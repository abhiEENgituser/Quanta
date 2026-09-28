#include "llama.h"
#include <nlohmann/json.hpp>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

using json = nlohmann::json;
using namespace std;
// Safety cap: prevents malicious/corrupt length prefixes from exhausting memory
static const uint32_t MAX_FRAME = 8u * 1024 * 1024;

// ---------------------------------------------------------------------------
// 1. Network Helpers
// Sockets can return partial data on read/write. We loop until ALL bytes arrive.
// Frame format: [4-byte payload size] + [JSON payload string]
// ---------------------------------------------------------------------------
static bool rw_exact(int fd, void* buf, size_t n, bool is_write) {
    char* p = static_cast<char*>(buf);
    size_t done = 0;
    while (done < n) {
        ssize_t r = is_write ? write(fd, p + done, n - done) : read(fd, p + done, n - done);
        if (r == 0) return false;      // Socket closed by peer
        if (r < 0) {
            if (errno == EINTR) continue; // Retry if interrupted by OS signal
            return false;
        }
        done += r;
    }
    return true;
}

// Reads the 4-byte size header first, then reads the exact JSON body
static bool read_frame(int fd, string& body) {
    uint32_t len = 0;
    if (!rw_exact(fd, &len, 4, false) || len > MAX_FRAME) return false;
    body.resize(len);
    return len == 0 || rw_exact(fd, &body[0], len, false);
}

// Sends the 4-byte size header first, then sends the JSON text
static bool write_frame(int fd, const string& body) {
    uint32_t len = static_cast<uint32_t>(body.size());
    return rw_exact(fd, &len, 4, true) && (len == 0 || rw_exact(fd, (void*)body.data(), len, true));
}

// Base64 helper: LLM token pieces can be raw partial UTF-8 bytes that break JSON strings
static string base64_encode(const uint8_t* data, size_t len) {
    static const char tbl[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    string out;
    out.reserve(((len + 2) / 3) * 4);
    for (size_t i = 0; i < len; i += 3) {
        uint32_t a = data[i], b = (i + 1 < len) ? data[i + 1] : 0, c = (i + 2 < len) ? data[i + 2] : 0;
        uint32_t triple = (a << 16) | (b << 8) | c;
        out.push_back(tbl[(triple >> 18) & 0x3f]);
        out.push_back(tbl[(triple >> 12) & 0x3f]);
        out.push_back((i + 1 < len) ? tbl[(triple >> 6) & 0x3f] : '=');
        out.push_back((i + 2 < len) ? tbl[triple & 0x3f] : '=');
    }
    return out;
}

// ---------------------------------------------------------------------------
// 2. Engine State
// Tracks position and the next token to emit for each independent sequence
// ---------------------------------------------------------------------------
struct seq_state {
    llama_pos   next_pos    = 0;     // Next token position index in the KV cache
    llama_token pending     = -1;    // Already sampled token waiting to be returned
    bool        has_pending = false; 
    bool        prefilled   = false; // Has this sequence ingested its prompt?
};

struct engine {
    llama_model*                      model  = nullptr;
    const llama_vocab*                vocab  = nullptr;
    llama_context*                    ctx    = nullptr;
    llama_sampler*                    smpl   = nullptr;
    llama_batch                       batch  = {};
    int                               n_ctx  = 2048; // Total context length across all sequences
    int                               n_seqs = 1;    // Max concurrent conversation streams
    unordered_map<llama_seq_id, seq_state> seqs; // Per-stream state tracker
};

static json err_json(const string& msg) { return {{"ok", false}, {"error", msg}}; }

// Converts an integer token ID back into human-readable raw text bytes
static bool get_token_piece(const engine& eng, llama_token tok, string& out) {
    char buf[256];
    int32_t n = llama_token_to_piece(eng.vocab, tok, buf, sizeof(buf), 0, true);
    if (n < 0) return false;
    out.assign(buf, n);
    return true;
}

// ---------------------------------------------------------------------------
// 3. Command Handlers
// ---------------------------------------------------------------------------

// "tokenize": converts plain text string -> vector of token IDs without touching KV memory
static json handle_tokenize(const engine& eng, const json& req) {
    if (!req.value("text", "").size()) return err_json("text missing or empty");
    string text = req["text"].get<string>();
    bool add_special = req.value("add_special", true);

    vector<llama_token> toks(text.size() + 16);
    int32_t n = llama_tokenize(eng.vocab, text.c_str(), text.size(), toks.data(), toks.size(), add_special, true);
    if (n < 0) { // Buffer was too small, resize to needed capacity (-n) and retry
        toks.resize(-n);
        n = llama_tokenize(eng.vocab, text.c_str(), text.size(), toks.data(), toks.size(), add_special, true);
    }
    toks.resize(n);
    return {{"ok", true}, {"tokens", toks}};
}

// "prefill": ingest all prompt tokens into the KV cache for a single sequence
static json handle_prefill(engine& eng, const json& req) {
    llama_seq_id seq = req.value("seq", -1);
    if (seq < 0 || seq >= eng.n_seqs) return err_json("seq out of range");

    vector<llama_token> toks = req.value("tokens", vector<llama_token>{});
    if (toks.empty()) return err_json("tokens empty");

    llama_pos start_pos = req.value("start_pos", 0);
    if (start_pos + (llama_pos)toks.size() > eng.n_ctx) return err_json("exceeds n_ctx");

    // Build the prefill batch
    eng.batch.n_tokens = toks.size();
    for (size_t i = 0; i < toks.size(); ++i) {
        eng.batch.token[i]     = toks[i];
        eng.batch.pos[i]       = start_pos + i;
        eng.batch.n_seq_id[i]  = 1;
        eng.batch.seq_id[i][0] = seq;
        eng.batch.logits[i]    = false; // Don't calculate probabilities for past tokens
    }
    eng.batch.logits[toks.size() - 1] = true; // ONLY compute logits for the final prompt token

    // Run forward pass to fill KV cache
    if (llama_decode(eng.ctx, eng.batch) != 0) return err_json("llama_decode failed");

    // IMPORTANT: Sample IMMEDIATELY. Next decode call for ANY sequence wipes out these logits!
    seq_state& st = eng.seqs[seq];
    st.next_pos    = start_pos + toks.size();
    st.prefilled   = true;
    st.pending     = llama_sampler_sample(eng.smpl, eng.ctx, toks.size() - 1);
    st.has_pending = true;
    return {{"ok", true}};
}

// "step": advance active sequences by 1 token in parallel
static json handle_step(engine& eng, const json& req) {
    vector<llama_seq_id> active = req.value("active", vector<llama_seq_id>{});
    if (active.empty()) return err_json("active list is empty");

    // Validate that all requested sequences have a prompt prefilled and a token ready
    for (llama_seq_id seq : active) {
        if (!eng.seqs.count(seq) || !eng.seqs[seq].has_pending)
            return err_json("seq not prefilled or missing pending token: " + to_string(seq));
    }

    json out_tokens = json::array();
    eng.batch.n_tokens = 0;
    vector<pair<llama_seq_id, int32_t>> owners;

    // STEP A: Emit the previously sampled tokens to the caller
    // And pack non-finished tokens into the next combined batch
    for (llama_seq_id seq : active) {
        seq_state& st = eng.seqs[seq];
        llama_token tok = st.pending;
        st.has_pending = false;

        string bytes;
        get_token_piece(eng, tok, bytes);
        bool finished = llama_vocab_is_eog(eng.vocab, tok); // End-of-generation reached?

        out_tokens.push_back({
            {"seq", seq}, {"id", tok},
            {"piece_b64", base64_encode((const uint8_t*)bytes.data(), bytes.size())},
            {"finished", finished}
        });

        if (finished) continue; // If sequence stopped, do not queue it for next decode

        // Assign a slot in the unified multi-sequence batch
        int32_t slot = eng.batch.n_tokens++;
        eng.batch.token[slot]     = tok;
        eng.batch.pos[slot]       = st.next_pos++;
        eng.batch.n_seq_id[slot]  = 1;
        eng.batch.seq_id[slot][0] = seq;
        eng.batch.logits[slot]    = true; // We need logits to predict the next word
        owners.push_back({seq, slot});
    }

    // STEP B: Run forward pass for all active tokens together, then sample next tokens
    if (eng.batch.n_tokens > 0) {
        if (llama_decode(eng.ctx, eng.batch) != 0) return err_json("decode failed");

        // Sample each sequence's NEXT token from its specific slot while logits are fresh
        for (auto& [seq, slot] : owners) {
            eng.seqs[seq].pending     = llama_sampler_sample(eng.smpl, eng.ctx, slot);
            eng.seqs[seq].has_pending = true;
        }
    }

    return {{"ok", true}, {"tokens", out_tokens}};
}

// "evict": delete a sequence's KV memory when a conversation finishes or is reset
static json handle_evict(engine& eng, const json& req) {
    llama_seq_id seq = req.value("seq", -1);
    if (seq < 0 || seq >= eng.n_seqs) return err_json("invalid seq");
    llama_pos p0 = req.value("p0", 0), p1 = req.value("p1", -1);

    // Free KV cache range [p0, p1) for this sequence ID
    bool removed = llama_memory_seq_rm(llama_get_memory(eng.ctx), seq, p0, p1);
    if (removed && p0 <= 0 && p1 < 0) eng.seqs.erase(seq); // Erase tracking if entirely cleared
    return {{"ok", true}, {"removed", removed}};
}

// Route JSON request to proper handler function based on the "op" field
static json dispatch(engine& eng, const string& body) {
    try {
        json req = json::parse(body);
        string op = req.value("op", "");
        if (op == "tokenize")  return handle_tokenize(eng, req);
        if (op == "prefill")   return handle_prefill(eng, req);
        if (op == "step")      return handle_step(eng, req);
        if (op == "evict")     return handle_evict(eng, req);
        if (op == "echo")      { req["ok"] = true; return req; }
        return err_json("unknown op: " + op);
    } catch (const exception& e) {
        return err_json(e.what());
    }
}

// ---------------------------------------------------------------------------
// 4. Server Entry Point (CLI args, model setup, and socket listen loop)
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    string model_path, sock_path = "/tmp/quanta.sock";
    int n_ctx = 2048, n_seqs = 1;

    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        if (a == "-m" && i + 1 < argc) model_path = argv[++i];
        else if (a == "-s" && i + 1 < argc) sock_path = argv[++i];
        else if (a == "-c" && i + 1 < argc) n_ctx = stoi(argv[++i]);
        else if (a == "-q" && i + 1 < argc) n_seqs = stoi(argv[++i]);
    }
    if (model_path.empty()) return 1;

    // Ignore SIGPIPE so the server doesn't crash if the client abruptly disconnects
    signal(SIGPIPE, SIG_IGN);
    llama_backend_init();

    // Initialize model, context, greedy sampler, and pre-allocated batch
    engine eng;
    eng.n_ctx = n_ctx;
    eng.n_seqs = n_seqs;
    eng.model = llama_model_load_from_file(model_path.c_str(), llama_model_default_params());
    if (!eng.model) return 1;

    eng.vocab = llama_model_get_vocab(eng.model);
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_batch = n_ctx;
    cparams.n_seq_max = n_seqs; // Partition KV cache across up to n_seqs streams
    eng.ctx = llama_init_from_model(eng.model, cparams);

    eng.smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(eng.smpl, llama_sampler_init_greedy());
    eng.batch = llama_batch_init(n_ctx, 0, 1);

    // Bind and listen on local Unix domain socket
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, sock_path.c_str(), sizeof(addr.sun_path) - 1);
    unlink(sock_path.c_str()); // Remove leftover socket file if program crashed earlier

    int srv = socket(AF_UNIX, SOCK_STREAM, 0);
    bind(srv, (struct sockaddr*)&addr, sizeof(addr));
    listen(srv, 1); // Single-client queue (dedicated daemon connection)

    // Accept loop: process one client request at a time synchronously
    while (true) {
        int conn = accept(srv, nullptr, nullptr);
        if (conn < 0) continue;
        string body;
        while (read_frame(conn, body)) {
            if (!write_frame(conn, dispatch(eng, body).dump())) break;
        }
        close(conn);
    }

    unlink(sock_path.c_str());
    return 0;
}