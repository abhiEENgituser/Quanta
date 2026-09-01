// quanta_shim: unix socket server that bridges quantad to llama.cpp.
//
// Step C — framing + JSON dispatch + tokenize/prefill/step/evict.
//
// Wire format (see docs/protocol.md):
//   [4-byte length, little-endian, unsigned][JSON body]
//
// Usage: quanta_shim -m <model.gguf> [-s socket_path] [-c n_ctx]

#include "llama.h"

#include <nlohmann/json.hpp>

#include <cerrno>
#include <climits>
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

// A length prefix larger than this is a protocol violation. Without this check,
// a corrupted or hostile length is an allocation of arbitrary size.
static const uint32_t MAX_FRAME = 8u * 1024 * 1024;

static const char * DEFAULT_SOCKET_PATH = "/tmp/quanta.sock";
static const int    DEFAULT_N_CTX       = 2048;

// ---------------------------------------------------------------------------
// Framing — proven in step A. Do not change without re-running the conformance
// tests in Testing/quanta_shim_client.py.
// ---------------------------------------------------------------------------

// Read exactly n bytes, or fail. A single read() may return fewer bytes than
// asked for — even on a local socket, even for small messages. Treating one
// read() as a whole message is the classic socket bug.
static bool read_exact(int fd, void * buf, size_t n) {
    char * p = static_cast<char *>(buf);
    size_t got = 0;

    while (got < n) {
        ssize_t r = read(fd, p + got, n - got);

        if (r == 0) {
            return false;               // peer closed the connection
        }
        if (r < 0) {
            if (errno == EINTR) {
                continue;               // interrupted by a signal, not an error
            }
            return false;
        }

        got += static_cast<size_t>(r);
    }

    return true;
}

// Write exactly n bytes, or fail. write() can also accept fewer bytes than
// offered, for the same reason read() can return fewer.
static bool write_exact(int fd, const void * buf, size_t n) {
    const char * p = static_cast<const char *>(buf);
    size_t sent = 0;

    while (sent < n) {
        ssize_t w = write(fd, p + sent, n - sent);

        if (w < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }

        sent += static_cast<size_t>(w);
    }

    return true;
}

static bool read_frame(int fd, std::string & body) {
    uint8_t hdr[4];
    if (!read_exact(fd, hdr, sizeof(hdr))) {
        return false;
    }

    // Decode little-endian explicitly rather than memcpy-ing into a uint32_t.
    // Both work on x86, but only this is correct on a big-endian host, and it
    // states the wire format in code instead of relying on the host's byte order.
    const uint32_t len =  static_cast<uint32_t>(hdr[0])
                       | (static_cast<uint32_t>(hdr[1]) <<  8)
                       | (static_cast<uint32_t>(hdr[2]) << 16)
                       | (static_cast<uint32_t>(hdr[3]) << 24);

    if (len > MAX_FRAME) {
        fprintf(stderr, "quanta_shim: frame too large (%u bytes), closing\n", len);
        return false;
    }

    body.resize(len);
    if (len > 0 && !read_exact(fd, &body[0], len)) {
        return false;
    }

    return true;
}

static bool write_frame(int fd, const std::string & body) {
    const uint32_t len = static_cast<uint32_t>(body.size());

    const uint8_t hdr[4] = {
        static_cast<uint8_t>( len        & 0xff),
        static_cast<uint8_t>((len >>  8) & 0xff),
        static_cast<uint8_t>((len >> 16) & 0xff),
        static_cast<uint8_t>((len >> 24) & 0xff),
    };

    if (!write_exact(fd, hdr, sizeof(hdr))) {
        return false;
    }
    if (len > 0 && !write_exact(fd, body.data(), len)) {
        return false;
    }

    return true;
}

// ---------------------------------------------------------------------------
// base64 — token text crosses the wire as raw bytes, because a single token can
// be a fragment of a multi-byte character and JSON strings must be valid UTF-8.
// Confirmed in practice: "café — 東京 🙂" produces byte-fallback pieces that are
// not valid UTF-8 on their own.
// ---------------------------------------------------------------------------

static std::string base64_encode(const uint8_t * data, size_t len) {
    static const char * tbl =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    std::string out;
    out.reserve(((len + 2) / 3) * 4);

    for (size_t i = 0; i < len; i += 3) {
        const uint32_t a = data[i];
        const uint32_t b = (i + 1 < len) ? data[i + 1] : 0u;
        const uint32_t c = (i + 2 < len) ? data[i + 2] : 0u;

        const uint32_t triple = (a << 16) | (b << 8) | c;

        out.push_back(tbl[(triple >> 18) & 0x3f]);
        out.push_back(tbl[(triple >> 12) & 0x3f]);
        out.push_back((i + 1 < len) ? tbl[(triple >> 6) & 0x3f] : '=');
        out.push_back((i + 2 < len) ? tbl[ triple       & 0x3f] : '=');
    }

    return out;
}

// ---------------------------------------------------------------------------
// Engine state
// ---------------------------------------------------------------------------

// Logits live only until the next llama_decode call — ANY next call, for any
// sequence. Holding a logits index across calls is therefore a use-after-
// invalidate bug the moment two sequences interleave (prefill A, prefill B:
// B's decode destroys A's pending logits — found the hard way, as an abort in
// llama_sampler_sample). So the shim samples EAGERLY, immediately after every
// decode while the logits are fresh, and stores the resulting pending TOKEN.
// A token id cannot go stale; a logits index can.
struct seq_state {
    llama_pos   next_pos    = 0;     // absolute position the next token will occupy
    llama_token pending     = -1;    // sampled but not yet emitted by a step
    bool        has_pending = false;
    bool        prefilled   = false;
};

struct engine {
    llama_model       * model  = nullptr;
    const llama_vocab * vocab  = nullptr;
    llama_context     * ctx    = nullptr;
    llama_sampler     * smpl   = nullptr;
    llama_batch         batch  = {};
    int                 n_ctx  = 0;
    int                 n_seqs = 1;

    // The shim owns each sequence's position because building a batch requires
    // it. Go keeps its own accounting and verifies against pos_range rather than
    // either side trusting the other blindly (docs/protocol.md, Constraint 2).
    std::unordered_map<llama_seq_id, seq_state> seqs;
};

static json make_error(const std::string & msg) {
    return json{{"ok", false}, {"error", msg}};
}

// A seq id outside [0, n_seqs) does not error inside llama.cpp — it ABORTS
// (GGML_ASSERT in llama_kv_cache::seq_rm and friends). These ids arrive over
// a socket, so an unchecked one is a remote kill switch for the server. Every
// handler that touches a sequence bounces bad ids here, before llama.cpp can
// see them.
static bool seq_in_range(const engine & eng, llama_seq_id seq) {
    return seq >= 0 && seq < static_cast<llama_seq_id>(eng.n_seqs);
}

static json seq_range_error(const engine & eng, llama_seq_id seq) {
    return make_error("sequence " + std::to_string(seq) + " out of range [0," +
                      std::to_string(eng.n_seqs) + ") — shim started with -q " +
                      std::to_string(eng.n_seqs));
}

// Decode one token id to its raw bytes. llama_token_to_piece returns a negative
// value when the buffer is too small — that must be checked before constructing
// a string, or the length is garbage.
static bool token_bytes(const engine & eng, llama_token tok, std::string & out) {
    char buf[256];

    const int32_t n = llama_token_to_piece(eng.vocab, tok, buf, sizeof(buf),
                                           /* lstrip */ 0, /* special */ true);
    if (n < 0) {
        return false;   // would need -n bytes; 256 is generous for a single token
    }

    out.assign(buf, static_cast<size_t>(n));
    return true;
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

// tokenize: text -> token ids. Deliberately separate from prefill, because Go
// needs the prompt's token count to make an admission decision *before* any KV
// memory is committed.
static json handle_tokenize(const engine & eng, const json & req) {
    if (!req.contains("text") || !req["text"].is_string()) {
        return make_error("tokenize: 'text' must be a string");
    }

    const std::string text        = req["text"].get<std::string>();
    const bool        add_special = req.value("add_special", true);

    // llama_tokenize does not report the token count up front. Guess a buffer
    // (byte-level fallback means never more than one token per input byte) plus
    // 2 for BOS/EOS, then retry if it was too small.
    std::vector<llama_token> toks(text.size() + 2);

    int32_t n = llama_tokenize(eng.vocab, text.c_str(), static_cast<int32_t>(text.size()),
                               toks.data(), static_cast<int32_t>(toks.size()),
                               add_special, /* parse_special */ true);

    if (n < 0) {
        // A too-small buffer is reported as the negated required size. INT32_MIN
        // cannot be negated without overflow — it stays negative, and resizing to
        // it would be catastrophic. `text` arrives over a socket, so this is no
        // longer input we control.
        if (n == INT32_MIN) {
            return make_error("tokenize: input too large to tokenize");
        }

        toks.resize(static_cast<size_t>(-n));

        n = llama_tokenize(eng.vocab, text.c_str(), static_cast<int32_t>(text.size()),
                           toks.data(), static_cast<int32_t>(toks.size()),
                           add_special, /* parse_special */ true);

        if (n < 0) {
            return make_error("tokenize: failed after resize");
        }
    }

    toks.resize(static_cast<size_t>(n));

    return json{{"ok", true}, {"tokens", toks}};
}

// prefill: submit a prompt for one sequence in a single decode pass, requesting
// logits only for the final token. start_pos lets a long prompt be split across
// several calls (chunked prefill) without a protocol change.
static json handle_prefill(engine & eng, const json & req) {
    if (!req.contains("seq") || !req["seq"].is_number_integer()) {
        return make_error("prefill: 'seq' must be an integer");
    }
    if (!req.contains("tokens") || !req["tokens"].is_array()) {
        return make_error("prefill: 'tokens' must be an array");
    }

    const llama_seq_id seq = req["seq"].get<llama_seq_id>();
    if (!seq_in_range(eng, seq)) {
        return seq_range_error(eng, seq);
    }

    std::vector<llama_token> toks;
    for (const auto & t : req["tokens"]) {
        if (!t.is_number_integer()) {
            return make_error("prefill: 'tokens' must contain only integers");
        }
        toks.push_back(t.get<llama_token>());
    }

    if (toks.empty()) {
        return make_error("prefill: 'tokens' must not be empty");
    }

    const llama_pos start_pos = req.value("start_pos", 0);
    if (start_pos < 0) {
        return make_error("prefill: 'start_pos' must be >= 0");
    }
    if (start_pos + static_cast<llama_pos>(toks.size()) > eng.n_ctx) {
        return make_error("prefill: start_pos + tokens exceeds n_ctx");
    }

    // Only the final token needs logits: the model computes hidden states for
    // every position anyway, but materialising a vocabulary-sized logits vector
    // for a prediction we discard is wasted work.
    eng.batch.n_tokens = static_cast<int32_t>(toks.size());
    for (size_t i = 0; i < toks.size(); ++i) {
        eng.batch.token[i]     = toks[i];
        eng.batch.pos[i]       = start_pos + static_cast<llama_pos>(i);
        eng.batch.n_seq_id[i]  = 1;
        eng.batch.seq_id[i][0] = seq;
        eng.batch.logits[i]    = false;
    }
    eng.batch.logits[toks.size() - 1] = true;

    const int32_t rc = llama_decode(eng.ctx, eng.batch);
    if (rc != 0) {
        // Positive codes are warnings (1 = no KV slot, potentially recoverable by
        // freeing memory); negatives are fatal. Pass the raw code through so Go
        // can tell the difference instead of guessing.
        json err = make_error("llama_decode failed during prefill");
        err["decode_rc"] = rc;
        return err;
    }

    seq_state & st = eng.seqs[seq];
    st.next_pos  = start_pos + static_cast<llama_pos>(toks.size());
    st.prefilled = true;

    // Sample NOW, while this decode's logits are still the current ones — the
    // next decode call for any sequence destroys them. The index is the batch
    // slot of the final token (llama_get_logits_ith resolves by batch
    // position, not output ordinal — that lesson stays load-bearing).
    st.pending     = llama_sampler_sample(eng.smpl, eng.ctx,
                                          static_cast<int32_t>(toks.size()) - 1);
    st.has_pending = true;

    return json{{"ok", true}};
}

// step: advance every listed sequence by exactly one decode pass. Go decides who
// is active and when to call; the shim never chooses.
//
// Emit-decode-sample: each sequence's PENDING token (sampled eagerly while its
// logits were fresh) is emitted, the pending tokens form the next batch, one
// decode advances everyone, and new pendings are sampled immediately from the
// just-computed logits — before any other call can invalidate them.
static json handle_step(engine & eng, const json & req) {
    if (!req.contains("active") || !req["active"].is_array()) {
        return make_error("step: 'active' must be an array");
    }

    std::vector<llama_seq_id> active;
    for (const auto & s : req["active"]) {
        if (!s.is_number_integer()) {
            return make_error("step: 'active' must contain only integers");
        }
        const llama_seq_id seq = s.get<llama_seq_id>();
        if (!seq_in_range(eng, seq)) {
            return seq_range_error(eng, seq);
        }
        active.push_back(seq);
    }

    if (active.empty()) {
        return make_error("step: 'active' must not be empty");
    }
    if (static_cast<int>(active.size()) > eng.n_ctx) {
        return make_error("step: more active sequences than batch capacity");
    }

    // Validate everything before mutating anything, so a bad request cannot leave
    // half the sequences advanced.
    for (const llama_seq_id seq : active) {
        auto it = eng.seqs.find(seq);
        if (it == eng.seqs.end() || !it->second.prefilled) {
            return make_error("step: sequence " + std::to_string(seq) + " has not been prefilled");
        }
        if (!it->second.has_pending) {
            return make_error("step: sequence " + std::to_string(seq) + " has no pending token (finished or invalidated — prefill again)");
        }
        if (it->second.next_pos >= eng.n_ctx) {
            return make_error("step: sequence " + std::to_string(seq) + " would exceed n_ctx");
        }
    }

    json out_tokens = json::array();

    eng.batch.n_tokens = 0;

    struct slot_owner { llama_seq_id seq; int32_t slot; };
    std::vector<slot_owner> owners;

    // Phase 1: emit each sequence's pending token and, unless it finished,
    // give that token a slot in the next batch.
    for (const llama_seq_id seq : active) {
        seq_state & st = eng.seqs[seq];

        const llama_token tok = st.pending;
        st.has_pending = false;

        std::string bytes;
        if (!token_bytes(eng, tok, bytes)) {
            return make_error("step: llama_token_to_piece needed a larger buffer");
        }

        const bool finished = llama_vocab_is_eog(eng.vocab, tok);

        out_tokens.push_back(json{
            {"seq",       seq},
            {"id",        tok},
            {"piece_b64", base64_encode(reinterpret_cast<const uint8_t *>(bytes.data()),
                                        bytes.size())},
            {"finished",  finished},
        });

        // 'finished' means the engine emitted an end-of-generation token, nothing
        // more. Length caps, timeouts and cancellation are policy and belong to
        // Go — a shim that stopped a sequence itself would be scheduling.
        if (finished) {
            continue;
        }

        const int32_t slot = eng.batch.n_tokens++;

        eng.batch.token[slot]     = tok;
        eng.batch.pos[slot]       = st.next_pos++;
        eng.batch.n_seq_id[slot]  = 1;
        eng.batch.seq_id[slot][0] = seq;
        eng.batch.logits[slot]    = true;

        owners.push_back({seq, slot});
    }

    // Phase 2: one decode advances every non-finished sequence, then new
    // pending tokens are sampled IMMEDIATELY — these logits die at the next
    // decode call, whoever makes it.
    if (eng.batch.n_tokens > 0) {
        const int32_t rc = llama_decode(eng.ctx, eng.batch);
        if (rc != 0) {
            json err = make_error("llama_decode failed during step");
            err["decode_rc"] = rc;
            return err;
        }

        for (const auto & o : owners) {
            seq_state & st = eng.seqs[o.seq];
            // Batch slot, not output ordinal — llama_get_logits_ith resolves
            // output_ids[slot] internally.
            st.pending     = llama_sampler_sample(eng.smpl, eng.ctx, o.slot);
            st.has_pending = true;
        }
    }

    return json{{"ok", true}, {"tokens", out_tokens}};
}

// evict: drop cached tokens for a sequence. Maps straight onto
// llama_memory_seq_rm, keeping llama.cpp's position conventions so there is no
// translation layer to get wrong: p0 < 0 means [0, p1), p1 < 0 means [p0, inf).
static json handle_evict(engine & eng, const json & req) {
    if (!req.contains("seq") || !req["seq"].is_number_integer()) {
        return make_error("evict: 'seq' must be an integer");
    }

    const llama_seq_id seq = req["seq"].get<llama_seq_id>();
    if (!seq_in_range(eng, seq)) {
        return seq_range_error(eng, seq);
    }
    const llama_pos    p0  = req.value("p0", 0);
    const llama_pos    p1  = req.value("p1", -1);

    llama_memory_t mem = llama_get_memory(eng.ctx);

    // 'removed' is reported separately from 'ok' because they answer different
    // questions: ok means the call executed, removed is seq_rm's own answer to
    // "was this removal possible".
    const bool removed = llama_memory_seq_rm(mem, seq, p0, p1);

    if (removed) {
        auto it = eng.seqs.find(seq);
        if (it != eng.seqs.end()) {
            if (p0 <= 0 && p1 < 0) {
                // Whole sequence cleared — drop our bookkeeping too, or the next
                // prefill would continue from a stale position.
                eng.seqs.erase(it);
            } else {
                // A partial evict leaves the cache in a state our cached
                // pending token no longer describes. Rather than guess, invalidate:
                // Go must prefill again before stepping this sequence.
                it->second.has_pending = false;
                if (p1 < 0 && p0 >= 0) {
                    it->second.next_pos = p0;
                }
            }
        }
    }

    return json{{"ok", true}, {"removed", removed}};
}

// pos_range: what the engine actually holds for a sequence. Not part of the
// Phase 1 protocol, but it is the verification primitive Constraint 2 calls for
// and it costs nothing to expose now.
static json handle_pos_range(engine & eng, const json & req) {
    if (!req.contains("seq") || !req["seq"].is_number_integer()) {
        return make_error("pos_range: 'seq' must be an integer");
    }

    const llama_seq_id seq = req["seq"].get<llama_seq_id>();
    if (!seq_in_range(eng, seq)) {
        return seq_range_error(eng, seq);
    }
    llama_memory_t     mem = llama_get_memory(eng.ctx);

    return json{
        {"ok",  true},
        {"min", llama_memory_seq_pos_min(mem, seq)},
        {"max", llama_memory_seq_pos_max(mem, seq)},
    };
}

// echo: returns the request unchanged. Exists so the framing conformance tests
// keep working as real operations are added — it touches no engine state and
// makes no decisions.
static json handle_echo(const json & req) {
    json resp = req;
    resp["ok"] = true;
    return resp;
}

static json dispatch(engine & eng, const std::string & body) {
    json req;
    try {
        req = json::parse(body);
    } catch (const json::exception & e) {
        // Recoverable: the length prefix already established where this message
        // ended, so the next frame starts exactly where expected.
        return make_error(std::string("malformed JSON: ") + e.what());
    }

    if (!req.is_object() || !req.contains("op") || !req["op"].is_string()) {
        return make_error("missing or non-string 'op'");
    }

    const std::string op = req["op"].get<std::string>();

    try {
        if (op == "tokenize")  return handle_tokenize (eng, req);
        if (op == "prefill")   return handle_prefill  (eng, req);
        if (op == "step")      return handle_step     (eng, req);
        if (op == "evict")     return handle_evict    (eng, req);
        if (op == "pos_range") return handle_pos_range(eng, req);
        if (op == "echo")      return handle_echo     (req);
        return make_error("unknown op: " + op);
    } catch (const std::exception & e) {
        // A handler throwing must not take the process down — the connection is
        // still framed correctly, so report it and carry on.
        return make_error(std::string("handler threw: ") + e.what());
    }
}

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

static void serve_connection(engine & eng, int conn) {
    // Strictly synchronous: one request in, one response out, repeat until the
    // peer closes or the stream desynchronises.
    std::string body;

    while (read_frame(conn, body)) {
        const json resp = dispatch(eng, body);

        if (!write_frame(conn, resp.dump())) {
            break;
        }
    }
}

static void engine_free(engine & eng) {
    if (eng.smpl)        llama_sampler_free(eng.smpl);
    if (eng.batch.token) llama_batch_free(eng.batch);
    if (eng.ctx)         llama_free(eng.ctx);
    if (eng.model)       llama_model_free(eng.model);
    llama_backend_free();
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    const char * sock_path  = DEFAULT_SOCKET_PATH;
    int          n_ctx      = DEFAULT_N_CTX;
    int          n_threads  = 0;   // 0 = llama.cpp default (GGML_DEFAULT_N_THREADS = 4)
    int          n_seqs     = 1;   // max concurrent sequences. NOTE: llama.cpp
                                   // DIVIDES n_ctx among sequences — n_ctx 2048
                                   // with -q 8 gives each sequence a 256-token
                                   // window. That division IS the KV budget.

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            sock_path = argv[++i];
        } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            n_ctx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-q") == 0 && i + 1 < argc) {
            n_seqs = atoi(argv[++i]);
        } else {
            fprintf(stderr, "usage: %s -m <model.gguf> [-s socket_path] [-c n_ctx] [-t threads] [-q max_seqs]\n", argv[0]);
            return 1;
        }
    }

    if (model_path == nullptr || n_ctx <= 0 || n_threads < 0 || n_seqs < 1) {
        fprintf(stderr, "usage: %s -m <model.gguf> [-s socket_path] [-c n_ctx] [-t threads] [-q max_seqs]\n", argv[0]);
        return 1;
    }

    // Writing to a socket whose peer has closed raises SIGPIPE, which by default
    // terminates the process silently. Ignoring it makes write() return EPIPE
    // instead, which the error paths already handle.
    signal(SIGPIPE, SIG_IGN);

    // ---- load the engine before listening: a shim that cannot serve should not
    // ---- accept a connection and then fail every request.

    llama_backend_init();

    engine eng;
    eng.n_ctx  = n_ctx;
    eng.n_seqs = n_seqs;

    llama_model_params model_params = llama_model_default_params();
    eng.model = llama_model_load_from_file(model_path, model_params);
    if (eng.model == nullptr) {
        fprintf(stderr, "quanta_shim: failed to load model: %s\n", model_path);
        engine_free(eng);
        return 1;
    }

    eng.vocab = llama_model_get_vocab(eng.model);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx     = static_cast<uint32_t>(n_ctx);
    ctx_params.n_batch   = static_cast<uint32_t>(n_ctx);
    ctx_params.n_seq_max = static_cast<uint32_t>(n_seqs);
    if (n_threads > 0) {
        // Both pools: n_threads drives decode (one token per sequence),
        // n_threads_batch drives prefill (many tokens at once). The -t 3 rule
        // reserves one core for the control plane, and it must apply to both
        // phases or the comparison measures a mixture.
        ctx_params.n_threads       = n_threads;
        ctx_params.n_threads_batch = n_threads;
    }

    eng.ctx = llama_init_from_model(eng.model, ctx_params);
    if (eng.ctx == nullptr) {
        fprintf(stderr, "quanta_shim: failed to create context\n");
        engine_free(eng);
        return 1;
    }

    // Greedy sampling: deterministic, so the same request always yields the same
    // tokens. That is what makes timings comparable across runs and lets output
    // be verified against probe.cpp.
    //
    // One shared sampler is safe only because greedy is stateless. Any sampler
    // with state (repetition penalty, a seeded RNG) needs one chain per sequence.
    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    eng.smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(eng.smpl, llama_sampler_init_greedy());

    eng.batch = llama_batch_init(n_ctx, 0, 1);

    fprintf(stderr, "quanta_shim: model loaded, vocab=%d, n_ctx=%d, threads=%d, seqs=%d (%d ctx/seq)\n",
            llama_vocab_n_tokens(eng.vocab), n_ctx,
            n_threads > 0 ? n_threads : -1 /* -1 = library default (4) */,
            n_seqs, n_ctx / n_seqs);

    // ---- socket setup

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;

    // sun_path is a fixed-size array (108 bytes on Linux). Overlong paths must be
    // rejected, not silently truncated into a different socket than intended.
    if (strlen(sock_path) >= sizeof(addr.sun_path)) {
        fprintf(stderr, "quanta_shim: socket path too long (max %zu)\n",
                sizeof(addr.sun_path) - 1);
        engine_free(eng);
        return 1;
    }
    strncpy(addr.sun_path, sock_path, sizeof(addr.sun_path) - 1);

    // A unix socket is a filesystem entry that outlives the process that made it.
    // Without this, bind() fails with EADDRINUSE after any unclean exit.
    unlink(sock_path);

    const int srv = socket(AF_UNIX, SOCK_STREAM, 0);
    if (srv < 0) {
        perror("socket");
        engine_free(eng);
        return 1;
    }

    if (bind(srv, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) < 0) {
        perror("bind");
        close(srv);
        engine_free(eng);
        return 1;
    }

    // Backlog of 1: quantad is the only client, and the shim serves one
    // connection at a time by design. Concurrency here would mean the shim
    // deciding what runs when, which is the scheduler's job.
    if (listen(srv, 1) < 0) {
        perror("listen");
        close(srv);
        engine_free(eng);
        return 1;
    }

    fprintf(stderr, "quanta_shim: listening on %s\n", sock_path);

    for (;;) {
        const int conn = accept(srv, nullptr, nullptr);
        if (conn < 0) {
            if (errno == EINTR) {
                continue;
            }
            perror("accept");
            break;
        }

        fprintf(stderr, "quanta_shim: client connected\n");
        serve_connection(eng, conn);
        fprintf(stderr, "quanta_shim: client disconnected\n");

        close(conn);
    }

    close(srv);
    unlink(sock_path);
    engine_free(eng);

    return 0;
}
