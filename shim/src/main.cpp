// quanta_shim: unix socket server that bridges quantad to llama.cpp.
//
// Stage 2 — framing + JSON dispatch + tokenize. Model and vocab are loaded, but
// no context and no decoding yet: prefill/step arrive in Stage 3.
//
// Wire format (see docs/protocol.md):
//   [4-byte length, little-endian, unsigned][JSON body]
//
// Usage: quanta_shim <model.gguf> [socket_path]

#include "llama.h"

#include <nlohmann/json.hpp>

#include <cerrno>
#include <climits>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

using json = nlohmann::json;

// A length prefix larger than this is a protocol violation. Without this check,
// a corrupted or hostile length is an allocation of arbitrary size.
static const uint32_t MAX_FRAME = 8u * 1024 * 1024;

static const char * DEFAULT_SOCKET_PATH = "/tmp/quanta.sock";

// ---------------------------------------------------------------------------
// Framing (unchanged from Stage 1)
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
// Engine state — loaded once at startup, shared by every request
// ---------------------------------------------------------------------------

struct engine {
    llama_model       * model = nullptr;
    const llama_vocab * vocab = nullptr;
};

static json make_error(const std::string & msg) {
    return json{{"ok", false}, {"error", msg}};
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

    // llama_tokenize does not tell you the token count up front. Guess a buffer
    // (a token is never more than one per input byte, since byte-level fallback
    // is the worst case) plus 2 for BOS/EOS, then retry if it was too small.
    std::vector<llama_token> toks(text.size() + 2);

    int32_t n = llama_tokenize(eng.vocab, text.c_str(), static_cast<int32_t>(text.size()),
                               toks.data(), static_cast<int32_t>(toks.size()),
                               add_special, /* parse_special */ true);

    if (n < 0) {
        // A too-small buffer is reported as the negated required size. INT32_MIN
        // cannot be negated without overflow — it stays negative, and resizing to
        // it would be catastrophic. Unreachable for sane input, but `text` now
        // arrives over a socket, so it is no longer our input to trust.
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

// echo: returns the request unchanged. Exists purely so the framing conformance
// tests keep working as real operations are added — it touches no engine state
// and makes no decisions.
static json handle_echo(const json & req) {
    json resp = req;
    resp["ok"] = true;
    return resp;
}

static json dispatch(const engine & eng, const std::string & body) {
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
        if (op == "tokenize") {
            return handle_tokenize(eng, req);
        }
        if (op == "echo") {
            return handle_echo(req);
        }
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

static void serve_connection(const engine & eng, int conn) {
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

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [socket_path]\n", argv[0]);
        return 1;
    }

    const char * model_path = argv[1];
    const char * sock_path  = (argc > 2) ? argv[2] : DEFAULT_SOCKET_PATH;

    // Writing to a socket whose peer has closed raises SIGPIPE, which by default
    // terminates the process silently. Ignoring it makes write() return EPIPE
    // instead, which the error paths above already handle.
    signal(SIGPIPE, SIG_IGN);

    // ---- load the engine before listening: a shim that cannot serve should not
    // ---- accept a connection and then fail every request.

    llama_backend_init();

    engine eng;

    llama_model_params model_params = llama_model_default_params();
    eng.model = llama_model_load_from_file(model_path, model_params);
    if (eng.model == nullptr) {
        fprintf(stderr, "quanta_shim: failed to load model: %s\n", model_path);
        llama_backend_free();
        return 1;
    }

    eng.vocab = llama_model_get_vocab(eng.model);

    fprintf(stderr, "quanta_shim: model loaded (%d tokens in vocab)\n",
            llama_vocab_n_tokens(eng.vocab));

    // ---- socket setup

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;

    // sun_path is a fixed-size array (108 bytes on Linux). Overlong paths must be
    // rejected, not silently truncated into a different socket than intended.
    if (strlen(sock_path) >= sizeof(addr.sun_path)) {
        fprintf(stderr, "quanta_shim: socket path too long (max %zu)\n",
                sizeof(addr.sun_path) - 1);
        llama_model_free(eng.model);
        llama_backend_free();
        return 1;
    }
    strncpy(addr.sun_path, sock_path, sizeof(addr.sun_path) - 1);

    // A unix socket is a filesystem entry that outlives the process that made it.
    // Without this, bind() fails with EADDRINUSE after any unclean exit.
    unlink(sock_path);

    const int srv = socket(AF_UNIX, SOCK_STREAM, 0);
    if (srv < 0) {
        perror("socket");
        llama_model_free(eng.model);
        llama_backend_free();
        return 1;
    }

    if (bind(srv, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) < 0) {
        perror("bind");
        close(srv);
        llama_model_free(eng.model);
        llama_backend_free();
        return 1;
    }

    // Backlog of 1: quantad is the only client, and the shim serves one
    // connection at a time by design.
    if (listen(srv, 1) < 0) {
        perror("listen");
        close(srv);
        llama_model_free(eng.model);
        llama_backend_free();
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

    llama_model_free(eng.model);
    llama_backend_free();

    return 0;
}