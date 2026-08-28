// probe: single-sequence greedy decode against llama.cpp, no common/ dependency.
//
// Usage: probe [-l prompt_tokens] [-n decode_steps]
//   -l  pad/truncate the prompt to exactly this many tokens (0 = natural length).
//       Padding tiles the natural prompt's tokens; for timing that is fine, since
//       prefill cost depends on position count, not which tokens sit there.
//   -n  number of decode steps (0 = prefill only).
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

static double ms_between(const struct timespec & t0, const struct timespec & t1) {
    return (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
}

int main(int argc, char ** argv) {
    const char * model_path = "/home/een/Abhishek/Learning/Projects/Quanta/models/qwen2.5-0.5b-q4km.gguf";
    const std::string prompt = "The capital of France is";
    const int n_ctx = 2048;          // KV cache size Limit in tokens (Max token model can hold in attention window at once)

    int n_decode_steps = 100;
    int prompt_len     = 0;          // 0 = use the natural prompt length

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_decode_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-l") == 0 && i + 1 < argc) {
            prompt_len = atoi(argv[++i]);
        } else {
            fprintf(stderr, "usage: %s [-l prompt_tokens] [-n decode_steps]\n", argv[0]);
            return 1;
        }
    }
    if (n_decode_steps < 0 || prompt_len < 0 || prompt_len + n_decode_steps > n_ctx) {
        fprintf(stderr, "error: prompt_len + decode_steps must fit in n_ctx (%d)\n", n_ctx);
        return 1;
    }

    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path, model_params);
    if (model == nullptr) {
        fprintf(stderr, "error: unable to load model: %s\n", model_path);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // tokenize the prompt (mirrors what common_tokenize does internally)
    std::vector<llama_token> tokens(prompt.size() + 8);
    int n_tokens = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(),tokens.data(), (int32_t) tokens.size(), true, true);
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(),
                                   tokens.data(), (int32_t) tokens.size(), true, true);
    }
    tokens.resize(n_tokens);

    // -l: pad or truncate to exactly prompt_len tokens. Padding tiles the natural
    // prompt's tokens. The token *values* barely matter for timing — prefill cost
    // is driven by how many positions are processed, not which tokens they hold —
    // but this changes what text gets generated, so -l runs are for measurement,
    // not for verifying output against the recorded probe text.
    if (prompt_len > 0) {
        std::vector<llama_token> padded(prompt_len);
        for (int i = 0; i < prompt_len; ++i) {
            padded[i] = tokens[i % tokens.size()];
        }
        tokens = padded;
        n_tokens = prompt_len;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx   = n_ctx;
    ctx_params.n_batch = n_ctx;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        fprintf(stderr, "error: failed to create the llama_context\n");
        return 1;
    }

    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    llama_batch batch = llama_batch_init(n_ctx, 0, 1);

    // prefill: submit the whole prompt, request logits only for the last token
    batch.n_tokens = n_tokens;
    for (int i = 0; i < n_tokens; ++i) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = false;
    }
    batch.logits[n_tokens - 1] = true;

    struct timespec t_prefill_0, t_prefill_1;
    clock_gettime(CLOCK_MONOTONIC, &t_prefill_0);
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "error: llama_decode() failed on prefill\n");
        return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &t_prefill_1);
    fprintf(stderr, "prefill: %.2f ms for %d tokens\n", ms_between(t_prefill_0, t_prefill_1), n_tokens);

    int n_cur = n_tokens;

    struct timespec t_decode_0, t_decode_1;
    clock_gettime(CLOCK_MONOTONIC, &t_decode_0);

    for (int step = 0; step < n_decode_steps; ++step) {
        const llama_token new_id = llama_sampler_sample(smpl, ctx, -1);

        char piece_buf[128];
        int piece_len = llama_token_to_piece(vocab, new_id, piece_buf, sizeof(piece_buf), 0, true);
        std::string piece(piece_buf, piece_len);
        printf("%d: %s\n", step, piece.c_str());

        batch.n_tokens      = 1;
        batch.token[0]      = new_id;
        batch.pos[0]        = n_cur;
        batch.n_seq_id[0]   = 1;
        batch.seq_id[0][0]  = 0;
        batch.logits[0]     = true;
        n_cur += 1;

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "error: llama_decode() failed at step %d\n", step);
            return 1;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t_decode_1);
    fprintf(stderr, "decode: %.2f ms for %d tokens\n", ms_between(t_decode_0, t_decode_1), n_decode_steps);

    llama_sampler_free(smpl);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
