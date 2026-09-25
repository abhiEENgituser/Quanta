#include "llama.h"
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>
#include <iostream>

using namespace std;

int main(int argc, char** argv) {
    const char* model_path = "/home/een/Abhishek/Learning/Projects/Quanta/models/qwen2.5-0.5b-q4km.gguf";
    const string prompt = "Tell me a joke about AI it must be funny and short. The joke is: ";
    const int n_ctx = 2048;

    int n_decode_steps = 100;
    int prompt_len = 0;

    // 1. Simple argument parsing
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "-n" && i + 1 < argc) n_decode_steps = stoi(argv[++i]);
        if (arg == "-l" && i + 1 < argc) prompt_len     = stoi(argv[++i]);
    }

    // 2. Initialize model and context
    llama_backend_init();
    llama_model* model = llama_model_load_from_file(model_path, llama_model_default_params());
    if (!model) return 1;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = n_ctx;
    llama_context* ctx = llama_init_from_model(model, ctx_params);

    const llama_vocab* vocab = llama_model_get_vocab(model);

    // 3. Tokenize prompt
    vector<llama_token> tokens(prompt.size() + 16);
    int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), tokens.size(), true, true);
    tokens.resize(n_tokens);

    // Optional padding/repeating if -l was passed
    if (prompt_len > 0) {
        vector<llama_token> padded(prompt_len);
        for (int i = 0; i < prompt_len; ++i) padded[i] = tokens[i % tokens.size()];
        tokens = move(padded);
        n_tokens = prompt_len;
    }

    // 4. Greedy sampler
    llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // ---------------- PREFILL ----------------
    // Use llama_batch_init and set tokens cleanly
    llama_batch batch = llama_batch_init(n_ctx, 0, 1);
    for (int i = 0; i < n_tokens; ++i) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = false;
    }
    batch.n_tokens = n_tokens;
    batch.logits[n_tokens - 1] = true; // calculate logits only for the last token

    auto t0 = chrono::steady_clock::now();
    llama_decode(ctx, batch);
    auto t1 = chrono::steady_clock::now();

    double prefill_ms = chrono::duration<double, milli>(t1 - t0).count();
    printf("Prefill: %.2f ms (%d tokens)\n\n", prefill_ms, n_tokens);

    // ---------------- DECODE ----------------
    auto t_decode_start = chrono::steady_clock::now();

    for (int step = 0; step < n_decode_steps; ++step) {
        // Sample next token
        llama_token new_id = llama_sampler_sample(smpl, ctx, -1);

        // Convert token to piece and print
        char buf[128];
        int len = llama_token_to_piece(vocab, new_id, buf, sizeof(buf), 0, true);
        printf("%.*s", len, buf);
        fflush(stdout);

        // Advance 1 token using helper: llama_batch_get_one(token*, n_tokens)
        // This handles pos, seq_id, and logits automatically!
        llama_batch one_token = llama_batch_get_one(&new_id, 1);
        if (llama_decode(ctx, one_token) != 0) break;
    }

    auto t_decode_end = chrono::steady_clock::now();
    double decode_ms = chrono::duration<double, milli>(t_decode_end - t_decode_start).count();
    printf("\n\nDecode: %.2f ms (%d steps)\n", decode_ms, n_decode_steps);

    // Cleanup
    llama_sampler_free(smpl);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}