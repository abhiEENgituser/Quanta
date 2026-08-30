# Baseline

Reference numbers for this machine. Every later performance claim gets checked against these.

## Environment

| | |
|---|---|
| CPU | Intel Core i7-8650U, 4c/8t, 15 W, 400–4200 MHz |
| RAM | 16 GB |
| GPU | none |
| Model | Qwen2.5-0.5B-Instruct, Q4_K_M |
| HF snapshot | `9217f5db79a29953eb74d5343926648285ec7e67` |
| llama.cpp | tag `b10121` (`555881ebc`), submodule at `shim/vendor/llama.cpp` |
| Build | `CMAKE_BUILD_TYPE=Release` (`-O3 -DNDEBUG`) |
| Threads | **4** (`GGML_DEFAULT_N_THREADS`) — the probe never calls `llama_set_n_threads` |

## Numbers

Source is `probe`; sweep results in `bench/results/baseline_sweep_20260828.csv`, produced by
`bench/configs/baseline_sweep.sh` (6 repeats, run 1 discarded, MHz sampled during runs).
Machine quiet, on AC, no swap activity, `performance` governor unless noted.

| Metric | Value | Sample |
|---|---|---|
| Decode step, batch=1, 100 steps | **17.24 ms/step** (58 t/s) | n=5, spread 7.4% |
| Decode step, batch=1, 20 steps | 18.03 ms/step | n=5, spread 3.7% — **`powersave` governor**, not directly comparable |
| Prefill slope | **≈ 6.0 ms per prompt token** | fit over 16/64/128/256/512 tokens, n=5 each, R² = 0.995 |
| Compute buffer | 300.25 MiB | `n_batch=2048` |

**Decode cost vs context length** (1 decode step taken immediately after an L-token prefill):

| Context | ms/step | MHz during run |
|---|---|---|
| 16 | 18.2 | ~3470 |
| 64 | 18.0 | ~3360 |
| 128 | 20.2 | ~3450 |
| 256 | 20.9 | ~3230 |
| 512 | **29.4** | ~2780 |

Flat to ~64 tokens, then real growth — each step's attention scans every cached key. The 512
figure is partly confounded: the long prefill preceding it drained the turbo budget (~2780 vs
~3470 MHz), which explains some but not all of the rise. Phase 2's cost model needs a length
term either way. An earlier n=1 claim of "20.27 ms/step at 100 tokens, 12% above baseline" did
**not** reproduce (n=5 gives 17.24) — growth is real but negligible below ~128 tokens.

**Prefill is roughly linear with mild superlinearity.** The least-squares intercept is negative
(−68 ms), which is the fit absorbing per-token cost that grows with length, not a real negative
fixed cost. Individual prefill points are noisy (spreads 8–50%; one 992 ms outlier in the
128-token series against a ~640 ms typical) — the slope is trustworthy, single points are not.
Prefill at ~6 ms/token vs decode at ~17 ms/token: on this 4-core CPU, prefill's parallelism buys
only ~3×, nothing like the gap a GPU would show.

## Phase 1 stack baselines (full HTTP→SSE path, per-run fresh servers)

From `make bench-single`: engine pinned per standing rules (t3 → cores 0-2, t4 → 0-3; control
plane on core 3), performance governor enforced by the script, 5 runs per cell, fresh shim and
quantad every run. Raw data: `bench/results/single/`. Values are per-run server-side p50s
(measured after the admission mutex — no queue wait in them), mean over 5 runs [min–max].

**Prefill (server `ttft_prefill`; includes tokenize+evict RTTs and the first decode step):**

| Prompt | 4 threads | 3 threads | `-t 3` cost |
|---|---|---|---|
| ~25 tok | 145 ms [140–151] | 199 ms [182–237] | **+37%** |
| ~105 tok | 670 ms [562–758] | 876 ms [865–886] | **+31%** |
| ~425 tok | 2975 ms [2964–3000] | 3484 ms [3412–3615] (n=4) | **+17%** |

**Decode ITL (client-observed, queue-free by construction — the engine is serial):**

| Context ≈ | 4 threads | 3 threads | `-t 3` cost |
|---|---|---|---|
| ~40 tok | 17.1 ms [17.0–17.5] | 20.8 ms [19.0–23.5] | **+22%** |
| ~120 tok | 20.8 ms [17.9–22.0] | 23.9 ms [23.8–24.0] | **+15%** |
| ~440 tok | 23.8 ms [23.5–23.9] | 26.1 ms [25.8–26.6] (n=4) | **+9%** |

**The `-t 3` reservation gets cheaper as context grows** — decode is increasingly
memory-bandwidth-bound, so the fourth compute thread is worth less exactly where a serving
workload spends its time. Reserving a core for the control plane costs ~10-20% in the regime
that matters, not the naive 33%.

Cross-checks: t4 short-context ITL (17.0-17.5) reproduces the bare-metal 17.24 through the full
stack; stack prefill slope ≈ 7.1 ms/token vs 6.0 bare-metal (quantad sharing the machine, as
designed).

**Excluded: `t3_rep85_run1`** — the machine suspended mid-run (screen lock → idle sleep). Its
engine-side numbers sat mid-pack, but the drift gate flagged it independently: mean drift grew
453 µs → 156 ms between run halves from the post-resume wakeup. A run the validity gate rejects
stays rejected, even when its numbers look fine — plausible-but-contaminated is the failure mode
this project keeps meeting. The t3/~425tok cells above are n=4. (`make bench-single` now holds a
systemd-inhibit lease so idle-suspend cannot recur mid-session.)

**Caveat on the client-side TTFT columns in those CSVs:** the session ran a fixed 0.5 req/s at
every prompt length, so utilization varied wildly per cell (~0.35 at short prompts, ~1.6 —
unstable overload — at ~425 tokens). Client TTFT there measures queue growth, not the engine;
honest data, wrong label. The engine tables above are queue-free. The script now scales rate
per prompt length so future TTFT curves are taken at comparable low utilization.

## Not yet measured

- **Decode vs batch size.** The sublinearity that justifies batching at all. (Phase 3 — needs
  multi-sequence support.)

## Measurement rules learned here

**Pin the cores.** Contention is expensive and was measured directly:

| Condition | ms/step |
|---|---|
| quiet machine | 18.1 – 18.8 |
| + 1 competing thread | 22.0 – 23.0 |
| + 4 competing threads | 30.8 – 32.5 |

**Repeats catch what averaging cannot.** An earlier 34.95 ms/step figure was recorded as trusted
from a single run. It was a 20-step average — precise, but wrong by 1.9×, because the machine was
busy and every step was equally contended. Averaging inside a run cannot detect that; only
repeating the run can.

**Log MHz during the run, not after.** Package temperature stayed at 30 °C in every condition
above, so the slowdown was never thermal — it was turbo budget (fewer active cores allow a higher
clock) plus CPU-time competition. That was only diagnosable because MHz was sampled while running.

**Known gap:** `GGML_NATIVE` and `GGML_OPENMP` were confirmed `ON` in the standalone llama.cpp
tree, not in `shim/build/`. Verify before comparing numbers across the two trees.
