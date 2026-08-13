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

Source is `probe` unless noted. Machine quiet, on AC, no swap activity.

| Metric | Value | Sample |
|---|---|---|
| Decode step, batch=1, 20 tokens | **18.03 ms/step** (55.5 t/s) | n=5, spread 3.7% |
| Decode step, batch=1, 100 tokens | **20.27 ms/step** (49.3 t/s) | n=1 |
| Prefill, 5 tokens | 45.74 ms | n=5, spread 28% — **unusable** |
| Compute buffer | 300.25 MiB | `n_batch=2048` |

**Decode cost grows with sequence length.** The 100-token average is 12% above the 20-token
average, because each step's attention scans every cached key. So "18 ms/step" is only valid at
short contexts — Phase 2's cost model needs a length term, not a constant.

**The 5-token prefill is not a usable measurement.** A 28% spread across repeats means it is
almost entirely fixed overhead with no signal about per-token cost. Needs a 16→1024 prompt-length
sweep to get a slope.

## Not yet measured

- **`-t 3` vs 4 threads.** Every benchmark in this project standardizes on `-t 3` (one core
  reserved for the control plane), but the numbers above are at 4. A slower `-t 3` figure is the
  expected cost of one fewer thread, not a regression.
- **Decode vs batch size.** The sublinearity that justifies batching at all.

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
