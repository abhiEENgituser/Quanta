# Baseline

Environment and reference numbers for Phase 0. Every later performance claim gets checked
against this — a plausible-sounding explanation that contradicts a recorded baseline is wrong,
not the baseline.

## Environment

| | |
|---|---|
| CPU | Intel Core i7-8650U, 4 cores / 8 threads, 15 W mobile TDP |
| CPU frequency range | 400 MHz – 4200 MHz (thermal throttling is real on this part — log `/proc/cpuinfo` MHz during any timed run) |
| RAM | 16 GB |
| GPU | none (Intel UHD 620 integrated graphics only, no CUDA/ROCm) |
| OS | Linux |

## Model

| | |
|---|---|
| Model | Qwen2.5-0.5B-Instruct |
| Quantization | Q4_K_M (confirmed from filename; not independently re-verified against the GGUF metadata) |
| HF snapshot hash | `9217f5db79a29953eb74d5343926648285ec7e67` |
| Symlink | `models/qwen2.5-0.5b-q4km.gguf` |

## llama.cpp

| | |
|---|---|
| Tag | `b10121` |
| Commit | `555881ebc8b0fc0402b30e09258a32a7bfd13c52` |
| Vendored at | `shim/vendor/llama.cpp` (git submodule) |

## Build configuration actually used

Two separate build trees exist and must not be confused:

- **`shim/build/`** — built via `shim/CMakeLists.txt` (`add_subdirectory(vendor/llama.cpp)`),
  produces `quanta_shim` and `probe`. **This is the tree the recorded numbers below came from.**
- **`shim/vendor/llama.cpp/build/`** — llama.cpp built standalone (its own CMakeLists as the
  top-level project), produces the full tool suite (`llama-cli`, `llama-bench`, etc.).

Verified from `CMakeCache.txt` in `shim/build/`:

```
CMAKE_BUILD_TYPE = Release   (-O3 -DNDEBUG)
```

**This is not something `shim/CMakeLists.txt` sets explicitly** — it's inherited from
llama.cpp's own top-level `CMakeLists.txt`, which defaults to `Release` whenever no build type
is specified and the platform isn't Xcode/MSVC (`CMakeLists.txt:10-12`). It happens to be
correct for a timed run, but it is an accident of upstream's default, not a guarantee our own
project makes. **Known gap:** once a debug/sanitizer build is needed (hard-won rule 6 —
separate `build-release`/`build-debug` directories, encoded in the Makefile), `shim/CMakeLists.txt`
should set `CMAKE_BUILD_TYPE` explicitly rather than continue relying on this default.

`GGML_NATIVE` and `GGML_OPENMP` were confirmed `ON` only in the standalone llama.cpp tree
(used for the `llama-cli` sanity check), not verified in `shim/build/`'s cache — worth
double-checking before trusting a cross-tree comparison.

## Recorded numbers

All `probe` runs use `GGML_DEFAULT_N_THREADS` = **4** — the probe never calls
`llama_set_n_threads`, so it inherits `llama_context_default_params()`'s built-in constant
(`ggml.h:232`). Not autodetected, and **not** the `-t 3` config Phase 1 standardizes on.

| Metric | Value | Source | Trust |
|---|---|---|---|
| Decode step, batch=1 | **18.03 ms/step** (55.5 t/s) — n=5, range 17.77–18.44, spread 3.7% | `probe`, 6 runs × 20 steps, run 1 discarded, 2 s cooldown | Good — tight spread, quiet machine |
| Prefill, 5 tokens | 45.74 ms — n=5, range 39.95–52.70, spread **28%** | same runs | **Unusable** — 28% spread confirms a 5-token prefill is nearly all fixed overhead; needs the 16→1024 sweep in Phase 2 |
| ~~Decode step, batch=1~~ | ~~34.95 ms/step (28.6 t/s)~~ | ~~`probe`, 20 steps, n=1~~ | **Superseded — does not reproduce.** See below |
| ~~Prefill, 5 tokens~~ | ~~82.39 ms~~ | ~~`probe`, n=1~~ | **Superseded — does not reproduce** |
| Compute buffer | 300.25 MiB | `probe`, `n_batch=2048` | — | Note: competes with the KV cache budget for RAM |
| `llama-cli` generation | 25.1–28.2 t/s | conversation mode, standalone tree | unset/default | Provisional only — tty-dependent behavior (hard-won rule 3) makes conversation-mode numbers suspect |
| Cost of `-t 3` vs default (4) | ~25–40% throughput | inferred, not measured | — | **Not yet measured — Phase 1 EXIT criterion** |

### Conditions of the n=5 measurement

| | |
|---|---|
| Power | on AC |
| Governor | **`powersave`** — violates the standing rule (`performance`); `intel_pstate` still boosted to near-max turbo, but this run is not fully rule-compliant |
| CPU MHz | 3777–4061 across runs, no downward drift |
| Swap | `si=0 so=0` — clean |
| Load avg | 0.43 at start |

**Methodological gap:** MHz was sampled *after* each run, not during. Throttling *within* a run
would be invisible. A compliant Phase 1 harness should sample MHz continuously.

### Why the original numbers were superseded

Two independent re-runs, then a 6-run series, all landed near 18 ms/step — roughly **1.9× faster
than the recorded 34.95**. Prefill moved by the same factor (82.39 → 45.74). A consistent ratio
across two independent metrics indicates a **systematic cause, not noise.**

**Cause: CPU contention. Confirmed by reproduction, not inferred.**

Ruled out first: thread count (identical binary, identical code path,
`GGML_DEFAULT_N_THREADS`=4 in every run) and terminal I/O (a run printing to a tty gave
19.68 ms/step, statistically indistinguishable from runs piped to `/dev/null`).

Then reproduced deliberately by adding competing CPU load, sampling MHz *during* each run:

| Condition | ms/step | MHz during run | Package temp |
|---|---|---|---|
| Clean machine | 18.06 – 18.79 | 3707 – 3734 | 30 °C |
| + 1 competing thread | 22.01 – 23.00 | 3424 – 3519 | 30 °C |
| **+ 4 competing threads** | **30.75 – 32.50** | 3002 – 3104 | 30 °C |
| *original recorded figure* | *34.95* | *not logged* | *unknown* |

Four competing threads land within 10% of the original 34.95, so a busy machine fully accounts
for it. The original was likely measured during a `-j3` build or alongside the runaway
`llama-cli` process that held a core for 18+ minutes that evening.

**An earlier draft of this document blamed thermal throttling. That was wrong.** Package
temperature held at 30 °C across every condition above — heat was never involved. The MHz drop
is turbo budget, not throttling: this part reaches ~4.2 GHz on one or two active cores but far
less across four, so *any* additional load lowers the achievable clock even on a cold machine.
Combined with straightforward CPU-time competition, that is the entire effect.

**Two consequences for benchmarking here:**

- The roadmap's `taskset` rule (engine on cores 0–2, control plane on core 3) is now
  empirically justified, not just prudent — a single competing thread costs ~22%.
- MHz must be sampled *during* runs, not after. The original number was undiagnosable purely
  because it wasn't, which is exactly why that standing rule exists.

**Carry forward:** the `-t 3` config (3 threads, reserving one core for the control plane) that
every other benchmark standardizes on has **still not been measured.** When Phase 1's `-t 3`
number comes in slower than **18.03 ms/step**, that is the expected cost of giving up a quarter
of the threads — not a regression to chase. Check the arithmetic against this number before
accepting any other explanation (hard-won rule 2).
