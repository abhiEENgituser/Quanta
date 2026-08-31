# Cost model

The calibrated `sleep()` that replaces the real engine in simulations. This
document records how it was fitted, how well it validates, and — most
importantly — where its claims end.

## Why it exists: the 40-hour arithmetic

A policy sweep worth running is ~20 configs × 5 repeats × 500 requests. At
~128 output tokens per request and ~25 ms/step under load, that is roughly
**40 hours of continuous decode** on a 4-core laptop that thermally cannot
hold peak clocks for 40 hours — the instrument would drift more than the
policies differ. Against the model on a virtual clock, the same sweep is
minutes, and *exact*: in a discrete-event system nothing happens between
events, so skipping the waiting loses no information.

## The model

Two lines, both in microseconds (`internal/engine/costmodel/params.json`,
regenerate with `make calibrate`):

```
prefill(n)  = a + b·n      n   = prompt tokens in one Prefill call
step(ctx)   = c + d·ctx    ctx = tokens already cached when the step runs
```

The context term in `step` is not optional: measured decode grows from ~17 to
~24 ms/step across 40→440 tokens of context, because each step's attention
scans every cached key. (An earlier n=1 claim that decode was flat died on
repeats; the growth survived them.)

## What is calibrated, exactly

**Backend op durations, measured through the real shim client** — socket
round-trip and JSON included — because the synthetic backend must impersonate
the engine *as the scheduler sees it*, not as a bare-metal probe sees it.

Conditions are part of the definition: `-t 3`, engine pinned to cores 0–2,
calibrator on core 3, performance governor (enforced by
`bench/configs/calibrate.sh`), and **thermal steady state** — 60 s of
sustained load before anything is recorded, because a serving engine is
continuously hot and hot equilibrium is the honest operating point. The
fitted parameters carry their own conditions (`meta.mhz_mean`, `engine_args`)
so a future reader can tell what machine state they describe.

## The calibration failure worth remembering (v1)

The first calibrator measured on a machine that heated as the sweep ran:
prefill@512 drifted 3243 → 4165 ms across sweeps (+21%). Worse, lengths ran
in ascending order, so **heat correlated with context length and poisoned the
step slope itself** — a confounded variable, R² 0.44, slope meaningless. The
diagnostics caught it; the fix became the method: thermal warmup to
equilibrium, per-sweep shuffled length order, MHz sampled during recording
with a decline warning, per-x medians across sweeps.

## Fit quality (current params)

```
prefill(n) = -9737us + 8355.8us·n    R² = 0.9998   max|res| = 43.9ms @ n≤512
step(ctx)  = 24122us + 9.02us·ctx    R² = 0.7637   max|res| = 3.2ms
```

- Prefill's negative intercept is the line absorbing mild superlinearity, not
  a real negative cost; `Line.At()` clamps at zero.
- Step's R² is mediocre and flagged at fit time. Per-step samples carry
  RTT/scheduler jitter and post-prefill turbo dips that medians of three
  sweeps cannot fully remove. Validation (below) is the arbiter of whether it
  matters.
- Hot steady state is slower than the cold numbers in `docs/baseline.md`
  (prefill slope 8.4 vs 6.0 µs/token; step intercept 24 vs ~17 ms). Neither is
  wrong — they describe different thermal regimes, and the model deliberately
  describes the serving regime.

## Validation: held-out lengths, hard threshold

`make validate` measures the real engine at prompt lengths **24/96/192/384 —
never used in fitting** — and compares against pure prediction. Agreement on
training points would prove only that least squares works; agreement in the
gaps tests the claim that cost is actually linear. Exit code is the verdict
(worst end-to-end error > 10% fails). Result:

| length | measured | predicted | error |
|---|---|---|---|
| 24 | 1071.4 ms | 974.1 ms | −9.1% |
| 96 | 1702.9 ms | 1596.5 ms | −6.3% |
| 192 | 2545.8 ms | 2426.4 ms | −4.7% |
| 384 | 4204.5 ms | 4086.1 ms | −2.8% |

**PASSED** (worst −9.1%, mean 5.7%). Plot: `bench/results/validation.png`.

Two properties of the error worth knowing when reading simulated results:

- **The bias is systematic and optimistic**: the model underpredicts latency
  by 3–9%, worst at short prompts (the fixed-overhead end of the lines).
  Absolute simulated latencies are ~5% rosy; *relative* comparisons between
  policies under the same model are largely unaffected.
- Per-op errors reach −14% but partially cancel over a full generation;
  end-to-end — the quantity a simulation integrates — stays within −9%.

## Where the model's claims end

Valid: single-sequence prefill and decode, prompt lengths ~16–512, context
to ~544, hot steady state, `-t 3` pinned config, this machine, this model
file. Outside that, it is guessing:

- **Batch size**: `synthetic.Step` refuses more than one sequence rather than
  guess — the batch cost curve is Phase 3's calibration, and it is the very
  quantity batching policies will be judged on.
- **Chunked prefill** is priced as an independent prefill of the chunk;
  attention back into existing cache is not modelled. Revisit in Phase 3.
- **Tokenize** is not cost-modelled (microseconds against 25 ms steps); token
  counts are approximated one-per-word, exact for the canonical test prompts.
- **Extrapolation** past ~550 context is unvalidated; the superlinearity that
  bent prefill's intercept will bend further out.
- **EOG**: the synthetic never finishes a sequence on its own — output length
  is policy, owned above the Backend interface.

Recalibrate (`make calibrate` + `make validate`) after: llama.cpp version
bumps, model file changes, thread-config changes, or any hardware change.
The params file's `meta` block says what it was fitted on; trust nothing it
does not claim.
