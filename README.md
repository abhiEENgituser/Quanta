# Quanta

**An operating system for LLM inference.**

One model. One machine. More demand than capacity. Quanta is the control plane that decides
which requests run, who gets memory, and what to reject when capacity runs out.

---

## The problem

`llama.cpp` gives you one function: *advance these sequences by one token*. It is a library,
not a server — no queue, no clients, no idea that anyone is waiting.

On this laptop (CPU-only, Qwen2.5-0.5B Q4_K_M) one decode step costs **~18 ms**. A typical
128-token answer is therefore ~2.3 seconds of generation.

Now let 20 people ask at once. Serve them the obvious way — one at a time, finish before
starting the next:

| Person | Sees their first word at |
|---|---|
| 1st | 0.1 s |
| 10th | ~21 s |
| 20th | **~44 s** |

Nothing is broken and the model is not slow. **Nobody is deciding anything.** That is the gap
this project fills.

Two facts make it genuinely hard rather than merely fiddly:

- **Every active request's KV cache grows with every token it generates.** Memory demand is a
  moving target, not a fixed reservation.
- **You cannot know how long an answer will be until it ends.** No prediction, no planning
  around it. If output lengths were known up front this would be a textbook scheduling
  problem.

So you are allocating an unpredictable resource to an unpredictable set of demands, in real
time, with people waiting.

## What Quanta decides

| Decision | Question it answers |
|---|---|
| **Admission** | Start now, queue, or reject outright? |
| **Batching** | Who joins the active set this cycle? Who fills a freed slot? |
| **Memory** | How is KV cache handed out, tracked, and shared? |
| **Eviction** | Memory is full — whose cache dies, and do we recompute it or stash it? |
| **Fairness** | One tenant is saturating the machine. Who still gets served? |

Concretely, the program is a loop. Every decode cycle:

```
anybody finish?        → free their KV memory
memory available?      → admit someone from the queue
out of memory?         → pick a victim, evict
anyone starved?        → move them up
                       → tell the engine: "advance sequences [3, 7, 11, 14]"
```

Every one of those decisions has a cost to somebody. Batch more sequences and throughput
rises while streaming gets choppier for everyone already running. Admit more and utilization
improves while queues lengthen. Evict someone and you free memory now but pay to recompute
later. **There is no setting that avoids these tensions** — the deliverable is measured curves
and a defensible argument about where to sit on them, not a single "optimized" number.

## Why this is a systems project, not an AI project

Delete `llama.cpp`. Replace it with `sleep(18ms)`, calibrated against real measurements.
**The scheduler cannot tell the difference** — because the scheduler was never doing AI work.

It never reads a token, never touches a logit, never multiplies a weight. The only facts it
knows about a request are: it is active and holds N blocks of memory, it has waited M
milliseconds, its owner has consumed X of their quota, and it is interactive or batch class.

Those are the facts an operating system knows about a process:

| OS | Quanta |
|---|---|
| processes | requests |
| CPU time slices | decode steps |
| RAM pages | KV cache blocks |
| page tables | per-sequence block tables |
| swap out a process | evict a sequence's cache |
| refuse to fork, out of memory | reject the request |

That substitution is not a rhetorical trick — it is the plan. Real inference on a 4-core
laptop is too slow to be a measurement instrument (a full policy sweep would run ~40 hours),
so the engine is calibrated once and then replaced by a cost model for the experiments.

## Borrowed vs. built here

**Borrowed:** `llama.cpp` provides `llama_decode` and owns the KV cache *mechanism* —
computing keys and values, storing them, using them in attention. That is already efficient
and there is no reason to rewrite it.

**Built here:** everything that decides *policy* over that mechanism — queues, batch
formation, the block allocator, victim selection, fairness rules, admission and rejection.
None of it exists in `llama.cpp`.

The boundary in one line: **llama.cpp is the MMU; Quanta is the allocator and the scheduler.**

`llama-server` is deliberately **not** used. It performs its own continuous batching
internally, which would mean benchmarking someone else's policy and calling it ours.

## Honest positioning

vLLM already does all of this, better, in ~100k lines. Quanta is ~5k lines of Go built to
understand the policy space — the same reason one writes a hash table from scratch. No claim
of novelty, and PagedAttention is vLLM's idea, not this project's.

The one area that is genuinely under-explored in open serving stacks is **multi-tenant
fairness across SLO classes**, and that is where this goes deepest.

## Architecture

```
        HTTP clients (many)
              |
     Go: admission control          <- SLO check, load shedding
              |
     Go: scheduler  <-->  KV block manager    <- queues, fairness,
              |                                  preemption, block tables
              |  unix socket (4-byte length prefix + JSON)
        C++ shim                     <- step-wise decode. zero policy.
              |  function call
     llama.cpp                       <- a library, not a server
```

Two decisions worth naming:

- **Separate process over a Unix socket, not cgo.** A decode step is ~18 ms and a socket
  round-trip is microseconds, so the isolation is nearly free — and if the engine segfaults,
  the scheduler's state survives.
- **`step` is synchronous and advances exactly one decode pass.** Go controls the clock. This
  is what keeps the shim a dumb executor rather than a second scheduler competing with the
  first.

## Status

**Phase 0 complete — engine and API validated.** `llama.cpp` pinned at tag `b10121`.
[`shim/src/probe.cpp`](shim/src/probe.cpp) drives generation one token at a time through 20
separate `llama_decode` calls, which is the thing the whole design depends on: if generation
could not be driven step by step, there would be nowhere for a scheduler to live.

Recorded: **18.03 ms/step**, single sequence, n=5 with a 3.7% spread. See
[`docs/baseline.md`](docs/baseline.md) for
the full environment and [`docs/protocol.md`](docs/protocol.md) for the exact `llama.cpp` API
surface this depends on, including where it can fail.

**Phase 1 next** — single-stream server, socket protocol, and the measurement instrument
(HDR histograms, TTFT split into queue wait vs. prefill, open-loop load generator).
Deliberately no scheduling yet: the instrument has to be trustworthy before any policy number
measured with it means anything.

Later phases, in order: cost model and virtual clock (so policy sweeps take minutes instead of
40 hours of real inference), continuous batching measured against a static-batching baseline,
the KV block manager with prefix sharing and preemption, multi-tenant fairness across SLO
classes, then trace replay and comparison against vLLM.

## Repo map

```
shim/
  src/probe.cpp        minimal token-by-token reproducer — kept permanently
  src/main.cpp         the socket server (Phase 1)
  vendor/llama.cpp     git submodule, pinned to b10121
docs/
  baseline.md          environment + reference numbers
  protocol.md          llama.cpp API surface used, and its failure modes
models/                gitignored — symlinks to GGUF weights
```

Go packages (`sched/`, `kv/`, `tenant/`) arrive with Phase 1 and after. This repo grows one
phase at a time rather than starting as an empty directory tree.
