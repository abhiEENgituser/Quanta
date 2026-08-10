# Protocol

The llama.cpp API surface this project depends on, and the shim contract built on top of it.
This file grows with each phase — Phase 0 documents the memory-management API discovered while
designing the `Backend` interface; Phase 1 will add the wire protocol between `quantad` and
`shim/src/main.cpp`.

Source of truth: `shim/vendor/llama.cpp/include/llama.h` at tag **`b10121`**
(commit `555881ebc8b0fc0402b30e09258a32a7bfd13c52`). Every signature below was read from that
file directly, not from memory or online tutorials — several of which still show deprecated
function names (`llama_load_model_from_file`, `llama_new_context_with_model`, etc.).

---

## The six `llama_memory_*` signatures the `Backend` interface is built on

These are the operations the KV-cache block manager (Phase 4) needs, and the ones the
`Backend` interface's `Evict`/`Fork`/`PosRange` methods map onto directly:

```c
llama_memory_t llama_get_memory(const struct llama_context * ctx);
```
Accessor — every other call below needs this handle, obtained from a live context. There is no
way to manipulate KV-cache memory without first going through the context that owns it.

```c
bool llama_memory_seq_rm(llama_memory_t mem, llama_seq_id seq_id,
                          llama_pos p0, llama_pos p1);
```
Removes all tokens belonging to `seq_id` with positions in `[p0, p1)`. Maps to `Evict`.
**Can return `false`** — see Constraint 1.

```c
void llama_memory_seq_cp(llama_memory_t mem, llama_seq_id seq_id_src,
                          llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1);
```
Copies cached tokens `[p0, p1)` from one sequence to another — real prefix sharing, not
simulator-only. Maps to `Fork`. **Returns `void`** — see Constraint 2.

```c
void llama_memory_seq_keep(llama_memory_t mem, llama_seq_id seq_id);
```
Removes everything that does *not* belong to `seq_id`. Not currently wired into the `Backend`
interface; useful for a "drop everyone else, keep this one sequence" fast path.

```c
llama_pos llama_memory_seq_pos_min(llama_memory_t mem, llama_seq_id seq_id);
llama_pos llama_memory_seq_pos_max(llama_memory_t mem, llama_seq_id seq_id);
```
Query the actual min/max cached position for a sequence. Maps to `PosRange`. This is the
**verification primitive** — Go's bookkeeping about what it thinks is cached should be checked
against these rather than assumed, especially after any `seq_cp`/`seq_rm` call.

**Convention across all of these:** `seq_id < 0` matches any sequence; `p0 < 0` means
`[0, p1)`; `p1 < 0` means `[p0, ∞)`.

**Not currently used by the `Backend` interface** (documented for completeness, since they're
part of the same family): `llama_memory_seq_add` and `llama_memory_seq_div` — relative
position-shift operations used for context-window sliding. Deferred; revisit if/when sliding
context is in scope.

---

## Constraint 1 — `seq_rm` returns `bool` and can genuinely fail

The header comment says: *"Returns false if a partial sequence cannot be removed. Removing a
whole sequence never fails."* Reading the actual implementations (`src/llama-kv-cache.cpp` and
`src/llama-memory-recurrent.cpp`) to find out *when*, rather than assuming:

**For a standard KV cache (what Qwen2.5-0.5B uses)** — `llama_kv_cache::seq_rm` — every token
position is its own independent cache cell. Removing an arbitrary sub-range just deletes those
cells. In this commit, the standard-cache code path has no failure branch at all; it always
returns `true`. **For our actual model, `seq_rm` cannot fail.**

**The failure case is architecturally specific to recurrent-state models** (Mamba, RWKV, and
hybrid architectures that mix recurrent state with attention) — `llama_memory_recurrent::seq_rm`.
Their "cache" is not a per-token array; it's a single rolling hidden-state vector that gets
mutated in place at every step. Once token 5's information has been folded into that rolling
state, there is no cell holding "just token 5" to delete — the model has already
irreversibly merged it into what comes after. The only two operations physically possible are:
roll back to an earlier state that was explicitly snapshotted (bounded by a configured
`n_rs_seq` budget), or wipe the sequence entirely. If the requested `[p0, p1)` range doesn't
line up with an available snapshot, the call refuses and returns `false`.

**Why the `Backend` interface still returns `(bool, error)` from `Evict` even though today's
model can't trigger the failure:** the interface is meant to stay architecture-agnostic. A
fallback path (evict the whole sequence when a partial evict is refused) plus a counter for how
often it fires costs almost nothing to write now and means the block manager doesn't silently
break the day a recurrent-family model gets swapped in.

## Constraint 2 — `seq_cp` returns `void`, with no failure signal at all

Unlike `seq_rm`, there's no way to ask "did the copy actually happen." The documented mitigation
(and the one this project will use): **after every `Fork`, call `PosRange` on the destination
sequence and verify it matches what was expected.** Bookkeeping drift between Go's view of the
KV cache and the engine's real state is called out in the roadmap as the nastiest bug class
expected in Phase 4 — and it's nasty specifically because it presents as garbled generated text,
not a crash or an error return. Cheap to assert now; expensive to debug later without the habit
already in place.

---

## Related finding: `llama_decode`'s own partial-failure semantics

Found while reading the `pos_min`/`pos_max` documentation (line numbers drift across commits;
this was near `llama.h:960` at this tag, not the original estimate of `967`). Worth recording
because `Step()` will need to handle it:

```
llama_decode() return values:
   0 - success
   1 - could not find a KV slot for the batch (reduce batch size or increase context)
   2 - aborted (processed ubatches remain in the context's memory)
  -1 - invalid input batch
 < -1 - fatal error (processed ubatches remain in the context's memory)
```

On any outcome other than a clean `0`, the comment is explicit: **query `pos_min`/`pos_max` to
find out what the memory state actually is**, rather than assume either "nothing happened" or
"everything happened." Positive return values are warnings, not fatal — worth distinguishing
from the negative (fatal/invalid) cases when `Step()` decides whether to retry, drop the
sequence, or propagate an error up to the scheduler.

---

## Non-memory API surface (from `probe.cpp`)

The full function-by-function breakdown of what `probe.cpp` calls — model loading, tokenization,
context/batch setup, greedy sampling, cleanup — is not duplicated here in full; see the
inline explanation already given for that file. Summary of the categories used:

| Category | Functions |
|---|---|
| Backend lifecycle | `llama_backend_init`, `llama_backend_free` |
| Model | `llama_model_default_params`, `llama_model_load_from_file`, `llama_model_get_vocab`, `llama_model_free` |
| Tokenization | `llama_tokenize`, `llama_token_to_piece` |
| Context | `llama_context_default_params`, `llama_init_from_model`, `llama_free` |
| Batch | `llama_batch_init`, `llama_batch_free`, `llama_decode` |
| Sampling | `llama_sampler_chain_default_params`, `llama_sampler_chain_init`, `llama_sampler_init_greedy`, `llama_sampler_chain_add`, `llama_sampler_sample`, `llama_sampler_free` |

**Modern names note** (from the roadmap, confirmed while reading the header): use
`llama_model_load_from_file`, `llama_init_from_model`, `llama_model_get_vocab` — online
tutorials frequently show the deprecated `llama_load_model_from_file` /
`llama_new_context_with_model` forms, which still exist in this header only as
`DEPRECATED(...)` wrappers.
