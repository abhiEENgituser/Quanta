# Protocol

The llama.cpp API surface this project depends on, and the shim contract built on top of it.
Phase 0 documents the memory-management API discovered while designing the `Backend` interface;
Phase 1 defines the wire protocol between `quantad` and `shim/src/main.cpp`.

Source of truth: `shim/vendor/llama.cpp/include/llama.h` at tag **`b10121`**
(commit `555881ebc8b0fc0402b30e09258a32a7bfd13c52`). Every signature below was read from that
file directly, not from memory or online tutorials — several of which still show deprecated
function names (`llama_load_model_from_file`, `llama_new_context_with_model`, etc.).

---

# Part 1 — The shim wire protocol (Phase 1)

The contract between `quantad` (Go) and `shim/src/main.cpp`. Both sides implement against this
document; neither side's behaviour is the specification.

## Transport

A **Unix domain socket**, not TCP. There is no network involved, so a network stack would add
latency, a port to collide on, and an attack surface for nothing in return. Filesystem
permissions become the access control.

**One connection, strictly synchronous.** `quantad` sends a request and reads exactly one
response before sending anything else. The shim handles a single connection with a blocking
accept loop — no threads, no `poll`, no concurrency of any kind. This is not a simplification to
be revisited later: it is what keeps the shim a dumb executor. Concurrency in the shim would mean
the shim deciding what runs when, which is the scheduler's job.

## Framing

```
[4-byte length, little-endian, unsigned][JSON body]
```

The length counts **the JSON body only** — it does not include the 4 prefix bytes.

Framing exists because a stream socket has no message boundaries. `read()` returns whatever
bytes happen to have arrived: half a message, two messages, or one and a half. Without an
explicit length there is no way to know where one message ends.

**Both sides must loop on read and write.** A single `read()` may return fewer bytes than asked
for, and a single `write()` may accept fewer bytes than offered — even on a local socket, even
for small messages. In Go use `io.ReadFull`; in C++ loop until the requested count is satisfied
or the peer closes. Treating a short read as a complete message is the classic socket bug and
presents as JSON parse failures under load rather than at rest.

**Maximum frame size: 8 MiB.** A length prefix larger than this is a protocol violation — close
the connection rather than attempting to allocate. An attacker-controlled or corrupted length is
otherwise an allocation of arbitrary size.

## Envelope

Every request carries an `op`. Field names are `snake_case`. Unknown fields must be **ignored**,
not rejected, so one side can add a field before the other consumes it.

Every response carries `ok`:

```json
{"ok": true,  ...payload}
{"ok": false, "error": "human-readable description"}
```

The distinction that matters is **whether the stream position is still trustworthy**, because that
decides whether the connection can continue:

| Condition | Response |
|---|---|
| Valid frame, body is not valid JSON | `ok: false` — keep the connection |
| Valid JSON, unknown `op` | `ok: false` — keep the connection |
| Valid `op`, engine call failed | `ok: false` — keep the connection |
| Length prefix exceeds `MAX_FRAME` | **close** — the length is untrustworthy, so the next frame boundary is unknown |
| EOF or error mid-frame | **close** — the peer is gone or the stream is truncated |

A malformed *body* is recoverable: the length prefix already established where the message ended,
so the next frame starts exactly where expected. Only a bad *length* desynchronises the stream,
because after that there is no way to know where one message stops and the next begins.

## Messages

### `tokenize`

```json
→ {"op":"tokenize","text":"The capital of France is","add_special":true}
← {"ok":true,"tokens":[791,6864,315,9822,374]}
```

Deliberately separate from `prefill`, because Go needs the prompt's token count to make an
admission decision *before* committing any KV memory. If the shim tokenized implicitly during
prefill, that number would never reach the scheduler.

`add_special` controls BOS/EOS insertion. `parse_special` is not exposed — the shim always passes
`true`, since text arriving from a client should have special-token markup interpreted
consistently.

### `prefill`

```json
→ {"op":"prefill","seq":0,"tokens":[791,6864,315,9822,374],"start_pos":0}
← {"ok":true}
```

Submits tokens for one sequence in a single `llama_decode` call, requesting logits **only for the
final token** — the model computes hidden states for every position regardless, but materializing
a vocabulary-sized logits vector for a position whose prediction is discarded is wasted work.

`start_pos` exists so that a long prompt can be submitted in several `prefill` calls at
increasing positions. That is chunked prefill (Phase 3), and having the field now means the
protocol does not change to support it.

### `step`

```json
→ {"op":"step","active":[0]}
← {"ok":true,"tokens":[{"seq":0,"id":12366,"piece_b64":"IFBhcmlz","finished":false}]}
```

Advances **exactly one decode pass** across the listed sequences. Go decides who is in `active`
and when to call; the shim never chooses.

`active` is an array even though Phase 1 only ever sends one sequence. Phase 3 needs multiple,
and widening the wire format later means changing both implementations at once.

`tokens` is an **array, not a map keyed by sequence id** — JSON object keys are strings, which
would force a string-to-int conversion on the Go side for no benefit.

`piece_b64` is base64 of the **raw bytes** from `llama_token_to_piece`. Those bytes are not
necessarily valid UTF-8: a single token can be a fragment of a multi-byte character, and JSON
strings must be valid UTF-8, so a partial character cannot legally be sent as a JSON string at
all. It would work on English ASCII and corrupt on anything else. Go accumulates the bytes and
decodes complete characters itself — which it has to do regardless, since an SSE frame must not
carry a split character either.

`finished` means **the engine emitted an end-of-generation token** (`llama_vocab_is_eog`) and
nothing else. Maximum output length, timeouts, and cancellation are policy, so they belong to Go.
A shim that stopped a sequence because it hit some length limit would be making a scheduling
decision.

On a failed decode, the raw return code is passed through:

```json
← {"ok":false,"error":"llama_decode failed","decode_rc":1}
```

That distinction matters: `1` means no KV slot was available and is potentially recoverable by
freeing memory and retrying, while `-1` means the batch itself was invalid and retrying is
pointless. See *`llama_decode`'s own partial-failure semantics* below.

### `evict`

```json
→ {"op":"evict","seq":0,"p0":0,"p1":-1}
← {"ok":true,"removed":true}
```

Maps directly to `llama_memory_seq_rm`. Position conventions are llama.cpp's, deliberately
unchanged so there is no translation layer to get wrong: `p0 < 0` means `[0, p1)`, `p1 < 0` means
`[p0, ∞)`.

`removed` is reported **separately from `ok`** because they answer different questions. `ok`
means the call executed; `removed` is `seq_rm`'s own return value, meaning "was this removal
possible." Collapsing them would discard the signal the block manager's fallback path depends on
(see Constraint 1).

In Phase 1 this is how a sequence gets reset between requests: `{"seq":0,"p0":0,"p1":-1}` clears
it completely.

## Engine state outlives the connection

Sequences, their cached tokens, and their positions belong to the **shim process**, not to the
socket connection. Disconnecting and reconnecting does not reset anything.

This is deliberate — a scheduler would not want a transient connection drop to silently discard
every in-flight sequence's KV cache. But it means the client owns cleanup: prefilling into a
position range a sequence already occupies fails with `decode_rc: -1` (invalid batch), not with a
helpful message about reuse. Either `evict` before reusing a sequence id, or allocate a fresh one.

## Not in Phase 1

`fork` (prefix sharing via `seq_cp`) and `pos_range` (bookkeeping verification via
`pos_min`/`pos_max`) are part of the `Backend` interface but are not needed until Phase 4. They
are omitted rather than stubbed — an unimplemented message that returns success is worse than one
that does not exist.

## Naming note

The roadmap's Phase 1 task list calls the second message `admit`. This document uses `prefill`,
matching the `Backend` interface. *Admission* is a decision Go makes — whether to start a request
at all. Naming a shim message after a policy it does not own invites someone later to put policy
behind it.

---

# Part 2 — llama.cpp API surface (Phase 0)

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
