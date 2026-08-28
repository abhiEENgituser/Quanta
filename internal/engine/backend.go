// Package engine defines the inference backend interface. Everything above this
// package (scheduler, KV manager, API) imports engine and never a concrete
// implementation — that seam is what lets the real llama.cpp shim, the
// cost-model mock, and the virtual-clock simulator swap with wiring changes only.
package engine

// SeqID identifies one sequence (one request's KV-cache lane) in the engine.
// Mirrors llama_seq_id, which is int32.
type SeqID int32

// Token is one generated token from one sequence.
type Token struct {
	Seq SeqID
	ID  int32
	// Piece is the token's raw bytes. Deliberately NOT a string: a single token
	// can be a fragment of a multi-byte UTF-8 character (confirmed empirically —
	// byte-level BPE fallback splits non-ASCII across tokens). Callers that
	// stream text must accumulate bytes and emit only complete characters.
	Piece []byte
	// Finished means the engine emitted an end-of-generation token. Nothing
	// else: length caps, timeouts and cancellation are policy, decided above
	// this interface.
	Finished bool
}

// StepResult carries one decode pass's output for every stepped sequence.
type StepResult struct {
	Tokens []Token
}

// Backend is the step-wise decode contract. It is intentionally dumb: the
// caller decides who is admitted, who steps, and who is evicted; the backend
// only executes. Phase 1 exposes exactly what the shim protocol implements —
// Fork (prefix sharing) joins in Phase 4 when the fork op exists end to end.
type Backend interface {
	// Tokenize converts text to token ids. Separate from Prefill because the
	// caller needs the token count to make an admission decision before
	// committing any KV memory.
	Tokenize(text string, addSpecial bool) ([]int32, error)

	// Prefill submits prompt tokens for one sequence in a single decode pass.
	// startPos > 0 continues an earlier partial prefill (chunked prefill).
	Prefill(seq SeqID, tokens []int32, startPos int32) error

	// Step advances every listed sequence by exactly one decode pass.
	Step(active []SeqID) (StepResult, error)

	// Evict removes cached tokens for seq in [p0, p1). llama.cpp conventions:
	// p0 < 0 means [0, p1); p1 < 0 means [p0, inf). removed=false means the
	// engine refused (possible on recurrent architectures; never for a standard
	// KV cache) — distinct from err, which means the call itself failed.
	Evict(seq SeqID, p0, p1 int32) (removed bool, err error)

	// PosRange reports the [min, max] cached positions the engine actually
	// holds for seq. This is the verification primitive: bookkeeping above this
	// interface should be asserted against it, not assumed.
	PosRange(seq SeqID) (min, max int32, err error)

	Close() error
}
