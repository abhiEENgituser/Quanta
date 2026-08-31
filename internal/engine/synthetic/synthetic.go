// Package synthetic implements engine.Backend as a calibrated cost model —
// the "replace llama.cpp with a sleep()" the whole project design rests on.
//
// One implementation, two behaviors, chosen by the injected Clock:
//
//	clock.Real     -> the MOCK: Sleep actually blocks, so quantad serves
//	                  requests at realistic speed with no model loaded
//	clock.Virtual  -> the SIMULATOR: Sleep jumps, so a 40-hour sweep of
//	                  simulated decode finishes in wall-clock seconds
//
// Keeping mock and sim as one type is a locked roadmap decision: two separate
// implementations would drift apart, and then the simulator validates nothing.
//
// The backend is honest about its calibration boundaries: multi-sequence
// batches are refused (the batch-size cost curve is Phase 3 work), and
// chunked prefill is priced as an independent prefill (documented limit).
// A cost model that silently guesses outside its data is how simulations lie.
package synthetic

import (
	"fmt"
	"sync"

	"github.com/abhiEENgituser/Quanta/internal/clock"
	"github.com/abhiEENgituser/Quanta/internal/engine"
	"github.com/abhiEENgituser/Quanta/internal/engine/costmodel"
)

type seqState struct {
	start   int32 // first cached position (0 unless chunked prefill began later)
	nextPos int32 // next position to be written = tokens currently cached
}

// Backend is safe for concurrent use; like the real shim it serves one
// operation at a time (the real engine is serial — pretending otherwise would
// make the fake more capable than the thing it imitates).
type Backend struct {
	clk clock.Clock
	p   costmodel.Params

	mu   sync.Mutex
	seqs map[engine.SeqID]*seqState
}

var _ engine.Backend = (*Backend)(nil)

func New(clk clock.Clock, p costmodel.Params) *Backend {
	return &Backend{clk: clk, p: p, seqs: make(map[engine.SeqID]*seqState)}
}

// Tokenize models token counts, not token identity: one token per
// whitespace-separated word. That is exact for this project's canonical
// prompts ("The capital of France is" -> 5, matching the real tokenizer) and
// close enough for admission-control arithmetic, which is all the scheduler
// uses counts for. Cost is not modelled: tokenization is microseconds against
// decode steps of tens of milliseconds.
func (b *Backend) Tokenize(text string, _ bool) ([]int32, error) {
	var toks []int32
	inWord := false
	for _, r := range text {
		if r == ' ' || r == '\n' || r == '\t' {
			inWord = false
			continue
		}
		if !inWord {
			toks = append(toks, int32(len(toks)))
			inWord = true
		}
	}
	return toks, nil
}

func (b *Backend) Prefill(seq engine.SeqID, tokens []int32, startPos int32) error {
	if len(tokens) == 0 {
		return fmt.Errorf("synthetic: prefill: empty tokens")
	}

	b.mu.Lock()
	st, ok := b.seqs[seq]
	if ok && startPos < st.nextPos {
		b.mu.Unlock()
		// Mirrors the real engine's behaviour (decode_rc -1 on occupied
		// positions): reusing a sequence without evicting is a caller bug the
		// fake must also catch, or code passes against the mock and dies
		// against the shim.
		return fmt.Errorf("synthetic: prefill: positions %d.. already occupied (evict first)", startPos)
	}
	if !ok {
		st = &seqState{start: startPos}
		b.seqs[seq] = st
	}
	st.nextPos = startPos + int32(len(tokens))
	b.mu.Unlock()

	// Calibration measured whole prompts at startPos 0. A continuation chunk
	// is priced as if independent — attention back into the existing cache is
	// NOT modelled. Documented limit; revisit with Phase 3's chunked prefill.
	b.clk.Sleep(b.p.Prefill.At(float64(len(tokens))))
	return nil
}

func (b *Backend) Step(active []engine.SeqID) (engine.StepResult, error) {
	if len(active) == 0 {
		return engine.StepResult{}, fmt.Errorf("synthetic: step: empty active set")
	}
	if len(active) > 1 {
		// The batch-size cost curve does not exist yet (Phase 3 calibrates
		// it). Guessing here would produce confident nonsense about exactly
		// the question the project exists to answer.
		return engine.StepResult{}, fmt.Errorf(
			"synthetic: step: %d sequences, but the batch cost curve is not calibrated yet (Phase 3)",
			len(active))
	}
	seq := active[0]

	b.mu.Lock()
	st, ok := b.seqs[seq]
	if !ok {
		b.mu.Unlock()
		return engine.StepResult{}, fmt.Errorf("synthetic: step: sequence %d not prefilled", seq)
	}
	ctx := st.nextPos // tokens cached when this step runs — same x the calibration measured
	st.nextPos++
	b.mu.Unlock()

	b.clk.Sleep(b.p.Step.At(float64(ctx)))

	// Deterministic fake token: id is the position it landed at, text is a
	// legible placeholder. Finished is never set — output length is policy,
	// decided above the Backend interface, and the fake has no EOG opinion.
	piece := fmt.Sprintf(" t%d", ctx)
	return engine.StepResult{Tokens: []engine.Token{{
		Seq:   seq,
		ID:    ctx,
		Piece: []byte(piece),
	}}}, nil
}

func (b *Backend) Evict(seq engine.SeqID, p0, p1 int32) (bool, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	st, ok := b.seqs[seq]
	if !ok {
		return true, nil // removing nothing succeeds, as with the real cache
	}
	switch {
	case p0 <= 0 && p1 < 0: // full clear
		delete(b.seqs, seq)
	case p1 < 0: // drop suffix [p0, inf)
		if p0 < st.nextPos {
			st.nextPos = p0
		}
	default:
		// Interior ranges would leave holes the simple state cannot express;
		// the real standard KV cache can do it, but nothing in the project
		// calls it yet. Refuse rather than misrepresent.
		return false, fmt.Errorf("synthetic: evict: interior range [%d,%d) not supported", p0, p1)
	}
	return true, nil
}

func (b *Backend) PosRange(seq engine.SeqID) (int32, int32, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	st, ok := b.seqs[seq]
	if !ok || st.nextPos == st.start {
		return -1, -1, nil // empty, matching llama.cpp's convention
	}
	return st.start, st.nextPos - 1, nil
}

func (b *Backend) Close() error { return nil }
