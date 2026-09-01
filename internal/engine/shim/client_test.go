package shim

import (
	"errors"
	"sync"
	"testing"

	"github.com/abhiEENgituser/Quanta/internal/engine"
)

const sock = "/tmp/quanta.sock"

// The recorded greedy output for "The capital of France is" — same golden text
// probe.cpp produces and the Python suite verifies. Greedy sampling is
// deterministic: anything other than an exact match means the client corrupted
// something in transit.
const golden = " Paris. It is the largest city in Europe and the second largest in the world. It is also"

func dialOrSkip(t *testing.T) *Client {
	t.Helper()
	// Dial, don't Stat: a socket FILE outlives its process (a killed shim
	// never unlinks), so existence proves nothing about liveness. Any dial
	// failure means "no usable shim" and these integration tests skip.
	c, err := Dial(sock)
	if err != nil {
		t.Skipf("shim not reachable (%v) — start it: ./shim/build/quanta_shim -m models/qwen2.5-0.5b-q4km.gguf -q 4", err)
	}
	t.Cleanup(func() { c.Close() })
	return c
}

func reset(t *testing.T, c *Client, seq engine.SeqID) {
	t.Helper()
	if _, err := c.Evict(seq, 0, -1); err != nil {
		t.Fatalf("evict: %v", err)
	}
}

func TestGenerateMatchesProbe(t *testing.T) {
	c := dialOrSkip(t)
	reset(t, c, 0)

	toks, err := c.Tokenize("The capital of France is", true)
	if err != nil {
		t.Fatalf("tokenize: %v", err)
	}
	if len(toks) != 5 {
		t.Fatalf("expected 5 prompt tokens, got %d: %v", len(toks), toks)
	}

	if err := c.Prefill(0, toks, 0); err != nil {
		t.Fatalf("prefill: %v", err)
	}

	// Accumulate raw bytes; decode once at the end. Per-token decoding would
	// corrupt any multi-byte character split across tokens.
	var out []byte
	for i := 0; i < 20; i++ {
		res, err := c.Step([]engine.SeqID{0})
		if err != nil {
			t.Fatalf("step %d: %v", i, err)
		}
		if len(res.Tokens) != 1 {
			t.Fatalf("step %d: expected 1 token, got %d", i, len(res.Tokens))
		}
		out = append(out, res.Tokens[0].Piece...)
		if res.Tokens[0].Finished {
			break
		}
	}

	if string(out) != golden {
		t.Fatalf("output diverged from probe:\n got: %q\nwant: %q", out, golden)
	}
}

func TestPosRangeReflectsPrefill(t *testing.T) {
	c := dialOrSkip(t)
	reset(t, c, 0)

	toks, err := c.Tokenize("The capital of France is", true)
	if err != nil {
		t.Fatalf("tokenize: %v", err)
	}
	if err := c.Prefill(0, toks, 0); err != nil {
		t.Fatalf("prefill: %v", err)
	}

	min, max, err := c.PosRange(0)
	if err != nil {
		t.Fatalf("pos_range: %v", err)
	}
	if min != 0 || max != int32(len(toks))-1 {
		t.Fatalf("pos_range = [%d,%d], want [0,%d]", min, max, len(toks)-1)
	}
}

func TestStepWithoutPrefillIsOpError(t *testing.T) {
	c := dialOrSkip(t)
	// Seq 3 is in range for -q 4 but never prefilled by any test. (Seq 9 would
	// be rejected as out of range — see the next test.)
	reset(t, c, 3)

	_, err := c.Step([]engine.SeqID{3})
	var opErr *OpError
	if !errors.As(err, &opErr) {
		t.Fatalf("expected *OpError, got %T: %v", err, err)
	}
	// The connection must survive an op-level error.
	if _, err := c.Tokenize("still alive?", true); err != nil {
		t.Fatalf("connection did not survive op error: %v", err)
	}
}

// A seq id past n_seq_max used to reach llama.cpp, whose response is
// GGML_ASSERT -> abort(): one bad integer over the socket killed the server.
// The shim must reject it as an op error and stay alive.
func TestOutOfRangeSeqIsRejectedNotFatal(t *testing.T) {
	c := dialOrSkip(t)

	var opErr *OpError
	if _, err := c.Evict(99, 0, -1); !errors.As(err, &opErr) {
		t.Fatalf("evict(99): want *OpError, got %T: %v", err, err)
	}
	if _, err := c.Step([]engine.SeqID{99}); !errors.As(err, &opErr) {
		t.Fatalf("step(99): want *OpError, got %T: %v", err, err)
	}
	if err := c.Prefill(99, []int32{1, 2, 3}, 0); !errors.As(err, &opErr) {
		t.Fatalf("prefill(99): want *OpError, got %T: %v", err, err)
	}
	// The server survived all three.
	if _, err := c.Tokenize("still alive?", true); err != nil {
		t.Fatalf("shim died on out-of-range seq: %v", err)
	}
}

// TestInterleavedSequencesMatchSolo is the batching correctness check, and
// greedy determinism makes it exact: two sequences advanced in ONE llama_decode
// call per step must produce byte-identical output to each sequence run alone.
// Any cross-sequence contamination — wrong logits index, attention leaking
// across seq_ids, position bookkeeping drift — breaks the equality.
//
// Requires the shim started with -q 2 or more; skips otherwise.
func TestInterleavedSequencesMatchSolo(t *testing.T) {
	c := dialOrSkip(t)

	promptA := "The capital of France is"
	promptB := "The sky is"
	const steps = 12

	gen := func(seq engine.SeqID, prompt string) []byte {
		reset(t, c, seq)
		toks, err := c.Tokenize(prompt, true)
		if err != nil {
			t.Fatalf("tokenize: %v", err)
		}
		if err := c.Prefill(seq, toks, 0); err != nil {
			t.Fatalf("prefill: %v", err)
		}
		var out []byte
		for i := 0; i < steps; i++ {
			res, err := c.Step([]engine.SeqID{seq})
			if err != nil {
				t.Fatalf("solo step: %v", err)
			}
			out = append(out, res.Tokens[0].Piece...)
		}
		return out
	}

	soloA := gen(0, promptA)
	soloB := gen(1, promptB)

	// Now both at once: one Step call advances both sequences per iteration.
	reset(t, c, 0)
	reset(t, c, 1)
	toksA, _ := c.Tokenize(promptA, true)
	toksB, _ := c.Tokenize(promptB, true)
	if err := c.Prefill(0, toksA, 0); err != nil {
		t.Fatalf("prefill A: %v", err)
	}
	if err := c.Prefill(1, toksB, 0); err != nil {
		var opErr *OpError
		if errors.As(err, &opErr) {
			t.Skipf("shim not started with -q 2+: %v", err)
		}
		t.Fatalf("prefill B: %v", err)
	}

	var batchA, batchB []byte
	for i := 0; i < steps; i++ {
		res, err := c.Step([]engine.SeqID{0, 1})
		if err != nil {
			t.Fatalf("batched step %d: %v", i, err)
		}
		if len(res.Tokens) != 2 {
			t.Fatalf("step %d returned %d tokens, want 2", i, len(res.Tokens))
		}
		for _, tok := range res.Tokens {
			if tok.Seq == 0 {
				batchA = append(batchA, tok.Piece...)
			} else {
				batchB = append(batchB, tok.Piece...)
			}
		}
	}

	if string(batchA) != string(soloA) {
		t.Errorf("seq A diverged under batching:\n solo:  %q\n batch: %q", soloA, batchA)
	}
	if string(batchB) != string(soloB) {
		t.Errorf("seq B diverged under batching:\n solo:  %q\n batch: %q", soloB, batchB)
	}
}

// TestConcurrentCallers exists for -race: the protocol allows one in-flight
// request, so the client serialises with a mutex. 8 goroutines × 5 calls will
// trip the race detector if that serialisation is wrong.
func TestConcurrentCallers(t *testing.T) {
	c := dialOrSkip(t)

	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 5; i++ {
				if _, err := c.Tokenize("hello world", true); err != nil {
					t.Errorf("tokenize: %v", err)
					return
				}
			}
		}()
	}
	wg.Wait()
}
