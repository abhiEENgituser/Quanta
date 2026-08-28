package shim

import (
	"errors"
	"os"
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
	if _, err := os.Stat(sock); err != nil {
		t.Skipf("shim not running (no %s) — start it: ./shim/build/quanta_shim -m models/qwen2.5-0.5b-q4km.gguf", sock)
	}
	c, err := Dial(sock)
	if err != nil {
		t.Fatalf("dial: %v", err)
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
	reset(t, c, 9)

	_, err := c.Step([]engine.SeqID{9})
	var opErr *OpError
	if !errors.As(err, &opErr) {
		t.Fatalf("expected *OpError, got %T: %v", err, err)
	}
	// The connection must survive an op-level error.
	if _, err := c.Tokenize("still alive?", true); err != nil {
		t.Fatalf("connection did not survive op error: %v", err)
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
