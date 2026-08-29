package api

import (
	"bufio"
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/abhiEENgituser/Quanta/internal/engine"
	"github.com/abhiEENgituser/Quanta/internal/metrics"
)

// fakeBackend scripts exact token bytes — the Backend seam means the whole API
// layer is testable with no shim, no socket and no model. This is the same
// mechanism Phase 2 uses to swap in the cost-model backend.
type fakeBackend struct {
	pieces  [][]byte // Step returns these in order; last one has Finished=true
	step    int
	evicts  int
	prefill int
}

func (f *fakeBackend) Tokenize(text string, _ bool) ([]int32, error) {
	return make([]int32, 5), nil
}
func (f *fakeBackend) Prefill(engine.SeqID, []int32, int32) error {
	f.prefill++
	return nil
}
func (f *fakeBackend) Step([]engine.SeqID) (engine.StepResult, error) {
	p := f.pieces[f.step]
	fin := f.step == len(f.pieces)-1
	f.step++
	return engine.StepResult{Tokens: []engine.Token{
		{Seq: 0, ID: int32(f.step), Piece: p, Finished: fin},
	}}, nil
}
func (f *fakeBackend) Evict(engine.SeqID, int32, int32) (bool, error) {
	f.evicts++
	return true, nil
}
func (f *fakeBackend) PosRange(engine.SeqID) (int32, int32, error) { return 0, 4, nil }
func (f *fakeBackend) Close() error                                { return nil }

// fixed clock: metrics never read time themselves, so tests control every
// instant — 10ms per tick, fully deterministic.
func fakeClock() func() time.Time {
	t := time.Unix(1000, 0)
	return func() time.Time {
		t = t.Add(10 * time.Millisecond)
		return t
	}
}

func doGenerate(t *testing.T, be engine.Backend, reg *metrics.Registry, body string) []string {
	t.Helper()
	srv := New(be, reg, fakeClock(), Config{MaxTokens: 64, MaxContext: 2048})

	req := httptest.NewRequest("POST", "/v1/generate", strings.NewReader(body))
	rec := httptest.NewRecorder()
	srv.Routes().ServeHTTP(rec, req)

	if rec.Code != 200 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	if ct := rec.Header().Get("Content-Type"); ct != "text/event-stream" {
		t.Fatalf("content-type = %q, want text/event-stream", ct)
	}

	var lines []string
	sc := bufio.NewScanner(rec.Body)
	for sc.Scan() {
		if l := sc.Text(); l != "" {
			lines = append(lines, l)
		}
	}
	return lines
}

// The golden UTF-8 case: "東" is 3 bytes (0xE6 0x9D 0xB1) split across two
// tokens. The stream must never emit the fragment — text arrives only once the
// rune is complete.
func TestSplitRuneAssembledAcrossTokens(t *testing.T) {
	be := &fakeBackend{pieces: [][]byte{
		[]byte("Tokyo: "),
		{0xE6, 0x9D},       // first 2 bytes of 東 — must NOT be emitted alone
		{0xB1, ' ', 'o', 'k'}, // completes 東, then " ok"
	}}
	reg := metrics.NewRegistry()

	lines := doGenerate(t, be, reg, `{"prompt":"hi"}`)

	var text strings.Builder
	var finish string
	var tokens float64
	for i, l := range lines {
		if !strings.HasPrefix(l, "data: ") {
			continue
		}
		var m map[string]any
		if err := json.Unmarshal([]byte(strings.TrimPrefix(l, "data: ")), &m); err != nil {
			t.Fatalf("line %d not json: %s", i, l)
		}
		if s, ok := m["text"].(string); ok {
			text.WriteString(s)
		}
		if fr, ok := m["finish_reason"].(string); ok {
			finish = fr
			tokens = m["tokens"].(float64)
		}
	}

	if got := text.String(); got != "Tokyo: 東 ok" {
		t.Fatalf("assembled text = %q, want %q", got, "Tokyo: 東 ok")
	}
	if finish != "stop" || tokens != 3 {
		t.Fatalf("finish=%q tokens=%v, want stop/3", finish, tokens)
	}

	// No fragment may have crossed the wire as its own event.
	for _, l := range lines {
		if strings.Contains(l, `�`) {
			t.Fatalf("replacement char leaked — fragment was emitted: %s", l)
		}
	}
}

func TestMetricsRecordedPerRequest(t *testing.T) {
	be := &fakeBackend{pieces: [][]byte{[]byte("a"), []byte("b"), []byte("c")}}
	reg := metrics.NewRegistry()

	doGenerate(t, be, reg, `{"prompt":"hi"}`)
	snap := reg.Snapshot()

	if n := snap[metrics.HistTTFT].Count; n != 1 {
		t.Errorf("ttft count = %d, want 1", n)
	}
	if n := snap[metrics.HistITL].Count; n != 2 {
		t.Errorf("itl count = %d, want 2 (3 tokens -> 2 gaps)", n)
	}
	if n := snap[metrics.HistQueueWait].Count; n != 1 {
		t.Errorf("queue_wait count = %d, want 1", n)
	}
	// Sequence hygiene: evict before prefill (stale state) and after (free
	// memory) — a leak here is leaked KV under a clamped budget.
	if be.evicts != 2 || be.prefill != 1 {
		t.Errorf("evicts=%d prefill=%d, want 2/1", be.evicts, be.prefill)
	}
}

func TestPromptTooLongRejected(t *testing.T) {
	be := &fakeBackend{pieces: [][]byte{[]byte("x")}}
	srv := New(be, metrics.NewRegistry(), fakeClock(), Config{MaxTokens: 64, MaxContext: 60})

	req := httptest.NewRequest("POST", "/v1/generate", strings.NewReader(`{"prompt":"hi"}`))
	rec := httptest.NewRecorder()
	srv.Routes().ServeHTTP(rec, req)

	if rec.Code != 400 {
		t.Fatalf("status = %d, want 400 (5 prompt + 64 max > 60 ctx)", rec.Code)
	}
	if be.prefill != 0 {
		t.Fatalf("prefill ran despite rejection — admission must precede memory commitment")
	}
}
