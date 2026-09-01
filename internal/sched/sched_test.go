package sched

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/abhiEENgituser/Quanta/internal/clock"
	"github.com/abhiEENgituser/Quanta/internal/engine/costmodel"
	"github.com/abhiEENgituser/Quanta/internal/engine/synthetic"
)

// Round-number params so every expected instant is exact mental arithmetic:
// prefill = 100ms flat, solo step = 10ms flat, batched step = 10ms + 5ms·B.
func testParams() costmodel.Params {
	var p costmodel.Params
	p.Prefill = costmodel.Line{InterceptUS: 100_000, SlopeUS: 0, N: 6, R2: 1}
	p.Step = costmodel.Line{InterceptUS: 10_000, SlopeUS: 0, N: 100, R2: 1}
	p.StepBatch = costmodel.Line{InterceptUS: 10_000, SlopeUS: 5_000, N: 18, R2: 1}
	p.BatchRefCtx = 4
	return p
}

type env struct {
	clk *clock.Virtual
	s   *Scheduler
	log []string
}

func newEnv(t *testing.T, cfg Config) *env {
	t.Helper()
	clk := clock.NewVirtual(time.Unix(0, 0))
	be := synthetic.New(clk, testParams())
	s, err := New(be, clk, cfg)
	if err != nil {
		t.Fatal(err)
	}
	return &env{clk: clk, s: s}
}

// submit adds a request whose lifecycle events land in the env log with their
// virtual timestamps — the deterministic event trace the assertions read.
func (e *env) submit(id string, promptTokens, maxTokens int) {
	r := &Request{
		ID:        id,
		Tokens:    make([]int32, promptTokens),
		MaxTokens: maxTokens,
		OnDone: func(reason string) {
			e.log = append(e.log, fmt.Sprintf("%dms %s done(%s)",
				e.clk.Now().Sub(time.Unix(0, 0)).Milliseconds(), id, reason))
		},
	}
	if err := e.s.Submit(r); err != nil {
		panic(err)
	}
}

func (e *env) run(t *testing.T, maxTicks int) {
	t.Helper()
	for i := 0; i < maxTicks; i++ {
		worked, err := e.s.Tick()
		if err != nil {
			t.Fatalf("tick %d: %v", i, err)
		}
		if !worked {
			return
		}
	}
	t.Fatalf("still working after %d ticks", maxTicks)
}

func (e *env) assertLog(t *testing.T, want ...string) {
	t.Helper()
	if len(e.log) != len(want) {
		t.Fatalf("log has %d events, want %d:\n got: %v\nwant: %v", len(e.log), len(want), e.log, want)
	}
	for i := range want {
		if e.log[i] != want[i] {
			t.Fatalf("event %d:\n got: %s\nwant: %s\nfull: %v", i, e.log[i], want[i], e.log)
		}
	}
}

// Continuous batching, one request: prefill 100ms + 3 solo steps at 10ms.
// Done at exactly 130ms — the virtual clock makes "exactly" literal.
func TestSingleRequestExactTiming(t *testing.T) {
	e := newEnv(t, Config{Slots: 2, Policy: Continuous})
	e.submit("a", 4, 3)
	e.run(t, 10)
	e.assertLog(t, "130ms a done(length)")
}

// Two requests, two slots, continuous: both admitted in the first tick
// (prefill 100+100 = 200ms, sequential — the engine is serial), then batched
// steps at 10+5·2 = 20ms each. a needs 2 steps (done at 240ms), b needs 4:
// two batched (240ms) then two solo at 10ms (260ms).
func TestContinuousBatchesAndDrains(t *testing.T) {
	e := newEnv(t, Config{Slots: 2, Policy: Continuous})
	e.submit("a", 4, 2)
	e.submit("b", 4, 4)
	e.run(t, 10)
	e.assertLog(t,
		"240ms a done(length)",
		"260ms b done(length)",
	)
}

// THE policy difference, as exact timestamps. Three requests, two slots.
//
// Static: [a,b] run to completion (240ms, per above), only THEN is c admitted:
// prefill ends 340ms, 2 solo steps -> done 360ms.
//
// Continuous: c takes a's slot the moment it frees at 240ms — same engine
// arithmetic here, but the admission DECISION happens a full batch earlier in
// queue-wait terms under load. With these numbers the completion times tie;
// the trace shows c's admission (its prefill) starting at 240ms either way —
// the difference explodes when more work is queued behind (see next test).
func TestStaticWaitsForWholeBatch(t *testing.T) {
	e := newEnv(t, Config{Slots: 2, Policy: Static})
	e.submit("a", 4, 2)
	e.submit("b", 4, 4)
	e.submit("c", 4, 2)
	e.run(t, 20)
	e.assertLog(t,
		"240ms a done(length)",
		"260ms b done(length)",
		"380ms c done(length)", // admitted only after b drained: 260+100 prefill+2·10
	)
}

func TestContinuousRefillsFreedSlot(t *testing.T) {
	e := newEnv(t, Config{Slots: 2, Policy: Continuous})
	e.submit("a", 4, 2)
	e.submit("b", 4, 4)
	e.submit("c", 4, 2)
	e.run(t, 20)
	// a frees its slot at 240ms; c is admitted immediately (prefill to 340ms)
	// while b keeps decoding IN THE SAME BATCH: b's remaining 2 steps run
	// batched with c at 20ms each -> b done 380... walk it precisely:
	// t=240: a done. tick: admit c (prefill 240->340), step [b,c] 20ms -> 360.
	// b produced 3/4. tick: step [b,c] -> 380: b done(4/4), c produced 2/2 done.
	e.assertLog(t,
		"240ms a done(length)",
		"380ms b done(length)",
		"380ms c done(length)",
	)
}

// The slot ledger must balance through cancellation: a cancelled active
// request frees its slot for the queue.
func TestCancellationFreesSlot(t *testing.T) {
	e := newEnv(t, Config{Slots: 1, Policy: Continuous})

	cancelled := false
	r := &Request{
		ID: "a", Tokens: make([]int32, 4), MaxTokens: 100,
		Cancelled: func() bool { return cancelled },
		OnDone: func(reason string) {
			e.log = append(e.log, fmt.Sprintf("%dms a done(%s)",
				e.clk.Now().Sub(time.Unix(0, 0)).Milliseconds(), reason))
		},
	}
	if err := e.s.Submit(r); err != nil {
		t.Fatal(err)
	}
	e.submit("b", 4, 1)

	// Two ticks: a admitted+prefilled+stepped, b still queued (1 slot).
	for i := 0; i < 2; i++ {
		if _, err := e.s.Tick(); err != nil {
			t.Fatal(err)
		}
	}
	if e.s.ActiveCount() != 1 || e.s.QueueDepth() != 1 {
		t.Fatalf("precondition: active=%d queue=%d, want 1/1", e.s.ActiveCount(), e.s.QueueDepth())
	}

	cancelled = true
	e.run(t, 10)

	if len(e.log) != 2 || !strings.Contains(e.log[0], "a done(cancelled)") {
		t.Fatalf("first event should be a's cancellation, got %v", e.log)
	}
	if !strings.Contains(e.log[1], "b done(length)") {
		t.Fatalf("b never completed after slot freed: %v", e.log)
	}
}

// Determinism is the property everything else here rests on: two identical
// runs must produce byte-identical event traces, timestamps included.
func TestSchedulerIsDeterministic(t *testing.T) {
	run := func() []string {
		e := newEnv(t, Config{Slots: 3, Policy: Continuous})
		for i := 0; i < 7; i++ {
			e.submit(fmt.Sprintf("r%d", i), 4+i, 2+i%3)
		}
		e.run(t, 50)
		return e.log
	}
	a, b := run(), run()
	if len(a) != len(b) {
		t.Fatalf("different event counts: %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("traces diverge at %d: %q vs %q", i, a[i], b[i])
		}
	}
}
