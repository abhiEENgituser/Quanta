package synthetic

import (
	"testing"
	"time"

	"github.com/abhiEENgituser/Quanta/internal/clock"
	"github.com/abhiEENgituser/Quanta/internal/engine"
	"github.com/abhiEENgituser/Quanta/internal/engine/costmodel"
)

// Round numbers so expected durations are exact arithmetic, not fit output.
func testParams() costmodel.Params {
	var p costmodel.Params
	p.Prefill = costmodel.Line{InterceptUS: 10_000, SlopeUS: 8_000, N: 6, R2: 1}
	p.Step = costmodel.Line{InterceptUS: 24_000, SlopeUS: 10, N: 100, R2: 1}
	return p
}

// The Phase 2 headline, as a test: a workload that would take ~28 wall-clock
// minutes runs in microseconds of real time, and virtual time advances by
// EXACTLY the model's prediction — the jump is exact, not approximate.
func TestSimulatedGenerationIsExactAndInstant(t *testing.T) {
	v := clock.NewVirtual(time.Unix(0, 0))
	be := New(v, testParams())

	realStart := time.Now()

	if err := be.Prefill(0, make([]int32, 512), 0); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 100; i++ {
		if _, err := be.Step([]engine.SeqID{0}); err != nil {
			t.Fatal(err)
		}
	}

	// prefill: 10ms + 8ms*512 = 4.106s
	// steps:   sum over ctx=512..611 of (24ms + 10us*ctx) = 100*24ms + 10us*sum(512..611)
	wantPrefill := 4106 * time.Millisecond
	wantSteps := 100*24*time.Millisecond + time.Duration(10*(512+611)*100/2)*time.Microsecond
	want := wantPrefill + wantSteps

	if got := v.Now().Sub(time.Unix(0, 0)); got != want {
		t.Fatalf("virtual elapsed %v, want exactly %v", got, want)
	}
	if real := time.Since(realStart); real > 100*time.Millisecond {
		t.Fatalf("simulation took %v of real time — the clock jump is the whole point", real)
	}
}

// Same type, Real clock = the mock: Sleep must actually block.
func TestMockModeActuallySleeps(t *testing.T) {
	var p costmodel.Params
	p.Prefill = costmodel.Line{InterceptUS: 20_000, N: 1, R2: 1} // flat 20ms
	p.Step = costmodel.Line{InterceptUS: 5_000, N: 1, R2: 1}

	be := New(clock.Real{}, p)
	start := time.Now()
	if err := be.Prefill(0, []int32{1, 2, 3}, 0); err != nil {
		t.Fatal(err)
	}
	if elapsed := time.Since(start); elapsed < 15*time.Millisecond {
		t.Fatalf("mock prefill returned in %v — it must really sleep ~20ms", elapsed)
	}
}

func TestRefusesUncalibratedBatch(t *testing.T) {
	be := New(clock.NewVirtual(time.Unix(0, 0)), testParams())
	_ = be.Prefill(0, []int32{1}, 0)
	_ = be.Prefill(1, []int32{1}, 0)

	if _, err := be.Step([]engine.SeqID{0, 1}); err == nil {
		t.Fatal("multi-sequence step must be refused until the batch curve is calibrated")
	}
}

// The fake must catch the same caller bug the real engine catches, or code
// passes against the mock and dies against the shim.
func TestPrefillOverOccupiedPositionsFails(t *testing.T) {
	be := New(clock.NewVirtual(time.Unix(0, 0)), testParams())
	if err := be.Prefill(0, make([]int32, 8), 0); err != nil {
		t.Fatal(err)
	}
	if err := be.Prefill(0, make([]int32, 8), 0); err == nil {
		t.Fatal("second prefill without evict must fail, mirroring decode_rc -1")
	}

	if removed, err := be.Evict(0, 0, -1); err != nil || !removed {
		t.Fatalf("evict: removed=%v err=%v", removed, err)
	}
	if err := be.Prefill(0, make([]int32, 8), 0); err != nil {
		t.Fatalf("prefill after evict: %v", err)
	}
}

func TestPosRangeMatchesBookkeeping(t *testing.T) {
	be := New(clock.NewVirtual(time.Unix(0, 0)), testParams())
	_ = be.Prefill(0, make([]int32, 5), 0)
	for i := 0; i < 3; i++ {
		_, _ = be.Step([]engine.SeqID{0})
	}

	min, max, err := be.PosRange(0)
	if err != nil || min != 0 || max != 7 {
		t.Fatalf("PosRange = [%d,%d] err=%v, want [0,7] (5 prompt + 3 generated)", min, max, err)
	}
}

func TestTokenizeWordCount(t *testing.T) {
	be := New(clock.NewVirtual(time.Unix(0, 0)), testParams())
	toks, _ := be.Tokenize("The capital of France is", true)
	if len(toks) != 5 {
		t.Fatalf("canonical prompt tokenized to %d, want 5 (matches the real tokenizer)", len(toks))
	}
}
