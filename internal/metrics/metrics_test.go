package metrics

import (
	"sync"
	"testing"
	"time"
)

// Deterministic timestamps: metrics never reads a clock, so tests can hand it
// exact instants and assert exact arithmetic. This is the same property that
// makes the package work under Phase 2's virtual clock.
func TestRequestPhaseMath(t *testing.T) {
	reg := NewRegistry()
	base := time.Unix(1000, 0)

	r := NewRequest(reg, base)
	r.MarkAdmitted(base.Add(40 * time.Millisecond))  // 40ms queued
	r.MarkToken(base.Add(340 * time.Millisecond))    // +300ms prefill -> TTFT 340ms
	r.MarkToken(base.Add(357 * time.Millisecond))    // ITL 17ms
	r.MarkToken(base.Add(375 * time.Millisecond))    // ITL 18ms
	r.Finish(base.Add(375 * time.Millisecond))

	snap := reg.Snapshot()

	// HDR guarantees ~0.1% relative error at 3 sigfigs; assert within 1%.
	within := func(name string, got, want time.Duration) {
		t.Helper()
		diff := got - want
		if diff < 0 {
			diff = -diff
		}
		if diff > want/100 {
			t.Errorf("%s: got %v, want ~%v", name, got, want)
		}
	}

	within(HistQueueWait, snap[HistQueueWait].P50, 40*time.Millisecond)
	within(HistPrefill, snap[HistPrefill].P50, 300*time.Millisecond)
	within(HistTTFT, snap[HistTTFT].P50, 340*time.Millisecond)
	within(HistE2E, snap[HistE2E].P50, 375*time.Millisecond)

	if snap[HistITL].Count != 2 {
		t.Errorf("ITL count = %d, want 2 (first token is TTFT, not ITL)", snap[HistITL].Count)
	}
	// TTFT must decompose: queue_wait + prefill = ttft.
	sum := snap[HistQueueWait].P50 + snap[HistPrefill].P50
	within("queue+prefill=ttft", sum, snap[HistTTFT].P50)
}

// The reason percentiles are never averaged, demonstrated rather than asserted:
// two windows, one fast and one slow. Averaging their p99s produces a number
// that is wrong by construction; merging the histograms produces the truth.
func TestMergeVsAveragingPercentiles(t *testing.T) {
	fast, slow := newHistogram(), newHistogram()

	for i := 0; i < 1000; i++ {
		fast.Record(1 * time.Millisecond) // a quiet window
		slow.Record(100 * time.Millisecond) // a loaded window
	}

	avgOfP99s := (fast.Snapshot().P99 + slow.Snapshot().P99) / 2 // ~50ms — fiction

	merged := newHistogram()
	merged.Merge(fast)
	merged.Merge(slow)
	realP99 := merged.Snapshot().P99 // ~100ms — 2000 samples, 1000 of them slow

	if realP99 < 99*time.Millisecond {
		t.Fatalf("merged p99 = %v, expected ~100ms", realP99)
	}
	// The averaged number is off by ~2x. If this assertion ever fails, the
	// demonstration has stopped demonstrating — investigate, don't delete.
	if avgOfP99s > 60*time.Millisecond {
		t.Fatalf("average-of-p99s = %v — expected the fiction to be ~50ms", avgOfP99s)
	}
	t.Logf("real p99 (merged) = %v; average of window p99s = %v — averaging hides half the truth",
		realP99, avgOfP99s)
}

func TestSaturationIsCounted(t *testing.T) {
	hg := newHistogram()
	hg.Record(2 * time.Minute) // past the 60s ceiling
	snap := hg.Snapshot()
	if snap.Saturated != 1 {
		t.Fatalf("saturated = %d, want 1", snap.Saturated)
	}
	if snap.Max < 59*time.Second {
		t.Fatalf("clamped max = %v, want ~60s", snap.Max)
	}
}

// For -race: histograms take concurrent writers (many request goroutines will
// share them), and the registry takes concurrent lookups.
func TestConcurrentRecording(t *testing.T) {
	reg := NewRegistry()
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 1000; i++ {
				reg.Histogram(HistITL).Record(17 * time.Millisecond)
			}
		}()
	}
	wg.Wait()
	if n := reg.Histogram(HistITL).Snapshot().Count; n != 8000 {
		t.Fatalf("count = %d, want 8000", n)
	}
}
