package clock

import (
	"testing"
	"time"
)

// The property the whole simulation premise rests on: sleeping simulated time
// costs no real time, and advances simulated time exactly.
func TestVirtualSleepJumpsExactly(t *testing.T) {
	start := time.Unix(0, 0)
	v := NewVirtual(start)

	realStart := time.Now()
	v.Sleep(40 * time.Hour) // a full Phase-1-sized sweep, in one call
	realElapsed := time.Since(realStart)

	if got := v.Now().Sub(start); got != 40*time.Hour {
		t.Fatalf("virtual time advanced %v, want exactly 40h", got)
	}
	if realElapsed > 100*time.Millisecond {
		t.Fatalf("virtual sleep took %v of real time — it must not block", realElapsed)
	}
}

func TestVirtualIsDeterministic(t *testing.T) {
	run := func() time.Time {
		v := NewVirtual(time.Unix(1000, 0))
		for i := 0; i < 1000; i++ {
			v.Sleep(17 * time.Millisecond) // a simulated decode step
		}
		return v.Now()
	}
	if a, b := run(), run(); !a.Equal(b) {
		t.Fatalf("two identical runs ended at different instants: %v vs %v", a, b)
	}
}

func TestVirtualNonPositiveSleepIsNoop(t *testing.T) {
	v := NewVirtual(time.Unix(0, 0))
	v.Sleep(0)
	v.Sleep(-time.Second)
	if !v.Now().Equal(time.Unix(0, 0)) {
		t.Fatalf("non-positive sleeps moved the clock to %v", v.Now())
	}
}

// Sanity only: Real delegates to the runtime. Kept tiny so the test suite
// spends its time on logic, not on actually sleeping.
func TestRealClockAdvances(t *testing.T) {
	var c Clock = Real{}
	a := c.Now()
	c.Sleep(10 * time.Millisecond)
	if b := c.Now(); !b.After(a) {
		t.Fatalf("real clock did not advance: %v -> %v", a, b)
	}
}
