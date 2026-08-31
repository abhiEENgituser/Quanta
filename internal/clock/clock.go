// Package clock abstracts time so that everything above it runs identically
// against wall-clock time and simulated time.
//
// This is the package the lint-clock rule exists to protect: nothing below
// cmd/ may call time.Now, time.Sleep or time.After directly, because one stray
// wall-clock read in the request path silently breaks the simulator — the
// numbers keep flowing, they are just wrong. All time flows through a Clock
// chosen at wiring time.
//
// The Real clock delegates to the runtime. The Virtual clock JUMPS: Sleep(d)
// advances now by exactly d and returns immediately. That jump is what turns a
// 40-hour policy sweep into minutes, and it is exact rather than approximate
// for one reason: in a discrete-event system nothing happens between events,
// so the time skipped over carries zero information. Simulated waiting is not
// a model of waiting — it is the same arithmetic with the idle gaps deleted.
package clock

import (
	"sync"
	"time"
)

// Clock is the minimal surface the serving and simulation code needs: read
// the current instant, and block for a duration. Kept deliberately small —
// every method added here must be implementable by BOTH the real runtime and
// a deterministic simulator, or the two stop being interchangeable.
type Clock interface {
	Now() time.Time
	Sleep(d time.Duration)
}

// Real is the wall clock. The only place in the repo (outside tests) that is
// allowed to touch the time package below cmd/.
type Real struct{}

var _ Clock = Real{}

func (Real) Now() time.Time        { return time.Now() }
func (Real) Sleep(d time.Duration) { time.Sleep(d) }

// Virtual is simulated time: Sleep advances the clock by exactly d and
// returns immediately. Deterministic — two runs with the same inputs see the
// same instants, which is what makes simulated experiments reproducible.
//
// This implementation serves sequential, single-goroutine simulation (the
// Phase 2 validation path). Coordinating many concurrently sleeping
// goroutines needs a waiter queue on top — that arrives with the Phase 3
// simulator, as its own type, when something actually needs it.
type Virtual struct {
	mu  sync.Mutex
	now time.Time
}

var _ Clock = (*Virtual)(nil)

// NewVirtual starts simulated time at the given instant. The epoch is
// arbitrary — simulations care about durations, not dates — but a fixed,
// explicit start keeps runs reproducible and logs legible.
func NewVirtual(start time.Time) *Virtual {
	return &Virtual{now: start}
}

func (v *Virtual) Now() time.Time {
	v.mu.Lock()
	defer v.mu.Unlock()
	return v.now
}

// Sleep advances virtual time by d. It never blocks: the caller "wakes"
// immediately in a world where d has passed.
func (v *Virtual) Sleep(d time.Duration) {
	if d <= 0 {
		return
	}
	v.mu.Lock()
	v.now = v.now.Add(d)
	v.mu.Unlock()
}
