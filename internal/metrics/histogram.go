// Package metrics aggregates latency observations into HDR histograms.
//
// Two rules define this package:
//
//  1. It never reads the clock. Callers pass timestamps in; this package only
//     does arithmetic on them. That keeps every consumer compatible with the
//     Phase 2 virtual clock without modification, and it is what the
//     lint-clock check enforces repo-wide.
//
//  2. Histograms merge; percentiles never do. The p99 of two windows averaged
//     is not the p99 of their union — it is a number with no meaning. So raw
//     histograms are the unit of aggregation, and quantiles are computed only
//     at read time, from a single (possibly merged) histogram.
package metrics

import (
	"sync"
	"time"

	hdrhistogram "github.com/HdrHistogram/hdrhistogram-go"
)

// Histogram bounds per the measurement plan: microsecond units, 100 µs floor,
// 60 s ceiling, 3 significant figures (~0.1% relative error on any quantile).
const (
	minUS   = 100
	maxUS   = 60_000_000
	sigfigs = 3
)

// Histogram is a concurrency-safe HDR histogram of durations in microseconds.
// The underlying library is not goroutine-safe; the mutex is not optional.
type Histogram struct {
	mu        sync.Mutex
	h         *hdrhistogram.Histogram
	saturated int64 // observations clamped to maxUS — a nonzero value in a
	// snapshot means the 60 s ceiling was hit and tails above it are invisible.
}

func newHistogram() *Histogram {
	return &Histogram{h: hdrhistogram.New(minUS, maxUS, sigfigs)}
}

// Record adds one duration. Values below the floor count as the floor (they
// carry no information at our resolution); values above the ceiling are
// clamped and counted in saturated rather than silently dropped.
func (hg *Histogram) Record(d time.Duration) {
	us := d.Microseconds()

	hg.mu.Lock()
	defer hg.mu.Unlock()

	switch {
	case us < minUS:
		us = minUS
	case us > maxUS:
		us = maxUS
		hg.saturated++
	}
	// After clamping, RecordValue cannot fail.
	_ = hg.h.RecordValue(us)
}

// Merge folds other into hg. This is the ONLY correct way to combine two
// measurement windows — merge the histograms, then read quantiles from the
// result.
func (hg *Histogram) Merge(other *Histogram) {
	other.mu.Lock()
	snap := hdrhistogram.Import(other.h.Export())
	sat := other.saturated
	other.mu.Unlock()

	hg.mu.Lock()
	defer hg.mu.Unlock()
	hg.h.Merge(snap)
	hg.saturated += sat
}

// Quantiles is a point-in-time read of one histogram.
type Quantiles struct {
	Count     int64
	P50       time.Duration
	P95       time.Duration
	P99       time.Duration
	P999      time.Duration
	Max       time.Duration
	Saturated int64
}

func (hg *Histogram) Snapshot() Quantiles {
	hg.mu.Lock()
	defer hg.mu.Unlock()

	us := func(v int64) time.Duration { return time.Duration(v) * time.Microsecond }
	return Quantiles{
		Count:     hg.h.TotalCount(),
		P50:       us(hg.h.ValueAtQuantile(50)),
		P95:       us(hg.h.ValueAtQuantile(95)),
		P99:       us(hg.h.ValueAtQuantile(99)),
		P999:      us(hg.h.ValueAtQuantile(99.9)),
		Max:       us(hg.h.Max()),
		Saturated: hg.saturated,
	}
}

// Registry is a named set of histograms, created on first use.
type Registry struct {
	mu    sync.Mutex
	hists map[string]*Histogram
}

func NewRegistry() *Registry {
	return &Registry{hists: make(map[string]*Histogram)}
}

// Histogram returns the named histogram, creating it if needed.
func (r *Registry) Histogram(name string) *Histogram {
	r.mu.Lock()
	defer r.mu.Unlock()

	hg, ok := r.hists[name]
	if !ok {
		hg = newHistogram()
		r.hists[name] = hg
	}
	return hg
}

// Snapshot reads every histogram.
func (r *Registry) Snapshot() map[string]Quantiles {
	// Copy the name->histogram pairs under the registry lock, then snapshot
	// each histogram outside it (each takes its own lock). Keeping the pairs
	// together end to end — no parallel slices — is the point: an earlier
	// version collected names and histograms separately, sorted one and not
	// the other, and served every value under the wrong name.
	r.mu.Lock()
	pairs := make(map[string]*Histogram, len(r.hists))
	for n, hg := range r.hists {
		pairs[n] = hg
	}
	r.mu.Unlock()

	out := make(map[string]Quantiles, len(pairs))
	for n, hg := range pairs {
		out[n] = hg.Snapshot()
	}
	return out
}
