// Package sched is the scheduler: the first component in this system that has
// opinions. Everything below it (shim, synthetic, metrics, clock) was built
// deliberately opinion-free so that the decisions made here — who is admitted,
// who advances, who is evicted — can be measured, simulated, and blamed with
// precision.
//
// The scheduler is a TICKABLE LIBRARY, not a running loop. Tick() executes one
// iteration-level scheduling cycle (Orca's idea): reap finished sequences,
// admit per policy, advance the active set by exactly one decode pass. Whoever
// owns the process drives Tick — a goroutine in real serving, a plain loop in
// the simulator, a test by hand. Single-threaded by construction: no mutexes,
// no goroutines, no time.Now — every instant comes from the injected Clock,
// every engine call goes through the injected Backend. That is what makes
// scheduling decisions deterministic, and determinism is what makes them
// testable to exact virtual timestamps.
package sched

import (
	"errors"
	"fmt"

	"github.com/abhiEENgituser/Quanta/internal/clock"
	"github.com/abhiEENgituser/Quanta/internal/engine"
	"github.com/abhiEENgituser/Quanta/internal/metrics"
)

// Policy decides when queued requests join the active set.
type Policy int

const (
	// Static batching: admit a full batch only when the engine is EMPTY, then
	// run that batch to completion before admitting anyone else. Great ITL
	// (the batch never changes mid-flight), terrible slot utilization
	// (finished-early slots idle while stragglers run). The honest baseline.
	Static Policy = iota
	// Continuous batching: a slot refills the moment it frees. Better
	// utilization and queue wait; the price is a prefill injected into the
	// running batch on every admission, stalling current decoders for its
	// duration — the ITL spike chunked prefill will later address.
	Continuous
)

func (p Policy) String() string {
	if p == Static {
		return "static"
	}
	return "continuous"
}

// Request is one unit of work handed to the scheduler. Tokens are already
// tokenized: admission decisions need the token count before any KV memory is
// committed, so tokenization happens at the edge (the API layer), not here.
type Request struct {
	ID        string
	Tokens    []int32
	MaxTokens int

	// OnToken and OnDone are called synchronously inside Tick, from the
	// driver's goroutine. They must be fast and non-blocking (hand off to a
	// channel or buffer); a slow callback stalls the entire batch — every
	// sequence, not just this one.
	OnToken func(tok engine.Token)
	OnDone  func(reason string) // "stop" | "length" | "cancelled" | "error"

	// Cancelled is polled each tick before work is spent on this request.
	// Nil means never cancelled.
	Cancelled func() bool

	// M, when non-nil, receives lifecycle marks (admitted, tokens, finish)
	// with instants from the scheduler's clock.
	M *metrics.Request
}

type Config struct {
	// Slots is the maximum concurrent sequences — the engine's -q value.
	// Remember the capacity semantics: llama.cpp divides n_ctx among slots,
	// so Slots is not free parallelism, it is a partition of the KV budget.
	Slots  int
	Policy Policy
}

type active struct {
	req      *Request
	seq      engine.SeqID
	produced int
}

// Scheduler owns the engine. Nothing else may touch the Backend while a
// Scheduler is driving it — the serial engine is exactly one party's to
// command, and that party is Tick.
type Scheduler struct {
	be  engine.Backend
	clk clock.Clock
	cfg Config

	queue    []*Request
	activeBy map[engine.SeqID]*active
	order    []engine.SeqID // active set in admission order, for stable stepping
	freeSeqs []engine.SeqID // LIFO pool of slot ids, 0..Slots-1
}

func New(be engine.Backend, clk clock.Clock, cfg Config) (*Scheduler, error) {
	if cfg.Slots < 1 {
		return nil, errors.New("sched: Slots must be >= 1")
	}
	s := &Scheduler{
		be:       be,
		clk:      clk,
		cfg:      cfg,
		activeBy: make(map[engine.SeqID]*active),
	}
	for i := cfg.Slots - 1; i >= 0; i-- { // pop order 0,1,2... for legibility
		s.freeSeqs = append(s.freeSeqs, engine.SeqID(i))
	}
	return s, nil
}

// Submit enqueues a request. It never blocks and never touches the engine —
// admission is Tick's decision, made under the policy, not the caller's.
func (s *Scheduler) Submit(r *Request) error {
	if len(r.Tokens) == 0 {
		return errors.New("sched: empty prompt")
	}
	if r.MaxTokens < 1 {
		return errors.New("sched: MaxTokens must be >= 1")
	}
	s.queue = append(s.queue, r)
	return nil
}

// QueueDepth and ActiveCount exist for observability and admission control
// above the scheduler (Phase 5 sheds load based on these).
func (s *Scheduler) QueueDepth() int  { return len(s.queue) }
func (s *Scheduler) ActiveCount() int { return len(s.activeBy) }

// Tick runs one scheduling cycle. Returns true if any work was done — a false
// return means idle (empty queue and empty engine), letting the driver decide
// whether to block, sleep, or exit.
//
// Order within a tick: cancellations first (never spend a step on a corpse),
// then admission (policy), then one decode pass, then completions. One Step
// call per tick, exactly — the tick IS the iteration in iteration-level
// scheduling.
func (s *Scheduler) Tick() (bool, error) {
	s.reapCancelled()
	s.admit()

	if len(s.order) == 0 {
		return len(s.queue) > 0, nil // queued work exists but cannot admit (shouldn't happen) or truly idle
	}

	res, err := s.be.Step(s.order)
	if err != nil {
		// A failed step leaves engine state unknowable for everyone in the
		// batch. Phase 4 refines this (decode_rc 1 = memory pressure becomes
		// a preemption signal); for now every active request fails loudly,
		// and the error propagates to the driver.
		s.failAll(err)
		return true, fmt.Errorf("sched: step: %w", err)
	}

	now := s.clk.Now()
	for _, tok := range res.Tokens {
		a, ok := s.activeBy[tok.Seq]
		if !ok {
			continue // engine returned a token for a sequence we dropped this tick
		}
		a.produced++
		if a.req.M != nil {
			a.req.M.MarkToken(now)
		}
		if a.req.OnToken != nil {
			a.req.OnToken(tok)
		}

		switch {
		case tok.Finished:
			s.finish(a, "stop")
		case a.produced >= a.req.MaxTokens:
			s.finish(a, "length")
		}
	}
	return true, nil
}

// admit moves queued requests into engine slots per the policy.
func (s *Scheduler) admit() {
	switch s.cfg.Policy {
	case Static:
		// Only when the engine is completely empty does the next batch load.
		if len(s.activeBy) > 0 {
			return
		}
		for len(s.queue) > 0 && len(s.freeSeqs) > 0 {
			s.admitOne()
		}
	case Continuous:
		// Any free slot is filled immediately. The prefill this triggers
		// stalls the running batch for its duration — that is the measured
		// cost of this policy, not an accident.
		for len(s.queue) > 0 && len(s.freeSeqs) > 0 {
			s.admitOne()
		}
	}
}

func (s *Scheduler) admitOne() {
	r := s.queue[0]
	s.queue = s.queue[1:]

	if r.Cancelled != nil && r.Cancelled() {
		done(r, "cancelled")
		return
	}

	seq := s.freeSeqs[len(s.freeSeqs)-1]
	s.freeSeqs = s.freeSeqs[:len(s.freeSeqs)-1]

	if r.M != nil {
		r.M.MarkAdmitted(s.clk.Now())
	}

	// Defensive evict: sequence slots are reused, and engine state outlives
	// everything (a lesson with a decode_rc -1 scar). An empty evict is cheap.
	if _, err := s.be.Evict(seq, 0, -1); err != nil {
		s.freeSeqs = append(s.freeSeqs, seq)
		done(r, "error")
		return
	}
	if err := s.be.Prefill(seq, r.Tokens, 0); err != nil {
		s.freeSeqs = append(s.freeSeqs, seq)
		done(r, "error")
		return
	}

	a := &active{req: r, seq: seq}
	s.activeBy[seq] = a
	s.order = append(s.order, seq)
}

func (s *Scheduler) reapCancelled() {
	// Queued cancellations: drop before they cost a slot.
	kept := s.queue[:0]
	for _, r := range s.queue {
		if r.Cancelled != nil && r.Cancelled() {
			done(r, "cancelled")
			continue
		}
		kept = append(kept, r)
	}
	s.queue = kept

	// Active cancellations: free the slot and its KV before stepping.
	for _, seq := range append([]engine.SeqID(nil), s.order...) {
		a := s.activeBy[seq]
		if a.req.Cancelled != nil && a.req.Cancelled() {
			s.finish(a, "cancelled")
		}
	}
}

// finish releases a sequence: evict its KV, return the slot, notify.
func (s *Scheduler) finish(a *active, reason string) {
	_, _ = s.be.Evict(a.seq, 0, -1) // a leaked slot is leaked KV
	delete(s.activeBy, a.seq)
	for i, seq := range s.order {
		if seq == a.seq {
			s.order = append(s.order[:i], s.order[i+1:]...)
			break
		}
	}
	s.freeSeqs = append(s.freeSeqs, a.seq)
	if a.req.M != nil {
		a.req.M.Finish(s.clk.Now())
	}
	done(a.req, reason)
}

func (s *Scheduler) failAll(err error) {
	for _, seq := range append([]engine.SeqID(nil), s.order...) {
		s.finish(s.activeBy[seq], "error")
	}
	_ = err
}

func done(r *Request, reason string) {
	if r.OnDone != nil {
		r.OnDone(reason)
	}
}
