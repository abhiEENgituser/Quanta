package metrics

import "time"

// Histogram names recorded by Request. Split TTFT is the point of this file:
// when TTFT regresses, one aggregate number cannot say whether the scheduler
// queued badly (queue_wait) or prefill got heavier (prefill). Days get lost
// guessing; two histograms answer it immediately.
const (
	HistTTFT      = "ttft"            // arrival -> first token
	HistQueueWait = "ttft_queue_wait" // arrival -> handed to engine (scheduler's share)
	HistPrefill   = "ttft_prefill"    // handed to engine -> first token (engine's share)
	HistITL       = "itl"             // between consecutive tokens
	HistE2E       = "e2e"             // arrival -> done
)

// Request tracks one request's phase timestamps and feeds the registry.
// It never reads a clock: every method takes the instant as an argument,
// captured by whoever owns time (real now, virtual in the simulator).
//
// Expected call order: NewRequest -> MarkAdmitted -> MarkToken... -> Finish.
// Not goroutine-safe; one request's lifecycle belongs to one goroutine.
type Request struct {
	reg       *Registry
	arrival   time.Time
	admitted  time.Time
	lastToken time.Time
	tokens    int
}

// NewRequest starts tracking at the moment the request arrived — before any
// queueing, which is exactly why arrival is captured here and not at admit.
func NewRequest(reg *Registry, arrival time.Time) *Request {
	return &Request{reg: reg, arrival: arrival}
}

// MarkAdmitted records the instant the request was handed to the engine.
// Everything between arrival and here is queue wait: the scheduler's share.
func (r *Request) MarkAdmitted(t time.Time) {
	r.admitted = t
	r.reg.Histogram(HistQueueWait).Record(t.Sub(r.arrival))
}

// MarkToken records one generated token. The first token closes out TTFT and
// its prefill component; every later token contributes an ITL sample.
func (r *Request) MarkToken(t time.Time) {
	if r.tokens == 0 {
		r.reg.Histogram(HistTTFT).Record(t.Sub(r.arrival))
		if !r.admitted.IsZero() {
			r.reg.Histogram(HistPrefill).Record(t.Sub(r.admitted))
		}
	} else {
		r.reg.Histogram(HistITL).Record(t.Sub(r.lastToken))
	}
	r.lastToken = t
	r.tokens++
}

// Finish closes the request at t, recording end-to-end latency.
func (r *Request) Finish(t time.Time) {
	r.reg.Histogram(HistE2E).Record(t.Sub(r.arrival))
}

// Tokens reports how many tokens were recorded — the caller needs it for
// throughput accounting, and tests use it.
func (r *Request) Tokens() int { return r.tokens }
