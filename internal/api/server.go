// Package api is quantad's HTTP front door: POST a prompt, receive tokens as
// Server-Sent Events. Phase 1 serves one request at a time — the mutex below
// IS the queue, and time spent blocked on it is recorded as queue wait.
package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/abhiEENgituser/Quanta/internal/engine"
	"github.com/abhiEENgituser/Quanta/internal/metrics"
)

type Config struct {
	// MaxTokens caps generation length per request (client may ask for less).
	MaxTokens int
	// MaxContext is the engine's n_ctx; prompt + generation must fit inside it.
	MaxContext int
}

type Server struct {
	be  engine.Backend
	reg *metrics.Registry
	// now is injected by cmd/quantad. This package never reads the clock
	// directly (lint-clock): with a virtual clock injected instead, the whole
	// API layer runs unmodified under the Phase 2 simulator.
	now func() time.Time
	cfg Config

	// mu serialises engine access. Phase 1 has no scheduler by design: this
	// mutex is the entire admission policy, and how long a request waits on it
	// is exactly what the ttft_queue_wait histogram records.
	mu sync.Mutex
}

func New(be engine.Backend, reg *metrics.Registry, now func() time.Time, cfg Config) *Server {
	return &Server{be: be, reg: reg, now: now, cfg: cfg}
}

func (s *Server) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/generate", s.handleGenerate)
	mux.HandleFunc("GET /v1/metrics", s.handleMetrics)
	return mux
}

type generateRequest struct {
	Prompt    string `json:"prompt"`
	MaxTokens int    `json:"max_tokens"` // 0 = server default
}

// sse writes one Server-Sent Event and flushes it to the wire immediately.
// Without the flush, Go buffers ~4KB of response — hundreds of tokens — and
// the client sees bursts at buffer boundaries instead of a stream. That would
// not just look wrong: it would make the ITL histogram measure buffer
// mechanics rather than generation.
func sse(w http.ResponseWriter, f http.Flusher, event string, payload any) {
	data, _ := json.Marshal(payload)
	if event != "" {
		fmt.Fprintf(w, "event: %s\n", event)
	}
	fmt.Fprintf(w, "data: %s\n\n", data)
	f.Flush()
}

func (s *Server) handleGenerate(w http.ResponseWriter, r *http.Request) {
	arrival := s.now() // before any queueing — that is the definition of arrival

	var req generateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad request body: "+err.Error(), http.StatusBadRequest)
		return
	}
	if req.Prompt == "" {
		http.Error(w, "prompt must not be empty", http.StatusBadRequest)
		return
	}
	maxTokens := req.MaxTokens
	if maxTokens <= 0 || maxTokens > s.cfg.MaxTokens {
		maxTokens = s.cfg.MaxTokens
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}

	mreq := metrics.NewRequest(s.reg, arrival)

	// ---- the queue. Everything above ran concurrently; from here on the
	// ---- request owns the engine.
	s.mu.Lock()
	defer s.mu.Unlock()
	mreq.MarkAdmitted(s.now()) // queue wait ends here; engine share begins

	toks, err := s.be.Tokenize(req.Prompt, true)
	if err != nil {
		http.Error(w, "tokenize: "+err.Error(), http.StatusBadGateway)
		return
	}
	if len(toks)+maxTokens > s.cfg.MaxContext {
		http.Error(w,
			fmt.Sprintf("prompt (%d tokens) + max_tokens (%d) exceeds context (%d)",
				len(toks), maxTokens, s.cfg.MaxContext),
			http.StatusBadRequest)
		return
	}

	// Phase 1 reuses sequence 0 for every request; engine state outlives
	// connections, so clear it first — a stale sequence makes prefill fail
	// with decode_rc -1 (occupied positions), or worse, pollutes generation.
	const seq engine.SeqID = 0
	if _, err := s.be.Evict(seq, 0, -1); err != nil {
		http.Error(w, "evict: "+err.Error(), http.StatusBadGateway)
		return
	}
	if err := s.be.Prefill(seq, toks, 0); err != nil {
		http.Error(w, "prefill: "+err.Error(), http.StatusBadGateway)
		return
	}

	// Headers commit the response as a stream; HTTP errors are impossible past
	// this point, so failures below travel as an "error" event instead.
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(http.StatusOK)

	var asm utf8Assembler
	ctx := r.Context()
	finish := "length"
	produced := 0

	for produced < maxTokens {
		// A disconnected client must stop engine work, not leave it generating
		// into the void. This is the cancellation path context exists for.
		select {
		case <-ctx.Done():
			finish = "cancelled"
		default:
		}
		if finish == "cancelled" {
			break
		}

		res, err := s.be.Step([]engine.SeqID{seq})
		if err != nil {
			sse(w, flusher, "error", map[string]string{"error": err.Error()})
			finish = "error"
			break
		}
		tok := res.Tokens[0]
		mreq.MarkToken(s.now())
		produced++

		if text := asm.Push(tok.Piece); text != "" {
			sse(w, flusher, "", map[string]string{"text": text})
		}
		if tok.Finished {
			finish = "stop"
			break
		}
	}

	if tail := asm.Flush(); tail != "" {
		sse(w, flusher, "", map[string]string{"text": tail})
	}

	// Free the KV cache regardless of how the stream ended — under a clamped
	// budget, a leaked sequence is leaked memory.
	if _, err := s.be.Evict(seq, 0, -1); err != nil && finish != "error" {
		sse(w, flusher, "error", map[string]string{"error": "evict: " + err.Error()})
	}

	mreq.Finish(s.now())
	sse(w, flusher, "done", map[string]any{
		"tokens":        produced,
		"prompt_tokens": len(toks),
		"finish_reason": finish,
	})
}

func (s *Server) handleMetrics(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	// Quantiles marshal with time.Duration fields as integer nanoseconds —
	// exact, and trivial for the bench harness to parse.
	_ = json.NewEncoder(w).Encode(s.reg.Snapshot())
}
