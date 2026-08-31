// Command quantad is the Quanta serving daemon. Phase 1: single-stream — one
// request owns the engine at a time, no scheduler. This is the baseline every
// scheduling policy gets measured against.
//
// Wiring happens here and only here: the API layer sees engine.Backend, never
// the shim client — swapping in the Phase 2 cost-model backend is a change to
// this file alone. Likewise time.Now is injected from here because nothing
// below cmd/ may read the clock (see lint-clock).
package main

import (
	"flag"
	"log"
	"net/http"
	"time"

	"github.com/abhiEENgituser/Quanta/internal/api"
	"github.com/abhiEENgituser/Quanta/internal/clock"
	"github.com/abhiEENgituser/Quanta/internal/engine"
	"github.com/abhiEENgituser/Quanta/internal/engine/costmodel"
	"github.com/abhiEENgituser/Quanta/internal/engine/shim"
	"github.com/abhiEENgituser/Quanta/internal/engine/synthetic"
	"github.com/abhiEENgituser/Quanta/internal/metrics"
)

func main() {
	var (
		engineKind = flag.String("engine", "shim", "engine backend: shim | mock")
		socket     = flag.String("socket", "/tmp/quanta.sock", "shim unix socket path (engine=shim)")
		paramsPath = flag.String("params", "internal/engine/costmodel/params.json", "cost model (engine=mock)")
		listen     = flag.String("listen", "127.0.0.1:8080", "HTTP listen address")
		maxTokens  = flag.Int("max-tokens", 256, "per-request generation cap")
		maxContext = flag.Int("max-context", 2048, "engine n_ctx; prompt+generation must fit")
	)
	flag.Parse()

	// The wiring point — and the ONLY difference between serving with a real
	// model and serving with a calibrated sleep(). Everything downstream of
	// engine.Backend cannot tell which one it got; that indistinguishability
	// is the proof this is a scheduler, not an AI program.
	var be engine.Backend
	switch *engineKind {
	case "shim":
		var err error
		be, err = shim.Dial(*socket)
		if err != nil {
			log.Fatalf("quantad: %v (is the shim running?)", err)
		}
	case "mock":
		p, err := costmodel.Load(*paramsPath)
		if err != nil {
			log.Fatalf("quantad: %v (run make calibrate)", err)
		}
		be = synthetic.New(clock.Real{}, p)
		log.Printf("quantad: MOCK engine — calibrated sleep(), no model loaded (fitted %s)", p.Meta.FittedAt)
	default:
		log.Fatalf("quantad: unknown -engine %q", *engineKind)
	}
	defer be.Close()

	reg := metrics.NewRegistry()
	srv := api.New(be, reg, time.Now, api.Config{
		MaxTokens:  *maxTokens,
		MaxContext: *maxContext,
	})

	log.Printf("quantad: engine=%s, serving http://%s", *engineKind, *listen)
	log.Fatal(http.ListenAndServe(*listen, srv.Routes()))
}
