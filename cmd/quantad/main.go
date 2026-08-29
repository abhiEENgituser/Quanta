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
	"github.com/abhiEENgituser/Quanta/internal/engine/shim"
	"github.com/abhiEENgituser/Quanta/internal/metrics"
)

func main() {
	var (
		socket     = flag.String("socket", "/tmp/quanta.sock", "shim unix socket path")
		listen     = flag.String("listen", "127.0.0.1:8080", "HTTP listen address")
		maxTokens  = flag.Int("max-tokens", 256, "per-request generation cap")
		maxContext = flag.Int("max-context", 2048, "engine n_ctx; prompt+generation must fit")
	)
	flag.Parse()

	be, err := shim.Dial(*socket)
	if err != nil {
		log.Fatalf("quantad: %v (is the shim running?)", err)
	}
	defer be.Close()

	reg := metrics.NewRegistry()
	srv := api.New(be, reg, time.Now, api.Config{
		MaxTokens:  *maxTokens,
		MaxContext: *maxContext,
	})

	log.Printf("quantad: engine on %s, serving http://%s", *socket, *listen)
	log.Fatal(http.ListenAndServe(*listen, srv.Routes()))
}
