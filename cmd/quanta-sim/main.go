// Command quanta-sim runs workloads against the calibrated cost model on a
// VIRTUAL clock: simulated time jumps from event to event, so hours of
// simulated inference finish in wall-clock seconds — exactly, not
// approximately, because in a discrete-event system nothing happens between
// events and the skipped waiting carries no information.
//
// Phase 2 scope: sequential single-stream replay, which is exactly what the
// validated cost model covers. Concurrent arrivals and batching arrive with
// Phase 3, alongside the batch-size calibration they require.
package main

import (
	"flag"
	"fmt"
	"log"
	"time"

	"github.com/abhiEENgituser/Quanta/internal/clock"
	"github.com/abhiEENgituser/Quanta/internal/engine"
	"github.com/abhiEENgituser/Quanta/internal/engine/costmodel"
	"github.com/abhiEENgituser/Quanta/internal/engine/synthetic"
	"github.com/abhiEENgituser/Quanta/internal/metrics"
)

func main() {
	var (
		paramsPath = flag.String("params", "internal/engine/costmodel/params.json", "fitted cost model")
		requests   = flag.Int("requests", 500, "number of requests to simulate")
		promptTok  = flag.Int("prompt-tokens", 128, "prompt length per request")
		maxTok     = flag.Int("max-tokens", 128, "generated tokens per request")
	)
	flag.Parse()

	p, err := costmodel.Load(*paramsPath)
	if err != nil {
		log.Fatal(err)
	}

	vclk := clock.NewVirtual(time.Unix(0, 0))
	be := synthetic.New(vclk, p)
	reg := metrics.NewRegistry()

	prompt := make([]int32, *promptTok)

	wallStart := time.Now()
	simStart := vclk.Now()

	for r := 0; r < *requests; r++ {
		mreq := metrics.NewRequest(reg, vclk.Now())

		if _, err := be.Evict(0, 0, -1); err != nil {
			log.Fatal(err)
		}
		mreq.MarkAdmitted(vclk.Now()) // single-stream: admitted immediately

		if err := be.Prefill(0, prompt, 0); err != nil {
			log.Fatal(err)
		}
		for i := 0; i < *maxTok; i++ {
			if _, err := be.Step([]engine.SeqID{0}); err != nil {
				log.Fatal(err)
			}
			mreq.MarkToken(vclk.Now())
		}
		mreq.Finish(vclk.Now())
	}

	simElapsed := vclk.Now().Sub(simStart)
	wallElapsed := time.Since(wallStart)

	snap := reg.Snapshot()
	fmt.Printf("simulated: %d requests × (%d prompt + %d generated tokens)\n",
		*requests, *promptTok, *maxTok)
	fmt.Printf("simulated time:  %v\n", simElapsed.Round(time.Second))
	fmt.Printf("wall time:       %v   (speedup ~%.0fx)\n",
		wallElapsed.Round(time.Millisecond),
		float64(simElapsed)/float64(wallElapsed))
	fmt.Printf("ttft  p50=%v p99=%v\n", snap[metrics.HistTTFT].P50, snap[metrics.HistTTFT].P99)
	fmt.Printf("itl   p50=%v p99=%v  (n=%d)\n",
		snap[metrics.HistITL].P50, snap[metrics.HistITL].P99, snap[metrics.HistITL].Count)
	fmt.Printf("e2e   p50=%v p99=%v\n", snap[metrics.HistE2E].P50, snap[metrics.HistE2E].P99)
	fmt.Printf("throughput (simulated): %.1f tokens/s\n",
		float64(*requests**maxTok)/simElapsed.Seconds())
}
