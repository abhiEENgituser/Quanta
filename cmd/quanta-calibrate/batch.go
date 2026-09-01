package main

import (
	"fmt"
	"log"
	"math/rand/v2"
	"os"
	"sort"
	"time"

	"github.com/abhiEENgituser/Quanta/internal/engine"
	"github.com/abhiEENgituser/Quanta/internal/engine/costmodel"
	"github.com/abhiEENgituser/Quanta/internal/engine/shim"
)

// runBatchMode measures the cost of one llama_decode advancing B sequences at
// once, for B = 1..batchMax, at a fixed per-sequence context (batchPrompt).
//
// This is the measurement the project's premise rests on: the ~380MB weight
// read happens once per decode call regardless of B, so batched cost should be
// strongly sublinear in B. Quantifying exactly how sublinear IS the economic
// case for batching — which is why it is measured, not assumed.
//
// Context is held fixed on purpose: step cost also grows with ctx (the fitted
// Step line covers that), and letting both vary at once would confound the two
// effects — the same mistake the v1 length-order/heat confound taught us to
// design against. Same discipline otherwise: thermal warmup, shuffled B order
// per sweep, per-B medians.
func runBatchMode(socket, out string, batchMax, batchPrompt, steps, sweeps int, warmFor time.Duration) {
	// The batch line joins the existing params file; the fitted lines must
	// already be there.
	p, err := costmodel.Load(out)
	if err != nil {
		log.Fatalf("batch mode needs the lines calibrated first: %v", err)
	}

	be, err := shim.Dial(socket)
	if err != nil {
		log.Fatalf("calibrate: %v (start the shim via bench/configs/calibrate.sh)", err)
	}
	defer be.Close()

	base, err := be.Tokenize("The capital of France is", true)
	if err != nil {
		log.Fatal(err)
	}
	prompt := make([]int32, batchPrompt)
	for i := range prompt {
		prompt[i] = base[i%len(base)]
	}

	// Prefill sequences 0..b-1 and run `steps` batched decode calls, timing
	// each call. Returns per-call durations in microseconds.
	round := func(b, nSteps int, record bool) []float64 {
		active := make([]engine.SeqID, b)
		for i := 0; i < b; i++ {
			seq := engine.SeqID(i)
			active[i] = seq
			if _, err := be.Evict(seq, 0, -1); err != nil {
				log.Fatal(err)
			}
			if err := be.Prefill(seq, prompt, 0); err != nil {
				log.Fatalf("prefill seq %d (is the shim running with -q %d+?): %v", i, batchMax, err)
			}
		}
		var out []float64
		for i := 0; i < nSteps; i++ {
			t0 := time.Now()
			if _, err := be.Step(active); err != nil {
				log.Fatalf("step batch=%d: %v", b, err)
			}
			if record {
				out = append(out, float64(time.Since(t0).Microseconds()))
			}
		}
		return out
	}

	fmt.Fprintf(os.Stderr, "thermal warmup: %s at batch=%d...\n", warmFor, batchMax)
	for end := time.Now().Add(warmFor); time.Now().Before(end); {
		round(batchMax, 8, false)
	}

	byB := map[int][]float64{}
	rng := rand.New(rand.NewPCG(43, 0))
	for sweep := 1; sweep <= sweeps; sweep++ {
		for _, idx := range rng.Perm(batchMax) {
			b := idx + 1
			byB[b] = append(byB[b], round(b, steps, true)...)
			fmt.Fprintf(os.Stderr, "sweep %d batch %d done\n", sweep, b)
		}
	}

	xs, ys := medianPoints(byB)

	line, err := costmodel.Fit(xs, ys)
	if err != nil {
		log.Fatal(err)
	}
	p.StepBatch = line
	p.BatchRefCtx = batchPrompt
	p.Meta.Notes += fmt.Sprintf(" | batch sweep: B=1..%d at ctx=%d, %d sweeps, %s",
		batchMax, batchPrompt, sweeps, time.Now().Format(time.RFC3339))

	if err := p.Save(out); err != nil {
		log.Fatal(err)
	}

	solo := medianOf(byB[1])
	fmt.Printf("step_batch(B) = %.0fus + %.0fus·B   R²=%.4f  (at ctx≈%d)\n",
		line.InterceptUS, line.SlopeUS, line.R2, batchPrompt)
	fmt.Printf("\n%-6s %12s %14s %18s\n", "batch", "cost/call", "cost/sequence", "vs B×solo")
	for _, b := range []int{1, 2, 3, 4, 5, 6} {
		if b > batchMax {
			break
		}
		m := medianOf(byB[b])
		fmt.Printf("%-6d %10.1fms %12.1fms %16.1f%%\n",
			b, m/1000, m/float64(b)/1000, m/(float64(b)*solo)*100)
	}
	fmt.Printf("\nsublinearity: batch %d costs %.0f%% of %d separate calls\n",
		batchMax, medianOf(byB[batchMax])/(float64(batchMax)*solo)*100, batchMax)
	fmt.Printf("wrote %s\n", out)
}

func medianOf(vs []float64) float64 {
	if len(vs) == 0 {
		return 0
	}
	xs := append([]float64(nil), vs...)
	sort.Float64s(xs)
	return xs[len(xs)/2]
}
