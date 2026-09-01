// Command quanta-calibrate fits the cost model by measuring the real engine
// through the same Backend client quantad uses — socket round-trip included,
// because the synthetic backend must impersonate the engine as the scheduler
// sees it, not as a bare-metal probe sees it.
//
// Calibration happens at THERMAL STEADY STATE. A serving engine is
// continuously busy, so hot equilibrium is the honest operating point — and
// the v1 calibrator proved the alternative: measuring on a machine that heats
// as the sweep proceeds produced a 21% drift across sweeps, and because
// lengths ran in ascending order, heat correlated with context length and
// poisoned the step slope itself (R² 0.44, confounded). Hence:
//
//   - a sustained warmup drives the engine to equilibrium before recording
//   - length order is shuffled per sweep (seeded), breaking the ctx↔time
//     correlation so any residual drift lands as noise, not slope
//   - CPU MHz is sampled during recording and stored in the params meta,
//     with a warning if it declines — conditions ride with the parameters
//   - fits use per-x medians across sweeps, damping sweep-level wobble
//
// The shim must already be running (bench/configs/calibrate.sh starts it
// pinned and configured; conditions are part of the measurement).
package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand/v2"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/abhiEENgituser/Quanta/internal/engine"
	"github.com/abhiEENgituser/Quanta/internal/engine/costmodel"
	"github.com/abhiEENgituser/Quanta/internal/engine/shim"
)

func main() {
	var (
		socket  = flag.String("socket", "/tmp/quanta.sock", "shim unix socket")
		mode    = flag.String("mode", "lines", "lines: prefill/step vs length | batch: step vs batch size")
		lengths = flag.String("lengths", "16,32,64,128,256,512", "prompt token lengths (mode=lines)")
		steps   = flag.Int("steps", 24, "decode steps measured per length")
		sweeps  = flag.Int("sweeps", 3, "recorded sweeps (after thermal warmup)")
		warmFor = flag.Duration("thermal-warmup", 60*time.Second, "sustained load before recording")
		out     = flag.String("out", "internal/engine/costmodel/params.json", "output path")
		args    = flag.String("engine-args", "", "engine config recorded in meta")

		batchMax    = flag.Int("batch-max", 6, "largest batch size (mode=batch; shim needs -q >= this)")
		batchPrompt = flag.Int("batch-prompt", 128, "prompt tokens per sequence (mode=batch)")
	)
	flag.Parse()

	if *mode == "batch" {
		runBatchMode(*socket, *out, *batchMax, *batchPrompt, *steps, *sweeps, *warmFor)
		return
	}

	var lens []int
	for _, s := range strings.Split(*lengths, ",") {
		n, err := strconv.Atoi(strings.TrimSpace(s))
		if err != nil || n < 1 {
			log.Fatalf("bad length %q", s)
		}
		lens = append(lens, n)
	}

	be, err := shim.Dial(*socket)
	if err != nil {
		log.Fatalf("calibrate: %v (start the shim via bench/configs/calibrate.sh)", err)
	}
	defer be.Close()

	base, err := be.Tokenize("The capital of France is", true)
	if err != nil {
		log.Fatalf("tokenize: %v", err)
	}
	// Token VALUES barely affect cost — position count does — so every target
	// length is tiled from one tokenized base, keeping lengths exact.
	tile := func(n int) []int32 {
		out := make([]int32, n)
		for i := range out {
			out[i] = base[i%len(base)]
		}
		return out
	}

	oneRound := func(n, nSteps int, record func(kind string, x int, d time.Duration)) {
		if _, err := be.Evict(0, 0, -1); err != nil {
			log.Fatalf("evict: %v", err)
		}
		t0 := time.Now()
		if err := be.Prefill(0, tile(n), 0); err != nil {
			log.Fatalf("prefill(%d): %v", n, err)
		}
		if record != nil {
			record("prefill", n, time.Since(t0))
		}
		for i := 0; i < nSteps; i++ {
			ctx := n + i // tokens cached when this step executes
			t0 = time.Now()
			res, err := be.Step([]engine.SeqID{0})
			if err != nil {
				log.Fatalf("step(len=%d,i=%d): %v", n, i, err)
			}
			if record != nil {
				record("step", ctx, time.Since(t0))
			}
			if res.Tokens[0].Finished {
				break
			}
		}
	}

	// ---- thermal warmup: hammer until equilibrium, record nothing ----------
	fmt.Fprintf(os.Stderr, "thermal warmup: %s of sustained load...\n", *warmFor)
	warmEnd := time.Now().Add(*warmFor)
	for time.Now().Before(warmEnd) {
		oneRound(256, 8, nil)
	}

	// ---- recorded sweeps, MHz sampled throughout ---------------------------
	mhzStop := make(chan struct{})
	mhzDone := make(chan struct{})
	var mhzSamples []float64
	go func() {
		defer close(mhzDone)
		tick := time.NewTicker(200 * time.Millisecond)
		defer tick.Stop()
		for {
			select {
			case <-mhzStop:
				return
			case <-tick.C:
				if v := readMHz(); v > 0 {
					mhzSamples = append(mhzSamples, v)
				}
			}
		}
	}()

	prefYs := map[int][]float64{} // length -> prefill us, one per sweep
	stepYs := map[int][]float64{} // ctx    -> step us, one per sweep

	rng := rand.New(rand.NewPCG(42, 0)) // seeded: reruns shuffle identically
	for sweep := 1; sweep <= *sweeps; sweep++ {
		order := rng.Perm(len(lens)) // break the ctx↔time correlation
		for _, idx := range order {
			n := lens[idx]
			oneRound(n, *steps, func(kind string, x int, d time.Duration) {
				us := float64(d.Microseconds())
				if kind == "prefill" {
					prefYs[x] = append(prefYs[x], us)
				} else {
					stepYs[x] = append(stepYs[x], us)
				}
			})
			fmt.Fprintf(os.Stderr, "sweep %d len %4d: prefill=%8.1fms\n",
				sweep, n, prefYs[n][len(prefYs[n])-1]/1000)
		}
	}
	close(mhzStop)
	<-mhzDone

	// ---- per-x medians across sweeps, then fit -----------------------------
	prefX, prefY := medianPoints(prefYs)
	stepX, stepY := medianPoints(stepYs)

	prefill, err := costmodel.Fit(prefX, prefY)
	if err != nil {
		log.Fatalf("prefill fit: %v", err)
	}
	step, err := costmodel.Fit(stepX, stepY)
	if err != nil {
		log.Fatalf("step fit: %v", err)
	}

	var p costmodel.Params
	p.Prefill = prefill
	p.Step = step
	p.Meta.FittedAt = time.Now().Format(time.RFC3339)
	p.Meta.Source = fmt.Sprintf(
		"quanta-calibrate v2: lengths=%s, steps=%d, sweeps=%d, thermal-warmup=%s, shuffled order, per-x medians",
		*lengths, *steps, *sweeps, *warmFor)
	p.Meta.EngineArgs = *args
	if len(mhzSamples) > 0 {
		mean, min := 0.0, mhzSamples[0]
		for _, v := range mhzSamples {
			mean += v
			if v < min {
				min = v
			}
		}
		p.Meta.MHzMean = mean / float64(len(mhzSamples))
		p.Meta.MHzMin = min

		// Declining clock during recording = still not at equilibrium.
		half := len(mhzSamples) / 2
		a, b := meanOf(mhzSamples[:half]), meanOf(mhzSamples[half:])
		if b < a*0.95 {
			p.Meta.Notes = fmt.Sprintf("WARNING: MHz declined during recording (%.0f -> %.0f) — thermal not settled", a, b)
			fmt.Fprintln(os.Stderr, p.Meta.Notes)
		}
	}

	if err := p.Save(*out); err != nil {
		log.Fatalf("save: %v", err)
	}

	fmt.Printf("prefill(n)  = %.0fus + %.1fus·n     R²=%.4f  max|res|=%.0fus  (n=%d medians)\n",
		prefill.InterceptUS, prefill.SlopeUS, prefill.R2, prefill.MaxAbsResUS, prefill.N)
	fmt.Printf("step(ctx)   = %.0fus + %.2fus·ctx   R²=%.4f  max|res|=%.0fus  (n=%d medians)\n",
		step.InterceptUS, step.SlopeUS, step.R2, step.MaxAbsResUS, step.N)
	fmt.Printf("mhz during recording: mean=%.0f min=%.0f\n", p.Meta.MHzMean, p.Meta.MHzMin)
	fmt.Printf("wrote %s\n", *out)

	if prefill.R2 < 0.98 {
		fmt.Fprintf(os.Stderr, "WARNING: prefill R²=%.4f — inspect residuals before trusting\n", prefill.R2)
	}
	if step.R2 < 0.80 {
		fmt.Fprintf(os.Stderr, "WARNING: step R²=%.4f — slope may not be trustworthy\n", step.R2)
	}
}

func medianPoints(m map[int][]float64) (xs, ys []float64) {
	keys := make([]int, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Ints(keys)
	for _, k := range keys {
		vs := append([]float64(nil), m[k]...)
		sort.Float64s(vs)
		xs = append(xs, float64(k))
		ys = append(ys, vs[len(vs)/2])
	}
	return
}

func meanOf(v []float64) float64 {
	if len(v) == 0 {
		return 0
	}
	s := 0.0
	for _, x := range v {
		s += x
	}
	return s / float64(len(v))
}

func readMHz() float64 {
	data, err := os.ReadFile("/proc/cpuinfo")
	if err != nil {
		return 0
	}
	var sum float64
	var n int
	for _, line := range strings.Split(string(data), "\n") {
		if strings.HasPrefix(line, "cpu MHz") {
			var v float64
			if _, err := fmt.Sscanf(line[strings.Index(line, ":")+1:], "%f", &v); err == nil {
				sum += v
				n++
			}
		}
	}
	if n == 0 {
		return 0
	}
	return sum / float64(n)
}
