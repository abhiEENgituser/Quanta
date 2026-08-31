// Command quanta-validate is the referee for the cost model: it measures the
// real engine at prompt lengths the model was NEVER FITTED ON and compares
// against pure prediction. Validating on training points proves nothing —
// interpolation on held-out points is what earns the model the right to
// replace the engine.
//
// The synthetic backend is not run here: its op durations ARE params.At(x) by
// construction, so timing its sleeps would only measure Go's sleep accuracy.
// Real measurements follow the same discipline as calibration (thermal
// warmup, shuffled order, per-x medians across sweeps) because both sides of
// a comparison must be taken under the same conditions.
//
// Exit code is the verdict: nonzero if end-to-end disagreement exceeds the
// threshold. Per-op tables and a CSV for plotting go to bench/results/.
package main

import (
	"flag"
	"fmt"
	"log"
	"math"
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
		params  = flag.String("params", "internal/engine/costmodel/params.json", "fitted model")
		lengths = flag.String("lengths", "24,96,192,384", "HELD-OUT prompt lengths (not calibration knots)")
		steps   = flag.Int("steps", 32, "decode steps per length")
		sweeps  = flag.Int("sweeps", 3, "recorded sweeps")
		warmFor = flag.Duration("thermal-warmup", 45*time.Second, "sustained load before recording")
		outCSV  = flag.String("out", "bench/results/validation.csv", "comparison CSV")
		maxErr  = flag.Float64("max-e2e-err", 10.0, "fail threshold, percent")
	)
	flag.Parse()

	p, err := costmodel.Load(*params)
	if err != nil {
		log.Fatal(err)
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
		log.Fatalf("validate: %v (start via bench/configs/validate.sh)", err)
	}
	defer be.Close()

	base, err := be.Tokenize("The capital of France is", true)
	if err != nil {
		log.Fatal(err)
	}
	tile := func(n int) []int32 {
		out := make([]int32, n)
		for i := range out {
			out[i] = base[i%len(base)]
		}
		return out
	}

	oneRound := func(n, nSteps int, rec func(kind string, x int, d time.Duration)) {
		if _, err := be.Evict(0, 0, -1); err != nil {
			log.Fatal(err)
		}
		t0 := time.Now()
		if err := be.Prefill(0, tile(n), 0); err != nil {
			log.Fatalf("prefill(%d): %v", n, err)
		}
		if rec != nil {
			rec("prefill", n, time.Since(t0))
		}
		for i := 0; i < nSteps; i++ {
			ctx := n + i
			t0 = time.Now()
			res, err := be.Step([]engine.SeqID{0})
			if err != nil {
				log.Fatal(err)
			}
			if rec != nil {
				rec("step", ctx, time.Since(t0))
			}
			if res.Tokens[0].Finished {
				break
			}
		}
	}

	fmt.Fprintf(os.Stderr, "thermal warmup: %s...\n", *warmFor)
	for end := time.Now().Add(*warmFor); time.Now().Before(end); {
		oneRound(256, 8, nil)
	}

	prefYs := map[int][]float64{}
	stepYs := map[int][]float64{}
	rng := rand.New(rand.NewPCG(7, 0))
	for sweep := 1; sweep <= *sweeps; sweep++ {
		for _, idx := range rng.Perm(len(lens)) {
			n := lens[idx]
			oneRound(n, *steps, func(kind string, x int, d time.Duration) {
				us := float64(d.Microseconds())
				if kind == "prefill" {
					prefYs[x] = append(prefYs[x], us)
				} else {
					stepYs[x] = append(stepYs[x], us)
				}
			})
			fmt.Fprintf(os.Stderr, "sweep %d len %d done\n", sweep, n)
		}
	}

	csv, err := os.Create(*outCSV)
	if err != nil {
		log.Fatal(err)
	}
	defer csv.Close()
	fmt.Fprintln(csv, "kind,x,measured_us,predicted_us,err_pct")

	report := func(kind string, ys map[int][]float64, predict func(float64) time.Duration) (meanAbs, worst float64) {
		xs := make([]int, 0, len(ys))
		for x := range ys {
			xs = append(xs, x)
		}
		sort.Ints(xs)
		var n int
		for _, x := range xs {
			vs := append([]float64(nil), ys[x]...)
			sort.Float64s(vs)
			measured := vs[len(vs)/2]
			predicted := float64(predict(float64(x)).Microseconds())
			errPct := (predicted - measured) / measured * 100
			fmt.Fprintf(csv, "%s,%d,%.0f,%.0f,%.2f\n", kind, x, measured, predicted, errPct)
			meanAbs += math.Abs(errPct)
			if math.Abs(errPct) > math.Abs(worst) {
				worst = errPct
			}
			n++
		}
		return meanAbs / float64(n), worst
	}

	prefMean, prefWorst := report("prefill", prefYs, p.Prefill.At)
	stepMean, stepWorst := report("step", stepYs, p.Step.At)

	// End-to-end: full generation (prefill + all steps) per held-out length —
	// the quantity a simulation actually integrates, where errors compound.
	fmt.Printf("%-8s %10s %12s %12s %8s\n", "e2e len", "measured", "predicted", "err", "")
	var e2eMean, e2eWorst float64
	for _, n := range lens {
		var measured float64
		for _, v := range medianAll(prefYs, n) {
			measured += v
		}
		for i := 0; i < *steps; i++ {
			measured += median(stepYs[n+i])
		}
		predicted := float64(p.Prefill.At(float64(n)).Microseconds())
		for i := 0; i < *steps; i++ {
			predicted += float64(p.Step.At(float64(n + i)).Microseconds())
		}
		errPct := (predicted - measured) / measured * 100
		fmt.Fprintf(csv, "e2e,%d,%.0f,%.0f,%.2f\n", n, measured, predicted, errPct)
		fmt.Printf("%-8d %9.1fms %11.1fms %+7.2f%%\n", n, measured/1000, predicted/1000, errPct)
		e2eMean += math.Abs(errPct) / float64(len(lens))
		if math.Abs(errPct) > math.Abs(e2eWorst) {
			e2eWorst = errPct
		}
	}

	fmt.Printf("\nper-op:  prefill mean|err|=%.1f%% worst=%+.1f%%   step mean|err|=%.1f%% worst=%+.1f%%\n",
		prefMean, prefWorst, stepMean, stepWorst)
	fmt.Printf("e2e:     mean|err|=%.1f%%  worst=%+.1f%%  (threshold %.0f%%)\n", e2eMean, e2eWorst, *maxErr)
	fmt.Printf("wrote %s\n", *outCSV)

	if math.Abs(e2eWorst) > *maxErr {
		fmt.Fprintf(os.Stderr, "VALIDATION FAILED: worst e2e error %+.1f%% exceeds %.0f%%\n", e2eWorst, *maxErr)
		os.Exit(1)
	}
	fmt.Println("VALIDATION PASSED")
}

func median(vs []float64) float64 {
	if len(vs) == 0 {
		return 0
	}
	s := append([]float64(nil), vs...)
	sort.Float64s(s)
	return s[len(s)/2]
}

func medianAll(m map[int][]float64, x int) []float64 {
	return []float64{median(m[x])}
}
