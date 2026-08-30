// Command quanta-bench is an open-loop load generator for quantad.
//
// Open loop means arrivals follow a predetermined schedule that never waits
// for responses. A closed loop (send, wait, send) lets a slow server pace its
// own examiner: a 2s stall is sampled once instead of by every request that
// would have arrived during it — coordinated omission, which systematically
// erases the tail latencies this project exists to measure.
//
// Arrivals are Poisson: inter-arrival gaps drawn from an exponential
// distribution, the memoryless model of independent users. The rate flag sets
// how hard the server is pushed; Poisson sets the texture of the pushing —
// natural bursts and lulls rather than a metronome that never lets a queue
// form.
//
// Every request's latency is measured from its INTENDED send time, and the
// intended-vs-actual gap (drift) is tracked per request. Growing drift means
// the generator itself could not keep its schedule, so the claimed offered
// load is false and the run is invalid — the instrument audits itself.
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math/rand/v2"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/abhiEENgituser/Quanta/internal/metrics"
)

type result struct {
	intended     time.Time
	drift        time.Duration
	ttft         time.Duration
	e2e          time.Duration
	tokens       int
	promptTokens int
	recorded     bool // false during warmup — sent to keep load real, not measured
	err          error
}

func main() {
	var (
		target   = flag.String("target", "http://127.0.0.1:8080", "quantad base URL")
		rate     = flag.Float64("rate", 1.0, "mean arrival rate, requests/second")
		duration = flag.Duration("duration", 30*time.Second, "measured run length (after warmup)")
		warmup   = flag.Duration("warmup", 3*time.Second, "requests sent but not recorded")
		prompt   = flag.String("prompt", "The capital of France is", "prompt text")
		repeat   = flag.Int("prompt-repeat", 1, "repeat the prompt N times (longer prompts)")
		maxTok   = flag.Int("max-tokens", 64, "max_tokens per request")
		outPath  = flag.String("out", "-", "per-request CSV output ('-' = stdout)")
	)
	flag.Parse()

	promptText := strings.TrimSpace(strings.Repeat(*prompt+" ", *repeat))
	body, _ := json.Marshal(map[string]any{"prompt": promptText, "max_tokens": *maxTok})

	// Record the conditions the numbers were taken under a number without
	// its conditions is not a measurement (see docs/baseline.md).
	fmt.Fprintf(os.Stderr, "# governor=%s load=%s\n", readFile("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"), readFile("/proc/loadavg"))

	mhzStop := make(chan struct{})
	var mhzMean, mhzMin float64
	go sampleMHz(mhzStop, &mhzMean, &mhzMin)

	reg := metrics.NewRegistry()

	var (
		mu      sync.Mutex
		results []result
		wg      sync.WaitGroup
	)

	// No Timeout on the client: it would cap the whole streamed response, and
	// a slow-but-alive stream is exactly what must be measured, not killed.
	client := &http.Client{}

	start := time.Now()
	endAt := start.Add(*warmup + *duration)
	intended := start

	for {
		// Exponential gap with mean 1/rate seconds: the Poisson schedule.
		gap := time.Duration(rand.ExpFloat64() / *rate * float64(time.Second))
		intended = intended.Add(gap)
		if intended.After(endAt) {
			break
		}

		// Sleep UNTIL the intended instant — never until "after the previous
		// response", which is what would close the loop. Firing happens in a
		// fresh goroutine so this scheduler is never busy when the next
		// arrival is due.
		if d := time.Until(intended); d > 0 {
			time.Sleep(d)
		}
		actual := time.Now()

		r := result{
			intended: intended,
			drift:    actual.Sub(intended),
			recorded: actual.Sub(start) >= *warmup,
		}

		wg.Add(1)
		go func(r result) {
			defer wg.Done()
			// ITL records live inside fire (one sample per token gap, across
			// all requests — smoothness is a property of the run); the
			// per-request summaries record here, once the outcome is known.
			fire(client, *target, body, &r, reg.Histogram("itl"))
			if r.recorded && r.err == nil {
				reg.Histogram("drift").Record(r.drift)
				reg.Histogram("ttft").Record(r.ttft)
				reg.Histogram("e2e").Record(r.e2e)
			}
			mu.Lock()
			results = append(results, r)
			mu.Unlock()
		}(r)
	}

	wg.Wait()
	close(mhzStop)
	wall := time.Since(start)

	writeCSV(*outPath, results)
	summarize(os.Stderr, reg, results, *rate, wall-*warmup, mhzMean, mhzMin)
	scrapeServerMetrics(os.Stderr, client, *target)

	// A run in which nothing succeeded must fail loudly. This binary once
	// exited 0 after 40 seconds of connection-refused — silence read as
	// success by everything above it.
	var okRecorded int
	for _, r := range results {
		if r.recorded && r.err == nil {
			okRecorded++
		}
	}
	if okRecorded == 0 {
		fmt.Fprintln(os.Stderr, "quanta-bench: FAILED — zero successful recorded requests")
		os.Exit(1)
	}
}

// fire sends one request and reads its SSE stream, timestamping every token
// event as it arrives at the client. TTFT and e2e are measured from the
// INTENDED send time: if the generator was late, that lateness is part of the
// latency a punctual user would have experienced — measuring from actual send
// would quietly forgive it (coordinated omission by the back door).
func fire(client *http.Client, target string, body []byte, r *result, itl *metrics.Histogram) {
	resp, err := client.Post(target+"/v1/generate", "application/json", bytes.NewReader(body))
	if err != nil {
		r.err = err
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		r.err = fmt.Errorf("status %d: %s", resp.StatusCode, msg)
		return
	}

	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	var last time.Time
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := line[len("data: "):]

		var ev struct {
			Text         *string `json:"text"`
			Tokens       int     `json:"tokens"`
			PromptTokens int     `json:"prompt_tokens"`
			FinishReason string  `json:"finish_reason"`
		}
		if json.Unmarshal([]byte(payload), &ev) != nil {
			continue
		}

		switch {
		case ev.Text != nil:
			now := time.Now()
			if r.ttft == 0 {
				r.ttft = now.Sub(r.intended)
			} else if r.recorded {
				itl.Record(now.Sub(last))
			}
			last = now
		case ev.FinishReason != "":
			r.tokens = ev.Tokens
			r.promptTokens = ev.PromptTokens
		}
	}
	r.e2e = time.Since(r.intended)
	if err := sc.Err(); err != nil {
		r.err = err
	}
}

func summarize(w io.Writer, reg *metrics.Registry, results []result, rate float64, window time.Duration, mhzMean, mhzMin float64) {
	var sent, ok, failed, tokens int
	var recorded []result
	for _, r := range results {
		sent++
		if r.err != nil {
			failed++
			continue
		}
		ok++
		if r.recorded {
			recorded = append(recorded, r)
			tokens += r.tokens
		}
	}

	fmt.Fprintf(w, "\n=== quanta-bench: rate=%.2f req/s window=%s ===\n", rate, window.Round(time.Second))
	fmt.Fprintf(w, "requests: sent=%d ok=%d failed=%d recorded=%d\n", sent, ok, failed, len(recorded))
	fmt.Fprintf(w, "cpu MHz during run: mean=%.0f min=%.0f\n", mhzMean, mhzMin)
	fmt.Fprintf(w, "throughput: %.1f tokens/s across recorded window\n", float64(tokens)/window.Seconds())

	snap := reg.Snapshot()
	pr := func(name string, q metrics.Quantiles) {
		fmt.Fprintf(w, "%-6s n=%-5d p50=%-10s p95=%-10s p99=%-10s max=%s\n",
			name, q.Count, q.P50, q.P95, q.P99, q.Max)
	}
	pr("ttft", snap["ttft"])
	pr("itl", snap["itl"])
	pr("e2e", snap["e2e"])
	pr("drift", snap["drift"])

	// Validity gate: drift growing across the run means the generator fell
	// behind its own schedule — the offered load was lower than claimed and
	// every number above has the wrong x-axis.
	if verdict := driftVerdict(recorded); verdict != "" {
		fmt.Fprintf(w, "%s\n", verdict)
	}
}

func driftVerdict(recorded []result) string {
	if len(recorded) < 10 {
		return "drift: too few recorded requests for a trend verdict"
	}
	half := len(recorded) / 2
	mean := func(rs []result) time.Duration {
		var s time.Duration
		for _, r := range rs {
			s += r.drift
		}
		return s / time.Duration(len(rs))
	}
	a, b := mean(recorded[:half]), mean(recorded[half:])
	if b > 2*a+time.Millisecond {
		return fmt.Sprintf("drift: INVALID RUN — grew %s -> %s between halves; generator could not hold the schedule", a, b)
	}
	return fmt.Sprintf("drift: OK (first half %s, second half %s)", a, b)
}

func writeCSV(path string, results []result) {
	out := os.Stdout
	if path != "-" {
		f, err := os.Create(path)
		if err != nil {
			fmt.Fprintf(os.Stderr, "csv: %v\n", err)
			return
		}
		defer f.Close()
		out = f
	}
	fmt.Fprintln(out, "intended_unix_us,drift_us,prompt_tokens,tokens,ttft_us,e2e_us,recorded,error")
	for _, r := range results {
		errs := ""
		if r.err != nil {
			errs = strings.ReplaceAll(r.err.Error(), ",", ";")
		}
		fmt.Fprintf(out, "%d,%d,%d,%d,%d,%d,%t,%s\n",
			r.intended.UnixMicro(), r.drift.Microseconds(), r.promptTokens,
			r.tokens, r.ttft.Microseconds(), r.e2e.Microseconds(), r.recorded, errs)
	}
}

// scrapeServerMetrics prints quantad's own view at run end, so the report
// shows the same requests from both sides of the wire: the client sees TTFT,
// the server splits it into queue wait vs prefill.
func scrapeServerMetrics(w io.Writer, client *http.Client, target string) {
	resp, err := client.Get(target + "/v1/metrics")
	if err != nil {
		return
	}
	defer resp.Body.Close()
	fmt.Fprintf(w, "\n--- server-side (/v1/metrics; accumulates since quantad start) ---\n")
	io.Copy(w, resp.Body)
	fmt.Fprintln(w)
}

func sampleMHz(stop <-chan struct{}, mean, min *float64) {
	var sum float64
	var n int
	*min = 1 << 30
	tick := time.NewTicker(100 * time.Millisecond)
	defer tick.Stop()
	for {
		select {
		case <-stop:
			if n > 0 {
				*mean = sum / float64(n)
			}
			return
		case <-tick.C:
			if v := readMHz(); v > 0 {
				sum += v
				n++
				if v < *min {
					*min = v
				}
			}
		}
	}
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

func readFile(path string) string {
	b, err := os.ReadFile(path)
	if err != nil {
		return "?"
	}
	return strings.TrimSpace(string(b))
}
