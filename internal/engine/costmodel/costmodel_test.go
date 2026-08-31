package costmodel

import (
	"math"
	"path/filepath"
	"testing"
	"time"
)

func TestFitRecoversExactLine(t *testing.T) {
	// y = 1000 + 6·x, exactly. The fit must recover it and report R²=1.
	xs := []float64{16, 64, 128, 256, 512}
	ys := make([]float64, len(xs))
	for i, x := range xs {
		ys[i] = 1000 + 6*x
	}

	l, err := Fit(xs, ys)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(l.InterceptUS-1000) > 1e-6 || math.Abs(l.SlopeUS-6) > 1e-9 {
		t.Fatalf("got y = %.3f + %.6f·x, want 1000 + 6·x", l.InterceptUS, l.SlopeUS)
	}
	if l.R2 < 0.999999 || l.MaxAbsResUS > 1e-6 {
		t.Fatalf("exact data should fit exactly: r2=%v maxres=%v", l.R2, l.MaxAbsResUS)
	}
}

func TestFitReportsImperfection(t *testing.T) {
	// Superlinear data (quadratic term) fitted with a line: R² should stay
	// high — the trap our real prefill fit demonstrated — while the residual
	// gives the game away. Both diagnostics must survive into the result.
	xs := []float64{16, 64, 128, 256, 512}
	ys := make([]float64, len(xs))
	for i, x := range xs {
		ys[i] = 1000 + 6*x + 0.004*x*x
	}

	l, err := Fit(xs, ys)
	if err != nil {
		t.Fatal(err)
	}
	if l.R2 > 0.99999 {
		t.Fatalf("curved data reported as a perfect line: r2=%v", l.R2)
	}
	if l.MaxAbsResUS < 10 {
		t.Fatalf("curvature should show in residuals, got max %v", l.MaxAbsResUS)
	}
}

func TestLineAtClampsNegative(t *testing.T) {
	// Our real prefill fit produced intercept -68ms. A cost model must never
	// hand the simulator a negative duration for a tiny prompt.
	l := Line{InterceptUS: -68000, SlopeUS: 6000}
	if d := l.At(5); d != 0 {
		t.Fatalf("At(5) = %v, want clamp to 0", d)
	}
	if d := l.At(100); d != 532*time.Millisecond {
		t.Fatalf("At(100) = %v, want 532ms", d)
	}
}

func TestSaveLoadRoundTrip(t *testing.T) {
	p := Params{}
	p.Prefill = Line{InterceptUS: 30000, SlopeUS: 6800, R2: 0.99, N: 20}
	p.Step = Line{InterceptUS: 16500, SlopeUS: 17, R2: 0.97, N: 120}
	p.Meta.Source = "test"
	p.Meta.EngineArgs = "-t 3"

	path := filepath.Join(t.TempDir(), "params.json")
	if err := p.Save(path); err != nil {
		t.Fatal(err)
	}
	got, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if got.Prefill != p.Prefill || got.Step != p.Step {
		t.Fatalf("round trip changed params:\n got %+v\nwant %+v", got, p)
	}
}

func TestLoadRejectsUnfitted(t *testing.T) {
	path := filepath.Join(t.TempDir(), "params.json")
	if err := (Params{}).Save(path); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(path); err == nil {
		t.Fatal("empty params loaded without error — synthetic would run on zeros")
	}
}
