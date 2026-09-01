// Package costmodel holds the fitted cost parameters that let a calibrated
// sleep() impersonate the real engine, plus the least-squares fitting used to
// produce them.
//
// The model is two lines, both demanded by measured data (docs/baseline.md):
//
//	prefill(n)  = a + b·n    n   = prompt tokens submitted in one Prefill
//	step(ctx)   = c + d·ctx  ctx = tokens already cached when the step runs
//
// Decode is NOT a constant — the context-length term exists because a single
// n=1 "20.27ms" claim died on repeats while the growth itself (17→24ms over
// 40→440 ctx) survived them.
//
// Params are calibrated against Backend OP durations measured through the
// real shim client — socket round-trip included — because the synthetic
// backend's job is to impersonate the engine as the scheduler sees it, not as
// a bare-metal probe sees it.
package costmodel

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"time"
)

// Line is y = Intercept + Slope·x, in microseconds, with its fit diagnostics
// carried alongside — a parameter file you cannot judge is a parameter file
// you cannot trust.
type Line struct {
	InterceptUS float64 `json:"intercept_us"`
	SlopeUS     float64 `json:"slope_us_per_token"`
	R2          float64 `json:"r2"`
	N           int     `json:"n_points"`
	MaxAbsResUS float64 `json:"max_abs_residual_us"`
}

// At evaluates the line, clamped at zero — a fitted negative intercept must
// never produce a negative duration for tiny x.
func (l Line) At(x float64) time.Duration {
	us := l.InterceptUS + l.SlopeUS*x
	if us < 0 {
		us = 0
	}
	return time.Duration(us) * time.Microsecond
}

type Params struct {
	// Prefill duration as a function of prompt tokens.
	Prefill Line `json:"prefill"`
	// Single decode-step duration as a function of current context length.
	Step Line `json:"step"`
	// Batched decode-step duration as a function of batch size, measured at
	// BatchRefCtx tokens of context per sequence. The sublinearity of this
	// line versus B × Step is the entire economic case for batching.
	// Optional until the batch sweep has run (Phase 3).
	StepBatch   Line `json:"step_batch,omitempty"`
	BatchRefCtx int  `json:"batch_ref_ctx,omitempty"`

	Meta struct {
		FittedAt   string  `json:"fitted_at"`
		Source     string  `json:"source"`
		EngineArgs string  `json:"engine_args"`
		MHzMean    float64 `json:"mhz_mean,omitempty"`
		MHzMin     float64 `json:"mhz_min,omitempty"`
		Notes      string  `json:"notes,omitempty"`
	} `json:"meta"`
}

func Load(path string) (Params, error) {
	var p Params
	data, err := os.ReadFile(path)
	if err != nil {
		return p, fmt.Errorf("costmodel: %w", err)
	}
	if err := json.Unmarshal(data, &p); err != nil {
		return p, fmt.Errorf("costmodel: parse %s: %w", path, err)
	}
	if p.Prefill.N == 0 || p.Step.N == 0 {
		return p, fmt.Errorf("costmodel: %s has unfitted lines — run make calibrate", path)
	}
	return p, nil
}

func (p Params) Save(path string) error {
	data, err := json.MarshalIndent(p, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0o644)
}

// Fit computes ordinary least squares for y = a + b·x, with R² and the
// largest absolute residual. The residual matters as much as R²: our own
// prefill fit scored R²=0.995 while hiding real curvature — R² near 1 is
// necessary, never sufficient. Callers print residuals; humans judge them.
func Fit(xs, ys []float64) (Line, error) {
	n := len(xs)
	if n != len(ys) {
		return Line{}, fmt.Errorf("costmodel: %d xs vs %d ys", n, len(ys))
	}
	if n < 3 {
		return Line{}, fmt.Errorf("costmodel: need >=3 points, got %d", n)
	}

	var sx, sy float64
	for i := range xs {
		sx += xs[i]
		sy += ys[i]
	}
	mx, my := sx/float64(n), sy/float64(n)

	var sxx, sxy float64
	for i := range xs {
		dx := xs[i] - mx
		sxx += dx * dx
		sxy += dx * (ys[i] - my)
	}
	if sxx == 0 {
		return Line{}, fmt.Errorf("costmodel: all x identical — cannot fit a slope")
	}

	b := sxy / sxx
	a := my - b*mx

	var ssRes, ssTot, maxRes float64
	for i := range xs {
		res := ys[i] - (a + b*xs[i])
		ssRes += res * res
		d := ys[i] - my
		ssTot += d * d
		if r := math.Abs(res); r > maxRes {
			maxRes = r
		}
	}
	r2 := 1.0
	if ssTot > 0 {
		r2 = 1 - ssRes/ssTot
	}

	return Line{InterceptUS: a, SlopeUS: b, R2: r2, N: n, MaxAbsResUS: maxRes}, nil
}
