#!/usr/bin/env python3
"""Plot the Phase 1 single-stream baseline from bench_single.sh output.

Reads the per-run CSVs (raw request rows), computes per-run percentiles from
recorded rows only, then plots mean across the 5 repeats with min-max error
bars. Variance is the point: a run without error bars is a single sample
wearing a costume (see docs/baseline.md for two numbers that died that way).

Usage: python3 bench/plot/plot_single.py bench/results/single
"""

import csv
import glob
import os
import re
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_runs(outdir):
    """-> {(threads, rep): [per-run dicts]}, each with percentiles + prompt tokens."""
    runs = {}
    for path in sorted(glob.glob(os.path.join(outdir, "t*_rep*_run*.csv"))):
        m = re.match(r"t(\d+)_rep(\d+)_run(\d+)\.csv", os.path.basename(path))
        if not m:
            continue
        threads, rep = int(m.group(1)), int(m.group(2))

        ttfts, e2es, ptoks = [], [], []
        with open(path) as f:
            for row in csv.DictReader(f):
                if row["recorded"] != "true" or row["error"]:
                    continue
                ttfts.append(int(row["ttft_us"]) / 1000.0)  # -> ms
                e2es.append(int(row["e2e_us"]) / 1000.0)
                ptoks.append(int(row["prompt_tokens"]))
        if len(ttfts) < 5:
            print(f"skip {path}: only {len(ttfts)} recorded rows", file=sys.stderr)
            continue

        ttfts.sort()
        q = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))]
        runs.setdefault((threads, rep), []).append({
            "p50": statistics.median(ttfts),
            "p95": q(ttfts, 0.95),
            "prompt_tokens": statistics.median(ptoks),
            "e2e_p50": statistics.median(sorted(e2es)),
        })
    return runs


def series(runs, threads, key):
    """-> (x prompt tokens, y mean, yerr [below, above]) across repeats."""
    xs, ys, lo, hi = [], [], [], []
    for (t, rep), rr in sorted(runs.items(), key=lambda kv: kv[0][1]):
        if t != threads:
            continue
        vals = [r[key] for r in rr]
        m = statistics.mean(vals)
        xs.append(statistics.median([r["prompt_tokens"] for r in rr]))
        ys.append(m)
        lo.append(m - min(vals))
        hi.append(max(vals) - m)
    return xs, ys, [lo, hi]


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "bench/results/single"
    runs = load_runs(outdir)
    if not runs:
        sys.exit(f"no usable CSVs in {outdir}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for threads, marker in ((4, "o"), (3, "s")):
        xs, ys, err = series(runs, threads, "p50")
        if xs:
            ax.errorbar(xs, ys, yerr=err, marker=marker, capsize=4,
                        label=f"TTFT p50, {threads} threads")
        xs, ys, err = series(runs, threads, "p95")
        if xs:
            ax.errorbar(xs, ys, yerr=err, marker=marker, capsize=4,
                        linestyle="--", alpha=0.7,
                        label=f"TTFT p95, {threads} threads")

    ax.set_xlabel("prompt length (tokens)")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("Single-stream TTFT vs prompt length (5 runs, min-max bars)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = os.path.join(outdir, "single_baseline.png")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")

    # The -t 3 headline number, printed so it can be written into
    # docs/baseline.md next to the plot.
    for key, label in (("p50", "TTFT p50"), ("e2e_p50", "e2e p50")):
        for (t, rep), rr in sorted(runs.items()):
            vals = [r[key] for r in rr]
            print(f"t={t} rep={rep:3d} ptok~{statistics.median([r['prompt_tokens'] for r in rr]):4.0f}  "
                  f"{label}: mean={statistics.mean(vals):8.1f}ms  "
                  f"min={min(vals):8.1f}  max={max(vals):8.1f}  n={len(vals)}")


if __name__ == "__main__":
    main()
