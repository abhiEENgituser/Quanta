#!/usr/bin/env python3
"""Synthetic-vs-real comparison plot from quanta-validate output.

This is the evidence that every later simulated number means something:
measured engine cost at HELD-OUT prompt lengths against the cost model's pure
prediction. Usage: python3 bench/plot/plot_validation.py [validation.csv]
"""

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "bench/results/validation.csv"
    rows = {"prefill": [], "step": [], "e2e": []}
    with open(path) as f:
        for row in csv.DictReader(f):
            rows[row["kind"]].append(
                (int(row["x"]), float(row["measured_us"]) / 1000,
                 float(row["predicted_us"]) / 1000, float(row["err_pct"])))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for ax, kind, unit in ((axes[0], "prefill", "ms"), (axes[1], "step", "ms"), (axes[2], "e2e", "ms")):
        pts = sorted(rows[kind])
        xs = [p[0] for p in pts]
        ax.plot(xs, [p[1] for p in pts], "o-", label="measured (real engine)")
        ax.plot(xs, [p[2] for p in pts], "s--", label="predicted (cost model)")
        worst = max(abs(p[3]) for p in pts)
        ax.set_title(f"{kind} — worst err {worst:.1f}%")
        ax.set_xlabel("prompt tokens" if kind != "step" else "context length")
        ax.set_ylabel(unit)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Cost model validation at held-out lengths (never fitted on)")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(path), "validation.png")
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
