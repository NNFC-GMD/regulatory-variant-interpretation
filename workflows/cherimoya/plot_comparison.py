#!/usr/bin/env python3
"""Plot Cherimoya against ChromBPNet, fold by fold, from summarize.py's paired table.

One panel per measure (default: counts Pearson and counts Spearman on the
test peaks), ChromBPNet on x and Cherimoya on y, one dot per dataset and fold,
so a dataset with all five folds scored is five dots. Above the diagonal,
Cherimoya did better; for MSE and JSD, where lower is better, below it.

Each dataset has its own colour and marker: Okabe-Ito's eight colourblind-safe
colours, repeated with a new marker shape every eight datasets, so up to 40
datasets stay distinguishable without relying on hue alone.

    python workflows/cherimoya/plot_comparison.py --paired <dir>/comparison_paired.tsv --out <dir>/comparison

writes <out>.pdf and <out>.png. Runs in the pixi `cherimoya` environment (matplotlib).
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

# Okabe & Ito (2008), in their order; black is last so the first datasets get hues.
OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
LABELS = {
    "counts_pearson": "counts Pearson r",
    "counts_spearman": "counts Spearman rho",
    "counts_mse": "counts MSE (log counts)",
    "median_jsd": "profile median JSD",
    "median_norm_jsd": "profile median normalised JSD",
}


def style(i: int) -> tuple[str, str]:
    """Colour and marker of the i-th dataset."""
    return OKABE_ITO[i % len(OKABE_ITO)], MARKERS[(i // len(OKABE_ITO)) % len(MARKERS)]


def read_paired(path: Path, measures: list[str]) -> dict[str, dict[str, list[tuple[float, float]]]]:
    """{dataset: {measure: [(chrombpnet, cherimoya) per fold]}}"""
    out: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    with open(path) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            for m in measures:
                out[row["dataset"]][m].append((float(row[f"{m}_chrombpnet"]), float(row[f"{m}_cherimoya"])))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paired", type=Path, required=True, help="comparison_paired.tsv from summarize.py")
    ap.add_argument("--out", type=Path, required=True, help="output path without extension")
    ap.add_argument("--measures", nargs="+", default=["counts_pearson", "counts_spearman"], choices=list(LABELS))
    ap.add_argument("--strip-prefix", default="", help="drop this prefix from dataset names in the legend")
    args = ap.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = read_paired(args.paired, args.measures)
    if not data:
        print(f"no rows in {args.paired}", file=sys.stderr)
        return 1
    datasets = sorted(data)
    n_folds = sum(len(data[d][args.measures[0]]) for d in datasets)

    fig, axes = plt.subplots(1, len(args.measures), figsize=(5.2 * len(args.measures) + 3.2, 5.4), squeeze=False)
    for ax, m in zip(axes[0], args.measures):
        values = [v for d in datasets for pair in data[d][m] for v in pair]
        lo, hi = min(values), max(values)
        pad = 0.05 * (hi - lo or 1)
        lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], color="#999999", lw=1, ls="--", zorder=0)
        for i, d in enumerate(datasets):
            colour, marker = style(i)
            x, y = zip(*data[d][m])
            ax.scatter(x, y, s=34, color=colour, marker=marker, edgecolor="#333333", linewidth=0.4,
                       alpha=0.9, label=d.removeprefix(args.strip_prefix))  # fmt: skip
        better = sum((b > a) if m not in ("counts_mse", "median_jsd") else (b < a)
                     for d in datasets for a, b in data[d][m])  # fmt: skip
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel(f"ChromBPNet {LABELS[m]}")
        ax.set_ylabel(f"Cherimoya {LABELS[m]}")
        ax.set_title(f"{LABELS[m]}: Cherimoya better on {better} of {n_folds} folds", fontsize=10)
        ax.grid(True, lw=0.3, color="#DDDDDD")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0 - 3.0 / fig.get_figwidth(), 0.5),
               fontsize=6.5, frameon=False, ncol=1 if len(labels) <= 20 else 2, markerscale=0.9,
               title=f"{len(datasets)} datasets x folds", title_fontsize=7)  # fmt: skip
    fig.suptitle("Test peaks, ChromBPNet's measures, same regions for both models", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1.0 - 3.2 / fig.get_figwidth(), 0.95))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {args.out}.pdf/.png ({n_folds} folds, {len(datasets)} datasets)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
