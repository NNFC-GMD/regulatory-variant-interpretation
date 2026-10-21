#!/usr/bin/env python3
"""Pool every fold's compare.tsv into one Cherimoya-vs-ChromBPNet table.

Each compare.tsv (compare_chrombpnet.py) holds both models' measures on the
same test peaks of one dataset and fold, so the two can be paired fold by fold.
This writes, in --out-dir:

  comparison.tsv          every fold's rows, plus Cherimoya's training time
                          and peak memory from fit.time
  comparison_paired.tsv   one row per (dataset, fold): each measure for both
                          models and the difference, Cherimoya minus ChromBPNet
  comparison_summary.txt  per measure, over all folds: both models' medians,
                          the median paired difference, and how many folds
                          each model did better on (higher Pearson/Spearman/
                          normalised JSD, lower MSE/JSD); then per dataset,
                          the fold means

    python workflows/train/summarize.py --root <make_jobs out-root> --out-dir <dir> [--datasets <list.txt>]

--datasets keeps only the datasets of a list (one name per line; blank lines and # comments skipped), e.g. the
datasets both models were interpreted on.
"""

from __future__ import annotations

import argparse
import csv
import logging
import statistics
import sys
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger("summarize")

MEASURES = ("counts_pearson", "counts_spearman", "counts_mse", "median_jsd", "median_norm_jsd")
HIGHER_IS_BETTER = {"counts_pearson": True, "counts_spearman": True, "counts_mse": False,
                    "median_jsd": False, "median_norm_jsd": True}  # fmt: skip
MODELS = ("chrombpnet", "cherimoya")


def read_datasets(path: Path) -> list[str]:
    """A dataset list: one name per line, blank lines and # comments skipped."""
    names = [line.split("#", 1)[0].strip() for line in Path(path).read_text().splitlines()]
    return [n for n in names if n]


def read_rows(root: Path, datasets: list[str] | None = None) -> list[dict]:
    rows = []
    for path in sorted(root.glob("*/fold_*/compare.tsv")):
        if datasets is not None and path.parent.parent.name not in datasets:
            continue
        fit_time = path.with_name("fit.time")
        seconds, rss_kb = ("NA", "NA")
        if fit_time.exists():
            parts = fit_time.read_text().split()
            seconds, rss_kb = (parts + ["NA", "NA"])[:2]
        with open(path) as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                is_cheri = row["model"] == "cherimoya"
                row["fit_seconds"] = seconds if is_cheri else "NA"
                row["fit_max_rss_gb"] = f"{int(rss_kb) / 1048576:.1f}" if is_cheri and rss_kb.isdigit() else "NA"
                rows.append(row)
    return rows


def pair(rows: list[dict]) -> list[dict]:
    by_fold: dict[tuple, dict] = defaultdict(dict)
    for r in rows:
        by_fold[(r["dataset"], r["fold"])][r["model"]] = r
    paired = []
    for (dataset, fold), models in sorted(by_fold.items()):
        if set(MODELS) - set(models):
            logger.warning("%s fold %s: only %s", dataset, fold, ", ".join(models))
            continue
        out = {"dataset": dataset, "fold": fold, "n_regions": models["cherimoya"]["n_regions"]}
        for m in MEASURES:
            a, b = float(models["chrombpnet"][m]), float(models["cherimoya"][m])
            out[f"{m}_chrombpnet"], out[f"{m}_cherimoya"], out[f"{m}_diff"] = a, b, b - a
        paired.append(out)
    return paired


def summary(paired: list[dict]) -> str:
    if not paired:
        return "no fold has both models scored yet\n"
    n_ds = len({p["dataset"] for p in paired})
    lines = [f"{len(paired)} fold(s) over {n_ds} dataset(s); difference = Cherimoya - ChromBPNet", ""]
    lines.append(
        f"{'measure':<17}{'ChromBPNet':>12}{'Cherimoya':>12}{'median diff':>13}  better on (Cherimoya / ChromBPNet / tie)"
    )
    for m in MEASURES:
        a = [p[f"{m}_chrombpnet"] for p in paired]
        b = [p[f"{m}_cherimoya"] for p in paired]
        d = [p[f"{m}_diff"] for p in paired]
        sign = 1 if HIGHER_IS_BETTER[m] else -1
        cheri = sum(sign * x > 0 for x in d)
        cbp = sum(sign * x < 0 for x in d)
        lines.append(f"{m:<17}{statistics.median(a):>12.4f}{statistics.median(b):>12.4f}{statistics.median(d):>+13.4f}"
                     f"  {cheri} / {cbp} / {len(d) - cheri - cbp}")  # fmt: skip
    lines += ["", "per dataset, mean over its folds (ChromBPNet -> Cherimoya):"]
    by_ds: dict[str, list] = defaultdict(list)
    for p in paired:
        by_ds[p["dataset"]].append(p)
    for ds, ps in sorted(by_ds.items()):
        parts = []
        for m in ("counts_pearson", "median_jsd"):
            parts.append(f"{m} {statistics.fmean(p[f'{m}_chrombpnet'] for p in ps):.3f} -> "
                         f"{statistics.fmean(p[f'{m}_cherimoya'] for p in ps):.3f}")  # fmt: skip
        lines.append(f"  {ds} ({len(ps)} fold(s)): " + "   ".join(parts))
    return "\n".join(lines) + "\n"


def write_tsv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="make_jobs.py's --out-root")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--datasets", type=Path, help="only the datasets of this list (one per line)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%F %T")

    datasets = read_datasets(args.datasets) if args.datasets else None
    rows = read_rows(args.root, datasets)
    if not rows:
        logger.error(
            "no compare.tsv under %s%s", args.root, f" for the datasets of {args.datasets}" if datasets else ""
        )
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(args.out_dir / "comparison.tsv", rows)
    paired = pair(rows)
    write_tsv(args.out_dir / "comparison_paired.tsv", paired)
    text = summary(paired)
    (args.out_dir / "comparison_summary.txt").write_text(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
