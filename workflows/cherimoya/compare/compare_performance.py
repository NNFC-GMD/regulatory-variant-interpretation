#!/usr/bin/env python3
"""Level 1 of the comparison: both models' performance on ChromBPNet's test peaks, fold by fold.

compare_chrombpnet.py has already scored each fold of both models on the same test peaks with ChromBPNet's measures
(<cheri-root>/<dataset>/fold_<k>/compare.tsv and compare_counts.npz). For the datasets of a list (the ones both
models were interpreted on), this writes in --out-dir (default <comparison-root>/performance):

  comparison.tsv, comparison_paired.tsv, comparison_summary.txt
                         summarize.py --datasets <list>: every fold's rows, the paired table (each measure for both
                         models and Cherimoya minus ChromBPNet), and its summary
  statistics.tsv         per measure: both models' medians over folds; the median of the fold differences; the
                         median over datasets of each dataset's mean difference, with a percentile bootstrap 95 % CI
                         (datasets resampled); two-sided Wilcoxon signed-rank tests over the folds and over the
                         dataset means (folds of one dataset share its peaks, so the dataset test is the one to
                         trust; with 10 datasets its exact two-sided p cannot go below 0.002); and how many folds and
                         datasets each model did better on (higher Pearson, Spearman and normalised JSD; lower MSE and
                         JSD)
  statistics.txt         the same, readable
  residuals_by_decile.tsv
                         per dataset, the test peaks of all its folds pooled (each peak is a test peak of exactly one
                         fold), cut into deciles of observed log(1 + counts) (equal-count bins, ties in file order);
                         per decile and model: the peaks, the observed range and mean, the mean residual (predicted
                         minus observed log counts), the mean absolute residual and the RMSE
  residuals_by_decile_summary.tsv
                         per decile, over datasets: each model's median RMSE and mean residual, the median RMSE
                         difference (Cherimoya minus ChromBPNet) with its bootstrap CI, the Wilcoxon p over datasets
                         and the datasets where Cherimoya's RMSE is lower
  scatter_<measure>.pdf/.png
                         plot_comparison.py for each measure: ChromBPNet on x, Cherimoya on y, one dot per fold,
                         each dataset in its Okabe-Ito colour and marker
  differences.pdf/.png   per measure, the fold differences by dataset (the dataset mean as a bar)
  residuals_by_decile.pdf/.png
                         each model's mean residual by decile (one line per dataset) and the RMSE difference

Every fold of each dataset's Cherimoya config must have its compare.tsv and compare_counts.npz; otherwise this exits
3 (waiting) and writes nothing, unless --allow-missing, which compares the folds there are. A done marker
(<out-dir>/.done/performance.json) records the inputs; a rerun with the same inputs and settings does nothing
(--force runs anyway).

    python workflows/cherimoya/compare/compare_performance.py --datasets <list.txt> [--cheri-root <root>]
        [--comparison-root <dir> | --out-dir <dir>] [--n-boot 10000] [--seed 0] [--force] [--allow-missing]

Runs in the pixi cherimoya environment (numpy, scipy, pandas, matplotlib), inside a box.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parent.parent)]

import common  # noqa: E402
import make_jobs  # noqa: E402
import plot_comparison  # noqa: E402
import summarize  # noqa: E402

logger = logging.getLogger("compare_performance")

FORMAT = 1
MEASURES = summarize.MEASURES
HIGHER_IS_BETTER = summarize.HIGHER_IS_BETTER
N_DECILES = 10


# ---------------------------------------------------------------------------------------------------------------------
# inputs


def fold_inputs(cheri_root: Path, datasets: list[str]) -> tuple[dict, list[str]]:
    """{(dataset, fold): {compare_tsv, counts_npz, fit_time}} for every fold of each dataset's config, and what is
    missing."""
    folds, missing = {}, []
    for ds in datasets:
        config = cheri_root / "configs" / ds / "config.yaml"
        try:
            ks = make_jobs.read_config(config)["folds"]
        except (OSError, ValueError) as exc:
            missing.append(f"{ds}: no readable {config} ({exc})")
            continue
        for k in ks:
            d = cheri_root / ds / f"fold_{k}"
            entry = {
                "compare_tsv": d / "compare.tsv",
                "counts_npz": d / "compare_counts.npz",
                "fit_time": d / "fit.time",
            }
            gone = [p.name for role, p in entry.items() if role != "fit_time" and not p.is_file()]
            if gone:
                missing.append(f"{ds} fold {k}: no {', '.join(gone)}")
                continue
            folds[(ds, str(k))] = entry
    return folds, missing


# ---------------------------------------------------------------------------------------------------------------------
# statistics


def measure_statistics(paired: list[dict], n_boot: int, seed: int) -> list[dict]:
    """One row per measure (see the module docstring)."""
    rows = []
    for m in MEASURES:
        sign = 1 if HIGHER_IS_BETTER[m] else -1
        by_ds: dict[str, list[float]] = defaultdict(list)
        for p in paired:
            by_ds[p["dataset"]].append(p[f"{m}_diff"])
        diffs = np.array([p[f"{m}_diff"] for p in paired], dtype=np.float64)
        ds_means = np.array([np.mean(v) for _, v in sorted(by_ds.items())], dtype=np.float64)
        est, lo, hi = common.bootstrap_ci(ds_means, np.median, n_boot=n_boot, seed=seed)
        w_f, p_f, n_f = common.wilcoxon(diffs)
        w_d, p_d, n_d = common.wilcoxon(ds_means)
        rows.append({
            "measure": m,
            "higher_is_better": HIGHER_IS_BETTER[m],
            "n_folds": len(paired),
            "n_datasets": len(by_ds),
            "chrombpnet_median": float(np.median([p[f"{m}_chrombpnet"] for p in paired])),
            "cherimoya_median": float(np.median([p[f"{m}_cherimoya"] for p in paired])),
            "median_fold_diff": float(np.median(diffs)),
            "median_dataset_diff": est,
            "ci95_lo": lo,
            "ci95_hi": hi,
            "mean_dataset_diff": float(np.mean(ds_means)),
            "wilcoxon_folds_stat": w_f,
            "wilcoxon_folds_p": p_f,
            "wilcoxon_datasets_stat": w_d,
            "wilcoxon_datasets_p": p_d,
            "cherimoya_better_folds": int((sign * diffs > 0).sum()),
            "chrombpnet_better_folds": int((sign * diffs < 0).sum()),
            "cherimoya_better_datasets": int((sign * ds_means > 0).sum()),
            "chrombpnet_better_datasets": int((sign * ds_means < 0).sum()),
        })  # fmt: skip
    return rows


def statistics_text(stats: list[dict], n_boot: int) -> str:
    if not stats:
        return "no fold has both models scored\n"
    s0 = stats[0]
    lines = [
        f"{s0['n_folds']} fold(s) over {s0['n_datasets']} dataset(s); difference = Cherimoya - ChromBPNet.",
        f"Dataset difference: the median over datasets of each dataset's mean fold difference, with a percentile "
        f"bootstrap 95 % CI ({n_boot} resamples of the datasets).",
        "Wilcoxon signed-rank, two-sided, over the fold differences and over the dataset means.",
        "",
        f"{'measure':<17}{'ChromBPNet':>11}{'Cherimoya':>11}{'dataset diff [95% CI]':>32}"
        f"{'p (folds)':>11}{'p (datasets)':>14}  better on, folds / datasets (Cherimoya-ChromBPNet)",
    ]
    for s in stats:
        ci = f"{s['median_dataset_diff']:+.4f} [{s['ci95_lo']:+.4f}, {s['ci95_hi']:+.4f}]"
        lines.append(
            f"{s['measure']:<17}{s['chrombpnet_median']:>11.4f}{s['cherimoya_median']:>11.4f}{ci:>32}"
            f"{s['wilcoxon_folds_p']:>11.2g}{s['wilcoxon_datasets_p']:>14.2g}  "
            f"{s['cherimoya_better_folds']}-{s['chrombpnet_better_folds']} / "
            f"{s['cherimoya_better_datasets']}-{s['chrombpnet_better_datasets']}"
        )
    return "\n".join(lines) + "\n"


def deciles(values: np.ndarray, n: int = N_DECILES) -> np.ndarray:
    """1..n: equal-count bins of the values in ascending order (ties in input order)."""
    order = np.argsort(values, kind="stable")
    out = np.empty(values.size, dtype=np.int64)
    out[order] = np.arange(values.size) * n // max(values.size, 1) + 1
    return out


def residual_rows(folds: dict, datasets: list[str]) -> list[dict]:
    """Per dataset (its folds' test peaks pooled), per decile of observed log counts and model."""
    rows = []
    for ds in datasets:
        parts = [np.load(folds[key]["counts_npz"], allow_pickle=False) for key in sorted(folds) if key[0] == ds]
        if not parts:
            continue
        observed = np.concatenate([p["observed"] for p in parts]).astype(np.float64)
        pred = {m: np.concatenate([p[m] for p in parts]).astype(np.float64) for m in common.MODELS}
        dec = deciles(observed)
        for d in range(1, N_DECILES + 1):
            sel = dec == d
            obs = observed[sel]
            for m in common.MODELS:
                res = pred[m][sel] - obs
                rows.append({
                    "dataset": ds, "decile": d, "model": m, "n_peaks": int(sel.sum()),
                    "observed_min": float(obs.min()), "observed_max": float(obs.max()),
                    "observed_mean": float(obs.mean()), "mean_residual": float(res.mean()),
                    "mean_abs_residual": float(np.abs(res).mean()), "rmse": float(np.sqrt((res**2).mean())),
                })  # fmt: skip
    return rows


def residual_summary(rows: list[dict], n_boot: int, seed: int) -> list[dict]:
    import pandas as pd

    df = pd.DataFrame(rows)
    out = []
    for d, part in df.groupby("decile", sort=True):
        wide = part.pivot(index="dataset", columns="model", values=["rmse", "mean_residual"])
        diff = (wide[("rmse", "cherimoya")] - wide[("rmse", "chrombpnet")]).to_numpy()
        est, lo, hi = common.bootstrap_ci(diff, np.median, n_boot=n_boot, seed=seed)
        _, p, _ = common.wilcoxon(diff)
        out.append({
            "decile": int(d), "n_datasets": int(len(wide)),
            "rmse_chrombpnet_median": float(wide[("rmse", "chrombpnet")].median()),
            "rmse_cherimoya_median": float(wide[("rmse", "cherimoya")].median()),
            "rmse_diff_median": est, "ci95_lo": lo, "ci95_hi": hi, "wilcoxon_datasets_p": p,
            "cherimoya_lower_rmse_datasets": int((diff < 0).sum()),
            "mean_residual_chrombpnet_median": float(wide[("mean_residual", "chrombpnet")].median()),
            "mean_residual_cherimoya_median": float(wide[("mean_residual", "cherimoya")].median()),
        })  # fmt: skip
    return out


# ---------------------------------------------------------------------------------------------------------------------
# figures


def plot_differences(paired: list[dict], stats: list[dict], out: Path) -> list[Path]:
    plt = common.matplotlib_pyplot()
    datasets = sorted({p["dataset"] for p in paired})
    styles, labels = common.dataset_styles(datasets), common.short_names(datasets)
    by = {s["measure"]: s for s in stats}
    fig, axes = plt.subplots(1, len(MEASURES), figsize=(3.0 * len(MEASURES), 3.6), squeeze=False)
    for ax, m in zip(axes[0], MEASURES):
        ax.axhline(0, color="#999999", lw=0.8, zorder=0)
        for i, ds in enumerate(datasets):
            colour, marker = styles[ds]
            y = np.array([p[f"{m}_diff"] for p in paired if p["dataset"] == ds])
            x = i + np.linspace(-0.18, 0.18, y.size) if y.size > 1 else np.array([i])
            ax.scatter(x, y, s=16, color=colour, marker=marker, edgecolor="#333333", linewidth=0.4, zorder=2)
            ax.plot([i - 0.3, i + 0.3], [y.mean(), y.mean()], color="#333333", lw=1.2, zorder=3)
        s = by[m]
        better = "higher" if HIGHER_IS_BETTER[m] else "lower"
        ax.set_title(f"{plot_comparison.LABELS[m]}\n{s['median_dataset_diff']:+.3f} "
                     f"[{s['ci95_lo']:+.3f}, {s['ci95_hi']:+.3f}], p = {s['wilcoxon_datasets_p']:.2g} (datasets)",
                     fontsize=7.5)  # fmt: skip
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels([labels[d] for d in datasets], rotation=90, fontsize=6)
        ax.set_ylabel(f"Cherimoya - ChromBPNet ({better} is better)", fontsize=7)
        ax.grid(axis="x", visible=False)
    common.dataset_legend(fig, styles, labels, f"{len(datasets)} datasets")
    fig.suptitle("Paired differences per fold (dots) and per dataset (bars), on ChromBPNet's test peaks", fontsize=9)
    fig.tight_layout()
    paths = common.save_figure(fig, out)
    plt.close(fig)
    return paths


def plot_residuals(rows: list[dict], out: Path) -> list[Path]:
    import pandas as pd

    plt = common.matplotlib_pyplot()
    df = pd.DataFrame(rows)
    datasets = sorted(df["dataset"].unique())
    styles, labels = common.dataset_styles(datasets), common.short_names(datasets)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), squeeze=False)
    ax_cbp, ax_che, ax_diff = axes[0]
    for ax, m in ((ax_cbp, "chrombpnet"), (ax_che, "cherimoya")):
        ax.axhline(0, color="#999999", lw=0.8, zorder=0)
        for ds in datasets:
            colour, marker = styles[ds]
            part = df[(df["dataset"] == ds) & (df["model"] == m)].sort_values("decile")
            ax.plot(part["decile"], part["mean_residual"], color=colour, marker=marker, lw=1.0, ms=4,
                    markeredgecolor="#333333", markeredgewidth=0.3)  # fmt: skip
        ax.set_title(f"{common.NAMES[m]}: mean residual by observed-count decile", fontsize=8)
        ax.set_xlabel("decile of observed log(1 + counts), pooled test peaks")
        ax.set_ylabel("predicted - observed log counts")
    lo = min(ax_cbp.get_ylim()[0], ax_che.get_ylim()[0])
    hi = max(ax_cbp.get_ylim()[1], ax_che.get_ylim()[1])
    for ax in (ax_cbp, ax_che):
        ax.set_ylim(lo, hi)
        ax.set_xticks(range(1, N_DECILES + 1))
    wide = df.pivot_table(index=["dataset", "decile"], columns="model", values="rmse").reset_index()
    wide["diff"] = wide["cherimoya"] - wide["chrombpnet"]
    ax_diff.axhline(0, color="#999999", lw=0.8, zorder=0)
    for ds in datasets:
        colour, marker = styles[ds]
        part = wide[wide["dataset"] == ds].sort_values("decile")
        ax_diff.scatter(part["decile"], part["diff"], s=14, color=colour, marker=marker, edgecolor="#333333",
                        linewidth=0.3, zorder=2)  # fmt: skip
    med = wide.groupby("decile")["diff"].median()
    ax_diff.plot(med.index, med.to_numpy(), color="#333333", lw=1.2, zorder=3, label="median over datasets")
    ax_diff.set_xticks(range(1, N_DECILES + 1))
    ax_diff.set_title("RMSE, Cherimoya - ChromBPNet (lower is better)", fontsize=8)
    ax_diff.set_xlabel("decile of observed log(1 + counts), pooled test peaks")
    ax_diff.set_ylabel("RMSE difference (log counts)")
    ax_diff.legend(fontsize=6.5, frameon=False, loc="best")
    common.dataset_legend(fig, styles, labels, f"{len(datasets)} datasets")
    fig.tight_layout()
    paths = common.save_figure(fig, out)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    import pandas as pd

    datasets = common.read_list(args.datasets)
    out_dir = args.out_dir or args.comparison_root / "performance"
    folds, missing = fold_inputs(args.cheri_root, datasets)
    if missing:
        for line in missing:
            logger.warning("missing: %s", line)
        if not args.allow_missing or not folds:
            common.result(f"waiting: {len(missing)} fold(s) without their comparison, e.g. {missing[0]}")
            return common.WAITING
    inputs = {"list": args.datasets}
    for (ds, k), entry in folds.items():
        for role, path in entry.items():
            inputs[f"{ds}/{k}/{role}"] = path
    settings = {"datasets": datasets, "n_boot": args.n_boot, "seed": args.seed, "deciles": N_DECILES,
                "allow_missing": bool(args.allow_missing)}  # fmt: skip
    marker = common.Marker(out_dir, "performance", FORMAT, inputs, settings)
    why = marker.current()
    if why is None and not args.force:
        logger.info("up to date (%s)", marker.path)
        common.result("up to date")
        return 0
    logger.info("level 1 for %d dataset(s), %d fold(s): %s", len(datasets), len(folds), why or "forced")
    marker.remove()
    out_dir.mkdir(parents=True, exist_ok=True)

    rc = summarize.main(["--root", str(args.cheri_root), "--out-dir", str(out_dir), "--datasets", str(args.datasets)])
    if rc != 0:
        raise common.Failed(f"summarize.py exited {rc}")
    rows = [r for r in summarize.read_rows(args.cheri_root, datasets) if (r["dataset"], r["fold"]) in folds]
    paired = summarize.pair(rows)
    unpaired = sorted(set(folds) - {(p["dataset"], p["fold"]) for p in paired})
    if unpaired:
        raise common.Failed(f"{len(unpaired)} fold(s) whose compare.tsv lacks a model's row, e.g. {unpaired[0]}")
    stats = measure_statistics(paired, args.n_boot, args.seed)
    common.write_table(out_dir / "statistics.tsv", pd.DataFrame(stats))
    text = statistics_text(stats, args.n_boot)
    common.write_text(out_dir / "statistics.txt", text)
    sys.stdout.write(text)

    res = residual_rows(folds, datasets)
    common.write_table(out_dir / "residuals_by_decile.tsv", pd.DataFrame(res))
    res_summary = residual_summary(res, args.n_boot, args.seed)
    common.write_table(out_dir / "residuals_by_decile_summary.tsv", pd.DataFrame(res_summary))

    outputs = {name: out_dir / name for name in ("comparison.tsv", "comparison_paired.tsv", "comparison_summary.txt",
                                                  "statistics.tsv", "statistics.txt", "residuals_by_decile.tsv",
                                                  "residuals_by_decile_summary.tsv")}  # fmt: skip
    paired_path = out_dir / "comparison_paired.tsv"
    for m in MEASURES:
        rc = plot_comparison.main(["--paired", str(paired_path), "--out", str(out_dir / f"scatter_{m}"),
                                   "--measures", m, "--strip-prefix", strip_prefix(datasets)])  # fmt: skip
        common.matplotlib_pyplot().close("all")  # plot_comparison.py leaves its figure open
        if rc != 0:
            raise common.Failed(f"plot_comparison.py exited {rc} for {m}")
        outputs[f"scatter_{m}.png"] = out_dir / f"scatter_{m}.png"
    for path in plot_differences(paired, stats, out_dir / "differences"):
        outputs[path.name] = path
    for path in plot_residuals(res, out_dir / "residuals_by_decile"):
        outputs[path.name] = path

    pear = next(s for s in stats if s["measure"] == "counts_pearson")
    jsd = next(s for s in stats if s["measure"] == "median_jsd")
    note = (f"{len(paired)} folds, {len({p['dataset'] for p in paired})} datasets: counts Pearson "
            f"{pear['chrombpnet_median']:.3f} -> {pear['cherimoya_median']:.3f} "
            f"(dataset diff {pear['median_dataset_diff']:+.3f}, p {pear['wilcoxon_datasets_p']:.2g}); median JSD "
            f"{jsd['chrombpnet_median']:.3f} -> {jsd['cherimoya_median']:.3f}")  # fmt: skip
    if missing:
        note += f"; {len(missing)} fold(s) missing"
    marker.write(outputs, note)
    common.result(note)
    return 0


def strip_prefix(datasets: list[str]) -> str:
    """The prefix short_names() drops, for plot_comparison.py's --strip-prefix."""
    ds = sorted(datasets)[0]
    return ds[: len(ds) - len(common.short_names(datasets)[ds])]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_root_args(ap)
    ap.add_argument("--datasets", type=Path, required=True, help="the datasets, one per line (e.g. shared10.txt)")
    ap.add_argument("--out-dir", type=Path, help="default <comparison-root>/performance")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="run even when the done marker is current")
    ap.add_argument("--allow-missing", action="store_true", help="compare the folds there are")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    common.setup_logging(args.verbose)
    common.require(args, "cheri_root")
    if args.out_dir is None:
        common.require(args, "comparison_root")
    try:
        return run(args)
    except common.Waiting as exc:
        common.result(f"waiting: {exc}")
        return common.WAITING
    except common.Failed as exc:
        logger.error("%s", exc)
        common.result(f"failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
