#!/usr/bin/env python3
"""The comparison's report: one page that reads levels 1-5 and says what they found.

Reads the levels' outputs under --comparison-root (performance/, attributions/, modisco/, compendium/, finemo/,
bias_control/), each fold's comparison under --cheri-root and the compendia under --compendia-root, computes the
statistics the levels leave out, and writes, under --out-dir (default <comparison-root>/report):

  index.html        the report page: the findings with their key numbers, one section per level with figures and
                    compact tables, the bias control, the caveats, and the methods with each input's time (ET) and
                    the code's commits. Self-contained (figures embedded as PNG data URIs, fonts from Google Fonts),
                    ready to publish as a private Artifact page
  statistics.tsv    every interval and test on the page, one row each: level, family, quantity, test (interval or
                    wilcoxon), over (datasets or folds), n, estimate (the median of the values, or of the paired
                    differences), ci95_lo and ci95_hi (percentile bootstrap, --n-boot resamples, --seed; datasets are
                    resampled, and for rows over folds whole datasets with their folds unless the note says
                    otherwise), statistic and p (two-sided Wilcoxon signed-rank), higher and lower (how many
                    datasets or folds the difference is above or below 0), note
  tables/*.tsv      the tables behind the numbers on the page (level<l>_*.tsv, bias_*.tsv, inputs.tsv, checks.tsv)
  figures/*.pdf/.png
                    the page's figures (Okabe-Ito blue for ChromBPNet and vermillion for Cherimoya in every panel;
                    quantities across the two models in ink)

What it computes that the levels do not:
  level 1   all datasets x 5 folds (and the shared and the other datasets apart): compare_performance.py's paired
            statistics (the median of the dataset means' differences with a bootstrap CI, Wilcoxon over the folds
            and over the dataset means, wins) and its residuals by observed-count decile
  level 2   the held-out contrast between the models (heldout_cherimoya - heldout_chrombpnet); the strata
            (signalValue decile, GC decile, the chromosome's role) across datasets with CIs, and tests of decile
            10 against decile 1 and of test against training chromosomes; top-50 Jaccard from per-peak means (the
            medians are quantized); each model's attribution magnitude (median per-peak L2 norm and sum)
  level 3   CIs on each model's own shares; tests of seqlet overlap (fold averages, and QC folds), of the overlap
            with matched partners, and of the Tn5/DNase-flagged patterns (fold averages and QC folds)
  bias      the bias control split by what the reference pattern is: the bias-included model's patterns are
            annotated here with level 3's rules (best MotifCompendium match; flagged when the best match in
            chrombpnet's motifs.meme is a Tn5 or DNase I entry at q <= 0.05)
  level 4   per dataset, the share of each model's patterns and seqlets in clusters holding both models'
            patterns, at the joint compendium's threshold and at the sensitivity threshold; the annotations of
            the reciprocal like-for-like pairs
  level 5   hits per 10,000 shared peaks per model; precision and recall by cluster class and by sign (the
            sign from joint_clusters.tsv); per-motif medians; the largest motifs; 11.0's motif reports side by side
The tests levels 2, 3 and 5 already ran are computed again from their tables (for the statistic as well as p) and
checked against the levels' statistics.tsv (tables/checks.tsv).

Pending parts are shown as pending, never as failures: folds without both models' comparison yet (level 1), the
supplement compendium (--supplement-tag; level 4's supplement.json) and the production hits (level 5's
finemo/<ds>/production/). A part that cannot be computed is shown on the page with its reason, everything else is
still written, and the script exits 1. Each run writes the page again; .done/report.json records what it read.

    python workflows/cherimoya/compare/report.py [--comparison-root <dir>] [--out-dir <dir>] [--cheri-root <dir>]
        [--compendia-root <dir>] [--datasets <shared list>] [--pending-eta "Saturday 2026-10-10"]
        [--repo <name>=<checkout> ...] [--n-boot 10000] [--seed 0] [--dpi 144]

The roots default to the environment compare.sh exports (COMPARISON_ROOT, CHERI_ROOT, COMPENDIA_ROOT), else to the
siblings of --comparison-root (<parent>/cherimoya, <parent>/compendia). --datasets defaults to
<compendia-root>/inputs/shared10.txt; level 1 reads it and <compendia-root>/inputs/rest29.txt (else every
<cheri-root>/configs/<dataset>/). Runs in the pixi cherimoya environment (numpy, scipy, pandas, matplotlib, h5py,
memelite), inside a box: about two minutes on 4 cores.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import html
import io
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parent.parent)]

import common  # noqa: E402
import compare_attributions  # noqa: E402
import compare_modisco  # noqa: E402
import compare_performance  # noqa: E402
import make_jobs  # noqa: E402
import motifs  # noqa: E402
import summarize  # noqa: E402

logger = logging.getLogger("report")

FORMAT = 1
MODELS = common.MODELS
NAMES = common.NAMES
MEASURES = summarize.MEASURES
HIGHER_IS_BETTER = summarize.HIGHER_IS_BETTER
MEASURE_NAMES = {
    "counts_pearson": "Counts Pearson r",
    "counts_spearman": "Counts Spearman ρ",
    "counts_mse": "Counts MSE (log counts)",
    "median_jsd": "Profile median JSD",
    "median_norm_jsd": "Profile median normalized JSD",
}
MEASURE_SHORT = {
    "counts_pearson": "counts Pearson r",
    "counts_spearman": "counts Spearman ρ",
    "counts_mse": "counts MSE",
    "median_jsd": "median JSD",
    "median_norm_jsd": "median normalized JSD",
}
COLOURS = {"chrombpnet": "#0072B2", "cherimoya": "#D55E00", "cross": "#3B3F45"}
"""Okabe-Ito blue (ChromBPNet) and vermillion (Cherimoya) in every figure; quantities across the models in ink."""
TN5 = "#E69F00"
"""Okabe-Ito orange: a Tn5/DNase bias pattern (bias control only)."""
INK, INK2, MUTED, PALE, GRID = "#16191D", "#3C434B", "#68717B", "#C3CAD1", "#E6EAEE"
SET_ORDER = ("all", "shared", "others")
CROSS_ORDER = (
    ("cross_fold", "A", "same\nfold", "cross"),
    ("cross_average", "B", "fold\naverages", "cross"),
    ("loo_chrombpnet", "C", "ChromBPNet\nceiling", "chrombpnet"),
    ("loo_cherimoya", "C", "Cherimoya\nceiling", "cherimoya"),
    ("cross_heldout", "D", "held-out\nfold", "cross"),
)
QUANTITY_NAMES = {
    "cross_fold": "A · cross-model, same fold",
    "cross_average": "B · cross-model, fold averages",
    "loo_chrombpnet": "C · ChromBPNet ceiling (leave one out)",
    "loo_cherimoya": "C · Cherimoya ceiling (leave one out)",
    "cross_heldout": "D · cross-model, held-out fold",
    "cross_train": "D · cross-model, training folds",
    "heldout_chrombpnet": "D · ChromBPNet, held-out fold",
    "train_chrombpnet": "D · ChromBPNet, training folds",
    "heldout_cherimoya": "D · Cherimoya, held-out fold",
    "train_cherimoya": "D · Cherimoya, training folds",
}
L2_TESTS = compare_attributions.PAIRED_TESTS + (
    ("heldout_cherimoya", "heldout_chrombpnet"),
    ("train_cherimoya", "train_chrombpnet"),
)
L2_METRICS = ("pearson", "spearman", "cosine", "jaccard_mean")
METRIC_NAMES = {
    "pearson": "Pearson r",
    "spearman": "Spearman ρ",
    "cosine": "Cosine",
    "jaccard": "Top-50 Jaccard (median)",
    "jaccard_mean": "Top-50 Jaccard (mean)",
}
CLASSES = ("shared", "chrombpnet_only", "cherimoya_only")
CLASS_NAMES = {"shared": "both models", "chrombpnet_only": "ChromBPNet only", "cherimoya_only": "Cherimoya only"}
CLASS_COLOURS = {"shared": COLOURS["cross"], "chrombpnet_only": COLOURS["chrombpnet"],
                 "cherimoya_only": COLOURS["cherimoya"]}  # fmt: skip
NAN = float("nan")


# ---------------------------------------------------------------------------------------------------------------------
# small helpers


def eastern() -> dt.tzinfo:
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo("America/New_York")
    except Exception:  # noqa: BLE001 - no tz database: EDT (the cluster's CEST minus 6 h)
        return dt.timezone(dt.timedelta(hours=-4))


ET = eastern()


def et(when) -> str:
    """A time (epoch seconds, or an ISO string with its offset) as 'Fri 2026-10-09 16:56 ET'."""
    if when is None:
        return "NA"
    try:
        if isinstance(when, str):
            t = dt.datetime.strptime(when, "%Y-%m-%dT%H:%M:%S%z") if "T" in when else None
            if t is None:
                return when
        else:
            t = dt.datetime.fromtimestamp(float(when), tz=dt.UTC)
    except (ValueError, OSError, OverflowError):
        return str(when)
    return t.astimezone(ET).strftime("%a %Y-%m-%d %H:%M ET")


def finite(values) -> np.ndarray:
    x = np.asarray(list(values), dtype=np.float64)
    return x[np.isfinite(x)]


def read_tsv(path: Path, **kw):
    import pandas as pd

    return pd.read_csv(path, sep="\t", keep_default_na=False, na_values=["NA", "", "nan", "NaN"], **kw)


def as_bool(series) -> np.ndarray:
    return compare_modisco.as_bool(series)


def isnum(x) -> bool:
    try:
        return x is not None and bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def png_size(data: bytes) -> tuple[int, int]:
    return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")


def git_info(path: Path) -> dict:
    def run(*cmd: str) -> str:
        try:
            out = subprocess.run(["git", "-C", str(path), *cmd], capture_output=True, text=True, timeout=20,
                                 check=False)  # fmt: skip
        except (OSError, subprocess.SubprocessError):
            return ""
        return out.stdout.strip()

    commit = run("rev-parse", "--short", "HEAD")
    return {
        "path": str(path),
        "commit": commit or None,
        "branch": run("branch", "--show-current") or None,
        "dirty": bool(run("status", "--porcelain", "--untracked-files=no")) if commit else None,
        "subject": run("log", "-1", "--format=%s") or None,
        "date": run("log", "-1", "--format=%cI") or None,
    }


# ---------------------------------------------------------------------------------------------------------------------
# statistics


class Stats:
    """Every interval and test of the page: the rows of statistics.tsv."""

    COLUMNS = ("level", "family", "quantity", "test", "over", "n", "estimate", "ci95_lo", "ci95_hi", "statistic", "p",
               "higher", "lower", "note")  # fmt: skip

    def __init__(self, n_boot: int, seed: int) -> None:
        self.n_boot, self.seed = n_boot, seed
        self.rows: list[dict] = []

    def ci(self, values) -> tuple[float, float, float]:
        return common.bootstrap_ci(finite(values), np.median, n_boot=self.n_boot, seed=self.seed)

    def cluster_ci(self, values, groups) -> tuple[float, float, float]:
        """The median of the values and its percentile bootstrap interval, the groups (datasets) resampled with all
        their values (folds)."""
        v = np.asarray(values, dtype=np.float64)
        g = np.asarray(groups, dtype=object)
        ok = np.isfinite(v)
        v, g = v[ok], g[ok]
        if v.size == 0:
            return (NAN, NAN, NAN)
        est = float(np.median(v))
        names = sorted(set(g))
        if len(names) < 2:
            return (est, NAN, NAN)
        width = max(int((g == n).sum()) for n in names)
        table = np.full((len(names), width), np.nan)
        for i, n in enumerate(names):
            x = v[g == n]
            table[i, : x.size] = x
        rng = np.random.default_rng(self.seed)
        draws = table[rng.integers(0, len(names), size=(self.n_boot, len(names)))].reshape(self.n_boot, -1)
        lo, hi = np.quantile(np.nanmedian(draws, axis=1), [0.025, 0.975])
        return (est, float(lo), float(hi))

    def add(self, **row) -> dict:
        full = {c: row.get(c, NAN) for c in self.COLUMNS}
        full["note"] = row.get("note", "")
        self.rows.append(full)
        return full

    def level(self, level: str, family: str, quantity: str, values, over: str = "datasets", note: str = "") -> dict:
        v = finite(values)
        est, lo, hi = self.ci(v)
        return self.add(level=level, family=family, quantity=quantity, test="interval", over=over, n=int(v.size),
                        estimate=est, ci95_lo=lo, ci95_hi=hi, note=note)  # fmt: skip

    def paired(self, level: str, family: str, quantity: str, diffs, over: str = "datasets", groups=None,
               note: str = "") -> dict:  # fmt: skip
        d = np.asarray(list(diffs), dtype=np.float64)
        if groups is not None:
            g = np.asarray(list(groups), dtype=object)
            ok = np.isfinite(d)
            d, g = d[ok], g[ok]
            est, lo, hi = self.cluster_ci(d, g)
        else:
            d = d[np.isfinite(d)]
            est, lo, hi = self.ci(d)
        w, p, _ = common.wilcoxon(d)
        return self.add(level=level, family=family, quantity=quantity, test="wilcoxon", over=over, n=int(d.size),
                        estimate=est, ci95_lo=lo, ci95_hi=hi, statistic=w, p=p, higher=int((d > 0).sum()),
                        lower=int((d < 0).sum()), note=note)  # fmt: skip

    def find(self, level: str, family: str, quantity: str, over: str = "datasets") -> dict | None:
        for r in reversed(self.rows):
            if r["level"] == level and r["family"] == family and r["quantity"] == quantity and r["over"] == over:
                return r
        return None


# ---------------------------------------------------------------------------------------------------------------------
# the run


class Report:
    """What the parts compute and the page shows: statistics, tables, figures, inputs, pending and failed parts."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root: Path = args.comparison_root
        self.out: Path = args.out_dir
        self.stats = Stats(args.n_boot, args.seed)
        self.tables: dict[str, object] = {}
        self.figures: dict[str, dict] = {}
        self.inputs: dict[str, Path] = {}
        self.seen: dict[str, dict] = {}
        self.problems: list[tuple[str, str]] = []
        self.pending: list[dict] = []
        self.checks: list[dict] = []
        self.shared = common.read_list(args.datasets)
        self.everything = self._everything()
        self.labels = common.short_names(self.everything)

    def _everything(self) -> list[str]:
        a = self.args
        lists = a.all_datasets or [p for p in (a.datasets, a.compendia_root / "inputs" / "rest29.txt") if p.is_file()]
        names: list[str] = []
        for path in lists:
            for ds in common.read_list(path):
                if ds not in names:
                    names.append(ds)
        if len(lists) < 2 and not a.all_datasets:
            configs = sorted(p.parent.name for p in (a.cheri_root / "configs").glob("*/config.yaml"))
            names += [ds for ds in configs if ds not in names]
        return names

    def use(self, role: str, path: Path) -> Path:
        path = Path(path)
        self.inputs[role] = path
        self.seen[role] = common.stat_entry(path)
        return path

    def table(self, name: str, df):
        self.tables[name] = df
        return df

    def check(self, what: str, pairs: list[tuple]) -> None:
        """Recomputed values against a level's own: the largest absolute difference (NaN on both sides agree)."""
        diffs, mismatched = [], 0
        for mine, theirs in pairs:
            a, b = isnum(mine), isnum(theirs)
            if a and b:
                diffs.append(abs(float(mine) - float(theirs)))
            elif a != b:
                mismatched += 1
        row = {"what": what, "values": len(pairs), "max_abs_diff": max(diffs) if diffs else NAN,
               "nan_mismatch": mismatched}  # fmt: skip
        self.checks.append(row)
        level = logging.WARNING if mismatched or (diffs and max(diffs) > 1e-4) else logging.INFO
        logger.log(level, "check %s: %d values, max |diff| %s, %d NaN mismatches", what, len(pairs),
                   f"{row['max_abs_diff']:.3g}", mismatched)  # fmt: skip

    def figure(self, fig, name: str, alt: str) -> None:
        """figures/<name>.pdf/.png (200 dpi) and the page's embedded PNG (--dpi)."""
        import matplotlib.pyplot as plt

        common.save_figure(fig, self.out / "figures" / name)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=self.args.dpi, bbox_inches="tight", facecolor="white",
                    pil_kwargs={"optimize": True})  # fmt: skip
        plt.close(fig)
        data = buf.getvalue()
        w, h = png_size(data)
        self.figures[name] = {"b64": base64.b64encode(data).decode("ascii"), "w": w, "h": h, "alt": alt}

    def part(self, name: str, fn, *args):
        try:
            return fn(self, *args)
        except Exception as exc:  # noqa: BLE001 - shown on the page; the script exits 1
            logger.exception("%s could not be computed", name)
            self.problems.append((name, f"{type(exc).__name__}: {exc}"))
            return None


def pyplot():
    plt = common.matplotlib_pyplot()
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlepad": 8, "axes.labelsize": 8.5, "axes.labelcolor": INK2,
        "axes.edgecolor": "#9AA2AA", "axes.linewidth": 0.7, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "xtick.labelsize": 7.8, "ytick.labelsize": 7.8, "xtick.color": INK2,
        "ytick.color": INK2, "xtick.major.size": 3, "ytick.major.size": 3, "text.color": INK,
        "legend.frameon": False, "legend.fontsize": 7.8, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })  # fmt: skip
    return plt


def median_bar(ax, x: float, values, rep: Report, colour: str = INK, half: float = 0.28) -> None:
    """A median bar at x and the bootstrap 95 % CI as a thin line beside it."""
    est, lo, hi = rep.stats.ci(values)
    if not np.isfinite(est):
        return
    ax.plot([x - half, x + half], [est, est], color=colour, lw=1.8, zorder=4, solid_capstyle="butt")
    if np.isfinite(lo):
        ax.plot([x + half + 0.06] * 2, [lo, hi], color=colour, lw=1.1, zorder=4, solid_capstyle="butt")


def strip(ax, x: float, values, colour: str, spread: float = 0.17, hollow: bool = False, size: float = 18) -> None:
    ys = finite(values)
    if ys.size == 0:
        return
    xs = x + (np.linspace(-spread, spread, ys.size) if ys.size > 1 else np.zeros(1))
    if hollow:
        ax.scatter(xs, ys, s=size, facecolor="white", edgecolor=colour, linewidth=1.0, zorder=3)
    else:
        ax.scatter(xs, ys, s=size, color=colour, edgecolor="white", linewidth=0.5, zorder=3)


# ---------------------------------------------------------------------------------------------------------------------
# level 1: performance


def expected_folds(cheri_root: Path, datasets: list[str]) -> int:
    n = 0
    for ds in datasets:
        try:
            n += len(make_jobs.read_config(cheri_root / "configs" / ds / "config.yaml")["folds"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return n


def level1(rep: Report) -> dict:
    import pandas as pd

    a = rep.args
    folds, missing = compare_performance.fold_inputs(a.cheri_root, rep.everything)
    expected = expected_folds(a.cheri_root, rep.everything)
    for (ds, k), entry in sorted(folds.items()):
        rep.use(f"level1/{ds}/{k}/compare_tsv", entry["compare_tsv"])
        rep.use(f"level1/{ds}/{k}/counts_npz", entry["counts_npz"])
    if missing:
        rep.pending.append(
            {
                "part": "Level 1",
                "what": f"{len(missing)} of {expected or '?'} folds without both "
                "models' comparison yet; the numbers below cover the folds there are",
                "detail": missing,
            }
        )
    rows = [r for r in summarize.read_rows(a.cheri_root, rep.everything) if (r["dataset"], r["fold"]) in folds]
    paired = summarize.pair(rows)
    if not paired:
        return {"state": "pending", "expected": expected, "missing": missing, "n_folds": 0}
    pdf = pd.DataFrame(paired)
    pdf["n_regions"] = pd.to_numeric(pdf["n_regions"], errors="coerce")
    pdf.insert(2, "shared", pdf["dataset"].isin(rep.shared))
    rep.table("level1_paired", pdf)
    members = {"all": set(rep.everything), "shared": set(rep.shared), "others": set(rep.everything) - set(rep.shared)}

    measures = []
    for key in SET_ORDER:
        sub = [p for p in paired if p["dataset"] in members[key]]
        if not sub:
            continue
        names = sorted({p["dataset"] for p in sub})
        fam = f"performance, {len(names)} datasets ({key})"
        for s in compare_performance.measure_statistics(sub, a.n_boot, a.seed):
            m = s["measure"]
            means = np.array([np.mean([p[f"{m}_diff"] for p in sub if p["dataset"] == ds]) for ds in names])
            better = "higher" if HIGHER_IS_BETTER[m] else "lower"
            rep.stats.add(
                level="1",
                family=fam,
                quantity=f"{m}: Cherimoya - ChromBPNet, dataset means",
                test="wilcoxon",
                over="datasets",
                n=len(names),
                estimate=s["median_dataset_diff"],
                ci95_lo=s["ci95_lo"],
                ci95_hi=s["ci95_hi"],
                statistic=s["wilcoxon_datasets_stat"],
                p=s["wilcoxon_datasets_p"],
                higher=int((means > 0).sum()),
                lower=int((means < 0).sum()),
                note=f"{better} is better",
            )
            fold = rep.stats.paired("1", fam, f"{m}: Cherimoya - ChromBPNet, folds", [p[f"{m}_diff"] for p in sub],
                                    over="folds", groups=[p["dataset"] for p in sub],
                                    note=f"{better} is better; the CI resamples datasets with their folds")  # fmt: skip
            s.update(set=key, fold_ci95_lo=fold["ci95_lo"], fold_ci95_hi=fold["ci95_hi"],
                     chrombpnet_mean=float(np.mean([p[f"{m}_chrombpnet"] for p in sub])),
                     cherimoya_mean=float(np.mean([p[f"{m}_cherimoya"] for p in sub])))  # fmt: skip
            measures.append(s)
    mdf = rep.table("level1_measures", pd.DataFrame(measures))

    agg = {"n_folds": ("fold", "size"), "test_peaks_mean": ("n_regions", "mean")}
    for m in MEASURES:
        for x in ("chrombpnet", "cherimoya", "diff"):
            agg[f"{m}_{x}"] = (f"{m}_{x}", "mean")
    per_ds = pdf.groupby("dataset", sort=True).agg(**agg).reset_index()
    per_ds.insert(1, "shared", per_ds["dataset"].isin(rep.shared))
    rep.table("level1_per_dataset", per_ds)

    res = rep.table("level1_residuals", pd.DataFrame(compare_performance.residual_rows(folds, rep.everything)))
    rsum = []
    for key in SET_ORDER:
        part = res[res["dataset"].isin(members[key])]
        if part.empty:
            continue
        fam = f"residuals, {part['dataset'].nunique()} datasets ({key})"
        for d, g in part.groupby("decile", sort=True):
            rm = g.pivot(index="dataset", columns="model", values="rmse")
            mr = g.pivot(index="dataset", columns="model", values="mean_residual")
            diff = (rm["cherimoya"] - rm["chrombpnet"]).to_numpy()
            row = rep.stats.paired(
                "1", fam, f"decile {int(d)}: RMSE Cherimoya - ChromBPNet", diff, note="lower is better"
            )
            obs = g[g["model"] == MODELS[0]]
            rsum.append({
                "set": key, "decile": int(d), "n_datasets": int(len(rm)),
                "observed_min_median": float(obs["observed_min"].median()),
                "observed_max_median": float(obs["observed_max"].median()),
                "observed_mean_median": float(obs["observed_mean"].median()),
                "rmse_chrombpnet_median": float(rm["chrombpnet"].median()),
                "rmse_cherimoya_median": float(rm["cherimoya"].median()),
                "rmse_diff_median": row["estimate"], "ci95_lo": row["ci95_lo"], "ci95_hi": row["ci95_hi"],
                "wilcoxon_stat": row["statistic"], "wilcoxon_p": row["p"],
                "cherimoya_lower_rmse_datasets": int((diff < 0).sum()),
                "mean_residual_chrombpnet_median": float(mr["chrombpnet"].median()),
                "mean_residual_cherimoya_median": float(mr["cherimoya"].median()),
            })  # fmt: skip
    rdf = rep.table("level1_residuals_summary", pd.DataFrame(rsum))

    # the level's own statistics of the shared datasets, computed from the same folds
    perf = rep.root / "performance"
    if (perf / "statistics.tsv").is_file() and "shared" in set(mdf["set"]):
        theirs = read_tsv(rep.use("level1/statistics", perf / "statistics.tsv"))
        mine = mdf[mdf["set"] == "shared"].set_index("measure")
        pairs = [
            (mine.loc[t["measure"], c], t[c])
            for _, t in theirs.iterrows()
            if t["measure"] in mine.index
            for c in (
                "chrombpnet_median",
                "cherimoya_median",
                "median_fold_diff",
                "median_dataset_diff",
                "ci95_lo",
                "ci95_hi",
                "wilcoxon_folds_p",
                "wilcoxon_datasets_p",
            )
        ]
        rep.check("level 1, shared datasets: performance/statistics.tsv", pairs)
    if (perf / "residuals_by_decile_summary.tsv").is_file() and "shared" in set(rdf["set"]):
        theirs = read_tsv(rep.use("level1/residuals_summary", perf / "residuals_by_decile_summary.tsv"))
        mine = rdf[rdf["set"] == "shared"].set_index("decile")
        pairs = [
            (mine.loc[int(t["decile"]), c1], t[c2])
            for _, t in theirs.iterrows()
            if int(t["decile"]) in mine.index
            for c1, c2 in (
                ("rmse_diff_median", "rmse_diff_median"),
                ("ci95_lo", "ci95_lo"),
                ("ci95_hi", "ci95_hi"),
                ("wilcoxon_p", "wilcoxon_datasets_p"),
            )
        ]
        rep.check("level 1, shared datasets: performance/residuals_by_decile_summary.tsv", pairs)

    plot_level1_scatter(rep, pdf)
    plot_level1_residuals(rep, res, rdf)
    return {"state": "done", "expected": expected, "missing": missing, "n_folds": len(paired),
            "n_datasets": pdf["dataset"].nunique(), "measures": mdf, "per_dataset": per_ds, "residuals": rdf,
            "paired": pdf}  # fmt: skip


def plot_level1_scatter(rep: Report, pdf) -> None:
    plt = pyplot()
    measures = ("counts_pearson", "counts_mse", "median_jsd")
    fig, axes = plt.subplots(1, len(measures), figsize=(10.6, 4.0), squeeze=False)
    shared = pdf["shared"].to_numpy(dtype=bool)
    n_sh, n_ot = pdf.loc[shared, "dataset"].nunique(), pdf.loc[~shared, "dataset"].nunique()
    for ax, m in zip(axes[0], measures):
        x, y = pdf[f"{m}_chrombpnet"].to_numpy(float), pdf[f"{m}_cherimoya"].to_numpy(float)
        lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
        pad = 0.06 * (hi - lo or 1.0)
        lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.9, zorder=1)
        if (~shared).any():
            ax.scatter(x[~shared], y[~shared], s=15, facecolor="white", edgecolor=INK2, linewidth=0.7, zorder=2,
                       label=f"folds of the {n_ot} other datasets")  # fmt: skip
        if shared.any():
            ax.scatter(x[shared], y[shared], s=15, color=INK, edgecolor="white", linewidth=0.4, zorder=3,
                       label=f"folds of the {n_sh} shared datasets")  # fmt: skip
        better = int(((y > x) if HIGHER_IS_BETTER[m] else (y < x)).sum())
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_title(f"{MEASURE_NAMES[m]}\nCherimoya better on {better} of {len(x)} folds")
        ax.set_xlabel("ChromBPNet")
        ax.set_ylabel("Cherimoya")
        if HIGHER_IS_BETTER[m]:
            ax.text(0.04, 0.96, "Cherimoya better", transform=ax.transAxes, ha="left", va="top", fontsize=7.5,
                    color=MUTED)  # fmt: skip
        else:
            ax.text(0.96, 0.04, "Cherimoya better", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
                    color=MUTED)  # fmt: skip
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    rep.figure(fig, "level1_scatter", "Three scatter plots of ChromBPNet against Cherimoya per test fold for counts "
               "Pearson r, counts MSE and profile median JSD, with the diagonal.")  # fmt: skip


def plot_level1_residuals(rep: Report, res, rdf) -> None:
    plt = pyplot()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 3.7))
    for m in MODELS:
        g = res[res["model"] == m].groupby("decile")["mean_residual"]
        med, q1, q3 = g.median(), g.quantile(0.25), g.quantile(0.75)
        ax1.fill_between(med.index, q1.to_numpy(), q3.to_numpy(), color=COLOURS[m], alpha=0.15, lw=0)
        ax1.plot(med.index, med.to_numpy(), color=COLOURS[m], lw=2, marker="o", ms=3.8, label=NAMES[m])
    ax1.axhline(0, color=MUTED, lw=0.8, zorder=1)
    ax1.set_title("Mean residual by decile of observed counts")
    ax1.set_xlabel("decile of observed log(1 + counts), test peaks of all folds")
    ax1.set_ylabel("predicted − observed log counts")
    ax1.set_xticks(range(1, 11))
    ax1.legend(loc="lower left")
    wide = res.pivot_table(index=["dataset", "decile"], columns="model", values="rmse").reset_index()
    wide["diff"] = wide["cherimoya"] - wide["chrombpnet"]
    for _, g in wide.groupby("dataset"):
        ax2.plot(g["decile"], g["diff"], color=PALE, lw=0.7, zorder=1)
    s = rdf[rdf["set"] == "all"].sort_values("decile")
    ax2.fill_between(s["decile"], s["ci95_lo"], s["ci95_hi"], color=INK, alpha=0.14, lw=0, zorder=2)
    ax2.plot(s["decile"], s["rmse_diff_median"], color=INK, lw=2, marker="o", ms=3.8, zorder=3,
             label=f"median over {int(s['n_datasets'].max())} datasets, 95% CI")  # fmt: skip
    ax2.axhline(0, color=MUTED, lw=0.8, zorder=1)
    ax2.set_title("RMSE, Cherimoya − ChromBPNet (below 0: Cherimoya closer)")
    ax2.set_xlabel("decile of observed log(1 + counts), test peaks of all folds")
    ax2.set_ylabel("RMSE difference (log counts)")
    ax2.set_xticks(range(1, 11))
    ax2.plot([], [], color=PALE, lw=0.9, label="one dataset")
    ax2.legend(loc="lower center")
    fig.tight_layout()
    rep.figure(fig, "level1_residuals", "Left: mean residual of each model by observed-count decile. Right: RMSE "
               "difference by decile, one grey line per dataset and the median with its confidence band.")  # fmt: skip


# ---------------------------------------------------------------------------------------------------------------------
# level 2: per-peak attributions


def level2(rep: Report) -> dict:
    import pandas as pd

    d = rep.root / "attributions"
    if not (d / "summary.tsv").is_file():
        rep.pending.append({"part": "Level 2", "what": "no summary yet (compare.sh 2 writes it)", "detail": []})
        return {"state": "pending"}
    summary = read_tsv(rep.use("level2/summary", d / "summary.tsv"), dtype={"level": str})
    info = {}
    for ds in rep.shared:
        p = d / ds / "dataset.json"
        if p.is_file() and (d / ".done" / f"{ds}.json").is_file():
            info[ds] = json.loads(rep.use(f"level2/{ds}/dataset", p).read_text())
    missing = [ds for ds in rep.shared if ds not in info]
    if missing:
        rep.pending.append({"part": "Level 2", "what": f"{len(missing)} of {len(rep.shared)} datasets not compared "
                            "yet", "detail": missing})  # fmt: skip
    if not info:
        return {"state": "pending"}
    summary = summary[summary["dataset"].isin(list(info))].copy()
    folds_by_ds = {ds: [str(k) for k in info[ds]["folds"]] for ds in info}
    q = quantities(summary, folds_by_ds)
    windows = sorted({w for w, _ in q}, key=lambda w: int(w[1:]))
    long = [
        {"window": w, "metric": m, "quantity": name, "dataset": ds, "value": v}
        for (w, m), qd in sorted(q.items())
        for name, vals in sorted(qd.items())
        for ds, v in sorted(vals.items())
    ]
    rep.table("level2_quantities", pd.DataFrame(long))

    for (w, m), qd in sorted(q.items()):
        fam = f"attributions, {w}, {m}"
        note = "means of the per-peak top-50 Jaccard (its medians are quantized)" if m == "jaccard_mean" else ""
        for name in sorted(qd):
            rep.stats.level("2", fam, name, list(qd[name].values()), note=note)
        for x, y in L2_TESTS:
            if x in qd and y in qd:
                both = sorted(set(qd[x]) & set(qd[y]))
                rep.stats.paired("2", fam, f"{x} - {y}", [qd[x][ds] - qd[y][ds] for ds in both], note=note)
    fold_rows = compare_attributions.fold_level_rows(summary)
    for (w, m), part in fold_rows.groupby(["window", "measure"]):
        wide = part.pivot_table(index=["dataset", "fold"], columns="kind", values="median")
        for model in MODELS:
            if "cross" not in wide or model not in wide:
                continue
            diff = (wide["cross"] - wide[model]).dropna()
            rep.stats.paired("2", f"attributions, {w}, {m}", f"cross_fold - loo_{model}", diff.to_numpy(), over="folds",
                             groups=diff.index.get_level_values("dataset"),
                             note="the CI resamples datasets with their folds")  # fmt: skip
    if (d / "statistics.tsv").is_file():
        theirs = read_tsv(rep.use("level2/statistics", d / "statistics.tsv"))
        pairs = []
        for _, t in theirs.iterrows():
            r = rep.stats.find("2", f"attributions, {t['window']}, {t['metric']}", t["quantity"], t["over"])
            if r is None:
                continue
            pairs += [(r["estimate"], t["median"]), (r["p"], t["wilcoxon_p"])]
            if t["over"] == "datasets":
                pairs += [(r["ci95_lo"], t["ci95_lo"]), (r["ci95_hi"], t["ci95_hi"])]
        rep.check("level 2: attributions/statistics.tsv (fold rows: median and p only)", pairs)

    # strata: each stratum level in place of 'all', through the level's own per-dataset quantities
    srows = []
    for (stratum, lev), part in summary[summary["stratum"] != "all"].groupby(["stratum", "level"], sort=False):
        for (w, m), qd in quantities(part.assign(stratum="all"), folds_by_ds).items():
            for name, vals in qd.items():
                for ds, v in vals.items():
                    srows.append({"stratum": stratum, "level": str(lev), "window": w, "metric": m, "quantity": name,
                                  "dataset": ds, "value": v})  # fmt: skip
    sdf = rep.table("level2_strata_per_dataset", pd.DataFrame(srows))
    ssum = []
    for key, g in sdf.groupby(["stratum", "level", "window", "metric", "quantity"], sort=False):
        est, lo, hi = rep.stats.ci(g["value"])
        ssum.append(
            dict(
                zip(("stratum", "level", "window", "metric", "quantity"), key),
                n=int(g["value"].notna().sum()),
                median=est,
                ci95_lo=lo,
                ci95_hi=hi,
            )
        )
    ssum = pd.DataFrame(ssum)
    ssum["order"] = pd.to_numeric(ssum["level"], errors="coerce").fillna(
        ssum["level"].map({"train": 1, "valid": 2, "test": 3}))  # fmt: skip
    ssum = rep.table("level2_strata", ssum.sort_values(["stratum", "window", "metric", "quantity", "order"])
                     .drop(columns="order").reset_index(drop=True))  # fmt: skip
    value = {k: dict(zip(g["dataset"], g["value"])) for k, g in sdf.groupby(["stratum", "level", "window", "metric",
                                                                                "quantity"])}  # fmt: skip
    for w in windows:
        for m in L2_METRICS:
            fam = f"attribution strata, {w}, {m}"
            for stratum in ("signal_decile", "gc_decile"):
                for name in ("cross_fold", "cross_average", "loo_chrombpnet", "loo_cherimoya"):
                    top, bottom = value.get((stratum, "10", w, m, name), {}), value.get((stratum, "1", w, m, name), {})
                    both = sorted(set(top) & set(bottom))
                    if both:
                        rep.stats.paired("2", fam, f"{stratum} 10 - 1: {name}", [top[ds] - bottom[ds] for ds in both])
            for name in ("cross_fold", "loo_chrombpnet", "loo_cherimoya"):
                test, train = value.get(("role", "test", w, m, name), {}), value.get(("role", "train", w, m, name), {})
                both = sorted(set(test) & set(train))
                if both:
                    rep.stats.paired("2", fam, f"role test - train: {name}", [test[ds] - train[ds] for ds in both])

    # magnitudes: each model's median per-peak L2 norm and sum of the projected score
    s_all = summary[summary["stratum"] == "all"]
    mags = []
    for ds in sorted(info):
        part = s_all[s_all["dataset"] == ds]
        for w in windows:
            pw = part[part["window"] == w]
            med = dict(zip(zip(pw["comparison"], pw["measure"]), pw["median"]))
            ks = folds_by_ds[ds]
            row = {"dataset": ds, "window": w}
            for m in MODELS:
                for kind in ("l2", "sum"):
                    row[f"average_{kind}_{m}"] = med.get(("average", f"{kind}_{m}"), NAN)
                    row[f"fold_{kind}_{m}"] = float(np.mean([med.get((f"fold_{k}", f"{kind}_{m}"), NAN) for k in ks]))
                row[f"loo_l2_fold_{m}"] = float(np.mean([med.get((f"loo_{m}_{k}", "l2_fold"), NAN) for k in ks]))
                row[f"loo_l2_others_{m}"] = float(np.mean([med.get((f"loo_{m}_{k}", "l2_others"), NAN) for k in ks]))
            row["average_l2_ratio"] = row["average_l2_cherimoya"] / row["average_l2_chrombpnet"]
            mags.append(row)
    mdf = rep.table("level2_magnitude", pd.DataFrame(mags))
    for w in windows:
        part = mdf[mdf["window"] == w]
        for m in MODELS:
            for col, what in ((f"average_l2_{m}", "median per-peak L2 norm, fold average"),
                              (f"average_sum_{m}", "median per-peak sum, fold average"),
                              (f"fold_l2_{m}", "median per-peak L2 norm, single folds (mean over folds)")):  # fmt: skip
                rep.stats.level("2", f"attribution magnitude, {w}", f"{NAMES[m]}: {what}", part[col])
            rep.stats.level(
                "2",
                f"attribution magnitude, {w}",
                f"{NAMES[m]}: L2 norm of the mean of 4 folds / of one fold",
                part[f"loo_l2_others_{m}"] / part[f"loo_l2_fold_{m}"],
            )
        rep.stats.level("2", f"attribution magnitude, {w}", "L2 norm of the fold averages, Cherimoya / ChromBPNet",
                        part["average_l2_ratio"], note="the two models' attribution scales differ")  # fmt: skip

    joins = rep.table("level2_joins", pd.DataFrame([{
        "dataset": ds, "compared_peaks": info[ds].get("compared_rows"), "chrombpnet_only_peaks":
        info[ds].get("chrombpnet_only_rows"), "v1_rows_checked_per_fold": info[ds].get("v1_rows_checked_per_fold"),
        "peaks_without_heldout_fold": info[ds].get("heldout_rows_without_test_fold"), "folds": len(folds_by_ds[ds])}
        for ds in sorted(info)]))  # fmt: skip
    f16 = None
    if (d / "float16.tsv").is_file():
        f16 = read_tsv(rep.use("level2/float16", d / "float16.tsv"))
        rep.table("level2_float16", f16)

    plot_level2_agreement(rep, q, windows[0])
    plot_level2_strata(rep, ssum, windows[0])
    return {
        "state": "done",
        "q": q,
        "windows": windows,
        "strata": ssum,
        "magnitude": mdf,
        "joins": joins,
        "float16": f16,
        "datasets": sorted(info),
        "missing": missing,
    }


def quantities(summary, folds_by_ds: dict) -> dict:
    """compare_attributions.per_dataset_quantities, plus top-50 Jaccard from the per-peak means ('jaccard_mean')."""
    q = compare_attributions.per_dataset_quantities(summary, folds_by_ds)
    jac = summary[summary["measure"].isin(("jaccard", "heldout_jaccard", "train_jaccard"))]
    if not jac.empty:
        for (w, _), v in compare_attributions.per_dataset_quantities(
            jac.assign(median=jac["mean"]), folds_by_ds
        ).items():
            q[(w, "jaccard_mean")] = v
    return q


def plot_level2_agreement(rep: Report, q: dict, window: str) -> None:
    plt = pyplot()
    panels = [m for m in L2_METRICS if (window, m) in q]
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.0), squeeze=False)
    for ax, metric in zip(axes.flat, panels):
        qd = q[(window, metric)]
        for x, (name, _, _, kind) in enumerate(CROSS_ORDER):
            vals = list((qd.get(name) or {}).values())
            strip(ax, x, vals, COLOURS[kind])
            median_bar(ax, x, vals, rep)
        ax.set_xticks(range(len(CROSS_ORDER)), [f"{letter}\n{desc}" for _, letter, desc, _ in CROSS_ORDER])
        ax.set_title(f"{METRIC_NAMES[metric]}, {window[1:]} bp around the summit")
        ax.set_ylabel("mean over peaks" if metric == "jaccard_mean" else "median over peaks")
        ax.grid(axis="x", visible=False)
        ax.set_xlim(-0.6, len(CROSS_ORDER) - 0.35)
    for ax in list(axes.flat)[len(panels) :]:
        ax.set_visible(False)
    fig.tight_layout(h_pad=2.2)
    rep.figure(
        fig,
        "level2_agreement",
        f"Four panels, one per agreement metric at {window[1:]} bp: per dataset, "
        "cross-model agreement on single folds (A), on fold averages (B), each model's leave-one-out ceiling "
        "(C) and cross-model agreement on held-out folds (D), with medians and confidence intervals.",
    )


def plot_level2_strata(rep: Report, ssum, window: str, metric: str = "pearson") -> None:
    plt = pyplot()
    lines = (("cross_fold", "A  cross-model, same fold", "cross", (0, (4, 2)), "white"),
             ("cross_average", "B  cross-model, fold averages", "cross", "-", None),
             ("loo_chrombpnet", "C  ChromBPNet ceiling", "chrombpnet", "-", None),
             ("loo_cherimoya", "C  Cherimoya ceiling", "cherimoya", "-", None))  # fmt: skip
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1), sharey=True, squeeze=False)
    titles = {"signal_decile": "signalValue decile", "gc_decile": "GC decile"}
    for ax, (stratum, xlabel) in zip(
        axes[0],
        (
            ("signal_decile", "decile of the peak's signalValue (narrowPeak column 7)"),
            ("gc_decile", "decile of the GC content of the 1,000-bp window"),
        ),
    ):
        for name, label, kind, ls, face in lines:
            part = ssum[(ssum["stratum"] == stratum) & (ssum["window"] == window) & (ssum["metric"] == metric)
                        & (ssum["quantity"] == name)].copy()  # fmt: skip
            if part.empty:
                continue
            part["x"] = part["level"].astype(int)
            part = part.sort_values("x")
            ax.fill_between(part["x"], part["ci95_lo"], part["ci95_hi"], color=COLOURS[kind], alpha=0.12, lw=0)
            ax.plot(part["x"], part["median"], color=COLOURS[kind], ls=ls, lw=1.7, marker="o", ms=3.6,
                    markerfacecolor=face or COLOURS[kind], label=label)  # fmt: skip
        ax.set_xticks(range(1, 11))
        ax.set_xlabel(xlabel)
        ax.set_title(f"{METRIC_NAMES[metric]}, {window[1:]}-bp window, by {titles[stratum]}")
    axes[0][0].set_ylabel("median over datasets, 95% CI")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    rep.figure(fig, "level2_strata", "Two line charts of attribution agreement by signal decile and by GC decile, for "
               "cross-model single folds, fold averages and both models' ceilings.")  # fmt: skip


# ---------------------------------------------------------------------------------------------------------------------
# level 3: TF-MoDISco


L3_PAIRED = (
    ("shared_cluster_count", "shared by cluster, by count: Cherimoya - ChromBPNet"),
    ("shared_cluster_seqlets", "shared by cluster, by seqlets: Cherimoya - ChromBPNet"),
    ("shared_tomtom_count", "shared by tomtom, by count: Cherimoya - ChromBPNet"),
    ("shared_tomtom_seqlets", "shared by tomtom, by seqlets: Cherimoya - ChromBPNet"),
    ("seqlet_overlap", "seqlets overlapped by the other model's seqlets: Cherimoya - ChromBPNet"),
    ("overlap_matched", "seqlets overlapped by matched partners' seqlets: Cherimoya - ChromBPNet"),
    ("bias_flagged", "Tn5/DNase-flagged patterns: Cherimoya - ChromBPNet"),
    ("bias_share", "share of patterns Tn5/DNase-flagged: Cherimoya - ChromBPNet"),
    ("bias_seqlet_share", "share of seqlets in Tn5/DNase-flagged patterns: Cherimoya - ChromBPNet"),
    ("qc_cross", "QC: patterns matched across models: Cherimoya - ChromBPNet"),
    ("qc_seqlet_overlap", "QC: seqlets overlapped across models: Cherimoya - ChromBPNet"),
    ("qc_bias_per_fold", "QC: Tn5/DNase-flagged patterns per fold: Cherimoya - ChromBPNet"),
)
L3_LEVELS = ("patterns", "seqlets", "shared_cluster_count", "shared_cluster_seqlets", "shared_tomtom_count",
             "shared_tomtom_seqlets", "seqlet_overlap", "overlap_matched", "bias_flagged", "bias_share", "qc_cross",
             "qc_within", "qc_seqlet_overlap", "qc_patterns_per_fold", "qc_bias_per_fold")  # fmt: skip


def level3(rep: Report) -> dict:
    import pandas as pd

    d = rep.root / "modisco"
    if not (d / "summary.tsv").is_file() or not (d / "qc_pairs.tsv").is_file():
        rep.pending.append({"part": "Level 3", "what": "no summary yet (compare.sh 3 writes it)", "detail": []})
        return {"state": "pending"}
    summary = read_tsv(rep.use("level3/summary", d / "summary.tsv"))
    summary = summary[summary["dataset"].isin(rep.shared)].reset_index(drop=True)
    missing = [ds for ds in rep.shared if ds not in set(summary["dataset"])]
    if missing:
        rep.pending.append({"part": "Level 3", "what": f"{len(missing)} of {len(rep.shared)} datasets not compared "
                            "yet", "detail": missing})  # fmt: skip
    if summary.empty:
        return {"state": "pending"}
    qc = read_tsv(rep.use("level3/qc_pairs", d / "qc_pairs.tsv"), dtype={"fold_a": str, "fold_b": str})
    qc = qc[qc["dataset"].isin(rep.shared)]
    pats, qcp = [], []
    for ds in summary["dataset"]:
        pats.append(read_tsv(rep.use(f"level3/{ds}/patterns", d / ds / "patterns.tsv")))
        if (d / ds / "qc_patterns.tsv").is_file():
            qcp.append(read_tsv(rep.use(f"level3/{ds}/qc_patterns", d / ds / "qc_patterns.tsv"), dtype={"fold": str}))
    patterns = pd.concat(pats, ignore_index=True)
    for col in ("bias_flag", "matched", "matched_cluster", "matched_tomtom"):
        patterns[col] = as_bool(patterns[col])
    qc_patterns = (
        pd.concat(qcp, ignore_index=True)
        if qcp
        else pd.DataFrame(columns=["dataset", "fold", "model", "pattern", "bias_flag"])
    )
    qc_patterns["bias_flag"] = as_bool(qc_patterns["bias_flag"])

    rows = []
    for _, r in summary.iterrows():
        ds = r["dataset"]
        for m in MODELS:
            part = patterns[(patterns["dataset"] == ds) & (patterns["model"] == m)]
            on = pd.to_numeric(part["seqlets_on_shared_peaks"], errors="coerce").to_numpy(float)
            om = pd.to_numeric(part["seqlet_overlap_matched"], errors="coerce").to_numpy(float)
            ok = np.isfinite(on) & np.isfinite(om)
            qq = qc_patterns[(qc_patterns["dataset"] == ds) & (qc_patterns["model"] == m)]
            per_fold = qq.groupby("fold").agg(n=("pattern", "size"), flagged=("bias_flag", "sum"))
            row = {
                "dataset": ds,
                "model": m,
                "patterns": int(r[f"{m}_pos"] + r[f"{m}_neg"]),
                "pos": int(r[f"{m}_pos"]),
                "neg": int(r[f"{m}_neg"]),
                "seqlets": int(r[f"{m}_seqlets"]),
                "bias_flagged": int(r[f"{m}_bias_flagged"]),
                "bias_seqlets": int(part.loc[part["bias_flag"], "n_seqlets"].sum()),
                "shared_cluster_count": r[f"{m}_shared_cluster_count"],
                "shared_cluster_seqlets": r[f"{m}_shared_cluster_seqlets"],
                "shared_tomtom_count": r[f"{m}_shared_tomtom_count"],
                "shared_tomtom_seqlets": r[f"{m}_shared_tomtom_seqlets"],
                "seqlet_overlap": r[f"{m}_seqlet_overlap"],
                "overlap_matched": float((om[ok] * on[ok]).sum() / on[ok].sum()) if on[ok].sum() else NAN,
                "cluster_unmatched": int((~part["matched_cluster"]).sum()),
                "tomtom_unmatched": int((~part["matched_tomtom"]).sum()),
                "cluster_unmatched_with_tomtom": int((~part["matched_cluster"] & part["matched_tomtom"]).sum()),
                "qc_cross": r.get(f"qc_cross_{m}", NAN),
                "qc_within": r.get(f"qc_within_{m}", NAN),
                "qc_seqlet_overlap": r.get(f"qc_seqlet_overlap_{m}", NAN),
                "qc_folds": int(len(per_fold)),
                "qc_patterns": int(len(qq)),
                "qc_bias_flagged": int(qq["bias_flag"].sum()),
                "qc_patterns_per_fold": float(per_fold["n"].mean()) if len(per_fold) else NAN,
                "qc_bias_per_fold": float(per_fold["flagged"].mean()) if len(per_fold) else NAN,
            }
            row["bias_share"] = row["bias_flagged"] / row["patterns"] if row["patterns"] else NAN
            row["bias_seqlet_share"] = row["bias_seqlets"] / row["seqlets"] if row["seqlets"] else NAN
            rows.append(row)
    per = rep.table("level3_per_dataset", pd.DataFrame(rows))
    wide = per.pivot(index="dataset", columns="model")

    for col in L3_LEVELS:
        for m in MODELS:
            rep.stats.level("3", "TF-MoDISco, per model", f"{NAMES[m]}: {col}", wide[(col, m)])
    for col, label in L3_PAIRED:
        rep.stats.paired(
            "3",
            "TF-MoDISco, Cherimoya - ChromBPNet",
            label,
            (wide[(col, "cherimoya")] - wide[(col, "chrombpnet")]).to_numpy(),
        )
    for m in MODELS:
        rep.stats.paired("3", "TF-MoDISco QC, cross-model - ceiling",
                         f"QC: {m} patterns matched across models - within {m} (ceiling)",
                         (wide[("qc_cross", m)] - wide[("qc_within", m)]).to_numpy())  # fmt: skip

    # per QC fold (dataset x fold): as compare_modisco.py's summary, and the seqlet overlap and bias flags
    qf = (
        qc_patterns.groupby(["dataset", "fold", "model"])
        .agg(patterns=("pattern", "size"), bias_flagged=("bias_flag", "sum"))
        .reset_index()
    )
    cross = qc[qc["kind"] == "cross"]
    fold_rows = []
    for m, col in (("chrombpnet", "matched_a"), ("cherimoya", "matched_b")):
        c = cross.set_index(["dataset", "fold_a"])[col]
        w = qc[qc["kind"] == f"within_{m}"]
        per_fold = pd.concat([w[["dataset", "fold_a", "matched_a"]].rename(columns={"matched_a": "v"}),
                              w[["dataset", "fold_b", "matched_b"]].rename(columns={"fold_b": "fold_a",
                                                                                     "matched_b": "v"})])  # fmt: skip
        within = per_fold.groupby(["dataset", "fold_a"])["v"].mean()
        diff = (c - within).dropna()
        rep.stats.paired("3", "TF-MoDISco QC, cross-model - ceiling",
                         f"QC: {m} patterns matched across models - within {m}, per fold", diff.to_numpy(),
                         over="folds", groups=diff.index.get_level_values("dataset"),
                         note="the CI resamples datasets with their folds")  # fmt: skip
        fold_rows.append(
            pd.DataFrame(
                {
                    "dataset": diff.index.get_level_values(0),
                    "fold": diff.index.get_level_values(1),
                    "model": m,
                    "cross": c.reindex(diff.index).to_numpy(),
                    "within": within.reindex(diff.index).to_numpy(),
                }
            )
        )
    if {"seqlet_overlap_a", "seqlet_overlap_b"} <= set(cross.columns):
        diff = (cross["seqlet_overlap_b"] - cross["seqlet_overlap_a"]).to_numpy(float)
        rep.stats.paired(
            "3",
            "TF-MoDISco QC, Cherimoya - ChromBPNet",
            "QC: seqlets overlapped across models, per fold",
            diff,
            over="folds",
            groups=cross["dataset"],
            note="the CI resamples datasets with their folds",
        )
    if not qf.empty:
        fw = qf.pivot_table(index=["dataset", "fold"], columns="model", values="bias_flagged").dropna()
        if set(MODELS) <= set(fw.columns):
            rep.stats.paired("3", "TF-MoDISco QC, Cherimoya - ChromBPNet",
                             "QC: Tn5/DNase-flagged patterns per fold", (fw["cherimoya"] - fw["chrombpnet"]).to_numpy(),
                             over="folds", groups=fw.index.get_level_values("dataset"),
                             note="the CI resamples datasets with their folds")  # fmt: skip
    rep.table("level3_qc_per_fold", qf)
    rep.table("level3_qc_ceilings_per_fold", pd.concat(fold_rows, ignore_index=True) if fold_rows else pd.DataFrame())

    if (d / "statistics.tsv").is_file():
        theirs = read_tsv(rep.use("level3/statistics", d / "statistics.tsv"))
        pairs = []
        for _, t in theirs.iterrows():
            r = next((x for x in reversed(rep.stats.rows) if x["level"] == "3" and x["quantity"] == t["comparison"]
                      and x["over"] == t["over"]), None)  # fmt: skip
            if r is None:
                continue
            pairs += [(r["estimate"], t["median_diff"]), (r["p"], t["wilcoxon_p"]), (r["higher"], t["first_higher"])]
            if t["over"] == "datasets":
                pairs += [(r["ci95_lo"], t["ci95_lo"]), (r["ci95_hi"], t["ci95_hi"])]
        rep.check("level 3: modisco/statistics.tsv (fold rows: median, p and counts only)", pairs)

    plot_level3(rep, per)
    return {"state": "done", "per": per, "wide": wide, "patterns": patterns, "qc_patterns": qc_patterns,
            "qc_folds": qf, "datasets": list(summary["dataset"]), "missing": missing}  # fmt: skip


def plot_level3(rep: Report, per) -> None:
    plt = pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.9), gridspec_kw={"width_ratios": [1.05, 1.15, 1.0]})
    ax1, ax2, ax3 = axes
    wide = per.pivot(index="dataset", columns="model")

    def paired_columns(ax, x0: float, col: str, gap: float = 0.36) -> list[float]:
        xs = [x0 - gap / 2, x0 + gap / 2]
        for ds in wide.index:
            ax.plot(xs, [wide.loc[ds, (col, MODELS[0])], wide.loc[ds, (col, MODELS[1])]], color=PALE, lw=0.7, zorder=1)
        for x, m in zip(xs, MODELS):
            strip(ax, x, wide[(col, m)], COLOURS[m], spread=0.0, size=20)
            median_bar(ax, x, wide[(col, m)], rep, half=0.13)
        return xs

    for i, col in enumerate(("shared_cluster_count", "shared_tomtom_count")):
        paired_columns(ax1, i, col)
    ax1.set_xticks([0, 1], ["same joint20\ncluster", "tomtom partner\n(q ≤ 0.05)"])
    ax1.set_ylim(0, 1.04)
    ax1.set_ylabel("share of the model's patterns")
    ax1.set_title("Fold-average patterns\nmatched in the other model")
    ax1.grid(axis="x", visible=False)

    for i, m in enumerate(MODELS):
        xs = [i * 1.5 - 0.3, i * 1.5 + 0.3]
        for ds in wide.index:
            ax2.plot(xs, [wide.loc[ds, ("qc_cross", m)], wide.loc[ds, ("qc_within", m)]], color=COLOURS[m], alpha=0.35,
                     lw=0.8, zorder=1)  # fmt: skip
        strip(ax2, xs[0], wide[("qc_cross", m)], COLOURS[m], spread=0.0, size=20)
        strip(ax2, xs[1], wide[("qc_within", m)], COLOURS[m], spread=0.0, size=20, hollow=True)
        median_bar(ax2, xs[0], wide[("qc_cross", m)], rep, half=0.12)
        median_bar(ax2, xs[1], wide[("qc_within", m)], rep, half=0.12)
    ax2.set_xticks([-0.3, 0.3, 1.2, 1.8], ["across\nthe models", "fold to fold\n(ceiling)"] * 2, fontsize=7.2)
    for i, m in enumerate(MODELS):
        ax2.text(i * 1.5, 1.0, NAMES[m], transform=ax2.get_xaxis_transform(), ha="center", va="bottom",
                 fontsize=7.8, fontweight="bold", color=INK2)  # fmt: skip
    ax2.set_xlim(-0.75, 2.25)
    ax2.set_ylabel("share of a fold's patterns matched")
    ax2.set_title("QC scale, per fold: across the models\nagainst each model's fold-to-fold ceiling", pad=18)
    ax2.grid(axis="x", visible=False)

    for i, col in enumerate(("seqlet_overlap", "overlap_matched", "qc_seqlet_overlap")):
        paired_columns(ax3, i, col)
    ax3.set_xticks([0, 1, 2], ["fold average,\nany seqlet", "fold average,\nmatched partners", "QC folds,\nany seqlet"],
                   fontsize=7.2)  # fmt: skip
    ax3.set_ylabel("share of the model's seqlets")
    ax3.set_title("Seqlets overlapped (≥ 50 %, reciprocal)\nby the other model's seqlets")
    ax3.set_ylim(0.5, 1.0)
    ax3.grid(axis="x", visible=False)
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], ls="", marker="o", color=COLOURS[m], markersize=6, label=NAMES[m]) for m in MODELS]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    rep.figure(fig, "level3_patterns", "Three panels: per dataset, the share of each model's fold-average patterns "
               "matched in the other model; the QC-scale cross-model match against each model's fold-to-fold ceiling; "
               "and the share of each model's seqlets overlapped by the other model's seqlets.")  # fmt: skip


# ---------------------------------------------------------------------------------------------------------------------
# bias control (level 3c)


def databases(rep: Report) -> dict:
    """The motif databases level 3 annotated with, from its done markers."""
    for ds in rep.shared:
        p = rep.root / "modisco" / ".done" / f"{ds}.average.json"
        try:
            inputs = json.loads(p.read_text())["inputs"]
            annot, bias = Path(inputs["motif_db"]["path"]), Path(inputs["bias_db"]["path"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if annot.is_file() and bias.is_file():
            return {"annot_path": rep.use("bias/motif_db", annot), "bias_path": rep.use("bias/bias_db", bias),
                    "annot": motifs.Database.read(annot), "bias": motifs.Database.read(bias)}  # fmt: skip
    raise FileNotFoundError("no level-3 done marker records the motif databases")


def bias_control(rep: Report, l3: dict | None) -> dict:
    import pandas as pd

    d = rep.root / "modisco"
    controls = []
    for status_path in sorted((rep.root / "bias_control").glob("*/status.json")):
        st = json.loads(rep.use(f"bias/{status_path.parent.name}/status", status_path).read_text())
        head = st.get("head", common.HEAD)
        h5 = status_path.parent / f"motif_qc_{head}" / "chrombpnet_modisco_results.h5"
        controls.append(
            {
                "name": status_path.parent.name,
                "state": st.get("state"),
                "head": head,
                "dataset": st.get("dataset"),
                "fold": st.get("fold"),
                "reason": st.get("reason"),
                "note": st.get("note"),
                "h5": h5 if st.get("state") == "done" and h5.is_file() else None,
            }
        )
    if not (d / "bias_control.tsv").is_file():
        return {"state": "pending", "controls": controls}
    bc = read_tsv(rep.use("bias/table", d / "bias_control.tsv"))
    bc["matched"] = as_bool(bc["matched"])
    bj = json.loads(rep.use("bias/json", d / "bias_control.json").read_text())
    refs = []
    if any(c["h5"] is not None for c in controls):
        dbs = databases(rep)
        for c in controls:
            if c["h5"] is None:
                continue
            pats, _ = motifs.read_modisco(rep.use(f"bias/{c['name']}/h5", c["h5"]), "reference", seqlets=False)
            for row in compare_modisco.annotate_patterns(pats, dbs["annot"], dbs["bias"]):
                row["reference"] = f"bias_control/{c['name']}"
                refs.append(row)
    ref = pd.DataFrame(refs)
    if not ref.empty:
        counts = (bc[bc["matched"]].groupby(["reference", "reference_match"])
                  .agg(cherimoya_patterns=("pattern", "size"), cherimoya_seqlets=("n_seqlets", "sum"),
                       datasets=("dataset", "nunique")).reset_index())  # fmt: skip
        ref = ref.merge(counts, left_on=["reference", "pattern"], right_on=["reference", "reference_match"],
                        how="left").drop(columns=["reference_match"])  # fmt: skip
        for col in ("cherimoya_patterns", "cherimoya_seqlets", "datasets"):
            ref[col] = ref[col].fillna(0).astype(int)
        ref["kind"] = np.where(ref["bias_flag"].astype(bool), "Tn5/DNase", "TF or other")
        kind = dict(zip(zip(ref["reference"], ref["pattern"]), ref["kind"]))
    else:
        kind = {}
    bc["reference_kind"] = [kind.get((r, p), "unknown") if mt else "no match"
                            for r, p, mt in zip(bc["reference"], bc["reference_match"], bc["matched"])]  # fmt: skip
    own = {}
    if l3 is not None:
        pt = l3["patterns"]
        pt = pt[pt["model"] == "cherimoya"]
        own.update({(ds, "fold average", p): f for ds, p, f in zip(pt["dataset"], pt["pattern"], pt["bias_flag"])})
        qp = l3["qc_patterns"]
        qp = qp[qp["model"] == "cherimoya"]
        own.update({(ds, f"QC fold {k}", p): f for ds, k, p, f in zip(qp["dataset"], qp["fold"], qp["pattern"],
                                                                      qp["bias_flag"])})  # fmt: skip
    bc["own_bias_flag"] = [own.get((ds, s, p)) for ds, s, p in zip(bc["dataset"], bc["scale"], bc["pattern"])]
    rep.table("bias_control_patterns", bc)
    rep.table("bias_reference_patterns", ref)
    split = (bc.groupby("reference_kind").agg(patterns=("pattern", "size"), seqlets=("n_seqlets", "sum"),
                                              db_flagged=("own_bias_flag", lambda s: int(sum(bool(x) for x in s
                                                                                             if x is not None))))
             .reset_index())  # fmt: skip
    rep.table("bias_control_split", split)
    if l3 is not None or not ref.empty:
        plot_bias(rep, l3, ref)
    return {"state": "done", "controls": controls, "table": bc, "json": bj, "reference": ref, "split": split}


def plot_bias(rep: Report, l3: dict | None, ref) -> None:
    plt = pyplot()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.0), gridspec_kw={"width_ratios": [0.8, 1.6]})
    if l3 is not None:
        wide = l3["wide"]
        for col, x0 in (("bias_flagged", 0.0), ("qc_bias_per_fold", 1.2)):
            xs = [x0 - 0.2, x0 + 0.2]
            for ds in wide.index:
                ax1.plot(xs, [wide.loc[ds, (col, MODELS[0])], wide.loc[ds, (col, MODELS[1])]], color=PALE, lw=0.7,
                         zorder=1)  # fmt: skip
            for x, m in zip(xs, MODELS):
                strip(ax1, x, wide[(col, m)], COLOURS[m], spread=0.0, size=20)
                median_bar(ax1, x, wide[(col, m)], rep, half=0.12)
        ax1.set_xticks([0, 1.2], ["fold averages", "QC folds\n(mean per fold)"])
        ax1.set_ylabel("Tn5/DNase-flagged patterns per dataset")
        ax1.set_title("Patterns whose best match in\nmotifs.meme is a Tn5/DNase entry")
        ax1.grid(axis="x", visible=False)
        from matplotlib.lines import Line2D

        ax1.legend(handles=[Line2D([], [], ls="", marker="o", color=COLOURS[m], markersize=6, label=NAMES[m])
                            for m in MODELS], loc="upper right")  # fmt: skip
    else:
        ax1.set_visible(False)
    if ref is not None and not ref.empty:
        r = ref.sort_values(["cherimoya_patterns", "n_seqlets"], ascending=[True, True]).reset_index(drop=True)
        colours = [TN5 if k == "Tn5/DNase" else "#8C959F" for k in r["kind"]]
        y = np.arange(len(r))
        ax2.barh(y, r["cherimoya_patterns"], color=colours, height=0.62, edgecolor="white", linewidth=0.8)
        labels = [f"{p}  {b if k == 'Tn5/DNase' else (a if isinstance(a, str) else 'no match')}"
                  for p, a, b, k in zip(r["pattern"], r["annotation"], r["bias_db_best"], r["kind"])]  # fmt: skip
        ax2.set_yticks(y, labels, fontsize=7.4)
        for yi, v in zip(y, r["cherimoya_patterns"]):
            ax2.text(v + 0.8, yi, str(int(v)), va="center", fontsize=7.4, color=INK2)
        ax2.set_xlabel("Cherimoya patterns without a cluster partner that match it")
        ax2.set_title("Reference: the bias-included ChromBPNet's patterns (profile head)")
        ax2.grid(axis="y", visible=False)
        ax2.set_xlim(0, max(1.0, float(r["cherimoya_patterns"].max())) * 1.15)
        from matplotlib.patches import Patch

        ax2.legend(handles=[Patch(color=TN5, label="Tn5/DNase pattern (motifs.meme, q ≤ 0.05)"),
                            Patch(color="#8C959F", label="TF or other pattern")], loc="lower right")  # fmt: skip
    else:
        ax2.set_visible(False)
    fig.tight_layout()
    rep.figure(fig, "bias_control", "Left: Tn5/DNase-flagged patterns per dataset for each model. Right: for each "
               "pattern of the bias-included ChromBPNet, how many unmatched Cherimoya patterns match it, coloured by "
               "whether the reference pattern is a Tn5/DNase pattern.")  # fmt: skip


# ---------------------------------------------------------------------------------------------------------------------
# level 4: compendia


def classify(df) -> np.ndarray:
    a = df["patterns_chrombpnet"].fillna(0).to_numpy() > 0
    b = df["patterns_cherimoya"].fillna(0).to_numpy() > 0
    return np.where(a & b, "shared", np.where(a, "chrombpnet_only", "cherimoya_only"))


def class_table(df, threshold: str) -> list[dict]:
    rows = []
    totals = {m: float(df[f"seqlets_{m}"].sum()) for m in MODELS}
    for c in CLASSES:
        part = df[df["class"] == c]
        row = {"threshold": threshold, "class": c, "clusters": int(len(part)),
               "single_pattern_clusters": int((part["patterns"] == 1).sum()),
               "annotated": int(part["annotation"].notna().sum())}  # fmt: skip
        for m in MODELS:
            row[f"patterns_{m}"] = int(part[f"patterns_{m}"].sum())
            row[f"seqlets_{m}"] = int(part[f"seqlets_{m}"].sum())
            row[f"seqlet_share_{m}"] = float(part[f"seqlets_{m}"].sum() / totals[m]) if totals[m] else NAN
        if "bias_share" in part:
            row["bias_clusters"] = int((part["bias_share"] >= 0.5).sum())
        rows.append(row)
    return rows


def level4(rep: Report) -> dict:
    import pandas as pd

    a = rep.args
    d = rep.root / "compendium"
    if not (d / "like_for_like.json").is_file():
        return {"state": "pending"}
    tags = {"cbp": a.cbp_tag, "cheri": a.cheri_tag, "joint": a.joint}
    supplement, sensitivity = a.supplement_tag, a.sensitivity_tag or f"{a.joint}_t090"
    try:
        settings = json.loads(rep.use("level4/marker", d / ".done" / "compendia.json").read_text())["settings"]
        tags.update(settings.get("tags") or {})
        supplement = settings.get("supplement") or supplement
        sensitivity = settings.get("sensitivity") or sensitivity
    except (OSError, ValueError, KeyError, TypeError):
        pass
    lfl = json.loads(rep.use("level4/like_for_like", d / "like_for_like.json").read_text())
    cl = {}
    for t in (tags["cbp"], tags["cheri"], tags["joint"], sensitivity):
        p = d / f"clusters_{t}.tsv"
        if p.is_file():
            cl[t] = read_tsv(rep.use(f"level4/clusters_{t}", p), dtype={"cluster_id": str})
    m_ab = read_tsv(rep.use("level4/matches", d / f"matches_{tags['cbp']}_vs_{tags['cheri']}.tsv"),
                    dtype={"cluster": str, "best": str})  # fmt: skip
    recip = m_ab[as_bool(m_ab["reciprocal"])].copy()
    cheri_ann = cl[tags["cheri"]].set_index("cluster_id")["annotation"] if tags["cheri"] in cl else pd.Series(dtype=str)
    recip["annotation_cherimoya"] = recip["best"].map(cheri_ann)
    recip = recip.rename(columns={"cluster": "cluster_chrombpnet", "best": "cluster_cherimoya",
                                  "annotation": "annotation_chrombpnet"})  # fmt: skip
    both = recip["annotation_chrombpnet"].notna() & recip["annotation_cherimoya"].notna()
    recip["agreement"] = np.where(
        ~both,
        "unannotated",
        np.where(
            recip["annotation_chrombpnet"] == recip["annotation_cherimoya"], "same annotation", "different annotation"
        ),
    )
    rep.table("level4_reciprocal_pairs", recip)
    agreement = {k: int(v) for k, v in recip["agreement"].value_counts().items()}
    by_sign = {k: int(v) for k, v in recip["posneg"].value_counts().items()}

    joint = read_tsv(rep.use("level4/joint_clusters", d / "joint_clusters.tsv"), dtype={"cluster_id": str})
    joint["class"] = classify(joint)
    classes = class_table(joint, "0.95")
    sens = cl.get(sensitivity)
    if sens is not None:
        sens = sens.copy()
        sens["class"] = classify(sens)
        classes += class_table(sens, "0.90")
    cdf = rep.table("level4_classes", pd.DataFrame(classes))

    # per dataset: the share of each model's patterns and seqlets in clusters holding both models' patterns
    shares = []
    for tag, frame, thr in ((tags["joint"], joint, "0.95"), (sensitivity, sens, "0.90")):
        meta = a.compendia_root / tag / "modisco_compendium_meta.tsv"
        if frame is None or not meta.is_file():
            continue
        mt = read_tsv(rep.use(f"level4/meta_{tag}", meta), dtype={"cluster_id": str})
        cls = dict(zip(frame["cluster_id"], frame["class"]))
        mt["class"] = mt["cluster_id"].map(cls)
        for label, part in mt.groupby("model"):
            prefix = next((p for p in compare_modisco.LABEL_PREFIX.items() if label.startswith(p[1])), None)
            if prefix is None:
                continue
            m, ds = prefix[0], label[len(prefix[1]) :]
            if ds not in rep.shared:
                continue
            sh = (part["class"] == "shared").to_numpy()
            seq = pd.to_numeric(part["num_seqlets"], errors="coerce").fillna(0).to_numpy(float)
            shares.append({"threshold": thr, "dataset": ds, "model": m, "patterns": int(len(part)),
                           "shared_patterns": float(sh.mean()), "shared_seqlets": float(seq[sh].sum() / seq.sum())
                           if seq.sum() else NAN})  # fmt: skip
    sdf = rep.table("level4_shares", pd.DataFrame(shares))
    for thr, part in sdf.groupby("threshold") if not sdf.empty else []:
        w = part.pivot(index="dataset", columns="model")
        fam = f"compendium shares, threshold {thr}"
        for col, what in (("shared_patterns", "patterns"), ("shared_seqlets", "seqlets")):
            for m in MODELS:
                rep.stats.level("4", fam, f"{NAMES[m]}: share of {what} in clusters of both models", w[(col, m)])
            rep.stats.paired("4", fam, f"share of {what} in clusters of both models: Cherimoya - ChromBPNet",
                             (w[(col, "cherimoya")] - w[(col, "chrombpnet")]).to_numpy())  # fmt: skip

    sup = None
    if (d / "supplement.json").is_file():
        sup = json.loads(rep.use("level4/supplement", d / "supplement.json").read_text())
    else:
        built = (a.compendia_root / supplement / "compendium.json").is_file()
        rep.pending.append({"part": "Level 4", "what": f"the {supplement} supplement (all of Cherimoya's datasets "
                            "against the shared ones)", "detail": [f"compendia/{supplement} is built; level 4 has not "
                            "run again yet" if built else f"compendia/{supplement} is not built yet"]})  # fmt: skip
    sens_json = json.loads(rep.use("level4/sensitivity", d / "sensitivity.json").read_text()) \
        if (d / "sensitivity.json").is_file() else None  # fmt: skip
    plot_level4(rep, cdf, sdf)
    return {"state": "done", "tags": tags, "supplement_tag": supplement, "sensitivity_tag": sensitivity, "lfl": lfl,
            "reciprocal": recip, "agreement": agreement, "by_sign": by_sign, "classes": cdf, "shares": sdf,
            "supplement": sup, "sensitivity": sens_json, "joint": joint}  # fmt: skip


def plot_level4(rep: Report, cdf, sdf) -> None:
    plt = pyplot()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 3.8), gridspec_kw={"width_ratios": [1.25, 1.0]})
    thresholds = [t for t in ("0.95", "0.90") if t in set(cdf["threshold"])]
    y = np.arange(len(CLASSES))
    height = 0.36
    for i, thr in enumerate(thresholds):
        part = cdf[cdf["threshold"] == thr].set_index("class").reindex(CLASSES)
        ys = y + (i - (len(thresholds) - 1) / 2) * (height + 0.04)
        bars = ax1.barh(ys, part["clusters"], height=height, color=[CLASS_COLOURS[c] for c in CLASSES],
                        alpha=1.0 if i == 0 else 0.45, edgecolor="white", linewidth=0.8)  # fmt: skip
        for b, v in zip(bars, part["clusters"]):
            ax1.text(b.get_width() + 2, b.get_y() + b.get_height() / 2,
                     f"{int(v)}  ({thr})", va="center", fontsize=7.4, color=INK2)  # fmt: skip
    ax1.set_yticks(y, [f"patterns of {CLASS_NAMES[c]}" for c in CLASSES])
    ax1.invert_yaxis()
    ax1.set_xlabel("clusters of the joint compendium")
    ax1.set_title("Joint compendium clusters by whose patterns they hold\n(solid: threshold 0.95; light: 0.90)")
    ax1.grid(axis="y", visible=False)
    ax1.set_xlim(0, float(cdf["clusters"].max()) * 1.32)
    if not sdf.empty:
        for i, thr in enumerate(thresholds):
            part = sdf[sdf["threshold"] == thr].pivot(index="dataset", columns="model", values="shared_patterns")
            xs = [i - 0.18, i + 0.18]
            for ds in part.index:
                ax2.plot(xs, [part.loc[ds, MODELS[0]], part.loc[ds, MODELS[1]]], color=PALE, lw=0.7, zorder=1)
            for x, m in zip(xs, MODELS):
                strip(ax2, x, part[m], COLOURS[m], spread=0.0, size=20)
                median_bar(ax2, x, part[m], rep, half=0.12)
        ax2.set_xticks(range(len(thresholds)), [f"threshold {t}" for t in thresholds])
        ax2.set_ylim(0, 1.02)
        ax2.set_ylabel("share of the dataset's patterns")
        ax2.set_title("Patterns per dataset in clusters\nholding both models' patterns")
        ax2.grid(axis="x", visible=False)
        from matplotlib.lines import Line2D

        ax2.legend(handles=[Line2D([], [], ls="", marker="o", color=COLOURS[m], markersize=6, label=NAMES[m])
                            for m in MODELS], loc="lower left")  # fmt: skip
    else:
        ax2.set_visible(False)
    fig.tight_layout()
    rep.figure(fig, "level4_compendium", "Left: clusters of the joint compendium holding patterns of both models, of "
               "ChromBPNet only or of Cherimoya only, at thresholds 0.95 and 0.90. Right: per dataset, the share of "
               "each model's patterns that sit in clusters of both models.")  # fmt: skip


# ---------------------------------------------------------------------------------------------------------------------
# level 5: Fi-NeMo hits


def pooled(g, tol: int = 5) -> dict:
    hc, hk = float(g["hits_chrombpnet"].sum()), float(g["hits_cherimoya"].sum())
    return {"hits_chrombpnet": int(hc), "hits_cherimoya": int(hk),
            f"precision_{tol}bp": float(g[f"matched_cherimoya_{tol}bp"].sum() / hk) if hk else NAN,
            f"recall_{tol}bp": float(g[f"matched_chrombpnet_{tol}bp"].sum() / hc) if hc else NAN}  # fmt: skip


def level5(rep: Report, l4: dict | None) -> dict:
    import pandas as pd

    d = rep.root / "finemo"
    if not (d / "summary.tsv").is_file():
        return {"state": "pending"}
    summ = read_tsv(rep.use("level5/summary", d / "summary.tsv"))
    summ = summ[summ["dataset"].isin(rep.shared)].reset_index(drop=True)
    missing = [ds for ds in rep.shared if ds not in set(summ["dataset"])]
    if missing:
        rep.pending.append({"part": "Level 5", "what": f"{len(missing)} of {len(rep.shared)} datasets not compared "
                            "yet", "detail": missing})  # fmt: skip
    mot = read_tsv(rep.use("level5/motifs", d / "motifs.tsv"))
    mot = mot[mot["dataset"].isin(rep.shared)].copy()
    hits = {ds: json.loads(rep.use(f"level5/{ds}/hits", d / ds / "hits.json").read_text()) for ds in summ["dataset"]}

    # totals per dataset
    tot = summ.copy()
    for m in MODELS:
        tot[f"per_peak_{m}"] = tot[f"hits_{m}"] / tot["shared_peaks"]
        tot[f"hits_all_peaks_{m}"] = [hits[ds].get("hits_all_peaks", {}).get(m, NAN) for ds in tot["dataset"]]
    tot["hits_ratio"] = tot["hits_chrombpnet"] / tot["hits_cherimoya"]
    tot["chrombpnet_hits_on_dropped_peaks"] = tot["hits_all_peaks_chrombpnet"] - tot["hits_chrombpnet"]
    rep.table("level5_per_dataset", tot)
    fam = "Fi-NeMo hits, joint vocabulary"
    for col in [c for c in summ.columns if c.startswith(("precision_", "recall_"))] + [
        "median_count_correlation",
        "coefficient_spearman",
    ]:
        rep.stats.level("5", fam, col, summ[col])
    rep.stats.paired("5", fam, "precision_5bp - recall_5bp", (summ["precision_5bp"] - summ["recall_5bp"]).to_numpy())
    for m in MODELS:
        rep.stats.level("5", fam, f"{NAMES[m]}: hits per shared peak", tot[f"per_peak_{m}"])
    rep.stats.paired("5", fam, "hits per shared peak: ChromBPNet - Cherimoya",
                     (tot["per_peak_chrombpnet"] - tot["per_peak_cherimoya"]).to_numpy())  # fmt: skip
    rep.stats.level("5", fam, "hits, ChromBPNet / Cherimoya", tot["hits_ratio"])
    if (d / "statistics.tsv").is_file():
        theirs = read_tsv(rep.use("level5/statistics", d / "statistics.tsv"))
        pairs = []
        for _, t in theirs.iterrows():
            r = rep.stats.find("5", fam, t["quantity"])
            if r is not None:
                pairs += [(r["estimate"], t["median"]), (r["ci95_lo"], t["ci95_lo"]), (r["ci95_hi"], t["ci95_hi"])]
                if "wilcoxon_p" in t and isnum(t["wilcoxon_p"]):
                    pairs.append((r["p"], t["wilcoxon_p"]))
        rep.check("level 5: finemo/statistics.tsv", pairs)

    # by cluster class and by sign (the sign of the cluster, from joint_clusters.tsv)
    joint = l4["joint"] if l4 is not None and "joint" in l4 else None
    if joint is None and (rep.root / "compendium" / "joint_clusters.tsv").is_file():
        joint = read_tsv(rep.root / "compendium" / "joint_clusters.tsv", dtype={"cluster_id": str})
    cid = mot["motif"].astype(str).str.split(".", n=1).str[-1]
    if joint is not None:
        mot["posneg"] = cid.map(dict(zip(joint["cluster_id"], joint["posneg"])))
        mot["cluster_class"] = mot["cluster_class"].fillna(cid.map(dict(zip(joint["cluster_id"], classify(joint)))))
    else:
        mot["posneg"] = np.where(mot["motif"].str.startswith("neg_"), "neg", "pos")
    mot["prefix_disagrees"] = mot["motif"].str.startswith("pos_patterns") & (mot["posneg"] == "neg")
    rows = []
    for ds, g in mot.groupby("dataset"):
        tc, tk = float(g["hits_chrombpnet"].sum()), float(g["hits_cherimoya"].sum())
        for key, col in (("class", "cluster_class"), ("sign", "posneg")):
            for value, h in g.groupby(col):
                row = {"dataset": ds, "by": key, "value": value, "motifs": int(len(h)), **pooled(h),
                       "share_of_chrombpnet_hits": float(h["hits_chrombpnet"].sum() / tc) if tc else NAN,
                       "share_of_cherimoya_hits": float(h["hits_cherimoya"].sum() / tk) if tk else NAN}  # fmt: skip
                for c in ("precision_5bp", "recall_5bp", "peak_jaccard", "coefficient_spearman"):
                    row[f"motif_median_{c}"] = float(pd.to_numeric(h[c], errors="coerce").median())
                rows.append(row)
        row = {"dataset": ds, "by": "all", "value": "all", "motifs": int(len(g)), **pooled(g),
               "share_of_chrombpnet_hits": 1.0, "share_of_cherimoya_hits": 1.0}  # fmt: skip
        for c in ("precision_5bp", "recall_5bp", "peak_jaccard", "coefficient_spearman"):
            row[f"motif_median_{c}"] = float(pd.to_numeric(g[c], errors="coerce").median())
        rows.append(row)
    groups = rep.table("level5_groups", pd.DataFrame(rows))
    for (by, value), part in groups.groupby(["by", "value"]):
        f = f"Fi-NeMo hits, {by} {value}"
        for c in ("precision_5bp", "recall_5bp", "share_of_chrombpnet_hits", "share_of_cherimoya_hits",
                  "motif_median_precision_5bp", "motif_median_recall_5bp", "motif_median_peak_jaccard",
                  "motif_median_coefficient_spearman"):  # fmt: skip
            if by == "all" and c.startswith("share_"):
                continue
            rep.stats.level("5", f, c, part[c])
    gw = {key: part.set_index("dataset") for key, part in groups.groupby(["by", "value"])}

    def contrast(label: str, a_key, a_col: str, b_key, b_col: str) -> None:
        if a_key in gw and b_key in gw:
            x, y = gw[a_key][a_col], gw[b_key][b_col]
            both = sorted(set(x.index) & set(y.index))
            rep.stats.paired("5", "Fi-NeMo hits, by cluster class", label, [x[ds] - y[ds] for ds in both])

    contrast("precision_5bp: shared clusters - Cherimoya-only clusters", ("class", "shared"), "precision_5bp",
             ("class", "cherimoya_only"), "precision_5bp")  # fmt: skip
    contrast("recall_5bp: shared clusters - ChromBPNet-only clusters", ("class", "shared"), "recall_5bp",
             ("class", "chrombpnet_only"), "recall_5bp")  # fmt: skip
    contrast("share of hits on Cherimoya-only clusters: Cherimoya - ChromBPNet", ("class", "cherimoya_only"),
             "share_of_cherimoya_hits", ("class", "cherimoya_only"), "share_of_chrombpnet_hits")  # fmt: skip
    contrast("share of hits on ChromBPNet-only clusters: ChromBPNet - Cherimoya", ("class", "chrombpnet_only"),
             "share_of_chrombpnet_hits", ("class", "chrombpnet_only"), "share_of_cherimoya_hits")  # fmt: skip
    contrast("precision_5bp: positive - negative clusters", ("sign", "pos"), "precision_5bp", ("sign", "neg"),
             "precision_5bp")  # fmt: skip

    # the largest motifs, pooled over datasets
    top = []
    for motif, g in mot.groupby("motif"):
        row = {
            "motif": motif,
            "datasets": int(g["dataset"].nunique()),
            **pooled(g),
            "annotation": g["cluster_annotation"].dropna().iloc[0] if g["cluster_annotation"].notna().any() else None,
            "class": g["cluster_class"].dropna().iloc[0] if g["cluster_class"].notna().any() else None,
            "posneg": g["posneg"].dropna().iloc[0] if g["posneg"].notna().any() else None,
            "per10k_chrombpnet": float(pd.to_numeric(g["per10k_chrombpnet"], errors="coerce").mean()),
            "per10k_cherimoya": float(pd.to_numeric(g["per10k_cherimoya"], errors="coerce").mean()),
        }
        top.append(row)
    top = pd.DataFrame(top)
    top["hits_total"] = top["hits_chrombpnet"] + top["hits_cherimoya"]
    top = rep.table("level5_motifs_pooled", top.sort_values("hits_total", ascending=False).reset_index(drop=True))

    # 11.0's motif reports, side by side
    reps = []
    for ds in summ["dataset"]:
        p = d / ds / "motif_report.tsv"
        if not p.is_file():
            continue
        mr = read_tsv(rep.use(f"level5/{ds}/motif_report", p))
        row = {"dataset": ds}
        for m in MODELS:
            tot_h = pd.to_numeric(mr.get(f"num_hits_total_{m}"), errors="coerce")
            res_h = pd.to_numeric(mr.get(f"num_hits_restricted_{m}"), errors="coerce")
            row[f"hits_total_{m}"] = float(tot_h.sum())
            row[f"restricted_share_{m}"] = float(res_h.sum() / tot_h.sum()) if tot_h.sum() else NAN
        sim = (
            mr[["cwm_similarity_chrombpnet", "cwm_similarity_cherimoya"]].apply(pd.to_numeric, errors="coerce").dropna()
        )
        row["motifs_with_both"] = int(len(sim))
        for m in MODELS:
            row[f"cwm_similarity_median_{m}"] = float(sim[f"cwm_similarity_{m}"].median()) if len(sim) else NAN
        row["cwm_similarity_median_diff"] = float((sim["cwm_similarity_cherimoya"] - sim["cwm_similarity_chrombpnet"])
                                                  .median()) if len(sim) else NAN  # fmt: skip
        reps.append(row)
    mrdf = rep.table("level5_motif_reports", pd.DataFrame(reps))
    if not mrdf.empty:
        for m in MODELS:
            rep.stats.level("5", "Fi-NeMo motif reports", f"{NAMES[m]}: median hit-seqlet CWM similarity",
                            mrdf[f"cwm_similarity_median_{m}"])  # fmt: skip
            rep.stats.level("5", "Fi-NeMo motif reports", f"{NAMES[m]}: share of hits restricted to TF-MoDISco's "
                            "regions", mrdf[f"restricted_share_{m}"])  # fmt: skip
        rep.stats.paired("5", "Fi-NeMo motif reports", "per-motif hit-seqlet CWM similarity: Cherimoya - ChromBPNet "
                         "(median per dataset)", mrdf["cwm_similarity_median_diff"])  # fmt: skip

    # the production hits (cbp10 against cheri39), when level 5 has compared them
    prod = []
    for ds in rep.shared:
        p = d / ds / "production" / "hits.json"
        if p.is_file():
            t = json.loads(rep.use(f"level5/{ds}/production", p).read_text())
            prod.append({"dataset": ds, "shared_peaks": t.get("shared_peaks"), **{f"hits_{m}": t.get("hits", {}).get(m)
                         for m in MODELS}, **{k: v for k, v in t.items() if k.startswith(("precision_", "recall_"))},
                         "median_count_correlation": t.get("median_count_correlation")})  # fmt: skip
    prod = pd.DataFrame(prod)
    if not prod.empty:
        rep.table("level5_production", prod)
        for col in [c for c in prod.columns if c.startswith(("precision_", "recall_"))]:
            rep.stats.level("5", "Fi-NeMo production hits (cbp10 vs cheri39)", col, prod[col])
    if len(prod) < len(rep.shared):
        rep.pending.append({"part": "Level 5", "what": "the production hits (ChromBPNet's on cbp10 against "
                            "Cherimoya's on cheri39, through level 4's cluster matches)", "detail":
                            [f"{len(prod)} of {len(rep.shared)} datasets compared; they need the cheri39 compendium, "
                             "Cherimoya's Fi-NeMo on it and level 4's matches"]})  # fmt: skip
    plot_level5(rep, groups, top)
    return {"state": "done", "totals": tot, "groups": groups, "motifs": mot, "top": top, "reports": mrdf,
            "production": prod, "missing": missing}  # fmt: skip


def plot_level5(rep: Report, groups, top) -> None:
    plt = pyplot()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.1), gridspec_kw={"width_ratios": [1.2, 1.0]})
    cats = [("all", "all"), ("class", "shared"), ("class", "chrombpnet_only"), ("class", "cherimoya_only")]
    names = ["all motifs", "clusters of\nboth models", "ChromBPNet-only\nclusters", "Cherimoya-only\nclusters"]
    for i, (by, value) in enumerate(cats):
        part = groups[(groups["by"] == by) & (groups["value"] == value)]
        for x, col, m in ((i - 0.17, "precision_5bp", "cherimoya"), (i + 0.17, "recall_5bp", "chrombpnet")):
            strip(ax1, x, part[col], COLOURS[m], spread=0.06, size=16)
            median_bar(ax1, x, part[col], rep, half=0.12)
    ax1.set_xticks(range(len(cats)), names)
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel("share of hits matched within 5 bp")
    ax1.set_title("Hits matched across the models, by cluster class")
    ax1.grid(axis="x", visible=False)
    from matplotlib.lines import Line2D

    ax1.legend(handles=[Line2D([], [], ls="", marker="o", color=COLOURS["cherimoya"], markersize=6,
                               label="of Cherimoya's hits (precision)"),
                        Line2D([], [], ls="", marker="o", color=COLOURS["chrombpnet"], markersize=6,
                               label="of ChromBPNet's hits (recall)")], loc="lower left")  # fmt: skip
    t = top.dropna(subset=["per10k_chrombpnet", "per10k_cherimoya"])
    floor = 0.01
    x = np.maximum(t["per10k_chrombpnet"].to_numpy(float), floor)
    y = np.maximum(t["per10k_cherimoya"].to_numpy(float), floor)
    for c in ("shared", "chrombpnet_only", "cherimoya_only"):
        sel = (t["class"] == c).to_numpy()
        ax2.scatter(x[sel], y[sel], s=14, color=CLASS_COLOURS[c], alpha=0.75, edgecolor="white", linewidth=0.4,
                    label=f"cluster of {CLASS_NAMES[c]}", zorder=3 if c != "shared" else 2)  # fmt: skip
    lo, hi = float(min(x.min(), y.min())) / 1.5, float(max(x.max(), y.max())) * 1.5
    ax2.plot([lo, hi], [lo, hi], color=MUTED, lw=0.9, zorder=1)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlim(lo, hi)
    ax2.set_ylim(lo, hi)
    ax2.set_xlabel("ChromBPNet hits per 10,000 shared peaks")
    ax2.set_ylabel("Cherimoya hits per 10,000 shared peaks")
    ax2.set_title("Each motif of the joint vocabulary (mean over datasets)")
    listing = []
    for i, (_, r) in enumerate(t.head(3).iterrows(), start=1):
        ax2.annotate(str(i), (max(r["per10k_chrombpnet"], floor), max(r["per10k_cherimoya"], floor)), xytext=(-7, 5),
                     textcoords="offset points", ha="right", fontsize=7.6, fontweight="bold", color=INK)  # fmt: skip
        listing.append(f"{i}  {r['annotation']} · {r['posneg']} · {CLASS_NAMES.get(r['class'], r['class'])}")
    ax2.text(0.03, 0.97, "\n".join(listing), transform=ax2.transAxes, ha="left", va="top", fontsize=7.4, color=INK2,
             linespacing=1.5)  # fmt: skip
    ax2.legend(loc="lower right", fontsize=7.2)
    fig.tight_layout()
    rep.figure(fig, "level5_hits", "Left: per dataset, the share of Cherimoya's hits matched by ChromBPNet (precision) "
               "and of ChromBPNet's hits matched by Cherimoya (recall), by cluster class. Right: hits per 10,000 peaks "
               "of each motif for the two models on log axes.")  # fmt: skip


# ---------------------------------------------------------------------------------------------------------------------
# provenance


def level_markers(root: Path) -> list[dict]:
    out = []
    for level, sub in ((1, "performance"), (2, "attributions"), (3, "modisco"), (4, "compendium"), (5, "finemo")):
        marks = []
        for p in sorted((root / sub / ".done").glob("*.json")):
            try:
                marks.append(json.loads(p.read_text()))
            except (OSError, ValueError):
                continue
        if not marks:
            out.append({"level": level, "dir": sub, "units": 0})
            continue
        written = sorted(m.get("written") or "" for m in marks)
        out.append({"level": level, "dir": sub, "units": len(marks), "first": written[0], "last": written[-1],
                    "commits": sorted({m.get("commit") or "?" for m in marks})})  # fmt: skip
    return out


def provenance(rep: Report) -> dict:
    a = rep.args
    repos = [("rvi-downstream (this code)", Path(__file__).resolve().parent)]
    if a.folds_dir is not None:
        repos.append(("igvf_tf_chrombpnet (REPO_ROOT, folds)", Path(a.folds_dir).parent))
    for spec in a.repo or []:
        name, _, path = spec.partition("=")
        if path:
            repos.append((name, Path(path)))
    commits = []
    for name, path in repos:
        if path.is_dir():
            commits.append({"name": name, **git_info(path)})
    engine = None
    p = a.cheri_root / "attribution_engine.json"
    if p.is_file():
        try:
            engine = json.loads(rep.use("provenance/attribution_engine", p).read_text())
        except (OSError, ValueError):
            engine = None
    compendia = []
    for tag in sorted({a.cbp_tag, a.cheri_tag, a.joint, a.sensitivity_tag or f"{a.joint}_t090", a.supplement_tag}):
        p = a.compendia_root / tag / "compendium.json"
        if not p.is_file():
            continue
        try:
            cj = json.loads(rep.use(f"provenance/compendium_{tag}", p).read_text())
        except (OSError, ValueError):
            continue
        settings = cj.get("settings")
        if isinstance(settings, str):
            try:
                settings = json.loads(settings.replace("'", '"'))
            except ValueError:
                settings = {}
        compendia.append(
            {
                "tag": tag,
                "created": cj.get("date_created"),
                "pipeline": cj.get("pipeline") or {},
                "software": cj.get("software") or {},
                "threshold": (settings or {}).get("threshold"),
            }
        )
    return {"commits": commits, "engine": engine, "markers": level_markers(rep.root), "compendia": compendia}


# ---------------------------------------------------------------------------------------------------------------------
# the page: formatting


def h(text) -> str:
    return html.escape("" if text is None else str(text), quote=True)


def num(x, d: int = 3, signed: bool = False) -> str:
    if not isnum(x):
        return "NA"
    s = f"{float(x):+.{d}f}" if signed else f"{float(x):.{d}f}"
    return s.replace("-", "−")


def ci(lo, hi, d: int = 3, signed: bool = False) -> str:
    if not (isnum(lo) and isnum(hi)):
        return ""
    return f"[{num(lo, d, signed)}, {num(hi, d, signed)}]"


def pval(p) -> str:
    if not isnum(p):
        return "NA"
    p = float(p)
    if p < 1e-3:
        mant, exp = f"{p:.1e}".split("e")
        return f"{mant}×10<sup>{int(exp)}</sup>".replace("-", "−")
    return f"{p:.2g}" if p < 0.1 else f"{p:.2f}"


def count(n) -> str:
    return f"{int(n):,}" if isnum(n) else "NA"


def pct(x, d: int = 1) -> str:
    return f"{100 * float(x):.{d}f}%" if isnum(x) else "NA"


def est_ci(row: dict | None, d: int = 3, signed: bool = False) -> str:
    if row is None:
        return "NA"
    c = ci(row["ci95_lo"], row["ci95_hi"], d, signed)
    return f'{num(row["estimate"], d, signed)}<span class="sub">{c}</span>' if c else num(row["estimate"], d, signed)


def diff_cell(r) -> str:
    """A level-1 measure's dataset difference and its CI."""
    return f'{num(r["median_dataset_diff"], 4, True)}<span class="sub">{ci(r["ci95_lo"], r["ci95_hi"], 4, True)}</span>'


def test_cell(row: dict | None, d: int = 3, total: int | None = None) -> str:
    """The median difference, its CI, how many were above 0, and p."""
    if row is None:
        return "NA"
    n = int(row["n"]) if isnum(row["n"]) else 0
    hi = int(row["higher"]) if isnum(row["higher"]) else 0
    return (f'{num(row["estimate"], d, True)}<span class="sub">{ci(row["ci95_lo"], row["ci95_hi"], d, True)} · '
            f'{hi}/{total or n} above 0 · p {pval(row["p"])}</span>')  # fmt: skip


def key(model: str) -> str:
    cls = {"chrombpnet": "cbp", "cherimoya": "cheri"}.get(model, "cross")
    name = NAMES.get(model, "Across the models")
    return f'<span class="key"><i class="sw {cls}"></i>{h(name)}</span>'


class Numbering:
    def __init__(self) -> None:
        self.fig = 0
        self.tab = 0

    def figure(self) -> int:
        self.fig += 1
        return self.fig

    def table(self) -> int:
        self.tab += 1
        return self.tab


def figure_html(rep: Report, n: Numbering, name: str, caption: str, wide: bool = True) -> str:
    f = rep.figures.get(name)
    if f is None:
        return ""
    k = n.figure()
    return (f'<figure class="fig" id="fig-{k}"><div class="plate"><img class="{"wide" if wide else ""}" '
            f'src="data:image/png;base64,{f["b64"]}" width="{f["w"]}" height="{f["h"]}" alt="{h(f["alt"])}" '
            f'decoding="async"></div><figcaption><b>Figure {k}.</b> {caption}</figcaption></figure>')  # fmt: skip


def table_html(n: Numbering, caption: str, header: list[str], rows: list[list[str]], left: int = 1,
               cls: str = "") -> str:  # fmt: skip
    k = n.table()
    head = "".join(f'<th scope="col"{' class="l"' if i < left else ""}>{c}</th>' for i, c in enumerate(header))
    body = "".join("<tr>" + "".join((f'<th scope="row" class="l">{c}</th>' if i == 0 else
                                     f'<td{" class=\"l\"" if i < left else ""}>{c}</td>') for i, c in enumerate(r))
                   + "</tr>" for r in rows)  # fmt: skip
    return (
        f'<div class="tblock"><p class="tcap"><b>Table {k}.</b> {caption}</p><div class="tbl {cls}">'
        f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div></div>"
    )


def note_html(state: str, label: str, body: str) -> str:
    return f'<div class="note"><span class="chip {state}">{h(label)}</span><div>{body}</div></div>'


def pending_html(rep: Report, part: str) -> str:
    out = []
    eta = f" Expected {h(rep.args.pending_eta)}." if rep.args.pending_eta else ""
    for p in rep.pending:
        if p["part"] != part:
            continue
        detail = p.get("detail") or []
        more = f" ({h('; '.join(str(x) for x in detail[:6]))}{' …' if len(detail) > 6 else ''})" if detail else ""
        out.append(note_html("pending", "Pending", f"<p><b>{h(p['what'])}</b>{more}.{eta}</p>"))
    return "".join(out)


def problem_html(rep: Report, part: str) -> str:
    return "".join(note_html("problem", "Not computed", f"<p><b>{h(name)}</b>: {h(msg)}</p>")
                   for name, msg in rep.problems if name == part)  # fmt: skip


def details(summary: str, body: str) -> str:
    return f"<details><summary>{summary}</summary>{body}</details>"


# ---------------------------------------------------------------------------------------------------------------------
# the page: sections


CSS = """
@import url("https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:ital,\
wght@0,400;0,500;0,600;1,400&family=Spectral:wght@500;600&display=swap");
/* Layout: one reading column (max 74rem) under a masthead; the sections follow the plan's levels in order; figures
   sit on light plates in both themes; every table scrolls inside its own frame. The two models' hues are the only
   saturated colours on the page. */
:root {
  --bg: #f5f6f7;
  --surface: #ffffff;
  --ink: #16191d;
  --ink-2: #3c434b;
  --muted: #68717b;
  --rule: #d3d9df;
  --rule-soft: #e6eaee;
  --plate: #ffffff;
  --cbp: #0072b2;
  --cheri: #d55e00;
  --cross: #3b3f45;
  --tn5: #e69f00;
  --done: #1d6a43;
  --done-bg: #e2f0e7;
  --pending: #7a5100;
  --pending-bg: #fbeed3;
  --problem: #9a2b2b;
  --problem-bg: #f7e1e1;
  --focus: #0072b2;
  --font-display: "Spectral", Georgia, "Times New Roman", serif;
  --font-body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --font-mono: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  --step--1: 0.8125rem;
  --step-0: 0.9375rem;
  --step-1: 1.125rem;
  --step-2: 1.5rem;
  --step-3: 2.25rem;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #101316; --surface: #161a1e; --ink: #e6e9ec; --ink-2: #c2c8ce; --muted: #939ca6; --rule: #2b3137;
    --rule-soft: #21262b; --plate: #f7f8f9; --cbp: #2f8fd0; --cheri: #e8762a; --cross: #aab1b8; --tn5: #e69f00;
    --done: #8fd0a8; --done-bg: #16291e; --pending: #efc06a; --pending-bg: #30240e; --problem: #f09a9a;
    --problem-bg: #361818; --focus: #6cb8ea; color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #101316; --surface: #161a1e; --ink: #e6e9ec; --ink-2: #c2c8ce; --muted: #939ca6; --rule: #2b3137;
  --rule-soft: #21262b; --plate: #f7f8f9; --cbp: #2f8fd0; --cheri: #e8762a; --cross: #aab1b8; --tn5: #e69f00;
  --done: #8fd0a8; --done-bg: #16291e; --pending: #efc06a; --pending-bg: #30240e; --problem: #f09a9a;
  --problem-bg: #361818; --focus: #6cb8ea; color-scheme: dark;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font-family: var(--font-body); font-size: var(--step-0);
  line-height: 1.6; -webkit-font-smoothing: antialiased; text-rendering: optimizeLegibility; }
.page { max-width: 74rem; margin-inline: auto; padding-inline: clamp(16px, 4vw, 48px); padding-block: 40px 80px; }
h1, h2 { font-family: var(--font-display); font-weight: 600; line-height: 1.15; margin: 0; text-wrap: balance;
  letter-spacing: -0.005em; }
h1 { font-size: var(--step-3); }
h2 { font-size: var(--step-2); }
h3 { font-size: var(--step-1); font-weight: 600; line-height: 1.3; margin: 0; text-wrap: balance; }
p { margin: 0; max-width: 74ch; }
a { color: inherit; text-decoration-color: var(--muted); text-underline-offset: 2px; }
a:hover { text-decoration-color: currentColor; }
a:focus-visible, summary:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; border-radius: 2px; }
code, .path { font-family: var(--font-mono); font-size: 0.84em; overflow-wrap: anywhere; }
sup { font-size: 0.72em; line-height: 0; }
.eyebrow { font-family: var(--font-mono); font-size: 0.74rem; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--muted); }
.masthead { display: grid; gap: 14px; padding-bottom: 26px; border-bottom: 1px solid var(--rule); }
.dek { font-size: var(--step-1); line-height: 1.5; color: var(--ink-2); max-width: 66ch; }
.meta { display: flex; flex-wrap: wrap; gap: 4px 20px; font-size: var(--step--1); color: var(--muted); }
.legend { display: flex; flex-wrap: wrap; gap: 6px 26px; font-size: var(--step--1); color: var(--ink-2); }
.lg { display: inline-flex; flex-wrap: wrap; align-items: baseline; gap: 2px 8px; min-width: 0; }
.lg .key { font-weight: 600; color: var(--ink); }
.lgd { color: var(--muted); }
.status { display: flex; flex-wrap: wrap; gap: 8px; }
.chip { display: inline-flex; align-items: center; gap: 6px; font-size: 0.76rem; font-weight: 500; line-height: 1.5;
  padding: 1px 9px 1px 8px; border-radius: 999px; background: var(--done-bg); color: var(--done); white-space: nowrap; }
.chip::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
.chip.pending { background: var(--pending-bg); color: var(--pending); }
.chip.problem { background: var(--problem-bg); color: var(--problem); }
.toc { display: flex; flex-wrap: wrap; gap: 6px 20px; padding-block: 14px; border-bottom: 1px solid var(--rule);
  font-size: var(--step--1); }
.toc a { color: var(--ink-2); text-decoration: none; }
.toc a:hover { color: var(--ink); text-decoration: underline; }
.toc .n { font-family: var(--font-mono); color: var(--muted); margin-right: 5px; }
section { display: grid; gap: 20px; padding-block: 44px 6px; }
section + section { border-top: 1px solid var(--rule); margin-top: 30px; }
section > * { min-width: 0; }
.head { display: grid; gap: 6px; }
.lead { font-size: 1.0rem; }
.findings { list-style: none; margin: 0; padding: 0; display: grid; }
.findings li { display: grid; grid-template-columns: 10.5rem minmax(0, 1fr); gap: 4px 20px; padding-block: 12px;
  border-top: 1px solid var(--rule-soft); }
.findings li:first-child { border-top: 0; }
.findings .topic { font-size: 0.74rem; font-weight: 600; letter-spacing: 0.07em; text-transform: uppercase;
  color: var(--muted); padding-top: 3px; }
.findings p { max-width: 70ch; }
.keys { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 15rem), 1fr)); gap: 0;
  border-top: 1.5px solid var(--ink); border-bottom: 1px solid var(--rule); }
.keys div { padding: 12px 16px 12px 0; display: grid; gap: 2px; align-content: start; min-width: 0; }
.keys dt { font-size: 0.74rem; color: var(--muted); letter-spacing: 0.03em; }
.keys dd { margin: 0; font-size: 1.3rem; font-weight: 500; line-height: 1.25; }
.keys dd small { display: block; font-size: 0.78rem; font-weight: 400; color: var(--ink-2); line-height: 1.4; }
figure { margin: 0; display: grid; gap: 8px; }
.plate { background: var(--plate); border: 1px solid var(--rule-soft); border-radius: 3px; padding: 10px;
  overflow-x: auto; }
.plate img { display: block; width: 100%; height: auto; max-width: 100%; }
.plate img.wide { min-width: 620px; max-width: none; }
figcaption { font-size: var(--step--1); color: var(--ink-2); max-width: 86ch; }
figcaption b, .tcap b { color: var(--ink); font-weight: 600; }
.tblock { display: grid; gap: 8px; min-width: 0; }
.tcap { font-size: var(--step--1); color: var(--ink-2); max-width: 86ch; }
.tbl { overflow-x: auto; border-top: 1.5px solid var(--ink); border-bottom: 1px solid var(--rule); }
table { border-collapse: collapse; width: 100%; font-size: 0.85rem; line-height: 1.4;
  font-variant-numeric: tabular-nums; }
th, td { padding: 7px 12px 7px 0; text-align: right; vertical-align: top; white-space: nowrap; }
th:last-child, td:last-child { padding-right: 2px; }
thead th { font-size: 0.76rem; font-weight: 600; color: var(--ink-2); border-bottom: 1px solid var(--rule);
  vertical-align: bottom; white-space: normal; min-width: 5.5rem; }
tbody th { font-weight: 500; }
tbody tr + tr > * { border-top: 1px solid var(--rule-soft); }
.l { text-align: left; }
td .sub, th .sub { display: block; font-size: 0.74rem; color: var(--muted); font-weight: 400; }
.tbl.compact td, .tbl.compact th { padding-block: 4px; }
.note { display: grid; grid-template-columns: auto minmax(0, 1fr); gap: 8px 12px; align-items: baseline;
  padding: 12px 14px; background: var(--surface); border: 1px solid var(--rule); border-radius: 4px; }
.note p { max-width: 80ch; }
.cols { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 24rem), 1fr)); gap: 20px 32px; }
.cols > * { min-width: 0; }
details { border-top: 1px solid var(--rule-soft); padding-top: 10px; }
details > * + * { margin-top: 12px; }
summary { cursor: pointer; font-size: var(--step--1); font-weight: 500; color: var(--ink-2); }
.key { display: inline-flex; align-items: center; gap: 6px; white-space: nowrap; }
.sw { width: 10px; height: 10px; border-radius: 2px; display: inline-block; background: var(--cross); }
.sw.cbp { background: var(--cbp); }
.sw.cheri { background: var(--cheri); }
.sw.tn5 { background: var(--tn5); }
ul.plain { margin: 0; padding-left: 1.1rem; display: grid; gap: 8px; max-width: 80ch; }
ul.plain li::marker { color: var(--muted); }
dl.defs { display: grid; grid-template-columns: minmax(8rem, max-content) minmax(0, 1fr); gap: 8px 22px; margin: 0;
  font-size: var(--step--1); }
dl.defs dt { font-weight: 600; }
dl.defs dd { margin: 0; min-width: 0; max-width: 80ch; }
@media (max-width: 640px) {
  .findings li { grid-template-columns: minmax(0, 1fr); }
  dl.defs { grid-template-columns: minmax(0, 1fr); }
  dl.defs dd { margin-bottom: 6px; }
  h1 { font-size: 1.85rem; }
}
"""


def masthead(rep: Report, R: dict, generated: str) -> str:
    l1 = R.get("l1") or {}
    chips = []

    def chip(label: str, state: str) -> None:
        chips.append(f'<span class="chip {state}">{h(label)}</span>')

    if l1.get("state") == "done":
        chip(f"Level 1 · {l1['n_folds']}/{l1['expected'] or l1['n_folds']} folds",
             "pending" if l1.get("missing") else "done")  # fmt: skip
    else:
        chip("Level 1 · " + ("not computed" if l1 == {} else "pending"), "problem" if l1 == {} else "pending")
    for i, k in ((2, "l2"), (3, "l3"), (4, "l4"), (5, "l5")):
        part = R.get(k)
        if part is None:
            chip(f"Level {i} · not computed", "problem")
        elif part.get("state") != "done":
            chip(f"Level {i} · pending", "pending")
        elif i == 4:
            chip("Level 4 · done", "done")
        else:
            miss = part.get("missing") or []
            chip(f"Level {i} · {len(rep.shared) - len(miss)}/{len(rep.shared)} datasets", "pending" if miss else "done")
    for p in rep.pending:
        if p["part"] in ("Level 4", "Level 5") and ("supplement" in p["what"] or "production" in p["what"]):
            chip(("cheri39 supplement" if "supplement" in p["what"] else "production hits") + " · pending", "pending")
    eta = f" (expected {h(rep.args.pending_eta)})" if rep.args.pending_eta and rep.pending else ""
    return f"""
<header class="masthead">
  <p class="eyebrow">AMSC · 10x multiome ATAC · model comparison</p>
  <h1>Cherimoya vs ChromBPNet</h1>
  <p class="dek">Two sequence-to-accessibility models trained on the same {len(rep.everything)} AMSC pseudobulk ATAC
  datasets with the same five chromosome folds. Performance covers every dataset; attributions, TF-MoDISco motifs,
  motif compendia and Fi-NeMo hits cover the {len(rep.shared)} datasets both models were interpreted on.</p>
  <div class="legend"><span class="lg">{key("chrombpnet")}<span class="lgd">bias-corrected model for attributions, full
  model for performance</span></span><span class="lg">{key("cherimoya")}<span class="lgd">no bias model</span></span>
  <span class="lg">{key("cross")}<span class="lgd">agreement between the two</span></span></div>
  <div class="status">{"".join(chips)}</div>
  <p class="meta"><span>Generated {h(generated)}</span><span>Data as of the newest input,
  {h(R.get("newest_input", "NA"))}</span><span>Pending parts{eta} are marked as such</span></p>
</header>"""


def toc() -> str:
    items = [
        ("summary", "", "Summary"),
        ("level-1", "1", "Performance"),
        ("level-2", "2", "Attributions"),
        ("level-3", "3", "TF-MoDISco"),
        ("level-4", "4", "Compendia"),
        ("level-5", "5", "Fi-NeMo hits"),
        ("bias-control", "", "Bias control"),
        ("caveats", "", "Caveats"),
        ("methods", "", "Methods and provenance"),
    ]
    links = "".join(f'<a href="#{i}">{f'<span class="n">{n}</span>' if n else ""}{h(t)}</a>' for i, n, t in items)
    return f'<nav class="toc" aria-label="Sections">{links}</nav>'


def s(rep: Report, level: str, family: str, quantity: str, over: str = "datasets") -> dict | None:
    return rep.stats.find(level, family, quantity, over)


def summary_section(rep: Report, R: dict) -> str:
    items, keys = [], []
    l1, l2, l3, l4, l5, bc = (R.get(k) for k in ("l1", "l2", "l3", "l4", "l5", "bias"))
    if l1 and l1.get("state") == "done":
        md = l1["measures"]
        a = md[md["set"] == "all"].set_index("measure")
        nf, nd = l1["n_folds"], l1["n_datasets"]
        pear, jsd = a.loc["counts_pearson"], a.loc["median_jsd"]
        full = [m for m in MEASURES if a.loc[m, "cherimoya_better_folds"] == nf]
        part = [m for m in MEASURES if m not in full]
        all_ds = all(a.loc[m, "cherimoya_better_datasets"] == nd for m in MEASURES)
        text = (f"Cherimoya predicts held-out accessibility better than ChromBPNet"
                f"{f' in all {nd} datasets, on all five measures' if all_ds else ''}. Over {nf} test folds the median "
                f"counts Pearson r rises from {num(pear['chrombpnet_median'])} to {num(pear['cherimoya_median'])} "
                f"(difference of dataset means {num(pear['median_dataset_diff'], 3, True)}, 95% CI "
                f"{ci(pear['ci95_lo'], pear['ci95_hi'], 3, True)}) and the median profile JSD falls from "
                f"{num(jsd['chrombpnet_median'])} to {num(jsd['cherimoya_median'])}.")  # fmt: skip
        if full:
            text += f" Cherimoya is better on every fold for {', '.join(MEASURE_SHORT[m] for m in full)}"
            text += (
                (
                    f"; for {', '.join(MEASURE_SHORT[m] for m in part)} on "
                    + ", ".join(f"{int(a.loc[m, 'cherimoya_better_folds'])}" for m in part)
                    + f" of {nf}."
                )
                if part
                else "."
            )
        items.append(("Performance", text))
        keys.append(
            (
                "Counts Pearson r, median of " + str(nf) + " folds",
                f"{num(pear['chrombpnet_median'])} → {num(pear['cherimoya_median'])}",
                f"ChromBPNet → Cherimoya; Cherimoya better on {int(pear['cherimoya_better_folds'])}/{nf} folds",
            )
        )
        rd = l1["residuals"]
        r = rd[rd["set"] == "all"]
        lower = [int(x) for x in r[(r["rmse_diff_median"] < 0) & (r["wilcoxon_p"] < 0.01)]["decile"]]
        higher = [int(x) for x in r[(r["rmse_diff_median"] > 0) & (r["wilcoxon_p"] < 0.01)]["decile"]]
        if lower:
            d1 = r[r["decile"] == min(lower)].iloc[0]
            text = (f"The gain sits at the extremes of accessibility. Cherimoya's RMSE is lower in deciles "
                    f"{', '.join(map(str, lower))} of observed counts (decile {int(d1['decile'])}: "
                    f"{num(d1['rmse_diff_median'], 3, True)} log counts)")  # fmt: skip
            text += (
                f" and slightly higher in deciles {', '.join(map(str, higher))} (at most "
                f"{num(r[r['decile'].isin(higher)]['rmse_diff_median'].max(), 3, True)})."
                if higher
                else "."
            )
            text += (" Both models over-predict the least accessible peaks and under-predict the most accessible; "
                     "Cherimoya less so at the low end.")  # fmt: skip
            items.append(("Where it gains", text))
    if l2 and l2.get("state") == "done":
        w = l2["windows"][0]
        fam = f"attributions, {w}, pearson"
        A, B = s(rep, "2", fam, "cross_fold"), s(rep, "2", fam, "cross_average")
        Cc, Ck = s(rep, "2", fam, "loo_chrombpnet"), s(rep, "2", fam, "loo_cherimoya")
        if A and B and Cc and Ck:
            text = (
                f"Per peak, the fold-averaged attributions of the two models agree about as well as each model "
                f"agrees with its own folds: Pearson r {num(B['estimate'])} in the {int(w[1:])}-bp window around "
                f"the summit, "
                f"against leave-one-out ceilings of {num(Cc['estimate'])} (ChromBPNet) and {num(Ck['estimate'])} "
                f"(Cherimoya). Single folds agree less across the models ({num(A['estimate'])}); part of that gap "
                f"is built in, because a ceiling compares one fold with the mean of the other four."
            )
            items.append(("Attributions", text))
            keys.append(
                (
                    "Attribution Pearson r, fold averages",
                    f"{num(B['estimate'])}",
                    f"ceilings {num(Cc['estimate'])} (ChromBPNet), {num(Ck['estimate'])} (Cherimoya)",
                )
            )
        ho = [s(rep, "2", fam, f"{a} - {b}") for a, b in (("heldout_chrombpnet", "train_chrombpnet"),
                                                       ("heldout_cherimoya", "train_cherimoya"),
                                                       ("cross_heldout", "cross_train"))]  # fmt: skip
        if all(ho):
            worst = max(abs(x["estimate"]) for x in ho)
            text = (f"On chromosomes a fold never trained on, agreement is within {num(worst, 4)} of agreement on its "
                    f"training chromosomes, for each model and across the models (Pearson r). The drop is consistent "
                    f"but tiny: no sign that either model memorized its training chromosomes.")  # fmt: skip
            items.append(("Held-out chromosomes", text))
    if l3 and l3.get("state") == "done":
        w = l3["wide"]
        tc, tk = w[("shared_tomtom_count", "chrombpnet")].median(), w[("shared_tomtom_count", "cherimoya")].median()
        sc, sk = w[("shared_tomtom_seqlets", "chrombpnet")].median(), w[("shared_tomtom_seqlets", "cherimoya")].median()
        qc_c, qc_k = w[("qc_cross", "chrombpnet")].median(), w[("qc_cross", "cherimoya")].median()
        qw_c, qw_k = w[("qc_within", "chrombpnet")].median(), w[("qc_within", "cherimoya")].median()
        text = (f"TF-MoDISco finds the same motifs in both. {pct(tc, 0)} of ChromBPNet's and {pct(tk, 0)} of "
                f"Cherimoya's fold-average patterns have a tomtom partner in the other model (medians over datasets), "
                f"and those patterns hold {pct(sc)} and {pct(sk)} of the seqlets. At QC scale, the cross-model match "
                f"({num(qc_c, 2)} and {num(qc_k, 2)}) sits just under each model's fold-to-fold ceiling "
                f"({num(qw_c, 2)} and {num(qw_k, 2)}).")  # fmt: skip
        items.append(("Motifs", text))
        keys.append(("Patterns with a partner in the other model", f"{pct(tc, 0)} / {pct(tk, 0)}",
                     "ChromBPNet / Cherimoya, tomtom q ≤ 0.05, median of datasets"))  # fmt: skip
    if l4 and l4.get("state") == "done":
        c = l4["classes"]
        c95 = c[c["threshold"] == "0.95"].set_index("class")
        c90 = c[c["threshold"] == "0.90"].set_index("class") if "0.90" in set(c["threshold"]) else None
        if set(CLASSES) <= set(c95.index):
            text = (f"In the joint compendium, {int(c95.loc['shared', 'clusters'])} clusters hold patterns of both "
                    f"models and carry {pct(c95.loc['shared', 'seqlet_share_chrombpnet'])} of ChromBPNet's and "
                    f"{pct(c95.loc['shared', 'seqlet_share_cherimoya'])} of Cherimoya's seqlets. The "
                    f"{int(c95.loc['chrombpnet_only', 'clusters'])} ChromBPNet-only and "
                    f"{int(c95.loc['cherimoya_only', 'clusters'])} Cherimoya-only clusters are small, mostly single "
                    f"patterns")  # fmt: skip
            if c90 is not None and set(CLASSES) <= set(c90.index):
                text += (f", and many merge when the compendium is rebuilt at threshold 0.90 "
                         f"({int(c90.loc['chrombpnet_only', 'clusters'])} and "
                         f"{int(c90.loc['cherimoya_only', 'clusters'])}, while the shared core keeps "
                         f"{int(c90.loc['shared', 'clusters'])}).")  # fmt: skip
            else:
                text += "."
            items.append(("Compendia", text))
    if l5 and l5.get("state") == "done":
        fam = "Fi-NeMo hits, joint vocabulary"
        P, Rr = s(rep, "5", fam, "precision_5bp"), s(rep, "5", fam, "recall_5bp")
        ratio = s(rep, "5", fam, "hits, ChromBPNet / Cherimoya")
        if P and Rr and ratio:
            text = (
                f"Called with one shared motif vocabulary, ChromBPNet has {num(ratio['estimate'], 2)}× as many "
                f"hits as Cherimoya. {pct(P['estimate'], 0)} of Cherimoya's hits match a ChromBPNet hit of the same "
                f"motif and strand within 5 bp (precision), and Cherimoya recovers {pct(Rr['estimate'], 0)} of "
                f"ChromBPNet's (recall). Agreement is highest on motifs both models found."
            )
            items.append(("Hits", text))
            keys.append(("Fi-NeMo hits matched within 5 bp", f"{pct(P['estimate'], 0)} / {pct(Rr['estimate'], 0)}",
                         "of Cherimoya's hits / of ChromBPNet's hits, median of datasets"))  # fmt: skip
    if bc and bc.get("state") == "done" and l3 and l3.get("state") == "done":
        per = l3["per"]
        fc, fk = (
            int(per[per["model"] == "chrombpnet"]["bias_flagged"].sum()),
            int(per[per["model"] == "cherimoya"]["bias_flagged"].sum()),
        )
        t = s(rep, "3", "TF-MoDISco, Cherimoya - ChromBPNet", "Tn5/DNase-flagged patterns: Cherimoya - ChromBPNet")
        split = bc["split"].set_index("reference_kind")
        n_checked = int(split["patterns"].sum())
        n_tn5 = int(split.loc["Tn5/DNase", "patterns"]) if "Tn5/DNase" in split.index else 0
        n_tf = int(split.loc["TF or other", "patterns"]) if "TF or other" in split.index else 0
        text = (
            f"Cherimoya, which has no bias model, carries more Tn5/DNase patterns: {fk} database-flagged "
            f"fold-average patterns over the {len(per['dataset'].unique())} datasets against {fc} for bias-corrected "
            f"ChromBPNet (p {pval(t['p']) if t else 'NA'}). Of the {n_checked} Cherimoya patterns without a "
            f"cluster partner, {n_tn5 + n_tf} match a pattern of the bias-included ChromBPNet, but only {n_tn5} "
            f"match one of its Tn5/DNase patterns; {n_tf} match its TF patterns."
        )
        items.append(("Bias", text))
    if rep.pending:
        what = sorted({p["what"].split(" (")[0] for p in rep.pending})
        eta = f" Expected {h(rep.args.pending_eta)}." if rep.args.pending_eta else ""
        items.append(("Pending", f"Not in this report yet: {'; '.join(what)}.{eta}"))
    if rep.problems:
        items.append(("Not computed", "; ".join(f"{n}: {m}" for n, m in rep.problems)))
    lis = "".join(f'<li><span class="topic">{h(t)}</span><p>{x}</p></li>' for t, x in items)
    kd = "".join(f"<div><dt>{h(a)}</dt><dd>{b}<small>{h(c)}</small></dd></div>" for a, b, c in keys)
    floor = ("<p class=\"tcap\">p-values are two-sided Wilcoxon signed-rank tests. With n paired values the exact p "
             "cannot fall below 2/2<sup>n</sup>: 0.002 for 10 datasets and 3.6×10<sup>−12</sup> for 39, while over 50 "
             "or 195 folds it saturates further. At those floors read p as “every dataset (or fold) went the same "
             "way”, not as a measure of effect size.</p>")  # fmt: skip
    return f"""
<section id="summary">
  <div class="head"><p class="eyebrow">Summary</p><h2>What the comparison shows</h2></div>
  <dl class="keys">{kd}</dl>
  <ul class="findings">{lis}</ul>
  {floor}
</section>"""


def section_level1(rep: Report, n: Numbering, l1: dict | None) -> str:
    head = '<div class="head"><p class="eyebrow">Level 1</p><h2>Performance on held-out test peaks</h2></div>'
    if l1 is None:
        return f'<section id="level-1">{head}{problem_html(rep, "Level 1")}</section>'
    if l1.get("state") != "done":
        return f'<section id="level-1">{head}{pending_html(rep, "Level 1")}</section>'
    md, nf, nd = l1["measures"], l1["n_folds"], l1["n_datasets"]
    lead = (
        f'<p class="lead">compare_chrombpnet.py scored both models on ChromBPNet\'s test peaks of each fold '
        f"(about {count(l1['paired']['n_regions'].mean())} peaks per fold) with ChromBPNet's own measures. "
        f"ChromBPNet is scored here as its evaluation scores it, with the full bias-included model. {nf} folds over "
        f"{nd} datasets; the {len(rep.shared)} shared datasets are the ones interpreted below.</p>"
    )
    a = md[md["set"] == "all"].set_index("measure")
    rows = []
    for m in MEASURES:
        r = a.loc[m]
        rows.append(
            [
                h(MEASURE_NAMES[m]),
                "higher" if HIGHER_IS_BETTER[m] else "lower",
                num(r["chrombpnet_median"]),
                num(r["cherimoya_median"]),
                diff_cell(r),
                f"{int(r['cherimoya_better_folds'])}/{nf}",
                f"{int(r['cherimoya_better_datasets'])}/{nd}",
                pval(r["wilcoxon_datasets_p"]),
                pval(r["wilcoxon_folds_p"]),
            ]
        )
    t1 = table_html(n, f"The five measures over all {nd} datasets. Medians over the {nf} folds; the difference is "
                    "Cherimoya minus ChromBPNet, as the median over datasets of each dataset's mean fold difference, "
                    "with its bootstrap 95% CI. p from Wilcoxon signed-rank tests over the dataset means and over the "
                    "folds.",
                    ["Measure", "Better", "ChromBPNet", "Cherimoya", "Difference [95% CI]", "Cherimoya better, folds",
                     "Cherimoya better, datasets", f"p, {nd} datasets", f"p, {nf} folds"], rows, left=2)  # fmt: skip
    sub_rows = []
    for key_ in ("shared", "others"):
        part = md[md["set"] == key_].set_index("measure")
        if part.empty:
            continue
        ndk = int(part["n_datasets"].iloc[0])
        nfk = int(part["n_folds"].iloc[0])
        label = f"{ndk} shared datasets" if key_ == "shared" else f"{ndk} other datasets"
        for m in MEASURES:
            r = part.loc[m]
            sub_rows.append(
                [
                    h(label) if m == MEASURES[0] else "",
                    h(MEASURE_NAMES[m]),
                    f"{num(r['chrombpnet_median'])} → {num(r['cherimoya_median'])}",
                    diff_cell(r),
                    f"{int(r['cherimoya_better_folds'])}/{nfk}",
                    f"{int(r['cherimoya_better_datasets'])}/{ndk}",
                    pval(r["wilcoxon_datasets_p"]),
                ]
            )
    t2 = (
        table_html(
            n,
            "The same for the shared datasets (the ones interpreted at levels 2-5) and for the others "
            "apart. The shared-dataset numbers equal level 1's own statistics.tsv.",
            [
                "Datasets",
                "Measure",
                "ChromBPNet → Cherimoya",
                "Difference [95% CI]",
                "Cherimoya better, folds",
                "Cherimoya better, datasets",
                "p, datasets",
            ],
            sub_rows,
            left=2,
            cls="compact",
        )
        if sub_rows
        else ""
    )
    rd = l1["residuals"]
    r_all = rd[rd["set"] == "all"].sort_values("decile")
    rrows = [
        [
            f"{int(r['decile'])}",
            num(r["observed_mean_median"], 2),
            num(r["rmse_chrombpnet_median"]),
            num(r["rmse_cherimoya_median"]),
            f'{num(r["rmse_diff_median"], 3, True)}<span class="sub">{ci(r["ci95_lo"], r["ci95_hi"], 3, True)}</span>',
            f"{int(r['cherimoya_lower_rmse_datasets'])}/{int(r['n_datasets'])}",
            pval(r["wilcoxon_p"]),
            num(r["mean_residual_chrombpnet_median"], 2, True),
            num(r["mean_residual_cherimoya_median"], 2, True),
        ]
        for _, r in r_all.iterrows()
    ]
    t3 = table_html(
        n,
        "Residuals by decile of observed log(1 + counts), each dataset's test peaks of all folds pooled. "
        "Medians over datasets; the RMSE difference is Cherimoya minus ChromBPNet (below 0: Cherimoya "
        "closer), with its bootstrap CI and the Wilcoxon p over datasets.",
        [
            "Decile",
            "Observed mean",
            "RMSE ChromBPNet",
            "RMSE Cherimoya",
            "Difference [95% CI]",
            "Cherimoya lower",
            "p",
            "Mean residual ChromBPNet",
            "Mean residual Cherimoya",
        ],
        rrows,
    )
    pd_ = l1["per_dataset"]
    drows = [
        [
            h(rep.labels.get(r["dataset"], r["dataset"])),
            "shared" if r["shared"] else "",
            f"{int(r['n_folds'])}",
            num(r["counts_pearson_chrombpnet"]),
            num(r["counts_pearson_cherimoya"]),
            num(r["counts_pearson_diff"], 3, True),
            num(r["median_jsd_chrombpnet"]),
            num(r["median_jsd_cherimoya"]),
            num(r["median_jsd_diff"], 3, True),
            num(r["counts_mse_diff"], 3, True),
        ]
        for _, r in pd_.iterrows()
    ]
    t4 = table_html(n, "Per dataset, the means over its folds (prefix amsc_ dropped).",
                    ["Dataset", "Set", "Folds", "Counts r ChromBPNet", "Counts r Cherimoya", "Difference",
                     "Median JSD ChromBPNet", "Median JSD Cherimoya", "Difference", "MSE difference"], drows, left=2,
                    cls="compact")  # fmt: skip
    f1 = figure_html(
        rep,
        n,
        "level1_scatter",
        "Each dot is one test fold (dataset × fold); filled dots are the "
        f"{len(rep.shared)} shared datasets, open dots the others. Above the diagonal Cherimoya's value is "
        "higher; for MSE and JSD lower is better, so Cherimoya is better below it.",
    )
    f2 = figure_html(
        rep,
        n,
        "level1_residuals",
        "Left: mean residual (predicted minus observed log counts) of each "
        "model by decile of observed counts, median over datasets with the interquartile range. Right: the "
        "RMSE difference per dataset (grey) and its median over datasets with the bootstrap 95% CI.",
    )
    return f"""
<section id="level-1">
  {head}
  {lead}
  {pending_html(rep, "Level 1")}
  {f1}
  {t1}
  {t2}
  {f2}
  {t3}
  {details(f"Per dataset ({len(pd_)} datasets)", t4)}
</section>"""


def l2_cell(rep: Report, w: str, m: str, quantity: str, d: int = 3) -> str:
    return est_ci(s(rep, "2", f"attributions, {w}, {m}", quantity), d)


def l2_test_cell(rep: Report, w: str, m: str, quantity: str, d: int = 4) -> str:
    return test_cell(s(rep, "2", f"attributions, {w}, {m}", quantity), d)


def section_level2(rep: Report, n: Numbering, l2: dict | None) -> str:
    head = '<div class="head"><p class="eyebrow">Level 2</p><h2>Per-peak attributions</h2></div>'
    if l2 is None:
        return f'<section id="level-2">{head}{problem_html(rep, "Level 2")}</section>'
    if l2.get("state") != "done":
        return f'<section id="level-2">{head}{pending_html(rep, "Level 2")}</section>'
    joins = l2["joins"]
    lead = (f"<p class=\"lead\">The counts-head DeepLIFT/SHAP scores of both models on every peak both attributed "
            f"({count(joins['compared_peaks'].min())}–{count(joins['compared_peaks'].max())} per dataset), joined by "
            "peak name. The projected score (the hypothetical score of the base present) is compared per peak within "
            "250 bp and 500 bp of the summit. <b>A</b> compares the two models fold by fold, <b>B</b> their fold "
            "averages, <b>C</b> each model's fold against the mean of its other four folds (the ceiling), <b>D</b> "
            "agreement on the fold that held the peak's chromosome out.</p>")  # fmt: skip
    blocks = []
    for wi, w in enumerate(l2["windows"]):
        rows = []
        for name in ("cross_fold", "cross_average", "loo_chrombpnet", "loo_cherimoya", "cross_heldout",
                     "heldout_chrombpnet", "heldout_cherimoya"):  # fmt: skip
            rows.append([h(QUANTITY_NAMES[name])] + [l2_cell(rep, w, m, name) for m in L2_METRICS])
        t_levels = table_html(n, f"Agreement within {int(w[1:]) // 2} bp of the summit ({w[1:]}-bp window): the median "
                              f"over datasets [bootstrap 95% CI] of each dataset's median over peaks (Jaccard: mean "
                              f"over peaks).", ["Quantity"] + [METRIC_NAMES[m] for m in L2_METRICS], rows)  # fmt: skip
        trows = []
        for x, y, label in (
            ("cross_fold", "loo_chrombpnet", "A − ChromBPNet ceiling"),
            ("cross_fold", "loo_cherimoya", "A − Cherimoya ceiling"),
            ("cross_average", "loo_chrombpnet", "B − ChromBPNet ceiling"),
            ("cross_average", "loo_cherimoya", "B − Cherimoya ceiling"),
            ("loo_cherimoya", "loo_chrombpnet", "Cherimoya ceiling − ChromBPNet ceiling"),
            ("heldout_chrombpnet", "train_chrombpnet", "ChromBPNet: held-out − training folds"),
            ("heldout_cherimoya", "train_cherimoya", "Cherimoya: held-out − training folds"),
            ("cross_heldout", "cross_train", "Cross-model: held-out − training folds"),
            ("heldout_cherimoya", "heldout_chrombpnet", "Held-out fold: Cherimoya − ChromBPNet"),
        ):
            trows.append([h(label)] + [l2_test_cell(rep, w, m, f"{x} - {y}") for m in L2_METRICS])
        for model in MODELS:
            cells = [test_cell(s(rep, "2", f"attributions, {w}, {m}", f"cross_fold - loo_{model}", "folds"), 4)
                     if m != "jaccard_mean" else "—" for m in L2_METRICS]  # fmt: skip
            trows.append([h(f"A − {NAMES[model]} ceiling, over the dataset × fold pairs")] + cells)
        t_tests = table_html(n, f"Paired differences ({w[1:]}-bp window), each over the datasets: the median "
                             "difference, its bootstrap 95% CI, how many datasets were above 0, and the Wilcoxon p.",
                             ["Contrast"] + [METRIC_NAMES[m] for m in L2_METRICS], trows)  # fmt: skip
        if wi == 0:
            blocks.append(
                figure_html(
                    rep,
                    n,
                    "level2_agreement",
                    f"Agreement within {int(w[1:]) // 2} bp of the "
                    "summit. Dots are datasets (cross-model quantities in ink, each model's ceiling in "
                    "its colour); bars are medians over datasets, the thin line beside each its "
                    "bootstrap 95% CI. Top-50 Jaccard uses per-peak means, since its medians take "
                    "only a few values (k/(100 − k)).",
                )
                + t_levels
                + t_tests
            )
        else:
            blocks.append(details(f"The same at {w[1:]} bp", t_levels + t_tests))
    w = l2["windows"][0]
    st = l2["strata"]
    srows = []
    for stratum, levels, label in (("signal_decile", ("1", "5", "10"), "signalValue decile"),
                                   ("gc_decile", ("1", "5", "10"), "GC decile"),
                                   ("role", ("train", "valid", "test"), "chromosome's role in the fold")):  # fmt: skip
        for lev in levels:
            cells = []
            for name in ("cross_fold", "cross_average", "loo_chrombpnet", "loo_cherimoya"):
                r = st[(st["stratum"] == stratum) & (st["level"] == lev) & (st["window"] == w)
                       & (st["metric"] == "pearson") & (st["quantity"] == name)]  # fmt: skip
                cells.append(
                    f"{num(r['median'].iloc[0])}"
                    f'<span class="sub">{ci(r["ci95_lo"].iloc[0], r["ci95_hi"].iloc[0])}</span>'
                    if len(r)
                    else "—"
                )
            srows.append([h(f"{label} {lev}")] + cells)
    t_strata = table_html(
        n,
        f"Pearson r by stratum ({w[1:]}-bp window), median over datasets [95% CI]. Deciles are "
        "taken over each dataset's joined peaks; B has no role stratum (fold averages have no "
        "held-out chromosomes).",
        ["Stratum", "A same fold", "B fold averages", "C ChromBPNet", "C Cherimoya"],
        srows,
    )
    sig = s(rep, "2", f"attribution strata, {w}, pearson", "signal_decile 10 - 1: cross_average")
    gc = s(rep, "2", f"attribution strata, {w}, pearson", "gc_decile 10 - 1: cross_average")

    def stv(stratum: str, lev: str, name: str) -> float:
        r = st[(st["stratum"] == stratum) & (st["level"] == lev) & (st["window"] == w) & (st["metric"] == "pearson")
               & (st["quantity"] == name)]  # fmt: skip
        return float(r["median"].iloc[0]) if len(r) else NAN

    strata_text = ""
    if sig and gc:
        gap = max((max(stv("signal_decile", str(k), "loo_chrombpnet"), stv("signal_decile", str(k), "loo_cherimoya"))
                   - stv("signal_decile", str(k), "cross_average")) for k in range(1, 11))  # fmt: skip
        strata_text = (
            f"<p>Agreement rises with peak strength. On fold averages (B), Pearson r is "
            f"{num(abs(sig['estimate']), 3)} {'higher' if sig['estimate'] > 0 else 'lower'} in the top "
            f"signalValue decile than in the bottom one ({int(sig['higher'])}/{int(sig['n'])} datasets, p "
            f"{pval(sig['p'])}); both ceilings move with it, and B stays within {num(gap, 3)} of the higher "
            f"ceiling in every decile. Single folds (A) change most, from "
            f"{num(stv('signal_decile', '1', 'cross_fold'))} to {num(stv('signal_decile', '10', 'cross_fold'))}. "
            f"Agreement {'falls' if gc['estimate'] < 0 else 'rises'} with GC content: B is "
            f"{num(abs(gc['estimate']), 3)} {'lower' if gc['estimate'] < 0 else 'higher'} in the most GC-rich "
            f"decile than in the least (p {pval(gc['p'])}), and there the ceilings part: ChromBPNet "
            f"{num(stv('gc_decile', '10', 'loo_chrombpnet'))}, Cherimoya "
            f"{num(stv('gc_decile', '10', 'loo_cherimoya'))}.</p>"
        )
    mrows = []
    for m in MODELS:
        mrows.append(
            [key(m)]
            + [
                est_ci(s(rep, "2", f"attribution magnitude, {w}", f"{NAMES[m]}: {what}"), 4)
                for what in (
                    "median per-peak L2 norm, fold average",
                    "median per-peak sum, fold average",
                    "median per-peak L2 norm, single folds (mean over folds)",
                    "L2 norm of the mean of 4 folds / of one fold",
                )
            ]
        )
    t_mag = table_html(
        n,
        f"Attribution magnitude per model ({w[1:]}-bp window): medians over datasets [95% CI] of each "
        "dataset's median per peak. The two models' scales differ, so the columns describe each model; "
        "the last column shows how much averaging four folds shrinks the norm (fold-to-fold noise).",
        [
            "Model",
            "L2 norm, fold average",
            "Sum, fold average",
            "L2 norm, single fold",
            "Mean of 4 folds / one fold (L2)",
        ],
        mrows,
    )
    ratio = s(rep, "2", f"attribution magnitude, {w}", "L2 norm of the fold averages, Cherimoya / ChromBPNet")
    f16 = l2.get("float16")
    f16_text = ""
    if f16 is not None and len(f16):
        ds16 = sorted(set(f16["dataset"]))
        f16_text = (
            f"<p>Float16 check on {h(', '.join(rep.labels.get(x, x) for x in ds16))}: computed again from "
            f"Cherimoya's float16 h5 instead of its float32 npz, the {len(f16)} medians of this level move by at "
            f"most {num(f16['median_diff'].abs().max(), 6)}.</p>"
        )
    return f"""
<section id="level-2">
  {head}
  {lead}
  {pending_html(rep, "Level 2")}
  {"".join(blocks)}
  {
        figure_html(
            rep,
            n,
            "level2_strata",
            "Pearson r of the projected scores by decile of peak signal (left) and of GC "
            "content (right): medians over datasets with bootstrap 95% CIs. The dashed line is A (single folds), "
            "the solid ink line B (fold averages).",
        )
    }
  {strata_text}
  {t_strata}
  {t_mag}
  {
        f"<p>The fold-averaged maps of Cherimoya have {num(ratio['estimate'], 2)} times the L2 norm of ChromBPNet's "
        f"(median "
        f"over datasets, {ci(ratio['ci95_lo'], ratio['ci95_hi'], 2)}).</p>"
        if ratio
        else ""
    }
  {f16_text}
</section>"""


def section_level3(rep: Report, n: Numbering, l3: dict | None) -> str:
    head = '<div class="head"><p class="eyebrow">Level 3</p><h2>TF-MoDISco patterns and seqlets</h2></div>'
    if l3 is None:
        return f'<section id="level-3">{head}{problem_html(rep, "Level 3")}</section>'
    if l3.get("state") != "done":
        return f'<section id="level-3">{head}{pending_html(rep, "Level 3")}</section>'
    per = l3["per"]
    lead = (
        '<p class="lead">Both models ran the same TF-MoDISco (08.0, 500-bp window, same flags) on their '
        "fold-average attributions, and the same QC-scale run (window 400, 5,000 seqlets) on each fold's 30K-peak "
        "subsample. A fold-average pattern is matched when the joint compendium puts a pattern of the other "
        "model's same dataset in its cluster (the primary rule), or when tomtom-lite finds it a partner among the "
        "other model's patterns at q ≤ 0.05 with the MotifCompendium database as decoys (secondary). Seqlets are "
        "placed in the genome through each model's own bed and count as shared at ≥ 50% reciprocal overlap.</p>"
    )
    fam_m, fam_d = "TF-MoDISco, per model", "TF-MoDISco, Cherimoya - ChromBPNet"

    def lvl(col: str, d: int = 3) -> list[str]:
        return [est_ci(s(rep, "3", fam_m, f"{NAMES[m]}: {col}"), d) for m in MODELS]

    def tst(label: str, d: int = 3) -> str:
        return test_cell(s(rep, "3", fam_d, label), d)

    totals = {m: per[per["model"] == m] for m in MODELS}
    rows = [
        ["Patterns, all datasets (pos / neg)"] + [f"{int(t['patterns'].sum())} ({int(t['pos'].sum())} / "
                                                  f"{int(t['neg'].sum())})" for t in totals.values()] + [""],
        ["Seqlets, all datasets"] + [count(t["seqlets"].sum()) for t in totals.values()] + [""],
        ["Matched by joint cluster, share of patterns"] + lvl("shared_cluster_count")
        + [tst("shared by cluster, by count: Cherimoya - ChromBPNet")],
        ["Matched by joint cluster, share of seqlets"] + lvl("shared_cluster_seqlets")
        + [tst("shared by cluster, by seqlets: Cherimoya - ChromBPNet")],
        ["Matched by tomtom, share of patterns"] + lvl("shared_tomtom_count")
        + [tst("shared by tomtom, by count: Cherimoya - ChromBPNet")],
        ["Matched by tomtom, share of seqlets"] + lvl("shared_tomtom_seqlets", 4)
        + [tst("shared by tomtom, by seqlets: Cherimoya - ChromBPNet", 4)],
        ["Seqlets overlapped by the other model's"] + lvl("seqlet_overlap")
        + [tst("seqlets overlapped by the other model's seqlets: Cherimoya - ChromBPNet")],
        ["Seqlets overlapped by matched partners'"] + lvl("overlap_matched")
        + [tst("seqlets overlapped by matched partners' seqlets: Cherimoya - ChromBPNet")],
        ["Tn5/DNase-flagged patterns per dataset"] + lvl("bias_flagged", 1)
        + [tst("Tn5/DNase-flagged patterns: Cherimoya - ChromBPNet", 1)],
    ]  # fmt: skip
    t1 = table_html(
        n,
        "Fold-average patterns, one value per dataset: medians over datasets [95% CI], and the paired "
        "difference Cherimoya − ChromBPNet (median, CI, datasets above 0, Wilcoxon p). Partner overlap is "
        "seqlet-weighted over the patterns that have a partner.",
        ["Quantity", "ChromBPNet", "Cherimoya", "Cherimoya − ChromBPNet"],
        [[h(r[0])] + r[1:] for r in rows],
    )
    unm = {m: totals[m] for m in MODELS}
    unmatched_text = (f"<p>By the cluster rule, {int(unm['chrombpnet']['cluster_unmatched'].sum())} ChromBPNet and "
                      f"{int(unm['cherimoya']['cluster_unmatched'].sum())} Cherimoya patterns are unmatched, but "
                      f"{int(unm['chrombpnet']['cluster_unmatched_with_tomtom'].sum())} and "
                      f"{int(unm['cherimoya']['cluster_unmatched_with_tomtom'].sum())} of them still have a tomtom "
                      f"partner; by tomtom only {int(unm['chrombpnet']['tomtom_unmatched'].sum())} and "
                      f"{int(unm['cherimoya']['tomtom_unmatched'].sum())} are unmatched. The cluster rule is strict: "
                      "it asks for the partner in the same compendium cluster, and at threshold 0.95 the joint "
                      "compendium keeps close variants of a motif in separate clusters (many merge at 0.90; level "
                      "4).</p>")  # fmt: skip
    famq = "TF-MoDISco QC, cross-model - ceiling"
    qrows = []
    for m in MODELS:
        qrows.append([key(m), est_ci(s(rep, "3", fam_m, f"{NAMES[m]}: qc_cross")),
                      est_ci(s(rep, "3", fam_m, f"{NAMES[m]}: qc_within")),
                      test_cell(s(rep, "3", famq, f"QC: {m} patterns matched across models - within {m} (ceiling)")),
                      test_cell(s(rep, "3", famq, f"QC: {m} patterns matched across models - within {m}, per fold",
                                  "folds")),
                      est_ci(s(rep, "3", fam_m, f"{NAMES[m]}: qc_seqlet_overlap")),
                      est_ci(s(rep, "3", fam_m, f"{NAMES[m]}: qc_patterns_per_fold"), 1)])  # fmt: skip
    t2 = table_html(n, "QC scale (each fold's 30K-peak subsample): the share of a fold's patterns matched by tomtom in "
                    "the other model's same fold, against the model's own fold-to-fold matching (every pair of its "
                    "folds; the ceiling). Medians over datasets [95% CI]; the differences are over the 10 datasets "
                    "and over the 50 folds (that CI resamples datasets with their folds). Seqlet overlap has no "
                    "ceiling, because the subsamples differ between folds.",
                    ["Model", "Across the models", "Fold to fold (ceiling)", "Across − ceiling, datasets",
                     "Across − ceiling, folds", "Seqlets overlapped across", "Patterns per fold"], qrows)  # fmt: skip
    return f"""
<section id="level-3">
  {head}
  {lead}
  {pending_html(rep, "Level 3")}
  {
        figure_html(
            rep,
            n,
            "level3_patterns",
            "Each dot is a dataset; lines join one dataset's two values. Left: share of "
            "each model's fold-average patterns matched in the other model, by the cluster rule and by tomtom. "
            "Middle: at QC scale, each model's patterns matched across the models (filled) against its fold-to-fold "
            "ceiling (open). Right: share of each model's seqlets overlapped by the other model's seqlets. Bars are "
            "medians, thin lines bootstrap 95% CIs.",
        )
    }
  {t1}
  {unmatched_text}
  {t2}
  <p>The fold-average scale has no ceiling (one fold average per model), so cross-model matching there can only be
  read against the QC-scale ceiling. The bias control (level 3c) follows the hits, in its own section below.</p>
</section>"""


def section_level4(rep: Report, n: Numbering, l4: dict | None) -> str:
    head = '<div class="head"><p class="eyebrow">Level 4</p><h2>Motif compendia</h2></div>'
    if l4 is None:
        return f'<section id="level-4">{head}{problem_html(rep, "Level 4")}{pending_html(rep, "Level 4")}</section>'
    if l4.get("state") != "done":
        return f'<section id="level-4">{head}{pending_html(rep, "Level 4")}</section>'
    tags, lfl = l4["tags"], l4["lfl"]
    ag, bs = l4["agreement"], l4["by_sign"]
    lead = (f"<p class=\"lead\">09.0 (MotifCompendium) clustered each model's fold-average patterns of the "
            f"{len(rep.shared)} shared datasets into a compendium per model ({h(tags['cbp'])}, {h(tags['cheri'])}) and "
            f"both models' patterns together into a joint one ({h(tags['joint'])}, threshold 0.95). Like-for-like "
            "clusters are paired by reciprocal best tomtom-lite match of their CWMs; every cluster has some best match "
            "at q ≤ 0.05 (there are no decoys), so only the reciprocal pairs say something.</p>")  # fmt: skip
    cl = lfl.get("clusters", {})
    ann, fam = lfl.get("annotations", {}), lfl.get("families", {})

    def size(v) -> str:
        return count(len(v)) if isinstance(v, list | tuple | set) else count(v)

    n_recip = int(lfl.get("matched", {}).get("reciprocal_pairs", sum(ag.values())))
    rows = [["Clusters", count(cl.get("cbp")), count(cl.get("cheri"))],
            ["Reciprocal best pairs", f"{n_recip} ({bs.get('pos', 0)} positive, {bs.get('neg', 0)} negative)", ""],
            ["… with the same annotation", count(ag.get("same annotation", 0)), ""],
            ["… with different annotations", count(ag.get("different annotation", 0)), ""],
            ["… with an unannotated side", count(ag.get("unannotated", 0)), ""],
            ["Annotations (shared / ChromBPNet only / Cherimoya only)",
             f"{size(ann.get('shared'))} / {size(ann.get('only_a'))} / {size(ann.get('only_b'))}",
             f"Jaccard {num(ann.get('jaccard'))}"],
            ["TF families (shared / ChromBPNet only / Cherimoya only)",
             f"{size(fam.get('shared'))} / {size(fam.get('only_a'))} / {size(fam.get('only_b'))}",
             f"Jaccard {num(fam.get('jaccard'))}"]]  # fmt: skip
    only = ""
    if isinstance(ann.get("only_a"), list) and isinstance(ann.get("only_b"), list):
        only = details("Annotations of one model's compendium only",
                       f"<p><b>ChromBPNet only:</b> {h(', '.join(ann['only_a']))}.</p>"
                       f"<p><b>Cherimoya only:</b> {h(', '.join(ann['only_b']))}.</p>")  # fmt: skip
    t1 = table_html(n, f"Like for like: {h(tags['cbp'])} (ChromBPNet) against {h(tags['cheri'])} (Cherimoya). "
                    "Annotations are each cluster's most frequent MotifCompendium annotation of score ≥ 0.7; a family "
                    "is the annotation without its _&lt;n&gt; suffix.",
                    ["", h(tags["cbp"]), h(tags["cheri"])], [[h(r[0])] + r[1:] for r in rows])  # fmt: skip
    c = l4["classes"]
    crow = []
    for _, r in c.iterrows():
        crow.append(
            [
                h(f"{r['threshold']} · {CLASS_NAMES[r['class']]}"),
                count(r["clusters"]),
                count(r["single_pattern_clusters"]),
                count(r["patterns_chrombpnet"]),
                count(r["patterns_cherimoya"]),
                pct(r["seqlet_share_chrombpnet"]),
                pct(r["seqlet_share_cherimoya"]),
                count(r["bias_clusters"]) if "bias_clusters" in r and isnum(r["bias_clusters"]) else "—",
            ]
        )
    t2 = table_html(
        n,
        f"The joint compendium's clusters by whose patterns they hold, at threshold 0.95 "
        f"({h(tags['joint'])}) and rebuilt at 0.90 ({h(l4['sensitivity_tag'])}). Seqlet shares are of each "
        "model's seqlets over all clusters. Bias clusters: at least half of the cluster's patterns flagged "
        "Tn5/DNase at level 3 (0.95 only).",
        [
            "Threshold · patterns of",
            "Clusters",
            "Single-pattern",
            "ChromBPNet patterns",
            "Cherimoya patterns",
            "ChromBPNet seqlets",
            "Cherimoya seqlets",
            "Bias clusters",
        ],
        crow,
    )
    srows = []
    for thr in ("0.95", "0.90"):
        fam_ = f"compendium shares, threshold {thr}"
        for what in ("patterns", "seqlets"):
            cells = [est_ci(s(rep, "4", fam_, f"{NAMES[m]}: share of {what} in clusters of both models"), 3)
                     for m in MODELS]  # fmt: skip
            if all(x == "NA" for x in cells):
                continue
            srows.append(
                [h(f"{thr} · {what}")]
                + cells
                + [test_cell(s(rep, "4", fam_, f"share of {what} in clusters of both models: Cherimoya - ChromBPNet"))]
            )
    t3 = (
        table_html(
            n,
            "Per dataset, the share of each model's patterns (and seqlets) in clusters that hold both "
            "models' patterns: medians over datasets [95% CI] and the paired difference.",
            ["Threshold · share of", "ChromBPNet", "Cherimoya", "Cherimoya − ChromBPNet"],
            srows,
        )
        if srows
        else ""
    )
    sup = l4.get("supplement")
    if sup:
        sup_html = note_html(
            "done",
            "Supplement",
            f"<p><b>{h(sup.get('compendium'))}</b>: {count(sup.get('clusters'))} "
            f"clusters, {count(sup.get('matched_in_shared'))} matched in {h(tags['cheri'])}, "
            f"{count(sup.get('not_in_shared'))} not; annotations it adds: "
            f"{h(', '.join(sup.get('annotations_added') or []) or 'none')}.</p>",
        )
    else:
        sup_html = pending_html(rep, "Level 4")
    return f"""
<section id="level-4">
  {head}
  {lead}
  {
        figure_html(
            rep,
            n,
            "level4_compendium",
            "Left: clusters of the joint compendium holding patterns of both models, "
            "of ChromBPNet only or of Cherimoya only (solid: threshold 0.95; light: 0.90). Right: per dataset, the "
            "share of each model's patterns in clusters of both models; lines join a dataset's two values, bars are "
            "medians with bootstrap 95% CIs.",
        )
    }
  {t1}
  {only}
  {t2}
  {t3}
  <h3>Supplement: what the other datasets add</h3>
  {sup_html}
</section>"""


def section_level5(rep: Report, n: Numbering, l5: dict | None) -> str:
    head = '<div class="head"><p class="eyebrow">Level 5</p><h2>Fi-NeMo hits</h2></div>'
    if l5 is None:
        return f'<section id="level-5">{head}{problem_html(rep, "Level 5")}{pending_html(rep, "Level 5")}</section>'
    if l5.get("state") != "done":
        return f'<section id="level-5">{head}{pending_html(rep, "Level 5")}</section>'
    tot = l5["totals"]
    lead = ("<p class=\"lead\">Both models' fold-average attributions were scanned by Fi-NeMo (10.0) with the same "
            "vocabulary, the joint compendium (the same compiled h5 for both). Only peaks both beds hold are "
            "compared. A hit is matched when the other model has a hit of the same motif on the same strand with its "
            "centre within 5 bp (0 and 10 bp are also reported). Precision is the share of Cherimoya's hits matched; "
            "recall the share of ChromBPNet's hits matched. There is no within-model ceiling here: Fi-NeMo ran on "
            "fold averages only.</p>")  # fmt: skip
    fam = "Fi-NeMo hits, joint vocabulary"
    rows = [[key(m), count(tot[f"hits_{m}"].sum()), est_ci(s(rep, "5", fam, f"{NAMES[m]}: hits per shared peak"), 2)]
            for m in MODELS]  # fmt: skip
    diff = s(rep, "5", fam, "hits per shared peak: ChromBPNet - Cherimoya")
    ratio = s(rep, "5", fam, "hits, ChromBPNet / Cherimoya")
    t1 = table_html(
        n,
        f"Hits on the shared peaks of the {len(tot)} datasets: in total, and per peak (median over "
        f"datasets [95% CI]). ChromBPNet − Cherimoya per peak: {test_cell(diff, 2)}; ChromBPNet / Cherimoya: "
        f"{est_ci(ratio, 3)}.",
        ["Model", "Hits", "Hits per peak"],
        rows,
    )
    arow = []
    for tol in (0, 5, 10):
        arow.append(
            [f"{tol} bp", est_ci(s(rep, "5", fam, f"precision_{tol}bp")), est_ci(s(rep, "5", fam, f"recall_{tol}bp"))]
        )
    t2 = table_html(
        n,
        "Agreement of the hits, medians over datasets [95% CI]. Per peak, the Pearson correlation of the "
        f"two models' motif-count vectors has median {est_ci(s(rep, '5', fam, 'median_count_correlation'))}; "
        f"on matched hits the Spearman correlation of the hit coefficients is "
        f"{est_ci(s(rep, '5', fam, 'coefficient_spearman'))}. Precision − recall at 5 bp: "
        f"{test_cell(s(rep, '5', fam, 'precision_5bp - recall_5bp'))}.",
        ["Tolerance", "Precision (of Cherimoya's hits)", "Recall (of ChromBPNet's hits)"],
        arow,
    )
    crow = []
    for by, value, label in (
        ("class", "shared", "clusters of both models"),
        ("class", "chrombpnet_only", "ChromBPNet-only clusters"),
        ("class", "cherimoya_only", "Cherimoya-only clusters"),
        ("sign", "pos", "positive clusters"),
        ("sign", "neg", "negative clusters"),
    ):
        f = f"Fi-NeMo hits, {by} {value}"
        crow.append(
            [
                h(label),
                est_ci(s(rep, "5", f, "share_of_chrombpnet_hits")),
                est_ci(s(rep, "5", f, "share_of_cherimoya_hits")),
                est_ci(s(rep, "5", f, "precision_5bp")),
                est_ci(s(rep, "5", f, "recall_5bp")),
                est_ci(s(rep, "5", f, "motif_median_peak_jaccard")),
                est_ci(s(rep, "5", f, "motif_median_coefficient_spearman")),
            ]
        )
    t3 = table_html(n, "By the class of the motif's joint cluster and by its sign (from joint_clusters.tsv, not the "
                    "motif's name prefix): the share of each model's hits, precision and recall at 5 bp pooled over "
                    "the class's hits, and the per-motif medians of the peak-set Jaccard and of the coefficient "
                    "Spearman correlation. Medians over datasets [95% CI].",
                    ["Motifs of", "Share of ChromBPNet's hits", "Share of Cherimoya's hits", "Precision", "Recall",
                     "Peak Jaccard, per motif", "Coefficient ρ, per motif"], crow)  # fmt: skip
    cc = [s(rep, "5", "Fi-NeMo hits, by cluster class", q) for q in (
        "share of hits on Cherimoya-only clusters: Cherimoya - ChromBPNet",
        "precision_5bp: shared clusters - Cherimoya-only clusters")]  # fmt: skip
    class_text = ""
    if all(cc):
        class_text = (
            f"<p>Cherimoya puts a {num(cc[0]['estimate'], 3)} larger share of its hits on Cherimoya-only motifs "
            f"than ChromBPNet does ({int(cc[0]['higher'])}/{int(cc[0]['n'])} datasets, p {pval(cc[0]['p'])}), "
            f"and those hits are the ones ChromBPNet reproduces least: precision is "
            f"{num(abs(cc[1]['estimate']), 3)} {'lower' if cc[1]['estimate'] > 0 else 'higher'} on "
            f"Cherimoya-only motifs than on motifs of both models ({int(cc[1]['higher'])}/{int(cc[1]['n'])} "
            f"datasets, p {pval(cc[1]['p'])}).</p>"
        )
    top = l5["top"].head(8)
    trow = [
        [
            h(r["annotation"] or r["motif"]),
            h(r["motif"]),
            h(CLASS_NAMES.get(r["class"], r["class"] or "")),
            h(r["posneg"] or ""),
            count(r["hits_chrombpnet"]),
            count(r["hits_cherimoya"]),
            num(r["precision_5bp"]),
            num(r["recall_5bp"]),
        ]
        for _, r in top.iterrows()
    ]
    total_hits = float(l5["top"]["hits_total"].sum())
    top2 = float(l5["top"]["hits_total"].head(2).sum())
    t4 = table_html(
        n,
        f"The largest motifs, pooled over datasets. The two largest hold {pct(top2 / total_hits)} of all "
        "hits, which is why the pooled precision and recall above are shown next to per-motif medians.",
        ["Annotation", "Motif", "Cluster of", "Sign", "ChromBPNet hits", "Cherimoya hits", "Precision", "Recall"],
        trow,
        left=4,
        cls="compact",
    )
    mr = l5["reports"]
    t5 = ""
    if mr is not None and not mr.empty:
        fr = "Fi-NeMo motif reports"
        rows5 = [
            [
                key(m),
                est_ci(s(rep, "5", fr, f"{NAMES[m]}: median hit-seqlet CWM similarity")),
                est_ci(s(rep, "5", fr, f"{NAMES[m]}: share of hits restricted to TF-MoDISco's regions")),
            ]
            for m in MODELS
        ]
        dd = s(rep, "5", fr, "per-motif hit-seqlet CWM similarity: Cherimoya - ChromBPNet (median per dataset)")
        t5 = table_html(n, "11.0's motif reports side by side: the cosine similarity of the hits' mean contribution "
                        "scores to the seqlets' (per motif, median over motifs with a value for both models), and the "
                        "share of hits inside TF-MoDISco's input regions. Medians over datasets [95% CI]. Cherimoya − "
                        f"ChromBPNet in CWM similarity: {test_cell(dd)}.",
                        ["Model", "Hit-seqlet CWM similarity", "Hits in TF-MoDISco's regions"], rows5)  # fmt: skip
    drop = tot["chrombpnet_hits_on_dropped_peaks"]
    prod = l5.get("production")
    if prod is not None and not prod.empty:
        prows = [
            [
                h(rep.labels.get(r["dataset"], r["dataset"])),
                count(r.get("hits_chrombpnet")),
                count(r.get("hits_cherimoya")),
                num(r.get("precision_5bp")),
                num(r.get("recall_5bp")),
            ]
            for _, r in prod.iterrows()
        ]
        prod_html = table_html(
            n,
            "Production hits (cbp10 against cheri39, through level 4's reciprocal cluster pairs).",
            ["Dataset", "ChromBPNet hits", "Cherimoya hits", "Precision 5 bp", "Recall 5 bp"],
            prows,
        )
        prod_html += pending_html(rep, "Level 5")
    else:
        prod_html = pending_html(rep, "Level 5")
    return f"""
<section id="level-5">
  {head}
  {lead}
  {
        figure_html(
            rep,
            n,
            "level5_hits",
            "Left: per dataset (dots), the share of Cherimoya's hits matched by ChromBPNet "
            "(precision, vermillion) and of ChromBPNet's hits matched by Cherimoya (recall, blue), over all motifs "
            "and by the class of the motif's joint cluster; bars are medians, thin lines bootstrap 95% CIs. Right: "
            "hits per 10,000 shared peaks of each motif, mean over datasets, on log axes, coloured by whose patterns "
            "its cluster holds; the three motifs with the most hits are numbered, and motifs without hits for a "
            "model sit at 0.01.",
        )
    }
  {t1}
  {t2}
  {t3}
  {class_text}
  {t4}
  {t5}
  <p>ChromBPNet also has {count(drop.min())}–{count(drop.max())} hits per dataset on the peaks Cherimoya dropped
  (windows with an N); they are left out of every number here.</p>
  <h3>Secondary: production hits</h3>
  {prod_html}
</section>"""


def section_bias(rep: Report, n: Numbering, bc: dict | None, l3: dict | None, l4: dict | None) -> str:
    head = '<div class="head"><p class="eyebrow">Level 3c</p><h2>Bias control</h2></div>'
    if bc is None:
        return f'<section id="bias-control">{head}{problem_html(rep, "Bias control")}</section>'
    if bc.get("state") != "done":
        return f'<section id="bias-control">{head}<p>No bias control has been compared yet.</p></section>'
    lead = ("<p class=\"lead\">ChromBPNet's attributions come from its bias-corrected model, while Cherimoya has no "
            "bias model, so Tn5 or DNase I sequence preference can surface as motifs on Cherimoya's side only. Two "
            "checks: the motif-database flag of level 3 (a pattern whose best match in chrombpnet's motifs.meme is a "
            "TN5_ or DNASE_ entry at q ≤ 0.05), and a control that attributes ChromBPNet's bias-included model and "
            "asks which of Cherimoya's unmatched patterns it also finds.</p>")  # fmt: skip
    ctrl = []
    for c in bc["controls"]:
        state = "done" if c["state"] == "done" else "problem"
        what = (f"<p><b>{h(c['name'])}</b>: {h(c['head'])} head of the bias-included model, "
                f"{h(rep.labels.get(c.get('dataset'), c.get('dataset')))} fold {h(c.get('fold'))}"
                + (f" — {h(c['reason'])}" if c.get("reason") else "") + "</p>")  # fmt: skip
        ctrl.append(note_html(state, "Used" if state == "done" else "Refused", what))
    per = l3["per"] if l3 else None
    t1 = ""
    if per is not None:
        rows = []
        for m in MODELS:
            p = per[per["model"] == m]
            rows.append(
                [
                    key(m),
                    f"{int(p['bias_flagged'].sum())} of {int(p['patterns'].sum())}",
                    f"{int(p['bias_flagged'].min())}–{int(p['bias_flagged'].max())}",
                    count(p["bias_seqlets"].sum()),
                    f"{int(p['qc_bias_flagged'].sum())} of {int(p['qc_patterns'].sum())}",
                ]
            )
        t = s(rep, "3", "TF-MoDISco, Cherimoya - ChromBPNet", "Tn5/DNase-flagged patterns: Cherimoya - ChromBPNet")
        tq = s(rep, "3", "TF-MoDISco QC, Cherimoya - ChromBPNet", "QC: Tn5/DNase-flagged patterns per fold", "folds")
        jt = ""
        if l4 is not None and l4.get("state") == "done":
            cc = l4["classes"]
            c95 = cc[cc["threshold"] == "0.95"].set_index("class")
            if "bias_clusters" in c95:
                jt = (
                    f" In the joint compendium, {int(c95.loc['cherimoya_only', 'bias_clusters'])} Cherimoya-only, "
                    f"{int(c95.loc['chrombpnet_only', 'bias_clusters'])} ChromBPNet-only and "
                    f"{int(c95.loc['shared', 'bias_clusters'])} shared clusters are mostly flagged patterns."
                )
        t1 = table_html(n, "Patterns flagged as Tn5/DNase by the motif database, summed over the datasets. Cherimoya − "
                        f"ChromBPNet per dataset: {test_cell(t, 1)}; per QC fold: {test_cell(tq, 1)}.{jt}",
                        ["Model", "Fold-average patterns flagged", "Per dataset", "Their seqlets",
                         "QC patterns flagged (all folds)"], rows)  # fmt: skip
    ref = bc["reference"]
    split = bc["split"].set_index("reference_kind")
    t2 = t3 = ""
    if ref is not None and not ref.empty:
        rr = ref.sort_values("cherimoya_patterns", ascending=False)
        rows = [[h(r["pattern"]), count(r["n_seqlets"]), h(r["consensus"]), h(r["annotation"]), pval(r["annotation_q"]),
                 h(r["bias_db_best"]), pval(r["bias_db_best_q"]), h(r["kind"]), count(r["cherimoya_patterns"]),
                 count(r["cherimoya_seqlets"])] for _, r in rr.iterrows()]  # fmt: skip
        t2 = table_html(
            n,
            "The reference: the bias-included ChromBPNet's TF-MoDISco patterns, annotated here with "
            "level 3's rules, and the unmatched Cherimoya patterns that match each (tomtom-lite, q ≤ 0.05, "
            "database decoys).",
            [
                "Pattern",
                "Seqlets",
                "Consensus",
                "MotifCompendium match",
                "q",
                "motifs.meme match",
                "q",
                "Kind",
                "Cherimoya patterns matched",
                "Their seqlets",
            ],
            rows,
            left=4,
            cls="compact",
        )
        srows = []
        for k in ("Tn5/DNase", "TF or other", "no match", "unknown"):
            if k in split.index:
                r = split.loc[k]
                srows.append([h({"Tn5/DNase": "a Tn5/DNase pattern", "TF or other": "a TF or other pattern",
                                 "no match": "no reference pattern", "unknown": "an unannotated reference"}[k]),
                              count(r["patterns"]), count(r["seqlets"]), count(r["db_flagged"])])  # fmt: skip
        t3 = table_html(
            n,
            f"The {int(split['patterns'].sum())} Cherimoya patterns checked (those without a cluster "
            "partner: every dataset's fold averages, and the control's own QC fold), by what they match.",
            ["Matches", "Cherimoya patterns", "Their seqlets", "Themselves flagged Tn5/DNase"],
            srows,
        )
    return f"""
<section id="bias-control">
  {head}
  {lead}
  {"".join(ctrl)}
  {
        figure_html(
            rep,
            n,
            "bias_control",
            "Left: Tn5/DNase-flagged patterns per dataset (lines join a dataset's two "
            "values), at fold averages and per QC fold. Right: each pattern of the bias-included ChromBPNet "
            "(profile head), labelled with its best MotifCompendium match, and how many unmatched Cherimoya patterns "
            "match it; orange marks Tn5/DNase patterns.",
        )
    }
  {t1}
  {t2}
  {t3}
  <p>Most of the Cherimoya patterns that the bias-included model also finds match its TF patterns (CEBP, AP-1, NFI,
  CTCF), not its Tn5 patterns: they are TF motifs that the strict cluster rule left unmatched, not bias. The control is
  one dataset and fold, on the profile head (DeepSHAP has no rule for the counts head of the full model), while every
  other pattern here comes from the counts head, so it bounds rather than measures the bias share.</p>
</section>"""


def section_caveats(rep: Report, R: dict) -> str:
    l2, l3, l4, l5 = (R.get(k) for k in ("l2", "l3", "l4", "l5"))
    bias_bits = []
    if l3 and l3.get("state") == "done":
        per = l3["per"]
        bias_bits.append(
            f"Tn5/DNase-flagged fold-average patterns: {int(per[per['model'] == 'cherimoya']['bias_flagged'].sum())} "
            f"for Cherimoya against {int(per[per['model'] == 'chrombpnet']['bias_flagged'].sum())} for ChromBPNet"
        )
    if l4 and l4.get("state") == "done":
        c95 = l4["classes"][l4["classes"]["threshold"] == "0.95"].set_index("class")
        if "bias_clusters" in c95:
            bias_bits.append(
                f"bias clusters in the joint compendium: {int(c95.loc['cherimoya_only', 'bias_clusters'])} "
                f"Cherimoya-only, {int(c95.loc['chrombpnet_only', 'bias_clusters'])} ChromBPNet-only, "
                f"{int(c95.loc['shared', 'bias_clusters'])} shared"
            )
    f16 = ""
    if l2 and l2.get("state") == "done" and l2.get("float16") is not None and len(l2["float16"]):
        f16 = (
            f" Computed again from Cherimoya's float16 h5 on one dataset, level 2's medians move by at most "
            f"{num(l2['float16']['median_diff'].abs().max(), 6)}."
        )
    drop = ""
    if l2 and l2.get("state") == "done":
        j = l2["joins"]
        drop = (
            f" {int(j['chrombpnet_only_peaks'].min())}–{int(j['chrombpnet_only_peaks'].max())} peaks per dataset "
            f"({int(j['chrombpnet_only_peaks'].sum())} over the {len(j)} datasets) are in ChromBPNet's beds only."
        )
    if l5 and l5.get("state") == "done":
        dd = l5["totals"]["chrombpnet_hits_on_dropped_peaks"]
        drop += f" ChromBPNet has {count(dd.min())}–{count(dd.max())} hits per dataset on them."
    items = [
        (
            "Bias model",
            "ChromBPNet's attributions (levels 2–5) come from its bias-corrected model (chrombpnet_nobias); "
            "Cherimoya attributes its log-counts head with no bias model, so Tn5/DNase motifs can appear on "
            "Cherimoya's side only. Level 1 is not affected: it scores ChromBPNet's full bias-included model against "
            "observed counts. "
            + ("; ".join(bias_bits) + ". " if bias_bits else "")
            + "The bias control used one dataset and fold and the "
            "profile head, so it bounds rather than measures how much of Cherimoya's extra signal is bias.",
        ),
        (
            "Float16",
            "Both models' attributions go through float16: ChromBPNet's 05.0 h5 stores them so, and Cherimoya's "
            "converted h5 files, which 06.0–08.0 and Fi-NeMo read, do too (06.0 sums float16)." + f16,
        ),
        (
            "Dropped N windows",
            "Cherimoya drops the windows that contain an N, so a few peaks have ChromBPNet attributions "
            "only. Levels 2, 3 and 5 compare only the peaks both models hold." + drop,
        ),
        (
            "Ceilings",
            "A ceiling (C) compares one fold with the mean of the other four. Cross-model agreement on single "
            "folds (A) compares one fold with one fold and so sits below a like-for-like ceiling; fold averages (B) "
            "compare a mean of five with a mean of five and sit above one. A single-fold-against-single-fold ceiling "
            "was not computed. At level 3 only the QC scale has a ceiling; levels 4 and 5 have none.",
        ),
        (
            "p-value floors",
            "With n paired values the exact two-sided Wilcoxon p cannot go below 2/2^n: 0.002 for 10 "
            "datasets, 3.6×10⁻¹² for 39. Folds of one dataset share its peaks, so the dataset-level test is the one "
            "to read; with 50 or 195 folds p saturates further. At those floors, read p as “all in the same "
            "direction”, not as an effect size.",
        ),
    ]
    lis = "".join(f"<li><b>{h(t)}.</b> {h(x)}</li>" for t, x in items)
    return f"""
<section id="caveats">
  <div class="head"><p class="eyebrow">Caveats</p><h2>Read the numbers with these in mind</h2></div>
  <ul class="plain">{lis}</ul>
</section>"""


def section_methods(rep: Report, R: dict, prov: dict, generated: str) -> str:
    a = rep.args
    defs = [
        ("Statistics", f"Every interval is a percentile bootstrap 95% CI of the median, {a.n_boot:,} resamples, seed "
         f"{a.seed}, resampling datasets; for rows over folds, whole datasets are resampled with their folds (level "
         "2 and 3 files resample folds instead). Tests are two-sided Wilcoxon signed-rank tests on paired values "
         "(scipy; exact for small n without ties or zeros). statistics.tsv lists all of them with n, the statistic, p "
         "and how many values were above and below 0."),
        ("Level 1", "compare_chrombpnet.py per fold: counts Pearson and Spearman and MSE of log counts, and the median "
         "JSD and normalized JSD of the profiles, on ChromBPNet's test peaks; ChromBPNet from its evaluation "
         "predictions (chrombpnet_predictions.h5, full model). Residuals: predicted minus observed log(1 + counts)."),
        ("Level 2", "compare_attributions.py: per peak, Pearson and Spearman of the projected scores, cosine of the "
         "4 × W hypothetical maps, Jaccard of the top 50 positions by |projected score|, and each map's sum and L2 "
         "norm, within 250 and 500 bp of the summit. Each dataset is summarized by its median over peaks; A and C "
         "are means over the 5 folds of those medians."),
        ("Level 3", "compare_modisco.py: tomtom-lite on trimmed PPMs, q-values over the targets with the "
         "MotifCompendium database as decoys; matched by joint20 cluster (primary) or tomtom (secondary); seqlets "
         "overlap at ≥ 50% reciprocal. Bias flag: best motifs.meme match is a TN5_ or DNASE_ entry at q ≤ 0.05."),
        ("Level 4", "compare_compendia.py: reciprocal best matches between cluster CWMs (trimmed, unit L2), "
         "Benjamini-Hochberg over the other compendium's clusters; classes from each cluster's patterns per model."),
        ("Level 5", "compare_finemo.py: hits of the same motif and strand with centres within 5 bp (0 and 10 bp too); "
         "precision over Cherimoya's hits, recall over ChromBPNet's; per-peak Pearson of the motif-count vectors."),
        ("Tables", "Under the output directory: statistics.tsv, tables/*.tsv (one per section and the inputs and "
         "checks below) and figures/*.pdf/.png."),
    ]  # fmt: skip
    dl = "".join(f"<dt>{h(x)}</dt><dd>{h(y)}</dd>" for x, y in defs)
    crow = []
    for c in prov["commits"]:
        crow.append(
            [
                h(c["name"]),
                f"<code>{h(c['commit'])}</code>{' (uncommitted changes)' if c.get('dirty') else ''}",
                h(c.get("branch") or ""),
                h(c.get("subject") or ""),
                h(et(c.get("date"))) if c.get("date") else "",
            ]
        )
    eng = prov.get("engine")
    if eng:
        crow.append([h("Cherimoya attribution engine (pin)"), f'<code>{h(str(eng.get("git_commit"))[:7])}</code>',
                     h(eng.get("git_branch") or ""), h(f"{eng.get('engine')} (attribution_engine.json)"),
                     h(et(eng.get("written")))])  # fmt: skip
    for c in prov["compendia"]:
        pl = c.get("pipeline") or {}
        sw = c.get("software") or {}
        crow.append(
            [
                h(f"compendium {c['tag']} (09.0)"),
                f"<code>{h(str(pl.get('commit') or '?')[:7])}</code>",
                h(pl.get("branch") or ""),
                h(f"MotifCompendium {sw.get('MotifCompendium', '?')}, threshold {c.get('threshold')}"),
                h(et(c.get("created"))) if c.get("created") else "",
            ]
        )
    n = Numbering()
    n.tab = 900  # these tables are not numbered with the findings
    commits = ('<div class="tbl compact"><table><thead><tr><th class="l">Code</th><th class="l">Commit</th>'
               '<th class="l">Branch</th><th class="l">Subject or note</th><th class="l">Time</th></tr></thead><tbody>'
               + "".join("<tr>" + "".join(f'<td class="l">{c}</td>' for c in r) + "</tr>" for r in crow)
               + "</tbody></table></div>")  # fmt: skip
    mrows = []
    for m in prov["markers"]:
        if not m.get("units"):
            mrows.append([f"{m['level']}", h(m["dir"]), "0", "", "", ""])
            continue
        mrows.append([f"{m['level']}", h(m["dir"]), str(m["units"]), h(et(m["first"])), h(et(m["last"])),
                      h(", ".join(m["commits"]))])  # fmt: skip
    markers = ('<div class="tbl compact"><table><thead><tr><th class="l">Level</th><th class="l">Directory</th>'
               "<th>Done units</th><th>First written</th><th>Last written</th><th>Code commits</th></tr></thead><tbody>"
               + "".join("<tr>" + "".join(f'<td class="{"l" if i < 2 else ""}">{c}</td>' for i, c in enumerate(r))
                         + "</tr>" for r in mrows) + "</tbody></table></div>")  # fmt: skip
    inputs = R.get("inputs_table")
    irows = []
    if inputs is not None:
        for _, r in inputs.iterrows():
            irows.append([f"<code>{h(r['path'])}</code>", h(r["files"]), h(r["size"]), h(r["modified"])])
    itab = ('<div class="tbl compact"><table><thead><tr><th class="l">Input</th><th>Files</th><th>Size</th>'
            "<th>Modified (newest)</th></tr></thead><tbody>"
            + "".join("<tr>" + "".join(f'<td class="{"l" if i == 0 else ""}">{c}</td>' for i, c in enumerate(r))
                      + "</tr>" for r in irows) + "</tbody></table></div>")  # fmt: skip
    chk = "".join(
        f"<li>{h(c['what'])}: {c['values']} values, largest difference "
        f"{h(f'{c["max_abs_diff"]:.2g}' if isnum(c['max_abs_diff']) else 'NA')}"
        f"{f', {c["nan_mismatch"]} NA mismatches' if c['nan_mismatch'] else ''}.</li>"
        for c in rep.checks
    )
    probs = "".join(f"<li>{h(p)}: {h(m)}</li>" for p, m in rep.problems) or "<li>None.</li>"
    return f"""
<section id="methods">
  <div class="head"><p class="eyebrow">Methods and provenance</p><h2>How the numbers were made</h2></div>
  <dl class="defs">{dl}</dl>
  <h3>Code</h3>
  {commits}
  <h3>Level outputs</h3>
  {markers}
  <h3>Inputs</h3>
  <p class="tcap">Paths relative to {h(str(rep.root.parent))}/. Times are US Eastern (the cluster runs on CEST,
  six hours ahead). Generated {h(generated)} into <code>{h(str(rep.out))}</code>.</p>
  {itab}
  <h3>Checks</h3>
  <p class="tcap">The statistics the levels already computed were computed again here from their tables; the largest
  absolute differences against the levels' own statistics.tsv (written to 6 significant digits):</p>
  <ul class="plain">{chk or "<li>None run.</li>"}</ul>
  <h3>Not computed</h3>
  <ul class="plain">{probs}</ul>
</section>"""


def inputs_table(rep: Report):
    """The inputs grouped for the page: per directory under the project root, the files, their size and the newest
    modification time."""
    import pandas as pd

    base = rep.root.parent
    groups: dict[str, list[dict]] = {}
    for role, entry in rep.seen.items():
        if entry.get("absent"):
            continue
        p = Path(entry["path"])
        try:
            rel = p.relative_to(base)
        except ValueError:
            rel = p
        parts = rel.parts
        if re.match(r"^level1/[^/]+/[^/]+/(compare_tsv|counts_npz)$", role) and len(parts) > 2:
            label = f"{parts[0]}/<dataset>/fold_<k>/{p.name}"
        elif len(parts) >= 3 and parts[2] in rep.shared:
            label = "/".join(parts[:2]) + "/<dataset>/" + "/".join(parts[3:])
        else:
            label = str(rel)
        groups.setdefault(label, []).append(entry)
    rows = []
    for label, entries in sorted(groups.items()):
        size = sum(e.get("size", 0) for e in entries)
        newest = max(e.get("mtime_ns", 0) for e in entries)
        rows.append({"path": label, "files": len(entries), "size": human(size), "modified": et(newest / 1e9),
                     "mtime_ns": newest})  # fmt: skip
    return pd.DataFrame(rows)


def human(n: float) -> str:
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000.0
    return f"{n:.1f} TB"


def render(rep: Report, R: dict, prov: dict, generated: str) -> str:
    n = Numbering()
    body = [
        masthead(rep, R, generated), toc(), summary_section(rep, R),
        section_level1(rep, n, R.get("l1")), section_level2(rep, n, R.get("l2")), section_level3(rep, n, R.get("l3")),
        section_level4(rep, n, R.get("l4")), section_level5(rep, n, R.get("l5")),
        section_bias(rep, n, R.get("bias"), R.get("l3"), R.get("l4")), section_caveats(rep, R),
        section_methods(rep, R, prov, generated),
    ]  # fmt: skip
    return (
        f"<title>Cherimoya vs ChromBPNet</title>\n<style>{CSS}</style>\n"
        f'<div class="page">\n' + "\n".join(body) + "\n</div>\n"
    )


# ---------------------------------------------------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    import pandas as pd

    rep = Report(args)
    logger.info("report on %d datasets (%d shared) from %s into %s", len(rep.everything), len(rep.shared), rep.root,
                rep.out)  # fmt: skip
    rep.out.mkdir(parents=True, exist_ok=True)
    R: dict = {}
    R["l1"] = rep.part("Level 1", level1)
    R["l2"] = rep.part("Level 2", level2)
    R["l3"] = rep.part("Level 3", level3)
    R["bias"] = rep.part("Bias control", bias_control, R["l3"] if R["l3"] and R["l3"].get("state") == "done" else None)
    R["l4"] = rep.part("Level 4", level4)
    R["l5"] = rep.part("Level 5", level5, R["l4"] if R["l4"] and R["l4"].get("state") == "done" else None)
    prov = provenance(rep)
    changed = [role for role, entry in rep.seen.items() if common.stat_entry(entry["path"]) != entry]
    if changed:
        rep.problems.append(("Inputs", f"{len(changed)} input(s) changed while the report ran (e.g. {changed[0]}); run "
                             "it again"))  # fmt: skip
    R["inputs_table"] = inputs_table(rep)
    newest = R["inputs_table"]["mtime_ns"].max() if len(R["inputs_table"]) else None
    R["newest_input"] = et(newest / 1e9) if newest else "NA"
    generated = et(dt.datetime.now(dt.UTC).timestamp())

    rep.table("inputs", R["inputs_table"].drop(columns=["mtime_ns"]))
    rep.table("checks", pd.DataFrame(rep.checks, columns=["what", "values", "max_abs_diff", "nan_mismatch"]))
    outputs: dict[str, Path] = {}
    for name, df in rep.tables.items():
        outputs[f"tables/{name}"] = common.write_table(rep.out / "tables" / f"{name}.tsv", df)
    outputs["statistics"] = common.write_table(rep.out / "statistics.tsv",
                                               pd.DataFrame(rep.stats.rows, columns=list(Stats.COLUMNS)))  # fmt: skip
    page = render(rep, R, prov, generated)
    outputs["index"] = common.write_text(rep.out / "index.html", page)
    size = (rep.out / "index.html").stat().st_size
    pend = sorted({p["what"].split(" (")[0] for p in rep.pending})
    note = (f"index.html {size / 1e6:.1f} MB, {len(rep.figures)} figures, {len(rep.tables)} tables, "
            f"{len(rep.stats.rows)} statistics")  # fmt: skip
    if R.get("l1") and R["l1"].get("state") == "done":
        note += f"; level 1 {R['l1']['n_folds']}/{R['l1']['expected']} folds"
    if pend:
        note += f"; pending: {'; '.join(pend)}"
    if rep.problems:
        note += f"; not computed: {', '.join(p for p, _ in rep.problems)}"
    settings = {"datasets": rep.shared, "everything": rep.everything, "n_boot": args.n_boot, "seed": args.seed,
                "pending_eta": args.pending_eta, "dpi": args.dpi}  # fmt: skip
    common.Marker(rep.out, "report", FORMAT, rep.inputs, settings).write(outputs, note)
    if size > 16e6:
        rep.problems.append(("Page", f"index.html is {size / 1e6:.1f} MB, over the 16 MB limit"))
    common.result(note)
    return 1 if rep.problems else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_root_args(ap)
    ap.add_argument("--out-dir", type=Path, help="default <comparison-root>/report")
    ap.add_argument(
        "--datasets",
        type=Path,
        help="the datasets both models were interpreted on (default <compendia-root>/inputs/shared10.txt)",
    )
    ap.add_argument(
        "--all-datasets",
        type=Path,
        nargs="+",
        help="level 1's dataset lists (default --datasets and <compendia-root>/inputs/rest29.txt)",
    )
    ap.add_argument("--joint", default="joint20", help="the joint compendium's tag")
    ap.add_argument("--cbp-tag", default="cbp10")
    ap.add_argument("--cheri-tag", default="cheri10")
    ap.add_argument("--supplement-tag", default="cheri39")
    ap.add_argument("--sensitivity-tag", help="default <joint>_t090")
    ap.add_argument("--pending-eta", help="when the pending parts are expected, shown with them (e.g. 'Saturday "
                    "2026-10-10')")  # fmt: skip
    ap.add_argument("--repo", action="append", metavar="NAME=PATH", help="a checkout whose commit the methods list "
                    "(repeatable; this code's checkout and --folds-dir's are always listed)")  # fmt: skip
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dpi", type=int, default=144, help="resolution of the figures embedded in the page")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    common.setup_logging(args.verbose)
    common.require(args, "comparison_root")
    parent = args.comparison_root.parent
    args.cheri_root = args.cheri_root or parent / "cherimoya"
    args.compendia_root = args.compendia_root or parent / "compendia"
    args.chrombpnet_root = args.chrombpnet_root or parent / "chrombpnet"
    args.out_dir = args.out_dir or args.comparison_root / "report"
    args.datasets = args.datasets or args.compendia_root / "inputs" / "shared10.txt"
    if not args.comparison_root.is_dir():
        raise SystemExit(f"no comparison root {args.comparison_root}")
    if not args.datasets.is_file():
        raise SystemExit(f"no dataset list {args.datasets}")
    os.environ.setdefault("NUMBA_NUM_THREADS", str(max(1, len(os.sched_getaffinity(0)))))
    try:
        return run(args)
    except common.Failed as exc:
        logger.error("%s", exc)
        common.result(f"failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
