#!/usr/bin/env python3
"""The comparison's report: one page that reads levels 1-6 and says what they found.

Reads the levels' outputs under --comparison-root (performance/, attributions/, modisco/, compendium/, finemo/,
bias_control/), each fold's comparison under --cheri-root, the compendia under --compendia-root and, for level 6,
qtl_compare.py's comparison of the models' variant scores with measured QTLs (--caqtl), computes the statistics the
levels leave out, and writes, under --out-dir (default <comparison-root>/report):

  index.html        the report page: the findings with their key numbers, one section per level with figures and
                    compact tables, the bias control, the caveats, and the methods with each input's time (ET) and
                    the code's commits. Self-contained (figures embedded as PNG data URIs, fonts from Google Fonts),
                    ready to publish as a private Artifact page
  statistics.tsv    every interval and test on the page, one row each: level, family, quantity, test (interval or
                    wilcoxon; level 6: interval, mcnemar or bootstrap), over (datasets or folds; level 6: variants),
                    n, estimate (the median of the values, or of the paired differences; level 6: the share or the
                    difference itself), ci95_lo and ci95_hi (percentile bootstrap, --n-boot resamples, --seed;
                    datasets are resampled, and for rows over folds whole datasets with their folds unless the note
                    says otherwise; level 6: qtl_compare.py's bootstrap over variants), statistic and p (two-sided
                    Wilcoxon signed-rank; level 6: McNemar's exact test, or the bootstrap p), p_method (how scipy's
                    'auto' computed p: exact, the signed-rank distribution; permutation, every sign flip, for up to
                    13 values with ties or zeros; normal, the normal approximation with tie and continuity
                    corrections, for more than 50 values or ties or zeros among 14-50; the page marks these with a
                    dagger; level 6: exact binomial or bootstrap), higher and lower (how many datasets or folds the
                    difference is above or below 0; the test drops the differences of 0; level 6, McNemar rows: the
                    leads only the first and only the second model calls the right way), note
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
  bias      the bias control split by what the reference pattern is: the uncorrected model's patterns are
            annotated here with level 3's rules (best MotifCompendium match; flagged when the best match in
            chrombpnet's motifs.meme is a Tn5 or DNase I entry at q <= 0.05)
  level 4   per dataset, the share of each model's patterns and seqlets in clusters holding both models'
            patterns, at the joint compendium's threshold and at the sensitivity threshold; the annotations of
            the reciprocal like-for-like pairs
  level 5   hits per 10,000 shared peaks per model; precision and recall by cluster class and by sign (the
            sign from joint_clusters.tsv); per-motif medians; the largest motifs; 11.0's motif reports side by side
  level 6   nothing new: qtl_compare.py's shares, intervals and tests as written (the shares are computed again from
            its rows.tsv.gz as a check), and per dataset how often Cherimoya is ahead
The tests levels 2, 3 and 5 already ran are computed again from their tables (for the statistic as well as p) and
checked against the levels' statistics.tsv (tables/checks.tsv).

Pending parts are shown as pending, never as failures: folds without both models' comparison yet (level 1), the
supplement compendium (--supplement-tag; level 4's supplement.json) and the production hits (level 5's
finemo/<ds>/production/). A part that cannot be computed is shown on the page with its reason, everything else is
still written, and the script exits 1. Each run writes the page again; .done/report.json records what it read.

    python workflows/cherimoya/compare/report.py [--comparison-root <dir>] [--out-dir <dir>] [--cheri-root <dir>]
        [--compendia-root <dir>] [--datasets <shared list>] [--caqtl <qtl_compare.py output>] [--pending-eta "<sentence>"]
        [--repo <name>=<checkout> ...] [--n-boot 10000] [--seed 0] [--dpi 144]

The roots default to the environment compare.sh exports (COMPARISON_ROOT, CHERI_ROOT, COMPENDIA_ROOT), else to the
siblings of --comparison-root (<parent>/cherimoya, <parent>/compendia). --datasets defaults to
<compendia-root>/inputs/shared10.txt; level 1 reads it and <compendia-root>/inputs/rest29.txt (else every
<cheri-root>/configs/<dataset>/). --caqtl defaults to the newest <comparison-root>/../variant_scoring/caqtl/out/<run>/
(by its run.json); without one, level 6 is shown as pending. Runs in the pixi cherimoya environment (numpy, scipy, pandas, matplotlib, h5py,
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

FORMAT = 2
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
VARIANT_MODELS = ("cherimoya", "cbp_nobias", "cbp_full", "cbp_bias")
"""qtl_compare.py's models: Cherimoya, ChromBPNet's corrected and uncorrected models and its bias model."""
VARIANT_STYLE = {"cherimoya": (COLOURS["cherimoya"], "-"), "cbp_nobias": (COLOURS["chrombpnet"], "-"),
                 "cbp_full": (COLOURS["chrombpnet"], ":"), "cbp_bias": (MUTED, "--")}  # fmt: skip
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


def window_name(w: str, full: bool = True) -> str:
    """A level-2 window by its width: 'w500' -> '500-bp window (summit ± 250 bp)' (full) or '500-bp window'."""
    width = int(w[1:])
    return f"{width:,}-bp window (summit ± {width // 2:,} bp)" if full else f"{width:,}-bp window"


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


def p_method(diffs) -> str:
    """How common.wilcoxon's scipy call (method 'auto') computes p for these paired differences: 'exact' (the
    signed-rank distribution: at most 50 values, no ties, no zeros), 'permutation' (every sign flip: at most 13 values
    with ties or
    zeros) or 'normal' (the normal approximation with tie and continuity corrections: more than 50 values, or ties or
    zeros among 14-50); '' when there is no test (no difference other than 0)."""
    d = finite(diffs)
    nz = d[d != 0]
    if nz.size == 0:
        return ""
    if d.size > 50:
        return "normal"
    if nz.size == d.size and np.unique(np.abs(nz)).size == nz.size:
        return "exact"
    return "permutation" if d.size <= 13 else "normal"


class Stats:
    """Every interval and test of the page: the rows of statistics.tsv."""

    COLUMNS = ("level", "family", "quantity", "test", "over", "n", "estimate", "ci95_lo", "ci95_hi", "statistic", "p",
               "p_method", "higher", "lower", "note")  # fmt: skip

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
        full["p_method"] = row.get("p_method", "")
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
                        estimate=est, ci95_lo=lo, ci95_hi=hi, statistic=w, p=p, p_method=p_method(d),
                        higher=int((d > 0).sum()), lower=int((d < 0).sum()), note=note)  # fmt: skip

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
                "short": f"{len(missing)} level-1 folds",
                "what": f"{len(missing)} of {expected or '?'} folds do not have both models' comparison yet; the "
                "numbers below cover the folds there are",
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
                p_method=p_method(means),
                higher=int((means > 0).sum()),
                lower=int((means < 0).sum()),
                note=f"{better} is better",
            )
            fold = rep.stats.paired("1", fam, f"{m}: Cherimoya - ChromBPNet, folds", [p[f"{m}_diff"] for p in sub],
                                    over="folds", groups=[p["dataset"] for p in sub],
                                    note=f"{better} is better; the CI resamples datasets with their folds")  # fmt: skip
            s.update(set=key, fold_ci95_lo=fold["ci95_lo"], fold_ci95_hi=fold["ci95_hi"],
                     wilcoxon_datasets_p_method=p_method(means), wilcoxon_folds_p_method=fold["p_method"],
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
                "wilcoxon_stat": row["statistic"], "wilcoxon_p": row["p"], "wilcoxon_p_method": row["p_method"],
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
        rep.pending.append({"part": "Level 2", "short": "level 2", "what": "Level 2 has not summarized any dataset yet",
                            "detail": []})  # fmt: skip
        return {"state": "pending"}
    summary = read_tsv(rep.use("level2/summary", d / "summary.tsv"), dtype={"level": str})
    info = {}
    for ds in rep.shared:
        p = d / ds / "dataset.json"
        if p.is_file() and (d / ".done" / f"{ds}.json").is_file():
            info[ds] = json.loads(rep.use(f"level2/{ds}/dataset", p).read_text())
    missing = [ds for ds in rep.shared if ds not in info]
    if missing:
        rep.pending.append({"part": "Level 2", "short": f"{len(missing)} level-2 datasets", "what": f"{len(missing)} "
                            f"of {len(rep.shared)} datasets are not compared yet", "detail": missing})  # fmt: skip
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
        ax.set_title(f"{METRIC_NAMES[metric]}, {window_name(window)}")
        ax.set_ylabel("mean over peaks" if metric == "jaccard_mean" else "median over peaks")
        ax.grid(axis="x", visible=False)
        ax.set_xlim(-0.6, len(CROSS_ORDER) - 0.35)
    for ax in list(axes.flat)[len(panels) :]:
        ax.set_visible(False)
    fig.tight_layout(h_pad=2.2)
    rep.figure(
        fig,
        "level2_agreement",
        f"Four panels, one per agreement metric in the {window_name(window)}: per dataset, "
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
    titles = {"signal_decile": "peak-signal decile", "gc_decile": "GC decile"}
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
        ax.set_title(f"{METRIC_NAMES[metric]}, {window_name(window, full=False)}, by {titles[stratum]}")
    axes[0][0].set_ylabel("median over datasets, 95% CI")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    rep.figure(fig, "level2_strata", "Two line charts of attribution agreement by peak-signal decile and by GC decile, "
               "for cross-model single folds, fold averages and both models' ceilings.")  # fmt: skip


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
        rep.pending.append({"part": "Level 3", "short": "level 3", "what": "Level 3 has not summarized any dataset yet",
                            "detail": []})  # fmt: skip
        return {"state": "pending"}
    summary = read_tsv(rep.use("level3/summary", d / "summary.tsv"))
    summary = summary[summary["dataset"].isin(rep.shared)].reset_index(drop=True)
    missing = [ds for ds in rep.shared if ds not in set(summary["dataset"])]
    if missing:
        rep.pending.append({"part": "Level 3", "short": f"{len(missing)} level-3 datasets", "what": f"{len(missing)} "
                            f"of {len(rep.shared)} datasets are not compared yet", "detail": missing})  # fmt: skip
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
    ax1.set_xticks([0, 1], ["same joint\ncluster", "tomtom partner\n(q ≤ 0.05)"])
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
    # each checked pattern's own level-3 flag, and whether ChromBPNet corrected has a tomtom partner for it (fold
    # averages: patterns.tsv's matched_tomtom; the control's QC fold: qc_patterns.tsv's matched_cross)
    own, partner = {}, {}
    if l3 is not None:
        pt = l3["patterns"]
        pt = pt[pt["model"] == "cherimoya"]
        own.update({(ds, "fold average", p): f for ds, p, f in zip(pt["dataset"], pt["pattern"], pt["bias_flag"])})
        partner.update({(ds, "fold average", p): bool(t) for ds, p, t in zip(pt["dataset"], pt["pattern"],
                                                                             pt["matched_tomtom"])})  # fmt: skip
        qp = l3["qc_patterns"]
        qp = qp[qp["model"] == "cherimoya"]
        own.update({(ds, f"QC fold {k}", p): f for ds, k, p, f in zip(qp["dataset"], qp["fold"], qp["pattern"],
                                                                      qp["bias_flag"])})  # fmt: skip
        if "matched_cross" in qp:
            partner.update({(ds, f"QC fold {k}", p): bool(t) for ds, k, p, t in zip(
                qp["dataset"], qp["fold"], qp["pattern"], as_bool(qp["matched_cross"]))})  # fmt: skip
    keys = list(zip(bc["dataset"], bc["scale"], bc["pattern"]))
    bc["own_bias_flag"] = [own.get(k) for k in keys]
    bc["chrombpnet_tomtom_partner"] = [partner.get(k) for k in keys]
    rep.table("bias_control_patterns", bc)
    rep.table("bias_reference_patterns", ref)

    def n_true(s) -> int:
        return int(sum(bool(x) for x in s if x is not None))

    split = (bc.groupby("reference_kind").agg(patterns=("pattern", "size"), seqlets=("n_seqlets", "sum"),
                                              db_flagged=("own_bias_flag", n_true),
                                              chrombpnet_partner=("chrombpnet_tomtom_partner", n_true))
             .reset_index())  # fmt: skip
    rep.table("bias_control_split", split)
    scales = {k: int(v) for k, v in bc["scale"].map(lambda x: "fold average" if x == "fold average" else "QC fold")
              .value_counts().items()}  # fmt: skip
    if l3 is not None or not ref.empty:
        plot_bias(rep, l3, ref)
    return {"state": "done", "controls": controls, "table": bc, "json": bj, "reference": ref, "split": split,
            "scales": scales}  # fmt: skip


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
        ax2.set_title("Reference: ChromBPNet uncorrected's patterns (profile head)")
        ax2.grid(axis="y", visible=False)
        ax2.set_axisbelow(True)
        ax2.set_xlim(0, max(1.0, float(r["cherimoya_patterns"].max())) * 1.15)
        from matplotlib.patches import Patch

        ax2.legend(handles=[Patch(color=TN5, label="Tn5/DNase pattern (motifs.meme, q ≤ 0.05)"),
                            Patch(color="#8C959F", label="TF or other pattern")], loc="lower right")  # fmt: skip
    else:
        ax2.set_visible(False)
    fig.tight_layout()
    rep.figure(fig, "bias_control", "Left: Tn5/DNase-flagged patterns per dataset for each model. Right: for each "
               "pattern of ChromBPNet uncorrected, labelled with its MotifCompendium match or, for Tn5/DNase "
               "patterns, its motifs.meme entry, how many unmatched Cherimoya patterns match it, coloured by "
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
        n_all = len(rep.everything)
        rep.pending.append({"part": "Level 4", "key": "supplement", "label": f"{n_all}-dataset supplement",
                            "short": f"the {n_all}-dataset Cherimoya supplement (level 4)",
                            "what": f"The supplement, Cherimoya's compendium of all {n_all} datasets set against its "
                            f"compendium of the {len(rep.shared)} shared ones, is not in this report yet",
                            "note": f"The {n_all}-dataset compendium is built, but level 4 has not compared it "
                            "yet" if built else f"The {n_all}-dataset compendium is not built yet",
                            "detail": []})  # fmt: skip
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
    ax1.set_axisbelow(True)
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
        rep.pending.append({"part": "Level 5", "short": f"{len(missing)} level-5 datasets", "what": f"{len(missing)} "
                            f"of {len(rep.shared)} datasets are not compared yet", "detail": missing})  # fmt: skip
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
        n_all = len(rep.everything)
        rep.pending.append({"part": "Level 5", "key": "production", "label": "production hits",
                            "short": "the production hits (level 5)",
                            "what": f"The production hits, ChromBPNet's on its own {len(rep.shared)}-dataset "
                            f"compendium against Cherimoya's on its {n_all}-dataset compendium, paired through level "
                            "4's cluster matches, are not in this report yet",
                            "note": f"{len(prod)} of {len(rep.shared)} datasets are compared; they need the "
                            f"{n_all}-dataset compendium, Cherimoya's Fi-NeMo run on it and level 4's matches",
                            "detail": []})  # fmt: skip
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
# level 6: the models' variant effects against measured QTLs (qtl_compare.py's outputs)


def newest_caqtl(root: Path) -> Path | None:
    """The newest qtl_compare.py output under <root>/out/ (by its run.json's modification time), or None."""
    runs = [p.parent for p in (root / "out").glob("*/run.json") if (p.parent / "paired.tsv").is_file()]
    return max(runs, key=lambda p: (p / "run.json").stat().st_mtime_ns) if runs else None


def level6(rep: Report) -> dict:
    import pandas as pd

    d = rep.args.caqtl
    if d is None or not (Path(d) / "run.json").is_file():
        rep.pending.append({"part": "Level 6", "key": "variants", "label": "Level 6", "short": "variant effects",
                            "what": "The models' variant scores are not compared with measured QTLs yet (no "
                            "qtl_compare.py output)"})  # fmt: skip
        return {"state": "pending"}
    d = Path(d)
    info = json.loads(rep.use("level6/run", d / "run.json").read_text())
    met = rep.table("level6_metrics", read_tsv(rep.use("level6/metrics", d / "metrics.tsv")))
    par = rep.table("level6_paired", read_tsv(rep.use("level6/paired", d / "paired.tsv")))
    per = rep.table("level6_per_dataset", read_tsv(rep.use("level6/per_dataset", d / "per_dataset.tsv")))
    cur = read_tsv(rep.use("level6/curves", d / "curves.tsv"))
    if "setting" not in par.columns:
        raise ValueError(f"{d / 'paired.tsv'} has no setting column: it predates qtl_compare.py's like-for-like "
                         "held-out setting; run qtl_compare.py again")  # fmt: skip
    st = info.get("settings") or {}
    subsets = {k: float(v) for k, v in (st.get("subsets") or {"in_peak": 250.0, "near": 500.0}).items()}
    primary = next(iter(subsets))
    labels = dict(zip(met["model"], met["model_label"]))

    fam = "caQTL sign concordance"
    for _, r in met.iterrows():
        rep.stats.add(level="6", family=f"{fam}, {r['setting']}, {r['subset']}",
                      quantity=f"{r['group']}: {r['model_label']}", test="interval", over="variants", n=int(r["n"]),
                      estimate=r["concordance"], ci95_lo=r["concordance_lo"], ci95_hi=r["concordance_hi"],
                      note="the share of leads whose logFC has the slope's sign; bootstrap over variants")  # fmt: skip
    for _, r in par.iterrows():
        mc = r["measure"] == "concordance"
        rep.stats.add(level="6", family=f"{fam}, {r['setting']}, {r['subset']}, paired",
                      quantity=f"{r['group']}: {r['measure']}, {labels.get(r['model_a'], r['model_a'])} - "
                      f"{labels.get(r['model_b'], r['model_b'])}", test="mcnemar" if mc else "bootstrap",
                      over="variants", n=int(r["n"]), estimate=r["a_minus_b"], ci95_lo=r["lo"], ci95_hi=r["hi"],
                      p=r["p_mcnemar"] if mc else r["p_boot"], p_method="exact binomial" if mc else "bootstrap",
                      higher=r["a_right_b_wrong"] if mc else NAN, lower=r["a_wrong_b_right"] if mc else NAN,
                      note="higher, lower: the leads only A, only B calls the right way" if mc else "")  # fmt: skip

    # The shares again from rows.tsv.gz (each lead's predictions), against metrics.tsv.
    if (d / "rows.tsv.gz").is_file():
        rows = pd.read_csv(rep.use("level6/rows", d / "rows.tsv.gz"), sep="\t", keep_default_na=False,
                           na_values=["NA", ""])  # fmt: skip
        pairs = []
        for setting in ("h2h", "heldout"):
            cols = [c for c in rows.columns if c.startswith(f"{setting}.")]
            for sub, maxbp in subsets.items():
                sel = (rows["dist_to_centre"] <= maxbp).to_numpy() & rows[cols].notna().all(axis=1).to_numpy()
                e = rows["effect"].to_numpy(float)[sel]
                for c in cols:
                    p = rows[c].to_numpy(float)[sel]
                    ok = (p != 0) & (e != 0)
                    theirs = met[(met["setting"] == setting) & (met["subset"] == sub) & (met["group"] == "all")
                                 & (met["model"] == c.split(".", 1)[1])]["concordance"]  # fmt: skip
                    pairs.append((float(np.mean(np.sign(p[ok]) == np.sign(e[ok]))) if ok.any() else NAN,
                                  float(theirs.iloc[0]) if len(theirs) else NAN))  # fmt: skip
        rep.check("level 6: sign concordance from rows.tsv.gz (h2h and heldout, every subset, all groups)", pairs)

    # Per dataset alone: how many datasets Cherimoya is ahead of ChromBPNet corrected in.
    wins = (0, 0)
    pp = per[per["subset"] == primary].pivot_table(index="dataset", columns="model", values="concordance")
    if {"cherimoya", "cbp_nobias"} <= set(pp.columns):
        both = pp.dropna(subset=["cherimoya", "cbp_nobias"])
        wins = (int((both["cherimoya"] > both["cbp_nobias"]).sum()), len(both))
    groups = [g for g in dict.fromkeys(met.loc[(met["setting"] == "h2h") & (met["subset"] == primary), "group"])
              if g != "all"]  # fmt: skip
    l6 = {"state": "done", "dir": d, "info": info, "met": met, "par": par, "per": per, "cur": cur, "labels": labels,
          "subsets": subsets, "primary": primary, "groups": groups, "wins": wins}  # fmt: skip
    plot_level6(rep, l6)
    return l6


def l6_metric(l6: dict, setting: str, group: str, model: str, subset: str | None = None) -> dict | None:
    m = l6["met"]
    x = m[(m["setting"] == setting) & (m["subset"] == (subset or l6["primary"])) & (m["group"] == group)
          & (m["model"] == model)]  # fmt: skip
    return x.iloc[0].to_dict() if len(x) else None


def l6_pair(l6: dict, setting: str, group: str, a: str, b: str, measure: str = "concordance",
            subset: str | None = None) -> dict | None:  # fmt: skip
    p = l6["par"]
    x = p[(p["setting"] == setting) & (p["subset"] == (subset or l6["primary"])) & (p["group"] == group)
          & (p["model_a"] == a) & (p["model_b"] == b) & (p["measure"] == measure)]  # fmt: skip
    return x.iloc[0].to_dict() if len(x) else None


def h2h_datasets(l6: dict) -> list[str]:
    return sorted({x for g in (l6["info"].get("datasets") or {}).values() for x in g.get("h2h", [])})


def group_name(g: str) -> str:
    return "all groups" if g == "all" else g.replace("_", " ")


def points(x, d: int = 1) -> str:
    """A difference of two shares in percentage points, signed: '+1.5'."""
    return num(100 * float(x), d, True) if isnum(x) else "NA"


def share_ci(lo, hi, d: int = 1) -> str:
    return f"[{num(100 * float(lo), d)}, {num(100 * float(hi), d)}]" if isnum(lo) and isnum(hi) else ""


def plain_p(p) -> str:
    """p without HTML (for the summary's tiles): '1.4×10⁻⁷'."""
    if not isnum(p):
        return "NA"
    p = float(p)
    if p < 1e-3:
        mant, exp = f"{p:.1e}".split("e")
        return f"{mant}×10{str(int(exp)).translate(str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹'))}"
    return f"{p:.2g}" if p < 0.1 else f"{p:.2f}"


def plot_level6(rep: Report, l6: dict) -> None:
    plt = pyplot()
    from matplotlib.lines import Line2D

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.8, 4.3), gridspec_kw={"width_ratios": [1.1, 1.35, 0.9]})
    sub = l6["primary"]
    cur = l6["cur"][l6["cur"]["subset"] == sub]
    for m in ("cbp_bias", "cbp_nobias", "cherimoya"):
        c = cur[cur["model"] == m].sort_values("top_fraction")
        if len(c):
            col, ls = VARIANT_STYLE[m]
            ax1.plot(100 * c["top_fraction"], c["concordance"], color=col, ls=ls, lw=1.7, marker="o", markersize=2.6,
                     label=l6["labels"].get(m, m))  # fmt: skip
    ax1.axhline(0.5, color=MUTED, lw=0.8, ls=":")
    ax1.text(99, 0.507, "chance", ha="right", va="bottom", fontsize=7.2, color=MUTED)
    ax1.set_xlim(0, 102)
    ax1.set_ylim(0.45, 1.01)
    ax1.set_xlabel("the model's most confident leads, % (by |logFC|)")
    ax1.set_ylabel("share with the measured direction")
    ax1.set_title("Direction right, by confidence")
    handles, names = ax1.get_legend_handles_labels()
    ax1.legend(handles[::-1], names[::-1], loc="center right")

    rows = ["all"] + l6["groups"]
    ys = np.arange(len(rows))[::-1].astype(float)
    ticks = []
    for g, y in zip(rows, ys):
        base = l6_metric(l6, "h2h", g, "cherimoya")
        ticks.append(f"{group_name(g)} ({count(base['n']) if base else 'NA'})")
        for setting, off, filled in (("h2h", 0.13, True), ("heldout", -0.13, False)):
            r = l6_pair(l6, setting, g, "cherimoya", "cbp_nobias")
            if r is None:
                continue
            ax2.plot([100 * r["lo"], 100 * r["hi"]], [y + off] * 2, color=INK, lw=1.1, solid_capstyle="butt")
            ax2.scatter([100 * r["a_minus_b"]], [y + off], s=24, color=INK if filled else "white", edgecolor=INK,
                        linewidth=1.1, zorder=3)  # fmt: skip
    ax2.axvline(0, color=MUTED, lw=0.9)
    ax2.set_yticks(ys, ticks)
    ax2.set_ylim(-0.6, len(rows) - 0.4)
    ax2.grid(axis="y", visible=False)
    ax2.set_xlabel("Cherimoya − ChromBPNet corrected, percentage points")
    ax2.set_title("Difference by caQTL group (leads)")
    ax2.legend(handles=[Line2D([], [], ls="", marker="o", color=INK, markersize=5, label="fold mean"),
                        Line2D([], [], ls="", marker="o", markerfacecolor="white", markeredgecolor=INK, markersize=5,
                               label="held-out fold")], loc="lower right")  # fmt: skip

    pp = l6["per"][l6["per"]["subset"] == sub].pivot_table(index="dataset", columns="model", values="concordance")
    if {"cherimoya", "cbp_nobias"} <= set(pp.columns):
        b = pp.dropna(subset=["cherimoya", "cbp_nobias"])
        x, y = 100 * b["cbp_nobias"].to_numpy(float), 100 * b["cherimoya"].to_numpy(float)
        lo, hi = float(np.floor(min(x.min(), y.min()) - 1)), float(np.ceil(max(x.max(), y.max()) + 1))
        ax3.plot([lo, hi], [lo, hi], color=MUTED, lw=0.9, zorder=1)
        ax3.scatter(x, y, s=26, color=COLOURS["cross"], edgecolor="white", linewidth=0.5, zorder=3)
        ax3.set_xlim(lo, hi)
        ax3.set_ylim(lo, hi)
    ax3.set_xlabel("ChromBPNet corrected, % right")
    ax3.set_ylabel("Cherimoya, % right")
    ax3.set_title("Each dataset alone")
    fig.tight_layout()
    rep.figure(fig, "level6_caqtl", "Left: the share of caQTL leads whose predicted direction matches the measured "
               "one, among each model's most confident leads. Middle: Cherimoya minus ChromBPNet corrected by caQTL "
               "group, with 95% CIs. Right: each dataset alone, ChromBPNet corrected against Cherimoya.")  # fmt: skip


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
    """x to d decimals; a value that is not 0 but rounds to 0 gets one more decimal when that shows it, and a value
    that still rounds to 0 is shown unsigned, so a CI end never reads −0.000."""
    if not isnum(x):
        return "NA"
    v = float(x)
    if v != 0 and round(v, d) == 0:
        first = -int(np.floor(np.log10(abs(v))))
        if first <= d + 1:
            d = first
    if round(v, d) == 0:
        return f"{0:.{d}f}"  # no sign on a zero
    s = f"{v:+.{d}f}" if signed else f"{v:.{d}f}"
    return s.replace("-", "−")


def ci(lo, hi, d: int = 3, signed: bool = False) -> str:
    if not (isnum(lo) and isnum(hi)):
        return ""
    return f"[{num(lo, d, signed)}, {num(hi, d, signed)}]"


APPROX = '<span class="approx" title="normal approximation">†</span>'
"""Marks a p-value from the normal approximation (p_method 'normal'); every other p on the page is exact."""


def pval(p, method: str = "") -> str:
    if not isnum(p):
        return "NA"
    p = float(p)
    mark = APPROX if method == "normal" else ""
    if p < 1e-3:
        mant, exp = f"{p:.1e}".split("e")
        return f"{mant}×10<sup>{int(exp)}</sup>".replace("-", "−") + mark
    return (f"{p:.2g}" if p < 0.1 else f"{p:.2f}") + mark


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


def sign_counts(row: dict) -> str:
    """How many differences were above 0 ('7/10 above 0'), and below and at 0 when some were 0 (the test drops those):
    '24 above, 3 below, 23 at 0'."""
    n = int(row["n"]) if isnum(row["n"]) else 0
    hi = int(row["higher"]) if isnum(row["higher"]) else 0
    lo = int(row["lower"]) if isnum(row["lower"]) else 0
    if hi + lo < n:
        return f"{hi} above, {lo} below, {n - hi - lo} at 0"
    return f"{hi}/{n} above 0"


def ds_counts(row: dict) -> str:
    """'10/10 datasets' (above 0), or sign_counts' form when some differences were 0."""
    n = int(row["n"]) if isnum(row["n"]) else 0
    hi = int(row["higher"]) if isnum(row["higher"]) else 0
    lo = int(row["lower"]) if isnum(row["lower"]) else 0
    return f"{hi}/{n} datasets" if hi + lo == n else sign_counts(row)


def test_cell(row: dict | None, d: int = 3) -> str:
    """The median difference, its CI, how many were above (below, at) 0, and p."""
    if row is None:
        return "NA"
    return (f'{num(row["estimate"], d, True)}<span class="sub">{ci(row["ci95_lo"], row["ci95_hi"], d, True)} · '
            f'{sign_counts(row)} · p {pval(row["p"], row.get("p_method", ""))}</span>')  # fmt: skip


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


def table_html(n: Numbering, caption: str, header: list[str], rows: list[list[str]], left: int | set[int] = 1,
               cls: str = "") -> str:  # fmt: skip
    """A numbered table; the first column heads the rows. left: how many leading columns are text (left-aligned), or
    the set of their indexes."""
    k = n.table()
    is_left = (lambda i: i < left) if isinstance(left, int) else (lambda i: i in left)
    head = "".join(f'<th scope="col"{' class="l"' if is_left(i) else ""}>{c}</th>' for i, c in enumerate(header))
    body = "".join("<tr>" + "".join((f'<th scope="row" class="l">{c}</th>' if i == 0 else
                                     f'<td{" class=\"l\"" if is_left(i) else ""}>{c}</td>') for i, c in enumerate(r))
                   + "</tr>" for r in rows)  # fmt: skip
    return (
        f'<div class="tblock"><p class="tcap"><b>Table {k}.</b> {caption}</p><div class="tbl {cls}">'
        f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div></div>"
    )


def note_html(state: str, label: str, body: str) -> str:
    return f'<div class="note"><span class="chip {state}">{h(label)}</span><div>{body}</div></div>'


def pending_eta(rep: Report) -> str:
    """--pending-eta as its own sentence (with a leading space), or ''."""
    eta = (rep.args.pending_eta or "").strip()
    if not eta:
        return ""
    return f" {h(eta if eta[-1] in '.!?' else eta + '.')}"


def pending_html(rep: Report, part: str) -> str:
    out = []
    for p in rep.pending:
        if p["part"] != part:
            continue
        names = p.get("detail") or []
        more = (f" Still missing: {h('; '.join(str(x) for x in names[:6]))}{' …' if len(names) > 6 else ''}."
                if names else "")  # fmt: skip
        note = f" {h(p['note'])}." if p.get("note") else ""
        out.append(note_html("pending", "Pending", f"<p>{h(p['what'])}.{note}{more}{pending_eta(rep)}</p>"))
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
   sit on light plates in both themes; every table scrolls inside its own frame. The two models' hues are the page's
   saturated colours, with the Tn5 orange of the bias control and the status chips' muted tints. */
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
.masthead > *, .head > *, figure > *, .findings li > * { min-width: 0; }
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
.note p { max-width: 80ch; overflow-wrap: anywhere; }
.approx { margin-left: 1px; }
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
  .plate img.wide { min-width: 900px; }  /* phones: scroll the plate at a legible scale */
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
    l6 = R.get("l6")
    if l6 is None:
        chip("Level 6 · not computed", "problem")
    elif l6.get("state") != "done":
        chip("Level 6 · pending", "pending")
    else:
        chip("Level 6 · done", "done")
    for p in rep.pending:
        if p.get("key") in ("supplement", "production"):
            chip(f"{p['label']} · pending", "pending")
    eta = pending_eta(rep) if rep.pending else ""
    variants = ""
    if l6 and l6.get("state") == "done":
        k = len(h2h_datasets(l6))
        variants = " The variant effects, compared with measured caQTLs, cover " + (
            "those datasets." if k == len(rep.shared) else f"the {k} of them that stand for a caQTL group."
        )
    return f"""
<header class="masthead">
  <p class="eyebrow">AMSC · 10x multiome ATAC · model comparison</p>
  <h1>Cherimoya vs ChromBPNet</h1>
  <p class="dek">Two sequence-to-accessibility models trained on the same {len(rep.everything)} AMSC pseudobulk ATAC
  datasets with the same five chromosome folds. Performance covers every dataset; attributions, TF-MoDISco motifs,
  motif compendia and Fi-NeMo hits cover the {len(rep.shared)} datasets both models were interpreted on.{variants}</p>
  <div class="legend"><span class="lg">{key("chrombpnet")}<span class="lgd">corrected model (chrombpnet_nobias.h5) for attributions and
  variant effects, uncorrected model (chrombpnet.h5) for performance</span></span><span class="lg">{key("cherimoya")}<span class="lgd">no bias model</span></span>
  <span class="lg">{key("cross")}<span class="lgd">agreement between the two</span></span></div>
  <div class="status">{"".join(chips)}</div>
  <p class="meta"><span>Generated {h(generated)}</span><span>Data as of the newest input,
  {h(R.get("newest_input", "NA"))}</span><span>Pending parts are marked as such.{eta}</span></p>
</header>"""


def toc() -> str:
    items = [
        ("summary", "", "Summary"),
        ("level-1", "1", "Performance"),
        ("level-2", "2", "Attributions"),
        ("level-3", "3", "TF-MoDISco"),
        ("level-4", "4", "Compendia"),
        ("level-5", "5", "Fi-NeMo hits"),
        ("level-6", "6", "Variant effects"),
        ("bias-control", "", "Bias control"),
        ("caveats", "", "Caveats"),
        ("methods", "", "Methods and provenance"),
    ]
    links = "".join(f'<a href="#{i}">{f'<span class="n">{n}</span>' if n else ""}{h(t)}</a>' for i, n, t in items)
    return f'<nav class="toc" aria-label="Sections">{links}</nav>'


def s(rep: Report, level: str, family: str, quantity: str, over: str = "datasets") -> dict | None:
    return rep.stats.find(level, family, quantity, over)


# ---------------------------------------------------------------------------------------------------------------------
# the findings: each level's sentences in one place, read by the Summary and by the level's lead


def ordinal(k: int) -> str:
    """'' for 1, 'second-' for 2, ... (as in 'the third-largest')."""
    return {1: "", 2: "second-", 3: "third-", 4: "fourth-", 5: "fifth-"}.get(k, f"{k}th-")


def ranges(values: list[int]) -> str:
    """[1, 2, 3, 5] -> '1–3 and 5'; [1, 2, 9, 10] -> '1, 2, 9 and 10'."""
    runs: list[list[int]] = []
    for v in sorted(values):
        if runs and v == runs[-1][-1] + 1:
            runs[-1].append(v)
        else:
            runs.append([v])
    parts = [p for r in runs for p in ([f"{r[0]}–{r[-1]}"] if len(r) > 2 else [str(x) for x in r])]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def change(x: float, d: int = 4) -> str:
    """'falls by 0.0011' or 'rises by 0.0011'."""
    return f"{'falls' if x < 0 else 'rises'} by {num(abs(x), d)}"


def relation(b: float, c1: float, c2: float) -> str:
    """Where b sits against two ceilings."""
    if b < min(c1, c2):
        return "sit below both ceilings"
    if b > max(c1, c2):
        return "sit above both ceilings"
    return "fall between the two ceilings"


def l1_finding(l1: dict) -> dict:
    """Level 1: 'headline' (who predicts better, by how much), 'detail' (the dataset difference, the folds) and
    'where' (where the gain sits, from the residual deciles; '' when no decile is lower at p < 0.01)."""
    md = l1["measures"]
    a = md[md["set"] == "all"].set_index("measure")
    nf, nd = l1["n_folds"], l1["n_datasets"]
    pear, jsd = a.loc["counts_pearson"], a.loc["median_jsd"]
    full = [m for m in MEASURES if a.loc[m, "cherimoya_better_folds"] == nf]
    part = [m for m in MEASURES if m not in full]
    all_ds = all(a.loc[m, "cherimoya_better_datasets"] == nd for m in MEASURES)
    headline = (f"Cherimoya predicts held-out accessibility better than ChromBPNet"
                f"{f' in all {nd} datasets, on all five measures' if all_ds else ''}: over {nf} test folds the median "
                f"counts Pearson r rises from {num(pear['chrombpnet_median'])} to {num(pear['cherimoya_median'])} and "
                f"the median profile JSD falls from {num(jsd['chrombpnet_median'])} to "
                f"{num(jsd['cherimoya_median'])}.")  # fmt: skip
    detail = (
        f"For counts Pearson r, the median over datasets of each dataset's mean fold difference is "
        f"{num(pear['median_dataset_diff'], 3, True)} (95% CI {ci(pear['ci95_lo'], pear['ci95_hi'], 3, True)})."
    )
    if full:
        detail += f" Cherimoya is better on every fold for {', '.join(MEASURE_SHORT[m] for m in full)}"
        detail += (f"; for {', '.join(MEASURE_SHORT[m] for m in part)} on "
                   + ", ".join(f"{int(a.loc[m, 'cherimoya_better_folds'])}" for m in part) + f" of {nf}."
                   if part else ".")  # fmt: skip
    rd = l1["residuals"]
    r = rd[rd["set"] == "all"].sort_values("decile")
    lower = [int(x) for x in r[(r["rmse_diff_median"] < 0) & (r["wilcoxon_p"] < 0.01)]["decile"]]
    higher = [int(x) for x in r[(r["rmse_diff_median"] > 0) & (r["wilcoxon_p"] < 0.01)]["decile"]]
    where = ""
    if lower:
        d1 = r[r["decile"] == min(lower)].iloc[0]
        where = "The gain sits at the extremes of accessibility. " if {1, 10} <= set(lower) else ""
        where += (f"Cherimoya's RMSE is lower in deciles "
                 f"{ranges(lower)} of observed counts (decile {int(d1['decile'])}: "
                 f"{num(d1['rmse_diff_median'], 3, True)} log counts)")  # fmt: skip
        where += (
            f" and slightly higher in deciles {ranges(higher)} (at most "
            f"{num(r[r['decile'].isin(higher)]['rmse_diff_median'].max(), 3, True)})."
            if higher
            else "."
        )
        first, last = r.iloc[0], r.iloc[-1]
        if min(first["mean_residual_chrombpnet_median"], first["mean_residual_cherimoya_median"]) > 0 and max(
                last["mean_residual_chrombpnet_median"], last["mean_residual_cherimoya_median"]) < 0:  # fmt: skip
            where += " Both models over-predict the least accessible peaks and under-predict the most accessible."
        below = [
            int(x) for x in r[r["mean_residual_cherimoya_median"] < r["mean_residual_chrombpnet_median"]]["decile"]
        ]
        if below:
            mid = r[r["decile"] == below[len(below) // 2]].iloc[0]
            where += (f" Cherimoya's predictions sit lower than ChromBPNet's in deciles {ranges(below)} (median mean "
                      f"residual in decile {int(mid['decile'])}: {num(mid['mean_residual_cherimoya_median'], 2, True)} "
                      f"against {num(mid['mean_residual_chrombpnet_median'], 2, True)}), a downward shift that shrinks "
                      "its over-prediction at the low end and leaves it under-predicting more in the middle deciles, "
                      "where its RMSE is higher.")  # fmt: skip
    return {"headline": headline, "detail": detail, "where": where, "lower": lower}


def l2_finding(rep: Report, l2: dict) -> dict | None:
    """Level 2: 'headline' and 'caveat' (fold averages against the ceilings), 'spread' (the other metrics), 'single'
    (single folds), 'lead' (one sentence for the section), 'heldout' (held-out against training chromosomes)."""
    w = l2["windows"][0]
    q = {(m, x): s(rep, "2", f"attributions, {w}, {m}", x) for m in L2_METRICS
         for x in ("cross_fold", "cross_average", "loo_chrombpnet", "loo_cherimoya")}  # fmt: skip
    A, B, Cc, Ck = (q[("pearson", x)] for x in ("cross_fold", "cross_average", "loo_chrombpnet", "loo_cherimoya"))
    if not (A and B and Cc and Ck):
        return None
    out = {
        "window": w,
        "headline": f"On Pearson r, the fold-averaged attributions of the two models agree per peak at "
        f"{num(B['estimate'])} in the {window_name(w)}, about as high as one fold agrees with the mean of its own "
        f"model's other four (the leave-one-out ceilings: {num(Cc['estimate'])} for ChromBPNet, "
        f"{num(Ck['estimate'])} for Cherimoya).",
        "caveat": "That comparison favours the fold averages, which average five folds on each side, so it does not "
        "show that the two models agree as well as each agrees with itself; a like-for-like ceiling was not computed.",
        "single": f"Single folds agree less across the models ({num(A['estimate'])}), a comparison tilted the other "
        "way, since one fold is noisier than the mean of four.",
        "lead": f"Per peak, the fold-averaged attributions of the two models correlate at Pearson r "
        f"{num(B['estimate'])} in the {window_name(w)}, close to each model's leave-one-out ceiling "
        f"({num(Cc['estimate'])} and {num(Ck['estimate'])}), but that comparison favours the fold averages (five "
        f"folds averaged on each side, against one fold against the mean of four); single folds agree at "
        f"{num(A['estimate'])}.",
    }
    spread = []
    for m, what in (("spearman", "Spearman ρ"), ("jaccard_mean", "the top-50 Jaccard")):
        b, c1, c2 = (q[(m, x)] for x in ("cross_average", "loo_chrombpnet", "loo_cherimoya"))
        if b and c1 and c2:
            spread.append(f"on {what} the fold averages ({num(b['estimate'])}) "
                          f"{relation(b['estimate'], c1['estimate'], c2['estimate'])} ({num(c1['estimate'])} for "
                          f"ChromBPNet, {num(c2['estimate'])} for Cherimoya)")  # fmt: skip
    out["spread"] = ("Other metrics give a mixed picture: " + "; ".join(spread) + ".") if spread else ""
    fam = f"attributions, {w}, pearson"
    ho = [s(rep, "2", fam, f"{x} - {y}") for x, y in (("heldout_chrombpnet", "train_chrombpnet"),
                                                     ("heldout_cherimoya", "train_cherimoya"),
                                                     ("cross_heldout", "cross_train"))]  # fmt: skip
    role_names = {"cross_fold": "cross-model agreement (A)", "loo_chrombpnet": "ChromBPNet's ceiling",
                  "loo_cherimoya": "Cherimoya's ceiling"}  # fmt: skip
    role = {k: s(rep, "2", f"attribution strata, {w}, pearson", f"role test - train: {k}") for k in role_names}
    role = {k: v for k, v in role.items() if v}
    out["heldout"] = ""
    if all(ho):
        rows = ho + list(role.values())
        worst = max(abs(x["estimate"]) for x in rows)
        est = [x["estimate"] for x in ho]
        if all(e < 0 for e in est) or all(e > 0 for e in est):
            by = (f"{'falls' if est[0] < 0 else 'rises'} by {num(abs(est[0]), 4)} for ChromBPNet, "
                  f"{num(abs(est[1]), 4)} for Cherimoya and {num(abs(est[2]), 4)} across the models")  # fmt: skip
        else:
            by = (f"{change(est[0])} for ChromBPNet, {change(est[1])} for Cherimoya and {change(est[2])} across "
                  "the models")  # fmt: skip
        text = (
            f"On chromosomes a fold never trained on, attribution agreement differs by at most {num(worst, 4)} in "
            f"Pearson r ({window_name(w, full=False)}). Per peak at the held-out fold against the folds that "
            f"trained on the chromosome, it {by}"
        )
        if role:
            k, r = max(role.items(), key=lambda kv: abs(kv[1]["estimate"]))
            text += f"; split by the chromosome's role in each fold, {role_names[k]} {change(r['estimate'])}"
        lower_all = all(x["estimate"] < 0 and int(x["higher"]) == 0 for x in rows)
        text += (". The drop holds in every dataset but is tiny: no sign of meaningful memorization." if lower_all
                 else ". The differences are tiny: no sign of meaningful memorization.")  # fmt: skip
        out["heldout"] = text
    return out


def l3_finding(rep: Report) -> dict | None:
    """Level 3: 'primary' (the cluster rule), 'secondary' (tomtom), 'qc' (the QC scale against its ceilings),
    'lead', and the tile's numbers."""
    fam = "TF-MoDISco, per model"

    def med(col: str, m: str) -> float:
        r = s(rep, "3", fam, f"{NAMES[m]}: {col}")
        return r["estimate"] if r else NAN

    v = {(col, m): med(col, m) for m in MODELS for col in ("shared_cluster_count", "shared_cluster_seqlets",
                                                           "shared_tomtom_count", "shared_tomtom_seqlets",
                                                           "qc_cross", "qc_within")}  # fmt: skip
    if not all(isnum(x) for x in v.values()):
        return None
    famq = "TF-MoDISco QC, cross-model - ceiling"
    dq = {m: s(rep, "3", famq, f"QC: {m} patterns matched across models - within {m} (ceiling)") for m in MODELS}

    def two(col: str, d: int = 1) -> tuple[str, str]:
        return pct(v[(col, "chrombpnet")], d), pct(v[(col, "cherimoya")], d)

    pc, pk = two("shared_cluster_count")
    sc, sk = two("shared_cluster_seqlets")
    tc, tk = two("shared_tomtom_count")
    ttc, ttk = two("shared_tomtom_seqlets")
    qc = (num(v[("qc_cross", "chrombpnet")], 2), num(v[("qc_cross", "cherimoya")], 2))
    qw = (num(v[("qc_within", "chrombpnet")], 2), num(v[("qc_within", "cherimoya")], 2))
    out = {
        "primary": f"By the primary rule, a partner in the same joint-compendium cluster, {pc} of ChromBPNet's and "
        f"{pk} of Cherimoya's fold-average patterns are matched (medians over datasets), and those hold {sc} and "
        f"{sk} of the seqlets.",
        "secondary": "The rule is strict, since at threshold 0.95 the joint compendium keeps close variants of a "
        f"motif apart: by tomtom (secondary), {tc} and {tk} of the patterns have a partner in the other model, "
        f"holding {ttc} and {ttk} of the seqlets.",
        "qc": f"At QC scale the cross-model match ({qc[0]} and {qc[1]}) comes close to each model's fold-to-fold "
        f"ceiling ({qw[0]} and {qw[1]}), but that comparison favours the cross-model match: both models' runs of a "
        "fold use the same 30K-peak subsample, while the ceiling compares folds whose subsamples differ.",
        "tile": (
            f"{pc} / {pk}",
            f"ChromBPNet / Cherimoya, median of datasets; by tomtom {tc} / {tk}. No ceiling here; "
            f"at QC scale {qc[0]} / {qc[1]} against fold-to-fold ceilings {qw[0]} / {qw[1]}",
        ),
    }
    lead = (f"{pc} of ChromBPNet's and {pk} of Cherimoya's fold-average patterns share a joint-compendium cluster with "
            f"a pattern of the other model (medians over datasets), and those patterns hold {sc} and {sk} of the "
            f"seqlets; by tomtom, {tc} and {tk} have a partner.")  # fmt: skip
    if all(dq.values()):
        lead += (f" At QC scale the cross-model match sits {num(abs(dq['chrombpnet']['estimate']), 3)} and "
                 f"{num(abs(dq['cherimoya']['estimate']), 3)} below each model's fold-to-fold ceiling, a comparison "
                 "that favours the cross-model match.")  # fmt: skip
    out["lead"] = lead
    return out


def l4_finding(rep: Report, l4: dict, l5: dict | None) -> dict | None:
    """Level 4: 'text' (the joint compendium's classes, the threshold 0.90 rebuild, the hits the model-only clusters
    draw) and 'lead'."""
    c = l4["classes"]
    c95 = c[c["threshold"] == "0.95"].set_index("class")
    c90 = c[c["threshold"] == "0.90"].set_index("class") if "0.90" in set(c["threshold"]) else None
    if not set(CLASSES) <= set(c95.index):
        return None
    n = {k: int(c95.loc[k, "clusters"]) for k in CLASSES}
    single = {k: int(c95.loc[k, "single_pattern_clusters"]) for k in CLASSES}
    text = (f"In the joint compendium, {n['shared']} clusters hold patterns of both models and carry "
            f"{pct(c95.loc['shared', 'seqlet_share_chrombpnet'])} of ChromBPNet's and "
            f"{pct(c95.loc['shared', 'seqlet_share_cherimoya'])} of Cherimoya's seqlets. The {n['chrombpnet_only']} "
            f"ChromBPNet-only and {n['cherimoya_only']} Cherimoya-only clusters are small by seqlets "
            f"({pct(c95.loc['chrombpnet_only', 'seqlet_share_chrombpnet'])} and "
            f"{pct(c95.loc['cherimoya_only', 'seqlet_share_cherimoya'])} of their model's) and mostly single patterns "
            f"({single['chrombpnet_only']} and {single['cherimoya_only']})")  # fmt: skip
    if c90 is not None and set(CLASSES) <= set(c90.index):
        text += (f", and many merge when the compendium is rebuilt at threshold 0.90 "
                 f"({int(c90.loc['chrombpnet_only', 'clusters'])} and {int(c90.loc['cherimoya_only', 'clusters'])}, "
                 f"while the shared core keeps {int(c90.loc['shared', 'clusters'])}).")  # fmt: skip
    else:
        text += "."
    hits = ""
    if l5 is not None and l5.get("state") == "done":
        hc = s(rep, "5", "Fi-NeMo hits, class chrombpnet_only", "share_of_chrombpnet_hits")
        hk = s(rep, "5", "Fi-NeMo hits, class cherimoya_only", "share_of_cherimoya_hits")
        if hc and hk:
            hits = (f" They matter more for the hits: their motifs draw {pct(hc['estimate'])} of ChromBPNet's and "
                    f"{pct(hk['estimate'])} of Cherimoya's Fi-NeMo hits (medians over datasets)")  # fmt: skip
            top = l5["top"].reset_index(drop=True)
            own = top[top["class"].isin(("chrombpnet_only", "cherimoya_only"))]
            if len(own):
                i = int(own.index[0])
                r = top.loc[i]
                m = "cherimoya" if r["class"] == "cherimoya_only" else "chrombpnet"
                hits += (f", and the {NAMES[m]}-only {h(r['annotation'] or r['motif'])} is the {ordinal(i + 1)}largest "
                         f"motif ({count(r[f'hits_{m}'])} {NAMES[m]} hits)")  # fmt: skip
            hits += "."
    lead = (f"{n['shared']} clusters of the joint compendium hold patterns of both models and carry "
            f"{pct(c95.loc['shared', 'seqlet_share_chrombpnet'])} and "
            f"{pct(c95.loc['shared', 'seqlet_share_cherimoya'])} of the two models' seqlets; the "
            f"{n['chrombpnet_only']} ChromBPNet-only and {n['cherimoya_only']} Cherimoya-only clusters are mostly "
            "single patterns with little of the seqlets.")  # fmt: skip
    return {"text": text + hits, "lead": lead + hits}


def l5_finding(rep: Report) -> dict | None:
    """Level 5: 'text' (hit counts, precision and recall, tolerance, by class) and 'lead'."""
    fam = "Fi-NeMo hits, joint vocabulary"
    P, Rr = s(rep, "5", fam, "precision_5bp"), s(rep, "5", fam, "recall_5bp")
    ratio = s(rep, "5", fam, "hits, ChromBPNet / Cherimoya")
    if not (P and Rr and ratio):
        return None
    tol = {t: (s(rep, "5", fam, f"precision_{t}bp"), s(rep, "5", fam, f"recall_{t}bp")) for t in (0, 10)}
    text = (f"Called with one shared motif vocabulary, ChromBPNet has {num(ratio['estimate'], 2)}× as many hits as "
            f"Cherimoya. {pct(P['estimate'], 0)} of Cherimoya's hits match a ChromBPNet hit of the same motif and "
            f"strand within 5 bp (precision), and Cherimoya recovers {pct(Rr['estimate'], 0)} of ChromBPNet's "
            f"(recall).")  # fmt: skip
    if all(a and b for a, b in tol.values()):
        text += (f" Matched hits sit at the same position: at 0 bp the shares are already {num(tol[0][0]['estimate'])} "
                 f"and {num(tol[0][1]['estimate'])}, and at 10 bp {num(tol[10][0]['estimate'])} and "
                 f"{num(tol[10][1]['estimate'])}.")  # fmt: skip
    cls = {c: (s(rep, "5", f"Fi-NeMo hits, class {c}", "precision_5bp"),
               s(rep, "5", f"Fi-NeMo hits, class {c}", "recall_5bp")) for c in CLASSES}  # fmt: skip
    by_class = ""
    if all(a and b for a, b in cls.values()):
        best_p = max(CLASSES, key=lambda c: cls[c][0]["estimate"])
        best_r = max(CLASSES, key=lambda c: cls[c][1]["estimate"])
        if best_p == best_r == "shared":
            by_class = (f" Agreement is highest on motifs of clusters both models hold (precision "
                        f"{num(cls['shared'][0]['estimate'])}, recall {num(cls['shared'][1]['estimate'])}) and "
                        f"Cherimoya's hits on Cherimoya-only motifs are the ones ChromBPNet reproduces least "
                        f"(precision {num(cls['cherimoya_only'][0]['estimate'])}).")  # fmt: skip
    lead = (
        f"With one shared motif vocabulary, ChromBPNet calls {num(ratio['estimate'], 2)}× as many hits as Cherimoya; "
        f"{pct(P['estimate'], 0)} of Cherimoya's hits have a ChromBPNet hit of the same motif and strand within 5 "
        f"bp (precision), and Cherimoya recovers {pct(Rr['estimate'], 0)} of ChromBPNet's (recall)"
        + (", with the highest agreement on motifs of clusters both models hold." if by_class else ".")
    )
    return {"text": text + by_class, "lead": lead, "tile": (P, Rr)}


def bias_finding(rep: Report, bc: dict, l3: dict) -> dict:
    """The bias control: 'flagged' (the database flags of both models), 'checked' (the control's split), 'lead',
    'closing' (what the split means), and the counts."""
    per = l3["per"]
    nd = per["dataset"].nunique()
    fc = int(per[per["model"] == "chrombpnet"]["bias_flagged"].sum())
    fk = int(per[per["model"] == "cherimoya"]["bias_flagged"].sum())
    t = s(rep, "3", "TF-MoDISco, Cherimoya - ChromBPNet", "Tn5/DNase-flagged patterns: Cherimoya - ChromBPNet")
    split = bc["split"].set_index("reference_kind")

    def get(kind: str, col: str) -> int:
        return int(split.loc[kind, col]) if kind in split.index and col in split.columns else 0

    n = {k: get(k, "patterns") for k in ("Tn5/DNase", "TF or other", "no match", "unknown")}
    partner = {k: get(k, "chrombpnet_partner") for k in n}
    flagged = {k: get(k, "db_flagged") for k in n}
    n_checked = int(split["patterns"].sum())
    scales = bc.get("scales") or {}
    n_fa, n_qc = scales.get("fold average", 0), scales.get("QC fold", 0)
    p = f"p {pval(t['p'], t['p_method'])}" if t else "p NA"
    out = {"fc": fc, "fk": fk, "n": n, "partner": partner, "db_flagged": flagged, "n_checked": n_checked}
    out["flag_text"] = (f"Cherimoya, which has no bias model, carries more Tn5/DNase-flagged patterns: {fk} "
                      f"fold-average patterns over the {nd} datasets against {fc} for ChromBPNet corrected ({p}). "
                      "ChromBPNet corrected has some too, so the difference is one of degree.")  # fmt: skip
    checked = (
        f"{n_fa} fold-average patterns without a cluster partner and {n_qc} of the control's own QC fold "
        f"without a tomtom partner"
        if n_qc
        else f"{n_fa} fold-average patterns without a cluster partner"
    )
    matched = n["Tn5/DNase"] + n["TF or other"]
    out["checked"] = (f"Of the {n_checked} Cherimoya patterns checked against ChromBPNet uncorrected "
                      f"({checked}), {matched} match one of its patterns: {n['Tn5/DNase']} its Tn5/DNase patterns "
                      f"and {n['TF or other']} its TF patterns, and {partner['TF or other']} of those "
                      f"{n['TF or other']} also have a tomtom partner in ChromBPNet corrected.")  # fmt: skip
    out["lead"] = (f"Cherimoya carries more Tn5/DNase-flagged patterns than ChromBPNet corrected ({fk} against "
                   f"{fc} over the {nd} datasets, {p}), but ChromBPNet corrected has some too, so the difference "
                   "is one of degree; and most of Cherimoya's unmatched patterns that ChromBPNet uncorrected "
                   f"also finds match its TF patterns ({n['TF or other']} of {matched}), not its Tn5 "
                   "patterns.")  # fmt: skip
    ref = bc["reference"]
    tf_names = []
    if ref is not None and not ref.empty:
        tf = ref[(ref["kind"] == "TF or other") & (ref["cherimoya_patterns"] > 0)]
        tf = tf.sort_values("cherimoya_patterns", ascending=False)
        tf_names = list(dict.fromkeys(str(x) for x in tf["annotation"].dropna()))
    names = f" ({', '.join(tf_names)})" if tf_names else ""
    out["closing"] = (
        f"Most of the Cherimoya patterns that the uncorrected model also finds match its TF patterns{names}, not "
        f"its Tn5 patterns: they are mostly TF motifs that the cluster rule left unmatched "
        f"({partner['TF or other']} of {n['TF or other']} have a tomtom partner in ChromBPNet corrected), "
        f"though {flagged['TF or other']} of them are themselves database-flagged Tn5/DNase. Of the "
        f"{n['Tn5/DNase']} that match its Tn5/DNase patterns, {partner['Tn5/DNase']} also have a tomtom partner in "
        f"ChromBPNet corrected, so it carries similar patterns too. The {n['no match']} checked patterns that "
        f"match no reference pattern are not cleared by this control: {partner['no match']} of them have a tomtom "
        f"partner in ChromBPNet corrected and {flagged['no match']} are database-flagged. The control is one "
        "dataset and fold, on the profile head (DeepSHAP has no rule for the counts head of the uncorrected model), while "
        "every other pattern here comes from the counts head, so it indicates, but does not measure, how much of "
        "Cherimoya's extra signal is bias."
    )
    return out


def summary_section(rep: Report, R: dict) -> str:
    items, keys = [], []
    l1, l2, l3, l4, l5, l6, bc = (R.get(k) for k in ("l1", "l2", "l3", "l4", "l5", "l6", "bias"))
    if l1 and l1.get("state") == "done":
        f1 = l1_finding(l1)
        items.append(("Performance", f"{f1['headline']} {f1['detail']}"))
        md = l1["measures"]
        pear = md[md["set"] == "all"].set_index("measure").loc["counts_pearson"]
        keys.append((f"Counts Pearson r, median of {l1['n_folds']} folds",
                     f"{num(pear['chrombpnet_median'])} → {num(pear['cherimoya_median'])}",
                     f"ChromBPNet → Cherimoya; Cherimoya better on {int(pear['cherimoya_better_folds'])}/"
                     f"{l1['n_folds']} folds"))  # fmt: skip
        if f1["where"]:
            items.append(("Where it gains", f1["where"]))
    if l2 and l2.get("state") == "done":
        f2 = l2_finding(rep, l2)
        if f2:
            items.append(("Attributions", " ".join(x for x in (f2["headline"], f2["caveat"], f2["spread"],
                                                                f2["single"]) if x)))  # fmt: skip
            fam = f"attributions, {f2['window']}, pearson"
            B = s(rep, "2", fam, "cross_average")
            Cc, Ck = s(rep, "2", fam, "loo_chrombpnet"), s(rep, "2", fam, "loo_cherimoya")
            keys.append(("Attribution Pearson r, fold averages (B)", num(B["estimate"]),
                         f"ceilings {num(Cc['estimate'])} (ChromBPNet), {num(Ck['estimate'])} (Cherimoya); not like "
                         "for like: B averages five folds on each side, a ceiling one fold against four"))  # fmt: skip
            if f2["heldout"]:
                items.append(("Held-out chromosomes", f2["heldout"]))
    if l3 and l3.get("state") == "done":
        f3 = l3_finding(rep)
        if f3:
            items.append(("Motifs", f"TF-MoDISco finds largely the same motifs in both models. {f3['primary']} "
                                    f"{f3['secondary']} {f3['qc']}"))  # fmt: skip
            keys.append(("Fold-average patterns in a joint cluster with the other model's", *f3["tile"]))
    f5 = l5_finding(rep) if l5 and l5.get("state") == "done" else None
    if l4 and l4.get("state") == "done":
        f4 = l4_finding(rep, l4, l5)
        if f4:
            items.append(("Compendia", f4["text"]))
    if f5:
        items.append(("Hits", f5["text"]))
        P, Rr = f5["tile"]
        keys.append(("Fi-NeMo hits matched within 5 bp", f"{pct(P['estimate'], 0)} / {pct(Rr['estimate'], 0)}",
                     "of Cherimoya's hits / of ChromBPNet's hits, median of datasets; no within-model ceiling (Fi-NeMo "
                     "ran on fold averages only)"))  # fmt: skip
    f6 = l6_finding(rep, l6) if l6 and l6.get("state") == "done" else None
    if f6:
        items.append(("Variant effects", f6["text"]))
        keys.append(f6["tile"])
    if bc and bc.get("state") == "done" and l3 and l3.get("state") == "done":
        fb = bias_finding(rep, bc, l3)
        items.append(("Bias", f"{fb['flag_text']} {fb['checked']}"))
    if rep.pending:
        what = list(dict.fromkeys(p.get("short") or p["what"] for p in rep.pending))
        items.append(("Pending", f"Not in this report yet: {h('; '.join(what))}.{pending_eta(rep)}"))
    if rep.problems:
        items.append(("Not computed", h("; ".join(f"{n}: {m}" for n, m in rep.problems))))
    lis = "".join(f'<li><span class="topic">{h(t)}</span><p>{x}</p></li>' for t, x in items)
    kd = "".join(f"<div><dt>{h(a)}</dt><dd>{b}<small>{h(c)}</small></dd></div>" for a, b, c in keys)
    n_sh, n_all = len(rep.shared), len(rep.everything)
    floors = ", ".join(f"{pval(2.0 / 2**k)} for {k} {what}" for k, what in ((n_sh, "datasets"), (n_all, "datasets"))
                       if k)  # fmt: skip
    floor = (f"<p class=\"tcap\">p-values are two-sided Wilcoxon signed-rank tests, except at level 6 (McNemar's "
             f"exact test on the variants, or a bootstrap p). With n paired values the exact Wilcoxon p "
             f"cannot fall below 2/2<sup>n</sup>: {floors}. p marked {APPROX} comes from the normal approximation, "
             f"used for more than 50 values (the folds of all datasets) or for ties or zeros among 14–50, and can fall "
             f"further. At those floors read p as “every dataset (or fold) went the same way”, not as a measure of "
             f"effect size.</p>")  # fmt: skip
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
    fnd = l1_finding(l1)
    gain = (" The gain is largest for the least and the most accessible peaks." if {1, 10} <= set(fnd["lower"])
            else "")  # fmt: skip
    lead = (
        f'<p class="lead">{fnd["headline"]}{gain} Both models were scored on ChromBPNet\'s test peaks of each fold '
        f"({count(round(l1['paired']['n_regions'].mean()))} peaks per fold on average) with ChromBPNet's own measures; "
        f"ChromBPNet is scored as its own evaluation scores it, with its uncorrected model (bias model included). {nf} folds over "
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
                pval(r["wilcoxon_datasets_p"], r["wilcoxon_datasets_p_method"]),
                pval(r["wilcoxon_folds_p"], r["wilcoxon_folds_p_method"]),
            ]
        )
    t1 = table_html(n, f"The five measures over all {nd} datasets. Medians over the {nf} folds; the difference is "
                    "Cherimoya minus ChromBPNet, as the median over datasets of each dataset's mean fold difference, "
                    f"with its bootstrap 95% CI. p from Wilcoxon signed-rank tests over the dataset means and over the "
                    f"folds ({APPROX}: normal approximation).",
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
                    pval(r["wilcoxon_datasets_p"], r["wilcoxon_datasets_p_method"]),
                    pval(r["wilcoxon_folds_p"], r["wilcoxon_folds_p_method"]),
                ]
            )
    t2 = (
        table_html(
            n,
            f"The same measures for the {len(rep.shared)} shared datasets (interpreted at levels 2–5) and for the "
            "other datasets, separately; Wilcoxon p over the dataset means and over the folds "
            f"({APPROX}: normal approximation). The shared-dataset values match level 1's own statistics.",
            [
                "Datasets",
                "Measure",
                "ChromBPNet → Cherimoya",
                "Difference [95% CI]",
                "Cherimoya better, folds",
                "Cherimoya better, datasets",
                "p, datasets",
                "p, folds",
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
            pval(r["wilcoxon_p"], r["wilcoxon_p_method"]),
            num(r["mean_residual_chrombpnet_median"], 2, True),
            num(r["mean_residual_cherimoya_median"], 2, True),
        ]
        for _, r in r_all.iterrows()
    ]
    below = [int(x) for x in r_all[r_all["mean_residual_cherimoya_median"]
                                   < r_all["mean_residual_chrombpnet_median"]]["decile"]]  # fmt: skip
    t3 = table_html(
        n,
        "Residuals by decile of observed log(1 + counts), each dataset's test peaks of all folds pooled. "
        "Medians over datasets. The RMSE difference is the median over datasets of each dataset's difference, "
        "Cherimoya minus ChromBPNet (below 0: Cherimoya closer), so it is not the difference of the two RMSE "
        "columns; with its bootstrap CI and the Wilcoxon p over datasets. The mean residual is predicted minus "
        "observed log counts"
        + (f": Cherimoya's sits below ChromBPNet's in deciles {ranges(below)}." if below else "."),
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
            "shared" if r["shared"] else "other",
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
        "One panel each for counts Pearson r, counts MSE (log counts) and profile median JSD; x: ChromBPNet, "
        "y: Cherimoya. Each dot is one test fold (dataset × fold); filled dots are the "
        f"{len(rep.shared)} shared datasets, open dots the others. Above the diagonal Cherimoya's value is "
        "higher; for MSE and JSD lower is better, so Cherimoya is better below it.",
    )
    f2 = figure_html(
        rep,
        n,
        "level1_residuals",
        "Left: mean residual (predicted minus observed log counts) of each "
        "model by decile of observed counts, median over datasets with the interquartile range. Right: the "
        "RMSE difference, Cherimoya minus ChromBPNet (below 0: Cherimoya closer), per dataset (grey) and its "
        "median over datasets with the bootstrap 95% CI.",
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


def float16_split(f16) -> dict:
    """The float16 check's medians: those that read Cherimoya's per-fold attribution files (fold_<k>, loo_cherimoya_<k>,
    heldout_cross: these can move) and the others (the fold averages and ChromBPNet's leave-one-out: these cannot)."""
    read = f16["comparison"].astype(str).str.match(r"^(fold_|loo_cherimoya_|heldout_cross)")
    diff = f16["median_diff"].abs()
    return {
        "n_read": int(read.sum()),
        "max_read": float(diff[read].max()) if read.any() else NAN,
        "n_other": int((~read).sum()),
        "max_other": float(diff[~read].max()) if (~read).any() else NAN,
    }


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
    fnd = l2_finding(rep, l2)
    wins = " and ".join(window_name(x) for x in l2["windows"])
    lead = (f"<p class=\"lead\">{fnd['lead'] + ' ' if fnd else ''}The counts-head DeepLIFT/SHAP scores of both models "
            f"on every peak both attributed ({count(joins['compared_peaks'].min())}–"
            f"{count(joins['compared_peaks'].max())} per dataset) are joined by peak name, and the projected score "
            f"(the hypothetical score of the base present) is compared per peak in the {wins}. <b>A</b> compares "
            "the two models fold by fold, <b>B</b> their fold averages, <b>C</b> each model's fold against the mean "
            "of its other four folds (the ceiling), and <b>D</b> looks at the fold that held the peak's chromosome "
            "out: the two models there (cross-model), and each model's fold against the mean of its other four there "
            "(per model).</p>")  # fmt: skip
    blocks = []
    for wi, w in enumerate(l2["windows"]):
        rows = []
        for name in ("cross_fold", "cross_average", "loo_chrombpnet", "loo_cherimoya", "cross_heldout",
                     "heldout_chrombpnet", "heldout_cherimoya"):  # fmt: skip
            rows.append([h(QUANTITY_NAMES[name])] + [l2_cell(rep, w, m, name) for m in L2_METRICS])
        t_levels = table_html(n, f"Agreement in the {window_name(w)}: the median over datasets [bootstrap 95% CI] of "
                              f"each dataset's median over peaks (Jaccard: mean over peaks).",
                              ["Quantity"] + [METRIC_NAMES[m] for m in L2_METRICS], rows)  # fmt: skip
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
        t_tests = table_html(n, f"Paired differences in the {window_name(w)}: the median difference over the "
                             f"{len(l2['datasets'])} datasets, its bootstrap 95% CI, how many datasets were above 0, "
                             f"and the Wilcoxon p. The last two rows are over the dataset × fold pairs instead (the CI "
                             f"resamples datasets with their folds; {APPROX}: normal approximation); — : not computed "
                             f"for the mean Jaccard.", ["Contrast"] + [METRIC_NAMES[m] for m in L2_METRICS],
                             trows)  # fmt: skip
        if wi == 0:
            blocks.append(
                figure_html(
                    rep,
                    n,
                    "level2_agreement",
                    f"Agreement in the {window_name(w)}. One panel per metric (Pearson r, Spearman ρ, "
                    "cosine, top-50 Jaccard); x: the comparisons A–D; y: each dataset's median over peaks "
                    "(Jaccard: mean, since its medians take only a few values, k/(100 − k)). Dots are datasets "
                    "(cross-model quantities in ink, each model's ceiling in its colour); bars are medians over "
                    "datasets, the thin line beside each its bootstrap 95% CI.",
                )
                + t_levels
                + t_tests
            )
        else:
            blocks.append(details(f"The same in the {window_name(w)}", t_levels + t_tests))
    w = l2["windows"][0]
    st = l2["strata"]
    srows = []
    role_label = {"train": "training", "valid": "validation", "test": "test"}
    for stratum, levels, label in (("signal_decile", ("1", "5", "10"), "Peak signal (narrowPeak signalValue) decile"),
                                   ("gc_decile", ("1", "5", "10"), "GC decile"),
                                   ("role", ("train", "valid", "test"), "Chromosome in the fold's")):  # fmt: skip
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
            srows.append([h(f"{label} {role_label[lev]} set" if stratum == "role" else f"{label} {lev}")] + cells)
    t_strata = table_html(
        n,
        f"Pearson r by stratum in the {window_name(w)}, median over datasets [95% CI]. Deciles are "
        "taken over each dataset's joined peaks; the GC decile is of the GC content of the "
        f"{window_name(l2['windows'][-1], full=False)}. B has no role stratum (fold averages have no "
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
            f"peak-signal decile than in the bottom one ({ds_counts(sig)}, p {pval(sig['p'], sig['p_method'])}); "
            f"both ceilings move with it, and B stays within {num(gap, 3)} of the higher "
            f"ceiling in every decile. Single folds (A) change most, from "
            f"{num(stv('signal_decile', '1', 'cross_fold'))} to {num(stv('signal_decile', '10', 'cross_fold'))}. "
            f"Agreement {'falls' if gc['estimate'] < 0 else 'rises'} with GC content: B is "
            f"{num(abs(gc['estimate']), 3)} {'lower' if gc['estimate'] < 0 else 'higher'} in the most GC-rich "
            f"decile than in the least (p {pval(gc['p'], gc['p_method'])}), and there the ceilings part: ChromBPNet "
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
        f"Attribution magnitude per model in the {window_name(w)}: medians over datasets [95% CI] of each "
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
        fx = float16_split(f16)
        f16_text = (
            f"<p>Float16 check on {h(', '.join(rep.labels.get(x, x) for x in ds16))}: computed again from "
            f"Cherimoya's float16 attribution files instead of its float32 ones, the {fx['n_read']} per-fold medians "
            f"that read them (cross-model per fold and on the held-out fold, and Cherimoya's leave-one-out ceiling; "
            f"both windows, four metrics) move by at most {num(fx['max_read'], 6)}. The other {fx['n_other']} "
            f"medians of the check (the fold averages, which come from float16 in both runs, and ChromBPNet's "
            f"ceiling) do not read them; the per-model held-out medians and the magnitudes were not computed "
            f"again.</p>"
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
            f"Pearson r of the projected scores in the {window_name(w)}, by decile of peak signal (left) "
            f"and of the GC content of the {window_name(l2['windows'][-1], full=False)} (right): medians over "
            "datasets with bootstrap 95% CIs. The dashed line is A (single folds), the solid ink line B (fold "
            "averages), the blue and vermillion lines the ceilings (C) of ChromBPNet and Cherimoya.",
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
    fnd = l3_finding(rep)
    lead = (
        f'<p class="lead">{fnd["lead"] + " " if fnd else ""}Both models ran the same TF-MoDISco (the same settings, '
        "500-bp window) on their fold-average attributions, and the same QC-scale run (window 400, 5,000 seqlets) "
        "on each fold's 30K-peak subsample. A fold-average pattern is matched when the joint compendium puts a "
        "pattern of the other model's same dataset in its cluster (the primary rule), or when tomtom-lite finds it a "
        "partner among the other model's patterns at q ≤ 0.05 with the MotifCompendium database as decoys "
        "(secondary). Seqlets are placed in the genome through each model's own peak file and count as shared at "
        "≥ 50% reciprocal overlap.</p>"
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
                      f"{int(unm['cherimoya']['cluster_unmatched'].sum())} Cherimoya fold-average patterns are "
                      "unmatched (the bias control below checks these "
                      f"{int(unm['cherimoya']['cluster_unmatched'].sum())} Cherimoya patterns), but "
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
                      test_cell(s(rep, "3", famq, f"QC: {m} patterns matched across models - within {m} (ceiling)"),
                                4),
                      test_cell(s(rep, "3", famq, f"QC: {m} patterns matched across models - within {m}, per fold",
                                  "folds"), 4),
                      est_ci(s(rep, "3", fam_m, f"{NAMES[m]}: qc_seqlet_overlap")),
                      est_ci(s(rep, "3", fam_m, f"{NAMES[m]}: qc_patterns_per_fold"), 1)])  # fmt: skip
    t2 = table_html(n, "QC scale (each fold's 30K-peak subsample): the share of a fold's patterns matched by tomtom in "
                    "the other model's same fold, against the model's own fold-to-fold matching (every pair of its "
                    "folds; the ceiling). The comparison favours the cross-model match: both models' runs of a fold "
                    "use the same subsample, while the ceiling compares folds whose subsamples differ. Medians over "
                    f"datasets [95% CI]; the differences are over the {len(l3['datasets'])} datasets and over the "
                    "dataset × fold pairs (that CI resamples datasets with their folds; "
                    f"{APPROX}: normal approximation). Seqlet overlap has no ceiling, because the subsamples differ "
                    "between folds.",
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
  read against the QC-scale ceiling, which is itself tilted towards the cross-model match. The bias control (level 3c)
  follows the hits, in its own section below.</p>
</section>"""


def section_level4(rep: Report, n: Numbering, l4: dict | None, l5: dict | None = None) -> str:
    head = '<div class="head"><p class="eyebrow">Level 4</p><h2>Motif compendia</h2></div>'
    if l4 is None:
        return f'<section id="level-4">{head}{problem_html(rep, "Level 4")}{pending_html(rep, "Level 4")}</section>'
    if l4.get("state") != "done":
        return f'<section id="level-4">{head}{pending_html(rep, "Level 4")}</section>'
    tags, lfl = l4["tags"], l4["lfl"]
    ag, bs = l4["agreement"], l4["by_sign"]
    fnd = l4_finding(rep, l4, l5)
    lead = (f"<p class=\"lead\">{fnd['lead'] + ' ' if fnd else ''}MotifCompendium clustered each model's fold-average "
            f"patterns of the {len(rep.shared)} shared datasets into a compendium per model ({h(tags['cbp'])}, "
            f"{h(tags['cheri'])}) and both models' patterns together into a joint one ({h(tags['joint'])}, threshold "
            "0.95). Like-for-like clusters are paired by reciprocal best tomtom-lite match of their CWMs; every "
            "cluster has some best match at q ≤ 0.05 (there are no decoys), so only the reciprocal pairs say "
            "something.</p>")  # fmt: skip
    cl = lfl.get("clusters", {})
    ann, fam = lfl.get("annotations", {}), lfl.get("families", {})

    def size(v) -> str:
        return count(len(v)) if isinstance(v, list | tuple | set) else count(v)

    n_recip = int(lfl.get("matched", {}).get("reciprocal_pairs", sum(ag.values())))
    rows = [["Clusters (ChromBPNet / Cherimoya)", f"{count(cl.get('cbp'))} / {count(cl.get('cheri'))}"],
            ["Reciprocal best pairs", f"{n_recip} ({bs.get('pos', 0)} positive, {bs.get('neg', 0)} negative)"],
            ["… with the same annotation", count(ag.get("same annotation", 0))],
            ["… with different annotations", count(ag.get("different annotation", 0))],
            ["… with an unannotated side", count(ag.get("unannotated", 0))],
            ["Annotations (shared / ChromBPNet only / Cherimoya only)",
             f"{size(ann.get('shared'))} / {size(ann.get('only_a'))} / {size(ann.get('only_b'))} (Jaccard "
             f"{num(ann.get('jaccard'))})"],
            ["TF families (shared / ChromBPNet only / Cherimoya only)",
             f"{size(fam.get('shared'))} / {size(fam.get('only_a'))} / {size(fam.get('only_b'))} (Jaccard "
             f"{num(fam.get('jaccard'))})"]]  # fmt: skip
    only = ""
    if isinstance(ann.get("only_a"), list) and isinstance(ann.get("only_b"), list):
        only = details("Annotations of one model's compendium only",
                       f"<p><b>ChromBPNet only:</b> {h(', '.join(ann['only_a']))}.</p>"
                       f"<p><b>Cherimoya only:</b> {h(', '.join(ann['only_b']))}.</p>")  # fmt: skip
    t1 = table_html(n, f"Like for like: {h(tags['cbp'])} (ChromBPNet) against {h(tags['cheri'])} (Cherimoya). "
                    "Annotations are each cluster's most frequent MotifCompendium annotation of score ≥ 0.7; a family "
                    "is the annotation without its _&lt;n&gt; suffix.",
                    ["", "Value"], [[h(r[0])] + r[1:] for r in rows])  # fmt: skip
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
    fnd = l5_finding(rep)
    lead = (f"<p class=\"lead\">{fnd['lead'] + ' ' if fnd else ''}Both models' fold-average attributions were scanned "
            "by Fi-NeMo with the same vocabulary, the joint compendium (the same compiled motif file for both). Only "
            "peaks both models' peak files hold are compared. A hit is matched when the other model has a hit of the "
            "same motif on the same strand with its centre within 5 bp (0 and 10 bp are also reported). Precision is "
            "the share of Cherimoya's hits matched; recall the share of ChromBPNet's hits matched. There is no "
            "within-model ceiling here: Fi-NeMo ran on fold averages only.</p>")  # fmt: skip
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
    t3 = table_html(n, "By the class of the motif's joint cluster and by its sign (from the joint compendium's cluster "
                    "table, not the motif's name prefix): the share of each model's hits, precision and recall at "
                    "5 bp pooled over the class's hits, and the per-motif medians of the peak-set Jaccard and of the "
                    "coefficient Spearman correlation. Medians over datasets [95% CI].",
                    ["Motifs of", "Share of ChromBPNet's hits", "Share of Cherimoya's hits", "Precision", "Recall",
                     "Peak Jaccard, per motif", "Coefficient ρ, per motif"], crow)  # fmt: skip
    cc = [s(rep, "5", "Fi-NeMo hits, by cluster class", q) for q in (
        "share of hits on Cherimoya-only clusters: Cherimoya - ChromBPNet",
        "precision_5bp: shared clusters - Cherimoya-only clusters")]  # fmt: skip
    class_text = ""
    if all(cc):
        class_text = (
            f"<p>Cherimoya puts a {num(cc[0]['estimate'], 3)} larger share of its hits on Cherimoya-only motifs "
            f"than ChromBPNet does ({int(cc[0]['higher'])}/{int(cc[0]['n'])} datasets, p "
            f"{pval(cc[0]['p'], cc[0]['p_method'])}), "
            f"and those hits are the ones ChromBPNet reproduces least: precision is "
            f"{num(abs(cc[1]['estimate']), 3)} {'lower' if cc[1]['estimate'] > 0 else 'higher'} on "
            f"Cherimoya-only motifs than on motifs of both models ({int(cc[1]['higher'])}/{int(cc[1]['n'])} "
            f"datasets, p {pval(cc[1]['p'], cc[1]['p_method'])}).</p>"
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
        t5 = table_html(n, "Fi-NeMo's motif reports side by side: the cosine similarity of the hits' mean contribution "
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
            f"Production hits: ChromBPNet's on its {len(rep.shared)}-dataset compendium against Cherimoya's on its "
            f"{len(rep.everything)}-dataset compendium, through level 4's reciprocal cluster pairs.",
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
            "each motif's hits per 10,000 shared peaks, mean over datasets (x: ChromBPNet, y: Cherimoya, log axes), "
            "coloured by whose patterns its cluster holds; the three motifs with the most hits are numbered, and "
            "motifs without hits for a model sit at 0.01.",
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


def l6_finding(rep: Report, l6: dict) -> dict | None:
    """Level 6: 'text' (the summary's sentences), 'lead' and 'tile'."""
    c, b = l6_metric(l6, "h2h", "all", "cherimoya"), l6_metric(l6, "h2h", "all", "cbp_nobias")
    d = l6_pair(l6, "h2h", "all", "cherimoya", "cbp_nobias")
    if not (c and b and d):
        return None
    full, bias = l6_metric(l6, "h2h", "all", "cbp_full"), l6_metric(l6, "h2h", "all", "cbp_bias")
    held = l6_pair(l6, "heldout", "all", "cherimoya", "cbp_nobias")
    k = len(h2h_datasets(l6))
    diffs = [l6_pair(l6, "h2h", g, "cherimoya", "cbp_nobias") for g in l6["groups"]]
    diffs = [x for x in diffs if x]
    ahead = sum(x["a_minus_b"] > 0 for x in diffs)
    clear = sum(x["lo"] > 0 for x in diffs)
    w, nd = l6["wins"]
    head = (f"On the {count(c['n'])} caQTL leads inside their peaks, scored on the same {k} datasets, the sign of "
            f"Cherimoya's predicted effect matches the measured one for {pct(c['concordance'])} of leads and "
            f"ChromBPNet corrected's for {pct(b['concordance'])}: {points(d['a_minus_b'])} percentage points "
            f"{share_ci(d['lo'], d['hi'])}, McNemar p {pval(d['p_mcnemar'])}.")  # fmt: skip
    more = (f" Cherimoya is ahead in {w}/{nd} datasets taken alone and in {ahead}/{len(diffs)} caQTL groups "
            f"({clear} with a CI above 0)")  # fmt: skip
    if held:
        more += (f", and on each dataset's held-out fold too ({points(held['a_minus_b'])} points, p "
                 f"{pval(held['p_mcnemar'])})")  # fmt: skip
    more += "."
    t1, t2 = c.get("top_0.1"), b.get("top_0.1")
    if isnum(t1) and isnum(t2):
        more += (f" Both are right on {pct(t1)} of their most confident 10% of leads, so the difference lies in the "
                 "weaker calls." if pct(t1) == pct(t2) else f" On their most confident 10% of leads Cherimoya is right "
                 f"on {pct(t1)} and ChromBPNet corrected on {pct(t2)}.")  # fmt: skip
    if full and bias:
        more += (f" ChromBPNet uncorrected scores like corrected ({pct(full['concordance'])}); its bias model alone "
                 f"gets {pct(bias['concordance'])} right.")  # fmt: skip
    tile = ("caQTL leads in peaks with the measured direction", f"{pct(b['concordance'])} → {pct(c['concordance'])}",
            f"ChromBPNet corrected → Cherimoya, {count(c['n'])} leads on the same {k} datasets; "
            f"{points(d['a_minus_b'])} points, McNemar p {plain_p(d['p_mcnemar'])}")  # fmt: skip
    return {"text": head + more, "lead": head, "tile": tile}


def section_level6(rep: Report, n: Numbering, l6: dict | None) -> str:
    head = '<div class="head"><p class="eyebrow">Level 6</p><h2>Variant effects against measured caQTLs</h2></div>'
    if l6 is None:
        return f'<section id="level-6">{head}{problem_html(rep, "Level 6")}</section>'
    if l6.get("state") != "done":
        return f'<section id="level-6">{head}{pending_html(rep, "Level 6")}</section>'
    info, labels, sub = l6["info"], l6["labels"], l6["primary"]
    cnt, af = info.get("counts") or {}, info.get("af_check") or {}
    dsets = info.get("datasets") or {}
    maxbp = l6["subsets"][sub]
    fnd = l6_finding(rep, l6)
    src = info.get("source") or "the QTL table given to qtl_compare.py"
    k = len(h2h_datasets(l6))
    af_text = (f"; the table's allele frequency follows the scored list's frequency of the genome's ALT allele (r = "
               f"{num(af.get('r_af_vs_list_alt_af'), 3)} on {count(af.get('n'))} variants), so the effect allele is "
               "the genome's ALT" if af else "")  # fmt: skip
    lead = (f'<p class="lead">{fnd["lead"] + " " if fnd else ""}The measured effects are {h(src)}. Each lead is put '
            f"on the genome's alleles ({count(cnt.get('swapped'))} of the {count(cnt.get('rows'))} rows name the "
            f"genome's ALT allele first in their variant id, {count(cnt.get('mismatch'))} match neither genome "
            f"base){af_text}. A model's prediction is its logFC, log<sub>2</sub> of the counts it predicts with the "
            "ALT allele over the REF allele, mean of its five folds, averaged over the datasets that stand for the "
            f"lead's group. Compared here are the leads within {maxbp:g} bp of their peak's centre, inside the peak, "
            f"on the {k} datasets that have both models' variant scores and stand for a caQTL group. 50% is "
            "chance.</p>")  # fmt: skip

    def vkey(m: str) -> str:
        cls = {"cherimoya": "cheri", "cbp_nobias": "cbp", "cbp_full": "cbp"}.get(m)
        sw = f'<i class="sw {cls}"></i>' if cls else ""
        return f'<span class="key">{sw}{h(labels.get(m, m))}</span>'

    def share(r: dict | None) -> str:
        if r is None:
            return "NA"
        return f'{pct(r["concordance"])}<span class="sub">{share_ci(r["concordance_lo"], r["concordance_hi"])}</span>'

    def diff(r: dict | None) -> str:
        if r is None:
            return "NA"
        return (f'{points(r["a_minus_b"])}<span class="sub">{share_ci(r["lo"], r["hi"])} · '
                f'{count(r["a_right_b_wrong"])} vs {count(r["a_wrong_b_right"])} · p {pval(r["p_mcnemar"])}</span>')  # fmt: skip

    models = [m for m in VARIANT_MODELS if l6_metric(l6, "h2h", "all", m)]
    rows1 = []
    for m in models:
        r = l6_metric(l6, "h2h", "all", m)
        rho = f'{num(r["spearman"])}<span class="sub">{ci(r["spearman_lo"], r["spearman_hi"])}</span>'
        rows1.append([vkey(m), count(r["n"]), share(r), pct(r.get("top_0.1")), pct(r.get("top_0.25")),
                      pct(r.get("top_0.5")), rho, num(r["pearson"])])  # fmt: skip
    cd = l6_pair(l6, "h2h", "all", "cherimoya", "cbp_nobias")
    cs = l6_pair(l6, "h2h", "all", "cherimoya", "cbp_nobias", "spearman")
    fu = l6_pair(l6, "h2h", "all", "cbp_nobias", "cbp_full")
    pc, pb = l6_metric(l6, "h2h", "all", "cherimoya"), l6_metric(l6, "h2h", "all", "cbp_nobias")
    tail = ""
    if cd:
        tail += (f" Cherimoya − ChromBPNet corrected on the same leads: {points(cd['a_minus_b'])} points "
                 f"{share_ci(cd['lo'], cd['hi'])}; {count(cd['a_right_b_wrong'])} leads only Cherimoya calls the right "
                 f"way against {count(cd['a_wrong_b_right'])} only ChromBPNet corrected does (McNemar p "
                 f"{pval(cd['p_mcnemar'])}).")  # fmt: skip
    if cs:
        tail += (f" Spearman ρ: {num(cs['a_minus_b'], 4, True)} {ci(cs['lo'], cs['hi'], 4, True)} (bootstrap p "
                 f"{pval(cs['p_boot'])}).")  # fmt: skip
    if pc and pb and isnum(pc["pearson"]) and isnum(pb["pearson"]) and pc["pearson"] < pb["pearson"]:
        tail += (f" Pearson r, which large effects weigh on most, is higher for ChromBPNet corrected "
                 f"({num(pb['pearson'])} against {num(pc['pearson'])}; no interval was computed for it).")  # fmt: skip
    if fu:
        tail += f" ChromBPNet corrected − uncorrected: {points(fu['a_minus_b'], 2)} points."
    t1 = table_html(n, f"Every model on the same {count(pc['n']) if pc else 'NA'} leads (fold means, {k} "
                    "datasets): the share whose predicted direction matches the measured one [95% CI], the same "
                    "share among the model's most confident 10%, 25% and 50% of leads (by |logFC|), and the "
                    "correlations of logFC with the measured slope." + tail,
                    ["Model", "Leads", "Direction right", "Top 10%", "Top 25%", "Top 50%", "Spearman ρ", "Pearson r"],
                    rows1)  # fmt: skip

    rows2 = []
    for g in l6["groups"]:
        dg = dsets.get(g) or {}
        mine = (dg.get("matched") or {}).get("cherimoya") or []
        r_c, r_b = l6_metric(l6, "h2h", g, "cherimoya"), l6_metric(l6, "h2h", g, "cbp_nobias")
        r_x = l6_metric(l6, "h2h", g, "cbp_bias")
        rows2.append([h(group_name(g)), f"{len(dg.get('h2h') or [])} of {len(mine)}", count(r_c["n"]) if r_c else "NA",
                      share(r_c), share(r_b), pct(r_x["concordance"]) if r_x else "NA",
                      diff(l6_pair(l6, "h2h", g, "cherimoya", "cbp_nobias"))])  # fmt: skip
    only = [g for g in dsets if not (dsets[g].get("h2h") or []) and l6_metric(l6, "matched", g, "cherimoya")]
    for g in only:
        r = l6_metric(l6, "matched", g, "cherimoya")
        mine = (dsets[g].get("matched") or {}).get("cherimoya") or []
        rows2.append([h(group_name(g)), f"0 of {len(mine)}", count(r["n"]), share(r), "—", "—",
                      "Cherimoya alone, on its datasets"])  # fmt: skip
    same = {}
    for g, dg in dsets.items():
        same.setdefault(tuple(sorted((dg.get("matched") or {}).get("cherimoya") or [])), []).append(g)
    shared_sets = [gs for gs in same.values() if len(gs) > 1]
    same_text = (" One prediction stands for several groups when their datasets are the same (e.g. "
                 f"{h(' and '.join(group_name(g) for g in shared_sets[0]))}), and a variant that leads in several "
                 "groups counts once in each; the bootstrap draws all its rows together."
                 if shared_sets else "")  # fmt: skip
    t2 = table_html(n, "By caQTL group (fold means, the leads inside their peaks): the datasets with both models' "
                    "variant scores out of the group's datasets, the leads, each model's share with the measured "
                    "direction [95% CI], and Cherimoya − ChromBPNet corrected in percentage points [95% CI] · the "
                    "leads only Cherimoya and only ChromBPNet corrected calls right · McNemar p." + same_text,
                    ["Group", "Datasets", "Leads", "Cherimoya", "ChromBPNet corrected", "Bias model",
                     "Cherimoya − corrected"], rows2, left=2)  # fmt: skip

    rows3 = []
    order = [sub] + [x for x in l6["subsets"] if x != sub]
    for s_, (setting, what) in ((x, y) for x in order for y in (("h2h", "fold mean"), ("heldout", "held-out fold"))):
        r_c, r_b = l6_metric(l6, setting, "all", "cherimoya", s_), l6_metric(l6, setting, "all", "cbp_nobias", s_)
        if not (r_c and r_b):
            continue
        rows3.append([h(f"within {l6['subsets'][s_]:g} bp, {what}"), count(r_c["n"]), share(r_c), share(r_b),
                      diff(l6_pair(l6, setting, "all", "cherimoya", "cbp_nobias", "concordance", s_)),
                      f"{num(r_c['spearman'])} / {num(r_b['spearman'])}"])  # fmt: skip
    mc = l6_metric(l6, "matched", "all", "cherimoya")
    ncheri = sorted({x for g in dsets.values() for x in (g.get("matched") or {}).get("cherimoya") or []})
    matched_text = (f" Cherimoya on all {len(ncheri)} of its datasets for these groups: {pct(mc['concordance'])} "
                    f"{share_ci(mc['concordance_lo'], mc['concordance_hi'])} of {count(mc['n'])} leads inside their "
                    "peaks." if mc else "")  # fmt: skip
    t3 = table_html(n, "The same comparison on each dataset's held-out fold (the fold whose model never trained on "
                    "the variant's chromosome) and on the leads farther from their peak's centre (still inside the "
                    "scored window). The same datasets and leads for both models in every row; Cherimoya − "
                    "ChromBPNet corrected as in the table above; Spearman ρ Cherimoya / ChromBPNet corrected."
                    + matched_text,
                    ["Leads", "n", "Cherimoya", "ChromBPNet corrected", "Cherimoya − corrected", "Spearman ρ"],
                    rows3)  # fmt: skip

    per = l6["per"][l6["per"]["subset"] == sub]
    rows4 = []
    for ds in dict.fromkeys(per["dataset"]):
        x = per[per["dataset"] == ds].set_index("model")
        if "cbp_nobias" not in x.index:
            continue
        rows4.append([h(rep.labels.get(ds, ds)), h(", ".join(group_name(g) for g in str(x["groups"].iloc[0])
                                                              .split(","))), count(x["n"].iloc[0])]
                     + [pct(x.loc[m, "concordance"]) if m in x.index else "NA" for m in models])  # fmt: skip
    t4 = details(f"Each dataset alone ({len(rows4)} datasets)", table_html(
        n, "Each dataset's own models on the leads of the groups it stands for (inside their peaks): the share with "
        "the measured direction.", ["Dataset", "Groups", "Leads"] + [h(labels.get(m, m)) for m in models], rows4,
        left=2, cls="compact")) if rows4 else ""  # fmt: skip
    return f"""
<section id="level-6">
  {head}
  {lead}
  {
        figure_html(
            rep,
            n,
            "level6_caqtl",
            "Left: among each model's most confident leads (the top x% by |logFC|), the share whose predicted "
            "direction matches the measured one; ChromBPNet uncorrected is left out, it lies on corrected. Middle: "
            "Cherimoya − ChromBPNet corrected by caQTL group (leads in brackets), with bootstrap 95% CIs over "
            "variants, from the fold means (filled) and from each dataset's held-out fold (hollow). Right: each "
            "dataset's own models on the leads of the groups it stands for; points above the diagonal favour "
            "Cherimoya.",
        )
    }
  {t1}
  {t2}
  {t3}
  {t4}
</section>"""


def control_sentence(rep: Report, bc: dict) -> str:
    """The controls used, in one sentence for the lead ('' when none was used)."""
    used = [c for c in bc["controls"] if c["state"] == "done"]
    if not used:
        return ""
    ref = bc.get("reference")
    names = {f"bias_control/{c['name']}" for c in used}
    n_pat = int(ref["reference"].isin(names).sum()) if ref is not None and not ref.empty else 0
    parts = [
        f"the {h(c['head'])} head of ChromBPNet's uncorrected model on "
        f"{h(rep.labels.get(c.get('dataset'), c.get('dataset')))} fold {h(c.get('fold'))}"
        for c in used
    ]
    heads = sorted({h(c["head"]) for c in bc["controls"] if c["state"] != "done"})
    return (
        f" The control attributes {' and '.join(parts)} with DeepSHAP, runs TF-MoDISco on it"
        + (f" ({n_pat} patterns)" if n_pat else "")
        + " and asks which of Cherimoya's unmatched patterns it also finds"
        + (f"; DeepSHAP cannot attribute that model's {' or '.join(heads)} head (see Methods)." if heads else ".")
    )


def section_bias(rep: Report, n: Numbering, bc: dict | None, l3: dict | None, l4: dict | None) -> str:
    head = '<div class="head"><p class="eyebrow">Level 3c</p><h2>Bias control</h2></div>'
    if bc is None:
        return f'<section id="bias-control">{head}{problem_html(rep, "Bias control")}</section>'
    if bc.get("state") != "done":
        return f'<section id="bias-control">{head}<p>No bias control has been compared yet.</p></section>'
    fnd = bias_finding(rep, bc, l3) if l3 else None
    lead = (f"<p class=\"lead\">{fnd['lead'] + ' ' if fnd else ''}ChromBPNet's attributions come from its "
            "corrected model; without a bias model, Cherimoya can learn Tn5 or DNase I sequence preference that "
            "ChromBPNet's bias model absorbs. Two checks: the motif-database flag of level 3 (a pattern whose best "
            "match in ChromBPNet's motif database, motifs.meme, is a TN5_ or DNASE_ entry at q ≤ 0.05), and a "
            f"control.{control_sentence(rep, bc)}</p>")  # fmt: skip
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
                b = {k: int(c95.loc[k, "bias_clusters"]) for k in CLASSES}
                jt = (f" In the joint compendium, {b['cherimoya_only']} Cherimoya-only, {b['chrombpnet_only']} "
                      f"ChromBPNet-only and {b['shared']} shared cluster{'s' if b['shared'] != 1 else ''} have at "
                      "least half of their patterns flagged.")  # fmt: skip
        t1 = table_html(n, "Patterns flagged as Tn5/DNase by the motif database, summed over the datasets. Cherimoya − "
                        f"ChromBPNet per dataset: {test_cell(t, 1)}; per QC fold (dataset × fold): {test_cell(tq, 1)} "
                        f"({APPROX}: normal approximation).{jt}",
                        ["Model", "Fold-average patterns flagged", "Per dataset", "Their seqlets",
                         "QC patterns flagged (all folds)"], rows)  # fmt: skip
    ref = bc["reference"]
    split = bc["split"].set_index("reference_kind")
    t2 = t3 = ""
    if ref is not None and not ref.empty:
        rr = ref.sort_values("cherimoya_patterns", ascending=False)
        rows = [[h(r["pattern"]), h(r["consensus"]), count(r["n_seqlets"]), h(r["annotation"]), pval(r["annotation_q"]),
                 h(r["bias_db_best"]), pval(r["bias_db_best_q"]), h(r["kind"]), count(r["cherimoya_patterns"]),
                 count(r["cherimoya_seqlets"])] for _, r in rr.iterrows()]  # fmt: skip
        t2 = table_html(
            n,
            "The reference: ChromBPNet uncorrected's TF-MoDISco patterns, annotated here with "
            "level 3's rules, and the unmatched Cherimoya patterns that match each (tomtom-lite, q ≤ 0.05, "
            "database decoys).",
            [
                "Pattern",
                "Consensus",
                "Seqlets",
                "MotifCompendium match",
                "MotifCompendium q",
                "motifs.meme match",
                "motifs.meme q",
                "Kind",
                "Cherimoya patterns matched",
                "Their seqlets",
            ],
            rows,
            left={0, 1, 3, 5, 7},
            cls="compact",
        )
        scales = bc.get("scales") or {}
        srows = []
        for k in ("Tn5/DNase", "TF or other", "no match", "unknown"):
            if k in split.index:
                r = split.loc[k]
                srows.append([h({"Tn5/DNase": "a Tn5/DNase pattern", "TF or other": "a TF or other pattern",
                                 "no match": "no reference pattern", "unknown": "an unannotated reference"}[k]),
                              count(r["patterns"]), count(r["seqlets"]), count(r["db_flagged"]),
                              count(r["chrombpnet_partner"]) if "chrombpnet_partner" in r else "NA"])  # fmt: skip
        what = (f"{scales.get('fold average', 0)} fold-average patterns without a cluster partner (level 3)"
                + (f" and {scales['QC fold']} of the control's own QC fold without a tomtom partner"
                   if scales.get("QC fold") else ""))  # fmt: skip
        t3 = table_html(
            n,
            f"The {int(split['patterns'].sum())} Cherimoya patterns checked, {what}, by what they match in the "
            "reference, with how many are themselves flagged Tn5/DNase and how many have a tomtom partner in "
            "ChromBPNet corrected (its fold average, or its same QC fold).",
            [
                "Matches",
                "Cherimoya patterns",
                "Their seqlets",
                "Themselves flagged Tn5/DNase",
                "With a partner in ChromBPNet corrected",
            ],
            srows,
        )
    return f"""
<section id="bias-control">
  {head}
  {lead}
  {
        figure_html(
            rep,
            n,
            "bias_control",
            "Left: Tn5/DNase-flagged patterns per dataset (lines join a dataset's two "
            "values), at fold averages and per QC fold. Right: each pattern of ChromBPNet uncorrected "
            "(profile head) and how many unmatched Cherimoya patterns match it; a bar is labelled with the pattern's "
            "best MotifCompendium match or, for the Tn5/DNase patterns (orange), with its motifs.meme Tn5/DNase "
            "entry.",
        )
    }
  {t1}
  {t2}
  {t3}
  {f"<p>{fnd['closing']}</p>" if fnd else ""}
</section>"""


def section_caveats(rep: Report, R: dict) -> str:
    """The plan's caveats (bias model, float16, dropped N windows), the ceilings and the p-value floors; each item's
    text is HTML."""
    l2, l3, l4, l5, l6 = (R.get(k) for k in ("l2", "l3", "l4", "l5", "l6"))
    bias_bits = []
    if l3 and l3.get("state") == "done":
        per = l3["per"]
        t = s(rep, "3", "TF-MoDISco, Cherimoya - ChromBPNet", "Tn5/DNase-flagged patterns: Cherimoya - ChromBPNet")
        bias_bits.append(
            f"Tn5/DNase-flagged fold-average patterns: {int(per[per['model'] == 'cherimoya']['bias_flagged'].sum())} "
            f"for Cherimoya against {int(per[per['model'] == 'chrombpnet']['bias_flagged'].sum())} for ChromBPNet"
            + (f" (p {pval(t['p'], t['p_method'])} over datasets)" if t else "")
        )
    if l4 and l4.get("state") == "done":
        c95 = l4["classes"][l4["classes"]["threshold"] == "0.95"].set_index("class")
        if "bias_clusters" in c95:
            bias_bits.append(
                f"clusters of the joint compendium with at least half of their patterns flagged: "
                f"{int(c95.loc['cherimoya_only', 'bias_clusters'])} Cherimoya-only, "
                f"{int(c95.loc['chrombpnet_only', 'bias_clusters'])} ChromBPNet-only, "
                f"{int(c95.loc['shared', 'bias_clusters'])} shared"
            )
    f16 = ""
    if l2 and l2.get("state") == "done" and l2.get("float16") is not None and len(l2["float16"]):
        fx = float16_split(l2["float16"])
        ds16 = sorted(set(l2["float16"]["dataset"]))
        f16 = (
            f" Level 2 reads Cherimoya's float32 per-fold files; computed again from its float16 files on "
            f"{h(', '.join(rep.labels.get(x, x) for x in ds16))}, the {fx['n_read']} level-2 medians that read them "
            f"move by at most {num(fx['max_read'], 6)}."
        )
    drop = ""
    if l2 and l2.get("state") == "done":
        j = l2["joins"]
        drop = (
            f" {int(j['chrombpnet_only_peaks'].min())}–{int(j['chrombpnet_only_peaks'].max())} peaks per dataset "
            f"({int(j['chrombpnet_only_peaks'].sum())} over the {len(j)} datasets) are in ChromBPNet's peak files only."
        )
    if l5 and l5.get("state") == "done":
        dd = l5["totals"]["chrombpnet_hits_on_dropped_peaks"]
        drop += f" ChromBPNet has {count(dd.min())}–{count(dd.max())} hits per dataset on them."
    l6_bias, l6_items = "", []
    if l6 and l6.get("state") == "done":
        fu = l6_pair(l6, "h2h", "all", "cbp_nobias", "cbp_full")
        bm = l6_metric(l6, "h2h", "all", "cbp_bias")
        if fu and bm:
            l6_bias = (f" For variant effects (level 6) the bias model cancels out of the allele difference: ChromBPNet "
                       f"uncorrected's share differs from corrected's by {num(100 * abs(fu['a_minus_b']), 2)} points. The bias "
                       f"model alone, which sees only Tn5's sequence preference, gets {pct(bm['concordance'])} of the "
                       "directions right, above chance: some measured effects follow how the allele changes Tn5's "
                       "preference, or something that goes with it.")  # fmt: skip
        l6_items.append((
            "caQTL leads",
            "Level 6's ground truth is each caQTL's lead variant as released, not a fine-mapped causal variant: a lead "
            "can tag the causal variant through linkage disequilibrium, and then no model can be right about it except "
            "by chance, so neither model is expected to reach 100%. Only significant caQTLs are released, so level 6 tests "
            "the direction of real effects, not whether a model tells caQTLs from variants without an effect (there are "
            "no negatives and no AUROC). The caQTLs were mapped in the source's own pseudobulks of each group's cells, "
            "while each model stands for a whole sample, so one dataset stands for every group of its condition.",
        ))  # fmt: skip
    n_sh, n_all = len(rep.shared), len(rep.everything)
    floors = [(n_sh, "datasets"), (n_all - n_sh, "datasets"), (n_all, "datasets"), (5 * n_sh, "folds")]
    floor_text = ", ".join(f"{pval(2.0 / 2**k)} for {k} {what}" for k, what in floors if k > 0)
    items = [
        (
            "Bias model",
            "ChromBPNet's attributions (levels 2–5) come from its corrected model; Cherimoya attributes its "
            "log-counts head with no bias model, so it can learn Tn5/DNase motifs that ChromBPNet's bias model "
            "absorbs. ChromBPNet corrected still shows some, so the difference is one of degree."
            + (" " + "; ".join(bias_bits) + "." if bias_bits else "")
            + " Level 1 is not affected: it scores ChromBPNet's uncorrected model (bias model included) against observed counts. "
            "The bias control used one dataset and fold and the profile head, so it indicates, but does not measure, "
            "how much of Cherimoya's extra signal is bias." + l6_bias,
        ),
        (
            "Float16",
            "Both models' attributions go through float16: ChromBPNet stores its attribution files in float16, and "
            "so do Cherimoya's converted files, from which the fold averages are computed and which TF-MoDISco and "
            "Fi-NeMo read." + f16,
        ),
        (
            "Dropped N windows",
            "Cherimoya drops the windows that contain an N, so a few peaks have ChromBPNet attributions only. "
            "Levels 2 and 5, and level 3's seqlet overlaps, compare only the peaks both models hold; the TF-MoDISco "
            "patterns (level 3) and the compendia (level 4) come from each model's full peak set." + drop,
        ),
        (
            "Ceilings",
            "A ceiling (C) compares one fold with the mean of the other four. Cross-model agreement on single "
            "folds (A) compares one fold with one fold and so sits below a like-for-like ceiling; fold averages (B) "
            "compare a mean of five with a mean of five and sit above one, so B close to C does not show that the "
            "models agree as well as each agrees with itself. A like-for-like ceiling (one fold against one fold, or "
            "a mean against a mean of as many folds) was not computed. At level 3 only the QC scale has a ceiling, "
            "and it is tilted the same way: both models' runs of a fold share one 30K-peak subsample, while the "
            "fold-to-fold ceiling compares folds whose subsamples differ. Levels 4 and 5 have no ceiling.",
        ),
        (
            "p-value floors",
            f"With n paired values the exact two-sided Wilcoxon p cannot go below 2/2<sup>n</sup>: {floor_text}. "
            f"Over more than 50 values, or with ties or zeros among 14–50, p comes from the normal approximation "
            f"(marked {APPROX}) and can fall further. The folds of one dataset share its library and most of their "
            "training data, so the dataset-level test is the one to read. At those floors, read p as “all in the "
            "same direction”, not as an effect size.",
        ),
    ] + l6_items
    lis = "".join(f"<li><b>{h(t)}.</b> {x}</li>" for t, x in items)
    return f"""
<section id="caveats">
  <div class="head"><p class="eyebrow">Caveats</p><h2>Read the numbers with these in mind</h2></div>
  <ul class="plain">{lis}</ul>
</section>"""


def section_methods(rep: Report, R: dict, prov: dict, generated: str) -> str:
    a = rep.args
    defs = [
        ("Statistics", f"Every interval of levels 1–5 is a percentile bootstrap 95% CI of the median, {a.n_boot:,} "
         f"resamples, seed {a.seed}, and every one resamples datasets (level 6's are described under it): for values over datasets, the datasets; for "
         "values over folds, whole datasets with all their folds. (The levels' own statistics files resample single "
         "folds for their fold rows; this page does not show those intervals, and the checks below compare only "
         "their medians and p.) The endpoints are Monte Carlo estimates: another seed moves them by up to about a "
         "fifth of the interval's width, so a fourth decimal is not stable. Tests are two-sided Wilcoxon signed-rank "
         "tests on paired values (scipy's 'auto' method): p is exact, from the signed-rank distribution or, for up to "
         "13 values with ties or zeros, from every sign flip, except where marked † (the normal approximation with "
         "tie and continuity corrections, used for more than 50 values or for ties or zeros among 14–50). The test "
         "drops differences of 0; where there are some, the counts show how many were above, below and at 0. "
         "statistics.tsv lists every interval and test with n, the statistic, p, how p was computed (p_method) and "
         "how many values were above and below 0."),
        ("Level 1", "compare_chrombpnet.py per fold: counts Pearson and Spearman and MSE of log counts, and the median "
         "JSD and normalized JSD of the profiles, on ChromBPNet's test peaks; ChromBPNet from its evaluation "
         "predictions (chrombpnet_predictions.h5, uncorrected model). Residuals: predicted minus observed log(1 + counts)."),
        ("Level 2", "compare_attributions.py: per peak, Pearson and Spearman of the projected scores, cosine of the "
         "4 × W hypothetical maps, Jaccard of the top 50 positions by |projected score|, and each map's sum and L2 "
         "norm, in the 500-bp and 1,000-bp windows (summit ± 250 and ± 500 bp). ChromBPNet from its 05.0 h5 and "
         "06.0 fold average, Cherimoya from its float32 per-fold npz and 06.0 fold average. Each dataset is "
         "summarized by its median over peaks; A and C are means over the 5 folds of those medians. Strata: deciles "
         "of signalValue and of the GC content of the 1,000-bp window, taken over each dataset's joined peaks, and "
         "the chromosome's role in each fold."),
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
    l6 = R.get("l6")
    if l6 and l6.get("state") == "done":
        st = l6["info"].get("settings") or {}
        sub = ", ".join(f"{k} ≤ {v:g} bp" for k, v in l6["subsets"].items())
        defs.insert(len(defs) - 1, ("Level 6", "qtl_compare.py (rvi-variant-scoring) on scores.py's variant summaries: "
                    "each QTL row's variant on the genome's alleles (REF is the genome base, the effect the slope "
                    "for the genome's ALT), a model's logFC (log2 of predicted counts, ALT over REF, mean of the "
                    "five folds, or the held-out fold's) averaged over the datasets that stand for the row's group, "
                    "and on the rows with every model's prediction: the share whose logFC has the slope's sign "
                    "(rows where either is 0 left out), that share among the model's top fractions by |logFC|, and "
                    f"Spearman and Pearson correlations. Subsets by distance to the peak's centre: {sub}. Intervals: "
                    f"percentile bootstrap 95% CIs over variants, {st.get('n_boot', 'NA')} resamples, seed "
                    f"{st.get('seed', 'NA')}, the rows of one variant drawn together; paired differences from the same "
                    "draws; McNemar's exact test (two-sided binomial on the leads only one of the two models calls "
                    f"right). Output read: {l6['dir']}."))  # fmt: skip
    items = [(h(x), h(y)) for x, y in defs]
    bc = R.get("bias")
    if bc and bc.get("controls"):
        used = [c for c in bc["controls"] if c["state"] == "done"]
        refused = [c for c in bc["controls"] if c["state"] != "done"]
        text = ("bias_control.sh runs 05.0's contribs_bw (DeepSHAP) on ChromBPNet's uncorrected model "
                "(chrombpnet.h5) over the fold's 30K-peak QC subsample, then TF-MoDISco as 04.5 does; "
                "compare_modisco.py --bias compares the result: "
                + "; ".join(f"{h(c['name'])}, the {h(c['head'])} head on {h(c.get('dataset'))} fold "
                            f"{h(c.get('fold'))} ({h(c.get('reason') or c.get('note') or 'done')})" for c in used)
                + ". Its patterns are annotated with level 3's rules, and Cherimoya's unmatched patterns (fold "
                "averages of every dataset, and the QC patterns of the control's own dataset and fold) are matched to "
                "them by tomtom-lite at q ≤ 0.05 with the database as decoys.")  # fmt: skip
        for c in refused:
            text += (f" Refused: {h(c['name'])}, the {h(c['head'])} head on {h(c.get('dataset'))} fold "
                     f"{h(c.get('fold'))}: <code>{h(c.get('reason') or c.get('state'))}</code>")  # fmt: skip
        items.insert(len(items) - 1, ("Bias control", text))  # before "Tables"
    dl = "".join(f"<dt>{x}</dt><dd>{y}</dd>" for x, y in items)
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
    if l6 and l6.get("state") == "done":
        info = l6["info"]
        crow.append([h("qtl_compare.py (level 6, rvi-variant-scoring)"), f"<code>{h(info.get('code') or '?')}</code>",
                     "", h(f"run {l6['dir'].name}"), h(info.get("finished_et") or "")])  # fmt: skip
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
    commits = ('<div class="tbl compact"><table><thead><tr><th scope="col" class="l">Code</th>'
               '<th scope="col" class="l">Commit</th><th scope="col" class="l">Branch</th>'
               '<th scope="col" class="l">Subject or note</th><th scope="col" class="l">Time</th></tr></thead><tbody>'
               + "".join("<tr>" + "".join(f'<td class="l">{c}</td>' for c in r) + "</tr>" for r in crow)
               + "</tbody></table></div>")  # fmt: skip
    mrows = []
    for m in prov["markers"]:
        if not m.get("units"):
            mrows.append([f"{m['level']}", h(m["dir"]), "0", "", "", ""])
            continue
        mrows.append([f"{m['level']}", h(m["dir"]), str(m["units"]), h(et(m["first"])), h(et(m["last"])),
                      h(", ".join(m["commits"]))])  # fmt: skip
    markers = ('<div class="tbl compact"><table><thead><tr><th scope="col" class="l">Level</th>'
               '<th scope="col" class="l">Directory</th><th scope="col">Done units</th><th scope="col">First written'
               '</th><th scope="col">Last written</th><th scope="col">Code commits</th></tr></thead><tbody>'
               + "".join("<tr>" + "".join(f'<td{" class=\"l\"" if i < 2 else ""}>{c}</td>' for i, c in enumerate(r))
                         + "</tr>" for r in mrows) + "</tbody></table></div>")  # fmt: skip
    inputs = R.get("inputs_table")
    irows = []
    if inputs is not None:
        for _, r in inputs.iterrows():
            irows.append([f"<code>{h(r['path'])}</code>", h(r["files"]), h(r["size"]), h(r["modified"])])
    itab = ('<div class="tbl compact"><table><thead><tr><th scope="col" class="l">Input</th><th scope="col">Files</th>'
            '<th scope="col">Size</th><th scope="col">Modified (newest)</th></tr></thead><tbody>'
            + "".join("<tr>" + "".join(f'<td{" class=\"l\"" if i == 0 else ""}>{c}</td>' for i, c in enumerate(r))
                      + "</tr>" for r in irows) + "</tbody></table></div>")  # fmt: skip
    chk = "".join(
        f"<li>{h(c['what'])}: {c['values']} values, largest difference "
        f"{h(f'{c["max_abs_diff"]:.2g}' if isnum(c['max_abs_diff']) else 'NA')}"
        f"{f', {c["nan_mismatch"]} NA mismatches' if c['nan_mismatch'] else ''}.</li>"
        for c in rep.checks
    )
    probs = "".join(f"<li>{h(p)}: {h(m)}</li>" for p, m in rep.problems)
    probs_html = f'<h3>Not computed</h3>\n  <ul class="plain">{probs}</ul>' if probs else ""
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
  <p class="tcap">A consistency check, not an independent one: the statistics the levels already computed were
  computed again here from their tables, with the same bootstrap and Wilcoxon code, and compared with the levels' own
  statistics.tsv (written to 6 significant digits). The largest absolute differences:</p>
  <ul class="plain">{chk or "<li>None run.</li>"}</ul>
  {probs_html}
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
        section_level4(rep, n, R.get("l4"), R.get("l5")), section_level5(rep, n, R.get("l5")),
        section_level6(rep, n, R.get("l6")), section_bias(rep, n, R.get("bias"), R.get("l3"), R.get("l4")),
        section_caveats(rep, R),
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
    R["l6"] = rep.part("Level 6", level6)
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
    pend = list(dict.fromkeys(p.get("short") or p["what"] for p in rep.pending))
    note = (f"index.html {size / 1e6:.1f} MB, {len(rep.figures)} figures, {len(rep.tables)} tables, "
            f"{len(rep.stats.rows)} statistics")  # fmt: skip
    if R.get("l1") and R["l1"].get("state") == "done":
        note += f"; level 1 {R['l1']['n_folds']}/{R['l1']['expected']} folds"
    if R.get("l6") and R["l6"].get("state") == "done":
        note += f"; level 6 from {R['l6']['dir']}"
    if pend:
        note += f"; pending: {'; '.join(pend)}"
    if rep.problems:
        note += f"; not computed: {', '.join(p for p, _ in rep.problems)}"
    settings = {"datasets": rep.shared, "everything": rep.everything, "n_boot": args.n_boot, "seed": args.seed,
                "pending_eta": args.pending_eta, "dpi": args.dpi, "caqtl": args.caqtl}  # fmt: skip
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
    ap.add_argument("--caqtl", type=Path, help="qtl_compare.py's output directory for level 6 (default the newest "
                    "<comparison-root>/../variant_scoring/caqtl/out/<run>/)")  # fmt: skip
    ap.add_argument("--pending-eta", help="a sentence on when the pending parts are expected, shown as given with them "
                    "(e.g. 'No date is set yet.')")  # fmt: skip
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
    args.caqtl = args.caqtl or newest_caqtl(parent / "variant_scoring" / "caqtl")
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
