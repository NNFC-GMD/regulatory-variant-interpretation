#!/usr/bin/env python3
"""Compare the models' variant scores, for the pilot report: agreement, ceilings, strands and the null.

Reads scores.py's per-fold tables for each dataset and model, joined by variant_id (every model and fold must hold the
same variants in the same order), and the variant list for each variant's stratum, type and callset. Every number is
on logFC (log2), the count score; abs_logfc.pval is the p-value read.

Per dataset, in every subset of the variants (all; each stratum; SNV, MNV and indel; each callset as the list gives
it, i.e. exome, genome or both):
  agreement between two models (--pairs; default Cherimoya against each ChromBPNet model)
    fold_mean   Pearson and Spearman of the two models' fold-mean logFC (B in the attribution comparison)
    same_fold   the same per fold, model A fold k against model B fold k, then the mean over folds (A)
    heldout     each model on the fold whose test chromosomes hold the variant (D)
    top_overlap the share of model A's top --top (1 %) variants by |fold-mean logFC| that are also model B's;
                independent rankings would share about --top of them
  each model's ceiling, leave one fold out
    loo         fold k against the mean of the model's other folds, Pearson and Spearman, then the mean over k (C);
                the reproducibility a cross-model number is read against, as in the attribution comparison. Not on
                equal terms with B, which compares two five-fold means where C compares one fold with four, so B
                can exceed C from averaging alone
    top_overlap the same top-share, fold k against the mean of the others, mean over k
  strand agreement, per model and fold, then the mean (and the worst fold)
    fwd_rev     Pearson and Spearman of the logFC from the forward strand alone against the reverse strand alone
    fwd_both    Pearson of forward-only logFC against the two-strand logFC, and their top-share: what scoring one
                strand only would change
  the null, per model
    null_abs_logfc_q99           the 99th percentile of the null rows' |logFC|, mean over folds
    frac_above_null_q99          the share of variants whose per-fold |logFC| exceeds that fold's null q99
    frac_heldout_p_lt_0.01       the share of variants with held-out abs_logfc.pval < 0.01

Outputs, in --out-dir:
  agreement.tsv   dataset, subset_kind, subset, n, model_a, model_b, measure, value (model_a = model_b for ceilings)
  strand.tsv      dataset, subset_kind, subset, n, model, measure, mean over folds, worst fold
  null.tsv        dataset, model, subset_kind, subset, n and the null measures
  report.md       the tables above for reading, with the figures
  fig_scatter.png      fold-mean logFC, Cherimoya against each ChromBPNet model, per dataset
  fig_agreement.png    cross-model Pearson by subset next to both models' leave-one-out ceilings
  fig_strand.png       forward-strand against reverse-strand logFC, per model (first fold)
  fig_null.png         how often |logFC| exceeds a value, the variants against the null, per model

    python workflows/variant_scoring/compare.py --scores-root <root>/scores --datasets <ds> [<ds> ...] \\
        --list <root>/lists/<variants>.tsv --variants <variants> --out-dir <root>/compare [--null null] \\
        [--folds 0 1 2 3 4] [--pairs cherimoya:cbp_full ...] [--top 0.01]

Runs in the pixi `cherimoya` environment (numpy, scipy, matplotlib); the statistics alone need numpy and scipy.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))

import scores as sc  # noqa: E402

logger = logging.getLogger("compare")

MODELS = ("cherimoya", "cbp_full", "cbp_nobias", "cbp_bias")
LABELS = {"cherimoya": "Cherimoya", "cbp_full": "ChromBPNet full", "cbp_nobias": "ChromBPNet bias-free",
          "cbp_bias": "ChromBPNet bias"}  # fmt: skip
PAIRS = ("cherimoya:cbp_full", "cherimoya:cbp_nobias", "cherimoya:cbp_bias")
STRATA = ("summit100", "flank500", "outside")
TYPES = ("SNV", "MNV", "indel")
SOURCES = ("exome", "genome", "both")
# Okabe & Ito (2008): one colour per model, the same in every figure. Checked with the dataviz validator (light
# mode, all pairs): bias-free and bias are 7.6 apart under deuteranopia, so every panel names its models.
COLOURS = {"cherimoya": "#D55E00", "cbp_full": "#0072B2", "cbp_nobias": "#009E73", "cbp_bias": "#CC79A7"}
CONTEXT = "#8C8C8C"  # the null, ceilings and reference lines
INK, MUTED, GRID = "#222222", "#666666", "#E6E6E6"


def label(model: str) -> str:
    return LABELS.get(model, model)


# --- statistics ------------------------------------------------------------------------------------------------


def correlations(x, y) -> dict:
    """Pearson and Spearman (average ranks) over the rows finite in both; NaN with fewer than 3 or no spread."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    if n < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return {"n": n, "pearson": math.nan, "spearman": math.nan}
    xs, ys = x[ok], y[ok]
    return {"n": n, "pearson": float(np.corrcoef(xs, ys)[0, 1]),
            "spearman": float(np.corrcoef(rankdata(xs), rankdata(ys))[0, 1])}  # fmt: skip


def top_overlap(x, y, frac: float) -> float:
    """The share of x's top `frac` rows by |value| that are also among y's top `frac` (rows finite in both)."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    ax, ay = np.abs(x[ok]), np.abs(y[ok])
    if not len(ax):
        return math.nan
    k = max(1, int(round(frac * len(ax))))
    top_x, top_y = np.argpartition(-ax, k - 1)[:k], np.argpartition(-ay, k - 1)[:k]
    return len(np.intersect1d(top_x, top_y)) / k


def leave_one_out(folds: np.ndarray, frac: float | None = None) -> dict:
    """Fold k against the mean of the other folds, for every k; the means over k (folds: [N, K])."""
    per_p, per_s, per_top = [], [], []
    for k in range(folds.shape[1]):
        others = np.delete(folds, k, axis=1).mean(axis=1)
        r = correlations(folds[:, k], others)
        per_p.append(r["pearson"])
        per_s.append(r["spearman"])
        if frac:
            per_top.append(top_overlap(folds[:, k], others, frac))
    out = {"pearson": _mean(per_p), "spearman": _mean(per_s), "per_fold_pearson": per_p, "per_fold_spearman": per_s}
    if frac:
        out["top_overlap"] = _mean(per_top)
    return out


def _mean(values) -> float:
    values = [v for v in values if not math.isnan(v)]
    return float(np.mean(values)) if values else math.nan


def _min(values) -> float:
    values = [v for v in values if not math.isnan(v)]
    return float(min(values)) if values else math.nan


def heldout(folds: np.ndarray, fold_test: np.ndarray, fold_ids: list[int]) -> np.ndarray:
    """Each row's value at the fold whose test set holds it; NaN when that fold is not among fold_ids."""
    out = np.full(folds.shape[0], np.nan)
    for j, k in enumerate(fold_ids):
        sel = fold_test == k
        out[sel] = folds[sel, j]
    return out


# --- inputs ----------------------------------------------------------------------------------------------------


@dataclass
class ModelScores:
    logfc: np.ndarray  # [N, K]
    logfc_fwd: np.ndarray | None  # [N, K], None when a fold has no strands
    logfc_rev: np.ndarray | None
    pval: np.ndarray  # abs_logfc.pval, [N, K]
    null_abs: list[np.ndarray]  # per fold, the null rows' |logFC|


def read_variant_list(path: Path) -> dict[str, dict]:
    """variant_id -> stratum, source, type (SNV, MNV or indel) and fold_test (-1 for none)."""
    out = {}
    with gzip.open(path, "rt", newline="") if path.suffix == ".gz" else open(path, newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            ref, alt = row["ref"], row["alt"]
            vtype = "SNV" if len(ref) == len(alt) == 1 else "MNV" if len(ref) == len(alt) else "indel"
            fold = (row.get("fold_test") or "").strip()
            out[row["variant_id"]] = {"stratum": row.get("stratum", ""), "source": row.get("source", ""),
                                      "type": vtype, "fold_test": int(fold) if fold.lstrip("-").isdigit() else -1}  # fmt: skip
    return out


def load_model(root: Path, dataset: str, model: str, variants: str, null: str, folds: list[int]) -> tuple:
    """(variant ids, ModelScores) from scores.py's per-fold tables; raises FileNotFoundError if a fold is missing."""
    ids, cols = None, {"logfc": [], "logfc_fwd": [], "logfc_rev": [], "abs_logfc.pval": []}
    null_abs = []
    for k in folds:
        table = sc.read_table(root / dataset / f"fold_{k}" / f"{model}.{variants}.tsv.gz")
        if ids is None:
            ids = table["variant_id"]
        elif not np.array_equal(ids, table["variant_id"]):
            raise ValueError(f"{dataset} {model}: fold {k} holds other variants than fold {folds[0]}")
        for c in cols:
            cols[c].append(table.get(c))
        null_abs.append(sc.read_table(root / dataset / f"fold_{k}" / f"{model}.{null}.tsv.gz")["abs_logfc"])
    stranded = all(v is not None for v in cols["logfc_fwd"] + cols["logfc_rev"])

    def stack(column):
        return np.stack(cols[column], axis=1) if column in ("logfc", "abs_logfc.pval") or stranded else None

    return ids, ModelScores(stack("logfc"), stack("logfc_fwd"), stack("logfc_rev"), stack("abs_logfc.pval"), null_abs)


def subsets(meta: list[dict]) -> list[tuple[str, str, np.ndarray]]:
    """(kind, name, mask) for every non-empty subset: all, strata, variant types, callsets."""
    out = [("all", "all", np.ones(len(meta), dtype=bool))]
    for kind, field, names in (("stratum", "stratum", STRATA), ("type", "type", TYPES), ("source", "source", SOURCES)):
        values = np.array([m[field] for m in meta])
        for name in names + tuple(sorted(set(values.tolist()) - set(names) - {""})):
            mask = values == name
            if mask.any():
                out.append((kind, name, mask))
    return out


# --- the comparison --------------------------------------------------------------------------------------------


def compare_dataset(dataset: str, data: dict[str, ModelScores], meta: list[dict], folds: list[int],
                    pairs: list[tuple[str, str]], frac: float) -> dict[str, list[dict]]:  # fmt: skip
    fold_test = np.array([m["fold_test"] for m in meta])
    means = {m: d.logfc.mean(axis=1) for m, d in data.items()}
    held = {m: heldout(d.logfc, fold_test, folds) for m, d in data.items()}
    rows: dict[str, list[dict]] = {"agreement": [], "strand": [], "null": []}
    for kind, name, mask in subsets(meta):
        base = {"dataset": dataset, "subset_kind": kind, "subset": name, "n": int(mask.sum())}
        agree = _collector(rows["agreement"], base)
        for a, b in pairs:
            r = correlations(means[a][mask], means[b][mask])
            agree(a, b, "pearson_fold_mean", r["pearson"])
            agree(a, b, "spearman_fold_mean", r["spearman"])
            same = [correlations(data[a].logfc[mask, j], data[b].logfc[mask, j]) for j in range(len(folds))]
            agree(a, b, "pearson_same_fold", _mean([s["pearson"] for s in same]))
            agree(a, b, "spearman_same_fold", _mean([s["spearman"] for s in same]))
            r = correlations(held[a][mask], held[b][mask])
            agree(a, b, "pearson_heldout", r["pearson"])
            agree(a, b, "spearman_heldout", r["spearman"])
            agree(a, b, "top_overlap_fold_mean", top_overlap(means[a][mask], means[b][mask], frac))
        for m, d in data.items():
            loo = leave_one_out(d.logfc[mask], frac)
            agree(m, m, "pearson_loo", loo["pearson"])
            agree(m, m, "spearman_loo", loo["spearman"])
            agree(m, m, "top_overlap_loo", loo["top_overlap"])
            if d.logfc_fwd is not None:
                rows["strand"] += strand_rows(base, m, d.logfc[mask], d.logfc_fwd[mask], d.logfc_rev[mask], frac)
            rows["null"].append({**base, "model": m, **null_measures(d, mask, fold_test, folds)})
    return rows


def strand_rows(base: dict, model: str, both: np.ndarray, fwd: np.ndarray, rev: np.ndarray, frac: float) -> list[dict]:
    """Per fold: forward-only against reverse-only logFC, and forward-only against both strands; mean and worst."""
    per: dict[str, list[float]] = {"pearson_fwd_rev": [], "spearman_fwd_rev": [], "pearson_fwd_both": [],
                                   "top_overlap_fwd_both": []}  # fmt: skip
    for j in range(both.shape[1]):
        r = correlations(fwd[:, j], rev[:, j])
        per["pearson_fwd_rev"].append(r["pearson"])
        per["spearman_fwd_rev"].append(r["spearman"])
        per["pearson_fwd_both"].append(correlations(fwd[:, j], both[:, j])["pearson"])
        per["top_overlap_fwd_both"].append(top_overlap(fwd[:, j], both[:, j], frac))
    return [{**base, "model": model, "measure": k, "mean": _mean(v), "worst_fold": _min(v)} for k, v in per.items()]


def null_measures(d: ModelScores, mask: np.ndarray, fold_test: np.ndarray, folds: list[int]) -> dict:
    """The null's |logFC| q99 (mean over folds); the share of (variant, fold) |logFC| above its fold's q99; the
    share of variants with held-out abs_logfc.pval < 0.01."""
    q99 = np.array([np.quantile(v, 0.99) for v in d.null_abs])
    above = np.abs(d.logfc[mask]) > q99[None, :]
    p = heldout(d.pval, fold_test, folds)[mask]
    p = p[np.isfinite(p)]
    return {"null_abs_logfc_q99": float(q99.mean()),
            "frac_above_null_q99": float(above.mean()) if above.size else math.nan,
            "frac_heldout_p_lt_0.01": float(np.mean(p < 0.01)) if len(p) else math.nan}  # fmt: skip


def _collector(rows: list[dict], base: dict):
    """A function that appends one agreement row for this subset."""

    def add(a: str, b: str, measure: str, value: float) -> None:
        rows.append({**base, "model_a": a, "model_b": b, "measure": measure, "value": value})

    return add


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})


# --- figures ---------------------------------------------------------------------------------------------------


def _style(ax) -> None:
    ax.grid(True, color=GRID, lw=0.6, ls="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#BBBBBB")
    ax.tick_params(colors=MUTED, labelsize=7)


def _square_limits(ax, *arrays) -> None:
    values = np.abs(np.concatenate([a[np.isfinite(a)] for a in arrays]))
    lim = max(1.05 * float(np.quantile(values, 0.999)), 1e-3) if len(values) else 1.0
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.plot([-lim, lim], [-lim, lim], color=CONTEXT, lw=0.8, zorder=1)
    ax.set_aspect("equal")


def fig_scatter(plt, out: Path, datasets, means, pairs, rows_by_ds) -> None:
    fig, axes = plt.subplots(len(datasets), len(pairs), figsize=(3.3 * len(pairs), 3.4 * len(datasets)), squeeze=False)
    for i, ds in enumerate(datasets):
        for j, (a, b) in enumerate(pairs):
            ax = axes[i][j]
            _style(ax)
            x, y = means[ds][b], means[ds][a]
            ax.scatter(x, y, s=2, color=COLOURS.get(b, INK), alpha=0.3, linewidths=0, rasterized=True, zorder=2)
            _square_limits(ax, x, y)
            r = _lookup(rows_by_ds[ds], a, b, "pearson_fold_mean")
            rho = _lookup(rows_by_ds[ds], a, b, "spearman_fold_mean")
            ax.set_title(f"{ds}\n{label(a)} vs {label(b)}\nr {r:.3f}, Spearman {rho:.3f}",
                         fontsize=7.5, color=INK)  # fmt: skip
            ax.set_xlabel(f"{label(b)} logFC, fold mean", fontsize=7.5, color=INK)
            ax.set_ylabel(f"{label(a)} logFC, fold mean", fontsize=7.5, color=INK)
    fig.suptitle("Fold-mean logFC (log2) per variant; axes cut at the 99.9th percentile of |logFC|; grey: y = x",
                 fontsize=9, color=INK)  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _lookup(rows, a, b, measure, kind="all", subset="all") -> float:
    for r in rows:
        if (r["model_a"], r["model_b"], r["measure"], r["subset_kind"], r["subset"]) == (a, b, measure, kind, subset):
            return r["value"]
    return math.nan


def fig_agreement(plt, out: Path, datasets, pairs, rows_by_ds) -> None:
    fig, axes = plt.subplots(len(datasets), len(pairs), figsize=(4.0 * len(pairs), 3.3 * len(datasets)), squeeze=False)
    for i, ds in enumerate(datasets):
        rows = rows_by_ds[ds]
        keys = []
        for r in rows:
            if r["measure"] == "pearson_fold_mean" and (r["subset_kind"], r["subset"]) not in keys:
                keys.append((r["subset_kind"], r["subset"]))
        x = np.arange(len(keys))
        for j, (a, b) in enumerate(pairs):
            ax = axes[i][j]
            _style(ax)
            cross = [_lookup(rows, a, b, "pearson_fold_mean", *key) for key in keys]
            loo_a = [_lookup(rows, a, a, "pearson_loo", *key) for key in keys]
            loo_b = [_lookup(rows, b, b, "pearson_loo", *key) for key in keys]
            ax.scatter(x - 0.15, loo_a, marker="D", s=22, color="#5A5A5A", edgecolor="white", linewidth=1.0,
                       zorder=3, label=f"{label(a)} ceiling (leave one fold out)")  # fmt: skip
            ax.scatter(x + 0.15, loo_b, marker="s", s=22, color="#A8A8A8", edgecolor="white", linewidth=1.0,
                       zorder=3, label=f"{label(b)} ceiling (leave one fold out)")  # fmt: skip
            ax.scatter(x, cross, marker="o", s=40, color=COLOURS.get(b, INK), edgecolor="white", linewidth=1.0,
                       zorder=4, label="between the models (fold means)")  # fmt: skip
            ax.set_xticks(x)
            ax.set_xticklabels([name for _, name in keys], rotation=45, ha="right", fontsize=7, color=INK)
            finite = [v for v in cross + loo_a + loo_b if not math.isnan(v)]
            ax.set_ylim(min(finite + [0.5]) - 0.05, 1.01)
            ax.set_ylabel("Pearson r of logFC", fontsize=7.5, color=INK)
            ax.set_title(f"{ds}\n{label(a)} vs {label(b)}", fontsize=7.5, color=INK)
            ax.legend(fontsize=6, frameon=False, loc="lower left")
    fig.suptitle("Agreement between the models against each model's own fold-to-fold ceiling", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_strand(plt, out: Path, datasets, data, folds) -> None:
    models = [m for m in MODELS if all(m in data[ds] and data[ds][m].logfc_fwd is not None for ds in datasets)]
    if not models:
        return
    fig, axes = plt.subplots(len(datasets), len(models), figsize=(3.2 * len(models), 3.4 * len(datasets)),
                             squeeze=False)  # fmt: skip
    for i, ds in enumerate(datasets):
        for j, m in enumerate(models):
            ax = axes[i][j]
            _style(ax)
            fwd, rev = data[ds][m].logfc_fwd[:, 0], data[ds][m].logfc_rev[:, 0]
            ax.scatter(fwd, rev, s=2, color=COLOURS.get(m, INK), alpha=0.3, linewidths=0, rasterized=True, zorder=2)
            _square_limits(ax, fwd, rev)
            r = correlations(fwd, rev)
            ax.set_title(f"{ds}\n{label(m)}, fold {folds[0]}: r {r['pearson']:.3f}", fontsize=7.5, color=INK)
            ax.set_xlabel("logFC, forward strand", fontsize=7.5, color=INK)
            ax.set_ylabel("logFC, reverse-complement strand", fontsize=7.5, color=INK)
    fig.suptitle("Strand agreement: each strand's logFC alone; grey: y = x", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _survival(values: np.ndarray, points: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """(x, share of values >= x) on a grid of quantiles, for a log-log plot."""
    v = np.sort(values[np.isfinite(values) & (values > 0)])
    if not len(v):
        return np.array([]), np.array([])
    idx = np.unique(np.round(np.geomspace(1, len(v), points)).astype(int) - 1)
    return v[idx], (len(v) - idx) / len(v)


def null_phrase(null: str) -> str:
    """How fig_null and its caption name the null: window set `null` is build_windows.py --null's shuffled one."""
    return "the shuffled null" if null == "null" else f"the null ({null})"


def fig_null(plt, out: Path, datasets, data, null: str = "null") -> None:
    models = [m for m in MODELS if all(m in data[ds] for ds in datasets)]
    if not models:
        return
    null_legend = "null (shuffled windows)" if null == "null" else f"null ({null})"
    fig, axes = plt.subplots(len(datasets), len(models), figsize=(3.2 * len(models), 3.0 * len(datasets)),
                             squeeze=False)  # fmt: skip
    for i, ds in enumerate(datasets):
        for j, m in enumerate(models):
            ax = axes[i][j]
            _style(ax)
            d = data[ds][m]
            curves = ((np.concatenate(d.null_abs), CONTEXT, null_legend),
                      (np.abs(d.logfc).ravel(), COLOURS.get(m, INK), "variants"))  # fmt: skip
            for values, colour, legend in curves:
                x, y = _survival(values)
                ax.plot(x, y, color=colour, lw=2, solid_capstyle="round", label=legend)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"{ds}\n{label(m)}", fontsize=7.5, color=INK)
            ax.set_xlabel("|logFC|, per fold", fontsize=7.5, color=INK)
            ax.set_ylabel("share of rows at or above", fontsize=7.5, color=INK)
            ax.legend(fontsize=6, frameon=False, loc="lower left")
    fig.suptitle(f"Effect sizes of the variants against {null_phrase(null)}, all folds pooled", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=200)
    plt.close(fig)


# --- the report ------------------------------------------------------------------------------------------------


def _f(value, digits=3) -> str:
    """A number for a table cell; a dash when it is missing."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    return f"{value:.{digits}f}"


def _table(header: list[str], body: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return lines + ["| " + " | ".join(row) + " |" for row in body] + [""]


def _now_et() -> str:
    """Now, in US Eastern time (the cluster's clock runs 6 h ahead of it)."""
    try:
        return datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M ET")
    except ZoneInfoNotFoundError:  # no time zone database
        return datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")


def _subset_keys(agree: list[dict]) -> list[tuple[str, str, int]]:
    """(kind, name, n) of every subset, in the order compare_dataset made them."""
    keys = []
    for r in agree:
        key = (r["subset_kind"], r["subset"], r["n"])
        if r["measure"] == "pearson_fold_mean" and key not in keys:
            keys.append(key)
    return keys


def _agreement_table(agree, keys, pairs, models, cross: str, ceiling: str) -> list[str]:
    """One row per subset: each pair's `cross` measure, then each model's `ceiling` measure."""
    header = ["subset", "n", *(f"{label(a)} vs {label(b)}" for a, b in pairs), *(f"ceiling {label(m)}" for m in models)]
    body = []
    for kind, name, n in keys:
        cells = [_f(_lookup(agree, a, b, cross, kind, name)) for a, b in pairs]
        cells += [_f(_lookup(agree, m, m, ceiling, kind, name)) for m in models]
        body.append([name, str(n), *cells])
    return _table(header, body)


def _strand_table(strand: list[dict], keys, models, top: float) -> list[str]:
    header = ["model", "subset", "fwd vs rev, Pearson (worst fold)", "fwd vs rev, Spearman",
              "fwd only vs both, Pearson", f"fwd only vs both, top {top:.0%} share"]  # fmt: skip
    body = []
    for m in models:
        for kind, name, _ in keys:
            got = {r["measure"]: r for r in strand if (r["model"], r["subset_kind"], r["subset"]) == (m, kind, name)}
            if kind == "source" or not got:
                continue
            fwd_rev = got["pearson_fwd_rev"]
            body.append([label(m), name, f"{_f(fwd_rev['mean'])} ({_f(fwd_rev['worst_fold'])})",
                         _f(got["spearman_fwd_rev"]["mean"]), _f(got["pearson_fwd_both"]["mean"]),
                         _f(got["top_overlap_fwd_both"]["mean"])])  # fmt: skip
    return _table(header, body)


def _null_table(null: list[dict]) -> list[str]:
    header = ["model", "subset", "n", "null q99 of abs logFC", "share above the null q99", "share held-out p < 0.01"]
    body = [[label(r["model"]), r["subset"], str(r["n"]), _f(r["null_abs_logfc_q99"], 4),
             _f(r["frac_above_null_q99"]), _f(r["frac_heldout_p_lt_0.01"])]
            for r in null if r["subset_kind"] in ("all", "stratum")]  # fmt: skip
    return _table(header, body)


def report(out_dir: Path, args, datasets, pairs, models, rows, sizes, figures) -> str:
    """Write report.md: per dataset the agreement, strand and null tables, then the figures."""
    top = f"{args.top:.0%}"
    n_folds = len(args.folds)
    lines = [
        f"# Variant scores: {', '.join(label(m) for m in models)}",
        "",
        f"- Generated {_now_et()} by `workflows/variant_scoring/compare.py`.",
        f"- Scores `{args.scores_root}`: variants `{args.variants}`, null `{args.null}`, folds "
        f"{', '.join(map(str, args.folds))}; variant list `{args.list}`.",
        "- Every number is on logFC, the log2 ratio of ALT to REF strand-averaged predicted counts.",
        "- B (between the models): the two models' fold-mean logFC. C (a model's ceiling): fold k against the mean of "
        "the model's other folds, averaged over k. A cross-model number near both ceilings means the two models agree "
        "as well as each agrees with itself.",
        f"- B and C are not on equal terms: B compares two {n_folds}-fold means, C one fold with the mean of "
        f"{n_folds - 1}, and A one fold with one. Averaging alone lifts B, so B can sit above both ceilings when the "
        "folds disagree; C is each model's fold-to-fold reproducibility, not an upper bound on B. These are the "
        "attribution comparison's definitions.",
        f"- Top share: of model A's top {top} of variants by |fold-mean logFC|, the share also among model B's; "
        f"independent rankings would share about {top}.",
        "",
    ]
    for ds in datasets:
        agree = [r for r in rows["agreement"] if r["dataset"] == ds]
        keys = _subset_keys(agree)
        lines += [f"## {ds}", ""] + ([sizes[ds], ""] if ds in sizes else [])
        lines += ["**Pearson r of logFC: fold means (B) and leave-one-fold-out ceilings (C)**", ""]
        lines += _agreement_table(agree, keys, pairs, models, "pearson_fold_mean", "pearson_loo")
        lines += ["**Spearman rho of logFC: fold means (B) and ceilings (C)**", ""]
        lines += _agreement_table(agree, keys, pairs, models, "spearman_fold_mean", "spearman_loo")
        lines += [f"**Top {top} share by |logFC|: fold means and ceilings**", ""]
        lines += _agreement_table(agree, keys, pairs, models, "top_overlap_fold_mean", "top_overlap_loo")
        measures = ("pearson_same_fold", "pearson_heldout", "spearman_heldout")
        body = [[f"{label(a)} vs {label(b)}", *(_f(_lookup(agree, a, b, m)) for m in measures)] for a, b in pairs]
        lines += ["**All variants: same fold (A, mean over folds) and held-out fold (D)**", ""]
        lines += _table(["pair", "same fold, Pearson", "held-out fold, Pearson", "held-out fold, Spearman"], body)
        strand = [r for r in rows["strand"] if r["dataset"] == ds]
        if strand:
            lines += ["**Strand agreement, mean over folds**", ""] + _strand_table(strand, keys, models, args.top)
        lines += ["**Against the null**", ""] + _null_table([r for r in rows["null"] if r["dataset"] == ds])
    lines += ["## Figures", ""]
    for name, caption in figures:
        lines += [f"**{caption}**", "", f"![{caption}]({name})", ""]
    text = "\n".join(lines) + "\n"
    (out_dir / "report.md").write_text(text)
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores-root", type=Path, required=True, help="scores.py's --out-root")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--list", type=Path, required=True, help="the variant list TSV")
    ap.add_argument("--variants", required=True, help="window-set name of the variants")
    ap.add_argument("--null", default="null", help="window-set name of the null (default null)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--pairs", nargs="+", default=list(PAIRS), help="model_a:model_b (default %(default)s)")
    ap.add_argument("--top", type=float, default=0.01, help="top share for the overlaps (default 0.01)")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%F %T")  # fmt: skip

    variant_meta = read_variant_list(args.list)
    wanted = [tuple(p.split(":", 1)) for p in args.pairs]
    data: dict[str, dict[str, ModelScores]] = {}
    metas: dict[str, list[dict]] = {}
    sizes: dict[str, str] = {}
    for ds in args.datasets:
        data[ds], ids = {}, None
        named = {m for pair in wanted for m in pair}
        for m in [m for m in MODELS if m in named] + sorted(named - set(MODELS)):
            try:
                m_ids, scores = load_model(args.scores_root, ds, m, args.variants, args.null, args.folds)
            except FileNotFoundError as err:
                logger.warning("%s %s: no scores (%s); left out", ds, m, err.filename)
                continue
            if ids is not None and not np.array_equal(ids, m_ids):
                logger.error("%s: %s holds other variants than the other models", ds, m)
                return 1
            ids, data[ds][m] = m_ids, scores
        if ids is None:
            logger.error("%s: no model has scores", ds)
            return 1
        missing = [v for v in ids.tolist() if v not in variant_meta]
        if missing:
            logger.error("%s: %d scored variant(s) are not in %s, e.g. %s", ds, len(missing), args.list, missing[0])
            return 1
        metas[ds] = [variant_meta[v] for v in ids.tolist()]
        meta_json = args.scores_root / ds / f"{next(iter(data[ds]))}.{args.variants}.json"
        if meta_json.exists():
            fold0 = json.loads(meta_json.read_text())["folds"].get(str(args.folds[0]), {})
            sizes[ds] = (f"{len(ids):,} variants scored; null {fold0.get('n_null', '?')} rows; AAQ reference "
                         f"{fold0.get('n_peaks', '?')} peak windows; smallest p {_f(fold0.get('min_p_right'), 6)} "
                         f"(abs_logfc), {_f(fold0.get('min_p_two_tailed'), 6)} (logfc).")  # fmt: skip
    pairs = [(a, b) for a, b in wanted if all(a in data[ds] and b in data[ds] for ds in args.datasets)]
    models = [m for m in MODELS if all(m in data[ds] for ds in args.datasets)]
    if not pairs:
        logger.error("no pair of models has scores for every dataset")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: dict[str, list[dict]] = {"agreement": [], "strand": [], "null": []}
    for ds in args.datasets:
        for key, values in compare_dataset(ds, data[ds], metas[ds], args.folds, pairs, args.top).items():
            rows[key] += values
    for key, values in rows.items():
        write_rows(args.out_dir / f"{key}.tsv", values)

    figures = []
    if not args.no_figures:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows_by_ds = {ds: [r for r in rows["agreement"] if r["dataset"] == ds] for ds in args.datasets}
        means = {ds: {m: d.logfc.mean(axis=1) for m, d in data[ds].items()} for ds in args.datasets}
        fig_scatter(plt, args.out_dir / "fig_scatter.png", args.datasets, means, pairs, rows_by_ds)
        fig_agreement(plt, args.out_dir / "fig_agreement.png", args.datasets, pairs, rows_by_ds)
        fig_strand(plt, args.out_dir / "fig_strand.png", args.datasets, data, args.folds)
        fig_null(plt, args.out_dir / "fig_null.png", args.datasets, data, args.null)
        figures = [("fig_scatter.png", "Fold-mean logFC, model against model"),
                   ("fig_agreement.png", "Agreement by subset against the leave-one-fold-out ceilings"),
                   ("fig_strand.png", "Forward against reverse-complement logFC"),
                   ("fig_null.png", f"Variants against {null_phrase(args.null)}")]  # fmt: skip
        figures = [f for f in figures if (args.out_dir / f[0]).exists()]
    report(args.out_dir, args, args.datasets, pairs, models, rows, sizes, figures)
    logger.info("wrote %s: %d datasets, pairs %s", args.out_dir / "report.md", len(args.datasets),
                ", ".join(f"{a}:{b}" for a, b in pairs))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
