#!/usr/bin/env python3
"""Compare the models' variant scores, for the pilot report: agreement, ceilings, strands and the null.

Reads scores.py's per-fold tables for each dataset and model through scores.read_table (their .npz when there is one,
and only the columns used), joined by variant_id (every model and fold must hold the same variants in the same order),
and the variant list for each variant's stratum, type and callset. Every number is on logFC (log2), the count score;
abs_logfc.pval is the p-value read. Each dataset is compared on the models it has scores for: a model or pair that one
dataset lacks (scores.py refused a fold, say) is a dash in that dataset's tables and a blank panel in its figures, not
a gap in every dataset's. A dataset with no scores, or whose tables do not fit together, is left out and the others are
written; the exit status is then 1.

Per dataset, in every subset of the variants (all; each stratum; SNV, MNV and indel; each callset as the list gives
it, e.g. exome, genome or both; a subset that holds every variant, such as the one source of a list from one callset,
repeats `all` and is left out):
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
  Every statistic, the r in a panel's title and the curves of fig_null use all variants. The scatter panels draw at
  most --figure-points of them (default 200,000), the same seeded random sample (--figure-seed) in every panel of a
  dataset; the captions say so.

Scale: each dataset is loaded, compared and reduced to its figure inputs on its own (in --workers processes, one per
dataset at a time), and only rows and figure inputs come back, so memory holds one dataset per worker. At 3.2M
variants, 4 models and 5 folds that is ~2 GB of scores (float64), ~1 GB of variant ids (0.5 GB each; read from each
model's first fold only), ~1 GB of the list's columns and ~1 GB of temporaries per worker, ~4.5-5 GB in all, and
~3-4 min of statistics per dataset (most of it Spearman's ranks); with a worker per dataset, 10 datasets take ~5 min
and ~50 GB, then ~2-3 min of figures in this process.

    python workflows/variant_scoring/compare.py --scores-root <root>/scores --datasets <ds> [<ds> ...] \\
        --list <root>/lists/<variants>.tsv --variants <variants> --out-dir <root>/compare [--null null] \\
        [--folds 0 1 2 3 4] [--pairs cherimoya:cbp_full ...] [--top 0.01] [--workers 10]

Runs in the pixi `cherimoya` environment (numpy, scipy, matplotlib); the statistics alone need numpy and scipy.
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import logging
import math
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
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
LABELS = {"cherimoya": "Cherimoya", "cbp_full": "ChromBPNet uncorrected", "cbp_nobias": "ChromBPNet corrected",
          "cbp_bias": "ChromBPNet bias model"}  # fmt: skip
# report.md's glossary of the ChromBPNet labels, corrected first: uncorrected is defined by it.
GLOSSARY = {"cbp_nobias": "chrombpnet_nobias.h5, the model with Tn5 bias removed, which variant scoring normally uses",
            "cbp_full": "chrombpnet.h5, the corrected model plus the bias model; it predicts the raw signal, Tn5 "
                        "bias included",
            "cbp_bias": "the Tn5 bias model alone"}  # fmt: skip
PAIRS = ("cherimoya:cbp_full", "cherimoya:cbp_nobias", "cherimoya:cbp_bias")
STRATA = ("summit100", "flank500", "outside")
TYPES = ("SNV", "MNV", "indel")
SOURCES = ("exome", "genome", "both")
# Okabe & Ito (2008): one colour per model, the same in every figure. Checked with the dataviz validator (light
# mode, all pairs): the corrected and the bias model are 7.6 apart under deuteranopia, so every panel names its models.
COLOURS = {"cherimoya": "#D55E00", "cbp_full": "#0072B2", "cbp_nobias": "#009E73", "cbp_bias": "#CC79A7"}
CONTEXT = "#8C8C8C"  # the null, ceilings and reference lines
INK, MUTED, GRID = "#222222", "#666666", "#E6E6E6"
# Points per scatter panel at most (a seeded sample of the variants), and the sample's seed.
FIGURE_POINTS = 200_000
FIGURE_SEED = 0
LOG_FORMAT = {"format": "[%(asctime)s] %(levelname)s %(message)s", "datefmt": "%F %T"}


def label(model: str) -> str:
    return LABELS.get(model, model)


# --- statistics ------------------------------------------------------------------------------------------------


def correlations(x, y, spearman: bool = True) -> dict:
    """Pearson and Spearman (average ranks) over the rows finite in both; NaN with fewer than 3 or no spread.
    spearman=False skips the ranks (two sorts, most of the cost at millions of rows) and reports NaN for it."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    if n < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return {"n": n, "pearson": math.nan, "spearman": math.nan}
    xs, ys = x[ok], y[ok]
    rho = float(np.corrcoef(rankdata(xs), rankdata(ys))[0, 1]) if spearman else math.nan
    return {"n": n, "pearson": float(np.corrcoef(xs, ys)[0, 1]), "spearman": rho}


def subsample(n: int, k: int, seed: int) -> np.ndarray:
    """The rows a scatter panel draws: all n of them when n <= k (or k < 1), else k drawn without replacement with
    default_rng(seed), in row order. The same arguments give the same rows."""
    if k < 1 or n <= k:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size=k, replace=False))


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


def read_variant_list(path: Path) -> dict[str, np.ndarray]:
    """The variant list's columns the comparison uses, one array each, in list order: variant_id, stratum, source,
    type (SNV, MNV or indel, from the allele lengths) and fold_test (-1 for none). Read once per process (again if the
    file changes); the arrays are shared, so callers index them and never write to them."""
    stat = Path(path).stat()
    return _read_variant_list(Path(path), stat.st_mtime_ns, stat.st_size)


@functools.lru_cache(maxsize=1)
def _read_variant_list(path: Path, mtime_ns: int, size: int) -> dict[str, np.ndarray]:
    cols = sc.read_list_columns(path, ("variant_id", "ref", "alt"), optional=("stratum", "source", "fold_test"))
    n = len(cols["variant_id"])
    # Allele lengths only: an array of the alleles themselves would be as wide as the longest insertion.
    ref = np.fromiter(map(len, cols["ref"]), dtype=np.int64, count=n)
    alt = np.fromiter(map(len, cols["alt"]), dtype=np.int64, count=n)

    def text(name: str) -> np.ndarray:
        return np.array(cols[name], dtype=str) if name in cols and n else np.full(n, "", dtype=str)

    return {"variant_id": text("variant_id"), "stratum": text("stratum"), "source": text("source"),
            "type": np.where((ref == 1) & (alt == 1), "SNV", np.where(ref == alt, "MNV", "indel")),
            "fold_test": sc.fold_numbers(cols["fold_test"]) if "fold_test" in cols else np.full(n, -1)}  # fmt: skip


def table_path(root: Path, dataset: str, fold: int, model: str, name: str) -> Path:
    """scores.py's per-fold table; read_table reads its .npz when there is one (always, with --no-fold-tsv)."""
    return root / dataset / f"fold_{fold}" / f"{model}.{name}.tsv.gz"


def first_missing(root: Path, dataset: str, model: str, variants: str, null: str, folds: list[int]) -> Path | None:
    """The first per-fold table of a model that is not there (neither its TSV nor its .npz), or None."""
    for k in folds:
        for name in (variants, null):
            path = table_path(root, dataset, k, model, name)
            if not (sc.npz_path(path).is_file() or path.is_file()):
                return path
    return None


def load_model(root: Path, dataset: str, model: str, variants: str, null: str, folds: list[int]) -> tuple:
    """(variant ids, ModelScores) from scores.py's per-fold tables; raises FileNotFoundError if a fold is missing. The
    ids are read from the first fold's table only (0.5 GB at 3.2M variants): scores.py writes every fold of a model
    from the same window ids, so a later fold is checked by its row count."""
    ids, cols = None, {"logfc": [], "logfc_fwd": [], "logfc_rev": [], "abs_logfc.pval": []}
    null_abs = []
    for k in folds:
        columns = (*cols,) if ids is not None else ("variant_id", *cols)
        table = sc.read_table(table_path(root, dataset, k, model, variants), columns=columns)
        if ids is None:
            ids = table["variant_id"]
        elif len(table["logfc"]) != len(ids):
            raise ValueError(f"{dataset} {model}: fold {k} holds {len(table['logfc'])} variants, fold {folds[0]} "
                             f"{len(ids)}")  # fmt: skip
        for c in cols:
            cols[c].append(table.get(c))
        null_abs.append(sc.read_table(table_path(root, dataset, k, model, null), columns=("abs_logfc",))["abs_logfc"])
    stranded = all(v is not None for v in cols["logfc_fwd"] + cols["logfc_rev"])

    def stack(column):
        return np.stack(cols[column], axis=1) if column in ("logfc", "abs_logfc.pval") or stranded else None

    return ids, ModelScores(stack("logfc"), stack("logfc_fwd"), stack("logfc_rev"), stack("abs_logfc.pval"), null_abs)


def subsets(meta: dict[str, np.ndarray]) -> list[tuple[str, str, np.ndarray]]:
    """(kind, name, mask) for all, the strata, the variant types and the callsets. A subset that is empty or holds
    every variant (it would repeat all) is left out."""
    n = len(meta["fold_test"])
    out = [("all", "all", np.ones(n, dtype=bool))]
    for kind, field, names in (("stratum", "stratum", STRATA), ("type", "type", TYPES), ("source", "source", SOURCES)):
        values = meta[field]
        for name in names + tuple(sorted(set(values.tolist()) - set(names) - {""})):
            mask = values == name
            if mask.any() and not mask.all():
                out.append((kind, name, mask))
    return out


# --- the comparison --------------------------------------------------------------------------------------------


def compare_dataset(dataset: str, data: dict[str, ModelScores], meta: dict[str, np.ndarray], folds: list[int],
                    pairs: list[tuple[str, str]], frac: float) -> dict[str, list[dict]]:  # fmt: skip
    """Every statistic of one dataset, on all of its variants (meta: read_variant_list's columns, row for row)."""
    fold_test = meta["fold_test"]
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
        per["pearson_fwd_both"].append(correlations(fwd[:, j], both[:, j], spearman=False)["pearson"])
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


# --- one dataset, in a worker ----------------------------------------------------------------------------------


def figure_data(data: dict[str, ModelScores], points: int, seed: int) -> dict:
    """What the figures draw of one dataset, small enough to send back from a worker: per model, the fold-mean logFC
    and fold 0's logFC of each strand at the rows subsample(n, points, seed) gives (the same rows for every model),
    fold 0's forward-reverse r over all rows, and fig_null's curves over all rows."""
    n = len(next(iter(data.values())).logfc)
    rows = subsample(n, points, seed)
    out: dict = {"n": n, "shown": len(rows), "means": {}, "strand": {}, "survival": {}}
    for m, d in data.items():
        out["means"][m] = d.logfc.mean(axis=1)[rows]
        if d.logfc_fwd is not None:
            fwd, rev = d.logfc_fwd[:, 0], d.logfc_rev[:, 0]
            out["strand"][m] = {"fwd": fwd[rows], "rev": rev[rows],
                                "pearson": correlations(fwd, rev, spearman=False)["pearson"]}  # fmt: skip
        out["survival"][m] = {"null": _survival(np.concatenate(d.null_abs)),
                              "variants": _survival(np.abs(d.logfc).ravel())}  # fmt: skip
    return out


def dataset_job(job: dict) -> dict:
    """One dataset from start to end: load its models' scores, place them in the variant list, compute every statistic
    and, unless job['figures'] is false, the figure inputs. Returns dataset and error (a message when the inputs do
    not fit together or cannot be read, else None), and with no error rows (compare_dataset's), n and figure
    (figure_data's, or None). An exception becomes the error, with its traceback in the log, so that one dataset's
    tables cannot stop the others'."""
    try:
        return _dataset_job(job)
    except Exception as err:  # a missing or truncated table, a column it lacks: this dataset only
        logger.exception("%s: comparing its scores", job["dataset"])
        return {"dataset": job["dataset"], "error": f"{job['dataset']}: {type(err).__name__}: {err}"}


def _dataset_job(job: dict) -> dict:
    ds = job["dataset"]
    data: dict[str, ModelScores] = {}
    ids = None
    for m in job["models"]:
        m_ids, scores = load_model(job["scores_root"], ds, m, job["variants"], job["null"], job["folds"])
        if ids is not None and not np.array_equal(ids, m_ids):
            return {"dataset": ds, "error": f"{ds}: {m} holds other variants than the other models"}
        ids, data[m] = m_ids, scores
    listed = read_variant_list(job["list"])
    try:
        at = sc.align(listed["variant_id"], ids)
    except ValueError as err:
        return {"dataset": ds, "error": f"{ds}: scored variants are not in {job['list']}: {err}"}
    meta = {key: listed[key][at] for key in ("stratum", "type", "source", "fold_test")}
    rows = compare_dataset(ds, data, meta, job["folds"], job["pairs"], job["top"])
    figure = figure_data(data, job["figure_points"], job["figure_seed"]) if job["figures"] else None
    logger.info("%s: %d variants, models %s", ds, len(ids), ", ".join(data))
    return {"dataset": ds, "error": None, "rows": rows, "n": len(ids), "figure": figure}


def _init_worker(level: int) -> None:
    logging.basicConfig(level=level, **LOG_FORMAT)


def run_jobs(jobs: list[dict], workers: int, level: int = logging.INFO) -> list[dict]:
    """dataset_job for every job, results in job order: in this process with one worker, else in up to `workers`
    processes. They are spawned, not forked, so none inherits this process's threads or memory; each reads the
    variant list once."""
    if workers <= 1 or len(jobs) <= 1:
        return [dataset_job(job) for job in jobs]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=context, initializer=_init_worker,
                             initargs=(level,)) as pool:  # fmt: skip
        return list(pool.map(dataset_job, jobs))


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


def _blank(ax, title: str, why: str = "no scores") -> None:
    """A panel for a model or pair this dataset has no scores for."""
    ax.axis("off")
    ax.set_title(f"{title}\n{why}", fontsize=7.5, color=MUTED)


def sample_note(figs: dict, seed: int) -> str:
    """How the scatter panels sample the variants, for their titles and captions; empty when every dataset is drawn
    whole."""
    sampled = [f for f in figs.values() if f["shown"] < f["n"]]
    if not sampled:
        return ""
    return (f"points: a random sample of {max(f['shown'] for f in sampled):,} variants per dataset (seed {seed}), the "
            "same in every panel of a dataset; r, Spearman and every table use all variants")  # fmt: skip


def _title(fig, text: str, note: str = "") -> None:
    """The figure's title, a sampling note on a second line when there is one, and the layout that leaves room."""
    fig.suptitle(text + (f"\n{note}" if note else ""), fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, min(0.96, 1 - 0.5 / fig.get_figheight()) if note else 0.96))


def fig_scatter(plt, out: Path, datasets, figs, pairs, rows_by_ds, note: str = "") -> None:
    fig, axes = plt.subplots(len(datasets), len(pairs), figsize=(3.3 * len(pairs), 3.4 * len(datasets)), squeeze=False)
    for i, ds in enumerate(datasets):
        for j, (a, b) in enumerate(pairs):
            ax = axes[i][j]
            if a not in figs[ds]["means"] or b not in figs[ds]["means"]:
                _blank(ax, f"{ds}\n{label(a)} vs {label(b)}")
                continue
            _style(ax)
            x, y = figs[ds]["means"][b], figs[ds]["means"][a]
            ax.scatter(x, y, s=2, color=COLOURS.get(b, INK), alpha=0.3, linewidths=0, rasterized=True, zorder=2)
            _square_limits(ax, x, y)
            r = _lookup(rows_by_ds[ds], a, b, "pearson_fold_mean")
            rho = _lookup(rows_by_ds[ds], a, b, "spearman_fold_mean")
            ax.set_title(f"{ds}\n{label(a)} vs {label(b)}\nr {r:.3f}, Spearman {rho:.3f}",
                         fontsize=7.5, color=INK)  # fmt: skip
            ax.set_xlabel(f"{label(b)} logFC, fold mean", fontsize=7.5, color=INK)
            ax.set_ylabel(f"{label(a)} logFC, fold mean", fontsize=7.5, color=INK)
    _title(fig, "Fold-mean logFC (log2) per variant; axes cut at the 99.9th percentile of the drawn |logFC|; "
           "grey: y = x", note)  # fmt: skip
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _lookup(rows, a, b, measure, kind="all", subset="all") -> float:
    for r in rows:
        if (r["model_a"], r["model_b"], r["measure"], r["subset_kind"], r["subset"]) == (a, b, measure, kind, subset):
            return r["value"]
    return math.nan


def fig_agreement(plt, out: Path, datasets, pairs, rows_by_ds, pairs_of: dict | None = None) -> None:
    """pairs_of: each dataset's own pairs (default: every one of pairs); a pair a dataset lacks is a blank panel."""
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
            if pairs_of is not None and (a, b) not in pairs_of[ds]:
                _blank(ax, f"{ds}\n{label(a)} vs {label(b)}")
                continue
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


def fig_strand(plt, out: Path, datasets, figs, folds, note: str = "") -> None:
    models = [m for m in MODELS if any(m in figs[ds]["strand"] for ds in datasets)]
    if not models:
        return
    fig, axes = plt.subplots(len(datasets), len(models), figsize=(3.2 * len(models), 3.4 * len(datasets)),
                             squeeze=False)  # fmt: skip
    for i, ds in enumerate(datasets):
        for j, m in enumerate(models):
            ax = axes[i][j]
            if m not in figs[ds]["strand"]:
                _blank(ax, f"{ds}\n{label(m)}", "no strand scores")
                continue
            _style(ax)
            strand = figs[ds]["strand"][m]
            fwd, rev = strand["fwd"], strand["rev"]
            ax.scatter(fwd, rev, s=2, color=COLOURS.get(m, INK), alpha=0.3, linewidths=0, rasterized=True, zorder=2)
            _square_limits(ax, fwd, rev)
            ax.set_title(f"{ds}\n{label(m)}, fold {folds[0]}: r {strand['pearson']:.3f}", fontsize=7.5, color=INK)
            ax.set_xlabel("logFC, forward strand", fontsize=7.5, color=INK)
            ax.set_ylabel("logFC, reverse-complement strand", fontsize=7.5, color=INK)
    _title(fig, "Strand agreement: each strand's logFC alone; grey: y = x", note)
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


def fig_null(plt, out: Path, datasets, figs, null: str = "null") -> None:
    models = [m for m in MODELS if any(m in figs[ds]["survival"] for ds in datasets)]
    if not models:
        return
    null_legend = "null (shuffled windows)" if null == "null" else f"null ({null})"
    fig, axes = plt.subplots(len(datasets), len(models), figsize=(3.2 * len(models), 3.0 * len(datasets)),
                             squeeze=False)  # fmt: skip
    for i, ds in enumerate(datasets):
        for j, m in enumerate(models):
            ax = axes[i][j]
            if m not in figs[ds]["survival"]:
                _blank(ax, f"{ds}\n{label(m)}")
                continue
            _style(ax)
            survival = figs[ds]["survival"][m]
            curves = ((survival["null"], CONTEXT, null_legend), (survival["variants"], COLOURS.get(m, INK), "variants"))
            for (x, y), colour, legend in curves:
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


def glossary(models) -> list[str]:
    """report.md's header bullet on what the ChromBPNet labels mean, for the ChromBPNet models among `models` only
    (none without one)."""
    named = [m for m in GLOSSARY if m in models]
    if not named:
        return []
    text = " ".join(f"{label(m)}: {GLOSSARY[m]}." for m in named)
    if "cherimoya" in models and "cbp_full" in models:
        text += " Cherimoya has no bias model and, like the uncorrected model, predicts the raw signal."
    elif "cherimoya" in models:
        text += " Cherimoya has no bias model and predicts the raw signal, Tn5 bias included."
    return [f"- {text}"]


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


def report(out_dir: Path, args, datasets, pairs, models, rows, sizes, figures, missing: dict | None = None,
           left_out: dict | None = None) -> str:  # fmt: skip
    """Write report.md: per dataset the agreement, strand and null tables, then the figures. missing: the models of
    `models` each dataset has no scores for; left_out: the datasets not compared, with why."""
    missing, left_out = missing or {}, left_out or {}
    top = f"{args.top:.0%}"
    n_folds = len(args.folds)
    lines = [
        f"# Variant scores: {', '.join(label(m) for m in models)}",
        "",
        f"- Generated {_now_et()} by `workflows/variant_scoring/compare.py`.",
        f"- Scores `{args.scores_root}`: variants `{args.variants}`, null `{args.null}`, folds "
        f"{', '.join(map(str, args.folds))}; variant list `{args.list}`.",
        *glossary(models),
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
        *(f"- **Left out:** {ds} ({why})." for ds, why in left_out.items()),
        "",
    ]
    for ds in datasets:
        agree = [r for r in rows["agreement"] if r["dataset"] == ds]
        keys = _subset_keys(agree)
        lines += [f"## {ds}", ""] + ([sizes[ds], ""] if ds in sizes else [])
        if missing.get(ds):
            lines += [f"Not scored for {ds}: {', '.join(label(m) for m in missing[ds])}; its columns show a dash "
                      "(compare.py's log names the missing table).", ""]  # fmt: skip
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
    ap.add_argument("--figure-points", type=int, default=FIGURE_POINTS,
                    help="variants per scatter panel at most, a seeded sample (default %(default)s)")  # fmt: skip
    ap.add_argument("--figure-seed", type=int, default=FIGURE_SEED, help="seed of that sample (default %(default)s)")
    ap.add_argument("--workers", type=int, default=1,
                    help="datasets compared at once, each in its own process (default 1: here)")  # fmt: skip
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, **LOG_FORMAT)

    wanted = [tuple(p.split(":", 1)) for p in args.pairs]
    named = {m for pair in wanted for m in pair}
    present: dict[str, list[str]] = {}
    left_out: dict[str, str] = {}
    for ds in args.datasets:
        present[ds] = []
        for m in [m for m in MODELS if m in named] + sorted(named - set(MODELS)):
            missing = first_missing(args.scores_root, ds, m, args.variants, args.null, args.folds)
            if missing is not None:
                logger.warning("%s %s: no scores (%s); left out of this dataset", ds, m, missing)
                continue
            present[ds].append(m)
        if not present[ds]:
            logger.error("%s: no model has scores; left out", ds)
            left_out[ds] = "no model has scores"
    # Each dataset on the models and pairs it has (module docstring).
    pairs_of = {ds: [(a, b) for a, b in wanted if a in present[ds] and b in present[ds]] for ds in args.datasets}
    jobs = [{"dataset": ds, "models": present[ds], "scores_root": args.scores_root, "list": args.list,
             "variants": args.variants, "null": args.null, "folds": args.folds, "pairs": pairs_of[ds],
             "top": args.top, "figures": not args.no_figures, "figure_points": args.figure_points,
             "figure_seed": args.figure_seed}
            for ds in args.datasets if ds not in left_out]  # fmt: skip
    if not jobs:
        logger.error("no dataset has scores")
        return 1
    results = {}
    for result in run_jobs(jobs, args.workers, level):
        if result["error"]:
            logger.error("%s; left out", result["error"])
            left_out[result["dataset"]] = result["error"]
        else:
            results[result["dataset"]] = result
    datasets = [ds for ds in args.datasets if ds in results]
    if not datasets:
        logger.error("no dataset could be compared")
        return 1
    pairs = [p for p in wanted if any(p in pairs_of[ds] for ds in datasets)]
    models = [m for m in MODELS if any(m in present[ds] for ds in datasets)]
    if not pairs:
        logger.error("no dataset has both models of any pair; only each model's own rows are written")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: dict[str, list[dict]] = {"agreement": [], "strand": [], "null": []}
    sizes: dict[str, str] = {}
    for ds in datasets:
        result = results[ds]
        for key, values in result["rows"].items():
            rows[key] += values
        meta_json = args.scores_root / ds / f"{present[ds][0]}.{args.variants}.json"
        if meta_json.exists():
            fold0 = json.loads(meta_json.read_text())["folds"].get(str(args.folds[0]), {})
            sizes[ds] = (f"{result['n']:,} variants scored; null {fold0.get('n_null', '?')} rows; AAQ reference "
                         f"{fold0.get('n_peaks', '?')} peak windows; smallest p {_f(fold0.get('min_p_right'), 6)} "
                         f"(abs_logfc), {_f(fold0.get('min_p_two_tailed'), 6)} (logfc).")  # fmt: skip
    for key, values in rows.items():
        write_rows(args.out_dir / f"{key}.tsv", values)

    figures = []
    if not args.no_figures:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figs = {ds: results[ds]["figure"] for ds in datasets}
        note = sample_note(figs, args.figure_seed)
        rows_by_ds = {ds: [r for r in rows["agreement"] if r["dataset"] == ds] for ds in datasets}
        if pairs:
            fig_scatter(plt, args.out_dir / "fig_scatter.png", datasets, figs, pairs, rows_by_ds, note)
            fig_agreement(plt, args.out_dir / "fig_agreement.png", datasets, pairs, rows_by_ds, pairs_of)
        fig_strand(plt, args.out_dir / "fig_strand.png", datasets, figs, args.folds, note)
        fig_null(plt, args.out_dir / "fig_null.png", datasets, figs, args.null)
        sampled = f" ({note})" if note else ""
        figures = [("fig_scatter.png", f"Fold-mean logFC, model against model{sampled}"),
                   ("fig_agreement.png", "Agreement by subset against the leave-one-fold-out ceilings"),
                   ("fig_strand.png", f"Forward against reverse-complement logFC{sampled}"),
                   ("fig_null.png", f"Variants against {null_phrase(args.null)}")]  # fmt: skip
        figures = [f for f in figures if (args.out_dir / f[0]).exists()]
    missing = {ds: [m for m in models if m not in present[ds]] for ds in datasets}
    report(args.out_dir, args, datasets, pairs, models, rows, sizes, figures, missing, left_out)
    logger.info("wrote %s: %d of %d datasets, pairs %s", args.out_dir / "report.md", len(datasets),
                len(args.datasets), ", ".join(f"{a}:{b}" for a, b in pairs))  # fmt: skip
    return 1 if left_out or not pairs else 0


if __name__ == "__main__":
    sys.exit(main())
