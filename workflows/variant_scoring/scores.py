#!/usr/bin/env python3
"""Count-based variant scores from each model's predictions, as kundajelab/variant-scorer (VS) computes them.

VS is kundajelab/variant-scorer, main @0e1e341; "VS file:line" refers to that commit. Only VS's count scores are
computed here. JSD and the other profile scores are left out on purpose.

Strand average
  VS predicts both strands and averages the predicted counts, exp(log counts), before taking any ratio
  (src/utils/helpers.py:169-170, 177-178, 195-197). So the strand-averaged log counts used here are
  log((exp(fwd) + exp(rev)) / 2), recomputed from the per-strand arrays whenever they are present. That is the log of
  the arithmetic mean of the counts, not the mean of the log counts. The predictor's own a1_logcounts / a2_logcounts
  are only checked against it: a file that holds the mean of the log counts is reported (and its per-strand arrays
  are used); a file that matches neither stops the run. Without a reverse strand (VS --forward_only), the forward
  log counts are used as they are.

Per fold and model (VS's column names; allele1 is REF, allele2 ALT)
  allele1_logcounts, allele2_logcounts   strand-averaged log counts (natural log); VS's allele*_pred_counts are exp()
  logfc                   log2(c2 / c1) of the strand-averaged counts (helpers.py:225)
  abs_logfc               |logfc|
  logfc.pval              two-tailed empirical p of logfc against the null rows' logfc (variant_scoring.py:279)
  abs_logfc.pval          right-tailed empirical p of |logfc| against the null rows' |logfc| (variant_scoring.py:280):
                          the two-sided test of the effect, and the p-value to read
  allele1_quantile, allele2_quantile
                          max(mean(peak_counts < c), 1 / P) among the same model's strand-averaged predictions on
                          the P peak windows (helpers.py:212-213)
  active_allele_quantile  AAQ, max(allele1_quantile, allele2_quantile) (variant_scoring.py:287), which is also the
                          quantile of the larger allele's counts
  quantile_change         allele2_quantile - allele1_quantile (variant_scoring.py:288), and quantile_change.pval,
                          two-tailed against the null's (:299-300)
  logfc_fwd, logfc_rev    the logFC from each strand alone, for the strand check (when both strands are present)
  The p-values are VS's get_pvals (helpers.py:319-335), unchanged: (rank + 1) / (n + 1) against the null's n rows, so
  the smallest p is 1 / (n + 1) right-tailed and 2 / (n + 1) two-tailed. Two-tailed p is not capped at 1.

The null
  VS's null rows are list rows whose window is dinucleotide-shuffled, with both alleles written back at the centre;
  the same model scores them, and their logFC is log2(c2 / c1) in the shuffled background (variant_generator.py:43-47,
  62-72; variant_scoring.py:77-89, 150-163). So the null's predictions need allele2 as well as allele1. A null file
  without a2 stops the run. The null's quantiles use the same peak predictions as the variants', as in VS (:150-155).
  --null may also name another window set built in variant mode (a1 and a2), such as the genomic null that
  build_windows.py --list builds from genomic_null.py's list (random substitutions away from peaks, in real sequence);
  a set without src_ids, as that one, gives null tables without the src_id column.

Across folds (VS's variant_summary_across_folds.py, plus the held-out fold)
  <score>.mean            the arithmetic mean over folds of each per-fold score (:37-43)
  <score>.mean.pval       the geometric mean of the per-fold p-values (:8-9, 44-47), a heuristic, not a calibrated p
  <score>.heldout, <score>.heldout.pval
                          the row of the fold whose test chromosomes hold the variant (the list's fold_test column),
                          NaN when that fold was not scored
  As VS (:29-35), every fold's rows must be the same variants in the same order.

The full ChromBPNet model (cbp_full)
  Its count head is LogSumExp over the bias-free and bias models' log counts, so the full model's predicted counts are
  the sum of the two parts' (fork chrombpnet/training/models/chrombpnet_with_bias_model.py:130-131,
  chrombpnet/training/utils/layers.py:15-16). derive_full() applies that per strand. When a dataset and fold have no
  cbp_full predictions but have both parts, they are derived here. When cbp_full predictions exist alongside both
  parts, they are compared with the derivation and the largest difference is recorded.

Non-finite predictions
  A NaN or infinite log count in any predictions file (variants, null or peaks; for cbp_full, either part) stops
  that model's scoring: np.sort puts NaN last, so the row would otherwise get the smallest p-value and an allele
  quantile of 1, and a non-finite null row would shift every variant's p-value. VS does not check this.

Inputs (FILE INTERFACES of the pipeline)
  <windows-root>/<name>.npz                 ids [N] (build_windows.py)
  <preds-root>/<dataset>/fold_<k>/<model>.<name>.npz
                                            a1_logcounts, a2_logcounts, a1_fwd, a1_rev, a2_fwd, a2_rev: float32 [N];
                                            peak windows have allele1 only
  --list                                    the variant list TSV; its fold_test column gives the held-out fold
  Window-set names may contain {dataset}, which is replaced by each dataset's name (e.g. --peaks 'peaks.{dataset}').

Outputs, under <out-root>/<dataset>/
  fold_<k>/<model>.<variants>.tsv.gz        per-fold variant scores (columns above)
  fold_<k>/<model>.<null>.tsv.gz            the null rows' scores
  <model>.<variants>.summary.tsv.gz         across folds
  <model>.<variants>.json                   inputs, sizes, the strand-average check, the cbp_full derivation
  Every table also goes to <name>.npz beside its <name>.tsv.gz: the same columns, one array each, float32 for floats,
  int64 for integers, numpy unicode for text. The TSV is written from those same float32 values, with 7 significant
  digits, so the two agree digit for digit. --no-fold-tsv leaves out the per-fold TSVs (and deletes one left by an
  earlier run); the .npz files and the summary TSVs are always written. read_table() reads a table's .npz when there is
  one, which at millions of rows takes seconds where parsing the TSV takes minutes, whatever the two files' ages: a TSV
  rewritten later by a scores.py that writes no .npz is not read while an older .npz sits beside it. Both files are
  written under a temporary name, then renamed.
  float32 rounds exact fractions: an AAQ of 45000/50000 is stored as 0.89999998, and a p-value k / (n + 1) equal to
  0.01 in float64 can be stored a hair below it. A threshold on a stored value is compared with a margin of a float32
  step (or at the null's own resolution, 1 / (n + 1)), not exactly.

    python workflows/variant_scoring/scores.py --datasets <ds> [<ds> ...] --preds-root <root>/preds \\
        --windows-root <root>/windows --list <root>/lists/<variants>.tsv --variants <variants> --null null \\
        --peaks 'peaks.{dataset}' --out-root <root>/scores [--folds 0 1 2 3 4] [--models ...] [--no-fold-tsv]

numpy only, so it imports in the Cherimoya and the ChromBPNet environments alike (and in the repository's default
environment). Holds one model's predictions for one fold at a time. Single-threaded: run one process per dataset to
score several at once. At 3.2M variants, a 100k-row null and 50k peaks, a dataset's four models take about 2-3 min with
--no-fold-tsv and about 5-6 min with the per-fold TSVs (each 3.2M-row TSV about 9 s to format and gzip).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import operator
import os
import sys
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np

logger = logging.getLogger("scores")

LN2 = float(np.log(2.0))
VS_COMMIT = "kundajelab/variant-scorer@0e1e34199e63112aa618748bb79a206fc491300a"
MODELS = ("cherimoya", "cbp_nobias", "cbp_bias", "cbp_full")
# Models whose predictions can be derived from others': the full ChromBPNet model from its two parts.
DERIVED = {"cbp_full": ("cbp_nobias", "cbp_bias")}
PRED_KEYS = ("a1_fwd", "a1_rev", "a2_fwd", "a2_rev", "a1_logcounts", "a2_logcounts")
# Scores aggregated across folds, and the tail of each p-value VS computes (variant_scoring.py:279-280, 299-300).
SCORES = ("logfc", "abs_logfc", "active_allele_quantile", "quantile_change")
PVALS = {"logfc": "both", "abs_logfc": "right", "quantile_change": "both"}
TEXT_COLUMNS = ("variant_id", "id", "src_id")
# How far a predictor's stored strand average may sit from a recomputation (float32 storage of log counts).
STRAND_TOL = 1e-5
# gzip level of the TSVs. On a 3.2M-row table, level 6 compresses at ~17 MB/s and level 1 at ~130 MB/s (measured, one
# core), for a file only ~10 % larger; rows are formatted a chunk at a time.
TSV_LEVEL = 1
TSV_CHUNK = 1 << 16


def strand_mean(fwd, rev) -> np.ndarray:
    """log((exp(fwd) + exp(rev)) / 2): the log of VS's strand-averaged counts (helpers.py:169-170, 195-197)."""
    return np.logaddexp(np.asarray(fwd, np.float64), np.asarray(rev, np.float64)) - LN2


def allele_logcounts(preds: dict, allele: str) -> np.ndarray:
    """An allele's strand-averaged log counts, recomputed from the strands when both are present."""
    fwd, rev = preds.get(f"{allele}_fwd"), preds.get(f"{allele}_rev")
    if fwd is not None and rev is not None:
        return strand_mean(fwd, rev)
    if fwd is not None:
        return np.asarray(fwd, np.float64)
    if f"{allele}_logcounts" in preds:
        return np.asarray(preds[f"{allele}_logcounts"], np.float64)
    raise KeyError(f"no {allele}_fwd or {allele}_logcounts in the predictions")


def stored_average(preds: dict, allele: str) -> tuple[str, float | None, float | None]:
    """Which strand average <allele>_logcounts holds, and its largest distance from each candidate:
    'counts' (VS's), 'log_counts' (the mean of the two strands' log counts), 'other', or 'unchecked' when the
    file lacks the field or a strand."""
    stored, fwd, rev = (preds.get(f"{allele}_{k}") for k in ("logcounts", "fwd", "rev"))
    if stored is None or fwd is None or rev is None:
        return "unchecked", None, None
    stored = np.asarray(stored, np.float64)
    log_mean = (np.asarray(fwd, np.float64) + np.asarray(rev, np.float64)) / 2
    off_counts = float(np.max(np.abs(stored - strand_mean(fwd, rev)), initial=0.0))
    off_logs = float(np.max(np.abs(stored - log_mean), initial=0.0))
    kind = "counts" if off_counts <= STRAND_TOL else "log_counts" if off_logs <= STRAND_TOL else "other"
    return kind, off_counts, off_logs


def logfc(lc1, lc2) -> np.ndarray:
    """log2(c2 / c1) from natural-log counts (VS helpers.py:225)."""
    return (np.asarray(lc2, np.float64) - np.asarray(lc1, np.float64)) / LN2


def allele_quantile(lc, peak_lc) -> np.ndarray:
    """VS's allele quantile, max(mean(peak_counts < c), 1 / P) (helpers.py:212-213), by binary search. Log counts
    order as counts do, so comparing them is comparing VS's counts."""
    ref = np.sort(np.asarray(peak_lc, np.float64))
    if not len(ref) or not np.isfinite(ref).all():
        raise ValueError(f"{len(ref)} peak predictions, {int((~np.isfinite(ref)).sum())} of them not finite")
    q = np.searchsorted(ref, np.asarray(lc, np.float64), side="left") / len(ref)
    return np.maximum(q, 1.0 / len(ref))


def get_pvals(obs, bg, tail: str) -> np.ndarray:
    """VS's empirical p-values (helpers.py:319-335), unchanged. right: (#{bg >= obs} + 1) / (n + 1); left:
    (#{bg <= obs} + 1) / (n + 1); both: twice the smaller of the two, not capped at 1."""
    if tail not in ("right", "left", "both"):
        raise ValueError(f"tail {tail!r}")
    sorted_bg = np.sort(np.asarray(bg, np.float64))
    if not np.isfinite(sorted_bg).all():
        # np.sort puts NaN last, so a non-finite null row would silently shift every right-tail p-value.
        raise ValueError(f"the null holds {int((~np.isfinite(sorted_bg)).sum())} non-finite score(s)")
    obs = np.asarray(obs, np.float64)
    n = len(sorted_bg)
    pval_right = (n - np.searchsorted(sorted_bg, obs, side="left") + 1) / (n + 1)
    pval_left = (np.searchsorted(sorted_bg, obs, side="right") + 1) / (n + 1)
    if tail == "right":
        return pval_right
    if tail == "left":
        return pval_left
    return np.minimum(pval_left, pval_right) * 2


def derive_full(nobias: dict, bias: dict) -> dict[str, np.ndarray]:
    """The full ChromBPNet model's log counts from its parts, float32 like a predictor's output.

    Per key, logaddexp(nobias, bias): the full model's LogSumExp count head, so the parts' counts add (fork
    chrombpnet_with_bias_model.py:130-131, layers.py:15-16). The strand averages are then recomputed from the
    derived strands (VS's average). Because VS averages counts, logaddexp of the parts' VS strand averages gives the
    same value; logaddexp of means of log counts would not, which is why the strands come first."""
    out: dict[str, np.ndarray] = {}
    for key in PRED_KEYS:
        if key in nobias and key in bias:
            out[key] = np.logaddexp(np.asarray(nobias[key], np.float64), np.asarray(bias[key], np.float64))
    for allele in ("a1", "a2"):
        if f"{allele}_fwd" in out and f"{allele}_rev" in out:
            out[f"{allele}_logcounts"] = strand_mean(out[f"{allele}_fwd"], out[f"{allele}_rev"])
    return {k: v.astype(np.float32) for k, v in out.items()}


def score(preds: dict, peak_lc: np.ndarray) -> dict[str, np.ndarray]:
    """VS's per-fold count scores of one window set, before p-values (variant_scoring.py:247-289)."""
    lc1, lc2 = allele_logcounts(preds, "a1"), allele_logcounts(preds, "a2")
    if lc1.shape != lc2.shape:
        raise ValueError(f"allele1 has {lc1.shape} predictions, allele2 {lc2.shape}")
    q1, q2 = allele_quantile(lc1, peak_lc), allele_quantile(lc2, peak_lc)
    fc = logfc(lc1, lc2)
    out = {
        "allele1_logcounts": lc1,
        "allele2_logcounts": lc2,
        "logfc": fc,
        "abs_logfc": np.abs(fc),
        "allele1_quantile": q1,
        "allele2_quantile": q2,
        "active_allele_quantile": np.maximum(q1, q2),
        "quantile_change": q2 - q1,
    }
    if all(k in preds for k in ("a1_fwd", "a1_rev", "a2_fwd", "a2_rev")):
        out["logfc_fwd"] = logfc(preds["a1_fwd"], preds["a2_fwd"])
        out["logfc_rev"] = logfc(preds["a1_rev"], preds["a2_rev"])
    return out


def score_fold(variants: dict, null: dict, peaks: dict) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """One model and fold: the variants' scores with their p-values against the null, and the null's scores."""
    if not ("a2_fwd" in null or "a2_logcounts" in null):
        raise ValueError(
            "the null predictions have no allele2: VS forms each null row's logFC between allele1 and allele2 in the "
            "same shuffled window (variant_generator.py:43-47, 62-72), so the null's a2 windows must be scored too"
        )
    peak_lc = allele_logcounts(peaks, "a1")
    var_scores, null_scores = score(variants, peak_lc), score(null, peak_lc)
    for name, tail in PVALS.items():
        var_scores[f"{name}.pval"] = get_pvals(var_scores[name], null_scores[name], tail)
    return var_scores, null_scores


def geo_mean(p: np.ndarray) -> np.ndarray:
    """VS's geo_mean_overflow over folds (variant_summary_across_folds.py:8-9)."""
    return np.exp(np.log(p).mean(axis=0))


def aggregate_folds(per_fold: dict[int, dict[str, np.ndarray]], fold_test: np.ndarray) -> dict[str, np.ndarray]:
    """VS's summary across folds plus the held-out fold (module docstring). fold_test: the fold whose test
    chromosomes hold each variant, -1 for none."""
    folds = sorted(per_fold)
    fold_test = np.asarray(fold_test, dtype=np.int64)
    out: dict[str, np.ndarray] = {"heldout_fold": fold_test}
    for name in SCORES:
        out[f"{name}.mean"] = np.stack([per_fold[k][name] for k in folds]).mean(axis=0)
        if f"{name}.pval" in per_fold[folds[0]]:
            out[f"{name}.mean.pval"] = geo_mean(np.stack([per_fold[k][f"{name}.pval"] for k in folds]))
    for name in SCORES + tuple(f"{s}.pval" for s in PVALS):
        if name not in per_fold[folds[0]]:
            continue
        heldout = np.full(len(fold_test), np.nan)
        for k in folds:
            sel = fold_test == k
            heldout[sel] = per_fold[k][name][sel]
        out[name.replace(".pval", ".heldout.pval") if name.endswith(".pval") else f"{name}.heldout"] = heldout
    return out


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def write_table(path: Path, columns: dict[str, np.ndarray]) -> None:
    """A gzipped TSV, one column per key: floats with 7 significant digits (NaN as nan), integers as integers,
    text as it is. Written under a temporary name, then renamed. Rows are formatted TSV_CHUNK at a time with one
    %-format per row, which writes each float as format(x, '.7g') does."""
    values, formats = [], []
    for v in columns.values():
        v = np.asarray(v)
        if v.dtype.kind == "f":
            formats.append("%.7g")
        elif v.dtype.kind in "biu":
            v = v.astype(np.int64)
            formats.append("%d")
        else:
            formats.append("%s")
        values.append(v)
    row = "\t".join(formats) + "\n"
    n = len(values[0]) if values else 0
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with gzip.open(partial, "wb", compresslevel=TSV_LEVEL) as fh:
        fh.write(("\t".join(columns) + "\n").encode())
        for start in range(0, n, TSV_CHUNK):
            chunk = [v[start : start + TSV_CHUNK].tolist() for v in values]
            fh.write("".join([row % r for r in zip(*chunk)]).encode())
    os.replace(partial, path)


def npz_path(path: Path) -> Path:
    """Where a table's .npz goes: <name>.tsv.gz (or <name>.tsv) -> <name>.npz, in the same directory."""
    path = Path(path)
    for suffix in (".tsv.gz", ".tsv"):
        if path.name.endswith(suffix):
            return path.with_name(path.name.removesuffix(suffix) + ".npz")
    return path.with_name(path.name + ".npz")


def table_arrays(columns: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The columns as a table stores them: floats float32, integers and booleans int64, anything else numpy unicode."""
    out = {}
    for name, values in columns.items():
        v = np.asarray(values)
        if v.dtype.kind == "f":
            v = v.astype(np.float32, copy=False)
        elif v.dtype.kind in "biu":
            v = v.astype(np.int64, copy=False)
        elif v.dtype.kind != "U":
            v = v.astype(str)
        out[name] = v
    return out


def write_npz_table(path: Path, columns: dict[str, np.ndarray]) -> None:
    """The columns, as table_arrays stores them, to npz_path(path): one array per column, in column order. Written
    under a temporary name, then renamed."""
    out = npz_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".partial")
    with open(partial, "wb") as fh:
        np.savez(fh, **table_arrays(columns))
    os.replace(partial, out)


def save_table(path: Path, columns: dict[str, np.ndarray], tsv: bool = True) -> None:
    """A table of scores: its .npz always, then its TSV from the same float32 values, so that the two agree digit for
    digit. tsv=False leaves the TSV out and deletes one an earlier run left at `path`, which would no longer match."""
    arrays = table_arrays(columns)
    write_npz_table(path, arrays)
    if tsv:
        write_table(path, arrays)
    else:
        Path(path).unlink(missing_ok=True)


def _number_column(values: np.ndarray) -> np.ndarray:
    """An .npz column as read_table returns it: float64, or str for text."""
    if values.dtype.kind in "fbiu":
        return values.astype(np.float64)
    return values if values.dtype.kind == "U" else values.astype(str)


def read_table(path: Path, columns: Iterable[str] | None = None) -> dict[str, np.ndarray]:
    """A scores table, from its .npz (npz_path) when there is one, else from the TSV at `path` (a write_table file or
    any TSV with a header); FileNotFoundError for `path` when there is neither. Either way, text columns come back as
    str and every other column as float64 (TEXT_COLUMNS are always text; a TSV column that does not parse as numbers
    is text too). `columns`: only these, those that the table has; the .npz then reads nothing else."""
    wanted = None if columns is None else list(columns)
    npz = npz_path(path)
    if npz.is_file():
        with np.load(npz, allow_pickle=False) as z:
            names = z.files if wanted is None else [c for c in wanted if c in z.files]
            return {name: _number_column(z[name]) for name in names}
    with gzip.open(path, "rt", newline="") if Path(path).suffix == ".gz" else open(path, newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader)
        table = list(zip(*reader)) or [()] * len(header)
    out: dict[str, np.ndarray] = {}
    for name, values in zip(header, table):
        if wanted is not None and name not in wanted:
            continue
        if name in TEXT_COLUMNS:
            out[name] = np.array(values, dtype=str)
            continue
        try:
            out[name] = np.array(values, dtype=np.float64)
        except ValueError:
            out[name] = np.array(values, dtype=str)
    return out


def read_list_columns(path: Path, names: Iterable[str], optional: Iterable[str] = ()) -> dict[str, tuple[str, ...]]:
    """Columns of a TSV with a header (the variant list), as tuples of text in file order: every one of `names`, which
    must be there, and those of `optional` that are. One C-level pass per row, so millions of rows take seconds."""
    names, optional = list(names), list(optional)
    with gzip.open(path, "rt", newline="") if Path(path).suffix == ".gz" else open(path, newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader, [])
        missing = [c for c in names if c not in header]
        if missing:
            raise ValueError(f"{path}: no column(s) {', '.join(missing)} in its header")
        found = [c for c in names + optional if c in header]
        index = [header.index(c) for c in found]
        rows = filter(None, reader)  # a blank line is an empty row
        if len(index) == 1:
            return {found[0]: tuple(row[index[0]] for row in rows)}
        table = list(zip(*map(operator.itemgetter(*index), rows))) or [()] * len(found)
    return dict(zip(found, table))


def fold_numbers(values: Iterable[str]) -> np.ndarray:
    """fold_test texts -> int64, -1 where one is empty or not an integer."""
    values = list(values)
    return np.fromiter((int(v) if v.strip().lstrip("-").isdigit() else -1 for v in values), dtype=np.int64,
                       count=len(values))  # fmt: skip


def align(ids: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    """The position in `ids` of each of `wanted` (ids unique): arange when they are the same, else one sort of ids.
    ValueError naming the first of `wanted` that ids lack."""
    ids, wanted = np.asarray(ids), np.asarray(wanted)
    if len(ids) == len(wanted) and np.array_equal(ids, wanted):
        return np.arange(len(ids))
    if not len(wanted):
        return np.zeros(0, dtype=np.int64)
    if not len(ids):
        raise ValueError(f"{len(wanted)} id(s) are not there, e.g. {wanted[0]}")
    order = np.argsort(ids, kind="stable")
    rows = order[np.minimum(np.searchsorted(ids, wanted, sorter=order), len(ids) - 1)]
    absent = np.flatnonzero(ids[rows] != wanted)
    if len(absent):
        raise ValueError(f"{len(absent)} id(s) are not there, e.g. {wanted[absent[0]]}")
    return rows


def read_fold_test(path: Path) -> dict[str, int]:
    """variant_id -> fold_test from the variant list; -1 where the column is empty or not an integer."""
    cols = read_list_columns(path, ("variant_id",), optional=("fold_test",))
    folds = fold_numbers(cols["fold_test"]) if "fold_test" in cols else np.full(len(cols["variant_id"]), -1)
    return dict(zip(cols["variant_id"], folds.tolist()))


def pred_path(root: Path, dataset: str, fold: int, model: str, name: str) -> Path:
    return root / dataset / f"fold_{fold}" / f"{model}.{name}.npz"


def _text(values) -> np.ndarray:
    values = np.asarray(values)
    return values if values.dtype.kind == "U" else values.astype(str)


def _check_rows(path: Path, preds: dict, ids: np.ndarray) -> None:
    """A predictions file must hold its windows' rows: as many, and the same ids in order when it stores ids."""
    for key, values in preds.items():
        if key in PRED_KEYS and len(values) != len(ids):
            raise ValueError(f"{path}: {key} has {len(values)} rows, its windows {len(ids)}")
    if "ids" in preds and not np.array_equal(_text(preds["ids"]), _text(ids)):
        raise ValueError(f"{path}: its ids are not its windows' ids, row for row")


def load_preds(root: Path, dataset: str, fold: int, model: str, name: str, ids: np.ndarray, notes: dict) -> dict:
    """A model's predictions on one window set (rows = the windows' ids), derived for cbp_full when only its parts
    exist; when both it and its parts exist, the largest gap to the derivation is noted."""
    path = pred_path(root, dataset, fold, model, name)
    part_paths = [pred_path(root, dataset, fold, p, name) for p in DERIVED.get(model, ())]
    parts = [load_npz(p) for p in part_paths] if part_paths and all(p.exists() for p in part_paths) else []
    for part_path, part in zip(part_paths, parts):
        _check_rows(part_path, part, ids)
    if path.exists():
        preds = load_npz(path)
        _check_rows(path, preds, ids)
        if parts:
            derived = derive_full(*parts)
            gaps = [float(np.max(np.abs(preds[k].astype(np.float64) - derived[k]), initial=0.0))
                    for k in derived if k in preds]  # fmt: skip
            notes[f"{name}.max_abs_vs_derived"] = max(gaps) if gaps else None
        notes[f"{name}.source"] = str(path)
    elif parts:
        preds = derive_full(*parts)
        notes[f"{name}.source"] = "derived: logaddexp of " + " and ".join(str(p) for p in part_paths)
    else:
        raise FileNotFoundError(path)
    # A non-finite prediction would be ranked as the most extreme row: np.sort puts NaN last, so its logFC gets the
    # smallest p-value get_pvals can give and its allele quantile is 1. Refuse it here, for the variant, null and peak
    # files alike and for the derived cbp_full (logaddexp carries a NaN of either part).
    for key in PRED_KEYS:
        if key in preds:
            bad = np.flatnonzero(~np.isfinite(np.asarray(preds[key], np.float64)))
            if len(bad):
                raise ValueError(f"{notes[f'{name}.source']}: {key} has {len(bad)} non-finite value(s), first at "
                                 f"row {bad[0]} ({ids[bad[0]]}); get_pvals and allele_quantile would rank them as "
                                 "the most extreme")  # fmt: skip
    for allele in ("a1", "a2"):
        kind, off_counts, off_logs = stored_average(preds, allele)
        notes[f"{name}.{allele}_logcounts"] = {"holds": kind, "max_abs_from_counts_mean": off_counts,
                                               "max_abs_from_log_mean": off_logs}  # fmt: skip
        if kind == "other":
            raise ValueError(f"{path}: {allele}_logcounts is neither strand average of {allele}_fwd and {allele}_rev "
                             f"(off by {off_counts:.3g} and {off_logs:.3g}); the file is inconsistent")  # fmt: skip
        if kind == "log_counts":
            logger.warning("%s: %s_logcounts holds the mean of the strands' log counts, not VS's mean of counts; "
                           "scoring uses the strands", path, allele)  # fmt: skip
    return preds


def window_ids(windows_root: Path, name: str) -> tuple[np.ndarray, np.ndarray | None]:
    with np.load(windows_root / f"{name}.npz", allow_pickle=False) as z:
        return z["ids"], (z["src_ids"] if "src_ids" in z.files else None)


def run_model(args, dataset: str, model: str, names: dict[str, str], fold_test: np.ndarray, ids: dict) -> dict:
    """Score one dataset and model on every fold; write the per-fold tables, the summary and the JSON."""
    out_dir = args.out_root / dataset
    meta: dict = {
        "dataset": dataset,
        "model": model,
        "names": names,
        "follows": VS_COMMIT,
        "strand_average": "log((exp(fwd) + exp(rev)) / 2), VS's mean of counts",
        "folds": {},
    }
    per_fold = {}
    for k in args.folds:
        notes: dict = {}
        variants = load_preds(args.preds_root, dataset, k, model, names["variants"], ids["variants"], notes)
        null = load_preds(args.preds_root, dataset, k, model, names["null"], ids["null"], notes)
        peaks = load_preds(args.preds_root, dataset, k, model, names["peaks"], ids["peaks"], notes)
        var_scores, null_scores = score_fold(variants, null, peaks)
        del variants, null, peaks
        save_table(out_dir / f"fold_{k}" / f"{model}.{names['variants']}.tsv.gz",
                   {"variant_id": ids["variants"], **var_scores}, tsv=args.fold_tsv)  # fmt: skip
        null_cols = {"id": ids["null"]} | ({"src_id": ids["null_src"]} if ids["null_src"] is not None else {})
        save_table(out_dir / f"fold_{k}" / f"{model}.{names['null']}.tsv.gz", null_cols | null_scores,
                   tsv=args.fold_tsv)  # fmt: skip
        notes.update(n_variants=len(ids["variants"]), n_null=len(ids["null"]), n_peaks=len(ids["peaks"]),
                     min_p_right=1 / (len(ids["null"]) + 1), min_p_two_tailed=2 / (len(ids["null"]) + 1),
                     null_abs_logfc_q95=float(np.quantile(null_scores["abs_logfc"], 0.95)),
                     null_abs_logfc_q99=float(np.quantile(null_scores["abs_logfc"], 0.99)))  # fmt: skip
        meta["folds"][str(k)] = notes
        # Only what the summary needs is kept across folds (at 3.2M rows each score is 26 MB).
        per_fold[k] = {key: v for key, v in var_scores.items() if key in SCORES or key.endswith(".pval")}
    summary = aggregate_folds(per_fold, fold_test)
    save_table(out_dir / f"{model}.{names['variants']}.summary.tsv.gz", {"variant_id": ids["variants"], **summary})
    meta["n_heldout"] = int(np.isin(fold_test, args.folds).sum())
    meta["fold_tsv"] = args.fold_tsv
    (out_dir / f"{model}.{names['variants']}.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--preds-root", type=Path, required=True, help="holds <dataset>/fold_<k>/<model>.<name>.npz")
    ap.add_argument("--windows-root", type=Path, required=True, help="holds build_windows.py's <name>.npz")
    ap.add_argument("--list", type=Path, required=True, help="the variant list TSV (for fold_test)")
    ap.add_argument("--variants", required=True, help="window-set name of the variants")
    ap.add_argument("--null", default="null", help="window-set name of the null (default null)")
    ap.add_argument("--peaks", default="peaks.{dataset}", help="window-set name of the peaks (default peaks.{dataset})")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--no-fold-tsv", dest="fold_tsv", action="store_false",
                    help="no per-fold .tsv.gz tables (their .npz and the summary TSVs are written)")  # fmt: skip
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%F %T")  # fmt: skip

    listed = read_list_columns(args.list, ("variant_id",), optional=("fold_test",))
    list_ids = np.array(listed["variant_id"], dtype=str)
    list_folds = fold_numbers(listed["fold_test"]) if "fold_test" in listed else np.full(len(list_ids), -1)
    del listed
    status = 0
    for dataset in args.datasets:
        names = {"variants": args.variants, "null": args.null, "peaks": args.peaks.replace("{dataset}", dataset)}
        ids: dict = {}
        ids["variants"], _ = window_ids(args.windows_root, names["variants"])
        ids["null"], ids["null_src"] = window_ids(args.windows_root, names["null"])
        ids["peaks"], _ = window_ids(args.windows_root, names["peaks"])
        try:
            fold_test = list_folds[align(list_ids, ids["variants"])]
        except ValueError as err:
            logger.error("window ids of %s are not in %s: %s", names["variants"], args.list, err)
            return 1
        for model in args.models:
            started = time.time()
            try:
                meta = run_model(args, dataset, model, names, fold_test, ids)
            except (FileNotFoundError, KeyError, ValueError) as err:
                logger.error("%s %s: %s", dataset, model, err)
                status = 1
                continue
            logger.info("%s %s: %d variants, %d folds, null %d, peaks %d, %.0f s", dataset, model,
                        len(ids["variants"]), len(meta["folds"]), len(ids["null"]), len(ids["peaks"]),
                        time.time() - started)  # fmt: skip
    return status


if __name__ == "__main__":
    sys.exit(main())
