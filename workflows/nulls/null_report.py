#!/usr/bin/env python3
"""The two nulls side by side: how wide each model's null is, what it calls and why, and the frequency gradient.

scores.py scores the variants twice: against VS's dinucleotide-shuffled null (window set `null`, tables under
<root>/scores) and against the GC-matched genomic null (window set `genomic_null`, tables under
<root>/scores_genomic_null; genomic_null.py); any other null <name> is read from <root>/scores_<name>. A null whose
scores directory is missing is left out with a note. Every number is on logFC (log2), the count score. A p-value is
VS's right-tailed empirical p of |logFC| against the fold's null rows, (rank + 1) / (n + 1) (scores.get_pvals), and a
call is p < 0.01.

Rows and models
  A shuffled-null row keeps its source variant's alleles at the centre of the shuffled window (build_windows.py
  --null), so it takes its source's type: its src_id is a variant_id of the list, whose ref and alt make it an SNV,
  MNV or indel and an SNV a transition (Ti: A<>G, C<>T) or a transversion (Tv). A row of a null without src_id (the
  genomic null) is typed from its own id, chrom:pos:ref:alt, or by the lengths in the list's compact form for long
  ids, chrom:pos:<len ref>><len alt>:<hash>. The genomic null holds SNVs only, so the like-for-like rows are each
  null's SNV rows. Models: Cherimoya (cherimoya), ChromBPNet uncorrected (cbp_full) and corrected (cbp_nobias); the
  bias model (cbp_bias) in A only. The pair compared is Cherimoya and ChromBPNet uncorrected.

Per dataset
  A  widths: per model, null and subset of its rows (all; SNV, and its Ti and Tv rows; MNV and indel when there are
     any), the |logFC| q50, q90 and q99.9 with the folds pooled, and q99 as the mean of the per-fold q99s (each fold's
     p-values come from its own null) with their range. The variants the same way (all, SNV, summit100, summit100
     SNV). q99_vs_cbp_full is the row's q99 over that of ChromBPNet uncorrected on the same rows.
  B  calls under each null: the per-fold share (the mean over folds, and its range) and the fold-mean calls (the
     summary's abs_logfc.mean.pval, VS's geometric mean of the per-fold p), for all variants (view all) and for the
     SNVs (SNV); when the null holds rows other than SNVs, also the SNVs against its SNV rows only (SNV_vs_SNV_rows:
     p recomputed per fold with scores.get_pvals, then scores.geo_mean). Cherimoya against ChromBPNet uncorrected: the
     calls of both and of one only, the shares, and the share in common when Cherimoya calls as many variants as
     ChromBPNet uncorrected (its smallest p first, ties by the larger |fold-mean logFC|). Each model across two nulls:
     its calls under each and under both.
  C  the p of a |logFC| of 0.25, 0.5, 0.75 and 1 (fold change 1.19, 1.41, 1.68 and 2) against each model's null,
     folds pooled; all rows and SNV rows.
  D  CpG: the CpG count of each null row's allele-1 window, a C (code 1) followed by a G (code 2) in
     <root>/windows/<null>.npz (a dinucleotide shuffle keeps a window's CpG count). The |logFC| q99 of the SNV rows
     in CpG bins 0-14, 15-29, 30-49, 50-79, 80-119 and 120+, folds pooled; and each null without src_id (the genomic
     one) reweighted to the CpG-bin mix of the SNV rows of the first null with src_id (the shuffled one). The weighted
     q99 is the smallest |logFC| whose cumulative weight reaches 99 % of the total.
  E  Cherimoya against ChromBPNet uncorrected on the same rows, per fold: Pearson r of logFC and the share of
     Cherimoya's top 1 % rows by |logFC| that are among the top 1 % of ChromBPNet uncorrected (compare.py's
     top_overlap); each null (all rows, SNV rows) and the variants (all, SNVs).
  F  held-out rows only: per fold k, the SNV rows on fold k's test chromosomes and their |logFC| q99 in fold k. A
     null row's chromosome is that of its src_id, or of its id without one; the list's fold_test, a function of the
     chromosome, maps it to a fold, and a row on a chromosome the list does not hold is left out. The mean over
     folds, against the mean of the per-fold q99s over all SNV rows. The variants' SNVs the same way.
  G  |logFC| q99 by allele-1 log-count quintile, folds pooled; the quintile edges per model from the nulls' SNV
     rows and the SNV variants, pooled (so the variants set them).
  H  frequency, the common end: the NFE minor allele frequency from the list's AF_nfe (2 decimals; MAF = min(AF,
     1 - AF), taken in hundredths), in bins 0.00 (below 0.5 % in NFE, common elsewhere), 0.01-0.04 and >= 0.05, and a
     non-numeric AF_nfe as `empty`. Per model, within mut_class and stratum (each, and all): n, the q50, q90 and q99
     of |logfc.mean| (the absolute fold-mean logFC; not abs_logfc.mean, the fold mean of |logFC|) and the share
     called under each null. Per null and mut_class, the MAF bins of the variants called by both models, by
     Cherimoya only and by ChromBPNet uncorrected only, against all variants.
  R  the decision numbers' sampling noise from the null rows alone, a bootstrap: --bootstrap replicates (default
     200), each drawing a null's SNV rows again with replacement, once for both models and every fold
     (default_rng(--bootstrap-seed)); its Ti and Tv rows are the drawn rows of each. The two nulls are drawn
     independently; the variants' ratio is held fixed (hundreds of thousands of rows). Per replicate the q99 ratio as A
     computes it, the excess removed and the rule's branch; reported: each ratio's point value, its 2.5 and 97.5 %
     quantiles (lo, hi) and SD over the replicates, and the share of replicates in each branch. Every dataset is scored
     against the same null rows, so a branch shared by all datasets does not confirm those rows' draw independently;
     R says how far the draw alone moves each number. About 5-15 s per null and dataset at 100,000 rows.

The decision rule
  Written before the genomic null's numbers were seen; RULE below, quoted in report.md. It reads A's Cherimoya /
  ChromBPNet uncorrected q99 ratio on the genomic null's rows: at most 1.15, the shuffled null's extra width for
  Cherimoya is an artefact of shuffled sequence; at least 1.40, a model property; in between, partly both. How much:
  the share of the shuffled null's excess over the variants' ratio (summit100 SNVs) that the genomic null removes,
  (shuffled - genomic) / (shuffled - variants). R gives each of these numbers an interval and the branch a share of
  bootstrap replicates.

Inputs
  <root>/scores[_<null>]/<dataset>/fold_<k>/<model>.<variants>.tsv.gz and .../fold_<k>/<model>.<null>.tsv.gz, and
  <root>/scores[_<null>]/<dataset>/<model>.<variants>.summary.tsv.gz: scores.py's tables. Each is read from its .npz
  sibling when there is one, and only the columns used (a table's ids alone are 0.5 GB at 3.2M variants); else with
  scores.read_table. The variants' logfc, abs_logfc and allele1_logcounts come from one null's directory (the models'
  predictions are the same in each), abs_logfc.pval and the summaries from each.
  <root>/windows/<null>.npz: ids and a1, for D.
  --list: variant_id, chrom, ref, alt, stratum and fold_test; mut_class and AF_nfe for H. Without mut_class the
  classes come from ref and alt, without the CpG split; without AF_nfe, H is left out.
  Every table of a dataset must hold the variants in one order, as scores.py writes them. Per dataset, the first
  null's summary ids are matched to the list, each other null's summary ids must equal them, and every other table
  must hold as many rows.

Outputs, in --out-dir, rewritten after every dataset (each under a temporary name, then renamed), so a run that
stops keeps the datasets it finished; report.md says how many:
  A_widths.tsv           dataset, null (or variants), subset, model, n (rows), folds, logc50 (the median allele-1 log
                         counts, folds pooled), q50, q90, q99, q99_min, q99_max, q999, q99_vs_cbp_full
  B_calls.tsv            dataset, null, view, model, n (variants), measure, value; the pair's rows have model
                         cherimoya+cbp_full, and a model across two nulls null <a>+<b>
  C_fixed_effect_p.tsv   dataset, null, subset, model, n (rows x folds), p_0.25, p_0.5, p_0.75, p_1
  D_cpg.tsv              dataset, null, model, cpg_bin (a bin, all, or reweighted), rows, share (of the SNV rows),
                         q99, cpg_mean, cpg_median, reference (the null reweighted to), max_weight, uncovered_share
                         (the reference's rows in bins this null lacks)
  E_agreement.tsv        dataset, set (a null or variants), subset, rows, r_mean, r_min, r_max, top_mean, top_min,
                         top_max
  F_heldout.tsv          dataset, set, subset, model, q99_heldout, q99_all, rows_min, rows_max (held-out rows per
                         fold), q99_fold_min, q99_fold_max, per_fold, q99_heldout_vs_cbp_full
  G_count_quintiles.tsv  dataset, model, group (a null's SNV rows, or variants: the SNVs), quintile, logc_lo, logc_hi,
                         n (rows x folds), q99
  H_frequency.tsv        dataset, part (effect or calls), null, model, group, mut_class, stratum, maf_bin, measure,
                         value
  R_rule_bootstrap.tsv   dataset, measure (q99_ratio, excess_removed, branch_share), null (a null, variants, or
                         <shuffled>+<genomic>), subset (SNV, SNV_Ti, SNV_Tv, summit100_SNV, or the branch), point (the
                         value on all rows; for branch_share the share of replicates), lo, hi, sd, reps (finite
                         replicates)
  report.md              the decision rule's outcome per dataset, then compact tables (medians over datasets for G
                         and H)
  Exit status 1 when a table or the list is missing or inconsistent, or no null has a scores directory; the outputs
  then hold what could be read.

    python workflows/nulls/null_report.py --root <root> --list <root>/lists/<list>.tsv \\
        --datasets <ds> [<ds> ...] --variants variants --nulls null genomic_null --out-dir <root>/analysis \\
        [--bootstrap 200] [--bootstrap-seed 0]

numpy only (the repository's default environment). One dataset in memory at a time: at 3.2M variants and 5 folds
about 3 GB (the variants' per-fold logFC, log counts and p-values in float32, the list and its index).
"""

from __future__ import annotations

import argparse
import csv
import functools
import gzip
import logging
import math
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("nulls", "variant_scoring")]

import scores as sc  # noqa: E402

logger = logging.getLogger("null_report")

MODELS = ("cherimoya", "cbp_full", "cbp_nobias")
WIDTH_MODELS = (*MODELS, "cbp_bias")  # A also shows the bias model's null
PAIR = ("cherimoya", "cbp_full")
PAIR_NAME = "+".join(PAIR)
LABELS = {"cherimoya": "Cherimoya", "cbp_full": "ChromBPNet uncorrected", "cbp_nobias": "ChromBPNet corrected",
          "cbp_bias": "ChromBPNet bias model"}  # fmt: skip
# report.md's glossary of the ChromBPNet labels, as compare.py's (corrected first: uncorrected is defined by it).
GLOSSARY = {"cbp_nobias": "chrombpnet_nobias.h5, the model with Tn5 bias removed, which variant scoring normally uses",
            "cbp_full": "chrombpnet.h5, the corrected model plus the bias model; it predicts the raw signal, Tn5 "
                        "bias included",
            "cbp_bias": "the Tn5 bias model alone"}  # fmt: skip
NULL_LABELS = {"null": "shuffled", "genomic_null": "genomic"}
ALPHA = 0.01  # a call: p < ALPHA
FIXED = (0.25, 0.5, 0.75, 1.0)  # C: the |logFC| values whose p is read off each null
TOP = 0.01  # E: the top share by |logFC|
CPG_EDGES = (0, 15, 30, 50, 80, 120)  # D: the lower edge of each CpG-count bin
QUINTILES = (0.2, 0.4, 0.6, 0.8)  # G: the inner edges
MIN_ROWS = 200  # report.md: a q99 over fewer rows (folds pooled) is shown as a dash
KINDS = ("SNV", "MNV", "indel")
SNV, MNV, INDEL = 0, 1, 2
TRANSITIONS = frozenset({("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")})
MUT_CLASSES = ("CpG_ti", "CpG_tv", "ti", "tv", "mnv", "indel")  # table_list.py's
STRATA = ("summit100", "flank500")
MAF_BINS = ("0.00", "0.01-0.04", ">=0.05", "empty")
# The decision rule's thresholds on the genomic null's Cherimoya / ChromBPNet uncorrected q99 ratio, and its branches.
ARTEFACT_MAX = 1.15
PROPERTY_MIN = 1.40
BRANCHES = ("null artefact", "partly both", "model property")
BOOTSTRAP = 200  # R: replicates of the null rows
BOOTSTRAP_SEED = 0
BASES = re.compile(r"[ACGTN]+")  # an allele of an id
# The rule as written, quoted verbatim in report.md; build_report then says which label its ChromBPNet-full is.
RULE = (
    "Decision rule, written before the genomic null's numbers were seen (2026-10-13 15:37 ET). The observable is the "
    "Cherimoya / ChromBPNet-full q99 ratio on like-for-like rows: the genomic null against the shuffled null's "
    "SNV-source rows (also within Ti and Tv). Shuffled SNV: 1.46-1.51. On real summit100 variants the q99 ratio is "
    "1.07-1.10.",
    "- genomic ratio near the variants' (<= ~1.15): the shuffled null's extra width for Cherimoya is an effect of "
    "shuffled (out-of-distribution) sequence; Cherimoya's fewer calls are a null artefact; cross-model p-values should "
    "use the genomic null.",
    "- genomic ratio still ~1.4-1.5: Cherimoya responds more strongly to single-base changes in background sequence, "
    "real or shuffled: a model property. Compare the models on effect sizes; Cherimoya stays conservative under either "
    "null.",
    "- in between: partly both; say how much.",
    "Call counts at p < 0.01 under the genomic null (B) read the same way.",
)
NULL_COLUMNS = ("abs_logfc", "logfc", "allele1_logcounts")
VARIANT_COLUMNS = ("abs_logfc", "logfc", "allele1_logcounts")
SUMMARY_COLUMNS = ("abs_logfc.mean.pval", "logfc.mean")
OUTPUTS = {"A": "A_widths.tsv", "B": "B_calls.tsv", "C": "C_fixed_effect_p.tsv", "D": "D_cpg.tsv",
           "E": "E_agreement.tsv", "F": "F_heldout.tsv", "G": "G_count_quintiles.tsv", "H": "H_frequency.tsv",
           "R": "R_rule_bootstrap.tsv"}  # fmt: skip
COLUMNS = {
    "A": ("dataset", "null", "subset", "model", "n", "folds", "logc50", "q50", "q90", "q99", "q99_min", "q99_max",
          "q999", "q99_vs_cbp_full"),
    "B": ("dataset", "null", "view", "model", "n", "measure", "value"),
    "C": ("dataset", "null", "subset", "model", "n", *(f"p_{x:g}" for x in FIXED)),
    "D": ("dataset", "null", "model", "cpg_bin", "rows", "share", "q99", "cpg_mean", "cpg_median", "reference",
          "max_weight", "uncovered_share"),
    "E": ("dataset", "set", "subset", "rows", "r_mean", "r_min", "r_max", "top_mean", "top_min", "top_max"),
    "F": ("dataset", "set", "subset", "model", "q99_heldout", "q99_all", "rows_min", "rows_max", "q99_fold_min",
          "q99_fold_max", "per_fold", "q99_heldout_vs_cbp_full"),
    "G": ("dataset", "model", "group", "quintile", "logc_lo", "logc_hi", "n", "q99"),
    "H": ("dataset", "part", "null", "model", "group", "mut_class", "stratum", "maf_bin", "measure", "value"),
    "R": ("dataset", "measure", "null", "subset", "point", "lo", "hi", "sd", "reps"),
}  # fmt: skip


def label(model: str) -> str:
    return LABELS.get(model, model)


def short(model: str) -> str:
    """The label without "ChromBPNet ", for a header that names several models: Cherimoya / uncorrected / corrected."""
    return label(model).removeprefix("ChromBPNet ")


def null_label(null: str) -> str:
    return NULL_LABELS.get(null, null)


def scores_dir(root: Path, null: str) -> Path:
    """pilot_steps.sh's layout: scores/ against the shuffled null (window set `null`), scores_<null>/ otherwise."""
    return root / ("scores" if null == "null" else f"scores_{null}")


# --- rows, frequencies and the rule (no numpy) ------------------------------------------------------------------


def variant_kind(ref: str, alt: str) -> int:
    """SNV, MNV (REF and ALT of one length above 1) or INDEL."""
    if len(ref) != len(alt):
        return INDEL
    return SNV if len(ref) == 1 else MNV


def parse_id(vid: str) -> tuple[str, int, bool]:
    """(chromosome, kind, transition) of chrom:pos:ref:alt, or of the list's compact form for ids over 40 characters,
    chrom:pos:<len ref>><len alt>:<hash>, typed by its lengths (a transition is never read off a compact id)."""
    parts = vid.split(":")
    if len(parts) != 4:
        raise ValueError(f"id {vid!r} is not chrom:pos:ref:alt")
    chrom, _, ref, alt = parts
    if ">" in ref:
        len_ref, _, len_alt = ref.partition(">")
        kind = INDEL if int(len_ref) != int(len_alt) else SNV if int(len_ref) == 1 else MNV
        return chrom, kind, False
    ref, alt = ref.upper(), alt.upper()
    if not (BASES.fullmatch(ref) and BASES.fullmatch(alt)):
        # e.g. a shuffled-null row's own id (chrom:pos:ref:alt|shuf<n>) where its src_id was meant: refused, not typed
        raise ValueError(f"id {vid!r}: {ref!r} and {alt!r} are not alleles of A, C, G, T and N")
    kind = variant_kind(ref, alt)
    return chrom, kind, kind == SNV and (ref, alt) in TRANSITIONS


def maf_bin(text: str) -> tuple[int, float]:
    """(index into MAF_BINS, MAF) of an allele frequency given with 2 decimals. MAF = min(AF, 1 - AF) is taken in
    hundredths, so that 0.96 gives 0.04 exactly; a text that is not a number in [0, 1] (NaN, empty) is `empty`."""
    try:
        af = float(text)
    except (TypeError, ValueError):
        return len(MAF_BINS) - 1, math.nan
    if not 0.0 <= af <= 1.0:  # also NaN
        return len(MAF_BINS) - 1, math.nan
    hundredths = round(af * 100)
    maf = min(hundredths, 100 - hundredths)
    return (0 if maf == 0 else 1 if maf <= 4 else 2), maf / 100


def decide(shuffled: float, genomic: float, variants: float) -> tuple[str, float]:
    """The rule's branch from the Cherimoya / ChromBPNet uncorrected q99 ratios of the shuffled null's SNV rows, the
    genomic null and the variants (summit100 SNVs), and how much of the shuffled null's excess over the variants' ratio
    the genomic null removes: (shuffled - genomic) / (shuffled - variants), NaN unless the shuffled ratio is larger."""
    if math.isnan(genomic):
        return "not evaluable", math.nan
    removed = (shuffled - genomic) / (shuffled - variants) if shuffled > variants else math.nan
    if genomic <= ARTEFACT_MAX:
        return "null artefact", removed
    if genomic >= PROPERTY_MIN:
        return "model property", removed
    return "partly both", removed


def cpg_bin_names() -> list[str]:
    """0-14, 15-29, 30-49, 50-79, 80-119, 120+."""
    return [f"{lo}-{hi - 1}" for lo, hi in zip(CPG_EDGES, CPG_EDGES[1:])] + [f"{CPG_EDGES[-1]}+"]


def _finite(values) -> list[float]:
    return [float(v) for v in values if v is not None and not math.isnan(v)]


def _mean(values) -> float:
    values = _finite(values)
    return sum(values) / len(values) if values else math.nan


def _min(values) -> float:
    values = _finite(values)
    return min(values) if values else math.nan


def _max(values) -> float:
    values = _finite(values)
    return max(values) if values else math.nan


def _median(values) -> float:
    values = sorted(_finite(values))
    if not values:
        return math.nan
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2


def _ratio(a: float, b: float) -> float:
    return a / b if b > 0 and not math.isnan(a) else math.nan


@dataclass
class Notes:
    """What the report says about its inputs: notes, and problems (a problem makes the exit status 1)."""

    notes: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def note(self, text: str) -> None:
        logger.info(text)
        self.notes.append(text)

    def problem(self, text: str) -> None:
        logger.error(text)
        self.problems.append(text)


# --- statistics -------------------------------------------------------------------------------------------------


def quantile(values, q: float) -> float:
    """numpy's default (type 7) quantile of all the values; NaN for none."""
    values = np.asarray(values, dtype=np.float64).ravel()
    return float(np.quantile(values, q)) if values.size else math.nan


def pearson(x, y) -> float:
    """Pearson r over the rows finite in both; NaN with fewer than 3 rows or no spread."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def top_overlap(x, y, frac: float) -> float:
    """The share of x's top `frac` rows by |value| that are also among y's top `frac` (rows finite in both), as
    compare.py's top_overlap (which this module does not import: compare.py needs scipy)."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    ax, ay = np.abs(x[ok]), np.abs(y[ok])
    if not len(ax):
        return math.nan
    k = max(1, int(round(frac * len(ax))))
    top_x, top_y = np.argpartition(-ax, k - 1)[:k], np.argpartition(-ay, k - 1)[:k]
    return len(np.intersect1d(top_x, top_y)) / k


def weighted_quantile(values, weights, q: float) -> float:
    """The smallest value whose cumulative weight, values ascending, reaches q of the total; NaN without weight."""
    values = np.asarray(values, dtype=np.float64).ravel()
    weights = np.asarray(weights, dtype=np.float64).ravel()
    keep = weights > 0
    values, weights = values[keep], weights[keep]
    if not values.size:
        return math.nan
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    i = int(np.searchsorted(cumulative, q * cumulative[-1], side="left"))
    return float(values[order[min(i, len(order) - 1)]])


def called(p) -> np.ndarray:
    """p < ALPHA, compared in float64: a float32 p would otherwise be compared with float32(0.01)."""
    return np.asarray(p, dtype=np.float64) < ALPHA


def cpg_counts(a1: np.ndarray, chunk: int = 1 << 14) -> np.ndarray:
    """CpG dinucleotides per row of base codes [N, L] (build_windows.py: A=0, C=1, G=2, T=3): a 1 followed by a 2."""
    out = np.zeros(len(a1), dtype=np.int64)
    for start in range(0, len(a1), chunk):
        block = np.asarray(a1[start : start + chunk])
        out[start : start + len(block)] = np.count_nonzero((block[:, :-1] == 1) & (block[:, 1:] == 2), axis=1)
    return out


def cpg_bins(counts) -> np.ndarray:
    """The index of each count's CpG bin (CPG_EDGES)."""
    return np.searchsorted(np.asarray(CPG_EDGES), np.asarray(counts), side="right") - 1


# --- inputs -----------------------------------------------------------------------------------------------------


def npz_of(path: Path) -> Path:
    """The .npz sibling scores.py writes beside each of its tables: the same path with .npz in place of .tsv.gz."""
    return path.with_name(path.name.removesuffix(".tsv.gz") + ".npz")


def has_table(path: Path) -> bool:
    """A table is there when its .tsv.gz or its .npz is (with scores.py --no-fold-tsv, only the .npz)."""
    return npz_of(path).exists() or path.exists()


def read_columns(path: Path, names, optional=()) -> dict[str, np.ndarray]:
    """The named columns of a scores.py table, and those of `optional` it has: from its .npz sibling when there is one
    (only these members are read), else with scores.read_table."""
    npz = npz_of(path)
    if npz.exists():
        source = npz
        with np.load(npz, allow_pickle=False) as z:
            got = {name: z[name] for name in (*names, *optional) if name in z.files}
    elif path.exists():
        source = path
        table = sc.read_table(path)
        got = {name: table[name] for name in (*names, *optional) if name in table}
    else:
        raise FileNotFoundError(f"no {path} or {npz.name}")
    absent = [name for name in names if name not in got]
    if absent:
        raise ValueError(f"{source}: no column {', '.join(absent)}")
    return got


@dataclass
class VariantList:
    """The variant list's columns this report reads, in the list's row order."""

    path: Path
    ids: np.ndarray  # variant_id
    kind: np.ndarray  # int8: SNV, MNV or INDEL, from ref and alt
    ti: np.ndarray  # bool: an SNV transition
    mut_class: np.ndarray  # int8, index into classes
    classes: tuple[str, ...]
    stratum: np.ndarray  # int8, index into strata
    strata: tuple[str, ...]
    maf_bin: np.ndarray | None  # int8, index into MAF_BINS; None without AF_nfe
    maf: np.ndarray | None  # float64 NFE MAF, NaN where AF_nfe is not a number
    fold_test: np.ndarray  # int64, -1 for none
    chrom_fold: dict[str, int]  # the fold_test of each chromosome's rows
    _index: dict[str, int] | None = None

    def index(self) -> dict[str, int]:
        """variant_id -> row, built on first use."""
        if self._index is None:
            self._index = {vid: i for i, vid in enumerate(self.ids.tolist())}
        return self._index


def read_list(path: Path, notes: Notes) -> VariantList:
    """The list's columns as arrays, read in one pass; the chromosome -> fold map from its fold_test."""
    need = ("variant_id", "chrom", "ref", "alt", "stratum", "fold_test")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader, [])
        col = {name: i for i, name in enumerate(header)}
        missing = [name for name in need if name not in col]
        if missing:
            raise ValueError(f"{path}: no column {', '.join(missing)}")
        i_id, i_chrom, i_ref, i_alt, i_stratum, i_fold = (col[name] for name in need)
        i_class, i_af = col.get("mut_class"), col.get("AF_nfe")
        classes = {name: i for i, name in enumerate(MUT_CLASSES)}
        strata = {name: i for i, name in enumerate(STRATA)}
        ids, kind, ti, cls, stratum, fold, bins, maf = [], [], [], [], [], [], [], []
        pairs: dict[tuple[str, int], int] = {}  # (chrom, fold_test) -> rows
        # Parsed once per distinct text (there are few): fold_test -> fold, AF_nfe -> (MAF bin, MAF).
        folds: dict[str, int] = {}
        mafs: dict[str, tuple[int, float]] = {}
        for row in reader:
            if not row:
                continue
            if len(row) < len(header):
                raise ValueError(f"{path} line {reader.line_num}: {len(row)} fields, the header has {len(header)}")
            ref, alt = row[i_ref].upper(), row[i_alt].upper()
            k = variant_kind(ref, alt)
            transition = k == SNV and (ref, alt) in TRANSITIONS
            if i_class is not None:
                name = row[i_class]
            else:
                name = ("ti" if transition else "tv") if k == SNV else "mnv" if k == MNV else "indel"
            value = row[i_fold]
            if value not in folds:
                folds[value] = int(value) if value.strip().lstrip("-").isdigit() else -1
            f = folds[value]
            ids.append(row[i_id])
            kind.append(k)
            ti.append(transition)
            cls.append(classes.setdefault(name, len(classes)))
            stratum.append(strata.setdefault(row[i_stratum], len(strata)))
            fold.append(f)
            key = (row[i_chrom], f)
            pairs[key] = pairs.get(key, 0) + 1
            if i_af is not None:
                text = row[i_af]
                if text not in mafs:
                    mafs[text] = maf_bin(text)
                b, m = mafs[text]
                bins.append(b)
                maf.append(m)
    per_chrom: dict[str, Counter] = {}
    for (chrom, f), count in pairs.items():
        per_chrom.setdefault(chrom, Counter())[f] += count
    if i_class is None:
        notes.note(f"{path}: no mut_class column; H's classes come from ref and alt, without the CpG split")
    if i_af is None:
        notes.note(f"{path}: no AF_nfe column; H is left out")
    chrom_fold = {}
    for chrom, counts in per_chrom.items():
        chrom_fold[chrom] = counts.most_common(1)[0][0]
        if len(counts) > 1:
            notes.problem(f"{path}: fold_test differs within {chrom} ({dict(counts)}); F takes {chrom_fold[chrom]}")
    return VariantList(
        path=path,
        ids=np.array(ids, dtype=str),
        kind=np.array(kind, dtype=np.int8),
        ti=np.array(ti, dtype=bool),
        mut_class=np.array(cls, dtype=np.int8),
        classes=tuple(classes),
        stratum=np.array(stratum, dtype=np.int8),
        strata=tuple(strata),
        maf_bin=np.array(bins, dtype=np.int8) if i_af is not None else None,
        maf=np.array(maf, dtype=np.float64) if i_af is not None else None,
        fold_test=np.array(fold, dtype=np.int64),
        chrom_fold=chrom_fold,
    )


@dataclass
class VariantView:
    """The list's columns in the score tables' row order."""

    n: int
    kind: np.ndarray
    ti: np.ndarray
    mut_class: np.ndarray
    classes: tuple[str, ...]
    stratum: np.ndarray
    strata: tuple[str, ...]
    maf_bin: np.ndarray | None
    maf: np.ndarray | None
    fold_test: np.ndarray


def view_of(ids: np.ndarray, vl: VariantList) -> VariantView:
    """The list's columns for the tables' variant ids: the list itself when the ids are its own, in its order (the
    usual case); else joined on variant_id (build_windows.py may drop rows it cannot build)."""
    if len(ids) == len(vl.ids) and np.array_equal(ids, vl.ids):
        rows = slice(None)
    else:
        index = vl.index()
        rows = np.fromiter((index.get(v, -1) for v in ids.tolist()), dtype=np.int64, count=len(ids))
        if (rows < 0).any():
            raise ValueError(f"{int((rows < 0).sum())} scored variant(s) are not in {vl.path}, e.g. "
                             f"{ids[int(np.argmax(rows < 0))]}")  # fmt: skip
    return VariantView(
        n=len(ids),
        kind=vl.kind[rows],
        ti=vl.ti[rows],
        mut_class=vl.mut_class[rows],
        classes=vl.classes,
        stratum=vl.stratum[rows],
        strata=vl.strata,
        maf_bin=None if vl.maf_bin is None else vl.maf_bin[rows],
        maf=None if vl.maf is None else vl.maf[rows],
        fold_test=vl.fold_test[rows],
    )


@dataclass
class NullRows:
    """A null's rows, the same in every dataset and fold: each row's type, the fold that holds it out, its CpG count."""

    ids: np.ndarray
    from_list: bool  # rows of the variant list (src_id), typed by their source variant
    kind: np.ndarray  # int8
    ti: np.ndarray  # bool
    fold: np.ndarray  # int64, -1 for a chromosome without a fold in the list
    cpg: np.ndarray | None = None  # CpG count of the allele-1 window; None without the windows


def null_rows(ids: np.ndarray, src_ids: np.ndarray | None, vl: VariantList) -> NullRows:
    """Type and held-out fold of each null row: from its source variant in the list when it has a src_id (refused when
    the list lacks it), else from its own id."""
    if src_ids is not None:
        index = vl.index()
        rows = np.fromiter((index.get(s, -1) for s in src_ids.tolist()), dtype=np.int64, count=len(src_ids))
        if (rows < 0).any():
            raise ValueError(f"{int((rows < 0).sum())} null src_id(s) are not variant_ids of {vl.path}, e.g. "
                             f"{src_ids[int(np.argmax(rows < 0))]}")  # fmt: skip
        kind, ti = vl.kind[rows], vl.ti[rows]
        chroms = [s.split(":", 1)[0] for s in src_ids.tolist()]
    else:
        parsed = [parse_id(v) for v in ids.tolist()]
        chroms = [p[0] for p in parsed]
        kind = np.array([p[1] for p in parsed], dtype=np.int8)
        ti = np.array([p[2] for p in parsed], dtype=bool)
    fold = np.array([vl.chrom_fold.get(c, -1) for c in chroms], dtype=np.int64)
    return NullRows(ids=ids, from_list=src_ids is not None, kind=kind, ti=ti, fold=fold)


def null_cpg(path: Path, ids: np.ndarray) -> np.ndarray:
    """The CpG count of each row's allele-1 window in a build_windows.py window set, in the order of `ids`."""
    with np.load(path, allow_pickle=False) as z:
        window_ids, a1 = z["ids"], z["a1"]
    counts = cpg_counts(a1)
    if len(window_ids) == len(ids) and np.array_equal(window_ids, ids):
        return counts
    where = {v: i for i, v in enumerate(window_ids.tolist())}
    rows = np.fromiter((where.get(v, -1) for v in ids.tolist()), dtype=np.int64, count=len(ids))
    if (rows < 0).any():
        raise ValueError(f"{path}: {int((rows < 0).sum())} null row(s) have no window, e.g. "
                         f"{ids[int(np.argmax(rows < 0))]}")  # fmt: skip
    return counts[rows]


def cached_null_rows(cache: dict, null: str, table: dict, vl: VariantList, root: Path, notes: Notes) -> NullRows:
    """The null's rows for a dataset's tables, reused while the tables hold the same ids (the null's windows are the
    same for every dataset)."""
    ids, src_ids = table["id"], table.get("src_id")
    hit = cache.get(null)
    if hit is not None and len(hit.ids) == len(ids) and np.array_equal(hit.ids, ids):
        return hit
    rows = null_rows(ids, src_ids, vl)
    path = root / "windows" / f"{null}.npz"
    if path.exists():
        try:
            rows.cpg = null_cpg(path, ids)
        except Exception as err:  # a missing member, other ids, an unreadable file: D only
            notes.problem(f"the {null_label(null)} null: no CpG counts ({type(err).__name__}: {err}); D leaves it out")
    else:
        notes.note(f"the {null_label(null)} null: no {path}; D leaves it out")
    off = int((rows.fold < 0).sum())
    if off:
        notes.note(f"the {null_label(null)} null: {off} of {len(ids)} rows lie on chromosomes without a fold in the "
                   "list; F leaves them out")  # fmt: skip
    cache[null] = rows
    return rows


@dataclass
class DatasetScores:
    """One dataset's scores. Per-fold arrays are [folds, rows]."""

    name: str
    folds: list[int]
    view: VariantView
    var: dict[str, dict[str, np.ndarray]]  # model -> VARIANT_COLUMNS
    pval: dict[tuple[str, str], np.ndarray]  # (null, model) -> the variants' per-fold abs_logfc.pval
    summary: dict[tuple[str, str], dict[str, np.ndarray]]  # (null, model) -> SUMMARY_COLUMNS
    nulls: dict[str, NullRows]  # the nulls with rows for this dataset
    null: dict[tuple[str, str], dict[str, np.ndarray]]  # (null, model) -> NULL_COLUMNS of the null's rows


def load_dataset(args, ds: str, nulls: list[str], vl: VariantList, cache: dict, notes: Notes) -> DatasetScores | None:
    """Every table this report reads for one dataset; a (null, model) whose tables are missing or inconsistent is
    left out with a problem noted."""
    var: dict = {}
    pval: dict = {}
    summary: dict = {}
    null_scores: dict = {}
    rows_of: dict = {}
    ref_ids = None
    for null in nulls:
        base = scores_dir(args.root, null) / ds
        order_checked = order_ok = False
        for model in WIDTH_MODELS:
            null_paths = [base / f"fold_{k}" / f"{model}.{null}.tsv.gz" for k in args.folds]
            var_paths = [base / f"fold_{k}" / f"{model}.{args.variants}.tsv.gz" for k in args.folds]
            summary_path = base / f"{model}.{args.variants}.summary.tsv.gz"
            needed = null_paths + (var_paths + [summary_path] if model in MODELS else [])
            missing = [p for p in needed if not has_table(p)]
            if missing:
                notes.problem(f"{ds}: {model} against the {null_label(null)} null: {len(missing)} of {len(needed)} "
                              f"tables missing, e.g. {missing[0]}")  # fmt: skip
                continue
            # The null's rows, the same for every model and fold.
            tables = [read_columns(null_paths[0], ("id", *NULL_COLUMNS), ("src_id",))]
            tables += [read_columns(p, NULL_COLUMNS) for p in null_paths[1:]]
            if null not in rows_of:
                rows_of[null] = cached_null_rows(cache, null, tables[0], vl, args.root, notes)
            elif not np.array_equal(tables[0]["id"], rows_of[null].ids):
                notes.problem(f"{ds}: {model}'s {null_label(null)} null rows are not the other models'; left out")
                continue
            m = len(rows_of[null].ids)
            if any(len(t[c]) != m for t in tables for c in NULL_COLUMNS):
                notes.problem(f"{ds}: {model}'s {null_label(null)} null tables do not all hold its {m} rows; left out")
                continue
            null_scores[(null, model)] = {c: np.stack([t[c] for t in tables]) for c in NULL_COLUMNS}
            if model not in MODELS:
                continue
            # The variants: their order checked once per null against the first null's, every length against it.
            if order_checked and not order_ok:
                continue
            s = read_columns(summary_path, SUMMARY_COLUMNS if order_checked else ("variant_id", *SUMMARY_COLUMNS))
            if not order_checked:
                order_checked = True
                ids = s.pop("variant_id")
                if ref_ids is None:
                    ref_ids = ids
                order_ok = len(ids) == len(ref_ids) and np.array_equal(ids, ref_ids)
                if not order_ok:
                    notes.problem(f"{ds}: the variants against the {null_label(null)} null are not in the order of "
                                  "the first null's; its p-values are left out")  # fmt: skip
                    continue
            n = len(ref_ids)
            first = model not in var
            columns = (*VARIANT_COLUMNS, "abs_logfc.pval") if first else ("abs_logfc.pval",)
            per_fold = [read_columns(p, columns) for p in var_paths]
            if any(len(v) != n for v in s.values()) or any(len(t[c]) != n for t in per_fold for c in columns):
                notes.problem(f"{ds}: {model}'s tables against the {null_label(null)} null do not all hold the {n} "
                              "variants; left out")  # fmt: skip
                continue
            summary[(null, model)] = s
            pval[(null, model)] = np.stack([t["abs_logfc.pval"] for t in per_fold])
            if first:
                var[model] = {c: np.stack([t[c] for t in per_fold]) for c in VARIANT_COLUMNS}
    if ref_ids is None or not var:
        notes.problem(f"{ds}: no variant scores under any null; left out")
        return None
    return DatasetScores(name=ds, folds=list(args.folds), view=view_of(ref_ids, vl), var=var, pval=pval,
                         summary=summary, nulls=rows_of, null=null_scores)  # fmt: skip


# --- the sections -----------------------------------------------------------------------------------------------


def null_subsets(rows: NullRows) -> list[tuple[str, np.ndarray]]:
    snv = rows.kind == SNV
    out = [("all", np.ones(len(snv), dtype=bool)), ("SNV", snv), ("SNV_Ti", snv & rows.ti),
           ("SNV_Tv", snv & ~rows.ti)]  # fmt: skip
    return out + [(KINDS[k], rows.kind == k) for k in (MNV, INDEL) if (rows.kind == k).any()]


def variant_subsets(view: VariantView) -> list[tuple[str, np.ndarray]]:
    snv = view.kind == SNV
    out = [("all", np.ones(view.n, dtype=bool)), ("SNV", snv)]
    if "summit100" in view.strata:
        summit = view.stratum == view.strata.index("summit100")
        out += [("summit100", summit), ("summit100_SNV", summit & snv)]
    return out


def width(values: np.ndarray, logc: np.ndarray, mask: np.ndarray) -> dict:
    """A's quantiles of |logFC| over the rows in mask (values and logc: [folds, rows])."""
    sub = np.asarray(values[:, mask], dtype=np.float64)
    out = {"n": int(sub.shape[1]), "folds": int(sub.shape[0])}
    if not sub.size:
        return out | dict.fromkeys(("logc50", "q50", "q90", "q99", "q99_min", "q99_max", "q999"), math.nan)
    q99 = np.quantile(sub, 0.99, axis=1)
    q50, q90, q999 = np.quantile(sub, (0.5, 0.9, 0.999))
    return out | {"logc50": float(np.median(logc[:, mask])), "q50": float(q50), "q90": float(q90),
                  "q99": float(q99.mean()), "q99_min": float(q99.min()), "q99_max": float(q99.max()),
                  "q999": float(q999)}  # fmt: skip


def section_a(ds: DatasetScores) -> list[dict]:
    """Null widths, and the variants' for reference."""
    rows = []
    for null, nr in ds.nulls.items():
        for subset, mask in null_subsets(nr):
            for model in WIDTH_MODELS:
                d = ds.null.get((null, model))
                if d is not None:
                    rows.append({"dataset": ds.name, "null": null, "subset": subset, "model": model,
                                 **width(d["abs_logfc"], d["allele1_logcounts"], mask)})  # fmt: skip
    for subset, mask in variant_subsets(ds.view):
        for model in MODELS:
            d = ds.var.get(model)
            if d is not None:
                rows.append({"dataset": ds.name, "null": "variants", "subset": subset, "model": model,
                             **width(d["abs_logfc"], d["allele1_logcounts"], mask)})  # fmt: skip
    full = {(r["null"], r["subset"]): r["q99"] for r in rows if r["model"] == "cbp_full"}
    for r in rows:
        r["q99_vs_cbp_full"] = _ratio(r["q99"], full.get((r["null"], r["subset"]), math.nan))
    return rows


def pair_overlap(p_a: np.ndarray, p_b: np.ndarray, abs_a: np.ndarray) -> dict:
    """PAIR[0]'s calls against PAIR[1]'s: in both, in one only, the shares, and the share in common when PAIR[0] calls
    as many as PAIR[1] (its smallest p first, ties by the larger |fold-mean logFC|), with the p it reaches there."""
    a, b = called(p_a), called(p_b)
    n_a, n_b, both = int(a.sum()), int(b.sum()), int((a & b).sum())
    first, second = PAIR
    out = {"both": both, f"{first}_only": n_a - both, f"{second}_only": n_b - both,
           f"share_of_{first}": both / n_a if n_a else math.nan,
           f"share_of_{second}": both / n_b if n_b else math.nan,
           "equal_count_shared": math.nan, "equal_count_p": math.nan}  # fmt: skip
    if n_b:
        p = np.asarray(p_a, dtype=np.float64)
        kth = np.partition(p, n_b - 1)[n_b - 1]
        candidates = np.flatnonzero(p <= kth)  # holds the first n_b of the full ranking, ties at kth included
        ranked = candidates[np.lexsort((-np.asarray(abs_a, dtype=np.float64)[candidates], p[candidates]))][:n_b]
        out["equal_count_shared"] = int(b[ranked].sum()) / n_b
        out["equal_count_p"] = float(p[ranked[-1]])
    return out


def section_b(ds: DatasetScores) -> list[dict]:
    """Calls under each null; the pair's overlap; each model across two nulls."""
    rows = []

    def add(null, view, model, n, measure, value):
        rows.append({"dataset": ds.name, "null": null, "view": view, "model": model, "n": n, "measure": measure,
                     "value": value})  # fmt: skip

    snv = ds.view.kind == SNV
    for null, nr in ds.nulls.items():
        null_snv = nr.kind == SNV
        views = [("all", None), ("SNV", snv)]
        if null_snv.any() and not null_snv.all():
            views.append(("SNV_vs_SNV_rows", snv))
        for view, mask in views:
            mean_p: dict[str, np.ndarray] = {}
            for model in MODELS:
                if (null, model) not in ds.pval:
                    continue
                if view == "SNV_vs_SNV_rows":
                    obs = ds.var[model]["abs_logfc"][:, mask]
                    bg = ds.null[(null, model)]["abs_logfc"][:, null_snv]
                    per_fold = np.stack([sc.get_pvals(obs[j], bg[j], "right") for j in range(len(ds.folds))])
                    fold_mean = sc.geo_mean(per_fold)
                else:
                    per_fold = ds.pval[(null, model)] if mask is None else ds.pval[(null, model)][:, mask]
                    fold_mean = ds.summary[(null, model)]["abs_logfc.mean.pval"]
                    fold_mean = fold_mean if mask is None else fold_mean[mask]
                calls = called(fold_mean)
                n = int(calls.size)
                shares = called(per_fold).mean(axis=1) if n else np.full(len(ds.folds), math.nan)
                add(null, view, model, n, "per_fold_share", float(shares.mean()))
                add(null, view, model, n, "per_fold_share_min", float(shares.min()))
                add(null, view, model, n, "per_fold_share_max", float(shares.max()))
                add(null, view, model, n, "fold_mean_calls", int(calls.sum()))
                add(null, view, model, n, "fold_mean_share", float(calls.mean()) if n else math.nan)
                mean_p[model] = np.asarray(fold_mean, dtype=np.float64)
            if all(m in mean_p for m in PAIR):
                abs_fc = np.abs(np.asarray(ds.summary[(null, PAIR[0])]["logfc.mean"], dtype=np.float64))
                abs_fc = abs_fc if mask is None else abs_fc[mask]
                n = len(mean_p[PAIR[0]])
                for measure, value in pair_overlap(mean_p[PAIR[0]], mean_p[PAIR[1]], abs_fc).items():
                    add(null, view, PAIR_NAME, n, measure, value)
    names = list(ds.nulls)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            for model in MODELS:
                if (a, model) in ds.summary and (b, model) in ds.summary:
                    in_a = called(ds.summary[(a, model)]["abs_logfc.mean.pval"])
                    in_b = called(ds.summary[(b, model)]["abs_logfc.mean.pval"])
                    n = len(in_a)
                    add(f"{a}+{b}", "all", model, n, f"calls_{a}", int(in_a.sum()))
                    add(f"{a}+{b}", "all", model, n, f"calls_{b}", int(in_b.sum()))
                    add(f"{a}+{b}", "all", model, n, "both", int((in_a & in_b).sum()))
    return rows


def section_c(ds: DatasetScores) -> list[dict]:
    """The p of a fixed |logFC| against each model's null, folds pooled."""
    rows = []
    for null, nr in ds.nulls.items():
        for subset, mask in (("all", np.ones(len(nr.kind), dtype=bool)), ("SNV", nr.kind == SNV)):
            for model in MODELS:
                d = ds.null.get((null, model))
                if d is None:
                    continue
                pooled = np.asarray(d["abs_logfc"][:, mask], dtype=np.float64).ravel()
                p = sc.get_pvals(np.asarray(FIXED), pooled, "right") if pooled.size else [math.nan] * len(FIXED)
                rows.append({"dataset": ds.name, "null": null, "subset": subset, "model": model, "n": int(pooled.size),
                             **{f"p_{x:g}": float(v) for x, v in zip(FIXED, p)}})  # fmt: skip
    return rows


def section_d(ds: DatasetScores) -> list[dict]:
    """|logFC| q99 of the SNV null rows by CpG bin; the nulls without src_id reweighted to the first with one."""
    rows = []
    names = cpg_bin_names()
    have = {null: nr for null, nr in ds.nulls.items() if nr.cpg is not None}
    reference = next((null for null, nr in have.items() if nr.from_list), None)
    ref_share = None
    if reference is not None:
        ref = have[reference]
        ref_counts = np.bincount(cpg_bins(ref.cpg[ref.kind == SNV]), minlength=len(names))
        ref_share = ref_counts / max(int(ref_counts.sum()), 1)
    for null, nr in have.items():
        snv = nr.kind == SNV
        cpg = nr.cpg[snv]
        bins = cpg_bins(cpg)
        counts = np.bincount(bins, minlength=len(names))
        total = int(snv.sum())
        for model in MODELS:
            d = ds.null.get((null, model))
            if d is None:
                continue
            values = np.asarray(d["abs_logfc"][:, snv], dtype=np.float64)  # [folds, SNV rows]
            base = {"dataset": ds.name, "null": null, "model": model}
            rows.append(base | {"cpg_bin": "all", "rows": total, "share": 1.0 if total else math.nan,
                                "q99": quantile(values, 0.99),
                                "cpg_mean": float(cpg.mean()) if total else math.nan,
                                "cpg_median": float(np.median(cpg)) if total else math.nan})  # fmt: skip
            for b, name in enumerate(names):
                rows.append(base | {"cpg_bin": name, "rows": int(counts[b]),
                                    "share": counts[b] / total if total else math.nan,
                                    "q99": quantile(values[:, bins == b], 0.99)})  # fmt: skip
            if ref_share is not None and not nr.from_list:
                own_share = counts / max(total, 1)
                weight = np.where(counts > 0, ref_share / np.where(counts > 0, own_share, 1.0), 0.0)
                rows.append(base | {"cpg_bin": "reweighted", "rows": total, "share": 1.0 if total else math.nan,
                                    "q99": weighted_quantile(values, np.broadcast_to(weight[bins], values.shape), 0.99),
                                    "reference": reference, "max_weight": float(weight.max()),
                                    "uncovered_share": float(ref_share[counts == 0].sum())})  # fmt: skip
    return rows


def section_e(ds: DatasetScores) -> list[dict]:
    """The pair on the same rows, per fold: Pearson r of logFC and the top-share by |logFC|."""
    a, b = PAIR
    sets = []
    for null, nr in ds.nulls.items():
        if (null, a) in ds.null and (null, b) in ds.null:
            sets.append((null, ds.null[(null, a)]["logfc"], ds.null[(null, b)]["logfc"], nr.kind == SNV))
    if a in ds.var and b in ds.var:
        sets.append(("variants", ds.var[a]["logfc"], ds.var[b]["logfc"], ds.view.kind == SNV))
    rows = []
    for name, x, y, snv in sets:
        for subset, mask in (("all", None), ("SNV", snv)):
            rs, tops = [], []
            for j in range(x.shape[0]):
                xj, yj = (x[j], y[j]) if mask is None else (x[j][mask], y[j][mask])
                rs.append(pearson(xj, yj))
                tops.append(top_overlap(xj, yj, TOP))
            rows.append({"dataset": ds.name, "set": name, "subset": subset,
                         "rows": int(x.shape[1] if mask is None else mask.sum()),
                         "r_mean": _mean(rs), "r_min": _min(rs), "r_max": _max(rs),
                         "top_mean": _mean(tops), "top_min": _min(tops), "top_max": _max(tops)})  # fmt: skip
    return rows


def heldout_row(
    ds: DatasetScores, name: str, model: str, values: np.ndarray, snv: np.ndarray, fold: np.ndarray
) -> dict:
    """F for one model and row set: per fold k, the q99 in fold k of the SNV rows on fold k's test chromosomes."""
    per, sizes, everything = [], [], []
    for j, k in enumerate(ds.folds):
        sel = snv & (fold == k)
        sizes.append(int(sel.sum()))
        per.append(quantile(values[j][sel], 0.99))
        everything.append(quantile(values[j][snv], 0.99))
    return {"dataset": ds.name, "set": name, "subset": "SNV", "model": model, "q99_heldout": _mean(per),
            "q99_all": _mean(everything), "rows_min": min(sizes), "rows_max": max(sizes), "q99_fold_min": _min(per),
            "q99_fold_max": _max(per), "per_fold": ",".join(f"{v:.4g}" for v in per)}  # fmt: skip


def section_f(ds: DatasetScores) -> list[dict]:
    """Held-out SNV rows only, each null and the variants."""
    rows = []
    for null, nr in ds.nulls.items():
        for model in MODELS:
            d = ds.null.get((null, model))
            if d is not None:
                rows.append(heldout_row(ds, null, model, d["abs_logfc"], nr.kind == SNV, nr.fold))
    for model in MODELS:
        if model in ds.var:
            rows.append(heldout_row(ds, "variants", model, ds.var[model]["abs_logfc"], ds.view.kind == SNV,
                                    ds.view.fold_test))  # fmt: skip
    full = {r["set"]: r["q99_heldout"] for r in rows if r["model"] == "cbp_full"}
    for r in rows:
        r["q99_heldout_vs_cbp_full"] = _ratio(r["q99_heldout"], full.get(r["set"], math.nan))
    return rows


def section_g(ds: DatasetScores) -> list[dict]:
    """|logFC| q99 by allele-1 log-count quintile: each null's SNV rows and the SNV variants, folds pooled."""
    rows = []
    for model in MODELS:
        groups = []
        for null, nr in ds.nulls.items():
            d = ds.null.get((null, model))
            if d is not None:
                snv = nr.kind == SNV
                groups.append((null, d["allele1_logcounts"][:, snv].ravel(), d["abs_logfc"][:, snv].ravel()))
        if model in ds.var:
            snv = ds.view.kind == SNV
            d = ds.var[model]
            groups.append(("variants", d["allele1_logcounts"][:, snv].ravel(), d["abs_logfc"][:, snv].ravel()))
        if not groups:
            continue
        pooled = np.concatenate([g[1] for g in groups]).astype(np.float64)
        if not pooled.size:
            continue
        edges = np.quantile(pooled, QUINTILES)
        bounds = [float(pooled.min()), *(float(e) for e in edges), float(pooled.max())]
        del pooled
        for name, logc, values in groups:
            which = np.searchsorted(edges, np.asarray(logc, dtype=np.float64), side="right")  # lo <= x < hi
            for i in range(len(QUINTILES) + 1):
                sel = which == i
                rows.append({"dataset": ds.name, "model": model, "group": name, "quintile": i + 1,
                             "logc_lo": bounds[i], "logc_hi": bounds[i + 1], "n": int(sel.sum()),
                             "q99": quantile(values[sel], 0.99)})  # fmt: skip
    return rows


def section_h(ds: DatasetScores) -> list[dict]:
    """Effect size and calls by NFE MAF bin, within mut_class and stratum; the MAF bins of the pair's calls."""
    v = ds.view
    if v.maf_bin is None:
        return []
    rows = []

    def add(part, null, model, group, mut_class, stratum, maf, measure, value):
        rows.append({"dataset": ds.name, "part": part, "null": null, "model": model, "group": group,
                     "mut_class": mut_class, "stratum": stratum, "maf_bin": maf, "measure": measure,
                     "value": value})  # fmt: skip

    classes = [(name, v.mut_class == i) for i, name in enumerate(v.classes)]
    classes = [("all", None)] + [(name, m) for name, m in classes if m.any()]
    strata = [(name, v.stratum == i) for i, name in enumerate(v.strata)]
    strata = [("all", None)] + [(name, m) for name, m in strata if m.any()]
    bins = [("all", None)] + [(name, v.maf_bin == i) for i, name in enumerate(MAF_BINS)]
    models = [m for m in MODELS if any((null, m) in ds.summary for null in ds.nulls)]
    abs_fc = {}
    for model in models:
        null = next(n for n in ds.nulls if (n, model) in ds.summary)
        abs_fc[model] = np.abs(np.asarray(ds.summary[(null, model)]["logfc.mean"], dtype=np.float64))
    calls = {key: called(s["abs_logfc.mean.pval"]) for key, s in ds.summary.items()}

    def both(m1, m2):
        return m2 if m1 is None else m1 if m2 is None else m1 & m2

    for c_name, c_mask in classes:
        for s_name, s_mask in strata:
            cs_mask = both(c_mask, s_mask)
            for b_name, b_mask in bins:
                mask = both(cs_mask, b_mask)
                n = v.n if mask is None else int(mask.sum())
                if not n:
                    continue
                for model in models:
                    x = abs_fc[model] if mask is None else abs_fc[model][mask]
                    q50, q90, q99 = np.quantile(x, (0.5, 0.9, 0.99))
                    cell = ("effect", "", model, "", c_name, s_name, b_name)
                    add(*cell, "n", n)
                    add(*cell, "abs_mean_logfc_q50", float(q50))
                    add(*cell, "abs_mean_logfc_q90", float(q90))
                    add(*cell, "abs_mean_logfc_q99", float(q99))
                    for null in ds.nulls:
                        if (null, model) in calls:
                            c = calls[(null, model)] if mask is None else calls[(null, model)][mask]
                            add("effect", null, model, "", c_name, s_name, b_name, "share_called", float(c.mean()))
    first, second = PAIR
    everything = np.ones(v.n, dtype=bool)
    for null in ds.nulls:
        if (null, first) not in calls or (null, second) not in calls:
            continue
        in_a, in_b = calls[(null, first)], calls[(null, second)]
        groups = (("all", everything), ("both", in_a & in_b), (f"{first}_only", in_a & ~in_b),
                  (f"{second}_only", in_b & ~in_a))  # fmt: skip
        for c_name, c_mask in classes:
            for g_name, g_mask in groups:
                sel = g_mask if c_mask is None else g_mask & c_mask
                n = int(sel.sum())
                counts = np.bincount(v.maf_bin[sel], minlength=len(MAF_BINS))
                maf = v.maf[sel]
                maf = maf[np.isfinite(maf)]
                cell = ("calls", null, PAIR_NAME, g_name, c_name, "all")
                add(*cell, "all", "n", n)
                add(*cell, "all", "mean_maf", float(maf.mean()) if maf.size else math.nan)
                for i, b_name in enumerate(MAF_BINS):
                    add(*cell, b_name, "n", int(counts[i]))
                    add(*cell, b_name, "share", counts[i] / n if n else math.nan)
    return rows


def mean_q99(values: np.ndarray, rows: np.ndarray) -> float:
    """The mean over folds of the per-fold |logFC| q99 of the given rows (values: [folds, rows]), as A computes q99."""
    sub = np.asarray(values[:, rows], dtype=np.float64)
    return float(np.quantile(sub, 0.99, axis=1).mean()) if sub.size else math.nan


def ratio_draws(x: np.ndarray, y: np.ndarray, snv: np.ndarray, ti: np.ndarray, reps: int, seed: int) -> dict:
    """R for one null: the q99 ratio of x over y (each [folds, rows] of |logFC|) on its SNV rows and on their Ti and
    Tv rows, as {subset: {"point": the ratio on all rows, "draws": [reps] ratios}}. A replicate draws the SNV rows
    again with replacement (default_rng(seed).integers), once for both models and every fold; its Ti and Tv rows are
    the drawn rows of each. A subset a replicate draws no row of gives NaN."""
    rows = np.flatnonzero(snv)
    is_ti = np.asarray(ti, dtype=bool)[rows]
    picks = {"SNV": None, "SNV_Ti": is_ti, "SNV_Tv": ~is_ti}
    out = {}
    for name, keep in picks.items():
        sel = rows if keep is None else rows[keep]
        out[name] = {"point": _ratio(mean_q99(x, sel), mean_q99(y, sel)), "draws": np.full(reps, math.nan)}
    if not len(rows):
        return out
    rng = np.random.default_rng(seed)
    for r in range(reps):
        drawn = rng.integers(0, len(rows), len(rows))
        for name, keep in picks.items():
            sel = rows[drawn] if keep is None else rows[drawn[keep[drawn]]]
            out[name]["draws"][r] = _ratio(mean_q99(x, sel), mean_q99(y, sel))
    return out


def spread(draws) -> dict:
    """lo and hi (the 2.5 and 97.5 % quantiles), sd and reps of the finite replicates."""
    values = np.asarray(draws, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if not n:
        return {"lo": math.nan, "hi": math.nan, "sd": math.nan, "reps": 0}
    lo, hi = (float(v) for v in np.quantile(values, (0.025, 0.975)))
    listed = values.tolist()
    mean = sum(listed) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in listed) / (n - 1)) if n > 1 else math.nan
    return {"lo": lo, "hi": hi, "sd": sd, "reps": n}


def section_r(ds: DatasetScores, reps: int = BOOTSTRAP, seed: int = BOOTSTRAP_SEED) -> list[dict]:
    """The decision numbers with the noise of the null rows' draw (module docstring)."""
    a, b = PAIR
    shuffled = next((null for null, nr in ds.nulls.items() if nr.from_list), None)
    genomic = next((null for null, nr in ds.nulls.items() if not nr.from_list), None)
    variants = math.nan
    if a in ds.var and b in ds.var and "summit100" in ds.view.strata:
        sel = np.flatnonzero((ds.view.stratum == ds.view.strata.index("summit100")) & (ds.view.kind == SNV))
        variants = _ratio(mean_q99(ds.var[a]["abs_logfc"], sel), mean_q99(ds.var[b]["abs_logfc"], sel))
    rows = [{"dataset": ds.name, "measure": "q99_ratio", "null": "variants", "subset": "summit100_SNV",
             "point": variants, "reps": 0}]  # fmt: skip
    snv: dict[str, dict] = {}
    for null in (shuffled, genomic):
        if null is None or (null, a) not in ds.null or (null, b) not in ds.null:
            continue
        nr = ds.nulls[null]
        got = ratio_draws(ds.null[(null, a)]["abs_logfc"], ds.null[(null, b)]["abs_logfc"], nr.kind == SNV, nr.ti,
                          reps, seed)  # fmt: skip
        for subset, d in got.items():
            rows.append({"dataset": ds.name, "measure": "q99_ratio", "null": null, "subset": subset,
                         "point": d["point"], **spread(d["draws"])})  # fmt: skip
        snv[null] = got["SNV"]
    if shuffled in snv and genomic in snv:
        s, g = snv[shuffled], snv[genomic]
        both = f"{shuffled}+{genomic}"
        outcomes = [decide(sv, gv, variants) for sv, gv in zip(s["draws"].tolist(), g["draws"].tolist())]
        rows.append({"dataset": ds.name, "measure": "excess_removed", "null": both, "subset": "SNV",
                     "point": decide(s["point"], g["point"], variants)[1],
                     **spread([removed for _, removed in outcomes])})  # fmt: skip
        counts = Counter(branch for branch, _ in outcomes)
        for branch in (*BRANCHES, *sorted(set(counts) - set(BRANCHES))):
            rows.append({"dataset": ds.name, "measure": "branch_share", "null": both, "subset": branch,
                         "point": counts[branch] / len(outcomes) if outcomes else math.nan,
                         "reps": len(outcomes)})  # fmt: skip
    return rows


SECTIONS = {"A": section_a, "B": section_b, "C": section_c, "D": section_d, "E": section_e, "F": section_f,
            "G": section_g, "H": section_h, "R": section_r}  # fmt: skip


# --- outputs ----------------------------------------------------------------------------------------------------


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_text(path: Path, text: str) -> None:
    """Write under a temporary name, then rename."""
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text)
    os.replace(partial, path)


def write_tsv(path: Path, columns, rows: list[dict]) -> None:
    """A TSV with a header: floats with 6 significant digits (nan when missing), an absent value empty."""
    lines = ["\t".join(columns)] + ["\t".join(_cell(row.get(c)) for c in columns) for row in rows]
    write_text(path, "\n".join(lines) + "\n")


def _f(value, digits: int = 3) -> str:
    """A number for a table cell; a dash when it is missing."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    return f"{value:.{digits}f}"


def _g(value, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    return f"{value:.{digits}g}"


def _pct(value, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    return f"{100 * value:.{digits}f} %"


def _int(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    return f"{int(value):,}"


def _table(header: list[str], body: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return lines + ["| " + " | ".join(row) + " |" for row in body] + [""]


def _now_et() -> str:
    """Now, in US Eastern time (the cluster's clock runs 6 h ahead of it)."""
    try:
        return datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M ET")
    except ZoneInfoNotFoundError:  # no time zone database
        return datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")


def _q99(row: dict | None, n_key: str = "n", folds: int = 1) -> float:
    """A row's q99, or NaN when it is missing or rests on fewer than MIN_ROWS rows (folds pooled)."""
    if row is None or row.get(n_key, 0) * folds < MIN_ROWS:
        return math.nan
    return row["q99"]


def glossary(models) -> list[str]:
    """report.md's header bullet on what the ChromBPNet labels mean, for the ChromBPNet models among `models` only
    (none without one), as compare.py's."""
    named = [m for m in GLOSSARY if m in models]
    if not named:
        return []
    text = " ".join(f"{label(m)}: {GLOSSARY[m]}." for m in named)
    if "cherimoya" in models and "cbp_full" in models:
        text += " Cherimoya has no bias model and, like the uncorrected model, predicts the raw signal."
    elif "cherimoya" in models:
        text += " Cherimoya has no bias model and predicts the raw signal, Tn5 bias included."
    return [f"- {text}"]


def build_report(args, vl: VariantList, nulls: list[str], cache: dict, done: list[str], rows: dict,
                 notes: Notes) -> str:  # fmt: skip
    """report.md: the rule's outcome per dataset first, then compact tables of A to H."""
    kinds = {null: cache[null].from_list for null in nulls if null in cache}
    shuffled = next((n for n in nulls if kinds.get(n) is True), None)
    genomic = next((n for n in nulls if kinds.get(n) is False), None)
    folds = len(args.folds)
    A = {(r["dataset"], r["null"], r["subset"], r["model"]): r for r in rows["A"]}
    B = {(r["dataset"], r["null"], r["view"], r["model"], r["measure"]): r["value"] for r in rows["B"]}
    C = {(r["dataset"], r["null"], r["subset"], r["model"]): r for r in rows["C"]}
    D = {(r["dataset"], r["null"], r["model"], r["cpg_bin"]): r for r in rows["D"]}
    E = {(r["dataset"], r["set"], r["subset"]): r for r in rows["E"]}
    F = {(r["dataset"], r["set"], r["model"]): r for r in rows["F"]}
    G = {(r["dataset"], r["model"], r["group"], r["quintile"]): r for r in rows["G"]}
    H = {(r["dataset"], r["part"], r["null"], r["model"], r["group"], r["mut_class"], r["stratum"], r["maf_bin"],
          r["measure"]): r["value"] for r in rows["H"]}  # fmt: skip
    R = {(r["dataset"], r["measure"], r["null"], r["subset"]): r for r in rows["R"]}
    both = f"{shuffled}+{genomic}"
    pair_names = " / ".join(label(m) for m in PAIR)  # Cherimoya / ChromBPNet uncorrected, in headers

    def ratio(ds, null, subset):
        r = A.get((ds, null, subset, "cherimoya"))
        return r["q99_vs_cbp_full"] if r else math.nan

    def interval(ds, measure, null, subset, fmt=_f):
        r = R.get((ds, measure, null, subset))
        return f"[{fmt(r['lo'])}-{fmt(r['hi'])}]" if r and r.get("reps") else ""

    def shares(ds):
        got = [(R[(ds, "branch_share", both, b)]["point"], b) for b in (*BRANCHES, "not evaluable")
               if (ds, "branch_share", both, b) in R]  # fmt: skip
        got = sorted(((v, b) for v, b in got if v > 0), reverse=True)
        return ", ".join(f"{_pct(v, 0)} {b}" for v, b in got)

    def calls(ds, null):
        c, f = (B.get((ds, null, "all", m, "fold_mean_calls")) for m in PAIR)
        if c is None or f is None:
            return "–"
        return f"{c:,} / {f:,} ({_f(_ratio(c, f), 2)})"

    n_kind = Counter(int(k) for k in vl.kind.tolist())
    sizes = ", ".join(f"{null_label(n)} (`{n}`, {len(cache[n].ids):,} rows, `{scores_dir(args.root, n).name}/`)"
                      for n in nulls if n in cache)  # fmt: skip
    lines = [
        "# Two nulls: what each model's null looks like, and what it calls",
        "",
        f"- Generated {_now_et()} by `workflows/nulls/null_report.py`: {len(done)} of "
        f"{len(args.datasets)} datasets" + ("." if len(done) == len(args.datasets) else " (partial)."),
        f"- Root `{args.root}`; variant list `{args.list}`: {len(vl.ids):,} variants ({n_kind[SNV]:,} SNVs, "
        f"{n_kind[INDEL]:,} indels, {n_kind[MNV]:,} MNVs); window set `{args.variants}`; folds "
        f"{', '.join(map(str, args.folds))}.",
        f"- Nulls: {sizes or 'none read'}.",
        *glossary({r["model"] for r in rows["A"]}),
        f"- logFC is the log2 ratio of ALT to REF strand-averaged predicted counts. A call is p < {ALPHA:g}, VS's "
        "right-tailed empirical p of |logFC| against the fold's null rows, (rank + 1) / (n + 1); fold-mean calls use "
        "the summary's geometric mean of the per-fold p. q99 is the mean of the per-fold 99th percentiles of |logFC| "
        f"unless a table says pooled. A dash: missing, or a quantile over fewer than {MIN_ROWS} rows (folds pooled).",
        "- Per dataset and section in full: A_widths.tsv to H_frequency.tsv and R_rule_bootstrap.tsv beside this file.",
        "",
    ]
    for text in notes.notes:
        lines.append(f"- Note: {text}")
    for text in notes.problems:
        lines.append(f"- **Problem:** {text}")
    if notes.notes or notes.problems:
        lines.append("")

    # --- the decision
    lines += ["## Decision rule (pre-registered)", ""] + [f"> {text}" for text in RULE] + [""]
    lines += [f"In this report's labels, the rule's ChromBPNet full is {label('cbp_full')}.", ""]
    lines += [
        "Ratios: Cherimoya's |logFC| q99 over that of ChromBPNet uncorrected on the same rows (A). The branch follows "
        f"the genomic null's ratio: at most {ARTEFACT_MAX}, null artefact; at least {PROPERTY_MIN}, model property; "
        "in between, partly both. Excess removed: (shuffled - genomic) / (shuffled - variants), the share of the "
        "shuffled null's extra Cherimoya width over the variants' ratio (summit100 SNVs) that real sequence takes "
        "away. Calls: fold-mean calls of all variants, Cherimoya / ChromBPNet uncorrected (their ratio). In brackets, "
        f"the bootstrap's 95 % interval ({args.bootstrap} replicates of each null's SNV rows, R_rule_bootstrap.tsv), "
        "and after the branch the share of replicates in each branch.",
        "",
    ]
    body, branches, high_anchor = [], Counter(), []
    for ds in done:
        var_all, var_snv = ratio(ds, "variants", "summit100"), ratio(ds, "variants", "summit100_SNV")
        s = [ratio(ds, shuffled, x) for x in ("SNV", "SNV_Ti", "SNV_Tv")]
        g = [ratio(ds, genomic, x) for x in ("SNV", "SNV_Ti", "SNV_Tv")]
        branch, removed = decide(s[0], g[0], var_snv)
        branches[branch] += 1
        if var_snv > ARTEFACT_MAX:
            high_anchor.append(ds)
        boot = shares(ds)
        body.append([ds, _f(var_all), _f(var_snv),
                     f"{_f(s[0])} {interval(ds, 'q99_ratio', shuffled, 'SNV')} ({_f(s[1])} / {_f(s[2])})",
                     f"{_f(g[0])} {interval(ds, 'q99_ratio', genomic, 'SNV')} ({_f(g[1])} / {_f(g[2])})",
                     f"**{branch}**" + (f" (bootstrap: {boot})" if boot else ""),
                     f"{_pct(removed, 0)} {interval(ds, 'excess_removed', both, 'SNV', lambda v: _pct(v, 0))}",
                     calls(ds, shuffled), calls(ds, genomic)])  # fmt: skip
    lines += _table(["dataset", "variants summit100", "variants summit100 SNV", "shuffled SNV rows (Ti / Tv)",
                     "genomic (Ti / Tv)", "branch", "excess removed", "calls, shuffled null",
                     "calls, genomic null"], body)  # fmt: skip
    lines += [f"Branches: {', '.join(f'{k} in {v}' for k, v in branches.most_common()) or 'none'} of {len(done)} "
              "datasets. Every dataset is scored against the same null rows (one shuffled draw, one genomic draw), so "
              "a branch that every dataset shares does not confirm those draws independently; the intervals show how "
              "far the draws alone move each number.", ""]  # fmt: skip
    body = []
    for ds in done:
        cells = [f"{_f(R[k]['point'])} {interval(ds, 'q99_ratio', k[2], k[3])}" if k in R else "–"
                 for k in ((ds, "q99_ratio", null, subset) for null in (shuffled, genomic)
                           for subset in ("SNV", "SNV_Ti", "SNV_Tv"))]  # fmt: skip
        if any(c != "–" for c in cells):
            body.append([ds, *cells])
    if body:
        lines += [f"Bootstrap intervals (95 %, {args.bootstrap} replicates) of the q99 ratios, SNV rows and their Ti "
                  "and Tv rows:", ""]  # fmt: skip
        lines += _table(["dataset", "shuffled SNV", "shuffled Ti", "shuffled Tv", "genomic SNV", "genomic Ti",
                         "genomic Tv"], body)  # fmt: skip
    if high_anchor:
        lines += [f"The variants' own ratio is above {ARTEFACT_MAX} in {', '.join(high_anchor)}: the rule was written "
                  "with the variants at 1.07-1.10, so its thresholds sit closer to the variants there.",
                  ""]  # fmt: skip

    # --- A
    lines += ["## A. Null widths: |logFC| q99", ""]
    body = []
    for ds in done:
        sets = []
        for null in nulls:
            subsets = ("all", "SNV", "indel") if kinds.get(null) else ("all",)
            sets += [(null, subset, f"{null_label(null)}, {subset}") for subset in subsets]
        sets.append(("variants", "summit100_SNV", "variants, summit100 SNVs"))
        for null, subset, name in sets:
            got = [A.get((ds, null, subset, m)) for m in WIDTH_MODELS]
            if not any(got):
                continue
            n = next(r["n"] for r in got if r)
            body.append([ds, name, _int(n), *(_f(r["q99"]) if r else "–" for r in got),
                         _f(ratio(ds, null, subset))])  # fmt: skip
    lines += _table(["dataset", "rows", "n", *(label(m) for m in WIDTH_MODELS), pair_names], body)

    # --- B
    lines += [
        f"## B. Calls at p < {ALPHA:g}",
        "",
        f"Per-fold share: the mean over folds of the share of variants with p < {ALPHA:g} in that fold. Fold-mean "
        f"calls: the summary's p < {ALPHA:g}. View SNV_vs_SNV_rows: the SNVs against the null's SNV rows only (p "
        "recomputed). "
        "Both: called by Cherimoya and ChromBPNet uncorrected. Equal count: the share in common when Cherimoya calls "
        "as many as ChromBPNet uncorrected.",
        "",
    ]
    body = []
    trio = " / ".join(short(m) for m in MODELS)
    for ds in done:
        for null in nulls:
            for view in ("all", "SNV", "SNV_vs_SNV_rows"):
                share = [B.get((ds, null, view, m, "per_fold_share")) for m in MODELS]
                count = [B.get((ds, null, view, m, "fold_mean_calls")) for m in MODELS]
                if all(x is None for x in count):
                    continue
                n = next((r["n"] for r in rows["B"] if (r["dataset"], r["null"], r["view"]) == (ds, null, view)), None)
                pair = {k: B.get((ds, null, view, PAIR_NAME, k)) for k in
                        ("both", "share_of_cherimoya", "share_of_cbp_full", "equal_count_shared")}  # fmt: skip
                body.append([ds, null_label(null), view, _int(n), " / ".join(_pct(x, 2) for x in share),
                             " / ".join(_int(x) for x in count), _int(pair["both"]),
                             _pct(pair["share_of_cherimoya"], 0), _pct(pair["share_of_cbp_full"], 0),
                             _pct(pair["equal_count_shared"], 0)])  # fmt: skip
    lines += _table(["dataset", "null", "view", "variants", f"per-fold share {trio}", f"fold-mean calls {trio}",
                     "both", "of Cherimoya's, also the uncorrected model's",
                     "of the uncorrected model's, also Cherimoya's", "equal count"], body)  # fmt: skip
    lines += ["Uncorrected and corrected: ChromBPNet uncorrected and ChromBPNet corrected.", ""]

    # --- C
    lines += [f"## C. p of a fixed |logFC| ({' / '.join(f'{x:g}' for x in FIXED)}; fold change "
              f"{' / '.join(f'{2**x:.2f}' for x in FIXED)}), folds pooled, all rows", ""]  # fmt: skip
    body = []
    for ds in done:
        for null in nulls:
            got = [C.get((ds, null, "all", m)) for m in PAIR]
            if any(got):
                body.append([ds, null_label(null), *(" / ".join(_g(r[f"p_{x:g}"]) for x in FIXED) if r else "–"
                                                     for r in got)])  # fmt: skip
    lines += _table(["dataset", "null", *(label(m) for m in PAIR)], body)

    # --- D
    names = cpg_bin_names()
    lines += ["## D. CpG", "", "CpG count of the allele-1 window, SNV rows:", ""]
    body = []
    for null in nulls:
        r = next((D[k] for k in D if k[1] == null and k[3] == "all"), None)
        if r is None:
            continue
        ds0, model0 = next(k for k in D if k[1] == null and k[3] == "all")[0::2]
        bins = [D.get((ds0, null, model0, name)) for name in names]
        body.append([null_label(null), _int(r["rows"]), _f(r["cpg_mean"], 1), _f(r["cpg_median"], 0),
                     *(_pct(b["share"], 0) if b else "–" for b in bins)])  # fmt: skip
    lines += _table(["null", "SNV rows", "mean", "median", *names], body)
    if genomic and shuffled:
        lines += ["|logFC| q99 of the SNV rows, folds pooled, Cherimoya / ChromBPNet uncorrected (ratio); the genomic "
                  "null reweighted to the shuffled null's CpG-bin mix. The reweighted q99 is the smallest |logFC| "
                  "whose weight reaches 99 %, the others numpy's interpolated quantile: the two rules can differ by "
                  "one order statistic (about 0.1 % of q99 at these sizes).", ""]  # fmt: skip
        body = []
        for ds in done:
            cells = []
            for null, bin_name in ((shuffled, "all"), (genomic, "all"), (genomic, "reweighted")):
                q = [_q99(D.get((ds, null, m, bin_name)), "rows", folds) for m in PAIR]
                cells.append(f"{_f(q[0])} / {_f(q[1])} ({_f(_ratio(q[0], q[1]), 2)})")
            rw = D.get((ds, genomic, PAIR[0], "reweighted"))
            body.append([ds, *cells, _f(rw["max_weight"], 1) if rw else "–",
                         _pct(rw["uncovered_share"], 1) if rw else "–"])  # fmt: skip
        lines += _table(["dataset", "shuffled SNV rows", "genomic", "genomic reweighted", "max weight",
                         "shuffled rows in bins the genomic null lacks"], body)  # fmt: skip

    # --- E
    sets = [("variants", "all"), ("variants", "SNV")]
    for null in nulls:
        sets += [(null, "all")] + ([(null, "SNV")] if kinds.get(null) else [])
    set_names = [("variants" if s == "variants" else null_label(s)) + ("" if sub == "all" else f" {sub}")
                 for s, sub in sets]  # fmt: skip
    lines += ["## E. Cherimoya against ChromBPNet uncorrected on the same rows (mean over folds)", ""]
    body = []
    for ds in done:
        got = [E.get((ds, s, sub)) for s, sub in sets]
        body.append([ds, " / ".join(_f(r["r_mean"]) if r else "–" for r in got),
                     " / ".join(_pct(r["top_mean"], 0) if r else "–" for r in got)])  # fmt: skip
    lines += _table(["dataset", f"Pearson r of logFC: {' / '.join(set_names)}", f"top {TOP:.0%} shared"], body)

    # --- F
    lines += ["## F. Held-out SNV rows: |logFC| q99 on each fold's test chromosomes, against all SNV rows", ""]
    body = []
    for ds in done:
        for null in [*nulls, "variants"]:
            got = [F.get((ds, null, m)) for m in PAIR]
            if not all(got):
                continue
            body.append([ds, "variants" if null == "variants" else null_label(null),
                         *(f"{_f(r['q99_heldout'])} / {_f(r['q99_all'])}" for r in got),
                         f"{_f(_ratio(got[0]['q99_heldout'], got[1]['q99_heldout']))} / "
                         f"{_f(_ratio(got[0]['q99_all'], got[1]['q99_all']))}",
                         f"{_int(got[0]['rows_min'])}-{_int(got[0]['rows_max'])}"])  # fmt: skip
    lines += _table(["dataset", "rows", *(f"{label(m)}: held-out / all" for m in PAIR), "ratio: held-out / all",
                     "held-out rows per fold"], body)  # fmt: skip

    # --- G
    lines += ["## G. |logFC| q99 by allele-1 log-count quintile (SNV rows, folds pooled; median over datasets)", "",
              "The quintile edges are per model, from the nulls' SNV rows and the SNV variants pooled. The variants "
              "are most of that pool, so the edges are theirs, and the nulls' rows (lower predicted counts) crowd "
              f"into the lower quintiles; an upper cell of a null can rest on few rows (a dash under {MIN_ROWS}).",
              ""]  # fmt: skip
    body = []
    for model in PAIR:
        for group in [*nulls, "variants"]:
            cells = []
            for q in range(1, len(QUINTILES) + 2):
                got = [G.get((ds, model, group, q)) for ds in done]
                got = [r for r in got if r]
                if not got:
                    cells.append("–")
                    continue
                cells.append(f"{_f(_median([_q99(r) for r in got]))} (n {_int(_median([r['n'] for r in got]))})")
            if any(c != "–" for c in cells):
                body.append([label(model), "variants SNV" if group == "variants" else null_label(group), *cells])
    lines += _table(["model", "rows", *(f"Q{q}" for q in range(1, len(QUINTILES) + 2))], body)

    # --- H
    lines += [
        "## H. Frequency: NFE minor allele frequency (median over datasets; stratum all)",
        "",
        "Purifying selection is expected to make large effects rarer: variants with larger predicted effects should "
        "sit at lower frequency, and calls that sit at higher frequency look less constrained. CpG transitions mutate "
        "far more often, so at the same constraint they reach higher frequencies; compare frequencies within a "
        "mut_class. MAF bins: 0.00 (below 0.5 % in NFE, common elsewhere), 0.01-0.04, >= 0.05, empty (no numeric "
        "AF_nfe). |logFC| here is that of the fold-mean logFC.",
        "",
    ]
    present = {k[5] for k in H}
    classes = [c for c in ("all", *vl.classes) if c in present]

    def h_med(part, null, model, group, mut_class, stratum, maf, measure):
        return _median([H.get((ds, part, null, model, group, mut_class, stratum, maf, measure)) for ds in done])

    body = []
    for c in classes:
        for b in MAF_BINS:
            n = h_med("effect", "", PAIR[0], "", c, "all", b, "n")
            if math.isnan(n):
                continue
            cells = [c, b, _int(n)]
            for m in PAIR:
                cells.append(" / ".join(_f(h_med("effect", "", m, "", c, "all", b, f"abs_mean_logfc_{q}"))
                                        for q in ("q50", "q99")))  # fmt: skip
            for null in nulls:
                cells.append(" / ".join(_pct(h_med("effect", null, m, "", c, "all", b, "share_called"), 2)
                                        for m in PAIR))  # fmt: skip
            body.append(cells)
    lines += _table(["mut_class", "MAF", "n", *(f"{label(m)} q50 / q99" for m in PAIR),
                     *(f"called {pair_names}, {null_label(null)}" for null in nulls)], body)  # fmt: skip
    groups = ("all", "both", f"{PAIR[0]}_only", f"{PAIR[1]}_only")
    lines += [f"The calls' MAF: mean MAF · share in {' / '.join(MAF_BINS[:3])} · n (median over datasets):", ""]
    body = []
    for null in nulls:
        for c in classes:
            cells = []
            for g in groups:
                n = h_med("calls", null, PAIR_NAME, g, c, "all", "all", "n")
                shares = [h_med("calls", null, PAIR_NAME, g, c, "all", b, "share") for b in MAF_BINS[:3]]
                mean_maf = h_med("calls", null, PAIR_NAME, g, c, "all", "all", "mean_maf")
                cells.append("–" if math.isnan(n) else
                             f"{_f(mean_maf)} · {'/'.join(_f(100 * s, 0) for s in shares)} % · {_int(n)}")  # fmt: skip
            if any(cell != "–" for cell in cells):
                body.append([null_label(null), c, *cells])
    lines += _table(["null", "mut_class", "all variants", "called by both", *(f"{label(m)} only" for m in PAIR)], body)
    return "\n".join(lines) + "\n"


def write_outputs(args, vl, nulls, cache, done, rows, notes) -> None:
    for key, name in OUTPUTS.items():
        write_tsv(args.out_dir / name, COLUMNS[key], rows[key])
    write_text(args.out_dir / "report.md", build_report(args, vl, nulls, cache, done, rows, notes))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="the pilot directory: scores*/ and windows/")
    ap.add_argument("--list", type=Path, required=True, help="the variant list TSV")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--variants", required=True, help="window-set name of the variants")
    ap.add_argument("--nulls", nargs="+", default=["null", "genomic_null"],
                    help="window-set names of the nulls (default %(default)s); null is read from <root>/scores, "
                    "any other from <root>/scores_<name>")  # fmt: skip
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--bootstrap", type=int, default=BOOTSTRAP,
                    help="R: replicates of each null's SNV rows (default %(default)s; 0: point values)")  # fmt: skip
    ap.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED, help="R's seed (default %(default)s)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    if args.bootstrap < 0:
        ap.error("--bootstrap must be 0 or more")
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%F %T")  # fmt: skip

    notes = Notes()
    nulls = []
    for null in args.nulls:
        if scores_dir(args.root, null).is_dir():
            nulls.append(null)
        else:
            notes.note(f"the {null_label(null)} null ({null}): no {scores_dir(args.root, null)}; left out")
    if not nulls:
        logger.error("no null has a scores directory under %s", args.root)
        return 1
    started = time.time()
    try:
        vl = read_list(args.list, notes)
    except (OSError, ValueError) as err:
        logger.error("the variant list: %s", err)
        return 1
    logger.info("list %s: %d variants, %.1f s", args.list, len(vl.ids), time.time() - started)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: dict[str, list[dict]] = {key: [] for key in OUTPUTS}
    cache: dict[str, NullRows] = {}
    done: list[str] = []
    for ds in args.datasets:
        started = time.time()
        try:
            data = load_dataset(args, ds, nulls, vl, cache, notes)
        except Exception as err:  # an unreadable table (a truncated npz is not an OSError) leaves this dataset out
            logger.exception("%s: reading the tables", ds)
            notes.problem(f"{ds}: {type(err).__name__}: {err}; left out")
            continue
        if data is None:
            continue
        logger.info("%s: tables read, %.1f s", ds, time.time() - started)
        sections = {**SECTIONS, "R": functools.partial(section_r, reps=args.bootstrap, seed=args.bootstrap_seed)}
        for key, section in sections.items():
            started = time.time()
            try:
                rows[key] += section(data)
            except Exception as err:  # one section's failure leaves the others' results; the log has the traceback
                logger.exception("%s, section %s", ds, key)
                notes.problem(f"{ds}: section {key} failed ({type(err).__name__}: {err})")
                continue
            logger.info("%s %s: %.1f s", ds, key, time.time() - started)
        done.append(ds)
        del data
        write_outputs(args, vl, nulls, cache, done, rows, notes)
    if not done:
        write_outputs(args, vl, nulls, cache, done, rows, notes)
    logger.info("wrote %s: %d of %d datasets, %d problem(s)", args.out_dir / "report.md", len(done),
                len(args.datasets), len(notes.problems))  # fmt: skip
    return 1 if notes.problems or not done else 0


if __name__ == "__main__":
    sys.exit(main())
