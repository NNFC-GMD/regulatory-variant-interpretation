#!/usr/bin/env python3
"""Compare the models' variant scores with measured QTL effects: does the predicted logFC point the way the QTL does?

Reads a table of QTL lead variants (one row per variant, phenotype and group, as TensorQTL's cis permutation pass
writes them, with an effect size per variant), puts each variant on the genome's alleles, finds its scores.py
summaries for every model and dataset that map to its group, and compares the fold-mean logFC (log2 ALT over REF)
with the QTL's effect size.

Alleles
  A QTL variant_id names its two alleles, c:pos:a1:a2 (chr prefix optional), in whatever order the QTL table uses:
  some write the genome's REF first, some sort the two. Each is put on the genome: REF is the --fasta base at pos,
  ALT the other allele; a row whose two alleles both differ from the genome base is dropped (counted as `mismatch`).
  --effect-allele says which allele the effect size counts: `genome_alt` (default; TensorQTL counts a VCF's ALT,
  which TOPMed imputation writes as the genome's ALT) or `second` (the variant_id's second allele; flipped to the
  genome's ALT where the id is the other way round). With --list and --list-af, the table's --af column is checked
  against the list's allele frequency of the genome's ALT on the variants both hold: the run refuses below
  --min-af-r, since a wrong effect allele would flip half the signs.

Groups and datasets
  --groups TSV, columns group and datasets (comma-separated): the datasets whose models stand for that QTL group
  (e.g. the samples of the same condition). Per group and model, the prediction is the mean of the fold-mean logFC
  over the group's datasets that the model has scores for (`matched`), and, for the head-to-head, over the group's
  datasets that every model in --models has (`h2h`, the same datasets for all; groups with none are left out of it).
  `heldout` is h2h with the held-out logFC (each dataset's fold that never trained on the variant's chromosome)
  instead of the fold mean: the same datasets and rows for every model.

Subsets
  by the variant's distance to its phenotype's centre (phenotype_id chrom:start-end): --subset NAME=MAXBP, default
  in_peak=250 (inside a 501-bp peak) and near=500 (inside the scored output window, which is centred on the variant).
  A variant farther away predicts accessibility at itself, not at its peak, and is left out.

Measures, per setting, group (and all groups pooled), subset and model
  concordance   the share of variants whose logFC has the QTL effect's sign (rows where either is 0 are left out);
                0.5 is chance
  top_<f>       the same among the model's top fraction f of variants by |logFC| (its confident calls), f in --top
  spearman, pearson   of logFC against the effect size
  Intervals are percentile bootstrap 95% CIs (--n-boot, --seed) over variants: rows sharing a variant_id are drawn
  together (one variant is often the lead of several groups). Paired differences between two models on the same
  rows (--pairs; h2h and heldout) come from the same draws, with a two-sided bootstrap p, and McNemar's exact test
  on the sign calls.

Inputs
  --qtl CSV or TSV (columns --variant-col, --effect-col, --group-col, --phenotype-col, optional --af-col, --se-col,
  --p-col); --fasta (with .fai); --groups; --scores-root (one or more: <root>/<dataset>/<model>.<variants>.summary.npz
  or .tsv.gz, scores.py's); --list (the scored variant list, for the allele-frequency check); --source (a sentence
  naming where the QTL table comes from, kept in run.json for the reports that read it).
Outputs, under --out-dir
  rows.tsv.gz   every QTL row: its genome alleles, orientation, distance, whether scored, and each model's predictions
  metrics.tsv, paired.tsv, curves.tsv (concordance by top fraction), per_dataset.tsv, run.json, report.md,
  concordance.png
Exit: 0, or 1 when nothing could be compared; 2 on bad input (the af check included).

    python workflows/variant_scoring/qtl_compare.py --qtl <table.csv> --fasta <genome.fa> --groups <groups.tsv> \\
        --scores-root <root>/scores [...] --list <root>/lists/<variants>.tsv --out-dir <dir>
    python workflows/variant_scoring/qtl_compare.py --selftest
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import logging
import math
import multiprocessing
import os
import subprocess
import sys
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from scipy.stats import binomtest, pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

import scores as sc  # noqa: E402

logger = logging.getLogger("qtl_compare")
ET = ZoneInfo("America/New_York")
MODELS = ("cherimoya", "cbp_nobias", "cbp_full", "cbp_bias")
LABELS = {"cherimoya": "Cherimoya", "cbp_nobias": "ChromBPNet corrected", "cbp_full": "ChromBPNet uncorrected",
          "cbp_bias": "ChromBPNet bias model"}  # fmt: skip
SCORE_COLUMNS = ("variant_id", "logfc.mean", "logfc.heldout")


def now_et() -> str:
    return datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S %Z")


# ------------------------------------------------------------------------------------------------ the genome
class Fasta:
    """Single bases from an indexed FASTA (its .fai), by seeking."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.index: dict[str, tuple[int, int, int, int]] = {}
        with open(f"{path}.fai") as fh:
            for line in fh:
                name, length, offset, bases, width = line.split("\t")[:5]
                self.index[name] = (int(length), int(offset), int(bases), int(width))
        self.fh = open(path, "rb")

    def base(self, chrom: str, pos: int) -> str | None:
        """The base at 1-based `pos`, upper case, or None off the sequence."""
        if chrom not in self.index:
            return None
        length, offset, bases, width = self.index[chrom]
        if not 1 <= pos <= length:
            return None
        i = pos - 1
        self.fh.seek(offset + (i // bases) * width + i % bases)
        return self.fh.read(1).decode().upper()


# ------------------------------------------------------------------------------------------------ the QTL table
@dataclass
class Qtl:
    variant: np.ndarray  # genome id chrom:pos:ref:alt
    group: np.ndarray
    phenotype: np.ndarray
    effect: np.ndarray  # per genome ALT
    dist: np.ndarray  # |pos - phenotype centre|
    orientation: np.ndarray  # same / swapped
    af: np.ndarray  # the table's, or NaN
    extra: dict[str, np.ndarray] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)


def parse_variant(text: str) -> tuple[str, int, str, str]:
    c, p, a, b = text.strip().split(":")
    return (c if c.startswith("chr") else f"chr{c}"), int(float(p)), a.upper(), b.upper()


def parse_phenotype(text: str) -> tuple[str, float]:
    """chrom:start-end -> (chrom, centre)."""
    chrom, span = text.strip().rsplit(":", 1)
    start, end = span.split("-")
    return chrom, (int(float(start)) + int(float(end))) / 2


def read_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="", encoding="utf-8-sig") as fh:
        first = fh.readline()
        fh.seek(0)
        reader = csv.reader(fh, delimiter="\t" if "\t" in first else ",")
        header = next(reader)
        return header, [row for row in reader if row and len(row) >= len(header) // 2]


def load_qtl(args, fasta: Fasta) -> Qtl:
    header, rows = read_rows(args.qtl)
    col = {}
    for i, name in enumerate(header):
        col.setdefault(name.strip(), i)  # the first of duplicated names
    need = [args.variant_col, args.effect_col, args.group_col, args.phenotype_col]
    missing = [c for c in need if c not in col]
    if missing:
        raise ValueError(f"{args.qtl}: no column(s) {', '.join(missing)}")
    keep = {k: [] for k in ("variant", "group", "phenotype", "effect", "dist", "orientation", "af")}
    extra_cols = [c for c in (args.se_col, args.p_col) if c and c in col]
    extra = {c: [] for c in extra_cols}
    counts = {"rows": len(rows), "mismatch": 0, "multibase": 0, "other_chrom": 0, "swapped": 0, "same": 0}
    for row in rows:
        chrom, pos, a, b = parse_variant(row[col[args.variant_col]])
        if len(a) != 1 or len(b) != 1:
            counts["multibase"] += 1  # QTL leads as SNVs; an indel would need the table's own allele convention
            continue
        g = fasta.base(chrom, pos)
        if g is None:
            counts["other_chrom"] += 1
            continue
        if g == a:
            ref, alt, orient = a, b, "same"
        elif g == b:
            ref, alt, orient = b, a, "swapped"
        else:
            counts["mismatch"] += 1
            continue
        counts[orient] += 1
        effect = float(row[col[args.effect_col]])
        if args.effect_allele == "second" and orient == "swapped":
            effect = -effect
        pchrom, centre = parse_phenotype(row[col[args.phenotype_col]])
        keep["variant"].append(f"{chrom}:{pos}:{ref}:{alt}")
        keep["group"].append(row[col[args.group_col]].strip())
        keep["phenotype"].append(row[col[args.phenotype_col]].strip())
        keep["effect"].append(effect)
        keep["dist"].append(abs(pos - centre) if pchrom == chrom else math.inf)
        keep["orientation"].append(orient)
        af = row[col[args.af_col]] if args.af_col and args.af_col in col else ""
        keep["af"].append(float(af) if af not in ("", "NA", "nan") else math.nan)
        for c in extra_cols:
            v = row[col[c]]
            extra[c].append(float(v) if v not in ("", "NA", "nan") else math.nan)
    return Qtl(
        variant=np.array(keep["variant"], dtype=str),
        group=np.array(keep["group"], dtype=str),
        phenotype=np.array(keep["phenotype"], dtype=str),
        effect=np.array(keep["effect"], dtype=np.float64),
        dist=np.array(keep["dist"], dtype=np.float64),
        orientation=np.array(keep["orientation"], dtype=str),
        af=np.array(keep["af"], dtype=np.float64),
        extra={c: np.array(v, dtype=np.float64) for c, v in extra.items()},
        counts=counts,
    )


def af_check(qtl: Qtl, list_path: Path, af_column: str) -> dict:
    """Pearson r of the table's allele frequency with the list's AF of the genome ALT, over the variants both hold."""
    wanted = {v: i for i, v in enumerate(qtl.variant) if not math.isnan(qtl.af[i])}
    pairs = []
    with gzip.open(list_path, "rt") if str(list_path).endswith(".gz") else open(list_path) as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader)
        iv, ia = header.index("variant_id"), header.index(af_column)
        for row in reader:
            i = wanted.get(row[iv])
            if i is not None:
                try:
                    pairs.append((qtl.af[i], float(row[ia])))
                except ValueError:
                    pass
    x = np.array([p for p in pairs if not math.isnan(p[1])], dtype=np.float64).reshape(-1, 2)
    r = float(pearsonr(x[:, 0], x[:, 1])[0]) if len(x) > 2 else math.nan
    return {"n": len(x), "r_af_vs_list_alt_af": r, "list_af_column": af_column}


# ------------------------------------------------------------------------------------------------ the scores
def summary_path(roots: list[Path], dataset: str, model: str, variants: str) -> Path | None:
    for root in roots:
        p = root / dataset / f"{model}.{variants}.summary.tsv.gz"
        if p.is_file() or sc.npz_path(p).is_file():
            return p
    return None


def _read_scores(job: tuple[str, str, str, tuple[str, ...]]) -> tuple[str, str, dict]:
    """One summary's logFC columns at the wanted variants (a worker)."""
    dataset, model, path, wanted = job
    t = sc.read_table(Path(path), SCORE_COLUMNS)
    ids = t["variant_id"]
    hit = np.flatnonzero(np.isin(ids, np.array(wanted, dtype=ids.dtype)))
    pos = {ids[i]: i for i in hit}
    rows = np.array([pos.get(v, -1) for v in wanted], dtype=np.int64)
    out = {}
    for c in ("logfc.mean", "logfc.heldout"):
        vals = np.full(len(wanted), np.nan)
        ok = rows >= 0
        if c in t:
            vals[ok] = t[c][rows[ok]]
        out[c] = vals
    out["n_rows"] = len(ids)
    return dataset, model, out


# ------------------------------------------------------------------------------------------------ measures
def concordance(pred: np.ndarray, effect: np.ndarray) -> float:
    ok = (pred != 0) & (effect != 0) & np.isfinite(pred) & np.isfinite(effect)
    return float(np.mean(np.sign(pred[ok]) == np.sign(effect[ok]))) if ok.any() else math.nan


def top_concordance(pred: np.ndarray, effect: np.ndarray, frac: float) -> float:
    ok = np.isfinite(pred) & np.isfinite(effect)
    p, e = pred[ok], effect[ok]
    if len(p) == 0:
        return math.nan
    k = max(1, int(round(frac * len(p))))
    order = np.argsort(-np.abs(p), kind="stable")[:k]
    return concordance(p[order], e[order])


def measures(pred: np.ndarray, effect: np.ndarray, tops: list[float]) -> dict[str, float]:
    ok = np.isfinite(pred) & np.isfinite(effect)
    p, e = pred[ok], effect[ok]
    out = {"n": int(ok.sum()), "concordance": concordance(p, e)}
    for f in tops:
        out[f"top_{f:g}"] = top_concordance(p, e, f)
    if len(p) > 2 and np.ptp(p) > 0 and np.ptp(e) > 0:
        out["spearman"] = float(spearmanr(p, e)[0])
        out["pearson"] = float(pearsonr(p, e)[0])
    else:
        out["spearman"] = out["pearson"] = math.nan
    return out


def cluster_draws(keys: np.ndarray, n_boot: int, seed: int) -> list[np.ndarray]:
    """Bootstrap draws of row indices, rows sharing a key drawn together."""
    uniq, inv = np.unique(keys, return_inverse=True)
    members = [[] for _ in range(len(uniq))]
    for i, k in enumerate(inv):
        members[k].append(i)
    members = [np.array(m, dtype=np.int64) for m in members]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        draws.append(np.concatenate([members[j] for j in pick]) if len(pick) else np.zeros(0, np.int64))
    return draws


def ci(values: list[float]) -> tuple[float, float]:
    v = np.array([x for x in values if not math.isnan(x)])
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (math.nan, math.nan)


def mcnemar(pa: np.ndarray, pb: np.ndarray, effect: np.ndarray) -> tuple[int, int, float]:
    """Rows A calls right and B wrong, the reverse, and McNemar's exact two-sided p."""
    ok = np.isfinite(pa) & np.isfinite(pb) & (effect != 0) & (pa != 0) & (pb != 0)
    ra = np.sign(pa[ok]) == np.sign(effect[ok])
    rb = np.sign(pb[ok]) == np.sign(effect[ok])
    b, c = int(np.sum(ra & ~rb)), int(np.sum(~ra & rb))
    p = float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0
    return b, c, p


# ------------------------------------------------------------------------------------------------ the run
def read_groups(path: Path) -> dict[str, list[str]]:
    groups = {}
    with open(path) as fh:
        for row in csv.DictReader((line for line in fh if not line.startswith("#")), delimiter="\t"):
            groups[row["group"].strip()] = [d.strip() for d in row["datasets"].split(",") if d.strip()]
    return groups


def fmt(x) -> str:
    if isinstance(x, float):
        return "NA" if math.isnan(x) else f"{x:.4g}"
    return str(x)


def write_tsv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(fmt(r.get(c, "")) for c in cols) + "\n")


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_commit() -> str:
    here = Path(__file__).resolve().parent
    try:
        out = subprocess.run(["git", "-C", str(here), "log", "-1", "--format=%h"], capture_output=True, text=True)
        dirty = subprocess.run(["git", "-C", str(here), "status", "--porcelain", "--", Path(__file__).name],
                               capture_output=True, text=True).stdout.strip()  # fmt: skip
        return out.stdout.strip() + (" (this file modified)" if dirty else "")
    except OSError:
        return "unknown"


_STATE: dict = {}


def _combo(job: tuple[str, str, float, str]) -> tuple[list[dict], list[dict], list[dict]]:
    """One setting, subset and group: each model's measures with bootstrap CIs, and the paired differences (h2h and
    heldout)."""
    setting, sub, maxbp, g = job
    qtl, models, groups, tops, args = (_STATE[k] for k in ("qtl", "models", "groups", "tops", "args"))
    preds = _STATE["settings"][setting]
    rows = (qtl.dist <= maxbp) & (np.isin(qtl.group, list(groups)) if g == "all" else qtl.group == g)
    same_rows = setting in ("h2h", "heldout")
    if same_rows:  # the same rows for every model
        rows &= np.all([np.isfinite(preds[m]) for m in models], axis=0)
    idx = np.flatnonzero(rows)
    if len(idx) < args.min_rows:
        return [], [], []
    keys = ("concordance", "spearman") + tuple(f"top_{f:g}" for f in tops)
    draws = cluster_draws(qtl.variant[idx], args.n_boot, args.seed)
    e = qtl.effect[idx]
    base, boots, m_rows, c_rows, p_rows = {}, {}, [], [], []
    for m in models:
        p = preds[m][idx]
        if not np.isfinite(p).any():
            continue
        base[m] = measures(p, e, tops)
        boots[m] = [measures(p[d], e[d], tops) for d in draws]
        row = {"setting": setting, "subset": sub, "group": g, "model": m, "model_label": LABELS[m], **base[m]}
        for k in keys:
            row[f"{k}_lo"], row[f"{k}_hi"] = ci([b[k] for b in boots[m]])
        m_rows.append(row)
        if setting == "h2h" and g == "all":
            for f in np.linspace(0.05, 1.0, 20):
                c_rows.append({"subset": sub, "model": m, "top_fraction": float(f),
                               "concordance": top_concordance(p, e, float(f))})  # fmt: skip
    if not same_rows:
        return m_rows, p_rows, c_rows
    for a, b in (pair.split(":") for pair in args.pairs):
        if a not in base or b not in base:
            continue
        for k in keys:
            bd = np.array([x[k] - y[k] for x, y in zip(boots[a], boots[b])], dtype=np.float64)
            bd = bd[np.isfinite(bd)]
            p_boot = float(min(1.0, 2 * min(np.mean(bd <= 0), np.mean(bd >= 0)))) if len(bd) else math.nan
            lo, hi = ci(list(bd))
            prow = {"setting": setting, "subset": sub, "group": g, "model_a": a, "model_b": b, "measure": k,
                    "a_minus_b": base[a][k] - base[b][k], "lo": lo, "hi": hi, "p_boot": p_boot, "n": len(idx)}  # fmt: skip
            if k == "concordance":
                prow["a_right_b_wrong"], prow["a_wrong_b_right"], prow["p_mcnemar"] = mcnemar(preds[a][idx],
                                                                                              preds[b][idx], e)  # fmt: skip
            p_rows.append(prow)
    return m_rows, p_rows, c_rows


def run(args) -> int:
    t0 = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fasta = Fasta(args.fasta)
    qtl = load_qtl(args, fasta)
    logger.info("%s: %s", args.qtl, qtl.counts)
    run_info: dict = {"started_et": now_et(), "argv": sys.argv, "code": code_commit(), "source": args.source,
                      "counts": dict(qtl.counts)}  # fmt: skip
    if args.list and args.list_af:
        chk = af_check(qtl, Path(args.list), args.list_af)
        run_info["af_check"] = chk
        logger.info("allele-frequency check: %s", chk)
        if not chk["r_af_vs_list_alt_af"] >= args.min_af_r:
            logger.error("the table's allele frequency does not follow the genome ALT's (r %s < %s): wrong effect "
                         "allele?", chk["r_af_vs_list_alt_af"], args.min_af_r)  # fmt: skip
            return 2
    groups = read_groups(args.groups)
    unknown = sorted(set(qtl.group) - set(groups))
    run_info["groups_without_datasets"] = unknown
    models = list(args.models)
    roots = [Path(r) for r in args.scores_root]
    jobs, have = [], {}
    wanted = tuple(sorted(set(qtl.variant)))
    all_datasets = sorted({d for dl in groups.values() for d in dl})
    for ds in all_datasets:
        for m in models:
            p = summary_path(roots, ds, m, args.variants)
            if p is not None:
                jobs.append((ds, m, str(p), wanted))
                have.setdefault(m, set()).add(ds)
    logger.info("%d summaries to read (%d variants each)", len(jobs), len(wanted))
    pred: dict[tuple[str, str], dict] = {}
    with ProcessPoolExecutor(max_workers=max(1, min(args.workers, len(jobs)))) as ex:
        for ds, m, res in ex.map(_read_scores, jobs):
            pred[(ds, m)] = res
    run_info["summaries"] = {f"{ds}/{m}": int(r["n_rows"]) for (ds, m), r in sorted(pred.items())}
    vindex = {v: i for i, v in enumerate(wanted)}
    vi = np.array([vindex[v] for v in qtl.variant], dtype=np.int64)

    def group_pred(model: str, datasets: list[str], column: str) -> np.ndarray:
        """Per QTL row: the mean over `datasets` of the model's column; NaN where none has the variant."""
        stack = [pred[(d, model)][column][vi] for d in datasets if (d, model) in pred]
        if not stack:
            return np.full(len(vi), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns (a variant no dataset has) stay NaN
            return np.nanmean(np.vstack(stack), axis=0) if len(stack) > 1 else stack[0]

    # Per row and setting, each model's prediction from the datasets of the row's group.
    settings = {"matched": {}, "h2h": {}, "heldout": {}}
    h2h_sets = {g: [d for d in ds if all(d in have.get(m, set()) for m in models)] for g, ds in groups.items()}
    for m in models:
        for name in settings:
            settings[name][m] = np.full(len(vi), np.nan)
        for g, ds in groups.items():
            rows = qtl.group == g
            if not rows.any():
                continue
            mine = [d for d in ds if d in have.get(m, set())]
            settings["matched"][m][rows] = group_pred(m, mine, "logfc.mean")[rows]
            settings["heldout"][m][rows] = group_pred(m, h2h_sets[g], "logfc.heldout")[rows]
            settings["h2h"][m][rows] = group_pred(m, h2h_sets[g], "logfc.mean")[rows]
    run_info["datasets"] = {g: {"matched": {m: [d for d in ds if d in have.get(m, set())] for m in models},
                                "h2h": h2h_sets[g]} for g, ds in groups.items()}  # fmt: skip

    subsets = [(n, float(b)) for n, b in (s.split("=") for s in args.subset)]
    tops = [float(f) for f in args.top]
    run_info["settings"] = {"models": models, "pairs": list(args.pairs), "subsets": dict(subsets), "tops": tops,
                            "n_boot": args.n_boot, "seed": args.seed, "min_rows": args.min_rows,
                            "effect_allele": args.effect_allele}  # fmt: skip
    group_names = ["all"] + sorted(set(qtl.group) & set(groups))
    combos = [(setting, sub, maxbp, g) for setting in settings for sub, maxbp in subsets for g in group_names]
    _STATE.update(qtl=qtl, settings=settings, models=models, groups=groups, tops=tops, args=args)
    metrics, paired, curves = [], [], []
    ctx = multiprocessing.get_context("fork")  # the workers read _STATE as the parent left it
    with ProcessPoolExecutor(max_workers=max(1, min(args.workers, len(combos))), mp_context=ctx) as ex:
        for m_rows, p_rows, c_rows in ex.map(_combo, combos):
            metrics += m_rows
            paired += p_rows
            curves += c_rows

    # Each dataset alone (every model it has), on the rows of the groups it stands for.
    per_dataset = []
    for sub, maxbp in subsets:
        for ds in all_datasets:
            gs = [g for g, dl in groups.items() if ds in dl]
            rows = (qtl.dist <= maxbp) & np.isin(qtl.group, gs)
            ms = [m for m in models if (ds, m) in pred]
            vals = {m: pred[(ds, m)]["logfc.mean"][vi] for m in ms}
            ok = rows & np.all([np.isfinite(vals[m]) for m in ms], axis=0) if ms else rows & False
            for m in ms:
                r = measures(vals[m][ok], qtl.effect[ok], tops)
                per_dataset.append({"subset": sub, "dataset": ds, "groups": ",".join(gs), "model": m, **r})

    write_tsv(out / "metrics.tsv", metrics)
    write_tsv(out / "paired.tsv", paired)
    write_tsv(out / "curves.tsv", curves)
    write_tsv(out / "per_dataset.tsv", per_dataset)
    with gzip.open(out / "rows.tsv.gz", "wt") as fh:
        cols = ["variant_id", "group", "phenotype_id", "orientation", "dist_to_centre", "effect"]
        cols += list(qtl.extra) + [f"{s}.{m}" for s in settings for m in models]
        fh.write("\t".join(cols) + "\n")
        for i in range(len(vi)):
            vals = [qtl.variant[i], qtl.group[i], qtl.phenotype[i], qtl.orientation[i], fmt(float(qtl.dist[i])),
                    fmt(float(qtl.effect[i]))]  # fmt: skip
            vals += [fmt(float(qtl.extra[c][i])) for c in qtl.extra]
            vals += [fmt(float(settings[s][m][i])) for s in settings for m in models]
            fh.write("\t".join(vals) + "\n")
    scored = np.isfinite(settings["matched"][models[0]])
    run_info["scored_rows"] = {sub: {"rows": int(np.sum(qtl.dist <= b)), "scored": int(np.sum((qtl.dist <= b) & scored))}
                               for sub, b in subsets}  # fmt: skip
    run_info["inputs"] = {"qtl": {"path": str(args.qtl), "md5": md5(Path(args.qtl))}, "groups": str(args.groups),
                          "scores_roots": [str(r) for r in roots], "fasta": str(args.fasta)}  # fmt: skip
    run_info["seconds"] = round(time.time() - t0, 1)
    run_info["finished_et"] = now_et()
    (out / "run.json").write_text(json.dumps(run_info, indent=1, default=str) + "\n")
    write_report(out, metrics, paired, run_info, models, tops)
    plot(out, curves, metrics, models)
    logger.info("wrote %s (%d metric rows, %d paired) in %.0f s", out, len(metrics), len(paired), time.time() - t0)
    return 0 if metrics else 1


def write_report(out: Path, metrics, paired, info, models, tops) -> None:
    lines = [f"# QTL concordance ({info['finished_et']})", "", f"QTL rows: {info['counts']}", ""]
    if "af_check" in info:
        lines += [f"Allele-frequency check: {info['af_check']}", ""]
    lines += [f"Scored rows by subset: {info['scored_rows']}", ""]
    for setting in ("h2h", "matched", "heldout"):
        rows = [r for r in metrics if r["setting"] == setting]
        if not rows:
            continue
        lines += [f"## {setting}", "", "| subset | group | model | n | concordance [95% CI] | top 10% | spearman |",
                  "|---|---|---|---|---|---|---|"]  # fmt: skip
        for r in rows:
            lines.append(f"| {r['subset']} | {r['group']} | {LABELS[r['model']]} | {r['n']} | "
                         f"{fmt(r['concordance'])} [{fmt(r['concordance_lo'])}, {fmt(r['concordance_hi'])}] | "
                         f"{fmt(r.get('top_0.1', math.nan))} | {fmt(r['spearman'])} |")  # fmt: skip
        lines.append("")
    for setting in ("h2h", "heldout"):
        rows = [r for r in paired if r["setting"] == setting]
        if not rows:
            continue
        lines += [f"## paired ({setting}, same rows)", "", "| subset | group | A − B | measure | diff [95% CI] | "
                  "p (boot) | McNemar p |", "|---|---|---|---|---|---|---|"]  # fmt: skip
        for r in rows:
            lines.append(f"| {r['subset']} | {r['group']} | {r['model_a']} − {r['model_b']} | {r['measure']} | "
                         f"{fmt(r['a_minus_b'])} [{fmt(r['lo'])}, {fmt(r['hi'])}] | {fmt(r['p_boot'])} | "
                         f"{fmt(r.get('p_mcnemar', ''))} |")  # fmt: skip
        lines.append("")
    (out / "report.md").write_text("\n".join(lines) + "\n")


def plot(out: Path, curves, metrics, models) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("no matplotlib: no figure")
        return
    subs = sorted({c["subset"] for c in curves})
    if not subs:
        return
    colors = {"cherimoya": "#d55e00", "cbp_nobias": "#0072b2", "cbp_full": "#56b4e9", "cbp_bias": "#999999"}
    fig, axes = plt.subplots(1, len(subs), figsize=(5.2 * len(subs), 4), squeeze=False)
    for ax, sub in zip(axes[0], subs):
        for m in models:
            c = [x for x in curves if x["subset"] == sub and x["model"] == m]
            if c:
                ax.plot([x["top_fraction"] for x in c], [x["concordance"] for x in c], color=colors.get(m),
                        label=LABELS[m], lw=2)  # fmt: skip
        ax.axhline(0.5, color="k", lw=0.8, ls=":")
        ax.set_xlabel("top fraction of variants by |predicted logFC|")
        ax.set_ylabel("sign concordance with the QTL effect")
        ax.set_title(f"{sub}, head to head, all groups")
        ax.set_ylim(0.4, 1.0)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "concordance.png", dpi=144)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ the selftest
def selftest() -> int:
    """A two-chromosome genome, a QTL table with sorted and genome-ordered ids, and fake summaries for two datasets."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        seq = {"chr1": "ACGTACGTAC" * 30, "chr2": "TTGCAGGCAT" * 30}
        with open(d / "g.fa", "w") as fh, open(d / "g.fa.fai", "w") as fai:
            off = 0
            for name, s in seq.items():
                head = f">{name}\n"
                off += len(head)
                lines = [s[i : i + 60] for i in range(0, len(s), 60)]
                fai.write(f"{name}\t{len(s)}\t{off}\t60\t61\n")
                fh.write(head + "\n".join(lines) + "\n")
                off += sum(len(x) + 1 for x in lines)
        fa = Fasta(d / "g.fa")
        assert fa.base("chr1", 1) == "A" and fa.base("chr1", 61) == seq["chr1"][60] and fa.base("chr2", 3) == "G"
        # chr1 pos 2 is C: "1:2:A:C" is sorted (genome C, so swapped) and "1:3:G:T" is genome order (G).
        qrows = [("1:2:A:C", 1.0, "g1", "chr1:0-6", 0.6), ("1:3:G:T", -2.0, "g1", "chr1:0-6", 0.3),
                 ("2:3:A:G", 0.5, "g2", "chr2:0-6", 0.4), ("2:4:A:T", 0.5, "g2", "chr2:0-6", 0.4),
                 ("2:5:A:G", 0.7, "g2", "chr2:900-1000", 0.4)]  # fmt: skip
        with open(d / "q.csv", "w") as fh:
            fh.write("variant_id,slope,experimental_group,phenotype_id,af\n")
            for r in qrows:
                fh.write(",".join(map(str, r)) + "\n")
        (d / "groups.tsv").write_text("group\tdatasets\ng1\tdsA,dsB\ng2\tdsB,dsC\n")
        # Genome ids: chr1:2:C:A (effect +1 for genome ALT A), chr1:3:G:T (-2), chr2:3:G:A (+0.5); chr2:4 is C in
        # the genome, neither A nor T: a mismatch; chr2:5 is A: same, but 945 bp from its peak's centre.
        ids = np.array(["chr1:2:C:A", "chr1:3:G:T", "chr2:3:G:A", "chr2:5:A:G", "chrX:1:A:C"])
        lf = {("dsA", "cherimoya"): [0.3, -0.4, 0.0, 0.1, 9], ("dsB", "cherimoya"): [0.1, -0.2, 0.6, 0.1, 9],
              ("dsA", "cbp_nobias"): [-0.3, -0.4, 0, 0.1, 9], ("dsB", "cbp_nobias"): [-0.1, 0.2, -0.6, 0.1, 9],
              ("dsC", "cherimoya"): [0.0, 0.0, -1.0, 0.1, 9]}  # fmt: skip
        for (ds, m), v in lf.items():
            (d / "scores" / ds).mkdir(parents=True, exist_ok=True)
            sc.save_table(d / "scores" / ds / f"{m}.variants.summary.tsv.gz",
                          {"variant_id": ids, "logfc.mean": np.array(v, np.float64),
                           "logfc.heldout": np.array(v, np.float64)}, tsv=False)  # fmt: skip
        args = build_parser().parse_args(["--qtl", str(d / "q.csv"), "--fasta", str(d / "g.fa"), "--groups",
                                          str(d / "groups.tsv"), "--scores-root", str(d / "scores"), "--out-dir",
                                          str(d / "out"), "--models", "cherimoya", "cbp_nobias", "--pairs",
                                          "cherimoya:cbp_nobias", "--n-boot", "50", "--min-rows", "1",
                                          "--subset", "in_peak=250", "--workers", "2"])  # fmt: skip
        q = load_qtl(args, fa)
        assert q.counts["mismatch"] == 1 and q.counts["swapped"] == 2 and q.counts["same"] == 2, q.counts
        assert list(q.variant[:3]) == ["chr1:2:C:A", "chr1:3:G:T", "chr2:3:G:A"], q.variant
        assert list(q.effect[:3]) == [1.0, -2.0, 0.5]  # genome_alt: the effect stays
        assert run(args) == 0
        m = {(r["setting"], r["group"], r["model"]): r for r in csv.DictReader(open(d / "out" / "metrics.tsv"),
                                                                              delimiter="\t")}  # fmt: skip
        # matched, g1: Cherimoya mean(0.3, 0.1) = 0.2 (+, effect +) and -0.3 (effect -): both right.
        assert float(m[("matched", "g1", "cherimoya")]["concordance"]) == 1.0
        # ChromBPNet corrected g1: -0.2 (effect +: wrong), -0.1 (effect -: right).
        assert float(m[("matched", "g1", "cbp_nobias")]["concordance"]) == 0.5
        # g2, matched: Cherimoya mean(dsB +0.6, dsC -1.0) = -0.2, effect +: wrong; ChromBPNet (dsB) -0.6: wrong.
        # chr2:5 is 945 bp off: not in in_peak.
        assert float(m[("matched", "g2", "cherimoya")]["concordance"]) == 0.0
        assert int(m[("matched", "g2", "cherimoya")]["n"]) == 1
        assert float(m[("matched", "g2", "cbp_nobias")]["concordance"]) == 0.0
        # h2h and heldout use the datasets both models have (g2: dsB only), so Cherimoya's +0.6 is right.
        for setting in ("h2h", "heldout"):
            assert float(m[(setting, "g2", "cherimoya")]["concordance"]) == 1.0, setting
            assert float(m[(setting, "all", "cherimoya")]["concordance"]) == 1.0, setting
        p = {r["setting"]: r for r in csv.DictReader(open(d / "out" / "paired.tsv"), delimiter="\t")
             if r["group"] == "all" and r["measure"] == "concordance"}  # fmt: skip
        assert set(p) == {"h2h", "heldout"}, p
        for r in p.values():
            assert abs(float(r["a_minus_b"]) - 2 / 3) < 1e-3, r  # the TSV holds 4 significant digits
            assert int(r["a_right_b_wrong"]) == 2 and int(r["a_wrong_b_right"]) == 0, r
        assert (d / "out" / "report.md").is_file() and (d / "out" / "rows.tsv.gz").is_file()
        # --effect-allele second flips the swapped rows.
        args2 = build_parser().parse_args(["--qtl", str(d / "q.csv"), "--fasta", str(d / "g.fa"), "--groups",
                                           str(d / "groups.tsv"), "--scores-root", str(d / "scores"),
                                           "--out-dir", str(d / "o2"), "--effect-allele", "second"])  # fmt: skip
        q2 = load_qtl(args2, fa)
        assert list(q2.effect[:3]) == [-1.0, -2.0, -0.5], q2.effect
    print("qtl_compare selftest: all checks passed")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--qtl", type=Path)
    ap.add_argument("--fasta", type=Path)
    ap.add_argument("--groups", type=Path)
    ap.add_argument("--source", default="", help="where the QTL table comes from (a sentence; kept in run.json)")
    ap.add_argument("--scores-root", nargs="+", default=[])
    ap.add_argument("--variants", default="variants", help="the window set's name in the summaries (variants)")
    ap.add_argument("--list", type=Path, help="the scored variant list, for the allele-frequency check")
    ap.add_argument("--list-af", default="", help="its allele-frequency column (e.g. AF_nfe); empty: no check")
    ap.add_argument("--min-af-r", type=float, default=0.8)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--variant-col", default="variant_id")
    ap.add_argument("--effect-col", default="slope")
    ap.add_argument("--se-col", default="slope_se")
    ap.add_argument("--p-col", default="pval_nominal")
    ap.add_argument("--group-col", default="experimental_group")
    ap.add_argument("--phenotype-col", default="phenotype_id")
    ap.add_argument("--af-col", default="af")
    ap.add_argument("--effect-allele", choices=("genome_alt", "second"), default="genome_alt")
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    ap.add_argument("--pairs", nargs="*", default=["cherimoya:cbp_nobias", "cherimoya:cbp_full",
                                                   "cbp_nobias:cbp_full", "cherimoya:cbp_bias"])  # fmt: skip
    ap.add_argument("--subset", nargs="+", default=["in_peak=250", "near=500"])
    ap.add_argument("--top", nargs="+", default=["0.1", "0.25", "0.5"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-rows", type=int, default=20)
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument("--selftest", action="store_true")
    return ap


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    if args.selftest:
        return selftest()
    for name in ("qtl", "fasta", "groups", "out_dir"):
        if getattr(args, name) is None:
            build_parser().error(f"--{name.replace('_', '-')} is required")
    if not args.scores_root:
        build_parser().error("--scores-root is required")
    try:
        return run(args)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        logger.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
