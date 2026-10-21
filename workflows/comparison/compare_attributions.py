#!/usr/bin/env python3
"""Level 2 of the comparison: per-peak agreement of ChromBPNet's and Cherimoya's attributions.

For one dataset (--dataset), every peak both models attributed is joined by peak name (never by row index):
ChromBPNet's row i is line i of its fold's interpretation.interpreted_regions.bed (05.0; the same bed for every fold),
Cherimoya's row j is the peaks-file line flatnonzero(idx)[j] (Cherimoya drops the windows holding a base other than
A, C, G or T). Cherimoya's W columns are ChromBPNet's L columns c0 .. c0 + W, c0 = L/2 - W/2 (557 at L = 2114 and
W = 1000): both are centred on the summit. On every joined row of every fold, Cherimoya's one-hot must equal
ChromBPNet's raw/seq[:, :, c0:c0 + W] (the plan's V1), or the dataset fails.

The maps compared, each (4, W) of hypothetical scores per peak:
  ChromBPNet   each fold's 05.0 h5 (shap/seq, float16) and 06.0's fold average
  Cherimoya    each fold's attribution (the float32 npz; with --cheri-source h5, the converted float16 h5 instead, the
               float16 sensitivity check) and 06.0's fold average
and their projected score, the hypothetical score of the base present (the map times the one-hot, summed over the
four bases). Within each window [summit - h, summit + h) of --windows (half-widths; default 250, MoDISco's window,
and 500, Fi-NeMo's), per peak:
  pearson, spearman  of the two projected scores (Spearman on average ranks)
  cosine             of the two 4 x 2h hypothetical maps
  jaccard            of the --top-k (50) positions with the largest |projected score|
  sum_<x>, l2_<x>    each map's projected score summed, and its L2 norm
for these comparisons (the files, under <out-dir>/<dataset>/):
  fold_<k>              (A) ChromBPNet fold k against Cherimoya fold k; `role` is the peak's chromosome's role in
                        fold k (train, valid or test, from igvf_tf_chrombpnet's folds/fold_<k>.json)
  average               (B) the two fold averages
  loo_<model>_<k>       (C) within a model, fold k against the mean of its other folds (float32): the
                        reproducibility ceiling every cross-model number is read against
  heldout_<model>       (D) within a model, per peak: the leave-one-out agreement at the fold whose test set holds
                        the peak's chromosome (heldout_*) and the mean of it over the folds that trained on it
                        (train_*, role train)
  heldout_cross         (D) the two models, each on the fold that held the peak's chromosome out
each as <name>.per_peak.tsv.gz with the peak's name, chrom, start, summit, signalValue and GC content per window.
summary.tsv holds, per comparison, window and metric, the count, mean, median and quartiles over all peaks and
within strata: the chromosome's role (per-fold comparisons), signalValue decile and GC decile (of the widest
window), deciles taken over the dataset's joined peaks. dataset.json records the joins, shapes, V1 counts and timings.

--summarize reads every dataset's summary.tsv (those whose unit is done) and writes, under <out-dir>:
  summary.tsv          all of them, with the dataset
  statistics.tsv       per window and metric, over datasets: each quantity's median with a bootstrap 95 % CI
                       (datasets resampled) -- cross-model per fold (A, the mean over folds of the per-fold
                       medians), on the averages (B), each model's leave-one-out ceiling (C), held-out and training
                       agreement (D) -- and the paired differences the plan reads: A against each ceiling, held-out
                       against training within each model and across the models, each with a bootstrap CI and a
                       two-sided Wilcoxon signed-rank test over the datasets; A against each ceiling also over the
                       folds (dataset x fold)
  float16.tsv          for a dataset run with --cheri-source h5 --tag float16: its medians against the float32 run's
  agreement_w<width>.pdf/.png, heldout_w<width>.pdf/.png
                       per metric, one dot per dataset (its Okabe-Ito colour and marker): cross-model agreement next
                       to both models' ceilings; held-out against training agreement

Waiting (exit 3, nothing written): a fold whose attribution attr_status.py does not call DONE (or, for
--cheri-source h5, whose conversion is not current), a missing ChromBPNet fold or average, Cherimoya's average without
downstream.sh's V2 marker (<output_dir>/.done/v2). Failed (exit 1): beds that differ between folds, a join that does
not hold, shapes that do not fit, a V1 mismatch. A done marker (<out-dir>/.done/<dataset>[.<tag>].json) records the
inputs; a rerun with the same inputs and settings does nothing (--force runs anyway).

    python workflows/comparison/compare_attributions.py --dataset <ds> [--windows 250,500] [--top-k 50]
        [--block-rows 4096] [--cheri-source npz|h5 --tag <tag>] [--max-rows N] [--out-dir <dir>] [--force]
    python workflows/comparison/compare_attributions.py --summarize --datasets <list.txt> [--out-dir <dir>]

The roots come from --cheri-root, --chrombpnet-root, --comparison-root and --folds-dir (or CHERI_ROOT,
CHROMBPNET_ROOT, COMPARISON_ROOT and REPO_ROOT/folds). Runs in the pixi cherimoya environment (numpy, scipy, pandas,
h5py, hdf5plugin), inside a box: about 2-3 GB of RAM at 4,096 rows per block, single-threaded, about half an hour
per dataset of 290,000 peaks.
"""

from __future__ import annotations

import argparse
import json
import logging
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1] / d) for d in ("comparison", "train", "attribution", "downstream")
]

import attr_status  # noqa: E402
import common  # noqa: E402
import to_chrombpnet_scores as tcs  # noqa: E402

logger = logging.getLogger("compare_attributions")

FORMAT = 1
AGREEMENT = ("pearson", "spearman", "cosine", "jaccard")
N_DECILES = 10


# ---------------------------------------------------------------------------------------------------------------------
# inputs


def chrombpnet_average(cbp_bed: Path, dataset: str, head: str) -> Path:
    """06.0's average for ChromBPNet: <output_dir>/contrib_scores/<ds>/<ds>_average_shaps.<head>.h5, output_dir being
    two levels above the fold's model directory (whose interpretation/ holds the bed)."""
    output_dir = cbp_bed.parent.parent.parent.parent
    return output_dir / "contrib_scores" / dataset / f"{dataset}_average_shaps.{head}.h5"


def resolve(args: argparse.Namespace, dataset: str) -> dict:
    """The dataset's inputs (paths), after checking that each is there and done; raises Waiting otherwise."""
    try:
        cfg, folds = tcs.dataset_folds(args.cheri_root, dataset)
    except tcs.Refused as exc:
        raise common.Waiting(str(exc)) from exc
    heads = {f["params"].get("output") for f in folds}
    if heads != {common.HEAD}:
        raise common.Failed(f"{dataset}: the folds attribute {sorted(map(str, heads))}, not the {common.HEAD} head")
    missing: list[str] = []
    out: dict = {"dataset": dataset, "cfg": cfg, "folds": [], "loci": None}
    for f in folds:
        k, fold_dir = f["fold"], f["where"]["fold_dir"]
        state, why = attr_status.check_fold(fold_dir)
        if state != "done":
            missing.append(f"fold {k}: attribution {state} ({'; '.join(why[:2])})")
            continue
        loci = Path(f["params"]["loci"][0])
        if out["loci"] is None:
            out["loci"] = loci
        elif loci != out["loci"]:
            raise common.Failed(f"{dataset}: fold {k} attributes {loci}, fold {folds[0]['fold']} {out['loci']}")
        entry = {"fold": k, "ohe": f["files"]["ohe"], "attr": f["files"]["attr"], "idx": f["files"]["idx"],
                 "meta": f["files"]["meta"], "h5": f["where"]["h5"], "fold_dir": fold_dir}  # fmt: skip
        if args.cheri_source == "h5":
            try:
                reason = tcs.up_to_date(
                    f["where"], tcs.stat_sources(tcs.fold_sources(f["params"], f["attribute_json"]))
                )
            except tcs.Refused as exc:
                reason = str(exc)
            if reason is not None:
                missing.append(f"fold {k}: conversion not current ({reason})")
        try:
            cbp = tcs.chrombpnet_interpretation(fold_dir, common.HEAD)
        except tcs.Refused as exc:
            raise common.Failed(str(exc)) from exc
        for role in ("h5", "bed"):
            if not cbp[role].is_file():
                missing.append(f"fold {k}: no ChromBPNet {cbp[role]}")
        entry["cbp_h5"], entry["cbp_bed"] = cbp["h5"], cbp["bed"]
        out["folds"].append(entry)
    if out["folds"]:
        out["cbp_average"] = chrombpnet_average(out["folds"][0]["cbp_bed"], dataset, common.HEAD)
        if not out["cbp_average"].is_file():
            missing.append(f"no ChromBPNet average {out['cbp_average']}")
    out["cheri_average"] = tcs.average_path(cfg, dataset, common.HEAD)
    v2 = Path(cfg["output_dir"]) / ".done" / "v2"
    out["cheri_v2"] = v2
    if not v2.is_file():
        missing.append(f"no Cherimoya average checked by V2 yet ({v2})")
    elif not out["cheri_average"].is_file():
        missing.append(f"no Cherimoya average {out['cheri_average']}")
    if missing:
        raise common.Waiting(f"{dataset}: " + "; ".join(missing[:4]) + (" ..." if len(missing) > 4 else ""))
    return out


def input_paths(inp: dict, source: str) -> dict[str, Path]:
    paths: dict[str, Path] = {"loci": inp["loci"], "cbp_average": inp["cbp_average"],
                              "cheri_average": inp["cheri_average"], "cheri_v2": inp["cheri_v2"]}  # fmt: skip
    for f in inp["folds"]:
        k = f["fold"]
        roles = ("ohe", "attr", "idx", "meta") if source == "npz" else ("idx", "meta", "h5")
        for role in roles:
            paths[f"fold_{k}/{role}"] = f[role]
        paths[f"fold_{k}/cbp_h5"] = f["cbp_h5"]
        paths[f"fold_{k}/cbp_bed"] = f["cbp_bed"]
    return paths


# ---------------------------------------------------------------------------------------------------------------------
# per-peak measures


def window_columns(width: int, half: int) -> tuple[int, int]:
    """Columns [a, b) of a W-wide summit-centred window that hold [summit - half, summit + half)."""
    if half <= 0 or 2 * half > width:
        raise common.Failed(f"window half-width {half} does not fit the {width}-column attributions")
    return width // 2 - half, width // 2 + half


def column_offset(length: int, width: int) -> int:
    """ChromBPNet's column of Cherimoya's column 0: L/2 - W/2 (557 at L = 2114, W = 1000)."""
    if width > length or (length - width) % 2:
        raise common.Failed(f"a {width}-column window cannot be centred in {length} columns")
    return length // 2 - width // 2


def rowwise_pearson(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    den = np.sqrt((x * x).sum(axis=1) * (y * y).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (x * y).sum(axis=1) / den
    r[den == 0] = np.nan
    return r


def average_ranks(x: np.ndarray) -> np.ndarray:
    """Row-wise average ranks (1-based; ties share their mean rank), as scipy's rankdata(method='average')."""
    from scipy.stats import rankdata

    return rankdata(x, method="average", axis=1)


def top_mask(p: np.ndarray, k: int) -> np.ndarray:
    k = min(k, p.shape[1])
    idx = np.argpartition(-np.abs(p), k - 1, axis=1)[:, :k]
    mask = np.zeros(p.shape, dtype=bool)
    np.put_along_axis(mask, idx, True, axis=1)
    return mask


class Map:
    """One (B, 4, W) map of a block, with its projected score and per-window features, computed once."""

    def __init__(self, hyp: np.ndarray, ohe: np.ndarray, top_k: int) -> None:
        self.hyp = hyp
        self.proj = (hyp * ohe).sum(axis=1, dtype=np.float64)
        self.top_k = top_k
        self.cache: dict = {}

    def feature(self, name: str, a: int, b: int):
        key = (name, a, b)
        if key not in self.cache:
            p = self.proj[:, a:b]
            if name == "rank":
                value = average_ranks(p)
            elif name == "top":
                value = top_mask(p, self.top_k)
            elif name == "sum":
                value = p.sum(axis=1)
            elif name == "l2":
                value = np.sqrt((p * p).sum(axis=1))
            elif name == "norm":
                h = self.hyp[:, :, a:b]
                value = np.sqrt((h * h).sum(axis=(1, 2), dtype=np.float64))
            else:
                raise KeyError(name)
            self.cache[key] = value
        return self.cache[key]


def pair_measures(x: Map, y: Map, a: int, b: int) -> dict[str, np.ndarray]:
    """The agreement of two maps over columns [a, b), per row."""
    px, py = x.proj[:, a:b], y.proj[:, a:b]
    den = x.feature("norm", a, b) * y.feature("norm", a, b)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = (x.hyp[:, :, a:b] * y.hyp[:, :, a:b]).sum(axis=(1, 2), dtype=np.float64) / den
    cosine[den == 0] = np.nan
    inter = (x.feature("top", a, b) & y.feature("top", a, b)).sum(axis=1)
    k = min(x.top_k, b - a)
    return {
        "pearson": rowwise_pearson(px, py),
        "spearman": rowwise_pearson(x.feature("rank", a, b), y.feature("rank", a, b)),
        "cosine": cosine,
        "jaccard": inter / (2 * k - inter),
    }


# ---------------------------------------------------------------------------------------------------------------------
# the pass over a dataset


class Columns:
    """Per-peak float32 columns of one comparison, filled block by block."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.data: dict[str, np.ndarray] = {}

    def put(self, name: str, j0: int, values: np.ndarray) -> None:
        if name not in self.data:
            self.data[name] = np.full(self.n, np.nan, dtype=np.float32)
        self.data[name][j0 : j0 + len(values)] = values


class Sources:
    """Block readers for every map: Cherimoya's folds (npz streams or converted h5) and average, ChromBPNet's folds
    and average, each returning the joined rows of the block as (B, 4, W)."""

    def __init__(self, inp: dict, source: str, cbp_rows: np.ndarray, c0: int, width: int) -> None:
        import h5py

        self.source, self.cbp_rows, self.c0, self.width = source, cbp_rows, c0, width
        self.closers = []
        self.cheri_ohe, self.cheri_attr, self.cheri_h5 = {}, {}, {}
        self.cbp = {}
        for f in inp["folds"]:
            k = f["fold"]
            if source == "npz":
                self.cheri_ohe[k] = tcs.NpzRows(Path(f["ohe"]))
                self.cheri_attr[k] = tcs.NpzRows(Path(f["attr"]))
                self.closers += [self.cheri_ohe[k], self.cheri_attr[k]]
            else:
                fh = h5py.File(f["h5"], "r")
                self.cheri_h5[k] = fh
                self.closers.append(fh)
            fh = h5py.File(f["cbp_h5"], "r")
            self.cbp[k] = fh
            self.closers.append(fh)
        self.cheri_avg = h5py.File(inp["cheri_average"], "r")
        self.cbp_avg = h5py.File(inp["cbp_average"], "r")
        self.closers += [self.cheri_avg, self.cbp_avg]

    def shapes(self) -> dict:
        out = {}
        for k, fh in self.cbp.items():
            out[f"chrombpnet fold {k}"] = (fh["raw/seq"].shape, fh["shap/seq"].shape)
        out["chrombpnet average"] = (self.cbp_avg["raw/seq"].shape, self.cbp_avg["shap/seq"].shape)
        if self.source == "npz":
            for k in self.cheri_attr:
                out[f"cherimoya fold {k}"] = (self.cheri_ohe[k].shape, self.cheri_attr[k].shape)
        else:
            for k, fh in self.cheri_h5.items():
                out[f"cherimoya fold {k}"] = (fh["raw/seq"].shape, fh["shap/seq"].shape)
        out["cherimoya average"] = (self.cheri_avg["raw/seq"].shape, self.cheri_avg["shap/seq"].shape)
        return out

    def _cbp_block(self, ds, j0: int, j1: int) -> np.ndarray:
        rows = self.cbp_rows[j0:j1]
        i0, i1 = int(rows[0]), int(rows[-1]) + 1
        return ds[i0:i1, :, self.c0 : self.c0 + self.width][rows - i0]

    def cherimoya(self, k: str, j0: int, j1: int) -> tuple[np.ndarray, np.ndarray]:
        if self.source == "npz":
            return self.cheri_ohe[k].read(j1 - j0), self.cheri_attr[k].read(j1 - j0)
        fh = self.cheri_h5[k]
        return fh["raw/seq"][j0:j1], fh["shap/seq"][j0:j1]

    def chrombpnet(self, k: str, j0: int, j1: int) -> tuple[np.ndarray, np.ndarray]:
        return self._cbp_block(self.cbp[k]["raw/seq"], j0, j1), self._cbp_block(self.cbp[k]["shap/seq"], j0, j1)

    def averages(self, j0: int, j1: int) -> dict:
        return {
            "cherimoya": (self.cheri_avg["raw/seq"][j0:j1], self.cheri_avg["shap/seq"][j0:j1]),
            "chrombpnet": (self._cbp_block(self.cbp_avg["raw/seq"], j0, j1),
                           self._cbp_block(self.cbp_avg["shap/seq"], j0, j1)),
        }  # fmt: skip

    def finish(self) -> None:
        for reader in list(self.cheri_ohe.values()) + list(self.cheri_attr.values()):
            if reader.rows_read == reader.shape[0]:
                reader.finish()

    def close(self) -> None:
        for c in self.closers:
            try:
                c.close()
            except Exception:  # noqa: BLE001 - closing is best effort
                pass


def compare_dataset(args: argparse.Namespace, inp: dict, out_dir: Path) -> dict:
    """The pass over one dataset; writes the per-peak files and summary.tsv into out_dir; returns dataset.json."""
    import pandas as pd

    began = time.perf_counter()
    dataset = inp["dataset"]
    folds = [f["fold"] for f in inp["folds"]]
    roles = common.fold_roles(args.folds_dir, folds)

    # rows: Cherimoya's kept peaks, joined by name to ChromBPNet's bed (the same bed for every fold)
    idx = None
    for f in inp["folds"]:
        this = np.load(f["idx"], allow_pickle=False)
        if idx is None:
            idx = this
        elif not np.array_equal(this, idx):
            raise common.Failed(f"{dataset}: fold {f['fold']} keeps other peaks than fold {folds[0]}")
    peaks = common.read_bed(inp["loci"])
    if len(peaks) != idx.size:
        raise common.Failed(f"{dataset}: {inp['loci']} has {len(peaks)} rows, the idx mask {idx.size}")
    cheri_bed = peaks[idx].reset_index(drop=True)
    first_bed = Path(inp["folds"][0]["cbp_bed"]).read_bytes()
    for f in inp["folds"][1:]:
        if Path(f["cbp_bed"]).read_bytes() != first_bed:
            raise common.Failed(f"{dataset}: ChromBPNet's fold {f['fold']} bed differs from fold {folds[0]}'s")
    cbp_bed = common.read_bed(inp["folds"][0]["cbp_bed"])
    li, ri = common.join_by_name(cheri_bed, cbp_bed, f"{dataset} Cherimoya against ChromBPNet")
    if len(li) != len(cheri_bed):
        raise common.Failed(f"{dataset}: {len(cheri_bed) - len(li)} Cherimoya peaks are not in ChromBPNet's bed")
    if not np.all(np.diff(ri) > 0):
        raise common.Failed(f"{dataset}: joined by name, the rows are not in ChromBPNet's order")
    n = len(li) if args.max_rows is None else min(len(li), args.max_rows)
    sources = None
    try:
        import h5py

        with h5py.File(inp["folds"][0]["cbp_h5"], "r") as fh:
            length = fh["shap/seq"].shape[2]
        if args.cheri_source == "npz":
            width = int(attr_status.npz_header(Path(inp["folds"][0]["attr"]))["shape"][2])
        else:
            with h5py.File(inp["folds"][0]["h5"], "r") as fh:
                width = fh["shap/seq"].shape[2]
        c0 = column_offset(length, width)
        windows = [(h, *window_columns(width, h)) for h in args.windows]
        sources = Sources(inp, args.cheri_source, ri, c0, width)
        shapes = sources.shapes()
        for what, (raw_shape, shap_shape) in shapes.items():
            want_rows = len(cbp_bed) if what.startswith("chrombpnet") else len(cheri_bed)
            want = (want_rows, 4, length if what.startswith("chrombpnet") else width)
            if tuple(raw_shape) != want or tuple(shap_shape) != want:
                raise common.Failed(f"{dataset}: {what} is {tuple(shap_shape)} (raw {tuple(raw_shape)}), not {want}")
        logger.info("%s: %d peaks joined (%d ChromBPNet-only), W %d of L %d (column offset %d), windows %s, %s "
                    "source, %d rows%s", dataset, len(li), len(cbp_bed) - len(li), width, length, c0,
                    [2 * h for h in args.windows], args.cheri_source, n,
                    "" if n == len(li) else f" (--max-rows of {len(li)})")  # fmt: skip

        comps: dict[str, Columns] = {}

        def comp(name: str) -> Columns:
            if name not in comps:
                comps[name] = Columns(n)
            return comps[name]

        gc = {h: np.full(n, np.nan, dtype=np.float32) for h in args.windows}
        v1_rows = 0
        t_read = t_measure = 0.0
        for j0 in range(0, n, args.block_rows):
            j1 = min(n, j0 + args.block_rows)
            t = time.perf_counter()
            hyp: dict[tuple[str, str], np.ndarray] = {}
            ohe = None
            for k in folds:
                c_ohe, c_hyp = sources.cherimoya(k, j0, j1)
                b_raw, b_hyp = sources.chrombpnet(k, j0, j1)
                if not np.array_equal(c_ohe, b_raw):
                    bad = int((c_ohe != b_raw).any(axis=(1, 2)).sum())
                    raise common.Failed(f"{dataset} fold {k}: V1 fails on {bad} of rows {j0}-{j1}: Cherimoya's "
                                        f"one-hot is not ChromBPNet's raw/seq[:, :, {c0}:{c0 + width}]")  # fmt: skip
                if ohe is None:
                    ohe = c_ohe
                elif not np.array_equal(c_ohe, ohe):
                    raise common.Failed(
                        f"{dataset}: fold {k}'s one-hot differs from fold {folds[0]}'s in rows {j0}-{j1}"
                    )
                hyp[("cherimoya", k)] = c_hyp.astype(np.float32)
                hyp[("chrombpnet", k)] = b_hyp.astype(np.float32)
            for model, (raw, shap) in sources.averages(j0, j1).items():
                if not np.array_equal(raw, ohe):
                    raise common.Failed(f"{dataset}: {model}'s average raw/seq is not the folds' one-hot in rows "
                                        f"{j0}-{j1}")  # fmt: skip
                hyp[(model, "avg")] = shap.astype(np.float32)
            v1_rows += j1 - j0
            t_read += time.perf_counter() - t
            t = time.perf_counter()
            ohe32 = ohe.astype(np.float32)
            for h, a, b in windows:
                gc[h][j0:j1] = ohe32[:, 1:3, a:b].sum(axis=(1, 2)) / (b - a)
            maps = {key: Map(v, ohe32, args.top_k) for key, v in hyp.items()}
            sums = {m: sum(hyp[(m, k)] for k in folds) for m in common.MODELS}
            for k in folds:
                loo = {m: Map((sums[m] - hyp[(m, k)]) / (len(folds) - 1), ohe32, args.top_k) for m in common.MODELS}
                pairs = [(f"fold_{k}", maps[("chrombpnet", k)], maps[("cherimoya", k)], "chrombpnet", "cherimoya")]
                pairs += [(f"loo_{m}_{k}", maps[(m, k)], loo[m], "fold", "others") for m in common.MODELS]
                for name, x, y, xn, yn in pairs:
                    put_pair(comp(name), j0, x, y, xn, yn, windows)
                for m in common.MODELS:
                    maps[(m, k)].cache.clear()
            put_pair(comp("average"), j0, maps[("chrombpnet", "avg")], maps[("cherimoya", "avg")],
                     "chrombpnet", "cherimoya", windows)  # fmt: skip
            t_measure += time.perf_counter() - t
            if (j0 // args.block_rows) % 10 == 0:
                logger.info("%s: rows %d-%d of %d (read %.0f s, measures %.0f s so far)", dataset, j0, j1, n, t_read,
                            t_measure)  # fmt: skip
        if n == len(li):
            sources.finish()
    finally:
        if sources is not None:
            sources.close()

    # per-peak tables
    t = time.perf_counter()
    rows = cheri_bed.iloc[:n]
    base = pd.DataFrame({
        "peak": rows["name"].to_numpy(), "chrom": rows["chrom"].to_numpy(), "start": rows["start"].to_numpy(),
        "summit": (rows["start"] + rows["summit"]).to_numpy(), "signal": rows["signal"].to_numpy(),
    })  # fmt: skip
    for h in args.windows:
        base[f"gc_w{2 * h}"] = gc[h]
    chrom = base["chrom"].to_numpy()
    role_of = {k: np.array([roles[k].get(c, "none") for c in chrom], dtype=object) for k in folds}
    held = np.array([common.heldout_fold(roles, c) or "" for c in chrom], dtype=object)
    strata = strata_of(base, args.windows)

    def comparison_table(name: str) -> pd.DataFrame:
        df = base.copy()
        k = name.rsplit("_", 1)[-1] if name.startswith(("fold_", "loo_")) else None
        if k is not None:
            df.insert(5, "role", role_of[k])
        for col, values in comps[name].data.items():
            df[col] = values
        return df

    builders = [(name, lambda name=name: comparison_table(name)) for name in comps]
    builders += [(f"heldout_{m}", lambda m=m: heldout_table(base, comps, m, folds, held, role_of, args.windows))
                 for m in common.MODELS]  # fmt: skip
    builders.append(("heldout_cross", lambda: heldout_cross_table(base, comps, folds, held, args.windows)))
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, build in builders:  # one table in memory at a time
        df = build()
        common.write_table(out_dir / f"{name}.per_peak.tsv.gz", df, float_format="%.5g")
        summary.extend(summarize_table(name, df, strata))
        del df
    common.write_table(out_dir / "summary.tsv", pd.DataFrame(summary))
    t_write = time.perf_counter() - t

    info = {
        "dataset": dataset,
        "cheri_source": args.cheri_source,
        "folds": folds,
        "peaks_file_rows": int(idx.size),
        "cherimoya_rows": int(len(cheri_bed)),
        "chrombpnet_rows": int(len(cbp_bed)),
        "joined_rows": int(len(li)),
        "compared_rows": int(n),
        "chrombpnet_only_rows": int(len(cbp_bed) - len(li)),
        "width": int(width),
        "length": int(length),
        "column_offset": int(c0),
        "windows": {f"w{2 * h}": [int(a), int(b)] for h, a, b in windows},
        "top_k": args.top_k,
        "v1_rows_checked_per_fold": int(v1_rows),
        "heldout_rows_without_test_fold": int((held == "").sum()),
        "timings_s": {
            "read": round(t_read, 1),
            "measures": round(t_measure, 1),
            "write": round(t_write, 1),
            "total": round(time.perf_counter() - began, 1),
        },  # fmt: skip
        "rss_max_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
    common.write_json(out_dir / "dataset.json", info)
    return info


def put_pair(cols: Columns, j0: int, x: Map, y: Map, xn: str, yn: str, windows) -> None:
    for h, a, b in windows:
        w = f"w{2 * h}"
        for metric, values in pair_measures(x, y, a, b).items():
            cols.put(f"{metric}_{w}", j0, values)
        for name, mp in ((xn, x), (yn, y)):
            cols.put(f"sum_{name}_{w}", j0, mp.feature("sum", a, b))
            cols.put(f"l2_{name}_{w}", j0, mp.feature("l2", a, b))


def heldout_table(base, comps: dict, model: str, folds: list[str], held: np.ndarray, role_of: dict, halves):
    """Per peak, the model's leave-one-out agreement at its held-out fold, and its mean over the folds whose training
    set held the peak's chromosome."""
    df = base.copy()
    df.insert(5, "heldout_fold", held)
    fold_pos = {k: i for i, k in enumerate(folds)}
    hi = np.array([fold_pos.get(h, -1) for h in held])
    rows = np.arange(len(held))
    train = np.stack([role_of[k] == "train" for k in folds])  # (folds, n)
    df["n_train_folds"] = train.sum(axis=0)
    for h in halves:
        for metric in AGREEMENT:
            col = f"{metric}_w{2 * h}"
            stack = np.stack([comps[f"loo_{model}_{k}"].data[col] for k in folds]).astype(np.float64)
            heldout = np.where(hi >= 0, stack[np.clip(hi, 0, None), rows], np.nan)
            with np.errstate(invalid="ignore", divide="ignore"):
                trained = np.where(train, stack, 0.0).sum(axis=0) / train.sum(axis=0)
            df[f"heldout_{col}"] = heldout
            df[f"train_{col}"] = trained
    return df


def heldout_cross_table(base, comps: dict, folds: list[str], held: np.ndarray, halves):
    """Per peak, the two models each on the fold that held the peak's chromosome out."""
    df = base.copy()
    df.insert(5, "heldout_fold", held)
    fold_pos = {k: i for i, k in enumerate(folds)}
    hi = np.array([fold_pos.get(h, -1) for h in held])
    rows = np.arange(len(held))
    names = [c for c in comps[f"fold_{folds[0]}"].data]
    for col in names:
        stack = np.stack([comps[f"fold_{k}"].data[col] for k in folds]).astype(np.float64)
        df[col] = np.where(hi >= 0, stack[np.clip(hi, 0, None), rows], np.nan)
    return df


def deciles_of(values: np.ndarray) -> np.ndarray:
    """1..10 by rank (equal-count bins, ties in row order); 0 where the value is not finite."""
    out = np.zeros(len(values), dtype=np.int64)
    ok = np.isfinite(values)
    order = np.argsort(values[ok], kind="stable")
    d = np.empty(ok.sum(), dtype=np.int64)
    d[order] = np.arange(ok.sum()) * N_DECILES // max(int(ok.sum()), 1) + 1
    out[ok] = d
    return out


def strata_of(base, halves) -> dict[str, np.ndarray]:
    widest = max(halves)
    return {"signal_decile": deciles_of(base["signal"].to_numpy(dtype=np.float64)),
            "gc_decile": deciles_of(base[f"gc_w{2 * widest}"].to_numpy(dtype=np.float64))}  # fmt: skip


def describe(values: np.ndarray) -> dict:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan, "q25": np.nan, "q75": np.nan}
    q25, med, q75 = np.quantile(v, [0.25, 0.5, 0.75])
    return {"n": int(v.size), "mean": float(v.mean()), "median": float(med), "q25": float(q25), "q75": float(q75)}


def summarize_table(name: str, df, strata: dict) -> list[dict]:
    """summary.tsv rows of one per-peak table: every measure column, over all peaks and within each stratum."""
    skip = {"peak", "chrom", "start", "summit", "signal", "role", "heldout_fold", "n_train_folds"}
    measures = [c for c in df.columns if c not in skip and not c.startswith("gc_w")]
    groups: list[tuple[str, str, np.ndarray]] = [("all", "all", np.ones(len(df), dtype=bool))]
    if "role" in df:
        for r in common.ROLES:
            groups.append(("role", r, (df["role"] == r).to_numpy()))
    for kind, values in strata.items():
        for d in range(1, N_DECILES + 1):
            groups.append((kind, str(d), values == d))
    rows = []
    for col in measures:
        metric, _, window = col.rpartition("_")
        values = df[col].to_numpy(dtype=np.float64)
        for kind, level, sel in groups:
            rows.append({"comparison": name, "window": window, "measure": metric, "stratum": kind, "level": level,
                         **describe(values[sel])})  # fmt: skip
    return rows


# ---------------------------------------------------------------------------------------------------------------------
# across datasets


def per_dataset_quantities(summary, folds_by_ds: dict) -> dict:
    """{(window, metric): {quantity: {dataset: value}}} from the datasets' summary rows (stratum all unless noted)."""
    import pandas as pd

    out: dict = {}
    s = summary
    held_measures = [f"{w}_{m}" for w in ("heldout", "train") for m in AGREEMENT]
    for (window, metric), part in s[s["measure"].isin(AGREEMENT)].groupby(["window", "measure"]):
        part = pd.concat([part, s[(s["window"] == window) & s["measure"].isin(held_measures)]])
        q: dict[str, dict[str, float]] = {}

        def put(quantity: str, ds: str, value: float, q: dict = q) -> None:
            q.setdefault(quantity, {})[ds] = value

        for ds, d in part.groupby("dataset"):
            allrows = d[d["stratum"] == "all"].set_index("comparison")["median"]
            role_rows = d[d["stratum"] == "role"]
            ks = folds_by_ds.get(ds, [])
            fold_vals = [allrows.get(f"fold_{k}") for k in ks]
            if fold_vals and all(v is not None and pd.notna(v) for v in fold_vals):
                put("cross_fold", ds, float(np.mean(fold_vals)))
            if "average" in allrows:
                put("cross_average", ds, float(allrows["average"]))
            for m in common.MODELS:
                vals = [allrows.get(f"loo_{m}_{k}") for k in ks]
                if vals and all(v is not None and pd.notna(v) for v in vals):
                    put(f"loo_{m}", ds, float(np.mean(vals)))
            for m in common.MODELS:
                ho = d[(d["comparison"] == f"heldout_{m}") & (d["stratum"] == "all")].set_index("measure")["median"]
                if f"heldout_{metric}" in ho and f"train_{metric}" in ho:
                    put(f"heldout_{m}", ds, float(ho[f"heldout_{metric}"]))
                    put(f"train_{m}", ds, float(ho[f"train_{metric}"]))
            for r in ("test", "train"):
                vals = role_rows[(role_rows["level"] == r) & role_rows["comparison"].isin([f"fold_{k}" for k in ks])]
                if len(vals) == len(ks) and len(ks):
                    put(f"cross_{'heldout' if r == 'test' else 'train'}", ds, float(vals["median"].mean()))
        out[(window, metric)] = q
    return out


PAIRED_TESTS = (
    ("cross_fold", "loo_chrombpnet"),
    ("cross_fold", "loo_cherimoya"),
    ("cross_average", "loo_chrombpnet"),
    ("cross_average", "loo_cherimoya"),
    ("loo_cherimoya", "loo_chrombpnet"),
    ("heldout_chrombpnet", "train_chrombpnet"),
    ("heldout_cherimoya", "train_cherimoya"),
    ("cross_heldout", "cross_train"),
)


def statistics_rows(quantities: dict, fold_rows, n_boot: int, seed: int) -> list[dict]:
    rows = []
    for (window, metric), q in sorted(quantities.items()):
        for name, values in sorted(q.items()):
            est, lo, hi = common.bootstrap_ci(list(values.values()), np.median, n_boot=n_boot, seed=seed)
            rows.append({"window": window, "metric": metric, "test": "level", "quantity": name, "over": "datasets",
                         "n": len(values), "median": est, "ci95_lo": lo, "ci95_hi": hi, "wilcoxon_p": np.nan,
                         "first_higher": np.nan})  # fmt: skip
        for a, b in PAIRED_TESTS:
            if a not in q or b not in q:
                continue
            common_ds = sorted(set(q[a]) & set(q[b]))
            diffs = [q[a][ds] - q[b][ds] for ds in common_ds]
            est, lo, hi = common.bootstrap_ci(diffs, np.median, n_boot=n_boot, seed=seed)
            _, p, _ = common.wilcoxon(diffs)
            rows.append({"window": window, "metric": metric, "test": "paired", "quantity": f"{a} - {b}",
                         "over": "datasets", "n": len(diffs), "median": est, "ci95_lo": lo, "ci95_hi": hi,
                         "wilcoxon_p": p, "first_higher": int(sum(d > 0 for d in diffs))})  # fmt: skip
    # over folds: A_k against each model's C_k (dataset x fold)
    if fold_rows is not None and len(fold_rows):
        for (window, metric), part in fold_rows.groupby(["window", "measure"]):
            wide = part.pivot_table(index=["dataset", "fold"], columns="kind", values="median")
            for m in common.MODELS:
                if "cross" not in wide or m not in wide:
                    continue
                diffs = (wide["cross"] - wide[m]).dropna().to_numpy()
                est, lo, hi = common.bootstrap_ci(diffs, np.median, n_boot=n_boot, seed=seed)
                _, p, _ = common.wilcoxon(diffs)
                rows.append({"window": window, "metric": metric, "test": "paired", "quantity": f"cross_fold - loo_{m}",
                             "over": "folds", "n": int(diffs.size), "median": est, "ci95_lo": lo, "ci95_hi": hi,
                             "wilcoxon_p": p, "first_higher": int((diffs > 0).sum())})  # fmt: skip
    return rows


def fold_level_rows(summary):
    """Per dataset, fold, window and agreement metric: the cross-model median (kind 'cross') and each model's
    leave-one-out median (kind = the model), stratum all."""
    s = summary[(summary["stratum"] == "all") & summary["measure"].isin(AGREEMENT)]
    s = s[s["comparison"].str.match(r"^(fold|loo_[a-z]+)_[^_]+$")].copy()
    s["fold"] = s["comparison"].str.rsplit("_", n=1).str[-1]
    s["kind"] = np.where(s["comparison"].str.startswith("fold_"), "cross",
                         s["comparison"].str.extract(r"^loo_([a-z]+)_")[0])  # fmt: skip
    return s


def plot_agreement(quantities: dict, out_dir: Path, datasets: list[str]) -> list[Path]:
    plt = common.matplotlib_pyplot()
    styles, labels = common.dataset_styles(datasets), common.short_names(datasets)
    order = [("cross_fold", "cross-model\nsame fold (A)"), ("cross_average", "cross-model\nfold averages (B)"),
             ("loo_chrombpnet", "ChromBPNet\nleave-one-out (C)"), ("loo_cherimoya", "Cherimoya\nleave-one-out (C)"),
             ("cross_heldout", "cross-model\nheld-out fold (D)")]  # fmt: skip
    paths = []
    for window in sorted({w for w, _ in quantities}, key=lambda w: int(w[1:])):
        fig, axes = plt.subplots(1, len(AGREEMENT), figsize=(3.4 * len(AGREEMENT), 3.8), squeeze=False)
        for ax, metric in zip(axes[0], AGREEMENT):
            q = quantities.get((window, metric), {})
            for x, (name, _) in enumerate(order):
                values = q.get(name, {})
                for i, ds in enumerate(sorted(values)):
                    colour, marker = styles.get(ds, ("#999999", "o"))
                    jitter = (i - (len(values) - 1) / 2) * 0.03
                    ax.scatter(x + jitter, values[ds], s=16, color=colour, marker=marker, edgecolor="#333333",
                               linewidth=0.3, zorder=2)  # fmt: skip
                if values:
                    med = float(np.median(list(values.values())))
                    ax.plot([x - 0.3, x + 0.3], [med, med], color="#333333", lw=1.2, zorder=3)
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels([label for _, label in order], fontsize=6, rotation=45, ha="right")
            ax.set_title(f"{metric}, {window[1:]} bp around the summit", fontsize=8)
            ax.set_ylabel(f"median per-peak {metric}")
            ax.grid(axis="x", visible=False)
        common.dataset_legend(fig, styles, labels, f"{len(datasets)} datasets")
        fig.suptitle("Attribution agreement: across the models next to each model's own ceiling", fontsize=9)
        fig.tight_layout()
        paths += common.save_figure(fig, out_dir / f"agreement_{window}")
        plt.close(fig)

        fig, axes = plt.subplots(1, len(AGREEMENT), figsize=(3.4 * len(AGREEMENT), 3.6), squeeze=False)
        pairs = [("train_chrombpnet", "heldout_chrombpnet", "ChromBPNet"),
                 ("train_cherimoya", "heldout_cherimoya", "Cherimoya"),
                 ("cross_train", "cross_heldout", "cross-model")]  # fmt: skip
        for ax, metric in zip(axes[0], AGREEMENT):
            q = quantities.get((window, metric), {})
            for x, (a, b, _) in enumerate(pairs):
                for ds in sorted(set(q.get(a, {})) & set(q.get(b, {}))):
                    colour, marker = styles.get(ds, ("#999999", "o"))
                    ya, yb = q[a][ds], q[b][ds]
                    ax.plot([2 * x, 2 * x + 1], [ya, yb], color=colour, lw=0.8, zorder=1)
                    ax.scatter([2 * x, 2 * x + 1], [ya, yb], s=14, color=colour, marker=marker, edgecolor="#333333",
                               linewidth=0.3, zorder=2)  # fmt: skip
            ax.set_xticks(range(2 * len(pairs)))
            ax.set_xticklabels([f"{lab}\n{w}" for _, _, lab in pairs for w in ("training", "held-out")], fontsize=6,
                               rotation=45, ha="right")  # fmt: skip
            ax.set_title(f"{metric}, {window[1:]} bp", fontsize=8)
            ax.set_ylabel(f"median per-peak {metric}")
            ax.grid(axis="x", visible=False)
        common.dataset_legend(fig, styles, labels, f"{len(datasets)} datasets")
        fig.suptitle("Agreement on the peaks a fold trained on against the peaks it held out", fontsize=9)
        fig.tight_layout()
        paths += common.save_figure(fig, out_dir / f"heldout_{window}")
        plt.close(fig)
    return paths


def summarize(args: argparse.Namespace, out_dir: Path) -> int:
    import pandas as pd

    datasets = common.read_list(args.datasets)
    done, waiting = [], []
    inputs: dict[str, Path] = {"list": args.datasets}
    for ds in datasets:
        marker = out_dir / ".done" / f"{ds}.json"
        if marker.is_file() and (out_dir / ds / "summary.tsv").is_file():
            done.append(ds)
            inputs[f"{ds}/marker"] = marker
        else:
            waiting.append(ds)
    tagged = sorted(p for p in (out_dir / ".done").glob("*.*.json")) if (out_dir / ".done").is_dir() else []
    for p in tagged:
        inputs[f"tagged/{p.stem}"] = p
    if not done:
        common.result(f"waiting: no dataset done yet (of {len(datasets)})")
        return common.WAITING
    settings = {"datasets": datasets, "n_boot": args.n_boot, "seed": args.seed}
    marker = common.Marker(out_dir, "summary", FORMAT, inputs, settings)
    why = marker.current()
    if why is None and not args.force and not waiting:
        common.result("up to date")
        return 0
    marker.remove()
    parts, folds_by_ds = [], {}
    for ds in done:
        part = pd.read_csv(out_dir / ds / "summary.tsv", sep="\t", keep_default_na=False, na_values=["NA"])
        part.insert(0, "dataset", ds)
        parts.append(part)
        info = json.loads((out_dir / ds / "dataset.json").read_text())
        folds_by_ds[ds] = info["folds"]
    summary = pd.concat(parts, ignore_index=True)
    summary["level"] = summary["level"].astype(str)
    common.write_table(out_dir / "summary.tsv", summary)
    quantities = per_dataset_quantities(summary, folds_by_ds)
    stats = statistics_rows(quantities, fold_level_rows(summary), args.n_boot, args.seed)
    common.write_table(out_dir / "statistics.tsv", pd.DataFrame(stats))
    outputs = {"summary": out_dir / "summary.tsv", "statistics": out_dir / "statistics.tsv"}

    f16 = []
    for p in tagged:
        ds, tag = p.stem.rsplit(".", 1)
        tagged_summary = out_dir / ds / tag / "summary.tsv"
        if ds not in done or not tagged_summary.is_file():
            continue
        a = summary[(summary["dataset"] == ds) & (summary["stratum"] == "all")]
        b = pd.read_csv(tagged_summary, sep="\t", keep_default_na=False, na_values=["NA"])
        b = b[b["stratum"] == "all"]
        m = a.merge(b, on=["comparison", "window", "measure", "stratum", "level"], suffixes=("_float32", f"_{tag}"))
        m = m[m["measure"].isin(AGREEMENT)]
        m["median_diff"] = m[f"median_{tag}"] - m["median_float32"]
        f16.append(m[["dataset", "comparison", "window", "measure", "n_float32", "median_float32", f"n_{tag}",
                      f"median_{tag}", "median_diff"]])  # fmt: skip
    if f16:
        common.write_table(out_dir / "float16.tsv", pd.concat(f16, ignore_index=True))
        outputs["float16"] = out_dir / "float16.tsv"
    for path in plot_agreement(quantities, out_dir, datasets):
        outputs[path.name] = path

    narrowest = min({w for w, _ in quantities}, key=lambda w: int(w[1:]))
    key = (narrowest, "pearson")
    q = quantities.get(key, {})

    def med(name: str) -> str:
        return f"{np.median(list(q[name].values())):.3f}" if q.get(name) else "NA"

    note = (f"{len(done)} of {len(datasets)} datasets: {key[1]} ({key[0]}) cross-model same fold {med('cross_fold')}, "
            f"averages {med('cross_average')}; ceilings ChromBPNet {med('loo_chrombpnet')}, Cherimoya "
            f"{med('loo_cherimoya')}")  # fmt: skip
    if waiting:
        note += f"; waiting for {len(waiting)}: {', '.join(waiting[:3])}{' ...' if len(waiting) > 3 else ''}"
        common.result(f"waiting: {note}")
        return common.WAITING
    marker.write(outputs, note)
    common.result(note)
    return 0


# ---------------------------------------------------------------------------------------------------------------------


def run_dataset(args: argparse.Namespace, out_root: Path) -> int:
    import pandas as pd

    ds = args.dataset
    tag = args.tag or ("" if args.cheri_source == "npz" else args.cheri_source)
    unit = f"{ds}.{tag}" if tag else ds
    out_dir = out_root / ds / tag if tag else out_root / ds
    inp = resolve(args, ds)
    settings = {"windows": args.windows, "top_k": args.top_k, "cheri_source": args.cheri_source,
                "max_rows": args.max_rows}  # fmt: skip
    marker = common.Marker(out_root, unit, FORMAT, input_paths(inp, args.cheri_source), settings)
    why = marker.current()
    if why is None and not args.force:
        common.result("up to date")
        return 0
    logger.info("%s: %s", unit, why or "forced")
    marker.remove()
    info = compare_dataset(args, inp, out_dir)
    outputs = {p.name: p for p in sorted(out_dir.glob("*.tsv*"))}
    outputs["dataset.json"] = out_dir / "dataset.json"
    summary = pd.read_csv(out_dir / "summary.tsv", sep="\t")
    pick = summary[(summary["stratum"] == "all") & (summary["measure"] == "pearson")]
    w = f"w{2 * min(args.windows)}"

    def med(comparison: str) -> str:
        row = pick[(pick["comparison"] == comparison) & (pick["window"] == w)]
        return f"{row['median'].iloc[0]:.3f}" if len(row) else "NA"

    k0 = info["folds"][0]
    note = (f"{info['compared_rows']} peaks ({info['chrombpnet_only_rows']} ChromBPNet-only), {w} Pearson: fold "
            f"{k0} {med(f'fold_{k0}')}, averages {med('average')}, ceilings {med(f'loo_chrombpnet_{k0}')} / "
            f"{med(f'loo_cherimoya_{k0}')}; {info['timings_s']['total']:.0f} s")  # fmt: skip
    marker.write(outputs, note, {"dataset": info})
    common.result(note)
    return 0


def parse_windows(text: str) -> list[int]:
    try:
        values = sorted({int(v) for v in text.split(",") if v.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--windows takes half-widths like 250,500, got {text!r}") from exc
    if not values or values[0] <= 0:
        raise argparse.ArgumentTypeError("--windows needs positive half-widths")
    return values


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_root_args(ap)
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--dataset", help="compare this dataset")
    what.add_argument("--summarize", action="store_true", help="summarize the datasets of --datasets")
    ap.add_argument("--datasets", type=Path, help="the dataset list (for --summarize)")
    ap.add_argument("--out-dir", type=Path, help="default <comparison-root>/attributions")
    ap.add_argument("--windows", type=parse_windows, default=[250, 500], help="half-widths around the summit")
    ap.add_argument("--top-k", type=int, default=50, help="positions in the Jaccard index (default 50)")
    ap.add_argument("--block-rows", type=int, default=4096, help="peaks per block (default 4096)")
    ap.add_argument("--cheri-source", choices=("npz", "h5"), default="npz",
                    help="Cherimoya's per-fold maps: the float32 npz (default) or the converted float16 h5")  # fmt: skip
    ap.add_argument("--tag", help="a name for this variant's outputs (default: none for npz, the source for h5)")
    ap.add_argument("--max-rows", type=int, help="only the first N joined peaks (a smoke test)")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    common.setup_logging(args.verbose)
    if args.out_dir is None:
        common.require(args, "comparison_root")
    out_root = args.out_dir or args.comparison_root / "attributions"
    try:
        if args.summarize:
            if args.datasets is None:
                ap.error("--summarize needs --datasets")
            return summarize(args, out_root)
        common.require(args, "cheri_root", "folds_dir")
        if args.block_rows < 1 or args.top_k < 1:
            ap.error("--block-rows and --top-k must be positive")
        return run_dataset(args, out_root)
    except common.Waiting as exc:
        common.result(f"waiting: {exc}")
        return common.WAITING
    except (common.Failed, tcs.Refused) as exc:
        logger.error("%s", exc)
        common.result(f"failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
