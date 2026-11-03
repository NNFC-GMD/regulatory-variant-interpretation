#!/usr/bin/env python3
"""DeepISA on the same sites under every dataset's models: the changes between groups and the series' trends, with a
paired bootstrap.

combine.py compares each dataset's own sites under its own models, so a change there mixes a change of grammar with a
change of which sites are bound and where. Here both groups' models score the same site pairs: matrix.py runs
coop_isa.py with every dataset's sites (--finemo-dir) under every dataset's models (--models), into
<matrix>/<sites>/<model>/. coop_isa.py's sites do not depend on the models, so the rows line up; this script checks
that they do.

Rows: the union of every dataset's site pairs. A pair found in several datasets' sites (the same chromosome, starts
and motifs) is kept once, from the first dataset in the groups' order; pairs whose sites differ by a few bases stay
apart.

Open windows: in a window a model predicts closed, every effect is small whatever the grammar, and a pair there would
look independent. A window is open under a model when its intact prediction a reaches the --open-quantile (0.1) of a
over the windows of that model's own sites. A comparison uses the pairs whose window is open under every model of both
groups; a series, under every model of the series. The bootstrap resamples windows (each dataset's sites' windows
apart), with the same weights for every model: the pairs are the same, so their noise largely cancels.

Per comparison and TF pair (or TF), with >= --min-n pairs (each group's pairs are the same):
  synergy   combine.py's changes() on these rows, paired (combine.synergy_boot's units): synergy_a, synergy_b, delta
            with CI, p, q, significant, the fold check, robust
  delta_i   the mean over the pairs of i(b) - i(a), each group's i the mean over its datasets' models (mean_i_a,
            mean_i_b): no threshold, so a weaker model's smaller effects do not move pairs across 0.1. CI, p and q (BH
            within the comparison) from the paired bootstrap; the fold check (delta_i_folds_agree, from i_f<k>);
            delta_i_significant, delta_i_robust
Per series and TF pair (or TF): combine.py's trend() on these rows, paired; and the slope of the mean i over the series
(mean_i_<group>, slope_i with CI, p, q, the fold check, slope_i_significant, slope_i_robust).

Activity, beside every score: a TF's mean e over its own sites in these pairs (a TF pair's: the mean of its two
sites'), each site's e the mean over the group's models: mean_e_a, mean_e_b and delta_e per comparison, mean_e_<group>
and slope_e per series. summary.json's activity_r: per comparison (and for the series), the correlation over the
tested TF pairs (or TFs) of each change (or slope) with that of the activity.

--adjust takes the sites' strength out of i before the same statistics, into --out-dir (combined/ is never written).
In log counts two strong sites are sub-additive even far apart (the model saturates), so a TF whose sites get stronger
looks less synergistic, and one whose sites weaken more. Each model's i (and i_f<k>) becomes:
  ratio    the official package's normalised interaction (deepISA calc_interaction(normalize=True)): i / (e_a + e_b +
           tau) on its qualified pairs: both e at least the 10th percentile of the non-negative e over the model's own
           sites (matrix.py's <model>/<model>/sites.tsv.gz), both effects given the other ablated >= 0. A comparison
           (or the series) uses the pairs qualified under all its models, so a motif that is inactive under any of
           them drops out. The package's tau is the median e_a + e_b of non-motif ablations, which coop_isa.py does not
           run: --tau (default 0). No synergy: its 0.1 is in log counts. i_f<k> is divided by the fold-mean e.
  distant  i minus the mean i of the pairs more than --far-gap (150) bp apart, the package's null, whose sites are as
           strong: per model, over its open pairs, in cells of the weaker site's e (--lo-bins quantiles, pooled over
           the models so that a cell is the same under each) and, within each, of the stronger site's (--hi-bins
           quantiles). A cell with fewer than --min-cell distant pairs takes its row's mean. i_f<k>: the same, with
           fold k's i. curves.tsv: per model and cell, the mean i of the pairs adjacent (gap <= 5 bp), near (<= 50),
           mid (<= --far-gap) and far, and the value subtracted.
The value subtracted (or divided by) is fixed, not refitted in each bootstrap replicate. The output columns keep
their names (mean_i_a, delta_i, slope_i, ...); summary.json's adjust says what i is.

Out (--out-dir, default <matrix>/combined/): changes_pairs.tsv, changes_tfs.tsv, with --series trend_pairs.tsv and
trend_tfs.tsv, open.tsv (per model: its threshold, its windows and how many are open), summary.json; with distant,
curves.tsv.

    paired.py --matrix <dir> --motifs <motifs.tsv> --groups <groups.tsv> --comparisons <comparisons.tsv>
        [--series <series.tsv>] [--adjust ratio|distant --out-dir <dir>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import combine as cb  # noqa: E402

ID = ["peak_id", "chr", "motif_a", "motif_b", "start_a", "start_b", "distance"]
"""A pair's columns that do not depend on the model: equal under every model of one dataset's sites."""
EXTRA = ["e_a", "e_b", "isa_both", "gap"]
"""coop_isa.py's columns kept beside i when the tables have them: the activity and --adjust read them."""


def load(matrix: Path, datasets: list[str], names: dict, open_quantile: float):
    """({model: its pairs over every dataset's sites, aligned row by row}, the pairs' identity table, {model: the
    threshold of an open window, its windows, open})."""
    per_model = {m: [] for m in datasets}
    ident, thresholds = [], {}
    for x in datasets:
        ref = None
        for m in datasets:
            f = matrix / x / m / "pairs.tsv.gz"
            if not f.is_file():
                raise FileNotFoundError(f"no {f}: matrix.py has not scored {x}'s sites with {m}'s models")
            df = pd.read_csv(f, sep="\t")
            if ref is None:
                ref = df[ID]
            elif not df[ID].equals(ref):
                raise ValueError(f"{x}'s sites: the pairs under {m}'s models are not those under {datasets[0]}'s")
            if x == m:
                a = df.drop_duplicates("peak_id")["a"]
                thresholds[m] = {"threshold": float(a.quantile(open_quantile)), "own_windows": len(a)}
            per_model[m].append(df)
        ident.append(ref.assign(sites=x))
    ident = pd.concat(ident, ignore_index=True)
    keep = ~ident.duplicated(subset=["chr", "start_a", "start_b", "motif_a", "motif_b"]).to_numpy()
    ident = ident[keep].reset_index(drop=True)
    ident["unit"] = ident["sites"] + ":" + ident["peak_id"].astype(str)
    for x in ("a", "b"):  # each site's TF from its motif, as combine.py names them
        ident[f"tf_{x}"] = ident[f"motif_{x}"].map(names).fillna(ident[f"motif_{x}"])
    sw = (ident["tf_a"] > ident["tf_b"]).to_numpy()
    ident.loc[sw, ["tf_a", "tf_b"]] = ident.loc[sw, ["tf_b", "tf_a"]].to_numpy()
    ident["pair"] = ident["tf_a"] + " + " + ident["tf_b"]
    tables = {}
    for m, dfs in per_model.items():
        df = pd.concat(dfs, ignore_index=True)[keep].reset_index(drop=True)
        cols = ["i", "a", *cb.fold_columns([df]), *[c for c in EXTRA if c in df]]
        tables[m] = pd.concat([ident[["unit", "sites", "peak_id", "tf_a", "tf_b", "pair", "distance"]], df[cols]],
                              axis=1)  # fmt: skip
        if "e_a" in df:  # each side's effect beside its TF's name, which the swap above may have moved
            tables[m].loc[sw, ["e_a", "e_b"]] = tables[m].loc[sw, ["e_b", "e_a"]].to_numpy()
        windows = df.assign(unit=ident["unit"]).drop_duplicates("unit")
        thresholds[m]["windows"] = len(windows)
        thresholds[m]["open"] = int((windows["a"] >= thresholds[m]["threshold"]).sum())
    return tables, ident, thresholds


def long_rows(base: pd.DataFrame, key: str, V: np.ndarray):
    """The labels, units and values (columns = rows of base) at `key`'s level: at the TF level each pair counts for
    both its TFs, as in combine.py."""
    if key == "tf":
        return (np.r_[base["tf_a"].to_numpy(), base["tf_b"].to_numpy()], np.r_[base["unit"].to_numpy(),
                base["unit"].to_numpy()], np.concatenate([V, V], axis=-1))  # fmt: skip
    return base[key].to_numpy(), base["unit"].to_numpy(), V


def boot_means(labels, unit_codes: np.ndarray, n_units: int, V: np.ndarray, n_boot: int, seed: int):
    """Per label: n, the mean of each row of V over its rows (labels x rows of V), and n_boot replicates with Poisson
    weights per unit (the same draws as combine.synergy_boot's with the same seed and units)."""
    codes, uniq = pd.factorize(labels)
    k = len(uniq)
    n = np.bincount(codes, minlength=k)
    point = np.stack([np.bincount(codes, v, k) for v in V], axis=1) / np.maximum(n, 1)[:, None]
    rng = np.random.default_rng(seed)
    reps = np.empty((n_boot, k, len(V)))
    for b in range(n_boot):
        w = rng.poisson(1.0, n_units)[unit_codes].astype(float)
        den = np.bincount(codes, w, k)
        with np.errstate(divide="ignore", invalid="ignore"):
            reps[b] = np.stack([np.bincount(codes, w * v, k) for v in V], axis=1) / den[:, None]
    return uniq, n, point, reps


def fold_means(labels, V_folds: list[np.ndarray]) -> list[np.ndarray]:
    codes, uniq = pd.factorize(labels)
    n = np.maximum(np.bincount(codes, minlength=len(uniq)), 1)
    return [np.stack([np.bincount(codes, v, len(uniq)) for v in V], axis=1) / n[:, None] for V in V_folds]


def group_i(sub: dict, models: list[str], col: str) -> np.ndarray:
    return np.mean([sub[m][col].to_numpy() for m in models], axis=0)


def delta_i(sub: dict, base: pd.DataFrame, ma: list, mb: list, key: str, units: pd.Index, args) -> pd.DataFrame:
    """Per TF pair (or TF): mean_i_a, mean_i_b, delta_i with its paired bootstrap CI, p, q and fold check."""
    V = np.stack([group_i(sub, ma, "i"), group_i(sub, mb, "i")])
    labels, unit_labels, V = long_rows(base, key, V)
    uniq, n, point, reps = boot_means(labels, units.get_indexer(unit_labels), len(units), V, args.n_boot, 11)
    out = pd.DataFrame({key: uniq, "n_pairs": n, "mean_i_a": point[:, 0], "mean_i_b": point[:, 1]})
    out["delta_i"] = point[:, 1] - point[:, 0]
    d = reps[:, :, 1] - reps[:, :, 0]
    out["delta_i_lo"], out["delta_i_hi"] = cb.quiet(np.nanpercentile, d, 2.5, axis=0), cb.quiet(np.nanpercentile, d, 97.5, axis=0)  # fmt: skip
    out["delta_i_p"] = cb.boot_p(out["delta_i"], d)
    cols = cb.fold_columns(sub.values())
    if cols:
        Vf = [long_rows(base, key, np.stack([group_i(sub, ma, c), group_i(sub, mb, c)]))[2] for c in cols]
        per_fold = np.stack([f[:, 1] - f[:, 0] for f in fold_means(labels, Vf)])
        cb.add_fold_check(out, "delta_i", per_fold)
        out = out.rename(columns={"folds_agree": "delta_i_folds_agree"}).drop(columns="n_folds")
    out = out[out["n_pairs"] >= args.min_n].copy()
    out["delta_i_q"] = cb.bh(out["delta_i_p"])
    out["delta_i_significant"] = out["delta_i_q"] < args.fdr
    if cols:
        out["delta_i_robust"] = out["delta_i_significant"] & (out["delta_i_folds_agree"] == len(cols))
    return out


def slope_i(sub: dict, base: pd.DataFrame, groups: dict, series: list, key: str, units: pd.Index, args):
    """Per TF pair (or TF): the mean i in each group of the series, and its slope with CI, p, q and fold check."""
    V = np.stack([group_i(sub, groups[g], "i") for g in series])
    labels, unit_labels, V = long_rows(base, key, V)
    uniq, n, point, reps = boot_means(labels, units.get_indexer(unit_labels), len(units), V, args.n_boot, 21)
    x = np.arange(len(series), dtype=float)
    w = (x - x.mean()) / ((x - x.mean()) ** 2).sum()
    out = pd.DataFrame({key: uniq, "n_pairs": n})
    for j, g in enumerate(series):
        out[f"mean_i_{g}"] = point[:, j]
    out["slope_i"] = point @ w
    r = reps @ w
    out["slope_i_lo"], out["slope_i_hi"] = cb.quiet(np.nanpercentile, r, 2.5, axis=0), cb.quiet(np.nanpercentile, r, 97.5, axis=0)  # fmt: skip
    out["slope_i_p"] = cb.boot_p(out["slope_i"], r)
    cols = cb.fold_columns(sub.values())
    if cols:
        Vf = [long_rows(base, key, np.stack([group_i(sub, groups[g], c) for g in series]))[2] for c in cols]
        per_fold = np.stack([f @ w for f in fold_means(labels, Vf)])
        cb.add_fold_check(out, "slope_i", per_fold)
        out = out.rename(columns={"folds_agree": "slope_i_folds_agree"}).drop(columns="n_folds")
    out = out[out["n_pairs"] >= args.min_n].copy()
    out["slope_i_q"] = cb.bh(out["slope_i_p"])
    out["slope_i_significant"] = out["slope_i_q"] < args.fdr
    if cols:
        out["slope_i_robust"] = out["slope_i_significant"] & (out["slope_i_folds_agree"] == len(cols))
    return out


def site_thresholds(matrix: Path, models: list[str], quantile: float = 0.1) -> dict:
    """Per model: the package's isa_thresh (deepISA get_isa_thresh), the `quantile` of the non-negative e over the
    model's own sites."""
    out = {}
    for m in models:
        e = pd.read_csv(matrix / m / m / "sites.tsv.gz", sep="\t", usecols=["e"])["e"]
        out[m] = float(e[e >= 0].quantile(quantile))
    return out


def adjust_ratio(tables: dict, thresholds: dict, tau: float) -> dict:
    """In place, per model: i and i_f<k> as i / (e_a + e_b + tau) on the qualified pairs (deepISA's
    _compute_interaction_per_track with normalize=True), NaN on the others; and the column qualified. Returns
    {model: its qualified pairs}."""
    n = {}
    for m, t in tables.items():
        ea, eb, both = (t[c].to_numpy() for c in ("e_a", "e_b", "isa_both"))
        den = ea + eb + tau
        q = (ea >= thresholds[m]) & (eb >= thresholds[m]) & (both - eb >= 0) & (both - ea >= 0) & (den > 0)
        for c in ["i", *cb.fold_columns([t])]:
            t[c] = np.where(q, t[c].to_numpy() / np.where(q, den, 1.0), np.nan)
        t["qualified"] = q
        n[m] = int(q.sum())
    return n


def quantile_edges(v: np.ndarray, k: int) -> np.ndarray:
    """The inner edges of k quantile bins of v (fewer where values tie)."""
    return np.unique(np.quantile(v, np.linspace(0, 1, k + 1)[1:-1])) if len(v) else np.empty(0)


def far_means(v: np.ndarray, cell: np.ndarray, b: np.ndarray, far: np.ndarray, n_hi: int, n_bins: int,
              min_cell: int) -> np.ndarray:  # fmt: skip
    """Per cell: the mean of v over the far rows in it; a cell with fewer than min_cell of them, its weaker-site bin's
    mean (and a bin with fewer, every far row's)."""
    n_bin = np.bincount(b[far], minlength=n_bins)
    by_bin = np.where(n_bin >= min_cell, np.bincount(b[far], v[far], n_bins) / np.maximum(n_bin, 1), v[far].mean())
    n_cell = np.bincount(cell[far], minlength=n_bins * n_hi)
    by_cell = np.bincount(cell[far], v[far], n_bins * n_hi) / np.maximum(n_cell, 1)
    return np.where(n_cell >= min_cell, by_cell, by_bin[np.arange(n_bins * n_hi) // n_hi])


def adjust_distant(tables: dict, is_open: dict, far_gap: float, n_lo: int, n_hi: int, min_cell: int):
    """In place, per model: i and i_f<k> minus the mean of the distant pairs (gap > far_gap, open under the model) in
    the same strength cell (the module doc). Returns (curves.tsv's table, a summary)."""
    strength = {m: np.sort(t[["e_a", "e_b"]].to_numpy(), axis=1) for m, t in tables.items()}  # weaker, stronger
    far = {m: is_open[m] & (t["gap"].to_numpy() > far_gap) for m, t in tables.items()}
    pooled = np.concatenate([strength[m][far[m]] for m in tables])
    lo_edges = quantile_edges(pooled[:, 0], n_lo)
    pooled_bin = np.searchsorted(lo_edges, pooled[:, 0], side="right")
    hi_edges = [quantile_edges(pooled[pooled_bin == b, 1], n_hi) for b in range(len(lo_edges) + 1)]
    n_bins, n_cells = len(lo_edges) + 1, (len(lo_edges) + 1) * n_hi
    curves, fallback = [], {}
    for m, t in tables.items():
        lo, hi = strength[m][:, 0], strength[m][:, 1]
        b = np.searchsorted(lo_edges, lo, side="right")
        h = np.zeros(len(t), dtype=np.int64)
        for k, e in enumerate(hi_edges):
            h[b == k] = np.searchsorted(e, hi[b == k], side="right")
        cell, f = b * n_hi + h, far[m]
        n_cell = np.bincount(cell[f], minlength=n_cells)
        raw = t["i"].to_numpy().copy()
        subtracted = far_means(raw, cell, b, f, n_hi, n_bins, min_cell)
        t["i"] = raw - subtracted[cell]
        for c in cb.fold_columns([t]):
            v = t[c].to_numpy()
            t[c] = v - far_means(v, cell, b, f, n_hi, n_bins, min_cell)[cell]
        fallback[m] = int(((n_cell[cell] < min_cell) & is_open[m]).sum())
        gap, o = t["gap"].to_numpy(), is_open[m]
        kind = np.select([gap <= 5, gap <= 50, gap <= far_gap], ["adjacent", "near", "mid"], "far")
        g = (pd.DataFrame({"cell": cell[o], "distance_class": kind[o], "i": raw[o], "weaker_e": lo[o],
                           "stronger_e": hi[o]})
             .groupby(["cell", "distance_class"])
             .agg(n=("i", "size"), mean_i=("i", "mean"), weaker_e=("weaker_e", "median"),
                  stronger_e=("stronger_e", "median"))
             .reset_index())  # fmt: skip
        g["subtracted"] = subtracted[g["cell"]]
        g["far_pairs_in_cell"] = n_cell[g["cell"]]
        g.insert(0, "model", m)
        curves.append(g)
    curves = pd.concat(curves, ignore_index=True)
    curves.insert(2, "weaker_bin", curves["cell"] // n_hi)
    curves.insert(3, "stronger_bin", curves["cell"] % n_hi)
    info = {"far_gap": far_gap, "lo_bins": n_bins, "hi_bins": n_hi, "min_cell": min_cell,
            "weaker_e_edges": [round(float(x), 5) for x in lo_edges],
            "far_pairs": {m: int(far[m].sum()) for m in tables},
            "open_pairs_in_sparse_cells": fallback}  # fmt: skip
    return curves.drop(columns="cell"), info


def mean_e(sub: dict, base: pd.DataFrame, models: list[str], key: str) -> pd.Series:
    """Per TF (the mean e of its own sites in these pairs) or TF pair (the mean of its two sites'), each site's e the
    mean over `models`."""
    ea, eb = group_i(sub, models, "e_a"), group_i(sub, models, "e_b")
    if key == "tf":
        labels, v = np.r_[base["tf_a"].to_numpy(), base["tf_b"].to_numpy()], np.r_[ea, eb]
    else:
        labels, v = base[key].to_numpy(), (ea + eb) / 2
    codes, uniq = pd.factorize(labels)
    return pd.Series(np.bincount(codes, v) / np.bincount(codes), index=uniq)


def corr(x, y) -> float | None:
    """Pearson's r over the rows where both are finite (None with fewer than 3, or no spread)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or x[ok].std() == 0 or y[ok].std() == 0:
        return None
    return round(float(np.corrcoef(x[ok], y[ok])[0, 1]), 4)


def counts(table: pd.DataFrame | None, cols: list[str]) -> dict:
    if table is None:
        return {"tested": 0}
    return {"tested": len(table), **{c: int(table[c].fillna(False).astype(bool).sum()) for c in cols if c in table}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", type=Path, required=True, help="matrix.py's output: <sites>/<model>/pairs.tsv.gz")
    ap.add_argument("--motifs", type=Path, required=True, help="coop_isa.py's --motifs (motif, cluster_annotation)")
    ap.add_argument("--groups", type=Path, required=True, help="TSV: group, datasets (comma-separated)")
    ap.add_argument("--comparisons", type=Path, required=True, help="TSV: name, group_a, group_b")
    ap.add_argument("--series", type=Path, help="TSV with column group: the groups in order, for the trends")
    ap.add_argument("--t-coop", type=float, default=0.1)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--min-n", type=int, default=50)
    ap.add_argument("--fdr", type=float, default=0.05)
    ap.add_argument("--open-quantile", type=float, default=0.1)
    ap.add_argument("--adjust", choices=("none", "ratio", "distant"), default="none",
                    help="take the sites' strength out of i first (the module doc); needs --out-dir")  # fmt: skip
    ap.add_argument("--out-dir", type=Path, help="default <matrix>/combined; required with --adjust")
    ap.add_argument("--tau", type=float, default=0.0, help="ratio: added to e_a + e_b (deepISA's tau)")
    ap.add_argument("--isa-quantile", type=float, default=0.1, help="ratio: the qualifying quantile of e")
    ap.add_argument("--far-gap", type=float, default=150, help="distant: the gap (bp) beyond which a pair is distant")
    ap.add_argument("--lo-bins", type=int, default=20, help="distant: quantile bins of the weaker site's e")
    ap.add_argument("--hi-bins", type=int, default=4, help="distant: quantile bins of the stronger one's, in each")
    ap.add_argument("--min-cell", type=int, default=50, help="distant: the distant pairs a cell needs for its mean")
    args = ap.parse_args(argv)
    if args.adjust != "none" and args.out_dir is None:
        ap.error("--adjust needs --out-dir: combined/ holds the unadjusted results")
    groups = {r["group"]: [d.strip() for d in r["datasets"].split(",")] for r in cb.read_tsv(args.groups)}
    comparisons = [(r["name"], r["group_a"], r["group_b"]) for r in cb.read_tsv(args.comparisons)]
    series = [r["group"] for r in cb.read_tsv(args.series)] if args.series else []
    for g in [x for _, a, b in comparisons for x in (a, b)] + series:
        if g not in groups:
            ap.error(f"group {g!r} is not in {args.groups}")
    datasets = list(dict.fromkeys(d for g in groups.values() for d in g))
    names = pd.read_csv(args.motifs, sep="\t", usecols=["motif", "cluster_annotation"]).drop_duplicates("motif")
    names = dict(zip(names["motif"], names["cluster_annotation"].str.replace(r"_\d+$", "", regex=True)))
    tables, ident, thresholds = load(args.matrix, datasets, names, args.open_quantile)
    units = pd.Index(ident["unit"].unique())
    is_open = {m: tables[m]["a"].to_numpy() >= thresholds[m]["threshold"] for m in datasets}
    has_e = all("e_a" in tables[m] for m in datasets)
    if args.adjust != "none" and not all(c in tables[m] for m in datasets for c in EXTRA):
        ap.error(f"--adjust needs coop_isa.py's {', '.join(EXTRA)} in every pairs.tsv.gz")
    out = args.out_dir or args.matrix / "combined"
    out.mkdir(parents=True, exist_ok=True)
    summary = {"pairs": len(ident), "windows": len(units), "open": thresholds, "comparisons": {}, "series": {},
               "activity_r": {"comparisons": {}, "series": {}}}  # fmt: skip
    use = is_open
    if args.adjust == "ratio":
        isa_thresh = site_thresholds(args.matrix, datasets, args.isa_quantile)
        qualified = adjust_ratio(tables, isa_thresh, args.tau)
        use = {m: is_open[m] & tables[m]["qualified"].to_numpy() for m in datasets}
        summary["adjust"] = {"method": "ratio", "i": "i / (e_a + e_b + tau) on the qualified pairs", "tau": args.tau,
                             "isa_quantile": args.isa_quantile, "isa_thresh": isa_thresh, "qualified": qualified}  # fmt: skip
    elif args.adjust == "distant":
        curves, info = adjust_distant(tables, is_open, args.far_gap, args.lo_bins, args.hi_bins, args.min_cell)
        curves.to_csv(out / "curves.tsv", sep="\t", index=False, float_format="%.5g")
        summary["adjust"] = {"method": "distant", "i": "i minus the mean i of distant pairs as strong", **info}
    pd.DataFrame([{"model": m, **v} for m, v in thresholds.items()]).to_csv(out / "open.tsv", sep="\t", index=False)
    print(f"{len(ident):,} site pairs in {len(units):,} windows; open windows: {thresholds}", flush=True)
    t = args.t_coop
    with_synergy = args.adjust != "ratio"  # the ratio has no scale for t

    for level, key in (("pairs", "pair"), ("tfs", "tf")):
        rows = []
        for name, ga, gb in comparisons:
            models = groups[ga] + groups[gb]
            mask = np.logical_and.reduce([use[m] for m in models])
            sub = {m: tables[m][mask].reset_index(drop=True) for m in models}
            base = ident[mask].reset_index(drop=True)
            syn = None
            if with_synergy:
                syn = cb.changes(sub, groups, [(name, ga, gb)], key, t, args.n_boot, args.min_n, fdr=args.fdr,
                                 units=units)  # fmt: skip
            di = delta_i(sub, base, groups[ga], groups[gb], key, units, args)
            if syn is None:
                m = di
            else:
                # Both give each group's mean i (every pair once per model, so pooling the group's tables or averaging
                # its models is the same mean): one column each.
                m = syn.merge(di, on=key, how="outer", suffixes=("", "_di"))
                for c in ("mean_i_a", "mean_i_b"):
                    m[c] = m[c].fillna(m.pop(f"{c}_di"))
            if has_e:
                m["mean_e_a"] = m[key].map(mean_e(sub, base, groups[ga], key))
                m["mean_e_b"] = m[key].map(mean_e(sub, base, groups[gb], key))
                m["delta_e"] = m["mean_e_b"] - m["mean_e_a"]
                r = summary["activity_r"]["comparisons"].setdefault(name, {})
                tested = m["delta_i_q"].notna()
                r[level] = {"delta_i": corr(m.loc[tested, "delta_i"], m.loc[tested, "delta_e"])}
                if syn is not None:
                    tested = m["q"].notna()
                    r[level]["synergy"] = corr(m.loc[tested, "delta"], m.loc[tested, "delta_e"])
            for c, v in (("comparison", name), ("group_a", ga), ("group_b", gb)):
                m[c] = v
            rows.append(m)
            c = summary["comparisons"].setdefault(name, {
                "pairs_open": int(np.logical_and.reduce([is_open[x] for x in models]).sum()),
                "pairs_used": int(mask.sum()), "windows_used": int(base["unit"].nunique())})  # fmt: skip
            c[level] = {
                "synergy": counts(syn, ["significant", "robust"]),
                "delta_i": counts(di, ["delta_i_significant", "delta_i_robust"]),
            }
        if rows:
            table = pd.concat(rows, ignore_index=True)
            lead = ["comparison", "group_a", "group_b", key]
            table[lead + [c for c in table.columns if c not in lead]].to_csv(
                out / f"changes_{level}.tsv", sep="\t", index=False, float_format="%.5g"
            )
        if series:
            models = [d for g in series for d in groups[g]]
            mask = np.logical_and.reduce([use[m] for m in models])
            sub = {m: tables[m][mask].reset_index(drop=True) for m in models}
            base = ident[mask].reset_index(drop=True)
            tr = None
            if with_synergy:
                tr = cb.trend(sub, groups, series, key, t, args.n_boot, args.min_n, fdr=args.fdr, units=units)
            si = slope_i(sub, base, groups, series, key, units, args)
            table = si if tr is None else tr.merge(si, on=key, how="outer")
            if has_e:
                x = np.arange(len(series), dtype=float)
                w = (x - x.mean()) / ((x - x.mean()) ** 2).sum()
                E = np.column_stack([table[key].map(mean_e(sub, base, groups[g], key)).to_numpy() for g in series])
                for j, g in enumerate(series):
                    table[f"mean_e_{g}"] = E[:, j]
                table["slope_e"] = E @ w
                tested = table["slope_i_q"].notna()
                r = {"slope_i": corr(table.loc[tested, "slope_i"], table.loc[tested, "slope_e"])}
                if tr is not None:
                    tested = table["slope_q"].notna()
                    r["synergy"] = corr(table.loc[tested, "slope"], table.loc[tested, "slope_e"])
                summary["activity_r"]["series"][level] = r
            table.to_csv(out / f"trend_{level}.tsv", sep="\t", index=False, float_format="%.5g")
            summary["series"]["pairs_open"] = int(np.logical_and.reduce([is_open[x] for x in models]).sum())
            summary["series"]["pairs_used"] = int(mask.sum())
            summary["series"][level] = {
                "synergy": cb.trend_counts(tr),
                "slope_i": counts(si, ["slope_i_significant", "slope_i_robust"]),
            }
    summary["settings"] = {"t_coop": t, "n_boot": args.n_boot, "min_n": args.min_n, "fdr": args.fdr,
                           "open_quantile": args.open_quantile, "series": series, "adjust": args.adjust}  # fmt: skip
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
