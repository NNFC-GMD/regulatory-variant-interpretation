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

Out (<matrix>/combined/): changes_pairs.tsv, changes_tfs.tsv, with --series trend_pairs.tsv and trend_tfs.tsv,
open.tsv (per model: its threshold, its windows and how many are open), summary.json.

    paired.py --matrix <dir> --motifs <motifs.tsv> --groups <groups.tsv> --comparisons <comparisons.tsv>
        [--series <series.tsv>]
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
    sw = ident["tf_a"] > ident["tf_b"]
    ident.loc[sw, ["tf_a", "tf_b"]] = ident.loc[sw, ["tf_b", "tf_a"]].to_numpy()
    ident["pair"] = ident["tf_a"] + " + " + ident["tf_b"]
    tables = {}
    for m, dfs in per_model.items():
        df = pd.concat(dfs, ignore_index=True)[keep].reset_index(drop=True)
        cols = ["i", "a", *cb.fold_columns([df])]
        tables[m] = pd.concat([ident[["unit", "sites", "peak_id", "tf_a", "tf_b", "pair", "distance"]], df[cols]],
                              axis=1)  # fmt: skip
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
    args = ap.parse_args(argv)
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
    out = args.matrix / "combined"
    out.mkdir(exist_ok=True)
    pd.DataFrame([{"model": m, **v} for m, v in thresholds.items()]).to_csv(out / "open.tsv", sep="\t", index=False)
    print(f"{len(ident):,} site pairs in {len(units):,} windows; open windows: {thresholds}", flush=True)
    summary = {"pairs": len(ident), "windows": len(units), "open": thresholds, "comparisons": {}, "series": {}}
    t = args.t_coop

    for level, key in (("pairs", "pair"), ("tfs", "tf")):
        rows = []
        for name, ga, gb in comparisons:
            models = groups[ga] + groups[gb]
            mask = np.logical_and.reduce([is_open[m] for m in models])
            sub = {m: tables[m][mask].reset_index(drop=True) for m in models}
            base = ident[mask].reset_index(drop=True)
            syn = cb.changes(sub, groups, [(name, ga, gb)], key, t, args.n_boot, args.min_n, fdr=args.fdr, units=units)
            di = delta_i(sub, base, groups[ga], groups[gb], key, units, args)
            if syn is None:
                m = di
            else:
                # Both give each group's mean i (every pair once per model, so pooling the group's tables or averaging
                # its models is the same mean): one column each.
                m = syn.merge(di, on=key, how="outer", suffixes=("", "_di"))
                for c in ("mean_i_a", "mean_i_b"):
                    m[c] = m[c].fillna(m.pop(f"{c}_di"))
            for c, v in (("comparison", name), ("group_a", ga), ("group_b", gb)):
                m[c] = v
            rows.append(m)
            c = summary["comparisons"].setdefault(name, {"pairs_open": int(mask.sum()),
                                                         "windows_open": int(base["unit"].nunique())})  # fmt: skip
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
            mask = np.logical_and.reduce([is_open[m] for m in models])
            sub = {m: tables[m][mask].reset_index(drop=True) for m in models}
            base = ident[mask].reset_index(drop=True)
            tr = cb.trend(sub, groups, series, key, t, args.n_boot, args.min_n, fdr=args.fdr, units=units)
            si = slope_i(sub, base, groups, series, key, units, args)
            table = si if tr is None else tr.merge(si, on=key, how="outer")
            table.to_csv(out / f"trend_{level}.tsv", sep="\t", index=False, float_format="%.5g")
            summary["series"]["pairs_open"] = int(mask.sum())
            summary["series"][level] = {
                "synergy": cb.trend_counts(tr),
                "slope_i": counts(si, ["slope_i_significant", "slope_i_robust"]),
            }
    summary["settings"] = {"t_coop": t, "n_boot": args.n_boot, "min_n": args.min_n, "fdr": args.fdr,
                           "open_quantile": args.open_quantile, "series": series}  # fmt: skip
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
