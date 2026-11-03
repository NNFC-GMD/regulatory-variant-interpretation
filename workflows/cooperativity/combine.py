#!/usr/bin/env python3
"""Across datasets: TF-pair synergy with bootstrap CIs, the changes between cell states, and the distance profile.

DeepISA's definitions (He et al. 2025, Methods; its code's Pd6_TF_cooperativity/get_cooperativity.py and the
tf_pair_synergy_score_*.csv columns): a site pair is cooperative when |i| > 0.1 (synergistic i > 0.1, redundant
i < -0.1), independent otherwise; independence score = independent / all pairs; synergy score over the cooperative
pairs; i_sum = sum |i| over them. Labels (TF pairs: i_sum >= 1, independence > 0.9; TFs: i_sum >= 5, > 0.95):
synergistic (synergy > 0.7), redundant (< 0.3), intermediate, independent (i_sum below and independence above), and
unlabelled otherwise; a TF pair or TF needs more than 10 pairs.

Reads <root>/<dataset>/pairs.tsv.gz (coop_isa.py). Each site's TF is named from its motif with --motifs, by
coop_isa.py's rule (cluster_annotation without its _<n> suffix; the motif's own name where it has none). The tf
columns are not read: the first run's coop_isa.py (20 October 2026) left them empty for motifs without an annotation
and sorted them apart from motif_a/motif_b. Synergy of a set of site pairs = sum(i > t) / sum(|i| > t)
(DeepISA's definition, t = 0.1 in log2 counts here). Intervals: Poisson bootstrap over peaks (each peak's pairs
weighted together, --n-boot replicates); a group of datasets pools their pairs, peaks kept distinct per dataset.

Groups (--groups TSV, columns group and datasets, comma-separated): the datasets pooled as one cell state.
Comparisons (--comparisons TSV, columns name, group_a and group_b): the change b - a in synergy. A change is reported
for TF pairs (and TFs) with at least --min-n pairs in each group, with its bootstrap 95% CI.

Significance: each change's two-sided p from its bootstrap SE (normal approximation, z = change / SE; 200 replicates
are too few for p-values counted from the replicates), and Benjamini-Hochberg q within each comparison
(scipy.stats.false_discovery_control). significant = q < --fdr (0.05); ci_excludes_0 is the earlier criterion, the
95% CI excluding 0, uncorrected. The fold check, when the pair tables have coop_isa.py's per-fold i (i_f<k>): the
change computed from fold k's models alone in both groups (the folds are the same chromosome splits in every
dataset), its smallest and largest value over the folds, folds_agree (the folds whose change has the sign of the
fold-mean change), and robust = significant and every fold agrees. pairs_by_dataset.tsv and tfs_by_dataset.tsv get
the per-fold synergy's range (synergy_fold_min, synergy_fold_max).

Series (--series TSV, column group, the groups in order, at positions 0, 1, ...): per TF pair (and TF) with at least
--min-n pairs in every group of the series, its synergy in each group; the slope over the positions (least squares)
with its bootstrap 95% CI, p and q; a heterogeneity test for any change along the series (the consecutive
differences, a Wald test with their bootstrap covariance, chi-square with G - 1 df), with q; peak and trough, the
groups with the highest and lowest synergy; the fold check of the slope. significant = slope q < --fdr,
het_significant = heterogeneity q < --fdr.

Out (<root>/combined/): pairs_by_dataset.tsv, tfs_by_dataset.tsv, changes_pairs.tsv, changes_tfs.tsv,
distance.tsv, summary.json, and with --series trend_pairs.tsv, trend_tfs.tsv.

paired.py uses this module's functions with a paired bootstrap (the `units` argument): one dataset's sites scored by
every dataset's models, the same resampling weights for every model.

--official adds the rules of the official deepISA package (anderssonlab/deepISA, pinned in pixi.toml), computed by
the package's own functions on the same ablations: per dataset, <root>/official/<dataset>/ holds the two tables its
QuickStart writes before its aggregate stage, motif_single_isa.csv (each site's isa_t0 = e) and motif_combi_isa.csv
(tf1, tf2, distance = the edge-to-edge gap, isa1_t0, isa2_t0, isa_both_t0, region = the peak), with only the pairs its
make_pairs_for_region keeps (1 <= gap <= --receptive-field, 255). Then, as QuickStart's aggregate stage does:
  calc_interaction(isa_thresh_quantile=0.1, normalize=False)  interaction_t0 = isa1 + isa2 - isa_both, NaN unless
      both single effects reach the 10th percentile of the non-negative ones and both effects given the other
      ablated are >= 0;
  calc_coop_score(level="tf_pair" and "tf", min_count=10, q_val_thresh=0.1)  coop_score = sum(i) / sum(|i|), a
      Mann-Whitney test against the pairs more than 150 bp apart, BH, and labels by the 70th and 30th percentiles of
      the significant scores (coop_tf_pair_t0.csv, coop_tf_t0.csv).
Its labels are relative within a dataset and its coop_score is NaN where the label is Independent, so changes
between groups (and the series' trends) are computed here from interaction_t0 instead: coop = sum(i) / sum(|i|) over
the qualified pairs, which is 2 x (this script's synergy at t = 0) - 1, with the same bootstrap, significance and fold
check (interaction_t0 = isa1 + isa2 - isa_both is coop_isa.py's i, so its per-fold i_f<k> apply). Differences from the package's own run:
its sites are every Fi-NeMo hit and its predictions one model on one strand; here they are coop_isa.py's (at most
--max-sites non-overlapping sites per sampled peak; both strands, the fold mean, log2 counts), so its 10th percentile
is taken over those sites. Out (<root>/official/): <dataset>/ as above, coop_tf_pairs_by_dataset.tsv,
coop_tfs_by_dataset.tsv, changes_pairs.tsv, changes_tfs.tsv, summary.json, and with --series trend_pairs.tsv,
trend_tfs.tsv.

    combine.py --root <coop_isa.py --out-dir> --motifs <motifs.tsv> --groups <groups.tsv> --comparisons <comparisons.tsv>
        [--series <series.tsv>] [--official]
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

DIST_BINS = [0, 20, 50, 100, 200, 500, 1001]
OFFICIAL = {"isa_thresh_quantile": 0.1, "min_count": 10, "q_val_thresh": 0.1}
"""The settings QuickStart.run_isa's aggregate stage passes (quickstart.py, deepISA @ the pixi.toml pin)."""


def read_tsv(path: Path) -> list[dict[str, str]]:
    """Rows of a tab-separated file with a header, blank lines and #-comments skipped."""
    lines = [x for x in path.read_text().splitlines() if x.strip() and not x.startswith("#")]
    head = lines[0].split("\t")
    return [dict(zip(head, (v.strip() for v in x.split("\t")))) for x in lines[1:]]


def fold_columns(dfs) -> list[str]:
    """coop_isa.py's per-fold i columns (i_f<k>) that every one of the tables has."""
    dfs = list(dfs)
    if not dfs:
        return []
    return [c for c in dfs[0].columns if c.startswith("i_f") and all(c in d.columns for d in dfs)]


def synergy_point(df: pd.DataFrame, key: str, t: float, col: str = "i") -> pd.Series:
    """Per value of `key`: sum(i > t) / sum(|i| > t) of column `col`, NaN without a cooperative pair."""
    codes, uniq = pd.factorize(df[key])
    i = df[col].to_numpy()
    k = len(uniq)
    pos = np.bincount(codes, np.where(i > t, i, 0.0), k)
    den = np.bincount(codes, np.where(np.abs(i) > t, np.abs(i), 0.0), k)
    return pd.Series(np.where(den > 0, pos / np.maximum(den, 1e-12), np.nan), index=uniq)


def quiet(f, *args, **kwargs):
    """f(*args, **kwargs) without numpy's all-NaN and empty-slice warnings (a NaN result is expected there)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return f(*args, **kwargs)


def boot_p(stat, reps: np.ndarray) -> np.ndarray:
    """Two-sided p of each statistic, its bootstrap replicates in the columns of reps: normal approximation, z = stat /
    the replicates' SD. A statistic whose replicates do not vary gets p = 0, or 1 when it is 0."""
    from scipy.stats import norm

    stat = np.asarray(stat, dtype=float)
    se = quiet(np.nanstd, reps, axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = 2 * norm.sf(np.abs(stat) / se)
    flat = se == 0
    p[flat] = np.where(stat[flat] == 0, 1.0, 0.0)
    p[~np.isfinite(stat)] = np.nan
    return p


def bh(p) -> np.ndarray:
    """Benjamini-Hochberg q-values (scipy.stats.false_discovery_control) over the finite p; NaN elsewhere."""
    from scipy.stats import false_discovery_control

    p = np.asarray(p, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        q[ok] = false_discovery_control(p[ok], method="bh")
    return q


def add_fold_check(m: pd.DataFrame, col: str, per_fold: np.ndarray) -> None:
    """Columns <col>_fold_min, <col>_fold_max, folds_agree and n_folds of m; per_fold holds `col` computed from each
    fold's models alone (folds x rows of m)."""
    ref = np.sign(m[col].to_numpy())
    m[f"{col}_fold_min"] = quiet(np.nanmin, per_fold, axis=0)
    m[f"{col}_fold_max"] = quiet(np.nanmax, per_fold, axis=0)
    m["folds_agree"] = ((np.sign(per_fold) == ref) & (ref != 0)).sum(axis=0)
    m["n_folds"] = len(per_fold)


def synergy_boot(df: pd.DataFrame, key: str, t: float, n_boot: int, seed: int, units: pd.Index | None = None):
    """Per value of `key`: n, synergy, independence, mean i, and the bootstrap replicates of synergy (columns b0..).
    The bootstrap resamples df's peaks; with `units`, df's unit column coded by that index, so that every call with
    the same seed and units draws the same weights (a paired bootstrap)."""
    codes, uniq = pd.factorize(df[key])
    if units is None:
        peaks, _ = pd.factorize(df["peak"])
        n_units = peaks.max() + 1
    else:
        peaks = units.get_indexer(df["unit"])
        if (peaks < 0).any():
            raise ValueError("synergy_boot: rows whose unit is not in `units`")
        n_units = len(units)
    i = df["i"].to_numpy()
    pos = np.where(i > t, i, 0.0)
    big = np.where(np.abs(i) > t, np.abs(i), 0.0)
    k = len(uniq)
    n = np.bincount(codes, minlength=k)
    syn = np.bincount(codes, pos, k) / np.maximum(np.bincount(codes, big, k), 1e-12)
    syn[np.bincount(codes, big, k) == 0] = np.nan
    ind = np.bincount(codes, (np.abs(i) <= t).astype(float), k) / n
    mean_i = np.bincount(codes, i, k) / n
    i_sum = np.bincount(codes, big, k)
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, k))
    for b in range(n_boot):
        w = rng.poisson(1.0, n_units)[peaks].astype(float)
        den = np.bincount(codes, w * big, k)
        boots[b] = np.where(den > 0, np.bincount(codes, w * pos, k) / np.maximum(den, 1e-12), np.nan)
    out = pd.DataFrame({key: uniq, "n": n, "synergy": syn, "independence": ind, "i_sum": i_sum, "mean_i": mean_i})
    c_sum, lin = (5.0, 0.95) if key == "tf" else (1.0, 0.9)
    out["label"] = np.select(
        [
            (out["i_sum"] >= c_sum) & (out["synergy"] > 0.7),
            (out["i_sum"] >= c_sum) & (out["synergy"] < 0.3),
            out["i_sum"] >= c_sum,
            (out["i_sum"] < c_sum) & (out["independence"] > lin),
        ],
        ["synergistic", "redundant", "intermediate", "independent"],
        default="unlabelled",
    )
    out["lo"], out["hi"] = quiet(np.nanpercentile, boots, 2.5, axis=0), quiet(np.nanpercentile, boots, 97.5, axis=0)
    return out, boots


def group_score(pairs: dict, groups: dict, g: str, key: str, t: float, n_boot: int, seed: int, transform=None,
                units: pd.Index | None = None):  # fmt: skip
    """Group g's datasets pooled: (synergy_boot's table, its replicates, {fold column: per-fold synergy}), or None
    when none of its datasets has pairs. `transform` maps a synergy to the score compared."""
    dfs = [pairs[d] for d in groups[g] if d in pairs]
    if not dfs:
        return None
    df = pd.concat(dfs)
    if key == "tf":
        df = pd.concat([df.assign(tf=df["tf_a"]), df.assign(tf=df["tf_b"])])
    score, boots = synergy_boot(df, key, t, n_boot, seed, units)
    folds = {c: synergy_point(df, key, t, c) for c in fold_columns(dfs)}
    if transform is not None:
        score = score.assign(synergy=transform(score["synergy"]), lo=transform(score["lo"]), hi=transform(score["hi"]))
        boots = transform(boots)
        folds = {c: pd.Series(transform(v), index=v.index) for c, v in folds.items()}
    return score, boots, folds


def changes(
    pairs: dict,
    groups: dict,
    comparisons: list,
    key: str,
    t: float,
    n_boot: int,
    min_n: int,
    transform=None,
    fdr: float = 0.05,
    units: pd.Index | None = None,
) -> pd.DataFrame | None:
    """Per comparison: each TF pair's (or TF's) synergy in group a and b, the change b - a with its bootstrap 95% CI,
    p, q (BH within the comparison) and the fold check, kept where both groups have >= min_n pairs. `transform` maps a
    synergy (and its replicates) to the score compared; `units` makes the bootstrap paired (synergy_boot)."""
    rows = []
    seeds = (11, 11) if units is not None else (11, 12)  # paired: the same weights in both groups
    for name, ga, gb in comparisons:
        ra = group_score(pairs, groups, ga, key, t, n_boot, seeds[0], transform, units)
        rb = group_score(pairs, groups, gb, key, t, n_boot, seeds[1], transform, units)
        if ra is None or rb is None:
            continue
        (a, boot_a, fold_a), (b, boot_b, fold_b) = ra, rb
        m = a.merge(b, on=key, suffixes=("_a", "_b"))
        ba = pd.DataFrame(boot_a, columns=a[key]).reindex(columns=m[key])
        bb = pd.DataFrame(boot_b, columns=b[key]).reindex(columns=m[key])
        delta = bb.to_numpy() - ba.to_numpy()
        m["delta"] = m["synergy_b"] - m["synergy_a"]
        m["delta_lo"], m["delta_hi"] = quiet(np.nanpercentile, delta, 2.5, axis=0), quiet(np.nanpercentile, delta, 97.5, axis=0)  # fmt: skip
        m["p"] = boot_p(m["delta"], delta)
        cols = [c for c in fold_a if c in fold_b]
        if cols:
            per_fold = np.stack([fold_b[c].reindex(m[key]).to_numpy() - fold_a[c].reindex(m[key]).to_numpy()
                                 for c in cols])  # fmt: skip
            add_fold_check(m, "delta", per_fold)
        m = m[(m["n_a"] >= min_n) & (m["n_b"] >= min_n)].copy()
        m.insert(0, "comparison", name)
        m.insert(1, "group_a", ga)
        m.insert(2, "group_b", gb)
        m["ci_excludes_0"] = (m["delta_lo"] > 0) | (m["delta_hi"] < 0)
        m["q"] = bh(m["p"])
        m["significant"] = m["q"] < fdr
        if cols:
            m["robust"] = m["significant"] & (m["folds_agree"] == len(cols))
        rows.append(m)
    return pd.concat(rows) if rows else None


def change_counts(ch: pd.DataFrame | None, comparisons: list) -> dict:
    out = {}
    for n, _, _ in comparisons:
        r = ch[ch["comparison"] == n] if ch is not None else None
        c = {"tested": 0 if r is None else len(r), "significant": 0 if r is None else int(r["significant"].sum())}
        for col in ("robust", "ci_excludes_0"):
            if r is not None and col in r:
                c[col] = int(r[col].sum())
        out[n] = c
    return out


def trend(
    pairs: dict,
    groups: dict,
    series: list,
    key: str,
    t: float,
    n_boot: int,
    min_n: int,
    transform=None,
    fdr: float = 0.05,
    units: pd.Index | None = None,
    score: str = "synergy",
) -> pd.DataFrame | None:
    """The series' trend per TF pair (or TF) with >= min_n pairs in every group of the series (the module doc): its
    score in each group (columns <score>_<group>), the slope over positions 0, 1, ... with CI, p and q, the
    heterogeneity test, peak and trough, and the fold check of the slope."""
    res = []
    for k, g in enumerate(series):
        r = group_score(pairs, groups, g, key, t, n_boot, 21 if units is not None else 21 + k, transform, units)
        if r is None:
            return None
        res.append(r)
    keep = None
    for s, _, _ in res:
        ok = set(s.loc[s["n"] >= min_n, key])
        keep = ok if keep is None else keep & ok
    keys = sorted(keep)
    if not keys:
        return None
    n_g = len(series)
    S = np.column_stack([s.set_index(key)["synergy"].reindex(keys).to_numpy() for s, _, _ in res])  # keys x groups
    B = np.stack([pd.DataFrame(b, columns=s[key]).reindex(columns=keys).to_numpy() for s, b, _ in res], axis=2)
    x = np.arange(n_g, dtype=float)
    w = (x - x.mean()) / ((x - x.mean()) ** 2).sum()  # least-squares slope = S @ w
    out = pd.DataFrame({key: keys})
    for g, (s, _, _) in zip(series, res):
        out[f"n_{g}"] = s.set_index(key)["n"].reindex(keys).to_numpy()
    for j, g in enumerate(series):
        out[f"{score}_{g}"] = S[:, j]
    reps = B @ w
    out["slope"] = S @ w
    out["slope_lo"], out["slope_hi"] = quiet(np.nanpercentile, reps, 2.5, axis=0), quiet(np.nanpercentile, reps, 97.5, axis=0)  # fmt: skip
    out["slope_p"] = boot_p(out["slope"], reps)
    out["het_Q"], out["het_p"] = heterogeneity(S, B)
    has = np.isfinite(S).any(axis=1)
    names = np.array(series, dtype=object)
    out["peak"] = np.where(has, names[np.where(np.isfinite(S), S, -np.inf).argmax(axis=1)], "")
    out["trough"] = np.where(has, names[np.where(np.isfinite(S), S, np.inf).argmin(axis=1)], "")
    cols = [c for c in res[0][2] if all(c in f for _, _, f in res)]
    if cols:
        per_fold = np.stack([np.column_stack([f[c].reindex(keys).to_numpy() for _, _, f in res]) @ w for c in cols])
        add_fold_check(out, "slope", per_fold)
    out["slope_q"], out["het_q"] = bh(out["slope_p"]), bh(out["het_p"])
    out["significant"] = out["slope_q"] < fdr
    out["het_significant"] = out["het_q"] < fdr
    if cols:
        out["robust"] = out["significant"] & (out["folds_agree"] == len(cols))
    return out


def heterogeneity(S: np.ndarray, B: np.ndarray):
    """Per row of S (keys x groups; B its replicates, n_boot x keys x groups): the Wald statistic that every
    consecutive difference is 0, with the replicates' covariance, and its chi-square p (groups - 1 df)."""
    from scipy.stats import chi2

    n_g = S.shape[1]
    C = np.diff(np.eye(n_g), axis=0)  # (groups - 1) x groups
    D, DB = S @ C.T, B @ C.T
    Q, p = np.full(len(S), np.nan), np.full(len(S), np.nan)
    for k in range(len(S)):
        r = DB[:, k, :]
        r = r[np.isfinite(r).all(axis=1)]
        if len(r) <= n_g or not np.isfinite(D[k]).all():
            continue
        cov = np.atleast_2d(np.cov(r, rowvar=False))
        Q[k] = float(D[k] @ np.linalg.pinv(cov) @ D[k])
        p[k] = float(chi2.sf(Q[k], n_g - 1))
    return Q, p


def trend_counts(tr: pd.DataFrame | None) -> dict:
    if tr is None:
        return {"tested": 0}
    out = {"tested": len(tr), "significant": int(tr["significant"].sum()),
           "het_significant": int(tr["het_significant"].sum())}  # fmt: skip
    if "robust" in tr:
        out["robust"] = int(tr["robust"].sum())
    return out


def fold_range(df: pd.DataFrame, key: str, t: float, keys) -> tuple | None:
    """The smallest and largest per-fold synergy of each of `keys` (None without per-fold columns)."""
    cols = fold_columns([df])
    if not cols:
        return None
    per_fold = np.column_stack([synergy_point(df, key, t, c).reindex(keys).to_numpy() for c in cols])
    return quiet(np.nanmin, per_fold, axis=1), quiet(np.nanmax, per_fold, axis=1)


def official_tables(df: pd.DataFrame, sites: pd.DataFrame, out: Path, receptive_field: int):
    """One dataset through the official package's aggregate stage: write its two input CSVs under `out`, run
    calc_interaction and calc_coop_score on them as QuickStart does, and return (the qualified pairs, with i =
    interaction_t0; the tf_pair table; the tf table)."""
    from deepISA.score import aggregate_isa as dag

    with _object_strings():
        return _official_tables(dag, df, sites, out, receptive_field)


def _object_strings():
    """pandas' object-dtype strings while the package runs: it was written for pandas 2 (np.minimum on its tf
    columns), and pandas 3 infers a string dtype by default."""
    import contextlib

    try:
        pd.get_option("future.infer_string")
    except (KeyError, pd.errors.OptionError):  # an older pandas: object strings already
        return contextlib.nullcontext()
    return pd.option_context("future.infer_string", False)


def _official_tables(dag, df: pd.DataFrame, sites: pd.DataFrame, out: Path, receptive_field: int):
    out.mkdir(parents=True, exist_ok=True)
    single_path, combi_path = out / "motif_single_isa.csv", out / "motif_combi_isa.csv"
    single = sites.rename(columns={"peak_id": "region", "e": "isa_t0"})
    single[["region", "chr", "start", "end", "motif", "tf", "isa_t0"]].to_csv(single_path, index=False)
    kept = df[(df["gap"] >= 1) & (df["gap"] <= receptive_field)]
    combi = pd.DataFrame({"region": kept["peak_id"], "tf1": kept["tf_a"], "tf2": kept["tf_b"],
                          "motif1": kept["motif_a"], "motif2": kept["motif_b"], "distance": kept["gap"],
                          "isa1_t0": kept["e_a"], "isa2_t0": kept["e_b"], "isa_both_t0": kept["isa_both"]})  # fmt: skip
    combi.to_csv(combi_path, index=False)
    dag.calc_interaction(combi_isa_path=str(combi_path), tracks=[0], tau_map=None,
                         isa_thresh_quantile=OFFICIAL["isa_thresh_quantile"], single_isa_path=str(single_path),
                         normalize=False)  # fmt: skip
    tables = {}
    for level, name in (("tf_pair", "coop_tf_pair_t0.csv"), ("tf", "coop_tf_t0.csv")):
        tables[level] = dag.calc_coop_score(str(combi_path), outpath=str(out / name), level=level, track_idx=0,
                                            min_count=OFFICIAL["min_count"], q_val_thresh=OFFICIAL["q_val_thresh"])  # fmt: skip
    scored = pd.read_csv(combi_path)
    q = kept.assign(i=scored["interaction_t0"].to_numpy())
    return q[np.isfinite(q["i"])].copy(), tables["tf_pair"], tables["tf"]


def run_official(args, pairs: dict, names: dict, groups: dict, comparisons: list, series: list) -> dict:
    out = args.root / "official"
    qualified, by_pair, by_tf, summary = {}, [], [], {}
    for ds, df in pairs.items():
        sites = pd.read_csv(args.root / ds / "sites.tsv.gz", sep="\t")
        sites["tf"] = sites["motif"].map(names).fillna(sites["motif"])
        q, tp, tf = official_tables(df, sites, out / ds, args.receptive_field)
        qualified[ds] = q
        for table, acc in ((tp, by_pair), (tf, by_tf)):
            if not table.empty:
                acc.append(table.assign(dataset=ds))
        i = q["i"].to_numpy()
        summary[ds] = {
            "pairs_within_receptive_field": int(((df["gap"] >= 1) & (df["gap"] <= args.receptive_field)).sum()),
            "qualified": len(q),
            "coop_score": float(i.sum() / np.abs(i).sum()) if len(i) and np.abs(i).sum() > 0 else None,
            "tf_pair_labels": tp["cooperativity"].value_counts().to_dict() if not tp.empty else {},
            "tf_labels": tf["cooperativity"].value_counts().to_dict() if not tf.empty else {},
        }
    for acc, name in ((by_pair, "coop_tf_pairs_by_dataset.tsv"), (by_tf, "coop_tfs_by_dataset.tsv")):
        if acc:
            table = pd.concat(acc)
            table.insert(0, "dataset", table.pop("dataset"))
            table.to_csv(out / name, sep="\t", index=False, float_format="%.5g")
    coop = lambda x: 2 * np.asarray(x, dtype=float) - 1  # noqa: E731  sum(i)/sum|i| from the synergy at t = 0
    for level, key in (("pairs", "pair"), ("tfs", "tf")):
        ch = changes(qualified, groups, comparisons, key, 0.0, args.n_boot, args.min_n, transform=coop, fdr=args.fdr)
        if ch is not None:
            ch = ch.rename(columns={"synergy_a": "coop_a", "synergy_b": "coop_b", "lo_a": "coop_lo_a",
                                    "hi_a": "coop_hi_a", "lo_b": "coop_lo_b", "hi_b": "coop_hi_b"})  # fmt: skip
            ch.drop(columns=[c for c in ch.columns if c.startswith(("label_", "independence_"))], inplace=True)
            ch.to_csv(out / f"changes_{level}.tsv", sep="\t", index=False, float_format="%.5g")
        summary[f"changes_{level}"] = change_counts(ch, comparisons)
        if series:
            tr = trend(qualified, groups, series, key, 0.0, args.n_boot, args.min_n, transform=coop, fdr=args.fdr,
                       score="coop")  # fmt: skip
            if tr is not None:
                tr.to_csv(out / f"trend_{level}.tsv", sep="\t", index=False, float_format="%.5g")
            summary[f"trend_{level}"] = trend_counts(tr)
    summary["settings"] = {**OFFICIAL, "receptive_field": args.receptive_field, "null": "pairs > 150 bp apart",
                           "fdr": args.fdr, "series": series}  # fmt: skip
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="coop_isa.py's --out-dir (<dataset>/pairs.tsv.gz)")
    ap.add_argument("--motifs", type=Path, required=True, help="coop_isa.py's --motifs (motif, cluster_annotation)")
    ap.add_argument("--groups", type=Path, required=True, help="TSV: group, datasets (comma-separated)")
    ap.add_argument("--comparisons", type=Path, required=True, help="TSV: name, group_a, group_b")
    ap.add_argument("--t-coop", type=float, default=0.1)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--min-n", type=int, default=50)
    ap.add_argument("--fdr", type=float, default=0.05, help="BH q below which a change or a trend is significant")
    ap.add_argument("--series", type=Path, help="TSV with column group: the groups in order, for the trends")
    ap.add_argument("--official", action="store_true",
                    help="also the official deepISA package's rules, in <root>/official/ (see the module doc)")  # fmt: skip
    ap.add_argument("--receptive-field", type=int, default=255,
                    help="--official: the largest edge-to-edge gap of a pair, as its make_pairs_for_region")  # fmt: skip
    args = ap.parse_args()
    groups = {r["group"]: [d.strip() for d in r["datasets"].split(",")] for r in read_tsv(args.groups)}
    comparisons = [(r["name"], r["group_a"], r["group_b"]) for r in read_tsv(args.comparisons)]
    for name, ga, gb in comparisons:
        for g in (ga, gb):
            if g not in groups:
                ap.error(f"comparison {name}: group {g!r} is not in {args.groups}")
    series = [r["group"] for r in read_tsv(args.series)] if args.series else []
    for g in series:
        if g not in groups:
            ap.error(f"series: group {g!r} is not in {args.groups}")
    if args.series and len(series) < 3:
        ap.error("a series needs at least 3 groups")
    out = args.root / "combined"
    out.mkdir(exist_ok=True)
    datasets = sorted({d for g in groups.values() for d in g})
    names = pd.read_csv(args.motifs, sep="\t", usecols=["motif", "cluster_annotation"]).drop_duplicates("motif")
    names = dict(zip(names["motif"], names["cluster_annotation"].str.replace(r"_\d+$", "", regex=True)))
    pairs = {}
    for ds in datasets:
        p = args.root / ds / "pairs.tsv.gz"
        if p.is_file():
            df = pd.read_csv(p, sep="\t")
            for x in ("a", "b"):  # each site's TF from its motif; a motif without an annotation keeps its own name
                df[f"tf_{x}"] = df[f"motif_{x}"].map(names).fillna(df[f"motif_{x}"])
            sw = df["tf_a"] > df["tf_b"]  # each pair's names in sorted order
            df.loc[sw, ["tf_a", "tf_b"]] = df.loc[sw, ["tf_b", "tf_a"]].to_numpy()
            df["peak"] = ds + ":" + df["peak_id"].astype(str)
            df["pair"] = df["tf_a"] + " + " + df["tf_b"]
            pairs[ds] = df
    print(len(pairs), "datasets read")
    t = args.t_coop

    by_ds, tfs_ds, dist_rows, summary = [], [], [], {}
    for ds, df in pairs.items():
        s, _ = synergy_boot(df, "pair", t, args.n_boot, 1)
        s.insert(0, "dataset", ds)
        long = pd.concat([df.assign(tf=df["tf_a"]), df.assign(tf=df["tf_b"])])
        st, _ = synergy_boot(long, "tf", t, args.n_boot, 2)
        st.insert(0, "dataset", ds)
        for table, src, key in ((s, df, "pair"), (st, long, "tf")):
            r = fold_range(src, key, t, table[key])
            if r is not None:
                table["synergy_fold_min"], table["synergy_fold_max"] = r
        by_ds.append(s[s["n"] > 10])
        tfs_ds.append(st[st["n"] > 10])
        df["dbin"] = pd.cut(df["distance"], DIST_BINS, right=False)
        for b, g in df.groupby("dbin", observed=True):
            i = g["i"].to_numpy()
            dist_rows.append(
                {
                    "dataset": ds,
                    "distance": str(b),
                    "n": len(g),
                    "synergistic": float(np.mean(i > t)),
                    "redundant": float(np.mean(i < -t)),
                    "independent": float(np.mean(np.abs(i) <= t)),
                    "mean_abs_i": float(np.mean(np.abs(i))),
                }
            )
        i = df["i"].to_numpy()
        summary[ds] = {
            "pairs": len(df),
            "independent": float(np.mean(np.abs(i) <= t)),
            "synergistic": float(np.mean(i > t)),
            "redundant": float(np.mean(i < -t)),
            "synergy": float(i[i > t].sum() / np.abs(i[np.abs(i) > t]).sum()),
            "median_abs_i": float(np.median(np.abs(i))),
            "median_abs_e": float(np.median(np.abs(np.r_[df["e_a"].to_numpy(), df["e_b"].to_numpy()]))),
            "median_a": float(df["a"].median()) if "a" in df else None,
            "tf_pair_labels": s[s["n"] > 10]["label"].value_counts().to_dict(),
            "tf_labels": st[st["n"] > 10]["label"].value_counts().to_dict(),
        }
    pd.concat(by_ds).to_csv(out / "pairs_by_dataset.tsv", sep="\t", index=False, float_format="%.5g")
    pd.concat(tfs_ds).to_csv(out / "tfs_by_dataset.tsv", sep="\t", index=False, float_format="%.5g")
    pd.DataFrame(dist_rows).to_csv(out / "distance.tsv", sep="\t", index=False, float_format="%.5g")

    for level, key in (("pairs", "pair"), ("tfs", "tf")):
        ch = changes(pairs, groups, comparisons, key, t, args.n_boot, args.min_n, fdr=args.fdr)
        if ch is not None:
            ch.to_csv(out / f"changes_{level}.tsv", sep="\t", index=False, float_format="%.5g")
        summary[f"changes_{level}"] = change_counts(ch, comparisons)
        if series:
            tr = trend(pairs, groups, series, key, t, args.n_boot, args.min_n, fdr=args.fdr)
            if tr is not None:
                tr.to_csv(out / f"trend_{level}.tsv", sep="\t", index=False, float_format="%.5g")
            summary[f"trend_{level}"] = trend_counts(tr)
    summary["settings"] = {"t_coop": t, "n_boot": args.n_boot, "min_n": args.min_n, "fdr": args.fdr, "series": series}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1)[:3000])
    if args.official:
        official = run_official(args, pairs, names, groups, comparisons, series)
        print("official:", json.dumps({k: v for k, v in official.items() if not k.startswith("changes")})[:2000])


if __name__ == "__main__":
    main()
