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

Out (<root>/combined/): pairs_by_dataset.tsv, tfs_by_dataset.tsv, changes_pairs.tsv, changes_tfs.tsv,
distance.tsv, summary.json.

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
between groups are computed here from interaction_t0 instead: coop = sum(i) / sum(|i|) over the qualified pairs,
which is 2 x (this script's synergy at t = 0) - 1, with the same bootstrap. Differences from the package's own run:
its sites are every Fi-NeMo hit and its predictions one model on one strand; here they are coop_isa.py's (at most
--max-sites non-overlapping sites per sampled peak; both strands, the fold mean, log2 counts), so its 10th percentile
is taken over those sites. Out (<root>/official/): <dataset>/ as above, coop_tf_pairs_by_dataset.tsv,
coop_tfs_by_dataset.tsv, changes_pairs.tsv, changes_tfs.tsv, summary.json.

    combine.py --root <coop_isa.py --out-dir> --motifs <motifs.tsv> --groups <groups.tsv> --comparisons <comparisons.tsv>
        [--official]
"""

import argparse
import json
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


def synergy_boot(df: pd.DataFrame, key: str, t: float, n_boot: int, seed: int) -> pd.DataFrame:
    """Per value of `key`: n, synergy, independence, mean i, and the bootstrap replicates of synergy (columns b0..)."""
    codes, uniq = pd.factorize(df[key])
    peaks, _ = pd.factorize(df["peak"])
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
        w = rng.poisson(1.0, peaks.max() + 1)[peaks].astype(float)
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
    out["lo"], out["hi"] = np.nanpercentile(boots, 2.5, axis=0), np.nanpercentile(boots, 97.5, axis=0)
    return out, boots


def changes(
    pairs: dict, groups: dict, comparisons: list, key: str, t: float, n_boot: int, min_n: int, transform=None
) -> pd.DataFrame | None:
    """Per comparison: each TF pair's (or TF's) synergy in group a and b, the change b - a and its bootstrap 95% CI,
    kept where both groups have >= min_n pairs. `transform` maps a synergy (and its replicates) to the score compared."""
    rows = []
    for name, ga, gb in comparisons:
        res = {}
        for g, seed in ((ga, 11), (gb, 12)):
            dfs = [pairs[d] for d in groups[g] if d in pairs]
            if not dfs:
                break
            df = pd.concat(dfs)
            if key == "tf":
                df = pd.concat([df.assign(tf=df["tf_a"]), df.assign(tf=df["tf_b"])])
            score, boots = synergy_boot(df, key, t, n_boot, seed)
            if transform is not None:
                score = score.assign(synergy=transform(score["synergy"]), lo=transform(score["lo"]),
                                     hi=transform(score["hi"]))  # fmt: skip
                boots = transform(boots)
            res[g] = score, boots
        if len(res) < 2:
            continue
        (a, boot_a), (b, boot_b) = res[ga], res[gb]
        m = a.merge(b, on=key, suffixes=("_a", "_b"))
        ba = pd.DataFrame(boot_a, columns=a[key]).reindex(columns=m[key])
        bb = pd.DataFrame(boot_b, columns=b[key]).reindex(columns=m[key])
        delta = bb.to_numpy() - ba.to_numpy()
        m["delta"] = m["synergy_b"] - m["synergy_a"]
        m["delta_lo"], m["delta_hi"] = np.nanpercentile(delta, 2.5, axis=0), np.nanpercentile(delta, 97.5, axis=0)
        m = m[(m["n_a"] >= min_n) & (m["n_b"] >= min_n)].copy()
        m.insert(0, "comparison", name)
        m.insert(1, "group_a", ga)
        m.insert(2, "group_b", gb)
        m["significant"] = (m["delta_lo"] > 0) | (m["delta_hi"] < 0)
        rows.append(m)
    return pd.concat(rows) if rows else None


def change_counts(ch: pd.DataFrame | None, comparisons: list) -> dict:
    return {
        n: {
            "tested": 0 if ch is None else int((ch["comparison"] == n).sum()),
            "significant": 0 if ch is None else int(ch[ch["comparison"] == n]["significant"].sum()),
        }
        for n, _, _ in comparisons
    }


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


def run_official(args, pairs: dict, names: dict, groups: dict, comparisons: list) -> dict:
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
        ch = changes(qualified, groups, comparisons, key, 0.0, args.n_boot, args.min_n, transform=coop)
        if ch is not None:
            ch = ch.rename(columns={"synergy_a": "coop_a", "synergy_b": "coop_b", "lo_a": "coop_lo_a",
                                    "hi_a": "coop_hi_a", "lo_b": "coop_lo_b", "hi_b": "coop_hi_b"})  # fmt: skip
            ch.drop(columns=[c for c in ch.columns if c.startswith(("label_", "independence_"))], inplace=True)
            ch.to_csv(out / f"changes_{level}.tsv", sep="\t", index=False, float_format="%.5g")
        summary[f"changes_{level}"] = change_counts(ch, comparisons)
    summary["settings"] = {**OFFICIAL, "receptive_field": args.receptive_field, "null": "pairs > 150 bp apart"}
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
        by_ds.append(s[s["n"] > 10])
        long = pd.concat([df.assign(tf=df["tf_a"]), df.assign(tf=df["tf_b"])])
        st, _ = synergy_boot(long, "tf", t, args.n_boot, 2)
        st.insert(0, "dataset", ds)
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
            "tf_pair_labels": s[s["n"] > 10]["label"].value_counts().to_dict(),
            "tf_labels": st[st["n"] > 10]["label"].value_counts().to_dict(),
        }
    pd.concat(by_ds).to_csv(out / "pairs_by_dataset.tsv", sep="\t", index=False, float_format="%.5g")
    pd.concat(tfs_ds).to_csv(out / "tfs_by_dataset.tsv", sep="\t", index=False, float_format="%.5g")
    pd.DataFrame(dist_rows).to_csv(out / "distance.tsv", sep="\t", index=False, float_format="%.5g")

    for level, key in (("pairs", "pair"), ("tfs", "tf")):
        ch = changes(pairs, groups, comparisons, key, t, args.n_boot, args.min_n)
        if ch is not None:
            ch.to_csv(out / f"changes_{level}.tsv", sep="\t", index=False, float_format="%.5g")
        summary[f"changes_{level}"] = change_counts(ch, comparisons)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1)[:3000])
    if args.official:
        official = run_official(args, pairs, names, groups, comparisons)
        print("official:", json.dumps({k: v for k, v in official.items() if not k.startswith("changes")})[:2000])


if __name__ == "__main__":
    main()
