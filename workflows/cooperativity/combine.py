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

    combine.py --root <coop_isa.py --out-dir> --motifs <motifs.tsv> --groups <groups.tsv> --comparisons <comparisons.tsv>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DIST_BINS = [0, 20, 50, 100, 200, 500, 1001]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="coop_isa.py's --out-dir (<dataset>/pairs.tsv.gz)")
    ap.add_argument("--motifs", type=Path, required=True, help="coop_isa.py's --motifs (motif, cluster_annotation)")
    ap.add_argument("--groups", type=Path, required=True, help="TSV: group, datasets (comma-separated)")
    ap.add_argument("--comparisons", type=Path, required=True, help="TSV: name, group_a, group_b")
    ap.add_argument("--t-coop", type=float, default=0.1)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--min-n", type=int, default=50)
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
                res[g] = synergy_boot(df, key, t, args.n_boot, seed)
            if len(res) < 2:
                continue
            (a, boot_a), (b, boot_b) = res[ga], res[gb]
            m = a.merge(b, on=key, suffixes=("_a", "_b"))
            ba = pd.DataFrame(boot_a, columns=a[key]).reindex(columns=m[key])
            bb = pd.DataFrame(boot_b, columns=b[key]).reindex(columns=m[key])
            delta = bb.to_numpy() - ba.to_numpy()
            m["delta"] = m["synergy_b"] - m["synergy_a"]
            m["delta_lo"], m["delta_hi"] = np.nanpercentile(delta, 2.5, axis=0), np.nanpercentile(delta, 97.5, axis=0)
            m = m[(m["n_a"] >= args.min_n) & (m["n_b"] >= args.min_n)].copy()
            m.insert(0, "comparison", name)
            m.insert(1, "group_a", ga)
            m.insert(2, "group_b", gb)
            m["significant"] = (m["delta_lo"] > 0) | (m["delta_hi"] < 0)
            rows.append(m)
        ch = pd.concat(rows)
        ch.to_csv(out / f"changes_{level}.tsv", sep="\t", index=False, float_format="%.5g")
        summary[f"changes_{level}"] = {
            n: {
                "tested": int((ch["comparison"] == n).sum()),
                "significant": int(ch[(ch["comparison"] == n)]["significant"].sum()),
            }
            for n, _, _ in comparisons
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1)[:3000])


if __name__ == "__main__":
    main()
