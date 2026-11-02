#!/usr/bin/env python3
"""TF cooperativity from Cherimoya by in-silico ablation (ISA), with DeepISA's definitions (He et al. 2025).

For one dataset: peaks with at least two activating motif instances (Fi-NeMo hits whose motif name starts with pos_,
--finemo-dir's hits_unique.tsv), a random --n-peaks of them (--seed), at most --max-sites strongest non-overlapping
sites each. Every peak's 2,114-bp window (centred on its 1,000-bp Fi-NeMo region) is predicted intact, with each site
ablated (its bases set to N, an all-zero one-hot column) and with each pair ablated; by every fold model (--models,
{fold} in the path), both strands (strand mean of the counts, as scores.py), the folds' log counts averaged; log2
units.

  a        log2 predicted counts of the intact window; a_-A with site A ablated; a_-AB with A and B
  e_A      a - a_-A                      (the site's effect)
  i_AB     e_A - e_A|-B = a - a_-A - a_-B + a_-AB   (symmetric in A and B; > 0 synergy, < 0 redundancy)

Per TF pair (--motifs' cluster_annotation without its _<n> suffix, or the motif's own name where it has none;
unordered: each pair's sides are in TF-name order, motif_a, tf_a and e_a being one site) and per TF (over all its
partners):
  n, independence = share with |i| <= --t-ind, synergy = sum(i > t) / sum(|i| > t), redundancy = 1 - synergy (t =
  --t-coop), mean i, median distance; pairs with n >= --min-n are reported. DeepISA's thresholds were set on its own
  log-scale outputs (0.01 and 0.1); here they are on log2 counts. combine.py holds the definitions for the results.

Out (--out-dir/<dataset>/): sites.tsv.gz, pairs.tsv.gz (one row per site pair), tf_pairs.tsv, tfs.tsv, run.json.
pairs.tsv.gz also holds what the official deepISA package's rules read (combine.py --official): gap, the
edge-to-edge distance in bp (the later site's start minus the earlier site's end; `distance` is centre to centre),
and isa_both = a - a_-AB, the pair's joint effect; e_a and e_b are its isa1 and isa2.

    coop_isa.py --dataset <name> --models <dir>/fold_{fold}/cherimoya.fold_{fold}.torch --finemo-dir <dir>
        --motifs <motifs.tsv> --fasta <genome.fa> --out-dir <dir>
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("qtl", "variant_scoring")]
import peak_scores as ps  # noqa: E402
import predict_cherimoya as pcm  # noqa: E402

HALF, L = 1057, 2114
LN2 = math.log(2.0)


def greedy_sites(h: pd.DataFrame, k: int) -> pd.DataFrame:
    """At most k non-overlapping sites, strongest (hit_coefficient) first."""
    keep, taken = [], []
    for r in h.sort_values("hit_coefficient", ascending=False).itertuples():
        if any(not (r.end <= s or r.start >= e) for s, e in taken):
            continue
        keep.append(r.Index)
        taken.append((r.start, r.end))
        if len(keep) == k:
            break
    return h.loc[keep].sort_values("start")


def aggregate(pairs: pd.DataFrame, keys: list[str], t_ind: float, t_coop: float, min_n: int) -> pd.DataFrame:
    rows = []
    for k, g in pairs.groupby(keys):
        i = g["i"].to_numpy()
        big = np.abs(i) > t_coop
        den = np.abs(i[big]).sum()
        rows.append(
            {
                **dict(zip(keys, k if isinstance(k, tuple) else (k,))),
                "n": len(g),
                "independence": float(np.mean(np.abs(i) <= t_ind)),
                "n_coop": int(big.sum()),
                "synergy": float(i[i > t_coop].sum() / den) if den > 0 else np.nan,
                "mean_i": float(i.mean()),
                "median_distance": float(g["distance"].median()),
            }
        )
    out = pd.DataFrame(rows)
    out["redundancy"] = 1 - out["synergy"]
    return out[out["n"] >= min_n].sort_values("n", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="the name of the output folder and of run.json's dataset")
    ap.add_argument("--models", required=True, help="the fold models' path, with {fold} where the fold goes")
    ap.add_argument("--finemo-dir", type=Path, required=True, help="Fi-NeMo's peaks_qc.tsv and hits_unique.tsv")
    ap.add_argument("--motifs", type=Path, required=True, help="TSV with columns motif and cluster_annotation")
    ap.add_argument("--fasta", type=Path, required=True, help="the genome the models were trained on (and its .fai)")
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-peaks", type=int, default=50000)
    ap.add_argument("--max-sites", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--t-ind", type=float, default=0.01)
    ap.add_argument("--t-coop", type=float, default=0.1)
    ap.add_argument("--min-n", type=int, default=11)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    t0 = time.time()
    ds = args.dataset
    out = args.out_dir / ds
    out.mkdir(parents=True, exist_ok=True)
    fin = args.finemo_dir
    peaks = pd.read_csv(fin / "peaks_qc.tsv", sep="\t", usecols=["peak_id", "chr", "peak_region_start"])
    hits = pd.read_csv(
        fin / "hits_unique.tsv",
        sep="\t",
        usecols=["chr", "start", "end", "motif_name", "strand", "hit_coefficient", "peak_id"],
    )
    hits = hits[hits["motif_name"].str.startswith("pos_")]
    ann = pd.read_csv(args.motifs, sep="\t", usecols=["motif", "cluster_annotation"]).drop_duplicates("motif")
    ann = dict(zip(ann["motif"], ann["cluster_annotation"].str.replace(r"_\d+$", "", regex=True)))
    hits["tf"] = hits["motif_name"].map(ann).fillna(hits["motif_name"])  # no annotation: the motif's own name
    counts = hits.groupby("peak_id").size()
    eligible = counts[counts >= 2].index.to_numpy()
    rng = np.random.default_rng(args.seed)
    chosen = np.sort(rng.choice(eligible, size=min(args.n_peaks, len(eligible)), replace=False))
    print(f"{ds}: {len(peaks)} peaks, {len(eligible)} with >= 2 activating sites, {len(chosen)} sampled", flush=True)

    genome = ps.Genome(args.fasta)
    pk = peaks.set_index("peak_id")
    hg = hits[hits["peak_id"].isin(chosen)].groupby("peak_id")
    seqs, site_rows, pair_rows, peak_rows = [], [], [], []
    for pid in chosen:
        chrom, rstart = pk.at[pid, "chr"], int(pk.at[pid, "peak_region_start"])
        w0 = rstart + 500 - HALF
        base = genome.codes(chrom, w0, w0 + L)
        if base is None:
            continue
        s = greedy_sites(hg.get_group(pid), args.max_sites)
        if len(s) < 2:
            continue
        spans = [(int(r.start) - w0, int(r.end) - w0) for r in s.itertuples()]
        i0 = len(seqs)
        seqs.append(base)
        single = []
        for a, b in spans:
            x = base.copy()
            x[a:b] = 4
            single.append(len(seqs))
            seqs.append(x)
        pair_idx = {}
        for p in range(len(spans)):
            for q in range(p + 1, len(spans)):
                x = base.copy()
                x[spans[p][0] : spans[p][1]] = 4
                x[spans[q][0] : spans[q][1]] = 4
                pair_idx[(p, q)] = len(seqs)
                seqs.append(x)
        peak_rows.append((pid, chrom, w0, i0, single, pair_idx, s))
    X = np.stack(seqs)
    print(f"{ds}: {len(peak_rows)} peaks, {len(X):,} sequences", flush=True)

    pcm.set_precision("default")
    acc = np.zeros(len(X), dtype=np.float64)
    for fold in args.folds:
        t1 = time.time()
        model = pcm.load_model(Path(args.models.format(fold=fold)), "cuda")
        pr = pcm.CountPredictor(model, "cuda").predict(X, args.batch)
        acc += ps.strand_mean(pr["fwd"], pr["rev"])
        del model
        print(f"{ds}: fold {fold} in {time.time() - t1:.0f} s", flush=True)
    lc = acc / len(args.folds) / LN2  # log2 counts, fold mean

    for pid, chrom, _w0, i0, single, pair_idx, s in peak_rows:
        a = lc[i0]
        tfs, motifs, starts = s["tf"].tolist(), s["motif_name"].tolist(), s["start"].tolist()
        ends = s["end"].tolist()
        e = [a - lc[j] for j in single]
        for k in range(len(single)):
            site_rows.append((pid, chrom, starts[k], ends[k], motifs[k], tfs[k], e[k], a))
        for (p, q), j in pair_idx.items():
            i = a - lc[single[p]] - lc[single[q]] + lc[j]
            dist = abs((starts[q] + ends[q]) / 2 - (starts[p] + ends[p]) / 2)
            gap = starts[q] - ends[p]  # sites are in start order and do not overlap (greedy_sites)
            u, v = (q, p) if tfs[q] < tfs[p] else (p, q)  # sides in TF-name order, each column with its own site
            pair_rows.append((pid, chrom, motifs[u], motifs[v], tfs[u], tfs[v], dist, gap, e[u], e[v], a - lc[j], i))
    sites = pd.DataFrame(site_rows, columns=["peak_id", "chr", "start", "end", "motif", "tf", "e", "a"])
    pairs = pd.DataFrame(
        pair_rows,
        columns=["peak_id", "chr", "motif_a", "motif_b", "tf_a", "tf_b", "distance", "gap", "e_a", "e_b", "isa_both",
                 "i"],
    )  # fmt: skip
    sites.to_csv(out / "sites.tsv.gz", sep="\t", index=False, float_format="%.5g")
    pairs.to_csv(out / "pairs.tsv.gz", sep="\t", index=False, float_format="%.5g")
    tfp = aggregate(pairs, ["tf_a", "tf_b"], args.t_ind, args.t_coop, args.min_n)
    tfp.to_csv(out / "tf_pairs.tsv", sep="\t", index=False, float_format="%.5g")
    long = pd.concat(
        [
            pairs.rename(columns={"tf_a": "tf"})[["tf", "i", "distance"]],
            pairs.rename(columns={"tf_b": "tf"})[["tf", "i", "distance"]],
        ]
    )
    tfs = aggregate(long, ["tf"], args.t_ind, args.t_coop, args.min_n)
    tfs.to_csv(out / "tfs.tsv", sep="\t", index=False, float_format="%.5g")
    info = {
        "dataset": ds,
        "folds": args.folds,
        "peaks": len(peak_rows),
        "sequences": int(len(X)),
        "sites": len(sites),
        "pairs": len(pairs),
        "tf_pairs_reported": len(tfp),
        "tfs_reported": len(tfs),
        "independence_all": float(np.mean(np.abs(pairs["i"]) <= args.t_ind)),
        "synergy_all": float(
            pairs["i"][pairs["i"] > args.t_coop].sum() / np.abs(pairs["i"][np.abs(pairs["i"]) > args.t_coop]).sum()
        ),
        "args": {k: str(v) for k, v in vars(args).items()},
        "seconds": round(time.time() - t0, 1),
    }
    (out / "run.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps({k: v for k, v in info.items() if k != "args"}), flush=True)


if __name__ == "__main__":
    main()
