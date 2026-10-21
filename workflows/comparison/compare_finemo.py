#!/usr/bin/env python3
"""Level 5 of the comparison: the two models' Fi-NeMo hits.

--dataset <ds>  both models' hits (10.0's hits.tsv) called with one shared vocabulary, the joint compendium
    (--joint, joint20): <output_dir>/finemo_<joint>/<ds>_<peak_type>/ of each model, made from the same
    modisco_compiled.h5 (their compendium.md5 must agree). hits.tsv names each hit's peak (peak_name, the narrowPeak's
    column 4) and its row in the bed Fi-NeMo read (peak_id: fold 0's interpreted_regions.bed); every hit is checked to
    name the peak of its row there, and only hits on peaks both beds hold are compared. A hit of one model is matched
    when the other model has a hit of the same motif on the same strand whose centre (the midpoint of the trimmed
    start and end) lies within --tolerance bp (default 5; 0 and 10 are also reported). Writes <out>/<ds>/:
      motifs.tsv        per motif: hits per 10,000 shared peaks for each model, matched hits, Cherimoya's precision
                        (its matched hits / its hits) and recall (ChromBPNet's matched hits / ChromBPNet's hits) at each
                        tolerance, the Jaccard index of the peaks with at least one hit of the motif, the Spearman
                        correlation of hit_coefficient_global over matched pairs (each hit with its nearest partner),
                        and, with level 4's joint_clusters.tsv, the cluster's class and annotation
      peaks.tsv.gz      per shared peak with a hit: each model's hit count and the Pearson correlation of the
                        two models' motif-count vectors (over every motif either model hit in the dataset)
      motif_report.tsv  11.0's finemo_report/motif_report.tsv of both models side by side, by motif
      hits.json         the totals: hits, shared peaks, precision and recall at each tolerance, the median per-peak
                        correlation
--dataset <ds> --production  the secondary comparison: each model's production hits (ChromBPNet's on --cbp-tag,
    cbp10; Cherimoya's on --cheri-tag, cheri39), whose vocabularies differ, compared through level 4's reciprocal
    cluster matches (<comparison-root>/compendium/matches_<cbp>_vs_<cheri>.tsv): a hit's motif is its matched pair, and
    hits of motifs without a pair are counted but never match. Writes <out>/<ds>/production/.
--summarize --datasets <list>  summary.tsv (one row per dataset), motifs.tsv (every dataset's), statistics.tsv
    (medians over datasets with bootstrap CIs) and agreement.pdf/.png.

Waiting (exit 3, nothing written): a model's hits.bed.gz (10.0's done marker) or compendium.md5 not there yet.
Failed (exit 1): the two models' hits came from different compendia, or a hit names another peak than its row.

    python workflows/comparison/compare_finemo.py --dataset <ds> [--joint joint20] [--tolerance 5]
    python workflows/comparison/compare_finemo.py --dataset <ds> --production [--cbp-tag cbp10 --cheri-tag cheri39]
    python workflows/comparison/compare_finemo.py --summarize --datasets <list.txt>

Runs in the pixi cherimoya environment (pandas, scipy, matplotlib), inside a box.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1] / d) for d in ("comparison", "train", "attribution", "downstream")
]

import common  # noqa: E402
import to_chrombpnet_scores as tcs  # noqa: E402

logger = logging.getLogger("compare_finemo")

FORMAT = 1
TOLERANCES = (0, 5, 10)
HIT_COLUMNS = ["chr", "start", "end", "motif_name", "hit_coefficient_global", "strand", "peak_name", "peak_id"]


# ---------------------------------------------------------------------------------------------------------------------
# where things are


def hits_dirs(cheri_root: Path, ds: str, tags: dict[str, str]) -> dict:
    """Each model's Fi-NeMo directory for its tag ({model: tag}) and the bed its peak ids index (fold 0's)."""
    try:
        cfg = tcs.read_cheri_config(cheri_root / "configs" / ds / "config.yaml")
    except tcs.Refused as exc:
        raise common.Waiting(str(exc)) from exc
    k0, pt = str(cfg["folds"][0]), cfg["peak_type"]
    try:
        cbp_bed = tcs.chrombpnet_interpretation(cheri_root / ds / f"fold_{k0}", common.HEAD)["bed"]
    except tcs.Refused as exc:
        raise common.Waiting(str(exc)) from exc
    cbp_fold = cbp_bed.parent.parent
    cbp_out = cbp_fold.parent.parent
    cbp_pt = cbp_fold.name[len(ds) + 1 : cbp_fold.name.rfind("_fold_")] or pt
    out = Path(cfg["output_dir"])
    return {
        "chrombpnet": {"dir": cbp_out / f"finemo_{tags['chrombpnet']}" / f"{ds}_{cbp_pt}", "bed": cbp_bed},
        "cherimoya": {"dir": out / f"finemo_{tags['cherimoya']}" / f"{ds}_{pt}",
                      "bed": out / "full_models" / f"{ds}_{pt}_fold_{k0}" / "interpretation" /
                      "interpretation.interpreted_regions.bed"},
    }  # fmt: skip


def read_hits(path: Path, bed):
    """hits.tsv's columns that matter, each hit checked to name the peak of its peak_id row in the bed."""
    import pandas as pd

    hits = pd.read_csv(path, sep="\t", usecols=lambda c: c in HIT_COLUMNS, dtype={"chr": str, "motif_name": str,
                       "strand": str, "peak_name": str})  # fmt: skip
    missing = [c for c in HIT_COLUMNS if c not in hits]
    if missing:
        raise common.Failed(f"{path} has no {', '.join(missing)} column")
    ids = hits["peak_id"].to_numpy(dtype=np.int64)
    if len(ids) and (ids.min() < 0 or ids.max() >= len(bed)):
        raise common.Failed(f"{path}: peak_id {int(ids.max())} beyond the {len(bed)} rows of its bed")
    named = bed["name"].to_numpy()[ids]
    bad = int((named != hits["peak_name"].to_numpy()).sum())
    if bad:
        raise common.Failed(f"{path}: {bad} hits name another peak than their row of the bed")
    return hits


# ---------------------------------------------------------------------------------------------------------------------
# matching


def match(a, b, tolerance: int) -> tuple[np.ndarray, np.ndarray]:
    """For each hit of a: whether b has a hit of the same motif, strand and chromosome whose centre is within
    `tolerance` bp, and the row of b's nearest one (-1 if none)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros(len(a), bool), np.full(len(a), -1, dtype=np.int64)
    keys = {k: i for i, k in enumerate(sorted(set(zip(a["chr"], a["motif"], a["strand"]))
                                             | set(zip(b["chr"], b["motif"], b["strand"]))))}  # fmt: skip
    ka = np.array([keys[k] for k in zip(a["chr"], a["motif"], a["strand"])], dtype=np.int64)
    kb = np.array([keys[k] for k in zip(b["chr"], b["motif"], b["strand"])], dtype=np.int64)
    ca = a["start"].to_numpy(np.int64) + a["end"].to_numpy(np.int64)  # twice the centre
    cb = b["start"].to_numpy(np.int64) + b["end"].to_numpy(np.int64)
    shift = np.int64(1) << 34
    za, zb = ka * shift + ca, kb * shift + cb
    order = np.argsort(zb, kind="stable")
    zs = zb[order]
    pos = np.searchsorted(zs, za)
    best = np.full(len(za), -1, dtype=np.int64)
    best_d = np.full(len(za), np.iinfo(np.int64).max)
    for cand in (pos - 1, pos):
        ok = (cand >= 0) & (cand < len(zs))
        c = np.where(ok, cand, 0)
        d = np.abs(zs[c] - za)
        better = ok & (zs[c] // shift == ka) & (d < best_d)
        best = np.where(better, order[c], best)
        best_d = np.where(better, d, best_d)
    hit = best >= 0
    hit &= best_d <= 2 * tolerance
    return hit, np.where(hit, best, -1)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman's rho; NaN for fewer than 3 pairs or a constant side."""
    from scipy.stats import spearmanr

    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(spearmanr(x, y)[0])


def jaccard(x: set, y: set) -> float:
    u = x | y
    return len(x & y) / len(u) if u else np.nan


def compare_hits(hits: dict, beds: dict, motif_of: dict | None, tolerance: int) -> dict:
    """Both models' hits on the peaks both beds hold, matched at each tolerance; per-motif and per-peak tables."""
    import pandas as pd

    shared = set(beds["chrombpnet"]["name"]) & set(beds["cherimoya"]["name"])
    h = {}
    for m in common.MODELS:
        df = hits[m]
        df = df[df["peak_name"].isin(shared)].reset_index(drop=True)
        names = df["motif_name"] if motif_of is None else df["motif_name"].map(motif_of[m])
        h[m] = df.assign(motif=names.fillna(f"unpaired:{m}:" + df["motif_name"]))
    a, b = h["chrombpnet"], h["cherimoya"]
    n_peaks = len(shared)
    matched = {}
    for t in sorted(set(TOLERANCES) | {tolerance}):
        matched[("chrombpnet", t)], partner_a = match(a, b, t)
        matched[("cherimoya", t)], _ = match(b, a, t)
        if t == tolerance:
            pa = partner_a
    rows = []
    motifs = sorted(set(a["motif"]) | set(b["motif"]))
    for motif in motifs:
        ia, ib = np.flatnonzero(a["motif"].to_numpy() == motif), np.flatnonzero(b["motif"].to_numpy() == motif)
        row = {"motif": motif, "hits_chrombpnet": len(ia), "hits_cherimoya": len(ib),
               "per10k_chrombpnet": 1e4 * len(ia) / n_peaks if n_peaks else np.nan,
               "per10k_cherimoya": 1e4 * len(ib) / n_peaks if n_peaks else np.nan}  # fmt: skip
        for t in sorted(set(TOLERANCES) | {tolerance}):
            ma, mb = matched[("chrombpnet", t)][ia].sum(), matched[("cherimoya", t)][ib].sum()
            row[f"matched_chrombpnet_{t}bp"], row[f"matched_cherimoya_{t}bp"] = int(ma), int(mb)
            row[f"precision_{t}bp"] = mb / len(ib) if len(ib) else np.nan
            row[f"recall_{t}bp"] = ma / len(ia) if len(ia) else np.nan
        row["peak_jaccard"] = jaccard(set(a["peak_name"].to_numpy()[ia]), set(b["peak_name"].to_numpy()[ib]))
        sel = ia[pa[ia] >= 0]
        row["coefficient_spearman"] = spearman(a["hit_coefficient_global"].to_numpy()[sel],
                                               b["hit_coefficient_global"].to_numpy()[pa[sel]])  # fmt: skip
        row["coefficient_pairs"] = int(len(sel))
        rows.append(row)
    motif_table = pd.DataFrame(rows)

    # per peak: the two models' motif-count vectors over the whole vocabulary
    vocab = len(motifs)
    counts = []
    for m, df in (("chrombpnet", a), ("cherimoya", b)):
        c = df.groupby(["peak_name", "motif"]).size().rename(m)
        counts.append(c)
    joined = pd.concat(counts, axis=1).fillna(0.0).reset_index()
    joined["xy"] = joined["chrombpnet"] * joined["cherimoya"]
    joined["xx"] = joined["chrombpnet"] ** 2
    joined["yy"] = joined["cherimoya"] ** 2
    per = joined.groupby("peak_name")[["chrombpnet", "cherimoya", "xy", "xx", "yy"]].sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = per["xy"] - per["chrombpnet"] * per["cherimoya"] / vocab
        vx = per["xx"] - per["chrombpnet"] ** 2 / vocab
        vy = per["yy"] - per["cherimoya"] ** 2 / vocab
        r = cov / np.sqrt(vx * vy)
    peaks = pd.DataFrame({"peak": per.index, "hits_chrombpnet": per["chrombpnet"].astype(int).to_numpy(),
                          "hits_cherimoya": per["cherimoya"].astype(int).to_numpy(),
                          "count_correlation": r.to_numpy()})  # fmt: skip
    totals = {"shared_peaks": n_peaks, "hits": {m: int(len(h[m])) for m in common.MODELS},
              "hits_all_peaks": {m: int(len(hits[m])) for m in common.MODELS}, "motifs": vocab,
              "peaks_with_hits": int(len(peaks)), "median_count_correlation": float(np.nanmedian(peaks["count_correlation"]))
              if len(peaks) else np.nan}  # fmt: skip
    for t in sorted(set(TOLERANCES) | {tolerance}):
        totals[f"precision_{t}bp"] = float(matched[("cherimoya", t)].mean()) if len(b) else np.nan
        totals[f"recall_{t}bp"] = float(matched[("chrombpnet", t)].mean()) if len(a) else np.nan
    sel = np.flatnonzero(pa >= 0)
    totals["coefficient_spearman"] = spearman(a["hit_coefficient_global"].to_numpy()[sel],
                                              b["hit_coefficient_global"].to_numpy()[pa[sel]])  # fmt: skip
    return {"motifs": motif_table, "peaks": peaks, "totals": totals}


# ---------------------------------------------------------------------------------------------------------------------
# units


def read_md5(path: Path) -> str:
    return path.read_text().split()[0] if path.is_file() and path.read_text().split() else ""


def run_dataset(args: argparse.Namespace, out_root: Path) -> int:
    import pandas as pd

    ds = args.dataset
    if args.production:
        tags = {"chrombpnet": args.cbp_tag, "cherimoya": args.cheri_tag}
        unit, od = f"{ds}.production", out_root / ds / "production"
        matches = args.comparison_root / "compendium" / f"matches_{args.cbp_tag}_vs_{args.cheri_tag}.tsv"
    else:
        tags = {m: args.joint for m in common.MODELS}
        unit, od, matches = ds, out_root / ds, None
    where = hits_dirs(args.cheri_root, ds, tags)
    missing = [f"{m} {where[m]['dir'] / name}" for m in common.MODELS for name in ("hits.bed.gz", "compendium.md5",
                                                                                     "hits.tsv")
               if not (where[m]["dir"] / name).is_file()]  # fmt: skip
    missing += [f"{m} bed {where[m]['bed']}" for m in common.MODELS if not where[m]["bed"].is_file()]
    if matches is not None and not matches.is_file():
        missing.append(f"level 4's {matches}")
    if missing:
        raise common.Waiting(f"{ds}: " + "; ".join(missing[:3]) + (" ..." if len(missing) > 3 else ""))
    reports = {m: where[m]["dir"] / "finemo_report" / "motif_report.tsv" for m in common.MODELS}
    joint_clusters = args.comparison_root / "compendium" / "joint_clusters.tsv" if args.comparison_root else None
    inputs = {f"{m}/{name}": where[m]["dir"] / name for m in common.MODELS for name in ("hits.tsv", "compendium.md5")}
    inputs.update({f"{m}/bed": where[m]["bed"] for m in common.MODELS})
    inputs.update({f"{m}/report": reports[m] for m in common.MODELS})
    inputs.update(matches=matches, joint_clusters=None if args.production else joint_clusters)
    marker = common.Marker(out_root, unit, FORMAT, inputs, {"tags": tags, "tolerance": args.tolerance})
    if marker.current() is None and not args.force:
        common.result("up to date")
        return 0
    marker.remove()
    md5 = {m: read_md5(where[m]["dir"] / "compendium.md5") for m in common.MODELS}
    if not args.production and md5["chrombpnet"] != md5["cherimoya"]:
        raise common.Failed(f"{ds}: the models' hits come from different compendia ({md5})")
    beds = {m: common.read_bed(where[m]["bed"]) for m in common.MODELS}
    hits = {m: read_hits(where[m]["dir"] / "hits.tsv", beds[m]) for m in common.MODELS}
    motif_of = None
    if matches is not None:
        mt = pd.read_csv(matches, sep="\t", dtype={"cluster": str, "best": str})
        mt = mt[mt["reciprocal"].astype(str).str.lower().isin(("true", "1"))]
        motif_of = {"chrombpnet": {}, "cherimoya": {}}
        for cbp_id, cheri_id in zip(mt["cluster"], mt["best"]):
            for group in ("pos_patterns", "neg_patterns"):
                pair = f"pair:{cbp_id}|{cheri_id}"
                motif_of["chrombpnet"][f"{group}.{cbp_id}"] = pair
                motif_of["cherimoya"][f"{group}.{cheri_id}"] = pair
    res = compare_hits(hits, beds, motif_of, args.tolerance)
    table = res["motifs"]
    if not args.production and joint_clusters is not None and joint_clusters.is_file():
        jc = pd.read_csv(joint_clusters, sep="\t", dtype={"cluster_id": str}).set_index("cluster_id")
        cid = table["motif"].str.split(".", n=1).str[-1]
        for col in ("class", "annotation", "bias_share"):
            if col in jc:
                table[f"cluster_{col}"] = cid.map(jc[col])
    common.write_table(od / "motifs.tsv", table)
    common.write_table(od / "peaks.tsv.gz", res["peaks"])
    outputs = {"motifs": od / "motifs.tsv", "peaks": od / "peaks.tsv.gz"}
    if all(p.is_file() for p in reports.values()):
        rep = [pd.read_csv(reports[m], sep="\t") for m in common.MODELS]
        key = rep[0].columns[0]
        side = rep[0].merge(rep[1], on=key, how="outer", suffixes=("_chrombpnet", "_cherimoya"))
        outputs["motif_report"] = common.write_table(od / "motif_report.tsv", side)
    totals = dict(res["totals"], dataset=ds, tags=tags, compendium_md5=md5, tolerance=args.tolerance)
    common.write_json(od / "hits.json", totals)
    outputs["hits"] = od / "hits.json"
    t = args.tolerance
    note = (f"{totals['hits']['chrombpnet']} ChromBPNet and {totals['hits']['cherimoya']} Cherimoya hits on "
            f"{totals['shared_peaks']} shared peaks; at {t} bp Cherimoya's precision {totals[f'precision_{t}bp']:.2f}, "
            f"recall {totals[f'recall_{t}bp']:.2f}; median per-peak count correlation "
            f"{totals['median_count_correlation']:.2f}")  # fmt: skip
    marker.write(outputs, note)
    common.result(note)
    return 0


def summarize(args: argparse.Namespace, out_root: Path) -> int:
    import pandas as pd

    datasets = common.read_list(args.datasets)
    rows, motif_tables, waiting = [], [], []
    inputs: dict[str, Path] = {"list": args.datasets}
    for ds in datasets:
        marker = out_root / ".done" / f"{ds}.json"
        inputs[ds] = marker
        if not marker.is_file():
            waiting.append(ds)
            continue
        t = json.loads((out_root / ds / "hits.json").read_text())
        row = {"dataset": ds, "shared_peaks": t["shared_peaks"], "motifs": t["motifs"],
               "hits_chrombpnet": t["hits"]["chrombpnet"], "hits_cherimoya": t["hits"]["cherimoya"],
               "median_count_correlation": t["median_count_correlation"],
               "coefficient_spearman": t["coefficient_spearman"]}  # fmt: skip
        for key, value in t.items():
            if key.startswith(("precision_", "recall_")):
                row[key] = value
        rows.append(row)
        motif_tables.append(pd.read_csv(out_root / ds / "motifs.tsv", sep="\t").assign(dataset=ds))
    if not rows:
        common.result(f"waiting: no dataset compared yet (of {len(datasets)})")
        return common.WAITING
    marker = common.Marker(out_root, "summary", FORMAT, inputs, {"n_boot": args.n_boot, "seed": args.seed})
    if marker.current() is None and not args.force and not waiting:
        common.result("up to date")
        return 0
    marker.remove()
    summary = pd.DataFrame(rows)
    common.write_table(out_root / "summary.tsv", summary)
    common.write_table(out_root / "motifs.tsv", pd.concat(motif_tables, ignore_index=True))
    stats = []
    for col in [c for c in summary.columns if c.startswith(("precision_", "recall_"))] + [
        "median_count_correlation", "coefficient_spearman"]:  # fmt: skip
        est, lo, hi = common.bootstrap_ci(summary[col].to_numpy(dtype=np.float64), np.median, n_boot=args.n_boot,
                                          seed=args.seed)  # fmt: skip
        stats.append({"quantity": col, "n_datasets": int(summary[col].notna().sum()), "median": est, "ci95_lo": lo,
                      "ci95_hi": hi})  # fmt: skip
    d = (summary["precision_5bp"] - summary["recall_5bp"]).dropna().to_numpy() if "precision_5bp" in summary else []
    if len(d):
        est, lo, hi = common.bootstrap_ci(d, np.median, n_boot=args.n_boot, seed=args.seed)
        _, p, _ = common.wilcoxon(d)
        stats.append({"quantity": "precision_5bp - recall_5bp", "n_datasets": int(len(d)), "median": est,
                      "ci95_lo": lo, "ci95_hi": hi, "wilcoxon_p": p})  # fmt: skip
    common.write_table(out_root / "statistics.tsv", pd.DataFrame(stats))
    outputs = {"summary": out_root / "summary.tsv", "motifs": out_root / "motifs.tsv",
               "statistics": out_root / "statistics.tsv"}  # fmt: skip
    for path in plot(summary, out_root, datasets):
        outputs[path.name] = path
    note = (f"{len(rows)} of {len(datasets)} datasets: at 5 bp Cherimoya's precision {summary['precision_5bp'].median():.2f}, "
            f"recall {summary['recall_5bp'].median():.2f} (medians over datasets)") if "precision_5bp" in summary else (
            f"{len(rows)} of {len(datasets)} datasets")  # fmt: skip
    if waiting:
        common.result(f"waiting: {note}; not yet: {', '.join(waiting[:4])}{' ...' if len(waiting) > 4 else ''}")
        return common.WAITING
    marker.write(outputs, note)
    common.result(note)
    return 0


def plot(summary, out_root: Path, datasets: list[str]) -> list[Path]:
    plt = common.matplotlib_pyplot()
    styles, labels = common.dataset_styles(datasets), common.short_names(datasets)
    cols = [f"{kind}_{t}bp" for kind in ("precision", "recall") for t in TOLERANCES if f"{kind}_{t}bp" in summary]
    cols += [c for c in ("median_count_correlation", "coefficient_spearman") if c in summary]
    fig, ax = plt.subplots(figsize=(1.0 * len(cols) + 2.5, 3.6))
    for x, col in enumerate(cols):
        for ds, v in zip(summary["dataset"], summary[col]):
            if not np.isfinite(v):
                continue
            colour, marker = styles.get(ds, ("#999999", "o"))
            ax.scatter(x + (sorted(datasets).index(ds) - len(datasets) / 2) * 0.03, v, s=16, color=colour,
                       marker=marker, edgecolor="#333333", linewidth=0.3, zorder=2)  # fmt: skip
        med = float(np.nanmedian(summary[col])) if summary[col].notna().any() else np.nan
        ax.plot([x - 0.3, x + 0.3], [med, med], color="#333333", lw=1.2, zorder=3)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c.replace("_", " ") for c in cols], rotation=45, ha="right", fontsize=7)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("Cherimoya against ChromBPNet")
    ax.set_title("Fi-NeMo hits on the joint vocabulary, peaks of both beds", fontsize=8)
    ax.grid(axis="x", visible=False)
    common.dataset_legend(fig, styles, labels, f"{len(datasets)} datasets")
    fig.tight_layout()
    paths = common.save_figure(fig, out_root / "agreement")
    plt.close(fig)
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_root_args(ap)
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--dataset", help="compare this dataset's hits")
    what.add_argument("--summarize", action="store_true", help="summarize the datasets of --datasets")
    ap.add_argument("--production", action="store_true", help="the production hits, through level 4's matches")
    ap.add_argument("--datasets", type=Path, help="the dataset list (for --summarize)")
    ap.add_argument("--joint", default="joint20", help="the shared vocabulary's tag (finemo_<tag>/)")
    ap.add_argument("--cbp-tag", default="cbp10", help="ChromBPNet's production compendium (--production)")
    ap.add_argument("--cheri-tag", default="cheri39", help="Cherimoya's production compendium (--production)")
    ap.add_argument("--tolerance", type=int, default=5, help="bp between matched hit centres (default 5)")
    ap.add_argument("--out-dir", type=Path, help="default <comparison-root>/finemo")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    common.setup_logging(args.verbose)
    common.require(args, "comparison_root")
    out_root = args.out_dir or args.comparison_root / "finemo"
    try:
        if args.summarize:
            if args.datasets is None:
                ap.error("--summarize needs --datasets")
            return summarize(args, out_root)
        common.require(args, "cheri_root")
        return run_dataset(args, out_root)
    except common.Waiting as exc:
        common.result(f"waiting: {exc}")
        return common.WAITING
    except (common.Failed, ValueError, KeyError) as exc:
        logger.error("%s", exc)
        common.result(f"failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
