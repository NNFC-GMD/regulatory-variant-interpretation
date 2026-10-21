#!/usr/bin/env python3
"""Level 3 of the comparison: the two models' TF-MoDISco patterns and seqlets.

--dataset <ds> --part average  (a) the fold averages: both models' 08.0 results (the same tool, environment, flags
    and 500-bp window), each pattern annotated with tomtom-lite against --motif-db (MotifCompendium's database; the
    best match, its p- and q-value) and flagged as bias when its best match in --bias-db (chrombpnet's motifs.meme) is
    one of its Tn5 or DNase I entries at q <= 0.05. A pattern of one model is matched in the other
      by cluster   when the joint compendium (--joint, e.g. joint20: both models' patterns of the shared datasets)
                   puts a pattern of the other model's same dataset in its cluster (primary), and
      by tomtom    when its best tomtom-lite target among the other model's patterns has q <= 0.05, the
                   patterns searched together with --motif-db's motifs, which make the background and the
                   multiple-testing burden those of a whole database (secondary; reciprocal pairs are listed).
    Seqlets are placed in the genome through each model's own bed (fold 0's interpreted_regions.bed, the rows of the
    averaged h5): [start + summit - window/2 + seqlet start, ... + seqlet end). On the peaks both beds hold, each
    model's seqlets overlapped at >= 50 % (reciprocal) by any seqlet of the other model, and by the seqlets of their
    matched patterns, are counted. Writes <out>/<ds>/patterns.tsv (one row per pattern), pairs.tsv (matched pattern
    pairs) and average.json (pattern counts, shared fractions by count and weighted by seqlets, seqlet overlap, the
    unmatched patterns with their annotations). Without the joint compendium the cluster columns are empty; the
    unit runs again once it is there.
--dataset <ds> --part qc  (b) per fold at the QC scale: ChromBPNet's 04.5 results (evaluation/motif_qc_counts/
    chrombpnet_nobias_modisco_results.h5) against Cherimoya's on the same 30K-peak subsample (qc_modisco_folds.sh:
    cherimoya_modisco_results.h5, with its done marker): the same annotation and flags, matching by tomtom (no
    compendium), seqlet overlap on the shared peaks (window 400); and each model's fold-to-fold matching (every pair
    of its folds, patterns only: the subsamples differ by fold), the ceiling. Writes <out>/<ds>/qc_patterns.tsv,
    qc_pairs.tsv (one row per comparison) and qc.json.
--bias  (c) the bias control: for every <comparison-root>/bias_control/*/ whose status.json is done, the bias-included
    model's TF-MoDISco patterns (its motif_qc_<head>/chrombpnet_modisco_results.h5), and every --bias-modisco h5 given
    (e.g. a bias model's own 03.3 QC run): which of Cherimoya's unmatched patterns (fold averages of every dataset
    done, and the QC patterns of the control's own dataset and fold) match one of them (tomtom, decoys as above).
    A control that contribs_bw refused is reported with its reason; the database flags of (a) still mark bias.
    Writes <out>/bias_control.tsv and bias_control.json.
--summarize --datasets <list>  per dataset and over datasets: summary.tsv, statistics.tsv (bootstrap CIs and Wilcoxon
    tests over datasets: Cherimoya's against ChromBPNet's shared fractions; at the QC scale, cross-model matching
    against each model's fold-to-fold ceiling, also over folds), unmatched.tsv, and figures.

Waiting (exit 3, nothing written): a results h5 or bed not there yet (Cherimoya's 08.0 without downstream.sh's
.done/08.0, a QC fold without its done marker, ChromBPNet's 08.0 h5 younger than --settle seconds). Done markers in
<out>/.done/ record the inputs; a rerun with the same inputs does nothing (--force runs anyway).

    python workflows/comparison/compare_modisco.py --dataset <ds> --part average|qc [--joint joint20]
    python workflows/comparison/compare_modisco.py --bias --datasets <list.txt> [--bias-modisco <h5> ...]
    python workflows/comparison/compare_modisco.py --summarize --datasets <list.txt>

--motif-db defaults to the reference database a compendium under --compendia-root records (compendium.json ref_db);
--bias-db to the motif database ChromBPNet's QC of the dataset's first fold used (its motif_qc.json). Runs in the
pixi cherimoya environment (h5py, memelite, pandas, matplotlib), inside a box; tomtom-lite runs on NUMBA_NUM_THREADS.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1] / d) for d in ("comparison", "train", "attribution", "downstream")
]

import common  # noqa: E402
import motifs  # noqa: E402
import to_chrombpnet_scores as tcs  # noqa: E402

logger = logging.getLogger("compare_modisco")

FORMAT = 1
LABEL_PREFIX = {"chrombpnet": "cbp_", "cherimoya": "cheri_"}
"""make_downstream.py's compendium labels: cbp_<dataset>, cheri_<dataset>."""
QC_NAMES = {"chrombpnet": "chrombpnet_nobias", "cherimoya": "cherimoya"}
"""The --prefix of each model's QC run (04.5's for ChromBPNet, qc_modisco_folds.sh's for Cherimoya)."""


# ---------------------------------------------------------------------------------------------------------------------
# where things are


def layout(cheri_root: Path, ds: str) -> dict:
    """The dataset's paths for both models: the fold-average results and beds, and per fold the QC results and beds.
    ChromBPNet's are found through each Cherimoya fold's compare.json (make_jobs.py), as the converter finds them."""
    try:
        cfg = tcs.read_cheri_config(cheri_root / "configs" / ds / "config.yaml")
    except tcs.Refused as exc:
        raise common.Waiting(str(exc)) from exc
    out, pt, folds = Path(cfg["output_dir"]), cfg["peak_type"], [str(k) for k in cfg["folds"]]
    cbp_fold = {}
    for k in folds:
        try:
            cbp_fold[k] = tcs.chrombpnet_interpretation(cheri_root / ds / f"fold_{k}", common.HEAD)["bed"].parent.parent
        except tcs.Refused as exc:
            raise common.Waiting(str(exc)) from exc
    k0 = folds[0]
    cbp_out = cbp_fold[k0].parent.parent
    lay = {
        "folds": folds,
        "average": {
            "chrombpnet": {
                "h5": cbp_out / "contrib_scores" / ds / "modisco" / f"modisco_{common.HEAD}_results.h5",
                "bed": cbp_fold[k0] / "interpretation" / "interpretation.interpreted_regions.bed",
            },
            "cherimoya": {
                "h5": out / "contrib_scores" / ds / "modisco" / f"modisco_{common.HEAD}_results.h5",
                "bed": out
                / "full_models"
                / f"{ds}_{pt}_fold_{k0}"
                / "interpretation"
                / "interpretation.interpreted_regions.bed",
                "done": out / ".done" / "08.0",
            },
        },  # fmt: skip
        "qc": {},
    }
    for k in folds:
        cheri_fold = out / "full_models" / f"{ds}_{pt}_fold_{k}"
        qc_dir = f"motif_qc_{common.HEAD}"
        lay["qc"][k] = {
            "chrombpnet": {"h5": cbp_fold[k] / "evaluation" / qc_dir / f"{QC_NAMES['chrombpnet']}_modisco_results.h5",
                           "bed": cbp_fold[k] / "auxiliary" / "interpret_subsample" /
                           f"{QC_NAMES['chrombpnet']}.interpreted_regions.bed",
                           "summary": cbp_fold[k] / "evaluation" / qc_dir / f"{QC_NAMES['chrombpnet']}_motif_qc.json"},
            "cherimoya": {"h5": cheri_fold / "evaluation" / qc_dir / f"{QC_NAMES['cherimoya']}_modisco_results.h5",
                          "bed": cheri_fold / "auxiliary" / "interpret_subsample" /
                          f"{QC_NAMES['cherimoya']}.interpreted_regions.bed",
                          "done": cheri_fold / "evaluation" / qc_dir / "cherimoya_qc.done.json"},
        }  # fmt: skip
    return lay


def settled(path: Path, seconds: float) -> bool:
    try:
        return time.time() - path.stat().st_mtime >= seconds
    except FileNotFoundError:
        return False


def default_motif_db(compendia_root: Path | None, joint: str) -> Path | None:
    """The reference database 09.0 annotated a compendium with (compendium.json ref_db.filepath): the joint one's,
    else any's."""
    if compendia_root is None:
        return None
    candidates = [compendia_root / joint / "compendium.json", *sorted(compendia_root.glob("*/compendium.json"))]
    for path in candidates:
        try:
            ref = json.loads(path.read_text())["ref_db"]["filepath"]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if Path(ref).is_file():
            return Path(ref)
    return None


def default_bias_db(cheri_root: Path, datasets: list[str]) -> Path | None:
    """The motif database ChromBPNet's per-fold QC used (chrombpnet's motifs.meme, with its Tn5 and DNase entries),
    from the motif_qc.json of a dataset's QC."""
    for ds in datasets:
        try:
            lay = layout(cheri_root, ds)
        except (common.Waiting, common.Failed):
            continue
        for k in lay["folds"]:
            try:
                db = json.loads(lay["qc"][k]["chrombpnet"]["summary"].read_text())["motif_db"]
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if Path(db).is_file():
                return Path(db)
    return None


def joint_meta(compendia_root: Path | None, joint: str) -> Path | None:
    return compendia_root / joint / "modisco_compendium_meta.tsv" if compendia_root is not None else None


def read_clusters(meta: Path | None, ds: str) -> dict[tuple[str, str], str] | None:
    """{(model, "<posneg>.pattern_<i>"): cluster id} for the dataset's patterns in the joint compendium's
    _meta.tsv (names <label>-<posneg>.pattern_<i>, labels cbp_<ds> and cheri_<ds>); None without the file."""
    import pandas as pd

    if meta is None or not meta.is_file():
        return None
    df = pd.read_csv(meta, sep="\t", dtype=str, keep_default_na=False)
    out = {}
    for model, prefix in LABEL_PREFIX.items():
        label = f"{prefix}{ds}"
        part = df[df["model"] == label]
        for name, cid in zip(part["name"], part["cluster_id"]):
            key = name[len(label) + 1 :] if name.startswith(f"{label}-") else name
            out[(model, key)] = cid
    return out


# ---------------------------------------------------------------------------------------------------------------------
# matching


def annotate_patterns(pats: list[motifs.Pattern], annot_db: motifs.Database, bias_db: motifs.Database) -> list[dict]:
    ann = motifs.annotate(pats, annot_db) if pats else []
    flags = motifs.bias_flags(pats, bias_db) if pats else []
    rows = []
    for p, a, f in zip(pats, ann, flags):
        a0, b0 = p.trim_bounds()
        rows.append({
            "model": p.model, "posneg": p.posneg, "pattern": p.key, "n_seqlets": p.n_seqlets,
            "width_trimmed": b0 - a0, "consensus": p.consensus(),
            "annotation": a["matches"][0] if a["matches"] else None, "annotation_p": a["p"][0] if a["p"] else np.nan,
            "annotation_q": a["q"][0] if a["q"] else np.nan, "annotation_top3": ",".join(a["matches"]),
            "bias_db_best": f["db_best"], "bias_db_best_q": f["db_best_q"], "bias_match": f["bias_best"],
            "bias_q": f["bias_q"], "bias_flag": f["bias_flag"],
        })  # fmt: skip
    return rows


def peaks_intervals(pats: list[motifs.Pattern], bed, half: int, shared: set[str]) -> dict[str, dict]:
    """{pattern key: its seqlets' intervals on the shared peaks}."""
    on_shared = bed["name"].isin(shared).to_numpy()
    out = {}
    for p in pats:
        iv = motifs.seqlet_intervals(p, bed, half)
        keep = on_shared[p.seqlets["example_idx"].astype(np.int64)]
        out[p.key] = {k: v[keep] for k, v in iv.items()}
    return out


def as_bool(series) -> np.ndarray:
    """A True/False column read back from a TSV (missing values are False)."""
    return series.astype(str).str.strip().str.lower().isin(("true", "1")).to_numpy()


def concat_intervals(parts: list[dict]) -> dict:
    if not parts:
        return {"chrom": np.array([], dtype=object), "start": np.array([], np.int64), "end": np.array([], np.int64),
                "peak": np.array([], dtype=object)}  # fmt: skip
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def compare_sets(pats: dict, beds: dict, halves: dict, decoys: motifs.Database, clusters=None,
                 seqlet_overlap: bool = True) -> dict:  # fmt: skip
    """Both directions of the matching between the two models' pattern sets (keys of `pats`: the two models, or two
    folds of one model): per pattern its partners and, with beds, the share of its seqlets overlapped."""
    a_name, b_name = list(pats)
    a, b = pats[a_name], pats[b_name]
    cm = motifs.cross_match(a, b, decoys)
    per: dict[str, dict] = {a_name: {}, b_name: {}}
    for side, mine, theirs, hits in ((a_name, a, b, cm["a"]), (b_name, b, a, cm["b"])):
        for p, (j, pv, qv) in zip(mine, hits):
            per[side][p.key] = {"tomtom_partner": theirs[j].key if j is not None else None, "tomtom_p": pv,
                                "tomtom_q": qv, "cluster_partners": []}  # fmt: skip
    if clusters is not None:
        by_cluster: dict[str, dict[str, list[str]]] = {}
        for side, mine in ((a_name, a), (b_name, b)):
            for p in mine:
                cid = clusters.get((side, p.key))
                per[side][p.key]["cluster"] = cid
                if cid is not None:
                    by_cluster.setdefault(cid, {}).setdefault(side, []).append(p.key)
        for side, other in ((a_name, b_name), (b_name, a_name)):
            for entry in per[side].values():
                cid = entry.get("cluster")
                entry["cluster_partners"] = by_cluster.get(cid, {}).get(other, []) if cid is not None else []
    pairs = [{"a": a[i].key, "b": b[j].key, "p_ab": pa, "p_ba": pb} for i, j, pa, pb in cm["reciprocal"]]
    out = {"per": per, "pairs": pairs}
    if seqlet_overlap and beds is not None:
        shared = set(beds[a_name]["name"]) & set(beds[b_name]["name"])
        iv = {side: peaks_intervals(pats[side], beds[side], halves[side], shared) for side in (a_name, b_name)}
        whole = {side: concat_intervals(list(iv[side].values())) for side in iv}
        out["shared_peaks"] = len(shared)
        for side, other in ((a_name, b_name), (b_name, a_name)):
            hit_any = motifs.reciprocal_overlap(whole[side], whole[other])
            out.setdefault("overlap", {})[side] = {"seqlets": int(len(hit_any)), "overlapped": int(hit_any.sum())}
            for key, intervals in iv[side].items():
                entry = per[side][key]
                entry["seqlets_on_shared_peaks"] = int(len(intervals["start"]))
                any_hit = motifs.reciprocal_overlap(intervals, whole[other])
                entry["overlap_any"] = float(any_hit.mean()) if len(any_hit) else np.nan
                partners = set(entry["cluster_partners"]) | ({entry["tomtom_partner"]} - {None})
                if partners:
                    hit = motifs.reciprocal_overlap(intervals, concat_intervals([iv[other][k] for k in partners]))
                    entry["overlap_matched"] = float(hit.mean()) if len(hit) else np.nan
                else:
                    entry["overlap_matched"] = np.nan
    return out


def shared_fractions(pats: list[motifs.Pattern], per: dict, use: str) -> dict:
    """The share of a model's patterns, and of its seqlets, in patterns matched in the other set (`use`: cluster or
    tomtom)."""
    if not pats:
        return {"patterns": 0, "matched": 0, "by_count": np.nan, "by_seqlets": np.nan}
    matched = [is_matched(per[p.key], use) for p in pats]
    seq = np.array([p.n_seqlets for p in pats], dtype=np.float64)
    return {"patterns": len(pats), "matched": int(sum(matched)), "by_count": float(np.mean(matched)),
            "by_seqlets": float(seq[np.array(matched)].sum() / seq.sum()) if seq.sum() else np.nan}  # fmt: skip


def is_matched(entry: dict, use: str) -> bool:
    if use == "cluster":
        return bool(entry.get("cluster_partners"))
    return entry.get("tomtom_partner") is not None


# ---------------------------------------------------------------------------------------------------------------------
# units


def databases(args: argparse.Namespace, datasets: list[str]) -> tuple[motifs.Database, motifs.Database, dict]:
    annot = args.motif_db or default_motif_db(args.compendia_root, args.joint)
    bias = args.bias_db or default_bias_db(args.cheri_root, datasets)
    if annot is None or bias is None:
        raise common.Waiting(f"no {'--motif-db' if annot is None else '--bias-db'} given and none recorded yet "
                             "(a compendium's compendium.json, ChromBPNet's motif_qc.json)")  # fmt: skip
    return motifs.Database.read(annot), motifs.Database.read(bias), {"motif_db": annot, "bias_db": bias}


def run_average(args: argparse.Namespace, out_root: Path) -> int:
    import pandas as pd

    ds = args.dataset
    lay = layout(args.cheri_root, ds)
    av = lay["average"]
    missing = [f"{m} {what} {p}" for m in common.MODELS for what, p in av[m].items() if not Path(p).is_file()]
    if not missing and not settled(av["chrombpnet"]["h5"], args.settle):
        missing.append(f"ChromBPNet's {av['chrombpnet']['h5']} is younger than {args.settle:.0f} s")
    if missing:
        raise common.Waiting(f"{ds}: " + "; ".join(missing[:3]))
    meta = joint_meta(args.compendia_root, args.joint)
    annot_path = args.motif_db or default_motif_db(args.compendia_root, args.joint)
    bias_path = args.bias_db or default_bias_db(args.cheri_root, [ds])
    inputs = {f"{m}_{what}": p for m in common.MODELS for what, p in av[m].items()}
    inputs.update(joint_meta=meta, motif_db=annot_path, bias_db=bias_path)
    settings = {"joint": args.joint, "q": motifs.Q_MATCH}
    marker = common.Marker(out_root, f"{ds}.average", FORMAT, inputs, settings)
    if marker.current() is None and not args.force:
        common.result("up to date")
        return 0
    marker.remove()
    annot_db, bias_db, dbs = databases(args, [ds])
    pats, halves, beds = {}, {}, {}
    for m in common.MODELS:
        pats[m], window = motifs.read_modisco(av[m]["h5"], m)
        if window is None:
            raise common.Failed(f"{av[m]['h5']} records no window_size")
        halves[m] = window // 2
        beds[m] = common.read_bed(av[m]["bed"])
    clusters = read_clusters(meta, ds)
    res = compare_sets(pats, beds, halves, annot_db, clusters)
    rows = []
    for m in common.MODELS:
        for row in annotate_patterns(pats[m], annot_db, bias_db):
            e = res["per"][m][row["pattern"]]
            row.update(dataset=ds, joint_cluster=e.get("cluster"),
                       cluster_partners=",".join(e.get("cluster_partners", [])) or None,
                       tomtom_partner=e["tomtom_partner"], tomtom_p=e["tomtom_p"], tomtom_q=e["tomtom_q"],
                       matched_cluster=is_matched(e, "cluster") if clusters is not None else None,
                       matched_tomtom=is_matched(e, "tomtom"),
                       matched=is_matched(e, "cluster" if clusters is not None else "tomtom"),
                       seqlets_on_shared_peaks=e.get("seqlets_on_shared_peaks"),
                       seqlet_overlap_any=e.get("overlap_any"), seqlet_overlap_matched=e.get("overlap_matched"))  # fmt: skip
            rows.append(row)
    df = pd.DataFrame(rows)
    front = ["dataset", "model", "posneg", "pattern", "n_seqlets"]
    df = df[front + [c for c in df.columns if c not in front]]
    od = out_root / ds
    common.write_table(od / "patterns.tsv", df)
    common.write_table(od / "pairs.tsv", pd.DataFrame(res["pairs"], columns=["a", "b", "p_ab", "p_ba"]).rename(
        columns={"a": "chrombpnet", "b": "cherimoya", "p_ab": "p_chrombpnet_to_cherimoya",
                 "p_ba": "p_cherimoya_to_chrombpnet"}))  # fmt: skip
    use = "cluster" if clusters is not None else "tomtom"
    summary = {"dataset": ds, "joint": args.joint if clusters is not None else None, "matching": use,
               "windows": {m: 2 * halves[m] for m in common.MODELS}, "shared_peaks": res.get("shared_peaks"),
               "databases": {k: str(v) for k, v in dbs.items()}, "models": {}}  # fmt: skip
    for m in common.MODELS:
        part = df[df["model"] == m]
        summary["models"][m] = {
            "patterns": {pn: int((part["posneg"] == pn).sum()) for pn in ("pos", "neg")},
            "seqlets": int(part["n_seqlets"].sum()),
            "bias_flagged": int(part["bias_flag"].sum()),
            "shared": {u: shared_fractions(pats[m], res["per"][m], u)
                       for u in (("cluster", "tomtom") if clusters is not None else ("tomtom",))},
            "seqlet_overlap": res.get("overlap", {}).get(m),
            "unmatched": part[~part[f"matched_{use}"].astype(bool)][["pattern", "n_seqlets", "annotation",
                                                                     "annotation_q", "bias_flag"]
                                                                    ].to_dict("records"),
        }  # fmt: skip
    common.write_json(od / "average.json", summary)
    s = summary["models"]
    note = (f"{s['chrombpnet']['patterns']['pos'] + s['chrombpnet']['patterns']['neg']} ChromBPNet and "
            f"{s['cherimoya']['patterns']['pos'] + s['cherimoya']['patterns']['neg']} Cherimoya patterns; matched by "
            f"{use}: {s['chrombpnet']['shared'][use]['by_count']:.2f} / {s['cherimoya']['shared'][use]['by_count']:.2f}"
            f" by count; Cherimoya bias-flagged {s['cherimoya']['bias_flagged']}")  # fmt: skip
    marker.write({"patterns": od / "patterns.tsv", "pairs": od / "pairs.tsv", "average": od / "average.json"}, note)
    common.result(note)
    return 0


def run_qc(args: argparse.Namespace, out_root: Path) -> int:
    import pandas as pd

    ds = args.dataset
    lay = layout(args.cheri_root, ds)
    missing = []
    for k in lay["folds"]:
        for m in common.MODELS:
            for what in ("h5", "bed") + (("done",) if m == "cherimoya" else ("summary",)):
                if not lay["qc"][k][m][what].is_file():
                    missing.append(f"fold {k} {m} {what}")
    if missing:
        raise common.Waiting(
            f"{ds}: QC not there yet: " + ", ".join(missing[:4]) + (" ..." if len(missing) > 4 else "")
        )
    annot_path = args.motif_db or default_motif_db(args.compendia_root, args.joint)
    bias_path = args.bias_db or default_bias_db(args.cheri_root, [ds])
    inputs = {f"fold_{k}/{m}/{what}": p for k in lay["folds"] for m in common.MODELS
              for what, p in lay["qc"][k][m].items()}  # fmt: skip
    inputs.update(motif_db=annot_path, bias_db=bias_path)
    marker = common.Marker(out_root, f"{ds}.qc", FORMAT, inputs, {"q": motifs.Q_MATCH})
    if marker.current() is None and not args.force:
        common.result("up to date")
        return 0
    marker.remove()
    annot_db, bias_db, _ = databases(args, [ds])
    pats: dict[tuple[str, str], list] = {}
    halves, beds = {}, {}
    rows = []
    for k in lay["folds"]:
        for m in common.MODELS:
            pats[(m, k)], window = motifs.read_modisco(lay["qc"][k][m]["h5"], m)
            if window is None:
                raise common.Failed(f"{lay['qc'][k][m]['h5']} records no window_size")
            halves[(m, k)] = window // 2
            beds[(m, k)] = common.read_bed(lay["qc"][k][m]["bed"])
            for row in annotate_patterns(pats[(m, k)], annot_db, bias_db):
                row.update(dataset=ds, fold=k)
                rows.append(row)
    comparisons = []
    for k in lay["folds"]:
        res = compare_sets({"chrombpnet": pats[("chrombpnet", k)], "cherimoya": pats[("cherimoya", k)]},
                           {"chrombpnet": beds[("chrombpnet", k)], "cherimoya": beds[("cherimoya", k)]},
                           {"chrombpnet": halves[("chrombpnet", k)], "cherimoya": halves[("cherimoya", k)]}, annot_db)  # fmt: skip
        for row in rows:
            if row["fold"] == k:
                e = res["per"][row["model"]][row["pattern"]]
                row.update(cross_partner=e["tomtom_partner"], cross_q=e["tomtom_q"],
                           matched_cross=e["tomtom_partner"] is not None,
                           seqlet_overlap_any=e.get("overlap_any"), seqlet_overlap_matched=e.get("overlap_matched"))  # fmt: skip
        comparisons.append(comparison_row("cross", "chrombpnet", k, "cherimoya", k, pats, res))
    for m in common.MODELS:
        for k, k2 in combinations(lay["folds"], 2):
            res = compare_sets({f"{m}:{k}": pats[(m, k)], f"{m}:{k2}": pats[(m, k2)]}, None, None, annot_db,
                               seqlet_overlap=False)  # fmt: skip
            res["per"] = {m_: v for m_, v in zip(("a", "b"), res["per"].values())}
            comparisons.append(comparison_row(f"within_{m}", m, k, m, k2, pats, res, sides=("a", "b")))
    od = out_root / ds
    df = pd.DataFrame(rows)
    front = ["dataset", "fold", "model", "posneg", "pattern", "n_seqlets"]
    df = df[front + [c for c in df.columns if c not in front]]
    common.write_table(od / "qc_patterns.tsv", df)
    cmp_df = pd.DataFrame(comparisons)
    common.write_table(od / "qc_pairs.tsv", cmp_df)
    summary = qc_summary(cmp_df)
    summary["dataset"] = ds
    common.write_json(od / "qc.json", summary)
    note = (f"{len(lay['folds'])} folds: Cherimoya patterns matched in ChromBPNet {summary['cross_cherimoya']:.2f}, "
            f"ChromBPNet in Cherimoya {summary['cross_chrombpnet']:.2f}; fold-to-fold ceilings ChromBPNet "
            f"{summary['within_chrombpnet']:.2f}, Cherimoya {summary['within_cherimoya']:.2f}")  # fmt: skip
    marker.write({"qc_patterns": od / "qc_patterns.tsv", "qc_pairs": od / "qc_pairs.tsv", "qc": od / "qc.json"}, note)
    common.result(note)
    return 0


def comparison_row(kind: str, ma: str, ka: str, mb: str, kb: str, pats: dict, res: dict,
                   sides: tuple[str, str] | None = None) -> dict:  # fmt: skip
    sa, sb = sides or (ma, mb)
    a, b = pats[(ma, ka)], pats[(mb, kb)]
    fa, fb = shared_fractions(a, res["per"][sa], "tomtom"), shared_fractions(b, res["per"][sb], "tomtom")
    row = {"kind": kind, "model_a": ma, "fold_a": ka, "model_b": mb, "fold_b": kb, "patterns_a": fa["patterns"],
           "patterns_b": fb["patterns"], "matched_a": fa["by_count"], "matched_b": fb["by_count"],
           "matched_seqlets_a": fa["by_seqlets"], "matched_seqlets_b": fb["by_seqlets"],
           "reciprocal_pairs": len(res["pairs"])}  # fmt: skip
    if "overlap" in res:
        for side, tag in ((sa, "a"), (sb, "b")):
            o = res["overlap"][side]
            row[f"seqlet_overlap_{tag}"] = o["overlapped"] / o["seqlets"] if o["seqlets"] else np.nan
    return row


def qc_summary(df) -> dict:
    """Means over folds (cross) and fold pairs (within, both directions)."""
    cross = df[df["kind"] == "cross"]
    out = {"cross_chrombpnet": float(cross["matched_a"].mean()), "cross_cherimoya": float(cross["matched_b"].mean())}
    for m in common.MODELS:
        w = df[df["kind"] == f"within_{m}"]
        out[f"within_{m}"] = float(np.mean(np.r_[w["matched_a"], w["matched_b"]])) if len(w) else np.nan
    if "seqlet_overlap_a" in cross:
        out["seqlet_overlap_chrombpnet"] = float(cross["seqlet_overlap_a"].mean())
        out["seqlet_overlap_cherimoya"] = float(cross["seqlet_overlap_b"].mean())
    return out


def run_bias(args: argparse.Namespace, out_root: Path) -> int:
    import pandas as pd

    datasets = common.read_list(args.datasets)
    controls = sorted((args.comparison_root / "bias_control").glob("*/status.json")) if args.comparison_root else []
    references, notes = [], []
    for status_path in controls:
        try:
            status = json.loads(status_path.read_text())
        except (OSError, ValueError) as exc:
            notes.append(f"{status_path.parent.name}: unreadable status ({exc})")
            continue
        head = status.get("head", common.HEAD)
        h5 = status_path.parent / f"motif_qc_{head}" / "chrombpnet_modisco_results.h5"
        if status.get("state") == "done" and h5.is_file():
            references.append({"name": f"bias_control/{status_path.parent.name}", "h5": h5,
                               "dataset": status.get("dataset"), "fold": str(status.get("fold"))})  # fmt: skip
        else:
            notes.append(f"{status_path.parent.name}: {status.get('state')} ({status.get('reason', 'no MoDISco')})")
    for h5 in args.bias_modisco or []:
        references.append({"name": str(h5), "h5": Path(h5), "dataset": None, "fold": None})
    inputs = {f"status/{p.parent.name}": p for p in controls}
    inputs.update({f"reference/{r['name']}": r["h5"] for r in references})
    for ds in datasets:
        inputs[f"{ds}/average"] = out_root / ".done" / f"{ds}.average.json"
        inputs[f"{ds}/qc"] = out_root / ".done" / f"{ds}.qc.json"
    marker = common.Marker(out_root, "bias", FORMAT, inputs, {"q": motifs.Q_MATCH})
    if marker.current() is None and not args.force:
        common.result("up to date")
        return 0
    marker.remove()
    report = {"controls": [str(p.parent) for p in controls], "references": [r["name"] for r in references],
              "unavailable": notes}  # fmt: skip
    rows = []
    if references:
        annot_db, _, _ = databases(args, datasets)
        # Cherimoya's unmatched patterns: the fold averages of every dataset done, and the QC patterns of each
        # reference's own dataset and fold
        cheri: list[tuple[dict, motifs.Pattern]] = []
        for ds in datasets:
            table = out_root / ds / "patterns.tsv"
            if not table.is_file():
                continue
            df = pd.read_csv(table, sep="\t", keep_default_na=False, na_values=["NA", ""])
            unmatched = set(df[(df["model"] == "cherimoya").to_numpy() & ~as_bool(df["matched"])]["pattern"])
            pats, _ = motifs.read_modisco(layout(args.cheri_root, ds)["average"]["cherimoya"]["h5"], "cherimoya",
                                          seqlets=False)  # fmt: skip
            cheri += [({"dataset": ds, "scale": "fold average"}, p) for p in pats if p.key in unmatched]
        for ref in references:
            if ref["dataset"] is None:
                continue
            table = out_root / ref["dataset"] / "qc_patterns.tsv"
            if not table.is_file():
                continue
            df = pd.read_csv(table, sep="\t", dtype={"fold": str}, keep_default_na=False, na_values=["NA", ""])
            sel = df[
                ((df["model"] == "cherimoya") & (df["fold"] == ref["fold"])).to_numpy() & ~as_bool(df["matched_cross"])
            ]
            pats, _ = motifs.read_modisco(layout(args.cheri_root, ref["dataset"])["qc"][ref["fold"]]["cherimoya"]["h5"],
                                          "cherimoya", seqlets=False)  # fmt: skip
            cheri += [({"dataset": ref["dataset"], "scale": f"QC fold {ref['fold']}"}, p) for p in pats
                      if p.key in set(sel["pattern"])]  # fmt: skip
        for ref in references:
            ref_pats, _ = motifs.read_modisco(ref["h5"], "reference", seqlets=False)
            res = motifs.cross_match([p for _, p in cheri], ref_pats, annot_db)
            for (where, p), (j, pv, qv) in zip(cheri, res["a"]):
                rows.append({"reference": ref["name"], **where, "pattern": p.key, "n_seqlets": p.n_seqlets,
                             "consensus": p.consensus(), "reference_match": ref_pats[j].key if j is not None else None,
                             "p": pv, "q": qv, "matched": j is not None})  # fmt: skip
    df = pd.DataFrame(rows, columns=["reference", "dataset", "scale", "pattern", "n_seqlets", "consensus",
                                     "reference_match", "p", "q", "matched"])  # fmt: skip
    common.write_table(out_root / "bias_control.tsv", df)
    report["cherimoya_unmatched_checked"] = int(df[["dataset", "scale", "pattern"]].drop_duplicates().shape[0])
    report["matched_a_reference"] = int(df[df["matched"].astype(bool)][["dataset", "scale", "pattern"]]
                                        .drop_duplicates().shape[0]) if len(df) else 0  # fmt: skip
    common.write_json(out_root / "bias_control.json", report)
    if references:
        note = (f"{len(references)} reference(s): {report['matched_a_reference']} of "
                f"{report['cherimoya_unmatched_checked']} unmatched Cherimoya patterns match one")  # fmt: skip
    else:
        note = "no bias reference with TF-MoDISco: " + ("; ".join(notes) if notes else "no bias control run")
        note += "; the motif-database bias flags of the patterns stand alone"
    marker.write({"bias_control": out_root / "bias_control.tsv", "report": out_root / "bias_control.json"}, note)
    common.result(note)
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# across datasets


def summarize(args: argparse.Namespace, out_root: Path) -> int:
    import pandas as pd

    datasets = common.read_list(args.datasets)
    rows, unmatched, qc_rows, waiting = [], [], [], []
    inputs: dict[str, Path] = {"list": args.datasets}
    for ds in datasets:
        avg_marker, qc_marker = out_root / ".done" / f"{ds}.average.json", out_root / ".done" / f"{ds}.qc.json"
        inputs[f"{ds}/average"], inputs[f"{ds}/qc"] = avg_marker, qc_marker
        row: dict = {"dataset": ds}
        if avg_marker.is_file():
            av = json.loads((out_root / ds / "average.json").read_text())
            row["matching"] = av["matching"]
            for m in common.MODELS:
                s = av["models"][m]
                row[f"{m}_pos"], row[f"{m}_neg"] = s["patterns"]["pos"], s["patterns"]["neg"]
                row[f"{m}_seqlets"] = s["seqlets"]
                row[f"{m}_bias_flagged"] = s["bias_flagged"]
                for use, f in s["shared"].items():
                    row[f"{m}_shared_{use}_count"], row[f"{m}_shared_{use}_seqlets"] = f["by_count"], f["by_seqlets"]
                o = s.get("seqlet_overlap") or {}
                row[f"{m}_seqlet_overlap"] = o["overlapped"] / o["seqlets"] if o.get("seqlets") else np.nan
                for u in s["unmatched"]:
                    unmatched.append({"dataset": ds, "model": m, **u})
        else:
            waiting.append(f"{ds} (average)")
        if qc_marker.is_file():
            qc = json.loads((out_root / ds / "qc.json").read_text())
            row.update({f"qc_{k}": v for k, v in qc.items() if k != "dataset"})
            qc_rows.append(pd.read_csv(out_root / ds / "qc_pairs.tsv", sep="\t", dtype={"fold_a": str, "fold_b": str})
                           .assign(dataset=ds))  # fmt: skip
        else:
            waiting.append(f"{ds} (QC)")
        rows.append(row)
    if all(len(r) == 1 for r in rows):
        common.result(f"waiting: no dataset compared yet (of {len(datasets)})")
        return common.WAITING
    marker = common.Marker(out_root, "summary", FORMAT, inputs, {"n_boot": args.n_boot, "seed": args.seed})
    if marker.current() is None and not args.force and not waiting:
        common.result("up to date")
        return 0
    marker.remove()
    summary = pd.DataFrame(rows)
    common.write_table(out_root / "summary.tsv", summary)
    common.write_table(out_root / "unmatched.tsv", pd.DataFrame(unmatched))
    stats = []

    def paired(a: str, b: str, label: str) -> None:
        if a not in summary or b not in summary:
            return
        d = (summary[a] - summary[b]).dropna().to_numpy()
        est, lo, hi = common.bootstrap_ci(d, np.median, n_boot=args.n_boot, seed=args.seed)
        _, p, _ = common.wilcoxon(d)
        stats.append({"comparison": label, "over": "datasets", "n": int(d.size), "median_diff": est, "ci95_lo": lo,
                      "ci95_hi": hi, "wilcoxon_p": p, "first_higher": int((d > 0).sum())})  # fmt: skip

    for use in ("cluster", "tomtom"):
        for unit in ("count", "seqlets"):
            paired(f"cherimoya_shared_{use}_{unit}", f"chrombpnet_shared_{use}_{unit}",
                   f"shared by {use}, by {unit}: Cherimoya - ChromBPNet")  # fmt: skip
    for m in common.MODELS:
        paired(f"qc_cross_{m}", f"qc_within_{m}", f"QC: {m} patterns matched across models - within {m} (ceiling)")
    if qc_rows:
        qc_all = pd.concat(qc_rows, ignore_index=True)
        common.write_table(out_root / "qc_pairs.tsv", qc_all)
        for m, col in (("chrombpnet", "matched_a"), ("cherimoya", "matched_b")):
            cross = qc_all[qc_all["kind"] == "cross"].set_index(["dataset", "fold_a"])[col]
            w = qc_all[qc_all["kind"] == f"within_{m}"]
            per_fold = pd.concat([w[["dataset", "fold_a", "matched_a"]].rename(columns={"matched_a": "v"}),
                                  w[["dataset", "fold_b", "matched_b"]].rename(columns={"fold_b": "fold_a",
                                                                                         "matched_b": "v"})])  # fmt: skip
            within = per_fold.groupby(["dataset", "fold_a"])["v"].mean()
            d = (cross - within).dropna().to_numpy()
            est, lo, hi = common.bootstrap_ci(d, np.median, n_boot=args.n_boot, seed=args.seed)
            _, p, _ = common.wilcoxon(d)
            stats.append({"comparison": f"QC: {m} patterns matched across models - within {m}, per fold",
                          "over": "folds", "n": int(d.size), "median_diff": est, "ci95_lo": lo, "ci95_hi": hi,
                          "wilcoxon_p": p, "first_higher": int((d > 0).sum())})  # fmt: skip
    common.write_table(out_root / "statistics.tsv", pd.DataFrame(stats))
    outputs = {"summary": out_root / "summary.tsv", "unmatched": out_root / "unmatched.tsv",
               "statistics": out_root / "statistics.tsv"}  # fmt: skip
    for path in plot_summary(summary, out_root, datasets):
        outputs[path.name] = path
    done = sum(1 for r in rows if "matching" in r)
    note = f"{done} of {len(datasets)} datasets with fold averages, {len(qc_rows)} with QC"
    if "cherimoya_shared_tomtom_count" in summary:
        note += (f"; matched by tomtom (median over datasets) ChromBPNet "
                 f"{summary['chrombpnet_shared_tomtom_count'].median():.2f}, Cherimoya "
                 f"{summary['cherimoya_shared_tomtom_count'].median():.2f}")  # fmt: skip
    if waiting:
        common.result(f"waiting: {note}; not yet: {', '.join(waiting[:4])}{' ...' if len(waiting) > 4 else ''}")
        return common.WAITING
    marker.write(outputs, note)
    common.result(note)
    return 0


def plot_summary(summary, out_root: Path, datasets: list[str]) -> list[Path]:
    plt = common.matplotlib_pyplot()
    styles, labels = common.dataset_styles(datasets), common.short_names(datasets)
    panels = [(f"{{m}}_shared_{use}_{unit}", f"matched by {use}, share of {unit}")
              for use in ("cluster", "tomtom") for unit in ("count", "seqlets")]  # fmt: skip
    panels += [("qc_cross_{m}", "QC scale: matched across models"), ("qc_within_{m}", "QC scale: fold to fold")]
    panels = [(c, t) for c, t in panels if all(c.format(m=m) in summary for m in common.MODELS)]
    if not panels:
        return []
    fig, axes = plt.subplots(1, len(panels), figsize=(2.8 * len(panels), 3.4), squeeze=False)
    for ax, (col, title) in zip(axes[0], panels):
        for x, m in enumerate(common.MODELS):
            values = summary[["dataset", col.format(m=m)]].dropna()
            for ds, v in zip(values["dataset"], values[col.format(m=m)]):
                colour, marker = styles.get(ds, ("#999999", "o"))
                ax.scatter(x + (sorted(datasets).index(ds) - len(datasets) / 2) * 0.025, v, s=16, color=colour,
                           marker=marker, edgecolor="#333333", linewidth=0.3, zorder=2)  # fmt: skip
            if len(values):
                med = float(values[col.format(m=m)].median())
                ax.plot([x - 0.3, x + 0.3], [med, med], color="#333333", lw=1.2, zorder=3)
        ax.set_xticks(range(len(common.MODELS)))
        ax.set_xticklabels([f"{common.NAMES[m]}'s\npatterns" for m in common.MODELS], fontsize=7)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(title, fontsize=7.5)
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("share")
    common.dataset_legend(fig, styles, labels, f"{len(datasets)} datasets")
    fig.suptitle("TF-MoDISco patterns found by both models", fontsize=9)
    fig.tight_layout()
    paths = common.save_figure(fig, out_root / "shared_patterns")
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_root_args(ap)
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--dataset", help="compare this dataset (with --part)")
    what.add_argument("--bias", action="store_true", help="the bias control (c)")
    what.add_argument("--summarize", action="store_true", help="summarize the datasets of --datasets")
    ap.add_argument("--part", choices=("average", "qc"), default="average", help="with --dataset: (a) or (b)")
    ap.add_argument("--datasets", type=Path, help="the dataset list (for --bias and --summarize)")
    ap.add_argument("--joint", default="joint20", help="the joint compendium's tag under --compendia-root")
    ap.add_argument("--motif-db", type=Path, help="MEME database to annotate with (default: see above)")
    ap.add_argument("--bias-db", type=Path, help="MEME database with TN5_/DNASE_ bias entries (default: see above)")
    ap.add_argument("--bias-modisco", type=Path, action="append", help="a bias reference's TF-MoDISco h5")
    ap.add_argument("--settle", type=float, default=600.0, help="seconds ChromBPNet's 08.0 h5 must be old")
    ap.add_argument("--out-dir", type=Path, help="default <comparison-root>/modisco")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    common.setup_logging(args.verbose)
    if args.out_dir is None:
        common.require(args, "comparison_root")
    out_root = args.out_dir or args.comparison_root / "modisco"
    try:
        if args.summarize or args.bias:
            if args.datasets is None:
                ap.error("--summarize and --bias need --datasets")
            if args.bias:
                common.require(args, "cheri_root")
                return run_bias(args, out_root)
            return summarize(args, out_root)
        common.require(args, "cheri_root")
        return run_average(args, out_root) if args.part == "average" else run_qc(args, out_root)
    except common.Waiting as exc:
        common.result(f"waiting: {exc}")
        return common.WAITING
    except (common.Failed, tcs.Refused, ValueError) as exc:
        logger.error("%s", exc)
        common.result(f"failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
