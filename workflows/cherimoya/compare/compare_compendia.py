#!/usr/bin/env python3
"""Level 4 of the comparison: the motif compendia (09.0) of the two models.

Each compendium is a directory under --compendia-root (09.0's outputs: modisco_compiled.h5, one CWM per cluster;
modisco_compendium_meta.tsv, one row per TF-MoDISco pattern with its label, seqlets, annotation and cluster; and
compendium.json). Writes, under --out-dir (default <comparison-root>/compendium):

  clusters_<tag>.tsv   per cluster: pos/neg, patterns, labels (datasets), seqlets, its annotation (the most frequent
                       annotation_name0 of its patterns whose annotation_score0 reaches the compendium's
                       min_annotation_score) and TF family (the annotation without its _<n> suffix)
  like_for_like.json, matches_<cbp>_vs_<cheri>.tsv
                       ChromBPNet's compendium of the shared datasets (--cbp-tag, cbp10) against Cherimoya's of the
                       same datasets (--cheri-tag, cheri10): reciprocal best matches between their cluster CWMs
                       (tomtom-lite on the CWMs trimmed as modisco-lite trims them and scaled to unit L2 norm, so the
                       two models' attribution scales do not count; q-values Benjamini-Hochberg over the other
                       compendium's clusters; a match needs q <= 0.05 both ways), and the overlap of the annotations
                       and of the TF families (shared, one model only, Jaccard)
  joint_clusters.tsv, joint_summary.tsv
                       the joint compendium (--joint, joint20: both models' patterns of the shared datasets): each
                       cluster's patterns, datasets and seqlets from each model (the model and num_seqlets columns of
                       _meta.tsv) and its class, shared (patterns of both models), ChromBPNet-only or Cherimoya-only,
                       with its annotation and its bias share: of its patterns that level 3 compared (its
                       patterns.tsv), the share flagged as Tn5/DNase bias; per class, the clusters, patterns and
                       seqlets of each model
  supplement.json, matches_<cheri>_vs_<all>.tsv
                       when --supplement-tag (cheri39, all of Cherimoya's datasets) is built: its clusters matched in
                       the shared datasets' compendium, and what the other datasets add; matches_<cbp>_vs_<all>.tsv
                       pairs ChromBPNet's clusters with it too (level 5 compares production hits through it)
  sensitivity.json     when --sensitivity-tag (the joint compendium rebuilt at threshold 0.90 under its own tag) is
                       built: its classes beside the joint compendium's
  compendia.pdf/.png   the joint clusters by class, and their seqlets by model

Waiting (exit 3, nothing written) while the like-for-like or the joint compendium is not built (no compendium.json).
A done marker (<out-dir>/.done/compendia.json) records the inputs; a rerun with the same inputs does nothing
(--force runs anyway).

    python workflows/cherimoya/compare/compare_compendia.py --datasets <list.txt> [--joint joint20]
        [--cbp-tag cbp10 --cheri-tag cheri10 --supplement-tag cheri39 --sensitivity-tag <tag>] [--out-dir <dir>]

Runs in the pixi cherimoya environment (h5py, memelite, pandas, matplotlib), inside a box.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parent.parent)]

import common  # noqa: E402
import motifs  # noqa: E402

logger = logging.getLogger("compare_compendia")

FORMAT = 1
LABEL_MODEL = (("cbp_", "chrombpnet"), ("cheri_", "cherimoya"))
CLASSES = ("shared", "chrombpnet_only", "cherimoya_only")
CLASS_COLOURS = {"shared": "#009E73", "chrombpnet_only": "#0072B2", "cherimoya_only": "#E69F00"}


def model_of(label: str) -> str | None:
    for prefix, model in LABEL_MODEL:
        if label.startswith(prefix):
            return model
    return None


def dataset_of(label: str) -> str:
    for prefix, _ in LABEL_MODEL:
        if label.startswith(prefix):
            return label[len(prefix) :]
    return label


def family(name: str | None) -> str | None:
    return re.sub(r"_\d+$", "", name) if isinstance(name, str) and name else None


def compendium_files(root: Path, tag: str) -> dict[str, Path]:
    d = root / tag
    return {"json": d / "compendium.json", "meta": d / "modisco_compendium_meta.tsv",
            "compiled": d / "modisco_compiled.h5"}  # fmt: skip


def read_meta(files: dict[str, Path]):
    import pandas as pd

    meta = pd.read_csv(files["meta"], sep="\t", keep_default_na=False, na_values=[""])
    meta["cluster_id"] = meta["cluster_id"].astype(str)
    meta["model_kind"] = meta["model"].map(model_of)
    meta["dataset"] = meta["model"].map(dataset_of)
    return meta


def min_score(files: dict[str, Path]) -> float:
    try:
        return float(json.loads(files["json"].read_text())["settings"]["min_annotation_score"])
    except (OSError, ValueError, KeyError, TypeError):
        return 0.0


def cluster_table(meta, threshold: float):
    """Per cluster: pos/neg (majority), patterns, labels, seqlets, annotation (most frequent annotation_name0 at or
    above the threshold) and family."""
    import pandas as pd

    rows = []
    for cid, part in meta.groupby("cluster_id", sort=False):
        good = part[pd.to_numeric(part["annotation_score0"], errors="coerce").fillna(0) >= threshold]
        names = Counter(good["annotation_name0"].dropna())
        top = sorted(names.items(), key=lambda kv: (-kv[1], kv[0]))
        annotation = top[0][0] if top else None
        row = {"cluster_id": cid, "posneg": Counter(part["posneg"]).most_common(1)[0][0], "patterns": len(part),
               "labels": part["model"].nunique(), "seqlets": int(part["num_seqlets"].sum()),
               "annotation": annotation, "family": family(annotation),
               "annotations": ",".join(f"{n}:{c}" for n, c in top)}  # fmt: skip
        for model in common.MODELS:
            mine = part[part["model_kind"] == model]
            row[f"patterns_{model}"] = len(mine)
            row[f"datasets_{model}"] = int(mine["dataset"].nunique())
            row[f"seqlets_{model}"] = int(mine["num_seqlets"].sum())
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.sort_values("cluster_id", key=lambda s: s.map(lambda c: int(c) if c.isdigit() else 1 << 30))


def normalized_cwms(compiled: Path) -> dict[str, np.ndarray]:
    """{cluster id: (4, w)}: each cluster's CWM trimmed as modisco-lite trims a pattern (positions whose summed |CWM|
    reaches 0.3 of the maximum) and scaled to unit L2 norm."""
    out = {}
    for _, cid, cwm in motifs.read_compiled(compiled):
        score = np.abs(cwm).sum(axis=1)
        keep = np.flatnonzero(score >= motifs.TRIM_THRESHOLD * score.max()) if score.max() > 0 else np.arange(len(cwm))
        core = cwm[keep.min() : keep.max() + 1]
        norm = np.linalg.norm(core)
        out[str(cid)] = np.ascontiguousarray((core / norm if norm > 0 else core).T)
    return out


def reciprocal_matches(a: dict[str, np.ndarray], b: dict[str, np.ndarray], q_max: float = motifs.Q_MATCH):
    """Best match both ways between two sets of normalized CWMs; q over the other set's clusters."""
    import pandas as pd

    ka, kb = list(a), list(b)
    pab = motifs.tomtom_p([a[k] for k in ka], [b[k] for k in kb])
    pba = motifs.tomtom_p([b[k] for k in kb], [a[k] for k in ka])
    qab, qba = motifs.bh(pab), motifs.bh(pba)
    best_ab = np.argmin(pab, axis=1) if len(kb) else np.array([], int)
    best_ba = np.argmin(pba, axis=1) if len(ka) else np.array([], int)
    rows = []
    for i, k in enumerate(ka):
        if not len(kb):
            rows.append({"cluster": k, "best": None, "p": np.nan, "q": np.nan, "match": False, "reciprocal": False})
            continue
        j = int(best_ab[i])
        ok = bool(qab[i, j] <= q_max)
        recip = ok and int(best_ba[j]) == i and bool(qba[j, i] <= q_max)
        rows.append({"cluster": k, "best": kb[j], "p": float(pab[i, j]), "q": float(qab[i, j]), "match": ok,
                     "reciprocal": recip})  # fmt: skip
    back = {kb[j]: (ka[int(best_ba[j])], bool(qba[j, int(best_ba[j])] <= q_max)) for j in range(len(kb)) if len(ka)}
    return pd.DataFrame(rows), back


def overlap(a: set, b: set) -> dict:
    a, b = {x for x in a if x}, {x for x in b if x}
    union = a | b
    return {"shared": sorted(a & b), "only_a": sorted(a - b), "only_b": sorted(b - a),
            "jaccard": len(a & b) / len(union) if union else np.nan}  # fmt: skip


def bias_shares(meta, level3: Path) -> dict[str, float]:
    """{cluster id: share of its patterns flagged as bias by level 3}, from <level3>/<dataset>/patterns.tsv."""
    import pandas as pd

    flags = {}
    for ds in meta["dataset"].unique():
        table = level3 / ds / "patterns.tsv"
        if not table.is_file():
            continue
        df = pd.read_csv(table, sep="\t", keep_default_na=False, na_values=["NA", ""])
        for model, pattern, flag in zip(df["model"], df["pattern"], df["bias_flag"]):
            flags[(model, ds, pattern)] = str(flag).strip().lower() in ("true", "1")
    out = {}
    for cid, part in meta.groupby("cluster_id"):
        seen = []
        for label, name, model in zip(part["model"], part["name"], part["model_kind"]):
            key = (model, dataset_of(label), name[len(label) + 1 :] if name.startswith(f"{label}-") else name)
            if key in flags:
                seen.append(flags[key])
        out[cid] = float(np.mean(seen)) if seen else np.nan
    return out


def classify(row) -> str:
    if row["patterns_chrombpnet"] and row["patterns_cherimoya"]:
        return "shared"
    return "chrombpnet_only" if row["patterns_chrombpnet"] else "cherimoya_only"


def class_summary(clusters) -> list[dict]:
    rows = []
    for cls in CLASSES:
        part = clusters[clusters["class"] == cls]
        rows.append({"class": cls, "clusters": len(part),
                     **{f"{what}_{m}": int(part[f"{what}_{m}"].sum()) for what in ("patterns", "seqlets")
                        for m in common.MODELS},
                     "annotated": int(part["annotation"].notna().sum()),
                     "bias_clusters": int((part["bias_share"] >= 0.5).sum())})  # fmt: skip
    return rows


def plot(summary, out: Path) -> list[Path]:
    plt = common.matplotlib_pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2), squeeze=False)
    ax, ax2 = axes[0]
    labels = {"shared": "both models", "chrombpnet_only": "ChromBPNet only", "cherimoya_only": "Cherimoya only"}
    x = np.arange(len(CLASSES))
    ax.bar(x, [int(summary.loc[c, "clusters"]) for c in CLASSES], color=[CLASS_COLOURS[c] for c in CLASSES],
           width=0.6)  # fmt: skip
    for i, c in enumerate(CLASSES):
        ax.annotate(str(int(summary.loc[c, "clusters"])), (i, summary.loc[c, "clusters"]), ha="center", va="bottom",
                    fontsize=7)  # fmt: skip
    ax.set_xticks(x)
    ax.set_xticklabels([labels[c] for c in CLASSES])
    ax.set_ylabel("clusters")
    ax.set_title("joint compendium clusters by the models whose patterns they hold", fontsize=8)
    ax.grid(axis="x", visible=False)
    from matplotlib.patches import Patch

    width = 0.38
    for i, m in enumerate(common.MODELS):
        values = [int(summary.loc[c, f"seqlets_{m}"]) for c in CLASSES]
        ax2.bar(x + (i - 0.5) * width, values, width=width - 0.04, color=[CLASS_COLOURS[c] for c in CLASSES],
                hatch=None if m == "chrombpnet" else "///", edgecolor="white", linewidth=0.5)  # fmt: skip
    ax2.legend(handles=[Patch(facecolor="#999999", label="ChromBPNet"),
                        Patch(facecolor="#999999", hatch="///", edgecolor="white", label="Cherimoya")],
               fontsize=7, frameon=False)  # fmt: skip
    ax2.set_xticks(x)
    ax2.set_xticklabels([labels[c] for c in CLASSES])
    ax2.set_ylabel("seqlets")
    ax2.set_title("their seqlets, by model (hatched: Cherimoya)", fontsize=8)
    ax2.grid(axis="x", visible=False)
    fig.tight_layout()
    paths = common.save_figure(fig, out)
    plt.close(fig)
    return paths


def run(args: argparse.Namespace) -> int:
    import pandas as pd

    root, out = args.compendia_root, args.out_dir or args.comparison_root / "compendium"
    tags = {"cbp": args.cbp_tag, "cheri": args.cheri_tag, "joint": args.joint}
    files = {k: compendium_files(root, t) for k, t in tags.items()}
    missing = [f"{tags[k]} ({f['json']})" for k, f in files.items() if not f["json"].is_file()]
    if missing:
        raise common.Waiting("not built yet: " + ", ".join(missing))
    optional = {"supplement": args.supplement_tag, "sensitivity": args.sensitivity_tag or f"{args.joint}_t090"}
    for k, t in optional.items():
        files[k] = compendium_files(root, t)
    inputs = {f"{k}/{role}": p for k, f in files.items() for role, p in f.items()}
    datasets = common.read_list(args.datasets)
    level3 = args.level3_dir or (args.comparison_root / "modisco" if args.comparison_root else None)
    for ds in datasets:
        if level3 is not None:
            inputs[f"level3/{ds}"] = level3 / ds / "patterns.tsv"
    marker = common.Marker(out, "compendia", FORMAT, inputs, {"tags": tags, **optional, "q": motifs.Q_MATCH})
    if marker.current() is None and not args.force:
        common.result("up to date")
        return 0
    marker.remove()
    outputs: dict[str, Path] = {}

    # clusters of each compendium
    metas, clusters = {}, {}
    for k, f in files.items():
        if not f["json"].is_file():
            continue
        metas[k] = read_meta(f)
        clusters[k] = cluster_table(metas[k], min_score(f))
        outputs[f"clusters_{k}"] = common.write_table(out / f"clusters_{tags.get(k, optional.get(k))}.tsv", clusters[k])

    # like for like
    cwm = {k: normalized_cwms(files[k]["compiled"]) for k in ("cbp", "cheri", "supplement")
           if k in metas and files[k]["compiled"].is_file()}  # fmt: skip
    m_ab, _ = reciprocal_matches(cwm["cbp"], cwm["cheri"])
    m_ba, _ = reciprocal_matches(cwm["cheri"], cwm["cbp"])
    for df, k in ((m_ab, "cbp"), (m_ba, "cheri")):
        ann = clusters[k].set_index("cluster_id")[["annotation", "posneg"]]
        df["annotation"] = df["cluster"].map(ann["annotation"])
        df["posneg"] = df["cluster"].map(ann["posneg"])
    pair_name = f"matches_{tags['cbp']}_vs_{tags['cheri']}"
    outputs[pair_name] = common.write_table(out / f"{pair_name}.tsv", m_ab)
    outputs[f"{pair_name}_back"] = common.write_table(out / f"matches_{tags['cheri']}_vs_{tags['cbp']}.tsv", m_ba)
    ann = {k: set(clusters[k]["annotation"].dropna()) for k in ("cbp", "cheri")}
    fam = {k: {family(a) for a in ann[k]} for k in ann}
    lfl = {
        "compendia": {k: tags[k] for k in ("cbp", "cheri")},
        "clusters": {k: int(len(clusters[k])) for k in ("cbp", "cheri")},
        "matched": {"chrombpnet_clusters_with_a_match": int(m_ab["match"].sum()),
                    "cherimoya_clusters_with_a_match": int(m_ba["match"].sum()),
                    "reciprocal_pairs": int(m_ab["reciprocal"].sum())},
        "annotations": overlap(ann["cbp"], ann["cheri"]),
        "families": overlap(fam["cbp"], fam["cheri"]),
    }  # fmt: skip
    outputs["like_for_like"] = common.write_json(out / "like_for_like.json", lfl)

    # the joint compendium
    joint = clusters["joint"].copy()
    joint["class"] = joint.apply(classify, axis=1)
    shares = bias_shares(metas["joint"], level3) if level3 is not None else {}
    joint["bias_share"] = joint["cluster_id"].map(shares).astype(float)
    outputs["joint_clusters"] = common.write_table(out / "joint_clusters.tsv", joint)
    summary = pd.DataFrame(class_summary(joint)).set_index("class")
    outputs["joint_summary"] = common.write_table(out / "joint_summary.tsv", summary.reset_index())
    for path in plot(summary, out / "compendia"):
        outputs[path.name] = path

    # the supplement: Cherimoya's compendium of every dataset against the shared datasets' one
    note_extra = ""
    if "supplement" in metas and "supplement" in cwm:
        sup, _ = reciprocal_matches(cwm["supplement"], cwm["cheri"])
        outputs["supplement_matches"] = common.write_table(
            out / f"matches_{optional['supplement']}_vs_{tags['cheri']}.tsv", sup)  # fmt: skip
        cbp_sup, _ = reciprocal_matches(cwm["cbp"], cwm["supplement"])
        outputs["cbp_vs_supplement"] = common.write_table(
            out / f"matches_{tags['cbp']}_vs_{optional['supplement']}.tsv", cbp_sup)  # fmt: skip
        sup_ann = set(clusters["supplement"]["annotation"].dropna())
        added = sup[~sup["match"]]
        report = {"compendium": optional["supplement"], "clusters": int(len(clusters["supplement"])),
                  "matched_in_shared": int(sup["match"].sum()), "not_in_shared": int(len(added)),
                  "annotations_added": sorted(sup_ann - ann["cheri"])}  # fmt: skip
        outputs["supplement"] = common.write_json(out / "supplement.json", report)
        note_extra += f"; {optional['supplement']}: {report['not_in_shared']} clusters not in {tags['cheri']}"
    if "sensitivity" in metas:
        sens = clusters["sensitivity"].copy()
        sens["class"] = sens.apply(classify, axis=1)
        report = {"compendium": optional["sensitivity"],
                  "classes": {c: int((sens["class"] == c).sum()) for c in CLASSES},
                  "joint_classes": {c: int((joint["class"] == c).sum()) for c in CLASSES}}  # fmt: skip
        outputs["sensitivity"] = common.write_json(out / "sensitivity.json", report)
        note_extra += f"; at {optional['sensitivity']}: " + "/".join(str(v) for v in report["classes"].values())

    n = {c: int(summary.loc[c, "clusters"]) for c in CLASSES}
    note = (f"{tags['cbp']} {lfl['clusters']['cbp']} vs {tags['cheri']} {lfl['clusters']['cheri']} clusters, "
            f"{lfl['matched']['reciprocal_pairs']} reciprocal pairs, annotation Jaccard "
            f"{lfl['annotations']['jaccard']:.2f}; {tags['joint']}: {n['shared']} shared, {n['chrombpnet_only']} "
            f"ChromBPNet-only, {n['cherimoya_only']} Cherimoya-only clusters{note_extra}")  # fmt: skip
    marker.write(outputs, note)
    common.result(note)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_root_args(ap)
    ap.add_argument("--datasets", type=Path, required=True, help="the shared datasets, one per line")
    ap.add_argument("--joint", default="joint20")
    ap.add_argument("--cbp-tag", default="cbp10")
    ap.add_argument("--cheri-tag", default="cheri10")
    ap.add_argument("--supplement-tag", default="cheri39")
    ap.add_argument("--sensitivity-tag", help="the joint compendium at threshold 0.90 (default <joint>_t090)")
    ap.add_argument("--level3-dir", type=Path, help="level 3's outputs (default <comparison-root>/modisco)")
    ap.add_argument("--motif-db", type=Path, help="accepted for compare.sh's sake; the compendia carry annotations")
    ap.add_argument("--bias-db", type=Path, help="accepted for compare.sh's sake; bias comes from level 3's flags")
    ap.add_argument("--out-dir", type=Path, help="default <comparison-root>/compendium")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    common.setup_logging(args.verbose)
    common.require(args, "compendia_root")
    if args.out_dir is None:
        common.require(args, "comparison_root")
    try:
        return run(args)
    except common.Waiting as exc:
        common.result(f"waiting: {exc}")
        return common.WAITING
    except (common.Failed, ValueError, KeyError) as exc:
        logger.error("%s", exc)
        common.result(f"failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
