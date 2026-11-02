#!/usr/bin/env python3
"""The motif -> TF table coop_isa.py and combine.py read (--motifs), from a motif compendium.

Fi-NeMo, run against a compendium (igvf_tf_chrombpnet 10.0), names each hit's motif <posneg>_patterns.<cluster_id>,
the compendium's cluster. A cluster is named as the comparison names it (comparison/compare_compendia.py's read_meta
and cluster_table): the most frequent annotation_name0 among its patterns whose annotation_score0 reaches the
compendium's min_annotation_score (compendium.json; 0 when it has none). A cluster none of whose patterns reaches it
gets no annotation, and coop_isa.py and combine.py then use the motif's own name.

A project with ChromBPNet and Cherimoya took this table from the comparison (comparison/finemo/motifs.tsv, the joint
compendium's clusters); this script needs only the compendium the project's Fi-NeMo hits were called with.

    motif_names.py --compendium <compendia root>/<tag> --out <motifs.tsv>

Out: TSV with columns motif, cluster_annotation (empty when the cluster has none), posneg, cluster_id, patterns,
datasets (the labels with a pattern in the cluster), annotations (every passing name with its count).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "comparison"))
import compare_compendia as cc  # noqa: E402


def motif_table(compendium: Path):
    files = cc.compendium_files(compendium.parent, compendium.name)
    for key in ("json", "meta"):
        if not files[key].is_file():
            raise FileNotFoundError(f"no {files[key]}: is {compendium} a finished compendium (09.0)?")
    meta = cc.read_meta(files)
    table = cc.cluster_table(meta, cc.min_score(files))
    table["motif"] = table["posneg"] + "_patterns." + table["cluster_id"].astype(str)
    table["cluster_annotation"] = table["annotation"].fillna("")
    table = table.rename(columns={"labels": "datasets"})
    return table[["motif", "cluster_annotation", "posneg", "cluster_id", "patterns", "datasets", "annotations"]]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compendium", type=Path, required=True, help="the compendium directory (<compendia>/<tag>)")
    ap.add_argument("--out", type=Path, required=True, help="the motifs.tsv to write")
    args = ap.parse_args(argv)
    table = motif_table(args.compendium)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".tmp")
    table.to_csv(tmp, sep="\t", index=False)
    tmp.replace(args.out)
    named = int((table["cluster_annotation"] != "").sum())
    print(f"{args.out}: {len(table)} motifs ({named} annotated) from {args.compendium}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
