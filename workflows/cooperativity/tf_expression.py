#!/usr/bin/env python3
"""TF expression per cell group beside DeepISA's cooperativity: is each TF family expressed in each group, and does
its expression change between the two groups of a comparison? A change in a TF's cooperativity between two cell
states can follow from the TF being expressed in one and not the other, rather than from a change in how it works
with its partners; these columns let the two be read apart.

    tf_expression.py expression --h5ad <counts.h5ad> --cells <cells.csv> --group-column <col> --out <dir>
                                [--id-column barcode] [--gene-column gene_symbols] [--chunk 20000]
    tf_expression.py annotate --expression <dir> --motif-metadata <metadata.tsv> --coop <root>/cherimoya/cooperativity
                              [--prefix <dataset prefix>] [--min-frac 0.05] [--min-fold 2]
    tf_expression.py fetch-metadata --reference-root <dir>      (setup.sh --references runs it; login node)

expression: raw counts (an AnnData h5ad, X a CSR matrix of cells x genes, read in --chunk rows at a time with h5py)
  summed per group of the --cells table (an ID column matching the h5ad's obs index, and a group column; comma-
  separated if .csv, else tab). Each group is named by its stem, as workflows/pipeline/pseudobulk.py names a
  pseudobulk, so it is the cooperativity group of that pseudobulk's sample. Writes <out>/genes_by_group.tsv.gz
  (group, gene, gene_id, counts, cpm, frac: the fraction of the group's cells with a count), <out>/groups.tsv (group,
  label, cells, cells_found, counts) and <out>/expression.json (inputs and counts).

annotate: TF families from MotifCompendium's metadata table (MotifCompendium-<version>-Database-Human.metadata.tsv,
  beside the MEME database in kundajelab/MotifCompendium's pipeline/data/; columns name and TF). A family is a motif
  name without its _<n> suffix, as combine.py names a site's TF; its genes are every TF of every motif of the family.
  Genes are matched case- and punctuation-blind (NKX61 = NKX6-1). In a group, the family's cpm and frac are its best
  member's (most counts per million; the largest fraction), top_gene that member, and expressed = frac >= --min-frac.
  Writes, under <coop>/expression/: tf_families.tsv (family, members, n_members, n_found), tf_expression.tsv (family,
  group, top_gene, cpm, frac, n_expressed, expressed), and a copy of each table of <coop>/combined/ and
  <coop>/official/ with these columns added, as <coop>/expression/<combined|official>/<table>.tsv:
    a comparison's TF (group_a, group_b, tf): top_gene, cpm, frac, expressed for each side (_a, _b); log2fc_expr =
      log2((cpm_b + 1) / (cpm_a + 1)); expression_changes = expressed on one side only, or |log2fc_expr| >=
      log2(--min-fold). The best member can be a broadly expressed relative that hides a lineage factor (TRPS1 for
      GATA1), so also members_up and members_down: the family's genes whose CPM changes at least --min-fold-fold
      between the sides and that are expressed (frac >= --min-frac) on the higher side, as GENE(xFOLD), and
      members_changed, how many;
    a comparison's TF pair (pair "A + B" or tf_pair "A|B"): each TF's expressed_a, expressed_b and log2fc_expr
      (tf1_, tf2_), and expression_changes if either TF's does;
    a dataset's TF or TF pair (dataset, tf, pair or tf_pair): its group = dataset without --prefix; the TF's, or each
      TF's, top_gene, cpm, frac, expressed, and both_expressed for a pair;
    a series' TF (trend tables, columns n_<group>): expressed_<group> and cpm_<group> for each group.
  A TF that is a motif's own name (no annotation) has no family: its columns are empty.

fetch-metadata: MotifCompendium's metadata table, from the commit and beside the MEME database the vendored
  download-references fetches (<dir>/motif/MotifCompendium-<version>-Database-Human.metadata.tsv), md5-checked.
  Standard library only.

expression and annotate need numpy, pandas and h5py (the cherimoya env): run them in a box, not on the login node.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pipeline"))

from pseudobulk import stem_of  # noqa: E402  # the group names pseudobulk.py gives the samples

PAIR_SEPARATORS = {"pair": " + ", "tf_pair": "|"}  # combine.py's and the official deepISA package's
MOTIF_METADATA_MD5 = "19ad4d572f3e8bc8f07f3c54367f87cc"  # MotifCompendium 1.0.19 (7e9d1c2), as the MEME database


def norm(gene: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(gene).upper())


def family_of(name: str) -> str:
    return re.sub(r"_\d+$", "", str(name))


# ---------------------------------------------------------------------------------------------------------------------
# expression


def _decode(x) -> str:
    return x.decode() if isinstance(x, bytes) else str(x)


def read_strings(node) -> list[str]:
    """An AnnData column: a string dataset, or a categorical group (categories, codes)."""
    import h5py

    if isinstance(node, h5py.Group):
        cats = [_decode(x) for x in node["categories"][:]]
        return [cats[c] if c >= 0 else "" for c in node["codes"][:]]
    return [_decode(x) for x in node[:]]


def index_of(frame) -> list[str]:
    return read_strings(frame[_decode(frame.attrs.get("_index", "_index"))])


def expression(args) -> dict:
    import h5py
    import numpy as np
    import pandas as pd

    t0 = time.time()
    sep = "," if args.cells.name.endswith(".csv") else "\t"
    cells = pd.read_csv(args.cells, sep=sep, usecols=[args.id_column, args.group_column], dtype=str)
    if cells[args.id_column].duplicated().any():
        raise SystemExit(f"{args.cells}: duplicate IDs in {args.id_column}")
    sizes = cells[args.group_column].value_counts()
    labels = sorted(sizes.index, key=lambda g: (-sizes[g], g))
    stems = [stem_of(g) for g in labels]
    if len(set(stems)) != len(stems) or not all(stems):
        raise SystemExit(f"group names give empty or repeated stems: {dict(zip(labels, stems))}")
    code = {g: i for i, g in enumerate(labels)}
    with h5py.File(args.h5ad, "r") as fh:
        obs = index_of(fh["obs"])
        var = fh["var"]
        gene_ids = index_of(var)
        genes = read_strings(var[args.gene_column]) if args.gene_column in var else gene_ids
        x = fh["X"]
        if not isinstance(x, h5py.Group) or _decode(x.attrs.get("encoding-type", "")) != "csr_matrix":
            raise SystemExit(f"{args.h5ad}: X is not a CSR matrix of cells x genes")
        n_rows, n_genes = (int(v) for v in x.attrs["shape"])
        if n_rows != len(obs) or n_genes != len(genes):
            raise SystemExit(f"{args.h5ad}: X is {n_rows} x {n_genes}, obs {len(obs)}, var {len(genes)}")
        pos = {c: i for i, c in enumerate(obs)}
        row_group = np.full(n_rows, -1, dtype=np.int64)
        found = np.zeros(len(labels), dtype=np.int64)
        missing = 0
        for cid, g in zip(cells[args.id_column], cells[args.group_column]):
            i = pos.get(cid)
            if i is None:
                missing += 1
                continue
            row_group[i] = code[g]
            found[code[g]] += 1
        if missing > 0.01 * len(cells):
            raise SystemExit(f"{missing} of {len(cells)} cells of {args.cells} are not in {args.h5ad}'s obs index")
        indptr = x["indptr"][:]
        n = len(labels) * n_genes
        sums = np.zeros(n, dtype=np.float64)
        nonzero = np.zeros(n, dtype=np.int64)
        for r0 in range(0, n_rows, args.chunk):
            r1 = min(r0 + args.chunk, n_rows)
            g = row_group[r0:r1]
            if (g < 0).all():
                continue
            a, b = int(indptr[r0]), int(indptr[r1])
            data = x["data"][a:b]
            idx = x["indices"][a:b].astype(np.int64)
            gg = np.repeat(g, np.diff(indptr[r0 : r1 + 1]))
            keep = gg >= 0
            key = gg[keep] * n_genes + idx[keep]
            vals = data[keep]
            sums += np.bincount(key, weights=vals, minlength=n)
            nonzero += np.bincount(key[vals > 0], minlength=n)
    sums = sums.reshape(len(labels), n_genes)
    nonzero = nonzero.reshape(len(labels), n_genes)
    totals = sums.sum(axis=1)
    args.out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame({
        "group": np.repeat(stems, n_genes),
        "gene": np.tile(genes, len(labels)),
        "gene_id": np.tile(gene_ids, len(labels)),
        "counts": sums.ravel(),
        "cpm": (sums / np.where(totals > 0, totals, 1)[:, None] * 1e6).ravel(),
        "frac": (nonzero / np.where(found > 0, found, 1)[:, None]).ravel(),
    })  # fmt: skip
    table.to_csv(args.out / "genes_by_group.tsv.gz", sep="\t", index=False, float_format="%.6g")
    groups = pd.DataFrame({"group": stems, "label": labels, "cells": [int(sizes[g]) for g in labels],
                           "cells_found": found, "counts": totals})  # fmt: skip
    groups.to_csv(args.out / "groups.tsv", sep="\t", index=False, float_format="%.0f")
    record = {"h5ad": str(args.h5ad), "h5ad_bytes": args.h5ad.stat().st_size, "cells": str(args.cells),
              "id_column": args.id_column, "group_column": args.group_column, "gene_column": args.gene_column,
              "n_cells": len(cells), "cells_not_in_h5ad": missing, "n_genes": n_genes,
              "seconds": round(time.time() - t0, 1)}  # fmt: skip
    (args.out / "expression.json").write_text(json.dumps(record, indent=1) + "\n")
    print(groups.to_string(index=False))
    print(f"{n_genes} genes x {len(labels)} groups -> {args.out}/genes_by_group.tsv.gz ({record['seconds']} s)")
    return record


# ---------------------------------------------------------------------------------------------------------------------
# annotate


def families(metadata: Path) -> dict[str, list[str]]:
    import pandas as pd

    meta = pd.read_csv(metadata, sep="\t", usecols=["name", "TF"], dtype=str)
    out: dict[str, set[str]] = {}
    for name, tfs in zip(meta["name"], meta["TF"].fillna("")):
        out.setdefault(family_of(name), set()).update(t.strip() for t in tfs.split(",") if t.strip())
    return {f: sorted(g) for f, g in out.items()}


def family_expression(expr, fams: dict[str, list[str]], min_frac: float):
    """(families table, per family and group: top_gene, cpm, frac, n_expressed, expressed)."""
    import pandas as pd

    expr = expr.assign(key=expr["gene"].map(norm))
    best = expr.groupby(["group", "key"], as_index=False).agg(gene=("gene", "first"), cpm=("cpm", "max"),
                                                              frac=("frac", "max"))  # fmt: skip
    known = set(best["key"])
    fam_rows, rows = [], []
    for fam, members in sorted(fams.items()):
        keys = {norm(m) for m in members}
        fam_rows.append({"family": fam, "members": ",".join(members), "n_members": len(members),
                         "n_found": len(keys & known)})  # fmt: skip
        sub = best[best["key"].isin(keys)]
        for group, g in sub.groupby("group"):
            top = g.loc[g["cpm"].idxmax()]
            frac = float(g["frac"].max())
            rows.append({"family": fam, "group": group, "top_gene": top["gene"], "cpm": float(top["cpm"]),
                         "frac": frac, "n_expressed": int((g["frac"] >= min_frac).sum()),
                         "expressed": frac >= min_frac})  # fmt: skip
    return pd.DataFrame(fam_rows), pd.DataFrame(rows, columns=["family", "group", "top_gene", "cpm", "frac",
                                                                "n_expressed", "expressed"])  # fmt: skip


def member_changes(members: dict, genes: dict, ga: str, gb: str, min_frac: float, min_fold: float):
    """(up, down): the members whose CPM changes at least min_fold-fold from group ga to gb and that are expressed
    on the higher side, as GENE(xFOLD), largest change first."""
    up, down = [], []
    for key in members:
        a, b = genes.get((ga, key)), genes.get((gb, key))
        if a is None or b is None:
            continue
        fold = (b["cpm"] + 1) / (a["cpm"] + 1)
        if fold >= min_fold and b["frac"] >= min_frac:
            up.append((fold, f"{b['gene']}(x{fold:.1f})"))
        elif fold <= 1 / min_fold and a["frac"] >= min_frac:
            down.append((1 / fold, f"{a['gene']}(x{1 / fold:.1f})"))
    return [g for _, g in sorted(up, reverse=True)], [g for _, g in sorted(down, reverse=True)]


def annotate_table(df, lookup, prefix: str, min_fold: float, members=None, genes=None, min_frac: float = 0.05):
    """df with the expression columns its shape takes (see the module docstring), or None if it takes none.
    members: family -> its genes' keys; genes: (group, key) -> gene, cpm, frac (for members_up and members_down)."""

    def get(tf, group, col):
        v = lookup.get((tf, group))
        return None if v is None else v[col]

    def log2fc(a, b):
        return None if a is None or b is None else math.log2((b + 1) / (a + 1))

    def changes(ea, eb, fc):
        if ea is None or eb is None:
            return None
        return bool(ea != eb or (fc is not None and abs(fc) >= math.log2(min_fold)))

    def either(a, b):  # True if either is, None if neither is known
        known = [v for v in (a, b) if v is not None]
        return any(known) if known else None

    cols = set(df.columns)
    pair_col = next((c for c in PAIR_SEPARATORS if c in cols), None)
    out = df.copy()
    if {"group_a", "group_b"} <= cols and "tf" in cols:
        vals = {}  # plain lists: a DataFrame column would turn None into NaN
        for side in ("a", "b"):
            g = df[f"group_{side}"]
            for col in ("top_gene", "cpm", "frac", "expressed"):
                vals[col, side] = [get(t, x, col) for t, x in zip(df["tf"], g)]
                out[f"tf_{col}_{side}"] = vals[col, side]
        fc = [log2fc(a, b) for a, b in zip(vals["cpm", "a"], vals["cpm", "b"])]
        out["log2fc_expr"] = fc
        out["expression_changes"] = [changes(a, b, f) for a, b, f in
                                     zip(vals["expressed", "a"], vals["expressed", "b"], fc)]  # fmt: skip
        if members is not None and genes is not None:
            moved = [member_changes(members[t], genes, x, y, min_frac, min_fold) if t in members else (None, None)
                     for t, x, y in zip(df["tf"], df["group_a"], df["group_b"])]  # fmt: skip
            out["members_up"] = [None if u is None else ",".join(u) for u, _ in moved]
            out["members_down"] = [None if d is None else ",".join(d) for _, d in moved]
            out["members_changed"] = [None if u is None else len(u) + len(d) for u, d in moved]
        return out
    if {"group_a", "group_b"} <= cols and pair_col:
        split = df[pair_col].str.split(PAIR_SEPARATORS[pair_col], n=1, regex=False)
        per_tf = []
        for k in (0, 1):
            tf = split.str[k].str.strip()
            ea = [get(t, x, "expressed") for t, x in zip(tf, df["group_a"])]
            eb = [get(t, x, "expressed") for t, x in zip(tf, df["group_b"])]
            fc = [log2fc(get(t, x, "cpm"), get(t, y, "cpm")) for t, x, y in zip(tf, df["group_a"], df["group_b"])]
            out[f"tf{k + 1}_expressed_a"], out[f"tf{k + 1}_expressed_b"], out[f"tf{k + 1}_log2fc_expr"] = ea, eb, fc
            per_tf.append([changes(a, b, f) for a, b, f in zip(ea, eb, fc)])
        out["expression_changes"] = [either(a, b) for a, b in zip(*per_tf)]
        return out
    if "dataset" in cols and ("tf" in cols or pair_col):
        group = df["dataset"].str.removeprefix(prefix) if prefix else df["dataset"]
        if "tf" in cols:
            for col in ("top_gene", "cpm", "frac", "expressed"):
                out[f"tf_{col}"] = [get(t, g, col) for t, g in zip(df["tf"], group)]
            return out
        split = df[pair_col].str.split(PAIR_SEPARATORS[pair_col], n=1, regex=False)
        flags = []
        for k in (0, 1):
            tf = split.str[k].str.strip()
            for col in ("top_gene", "expressed"):
                out[f"tf{k + 1}_{col}"] = [get(t, g, col) for t, g in zip(tf, group)]
            flags.append(out[f"tf{k + 1}_expressed"])
        out["both_expressed"] = [None if a is None or b is None else bool(a and b) for a, b in zip(*flags)]
        return out
    groups = [c[2:] for c in df.columns if c.startswith("n_")]
    if "tf" in cols and groups:
        for g in groups:
            out[f"expressed_{g}"] = [get(t, g, "expressed") for t in df["tf"]]
            out[f"cpm_{g}"] = [get(t, g, "cpm") for t in df["tf"]]
        return out
    if pair_col and groups:
        split = df[pair_col].str.split(PAIR_SEPARATORS[pair_col], n=1, regex=False)
        tf1, tf2 = split.str[0].str.strip(), split.str[1].str.strip()
        for g in groups:
            e1 = [get(t, g, "expressed") for t in tf1]
            e2 = [get(t, g, "expressed") for t in tf2]
            out[f"both_expressed_{g}"] = [None if a is None or b is None else bool(a and b) for a, b in zip(e1, e2)]
        return out
    return None


def annotate(args) -> dict:
    import pandas as pd

    expr = pd.read_csv(args.expression / "genes_by_group.tsv.gz", sep="\t", dtype={"group": str, "gene": str})
    fams = families(args.motif_metadata)
    fam_table, per_group = family_expression(expr, fams, args.min_frac)
    out_dir = args.coop / "expression"
    out_dir.mkdir(parents=True, exist_ok=True)
    fam_table.to_csv(out_dir / "tf_families.tsv", sep="\t", index=False)
    per_group.to_csv(out_dir / "tf_expression.tsv", sep="\t", index=False, float_format="%.6g")
    lookup = {(r.family, r.group): r._asdict() for r in per_group.itertuples(index=False)}
    members = {f: {norm(m) for m in ms} for f, ms in fams.items()}
    genes = {}
    for r in expr.itertuples(index=False):  # per group and gene key, the best of genes sharing a key
        k = (r.group, norm(r.gene))
        if k not in genes or r.cpm > genes[k]["cpm"]:
            genes[k] = {"gene": r.gene, "cpm": float(r.cpm), "frac": float(r.frac)}
    written = []
    for sub in ("combined", "official"):
        for table in sorted((args.coop / sub).glob("*.tsv")):
            df = pd.read_csv(table, sep="\t", dtype=str)
            annotated = annotate_table(df, lookup, args.prefix, args.min_fold, members, genes, args.min_frac)
            if annotated is None:
                continue
            dest = out_dir / sub / table.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            annotated.to_csv(dest, sep="\t", index=False, float_format="%.6g")
            written.append(str(dest))
    groups = sorted(per_group["group"].unique())
    record = {"expression": str(args.expression), "motif_metadata": str(args.motif_metadata),
              "min_frac": args.min_frac, "min_fold": args.min_fold, "prefix": args.prefix, "groups": groups,
              "families": len(fams), "written": written}  # fmt: skip
    (out_dir / "annotate.json").write_text(json.dumps(record, indent=1) + "\n")
    print(f"{len(fams)} families, {len(groups)} groups; {len(written)} tables annotated under {out_dir}")
    return record


def metadata_path(reference_root: Path) -> tuple[str, Path]:
    """(URL, local path) of MotifCompendium's metadata table, beside the MEME database the vendored layout names."""
    sys.path.insert(0, str(HERE.parent / "pipeline"))
    import project  # noqa: PLC0415  # stdlib only; loads the vendored references module

    refs = project._vendored_references()
    url = refs.MOTIF_DB_URL.replace(".meme.txt", ".metadata.tsv")
    meme = Path(refs.layout(str(reference_root))["ref_db_meme"])
    return url, meme.with_name(meme.name.replace(".meme.txt", ".metadata.tsv"))


def fetch_metadata(args) -> Path:
    import hashlib
    import urllib.request

    url, dest = metadata_path(args.reference_root)
    if dest.is_file() and hashlib.md5(dest.read_bytes()).hexdigest() == MOTIF_METADATA_MD5:
        print(f"in place: {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as r:
        data = r.read()
    md5 = hashlib.md5(data).hexdigest()
    if md5 != MOTIF_METADATA_MD5:
        raise SystemExit(f"{url}: md5 {md5}, expected {MOTIF_METADATA_MD5}")
    tmp = dest.with_name(dest.name + ".partial")
    tmp.write_bytes(data)
    tmp.replace(dest)
    print(f"fetched: {dest}")
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    e = sub.add_parser("expression")
    e.add_argument("--h5ad", type=Path, required=True)
    e.add_argument("--cells", type=Path, required=True)
    e.add_argument("--group-column", required=True)
    e.add_argument("--id-column", default="barcode")
    e.add_argument("--gene-column", default="gene_symbols")
    e.add_argument("--chunk", type=int, default=20000)
    e.add_argument("--out", type=Path, required=True)
    a = sub.add_parser("annotate")
    a.add_argument("--expression", type=Path, required=True)
    a.add_argument("--motif-metadata", type=Path, required=True)
    a.add_argument("--coop", type=Path, required=True)
    a.add_argument("--prefix", default="")
    a.add_argument("--min-frac", type=float, default=0.05)
    a.add_argument("--min-fold", type=float, default=2.0)
    f = sub.add_parser("fetch-metadata")
    f.add_argument("--reference-root", type=Path, required=True)
    args = ap.parse_args(argv)
    {"expression": expression, "annotate": annotate, "fetch-metadata": fetch_metadata}[args.command](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
