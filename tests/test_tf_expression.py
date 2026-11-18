"""workflows/cooperativity/tf_expression.py: counts per cell group from an h5ad, TF families from MotifCompendium's
metadata, and the cooperativity tables annotated with each TF's expression on each side of a comparison."""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
h5py = pytest.importorskip("h5py")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflows" / "cooperativity"))

import tf_expression as te  # noqa: E402

CELLS = ["w1_A", "w1_B", "w1_C", "w2_A", "w2_D", "w3_X"]
GENES = ["GATA1", "GATA2", "NKX6-1", "ACTB"]
DENSE = np.array([[0, 0, 2, 10], [0, 0, 0, 5], [0, 0, 1, 3],  # HSC
                  [4, 1, 1, 6], [2, 0, 1, 2],  # MEP/early
                  [9, 9, 9, 9]], dtype=np.float32)  # unlabelled  # fmt: skip
LABELS = {"w1_A": "HSC", "w1_B": "HSC", "w1_C": "HSC", "w2_A": "MEP/early", "w2_D": "MEP/early"}


def _h5ad(path: Path) -> Path:
    """The AnnData layout: obs and var with an _index, gene_symbols categorical, X a CSR matrix."""
    with h5py.File(path, "w") as fh:
        obs = fh.create_group("obs")
        obs.attrs["_index"] = "_index"
        obs.create_dataset("_index", data=np.array(CELLS, dtype="S"))
        var = fh.create_group("var")
        var.attrs["_index"] = "_index"
        var.create_dataset("_index", data=np.array([f"ENSG{i}" for i in range(len(GENES))], dtype="S"))
        sym = var.create_group("gene_symbols")
        sym.create_dataset("categories", data=np.array(GENES, dtype="S"))
        sym.create_dataset("codes", data=np.arange(len(GENES), dtype=np.int8))
        x = fh.create_group("X")
        x.attrs["encoding-type"] = "csr_matrix"
        x.attrs["shape"] = DENSE.shape
        rows, cols = np.nonzero(DENSE)
        x.create_dataset("data", data=DENSE[rows, cols])
        x.create_dataset("indices", data=cols.astype(np.int32))
        x.create_dataset("indptr", data=np.concatenate([[0], np.cumsum(np.count_nonzero(DENSE, axis=1))]))
    return path


def _expression(tmp_path: Path, cells: dict[str, str] = LABELS) -> Path:
    csv = tmp_path / "cells.csv"
    csv.write_text("barcode,umap,cell_type\n" + "".join(f"{c},0.1,{g}\n" for c, g in cells.items()))
    out = tmp_path / "expr"
    te.main(["expression", "--h5ad", str(_h5ad(tmp_path / "x.h5ad")), "--cells", str(csv), "--group-column",
             "cell_type", "--out", str(out), "--chunk", "2"])  # fmt: skip
    return out


def test_counts_cpm_and_fraction_per_group_match_the_matrix(tmp_path):
    out = _expression(tmp_path)
    table = pd.read_csv(out / "genes_by_group.tsv.gz", sep="\t").set_index(["group", "gene"])
    for group, rows in (("hsc", [0, 1, 2]), ("mep_early", [3, 4])):  # the stems pseudobulk.py gives
        sub = DENSE[rows]
        for j, gene in enumerate(GENES):
            r = table.loc[(group, gene)]
            assert r["counts"] == pytest.approx(sub[:, j].sum())
            assert r["cpm"] == pytest.approx(sub[:, j].sum() / sub.sum() * 1e6, rel=1e-5)
            assert r["frac"] == pytest.approx((sub[:, j] > 0).mean())
    groups = pd.read_csv(out / "groups.tsv", sep="\t").set_index("group")
    assert groups.loc["hsc", "cells_found"] == 3 and groups.loc["mep_early", "label"] == "MEP/early"
    assert "w3_X" not in gzip.open(out / "genes_by_group.tsv.gz", "rt").read()  # unlabelled cells are left out


def test_cells_missing_from_the_h5ad_stop_it(tmp_path):
    with pytest.raises(SystemExit, match="not in"):
        _expression(tmp_path, {**LABELS, "w9_Z": "HSC"})


def _coop(tmp_path: Path) -> Path:
    coop = tmp_path / "coop"
    (coop / "combined").mkdir(parents=True)
    (coop / "official").mkdir()
    (coop / "combined" / "changes_tfs.tsv").write_text(
        "comparison\tgroup_a\tgroup_b\ttf\tn_a\tn_b\tdelta\n"
        "hsc_vs_mep\thsc\tmep_early\tGATA\t60\t70\t0.2\n"
        "hsc_vs_mep\thsc\tmep_early\tNKX\t80\t90\t-0.1\n"
        "hsc_vs_mep\thsc\tmep_early\tpos_patterns.3\t55\t66\t0.0\n"
    )
    (coop / "combined" / "changes_pairs.tsv").write_text(
        "comparison\tgroup_a\tgroup_b\tpair\tdelta\nhsc_vs_mep\thsc\tmep_early\tGATA + NKX\t0.3\n"
    )
    (coop / "combined" / "tfs_by_dataset.tsv").write_text("dataset\ttf\tn\tsynergy\njl_hsc\tGATA\t60\t0.4\n")
    (coop / "combined" / "trend_tfs.tsv").write_text("tf\tn_hsc\tn_mep_early\tslope\nNKX\t80\t90\t0.1\n")
    (coop / "combined" / "distance.tsv").write_text("bin\tshare\n0-50\t0.2\n")  # no TF: left alone
    (coop / "official" / "coop_tf_pairs_by_dataset.tsv").write_text("dataset\ttf_pair\tn\njl_mep_early\tGATA|NKX\t20\n")
    return coop


def _metadata(tmp_path: Path) -> Path:
    path = tmp_path / "metadata.tsv"
    path.write_text("name\tsource_cluster\tTF\n" + "GATA_0\t0\tGATA1,GATA2\n" + "GATA_1\t1\tGATA3\n"
                    + "NKX_0\t2\tNKX61\n" + "OTHER_0\t3\tFOO\n")  # fmt: skip
    return path


def test_tables_get_each_tfs_expression_on_each_side(tmp_path):
    expr = _expression(tmp_path)
    coop = _coop(tmp_path)
    te.main(["annotate", "--expression", str(expr), "--motif-metadata", str(_metadata(tmp_path)), "--coop",
             str(coop), "--prefix", "jl_"])  # fmt: skip
    out = coop / "expression"
    fams = pd.read_csv(out / "tf_families.tsv", sep="\t").set_index("family")
    assert fams.loc["GATA", "members"] == "GATA1,GATA2,GATA3" and fams.loc["GATA", "n_found"] == 2
    assert fams.loc["NKX", "n_found"] == 1  # NKX61 is NKX6-1
    tfs = pd.read_csv(out / "combined" / "changes_tfs.tsv", sep="\t").set_index("tf")
    gata, nkx = tfs.loc["GATA"], tfs.loc["NKX"]
    assert not gata["tf_expressed_a"] and gata["tf_expressed_b"] and gata["expression_changes"]
    assert gata["tf_top_gene_b"] == "GATA1" and gata["tf_cpm_b"] == pytest.approx(6 / 17 * 1e6, rel=1e-5)
    assert nkx["tf_expressed_a"] and nkx["tf_expressed_b"] and not nkx["expression_changes"]  # 3/21 vs 2/17
    assert pd.isna(tfs.loc["pos_patterns.3", "tf_expressed_a"])  # a motif without annotation has no family
    assert list(tfs.columns[:6]) == ["comparison", "group_a", "group_b", "n_a", "n_b", "delta"]  # originals first
    pairs = pd.read_csv(out / "combined" / "changes_pairs.tsv", sep="\t").iloc[0]
    assert not pairs["tf1_expressed_a"] and pairs["tf2_expressed_a"] and pairs["expression_changes"]
    by_ds = pd.read_csv(out / "combined" / "tfs_by_dataset.tsv", sep="\t").iloc[0]
    assert by_ds["tf_expressed"] == False  # noqa: E712  # GATA in hsc
    trend = pd.read_csv(out / "combined" / "trend_tfs.tsv", sep="\t").iloc[0]
    assert trend["expressed_hsc"] and trend["expressed_mep_early"]
    official = pd.read_csv(out / "official" / "coop_tf_pairs_by_dataset.tsv", sep="\t").iloc[0]
    assert official["both_expressed"]  # GATA and NKX in mep_early
    assert not (out / "combined" / "distance.tsv").exists()
    assert (coop / "combined" / "changes_tfs.tsv").read_text().count("\t") == 4 * 6  # the original is untouched


def test_the_metadata_comes_from_the_motif_database_s_commit_and_sits_beside_it(tmp_path):
    url, dest = te.metadata_path(tmp_path)
    assert url.startswith("https://raw.githubusercontent.com/kundajelab/MotifCompendium/7e9d1c2")
    assert url.endswith("/pipeline/data/MotifCompendium-Database-Human.metadata.tsv")
    assert dest == tmp_path / "motif" / "MotifCompendium-1.0.19-Database-Human.metadata.tsv"
