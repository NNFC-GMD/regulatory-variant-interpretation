"""compare_compendia.py, the comparison's level 4 (CPU only, synthetic compendia from compare_fixtures).

ChromBPNet's compendium of the shared datasets holds clusters of TF0-TF3 (pos) and TF4 (neg); Cherimoya's holds TF0,
TF1 at three times the attribution scale, TF5, TN5_1 (pos) and TF4 (neg). So three clusters pair up, whatever the
models' scales. The joint compendium has two shared clusters, one of each model alone; Cherimoya's compendium of
every dataset adds a TF7 cluster.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("h5py")
pytest.importorskip("memelite")

TESTS = Path(__file__).resolve().parent
sys.path[:0] = [
    str(TESTS),
    str(TESTS.parent / "workflows" / "cherimoya"),
    str(TESTS.parent / "workflows" / "cherimoya" / "compare"),
]

import compare_compendia as cc  # noqa: E402
from compare_fixtures import make_compendium, motif_db  # noqa: E402


def meta(rows: list[tuple[str, str, int, str, str, float, str, str]]) -> list[tuple]:
    """(label, pattern, seqlets, posneg, annotation, score, cluster) -> _meta.tsv rows."""
    return [(f"{label}-{pattern}", n, label, posneg, score, annotation, cid)
            for label, pattern, n, posneg, annotation, score, cid in rows]  # fmt: skip


@pytest.fixture
def world(tmp_path):
    _, db = motif_db(tmp_path)
    root = tmp_path / "compendia"
    make_compendium(root, "cbp10", [("0", "pos", db["TF0_0"], 1.0), ("1", "pos", db["TF1_0"], 1.0),
                                    ("2", "pos", db["TF2_0"], 1.0), ("3", "pos", db["TF3_0"], 1.0),
                                    ("4", "neg", db["TF4_0"], 1.0)],
                    meta([("cbp_ds_a", f"pos.pattern_{i}", 100, "pos", f"TF{i}_0", 0.9, str(i)) for i in range(4)]
                         + [("cbp_ds_a", "neg.pattern_0", 50, "neg", "TF4_0", 0.9, "4")]))  # fmt: skip
    make_compendium(root, "cheri10", [("0", "pos", db["TF0_0"], 3.0), ("1", "pos", db["TF1_0"], 3.0),
                                      ("2", "pos", db["TF5_0"], 3.0), ("3", "pos", db["TN5_1"], 3.0),
                                      ("4", "neg", db["TF4_0"], 3.0)],
                    meta([("cheri_ds_a", "pos.pattern_0", 120, "pos", "TF0_0", 0.9, "0"),
                          ("cheri_ds_a", "pos.pattern_1", 80, "pos", "TF1_1", 0.8, "1"),
                          ("cheri_ds_a", "pos.pattern_2", 60, "pos", "TF5_0", 0.95, "2"),
                          ("cheri_ds_a", "pos.pattern_3", 300, "pos", "ZNF9_0", 0.4, "3"),  # below 0.7: none
                          ("cheri_ds_a", "neg.pattern_0", 40, "neg", "TF4_0", 0.9, "4")]))  # fmt: skip
    make_compendium(root, "joint20", [("0", "pos", db["TF0_0"], 1.0), ("1", "pos", db["TF2_0"], 1.0),
                                      ("2", "pos", db["TN5_1"], 1.0), ("3", "neg", db["TF4_0"], 1.0)],
                    meta([("cbp_ds_a", "pos.pattern_0", 100, "pos", "TF0_0", 0.9, "0"),
                          ("cbp_ds_b", "pos.pattern_0", 90, "pos", "TF0_0", 0.9, "0"),
                          ("cheri_ds_a", "pos.pattern_0", 120, "pos", "TF0_0", 0.9, "0"),
                          ("cbp_ds_a", "pos.pattern_2", 70, "pos", "TF2_0", 0.9, "1"),
                          ("cheri_ds_a", "pos.pattern_3", 300, "pos", "ZNF9_0", 0.4, "2"),
                          ("cheri_ds_b", "pos.pattern_1", 200, "pos", "ZNF9_0", 0.4, "2"),
                          ("cbp_ds_a", "neg.pattern_0", 50, "neg", "TF4_0", 0.9, "3"),
                          ("cheri_ds_a", "neg.pattern_0", 40, "neg", "TF4_0", 0.9, "3")]))  # fmt: skip
    level3 = tmp_path / "comparison" / "modisco" / "ds_a"
    level3.mkdir(parents=True)
    pd.DataFrame({"model": ["cherimoya", "cherimoya", "chrombpnet"],
                  "pattern": ["pos.pattern_3", "pos.pattern_0", "pos.pattern_0"],
                  "bias_flag": [True, False, False]}).to_csv(level3 / "patterns.tsv", sep="\t", index=False)  # fmt: skip
    listing = tmp_path / "list.txt"
    listing.write_text("ds_a\nds_b\n")
    return {"root": root, "db": db, "list": listing}


def run(tmp_path: Path, world: dict, *extra: str, capsys=None) -> tuple[int, str]:
    rc = cc.main(["--compendia-root", str(world["root"]), "--comparison-root", str(tmp_path / "comparison"),
                  "--datasets", str(world["list"]), *extra])  # fmt: skip
    return rc, capsys.readouterr().out if capsys else ""


def result_line(out: str) -> str:
    lines = [line for line in out.splitlines() if line.startswith("RESULT: ")]
    assert lines, out
    return lines[-1][len("RESULT: ") :]


def test_like_for_like_and_the_joint_classes(tmp_path, world, capsys):
    rc, out = run(tmp_path, world, capsys=capsys)
    assert rc == 0, out
    od = tmp_path / "comparison" / "compendium"
    lfl = json.loads((od / "like_for_like.json").read_text())
    assert lfl["clusters"] == {"cbp": 5, "cheri": 5}
    assert lfl["matched"]["reciprocal_pairs"] == 3  # TF0, TF1 (at another scale) and TF4
    m = pd.read_csv(od / "matches_cbp10_vs_cheri10.tsv", sep="\t", dtype={"cluster": str, "best": str})
    pairs = {(c, b) for c, b, r in zip(m["cluster"], m["best"], m["reciprocal"]) if r}
    assert pairs == {("0", "0"), ("1", "1"), ("4", "4")}
    assert lfl["annotations"]["shared"] == ["TF0_0", "TF4_0"]
    assert lfl["annotations"]["jaccard"] == pytest.approx(2 / 7)
    assert lfl["families"]["shared"] == ["TF0", "TF1", "TF4"] and lfl["families"]["jaccard"] == pytest.approx(0.5)
    joint = pd.read_csv(od / "joint_clusters.tsv", sep="\t", dtype={"cluster_id": str})
    cls = dict(zip(joint["cluster_id"], joint["class"]))
    assert cls == {"0": "shared", "1": "chrombpnet_only", "2": "cherimoya_only", "3": "shared"}
    row = joint.set_index("cluster_id").loc["0"]
    assert (row["patterns_chrombpnet"], row["datasets_chrombpnet"], row["seqlets_chrombpnet"]) == (2, 2, 190)
    assert (row["patterns_cherimoya"], row["seqlets_cherimoya"]) == (1, 120)
    assert joint.set_index("cluster_id").loc["2", "bias_share"] == 1.0  # ds_a's pattern flagged; ds_b has no table
    assert pd.isna(joint.set_index("cluster_id").loc["2", "annotation"])  # its annotation scores below 0.7
    summary = pd.read_csv(od / "joint_summary.tsv", sep="\t").set_index("class")
    assert summary.loc["shared", "clusters"] == 2 and summary.loc["cherimoya_only", "seqlets_cherimoya"] == 500
    assert (od / "compendia.png").stat().st_size > 0
    assert result_line(out).startswith("cbp10 5 vs cheri10 5 clusters, 3 reciprocal pairs, annotation Jaccard 0.29; "
                                       "joint20: 2 shared, 1 ChromBPNet-only, 1 Cherimoya-only clusters")  # fmt: skip


def test_supplement_sensitivity_and_rerun(tmp_path, world, capsys):
    assert run(tmp_path, world, capsys=capsys)[0] == 0
    rc, out = run(tmp_path, world, capsys=capsys)
    assert rc == 0 and result_line(out) == "up to date"
    db = world["db"]
    make_compendium(world["root"], "cheri39", [("0", "pos", db["TF0_0"], 2.0), ("1", "pos", db["TF5_0"], 2.0),
                                               ("2", "pos", db["TF7_0"], 2.0)],
                    meta([("cheri_ds_a", "pos.pattern_0", 100, "pos", "TF0_0", 0.9, "0"),
                          ("cheri_ds_a", "pos.pattern_2", 60, "pos", "TF5_0", 0.95, "1"),
                          ("cheri_ds_z", "pos.pattern_0", 60, "pos", "TF7_0", 0.95, "2")]))  # fmt: skip
    make_compendium(world["root"], "joint20_t090", [("0", "pos", db["TF0_0"], 1.0)],
                    meta([("cbp_ds_a", "pos.pattern_0", 100, "pos", "TF0_0", 0.9, "0"),
                          ("cheri_ds_a", "pos.pattern_0", 120, "pos", "TF0_0", 0.9, "0")]))  # fmt: skip
    rc, out = run(tmp_path, world, capsys=capsys)
    assert rc == 0, out
    od = tmp_path / "comparison" / "compendium"
    sup = json.loads((od / "supplement.json").read_text())
    assert sup["not_in_shared"] == 1 and sup["annotations_added"] == ["TF7_0"]
    assert (od / "matches_cbp10_vs_cheri39.tsv").is_file()
    sens = json.loads((od / "sensitivity.json").read_text())
    assert sens["classes"] == {"shared": 1, "chrombpnet_only": 0, "cherimoya_only": 0}
    assert "cheri39: 1 clusters not in cheri10" in result_line(out) and "at joint20_t090: 1/0/0" in result_line(out)


def test_waits_for_the_compendia(tmp_path, world, capsys):
    (world["root"] / "joint20" / "compendium.json").unlink()
    rc, out = run(tmp_path, world, capsys=capsys)
    assert rc == 3 and result_line(out).startswith("waiting: not built yet: joint20")
    assert not (tmp_path / "comparison" / "compendium").exists()


def test_family():
    assert cc.family("FOS-JUN_0") == "FOS-JUN" and cc.family("ZNF_437") == "ZNF" and cc.family(None) is None
    assert cc.model_of("cbp_x") == "chrombpnet" and cc.model_of("cheri_x") == "cherimoya" and cc.model_of("x") is None
