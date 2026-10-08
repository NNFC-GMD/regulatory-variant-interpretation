"""compare_modisco.py and motifs.py, the comparison's level 3 (CPU only, synthetic TF-MoDISco results).

compare_fixtures.make_modisco_dataset writes both models' fold-average and per-fold QC results from motifs of one
database (40 distinct motifs, TN5_1 and DNASE_1): ChromBPNet finds TF0-TF3 (pos) and TF4 (neg), Cherimoya TF0-TF2,
TN5_1 (pos) and TF5 (neg). Their seqlets sit at known peaks and offsets, so which patterns match, which seqlets
overlap and which pattern is bias are known in advance. Cherimoya's beds lack two peaks, so a seqlet's row differs
between the models and only the join through each model's own bed puts them on the same peak.
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
pytest.importorskip("yaml")

TESTS = Path(__file__).resolve().parent
sys.path[:0] = [
    str(TESTS),
    str(TESTS.parent / "workflows" / "cherimoya"),
    str(TESTS.parent / "workflows" / "cherimoya" / "compare"),
]

import compare_modisco as cm  # noqa: E402
import motifs  # noqa: E402
from compare_fixtures import (  # noqa: E402
    make_modisco_dataset,
    motif_db,
    seqlets_at,
    write_joint,
    write_modisco,
)


def run(tmp_path: Path, *args: str, capsys=None) -> tuple[int, str]:
    rc = cm.main(["--cheri-root", str(tmp_path / "cheri"), "--comparison-root", str(tmp_path / "comparison"),
                  "--compendia-root", str(tmp_path / "compendia"), "--n-boot", "300", *args])  # fmt: skip
    return rc, capsys.readouterr().out if capsys else ""


def result_line(out: str) -> str:
    lines = [line for line in out.splitlines() if line.startswith("RESULT: ")]
    assert lines, out
    return lines[-1][len("RESULT: ") :]


@pytest.fixture
def world(tmp_path):
    db_path, db = motif_db(tmp_path)
    data = make_modisco_dataset(tmp_path, "ds_a", db)
    return {"db_path": db_path, "db": db, "data": data}


def patterns_table(tmp_path: Path, ds: str = "ds_a") -> pd.DataFrame:
    return pd.read_csv(tmp_path / "comparison" / "modisco" / ds / "patterns.tsv", sep="\t", keep_default_na=False,
                       na_values=["NA", ""]).set_index(["model", "pattern"])  # fmt: skip


def test_fold_averages_annotations_matches_and_seqlets(tmp_path, world, capsys):
    write_joint(tmp_path / "compendia", ["ds_a"], world["db_path"])
    rc, out = run(tmp_path, "--dataset", "ds_a", "--part", "average", capsys=capsys)
    assert rc == 0, out
    df = patterns_table(tmp_path)
    want = {("chrombpnet", "pos.pattern_0"): "TF0_0", ("chrombpnet", "pos.pattern_3"): "TF3_0",
            ("chrombpnet", "neg.pattern_0"): "TF4_0", ("cherimoya", "pos.pattern_3"): "TN5_1",
            ("cherimoya", "neg.pattern_0"): "TF5_0"}  # fmt: skip
    for key, motif in want.items():
        assert df.loc[key, "annotation"] == motif and df.loc[key, "annotation_q"] < 0.05
    assert df["bias_flag"].astype(str).eq("True").sum() == 1 and str(df.loc[("cherimoya", "pos.pattern_3"),
                                                                           "bias_flag"]) == "True"  # fmt: skip
    matched = df["matched_cluster"].astype(str).eq("True")
    assert sorted(df.index[matched]) == sorted([(m, f"pos.pattern_{i}") for m in ("chrombpnet", "cherimoya")
                                                for i in range(3)])  # fmt: skip
    assert df.loc[("cherimoya", "pos.pattern_1"), "tomtom_partner"] == "pos.pattern_1"
    assert pd.isna(df.loc[("chrombpnet", "pos.pattern_3"), "tomtom_partner"])  # TF3: only the database has it
    # seqlets: TF0 at the same positions (all overlapped), TF1 20 bp apart (none), TF2 10 bp apart (20 of 30 bp);
    # the peak Cherimoya dropped (5) is not counted
    assert df.loc[("chrombpnet", "pos.pattern_0"), "seqlets_on_shared_peaks"] == 9
    assert df.loc[("chrombpnet", "pos.pattern_0"), "seqlet_overlap_matched"] == 1.0
    assert df.loc[("chrombpnet", "pos.pattern_1"), "seqlet_overlap_any"] == 0.0
    assert df.loc[("cherimoya", "pos.pattern_2"), "seqlet_overlap_matched"] == 1.0
    assert df.loc[("chrombpnet", "neg.pattern_0"), "seqlet_overlap_any"] == 1.0  # TF4 sits where Cherimoya's TN5 does
    pairs = pd.read_csv(tmp_path / "comparison" / "modisco" / "ds_a" / "pairs.tsv", sep="\t")
    assert sorted(zip(pairs["chrombpnet"], pairs["cherimoya"])) == [(f"pos.pattern_{i}", f"pos.pattern_{i}")
                                                                    for i in range(3)]  # fmt: skip
    av = json.loads((tmp_path / "comparison" / "modisco" / "ds_a" / "average.json").read_text())
    assert av["matching"] == "cluster" and av["windows"] == {"chrombpnet": 500, "cherimoya": 500}
    cbp, che = av["models"]["chrombpnet"], av["models"]["cherimoya"]
    assert cbp["patterns"] == {"pos": 4, "neg": 1} and che["patterns"] == {"pos": 4, "neg": 1}
    assert cbp["shared"]["cluster"]["by_count"] == pytest.approx(0.6)
    assert cbp["shared"]["cluster"]["by_seqlets"] == pytest.approx(25 / 35)
    assert che["shared"]["cluster"]["by_seqlets"] == pytest.approx(23 / 38)
    assert {u["pattern"] for u in che["unmatched"]} == {"pos.pattern_3", "neg.pattern_0"}
    assert che["bias_flagged"] == 1 and cbp["bias_flagged"] == 0
    assert result_line(out).startswith("5 ChromBPNet and 5 Cherimoya patterns; matched by cluster: 0.60 / 0.60")


def test_without_the_joint_compendium_it_matches_by_tomtom_and_reruns_once_it_appears(tmp_path, world, capsys):
    rc, out = run(tmp_path, "--dataset", "ds_a", "--part", "average", "--motif-db", str(world["db_path"]),
                  capsys=capsys)  # fmt: skip
    assert rc == 0, out
    av = json.loads((tmp_path / "comparison" / "modisco" / "ds_a" / "average.json").read_text())
    assert av["matching"] == "tomtom" and av["joint"] is None
    assert av["models"]["cherimoya"]["shared"]["tomtom"]["by_count"] == pytest.approx(0.6)
    rc, out = run(tmp_path, "--dataset", "ds_a", "--part", "average", "--motif-db", str(world["db_path"]),
                  capsys=capsys)  # fmt: skip
    assert result_line(out) == "up to date"
    write_joint(tmp_path / "compendia", ["ds_a"], world["db_path"])
    rc, out = run(tmp_path, "--dataset", "ds_a", "--part", "average", "--motif-db", str(world["db_path"]),
                  capsys=capsys)  # fmt: skip
    assert rc == 0 and "matched by cluster" in result_line(out)


def test_qc_folds_cross_model_and_fold_to_fold(tmp_path, world, capsys):
    write_joint(tmp_path / "compendia", ["ds_a"], world["db_path"])
    rc, out = run(tmp_path, "--dataset", "ds_a", "--part", "qc", capsys=capsys)
    assert rc == 0, out
    od = tmp_path / "comparison" / "modisco" / "ds_a"
    pairs = pd.read_csv(od / "qc_pairs.tsv", sep="\t", dtype={"fold_a": str, "fold_b": str})
    cross = pairs[pairs["kind"] == "cross"]
    assert list(cross["fold_a"]) == ["0", "1", "2"]
    assert np.allclose(cross["matched_a"], 2 / 3) and np.allclose(cross["matched_b"], 2 / 3)
    within = pairs[pairs["kind"] == "within_cherimoya"]
    assert len(within) == 3 and np.allclose(within["matched_a"], 1.0)
    assert (cross["seqlet_overlap_a"] > 0.5).all()  # TF0 and TF1 seqlets coincide; TF2's sit on TN5's
    qc = json.loads((od / "qc.json").read_text())
    assert qc["within_chrombpnet"] == 1.0 and qc["cross_cherimoya"] == pytest.approx(2 / 3)
    patterns = pd.read_csv(od / "qc_patterns.tsv", sep="\t", dtype={"fold": str})
    tn5 = patterns[(patterns["model"] == "cherimoya") & (patterns["annotation"] == "TN5_1")]
    assert len(tn5) == 3 and tn5["bias_flag"].astype(str).eq("True").all()
    assert result_line(out).startswith("3 folds: Cherimoya patterns matched in ChromBPNet 0.67")


@pytest.mark.parametrize("what", ["cheri_08", "young_cbp", "qc_fold"])
def test_missing_or_unsettled_inputs_wait(tmp_path, world, capsys, what):
    write_joint(tmp_path / "compendia", ["ds_a"], world["db_path"])
    data = world["data"]
    part = "average"
    if what == "cheri_08":
        (data["cheri_out"] / ".done" / "08.0").unlink()
    elif what == "young_cbp":
        (data["cbp_out"] / "contrib_scores" / "ds_a" / "modisco" / "modisco_counts_results.h5").touch()
    else:
        part = "qc"
        (data["cheri_out"] / "full_models" / "ds_a_all_fold_1" / "evaluation" / "motif_qc_counts" /
         "cherimoya_qc.done.json").unlink()  # fmt: skip
    rc, out = run(tmp_path, "--dataset", "ds_a", "--part", part, capsys=capsys)
    assert rc == 3, out
    assert result_line(out).startswith("waiting: ds_a: ")
    assert not (tmp_path / "comparison" / "modisco" / "ds_a").exists()


def test_bias_control_rejected_then_a_reference(tmp_path, world, capsys):
    write_joint(tmp_path / "compendia", ["ds_a"], world["db_path"])
    assert run(tmp_path, "--dataset", "ds_a", "--part", "average", capsys=capsys)[0] == 0
    assert run(tmp_path, "--dataset", "ds_a", "--part", "qc", capsys=capsys)[0] == 0
    control = tmp_path / "comparison" / "bias_control" / "ctl_fold0"
    control.mkdir(parents=True)
    (control / "status.json").write_text(json.dumps({"state": "rejected", "reason": "counts head refused",
                                                     "dataset": "ds_a", "fold": "0", "head": "counts"}))  # fmt: skip
    listing = tmp_path / "list.txt"
    listing.write_text("ds_a\n")
    rc, out = run(tmp_path, "--bias", "--datasets", str(listing), capsys=capsys)
    assert rc == 0, out
    assert "no bias reference with TF-MoDISco: ctl_fold0: rejected (counts head refused)" in result_line(out)
    report = json.loads((tmp_path / "comparison" / "modisco" / "bias_control.json").read_text())
    assert report["references"] == [] and report["unavailable"]
    # the control is run again and its MoDISco done: Cherimoya's TN5 pattern is found among its patterns
    (control / "status.json").write_text(json.dumps({"state": "done", "dataset": "ds_a", "fold": "0",
                                                     "head": "counts"}))  # fmt: skip
    peaks = {i: i for i in range(10)}
    write_modisco(control / "motif_qc_counts" / "chrombpnet_modisco_results.h5",
                  {"pos": [(world["db"]["TN5_1"], seqlets_at(list(range(10)), peaks, 400)),
                           (world["db"]["TF9_0"], seqlets_at(list(range(5)), peaks, 400))]}, 400)  # fmt: skip
    rc, out = run(tmp_path, "--bias", "--datasets", str(listing), capsys=capsys)
    assert rc == 0, out
    df = pd.read_csv(tmp_path / "comparison" / "modisco" / "bias_control.tsv", sep="\t")
    # checked: the fold averages' TN5_1 and TF5 (neg) patterns, and fold 0's QC TN5_1 pattern
    assert set(zip(df["scale"], df["pattern"])) == {("fold average", "pos.pattern_3"), ("fold average", "neg.pattern_0"),
                                                    ("QC fold 0", "pos.pattern_2")}  # fmt: skip
    hits = df[df["matched"].astype(str) == "True"]
    assert set(zip(hits["scale"], hits["pattern"])) == {
        ("fold average", "pos.pattern_3"),
        ("QC fold 0", "pos.pattern_2"),
    }
    assert result_line(out).startswith("1 reference(s): 2 of 3 unmatched Cherimoya patterns match one")


def test_summary_over_datasets(tmp_path, world, capsys):
    make_modisco_dataset(tmp_path, "ds_b", world["db"], seed=1)
    write_joint(tmp_path / "compendia", ["ds_a", "ds_b"], world["db_path"])
    listing = tmp_path / "list.txt"
    listing.write_text("ds_a\nds_b\n")
    for ds in ("ds_a", "ds_b"):
        for part in ("average", "qc"):
            assert run(tmp_path, "--dataset", ds, "--part", part, capsys=capsys)[0] == 0
    rc, out = run(tmp_path, "--summarize", "--datasets", str(listing), capsys=capsys)
    assert rc == 0, out
    od = tmp_path / "comparison" / "modisco"
    summary = pd.read_csv(od / "summary.tsv", sep="\t")
    assert list(summary["dataset"]) == ["ds_a", "ds_b"]
    assert np.allclose(summary["cherimoya_shared_cluster_count"], 0.6)
    assert np.allclose(summary["qc_within_cherimoya"], 1.0)
    stats = pd.read_csv(od / "statistics.tsv", sep="\t")
    assert (stats["over"] == "folds").sum() == 2 and set(stats[stats["over"] == "folds"]["n"]) == {6}
    unmatched = pd.read_csv(od / "unmatched.tsv", sep="\t")
    assert len(unmatched) == 2 * 4  # per dataset: ChromBPNet TF3, TF4; Cherimoya TN5, TF5
    assert (od / "shared_patterns.png").stat().st_size > 0
    assert result_line(out).startswith("2 of 2 datasets with fold averages, 2 with QC")


def test_motif_helpers(tmp_path, world):
    p = np.array([[0.01, 0.04, 0.03, 0.5], [0.2, 0.001, 0.9, 0.04]])
    q = motifs.bh(p)
    for i in range(2):
        order = np.argsort(p[i])
        ranked = p[i][order] * 4 / np.arange(1, 5)
        want = np.minimum.accumulate(ranked[::-1])[::-1]
        assert np.allclose(q[i][order], want)
    names, mats = motifs.read_meme(world["db_path"])
    assert names[:2] == ["TF0_0", "TF1_0"] and names[-2:] == ["TN5_1", "DNASE_1"] and mats[0].shape == (4, 12)
    assert motifs.is_bias("TN5_3") and motifs.is_bias("DNASE_1") and not motifs.is_bias("TF3_0")
    a = {
        "chrom": np.array(["chr1", "chr1", "chr2"]),
        "start": np.array([100, 200, 100]),
        "end": np.array([130, 230, 130]),
    }
    b = {
        "chrom": np.array(["chr1", "chr2", "chr1"]),
        "start": np.array([114, 140, 216]),
        "end": np.array([144, 170, 246]),
    }
    # 16 of 30 bp overlap (>= 15), 14 of 30 (< 15), and chr2 too far
    assert motifs.reciprocal_overlap(a, b).tolist() == [True, False, False]
    pats, window = motifs.read_modisco(tmp_path / "chrombpnet" / "ds_a" / "contrib_scores" / "ds_a" / "modisco" /
                                       "modisco_counts_results.h5", "chrombpnet")  # fmt: skip
    assert window == 500 and [p.key for p in pats] == [f"pos.pattern_{i}" for i in range(4)] + ["neg.pattern_0"]
    assert pats[0].trim_bounds() == (9, 21) and len(pats[0].consensus()) == 12
