"""compare_finemo.py, the comparison's level 5 (CPU only, synthetic Fi-NeMo hits).

The dataset is compare_fixtures.make_modisco_dataset's (40 peaks; Cherimoya's beds lack peaks 5 and 17, so a peak's
row, and Fi-NeMo's peak_id, differ between the models). Hits of the joint vocabulary:
  pos_patterns.0  both models at the summits of peaks 0-9 (ChromBPNet's peak 5 only it has), coefficients i and 2i
  pos_patterns.1  both at peaks 10-19, Cherimoya's 3 bp off: matched at 5 and 10 bp, not at 0
  pos_patterns.2  both at peaks 20-24, on opposite strands: never matched
  pos_patterns.3  Cherimoya alone, at peaks 30-39
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("scipy")
pytest.importorskip("yaml")

TESTS = Path(__file__).resolve().parent
sys.path[:0] = [
    str(TESTS),
    str(TESTS.parent / "workflows" / "cherimoya"),
    str(TESTS.parent / "workflows" / "cherimoya" / "compare"),
]

import compare_finemo as cf  # noqa: E402
from compare_fixtures import DROPPED3, N_PEAKS3, make_modisco_dataset, motif_db, peaks_bed, write_hits  # noqa: E402

KEPT = [i for i in range(N_PEAKS3) if i not in DROPPED3]
ROW = {p: j for j, p in enumerate(KEPT)}  # Cherimoya's bed row of a peak


def hit_lists(names=("pos_patterns.0", "pos_patterns.1", "pos_patterns.2", "pos_patterns.3")) -> dict:
    m0, m1, m2, m3 = names
    cbp = [(i, m0, "+", 0, float(i)) for i in range(10)] + [(i, m1, "+", 0, 1.0) for i in range(10, 20)]
    cbp += [(i, m2, "+", 0, 1.0) for i in range(20, 25)]
    che = [(ROW[i], m0, "+", 0, 2.0 * i) for i in range(10) if i in ROW]
    che += [(ROW[i], m1, "+", 3, 1.0) for i in range(10, 20) if i in ROW]
    che += [(ROW[i], m2, "-", 0, 1.0) for i in range(20, 25)] + [(ROW[i], m3, "+", 0, 1.0) for i in range(30, 40)]
    return {"chrombpnet": cbp, "cherimoya": che}


@pytest.fixture
def world(tmp_path):
    _, db = motif_db(tmp_path)
    data = make_modisco_dataset(tmp_path, "ds_a", db)
    lines = peaks_bed(N_PEAKS3)
    beds = {"chrombpnet": lines, "cherimoya": [lines[i] for i in KEPT]}
    dirs = {"chrombpnet": data["cbp_out"] / "finemo_joint20" / "ds_a_all",
            "cherimoya": data["cheri_out"] / "finemo_joint20" / "ds_a_all"}  # fmt: skip
    hits = hit_lists()
    for m in ("chrombpnet", "cherimoya"):
        write_hits(dirs[m], beds[m], hits[m])
    return {"data": data, "dirs": dirs, "beds": beds}


def run(tmp_path: Path, *args: str, capsys=None) -> tuple[int, str]:
    rc = cf.main(["--cheri-root", str(tmp_path / "cheri"), "--comparison-root", str(tmp_path / "comparison"),
                  "--n-boot", "300", *args])  # fmt: skip
    return rc, capsys.readouterr().out if capsys else ""


def result_line(out: str) -> str:
    lines = [line for line in out.splitlines() if line.startswith("RESULT: ")]
    assert lines, out
    return lines[-1][len("RESULT: ") :]


def test_hits_matched_per_motif_and_peak(tmp_path, world, capsys):
    rc, out = run(tmp_path, "--dataset", "ds_a", capsys=capsys)
    assert rc == 0, out
    od = tmp_path / "comparison" / "finemo" / "ds_a"
    t = json.loads((od / "hits.json").read_text())
    assert t["shared_peaks"] == 38 and t["hits"] == {"chrombpnet": 23, "cherimoya": 33}
    assert t["hits_all_peaks"]["chrombpnet"] == 25  # peaks 5 and 17 are ChromBPNet's alone
    assert t["recall_5bp"] == pytest.approx(18 / 23) and t["precision_5bp"] == pytest.approx(18 / 33)
    assert t["recall_0bp"] == pytest.approx(9 / 23) and t["precision_10bp"] == pytest.approx(18 / 33)
    motifs = pd.read_csv(od / "motifs.tsv", sep="\t").set_index("motif")
    assert motifs.loc["pos_patterns.0", "recall_0bp"] == 1.0 and motifs.loc["pos_patterns.1", "recall_0bp"] == 0.0
    assert motifs.loc["pos_patterns.1", "recall_5bp"] == 1.0
    assert motifs.loc["pos_patterns.2", "recall_10bp"] == 0.0  # the other strand never matches
    assert motifs.loc["pos_patterns.3", "precision_5bp"] == 0.0 and motifs.loc["pos_patterns.3", "hits_chrombpnet"] == 0
    assert motifs.loc["pos_patterns.0", "coefficient_spearman"] == pytest.approx(1.0)
    assert motifs.loc["pos_patterns.0", "per10k_chrombpnet"] == pytest.approx(1e4 * 9 / 38)
    assert motifs.loc["pos_patterns.2", "peak_jaccard"] == 1.0  # the same peaks, whatever the strand
    peaks = pd.read_csv(od / "peaks.tsv.gz", sep="\t").set_index("peak")
    assert peaks.loc["peak_0", "count_correlation"] == pytest.approx(1.0)
    assert np.isnan(peaks.loc["peak_30", "count_correlation"])  # ChromBPNet has no hit there
    assert t["median_count_correlation"] == pytest.approx(1.0)
    report = pd.read_csv(od / "motif_report.tsv", sep="\t")
    assert {"num_hits_total_chrombpnet", "num_hits_total_cherimoya"} <= set(report.columns)
    assert result_line(out).startswith("23 ChromBPNet and 33 Cherimoya hits on 38 shared peaks; at 5 bp Cherimoya's "
                                       "precision 0.55, recall 0.78")  # fmt: skip
    rc, out = run(tmp_path, "--dataset", "ds_a", capsys=capsys)
    assert rc == 0 and result_line(out) == "up to date"


def test_joint_cluster_annotations_join(tmp_path, world, capsys):
    comp = tmp_path / "comparison" / "compendium"
    comp.mkdir(parents=True)
    comp.joinpath("joint_clusters.tsv").write_text("cluster_id\tclass\tannotation\tbias_share\n0\tshared\tTF0_0\t0\n"
                                                   "3\tcherimoya_only\t\t1\n")  # fmt: skip
    assert run(tmp_path, "--dataset", "ds_a", capsys=capsys)[0] == 0
    motifs = pd.read_csv(tmp_path / "comparison" / "finemo" / "ds_a" / "motifs.tsv", sep="\t").set_index("motif")
    assert motifs.loc["pos_patterns.0", "cluster_class"] == "shared"
    assert motifs.loc["pos_patterns.3", "cluster_bias_share"] == 1.0


def test_wrong_peaks_or_compendia_fail(tmp_path, world, capsys):
    hits = world["dirs"]["cherimoya"] / "hits.tsv"
    text = hits.read_text().replace("\tpeak_6\t", "\tpeak_7\t", 1)
    hits.write_text(text)
    rc, out = run(tmp_path, "--dataset", "ds_a", capsys=capsys)
    assert rc == 1 and "name another peak than their row" in result_line(out)
    write_hits(world["dirs"]["cherimoya"], world["beds"]["cherimoya"], hit_lists()["cherimoya"], md5="other")
    rc, out = run(tmp_path, "--dataset", "ds_a", capsys=capsys)
    assert rc == 1 and "different compendia" in result_line(out)


def test_missing_hits_wait(tmp_path, world, capsys):
    (world["dirs"]["chrombpnet"] / "hits.bed.gz").unlink()
    rc, out = run(tmp_path, "--dataset", "ds_a", capsys=capsys)
    assert rc == 3 and result_line(out).startswith("waiting: ds_a: chrombpnet")
    assert not (tmp_path / "comparison" / "finemo" / "ds_a").exists()


def test_production_hits_through_level_4_matches(tmp_path, world, capsys):
    data = world["data"]
    cbp = hit_lists(("pos_patterns.0", "pos_patterns.1", "pos_patterns.2", "pos_patterns.3"))["chrombpnet"]
    che = hit_lists(("pos_patterns.7", "pos_patterns.8", "pos_patterns.9", "pos_patterns.0"))["cherimoya"]
    write_hits(data["cbp_out"] / "finemo_cbp10" / "ds_a_all", world["beds"]["chrombpnet"], cbp, md5="m1")
    write_hits(data["cheri_out"] / "finemo_cheri39" / "ds_a_all", world["beds"]["cherimoya"], che, md5="m2")
    rc, out = run(tmp_path, "--dataset", "ds_a", "--production", capsys=capsys)
    assert rc == 3 and "matches_cbp10_vs_cheri39.tsv" in result_line(out)
    comp = tmp_path / "comparison" / "compendium"
    comp.mkdir(parents=True)
    # ChromBPNet's 0 and 1 pair with Cherimoya's 7 and 8; Cherimoya's 0 is another motif than ChromBPNet's 0
    comp.joinpath("matches_cbp10_vs_cheri39.tsv").write_text("cluster\tbest\tp\tq\tmatch\treciprocal\n"
                                                             "0\t7\t1e-9\t1e-7\tTrue\tTrue\n1\t8\t1e-9\t1e-7\tTrue\tTrue\n"
                                                             "2\t9\t0.5\t0.9\tFalse\tFalse\n")  # fmt: skip
    rc, out = run(tmp_path, "--dataset", "ds_a", "--production", capsys=capsys)
    assert rc == 0, out
    t = json.loads((tmp_path / "comparison" / "finemo" / "ds_a" / "production" / "hits.json").read_text())
    assert t["recall_5bp"] == pytest.approx(18 / 23) and t["precision_5bp"] == pytest.approx(18 / 33)
    motifs = pd.read_csv(tmp_path / "comparison" / "finemo" / "ds_a" / "production" / "motifs.tsv", sep="\t")
    assert "pair:0|7" in set(motifs["motif"]) and "unpaired:cherimoya:pos_patterns.0" in set(motifs["motif"])


def test_summary(tmp_path, world, capsys):
    listing = tmp_path / "list.txt"
    listing.write_text("ds_a\nds_b\n")
    assert run(tmp_path, "--dataset", "ds_a", capsys=capsys)[0] == 0
    rc, out = run(tmp_path, "--summarize", "--datasets", str(listing), capsys=capsys)
    assert rc == 3 and "not yet: ds_b" in result_line(out)
    od = tmp_path / "comparison" / "finemo"
    summary = pd.read_csv(od / "summary.tsv", sep="\t")
    assert list(summary["dataset"]) == ["ds_a"] and summary["recall_5bp"].iloc[0] == pytest.approx(18 / 23)
    stats = pd.read_csv(od / "statistics.tsv", sep="\t").set_index("quantity")
    assert stats.loc["precision_5bp", "median"] == pytest.approx(18 / 33)
    assert (od / "agreement.png").stat().st_size > 0


def test_match_function():
    a = pd.DataFrame({"chr": ["chr1"] * 4, "motif": ["m", "m", "m", "n"], "strand": ["+", "+", "-", "+"],
                      "start": [100, 200, 300, 100], "end": [110, 210, 310, 110]})  # fmt: skip
    b = pd.DataFrame({"chr": ["chr1"] * 3, "motif": ["m", "m", "n"], "strand": ["+", "-", "+"],
                      "start": [104, 296, 120], "end": [114, 306, 130]})  # fmt: skip
    hit, partner = cf.match(a, b, 5)
    assert hit.tolist() == [True, False, True, False] and partner.tolist() == [0, -1, 1, -1]
    hit, _ = cf.match(a, b, 3)
    assert hit.tolist() == [False, False, False, False]
    hit, partner = cf.match(a, b.iloc[:0], 5)
    assert hit.tolist() == [False] * 4 and partner.tolist() == [-1] * 4
