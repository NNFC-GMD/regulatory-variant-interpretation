"""workflows/cooperativity/: motif_names.py (the motif -> TF table from a compendium) and combine.py, ours and
--official (the official deepISA package's rules, by its own functions), on toy pair tables. CPU only."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

COOP = Path(__file__).resolve().parents[1] / "workflows" / "cooperativity"
sys.path.insert(0, str(COOP))

import motif_names as mn  # noqa: E402


def _compendium(root: Path, rows: list[tuple], min_score: float = 0.7) -> Path:
    """A 09.0 compendium's meta and json: rows of (label, posneg, cluster, annotation, score, seqlets)."""
    d = root / "tag5"
    d.mkdir(parents=True)
    lines = ["name\tnum_seqlets\tmodel\tposneg\tavg_dist_from_summit\tannotation_score0\tannotation_name0\tcluster_id"]
    for i, (label, posneg, cid, ann, score, n) in enumerate(rows):
        lines.append(f"{label}-{posneg}.pattern_{i}\t{n}\t{label}\t{posneg}\t0\t{score}\t{ann}\t{cid}")
    (d / "modisco_compendium_meta.tsv").write_text("\n".join(lines) + "\n")
    (d / "compendium.json").write_text(json.dumps({"settings": {"min_annotation_score": min_score}}))
    return d


def test_motif_names_names_clusters_as_the_comparison_does(tmp_path):
    d = _compendium(tmp_path, [
        ("cheri_ds_a", "pos", "0", "CTCF_1", 0.9, 100), ("cheri_ds_b", "pos", "0", "CTCF_2", 0.95, 80),
        ("cheri_ds_c", "pos", "0", "CTCF_1", 0.8, 70),  # CTCF_1 twice, CTCF_2 once: CTCF_1
        ("cheri_ds_a", "pos", "1", "ZNF9_0", 0.4, 300),  # below the compendium's 0.7: no annotation
        ("cheri_ds_a", "neg", "2", "TF4_0", 0.9, 40),
    ])  # fmt: skip
    out = tmp_path / "motifs.tsv"
    assert mn.main(["--compendium", str(d), "--out", str(out)]) == 0
    t = pd.read_csv(out, sep="\t", keep_default_na=False).set_index("motif")
    assert list(t.index) == ["pos_patterns.0", "pos_patterns.1", "neg_patterns.2"]
    assert t.loc["pos_patterns.0", "cluster_annotation"] == "CTCF_1" and t.loc["pos_patterns.0", "datasets"] == 3
    assert t.loc["pos_patterns.1", "cluster_annotation"] == ""
    assert t.loc["neg_patterns.2", "cluster_annotation"] == "TF4_0"
    with pytest.raises(FileNotFoundError, match="finished compendium"):
        mn.motif_table(tmp_path / "nothing")


# ---------------------------------------------------------------------------------------------------------------------
# combine.py


def _world(tmp_path: Path, n_peaks: int = 60, seed: int = 0) -> dict:
    """coop_isa.py's outputs for two datasets: TF pair A + B synergistic in ds_1 and redundant in ds_2, C + D mixed."""
    rng = np.random.default_rng(seed)
    root = tmp_path / "coop"
    for ds, sign in (("ds_1", 1.0), ("ds_2", -1.0)):
        sites, pairs = [], []
        for k in range(n_peaks):
            pid = f"peak_{k}"
            for j, (motif, tf) in enumerate((("pos_patterns.0", "A"), ("pos_patterns.1", "B"))):
                sites.append((pid, "chr1", 100 + 20 * j, 110 + 20 * j, motif, tf, 0.5 + rng.random(), 3.0))
            e_a, e_b = sites[-2][6], sites[-1][6]
            i = sign * (0.2 + 0.1 * rng.random())
            pairs.append((pid, "chr1", "pos_patterns.0", "pos_patterns.1", "A", "B", 20.0, 10, e_a, e_b,
                          e_a + e_b - i, i))  # fmt: skip
            e_c, e_d = 0.4 + rng.random(), 0.4 + rng.random()
            i2 = rng.normal(0, 0.2)
            gap = 200 if k % 3 == 0 else 30  # some pairs beyond 150 bp: the package's null
            pairs.append((pid, "chr1", "pos_patterns.2", "pos_patterns.3", "C", "D", gap + 10.0, gap, e_c, e_d,
                          e_c + e_d - i2, i2))  # fmt: skip
            sites += [(pid, "chr1", 300, 310, "pos_patterns.2", "C", e_c, 3.0),
                      (pid, "chr1", 310 + gap, 320 + gap, "pos_patterns.3", "D", e_d, 3.0)]  # fmt: skip
        d = root / ds
        d.mkdir(parents=True)
        pd.DataFrame(sites, columns=["peak_id", "chr", "start", "end", "motif", "tf", "e", "a"]).to_csv(
            d / "sites.tsv.gz", sep="\t", index=False
        )
        pd.DataFrame(pairs, columns=["peak_id", "chr", "motif_a", "motif_b", "tf_a", "tf_b", "distance", "gap", "e_a",
                                     "e_b", "isa_both", "i"]).to_csv(d / "pairs.tsv.gz", sep="\t", index=False)  # fmt: skip
    motifs = tmp_path / "motifs.tsv"
    motifs.write_text(
        "motif\tcluster_annotation\n" + "".join(f"pos_patterns.{n}\t{t}_1\n" for n, t in enumerate("ABCD"))
    )
    groups = tmp_path / "groups.tsv"
    groups.write_text("group\tdatasets\none\tds_1\ntwo\tds_2\n")
    comparisons = tmp_path / "comparisons.tsv"
    comparisons.write_text("name\tgroup_a\tgroup_b\none_two\tone\ttwo\n")
    return {"root": root, "motifs": motifs, "groups": groups, "comparisons": comparisons}


def _combine(w: dict, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(COOP / "combine.py"), "--root", str(w["root"]), "--motifs", str(w["motifs"]),
         "--groups", str(w["groups"]), "--comparisons", str(w["comparisons"]), "--n-boot", "50", "--min-n", "20",
         *extra],
        capture_output=True, text=True,
    )  # fmt: skip


def test_combine_ours_finds_the_switch(tmp_path):
    w = _world(tmp_path)
    r = _combine(w)
    assert r.returncode == 0, r.stderr
    ch = pd.read_csv(w["root"] / "combined" / "changes_pairs.tsv", sep="\t").set_index("pair")
    ab = ch.loc["A + B"]
    assert ab["synergy_a"] == 1.0 and ab["synergy_b"] == 0.0 and bool(ab["significant"])
    assert not (w["root"] / "official").exists()


def test_combine_official_is_the_package_on_our_tables(tmp_path):
    pytest.importorskip("deepISA.score.aggregate_isa")
    from deepISA.score import aggregate_isa as dag

    w = _world(tmp_path)
    r = _combine(w, "--official")
    assert r.returncode == 0, r.stderr
    off = w["root"] / "official"
    combi = pd.read_csv(off / "ds_1" / "motif_combi_isa.csv")
    # only the pairs make_pairs_for_region keeps: 1 <= edge gap <= 255
    assert combi["distance"].between(1, 255).all() and set(combi.columns) >= {"tf1", "tf2", "isa1_t0", "isa_both_t0"}
    # the package itself, called directly on fresh copies of the two inputs, gives the same tables
    direct = tmp_path / "direct"
    direct.mkdir()
    single = pd.read_csv(off / "ds_1" / "motif_single_isa.csv")
    single.to_csv(direct / "single.csv", index=False)
    combi.drop(columns=["interaction_t0"]).to_csv(direct / "combi.csv", index=False)
    dag.calc_interaction(str(direct / "combi.csv"), tracks=[0], isa_thresh_quantile=0.1,
                         single_isa_path=str(direct / "single.csv"), normalize=False)  # fmt: skip
    for level, name in (("tf_pair", "coop_tf_pair_t0.csv"), ("tf", "coop_tf_t0.csv")):
        want = dag.calc_coop_score(str(direct / "combi.csv"), str(direct / name), level=level, min_count=10,
                                   q_val_thresh=0.1)  # fmt: skip
        got = pd.read_csv(off / "ds_1" / name)
        pd.testing.assert_frame_equal(got, pd.read_csv(direct / name), check_exact=False, rtol=1e-6)
        assert len(want) == len(got)
    # A + B's interactions are all positive in ds_1 and negative in ds_2: coop +1 and -1, a change of -2
    ch = pd.read_csv(off / "changes_pairs.tsv", sep="\t").set_index("pair")
    ab = ch.loc["A + B"]
    assert ab["coop_a"] == pytest.approx(1.0) and ab["coop_b"] == pytest.approx(-1.0) and bool(ab["significant"])
    summary = json.loads((off / "summary.json").read_text())
    assert summary["settings"]["receptive_field"] == 255 and summary["ds_1"]["qualified"] > 0
