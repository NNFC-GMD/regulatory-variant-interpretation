"""DeepISA's significance, fold check and trends (combine.py), the same sites under every model (paired.py,
matrix.py), and the coop_matrix stage's project settings, on toy tables. CPU only.

Not in test-pipeline yet (the preflight's required tests): it joins them once it has passed in a box."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("scipy")
yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
COOP = ROOT / "workflows" / "cooperativity"
sys.path[:0] = [str(COOP), str(ROOT / "workflows" / "pipeline")]

import combine as cb  # noqa: E402
import matrix as mx  # noqa: E402
import paired as pa  # noqa: E402
import project as pj  # noqa: E402

MOTIF = {tf: f"pos_patterns.{n}" for n, tf in enumerate("ABCDEF")}
OFFSETS = (-0.02, -0.01, 0.0, 0.01, 0.02)  # i_f<k> = i + offset: their mean is i, and each keeps i's sign


def _pairs(fracs: dict, n: int = 200, prefix: str = "peak", start0: int = 0, a: float = 3.0) -> pd.DataFrame:
    """One pair per peak and TF pair; fracs: TF pair ("A", "B") -> the share of its n pairs with i = +0.3 (the first
    ones; the others -0.3), so its synergy is exactly that share."""
    rows = []
    for k in range(n):
        for j, ((ta, tb), frac) in enumerate(fracs.items()):
            i = 0.3 if k < round(frac * n) else -0.3
            s = start0 + 1000 * k + 100 * j
            gap = 200 if ta == "C" else 20  # C + D beyond 150 bp: the official package's null (calc_coop_score)
            row = {"peak_id": f"{prefix}_{k}", "chr": "chr1", "motif_a": MOTIF[ta], "motif_b": MOTIF[tb], "tf_a": ta,
                   "tf_b": tb, "distance": gap + 10.0, "gap": gap, "e_a": 0.6, "e_b": 0.5, "isa_both": 1.1 - i,
                   "i": i, "start_a": s, "start_b": s + gap + 10, "a": a}  # fmt: skip
            row.update({f"i_f{f}": i + off for f, off in enumerate(OFFSETS)})
            rows.append(row)
    return pd.DataFrame(rows)


def _sites(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in pairs.itertuples():
        rows += [(r.peak_id, r.chr, r.start_a, r.start_a + 10, r.motif_a, r.tf_a, r.e_a, r.a),
                 (r.peak_id, r.chr, r.start_b, r.start_b + 10, r.motif_b, r.tf_b, r.e_b, r.a)]  # fmt: skip
    return pd.DataFrame(rows, columns=["peak_id", "chr", "start", "end", "motif", "tf", "e", "a"])


def _inputs(tmp_path: Path, datasets: list[str], comparisons: list[tuple]) -> dict:
    motifs = tmp_path / "motifs.tsv"
    motifs.write_text("motif\tcluster_annotation\n" + "".join(f"{m}\t{tf}_1\n" for tf, m in MOTIF.items()))
    groups = tmp_path / "groups.tsv"
    groups.write_text("group\tdatasets\n" + "".join(f"{d}\t{d}\n" for d in datasets))
    comps = tmp_path / "comparisons.tsv"
    comps.write_text("name\tgroup_a\tgroup_b\n" + "".join("\t".join(c) + "\n" for c in comparisons))
    series = tmp_path / "series.tsv"
    series.write_text("group\n" + "".join(f"{d}\n" for d in datasets))
    return {"motifs": motifs, "groups": groups, "comparisons": comps, "series": series}


RISING = (0.1, 0.35, 0.6, 0.85)  # A + B: synergy up by 0.25 per step
UP_DOWN = (0.1, 0.9, 0.9, 0.1)  # E + F: no slope, a clear change along the series


def _own_world(tmp_path: Path) -> dict:
    datasets = ["t0", "t1", "t2", "t3"]
    root = tmp_path / "coop"
    for k, ds in enumerate(datasets):
        pairs = _pairs({("A", "B"): RISING[k], ("C", "D"): 0.5, ("E", "F"): UP_DOWN[k]})
        (root / ds).mkdir(parents=True)
        pairs.to_csv(root / ds / "pairs.tsv.gz", sep="\t", index=False)
        _sites(pairs).to_csv(root / ds / "sites.tsv.gz", sep="\t", index=False)
    return {"root": root, **_inputs(tmp_path, datasets, [("t0_t3", "t0", "t3")])}


def test_boot_p_and_bh():
    reps = np.random.default_rng(0).normal(0, 1, (400, 3))
    p = cb.boot_p([0.0, 1.96, -10.0], reps)
    assert p[0] == pytest.approx(1.0) and 0.02 < p[1] < 0.1 and p[2] < 1e-10
    flat = np.zeros((10, 3))
    assert list(cb.boot_p([0.0, 0.5, np.nan], flat)[:2]) == [1.0, 0.0] and np.isnan(cb.boot_p([np.nan], flat[:, :1]))
    multitest = pytest.importorskip("statsmodels.stats.multitest")
    raw = np.array([0.001, 0.01, 0.02, 0.04, 0.3, np.nan])
    want = multitest.multipletests(raw[:-1], method="fdr_bh")[1]
    got = cb.bh(raw)
    assert np.allclose(got[:-1], want) and np.isnan(got[-1])


def test_combine_trends_fdr_and_folds(tmp_path):
    pytest.importorskip("deepISA.score.aggregate_isa")
    w = _own_world(tmp_path)
    r = subprocess.run(
        [sys.executable, str(COOP / "combine.py"), "--root", str(w["root"]), "--motifs", str(w["motifs"]),
         "--groups", str(w["groups"]), "--comparisons", str(w["comparisons"]), "--series", str(w["series"]),
         "--n-boot", "100", "--min-n", "20", "--official"],
        capture_output=True, text=True,
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    out = w["root"] / "combined"
    tr = pd.read_csv(out / "trend_pairs.tsv", sep="\t").set_index("pair")
    ab, cd, ef = tr.loc["A + B"], tr.loc["C + D"], tr.loc["E + F"]
    assert ab["slope"] == pytest.approx(0.25) and ab["significant"] and ab["robust"] and ab["folds_agree"] == 5
    assert (ab["peak"], ab["trough"]) == ("t3", "t0") and ab["synergy_t1"] == pytest.approx(0.35)
    assert cd["slope_p"] > 0.5 and not cd["significant"] and not cd["het_significant"]
    assert abs(ef["slope"]) < 1e-9 and not ef["significant"] and ef["het_significant"]
    ch = pd.read_csv(out / "changes_pairs.tsv", sep="\t").set_index("pair")
    assert ch.loc["A + B", "delta"] == pytest.approx(0.75) and ch.loc["A + B", "robust"]
    assert ch.loc["A + B", "ci_excludes_0"] and ch.loc["A + B", "q"] < 0.05
    assert ch.loc["C + D", "p"] == pytest.approx(1.0) and not ch.loc["C + D", "significant"]
    by_ds = pd.read_csv(out / "pairs_by_dataset.tsv", sep="\t")
    assert {"synergy_fold_min", "synergy_fold_max"} <= set(by_ds.columns)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["trend_pairs"]["significant"] >= 1 and summary["changes_pairs"]["t0_t3"]["robust"] >= 1
    assert summary["t0"]["median_a"] == pytest.approx(3.0)
    off = pd.read_csv(w["root"] / "official" / "trend_pairs.tsv", sep="\t").set_index("pair")
    assert off.loc["A + B", "slope"] == pytest.approx(0.5) and off.loc["A + B", "significant"]  # coop = 2 synergy - 1


def _matrix_world(tmp_path: Path, n: int = 120) -> dict:
    """Three datasets' sites, each under the three datasets' models. A + B's i rises with the model, on the same
    pairs; C + D's does not change. m0's last 10 windows are closed under m2's models. One pair is in two site sets."""
    datasets = ["m0", "m1", "m2"]
    frac = {"m0": 0.2, "m1": 0.5, "m2": 0.8}
    root = tmp_path / "matrix"
    for x_k, x in enumerate(datasets):
        for m in datasets:
            pairs = _pairs({("A", "B"): frac[m], ("C", "D"): 0.5}, n=n, prefix=x, start0=10**6 * x_k)
            if x == "m1":  # the same genomic pair as m0's first A + B pair
                pairs.loc[0, ["start_a", "start_b"]] = [0, 30]
            if x == "m0" and m == "m2":
                pairs.loc[pairs["peak_id"].isin([f"m0_{k}" for k in range(n - 10, n)]), "a"] = 0.5
            (root / x / m).mkdir(parents=True)
            pairs.to_csv(root / x / m / "pairs.tsv.gz", sep="\t", index=False)
    comparisons = [("m0_m2", "m0", "m2"), ("m0_m1", "m0", "m1")]
    return {"root": root, "n": n, **_inputs(tmp_path, datasets, comparisons)}


def _paired(w: dict, *extra: str) -> int:
    return pa.main(["--matrix", str(w["root"]), "--motifs", str(w["motifs"]), "--groups", str(w["groups"]),
                    "--comparisons", str(w["comparisons"]), "--series", str(w["series"]), "--n-boot", "100",
                    "--min-n", "20", *extra])  # fmt: skip


def test_paired_same_sites_under_every_model(tmp_path):
    w = _matrix_world(tmp_path)
    assert _paired(w) == 0
    out = w["root"] / "combined"
    summary = json.loads((out / "summary.json").read_text())
    assert summary["pairs"] == 3 * 2 * w["n"] - 1  # the pair in two site sets is kept once
    assert summary["open"]["m2"]["open"] == summary["open"]["m2"]["windows"] - 10
    assert summary["comparisons"]["m0_m2"]["pairs_open"] == summary["pairs"] - 2 * 10
    assert summary["comparisons"]["m0_m1"]["pairs_open"] == summary["pairs"]  # m2's closed windows do not matter
    ch = pd.read_csv(out / "changes_pairs.tsv", sep="\t")
    assert not [c for c in ch.columns if c.endswith(("_x", "_y", "_di"))]  # one mean_i_a, mean_i_b
    ab = ch[(ch["comparison"] == "m0_m2") & (ch["pair"] == "A + B")].iloc[0]
    cd = ch[(ch["comparison"] == "m0_m2") & (ch["pair"] == "C + D")].iloc[0]
    assert ab["delta_i"] > 0.3 and ab["delta_i_significant"] and ab["delta_i_robust"] and ab["delta_i_folds_agree"] == 5
    assert ab["delta"] > 0.5 and ab["significant"]  # the paired synergy change
    assert cd["delta_i"] == pytest.approx(0.0) and not cd["delta_i_significant"] and not cd["significant"]
    tr = pd.read_csv(out / "trend_pairs.tsv", sep="\t").set_index("pair")
    assert tr.loc["A + B", "slope"] > 0.2 and tr.loc["A + B", "significant"] and tr.loc["A + B", "slope_i_significant"]
    assert not tr.loc["C + D", "slope_i_significant"]
    assert (out / "trend_tfs.tsv").is_file() and (out / "open.tsv").is_file()


def test_paired_refuses_rows_that_do_not_line_up(tmp_path):
    w = _matrix_world(tmp_path, n=30)
    f = w["root"] / "m1" / "m2" / "pairs.tsv.gz"
    df = pd.read_csv(f, sep="\t")
    df.loc[3, "start_a"] += 1
    df.to_csv(f, sep="\t", index=False)
    with pytest.raises(ValueError, match="not those under"):
        _paired(w)


def test_matrix_args_and_cpu_blocks(tmp_path):
    root = tmp_path / "coop"
    (root / "args").mkdir(parents=True)
    for d in ("x", "y"):
        (root / "args" / f"{d}.args").write_text(
            f"--dataset\n{d}\n--models\n/m/{d}/fold_{{fold}}.torch\n--finemo-dir\n/f/{d}\n--out-dir\n{root}\n"
            "--folds\n0\n1\n"
        )
    runs = mx.write_matrix_args(root, ["x", "y"])
    assert [(a, b) for a, b, _ in runs] == [("x", "x"), ("x", "y"), ("y", "x"), ("y", "y")]
    xy = mx.read_args(root / "matrix" / "args" / "x__y.args")
    assert mx.option(xy, "--finemo-dir") == "/f/x" and mx.option(xy, "--models") == "/m/y/fold_{fold}.torch"
    assert mx.option(xy, "--dataset") == "y" and mx.option(xy, "--out-dir") == str(root / "matrix" / "x")
    assert xy[-3:] == ["--folds", "0", "1"]
    blocks = mx.cpu_blocks(3, 1)
    assert len(blocks) == 3 and all(b for b in blocks)


def _project(tmp_path: Path, **coop) -> Path:
    doc = {"project": "toy", "root": str(tmp_path / "root"), "prefix": "tx_", "compendium": "tx3",
           "samples": [{"stem": s, "fragments": f"/data/{s}.fragments.tsv.gz"} for s in ("a", "b", "c")],
           "cooperativity": {"comparisons": [["a_c", "a", "c"]], **coop}}  # fmt: skip
    path = tmp_path / "project.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path


def test_project_series_and_the_optional_stage(tmp_path):
    p = pj.load(_project(tmp_path))
    assert "coop_matrix" not in p["stages"] and p["cooperativity"]["series"] == []
    p = pj.load(_project(tmp_path, series=["a", "b", "c"]))
    assert p["cooperativity"]["series"] == ["a", "b", "c"]
    written = pj.write_coop_inputs(p)
    coop = pj.paths(p)["coop"]
    assert coop / "series.tsv" in written and (coop / "series.tsv").read_text() == "group\na\nb\nc\n"
    for bad in (["a", "b"], ["a", "b", "zz"], ["a", "a", "b"]):
        with pytest.raises(pj.ProjectError, match="series"):
            pj.load(_project(tmp_path, series=bad))
    path = _project(tmp_path, series=["a", "b", "c"])
    raw = yaml.safe_load(path.read_text())
    path.write_text(yaml.safe_dump({**raw, "stages": ["coop_matrix", "prep"]}))
    p = pj.load(path)
    assert p["stages"] == ["prep", "coop_matrix"]
    summary = pj.paths(p)["coop_matrix"] / "combined" / "summary.json"
    assert pj.check(p, "coop_matrix") == [str(summary)]
    summary.parent.mkdir(parents=True)
    summary.write_text("{}")
    (coop / "combined").mkdir(parents=True)
    (coop / "combined" / "summary.json").write_text("{}")
    assert "no trends" in pj.check(p, "coop_matrix")[0]  # a series needs combine.py's trends too
    (coop / "combined" / "summary.json").write_text('{"trend_pairs": {"tested": 1}}')
    assert pj.check(p, "coop_matrix") == []
