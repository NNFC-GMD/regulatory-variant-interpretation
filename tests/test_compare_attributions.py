"""compare_attributions.py, the comparison's level 2 (CPU only, synthetic datasets from compare_fixtures).

Each dataset is test_check_dataset's (W = 12 of L = 2114, so ChromBPNet's column of Cherimoya's column 0 is 1051;
24 peaks, 2 of them dropped by Cherimoya for an N) spread over three chromosomes, with ChromBPNet's scores and both
fold averages. The per-peak measures are checked against a direct computation with scipy, row by row, joined by
peak name; the held-out fold of chr<c> is fold c - 1 (compare_fixtures.write_folds_dir).
"""

from __future__ import annotations

import gzip
import json
import os
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
h5py = pytest.importorskip("h5py")
pytest.importorskip("hdf5plugin")
pytest.importorskip("scipy")
pytest.importorskip("yaml")

TESTS = Path(__file__).resolve().parent
sys.path[:0] = [
    str(TESTS),
    str(TESTS.parent / "workflows" / "cherimoya"),
    str(TESTS.parent / "workflows" / "cherimoya" / "compare"),
]

import compare_attributions as ca  # noqa: E402
from compare_fixtures import FOLDS, make_attribution_dataset, write_folds_dir  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

W, L = 12, 2114
C0 = L // 2 - W // 2


def run(tmp_path: Path, data: dict, *extra: str, capsys=None) -> tuple[int, str]:
    rc = ca.main(["--cheri-root", str(data["root"]), "--comparison-root", str(tmp_path / "comparison"),
                  "--folds-dir", str(write_folds_dir(tmp_path / "igvf" / "folds")), "--windows", "3,6",
                  "--top-k", "3", "--block-rows", "5", "--dataset", data["ds"], *extra])  # fmt: skip
    return rc, capsys.readouterr().out if capsys else ""


def result_line(out: str) -> str:
    lines = [line for line in out.splitlines() if line.startswith("RESULT: ")]
    assert lines, out
    return lines[-1][len("RESULT: ") :]


def table(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt") as fh:
        return pd.read_csv(fh, sep="\t", keep_default_na=False, na_values=["NA"])


def cheri_attr(data: dict, k: str) -> np.ndarray:
    with np.load(data["folds"][k]["params"]["attr_filename"]) as z:
        return z["arr_0"].astype(np.float64)


def expected(x: np.ndarray, y: np.ndarray, ohe: np.ndarray, half: int, k: int) -> dict:
    a, b = W // 2 - half, W // 2 + half
    px, py = (x * ohe).sum(1)[:, a:b], (y * ohe).sum(1)[:, a:b]
    out = {"pearson": [], "spearman": [], "cosine": [], "jaccard": []}
    for i in range(len(x)):
        out["pearson"].append(pearsonr(px[i], py[i])[0])
        out["spearman"].append(spearmanr(px[i], py[i])[0])
        hx, hy = x[i][:, a:b].ravel(), y[i][:, a:b].ravel()
        out["cosine"].append(hx @ hy / np.linalg.norm(hx) / np.linalg.norm(hy))
        tx, ty = set(np.argsort(-np.abs(px[i]))[:k]), set(np.argsort(-np.abs(py[i]))[:k])
        out["jaccard"].append(len(tx & ty) / len(tx | ty))
    out["sum_x"], out["l2_x"] = px.sum(1), np.sqrt((px**2).sum(1))
    out["sum_y"], out["l2_y"] = py.sum(1), np.sqrt((py**2).sum(1))
    return {key: np.asarray(v, dtype=np.float64) for key, v in out.items()}


@pytest.fixture
def data(tmp_path):
    return make_attribution_dataset(tmp_path, "ds_a", seed=0)


def test_measures_match_a_direct_computation(tmp_path, data, capsys):
    rc, out = run(tmp_path, data, capsys=capsys)
    assert rc == 0, out
    od = tmp_path / "comparison" / "attributions" / "ds_a"
    info = json.loads((od / "dataset.json").read_text())
    assert info["joined_rows"] == 22 and info["chrombpnet_only_rows"] == 2 and info["column_offset"] == C0
    assert info["windows"] == {"w6": [3, 9], "w12": [0, 12]}
    kept = np.flatnonzero(data["idx"])
    ohe = data["ohe"].astype(np.float64)
    names = [line.split("\t")[3] for line in data["peaks"].read_text().splitlines()]
    cbp = {k: data["cbp_shap"][k][kept][:, :, C0 : C0 + W].astype(np.float64) for k in FOLDS}
    che = {k: cheri_attr(data, k) for k in FOLDS}

    # (A) same fold, joined by name: ChromBPNet's rows are the peaks file's, Cherimoya's the kept ones
    for k in ("0", "3"):
        df = table(od / f"fold_{k}.per_peak.tsv.gz")
        assert df["peak"].tolist() == [names[i] for i in kept]
        assert df["chrom"].tolist() == [f"chr{1 + i % 3}" for i in kept]
        assert df["role"].tolist() == [{"chr1": "test" if k == "0" else "train", "chr2": "test" if k == "1" else
                                        ("valid" if k == "0" else "train"), "chr3": "test" if k == "2" else
                                        ("valid" if k == "1" else "train")}[f"chr{1 + i % 3}"] for i in kept]  # fmt: skip
        for half in (3, 6):
            e = expected(cbp[k], che[k], ohe, half, 3)
            w = f"w{2 * half}"
            for m in ("pearson", "spearman", "cosine", "jaccard"):
                assert np.allclose(df[f"{m}_{w}"], e[m], rtol=2e-4, atol=2e-5), (k, m, w)
            assert np.allclose(df[f"sum_chrombpnet_{w}"], e["sum_x"], rtol=2e-4, atol=1e-6)
            assert np.allclose(df[f"l2_cherimoya_{w}"], e["l2_y"], rtol=2e-4, atol=1e-6)
            gc = (data["ohe"][:, 1:3, 6 - half : 6 + half].sum((1, 2)) / (2 * half)).astype(np.float64)
            assert np.allclose(df[f"gc_{w}"], gc, atol=1e-4)  # written with 5 digits

    # (B) the averages: ChromBPNet's float16 mean, Cherimoya's 06.0 average
    with h5py.File(data["out"] / "contrib_scores" / "ds_a" / "ds_a_average_shaps.counts.h5") as fh:
        che_avg = fh["shap/seq"][()].astype(np.float64)
    cbp_avg = data["cbp_average"][kept][:, :, C0 : C0 + W].astype(np.float64)
    df = table(od / "average.per_peak.tsv.gz")
    e = expected(cbp_avg, che_avg, ohe, 6, 3)
    assert np.allclose(df["pearson_w12"], e["pearson"], rtol=2e-4, atol=2e-5)
    assert np.allclose(df["cosine_w12"], e["cosine"], rtol=2e-4, atol=2e-5)

    # (C) leave one out: fold k against the mean of the other four
    for m, maps in (("cherimoya", che), ("chrombpnet", cbp)):
        df = table(od / f"loo_{m}_2.per_peak.tsv.gz")
        others = np.mean([maps[j] for j in FOLDS if j != "2"], axis=0)
        e = expected(maps["2"], others, ohe, 3, 3)
        assert np.allclose(df["spearman_w6"], e["spearman"], rtol=2e-4, atol=2e-5), m
        assert np.allclose(df["sum_others_w6"], e["sum_y"], rtol=2e-4, atol=1e-6), m

    # (D) held out: chr1's peaks were held out by fold 0, chr2's by fold 1, chr3's by fold 2
    held = table(od / "heldout_cherimoya.per_peak.tsv.gz")
    assert held["heldout_fold"].tolist() == [int(1 + i % 3) - 1 for i in kept]
    loo = {k: table(od / f"loo_cherimoya_{k}.per_peak.tsv.gz") for k in FOLDS}
    for j, i in enumerate(kept):
        c = 1 + i % 3
        assert held["heldout_pearson_w12"][j] == pytest.approx(loo[str(c - 1)]["pearson_w12"][j], abs=1e-4)
        train = [k for k in FOLDS if k not in (str(c - 1), str((c - 2) % 5))]  # tested by c-1, validated by c-2
        assert held["n_train_folds"][j] == len(train)
        want = np.mean([loo[k]["pearson_w12"][j] for k in train])
        assert held["train_pearson_w12"][j] == pytest.approx(want, abs=1e-4)
    cross = table(od / "heldout_cross.per_peak.tsv.gz")
    folds_tables = {k: table(od / f"fold_{k}.per_peak.tsv.gz") for k in FOLDS}
    for j, i in enumerate(kept):
        k = str(i % 3)
        assert cross["cosine_w6"][j] == pytest.approx(folds_tables[k]["cosine_w6"][j], abs=1e-6)

    summary = pd.read_csv(od / "summary.tsv", sep="\t", keep_default_na=False, na_values=["NA"])
    row = summary[(summary["comparison"] == "fold_0") & (summary["window"] == "w12") & (summary["measure"] ==
                  "pearson") & (summary["stratum"] == "all")].iloc[0]  # fmt: skip
    assert row["n"] == 22 and row["median"] == pytest.approx(np.median(folds_tables["0"]["pearson_w12"]), abs=1e-4)
    roles = summary[(summary["comparison"] == "fold_0") & (summary["stratum"] == "role")]
    assert set(roles["level"]) == {"train", "valid", "test"}
    assert set(summary[summary["stratum"] == "signal_decile"]["level"].astype(str)) == {str(d) for d in range(1, 11)}
    assert result_line(out).startswith("22 peaks (2 ChromBPNet-only), w6 Pearson: fold 0")
    marker = json.loads((tmp_path / "comparison" / "attributions" / ".done" / "ds_a.json").read_text())
    assert marker["dataset"]["joined_rows"] == 22


def test_rerun_skips_and_a_new_attribution_runs_again(tmp_path, data, capsys):
    assert run(tmp_path, data, capsys=capsys)[0] == 0
    rc, out = run(tmp_path, data, capsys=capsys)
    assert rc == 0 and result_line(out) == "up to date"
    rc, out = run(tmp_path, data, "--top-k", "2", capsys=capsys)  # another setting
    assert rc == 0 and result_line(out).startswith("22 peaks")
    attr = Path(data["folds"]["4"]["params"]["attr_filename"])
    st = attr.stat()
    os.utime(attr, ns=(st.st_atime_ns, st.st_mtime_ns - 10**9))  # still older than the meta: DONE, but changed
    rc, out = run(tmp_path, data, "--top-k", "2", capsys=capsys)
    assert rc == 0 and result_line(out).startswith("22 peaks")


def test_max_rows_and_blocks(tmp_path, data, capsys):
    rc, out = run(tmp_path, data, "--max-rows", "7", "--block-rows", "3", capsys=capsys)
    assert rc == 0, out
    od = tmp_path / "comparison" / "attributions" / "ds_a"
    assert len(table(od / "fold_1.per_peak.tsv.gz")) == 7
    whole = tmp_path / "whole"
    rc = ca.main(["--cheri-root", str(data["root"]), "--out-dir", str(whole), "--folds-dir",
                  str(tmp_path / "igvf" / "folds"), "--windows", "3,6", "--top-k", "3", "--block-rows", "22",
                  "--dataset", "ds_a"])  # fmt: skip
    assert rc == 0
    a = table(od / "fold_1.per_peak.tsv.gz")
    b = table(whole / "ds_a" / "fold_1.per_peak.tsv.gz").iloc[:7]
    assert np.allclose(a["spearman_w12"], b["spearman_w12"])  # blocks do not change the numbers


@pytest.mark.parametrize("what", ["fold", "v2", "chrombpnet_average", "chrombpnet_fold"])
def test_missing_inputs_wait_and_write_nothing(tmp_path, data, capsys, what):
    if what == "fold":
        import attr_status

        attr_status.fold_files(data["folds"]["4"]["params"])["meta"].unlink()  # the fold is no longer DONE
    elif what == "v2":
        (data["out"] / ".done" / "v2").unlink()
    elif what == "chrombpnet_average":
        data["cbp_average_path"].unlink()
    else:
        (data["chrombpnet"] / "ds_a_all_fold_2" / "interpretation" / "interpretation.counts_scores.h5").unlink()
    rc, out = run(tmp_path, data, capsys=capsys)
    assert rc == 3, out
    assert result_line(out).startswith("waiting: ds_a: ")
    assert not (tmp_path / "comparison" / "attributions" / "ds_a").exists()


def test_v1_mismatch_fails(tmp_path, data, capsys):
    h5 = data["chrombpnet"] / "ds_a_all_fold_3" / "interpretation" / "interpretation.counts_scores.h5"
    with h5py.File(h5, "a") as fh:
        raw = fh["raw/seq"][()]
        raw[9, :, C0 + 2] = np.roll(raw[9, :, C0 + 2], 1)  # one base of a joined row
        fh["raw/seq"][...] = raw
    rc, out = run(tmp_path, data, capsys=capsys)
    assert rc == 1
    assert "V1 fails on 1 of rows" in result_line(out) and "fold 3" in result_line(out)
    assert not (tmp_path / "comparison" / "attributions" / ".done" / "ds_a.json").exists()


def test_beds_that_differ_between_folds_fail(tmp_path, data, capsys):
    bed = data["chrombpnet"] / "ds_a_all_fold_2" / "interpretation" / "interpretation.interpreted_regions.bed"
    lines = bed.read_text().splitlines()
    bed.write_text("\n".join(lines[:-1]) + "\n")
    rc, out = run(tmp_path, data, capsys=capsys)
    assert rc == 1 and "fold 2 bed differs" in result_line(out)


def test_float16_source_and_summary(tmp_path, data, capsys):
    other = make_attribution_dataset(tmp_path, "ds_b", seed=1)
    assert run(tmp_path, data, capsys=capsys)[0] == 0
    rc, out = run(tmp_path, data, "--cheri-source", "h5", "--tag", "float16", capsys=capsys)
    assert rc == 0, out
    od = tmp_path / "comparison" / "attributions"
    assert (od / ".done" / "ds_a.float16.json").is_file()
    a = table(od / "ds_a" / "fold_0.per_peak.tsv.gz")
    b = table(od / "ds_a" / "float16" / "fold_0.per_peak.tsv.gz")
    assert np.allclose(a["pearson_w12"], b["pearson_w12"], atol=5e-3)  # float16 changes little
    listing = tmp_path / "list.txt"
    listing.write_text("ds_a\nds_b\n")
    summarize = ["--comparison-root", str(tmp_path / "comparison"), "--summarize", "--datasets", str(listing),
                 "--n-boot", "500"]  # fmt: skip
    rc = ca.main(summarize)
    out = capsys.readouterr().out
    assert rc == 3 and "waiting for 1: ds_b" in result_line(out)  # written, not marked done
    assert (od / "statistics.tsv").is_file() and not (od / ".done" / "summary.json").exists()
    assert run(tmp_path, other, capsys=capsys)[0] == 0
    rc = ca.main(summarize)
    out = capsys.readouterr().out
    assert rc == 0, out
    assert result_line(out).startswith("2 of 2 datasets: pearson (w6) cross-model same fold")
    stats = pd.read_csv(od / "statistics.tsv", sep="\t")
    level = stats[(stats["test"] == "level") & (stats["window"] == "w12") & (stats["metric"] == "pearson")]
    assert set(level["quantity"]) >= {
        "cross_fold",
        "cross_average",
        "loo_chrombpnet",
        "loo_cherimoya",
        "heldout_cherimoya",
        "train_cherimoya",
        "cross_heldout",
        "cross_train",
    }
    paired = stats[(stats["test"] == "paired") & (stats["over"] == "folds")]
    assert set(paired["n"]) == {10}  # 2 datasets x 5 folds
    s = pd.read_csv(od / "summary.tsv", sep="\t", keep_default_na=False, na_values=["NA"])
    assert set(s["dataset"]) == {"ds_a", "ds_b"}
    # cross_fold of ds_a is the mean over folds of its per-fold medians
    want = np.mean(
        [
            s[
                (s["dataset"] == "ds_a")
                & (s["comparison"] == f"fold_{k}")
                & (s["window"] == "w12")
                & (s["measure"] == "pearson")
                & (s["stratum"] == "all")
            ]["median"].iloc[0]
            for k in FOLDS
        ]
    )
    q = ca.per_dataset_quantities(s, {"ds_a": list(FOLDS), "ds_b": list(FOLDS)})
    assert q[("w12", "pearson")]["cross_fold"]["ds_a"] == pytest.approx(want)
    f16 = pd.read_csv(od / "float16.tsv", sep="\t")
    assert set(f16["dataset"]) == {"ds_a"}
    assert f16[f16["measure"] != "jaccard"]["median_diff"].abs().max() < 5e-3  # Jaccard moves in steps
    for w in ("w6", "w12"):
        for name in ("agreement", "heldout"):
            assert (od / f"{name}_{w}.png").stat().st_size > 0
    rc = ca.main(summarize)
    assert rc == 0 and result_line(capsys.readouterr().out) == "up to date"


def test_helpers():
    assert ca.column_offset(2114, 1000) == 557
    assert ca.window_columns(1000, 250) == (250, 750) and ca.window_columns(1000, 500) == (0, 1000)
    with pytest.raises(ca.common.Failed):
        ca.window_columns(1000, 501)
    with pytest.raises(ca.common.Failed):
        ca.column_offset(2114, 1001)
    x = np.array([[1.0, 2.0, 2.0, 3.0], [5.0, 5.0, 5.0, 5.0]])
    assert ca.average_ranks(x).tolist() == [[1.0, 2.5, 2.5, 4.0], [2.5, 2.5, 2.5, 2.5]]
    r = ca.rowwise_pearson(x, np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]))
    assert r[0] == pytest.approx(pearsonr(x[0], [1, 2, 3, 4])[0]) and np.isnan(r[1])
    mask = ca.top_mask(np.array([[0.1, -3.0, 2.0, 0.5]]), 2)
    assert mask.tolist() == [[False, True, True, False]]
    d = ca.deciles_of(np.array([3.0, np.nan, 1.0, 2.0]))
    assert d[1] == 0 and d[2] < d[3] < d[0]
