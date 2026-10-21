"""compare_performance.py, the comparison's level 1 (CPU only, synthetic folds from compare_fixtures).

The folds' compare.tsv files have known Cherimoya-minus-ChromBPNet differences, and their compare_counts.npz files
known residuals (ChromBPNet predicts observed + 1, Cherimoya observed - 0.5), so the statistics, the decile table and
the done marker can be checked exactly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("scipy")
pytest.importorskip("matplotlib")

TESTS = Path(__file__).resolve().parent
sys.path[:0] = [
    str(TESTS),
    *(str(TESTS.parent / "workflows" / d) for d in ("train", "attribution", "downstream")),
    str(TESTS.parent / "workflows" / "comparison"),
]

import compare_performance as cp  # noqa: E402
import summarize  # noqa: E402
from compare_fixtures import FOLDS, MEASURES, N_PEAKS, diff_of, make_performance_root  # noqa: E402


def run(data: dict, tmp_path: Path, *extra: str, capsys=None) -> tuple[int, str]:
    rc = cp.main(["--cheri-root", str(data["root"]), "--comparison-root", str(tmp_path / "comparison"),
                  "--datasets", str(data["list"]), "--n-boot", "2000", *extra])  # fmt: skip
    out = capsys.readouterr().out if capsys else ""
    return rc, out


def result_line(out: str) -> str:
    lines = [line for line in out.splitlines() if line.startswith("RESULT: ")]
    assert lines, out
    return lines[-1][len("RESULT: ") :]


def test_statistics_and_tables(tmp_path, capsys):
    data = make_performance_root(tmp_path)
    rc, out = run(data, tmp_path, capsys=capsys)
    assert rc == 0, out
    perf = tmp_path / "comparison" / "performance"
    paired = pd.read_csv(perf / "comparison_paired.tsv", sep="\t")
    assert sorted(paired["dataset"].unique()) == data["datasets"]  # the dataset not listed is left out
    assert len(paired) == len(data["datasets"]) * len(FOLDS)
    stats = pd.read_csv(perf / "statistics.tsv", sep="\t").set_index("measure")
    assert list(stats.index) == list(MEASURES)
    from scipy.stats import wilcoxon

    for m in MEASURES:
        diffs = np.array([diff_of(i, k) for i in range(len(data["datasets"])) for k in FOLDS])
        means = np.array([np.mean([diff_of(i, k) for k in FOLDS]) for i in range(len(data["datasets"]))])
        s = stats.loc[m]
        assert s["n_folds"] == 15 and s["n_datasets"] == 3
        assert s["median_fold_diff"] == pytest.approx(np.median(diffs), abs=2e-6)
        assert s["median_dataset_diff"] == pytest.approx(np.median(means), abs=2e-6)
        assert s["ci95_lo"] <= s["median_dataset_diff"] <= s["ci95_hi"]
        assert s["wilcoxon_folds_p"] == pytest.approx(wilcoxon(diffs).pvalue, rel=1e-3)
        assert s["wilcoxon_datasets_p"] == pytest.approx(wilcoxon(means).pvalue, rel=1e-3)
        higher = summarize.HIGHER_IS_BETTER[m]
        # 14 folds have Cherimoya - ChromBPNet > 0, one < 0
        assert s["cherimoya_better_folds"] == (14 if higher else 1)
        assert s["chrombpnet_better_folds"] == (1 if higher else 14)
        assert s["cherimoya_better_datasets"] == (3 if higher else 0)
    text = (perf / "statistics.txt").read_text()
    assert "counts_pearson" in text and "bootstrap" in text
    note = result_line(out)
    assert note.startswith("15 folds, 3 datasets: counts Pearson")
    for m in MEASURES:
        assert (perf / f"scatter_{m}.png").stat().st_size > 0
        assert (perf / f"scatter_{m}.pdf").stat().st_size > 0
    for name in ("differences", "residuals_by_decile"):
        assert (perf / f"{name}.png").stat().st_size > 0 and (perf / f"{name}.pdf").stat().st_size > 0


def test_residuals_by_decile_exact(tmp_path, capsys):
    data = make_performance_root(tmp_path)
    assert run(data, tmp_path, capsys=capsys)[0] == 0
    perf = tmp_path / "comparison" / "performance"
    res = pd.read_csv(perf / "residuals_by_decile.tsv", sep="\t")
    assert len(res) == 3 * 10 * 2
    assert set(res["n_peaks"]) == {len(FOLDS) * N_PEAKS // 10}  # every test peak of the 5 folds, 10 equal bins
    cbp, che = res[res["model"] == "chrombpnet"], res[res["model"] == "cherimoya"]
    assert np.allclose(cbp["mean_residual"], 1.0) and np.allclose(cbp["rmse"], 1.0)
    assert np.allclose(che["mean_residual"], -0.5) and np.allclose(che["rmse"], 0.5)
    first = res[(res["dataset"] == "ds_a") & (res["model"] == "cherimoya")].sort_values("decile")
    assert (np.diff(first["observed_mean"]) > 0).all()  # deciles in increasing order of observed counts
    summary = pd.read_csv(perf / "residuals_by_decile_summary.tsv", sep="\t")
    assert len(summary) == 10
    assert np.allclose(summary["rmse_diff_median"], -0.5)
    assert (summary["cherimoya_lower_rmse_datasets"] == 3).all()


def test_deciles_equal_count_ties_in_order():
    x = np.array([5.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 1.0, 9.0, 0.0])
    d = cp.deciles(x, n=5)
    assert np.bincount(d)[1:].tolist() == [2, 2, 2, 2, 2]
    # ties keep their input order: the four 1.0s are rows 1, 2, 3, 7; after the 0.0 come rows 1 then 2 (bin 1 is
    # full after row 9 and row 1), then 3 and 7
    assert d[9] == 1 and d[1] == 1 and d[2] == 2 and d[3] == 2 and d[7] == 3
    assert d[8] == 5


def test_rerun_is_up_to_date_and_a_changed_input_runs_again(tmp_path, capsys):
    data = make_performance_root(tmp_path)
    assert run(data, tmp_path, capsys=capsys)[0] == 0
    perf = tmp_path / "comparison" / "performance"
    stamp = (perf / "statistics.tsv").stat().st_mtime_ns
    rc, out = run(data, tmp_path, capsys=capsys)
    assert rc == 0 and result_line(out) == "up to date"
    assert (perf / "statistics.tsv").stat().st_mtime_ns == stamp
    tsv = data["root"] / "ds_b" / "fold_3" / "compare.tsv"
    os.utime(tsv, ns=(tsv.stat().st_atime_ns, tsv.stat().st_mtime_ns + 10**9))
    rc, out = run(data, tmp_path, capsys=capsys)
    assert rc == 0 and result_line(out).startswith("15 folds")
    rc, out = run(data, tmp_path, "--force", capsys=capsys)
    assert rc == 0 and result_line(out).startswith("15 folds")


def test_a_missing_fold_waits_and_writes_nothing(tmp_path, capsys):
    data = make_performance_root(tmp_path)
    (data["root"] / "ds_c" / "fold_2" / "compare_counts.npz").unlink()
    rc, out = run(data, tmp_path, capsys=capsys)
    assert rc == 3
    assert result_line(out).startswith("waiting: 1 fold(s)")
    assert not (tmp_path / "comparison" / "performance").exists()
    rc, out = run(data, tmp_path, "--allow-missing", capsys=capsys)
    assert rc == 0
    assert result_line(out).startswith("14 folds") and "1 fold(s) missing" in result_line(out)


def test_a_broken_table_fails(tmp_path, capsys):
    data = make_performance_root(tmp_path)
    (data["root"] / "ds_a" / "fold_1" / "compare.tsv").write_text("dataset\tfold\tmodel\n")
    rc, out = run(data, tmp_path, capsys=capsys)
    assert rc == 1


def test_summarize_keeps_only_the_listed_datasets(tmp_path):
    data = make_performance_root(tmp_path)
    out = tmp_path / "s"
    assert summarize.main(["--root", str(data["root"]), "--out-dir", str(out), "--datasets", str(data["list"])]) == 0
    rows = pd.read_csv(out / "comparison.tsv", sep="\t")
    assert sorted(rows["dataset"].unique()) == data["datasets"]
    assert summarize.main(["--root", str(data["root"]), "--out-dir", str(tmp_path / "all")]) == 0
    assert "ds_other" in set(pd.read_csv(tmp_path / "all" / "comparison.tsv", sep="\t")["dataset"])


def test_short_names_and_styles():
    import common

    names = common.short_names(["amsc_x_d0", "amsc_x_d14", "amsc_y_d0"])
    assert names == {"amsc_x_d0": "x_d0", "amsc_x_d14": "x_d14", "amsc_y_d0": "y_d0"}
    assert common.short_names(["alone"]) == {"alone": "alone"}
    styles = common.dataset_styles([f"d{i:02d}" for i in range(10)])
    assert styles["d00"] == ("#E69F00", "o") and styles["d08"] == ("#E69F00", "s")  # plot_comparison's order
