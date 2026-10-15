"""scores.py's count scores are kundajelab/variant-scorer's, and compare.py's statistics do what they say.

The references are line-by-line transcriptions of VS (main @0e1e341): the strand average of fetch_variant_predictions
(src/utils/helpers.py:169-170, 177-178, 195-197), logFC (:225), the allele quantiles (:212-213), get_pvals
(:319-335) and the fold summary (src/variant_summary_across_folds.py:8-9, 37-47). All of it runs on toy arrays.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflows" / "variant_scoring"))

import compare as cmp  # noqa: E402
import scores as sc  # noqa: E402

# --- VS, transcribed -------------------------------------------------------------------------------------------


def vs_strand_average(fwd_logcounts, rev_logcounts):
    """helpers.py:169-170, 177-178, 195-197: counts are exp(log counts), averaged over the two strands."""
    pred_counts = np.exp(fwd_logcounts)
    revcomp_counts = np.exp(rev_logcounts)
    return np.average([pred_counts, revcomp_counts], axis=0)


def vs_logfc(allele1_pred_counts, allele2_pred_counts):
    """helpers.py:225."""
    return np.squeeze(np.log2(allele2_pred_counts / allele1_pred_counts))


def vs_quantile(allele_pred_counts, pred_counts):
    """helpers.py:212-213."""
    return np.array([np.max([np.mean(pred_counts < x), (1 / len(pred_counts))]) for x in allele_pred_counts])


def vs_get_pvals(obs, bg, tail):
    """helpers.py:319-335."""
    sorted_bg = np.sort(bg)
    if tail == "right" or tail == "both":
        rank_right = len(sorted_bg) - np.searchsorted(sorted_bg, obs, side="left")
        pval_right = (rank_right + 1) / (len(sorted_bg) + 1)
        if tail == "right":
            return pval_right
    if tail == "left" or tail == "both":
        rank_left = np.searchsorted(sorted_bg, obs, side="right")
        pval_left = (rank_left + 1) / (len(sorted_bg) + 1)
        if tail == "left":
            return pval_left
    assert tail == "both"
    min_pval = np.minimum(pval_left, pval_right)
    pval_both = min_pval * 2

    return pval_both


def vs_geo_mean_overflow(iterable, axis=0):
    """variant_summary_across_folds.py:8-9."""
    return np.exp(np.log(iterable).mean(axis=0))


# --- toy predictions -------------------------------------------------------------------------------------------


def toy_preds(rng, n, alleles=("a1", "a2"), effect=0.0, strand_noise=0.05):
    """A predictor's output: per-strand float32 log counts plus VS's strand average in <allele>_logcounts."""
    base = rng.normal(4.0, 1.0, size=n)
    out = {}
    for allele in alleles:
        shift = effect * rng.normal(size=n) if allele == "a2" else 0.0
        fwd = (base + shift + strand_noise * rng.normal(size=n)).astype(np.float32)
        rev = (base + shift + strand_noise * rng.normal(size=n)).astype(np.float32)
        out[f"{allele}_fwd"], out[f"{allele}_rev"] = fwd, rev
        out[f"{allele}_logcounts"] = np.log(vs_strand_average(fwd.astype(np.float64), rev)).astype(np.float32)
    return out


def test_strand_mean_is_the_log_of_vs_average_counts():
    rng = np.random.default_rng(0)
    fwd, rev = rng.normal(3, 2, size=1000), rng.normal(3, 2, size=1000)
    np.testing.assert_allclose(sc.strand_mean(fwd, rev), np.log(vs_strand_average(fwd, rev)), rtol=0, atol=1e-12)
    # It is not the mean of the log counts, unless the strands agree.
    assert np.max(np.abs(sc.strand_mean(fwd, rev) - (fwd + rev) / 2)) > 0.1
    np.testing.assert_allclose(sc.strand_mean(fwd, fwd), fwd, rtol=0, atol=1e-12)


def test_logfc_is_vs_logfc():
    rng = np.random.default_rng(1)
    preds = toy_preds(rng, 500, effect=0.5)
    ours = sc.logfc(sc.allele_logcounts(preds, "a1"), sc.allele_logcounts(preds, "a2"))
    c1 = vs_strand_average(preds["a1_fwd"].astype(np.float64), preds["a1_rev"].astype(np.float64))
    c2 = vs_strand_average(preds["a2_fwd"].astype(np.float64), preds["a2_rev"].astype(np.float64))
    np.testing.assert_allclose(ours, vs_logfc(c1, c2), rtol=0, atol=1e-10)


def test_allele_logcounts_fall_back_to_the_forward_strand_and_the_stored_field():
    fwd = np.array([1.0, 2.0], dtype=np.float32)
    assert sc.allele_logcounts({"a1_fwd": fwd}, "a1").tolist() == [1.0, 2.0]
    assert sc.allele_logcounts({"a1_logcounts": fwd + 1}, "a1").tolist() == [2.0, 3.0]
    with pytest.raises(KeyError):
        sc.allele_logcounts({"a1_fwd": fwd}, "a2")


def test_quantiles_are_vs_quantiles():
    rng = np.random.default_rng(2)
    peaks = np.round(rng.normal(5, 1, size=400), 1)  # many ties
    lc = np.concatenate([rng.normal(5, 1.5, size=300), peaks[:20], [peaks.min() - 1, peaks.max() + 1, peaks.min()]])
    ours = sc.allele_quantile(lc, peaks)
    np.testing.assert_allclose(ours, vs_quantile(np.exp(lc), np.exp(peaks)), rtol=0, atol=1e-12)
    assert ours[-3] == ours[-1] == 1 / len(peaks)  # below or at the lowest peak: the 1 / P floor
    assert ours[-2] == 1.0


def test_aaq_is_the_quantile_of_the_stronger_allele():
    rng = np.random.default_rng(3)
    peaks, lc1, lc2 = rng.normal(5, 1, size=300), rng.normal(5, 1, size=200), rng.normal(5, 1, size=200)
    scores = sc.score({"a1_logcounts": lc1, "a2_logcounts": lc2}, peaks)
    np.testing.assert_array_equal(scores["active_allele_quantile"], sc.allele_quantile(np.maximum(lc1, lc2), peaks))
    np.testing.assert_array_equal(scores["quantile_change"], scores["allele2_quantile"] - scores["allele1_quantile"])


@pytest.mark.parametrize("tail", ["right", "left", "both"])
def test_get_pvals_is_vs_get_pvals(tail):
    rng = np.random.default_rng(4)
    bg = np.round(rng.normal(size=999), 2)
    obs = np.concatenate([rng.normal(size=500) * 2, bg[:50], [bg.min() - 1, bg.max() + 1]])
    np.testing.assert_array_equal(sc.get_pvals(obs, bg, tail), vs_get_pvals(obs, bg, tail))


def test_get_pvals_counts_and_limits():
    bg = np.arange(1.0, 100.0)  # 99 null rows
    right = sc.get_pvals(np.array([0.0, 50.0, 1000.0]), bg, "right")
    assert right.tolist() == [1.0, (50 + 1) / 100, 1 / 100]  # (#bg >= obs + 1) / (n + 1)
    both = sc.get_pvals(np.array([-5.0, 50.0, 1000.0]), bg, "both")
    assert both[0] == both[2] == 2 / 100 and both[1] > 1  # 2 / (n + 1) at best; not capped at 1
    with pytest.raises(ValueError):
        sc.get_pvals(bg, bg, "two-sided")


def test_score_fold_takes_p_values_against_the_null():
    rng = np.random.default_rng(5)
    variants, null = toy_preds(rng, 300, effect=0.8), toy_preds(rng, 2000, effect=0.1)
    peaks = toy_preds(rng, 500, alleles=("a1",))
    var_scores, null_scores = sc.score_fold(variants, null, peaks)
    np.testing.assert_array_equal(var_scores["abs_logfc.pval"],
                                  vs_get_pvals(var_scores["abs_logfc"], null_scores["abs_logfc"], "right"))  # fmt: skip
    np.testing.assert_array_equal(var_scores["logfc.pval"],
                                  vs_get_pvals(var_scores["logfc"], null_scores["logfc"], "both"))  # fmt: skip
    np.testing.assert_array_equal(var_scores["quantile_change.pval"],
                                  vs_get_pvals(var_scores["quantile_change"], null_scores["quantile_change"], "both"))  # fmt: skip
    np.testing.assert_allclose(var_scores["logfc_fwd"], (variants["a2_fwd"] - variants["a1_fwd"]) / np.log(2.0),
                               rtol=0, atol=1e-6)  # fmt: skip
    assert "logfc.pval" not in null_scores
    # Larger effects than the null's give smaller p-values.
    assert np.median(var_scores["abs_logfc.pval"]) < 0.5


def test_a_null_without_allele2_is_refused():
    rng = np.random.default_rng(6)
    variants, peaks = toy_preds(rng, 10), toy_preds(rng, 10, alleles=("a1",))
    with pytest.raises(ValueError, match="allele2"):
        sc.score_fold(variants, toy_preds(rng, 10, alleles=("a1",)), peaks)


def test_derived_full_model_adds_the_parts_counts():
    rng = np.random.default_rng(7)
    nobias, bias = toy_preds(rng, 400, effect=0.5), toy_preds(rng, 400, effect=0.5)
    for part in (nobias, bias):  # a predictor that stored the mean of the log counts: must not matter
        for allele in ("a1", "a2"):
            part[f"{allele}_logcounts"] = ((part[f"{allele}_fwd"] + part[f"{allele}_rev"]) / 2).astype(np.float32)
    full = sc.derive_full(nobias, bias)
    assert all(v.dtype == np.float32 for v in full.values())
    for key in ("a1_fwd", "a1_rev", "a2_fwd", "a2_rev"):
        np.testing.assert_allclose(np.exp(full[key].astype(np.float64)),
                                   np.exp(nobias[key].astype(np.float64)) + np.exp(bias[key].astype(np.float64)),
                                   rtol=3e-6)  # fmt: skip
    # Strand averages commute with the sum of counts, so the full model's strand average is the sum of the parts'.
    for allele in ("a1", "a2"):
        parts = [sc.allele_logcounts(p, allele) for p in (nobias, bias)]
        np.testing.assert_allclose(np.exp(full[f"{allele}_logcounts"].astype(np.float64)),
                                   np.exp(parts[0]) + np.exp(parts[1]), rtol=3e-6)  # fmt: skip
        np.testing.assert_allclose(sc.allele_logcounts(full, allele), np.logaddexp(*parts), rtol=0, atol=1e-5)


def test_stored_strand_average_is_recognised():
    rng = np.random.default_rng(8)
    preds = toy_preds(rng, 200, strand_noise=0.5)
    assert sc.stored_average(preds, "a1")[0] == "counts"
    preds["a1_logcounts"] = ((preds["a1_fwd"] + preds["a1_rev"]) / 2).astype(np.float32)
    assert sc.stored_average(preds, "a1")[0] == "log_counts"
    preds["a1_logcounts"] = preds["a1_logcounts"][::-1].copy()
    assert sc.stored_average(preds, "a1")[0] == "other"
    assert sc.stored_average({"a1_fwd": preds["a1_fwd"]}, "a1")[0] == "unchecked"


def test_aggregate_folds_is_vs_summary_plus_the_heldout_fold():
    rng = np.random.default_rng(9)
    n, folds = 50, [0, 1, 2]
    per_fold = {k: {name: rng.normal(size=n) for name in sc.SCORES} for k in folds}
    for k in folds:
        for name in sc.PVALS:
            per_fold[k][f"{name}.pval"] = rng.uniform(0.001, 1, size=n)
    fold_test = rng.integers(-1, 5, size=n)  # -1: no fold; 3 and 4: folds not scored
    out = sc.aggregate_folds(per_fold, fold_test)
    for name in sc.SCORES:
        np.testing.assert_allclose(out[f"{name}.mean"], np.mean([per_fold[k][name] for k in folds], axis=0))
    for name in sc.PVALS:
        np.testing.assert_allclose(out[f"{name}.mean.pval"],
                                   vs_geo_mean_overflow([per_fold[k][f"{name}.pval"] for k in folds]))  # fmt: skip
    for i, k in enumerate(fold_test):
        if k in folds:
            assert out["logfc.heldout"][i] == per_fold[k]["logfc"][i]
            assert out["abs_logfc.heldout.pval"][i] == per_fold[k]["abs_logfc.pval"][i]
        else:
            assert np.isnan(out["logfc.heldout"][i]) and np.isnan(out["abs_logfc.heldout.pval"][i])
    assert out["heldout_fold"].tolist() == fold_test.tolist()


def test_tables_round_trip(tmp_path):
    cols = {"variant_id": np.array(["chr1:5:A:G", "chr2:7:C:CT"]), "logfc": np.array([0.123456789, np.nan]),
            "heldout_fold": np.array([3, -1]), "flag": np.array([True, False])}  # fmt: skip
    sc.write_table(tmp_path / "t.tsv.gz", cols)
    back = sc.read_table(tmp_path / "t.tsv.gz")
    assert back["variant_id"].tolist() == cols["variant_id"].tolist()
    assert back["logfc"][0] == pytest.approx(0.1234568, abs=1e-7) and np.isnan(back["logfc"][1])
    assert back["heldout_fold"].tolist() == [3.0, -1.0] and back["flag"].tolist() == [1.0, 0.0]


def test_npz_table_matches_its_tsv_digit_for_digit(tmp_path):
    """save_table writes the .npz (float32, int64, unicode) and the TSV from the same values: every TSV cell is the
    .npz value written with 7 significant digits, so the two agree to 7 significant digits."""
    rng = np.random.default_rng(13)
    n = 2000
    cols = {"variant_id": np.array([f"chr1:{1000 + i}:A:G" for i in range(n)]),
            "logfc": rng.normal(size=n) * 10.0 ** rng.integers(-9, 9, size=n), "pval": rng.uniform(1e-6, 1, size=n),
            "heldout_fold": rng.integers(-1, 5, size=n), "flag": rng.random(n) < 0.5}  # fmt: skip
    cols["logfc"][[3, 7, 11]] = [np.nan, -0.0, np.inf]
    path = tmp_path / "t.tsv.gz"
    sc.save_table(path, cols)
    with np.load(tmp_path / "t.npz", allow_pickle=False) as z:
        stored = {k: z[k] for k in z.files}
    assert list(stored) == list(cols)
    assert stored["logfc"].dtype == np.float32 and stored["pval"].dtype == np.float32
    assert stored["heldout_fold"].dtype == np.int64 and stored["flag"].dtype == np.int64
    assert stored["variant_id"].dtype.kind == "U" and stored["variant_id"].tolist() == cols["variant_id"].tolist()
    with gzip.open(path, "rt") as fh:
        header, *lines = fh.read().splitlines()
    assert header.split("\t") == list(cols) and len(lines) == n
    cells = list(zip(*(line.split("\t") for line in lines)))
    for j, (name, values) in enumerate(stored.items()):
        as_text = (lambda v: format(v, ".7g")) if values.dtype.kind == "f" else str
        written = [as_text(v) for v in values.tolist()]  # format(x, ".7g") writes what "%.7g" % x does
        assert list(cells[j]) == written, name
        if values.dtype.kind == "f":  # 7 significant digits: within 5e-7 of the value, relatively
            np.testing.assert_allclose(np.array(cells[j], dtype=np.float64), values, rtol=6e-7)
    assert cells[1][3] == "nan" and cells[1][7] == "-0" and cells[1][11] == "inf"
    # read_table reads the .npz: floats as float64, integers too, text as str
    back = sc.read_table(path)
    assert list(back) == list(cols) and all(back[k].dtype == np.float64 for k in ("logfc", "heldout_fold", "flag"))
    np.testing.assert_array_equal(back["logfc"], stored["logfc"].astype(np.float64))
    assert back["heldout_fold"].tolist() == cols["heldout_fold"].tolist()
    assert back["variant_id"].tolist() == cols["variant_id"].tolist()


def test_read_table_prefers_the_npz(tmp_path):
    path = tmp_path / "t.tsv.gz"
    sc.save_table(path, {"variant_id": np.array(["a", "b"]), "logfc": np.array([1.0, 2.0])})
    sc.write_table(path, {"variant_id": np.array(["a", "b"]), "logfc": np.array([5.0, 6.0])})  # a TSV unlike its npz
    assert sc.read_table(path)["logfc"].tolist() == [1.0, 2.0]
    assert set(sc.read_table(path, columns=["logfc", "absent"])) == {"logfc"}
    (tmp_path / "t.npz").unlink()
    assert sc.read_table(path)["logfc"].tolist() == [5.0, 6.0]  # no .npz: the TSV
    assert set(sc.read_table(path, columns=["logfc"])) == {"logfc"}
    path.unlink()
    with pytest.raises(FileNotFoundError):
        sc.read_table(path)
    # tsv=False (scores.py --no-fold-tsv): the .npz alone, read through the TSV's path; a stale TSV is deleted
    path.write_bytes(b"stale")
    sc.save_table(path, {"variant_id": np.array(["a"]), "logfc": np.array([3.0])}, tsv=False)
    assert not path.exists() and sc.read_table(path)["logfc"].tolist() == [3.0]
    assert sc.npz_path(tmp_path / "x.summary.tsv.gz") == tmp_path / "x.summary.npz"
    assert sc.npz_path(tmp_path / "agreement.tsv") == tmp_path / "agreement.npz"


def test_list_columns_fold_numbers_and_align(tmp_path):
    lines = ["variant_id\tchrom\tpos\tref\talt\tsource\tstratum\taf\tdist_to_summit\tfold_test",
             "chr1:5:A:G\tchr1\t5\tA\tG\tsrc\tsummit100\t0.1\t3\t0",
             "chr1:9:AC:GT\tchr1\t9\tAC\tGT\tsrc\tflank500\t\t150\t",
             "chr2:7:C:CTTT\tchr2\t7\tC\tCTTT\tsrc\tsummit100\tnan\t10\t3"]  # fmt: skip
    (tmp_path / "l.tsv").write_text("\n".join(lines) + "\n\n")  # a blank last line, skipped
    cols = sc.read_list_columns(tmp_path / "l.tsv", ("variant_id",), optional=("fold_test", "absent"))
    assert set(cols) == {"variant_id", "fold_test"} and cols["fold_test"] == ("0", "", "3")
    assert sc.fold_numbers(cols["fold_test"]).tolist() == [0, -1, 3]
    assert sc.read_fold_test(tmp_path / "l.tsv") == {"chr1:5:A:G": 0, "chr1:9:AC:GT": -1, "chr2:7:C:CTTT": 3}
    with pytest.raises(ValueError, match="absent"):
        sc.read_list_columns(tmp_path / "l.tsv", ("absent",))
    listed = cmp.read_variant_list(tmp_path / "l.tsv")
    assert listed["type"].tolist() == ["SNV", "MNV", "indel"] and listed["fold_test"].tolist() == [0, -1, 3]
    assert listed["stratum"].tolist() == ["summit100", "flank500", "summit100"]
    ids = listed["variant_id"]
    assert sc.align(ids, ids).tolist() == [0, 1, 2]
    assert sc.align(ids, ids[[2, 0]]).tolist() == [2, 0]
    with pytest.raises(ValueError, match="chr9:1:A:C"):
        sc.align(ids, np.array(["chr1:5:A:G", "chr9:1:A:C"]))


# --- the command line, end to end ------------------------------------------------------------------------------


def _write_inputs(root: Path, n_var=120, n_null=400, n_peaks=300, folds=(0, 1)):
    rng = np.random.default_rng(10)
    ids = np.array([f"chr{1 + i % 3}:{1000 + i}:A:G" for i in range(n_var)])
    (root / "windows").mkdir(parents=True)
    for name, n in (("variants", n_var), ("null", n_null), ("peaks.dsA", n_peaks)):
        names = ids if name == "variants" else np.array([f"{name}_{i}" for i in range(n)])
        extra = {"src_ids": ids[rng.integers(0, n_var, size=n)]} if name == "null" else {}
        np.savez(root / "windows" / f"{name}.npz", ids=names, **extra)
    fold_test = {f"chr{c}": t for c, t in ((1, 0), (2, 1), (3, 4))}  # chr3's fold is not scored here
    lines = ["variant_id\tchrom\tpos\tref\talt\tsource\tstratum\taf\tdist_to_summit\tfold_test"]
    lines += [f"{v}\t{v.split(':')[0]}\t{v.split(':')[1]}\tA\tG\tgenome\tsummit100\t0.1\t0\t{fold_test[v.split(':')[0]]}"
              for v in ids]  # fmt: skip
    (root / "list.tsv").write_text("\n".join(lines) + "\n")
    for k in folds:
        d = root / "preds" / "dsA" / f"fold_{k}"
        d.mkdir(parents=True)
        for model in ("cherimoya", "cbp_nobias", "cbp_bias"):
            for name, n, alleles in (("variants", n_var, ("a1", "a2")), ("null", n_null, ("a1", "a2")),
                                     ("peaks.dsA", n_peaks, ("a1",))):  # fmt: skip
                np.savez(d / f"{model}.{name}.npz", **toy_preds(rng, n, alleles=alleles, effect=0.3))
    return ids


def test_cli_end_to_end(tmp_path):
    ids = _write_inputs(tmp_path)
    args = ["--datasets", "dsA", "--preds-root", str(tmp_path / "preds"), "--windows-root", str(tmp_path / "windows"),
            "--list", str(tmp_path / "list.tsv"), "--variants", "variants", "--out-root", str(tmp_path / "scores"),
            "--folds", "0", "1"]  # fmt: skip
    assert sc.main(args) == 0
    out = tmp_path / "scores" / "dsA"
    for model in sc.MODELS:
        per_fold = {}
        for k in (0, 1):
            table = sc.read_table(out / f"fold_{k}" / f"{model}.variants.tsv.gz")
            assert table["variant_id"].tolist() == ids.tolist()
            if model != "cbp_full":
                d = tmp_path / "preds" / "dsA" / f"fold_{k}"
                expected, _ = sc.score_fold(*(sc.load_npz(d / f"{model}.{name}.npz")
                                              for name in ("variants", "null", "peaks.dsA")))  # fmt: skip
                for col in ("logfc", "abs_logfc.pval", "active_allele_quantile", "logfc_rev"):
                    np.testing.assert_allclose(table[col], expected[col], rtol=1e-6, atol=1e-7)
            per_fold[k] = table
            null = sc.read_table(out / f"fold_{k}" / f"{model}.null.tsv.gz")
            assert len(null["id"]) == 400 and "src_id" in null
        summary = sc.read_table(out / f"{model}.variants.summary.tsv.gz")
        np.testing.assert_allclose(summary["logfc.mean"], (per_fold[0]["logfc"] + per_fold[1]["logfc"]) / 2,
                                   rtol=1e-6, atol=1e-6)  # fmt: skip
        chrom = np.array([v.split(":")[0] for v in ids])
        np.testing.assert_array_equal(summary["logfc.heldout"][chrom == "chr1"], per_fold[0]["logfc"][chrom == "chr1"])
        np.testing.assert_array_equal(summary["logfc.heldout"][chrom == "chr2"], per_fold[1]["logfc"][chrom == "chr2"])
        assert np.isnan(summary["logfc.heldout"][chrom == "chr3"]).all()
        meta = json.loads((out / f"{model}.variants.json").read_text())
        source = meta["folds"]["0"]["variants.source"]
        assert source.startswith("derived") == (model == "cbp_full")
        assert meta["folds"]["0"]["variants.a1_logcounts"]["holds"] == "counts"


def test_cli_no_fold_tsv(tmp_path):
    """--no-fold-tsv: per-fold tables as .npz only (a stale per-fold TSV is deleted), summaries in both; compare.py
    reads the same tables through read_table."""
    ids = _write_inputs(tmp_path)
    out = tmp_path / "scores" / "dsA"
    stale = out / "fold_0" / "cherimoya.variants.tsv.gz"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"from an earlier run")
    args = ["--datasets", "dsA", "--preds-root", str(tmp_path / "preds"), "--windows-root", str(tmp_path / "windows"),
            "--list", str(tmp_path / "list.tsv"), "--variants", "variants", "--out-root", str(tmp_path / "scores"),
            "--folds", "0", "1", "--no-fold-tsv"]  # fmt: skip
    assert sc.main(args) == 0
    for k in (0, 1):
        assert not list((out / f"fold_{k}").glob("*.tsv.gz"))
        assert {p.name for p in (out / f"fold_{k}").glob("*.npz")} == {
            f"{m}.{name}.npz" for m in sc.MODELS for name in ("variants", "null")}  # fmt: skip
    for model in sc.MODELS:
        for suffix in (".tsv.gz", ".npz"):
            assert (out / f"{model}.variants.summary{suffix}").is_file()
        assert sc.read_table(out / "fold_1" / f"{model}.variants.tsv.gz")["variant_id"].tolist() == ids.tolist()
        assert json.loads((out / f"{model}.variants.json").read_text())["fold_tsv"] is False
    compared = ["--scores-root", str(tmp_path / "scores"), "--datasets", "dsA", "--list", str(tmp_path / "list.tsv"),
                "--variants", "variants", "--out-dir", str(tmp_path / "compare"), "--folds", "0", "1",
                "--no-figures"]  # fmt: skip
    assert cmp.main(compared) == 0
    agreement = sc.read_table(tmp_path / "compare" / "agreement.tsv")
    assert (agreement["n"][agreement["subset"] == "all"] == len(ids)).all()


def test_cli_reports_missing_predictions(tmp_path):
    _write_inputs(tmp_path)
    args = ["--datasets", "dsA", "--preds-root", str(tmp_path / "preds"), "--windows-root", str(tmp_path / "windows"),
            "--list", str(tmp_path / "list.tsv"), "--variants", "variants", "--out-root", str(tmp_path / "scores")]  # fmt: skip
    assert sc.main(args) == 1  # folds 2-4 have no predictions


def test_predictions_must_be_their_windows_rows(tmp_path):
    ids = _write_inputs(tmp_path)
    d = tmp_path / "preds" / "dsA" / "fold_0"
    preds = sc.load_npz(d / "cherimoya.variants.npz")
    np.savez(d / "cherimoya.variants.npz", ids=ids[::-1], **preds)  # the predictor's ids, in another order
    with pytest.raises(ValueError, match="ids"):
        sc.load_preds(tmp_path / "preds", "dsA", 0, "cherimoya", "variants", ids, {})
    np.savez(d / "cherimoya.variants.npz", ids=ids, **preds)
    assert sc.load_preds(tmp_path / "preds", "dsA", 0, "cherimoya", "variants", ids, {})["a1_fwd"].shape == (len(ids),)
    with pytest.raises(ValueError, match="rows"):
        sc.load_preds(tmp_path / "preds", "dsA", 0, "cherimoya", "variants", ids[:-1], {})


def test_non_finite_predictions_are_refused(tmp_path):
    """A NaN would sort last and be ranked as the most extreme row (p = 1 / (n + 1), AAQ = 1): refused instead."""
    ids = _write_inputs(tmp_path, folds=(0,))
    d = tmp_path / "preds" / "dsA" / "fold_0"
    preds = sc.load_npz(d / "cherimoya.variants.npz")
    preds["a1_fwd"][7] = np.nan
    np.savez(d / "cherimoya.variants.npz", **preds)
    with pytest.raises(ValueError, match="non-finite"):
        sc.load_preds(tmp_path / "preds", "dsA", 0, "cherimoya", "variants", ids, {})
    # The derived full model inherits a non-finite part.
    null_ids, _ = sc.window_ids(tmp_path / "windows", "null")
    bias = sc.load_npz(d / "cbp_bias.null.npz")
    bias["a2_rev"][3] = np.inf
    np.savez(d / "cbp_bias.null.npz", **bias)
    with pytest.raises(ValueError, match="non-finite"):
        sc.load_preds(tmp_path / "preds", "dsA", 0, "cbp_full", "null", null_ids, {})
    # The command line skips those models, scores the others and exits 1.
    args = ["--datasets", "dsA", "--preds-root", str(tmp_path / "preds"), "--windows-root", str(tmp_path / "windows"),
            "--list", str(tmp_path / "list.tsv"), "--variants", "variants", "--out-root", str(tmp_path / "scores"),
            "--folds", "0"]  # fmt: skip
    assert sc.main(args) == 1
    out = tmp_path / "scores" / "dsA"
    assert (out / "cbp_nobias.variants.summary.tsv.gz").exists()
    for model in ("cherimoya", "cbp_bias", "cbp_full"):
        assert not (out / f"{model}.variants.summary.tsv.gz").exists()
    with pytest.raises(ValueError, match="non-finite"):
        sc.get_pvals(np.array([0.5, 1.0]), np.array([0.1, np.nan, 0.3]), "right")


def test_compare_end_to_end(tmp_path):
    _write_inputs(tmp_path)
    scored = ["--datasets", "dsA", "--preds-root", str(tmp_path / "preds"), "--windows-root", str(tmp_path / "windows"),
              "--list", str(tmp_path / "list.tsv"), "--variants", "variants", "--out-root", str(tmp_path / "scores"),
              "--folds", "0", "1"]  # fmt: skip
    assert sc.main(scored) == 0
    out = tmp_path / "compare"
    args = ["--scores-root", str(tmp_path / "scores"), "--datasets", "dsA", "--list", str(tmp_path / "list.tsv"),
            "--variants", "variants", "--out-dir", str(out), "--folds", "0", "1"]  # fmt: skip
    figures = importlib.util.find_spec("matplotlib") is not None
    assert cmp.main(args + ([] if figures else ["--no-figures"])) == 0
    agreement = sc.read_table(out / "agreement.tsv")
    loo = (agreement["measure"] == "pearson_loo") & (agreement["subset"] == "all")
    assert sorted(agreement["model_a"][loo].tolist()) == sorted(sc.MODELS)
    strand = sc.read_table(out / "strand.tsv")
    assert set(strand["measure"].tolist()) == {"pearson_fwd_rev", "spearman_fwd_rev", "pearson_fwd_both",
                                               "top_overlap_fwd_both"}  # fmt: skip
    report = (out / "report.md").read_text()
    assert "Cherimoya vs ChromBPNet uncorrected" in report and "ceiling Cherimoya" in report
    assert "- ChromBPNet corrected: chrombpnet_nobias.h5" in report and "ChromBPNet full" not in report
    for name in ("fig_scatter.png", "fig_agreement.png", "fig_strand.png", "fig_null.png"):
        assert (out / name).exists() == figures


def test_report_glossary_names_only_the_models_run():
    """report.md's header says what each ChromBPNet label means, for the ChromBPNet models of the run only."""
    every = cmp.glossary(cmp.MODELS)
    assert len(every) == 1 and every[0].startswith("- ChromBPNet corrected: chrombpnet_nobias.h5, the model with Tn5")
    assert "ChromBPNet uncorrected: chrombpnet.h5" in every[0] and "ChromBPNet bias model: the Tn5" in every[0]
    assert every[0].endswith("Cherimoya has no bias model and, like the uncorrected model, predicts the raw signal.")
    some = cmp.glossary(["cherimoya", "cbp_nobias"])[0]
    assert "ChromBPNet uncorrected" not in some and "ChromBPNet bias model" not in some
    assert some.endswith("Cherimoya has no bias model and predicts the raw signal, Tn5 bias included.")
    assert cmp.glossary(["cherimoya"]) == []


# --- compare.py's statistics -----------------------------------------------------------------------------------


def test_correlations_skip_missing_values():
    x = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
    y = np.array([2.0, 4.0, 6.0, 1.0, np.inf])
    r = cmp.correlations(x, y)
    assert r["n"] == 3 and r["pearson"] == pytest.approx(1.0) and r["spearman"] == pytest.approx(1.0)
    assert np.isnan(cmp.correlations(x[:2], y[:2])["pearson"])  # too few values


def test_top_overlap():
    rng = np.random.default_rng(11)
    x = rng.normal(size=1000)
    assert cmp.top_overlap(x, x, 0.01) == 1.0
    assert cmp.top_overlap(x, -x, 0.01) == 1.0  # by magnitude
    assert cmp.top_overlap(x, rng.normal(size=1000), 0.01) < 0.5


def test_leave_one_fold_out_ceiling():
    rng = np.random.default_rng(12)
    truth = rng.normal(size=5000)
    folds = np.stack([truth + 0.5 * rng.normal(size=5000) for _ in range(5)], axis=1)  # [N, K]
    ceiling = cmp.leave_one_out(folds)
    # Fold k against the mean of the other four: var(truth) = 1, noise 0.25 and 0.0625.
    expected = 1 / np.sqrt(1.25 * 1.0625)
    assert ceiling["pearson"] == pytest.approx(expected, abs=0.02) and len(ceiling["per_fold_pearson"]) == 5


def test_figure_sample_is_seeded_and_statistics_use_every_row():
    """The scatter panels draw a seeded sample of at most --figure-points variants; the statistics, a panel's r and the
    null figure's curves use every variant."""
    sample = cmp.subsample(1000, 100, 7)
    assert len(sample) == 100 and np.array_equal(sample, cmp.subsample(1000, 100, 7)) and (np.diff(sample) > 0).all()
    assert not np.array_equal(sample, cmp.subsample(1000, 100, 8))
    assert np.array_equal(cmp.subsample(50, 100, 7), np.arange(50)) and np.array_equal(cmp.subsample(50, 0, 7),
                                                                                        np.arange(50))  # fmt: skip
    rng = np.random.default_rng(14)
    n, k = 600, 3
    truth = rng.normal(size=n)

    def model(noise: float) -> cmp.ModelScores:
        logfc = truth[:, None] + noise * rng.normal(size=(n, k))
        fwd, rev = logfc + 0.1 * rng.normal(size=(n, k)), logfc + 0.1 * rng.normal(size=(n, k))
        null_abs = [np.abs(rng.normal(size=200)) for _ in range(k)]
        return cmp.ModelScores(logfc, fwd, rev, rng.uniform(size=(n, k)), null_abs)

    data = {"cherimoya": model(0.3), "cbp_full": model(0.5)}
    fig = cmp.figure_data(data, points=50, seed=3)
    drawn = cmp.subsample(n, 50, 3)
    assert fig["n"] == n and fig["shown"] == 50
    np.testing.assert_array_equal(fig["means"]["cherimoya"], data["cherimoya"].logfc.mean(axis=1)[drawn])
    strand, full = fig["strand"]["cbp_full"], data["cbp_full"]
    assert len(strand["fwd"]) == len(strand["rev"]) == 50
    assert strand["pearson"] == pytest.approx(np.corrcoef(full.logfc_fwd[:, 0], full.logfc_rev[:, 0])[0, 1])
    x, y = fig["survival"]["cherimoya"]["variants"]
    assert y[0] == 1.0 and x[-1] == np.abs(data["cherimoya"].logfc).max()  # every (variant, fold), the largest too
    assert cmp.sample_note({"dsA": fig}, 3).startswith("points: a random sample of 50 variants")
    assert cmp.sample_note({"dsA": {**fig, "shown": n}}, 3) == ""
    # The statistics: every row, whatever the figures draw. One type and one source hold every variant: left out.
    meta = {"stratum": np.array(["summit100", "flank500"] * (n // 2)), "type": np.full(n, "SNV"),
            "source": np.full(n, "both"), "fold_test": np.arange(n) % k}  # fmt: skip
    rows = cmp.compare_dataset("dsA", data, meta, list(range(k)), [("cherimoya", "cbp_full")], 0.01)
    assert {r["subset"] for r in rows["agreement"]} == {"all", "summit100", "flank500"}
    b = next(r for r in rows["agreement"] if r["subset"] == "all" and r["measure"] == "pearson_fold_mean")
    assert b["n"] == n
    means = [data[m].logfc.mean(axis=1) for m in ("cherimoya", "cbp_full")]
    assert b["value"] == pytest.approx(np.corrcoef(*means)[0, 1])
    fwd_both = next(r for r in rows["strand"] if r["subset"] == "all" and r["measure"] == "pearson_fwd_both")
    assert np.isfinite(fwd_both["mean"])


def test_compare_workers_match_one_process(tmp_path):
    """--workers N compares the datasets in spawned processes and writes what one process writes."""
    _write_inputs(tmp_path)
    scored = ["--datasets", "dsA", "--preds-root", str(tmp_path / "preds"), "--windows-root", str(tmp_path / "windows"),
              "--list", str(tmp_path / "list.tsv"), "--variants", "variants", "--out-root", str(tmp_path / "scores"),
              "--folds", "0", "1"]  # fmt: skip
    assert sc.main(scored) == 0
    shutil.copytree(tmp_path / "scores" / "dsA", tmp_path / "scores" / "dsB")  # a second dataset, the same scores
    figures = importlib.util.find_spec("matplotlib") is not None
    common = ["--scores-root", str(tmp_path / "scores"), "--datasets", "dsA", "dsB", "--list",
              str(tmp_path / "list.tsv"), "--variants", "variants", "--folds", "0", "1", "--figure-points",
              "50"]  # fmt: skip
    common += [] if figures else ["--no-figures"]
    assert cmp.main([*common, "--out-dir", str(tmp_path / "one")]) == 0
    assert cmp.main([*common, "--out-dir", str(tmp_path / "two"), "--workers", "2"]) == 0
    for name in ("agreement.tsv", "strand.tsv", "null.tsv"):
        assert (tmp_path / "one" / name).read_text() == (tmp_path / "two" / name).read_text()
    one = (tmp_path / "one" / "agreement.tsv").read_text()
    assert "\ndsA\tall\t" in one and "\ndsB\tall\t" in one
    if figures:
        assert "random sample of 50 variants" in (tmp_path / "two" / "report.md").read_text()


def test_compare_keeps_each_dataset_on_its_own_models(tmp_path):
    """A model one dataset has no scores for is left out of that dataset only; a dataset without scores is left out,
    the others are written, and the exit status is 1."""
    _write_inputs(tmp_path)
    scored = ["--datasets", "dsA", "--preds-root", str(tmp_path / "preds"), "--windows-root", str(tmp_path / "windows"),
              "--list", str(tmp_path / "list.tsv"), "--variants", "variants", "--out-root", str(tmp_path / "scores"),
              "--folds", "0", "1"]  # fmt: skip
    assert sc.main(scored) == 0
    shutil.copytree(tmp_path / "scores" / "dsA", tmp_path / "scores" / "dsB")
    for path in (tmp_path / "scores" / "dsB").glob("fold_*/cbp_full.*"):  # as when scores.py refuses a fold
        path.unlink()
    common = ["--scores-root", str(tmp_path / "scores"), "--list", str(tmp_path / "list.tsv"), "--variants",
              "variants", "--folds", "0", "1", "--no-figures"]  # fmt: skip
    assert cmp.main([*common, "--datasets", "dsA", "dsB", "--out-dir", str(tmp_path / "two")]) == 0
    agreement = sc.read_table(tmp_path / "two" / "agreement.tsv")
    full = (agreement["model_a"] == "cherimoya") & (agreement["model_b"] == "cbp_full")
    nobias = (agreement["model_a"] == "cherimoya") & (agreement["model_b"] == "cbp_nobias")
    assert set(agreement["dataset"][full].tolist()) == {"dsA"}
    assert set(agreement["dataset"][nobias].tolist()) == {"dsA", "dsB"}
    report = (tmp_path / "two" / "report.md").read_text()
    assert "Cherimoya vs ChromBPNet uncorrected" in report and "Not scored for dsB: ChromBPNet uncorrected" in report
    assert cmp.main([*common, "--datasets", "dsA", "dsB", "dsC", "--out-dir", str(tmp_path / "three")]) == 1
    three = sc.read_table(tmp_path / "three" / "agreement.tsv")
    assert set(three["dataset"].tolist()) == {"dsA", "dsB"}
    assert "**Left out:** dsC (no model has scores)" in (tmp_path / "three" / "report.md").read_text()
