"""compare_chrombpnet.py's measures are ChromBPNet's, vectorised.

The comparison is only fair if both models go through ChromBPNet's own
definitions, so the vectorised versions here are checked against a line-by-line
transcription of chrombpnet/training/metrics.py (profile_metrics, counts_metrics)
and metrics_utils.py (jsd_min_max_bounds, get_min_max_normalized_value),
including a region with no reads.

For models with control tracks: the table keeps its original layout without
controls, the controls are found (compare.json, else fit.json) and read as
training reads them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflows" / "cherimoya"))

import compare_chrombpnet as cc  # noqa: E402


def _chrombpnet_profile(true_counts, pred_probs, pseudocount=0.001):
    jsd_pw, jsd_norm = [], []
    for idx in range(true_counts.shape[0]):
        cur = jensenshannon(true_counts[idx] / (pseudocount + np.nansum(true_counts[idx])), pred_probs[idx])
        jsd_pw.append(cur)
        uniform = np.ones(len(true_counts[idx])) * (1.0 / len(true_counts[idx]))
        max_jsd = jensenshannon(true_counts[idx] / np.sum(true_counts[idx]), uniform)
        val = (cur - max_jsd) / (0.0 - max_jsd)
        jsd_norm.append(0 if val < 0 else 1 if val > 1 else val)
    return float(np.nanmedian(jsd_pw)), float(np.nanmedian(jsd_norm))


def _data(seed=0, n=300, width=1000):
    rng = np.random.default_rng(seed)
    observed = rng.poisson(rng.gamma(0.3, 1.0, size=(n, 1)) * np.ones((1, width))).astype(np.float64)
    observed[7] = 0.0  # no reads: NaN in both JSDs, skipped by the medians
    logits = rng.normal(size=(n, width))
    probs = np.exp(logits - logits.mean(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    return observed, probs, rng.normal(3, 1, size=n)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_profile_measures_match_chrombpnet():
    observed, probs, _ = _data()
    ours = cc.profile_measures(observed, probs)
    ref_jsd, ref_norm = _chrombpnet_profile(observed, probs)
    assert ours["median_jsd"] == pytest.approx(ref_jsd, abs=1e-12)
    assert ours["median_norm_jsd"] == pytest.approx(ref_norm, abs=1e-12)


def test_counts_measures_match_chrombpnet():
    observed, _, pred = _data(seed=1)
    true = np.log(1 + observed.sum(1))
    ours = cc.counts_measures(true, pred)
    assert ours["counts_pearson"] == pytest.approx(pearsonr(true, pred)[0])
    assert ours["counts_spearman"] == pytest.approx(spearmanr(true, pred)[0])
    assert ours["counts_mse"] == pytest.approx(((true - pred) ** 2).mean())


def test_perfect_prediction_scores_as_perfect():
    observed, _, _ = _data(seed=2)
    observed = observed[observed.sum(1) > 0]
    probs = observed / observed.sum(1, keepdims=True)
    m = cc.measures(observed, probs, np.log(1 + observed.sum(1)))
    assert m["median_jsd"] == pytest.approx(0.0, abs=1e-6)
    assert m["median_norm_jsd"] == pytest.approx(1.0, abs=1e-6)
    assert m["counts_pearson"] == pytest.approx(1.0) and m["counts_mse"] == pytest.approx(0.0)


def test_one_hot():
    x = cc.one_hot(["ACGTN", "acgtx"])
    assert x.shape == (2, 4, 5) and x.dtype == np.float32
    np.testing.assert_array_equal(x[0][:, :4], np.eye(4))
    np.testing.assert_array_equal(x[0], x[1])
    assert x[0][:, 4].sum() == 0


def _rows():
    return {
        "chrombpnet": dict(zip(cc.MEASURES, (0.741841, 0.694348, 0.524043, 0.251388, 0.57763))),
        "cherimoya": dict(zip(cc.MEASURES, (0.764252, 0.731391, 0.50869, 0.232805, 0.609328))),
    }


def test_table_without_control_is_the_original_layout():
    expected = (
        "dataset\tfold\tmodel\tn_regions\tcounts_pearson\tcounts_spearman\tcounts_mse\tmedian_jsd\tmedian_norm_jsd\n"
        + "ds\t0\tchrombpnet\t64772\t0.741841\t0.694348\t0.524043\t0.251388\t0.57763\n"
        + "ds\t0\tcherimoya\t64772\t0.764252\t0.731391\t0.50869\t0.232805\t0.609328\n"
    )
    assert cc.format_table({"dataset": "ds", "fold": "0"}, _rows(), 64772) == expected


def test_table_with_control_adds_a_last_column():
    lines = cc.format_table({"dataset": "ds", "fold": 0}, _rows(), 10, control=True).splitlines()
    assert lines[0].split("\t")[-1] == "control" and lines[0].split("\t")[:-1][-1] == "median_norm_jsd"
    assert [line.split("\t")[-1] for line in lines[1:]] == ["n/a", "real"]


def test_controls_come_from_compare_json_then_fit_json(tmp_path):
    compare = tmp_path / "compare.json"
    assert cc.resolve_controls({}, compare) == (None, "")
    (tmp_path / "fit.json").write_text(json.dumps({"controls": None}))
    assert cc.resolve_controls({}, compare) == (None, "")
    (tmp_path / "fit.json").write_text(json.dumps({"controls": ["bias.bw", ["s+.bw", "s-.bw"]]}))
    files, source = cc.resolve_controls({}, compare)
    assert files == [str(tmp_path / f) for f in ("bias.bw", "s+.bw", "s-.bw")] and source == str(tmp_path / "fit.json")
    files, source = cc.resolve_controls({"controls": ["/abs/own.bw"]}, compare)
    assert files == ["/abs/own.bw"] and source == str(compare)
    with pytest.raises(ValueError, match="list"):
        cc.resolve_controls({"controls": "/abs/own.bw"}, compare)


def test_read_controls_matches_the_training_reader(tmp_path):
    pyBigWig = pytest.importorskip("pyBigWig")
    extract_loci = pytest.importorskip("tangermeme.io").extract_loci

    rng = np.random.default_rng(3)
    sizes = {"chrA": 6000, "chrB": 5000}
    fasta = tmp_path / "genome.fa"
    fasta.write_text("".join(f">{c}\n" + "".join(rng.choice(list("ACGT"), n)) + "\n" for c, n in sizes.items()))
    track = tmp_path / "bias.bw"
    values = {c: rng.uniform(0, 3, n) for c, n in sizes.items()}
    values["chrB"][2000:2100] = np.nan  # a gap: read as 0 by both
    with pyBigWig.open(str(track), "w") as bw:
        bw.addHeader(list(sizes.items()), maxZooms=0)
        for c, v in values.items():
            keep = ~np.isnan(v)
            starts = np.flatnonzero(keep)
            bw.addEntries([c] * len(starts), starts.tolist(), ends=(starts + 1).tolist(), values=v[keep].tolist())
    chroms, centers = np.array(["chrA", "chrB", "chrB"]), np.array([3000, 2050, 3100])
    ours = cc.read_controls([str(track)], chroms, centers)
    loci = tmp_path / "loci.bed"
    loci.write_text("".join(f"{c}\t{m - 500}\t{m + 500}\n" for c, m in zip(chroms, centers)))
    _, theirs = extract_loci(loci=str(loci), sequences=str(fasta), in_signals=[str(track)], in_window=cc.IN_WINDOW)
    assert ours.shape == (3, 1, cc.IN_WINDOW) and ours.dtype == np.float32
    np.testing.assert_array_equal(ours, theirs.numpy())
    assert ours[1, 0, 1007:1107].sum() == 0
