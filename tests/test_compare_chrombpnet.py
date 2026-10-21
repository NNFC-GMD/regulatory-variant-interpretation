"""compare_chrombpnet.py's measures are ChromBPNet's, vectorised.

The comparison is only fair if both models go through ChromBPNet's own
definitions, so the vectorised versions here are checked against a line-by-line
transcription of chrombpnet/training/metrics.py (profile_metrics, counts_metrics)
and metrics_utils.py (jsd_min_max_bounds, get_min_max_normalized_value),
including a region with no reads.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, spearmanr

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1] / "workflows" / d) for d in ("train", "attribution", "downstream")
]

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
