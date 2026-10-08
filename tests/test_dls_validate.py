"""The gate arithmetic of workflows/cherimoya/dls_validate.py, and its `gates` report on synthetic runs (CPU only).

The harness's runs need a GPU and the real model, and take tens of minutes; its gates need neither. These tests check
the pieces the verdicts rest on: the per-pair and per-peak rel-L2, the Pearson correlation, the floor-relative limits of
E3/E4/E6/E10, E8's bounds and E7(a)'s tie-breaker. They also run `gates` end to end on a synthetic validation directory,
with every run the report reads, and check that an engine with a real error fails while one at the floor passes. That
includes E1 and E10 as revised after round 2 (design/revisions.md): E1 in reference mode within 2x the stock forward's
rerun noise and 1.2x its distance to the fp64 truth, E10 in fp32 mode in E4's form against F_hh, with the original
bounds reported but not gating.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "workflows" / "cherimoya"))

import dls_validate as dv  # noqa: E402


def test_rel_l2_per_pair_and_per_peak():
    rng = np.random.default_rng(0)
    b = rng.normal(size=(5, 3, 4, 11)).astype(np.float32)
    a = b.copy()
    a[2, 1] *= 1.5  # one pair off by 50%
    per_pair = dv.rel_l2(a, b, lead=2, chunk=2)
    assert per_pair.shape == (15,)
    assert per_pair[2 * 3 + 1] == pytest.approx(0.5, rel=1e-6)
    assert np.count_nonzero(per_pair) == 1
    per_peak = dv.rel_l2(a[:, 0], b[:, 0], lead=1)
    assert per_peak.shape == (5,) and not per_peak.any()
    with pytest.raises(ValueError):
        dv.rel_l2(a[:4], b, lead=2)


def test_pearson_and_stats():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, 4, 10))
    assert np.allclose(dv.pearson_rows(x, 3 * x + 1), 1.0)
    assert np.allclose(dv.pearson_rows(x, -x), -1.0)
    s = dv.stats(np.arange(101, dtype=float))
    assert s == {"n": 101, "q50": 50.0, "q99": 99.0, "max": 100.0}


def test_floor_relative_bounds_follow_the_acceptance_criteria():
    # E3: median <= max(2 F(0.5), 1e-5), p99 <= max(3 F(0.99), 1e-4)
    assert dv.floor_bounds({"q50": 1e-4, "q99": 1e-3}) == {"q50": 2e-4, "q99": pytest.approx(3e-3)}
    assert dv.floor_bounds({"q50": 1e-7, "q99": 1e-6}) == {"q50": 1e-5, "q99": 1e-4}
    # E8: fraction <= max(2 x R1's, 0.3%), max <= max(2 x R1's max, 4e-3); the pilot's numbers give 0.306% and 4e-3
    b = dv.e8_bounds(153, 99_980, 1.9636e-3)
    assert b["fraction"] == pytest.approx(2 * 153 / 99_980) and b["max"] == 4e-3
    assert dv.e8_bounds(1, 1000, 1e-3) == {"fraction": 0.003, "max": 4e-3}
    # E7(a): ours within 1.2 x the reference's distance to the truth at the median and the max
    assert dv.accuracy_limited({"q50": 1.2, "max": 2.0}, {"q50": 1.0, "max": 2.0}) == {"q50": True, "max": True}
    assert dv.accuracy_limited({"q50": 1.3, "max": 2.0}, {"q50": 1.0, "max": 2.0}) == {"q50": False, "max": True}


N, K, L, W, N_ALL = 16, 4, 60, 12, 40
TRUTH = np.array([0, 5, 10, 15])


def _write_run(root: Path, name: str, kind: str, spec: dict, arrays: dict, extra: dict) -> None:
    d = root / "runs" / name
    d.mkdir(parents=True)
    for key, value in arrays.items():
        np.save(d / f"{key}.npy", value)
    deltas = arrays["deltas"]
    record = {
        "kind": kind,
        "run": name,
        "spec": spec,
        "args": {"batch_size": 20},
        "timings_s": {"total": 1.0, "run": 1.0, "deep_lift_shap": 1.0},
        "deltas": dv._delta_summary(deltas, 1e-3),
        **extra,
    }
    (d / "record.json").write_text(json.dumps(record))


def _synthetic(root: Path, engine_noise: float, forward_noise: float = 1.0) -> None:
    """A validation directory: the truth, references at 1e-4 rerun noise, ours at `engine_noise` from R1.

    The counts outputs y carry 1e-6 relative noise per run, ours `forward_noise` times that.
    """
    rng = np.random.default_rng(2)
    inputs = root / "inputs"
    inputs.mkdir(parents=True)
    meta = {
        "parameters": {"warning_threshold": 1e-3},
        "n_kept": N_ALL,
        "length": L,
        "window": [24, 36],
        "p256_rule": "synthetic",
        "truth_rule": "synthetic",
        "truth_kept_indices": TRUTH.tolist(),
        "G0": {"pairs": N * K, "bitwise_equal": N * K, "pass": True},
        "versions": {},
    }
    (inputs / "meta.json").write_text(json.dumps(meta))
    np.save(inputs / "truth.npy", TRUTH)

    m_true = rng.normal(size=(N, K, 4, L))
    h_true = rng.normal(size=(N, 4, W))
    y_true = rng.normal(size=(N, K + 1))
    # The reference-precision runs share one fixed "TF32" offset from the truth, 1e-3; the fp32 runs have none. So
    # F_tf is about 1e-3, F_rr about 1.4e-4 and F_hh about 1.4e-5.
    tf32 = {k: 1e-3 * rng.normal(size=v.shape) for k, v in (("m", m_true), ("hyp", h_true), ("y", y_true))}

    def noisy(scale: float, reference_class: bool, peaks=slice(None), y_scale: float = 1e-6):
        def jitter(x, key, s=scale):
            bias = tf32[key][peaks] if reference_class else 0.0
            return (x * (1 + bias + s * rng.normal(size=x.shape))).astype(np.float32)

        y = jitter(y_true[peaks], "y", y_scale)
        return {
            "m": jitter(m_true[peaks], "m"),
            "hyp": jitter(h_true[peaks], "hyp"),
            "deltas": np.abs(rng.normal(scale=1e-5, size=(len(y), K))).astype(np.float32),
            "y_x": y[:, 0].copy(),
            "y_r": y[:, 1:].copy(),
        }

    flags = {"torch.get_float32_matmul_precision()": "high"}
    oracle_extra = {"gpu_memory_attribution": {"max_allocated_bytes": 5e9, "max_reserved_bytes": 6e9}}
    oracle_extra["precision_flags"] = flags
    for name, spec in dv.ORACLE_RUNS.items():
        scale = 1e-4 if spec["tf32"] else 1e-5
        peaks = TRUTH if spec["peaks"] == "truth" else slice(None)
        _write_run(root, name, "oracle", spec, noisy(scale, spec["tf32"], peaks), oracle_extra)
    meta_ours = {"seqs_per_step": 4, "compile_s": 1.0, "steady_pairs_per_s": 1e3, "step_peak_bytes": 3e9, "oom": []}
    meta_ours["precision_flags"] = flags
    ours_extra = {"meta": meta_ours, "gpu_memory": {"max_allocated_bytes": 4e9, "max_reserved_bytes": 4e9}}
    for name, spec in dv.OURS_RUNS.items():
        if spec["peaks"] == "all":
            deltas = np.full((N_ALL, K), 1e-5, np.float32)
            deltas[0, :3] = 2e-3  # 3 of 160 pairs, 1.9%: above E8's 0.306%
            arrays = {"hyp": np.zeros((N_ALL, 4, W), np.float32), "deltas": deltas}
            arrays["y_x"], arrays["y_r"] = np.zeros(N_ALL, np.float32), np.zeros((N_ALL, K), np.float32)
        else:
            reference_class = spec["precision"] == "reference"
            arrays = noisy(
                (1e-4 if reference_class else 1e-5) * engine_noise, reference_class, y_scale=1e-6 * forward_noise
            )
        _write_run(root, name, "ours", spec, arrays, ours_extra)
    truth = noisy(0.0, False, TRUTH)
    truth = {k: np.asarray(v, np.float64) for k, v in truth.items()}
    truth["m"], truth["hyp"] = m_true[TRUTH], h_true[TRUTH]
    _write_run(root, dv.TRUTH_RUN, "truth", {"peaks": "truth"}, truth, {})


def _gates(root: Path) -> dict:
    rc = dv.main(["gates", "--out-dir", str(root)])
    report = json.loads((root / "report.json").read_text())
    assert rc == (0 if report["all_gates_pass"] else 1)
    assert (root / "report.tsv").read_text().startswith("gate\tcomparison")
    assert (root / "report.md").read_text().startswith("# cheri_dls GPU validation")
    return report


def test_gates_pass_an_engine_at_the_floor_and_read_e8_against_the_pilot(tmp_path):
    _synthetic(tmp_path, engine_noise=1.0)
    report = _gates(tmp_path)
    s = report["summary"]
    assert s["G0"] is True and s["E2"] is True
    assert set(report["floors"]) == {"F_rr", "F_hh", "F_tf"}
    assert report["floors"]["F_rr"]["raw"]["n"] == N * K and report["floors"]["F_rr"]["hyp"]["n"] == N
    assert s["E3"] in (True, "via E7(a)") and s["E4"] in (True, "via E7(a)")
    assert s["E5"] is True and s["E6"] is not False
    assert s["E8"] is False  # the synthetic P5000 run has 1.9% of its deltas above 1e-3
    assert report["distance_to_T64"]["T1"]["raw"]["n"] == len(TRUTH) * K
    assert s["E1"] is True and s["E10"] is True
    _check_revised_rows(report)


def _check_revised_rows(report: dict) -> None:
    """E1 (reference) and E10 (fp32) gate with the revised bounds; the original bounds are reported, not gated."""
    e1 = report["gates"]["E1"]
    floor = report["E1_context"]["max_abs_between_runs"]["R1p vs R1"]
    to_truth = report["E1_context"]["max_abs_to_T64_truth_peaks"]
    for run in ("T1", "T3"):
        rows = [r for r in e1 if r["comparison"].startswith(f"{run} y")]
        assert [r["bound"] for r in rows] == [2 * floor, 1.2 * to_truth["R1"]]
        assert rows[1]["measured"] == to_truth[run]
    assert [r["bound"] for r in e1 if r["comparison"].startswith("T2 y")] == [5e-5]
    hh = report["floors"]["F_hh"]["raw"]
    e10_fp32 = {r["statistic"]: r for r in report["gates"]["E10"] if r["comparison"].startswith("T2_S8")}
    assert set(e10_fp32) == {"q50", "q99"} and all(r["metric"] == "raw" for r in e10_fp32.values())
    assert e10_fp32["q50"]["bound"] == max(2 * hh["q50"], 1e-5) and e10_fp32["q99"]["bound"] == max(3 * hh["q99"], 1e-4)
    originals = {(r["gate"], r["comparison"]): r for r in report["original_bounds"]}
    assert set(originals) == {
        ("E1 original (reported only)", "T1 y vs R1 module y"),
        ("E1 original (reported only)", "T3 y vs R1 module y"),
        ("E10 original (reported only)", "T2_S8 vs T2 (fp32)"),
    }
    assert originals["E10 original (reported only)", "T2_S8 vs T2 (fp32)"]["bound"] == 1e-5
    for gate in ("E1", "E10"):
        assert not any("original" in r["gate"] for r in report["gates"][gate])
    revisions = report["revisions"]
    assert revisions["E1_reference"]["original"]["bound"] == 2e-4 and set(revisions["E1_reference"]["runs"]) == {
        "T1",
        "T3",
    }
    assert revisions["E10_fp32"]["original"]["bound"] == 1e-5


def test_the_original_bounds_of_revised_criteria_do_not_gate(tmp_path):
    """An engine at the floor of a synthetic directory whose fp32 floor F_hh is above 1e-5 per pair at p99: the original
    E10 bound (p99 <= 1e-5) fails, as it did on the real model, and the revised one passes; E10 passes."""
    _synthetic(tmp_path, engine_noise=1.0)
    report = _gates(tmp_path)
    original = report["revisions"]["E10_fp32"]["original"]
    assert original["pass"] is False and original["measured"] > 1e-5
    assert report["summary"]["E10"] is True
    md = (tmp_path / "report.md").read_text()
    assert "## Revised criteria" in md and "| E10 fp32 | T2_S8 vs T2, raw | q99 |" in md


def test_e1_fails_a_forward_far_above_the_rerun_noise(tmp_path):
    """Our counts outputs 10^4 x noisier than a stock rerun: E1 fails in reference mode on both of its clauses."""
    _synthetic(tmp_path, engine_noise=1.0, forward_noise=1e4)
    report = _gates(tmp_path)
    assert report["summary"]["E1"] is False
    failed = {(r["comparison"].split()[0], r["statistic"]) for r in report["gates"]["E1"] if r["pass"] is False}
    assert {("T1", "max abs"), ("T3", "max abs"), ("T1", "max abs to T64"), ("T3", "max abs to T64")} <= failed
    assert any(line.startswith("E1 failed") for line in report["open_issues"])


def test_e1_is_not_evaluated_without_the_stock_rerun(tmp_path):
    """Without R1' the floor-relative E1 cannot be computed: the verdict is None, not a pass."""
    _synthetic(tmp_path, engine_noise=1.0)
    for path in (tmp_path / "runs" / "R1p").iterdir():
        path.unlink()
    (tmp_path / "runs" / "R1p").rmdir()
    report = _gates(tmp_path)
    assert report["summary"]["E1"] is None and report["all_gates_pass"] is False


def test_gates_fail_an_engine_far_above_the_floor(tmp_path):
    _synthetic(tmp_path, engine_noise=300.0)  # 3e-2 relative noise: above every floor-relative limit and the caps
    report = _gates(tmp_path)
    s = report["summary"]
    assert s["E3"] is False and s["E4"] is False  # E7(a) cannot rescue it: it is 300x farther from the truth
    assert s["E5"] is False and s["E10"] is False
    assert any(r["gate"] == "E7a" and r["pass"] is False for r in report["gates"]["E7"])
    assert report["all_gates_pass"] is False
