"""The gate arithmetic of workflows/cherimoya/dls_validate.py, and its `gates` report on synthetic runs (CPU only).

The harness's runs need a GPU and the real model, and take tens of minutes; its gates need neither. These tests check
the pieces the verdicts rest on: the per-pair and per-peak rel-L2, the Pearson correlation, the floor-relative limits of
E3/E4/E6/E10, E8's bounds and E7(a)'s tie-breaker. They also run `gates` end to end on a synthetic validation directory,
with every run the report reads, and check that an engine with a real error fails while one at the floor passes. That
includes E1 and E10 as revised after round 2 (design/revisions.md): E1 in reference mode within 2x the stock forward's
rerun noise and 1.2x its distance to the fp64 truth, E10 in fp32 mode in E4's form against F_hh; and E3's precision-
policy check as revised after round 3, median <= max(0.3 F_tf(0.5), 1.5 F_rr(0.5)), which E6 shares. The original
bounds are reported but do not gate.
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
    # E3's policy check as revised after round 3: median <= max(0.3 F_tf(0.5), 1.5 F_rr(0.5)), either term binding
    b = dv.policy_bound(1e-3, 2.5e-4)
    assert b["bound"] == pytest.approx(3.75e-4) and b["binding"] == "1.5*F_rr(0.5)"
    assert b["tf_term"] == pytest.approx(3e-4) and b["rr_term"] == pytest.approx(3.75e-4)
    b = dv.policy_bound(1e-3, 1e-4)
    assert b["bound"] == pytest.approx(3e-4) and b["binding"] == "0.3*F_tf(0.5)"


def test_the_policy_row_gates_with_the_revised_bound_and_reports_the_original():
    """Between 0.3 F_tf(0.5) and 1.5 F_rr(0.5): the revised policy row passes, the original row fails, reported only."""
    floor = {m: {"q50": 2.5e-4, "q99": 1e-3} for m in ("raw", "hyp")}
    tf = {m: {"q50": 1e-3} for m in ("raw", "hyp")}
    cmp = {"a": "T1", "b": "R1", **{m: {"q50": 3.3e-4, "q99": 1e-3} for m in ("raw", "hyp")}}
    rows = dv._floor_rows("E3", cmp, floor, "F_rr", tf)
    policy = [r for r in rows if r["statistic"] == "q50 policy"]
    assert [r["metric"] for r in policy] == ["raw", "hyp"]
    for r in policy:
        assert r["bound"] == pytest.approx(3.75e-4) and r["pass"] is True and r["gate"] == "E3"
        assert r["rule"] == "max(0.3*F_tf(0.5), 1.5*F_rr(0.5))" and r["note"] == "binding: 1.5*F_rr(0.5)"
    originals = dv._policy_original_rows("E3", cmp, tf)
    for r in originals:
        assert r["gate"] == "E3 original (reported only)" and r["statistic"] == "q50 policy"
        assert r["bound"] == pytest.approx(3e-4) and r["pass"] is False
    # Above both terms the revised row fails too.
    cmp_far = {"a": "T1", "b": "R1", **{m: {"q50": 4e-4, "q99": 1e-3} for m in ("raw", "hyp")}}
    assert all(
        r["pass"] is False for r in dv._floor_rows("E3", cmp_far, floor, "F_rr", tf) if r["statistic"] == "q50 policy"
    )


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


def _synthetic(
    root: Path,
    engine_noise: float,
    forward_noise: float = 1.0,
    runs: list[str] | None = None,
    rerun_noise: float = 1e-4,
    model: str = "/data/amsc_x/fold_0/cherimoya.fold_0.torch",
    seed: int = 2,
) -> None:
    """A validation directory: the truth, references at `rerun_noise` (1e-4) rerun noise, ours at `engine_noise` from R1.

    The counts outputs y carry 1e-6 relative noise per run, ours `forward_noise` times that. `runs` limits the runs
    written (default: every run of plan "full"); every engine run records the S of its spec in raw() and run().
    """
    rng = np.random.default_rng(seed)
    wanted = set(dv.plan_runs("full") if runs is None else runs)
    inputs = root / "inputs"
    inputs.mkdir(parents=True)
    meta = {
        "parameters": {"warning_threshold": 1e-3, "model": model, "loci": ["/data/amsc_x/peaks.narrowPeak"]},
        "n_kept": N_ALL,
        "length": L,
        "window": [24, 36],
        "p256_rule": "synthetic",
        "truth_rule": "synthetic",
        "p256_first_last": [0, N_ALL - 1],
        "truth_kept_indices": TRUTH.tolist(),
        "G0": {"pairs": N * K, "bitwise_equal": N * K, "pass": True},
        "versions": {},
    }
    (inputs / "meta.json").write_text(json.dumps(meta))
    np.save(inputs / "truth.npy", TRUTH)
    idx = np.ones(N_ALL + 2, dtype=bool)
    idx[[3, 7]] = False  # 2 loci dropped: N_ALL kept out of N_ALL + 2 read
    np.save(inputs / "idx.npy", idx)

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
    gpu = {"env": {"CUDA_VISIBLE_DEVICES": "3"}, "versions": {"gpu": "NVIDIA H100 80GB HBM3"}}
    oracle_extra = {"gpu_memory_attribution": {"max_allocated_bytes": 5e9, "max_reserved_bytes": 6e9}, **gpu}
    oracle_extra["precision_flags"] = flags
    for name, spec in dv.ORACLE_RUNS.items():
        scale = rerun_noise if spec["tf32"] else 1e-5
        peaks = TRUTH if spec["peaks"] == "truth" else slice(None)
        arrays = noisy(scale, spec["tf32"], peaks)  # drawn whether or not it is written: every run sees the same noise
        if name in wanted:
            _write_run(root, name, "oracle", spec, arrays, oracle_extra)
    for name, spec in dv.OURS_RUNS.items():
        meta_ours = {"compile_s": 1.0, "steady_pairs_per_s": 1e3, "step_peak_bytes": 3e9, "oom": []}
        meta_ours.update(seqs_per_step=spec["S"], precision_flags=flags)
        ours_extra = {"meta": meta_ours, "gpu_memory": {"max_allocated_bytes": 4e9, "max_reserved_bytes": 4e9}, **gpu}
        if spec["peaks"] != "all":
            ours_extra["raw"] = {"seqs_per_step": spec["S"], "oom": []}
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
        if name in wanted:
            _write_run(root, name, "ours", spec, arrays, ours_extra)
    truth = noisy(0.0, False, TRUTH)
    truth = {k: np.asarray(v, np.float64) for k, v in truth.items()}
    truth["m"], truth["hyp"] = m_true[TRUTH], h_true[TRUTH]
    if dv.TRUTH_RUN in wanted:
        _write_run(root, dv.TRUTH_RUN, "truth", {"peaks": "truth"}, truth, {})


def _gates(root: Path, *extra: str) -> dict:
    rc = dv.main(["gates", "--out-dir", str(root), *extra])
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
    e10_fp32 = {r["statistic"]: r for r in report["gates"]["E10"] if r["comparison"].startswith("T2_S8 vs")}
    assert set(e10_fp32) == {"q50", "q99"} and all(r["metric"] == "raw" for r in e10_fp32.values())
    assert e10_fp32["q50"]["bound"] == max(2 * hh["q50"], 1e-5) and e10_fp32["q99"]["bound"] == max(3 * hh["q99"], 1e-4)
    originals = {(r["gate"], r["comparison"]): r for r in report["original_bounds"]}
    assert set(originals) == {
        ("E1 original (reported only)", "T1 y vs R1 module y"),
        ("E1 original (reported only)", "T3 y vs R1 module y"),
        ("E10 original (reported only)", "T2_S8 vs T2 (fp32)"),
        ("E3 original (reported only)", "T1 vs R1"),
        ("E6 original (reported only)", "T3 vs T1"),
    }
    _check_policy_rows(report, ("E3", "E6"))
    assert originals["E10 original (reported only)", "T2_S8 vs T2 (fp32)"]["bound"] == 1e-5
    for gate in ("E1", "E10"):
        assert not any("original" in r["gate"] for r in report["gates"][gate])
    revisions = report["revisions"]
    assert revisions["E1_reference"]["original"]["bound"] == 2e-4 and set(revisions["E1_reference"]["runs"]) == {
        "T1",
        "T3",
    }
    assert revisions["E10_fp32"]["original"]["bound"] == 1e-5


def _check_policy_rows(report: dict, gates: tuple[str, ...]) -> None:
    """E3's (and E6's) policy row gates with max(0.3 F_tf(0.5), 1.5 F_rr(0.5)); the original 0.3 F_tf(0.5) row is
    reported only, under its own gate name, and the revision record has both."""
    floors = report["floors"]
    rec = report["revisions"]["E3_policy"]
    assert set(rec["checks"]) == set(gates)
    for gate in gates:
        for metric in ("raw", "hyp"):
            tf, rr = floors["F_tf"][metric]["q50"], floors["F_rr"][metric]["q50"]
            row = next(r for r in report["gates"][gate] if r["metric"] == metric and r["statistic"] == "q50 policy")
            assert row["bound"] == pytest.approx(max(0.3 * tf, 1.5 * rr))
            assert row["rule"] == "max(0.3*F_tf(0.5), 1.5*F_rr(0.5))"
            original = next(
                r
                for r in report["original_bounds"]
                if r["gate"] == f"{gate} original (reported only)" and r["metric"] == metric
            )
            assert original["bound"] == pytest.approx(0.3 * tf) and original["measured"] == row["measured"]
            check = rec["checks"][gate][metric]
            assert check["bound"] == row["bound"] and check["pass"] is row["pass"]
            assert check["original_bound"] == original["bound"] and check["original_pass"] is original["pass"]
            assert check["binding"] == ("1.5*F_rr(0.5)" if 1.5 * rr > 0.3 * tf else "0.3*F_tf(0.5)")
        assert not any("original" in r["gate"] for r in report["gates"][gate])


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


CHECKPOINT_RUNS = dv.plan_runs("checkpoint")


def test_the_checkpoint_plan_gates_only_its_own_gates(tmp_path):
    """Plan "checkpoint" on a directory with only its runs: the verdict covers G0, E1-E5 and E10 in reference mode and
    passes; E6 and E8 are out of scope, not failures. E10 checks that T1 and T1_S8 ran at S = 4 and 8."""
    _synthetic(tmp_path, engine_noise=1.0, runs=CHECKPOINT_RUNS)
    report = _gates(tmp_path, "--plan", "checkpoint")
    assert list(report["summary"]) == ["G0", "E1", "E2", "E3", "E4", "E5", "E10"]
    assert report["plan"]["out_of_scope"] == ["E6", "E8"] and report["missing_runs"] == []
    assert report["all_gates_pass"] is True and report["E8_reference"] is None
    assert all(v in (True, "via E7(a)") for v in report["summary"].values())
    steps = {r["comparison"]: r for r in report["gates"]["E10"] if r["metric"] == "seqs_per_step"}
    assert {k: (v["measured"], v["bound"], v["pass"]) for k, v in steps.items()} == {
        "T1 steps as run": (4, 4, True),
        "T1_S8 steps as run": (8, 8, True),
    }
    assert not any(r["comparison"].startswith("T2_S8") for r in report["gates"]["E10"])  # fp32 mode: out of scope
    e1 = {r["comparison"] for r in report["gates"]["E1"]}
    assert e1 == {"T1 y vs R1 module y", "T1 y vs T64 y (truth peaks), against 1.2 x R1's", "T2 y vs R2 module y"}
    md = (tmp_path / "report.md").read_text()
    assert "Out of the scope of plan 'checkpoint', not in the verdict: E6, E8." in md
    assert "## E3 precision policy: margins" in md and "## E7(a) context" in md
    assert not any("E8" in issue for issue in report["open_issues"])


def test_the_full_plan_does_not_pass_the_checkpoint_runs(tmp_path):
    """The same directory under the default plan: E6 and E8 lack their runs, so they are not run and the verdict fails.
    The default plan is the one rounds 2 and 3 ran."""
    _synthetic(tmp_path, engine_noise=1.0, runs=CHECKPOINT_RUNS)
    report = _gates(tmp_path)
    assert report["plan"]["name"] == "full" and report["plan"]["out_of_scope"] == []
    assert report["summary"]["E6"] is None and report["summary"]["E8"] is None
    assert report["not_run"]["E6"] == ["T3"] and report["not_run"]["E8"] == ["P5000_ref"]
    assert report["all_gates_pass"] is False
    assert {"T1p", "T3", "T2_S8", "P5000_ref", "P5000_fp32", "R1_8"} == set(report["missing_runs"])


def test_a_planned_run_that_is_missing_leaves_its_gates_not_run(tmp_path):
    """Plan "checkpoint" without T1_S8: E10 and E5 read it, so both are not run rather than passing on what is left."""
    _synthetic(tmp_path, engine_noise=1.0, runs=[r for r in CHECKPOINT_RUNS if r != "T1_S8"])
    report = _gates(tmp_path, "--plan", "checkpoint")
    assert report["missing_runs"] == ["T1_S8"]
    assert report["summary"]["E10"] is None and report["summary"]["E5"] is None
    assert report["summary"]["E3"] in (True, "via E7(a)") and report["all_gates_pass"] is False
    assert any(issue.startswith("E10 was not evaluated (missing runs: ['T1_S8'])") for issue in report["open_issues"])


def test_e10_fails_a_run_whose_step_was_halved(tmp_path):
    """T1_S8 out of memory in run() and halved to S = 4: E10 would compare T1 with itself, so it fails."""
    _synthetic(tmp_path, engine_noise=1.0, runs=CHECKPOINT_RUNS)
    path = tmp_path / "runs" / "T1_S8" / "record.json"
    record = json.loads(path.read_text())
    record["meta"].update(seqs_per_step=4, oom=[{"step": "warm-up", "from": 8, "to": 4}])
    path.write_text(json.dumps(record))
    report = _gates(tmp_path, "--plan", "checkpoint")
    assert report["summary"]["E10"] is False
    row = next(r for r in report["gates"]["E10"] if r["comparison"] == "T1_S8 steps as run")
    assert (row["measured"], row["bound"], row["pass"]) == (4, 8, False) and row["note"].startswith("1 halvings")
    assert any(issue.startswith("E10 failed: T1_S8 steps as run") for issue in report["open_issues"])


def test_e3_policy_margins_and_e7_context(tmp_path):
    """E3's policy margins are T1 vs R1's median and F_rr's median over 0.3 F_tf(0.5); E7(a)'s ratios are
    d(ours, T64) / d(R, T64). Both are reported whether or not E3 and E4 pass."""
    _synthetic(tmp_path, engine_noise=1.0, runs=CHECKPOINT_RUNS)
    report = _gates(tmp_path, "--plan", "checkpoint")
    floors, policy = report["floors"], report["E3_policy"]
    assert set(policy) == {"raw", "hyp"}
    for metric, v in policy.items():
        tf, rr = floors["F_tf"][metric]["q50"], floors["F_rr"][metric]["q50"]
        bound = max(0.3 * tf, 1.5 * rr)
        row = next(r for r in report["gates"]["E3"] if r["metric"] == metric and r["statistic"] == "q50 policy")
        assert v["bound"] == pytest.approx(bound) and v["ours_q50"] == row["measured"]
        assert v["ours_ratio"] == pytest.approx(row["measured"] / bound)
        assert v["stock_rerun_ratio"] == pytest.approx(rr / bound)
        assert v["pass"] is (v["ours_ratio"] <= 1)
        assert v["original_bound"] == pytest.approx(0.3 * tf) and v["F_tf_q50"] == tf
        assert v["ours_ratio_original"] == pytest.approx(row["measured"] / (0.3 * tf))
        assert v["stock_rerun_ratio_original"] == pytest.approx(rr / (0.3 * tf))
        assert v["ours_over_F_rr"] == pytest.approx(row["measured"] / rr)
        assert v["rr_term"] == pytest.approx(1.5 * rr) and v["pass_original"] is (v["ours_ratio_original"] <= 1)
    assert set(report["E7_context"]) == {"E3", "E4"}
    for gate, (ours, ref) in {"E3": ("T1", "R1"), "E4": ("T2", "R2")}.items():
        for metric in ("raw", "hyp"):
            v = report["E7_context"][gate]["metrics"][metric]
            d_ours, d_ref = report["distance_to_T64"][ours][metric], report["distance_to_T64"][ref][metric]
            assert v["ratio"]["q50"] == pytest.approx(d_ours["q50"] / d_ref["q50"])
            assert v["ratio"]["max"] == pytest.approx(d_ours["max"] / d_ref["max"])
            assert v["within"] is (max(v["ratio"].values()) <= 1.2)


def test_e3_policy_passes_at_the_stock_rerun_noise_where_the_original_bound_failed(tmp_path):
    """References whose rerun noise is a large share of F_tf (F_rr / F_tf about 0.4): an engine at about the rerun
    noise fails the original 0.3 F_tf(0.5), as 3 of the 4 checkpoints did, and passes the revised bound, which the
    1.5 F_rr(0.5) term sets. E3 passes directly, and the original rows stay out of the verdict."""
    _synthetic(tmp_path, engine_noise=2.0, rerun_noise=3e-4, runs=CHECKPOINT_RUNS)
    report = _gates(tmp_path, "--plan", "checkpoint")
    _check_policy_rows(report, ("E3",))
    for metric, v in report["E3_policy"].items():
        assert v["binding"] == "1.5*F_rr(0.5)" and v["pass"] is True and v["pass_original"] is False, (metric, v)
    assert report["summary"]["E3"] is True and report["all_gates_pass"] is True
    md = (tmp_path / "report.md").read_text()
    assert "| E3 policy | T1 vs R1, raw | q50 |" in md and "(1.5*F_rr(0.5) binds)" in md
    tsv = (tmp_path / "report.tsv").read_text()
    assert "E3 original (reported only)\tT1 vs R1\traw\tq50 policy" in tsv


def test_e3_policy_reads_a_report_without_the_new_fields():
    """summarize reads the pilot's report, written before E3_policy existed: the margins come from E3's rows."""
    floors = {"F_rr": {"raw": {"q50": 1.59e-4}, "hyp": {"q50": 7.2e-5}}, "F_tf": {"raw": {"q50": 6.32e-4}}}
    rows = [
        dv._row("E3", "T1 vs R1", "raw", "q50", 1.7e-4, 3.2e-4, "x"),
        dv._row("E3", "T1 vs R1", "raw", "q50 policy", 1.773e-4, 0.3 * 6.32e-4, "0.3*F_tf(0.5)"),
    ]
    policy = dv.e3_policy(floors, rows)
    assert set(policy) == {"raw"}
    assert policy["raw"]["ours_ratio"] == pytest.approx(1.773e-4 / 1.896e-4)
    assert policy["raw"]["stock_rerun_ratio"] == pytest.approx(1.59e-4 / 1.896e-4) and policy["raw"]["pass"] is True


def _fold(tmp_path: Path, name: str = "cherimoya.fold_2") -> tuple[Path, Path]:
    fold = tmp_path / "amsc_y" / "fold_2"
    fold.mkdir(parents=True)
    (fold / "cherimoya.fold_2.torch").write_bytes(b"weights")
    peaks = tmp_path / "amsc_y_peaks.narrowPeak"
    peaks.write_text("chr1\t100\t1100\n")
    (fold / "fit.json").write_text(json.dumps({"name": name, "loci": [str(peaks)], "random_state": 1236}))
    return fold, peaks


BASE = {
    "model": "/pilot/fold_0/cherimoya.fold_0.torch",
    "loci": ["/pilot/peaks_5000.narrowPeak"],
    "chroms": dv.ALL_CHROMS,
    "batch_size": 64,
    "random_state": 0,
    "ohe_filename": "/pilot/a.ohe.npz",
    "attr_filename": "/pilot/a.attr.npz",
    "idx_filename": "/pilot/a.idx.npy",
}


def test_checkpoint_parameters_change_only_the_model_the_loci_and_the_outputs(tmp_path):
    fold, peaks = _fold(tmp_path)
    out = tmp_path / "out"
    p = dv.checkpoint_parameters(BASE, fold, out)
    assert p["model"] == str(fold / "cherimoya.fold_2.torch") and p["loci"] == [str(peaks)]
    assert p["ohe_filename"] == str(out / "stock_unused" / "attr.ohe.npz")
    assert p["idx_filename"] == str(out / "stock_unused" / "attr.idx.npy")
    changed = {k for k in BASE if p[k] != BASE[k]}
    assert changed == {"model", "loci", "ohe_filename", "attr_filename", "idx_filename"}
    assert list(p) == list(BASE) and len(p["chroms"]) == 24
    with pytest.raises(ValueError, match="24 chromosomes"):
        dv.checkpoint_parameters(dict(BASE, chroms=dv.ALL_CHROMS[1:]), fold, out)
    (fold / "cherimoya.fold_2.torch").unlink()
    with pytest.raises(FileNotFoundError, match="cherimoya.fold_2.torch"):
        dv.checkpoint_parameters(BASE, fold, out)


def test_checkpoint_parameters_refuse_a_fit_json_that_names_another_model(tmp_path):
    fold, _ = _fold(tmp_path, name="cherimoya.fold_9")
    with pytest.raises(ValueError, match="cherimoya.fold_9"):
        dv.checkpoint_parameters(BASE, fold, tmp_path / "out")


def test_checkpoint_json_writes_once_and_refuses_to_change_a_used_json(tmp_path):
    fold, _ = _fold(tmp_path)
    base = tmp_path / "base.json"
    base.write_text(json.dumps(BASE))
    out = tmp_path / "out"
    argv = ["checkpoint-json", "--base", str(base), "--fold-dir", str(fold), "--out-dir", str(out)]
    assert dv.main(argv) == 0
    text = (out / "attribute.json").read_text()
    assert json.loads(text) == dv.checkpoint_parameters(BASE, fold, out) and text.endswith("}\n")
    assert dv.main(argv) == 0  # the same JSON again: kept
    base.write_text(json.dumps(dict(BASE, random_state=1)))
    assert dv.main(argv) == 1 and (out / "attribute.json").read_text() == text


def test_p256_table_maps_kept_indices_to_the_loci_rows(tmp_path):
    """Kept index -> row of the loci extract_loci read (np.flatnonzero(idx)) -> the file's coordinates."""
    pytest.importorskip("tangermeme")
    peaks = tmp_path / "peaks.narrowPeak"
    rows = [("chr1", 100 * i, 100 * i + 50) for i in range(10)]
    peaks.write_text("".join(f"{c}\t{s}\t{e}\tpeak\t0\t.\t1\t1\t1\t25\n" for c, s, e in rows))
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "meta.json").write_text(json.dumps({"parameters": {"loci": [str(peaks)], "chroms": dv.ALL_CHROMS}}))
    idx = np.array([True, False, True, True, False, True, True, True, False, True])  # 7 kept
    np.save(inputs / "idx.npy", idx)
    np.save(inputs / "p256.npy", np.array([0, 3, 6]))  # kept 0, 3, 6 -> rows 0, 5, 9
    np.save(inputs / "truth.npy", np.array([0, 2]))
    assert dv.main(["p256-table", "--out-dir", str(tmp_path)]) == 0
    lines = (inputs / "p256.tsv").read_text().splitlines()
    assert lines == [
        "p256\tkept_index\tloci_row\tchrom\tstart\tend\ttruth",
        "0\t0\t0\tchr1\t0\t50\t1",
        "1\t3\t5\tchr1\t500\t550\t0",
        "2\t6\t9\tchr1\t900\t950\t1",
    ]
    np.save(inputs / "idx.npy", idx[:-1])
    with pytest.raises(RuntimeError, match="mask has 9 entries but the loci have 10 rows"):
        dv.write_p256_table(tmp_path)


def test_summarize_puts_checkpoints_side_by_side(tmp_path):
    """Two checkpoints under plan "checkpoint" and a full-plan reference: the verdicts, floors, E3's policy margins
    (ours and the stock rerun), E7(a)'s ratios and every gate check's measured value against its bound."""
    dirs = []
    for i, ds in enumerate(("amsc_a", "amsc_b")):
        d = tmp_path / f"{ds}_fold_{i}"
        _synthetic(d, engine_noise=1.0, runs=CHECKPOINT_RUNS, model=f"/data/{ds}/fold_{i}/cherimoya.fold_{i}.torch")
        _gates(d, "--plan", "checkpoint")
        dirs.append(d)
    ref = tmp_path / "pilot"
    _synthetic(ref, engine_noise=1.0, model="/data/amsc_pilot/fold_0/cherimoya.fold_0.torch")
    _gates(ref)
    # The pilot's real report predates plans, the margins and E10's step rows: strip them, as summarize must read it
    old = json.loads((ref / "report.json").read_text())
    for key in ("plan", "not_run", "E3_policy", "E7_context", "harness"):
        del old[key]
    old["gates"]["E10"] = [r for r in old["gates"]["E10"] if r["metric"] != "seqs_per_step"]
    (ref / "report.json").write_text(json.dumps(old))
    out = tmp_path / "summary" / "summary.md"
    rc = dv.main(["summarize", "--out", str(out), *map(str, dirs), "--reference-dir", str(ref), "--note", "a note"])
    assert rc == 0
    md = out.read_text()
    summary = json.loads(out.with_suffix(".json").read_text())
    assert [e["label"] for e in summary["checkpoints"]] == ["amsc_a fold_0", "amsc_b fold_1"]
    assert summary["reference"]["label"] == "amsc_pilot fold_0" and summary["reference"]["plan"] == "full"
    a = summary["checkpoints"][0]
    assert (
        a["n_loci_read"] == N_ALL + 2
        and a["n_kept"] == N_ALL
        and a["gpu_runs"] == {"3 (NVIDIA H100 80GB HBM3)": ["R1", "R1p", "R2", "R2p", "T1", "T2", "T1_S8"]}
    )
    report_a = json.loads((dirs[0] / "report.json").read_text())
    assert a["E3_policy"] == report_a["E3_policy"] and a["summary"] == report_a["summary"]
    raw = a["E3_policy"]["raw"]
    assert (
        f"| amsc_a fold_0 | raw | {dv._fmt(raw['bound'])} | {raw['binding']} | {dv._fmt(raw['ours_q50'])} | "
        f"{raw['ours_ratio']:.3f} | {raw['ours_over_F_rr']:.3f} | {dv._fmt(raw['stock_rerun_q50'])} | "
        f"{raw['stock_rerun_ratio']:.3f} | {dv._fmt(raw['original_bound'])} | {raw['ours_ratio_original']:.3f} | "
        f"{raw['stock_rerun_ratio_original']:.3f} | {raw['pass']} |" in md
    )
    assert "Against the original bound 0.3 F_tf(0.5), reported only: ours" in md
    f_rr = report_a["floors"]["F_rr"]["raw"]
    assert f"| amsc_a fold_0 | F_rr | {dv._fmt(f_rr['q50'])} | {dv._fmt(f_rr['q99'])} | {dv._fmt(f_rr['max'])} |" in md
    verdicts = [line for line in md.splitlines() if line.startswith("| amsc_a fold_0 | PASS | PASS | PASS |")]
    assert len(verdicts) == 1 and verdicts[0].endswith("| out of scope | out of scope | PASS | True |")
    pilot = next(line for line in md.splitlines() if line.startswith("| amsc_pilot fold_0 (reference) | PASS |"))
    assert pilot.endswith("| FAIL | PASS | False |")  # the synthetic P5000 fails E8; plan full has E6 and E8
    assert summary["reference"]["E3_policy"] == summary["checkpoints"][0]["E3_policy"]  # same synthetic numbers
    # one row per gate check of the checkpoints, the reference's value in the last column
    e10 = next(line for line in md.splitlines() if line.startswith("| E10 | T1_S8 steps as run |"))
    assert e10.count("8 == 8, pass") == 2 and e10.endswith("|  |")  # the old report has no step rows
    g0 = next(line for line in md.splitlines() if line.startswith("| G0 |"))
    assert g0.count(f"{N * K} == {N * K}, pass") == 3
    assert "## Notes" in md and "- a note" in md
    assert "E3 policy margin, raw: ours / bound" in md
