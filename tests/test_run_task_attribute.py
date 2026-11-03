"""run_task.sh's attribute stage (CPU only), with a fake `pixi`, a fake attribute_fast.py and a fake `cherimoya`.

The fake engine writes the files the real one writes (test_attr_status._write_outputs), and the fake `cherimoya
attribute` the three the stock command writes, so the stage's waiting, skipping, refusing, cleaning and exit codes
are checked without a GPU, for both engines (CHERI_ATTR_ENGINE=rvi|cherimoya).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")

TESTS = Path(__file__).resolve().parent
sys.path[:0] = [str(TESTS.parent / "workflows" / d) for d in ("train", "attribution", "downstream")]
sys.path.insert(0, str(TESTS))

import attr_status as st  # noqa: E402
from test_attr_status import WORKFLOWS, _engine, _fold, _refit  # noqa: E402

FAKE_PIXI = """#!/bin/bash
# pixi run --frozen --manifest-path M -e ENV cmd... -> cmd..., with python = the test's interpreter
echo "$*" >> "${FAKE_LOG}"
shift
while (( $# )); do
    case "$1" in --frozen) shift ;; --manifest-path|-e) shift 2 ;; *) break ;; esac
done
if [[ "$1" == python ]]; then shift; exec "${FAKE_PYTHON}" "$@"; fi
exec "$@"
"""

FAKE_ENGINE = """import json, os, sys
sys.path.insert(0, os.environ["FAKE_TESTS"])
import numpy as np
from pathlib import Path
import test_attr_status as t
params = json.loads(Path(sys.argv[sys.argv.index("-p") + 1]).read_text())
with open(os.environ["FAKE_LOG"], "a") as fh:
    fh.write("engine " + " ".join(sys.argv[1:]) + " threads=" + os.environ.get("TORCHINDUCTOR_COMPILE_THREADS", "") + "\\n")
fail = os.environ.get("FAKE_AUDIT_FAIL") == "1"
t._write_outputs(params, Path(__file__).parent, meta_extra={"audit": {"pass": not fail}})
sys.exit(3 if fail else 0)
"""

FAKE_CHERIMOYA = """#!/bin/bash
# cherimoya attribute -p attribute.json: the stock command's three files (stock_record.py writes the record)
exec "${FAKE_PYTHON}" - "$@" <<'PY'
import json, os, sys
sys.path.insert(0, os.environ["FAKE_TESTS"])
import numpy as np
from pathlib import Path
import test_attr_status as t
assert sys.argv[1] == "attribute", sys.argv
params = json.loads(Path(sys.argv[sys.argv.index("-p") + 1]).read_text())
with open(os.environ["FAKE_LOG"], "a") as fh:
    fh.write("cherimoya " + " ".join(sys.argv[1:]) + "\\n")
idx = np.ones(t.N_ROWS, dtype=bool)
idx[list(t.DROPPED)] = False
attr = np.ones((t.KEPT, 4, t.W), np.float32)
if os.environ.get("FAKE_NAN") == "1":
    attr[0, 0, 0] = np.nan
np.savez_compressed(params["ohe_filename"], np.zeros((t.KEPT, 4, t.W), np.int8))
np.savez_compressed(params["attr_filename"], attr)
np.save(params["idx_filename"], idx)
if os.environ.get("FAKE_FALLBACK") == "1":
    print("x.py:1: RuntimeWarning: deep_lift_shap is using the default path because the fast path disagreed "
          "with tangermeme's on the first sequences. Pass fast=False to skip the fast path.", file=sys.stderr)
print("x.py:2: RuntimeWarning: 1 convergence deltas above 0.001", file=sys.stderr)
sys.exit(int(os.environ.get("FAKE_EXIT", "0")))
PY
"""


@pytest.fixture
def stage(tmp_path):
    """Run run_task.sh <fold> attribute with the fakes; returns (run, fold, params, log)."""
    if shutil.which("flock") is None:
        pytest.skip("no flock")
    fold, params = _fold(tmp_path, outputs=False, pin=False)
    engine = tmp_path / "engine"
    (engine / "attribute_fast.py").write_text(FAKE_ENGINE)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, text in (("pixi", FAKE_PIXI), ("cherimoya", FAKE_CHERIMOYA)):
        (bin_dir / name).write_text(text)
        (bin_dir / name).chmod(0o755)
    log = tmp_path / "calls.log"
    env = dict(
        os.environ,
        PATH=f"{bin_dir}:{os.environ['PATH']}",
        FAKE_LOG=str(log),
        FAKE_PYTHON=sys.executable,
        FAKE_TESTS=str(TESTS),
        CHERI_ATTRIBUTE_FAST=str(engine / "attribute_fast.py"),
        CHERI_PIXI_MANIFEST="/main/checkout/pixi.toml",
        CUDA_VISIBLE_DEVICES="",
        TMPDIR=str(tmp_path / "tmp"),
    )
    for knob in ("CHERI_ATTR_ARGS", "CHERI_ATTR_ENGINE", "CHERI_PIXI_ENV", "CHERI_ATTR_PIXI_ENV"):
        env.pop(knob, None)

    def run(**extra):
        return subprocess.run(
            ["bash", str(WORKFLOWS / "train" / "run_task.sh"), str(fold), "attribute"],
            env={**env, **extra},
            capture_output=True,
            text=True,
        )

    return run, fold, params, log


def test_stage_waits_for_the_fit_and_refuses_an_unpinned_engine(stage, tmp_path):
    run, fold, params, log = stage
    performance = fold / "cherimoya.fold_0.test.performance.tsv"
    performance.unlink()
    r = run()
    assert r.returncode == 3 and "the fit is not done; attribute waits" in r.stdout
    performance.write_text("x\n")
    r = run()
    assert r.returncode == 1 and "refusing to attribute with an engine other than the pinned one" in r.stderr
    assert "no engine pin" in r.stderr and not Path(params["idx_filename"]).exists()
    assert "--manifest-path /main/checkout/pixi.toml -e cherimoya" in log.read_text()


def test_stage_runs_skips_and_keeps_a_failed_audit(stage, tmp_path):
    run, fold, params, log = stage
    assert st.write_pin(tmp_path / "engine", tmp_path / "root" / st.PIN_NAME)[0] == "written"
    r = run(CHERI_ATTR_ARGS="--seqs-per-step 4 --audit 8")
    assert r.returncode == 0, r.stderr
    assert "attribute done" in r.stdout and st.check_fold(fold) == ("done", [])
    engine_calls = [line for line in log.read_text().splitlines() if line.startswith("engine ")]
    assert engine_calls == ["engine -p attribute.json --seqs-per-step 4 --audit 8 threads=1"]
    # the engine refuses Cherimoya 0.3.1, so it runs in the env that keeps 0.3.0; the status checks do not
    assert "-e cherimoya-030 python " + str(tmp_path / "engine" / "attribute_fast.py") in log.read_text()
    assert "-e cherimoya python " in log.read_text()
    assert (fold / "attribute.time").read_text().count("\t") == 1
    r = run()
    assert r.returncode == 0 and "attribute done, skipped" in r.stdout
    assert len([line for line in log.read_text().splitlines() if line.startswith("engine ")]) == 1

    # a refit: the attribution is stale, so it is removed and redone
    _refit(params)
    assert st.check_fold(fold)[0] == "stale"
    r = run()
    assert r.returncode == 0 and "attribute (stale:" in r.stdout and st.check_fold(fold)[0] == "done"

    # the audit fails: exit 1, outputs kept, and a later run leaves the fold alone until asked
    _refit(params)
    r = run(FAKE_AUDIT_FAIL="1")
    assert r.returncode == 1 and "audit failed; the outputs are kept for inspection" in r.stderr
    assert st.check_fold(fold)[0] == "failed" and Path(params["attr_filename"]).exists()
    n_calls = len([line for line in log.read_text().splitlines() if line.startswith("engine ")])
    r = run()
    assert r.returncode == 1 and "an earlier attribution failed and is kept for inspection" in r.stderr
    assert len([line for line in log.read_text().splitlines() if line.startswith("engine ")]) == n_calls
    r = run(CHERI_ATTR_RERUN_FAILED="1")
    assert r.returncode == 0 and st.check_fold(fold)[0] == "done"


def test_stage_refuses_an_engine_that_differs_from_the_pin(stage, tmp_path):
    run, fold, params, log = stage
    other = _engine(tmp_path / "pinned_engine", tag="v0")
    assert st.write_pin(other, tmp_path / "root" / st.PIN_NAME)[0] == "written"
    r = run()
    assert r.returncode == 1 and "attribute_fast.py in" in r.stderr and not Path(params["idx_filename"]).exists()


def _stock_record(params: dict) -> dict:
    import json

    return json.loads(st.fold_files(params)["stock"].read_text())


def test_the_cherimoya_engine_needs_no_pin_and_writes_the_stock_record(stage, tmp_path):
    run, fold, params, log = stage  # the fixture's fold has no pin: the rvi engine would refuse it
    r = run(CHERI_ATTR_ENGINE="cherimoya")
    assert r.returncode == 0, r.stderr
    assert "with cherimoya" in r.stdout and "attribute done" in r.stdout and "(fast path;" in r.stdout
    assert st.check_fold(fold) == ("done", [])
    record = _stock_record(params)
    assert (record["engine"], record["tool"], record["exit"], record["attr_finite"]) == (
        "stock",
        "cherimoya attribute",
        0,
        True,
    )
    assert record["fast_path"] is True and record["fallback"] is None and record["convergence_warnings"] == 1
    assert record["parameters"] == __import__("json").loads((fold / "attribute.json").read_text())
    calls = log.read_text()
    assert "-e cherimoya cherimoya attribute -p attribute.json" in calls and "cherimoya-030" not in calls
    assert (fold / "attribute.log").exists() and (fold / "attribute.time").read_text().count("\t") == 1
    r = run(CHERI_ATTR_ENGINE="cherimoya")
    assert r.returncode == 0 and "attribute done, skipped" in r.stdout
    fake_runs = [line for line in log.read_text().splitlines() if line.startswith("cherimoya attribute")]
    assert len(fake_runs) == 1  # the rerun ran nothing


def test_the_cherimoya_engine_records_a_fallback_and_refuses_non_finite_scores(stage, tmp_path):
    run, fold, params, log = stage
    r = run(CHERI_ATTR_ENGINE="cherimoya", FAKE_FALLBACK="1")
    assert r.returncode == 0, r.stderr
    assert "FELL BACK to tangermeme's path" in r.stdout and st.check_fold(fold)[0] == "done"
    record = _stock_record(params)
    assert record["fast_path"] is False and "disagreed with tangermeme's" in record["fallback"]

    _refit(params)
    r = run(CHERI_ATTR_ENGINE="cherimoya", FAKE_NAN="1")
    assert r.returncode == 1 and "non-finite attributions" in r.stderr and "stock_record.py refused" in r.stderr
    assert not st.fold_files(params)["stock"].exists() and st.check_fold(fold)[0] == "missing"

    r = run(CHERI_ATTR_ENGINE="cherimoya", FAKE_EXIT="2")
    assert r.returncode == 1 and "cherimoya attribute failed (exit 2)" in r.stderr
    assert not st.fold_files(params)["stock"].exists()


def test_an_unknown_engine_is_refused(stage):
    run, fold, params, log = stage
    r = run(CHERI_ATTR_ENGINE="other")
    assert r.returncode == 1 and "CHERI_ATTR_ENGINE is other, not rvi or cherimoya" in r.stderr
