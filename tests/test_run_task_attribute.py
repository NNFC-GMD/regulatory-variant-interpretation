"""run_task.sh's attribute stage (CPU only), with a fake `pixi` and a fake attribute_fast.py.

The fake engine writes the files the real one writes (test_attr_status._write_outputs), so the stage's waiting,
skipping, refusing, cleaning and exit codes are checked without a GPU.
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
sys.path.insert(0, str(TESTS.parent / "workflows" / "cherimoya"))
sys.path.insert(0, str(TESTS))

import attr_status as st  # noqa: E402
from test_attr_status import CHERIMOYA, _engine, _fold, _refit  # noqa: E402

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
    (bin_dir / "pixi").write_text(FAKE_PIXI)
    (bin_dir / "pixi").chmod(0o755)
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
    env.pop("CHERI_ATTR_ARGS", None)

    def run(**extra):
        return subprocess.run(
            ["bash", str(CHERIMOYA / "run_task.sh"), str(fold), "attribute"],
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
