"""compare.sh, the comparison's driver (CPU only): levels, units, the record, exit codes and signals.

The level scripts are the real ones, run by a fake `pixi` with this test's interpreter (compare_fixtures.fake_bin);
the inputs are synthetic. Every run has its own SLURM_JOB_ID, so its node-local scratch is its own.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")
pytest.importorskip("matplotlib")

TESTS = Path(__file__).resolve().parent
sys.path[:0] = [
    str(TESTS),
    str(TESTS.parent / "workflows" / "cherimoya"),
    str(TESTS.parent / "workflows" / "cherimoya" / "compare"),
]

from compare_fixtures import COMPARE, compare_env, make_performance_root  # noqa: E402

SCRIPT = COMPARE / "compare.sh"


def compare_sh(env: dict, *args: str, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SCRIPT), *args], env=env, capture_output=True, text=True, timeout=timeout)


def records(env: dict) -> list[list[str]]:
    path = Path(env["BOX_DIR"]) / "compare_steps.tsv"
    lines = path.read_text().splitlines()
    assert lines[0].split("\t") == ["level", "unit", "exit", "seconds", "max_rss_kb", "started", "note"]
    return [line.split("\t") for line in lines[1:]]


def processes_of(env: dict, pattern: str) -> list[str]:
    """The pids of this run's processes (by its SLURM_JOB_ID) whose command line matches the pattern."""
    out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout.split()
    mine = []
    for pid in out:
        try:
            if env["SLURM_JOB_ID"].encode() in Path(f"/proc/{pid}/environ").read_bytes():
                mine.append(pid)
        except OSError:
            continue
    return mine


@pytest.fixture
def perf(tmp_path):
    data = make_performance_root(tmp_path)
    env = compare_env(tmp_path, data["root"])
    return data, env


def test_level_1_runs_records_and_skips_when_done(perf, tmp_path):
    data, env = perf
    p = compare_sh(env, "1", "--datasets", str(data["list"]))
    assert p.returncode == 0, p.stdout + p.stderr
    rows = records(env)
    assert len(rows) == 1
    level, unit, rc, seconds, rss, started, note = rows[0]
    assert (level, unit, rc) == ("1", "performance", "0")
    assert rss.isdigit() and int(rss) > 0 and seconds.isdigit()
    assert note.startswith("15 folds, 3 datasets: counts Pearson")
    log = Path(env["COMPARISON_ROOT"]) / "logs" / "level1" / "performance.log"
    assert log.read_text().startswith("==== [")
    assert (Path(env["COMPARISON_ROOT"]) / "performance" / "statistics.tsv").is_file()
    assert "1 unit(s) done, 0 waiting for inputs, 0 failed" in p.stdout
    fake = Path(env["FAKE_LOG"]).read_text()
    assert "-e cherimoya python" in fake and "compare_performance.py --datasets" in fake
    p = compare_sh(env, "1", "--datasets", str(data["list"]))
    assert p.returncode == 0
    assert records(env)[-1][6] == "up to date"
    p = compare_sh(env, "1", "--datasets", str(data["list"]), "--force")
    assert p.returncode == 0
    assert records(env)[-1][6].startswith("15 folds")
    # a unit's note comes from this run's lines only, never an earlier run's
    assert len(records(env)) == 3


def test_waiting_and_failing_units_set_the_exit(perf, tmp_path):
    data, env = perf
    (data["root"] / "ds_a" / "fold_0" / "compare.tsv").unlink()
    p = compare_sh(env, "1", "--datasets", str(data["list"]))
    assert p.returncode == 3, p.stdout + p.stderr
    assert records(env)[-1][2] == "3" and records(env)[-1][6].startswith("waiting: 1 fold(s)")
    assert "0 unit(s) done, 1 waiting for inputs, 0 failed" in p.stdout
    (data["root"] / "ds_a" / "fold_0" / "compare.tsv").write_text("dataset\tfold\tmodel\n")
    p = compare_sh(env, "1", "--datasets", str(data["list"]))
    assert p.returncode == 1
    assert records(env)[-1][2] == "1" and records(env)[-1][6].startswith("failed:")


def test_the_default_list_is_the_compendia_inputs(perf, tmp_path):
    data, env = perf
    inputs = Path(env["COMPENDIA_ROOT"]) / "inputs"
    inputs.mkdir(parents=True)
    (inputs / "shared10.txt").write_text(data["list"].read_text())
    p = compare_sh(env, "1")
    assert p.returncode == 0, p.stdout + p.stderr
    assert f"3 dataset(s) from {inputs / 'shared10.txt'}" in p.stdout


@pytest.mark.parametrize(
    ("args", "unset", "message"),
    [
        (["6"], None, "levels must be comma-separated digits 1-5"),
        (["1,,2"], None, "levels must be comma-separated digits 1-5"),
        (["1", "--bogus"], None, "unknown option '--bogus'"),
        (["1"], "CHERI_ROOT", "set CHERI_ROOT"),
        (["1", "--only", "ds_zzz"], None, "--only names ds_zzz, which is not in"),
        (["1", "--workers", "0"], None, "must be positive integers"),
        ([], None, "usage: compare.sh"),
    ],
)
def test_usage_errors_exit_2(perf, args, unset, message):
    data, env = perf
    if unset:
        env.pop(unset)
    extra = ["--datasets", str(data["list"])] if args and args[0] == "1" else []
    p = compare_sh(env, *args, *extra)
    assert p.returncode == 2, p.stdout + p.stderr
    assert message in p.stderr
    assert not (Path(env["BOX_DIR"]) / "compare_steps.tsv").exists()


def test_a_level_script_that_is_not_there_fails_that_unit(perf, tmp_path, monkeypatch):
    data, env = perf
    # a copy of the driver beside no level scripts: every unit is "not built"
    lonely = tmp_path / "repo" / "workflows" / "cherimoya" / "compare"
    lonely.mkdir(parents=True)
    (lonely / "compare.sh").write_text(SCRIPT.read_text())
    p = subprocess.run(["bash", str(lonely / "compare.sh"), "4", "--datasets", str(data["list"])], env=env,
                       capture_output=True, text=True, timeout=60)  # fmt: skip
    assert p.returncode == 1
    assert records(env)[-1][:3] == ["4", "compendia", "1"]
    assert records(env)[-1][6] == "not built: no compare_compendia.py"


def test_sigterm_stops_the_units_and_exits_143(perf, tmp_path):
    data, env = perf
    env["FAKE_PIXI_SLEEP"] = "60"
    proc = subprocess.Popen(["bash", str(SCRIPT), "1", "--datasets", str(data["list"])], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)  # fmt: skip
    deadline = time.time() + 30
    sleeper = None
    while time.time() < deadline and sleeper is None:
        sleeper = next(iter(processes_of(env, "sleep 60")), None)
        time.sleep(0.2)
    assert sleeper, "the unit never started"
    proc.send_signal(signal.SIGTERM)
    stdout, stderr = proc.communicate(timeout=30)
    assert proc.returncode == 143, stdout + stderr
    assert "signal: stopping every unit" in stdout
    time.sleep(0.5)
    assert processes_of(env, "sleep 60") == []
    assert not (Path(env["COMPARISON_ROOT"]) / "performance" / ".done" / "performance.json").exists()
    assert not list(Path(f"/dev/shm/{os.environ.get('USER', 'x')}/box.{env['SLURM_JOB_ID']}").glob("compare.*"))


@pytest.mark.parametrize("script", sorted(p.name for p in COMPARE.glob("compare_*.py")))
def test_each_level_script_starts_on_its_own(script):
    """Run as compare.sh runs it (a fresh interpreter, no test paths): its imports resolve from its own directory."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = os.environ.get("PYTHONPATH", "")  # pytest's shim only, never the test directories
    p = subprocess.run([sys.executable, str(COMPARE / script), "--help"], env=env, capture_output=True, text=True,
                       cwd="/", timeout=120)  # fmt: skip
    assert p.returncode == 0, p.stderr
    assert "usage:" in p.stdout
