"""downstream.sh (CPU only): synthetic datasets, the real converter and checks, fake igvf steps and a fake `pixi`.

The datasets come from test_check_dataset.make_dataset (5 DONE folds on a synthetic genome, a fake ChromBPNet 05.0
tree), not converted yet. The fake 06.0 averages the converted folds exactly as average_contrib_scores.py does, so the
real V2 passes on it; the fake 07.0 writes a bigwig; the fake 08.0 writes a results h5 with pos_patterns (or not), and
a report, and can sleep and exit 1. Each fake step appends a line to a log with what it was given (threads, GPUs,
CPU mask), so the tests can see what ran, in which order, and how. Every run gets its own SLURM_JOB_ID, so its
node-local locks are its own.
"""

from __future__ import annotations

import fcntl
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
h5py = pytest.importorskip("h5py")
pytest.importorskip("hdf5plugin")
pytest.importorskip("yaml")

TESTS = Path(__file__).resolve().parent
WORKFLOWS = TESTS.parent / "workflows"
sys.path[:0] = [str(WORKFLOWS / d) for d in ("train", "attribution", "downstream")]
sys.path.insert(0, str(TESTS))

import attr_status as st  # noqa: E402
from test_check_dataset import FOLDS, make_dataset, write_attribution  # noqa: E402

FAKE_PIXI = """#!/bin/bash
# pixi run --frozen --manifest-path M -e ENV cmd... -> cmd..., with python = the test's interpreter
shift
while (( $# )); do
    case "$1" in --frozen) shift ;; --manifest-path|-e) shift 2 ;; *) break ;; esac
done
if [[ "$1" == python ]]; then shift; exec "${FAKE_PYTHON}" "$@"; fi
exec "$@"
"""

STEP_HEAD = """#!/bin/bash
cfg="${DATASET_CONFIG}"
out="$(sed -n 's/^output_dir: "\\(.*\\)"$/\\1/p' "${cfg}")"
ds="$(sed -n 's/^dataset_name: \\(.*\\)$/\\1/p' "${cfg}")"
avg="${out}/contrib_scores/${ds}"
say() { echo "$1 ${ds} $2 $(date +%s.%N) threads=${SLURM_CPUS_PER_TASK} omp=${OMP_NUM_THREADS} numba=${NUMBA_NUM_THREADS}" \\
    "cuda=[${CUDA_VISIBLE_DEVICES-unset}] cpus=$(nproc) task=${SLURM_ARRAY_TASK_ID} repo=${REPO_ROOT}" >> "${FAKE_LOG}"; }
"""

FAKE_STEPS = {
    "06.0.average_contrib_scores": STEP_HEAD
    + """say 06.0 start
[[ "${FAKE_06_EXIT:-0}" == 0 ]] || exit "${FAKE_06_EXIT}"
"${FAKE_PYTHON}" -c 'import sys; sys.path.insert(0, sys.argv[1]); import test_check_dataset as t; t.average_from_config(sys.argv[2])' \\
    "${FAKE_TESTS}" "${cfg}" || exit 1
say 06.0 end
""",
    "07.0.contribs_to_bigwig": STEP_HEAD
    + """say 07.0 start
echo bigwig > "${avg}/${ds}_average_shaps.counts.bw"
say 07.0 end
""",
    "08.0.run_modisco": STEP_HEAD
    + """say 08.0 start
mkdir -p "${avg}/modisco/counts_report"  # as 08.0 does before modisco motifs
sleep "${FAKE_08_SLEEP:-0}"
if [[ "${FAKE_08_NOH5:-0}" != 1 ]]; then
    "${FAKE_PYTHON}" -c 'import sys, h5py
with h5py.File(sys.argv[1], "w") as fh:
    fh.create_group("pos_patterns/pattern_0")
    fh.attrs["window_size"] = 500' "${avg}/modisco/modisco_counts_results.h5" || exit 1
fi
[[ "${FAKE_08_EXIT:-0}" == 0 ]] || { say 08.0 failed; exit "${FAKE_08_EXIT}"; }
echo report > "${avg}/modisco/counts_report/motifs.html"
say 08.0 end
""",
}


class Box:
    """A box directory, a fake igvf checkout and the environment downstream.sh runs with."""

    def __init__(self, tmp_path: Path, shared: tuple[str, ...] = ()):
        self.tmp = tmp_path
        self.dir = tmp_path / "box"
        self.dir.mkdir()
        (self.dir / "attribution_datasets.txt").write_text("".join(f"{ds}\n" for ds in shared))
        self.repo = tmp_path / "igvf"
        steps = self.repo / "workflows" / "SLURM"
        steps.mkdir(parents=True)
        (self.repo / "lib" / "bash").mkdir(parents=True)
        (self.repo / "lib" / "bash" / "common.sh").write_text("")
        for name, text in FAKE_STEPS.items():
            (steps / f"{name}.sh").write_text(text)
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "pixi").write_text(FAKE_PIXI)
        (bin_dir / "pixi").chmod(0o755)
        self.log = tmp_path / "steps.log"
        self.log.touch()
        self.job = f"pytest{os.getpid()}.{tmp_path.name}"
        self.scratch = Path(f"/dev/shm/{os.environ.get('USER', 'user')}/box.{self.job}")
        self.root = tmp_path / "root"
        self.env = dict(
            os.environ,
            PATH=f"{bin_dir}:{os.environ['PATH']}",
            FAKE_LOG=str(self.log),
            FAKE_PYTHON=sys.executable,
            FAKE_TESTS=str(TESTS),
            CHERI_ROOT=str(self.root),
            BOX_DIR=str(self.dir),
            REPO_ROOT=str(self.repo),
            CHERI_PIXI_MANIFEST="/main/checkout/pixi.toml",
            CHERI_08_MAX="6",
            CHERI_08_THREADS="2",
            CHERI_06_THREADS="2",
            CHERI_07_THREADS="2",
            CHERI_DOWNSTREAM_POLL="1",
            CHERI_DOWNSTREAM_RETRY="1",
            CHERI_DOWNSTREAM_PRODUCERS="",
            CHERI_PICK_SAMPLE="0.05",
            SLURM_JOB_ID=self.job,
        )
        for key in ("CHERI_08_MAX_FILE", "CHERI_SHARED_LIST", "CHERI_DROP_FOLD_H5", "METADATA_DIR"):
            self.env.pop(key, None)

    def lists(self, *datasets: str) -> Path:
        path = self.tmp / f"list.{'.'.join(datasets)}.txt"
        path.write_text("# a comment\n\n" + "".join(f"{ds}\n" for ds in datasets))
        return path

    def command(self, *datasets: str) -> list[str]:
        return ["bash", str(WORKFLOWS / "downstream" / "downstream.sh"), str(self.lists(*datasets))]

    def run(self, *datasets: str, timeout: float = 240, **env) -> subprocess.CompletedProcess:
        return subprocess.run(self.command(*datasets), env={**self.env, **env}, capture_output=True, text=True,
                              timeout=timeout)  # fmt: skip

    def start(self, *datasets: str, **env) -> subprocess.Popen:
        return subprocess.Popen(self.command(*datasets), env={**self.env, **env}, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)  # fmt: skip

    def steps(self) -> list[list[str]]:
        return [line.split() for line in self.log.read_text().splitlines()]

    def ran(self, step: str, ds: str = "ds", event: str = "start") -> int:
        return sum(1 for f in self.steps() if f[:3] == [step, ds, event])

    def record(self) -> list[list[str]]:
        lines = (self.dir / "cheri_downstream.tsv").read_text().splitlines()
        assert lines[0].split("\t") == ["dataset", "step", "exit", "seconds", "max_rss_kb", "started", "note"]
        return [line.split("\t") for line in lines[1:]]

    def done(self, ds: str = "ds") -> Path:
        return self.root / ds / "downstream" / ".done"

    def avg(self, ds: str = "ds") -> Path:
        return self.root / ds / "downstream" / "contrib_scores" / ds


@pytest.fixture
def box(tmp_path):
    if shutil.which("flock") is None or shutil.which("taskset") is None or not os.access("/usr/bin/time", os.X_OK):
        pytest.skip("needs flock, taskset and /usr/bin/time")
    made = []

    def make(shared: tuple[str, ...] = ()) -> Box:
        made.append(Box(tmp_path, shared))
        return made[-1]

    yield make
    for b in made:
        shutil.rmtree(b.scratch, ignore_errors=True)


def _wait_for(predicate, timeout: float = 120, what: str = "") -> None:
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return
        time.sleep(0.2)
    raise AssertionError(f"timed out waiting for {what}")


def _out(result: subprocess.CompletedProcess) -> str:
    return result.stdout + result.stderr


ALL_MARKERS = {f"convert_{k}" for k in FOLDS} | {"check", "06.0", "v2", "07.0", "08.0"}


# ---------------------------------------------------------------------------------------------------------------------
# a dataset, start to end


def test_a_dataset_goes_through_every_step_once_and_a_rerun_does_nothing(box, tmp_path):
    b = box()
    make_dataset(tmp_path, "ds", convert=False)
    result = b.run("ds")
    assert result.returncode == 0, _out(result)
    assert {p.name for p in b.done().iterdir()} == ALL_MARKERS
    assert (b.root / "ds" / "downstream" / "validation" / "check_dataset.json").is_file()
    assert (b.root / "ds" / "downstream" / "validation" / "check_average.json").is_file()
    steps = b.steps()
    order = [f[0] for f in steps if f[2] == "start"]
    assert order[0] == "06.0" and sorted(order[1:]) == ["07.0", "08.0"]
    for f in steps:
        fields = dict(item.split("=", 1) for item in f[4:])
        assert fields["threads"] == fields["omp"] == fields["numba"] == fields["cpus"] == "2"
        assert fields["cuda"] == "[]" and fields["task"] == "0" and fields["repo"] == str(b.repo)
    rows = b.record()
    assert [r[1] for r in rows if r[1].startswith("convert")] and all(r[2] == "0" for r in rows)
    assert {r[1] for r in rows} == ALL_MARKERS
    for r in rows:
        assert r[0] == "ds" and r[3].isdigit() and r[4].isdigit(), r
    assert "against fasta" in {r[6] for r in rows if r[1] == "check"}
    assert "1 pos patterns, window_size 500" in (b.done() / "08.0").read_text()
    assert "ds: done" in result.stdout

    before = b.log.read_text(), len(rows)
    again = b.run("ds")
    assert again.returncode == 0, _out(again)
    assert (b.log.read_text(), len(b.record())) == before
    assert "1 of 1 dataset(s) done, 0 given up" in again.stdout


def test_a_shared_dataset_is_checked_against_chrombpnet(box, tmp_path):
    b = box(shared=("ds",))
    make_dataset(tmp_path, "ds", convert=False)
    result = b.run("ds")
    assert result.returncode == 0, _out(result)
    assert "against chrombpnet" in (b.done() / "check").read_text()


def test_08_0_report_failure_with_a_good_h5_counts_as_done(box, tmp_path):
    b = box()
    make_dataset(tmp_path, "ds", convert=False)
    result = b.run("ds", FAKE_08_EXIT="1")
    assert result.returncode == 0, _out(result)
    assert "report_failed: 1 pos patterns" in (b.done() / "08.0").read_text()
    assert [r[2] for r in b.record() if r[1] == "08.0"] == ["1"]


def test_08_0_without_results_gives_the_dataset_up_and_the_others_go_on(box, tmp_path):
    b = box()
    make_dataset(tmp_path, "ds", convert=False)
    make_dataset(tmp_path, "ds2", convert=False)
    result = b.run("ds", "ds2", FAKE_08_NOH5="1", FAKE_08_EXIT="1")
    assert result.returncode == 1
    assert not (b.done() / "08.0").exists() and (b.done() / "07.0").exists()
    assert "given up: ds: 07.0 or 08.0 failed" in result.stdout
    assert [r[6] for r in b.record() if r[1] == "08.0"] == ["failed", "failed"]


def test_a_failed_check_stops_the_dataset_before_06_0(box, tmp_path):
    b = box()
    data = make_dataset(tmp_path, "ds", convert=False)
    fasta = Path(data["folds"]["0"]["params"]["sequences"])
    fasta.write_text(fasta.read_text().replace("N", "A"))
    result = b.run("ds")
    assert result.returncode == 1
    assert b.ran("06.0") == 0 and not (b.done() / "check").exists()
    assert "--check-dataset failed" in result.stdout


# ---------------------------------------------------------------------------------------------------------------------
# markers, resume


def test_an_output_without_its_marker_is_moved_aside_and_its_step_and_the_later_ones_redone(box, tmp_path):
    b = box()
    make_dataset(tmp_path, "ds", convert=False)
    assert b.run("ds").returncode == 0
    (b.done() / "08.0").unlink()
    result = b.run("ds")
    assert result.returncode == 0, _out(result)
    assert (b.ran("06.0"), b.ran("07.0"), b.ran("08.0")) == (1, 1, 2)
    modisco = b.avg() / "modisco"
    assert len(list(modisco.glob("modisco_counts_results.h5.incomplete.*"))) == 1
    assert len(list(modisco.glob("counts_report.incomplete.*"))) == 1

    (b.done() / "06.0").unlink()
    result = b.run("ds")
    assert result.returncode == 0, _out(result)
    assert (b.ran("06.0"), b.ran("07.0"), b.ran("08.0")) == (2, 2, 3)
    assert len(list(b.avg().glob("ds_average_shaps.counts.h5.incomplete.*"))) == 1
    assert len(list(b.avg().glob("ds_average_shaps.counts.bw.incomplete.*"))) == 1
    assert len(list(modisco.glob("modisco_counts_results.h5.incomplete.*"))) == 2
    assert {p.name for p in b.done().iterdir()} == ALL_MARKERS


def test_a_marker_without_its_output_reruns_the_step(box, tmp_path):
    b = box()
    make_dataset(tmp_path, "ds", convert=False)
    assert b.run("ds").returncode == 0
    (b.avg() / "ds_average_shaps.counts.bw").unlink()
    result = b.run("ds")
    assert result.returncode == 0, _out(result)
    assert (b.ran("06.0"), b.ran("07.0"), b.ran("08.0")) == (1, 2, 1)


def test_a_fold_attributed_again_after_06_0_stops_the_dataset(box, tmp_path):
    b = box()
    data = make_dataset(tmp_path, "ds", convert=False)
    assert b.run("ds").returncode == 0
    (b.done() / "08.0").unlink()
    attr = Path(data["folds"]["2"]["params"]["attr_filename"])
    later = time.time() + 5
    os.utime(attr, (later, later))
    result = b.run("ds")
    assert result.returncode == 1
    assert "attributed again after the conversion" in result.stdout and b.ran("08.0") == 1


def test_sigterm_stops_the_steps_and_writes_no_marker_for_them(box, tmp_path):
    b = box()
    make_dataset(tmp_path, "ds", convert=False)
    proc = b.start("ds", FAKE_08_SLEEP="60")
    try:
        _wait_for(lambda: b.ran("08.0") == 1 and (b.done() / "07.0").exists(), what="08.0 to start")
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=30) == 143
    finally:
        proc.kill()
    out = proc.stdout.read()
    assert "signal: stopping every step" in out
    step = str(b.repo / "workflows" / "SLURM" / "08.0")
    _wait_for(lambda: subprocess.run(["pgrep", "-f", step], capture_output=True).returncode == 1, 10, "08.0 to end")
    assert {p.name for p in b.done().iterdir()} == ALL_MARKERS - {"08.0"}
    result = b.run("ds")
    assert result.returncode == 0, _out(result)
    assert (b.ran("06.0"), b.ran("07.0"), b.ran("08.0")) == (1, 1, 2)
    assert not list((b.avg() / "modisco").glob("*.incomplete.*"))  # the stopped 08.0 left an empty report dir


# ---------------------------------------------------------------------------------------------------------------------
# waiting, other runs, 08.0 slots


def test_a_dataset_waits_for_its_attribution_while_a_producer_holds_the_box(box, tmp_path):
    b = box()
    data = make_dataset(tmp_path, "ds", convert=False, done_folds=FOLDS[:4])
    hold = b.dir / ".hold.cherimoya.attr.A"
    hold.touch()
    proc = b.start("ds", CHERI_DOWNSTREAM_PRODUCERS=str(b.dir / ".hold.cherimoya.attr.*"))
    try:
        time.sleep(3)
        assert proc.poll() is None and b.ran("06.0") == 0
        fold = data["folds"]["4"]
        write_attribution(fold, fold["ohe"], fold["attr"], fold["idx"], stamp=fold["stamp"])
        assert st.check_fold(fold["dir"])[0] == "done"
        assert proc.wait(timeout=180) == 0
    finally:
        proc.kill()
    out = proc.stdout.read()
    assert "ds: waiting for its attribution (" in out and "4 done" in out
    assert {p.name for p in b.done().iterdir()} == ALL_MARKERS


def test_without_a_producer_an_unfinished_attribution_is_given_up_and_the_rest_goes_on(box, tmp_path):
    b = box()
    make_dataset(tmp_path, "ds", convert=False, done_folds=FOLDS[:3])
    make_dataset(tmp_path, "ds2", convert=False)
    result = b.run("ds", "ds2")
    assert result.returncode == 1
    assert "ds: given up: its attribution is not done" in result.stdout
    assert b.ran("08.0", "ds2") == 1 and b.ran("06.0", "ds") == 0


def test_a_dataset_another_run_holds_is_left_until_it_is_free(box, tmp_path):
    b = box()
    make_dataset(tmp_path, "ds", convert=False)
    lock_file = b.scratch / "cheri_downstream" / "locks" / "ds.lock"
    lock_file.parent.mkdir(parents=True)
    fd = os.open(lock_file, os.O_WRONLY | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX)
    proc = b.start("ds")
    try:
        time.sleep(3)
        assert proc.poll() is None and b.ran("06.0") == 0
        os.close(fd)
        assert proc.wait(timeout=180) == 0
    finally:
        proc.kill()
    assert "ds: another run is working on it" in proc.stdout.read()


def test_08_0_runs_are_capped_and_go_in_list_order(box, tmp_path):
    b = box()
    names = ("ds_c", "ds_a", "ds_b")
    for ds in names:
        make_dataset(tmp_path, ds, convert=False)
    result = b.run(*names, CHERI_08_MAX="1", FAKE_08_SLEEP="8")
    assert result.returncode == 0, _out(result)
    runs = {}
    for f in b.steps():
        if f[0] == "08.0":
            runs.setdefault(f[1], {})[f[2]] = float(f[3])
    spans = sorted((r["start"], r["end"], ds) for ds, r in runs.items())
    assert [ds for _, _, ds in spans] == list(names)
    assert all(end <= next_start for (_, end, _), (next_start, _, _) in zip(spans, spans[1:]))


def test_the_08_0_cap_is_read_from_its_file(box, tmp_path):
    b = box()
    for ds in ("ds", "ds2"):
        make_dataset(tmp_path, ds, convert=False)
    (b.dir / "cheri_downstream.08_max").write_text("1\n")
    result = b.run("ds", "ds2", CHERI_08_MAX="6", FAKE_08_SLEEP="8")
    assert result.returncode == 0, _out(result)
    starts = {f[1]: float(f[3]) for f in b.steps() if f[0] == "08.0" and f[2] == "start"}
    ends = {f[1]: float(f[3]) for f in b.steps() if f[0] == "08.0" and f[2] == "end"}
    assert ends["ds"] <= starts["ds2"]


def test_the_per_fold_h5_are_dropped_only_when_asked_and_only_for_datasets_chrombpnet_did_not_interpret(box, tmp_path):
    b = box(shared=("ds",))
    for ds in ("ds", "ds2"):
        make_dataset(tmp_path, ds, convert=False)
    result = b.run("ds", "ds2", CHERI_DROP_FOLD_H5="1")
    assert result.returncode == 0, _out(result)

    def per_fold(ds):
        return sorted((b.root / ds / "downstream" / "full_models").glob("*/interpretation/*.counts_scores.h5"))

    assert len(per_fold("ds")) == 5 and per_fold("ds2") == []
    assert (b.done("ds2") / "fold_h5_dropped").exists() and not (b.done("ds") / "fold_h5_dropped").exists()
    assert len(list((b.root / "ds2" / "downstream" / "full_models").glob("*/interpretation/*.provenance.json"))) == 5
    again = b.run("ds", "ds2")
    assert again.returncode == 0, _out(again)
    assert per_fold("ds2") == []  # a done dataset is not converted again


def test_bad_arguments_and_environment_are_refused(box, tmp_path):
    b = box()
    make_dataset(tmp_path, "ds", convert=False)
    assert b.run("ds", CHERI_08_MAX="0").returncode == 1
    no_list = subprocess.run(
        ["bash", str(WORKFLOWS / "downstream" / "downstream.sh")], env=b.env, capture_output=True, text=True
    )
    assert no_list.returncode == 2
    unknown = b.run("nope")
    assert unknown.returncode == 1 and "no " in unknown.stderr and "nope/config.yaml" in unknown.stderr
    env = dict(b.env)
    env.pop("REPO_ROOT")
    missing = subprocess.run(b.command("ds"), env=env, capture_output=True, text=True)
    assert missing.returncode != 0 and "REPO_ROOT" in missing.stderr
