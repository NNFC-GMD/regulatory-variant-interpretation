"""qc_modisco_folds.sh (CPU only): synthetic datasets, the real subset_scores.py, fake igvf files and tools.

The datasets come from test_check_dataset.make_dataset (5 DONE folds, a fake ChromBPNet tree), with ChromBPNet's QC
subsample of each fold from test_subset_scores.write_subsample. The fake igvf checkout's lib/bash/config.sh sets the
few names the steps read from DATASET_CONFIG (and activate_env, gpu_env and require_gpu that only log); its
src/motif_qc.py opens the scores like the real one and writes the outputs 04.5 checks, and can sleep or fail. A fake
`chrombpnet contribs_bw` writes 05.0's outputs or refuses the model as chrombpnet's DeepSHAP refuses a bias-included
model's counts head, and a fake `nvidia-smi` lists two GPUs (test_bias_control.py uses both). Each fake appends a line to a log with what it was given, so
the tests can see what ran, with which threads, CPUs, niceness and GPU. Every run gets its own SLURM_JOB_ID, so its
node-local locks, CPU reservations and GPU claims are its own.
"""

from __future__ import annotations

import json
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

from test_check_dataset import FOLDS, make_dataset, write_attribution  # noqa: E402
from test_subset_scores import write_subsample  # noqa: E402

FAKE_PIXI = """#!/bin/bash
# pixi run --frozen --manifest-path M -e ENV cmd... -> cmd..., with python = the test's interpreter
shift
while (( $# )); do
    case "$1" in --frozen) shift ;; --manifest-path|-e) shift 2 ;; *) break ;; esac
done
if [[ "$1" == python ]]; then shift; exec "${FAKE_PYTHON}" "$@"; fi
exec "$@"
"""

FAKE_PYTHON = """#!/bin/bash
exec "${FAKE_PYTHON}" "$@"
"""

FAKE_CONFIG_SH = """# fake lib/bash/config.sh: what the QC steps read, from DATASET_CONFIG
cfg="${DATASET_CONFIG:?}"
dataset_name="$(sed -n 's/^dataset_name: *\\(.*\\)$/\\1/p' "${cfg}")"
output_dir="$(sed -n 's/^output_dir: "\\(.*\\)".*$/\\1/p' "${cfg}")"
peak_type=all
read -r -a folds <<< "$(sed -n 's/^folds: *\\[\\(.*\\)\\].*/\\1/p' "${cfg}" | tr -d '",')"
full_model_dir="${output_dir}/full_models"
src_dir="${REPO_ROOT}/src"
chrombpnet_env="pixi:${CHROMBPNET_REPO:-none}/pyproject.toml#${CHROMBPNET_PIXI_ENV:-none}"
chrombpnet_motifs_meme="${FAKE_MOTIF_DB}"
genome_fa="${FAKE_GENOME}"
chrom_sizes="${FAKE_SIZES}"
activate_env() { echo "activate_env $1 ${dataset_name}" >> "${FAKE_LOG}"; }
gpu_env() { export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"; }
require_gpu() { echo "require_gpu $1 cuda=[${CUDA_VISIBLE_DEVICES}]" >> "${FAKE_LOG}"; }
"""

FAKE_MOTIF_QC = """import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py

ap = argparse.ArgumentParser()
for flag in ("--scores", "--motif-db", "--output-dir", "--prefix"):
    ap.add_argument(flag)
for flag in ("--window", "--max-seqlets", "--threads"):
    ap.add_argument(flag, type=int)
a = ap.parse_args()


def say(event):
    with open(os.environ["FAKE_LOG"], "a") as fh:
        print(f"motif_qc {event} {time.time():.3f} out={a.output_dir} prefix={a.prefix} window={a.window} "
              f"seqlets={a.max_seqlets} threads={a.threads} omp={os.environ.get('OMP_NUM_THREADS')} "
              f"cuda=[{os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}] cpus={len(os.sched_getaffinity(0))} "
              f"nice={os.nice(0)} db={a.motif_db} scores={a.scores}", file=fh)


say("start")
with h5py.File(a.scores, "r") as fh:  # what the real one reads
    n = fh["shap/seq"].shape[0]
    assert fh["raw/seq"].shape == fh["shap/seq"].shape
time.sleep(float(os.environ.get("FAKE_QC_SLEEP", "0")))
if os.environ.get("FAKE_QC_FAIL") and os.environ["FAKE_QC_FAIL"] in a.output_dir:
    say("failed")
    sys.exit(1)
out = Path(a.output_dir)
out.mkdir(parents=True, exist_ok=True)
px = f"{a.prefix}_"
(out / f"{px}modisco_results.h5").write_bytes(b"h5")
(out / f"{px}modisco_motifs.meme").write_text("MEME version 5\\n")
(out / "report").mkdir(exist_ok=True)
(out / "report" / "report.html").write_text("<html></html>\\n")
(out / f"{px}motif_qc.json").write_text(json.dumps({
    "scores": a.scores, "window": a.window, "max_seqlets": a.max_seqlets, "motif_db": a.motif_db,
    "modisco_flags": "-l 2 -z 20 -f 5 -t 20 -g 5 -j 0", "seconds": {"modisco": 1.0, "report": 1.0},
    "pos_patterns": n, "neg_patterns": 1}))
say("end")
"""

FAKE_CHROMBPNET = """#!/bin/bash
# chrombpnet contribs_bw -m M -r R -g G -c C -op P -pc H --shap-seed S
args="$*"
shift
while (( $# )); do
    case "$1" in
        -m) model="$2" ;; -r) regions="$2" ;; -op) prefix="$2" ;; -pc) head="$2" ;; --shap-seed) seed="$2" ;;
    esac
    shift 2
done
echo "chrombpnet ${args} cuda=[${CUDA_VISIBLE_DEVICES-unset}] gpu_claim=${GPU_CLAIM-unset}" \\
     "fraction=${XLA_PYTHON_CLIENT_MEM_FRACTION-unset} preallocate=${XLA_PYTHON_CLIENT_PREALLOCATE-unset}" \\
     "threads=${OMP_NUM_THREADS} cpus=$(nproc) nice=$(nice) claims=$(ls "${FAKE_CLAIMS}" | tr '\\n' ,)" >> "${FAKE_LOG}"
echo "{\\"chrombpnet_version\\": \\"2.0.0.dev0\\", \\"jax_backend\\": \\"gpu\\", \\"seed\\": ${seed}}" > "${prefix}.interpret.args.json"
sleep "${FAKE_CONTRIBS_SLEEP:-0}"
case "${FAKE_CONTRIBS:-ok}" in
    reject)
        echo "Traceback (most recent call last):" >&2
        echo '  File "explainer.py", line 87, in check_model' >&2
        echo "NotImplementedError: counts-head DeepSHAP of a full chrombpnet model (logsumexp of the bias and" \\
             "no-bias counts) is not supported: interpret the bias-corrected *_nobias.h5 model instead (as the" \\
             "chrombpnet pipeline does). The profile head of the full model is supported." >&2
        exit 1 ;;
    fail) echo "RuntimeError: something else" >&2; exit 1 ;;
esac
cp "${regions}" "${prefix}.interpreted_regions.bed"
"${FAKE_PYTHON}" - "${prefix}.${head}_scores.h5" "$(wc -l < "${regions}")" <<'EOF'
import sys

import h5py
import numpy as np

n = int(sys.argv[2])
with h5py.File(sys.argv[1], "w") as fh:
    fh.create_dataset("raw/seq", data=np.zeros((n, 4, 2114), np.int8))
    fh.create_dataset("shap/seq", data=np.zeros((n, 4, 2114), np.float16))
    fh.create_dataset("projected_shap/seq", data=np.zeros((n, 4, 2114), np.float16))
EOF
echo bigwig > "${prefix}.${head}_scores.bw"
"""

FAKE_NVIDIA_SMI = """#!/bin/bash
case "$*" in
    *-L*) echo "GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-0)"; echo "GPU 1: NVIDIA H100 80GB HBM3 (UUID: GPU-1)" ;;
    *--query-gpu*) printf '0, %s\\n1, %s\\n' "${FAKE_MEM0:-500}" "${FAKE_MEM1:-100}" ;;
esac
"""

CHROMBPNET_SETTINGS = {"window": 400, "max_seqlets": 5000, "modisco_flags": "-l 2 -z 20 -f 5 -t 20 -g 5 -j 0"}


class Box:
    """A box directory, a fake igvf checkout, fake tools and the environment the scripts run with."""

    def __init__(self, tmp_path: Path, shared: tuple[str, ...] = ()):
        self.tmp = tmp_path
        self.dir = tmp_path / "box"
        self.dir.mkdir()
        (self.dir / "attribution_datasets.txt").write_text("".join(f"{ds}\n" for ds in shared))
        self.repo = tmp_path / "igvf"
        (self.repo / "lib" / "bash").mkdir(parents=True)
        (self.repo / "lib" / "bash" / "common.sh").write_text("")
        (self.repo / "lib" / "bash" / "config.sh").write_text(FAKE_CONFIG_SH)
        (self.repo / "src").mkdir()
        (self.repo / "src" / "motif_qc.py").write_text(FAKE_MOTIF_QC)
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        for name, text in (("pixi", FAKE_PIXI), ("python", FAKE_PYTHON), ("chrombpnet", FAKE_CHROMBPNET),
                           ("nvidia-smi", FAKE_NVIDIA_SMI)):  # fmt: skip
            (bin_dir / name).write_text(text)
            (bin_dir / name).chmod(0o755)
        self.log = tmp_path / "steps.log"
        self.log.touch()
        refs = tmp_path / "refs"
        refs.mkdir()
        for name in ("motifs.meme", "genome.fa", "chrom.sizes"):
            (refs / name).write_text("x\n")
        self.job = f"pytest{os.getpid()}.{tmp_path.name}"
        self.scratch = Path(f"/dev/shm/{os.environ.get('USER', 'user')}/box.{self.job}")
        self.root = tmp_path / "root"
        self.env = dict(
            os.environ,
            PATH=f"{bin_dir}:{os.environ['PATH']}",
            FAKE_LOG=str(self.log),
            FAKE_PYTHON=sys.executable,
            FAKE_MOTIF_DB=str(refs / "motifs.meme"),
            FAKE_GENOME=str(refs / "genome.fa"),
            FAKE_SIZES=str(refs / "chrom.sizes"),
            FAKE_CLAIMS=str(self.scratch / "gpu_claims"),
            CHROMBPNET_REPO="/fake/chrombpnet",
            CHROMBPNET_PIXI_ENV="cuda13-dev",
            CHERI_ROOT=str(self.root),
            BOX_DIR=str(self.dir),
            REPO_ROOT=str(self.repo),
            CHERI_PIXI_MANIFEST="/main/checkout/pixi.toml",
            CHERI_QC_MAX="3",
            CHERI_QC_THREADS="2",
            CHERI_BIAS_THREADS="2",
            CHERI_QC_POLL="1",
            CHERI_QC_RETRY="1",
            CHERI_QC_PRODUCERS="",
            CHERI_PICK_SAMPLE="0.05",
            SLURM_JOB_ID=self.job,
        )
        for key in ("CHERI_SHARED_LIST", "CHERI_QC_CORES_DIR", "METADATA_DIR", "MOTIF_QC_WINDOW",
                    "MOTIF_QC_MAX_SEQLETS", "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_PREALLOCATE",
                    "CHERI_BIAS_HEAD", "CHERI_BIAS_SHAP_SEED", "CHERI_QC_NICE"):  # fmt: skip
            self.env.pop(key, None)

    def dataset(self, ds: str = "ds", *, qc_json: dict[str, dict] | None = None, **kwargs) -> dict:
        """A dataset with ChromBPNet's QC subsample in every fold, and ChromBPNet's QC summaries of some folds."""
        data = make_dataset(self.tmp, ds, convert=False, **kwargs)
        for k in FOLDS:
            write_subsample(data, k)
        for k, settings in (qc_json or {}).items():
            qc = data["chrombpnet"] / f"{ds}_all_fold_{k}" / "evaluation" / "motif_qc_counts"
            qc.mkdir(parents=True)
            summary = {**CHROMBPNET_SETTINGS, "motif_db": self.env["FAKE_MOTIF_DB"], "pos_patterns": 9, **settings}
            (qc / "chrombpnet_nobias_motif_qc.json").write_text(json.dumps(summary))
        return data

    def lists(self, *datasets: str) -> Path:
        path = self.tmp / f"list.{'.'.join(datasets)}.txt"
        path.write_text("# a comment\n\n" + "".join(f"{ds}\n" for ds in datasets))
        return path

    def qc_command(self, *datasets: str) -> list[str]:
        return [
            "bash",
            str(WORKFLOWS / "downstream" / "qc_modisco_folds.sh"),
            *([str(self.lists(*datasets))] if datasets else []),
        ]

    def run(self, *datasets: str, timeout: float = 300, **env) -> subprocess.CompletedProcess:
        return subprocess.run(self.qc_command(*datasets), env={**self.env, **env}, capture_output=True, text=True,
                              timeout=timeout)  # fmt: skip

    def start(self, *datasets: str, **env) -> subprocess.Popen:
        return subprocess.Popen(self.qc_command(*datasets), env={**self.env, **env}, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)  # fmt: skip

    def events(self) -> list[dict]:
        """The fake motif_qc.py's lines: event, time and the key=value fields."""
        out = []
        for line in self.log.read_text().splitlines():
            fields = line.split()
            if fields and fields[0] == "motif_qc":
                out.append({"event": fields[1], "time": float(fields[2]),
                            **dict(f.split("=", 1) for f in fields[3:] if "=" in f)})  # fmt: skip
        return out

    def calls(self) -> list[tuple[str, dict]]:
        """The fake chrombpnet's lines: its arguments, and the key=value fields after them."""
        out = []
        for line in self.log.read_text().splitlines():
            if line.startswith("chrombpnet "):
                args, _, rest = line[len("chrombpnet ") :].partition(" cuda=")
                out.append((args, dict(f.split("=", 1) for f in ("cuda=" + rest).split() if "=" in f)))
        return out

    def record(self) -> list[list[str]]:
        lines = (self.dir / "cheri_qc.tsv").read_text().splitlines()
        assert lines[0].split("\t") == ["dataset", "step", "exit", "seconds", "max_rss_kb", "started", "note"]
        return [line.split("\t") for line in lines[1:]]

    def qc_dir(self, ds: str, k: str) -> Path:
        return self.root / ds / "downstream" / "full_models" / f"{ds}_all_fold_{k}" / "evaluation" / "motif_qc_counts"

    def marker(self, ds: str, k: str) -> Path:
        return self.qc_dir(ds, k) / "cherimoya_qc.done.json"


def boxes(tmp_path: Path):
    """The `box` fixture's body (test_bias_control.py's too): yields a maker of Boxes; removes their scratch after."""
    for tool in ("flock", "taskset", "nice", "pgrep"):
        if shutil.which(tool) is None:
            pytest.skip(f"needs {tool}")
    if not os.access("/usr/bin/time", os.X_OK):
        pytest.skip("needs /usr/bin/time")
    made = []

    def make(shared: tuple[str, ...] = ()) -> Box:
        made.append(Box(tmp_path, shared))
        return made[-1]

    yield make
    for b in made:
        shutil.rmtree(b.scratch, ignore_errors=True)


@pytest.fixture
def box(tmp_path):
    yield from boxes(tmp_path)


def _wait_for(predicate, timeout: float = 120, what: str = "") -> None:
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return
        time.sleep(0.2)
    raise AssertionError(f"timed out waiting for {what}")


def _out(result: subprocess.CompletedProcess) -> str:
    return result.stdout + result.stderr


def _max_overlap(events: list[dict]) -> int:
    points = sorted((e["time"], 1 if e["event"] == "start" else -1) for e in events if e["event"] in ("start", "end"))
    now = peak = 0
    for _, step in points:
        now += step
        peak = max(peak, now)
    return peak


# ---------------------------------------------------------------------------------------------------------------------
# qc_modisco_folds.sh


def test_every_fold_runs_once_as_04_5_runs_it_and_a_rerun_does_nothing(box):
    b = box()
    b.dataset("ds", qc_json={"0": {}, "1": {"window": 300}})
    b.dataset("ds2")
    result = b.run("ds", "ds2", FAKE_QC_SLEEP="1")
    assert result.returncode == 0, _out(result)
    for ds in ("ds", "ds2"):
        for k in FOLDS:
            assert b.marker(ds, k).is_file(), (ds, k)
    ends = [e for e in b.events() if e["event"] == "end"]
    assert len(ends) == 10 and len({e["out"] for e in ends}) == 10
    for e in b.events():
        assert (
            e["prefix"] == "cherimoya"
            and e["window"] == "400"
            and e["seqlets"] == "5000"
            and e["db"] == b.env["FAKE_MOTIF_DB"]
        )
        assert e["threads"] == e["omp"] == e["cpus"] == "2" and e["cuda"] == "[]" and e["nice"] == "19"
        assert e["scores"].endswith("/auxiliary/interpret_subsample/cherimoya.counts_scores.h5")
    assert 2 <= _max_overlap(b.events()) <= 3
    assert "activate_env pixi:/fake/chrombpnet/pyproject.toml#cuda13-dev ds" in b.log.read_text()
    rows = b.record()
    assert sorted(r[1] for r in rows if r[0] == "ds") == sorted([f"subset_{k}" for k in FOLDS] +
                                                                [f"modisco_{k}" for k in FOLDS])  # fmt: skip
    assert all(r[2] == "0" and r[3].isdigit() and r[4].isdigit() for r in rows), rows
    notes = {r[1]: r[6] for r in rows if r[0] == "ds"}
    assert notes["subset_0"] == "written"
    assert notes["modisco_0"] == "13 pos, 1 neg patterns from 13 peaks; settings as ChromBPNet's QC"
    assert notes["modisco_1"].endswith("settings differ from ChromBPNet's QC: window")
    assert notes["modisco_2"].endswith("no ChromBPNet QC summary to compare settings with")
    marker = json.loads(b.marker("ds", "0").read_text())
    assert marker["chrombpnet"]["settings_differ"] == [] and marker["summary"]["window"] == 400

    before = b.log.read_text(), len(rows)
    again = b.run("ds", "ds2")
    assert again.returncode == 0, _out(again)
    assert (b.log.read_text(), len(b.record())) == before
    assert "10 of 10 fold(s) done, 0 given up" in again.stdout


def test_with_no_list_the_shared_list_runs(box):
    b = box(shared=("ds",))
    b.dataset("ds")
    result = b.run()
    assert result.returncode == 0, _out(result)
    assert all(b.marker("ds", k).is_file() for k in FOLDS)


def test_a_fold_waits_for_its_attribution_while_a_producer_holds_the_box(box):
    b = box()
    data = b.dataset("ds", done_folds=FOLDS[:4])
    hold = b.dir / ".hold.cherimoya.attr.A"
    hold.touch()
    proc = b.start("ds", CHERI_QC_PRODUCERS=str(b.dir / ".hold.cherimoya.attr.*"))
    try:
        _wait_for(lambda: all(b.marker("ds", k).is_file() for k in FOLDS[:4]), what="the 4 done folds")
        time.sleep(2)
        assert proc.poll() is None and not b.marker("ds", "4").exists()
        fold = data["folds"]["4"]
        write_attribution(fold, data["ohe"], fold["attr"], data["idx"], stamp=time.time() - 5)
        _wait_for(lambda: b.marker("ds", "4").is_file(), what="fold 4 once attributed")
        hold.unlink()
        out, _ = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == 0, out
    assert "ds fold 4: waiting for its attribution (missing)" in out


def test_a_fold_with_no_producer_left_is_given_up_and_the_others_go_on(box):
    b = box()
    b.dataset("ds", done_folds=FOLDS[:4])
    result = b.run("ds")
    assert result.returncode == 1
    assert all(b.marker("ds", k).is_file() for k in FOLDS[:4]) and not b.marker("ds", "4").exists()
    assert "given up: ds fold 4: attribution not done (missing)" in result.stdout


def test_a_failed_motif_qc_gives_the_fold_up_and_the_others_go_on(box):
    b = box()
    b.dataset("ds")
    result = b.run("ds", FAKE_QC_FAIL="ds_all_fold_2/")
    assert result.returncode == 1
    assert [k for k in FOLDS if b.marker("ds", k).exists()] == ["0", "1", "3", "4"]
    assert "given up: ds fold 2: subset or MoDISco failed (exit 1)" in result.stdout
    assert [r[2] for r in b.record() if r[1] == "modisco_2"] == ["1"]


def test_a_missing_subsample_gives_the_fold_up(box):
    b = box()
    data = b.dataset("ds")
    (data["chrombpnet"] / "ds_all_fold_3" / "auxiliary" / "30K_subsample_peaks.bed").unlink()
    result = b.run("ds")
    assert result.returncode == 1
    assert not b.marker("ds", "3").exists() and b.marker("ds", "4").exists()
    assert [r[6] for r in b.record() if r[1] == "subset_3"] == ["failed"]


def test_a_fold_attributed_again_is_redone_and_its_old_qc_kept_aside(box):
    b = box()
    data = b.dataset("ds")
    assert b.run("ds").returncode == 0
    fold = data["folds"]["0"]
    write_attribution(fold, data["ohe"], (fold["attr"] * 3).astype(np.float32), data["idx"], stamp=time.time() - 5)
    n_before = len(b.events())
    result = b.run("ds")
    assert result.returncode == 0, _out(result)
    new = b.events()[n_before:]
    assert {e["out"] for e in new} == {str(b.qc_dir("ds", "0"))}
    aside = list(b.qc_dir("ds", "0").parent.glob("motif_qc_counts.incomplete.*"))
    assert len(aside) == 1 and (aside[0] / "cherimoya_qc.done.json").is_file()
    assert b.marker("ds", "0").is_file()
    assert [r[6] for r in b.record() if r[1] == "subset_0"] == ["written", "written"]


def test_sigterm_stops_the_steps_and_writes_no_marker(box):
    b = box()
    b.dataset("ds")
    proc = b.start("ds", FAKE_QC_SLEEP="120")
    try:
        _wait_for(lambda: len(b.events()) >= 3, what="three MoDISco runs")
        t0 = time.time()
        proc.send_signal(signal.SIGTERM)
        out, _ = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == 143, out
    assert time.time() - t0 < 30
    assert not any(b.marker("ds", k).exists() for k in FOLDS)
    left = subprocess.run(["pgrep", "-f", str(b.repo / "src" / "motif_qc.py")], capture_output=True, text=True)
    assert left.stdout.strip() == ""


def test_bad_settings_are_refused(box):
    b = box()
    b.dataset("ds")
    result = b.run("ds", CHERI_QC_MAX="0")
    assert result.returncode == 1 and "positive integers" in result.stderr
    result = b.run("nope")
    assert result.returncode == 1 and "no " in result.stderr
