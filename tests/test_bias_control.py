"""bias_control.sh (CPU only): ChromBPNet's bias-included model on a fold's QC subsample, with fake tools.

The fakes are test_qc_modisco_folds.py's: a fake igvf checkout (config.sh, motif_qc.py), a fake `chrombpnet
contribs_bw` that writes 05.0's outputs or refuses the model's counts head as chrombpnet's DeepSHAP refuses a
bias-included model's, and a fake `nvidia-smi` listing two GPUs, GPU 1 with less memory in use. Each run gets its own
SLURM_JOB_ID, so its GPU claims and CPU reservations are its own.
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("h5py")
pytest.importorskip("hdf5plugin")
pytest.importorskip("yaml")

TESTS = Path(__file__).resolve().parent
CHERIMOYA = TESTS.parent / "workflows" / "cherimoya"
sys.path.insert(0, str(TESTS))

from test_check_dataset import FOLDS  # noqa: E402
from test_qc_modisco_folds import Box, _out, _wait_for, boxes  # noqa: E402


@pytest.fixture
def box(tmp_path):
    yield from boxes(tmp_path)


def _bias_setup(b: Box) -> tuple[dict, Path, Path]:
    """ChromBPNet's side of a dataset: its config, the bias-included model of every fold and the QC subsamples."""
    data = b.dataset("ds")
    for k in FOLDS:
        models = data["chrombpnet"] / f"ds_all_fold_{k}" / "models"
        models.mkdir()
        (models / "chrombpnet.h5").write_bytes(b"model")
    config = b.tmp / "cbp_configs" / "ds" / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        f'dataset_name: ds\noutput_dir: "{data["chrombpnet"].parent}"\nfolds: ["0", "1", "2", "3", "4"]\n'
    )
    return data, config, b.tmp / "bias_control" / "ds_fold0"


def _bias(b: Box, config: Path, out: Path, fold: str = "0", **env) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(CHERIMOYA / "bias_control.sh"), str(config), fold, str(out)],
                          env={**b.env, **env}, capture_output=True, text=True, timeout=300)  # fmt: skip


def test_bias_control_attributes_on_a_claimed_gpu_then_runs_motif_qc_and_a_rerun_does_nothing(box):
    b = box()
    data, config, out = _bias_setup(b)
    result = _bias(b, config, out)
    assert result.returncode == 0, _out(result)
    [(args, fields)] = b.calls()
    fold_dir = data["chrombpnet"] / "ds_all_fold_0"
    assert args == (f"contribs_bw -m {fold_dir}/models/chrombpnet.h5 -r {fold_dir}/auxiliary/30K_subsample_peaks.bed "
                    f"-g {b.env['FAKE_GENOME']} -c {b.env['FAKE_SIZES']} -op {out}/interpretation -pc counts "
                    "--shap-seed 1234")  # fmt: skip
    assert fields["cuda"] == "[1]" and fields["gpu_claim"] == "1"  # GPU 1: no claims, less memory in use
    assert fields["fraction"] == "0.09" and fields["preallocate"] == "false" and fields["nice"] == "19"
    assert fields["threads"] == "2" and fields["claims"].startswith("1.") and fields["claims"].endswith(".claim,")
    assert not list((b.scratch / "gpu_claims").glob("*.claim"))
    assert "require_gpu jax cuda=[1]" in b.log.read_text()
    qc = [e for e in b.events() if e["event"] == "end"]
    assert len(qc) == 1 and qc[0]["prefix"] == "chrombpnet" and qc[0]["out"] == str(out / "motif_qc_counts")
    assert qc[0]["scores"] == str(out / "interpretation.counts_scores.h5") and qc[0]["cuda"] == "[]"
    status = json.loads((out / "status.json").read_text())
    assert status["state"] == "done" and status["gpu"] == "1" and status["shap_seed"] == 1234
    assert status["interpret_args"]["chrombpnet_version"] == "2.0.0.dev0"
    rows = b.record()
    assert [(r[0], r[1], r[2]) for r in rows] == [("ds", "bias_contribs_0", "0"), ("ds", "bias_modisco_0", "0")]

    before = b.log.read_text()
    again = _bias(b, config, out)
    assert again.returncode == 0, _out(again)
    assert b.log.read_text() == before and "done before" in again.stdout


def test_bias_control_records_a_refused_model_and_skips_modisco(box):
    b = box()
    _, config, out = _bias_setup(b)
    result = _bias(b, config, out, FAKE_CONTRIBS="reject")
    assert result.returncode == 3, _out(result)
    status = json.loads((out / "status.json").read_text())
    assert status["state"] == "rejected" and status["gpu"] == "1"
    assert "counts-head DeepSHAP of a full chrombpnet model" in status["reason"]
    assert not b.events() and not (out / "motif_qc_counts").exists() and not (out / ".done" / "contribs.json").exists()
    assert not list((b.scratch / "gpu_claims").glob("*.claim"))
    [row] = b.record()
    assert row[1:3] == ["bias_contribs_0", "1"] and row[6].startswith("rejected: counts-head DeepSHAP")


def test_bias_control_takes_the_gpu_with_fewer_claims_first(box):
    b = box()
    _, config, out = _bias_setup(b)
    claims = b.scratch / "gpu_claims"
    claims.mkdir(parents=True)
    (claims / "1.999999.1.claim").touch()
    result = _bias(b, config, out)
    assert result.returncode == 0, _out(result)
    [(_, fields)] = b.calls()
    assert fields["cuda"] == "[0]"
    assert [p.name for p in claims.glob("*.claim")] == ["1.999999.1.claim"]


def test_bias_control_fails_on_another_error_and_on_bad_arguments(box):
    b = box()
    _, config, out = _bias_setup(b)
    result = _bias(b, config, out, FAKE_CONTRIBS="fail")
    assert result.returncode == 1
    assert json.loads((out / "status.json").read_text())["state"] == "failed"
    assert _bias(b, config, out, fold="7").returncode == 1
    usage = subprocess.run(["bash", str(CHERIMOYA / "bias_control.sh"), str(config)], env=b.env, capture_output=True,
                           text=True)  # fmt: skip
    assert usage.returncode == 2


def test_bias_control_sigterm_releases_the_claim(box):
    b = box()
    _, config, out = _bias_setup(b)
    proc = subprocess.Popen(["bash", str(CHERIMOYA / "bias_control.sh"), str(config), "0", str(out)],
                            env={**b.env, "FAKE_CONTRIBS_SLEEP": "120"}, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)  # fmt: skip
    try:
        _wait_for(lambda: bool(b.calls()), what="contribs_bw to start")
        proc.send_signal(signal.SIGTERM)
        out_text, _ = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == 143, out_text
    assert not list((b.scratch / "gpu_claims").glob("*.claim"))
    assert not (out / ".done" / "contribs.json").exists()
