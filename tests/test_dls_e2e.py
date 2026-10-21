"""The end-to-end driver workflows/attribution/dls_e2e.py on synthetic files (CPU only; no torch, no GPU).

E9's report rests on three pieces of dls_e2e.py, checked here:
- prepare writes a stock and a fast JSON that differ only in their three output paths;
- stock_warnings counts the stock run's per-batch convergence warnings and reads the deltas they print, in torch's
  scientific and fixed notations, through progress-bar noise;
- compare passes outputs that differ from the stock's at the floor and fails a wrong ohe, idx or attribution.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1] / "workflows" / d) for d in ("train", "attribution", "downstream")
]

import dls_e2e  # noqa: E402

FLOORS = {
    "floors": {
        "F_rr": {"hyp": {"n": 256, "q50": 7.0e-5, "q99": 1.2e-4, "max": 1.3e-4}},
        "F_tf": {"hyp": {"n": 256, "q50": 5.0e-4, "q99": 1.0e-3, "max": 1.5e-3}},
    }
}

STOCK_LOG = (
    "Loading Loci: 100%|##########| 5/5 [00:00<00:00]\n"
    "  0%|          | 0/100 [00:00<?, ?it/s]\r 20%|##        | 20/100 [00:01<00:04, 19.6it/s]"
    "/x/tangermeme/deep_lift_shap.py:745: RuntimeWarning: Convergence deltas too high: tensor([1.2000e-04, "
    "1.9636e-03,\n        3.0000e-05, 1.0010e-03], device='cuda:0', grad_fn=<AbsBackward0>)\n"
    '  warnings.warn("Convergence deltas too high: " +\n'
    "\r 40%|####      | 40/100 [00:02<00:03, 19.6it/s]"
    "/x/tangermeme/deep_lift_shap.py:745: RuntimeWarning: Convergence deltas too high: tensor([0.0011, 0.0004, "
    "0.0009], device='cuda:0', grad_fn=<AbsBackward0>)\n"
    '  warnings.warn("Convergence deltas too high: " +\n'
)


def test_stock_warnings_reads_both_notations():
    w = dls_e2e.stock_warnings(STOCK_LOG, 1e-3)
    assert w["warnings"] == 2 and w["tensors_parsed"] == 2
    assert w["tensors_in_fixed_notation"] == 1
    assert w["printed_values"] == 7
    assert w["printed_above_threshold"] == 3  # 1.9636e-03, 1.0010e-03 and 0.0011
    assert w["printed_max"] == pytest.approx(1.9636e-3)
    assert dls_e2e.stock_warnings("no warnings here", 1e-3)["printed_max"] is None


def _prepare(tmp_path: Path) -> Path:
    base = {
        "model": "m.torch",
        "sequences": "g.fa",
        "loci": ["p.bed"],
        "batch_size": 64,
        "n_shuffles": 2,
        "warning_threshold": 1e-3,
        "ohe_filename": "/elsewhere/attr.ohe.npz",
        "attr_filename": "/elsewhere/attr.attr.npz",
        "idx_filename": "/elsewhere/attr.idx.npy",
    }
    path = tmp_path / "attribute.json"
    path.write_text(json.dumps(base))
    out = tmp_path / "e9"
    argv = ["prepare", "-p", str(path), "--out-dir", str(out), "--set", "batch_size=20", "--fast", '{"ref_workers": 1}']
    assert dls_e2e.main(argv) == 0
    return out


def test_prepare_writes_jsons_that_differ_only_in_their_outputs(tmp_path):
    out = _prepare(tmp_path)
    stock, fast = (json.loads((out / f"{side}.json").read_text()) for side in ("stock", "fast"))
    assert {k for k in stock if stock[k] != fast[k]} == {"ohe_filename", "attr_filename", "idx_filename"}
    assert stock["batch_size"] == fast["batch_size"] == 20
    assert stock["fast"] == fast["fast"] == {"ref_workers": 1}
    assert stock["attr_filename"] == str(out / "stock" / "attr.attr.npz")
    assert fast["idx_filename"] == str(out / "fast" / "attr.idx.npy")
    assert (out / "stock").is_dir() and (out / "fast").is_dir()


def _write_side(params: dict, ohe, attr, idx) -> None:
    np.savez_compressed(params["ohe_filename"], ohe)
    np.savez_compressed(params["attr_filename"], attr)
    np.save(params["idx_filename"], idx)


def _synthetic(tmp_path: Path, *, noise: float, ohe_flip: bool = False, idx_flip: bool = False) -> Path:
    out = _prepare(tmp_path)
    params = {side: json.loads((out / f"{side}.json").read_text()) for side in ("stock", "fast")}
    rng = np.random.default_rng(0)
    ohe = np.eye(4, dtype=np.int8)[rng.integers(0, 4, (30, 50))].transpose(0, 2, 1).copy()
    idx = np.ones(32, dtype=bool)
    idx[[3, 17]] = False
    attr = rng.normal(size=(30, 4, 50)).astype(np.float32)
    _write_side(params["stock"], ohe, attr, idx)
    fast_attr = (attr * (1 + noise * rng.normal(size=attr.shape))).astype(np.float32)
    fast_ohe, fast_idx = ohe.copy(), idx.copy()
    if ohe_flip:
        fast_ohe[0, :, 0] = np.roll(fast_ohe[0, :, 0], 1)
    if idx_flip:
        fast_idx[0] = False
    _write_side(params["fast"], fast_ohe, fast_attr, fast_idx)
    (out / "stock.log").write_text(STOCK_LOG)
    (tmp_path / "report.json").write_text(json.dumps(FLOORS))
    return out


def _compare(tmp_path: Path, out: Path) -> tuple[int, dict]:
    notes = tmp_path / "notes.md"
    notes.write_text("## Notes\n\n- a note\n")
    argv = ["compare", "--out-dir", str(out), "--floors", str(tmp_path / "report.json"), "--notes", str(notes)]
    status = dls_e2e.main(argv)
    return status, json.loads((out / "e9_report.json").read_text())


def test_compare_passes_outputs_at_the_floor(tmp_path):
    out = _synthetic(tmp_path, noise=5e-5)
    status, report = _compare(tmp_path, out)
    assert status == 0 and report["pass"]
    assert report["stock_warnings"]["warnings"] == 2
    assert report["E3_bounds"]["q50"] == pytest.approx(1.4e-4) and report["E3_bounds"]["policy"] == pytest.approx(
        1.5e-4
    )
    text = (out / "e9_report.md").read_text()
    assert "Verdict: **PASS**" in text and "## Notes" in text and "30 peaks" in text


@pytest.mark.parametrize(
    "change, failing",
    [
        (
            {"noise": 3e-2},  # rel-L2 about 3e-2 and 1 - r about 4.5e-4: every E3 limit and both E5 caps fail
            {
                "E3 hyp median",
                "E3 hyp p99",
                "E3 policy (hyp median)",
                "E5 Pearson (min over peaks)",
                "E5 rel-L2 (max over peaks)",
            },
        ),
        ({"noise": 5e-5, "ohe_flip": True}, {"ohe"}),
        ({"noise": 5e-5, "idx_flip": True}, {"idx"}),
    ],
    ids=["attr-3e-2", "ohe", "idx"],
)
def test_compare_fails_wrong_outputs(tmp_path, change, failing):
    out = _synthetic(tmp_path, **change)
    status, report = _compare(tmp_path, out)
    assert status == 1 and not report["pass"]
    assert {r["criterion"] for r in report["criteria"] if not r["pass"]} == failing
