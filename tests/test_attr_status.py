"""attr_status.py's DONE rule and engine pin (CPU only).

Each clause of the DONE rule is broken on its own in a synthetic fold that is otherwise done. The helpers here also
serve test_run_task_attribute.py, whose fake attribute_fast.py writes its files with _write_outputs.
"""

from __future__ import annotations

import ast
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

REPO = Path(__file__).resolve().parents[1]
CHERIMOYA = REPO / "workflows" / "cherimoya"
sys.path.insert(0, str(CHERIMOYA))

import attr_status as st  # noqa: E402

N_ROWS, DROPPED, W, K = 6, (2,), 10, 3
KEPT = N_ROWS - len(DROPPED)


def _engine(path: Path, tag: str = "v1") -> Path:
    path.mkdir(parents=True, exist_ok=True)
    for name in st.CODE_FILES:
        (path / name).write_text(f"# {name} {tag}\n")
    return path


def _hashes(engine: Path) -> dict:
    return {name: hashlib.sha256((engine / name).read_bytes()).hexdigest() for name in st.CODE_FILES}


def _write_outputs(params: dict, engine: Path, *, meta_extra: dict | None = None) -> dict:
    """The files attribute_fast.py writes for `params`, in its order (ohe, attr, idx, deltas, then the meta)."""
    files = st.fold_files(params)
    idx = np.ones(N_ROWS, dtype=bool)
    idx[list(DROPPED)] = False
    np.savez_compressed(params["ohe_filename"], np.zeros((KEPT, 4, W), np.int8))
    np.savez_compressed(params["attr_filename"], np.ones((KEPT, 4, W), np.float32))
    np.save(params["idx_filename"], idx)
    np.save(files["deltas"], np.zeros((KEPT, K), np.float32))
    meta = {
        "parameters": dict(params, skip=False),
        "random_state": params["random_state"],
        "n_shuffles": params["n_shuffles"],
        "inputs": {"loci": N_ROWS, "extracted": N_ROWS, "kept": KEPT, "length": 2114, "window": [0, W]},
        "outputs": {
            "ohe": params["ohe_filename"],
            "attr": params["attr_filename"],
            "idx": params["idx_filename"],
            "attr_shape": [KEPT, 4, W],
            "attr_finite": True,
        },
        "deltas": {"finite": True},
        "audit": {"peaks": 8, "pass": True},
        "code": {"files_sha256": _hashes(engine), "git_commit": "abc"},
    }
    for key, value in (meta_extra or {}).items():
        if isinstance(value, dict) and isinstance(meta.get(key), dict):
            meta[key] = {**meta[key], **value}
        else:
            meta[key] = value
    files["meta"].write_text(json.dumps(meta))
    return meta


def _fold(tmp_path: Path, *, outputs: bool = True, pin: bool = True, fit_done: bool = True) -> tuple[Path, dict]:
    """<tmp>/root/ds/fold_0, done by default: the model, its test performance, the outputs and a matching pin."""
    root = tmp_path / "root"
    fold = root / "ds" / "fold_0"
    fold.mkdir(parents=True)
    peaks = tmp_path / "peaks.narrowPeak"
    peaks.write_text("".join(f"chr1\t{i * 100}\t{i * 100 + 50}\tp{i}\t0\t.\t1\t1\t1\t25\n" for i in range(N_ROWS)))
    (fold / "fit.json").write_text("{}")
    (fold / "compare.json").write_text("{}")
    model = fold / "cherimoya.fold_0.torch"
    model.write_bytes(b"weights")
    if fit_done:
        (fold / "cherimoya.fold_0.test.performance.tsv").write_text("x\n")
    params = {
        "model": str(model),
        "sequences": "/data/genome.fa",
        "loci": [str(peaks)],
        "chroms": ["chr1"],
        "in_window": 2114,
        "attr_window": W,
        "exclusion_lists": None,
        "algorithm": "deep_lift_shap",
        "output": "counts",
        "group": 0,
        "n_shuffles": K,
        "batch_size": 64,
        "warning_threshold": 0.001,
        "random_state": 1234,
        "dtype": "float32",
        "device": "cuda",
        "compile": False,
        "verbose": False,
        "ohe_filename": str(fold / "attr.ohe.npz"),
        "attr_filename": str(fold / "attr.attr.npz"),
        "idx_filename": str(fold / "attr.idx.npy"),
        "fast": {"precision": "reference", "mem_budget_gb": 7.5, "ref_workers": 3, "audit": 8},
    }
    (fold / "attribute.json").write_text(json.dumps(params, indent=4))
    engine = _engine(tmp_path / "engine")
    if pin:
        assert st.write_pin(engine, root / st.PIN_NAME)[0] == "written"
    if outputs:
        time.sleep(0.01)  # the outputs are newer than the model
        _write_outputs(params, engine)
    return fold, params


def _touch_later(path: Path, seconds: float = 5.0) -> None:
    t = time.time() + seconds
    os.utime(path, (t, t))


def _refit(params: dict) -> None:
    """Age the fold's attribution files by 100 s and its model by 50 s: a refit after the attribution, in the past."""
    t = time.time()
    for path in st.fold_files(params).values():
        if path.exists():
            os.utime(path, (t - 100, t - 100))
    os.utime(params["model"], (t - 50, t - 50))


def test_a_complete_fold_is_done(tmp_path):
    fold, params = _fold(tmp_path)
    assert st.check_fold(fold) == ("done", [])
    assert st.fold_state(fold) == ("done", [])
    files = st.fold_files(params)
    assert files["deltas"] == fold / "attr.attr.deltas.npy" and files["meta"] == fold / "attr.attr.meta.json"
    assert files["stock"] == fold / "attr.attr.stock.json"


@pytest.mark.parametrize("name", ["ohe", "attr", "idx", "deltas", "meta"])
def test_a_missing_or_empty_file_is_missing(tmp_path, name):
    fold, params = _fold(tmp_path)
    path = st.fold_files(params)[name]
    path.unlink()
    assert st.check_fold(fold)[0] == "missing"
    if name != "meta":
        path.write_bytes(b"")
        assert st.check_fold(fold)[0] == "missing"


def test_a_refit_makes_the_attribution_stale(tmp_path):
    fold, params = _fold(tmp_path)
    _touch_later(Path(params["model"]))
    state, reasons = st.check_fold(fold)
    assert state == "stale" and "refit since the attribution" in reasons[0]


def test_outputs_newer_than_the_meta_are_stale(tmp_path):
    fold, params = _fold(tmp_path)
    _touch_later(Path(params["attr_filename"]))
    state, reasons = st.check_fold(fold)
    assert state == "stale" and "attr.attr.npz newer than attr.attr.meta.json" in reasons[0]


@pytest.mark.parametrize(
    "extra, message",
    [
        ({"audit": {"pass": False}}, "audit.pass is False"),
        ({"audit": {"pass": None, "status": "pending"}}, "audit.pass is None (pending)"),
        ({"audit": {"pass": None, "status": "off (audit 0)"}}, "(off (audit 0))"),
        ({"outputs": {"attr_finite": False}}, "outputs.attr_finite is False"),
        ({"deltas": {"finite": False}}, "deltas.finite is False"),
        ({"outputs": {"attr_shape": [KEPT, 4, W + 2]}}, "outputs.attr_shape"),
        ({"inputs": {"kept": KEPT - 1}, "outputs": {"attr_shape": [KEPT - 1, 4, W]}}, "the idx mask keeps"),
    ],
)
def test_meta_clauses_that_fail(tmp_path, extra, message):
    fold, params = _fold(tmp_path, outputs=False)
    _write_outputs(params, tmp_path / "engine", meta_extra=extra)
    state, reasons = st.check_fold(fold)
    assert state == "failed" and any(message in r for r in reasons), reasons


@pytest.mark.parametrize(
    "extra, message",
    [
        ({"random_state": 1235}, "meta random_state is 1235"),
        ({"n_shuffles": 20}, "meta n_shuffles is 20"),
        ({"inputs": {"loci": N_ROWS + 1}}, f"inputs.loci is {N_ROWS + 1}"),
        ({"code": {"files_sha256": {"attribute_fast.py": "0"}}}, "code.files_sha256 differs from the pin"),
        ({"outputs": {"attr": "/elsewhere/attr.attr.npz"}}, "meta outputs.attr is"),
    ],
)
def test_meta_clauses_that_make_it_stale(tmp_path, extra, message):
    fold, params = _fold(tmp_path, outputs=False)
    _write_outputs(params, tmp_path / "engine", meta_extra=extra)
    state, reasons = st.check_fold(fold)
    assert state == "stale" and any(message in r for r in reasons), reasons


@pytest.mark.parametrize("key, value", [("attr_window", 400), ("random_state", 1334), ("chroms", ["chr1", "chr2"])])
def test_an_attribute_json_changed_since_the_run_is_stale(tmp_path, key, value):
    fold, params = _fold(tmp_path)
    (fold / "attribute.json").write_text(json.dumps(dict(params, **{key: value})))
    state, reasons = st.check_fold(fold)
    assert state == "stale" and any(f"meta parameters.{key}" in r for r in reasons), reasons


def test_peaks_that_changed_are_stale(tmp_path):
    fold, params = _fold(tmp_path)
    with open(params["loci"][0], "a") as fh:
        fh.write("chr1\t900\t950\tp9\t0\t.\t1\t1\t1\t25\n")
    state, reasons = st.check_fold(fold)
    assert state == "stale"
    assert any(f"the idx mask has {N_ROWS} entries, the peaks file {N_ROWS + 1} rows" in r for r in reasons)
    assert any(f"inputs.loci is {N_ROWS}, the peaks file has {N_ROWS + 1} rows" in r for r in reasons)


def test_files_that_disagree_with_the_meta_fail(tmp_path):
    fold, params = _fold(tmp_path)
    meta = st.fold_files(params)["meta"]
    stamp = meta.stat().st_mtime
    np.savez_compressed(params["attr_filename"], np.ones((KEPT, 4, W), np.float16))  # wrong dtype
    np.save(st.fold_files(params)["deltas"], np.zeros((KEPT, K + 1), np.float32))  # wrong K
    os.utime(meta, (stamp + 5, stamp + 5))
    state, reasons = st.check_fold(fold)
    assert state == "failed"
    assert any("the attr npz holds <f2" in r for r in reasons) and any("the deltas hold" in r for r in reasons)
    Path(params["attr_filename"]).write_bytes(b"not a zip")
    os.utime(meta, (stamp + 10, stamp + 10))
    assert st.check_fold(fold)[0] == "failed"


def test_the_pin_is_required_and_compared_by_file_hashes(tmp_path):
    fold, params = _fold(tmp_path, pin=False)
    state, reasons = st.check_fold(fold)
    assert state == "stale" and any("no engine pin" in r for r in reasons)
    other = _engine(tmp_path / "other_engine", tag="v2")
    pin = tmp_path / "root" / st.PIN_NAME
    assert st.write_pin(other, pin)[0] == "written"
    assert st.check_fold(fold)[0] == "stale"
    assert st.write_pin(tmp_path / "engine", pin)[0] == "refused"  # another pin is there
    assert st.write_pin(tmp_path / "engine", pin, replace=True)[0] == "written"
    assert st.check_fold(fold) == ("done", [])
    assert st.write_pin(tmp_path / "engine", pin)[0] == "unchanged"
    assert json.loads(pin.read_text())["files_sha256"] == _hashes(tmp_path / "engine")
    # an explicit pin elsewhere
    assert st.check_fold(fold, tmp_path / "nowhere.json")[0] == "stale"


def test_check_engine_and_write_pin_refusals(tmp_path):
    engine = _engine(tmp_path / "engine")
    pin = tmp_path / st.PIN_NAME
    assert st.check_engine(engine, pin)[0].startswith("no engine pin")
    assert st.write_pin(engine, pin)[0] == "written"
    assert st.check_engine(engine, pin) == []
    (engine / "cheri_dls.py").write_text("# changed\n")
    assert len(st.check_engine(engine, pin)) == 1 and "cheri_dls.py" in st.check_engine(engine, pin)[0]
    (engine / "dls_refs.py").unlink()
    assert st.write_pin(engine, tmp_path / "new.json")[0] == "refused"


def test_write_pin_refuses_uncommitted_engine_files(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("no git")
    engine = _engine(tmp_path / "repo")
    git = ["git", "-C", str(engine), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", *st.CODE_FILES], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "engine"], check=True)
    (engine / "attribute_fast.py").write_text("# edited\n")
    status, message = st.write_pin(engine, tmp_path / st.PIN_NAME)
    assert status == "refused" and "commit them before signing off" in message
    subprocess.run([*git, "commit", "-q", "-am", "edit"], check=True)
    status, _ = st.write_pin(engine, tmp_path / st.PIN_NAME)
    pin = json.loads((tmp_path / st.PIN_NAME).read_text())
    assert status == "written" and len(pin["git_commit"]) == 40


def test_a_stock_record_replaces_the_meta(tmp_path):
    fold, params = _fold(tmp_path, outputs=False)
    files = st.fold_files(params)
    _write_outputs(params, tmp_path / "engine", meta_extra={"audit": {"pass": False}})
    assert st.check_fold(fold)[0] == "failed"
    files["deltas"].unlink()  # stock writes no deltas
    record = {"engine": "stock", "exit": 0, "attr_finite": True, "parameters": dict(params, batch_size=20)}
    files["stock"].write_text(json.dumps(record))
    _touch_later(files["stock"])
    assert st.check_fold(fold) == ("done", [])
    files["stock"].write_text(json.dumps(dict(record, exit=1)))
    _touch_later(files["stock"])
    assert st.check_fold(fold)[0] == "failed"
    files["stock"].write_text(json.dumps(dict(record, parameters=dict(params, random_state=7))))
    _touch_later(files["stock"])
    assert st.check_fold(fold)[0] == "stale"


def test_listing_states_running_and_waiting(tmp_path):
    fold, _ = _fold(tmp_path, outputs=False)
    assert st.check_fold(fold)[0] == "missing"
    assert st.fold_state(fold)[0] == "missing"
    with open(fold / st.LOCK_NAME, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert st.fold_state(fold) == ("running", [".task.lock is held"])
        assert st.fold_state(fold, locks=False)[0] == "missing"
        assert st.check_fold(fold)[0] == "missing"  # --check never looks at the lock
    assert not st.lock_held(fold)
    (fold / "cherimoya.fold_0.test.performance.tsv").unlink()
    assert st.fold_state(fold)[0] == "waiting"
    Path(json.loads((fold / "attribute.json").read_text())["model"]).unlink()
    assert st.fold_state(fold)[0] == "waiting"


def test_the_command_line(tmp_path, capsys):
    fold, params = _fold(tmp_path)
    root = tmp_path / "root"
    other = root / "ds2" / "fold_1"
    other.mkdir(parents=True)
    (other / "fit.json").write_text("{}")
    assert st.main(["--check", str(fold)]) == 0
    assert capsys.readouterr().out == "ds\tfold_0\tdone\t\n"
    assert st.main(["--files", str(fold)]) == 0
    assert capsys.readouterr().out.split() == [str(p) for p in st.fold_files(params).values()]
    assert st.main(["--root", str(root), "--summary"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "dataset\tfolds\tdone\trunning\twaiting\tstale\tfailed\tmissing"
    assert out[1:] == ["ds\t1\t1\t0\t0\t0\t0\t0", "ds2\t1\t0\t0\t0\t0\t0\t1", "total\t2\t1\t0\t0\t0\t0\t1"]
    assert st.main(["--root", str(root), "--only", "ds", "--settled"]) == 0
    assert capsys.readouterr().out.startswith("ds\tfold_0\tdone")
    assert st.main(["--root", str(root), "--settled"]) == 1
    capsys.readouterr()
    assert st.main(["--check-engine", str(tmp_path / "engine"), "--root", str(root)]) == 0
    assert st.main(["--write-pin", str(tmp_path / "engine"), "--root", str(root)]) == 0
    assert capsys.readouterr().out.startswith("unchanged: ")
    _touch_later(Path(params["model"]))
    assert st.main(["--check", str(fold)]) == 1
    assert capsys.readouterr().out.startswith("ds\tfold_0\tstale\tthe model cherimoya.fold_0.torch is newer")


def test_code_files_are_attribute_fast_pys():
    """The pin covers exactly the files attribute_fast.py records in its meta (read from its source, not imported)."""
    tree = ast.parse((CHERIMOYA / "attribute_fast.py").read_text())
    value = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "CODE_FILES" for t in node.targets)
    )
    assert tuple(ast.literal_eval(value)) == st.CODE_FILES
