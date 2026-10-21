"""to_chrombpnet_scores.py: one fold's attribution in ChromBPNet's 05.0 per-fold layout (CPU only).

It needs h5py and hdf5plugin (the cherimoya environment has both); without them the module is skipped. Each test builds
a synthetic fold that attr_status.py calls DONE: a peaks file, random one-hot ohe, float32 attributions, an idx mask that
drops the first, a middle and the last row, the deltas, a meta with a passed audit, a pin of a fake engine, and the
dataset's Cherimoya config. Every attribution row carries its peaks-file row number (attr[:, 0, 0] = row + 1, exact in
float16), so an h5 row can be traced back to its bed line and its peaks-file line. A few values are planted: 1e-9
(0 in float16) and 1e-6 (a float16 subnormal) at observed bases, and exact zeros.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
h5py = pytest.importorskip("h5py")
pytest.importorskip("hdf5plugin")
pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO / "workflows"
sys.path[:0] = [str(WORKFLOWS / d) for d in ("train", "attribution", "downstream")]

import attr_status as st  # noqa: E402
import to_chrombpnet_scores as tcs  # noqa: E402

W, K = 12, 3
N_ROWS, DROPPED = 45, (0, 17, 44)
N_KEPT = N_ROWS - len(DROPPED)
TINY, SUBNORMAL = 1e-9, 1e-6
N_TINY = 3
KEYS = {"raw", "shap", "projected_shap"}


# ---------------------------------------------------------------------------------------------------------------------
# a synthetic dataset


def _root(tmp_path: Path, folds=("0", "1", "2", "3", "4")) -> Path:
    """<tmp>/root with the pin of a fake engine and the Cherimoya config of dataset `ds`."""
    root = tmp_path / "root"
    engine = tmp_path / "engine"
    engine.mkdir(parents=True)
    for name in st.CODE_FILES:
        (engine / name).write_text(f"# {name}\n")
    root.mkdir()
    assert st.write_pin(engine, root / st.PIN_NAME)[0] == "written"
    config = root / "configs" / "ds" / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        'dataset_name: ds\nassay: ATAC\nsignal_path: "/data/ds.bw"\n'
        f'regions: "{root}/ds/peaks.narrowPeak"\noutput_dir: "{root}/ds/downstream"\npeak_type: all\n'
        f'folds: {json.dumps(list(folds))}\ncontrib_heads: ["counts"]\nmodisco_max_seqlets: 500000\n'
    )
    return root


def _peaks(path: Path, n_rows: int) -> None:
    if not path.exists():
        path.write_text(
            "".join(
                f"chr1\t{1000 * i}\t{1000 * i + 1000}\tpeak_{i + 1}\t0\t.\t{i % 7 + 1}.5\t-1\t-1\t500\n"
                for i in range(n_rows)
            )
        )


def _arrays(idx: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows = np.flatnonzero(idx)
    n = rows.size
    bases = rng.integers(0, 4, size=(n, W))
    ohe = np.zeros((n, 4, W), np.int8)
    ohe[np.arange(n)[:, None], bases, np.arange(W)[None, :]] = 1
    sign = rng.choice(np.array([-1.0, 1.0], np.float32), size=(n, 4, W))
    attr = (rng.uniform(1e-3, 1.0, size=(n, 4, W)).astype(np.float32) * sign).astype(np.float32)
    attr[:, 0, 0] = rows + 1
    if n > N_TINY + 4:
        for j in range(N_TINY):
            attr[1 + j, bases[1 + j, 1 + j], 1 + j] = TINY * (-1) ** j
        attr[4, bases[4, 4], 4] = SUBNORMAL
        attr[5, :, 6] = 0.0
    return ohe, attr


def _hashes(engine: Path) -> dict:
    return {name: hashlib.sha256((engine / name).read_bytes()).hexdigest() for name in st.CODE_FILES}


def _write_attribution(fold: dict, ohe, attr, idx, *, stamp: float, stock: bool = False) -> None:
    """The files attribute_fast.py writes (or the stock command and its record), with ordered mtimes."""
    params, files = fold["params"], st.fold_files(fold["params"])
    np.savez_compressed(params["ohe_filename"], ohe)
    np.savez_compressed(params["attr_filename"], attr)
    np.save(params["idx_filename"], idx)
    outputs = [files["ohe"], files["attr"], files["idx"]]
    if stock:
        files["stock"].write_text(json.dumps({
            "engine": "stock", "exit": 0, "attr_finite": True, "parameters": dict(params, batch_size=20),
            "versions": {"cherimoya": "0.3.0"},
        }))  # fmt: skip
        record = files["stock"]
    else:
        np.save(files["deltas"], np.zeros((len(attr), K), np.float32))
        outputs.append(files["deltas"])
        n_rows, n_kept = int(idx.size), int(idx.sum())
        files["meta"].write_text(json.dumps({
            "tool": "attribute_fast.py",
            "parameters": dict(params, skip=False),
            "random_state": params["random_state"],
            "n_shuffles": params["n_shuffles"],
            "inputs": {"loci": n_rows, "extracted": n_rows, "kept": n_kept, "length": 2114,
                       "window": [1057 - W // 2, 1057 + W // 2]},
            "outputs": {"ohe": params["ohe_filename"], "attr": params["attr_filename"],
                        "idx": params["idx_filename"], "attr_shape": [n_kept, 4, W], "attr_finite": True},
            "deltas": {"finite": True},
            "audit": {"peaks": 8, "pass": True, "status": "pass", "min_pearson": 0.99999},
            "code": {"files_sha256": _hashes(fold["root"].parent / "engine"), "git_commit": "abc"},
        }))  # fmt: skip
        record = files["meta"]
    for path in outputs:
        os.utime(path, (stamp, stamp))
    os.utime(record, (stamp + 1, stamp + 1))


def _fold(root: Path, k: int = 0, *, n_rows: int = N_ROWS, dropped=DROPPED, seed: int = 0, stock: bool = False):
    """<root>/ds/fold_<k>, DONE: model, fit done, attribute.json and its outputs."""
    fold_dir = root / "ds" / f"fold_{k}"
    fold_dir.mkdir(parents=True)
    peaks = root / "ds" / "peaks.narrowPeak"
    _peaks(peaks, n_rows)
    model = fold_dir / f"cherimoya.fold_{k}.torch"
    model.write_bytes(b"weights")
    (fold_dir / f"cherimoya.fold_{k}.test.performance.tsv").write_text("x\n")
    (fold_dir / "fit.json").write_text("{}")
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
        "random_state": 1234 + 100 * k,
        "dtype": "float32",
        "device": "cuda",
        "compile": False,
        "verbose": False,
        "ohe_filename": str(fold_dir / "attr.ohe.npz"),
        "attr_filename": str(fold_dir / "attr.attr.npz"),
        "idx_filename": str(fold_dir / "attr.idx.npy"),
        "fast": {"precision": "reference", "mem_budget_gb": 7.5, "ref_workers": 3, "audit": 8},
    }
    (fold_dir / "attribute.json").write_text(json.dumps(params, indent=4))
    idx = np.ones(n_rows, dtype=bool)
    idx[list(dropped)] = False
    ohe, attr = _arrays(idx, seed)
    now = time.time()
    os.utime(model, (now - 1000, now - 1000))
    fold = {"root": root, "dir": fold_dir, "params": params, "json": fold_dir / "attribute.json", "peaks": peaks,
            "ohe": ohe, "attr": attr, "idx": idx, "stamp": now - 500}  # fmt: skip
    _write_attribution(fold, ohe, attr, idx, stamp=fold["stamp"], stock=stock)
    assert st.check_fold(fold_dir)[0] == "done", st.check_fold(fold_dir)
    return fold


def _out(root: Path, k: int = 0) -> Path:
    return root / "ds" / "downstream" / "full_models" / f"ds_all_fold_{k}" / "interpretation"


def _paths(root: Path, k: int = 0) -> dict:
    out = _out(root, k)
    return {
        "h5": out / "interpretation.counts_scores.h5",
        "bed": out / "interpretation.interpreted_regions.bed",
        "provenance": out / "interpretation.provenance.json",
    }


def _no_partials(root: Path) -> bool:
    return not list(root.rglob("*.partial"))


def _read(path: Path) -> dict:
    with h5py.File(path, "r") as fh:
        return {key: fh[key]["seq"][:] for key in KEYS}


def _f16_stats(x32: np.ndarray) -> dict:
    x16 = x32.astype(np.float16)
    nonzero = x32 != 0
    return {
        "values": int(x32.size),
        "nonzero": int(nonzero.sum()),
        "underflow_to_zero": int((nonzero & (x16 == 0)).sum()),
        "subnormal": int(((x16 != 0) & (np.abs(x16) < 2.0**-14)).sum()),
        "max_abs": float(np.abs(x32).max()),
        "above_float16_max": int((np.abs(x32) > 65504).sum()),
    }


# ---------------------------------------------------------------------------------------------------------------------
# the layout


def test_the_h5_has_05_0s_keys_dtypes_shapes_chunks_and_filter(tmp_path):
    root = _root(tmp_path)
    _fold(root)
    assert tcs.main(["--attribute-json", str(root / "ds" / "fold_0" / "attribute.json"), "--block-rows", "10"]) == 0
    paths = _paths(root)
    with h5py.File(paths["h5"], "r") as fh:
        assert set(fh.keys()) == KEYS  # no hyp_scores: modisco would read that first
        for key, dtype in (("raw", np.int8), ("shap", np.float16), ("projected_shap", np.float16)):
            assert list(fh[key].keys()) == ["seq"]
            ds = fh[key]["seq"]
            assert ds.dtype == dtype and ds.shape == (N_KEPT, 4, W) and ds.chunks == (16, 4, W)
            plist = ds.id.get_create_plist()
            assert plist.get_nfilters() == 1
            code, _flags, values, _name = plist.get_filter(0)
            # Blosc: filter revision, blosc format, type size, chunk bytes, clevel 9, byte shuffle, blosclz; h5ls -v
            # prints ChromBPNet's 05.0 files as `blosc-32001 OPT {2, 2, <type size>, <chunk bytes>, 9, 1, 0}`
            itemsize = np.dtype(dtype).itemsize
            assert code == 32001
            assert tuple(values) == (2, 2, itemsize, 16 * 4 * W * itemsize, 9, 1, 0)
    assert _no_partials(root)


def test_raw_shap_and_projected_are_the_npz_cast_as_05_0_casts_them(tmp_path):
    root = _root(tmp_path)
    fold = _fold(root)
    assert tcs.convert(fold["json"], block_rows=10)[0] == "converted"
    got = _read(_paths(root)["h5"])
    ohe, attr = fold["ohe"], fold["attr"]
    assert np.array_equal(got["raw"], ohe)
    assert np.array_equal(got["shap"].view(np.uint16), attr.astype(np.float16).view(np.uint16))
    expected = (ohe.astype(np.float32) * attr).astype(np.float16)  # scores_io.score_arrays: multiply, then cast
    assert np.array_equal(got["projected_shap"].view(np.uint16), expected.view(np.uint16))
    assert np.array_equal(got["projected_shap"], got["shap"] * ohe)  # non-zero only at the observed base
    assert (got["projected_shap"][ohe == 0] == 0).all()


def test_h5_rows_follow_the_bed_and_the_bed_is_the_kept_peaks_lines(tmp_path):
    root = _root(tmp_path)
    fold = _fold(root)
    assert tcs.convert(fold["json"], block_rows=10)[0] == "converted"
    paths = _paths(root)
    rows = np.flatnonzero(fold["idx"])
    peak_lines = fold["peaks"].read_bytes().splitlines(keepends=True)
    bed = paths["bed"].read_bytes()
    assert bed == b"".join(peak_lines[i] for i in rows)  # byte for byte, in file order
    names = [line.split(b"\t")[3].decode() for line in bed.splitlines()]
    assert names == [f"peak_{i + 1}" for i in rows]
    shap = _read(paths["h5"])["shap"]
    assert np.array_equal(shap[:, 0, 0].astype(np.int64), rows + 1)  # h5 row j is peaks row flatnonzero(idx)[j]
    prov = json.loads(paths["provenance"].read_text())
    assert prov["rows"]["dropped"] == [{"row": i, "name": f"peak_{i + 1}"} for i in DROPPED]


def test_a_fold_that_drops_nothing_gets_the_peaks_file_itself(tmp_path):
    root = _root(tmp_path)
    fold = _fold(root, dropped=())
    assert tcs.convert(fold["json"])[0] == "converted"
    assert _paths(root)["bed"].read_bytes() == fold["peaks"].read_bytes()


def test_few_rows_get_chunks_of_that_many_rows(tmp_path):
    root = _root(tmp_path)
    _fold(root, n_rows=6, dropped=(2,))
    assert tcs.main(["--attribute-json", str(root / "ds" / "fold_0" / "attribute.json")]) == 0
    with h5py.File(_paths(root)["h5"], "r") as fh:
        assert all(fh[key]["seq"].chunks == (5, 4, W) for key in KEYS)  # scores_io: (min(N, 16), 4, L)


def test_block_size_does_not_change_the_output(tmp_path):
    root = _root(tmp_path)
    fold = _fold(root)
    _, prov_small = tcs.convert(fold["json"], block_rows=7)
    small = _read(_paths(root)["h5"])
    _, prov_big = tcs.convert(fold["json"], force=True)
    big = _read(_paths(root)["h5"])
    assert all(np.array_equal(small[k].view(np.uint8), big[k].view(np.uint8)) for k in KEYS)
    assert prov_small["float16"] == prov_big["float16"]


def test_provenance_records_sources_rows_audit_and_float16_statistics(tmp_path):
    root = _root(tmp_path)
    fold = _fold(root)
    assert tcs.convert(fold["json"], block_rows=10)[0] == "converted"
    paths = _paths(root)
    prov = json.loads(paths["provenance"].read_text())
    files = st.fold_files(fold["params"])
    for role, path in (("attribute_json", fold["json"]), ("peaks", fold["peaks"]), ("ohe", files["ohe"]),
                       ("attr", files["attr"]), ("idx", files["idx"]), ("record", files["meta"])):  # fmt: skip
        stat = os.stat(path)
        assert prov["sources"][role] == {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    meta = json.loads(files["meta"].read_text())
    att = prov["attribution"]
    assert att["record"] == "meta" and att["audit"] == meta["audit"] and att["code"] == meta["code"]
    assert att["random_state"] == 1234 and att["attr_window"] == W and att["n_shuffles"] == K
    assert att["window"] == [1057 - W // 2, 1057 + W // 2] and att["summit_column"] == W // 2
    rows = prov["rows"]
    assert (rows["n_rows"], rows["n_kept"], rows["n_dropped"]) == (N_ROWS, N_KEPT, len(DROPPED))
    assert rows["idx_sha256"] == hashlib.sha256(fold["idx"].astype(np.uint8).tobytes()).hexdigest()
    assert prov["outputs"]["bed"]["sha256"] == hashlib.sha256(paths["bed"].read_bytes()).hexdigest()
    for name in ("h5", "bed"):
        stat = os.stat(paths[name])
        assert (prov["outputs"][name]["size"], prov["outputs"][name]["mtime_ns"]) == (stat.st_size, stat.st_mtime_ns)
    assert set(prov["outputs"]["h5"]["datasets"]) == {f"{k}/seq" for k in KEYS}
    assert (
        prov["converter"]["files_sha256"]["to_chrombpnet_scores.py"]
        == hashlib.sha256((WORKFLOWS / "downstream" / "to_chrombpnet_scores.py").read_bytes()).hexdigest()
    )
    assert prov["converter"]["git_commit"] and prov["written"] and prov["written_et"]
    ohe, attr = fold["ohe"], fold["attr"]
    for key, x32 in (("shap", attr), ("projected_shap", ohe.astype(np.float32) * attr)):
        f16 = prov["float16"][key]
        assert {k: f16[k] for k in _f16_stats(x32)} == _f16_stats(x32)
        assert f16["underflow_to_zero"] == N_TINY and f16["subnormal"] == 1 and f16["above_float16_max"] == 0
        assert f16["underflow_fraction"] == pytest.approx(N_TINY / f16["nonzero"])
        assert f16["underflow_flagged"] is (f16["underflow_fraction"] > 1e-3)
    assert prov["float16"]["shap"]["underflow_flagged"] is True  # 3 of ~2,000 non-zero values
    assert prov["float16"]["shap"]["max_abs"] == float(np.flatnonzero(fold["idx"]).max() + 1)


def test_the_default_out_dir_is_the_configs_full_models_tree(tmp_path):
    root = _root(tmp_path)
    fold = _fold(root)
    status, prov = tcs.convert(fold["json"])
    assert status == "converted"
    assert prov["outputs"]["h5"]["path"] == str(_paths(root)["h5"])
    assert prov["config"] == str(root / "configs" / "ds" / "config.yaml")
    other = tmp_path / "elsewhere" / "ds_all_fold_0" / "interpretation"
    assert tcs.convert(fold["json"], out_dir=other)[0] == "converted"
    assert (other / "interpretation.counts_scores.h5").is_file()


def test_an_out_dir_06_0_would_not_read_is_refused(tmp_path):
    root = _root(tmp_path)
    fold = _fold(root)
    for bad in (tmp_path / "x" / "interpretation", tmp_path / "ds_all_fold_1" / "interpretation",
                tmp_path / "ds_all_fold_0" / "scores"):  # fmt: skip
        with pytest.raises(tcs.Refused, match="06.0 reads those"):
            tcs.convert(fold["json"], out_dir=bad)


# ---------------------------------------------------------------------------------------------------------------------
# refusals: nothing is written


def test_a_fold_that_is_not_done_is_refused(tmp_path):
    root = _root(tmp_path)
    fold = _fold(root)
    (root / st.PIN_NAME).unlink()
    with pytest.raises(tcs.Refused, match="is stale, not DONE: .*no engine pin"):
        tcs.convert(fold["json"])
    assert not (root / "ds" / "downstream").exists()
    assert tcs.main(["--attribute-json", str(fold["json"])]) == 1


def test_another_pin_can_be_named(tmp_path):
    root = _root(tmp_path)
    fold = _fold(root)
    pin = tmp_path / "provisional_pin.json"
    os.replace(root / st.PIN_NAME, pin)
    status, prov = tcs.convert(fold["json"], pin=pin)
    assert status == "converted" and prov["pin"] == str(pin)


@pytest.mark.parametrize(
    "value, message",
    [(65504.0, None), (65505.0, "the largest float16"), (np.inf, "non-finite"), (np.nan, "non-finite")],
)
def test_attr_beyond_float16_or_not_finite_fails_the_fold(tmp_path, value, message):
    root = _root(tmp_path)
    fold = _fold(root)
    attr = fold["attr"].copy()
    attr[30, 1, 5] = value
    _write_attribution(fold, fold["ohe"], attr, fold["idx"], stamp=fold["stamp"] + 10)
    if message is None:
        status, prov = tcs.convert(fold["json"], block_rows=10)
        assert status == "converted" and prov["float16"]["shap"]["max_abs"] == 65504.0
        assert prov["float16"]["shap"]["above_float16_max"] == 0
        return
    with pytest.raises(tcs.Refused, match=message):
        tcs.convert(fold["json"], block_rows=10)
    assert not any(p.exists() for p in _paths(root).values()) and _no_partials(root)


@pytest.mark.parametrize("column", [[0, 0, 0, 0], [1, 1, 0, 0], [2, -1, 0, 0]])
def test_an_ohe_column_that_is_not_one_hot_is_refused(tmp_path, column):
    root = _root(tmp_path)
    fold = _fold(root)
    ohe = fold["ohe"].copy()
    ohe[25, :, 3] = column
    _write_attribution(fold, ohe, fold["attr"], fold["idx"], stamp=fold["stamp"] + 10)
    with pytest.raises(tcs.Refused, match="ohe row 25 is not one-hot"):
        tcs.convert(fold["json"], block_rows=10)
    assert not any(p.exists() for p in _paths(root).values()) and _no_partials(root)


def test_a_blank_line_in_the_peaks_file_is_refused(tmp_path):
    root = _root(tmp_path)
    fold = _fold(root)
    lines = fold["peaks"].read_bytes().splitlines(keepends=True)
    fold["peaks"].write_bytes(b"".join(lines[:10] + [b"\n"] + lines[10:]))  # attr_status counts 45 rows still
    assert st.check_fold(fold["dir"])[0] == "done"
    with pytest.raises(tcs.Refused, match="blank line at row 11"):
        tcs.convert(fold["json"])


def test_a_config_that_renames_full_models_is_refused(tmp_path):
    root = _root(tmp_path)
    fold = _fold(root)
    config = root / "configs" / "ds" / "config.yaml"
    config.write_text(config.read_text() + "full_model_precision: mixed\n")
    with pytest.raises(tcs.Refused, match="full_model_precision"):
        tcs.convert(fold["json"])


# ---------------------------------------------------------------------------------------------------------------------
# siblings


def test_a_sibling_fold_with_other_rows_is_refused(tmp_path):
    root = _root(tmp_path)
    assert tcs.convert(_fold(root, 0)["json"])[0] == "converted"
    status, prov = tcs.convert(_fold(root, 1, seed=1)["json"])
    assert status == "converted" and prov["siblings"] == [
        {"fold": "0", "provenance": str(_paths(root, 0)["provenance"]), "compared": True}
    ]
    other = _fold(root, 2, dropped=(*DROPPED, 5), seed=2)
    with pytest.raises(tcs.Refused, match="fold 0 .* kept other rows"):
        tcs.convert(other["json"])
    assert not _out(root, 2).exists() or not any(_out(root, 2).iterdir())


def test_a_sibling_whose_sources_changed_is_not_compared(tmp_path):
    root = _root(tmp_path)
    first = _fold(root, 0)
    assert tcs.convert(first["json"])[0] == "converted"
    st.fold_files(first["params"])["attr"].touch()  # fold 0 is being attributed again
    status, prov = tcs.convert(_fold(root, 1, dropped=(3,), seed=1)["json"])
    assert status == "converted"
    assert prov["siblings"][0]["compared"] is False


# ---------------------------------------------------------------------------------------------------------------------
# atomic and idempotent


def test_a_second_run_is_up_to_date_and_touches_nothing(tmp_path):
    root = _root(tmp_path)
    fold = _fold(root)
    assert tcs.convert(fold["json"])[0] == "converted"
    paths = _paths(root)
    before = {name: os.stat(path) for name, path in paths.items()}
    status, info = tcs.convert(fold["json"])
    assert status == "up to date" and "matches" in info["reason"]
    after = {name: os.stat(path) for name, path in paths.items()}
    assert {n: (s.st_ino, s.st_mtime_ns) for n, s in before.items()} == {
        n: (s.st_ino, s.st_mtime_ns) for n, s in after.items()
    }
    assert tcs.main(["--attribute-json", str(fold["json"])]) == 0
    assert os.stat(paths["h5"]).st_ino == before["h5"].st_ino
    status, _ = tcs.convert(fold["json"], force=True)
    assert status == "converted" and os.stat(paths["h5"]).st_ino != before["h5"].st_ino


def test_a_new_attribution_is_converted_again(tmp_path):
    root = _root(tmp_path)
    fold = _fold(root)
    assert tcs.convert(fold["json"])[0] == "converted"
    ohe, attr = _arrays(fold["idx"], seed=7)
    _write_attribution(fold, ohe, attr, fold["idx"], stamp=fold["stamp"] + 10)
    status, prov = tcs.convert(fold["json"])
    assert status == "converted"
    got = _read(_paths(root)["h5"])
    assert np.array_equal(got["raw"], ohe) and np.array_equal(got["shap"], attr.astype(np.float16))
    assert prov["sources"]["attr"]["mtime_ns"] == os.stat(st.fold_files(fold["params"])["attr"]).st_mtime_ns


def test_an_output_changed_after_the_conversion_is_converted_again(tmp_path):
    root = _root(tmp_path)
    fold = _fold(root)
    assert tcs.convert(fold["json"])[0] == "converted"
    bed = _paths(root)["bed"]
    bed.write_bytes(b"chr1\t0\t1\n")
    assert tcs.convert(fold["json"])[0] == "converted"
    assert bed.read_bytes() == b"".join(
        fold["peaks"].read_bytes().splitlines(keepends=True)[i] for i in np.flatnonzero(fold["idx"])
    )


def test_a_failure_while_writing_leaves_no_partial_and_no_provenance(tmp_path, monkeypatch):
    root = _root(tmp_path)
    fold = _fold(root)
    assert tcs.convert(fold["json"])[0] == "converted"
    ohe, attr = _arrays(fold["idx"], seed=3)
    _write_attribution(fold, ohe, attr, fold["idx"], stamp=fold["stamp"] + 10)
    real, calls = tcs.write_block, []

    def failing(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("disk full")
        return real(*args, **kwargs)

    monkeypatch.setattr(tcs, "write_block", failing)
    with pytest.raises(RuntimeError, match="disk full"):
        tcs.convert(fold["json"], block_rows=10)
    paths = _paths(root)
    assert _no_partials(root)
    assert not paths["provenance"].exists()  # the old outputs are left, described by no provenance
    assert paths["h5"].exists() and paths["bed"].exists()
    monkeypatch.setattr(tcs, "write_block", real)
    assert tcs.convert(fold["json"])[0] == "converted"
    assert np.array_equal(_read(paths["h5"])["shap"], attr.astype(np.float16))


def test_stale_partials_of_a_killed_run_are_replaced(tmp_path):
    root = _root(tmp_path)
    fold = _fold(root)
    out = _out(root)
    out.mkdir(parents=True)
    for path in _paths(root).values():
        Path(f"{path}.partial").write_bytes(b"left by a killed run")
    assert tcs.convert(fold["json"])[0] == "converted"
    assert _no_partials(root)
    assert np.array_equal(_read(_paths(root)["h5"])["raw"], fold["ohe"])


def test_a_source_that_changes_during_the_conversion_is_refused(tmp_path, monkeypatch):
    root = _root(tmp_path)
    fold = _fold(root)
    real = tcs.write_block
    attr_npz = st.fold_files(fold["params"])["attr"]

    def touching(*args, **kwargs):
        later = time.time() + 5
        os.utime(attr_npz, (later, later))
        return real(*args, **kwargs)

    monkeypatch.setattr(tcs, "write_block", touching)
    with pytest.raises(tcs.Refused, match="attr changed during the conversion"):
        tcs.convert(fold["json"])
    assert not any(p.exists() for p in _paths(root).values()) and _no_partials(root)


def test_a_stock_attribution_is_converted_with_its_record(tmp_path):
    root = _root(tmp_path)
    fold = _fold(root, stock=True)
    status, prov = tcs.convert(fold["json"])
    assert status == "converted"
    att = prov["attribution"]
    assert att["record"] == "stock" and att["engine"] == "stock" and att["versions"] == {"cherimoya": "0.3.0"}
    assert prov["sources"]["record"]["path"].endswith("attr.attr.stock.json")


def test_lock_busy_exits_4(tmp_path):
    root = _root(tmp_path)
    fold = _fold(root)
    fd = os.open(fold["dir"] / st.LOCK_NAME, os.O_WRONLY | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert tcs.main(["--attribute-json", str(fold["json"]), "--lock"]) == tcs.LOCK_BUSY_EXIT
        assert not _paths(root)["h5"].exists()
    finally:
        os.close(fd)
    assert tcs.main(["--attribute-json", str(fold["json"]), "--lock"]) == 0


def test_the_command_line(tmp_path):
    root = _root(tmp_path)
    fold = _fold(root)
    cmd = [
        sys.executable,
        str(WORKFLOWS / "downstream" / "to_chrombpnet_scores.py"),
        "--attribute-json",
        str(fold["json"]),
    ]
    first = subprocess.run(cmd, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    assert "wrote" in first.stderr and "dropped peak_1, peak_18, peak_45" in first.stderr
    second = subprocess.run(cmd, capture_output=True, text=True)
    assert second.returncode == 0 and "up to date" in second.stderr
