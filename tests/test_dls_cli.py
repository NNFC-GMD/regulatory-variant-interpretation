"""attribute_fast.py against the stock `cherimoya attribute`, on the CPU (acceptance criterion C11; design steps 3 and 7).

attribute_fast.py reads the stock command's attribute.json and writes the stock command's three files, with the
attributions computed by cheri_dls.Engine. These tests run both commands on the same synthetic inputs:
- A FASTA of 3 contigs of 3 kb, chr1-chr3, with a run of 20 Ns in chr2.
- 12 loci. One sits so close to the end of chr3 that its window runs off the contig, so extract_loci drops it. One
  covers the N run, so the N filter drops it. 10 are kept.
- A tiny Cherimoya, tests/dls_tiny.py's config A (16 filters, 3 layers, weights redrawn so the rules matter), saved
  with model.save() and loaded by both commands from that file.
- The JSON: in_window 256, attr_window 64, n_shuffles 4, random_state 0, device cpu, float32, batch_size 64 or 7.
  Batches of 7 split a peak's 4 pairs across batches in the stock run.

C11: the ohe (int8) and idx (bool) files are numpy.array_equal to the stock's, every npz holds the one key arr_0, the
shapes are equal, and the attributions are float32 with a per-locus rel-L2 of at most 1e-5 against the stock's.

The audit (--audit N, design §14): every run above re-attributes 8 evenly spaced peaks with the stock deep_lift_shap,
the engine's own references injected, at batch_size min(JSON, 20), and passes. A run whose engine output is corrupted
on one audited peak (scaled by 1.05, or swapped with another peak's) fails it after writing every output, the meta
records the failure, and main() exits 3; so do references that differ from tangermeme's draw. --audit 0 runs none.
The per-peak comparison and its 3e-4 median warning are checked on synthetic arrays.

Also covered:
- The order of the writes: ohe, attr, then idx (the done marker), then the deltas sidecar; the meta sidecar and the
  self-check's record; a stale done marker is removed before the new files are written.
- skip returns before anything is loaded; saturation_mutagenesis and output "profile" run the stock command (profile
  files array_equal to the stock's); bfloat16 and float16 are refused, float64 only on the CPU, both before the model
  is loaded.
- A JSON with a "fast" block gives the stock command the same files as one without it.
- The "fast" options: defaults, the JSON block, flag overrides, unknown and later-round keys refused.
- One summary convergence warning, none for warning_threshold null; the stock group error for a multi-group model
  with group null; random_state null records the drawn seed, and a rerun with it reproduces the attributions.
- A failed forward self-check aborts before anything is written.
- NUMBA_CACHE_DIR defaults to a node-local directory, exported before cherimoya is imported, and
  TRITON_CACHE_AUTOTUNING to 1, so cherimoya's autotuned kernels cache their timings; a value already set is kept.

Measured values are written to the terminal.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("tangermeme")
pytest.importorskip("cherimoya")
pytest.importorskip("cherimoya_cli")
pytest.importorskip("pyfaidx")

ATTRIBUTION = Path(__file__).resolve().parents[1] / "workflows" / "attribution"
sys.path.insert(0, str(ATTRIBUTION))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import attribute_fast  # noqa: E402
import cheri_dls  # noqa: E402
import dls_tiny  # noqa: E402
from cherimoya_cli.commands import attribute as stock_attribute  # noqa: E402

CONTIG_LENGTH = 3000
CHROMS = ("chr1", "chr2", "chr3")
N_RUN = ("chr2", 1500, 1520)
IN_WINDOW, ATTR_WINDOW, N_SHUFFLES = 256, 64, 4
LOCI = (  # (contig, centre); every locus spans centre +- 100, so extract_loci centres its window on `centre`
    ("chr1", 300),
    ("chr1", 900),
    ("chr1", 1600),
    ("chr1", 2400),
    ("chr2", 400),
    ("chr2", 1000),
    ("chr2", 1510),  # its window [1382, 1638) covers the N run: dropped by the N filter
    ("chr2", 2200),
    ("chr2", 2700),
    ("chr3", 500),
    ("chr3", 1800),
    ("chr3", 2950),  # its window ends at 3078, past the contig: dropped by extract_loci
)
KEPT = np.array([True] * 6 + [False] + [True] * 4 + [False])
C11_REL_L2 = 1e-5
MISSING_MODEL = "/nonexistent/cherimoya.torch"  # a test that passes this proves the model is never loaded


def _report(config: pytest.Config, line: str) -> None:
    """Write a measurement to the terminal, past pytest's output capture."""
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(f"[attribute_fast] {line}")


@pytest.fixture(scope="module")
def inputs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    """The FASTA, the loci, and the tiny models saved with model.save(): config A, and config B with 2 groups."""
    root = tmp_path_factory.mktemp("inputs")
    rng = np.random.RandomState(0)
    fasta = root / "genome.fa"
    with open(fasta, "w") as out:
        for name in CHROMS:
            seq = list(rng.choice(list("ACGT"), CONTIG_LENGTH))
            if name == N_RUN[0]:
                seq[N_RUN[1] : N_RUN[2]] = "N" * (N_RUN[2] - N_RUN[1])
            out.write(f">{name}\n")
            text = "".join(seq)
            out.write("\n".join(text[i : i + 60] for i in range(0, len(text), 60)) + "\n")
    loci = root / "peaks.narrowPeak"
    with open(loci, "w") as out:
        for i, (chrom, centre) in enumerate(LOCI):
            out.write(f"{chrom}\t{centre - 100}\t{centre + 100}\tpeak{i}\t0\t.\t1.0\t1.0\t1.0\t100\n")
    model_a, model_b = root / "tiny_a.torch", root / "tiny_b.torch"
    dls_tiny.tiny_model("A", seed=0).save(str(model_a))
    dls_tiny.tiny_model("B", seed=1).save(str(model_b))
    return {"fasta": str(fasta), "loci": str(loci), "model": str(model_a), "model_b": str(model_b)}


def write_params(inputs: dict[str, str], out_dir: Path, name: str, **overrides: object) -> Path:
    """An attribute.json for the synthetic inputs, writing <out_dir>/<name>.{ohe.npz,attr.npz,idx.npy}."""
    params = {
        "model": inputs["model"],
        "sequences": inputs["fasta"],
        "loci": [inputs["loci"]],
        "chroms": list(CHROMS),
        "in_window": IN_WINDOW,
        "attr_window": ATTR_WINDOW,
        "n_shuffles": N_SHUFFLES,
        "batch_size": 64,
        "random_state": 0,
        "dtype": "float32",
        "device": "cpu",
        "output": "counts",
        "algorithm": "deep_lift_shap",
        "verbose": False,
        "ohe_filename": str(out_dir / f"{name}.ohe.npz"),
        "attr_filename": str(out_dir / f"{name}.attr.npz"),
        "idx_filename": str(out_dir / f"{name}.idx.npy"),
    }
    params.update(overrides)
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(params, indent=2))
    return path


def _params(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def load_outputs(path: Path) -> dict:
    """The three files a run of the JSON at `path` wrote: arrays and the npz keys."""
    p = _params(path)
    out = {}
    for name in ("ohe", "attr"):
        with np.load(p[f"{name}_filename"]) as npz:
            out[f"{name}_keys"] = sorted(npz.files)
            out[name] = npz["arr_0"]
    out["idx"] = np.load(p["idx_filename"])
    return out


def output_paths(path: Path) -> list[Path]:
    p = _params(path)
    base = attribute_fast.sidecar_base(p["attr_filename"])
    names = [p["ohe_filename"], p["attr_filename"], p["idx_filename"], base + ".deltas.npy", base + ".meta.json"]
    return [Path(n) for n in names]


def per_locus_rel_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """||a - b|| / ||b|| per locus, in float64."""
    a = a.astype(np.float64).reshape(len(a), -1)
    b = b.astype(np.float64).reshape(len(b), -1)
    return np.linalg.norm(a - b, axis=1) / np.maximum(np.linalg.norm(b, axis=1), 1e-30)


def ns(path: Path) -> argparse.Namespace:
    return argparse.Namespace(parameters=str(path))


@pytest.fixture(scope="module")
def stock_runs(inputs: dict[str, str], tmp_path_factory: pytest.TempPathFactory):
    """The stock `cherimoya attribute` run of the synthetic JSON, once per batch size (and set of extra keys)."""
    cache: dict = {}

    def get(batch_size: int, **overrides: object) -> dict:
        key = (batch_size, json.dumps(overrides, sort_keys=True))
        if key not in cache:
            out = tmp_path_factory.mktemp(f"stock{batch_size}")
            path = write_params(inputs, out, "stock", batch_size=batch_size, **overrides)
            stock_attribute.run(ns(path))
            cache[key] = load_outputs(path)
        return cache[key]

    return get


# ---------------------------------------------------------------------------------------------------------------------
# C11


@pytest.mark.parametrize("batch_size, ref_workers", [(64, 2), (7, 0)], ids=["batch64-pool", "batch7-inline"])
def test_matches_the_stock_command(request, inputs, stock_runs, tmp_path, batch_size, ref_workers):
    stock = stock_runs(batch_size)
    path = write_params(inputs, tmp_path, "fast", batch_size=batch_size, fast={"ref_workers": ref_workers})
    record = attribute_fast.run(ns(path))
    ours = load_outputs(path)

    for name in ("ohe", "attr"):
        assert ours[f"{name}_keys"] == stock[f"{name}_keys"] == ["arr_0"]
    assert ours["ohe"].dtype == stock["ohe"].dtype == np.int8
    assert ours["ohe"].shape == stock["ohe"].shape == (KEPT.sum(), 4, ATTR_WINDOW)
    assert np.array_equal(ours["ohe"], stock["ohe"])
    assert ours["idx"].dtype == stock["idx"].dtype == np.bool_
    assert np.array_equal(ours["idx"], stock["idx"]) and np.array_equal(ours["idx"], KEPT)
    assert ours["attr"].dtype == stock["attr"].dtype == np.float32
    assert ours["attr"].shape == stock["attr"].shape == (KEPT.sum(), 4, ATTR_WINDOW)
    rel = per_locus_rel_l2(ours["attr"], stock["attr"])
    _report(
        request.config,
        f"C11 batch_size {batch_size}, ref_workers {ref_workers}: attr per-locus rel-L2 vs stock median "
        f"{np.median(rel):.3g}, max {rel.max():.3g} (bound {C11_REL_L2:g}); bitwise equal loci "
        f"{int((rel == 0).sum())}/{len(rel)}; self-check max |dy| {record['self_check']['max_abs']:.3g}",
    )
    assert rel.max() <= C11_REL_L2

    ohe, attr, idx, deltas_file, meta_file = output_paths(path)
    deltas = np.load(deltas_file)
    assert deltas.dtype == np.float32 and deltas.shape == (KEPT.sum(), N_SHUFFLES)
    assert np.isfinite(deltas).all()
    meta = json.loads(meta_file.read_text())
    assert meta["n_shuffles"] == N_SHUFFLES and meta["head_rows"] == 2 * batch_size
    assert meta["random_state"] == 0 and meta["random_state_drawn"] is False
    assert meta["precision"] == "reference" and meta["dtype"] == "float32"
    assert meta["forward_stats"] == "cpu_mirror" and meta["compile"] == "none"
    assert meta["inputs"] == {
        "loci": len(LOCI),
        "extracted": len(LOCI) - 1,
        "kept": int(KEPT.sum()),
        "length": IN_WINDOW,
        "window": [IN_WINDOW // 2 - ATTR_WINDOW // 2, IN_WINDOW // 2 + ATTR_WINDOW // 2],
    }
    assert meta["engine"]["references"] == "produced"
    assert meta["deltas"]["pairs"] == KEPT.sum() * N_SHUFFLES and meta["deltas"]["finite"]
    for key in ("load", "extract", "refs", "self_check", "run", "write", "total", "compile", "steady_pairs_per_s"):
        assert key in meta["timings_s"]
    assert meta["memory"]["rss_max_bytes"] > 0
    assert meta["versions"]["cherimoya"] == "0.3.0" and meta["versions"]["tangermeme"] == "1.5.0"
    check = meta["self_check"]
    assert check["pass"] and check["peaks"] == 4 and check["bound"] == attribute_fast.SELF_CHECK_MAX_ABS
    assert check["max_abs"] <= 1e-6  # on the CPU both sides run the reference CPU path, op for op
    assert check["run_outputs"]["max_abs"] <= 1e-6
    assert meta["outputs"]["attr_shape"] == [int(KEPT.sum()), 4, ATTR_WINDOW]

    audit = meta["audit"]
    _report(
        request.config,
        f"audit, batch_size {batch_size}: {audit['peaks']} peaks {audit['indices']} at stock batch "
        f"{audit['batch_size']}: min Pearson {audit['min_pearson']:.10f}, rel-L2 median {audit['median_rel_l2']:.3g} "
        f"max {audit['max_rel_l2']:.3g}; {audit['seconds']:.2f} s",
    )
    assert audit == record["audit"] and audit["pass"] is True and audit["warn"] is False
    assert audit["indices"] == np.round(np.linspace(0, KEPT.sum() - 1, 8)).astype(int).tolist()
    assert audit["batch_size"] == min(batch_size, attribute_fast.AUDIT_MAX_BATCH) and audit["dtype"] == "float32"
    assert audit["references_match_tangermeme"] is True and audit["n_shuffles"] == N_SHUFFLES
    assert audit["precision"] == "reference" and audit["failed_positions"] == []
    assert audit["max_rel_l2"] <= C11_REL_L2 and audit["min_pearson"] >= 0.99999
    assert "audit" in meta["timings_s"] and meta["triton_autotune"]["TRITON_CACHE_AUTOTUNING"]
    assert meta["engine"]["reference_stats"]["first_wait_s"] is not None


def test_writes_in_the_stock_order(inputs, tmp_path, monkeypatch):
    """ohe, attr, idx as attribute.py:115-117 writes them, so the idx file marks a finished run; then the sidecars."""
    path = write_params(inputs, tmp_path, "order", fast={"ref_workers": 0})
    calls = []
    real_save, real_savez = np.save, np.savez_compressed

    def save(file, *args, **kwargs):
        calls.append(("save", str(file)))
        return real_save(file, *args, **kwargs)

    def savez(file, *args, **kwargs):
        calls.append(("savez_compressed", str(file)))
        return real_savez(file, *args, **kwargs)

    monkeypatch.setattr(np, "save", save)
    monkeypatch.setattr(np, "savez_compressed", savez)
    attribute_fast.run(ns(path))
    ohe, attr, idx, deltas, meta = output_paths(path)
    assert calls == [
        ("savez_compressed", str(ohe)),
        ("savez_compressed", str(attr)),
        ("save", str(idx)),
        ("save", str(deltas)),
    ]
    assert meta.exists()


def test_a_stale_done_marker_is_removed_before_writing(inputs, tmp_path, monkeypatch):
    path = write_params(inputs, tmp_path, "stale", fast={"ref_workers": 0})
    ohe, attr, idx, _, _ = output_paths(path)
    idx.write_bytes(b"an earlier run's idx file")
    real_savez = np.savez_compressed

    def savez(file, *args, **kwargs):
        if str(file) == str(attr):
            raise OSError("disk full")
        return real_savez(file, *args, **kwargs)

    monkeypatch.setattr(np, "savez_compressed", savez)
    with pytest.raises(OSError, match="disk full"):
        attribute_fast.run(ns(path))
    assert ohe.exists() and not idx.exists()


# ---------------------------------------------------------------------------------------------------------------------
# the stock command's behaviour, and what attribute_fast.py refuses


def test_skip_returns_before_anything_is_loaded(inputs, tmp_path):
    path = write_params(inputs, tmp_path, "skip", skip=True, model=MISSING_MODEL)
    assert attribute_fast.run(ns(path)) is None
    assert not any(p.exists() for p in output_paths(path))


def test_saturation_mutagenesis_runs_the_stock_command(inputs, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(stock_attribute, "run", lambda args: calls.append(args))

    def forbidden(*args, **kwargs):
        raise AssertionError("the engine must not run for saturation_mutagenesis")

    monkeypatch.setattr(cheri_dls, "Engine", forbidden)
    path = write_params(inputs, tmp_path, "ism", algorithm="saturation_mutagenesis")
    args = ns(path)
    assert attribute_fast.run(args) is None
    assert calls == [args]


def test_an_unknown_algorithm_raises_the_stock_error(inputs, tmp_path):
    path = write_params(inputs, tmp_path, "algo", algorithm="integrated_gradients", model=MISSING_MODEL)
    with pytest.raises(ValueError, match="algorithm must be either `deep_lift_shap` or `saturation_mutagenesis`"):
        attribute_fast.run(ns(path))


@pytest.mark.parametrize(
    "dtype, device, message",
    [
        ("bfloat16", "cpu", "autocast"),
        ("float16", "cpu", "autocast"),
        ("bfloat16", "cuda", "autocast"),
        ("float64", "cuda", "CPU only"),
        ("float64", "cuda:0", "CPU only"),
        ("double", "cpu", "float32 or null"),
    ],
)
def test_dtypes_are_refused_before_the_model_is_loaded(inputs, tmp_path, dtype, device, message):
    path = write_params(inputs, tmp_path, "dtype", dtype=dtype, device=device, model=MISSING_MODEL)
    with pytest.raises(ValueError, match=message):
        attribute_fast.run(ns(path))


def test_float64_runs_on_the_cpu(request, inputs, stock_runs, tmp_path):
    stock = stock_runs(64)
    path = write_params(inputs, tmp_path, "f64", dtype="float64", fast={"ref_workers": 0})
    record = attribute_fast.run(ns(path))
    ours = load_outputs(path)
    assert record["dtype"] == "float64" and ours["attr"].dtype == np.float32
    assert np.array_equal(ours["ohe"], stock["ohe"]) and np.array_equal(ours["idx"], stock["idx"])
    rel = per_locus_rel_l2(ours["attr"], stock["attr"])
    _report(
        request.config,
        f"float64 engine vs the float32 stock: attr per-locus rel-L2 max {rel.max():.3g}; "
        f"self-check (float64 ours vs float32 module) {record['self_check']['max_abs']:.3g}",
    )
    assert rel.max() <= C11_REL_L2
    assert record["self_check"]["pass"]


def test_profile_runs_the_stock_command(inputs, stock_runs, tmp_path, caplog, monkeypatch):
    """output "profile" is handed to the stock `cherimoya attribute`, with a warning that says so: its three files
    equal, array for array, a stock run of the same JSON, and attribute_fast.py writes no sidecar of its own."""

    def forbidden(*args, **kwargs):
        raise AssertionError("the engine must not run for the profile head")

    monkeypatch.setattr(cheri_dls, "Engine", forbidden)
    stock = stock_runs(64, output="profile")
    path = write_params(inputs, tmp_path, "profile", output="profile", fast={"precision": "fp32", "audit": 8})
    with caplog.at_level("WARNING", logger="attribute_fast"):
        assert attribute_fast.run(ns(path)) is None
    assert any("profile head runs the stock" in r.getMessage() for r in caplog.records)
    ours = load_outputs(path)
    for name in ("ohe", "attr", "idx"):
        assert ours[name].dtype == stock[name].dtype and np.array_equal(ours[name], stock[name]), name
    assert ours["attr"].shape == (KEPT.sum(), 4, ATTR_WINDOW) and ours["attr_keys"] == ["arr_0"]
    _, _, _, deltas_file, meta_file = output_paths(path)
    assert not deltas_file.exists() and not meta_file.exists()


def test_a_missing_output_directory_is_refused_before_the_model_is_loaded(inputs, tmp_path):
    missing = tmp_path / "missing" / "x.idx.npy"
    path = write_params(inputs, tmp_path, "dirs", idx_filename=str(missing), model=MISSING_MODEL)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        attribute_fast.run(ns(path))


def test_a_multi_group_model_without_a_group_raises_the_stock_error(inputs, tmp_path):
    ours_path = write_params(inputs, tmp_path, "ours_group", model=inputs["model_b"], group=None)
    stock_path = write_params(inputs, tmp_path, "stock_group", model=inputs["model_b"], group=None)
    with pytest.raises(ValueError, match="deep_lift_shap attributes one output") as ours:
        attribute_fast.run(ns(ours_path))
    with pytest.raises(ValueError, match="deep_lift_shap attributes one output") as stock:
        stock_attribute.run(ns(stock_path))
    assert str(ours.value) == str(stock.value)


def test_the_stock_command_ignores_a_fast_block(inputs, stock_runs):
    plain = stock_runs(64)
    fast = {"precision": "fp32", "seqs_per_step": 3, "mem_budget_gb": 4, "ref_workers": 0, "deterministic": True}
    with_block = stock_runs(64, fast=fast)
    for name in ("ohe", "attr", "idx"):
        assert np.array_equal(with_block[name], plain[name])


# ---------------------------------------------------------------------------------------------------------------------
# the "fast" options


def test_fast_options_resolve_defaults_block_and_flags():
    assert attribute_fast.fast_options(None) == attribute_fast.FAST_DEFAULTS
    flags = argparse.Namespace(precision="fp32", seqs_per_step=None, ref_workers=0)
    options = attribute_fast.fast_options({"precision": "reference", "seqs_per_step": 4, "mem_budget_gb": None}, flags)
    assert options["precision"] == "fp32"  # the flag wins
    assert options["seqs_per_step"] == 4  # the block wins over the default
    assert options["mem_budget_gb"] == attribute_fast.FAST_DEFAULTS["mem_budget_gb"]  # null keeps the default
    assert options["ref_workers"] == 0
    assert options["audit"] == 8  # the default
    assert attribute_fast.fast_options({"audit": 3})["audit"] == 3
    assert attribute_fast.fast_options({"audit": 3}, argparse.Namespace(audit=0))["audit"] == 0  # the flag wins
    assert attribute_fast.fast_options({"audit": None})["audit"] == 8
    # keys of later rounds are accepted only with this version's behaviour
    attribute_fast.fast_options({"engine": "dls", "self_check": True, "resume": None})
    with pytest.raises(ValueError, match="not implemented"):
        attribute_fast.fast_options({"engine": "tangermeme"})
    with pytest.raises(ValueError, match="not implemented"):
        attribute_fast.fast_options({"resume": True})
    with pytest.raises(ValueError, match='unknown key "fast.precison"'):
        attribute_fast.fast_options({"precison": "fp32"})
    bad = [
        {"precision": "bf16"},
        {"seqs_per_step": 0},
        {"seqs_per_step": True},
        {"seqs_per_step": "4"},
        {"mem_budget_gb": -1},
        {"compile": "max-autotune"},
        {"ref_workers": -1},
        {"forward_stats": "three_pass"},
        {"deterministic": "yes"},
        {"audit": -1},
        {"audit": True},
        {"audit": 1.5},
        {"audit": "8"},
    ]
    for block in bad:
        with pytest.raises(ValueError):
            attribute_fast.fast_options(block)
    with pytest.raises(ValueError, match="JSON object"):
        attribute_fast.fast_options(["precision", "fp32"])


def test_a_bad_fast_block_is_refused_before_the_model_is_loaded(inputs, tmp_path):
    path = write_params(inputs, tmp_path, "badfast", fast={"precison": "fp32"}, model=MISSING_MODEL)
    with pytest.raises(ValueError, match="unknown key"):
        attribute_fast.run(ns(path))


def test_flags_override_the_fast_block(inputs, stock_runs, tmp_path):
    stock = stock_runs(64)
    path = write_params(inputs, tmp_path, "flags", fast={"seqs_per_step": 5, "ref_workers": 0})
    assert attribute_fast.main(["-p", str(path), "--seqs-per-step", "3", "--precision", "fp32"]) == 0
    meta = json.loads(output_paths(path)[-1].read_text())
    assert meta["fast"]["seqs_per_step"] == 3 and meta["seqs_per_step"] == 3 and meta["engine"]["steps"] == 4
    assert meta["precision"] == "fp32" and meta["fast"]["ref_workers"] == 0
    rel = per_locus_rel_l2(load_outputs(path)["attr"], stock["attr"])
    assert rel.max() <= C11_REL_L2


# ---------------------------------------------------------------------------------------------------------------------
# warnings, seeds and the self-check


def _delta_warnings(caught: list) -> list:
    return [
        w
        for w in caught
        if issubclass(w.category, RuntimeWarning) and str(w.message).startswith("Convergence deltas too high")
    ]


@pytest.mark.parametrize("threshold, expected", [(1e-12, 1), (None, 0)], ids=["tiny-threshold", "null"])
def test_one_summary_warning_or_none(request, inputs, tmp_path, threshold, expected):
    path = write_params(inputs, tmp_path, "warn", warning_threshold=threshold, fast={"ref_workers": 0})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        record = attribute_fast.run(ns(path))
    found = _delta_warnings(caught)
    if found:
        _report(request.config, f"the summary warning: {found[0].message}")
    assert len(found) == expected
    if threshold is None:
        assert record["deltas"]["above"] is None
    else:
        assert record["deltas"]["above"] >= 1


def test_random_state_null_records_the_drawn_seed(inputs, tmp_path):
    drawn_path = write_params(inputs, tmp_path, "drawn", random_state=None, fast={"ref_workers": 0})
    record = attribute_fast.run(ns(drawn_path))
    seed = record["random_state"]
    assert isinstance(seed, int) and record["random_state_drawn"] is True
    assert json.loads(output_paths(drawn_path)[-1].read_text())["random_state"] == seed
    again_path = write_params(inputs, tmp_path, "again", random_state=seed, fast={"ref_workers": 0})
    attribute_fast.run(ns(again_path))
    assert np.array_equal(load_outputs(again_path)["attr"], load_outputs(drawn_path)["attr"])


def test_a_failed_self_check_aborts_before_anything_is_written(inputs, tmp_path, monkeypatch):
    """Weights that are not the model's (the count Linear scaled by 1.05) must fail the check, before the run."""
    original = cheri_dls.CheriWeights.from_model

    def perturbed(cls, model, device=None, **kwargs):
        weights = original(model, device, **kwargs)
        weights.Wl.mul_(1.05)
        return weights

    monkeypatch.setattr(cheri_dls.CheriWeights, "from_model", classmethod(perturbed))
    runs = []
    monkeypatch.setattr(cheri_dls.Engine, "run", lambda self, *args, **kwargs: runs.append(args))
    path = write_params(inputs, tmp_path, "selfcheck", fast={"ref_workers": 0})
    with pytest.raises(RuntimeError, match="forward self-check failed"):
        attribute_fast.run(ns(path))
    assert not runs
    assert not any(p.exists() for p in output_paths(path))


# ---------------------------------------------------------------------------------------------------------------------
# the audit


def _corrupting_run(monkeypatch, corrupt) -> list:
    """Make Engine.run hand back its result with corrupt(attr) applied: an engine whose output is wrong."""
    real_run = cheri_dls.Engine.run
    calls = []

    def run(self, *args, **kwargs):
        result = real_run(self, *args, **kwargs)
        calls.append(kwargs.get("keep_references"))
        return result._replace(attr=corrupt(result.attr.clone()))

    monkeypatch.setattr(cheri_dls.Engine, "run", run)
    return calls


def _scale_one_peak(attr):
    attr[3] *= 1.05  # rel-L2 0.05 > 1e-2, Pearson still 1
    return attr


def _swap_two_peaks(attr):
    attr[[3, 7]] = attr[[7, 3]]  # Pearson far below 0.9999
    return attr


@pytest.mark.parametrize("corrupt", [_scale_one_peak, _swap_two_peaks], ids=["scaled", "swapped"])
def test_a_corrupted_engine_output_fails_the_audit_after_writing(request, inputs, tmp_path, monkeypatch, corrupt):
    calls = _corrupting_run(monkeypatch, corrupt)
    path = write_params(inputs, tmp_path, "corrupt", fast={"ref_workers": 0})
    with pytest.raises(attribute_fast.AuditFailed, match="audit failed") as failed:
        attribute_fast.run(ns(path))
    assert all(p.exists() for p in output_paths(path)), "every output is written before the audit fails"
    meta = json.loads(output_paths(path)[-1].read_text())
    audit = meta["audit"]
    _report(request.config, f"audit of a {corrupt.__name__} output: {failed.value}")
    peaks = np.round(np.linspace(0, KEPT.sum() - 1, 8)).astype(int).tolist()
    assert len(calls) == 1 and np.array_equal(calls[0], peaks)
    bad = [peaks.index(i) for i in (3, 7) if i in peaks and (corrupt is _swap_two_peaks or i == 3)]
    assert audit["pass"] is False and audit["failed_positions"] == bad
    assert audit["references_match_tangermeme"] is True
    if corrupt is _scale_one_peak:
        assert audit["rel_l2"][bad[0]] == pytest.approx(0.05, rel=1e-3) and audit["min_pearson"] > 0.99999
    else:
        assert max(audit["pearson"][i] for i in bad) < 0.9999
    # through the command line: the same outputs, and exit status 3
    assert attribute_fast.main(["-p", str(path)]) == attribute_fast.AUDIT_FAILED_EXIT


def test_references_unlike_tangermemes_fail_the_audit(inputs, tmp_path, monkeypatch):
    """If the engine's references for the audited peaks were not tangermeme's draw, the audit fails even though the
    stock run, given the same references, agrees with the engine."""
    real = attribute_fast._tangermeme_references

    def shifted(X, peaks, n_shuffles, random_state):
        return real(X, peaks, n_shuffles, random_state + 1)

    monkeypatch.setattr(attribute_fast, "_tangermeme_references", shifted)
    path = write_params(inputs, tmp_path, "refs", fast={"ref_workers": 0})
    with pytest.raises(attribute_fast.AuditFailed, match="references equal to tangermeme's: False"):
        attribute_fast.run(ns(path))
    audit = json.loads(output_paths(path)[-1].read_text())["audit"]
    assert audit["references_match_tangermeme"] is False and audit["failed_positions"] == []
    assert audit["references_mismatched_positions"] == list(range(8))


def test_audit_zero_runs_no_audit(inputs, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("--audit 0 must not audit")

    monkeypatch.setattr(attribute_fast, "run_audit", forbidden)
    calls = _corrupting_run(monkeypatch, lambda attr: attr)
    path = write_params(inputs, tmp_path, "noaudit", fast={"ref_workers": 0, "audit": 8})
    assert attribute_fast.main(["-p", str(path), "--audit", "0"]) == 0
    meta = json.loads(output_paths(path)[-1].read_text())
    assert meta["fast"]["audit"] == 0 and meta["audit"] == {"peaks": 0, "pass": None, "status": "off (audit 0)"}
    assert calls == [None]  # the run keeps no references


def test_audit_peaks_are_evenly_spaced_and_distinct():
    assert attribute_fast.audit_peaks(10, 8).tolist() == [0, 1, 3, 4, 5, 6, 8, 9]
    assert attribute_fast.audit_peaks(4999, 8).tolist() == [0, 714, 1428, 2142, 2856, 3570, 4284, 4998]
    assert attribute_fast.audit_peaks(5, 8).tolist() == [0, 1, 2, 3, 4]
    assert attribute_fast.audit_peaks(5, 0).tolist() == [] and attribute_fast.audit_peaks(0, 8).tolist() == []
    assert attribute_fast.audit_peaks(1, 1).tolist() == [0]
    for n in range(1, 60):
        for k in range(1, n + 1):
            peaks = attribute_fast.audit_peaks(n, k)
            assert len(np.unique(peaks)) == k and peaks[0] == 0 and peaks[-1] == (n - 1 if k > 1 else 0)


def test_compare_attributions_caps_and_warning(caplog):
    rng = np.random.default_rng(0)
    stock = rng.normal(size=(6, 4, 50))
    same = attribute_fast.compare_attributions(stock, stock)
    assert same["pass"] and not same["warn"] and same["max_rel_l2"] == 0.0 and same["min_pearson"] == pytest.approx(1)

    # a median rel-L2 of 5e-4: within the caps, so it passes, but above 3e-4, so it warns
    noise = rng.normal(size=stock.shape)
    noise *= (
        5e-4
        * np.linalg.norm(stock.reshape(6, -1), axis=1)[:, None, None]
        / np.linalg.norm(noise.reshape(6, -1), axis=1)[:, None, None]
    )
    near = attribute_fast.compare_attributions(stock + noise, stock)
    assert near["pass"] and near["warn"] and near["median_rel_l2"] == pytest.approx(5e-4, rel=1e-6)

    # a peak with a large constant offset: its centred values decorrelate (Pearson < 0.9999) while its rel-L2,
    # relative to the offset's norm, stays under 1e-2: the Pearson cap alone fails it
    offset = stock.copy()
    offset[2] = 100.0 + 0.1 * rng.normal(size=offset[2].shape)
    ours = offset.copy()
    ours[2] += 0.02 * rng.normal(size=ours[2].shape)
    pearson_only = attribute_fast.compare_attributions(ours, offset)
    assert pearson_only["rel_l2"][2] <= 1e-2 and pearson_only["pearson"][2] < 0.9999
    assert not pearson_only["pass"] and pearson_only["failed_positions"] == [2]

    nan = stock.copy()
    nan[1, 0, 0] = np.nan
    assert attribute_fast.compare_attributions(nan, stock)["failed_positions"] == [1]
    with pytest.raises(ValueError, match="equal, non-empty"):
        attribute_fast.compare_attributions(stock[:2], stock)

    audit = dict(near, peaks=6, batch_size=20, precision="reference", references_match_tangermeme=True, seconds=1.0)
    with caplog.at_level("INFO", logger="attribute_fast"):
        attribute_fast._log_audit(audit)
    assert [r.levelname for r in caplog.records] == ["WARNING", "INFO"]
    assert "above 0.0003" in caplog.records[0].getMessage()


# ---------------------------------------------------------------------------------------------------------------------
# NUMBA_CACHE_DIR


def test_the_numba_cache_defaults_to_a_node_local_directory(tmp_path, monkeypatch):
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    assert attribute_fast.default_numba_cache_dir() == str(tmp_path / "numba_cache")
    assert attribute_fast.ensure_numba_cache_dir() == str(tmp_path / "numba_cache")
    assert os.environ["NUMBA_CACHE_DIR"] == str(tmp_path / "numba_cache") and (tmp_path / "numba_cache").is_dir()
    monkeypatch.delenv("TMPDIR")
    assert attribute_fast.default_numba_cache_dir() == os.path.join("/dev/shm", attribute_fast._user(), "numba_cache")
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(tmp_path / "set"))
    assert attribute_fast.ensure_numba_cache_dir() == str(tmp_path / "set")  # a set variable is kept


def test_the_numba_cache_is_exported_before_cherimoya_is_imported(tmp_path):
    """In a fresh process: importing attribute_fast imports nothing heavy and exports NUMBA_CACHE_DIR, so the numba
    kernel tangermeme compiles when cheri_dls imports cherimoya is cached there, not in the installed environment."""
    code = textwrap.dedent(
        f"""
        import os, sys
        sys.path.insert(0, {str(ATTRIBUTION)!r})
        import attribute_fast
        heavy = sorted(m for m in ("torch", "numba", "cherimoya", "tangermeme") if m in sys.modules)
        assert not heavy, heavy
        import cheri_dls
        from tangermeme import ersatz
        print(os.environ["NUMBA_CACHE_DIR"])
        print(ersatz._fast_shuffle._cache.cache_path)
        """
    )
    env = {k: v for k, v in os.environ.items() if k != "NUMBA_CACHE_DIR"}
    env["TMPDIR"] = str(tmp_path)
    done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-3000:]
    exported, kernel_cache = done.stdout.split()[-2:]
    assert exported == str(tmp_path / "numba_cache")
    assert Path(kernel_cache).resolve().is_relative_to((tmp_path / "numba_cache").resolve())


@pytest.mark.parametrize("preset", [None, "0"], ids=["unset", "set-to-0"])
def test_triton_autotune_caching_is_exported_before_cherimoya_is_imported(tmp_path, preset):
    """In a fresh process: importing attribute_fast exports TRITON_CACHE_AUTOTUNING=1 when it is unset (and keeps a
    value already set), before cherimoya is imported, so cherimoya's autotuned kernels cache their timings."""
    code = textwrap.dedent(
        f"""
        import os, sys
        sys.path.insert(0, {str(ATTRIBUTION)!r})
        import attribute_fast
        print(os.environ.get("TRITON_CACHE_AUTOTUNING"))
        print(attribute_fast.TRITON_AUTOTUNE["set_by_attribute_fast"], attribute_fast.TRITON_AUTOTUNE["cherimoya_imported"])
        import cherimoya.cheri as cheri
        print(cheri.HAS_TRITON and cheri._fwd_stats_kernel.cache_results)
        """
    )
    env = {k: v for k, v in os.environ.items() if k != "TRITON_CACHE_AUTOTUNING"}
    env["TMPDIR"] = str(tmp_path)
    env.setdefault("NUMBA_CACHE_DIR", str(tmp_path / "numba_cache"))
    if preset is not None:
        env["TRITON_CACHE_AUTOTUNING"] = preset
    done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-3000:]
    value, flags, cached = done.stdout.strip().splitlines()[-3:]
    if preset is None:
        assert (value, flags, cached) == ("1", "True False", "True")
    else:
        assert (value, flags, cached) == ("0", "False False", "False")
