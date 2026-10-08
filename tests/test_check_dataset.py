"""to_chrombpnet_scores.py --check-dataset and --check-average (CPU only, synthetic).

Each test builds a dataset whose 5 folds attr_status.py calls DONE and the converter has converted, on a synthetic
genome: peaks 3,000 bp apart on one chromosome, so no two 2,114-bp windows overlap; each kept row's ohe is the one-hot
of the central W bases of its window, and the dropped rows' windows hold an N (one inside the central W, one outside
it). A fake ChromBPNet 05.0 tree beside it holds, per fold, the peaks file as interpreted_regions.bed and the one-hot
of every full window as raw/seq, with a column of zeros at each N; each fold's compare.json points at it, as
make_jobs.py writes it.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
h5py = pytest.importorskip("h5py")
pytest.importorskip("hdf5plugin")
pytest.importorskip("yaml")

TESTS = Path(__file__).resolve().parent
CHERIMOYA = TESTS.parent / "workflows" / "cherimoya"
sys.path.insert(0, str(CHERIMOYA))
sys.path.insert(0, str(TESTS))

import attr_status as st  # noqa: E402
import to_chrombpnet_scores as tcs  # noqa: E402

W, L, K = 12, 2114, 3
FOLDS = ("0", "1", "2", "3", "4")
N_ROWS = 24
N_ROWS_DROPPED = {5: 0, 17: -900}  # row: offset of its N from the window's centre (inside / outside the central W)
OFFSET, SPACING = 5000, 3000
ACGT = "ACGT"


# ---------------------------------------------------------------------------------------------------------------------
# a synthetic dataset


def _rows(n_rows: int) -> list[tuple[int, int]]:
    return [(OFFSET + SPACING * i, OFFSET + SPACING * i + 1000) for i in range(n_rows)]


def _genome(path: Path, n_rows: int, dropped: dict[int, int], seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    seq = list("".join(rng.choice(list(ACGT), size=OFFSET + SPACING * n_rows + 5000)))
    for i, offset in dropped.items():
        start, end = _rows(n_rows)[i]
        mid = start + (end - start) // 2
        seq[mid + offset] = "N"
    text = "".join(seq)
    path.write_text(">chr1\n" + "\n".join(text[i : i + 60] for i in range(0, len(text), 60)) + "\n")
    return text


def _onehot(text: str) -> np.ndarray:
    out = np.zeros((4, len(text)), np.int8)
    for j, base in enumerate(text):
        if base in ACGT:
            out[ACGT.index(base), j] = 1
    return out


def _window(genome: str, row: tuple[int, int], width: int) -> str:
    mid = row[0] + (row[1] - row[0]) // 2
    return genome[mid - width // 2 : mid + width // 2]


def make_root(tmp_path: Path) -> Path:
    """<tmp>/root with the pin of a fake engine (in <tmp>/engine, as test_to_chrombpnet_scores makes it)."""
    root = tmp_path / "root"
    engine = tmp_path / "engine"
    if not root.exists():
        engine.mkdir(parents=True)
        for name in st.CODE_FILES:
            (engine / name).write_text(f"# {name}\n")
        root.mkdir()
        assert st.write_pin(engine, root / st.PIN_NAME)[0] == "written"
    return root


def make_dataset(
    tmp_path: Path,
    ds: str = "ds",
    *,
    n_rows: int = N_ROWS,
    dropped: dict[int, int] | None = None,
    chrombpnet: bool = True,
    convert: bool = True,
    done_folds: tuple[str, ...] = FOLDS,
    seed: int = 0,
) -> dict:
    """A dataset of 5 folds under <tmp>/root: config, genome, peaks, DONE attributions, conversions, ChromBPNet."""
    dropped = dict(N_ROWS_DROPPED if dropped is None else dropped)
    root = make_root(tmp_path)
    data = tmp_path / "data" / ds
    data.mkdir(parents=True)
    genome = _genome(data / "genome.fa", n_rows, dropped, seed)
    peaks = data / "peaks.narrowPeak"
    rows = _rows(n_rows)
    peaks.write_text("".join(f"chr1\t{s}\t{e}\t{ds}_peak_{i + 1}\t0\t.\t{i % 9 + 1}.5\t-1\t-1\t500\n"
                             for i, (s, e) in enumerate(rows)))  # fmt: skip
    config = root / "configs" / ds / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        f'dataset_name: {ds}\nassay: ATAC\nsignal_path: "/data/{ds}.bw"   # never read\n'
        f'regions: "{peaks}"   # never read\noutput_dir: "{root}/{ds}/downstream"\npeak_type: all\n'
        f'folds: {json.dumps(list(FOLDS))}\ncontrib_heads: ["counts"]\nmodisco_max_seqlets: 500000\n'
    )
    idx = np.ones(n_rows, dtype=bool)
    idx[list(dropped)] = False
    kept = np.flatnonzero(idx)
    ohe = np.stack([_onehot(_window(genome, rows[i], W)) for i in kept]) if kept.size else np.zeros((0, 4, W), np.int8)
    cbp_root = tmp_path / "chrombpnet" / ds / "full_models"
    folds = {}
    for k in FOLDS:
        fold_dir = root / ds / f"fold_{k}"
        fold_dir.mkdir(parents=True)
        model = fold_dir / f"cherimoya.fold_{k}.torch"
        model.write_bytes(b"weights")
        (fold_dir / f"cherimoya.fold_{k}.test.performance.tsv").write_text("x\n")
        (fold_dir / "fit.json").write_text("{}")
        evaluation = cbp_root / f"{ds}_all_fold_{k}" / "evaluation"
        (fold_dir / "compare.json").write_text(json.dumps({"chrombpnet_predictions": str(evaluation / "p.h5")}))
        params = {
            "model": str(model), "sequences": str(data / "genome.fa"), "loci": [str(peaks)], "chroms": ["chr1"],
            "in_window": L, "attr_window": W, "exclusion_lists": None, "algorithm": "deep_lift_shap",
            "output": "counts", "group": 0, "n_shuffles": K, "batch_size": 64, "warning_threshold": 0.001,
            "random_state": 1234 + 100 * int(k), "dtype": "float32", "device": "cuda", "compile": False,
            "verbose": False, "ohe_filename": str(fold_dir / "attr.ohe.npz"),
            "attr_filename": str(fold_dir / "attr.attr.npz"), "idx_filename": str(fold_dir / "attr.idx.npy"),
            "fast": {"precision": "reference", "mem_budget_gb": 7.5, "ref_workers": 3, "audit": 8},
        }  # fmt: skip
        (fold_dir / "attribute.json").write_text(json.dumps(params, indent=4))
        now = time.time()
        os.utime(model, (now - 1000, now - 1000))
        rng = np.random.default_rng(100 + int(k) + seed)
        attr = (rng.uniform(-1.0, 1.0, size=ohe.shape)).astype(np.float32)
        fold = {"root": root, "dir": fold_dir, "params": params, "json": fold_dir / "attribute.json",
                "ohe": ohe, "attr": attr, "idx": idx, "stamp": now - 500}  # fmt: skip
        if k in done_folds:
            write_attribution(fold, ohe, attr, idx, stamp=fold["stamp"])
            assert st.check_fold(fold_dir)[0] == "done", st.check_fold(fold_dir)
            if convert:
                assert tcs.convert(fold["json"])[0] == "converted"
        if chrombpnet:
            interp = cbp_root / f"{ds}_all_fold_{k}" / "interpretation"
            interp.mkdir(parents=True)
            shutil.copy(peaks, interp / "interpretation.interpreted_regions.bed")
            raw = np.stack([_onehot(_window(genome, r, L)) for r in rows])
            with h5py.File(interp / "interpretation.counts_scores.h5", "w") as fh:
                fh.create_dataset("raw/seq", data=raw, chunks=(min(16, n_rows), 4, L))
        folds[k] = fold
    return {"root": root, "ds": ds, "config": config, "genome": genome, "peaks": peaks, "rows": rows, "idx": idx,
            "ohe": ohe, "folds": folds, "chrombpnet": cbp_root, "out": root / ds / "downstream"}  # fmt: skip


def write_attribution(fold: dict, ohe, attr, idx, *, stamp: float) -> None:
    """What attribute_fast.py leaves: the npz files, the idx, the deltas, a meta with a passed audit."""
    params, files = fold["params"], st.fold_files(fold["params"])
    np.savez_compressed(params["ohe_filename"], ohe)
    np.savez_compressed(params["attr_filename"], attr)
    np.save(params["idx_filename"], idx)
    np.save(files["deltas"], np.zeros((len(attr), K), np.float32))
    n_rows, n_kept = int(idx.size), int(idx.sum())
    engine = fold["root"].parent / "engine"
    files["meta"].write_text(json.dumps({
        "tool": "attribute_fast.py", "parameters": dict(params, skip=False),
        "random_state": params["random_state"], "n_shuffles": params["n_shuffles"],
        "inputs": {"loci": n_rows, "extracted": n_rows, "kept": n_kept, "length": L},
        "outputs": {"ohe": params["ohe_filename"], "attr": params["attr_filename"], "idx": params["idx_filename"],
                    "attr_shape": [n_kept, 4, W], "attr_finite": True},
        "deltas": {"finite": True}, "audit": {"peaks": 8, "pass": True, "status": "pass"},
        "code": {"files_sha256": st.engine_hashes(engine), "git_commit": "abc"},
    }))  # fmt: skip
    for path in (files["ohe"], files["attr"], files["idx"], files["deltas"]):
        os.utime(path, (stamp, stamp))
    os.utime(files["meta"], (stamp + 1, stamp + 1))


def average_as_06_0(data: dict, order: tuple[str, ...] = FOLDS, path: Path | None = None) -> Path:
    """06.0's average_contrib_scores.py on the converted folds, line for line: float16 copy, +=, /= n, gzip."""
    ds = data["ds"]
    out = path or data["out"] / "contrib_scores" / ds / f"{ds}_average_shaps.counts.h5"
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = avg_projected = avg_shap = None
    n = 0
    for k in order:
        h5 = data["out"] / "full_models" / f"{ds}_all_fold_{k}" / "interpretation" / "interpretation.counts_scores.h5"
        with h5py.File(h5, "r") as fh:
            projected = np.array(fh["projected_shap"]["seq"])
            shap = np.array(fh["shap"]["seq"])
            if raw is None:
                raw = np.array(fh["raw"]["seq"])
                avg_projected = projected.copy()
                avg_shap = shap.copy()
            else:
                avg_projected += projected
                avg_shap += shap
        n += 1
    avg_projected /= n
    avg_shap /= n
    with h5py.File(out, "w") as fh:
        fh.create_dataset("raw/seq", data=raw, compression="gzip")
        fh.create_dataset("shap/seq", data=avg_shap, compression="gzip")
        fh.create_dataset("projected_shap/seq", data=avg_projected, compression="gzip")
    return out


def average_from_config(config: str) -> Path:
    """test_downstream's fake 06.0: average_as_06_0 on the dataset of a Cherimoya config, in its fold order."""
    import yaml

    cfg = yaml.safe_load(Path(config).read_text())
    return average_as_06_0({"ds": cfg["dataset_name"], "out": Path(cfg["output_dir"])}, order=tuple(cfg["folds"]))


def _cbp_h5(data: dict, k: str) -> Path:
    return data["chrombpnet"] / f"{data['ds']}_all_fold_{k}" / "interpretation" / "interpretation.counts_scores.h5"


def _check(data: dict, **kwargs) -> dict:
    return tcs.check_dataset(data["root"], data["ds"], block_rows=7, **kwargs)


def _failures(report: dict) -> str:
    return " | ".join(report["conversions"]["failures"] + [m for r in report["v1"].values() for m in r["failures"]])


# ---------------------------------------------------------------------------------------------------------------------
# --check-dataset


def test_v1_against_chrombpnet_passes_on_every_row_of_the_first_fold_and_samples_of_the_others(tmp_path):
    data = make_dataset(tmp_path)
    report = _check(data, against="chrombpnet", n_sample=5)
    assert report["pass"], _failures(report)
    first = report["v1"]["0"]
    n_kept = N_ROWS - len(N_ROWS_DROPPED)
    assert first["rows"] == "all" and first["rows_compared"] == n_kept
    assert first["window_columns"] == [L // 2 - W // 2, L // 2 + W // 2]
    assert first["cbp_only_rows"] == len(N_ROWS_DROPPED) and first["zero_column_rows"] == len(N_ROWS_DROPPED)
    for k in FOLDS[1:]:
        assert report["v1"][k]["rows_compared"] == 5 and report["v1"][k]["cbp_only_rows_with_zero_column"] == 2
    assert {r["idx_sha256"] for r in report["conversions"]["folds"].values()} == {
        report["conversions"]["folds"]["0"]["idx_sha256"]
    }


def test_v1_against_the_genome_passes(tmp_path):
    data = make_dataset(tmp_path, chrombpnet=False)
    report = _check(data, against="fasta", n_sample=8)
    assert report["pass"], _failures(report)
    assert all(r["rows_compared"] == 8 and r["dropped_rows"] == 2 for r in report["v1"].values())


def test_auto_takes_chrombpnet_when_every_fold_has_its_files_and_the_genome_otherwise(tmp_path):
    data = make_dataset(tmp_path)
    assert _check(data)["against"] == "chrombpnet"
    _cbp_h5(data, "3").unlink()
    report = _check(data)
    assert report["against"] == "fasta" and report["pass"], _failures(report)
    failed = _check(data, against="chrombpnet")
    assert not failed["pass"] and "no " in failed["v1"]["3"]["failures"][0]


@pytest.mark.parametrize("shift", [1, -1])
def test_a_window_off_centre_fails_v1(tmp_path, shift):
    data = make_dataset(tmp_path)
    for k in FOLDS:
        with h5py.File(_cbp_h5(data, k), "r+") as fh:
            fh["raw/seq"][...] = np.roll(fh["raw/seq"][...], shift, axis=2)
    report = _check(data, against="chrombpnet", n_sample=5)
    assert not report["pass"]
    assert "the ohe is not ChromBPNet's raw/seq[:, :, 1051:1063]" in report["v1"]["0"]["failures"][0]


def test_another_base_order_fails_v1(tmp_path):
    data = make_dataset(tmp_path)
    with h5py.File(_cbp_h5(data, "0"), "r+") as fh:
        fh["raw/seq"][...] = fh["raw/seq"][...][:, [1, 0, 2, 3], :]  # CAGT
    report = _check(data, against="chrombpnet", n_sample=5)
    assert not report["pass"] and report["v1"]["0"]["ohe_mismatch_rows"] > 0
    assert all(report["v1"][k]["pass"] for k in FOLDS[1:])


def test_a_dropped_row_chrombpnet_reads_as_all_acgt_fails_v1(tmp_path):
    data = make_dataset(tmp_path)
    for k in FOLDS:
        with h5py.File(_cbp_h5(data, k), "r+") as fh:
            raw = fh["raw/seq"][...]
            raw[5] = _onehot("A" * L)
            fh["raw/seq"][...] = raw
    report = _check(data, against="chrombpnet", n_sample=5)
    assert not report["pass"]
    assert all("column of zeros" in " ".join(report["v1"][k]["failures"]) for k in FOLDS)


def test_rows_joined_by_name_must_agree_in_coordinates(tmp_path):
    data = make_dataset(tmp_path)
    bed = data["chrombpnet"] / "ds_all_fold_2" / "interpretation" / "interpretation.interpreted_regions.bed"
    lines = bed.read_text().splitlines(keepends=True)
    lines[3] = lines[3].replace(f"\t{data['rows'][3][0]}\t", f"\t{data['rows'][3][0] + 1}\t")
    bed.write_text("".join(lines))
    report = _check(data, against="chrombpnet", n_sample=5)
    assert not report["pass"] and report["v1"]["2"]["coordinate_mismatches"] == 1


def test_the_genome_check_finds_a_base_that_differs(tmp_path):
    data = make_dataset(tmp_path, chrombpnet=False)
    fasta = Path(data["folds"]["0"]["params"]["sequences"])
    start, end = data["rows"][3]
    mid = start + (end - start) // 2
    genome = list(data["genome"])
    genome[mid] = {"A": "C", "C": "A", "G": "T", "T": "G"}[genome[mid]]
    text = "".join(genome)
    fasta.write_text(">chr1\n" + "\n".join(text[i : i + 60] for i in range(0, len(text), 60)) + "\n")
    Path(f"{fasta}.fai").unlink(missing_ok=True)
    report = _check(data, against="fasta", n_sample=N_ROWS)
    assert not report["pass"]
    assert all(r["onehot_mismatch_rows"] == 1 for r in report["v1"].values())


def test_the_genome_check_finds_a_dropped_row_without_an_n(tmp_path):
    data = make_dataset(tmp_path, chrombpnet=False)
    fasta = Path(data["folds"]["0"]["params"]["sequences"])
    fasta.write_text(fasta.read_text().replace("N", "A"))
    Path(f"{fasta}.fai").unlink(missing_ok=True)
    report = _check(data, against="fasta", n_sample=4)
    assert not report["pass"] and report["v1"]["0"]["dropped_all_acgt"] == [5, 17]


def test_a_fold_not_converted_fails(tmp_path):
    data = make_dataset(tmp_path)
    prov = data["out"] / "full_models" / "ds_all_fold_4" / "interpretation" / "interpretation.provenance.json"
    prov.unlink()
    report = _check(data, against="chrombpnet", n_sample=5)
    assert not report["pass"] and "fold 4: not converted" in report["conversions"]["failures"][0]


def test_a_conversion_older_than_its_attribution_fails(tmp_path):
    data = make_dataset(tmp_path)
    attr = Path(data["folds"]["1"]["params"]["attr_filename"])
    later = time.time() + 5
    os.utime(attr, (later, later))
    report = _check(data, against="chrombpnet", n_sample=5)
    assert not report["pass"]
    assert "fold 1: the conversion is not current: sources changed since the conversion: attr" in _failures(report)


def test_beds_that_differ_between_folds_fail(tmp_path):
    data = make_dataset(tmp_path)
    bed = data["out"] / "full_models" / "ds_all_fold_3" / "interpretation" / "interpretation.interpreted_regions.bed"
    text = bed.read_bytes()
    bed.write_bytes(text.replace(b"ds_peak_1\t", b"ds_peak_X\t"))
    report = _check(data, against="chrombpnet", n_sample=5)
    failures = _failures(report)
    assert not report["pass"] and "fold 3: the conversion is not current" in failures
    assert "fold 3's bed differs from fold 0's (cmp)" in failures


def test_the_command_line_writes_the_report_and_exits_by_the_verdict(tmp_path):
    data = make_dataset(tmp_path)
    argv = ["--check-dataset", "ds", "--root", str(data["root"]), "--against", "chrombpnet", "--sample-rows", "5"]
    assert tcs.main(argv) == 0
    report = json.loads((data["out"] / "validation" / "check_dataset.json").read_text())
    assert report["pass"] and report["mode"] == "check-dataset" and report["against"] == "chrombpnet"
    _cbp_h5(data, "0").unlink()
    assert tcs.main([*argv, "--report", str(tmp_path / "r.json")]) == 1
    assert json.loads((tmp_path / "r.json").read_text())["pass"] is False
    with pytest.raises(SystemExit):
        tcs.main(["--check-dataset", "ds"])  # no --root


# ---------------------------------------------------------------------------------------------------------------------
# --check-average (V2)


def test_v2_passes_on_06_0s_average(tmp_path):
    data = make_dataset(tmp_path, chrombpnet=False)
    average_as_06_0(data)
    report = tcs.check_average(data["root"], "ds", n_sample=10, block_rows=7)
    assert report["pass"], report["failures"]
    assert report["rows_compared"] == 10 and report["raw_mismatch_rows"] == 0
    assert report["shap_rows_not_bitwise_equal"] == 0 and report["projected_shap_rows_not_bitwise_equal"] == 0
    everything = tcs.check_average(data["root"], "ds", n_sample=10_000, threads=1)
    assert everything["pass"] and everything["rows_compared"] == N_ROWS - len(N_ROWS_DROPPED)


def test_v2_finds_one_float16_step_off(tmp_path):
    data = make_dataset(tmp_path, chrombpnet=False)
    avg = average_as_06_0(data)
    with h5py.File(avg, "r+") as fh:
        shap = fh["shap/seq"][...]
        shap[4, 2, 7] = np.nextafter(shap[4, 2, 7], np.float16(np.inf))
        fh["shap/seq"][...] = shap
    report = tcs.check_average(data["root"], "ds", n_sample=10_000)
    assert not report["pass"] and report["shap_rows_not_bitwise_equal"] == 1
    assert report["projected_shap_rows_not_bitwise_equal"] == 0


def test_v2_holds_the_folds_to_the_configs_order(tmp_path):
    data = make_dataset(tmp_path, chrombpnet=False, n_rows=60, dropped={})
    average_as_06_0(data, order=tuple(reversed(FOLDS)))
    report = tcs.check_average(data["root"], "ds", n_sample=10_000)
    assert not report["pass"]  # float16 sums are not associative: another order rounds some values otherwise


def test_v2_finds_raw_rows_out_of_order_and_a_missing_row(tmp_path):
    data = make_dataset(tmp_path, chrombpnet=False)
    avg = average_as_06_0(data)
    with h5py.File(avg, "r+") as fh:
        raw = fh["raw/seq"][...]
        fh["raw/seq"][...] = raw[::-1]
    report = tcs.check_average(data["root"], "ds", n_sample=10)
    assert not report["pass"] and report["raw_mismatch_rows"] > 0
    with h5py.File(avg, "w") as fh:
        for key in ("raw", "shap", "projected_shap"):
            fh.create_dataset(f"{key}/seq", data=np.zeros((N_ROWS - 3, 4, W), tcs.DTYPES[key]))
    report = tcs.check_average(data["root"], "ds")
    assert not report["pass"] and "not ('int8', [22, 4, 12])" in report["failures"][0]


def test_v2_without_the_average_fails_and_the_command_line_reports_it(tmp_path):
    data = make_dataset(tmp_path, chrombpnet=False)
    argv = ["--check-average", "ds", "--root", str(data["root"]), "--sample-rows", "5"]
    assert tcs.main(argv) == 1
    report = json.loads((data["out"] / "validation" / "check_average.json").read_text())
    assert report["failures"][0].startswith("no ")
    average_as_06_0(data)
    assert tcs.main(argv) == 0
