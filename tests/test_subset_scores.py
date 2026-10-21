"""subset_scores.py (CPU only, synthetic): a fold's attribution on ChromBPNet's QC subsample of the same fold.

The datasets come from test_check_dataset.make_dataset (5 DONE folds on a synthetic genome, two of whose 24 peaks hold
an N, and a fake ChromBPNet tree). write_subsample adds what ChromBPNet's 04.4 leaves in a fold: the subsample (some
of the peaks, shuffled, with the non-coordinate columns rewritten as pandas writes them) and, unless told not to, its
QC scores (the one-hot of each peak's full window as raw/seq) and their bed.
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
WORKFLOWS = TESTS.parent / "workflows"
sys.path[:0] = [str(WORKFLOWS / d) for d in ("train", "attribution", "downstream")]
sys.path.insert(0, str(TESTS))

import attr_status as st  # noqa: E402
import subset_scores as ss  # noqa: E402
from test_check_dataset import FOLDS, N_ROWS_DROPPED, L, _onehot, _window, make_dataset, write_attribution  # noqa: E402

KEYS = ("raw", "shap", "projected_shap")
SUBSAMPLE_ROWS = [20, 5, 3, 17, 0, 11, 8, 23, 2, 14, 9, 1, 16, 6, 21]
"""Peaks-file rows of the subsample, in its order; 5 and 17 are the windows with an N."""


def write_subsample(data: dict, k: str, rows: list[int] | None = None, *, scores: bool = True) -> Path:
    """ChromBPNet's 04.4 files of fold k: the subsample of these peaks-file rows, in this order (rows rotated by the
    fold by default), with signalValue and pValue rewritten (1.5 -> 1.50, -1 -> -1.0); its QC scores and their bed."""
    if rows is None:
        rows = SUBSAMPLE_ROWS[int(k) :] + SUBSAMPLE_ROWS[: int(k)]
    aux = data["chrombpnet"] / f"{data['ds']}_all_fold_{k}" / "auxiliary"
    aux.mkdir(parents=True, exist_ok=True)
    peak_lines = data["peaks"].read_text().splitlines()
    out = []
    for i in rows:
        fields = peak_lines[i].split("\t")
        fields[6] = f"{float(fields[6]):.2f}"
        fields[7] = f"{float(fields[7]):.1f}"
        out.append("\t".join(fields) + "\n")
    bed = aux / "30K_subsample_peaks.bed"
    bed.write_text("".join(out))
    if scores:
        interp = aux / "interpret_subsample"
        interp.mkdir(exist_ok=True)
        shutil.copy(bed, interp / "chrombpnet_nobias.interpreted_regions.bed")
        raw = np.stack([_onehot(_window(data["genome"], data["rows"][i], L)) for i in rows])
        with h5py.File(interp / "chrombpnet_nobias.counts_scores.h5", "w") as fh:
            fh.create_dataset("raw/seq", data=raw, chunks=(min(4, len(rows)), 4, L))
    return bed


def subset_dir(data: dict, k: str) -> Path:
    return data["out"] / "full_models" / f"{data['ds']}_all_fold_{k}" / "auxiliary" / "interpret_subsample"


def run(data: dict, k: str, *extra: str) -> int:
    return ss.main(["--attribute-json", str(data["folds"][k]["json"]), "--block-rows", "5", *extra])


def converted_rows(data: dict, k: str, names: list[str]) -> dict:
    """The converted fold's rows of these peaks (to_chrombpnet_scores.py's h5, joined by name through its bed)."""
    interp = data["out"] / "full_models" / f"{data['ds']}_all_fold_{k}" / "interpretation"
    row = {line.split("\t")[3]: r for r, line in enumerate(interp.joinpath("interpretation.interpreted_regions.bed")
                                                           .read_text().splitlines())}  # fmt: skip
    with h5py.File(interp / "interpretation.counts_scores.h5", "r") as fh:
        return {key: fh[f"{key}/seq"][...][[row[n] for n in names]] for key in KEYS}


# ---------------------------------------------------------------------------------------------------------------------
# the subset


def test_the_subset_is_the_subsample_in_its_order_bit_for_bit_the_converted_rows(tmp_path):
    data = make_dataset(tmp_path)
    bed = write_subsample(data, "2")
    assert run(data, "2") == 0
    out = subset_dir(data, "2")
    rows = SUBSAMPLE_ROWS[2:] + SUBSAMPLE_ROWS[:2]
    lines = bed.read_text().splitlines(keepends=True)
    kept = [line for line, i in zip(lines, rows) if data["idx"][i]]
    assert (out / "cherimoya.interpreted_regions.bed").read_text() == "".join(kept)  # the subsample's own lines
    names = [line.split("\t")[3] for line in kept]
    expected = converted_rows(data, "2", names)
    with h5py.File(out / "cherimoya.counts_scores.h5", "r") as fh:
        assert sorted(fh) == sorted(KEYS)
        for key in KEYS:
            ds = fh[f"{key}/seq"]
            assert ds.dtype == expected[key].dtype and ds.shape == (len(kept), 4, data["ohe"].shape[2])
            assert ds.compression == "gzip" and ds.compression_opts == ss.GZIP_LEVEL and ds.chunks[0] == 13
            assert np.array_equal(ds[...].view(np.uint8), expected[key].view(np.uint8)), key
    prov = json.loads((out / "cherimoya.provenance.json").read_text())
    assert prov["rows"]["subsample_lines"] == len(rows) and prov["rows"]["kept"] == len(kept)
    assert [d["name"] for d in prov["rows"]["dropped"]] == [f"ds_peak_{i + 1}" for i in rows if i in N_ROWS_DROPPED]
    check = prov["chrombpnet_check"]
    assert check["pass"] and check["rows_compared"] == len(kept) and check["dropped_with_zero_column"] == 2
    assert set(prov["sources"]) == {"attribute_json", "peaks", "ohe", "attr", "idx", "record", "subsample",
                                    "chrombpnet_h5", "chrombpnet_bed"}  # fmt: skip
    assert prov["float16"]["shap"]["above_float16_max"] == 0


def test_a_rerun_with_nothing_changed_writes_nothing_and_a_changed_source_writes_again(tmp_path):
    data = make_dataset(tmp_path, convert=False)
    bed = write_subsample(data, "0")
    assert run(data, "0") == 0
    out = subset_dir(data, "0")
    before = {p.name: p.stat().st_mtime_ns for p in out.iterdir()}
    assert run(data, "0") == 0
    assert {p.name: p.stat().st_mtime_ns for p in out.iterdir()} == before
    later = time.time() + 5
    os.utime(bed, (later, later))
    assert run(data, "0") == 0
    after = {p.name: p.stat().st_mtime_ns for p in out.iterdir()}
    assert after.keys() == before.keys() and all(after[n] != before[n] for n in before)
    assert run(data, "0", "--force") == 0
    assert (out / "cherimoya.counts_scores.h5").stat().st_mtime_ns != after["cherimoya.counts_scores.h5"]


def test_a_fold_not_done_exits_3_and_writes_nothing(tmp_path):
    data = make_dataset(tmp_path, convert=False, done_folds=FOLDS[:4])
    write_subsample(data, "4")
    assert run(data, "4") == ss.NOT_READY_EXIT
    assert not subset_dir(data, "4").exists()


def test_without_chrombpnet_qc_scores_the_check_is_skipped(tmp_path):
    data = make_dataset(tmp_path, convert=False)
    write_subsample(data, "1", scores=False)
    assert run(data, "1") == 0
    prov = json.loads((subset_dir(data, "1") / "cherimoya.provenance.json").read_text())
    assert "skipped" in prov["chrombpnet_check"] and "chrombpnet_h5" not in prov["sources"]


def test_another_subsample_and_directory_can_be_named(tmp_path):
    data = make_dataset(tmp_path, convert=False)
    bed = write_subsample(data, "3", rows=[4, 2, 17])
    moved = tmp_path / "elsewhere" / "sub.bed"
    moved.parent.mkdir()
    shutil.copy(bed, moved)
    out = tmp_path / "out"
    assert run(data, "3", "--subsample", str(moved), "--out-dir", str(out)) == 0
    assert (out / "cherimoya.interpreted_regions.bed").read_text().count("\n") == 2
    prov = json.loads((out / "cherimoya.provenance.json").read_text())
    assert prov["sources"]["subsample"]["path"] == str(moved) and "skipped" in prov["chrombpnet_check"]


# ---------------------------------------------------------------------------------------------------------------------
# refused


def _edit_line(path: Path, i: int, edit) -> None:
    lines = path.read_text().splitlines(keepends=True)
    lines[i] = edit(lines[i])
    path.write_text("".join(lines))


@pytest.mark.parametrize(
    "case, edit, message",
    [
        ("unknown name", lambda s: s.replace("ds_peak_", "other_peak_"), "is not in"),
        ("coordinates", lambda s: s.replace(s.split("\t")[1], str(int(s.split("\t")[1]) + 1), 1), "chrom, start"),
        ("repeated name", None, "repeats"),
    ],
)
def test_a_subsample_that_does_not_join_is_refused(tmp_path, caplog, case, edit, message):
    data = make_dataset(tmp_path, convert=False)
    bed = write_subsample(data, "0", scores=False)
    if edit is None:
        bed.write_text(bed.read_text() + bed.read_text().splitlines(keepends=True)[2])
    else:
        _edit_line(bed, 3, edit)
    assert run(data, "0") == 1
    assert message in caplog.text, case
    assert not subset_dir(data, "0").exists() or not any(subset_dir(data, "0").iterdir())


def test_a_peak_whose_window_centres_differ_is_refused(tmp_path, caplog):
    data = make_dataset(tmp_path, convert=False)
    for path in (data["peaks"], write_subsample(data, "0", rows=[1, 2, 3], scores=False)):
        _edit_line(path, 2 if path == data["peaks"] else 1, lambda s: s.rstrip("\n").rsplit("\t", 1)[0] + "\t499\n")
    assert st.check_fold(data["folds"]["0"]["dir"])[0] == "done"
    assert run(data, "0") == 1
    assert "window centre" in caplog.text


def test_scores_that_disagree_with_chrombpnet_are_refused(tmp_path, caplog):
    data = make_dataset(tmp_path, convert=False)
    write_subsample(data, "0")
    h5 = (
        data["chrombpnet"]
        / "ds_all_fold_0"
        / "auxiliary"
        / "interpret_subsample"
        / "chrombpnet_nobias.counts_scores.h5"
    )
    with h5py.File(h5, "r+") as fh:
        fh["raw/seq"][...] = np.roll(fh["raw/seq"][...], 1, axis=2)
    assert run(data, "0") == 1
    assert "against ChromBPNet's QC scores" in caplog.text
    assert not (subset_dir(data, "0") / "cherimoya.provenance.json").exists()


def test_no_subsample_is_refused(tmp_path, caplog):
    data = make_dataset(tmp_path, convert=False)
    assert run(data, "0") == 1
    assert "no subsample" in caplog.text


def test_a_fold_attributed_again_is_taken_again(tmp_path):
    data = make_dataset(tmp_path, convert=False)
    write_subsample(data, "0")
    assert run(data, "0") == 0
    fold = data["folds"]["0"]
    attr = (fold["attr"] * 2).astype(np.float32)
    write_attribution(fold, data["ohe"], attr, data["idx"], stamp=time.time() - 5)
    assert run(data, "0") == 0
    with h5py.File(subset_dir(data, "0") / "cherimoya.counts_scores.h5", "r") as fh:
        rows = [i for i in SUBSAMPLE_ROWS if data["idx"][i]]
        rank = np.cumsum(data["idx"]) - 1
        assert np.array_equal(fh["shap/seq"][...], attr[rank[rows]].astype(np.float16))
