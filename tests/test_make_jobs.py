"""make_jobs.py gives Cherimoya exactly what ChromBPNet trained on, fold by fold."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflows" / "cherimoya"))

import make_jobs as mj  # noqa: E402

SPLIT = {"train": ["chr2", "chr4"], "valid": ["chr8"], "test": ["chr1", "chr3"]}


def _tree(tmp_path: Path, with_negatives=True, with_model=False) -> list[str]:
    out = tmp_path / "results" / "ds"
    (out / "preprocessing" / "signal").mkdir(parents=True)
    (out / "preprocessing" / "signal" / "data_unstranded.bw").touch()
    (out / "preprocessing" / "peaks").mkdir()
    (out / "preprocessing" / "peaks" / "ds_all_peaks_no_blacklist.narrowPeak").touch()
    (out / "preprocessing" / "ds").mkdir()
    if with_negatives:
        for f in "01":
            (out / "preprocessing" / "ds" / f"output_all_fold_{f}_negatives.bed").touch()
    if with_model:
        ev = out / "full_models" / "ds_all_fold_0" / "evaluation"
        ev.mkdir(parents=True)
        (ev / "chrombpnet_predictions.h5").touch()
    cfg = tmp_path / "configs" / "ds" / "config.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(f'dataset_name: ds\noutput_dir: "{out}"\npeak_type: all\nfolds: ["0", "1"]\n')
    folds = tmp_path / "folds"
    folds.mkdir()
    for f in "01":
        (folds / f"fold_{f}.json").write_text(json.dumps(SPLIT))
    genome = tmp_path / "genome.fa"
    genome.touch()
    return ["--chrombpnet-configs", str(cfg), "--folds-dir", str(folds), "--genome", str(genome),
            "--out-root", str(tmp_path / "cheri")]  # fmt: skip


def test_fit_json_matches_chrombpnet_inputs(tmp_path):
    assert mj.main(_tree(tmp_path)) == 0
    fit = json.loads((tmp_path / "cheri" / "ds" / "fold_1" / "fit.json").read_text())
    pre = tmp_path / "results" / "ds" / "preprocessing"
    assert fit["name"] == "cherimoya.fold_1"
    assert fit["signals"] == [str(pre / "signal" / "data_unstranded.bw")]
    assert fit["loci"] == [str(pre / "peaks" / "ds_all_peaks_no_blacklist.narrowPeak")]
    assert fit["negatives"] == [str(pre / "ds" / "output_all_fold_1_negatives.bed")]
    assert (fit["training_chroms"], fit["validation_chroms"], fit["test_chroms"]) == (
        SPLIT["train"],
        SPLIT["valid"],
        SPLIT["test"],
    )
    assert fit["random_state"] == 1235
    assert fit["controls"] is None and fit["exclusion_lists"] is None
    assert "dtype" not in fit  # Cherimoya's default unless --set


def test_overrides_and_forbidden_overrides(tmp_path):
    args = _tree(tmp_path)
    assert mj.main(args + ["--set", "dtype=bfloat16", "--set", "max_epochs=5"]) == 0
    fit = json.loads((tmp_path / "cheri" / "ds" / "fold_0" / "fit.json").read_text())
    assert fit["dtype"] == "bfloat16" and fit["max_epochs"] == 5
    with pytest.raises(ValueError, match="test_chroms"):
        mj.main(args + ["--set", 'test_chroms=["chr2"]'])


def test_tasks_tsv_marks_readiness(tmp_path):
    assert mj.main(_tree(tmp_path, with_model=True)) == 0
    lines = (tmp_path / "cheri" / "tasks.tsv").read_text().splitlines()
    assert lines[0].split("\t") == ["dataset", "fold", "workdir", "inputs_ready", "chrombpnet_ready"]
    rows = {r.split("\t")[1]: r.split("\t") for r in lines[1:]}
    assert rows["0"][3:] == ["yes", "yes"] and rows["1"][3:] == ["yes", "no"]
    compare = json.loads((tmp_path / "cheri" / "ds" / "fold_0" / "compare.json").read_text())
    assert compare["model"].endswith("ds/fold_0/cherimoya.fold_0.torch")


def test_missing_negatives_are_reported_not_fatal(tmp_path):
    assert mj.main(_tree(tmp_path, with_negatives=False)) == 0
    lines = (tmp_path / "cheri" / "tasks.tsv").read_text().splitlines()[1:]
    assert all(line.split("\t")[3] == "no" for line in lines)


def test_fold_split_overlap_is_refused(tmp_path):
    (tmp_path / "fold_0.json").write_text(json.dumps({**SPLIT, "valid": ["chr2"]}))
    with pytest.raises(ValueError, match="chr2"):
        mj.read_fold(tmp_path, "0")
