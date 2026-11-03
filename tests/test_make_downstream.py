"""make_downstream.py writes each fold's attribution JSON, each dataset's pipeline config and the compendium lists."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1] / "workflows" / d) for d in ("train", "attribution", "downstream")
]

import make_downstream as md  # noqa: E402

SPLITS = {  # fold -> (training, validation, test)
    "0": (["chr2", "chrX"], ["chr10"], ["chr1"]),
    "1": (["chr1", "chr10"], ["chrX"], ["chr2"]),
}
PEAK_CHROMS = ["chr1", "chr2", "chr10", "chrX"]


def _peaks(path: Path, chroms=PEAK_CHROMS) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{c}\t{1000 * i}\t{1000 * i + 1000}\tpeak_{i}\t0\t.\t1\t1\t1\t500\n" for i, c in enumerate(chroms))
    )
    return path


def _tree(tmp_path: Path, datasets=("ds_a", "ds_b"), shared=("ds_b",), interpret: dict | None = None) -> list[str]:
    """ChromBPNet configs and outputs, and make_jobs.py's Cherimoya folds, for `datasets` with folds 0 and 1."""
    cheri = tmp_path / "cheri"
    configs = []
    for ds in datasets:
        out = tmp_path / "cbp" / ds
        peaks = _peaks(out / "preprocessing" / "peaks" / f"{ds}_all_peaks_no_blacklist.narrowPeak")
        cfg = tmp_path / "configs" / ds / "config.yaml"
        cfg.parent.mkdir(parents=True)
        cfg.write_text(
            f'dataset_name: {ds}\nassay: ATAC\nsignal_path: "/data/{ds}.fragments.tsv.gz"\n'
            f'output_dir: "{out}"\npeak_type: all\nfolds: ["0", "1"]\nmodisco_max_seqlets: 5000\n'
            'fold_bias_suffix:\n  "0": "_07"\n'
        )
        configs.append(str(cfg))
        for fold, (train, valid, test) in SPLITS.items():
            fold_dir = cheri / ds / f"fold_{fold}"
            fold_dir.mkdir(parents=True)
            fit = {
                "name": f"cherimoya.fold_{fold}",
                "sequences": "/data/genome.fa",
                "loci": [str(peaks)],
                "training_chroms": train,
                "validation_chroms": valid,
                "test_chroms": test,
                "random_state": 1234 + int(fold),
            }
            (fold_dir / "fit.json").write_text(json.dumps(fit))
            if interpret is not None:
                args_dir = out / "full_models" / f"{ds}_all_fold_{fold}" / "interpretation"
                args_dir.mkdir(parents=True)
                record = {"num_shuffles": 20, "regions": str(peaks), **interpret}
                (args_dir / "interpretation.interpret.args.json").write_text(json.dumps(record))
    shared_txt = tmp_path / "shared.txt"
    shared_txt.write_text("".join(f"{ds}\n" for ds in shared))
    return ["--cheri-root", str(cheri), "--chrombpnet-configs", *configs, "--shared", str(shared_txt),
            "--compendia", str(tmp_path / "compendia")]  # fmt: skip


def _attribute(tmp_path: Path, ds: str, fold: str) -> dict:
    return json.loads((tmp_path / "cheri" / ds / f"fold_{fold}" / "attribute.json").read_text())


def test_attribute_json_fields_seeds_and_outputs(tmp_path):
    argv = _tree(tmp_path, interpret={})
    assert md.main(argv) == 0
    peaks = tmp_path / "cbp" / "ds_a" / "preprocessing" / "peaks" / "ds_a_all_peaks_no_blacklist.narrowPeak"
    for fold in "01":
        fold_dir = tmp_path / "cheri" / "ds_a" / f"fold_{fold}"
        p = _attribute(tmp_path, "ds_a", fold)
        assert p["model"] == str(fold_dir / f"cherimoya.fold_{fold}.torch")
        assert p["loci"] == [str(peaks)] and p["sequences"] == "/data/genome.fa"
        assert p["chroms"] == ["chr1", "chr2", "chr10", "chrX"]  # the fold's three lists, in chromosome order
        assert (p["in_window"], p["attr_window"], p["n_shuffles"], p["batch_size"]) == (2114, 1000, 20, 64)
        assert (p["algorithm"], p["output"], p["group"], p["dtype"], p["device"]) == (
            "deep_lift_shap",
            "counts",
            0,
            "float32",
            "cuda",
        )
        assert p["exclusion_lists"] is None and p["compile"] is False and p["warning_threshold"] == 0.001
        assert p["random_state"] == 1234 + 100 * int(fold)
        assert p["ohe_filename"] == str(fold_dir / "attr.ohe.npz")
        assert p["attr_filename"] == str(fold_dir / "attr.attr.npz")
        assert p["idx_filename"] == str(fold_dir / "attr.idx.npy")
        assert p["fast"] == {"precision": "reference", "mem_budget_gb": 7.5, "ref_workers": 3, "audit": 8}
        assert md.output_files(p)[3:] == [fold_dir / "attr.attr.deltas.npy", fold_dir / "attr.attr.meta.json"]
    fit = json.loads((tmp_path / "cheri" / "ds_a" / "fold_0" / "fit.json").read_text())
    assert fit["random_state"] == 1234  # fit.json is read, never written


def test_seed_options(tmp_path):
    argv = _tree(tmp_path)
    assert md.main([*argv, "--seed-base", "7", "--seed-stride", "20", "--attr-window", "400"]) == 0
    assert [_attribute(tmp_path, "ds_b", f)["random_state"] for f in "01"] == [7, 27]
    assert _attribute(tmp_path, "ds_b", "1")["attr_window"] == 400
    with pytest.raises(SystemExit):  # adjacent folds would share 19 of 20 references
        md.main([*argv, "--seed-stride", "1"])
    for window in ("999", "2116", "0"):
        with pytest.raises(SystemExit):
            md.main([*argv, "--attr-window", window])


def test_every_peak_must_lie_on_a_listed_chromosome(tmp_path):
    argv = _tree(tmp_path)
    _peaks(tmp_path / "cbp" / "ds_a" / "preprocessing" / "peaks" / "ds_a_all_peaks_no_blacklist.narrowPeak",
           [*PEAK_CHROMS, "chrUn_KI270742v1"])  # fmt: skip
    with pytest.raises(ValueError, match=r"peaks on \['chrUn_KI270742v1'\]"):
        md.main(argv)


def test_a_listed_chromosome_without_peaks_is_allowed(tmp_path, caplog):
    argv = _tree(tmp_path)
    _peaks(tmp_path / "cbp" / "ds_a" / "preprocessing" / "peaks" / "ds_a_all_peaks_no_blacklist.narrowPeak",
           ["chr1", "chr2", "chr10"])  # fmt: skip
    assert md.main(argv) == 0
    assert _attribute(tmp_path, "ds_a", "0")["chroms"] == ["chr1", "chr2", "chr10", "chrX"]
    assert "no peaks on ['chrX']" in caplog.text


def test_fit_json_problems_are_refused(tmp_path):
    argv = _tree(tmp_path)
    fit_path = tmp_path / "cheri" / "ds_a" / "fold_1" / "fit.json"
    fit = json.loads(fit_path.read_text())
    for bad, message in (
        (dict(fit, loci=fit["loci"] * 2), "exactly one loci file"),
        (dict(fit, name="cherimoya.fold_0"), "not cherimoya.fold_1"),
        (dict(fit, test_chroms=["chr1"]), "chr1 is in both"),
        (dict(fit, loci=["/elsewhere.narrowPeak"]), "differs from the dataset's other folds"),
    ):
        fit_path.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match=message):
            md.main(argv)


def test_n_shuffles_and_regions_checked_against_chrombpnet(tmp_path):
    argv = _tree(tmp_path, interpret={"num_shuffles": 10})
    with pytest.raises(ValueError, match="num_shuffles 10, not 20"):
        md.main(argv)
    argv = _tree(tmp_path / "other", interpret={"regions": "/data/other.narrowPeak"})
    with pytest.raises(ValueError, match="ChromBPNet scored /data/other.narrowPeak"):
        md.main(argv)


def test_writes_are_idempotent_and_a_used_json_is_kept(tmp_path, caplog):
    argv = _tree(tmp_path)
    assert md.main(argv) == 0
    files = sorted(p for d in ("cheri", "compendia") for p in (tmp_path / d).rglob("*") if p.is_file())
    stamps = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in files}
    caplog.set_level(logging.INFO)
    caplog.clear()
    assert md.main(argv) == 0
    assert {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in files} == stamps
    assert "attribute.json 0 written, 4 unchanged; configs 0 written, 2 unchanged" in caplog.text
    assert not list((tmp_path / "cheri").rglob(".*.tmp.*"))

    # A changed parameter rewrites the JSON, unless an attribution has already run with it
    target = tmp_path / "cheri" / "ds_a" / "fold_0" / "attribute.json"
    (tmp_path / "cheri" / "ds_a" / "fold_0" / "attr.attr.meta.json").write_text("{}")
    with pytest.raises(RuntimeError, match="an attribution ran with it"):
        md.main([*argv, "--audit", "16"])
    assert target.read_bytes() == stamps[target][1]
    assert md.main([*argv, "--audit", "16", "--replace-used"]) == 0
    assert json.loads(target.read_text())["fast"]["audit"] == 16


def test_configs_for_the_pipeline_steps(tmp_path):
    argv = _tree(tmp_path)
    assert md.main(argv) == 0
    text = (tmp_path / "cheri" / "configs" / "ds_a" / "config.yaml").read_text()
    assert text.startswith("# Generated by regulatory-variant-interpretation workflows/downstream/make_downstream.py")
    cfg = yaml.safe_load(text)
    assert cfg == {
        "dataset_name": "ds_a",
        "assay": "ATAC",
        "signal_path": "/data/ds_a.fragments.tsv.gz",
        "regions": str(
            tmp_path / "cbp" / "ds_a" / "preprocessing" / "peaks" / "ds_a_all_peaks_no_blacklist.narrowPeak"
        ),
        "output_dir": str(tmp_path / "cheri" / "ds_a" / "downstream"),
        "peak_type": "all",
        "folds": ["0", "1"],
        "contrib_heads": ["counts"],
        "modisco_max_seqlets": 5000,
    }  # no fold_bias_suffix (03.x-05.0 fail fast with it) and no full_model_* key (they rename full_models/)
    # Every value line also reads in igvf_tf_chrombpnet's stdlib parser: one `key: value` per line, quoted strings, flow
    # lists, comments after " #".
    for line in text.splitlines():
        if line and not line.startswith("#"):
            key, _, value = line.split(" #", 1)[0].partition(": ")
            assert key in cfg and value.strip()


def test_compendium_lists(tmp_path):
    argv = _tree(tmp_path, datasets=("ds_a", "ds_b", "ds_c"), shared=("ds_c", "ds_a"))
    assert md.main(argv) == 0
    inputs = tmp_path / "compendia" / "inputs"
    assert sorted(p.name for p in inputs.iterdir()) == [
        "cbp2.tsv",
        "cheri2.tsv",
        "cheri3.tsv",
        "joint4.tsv",
        "rest1.txt",
        "shared2.txt",
    ]
    assert (inputs / "shared2.txt").read_text() == "ds_c\nds_a\n"  # the shared list's order
    assert (inputs / "rest1.txt").read_text() == "ds_b\n"
    cbp = [f"cbp_{ds}\t{os.path.abspath(tmp_path / 'configs' / ds / 'config.yaml')}" for ds in ("ds_c", "ds_a")]
    cheri = [f"cheri_{ds}\t{tmp_path / 'cheri' / 'configs' / ds / 'config.yaml'}" for ds in ("ds_c", "ds_a", "ds_b")]
    assert (inputs / "cbp2.tsv").read_text().splitlines() == cbp
    assert (inputs / "cheri2.tsv").read_text().splitlines() == cheri[:2]
    assert (inputs / "cheri3.tsv").read_text().splitlines() == cheri
    assert (inputs / "joint4.tsv").read_text().splitlines() == cbp + cheri[:2]


def test_lists_refuse_unknown_and_bad_names(tmp_path):
    argv = _tree(tmp_path)
    (tmp_path / "shared.txt").write_text("ds_a\nds_z\n")
    with pytest.raises(ValueError, match=r"without a ChromBPNet config: \['ds_z'\]"):
        md.main(argv)
    (tmp_path / "shared.txt").write_text("ds_a\nds_a\n")
    with pytest.raises(ValueError, match="listed twice"):
        md.main(argv)
    with pytest.raises(ValueError, match="does not match"):
        md.compendium_rows("cbp", ["ds-a"], {"ds-a": "/x/config.yaml"})


def test_chromosome_order():
    names = ["chrX", "chr10", "chrM", "chr2", "chrY", "chr1", "chr22"]
    assert sorted(names, key=md.chrom_key) == ["chr1", "chr2", "chr10", "chr22", "chrM", "chrX", "chrY"]


def test_engine_cherimoya_and_an_empty_shared_list(tmp_path):
    """--engine cherimoya: `cherimoya attribute` (>= 0.3.1) refuses anything but a bool in "fast"; the rest of the
    JSON is the rvi engine's. A project without ChromBPNet passes an empty --shared and gets cheri<D>.tsv."""
    argv = _tree(tmp_path, shared=())
    assert md.main([*argv, "--engine", "cherimoya"]) == 0
    rvi = tmp_path / "rvi"
    assert md.main(_tree(rvi, shared=())) == 0
    for fold in "01":
        p, q = _attribute(tmp_path, "ds_a", fold), _attribute(rvi, "ds_a", fold)
        assert p["fast"] is True and isinstance(q["fast"], dict)
        assert p["output"] == "counts"  # the converted scores' names and the downstream configs are keyed on it
        same = {k: v for k, v in p.items() if k != "fast"}
        other = {k: str(v).replace(str(rvi), str(tmp_path)) for k, v in q.items() if k != "fast"}
        assert {k: str(v) for k, v in same.items()} == other
    inputs = tmp_path / "compendia" / "inputs"
    assert (inputs / "shared0.txt").read_text() == ""
    rows = (inputs / "cheri2.tsv").read_text().splitlines()
    assert [r.split("\t")[0] for r in rows] == ["cheri_ds_a", "cheri_ds_b"]
