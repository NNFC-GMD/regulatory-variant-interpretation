#!/usr/bin/env python3
"""List, and check, every model file and input a variant-scoring run reads.

For each dataset (one igvf_tf_chrombpnet config.yaml) and each of its folds
this records, with size and mtime (and md5 with --md5):

  cherimoya    `<cherimoya-root>/<dataset>/fold_<k>/cherimoya.fold_<k>.torch`, the
               checkpoint workflows/train/ trains and attribution reads
               (the `model` of that fold's attribute.json, when present).
  cbp_nobias   `<output_dir>/<models-subdir>/<dataset>_<peak_type>_fold_<k>/models/chrombpnet_nobias.h5`
  cbp_bias     `.../models/bias_model_scaled.h5`, the scaled bias model that
               chrombpnet.h5 freezes inside itself. ChromBPNet's pipeline moves it
               from auxiliary/ to models/ and rewrites `bias_model_path` in
               logs/chrombpnet_model_params.tsv to point at it
               (chrombpnet/pipelines.py), so that file is checked to name it.
               logs/chrombpnet.args.json's `bias_model_path` is the UNSCALED
               03.0 bias model; it is recorded as provenance only and must not
               be scored.
  cbp_full     `.../models/chrombpnet.h5`. Variant scoring derives the full model
               from cbp_nobias + cbp_bias; this file is listed for the one-off
               check that the derivation reproduces it.
  peaks        00.1's training peaks,
               `<output_dir>/preprocessing/peaks/<dataset>_<peak_type>_peaks_no_blacklist.narrowPeak`
  chroms       the fold's train/valid/test chromosomes from `<folds-dir>/fold_<k>.json`.

Checks (a failure is reported and makes the exit status 1):
  - every listed file exists and is not empty;
  - model_params.tsv's bias_model_path is the cbp_bias file;
  - args.json's chr_fold_path is the fold file read here;
  - the Cherimoya fit.json's test/validation/training chromosomes equal the
    fold file's, and its `loci` is the peaks file;
  - attribute.json's `model`, when present, is the cherimoya checkpoint;
  - the fold files' test sets do not overlap.

    python workflows/variant_scoring/make_manifest.py \\
        --chrombpnet-configs <configs>/<dataset>/config.yaml ... \\
        --cherimoya-root <cherimoya out-root> --folds-dir <igvf_tf_chrombpnet>/folds \\
        --genome <genome.fasta> --md5 --out <dir>/manifest.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

logger = logging.getLogger("make_manifest")

ET = ZoneInfo("America/New_York")
SPLITS = ("train", "valid", "test")
CHERIMOYA_SPLITS = {"train": "training_chroms", "valid": "validation_chroms", "test": "test_chroms"}


def now_et() -> str:
    return dt.datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S %Z")


def md5sum(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def describe(path: Path, md5: bool) -> dict:
    """Path, size and mtime (ET) of a file, plus its md5 when asked; `exists` False if absent."""
    out: dict = {"path": str(path), "exists": path.is_file()}
    if out["exists"]:
        st = path.stat()
        out["size"] = st.st_size
        out["mtime_et"] = dt.datetime.fromtimestamp(st.st_mtime, ET).strftime("%Y-%m-%d %H:%M:%S %Z")
        if md5:
            out["md5"] = md5sum(path)
    return out


def read_config(path: Path) -> dict:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    out = {}
    for key in ("dataset_name", "output_dir"):
        value = cfg.get(key)
        if not value:
            raise ValueError(f"{path}: no {key}")
        value = os.path.expandvars(str(value))
        if "$" in value:
            raise ValueError(f"{path}: {key} has an unexpanded variable: {value}")
        out[key] = value
    out["peak_type"] = str(cfg.get("peak_type", "all"))
    out["folds"] = [str(f) for f in cfg.get("folds", ["0", "1", "2", "3", "4"])]
    return out


def read_fold(folds_dir: Path, fold: str) -> dict:
    with open(folds_dir / f"fold_{fold}.json") as fh:
        split = json.load(fh)
    missing = set(SPLITS) - set(split)
    if missing:
        raise ValueError(f"fold_{fold}.json lacks {sorted(missing)}")
    return {k: list(split[k]) for k in SPLITS}


def read_params_tsv(path: Path) -> dict:
    out = {}
    with open(path) as fh:
        for line in fh:
            key, _, value = line.rstrip("\n").partition("\t")
            if key:
                out[key] = value
    return out


def peaks_path(cfg: dict) -> Path:
    """00.1's training peaks for a dataset."""
    ds, pt = cfg["dataset_name"], cfg["peak_type"]
    return Path(cfg["output_dir"]) / "preprocessing" / "peaks" / f"{ds}_{pt}_peaks_no_blacklist.narrowPeak"


def same_file(a, b) -> bool:
    return os.path.realpath(str(a)) == os.path.realpath(str(b))


def count_lines(path: Path) -> int:
    with open(path, "rb") as fh:
        return sum(1 for _ in fh)


def fold_entry(cfg, fold, chroms, args, problems) -> dict:
    ds, pt = cfg["dataset_name"], cfg["peak_type"]
    tag = f"{ds} fold {fold}"
    run = Path(cfg["output_dir"]) / args.models_subdir / f"{ds}_{pt}_fold_{fold}"
    cheri_dir = args.cherimoya_root / ds / f"fold_{fold}"
    peaks = peaks_path(cfg)
    models = {
        "cherimoya": describe(cheri_dir / f"cherimoya.fold_{fold}.torch", args.md5),
        "cbp_nobias": describe(run / "models" / "chrombpnet_nobias.h5", args.md5),
        "cbp_bias": describe(run / "models" / "bias_model_scaled.h5", args.md5),
        "cbp_full": describe(run / "models" / "chrombpnet.h5", args.md5),
    }
    models["cbp_full"]["role"] = "reference only: scoring derives it from cbp_nobias + cbp_bias"
    models["cbp_bias"]["role"] = "the scaled bias model frozen inside chrombpnet.h5"
    for name, info in models.items():
        if not info["exists"] or info.get("size", 0) == 0:
            problems.append(f"{tag}: {name} missing or empty: {info['path']}")

    checks: dict = {}
    provenance: dict = {}
    params_tsv = run / "logs" / "chrombpnet_model_params.tsv"
    args_json = run / "logs" / "chrombpnet.args.json"
    provenance["chrombpnet_model_params"] = str(params_tsv)
    provenance["chrombpnet_args"] = str(args_json)
    if params_tsv.is_file():
        bias_in_params = read_params_tsv(params_tsv).get("bias_model_path", "")
        checks["model_params_names_cbp_bias"] = same_file(bias_in_params, models["cbp_bias"]["path"])
    else:
        checks["model_params_names_cbp_bias"] = False
    if args_json.is_file():
        run_args = json.loads(args_json.read_text())
        provenance["unscaled_bias_model_not_used"] = run_args.get("bias_model_path")
        checks["args_fold_file_matches"] = same_file(
            run_args.get("chr_fold_path", ""), args.folds_dir / f"fold_{fold}.json"
        )
        provenance["genome_trained_with"] = run_args.get("genome")
    else:
        checks["args_fold_file_matches"] = False

    fit_json = cheri_dir / "fit.json"
    provenance["cherimoya_fit"] = str(fit_json)
    if fit_json.is_file():
        fit = json.loads(fit_json.read_text())
        checks["cherimoya_fit_chroms_match"] = all(
            sorted(fit.get(CHERIMOYA_SPLITS[k], [])) == sorted(chroms[k]) for k in SPLITS
        )
        checks["cherimoya_fit_loci_is_peaks"] = [os.path.realpath(p) for p in fit.get("loci", [])] == [
            os.path.realpath(peaks)
        ]
    else:
        checks["cherimoya_fit_chroms_match"] = False
        checks["cherimoya_fit_loci_is_peaks"] = False
    attribute_json = cheri_dir / "attribute.json"
    if attribute_json.is_file():
        provenance["cherimoya_attribute"] = str(attribute_json)
        model_in_attr = json.loads(attribute_json.read_text()).get("model", "")
        checks["attribute_model_is_cherimoya"] = same_file(model_in_attr, models["cherimoya"]["path"])
    for name, ok in checks.items():
        if not ok:
            problems.append(f"{tag}: check failed: {name}")

    qc = run / "evaluation" / "chrombpnet_nobias_max_bias_response.txt"
    entry = {
        "chroms": chroms,
        "fold_json": str(args.folds_dir / f"fold_{fold}.json"),
        "models": models,
        "checks": checks,
        "provenance": provenance,
    }
    if qc.is_file():
        entry["nobias_max_bias_response"] = qc.read_text().strip()
    return entry


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chrombpnet-configs", nargs="+", type=Path, required=True,
                    help="igvf_tf_chrombpnet config.yaml files, one per dataset")  # fmt: skip
    ap.add_argument("--cherimoya-root", type=Path, required=True,
                    help="workflows/train/make_jobs.py's --out-root (<dataset>/fold_<k>/ under it)")  # fmt: skip
    ap.add_argument("--folds-dir", type=Path, required=True, help="the folds/ directory ChromBPNet trained with")
    ap.add_argument("--genome", type=Path, required=True, help="the FASTA both models trained with (.fai beside it)")
    ap.add_argument("--models-subdir", default="full_models",
                    help="ChromBPNet's full-model directory under output_dir (default full_models)")  # fmt: skip
    ap.add_argument("--md5", action="store_true", help="also record md5 of the genome and every model file")
    ap.add_argument("--out", type=Path, required=True, help="manifest.json to write")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%F %T")  # fmt: skip

    problems: list[str] = []
    started = now_et()
    genome = describe(args.genome, args.md5)
    genome["fai"] = describe(Path(f"{args.genome}.fai"), False)
    if not genome["exists"] or not genome["fai"]["exists"]:
        problems.append(f"genome or its .fai missing: {args.genome}")

    fold_ids = sorted({f for c in args.chrombpnet_configs for f in read_config(c)["folds"]}, key=int)
    folds = {f: read_fold(args.folds_dir, f) for f in fold_ids}
    seen: dict[str, str] = {}
    for f, split in folds.items():
        for c in split["test"]:
            if c in seen:
                problems.append(f"{c} is a test chromosome of folds {seen[c]} and {f}")
            seen[c] = f

    datasets = {}
    for config in args.chrombpnet_configs:
        cfg = read_config(config)
        ds = cfg["dataset_name"]
        logger.info("%s: %s", ds, config)
        peaks = peaks_path(cfg)
        peaks_info = describe(peaks, args.md5)
        if peaks_info["exists"]:
            peaks_info["n_peaks"] = count_lines(peaks)
        else:
            problems.append(f"{ds}: peaks missing: {peaks}")
        datasets[ds] = {
            "config": describe(config, False),
            "output_dir": cfg["output_dir"],
            "peak_type": cfg["peak_type"],
            "peaks": peaks_info,
            "folds": {f: fold_entry(cfg, f, folds[f], args, problems) for f in cfg["folds"]},
        }

    manifest = {
        "created_et": started,
        "written_by": "workflows/variant_scoring/make_manifest.py",
        "argv": sys.argv[1:] if argv is None else list(argv),
        "genome": genome,
        "folds_dir": str(args.folds_dir),
        "test_chrom_to_fold": {c: f for f, split in folds.items() for c in split["test"]},
        "datasets": datasets,
        "problems": problems,
        "ok": not problems,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    tmp.replace(args.out)
    for p in problems:
        logger.error("%s", p)
    n_models = sum(len(f["models"]) for d in datasets.values() for f in d["folds"].values())
    logger.info("wrote %s: %d datasets, %d model files, %d problems", args.out, len(datasets), n_models, len(problems))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
