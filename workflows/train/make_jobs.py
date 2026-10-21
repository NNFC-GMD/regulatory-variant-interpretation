#!/usr/bin/env python3
"""Write Cherimoya training jobs for datasets the ChromBPNet pipeline has prepared.

Cherimoya (jmschrei/cherimoya) is trained here on exactly what ChromBPNet
trains on, so the two can be compared fold by fold:

  signal     00.0's bigwig, `<output_dir>/preprocessing/signal/data_unstranded.bw`.
             It already carries the +4/-4 Tn5 correction, so no shift is applied
             again (Cherimoya only shifts when it converts reads itself).
  peaks      00.1's summit windows, `<dataset>_<peak_type>_peaks_no_blacklist.narrowPeak`
             (1000 bp, summit at the midpoint, so `summits` stays false).
  negatives  01.0's GC-matched non-peaks for the fold,
             `output_<peak_type>_fold_<k>_negatives.bed` (2114 bp, centred).
  splits     the same `folds/fold_<k>.json` ChromBPNet trained with: `train`,
             `valid` and `test` become `training_chroms`, `validation_chroms`
             and `test_chroms`. Fold 0 equals Cherimoya's own default split.

Both peak and negative files are already blacklist-filtered by the ChromBPNet
steps, so `exclusion_lists` stays null unless --exclusion-list is given.
Everything else is Cherimoya's default (cherimoya_cli/defaults.py), except
`random_state`, which is --seed-base + fold: the default 0 would start all
five folds from the same initialisation. --set overrides any fit parameter.

For every dataset and fold this writes, under --out-root:

  <dataset>/fold_<k>/fit.json      `cherimoya fit -p fit.json`, run from that
                                   directory: the model, logs and the
                                   validation/test performance land beside it
                                   as cherimoya.fold_<k>.*
  <dataset>/fold_<k>/compare.json  compare_chrombpnet.py's inputs: the
                                   Cherimoya model, ChromBPNet's test-set
                                   predictions for the same fold, the bigwig
                                   and the genome
  tasks.tsv                        one row per (dataset, fold), with whether
                                   the inputs exist; run_box.sh works through it

Inputs missing at writing time are reported and the row is marked
`inputs_ready=no`; rerun once the ChromBPNet steps have made them.

    python workflows/train/make_jobs.py \\
        --chrombpnet-configs <configs>/*/config.yaml \\
        --folds-dir <igvf_tf_chrombpnet>/folds \\
        --genome <genome.fasta> --out-root <dir>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

logger = logging.getLogger("make_jobs")

FOLD_KEYS = {"train": "training_chroms", "valid": "validation_chroms", "test": "test_chroms"}


def read_config(path: Path) -> dict:
    """The fields of an igvf_tf_chrombpnet config.yaml this needs, with
    environment variables expanded (the pipeline's configs may use them)."""
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
    """A ChromBPNet fold file as Cherimoya's three chromosome lists."""
    with open(folds_dir / f"fold_{fold}.json") as fh:
        split = json.load(fh)
    missing = set(FOLD_KEYS) - set(split)
    if missing:
        raise ValueError(f"fold_{fold}.json lacks {sorted(missing)}")
    chroms = {FOLD_KEYS[k]: list(split[k]) for k in FOLD_KEYS}
    seen: dict[str, str] = {}
    for key, values in chroms.items():
        for c in values:
            if c in seen:
                raise ValueError(f"fold_{fold}.json puts {c} in both {seen[c]} and {key}")
            seen[c] = key
    return chroms


def dataset_paths(cfg: dict, fold: str, models_subdir: str) -> dict:
    out, ds, pt = Path(cfg["output_dir"]), cfg["dataset_name"], cfg["peak_type"]
    prefix = f"{ds}_{pt}_fold_{fold}"
    evaluation = out / models_subdir / prefix / "evaluation"
    return {
        "signal": out / "preprocessing" / "signal" / "data_unstranded.bw",
        "peaks": out / "preprocessing" / "peaks" / f"{ds}_{pt}_peaks_no_blacklist.narrowPeak",
        "negatives": out / "preprocessing" / ds / f"output_{pt}_fold_{fold}_negatives.bed",
        "chrombpnet_predictions": evaluation / "chrombpnet_predictions.h5",
        "chrombpnet_metrics": evaluation / "chrombpnet_metrics.json",
    }


def parse_set(items: list[str]) -> dict:
    """--set KEY=VALUE pairs; VALUE is read as JSON when it parses, else kept as a string."""
    out = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"--set wants KEY=VALUE, got {item!r}")
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def fit_parameters(cfg, fold, paths, chroms, genome, exclusion_list, seed_base, overrides):
    params = {
        "name": f"cherimoya.fold_{fold}",
        "sequences": str(genome),
        "loci": [str(paths["peaks"])],
        "negatives": [str(paths["negatives"])],
        "signals": [str(paths["signal"])],
        "controls": None,
        "exclusion_lists": [str(exclusion_list)] if exclusion_list else None,
        **chroms,
        "random_state": seed_base + int(fold),
        "verbose": True,
    }
    for key in overrides:
        if key in ("name", "loci", "negatives", "signals", "sequences", *FOLD_KEYS.values()):
            raise ValueError(f"--set {key} would change what is being compared; not allowed")
    params.update(overrides)
    return params


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chrombpnet-configs", nargs="+", type=Path, required=True,
                    help="igvf_tf_chrombpnet config.yaml files, one per dataset")  # fmt: skip
    ap.add_argument("--folds-dir", type=Path, required=True,
                    help="the folds/ directory ChromBPNet trained with (fold_<k>.json)")  # fmt: skip
    ap.add_argument("--genome", type=Path, required=True, help="the FASTA ChromBPNet trained with")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--models-subdir", default="full_models",
                    help="ChromBPNet's full-model directory under output_dir (default full_models)")  # fmt: skip
    ap.add_argument("--exclusion-list", type=Path, default=None,
                    help="a BED of regions Cherimoya should also drop (default: none; inputs are filtered)")  # fmt: skip
    ap.add_argument("--seed-base", type=int, default=1234,
                    help="fold k trains with random_state seed-base + k (default 1234)")  # fmt: skip
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                    help="override a Cherimoya fit parameter, e.g. --set dtype=bfloat16")  # fmt: skip
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%F %T")  # fmt: skip

    overrides = parse_set(args.overrides)
    if not args.genome.exists():
        logger.error("genome not found: %s", args.genome)
        return 1
    rows, n_missing = [], 0
    for config in sorted(args.chrombpnet_configs):
        cfg = read_config(config)
        for fold in cfg["folds"]:
            chroms = read_fold(args.folds_dir, fold)
            paths = dataset_paths(cfg, fold, args.models_subdir)
            workdir = args.out_root / cfg["dataset_name"] / f"fold_{fold}"
            workdir.mkdir(parents=True, exist_ok=True)
            params = fit_parameters(cfg, fold, paths, chroms, args.genome, args.exclusion_list,
                                    args.seed_base, overrides)  # fmt: skip
            (workdir / "fit.json").write_text(json.dumps(params, indent=4, sort_keys=True) + "\n")
            compare = {
                "dataset": cfg["dataset_name"],
                "fold": fold,
                "model": str(workdir / f"{params['name']}.torch"),
                "sequences": str(args.genome),
                "signal": str(paths["signal"]),
                "chrombpnet_predictions": str(paths["chrombpnet_predictions"]),
                "chrombpnet_metrics": str(paths["chrombpnet_metrics"]),
                "output": str(workdir / "compare.tsv"),
            }
            (workdir / "compare.json").write_text(json.dumps(compare, indent=4) + "\n")
            missing = [k for k in ("signal", "peaks", "negatives") if not paths[k].exists()]
            for k in missing:
                logger.warning("%s fold %s: no %s at %s", cfg["dataset_name"], fold, k, paths[k])
            n_missing += bool(missing)
            rows.append((cfg["dataset_name"], fold, str(workdir), "no" if missing else "yes",
                         "yes" if paths["chrombpnet_predictions"].exists() else "no"))  # fmt: skip

    args.out_root.mkdir(parents=True, exist_ok=True)
    with open(args.out_root / "tasks.tsv", "w") as fh:
        fh.write("dataset\tfold\tworkdir\tinputs_ready\tchrombpnet_ready\n")
        for row in rows:
            fh.write("\t".join(row) + "\n")
    logger.info("%d task(s) in %s; %d lack a training input", len(rows), args.out_root / "tasks.tsv", n_missing)
    return 0


if __name__ == "__main__":
    sys.exit(main())
