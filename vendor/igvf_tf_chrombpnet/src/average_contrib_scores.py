#!/usr/bin/env python3
"""
average_contrib_scores.py
Average DeepSHAP contribution scores across the cross-validation folds for a
given dataset (06.0). Produces a single "average_shaps.{score_type}.h5" per
dataset with reduced fold-specific noise, for 07.0's bigwigs, motif discovery
in 08.0 and Fi-NeMo hit calling in 10.0 (which reads counts). Handles both the
"counts" and "profile" contribution score heads via --score-type; the two H5
types share the same key structure. Folds with no scores file are skipped
with a warning; an existing output is left alone.

Why average:
  Each fold's model was trained on a different 80% of peaks, so its
  contribution scores reflect slightly different random variation. Averaging
  cancels this noise while preserving the signal that is consistently
  important across all models.

Input per fold (05.0, `chrombpnet contribs_bw`):
  {full_model_dir}/{dataset}_{peak_type}_fold_{fold}/interpretation/
      interpretation.{score_type}_scores.h5

  Expected H5 keys (shape: n_peaks x 4 x seq_len):
      raw/seq             - one-hot encoded sequences (identical across folds)
      shap/seq            - hypothetical contribution scores
      projected_shap/seq  - projected contribution scores (used by MoDISco)

  chrombpnet 1.x (deepdish) and 2.x (h5py) both write these float16 and
  Blosc-compressed, which is why hdf5plugin is imported before reading.

Output:
  {averaged_dir}/{dataset}/{dataset}_average_shaps.{score_type}.h5
      same keys, float16, gzip-compressed (readable without hdf5plugin)

Usage:
  python average_contrib_scores.py \\
      --dataset igvf3_cardiomyocyte \\
      --folds 0 1 2 3 4 \\
      --full-model-dir results/full_models \\
      --peak-type all \\
      --score-type profile \\
      --out-dir results/contrib_scores/igvf3_cardiomyocyte
"""

import argparse
import os
import sys
from pathlib import Path

import h5py

# Make lib/python importable without an install step (works under the cluster
# conda envs, under pixi, and under a bare python).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib" / "python"))

import hdf5plugin  # noqa: F401  # registers the HDF5 filters chrombpnet's H5s use
import numpy as np

from utils import log  # noqa: E402

logger = log.get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="dataset label, e.g. igvf3_cardiomyocyte")
    p.add_argument(
        "--folds",
        required=True,
        nargs="+",
        type=str,
        help="Fold indices, e.g. 0 1 2 3 4",
    )
    p.add_argument(
        "--full-model-dir",
        required=True,
        help="Directory holding the {dataset}_{peak-type}_fold_{fold}/ model dirs (${full_model_dir})",
    )
    p.add_argument("--peak-type", default="all")
    p.add_argument(
        "--score-type",
        default="counts",
        choices=["counts", "profile"],
        help="Which contribution score head to average (default: counts)",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for average_shaps.{score-type}.h5",
    )
    log.add_logging_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    log.setup_from_args(args)
    os.makedirs(args.out_dir, exist_ok=True)

    out_path = os.path.join(args.out_dir, f"{args.dataset}_average_shaps.{args.score_type}.h5")
    if os.path.exists(out_path):
        logger.warning(f"Output already exists, skipping: {out_path}")
        return

    # Locate available fold H5 files
    fold_h5s = []
    for fold in args.folds:
        h5_path = os.path.join(
            args.full_model_dir,
            f"{args.dataset}_{args.peak_type}_fold_{fold}",
            "interpretation",
            f"interpretation.{args.score_type}_scores.h5",
        )
        if os.path.exists(h5_path):
            fold_h5s.append((fold, h5_path))
        else:
            logger.warning(f"not found, skipping fold {fold}: {h5_path}")

    if len(fold_h5s) == 0:
        sys.exit(f"ERROR: no contribution H5 files found for dataset={args.dataset}")

    logger.info(f"  Averaging {len(fold_h5s)} folds: {[f for f, _ in fold_h5s]}")

    # Accumulate sums
    avg_projected = None
    avg_shap = None
    raw = None
    n = 0

    for fold, h5_path in fold_h5s:
        logger.info(f"  Loading fold {fold}: {h5_path}")
        with h5py.File(h5_path, "r") as fh:
            avg_projected_shap = np.array(fh["projected_shap"]["seq"])
            shap = np.array(fh["shap"]["seq"])

            if raw is None:
                raw = np.array(fh["raw"]["seq"])
                avg_projected = avg_projected_shap.copy()
                avg_shap = shap.copy()
            else:
                assert raw.shape == avg_projected_shap.shape, (
                    f"Shape mismatch at fold {fold}: {avg_projected_shap.shape} vs {raw.shape}"
                )
                avg_projected += avg_projected_shap
                avg_shap += shap
        n += 1

    avg_projected /= n
    avg_shap /= n

    logger.info(f"Saving averaged SHAP scores to {out_path}")
    with h5py.File(out_path, "w") as fh:
        fh.create_dataset("raw/seq", data=raw, compression="gzip")
        fh.create_dataset("shap/seq", data=avg_shap, compression="gzip")
        fh.create_dataset("projected_shap/seq", data=avg_projected, compression="gzip")

    logger.info("Done!")


if __name__ == "__main__":
    main()
