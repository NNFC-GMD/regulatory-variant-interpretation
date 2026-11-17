#!/usr/bin/env python3
"""
contribs_to_bigwig.py
Convert one head's fold-averaged contribution score H5, produced by
average_contrib_scores.py (06.0), into a bigwig file (07.0, once per head).

The averaged H5 stores projected_shap/seq with shape (N, 4, seqlen).
Summing over the 4-base axis yields the per-position contribution score
for the actual nucleotide at that position (identical to what
chrombpnet's importance_hdf5_to_bigwig.py does on a per-fold H5). The
region and bigwig handling is chrombpnet's own bigwig_helper, so this runs
in the chrombpnet environment. An existing output is left alone.

Input:
  {averaged_dir}/{dataset}/{dataset}_average_shaps.{counts,profile}.h5
  interpretation.interpreted_regions.bed  (05.0, from any single fold;
      rows must match the averaged H5 - all folds use the same peaks)

Output:
  {averaged_dir}/{dataset}/{dataset}_average_shaps.{counts,profile}.bw

Usage:
  python contribs_to_bigwig.py \\
      --h5          results/contrib_scores/igvf3_cardiomyocyte/igvf3_cardiomyocyte_average_shaps.counts.h5 \\
      --regions     results/full_models/igvf3_cardiomyocyte_all_fold_0/interpretation/interpretation.interpreted_regions.bed \\
      --chrom-sizes hg38.chrom.sizes \\
      --output-bw   results/contrib_scores/igvf3_cardiomyocyte/igvf3_cardiomyocyte_average_shaps.counts.bw
"""

import argparse
import os
import sys
from pathlib import Path

import chrombpnet.evaluation.make_bigwigs.bigwig_helper as bigwig_helper
import h5py

# Make lib/python importable without an install step (works under the cluster
# conda envs, under pixi, and under a bare python).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib" / "python"))

import numpy as np

from utils import log  # noqa: E402

logger = log.get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", required=True, help="Path to average_shaps.{counts,profile}.h5 (06.0)")
    p.add_argument(
        "--regions",
        required=True,
        help="10-column BED file of interpreted regions "
        "(interpretation.interpreted_regions.bed from any fold)",
    )
    p.add_argument("--chrom-sizes", required=True, help="Chromosome sizes 2-column TSV")
    p.add_argument("--output-bw", required=True, help="Output bigwig path")
    p.add_argument(
        "--debug-chr",
        type=str,
        default=None,
        help="Restrict to one chromosome for debugging (e.g. chr1)",
    )
    log.add_logging_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    log.setup_from_args(args)

    if os.path.exists(args.output_bw):
        logger.info(f"Output already exists, skipping: {args.output_bw}")
        return

    logger.info(f"Loading projected SHAP scores from: {args.h5}")
    with h5py.File(args.h5, "r") as fh:
        projected = np.array(fh["projected_shap"]["seq"])  # (N, 4, seqlen)

    seqlen = projected.shape[2]
    assert seqlen % 2 == 0, f"seqlen must be even, got {seqlen}"
    logger.info(f"  Shape: {projected.shape}  (N={projected.shape[0]}, seqlen={seqlen})")

    # Sum over 4 bases → per-position contribution score  (N, seqlen)
    scores = projected.sum(axis=1).astype(np.float32)

    gs = bigwig_helper.read_chrom_sizes(args.chrom_sizes)
    regions = bigwig_helper.get_regions(args.regions, seqlen)

    assert scores.shape[0] == len(regions), (
        f"Row count mismatch: H5 has {scores.shape[0]} peaks but "
        f"regions BED has {len(regions)} rows."
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_bw)), exist_ok=True)
    bigwig_helper.write_bigwig(scores, regions, gs, args.output_bw, debug_chr=args.debug_chr)
    logger.info(f"Wrote bigwig: {args.output_bw}")


if __name__ == "__main__":
    main()
