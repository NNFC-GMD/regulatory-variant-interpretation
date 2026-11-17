#!/usr/bin/env python3
"""Detect the Tn5 shift already present in every fragments file, once, genome-wide.

The configs take the shift PRESENT in the data (plus_shift/minus_shift); 00.0
and 00.0.call_peaks convert it to chrombpnet's +4/-4. This runs the pipeline's
own detector (utils.shift, vendored from scPrinter: base composition at the
fragment ends against reference Tn5 bias matrices) once per file -- and, with
--chrombpnet-python, chrombpnet's own (auto_shift_detect.compute_shift, the
check `chrombpnet pipeline` runs) on the same sample -- so the answer can be
pinned in every config rather than re-detected by each step. The two must
agree, or the file is reported.

Why not ``shift.detect_shift_raw`` as is: for a file over 2 GB it samples only
the first million lines, which in a coordinate-sorted fragments file is the
start of chr1 -- telomeric repeat and few distinct loci. On these files that
gave (0, 0) with "may not be accurate" for 38 of 39. Here every file is thinned
across the WHOLE genome (``bgzip -dc | mawk 'NR % every == 0'``), then a seeded
sample of the kept fragments goes to the same detector. Each file is one full
decompression pass, so run it on a compute node (workflows/dcai/run_box.sh
does, as its first phase), from the ``peaks`` environment (bgzip, mawk, pyfaidx):

    python workflows/dcai/detect_shifts.py --fragments-dir <dir> \\
        --genome "$genome_fa" --out shifts.json --jobs 39

Output: {"<file stem>": {"plus_shift": 4, "minus_shift": -5, "mse_plus": ...,
"mse_minus": ..., "chrombpnet": [4, -5], "n_fragments": ...}, ...}. A fit
whose minimum MSE is above 0.002 (scPrinter's own threshold), or a
disagreement with chrombpnet, is listed at the end and makes the exit status
2, so nothing pins a shift nobody believes.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib" / "python"))

from utils import log, shift  # noqa: E402

logger = log.get_logger(__name__)

SUFFIX = ".fragments.tsv.gz"
#: scPrinter's own threshold for a trustworthy fit (shift.circular_detect).
MAX_MSE = 0.002


def _fit(ref, query):
    """shift.circular_detect, also returning the minimum MSE instead of printing it."""
    radius = ref.shape[0] // 2
    mses = np.array(
        [shift.mse(shift.circular_shift(ref, s), query) for s in range(-radius, radius)]
    )
    return int(np.argmin(mses) - radius), float(mses.min())


#: chrombpnet's detector, run with the chrombpnet environment's python.
_CHROMBPNET_SHIFT = """
import sys, warnings
warnings.simplefilter("ignore")
from chrombpnet.helpers.preprocessing import auto_shift_detect as a
from chrombpnet.data import get_default_data_path, DefaultDataFile
ps, ms = a.compute_shift(None, sys.argv[1], None, int(sys.argv[3]), sys.argv[2], "ATAC",
                         get_default_data_path(DefaultDataFile.atac_ref_motifs), 1234)
print(ps, ms)
"""


def _detect(job):
    path, genome, every, n_sample, seed, tmp_dir, chrombpnet_python = job
    with tempfile.NamedTemporaryFile("w+", suffix=".tsv", dir=tmp_dir) as thin:
        bgzip = subprocess.Popen(["bgzip", "-dc", "-@", "2", str(path)], stdout=subprocess.PIPE)
        subprocess.run(
            [
                "mawk",
                "-v",
                f"k={every}",
                "-v",
                "OFS=\t",
                "!/^#/ && ++n % k == 0 { print $1, $2, $3 }",
            ],
            stdin=bgzip.stdout,
            stdout=thin,
            check=True,
        )
        bgzip.stdout.close()
        if bgzip.wait() != 0:
            raise RuntimeError(f"bgzip -dc {path} exited {bgzip.returncode}")
        thin.flush()
        frags = pd.read_csv(thin.name, sep="\t", header=None, names=["chrom", "start", "end"])
        chrombpnet = None
        if chrombpnet_python:
            run = subprocess.run(
                [chrombpnet_python, "-c", _CHROMBPNET_SHIFT, thin.name, genome, str(n_sample)],
                capture_output=True,
                text=True,
                check=True,
            )
            chrombpnet = [int(x) for x in run.stdout.split()[-2:]]
    n_total = len(frags)
    if n_total > n_sample:
        frags = frags.sample(n_sample, random_state=seed)
    forward, reverse = shift.get_nucleotide_freq(frags, genome, paired=True)
    d_plus, mse_plus = _fit(shift.ref_forward_bias, forward)
    d_minus, mse_minus = _fit(shift.ref_reverse_bias, reverse)
    plus, minus = shift.shift_present(d_plus, d_minus)
    stem = path.name[: -len(SUFFIX)]
    return stem, {
        "plus_shift": plus,
        "minus_shift": minus,
        "mse_plus": round(mse_plus, 6),
        "mse_minus": round(mse_minus, 6),
        "chrombpnet": chrombpnet,
        "n_fragments": int(len(frags)),
        "n_thinned": int(n_total),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--fragments-dir", required=True, type=Path)
    ap.add_argument("--genome", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--every", type=int, default=2000, help="keep every Nth fragment")
    ap.add_argument("--n-sample", type=int, default=100_000, help="fragments fed to the fit")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tmp-dir", default=None)
    ap.add_argument("--only", nargs="*", default=None, help="file stems to restrict to")
    ap.add_argument(
        "--chrombpnet-python",
        default=None,
        help="python of the chrombpnet environment; also run chrombpnet's detector",
    )
    log.add_logging_args(ap)
    args = ap.parse_args(argv)
    log.setup_from_args(args)

    files = sorted(args.fragments_dir.glob(f"*{SUFFIX}"))
    if args.only:
        files = [f for f in files if f.name[: -len(SUFFIX)] in set(args.only)]
    if not files:
        logger.error("no *%s under %s", SUFFIX, args.fragments_dir)
        return 1
    logger.info("detecting the Tn5 shift in %d file(s) on %d process(es)", len(files), args.jobs)
    jobs = [
        (f, args.genome, args.every, args.n_sample, args.seed, args.tmp_dir, args.chrombpnet_python)
        for f in files
    ]
    shifts = {}
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for stem, rec in pool.map(_detect, jobs):
            shifts[stem] = rec
            logger.info(
                "%-28s %+d/%+d  mse %.4f/%.4f  chrombpnet %s  (%d fragments)",
                stem, rec["plus_shift"], rec["minus_shift"],
                rec["mse_plus"], rec["mse_minus"], rec["chrombpnet"], rec["n_fragments"],
            )  # fmt: skip
    args.out.write_text(json.dumps(shifts, indent=2, sort_keys=True) + "\n")
    logger.info("-> %s", args.out)

    seen = {(r["plus_shift"], r["minus_shift"]) for r in shifts.values()}
    if len(seen) > 1:
        logger.warning("files disagree on the shift: %s", sorted(seen))
    shaky = [s for s, r in shifts.items() if max(r["mse_plus"], r["mse_minus"]) > MAX_MSE]
    if shaky:
        logger.error("fit above MSE %.3f (unreliable) for: %s", MAX_MSE, ", ".join(shaky))
    disagree = [
        s
        for s, r in shifts.items()
        if r["chrombpnet"] is not None and r["chrombpnet"] != [r["plus_shift"], r["minus_shift"]]
    ]
    if disagree:
        logger.error("scPrinter and chrombpnet disagree for: %s", ", ".join(disagree))
    if shaky or disagree:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
