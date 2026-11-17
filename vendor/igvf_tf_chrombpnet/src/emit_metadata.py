#!/usr/bin/env python3
"""Write a step's run-metadata JSON from a shell step.

The Python steps use ``utils.metadata.record()`` directly; the sbatch wrappers
that shell out to chrombpnet / modisco / finemo call this instead, so both
produce the identical schema and land in the same DuckDB table.

Called from the EXIT trap installed by ``metadata_start`` in lib/bash/common.sh,
so it runs whether the step succeeded, failed, or was cancelled with SIGTERM.
A SIGKILL (SLURM OOM, hard preemption) leaves no record at all -- that absence
is the only available signal.

Imports nothing beyond the standard library and ``utils.metadata``, so it runs
under any of the four conda environments, including ones without pyranges1.

Usage (see metadata_emit in lib/bash/common.sh, which builds this call):
  python emit_metadata.py \\
      --step 04.0.train_full_model --dataset igvf3_cardiomyocyte \\
      --out-dir "${metadata_dir}" --script "${BASH_SOURCE[0]}" \\
      --started-at 1758400000 --exit-status 0 \\
      --input  fragments=/path/frags.tsv.gz --input peaks=/path/peaks.narrowPeak \\
      --output model=/path/chrombpnet_nobias.h5 \\
      --param  fold=0 --param bias_suffix=_08 \\
      --tool   "chrombpnet=1.0.1"
"""

from __future__ import annotations  # PEP 585/604 annotations on the >= 3.9 bootstrap python

import argparse
import sys
import time
from pathlib import Path

# Make lib/python importable without an install step (works under the cluster
# conda envs, under pixi, and under a bare python).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib" / "python"))

from utils import log, metadata  # noqa: E402

logger = log.get_logger(__name__)


def pair(text: str) -> tuple[str, str]:
    """Parse ``key=value``; the value may itself contain '='."""
    key, sep, value = text.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError(f"expected key=value, got {text!r}")
    return key.strip(), value


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--step", required=True, help="Step name, e.g. 00.1.preprocess_peaks")
    p.add_argument("--dataset", default=None)
    p.add_argument("--out-dir", required=True, help="Metadata directory (metadata_dir)")
    p.add_argument("--script", default=None, help="Step script, for the GitHub permalink")
    p.add_argument("--status", default=None, choices=["ok", "failed"])
    p.add_argument("--exit-status", type=int, default=0)
    p.add_argument(
        "--started-at",
        type=float,
        default=None,
        help="Unix epoch when the step began (date +%%s), for the duration",
    )
    p.add_argument(
        "--input", dest="inputs", action="append", type=pair, default=[], metavar="ROLE=PATH"
    )
    p.add_argument(
        "--output", dest="outputs", action="append", type=pair, default=[], metavar="ROLE=PATH"
    )
    p.add_argument(
        "--param", dest="params", action="append", type=pair, default=[], metavar="KEY=VALUE"
    )
    p.add_argument(
        "--metric", dest="metrics", action="append", type=pair, default=[], metavar="KEY=VALUE"
    )
    p.add_argument(
        "--tool", dest="tools", action="append", type=pair, default=[], metavar="NAME=VERSION"
    )
    p.add_argument("--command", default=None, help="The command the step ran, for the record")
    log.add_logging_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    log.setup_from_args(args)

    md = metadata.StepMetadata(
        args.step, dataset=args.dataset, out_dir=args.out_dir, script=args.script
    )

    # The trap knows when the step began; without it the duration is just this
    # process's lifetime, which would be misleading, so leave it unset instead.
    if args.started_at:
        md.started_monotonic = time.monotonic() - max(0.0, time.time() - args.started_at)
        md.started_at = metadata._utc(args.started_at)

    md.exit_status = args.exit_status
    md.status = args.status or ("ok" if args.exit_status == 0 else "failed")
    if md.status == "failed" and not md.error:
        md.error = f"step exited {args.exit_status}"

    for role, path in args.inputs:
        md.add_input(role, path)
    for role, path in args.outputs:
        md.add_output(role, path)
    for key, value in args.params:
        md.add_param(key, value)
    for key, value in args.metrics:
        md.add_metric(key, value)
        # peak_rss_gb is a top-level field, not just a metric: it is the one
        # number SLURM sizing reads, and burying it in the parameters list
        # would mean an UNNEST for every --mem question. The step measured it
        # in the process that did the work, so unlike this emitter's own RSS
        # it is real. See StepMetadata.peak_rss_gb.
        if key == "peak_rss_gb":
            try:
                md.peak_rss_gb = float(value)
            except (TypeError, ValueError):
                pass
    for name, version in args.tools:
        md.add_tool(name, version)
    if args.command:
        md.add_param("command", args.command)

    # Provenance must never be the reason a step fails.
    try:
        md.write()
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not write run metadata: %s", exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
