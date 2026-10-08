#!/usr/bin/env python3
"""Time the cheri_dls engine on real peaks: warm-up (compile) time, steady pairs/s and memory, in one fresh process.

This is the first piece of the benchmark runner of the design (analyses/attr_pilot/design/final_design.md §4, §16 and
the benchmark plan). It reads an attribute.json the way `cherimoya attribute` does (cherimoya_cli/commands/
attribute.py:21-86): the same parameter merge, the same extract_loci call, the same N filter and the same centred
window. It then attributes the first --peaks kept peaks with cheri_dls.Engine on CUDA, drawing the references with
dls_refs.RefProducer as the CLI will, and writes one JSON record:
- the engine's record (meta): step size, warm-up and compile seconds, steady pairs/s (pairs of every step after the
  first over the wall time between the first and the last step's arrival), host synchronizations, out-of-memory
  halvings, the most a step allocated against cheri_dls.step_bytes, the precision flags, the reference queue's stalls
  and a summary of the convergence deltas;
- peak allocated and reserved GPU memory of the whole process, wall times of each stage, and the co-tenant processes
  on the GPU before and after (nvidia-smi);
- with --stock-attr, the per-peak rel-L2 and Pearson correlation of the attributions against a stock
  `cherimoya attribute` run of the same JSON, an early look at criteria E3/E5 (not a gate).

Compile time depends on the caches. A cold compile needs empty TORCHINDUCTOR_CACHE_DIR and TRITON_CACHE_DIR
directories; a second process with the same directories measures the warm one. The engine keeps one Inductor cache
directory per precision mode under TORCHINDUCTOR_CACHE_DIR.

The GPUs are shared, so the process caps its CUDA caching allocator (--gpu-mem-cap-gb), and the engine plans its
steps within --mem-budget-gb. Set TORCHINDUCTOR_COMPILE_THREADS=1 to keep Inductor from forking a compile pool next
to the reference workers. Run it inside a box, on one GPU and pinned cores:

    CUDA_VISIBLE_DEVICES=<GPU> TORCHINDUCTOR_COMPILE_THREADS=1 NUMBA_CACHE_DIR=<node-local> taskset -c <cores> \\
        python workflows/cherimoya/dls_bench.py -p attribute.json --peaks 512 --seqs-per-step 4 --out run.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger("dls_bench")


def _now_et() -> str:
    """The time in ET (America/New_York), where the user is; the cluster clock runs on CEST."""
    from zoneinfo import ZoneInfo

    return dt.datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds")


def _compute_apps() -> list[dict]:
    """Compute processes on the visible GPU, from nvidia-smi: pid and used memory in MiB."""
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not gpu or "," in gpu:
        return []
    try:
        uuid = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "-i", gpu],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        ).stdout.strip()
        rows = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        ).stdout.splitlines()
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("nvidia-smi failed: %s", exc)
        return []
    apps = []
    for row in rows:
        parts = [p.strip() for p in row.split(",")]
        if len(parts) == 3 and parts[0] == uuid:
            apps.append({"pid": int(parts[1]), "used_mib": int(parts[2])})
    return apps


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-p", "--parameters", required=True, help="the attribute.json of `cherimoya attribute`")
    parser.add_argument("--peaks", type=int, default=512, help="attribute the first PEAKS kept peaks (default 512)")
    parser.add_argument("--seqs-per-step", default="auto", help="peaks per step, or auto (default)")
    parser.add_argument("--mem-budget-gb", type=float, default=8.0, help="the engine's budget in GB (default 8)")
    parser.add_argument("--gpu-mem-cap-gb", type=float, default=7.5, help="cap of this process's CUDA allocator, GB")
    parser.add_argument("--precision", default="reference", choices=("reference", "fp32"))
    parser.add_argument("--compile", default="default", choices=("default", "none"))
    parser.add_argument("--ref-workers", type=int, default=2, help="RefProducer worker processes (default 2)")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--stock-attr", help="attr .npz of a stock run of the same JSON, to compare against")
    parser.add_argument("--label", default="", help="a free-text label stored in the record")
    parser.add_argument("--out", required=True, help="where to write the JSON record")
    args = parser.parse_args(argv)
    if args.seqs_per_step != "auto":
        args.seqs_per_step = int(args.seqs_per_step)
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    began = time.perf_counter()
    record: dict = {"args": vars(args), "started_et": _now_et(), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES")}
    record["env"] = {
        k: os.environ.get(k)
        for k in (
            "TORCHINDUCTOR_CACHE_DIR",
            "TRITON_CACHE_DIR",
            "TORCHINDUCTOR_COMPILE_THREADS",
            "NUMBA_CACHE_DIR",
            "PYTORCH_CUDA_ALLOC_CONF",
        )
    }
    record["co_tenants_before"] = _compute_apps()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import dls_refs  # noqa: E402  (exports a node-local NUMBA_CACHE_DIR before cherimoya imports tangermeme)

    dls_refs.ensure_numba_cache_dir()
    import cheri_dls  # noqa: E402
    import numpy as np  # noqa: E402
    import torch  # noqa: E402
    from cherimoya_cli.defaults import default_attribute_parameters  # noqa: E402
    from cherimoya_cli.utils import merge_parameters  # noqa: E402
    from tangermeme.io import extract_loci  # noqa: E402

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device is visible; run inside a box with CUDA_VISIBLE_DEVICES set")
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.gpu_mem_cap_gb * 1e9 / total), 0)
    record["timings_s"] = {"imports": time.perf_counter() - began}

    # attribute.py:21-86, verbatim in effect
    t = time.perf_counter()
    parameters = merge_parameters(args.parameters, default_attribute_parameters)
    if parameters["algorithm"] != "deep_lift_shap" or parameters["output"] != "counts":
        raise SystemExit("this benchmark runs deep_lift_shap on the counts head")
    model = cheri_dls.load_model(parameters["model"])
    record["timings_s"]["load"] = time.perf_counter() - t
    t = time.perf_counter()
    X, idxs = extract_loci(
        sequences=parameters["sequences"],
        loci=parameters["loci"],
        chroms=parameters["chroms"],
        in_window=parameters["in_window"],
        max_jitter=0,
        exclusion_lists=parameters["exclusion_lists"],
        ignore=list("QWERYUIOPSDFHJKLZXVBNM"),
        return_mask=True,
        verbose=False,
    )
    n_idxs = X.sum(dim=(1, 2)) == X.shape[-1]
    X = X[n_idxs]
    record["timings_s"]["extract_loci"] = time.perf_counter() - t
    mid = X.shape[-1] // 2
    start = mid - parameters["attr_window"] // 2
    end = start + parameters["attr_window"]
    n = min(args.peaks, len(X))
    record["peaks"] = {"kept": len(X), "attributed": n, "length": X.shape[-1], "window": [start, end]}

    engine = cheri_dls.Engine(
        model,
        output="counts",
        group=parameters["group"],
        device="cuda",
        precision=args.precision,
        n_shuffles=parameters["n_shuffles"],
        random_state=parameters["random_state"],
        head_rows=2 * parameters["batch_size"],
        seqs_per_step=args.seqs_per_step,
        mem_budget_gb=args.mem_budget_gb,
        compile=args.compile,
        warning_threshold=parameters["warning_threshold"],
        ref_workers=args.ref_workers,
        deterministic=args.deterministic,
    )
    t = time.perf_counter()
    result = engine.run(X[:n], start, end)
    record["timings_s"]["engine_run"] = time.perf_counter() - t
    meta = result.meta
    record["meta"] = meta
    record["gpu_memory"] = {
        "max_allocated_bytes": torch.cuda.max_memory_allocated(),
        "max_reserved_bytes": torch.cuda.max_memory_reserved(),
        "step_peak_bytes": meta["step_peak_bytes"],
        "step_estimate_bytes": meta["step_estimate_bytes"],
        "step_peak_over_estimate": meta["step_peak_bytes"] / meta["step_estimate_bytes"],
    }
    record["co_tenants_after"] = _compute_apps()
    attr = result.attr.float()
    record["attr_finite"] = bool(torch.isfinite(attr).all())

    if args.stock_attr:
        stock = torch.from_numpy(np.load(args.stock_attr)["arr_0"][:n]).double()
        ours = attr.double()
        diff = (ours - stock).flatten(1)
        rel = diff.norm(dim=1) / stock.flatten(1).norm(dim=1).clamp_min(1e-30)
        a = ours.flatten(1) - ours.flatten(1).mean(dim=1, keepdim=True)
        b = stock.flatten(1) - stock.flatten(1).mean(dim=1, keepdim=True)
        pearson = (a * b).sum(dim=1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-300)
        q = torch.tensor([0.5, 0.99, 1.0], dtype=torch.float64)
        record["vs_stock"] = {
            "stock_attr": args.stock_attr,
            "peaks": n,
            "rel_l2_q50_q99_max": [float(v) for v in torch.quantile(rel, q)],
            "pearson_min": float(pearson.min()),
            "pearson_median": float(pearson.median()),
        }
    record["timings_s"]["total"] = time.perf_counter() - began
    record["finished_et"] = _now_et()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, default=str) + "\n")
    logger.info(
        "S=%s: warm-up %.1f s (compile %.1f s), steady %.0f pairs/s, %d steps, %d syncs, step peak %.2f GB = %.2f x "
        "estimate, oom %s; wrote %s",
        meta["seqs_per_step"],
        meta["warmup_s"],
        meta["compile_s"],
        meta["steady_pairs_per_s"] or float("nan"),
        meta["steps"],
        meta["host_syncs"],
        meta["step_peak_bytes"] / 1e9,
        record["gpu_memory"]["step_peak_over_estimate"],
        meta["oom"],
        out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
