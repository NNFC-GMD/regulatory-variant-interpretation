#!/usr/bin/env python3
"""Forward passes per second of each model at several batch sizes, with 1 and 2 processes per GPU.

A probe point is (model, processes per GPU). Its processes are copies of the model's predictor in --probe mode
(predict_cherimoya.py, predict_chrombpnet.py), all on one GPU, each pinned to its own CPUs (taskset) with its GPU
memory capped. They time the same code path the predictions run (random 2114-bp code rows, both strands, the
predictor's batching and padding) and start and end every measurement together (predict_common.barrier), so their
rates add up to what the GPU gives at that packing. Points run in parallel, one per GPU.

Models:
  cherimoya    predict_cherimoya.py --model <torch>: --batches, plus --cherimoya-extra-batches (default 2048)
  cbp_nobias   predict_chrombpnet.py --nobias <h5> --components nobias: --batches
  cbp_bias     predict_chrombpnet.py --bias <h5> --components bias: --batches
A pass is one sequence through one model.

Output (--out probe.json):
  the settings, and per point: the GPU, each process's exit status, and per batch the per-process passes/s, their
  total, warm-up seconds, and peak GPU memory; `best`: per model, the batch and packing with the highest total.
  probe.tsv beside it, one row per point and batch. <work-dir> (default <out dir>/work) keeps each process's JSON
  and log.

    python workflows/variant_scoring/probe_throughput.py --out <root>/probe/probe.json \\
        --cherimoya-python <python of the cherimoya env> --chrombpnet-python <python of the fork's cuda13 env> \\
        --cherimoya-model <cherimoya.fold_0.torch> --nobias-model <chrombpnet_nobias.h5> \\
        --bias-model <bias_model_scaled.h5> --gpus 0,1,2,3,4,5

Standard library only; any Python 3.10 or newer runs it.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("probe_throughput")

HERE = Path(__file__).resolve().parent
MODELS = ("cherimoya", "cbp_nobias", "cbp_bias")


@dataclass
class Point:
    model: str
    procs: int
    batches: list[int]
    gpu: str | None = None
    exit_codes: list[int | None] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def tag(self) -> str:
        return f"{self.model}.p{self.procs}"


def int_list(text: str) -> list[int]:
    values = [int(v) for v in text.split(",") if v.strip()]
    if not values or min(values) < 1:
        raise argparse.ArgumentTypeError(f"expected positive comma-separated integers, got {text!r}")
    return values


def allowed_cpus() -> list[int]:
    try:
        return sorted(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        return list(range(os.cpu_count() or 1))


def merge_alloc_conf(existing: str, fraction: float) -> str:
    """PYTORCH_CUDA_ALLOC_CONF with per_process_memory_fraction set to `fraction`, other settings kept."""
    parts = [p for p in existing.split(",") if p and not p.startswith("per_process_memory_fraction")]
    return ",".join([*parts, f"per_process_memory_fraction:{fraction:.3f}"])


def worker_env(base: dict, gpu: str, n_cpus: int, fraction: float) -> dict:
    """The environment of one probe process: its GPU, thread counts, and GPU memory caps for torch and JAX.
    jaxlib reads XLA_CLIENT_MEM_FRACTION or the deprecated XLA_PYTHON_CLIENT_MEM_FRACTION, and refuses both."""
    env = dict(base)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[key] = str(n_cpus)
    env["PYTORCH_CUDA_ALLOC_CONF"] = merge_alloc_conf(base.get("PYTORCH_CUDA_ALLOC_CONF", ""), fraction)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)
    env["XLA_CLIENT_MEM_FRACTION"] = f"{fraction:.3f}"
    return env


def worker_cmd(args, point: Point, json_path: Path, barrier_dir: Path) -> list[str]:
    common = ["--probe", "--probe-json", str(json_path), "--probe-batches", ",".join(map(str, point.batches)),
              "--probe-seconds", str(args.seconds), "--probe-warmup", str(args.warmup),
              "--barrier", str(barrier_dir), "--barrier-n", str(point.procs)]  # fmt: skip
    if point.model == "cherimoya":
        return [args.cherimoya_python, str(HERE / "predict_cherimoya.py"), "--model", str(args.cherimoya_model),
                "--precision", args.cherimoya_precision, *common]  # fmt: skip
    part, model = ("nobias", args.nobias_model) if point.model == "cbp_nobias" else ("bias", args.bias_model)
    return [args.chrombpnet_python, str(HERE / "predict_chrombpnet.py"), f"--{part}", str(model),
            "--components", part, "--precision", args.chrombpnet_precision, *common]  # fmt: skip


def run_point(args, point: Point, gpu: str, cpu_blocks: list[list[int]], work: Path) -> Point:
    """Run one point's processes on `gpu`, pinned to cpu_blocks[rank]; record their exit codes."""
    point.gpu = gpu
    barrier_dir = work / f"{point.tag}.barrier"
    shutil.rmtree(barrier_dir, ignore_errors=True)
    fraction = args.gpu_fraction / point.procs
    procs, logs = [], []
    t0 = time.monotonic()
    for rank in range(point.procs):
        json_path = work / f"{point.tag}.r{rank}.json"
        json_path.unlink(missing_ok=True)
        cpus = cpu_blocks[rank]
        cmd = ["taskset", "-c", ",".join(map(str, cpus)), *worker_cmd(args, point, json_path, barrier_dir)]
        log = open(work / f"{point.tag}.r{rank}.log", "w")  # closed below, after the process ends
        logs.append(log)
        log.write(" ".join(cmd) + "\n")
        log.flush()
        procs.append(subprocess.Popen(cmd, env=worker_env(os.environ, gpu, len(cpus), fraction), stdout=log,
                                      stderr=subprocess.STDOUT))  # fmt: skip
    # Poll rather than wait in turn: a process that fails would otherwise leave its partner waiting at a barrier.
    deadline = t0 + args.timeout
    while True:
        codes = [p.poll() for p in procs]
        if all(c is not None for c in codes):
            break
        failed = any(c not in (None, 0) for c in codes)
        if failed or time.monotonic() > deadline:
            logger.error("%s on GPU %s: %s; stopping its other processes", point.tag, gpu,
                         f"exit {codes}" if failed else f"timed out after {args.timeout:.0f} s")  # fmt: skip
            for p in procs:
                if p.poll() is None:
                    p.kill()
            for p in procs:
                p.wait()
            break
        time.sleep(1.0)
    for log in logs:
        log.close()
    point.exit_codes = [p.returncode for p in procs]
    point.seconds = round(time.monotonic() - t0, 1)
    logger.info("%s on GPU %s: exit %s in %.0f s", point.tag, gpu, point.exit_codes, point.seconds)
    return point


def collect(point: Point, work: Path) -> dict:
    """The point's per-batch totals from its processes' JSON files."""
    per_rank = []
    for rank in range(point.procs):
        path = work / f"{point.tag}.r{rank}.json"
        try:
            per_rank.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            per_rank.append(None)
    batches = []
    for b in point.batches:
        rates, warm, errors = [], [], []
        for payload in per_rank:
            hit = next((r for r in (payload or {}).get("results", []) if r.get("batch") == b), None) or {}
            rates.append(hit.get("passes_per_second"))
            warm.append(hit.get("warmup_seconds"))
            errors.append(hit.get("error") if hit else "no result (the process ended before this batch size)")
        complete = all(r is not None for r in rates)
        batches.append({"batch": b, "per_process_passes_per_second": rates,
                        "total_passes_per_second": round(sum(rates), 1) if complete else None,
                        "warmup_seconds": warm, "complete": complete,
                        "errors": [e for e in errors if e] or None})  # fmt: skip
    first = next((p for p in per_rank if p), {})
    return {"model": point.model, "procs_per_gpu": point.procs, "gpu": point.gpu, "exit_codes": point.exit_codes,
            "seconds": point.seconds, "batches": batches,
            "peak_gpu_memory_gb": [(p or {}).get("peak_gpu_memory_gb") for p in per_rank],
            "device": first.get("device") or first.get("jax_devices"), "precision": first.get("precision"),
            "forward_path": first.get("forward_path"), "versions": {k: first.get(k) for k in
                                                                    ("torch", "cherimoya", "keras", "jax")
                                                                    if k in first}}  # fmt: skip


def best_per_model(points: list[dict]) -> dict:
    best: dict = {}
    for p in points:
        for b in p["batches"]:
            total = b["total_passes_per_second"]
            if total is None:
                continue
            if p["model"] not in best or total > best[p["model"]]["total_passes_per_second"]:
                best[p["model"]] = {"batch": b["batch"], "procs_per_gpu": p["procs_per_gpu"],
                                    "total_passes_per_second": total}  # fmt: skip
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="probe.json; probe.tsv goes beside it")
    ap.add_argument("--work-dir", type=Path, help="per-process JSON and logs (default <out dir>/work)")
    ap.add_argument("--models", default=",".join(MODELS), help=f"comma-separated subset of {', '.join(MODELS)}")
    ap.add_argument("--cherimoya-python", help="python of the cherimoya environment")
    ap.add_argument("--chrombpnet-python", help="python of the chrombpnet fork's cuda13 environment")
    ap.add_argument("--cherimoya-model", type=Path)
    ap.add_argument("--nobias-model", type=Path)
    ap.add_argument("--bias-model", type=Path)
    ap.add_argument("--batches", type=int_list, default=int_list("64,256,512,1024"))
    ap.add_argument("--cherimoya-extra-batches", type=int_list, default=int_list("2048"))
    ap.add_argument("--procs", type=int_list, default=int_list("1,2"), help="processes per GPU (default 1,2)")
    ap.add_argument("--gpus", required=True, help="comma-separated GPU indices; one point runs per GPU at a time")
    ap.add_argument("--cpus-per-proc", type=int, default=8)
    ap.add_argument("--gpu-fraction", type=float, default=0.9,
                    help="share of a GPU's memory a point may use, split among its processes (default 0.9)")  # fmt: skip
    ap.add_argument("--seconds", type=float, default=20.0, help="timed seconds per batch size (default 20)")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=1800.0, help="seconds a point may take (default 1800)")
    ap.add_argument("--cherimoya-precision", default="default")
    ap.add_argument("--chrombpnet-precision", default="default")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%F %T")  # fmt: skip

    models = [m for m in args.models.split(",") if m]
    if not models or any(m not in MODELS for m in models):
        ap.error(f"--models takes a subset of {', '.join(MODELS)}")
    needs = {"cherimoya": ("cherimoya_python", "cherimoya_model"), "cbp_nobias": ("chrombpnet_python", "nobias_model"),
             "cbp_bias": ("chrombpnet_python", "bias_model")}  # fmt: skip
    for m in models:
        for attr in needs[m]:
            if not getattr(args, attr):
                ap.error(f"{m} needs --{attr.replace('_', '-')}")
    gpus = [g for g in args.gpus.split(",") if g]
    max_procs = max(args.procs)
    cpus = allowed_cpus()
    need = len(gpus) * max_procs * args.cpus_per_proc
    if need > len(cpus):
        ap.error(f"{len(gpus)} GPUs x {max_procs} processes x {args.cpus_per_proc} CPUs = {need}, "
                 f"but only {len(cpus)} CPUs are allowed")  # fmt: skip
    # CPU blocks: GPU slot i owns cpus[i * max_procs * k : (i + 1) * max_procs * k], one block of k per process.
    k = args.cpus_per_proc
    blocks = {g: [cpus[(i * max_procs + r) * k:(i * max_procs + r + 1) * k] for r in range(max_procs)]
              for i, g in enumerate(gpus)}  # fmt: skip

    points = [Point(m, p, args.batches + (args.cherimoya_extra_batches if m == "cherimoya" else []))
              for m in models for p in sorted(args.procs, reverse=True)]  # fmt: skip
    work = args.work_dir or args.out.parent / "work"
    work.mkdir(parents=True, exist_ok=True)
    free: queue.Queue[str] = queue.Queue()
    for g in gpus:
        free.put(g)
    lock = threading.Lock()

    def task(point: Point) -> Point:
        gpu = free.get()
        try:
            with lock:
                logger.info("%s: GPU %s, batches %s", point.tag, gpu, point.batches)
            return run_point(args, point, gpu, blocks[gpu], work)
        finally:
            free.put(gpu)

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        done = list(pool.map(task, points))
    results = [collect(p, work) for p in done]
    payload = {"written": time.strftime("%Y-%m-%d %H:%M:%S %z"), "host": os.uname().nodename,
               "seconds": round(time.monotonic() - t0, 1),
               "settings": {"models": models, "batches": args.batches,
                            "cherimoya_extra_batches": args.cherimoya_extra_batches, "procs_per_gpu": args.procs,
                            "gpus": gpus, "cpus_per_proc": k, "gpu_fraction_per_point": args.gpu_fraction,
                            "timed_seconds_per_batch": args.seconds, "warmup_calls": args.warmup,
                            "cherimoya_precision": args.cherimoya_precision,
                            "chrombpnet_precision": args.chrombpnet_precision,
                            "pass": "one 2114-bp sequence through one model; both strands are timed"},
               "points": results, "best": best_per_model(results)}  # fmt: skip
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_name(args.out.name + ".partial")
    partial.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(partial, args.out)
    with open(args.out.with_suffix(".tsv"), "w") as fh:
        fh.write("model\tprocs_per_gpu\tbatch\ttotal_passes_per_second\tper_process\tgpu\tpeak_gpu_memory_gb\terror\n")
        for r in results:
            for b in r["batches"]:
                error = (b["errors"] or [""])[0].replace("\t", " ").replace("\n", " ")[:200]
                fh.write("\t".join(map(str, [r["model"], r["procs_per_gpu"], b["batch"], b["total_passes_per_second"],
                                             ",".join(map(str, b["per_process_passes_per_second"])), r["gpu"],
                                             ",".join(map(str, r["peak_gpu_memory_gb"])), error])) + "\n")  # fmt: skip
    for model, b in payload["best"].items():
        logger.info("best %s: %.0f passes/s at batch %d, %d process(es) per GPU", model, b["total_passes_per_second"],
                    b["batch"], b["procs_per_gpu"])  # fmt: skip
    failed = [r for r in results if any(c != 0 for c in r["exit_codes"])]
    for r in failed:
        logger.error("%s x%d: exit %s; see %s", r["model"], r["procs_per_gpu"], r["exit_codes"], work)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
