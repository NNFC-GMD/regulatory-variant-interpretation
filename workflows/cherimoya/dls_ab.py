#!/usr/bin/env python3
"""A/B benchmark: the stock `cherimoya attribute` against attribute_fast.py under one condition (design step 10).

The benchmark plan (analyses/attr_pilot/design/final_rest.json, benchmark_plan; criterion G1) times both commands on
the same GPU, the same pinned cores and the same model, genome and JSON settings. Each run is a fresh process, and
the tools alternate (stock, ours, stock, ours) so that load drifting on a shared GPU falls on both. A speed-up is
quoted only within one condition: C1 is a GPU shared with the node's training jobs, as today, and C2 a free GPU.

Subcommands, all run inside a box. `run` needs the GPU; `report` reads the outputs with numpy.
- prepare: writes one JSON per run of --order (default stock,fast,stock,fast) into <out>/json/, from a base
  attribute.json with --set KEY=JSON applied and the same --fast block (the stock command ignores it). The JSONs
  differ only in the loci and the three output paths (<out>/<tag>/). The stock runs attribute the first
  --stock-peaks lines of the base's loci file (copied to <out>/peaks_first<N>.<ext>), and ours all of it.
  --stock-env and --fast-env KEY=VALUE give each tool's environment, e.g. its cache directories. Writes
  <out>/plan.json: tags, tools, repetitions, commands and environments. Stock runs as users run it, through the
  `cherimoya` console script of this Python's environment.
- run: runs the plan in order, one process at a time, each with this process's environment plus its tool's. It
  records the state of every GPU once at the start, then for each run, in <out>/runs.json:
  - wall time, exit status, max RSS, CPU time and context switches (os.wait4). These cover the process and the
    children it waited for, which for ours are the reference workers; max RSS is the largest single process.
  - the GPU (CUDA_VISIBLE_DEVICES, a single index): its processes before and after the run, with memory, name and
    command line (nvidia-smi --query-compute-apps, /proc); the most memory each used during the run; the GPU's
    utilization, polled every --poll-s seconds; and the run's own peak memory.
  - the pinned cores (this process's affinity, which the run inherits): their busy and run-queue-delay seconds over
    the run, from /proc/schedstat. Subtracting the run's and this driver's CPU time leaves what other processes took
    on those cores.
  - stock: every tqdm update, dated by this process's clock as it arrives through the pipe (the output is copied to
    <tag>.log unchanged, the updates to <tag>.progress.tsv). Ours: a summary of its meta sidecar.
  - each cache directory's file count before the run (cold or warm), and start and end in ET.
- report: writes --tsv, one row per run, and --md. The Markdown gives the per-run measurements, steady pairs/s per
  tool and the speed-up within the condition: the ratio of the tools' mean steady rates, and per repetition. It
  also gives G1's verdict for the condition, projections to a --fold-peaks fold and --models models, co-tenancy, and
  as context the agreement of the outputs on the peaks both tools attributed. --notes appends a Markdown file.

Steady pairs/s. Stock: the least-squares slope of its tqdm pair counter against the arrival times of the updates.
It starts after the first 5 batches (n >= 5 x batch_size) and runs to the last update. The first batch's seconds are
its warm-up: cuDNN and Triton autotuning, since it compiles nothing. The same slope over tqdm's own elapsed field
(1 s resolution) is kept as a check. Ours: steady_pairs_per_s of the meta sidecar, the pairs of every step after the
first over the time between the first and the last step's arrival. Its compile (warm-up) time is the meta's.

    python workflows/cherimoya/dls_ab.py prepare -p attribute.json --out-dir <c1> --stock-peaks 1000 \\
        --set batch_size=20 --fast '{"mem_budget_gb": 7.5, "ref_workers": 3}' --fast-env NUMBA_CACHE_DIR=<dir>
    CUDA_VISIBLE_DEVICES=<GPU> taskset -c <cores> python workflows/cherimoya/dls_ab.py run --out-dir <c1>
    python workflows/cherimoya/dls_ab.py report --out-dir <c1> --condition C1 --tsv c1.tsv --md c1.md
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import resource
import socket
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import dls_e2e  # noqa: E402  (stdlib only at import)
from dls_bench import _now_et  # noqa: E402

logger = logging.getLogger("dls_ab")

TOOLS = ("stock", "fast")
SKIP_BATCHES = 5
"""Stock's steady rate starts after this many batches (the benchmark plan: 'after the first 5 batches')."""
FOLD_PEAKS, MODELS = 290_000, 195
"""A fold's peaks and the number of models (39 samples x 5 folds), for the projections."""
G1_SHARED = 5.0
"""G1, shared GPU: ours >= 5 x stock under the same co-tenancy, A/B interleaved."""
ENV_KEYS = (*dls_e2e.ENV_KEYS, "PYTHONPATH")
COLUMNS = (
    "condition",
    "run",
    "tag",
    "tool",
    "rep",
    "cache",
    "precision",
    "head",
    "batch_size",
    "S",
    "peaks",
    "pairs",
    "steady_pairs_per_s",
    "steady_method",
    "wall_s",
    "startup_s",
    "compile_s",
    "self_check_s",
    "write_s",
    "peak_rss_gb",
    "peak_gpu_mib",
    "gpu_alloc_max_gb",
    "gpu_util_mean_pct",
    "gpu_idle_pct",
    "co_tenants",
    "co_tenant_mib",
    "other_cpu_cores",
    "cpu_delay_cores",
    "start_et",
    "end_et",
    "exit",
)


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    tmp.replace(path)


# ---------------------------------------------------------------------------------------------------------------------
# rates and accounting (pure)


def attribution_bar(progress: list) -> tuple[int | None, list[tuple[float, int, float | None]]]:
    """The attribution bar among a run's tqdm updates: the unnamed bar with the most updates (deep_lift_shap's trange;
    extract_loci's bar is named 'Loading Loci'). Returns its total and its (t, n, elapsed_s) updates."""
    groups: dict[int, list] = {}
    for t, desc, n, total, elapsed in progress:
        if desc == "":
            groups.setdefault(total, []).append((t, n, elapsed))
    if not groups:
        return None, []
    total = max(groups, key=lambda k: len(groups[k]))
    return total, groups[total]


def steady_rate(points: list[tuple[float, int]], batch_size: int, skip_batches: int = SKIP_BATCHES) -> dict | None:
    """The steady rate of a progress counter: points (t, n) in arrival order, t in seconds.

    Each n keeps the time it was first seen (tqdm prints its final state twice). The window holds the points with
    n >= skip_batches x batch_size; rate_ls is the least-squares slope of n against t there, and rate_two_point is
    (n_last - n_first) / (t_last - t_first). first_batch_s is the time from the bar's 0 to its first n > 0.
    """
    first: dict[int, float] = {}
    for t, n in points:
        first.setdefault(n, t)
    pts = sorted((t, n) for n, t in first.items())
    if not pts:
        return None
    zero = first.get(0)
    positive = [(t, n) for t, n in pts if n > 0]
    out = {
        "bar_start_s": zero,
        "first_batch_s": positive[0][0] - zero if zero is not None and positive else None,
        "t_end_s": pts[-1][0],
        "n_end": pts[-1][1],
        "skip_batches": skip_batches,
        "rate_ls": None,
        "rate_two_point": None,
    }
    window = [(t, n) for t, n in pts if n >= skip_batches * batch_size]
    if len(window) >= 3 and window[-1][0] > window[0][0]:
        ts = [float(t) for t, _ in window]
        ns = [float(n) for _, n in window]
        mt, mn = statistics.fmean(ts), statistics.fmean(ns)
        sxx = sum((t - mt) ** 2 for t in ts)
        sxy = sum((t - mt) * (n - mn) for t, n in zip(ts, ns))
        out.update(
            rate_ls=sxy / sxx,
            rate_two_point=(ns[-1] - ns[0]) / (ts[-1] - ts[0]),
            window_start_n=int(ns[0]),
            window_pairs=int(ns[-1] - ns[0]),
            window_s=ts[-1] - ts[0],
            window_points=len(window),
        )
    return out


def stock_steady(progress: list, batch_size: int) -> dict:
    """Stock's rates from its dated tqdm updates: on this driver's clock (the measurement) and on tqdm's elapsed field
    (a 1 s-resolution check), and the span of extract_loci's 'Loading Loci' bar."""
    total, bar = attribution_bar(progress)
    loading = [u[0] for u in progress if u[1] == "Loading Loci"]
    return {
        "pairs_total": total,
        "host_clock": steady_rate([(t, n) for t, n, _ in bar], batch_size),
        "tqdm_elapsed": steady_rate([(e, n) for _, n, e in bar if e is not None], batch_size),
        "extract_s": loading[-1] - loading[0] if len(loading) > 1 else None,
    }


def read_schedstat(cpus, path: str = "/proc/schedstat") -> dict[int, tuple[int, int]]:
    """Per CPU in cpus: (ns tasks ran on it, ns tasks waited to run on it), from /proc/schedstat (version 15: the
    7th and 8th numbers after the cpuN label)."""
    wanted, out = set(cpus), {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("cpu"):
                parts = line.split()
                cpu = int(parts[0][3:])
                if cpu in wanted:
                    out[cpu] = (int(parts[7]), int(parts[8]))
    return out


def cpu_account(before: dict, after: dict, tool_cpu_s: float, driver_cpu_s: float, wall_s: float) -> dict:
    """CPU time on the pinned cores over a run: busy and run-queue-delay seconds summed over the cores, and what is
    left for other processes after the run's own CPU time and this driver's (its threads and nvidia-smi calls)."""
    cpus = sorted(set(before) & set(after))
    busy = sum(after[c][0] - before[c][0] for c in cpus) / 1e9
    delay = sum(after[c][1] - before[c][1] for c in cpus) / 1e9
    other = busy - tool_cpu_s - driver_cpu_s
    return {
        "cpus": cpus,
        "busy_s": busy,
        "run_delay_s": delay,
        "tool_cpu_s": tool_cpu_s,
        "driver_cpu_s": driver_cpu_s,
        "other_cpu_s": other,
        "other_cpu_cores": other / wall_s if wall_s > 0 else None,
        "run_delay_cores": delay / wall_s if wall_s > 0 else None,
        "tool_cpu_cores": tool_cpu_s / wall_s if wall_s > 0 else None,
    }


def project(row: dict, fold_peaks: int, models: int, n_shuffles: int) -> dict:
    """A run's projection to one fold of fold_peaks peaks and to `models` folds.

    steady: the fold's pairs at the run's steady rate alone. end_to_end: the run's measured wall time, extended at
    its steady rate by the fold's extra pairs and at its per-peak cost (extract and write) by the extra peaks; at
    the run's own size it gives back its wall time.
    """
    rate, pairs, peaks = row["steady_pairs_per_s"], row["pairs"], row["peaks"]
    fold_pairs = fold_peaks * n_shuffles
    steady_s = fold_pairs / rate
    e2e_s = row["wall_s"] + (fold_pairs - pairs) / rate + row["per_peak_s"] * (fold_peaks - peaks)
    return {
        "fold_pairs": fold_pairs,
        "steady_min": steady_s / 60,
        "end_to_end_min": e2e_s / 60,
        "steady_gpu_h": steady_s * models / 3600,
        "end_to_end_gpu_h": e2e_s * models / 3600,
    }


def speedups(rows: list[dict]) -> dict:
    """Ours over stock within one condition: the ratio of the mean steady rates, per repetition, and the extremes
    (slowest ours over fastest stock and the reverse)."""
    rows = [r for r in rows if r["steady_pairs_per_s"]]
    rate = {tool: [r["steady_pairs_per_s"] for r in rows if r["tool"] == tool] for tool in TOOLS}
    if not rate["stock"] or not rate["fast"]:
        return {}
    by_rep = {}
    for r in rows:
        by_rep.setdefault(r["rep"], {})[r["tool"]] = r["steady_pairs_per_s"]
    per_rep = {rep: v["fast"] / v["stock"] for rep, v in sorted(by_rep.items()) if "fast" in v and "stock" in v}
    mean = {tool: statistics.fmean(v) for tool, v in rate.items()}
    return {
        "stock_mean": mean["stock"],
        "fast_mean": mean["fast"],
        "ratio_of_means": mean["fast"] / mean["stock"],
        "per_rep": per_rep,
        "min": min(rate["fast"]) / max(rate["stock"]),
        "max": max(rate["fast"]) / min(rate["stock"]),
    }


def g1_shared(speed: dict) -> bool:
    """G1's shared-GPU clause: the ratio of the mean steady rates and every repetition's ratio at least G1_SHARED."""
    return (
        bool(speed) and speed["ratio_of_means"] >= G1_SHARED and all(v >= G1_SHARED for v in speed["per_rep"].values())
    )


# ---------------------------------------------------------------------------------------------------------------------
# prepare


def first_lines(src: Path, dst: Path, n: int) -> int:
    """Copy the first n lines of src to dst; it is an error for src to have fewer."""
    lines = []
    with open(src) as fh:
        for line in fh:
            if len(lines) == n:
                break
            lines.append(line)
    if len(lines) < n:
        raise SystemExit(f"{src} has {len(lines)} lines, fewer than {n}")
    dst.write_text("".join(lines))
    return n


def _pairs(items: list[str] | None, what: str) -> dict:
    out = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not key or not sep:
            raise SystemExit(f"{what} takes KEY=VALUE, got {item!r}")
        out[key] = value
    return out


def cmd_prepare(args: argparse.Namespace) -> int:
    out = Path(args.out_dir).resolve()
    base = json.loads(Path(args.parameters).read_text())
    for key, value in _pairs(args.set, "--set").items():
        base[key] = json.loads(value)
    if args.fast is not None:
        base["fast"] = json.loads(args.fast)
    loci = [base["loci"]] if isinstance(base["loci"], str) else list(base["loci"])
    if len(loci) != 1:
        raise SystemExit(f"prepare needs one loci file in the base JSON, got {loci}")
    order = [t.strip() for t in args.order.split(",") if t.strip()]
    if not order or any(t not in TOOLS for t in order):
        raise SystemExit(f"--order takes a comma-separated list of {TOOLS}, got {args.order!r}")
    for key in ("batch_size", "n_shuffles"):
        if not isinstance(base.get(key), int):
            raise SystemExit(f"the base JSON (after --set) needs an integer {key}")
    src = Path(loci[0]).resolve()
    with open(src) as fh:
        all_lines = sum(1 for _ in fh)
    out.mkdir(parents=True, exist_ok=True)
    stock_loci = out / f"peaks_first{args.stock_peaks}{''.join(src.suffixes) or '.bed'}"
    first_lines(src, stock_loci, args.stock_peaks)
    bindir = Path(sys.executable).parent
    env = {"stock": _pairs(args.stock_env, "--stock-env"), "fast": _pairs(args.fast_env, "--fast-env")}
    reps = dict.fromkeys(TOOLS, 0)
    runs, written = [], []
    for i, tool in enumerate(order, 1):
        reps[tool] += 1
        tag = f"{i}_{tool}"
        params = dict(base)
        params["loci"] = [str(stock_loci if tool == "stock" else src)]
        (out / tag).mkdir(exist_ok=True)
        for key in dls_e2e.OUTPUTS:
            params[key] = str(out / tag / Path(base[key]).name)
        path = out / "json" / f"{tag}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(params, indent=4) + "\n")
        written.append(params)
        command = (
            [str(bindir / "cherimoya"), "attribute", "-p", str(path)]
            if tool == "stock"
            else [sys.executable, str(HERE / "attribute_fast.py"), "-p", str(path)]
        )
        runs.append(
            {
                "tag": tag,
                "tool": tool,
                "rep": reps[tool],
                "json": str(path),
                "command": command,
                "env": env[tool],
                "loci_lines": args.stock_peaks if tool == "stock" else all_lines,
                "batch_size": params["batch_size"],
                "n_shuffles": params["n_shuffles"],
            }
        )
    differing = sorted({k for p in written for k in p if any(p[k] != q.get(k) for q in written)})
    assert set(differing) <= {"loci", *dls_e2e.OUTPUTS}, differing
    _write_json(
        out / "plan.json",
        {
            "base": str(Path(args.parameters).resolve()),
            "set": args.set,
            "fast": base.get("fast"),
            "order": order,
            "stock_loci": str(stock_loci),
            "fast_loci": str(src),
            "differ_only_in": differing,
            "prepared_et": _now_et(),
            "runs": runs,
        },
    )
    logger.info("wrote %d JSONs and %s; they differ only in %s", len(runs), out / "plan.json", differing)
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# run


def _smi(query: list[str]) -> list[str]:
    return subprocess.run(
        ["nvidia-smi", *query, "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    ).stdout.splitlines()


def _cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        return ""


def process_kind(cmdline: str) -> str:
    """A short label for a GPU co-tenant, from its command line."""
    low = cmdline.lower()
    for needle, label in (
        ("cherimoya fit", "Cherimoya training"),
        ("attribute_fast", "attribute_fast.py"),
        ("cherimoya attribute", "cherimoya attribute"),
        ("modisco", "TF-MoDISco"),
        ("chrombpnet", "ChromBPNet"),
    ):
        if needle in low:
            return label
    return Path(cmdline.split(" ", 1)[0]).name if cmdline else "?"


def gpu_processes(uuid: str) -> list[dict]:
    """The compute processes on one GPU: pid, MiB, nvidia-smi's process name, command line and a label."""
    procs = []
    for row in _smi(["--query-compute-apps=gpu_uuid,pid,process_name,used_memory"]):
        parts = [p.strip() for p in row.split(",")]
        if len(parts) < 4 or parts[0] != uuid or not parts[1].isdigit():
            continue
        pid, mib = int(parts[1]), parts[-1]
        cmd = _cmdline(pid)
        procs.append(
            {
                "pid": pid,
                "used_mib": int(mib) if mib.isdigit() else None,
                "name": ",".join(parts[2:-1]),
                "kind": process_kind(cmd),
                "cmdline": cmd[:300],
            }
        )
    return procs


def all_gpus() -> list[dict]:
    """Every GPU's index, UUID, memory in use and utilization, and its number of compute processes."""
    counts: dict[str, int] = {}
    for gpu, _, _ in dls_e2e.parse_compute_apps(_smi(["--query-compute-apps=gpu_uuid,pid,used_memory"])):
        counts[gpu] = counts.get(gpu, 0) + 1
    gpus = []
    for row in _smi(["--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu"]):
        index, uuid, used, total, util = (p.strip() for p in row.split(","))
        gpus.append(
            {
                "index": int(index),
                "uuid": uuid,
                "memory_used_mib": int(used),
                "memory_total_mib": int(total),
                "utilization_pct": int(util) if util.isdigit() else None,
                "processes": counts.get(uuid, 0),
            }
        )
    return gpus


def cache_state(env: dict) -> dict:
    """The number of files under each *_CACHE_DIR of env (None if the directory does not exist)."""
    state = {}
    for key, value in sorted(env.items()):
        if key.endswith("_CACHE_DIR"):
            path = Path(value)
            state[key] = sum(len(files) for _, _, files in os.walk(path)) if path.is_dir() else None
    return state


def fast_meta_summary(params: dict) -> dict | None:
    """The parts of attribute_fast.py's meta sidecar the benchmark reports."""
    path = Path(params["attr_filename"].removesuffix(".npz") + ".meta.json")
    if not path.exists():
        return None
    m = json.loads(path.read_text())
    return {
        "file": str(path),
        "precision": m["precision"],
        "seqs_per_step": m["seqs_per_step"],
        "head_rows": m["head_rows"],
        "compile": m["compile"],
        "n_shuffles": m["n_shuffles"],
        "inputs": m["inputs"],
        "timings_s": m["timings_s"],
        "memory": m["memory"],
        "engine": {k: m["engine"].get(k) for k in ("steps", "padded_peaks", "host_syncs", "oom", "reference_stats")},
        "deltas": m["deltas"],
        "self_check": {k: m["self_check"].get(k) for k in ("pass", "max_abs", "seconds")},
        "gpu": {k: m["gpu"].get(k) for k in ("name", "driver", "CUDA_VISIBLE_DEVICES")},
        "code": m["code"],
    }


def _cpu_s(usage: resource.struct_rusage) -> float:
    return usage.ru_utime + usage.ru_stime


def _write_progress(path: Path, progress: list) -> None:
    lines = ["t_s\tdesc\tn\ttotal\telapsed_s"]
    lines += [f"{t:.4f}\t{desc}\t{n}\t{total}\t{'' if e is None else e}" for t, desc, n, total, e in progress]
    path.write_text("\n".join(lines) + "\n")


def cmd_run(args: argparse.Namespace) -> int:
    out = Path(args.out_dir).resolve()
    plan = json.loads((out / "plan.json").read_text())
    runs_path = out / "runs.json"
    runs = json.loads(runs_path.read_text()) if runs_path.exists() else {}
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    uuid = None
    if gpu.isdigit():
        uuid = _smi(["--query-gpu=uuid", "-i", gpu])[0].strip()
    elif gpu:
        raise SystemExit(f"CUDA_VISIBLE_DEVICES must be one GPU index, got {gpu!r}")
    cpus = sorted(os.sched_getaffinity(0))
    runs.setdefault("setups", []).append(
        {
            "started_et": _now_et(),
            "host": socket.gethostname(),
            "CUDA_VISIBLE_DEVICES": gpu,
            "gpu_uuid": uuid,
            "cpus": cpus,
            "env": {k: os.environ.get(k) for k in ENV_KEYS},
            "gpus_at_start": all_gpus() if uuid else [],
            "only": args.only,
        }
    )
    runs.setdefault("runs", {})
    for entry in plan["runs"]:
        tag = entry["tag"]
        if args.only and tag not in args.only:
            continue
        env = dict(os.environ)
        env.update(entry["env"])
        for key, value in entry["env"].items():
            if key.endswith("_CACHE_DIR"):
                Path(value).mkdir(parents=True, exist_ok=True)
        record = {k: entry[k] for k in ("tag", "tool", "rep", "json", "env", "batch_size", "n_shuffles")}
        record["cache_files_before"] = cache_state(entry["env"])
        record["gpu_before"] = gpu_processes(uuid) if uuid else []
        self0, kids0 = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
        sched0 = read_schedstat(cpus)
        result = dls_e2e._run_one(
            tag, entry["command"], out / f"{tag}.log", args.poll_s, env=env, progress=True, uuid=uuid
        )
        sched1 = read_schedstat(cpus)
        self1, kids1 = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
        record["gpu_after"] = gpu_processes(uuid) if uuid else []
        progress = result.pop("progress")
        _write_progress(out / f"{tag}.progress.tsv", progress)
        record.update(result)
        tool_cpu = result["cpu_user_s"] + result["cpu_system_s"]
        driver_cpu = (_cpu_s(self1) - _cpu_s(self0)) + (_cpu_s(kids1) - _cpu_s(kids0)) - tool_cpu
        record["cpu"] = cpu_account(sched0, sched1, tool_cpu, driver_cpu, result["wall_s"])
        params = json.loads(Path(entry["json"]).read_text())
        if entry["tool"] == "stock":
            record["steady"] = stock_steady(progress, entry["batch_size"])
        else:
            record["meta"] = fast_meta_summary(params)
        runs["runs"][tag] = record
        _write_json(runs_path, runs)
        if result["returncode"] != 0:
            logger.error("%s failed (exit %d); see %s", tag, result["returncode"], result["log"])
            return 1
    runs["setups"][-1]["finished_et"] = _now_et()
    _write_json(runs_path, runs)
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# report


def _cache_label(counts: dict) -> str:
    values = [v or 0 for v in counts.values()]
    if not values:
        return "-"
    if all(v == 0 for v in values):
        return "cold"
    return "warm" if all(v > 0 for v in values) else "mixed"


def run_row(rec: dict, run_no: int, condition: str) -> dict:
    """One TSV row from a run record (see COLUMNS); also the per-peak cost the projections use."""
    cpu, apps = rec["cpu"], rec["gpu_before"]
    util = rec.get("gpu_util") or []
    row = {
        "condition": condition,
        "run": run_no,
        "tag": rec["tag"],
        "tool": rec["tool"],
        "rep": rec["rep"],
        "cache": _cache_label(rec["cache_files_before"]),
        "head": "counts",
        "batch_size": rec["batch_size"],
        "wall_s": rec["wall_s"],
        "peak_rss_gb": rec["max_rss_bytes"] / 1e9,
        "peak_gpu_mib": rec["gpu_max_used_mib"],
        "gpu_util_mean_pct": statistics.fmean(util) if util else None,
        "gpu_idle_pct": "NA",
        "co_tenants": len(apps),
        "co_tenant_mib": sum(a["used_mib"] or 0 for a in apps),
        "other_cpu_cores": cpu["other_cpu_cores"],
        "cpu_delay_cores": cpu["run_delay_cores"],
        "start_et": rec["started_et"],
        "end_et": rec["finished_et"],
        "exit": rec["returncode"],
    }
    if rec["tool"] == "stock":
        s, host = rec["steady"], rec["steady"]["host_clock"]
        peaks = host["n_end"] // rec["n_shuffles"]
        post_bar = rec["wall_s"] - host["t_end_s"]
        row.update(
            precision="float32",
            S="-",
            peaks=peaks,
            pairs=host["n_end"],
            steady_pairs_per_s=host["rate_ls"],
            steady_method=f"tqdm slope, n >= {SKIP_BATCHES} batches",
            startup_s=host["bar_start_s"],
            compile_s=host["first_batch_s"],
            self_check_s="-",
            write_s=post_bar,
            gpu_alloc_max_gb="-",
            per_peak_s=((s["extract_s"] or 0.0) + post_bar) / peaks,
        )
    else:
        m = rec["meta"]
        t = m["timings_s"]
        kept = m["inputs"]["kept"]
        row.update(
            precision=m["precision"],
            S=m["seqs_per_step"],
            peaks=kept,
            pairs=m["deltas"]["pairs"],
            steady_pairs_per_s=t["steady_pairs_per_s"],
            steady_method="meta: steps 2..last",
            startup_s=t["imports"] + t["load"] + t["extract"],
            compile_s=t["compile"],
            self_check_s=t["self_check"],
            write_s=t["write"],
            gpu_alloc_max_gb=m["memory"]["gpu_max_allocated_bytes"] / 1e9,
            per_peak_s=(t["extract"] + t["write"]) / kept,
        )
    return row


def _cell(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}" if abs(value) < 1e4 else f"{value:.0f}"
    return str(value)


def agreement(plan: dict, runs: dict) -> list[dict]:
    """Context, not a gate: per-peak rel-L2 and Pearson (float64, over 4 x W) between the runs' attributions, on the
    peaks both attributed (the stock runs' peaks are the first lines of ours), when the ohe files agree there."""
    import dls_validate as dv
    import numpy as np

    outputs = {}
    for entry in plan["runs"]:
        rec = runs["runs"].get(entry["tag"])
        if rec is None or rec["returncode"] != 0:
            continue
        params = json.loads(Path(entry["json"]).read_text())
        try:
            with np.load(params["ohe_filename"]) as f:
                ohe = f["arr_0"]
            with np.load(params["attr_filename"]) as f:
                attr = f["arr_0"]
        except OSError:
            continue
        outputs[entry["tag"]] = (entry["tool"], ohe, attr)
    stock = [t for t, v in outputs.items() if v[0] == "stock"]
    fast = [t for t, v in outputs.items() if v[0] == "fast"]
    pairs = [(a, b, "stock rerun") for a, b in zip(stock, stock[1:])]
    pairs += [(a, b, "ours rerun") for a, b in zip(fast, fast[1:])]
    pairs += [(f, stock[0], "ours vs stock") for f in fast] if stock else []
    rows = []
    for a, b, label in pairs:
        (_, ohe_a, attr_a), (_, ohe_b, attr_b) = outputs[a], outputs[b]
        n = min(len(attr_a), len(attr_b))
        row = {"label": label, "a": a, "b": b, "peaks": n, "ohe_equal": bool(np.array_equal(ohe_a[:n], ohe_b[:n]))}
        if row["ohe_equal"]:
            rel = dv.rel_l2(attr_a[:n], attr_b[:n], lead=1)
            pearson = dv.pearson_rows(attr_a[:n], attr_b[:n])
            row.update(
                rel_l2=dv.stats(rel),
                pearson_min=float(pearson.min()),
                pearson_median=float(np.median(pearson)),
                bitwise_peaks=int((rel == 0).sum()),
            )
        rows.append(row)
    return rows


def cmd_report(args: argparse.Namespace) -> int:
    out = Path(args.out_dir).resolve()
    plan = json.loads((out / "plan.json").read_text())
    runs = json.loads((out / "runs.json").read_text())
    rows, records = [], []
    for run_no, entry in enumerate(plan["runs"], 1):
        rec = runs["runs"].get(entry["tag"])
        if rec is None or rec["returncode"] != 0:
            continue
        rows.append(run_row(rec, run_no, args.condition))
        records.append(rec)
    if not rows:
        raise SystemExit(f"no successful runs in {out / 'runs.json'}")
    tsv = ["\t".join(COLUMNS)] + ["\t".join(_cell(r[c]) for c in COLUMNS) for r in rows]
    Path(args.tsv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.tsv).write_text("\n".join(tsv) + "\n")
    n_shuffles = plan["runs"][0]["n_shuffles"]
    for r in rows:
        r["projection"] = project(r, args.fold_peaks, args.models, n_shuffles) if r["steady_pairs_per_s"] else None
    speed = speedups(rows)
    agree = [] if args.no_agreement else agreement(plan, runs)
    notes = Path(args.notes).read_text() if args.notes else None
    report = {
        "generated_et": _now_et(),
        "condition": args.condition,
        "rows": rows,
        "speedup": speed,
        "g1_shared_pass": g1_shared(speed),
        "agreement": agree,
        "fold_peaks": args.fold_peaks,
        "models": args.models,
    }
    _write_json(Path(args.md).with_suffix(".json"), report)
    Path(args.md).write_text(markdown(report, records, runs, plan, notes))
    logger.info(
        "%s: stock %.1f, ours %.1f pairs/s (means); speed-up %.2fx (per rep %s); G1 shared %s; wrote %s and %s",
        args.condition,
        speed.get("stock_mean", float("nan")),
        speed.get("fast_mean", float("nan")),
        speed.get("ratio_of_means", float("nan")),
        ", ".join(f"{v:.2f}" for v in speed.get("per_rep", {}).values()),
        "PASS" if report["g1_shared_pass"] else "FAIL",
        args.tsv,
        args.md,
    )
    return 0


def _f(value, spec: str = ".3g") -> str:
    return "-" if value is None else format(value, spec)


def markdown(report: dict, records: list[dict], runs: dict, plan: dict, notes: str | None) -> str:
    rows, speed, cond = report["rows"], report["speedup"], report["condition"]
    setup = runs["setups"][-1]
    gpu = next((g for g in setup["gpus_at_start"] if g["uuid"] == setup["gpu_uuid"]), {})
    name = next((r["meta"]["gpu"]["name"] for r in records if r.get("meta")), "GPU")
    lines = [
        f"# {cond}: stock `cherimoya attribute` against attribute_fast.py (counts head)",
        "",
        f"Generated {report['generated_et']} (ET). {cond} is a **shared GPU**: GPU {setup['CUDA_VISIBLE_DEVICES']} "
        f"({name}), {gpu.get('processes', '?')} other processes and {gpu.get('memory_used_mib', '?')} MiB in use when "
        f"the runs started, cores {_cores(setup['cpus'])}. The runs alternated {', '.join(plan['order'])} in fresh "
        "processes, one at a time. **Every rate, speed-up and projection below holds for this condition only. None "
        "is a free-GPU number, and none may be compared with one.**",
        "",
    ]
    if speed:
        verdict = "PASS" if report["g1_shared_pass"] else "FAIL"
        lines += [
            "## Result",
            "",
            f"- Steady pairs/s, stock: {', '.join(_f(r['steady_pairs_per_s'], '.1f') for r in rows if r['tool'] == 'stock')}"
            f" (mean {speed['stock_mean']:.1f}); ours: "
            f"{', '.join(_f(r['steady_pairs_per_s'], '.0f') for r in rows if r['tool'] == 'fast')} (mean "
            f"{speed['fast_mean']:.0f}).",
            f"- Speed-up within {cond}: **{speed['ratio_of_means']:.2f}x** (ratio of the means); per repetition "
            f"{', '.join(f'{k}: {v:.2f}x' for k, v in speed['per_rep'].items())}; extremes {speed['min']:.2f}x "
            f"(slowest ours / fastest stock) to {speed['max']:.2f}x.",
            f"- G1, shared-GPU clause (ours >= {G1_SHARED:g}x stock under the same co-tenancy, A/B interleaved): "
            f"**{verdict}** (the ratio of the means and every repetition).",
            "",
        ]
    lines += [
        "## Runs",
        "",
        "| run | tool | cache | peaks | pairs | steady pairs/s | wall s | startup s | warm-up/compile s | self-check s |"
        " write s | peak RSS GB | peak GPU MiB | GPU co-tenants (MiB) | others' CPU on our cores | start - end (ET) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['run']} | {r['tool']} | {r['cache']} | {r['peaks']} | {r['pairs']} | "
            f"{_f(r['steady_pairs_per_s'], '.1f')} | {_f(r['wall_s'], '.1f')} | {_f(r['startup_s'], '.1f')} | "
            f"{_f(r['compile_s'], '.1f')} | {_cell(r['self_check_s'])} | {_f(r['write_s'], '.1f')} | "
            f"{_f(r['peak_rss_gb'], '.2f')} | {r['peak_gpu_mib']} | {r['co_tenants']} ({r['co_tenant_mib']}) | "
            f"{_f(r['other_cpu_cores'], '.2f')} cores | {r['start_et'][11:19]} - {r['end_et'][11:19]} |"
        )
    lines += [
        "",
        "- Steady pairs/s, stock: the least-squares slope of its tqdm pair counter against the arrival time of each "
        f"update, after the first {SKIP_BATCHES} batches. Ours: the meta sidecar's steady_pairs_per_s (steps 2 to "
        "last). Startup: stock's time to its first progress bar (imports, model load, extract_loci); ours' imports, "
        "load and extract. Warm-up/compile: stock's first batch (cuDNN and Triton autotuning; it compiles nothing); "
        "ours' first step (torch.compile). Write: stock's time after its last update; ours' write. Peak RSS: the "
        "largest single process (os.wait4). Peak GPU: the process's own memory in nvidia-smi.",
        "- Others' CPU: the CPU time that processes other than the run and this driver took on the run's cores, in "
        "cores' worth over the run (/proc/schedstat).",
        "- GPU idle %: not measurable under co-tenancy. nvidia-smi's utilization counts every process's kernels: "
        + ", ".join(f"run {r['run']} {_f(r['gpu_util_mean_pct'], '.0f')}%" for r in rows)
        + " on average.",
        "",
    ]
    checks = []
    for rec in records:
        if rec["tool"] == "stock":
            h, e = rec["steady"]["host_clock"], rec["steady"]["tqdm_elapsed"]
            checks.append(
                f"- {rec['tag']}: slope {_f(h['rate_ls'], '.2f')} pairs/s over {_f(h.get('window_s'), '.1f')} s and "
                f"{h.get('window_pairs')} pairs (two-point {_f(h['rate_two_point'], '.2f')}); on tqdm's own elapsed "
                f"field {_f(e['rate_ls'] if e else None, '.2f')}; first batch {_f(h['first_batch_s'], '.1f')} s."
            )
        else:
            m = rec["meta"]
            t, mem, rs = m["timings_s"], m["memory"], m["engine"]["reference_stats"] or {}
            checks.append(
                f"- {rec['tag']}: S = {m['seqs_per_step']}, {m['engine']['steps']} steps, {m['engine']['host_syncs']} "
                f"host syncs, precision {m['precision']}; run {_f(t['run'], '.1f')} s (compile {_f(t['compile'], '.1f')}"
                f" s); reference queue: {rs.get('stalls')} stalls, {_f(rs.get('wait_s'), '.1f')} s waited; GPU "
                f"allocator max {mem['gpu_max_allocated_bytes'] / 1e9:.2f} GB, reserved "
                f"{mem['gpu_max_reserved_bytes'] / 1e9:.2f} GB; deltas {m['deltas']['above']} of {m['deltas']['pairs']}"
                f" above {m['deltas']['threshold']}, max {_f(m['deltas']['max'], '.4g')}; self-check "
                f"{'passed' if m['self_check']['pass'] else 'FAILED'} ({_f(m['self_check']['max_abs'], '.3g')})."
            )
    lines += ["Rate checks and run details:", "", *checks, ""]
    lines += [
        f"## Projections ({cond} rates, shared GPU)",
        "",
        f"A fold is {report['fold_peaks']:,} peaks, {report['fold_peaks'] * plan['runs'][0]['n_shuffles']:,} pairs; "
        f"{report['models']} "
        "models are 39 samples x 5 folds. Steady: the fold's pairs at the measured steady rate. End to end: the run's "
        "measured wall time, extended at its steady rate by the extra pairs and at its per-peak extract and write "
        f"cost by the extra peaks. **These are {cond} (shared-GPU) numbers, under the co-tenancy recorded below. A "
        "free GPU is expected to be faster for both tools and is measured separately (C2).**",
        "",
        "| run | tool | steady pairs/s | min per fold (steady) | min per fold (end to end) | GPU-hours, "
        f"{report['models']} models (steady) | GPU-hours (end to end) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in (r for r in rows if r["projection"]):
        p = r["projection"]
        lines.append(
            f"| {r['run']} | {r['tool']} | {_f(r['steady_pairs_per_s'], '.1f')} | {p['steady_min']:.0f} | "
            f"{p['end_to_end_min']:.0f} | {p['steady_gpu_h']:.0f} | {p['end_to_end_gpu_h']:.0f} |"
        )
    for tool in TOOLS:
        sel = [r["projection"] for r in rows if r["tool"] == tool and r["projection"]]
        if sel:
            rates = [r["steady_pairs_per_s"] for r in rows if r["tool"] == tool and r["projection"]]
            lines.append(
                f"| mean | {tool} | {statistics.fmean(rates):.1f}"
                f" | {statistics.fmean(p['steady_min'] for p in sel):.0f} | "
                f"{statistics.fmean(p['end_to_end_min'] for p in sel):.0f} | "
                f"{statistics.fmean(p['steady_gpu_h'] for p in sel):.0f} | "
                f"{statistics.fmean(p['end_to_end_gpu_h'] for p in sel):.0f} |"
            )
    lines += ["", "## Co-tenancy", "", f"All GPUs when the runs started ({setup['started_et']}):", ""]
    lines += ["| GPU | MiB in use | utilization | processes |", "|---|---|---|---|"]
    for g in setup["gpus_at_start"]:
        mark = " (used)" if g["uuid"] == setup["gpu_uuid"] else ""
        lines.append(f"| {g['index']}{mark} | {g['memory_used_mib']} | {g['utilization_pct']}% | {g['processes']} |")
    lines += [""]
    for rec in records:
        before = ", ".join(f"{a['kind']} {a['pid']} ({a['used_mib']} MiB)" for a in rec["gpu_before"]) or "none"
        pids_before = {a["pid"] for a in rec["gpu_before"]}
        came = [p for p in rec.get("gpu_others_max_mib", {}) if int(p) not in pids_before]
        gone = sorted(pids_before - {a["pid"] for a in rec["gpu_after"]})
        c = rec["cpu"]
        lines.append(
            f"- {rec['tag']}: GPU co-tenants at the start: {before}. During the run, others' memory peaked at "
            f"{sum(rec.get('gpu_others_max_mib', {}).values())} MiB in total; new processes {came or 'none'}, gone "
            f"{gone or 'none'}. On cores {_cores(c['cpus'])}: the run used {_f(c['tool_cpu_cores'], '.2f')} cores' "
            f"worth, other processes {_f(c['other_cpu_cores'], '.2f')}, this driver {c['driver_cpu_s']:.1f} s; tasks "
            f"waited for a core {_f(c['run_delay_cores'], '.2f')} cores' worth (run-queue delay); the run was "
            f"preempted {rec['involuntary_switches']} times."
        )
    if report["agreement"]:
        lines += [
            "",
            "## Agreement (context, not a gate)",
            "",
            "Per-peak rel-L2 and Pearson r of the attributions, float64 over 4 x W, on the peaks both runs "
            "attributed. The stock runs' peaks are the first lines of ours, and their ohe files must agree there.",
            "",
            "| pair | runs | peaks | ohe equal | rel-L2 q50 | q99 | max | Pearson min |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for a in report["agreement"]:
            r = a.get("rel_l2") or {}
            lines.append(
                f"| {a['label']} | {a['a']} / {a['b']} | {a['peaks']} | {a['ohe_equal']} | {_f(r.get('q50'))} | "
                f"{_f(r.get('q99'))} | {_f(r.get('max'))} | {_f(a.get('pearson_min'), '.8f')} |"
            )
    lines += ["", "## Environment", "", "```", json.dumps(setup["env"], indent=2), "```", ""]
    lines += [
        "Per tool (added to the environment above):",
        "",
        *(f"- {e['tag']}: `{' '.join(e['command'])}` with {json.dumps(e['env'])}" for e in plan["runs"]),
        "",
    ]
    if notes:
        lines += [notes.rstrip(), ""]
    return "\n".join(lines)


def _cores(cpus: list[int]) -> str:
    if not cpus:
        return "-"
    spans, start, prev = [], cpus[0], cpus[0]
    for c in cpus[1:] + [None]:
        if c is not None and c == prev + 1:
            prev = c
            continue
        spans.append(f"{start}-{prev}" if prev > start else f"{start}")
        if c is not None:
            start = prev = c
    return ",".join(spans)


# ---------------------------------------------------------------------------------------------------------------------


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="write the runs' JSONs and plan.json")
    p.add_argument("-p", "--parameters", required=True, help="the base attribute.json")
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--order", default="stock,fast,stock,fast", help="tools in run order (default stock,fast,stock,fast)"
    )
    p.add_argument("--stock-peaks", type=int, required=True, help="stock attributes this many first lines of the loci")
    p.add_argument("--set", action="append", help="KEY=JSON, applied to every JSON (repeatable)")
    p.add_argument("--fast", help='a JSON object, the "fast" block of every JSON')
    p.add_argument("--stock-env", action="append", help="KEY=VALUE for the stock runs' environment (repeatable)")
    p.add_argument("--fast-env", action="append", help="KEY=VALUE for our runs' environment (repeatable)")
    r = sub.add_parser("run", help="run the plan, one process at a time")
    r.add_argument("--out-dir", required=True)
    r.add_argument("--only", nargs="+", help="run these tags only")
    r.add_argument("--poll-s", type=float, default=2.0, help="nvidia-smi polling interval (default 2 s)")
    c = sub.add_parser("report", help="write the TSV and the Markdown report")
    c.add_argument("--out-dir", required=True)
    c.add_argument("--condition", required=True, help="the condition's name, e.g. C1")
    c.add_argument("--tsv", required=True)
    c.add_argument("--md", required=True, help="the Markdown report; a JSON copy goes next to it")
    c.add_argument("--fold-peaks", type=int, default=FOLD_PEAKS)
    c.add_argument("--models", type=int, default=MODELS)
    c.add_argument("--notes", help="a Markdown file appended to the report")
    c.add_argument("--no-agreement", action="store_true", help="skip the outputs' agreement (needs numpy)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    return {"prepare": cmd_prepare, "run": cmd_run, "report": cmd_report}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
