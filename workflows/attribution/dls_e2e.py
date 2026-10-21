#!/usr/bin/env python3
"""End to end: attribute_fast.py against the stock `cherimoya attribute` on one attribute.json (criterion E9).

E9 (final_rest.json) runs both commands on the same JSON and the same peaks, each in a fresh process, as users run
them. It asks for the ohe (int8) and idx (bool) files to be numpy.array_equal, and for the attributions (float32) to
meet E3 against the stock reference's measured hyp floors and E5's caps. The floors come from the GPU validation
harness (dls_validate.py gates, report.json). This script drives that comparison.

Subcommands, all run inside a box; only `run` needs a GPU:
- prepare: writes <out>/stock.json and <out>/fast.json from a base attribute.json. They are identical except for the
  three output paths, which point into <out>/stock/ and <out>/fast/. --set KEY=JSON applies the same override to
  both, e.g. batch_size=20. --fast JSON puts the same "fast" block into both; the stock command ignores it.
- run: the stock command, ``python -m cherimoya_cli attribute -p stock.json``, then
  ``python attribute_fast.py -p fast.json``. Each runs in a fresh process, one after the other, never concurrently,
  with this process's environment. For each it records the wall time, the exit status, the process's max RSS
  (os.wait4) and the most GPU memory nvidia-smi reports for the process (polled every --poll-s seconds) in
  <out>/runs.json. Their output goes to <out>/stock.log and <out>/fast.log. --only runs one side.
- compare: reads both sets of files and writes <out>/e9_report.json and <out>/e9_report.md:
  - ohe and idx: npz keys, dtypes, shapes, numpy.array_equal.
  - attr: dtype and shape; per-peak rel-L2 ||fast - stock|| / ||stock|| and Pearson over 4 x W, in float64;
    finiteness. E3 on the hyp metric: median <= max(2 F_rr(0.5), 1e-5), p99 <= max(3 F_rr(0.99), 1e-4), and the
    precision-policy check median <= 0.3 F_tf(0.5), with F_rr and F_tf from --floors (report.json). E5: every peak's
    Pearson >= 0.9999 and rel-L2 <= 1e-2, every value finite.
  - The stock run's printed warnings: deep_lift_shap warns once per batch whose convergence deltas exceed the
    threshold, printing that batch's deltas. The report counts the warnings and the printed deltas above the
    threshold (torch prints them with 4 significant digits), and reads E8's bounds against them as context.
  - Context: attribute_fast's meta sidecar (step size, timings, self-check, deltas), the wall times and peak memory
    of both runs, and, with --engine-attr, the attributions against an earlier engine run of the same peaks.
    --notes appends a Markdown file (deviations, notes) to the report.

    python workflows/attribution/dls_e2e.py prepare -p attribute.json --out-dir <e9> --set batch_size=20 \\
        --fast '{"mem_budget_gb": 7.5, "ref_workers": 3}'
    CUDA_VISIBLE_DEVICES=<GPU> NUMBA_CACHE_DIR=<node-local> TRITON_CACHE_DIR=<node-local> taskset -c <cores> \\
        python workflows/attribution/dls_e2e.py run --out-dir <e9>
    python workflows/attribution/dls_e2e.py compare --out-dir <e9> --floors <validation>/report.json
"""

from __future__ import annotations

import argparse
import codecs
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dls_bench import _now_et  # noqa: E402  (stdlib only at import)

logger = logging.getLogger("dls_e2e")

OUTPUTS = ("ohe_filename", "attr_filename", "idx_filename")
SIDES = ("stock", "fast")
ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "NUMBA_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "TORCHINDUCTOR_COMPILE_THREADS",
    "TRITON_CACHE_AUTOTUNING",
    "OMP_NUM_THREADS",
    "TMPDIR",
    "PYTORCH_CUDA_ALLOC_CONF",
)
WARNING_TEXT = "RuntimeWarning: Convergence deltas too high"
_TENSOR = re.compile(r"Convergence deltas too high: tensor\(\[(.*?)\]", re.S)
_FLOAT = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
_TQDM = re.compile(r"(?P<prefix>[^\r\n]*?)(?P<n>\d+)/(?P<total>\d+) \[(?P<stats>[^\]\r\n]*)\]")
_TQDM_PCT = re.compile(r"\s*\d+%\|.*$")


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    tmp.replace(path)


# ---------------------------------------------------------------------------------------------------------------------
# prepare


def cmd_prepare(args: argparse.Namespace) -> int:
    out = Path(args.out_dir).resolve()
    base = json.loads(Path(args.parameters).read_text())
    for item in args.set or []:
        key, sep, value = item.partition("=")
        if not key or not sep:
            raise SystemExit(f"--set takes KEY=JSON, got {item!r}")
        base[key] = json.loads(value)
    if args.fast is not None:
        base["fast"] = json.loads(args.fast)
    written = {}
    for side in SIDES:
        (out / side).mkdir(parents=True, exist_ok=True)
        params = dict(base)
        for key in OUTPUTS:
            params[key] = str(out / side / Path(base[key]).name)
        path = out / f"{side}.json"
        path.write_text(json.dumps(params, indent=4) + "\n")
        written[side] = params
    differing = sorted(k for k in written["stock"] if written["stock"][k] != written["fast"][k])
    assert differing == sorted(OUTPUTS), differing
    _write_json(
        out / "prepare.json",
        {
            "base": str(Path(args.parameters).resolve()),
            "set": args.set,
            "fast": base.get("fast"),
            "differ_only_in": differing,
            "prepared_et": _now_et(),
        },
    )
    logger.info("wrote %s and %s; they differ only in %s", out / "stock.json", out / "fast.json", differing)
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# run


def parse_compute_apps(rows: list[str]) -> list[tuple[str, int, int]]:
    """(gpu_uuid, pid, MiB) from `nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory` CSV rows (noheader,
    nounits); rows that do not parse, such as a memory of [N/A], are skipped."""
    apps = []
    for row in rows:
        parts = [p.strip() for p in row.split(",")]
        if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
            apps.append((parts[0], int(parts[1]), int(parts[2])))
    return apps


class GpuPoller(threading.Thread):
    """Polls nvidia-smi for the GPU memory one process uses; keeps the maximum (MiB) and the number of samples.

    With uuid (the UUID of the GPU the process runs on) it also keeps, for every other process on that GPU, the most
    memory it used during the polling (others: pid -> MiB), and the GPU's utilization.gpu samples (util, percent:
    the share of the sample period in which a kernel of any process was running).
    """

    def __init__(self, pid: int, interval: float, uuid: str | None = None) -> None:
        super().__init__(daemon=True)
        self.pid, self.interval, self.uuid = pid, interval, uuid
        self.max_mib, self.samples, self.errors = 0, 0, 0
        self.others: dict[int, int] = {}
        self.util: list[int] = []
        self._halt = threading.Event()  # not _stop: threading.Thread has a method of that name

    @staticmethod
    def _smi(query: list[str]) -> list[str]:
        return subprocess.run(
            ["nvidia-smi", *query, "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        ).stdout.splitlines()

    def run(self) -> None:
        while not self._halt.wait(self.interval):
            try:
                apps = parse_compute_apps(self._smi(["--query-compute-apps=gpu_uuid,pid,used_memory"]))
                util = self._smi(["--query-gpu=utilization.gpu", "-i", self.uuid]) if self.uuid else []
            except (OSError, subprocess.SubprocessError):
                self.errors += 1
                continue
            for gpu, pid, mib in apps:
                if pid == self.pid:
                    self.max_mib = max(self.max_mib, mib)
                    self.samples += 1
                elif self.uuid is not None and gpu == self.uuid:
                    self.others[pid] = max(self.others.get(pid, 0), mib)
            if util and util[0].strip().isdigit():
                self.util.append(int(util[0].strip()))

    def stop(self) -> None:
        self._halt.set()
        self.join(timeout=120)


def _elapsed_s(stats: str) -> float | None:
    """tqdm's elapsed time, the [MM:SS] or [H:MM:SS] before '<' in a bar's statistics, in seconds (1 s resolution)."""
    head = stats.split("<", 1)[0].strip()
    parts = head.split(":")
    if not 2 <= len(parts) <= 3 or not all(p.isdigit() for p in parts):
        return None
    seconds = 0
    for part in parts:
        seconds = 60 * seconds + int(part)
    return float(seconds)


class OutputTee(threading.Thread):
    """Copies a child's output pipe into its log, byte for byte, and dates every tqdm progress update in it.

    tqdm redraws its bar in place: each update is '\\r' plus the bar, written and flushed at once, so it arrives in
    one read of the pipe, and the clock at that read dates it. updates holds (t, desc, n, total, elapsed_s) for every
    complete bar in arrival order; desc is the bar's description ('' for an unnamed bar such as deep_lift_shap's)
    and elapsed_s tqdm's own elapsed field. feed() is the parser alone, for text that has already been read.
    """

    def __init__(self, pipe, log, clock: Callable[[], float] = time.perf_counter) -> None:
        super().__init__(daemon=True)
        self.pipe, self.log, self.clock = pipe, log, clock
        self.updates: list[tuple[float, str, int, int, float | None]] = []
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._buffer = ""

    def run(self) -> None:
        fd = self.pipe.fileno()
        while True:
            chunk = os.read(fd, 1 << 16)
            if not chunk:
                break
            t = self.clock()
            self.log.write(chunk)
            self.log.flush()
            self.feed(self._decoder.decode(chunk), t)
        self.feed(self._decoder.decode(b"", final=True), self.clock())

    def feed(self, text: str, t: float) -> None:
        self._buffer += text
        end = 0
        for match in _TQDM.finditer(self._buffer):
            desc = _TQDM_PCT.sub("", match.group("prefix")).strip().rstrip(":").strip()
            n, total = int(match.group("n")), int(match.group("total"))
            self.updates.append((t, desc, n, total, _elapsed_s(match.group("stats"))))
            end = match.end()
        rest = self._buffer[end:]
        cut = max(rest.rfind("\r"), rest.rfind("\n"))
        self._buffer = rest[cut:] if cut >= 0 else rest  # a bar never spans a line break: keep the open line only


def _run_one(
    side: str,
    command: list[str],
    log_path: Path,
    poll_s: float,
    *,
    env: dict[str, str] | None = None,
    progress: bool = False,
    uuid: str | None = None,
) -> dict:
    """Run `command` in a fresh process with its output in log_path; wall time, exit status, max RSS, GPU MiB.

    env replaces the environment (default: this process's). progress=True reads the output through a pipe
    (OutputTee) and records the tqdm updates, dated in seconds since the start, as `progress`. uuid also records the
    other processes on that GPU and its utilization (GpuPoller). CPU times and context switches come from os.wait4:
    the process and the children it waited for.
    """
    logger.info("%s: %s", side, " ".join(command))
    record = {"command": command, "log": str(log_path), "started_et": _now_et()}
    with open(log_path, "wb" if progress else "w") as log:
        began = time.perf_counter()
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE if progress else log,
            stderr=subprocess.STDOUT,
            cwd=str(HERE),
            env=env,
        )
        tee = None
        if progress:
            tee = OutputTee(proc.stdout, log, clock=lambda: time.perf_counter() - began)
            tee.start()
        poller = GpuPoller(proc.pid, poll_s, uuid=uuid)
        poller.start()
        _, status, usage = os.wait4(proc.pid, 0)
        wall = time.perf_counter() - began
        proc.returncode = os.waitstatus_to_exitcode(status)
        poller.stop()
        if tee is not None:
            tee.join(timeout=120)  # the pipe closes when the last holder of its write end exits
            if tee.is_alive():
                logger.warning("%s: output still open 120 s after the process exited", side)
            proc.stdout.close()
    record.update(
        finished_et=_now_et(),
        pid=proc.pid,
        returncode=proc.returncode,
        wall_s=wall,
        max_rss_bytes=usage.ru_maxrss * 1024,
        cpu_user_s=usage.ru_utime,
        cpu_system_s=usage.ru_stime,
        voluntary_switches=usage.ru_nvcsw,
        involuntary_switches=usage.ru_nivcsw,
        gpu_max_used_mib=poller.max_mib,
        gpu_samples=poller.samples,
        gpu_poll_errors=poller.errors,
        gpu_poll_s=poll_s,
    )
    if uuid is not None:
        record.update(gpu_others_max_mib={str(k): v for k, v in sorted(poller.others.items())}, gpu_util=poller.util)
    if tee is not None:
        record["progress"] = [list(u) for u in tee.updates]
    logger.info(
        "%s: exit %d after %.1f s; max RSS %.2f GB; GPU at most %d MiB (%d samples)",
        side,
        proc.returncode,
        wall,
        record["max_rss_bytes"] / 1e9,
        poller.max_mib,
        poller.samples,
    )
    return record


def cmd_run(args: argparse.Namespace) -> int:
    out = Path(args.out_dir).resolve()
    runs_path = out / "runs.json"
    runs = json.loads(runs_path.read_text()) if runs_path.exists() else {}
    runs["env"] = {k: os.environ.get(k) for k in ENV_KEYS}
    commands = {
        "stock": [sys.executable, "-m", "cherimoya_cli", "attribute", "-p", str(out / "stock.json")],
        "fast": [sys.executable, str(HERE / "attribute_fast.py"), "-p", str(out / "fast.json")],
    }
    for side in SIDES:
        if args.only and side != args.only:
            continue
        runs[side] = _run_one(side, commands[side], out / f"{side}.log", args.poll_s)
        _write_json(runs_path, runs)
        if runs[side]["returncode"] != 0:
            logger.error("%s failed; see %s", side, runs[side]["log"])
            return 1
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# compare


def stock_warnings(log_text: str, threshold: float) -> dict:
    """The stock run's per-batch convergence warnings: how many, and the printed deltas above the threshold."""
    values, fixed = [], 0
    for match in _TENSOR.finditer(log_text):
        body = match.group(1)
        numbers = [float(v) for v in _FLOAT.findall(body)]
        if "e" not in body.lower():
            fixed += 1
        values.extend(numbers)
    above = [v for v in values if v > threshold]
    return {
        "warnings": log_text.count(WARNING_TEXT),
        "tensors_parsed": len(_TENSOR.findall(log_text)),
        "tensors_in_fixed_notation": fixed,
        "printed_values": len(values),
        "printed_above_threshold": len(above),
        "printed_max": max(values) if values else None,
        "threshold": threshold,
    }


def _load_npz(path: str) -> tuple[list[str], object]:
    import numpy as np

    with np.load(path) as npz:
        return sorted(npz.files), npz["arr_0"]


def _row(name: str, measured, bound, ok: bool, rule: str) -> dict:
    return {"criterion": name, "measured": measured, "bound": bound, "pass": bool(ok), "rule": rule}


def cmd_compare(args: argparse.Namespace) -> int:
    import dls_validate as dv
    import numpy as np

    out = Path(args.out_dir).resolve()
    params = {side: json.loads((out / f"{side}.json").read_text()) for side in SIDES}
    runs = json.loads((out / "runs.json").read_text()) if (out / "runs.json").exists() else {}
    floors = json.loads(Path(args.floors).read_text())["floors"]
    report: dict = {"generated_et": _now_et(), "out_dir": str(out), "floors_file": str(Path(args.floors).resolve())}
    rows: list[dict] = []

    arrays: dict = {}
    for side in SIDES:
        p = params[side]
        ohe_keys, ohe = _load_npz(p["ohe_filename"])
        attr_keys, attr = _load_npz(p["attr_filename"])
        idx_file = p["idx_filename"] if p["idx_filename"].endswith(".npy") else p["idx_filename"] + ".npy"
        arrays[side] = {
            "ohe_keys": ohe_keys,
            "ohe": ohe,
            "attr_keys": attr_keys,
            "attr": attr,
            "idx": np.load(idx_file),
        }
    s, f = arrays["stock"], arrays["fast"]
    report["files"] = {
        side: {
            "ohe": {"keys": a["ohe_keys"], "dtype": str(a["ohe"].dtype), "shape": list(a["ohe"].shape)},
            "attr": {"keys": a["attr_keys"], "dtype": str(a["attr"].dtype), "shape": list(a["attr"].shape)},
            "idx": {"dtype": str(a["idx"].dtype), "shape": list(a["idx"].shape), "true": int(a["idx"].sum())},
        }
        for side, a in arrays.items()
    }
    keys_ok = s["ohe_keys"] == f["ohe_keys"] == s["attr_keys"] == f["attr_keys"] == ["arr_0"]
    rows.append(_row("npz keys", f"{f['ohe_keys']}/{f['attr_keys']}", "['arr_0']", keys_ok, "every npz holds arr_0"))
    ohe_ok = s["ohe"].dtype == f["ohe"].dtype == np.int8 and np.array_equal(s["ohe"], f["ohe"])
    rows.append(
        _row("ohe", f"{f['ohe'].dtype} {list(f['ohe'].shape)}", "int8, array_equal", ohe_ok, "numpy.array_equal")
    )
    idx_ok = s["idx"].dtype == f["idx"].dtype == np.bool_ and np.array_equal(s["idx"], f["idx"])
    rows.append(
        _row("idx", f"{f['idx'].dtype} {list(f['idx'].shape)}", "bool, array_equal", idx_ok, "numpy.array_equal")
    )
    shape_ok = f["attr"].dtype == s["attr"].dtype == np.float32 and f["attr"].shape == s["attr"].shape
    rows.append(_row("attr dtype/shape", f"{f['attr'].dtype} {list(f['attr'].shape)}", "float32, equal", shape_ok, ""))

    rel = dv.rel_l2(f["attr"], s["attr"], lead=1)
    pearson = dv.pearson_rows(f["attr"], s["attr"])
    finite = {side: int((~np.isfinite(a["attr"])).sum()) for side, a in arrays.items()}
    hyp = dv.stats(rel)
    report["attr"] = {
        "rel_l2": hyp,
        "pearson": {"min": float(pearson.min()), "q50": float(np.median(pearson))},
        "non_finite": finite,
        "bitwise_equal_peaks": int((rel == 0).sum()),
        "worst_peaks": [int(i) for i in np.argsort(rel)[::-1][:5]],
    }
    f_rr, f_tf = floors["F_rr"]["hyp"], floors["F_tf"]["hyp"]
    bounds = dv.floor_bounds(f_rr)
    policy = dv.POLICY_FACTOR * f_tf["q50"]
    report["E3_bounds"] = {"from": {"F_rr.hyp": f_rr, "F_tf.hyp": f_tf}, **bounds, "policy": policy}
    rows.append(_row("E3 hyp median", hyp["q50"], bounds["q50"], hyp["q50"] <= bounds["q50"], "<= max(2 F_rr, 1e-5)"))
    rows.append(_row("E3 hyp p99", hyp["q99"], bounds["q99"], hyp["q99"] <= bounds["q99"], "<= max(3 F_rr, 1e-4)"))
    rows.append(_row("E3 policy (hyp median)", hyp["q50"], policy, hyp["q50"] <= policy, "<= 0.3 F_tf(0.5)"))
    rows.append(
        _row(
            "E5 Pearson (min over peaks)",
            float(pearson.min()),
            dv.PEARSON_MIN,
            pearson.min() >= dv.PEARSON_MIN,
            ">= 0.9999 every peak",
        )
    )
    rows.append(_row("E5 rel-L2 (max over peaks)", hyp["max"], dv.HYP_REL_MAX, hyp["max"] <= dv.HYP_REL_MAX, "<= 1e-2"))
    rows.append(_row("E5 finite", finite, 0, not any(finite.values()), "every value finite"))

    if args.engine_attr:
        engine = np.load(args.engine_attr)
        er = dv.rel_l2(f["attr"], engine, lead=1)
        report["vs_engine_attr"] = {"file": args.engine_attr, "rel_l2": dv.stats(er), "bitwise": int((er == 0).sum())}

    threshold = params["stock"].get("warning_threshold", 1e-3)
    threshold = 1e-3 if threshold is None else float(threshold)
    log_path = out / "stock.log"
    if log_path.exists():
        report["stock_warnings"] = stock_warnings(log_path.read_text(errors="replace"), threshold)
    fast_meta_path = Path(params["fast"]["attr_filename"].removesuffix(".npz") + ".meta.json")
    fast_meta = json.loads(fast_meta_path.read_text()) if fast_meta_path.exists() else None
    if fast_meta is not None:
        report["fast_meta"] = {
            "file": str(fast_meta_path),
            "seqs_per_step": fast_meta["seqs_per_step"],
            "n_shuffles": fast_meta["n_shuffles"],
            "head_rows": fast_meta["head_rows"],
            "precision": fast_meta["precision"],
            "compile": fast_meta["compile"],
            "timings_s": fast_meta["timings_s"],
            "memory": fast_meta["memory"],
            "deltas": fast_meta["deltas"],
            "self_check": {
                "pass": fast_meta["self_check"]["pass"],
                "max_abs_fp32": fast_meta["self_check"]["max_abs"],
                "comparisons": {k: v["max_abs"] for k, v in fast_meta["self_check"]["comparisons"].items()},
                "run_outputs_max_abs": fast_meta["self_check"]["run_outputs"]["max_abs"],
                "seconds": fast_meta["self_check"]["seconds"],
            },
            "gpu": fast_meta["gpu"],
            "code": fast_meta["code"],
        }
        stock_w = report.get("stock_warnings")
        if stock_w and stock_w["printed_values"]:
            pairs = fast_meta["deltas"]["pairs"]
            e8 = dv.e8_bounds(stock_w["printed_above_threshold"], pairs, stock_w["printed_max"])
            ours_fraction = fast_meta["deltas"]["above"] / pairs
            report["E8_context"] = {
                "stock": {
                    "above": stock_w["printed_above_threshold"],
                    "pairs": pairs,
                    "max": stock_w["printed_max"],
                },
                "fast": {"above": fast_meta["deltas"]["above"], "max": fast_meta["deltas"]["max"]},
                "bounds": e8,
                "pass": bool(ours_fraction <= e8["fraction"] and fast_meta["deltas"]["max"] <= e8["max"]),
            }
    report["runs"] = runs
    report["criteria"] = rows
    report["pass"] = all(r["pass"] for r in rows)
    _write_json(out / "e9_report.json", report)
    notes = Path(args.notes).read_text() if args.notes else None
    (out / "e9_report.md").write_text(_markdown(report, params, notes))
    for r in rows:
        logger.info(
            "%-28s %-6s measured %s, bound %s",
            r["criterion"],
            "PASS" if r["pass"] else "FAIL",
            r["measured"],
            r["bound"],
        )
    logger.info("E9 %s; wrote %s", "PASS" if report["pass"] else "FAIL", out / "e9_report.md")
    return 0 if report["pass"] else 1


def _fmt(value) -> str:
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _gb(value) -> str:
    return "-" if value is None else f"{value / 1e9:.2f} GB"


def _markdown(report: dict, params: dict, notes: str | None = None) -> str:
    runs, w = report.get("runs", {}), report.get("stock_warnings")
    meta = report.get("fast_meta")
    lines = [
        f"# E9: attribute_fast.py against `cherimoya attribute` (counts, {report['attr']['rel_l2']['n']} peaks)",
        "",
        f"Generated {report['generated_et']} (ET). Verdict: **{'PASS' if report['pass'] else 'FAIL'}**.",
        "",
        "Both commands ran the same JSON in fresh processes, one after the other; the two JSONs differ only in their "
        "three output paths (`stock.json`, `fast.json`).",
        "",
        "## Criteria",
        "",
        "| criterion | measured | bound | rule | pass |",
        "|---|---|---|---|---|",
    ]
    for r in report["criteria"]:
        lines.append(
            f"| {r['criterion']} | {_fmt(r['measured'])} | {_fmt(r['bound'])} | {r['rule']} | "
            f"{'PASS' if r['pass'] else 'FAIL'} |"
        )
    a = report["attr"]
    lines += [
        "",
        f"attr per-peak rel-L2 (fast vs stock): q50 {a['rel_l2']['q50']:.3g}, q99 {a['rel_l2']['q99']:.3g}, "
        f"max {a['rel_l2']['max']:.3g} over {a['rel_l2']['n']} peaks; Pearson min {a['pearson']['min']:.8f}, "
        f"median {a['pearson']['q50']:.10f}; {a['bitwise_equal_peaks']} peaks bitwise equal. E3's bounds come from "
        f"the stock floors in `{report['floors_file']}` (F_rr hyp q50 {report['E3_bounds']['from']['F_rr.hyp']['q50']:.3g}, "
        f"q99 {report['E3_bounds']['from']['F_rr.hyp']['q99']:.3g}; F_tf hyp q50 "
        f"{report['E3_bounds']['from']['F_tf.hyp']['q50']:.3g}).",
    ]
    if "vs_engine_attr" in report:
        v = report["vs_engine_attr"]
        lines += [
            "",
            f"Context: against the validation harness's own engine run of these peaks (`{v['file']}`): rel-L2 q50 "
            f"{v['rel_l2']['q50']:.3g}, max {v['rel_l2']['max']:.3g}.",
        ]
    lines += ["", "## Runs", "", "| | stock `cherimoya attribute` | `attribute_fast.py` |", "|---|---|---|"]
    st, fa = runs.get("stock", {}), runs.get("fast", {})
    lines.append(f"| wall time | {_fmt(st.get('wall_s'))} s | {_fmt(fa.get('wall_s'))} s |")
    lines.append(f"| exit status | {st.get('returncode')} | {fa.get('returncode')} |")
    lines.append(f"| max RSS | {_gb(st.get('max_rss_bytes'))} | {_gb(fa.get('max_rss_bytes'))} |")
    lines.append(
        f"| GPU memory (nvidia-smi, max) | {st.get('gpu_max_used_mib')} MiB | {fa.get('gpu_max_used_mib')} MiB |"
    )
    lines.append(f"| started (ET) | {st.get('started_et')} | {fa.get('started_et')} |")
    if w:
        lines.append(
            f"| printed warnings | {w['warnings']} (`Convergence deltas too high`, one per batch) | 1 summary |"
        )
    if st.get("wall_s") and fa.get("wall_s"):
        lines += ["", f"Wall-time ratio stock / fast: {st['wall_s'] / fa['wall_s']:.2f}."]
    if w:
        lines += [
            "",
            f"Stock warnings: {w['warnings']} batches warned; their printed deltas hold {w['printed_values']} values, "
            f"{w['printed_above_threshold']} above {w['threshold']:g} (max {_fmt(w['printed_max'])}; torch prints 4 "
            f"significant digits; {w['tensors_in_fixed_notation']} tensors in fixed notation).",
        ]
    if meta:
        t, m, d, sc = meta["timings_s"], meta["memory"], meta["deltas"], meta["self_check"]
        lines += [
            "",
            "## attribute_fast.py (meta sidecar)",
            "",
            f"- S = {meta['seqs_per_step']} peaks per step, K = {meta['n_shuffles']}, head_rows {meta['head_rows']}, "
            f"precision {meta['precision']}, compile {meta['compile']}.",
            f"- Timings (s): imports {_fmt(t.get('imports'))}, load {_fmt(t.get('load'))}, extract "
            f"{_fmt(t.get('extract'))}, self-check {_fmt(t.get('self_check'))}, run {_fmt(t.get('run'))} (warm-up/"
            f"compile {_fmt(t.get('compile'))}, waits on references {_fmt(t.get('refs'))} s in "
            f"{t.get('refs_stalls')} stalls), write {_fmt(t.get('write'))}, total {_fmt(t.get('total'))}. Steady "
            f"{_fmt(t.get('steady_pairs_per_s'))} pairs/s; overall {_fmt(t.get('pairs_per_s'))} pairs/s.",
            f"- GPU memory: steps at most {_gb(m.get('step_peak_bytes'))} (estimate {_gb(m.get('step_estimate_bytes'))}"
            f"); process max allocated {_gb(m.get('gpu_max_allocated_bytes'))}, reserved "
            f"{_gb(m.get('gpu_max_reserved_bytes'))}; before the run {_gb(m.get('gpu_max_allocated_bytes_before_run'))}"
            f". Max RSS {_gb(m.get('rss_max_bytes'))}.",
            f"- Deltas: {d.get('above')} of {d.get('pairs')} pairs above {d.get('threshold')} (max {_fmt(d.get('max'))}, "
            f"p50 {_fmt(d.get('p50'))}, p99 {_fmt(d.get('p99'))}).",
            f"- Self-check: {'passed' if sc['pass'] else 'FAILED'}; strict fp32 max |dy| {sc['max_abs_fp32']:.3g} "
            f"(gate 2.5e-4); per precision {', '.join(f'{k} {v:.3g}' for k, v in sc['comparisons'].items())}; the "
            f"run's own outputs vs the module {sc['run_outputs_max_abs']:.3g} (recorded, not gated); "
            f"{sc['seconds']:.1f} s.",
            f"- GPU: {meta['gpu'].get('name')}, driver {meta['gpu'].get('driver')}, CUDA_VISIBLE_DEVICES "
            f"{meta['gpu'].get('CUDA_VISIBLE_DEVICES')}; code {meta['code'].get('git_commit')} "
            f"(dirty: {meta['code'].get('git_dirty')}).",
        ]
    if "E8_context" in report:
        e8 = report["E8_context"]
        lines += [
            "",
            f"E8 read against this stock run (context): stock {e8['stock']['above']} of {e8['stock']['pairs']} deltas "
            f"above the threshold, max {_fmt(e8['stock']['max'])}; fast {e8['fast']['above']}, max "
            f"{_fmt(e8['fast']['max'])}; bounds fraction <= {e8['bounds']['fraction']:.4g}, max <= "
            f"{e8['bounds']['max']:.4g}: {'PASS' if e8['pass'] else 'FAIL'}.",
        ]
    lines += ["", "## Environment", "", "```", json.dumps(runs.get("env", {}), indent=2), "```", ""]
    if notes:
        lines += [notes.rstrip(), ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------------------


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="write stock.json and fast.json")
    p.add_argument("-p", "--parameters", required=True, help="the base attribute.json")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--set", action="append", help="KEY=JSON, applied to both JSONs (repeatable)")
    p.add_argument("--fast", help='a JSON object, the "fast" block of both JSONs')
    r = sub.add_parser("run", help="the stock command, then attribute_fast.py, in fresh processes")
    r.add_argument("--out-dir", required=True)
    r.add_argument("--only", choices=SIDES, help="run one side only")
    r.add_argument("--poll-s", type=float, default=2.0, help="nvidia-smi polling interval (default 2 s)")
    c = sub.add_parser("compare", help="E9's checks; writes e9_report.json and e9_report.md")
    c.add_argument("--out-dir", required=True)
    c.add_argument("--floors", required=True, help="the validation harness's report.json (floors)")
    c.add_argument("--engine-attr", help="an earlier engine run's attributions (.npy), for context")
    c.add_argument("--notes", help="a Markdown file appended to e9_report.md (deviations, notes)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    return {"prepare": cmd_prepare, "run": cmd_run, "compare": cmd_compare}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
