#!/usr/bin/env python3
"""attribute_fast.py: `cherimoya attribute` for DeepLIFT/DeepSHAP on the counts head, computed by cheri_dls.

`cherimoya attribute -p attribute.json` runs tangermeme's hook-based deep_lift_shap. It forwards every sequence once
per reference and runs the backward over both halves of each batch, about 130 pairs/s on a shared H100. This script
reads the same JSON and writes the same three files. The attributions come from cheri_dls.Engine instead, whose
hand-written forward and backward compute the same multipliers several times faster (design
analyses/attr_pilot/design/final_design.md, §13-§14; implementation steps 3 and 7 of final_rest.json).

    python workflows/cherimoya/attribute_fast.py -p attribute.json [--precision reference|fp32] [--seqs-per-step N]
        [--mem-budget-gb GB] [--compile auto|default|none] [--ref-workers N] [--forward-stats auto|two_pass|...]
        [--deterministic] [--allow-version-drift]

Run it inside a box, never on the login node. The same JSON still runs under `cherimoya attribute` unchanged: the
stock command's merge_parameters passes unknown keys through, so the optional "fast" block is ignored there.

What it does, in the stock command's order (attribute.py below is cherimoya_cli/commands/attribute.py):
1. merge_parameters with the stock defaults. `skip` returns at once (attribute.py:21-23).
2. The stock algorithm check. saturation_mutagenesis is handed to the stock command unchanged.
3. Checks that need nothing loaded:
   - The "fast" options are resolved: the JSON's optional "fast" object, overridden by the flags above.
   - dtype must be float32 or null (the model's dtype, float32). bfloat16 and float16 are refused: tangermeme runs
     them under autocast, which computes a different function. float64 is accepted on the CPU only (tests).
   - output "profile" is refused for now: cheri_dls implements the counts head only.
   - The three output directories must exist (the stock command would fail on that only after attributing).
4. ``Cherimoya.load(model, device="cpu", compile=False)`` (cheri_dls.load_model); the engine copies the weights to
   the device, and the module itself moves there for the self-check. JSON `compile` is ignored, as the stock command
   ignores it for deep_lift_shap. Then the stock group check (attribute.py:42-47).
5. extract_loci with the stock's arguments, the N filter and the mask update (attribute.py:49-63).
6. The stock output check, the attr_window check and the centred window (attribute.py:66-86).
7. cheri_dls.Engine with the JSON's n_shuffles, random_state, warning_threshold, group and device. head_rows is
   2 x batch_size, the row count of the stock command's full batches; nothing else depends on batch_size.
8. The forward self-check (§14), always on. See forward_self_check(). A failure aborts before anything is written.
9. Engine.run: the hypothetical attributions over the window, averaged over each peak's references. One summary
   warning if any convergence delta exceeds warning_threshold (none for null).
10. The stock command's three writes, in its order: ``numpy.savez_compressed(ohe, X[:, :, start:end])`` from the same
   int8 torch slice, ``numpy.savez_compressed(attr, X_attr)`` with X_attr float32, ``numpy.save(idx, idxs)``. The idx
   file is written last, so it marks a finished run. Then two sidecars next to the attr file:
   ``<attr minus .npz>.deltas.npy``, the (n_kept, K) float32 convergence deltas, and ``<attr minus .npz>.meta.json``:
   versions, GPU, flags, S, K, precision, timings, peak GPU memory and RSS, a delta summary, the random_state used
   and the self-check result.

The "fast" block (all optional; a flag overrides the block, and the block overrides the default):

| key | default | |
|---|---|---|
| precision | "reference" | "reference": the TF32 GEMMs `cherimoya attribute` computes with; "fp32": strict IEEE fp32 |
| seqs_per_step | "auto" | peaks per step; "auto" sizes steps from free GPU memory and mem_budget_gb |
| mem_budget_gb | 12.0 | the GPU memory "auto" plans a step for, in GB of 1e9 bytes |
| compile | "auto" | "default" (torch.compile) on CUDA, "none" (eager) on the CPU |
| ref_workers | null | reference worker processes; null for min(8, CPUs - 2), 0 to shuffle in this process |
| forward_stats | "auto" | "two_pass" on CUDA, "cpu_mirror" on the CPU; "one_pass" is diagnostic |
| deterministic | false | bitwise reruns (cuDNN deterministic, no benchmark, deterministic cuBLAS and Inductor) |
| allow_version_drift | false | run with cherimoya/tangermeme versions other than the ones cheri_dls was checked against |

Keys the design names for later rounds (audit, resume, part files, the tangermeme engine, conv_class, other sidecar
names) are refused rather than ignored, and so is any unknown key.

Deliberate differences from the stock command:
- The forward self-check gates in strict fp32. Both sides, our eager forward and the stock module through
  LogCountWrapper(ControlWrapper(model), group) under torch.enable_grad() (the Triton training path), run under
  cheri_dls.precision_mode("fp32"), and the run aborts if they differ by more than 2.5e-4 on any of the first 4 peaks.
  2.5e-4 is cherimoya's documented fp32 cross-path bound (cherimoya/cheri.py:30-34), and in strict fp32 the two
  agree to a few 1e-6 (validation E1). In the default reference precision (TF32 GEMMs) the stock module disagrees
  with itself by up to 4.5e-4 between reruns, and by up to 5.7e-4 between identical rows of one batch, so a 2.5e-4
  gate there would abort valid runs at random: on the pilot's first 64 peaks the run's own outputs differed from a
  4-row forward of the module by 6.5e-4, while the strict-fp32 gate measured 9.5e-7. The comparison in the run's
  own precision, and the run's own counts outputs against the module, are recorded in the meta and logged above
  2.5e-4 (as a warning in fp32 precision), but do not gate.
- A stale idx file (the done marker of an earlier run) is removed before the new ohe file is written, so the marker
  never sits next to a partial set of new files.
- output "profile" and dtypes other than float32 are refused, and the output directories are checked, before the
  model is loaded.

NUMBA_CACHE_DIR: importing this module exports a node-local default, $TMPDIR/numba_cache or else
/dev/shm/$USER/numba_cache, if it is unset. tangermeme compiles its numba shuffle kernel with cache=True when it is
imported, which fixes the cache location then; unset, numba would write into the installed environment. So the
variable must be set before cherimoya or cheri_dls is imported, and this module imports both only inside run().
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import logging
import numbers
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))  # cheri_dls, dls_refs and dls_bench are flat modules next to this one

logger = logging.getLogger("attribute_fast")

SELF_CHECK_PEAKS = 4
SELF_CHECK_MAX_ABS = 2.5e-4
"""§14: abort if our counts output and the stock module's differ by more than this on the first 4 peaks."""

FAST_DEFAULTS = {
    "precision": "reference",
    "seqs_per_step": "auto",
    "mem_budget_gb": 12.0,
    "compile": "auto",
    "ref_workers": None,
    "forward_stats": "auto",
    "deterministic": False,
    "allow_version_drift": False,
}
"""The "fast" options and their defaults (the Engine's own defaults)."""

FAST_LATER = {
    "engine": "dls",
    "self_check": True,
    "audit": None,
    "resume": None,
    "part_peaks": None,
    "chunk_peaks": None,
    "conv_class": "inherit",
    "deltas_filename": None,
    "meta_filename": None,
}
"""Keys the design names for later rounds, with the one value each accepts now (the behaviour of this version)."""

PRECISIONS = ("reference", "fp32")
COMPILE_MODES = ("auto", "default", "none")
FORWARD_STATS = ("auto", "two_pass", "one_pass", "cpu_mirror")
CODE_FILES = ("attribute_fast.py", "cheri_dls.py", "dls_refs.py")


def _user() -> str:
    try:
        return getpass.getuser()
    except (KeyError, OSError):
        return str(os.getuid())


def default_numba_cache_dir() -> str:
    """The node-local default for NUMBA_CACHE_DIR: $TMPDIR/numba_cache, or /dev/shm/$USER/numba_cache."""
    base = os.environ.get("TMPDIR") or os.path.join("/dev/shm", _user())
    return os.path.join(base, "numba_cache")


def ensure_numba_cache_dir() -> str:
    """Export NUMBA_CACHE_DIR=default_numba_cache_dir() if it is unset, create the directory, and return it.

    It must run before cherimoya, tangermeme or cheri_dls is imported (see the module doc); this module calls it when
    it is imported. A variable that is already set is kept.
    """
    path = os.environ.get("NUMBA_CACHE_DIR")
    if not path:
        path = default_numba_cache_dir()
        os.environ["NUMBA_CACHE_DIR"] = path
    os.makedirs(path, exist_ok=True)
    return path


ensure_numba_cache_dir()


# ---------------------------------------------------------------------------------------------------------------------
# options and checks that need nothing loaded


def fast_options(block: object, args: argparse.Namespace | None = None) -> dict:
    """Resolve the "fast" options: FAST_DEFAULTS, then the JSON's "fast" object, then the command-line flags.

    A null value in the block, or a flag left out, keeps the value before it. Unknown keys raise ValueError, and so
    do keys of later rounds (FAST_LATER) set to anything but the behaviour of this version.
    """
    if block is None:
        block = {}
    if not isinstance(block, dict):
        raise ValueError(f'"fast" must be a JSON object, got {type(block).__name__}')
    options = dict(FAST_DEFAULTS)
    for key, value in block.items():
        if key in FAST_DEFAULTS:
            if value is not None:
                options[key] = value
        elif key in FAST_LATER:
            if value is not None and value != FAST_LATER[key]:
                raise ValueError(
                    f'"fast.{key}" is not implemented in this version of attribute_fast.py '
                    f"(only {FAST_LATER[key]!r} or null is accepted, got {value!r})"
                )
        else:
            raise ValueError(f'unknown key "fast.{key}"; the "fast" options are {sorted(FAST_DEFAULTS)}')
    for key in FAST_DEFAULTS:
        value = getattr(args, key, None)
        if value is not None:
            options[key] = value
    _check_fast(options)
    return options


def _check_fast(options: dict) -> None:
    """Type and range checks of the resolved options, before anything is loaded. The Engine checks them again."""

    def bad(key: str, want: str) -> ValueError:
        return ValueError(f'"fast.{key}" must be {want}, got {options[key]!r}')

    if options["precision"] not in PRECISIONS:
        raise bad("precision", f"one of {PRECISIONS}")
    S = options["seqs_per_step"]
    if S != "auto" and (isinstance(S, bool) or not isinstance(S, numbers.Integral) or S < 1):
        raise bad("seqs_per_step", '"auto" or a positive integer')
    budget = options["mem_budget_gb"]
    if isinstance(budget, bool) or not isinstance(budget, numbers.Real) or not budget > 0:
        raise bad("mem_budget_gb", "a positive number")
    if options["compile"] not in COMPILE_MODES:
        raise bad("compile", f"one of {COMPILE_MODES}")
    workers = options["ref_workers"]
    if workers is not None and (isinstance(workers, bool) or not isinstance(workers, numbers.Integral) or workers < 0):
        raise bad("ref_workers", "null or a non-negative integer")
    if options["forward_stats"] not in FORWARD_STATS:
        raise bad("forward_stats", f"one of {FORWARD_STATS}")
    for key in ("deterministic", "allow_version_drift"):
        if not isinstance(options[key], bool):
            raise bad(key, "true or false")


def engine_dtype(dtype: object, device: object) -> str:
    """The dtype the engine runs in, "float32" or "float64", from the JSON's dtype and device; raise if refused.

    null means the model's dtype, float32. bfloat16 and float16 are refused: tangermeme runs them under autocast
    (tangermeme/deep_lift_shap.py:719-735), a different function from the one cheri_dls computes. float64 runs on the
    CPU only, for tests.
    """
    if dtype is None or dtype == "float32":
        return "float32"
    if dtype in ("bfloat16", "float16", "half"):
        raise ValueError(
            f"dtype {dtype!r} is not supported by attribute_fast.py: tangermeme runs it under autocast, which computes "
            "a different function from the engine's. Use float32 (or null), or run `cherimoya attribute`."
        )
    if dtype == "float64":
        if str(device).split(":")[0] != "cpu":
            raise ValueError(f"dtype float64 runs on the CPU only (tests), not on device {device!r}; use float32")
        return "float64"
    raise ValueError(f"dtype must be float32 or null, got {dtype!r}")


def _npy_path(name: str) -> str:
    """The file numpy.save writes for `name`: it appends .npy unless the name already ends with it."""
    return name if name.endswith(".npy") else name + ".npy"


def sidecar_base(attr_filename: str) -> str:
    """<attr minus .npz>: numpy.savez_compressed appends .npz to a name without it, so this is the file's stem."""
    return attr_filename[: -len(".npz")] if attr_filename.endswith(".npz") else attr_filename


def _check_output_dirs(parameters: dict) -> None:
    for key in ("ohe_filename", "attr_filename", "idx_filename"):
        directory = os.path.dirname(os.path.abspath(parameters[key]))
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"{key}: the directory {directory} does not exist")


# ---------------------------------------------------------------------------------------------------------------------
# the forward self-check (§14)


def _module_counts(wrapper, X, precision: str):
    """The stock module's attributed output: wrapper(X)[:, 0] under torch.enable_grad() (the Triton training path on
    CUDA; no_grad would pick the bf16 inference kernel), in `precision`'s flags, as float64 on the CPU."""
    import cheri_dls
    import torch

    with cheri_dls.precision_mode(precision, inductor_cache=False), torch.enable_grad():
        return wrapper(X)[:, 0].detach().cpu().double()


def _our_counts(engine, X, precision: str):
    """Our attributed output: cheri_dls.model_forward, eager, with the engine's weights and statistics."""
    import cheri_dls
    import torch

    with cheri_dls.precision_mode(precision, inductor_cache=False), torch.no_grad():
        out = cheri_dls.model_forward(
            X, engine.weights, forward_stats=engine.forward_stats, output=engine.output, group=engine.group
        )
    return out.y.detach().cpu().double()


def _max_abs(a, b) -> float:
    return float((a - b).abs().max())


def forward_self_check(engine, model, X, *, peaks: int = SELF_CHECK_PEAKS, bound: float = SELF_CHECK_MAX_ABS) -> dict:
    """§14's forward self-check on the first `peaks` peaks of X; returns a record, with "pass", for the meta.

    The stock module is ``LogCountWrapper(ControlWrapper(model), group)``, built by cheri_dls.make_wrapper with
    cherimoya's own classes, in eval mode on the engine's device. `model` is the float32 Cherimoya. It is run under
    torch.enable_grad(), as deep_lift_shap runs it. Ours is cheri_dls.model_forward on the engine's weights, eager.
    - Gate: both in strict fp32 (precision_mode("fp32")); "pass" is max |ours - module| <= bound (NaN fails).
    - Recorded, not gated, when the engine's precision is not fp32: the same comparison in the engine's precision.
      In "reference" precision the module's own reruns differ by up to 4.5e-4 (see the module doc).
    The module's outputs in the engine's precision are kept (module_y_run), so that after the run its own outputs
    can be compared with them (compare_run_outputs).
    """
    import torch

    began = time.perf_counter()
    n = min(int(peaks), len(X))
    device = engine.weights.device
    X4 = torch.as_tensor(X[:n]).to(device=device, dtype=torch.float32)
    import cheri_dls

    wrapper = cheri_dls.make_wrapper(model, "counts", engine.group).to(device).eval()
    record: dict = {"peaks": n, "bound": bound, "gate_precision": "fp32", "comparisons": {}}
    modes = ["fp32"] + ([engine.precision] if engine.precision != "fp32" else [])
    for mode in modes:
        module_y = _module_counts(wrapper, X4, mode)
        ours_y = _our_counts(engine, X4, mode)
        record["comparisons"][mode] = {
            "max_abs": _max_abs(ours_y, module_y),
            "module_y": module_y.tolist(),
            "ours_y": ours_y.tolist(),
        }
        if mode == engine.precision:
            record["module_y_run"] = module_y.tolist()
    gate = record["comparisons"]["fp32"]["max_abs"]
    record["max_abs"] = gate
    record["pass"] = bool(gate <= bound)
    record["seconds"] = time.perf_counter() - began
    for mode in modes[1:]:
        value = record["comparisons"][mode]["max_abs"]
        if not value <= bound:
            logger.info(
                "self-check, %s precision (recorded, not gated): our counts output differs from the stock module's by "
                "%.3g > %g on %d peaks. The stock module's own reruns differ by up to 4.5e-4 in this precision; the "
                "gated strict-fp32 comparison gave %.3g.",
                mode,
                value,
                bound,
                n,
                gate,
            )
    return record


def compare_run_outputs(check: dict, y_x, precision: str, bound: float = SELF_CHECK_MAX_ABS) -> dict:
    """The run's own counts outputs for the self-check's peaks against the module's, in the run's precision (recorded,
    not gated).

    Above `bound` this logs a warning in fp32 precision, where the two agree to about 1e-5 (validation E1), and an info
    line in reference precision, where the module's own TF32 noise is larger than the bound: its outputs depend on the
    batch shape it runs at (4 rows here, S(K + 1) rows in the run), because cuDNN and cuBLAS pick kernels per shape.
    """
    import torch

    n = check["peaks"]
    module_y = torch.tensor(check["module_y_run"], dtype=torch.float64)
    run_y = y_x[:n].detach().cpu().double()
    value = _max_abs(run_y, module_y)
    if not value <= bound:
        (logger.warning if precision == "fp32" else logger.info)(
            "the run's counts outputs differ from the stock module's by %.3g > %g on the self-check's %d peaks, in %s "
            "precision (recorded, not gated)",
            value,
            bound,
            n,
            precision,
        )
    return {"precision": precision, "max_abs": value, "run_y": run_y.tolist()}


# ---------------------------------------------------------------------------------------------------------------------
# the record


def _now_et() -> str:
    from dls_bench import _now_et as now_et

    return now_et()


def _versions() -> dict:
    import importlib.metadata as metadata

    import torch

    out = {"python": sys.version.split()[0], "torch.version.cuda": torch.version.cuda}
    try:
        out["cudnn"] = torch.backends.cudnn.version()
    except Exception as exc:  # recorded, not fatal
        out["cudnn"] = f"<{type(exc).__name__}: {exc}>"
    for package in ("torch", "cherimoya", "tangermeme", "numpy", "numba", "triton", "pyfaidx"):
        try:
            out[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            out[package] = None
    return out


def _code() -> dict:
    """The code that ran: the git commit of this checkout, whether these files differ from it, and their sha256."""
    out: dict = {"files_sha256": {}}
    for name in CODE_FILES:
        try:
            out["files_sha256"][name] = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        except OSError:
            out["files_sha256"][name] = None
    try:
        out["git_commit"] = subprocess.run(
            ["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=30
        ).stdout.strip()
        changed = subprocess.run(
            ["git", "-C", str(HERE), "status", "--porcelain", "--", *CODE_FILES],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
        out["git_dirty"] = bool(changed)
    except (OSError, subprocess.SubprocessError):
        out["git_commit"], out["git_dirty"] = None, None
    return out


def _device_record(device) -> dict:
    import torch

    dev = torch.device(device)
    out: dict = {"device": str(dev), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}
    if dev.type != "cuda":
        return out
    index = dev.index if dev.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    out.update(name=props.name, total_memory_bytes=props.total_memory, capability=f"{props.major}.{props.minor}")
    try:
        out["driver"] = (
            subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True,
                timeout=60,
            )
            .stdout.splitlines()[0]
            .strip()
        )
    except (OSError, subprocess.SubprocessError, IndexError):
        out["driver"] = None
    from dls_bench import _compute_apps

    out["co_tenants"] = _compute_apps()
    return out


def _rss() -> dict:
    import resource

    return {
        "rss_max_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "children_rss_max_bytes": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024,
    }


def _write_json(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as out:
        out.write(json.dumps(obj, indent=2, default=str) + "\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------------------------------------------------
# the command


def run(args: argparse.Namespace) -> dict | None:
    """Attribute the JSON's loci as `cherimoya attribute` does, with cheri_dls; see the module doc for the order.

    args.parameters is the JSON's path (or an already parsed dict, as the stock command accepts); the other attributes
    of args are the flags, and any that are missing or None leave the "fast" block's value. Returns the meta record,
    or None when `skip` is set or the stock command ran (saturation_mutagenesis).
    """
    from cherimoya_cli.defaults import default_attribute_parameters
    from cherimoya_cli.utils import merge_parameters

    began = time.perf_counter()
    started_et = _now_et()
    parameters_file = args.parameters if isinstance(args.parameters, str) else None
    parameters = merge_parameters(args.parameters, default_attribute_parameters)
    if parameters["skip"]:
        return None

    algorithm = parameters["algorithm"]
    if algorithm not in ("deep_lift_shap", "saturation_mutagenesis"):
        raise ValueError(f"algorithm must be either `deep_lift_shap` or `saturation_mutagenesis`, got {algorithm!r}")
    if algorithm == "saturation_mutagenesis":
        from cherimoya_cli.commands import attribute as stock_attribute

        logger.info("algorithm saturation_mutagenesis: running the stock `cherimoya attribute` unchanged")
        return stock_attribute.run(args)

    options = fast_options(parameters.get("fast"), args)
    dtype = engine_dtype(parameters["dtype"], parameters["device"])
    if parameters["output"] == "profile":
        raise NotImplementedError(
            "attribute_fast.py attributes the counts head only; output 'profile' is not implemented in cheri_dls "
            "yet. Run `cherimoya attribute -p <json>` for profile attributions."
        )
    _check_output_dirs(parameters)

    import cheri_dls  # imports cherimoya and tangermeme, after NUMBA_CACHE_DIR is set (module doc)
    import numpy
    import torch
    from tangermeme.io import extract_loci

    timings: dict = {"imports": time.perf_counter() - began}

    # attribute.py:30-47: the model, loaded on the CPU without compiling; then the group check
    t = time.perf_counter()
    model = cheri_dls.load_model(parameters["model"], allow_version_drift=options["allow_version_drift"])
    group = parameters["group"]
    if parameters["output"] == "counts" and group is None and len(model.signal_groups) > 1:
        raise ValueError(
            "deep_lift_shap attributes one output; set `group` "
            f"to one of the model's {len(model.signal_groups)} signal groups"
        )
    timings["load"] = time.perf_counter() - t

    # attribute.py:49-63
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
        verbose=parameters["verbose"],
    )
    n_extracted = len(X)
    n_idxs = X.sum(dim=(1, 2)) == X.shape[-1]
    X = X[n_idxs]
    idxs[idxs.clone()] = n_idxs
    timings["extract"] = time.perf_counter() - t

    # attribute.py:65-86
    if parameters["output"] not in ("counts", "profile"):
        raise ValueError("output must be either `counts` or `profile`.")
    attr_window = parameters["attr_window"]
    if attr_window > X.shape[-1]:
        raise ValueError(
            f"attr_window ({attr_window}) is wider than the extracted sequence ({X.shape[-1]}); "
            "lower attr_window or raise in_window"
        )
    mid = X.shape[-1] // 2
    start = mid - attr_window // 2
    end = start + attr_window

    engine_model = model if dtype == "float32" else _float64_copy(model)
    engine = cheri_dls.Engine(
        engine_model,
        output=parameters["output"],
        group=group,
        device=parameters["device"],
        precision=options["precision"],
        n_shuffles=parameters["n_shuffles"],
        random_state=parameters["random_state"],
        head_rows=2 * parameters["batch_size"],
        seqs_per_step=options["seqs_per_step"],
        mem_budget_gb=options["mem_budget_gb"],
        compile=options["compile"],
        forward_stats=options["forward_stats"],
        warning_threshold=parameters["warning_threshold"],
        ref_workers=options["ref_workers"],
        deterministic=options["deterministic"],
        allow_version_drift=options["allow_version_drift"],
    )
    device = engine.weights.device
    cuda = device.type == "cuda"

    t = time.perf_counter()
    check = forward_self_check(engine, model.to(device), X)
    timings["self_check"] = time.perf_counter() - t
    if not check["pass"]:
        raise RuntimeError(
            f"forward self-check failed: on the first {check['peaks']} peaks, in strict fp32, our counts output differs "
            f"from the stock module's by {check['max_abs']:.4g}, above {check['bound']:g}. Nothing was written."
        )
    logger.info(
        "forward self-check passed: max |ours - module| %.3g (strict fp32, %d peaks, bound %g)%s",
        check["max_abs"],
        check["peaks"],
        check["bound"],
        "".join(
            f"; {mode} precision {c['max_abs']:.3g} (recorded)"
            for mode, c in check["comparisons"].items()
            if mode != "fp32"
        ),
    )
    gpu_before_run = torch.cuda.max_memory_allocated(device) if cuda else None

    t = time.perf_counter()
    result = engine.run(X, start, end)
    timings["run"] = time.perf_counter() - t
    check["run_outputs"] = compare_run_outputs(check, result.y_x, engine.precision)
    X_attr = result.attr.float()

    # attribute.py:115-117, then the sidecars
    t = time.perf_counter()
    idx_file = _npy_path(parameters["idx_filename"])
    if os.path.exists(idx_file):
        logger.info("removing the done marker of an earlier run, %s, before writing", idx_file)
        os.remove(idx_file)
    numpy.savez_compressed(parameters["ohe_filename"], X[:, :, start:end])
    numpy.savez_compressed(parameters["attr_filename"], X_attr)
    numpy.save(parameters["idx_filename"], idxs)
    base = sidecar_base(parameters["attr_filename"])
    deltas_file, meta_file = base + ".deltas.npy", base + ".meta.json"
    numpy.save(deltas_file, result.deltas.float().numpy())
    timings["write"] = time.perf_counter() - t

    meta = result.meta
    deltas = dict(meta["deltas"])
    if deltas.get("above") is not None:
        deltas["fraction"] = deltas["above"] / deltas["pairs"]
    deltas["finite"] = bool(torch.isfinite(result.deltas).all())
    reference_stats = meta.get("reference_stats") or {}
    timings.update(
        compile=meta["compile_s"],
        warmup=meta["warmup_s"],
        refs=reference_stats.get("wait_s"),  # seconds the steps waited for the reference pool
        refs_stalls=reference_stats.get("stalls"),
        steady_pairs_per_s=meta["steady_pairs_per_s"],
        pairs_per_s=meta["pairs_per_s"],
    )
    timings["total"] = time.perf_counter() - began
    record = {
        "tool": "attribute_fast.py",
        "started_et": started_et,
        "finished_et": _now_et(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "argv": list(sys.argv),
        "parameters_file": parameters_file,
        "parameters": parameters,
        "fast": options,
        "dtype": dtype,
        "versions": _versions(),
        "code": _code(),
        "gpu": _device_record(device),
        "precision": engine.precision,
        "precision_flags": meta["precision_flags"],
        "seqs_per_step": meta["seqs_per_step"],
        "n_shuffles": meta["n_shuffles"],
        "head_rows": engine.head_rows,
        "compile": engine.compile,
        "forward_stats": engine.forward_stats,
        "deterministic": engine.deterministic,
        "random_state": engine.random_state,
        "random_state_drawn": parameters["random_state"] is None,
        "inputs": {
            "loci": int(len(idxs)),
            "extracted": int(n_extracted),
            "kept": int(len(X)),
            "length": int(X.shape[-1]),
            "window": [int(start), int(end)],
        },
        "engine": {
            key: meta.get(key)
            for key in ("steps", "padded_peaks", "host_syncs", "oom", "references", "reference_stats", "seconds")
        },
        "timings_s": timings,
        "memory": {
            "gpu_max_allocated_bytes_before_run": gpu_before_run,
            "gpu_max_allocated_bytes": torch.cuda.max_memory_allocated(device) if cuda else None,
            "gpu_max_reserved_bytes": torch.cuda.max_memory_reserved(device) if cuda else None,
            "step_peak_bytes": meta.get("step_peak_bytes"),
            "step_estimate_bytes": meta.get("step_estimate_bytes"),
            **_rss(),
        },
        "deltas": deltas,
        "self_check": check,
        "outputs": {
            "ohe": parameters["ohe_filename"],
            "attr": parameters["attr_filename"],
            "idx": parameters["idx_filename"],
            "deltas": deltas_file,
            "meta": meta_file,
            "attr_shape": list(X_attr.shape),
            "attr_finite": bool(torch.isfinite(X_attr).all()),
        },
    }
    _write_json(meta_file, record)
    logger.info(
        "wrote %s, %s and %s (%d of %d loci kept, window [%d, %d)); sidecars %s and %s. %d pairs in %.1f s "
        "(steady %s pairs/s, S=%d); total %.1f s",
        parameters["ohe_filename"],
        parameters["attr_filename"],
        parameters["idx_filename"],
        len(X),
        len(idxs),
        start,
        end,
        deltas_file,
        meta_file,
        meta["pairs"],
        timings["run"],
        "-" if meta["steady_pairs_per_s"] is None else f"{meta['steady_pairs_per_s']:.0f}",
        meta["seqs_per_step"],
        timings["total"],
    )
    return record


def _float64_copy(model):
    """A float64 copy of `model` for the engine (CPU tests). The engine never runs the module's forward; the stock
    Cherimoya 0.3.0 forward cannot run in float64 at all, so the self-check keeps the float32 original."""
    import copy

    return copy.deepcopy(model).double()


def _seqs_per_step(value: str) -> int | str:
    if value == "auto":
        return value
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected 'auto' or an integer, got {value!r}") from None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="attribute_fast.py",
        description="`cherimoya attribute` (DeepLIFT/DeepSHAP, counts head) computed by the cheri_dls engine.",
    )
    parser.add_argument("-p", "--parameters", required=True, help="the attribute.json of `cherimoya attribute`")
    parser.add_argument("--precision", choices=PRECISIONS, help="reference (default; TF32 GEMMs) or fp32")
    parser.add_argument("--seqs-per-step", dest="seqs_per_step", type=_seqs_per_step, help="peaks per step, or auto")
    parser.add_argument("--mem-budget-gb", dest="mem_budget_gb", type=float, help="GPU memory 'auto' plans for")
    parser.add_argument("--compile", choices=COMPILE_MODES, help="auto (default), default or none")
    parser.add_argument("--ref-workers", dest="ref_workers", type=int, help="reference worker processes (0: inline)")
    parser.add_argument("--forward-stats", dest="forward_stats", choices=FORWARD_STATS, help="normalization statistics")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, help="bitwise-reproducible reruns")
    parser.add_argument(
        "--allow-version-drift",
        dest="allow_version_drift",
        action=argparse.BooleanOptionalAction,
        help="run with other cherimoya/tangermeme versions than cheri_dls was checked against",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
