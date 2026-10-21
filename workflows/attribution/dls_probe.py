#!/usr/bin/env python3
"""GPU probes P0-P5 for the cheri_dls DeepLIFT/DeepSHAP engine (final_design.md §15.2).

The fast engine is judged against `cherimoya attribute`, whose "float32" run is
shaped by process-global flags that `import cherimoya` sets (cherimoya.py:15-16:
cudnn.benchmark=True and float32 matmul precision 'high') and by library choices
the installed sources cannot settle: whether cuBLAS and cuDNN run each GEMM and
convolution in IEEE fp32 or in TF32, how TF32 rounds its operands, which kernel
PyTorch picks for the rule's depthwise convolution, what cuBLAS does with a K=1
product, and how far Inductor's GELU is from the eager kernel. Each answer
decides a branch of the design (§7.1, §8.1, §12, §17), so it is measured here,
on the GPU, before any parity gate is read:

  P0  environment: versions, GPU, driver, co-tenant processes, and every
      precision flag before and after `import cherimoya`
  P1  class of the block GEMMs, forward (F.linear) and input-grad (matmul with
      W), at 128 and 240 rows of 2114 positions
  P2  kernel and class of `cherimoya.cheri._cheri_conv`, forward and its
      autograd input-grad (the rule's conv, deep_lift_shap.py:143-148), for
      dilations 1..256, against an fp64 shift-add
  P3  cuDNN class of iconv (4->128, k21) and fconv (128->1, k75), forward and
      input-grad (`aten.convolution_backward`, input-only mask, real contiguous
      input), at the reference's 128 rows vs our 260, repeated in fresh
      processes (3 by default) because cudnn.benchmark picks an algorithm by
      timing in each process
  P4  whether cuBLAS returns W bit for bit from ones(M,1) @ W, the count head's
      K=1 backward GEMM, and the whole head chain of §7.1
  P5  ulp histogram of Inductor against eager GELU(tanh) and gelu_backward

A class is identified positively rather than by a threshold alone. The GPU's
fp32 result is compared with fp64 products of (a) the exact operands and of the
operands rounded to TF32 (10 explicit mantissa bits) with bit operations by
(b) round-to-nearest-even, (c) round-to-nearest-away (CUTLASS's half-ulp
truncate) and (d) truncation. The matching emulation lands at fp32 accumulation
error (~1e-7) and the others near the TF32 unit roundoff (~1e-4). Realistic
operands cannot tell (b) from (c), because they differ only on exact ties, so
each GEMM and convolution is also run on operands whose low 13 bits are all
0x1000 (exact ties), one operand at a time with the other TF32-exact. That
separates RNE from RNA and tells which operand the kernel rounds.

Each probe runs in its own fresh subprocess that imports cherimoya before
anything else, so the flags are exactly those a user of `cherimoya attribute`
gets. The driver never imports torch. Run it inside a box on one GPU:

    CUDA_VISIBLE_DEVICES=<least-loaded GPU> taskset -c <4 cores> \\
        python workflows/attribution/dls_probe.py --out-dir <dir> --model <cherimoya.fold_0.torch>

It writes <out-dir>/dls_probes.json (every measurement plus the conclusions),
<out-dir>/dls_probes.md (a short summary), and the per-probe raw JSON and logs
under <out-dir>/raw and <out-dir>/logs. The §12 conv_class verdict is given per
head: a convolution needs a pin when the reference's row count gives one class
in every process while ours gives another or flips between classes; nothing
can be pinned when the reference itself flips. `--probes` runs a subset,
`--p3-first-run` adds P3 samples next to earlier ones, and `--merge-only`
rebuilds the report from the raw files.
Each child caps its CUDA caching allocator (`--gpu-mem-gb`), sets
TORCHINDUCTOR_COMPILE_THREADS=1 so Inductor does not fork a compile pool, and
sets nothing that changes numerics.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
import logging
import math
import os
import subprocess
import sys
import time
import traceback
from collections import Counter
from functools import partial
from pathlib import Path

logger = logging.getLogger("dls_probe")

PROBES = ("P0", "P1", "P2", "P3", "P4", "P5")
TF32_MODES = ("rne", "rna", "rz")
CANDIDATES = ("fp32", "tf32_rne", "tf32_rna", "tf32_rz")
# Realistic operands cannot separate RNE from RNA (they differ only on exact ties), so those two form one family.
FAMILY = {"fp32": "fp32", "tf32_rne": "tf32_rn", "tf32_rna": "tf32_rn", "tf32_rz": "tf32_rz"}
IDENTITY = {c: c for c in CANDIDATES}
CLASS_MAX_ERR = 1e-5  # the matching candidate sits at fp32 accumulation error, far below this
CLASS_MIN_MARGIN = 10.0  # and is at least 10x closer than any candidate of another family
CHUNK_ROWS = 32  # fp64 references are computed this many rows at a time to bound memory
CONV_OPS = ("iconv_fwd", "iconv_dgrad", "fconv_fwd", "fconv_dgrad")
# The counts head's gradient passes through iconv only; fconv enters only the profile head (cherimoya.py:455-470).
HEAD_OPS = {"counts": ("iconv_fwd", "iconv_dgrad"), "profile": CONV_OPS}
# Layout transforms, scaling, fills and copies around the compute kernel, left out when naming "the" kernel.
HELPER_KERNELS = ("elementwise", "nchwtonhwc", "nhwctonchw", "scalepackedtensor", "memset", "memcpy")
ULP_BINS = ((0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 8), (9, 16), (17, 64), (65, 1024), (1025, None))
# Variables that change numerics or algorithm choice without anyone noticing; P0 records each (None = unset).
NUMERICS_ENV = (
    "NVIDIA_TF32_OVERRIDE",
    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
    "CUBLAS_WORKSPACE_CONFIG",
    "CUBLASLT_WORKSPACE_SIZE",
    "CUDNN_CONV_WSCAP_DBG",
    "TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT",
    "TORCH_CUDNN_V8_API_DISABLED",
    "TORCH_BLAS_PREFER_CUBLASLT",
    "PYTORCH_CUDA_ALLOC_CONF",
    "CUDA_LAUNCH_BLOCKING",
)
ENV_PREFIXES = ("NVIDIA_", "TORCH", "PYTORCH", "CUDNN", "CUBLAS", "CUDA_", "TRITON", "OMP_", "MKL_", "NUMBA_")


# ---------------------------------------------------------------------------------------------------------------------
# helpers shared by the children (torch is imported lazily so the driver never loads it)


def _now_et() -> str:
    try:
        from zoneinfo import ZoneInfo

        return dt.datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds")
    except Exception:  # no tz database: fall back to UTC, still unambiguous
        return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _ints(text: str) -> list[int]:
    return [int(t) for t in str(text).split(",") if t.strip()]


def _jsonable(obj):
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if obj is None or isinstance(obj, bool | int | str):
        return obj
    return str(obj)


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(_jsonable(obj), indent=1) + "\n")
    tmp.replace(path)


def _smi(query: str, kind: str = "gpu") -> list[list[str]]:
    flag = "--query-gpu" if kind == "gpu" else "--query-compute-apps"
    try:
        res = subprocess.run(
            ["nvidia-smi", f"{flag}={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("nvidia-smi failed: %s", exc)
        return []
    return [[f.strip() for f in line.split(",")] for line in res.stdout.splitlines() if line.strip()]


def _compute_apps(uuid: str | None = None) -> list[dict]:
    apps = []
    for row in _smi("gpu_uuid,pid,process_name,used_memory", kind="apps"):
        if len(row) < 4:
            continue
        if uuid and row[0].removeprefix("GPU-") != uuid.removeprefix("GPU-"):
            continue
        mem = int(row[3]) if row[3].isdigit() else row[3]
        apps.append({"gpu_uuid": row[0], "pid": int(row[1]), "process_name": row[2], "used_memory_mib": mem})
    return apps


def _import_torch_as_user(gpu_mem_gb: float):
    """Import cherimoya first, exactly as `cherimoya attribute` does, then cap this process's caching allocator."""
    import cherimoya  # noqa: F401  (cherimoya.py:15-16 set cudnn.benchmark and the matmul precision here)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device is visible; run inside a box with CUDA_VISIBLE_DEVICES set")
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, gpu_mem_gb * 2**30 / total), 0)
    return torch


def _gen(torch, seed: int):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    return gen


def _onehot(torch, rows: int, length: int, gen):
    idx = torch.randint(0, 4, (rows, length), generator=gen, device="cuda")
    return torch.nn.functional.one_hot(idx, 4).permute(0, 2, 1).to(torch.float32).contiguous()


def _weights(torch, args) -> dict:
    """The tensors the probes multiply: the model's when --model is given, else draws of the real shapes.

    The draws use the scales of the design's weight redraw (§15.1) so nothing is degenerate.
    """
    if args.model:
        from cherimoya import Cherimoya

        m = Cherimoya.load(str(args.model), device="cpu", compile=False)
        blocks = list(m.blocks)
        w = {
            "source": str(args.model),
            "W0": m.iconv.weight,
            "b0": m.iconv.bias,
            "pad0": int(m.iconv.padding[0]),
            "Wf": m.fconv.weight,
            "bf": m.fconv.bias,
            "padf": int(m.fconv.padding[0]),
            "Wl": m.linear.weight,
            "bl": m.linear.bias,
            "W1": blocks[0].linear1.weight,
            "W2": blocks[0].linear2.weight,
            "cw": [b.conv.conv_weight for b in blocks],
            "dilations": [int(b.dilation) for b in blocks],
            "T": int(m.trimming),
            "n_ctl": int(m.n_control_tracks),
            "C": int(m.n_filters),
            "H": int(m.expansion * m.n_filters),
        }
    else:
        g = torch.Generator(device="cpu")
        g.manual_seed(1234)

        def rn(*shape, std):
            return torch.randn(*shape, generator=g) * std

        C, H, n_out = 128, 256, 1
        w = {
            "source": "random (design §15.1 scales)",
            "W0": rn(C, 4, 21, std=0.5),
            "b0": rn(C, std=0.3),
            "pad0": 10,
            "Wf": rn(n_out, C, 75, std=0.1),
            "bf": rn(n_out, std=0.1),
            "padf": 37,
            "Wl": rn(1, C, std=1.0),
            "bl": rn(1, std=0.1),
            "W1": rn(H, C, std=1.5 / C**0.5),
            "W2": rn(C, H, std=1.0 / H**0.5),
            "cw": [rn(3, C, std=0.8) for _ in range(9)],
            "dilations": [2**i for i in range(9)],
            "T": 557,
            "n_ctl": 0,
            "C": C,
            "H": H,
        }
    for k, v in list(w.items()):
        if isinstance(v, torch.Tensor):
            w[k] = v.detach().to("cuda", torch.float32).contiguous()
        elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            w[k] = [t.detach().to("cuda", torch.float32).contiguous() for t in v]
    return w


def _cuda_kernels(torch, fn) -> list[str]:
    """Names of the CUDA kernels (and memcpy/memset) that one call of fn launches, in order, with repeat counts."""
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    try:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
    except Exception as exc:  # profiler unavailable: say so rather than lose the probe
        return [f"<profiler failed: {exc!r}>"]
    counts: dict[str, int] = {}
    for evt in sorted(prof.events(), key=lambda e: e.time_range.start):
        if evt.device_type == DeviceType.CUDA:
            counts[evt.name] = counts.get(evt.name, 0) + 1
    return [f"{n} [x{c}]" if c > 1 else n for n, c in counts.items()]


def _tf32(torch, x, mode: str | None):
    """x rounded to TF32 (fp32 with the low 13 mantissa bits cleared) by bit operations.

    rne: round to nearest, ties to even. rna: round to nearest, ties away from zero (add half an ulp, truncate).
    rz: truncation. The int32 view adds to the magnitude bits, so the same expression serves both signs.
    """
    if mode is None:
        return x
    b = x.contiguous().view(torch.int32)
    if mode == "rz":
        r = b & -0x2000
    elif mode == "rna":
        r = (b + 0x1000) & -0x2000
    elif mode == "rne":
        r = (b + (0x0FFF + ((b >> 13) & 1))) & -0x2000
    else:
        raise ValueError(f"unknown TF32 rounding {mode!r}")
    return r.view(torch.float32)


def _ties(torch, x):
    """x with its low 13 bits set to 0x1000, so every element lies exactly halfway between two TF32 values."""
    b = x.contiguous().view(torch.int32)
    return ((b & -0x2000) | 0x1000).view(torch.float32)


def _mode(candidate: str) -> str | None:
    return None if candidate == "fp32" else candidate.removeprefix("tf32_")


def _compare(torch, y, ref_fn, n_rows: int, chunk: int = CHUNK_ROWS) -> dict:
    """rel-L2 and max-abs error of the fp32 tensor y (rows first) against an fp64 reference built chunk by chunk."""
    num = den = 0.0
    max_abs = ref_max = 0.0
    for i in range(0, n_rows, chunk):
        j = min(n_rows, i + chunk)
        ref = ref_fn(i, j)
        d = y[i:j].double() - ref
        num += float(torch.sum(d * d))
        den += float(torch.sum(ref * ref))
        max_abs = max(max_abs, float(d.abs().max()))
        ref_max = max(ref_max, float(ref.abs().max()))
        del ref, d
    return {
        "rel_l2": math.sqrt(num / den) if den > 0 else math.nan,
        "max_abs": max_abs,
        "max_abs_over_max_ref": max_abs / ref_max if ref_max > 0 else math.nan,
    }


def _classify(rel: dict[str, float], family: dict[str, str]) -> dict:
    best = min(rel, key=rel.__getitem__)
    fam = family[best]
    rivals = [v for k, v in rel.items() if family[k] != fam]
    margin = min(rivals) / rel[best] if rivals and rel[best] > 0 else math.inf
    ok = rel[best] <= CLASS_MAX_ERR and margin >= CLASS_MIN_MARGIN
    return {"class": fam if ok else "unclassified", "closest": best, "margin": margin}


def _reference(ref_fn, a, w, mode_a, mode_w, i, j):
    import torch

    return ref_fn(_tf32(torch, a[i:j], mode_a).double(), _tf32(torch, w, mode_w).double(), None)


def _errors(torch, y, ref_fn, a, w, which: str | None) -> dict:
    """Errors of y against the four candidates, rounding both operands (which=None) or only one ('a' or 'w')."""
    errs = {}
    for cand in CANDIDATES:
        m = _mode(cand)
        mode_a = m if which in (None, "a") else None
        mode_w = m if which in (None, "w") else None
        errs[cand] = _compare(torch, y, partial(_reference, ref_fn, a, w, mode_a, mode_w), a.shape[0])
    return errs


def _sha(torch, y) -> str:
    h = hashlib.sha256()
    for i in range(0, y.shape[0], CHUNK_ROWS):
        h.update(y[i : i + CHUNK_ROWS].contiguous().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def _measure(torch, fn, a, w, carrier, *, gen, ref_fn=None, tie_tests: bool | None = True, sha: bool = False) -> dict:
    """Run fn(a, w, carrier) on the GPU in fp32 and identify its arithmetic class.

    The first call lets cudnn.benchmark choose; the kernels are read from a later call. ref_fn (default fn) computes
    the fp64 references chunk by chunk. tie_tests=None runs the tie tests only if the realistic class is not fp32.
    """
    ref_fn = ref_fn or fn
    out = {"shape_a": list(a.shape), "shape_w": list(w.shape)}
    t0 = time.perf_counter()
    y = fn(a, w, carrier)
    torch.cuda.synchronize()
    out["first_call_s"] = time.perf_counter() - t0
    y2 = fn(a, w, carrier)
    out["repeat_bitwise_equal"] = bool(torch.equal(y, y2))
    del y2
    out["kernels"] = _cuda_kernels(torch, partial(fn, a, w, carrier))
    out["shape_y"] = list(y.shape)
    if sha:
        out["sha256_16"] = _sha(torch, y)
    errs = _errors(torch, y, ref_fn, a, w, None)
    out["realistic"] = {"rel_l2": {c: e["rel_l2"] for c, e in errs.items()}, "detail": errs}
    out["realistic"].update(_classify(out["realistic"]["rel_l2"], FAMILY))
    del y
    torch.cuda.empty_cache()
    if tie_tests is None:
        tie_tests = out["realistic"]["class"] != "fp32"
    if tie_tests:
        a_tie = _ties(torch, torch.randn(a.shape, generator=gen, device="cuda"))
        w_exact = _tf32(torch, w, "rz")
        out["tie_a"] = _tie_case(torch, fn, ref_fn, a_tie, w_exact, carrier, "a")
        del a_tie
        a_exact = _tf32(torch, a, "rz")
        w_tie = _ties(torch, torch.randn(w.shape, generator=gen, device="cuda") * w.std())
        out["tie_w"] = _tie_case(torch, fn, ref_fn, a_exact, w_tie, carrier, "w")
        del a_exact
        torch.cuda.empty_cache()
    return out


def _tie_case(torch, fn, ref_fn, a, w, carrier, which: str) -> dict:
    y = fn(a, w, carrier)
    errs = _errors(torch, y, ref_fn, a, w, which)
    rel = {c: e["rel_l2"] for c, e in errs.items()}
    return {"rel_l2": rel, **_classify(rel, IDENTITY)}


def _signature(case: dict) -> str:
    """One string per measured case: the realistic class and, when run, the per-operand tie-test classes."""
    if "error" in case:
        return "error"
    sig = case["realistic"]["class"]
    if "tie_a" in case:
        sig += f" (a:{case['tie_a']['class']}, w:{case['tie_w']['class']})"
    return sig


# ---------------------------------------------------------------------------------------------------------------------
# the operations under test (module level so partial() binds them without closures)


def _linear(a, w, carrier=None):
    import torch.nn.functional as F

    return F.linear(a, w)  # the forward of linear1/linear2 (cheri.py:1180): a @ w.T through cuBLAS


def _matmul(a, w, carrier=None):
    import torch

    return torch.matmul(a, w)  # the input-grad of a bias-free Linear: grad @ W (design §8.1)


def _conv_fwd(a, w, carrier=None, *, bias, padding):
    import torch.nn.functional as F

    return F.conv1d(a, w, bias.to(a.dtype), padding=padding)


def _conv_dgrad(a, w, carrier=None, *, padding):
    """Input-grad of a stride-1 Conv1d as autograd's ConvolutionBackward0 computes it (output_mask = T, F, F)."""
    import torch

    if carrier is None or carrier.dtype != a.dtype or carrier.shape[0] != a.shape[0]:
        carrier = a.new_zeros((a.shape[0], w.shape[1], a.shape[2]))  # the values are never read, only the shape
    mask = [True, False, False]
    return torch.ops.aten.convolution_backward(a, carrier, w, None, [1], [padding], [1], False, [0], 1, mask)[0]


def _cheri_fwd(a, w, carrier=None, *, d):
    from cherimoya.cheri import _cheri_conv

    return _cheri_conv(a, w, d)


def _cheri_dgrad(a, w, carrier=None, *, d):
    """The rule's push-back through the depthwise conv: autograd.grad of _cheri_conv (CH/deep_lift_shap.py:141-148)."""
    import torch
    from cherimoya.cheri import _cheri_conv

    with torch.enable_grad():
        x = carrier.detach().requires_grad_()
        y = _cheri_conv(x, w, d)
        (g,) = torch.autograd.grad(y, x, grad_outputs=a)
    return g


def _shift_add(a, w, carrier=None, *, d, transpose=False):
    """The 3-tap depthwise conv on (N, L, C) by shifts: y[l] = w0 a[l-d] + w1 a[l] + w2 a[l+d] (cheri.py:174-193).

    transpose gives the input-grad, g[l] = w0 a[l+d] + w1 a[l] + w2 a[l-d] (cheri.py:410-411).
    """
    import torch.nn.functional as F

    n = a.shape[1]
    p = F.pad(a, (0, 0, d, d))
    left, mid, right = p[:, :n], p[:, d : d + n], p[:, 2 * d : 2 * d + n]
    if transpose:
        left, right = right, left
    return left * w[0] + mid * w[1] + right * w[2]


# ---------------------------------------------------------------------------------------------------------------------
# P0-P5 (each runs in its own fresh process and fills `out` as it goes, so a failure keeps what was measured)


def _flag_snapshot(torch) -> dict:
    import warnings

    b = torch.backends
    getters = {
        "torch.get_float32_matmul_precision()": torch.get_float32_matmul_precision,
        "torch.backends.fp32_precision": lambda: b.fp32_precision,
        "torch.backends.cuda.matmul.allow_tf32": lambda: b.cuda.matmul.allow_tf32,
        "torch.backends.cuda.matmul.fp32_precision": lambda: b.cuda.matmul.fp32_precision,
        "torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction": (
            lambda: b.cuda.matmul.allow_fp16_reduced_precision_reduction
        ),
        "torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction": (
            lambda: b.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
        "torch.backends.cuda.preferred_blas_library()": lambda: str(b.cuda.preferred_blas_library()),
        "torch.backends.cudnn.enabled": lambda: b.cudnn.enabled,
        "torch.backends.cudnn.allow_tf32": lambda: b.cudnn.allow_tf32,
        "torch.backends.cudnn.fp32_precision": lambda: b.cudnn.fp32_precision,
        "torch.backends.cudnn.conv.fp32_precision": lambda: b.cudnn.conv.fp32_precision,
        "torch.backends.cudnn.rnn.fp32_precision": lambda: b.cudnn.rnn.fp32_precision,
        "torch.backends.cudnn.benchmark": lambda: b.cudnn.benchmark,
        "torch.backends.cudnn.benchmark_limit": lambda: b.cudnn.benchmark_limit,
        "torch.backends.cudnn.deterministic": lambda: b.cudnn.deterministic,
        "torch.backends.cudnn.depthwise_kernel": lambda: b.cudnn.depthwise_kernel,
        "torch.are_deterministic_algorithms_enabled()": torch.are_deterministic_algorithms_enabled,
        "torch.backends.mkldnn.matmul.fp32_precision": lambda: b.mkldnn.matmul.fp32_precision,
        "torch.backends.mkldnn.conv.fp32_precision": lambda: b.mkldnn.conv.fp32_precision,
    }
    snap: dict = {}
    notes: dict = {}
    for name, get in getters.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                value = get()
                snap[name] = value if isinstance(value, bool | int | float | str) or value is None else str(value)
            except Exception as exc:
                snap[name] = f"<{type(exc).__name__}: {exc}>"
        if caught:
            notes[name] = sorted({str(c.message) for c in caught})
    if notes:
        snap["_warnings"] = notes
    return snap


def probe_p0(args, out: dict) -> None:
    import importlib.metadata as md
    import platform
    import socket

    import torch

    out["flags_after_import_torch"] = _flag_snapshot(torch)
    import cherimoya

    out["flags_after_import_cherimoya"] = _flag_snapshot(torch)
    for mod in ("tangermeme.deep_lift_shap", "cherimoya.deep_lift_shap", "cherimoya_cli.commands.attribute"):
        importlib.import_module(mod)  # what `cherimoya attribute` imports next (attribute.py:7-18)
    out["flags_after_cli_imports"] = _flag_snapshot(torch)
    props = torch.cuda.get_device_properties(0)
    out["flags_after_cuda_init"] = _flag_snapshot(torch)
    steps = ("import_torch", "import_cherimoya", "cli_imports", "cuda_init")
    for prev, step in zip(steps, steps[1:]):
        before, after = out[f"flags_after_{prev}"], out[f"flags_after_{step}"]
        out[f"changed_by_{step}"] = {
            k: [before.get(k), after[k]] for k in after if k != "_warnings" and before.get(k) != after[k]
        }
    try:
        out["flags_after_cuda_init"]["torch.backends.cuda.cublas_workspace_size()"] = (
            torch.backends.cuda.cublas_workspace_size()
        )
    except Exception as exc:
        out["flags_after_cuda_init"]["torch.backends.cuda.cublas_workspace_size()"] = f"<{exc!r}>"

    import numba
    import numpy
    import tangermeme
    import triton

    out["versions"] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch.version.cuda": torch.version.cuda,
        "torch.version.git": torch.version.git_version,
        "cudnn": torch.backends.cudnn.version(),
        "triton": triton.__version__,
        "cherimoya": cherimoya.__version__,
        "tangermeme": tangermeme.__version__,
        "numpy": numpy.__version__,
        "numba": numba.__version__,
        "nvidia_wheels": sorted(
            f"{d.metadata['Name']}=={d.version}"
            for d in md.distributions()
            if (d.metadata["Name"] or "").lower().startswith("nvidia")
        ),
    }
    uuid = str(props.uuid)
    gpu = {
        "torch_name": props.name,
        "capability": f"{props.major}.{props.minor}",
        "total_memory_gib": props.total_memory / 2**30,
        "multi_processor_count": props.multi_processor_count,
        "uuid": uuid,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    fields = (
        "index,uuid,name,driver_version,memory.used,memory.total,utilization.gpu,clocks.sm,clocks.max.sm,power.draw"
    )
    for row in _smi(fields):
        if row[1].removeprefix("GPU-") == uuid.removeprefix("GPU-"):
            gpu["nvidia_smi"] = dict(zip(fields.split(","), row))
    out["gpu"] = gpu
    apps = _compute_apps(uuid)
    out["co_tenants"] = [a for a in apps if a["pid"] != os.getpid()]
    out["co_tenant_summary"] = {
        "n_processes": len(out["co_tenants"]),
        "used_memory_mib": sum(
            a["used_memory_mib"] for a in out["co_tenants"] if isinstance(a["used_memory_mib"], int)
        ),
    }
    out["host"] = {
        "hostname": socket.gethostname(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "torch_num_threads": torch.get_num_threads(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    out["numerics_env"] = {k: os.environ.get(k) for k in NUMERICS_ENV}
    out["env"] = {k: v for k, v in sorted(os.environ.items()) if k.startswith(ENV_PREFIXES)}


def probe_p1(args, out: dict) -> None:
    torch = _import_torch_as_user(args.gpu_mem_gb)
    w = _weights(torch, args)
    out["weights"] = w["source"]
    gen = _gen(torch, 11)
    L, C, H = args.length, w["C"], w["H"]
    out["cases"] = cases = []
    for rows in _ints(args.p1_rows):
        specs = (
            ("fwd_linear1", _linear, (rows, L, C), w["W1"], "F.linear(yhat, W1): (rows*L,C) @ (C,H)"),
            ("fwd_linear2", _linear, (rows, L, H), w["W2"], "F.linear(gelu(u), W2): (rows*L,H) @ (H,C)"),
            ("bwd_linear2_dgrad", _matmul, (rows, L, C), w["W2"], "matmul(G*rs, W2): (rows*L,C) @ (C,H)"),
            ("bwd_linear1_dgrad", _matmul, (rows, L, H), w["W1"], "matmul(Gu, W1): (rows*L,H) @ (H,C)"),
        )
        for name, fn, shape, weight, what in specs:
            logger.info("P1 %s rows=%d", name, rows)
            case = {"name": name, "rows": rows, "M": rows * L, "what": what}
            try:
                a = torch.randn(shape, generator=gen, device="cuda")
                case.update(_measure(torch, fn, a, weight, None, gen=gen))
                del a
            except Exception as exc:
                case["error"] = repr(exc)
                logger.exception("P1 %s rows=%d failed", name, rows)
            torch.cuda.empty_cache()
            cases.append(case)


def probe_p2(args, out: dict) -> None:
    torch = _import_torch_as_user(args.gpu_mem_gb)
    w = _weights(torch, args)
    out["weights"] = w["source"]
    out["torch.backends.cudnn.depthwise_kernel"] = str(torch.backends.cudnn.depthwise_kernel)
    gen = _gen(torch, 22)
    L, C = args.length, w["C"]
    out["cases"] = cases = []
    for rows in _ints(args.p2_rows):
        x = torch.randn((rows, L, C), generator=gen, device="cuda")
        g = torch.randn((rows, L, C), generator=gen, device="cuda")
        for cw, d in zip(w["cw"], w["dilations"]):
            logger.info("P2 rows=%d d=%d", rows, d)
            case = {"rows": rows, "dilation": d}
            try:
                fwd_ref = partial(_shift_add, d=d)
                case["fwd"] = _measure(
                    torch, partial(_cheri_fwd, d=d), x, cw, None, gen=gen, ref_fn=fwd_ref, tie_tests=None
                )
                bwd_ref = partial(_shift_add, d=d, transpose=True)
                bwd_fn = partial(_cheri_dgrad, d=d)
                case["dgrad"] = _measure(torch, bwd_fn, g, cw, x, gen=gen, ref_fn=bwd_ref, tie_tests=None)
                fwd_names = {k.split(" [x")[0] for k in case["fwd"]["kernels"]}
                case["dgrad"]["kernels_not_in_fwd"] = [
                    k for k in case["dgrad"]["kernels"] if k.split(" [x")[0] not in fwd_names
                ]
            except Exception as exc:
                case["error"] = repr(exc)
                logger.exception("P2 rows=%d d=%d failed", rows, d)
            torch.cuda.empty_cache()
            cases.append(case)
        del x, g


def probe_p3(args, out: dict) -> None:
    torch = _import_torch_as_user(args.gpu_mem_gb)
    w = _weights(torch, args)
    out["weights"] = w["source"]
    out["torch.backends.cudnn.benchmark"] = bool(torch.backends.cudnn.benchmark)
    out["torch.backends.cudnn.allow_tf32"] = bool(torch.backends.cudnn.allow_tf32)
    gen = _gen(torch, 33)  # the same seed in every run, so identical algorithms give identical bits (sha256_16)
    L, T = args.length, w["T"]
    W0, Wf = w["W0"], w["Wf"]
    out["cases"] = cases = []
    for rows in _ints(args.p3_rows):
        onehot = _onehot(torch, rows, L, gen)  # iconv input: the one-hot sequences and references
        x_f = torch.randn((rows, Wf.shape[1], L), generator=gen, device="cuda")  # fconv input: the residual stream
        g_i = torch.randn((rows, W0.shape[0], L), generator=gen, device="cuda")  # cotangent at the iconv output
        g_f = torch.randn((rows, Wf.shape[0], L), generator=gen, device="cuda")  # profile cotangent, zero off-window
        g_f[..., :T] = 0
        g_f[..., L - T :] = 0
        specs = (
            ("iconv_fwd", partial(_conv_fwd, bias=w["b0"], padding=w["pad0"]), onehot, W0, None),
            ("iconv_dgrad", partial(_conv_dgrad, padding=w["pad0"]), g_i, W0, onehot),
            ("fconv_fwd", partial(_conv_fwd, bias=w["bf"], padding=w["padf"]), x_f, Wf, None),
            ("fconv_dgrad", partial(_conv_dgrad, padding=w["padf"]), g_f, Wf, x_f),
        )
        for name, fn, a, weight, carrier in specs:
            logger.info("P3 run %d %s rows=%d", args.run_index, name, rows)
            case = {"op": name, "rows": rows}
            try:
                case.update(_measure(torch, fn, a, weight, carrier, gen=gen, sha=True))
            except Exception as exc:
                case["error"] = repr(exc)
                logger.exception("P3 %s rows=%d failed", name, rows)
            torch.cuda.empty_cache()
            cases.append(case)
        del onehot, x_f, g_i, g_f
        torch.cuda.empty_cache()


def _ulp(torch, a, b):
    """|a - b| in units in the last place of fp32, counting representable values between them across zero."""

    def key(t):
        i = t.contiguous().view(torch.int32).to(torch.int64)
        return torch.where(i < 0, -(i & 0x7FFFFFFF), i)

    return (key(a) - key(b)).abs()


def probe_p4(args, out: dict) -> None:
    torch = _import_torch_as_user(args.gpu_mem_gb)
    import torch.nn.functional as F

    w = _weights(torch, args)
    out["weights"] = w["source"]
    gen = _gen(torch, 44)
    C, L, T = w["C"], args.length, w["T"]
    Wl, bl = w["Wl"], w["bl"]
    weights = {
        "count_linear_row0": Wl[0:1].contiguous(),
        "randn": torch.randn((1, Wl.shape[1]), generator=gen, device="cuda"),
    }
    out["mm"] = mm = []
    out["head_chain"] = chain = []
    for M in _ints(args.p4_rows):
        ones = torch.ones(M, 1, device="cuda")
        for label, W in weights.items():
            res = ones @ W
            exact = W.expand(M, -1)
            mm.append(
                {
                    "M": M,
                    "weight": label,
                    "bitwise_equal_to_W": bool(torch.equal(res, exact)),
                    "n_differing": int((res != exact).sum()),
                    "max_ulp": int(_ulp(torch, res, exact).max()),
                    "equal_to_tf32_W": {
                        m: bool(torch.equal(res, _tf32(torch, W, m).expand(M, -1))) for m in TF32_MODES
                    },
                    "kernels": _cuda_kernels(torch, partial(torch.matmul, ones, W)),
                }
            )
        # §7.1: the reference's own head chain under autograd, compared with MeanBackward applied to the exact W.
        Hd = torch.zeros(M, C, L, device="cuda", requires_grad=True)
        pooled = torch.mean(Hd[:, :, T : L - T].float(), dim=2)
        if w["n_ctl"]:
            pooled = torch.cat([pooled, torch.log(pooled.new_zeros(M, 1) + 1)], -1)
        y = F.linear(pooled, Wl, bl)[:, 0:1][:, 0]
        (gH,) = torch.autograd.grad(y.sum(), Hd)
        Hd2 = torch.zeros(M, C, L, device="cuda", requires_grad=True)
        pooled2 = torch.mean(Hd2[:, :, T : L - T].float(), dim=2)
        (gH2,) = torch.autograd.grad(pooled2, Hd2, grad_outputs=Wl[0:1, :C].expand(M, C).contiguous())
        win = gH[:, :, T : L - T]
        n = L - 2 * T
        chain.append(
            {
                "M": M,
                "equal_to_mean_backward_of_exact_W": bool(torch.equal(gH, gH2)),
                "max_ulp_vs_exact_chain": int(_ulp(torch, gH, gH2).max()),
                "window_equal_to_W_div_n": bool(torch.equal(win, (Wl[0, :C] / n)[None, :, None].expand_as(win))),
                "window_equal_to_W_mul_inv_n": bool(
                    torch.equal(win, (Wl[0, :C] * torch.tensor(1.0 / n, device="cuda"))[None, :, None].expand_as(win))
                ),
                "zero_outside_window": bool((gH[:, :, :T] == 0).all() and (gH[:, :, L - T :] == 0).all()),
                "rows_identical": bool((gH == gH[:1]).all()),
            }
        )
        del Hd, Hd2, gH, gH2, pooled, pooled2, y


def _ulp_stats(torch, a, b, x) -> dict:
    u = _ulp(torch, a, b)
    n = u.numel()
    hist = {}
    for lo, hi in ULP_BINS:
        sel = (u >= lo) if hi is None else ((u >= lo) & (u <= hi))
        hist[str(lo) if lo == hi else (f">={lo}" if hi is None else f"{lo}-{hi}")] = int(sel.sum())
    k = int(torch.argmax(u))
    diff = (a.double() - b.double()).abs()
    kd = int(torch.argmax(diff))
    big = b.abs() >= 1e-6
    ref_max = float(b.abs().max())
    ulp_of_max = 2.0 ** (math.floor(math.log2(ref_max)) - 23) if ref_max > 0 else math.nan
    return {
        "n": n,
        "hist": hist,
        "fraction_exact": hist["0"] / n,
        "max_ulp": int(u[k]),
        "x_at_max_ulp": float(x[k]),
        "values_at_max_ulp": [float(a[k]), float(b[k])],
        "max_ulp_where_ref_abs_ge_1e-6": int(u[big].max()) if bool(big.any()) else 0,
        "max_abs_diff": float(diff[kd]),
        "max_abs_ref": ref_max,
        "max_abs_diff_in_ulps_of_max_ref": float(diff[kd]) / ulp_of_max,
        "x_at_max_abs_diff": float(x[kd]),
        "sign_disagreements": int((torch.sign(a) != torch.sign(b)).sum()),
    }


def _tanh_lines(codes) -> list[str]:
    lines = []
    for code in codes:
        lines += [ln.strip() for ln in code.splitlines() if "tanh" in ln and not ln.strip().startswith("#")]
    return lines[:20]


def probe_p5(args, out: dict) -> None:
    torch = _import_torch_as_user(args.gpu_mem_gb)
    import torch.nn.functional as F
    from torch._inductor.utils import run_and_get_code

    # Closures over the already-imported modules: Dynamo traces these with fullgraph=True, so no import inside.
    def gelu(t):
        return F.gelu(t, approximate="tanh")

    def gelu_backward(g, t):
        return torch.ops.aten.gelu_backward(g, t, approximate="tanh")

    n = args.p5_n
    x = torch.linspace(-6.0, 6.0, n, device="cuda", dtype=torch.float32)
    gen = _gen(torch, 55)
    out["n"] = n
    out["inductor_config"] = {
        k: str(getattr(torch._inductor.config, k, "<absent>"))
        for k in ("compile_threads", "emulate_precision_casts", "use_fast_math", "fx_graph_cache")
    }
    beta = math.sqrt(2.0 / math.pi)
    x64 = x.double()
    th = torch.tanh(beta * (x64 + 0.044715 * x64**3))
    truth_f = (0.5 * x64 * (1.0 + th)).float()
    dtruth = 0.5 * (1.0 + th) + 0.5 * x64 * (1.0 - th * th) * beta * (1.0 + 3.0 * 0.044715 * x64 * x64)
    del th

    fwd_c = torch.compile(gelu, fullgraph=True, dynamic=False)
    y_c, codes = run_and_get_code(fwd_c, x)
    y_e = gelu(x)
    out["gelu"] = {
        "compiled_vs_eager": _ulp_stats(torch, y_c, y_e, x),
        "eager_vs_fp64": _ulp_stats(torch, y_e, truth_f, x),
        "compiled_vs_fp64": _ulp_stats(torch, y_c, truth_f, x),
        "inductor_tanh_lines": _tanh_lines(codes),
    }
    del y_c, y_e, truth_f

    bwd_c = torch.compile(gelu_backward, fullgraph=True, dynamic=False)
    grads = {"ones": torch.ones_like(x), "randn": torch.randn(x.shape, generator=gen, device="cuda")}
    out["gelu_backward"] = {}
    for label, g in grads.items():
        if label == "ones":
            b_c, codes = run_and_get_code(bwd_c, g, x)
            tanh_lines = _tanh_lines(codes)
        else:
            b_c, tanh_lines = bwd_c(g, x), None
        b_e = gelu_backward(g, x)
        truth_b = (g.double() * dtruth).float()
        out["gelu_backward"][f"grad_{label}"] = {
            "compiled_vs_eager": _ulp_stats(torch, b_c, b_e, x),
            "eager_vs_fp64": _ulp_stats(torch, b_e, truth_b, x),
            "compiled_vs_fp64": _ulp_stats(torch, b_c, truth_b, x),
            "inductor_tanh_lines": tanh_lines,
        }
        del b_c, b_e, truth_b


PROBE_FUNCS = {"P0": probe_p0, "P1": probe_p1, "P2": probe_p2, "P3": probe_p3, "P4": probe_p4, "P5": probe_p5}


def _resources() -> dict:
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_initialized():
        return {}
    torch.cuda.synchronize()
    mine = [a for a in _compute_apps() if a["pid"] == os.getpid()]
    return {
        "max_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "max_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "nvidia_smi_process_mib_at_end": mine[0]["used_memory_mib"] if mine else None,
    }


def _child(args) -> int:
    rec = {"probe": args.run_probe, "run_index": args.run_index, "pid": os.getpid(), "started_et": _now_et()}
    rec["args"] = {k: str(v) for k, v in vars(args).items()}
    out: dict = {}
    t0 = time.time()
    try:
        PROBE_FUNCS[args.run_probe](args, out)
        rec["status"] = "ok"
    except Exception:
        rec["status"] = "error"
        rec["error"] = traceback.format_exc()
        logger.exception("%s failed", args.run_probe)
    rec["result"] = out
    rec["wall_s"] = time.time() - t0
    try:
        rec["resources"] = _resources()
    except Exception as exc:
        rec["resources"] = {"error": repr(exc)}
    rec["finished_et"] = _now_et()
    _write_json(args.result, rec)
    return 0 if rec["status"] == "ok" else 1


# ---------------------------------------------------------------------------------------------------------------------
# driver: one fresh subprocess per probe (P3 once per repeat), then the merged report


def _passthrough(args) -> list[str]:
    argv = [
        "--length", str(args.length),
        "--p1-rows", args.p1_rows,
        "--p2-rows", args.p2_rows,
        "--p3-rows", args.p3_rows,
        "--p4-rows", args.p4_rows,
        "--p5-n", str(args.p5_n),
        "--gpu-mem-gb", str(args.gpu_mem_gb),
    ]  # fmt: skip
    if args.model:
        argv += ["--model", str(args.model)]
    return argv


def _child_env(args) -> dict:
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "4")
    env["TORCHINDUCTOR_COMPILE_THREADS"] = "1"  # otherwise Inductor forks up to 32 compile workers
    cache = Path(args.cache_dir)
    env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache / "inductor"))
    env.setdefault("TRITON_CACHE_DIR", str(cache / "triton"))
    return env


def _tail(path: Path, n: int = 40) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def _run_child(args, probe: str, run_index: int, raw_dir: Path, log_dir: Path) -> None:
    name = f"{probe}_run{run_index}" if run_index else probe
    result, log_path = raw_dir / f"{name}.json", log_dir / f"{name}.log"
    result.unlink(missing_ok=True)
    cmd = [sys.executable, str(Path(__file__).resolve()), "--run-probe", probe, "--result", str(result)]
    cmd += ["--run-index", str(run_index), *_passthrough(args)]
    logger.info("%s: starting (log %s)", name, log_path)
    t0 = time.time()
    with open(log_path, "w") as log:
        try:
            rc = subprocess.run(
                cmd, stdout=log, stderr=subprocess.STDOUT, env=_child_env(args), timeout=args.timeout, check=False
            ).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    if not result.exists():
        _write_json(
            result,
            {
                "probe": probe,
                "run_index": run_index,
                "status": "crashed",
                "returncode": rc,
                "log_tail": _tail(log_path),
            },
        )
    logger.info("%s: done in %.0f s (exit %s)", name, time.time() - t0, rc)


def _conclude(raw: dict) -> dict:
    """Turn the raw probe records into the findings the design asks for."""
    con: dict = {}
    p0 = raw.get("P0", {}).get("result", {})
    if p0:
        flags = p0.get("flags_after_import_cherimoya", {})
        con["P0"] = {
            "flags_after_import_cherimoya": {k: v for k, v in flags.items() if k != "_warnings"},
            "changed_by_import_cherimoya": p0.get("changed_by_import_cherimoya"),
            "changed_by_cli_imports": p0.get("changed_by_cli_imports"),
            "changed_by_cuda_init": p0.get("changed_by_cuda_init"),
            "gpu": p0.get("gpu", {}).get("nvidia_smi", p0.get("gpu")),
            "co_tenants": p0.get("co_tenant_summary"),
            "numerics_env_set": {k: v for k, v in p0.get("numerics_env", {}).items() if v is not None},
            "versions": {k: v for k, v in p0.get("versions", {}).items() if k != "nvidia_wheels"},
        }

    p1 = raw.get("P1", {}).get("result", {}).get("cases", [])
    if p1:
        sigs = {f"{c['name']}@{c['rows']}": _signature(c) for c in p1}
        con["P1"] = {"classes": sigs, "one_class_for_all_shapes": len(set(sigs.values())) == 1}

    p2 = raw.get("P2", {}).get("result", {}).get("cases", [])
    if p2:
        fwd = {f"d{c['dilation']}@{c['rows']}": _signature(c["fwd"]) for c in p2 if "fwd" in c}
        bwd = {f"d{c['dilation']}@{c['rows']}": _signature(c["dgrad"]) for c in p2 if "dgrad" in c}
        kernels = sorted(
            {k.split(" [x")[0] for c in p2 for part in ("fwd", "dgrad") for k in c.get(part, {}).get("kernels", [])}
        )
        all_fp32 = all(s == "fp32" for s in [*fwd.values(), *bwd.values()]) and len(fwd) + len(bwd) == 2 * len(p2)
        con["P2"] = {
            "fwd_classes": fwd,
            "dgrad_classes": bwd,
            "kernels_seen": kernels,
            "rule_conv_contingency_needed": "no" if all_fp32 else "yes",
            "max_rel_l2_vs_fp64": max(
                (c[p]["realistic"]["rel_l2"]["fp32"] for c in p2 for p in ("fwd", "dgrad") if p in c), default=None
            ),
        }

    runs = sorted((v for k, v in raw.items() if k.startswith("P3")), key=lambda r: r.get("run_index", 0))
    if runs:
        con["P3"] = _conclude_p3(runs)

    p4 = raw.get("P4", {}).get("result", {})
    if p4:
        con["P4"] = {
            "mm_bitwise_equal": {f"{r['weight']}@M={r['M']}": r["bitwise_equal_to_W"] for r in p4.get("mm", [])},
            "head_chain_equal_to_exact": {
                f"M={r['M']}": r["equal_to_mean_backward_of_exact_W"] for r in p4.get("head_chain", [])
            },
        }

    p5 = raw.get("P5", {}).get("result", {})
    if p5:
        summ = {}
        if "gelu" in p5:
            summ["gelu"] = p5["gelu"]["compiled_vs_eager"]
        for k, v in p5.get("gelu_backward", {}).items():
            summ[f"gelu_backward_{k}"] = v["compiled_vs_eager"]
        con["P5"] = {
            name: {
                "fraction_exact": s["fraction_exact"],
                "max_ulp": s["max_ulp"],
                "max_ulp_where_ref_abs_ge_1e-6": s["max_ulp_where_ref_abs_ge_1e-6"],
                "max_abs_diff": s["max_abs_diff"],
                "max_abs_diff_in_ulps_of_max_ref": s.get("max_abs_diff_in_ulps_of_max_ref"),
            }
            for name, s in summ.items()
        }
    return con


def _compute_kernels(names: list[str]) -> list[str]:
    return [k for k in names if not any(h in k.lower() for h in HELPER_KERNELS)]


def _op_state(by_rows: dict, ref_rows) -> dict:
    """Where a convolution's class sits at the reference's row count against ours, over fresh processes.

    Pinning helps when the reference's shape gives one class in every process and ours gives another class or
    flips between classes ('ours unstable'). When the reference itself flips, no pin matches every reference run;
    the class it picks most often is then only a suggestion, which removes our own flips.
    """
    counts = {rows: dict(Counter(c["class"] for c in cells)) for rows, cells in by_rows.items()}
    ref = counts.get(ref_rows, {})
    ours = {rows: cls for rows, cls in counts.items() if rows != ref_rows}
    bad = any(k == "error" or k.startswith("unclassified") for cls in counts.values() for k in cls)
    pin = suggest = None
    if bad or not ref or not ours:
        state = "unclear"
    elif len(ref) > 1:
        state = "reference unstable"
        majority = max(ref, key=ref.__getitem__)
        suggest = f"{majority.split(' ')[0]} (the reference's class in {ref[majority]}/{sum(ref.values())})"
    else:
        ref_class = next(iter(ref))
        if all(set(cls) == {ref_class} for cls in ours.values()):
            state = "match"
        else:
            state = "mismatch" if all(len(cls) == 1 for cls in ours.values()) else "ours unstable"
            pin = ref_class.split(" ")[0]  # the class itself, without the tie-test bracket
    return {
        "state": state,
        "pin_to": pin,
        "suggested_pin": suggest,
        "reference_rows": ref_rows,
        "class_counts_by_rows": counts,
    }


def _conclude_p3(runs: list[dict]) -> dict:
    table: dict = {}
    ref_rows = None
    for run in runs:
        for case in run.get("result", {}).get("cases", []):
            ref_rows = case["rows"] if ref_rows is None else ref_rows  # the first --p3-rows value is the reference's
            table.setdefault(case["op"], {}).setdefault(case["rows"], []).append(
                {
                    "run": run.get("run_index"),
                    "class": _signature(case),
                    "deterministic_in_process": case.get("repeat_bitwise_equal"),
                    "sha256_16": case.get("sha256_16"),
                    "compute_kernels": _compute_kernels(case.get("kernels", [])),
                }
            )
    ops = {}
    for op, by_rows in table.items():
        ops[op] = _op_state(by_rows, ref_rows)
        ops[op]["same_bits_across_runs_by_rows"] = {
            rows: len({c["sha256_16"] for c in cells}) == 1 for rows, cells in by_rows.items()
        }
        ops[op]["deterministic_in_process_by_rows"] = {
            rows: all(c["deterministic_in_process"] for c in cells) for rows, cells in by_rows.items()
        }
        ops[op]["kernel_counts_by_rows"] = {
            rows: dict(Counter(" + ".join(c["compute_kernels"]) or "-" for c in cells))
            for rows, cells in by_rows.items()
        }
    heads = {}
    for head, names in HEAD_OPS.items():
        states = {op: ops[op]["state"] for op in names if op in ops}
        if len(states) < len(names) or any(st in ("unclear", "reference unstable") for st in states.values()):
            needed = "unclear"
        elif any(st in ("mismatch", "ours unstable") for st in states.values()):
            needed = "yes"
        else:
            needed = "no"
        heads[head] = {
            "conv_class_needed": needed,
            "states": states,
            "pins": {op: ops[op]["pin_to"] for op in names if op in ops and ops[op]["pin_to"]},
            "suggested_pins": {op: ops[op]["suggested_pin"] for op in names if op in ops and ops[op]["suggested_pin"]},
        }
    return {"n_runs": len(runs), "reference_rows": ref_rows, "ops": ops, "heads": heads, "table": table}


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.2e}"
    return str(v)


def _short(name: str, width: int = 90) -> str:
    name = name.removeprefix("void ")
    return (name if len(name) <= width else name[: width - 3] + "...").replace("|", "/")


def _short_kernels(names: list[str], n: int = 2) -> str:
    return "; ".join(_short(k) for k in _compute_kernels(names)[:n]) or "-"


def _tie_rounding(cases: list[dict]) -> dict:
    """How the TF32 kernels among `cases` round exact ties, per operand, from the tie tests."""
    seen: Counter = Counter()
    for c in cases:
        if "error" in c or not c.get("realistic", {}).get("class", "").startswith("tf32") or "tie_a" not in c:
            continue
        seen[f"a:{c['tie_a']['class']}"] += 1
        seen[f"w:{c['tie_w']['class']}"] += 1
    return dict(seen)


def _key_findings(con: dict, raw: dict) -> list[str]:
    """One line per probe, derived from the measurements, for the top of the summary."""
    out = []
    p1_cases = raw.get("P1", {}).get("result", {}).get("cases", [])
    p3_cases = [c for k, v in raw.items() if k.startswith("P3") for c in v.get("result", {}).get("cases", [])]
    if "P0" in con:
        changed = ", ".join(
            f"{k} {v[0]} -> {v[1]}" for k, v in (con["P0"].get("changed_by_import_cherimoya") or {}).items()
        )
        flags = con["P0"].get("flags_after_import_cherimoya") or {}
        out.append(
            f"P0: `import cherimoya` changes {changed or 'nothing'}; cudnn.allow_tf32 = "
            f"{flags.get('torch.backends.cudnn.allow_tf32')} and cudnn.conv.fp32_precision = "
            f"{flags.get('torch.backends.cudnn.conv.fp32_precision')} are torch's own process defaults."
        )
    if "P1" in con:
        sigs = Counter(con["P1"]["classes"].values())
        out.append(
            f"P1: block GEMMs through cuBLAS, {sum(sigs.values())} shape/form cases: "
            + "; ".join(f"{sig} in {n}" for sig, n in sigs.items())
            + "."
        )
    if "P2" in con:
        p2 = con["P2"]
        out.append(
            f"P2: the rule's depthwise conv runs PyTorch's native conv_depthwise2d kernels; forward "
            f"{dict(Counter(p2['fwd_classes'].values()))}, input-grad {dict(Counter(p2['dgrad_classes'].values()))}, "
            f"max rel-L2 {_fmt(p2['max_rel_l2_vs_fp64'])} against the fp64 shift-add."
        )
    if p1_cases or p3_cases:
        out.append(
            f"TF32 tie rounding per operand: cuBLAS (P1) {_tie_rounding(p1_cases) or 'no TF32 case'}; "
            f"cuDNN (P3) {_tie_rounding(p3_cases) or 'no TF32 case'}."
        )
    if "P3" in con:
        nondet = [
            f"{op}@{rows}"
            for op, v in con["P3"]["ops"].items()
            for rows, det in v["deterministic_in_process_by_rows"].items()
            if not det
        ]
        out.append(f"P3: convolutions whose bits change from call to call in one process: {nondet or 'none'}.")
    if "P4" in con:
        mm, chain = con["P4"]["mm_bitwise_equal"], con["P4"]["head_chain_equal_to_exact"]
        out.append(
            f"P4: ones(M,1) @ W is bitwise W in {sum(mm.values())}/{len(mm)} cases; the §7.1 head chain equals "
            f"MeanBackward of the exact W in {sum(chain.values())}/{len(chain)}."
        )
    if "P5" in con:
        parts = [
            f"{name} {v['fraction_exact']:.3f} exact, max abs diff {_fmt(v['max_abs_diff'])}"
            for name, v in con["P5"].items()
        ]
        out.append("P5: Inductor against eager: " + "; ".join(parts) + ".")
    return out


def _caveats(con: dict, args: dict) -> list[str]:
    p0 = con.get("P0", {})
    gpu, co = p0.get("gpu") or {}, p0.get("co_tenants") or {}
    return [
        f"Each child capped its CUDA caching allocator at {args.get('gpu_mem_gb')} GiB, so cuDNN plans needing more "
        "workspace than the cap left were not candidates in P3; an uncapped `cherimoya attribute` could pick them.",
        f"cudnn.benchmark timed its candidates on a GPU shared with {co.get('n_processes')} other processes at "
        f"{gpu.get('utilization.gpu', '?')}% utilization (P0), the load the pilot ran under; picks on a free GPU may "
        "differ.",
        f"P3 probed rows {args.get('p3_rows')} only: the engine's real shapes (R = S(K+1) rows forward, P = SK rows "
        "input-grad) were not probed, and the reference's final partial batch (24 rows in the pilot) was probed in "
        "P2 and P4 but not for the cuDNN iconv/fconv convolutions.",
    ]


def _markdown(rep: dict) -> str:
    con, raw = rep["conclusions"], rep["probes"]
    md = ["# cheri_dls GPU probes P0-P5 (final_design.md §15.2)", ""]
    md.append(
        f"Generated {rep['generated_et']} (ET) by `workflows/attribution/dls_probe.py`; every number is in "
        "`dls_probes.json`, raw records and logs under `raw/` and `logs/`."
    )
    md.append("")
    if "P3" in con:
        p3 = con["P3"]
        md.append(
            f"**§12 conv_class contingency** ({p3['n_runs']} fresh P3 processes; the reference's shape is "
            f"{p3['reference_rows']} rows):"
        )
        md.append("")
        for head, h in p3["heads"].items():
            pins = ", ".join(f"pin {op} to `{c}`" for op, c in h["pins"].items())
            pins += "".join(f", optional pin {op} to {c}" for op, c in h.get("suggested_pins", {}).items())
            pins = pins.removeprefix(", ")
            states = "; ".join(f"{op} {st}" for op, st in h["states"].items())
            md.append(f"- {head} head: **{h['conv_class_needed']}** ({states}){'. ' + pins if pins else ''}")
        md.append("")
    if "P2" in con:
        md.append(f"**§17 rule_conv contingency (P2) needed: {con['P2']['rule_conv_contingency_needed']}.**")
        md.append("")
    if con.get("key_findings"):
        md += ["## Key findings", ""] + [f"- {line}" for line in con["key_findings"]] + [""]
    if con.get("caveats"):
        md += ["## Caveats", ""] + [f"- {line}" for line in con["caveats"]] + [""]
    if con.get("notes"):
        md += ["## Notes", ""] + [f"- {line}" for line in con["notes"]] + [""]
    md.append("Class key: `fp32` = IEEE fp32 products; `tf32_rn` / `tf32_rz` = operands rounded to TF32 to nearest /")
    md.append("by truncation. The bracket gives the tie tests per operand (a = data or cotangent, w = weight), which")
    md.append("separate ties-to-even (`tf32_rne`) from ties-away (`tf32_rna`).")
    md.append("")

    if "P0" in con:
        p0 = con["P0"]
        md += ["## P0 environment", ""]
        gpu = p0.get("gpu") or {}
        md.append(
            f"- GPU {gpu.get('index', '?')}: {gpu.get('name', '?')}, driver {gpu.get('driver_version', '?')}, "
            f"{gpu.get('memory.used', '?')}/{gpu.get('memory.total', '?')} MiB used, "
            f"utilization {gpu.get('utilization.gpu', '?')}% at the start of P0."
        )
        co = p0.get("co_tenants") or {}
        md.append(
            f"- Co-tenant compute processes on it: {co.get('n_processes')}, using {co.get('used_memory_mib')} MiB."
        )
        md.append("- Versions: " + ", ".join(f"{k} {v}" for k, v in (p0.get("versions") or {}).items()) + ".")
        md.append(f"- Numerics-relevant environment variables set: {p0.get('numerics_env_set') or 'none'}.")
        later = {**(p0.get("changed_by_cli_imports") or {}), **(p0.get("changed_by_cuda_init") or {})}
        md.append(f"- Flags changed by the CLI's other imports or by CUDA initialization: {later or 'none'}.")
        md += ["", "| flag after `import cherimoya` | value | before the import |", "|---|---|---|"]
        changed = p0.get("changed_by_import_cherimoya") or {}
        for k, v in (p0.get("flags_after_import_cherimoya") or {}).items():
            md.append(f"| `{k}` | `{v}` | {'`' + str(changed[k][0]) + '`' if k in changed else 'same'} |")
        md.append("")

    if "P1" in raw:
        md += ["## P1 GEMM class (cuBLAS, after `import cherimoya`)", ""]
        md += [
            "| GEMM | rows | M | class | rel-L2 vs fp64 | vs TF32-RNE | vs TF32-RZ | kernel |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for c in raw["P1"].get("result", {}).get("cases", []):
            if "error" in c:
                md.append(f"| {c['name']} | {c['rows']} | {c['M']} | error: {_short(c['error'], 60)} | | | | |")
                continue
            r = c["realistic"]["rel_l2"]
            md.append(
                f"| {c['name']} | {c['rows']} | {c['M']} | {_signature(c)} | {_fmt(r['fp32'])} | "
                f"{_fmt(r['tf32_rne'])} | {_fmt(r['tf32_rz'])} | `{_short_kernels(c['kernels'], 1)}` |"
            )
        md.append("")

    if "P2" in con:
        p2 = con["P2"]
        md += ["## P2 rule depthwise conv (`cherimoya.cheri._cheri_conv`, forward and autograd input-grad)", ""]
        md.append(
            f"- Classes over every dilation and row count: forward {dict(Counter(p2['fwd_classes'].values()))}, "
            f"input-grad {dict(Counter(p2['dgrad_classes'].values()))}; max rel-L2 against the fp64 shift-add "
            f"{_fmt(p2['max_rel_l2_vs_fp64'])}."
        )
        md.append(
            "- Compute kernels: " + "; ".join(f"`{_short(k, 110)}`" for k in _compute_kernels(p2["kernels_seen"]))
        )
        md.append("")

    if "P3" in con:
        p3 = con["P3"]
        md += ["## P3 cuDNN classes for iconv and fconv (cudnn.benchmark on, fresh process per run)", ""]
        md += [
            "| op | rows | class: processes | deterministic in-process | same bits across runs | compute kernel: processes |",
            "|---|---|---|---|---|---|",
        ]
        for op, v in p3["ops"].items():
            for rows, cls in v["class_counts_by_rows"].items():
                classes = "; ".join(f"{c}: {n}" for c, n in cls.items())
                kernels = "; ".join(f"`{_short(k, 80)}`: {n}" for k, n in v["kernel_counts_by_rows"][rows].items())
                md.append(
                    f"| {op} | {rows} | {classes} | {v['deterministic_in_process_by_rows'][rows]} | "
                    f"{v['same_bits_across_runs_by_rows'][rows]} | {kernels} |"
                )
        md.append("")
        md.append("A kernel that is not deterministic in-process gives different bits on every call, so its rows show")
        md.append("'same bits across runs: False' without any change of algorithm.")
        md.append("")

    if "P4" in raw:
        md += ["## P4 count head K=1 GEMM", ""]
        p4 = raw["P4"].get("result", {})
        for r in p4.get("mm", []):
            md.append(
                f"- ones({r['M']},1) @ W [{r['weight']}]: bitwise equal to W = {r['bitwise_equal_to_W']} "
                f"({r['n_differing']} elements differ, max {r['max_ulp']} ulp; equal to TF32-rounded W: "
                f"{r['equal_to_tf32_W']}); kernel `{_short_kernels(r['kernels'], 1)}`"
            )
        for r in p4.get("head_chain", []):
            md.append(
                f"- §7.1 head chain at M={r['M']}: equal to MeanBackward of the exact W = "
                f"{r['equal_to_mean_backward_of_exact_W']}; zero off the window = {r['zero_outside_window']}; "
                f"rows identical = {r['rows_identical']}"
            )
        md.append("")

    if "P5" in raw:
        md += ["## P5 GELU(tanh): Inductor against eager CUDA, 1e7 points in [-6, 6]", ""]
        p5 = raw["P5"].get("result", {})
        hdr = list(p5.get("gelu", {}).get("compiled_vs_eager", {}).get("hist") or {})
        md += ["| function | " + " | ".join(f"{h} ulp" for h in hdr) + " | max abs diff | in ulp of max abs value |"]
        md += ["|---|" + "---|" * (len(hdr) + 2)]
        named = [("gelu", p5.get("gelu", {}))]
        named += [(f"gelu_backward {k}", v) for k, v in p5.get("gelu_backward", {}).items()]
        for name, v in named:
            st = v.get("compiled_vs_eager")
            if not st:
                continue
            md.append(
                f"| {name} | "
                + " | ".join(str(st["hist"].get(h, 0)) for h in hdr)
                + f" | {_fmt(st['max_abs_diff'])} | {_fmt(st.get('max_abs_diff_in_ulps_of_max_ref', math.nan))} |"
            )
        md.append("")
        md.append(
            "Large ulp counts sit where the value is near zero (gelu_backward crosses zero near x = -0.75), so the"
        )
        md.append(
            "absolute difference and its size in ulps of the function's largest value are the meaningful measures."
        )
        for name, v in named:
            e, c = v.get("eager_vs_fp64"), v.get("compiled_vs_fp64")
            if e and c:
                md.append(
                    f"- {name} against fp64 truth: eager exact {e['fraction_exact']:.4f}, max abs diff "
                    f"{_fmt(e['max_abs_diff'])}; compiled exact {c['fraction_exact']:.4f}, max abs diff "
                    f"{_fmt(c['max_abs_diff'])}"
                )
        md.append("")

    md += ["## Resources per child", ""]
    for name, rec in raw.items():
        res = rec.get("resources") or {}
        md.append(
            f"- {name}: {rec.get('status')}, {_fmt(rec.get('wall_s', math.nan))} s, peak reserved "
            f"{_fmt(res.get('max_reserved_gib', math.nan))} GiB, nvidia-smi {res.get('nvidia_smi_process_mib_at_end')} "
            "MiB at the end"
        )
    md.append("")
    return "\n".join(md)


def _merge(args, raw_dir: Path) -> dict:
    notes_path = args.out_dir / "notes.txt"
    raw = {}
    for path in sorted(raw_dir.glob("*.json")):
        try:
            raw[path.stem] = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raw[path.stem] = {"status": "unreadable", "error": repr(exc)}
    run_args = {k: str(v) for k, v in vars(args).items() if not k.startswith("run_")}
    con = _conclude(raw)
    con["key_findings"] = _key_findings(con, raw)
    # The children's own arguments describe what was measured; the merge's are the fallback for older records.
    p3_args = next((r["args"] for k, r in sorted(raw.items()) if k.startswith("P3") and r.get("args")), None)
    con["caveats"] = _caveats(con, p3_args or run_args)
    con["notes"] = notes_path.read_text().splitlines() if notes_path.exists() else []
    return {
        "generated_et": _now_et(),
        "design": "final_design.md §15.2 (P0-P5), §12 conv_class, §17 rule_conv, §7.1 counts cotangent",
        "args": run_args,
        "conclusions": con,
        "probes": raw,
    }


def _parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, help="where dls_probes.json, dls_probes.md, raw/ and logs/ go")
    ap.add_argument("--probes", default=",".join(PROBES), help="comma-separated subset of P0..P5 (default all)")
    ap.add_argument("--model", type=Path, help="a Cherimoya checkpoint whose weights the probes use (default: draws)")
    ap.add_argument("--length", type=int, default=2114, help="input length L (default 2114)")
    ap.add_argument("--p1-rows", default="128,240", help="P1 rows; M = rows * L (default 128,240)")
    ap.add_argument("--p2-rows", default="128,24", help="P2 rows: the reference's 2*batch and last batch (128,24)")
    ap.add_argument("--p3-rows", default="128,260", help="P3 rows: the reference's 128 against ours (128,260)")
    ap.add_argument("--p3-repeats", type=int, default=3, help="fresh processes for P3 (default 3)")
    ap.add_argument("--p3-first-run", type=int, default=1, help="index of the first P3 run, to add samples (default 1)")
    ap.add_argument("--p4-rows", default="128,24", help="P4 M values (default 128,24)")
    ap.add_argument("--p5-n", type=int, default=10**7, help="P5 sample count in [-6, 6] (default 1e7)")
    ap.add_argument("--gpu-mem-gb", type=float, default=6.5, help="cap on each child's CUDA caching allocator")
    ap.add_argument("--timeout", type=int, default=900, help="seconds before a child is killed")
    ap.add_argument(
        "--cache-dir",
        default=f"/dev/shm/{os.environ.get('USER', 'user')}/dls_probe_cache",
        help="node-local parent of TORCHINDUCTOR_CACHE_DIR and TRITON_CACHE_DIR, unless those are set",
    )
    ap.add_argument("--merge-only", action="store_true", help="only rebuild the report from <out-dir>/raw")
    ap.add_argument(
        "--note",
        action="append",
        default=[],
        help="a line for the report's Notes section, e.g. evidence from runs no longer in raw/; it is appended to "
        "<out-dir>/notes.txt, which every later merge reads",
    )
    ap.add_argument("--run-probe", choices=PROBES, help=argparse.SUPPRESS)
    ap.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    ap.add_argument("--run-index", type=int, default=0, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if not args.run_probe and not args.out_dir:
        ap.error("--out-dir is required")
    return args


def main(argv=None) -> int:
    args = _parse(argv)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%F %T")
    if args.run_probe:
        return _child(args)
    raw_dir, log_dir = args.out_dir / "raw", args.out_dir / "logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    if args.note:
        with open(args.out_dir / "notes.txt", "a") as fh:
            fh.writelines(" ".join(note.split()) + "\n" for note in args.note)
    if not args.merge_only:
        wanted = [p.strip().upper() for p in args.probes.split(",") if p.strip()]
        unknown = sorted(set(wanted) - set(PROBES))
        if unknown:
            logger.error("unknown probes: %s", unknown)
            return 2
        for probe in wanted:
            first = args.p3_first_run
            for run_index in range(first, first + args.p3_repeats) if probe == "P3" else (0,):
                _run_child(args, probe, run_index, raw_dir, log_dir)
    report = _merge(args, raw_dir)
    _write_json(args.out_dir / "dls_probes.json", report)
    (args.out_dir / "dls_probes.md").write_text(_markdown(report))
    failed = [k for k, v in report["probes"].items() if v.get("status") != "ok"]
    if failed:
        logger.warning("probes not ok: %s", failed)
    logger.info("wrote %s and %s", args.out_dir / "dls_probes.json", args.out_dir / "dls_probes.md")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
