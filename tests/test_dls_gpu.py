"""cheri_dls on the GPU: compiled passes, precision modes, step memory and determinism (acceptance criteria D1-D5).

Every test here is marked ``gpu`` and runs only with ``-m gpu``, inside a box, on one GPU:

    CUDA_VISIBLE_DEVICES=<least-loaded GPU> TORCHINDUCTOR_COMPILE_THREADS=1 taskset -c <4 cores> \\
        python -m pytest -q -m gpu tests/test_dls_gpu.py

The GPUs are shared with training jobs. So this module caps its process's caching allocator at GPU_MEM_CAP_GB
(torch.cuda.set_per_process_memory_fraction), every engine plans its steps within MEM_BUDGET_GB, and the D5 child
processes cap themselves the same way. TORCHINDUCTOR_COMPILE_THREADS=1 keeps Inductor from forking a compile pool.

D1. Compiled against eager, both on the GPU in the strict "fp32" precision mode (cuBLAS and cuDNN without TF32), on
    the tiny models A, B and C of tests/dls_tiny.py: 13 sequences, K = 5 references, shuffled or random. As revised
    after round 2 (analyses/attr_pilot/design/revisions.md), the bound is accuracy-relative. Against an fp64 truth on
    the same inputs and references (the engine on the CPU in float64, two_pass, dls_tiny.enable_float64), compiled
    must be no farther than 1.2 x eager, at the median and at the max of the per-pair raw rel-L2. A sanity ceiling
    holds the two paths within 1e-3 of each other per pair. The original bound, per-pair rel-L2 <= 1e-6 between the
    paths, is reported only: every fp32 evaluation of these models sits 1e-6 to 1e-4 per pair from the fp64 truth,
    because the rescale rule's secants (g(z_x) - g(z_r)) / (z_x - z_r) turn ulp-level differences in a GELU input into
    relative errors of about ulp / |z_x - z_r|. Bitwise equality is not expected either: Inductor's fused reductions
    sum in another order than eager's, and inside compiled code gelu_backward is a decomposition, within about an ulp
    of the eager kernel (probe P5). run()'s attributions are reported per sequence.
D2. The real architecture (C = 128, H = 256, 9 blocks, L = 2114, T = 557) with random weights, K = 20, steps of
    S = 4 peaks, in the default "reference" precision mode. Two run()s of 13 peaks each, each a warm-up step and 4
    steps:
    - fwd_c, bwd_c and epi_c are compiled exactly once each: Dynamo's unique_graphs counter is 3 after the first run
      and still 3 after the second, which runs under error_on_recompile. No graph break is recorded, and
      fullgraph=True would have raised on one.
    - Their generated code calls cuBLAS through extern_kernels.mm, 2 GEMMs per block each way (so the 9 blocks are
      unrolled into one graph), and holds no Triton GEMM (no tl.dot, no template kernel) and no convolution.
D3. On that engine, the most memory its second run allocates above what was allocated before it
    (torch.cuda.max_memory_allocated) is <= 1.3 x cheri_dls.step_bytes at S = 4: the analytic estimate of design §6.4,
    the same one auto-sizing uses.
D4. The forward's two-pass normalization (dw3, then (y - mu) * v with the rule's own statistics), eager and compiled,
    against cherimoya's Triton FusedDilatedConvNormFunc forward under torch.enable_grad, on zero-mean N(0, 1) input of
    shape (4, 2114, 128) at every dilation 1..256: max over elements of |ours - theirs| / max(1, |theirs|) <= 1e-5.
D5. deterministic=True: run() twice on one engine and once on a fresh engine, in each of two fresh processes, gives
    bitwise identical attributions, deltas and outputs, in the default "reference" mode. Each child process sets
    CUBLAS_WORKSPACE_CONFIG before CUDA initializes, which this pytest process, having used CUDA, cannot.

Also covered:
- The eager engine on the GPU against the eager engine on the CPU, both in fp32 mode with two_pass statistics, on the
  same accuracy-relative criterion as D1 (revised after round 2): the GPU no farther from the fp64 truth than 1.2 x the
  CPU at the median and the max of the per-pair raw rel-L2, and the two within 1e-3 per pair. The original bound,
  per-pair rel-L2 <= 1e-5 between them, is reported only, and so is the CPU engine's cpu_mirror path, the one checked
  against tangermeme.
- forward_stats="cpu_mirror" is refused on CUDA, and "auto" resolves to "two_pass" there.
- One host synchronization per step: D2's second run runs under torch.cuda.set_sync_debug_mode("error"), which
  raises on any implicit synchronization (.item(), a blocking copy). The engine's one wait per step, on the previous
  step's event, is counted and must equal the number of steps.
- seqs_per_step="auto" takes the largest S whose step_bytes fits mem_budget_gb and 0.6 of the available memory.
- Out of memory: an OutOfMemoryError in the warm-up or in a step halves S and retries. Injected in the warm-up, all
  steps run at the halved S and the results equal, bit for bit, a deterministic engine that ran at that S from the
  start. Injected in the second step, the first step's peaks keep S = 4 and the rest run at 2, bit for bit as the
  S = 4 and S = 2 engines compute them, and every peak is attributed once.
- The reference pool starts before the warm-up. On a small deterministic run with produced references (2 workers),
  whose warm-up is held to at least WARMUP_FLOOR_S, the first step does not wait for references (first_wait_s 0, no
  stall at all), and the results equal, bit for bit, the same engine's run with the same references injected. With
  the pool started after the warm-up, as before, the first step would wait for the workers to spawn.
- The pool draws chunks of the planned S before the warm-up. A warm-up that halves S = 5 to 2 re-cuts them: the run
  takes the 7 steps and 1 padded peak of an engine planned at S = 2, with its results bit for bit.

Measured values are written to the terminal.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from unittest import mock

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("tangermeme")
pytest.importorskip("cherimoya")

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "workflows" / d) for d in ("train", "attribution", "downstream")]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cheri_dls  # noqa: E402
import dls_refs  # noqa: E402
import dls_tiny  # noqa: E402
from cherimoya import Cherimoya  # noqa: E402

pytestmark = pytest.mark.gpu

GPU_MEM_CAP_GB = 6.0  # this process's caching allocator, in GB of 1e9 bytes
MEM_BUDGET_GB = 6.0  # every engine's mem_budget_gb
K_TINY = 5
HEAD_ROWS = 14  # the CPU engine tests' head rows, 2 x their oracle batch of 7
D1_ORIGINAL_MAX = 1e-6  # D1's original bound between compiled and eager, per pair; reported only since the revision
GPU_CPU_ORIGINAL_MAX = 1e-5  # the original GPU-CPU bound, per pair; reported only since the revision
ACCURACY_FACTOR = 1.2  # revised D1 and GPU-CPU: d(a, truth) <= 1.2 x d(b, truth) at the median and the max
SANITY_MAX = 1e-3  # revised D1 and GPU-CPU: per-pair raw rel-L2 between the two paths, at the max
RERUN_MAX = 1e-6  # two run()s of one engine in reference mode, per sequence
D3_FACTOR = 1.3
D4_MAX = 1e-5
REAL_K, REAL_S, REAL_PEAKS = 20, 4, 13  # 13 peaks in steps of 4: 3 full steps and 1 padded
REAL_LENGTH = 2114
WARMUP_FLOOR_S = 15.0  # the pool-overlap test holds its warm-up to at least this; a real-model compile takes 5-45 s
CHILD_TIMEOUT_S = 1800
RETAINED_MAX = 100e6  # bytes a run may leave allocated: cotangent, weights, cuBLAS workspaces; a step is 3 GB


def _report(config: pytest.Config, line: str) -> None:
    """Write a measurement to the terminal, past pytest's output capture."""
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(f"[cheri_dls gpu] {line}")


def _rows_rel_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """||a - b||_2 / max(||b||_2, 1e-30) for each row (first axis), in float64."""
    a, b = a.double().flatten(1), b.double().flatten(1)
    return (a - b).norm(dim=1) / b.norm(dim=1).clamp_min(1e-30)


def _cap_allocator() -> None:
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, GPU_MEM_CAP_GB * 1e9 / total), 0)


@pytest.fixture(scope="module", autouse=True)
def cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device is visible; run inside a box with CUDA_VISIBLE_DEVICES set")
    _cap_allocator()


def _window(length: int, width: int) -> tuple[int, int]:
    """attribute.py:84-86's centred window of `width` positions."""
    start = length // 2 - width // 2
    return start, start + width


_CASES: dict = {}


def _case(config: str, kind: str) -> tuple[torch.Tensor, torch.Tensor]:
    """A tiny config's 13 sequences, int8 (13, 4, L), and K_TINY references of one kind, float32."""
    if (config, kind) not in _CASES:
        X = dls_tiny.sequences(config)
        if kind == "shuffled":
            R = dls_tiny.shuffled_references(X, K_TINY)
        elif kind == "random":
            R = dls_tiny.random_references(X, K_TINY, seed=3)
        else:
            raise ValueError(kind)
        _CASES[config, kind] = (X, R)
    return _CASES[config, kind]


def _tiny_engine(config: str, **kwargs) -> cheri_dls.Engine:
    model = dls_tiny.tiny_model(config)
    kwargs.setdefault("device", "cuda")
    kwargs.setdefault("mem_budget_gb", MEM_BUDGET_GB)
    return cheri_dls.Engine(model, group=dls_tiny.CONFIGS[config].group, head_rows=HEAD_ROWS, **kwargs)


# D1 and GPU-CPU parity -----------------------------------------------------------------------------------------------

_RAW: dict = {}


def _raw(config: str, kind: str, which: str) -> cheri_dls.RawResult:
    """Engine.raw on a tiny config's case, computed once per session.

    which: "gpu_eager" and "gpu_compiled" (CUDA, fp32 mode), "cpu_two_pass" and "cpu_mirror" (CPU, fp32 mode), or
    "truth" (CPU, float64, two_pass: exact arithmetic for this comparison, design section 15.3's T64).
    """
    if (config, kind, which) not in _RAW:
        X, R = _case(config, kind)
        if which == "truth":
            model = dls_tiny.tiny_model(config, dtype=torch.float64)
            engine = cheri_dls.Engine(
                model, group=dls_tiny.CONFIGS[config].group, head_rows=HEAD_ROWS, forward_stats="two_pass"
            )
        else:
            kwargs = {
                "gpu_eager": dict(compile="none"),
                "gpu_compiled": dict(compile="default"),
                "cpu_two_pass": dict(device="cpu", forward_stats="two_pass"),
                "cpu_mirror": dict(device="cpu"),
            }[which]
            engine = _tiny_engine(config, precision="fp32", **kwargs)
        _RAW[config, kind, which] = engine.raw(X, R)
    return _RAW[config, kind, which]


def _pairs(result: cheri_dls.RawResult) -> torch.Tensor:
    return result.m.flatten(0, 1)


def _q(t: torch.Tensor) -> str:
    return f"median {float(t.median()):.3g} max {float(t.max()):.3g}"


def _accuracy_relative(name: str, a: torch.Tensor, b: torch.Tensor, between: torch.Tensor) -> list[str]:
    """The revised D1 / GPU-CPU criterion; returns the violations (none: it passes).

    a and b are two paths' per-pair raw rel-L2 to the fp64 truth, `between` the per-pair raw rel-L2 of path a against
    path b. a must be no farther from the truth than 1.2 x b at the median and at the max, and `between` must stay
    within the sanity ceiling at the max.
    """
    failures = []
    for stat, fn in (("median", torch.median), ("max", torch.max)):
        da, db = float(fn(a)), float(fn(b))
        if not da <= ACCURACY_FACTOR * db:
            failures.append(f"{name}: d(., truth) {stat} {da:.3g} > {ACCURACY_FACTOR} x {db:.3g}")
    if not float(between.max()) <= SANITY_MAX:
        failures.append(f"{name}: per-pair rel-L2 between the paths max {float(between.max()):.3g} > {SANITY_MAX:g}")
    return failures


@pytest.mark.parametrize("kind", ["shuffled", "random"])
@pytest.mark.parametrize("config", list(dls_tiny.CONFIGS))
def test_d1_compiled_matches_eager_in_fp32_mode(config, kind, pytestconfig):
    """D1 as revised after round 2: compiled no farther from the fp64 truth than 1.2 x eager, within 1e-3 of eager.

    The absolute numbers, and the original bound of 1e-6 between the paths, are written to the terminal.
    """
    X, R = _case(config, kind)
    L = X.shape[-1]
    start, end = _window(L, L // 4)
    eager, compiled, truth = (_raw(config, kind, which) for which in ("gpu_eager", "gpu_compiled", "truth"))
    rel = _rows_rel_l2(_pairs(compiled), _pairs(eager))
    to_truth = {name: _rows_rel_l2(_pairs(r), _pairs(truth)) for name, r in (("compiled", compiled), ("eager", eager))}
    run_e = _tiny_engine(config, precision="fp32", compile="none").run(X, start, end, references=R)
    run_c = _tiny_engine(config, precision="fp32", compile="default").run(X, start, end, references=R)
    rel_attr = _rows_rel_l2(run_c.attr, run_e.attr)
    d_out = max(
        float(((compiled.y_x - eager.y_x).abs() / eager.y_x.abs().clamp_min(1)).max()),
        float(((compiled.y_r - eager.y_r).abs() / eager.y_r.abs().clamp_min(1)).max()),
    )
    ratio = {
        s: float(f(to_truth["compiled"])) / float(f(to_truth["eager"]))
        for s, f in (("median", torch.median), ("max", torch.max))
    }
    line = (
        f"D1 config {config} {kind}: compiled vs eager, fp32 mode: raw per-pair rel-L2 {_q(rel)} (sanity max "
        f"{SANITY_MAX:g}; original bound {D1_ORIGINAL_MAX:g} {'met' if float(rel.max()) <= D1_ORIGINAL_MAX else 'not met'}"
        f", reported only; bitwise {'yes' if torch.equal(compiled.m, eager.m) else 'no'}); vs the fp64 truth: compiled "
        f"{_q(to_truth['compiled'])}, eager {_q(to_truth['eager'])}, ratio median {ratio['median']:.3f} max "
        f"{ratio['max']:.3f} (bound {ACCURACY_FACTOR}); run() attr per-sequence {_q(rel_attr)}; outputs max "
        f"|dy|/max(1,|y|) {d_out:.3g}"
    )
    _report(pytestconfig, line)
    for result in (eager, compiled):
        assert result.debug["precision"] == "fp32" and result.debug["device"].startswith("cuda")
        flags = result.debug["precision_flags"]
        assert flags["torch.get_float32_matmul_precision()"] == "highest"
        assert flags["torch.backends.cudnn.allow_tf32"] is False
    assert (eager.debug["compile"], compiled.debug["compile"], compiled.debug["forward_stats"]) == (
        "none",
        "default",
        "two_pass",
    )
    assert torch.equal(run_c.y_x, compiled.y_x) and torch.equal(run_c.y_r, compiled.y_r)  # one forward, run or raw
    assert torch.equal(run_e.y_x, eager.y_x) and torch.equal(run_e.y_r, eager.y_r)
    failures = _accuracy_relative("compiled vs eager", to_truth["compiled"], to_truth["eager"], rel)
    assert not failures, f"{failures}; {line}"


@pytest.mark.parametrize("config", list(dls_tiny.CONFIGS))
def test_gpu_eager_engine_matches_the_cpu_engine_in_fp32_mode(config, pytestconfig):
    """The same two_pass arithmetic on two devices; only GEMM blocking, reduction order and cuDNN differ.

    As revised after round 2: the GPU no farther from the fp64 truth than 1.2 x the CPU, and within 1e-3 of it per pair.
    The absolute numbers, the original bound of 1e-5 and the cpu_mirror path are written to the terminal.
    """
    gpu, cpu, mirror, truth = (
        _raw(config, "shuffled", which) for which in ("gpu_eager", "cpu_two_pass", "cpu_mirror", "truth")
    )
    assert mirror.debug["forward_stats"] == "cpu_mirror" and gpu.debug["forward_stats"] == "two_pass"
    assert cpu.debug["forward_stats"] == "two_pass" and cpu.debug["device"] == "cpu"
    rel = _rows_rel_l2(_pairs(gpu), _pairs(cpu))
    rel_mirror = _rows_rel_l2(_pairs(gpu), _pairs(mirror))
    to_truth = {name: _rows_rel_l2(_pairs(r), _pairs(truth)) for name, r in (("gpu", gpu), ("cpu", cpu))}
    ratio = {
        s: float(f(to_truth["gpu"])) / float(f(to_truth["cpu"]))
        for s, f in (("median", torch.median), ("max", torch.max))
    }
    line = (
        f"GPU eager vs CPU, config {config}, fp32 mode: per-pair rel-L2 vs CPU two_pass {_q(rel)} (sanity max "
        f"{SANITY_MAX:g}; original bound {GPU_CPU_ORIGINAL_MAX:g} "
        f"{'met' if float(rel.max()) <= GPU_CPU_ORIGINAL_MAX else 'not met'}, reported only); vs CPU cpu_mirror "
        f"(reported only) {_q(rel_mirror)}; vs the fp64 truth: GPU {_q(to_truth['gpu'])}, CPU {_q(to_truth['cpu'])}, "
        f"ratio median {ratio['median']:.3f} max {ratio['max']:.3f} (bound {ACCURACY_FACTOR})"
    )
    _report(pytestconfig, line)
    failures = _accuracy_relative("GPU vs CPU", to_truth["gpu"], to_truth["cpu"], rel)
    assert not failures, f"{failures}; {line}"


def test_cpu_mirror_is_refused_on_cuda():
    model = dls_tiny.tiny_model("A")
    with pytest.raises(ValueError, match="cpu_mirror"):
        cheri_dls.Engine(model, device="cuda", forward_stats="cpu_mirror")
    with pytest.raises(ValueError, match="cpu_mirror"):
        cheri_dls.resolve_forward_stats("cpu_mirror", "cuda")
    engine = cheri_dls.Engine(model, device="cuda")
    assert engine.forward_stats == "two_pass" and engine.compile == "default" and engine.precision == "reference"
    w = engine.weights
    z0 = cheri_dls.stem_forward(dls_tiny.sequences("A")[:2].float().cuda(), w.W0, w.b0)
    with pytest.raises(ValueError, match="cpu_mirror"):
        cheri_dls.forward_save(z0, w.blocks, w.rs, w.eps, "cpu_mirror")


def test_weights_are_cloned_on_the_engine_device():
    model = dls_tiny.tiny_model("A")
    for device in ("cuda", "cpu"):
        w = cheri_dls.Engine(model, device=device).weights
        assert w.device.type == device
        assert w.W0.data_ptr() != model.iconv.weight.data_ptr()
        assert w.blocks[0].W1.data_ptr() != model.blocks[0].linear1.weight.data_ptr()
        assert torch.equal(w.W0.cpu(), model.iconv.weight.detach())


# D2, D3 and one sync per step: the real architecture -----------------------------------------------------------------


@pytest.fixture(scope="module")
def real_runs():
    """Two run()s of an engine on the real architecture, recording compiles, generated code, memory and syncs."""
    from torch._dynamo.utils import counters
    from torch._inductor.graph import GraphLowering

    model = dls_tiny.redraw_weights(Cherimoya(compile=False, random_state=0, verbose=False), 0)
    X = dls_tiny.random_onehot(REAL_PEAKS, REAL_LENGTH, seed=11)
    R = dls_tiny.shuffled_references(X, REAL_K)
    start, end = _window(REAL_LENGTH, 400)
    engine = cheri_dls.Engine(model, device="cuda", seqs_per_step=REAL_S, mem_budget_gb=MEM_BUDGET_GB)

    codes: list[str] = []
    torch.cuda.empty_cache()
    at_start = torch.cuda.memory_allocated()
    counters.clear()
    with mock.patch.object(GraphLowering, "save_output_code", codes.append):
        first = engine.run(X, start, end, references=R)
    graphs_first = counters["stats"]["unique_graphs"]
    breaks = {str(k): v for k, v in counters["graph_break"].items()}

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    torch.cuda.set_sync_debug_mode("error")
    try:
        with torch._dynamo.config.patch(error_on_recompile=True):
            second = engine.run(X, start, end, references=R)
    finally:
        torch.cuda.set_sync_debug_mode(0)
    peak = torch.cuda.max_memory_allocated() - before
    out = {
        "engine": engine,
        "first": first,
        "second": second,
        "codes": codes,
        "graphs_first": graphs_first,
        "graphs_second": counters["stats"]["unique_graphs"],
        "breaks": breaks,
        "peak": peak,
        "before": before,
        "at_start": at_start,
        "estimate": cheri_dls.step_bytes(REAL_S, REAL_K, REAL_LENGTH, 128, 256, 9),
    }
    yield out
    out.clear()
    torch.cuda.empty_cache()


def test_d2_one_fullgraph_compile_per_function_with_extern_mm_and_no_convolution(real_runs, pytestconfig):
    first, second, codes = real_runs["first"], real_runs["second"], real_runs["codes"]
    n_blocks = 9
    mm_counts = [len(re.findall(r"extern_kernels\.(?:mm|bmm|addmm)\(", code)) for code in codes]
    _report(
        pytestconfig,
        f"D2 real architecture, S={REAL_S}, K={REAL_K}: unique graphs after run 1 {real_runs['graphs_first']}, after "
        f"run 2 (error_on_recompile) {real_runs['graphs_second']}; graph breaks {real_runs['breaks'] or 0}; "
        f"{len(codes)} generated modules with extern mm calls {mm_counts}; steps {first.meta['steps']} + "
        f"{second.meta['steps']}; compile (warm-up) {first.meta['compile_s']:.1f} s, then {second.meta['compile_s']:.2f} s",
    )
    assert first.meta["steps"] == second.meta["steps"] == 4 and first.meta["seqs_per_step"] == REAL_S
    assert real_runs["graphs_first"] == 3, "fwd_c, bwd_c and epi_c must each compile exactly once"
    assert real_runs["graphs_second"] == 3, "the second run must not recompile"
    assert real_runs["breaks"] == {}
    assert len(codes) == 3
    assert sorted(mm_counts) == [0, 2 * n_blocks, 2 * n_blocks]
    for code in codes:
        assert "tl.dot" not in code and "triton_tem" not in code and "triton_mm" not in code
        assert not re.search(r"convolution|conv1d|conv2d|cudnn", code, flags=re.IGNORECASE)


def test_a_compiling_run_gives_its_memory_back(real_runs, pytestconfig):
    """After the first run, which compiled, only the engine's own state stays allocated.

    Compiling leaves the traced step's activations (2.5 GB here) in reference cycles that only the cyclic garbage
    collector frees; the engine collects them. What may stay: the cotangent, the weights, cuBLAS workspaces.
    """
    held = real_runs["before"] - real_runs["at_start"]
    _report(pytestconfig, f"memory held after a compiling run: {held / 1e6:.1f} MB (tol {RETAINED_MAX / 1e6:g} MB)")
    assert held <= RETAINED_MAX


def test_d3_step_memory_is_within_the_estimate(real_runs, pytestconfig):
    peak, estimate = real_runs["peak"], real_runs["estimate"]
    meta = real_runs["second"].meta
    _report(
        pytestconfig,
        f"D3 real architecture, S={REAL_S}, K={REAL_K}: max_memory_allocated above the {real_runs['before'] / 1e6:.1f} "
        f"MB held before the run ({real_runs['at_start'] / 1e6:.1f} MB before the first run): {peak / 1e9:.3f} GB; "
        f"step_bytes estimate {estimate / 1e9:.3f} GB; ratio "
        f"{peak / estimate:.3f} (tol {D3_FACTOR}); the engine's own steady-loop measurement "
        f"{meta['step_peak_bytes'] / 1e9:.3f} GB",
    )
    assert meta["step_estimate_bytes"] == estimate
    assert peak <= D3_FACTOR * estimate


def test_one_host_sync_per_step(real_runs, pytestconfig):
    """The second run ran under sync debug mode "error": it raised on no implicit synchronization."""
    meta = real_runs["second"].meta
    _report(
        pytestconfig,
        f"syncs: {meta['host_syncs']} event waits for {meta['steps']} steps under sync debug mode 'error'; "
        f"steady {meta['steady_pairs_per_s']:.0f} pairs/s on the shared GPU",
    )
    assert meta["host_syncs"] == meta["steps"] == 4
    assert real_runs["first"].meta["host_syncs"] == real_runs["first"].meta["steps"]


def test_the_two_runs_agree(real_runs):
    """Same engine, same inputs, reference mode: cudnn.benchmark has its algorithm cached, so the outputs agree."""
    first, second = real_runs["first"], real_runs["second"]
    for name in ("y_x", "y_r"):
        assert torch.equal(getattr(first, name), getattr(second, name)), name
    rel = _rows_rel_l2(second.attr, first.attr)
    assert float(rel.max()) <= RERUN_MAX  # iconv's input gradient is not bitwise repeatable in cuDNN (probe P3)
    assert bool(torch.isfinite(first.attr).all())


# D4 ------------------------------------------------------------------------------------------------------------------


def test_d4_two_pass_normalization_matches_the_triton_forward(pytestconfig):
    from cherimoya.cheri import HAS_TRITON, FusedDilatedConvNormFunc

    assert HAS_TRITON
    gen = torch.Generator(device="cuda").manual_seed(4)
    x = torch.randn(4, REAL_LENGTH, 128, generator=gen, device="cuda")
    x = (x - x.mean(dim=(1, 2), keepdim=True)).contiguous()
    cw = torch.randn(3, 128, generator=gen, device="cuda") * 0.8
    eps = cheri_dls.CONV_NORM_EPS

    def ours(h: torch.Tensor, w: torch.Tensor, d: int) -> torch.Tensor:
        return cheri_dls.norm_two_pass(cheri_dls.dw3(h, w, d), eps)[0]

    compiled = torch.compile(ours, fullgraph=True, dynamic=False, recompile_limit=16)  # one graph per dilation
    lines, worst = [], 0.0
    for i in range(9):
        d = 2**i
        with torch.enable_grad():
            theirs = FusedDilatedConvNormFunc.apply(x.requires_grad_(), cw.requires_grad_(), d).detach()
        x, cw = x.detach(), cw.detach()
        with torch.no_grad():
            got = {"eager": ours(x, cw, d), "compiled": compiled(x, cw, d)}
        errs = {k: float(((v - theirs).abs() / theirs.abs().clamp_min(1)).max()) for k, v in got.items()}
        worst = max(worst, *errs.values())
        lines.append(f"d={d} eager {errs['eager']:.2g} compiled {errs['compiled']:.2g}")
    _report(pytestconfig, f"D4 max |ours - triton|/max(1,|triton|) (tol {D4_MAX:g}): " + "; ".join(lines))
    assert worst <= D4_MAX


# D5 ------------------------------------------------------------------------------------------------------------------

_D5_CHILD = textwrap.dedent(
    """
    import sys
    from pathlib import Path

    repo, out = Path(sys.argv[1]), sys.argv[2]
    sys.path[:0] = [str(repo / "workflows" / d) for d in ("train", "attribution", "downstream")]
    sys.path.insert(0, str(repo / "tests"))
    import cheri_dls
    import dls_tiny
    import torch

    assert not torch.cuda.is_initialized()
    model = dls_tiny.tiny_model("C")
    X = dls_tiny.sequences("C")
    R = dls_tiny.shuffled_references(X, 5)
    L = X.shape[-1]
    start = L // 2 - L // 8
    kwargs = dict(device="cuda", group=0, head_rows=14, seqs_per_step=4, deterministic=True, mem_budget_gb=4.0)
    engine = cheri_dls.Engine(model, **kwargs)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, 4e9 / total), 0)
    runs = [engine.run(X, start, start + L // 4, references=R) for _ in range(2)]
    runs.append(cheri_dls.Engine(model, **kwargs).run(X, start, start + L // 4, references=R))
    torch.save(
        {
            "results": [{k: getattr(r, k) for k in ("attr", "deltas", "y_x", "y_r")} for r in runs],
            "flags": runs[0].meta["precision_flags"],
            "steps": [r.meta["steps"] for r in runs],
        },
        out,
    )
    """
)


def test_d5_deterministic_reruns_are_bitwise_identical(tmp_path, pytestconfig):
    torch.cuda.empty_cache()
    env = os.environ.copy()
    env.pop("CUBLAS_WORKSPACE_CONFIG", None)  # the engine must set it itself, before CUDA initializes
    outs = []
    for i in range(2):
        out = tmp_path / f"child{i}.pt"
        proc = subprocess.run(
            [sys.executable, "-c", _D5_CHILD, str(REPO), str(out)],
            env=env,
            capture_output=True,
            text=True,
            timeout=CHILD_TIMEOUT_S,
        )
        assert proc.returncode == 0, proc.stderr[-4000:]
        outs.append(torch.load(out, weights_only=False))
    flags = outs[0]["flags"]
    reference = outs[0]["results"][0]
    same = [
        all(torch.equal(result[k], reference[k]) for k in reference) for child in outs for result in child["results"]
    ]
    _report(
        pytestconfig,
        f"D5 deterministic: 2 processes x 3 runs bitwise identical to the first: {same}; flags cudnn.deterministic "
        f"{flags['torch.backends.cudnn.deterministic']}, benchmark {flags['torch.backends.cudnn.benchmark']}, "
        f"CUBLAS_WORKSPACE_CONFIG {flags['$CUBLAS_WORKSPACE_CONFIG']}, matmul {flags['torch.get_float32_matmul_precision()']}",
    )
    assert flags["torch.backends.cudnn.deterministic"] is True and flags["torch.backends.cudnn.benchmark"] is False
    assert flags["$CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert flags["torch.get_float32_matmul_precision()"] == "high"  # deterministic, still the reference mode
    assert all(child["steps"] == [4, 4, 4] for child in outs)
    assert all(same)


# Step sizing and out-of-memory recovery ------------------------------------------------------------------------------


def test_auto_step_size_fits_the_budget():
    model = dls_tiny.redraw_weights(Cherimoya(compile=False, random_state=0, verbose=False), 0)
    per_peak = cheri_dls.step_bytes(1, REAL_K, REAL_LENGTH, 128, 256, 9)
    free, _ = torch.cuda.mem_get_info()
    for budget, expected in ((2.0, 2), (0.5, 1), (4.0, 4)):
        assert 0.6 * free > budget * 1e9, "the GPU has too little free memory for this test"
        engine = cheri_dls.Engine(model, device="cuda", mem_budget_gb=budget)
        assert max(1, int(budget * 1e9 // per_peak)) == expected
        assert engine.step_size(100, REAL_LENGTH, REAL_K) == expected
        assert engine.step_size(1, REAL_LENGTH, REAL_K) == 1
    fixed = cheri_dls.Engine(model, device="cuda", seqs_per_step=3, mem_budget_gb=0.5)
    assert fixed.step_size(100, REAL_LENGTH, REAL_K) == 3  # an explicit S is kept, with a warning


def _inject_oom(engine: cheri_dls.Engine, at_call: int, when_s: int) -> list[int]:
    """Make engine's at_call-th device step raise torch.cuda.OutOfMemoryError if it runs S = when_s peaks."""
    calls: list[int] = []
    step = engine._device_step

    def failing(tokens, S, K, G0, window):
        calls.append(S)
        if len(calls) == at_call and S == when_s:
            raise torch.cuda.OutOfMemoryError("injected by the test")
        return step(tokens, S, K, G0, window)

    engine._device_step = failing
    return calls


def test_oom_in_the_warm_up_halves_the_step_before_the_run(pytestconfig):
    X, R = _case("C", "shuffled")
    start, end = _window(X.shape[-1], X.shape[-1] // 4)
    kwargs = dict(precision="fp32", deterministic=True)
    engine = _tiny_engine("C", seqs_per_step=4, **kwargs)
    calls = _inject_oom(engine, at_call=1, when_s=4)
    got = engine.run(X, start, end, references=R)
    expected = _tiny_engine("C", seqs_per_step=2, **kwargs).run(X, start, end, references=R)
    _report(pytestconfig, f"OOM in the warm-up: device steps by S {calls}; oom record {got.meta['oom']}")
    assert got.meta["oom"] == [{"step": "warm-up", "from": 4, "to": 2}]
    assert got.meta["seqs_per_step"] == 2 and got.meta["steps"] == 7 and calls == [4, 2] + [2] * 7
    for name in ("attr", "deltas", "y_x", "y_r"):
        assert torch.equal(getattr(got, name), getattr(expected, name)), name


def test_oom_in_a_step_halves_the_step_and_retries_it(pytestconfig):
    """An OOM in step 2 of 4-peak steps: peaks 0-3 keep their step of 4, peaks 4-12 run in steps of 2.

    Deterministic engines in fp32 mode reproduce a step bit for bit, so peaks 0-3 must equal an S = 4 engine's and
    peaks 4-12 an S = 2 engine's: [4, 6), [6, 8), [8, 10), [10, 12) and [12, 13), padded, are the steps of both.
    """
    X, R = _case("C", "shuffled")
    kwargs = dict(precision="fp32", deterministic=True)
    engine = _tiny_engine("C", seqs_per_step=4, **kwargs)
    calls = _inject_oom(engine, at_call=3, when_s=4)  # the warm-up, step 1, then step 2 fails
    got = engine.raw(X, R)
    by_4 = _tiny_engine("C", seqs_per_step=4, **kwargs).raw(X, R)
    by_2 = _tiny_engine("C", seqs_per_step=2, **kwargs).raw(X, R)
    _report(pytestconfig, f"OOM in step 2: device steps by S {calls}; oom record {got.debug['oom']}")
    assert got.debug["oom"] == [{"step": 2, "from": 4, "to": 2}]
    assert calls == [4, 4, 4, 2, 2, 2, 2, 2]
    assert got.debug["steps"] == 6 and got.debug["padded_peaks"] == 1 and got.debug["seqs_per_step"] == 2
    assert got.debug["host_syncs"] == got.debug["steps"]
    for name in ("m", "y_x", "y_r", "deltas"):
        ours = getattr(got, name)
        assert torch.equal(ours[:4], getattr(by_4, name)[:4]), name
        assert torch.equal(ours[4:], getattr(by_2, name)[4:]), name


# The reference pool and the warm-up ---------------------------------------------------------------------------------


def _hold_warm_up(engine: cheri_dls.Engine, seconds: float) -> dict:
    """Make engine's warm-up last at least `seconds`, as a cold compile of the real model does; returns its timings."""
    timing: dict = {}
    warm_up = engine._warm_up

    def held(*args, **kwargs):
        began = time.perf_counter()
        S = warm_up(*args, **kwargs)
        timing["warm_up_s"] = time.perf_counter() - began
        time.sleep(max(0.0, seconds - timing["warm_up_s"]))
        timing["held_s"] = time.perf_counter() - began
        return S

    engine._warm_up = held
    return timing


def test_the_reference_pool_draws_during_the_warm_up(pytestconfig):
    """Produced references, 2 workers: the first step finds its references drawn, and the results are unchanged."""
    X, R = _case("C", "shuffled")  # dls_tiny's shuffles are deep_lift_shap's own draw, with random_state 0
    start, end = _window(X.shape[-1], X.shape[-1] // 4)
    engine = _tiny_engine(
        "C", seqs_per_step=4, precision="fp32", deterministic=True, n_shuffles=K_TINY, random_state=0, ref_workers=2
    )
    timing = _hold_warm_up(engine, WARMUP_FLOOR_S)
    produced = engine.run(X, start, end)
    del engine._warm_up  # the injected run warms up as usual
    stats = produced.meta["reference_stats"]
    injected = engine.run(X, start, end, references=R)
    _report(
        pytestconfig,
        f"pool during the warm-up: warm-up {timing['warm_up_s']:.1f} s, held to {timing['held_s']:.1f} s; reference "
        f"stats {stats}; steps {produced.meta['steps']}, padded {produced.meta['padded_peaks']}",
    )
    assert produced.meta["references"] == "produced" and injected.meta["references"] == "injected"
    assert stats["first_wait_s"] == 0.0 and stats["stalls"] == 0 and stats["chunks"] == 4
    for name in ("attr", "deltas", "y_x", "y_r"):
        assert torch.equal(getattr(produced, name), getattr(injected, name)), name
    for key in ("steps", "padded_peaks", "seqs_per_step", "host_syncs"):
        assert produced.meta[key] == injected.meta[key], key


def test_a_warm_up_halving_recuts_the_pools_chunks(pytestconfig):
    """S = 5 halves to 2 in the warm-up, after the pool started drawing chunks of 5 peaks."""
    X, R = _case("C", "shuffled")
    start, end = _window(X.shape[-1], X.shape[-1] // 4)
    kwargs = dict(precision="fp32", deterministic=True, n_shuffles=K_TINY, random_state=0)
    engine = _tiny_engine("C", seqs_per_step=5, ref_workers=2, **kwargs)
    calls = _inject_oom(engine, at_call=1, when_s=5)
    got = engine.run(X, start, end, keep_references=list(range(len(X))))
    expected = _tiny_engine("C", seqs_per_step=2, **kwargs).run(X, start, end, references=R)
    _report(
        pytestconfig,
        f"warm-up halving 5 -> 2 with the pool drawing chunks of 5: device steps by S {calls}; steps "
        f"{got.meta['steps']}, padded {got.meta['padded_peaks']} (an S = 2 engine: {expected.meta['steps']}, "
        f"{expected.meta['padded_peaks']})",
    )
    assert got.meta["oom"] == [{"step": "warm-up", "from": 5, "to": 2}]
    assert calls == [5, 2] + [2] * 7
    steps = (got.meta["steps"], got.meta["padded_peaks"])
    assert steps == (expected.meta["steps"], expected.meta["padded_peaks"]) == (7, 1)
    np.testing.assert_array_equal(got.references, dls_refs.onehot_to_tokens(R))
    for name in ("attr", "deltas", "y_x", "y_r"):
        assert torch.equal(getattr(got, name), getattr(expected, name)), name
