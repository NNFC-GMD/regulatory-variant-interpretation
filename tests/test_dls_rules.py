"""cheri_dls's backward rules against tangermeme's and cherimoya's own functions, on CPU (acceptance criterion C2).

C2 checks each piece of the hand-written backward on its own, against the reference function it replaces, with the
same inputs on both sides, in float64:
(i) gelu_rescale, the rescale rule through GELU(tanh), against tangermeme's _nonlinear
    (tangermeme/_deep_lift_utils.py:116-128) at |dz| = 0, 5e-7, 9.999e-7, exactly 1e-6, 1.0001e-6, 1e-3 and 1.
    It must take the same branch (the gradient where |dz| < 1e-6, strictly; the secant otherwise), with values
    within 1 ulp.
(ii) ln_rule, the normalization rule inside bwd_block, against tangermeme's _layer_normalization_helper on
    cherimoya's _ConvNormView, as conv_norm_op calls it (cherimoya/deep_lift_shap.py:137-150): <= 1e-13 relative.
(iii) dw3 and its transpose dw3_t against cherimoya's _cheri_conv and autograd through it: <= 1e-14 relative.
(iv) counts_cotangent against autograd through the real LogCountWrapper(ControlWrapper(model)): torch.equal.
(v) the epilogue's window of hypothetical scores against tangermeme's hypothetical_attributions, sliced: torch.equal.
(vi) the host-side mean over references against tangermeme's full-length stack-and-mean, sliced: torch.equal.

The stubs. tangermeme's rules read the activations off the module they correct: _nonlinear reads `input` and
`output`; _layer_normalization_helper reads `input`, `eps`, `normalized_shape` and `weight`; conv_norm_op reads
`input`, `conv_weight` and `dilation`. Each stub holds tangermeme's joint layout: one copy of the sequence per pair,
then the references (tangermeme/deep_lift_shap.py:717). Pairs are example-major. Only the first half of a rule's
output, the sequence half, is compared: that is all the engine computes.

GELU on CPU. The CPU kernels for GELU and its gradient compute a vectorized body and a scalar tail, and the two round
differently: on this AVX512 node about 5% of elements differ by an ulp (measured). Which elements fall in the tail
depends on the tensor's shape and on how ATen splits it across threads. The rule's own arithmetic is what (i) tests,
so its tensors keep every contiguous run a multiple of 16 elements and stay below ATen's parallel grain size (32768
elements): every element then takes the vectorized path on both sides. The composite tests of bwd_block and
stem_rule run at the default thread count on the tiny models' real shapes, and are bounded by relative error.

Beyond C2, the file also checks:
- that bwd_block and stem_rule compose the rules as tangermeme does, for one block or the stem at a time;
- that input_multipliers is autograd's input gradient of iconv;
- that the epilogue's convergence deltas are tangermeme's expression;
- the argument checks.

Measured values are written to the terminal.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tangermeme")
pytest.importorskip("cherimoya")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflows" / "cherimoya"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cheri_dls  # noqa: E402
import dls_tiny  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from cherimoya import ControlWrapper, LogCountWrapper  # noqa: E402
from cherimoya.cheri import CONV_NORM_EPS, _cheri_conv  # noqa: E402
from cherimoya.deep_lift_shap import _ConvNormView, conv_norm_op  # noqa: E402
from tangermeme._deep_lift_utils import _layer_normalization_helper, _nonlinear  # noqa: E402

F64 = torch.float64
DTYPES = [pytest.param(torch.float32, id="float32"), pytest.param(torch.float64, id="float64")]
CONFIG_NAMES = list(dls_tiny.CONFIGS)
S, K = 2, 3  # sequences and references per sequence in the composite tests: R = S + S*K = 8 rows

LN_TOL = 1e-13  # C2 (ii)
DW3_TOL = 1e-14  # C2 (iii)
COMPOSITE_TOL = 1e-13  # bwd_block and stem_rule: no design bound; C2's loosest rule bound, see the tests


def _report(config: pytest.Config, line: str) -> None:
    """Write a measurement to the terminal, past pytest's output capture."""
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(f"[cheri_dls] {line}")


@contextlib.contextmanager
def _threads(n: int):
    previous = torch.get_num_threads()
    torch.set_num_threads(n)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _ulps(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Elementwise distance in units in the last place between two float64 tensors, as float64.

    The sign-magnitude bit patterns are mapped onto one monotonic integer line, on which adjacent doubles differ by 1
    and -0.0 equals +0.0.
    """
    assert a.dtype == b.dtype == F64
    lowest = torch.iinfo(torch.int64).min

    def line(t: torch.Tensor) -> torch.Tensor:
        bits = t.contiguous().view(torch.int64)
        return torch.where(bits < 0, lowest - bits, bits).to(F64)

    return (line(a) - line(b)).abs()


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    """max over rows (the first axis) of ||a - b||_2 / ||b||_2."""
    a, b = a.flatten(1), b.flatten(1)
    return float(((a - b).norm(dim=1) / b.norm(dim=1)).max())


def _joint(t: torch.Tensor) -> torch.Tensor:
    """Our rows [x_0..x_{S-1}, r_00..r_{S-1,K-1}] in tangermeme's joint layout: cat([x once per pair, references])."""
    return torch.cat([t[:S].repeat_interleave(K, dim=0), t[S:]])


def _joint_batch(config: str, dtype: torch.dtype) -> torch.Tensor:
    """S sequences (one random, the AT-rich one), then their K dinucleotide shuffles: (S + S*K, 4, L)."""
    X = dls_tiny.sequences(config)[[0, 12]]
    refs = dls_tiny.shuffled_references(X, K)
    return torch.cat([X.float(), refs.flatten(0, 1)]).to(dtype)


def _gelu_reference(z_joint: torch.Tensor, g_joint: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """tangermeme's _nonlinear on an nn.GELU(tanh) stub over the joint rows: (result, grad_input, module output).

    grad_input is what the full backward hook receives: nn.GELU's own autograd gradient of g_joint.
    """
    leaf = z_joint.detach().clone().requires_grad_()
    with torch.enable_grad():
        out = torch.nn.GELU(approximate="tanh")(leaf)
        (grad_input,) = torch.autograd.grad(out, leaf, g_joint)
    stub = SimpleNamespace(input=z_joint, output=out.detach())
    (result,) = _nonlinear(stub, (grad_input,), (g_joint,))
    return result, grad_input, out.detach()


# (i) ------------------------------------------------------------------------------------------------------------

DZ = [0.0, 5e-7, 9.999e-7, 1e-6, 1.0001e-6, 1e-3, 1.0]
DZ_IDS = ["0", "5e-7", "9.999e-7", "exactly-1e-6", "1.0001e-6", "1e-3", "1"]


@pytest.mark.parametrize("dz", DZ, ids=DZ_IDS)
def test_c2_i_gelu_rescale_matches_tangermeme_nonlinear(dz, pytestconfig):
    """|z_x - z_r| = dz everywhere, with random signs. The gradient branch is the strict |dz| < 1e-6 (rule :122).

    |dz| exactly 1e-6 needs base points below about 1.9e-6: 1e-6's lowest mantissa bit is 2**-72, so z_x - z_r is
    exact only while |z| < 2**-19. Those base points are multiples of 2**-40 below 0.85e-6. Elsewhere they are
    uniform in [-4, 4].

    The branch is checked by value. The expected branch's value (autograd's grad_input, or G times tangermeme's own
    secant) must be within 1 ulp of ours and of the reference's. Where the other branch's value is more than 2 ulps
    away, ours is therefore not the other branch. At least 90% of elements must be so distinguishable.
    """
    n_s, n_k, inner = 3, 4, (16, 8)  # 128-element rows: see the module docstring on GELU
    P = n_s * n_k
    gen = torch.Generator().manual_seed(int(dz * 1e7) + 7)
    if dz == 1e-6:
        zx = torch.randint(-900_000, 900_001, (n_s, *inner), generator=gen).to(F64) * 2.0**-40
    else:
        zx = torch.rand((n_s, *inner), generator=gen, dtype=F64) * 8 - 4
    sign = torch.randint(0, 2, (n_s, n_k, *inner), generator=gen).to(F64) * 2 - 1
    zr = zx.unsqueeze(1) - sign * dz
    din = zx.unsqueeze(1) - zr
    small = dz < cheri_dls.RESCALE_EPS
    if dz == 1e-6:
        assert torch.equal(din.abs(), torch.full_like(din, 1e-6))
    assert bool(((din.abs() < 1e-6) == small).all())

    G = torch.randn((n_s, n_k, *inner), generator=gen, dtype=F64)
    G_ref_half = torch.randn((P, *inner), generator=gen, dtype=F64)
    z_joint = torch.cat([zx.repeat_interleave(n_k, dim=0), zr.flatten(0, 1)])
    g_joint = torch.cat([G.flatten(0, 1), G_ref_half])
    ref, grad_input, out = _gelu_reference(z_joint, g_joint)
    ref = ref[:P]

    z = torch.cat([zx, zr.flatten(0, 1)])
    ours = cheri_dls.gelu_rescale(G, z, n_s, n_k).reshape(P, *inner)
    assert ours.dtype == F64 and bool(torch.isfinite(ours).all())

    tangent = grad_input[:P]
    secant = g_joint[:P] * ((out[:P] - out[P:]) / (z_joint[:P] - z_joint[P:]))  # the rule's own expressions
    expected, other = (tangent, secant) if small else (secant, tangent)
    ulps = _ulps(ours, ref)
    max_ulps, bitwise = float(ulps.max()), torch.equal(ours, ref)
    assert float(_ulps(ref, expected).max()) <= 1, "the reference did not take the expected branch"
    assert float(_ulps(ours, expected).max()) <= 1
    if dz == 0:
        assert bool(torch.isnan(other).all())  # 0/0: only the gradient branch is defined
        distinguishable = 1.0
    else:
        far = _ulps(expected, other) > 2
        distinguishable = float(far.to(F64).mean())
        assert distinguishable >= 0.9
        assert bool((_ulps(ours, other)[far] > 1).all())

    # Half-independence (R8): the sequence half reads only sequence-half cotangents.
    again, _, _ = _gelu_reference(z_joint, torch.cat([G.flatten(0, 1), -3 * G_ref_half]))
    assert torch.equal(again[:P], ref)

    _report(
        pytestconfig,
        f"C2(i) GELU |dz|={dz:g}: branch {'gradient' if small else 'secant'}, max {max_ulps:g} ulp vs _nonlinear "
        f"(bitwise {'yes' if bitwise else 'no'}), {distinguishable:.1%} of elements tell the branches apart",
    )
    assert max_ulps <= 1


# (ii) -----------------------------------------------------------------------------------------------------------


def _ln_reference(y: torch.Tensor, Gn: torch.Tensor, G_ref_half: torch.Tensor) -> torch.Tensor:
    """_layer_normalization_helper(_ConvNormView(cat(y_x, y_r)), None, (cat(g, g_r),))[:P], conv_norm_op's call."""
    view = _ConvNormView(_joint(y))
    assert (view.eps, view.normalized_shape, view.weight) == (CONV_NORM_EPS, list(y.shape[1:]), None)
    (result,) = _layer_normalization_helper(view, None, (torch.cat([Gn.flatten(0, 1), G_ref_half]),))
    return result[: S * K]


@pytest.mark.parametrize("config", CONFIG_NAMES)
def test_c2_ii_ln_rule_matches_layer_normalization_helper(config, pytestconfig):
    """Every block's real conv output from a float64 forward, plus a synthetic y far from zero mean.

    The synthetic rows have mean up to 50 and std down to 0.01, so the mean-centring and the variance term matter.
    """
    cfg = dls_tiny.CONFIGS[config]
    w = cheri_dls.CheriWeights.from_model(dls_tiny.tiny_model(config, dtype=F64))
    with torch.no_grad():
        fwd = cheri_dls.model_forward(_joint_batch(config, F64), w, forward_stats="cpu_mirror", group=cfg.group)
    gen = torch.Generator().manual_seed(11)
    L, C = cfg.length, w.C
    R = S + S * K
    ys = [saved.y for saved in fwd.saved]
    offset = torch.rand((R, 1, 1), generator=gen, dtype=F64) * 100 - 50
    scale = 10 ** (torch.rand((R, 1, 1), generator=gen, dtype=F64) * 3 - 2)
    ys.append(offset + scale * torch.randn((R, L, C), generator=gen, dtype=F64))

    errs = []
    for y in ys:
        mu, _, v = cheri_dls.rule_stats(y, w.eps)
        Gn = torch.randn((S, K, L, C), generator=gen, dtype=F64)
        G_ref_half = torch.randn((S * K, L, C), generator=gen, dtype=F64)
        ours = cheri_dls.ln_rule(Gn, y, mu.view(R), v.view(R), S, K)
        assert ours.shape == (S, K, L, C) and ours.dtype == F64
        ref = _ln_reference(y, Gn, G_ref_half)
        assert torch.equal(_ln_reference(y, Gn, 2 * G_ref_half + 1), ref)  # half-independence (R8)
        errs.append(_rel_l2(ours.flatten(0, 1), ref))
    _report(
        pytestconfig,
        f"C2(ii) LN rule config {config}: per-pair rel-L2 vs _layer_normalization_helper, "
        f"blocks {', '.join(f'{e:.3g}' for e in errs[:-1])}; synthetic {errs[-1]:.3g} (tol {LN_TOL:g})",
    )
    assert max(errs) <= LN_TOL


# (iii) ----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(2114, 128), (100, 8)], ids=["L2114xC128", "L100xC8"])
@pytest.mark.parametrize("d", [2**i for i in range(9)])
def test_c2_iii_dw3_and_its_transpose_match_cheri_conv(d, shape, pytestconfig):
    """dw3 against _cheri_conv, and dw3_t against autograd through _cheri_conv, which is how conv_norm_op pushes the
    rule's multipliers back to the block input (cherimoya/deep_lift_shap.py:141-148). L=100 with d >= 128 puts both
    outer taps outside the sequence everywhere.
    """
    L, C = shape
    gen = torch.Generator().manual_seed(d)
    cw = torch.randn((3, C), generator=gen, dtype=F64)
    h = torch.randn((S + S * K, L, C), generator=gen, dtype=F64)
    y_err = _rel_l2(cheri_dls.dw3(h, cw, d), _cheri_conv(h, cw, d))

    G = torch.randn((S, K, L, C), generator=gen, dtype=F64)
    leaf = torch.randn((S * K, L, C), generator=gen, dtype=F64, requires_grad=True)
    with torch.enable_grad():
        (expected,) = torch.autograd.grad(_cheri_conv(leaf, cw, d), leaf, G.flatten(0, 1))
    ours = cheri_dls.dw3_t(G, cw, d)
    assert ours.shape == G.shape
    t_err = _rel_l2(ours.flatten(0, 1), expected)
    _report(
        pytestconfig,
        f"C2(iii) d={d} L={L} C={C}: per-row rel-L2 dw3 {y_err:.3g}, transpose {t_err:.3g} (tol {DW3_TOL:g})",
    )
    assert y_err <= DW3_TOL and t_err <= DW3_TOL


# (iv) -----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "config, group", [("A", 0), ("B", 0), ("B", 1), ("B", None), ("C", 0)], ids=["A", "B-g0", "B-g1", "B-None", "C"]
)
def test_c2_iv_counts_cotangent_is_autograd_through_the_real_head(config, group, dtype):
    """The gradient of the attributed count with respect to the last block's output, from the real module.

    The module runs as deep_lift_shap runs it, LogCountWrapper(ControlWrapper(model), group)(X)[:, 0], and autograd
    differentiates the sum. In float64 the module is dls_tiny's enable_float64 one. Config B has a control track and
    two groups; group None attributes target 0.
    """
    model = dls_tiny.tiny_model(config, dtype=dtype)
    w = cheri_dls.CheriWeights.from_model(model)
    X1h = _joint_batch(config, dtype)
    L, T = X1h.shape[-1], w.T
    captured = {}
    handle = model.blocks[-1].register_forward_hook(lambda m, args, out: captured.update(hL=out))
    try:
        with torch.enable_grad():
            y = LogCountWrapper(ControlWrapper(model), group=group)(X1h.clone().requires_grad_())[:, 0]
            (expected,) = torch.autograd.grad(y.sum(), captured["hL"])
    finally:
        handle.remove()
    assert expected.shape == (len(X1h), L, w.C)

    for rows in (len(X1h), cheri_dls.DEFAULT_HEAD_ROWS):
        ours = cheri_dls.counts_cotangent(w, group=group, length=L, rows=rows)
        assert ours.shape == (L, w.C) and ours.dtype == dtype and ours.is_contiguous()
        assert torch.equal(expected, ours.expand_as(expected))

    # What it is: the count Linear's row over the trimmed mean, zero outside [T, L - T) (cherimoya.py:458, 470).
    row = w.Wl[0 if group is None else group, : w.C]
    assert torch.equal(ours[T : L - T], (row / (L - 2 * T)).expand(L - 2 * T, w.C))
    assert not bool(ours[:T].any()) and not bool(ours[L - T :].any())


def test_counts_cotangent_arguments():
    w = cheri_dls.CheriWeights.from_model(dls_tiny.tiny_model("B"))
    with torch.no_grad():  # the engine runs under no_grad; the cotangent turns grad mode on for itself
        assert cheri_dls.counts_cotangent(w, group=1, length=200).shape == (200, 8)
    with torch.inference_mode(), pytest.raises(RuntimeError, match="inference"):
        cheri_dls.counts_cotangent(w, group=1, length=200)
    with pytest.raises(ValueError, match="rows"):
        cheri_dls.counts_cotangent(w, group=1, length=200, rows=0)
    with pytest.raises(ValueError, match="group"):
        cheri_dls.counts_cotangent(w, group=2, length=200)
    with pytest.raises(ValueError, match="trimming"):
        cheri_dls.counts_cotangent(w, group=1, length=2 * w.T)


# (v) and (vi) -----------------------------------------------------------------------------------------------------


def _window(L: int, W: int) -> tuple[int, int]:
    """The attribution window of cherimoya_cli/commands/attribute.py:84-86."""
    start = L // 2 - W // 2
    return start, start + W


def _pairs(L: int, n_s: int, n_k: int, dtype: torch.dtype, seed: int):
    """Sequences, one-hot references with a few N (all-zero) columns, multipliers and outputs for n_s * n_k pairs.

    Returns the joint rows X1h (n_s + n_s*n_k, 4, L), m (n_s*n_k, 4, L) and y (n_s + n_s*n_k,). m holds some exact
    zeros of both signs.
    """
    gen = torch.Generator().manual_seed(seed)
    X = dls_tiny.random_onehot(n_s, L, seed=seed).to(dtype)
    refs = dls_tiny.random_references(X, n_k, seed=seed + 1).to(dtype).flatten(0, 1)
    refs[:, :, torch.randint(0, L, (5,), generator=gen)] = 0
    m = torch.randn((n_s * n_k, 4, L), generator=gen, dtype=F64).to(dtype)
    m[m.abs() < 0.05] = 0.0
    m[:, :, ::7] *= -1  # negative zeros among them
    y = torch.randn(n_s + n_s * n_k, generator=gen, dtype=F64).to(dtype)
    return torch.cat([X, refs]), m, y


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("L, W", [(2114, 400), (256, 64), (200, 50), (2114, 100)])
def test_c2_v_window_hyp_is_hypothetical_attributions_sliced(L, W, dtype):
    """For one-hot (or N) references both sides round once, to fl(m_i - m_b), so the window is bitwise tangermeme's
    full-length projection (tangermeme/deep_lift_shap.py:93-103), sliced afterwards.
    """
    n_s, n_k = 2, 3
    X1h, m, y = _pairs(L, n_s, n_k, dtype, seed=L + W)
    start, end = _window(L, W)
    hyp, _ = cheri_dls.epilogue(m, X1h, y[:n_s], y[n_s:].view(n_s, n_k), start, end, n_s, n_k)
    assert hyp.shape == (n_s, n_k, 4, W) and hyp.dtype == dtype and hyp.is_contiguous()
    Xp = X1h[:n_s].repeat_interleave(n_k, dim=0)
    (full,) = dls_tiny.TDLS.hypothetical_attributions((m,), (Xp,), (X1h[n_s:],))
    assert torch.equal(hyp.flatten(0, 1), full[..., start:end])


def test_epilogue_deltas_are_tangermemes_expression(pytestconfig):
    """|(y_x - y_r) - sum((x - r) * m)| over the full length (tangermeme/deep_lift_shap.py:739-742)."""
    for dtype in (torch.float32, F64):
        n_s, n_k, L = 3, 4, 256
        X1h, m, y = _pairs(L, n_s, n_k, dtype, seed=3)
        _, delta = cheri_dls.epilogue(m, X1h, y[:n_s], y[n_s:].view(n_s, n_k), 96, 160, n_s, n_k)
        assert delta.shape == (n_s, n_k) and delta.dtype == dtype
        y_joint = torch.cat([y[:n_s].repeat_interleave(n_k), y[n_s:]])
        Xp, r = X1h[:n_s].repeat_interleave(n_k, dim=0), X1h[n_s:]
        expected = abs(torch.sub(*torch.chunk(y_joint, 2)) - torch.sum((Xp - r) * m, dim=(1, 2)))
        bitwise = torch.equal(delta.flatten(), expected)
        _report(
            pytestconfig,
            f"epilogue deltas {str(dtype)[6:]}: max |diff| {float((delta.flatten() - expected).abs().max()):.3g} "
            f"(bitwise {'yes' if bitwise else 'no'})",
        )
        assert bitwise


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "n_k, L, W", [(20, 2114, 400), (4, 256, 64), (5, 200, 50), (20, 2114, 100), (5, 256, 256), (3, 2114, 2100)]
)
def test_c2_vi_mean_over_references_is_tangermemes_stack_mean_sliced(n_k, L, W, dtype, pytestconfig):
    """tangermeme averages a sequence's K full-length hypothetical scores: torch.stack(list of (4, L)).mean(dim=0)
    (tangermeme/deep_lift_shap.py:760-770). Ours averages the windows that the epilogue hands the host.

    The reference side runs at 1, 4 and 8 threads. The shapes include the real one (K=20, L=2114, W=400), the CLI
    tests' (K=4, L=256, W=64), a window that is not a multiple of 16 positions (W=100), the full length, and a window
    reaching within 7 positions of the end.

    Also reported, not asserted: the design's literal op, the mean over each contiguous (K, 4, W) window block alone.
    """
    n_s = 2
    X1h, m, y = _pairs(L, n_s, n_k, dtype, seed=n_k * L + W)
    start, end = _window(L, W)
    hyp, _ = cheri_dls.epilogue(m, X1h, y[:n_s], y[n_s:].view(n_s, n_k), start, end, n_s, n_k)
    Xp = X1h[:n_s].repeat_interleave(n_k, dim=0)
    (full,) = dls_tiny.TDLS.hypothetical_attributions((m,), (Xp,), (X1h[n_s:],))
    attr_ = list(full)  # tangermeme's per-pair list (deep_lift_shap.py:760)

    ours = cheri_dls.mean_over_references(hyp, start, L)
    assert ours.shape == (n_s, 4, W) and ours.dtype == dtype
    for threads in (1, 4, 8):
        with _threads(threads):
            ref = torch.stack(
                [torch.stack(attr_[s * n_k : (s + 1) * n_k]).mean(dim=0)[:, start:end] for s in range(n_s)]
            )
        assert torch.equal(ours, ref), f"{threads} threads"

    literal = torch.stack([hyp[s].mean(dim=0) for s in range(n_s)])
    _report(
        pytestconfig,
        f"C2(vi) {str(dtype)[6:]} K={n_k} L={L} W={W}: mean_over_references bitwise at 1/4/8 reference threads; "
        f"window-only (K,4,W).mean(0) differs in {int((literal != ref).sum())} of {ref.numel()} elements",
    )


def test_epilogue_and_mean_arguments():
    X1h, m, y = _pairs(64, 2, 3, F64, seed=0)
    for start, end in ((10, 10), (-1, 20), (0, 65), (40, 30)):
        with pytest.raises(ValueError, match="window"):
            cheri_dls.epilogue(m, X1h, y[:2], y[2:].view(2, 3), start, end, 2, 3)
    hyp, _ = cheri_dls.epilogue(m, X1h, y[:2], y[2:].view(2, 3), 20, 40, 2, 3)
    with pytest.raises(ValueError, match="window"):
        cheri_dls.mean_over_references(hyp, 50, 64)
    with pytest.raises(ValueError, match="CPU"):
        cheri_dls.mean_over_references(hyp.to("meta"), 20, 64)
    with pytest.raises(ValueError, match="rows"):
        cheri_dls.split_rows(X1h[:-1], 2, 3)


# Composites: bwd_block, stem_rule and input_multipliers --------------------------------------------------------------


def _reference_block_backward(
    G: torch.Tensor, G_ref_half: torch.Tensor, h: torch.Tensor, u: torch.Tensor, block, rs: float
) -> torch.Tensor:
    """One CheriBlock's backward as tangermeme runs it, on the joint rows, sequence half returned.

    The block computes ``h + linear2(GELU(linear1(conv(h)))) * rs`` (cherimoya/cheri.py:1179-1181). Ordinary gradients
    run through the residual, the scale and both Linears; tangermeme's _nonlinear corrects the GELU, and cherimoya's own
    conv_norm_op corrects FusedDilatedConvNorm. h and u are the block input and the linear1 output over our rows.
    """
    P = S * K
    g_out = torch.cat([G.flatten(0, 1), G_ref_half])
    u_joint = _joint(u)
    with torch.enable_grad():
        q = torch.zeros_like(u_joint, requires_grad=True)  # linear2's input; its value does not enter the gradient
        (g_q,) = torch.autograd.grad(F.linear(q, block.W2) * rs, q, g_out)
    g_u, _, _ = _gelu_reference(u_joint, g_q)
    with torch.enable_grad():
        n = torch.zeros(u_joint.shape[:-1] + (block.W1.shape[1],), dtype=u.dtype, requires_grad=True)
        (g_n,) = torch.autograd.grad(F.linear(n, block.W1), n, g_u)
    stub = SimpleNamespace(input=_joint(h), conv_weight=block.cw, dilation=block.d)
    (g_h,) = conv_norm_op(stub, None, (g_n,))
    return (g_out + g_h)[:P]


@pytest.mark.parametrize("config", CONFIG_NAMES)
def test_bwd_block_composes_the_reference_rules(config, pytestconfig):
    """Each block of a float64 tiny model, on its real inputs from our forward, against _reference_block_backward.

    No design bound exists for a whole block. Both sides use the same activations, and differ only in GEMM row counts,
    the shift-add transpose and reduction order. COMPOSITE_TOL is C2's loosest rule bound, the LN rule's 1e-13.
    """
    cfg = dls_tiny.CONFIGS[config]
    w = cheri_dls.CheriWeights.from_model(dls_tiny.tiny_model(config, dtype=F64))
    gen = torch.Generator().manual_seed(5)
    L, C = cfg.length, w.C
    errs = []
    with torch.no_grad():
        z0 = cheri_dls.stem_forward(_joint_batch(config, F64), w.W0, w.b0)
        _, saved = cheri_dls.forward_save(z0, w.blocks, w.rs, w.eps, "cpu_mirror")
        for i, block in enumerate(w.blocks):
            h, _ = cheri_dls.forward_save(z0, w.blocks[:i], w.rs, w.eps, "cpu_mirror")  # block i's input
            G = torch.randn((S, K, L, C), generator=gen, dtype=F64)
            ours = cheri_dls.bwd_block(G, saved[i], block, w.rs, S, K)
            assert ours.shape == (S, K, L, C) and ours.dtype == F64
            G_ref_half = torch.randn((S * K, L, C), generator=gen, dtype=F64)
            ref = _reference_block_backward(G, G_ref_half, h, saved[i].u, block, w.rs)
            errs.append(_rel_l2(ours.flatten(0, 1), ref))
            # The counts cotangent reaches the last block broadcast over pairs, never materialized K-fold.
            if i == len(w.blocks) - 1:
                G0 = cheri_dls.counts_cotangent(w, group=cfg.group, length=L)
                broadcast = cheri_dls.bwd_block(G0.expand(S, K, L, C), saved[i], block, w.rs, S, K)
                assert torch.equal(broadcast, cheri_dls.bwd_block(G0.repeat(S, K, 1, 1), saved[i], block, w.rs, S, K))
    _report(
        pytestconfig,
        f"bwd_block config {config}: per-pair rel-L2 vs the reference's rules, blocks "
        + ", ".join(f"{e:.3g}" for e in errs)
        + f" (tol {COMPOSITE_TOL:g})",
    )
    assert max(errs) <= COMPOSITE_TOL


@pytest.mark.parametrize("config", CONFIG_NAMES)
def test_stem_rule_and_input_multipliers(config, pytestconfig):
    """stem_rule against _nonlinear on the igelu stub, and input_multipliers against autograd through iconv.

    tangermeme's igelu hook gets its cotangent as autograd delivers it through ``.transpose(1, 2).contiguous()``
    (cherimoya.py:444-445): a transposed view of the block-0 input cotangent. Shuffled references take the secant
    almost everywhere; r = x takes the gradient everywhere.
    """
    cfg = dls_tiny.CONFIGS[config]
    w = cheri_dls.CheriWeights.from_model(dls_tiny.tiny_model(config, dtype=F64))
    gen = torch.Generator().manual_seed(9)
    L, C = cfg.length, w.C
    X1h = _joint_batch(config, F64)
    same = torch.cat([X1h[:S], X1h[:S].repeat_interleave(K, dim=0)])  # r = x
    lines = []
    for name, rows in (("shuffled", X1h), ("r = x", same)):
        with torch.no_grad():
            z0 = cheri_dls.stem_forward(rows, w.W0, w.b0)
            G = torch.randn((S, K, L, C), generator=gen, dtype=F64)
            ours = cheri_dls.stem_rule(G, z0, S, K)
        assert ours.shape == (S * K, C, L) and ours.is_contiguous()
        g_joint = torch.cat([G.flatten(0, 1), torch.randn((S * K, L, C), generator=gen, dtype=F64)]).transpose(1, 2)
        ref, _, _ = _gelu_reference(_joint(z0), g_joint)
        ref = ref[: S * K]
        err, max_ulps = _rel_l2(ours, ref), float(_ulps(ours, ref).max())
        lines.append(f"{name} rel-L2 {err:.3g}, max {max_ulps:g} ulp")
        assert err <= COMPOSITE_TOL

        with torch.no_grad():
            m = cheri_dls.input_multipliers(ours, rows[S:], w.W0)
        leaf = rows[S:].clone().requires_grad_()
        with torch.enable_grad():
            (expected,) = torch.autograd.grad(F.conv1d(leaf, w.W0, w.b0, padding=10), leaf, ours)
        assert m.shape == (S * K, 4, L) and torch.equal(m, expected)
    _report(pytestconfig, f"stem_rule config {config} vs _nonlinear: " + "; ".join(lines))
