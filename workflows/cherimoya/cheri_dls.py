"""cheri_dls: a fast, numerically faithful DeepLIFT/DeepSHAP engine for Cherimoya models.

`cherimoya attribute` runs tangermeme's hook-based deep_lift_shap over 2B-row batches: each sequence is forwarded
once per reference, and the backward runs over both halves of the batch. This engine computes the same multipliers
from a hand-written forward and backward. The forward runs over a joint batch per step: each peak's sequence once,
then its K references. The backward runs over the sequence half only. The design is
analyses/attr_pilot/design/final_design.md; the section numbers below refer to it. CH/ is the installed cherimoya
package and TM/ the installed tangermeme.

This module holds the weights, the forward pass, the backward rules, the epilogue and the Engine that runs them over
a whole set of peaks (§4-§11). Every function runs eagerly, and only the CPU path is validated so far; compilation and
the other GPU features come later.

- check_versions() and load_model(). The model is loaded with the reference's own loader,
  ``Cherimoya.load(path, device="cpu", compile=False)`` (cherimoya_cli/commands/attribute.py:35-38). Its strict
  load_state_dict refuses missing, unexpected or mis-shaped keys. It also accepts the depthwise weight under either
  spelling, ``blocks.{i}.conv_weight`` on disk or ``blocks.{i}.conv.conv_weight`` (CH/cheri.py:1045-1061).
- CheriWeights. It reads module attributes, never raw state-dict keys, and checks the architecture contract the
  hand-written passes rely on: GELU(tanh) everywhere, the iconv and fconv geometry, dilations 2**i, no MLP biases,
  and consistent shapes. Anything else is refused.
- make_wrapper(). It builds the reference's own LogCountWrapper/ProfileWrapper(ControlWrapper(model), group), so a
  bad group fails exactly as `cherimoya attribute` fails (attribute.py:42-47; CH/wrappers.py:67-85).
- stem_forward(), forward_save() and head_forward(). The forward runs over any batch of rows. It keeps what the
  backward rules read: per block the conv output y, the linear1 output u, and the rule's mean and inverse std.
  forward_stats picks how the per-row normalization is computed:
  - "cpu_mirror": op for op the reference's CPU path (CH/cheri.py:54-87, 124-132). The default on CPU; used by the
    tests.
  - "two_pass": one set of two-pass statistics serves both the forward and the rule (§6.1). The default on CUDA.
  - "one_pass": the Triton forward's E[y^2] - mean^2 (CH/cheri.py:219-222). Diagnostic only.
- head_forward(). The counts head with the reference's ops on the reference's layout (§6.3). Control tracks are fed
  as zeros, as ControlWrapper does, and the group is sliced as LogCountWrapper does. The profile head is not
  implemented yet.
- counts_cotangent() (§7.1). The cotangent of the attributed count at the final residual stream: autograd through
  head_forward, the reference's own eager head ops, once per model.
- The backward (§8), over the sequence half of each pair only. tangermeme's rule for a sequence row reads the
  reference row's activations but never its cotangent (TM/_deep_lift_utils.py:128, 554-557), so the reference half
  is never needed. Rows are laid out [x_0..x_{S-1}, r_00..r_{S-1,K-1}]; split_rows() views them as sequences
  (S, 1, ...) and references (S, K, ...), and a sequence row broadcasts over its K pairs without being copied.
  - gelu_rescale(): the rescale rule of tangermeme's _nonlinear for GELU(tanh) (TM/_deep_lift_utils.py:116-128).
  - ln_rule(): the normalization rule that cherimoya's conv_norm_op takes from tangermeme
    (CH/deep_lift_shap.py:137-150; TM/_deep_lift_utils.py:500-557).
  - dw3_t(): the depthwise conv's transpose, which takes that rule's output back to the block input.
  - bwd_block(): one CheriBlock, composed from the three; stem_rule(): the stem GELU; input_multipliers(): iconv's
    input gradient, giving the multipliers at the one-hot input.
- epilogue() (§9): per-pair convergence deltas, and hypothetical scores over the attribution window.
  mean_over_references() averages the windows over each sequence's references on the CPU, bitwise as tangermeme does.
- Engine (§4, §11): runs all of the above over a set of peaks, in steps of S whole peaks with their K references.
  Each step forwards its joint rows once and runs the backward over its S*K pairs. A short last step is padded by
  repeating its last peak, and the padding is dropped. References are injected, or drawn by dls_refs.RefProducer
  exactly as deep_lift_shap draws them.
  - Engine.raw() returns each pair's multipliers, outputs and convergence delta (tests and validation).
  - Engine.run() returns the hypothetical attributions over a window, averaged over references, as
    `cherimoya attribute` computes them, plus the deltas and a summary of them. It warns once if deltas are high.

Deliberate differences from the reference:
- The counts head casts the pooled stream to the count Linear's dtype, where CH/cherimoya.py:458 calls ``.float()``.
  Every supported model has fp32 parameters, so the two are the same operation. A float64 model on CPU (tests and the
  fp64 "truth" run only) gets a float64 head. With ``.float()`` the stock module cannot run a float64 forward at all:
  the float32 pooled stream meets the float64 Linear and raises. counts_cotangent() inherits the cast.
- mean_over_references() writes each window into a zeroed full-length block before averaging, where design §9 averages
  the (K, 4, W) window block alone. The CPU mean sums a column in an order that depends on where the column sits in
  its block, and the window alone moves some columns. Measured: for K=20, L=2114, W=100 in float32, the window-only
  mean differs from tangermeme's in 9 of 400 elements; the full-length block matches it bitwise.
- The Engine returns its results in the model's dtype, where `cherimoya attribute` casts the attributions with
  ``.float()`` after deep_lift_shap returns (attribute.py:101); the caller casts. It warns once per run with a summary
  of the deltas, where tangermeme warns once per batch with the batch's deltas, and a warning_threshold of None means
  no warning.
"""

from __future__ import annotations

import contextlib
import logging
import numbers
import operator
import time
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from os import PathLike
from typing import NamedTuple

import cherimoya
import dls_refs
import numpy as np
import tangermeme
import torch
import torch.nn.functional as F
from cherimoya import Cherimoya, ControlWrapper, LogCountWrapper, ProfileWrapper
from cherimoya.cheri import CONV_NORM_EPS, _cheri_conv

logger = logging.getLogger(__name__)

EXPECTED_VERSIONS = {"cherimoya": "0.3.0", "tangermeme": "1.5.0"}
"""Versions whose sources the design was checked against. Any other version raises unless drift is allowed."""

FORWARD_STATS = ("two_pass", "one_pass", "cpu_mirror")
OUTPUTS = ("counts", "profile")

ICONV_KERNEL, ICONV_PADDING = 21, 10  # CH/cherimoya.py:245
FCONV_KERNEL, FCONV_PADDING = 75, 37  # CH/cherimoya.py:254-255

RESCALE_EPS = 1e-6
"""The rescale rule uses the plain gradient where |z_x - z_r| < RESCALE_EPS, strictly (TM/_deep_lift_utils.py:122)."""

DEFAULT_HEAD_ROWS = 128
"""Rows of the head that counts_cotangent() differentiates: 2 x batch_size 64, the reference's full batch (§7.1)."""

AUTO_SEQS_PER_STEP = 8
"""Peaks per step for seqs_per_step="auto". Sizing steps from free GPU memory (§11) comes with the GPU engine."""

SEED_RANGE = 9_999_999
"""random_state None draws a base seed from [0, SEED_RANGE), once per Engine (§10)."""


def check_versions(allow_drift: bool = False) -> dict[str, str]:
    """Return the installed cherimoya and tangermeme versions. Raise if either differs from EXPECTED_VERSIONS.

    The engine reproduces these packages' arithmetic from their sources, so a different version may compute
    something else. With allow_drift=True a mismatch is logged as a warning instead.
    """
    found = {"cherimoya": cherimoya.__version__, "tangermeme": tangermeme.__version__}
    drift = [
        f"{name} {found[name]} (expected {want})" for name, want in EXPECTED_VERSIONS.items() if found[name] != want
    ]
    if drift:
        message = "cheri_dls was validated against other package versions: " + ", ".join(drift)
        if not allow_drift:
            raise RuntimeError(message + ". Pass allow_version_drift=True to run anyway.")
        logger.warning("%s; running anyway because version drift is allowed", message)
    return found


def load_model(path: str | PathLike, *, allow_version_drift: bool = False) -> Cherimoya:
    """Load a checkpoint the way `cherimoya attribute` does: ``Cherimoya.load(path, device="cpu", compile=False)``.

    JSON `compile` is ignored, as at attribute.py:35: DeepLIFT is never compiled. Move the weights to a device with
    CheriWeights.from_model().
    """
    check_versions(allow_version_drift)
    return Cherimoya.load(str(path), device="cpu", compile=False)


class BlockWeights(NamedTuple):
    """One CheriBlock's tensors (CH/cheri.py:984-987)."""

    cw: torch.Tensor  # (3, C) depthwise taps; row 0 reads l - d, row 2 reads l + d
    W1: torch.Tensor  # (H, C) linear1, no bias
    W2: torch.Tensor  # (C, H) linear2, no bias
    d: int  # dilation, 2**i


@dataclass(frozen=True, eq=False)
class CheriWeights:
    """The tensors of a loaded Cherimoya that the forward and backward read, detached, on one device (§5)."""

    W0: torch.Tensor  # (C, 4, 21) iconv weight
    b0: torch.Tensor  # (C,) iconv bias
    blocks: tuple[BlockWeights, ...]
    rs: float  # residual_scale, shared by every block
    Wf: torch.Tensor  # (n_out, C + n_ctl, 75) fconv weight
    bf: torch.Tensor  # (n_out,) fconv bias
    Wl: torch.Tensor  # (n_groups, C + (n_ctl > 0)) count Linear weight
    bl: torch.Tensor  # (n_groups,) count Linear bias
    T: int  # trimming: both heads read positions [T, L - T)
    eps: float  # CONV_NORM_EPS
    signal_groups: tuple[int, ...]
    n_ctl: int  # n_control_tracks

    @classmethod
    def from_model(
        cls, model: Cherimoya, device: str | torch.device | None = None, *, allow_version_drift: bool = False
    ) -> CheriWeights:
        """Read `model`'s weights after checking the contract the engine relies on. Raise ValueError if it fails.

        The tensors are ``.detach()``-ed (requires_grad False) and moved to `device`, by default the model's own. On
        the model's own device they share its storage. The parameters must all be float32. float64 is accepted on
        CPU only, for tests.
        """
        check_versions(allow_version_drift)
        problems = contract_violations(model)
        if problems:
            raise ValueError("the model breaks the cheri_dls contract:\n  " + "\n  ".join(problems))

        dev = torch.device(device) if device is not None else model.iconv.weight.device
        if model.iconv.weight.dtype == torch.float64 and dev.type != "cpu":
            raise ValueError(f"float64 models run on CPU only (tests), not on {dev}")

        def get(t: torch.Tensor) -> torch.Tensor:
            return t.detach().to(dev)

        return cls(
            W0=get(model.iconv.weight),
            b0=get(model.iconv.bias),
            blocks=tuple(
                BlockWeights(get(b.conv.conv_weight), get(b.linear1.weight), get(b.linear2.weight), int(b.dilation))
                for b in model.blocks
            ),
            rs=float(model.residual_scale),
            Wf=get(model.fconv.weight),
            bf=get(model.fconv.bias),
            Wl=get(model.linear.weight),
            bl=get(model.linear.bias),
            T=int(model.trimming),
            eps=float(CONV_NORM_EPS),
            signal_groups=tuple(int(g) for g in model.signal_groups),
            n_ctl=int(model.n_control_tracks),
        )

    @property
    def C(self) -> int:
        return self.W0.shape[0]

    @property
    def H(self) -> int:
        return self.blocks[0].W1.shape[0] if self.blocks else 0

    @property
    def n_out(self) -> int:
        return self.Wf.shape[0]

    @property
    def n_groups(self) -> int:
        return self.Wl.shape[0]

    @property
    def dtype(self) -> torch.dtype:
        return self.W0.dtype

    @property
    def device(self) -> torch.device:
        return self.W0.device


def contract_violations(model: object) -> list[str]:
    """Every way `model` departs from the architecture the hand-written passes implement; empty if none (§5)."""
    if not isinstance(model, Cherimoya):
        return [f"expected a cherimoya.Cherimoya, got {type(model).__name__}"]

    problems = []
    C, n_layers, n_ctl = model.n_filters, model.n_layers, model.n_control_tracks
    H = model.expansion * C
    groups = list(model.signal_groups)
    n_out, n_groups = sum(groups), len(groups)

    gelus = [("igelu", model.igelu)] + [(f"blocks.{i}.activation", b.activation) for i, b in enumerate(model.blocks)]
    for name, module in gelus:
        if not isinstance(module, torch.nn.GELU) or module.approximate != "tanh":
            problems.append(f"{name} must be GELU(approximate='tanh'), got {module!r}")

    dtypes = sorted({str(p.dtype) for p in model.parameters()})
    if dtypes not in (["torch.float32"], ["torch.float64"]):
        problems.append(f"parameters must all be float32 (float64 on CPU for tests), got {dtypes}")

    problems += _conv_violations(
        "iconv", model.iconv, in_channels=4, out_channels=C, kernel=ICONV_KERNEL, padding=ICONV_PADDING
    )
    problems += _conv_violations(
        "fconv", model.fconv, in_channels=C + n_ctl, out_channels=n_out, kernel=FCONV_KERNEL, padding=FCONV_PADDING
    )

    if len(model.blocks) != n_layers:
        problems.append(f"{len(model.blocks)} blocks for n_layers={n_layers}")
    for i, block in enumerate(model.blocks):
        name = f"blocks.{i}"
        if block.dilation != 2**i or block.conv.dilation != 2**i:
            problems.append(f"{name} dilation {block.dilation}/{block.conv.dilation}, expected {2**i}")
        if block.residual_scale != model.residual_scale:
            problems.append(f"{name} residual_scale {block.residual_scale} != model's {model.residual_scale}")
        problems += _shape_violations(f"{name}.conv.conv_weight", block.conv.conv_weight, (3, C))
        for layer, shape in (("linear1", (H, C)), ("linear2", (C, H))):
            linear = getattr(block, layer)
            if not isinstance(linear, torch.nn.Linear) or linear.bias is not None:
                problems.append(f"{name}.{layer} must be an nn.Linear without bias, got {linear!r}")
            else:
                problems += _shape_violations(f"{name}.{layer}.weight", linear.weight, shape)

    if not isinstance(model.linear, torch.nn.Linear) or model.linear.bias is None:
        problems.append(f"linear must be an nn.Linear with bias, got {model.linear!r}")
    else:
        problems += _shape_violations("linear.weight", model.linear.weight, (n_groups, C + (n_ctl > 0)))
        problems += _shape_violations("linear.bias", model.linear.bias, (n_groups,))

    if not isinstance(model.trimming, int) or model.trimming < 0:
        problems.append(f"trimming must be a non-negative int, got {model.trimming!r}")
    return problems


def _shape_violations(name: str, tensor: torch.Tensor, shape: tuple[int, ...]) -> list[str]:
    return [] if tuple(tensor.shape) == shape else [f"{name} has shape {tuple(tensor.shape)}, expected {shape}"]


def _conv_violations(
    name: str, conv: object, *, in_channels: int, out_channels: int, kernel: int, padding: int
) -> list[str]:
    """The Conv1d geometry the engine hard-codes: F.conv1d(x, W, b, padding=`padding`) with stride and dilation 1."""
    if not isinstance(conv, torch.nn.Conv1d):
        return [f"{name} must be an nn.Conv1d, got {type(conv).__name__}"]
    want = {
        "kernel_size": (kernel,),
        "padding": (padding,),
        "stride": (1,),
        "dilation": (1,),
        "groups": 1,
        "padding_mode": "zeros",
    }
    problems = [
        f"{name}.{key} is {getattr(conv, key)!r}, expected {value!r}"
        for key, value in want.items()
        if getattr(conv, key) != value
    ]
    problems += _shape_violations(f"{name}.weight", conv.weight, (out_channels, in_channels, kernel))
    if conv.bias is None:
        problems.append(f"{name} must have a bias")
    else:
        problems += _shape_violations(f"{name}.bias", conv.bias, (out_channels,))
    return problems


def make_wrapper(model: Cherimoya, output: str = "counts", group: int | None = 0) -> torch.nn.Module:
    """The wrapper `cherimoya attribute` attributes, built from cherimoya's own classes and checks.

    A count head with several groups needs a group (attribute.py:43-47). The wrapper's constructor validates the
    group itself (CH/wrappers.py:67-85).
    """
    if output not in OUTPUTS:
        raise ValueError("output must be either `counts` or `profile`.")
    if output == "counts" and group is None and len(model.signal_groups) > 1:
        raise ValueError(
            "deep_lift_shap attributes one output; set `group` to one of the model's "
            f"{len(model.signal_groups)} signal groups"
        )
    inner = ControlWrapper(model)
    return LogCountWrapper(inner, group=group) if output == "counts" else ProfileWrapper(inner, group=group)


def resolve_forward_stats(forward_stats: str, device: str | torch.device) -> str:
    """Resolve "auto": "two_pass" on CUDA and "cpu_mirror" on CPU (§4)."""
    if forward_stats == "auto":
        return "two_pass" if torch.device(device).type == "cuda" else "cpu_mirror"
    if forward_stats not in FORWARD_STATS:
        raise ValueError(f"forward_stats must be 'auto' or one of {FORWARD_STATS}, got {forward_stats!r}")
    return forward_stats


class BlockSaved(NamedTuple):
    """What one block's backward rules read (§6.4). Nothing else is kept: yhat and the GELU output are recomputed."""

    y: torch.Tensor  # (R, L, C) depthwise conv output, the normalization's input
    u: torch.Tensor  # (R, L, H) linear1 output, the GELU's input
    mu: torch.Tensor  # (R,) the rule's mean of y over the (L, C) plane
    v: torch.Tensor  # (R,) the rule's inverse std, (var + eps) ** -0.5, from the two-pass biased variance


class Forward(NamedTuple):
    """model_forward()'s results for a batch of R rows."""

    y: torch.Tensor  # (R,) the attributed output, one scalar per row
    z0: torch.Tensor  # (R, C, L) iconv output, the stem GELU's input
    hL: torch.Tensor  # (R, L, C) the final residual stream, channels last
    saved: list[BlockSaved]


def stem_forward(X1h: torch.Tensor, W0: torch.Tensor, b0: torch.Tensor) -> torch.Tensor:
    """iconv: the op nn.Conv1d(4, C, 21, padding=10) calls (CH/cherimoya.py:444). X1h (R, 4, L) -> z0 (R, C, L)."""
    return F.conv1d(X1h, W0, b0, padding=ICONV_PADDING)


def dw3(h: torch.Tensor, cw: torch.Tensor, d: int) -> torch.Tensor:
    """The 3-tap dilated depthwise conv as a shift-add (CH/cheri.py:174-193).

    h has shape (..., L, C), channels last, and cw has shape (3, C). The output is
    ``y[l] = cw[0] * h[l - d] + cw[1] * h[l] + cw[2] * h[l + d]``, with zeros outside [0, L).
    """
    L = h.shape[-2]
    hp = F.pad(h, (0, 0, d, d))
    return hp[..., :L, :] * cw[0] + hp[..., d : d + L, :] * cw[1] + hp[..., 2 * d : 2 * d + L, :] * cw[2]


def rule_stats(y: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The statistics conv_norm_op's rule takes over each row's (L, C) plane: (mu, y - mu, v), keepdim.

    The expressions are those of tangermeme's _layer_normalization_helper (TM/_deep_lift_utils.py:514-527), applied
    as cherimoya's _ConvNormView sets it up: normalized_shape (L, C), eps 1e-3 (CH/deep_lift_shap.py:68-72). They are
    computed in y's dtype.
    """
    mu = y.mean(dim=(-2, -1), keepdim=True)
    a = y - mu
    var = (a**2).mean(dim=(-2, -1), keepdim=True)
    v = (var + eps) ** (-0.5)
    return mu, a, v


def _cpu_norm(y: torch.Tensor, eps: float) -> torch.Tensor:
    """CH/cheri.py:126-132 verbatim: the reference's CPU normalization, fp32 cast included even for float64."""
    N, L, C = y.shape
    flat = y.reshape(N, -1).float()
    mean = flat.mean(dim=1, keepdim=True)
    var = flat.var(dim=1, keepdim=True, unbiased=False)
    rstd = (var + eps).rsqrt()
    flat = (flat - mean) * rstd
    return flat.reshape(N, L, C).to(y.dtype)


def _one_pass_norm(y: torch.Tensor, eps: float) -> torch.Tensor:
    """The Triton forward's formula (CH/cheri.py:219-222, 251): var = E[y^2] - mean^2, rstd = 1 / sqrt(var + eps)."""
    D = y.shape[-2] * y.shape[-1]
    mean = y.sum(dim=(-2, -1), keepdim=True) / D
    var = (y * y).sum(dim=(-2, -1), keepdim=True) / D - mean * mean
    return (y - mean) * (1.0 / torch.sqrt(var + eps))


def forward_save(
    z0: torch.Tensor, blocks: tuple[BlockWeights, ...], rs: float, eps: float, stats: str
) -> tuple[torch.Tensor, list[BlockSaved]]:
    """Run the stem GELU and the blocks on z0 (R, C, L), keeping what the rules read (§6.1, §6.2).

    Returns the final residual stream (R, L, C), channels last, and one BlockSaved per block. The function is
    row-wise: a batch can hold sequences and references in any order. `stats` picks the normalization:
    - "cpu_mirror": y from cherimoya's own _cheri_conv and the normalization of CH/cheri.py:127-132, so the forward is
      the reference CPU path op for op.
    - "two_pass": y from the shift-add dw3; the forward normalizes with the rule's own statistics, yhat = (y - mu) * v.
    - "one_pass": y from dw3; the forward uses the Triton formula. Diagnostic only.
    In every mode the saved mu and v are the rule's two-pass statistics, computed in the model's dtype.
    """
    if stats not in FORWARD_STATS:
        raise ValueError(f"stats must be one of {FORWARD_STATS}, got {stats!r}")
    h = F.gelu(z0, approximate="tanh").transpose(1, 2).contiguous()  # CH/cherimoya.py:444-445
    saved = []
    for cw, W1, W2, d in blocks:
        R = h.shape[0]
        y = _cheri_conv(h, cw, d) if stats == "cpu_mirror" else dw3(h, cw, d)
        mu, a, v = rule_stats(y, eps)
        if stats == "cpu_mirror":
            yhat = _cpu_norm(y, eps)
        elif stats == "one_pass":
            yhat = _one_pass_norm(y, eps)
        else:
            yhat = a * v
        u = F.linear(yhat, W1)  # CH/cheri.py:1180, then the residual of :1181 in its op order
        h = h + F.linear(F.gelu(u, approximate="tanh"), W2) * rs
        saved.append(BlockSaved(y, u, mu.view(R), v.view(R)))
    return h, saved


def head_forward(hL: torch.Tensor, w: CheriWeights, *, output: str = "counts", group: int | None = 0) -> torch.Tensor:
    """The attributed scalar per row, from the final residual stream hL (R, L, C) (§6.3).

    Counts: the reference's head (CH/cherimoya.py:449, 458-470) on the reference's (R, C, L) layout. Then
    LogCountWrapper's group slice (CH/wrappers.py:201-202) and deep_lift_shap's target 0
    (TM/deep_lift_shap.py:733). Control tracks are zeros, as ControlWrapper feeds them (CH/wrappers.py:62-64), so
    their count feature is log(0 + 1) = 0. The pooled stream is cast to the count Linear's dtype, where the
    reference calls ``.float()``. That is the same op for fp32 parameters and keeps a float64 model in float64.
    """
    if output == "profile":
        raise NotImplementedError("the profile head (ProfileWrapper) is not implemented in cheri_dls yet; use counts")
    if output != "counts":
        raise ValueError("output must be either `counts` or `profile`.")
    if group is not None and (isinstance(group, bool) or not isinstance(group, int) or not 0 <= group < w.n_groups):
        raise ValueError(f"group must be None or an int in [0, {w.n_groups}), got {group!r}")
    R, L, _ = hL.shape
    T = w.T
    if L <= 2 * T:
        raise ValueError(f"input length {L} leaves no positions after trimming {T} from each end")

    H = hL.transpose(1, 2).contiguous()  # CH/cherimoya.py:449
    pooled = torch.mean(H[:, :, T : L - T].to(w.Wl.dtype), dim=2)  # :458
    if w.n_ctl:
        ctl = H.new_zeros(R, w.n_ctl, L)
        ctl = torch.sum(ctl[:, :, T : L - T].to(w.Wl.dtype), dim=(1, 2)).unsqueeze(-1)  # :460-461
        pooled = torch.cat([pooled, torch.log(ctl + 1)], dim=-1)  # :468
    y = F.linear(pooled, w.Wl, w.bl)  # :470
    if group is not None:
        y = y[:, group : group + 1]
    return y[:, 0]


def model_forward(
    X1h: torch.Tensor,
    w: CheriWeights,
    *,
    forward_stats: str = "auto",
    output: str = "counts",
    group: int | None = 0,
) -> Forward:
    """The whole forward on a batch of one-hot rows X1h (R, 4, L), cast to the weights' dtype and device.

    Run it under torch.no_grad() unless a graph is wanted. The engine never needs autograd for its forward.
    """
    stats = resolve_forward_stats(forward_stats, w.device)
    X1h = X1h.to(device=w.device, dtype=w.dtype)
    z0 = stem_forward(X1h, w.W0, w.b0)
    hL, saved = forward_save(z0, w.blocks, w.rs, w.eps, stats)
    return Forward(head_forward(hL, w, output=output, group=group), z0, hL, saved)


def counts_cotangent(
    w: CheriWeights, *, group: int | None = 0, length: int, rows: int = DEFAULT_HEAD_ROWS
) -> torch.Tensor:
    """The cotangent of the attributed count at the final residual stream, (L, C), shared by every row (§7.1).

    It is autograd's gradient of ``y.sum()`` through head_forward(), which runs the reference's own head ops on the
    reference's layout, at `rows` rows: by default 2 x batch_size, the reference's full batch. So the backward of the
    count Linear is the reference's own mm over that many rows, with K = n_groups (1 for one group), and the mean's
    backward divides by L - 2T exactly as the reference's does. The result is zero outside [T, L - T). Compute it once
    per model and broadcast it over the pairs. It turns grad mode on for itself, so the engine can run under no_grad,
    but not under torch.inference_mode().
    """
    if torch.is_inference_mode_enabled():
        raise RuntimeError("counts_cotangent needs autograd, which torch.inference_mode() switches off")
    if rows < 1:
        raise ValueError(f"rows must be at least 1, got {rows}")
    with torch.enable_grad():
        hL = torch.zeros(rows, length, w.C, device=w.device, dtype=w.dtype, requires_grad=True)
        y = head_forward(hL, w, output="counts", group=group)
        (grad,) = torch.autograd.grad(y.sum(), hL)
    return grad[0].contiguous()


def split_rows(t: torch.Tensor, S: int, K: int) -> tuple[torch.Tensor, torch.Tensor]:
    """View rows [x_0..x_{S-1}, r_00..r_{S-1,K-1}] as sequences (S, 1, ...) and references (S, K, ...).

    Both are views, so `t` must be contiguous along its rows. A sequence row broadcasts over its K pairs.
    """
    if t.shape[0] != S + S * K:
        raise ValueError(f"expected S + S*K = {S + S * K} rows for S={S}, K={K}, got {t.shape[0]}")
    return t[:S].unsqueeze(1), t[S:].view(S, K, *t.shape[1:])


def gelu_rescale(G: torch.Tensor, z: torch.Tensor, S: int, K: int) -> torch.Tensor:
    """DeepLIFT's rescale rule through GELU(tanh) on the sequence half: tangermeme's _nonlinear (§8.1).

    G (S, K, ...) is each pair's cotangent at the GELU's output, and z (S + S*K, ...) the GELU's input over the joint
    rows. The result, (S, K, ...), is the cotangent at the GELU's input (TM/_deep_lift_utils.py:116-128):
    - G times the secant (gelu(z_x) - gelu(z_r)) / (z_x - z_r);
    - where |z_x - z_r| < 1e-6, strictly, the plain gradient instead: what nn.GELU's autograd hands tangermeme as
      grad_input, gelu_backward(G, z_x).
    The GELU outputs are recomputed from z over all rows at once, as the forward computed them, so the secants use the
    forward's own values.
    """
    zx, zr = split_rows(z, S, K)
    gx, gr = split_rows(F.gelu(z, approximate="tanh"), S, K)
    din = zx - zr
    small = din.abs() < RESCALE_EPS
    secant = (gx - gr) / din.masked_fill(small, 1.0)  # the masked entries are never used; this keeps them finite
    gradient = torch.ops.aten.gelu_backward(G, zx.expand_as(G), approximate="tanh")
    return torch.where(small, gradient, G * secant)


def ln_rule(Gn: torch.Tensor, y: torch.Tensor, mu: torch.Tensor, v: torch.Tensor, S: int, K: int) -> torch.Tensor:
    """The normalization rule of conv_norm_op on the sequence half (§8.1).

    cherimoya's conv_norm_op applies tangermeme's _layer_normalization_helper (TM/_deep_lift_utils.py:500-557) to the
    depthwise conv's output through _ConvNormView: the whole (L, C) plane of each row, D = L*C, eps 1e-3, no affine
    weight (CH/deep_lift_shap.py:68-72, 137-150). The helper's gamma of 1.0 multiplies exactly and is left out.

    Gn (S, K, L, C) is each pair's cotangent at the normalization's output. y (S + S*K, L, C) is its input over the
    joint rows, and mu and v (S + S*K,) are the rule statistics of rule_stats(). Returns the cotangent at y,
    (S, K, L, C). The expressions and their order are the helper's (:534-552).
    """
    L, C = y.shape[-2:]
    D = L * C
    yx, yr = split_rows(y, S, K)
    mx, mr = (t[..., None, None] for t in split_rows(mu, S, K))
    vx, vr = (t[..., None, None] for t in split_rows(v, S, K))
    v_avg = (vx + vr) / 2
    a_sum = (yx - mx) + (yr - mr)
    ratio = (-(vx**2) * (vr**2)) / (2 * v_avg)
    variance_term = ratio * a_sum / (2 * D)
    term1 = v_avg * (Gn - Gn.mean(dim=(-2, -1), keepdim=True))
    dot = (Gn * a_sum).sum(dim=(-2, -1), keepdim=True)
    return term1 + variance_term * dot


def dw3_t(G: torch.Tensor, cw: torch.Tensor, d: int) -> torch.Tensor:
    """The transpose of dw3, i.e. the depthwise conv's input gradient, as a shift-add (§8.1).

    G has shape (..., L, C). ``g[l] = cw[1] * G[l] + cw[0] * G[l + d] + cw[2] * G[l - d]``, with zeros outside
    [0, L), in the term order of the Triton backward (CH/cheri.py:410-411).
    """
    L = G.shape[-2]
    Gp = F.pad(G, (0, 0, d, d))
    return Gp[..., d : d + L, :] * cw[1] + Gp[..., 2 * d : 2 * d + L, :] * cw[0] + Gp[..., :L, :] * cw[2]


def bwd_block(G: torch.Tensor, saved: BlockSaved, block: BlockWeights, rs: float, S: int, K: int) -> torch.Tensor:
    """One CheriBlock's DeepLIFT backward on the sequence half: G (S, K, L, C) at its output -> at its input (§8.1).

    The block computes ``h + linear2(GELU(linear1(conv_norm(h)))) * rs`` (CH/cheri.py:1179-1181). Backwards: the
    ordinary gradient through the scale and linear2, the rescale rule at the GELU, the ordinary gradient through
    linear1, conv_norm_op's normalization rule pushed back through the depthwise conv, and the residual. G may be a
    broadcast view, such as the counts cotangent expanded over the pairs.
    """
    Gu = gelu_rescale(torch.matmul(G * rs, block.W2), saved.u, S, K)
    Gy = ln_rule(torch.matmul(Gu, block.W1), saved.y, saved.mu, saved.v, S, K)
    return G + dw3_t(Gy, block.cw, block.d)


def stem_rule(G: torch.Tensor, z0: torch.Tensor, S: int, K: int) -> torch.Tensor:
    """The stem GELU's rescale rule on the sequence half: G (S, K, L, C) at h0 -> (S*K, C, L) at iconv's output (§8.2).

    G is transposed back to the GELU's (C, L) layout (CH/cherimoya.py:444-445). z0 (S + S*K, C, L) is iconv's output
    over the joint rows. The result is contiguous, ready for input_multipliers().
    """
    Gz = gelu_rescale(G.transpose(-1, -2), z0, S, K)
    return Gz.reshape(S * K, *Gz.shape[-2:]).contiguous()


def input_multipliers(Gz: torch.Tensor, X_refs: torch.Tensor, W0: torch.Tensor) -> torch.Tensor:
    """iconv's input gradient: the multipliers m (S*K, 4, L) at the one-hot input (§8.2).

    This is the call autograd's ConvolutionBackward0 makes for ``F.conv1d(X, W0, b0, padding=10)`` when only the input
    needs a gradient: aten.convolution_backward with output mask (True, False, False). X_refs, the reference rows
    (S*K, 4, L), is a real contiguous tensor of the input's shape; the input gradient does not read its values.
    """
    return torch.ops.aten.convolution_backward(
        Gz, X_refs, W0, None, [1], [ICONV_PADDING], [1], False, [0], 1, [True, False, False]
    )[0]


def convergence_deltas(
    m: torch.Tensor, X1h: torch.Tensor, y_x: torch.Tensor, y_r: torch.Tensor, S: int, K: int
) -> torch.Tensor:
    """tangermeme's convergence delta per pair, |(y_x - y_r) - sum((x - r) * m)| over the full length (§9).

    m (S*K, 4, L) holds the raw multipliers, X1h (S + S*K, 4, L) the joint one-hot rows, y_x (S,) and y_r (S, K) the
    attributed outputs. Returns (S, K). The expression is TM/deep_lift_shap.py:739-742's.
    """
    x, r = split_rows(X1h, S, K)
    m = m.view(S, K, *m.shape[1:])
    return ((y_x[:, None] - y_r) - ((x - r) * m).sum(dim=(-2, -1))).abs()


def epilogue(
    m: torch.Tensor,
    X1h: torch.Tensor,
    y_x: torch.Tensor,
    y_r: torch.Tensor,
    start: int,
    end: int,
    S: int,
    K: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convergence deltas over the full length, and hypothetical scores over the window [start, end) (§9).

    m (S*K, 4, L) holds the multipliers, X1h (S + S*K, 4, L) the joint one-hot rows, y_x (S,) and y_r (S, K) the
    attributed outputs. Returns:
    - hyp (S, K, 4, end - start), contiguous. For one-hot references (an N column of zeros included) it is bitwise
      tangermeme's hypothetical_attributions (TM/deep_lift_shap.py:93-103) sliced to the window: both round once, to
      fl(m_i - m_b) for reference base b.
    - delta (S, K): convergence_deltas() on the raw multipliers.
    """
    L = X1h.shape[-1]
    if not 0 <= start < end <= L:
        raise ValueError(f"the window [{start}, {end}) must be non-empty and inside [0, {L})")
    delta = convergence_deltas(m, X1h, y_x, y_r, S, K)
    _, r = split_rows(X1h, S, K)
    m = m.view(S, K, *m.shape[1:])
    mw, rw = m[..., start:end], r[..., start:end]
    return (mw - (rw * mw).sum(dim=-2, keepdim=True)).contiguous(), delta


def mean_over_references(hyp: torch.Tensor, start: int, length: int) -> torch.Tensor:
    """Each sequence's mean over its K references, on the CPU, bitwise as tangermeme computes it (§9).

    hyp (S, K, 4, W) holds the windows [start, start + W) of full-length hypothetical scores. Returns (S, 4, W).

    tangermeme stacks a sequence's K full-length scores into a contiguous (K, 4, length) block and takes
    ``.mean(dim=0)`` on the CPU (TM/deep_lift_shap.py:760-770). The CPU kernel's summation order for a column depends on
    where the column sits in that block: its vectorized body and its remainder group the K terms differently. So each
    window is written at its own positions into a zeroed (K, 4, length) block, averaged there, and sliced. Columns are
    summed independently, so the zeros outside the window change nothing inside it.
    """
    if hyp.device.type != "cpu":
        raise ValueError(f"the mean over references runs on the CPU, as tangermeme's does; got {hyp.device}")
    S, K, A, W = hyp.shape
    if not 0 <= start <= start + W <= length:
        raise ValueError(f"the window [{start}, {start + W}) must lie inside [0, {length})")
    out = hyp.new_empty(S, A, W)
    block = hyp.new_zeros(K, A, length)
    for s in range(S):
        block[:, :, start : start + W] = hyp[s]
        out[s] = block.mean(dim=0)[:, start : start + W]
    return out


class RawResult(NamedTuple):
    """Engine.raw()'s output: CPU tensors in the engine's dtype."""

    m: torch.Tensor  # (N, K, 4, L) multipliers at the one-hot input; pair (e, k) at [e, k]
    y_x: torch.Tensor  # (N,) the attributed output of each sequence
    y_r: torch.Tensor  # (N, K) the attributed output of each reference
    deltas: torch.Tensor  # (N, K) convergence deltas
    debug: dict  # what ran: step size, steps, padding, statistics mode, reference source, delta summary


class Result(NamedTuple):
    """Engine.run()'s output: CPU tensors in the engine's dtype. `cherimoya attribute` saves attr.float() (:101)."""

    attr: torch.Tensor  # (N, 4, W) hypothetical attributions over the window, averaged over each sequence's references
    deltas: torch.Tensor  # (N, K) convergence deltas
    y_x: torch.Tensor  # (N,)
    y_r: torch.Tensor  # (N, K)
    meta: dict  # debug's keys, plus the window, the wall time and pairs/s


class Engine:
    """DeepLIFT/DeepSHAP attributions of a Cherimoya model, as `cherimoya attribute` computes them with tangermeme.

    The engine works in steps of S whole peaks, each with its K references (§3, §11):
    - The step's joint rows [x_0..x_{S-1}, r_00..r_{S-1,K-1}] are forwarded once: each sequence once, not once per
      reference as in tangermeme's batches. Nothing in the forward mixes rows.
    - The backward runs over the sequence half only, as S*K pairs. The counts cotangent enters broadcast over the
      pairs; each block's rules, the stem rule and iconv's input gradient follow (bwd_block, stem_rule,
      input_multipliers).
    - A last step with fewer than S peaks is padded by repeating its last peak, sequence and references, and the
      padded pairs are dropped. So every step has the same shapes.
    - run() takes the hypothetical scores over the window [start, end) and averages them over each sequence's K
      references on the CPU, bitwise as tangermeme does (mean_over_references).

    Arguments:
    - model: a loaded Cherimoya (load_model). Its weights are read once (CheriWeights) and never changed.
    - output, group: what is attributed, as in attribute.json. The real LogCountWrapper(ControlWrapper(model), group)
      is built, so a bad group fails exactly as `cherimoya attribute` fails. output="profile" is not implemented yet.
    - device: where the weights and steps live; by default the model's device. Only the CPU path is validated so far.
      The GPU features of the design (compiled passes, precision modes, pinned transfers, OOM backoff, step sizing from
      free memory) are not implemented yet.
    - n_shuffles, random_state: K references per sequence when none are passed in: pair (e, k) gets
      ``dinucleotide_shuffle(X[e:e+1].float(), n=1, random_state=random_state + k)[:, 0]``, drawn by
      dls_refs.RefProducer exactly as deep_lift_shap draws it (R4). random_state None draws a base seed once, records
      it, and makes the run reproducible from it.
    - head_rows: rows of the head that counts_cotangent() differentiates, by default 2 x 64. `cherimoya attribute`
      differentiates 2 x batch_size rows, so pass 2 x the JSON batch_size. Nothing else depends on batch_size.
    - seqs_per_step: S, or "auto" for AUTO_SEQS_PER_STEP. A run with fewer peaks uses steps of all of them.
    - forward_stats: the normalization's statistics (forward_save). "auto" is "cpu_mirror" on CPU, "two_pass" on CUDA.
    - warning_threshold: run() warns once, with a summary, if any convergence delta exceeds it. None: no warning.
    - ref_workers: worker processes for RefProducer; None for its default, 0 to draw in this process.
    - allow_version_drift: run with other cherimoya or tangermeme versions than the ones the engine was checked against.
    """

    def __init__(
        self,
        model: Cherimoya,
        *,
        output: str = "counts",
        group: int | None = 0,
        device: str | torch.device | None = None,
        n_shuffles: int = 20,
        random_state: int | None = 0,
        head_rows: int = DEFAULT_HEAD_ROWS,
        seqs_per_step: int | str = "auto",
        forward_stats: str = "auto",
        warning_threshold: float | None = 1e-3,
        ref_workers: int | None = None,
        allow_version_drift: bool = False,
    ) -> None:
        if output not in OUTPUTS:
            raise ValueError("output must be either `counts` or `profile`.")
        if output == "profile":
            raise NotImplementedError("the profile head is not implemented in cheri_dls yet; attribute counts")
        self.weights = CheriWeights.from_model(model, device, allow_version_drift=allow_version_drift)
        self.wrapper = make_wrapper(model, output, group)  # the reference's group checks; kept for self-checks
        self.output, self.group = output, group
        self.forward_stats = resolve_forward_stats(forward_stats, self.weights.device)
        self.n_shuffles = _positive_int(n_shuffles, "n_shuffles")
        self.head_rows = _positive_int(head_rows, "head_rows")
        if seqs_per_step != "auto":
            seqs_per_step = _positive_int(seqs_per_step, "seqs_per_step", "'auto' or ")
        self.seqs_per_step = seqs_per_step
        if random_state is None:
            random_state = int(np.random.randint(0, SEED_RANGE))
            logger.info("random_state is None: drew the base seed %d", random_state)
        if isinstance(random_state, bool) or not isinstance(random_state, numbers.Integral):
            raise ValueError(f"random_state must be an integer or None, got {random_state!r}")
        self.random_state = int(random_state)
        if warning_threshold is not None and (
            isinstance(warning_threshold, bool) or not isinstance(warning_threshold, numbers.Real)
        ):
            raise ValueError(f"warning_threshold must be a number or None, got {warning_threshold!r}")
        self.warning_threshold = None if warning_threshold is None else float(warning_threshold)
        if ref_workers is not None and (
            isinstance(ref_workers, bool) or not isinstance(ref_workers, numbers.Integral) or ref_workers < 0
        ):
            raise ValueError(f"ref_workers must be None or a non-negative integer, got {ref_workers!r}")
        self.ref_workers = None if ref_workers is None else int(ref_workers)
        self._cotangents: dict[int, torch.Tensor] = {}

    def raw(self, X: torch.Tensor | np.ndarray, references: torch.Tensor | np.ndarray | None = None) -> RawResult:
        """Each pair's multipliers, outputs and convergence delta: what deep_lift_shap(raw_outputs=True) computes.

        For tests and validation: the multipliers take N*K*4*L values. X is an (N, 4, L) one-hot such as
        extract_loci's int8 one, with no N. references is an (N, K, 4, L) one-hot whose columns may be all zero (N), or
        None to draw n_shuffles per sequence (see the class doc). Injected references set K, as in deep_lift_shap.
        """
        x_tokens, ref_tokens, K = self._tokens(X, references)
        N, L = x_tokens.shape
        S = self._step_size(N)
        G0 = self._counts_cotangent(L)
        dtype = self.weights.dtype
        m = torch.empty(N, K, 4, L, dtype=dtype)
        y_x, y_r, deltas = torch.empty(N, dtype=dtype), torch.empty(N, K, dtype=dtype), torch.empty(N, K, dtype=dtype)
        debug = self._describe(N, L, K, S)
        with self._reference_chunks(x_tokens, ref_tokens, K, S, debug) as chunks, torch.no_grad():
            for a, b, refs in chunks:
                n = b - a
                X1h = self._joint_rows(x_tokens[a:b], refs, S)
                fwd, mult = self._step(X1h, G0, S, K)
                ys_x, ys_r = fwd.y[:S], fwd.y[S:].view(S, K)
                delta = convergence_deltas(mult, X1h, ys_x, ys_r, S, K)
                m[a:b] = mult.view(S, K, 4, L)[:n].cpu()
                y_x[a:b], y_r[a:b], deltas[a:b] = ys_x[:n].cpu(), ys_r[:n].cpu(), delta[:n].cpu()
                debug["steps"] += 1
                debug["padded_peaks"] += S - n
        debug["deltas"] = delta_summary(deltas, self.warning_threshold)
        return RawResult(m, y_x, y_r, deltas, debug)

    def run(
        self,
        X: torch.Tensor | np.ndarray,
        start: int,
        end: int,
        *,
        references: torch.Tensor | np.ndarray | None = None,
    ) -> Result:
        """Hypothetical attributions over the window [start, end), averaged over each sequence's references.

        This is ``deep_lift_shap(wrapper, X, hypothetical=True, ...)[:, :, start:end]`` as `cherimoya attribute` calls
        it (attribute.py:88-101), before its ``.float()``. X and references are as in raw(). Also returns the
        convergence deltas and outputs per pair, and warns once if any delta exceeds warning_threshold.
        """
        x_tokens, ref_tokens, K = self._tokens(X, references)
        N, L = x_tokens.shape
        if not 0 <= operator.index(start) < operator.index(end) <= L:
            raise ValueError(f"the window [{start}, {end}) must be non-empty and inside [0, {L})")
        S = self._step_size(N)
        G0 = self._counts_cotangent(L)
        dtype = self.weights.dtype
        attr = torch.empty(N, 4, end - start, dtype=dtype)
        y_x, y_r, deltas = torch.empty(N, dtype=dtype), torch.empty(N, K, dtype=dtype), torch.empty(N, K, dtype=dtype)
        meta = self._describe(N, L, K, S) | {"window": [int(start), int(end)]}
        began = time.perf_counter()
        with self._reference_chunks(x_tokens, ref_tokens, K, S, meta) as chunks, torch.no_grad():
            for a, b, refs in chunks:
                n = b - a
                X1h = self._joint_rows(x_tokens[a:b], refs, S)
                fwd, mult = self._step(X1h, G0, S, K)
                ys_x, ys_r = fwd.y[:S], fwd.y[S:].view(S, K)
                hyp, delta = epilogue(mult, X1h, ys_x, ys_r, start, end, S, K)
                attr[a:b] = mean_over_references(hyp[:n].cpu(), start, L)
                y_x[a:b], y_r[a:b], deltas[a:b] = ys_x[:n].cpu(), ys_r[:n].cpu(), delta[:n].cpu()
                meta["steps"] += 1
                meta["padded_peaks"] += S - n
        meta["seconds"] = time.perf_counter() - began
        meta["pairs_per_s"] = N * K / meta["seconds"]
        meta["deltas"] = delta_summary(deltas, self.warning_threshold)
        logger.info(
            "cheri_dls: %d sequences x %d references in %d steps of %d peaks, %.1f s (%.0f pairs/s)",
            N,
            K,
            meta["steps"],
            S,
            meta["seconds"],
            meta["pairs_per_s"],
        )
        if meta["deltas"]["above"]:
            warnings.warn(delta_warning(meta["deltas"]), RuntimeWarning, stacklevel=2)
        return Result(attr, deltas, y_x, y_r, meta)

    def _step(self, X1h: torch.Tensor, G0: torch.Tensor, S: int, K: int) -> tuple[Forward, torch.Tensor]:
        """One step: the forward over the joint rows, then the backward over the S*K pairs. Returns (fwd, m).

        m (S*K, 4, L) holds the multipliers at the one-hot input.
        """
        w = self.weights
        L = X1h.shape[-1]
        fwd = model_forward(X1h, w, forward_stats=self.forward_stats, output=self.output, group=self.group)
        G = G0.expand(S, K, L, w.C)
        for saved, block in zip(reversed(fwd.saved), reversed(w.blocks)):
            G = bwd_block(G, saved, block, w.rs, S, K)
        return fwd, input_multipliers(stem_rule(G, fwd.z0, S, K), X1h[S:], w.W0)

    def _joint_rows(self, x_tokens: np.ndarray, ref_tokens: np.ndarray, S: int) -> torch.Tensor:
        """The step's one-hot rows [x_0..x_{S-1}, r_00..r_{S-1,K-1}], (S + S*K, 4, L), in the engine's dtype.

        x_tokens (n, L) and ref_tokens (n, K, L) are uint8 tokens (dls_refs). With n < S, the last peak's sequence and
        references are repeated to fill the step.
        """
        n = len(x_tokens)
        if n < S:
            keep = np.concatenate([np.arange(n), np.full(S - n, n - 1)])
            x_tokens, ref_tokens = x_tokens[keep], ref_tokens[keep]
        tokens = np.concatenate([x_tokens, ref_tokens.reshape(-1, x_tokens.shape[-1])])
        return dls_refs.tokens_to_onehot(torch.from_numpy(tokens).to(self.weights.device), self.weights.dtype)

    @contextlib.contextmanager
    def _reference_chunks(
        self, x_tokens: np.ndarray, ref_tokens: np.ndarray | None, K: int, S: int, record: dict
    ) -> Iterator[Iterator[tuple[int, int, np.ndarray]]]:
        """An iterator of (a, b, references of peaks [a, b) as (b - a, K, L) tokens), over chunks of S peaks in order.

        Injected references are sliced. Otherwise a RefProducer draws them, ahead of use, and closes its pool on exit.
        """
        N = len(x_tokens)
        if ref_tokens is not None:
            record["references"] = "injected"
            yield ((a, min(a + S, N), ref_tokens[a : a + S]) for a in range(0, N, S))
            return
        record["references"] = "produced"
        with dls_refs.RefProducer(self.ref_workers) as producer:
            yield producer.chunks(x_tokens, K, self.random_state, S)
            record["reference_stats"] = dict(producer.stats)

    def _tokens(
        self, X: torch.Tensor | np.ndarray, references: torch.Tensor | np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray | None, int]:
        """Check X and references as deep_lift_shap does, and encode them as uint8 tokens.

        Returns X's tokens (N, L), the references' tokens (N, K, L) or None, and K.

        X must be a one-hot with exactly one base per position. references may have all-zero (N) columns
        (TM/deep_lift_shap.py:681-684). The tokens are lossless, so the one-hot rows of a step equal the inputs.
        """
        x = torch.as_tensor(X)
        if x.dim() != 3 or x.shape[1] != 4:
            raise ValueError(f"X must have shape (N, 4, L), got {tuple(x.shape)}")
        if x.shape[0] == 0:
            raise ValueError("deep_lift_shap requires at least one example; got X with shape[0] == 0.")
        x_tokens = dls_refs.onehot_to_tokens(x)
        if bool((x_tokens == dls_refs.N_TOKEN).any()):
            raise ValueError("X must be one-hot encoded and cannot have unknown characters (all-zero columns)")
        if references is None:
            return x_tokens, None, self.n_shuffles
        r = torch.as_tensor(references)
        N, _, L = x.shape
        if r.dim() != 4 or r.shape[0] != N or r.shape[1] < 1 or tuple(r.shape[2:]) != (4, L):
            raise ValueError(f"references must have shape ({N}, K >= 1, 4, {L}), got {tuple(r.shape)}")
        return x_tokens, dls_refs.onehot_to_tokens(r), r.shape[1]

    def _step_size(self, n_peaks: int) -> int:
        return min(AUTO_SEQS_PER_STEP if self.seqs_per_step == "auto" else self.seqs_per_step, n_peaks)

    def _counts_cotangent(self, length: int) -> torch.Tensor:
        """counts_cotangent() at the input length, computed once per length (§7.1)."""
        if length not in self._cotangents:
            self._cotangents[length] = counts_cotangent(
                self.weights, group=self.group, length=length, rows=self.head_rows
            )
        return self._cotangents[length]

    def _describe(self, N: int, L: int, K: int, S: int) -> dict:
        return {
            "output": self.output,
            "group": self.group,
            "device": str(self.weights.device),
            "dtype": str(self.weights.dtype).removeprefix("torch."),
            "forward_stats": self.forward_stats,
            "head_rows": self.head_rows,
            "n_sequences": N,
            "length": L,
            "n_shuffles": K,
            "pairs": N * K,
            "random_state": self.random_state,
            "seqs_per_step": S,
            "steps": 0,
            "padded_peaks": 0,
        }


def delta_summary(deltas: torch.Tensor, threshold: float | None) -> dict:
    """Convergence deltas in brief (§9): pairs, threshold, how many exceed it, max, p50, p99 and p99.9.

    A delta exceeds the threshold as tangermeme decides it, ``deltas > threshold`` in the deltas' dtype
    (TM/deep_lift_shap.py:744). `above` is None when the threshold is.
    """
    values = deltas.detach().cpu()
    p50, p99, p999 = np.quantile(values.double().flatten().numpy(), [0.5, 0.99, 0.999])
    return {
        "pairs": values.numel(),
        "threshold": threshold,
        "above": None if threshold is None else int((values > threshold).sum()),
        "max": float(values.max()),
        "p50": float(p50),
        "p99": float(p99),
        "p99.9": float(p999),
    }


def delta_warning(summary: dict) -> str:
    """The one warning a run gives for high deltas, where tangermeme warns per batch with the whole batch's deltas."""
    return (
        f"Convergence deltas too high: {summary['above']} of {summary['pairs']} pairs > {summary['threshold']:g} "
        f"(max {summary['max']:.4g}, p50/p99/p99.9 {summary['p50']:.3g}/{summary['p99']:.3g}/{summary['p99.9']:.3g})"
    )


def _positive_int(value: object, name: str, alternatives: str = "") -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f"{name} must be {alternatives}a positive integer, got {value!r}")
    return int(value)
