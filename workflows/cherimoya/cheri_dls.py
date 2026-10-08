"""cheri_dls: a fast, numerically faithful DeepLIFT/DeepSHAP engine for Cherimoya models.

`cherimoya attribute` runs tangermeme's hook-based deep_lift_shap over 2B-row batches: each sequence is forwarded
once per reference, and the backward runs over both halves of the batch. This engine computes the same multipliers
from a hand-written forward and backward. The forward runs over a joint batch per step: each peak's sequence once,
then its K references. The backward runs over the sequence half only. The design is
analyses/attr_pilot/design/final_design.md; the section numbers below refer to it. CH/ is the installed cherimoya
package and TM/ the installed tangermeme.

This module holds the weights, the forward pass, the backward rules and the epilogue (§5-§9). Every function runs
eagerly; compilation comes later.

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

Two deliberate differences from the reference:
- The counts head casts the pooled stream to the count Linear's dtype, where CH/cherimoya.py:458 calls ``.float()``.
  Every supported model has fp32 parameters, so the two are the same operation. A float64 model on CPU (tests and the
  fp64 "truth" run only) gets a float64 head. With ``.float()`` the stock module cannot run a float64 forward at all:
  the float32 pooled stream meets the float64 Linear and raises. counts_cotangent() inherits the cast.
- mean_over_references() writes each window into a zeroed full-length block before averaging, where design §9 averages
  the (K, 4, W) window block alone. The CPU mean sums a column in an order that depends on where the column sits in
  its block, and the window alone moves some columns. Measured: for K=20, L=2114, W=100 in float32, the window-only
  mean differs from tangermeme's in 9 of 400 elements; the full-length block matches it bitwise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from os import PathLike
from typing import NamedTuple

import cherimoya
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
    - delta (S, K): |(y_x - y_r) - sum((x - r) * m)| on the raw multipliers, tangermeme's convergence delta
      (TM/deep_lift_shap.py:739-742).
    """
    L = X1h.shape[-1]
    if not 0 <= start < end <= L:
        raise ValueError(f"the window [{start}, {end}) must be non-empty and inside [0, {L})")
    x, r = split_rows(X1h, S, K)
    m = m.view(S, K, *m.shape[1:])
    delta = ((y_x[:, None] - y_r) - ((x - r) * m).sum(dim=(-2, -1))).abs()
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
