"""Tiny Cherimoya models, one-hot inputs, references and a tangermeme oracle for the cheri_dls tests.

This is design §15.1. Import it only after ``pytest.importorskip("torch")``: it imports torch, tangermeme and
cherimoya at module level. CH/ below is the installed cherimoya package and TM/ the installed tangermeme.

Models. CONFIGS holds the three configurations, each built with ``Cherimoya(..., compile=False, random_state=seed)``:
- A: C=16, 3 layers, expansion 2, groups [1], no controls, L=256, trimming 53.
- B: C=8, 2 layers, expansion 1, groups [1, 2], one control track, L=200. Both heads attribute group 1.
- C: C=128, 2 layers, expansion 2, groups [1], L=512. These are the real GEMM widths, C=128 and H=256.

The default init (trunc_normal, std 0.02; CH/cherimoya.py:263-268, CH/cheri.py:989-994) leaves GELU nearly linear.
redraw_weights() therefore redraws every weight the forward reads, from N(0, std) with one seeded generator:

| weights | std |
|---|---|
| iconv weight | 0.5 |
| iconv bias | 0.3 |
| conv_weight | 0.8 |
| linear1 | 1.5/sqrt(C) |
| linear2 | 1/sqrt(H) |
| fconv weight | 0.1 |
| fconv bias | 0.1 |
| count linear weight | 1 |
| count linear bias | 0.1 |

A float64 model is the float32 one after ``.double()``, so its weights hold the same values exactly.

float64 forwards. Stock Cherimoya 0.3.0 cannot run a float64 forward. _forward_impl always computes the counts head,
and ``torch.mean(X[..].float())`` (CH/cherimoya.py:458) hands a float32 tensor to the float64 count Linear, which
raises "mat1 and mat2 must have the same dtype". This happens under either wrapper. enable_float64() rebinds the
forward of one model instance to the module's own _forward_impl source, with its two ``.float()`` casts
(CH/cherimoya.py:458, 460) replaced by casts to the count Linear's dtype:
- For float32 parameters that is the same op: ``.to(float32)`` returns a float32 tensor itself, as ``.float()`` does.
- Every other line is unchanged, and so is every module call, so tangermeme's hooks still fire.
- The normalization's own fp32 cast (CH/cheri.py:127) is untouched. That cast is what bounds a float64 oracle.

Inputs: sequences() gives 12 random sequences and 1 AT-rich one, as int8 one-hots like extract_loci's.

References, each (N, K, 4, L) float32, injected identically into both sides of a comparison:
- shuffled_references(): exactly the dinucleotide shuffles deep_lift_shap draws itself.
- identical_references(): r = x.
- mutated_references(): x with 3 positions changed.
- random_references(): random one-hots.

Oracle: oracle() runs tangermeme's deep_lift_shap as `cherimoya attribute` calls it, but with injected references
and on CPU. It returns the result together with tangermeme's own per-pair convergence deltas. capture_deltas()
catches those deltas through a module-global ``print`` (§14).
"""

from __future__ import annotations

import builtins
import contextlib
import importlib
import inspect
import math
import sys
import textwrap
import types
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch
from cherimoya import Cherimoya, ControlWrapper, LogCountWrapper, ProfileWrapper
from cherimoya.deep_lift_shap import attribution_ops
from tangermeme.ersatz import dinucleotide_shuffle

# The submodule, not the function of the same name.
TDLS = importlib.import_module("tangermeme.deep_lift_shap")


@dataclass(frozen=True)
class TinyConfig:
    """A tiny Cherimoya architecture, its input length, and the group its heads attribute."""

    name: str
    n_filters: int
    n_layers: int
    expansion: int
    signal_groups: tuple[int, ...]
    n_control_tracks: int
    length: int
    group: int


CONFIGS = {
    "A": TinyConfig("A", 16, 3, 2, (1,), 0, 256, 0),
    "B": TinyConfig("B", 8, 2, 1, (1, 2), 1, 200, 1),
    "C": TinyConfig("C", 128, 2, 2, (1,), 0, 512, 0),
}

N_RANDOM, N_AT_RICH = 12, 1


def _config(config: str | TinyConfig) -> TinyConfig:
    return CONFIGS[config] if isinstance(config, str) else config


def redraw_weights(model: Cherimoya, seed: int) -> Cherimoya:
    """Redraw, in place, every weight the forward reads from N(0, std), with the stds in the module docstring."""
    gen = torch.Generator().manual_seed(seed)
    C = model.n_filters
    H = model.expansion * C

    def draw(param: torch.Tensor, std: float) -> None:
        param.copy_(torch.randn(param.shape, generator=gen, dtype=torch.float64) * std)

    with torch.no_grad():
        draw(model.iconv.weight, 0.5)
        draw(model.iconv.bias, 0.3)
        for block in model.blocks:
            draw(block.conv.conv_weight, 0.8)
            draw(block.linear1.weight, 1.5 / math.sqrt(C))
            draw(block.linear2.weight, 1.0 / math.sqrt(H))
        draw(model.fconv.weight, 0.1)
        draw(model.fconv.bias, 0.1)
        draw(model.linear.weight, 1.0)
        draw(model.linear.bias, 0.1)
    return model


def tiny_model(config: str | TinyConfig = "A", *, seed: int = 0, dtype: torch.dtype = torch.float32) -> Cherimoya:
    """A tiny Cherimoya with redrawn weights, float32, or float64 with enable_float64() applied."""
    cfg = _config(config)
    model = Cherimoya(
        n_filters=cfg.n_filters,
        n_layers=cfg.n_layers,
        signal_groups=list(cfg.signal_groups),
        n_control_tracks=cfg.n_control_tracks,
        expansion=cfg.expansion,
        verbose=False,
        compile=False,
        random_state=seed,
    )
    redraw_weights(model, seed)
    if dtype == torch.float64:
        return enable_float64(model.double())
    if dtype != torch.float32:
        raise ValueError(f"tiny models are float32 or float64, not {dtype}")
    return model


def enable_float64(model: Cherimoya) -> Cherimoya:
    """Let `model` run its own forward in float64: its _forward_impl with the two head casts widened (see module doc).

    Only this instance changes. Cherimoya.forward calls ``self._forward_fn`` (CH/cherimoya.py:411), which an eager
    model sets to its bound _forward_impl, and this rebinds that one attribute.
    """
    source = textwrap.dedent(inspect.getsource(Cherimoya._forward_impl))
    if source.count(".float()") != 2:
        raise RuntimeError("Cherimoya._forward_impl no longer has exactly the two .float() casts this widens")
    source = source.replace(".float()", ".to(self.linear.weight.dtype)")
    namespace = dict(vars(sys.modules[Cherimoya.__module__]))
    exec(compile(source, "<Cherimoya._forward_impl with widened head casts>", "exec"), namespace)
    model._forward_fn = types.MethodType(namespace["_forward_impl"], model)
    return model


def onehot(tokens: np.ndarray) -> torch.Tensor:
    """Base indices 0-3 of shape (..., L) to an int8 one-hot (..., 4, L): extract_loci's dtype."""
    expanded = np.eye(4, dtype=np.int8)[np.asarray(tokens)]
    return torch.from_numpy(np.ascontiguousarray(np.moveaxis(expanded, -1, -2)))


def random_onehot(n: int, length: int, *, seed: int) -> torch.Tensor:
    """n uniformly random sequences, int8 (n, 4, length)."""
    return onehot(np.random.RandomState(seed).randint(0, 4, (n, length)))


def at_rich_onehot(n: int, length: int, *, seed: int, at_fraction: float = 0.9) -> torch.Tensor:
    """n sequences in which each position is A or T with probability at_fraction, int8 (n, 4, length)."""
    rng = np.random.RandomState(seed)
    at = rng.random_sample((n, length)) < at_fraction
    return onehot(np.where(at, rng.choice([0, 3], (n, length)), rng.choice([1, 2], (n, length))))


def sequences(config: str | TinyConfig, *, seed: int = 0) -> torch.Tensor:
    """The inputs of §15.1 at the config's length: 12 random sequences, then 1 AT-rich one, int8 (13, 4, L)."""
    length = _config(config).length
    return torch.cat([random_onehot(N_RANDOM, length, seed=seed), at_rich_onehot(N_AT_RICH, length, seed=seed + 1)])


def _tokens(X: torch.Tensor) -> np.ndarray:
    """The base index at each position of a one-hot (..., 4, L) with exactly one base per position."""
    x = torch.as_tensor(X)
    if not bool((x.sum(dim=-2) == 1).all()):
        raise ValueError("expected a one-hot with exactly one base at every position")
    return x.argmax(dim=-2).numpy()


def shuffled_references(X: torch.Tensor, k: int = 5, random_state: int = 0) -> torch.Tensor:
    """The references deep_lift_shap draws itself, float32 (N, k, 4, L).

    Pair (e, j) gets ``dinucleotide_shuffle(X[e:e+1].float(), n=1, random_state=random_state + j)[:, 0]``
    (TM/deep_lift_shap.py:706-708).
    """
    return torch.stack(
        [
            torch.cat(
                [dinucleotide_shuffle(X[e : e + 1].float(), n=1, random_state=random_state + j)[:, 0] for j in range(k)]
            )
            for e in range(len(X))
        ]
    )


def identical_references(X: torch.Tensor, k: int) -> torch.Tensor:
    """r = x for every pair, float32 (N, k, 4, L). Every rule then falls back to the ordinary gradient."""
    return X.float().unsqueeze(1).repeat(1, k, 1, 1)


def mutated_references(X: torch.Tensor, k: int, *, n_mutations: int = 3, seed: int = 0) -> torch.Tensor:
    """x with `n_mutations` distinct positions changed to another base, float32 (N, k, 4, L)."""
    tokens = _tokens(X)
    n, length = tokens.shape
    rng = np.random.RandomState(seed)
    out = np.repeat(tokens[:, None], k, axis=1)
    for e in range(n):
        for j in range(k):
            pos = rng.choice(length, n_mutations, replace=False)
            out[e, j, pos] = (out[e, j, pos] + rng.randint(1, 4, n_mutations)) % 4
    return onehot(out).float()


def random_references(X: torch.Tensor, k: int, *, seed: int = 0) -> torch.Tensor:
    """Random one-hots, unrelated to X, float32 (N, k, 4, L)."""
    n, _, length = X.shape
    return random_onehot(n * k, length, seed=seed).view(n, k, 4, length).float()


def wrapper(model: Cherimoya, output: str = "counts", group: int | None = 0) -> torch.nn.Module:
    """The module `cherimoya attribute` attributes: LogCountWrapper or ProfileWrapper over ControlWrapper."""
    inner = ControlWrapper(model)
    if output == "counts":
        return LogCountWrapper(inner, group=group)
    if output == "profile":
        return ProfileWrapper(inner, group=group)
    raise ValueError(f"output must be counts or profile, got {output!r}")


class OracleResult(NamedTuple):
    """oracle()'s output."""

    attributions: torch.Tensor  # raw_outputs: the multipliers (N, K, 4, L); otherwise hypothetical scores (N, 4, L)
    deltas: torch.Tensor  # (N, K) tangermeme's own convergence deltas, pair (e, k) at [e, k]


@contextlib.contextmanager
def capture_deltas() -> Iterator[list[torch.Tensor]]:
    """Capture, as tensors, the convergence deltas deep_lift_shap prints with print_convergence_deltas=True (§14).

    ``print(convergence_deltas)`` (TM/deep_lift_shap.py:749) is that module's only print. A module-global ``print``
    shadows the builtin there. The sink appends a detached copy of each per-batch delta tensor to the yielded list,
    and passes any other call on to the builtin. The global is removed on exit, also when the body raises.
    """
    if "print" in vars(TDLS):
        raise RuntimeError("tangermeme.deep_lift_shap already has a module-global print")
    captured: list[torch.Tensor] = []

    def sink(*args: object, **kwargs: object) -> None:
        if len(args) == 1 and not kwargs and isinstance(args[0], torch.Tensor):
            captured.append(args[0].detach().clone())
        else:
            builtins.print(*args, **kwargs)

    TDLS.print = sink
    try:
        yield captured
    finally:
        vars(TDLS).pop("print", None)


def oracle(
    model: torch.nn.Module,
    X: torch.Tensor,
    references: torch.Tensor,
    *,
    raw_outputs: bool,
    dtype: torch.dtype | str,
    batch_size: int = 7,
    warning_threshold: float = 1e-3,
    num_threads: int = 1,
) -> OracleResult:
    """tangermeme's deep_lift_shap on CPU with injected references (N, K, 4, L), plus its convergence deltas.

    The call is `cherimoya attribute`'s (attribute.py:88-101): hypothetical=True and cherimoya's attribution_ops(). The
    differences: the references are injected, the device is CPU, and the batch size defaults to 7, which splits
    sequences across batches. `model` is the wrapper to attribute; see wrapper(). Pass a float64 model with
    dtype=torch.float64. tangermeme's per-batch "Convergence deltas too high" warnings are silenced, because the
    deltas themselves come back. torch runs `num_threads` intra-op threads during the call (1, as §15.1 asks), and the
    previous setting is restored afterwards.
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    try:
        with warnings.catch_warnings(), capture_deltas() as captured:
            warnings.filterwarnings("ignore", message="Convergence deltas too high", category=RuntimeWarning)
            out = TDLS.deep_lift_shap(
                model,
                X,
                references=references,
                batch_size=batch_size,
                hypothetical=True,
                raw_outputs=raw_outputs,
                warning_threshold=warning_threshold,
                additional_nonlinear_ops=attribution_ops(),
                print_convergence_deltas=True,
                dtype=dtype,
                device="cpu",
            )
    finally:
        torch.set_num_threads(previous)
    deltas = torch.cat(captured).reshape(len(X), references.shape[1])
    return OracleResult(out, deltas)
