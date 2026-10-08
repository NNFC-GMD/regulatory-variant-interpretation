"""cheri_dls's forward pass and whole engine against Cherimoya's own forward and tangermeme's deep_lift_shap, on CPU.

Part 1, the weights and the forward pass (acceptance criterion C1)
------------------------------------------------------------------
C1. The "cpu_mirror" forward runs the reference CPU path op for op (cherimoya/cheri.py:54-87, 124-132). On the same
rows it must therefore give the module's own output:
- float64: |dy| <= 1e-12 * max(1, |y|).
- float32: |dy| <= 2e-6 * max(1, |y|). Bitwise equality is expected and reported.

The module runs as deep_lift_shap runs it: LogCountWrapper(ControlWrapper(model), group)(X)[:, 0], under grad mode,
on inputs that require grad. Ours runs under no_grad. Both see the joint batch the engine forwards: 13 sequences
(12 random, 1 AT-rich), then the 5 dinucleotide shuffles of each. A, B and C are the tiny models of
tests/dls_tiny.py. B has a control track and two count groups, and attributes group 1.

Besides the counts output y, the tests compare:
- the final residual stream, which the profile head would read, against fconv's input;
- each block's saved state against the module's own activations, captured with forward hooks:
  - y against cherimoya's _cheri_conv of the block's input, the conv the DeepLIFT rule recomputes
    (cherimoya/deep_lift_shap.py:143);
  - u against linear1's output;
  - mu and v against tangermeme's own expressions for the rule's statistics.

float64 needs one adjustment. Stock Cherimoya 0.3.0 cannot run a float64 forward (see dls_tiny.enable_float64), so
the float64 module is the module's own forward with its two head casts widened. A test pins that this widening
changes nothing in float32.

Also covered:
- two_pass and one_pass against cpu_mirror;
- head_forward's group slicing and control zeros;
- CheriWeights: it matches the module, both depthwise key spellings load, and bad shapes, version drift and
  architecture changes raise;
- the dls_tiny oracle and its delta sink.

Part 2, the whole engine: Engine.raw and Engine.run (acceptance criteria C3-C10)
---------------------------------------------------------------------------------
Both sides get the same 13 sequences and the same injected references, K = 5 per sequence, of four kinds (dls_tiny):
dinucleotide shuffles, random one-hots, r = x, and x with 3 mutated positions.
- The oracle is deep_lift_shap as `cherimoya attribute` calls it, on CPU, at batch_size 7 and 1 thread. 7 is odd, so
  a sequence's pairs split across batches. Its convergence deltas come from its print sink.
- The engine runs at the default thread count, in steps of 8 peaks unless a test sets the step size. 13 peaks then
  fill one step and pad the second with 3 copies of the last peak. Its counts cotangent is taken at the oracle's
  full-batch row count, 2 x 7.

forward_stats is "cpu_mirror" everywhere except C6 ("two_pass") and C9 (both). cpu_mirror normalizes in float32, as
the module does (cherimoya/cheri.py:127), even in a float64 model. So its forward is the oracle's, while two_pass sits
about 1e-8 away from the oracle in float64. Conversely, two_pass is conservative to float64 accuracy, and cpu_mirror
only to float32 accuracy.

- C3: per-pair rel-L2 of the raw multipliers over (4, L) against deep_lift_shap(raw_outputs=True).
  - float64: max <= 1e-9.
  - float32, shuffled and random references only: median <= 1e-6 and max <= 1e-5.
- C4: per-sequence rel-L2 of the hypothetical attributions against deep_lift_shap(hypothetical=True), over the full
  length and over attribute.py's centred window. float64: <= 1e-9; float32: <= 1e-5.
- C5: in float64, |delta_ours - delta_ref| <= 1e-9 * max(1, |y_x - y_r|) for every pair. float32 deltas are reported,
  not bounded: they are the size of an ulp of y.
- C6: in float64 with two_pass, |delta| <= 1e-10 * max(1, |y_x - y_r|) for every pair.
- C7: with r = x the multipliers are the plain gradient, autograd through the float64 wrapper: per-pair rel-L2 <= 1e-6.
  The autograd path's float32 normalization sets the bound.
- C8: with 3 mutated positions, some GELU inputs differ by less than 1e-6 between a sequence and its reference, so the
  rule falls back to the gradient there. At least one such element must exist, and C3's float64 bound must hold.
- C9: in float64, steps of S = 3 and 8 peaks against S = 1, for 13 peaks: per-pair and per-sequence rel-L2 <= 1e-12,
  and the same relative bound on the deltas and outputs.
- C10: on every model, the DeepLIFT multipliers for shuffled references differ from the plain gradient by at least 10%
  per-pair rel-L2. So the rules are exercised.

Also covered:
- references produced by dls_refs.RefProducer, in process and in a 2-process pool, equal the injected shuffles bit
  for bit;
- the engine's argument checks, numpy inputs, and group None on a one-group model;
- a single summary warning for high deltas, with tangermeme's strict comparison against the threshold.

Measured values are written to the terminal.
"""

from __future__ import annotations

import copy
import functools
import re
import sys
import warnings
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tangermeme")
pytest.importorskip("cherimoya")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflows" / "cherimoya"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cheri_dls  # noqa: E402
import dls_tiny  # noqa: E402
from cherimoya import ControlWrapper, LogCountWrapper  # noqa: E402
from cherimoya.cheri import CONV_NORM_EPS, _cheri_conv  # noqa: E402
from cherimoya.deep_lift_shap import attribution_ops  # noqa: E402

K = 5  # references per sequence in the joint batch
C1_TOL = {torch.float32: 2e-6, torch.float64: 1e-12}  # acceptance criterion C1
E1_FP32_TOL = 5e-5  # E1's strict-fp32 forward bound: a different statistics formula and reduction order
DTYPES = [pytest.param(torch.float32, id="float32"), pytest.param(torch.float64, id="float64")]
CONFIG_NAMES = list(dls_tiny.CONFIGS)


def _report(config: pytest.Config, line: str) -> None:
    """Write a measurement to the terminal, past pytest's output capture."""
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(f"[cheri_dls] {line}")


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """max over elements of |a - b| / max(1, |b|)."""
    return float(((a - b).abs() / b.abs().clamp_min(1)).max())


def _tm_rule_stats(y: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row mu and v as _layer_normalization_helper computes them (tangermeme/_deep_lift_utils.py:503-527).

    The helper is applied to y of shape (R, L, C) through cherimoya's _ConvNormView, with normalized_shape [L, C].
    """
    norm_dims = list(range(-2, 0))
    mu = y.mean(dim=norm_dims, keepdim=True)
    a = y - mu
    var = (a**2).mean(dim=norm_dims, keepdim=True)
    v = (var + eps) ** (-0.5)
    return mu.flatten(), v.flatten()


@pytest.fixture(scope="module")
def batches() -> dict[str, torch.Tensor]:
    """Per config, the engine's joint batch: 13 sequences, then their K shuffles, float32 (13 + 13K, 4, L)."""
    out = {}
    for name in CONFIG_NAMES:
        X = dls_tiny.sequences(name)
        out[name] = torch.cat([X.float(), dls_tiny.shuffled_references(X, K).flatten(0, 1)])
    return out


def _module_forward(model: torch.nn.Module, X1h: torch.Tensor, group: int | None) -> tuple[torch.Tensor, dict]:
    """The module's counts output as deep_lift_shap computes it, plus activations captured with forward hooks.

    The captured activations are fconv's input (the residual stream, then any control zeros), each block's input,
    and each block's linear1 output.
    """
    acts: dict = {"block_in": [], "u": []}
    handles = [model.fconv.register_forward_pre_hook(lambda m, args: acts.update(fconv_in=args[0].detach().clone()))]
    for block in model.blocks:
        handles.append(block.conv.register_forward_pre_hook(lambda m, args: acts["block_in"].append(args[0].detach())))
        handles.append(block.linear1.register_forward_hook(lambda m, args, out: acts["u"].append(out.detach())))
    try:
        with torch.enable_grad():
            y = LogCountWrapper(ControlWrapper(model), group=group)(X1h.clone().requires_grad_())[:, 0]
    finally:
        for handle in handles:
            handle.remove()
    return y.detach(), acts


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("config", CONFIG_NAMES)
def test_c1_cpu_mirror_forward_matches_module(config, dtype, batches, pytestconfig):
    cfg = dls_tiny.CONFIGS[config]
    model = dls_tiny.tiny_model(config, dtype=dtype)
    X1h = batches[config].to(dtype)
    y_ref, acts = _module_forward(model, X1h, cfg.group)

    w = cheri_dls.CheriWeights.from_model(model)
    with torch.no_grad():
        ours = cheri_dls.model_forward(X1h, w, forward_stats="cpu_mirror", output="counts", group=cfg.group)
    assert ours.y.dtype == dtype and ours.y.shape == y_ref.shape == (len(X1h),)
    assert len(ours.saved) == len(acts["block_in"]) == len(acts["u"]) == cfg.n_layers

    trunk = (ours.hL.transpose(1, 2), acts["fconv_in"][:, : w.C])  # fconv reads cat([H, controls]) (cherimoya.py:453)
    saved = []
    for s, block, block_in, u in zip(ours.saved, w.blocks, acts["block_in"], acts["u"]):
        y_rule = _cheri_conv(block_in, block.cw, block.d)
        mu, v = _tm_rule_stats(y_rule, w.eps)
        saved += [(s.y, y_rule), (s.u, u), (s.mu, mu), (s.v, v)]

    err_y, err_trunk = _rel(ours.y, y_ref), _rel(*trunk)
    err_saved = max(_rel(a, b) for a, b in saved)
    bitwise = {
        "y": torch.equal(ours.y, y_ref),
        "trunk": torch.equal(*trunk),
        "saved": all(torch.equal(a, b) for a, b in saved),
    }
    flags = ", ".join(f"{name} {'yes' if equal else 'no'}" for name, equal in bitwise.items())
    _report(
        pytestconfig,
        f"C1 config {config} {str(dtype)[6:]}: {len(X1h)} rows, max |dy|/max(1,|y|) = {err_y:.3g} "
        f"(tol {C1_TOL[dtype]:g}), trunk {err_trunk:.3g}, saved y/u/mu/v {err_saved:.3g}; bitwise: {flags}",
    )
    assert err_y <= C1_TOL[dtype]
    assert err_trunk <= C1_TOL[dtype]
    assert err_saved <= C1_TOL[dtype]


def test_float64_patch_is_the_stock_forward_in_float32(batches):
    """enable_float64 widens only the head casts: in float32 both heads are bitwise the stock forward's."""
    for name in CONFIG_NAMES:
        stock = dls_tiny.tiny_model(name)
        patched = dls_tiny.enable_float64(copy.deepcopy(stock))
        X1h = batches[name][:20]
        with torch.enable_grad():
            profile, counts = ControlWrapper(stock)(X1h.clone().requires_grad_())
            profile_p, counts_p = ControlWrapper(patched)(X1h.clone().requires_grad_())
        assert torch.equal(profile, profile_p) and torch.equal(counts, counts_p)
        assert patched._forward_fn.__func__ is not type(stock)._forward_impl


def test_stock_module_cannot_run_float64(batches):
    """Why enable_float64 exists: cherimoya.py:458 feeds a float32 mean to the float64 count Linear.

    If a later cherimoya fixes this, this test fails, and the float64 comparisons can use the stock module again.
    """
    model = dls_tiny.tiny_model("A").double()
    with pytest.raises(RuntimeError, match="same dtype"):
        ControlWrapper(model)(batches["A"][:2].double())
    with torch.no_grad():
        dls_tiny.enable_float64(model)
        profile, counts = ControlWrapper(model)(batches["A"][:2].double())
    assert profile.dtype == counts.dtype == torch.float64


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("config", CONFIG_NAMES)
def test_two_pass_and_one_pass_compute_the_same_function(config, dtype, batches, pytestconfig):
    """two_pass and one_pass differ from cpu_mirror only in the statistics formula and the reduction order.

    The bound is E1's strict-fp32 forward bound, which covers exactly that kind of difference. In float64,
    cpu_mirror still normalizes in fp32 (cherimoya/cheri.py:127), so its gap to the float64 modes stays at fp32 level.
    one_pass and two_pass are the same function in exact arithmetic, so in float64 they must agree to float64
    accuracy. Block 0 sees the same input in every mode. Its shift-add conv dw3 must therefore match _cheri_conv to
    C2's 1e-14 in float64.
    """
    cfg = dls_tiny.CONFIGS[config]
    w = cheri_dls.CheriWeights.from_model(dls_tiny.tiny_model(config, dtype=dtype))
    X1h = batches[config].to(dtype)
    with torch.no_grad():
        runs = {s: cheri_dls.model_forward(X1h, w, forward_stats=s, group=cfg.group) for s in cheri_dls.FORWARD_STATS}
    mirror = runs["cpu_mirror"]
    errs = {s: _rel(runs[s].y, mirror.y) for s in ("two_pass", "one_pass")}
    errs["dw3 vs _cheri_conv, block 0"] = _rel(runs["two_pass"].saved[0].y, mirror.saved[0].y)
    if dtype == torch.float64:
        errs["one_pass vs two_pass"] = _rel(runs["one_pass"].y, runs["two_pass"].y)
    _report(
        pytestconfig,
        f"forward_stats config {config} {str(dtype)[6:]}, max |d|/max(1,|ref|): "
        + ", ".join(f"{name} {value:.3g}" for name, value in errs.items()),
    )
    assert errs["two_pass"] <= E1_FP32_TOL and errs["one_pass"] <= E1_FP32_TOL
    if dtype == torch.float64:
        assert errs["one_pass vs two_pass"] <= C1_TOL[torch.float64]
        assert errs["dw3 vs _cheri_conv, block 0"] <= 1e-14
    for s in ("two_pass", "one_pass"):  # the saved statistics are the rule's, whatever the forward's formula
        for got in runs[s].saved:
            mu, v = _tm_rule_stats(got.y, w.eps)
            assert torch.equal(got.mu, mu) and torch.equal(got.v, v)


def test_counts_head_groups_and_controls(batches):
    """Config B (two count groups, one control track): every group, and group None (target 0), as the module does."""
    model = dls_tiny.tiny_model("B")
    w = cheri_dls.CheriWeights.from_model(model)
    X1h = batches["B"]
    with torch.no_grad():
        hL = cheri_dls.model_forward(X1h, w, forward_stats="cpu_mirror", group=0).hL
    for group in (0, 1, None):
        y_ref, _ = _module_forward(model, X1h, group)
        with torch.no_grad():
            y = cheri_dls.head_forward(hL, w, output="counts", group=group)
        assert _rel(y, y_ref) <= C1_TOL[torch.float32]

    with torch.no_grad():
        y0, y1 = (cheri_dls.head_forward(hL, w, group=g) for g in (0, 1))
    assert not torch.equal(y0, y1)
    with pytest.raises(NotImplementedError, match="profile"):
        cheri_dls.head_forward(hL, w, output="profile", group=1)
    for bad in (2, -1, True, 0.0):
        with pytest.raises(ValueError, match="group"):
            cheri_dls.head_forward(hL, w, group=bad)
    with pytest.raises(ValueError, match="output"):
        cheri_dls.head_forward(hL, w, output="logits")
    with pytest.raises(ValueError, match="trimming"):
        cheri_dls.head_forward(hL[:, : 2 * w.T], w)


def test_make_wrapper_runs_the_reference_group_checks():
    model = dls_tiny.tiny_model("B")
    for output, group, cls in (("counts", 1, "LogCountWrapper"), ("profile", None, "ProfileWrapper")):
        wrapped = cheri_dls.make_wrapper(model, output, group)
        assert type(wrapped).__name__ == cls and isinstance(wrapped.model, ControlWrapper)
    with pytest.raises(ValueError, match="attributes one output"):
        cheri_dls.make_wrapper(model, "counts", None)  # attribute.py:43-47
    for output in ("counts", "profile"):
        for bad in (2, True):
            with pytest.raises(ValueError, match="group must be"):
                cheri_dls.make_wrapper(model, output, bad)  # wrappers.py:67-85
    with pytest.raises(ValueError, match="output"):
        cheri_dls.make_wrapper(model, "logits", 0)


def test_forward_stats_resolution_and_validation(batches):
    assert cheri_dls.resolve_forward_stats("auto", "cpu") == "cpu_mirror"
    assert cheri_dls.resolve_forward_stats("auto", torch.device("cuda", 0)) == "two_pass"
    assert cheri_dls.resolve_forward_stats("one_pass", "cpu") == "one_pass"
    w = cheri_dls.CheriWeights.from_model(dls_tiny.tiny_model("A"))
    for bad in ("three_pass", "", None):
        with pytest.raises(ValueError, match="forward_stats"):
            cheri_dls.resolve_forward_stats(bad, "cpu")
    z0 = cheri_dls.stem_forward(batches["A"][:2], w.W0, w.b0)
    with pytest.raises(ValueError, match="stats"):
        cheri_dls.forward_save(z0, w.blocks, w.rs, w.eps, "auto")


def _module_tensors(model) -> list[torch.Tensor]:
    tensors = [model.iconv.weight, model.iconv.bias]
    for block in model.blocks:
        tensors += [block.conv.conv_weight, block.linear1.weight, block.linear2.weight]
    return tensors + [model.fconv.weight, model.fconv.bias, model.linear.weight, model.linear.bias]


def _weights_tensors(w: cheri_dls.CheriWeights) -> list[torch.Tensor]:
    tensors = [w.W0, w.b0]
    for block in w.blocks:
        tensors += [block.cw, block.W1, block.W2]
    return tensors + [w.Wf, w.bf, w.Wl, w.bl]


@pytest.mark.parametrize("config", CONFIG_NAMES)
def test_weights_are_the_module_tensors(config):
    cfg = dls_tiny.CONFIGS[config]
    for dtype in (torch.float32, torch.float64):
        model = dls_tiny.tiny_model(config, dtype=dtype)
        w = cheri_dls.CheriWeights.from_model(model)
        for ours, theirs in zip(_weights_tensors(w), _module_tensors(model), strict=True):
            assert torch.equal(ours, theirs) and ours.dtype == dtype and not ours.requires_grad
        assert [b.d for b in w.blocks] == [2**i for i in range(cfg.n_layers)]
        assert (w.T, w.rs, w.eps) == (model.trimming, model.residual_scale, CONV_NORM_EPS)
        assert (w.C, w.H, w.n_ctl) == (cfg.n_filters, cfg.expansion * cfg.n_filters, cfg.n_control_tracks)
        assert (w.n_out, w.n_groups, w.signal_groups) == (
            sum(cfg.signal_groups),
            len(cfg.signal_groups),
            cfg.signal_groups,
        )
        assert (w.dtype, w.device) == (dtype, torch.device("cpu"))
    with pytest.raises(ValueError, match="CPU only"):
        cheri_dls.CheriWeights.from_model(model, device="meta")  # float64 is for CPU tests only


def test_both_depthwise_key_spellings_load(tmp_path, batches):
    """Cherimoya.load takes blocks.{i}.conv_weight (as saved) and blocks.{i}.conv.conv_weight (cheri.py:1045-1061)."""
    model = dls_tiny.tiny_model("A")
    saved = tmp_path / "saved.torch"
    model.save(saved)
    payload = torch.load(saved, weights_only=True)
    old = re.compile(r"^blocks\.(\d+)\.conv_weight$")
    assert sum(bool(old.match(key)) for key in payload["state_dict"]) == model.n_layers
    renamed = tmp_path / "renamed.torch"
    state = {old.sub(r"blocks.\1.conv.conv_weight", key): value for key, value in payload["state_dict"].items()}
    assert sum(key.endswith(".conv.conv_weight") for key in state) == model.n_layers
    torch.save({"config": payload["config"], "state_dict": state}, renamed)

    outputs = []
    for path in (saved, renamed):
        loaded = cheri_dls.load_model(path)
        w = cheri_dls.CheriWeights.from_model(loaded)
        for ours, theirs in zip(_weights_tensors(w), _module_tensors(model), strict=True):
            assert torch.equal(ours, theirs)
        with torch.no_grad():
            outputs.append(cheri_dls.model_forward(batches["A"], w, forward_stats="cpu_mirror").y)
    assert torch.equal(*outputs)


def test_bad_shapes_and_architecture_changes_are_refused(tmp_path):
    model = dls_tiny.tiny_model("A")
    C, H = 16, 32

    payload = model._checkpoint()
    payload["state_dict"]["blocks.1.conv_weight"] = torch.zeros(3, C + 1)
    bad = tmp_path / "bad.torch"
    torch.save(payload, bad)
    with pytest.raises(RuntimeError, match="size mismatch"):
        cheri_dls.load_model(bad)  # the reference's strict loader refuses it

    def replace(path: str, value):
        def tamper(m):
            *parents, name = path.split(".")
            target = m
            for part in parents:
                target = target[int(part)] if part.isdigit() else getattr(target, part)
            setattr(target, name, value)

        return tamper

    tampers = {
        "depthwise taps": replace("blocks.0.conv.conv_weight", torch.nn.Parameter(torch.zeros(3, C + 1))),
        "linear1 with a bias": replace("blocks.1.linear1", torch.nn.Linear(C, H)),
        "linear2 shape": replace("blocks.2.linear2", torch.nn.Linear(H + 1, C, bias=False)),
        "count head shape": replace("linear", torch.nn.Linear(C + 1, 1)),
        "count head without bias": replace("linear", torch.nn.Linear(C, 1, bias=False)),
        "fconv channels": replace("fconv", torch.nn.Conv1d(C + 1, 1, 75, padding=37)),
        "fconv padding": replace("fconv", torch.nn.Conv1d(C, 1, 75, padding=36)),
        "iconv kernel": replace("iconv", torch.nn.Conv1d(4, C, 19, padding=9)),
        "iconv dilation": replace("iconv", torch.nn.Conv1d(4, C, 21, padding=10, dilation=2)),
        "erf GELU in a block": replace("blocks.1.activation", torch.nn.GELU()),
        "erf GELU in the stem": replace("igelu", torch.nn.GELU()),
        "dilation": replace("blocks.2.dilation", 3),
        "residual scale": replace("blocks.0.residual_scale", 0.2),
        "a block missing": replace("blocks", model.blocks[:2]),
        "float16": lambda m: m.half(),
        "mixed dtypes": lambda m: m.linear.double(),
    }
    for name, tamper in tampers.items():
        changed = copy.deepcopy(model)
        tamper(changed)
        with pytest.raises(ValueError, match="contract") as raised:
            cheri_dls.CheriWeights.from_model(changed)
        assert str(raised.value).count("\n") >= 1, name
    with pytest.raises(ValueError, match="Cherimoya"):
        cheri_dls.CheriWeights.from_model(ControlWrapper(model))
    assert cheri_dls.contract_violations(model) == []


def test_version_drift_raises(tmp_path, monkeypatch, caplog):
    model = dls_tiny.tiny_model("A")
    path = tmp_path / "model.torch"
    model.save(path)
    for package in ("cherimoya", "tangermeme"):
        with monkeypatch.context() as patch:
            patch.setattr(sys.modules[package], "__version__", "9.9.9")
            with pytest.raises(RuntimeError, match=f"{package} 9.9.9"):
                cheri_dls.CheriWeights.from_model(model)
            with pytest.raises(RuntimeError, match=f"{package} 9.9.9"):
                cheri_dls.load_model(path)
            caplog.clear()
            with caplog.at_level("WARNING", logger=cheri_dls.logger.name):
                cheri_dls.CheriWeights.from_model(
                    cheri_dls.load_model(path, allow_version_drift=True), allow_version_drift=True
                )
            assert f"{package} 9.9.9" in caplog.text
    assert cheri_dls.check_versions() == cheri_dls.EXPECTED_VERSIONS


def test_tiny_inputs_and_references():
    for name, cfg in dls_tiny.CONFIGS.items():
        X = dls_tiny.sequences(name)
        assert X.dtype == torch.int8 and X.shape == (13, 4, cfg.length)
        assert bool((X.sum(dim=1) == 1).all())
        at = X[:, [0, 3]].sum(dim=(1, 2)).double() / cfg.length
        assert float(at[-1]) >= 0.85 and bool(((at[:-1] > 0.35) & (at[:-1] < 0.65)).all())
        assert torch.equal(X, dls_tiny.sequences(name))

    X = dls_tiny.sequences("A")
    same = dls_tiny.identical_references(X, 3)
    mutated = dls_tiny.mutated_references(X, 3)
    random = dls_tiny.random_references(X, 3)
    shuffled = dls_tiny.shuffled_references(X, 3)
    for refs in (same, mutated, random, shuffled):
        assert refs.dtype == torch.float32 and refs.shape == (13, 3, 4, 256)
        assert bool((refs.sum(dim=2) == 1).all())
    assert torch.equal(same, X.float()[:, None].expand_as(same))
    assert bool(((mutated != X.float()[:, None]).any(dim=2).sum(dim=-1) == 3).all())
    assert bool((shuffled[..., 0] == X.float()[:, None, :, 0]).all())  # the shuffle keeps the first base

    first = dls_tiny.tiny_model("C")
    assert all(torch.equal(a, b) for a, b in zip(_module_tensors(first), _module_tensors(dls_tiny.tiny_model("C"))))
    assert not torch.equal(first.iconv.weight, dls_tiny.tiny_model("C", seed=1).iconv.weight)
    std = float(first.blocks[1].linear1.weight.detach().std())
    assert abs(std / (1.5 / 128**0.5) - 1) < 0.05


def test_shuffled_references_are_deep_lift_shaps_own():
    """With no references passed, deep_lift_shap draws exactly dls_tiny.shuffled_references."""
    model = dls_tiny.tiny_model("A")
    X = dls_tiny.sequences("A")[[0, 1, 12]]  # 12 is the AT-rich one
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Convergence deltas too high", category=RuntimeWarning)
        drawn = dls_tiny.TDLS.deep_lift_shap(
            dls_tiny.wrapper(model),
            X,
            n_shuffles=2,
            random_state=0,
            return_references=True,
            hypothetical=True,
            additional_nonlinear_ops=attribution_ops(),
            dtype="float32",
            device="cpu",
        ).references
    assert torch.equal(drawn, dls_tiny.shuffled_references(X, 2, random_state=0))


def test_oracle_returns_tangermemes_deltas_in_pair_order(pytestconfig):
    """The print sink returns (N, K) deltas: |(y_x - y_r) - sum((x - r) * m)|, per pair, in pair order.

    The deltas are recomputed per batch of 7 pairs, packed example-major as deep_lift_shap packs them
    (tangermeme/deep_lift_shap.py:686-692, 739-742). The recomputation uses the same rows, batch shapes, ops and
    thread count, so equality must be bitwise. A tolerance would hide a pair-order error: an ulp of y is about the
    size of the deltas themselves.
    """
    model = dls_tiny.tiny_model("A")
    X = dls_tiny.sequences("A")
    R = dls_tiny.shuffled_references(X, K)
    wrapped = dls_tiny.wrapper(model, "counts", 0)
    threads = torch.get_num_threads()
    result = dls_tiny.oracle(wrapped, X, R, raw_outputs=True, dtype=torch.float32, batch_size=7)
    assert torch.get_num_threads() == threads and "print" not in vars(dls_tiny.TDLS)
    assert result.attributions.shape == (13, K, 4, 256) and result.deltas.shape == (13, K)

    pairs = [(e, j) for e in range(len(X)) for j in range(K)]
    expected = []
    torch.set_num_threads(1)
    try:
        for b in range(0, len(pairs), 7):
            e, j = (list(t) for t in zip(*pairs[b : b + 7]))
            x, r = X[e].float(), R[e, j]
            with torch.enable_grad():
                y = wrapped(torch.cat([x, r]).requires_grad_())[:, 0].detach()
            dy = y[: len(e)] - y[len(e) :]
            expected.append((dy - ((x - r) * result.attributions[e, j]).sum(dim=(1, 2))).abs())
    finally:
        torch.set_num_threads(threads)
    expected = torch.cat(expected).view(len(X), K)
    bitwise = torch.equal(result.deltas, expected)
    _report(
        pytestconfig,
        f"oracle sink, config A float32: {result.deltas.numel()} deltas, max {float(result.deltas.max()):.3g}, "
        f"recomputed per batch: max diff {float((result.deltas - expected).abs().max()):.3g} "
        f"(bitwise {'yes' if bitwise else 'no'})",
    )
    assert bitwise

    class Boom(torch.nn.Module):
        def forward(self, X):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        dls_tiny.oracle(Boom(), X[:2], R[:2], raw_outputs=True, dtype=torch.float32)
    assert torch.get_num_threads() == threads and "print" not in vars(dls_tiny.TDLS)


def test_oracle_runs_a_float64_model(pytestconfig):
    """The float64 module (enable_float64) attributes under tangermeme's hooks: config B, group 1, controls."""
    model = dls_tiny.tiny_model("B", dtype=torch.float64)
    X = dls_tiny.sequences("B")
    R = dls_tiny.mutated_references(X, 2)
    result = dls_tiny.oracle(dls_tiny.wrapper(model, "counts", 1), X, R, raw_outputs=True, dtype=torch.float64)
    assert result.attributions.dtype == result.deltas.dtype == torch.float64
    assert result.deltas.shape == (13, 2) and bool(torch.isfinite(result.attributions).all())
    _report(pytestconfig, f"oracle float64, config B: max convergence delta {float(result.deltas.max()):.3g}")


# Part 2: the whole engine against tangermeme's deep_lift_shap (C3-C10) -----------------------------------------------

F64 = torch.float64
ORACLE_BATCH = 7  # odd, so a sequence's pairs split across batches (design §15.1)
HEAD_ROWS = 2 * ORACLE_BATCH  # the oracle's full batch: 2 x batch_size rows (§4 head_rows)
N_SEQUENCES = dls_tiny.N_RANDOM + dls_tiny.N_AT_RICH  # 13: no step size tested here divides it

C3_FP64_MAX, C3_FP32_MEDIAN, C3_FP32_MAX = 1e-9, 1e-6, 1e-5
C4_TOL = {torch.float32: 1e-5, F64: 1e-9}
C5_FP64 = 1e-9
C6_FP64 = 1e-10
C7_TOL = 1e-6
C9_TOL = 1e-12
C10_MIN = 0.1

PARITY_CASES = [
    pytest.param(torch.float32, "shuffled", id="float32-shuffled"),
    pytest.param(torch.float32, "random", id="float32-random"),
    pytest.param(F64, "shuffled", id="float64-shuffled"),
    pytest.param(F64, "random", id="float64-random"),
    pytest.param(F64, "identical", id="float64-identical"),  # r = x, C7's references
    pytest.param(F64, "mutated", id="float64-mutated"),  # 3 mutated positions, C8's references
]


@functools.cache
def _case(config: str, kind: str) -> tuple[torch.Tensor, torch.Tensor]:
    """A config's 13 sequences, int8 (13, 4, L), and their K references of one kind, float32 (13, K, 4, L)."""
    X = dls_tiny.sequences(config)
    if kind == "shuffled":
        return X, dls_tiny.shuffled_references(X, K)
    if kind == "random":
        return X, dls_tiny.random_references(X, K, seed=3)
    if kind == "identical":
        return X, dls_tiny.identical_references(X, K)
    if kind == "mutated":
        return X, dls_tiny.mutated_references(X, K, seed=5)
    raise ValueError(kind)


def _window(config: str) -> tuple[int, int]:
    """attribute.py:84-86's window, for an attr_window of a quarter of the input."""
    L = dls_tiny.CONFIGS[config].length
    start = L // 2 - (L // 4) // 2
    return start, start + L // 4


def _engine(config: str, dtype: torch.dtype, **kwargs) -> cheri_dls.Engine:
    """An Engine on a fresh tiny model, attributing the config's group, with the oracle's head rows."""
    model = dls_tiny.tiny_model(config, dtype=dtype)
    return cheri_dls.Engine(model, group=dls_tiny.CONFIGS[config].group, head_rows=HEAD_ROWS, **kwargs)


@functools.cache
def _ours_raw(config: str, dtype: torch.dtype, kind: str, stats: str = "cpu_mirror", step: int | str = "auto"):
    X, R = _case(config, kind)
    return _engine(config, dtype, forward_stats=stats, seqs_per_step=step).raw(X, R)


@functools.cache
def _ours_run(
    config: str,
    dtype: torch.dtype,
    kind: str,
    start: int,
    end: int,
    stats: str = "cpu_mirror",
    step: int | str = "auto",
):
    X, R = _case(config, kind)
    return _engine(config, dtype, forward_stats=stats, seqs_per_step=step).run(X, start, end, references=R)


@functools.cache
def _theirs(config: str, dtype: torch.dtype, kind: str, raw_outputs: bool) -> dls_tiny.OracleResult:
    """tangermeme's deep_lift_shap on the same injected references, batch_size 7, 1 thread."""
    X, R = _case(config, kind)
    wrapped = dls_tiny.wrapper(dls_tiny.tiny_model(config, dtype=dtype), "counts", dls_tiny.CONFIGS[config].group)
    return dls_tiny.oracle(wrapped, X, R, raw_outputs=raw_outputs, dtype=dtype, batch_size=ORACLE_BATCH)


@functools.cache
def _gradient(config: str) -> torch.Tensor:
    """The plain gradient of the attributed count at each sequence, (13, 4, L): autograd through the float64 wrapper.

    No DeepLIFT hooks are involved. The module normalizes in float32 (cherimoya/cheri.py:127), in the backward too.
    """
    wrapped = dls_tiny.wrapper(dls_tiny.tiny_model(config, dtype=F64), "counts", dls_tiny.CONFIGS[config].group)
    X = dls_tiny.sequences(config).to(F64).requires_grad_()
    with torch.enable_grad():
        (grad,) = torch.autograd.grad(wrapped(X)[:, 0].sum(), X)
    return grad


def _rows_rel_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """||a - b||_2 / max(||b||_2, 1e-30) for each row (first axis), in float64."""
    a, b = a.double().flatten(1), b.double().flatten(1)
    return (a - b).norm(dim=1) / b.norm(dim=1).clamp_min(1e-30)


def _dy(result) -> torch.Tensor:
    """|y_x - y_r| per pair, (N, K), in float64."""
    return (result.y_x[:, None] - result.y_r).double().abs()


def _name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


@pytest.mark.parametrize("dtype, kind", PARITY_CASES)
@pytest.mark.parametrize("config", CONFIG_NAMES)
def test_c3_c5_raw_multipliers_and_deltas_match_tangermeme(config, dtype, kind, pytestconfig):
    """Engine.raw against deep_lift_shap(raw_outputs=True) and its print-sink deltas, on the same references."""
    L = dls_tiny.CONFIGS[config].length
    ours, theirs = _ours_raw(config, dtype, kind), _theirs(config, dtype, kind, True)
    assert ours.m.shape == theirs.attributions.shape == (N_SEQUENCES, K, 4, L)
    assert ours.deltas.shape == ours.y_r.shape == (N_SEQUENCES, K) and ours.y_x.shape == (N_SEQUENCES,)
    assert ours.m.dtype == ours.deltas.dtype == ours.y_x.dtype == ours.y_r.dtype == dtype
    assert ours.debug["seqs_per_step"] == 8 and ours.debug["steps"] == 2 and ours.debug["padded_peaks"] == 3

    rel = _rows_rel_l2(ours.m.flatten(0, 1), theirs.attributions.flatten(0, 1))
    d_delta = (ours.deltas.double() - theirs.deltas.double()).abs() / _dy(ours).clamp_min(1)
    _report(
        pytestconfig,
        f"C3/C5 config {config} {_name(dtype)} {kind}: raw per-pair rel-L2 median {float(rel.median()):.3g} max "
        f"{float(rel.max()):.3g} (tol "
        + (f"max {C3_FP64_MAX:g}" if dtype == F64 else f"median {C3_FP32_MEDIAN:g}, max {C3_FP32_MAX:g}")
        + f"; bitwise {'yes' if torch.equal(ours.m, theirs.attributions) else 'no'}); "
        f"|d delta|/max(1,|dy|) max {float(d_delta.max()):.3g}"
        + (f" (tol {C5_FP64:g})" if dtype == F64 else " (reported only)")
        + f"; deltas max ours {float(ours.deltas.max()):.3g}, tangermeme {float(theirs.deltas.max()):.3g}",
    )
    if dtype == F64:
        assert float(rel.max()) <= C3_FP64_MAX
        assert float(d_delta.max()) <= C5_FP64
    else:
        assert float(rel.median()) <= C3_FP32_MEDIAN
        assert float(rel.max()) <= C3_FP32_MAX


@pytest.mark.parametrize("dtype, kind", PARITY_CASES)
@pytest.mark.parametrize("config", CONFIG_NAMES)
def test_c4_hypothetical_attributions_match_tangermeme(config, dtype, kind, pytestconfig):
    """Engine.run against deep_lift_shap(hypothetical=True), per sequence, over the full length and the window.

    The window run must also be the full-length run sliced, bit for bit, and both must carry Engine.raw's deltas and
    outputs: the three share every step.
    """
    L = dls_tiny.CONFIGS[config].length
    start, end = _window(config)
    full, window = _ours_run(config, dtype, kind, 0, L), _ours_run(config, dtype, kind, start, end)
    theirs = _theirs(config, dtype, kind, False).attributions
    assert full.attr.shape == theirs.shape == (N_SEQUENCES, 4, L)
    assert window.attr.shape == (N_SEQUENCES, 4, end - start)
    assert full.attr.dtype == window.attr.dtype == dtype
    assert torch.equal(window.attr, full.attr[..., start:end])
    raw = _ours_raw(config, dtype, kind)
    for result in (full, window):
        for name in ("deltas", "y_x", "y_r"):
            assert torch.equal(getattr(result, name), getattr(raw, name)), name

    rel_full = _rows_rel_l2(full.attr, theirs)
    rel_window = _rows_rel_l2(window.attr, theirs[..., start:end])
    worst = max(float(rel_full.max()), float(rel_window.max()))
    _report(
        pytestconfig,
        f"C4 config {config} {_name(dtype)} {kind}: hyp per-sequence rel-L2 max, full length "
        f"{float(rel_full.max()):.3g}, window [{start}, {end}) {float(rel_window.max()):.3g} (tol {C4_TOL[dtype]:g}); "
        f"bitwise {'yes' if torch.equal(full.attr, theirs) else 'no'}",
    )
    assert worst <= C4_TOL[dtype]


@pytest.mark.parametrize("kind", ["shuffled", "random", "mutated"])
@pytest.mark.parametrize("config", CONFIG_NAMES)
def test_c6_conservation_with_two_pass_statistics(config, kind, pytestconfig):
    """With two_pass the forward normalizes with the rule's own statistics, so the rules sum exactly to y_x - y_r."""
    ours = _ours_raw(config, F64, kind, "two_pass")
    ratio = ours.deltas / _dy(ours).clamp_min(1)
    mirror = _ours_raw(config, F64, kind, "cpu_mirror")
    mirror_ratio = mirror.deltas / _dy(mirror).clamp_min(1)
    _report(
        pytestconfig,
        f"C6 config {config} {kind}: two_pass max |delta|/max(1,|dy|) {float(ratio.max()):.3g} (tol {C6_FP64:g}); "
        f"cpu_mirror, for contrast: {float(mirror_ratio.max()):.3g} (not bounded)",
    )
    assert float(ratio.max()) <= C6_FP64


@pytest.mark.parametrize("config", CONFIG_NAMES)
def test_c7_identical_references_give_the_gradient(config, pytestconfig):
    """r = x: every rule falls back to the ordinary gradient, so the multipliers are autograd's gradient at x."""
    grad = _gradient(config).repeat_interleave(K, dim=0)
    lines = []
    for stats in ("cpu_mirror", "two_pass"):
        ours = _ours_raw(config, F64, "identical", stats)
        rel = _rows_rel_l2(ours.m.flatten(0, 1), grad)
        lines.append(f"{stats} max {float(rel.max()):.3g}, max delta {float(ours.deltas.max()):.3g}")
        assert float(rel.max()) <= C7_TOL, stats
    _report(pytestconfig, f"C7 config {config}: per-pair rel-L2 vs the gradient (tol {C7_TOL:g}): " + "; ".join(lines))


@pytest.mark.parametrize("config", CONFIG_NAMES)
def test_c8_near_identical_references_exercise_the_gradient_fallback(config, pytestconfig):
    """3 mutated positions: the GELU inputs of a sequence and its reference often differ by less than 1e-6.

    The count is taken from a forward over the same joint rows: where |z_x - z_r| < 1e-6, strictly, tangermeme's
    rescale rule and ours take the gradient instead of the secant. C8 asks for one such element. Every GELU must have
    one here, the blocks' as well as the stem's: away from the mutations the stem's inputs are equal, while a block's
    differ through the per-row normalization, by amounts that only sometimes fall below 1e-6. C3's float64 bound must
    still hold.
    """
    cfg = dls_tiny.CONFIGS[config]
    X, R = _case(config, "mutated")
    w = cheri_dls.CheriWeights.from_model(dls_tiny.tiny_model(config, dtype=F64))
    X1h = torch.cat([X, R.flatten(0, 1)]).to(F64)
    with torch.no_grad():
        fwd = cheri_dls.model_forward(X1h, w, forward_stats="cpu_mirror", group=cfg.group)
    sites = [("stem", fwd.z0)] + [(f"block {i}", saved.u) for i, saved in enumerate(fwd.saved)]
    counts = {}
    for name, z in sites:
        zx, zr = cheri_dls.split_rows(z, N_SEQUENCES, K)
        counts[name] = int(((zx - zr).abs() < cheri_dls.RESCALE_EPS).sum())

    theirs = _theirs(config, F64, "mutated", True).attributions
    rel = _rows_rel_l2(_ours_raw(config, F64, "mutated").m.flatten(0, 1), theirs.flatten(0, 1))
    _report(
        pytestconfig,
        f"C8 config {config}: GELU elements with |dz| < 1e-6 over {N_SEQUENCES * K} pairs: "
        + ", ".join(f"{name} {n}" for name, n in counts.items())
        + f"; raw per-pair rel-L2 max {float(rel.max()):.3g} (tol {C3_FP64_MAX:g})",
    )
    assert min(counts.values()) >= 1
    assert float(rel.max()) <= C3_FP64_MAX


@pytest.mark.parametrize("kind", ["shuffled", "mutated"])
@pytest.mark.parametrize("stats", ["cpu_mirror", "two_pass"])
@pytest.mark.parametrize("config", CONFIG_NAMES)
def test_c9_results_do_not_depend_on_the_step_size(config, stats, kind, pytestconfig):
    """Steps of 3 and 8 peaks against steps of 1, for 13 peaks: the last step is padded with copies of the last peak.

    Rows are independent in every op, so only reassociation can differ. Multipliers are compared per pair and the
    attributions per sequence (rel-L2); deltas relative to max(1, |dy|), and outputs relative to max(1, |y|).
    """
    start, end = _window(config)
    base_raw = _ours_raw(config, F64, kind, stats, 1)
    base_run = _ours_run(config, F64, kind, start, end, stats, 1)
    assert (base_raw.debug["steps"], base_raw.debug["padded_peaks"]) == (N_SEQUENCES, 0)
    lines = []
    for step in (3, 8):
        raw = _ours_raw(config, F64, kind, stats, step)
        run = _ours_run(config, F64, kind, start, end, stats, step)
        n_steps = -(-N_SEQUENCES // step)
        assert (raw.debug["seqs_per_step"], raw.debug["steps"], raw.debug["padded_peaks"]) == (
            step,
            n_steps,
            n_steps * step - N_SEQUENCES,
        )
        assert (run.meta["seqs_per_step"], run.meta["steps"]) == (step, n_steps)
        y_ref = torch.cat([base_raw.y_x[:, None], base_raw.y_r], dim=1).double()
        y = torch.cat([raw.y_x[:, None], raw.y_r], dim=1).double()
        errs = {
            "m": float(_rows_rel_l2(raw.m.flatten(0, 1), base_raw.m.flatten(0, 1)).max()),
            "attr": float(_rows_rel_l2(run.attr, base_run.attr).max()),
            "deltas": float(((raw.deltas - base_raw.deltas).abs() / _dy(base_raw).clamp_min(1)).max()),
            "y": float(((y - y_ref).abs() / y_ref.abs().clamp_min(1)).max()),
        }
        bitwise = torch.equal(raw.m, base_raw.m) and torch.equal(run.attr, base_run.attr)
        lines.append(
            f"S={step}: "
            + ", ".join(f"{k} {v:.3g}" for k, v in errs.items())
            + f" (bitwise {'yes' if bitwise else 'no'})"
        )
        assert max(errs.values()) <= C9_TOL, step
    _report(pytestconfig, f"C9 config {config} {stats} {kind} vs S=1 (tol {C9_TOL:g}): " + "; ".join(lines))


@pytest.mark.parametrize("config", CONFIG_NAMES)
def test_c10_deeplift_multipliers_are_not_the_gradient(config, pytestconfig):
    """Shuffled references: every pair's multipliers differ from the gradient at its sequence by >= 10% rel-L2."""
    rel = _rows_rel_l2(_ours_raw(config, F64, "shuffled").m.flatten(0, 1), _gradient(config).repeat_interleave(K, 0))
    _report(
        pytestconfig,
        f"C10 config {config}: per-pair rel-L2 DeepLIFT vs gradient, min {float(rel.min()):.3g} "
        f"(bound >= {C10_MIN:g}), median {float(rel.median()):.3g}, max {float(rel.max()):.3g}",
    )
    assert float(rel.min()) >= C10_MIN


@pytest.mark.parametrize("workers", [0, 2])
def test_produced_references_are_the_injected_shuffles(workers, pytestconfig):
    """references=None: dls_refs.RefProducer draws them, in this process (0 workers) or in a spawn pool.

    dls_tiny.shuffled_references is deep_lift_shap's own draw (test_shuffled_references_are_deep_lift_shaps_own), so
    bitwise equality with it covers the seeds, the token round trip and the order of the chunks. Config B, with steps
    of 4 peaks: 13 peaks make 4 steps, the last padded.
    """
    rs, k = 1234, 3
    X = dls_tiny.sequences("B")
    R = dls_tiny.shuffled_references(X, k, random_state=rs)
    engine = _engine("B", torch.float32, n_shuffles=k, random_state=rs, ref_workers=workers, seqs_per_step=4)
    produced, injected = engine.raw(X), engine.raw(X, R)
    for name in ("m", "y_x", "y_r", "deltas"):
        assert torch.equal(getattr(produced, name), getattr(injected, name)), name
    start, end = _window("B")
    produced_run, injected_run = engine.run(X, start, end), engine.run(X, start, end, references=R)
    assert torch.equal(produced_run.attr, injected_run.attr) and torch.equal(produced_run.deltas, injected_run.deltas)
    assert (produced_run.meta["references"], injected_run.meta["references"]) == ("produced", "injected")
    assert produced_run.meta["random_state"] == rs and produced_run.meta["n_shuffles"] == k
    _report(
        pytestconfig,
        f"RefProducer, {workers} worker(s): {N_SEQUENCES * k} pairs bitwise equal to the injected shuffles "
        f"(raw and run); reference stats {produced_run.meta['reference_stats']}",
    )


def test_engine_arguments():
    model = dls_tiny.tiny_model("B")
    X = dls_tiny.sequences("B")[:3]
    L = X.shape[-1]
    R = dls_tiny.shuffled_references(X, 2)

    with pytest.raises(NotImplementedError, match="profile"):
        cheri_dls.Engine(model, output="profile", group=1)
    with pytest.raises(ValueError, match="output"):
        cheri_dls.Engine(model, output="logits", group=1)
    with pytest.raises(ValueError, match="attributes one output"):
        cheri_dls.Engine(model, group=None)  # attribute.py:43-47
    with pytest.raises(ValueError, match="contract"):
        cheri_dls.Engine(ControlWrapper(model), group=1)
    for bad in (2, True):
        with pytest.raises(ValueError, match="group must be"):
            cheri_dls.Engine(model, group=bad)  # wrappers.py:67-85
    for bad in (0, -1, "big", 2.5, True):
        with pytest.raises(ValueError, match="seqs_per_step"):
            cheri_dls.Engine(model, group=1, seqs_per_step=bad)
    for name in ("n_shuffles", "head_rows"):
        for bad in (0, 1.5, True):
            with pytest.raises(ValueError, match=name):
                cheri_dls.Engine(model, group=1, **{name: bad})
    with pytest.raises(ValueError, match="forward_stats"):
        cheri_dls.Engine(model, group=1, forward_stats="three_pass")
    with pytest.raises(ValueError, match="random_state"):
        cheri_dls.Engine(model, group=1, random_state=1.5)
    with pytest.raises(ValueError, match="warning_threshold"):
        cheri_dls.Engine(model, group=1, warning_threshold="high")
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError, match="ref_workers"):
            cheri_dls.Engine(model, group=1, ref_workers=bad)

    engine = cheri_dls.Engine(model, group=1, n_shuffles=2, ref_workers=0)
    assert (engine.forward_stats, engine.random_state, engine.head_rows) == ("cpu_mirror", 0, 128)
    with_n = X.clone()
    with_n[1, :, 7] = 0
    bad_inputs = {
        "not one-hot": (X * 2, None, "one-hot"),
        "an N in X": (with_n, R, "unknown"),  # with references=None tangermeme's shuffle would refuse it as well
        "an N in X, references drawn": (with_n, None, "unknown"),
        "X of 3 channels": (X[:, :3], None, "shape"),
        "no sequences": (X[:0], None, "at least one"),
        "references for other sequences": (X, R[:2], "references"),
        "references of another length": (X, R[..., :-1], "references"),
        "no references": (X, R[:, :0], "references"),
        "references not one-hot": (X, R * 2, "one-hot"),
        "an input too short for the trimming": (X[..., : 2 * model.trimming], None, "trimming"),
    }
    for x, r, match in bad_inputs.values():
        with pytest.raises(ValueError, match=match):
            engine.raw(x, r)
        with pytest.raises(ValueError, match=match):
            engine.run(x, 0, x.shape[-1] if x.dim() == 3 else 1, references=r)
    for start, end in ((10, 10), (-1, 20), (0, L + 1), (50, 40)):
        with pytest.raises(ValueError, match="window"):
            engine.run(X, start, end, references=R)

    # tangermeme takes K from injected references, whatever n_shuffles says (deep_lift_shap.py:681-684), and allows
    # their N (all-zero) columns.
    R_n = R.clone()
    R_n[0, 1, :, 5] = 0
    raw = cheri_dls.Engine(model, group=1, n_shuffles=20).raw(X, R_n)
    assert raw.m.shape == (3, 2, 4, L) and bool(torch.isfinite(raw.m).all())

    # random_state None draws a base seed once, records it, and draws the references from it.
    drawn = cheri_dls.Engine(model, group=1, n_shuffles=2, random_state=None, ref_workers=0)
    rs = drawn.random_state
    assert isinstance(rs, int) and 0 <= rs < 9_999_999
    result = drawn.run(X, 0, L)
    assert result.meta["random_state"] == rs
    expected = drawn.run(X, 0, L, references=dls_tiny.shuffled_references(X, 2, random_state=rs))
    assert torch.equal(result.attr, expected.attr)


def test_engine_takes_numpy_inputs_and_group_none():
    """numpy one-hots work like tensors; group None on a one-group model attributes that group (target 0)."""
    model = dls_tiny.tiny_model("A")
    X = dls_tiny.sequences("A")[:3]
    R = dls_tiny.shuffled_references(X, 2)
    expected = cheri_dls.Engine(model, group=0).raw(X, R)
    for group, x, r in ((0, X.numpy(), R.numpy()), (None, X, R)):
        got = cheri_dls.Engine(model, group=group).raw(x, r)
        for name in ("m", "y_x", "y_r", "deltas"):
            assert torch.equal(getattr(got, name), getattr(expected, name)), (group, name)


def test_run_warns_once_for_high_deltas():
    """One summary RuntimeWarning per run when any delta exceeds the threshold, none for a threshold of None (§9).

    A delta exceeds the threshold only if it is strictly greater, as in tangermeme (deep_lift_shap.py:744). So a
    threshold equal to the run's largest delta gives no warning.
    """
    model = dls_tiny.tiny_model("A")
    X = dls_tiny.sequences("A")[:4]
    R = dls_tiny.shuffled_references(X, 3)
    largest = float(cheri_dls.Engine(model, warning_threshold=None).run(X, 0, X.shape[-1], references=R).deltas.max())
    for threshold, warns in ((0.0, True), (None, False), (1.0, False), (largest, False)):
        engine = cheri_dls.Engine(model, warning_threshold=threshold, seqs_per_step=3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = engine.run(X, 0, X.shape[-1], references=R)
        messages = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
        summary = result.meta["deltas"]
        assert summary["pairs"] == 12 and summary["threshold"] == threshold
        assert summary["max"] == float(result.deltas.max()) == largest
        if warns:
            assert len(messages) == 1, messages
            assert re.fullmatch(
                r"Convergence deltas too high: \d+ of 12 pairs > 0 \(max \S+, p50/p99/p99\.9 \S+/\S+/\S+\)", messages[0]
            )
            assert summary["above"] == int((result.deltas > 0).sum())
        else:
            assert messages == []
            assert summary["above"] == (None if threshold is None else 0)
