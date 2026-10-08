"""cheri_dls's weights and forward pass against the Cherimoya module's own CPU forward (acceptance criterion C1).

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

Measured values are written to the terminal.
"""

from __future__ import annotations

import copy
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
    (tangermeme/deep_lift_shap.py:686-692, 739-742), with the same thread count.
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
    assert _rel(result.deltas, expected) <= 1e-6

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
