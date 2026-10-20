"""Backend checks of the variant-scoring predictors, with tiny random models, on the CPU.

tests/test_variant_predict.py runs this file in the interpreter of each backend's environment, because the test
environment has neither torch nor keras and the backend environments have no pytest:

    <python of the cherimoya env>        tests/variant_predict_backends.py cherimoya  <empty dir>
    <python of the fork's cuda13 env>    tests/variant_predict_backends.py chrombpnet <empty dir>

It prints one JSON line ({"backend", "ok", "checks", versions}) and exits 0 when every check passes; a failed check
raises, and the traceback goes to stderr with exit status 1.

Cherimoya: a 2-layer, 16-filter Cherimoya built with Cherimoya(..., compile=False, random_state=0), its weights redrawn
larger than the default init (which leaves the model almost constant), saved and reloaded through
predict_cherimoya.load_model; then the same with one control track (n_control_tracks=1), which gets a zero control.
ChromBPNet: the fork's own architecture builders and save path, as its training writes
the three files (bpnet_model / chrombpnet_with_bias_model.getModelGivenModelOptionsAndWeightInits, model.save,
save_model_without_bias; fork tests/test_models.py builds them the same way), with 4- and 8-filter models.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
import types
from pathlib import Path

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflows" / "variant_scoring"))

import predict_common as pc  # noqa: E402

PASSED: list[str] = []


def passed(name: str) -> None:
    PASSED.append(name)


def close(a, b, atol, what: str, rtol: float = 1e-6) -> None:
    """|a - b| <= atol + rtol * |b| everywhere (rtol covers float32 rounding of larger log counts)."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    assert a.shape == b.shape, f"{what}: shapes {a.shape} and {b.shape}"
    excess = np.abs(a - b) - (atol + rtol * np.abs(b))
    worst = float(np.max(np.abs(a - b), initial=0.0))
    assert not (excess > 0).any(), f"{what}: max |difference| {worst:.3g} > {atol:g} + {rtol:g} x |value|"


def tokens(n: int, seed: int) -> np.ndarray:
    """Random code rows with an N stretch, a whole-N row and a soft cluster of one base, so rows differ."""
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 4, size=(n, pc.WINDOW), dtype=np.uint8)
    t[1, 900:1100] = pc.N_CODE
    if n > 4:
        t[4, :] = pc.N_CODE
    if n > 3:
        t[3, 1000:1114] = 2
    return t


def write_windows(directory: Path) -> tuple[Path, Path, np.ndarray, np.ndarray]:
    """A variant set (a1, a2: an SNV at 1057 in each row) and a peak set (a1 only), with builder .json sidecars."""
    directory.mkdir(parents=True, exist_ok=True)
    a1 = tokens(9, 1)
    a2 = a1.copy()
    a2[:, pc.ALLELE_INDEX] = (a1[:, pc.ALLELE_INDEX] + 1) % 4
    variants = directory / "variants.npz"
    np.savez(variants, a1=a1, a2=a2, ids=np.array([f"chrT:{i}:A:C" for i in range(len(a1))]),
             nonacgt=np.zeros(len(a1), bool))  # fmt: skip
    variants.with_suffix(".json").write_text(json.dumps({"mode": "variants", "fasta_md5": "0" * 32, "n_kept": 9}))
    peaks_a1 = tokens(5, 2)
    peaks = directory / "peaks.tiny.npz"
    np.savez(peaks, a1=peaks_a1, ids=np.array([f"peak{i}" for i in range(5)]))
    return variants, peaks, a1, a2


def check_outputs(out_dir: Path, models: list[str], sets: dict[str, list[str]]) -> dict:
    """Every <model>.<set>.npz exists with float32 arrays of the right alleles and VS's strand average."""
    loaded = {}
    for model in models:
        for name, alleles in sets.items():
            path = pc.output_path(out_dir, model, name)
            assert path.is_file() and path.with_suffix(".json").is_file(), f"missing {path} or its .json"
            p = pc.read_predictions(path)
            for allele in pc.ALLELES:
                keys = [f"{allele}_{k}" for k in ("logcounts", "fwd", "rev")]
                if allele in alleles:
                    assert all(k in p and p[k].dtype == np.float32 for k in keys), f"{path}: {allele} arrays"
                    close(p[f"{allele}_logcounts"], np.logaddexp(p[f"{allele}_fwd"].astype(np.float64),
                                                                 p[f"{allele}_rev"]) - np.log(2), 1e-6,
                          f"{path}: {allele} strand average")  # fmt: skip
                else:
                    assert not any(k in p for k in keys), f"{path}: unexpected {allele} arrays"
            meta = json.loads(path.with_suffix(".json").read_text())
            assert meta["model"] == model and meta["strand_average"] == pc.STRAND_AVERAGE, f"{path}: meta"
            loaded[(model, name)] = p
    return loaded


# ---------------------------------------------------------------------------------------------------- Cherimoya
def check_cherimoya(tmp: Path) -> dict:
    import predict_cherimoya as pch
    import torch
    from cherimoya import Cherimoya

    model = Cherimoya(n_filters=16, n_layers=2, compile=False, random_state=0)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * (0.3 if p.ndim > 1 else 0.1))
    path = tmp / "tiny.torch"
    model.save(str(path))
    loaded = pch.load_model(path, "cpu")
    assert not loaded.training and not any(p.requires_grad for p in loaded.parameters())
    passed("load_model: eval mode, no grad")

    predictor = pch.CountPredictor(loaded, "cpu")
    t = tokens(7, 0)
    x = predictor.one_hot(torch.from_numpy(t))
    assert torch.equal(x, torch.from_numpy(pc.one_hot(t, "NCL"))), "torch one-hot != predict_common.one_hot NCL"
    assert torch.equal(x[:, :, pc.ALLELE_INDEX], torch.from_numpy(pc.one_hot(t, "NLC")[:, pc.ALLELE_INDEX, :])), (
        "Cherimoya's and ChromBPNet's layouts disagree at the allele index"
    )
    rc = predictor.reverse_complement(torch.from_numpy(t)).numpy()
    assert np.array_equal(rc, pc.reverse_complement(t)), "torch reverse complement != numpy"
    passed("one-hot and reverse complement equal predict_common's")

    out = predictor.predict(t, batch=3)
    with torch.no_grad():
        ref_fwd = loaded(torch.from_numpy(pc.one_hot(t, "NCL")))[1][:, 0].numpy()
        ref_rev = loaded(torch.from_numpy(pc.one_hot(pc.reverse_complement(t), "NCL")))[1][:, 0].numpy()
        ref_flip = loaded(torch.from_numpy(pc.one_hot(t, "NCL")).flip(dims=(1, 2)))[1][:, 0].numpy()
    close(out["fwd"], ref_fwd, 1e-5, "forward strand vs model(X)[1]")
    close(out["rev"], ref_rev, 1e-5, "reverse strand vs model(revcomp X)[1]")
    close(ref_rev, ref_flip, 1e-6, "revcomp of codes vs flip(dims=(1, 2)) of the one-hot")
    assert np.std(ref_fwd) > 1e-3 and np.max(np.abs(ref_fwd - ref_rev)) > 1e-4, "tiny model too flat to test with"
    passed("count head on both strands equals the model's own forward")

    for b in (1, 2, 7, 16):
        o = predictor.predict(t, batch=b)
        close(o["fwd"], out["fwd"], 1e-5, f"batch {b} (padded last batch) vs batch 3")
        close(o["rev"], out["rev"], 1e-5, f"batch {b} reverse")
    passed("results do not depend on the batch size or the padding")

    grad = pch.CountPredictor(loaded, "cpu", grad_path=True)
    assert grad.path == "cpu" and predictor.path == "cpu"
    close(grad.predict(t, 3)["fwd"], out["fwd"], 1e-6, "grad-enabled forward vs no_grad on the CPU")
    strict = pch.CountPredictor(loaded, "cpu", precision="strict-fp32")
    assert strict.grad_path, "strict-fp32 must take the grad path"
    bf16 = pch.CountPredictor(loaded, "cpu", precision="bf16").predict(t, 3)["fwd"]
    assert bf16.shape == out["fwd"].shape and np.isfinite(bf16).all(), "bf16 autocast gave no finite outputs"
    passed("grad path, strict-fp32 and bf16 options run")  # bf16's accuracy is measured on the GPU (V4)

    variants, peaks, a1, a2 = write_windows(tmp / "windows")
    out_dir = tmp / "preds"
    argv = ["--model", str(path), "--device", "cpu", "--windows", str(variants), "--windows", str(peaks),
            "--out-dir", str(out_dir), "--batch", "4"]  # fmt: skip
    assert pch.main(argv) == 0
    res = check_outputs(out_dir, ["cherimoya"], {"variants": ["a1", "a2"], "peaks.tiny": ["a1"]})
    full = res[("cherimoya", "variants")]
    close(full["a1_fwd"], predictor.predict(a1, 5)["fwd"], 1e-5, "CLI a1 vs predictor")
    close(full["a2_rev"], predictor.predict(a2, 5)["rev"], 1e-5, "CLI a2 reverse vs predictor")
    assert np.array_equal(full["ids"], np.load(variants)["ids"]), "ids not in window order"
    meta = json.loads((out_dir / "cherimoya.variants.json").read_text())
    assert meta["windows"]["builder"]["fasta_md5"] == "0" * 32 and meta["model_file"]["bytes"] > 0
    assert meta["control"] == "none" and meta["control_tracks"] == 0 and meta["head"] == pch.HEADS["none"]
    passed("CLI writes every set with ids, VS's strand average and the builder's provenance")

    target = out_dir / "cherimoya.variants.npz"
    os.utime(target, ns=(1, 1))  # an old mtime, so the checks do not depend on the filesystem's time resolution
    assert pch.main(argv) == 0 and target.stat().st_mtime_ns == 1, "an existing output was rewritten"
    assert pch.main([*argv, "--overwrite"]) == 0 and target.stat().st_mtime_ns != 1, "--overwrite did not rewrite"
    passed("existing outputs are kept unless --overwrite")

    assert pch.main([*argv[:-4], "--out-dir", str(tmp / "perm"), "--batch", "2", "--permute-seed", "5"]) == 0
    perm = pc.read_predictions(tmp / "perm" / "cherimoya.variants.npz")
    assert np.array_equal(perm["ids"], full["ids"])
    for k in ("a1_fwd", "a1_rev", "a2_fwd", "a2_rev"):
        close(perm[k], full[k], 1e-5, f"--permute-seed {k}")
    assert pch.main([*argv[:-4], "--out-dir", str(tmp / "sample"), "--batch", "4", "--sample", "4",
                     "--sample-seed", "2"]) == 0  # fmt: skip
    sample = pc.read_predictions(tmp / "sample" / "cherimoya.variants.npz")
    rows = pc.select_rows(9, 4, 2)
    assert np.array_equal(sample["ids"], full["ids"][rows])
    close(sample["a1_logcounts"], full["a1_logcounts"][rows], 1e-5, "--sample rows")
    passed("--permute-seed and --sample keep each row's values")

    probe_json = tmp / "probe.json"
    assert pch.main(["--model", str(path), "--device", "cpu", "--probe", "--probe-json", str(probe_json),
                     "--probe-batches", "2,4", "--probe-seconds", "0.1", "--probe-rows", "8"]) == 0  # fmt: skip
    probe = json.loads(probe_json.read_text())
    assert [r["batch"] for r in probe["results"]] == [2, 4] and all(r["passes"] > 0 for r in probe["results"])
    passed("probe mode")

    # A model trained with one control track: fconv and linear take one input more (cherimoya.py:254-258), and the
    # predictor feeds it an all-zero control through cherimoya.wrappers.ControlWrapper.
    ctl = Cherimoya(n_filters=16, n_layers=2, n_control_tracks=1, compile=False, random_state=0)
    with torch.no_grad():
        for p in ctl.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * (0.3 if p.ndim > 1 else 0.1))
        ctl.linear.weight[:, 16] = 0.5  # the control's count weight, set so that a real control surely moves counts
    ctl_path = tmp / "ctl.torch"
    ctl.save(str(ctl_path))
    ctl_loaded = pch.load_model(ctl_path, "cpu")
    assert ctl_loaded.n_control_tracks == 1 and tuple(ctl_loaded.linear.weight.shape) == (1, 17)
    assert float(ctl_loaded.linear.weight[0, 16]) == 0.5, "the control's count weight did not survive save and load"
    ctl_pred = pch.CountPredictor(ctl_loaded, "cpu")
    assert ctl_pred.control == "zero" and ctl_pred.control_tracks == 1 and predictor.control == "none"
    ctl_out = ctl_pred.predict(t, batch=3)
    zeros = torch.zeros(len(t), 1, pc.WINDOW)
    with torch.no_grad():
        ref_fwd = ctl_loaded(torch.from_numpy(pc.one_hot(t, "NCL")), zeros)[1][:, 0].numpy()
        ref_rev = ctl_loaded(torch.from_numpy(pc.one_hot(pc.reverse_complement(t), "NCL")), zeros)[1][:, 0].numpy()
    close(ctl_out["fwd"], ref_fwd, 1e-5, "control model: forward strand vs model(X, zeros)[1]")
    close(ctl_out["rev"], ref_rev, 1e-5, "control model: reverse strand vs model(revcomp X, zeros)[1]")
    assert np.std(ref_fwd) > 1e-3, "tiny control model too flat to test with"
    passed("a model with a control track loads; its count head is model(X, zeros)[1] on both strands")

    # The zero control is the sequence branch alone: the same weights without the control's input channel and count
    # column, in a model without control tracks, give the same counts.
    state = {k: v.clone() for k, v in ctl_loaded.state_dict().items()}
    state["fconv.weight"] = state["fconv.weight"][:, :16]
    state["linear.weight"] = state["linear.weight"][:, :16]
    seq_only = Cherimoya(n_filters=16, n_layers=2, compile=False)
    seq_only.load_state_dict(state)
    seq_only.eval().requires_grad_(False)
    seq_out = pch.CountPredictor(seq_only, "cpu").predict(t, batch=3)
    close(ctl_out["fwd"], seq_out["fwd"], 1e-5, "zero control vs the sequence branch alone, forward")
    close(ctl_out["rev"], seq_out["rev"], 1e-5, "zero control vs the sequence branch alone, reverse")
    passed("the zero control's counts are the sequence branch's alone")

    # logFC with the zero control equals logFC with any control both alleles share (the reference's), on both
    # strands: the control enters the count head as one w * log(sum + 1) term, the same for both alleles. The reverse
    # strand gets the control reversed, as training's reverse-complement augmentation flips it (cherimoya/io.py).
    v1 = tokens(6, 3)
    v2 = v1.copy()
    centre = slice(pc.ALLELE_INDEX - 10, pc.ALLELE_INDEX + 11)  # 21 bases changed, so the alleles' counts surely differ
    v2[:, centre] = (v1[:, centre] + 1) % 4
    shared = torch.rand(len(v1), 1, pc.WINDOW, generator=g) * 3

    def counts(x: np.ndarray, control) -> dict[str, np.ndarray]:
        with torch.no_grad():
            fwd = ctl_loaded(torch.from_numpy(pc.one_hot(x, "NCL")), control)[1][:, 0].numpy()
            rev = ctl_loaded(torch.from_numpy(pc.one_hot(pc.reverse_complement(x), "NCL")), control.flip(-1))[1]
        return {"fwd": fwd, "rev": rev[:, 0].numpy()}

    real = {"a1": counts(v1, shared), "a2": counts(v2, shared)}
    zero = {"a1": counts(v1, torch.zeros_like(shared)), "a2": counts(v2, torch.zeros_like(shared))}
    close(zero["a1"]["fwd"], ctl_pred.predict(v1, 4)["fwd"], 1e-5, "control model: predictor vs model(X, zeros)[1]")
    for s in pc.STRANDS:
        close(zero["a2"][s] - zero["a1"][s], real["a2"][s] - real["a1"][s], 1e-5,
              f"zero-control logFC vs a shared real control's, {s}")  # fmt: skip
    assert np.max(np.abs(real["a1"]["fwd"] - zero["a1"]["fwd"])) > 1e-3, "the real control moved no count: no test"
    assert np.max(np.abs(zero["a2"]["fwd"] - zero["a1"]["fwd"])) > 1e-5, "the alleles' counts do not differ: no test"
    passed("zero-control logFC equals a shared real control's on both strands")

    bf16 = pch.CountPredictor(ctl_loaded, "cpu", precision="bf16").predict(t, 3)["fwd"]
    assert bf16.shape == ctl_out["fwd"].shape and np.isfinite(bf16).all(), "control model: bf16 gave no finite outputs"
    try:
        pch.CountPredictor(ctl_loaded, "cpu", control="real")
    except ValueError:
        pass
    else:
        raise AssertionError("a control other than zero was accepted")
    ctl_dir = tmp / "ctl_preds"
    assert pch.main(["--model", str(ctl_path), "--model-name", "ctl", "--control", "zero", "--device", "cpu",
                     "--windows", str(variants), "--out-dir", str(ctl_dir), "--batch", "4"]) == 0  # fmt: skip
    ctl_res = check_outputs(ctl_dir, ["ctl"], {"variants": ["a1", "a2"]})[("ctl", "variants")]
    close(ctl_res["a1_fwd"], ctl_pred.predict(a1, 5)["fwd"], 1e-5, "control model: CLI a1 vs predictor")
    ctl_meta = json.loads((ctl_dir / "ctl.variants.json").read_text())
    assert ctl_meta["control"] == "zero" and ctl_meta["control_tracks"] == 1 and ctl_meta["head"] == pch.HEADS["zero"]
    passed("control model: bf16 runs, only the zero control is accepted, the CLI records control zero")
    import cherimoya

    return {"torch": torch.__version__, "cherimoya": getattr(cherimoya, "__version__", None)}


# ---------------------------------------------------------------------------------------------------- ChromBPNet
def check_chrombpnet(tmp: Path) -> dict:
    import chrombpnet  # noqa: F401  (KERAS_BACKEND=jax before keras)
    import chrombpnet.training.models.bpnet_model as bpnet_model
    import chrombpnet.training.models.chrombpnet_with_bias_model as cwb
    import keras
    import predict_chrombpnet as pcb

    params = {"filters": "4", "n_dil_layers": "2", "counts_loss_weight": "10", "inputlen": str(pc.WINDOW),
              "outputlen": "1000"}  # fmt: skip
    bias_h5, full_h5, prefix = tmp / "bias_model_scaled.h5", tmp / "chrombpnet.h5", tmp / "chrombpnet"
    bpnet_model.getModelGivenModelOptionsAndWeightInits(types.SimpleNamespace(seed=7, learning_rate=1e-3),
                                                        params).save(str(bias_h5))  # fmt: skip
    full = cwb.getModelGivenModelOptionsAndWeightInits(types.SimpleNamespace(seed=1, learning_rate=1e-3),
                                                       {**params, "filters": "8", "bias_model_path": str(bias_h5)})  # fmt: skip
    full.save(str(full_h5))
    cwb.save_model_without_bias(full, str(prefix))
    nobias_h5 = Path(f"{prefix}_nobias.h5")

    runtime, model_io, backend = pcb.import_backend("cpu")
    assert backend == "cpu", backend
    runtime.configure_precision("default")
    nobias = pcb.load_model(model_io, nobias_h5)
    bias = pcb.load_model(model_io, bias_h5)
    fullm = pcb.load_model(model_io, full_h5)
    passed("the fork's loader reads the three files")

    t = tokens(7, 0)
    x = pc.one_hot(t, "NLC")
    assert np.array_equal(np.asarray(pcb.device_one_hot(t)), x), "jax one-hot != predict_common.one_hot NLC"
    passed("one-hot equals predict_common's")

    counts = pcb.CountsFunction({"cbp_nobias": nobias, "cbp_bias": bias})
    out = counts.predict(t, batch=3)
    xr = np.ascontiguousarray(x[:, ::-1, ::-1])
    for name, m in (("cbp_nobias", nobias), ("cbp_bias", bias)):
        close(out[name]["fwd"], m.predict(x, verbose=0)[1][:, 0], 1e-5, f"{name} forward vs keras predict")
        close(out[name]["rev"], m.predict(xr, verbose=0)[1][:, 0], 1e-5, f"{name} reverse vs keras predict")
    ref_full_fwd = fullm.predict(x, verbose=0)[1][:, 0]
    ref_full_rev = fullm.predict(xr, verbose=0)[1][:, 0]
    assert np.std(ref_full_fwd) > 1e-4 and np.max(np.abs(ref_full_fwd - ref_full_rev)) > 1e-5, "models too flat"
    passed("stateless jitted count heads equal Keras predict on both strands")

    close(pc.combine_logcounts(out["cbp_nobias"]["fwd"], out["cbp_bias"]["fwd"]), ref_full_fwd, 1e-5,
          "derived full vs chrombpnet.h5, forward")  # fmt: skip
    close(pc.combine_logcounts(out["cbp_nobias"]["rev"], out["cbp_bias"]["rev"]), ref_full_rev, 1e-5,
          "derived full vs chrombpnet.h5, reverse")  # fmt: skip
    direct = pcb.predict_keras(fullm, t, 3)
    close(direct["fwd"], ref_full_fwd, 1e-5, "predict_keras forward")
    close(direct["rev"], ref_full_rev, 1e-5, "predict_keras reverse")
    passed("log(exp(nobias) + exp(bias)) reproduces chrombpnet.h5's count head")

    for b in (1, 7, 16):
        o = counts.predict(t, batch=b)
        close(o["cbp_nobias"]["fwd"], out["cbp_nobias"]["fwd"], 1e-5, f"batch {b} vs batch 3")
        close(o["cbp_bias"]["rev"], out["cbp_bias"]["rev"], 1e-5, f"batch {b} vs batch 3, bias reverse")
    passed("results do not depend on the batch size or the padding")

    variants, peaks, a1, a2 = write_windows(tmp / "windows")
    out_dir = tmp / "preds"
    argv = ["--nobias", str(nobias_h5), "--bias", str(bias_h5), "--full", str(full_h5), "--device", "cpu",
            "--windows", str(variants), "--windows", str(peaks), "--out-dir", str(out_dir), "--batch", "4"]  # fmt: skip
    assert pcb.main(argv) == 0
    models = ["cbp_nobias", "cbp_bias", "cbp_full", "cbp_full_direct"]
    res = check_outputs(out_dir, models, {"variants": ["a1", "a2"], "peaks.tiny": ["a1"]})
    for name in ("variants", "peaks.tiny"):
        derived, direct = res[("cbp_full", name)], res[("cbp_full_direct", name)]
        for k in derived:
            if k != "ids":
                close(derived[k], direct[k], 1e-5, f"{name}: derived cbp_full {k} vs chrombpnet.h5")
    meta = json.loads((out_dir / "cbp_full.variants.json").read_text())
    assert meta["derived_from"] == ["cbp_nobias", "cbp_bias"]
    close(res[("cbp_nobias", "variants")]["a2_fwd"], counts.predict(a2, 5)["cbp_nobias"]["fwd"], 1e-5, "CLI a2")
    passed("CLI writes the parts, the derived full model and chrombpnet.h5's own predictions")

    assert pcb.main([*argv[:-4], "--out-dir", str(tmp / "perm"), "--batch", "2", "--permute-seed", "3"]) == 0
    perm = pc.read_predictions(tmp / "perm" / "cbp_nobias.variants.npz")
    base = res[("cbp_nobias", "variants")]
    assert np.array_equal(perm["ids"], base["ids"])
    for k in ("a1_fwd", "a1_rev", "a2_fwd", "a2_rev"):
        close(perm[k], base[k], 1e-5, f"--permute-seed {k}")
    target = out_dir / "cbp_bias.variants.npz"
    os.utime(target, ns=(1, 1))
    assert pcb.main(argv) == 0 and target.stat().st_mtime_ns == 1, "an existing output was rewritten"
    passed("--permute-seed keeps each row's values; existing outputs are kept")

    probe_json = tmp / "probe.json"
    assert pcb.main(["--bias", str(bias_h5), "--components", "bias", "--device", "cpu", "--probe", "--probe-json",
                     str(probe_json), "--probe-batches", "2", "--probe-seconds", "0.1", "--probe-rows", "4"]) == 0  # fmt: skip
    probe = json.loads(probe_json.read_text())
    assert probe["models"] == ["cbp_bias"] and probe["results"][0]["passes"] > 0
    passed("probe mode")
    import jax

    return {"keras": keras.__version__, "jax": jax.__version__}


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[0] not in ("cherimoya", "chrombpnet"):
        print("usage: variant_predict_backends.py {cherimoya,chrombpnet} <empty dir>", file=sys.stderr)
        return 2
    backend, tmp = argv[0], Path(argv[1])
    tmp.mkdir(parents=True, exist_ok=True)
    pc.setup_logging(False)
    try:
        versions = check_cherimoya(tmp) if backend == "cherimoya" else check_chrombpnet(tmp)
    except Exception:
        traceback.print_exc()
        print(json.dumps({"backend": backend, "ok": False, "checks": PASSED}))
        return 1
    print(json.dumps({"backend": backend, "ok": True, "checks": PASSED, **versions}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
