#!/usr/bin/env python3
"""ChromBPNet's predicted log counts for variant, null and peak windows, on both strands: the bias-free model, the
bias model, and the full model derived from the two.

Models, from one fold's models/ directory:
  chrombpnet_nobias.h5   --nobias, written as cbp_nobias
  bias_model_scaled.h5   --bias, written as cbp_bias: the scaled bias model chrombpnet.h5 was trained around and
                         froze (every fold's logs/chrombpnet_model_params.tsv names it as bias_model_path; the
                         unscaled 03.0 bias model of chrombpnet.args.json is not this one)
  chrombpnet.h5          --full, optional: the full model itself, written as cbp_full_direct, only to check the
                         derivation below (V3)
Loading: the fork's one entry point for every ChromBPNet model file, chrombpnet.training.utils.model_io.load_model
(fork chrombpnet/training/utils/model_io.py:122-134: Keras 3 .h5 files go through keras.saving.load_model with
compile=False and the fork's custom objects). It is what igvf_tf_chrombpnet's 04.3 predictions use, through
chrombpnet.helpers.hyperparameters.param_utils.load_model_wrapper (param_utils.py:69-73 -> model_io.py:137-142;
igvf_tf_chrombpnet src/predict_and_avg.py:50, 132), and what the fork's own predict uses (training/predict.py:98).
`chrombpnet` is imported before anything imports keras: that selects the Keras JAX backend and turns JAX's GPU
preallocation off, both by os.environ.setdefault (chrombpnet/__init__.py:7, 10), so the environment can override
them.

Prediction: one jax.jit function runs every requested part on a batch and on its reverse complement through Keras's
stateless_call, as the fork's DeepSHAP explainer runs its models (chrombpnet/evaluation/interpret/explainer.py:245,
307-308). Only the uint8 codes go to the GPU; the one-hot (B, 2114, 4), code 4 as an all-zero row, and the reverse
complement x[:, ::-1, ::-1] (VS src/utils/helpers.py:138-139) are built there, and only the count heads come back.
JAX compiles once per input shape, so a short last batch is padded to the full batch size and its padded rows
dropped, as the fork's predict_on_batch_wrapper does (chrombpnet/training/predict.py:68-78).

The full model, derived: chrombpnet.h5 runs the bias-free and the bias model on the same input, adds their profile
logits and takes a logsumexp over their log counts (chrombpnet/training/models/chrombpnet_with_bias_model.py:114-131;
LogSumExp is ops.logsumexp(axis=-1, keepdims=True), chrombpnet/training/utils/layers.py:15-16), with every bias layer
frozen (:14-21). So, strand by strand,
    logcounts_full = log(exp(logcounts_nobias) + exp(logcounts_bias)),
which predict_common.derive_full computes in float64; the strand average then follows, as for any model. Run with
--full, chrombpnet.h5 itself goes through Keras's predict_on_batch on the host-built one-hot, the fork's own prediction
path (training/predict.py:68-78), so the check compares two routes as well as the formula.

Precision (fork chrombpnet.training.runtime.configure_precision, runtime.py:12-30), set before the first trace:
  default   JAX's default: TF32 for float32 matmuls and convolutions on H100s, as TensorFlow did for chrombpnet 1.x
  highest   full float32 (jax_default_matmul_precision='highest'), the setting of the fork's parity checks
bf16 inference is not offered: these models were saved in float32, and running them in bf16 would need per-layer
dtype policies that nobody has checked for prediction.

Inputs and outputs are predict_common's (its docstring has the formats):
  --windows [NAME=]<dir>/<NAME>.npz ...   build_windows.py's arrays (a1, a2 when present, ids)
  <out-dir>/cbp_nobias.<NAME>.npz, cbp_bias.<NAME>.npz, cbp_full.<NAME>.npz (derived, when both parts run),
  cbp_full_direct.<NAME>.npz (with --full); a .json beside each
  --probe                                 instead: forward passes per second on random windows, to --probe-json

    python workflows/variant_scoring/predict_chrombpnet.py \\
        --nobias <fold>/models/chrombpnet_nobias.h5 --bias <fold>/models/bias_model_scaled.h5 \\
        --windows <root>/windows/variants.npz --windows <root>/windows/null.npz \\
        --windows <root>/windows/peaks.<dataset>.npz --out-dir <root>/preds/<dataset>/fold_0 [--batch 512]
    python workflows/variant_scoring/predict_chrombpnet.py --nobias ... --components nobias --probe \\
        --probe-json probe.json [--probe-batches 64,256,512,1024]

Runs in the chrombpnet fork's pixi `cuda13` environment, on a GPU (--device cpu for tests). When several processes
share a GPU, cap each with XLA_CLIENT_MEM_FRACTION=<f> (jaxlib reads it, and the deprecated
XLA_PYTHON_CLIENT_MEM_FRACTION, whether or not memory is preallocated; setting both is an error) and keep
XLA_PYTHON_CLIENT_PREALLOCATE=false.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import predict_common as pc  # noqa: E402

logger = logging.getLogger("predict_chrombpnet")

PRECISIONS = ("default", "highest")
COMPONENTS = ("nobias", "bias")


def import_backend(device: str):
    """Import the fork (Keras JAX backend), check the device, and return (runtime, model_io, backend name)."""
    import chrombpnet  # noqa: F401  (before keras: KERAS_BACKEND=jax, no GPU preallocation, __init__.py:7, 10)
    from chrombpnet.training import runtime
    from chrombpnet.training.utils import model_io

    model_io.require_jax_backend()
    backend = runtime.assert_gpu_if_requested(device, initialize=True)
    return runtime, model_io, backend


def load_model(model_io, path: Path, window: int = pc.WINDOW):
    """model_io.load_model(path, compile=False), checked to take (window, 4) and to return [profile, (1,) counts]."""
    model = model_io.load_model(str(path), compile=False)
    shape = tuple(model.input_shape)
    if shape[1:] != (window, 4):
        raise ValueError(f"{path}: input shape {shape}, expected (None, {window}, 4)")
    outputs = list(model.outputs)
    if len(outputs) != 2 or tuple(outputs[1].shape)[1:] != (1,):
        raise ValueError(f"{path}: expected outputs [profile logits, log counts (1,)], got "
                         f"{[tuple(o.shape) for o in outputs]}")  # fmt: skip
    return model


def device_one_hot(tokens):
    """(B, L) uint8 codes -> (B, L, 4) float32 one-hot, as a jax array; code 4 matches no channel (all zero).
    The same as predict_common.one_hot(tokens, 'NLC'), computed where the tokens are."""
    import jax.numpy as jnp

    tokens = jnp.asarray(tokens)
    return (tokens[..., None] == jnp.arange(4, dtype=tokens.dtype)).astype(jnp.float32)


class CountsFunction:
    """The count heads of several Keras models on uint8 code rows, both strands, in one jitted call per batch."""

    def __init__(self, models: dict):
        import jax
        import jax.numpy as jnp

        self.names = list(models)
        self._models = list(models.values())
        # Copies, not the variables' own buffers: Keras's JAX predict donates those (keras/src/backend/jax/trainer.py,
        # make_predict_function, donate_argnums=0), so a Keras predict on the same model would free arrays held here.
        self.states = [([jnp.array(v.value) for v in m.trainable_variables],
                        [jnp.array(v.value) for v in m.non_trainable_variables]) for m in self._models]  # fmt: skip
        keras_models = self._models

        def run(states, tokens):
            x = device_one_hot(tokens)
            views = (x, x[:, ::-1, ::-1])
            out = []
            for model, (tv, ntv) in zip(keras_models, states):
                for view in views:
                    outputs, _ = model.stateless_call(tv, ntv, view, training=False)
                    out.append(outputs[1][:, 0].astype(jnp.float32))
            return jnp.stack(out)

        self._run = jax.jit(run)

    def predict(self, tokens: np.ndarray, batch: int) -> dict[str, dict[str, np.ndarray]]:
        n = len(tokens)
        if n == 0:
            return {name: {s: np.zeros(0, np.float32) for s in pc.STRANDS} for name in self.names}
        pending = []
        for start, stop in pc.batch_slices(n, batch):
            chunk = pc.pad_rows(np.ascontiguousarray(tokens[start:stop], dtype=np.uint8), batch)
            pending.append((stop - start, self._run(self.states, chunk)))  # dispatched; collected below
        stacked = np.concatenate([np.asarray(r)[:, :k] for k, r in pending], axis=1)
        return {name: {"fwd": stacked[2 * i], "rev": stacked[2 * i + 1]} for i, name in enumerate(self.names)}


def predict_keras(model, tokens: np.ndarray, batch: int) -> dict[str, np.ndarray]:
    """A model's count head through Keras's predict_on_batch on host-built one-hot batches, padded like
    chrombpnet/training/predict.py:68-78: the fork's own prediction route."""
    n = len(tokens)
    out = {s: np.zeros(n, np.float32) for s in pc.STRANDS}
    for start, stop in pc.batch_slices(n, batch):
        k = stop - start
        x = pc.one_hot(pc.pad_rows(tokens[start:stop], batch), "NLC")
        for strand, view in (("fwd", x), ("rev", x[:, ::-1, ::-1])):
            preds = model.predict_on_batch(np.ascontiguousarray(view))
            out[strand][start:stop] = np.asarray(preds[1])[:k, 0]
    return out


def peak_memory_gb() -> float | None:
    import jax

    try:
        stats = jax.devices()[0].memory_stats() or {}
    except Exception:  # memory_stats is backend-dependent
        return None
    peak = stats.get("peak_bytes_in_use")
    return round(peak / 2**30, 3) if peak is not None else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nobias", type=Path, help="chrombpnet_nobias.h5")
    ap.add_argument("--bias", type=Path, help="bias_model_scaled.h5, the scaled bias model chrombpnet.h5 freezes")
    ap.add_argument("--full", type=Path, help="chrombpnet.h5: also run it, as <prefix>_full_direct (the V3 check)")
    ap.add_argument("--components", default="nobias,bias",
                    help="the parts to run: nobias, bias or nobias,bias (default; the full model is then derived)")  # fmt: skip
    ap.add_argument("--prefix", default="cbp", help="output model-name prefix (default cbp)")
    ap.add_argument("--precision", choices=PRECISIONS, default="default", help="see above (default: JAX's)")
    ap.add_argument("--device", choices=("gpu", "cpu", "auto"), default="gpu", help="JAX device (default gpu)")
    pc.add_common_args(ap, default_batch=512, default_probe_batches="64,256,512,1024")
    args = ap.parse_args(argv)
    pc.setup_logging(args.verbose)
    specs = pc.check_common_args(ap, args)
    components = [c for c in args.components.split(",") if c]
    if not components or any(c not in COMPONENTS for c in components) or len(set(components)) != len(components):
        ap.error(f"--components takes nobias, bias or nobias,bias, got {args.components!r}")
    paths = {"nobias": args.nobias, "bias": args.bias}
    for c in components:
        if paths[c] is None:
            ap.error(f"--components {args.components} needs --{c}")
    if args.probe and args.full:
        ap.error("--probe times the parts only; leave out --full")

    runtime, model_io, backend = import_backend(args.device)
    runtime.configure_precision(args.precision)  # before the first trace: JAX bakes the precision into it
    names = {c: f"{args.prefix}_{c}" for c in components}
    models = {names[c]: load_model(model_io, paths[c]) for c in components}
    counts = CountsFunction(models)
    full_name = f"{args.prefix}_full_direct"
    full_model = load_model(model_io, args.full) if args.full else None
    info = {"backend": "keras-jax", **runtime.runtime_info(), "precision": args.precision,
            "jax_backend_checked": backend}  # fmt: skip
    logger.info("%s on %s, precision %s, batch %d", " + ".join(models), info.get("jax_devices"), args.precision,
                args.batch)  # fmt: skip

    def predict(tokens: np.ndarray, batch: int) -> dict[str, dict[str, np.ndarray]]:
        out = counts.predict(tokens, batch)
        if full_model is not None:
            out[full_name] = predict_keras(full_model, tokens, batch)
        return out

    if args.probe:

        def save(results: list[dict], complete: bool = False) -> None:
            pc.write_json(args.probe_json, {"model": "+".join(models), "models": list(models),
                                            "model_files": {n: pc.file_info(paths[c]) for c, n in names.items()},
                                            **info, "results": results, "complete": complete,
                                            "peak_gpu_memory_gb": peak_memory_gb(), **pc.run_info()})  # fmt: skip

        results = pc.probe(predict, args.probe_batches, args.probe_seconds, passes_per_row=2 * len(models),
                           warmup=args.probe_warmup, rows_per_call=args.probe_rows, barrier_dir=args.barrier,
                           barrier_n=args.barrier_n, on_result=save)  # fmt: skip
        save(results, complete=True)
        return 0 if any("error" not in r for r in results) else 1

    window_sets = [pc.load_windows(path, name) for name, path in specs]
    model_names = list(models) + ([full_name] if full_model is not None else [])
    model_files = {names[c]: pc.file_info(paths[c]) for c in components}
    if full_model is not None:
        model_files[full_name] = pc.file_info(args.full)
    derived = {f"{args.prefix}_full": (names["nobias"], names["bias"])} if len(components) == 2 else {}
    meta = {"script": "workflows/variant_scoring/predict_chrombpnet.py", **info,
            "loader": "chrombpnet.training.utils.model_io.load_model(path, compile=False)",
            "prediction": "jax.jit over keras stateless_call of each part" + (
                f"; {full_name}: keras predict_on_batch" if full_model is not None else ""),
            "model_files": model_files}  # fmt: skip
    pc.run_window_sets(window_sets, predict, model_names, args.out_dir, args.batch, sample=args.sample,
                       sample_seed=args.sample_seed, permute_seed=args.permute_seed, overwrite=args.overwrite,
                       meta=meta, derived=derived)  # fmt: skip
    logger.info("peak GPU memory %s GB", peak_memory_gb())
    return 0


if __name__ == "__main__":
    sys.exit(main())
