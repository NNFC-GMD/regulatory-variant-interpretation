#!/usr/bin/env python3
"""Cherimoya's predicted log counts for variant, null and peak windows, on both strands.

The model is one fold's checkpoint, cherimoya.fold_<k>.torch: the weights the ChromBPNet comparison and the
attribution read. It is loaded with Cherimoya's own loader, without torch.compile, and put in eval mode before the
first call: Cherimoya.load(path, device, compile=False).eval() (site-packages cherimoya/cherimoya.py:343-389), as
workflows/cherimoya/compare_chrombpnet.py loads it. eval() builds the blocks' bf16 weight cache that the inference
megakernel reads (cheri.py:1063-1086); without it every call recasts the weights.

Only the count head is used: what cherimoya.wrappers.LogCountWrapper returns, model(X)[1] (wrappers.py:168-204), on
the scale the head is trained on, log(count + 1) (losses.py:158). The model must have one signal group and no control
tracks. Inputs are one-hot (N, 4, 2114) built on the GPU from the uint8 codes, code 4 as an all-zero column. The
reverse complement is the flip of the length and channel axes, tensor.flip(dims=(1, 2)), which for codes A, C, G, T
= 0..3 is the complement and the reversal (done here on the codes, before the one-hot). Every batch has the same
shape: a short last batch is padded with all-zero rows whose outputs are dropped; Cherimoya has no cross-row op, so
padding cannot change the other rows.

Forward path (each Cheri block picks one per call, cheri.py:1105-1118, 1171-1181):
  default       under torch.no_grad() on a GPU: the fused Triton inference megakernel. For fp32 input it has two
                bf16 steps: its MLP dots take bf16 weights (cheri.py:1063-1086, 1120-1169), and each block stores its
                un-normalised conv output in bf16 before the norm reads it (_fwd_inf_forward, cheri.py:779-795, with
                _fwd_inf_run_stats, :726-730). The other ops are fp32, with the TF32 matmuls and cudnn.benchmark that
                `import cherimoya` switches on for the whole process (cherimoya.py:15-16). This is the stack default
                that every Cherimoya rate so far was measured on. The megakernel-vs-grad-path check (V4) measures the
                two bf16 steps together.
  --grad-path   the normal autograd forward (the training Triton path: fused conv + norm, PyTorch MLP), for the
                megakernel-vs-grad-path check. No parameter requires grad, so autograd keeps no graph.
Precision (--precision):
  default       as above
  strict-fp32   TF32 off for matmuls and cuDNN (set_float32_matmul_precision('highest'), allow_tf32 = False), and
                the grad path, since the megakernel's two bf16 steps do not depend on the matmul setting
                (TF32-vs-fp32 check)
  bf16          bf16 inputs under torch.autocast(bfloat16), as tangermeme.predict runs dtype=bfloat16. Only for an
                agreement check before bf16 is considered at scale; not the pilot's setting.

Inputs and outputs are predict_common's (its docstring has the formats):
  --windows [NAME=]<dir>/<NAME>.npz ...   build_windows.py's arrays (a1, a2 when present, ids)
  <out-dir>/<model-name>.<NAME>.npz       a1_logcounts, a1_fwd, a1_rev (and a2_*), ids; .json beside it
  --probe                                 instead: forward passes per second on random windows, to --probe-json

    python workflows/variant_scoring/predict_cherimoya.py --model <dir>/fold_0/cherimoya.fold_0.torch \\
        --windows <root>/windows/variants.npz --windows <root>/windows/null.npz \\
        --windows <root>/windows/peaks.<dataset>.npz --out-dir <root>/preds/<dataset>/fold_0 [--batch 1024]
    python workflows/variant_scoring/predict_cherimoya.py --model ... --probe --probe-json probe.json \\
        [--probe-batches 64,256,512,1024,2048] [--barrier <dir> --barrier-n 2]

Runs in this repository's pixi `cherimoya` environment, on a GPU (--device cpu for tests). Cap its GPU memory with
PYTORCH_CUDA_ALLOC_CONF=per_process_memory_fraction:<f> when several processes share a GPU.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import predict_common as pc  # noqa: E402

logger = logging.getLogger("predict_cherimoya")

PRECISIONS = ("default", "strict-fp32", "bf16")


def set_precision(precision: str) -> None:
    """Apply --precision to the process. Call after `import cherimoya`, which sets TF32 matmuls at import."""
    import torch

    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    if precision == "strict-fp32":
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def load_model(path: Path, device: str):
    """Cherimoya.load(path, device, compile=False).eval(), checked to be one unstranded track without controls."""
    import cherimoya  # noqa: F401  (sets TF32 matmuls and cudnn.benchmark, cherimoya.py:15-16)
    from cherimoya import Cherimoya

    model = Cherimoya.load(str(path), device=device, compile=False).eval()
    if model.n_control_tracks:
        raise ValueError(f"{path}: the model takes {model.n_control_tracks} control track(s); windows have none")
    if list(model.signal_groups) != [1]:
        raise ValueError(f"{path}: expected one unstranded track (signal_groups [1]), got {model.signal_groups}")
    model.requires_grad_(False)
    return model


class CountPredictor:
    """Cherimoya's count head on uint8 code rows, forward and reverse complement, one forward call per batch and
    strand. Outputs stay on the device until the end of a call, so the CPU queues batches ahead of the GPU."""

    def __init__(self, model, device: str, precision: str = "default", grad_path: bool = False):
        import torch
        from cherimoya.wrappers import LogCountWrapper

        self.torch = torch
        self.device = torch.device(device)
        self.head = LogCountWrapper(model)
        self.precision = precision
        self.grad_path = grad_path or precision == "strict-fp32"
        self.dtype = torch.bfloat16 if precision == "bf16" else torch.float32
        self.codes = torch.arange(4, dtype=torch.uint8, device=self.device).view(1, 4, 1)
        self.complement = torch.as_tensor(pc.COMPLEMENT, device=self.device)

    @property
    def path(self) -> str:
        """Which Cheri block forward runs: 'megakernel', 'grad' (training Triton path) or 'cpu'."""
        if self.device.type != "cuda":
            return "cpu"
        return "grad" if self.grad_path else "megakernel"

    def one_hot(self, tokens):
        """(B, L) uint8 codes -> (B, 4, L) one-hot; code 4 matches no channel, so its column is all zero."""
        return (tokens.unsqueeze(1) == self.codes).to(self.dtype)

    def reverse_complement(self, tokens):
        return self.complement[tokens.long()].flip(-1)

    def _contexts(self):
        torch = self.torch
        grad = torch.enable_grad() if self.grad_path else torch.no_grad()
        if self.precision == "bf16":
            return grad, torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
        return grad, contextlib.nullcontext()

    def forward(self, tokens):
        grad, amp = self._contexts()
        with grad, amp:
            y = self.head(self.one_hot(tokens))
        return y[:, 0].detach().float()

    def predict(self, tokens: np.ndarray, batch: int) -> dict[str, np.ndarray]:
        torch = self.torch
        n = len(tokens)
        out = {s: torch.empty(n, dtype=torch.float32, device=self.device) for s in pc.STRANDS}
        if n == 0:
            return {s: v.cpu().numpy() for s, v in out.items()}
        host = torch.from_numpy(np.ascontiguousarray(tokens, dtype=np.uint8))
        if self.device.type == "cuda":
            host = host.pin_memory()
        for start, stop in pc.batch_slices(n, batch):
            k = stop - start
            tok = host[start:stop].to(self.device, non_blocking=True)
            if k < batch:
                tok = torch.cat([tok, torch.full((batch - k, tok.shape[1]), pc.N_CODE, dtype=tok.dtype,
                                                 device=self.device)])  # fmt: skip
            out["fwd"][start:stop] = self.forward(tok)[:k]
            out["rev"][start:stop] = self.forward(self.reverse_complement(tok))[:k]
        return {s: v.cpu().numpy() for s, v in out.items()}


def backend_info(predictor: CountPredictor) -> dict:
    torch = predictor.torch
    info = {
        "backend": "torch",
        "torch": torch.__version__,
        "cherimoya": importlib.metadata.version("cherimoya"),
        "precision": predictor.precision,
        "forward_path": predictor.path,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32": {"matmul": torch.backends.cuda.matmul.allow_tf32, "cudnn": torch.backends.cudnn.allow_tf32},
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
    if predictor.device.type == "cuda":
        info["device"] = torch.cuda.get_device_name(predictor.device)
        try:
            info["triton"] = importlib.metadata.version("triton")
        except importlib.metadata.PackageNotFoundError:
            info["triton"] = None
    else:
        info["device"] = "cpu"
    return info


def peak_memory_gb(predictor: CountPredictor) -> float | None:
    torch = predictor.torch
    if predictor.device.type != "cuda":
        return None
    return round(torch.cuda.max_memory_allocated(predictor.device) / 2**30, 3)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True, help="cherimoya.fold_<k>.torch")
    ap.add_argument("--model-name", default="cherimoya", help="output file prefix (default cherimoya)")
    ap.add_argument("--precision", choices=PRECISIONS, default="default", help="see above (default: the stack's)")
    ap.add_argument("--grad-path", action="store_true", help="the autograd forward instead of the megakernel")
    ap.add_argument("--device", default="cuda", help="cuda (default) or cpu")
    pc.add_common_args(ap, default_batch=1024, default_probe_batches="64,256,512,1024,2048")
    args = ap.parse_args(argv)
    pc.setup_logging(args.verbose)
    specs = pc.check_common_args(ap, args)

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logger.error("no CUDA device is visible (CUDA_VISIBLE_DEVICES=%s); use --device cpu for a CPU run",
                     pc.run_info()["cuda_visible_devices"])  # fmt: skip
        return 1
    model = load_model(args.model, args.device)
    set_precision(args.precision)
    predictor = CountPredictor(model, args.device, args.precision, args.grad_path)
    info = backend_info(predictor)
    logger.info("%s on %s: %s path, precision %s, batch %d", args.model, info["device"], predictor.path,
                args.precision, args.batch)  # fmt: skip

    def predict(tokens: np.ndarray, batch: int) -> dict[str, dict[str, np.ndarray]]:
        return {args.model_name: predictor.predict(tokens, batch)}

    if args.probe:

        def save(results: list[dict], complete: bool = False) -> None:
            pc.write_json(args.probe_json, {"model": args.model_name, "model_file": pc.file_info(args.model), **info,
                                            "results": results, "complete": complete,
                                            "peak_gpu_memory_gb": peak_memory_gb(predictor), **pc.run_info()})  # fmt: skip

        results = pc.probe(predict, args.probe_batches, args.probe_seconds, passes_per_row=2,
                           warmup=args.probe_warmup, rows_per_call=args.probe_rows, barrier_dir=args.barrier,
                           barrier_n=args.barrier_n, on_result=save)  # fmt: skip
        save(results, complete=True)
        return 0 if any("error" not in r for r in results) else 1

    window_sets = [pc.load_windows(path, name) for name, path in specs]
    meta = {"script": "workflows/variant_scoring/predict_cherimoya.py", **info,
            "head": "cherimoya.wrappers.LogCountWrapper: model(X)[1], log(count + 1) scale",
            "model_files": {args.model_name: pc.file_info(args.model)}}  # fmt: skip
    pc.run_window_sets(window_sets, predict, [args.model_name], args.out_dir, args.batch, sample=args.sample,
                       sample_seed=args.sample_seed, permute_seed=args.permute_seed, overwrite=args.overwrite,
                       meta=meta)  # fmt: skip
    logger.info("peak GPU memory %s GB", peak_memory_gb(predictor))
    return 0


if __name__ == "__main__":
    sys.exit(main())
