"""Shared helpers of the variant-scoring predictors: window arrays in, per-strand log counts out.

predict_cherimoya.py (the pixi `cherimoya` environment of this repository) and predict_chrombpnet.py (the chrombpnet
fork's `cuda13` environment) both import this module, so it needs numpy and the standard library only. The model code
of each predictor supplies one function, predict(tokens, batch), and everything around it is here: reading and
checking the windows, the row sample and processing order of the checks, both strands, the strand average, the full
ChromBPNet model derived from its parts, writing the outputs, and the timed probe.

VS is kundajelab/variant-scorer, main @0e1e341; "VS file:line" refers to that commit.

Windows (build_windows.py), <name>.npz
  a1, a2   uint8 [N, 2114] base codes, A=0, C=1, G=2, T=3, anything else 4; one-hot as eye(5)[:, :4], so code 4 is an
           all-zero column, as VS's one_hot.dna_to_one_hot makes it (src/utils/one_hot.py:16, 31). The allele starts
           at index 1057. a2 is absent for peak windows, which have one sequence per row.
  ids      str [N]
  Other arrays (nonacgt, src_ids, seeds, chrom, summit) are ignored. <name>.json beside it holds the builder's
  settings and the FASTA md5; the parts of it that say what was built are copied into every output's .json.

Predictions, <out-dir>/<model>.<name>.npz
  a1_fwd, a1_rev     float32 [N]: the model's log counts for the window and for its reverse complement
  a1_logcounts       float32 [N]: the two strands averaged as VS averages them, in counts, not in log counts:
                     log((exp(fwd) + exp(rev)) / 2) (VS src/utils/helpers.py:169-170, 195-197), computed in float64.
                     So (a2_logcounts - a1_logcounts) / ln 2 is VS's logfc, log2(c2 / c1) (helpers.py:225).
  a2_fwd, a2_rev, a2_logcounts
                     the same for allele 2, when the windows have a2 (variant and null windows)
  ids                str [N]: the windows' ids, row for row
  <model>.<name>.json  model file, windows, settings, timing, versions. The .npz is written last, under a temporary
                     name, so its presence means the set is done; an existing one is kept unless --overwrite.

Derived full ChromBPNet model (cbp_full): chrombpnet.h5 adds the two parts' profile logits and takes a logsumexp over
their log counts (fork chrombpnet/training/models/chrombpnet_with_bias_model.py:129-131; LogSumExp is
ops.logsumexp(axis=-1, keepdims=True), chrombpnet/training/utils/layers.py:15-16). So, strand by strand,
logcounts_full = log(exp(logcounts_nobias) + exp(logcounts_bias)): the parts' counts add. VS's strand average is a mean
of counts, so it commutes with that sum, and the strand-averaged full model is the sum of the strand-averaged parts too.

The checks of the pilot need two more knobs, applied identically by both predictors:
  --sample N, --sample-seed S   score N rows drawn without replacement (in window order) instead of all of them
  --permute-seed S              run the rows through the model in a shuffled order and put the outputs back, so that
                                every row sits in a different batch and batch position than in a plain run (V4)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import socket
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger("predict_common")

WINDOW = 2114
ALLELE_INDEX = WINDOW // 2  # 1057, where the allele starts (VS variant_generator.py:24, 46-47)
N_CODE = 4
ALLELES = ("a1", "a2")
STRANDS = ("fwd", "rev")
LN2 = math.log(2.0)
# Complement of each code: A<->T, C<->G, other stays other. With A, C, G, T = 0..3 that is 3 - code, so in one-hot
# space the reverse complement is a flip of both the length and the channel axis (VS helpers.py:138-139).
COMPLEMENT = np.array([3, 2, 1, 0, N_CODE], dtype=np.uint8)
VS_COMMIT = "kundajelab/variant-scorer@0e1e34199e63112aa618748bb79a206fc491300a"
STRAND_AVERAGE = ("log((exp(fwd) + exp(rev)) / 2): the log of the strand-averaged counts, as VS averages "
                  "exp(log counts) (src/utils/helpers.py:169-170, 195-197)")  # fmt: skip
LOGFC = "logfc = (a2_logcounts - a1_logcounts) / ln 2 = log2(c2 / c1), VS's logfc (src/utils/helpers.py:225)"
# Builder settings worth carrying into each output's .json (the rest of build_windows.py's .json stays there).
BUILDER_KEYS = ("mode", "created", "code_commit", "fasta", "fasta_md5", "list", "list_md5", "peaks", "peaks_md5",
                "n_input", "n_kept", "seed", "input_len", "allele_index")  # fmt: skip

# A predictor's model code: (tokens uint8 [n, WINDOW], batch) -> {model name: {"fwd": float32 [n], "rev": float32 [n]}}
PredictFn = Callable[[np.ndarray, int], dict[str, dict[str, np.ndarray]]]


# ---------------------------------------------------------------------------------------------------- windows
@dataclass
class WindowSet:
    """One window file: its name (the file's stem unless given), ids and allele arrays, and the builder's .json."""

    name: str
    path: Path
    ids: np.ndarray
    alleles: dict[str, np.ndarray]
    builder: dict

    def __len__(self) -> int:
        return len(self.ids)


def parse_window_spec(spec: str) -> tuple[str, Path]:
    """'NAME=path.npz' -> (NAME, path); 'dir/peaks.x.npz' -> ('peaks.x', path): the name is the file's stem."""
    name, sep, path = spec.partition("=")
    if not sep:
        path = spec
        name = Path(spec).name.removesuffix(".npz")
    if not name or not path.endswith(".npz"):
        raise ValueError(f"--windows takes [NAME=]<file>.npz, got {spec!r}")
    return name, Path(path)


def load_windows(path: Path, name: str | None = None, window: int = WINDOW) -> WindowSet:
    """Read and check a build_windows.py npz: a1 (and a2) uint8 [N, window] with codes 0-4, ids [N]."""
    path = Path(path)
    name = name or path.name.removesuffix(".npz")
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in ("a1", "ids") if k not in z.files]
        if missing:
            raise ValueError(f"{path}: no {', '.join(missing)} (has {', '.join(z.files)})")
        ids = np.asarray(z["ids"]).astype(str)
        alleles = {a: np.ascontiguousarray(z[a]) for a in ALLELES if a in z.files}
    if ids.ndim != 1:
        raise ValueError(f"{path}: ids must be 1-D, got shape {ids.shape}")
    for allele, x in alleles.items():
        if x.dtype != np.uint8 or x.ndim != 2 or x.shape != (len(ids), window):
            raise ValueError(f"{path}: {allele} must be uint8 [{len(ids)}, {window}], got {x.dtype} {list(x.shape)}")
        if x.size and int(x.max()) > N_CODE:
            raise ValueError(f"{path}: {allele} holds codes above {N_CODE} (max {int(x.max())})")
    builder: dict = {}
    sidecar = path.with_suffix(".json")
    if sidecar.is_file():
        try:
            builder = json.loads(sidecar.read_text())
        except json.JSONDecodeError as err:
            logger.warning("%s: unreadable (%s); outputs will not record the builder's settings", sidecar, err)
    return WindowSet(name=name, path=path, ids=ids, alleles=alleles, builder=builder)


def reverse_complement(tokens: np.ndarray) -> np.ndarray:
    """Reverse complement of uint8 code rows [n, L] (code 4 stays 4)."""
    return COMPLEMENT[tokens][:, ::-1]


def one_hot(tokens: np.ndarray, layout: str = "NLC", dtype=np.float32) -> np.ndarray:
    """One-hot of code rows [n, L]: 'NLC' (n, L, 4), ChromBPNet's layout, or 'NCL' (n, 4, L), Cherimoya's.
    Code 4 is an all-zero column."""
    eye = np.vstack([np.eye(4, dtype=dtype), np.zeros((1, 4), dtype=dtype)])
    x = eye[tokens]
    if layout == "NLC":
        return x
    if layout == "NCL":
        return np.ascontiguousarray(x.transpose(0, 2, 1))
    raise ValueError(f"layout must be NLC or NCL, got {layout!r}")


# ---------------------------------------------------------------------------------------------------- counts
def strand_average(fwd, rev) -> np.ndarray:
    """log((exp(fwd) + exp(rev)) / 2) in float64, returned as float32: VS's strand average, in counts."""
    return (np.logaddexp(np.asarray(fwd, np.float64), np.asarray(rev, np.float64)) - LN2).astype(np.float32)


def combine_logcounts(nobias, bias) -> np.ndarray:
    """The full ChromBPNet model's log counts from its parts': log(exp(nobias) + exp(bias)), float64 -> float32."""
    return np.logaddexp(np.asarray(nobias, np.float64), np.asarray(bias, np.float64)).astype(np.float32)


def prediction_arrays(strands: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """{'a1': {'fwd': x, 'rev': y}, ...} -> a1_logcounts, a1_fwd, a1_rev (and a2_*), all float32."""
    out: dict[str, np.ndarray] = {}
    for allele in ALLELES:
        if allele not in strands:
            continue
        fwd = np.asarray(strands[allele]["fwd"], np.float32)
        rev = np.asarray(strands[allele]["rev"], np.float32)
        if fwd.shape != rev.shape or fwd.ndim != 1:
            raise ValueError(f"{allele}: strands must be equal-length 1-D arrays, got {fwd.shape} and {rev.shape}")
        out[f"{allele}_logcounts"] = strand_average(fwd, rev)
        out[f"{allele}_fwd"] = fwd
        out[f"{allele}_rev"] = rev
    return out


def derive_full(nobias: dict[str, np.ndarray], bias: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """cbp_full from cbp_nobias and cbp_bias predictions: per strand combine_logcounts, then the strand average."""
    strands = {}
    for allele in ALLELES:
        if f"{allele}_fwd" in nobias:
            strands[allele] = {s: combine_logcounts(nobias[f"{allele}_{s}"], bias[f"{allele}_{s}"]) for s in STRANDS}
    return prediction_arrays(strands)


# ---------------------------------------------------------------------------------------------------- rows
def select_rows(n: int, sample: int | None, seed: int) -> np.ndarray:
    """All rows, or `sample` of them drawn without replacement with default_rng(seed), in window order."""
    if sample is None or sample >= n:
        return np.arange(n)
    if sample < 1:
        raise ValueError(f"--sample must be at least 1, got {sample}")
    return np.sort(np.random.default_rng(seed).choice(n, size=sample, replace=False))


def processing_order(n: int, seed: int | None) -> np.ndarray | None:
    """The order rows go through the model in: None (as they are) or a permutation from default_rng(seed)."""
    return None if seed is None else np.random.default_rng(seed).permutation(n)


def restore_order(values: np.ndarray, order: np.ndarray | None) -> np.ndarray:
    """Undo processing_order: row order[i] was processed i-th."""
    if order is None:
        return values
    out = np.empty_like(values)
    out[order] = values
    return out


def batch_slices(n: int, batch: int) -> Iterator[tuple[int, int]]:
    if batch < 1:
        raise ValueError(f"batch must be at least 1, got {batch}")
    for start in range(0, n, batch):
        yield start, min(start + batch, n)


def pad_rows(chunk: np.ndarray, batch: int, fill: int = N_CODE) -> np.ndarray:
    """A short last batch padded to `batch` rows (of code `fill`), so every call has one shape and one compiled
    kernel set; the padded rows' outputs are dropped. As the fork's predict_on_batch_wrapper pads
    (chrombpnet/training/predict.py:68-78)."""
    if len(chunk) == batch:
        return chunk
    out = np.full((batch,) + chunk.shape[1:], fill, dtype=chunk.dtype)
    out[: len(chunk)] = chunk
    return out


# ---------------------------------------------------------------------------------------------------- outputs
def output_path(out_dir: Path, model: str, name: str) -> Path:
    return Path(out_dir) / f"{model}.{name}.npz"


def write_predictions(path: Path, ids: np.ndarray, arrays: dict[str, np.ndarray], meta: dict) -> None:
    """<model>.<name>.npz (ids + arrays) and its .json; the .npz last, through a temporary name."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    partial = path.with_name(path.name + ".partial")
    with open(partial, "wb") as fh:
        np.savez(fh, ids=np.asarray(ids).astype(str), **arrays)
    os.replace(partial, path)


def read_predictions(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def file_info(path: Path) -> dict:
    path = Path(path)
    st = path.stat()
    return {"path": str(path.resolve()), "bytes": st.st_size,
            "mtime": time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime(st.st_mtime))}  # fmt: skip


def run_info() -> dict:
    """Where and when: host, time (with its UTC offset), python, CUDA_VISIBLE_DEVICES, the CPUs allowed."""
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        cpus = []
    return {"host": socket.gethostname(), "written": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "python": sys.version.split()[0], "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "n_cpus_allowed": len(cpus), "pid": os.getpid()}  # fmt: skip


def run_window_sets(window_sets: list[WindowSet], predict: PredictFn, models: list[str], out_dir: Path, batch: int,
                    *, sample: int | None = None, sample_seed: int = 0, permute_seed: int | None = None,
                    overwrite: bool = False, meta: dict | None = None,
                    derived: dict[str, tuple[str, str]] | None = None) -> list[dict]:  # fmt: skip
    """Predict every window set with `predict` and write <model>.<set>.npz for each model it returns (and for each
    derived model: name -> (nobias part, bias part)). `meta` is copied into every .json; meta['model_files'] may map a
    model name to its file. Returns one summary per set."""
    derived = derived or {}
    meta = meta or {}
    summaries = []
    for ws in window_sets:
        paths = {m: output_path(out_dir, m, ws.name) for m in [*models, *derived]}
        if not overwrite and all(p.exists() for p in paths.values()):
            logger.info("%s: every output exists (%s); skipped", ws.name, ", ".join(sorted(paths)))
            summaries.append({"set": ws.name, "skipped": True})
            continue
        rows = select_rows(len(ws), sample, sample_seed)
        order = processing_order(len(rows), permute_seed)
        strands: dict[str, dict[str, dict[str, np.ndarray]]] = {m: {} for m in models}
        t0 = time.perf_counter()
        for allele, tokens in ws.alleles.items():
            x = tokens if len(rows) == len(ws) else tokens[rows]
            if order is not None:
                x = x[order]
            pred = predict(np.ascontiguousarray(x), batch)
            missing = [m for m in models if m not in pred]
            if missing:
                raise KeyError(f"the predictor returned no {', '.join(missing)}")
            for m in models:
                strands[m][allele] = {s: restore_order(np.asarray(pred[m][s], np.float32), order) for s in STRANDS}
        seconds = time.perf_counter() - t0
        passes = 2 * len(rows) * len(ws.alleles)  # per model: both strands of every allele
        arrays = {m: prediction_arrays(strands[m]) for m in models}
        for name, (nobias, bias) in derived.items():
            arrays[name] = derive_full(arrays[nobias], arrays[bias])
        ids = ws.ids[rows]
        common = {
            **{k: v for k, v in meta.items() if k != "model_files"},
            "windows": {"name": ws.name, **file_info(ws.path), "n_rows": len(ws), "alleles": list(ws.alleles),
                        "builder": {k: ws.builder[k] for k in BUILDER_KEYS if k in ws.builder}},  # fmt: skip
            "rows": len(rows), "sample": sample, "sample_seed": sample_seed if sample else None,
            "permute_seed": permute_seed, "batch": batch,
            "seconds": round(seconds, 3), "passes_per_model": passes,
            "passes_per_second_per_model": round(passes / seconds, 1) if seconds > 0 else None,
            "models_in_this_pass": list(models),
            "strand_average": STRAND_AVERAGE, "logfc": LOGFC, "vs_commit": VS_COMMIT, **run_info(),
        }  # fmt: skip
        for m, arr in arrays.items():
            info = {"model": m, **common}
            if m in derived:
                info["derived_from"] = list(derived[m])
                info["derivation"] = "per strand log(exp(nobias) + exp(bias)), then the strand average"
            elif m in meta.get("model_files", {}):
                info["model_file"] = meta["model_files"][m]
            write_predictions(paths[m], ids, arr, info)
        logger.info("%s: %d rows x %d allele(s) x 2 strands, %s, in %.1f s (%.0f passes/s per model)", ws.name,
                    len(rows), len(ws.alleles), "+".join([*models, *derived]), seconds,
                    passes / seconds if seconds > 0 else float("nan"))  # fmt: skip
        summaries.append({"set": ws.name, "rows": len(rows), "seconds": seconds, "passes_per_model": passes})
    return summaries


# ---------------------------------------------------------------------------------------------------- probe
def barrier(directory: Path, tag: str, n: int, timeout: float = 900.0, poll: float = 0.05) -> None:
    """Wait until n processes have reached `tag`: each touches <directory>/<tag>.<pid> and polls for n such files."""
    if n <= 1:
        return
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{tag}.{os.getpid()}").touch()
    t0 = time.monotonic()
    while len(list(directory.glob(f"{tag}.*"))) < n:
        if time.monotonic() - t0 > timeout:
            raise TimeoutError(f"barrier {tag}: fewer than {n} processes arrived within {timeout:.0f} s")
        time.sleep(poll)


def probe(predict: PredictFn, batches: list[int], seconds: float, passes_per_row: int, *, warmup: int = 2,
          rows_per_call: int | None = None, barrier_dir: Path | None = None, barrier_n: int = 1, seed: int = 0,
          window: int = WINDOW, on_result: Callable[[list[dict]], None] | None = None) -> list[dict]:  # fmt: skip
    """Forward passes per second of `predict` at each batch size, on random code rows.

    For each batch size: `warmup` calls on one batch (compilation, autotuning, allocator growth), then calls on
    rows_per_call rows (default max(4 x batch, 2048), whole batches) until `seconds` have passed. Each call returns
    numpy arrays, so it has finished on the device when it returns. With barrier_dir, barrier_n processes start and
    end each measurement together, so 2 processes on one GPU are timed over the same interval. A pass is one
    sequence through one model; passes_per_row is 2 x the number of models a call runs (both strands).
    A batch size that fails (out of GPU memory, say) is recorded with its error, and the process still meets the
    others at that size's barriers; on_result(results so far) runs after every batch size."""
    rng = np.random.default_rng(seed)
    results = []
    for b in batches:
        rows = rows_per_call or max(4 * b, 2048)
        rows = -(-rows // b) * b
        entry: dict = {"batch": b, "rows_per_call": rows}
        tokens = rng.integers(0, 4, size=(rows, window), dtype=np.uint8)
        t0 = time.perf_counter()
        try:
            for _ in range(max(warmup, 1)):
                predict(tokens[:b], b)
        except Exception as err:  # recorded, so that the other batch sizes and processes go on
            logger.exception("batch %d: warm-up failed", b)
            entry["error"] = f"{type(err).__name__}: {err}"[:1000]
        entry["warmup_seconds"] = round(time.perf_counter() - t0, 2)
        if barrier_dir is not None:
            barrier(barrier_dir, f"b{b}.start", barrier_n)
        if "error" not in entry:
            calls = 0
            t0 = time.perf_counter()
            try:
                while True:
                    predict(tokens, b)
                    calls += 1
                    elapsed = time.perf_counter() - t0
                    if elapsed >= seconds:
                        break
                passes = calls * rows * passes_per_row
                entry.update(calls=calls, passes=passes, seconds=round(elapsed, 3),
                             passes_per_second=round(passes / elapsed, 1))  # fmt: skip
                logger.info("batch %d: %.0f passes/s (%d calls of %d rows in %.1f s; warm-up %.1f s)", b,
                            passes / elapsed, calls, rows, elapsed, entry["warmup_seconds"])  # fmt: skip
            except Exception as err:
                logger.exception("batch %d: timed calls failed", b)
                entry["error"] = f"{type(err).__name__}: {err}"[:1000]
        if barrier_dir is not None:
            barrier(barrier_dir, f"b{b}.end", barrier_n)
        results.append(entry)
        if on_result is not None:
            on_result(results)
    return results


# ---------------------------------------------------------------------------------------------------- CLI
def int_list(text: str) -> list[int]:
    try:
        values = [int(v) for v in text.split(",") if v.strip()]
    except ValueError as err:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers, got {text!r}") from err
    if not values or min(values) < 1:
        raise argparse.ArgumentTypeError(f"expected positive integers, got {text!r}")
    return values


def add_common_args(ap: argparse.ArgumentParser, default_batch: int, default_probe_batches: str) -> None:
    io = ap.add_argument_group("windows and outputs")
    io.add_argument("--windows", action="append", default=[], metavar="[NAME=]NPZ",
                    help="a build_windows.py window file; repeat for several. NAME defaults to the file's stem "
                         "(windows/peaks.<dataset>.npz -> peaks.<dataset>)")  # fmt: skip
    io.add_argument("--out-dir", type=Path, help="writes <model>.<NAME>.npz and .json here")
    io.add_argument("--batch", type=int, default=default_batch, help=f"rows per forward call (default {default_batch})")
    io.add_argument("--overwrite", action="store_true", help="redo sets whose outputs exist (default: skip them)")
    checks = ap.add_argument_group("checks")
    checks.add_argument("--sample", type=int, help="score this many rows of each set, drawn with --sample-seed")
    checks.add_argument("--sample-seed", type=int, default=0)
    checks.add_argument("--permute-seed", type=int,
                        help="run rows through the model in an order shuffled with this seed (outputs keep the "
                             "windows' order)")  # fmt: skip
    pr = ap.add_argument_group("probe (instead of predicting windows)")
    pr.add_argument("--probe", action="store_true", help="time forward passes on random windows; write --probe-json")
    pr.add_argument("--probe-json", type=Path)
    pr.add_argument("--probe-batches", type=int_list, default=int_list(default_probe_batches),
                    help=f"comma-separated (default {default_probe_batches})")  # fmt: skip
    pr.add_argument("--probe-seconds", type=float, default=20.0, help="timed seconds per batch size (default 20)")
    pr.add_argument("--probe-rows", type=int, help="rows per timed call (default max(4 x batch, 2048))")
    pr.add_argument("--probe-warmup", type=int, default=2, help="untimed calls per batch size (default 2)")
    pr.add_argument("--barrier", type=Path, help="directory the processes of one probe point synchronise in")
    pr.add_argument("--barrier-n", type=int, default=1, help="processes that meet at --barrier (default 1)")
    ap.add_argument("-v", "--verbose", action="store_true")


def check_common_args(ap: argparse.ArgumentParser, args: argparse.Namespace) -> list[tuple[str, Path]]:
    """Validate the shared options; returns the parsed --windows specs (empty in --probe mode)."""
    if args.batch < 1:
        ap.error("--batch must be at least 1")
    if args.probe:
        if args.probe_json is None:
            ap.error("--probe needs --probe-json")
        return []
    if not args.windows or args.out_dir is None:
        ap.error("give --windows (one or more) and --out-dir, or --probe")
    try:
        specs = [parse_window_spec(s) for s in args.windows]
    except ValueError as err:
        ap.error(str(err))
    names = [n for n, _ in specs]
    if len(set(names)) != len(names):
        ap.error(f"window-set names repeat: {names}")
    return specs


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s", datefmt="%F %T")  # fmt: skip


def write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    os.replace(partial, path)
