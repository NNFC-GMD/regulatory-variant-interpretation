#!/usr/bin/env python3
"""Teach a trained Cherimoya model's heads a control track, with its trunk frozen (the head-only fine-tune arm).

The base is a fold trained without controls. Its checkpoint is rebuilt with control tracks, every weight copied over,
and the two head weights that read the control widened with zeros: fconv.weight gains the control channels at input
channels n_filters.. (the profile head reads cat([trunk, control], dim=1), cherimoya.py _forward_impl) and
linear.weight one column for log(1 + control total), the count head's control input whatever the number of tracks
(n_count_control in Cherimoya.__init__). With those zeros the model predicts exactly what the base does. Then only
fconv.*, linear.*, lw0 and lw1 train; iconv and the blocks keep the base's weights (requires_grad False). The zero
weights can learn: their gradient is the control itself times the gradient at the head's output (fconv), or
log(1 + control total) times the count error (linear), so a nonzero track moves them from the first step.

Everything else is `cherimoya fit -p fit.json` (cherimoya_cli/commands/fit.py run(), whose steps are copied here
in order, cited by line): the same PeakGenerator with the controls, the same validation peaks and negatives with the
controls as in_signals, cherimoya.training.fit with fit.json's optimizers and rates (Muon, AdamW, SGD for lw0/lw1),
EMA and checkpoint rule, then the validation and test evaluates. Only the length differs: --epochs (default 5) with
--warmup-epochs (default 1), the schedule lengths as fit.py computes them, without its min_total_steps floor.

Why freezing needs nothing from training.py:
  optimizers  _split_parameters still gives the frozen weights to Muon and AdamW; both skip a parameter whose grad
              is None (torch/optim/_muon.py _init_group, adam.py _init_group), so a frozen weight gets no update and no
              weight decay. lw0/lw1 stay trainable, since training_step reads lw0.grad.
  EMA         built at on_fit_start, after the freeze, it shadows only parameters that require grad (cherimoya.py
              EMA.__init__), so the checkpoints hold the EMA of the heads and the base's trunk unchanged. That is
              checked after the run, tensor by tensor against the base.
  checkpoint  as in a normal fit: <name>.torch is the EMA at the epoch with the best validation count Pearson (a
              strict improvement, so epoch 0 is always saved), <name>.final.torch the EMA at the end; the evaluates
              score <name>.torch, as fit's do, and so does everything downstream.
  compile     the fine-tune trains eagerly, whatever fit.json's `compile` says (--compile compiles it). With
              the trunk frozen the backward pass is the heads' alone, the step is CPU-bound as a normal fit's is, and
              five epochs do not pay back max-autotune's compile of the training and validation graphs (about 4.5 min
              in a normal fit's first epoch), nor its CUDA-graph pools on a shared GPU. The evaluates keep fit.json's
              `compile`, so the performance files come from the same path as every other fit's.
  float16     refused: its GradScaler fails on an optimizer none of whose parameters has a gradient (Muon's here).

Inputs:
  --base      a control-free Cherimoya checkpoint (Cherimoya.save format)
  --fit-json  the fit.json of a fit with controls (non-null `controls`, devices 1): the data, the splits and the
              training settings. Its name's basename names the outputs; they go to --out-dir whatever its directory.

Outputs, in --out-dir, the files a normal fit leaves (<name> = the basename of fit.json's name):
  <name>.torch, <name>.final.torch, <name>.log, <name>.detailed.log,
  <name>.{validation,test}.evaluate.json, <name>.{validation,test}.performance.tsv
  finetune.json  written last (tmp + rename), the done marker:
                 status ok | no-op | failed, exit, reason; base (path, size, mtime, sha256), fit_json, inputs
                 (what a rerun is matched against), name, controls; epochs, warmup_epochs, steps (per_epoch,
                 warmup, decay, scheduled, taken); compile (fit, evaluate, why); trainable (names, count, total),
                 ema_shadow; data sizes; checkpoint_used (the .torch path), best_epoch, epochs_log (per epoch:
                 validation MNLL, profile and count Pearson, saved);
                 control_channel_norms {"torch": {"fconv", "linear"}, "final.torch": {...}} (the norms of
                 fconv.weight[:, n_filters:, :] and linear.weight[:, n_filters:]); checkpoints (per file: those norms,
                 trunk_unchanged, how far the heads' trunk-side weights moved, lw0, lw1, finite); seconds (load, fit,
                 evaluate, total); finished_et

A run whose finetune.json, checkpoints and test performance exist, for the same base (path, size, mtime), fit.json
(sha256), control tracks (size, mtime) and epochs, is skipped and exits as it did (--force runs again). Otherwise the
outputs above are removed before training, once the base and fit.json are found to describe the same model.

    python workflows/cherimoya/tn5/finetune_head.py --base <fold>/cherimoya.fold_<k>.torch \\
        --fit-json <out-dir>/fit.json --out-dir <out-dir> [--epochs 5] [--warmup-epochs 1] [--compile] [--force]
    python workflows/cherimoya/tn5/finetune_head.py --selftest

Exit 0 when fine-tuned (or already done); 3 when the checkpoint used has a zero control channel, so the arm is the
base model (finetune.json says so, status no-op); 1 when refused or failed (a changed trunk or non-finite weights
included). Run it in the cherimoya environment, on a GPU, as `cherimoya fit` runs. --selftest checks the widening, the
freeze, a two-epoch run on synthetic files through this same code, the checkpoints and the wrappers, on the CPU in a
temporary directory (about a minute).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

logger = logging.getLogger("finetune_head")

FORMAT = 1
"""The layout of finetune.json."""
HEAD_PREFIXES = ("fconv.", "linear.")
LOSS_WEIGHTS = ("lw0", "lw1")
OUTPUT_SUFFIXES = ("torch", "final.torch", "log", "detailed.log", "validation.evaluate.json", "test.evaluate.json",
                   "validation.performance.tsv", "test.performance.tsv")  # fmt: skip
COMPILE_WHY = (
    "eager fine-tune: the backward pass is the heads' alone and the step CPU-bound, so 5 epochs do not pay back "
    "max-autotune's compile or its CUDA-graph pools; the evaluates keep fit.json's compile"
)


def now_et() -> str:
    """The time in ET (America/New_York), where the user is; the cluster clock runs on CEST."""
    return datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds")


def is_head(name: str) -> bool:
    """Whether a parameter (or state-dict key) trains in the fine-tune: the profile head (fconv), the count head
    (linear) and the Kendall loss weights. Everything else, iconv and the blocks, is the frozen trunk."""
    return name.startswith(HEAD_PREFIXES) or name in LOSS_WEIGHTS


def widen_state_dict(state_dict: dict, n_filters: int, n_control_tracks: int) -> dict:
    """A control-free model's state dict for the same model with `n_control_tracks` control tracks, the new weights
    zero: fconv.weight (out, n_filters, k) gains n_control_tracks input channels after the trunk's, linear.weight
    (groups, n_filters) one column. Every other tensor is the base's own."""
    import torch

    out = copy.copy(state_dict)  # keeps the OrderedDict's _metadata, which load_state_dict reads
    for key, extra in (("fconv.weight", n_control_tracks), ("linear.weight", 1)):
        w = state_dict[key]
        if w.shape[1] != n_filters:
            raise ValueError(f"the base's {key} has {w.shape[1]} inputs, not n_filters = {n_filters}: it has controls")
        out[key] = torch.cat([w, w.new_zeros(w.shape[0], extra, *w.shape[2:])], dim=1)
    return out


def freeze_trunk(model) -> list[str]:
    """Leave only the heads (is_head) trainable; the names of the trainable parameters."""
    model.requires_grad_(False)
    names = []
    for name, p in model.named_parameters():
        if is_head(name):
            p.requires_grad_(True)
            names.append(name)
    return names


def build_head_model(payload: dict, n_control_tracks: int, name: str, compile: bool, compile_mode: str,
                     random_state, verbose: bool):  # fmt: skip
    """The base checkpoint's model (`Cherimoya.save`'s payload: config and state dict, read as Cherimoya.load reads
    it) rebuilt with `n_control_tracks` control tracks and the given name, its weights the base's widened with zeros,
    its trunk frozen. Returns the model and the names of its trainable parameters."""
    from cherimoya import Cherimoya

    config = dict(payload["config"])
    if config.get("n_control_tracks", 0):
        raise ValueError(f"the base already takes {config['n_control_tracks']} control track(s)")
    config.update(n_control_tracks=n_control_tracks, name=name, verbose=verbose)
    model = Cherimoya(**config, compile=compile, compile_mode=compile_mode, random_state=random_state)
    model.load_state_dict(widen_state_dict(payload["state_dict"], config["n_filters"], n_control_tracks))
    return model, freeze_trunk(model)


def schedule_steps(n_examples: int, batch_size: int, epochs: int, warmup_epochs: int) -> dict:
    """fit.py's learning-rate schedule lengths (run(), L309-312 and L359-360) for `epochs`, without its
    min_total_steps floor. A trailing partial batch counts as a step there, although training drops it."""
    per_epoch = -(-n_examples // batch_size)
    return {
        "per_epoch": per_epoch,
        "warmup": per_epoch * warmup_epochs,
        "decay": per_epoch * max(1, epochs - warmup_epochs),
        "scheduled": per_epoch * epochs,
    }


def inspect_checkpoint(path: Path, base_state: dict, n_filters: int) -> dict:
    """What a fine-tuned checkpoint holds next to its base: the norms of its control-channel weights
    (fconv.weight[:, n_filters:, :], linear.weight[:, n_filters:]), whether every trunk tensor is the base's bit for
    bit, how far the heads' trunk-side weights moved (relative L2), lw0, lw1, and whether every tensor is finite."""
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
    trunk = [k for k in base_state if not is_head(k)]
    changed = [k for k in trunk if k not in state or not torch.equal(state[k], base_state[k])]

    def norm(x) -> float:
        return float(torch.linalg.vector_norm(x.double()))

    def moved(key: str, part) -> float:
        return norm(state[key][part] - base_state[key]) / max(norm(base_state[key]), 1e-30)

    every = slice(None)
    return {
        "control_norm": {
            "fconv": norm(state["fconv.weight"][:, n_filters:]),
            "linear": norm(state["linear.weight"][:, n_filters:]),
        },
        "trunk_unchanged": not changed,
        "trunk_changed": changed[:10],
        "head_change": {
            "fconv_weight_trunk_side": moved("fconv.weight", (every, slice(0, n_filters))),
            "fconv_bias": moved("fconv.bias", every),
            "linear_weight_trunk_side": moved("linear.weight", (every, slice(0, n_filters))),
            "linear_bias": moved("linear.bias", every),
        },
        "lw0": state["lw0"].tolist(),
        "lw1": state["lw1"].tolist(),
        "finite": all(bool(torch.isfinite(v).all()) for v in state.values() if v.is_floating_point()),
    }


def judge(info: dict) -> tuple[str, int, str]:
    """(status, exit code, reason) for the checkpoint used: failed (1) when a weight is not finite or the trunk moved,
    no-op (3) when either control-channel norm is zero, else ok (0)."""
    if not info["finite"]:
        return "failed", 1, "the checkpoint holds non-finite weights"
    if not info["trunk_unchanged"]:
        return "failed", 1, f"the frozen trunk changed: {', '.join(info['trunk_changed'])}"
    n = info["control_norm"]
    if not (n["fconv"] > 0 and n["linear"] > 0):
        return "no-op", 3, (f"a control-channel norm is zero (fconv {n['fconv']:.3g}, linear {n['linear']:.3g}): "
                            "the control is unused and the arm is the base model")  # fmt: skip
    return "ok", 0, ""


def read_log(path: Path) -> list[dict]:
    """The training log's rows (training.py _LOG_COLUMNS) that finetune.json reports. pandas writes NaN as an empty
    field, read back as None."""
    import csv

    def number(text: str):
        return float(text) if text.strip() else None

    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    return [{"epoch": int(r["Epoch"]), "iteration": int(r["Iteration"]),
             "valid_mnll": number(r["Validation MNLL"]),
             "valid_profile_pearson": number(r["Validation Profile Pearson"]),
             "valid_count_pearson": number(r["Validation Count Pearson"]),
             "saved": r["Saved?"] == "True"} for r in rows]  # fmt: skip


def file_record(path: Path, digest: bool = True) -> dict:
    st = path.stat()
    out = {"path": str(path), "size": st.st_size, "mtime": st.st_mtime}
    if digest:
        out["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def input_identity(base: Path, fit_json: Path, control_files: list[str]) -> dict:
    """What a rerun is matched against: the base (path, size, mtime), fit.json's sha256, and every control track's
    size and mtime. A track rewritten in place (tn5_tracks.py replaces its tracks for a new offset) is new data, so
    the heads trained on the old one are not done."""

    def stat(path, what: str) -> dict:
        try:
            st = Path(path).stat()
        except FileNotFoundError:
            raise FileNotFoundError(f"no {what} at {path}") from None
        return {"path": str(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns}

    return {
        "base": stat(base, "base checkpoint"),
        "fit_json_sha256": hashlib.sha256(fit_json.read_bytes()).hexdigest(),
        "controls": [stat(f, "control track") for f in control_files],
    }


def write_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def finetune(base: Path, fit_json: Path, out_dir: Path, epochs: int = 5, warmup_epochs: int = 1,
             compile_fit: bool = False, force: bool = False) -> int:  # fmt: skip
    """Fine-tune `base`'s heads on fit.json's data and controls into `out_dir`; the exit code (see the header)."""
    t_start = time.time()
    if epochs < 1 or warmup_epochs < 0:
        raise ValueError(f"--epochs must be >= 1 and --warmup-epochs >= 0, got {epochs} and {warmup_epochs}")
    base, fit_json, out_dir = base.absolute(), fit_json.absolute(), out_dir.absolute()
    if not base.is_file():
        raise FileNotFoundError(f"no base checkpoint at {base}")

    from cherimoya.io import normalize_signal_groups
    from cherimoya_cli.defaults import default_fit_parameters
    from cherimoya_cli.utils import merge_parameters

    parameters = merge_parameters(str(fit_json), default_fit_parameters)
    if parameters["skip"]:
        raise ValueError(f"{fit_json} sets skip")
    if parameters["devices"] != 1:
        raise ValueError(f"{fit_json} asks for {parameters['devices']} devices; the fine-tune runs on one")
    if parameters["dtype"] == "float16":
        raise ValueError("dtype float16: its GradScaler refuses an optimizer without gradients, as Muon is here")

    # fit.py L127-134: a chromosome in two splits would score the model on data it was trained on or selected with.
    splits = {s: parameters[s] or [] for s in ("training_chroms", "validation_chroms", "test_chroms")}
    for (a, chroms_a), (b, chroms_b) in itertools.combinations(splits.items(), 2):
        if set(chroms_a) & set(chroms_b):
            raise ValueError(f"{a} and {b} share {sorted(set(chroms_a) & set(chroms_b))}")

    # fit.py L183-184.
    signal_files, signal_groups = normalize_signal_groups(parameters["signals"])
    control_files, control_groups = normalize_signal_groups(parameters["controls"])
    if not control_files:
        raise ValueError(f"{fit_json} has no controls; the fine-tune teaches the heads one")

    # Every output goes to out_dir under the fit's own basename, so a fit.json copied from another arm cannot write
    # into that arm's directory.
    name = out_dir / os.path.basename(parameters["name"] or "cherimoya")
    if str(name) != parameters["name"]:
        logger.info("writing the model as %s (fit.json names it %s)", name, parameters["name"])
    parameters["name"] = str(name)
    outputs = [Path(f"{name}.{s}") for s in OUTPUT_SUFFIXES]
    record_path = out_dir / "finetune.json"
    if base.resolve().parent == out_dir.resolve() or base.resolve() in {p.resolve() for p in outputs}:
        raise ValueError(f"--base {base} is in --out-dir {out_dir}: its outputs would overwrite or delete it")

    identity = input_identity(base, fit_json, control_files)
    if not force and record_path.is_file():
        try:
            done = json.loads(record_path.read_text())
        except json.JSONDecodeError:
            done = {}
        same = (done.get("inputs") == identity and done.get("fit_json") == str(fit_json)
                and done.get("epochs") == epochs and done.get("warmup_epochs") == warmup_epochs)  # fmt: skip
        needed = [Path(f"{name}.torch"), Path(f"{name}.final.torch")]
        if parameters["test_chroms"]:
            needed.append(Path(f"{name}.test.performance.tsv"))
        if same and done.get("status") in ("ok", "no-op") and all(p.is_file() for p in needed):
            print(f"finetune_head: {out_dir} done ({done['status']}), skipped; --force runs it again")
            return int(done.get("exit", 0))

    # The base's own config is the architecture; fit.json must describe the same model (fit.py L283-298 builds it
    # from fit.json). Checked before the earlier outputs are removed and the data loaded, which takes minutes.
    import torch

    payload = torch.load(base, map_location="cpu", weights_only=True)
    config = payload["config"]
    expected = {k: parameters[k] for k in ("n_filters", "n_layers", "expansion", "residual_scale")}
    expected.update(signal_groups=signal_groups, trimming=(parameters["in_window"] - parameters["out_window"]) // 2)
    mismatched = {k: {"base": config.get(k), "fit.json": v} for k, v in expected.items() if config.get(k) != v}
    if mismatched:
        raise ValueError(f"the base and {fit_json} describe different models: {mismatched}")

    out_dir.mkdir(parents=True, exist_ok=True)
    for path in [*outputs, record_path]:
        if path.exists():
            logger.info("removing the earlier %s", path)
            path.unlink()

    parameters["max_epochs"] = epochs
    parameters["n_warmup_epochs"] = warmup_epochs
    parameters["min_total_steps"] = None

    # fit.py L98 and L148-172, for one process: a null seed is drawn, printed and kept in the evaluate JSONs.
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
    import lightning
    import numpy
    from cherimoya.io import PeakGenerator
    from cherimoya.training import fit
    from cherimoya_cli.commands import evaluate as evaluate_cmd
    from tangermeme.io import _interleave_loci, extract_loci

    if parameters["random_state"] is None:
        parameters["random_state"] = int(numpy.random.randint(0, 2**31 - 1))
        print(f"Drew random_state={parameters['random_state']}; set it in the JSON to repeat this run.")
    lightning.seed_everything(parameters["random_state"], verbose=False)

    logger.info("base %s; data and controls from %s: %s", base, fit_json, ", ".join(control_files))

    # fit.py L204-222.
    training_data = PeakGenerator(
        peaks=parameters["loci"],
        negatives=parameters["negatives"],
        sequences=parameters["sequences"],
        signals=parameters["signals"],
        controls=parameters["controls"],
        chroms=parameters["training_chroms"],
        in_window=parameters["in_window"],
        out_window=parameters["out_window"],
        max_jitter=parameters["max_jitter"],
        negative_ratio=parameters["negative_ratio"],
        reverse_complement=parameters["reverse_complement"],
        summits=parameters["summits"],
        exclusion_lists=parameters["exclusion_lists"],
        random_state=parameters["random_state"],
        verbose=parameters["verbose"],
        signal_groups=signal_groups,
        control_groups=control_groups,
    ).dataset

    # fit.py L225-263: validation peaks centred as the training peaks are, then every validation negative, labeled 0.
    valid_data = extract_loci(
        sequences=parameters["sequences"],
        signals=signal_files,
        in_signals=control_files,
        loci=parameters["loci"],
        chroms=parameters["validation_chroms"],
        in_window=parameters["in_window"],
        out_window=parameters["out_window"],
        max_jitter=0,
        summits=parameters["summits"],
        exclusion_lists=parameters["exclusion_lists"],
        ignore=list("QWERYUIOPSDFHJKLZXVBNM"),
        verbose=parameters["verbose"],
    )
    n_valid_peaks, n_valid_negatives = len(valid_data[0]), 0
    valid_labels = None
    if parameters["negatives"] is not None and len(_interleave_loci(
            parameters["negatives"], parameters["validation_chroms"])) > 0:  # fmt: skip
        negative_data = extract_loci(
            sequences=parameters["sequences"],
            signals=signal_files,
            in_signals=control_files,
            loci=parameters["negatives"],
            chroms=parameters["validation_chroms"],
            in_window=parameters["in_window"],
            out_window=parameters["out_window"],
            max_jitter=0,
            exclusion_lists=parameters["exclusion_lists"],
            ignore=list("QWERYUIOPSDFHJKLZXVBNM"),
            verbose=parameters["verbose"],
        )
        n_valid_negatives = len(negative_data[0])
        valid_data = [torch.cat(pair) for pair in zip(valid_data, negative_data)]
        valid_labels = torch.cat([torch.ones(n_valid_peaks), torch.zeros(n_valid_negatives)])

    # fit.py L275-277 (controls are required here).
    valid_sequences, valid_signals, valid_controls = valid_data
    data = {"training_peaks": int(training_data.peak_sequences.shape[0]),
            "training_negatives": int(training_data.negative_sequences.shape[0]),
            "examples_per_epoch": len(training_data), "validation_peaks": n_valid_peaks,
            "validation_negatives": n_valid_negatives}  # fmt: skip
    logger.info("data: %s", data)

    model, trainable = build_head_model(payload, len(control_files), str(name), compile=compile_fit,
                                        compile_mode=parameters["compile_mode"],
                                        random_state=parameters["random_state"],
                                        verbose=parameters["verbose"])  # fmt: skip
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info("training %d of %d parameters: %s", n_trainable, n_total, ", ".join(trainable))

    # fit.py L306-360 for the given length, without the floor.
    steps = schedule_steps(len(training_data), parameters["batch_size"], epochs, warmup_epochs)
    logger.info("%d epoch(s), %d warm-up; %d steps per epoch; Muon lr %s wd %s, AdamW lr %s wd %s; compile %s",
                epochs, warmup_epochs, steps["per_epoch"], parameters["muon_lr"], parameters["muon_wd"],
                parameters["adam_lr"], parameters["adam_wd"], compile_fit)  # fmt: skip
    t_load = time.time() - t_start

    # fit.py L340-369.
    accelerator = {"cuda": "gpu"}.get(parameters["device"], parameters["device"])
    t_fit = time.time()
    trainer = fit(
        model,
        training_data,
        valid_sequences,
        valid_signals,
        X_ctl_valid=valid_controls,
        labels_valid=valid_labels,
        max_epochs=epochs,
        early_stopping=parameters["early_stopping"],
        dtype=parameters["dtype"],
        accelerator=accelerator,
        devices=parameters["devices"],
        verbose=parameters["verbose"],
        progress_bar=parameters["progress_bar"],
        batch_size=parameters["batch_size"],
        num_workers=parameters["num_workers"],
        n_warmup_steps=steps["warmup"],
        n_decay_steps=steps["decay"],
        muon_lr=parameters["muon_lr"],
        muon_wd=parameters["muon_wd"],
        adam_lr=parameters["adam_lr"],
        adam_wd=parameters["adam_wd"],
        lw_lr=parameters["lw_lr"],
        lw_wd=parameters["lw_wd"],
        lw_momentum=parameters["lw_momentum"],
        loss_weights=parameters["loss_weights"],
    )
    t_fit = time.time() - t_fit
    ema_shadow = sorted(trainer.lightning_module.ema.shadow)

    # fit.py L376-394: the validation chromosomes chose the checkpoint, so only the test chromosomes give an estimate
    # that took no part in training.
    t_eval = time.time()
    for split in ("validation", "test"):
        if not parameters[split + "_chroms"]:
            continue
        evaluate_parameters = copy.deepcopy(parameters)
        evaluate_parameters["chroms"] = parameters[split + "_chroms"]
        evaluate_parameters["model"] = f"{name}.torch"
        evaluate_parameters["performance_filename"] = f"{name}.{split}.performance.tsv"
        fname = f"{name}.{split}.evaluate.json"
        with open(fname, "w") as outfile:
            outfile.write(json.dumps(evaluate_parameters, sort_keys=True, indent=4))
        evaluate_cmd.run(argparse.Namespace(parameters=fname))
    t_eval = time.time() - t_eval

    n_filters = config["n_filters"]
    checkpoints = {s: inspect_checkpoint(Path(f"{name}.{s}"), payload["state_dict"], n_filters)
                   for s in ("torch", "final.torch")}  # fmt: skip
    status, code, reason = judge(checkpoints["torch"])
    if status == "ok" and not checkpoints["final.torch"]["trunk_unchanged"]:
        status, code, reason = "failed", 1, "the frozen trunk changed in .final.torch"
    epochs_log = read_log(Path(f"{name}.log"))
    saved = [r["epoch"] for r in epochs_log if r["saved"]]
    record = {
        "format": FORMAT,
        "status": status,
        "exit": code,
        "reason": reason,
        "base": file_record(base),
        "fit_json": str(fit_json),
        "inputs": identity,
        "name": str(name),
        "controls": parameters["controls"],
        "epochs": epochs,
        "warmup_epochs": warmup_epochs,
        "steps": {**steps, "taken": epochs_log[-1]["iteration"] if epochs_log else None},
        "compile": {"fit": compile_fit, "evaluate": parameters["compile"], "why": "" if compile_fit else COMPILE_WHY},
        "trainable": {"names": trainable, "count": n_trainable, "total": n_total},
        "ema_shadow": ema_shadow,
        "data": data,
        "checkpoint_used": f"{name}.torch",
        "checkpoint_rule": "best validation count Pearson (EMA weights), strict improvement, as cherimoya fit",
        "best_epoch": saved[-1] if saved else None,
        "epochs_log": epochs_log,
        "control_channel_norms": {s: c["control_norm"] for s, c in checkpoints.items()},
        "checkpoints": checkpoints,
        "seconds": {
            "load": round(t_load, 1),
            "fit": round(t_fit, 1),
            "evaluate": round(t_eval, 1),
            "total": round(time.time() - t_start, 1),
        },
        "finished_et": now_et(),
    }
    write_json(record_path, record)
    n = record["control_channel_norms"]["torch"]
    print(f"finetune_head: {status}{': ' + reason if reason else ''}; best epoch {record['best_epoch']} of "
          f"{epochs}, control-channel norms fconv {n['fconv']:.4g} linear {n['linear']:.4g}; "
          f"{record['seconds']['total']:.0f} s; {record_path}")  # fmt: skip
    return code


# ---------------------------------------------------------------------------------------------------------------------
# --selftest


def _write_synthetic_inputs(d: Path, rng, sizes: dict, peaks: dict, negatives: dict, in_window: int) -> dict:
    """A genome, peak and negative beds, a signal bigwig with bumps at the peaks and a positive control bigwig."""
    import numpy as np
    import pyBigWig

    fasta = d / "genome.fa"
    with open(fasta, "w") as fh:
        for c, n in sizes.items():
            seq = "".join(rng.choice(list("ACGT"), n))
            fh.write(f">{c}\n" + "".join(seq[i : i + 60] + "\n" for i in range(0, n, 60)))
    signal = {c: rng.poisson(0.2, n).astype(np.float64) for c, n in sizes.items()}
    for c, mids in peaks.items():
        for m in mids:
            signal[c][m - 30 : m + 30] += rng.poisson(2.0, 60)
    control = {c: rng.uniform(0.5, 3.0, n) for c, n in sizes.items()}
    paths = {"fasta": fasta, "signal": d / "signal.bw", "control": d / "control.bw"}
    for key, values in (("signal", signal), ("control", control)):
        bw = pyBigWig.open(str(paths[key]), "w")
        bw.addHeader(list(sizes.items()), maxZooms=0)
        for c, v in values.items():
            bw.addEntries(c, 0, values=v.tolist(), span=1, step=1)
        bw.close()
    for key, loci, half in (("peaks", peaks, 50), ("negatives", negatives, in_window // 2)):
        paths[key] = d / f"{key}.bed"
        paths[key].write_text("".join(f"{c}\t{m - half}\t{m + half}\n" for c, mids in loci.items() for m in mids))
    return paths


def selftest() -> int:
    """The checks in the header's last paragraph; 0 when every one holds."""
    import numpy as np
    import torch
    from cherimoya import Cherimoya, ControlWrapper, LogCountWrapper, ProfileWrapper

    failures = []

    def check(ok, what: str) -> None:
        logger.info("%s: %s", "ok" if ok else "FAILED", what)
        if not ok:
            failures.append(what)

    n_filters, n_layers, in_window, out_window = 16, 2, 200, 100
    trimming = (in_window - out_window) // 2  # 50 > fconv's padding of 37, as 557 is for the real model
    rng = np.random.default_rng(1234)
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="finetune_head.") as tmp:
        tmp = Path(tmp)
        base = Cherimoya(n_filters, n_layers, trimming=trimming, name=str(tmp / "base"), compile=False, random_state=0)
        base.save(str(tmp / "base.torch"))
        payload = torch.load(tmp / "base.torch", map_location="cpu", weights_only=True)

        # The widening and the freeze, before any training.
        wide, trainable = build_head_model(payload, 1, str(tmp / "wide"), compile=False, compile_mode="max-autotune",
                                           random_state=1, verbose=False)  # fmt: skip
        check(set(trainable) == {"fconv.weight", "fconv.bias", "linear.weight", "linear.bias", "lw0", "lw1"},
              f"only the heads train: {trainable}")  # fmt: skip
        X = torch.nn.functional.one_hot(torch.randint(0, 4, (8, in_window)), 4).transpose(1, 2).float()
        X_ctl = torch.rand(8, 1, in_window) * 3
        base.eval()
        wide.eval()
        with torch.no_grad():
            (yb, cb), (yw, cw) = base(X), wide(X, X_ctl)
        check(torch.allclose(yb, yw, atol=1e-6) and torch.allclose(cb, cw, atol=1e-6),
              "with zero control weights the widened model predicts what the base does")  # fmt: skip
        wide.save(str(tmp / "wide.torch"))
        status, code, _ = judge(inspect_checkpoint(tmp / "wide.torch", payload["state_dict"], n_filters))
        check((status, code) == ("no-op", 3), f"an untrained control channel is a no-op (exit 3): {status} {code}")

        # With a control channel that is far from zero, a constant control (the zeros attribution feeds) is a constant
        # logit shift over the output window, which the profile target's mean-centring and softmax remove, while a
        # varying control changes it.
        with torch.no_grad():
            wide.fconv.weight[:, n_filters:] = torch.randn(wide.fconv.weight[:, n_filters:].shape) * 0.5
            profile = ProfileWrapper(ControlWrapper(wide))
            p_zero, p_flat, p_real = profile(X), profile(X, torch.full_like(X_ctl, 1.7)), profile(X, X_ctl)
        check(torch.allclose(p_flat, p_zero, atol=1e-5), "a constant control leaves the profile target unchanged")
        check(not torch.allclose(p_real, p_zero, atol=1e-3), "a varying control changes the profile target")

        # Two epochs on synthetic files through finetune() itself, on the CPU.
        sizes = {"chrT": 40000, "chrV": 12000, "chrE": 12000}
        peaks = {"chrT": [500 + 240 * i for i in range(160)], "chrV": [500 + 300 * i for i in range(32)],
                 "chrE": [500 + 300 * i for i in range(32)]}  # fmt: skip
        negatives = {"chrT": [620 + 600 * i for i in range(64)], "chrV": [650 + 600 * i for i in range(16)],
                     "chrE": [650 + 600 * i for i in range(16)]}  # fmt: skip
        paths = _write_synthetic_inputs(tmp, rng, sizes, peaks, negatives, in_window)
        out_dir = tmp / "arm" / "fold_0"
        out_dir.mkdir(parents=True)
        fit_json = out_dir / "fit.json"
        fit_json.write_text(json.dumps({
            "name": "cherimoya.fold_0", "sequences": str(paths["fasta"]), "loci": [str(paths["peaks"])],
            "negatives": [str(paths["negatives"])], "signals": [str(paths["signal"])],
            "controls": [str(paths["control"])], "exclusion_lists": None, "training_chroms": ["chrT"],
            "validation_chroms": ["chrV"], "test_chroms": ["chrE"], "random_state": 1234, "verbose": False,
            "n_filters": n_filters, "n_layers": n_layers, "in_window": in_window, "out_window": out_window,
            "max_jitter": 10, "batch_size": 16, "num_workers": 0, "device": "cpu", "compile": False,
        }, indent=4))  # fmt: skip
        code = finetune(tmp / "base.torch", fit_json, out_dir, epochs=2, warmup_epochs=1)
        check(code == 0, f"finetune() exits 0: {code}")
        record = json.loads((out_dir / "finetune.json").read_text())
        name = out_dir / "cherimoya.fold_0"
        check(all(Path(f"{name}.{s}").is_file() for s in OUTPUT_SUFFIXES), "a normal fit's files are all there")
        check(record["status"] == "ok" and record["exit"] == 0, f"finetune.json: {record['status']} {record['reason']}")
        check(record["steps"]["taken"] >= 20, f"at least 20 steps taken: {record['steps']}")
        check(record["ema_shadow"] == sorted(trainable), f"the EMA shadows the heads alone: {record['ema_shadow']}")
        check(record["compile"]["fit"] is False and record["checkpoint_used"] == f"{name}.torch", "eager, .torch")
        for s, info in record["checkpoints"].items():
            n = info["control_norm"]
            check(
                info["trunk_unchanged"] and n["fconv"] > 0 and n["linear"] > 0,
                f"{s}: trunk as the base, control channel moved (fconv {n['fconv']:.3g}, linear {n['linear']:.3g})",
            )
            check(info["head_change"]["fconv_weight_trunk_side"] > 0, f"{s}: the profile head's trunk side trained")
        evaluate = json.loads(Path(f"{name}.test.evaluate.json").read_text())
        check(evaluate["model"] == f"{name}.torch" and evaluate["max_epochs"] == 2, "the evaluate JSON scores .torch")
        check(Path(f"{name}.test.performance.tsv").read_text().startswith("profile_mnll"), "test performance written")
        mtime = Path(f"{name}.torch").stat().st_mtime
        code = finetune(tmp / "base.torch", fit_json, out_dir, epochs=2, warmup_epochs=1)
        check(code == 0 and Path(f"{name}.torch").stat().st_mtime == mtime, "a finished run is skipped when rerun")

        # The checkpoint through Cherimoya.save/load and the wrappers attribution uses.
        model = Cherimoya.load(f"{name}.torch", compile=False)
        model.save(str(tmp / "resaved.torch"))
        model = Cherimoya.load(str(tmp / "resaved.torch"), compile=False).eval()
        check(model.n_control_tracks == 1, "the checkpoint loads with one control track")
        state = model.state_dict()
        check(all(torch.equal(v, state[k]) for k, v in base.state_dict().items() if not is_head(k)),
              "every frozen tensor of the reloaded checkpoint is the base's")  # fmt: skip
        X = torch.nn.functional.one_hot(torch.randint(0, 4, (8, in_window)), 4).transpose(1, 2).float()
        with torch.no_grad():
            profile = ProfileWrapper(ControlWrapper(model))
            counts = LogCountWrapper(ControlWrapper(model))
            p0, c0 = profile(X), counts(X)
            finite = bool(torch.isfinite(p0).all() & torch.isfinite(c0).all())
            check(p0.shape == (8, 1) and c0.shape == (8, 1) and finite,
                  "ProfileWrapper and LogCountWrapper over ControlWrapper run with no control")  # fmt: skip
            # The EMA'd control weights are small after 24 steps, so these ask for any difference at all.
            check(bool((counts(X, X_ctl) != c0).any()), "the real control moves the counts")
            moved = ControlWrapper(model)(X, X_ctl)[0] != ControlWrapper(model)(X)[0]
            check(bool(moved.any()), "the real control moves the profile logits")
    print(
        f"finetune_head selftest: {'ok' if not failures else 'FAILED: ' + '; '.join(failures)} "
        f"({time.time() - t0:.0f} s)"
    )
    return 1 if failures else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, help="the control-free checkpoint to fine-tune (<name>.torch)")
    ap.add_argument("--fit-json", type=Path, help="fit.json of a fit with controls: data, splits, settings")
    ap.add_argument("--out-dir", type=Path, help="where the outputs go (<name>.* and finetune.json)")
    ap.add_argument("--epochs", type=int, default=5, help="epochs to train (default 5)")
    ap.add_argument("--warmup-epochs", type=int, default=1, help="epochs of learning-rate warm-up (default 1)")
    ap.add_argument("--compile", action="store_true",
                    help="train under torch.compile (fit.json's compile_mode) instead of eagerly")  # fmt: skip
    ap.add_argument("--force", action="store_true", help="run again although finetune.json says it is done")
    ap.add_argument("--selftest", action="store_true", help="the synthetic checks, on the CPU in a temp dir")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%F %T")  # fmt: skip
    if args.selftest:
        return selftest()
    if args.base is None or args.fit_json is None or args.out_dir is None:
        ap.error("--base, --fit-json and --out-dir are required (or --selftest)")
    return finetune(args.base, args.fit_json, args.out_dir, args.epochs, args.warmup_epochs, args.compile, args.force)


if __name__ == "__main__":
    sys.exit(main())
