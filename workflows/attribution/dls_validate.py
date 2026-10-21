#!/usr/bin/env python3
"""GPU validation of cheri_dls on the real model: the stock reference's noise floors, an fp64 truth, and the gates.

This is the harness of design §15.3 (analyses/attr_pilot/design/final_design.md), step 9 of final_rest.json, for the
counts head: acceptance criteria E1-E8 and E10, and G0, the GPU half of B1. The stock reference is what
`cherimoya attribute` runs: tangermeme's deep_lift_shap through cherimoya's LogCountWrapper(ControlWrapper(model),
group) with attribution_ops(), in float32 on CUDA. The fast engine is cheri_dls.Engine. Both see the same peaks and
the same injected references. Faithfulness is judged relative to the reference's own noise. Two runs of the
reference differ: cuDNN's input-grad is not repeatable, cudnn.benchmark and Triton's autotuner pick kernels by timing
in each process, and the rescale rule's secants amplify ulp-level differences. A fixed tolerance would either pass a
real error or fail an exact engine.

Subcommands, in order. Every one runs inside a box. All of them import torch except `checkpoint-json`, `summarize`
and the drivers of `oracle` and `ours`, which only start and watch one child process per run, one at a time.

- checkpoint-json: the attribute.json of another trained checkpoint. It copies a base JSON (the pilot's) and changes
  only "model", ``<fold>/cherimoya.<fold>.torch``, and "loci", the fold's fit.json "loci". The three output paths point
  under --out-dir and stay unused. The chromosome list is the base's and must name all 24 chromosomes: cherimoya's
  default list is the pilot fold's training and validation chromosomes, 21 of them.
- prepare: reads the attribute.json the way `cherimoya attribute` does (attribute.py:21-86: merge_parameters,
  extract_loci, the N filter, the centred window). P256 is 256 peaks evenly spaced over the kept peaks,
  round(linspace(0, n - 1, 256)). The 8 truth peaks are evenly spaced over P256, round(linspace(0, 255, 8)).
  P256's references are drawn with dls_refs.RefProducer, the engine's own path. Every pair is checked against
  tangermeme's ``dinucleotide_shuffle(X[e:e+1].float(), n=1, random_state=rs + k)[:, 0]`` (G0). Writes <out>/inputs/:
  the kept peaks as uint8 tokens, the indices, P256's reference tokens and meta.json, and p256.tsv (p256-table).
- p256-table: writes <out>/inputs/p256.tsv from what prepare saved. Each P256 peak gets its kept index, its row among
  the loci extract_loci read, its coordinates and whether it is a truth peak. The loci are tangermeme's own
  _interleave_loci over the JSON's loci and chroms; with one loci file and every chromosome listed, the row is the
  file's line.
- oracle: the stock runs, each in a fresh subprocess, so each picks its own cuDNN algorithms and Triton
  configurations, as separate `cherimoya attribute` runs do.
  - R1, R1p (R1'): the flags as `import cherimoya` leaves them (TF32 GEMMs, cudnn.allow_tf32, cudnn.benchmark).
  - R2, R2p (R2'): the same, then ``torch.backends.cuda.matmul.allow_tf32 = False`` and
    ``torch.backends.cudnn.allow_tf32 = False`` right after the import.
  - R1_8: R1's settings on the 8 truth peaks alone.

  Each child calls ``deep_lift_shap(wrapper, X, references=R, raw_outputs=True, hypothetical=True,
  additional_nonlinear_ops=attribution_ops(), dtype="float32", device="cuda", batch_size=--batch-size)``. The
  per-pair convergence deltas come through a module-global ``print`` in tangermeme.deep_lift_shap, whose only print is
  the delta print (TM/deep_lift_shap.py:749). The hypothetical attributions are computed on the CPU from the raw
  multipliers: tangermeme's own hypothetical_attributions per pair, then torch.stack and mean over the references,
  then the window slice. These are the ops deep_lift_shap and attribute.py apply (TM/deep_lift_shap.py:754, 767-770;
  attribute.py:88-101). Last, the wrapper's own forward runs again on deep_lift_shap's own batches, under
  torch.enable_grad(), the Triton training path that deep_lift_shap runs, and gives E1's reference outputs. no_grad
  would select the bf16 megakernel (CH/cheri.py:1105-1118).
- ours: cheri_dls.Engine runs, each in a fresh subprocess, on CUDA with mem_budget_gb 8.
  - T1, T1p (T1'): reference precision, compiled, S = 4. T1p repeats T1, which measures the engine's own rerun noise.
  - T2: fp32 precision (strict IEEE), compiled, S = 4.
  - T3: reference precision, eager (compile "none"), S = 4.
  - T1_S8, T2_S8: T1 and T2 at S = 8, for E10.
  - P5000_ref, P5000_fp32: every kept peak, with the references the engine draws itself (RefProducer), S = 4, for E8.

  The P256 runs call Engine.raw() (multipliers, outputs, deltas) and Engine.run() (the windowed attributions
  `cherimoya attribute` saves), both with P256's injected references.
- truth: T64, the engine on the CPU in float64 with two-pass statistics, on the 8 truth peaks. The model is
  ``.double()``'d and passed through tests/dls_tiny.enable_float64. Its hypothetical attributions come from its
  multipliers through the oracle's CPU path, in float64.
- gates: reads every run and writes report.json, report.tsv and report.md into --out-dir.
  - E2 floors, per head: F_rr = R1' vs R1, F_hh = R2' vs R2, F_tf = R2 vs R1. Each is measured on the per-pair
    rel-L2 of the raw multipliers over 4 x L and the per-peak rel-L2 of the windowed attributions over 4 x W, at
    q = 0.5, 0.99 and max. "a vs b" always means ||a - b|| / ||b||, so the floors are relative to R1 (R2 for F_hh)
    and every gate relative to the reference run it names.
  - The gates: E1 forward parity, E3, E4, E5, E6, E7 (applied only when E3 or E4 fails), E8, E10, and G0.
  - E1 in reference mode and E10 in fp32 mode are gated as revised after round 2 (analyses/attr_pilot/design/
    revisions.md). Their original fixed bounds sat below the stock reference's own rerun noise:
    - E1, reference mode: max |y_ours - y_R1| <= 2 x max |y_R1' - y_R1|, the stock forward's own rerun noise on the
      same rows, and max |y_ours - y_T64| <= 1.2 x max |y_R1 - y_T64| on the truth peaks. fp32 mode keeps 5e-5.
    - E10, fp32 mode: E4's form against F_hh, per-pair raw q99 <= max(3 F_hh(0.99), 1e-4) and q50 <=
      max(2 F_hh(0.5), 1e-5).
    The original bounds (2e-4; p99 <= 1e-5) are still computed and reported, in report.json's "revisions" and the
    report's "Revised criteria" section, but do not gate.
  - Context, not gated: T1' vs T1, R1_8 vs R1, every run's distance to T64, every run's forward error against T64,
    every run's deltas, and with --probe-json the cuDNN classes probe P3 (dls_probe.py) found at the row counts these
    runs used. Also E3's precision-policy margins: T1 vs R1's median, and the stock rerun's (F_rr), as fractions of
    0.3 F_tf(0.5). And E7(a)'s ratios d(ours, T64) / d(R, T64), computed whether or not E3 and E4 pass.
  - E10 also checks that each run it compares ran at its specified S, in raw() and in run(). An out-of-memory halving
    would otherwise compare a run with itself.
  - --plan names the runs and gates of a validation (PLANS). "full" (the default) is everything above, on the pilot
    model. "checkpoint" validates another checkpoint: the oracle R1, R1', R2 and R2', ours T1, T2 and T1_S8, the truth.
    It gates G0, E1, E2, E3, E4, E5 and E10 in reference mode, with E7(a) as context. The other gates are out of scope:
    they are listed as such and do not count toward the verdict. A gate that reads a planned run that is missing is
    "not run".
- summarize: one summary.md over several validation directories (one per checkpoint), from their report.json and
  inputs/meta.json. It has the verdicts, the floors, E3's policy margins, E7(a)'s ratios and every gate check's
  measured value against its bound. --reference-dir adds a column for another validation, such as the pilot's.

Deviations from the design, forced by the rules for the shared GPUs and recorded in the report:
- The oracle's batch_size. Stock deep_lift_shap at the pilot's batch_size 64 allocates 15.5 GB on this model, and
  our processes must stay under 8 GB on the shared GPUs. The oracle therefore runs at --batch-size: 20 in the
  recorded run, one peak's 20 pairs per batch, 40 rows, and no partial batch. Nothing in the reference mixes rows:
  the normalization is per example, the head per row, and the K = 1 head GEMM is exact at any row count (probe P4).
  A batch size can change only which kernels cuDNN and Triton pick for the shape. A rerun shows the same kind of
  difference.
- E10 compares S = 4 with S = 8, the largest step within the 8 GB budget. step_bytes puts S = 16 at 14.9 GB.

Run it in a box, on one GPU and four pinned cores, one GPU job at a time:

    CUDA_VISIBLE_DEVICES=<GPU> TORCHINDUCTOR_COMPILE_THREADS=1 NUMBA_CACHE_DIR=<node-local> taskset -c <cores> \\
        python workflows/attribution/dls_validate.py prepare -p attribute.json --out-dir <dir>
    ... dls_validate.py oracle --out-dir <dir> --runs R1,R1p,R2,R2p,R1_8 --batch-size 20
    ... dls_validate.py ours --out-dir <dir> --runs T1,T1p,T2,T3,T1_S8,T2_S8,P5000_ref,P5000_fp32
    CUDA_VISIBLE_DEVICES= ... dls_validate.py truth --out-dir <dir>
    CUDA_VISIBLE_DEVICES= ... dls_validate.py gates --out-dir <dir>

Another checkpoint, one directory per checkpoint, one checkpoint at a time:

    dls_validate.py checkpoint-json --base attr_pilot/attribute.json --fold-dir <dataset>/<fold> --out-dir <dir>
    ... dls_validate.py prepare -p <dir>/attribute.json --out-dir <dir>
    ... dls_validate.py oracle --plan checkpoint --out-dir <dir> --batch-size 20
    ... dls_validate.py ours --plan checkpoint --out-dir <dir> --mem-budget-gb 7.5
    CUDA_VISIBLE_DEVICES= ... dls_validate.py truth --out-dir <dir>
    CUDA_VISIBLE_DEVICES= ... dls_validate.py gates --plan checkpoint --out-dir <dir>
    dls_validate.py summarize --out <summary.md> <dir> [<dir> ...] [--reference-dir attr_pilot/validation]
"""

from __future__ import annotations

import argparse
import builtins
import contextlib
import hashlib
import json
import logging
import math
import os
import socket
import subprocess
import sys
import time
import warnings
from collections.abc import Iterator
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from dls_bench import _compute_apps, _now_et  # noqa: E402  (stdlib only at import)

logger = logging.getLogger("dls_validate")

ORACLE_RUNS = {
    "R1": {"tf32": True, "peaks": "p256", "about": "stock; the flags as `import cherimoya` leaves them"},
    "R1p": {"tf32": True, "peaks": "p256", "about": "R1': a rerun of R1"},
    "R2": {"tf32": False, "peaks": "p256", "about": "stock; cuda.matmul.allow_tf32 and cudnn.allow_tf32 off"},
    "R2p": {"tf32": False, "peaks": "p256", "about": "R2': a rerun of R2"},
    "R1_8": {"tf32": True, "peaks": "truth", "about": "R1's settings on the 8 truth peaks alone"},
}
OURS_RUNS = {
    "T1": {"precision": "reference", "compile": "default", "S": 4, "peaks": "p256"},
    "T1p": {"precision": "reference", "compile": "default", "S": 4, "peaks": "p256"},
    "T2": {"precision": "fp32", "compile": "default", "S": 4, "peaks": "p256"},
    "T3": {"precision": "reference", "compile": "none", "S": 4, "peaks": "p256"},
    "T1_S8": {"precision": "reference", "compile": "default", "S": 8, "peaks": "p256"},
    "T2_S8": {"precision": "fp32", "compile": "default", "S": 8, "peaks": "p256"},
    "P5000_ref": {"precision": "reference", "compile": "default", "S": 4, "peaks": "all"},
    "P5000_fp32": {"precision": "fp32", "compile": "default", "S": 4, "peaks": "all"},
}
TRUTH_RUN = "T64"
REFERENCE_OF = {"reference": "R1", "fp32": "R2"}  # the stock run a mode is judged against

GATES = ("G0", "E1", "E2", "E3", "E4", "E5", "E6", "E8", "E10")
# The runs each gate reads. When a run the plan has is missing, the gates that read it are "not run" (None).
GATE_RUNS = {
    "G0": (),
    "E1": ("R1", "R1p", "R2", "T1", "T2", "T3", TRUTH_RUN),
    "E2": ("R1", "R1p", "R2", "R2p"),
    "E3": ("T1", "R1", "R1p", "R2"),
    "E4": ("T2", "R2", "R2p"),
    "E5": ("R1", "R2", "T1", "T1p", "T3", "T1_S8", "T2", "T2_S8"),
    "E6": ("T3", "T1", "R1", "R1p", "R2"),
    "E8": ("P5000_ref",),
    "E10": ("T1_S8", "T1", "R1", "R1p", "T2_S8", "T2", "R2", "R2p"),
}
# A validation's runs and the gates its verdict covers. "full" is design §15.3 on the pilot model, as rounds 2 and 3
# ran it. "checkpoint" repeats the floors, T1, T2, T1_S8 and the truth on another trained checkpoint. Without T3, P5000
# and T2_S8, E6, E8 and E10's fp32 mode are out of its scope.
PLANS = {
    "full": {
        "about": "design §15.3: every stock and engine run, the fp64 truth, every gate",
        "oracle": tuple(ORACLE_RUNS),
        "ours": tuple(OURS_RUNS),
        "gates": GATES,
    },
    "checkpoint": {
        "about": "another checkpoint: R1, R1', R2, R2'; T1, T2, T1_S8; the fp64 truth. G0, E1, E2, E3, E4, E5 and E10 "
        "in reference mode; E7(a) as context",
        "oracle": ("R1", "R1p", "R2", "R2p"),
        "ours": ("T1", "T2", "T1_S8"),
        "gates": ("G0", "E1", "E2", "E3", "E4", "E5", "E10"),
    },
}


def plan_runs(plan: str) -> list[str]:
    """Every run a plan has, the truth included, in the order the report lists runs."""
    return [*PLANS[plan]["oracle"], *PLANS[plan]["ours"], TRUTH_RUN]


# Acceptance constants (final_rest.json acceptance_criteria E1-E10; design §15.4), as revised after round 2 for E1 in
# reference mode and E10 in fp32 mode (design/revisions.md). Nothing else is relaxed here.
MED_FACTOR, P99_FACTOR = 2.0, 3.0  # E3/E4/E6: median <= max(2 F(0.5), 1e-5), p99 <= max(3 F(0.99), 1e-4)
MED_MIN, P99_MIN = 1e-5, 1e-4
POLICY_FACTOR = 0.3  # E3: median <= 0.3 F_tf(0.5)
PEARSON_MIN, HYP_REL_MAX = 0.9999, 1e-2  # E5 caps, per peak
E1_MAX = {"reference": 2e-4, "fp32": 5e-5}  # max-abs on the counts output; reference mode's is the original, reported
E1_FLOOR_FACTOR = 2.0  # revised E1, reference mode: max|y_ours - y_R1| <= 2 x max|y_R1' - y_R1|
ACCURACY_FACTOR = 1.2  # E7(a); also revised E1's accuracy clause: max|y_ours - y_T64| <= 1.2 x max|y_R1 - y_T64|
E8_FRACTION_FLOOR, E8_MAX_FLOOR = 0.003, 4e-3  # E8: fraction <= max(2 R1's, 0.3%), max <= max(2 R1's max, 4e-3)
E8_FP32_PREDICTION = 1e-4  # E8, fp32 mode: a prediction, not a gate
E10_FP32_P99_MAX = 1e-5  # the original E10 fp32 bound, reported only; revised: E4's form against F_hh
REVISIONS_DOC = "analyses/attr_pilot/design/revisions.md"
# The pilot's stock run of all 4,999 kept peaks (attr_pilot/attribute.log, recounted in final_rest.json's corrections).
PILOT = {"above": 153, "pairs": 99_980, "max": 1.9636e-3}

QUANTILES = (("q50", 0.5), ("q99", 0.99))
TINY = 1e-30
ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_COMPILE_THREADS",
    "TRITON_CACHE_AUTOTUNING",
    "NUMBA_CACHE_DIR",
    "OMP_NUM_THREADS",
    "PYTORCH_CUDA_ALLOC_CONF",
    "CUBLAS_WORKSPACE_CONFIG",
    "NVIDIA_TF32_OVERRIDE",
)


# ---------------------------------------------------------------------------------------------------------------------
# metrics (numpy only; shared by the gates and the tests)


def rel_l2(a, b, lead: int, chunk: int = 8):
    """||a - b||_2 / max(||b||_2, 1e-30) in float64 for each index of the first `lead` axes, flattened.

    a and b have equal shapes (memory-mapped arrays work: they are read `chunk` entries of the first axis at a time).
    lead=2 on multipliers (N, K, 4, L) gives one value per pair; lead=1 on attributions (N, 4, W) one per peak.
    """
    import numpy as np

    if a.shape != b.shape:
        raise ValueError(f"shapes differ: {a.shape} vs {b.shape}")
    out = []
    for i in range(0, a.shape[0], chunk):
        x = np.asarray(a[i : i + chunk], dtype=np.float64)
        y = np.asarray(b[i : i + chunk], dtype=np.float64)
        x = x.reshape(x.shape[:lead] + (-1,))
        y = y.reshape(y.shape[:lead] + (-1,))
        out.append((np.linalg.norm(x - y, axis=-1) / np.maximum(np.linalg.norm(y, axis=-1), TINY)).ravel())
    return np.concatenate(out)


def pearson_rows(a, b):
    """Pearson correlation of each row (first axis) of a and b over all its other values, in float64."""
    import numpy as np

    x = np.asarray(a, dtype=np.float64).reshape(len(a), -1)
    y = np.asarray(b, dtype=np.float64).reshape(len(b), -1)
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    return (x * y).sum(axis=1) / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-300)


def stats(values) -> dict:
    """n, q50, q99 and max of a vector (numpy's default linear quantiles)."""
    import numpy as np

    v = np.asarray(values, dtype=np.float64).ravel()
    out = {"n": int(v.size)}
    out.update({name: float(np.quantile(v, q)) for name, q in QUANTILES})
    out["max"] = float(v.max())
    return out


def floor_bounds(floor: dict) -> dict:
    """E3/E4's limits from one floor's stats on one metric: median and p99 bounds (final_rest.json E3)."""
    return {
        "q50": max(MED_FACTOR * floor["q50"], MED_MIN),
        "q99": max(P99_FACTOR * floor["q99"], P99_MIN),
    }


def e8_bounds(above: int, pairs: int, maximum: float) -> dict:
    """E8's upper bounds from the reference's exceedance count, pair count and max delta."""
    return {
        "fraction": max(2.0 * above / pairs, E8_FRACTION_FLOOR),
        "max": max(2.0 * maximum, E8_MAX_FLOOR),
    }


def accuracy_limited(ours: dict, ref: dict, factor: float = ACCURACY_FACTOR) -> dict:
    """E7(a): d(ours, T64) <= factor x d(R, T64) at the median and the max; ours and ref are stats() of one metric."""
    return {stat: bool(ours[stat] <= factor * ref[stat]) for stat in ("q50", "max")}


def e3_policy(floors: dict, e3_rows: list[dict]) -> dict:
    """E3's precision-policy check, median <= 0.3 F_tf(0.5), per metric, with its margins.

    T1 vs R1's median and the stock rerun's, F_rr(0.5) (R1' vs R1), are each given as a fraction of the bound. A run
    under the reference's precision policy differs from R1 by about what a rerun does; one under another policy lands
    near F_tf. Read from the floors and E3's rows, which every report.json has.
    """
    out = {}
    f_rr = floors.get("F_rr")
    for row in e3_rows:
        if row["statistic"] != "q50 policy" or not row["bound"]:
            continue
        metric, bound = row["metric"], row["bound"]
        rerun = f_rr[metric]["q50"] if f_rr else None
        out[metric] = {
            "bound": bound,
            "rule": row["rule"],
            "ours_q50": row["measured"],
            "ours_ratio": row["measured"] / bound,
            "stock_rerun_q50": rerun,
            "stock_rerun_ratio": None if rerun is None else rerun / bound,
            "pass": row["pass"],
        }
    return out


def e7_context(e7_computed: dict) -> dict:
    """E7(a)'s tie-breaker as context: per gate and metric, d(ours, T64) / d(R, T64) at the median and the max, and
    whether both are within 1.2. Read from report.json's "E7_computed", which gates fills whenever T64 exists."""
    out = {}
    for gate, tb in e7_computed.items():
        metrics = {}
        for metric, v in tb["metrics"].items():
            metrics[metric] = {
                "d_ours": {s: v["d_ours"][s] for s in ("q50", "max")},
                "d_ref": {s: v["d_ref"][s] for s in ("q50", "max")},
                "ratio": {s: v["d_ours"][s] / max(v["d_ref"][s], TINY) for s in ("q50", "max")},
                "within": bool(all(v["pass"].values())),
            }
        out[gate] = {"ours": tb["ours"], "reference": tb["reference"], "metrics": metrics}
    return out


ALL_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
UNUSED_OUTPUTS = (
    ("ohe_filename", "attr.ohe.npz"),
    ("attr_filename", "attr.attr.npz"),
    ("idx_filename", "attr.idx.npy"),
)


def checkpoint_parameters(base: dict, fold_dir: Path, out_dir: Path) -> dict:
    """The attribute.json of the checkpoint trained in fold_dir: `base` with "model" ``<fold_dir>/cherimoya.<fold>.torch``
    and "loci" the fold's fit.json "loci", and its output paths under out_dir/stock_unused. Nothing else changes.

    The base must list all 24 chromosomes: cherimoya's default list is the pilot fold's training and validation
    chromosomes, 21 of them, and "all 24 chromosomes" is the design's peak set (§15.3).
    """
    fold_dir = Path(fold_dir)
    fold = fold_dir.name
    fit = json.loads((fold_dir / "fit.json").read_text())
    if fit.get("name", f"cherimoya.{fold}") != f"cherimoya.{fold}":
        raise ValueError(f"{fold_dir / 'fit.json'} names the model {fit['name']!r}, not cherimoya.{fold}")
    model = fold_dir / f"cherimoya.{fold}.torch"
    loci = [str(p) for p in (fit["loci"] if isinstance(fit["loci"], list) else [fit["loci"]])]
    missing = [str(p) for p in (model, *loci) if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"missing: {missing}")
    chroms = base.get("chroms") or []
    if sorted(chroms) != sorted(ALL_CHROMS):
        raise ValueError(f"the base JSON must list the 24 chromosomes {ALL_CHROMS}, not {chroms}")
    parameters = dict(base)
    parameters["model"], parameters["loci"] = str(model), loci
    for key, name in UNUSED_OUTPUTS:
        parameters[key] = str(Path(out_dir) / "stock_unused" / name)
    return parameters


def write_p256_table(out_dir: Path) -> Path:
    """<out>/inputs/p256.tsv: each P256 peak's position in P256, its kept index, its row among the loci extract_loci
    read, its coordinates as the loci file gives them, and whether it is a truth peak.

    The loci are tangermeme's own _interleave_loci over the JSON's loci and chroms, as extract_loci reads them, and the
    row is the kept-mask position (np.flatnonzero(idx)[kept index]). With one loci file and every chromosome listed,
    that row is the file's line, counting from 0.
    """
    import numpy as np
    from tangermeme.io import _interleave_loci

    d = Path(out_dir) / "inputs"
    parameters = json.loads((d / "meta.json").read_text())["parameters"]
    idx = np.load(d / "idx.npy")
    p256 = np.load(d / "p256.npy")
    truth = set(np.load(d / "truth.npy").tolist())
    loci = _interleave_loci(parameters["loci"], parameters["chroms"])
    if len(loci) != len(idx):
        raise RuntimeError(f"extract_loci's mask has {len(idx)} entries but the loci have {len(loci)} rows")
    rows = np.flatnonzero(idx)[p256]
    lines = ["p256\tkept_index\tloci_row\tchrom\tstart\tend\ttruth"]
    for i, (kept, row) in enumerate(zip(p256.tolist(), rows.tolist())):
        locus = loci.iloc[row]
        lines.append(f"{i}\t{kept}\t{row}\t{locus['chrom']}\t{locus['start']}\t{locus['end']}\t{int(i in truth)}")
    path = d / "p256.tsv"
    path.write_text("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------------------------------------------------
# helpers shared by the children


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    tmp.replace(path)


def _run_dir(out_dir: Path, name: str) -> Path:
    path = Path(out_dir) / "runs" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _base_record(kind: str, name: str, spec: dict, args: argparse.Namespace) -> dict:
    return {
        "kind": kind,
        "run": name,
        "spec": spec,
        "args": {k: v if isinstance(v, int | float | str | bool | None) else str(v) for k, v in vars(args).items()},
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "started_et": _now_et(),
        "env": {k: os.environ.get(k) for k in ENV_KEYS},
    }


def _harness_version() -> dict:
    """This file's path, its git commit and whether workflows/attribution differs from it (None outside a git checkout)."""
    out: dict = {"file": str(Path(__file__).resolve()), "commit": None, "dirty": None}
    try:
        git = ["git", "-C", str(HERE)]
        out["commit"] = subprocess.run(
            [*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=30
        ).stdout.strip()
        status = subprocess.run(
            [*git, "status", "--porcelain", "--", "."], capture_output=True, text=True, check=True, timeout=30
        )
        out["dirty"] = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return out


def _versions(torch) -> dict:
    from importlib.metadata import version

    out = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch.version.cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    for package in ("cherimoya", "tangermeme", "numpy", "triton"):
        try:
            out[package] = version(package)
        except Exception as exc:  # recorded, not fatal
            out[package] = f"<{type(exc).__name__}>"
    if torch.cuda.is_available():
        out["gpu"] = torch.cuda.get_device_name(0)
    return out


def _cap_allocator(torch, cap_gb: float, record: dict) -> None:
    """Cap this process's CUDA caching allocator at cap_gb (1e9 bytes): the shared GPUs' budget."""
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, cap_gb * 1e9 / total), 0)
    record["gpu_mem_cap_gb"] = cap_gb


def _gpu_memory(torch) -> dict:
    return {
        "max_allocated_bytes": torch.cuda.max_memory_allocated(),
        "max_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


class Inputs:
    """What `prepare` wrote: the kept peaks as tokens, P256, the truth peaks, P256's references and the metadata."""

    def __init__(self, out_dir: Path) -> None:
        import numpy as np

        d = Path(out_dir) / "inputs"
        self.meta = json.loads((d / "meta.json").read_text())
        self.parameters = self.meta["parameters"]
        self.x_tokens = np.load(d / "x_tokens.npy")  # (n_kept, L) uint8
        self.p256 = np.load(d / "p256.npy")  # indices into the kept peaks
        self.truth = np.load(d / "truth.npy")  # positions within P256
        self.refs = np.load(d / "refs_p256.npy")  # (len(p256), K, L) uint8
        self.start, self.end = self.meta["window"]

    def select(self, peaks: str):
        """(x tokens, reference tokens or None) of a peak set: "p256", "truth" (8 of P256) or "all" (P5000)."""
        if peaks == "p256":
            return self.x_tokens[self.p256], self.refs
        if peaks == "truth":
            return self.x_tokens[self.p256[self.truth]], self.refs[self.truth]
        if peaks == "all":
            return self.x_tokens, None
        raise ValueError(f"unknown peak set {peaks!r}")

    def check(self, peaks: str, X_int8) -> str:
        """Check that the int8 one-hot rebuilt from tokens equals, bit for bit, what extract_loci returned."""
        digest = _sha256(X_int8.numpy().tobytes())
        expected = self.meta["sha256_x_int8"][peaks]
        if digest != expected:
            raise RuntimeError(f"the rebuilt {peaks} one-hot differs from extract_loci's: {digest} != {expected}")
        return digest


def _save_arrays(run_dir: Path, arrays: dict) -> dict:
    """Save each tensor or array as <name>.npy; returns {name: [shape, dtype]}."""
    import numpy as np

    saved = {}
    for name, value in arrays.items():
        arr = value.numpy() if hasattr(value, "numpy") else np.asarray(value)
        np.save(run_dir / f"{name}.npy", arr)
        saved[name] = [list(arr.shape), str(arr.dtype)]
    return saved


def hyp_window(m, X, R, start: int, end: int):
    """Per-peak hypothetical attributions over [start, end), as deep_lift_shap(hypothetical=True) and attribute.py
    compute them, from the raw multipliers on the CPU.

    m (N, K, 4, L) are the multipliers, X (N, 4, L) the one-hot sequences, R (N, K, 4, L) the references. Per peak:
    tangermeme's own hypothetical_attributions over the K pairs, with the sequence repeated per pair as deep_lift_shap's
    ``X[Xi].type(dtype)`` does (TM/deep_lift_shap.py:688, 754); then ``torch.stack`` and ``.mean(dim=0)`` over the K
    pairs (:767-770); then the window (attribute.py:101). The result has m's dtype.
    """
    import torch
    from tangermeme.deep_lift_shap import hypothetical_attributions

    N, K = m.shape[:2]
    out = torch.empty(N, m.shape[2], end - start, dtype=m.dtype)
    for e in range(N):
        x = X[e].to(m.dtype).unsqueeze(0).expand(K, -1, -1).contiguous()
        (h,) = hypothetical_attributions((m[e],), (x,), (R[e].to(m.dtype),))
        out[e] = torch.stack(list(h)).mean(dim=0)[:, start:end]
    return out


@contextlib.contextmanager
def _delta_sink(torch) -> Iterator[tuple[list, list]]:
    """deep_lift_shap's convergence deltas, and when each arrived, through a module-global ``print`` (§14).

    ``print(convergence_deltas)`` is tangermeme.deep_lift_shap's only print (TM/deep_lift_shap.py:749); a module
    global shadows the builtin there. tests/dls_tiny.capture_deltas() is the same sink without the times.
    """
    import importlib

    module = importlib.import_module("tangermeme.deep_lift_shap")
    if "print" in vars(module):
        raise RuntimeError("tangermeme.deep_lift_shap already has a module-global print")
    captured: list = []
    times: list = []

    def sink(*args: object, **kwargs: object) -> None:
        if len(args) == 1 and not kwargs and isinstance(args[0], torch.Tensor):
            captured.append(args[0].detach().clone())
            times.append(time.perf_counter())
        else:
            builtins.print(*args, **kwargs)

    module.print = sink
    try:
        yield captured, times
    finally:
        vars(module).pop("print", None)


def _module_outputs(torch, wrapper, X, R, batch_size: int):
    """E1's reference outputs: the wrapper's own forward on deep_lift_shap's own batches, under torch.enable_grad().

    Each batch is deep_lift_shap's: batch_size pairs in pair order, their sequences then their references, as float32
    tensors that require grad (TM/deep_lift_shap.py:686-733). So every forward has the shape of one of the
    attribution's forwards, and in this process the same cuDNN algorithms and Triton configurations, which are picked
    per shape. A batch of another shape can get another algorithm. Forwarding the rows in plain batches of 40 left the
    run's last 16 rows in a batch of their own, and those rows alone differed from the engine's outputs by up to
    1.2e-3, 2.6 to 2.9 times as much as any row of a 40-row batch (R1, R1').
    Under enable_grad on CUDA, Cherimoya runs its Triton training path (CH/cheri.py:929); nothing is differentiated.

    Returns y_x (N,), the first of each sequence's K copies, y_r (N, K), and the largest difference between a
    sequence's copies.
    """
    wrapper.eval()
    N, K = R.shape[:2]
    y_x = torch.empty(N, K)
    y_r = torch.empty(N, K)
    pairs = torch.arange(N * K)
    with torch.enable_grad():
        for a in range(0, N * K, batch_size):
            idx = pairs[a : a + batch_size]
            Xi, rj = idx // K, idx % K
            _X = X[Xi].float().to("cuda").requires_grad_()
            _R = R[Xi, rj].float().to("cuda").requires_grad_()
            y = wrapper(torch.cat([_X, _R]))[:, 0].detach().cpu()
            y_x[Xi, rj] = y[: len(idx)]
            y_r[Xi, rj] = y[len(idx) :]
            del _X, _R
    spread = float((y_x - y_x[:, :1]).abs().max())
    return y_x[:, 0].clone(), y_r, spread


def _delta_summary(deltas, threshold: float) -> dict:
    """Pairs, how many deltas exceed the threshold (in their own dtype, as tangermeme compares), max and quantiles."""
    import numpy as np

    d = np.asarray(deltas)
    v = d.astype(np.float64).ravel()
    p50, p99, p999 = np.quantile(v, [0.5, 0.99, 0.999])
    above = int((d > threshold).sum())
    return {
        "pairs": int(v.size),
        "threshold": threshold,
        "above": above,
        "fraction": above / v.size,
        "max": float(v.max()),
        "p50": float(p50),
        "p99": float(p99),
        "p99.9": float(p999),
        "finite": bool(np.isfinite(v).all()),
    }


# ---------------------------------------------------------------------------------------------------------------------
# prepare


def cmd_prepare(args: argparse.Namespace) -> int:
    """extract_loci as `cherimoya attribute` does, P256, the truth peaks, P256's references and G0."""
    import dls_refs

    dls_refs.ensure_numba_cache_dir()
    import numpy as np
    import torch
    from cherimoya_cli.defaults import default_attribute_parameters
    from cherimoya_cli.utils import merge_parameters
    from tangermeme.ersatz import dinucleotide_shuffle
    from tangermeme.io import extract_loci

    torch.set_num_threads(args.threads)
    began = time.perf_counter()
    out = Path(args.out_dir) / "inputs"
    out.mkdir(parents=True, exist_ok=True)
    meta: dict = {"started_et": _now_et(), "parameters_file": str(Path(args.parameters).resolve())}

    # attribute.py:21-86, in effect verbatim
    parameters = merge_parameters(args.parameters, default_attribute_parameters)
    if parameters["algorithm"] != "deep_lift_shap" or parameters["output"] != "counts":
        raise SystemExit("this harness validates deep_lift_shap on the counts head")
    X, idxs = extract_loci(
        sequences=parameters["sequences"],
        loci=parameters["loci"],
        chroms=parameters["chroms"],
        in_window=parameters["in_window"],
        max_jitter=0,
        exclusion_lists=parameters["exclusion_lists"],
        ignore=list("QWERYUIOPSDFHJKLZXVBNM"),
        return_mask=True,
        verbose=False,
    )
    n_loci = int(idxs.sum())
    n_idxs = X.sum(dim=(1, 2)) == X.shape[-1]
    X = X[n_idxs]
    idxs[idxs.clone()] = n_idxs
    attr_window = parameters["attr_window"]
    mid = X.shape[-1] // 2
    start = mid - attr_window // 2
    end = start + attr_window
    n, L = len(X), X.shape[-1]
    if args.n_peaks > n:
        raise SystemExit(f"--n-peaks {args.n_peaks} exceeds the {n} kept peaks")
    p256 = np.round(np.linspace(0, n - 1, args.n_peaks)).astype(np.int64)
    truth = np.round(np.linspace(0, args.n_peaks - 1, args.n_truth)).astype(np.int64)
    if len(np.unique(p256)) != len(p256) or len(np.unique(truth)) != len(truth):
        raise SystemExit("the evenly spaced peak indices are not distinct")
    meta.update(
        parameters=parameters,
        n_loci_read=len(idxs),
        n_regions_in_chroms=n_loci,
        n_kept=n,
        length=L,
        window=[start, end],
        p256_rule=f"round(linspace(0, {n - 1}, {args.n_peaks})) over the kept peaks",
        truth_rule=f"round(linspace(0, {args.n_peaks - 1}, {args.n_truth})) over P256",
        p256_first_last=[int(p256[0]), int(p256[-1])],
        truth_positions=truth.tolist(),
        truth_kept_indices=p256[truth].tolist(),
        sha256_x_int8={
            "all": _sha256(X.numpy().tobytes()),
            "p256": _sha256(X[p256].numpy().tobytes()),
            "truth": _sha256(X[p256[truth]].numpy().tobytes()),
        },
        timings_s={"extract_loci": time.perf_counter() - began},
    )
    x_tokens = dls_refs.onehot_to_tokens(X)
    if not torch.equal(dls_refs.tokens_to_onehot(x_tokens, torch.int8), X):
        raise RuntimeError("the tokens of the kept peaks do not round-trip to extract_loci's one-hot")

    # G0: the engine's references (RefProducer, worker processes) against tangermeme's own, pair by pair
    K, rs = int(parameters["n_shuffles"]), int(parameters["random_state"])
    t = time.perf_counter()
    with dls_refs.RefProducer(args.ref_workers) as producer:
        chunks = list(producer.chunks(x_tokens[p256], K, rs, args.chunk_peaks))
        producer_stats = dict(producer.stats)
    refs = np.concatenate([r for _, _, r in chunks])
    meta["timings_s"]["ref_producer"] = time.perf_counter() - t
    t = time.perf_counter()
    equal = np.zeros((len(p256), K), dtype=bool)
    for i, e in enumerate(p256):
        x = X[e : e + 1].float()
        for k in range(K):
            theirs = dinucleotide_shuffle(x, n=1, random_state=rs + k)[:, 0]
            ours = dls_refs.tokens_to_onehot(refs[i, k][None], torch.float32)
            equal[i, k] = torch.equal(ours, theirs)
    meta["timings_s"]["tangermeme_shuffles"] = time.perf_counter() - t
    meta["G0"] = {
        "pairs": int(equal.size),
        "bitwise_equal": int(equal.sum()),
        "pass": bool(equal.all()),
        "first_mismatch": None if equal.all() else [int(v) for v in np.argwhere(~equal)[0]],
        "ref_workers": args.ref_workers,
        "producer_stats": producer_stats,
        "rule": "torch.equal(our reference, dinucleotide_shuffle(X[e:e+1].float(), n=1, random_state=rs + k)[:, 0])",
    }
    np.save(out / "x_tokens.npy", x_tokens)
    np.save(out / "idx.npy", idxs.numpy())
    np.save(out / "p256.npy", p256)
    np.save(out / "truth.npy", truth)
    np.save(out / "refs_p256.npy", refs)
    meta["versions"] = _versions(torch)
    meta["timings_s"]["total"] = time.perf_counter() - began
    meta["finished_et"] = _now_et()
    _write_json(out / "meta.json", meta)
    write_p256_table(args.out_dir)
    logger.info(
        "prepare: %d kept peaks of length %d, window [%d, %d); P256 %d..%d; G0 %d/%d pairs bitwise equal; wrote %s",
        n,
        L,
        start,
        end,
        p256[0],
        p256[-1],
        meta["G0"]["bitwise_equal"],
        meta["G0"]["pairs"],
        out,
    )
    return 0 if meta["G0"]["pass"] else 1


def cmd_checkpoint_json(args: argparse.Namespace) -> int:
    """Write <out>/attribute.json for the checkpoint in --fold-dir (checkpoint_parameters). An existing file is kept
    when it says the same and refused when it differs: runs may already have read it."""
    base = json.loads(Path(args.base).read_text())
    parameters = checkpoint_parameters(base, args.fold_dir, args.out_dir)
    text = json.dumps(parameters, indent=4) + "\n"
    path = Path(args.out_dir) / "attribute.json"
    if path.exists():
        if path.read_text() != text:
            logger.error("%s exists and differs from the JSON for %s; not overwritten", path, args.fold_dir)
            return 1
        logger.info("%s already says the same", path)
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    logger.info("wrote %s: model %s, loci %s", path, parameters["model"], parameters["loci"])
    return 0


def cmd_p256_table(args: argparse.Namespace) -> int:
    """Write <out>/inputs/p256.tsv for a directory prepare has filled."""
    logger.info("wrote %s", write_p256_table(args.out_dir))
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# drivers: one fresh child process per run, one at a time


def _drive(kind: str, args: argparse.Namespace) -> int:
    table = ORACLE_RUNS if kind == "oracle" else OURS_RUNS
    runs = args.runs if args.runs is not None else ",".join(PLANS[args.plan][kind])
    names = [r.strip() for r in runs.split(",") if r.strip()]
    unknown = [r for r in names if r not in table]
    if unknown:
        logger.error("unknown %s runs %s; known: %s", kind, unknown, list(table))
        return 2
    for name in names:
        rc = _spawn(kind, name, args)
        if rc != 0:
            logger.error("%s run %s failed (exit %d); stopping", kind, name, rc)
            return rc
    return 0


def _spawn(kind: str, name: str, args: argparse.Namespace) -> int:
    """Run one child in a fresh process; its log goes to runs/<name>/child.log, the driver's record to driver.json."""
    run_dir = _run_dir(args.out_dir, name)
    cmd = [sys.executable, str(Path(__file__).resolve()), f"_{kind}", "--out-dir", str(args.out_dir), "--run", name]
    cmd += ["--gpu-mem-cap-gb", str(args.gpu_mem_cap_gb)]
    if kind == "oracle":
        cmd += ["--batch-size", str(args.batch_size)]
    else:
        cmd += ["--mem-budget-gb", str(args.mem_budget_gb), "--ref-workers", str(args.ref_workers)]
    env = dict(os.environ)
    env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")  # no Inductor compile pool next to the reference workers
    env["PYTHONUNBUFFERED"] = "1"
    if kind == "oracle":
        env.pop("TRITON_CACHE_AUTOTUNING", None)  # stock runs exactly as users run it: autotune in every process
    for stale in ("record.json", "driver.json"):
        (run_dir / stale).unlink(missing_ok=True)
    record = {"cmd": cmd, "started_et": _now_et(), "co_tenants_before": _compute_apps()}
    logger.info("%s %s: starting %s", kind, name, " ".join(cmd))
    began = time.perf_counter()
    try:
        with open(run_dir / "child.log", "w") as log:
            proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout, check=False)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        rc = -9
        logger.error("%s %s: killed after %d s", kind, name, args.timeout)
    record.update(
        returncode=rc,
        wall_s=time.perf_counter() - began,
        finished_et=_now_et(),
        co_tenants_after=_compute_apps(),
    )
    _write_json(run_dir / "driver.json", record)
    tail = (run_dir / "child.log").read_text(errors="replace").splitlines()[-(25 if rc else 3) :]
    logger.info("%s %s: exit %d after %.1f s\n  %s", kind, name, rc, record["wall_s"], "\n  ".join(tail))
    return rc


# ---------------------------------------------------------------------------------------------------------------------
# children


def child_oracle(args: argparse.Namespace) -> int:
    """One stock run (ORACLE_RUNS[args.run]) in this fresh process."""
    spec = ORACLE_RUNS[args.run]
    run_dir = _run_dir(args.out_dir, args.run)
    record = _base_record("oracle", args.run, spec, args)
    began = time.perf_counter()
    import dls_refs

    dls_refs.ensure_numba_cache_dir()
    import cheri_dls  # precision_flags() only; importing it sets no flag
    import cherimoya  # noqa: F401  (sets the flags `cherimoya attribute` runs under, CH/cherimoya.py:15-16)
    import torch
    from cherimoya import Cherimoya, ControlWrapper, LogCountWrapper
    from cherimoya.deep_lift_shap import attribution_ops
    from tangermeme.deep_lift_shap import deep_lift_shap

    record["precision_flags_imported"] = cheri_dls.precision_flags()
    if not spec["tf32"]:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    record["precision_flags"] = flags = cheri_dls.precision_flags()
    # The CUDA getters decide. After the legacy cuda.matmul.allow_tf32 setter, torch 2.14's global
    # get_float32_matmul_precision() raises instead of answering: `import cherimoya` left the CPU (mkldnn) backend at
    # tf32 through set_float32_matmul_precision('high'), so the backends disagree. precision_flags() records the error.
    keys = (
        "torch.backends.cuda.matmul.allow_tf32",
        "torch.backends.cuda.matmul.fp32_precision",
        "torch.backends.cudnn.allow_tf32",
    )
    actual = tuple(flags[k] for k in keys)
    expected = (True, "tf32", True) if spec["tf32"] else (False, "ieee", False)
    if actual != expected:
        raise RuntimeError(f"precision flags {dict(zip(keys, actual))} are not the ones {args.run} needs, {expected}")
    _cap_allocator(torch, args.gpu_mem_cap_gb, record)
    record["versions"] = _versions(torch)
    record["co_tenants_before"] = _compute_apps()

    inputs = Inputs(args.out_dir)
    p = inputs.parameters
    x_tokens, r_tokens = inputs.select(spec["peaks"])
    X = dls_refs.tokens_to_onehot(x_tokens, torch.int8)  # extract_loci's dtype, as attribute.py passes it
    record["x_sha256"] = inputs.check(spec["peaks"], X)
    R = dls_refs.tokens_to_onehot(r_tokens, torch.float32)
    N, K = R.shape[:2]
    model = Cherimoya.load(p["model"], device=p["device"], compile=False, compile_mode=p["compile_mode"])
    wrapper = LogCountWrapper(ControlWrapper(model), group=p["group"])  # attribute.py:69-75
    record["timings_s"] = {"setup": time.perf_counter() - began}

    logger.info("deep_lift_shap on %d peaks x %d references, batch_size %d", N, K, args.batch_size)
    with warnings.catch_warnings(), _delta_sink(torch) as (captured, arrivals):
        warnings.filterwarnings("ignore", message="Convergence deltas too high", category=RuntimeWarning)
        torch.cuda.synchronize()
        t = time.perf_counter()
        m = deep_lift_shap(
            wrapper,
            X,
            references=R,
            n_shuffles=K,
            batch_size=args.batch_size,
            hypothetical=True,
            raw_outputs=True,
            warning_threshold=p["warning_threshold"],
            additional_nonlinear_ops=attribution_ops(),
            print_convergence_deltas=True,
            dtype=p["dtype"],
            device=p["device"],
            random_state=p["random_state"],
            verbose=False,
        )
        torch.cuda.synchronize()
        record["timings_s"]["deep_lift_shap"] = time.perf_counter() - t
    deltas = torch.cat([d.cpu() for d in captured]).reshape(N, K)
    batches = len(arrivals)
    if batches > 2:  # the first batch carries Triton's autotuning and cuDNN's benchmark; each wait is a sync
        steady_pairs = N * K - args.batch_size
        record["first_batch_s"] = arrivals[0] - t
        record["steady_pairs_per_s"] = steady_pairs / (arrivals[-1] - arrivals[0])
    record["batches"] = batches
    record["gpu_memory_attribution"] = _gpu_memory(torch)

    t = time.perf_counter()
    hyp = hyp_window(m, X, R, inputs.start, inputs.end).float()
    record["timings_s"]["hyp_cpu"] = time.perf_counter() - t
    t = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    y_x, y_r, record["module_y_x_copy_spread"] = _module_outputs(torch, wrapper, X, R, args.batch_size)
    record["timings_s"]["module_forward"] = time.perf_counter() - t
    record["gpu_memory_forward"] = _gpu_memory(torch)

    record["deltas"] = _delta_summary(deltas.numpy(), float(p["warning_threshold"]))
    record["arrays"] = _save_arrays(run_dir, {"m": m, "hyp": hyp, "deltas": deltas, "y_x": y_x, "y_r": y_r})
    record["co_tenants_after"] = _compute_apps()
    record["timings_s"]["total"] = time.perf_counter() - began
    record["finished_et"] = _now_et()
    _write_json(run_dir / "record.json", record)
    logger.info(
        "%s: %d pairs in %.1f s (steady %s pairs/s), %d of %d deltas > %g (max %.4g), GPU max allocated %.2f GB",
        args.run,
        N * K,
        record["timings_s"]["deep_lift_shap"],
        f"{record.get('steady_pairs_per_s', float('nan')):.0f}",
        record["deltas"]["above"],
        record["deltas"]["pairs"],
        record["deltas"]["threshold"],
        record["deltas"]["max"],
        record["gpu_memory_attribution"]["max_allocated_bytes"] / 1e9,
    )
    return 0


def child_ours(args: argparse.Namespace) -> int:
    """One cheri_dls run (OURS_RUNS[args.run]) in this fresh process."""
    spec = OURS_RUNS[args.run]
    run_dir = _run_dir(args.out_dir, args.run)
    record = _base_record("ours", args.run, spec, args)
    began = time.perf_counter()
    import dls_refs

    dls_refs.ensure_numba_cache_dir()
    import cheri_dls
    import torch

    _cap_allocator(torch, args.gpu_mem_cap_gb, record)
    record["versions"] = _versions(torch)
    record["co_tenants_before"] = _compute_apps()
    inputs = Inputs(args.out_dir)
    p = inputs.parameters
    x_tokens, r_tokens = inputs.select(spec["peaks"])
    X = dls_refs.tokens_to_onehot(x_tokens, torch.int8)
    record["x_sha256"] = inputs.check(spec["peaks"], X)
    R = None if r_tokens is None else dls_refs.tokens_to_onehot(r_tokens, torch.int8)
    model = cheri_dls.load_model(p["model"])
    engine = cheri_dls.Engine(
        model,
        output="counts",
        group=p["group"],
        device="cuda",
        precision=spec["precision"],
        n_shuffles=p["n_shuffles"],
        random_state=p["random_state"],
        head_rows=2 * p["batch_size"],
        seqs_per_step=spec["S"],
        mem_budget_gb=args.mem_budget_gb,
        compile=spec["compile"],
        warning_threshold=p["warning_threshold"],
        ref_workers=args.ref_workers,
    )
    record["timings_s"] = {"setup": time.perf_counter() - began}
    arrays: dict = {}
    if R is not None:
        t = time.perf_counter()
        raw = engine.raw(X, R)
        record["timings_s"]["raw"] = time.perf_counter() - t
        record["raw"] = raw.debug
        arrays.update(m=raw.m, deltas=raw.deltas, y_x=raw.y_x, y_r=raw.y_r)
    t = time.perf_counter()
    result = engine.run(X, inputs.start, inputs.end, references=R)
    record["timings_s"]["run"] = time.perf_counter() - t
    record["meta"] = result.meta
    arrays["hyp"] = result.attr.float()
    if R is None:
        arrays.update(deltas=result.deltas, y_x=result.y_x, y_r=result.y_r)
    else:
        arrays.update(deltas_run=result.deltas, y_x_run=result.y_x, y_r_run=result.y_r)
        record["run_vs_raw"] = {
            "y_bitwise_equal": bool(torch.equal(raw.y_x, result.y_x) and torch.equal(raw.y_r, result.y_r)),
            "y_max_abs_diff": float(max((raw.y_x - result.y_x).abs().max(), (raw.y_r - result.y_r).abs().max()).item()),
            "deltas_max_abs_diff": float((raw.deltas - result.deltas).abs().max()),
        }
    record["gpu_memory"] = _gpu_memory(torch)
    record["deltas"] = _delta_summary(arrays["deltas"].numpy(), float(p["warning_threshold"]))
    record["arrays"] = _save_arrays(run_dir, arrays)
    record["co_tenants_after"] = _compute_apps()
    record["timings_s"]["total"] = time.perf_counter() - began
    record["finished_et"] = _now_et()
    _write_json(run_dir / "record.json", record)
    meta = result.meta
    logger.info(
        "%s: %d pairs, run() %.1f s (steady %s pairs/s, S=%d, compile %s s), %d of %d deltas > %g (max %.4g), "
        "process GPU max allocated %.2f GB",
        args.run,
        meta["pairs"],
        record["timings_s"]["run"],
        "-" if meta["steady_pairs_per_s"] is None else f"{meta['steady_pairs_per_s']:.0f}",
        meta["seqs_per_step"],
        "-" if meta["compile_s"] is None else f"{meta['compile_s']:.1f}",
        record["deltas"]["above"],
        record["deltas"]["pairs"],
        record["deltas"]["threshold"],
        record["deltas"]["max"],
        record["gpu_memory"]["max_allocated_bytes"] / 1e9,
    )
    return 0


def cmd_truth(args: argparse.Namespace) -> int:
    """T64: the engine on the CPU in float64, two-pass statistics, on the 8 truth peaks."""
    run_dir = _run_dir(args.out_dir, TRUTH_RUN)
    spec = {"precision": "float64", "device": "cpu", "forward_stats": "two_pass", "compile": "none", "peaks": "truth"}
    record = _base_record("truth", TRUTH_RUN, spec, args)
    began = time.perf_counter()
    import dls_refs

    dls_refs.ensure_numba_cache_dir()
    import cheri_dls
    import torch

    sys.path.insert(0, str(REPO / "tests"))
    import dls_tiny  # enable_float64: the stock forward of a float64 model (see its docstring)

    torch.set_num_threads(args.threads)
    record["versions"] = _versions(torch)
    inputs = Inputs(args.out_dir)
    p = inputs.parameters
    x_tokens, r_tokens = inputs.select("truth")
    X = dls_refs.tokens_to_onehot(x_tokens, torch.int8)
    record["x_sha256"] = inputs.check("truth", X)
    R = dls_refs.tokens_to_onehot(r_tokens, torch.int8)
    model = dls_tiny.enable_float64(cheri_dls.load_model(p["model"]).double())
    engine = cheri_dls.Engine(
        model,
        output="counts",
        group=p["group"],
        device="cpu",
        precision="fp32",  # no TF32 anywhere; the arithmetic is float64 in any case
        n_shuffles=p["n_shuffles"],
        random_state=p["random_state"],
        head_rows=2 * p["batch_size"],
        seqs_per_step=args.seqs_per_step,
        compile="none",
        forward_stats="two_pass",
        warning_threshold=p["warning_threshold"],
    )
    t = time.perf_counter()
    raw = engine.raw(X, R)
    record["timings_s"] = {"raw": time.perf_counter() - t}
    record["raw"] = raw.debug
    if raw.m.dtype != torch.float64:
        raise RuntimeError(f"the truth run is not float64: {raw.m.dtype}")
    t = time.perf_counter()
    hyp = hyp_window(raw.m, X, R, inputs.start, inputs.end)
    record["timings_s"]["hyp_cpu"] = time.perf_counter() - t
    arrays = {"m": raw.m, "hyp": hyp, "deltas": raw.deltas, "y_x": raw.y_x, "y_r": raw.y_r}
    record["deltas"] = _delta_summary(raw.deltas.numpy(), float(p["warning_threshold"]))
    record["arrays"] = _save_arrays(run_dir, arrays)
    record["timings_s"]["total"] = time.perf_counter() - began
    record["finished_et"] = _now_et()
    _write_json(run_dir / "record.json", record)
    logger.info(
        "T64: %d pairs in %.1f s, deltas max %.3g (conservation in float64)",
        raw.deltas.numel(),
        record["timings_s"]["raw"],
        record["deltas"]["max"],
    )
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# gates


class Runs:
    """The saved runs under <out>/runs, read lazily and memory-mapped."""

    def __init__(self, out_dir: Path, truth_positions) -> None:
        self.root = Path(out_dir) / "runs"
        self.truth = truth_positions
        self._records: dict = {}
        self._comparisons: dict = {}

    def has(self, name: str) -> bool:
        return (self.root / name / "record.json").exists()

    def record(self, name: str) -> dict:
        if name not in self._records:
            self._records[name] = json.loads((self.root / name / "record.json").read_text())
        return self._records[name]

    def array(self, name: str, key: str, peaks: str = "own"):
        """A saved array; peaks="truth" takes the truth peaks out of a P256 run (rows never mix, so this is exact)."""
        import numpy as np

        arr = np.load(self.root / name / f"{key}.npy", mmap_mode="r")
        if peaks == "truth" and self.record(name)["spec"].get("peaks") == "p256":
            return np.asarray(arr[self.truth])
        return arr

    def compare(self, a: str, b: str, peaks: str = "own") -> dict:
        """a vs b: per-pair raw rel-L2 (4 x L), per-peak attribution rel-L2 (4 x W) and Pearson, relative to b."""
        import numpy as np

        key = (a, b, peaks)
        if key not in self._comparisons:
            raw = rel_l2(self.array(a, "m", peaks), self.array(b, "m", peaks), lead=2)
            ha, hb = self.array(a, "hyp", peaks), self.array(b, "hyp", peaks)
            hyp = rel_l2(ha, hb, lead=1)
            r = pearson_rows(ha, hb)
            self._comparisons[key] = {
                "a": a,
                "b": b,
                "peaks": "truth" if peaks == "truth" else "p256",
                "raw": stats(raw),
                "hyp": stats(hyp),
                "pearson": {"min": float(r.min()), "q50": float(np.median(r))},
                "_values": {"raw": raw, "hyp": hyp, "pearson": r},
            }
        return self._comparisons[key]


def _row(gate, comparison, metric, statistic, measured, bound, rule, op="<=", note="") -> dict:
    if measured is None or bound is None:
        ok = None
    elif op == "<=":
        ok = bool(measured <= bound)
    elif op == ">=":
        ok = bool(measured >= bound)
    elif op == "==":
        ok = bool(measured == bound)
    else:
        raise ValueError(op)
    return {
        "gate": gate,
        "comparison": comparison,
        "metric": metric,
        "statistic": statistic,
        "measured": measured,
        "op": op,
        "bound": bound,
        "rule": rule,
        "pass": ok,
        "note": note,
    }


def _public(comparison: dict) -> dict:
    return {k: v for k, v in comparison.items() if not k.startswith("_")}


def _floor_rows(gate: str, cmp: dict, floor: dict, floor_name: str, tf: dict | None) -> list[dict]:
    """E3-style limits for one comparison: median, p99 and (with tf) the precision-policy check, on both metrics."""
    rows = []
    label = f"{cmp['a']} vs {cmp['b']}"
    for metric in ("raw", "hyp"):
        bounds = floor_bounds(floor[metric])
        rows.append(
            _row(gate, label, metric, "q50", cmp[metric]["q50"], bounds["q50"], f"max(2*{floor_name}(0.5), 1e-5)")
        )
        rows.append(
            _row(gate, label, metric, "q99", cmp[metric]["q99"], bounds["q99"], f"max(3*{floor_name}(0.99), 1e-4)")
        )
        if tf is not None:
            rows.append(
                _row(
                    gate,
                    label,
                    metric,
                    "q50 policy",
                    cmp[metric]["q50"],
                    POLICY_FACTOR * tf[metric]["q50"],
                    "0.3*F_tf(0.5)",
                )
            )
    return rows


def _finite(runs: Runs, name: str) -> int:
    """Non-finite values among a run's saved outputs."""
    import numpy as np

    bad = 0
    for key in ("m", "hyp", "deltas", "y_x", "y_r"):
        path = runs.root / name / f"{key}.npy"
        if path.exists():
            arr = np.load(path, mmap_mode="r")
            bad += int(arr.size - sum(int(np.isfinite(arr[i : i + 16]).sum()) for i in range(0, len(arr), 16)))
    return bad


def _cap_rows(runs: Runs, a: str, b: str) -> list[dict]:
    cmp = runs.compare(a, b)
    label = f"{a} vs {b}"
    return [
        _row("E5", label, "hyp", "min Pearson", cmp["pearson"]["min"], PEARSON_MIN, "every peak >= 0.9999", ">="),
        _row("E5", label, "hyp", "max rel-L2", cmp["hyp"]["max"], HYP_REL_MAX, "every peak <= 1e-2"),
    ]


def _steps_row(runs: Runs, name: str) -> dict:
    """E10's check that `name` ran at its specified S in raw() (the multipliers) and run() (the attributions).

    An out-of-memory halving lowers the S a call records (cheri_dls.Engine._after_oom), so the smaller of the two must
    equal the spec's. Otherwise E10 would compare a run with one at its own step size.
    """
    rec = runs.record(name)
    spec = OURS_RUNS[name]["S"]
    calls = [d for d in (rec.get("raw"), rec.get("meta")) if d]
    used = [d.get("seqs_per_step") for d in calls]
    measured = min(used) if used and None not in used else None
    halvings = sum(len(d.get("oom") or []) for d in calls)
    return _row(
        "E10",
        f"{name} steps as run",
        "seqs_per_step",
        "S in raw() and run()",
        measured,
        spec,
        f"S = {spec} as specified, no out-of-memory halving",
        "==",
        note=f"{halvings} halvings over {len(calls)} calls",
    )


def _outputs(runs: Runs, name: str, peaks: str = "own"):
    """A run's attributed outputs, the sequences' then the references', as one float64 vector."""
    import numpy as np

    return np.concatenate(
        [
            np.asarray(runs.array(name, "y_x", peaks), np.float64),
            np.asarray(runs.array(name, "y_r", peaks), np.float64).ravel(),
        ]
    )


def _gate_pass(rows: list[dict]) -> bool | None:
    """A gate's verdict: None when it has no rows or a row could not be evaluated (a run it needs is missing)."""
    verdicts = [r["pass"] for r in rows]
    return None if not verdicts or None in verdicts else all(verdicts)


def cmd_gates(args: argparse.Namespace) -> int:
    """Floors, gates and context from the saved runs; writes report.json, report.tsv and report.md."""
    import numpy as np

    out_dir = Path(args.out_dir)
    inputs_meta = json.loads((out_dir / "inputs" / "meta.json").read_text())
    truth = np.load(out_dir / "inputs" / "truth.npy")
    runs = Runs(out_dir, truth)
    threshold = float(inputs_meta["parameters"]["warning_threshold"])
    report: dict = {"generated_et": _now_et(), "out_dir": str(out_dir), "head": "counts"}
    report["harness"] = _harness_version()
    report["inputs"] = {
        k: inputs_meta[k]
        for k in ("n_kept", "length", "window", "p256_rule", "truth_rule", "truth_kept_indices", "G0", "versions")
    }
    report["inputs"]["parameters_file"] = inputs_meta.get("parameters_file")
    report["inputs"]["model"] = inputs_meta["parameters"].get("model")
    report["inputs"]["loci"] = inputs_meta["parameters"].get("loci")
    plan = PLANS[args.plan]
    planned = plan_runs(args.plan)
    report["plan"] = {
        "name": args.plan,
        "about": plan["about"],
        "runs": planned,
        "gates": list(plan["gates"]),
        "out_of_scope": [g for g in GATES if g not in plan["gates"]],
    }
    missing = [n for n in planned if not runs.has(n)]
    report["missing_runs"] = missing
    present = [n for n in [*ORACLE_RUNS, *OURS_RUNS, TRUTH_RUN] if runs.has(n)]
    report["runs"] = {n: _run_summary(runs, n) for n in present}
    gates: dict = {}
    rows: list[dict] = []

    def need(*names: str) -> bool:
        return all(runs.has(n) for n in names)

    # G0
    g0 = inputs_meta["G0"]
    gates["G0"] = [
        _row(
            "G0",
            "RefProducer vs tangermeme",
            "references",
            "bitwise equal pairs",
            g0["bitwise_equal"],
            g0["pairs"],
            "all P256 x K pairs torch.equal",
            "==",
        ),
    ]

    # E2 floors
    floors: dict = {}
    for name, a, b in (("F_rr", "R1p", "R1"), ("F_hh", "R2p", "R2"), ("F_tf", "R2", "R1")):
        if need(a, b):
            floors[name] = _public(runs.compare(a, b))
    report["floors"] = floors
    gates["E2"] = [
        _row("E2", f"{f['a']} vs {f['b']}", metric, stat, f[metric][stat], None, f"floor {name}")
        for name, f in floors.items()
        for metric in ("raw", "hyp")
        for stat in ("q50", "q99", "max")
    ]

    # Forward outputs: every run's against the others' and against the fp64 truth on the 8 truth peaks (168 rows)
    context_rows = []
    original_rows = []  # the original bounds of revised criteria: computed and reported, never gating
    revisions: dict = {"document": REVISIONS_DOC}
    forward_noise = {}
    for a, b in (("R1p", "R1"), ("R2p", "R2"), ("R2", "R1"), ("T1p", "T1"), ("T2", "R1"), ("T1", "R1p"), ("T2", "R2p")):
        if need(a, b):
            diff = float(np.abs(_outputs(runs, a) - _outputs(runs, b)).max())
            forward_noise[f"{a} vs {b}"] = diff
            context_rows.append(_row("context E1", f"{a} y vs {b} y", "counts y", "max abs", diff, None, "reported"))
    forward_to_truth = {}
    if runs.has(TRUTH_RUN):
        y_truth = _outputs(runs, TRUTH_RUN)
        for name in ("R1", "R1p", "R2", "R2p", "R1_8", "T1", "T1p", "T2", "T3", "T1_S8", "T2_S8"):
            if runs.has(name):
                diff = float(np.abs(_outputs(runs, name, "truth") - y_truth).max())
                forward_to_truth[name] = diff
                context_rows.append(
                    _row(
                        "context E1", f"{name} y vs T64 y (truth peaks)", "counts y", "max abs", diff, None, "reported"
                    )
                )
    forward_accuracy = {}
    for ours, ref in (("T1", "R1"), ("T3", "R1"), ("T2", "R2")):
        if ours in forward_to_truth and ref in forward_to_truth:
            d_ours, d_ref = forward_to_truth[ours], forward_to_truth[ref]
            forward_accuracy[f"{ours} vs {ref}"] = {
                "ours_to_T64": d_ours,
                "reference_to_T64": d_ref,
                "within_1.2x": bool(d_ours <= ACCURACY_FACTOR * d_ref),
            }

    # E1 forward parity: our outputs against the module's own forward under enable_grad, per mode. Reference mode as
    # revised (design/revisions.md): within 2x the stock forward's own rerun noise on the same rows, and no farther
    # from the fp64 truth than 1.2x the stock run. The original 2e-4 is reported only. fp32 mode keeps 5e-5.
    e1 = []
    floor_y = forward_noise.get("R1p vs R1")
    e1_revision: dict = {
        "original": {"rule": f"max |y_ours - y_R1| <= {E1_MAX['reference']:g}", "bound": E1_MAX["reference"]},
        "revised": {
            "rule": "max |y_ours - y_R1| <= 2 x max |y_R1' - y_R1| (the stock forward's rerun noise on the same "
            "rows), and max |y_ours - y_T64| <= 1.2 x max |y_R1 - y_T64| on the 8 truth peaks",
            "floor_R1p_vs_R1": floor_y,
            "floor_bound": None if floor_y is None else E1_FLOOR_FACTOR * floor_y,
            "R1_to_T64": forward_to_truth.get("R1"),
            "accuracy_bound": None if "R1" not in forward_to_truth else ACCURACY_FACTOR * forward_to_truth["R1"],
        },
        "runs": {},
    }
    for ours, mode in (("T1", "reference"), ("T3", "reference"), ("T2", "fp32")):
        ref = REFERENCE_OF[mode]
        if not need(ours, ref):
            continue
        diff = float(np.abs(_outputs(runs, ours) - _outputs(runs, ref)).max())
        label = f"{ours} y vs {ref} module y"
        if mode == "fp32":
            e1.append(_row("E1", label, "counts y", "max abs", diff, E1_MAX[mode], f"{mode}: <= {E1_MAX[mode]:g}"))
            continue
        original = _row(
            "E1 original (reported only)",
            label,
            "counts y",
            "max abs",
            diff,
            E1_MAX[mode],
            f"{mode}: <= {E1_MAX[mode]:g}; superseded by the revised E1",
        )
        original_rows.append(original)
        floor_row = _row(
            "E1",
            label,
            "counts y",
            "max abs",
            diff,
            e1_revision["revised"]["floor_bound"],
            "reference: <= 2 x max|y_R1' - y_R1|",
        )
        accuracy_row = _row(
            "E1",
            f"{ours} y vs T64 y (truth peaks), against 1.2 x R1's",
            "counts y",
            "max abs to T64",
            forward_to_truth.get(ours),
            e1_revision["revised"]["accuracy_bound"],
            "reference: <= 1.2 x max|y_R1 - y_T64|",
        )
        e1 += [floor_row, accuracy_row]
        e1_revision["runs"][ours] = {
            "max_abs_vs_R1": diff,
            "max_abs_to_T64": accuracy_row["measured"],
            "original_pass": original["pass"],
            "floor_pass": floor_row["pass"],
            "accuracy_pass": accuracy_row["pass"],
        }
    gates["E1"] = e1
    revisions["E1_reference"] = e1_revision
    report["E1_context"] = {
        "max_abs_between_runs": forward_noise,
        "max_abs_to_T64_truth_peaks": forward_to_truth,
        "accuracy": forward_accuracy,
        "module_y_x_copy_spread": {n: runs.record(n).get("module_y_x_copy_spread") for n in ORACLE_RUNS if runs.has(n)},
    }

    # E3, E4, E6, with E7(a) as the tie-breaker of E3 and E4
    truth_ready = runs.has(TRUTH_RUN)
    e7: dict = {}

    def tie_breaker(ours: str, ref: str) -> dict | None:
        if not (truth_ready and need(ours, ref)):
            return None
        d_ours = runs.compare(ours, TRUTH_RUN, "truth")
        d_ref = runs.compare(ref, TRUTH_RUN, "truth")
        result = {"ours": ours, "reference": ref, "metrics": {}}
        for metric in ("raw", "hyp"):
            verdict = accuracy_limited(d_ours[metric], d_ref[metric])
            result["metrics"][metric] = {
                "d_ours": d_ours[metric],
                "d_ref": d_ref[metric],
                "pass": verdict,
            }
        return result

    for gate, ours, ref, floor_name, with_policy in (
        ("E3", "T1", "R1", "F_rr", True),
        ("E4", "T2", "R2", "F_hh", False),
    ):
        if not (need(ours, ref) and floor_name in floors):
            gates[gate] = []
            continue
        cmp = runs.compare(ours, ref)
        gate_rows = _floor_rows(gate, cmp, floors[floor_name], floor_name, floors.get("F_tf") if with_policy else None)
        gates[gate] = gate_rows
        tb = tie_breaker(ours, ref)
        if tb is not None:
            e7[gate] = tb
    gates["E6"] = (
        _floor_rows("E6", runs.compare("T3", "T1"), floors["F_rr"], "F_rr", floors.get("F_tf"))
        if need("T3", "T1") and "F_rr" in floors
        else []
    )

    # E7: applied to a failing E3/E4 on the metrics that failed
    e7_rows = []
    verdicts: dict = {}
    for gate in ("E3", "E4"):
        direct = _gate_pass(gates.get(gate, []))
        verdicts[gate] = {"direct": direct, "via_E7a": None}
        if direct is False:
            failed_metrics = sorted({r["metric"] for r in gates[gate] if r["pass"] is False})
            tb = e7.get(gate)
            if tb is None:
                verdicts[gate]["via_E7a"] = False
                continue
            ok = True
            for metric in failed_metrics:
                for stat in ("q50", "max"):
                    d_o, d_r = tb["metrics"][metric]["d_ours"][stat], tb["metrics"][metric]["d_ref"][stat]
                    row = _row(
                        "E7a",
                        f"d({tb['ours']},T64) vs 1.2 d({tb['reference']},T64) [for {gate}]",
                        metric,
                        stat,
                        d_o,
                        ACCURACY_FACTOR * d_r,
                        "d(ours,T64) <= 1.2 d(R,T64)",
                    )
                    e7_rows.append(row)
                    ok = ok and bool(row["pass"])
            verdicts[gate]["via_E7a"] = ok
    gates["E7"] = e7_rows
    report["E7_computed"] = e7
    report["E7_context"] = e7_context(e7)
    report["E3_E4_verdicts"] = verdicts
    report["E3_policy"] = e3_policy(floors, gates.get("E3", []))

    # E4's side report: T2 vs R1 must meet only E5
    if need("T2", "R1"):
        report["T2_vs_R1"] = _public(runs.compare("T2", "R1"))

    # E5 caps on every run of ours
    e5 = []
    for ours in ("T1", "T1p", "T3", "T1_S8"):
        if need(ours, "R1"):
            e5 += _cap_rows(runs, ours, "R1")
    for ours in ("T2", "T2_S8"):
        for ref in ("R2", "R1"):
            if need(ours, ref):
                e5 += _cap_rows(runs, ours, ref)
    for name in [n for n in OURS_RUNS if runs.has(n)]:
        e5.append(
            _row("E5", name, "all outputs", "non-finite values", _finite(runs, name), 0, "every output finite", "==")
        )
    gates["E5"] = e5

    # E8: our P5000 deltas against the pilot's stock run (the pilot model's numbers: only a plan with E8 reads them)
    e8 = []
    e8_in_scope = "E8" in plan["gates"]
    pilot_bounds = e8_bounds(PILOT["above"], PILOT["pairs"], PILOT["max"])
    report["E8_reference"] = (
        {
            "pilot": PILOT,
            "pilot_fraction": PILOT["above"] / PILOT["pairs"],
            "bounds": pilot_bounds,
            "stock_p5000_run": args.stock_deltas or None,
        }
        if e8_in_scope
        else None
    )
    if e8_in_scope and runs.has("P5000_ref"):
        s = runs.record("P5000_ref")["deltas"]
        e8.append(
            _row(
                "E8",
                "P5000_ref deltas",
                "deltas",
                f"fraction > {threshold:g}",
                s["fraction"],
                pilot_bounds["fraction"],
                "max(2 x pilot 153/99,980, 0.3%)",
            )
        )
        e8.append(
            _row(
                "E8",
                "P5000_ref deltas",
                "deltas",
                "max",
                s["max"],
                pilot_bounds["max"],
                "max(2 x pilot 1.9636e-3, 4e-3)",
            )
        )
        if args.stock_deltas:
            stock = np.load(args.stock_deltas)
            st = _delta_summary(stock, threshold)
            b = e8_bounds(st["above"], st["pairs"], st["max"])
            e8.append(
                _row(
                    "E8",
                    "P5000_ref vs stock P5000",
                    "deltas",
                    "fraction",
                    s["fraction"],
                    b["fraction"],
                    "max(2 x stock, 0.3%)",
                )
            )
            e8.append(
                _row("E8", "P5000_ref vs stock P5000", "deltas", "max", s["max"], b["max"], "max(2 x stock max, 4e-3)")
            )
    if runs.has("P5000_fp32"):
        s = runs.record("P5000_fp32")["deltas"]
        context_rows.append(
            _row(
                "E8 fp32 (prediction, not a gate)",
                "P5000_fp32 deltas",
                "deltas",
                "max",
                s["max"],
                E8_FP32_PREDICTION,
                "predicted <= 1e-4",
            )
        )
    gates["E8"] = e8

    # E10: step-size invariance, S = 4 against S = 8
    e10 = []
    if need("T1_S8", "T1") and "F_rr" in floors:
        e10 += [_steps_row(runs, "T1"), _steps_row(runs, "T1_S8")]
        cmp = runs.compare("T1_S8", "T1")
        for metric in ("raw", "hyp"):
            e10.append(
                _row(
                    "E10",
                    "T1_S8 vs T1 (reference)",
                    metric,
                    "q50",
                    cmp[metric]["q50"],
                    floor_bounds(floors["F_rr"][metric])["q50"],
                    "max(2*F_rr(0.5), 1e-5)",
                )
            )
    if need("T2_S8", "T2"):
        # fp32 mode as revised (design/revisions.md): E4's form against F_hh, the stock's own strict-fp32 rerun, on
        # the per-pair raw rel-L2. The original p99 <= 1e-5 is reported only; hyp is reported in the same form.
        e10 += [_steps_row(runs, "T2"), _steps_row(runs, "T2_S8")]
        cmp = runs.compare("T2_S8", "T2")
        label = "T2_S8 vs T2 (fp32)"
        original = _row(
            "E10 original (reported only)",
            label,
            "raw",
            "q99",
            cmp["raw"]["q99"],
            E10_FP32_P99_MAX,
            "per-pair p99 <= 1e-5; superseded by the revised E10",
        )
        original_rows.append(original)
        hh = floors.get("F_hh")
        revised = {}
        for metric in ("raw", "hyp"):
            bounds = floor_bounds(hh[metric]) if hh else {"q50": None, "q99": None}
            for stat, rule in (("q50", "max(2*F_hh(0.5), 1e-5)"), ("q99", "max(3*F_hh(0.99), 1e-4)")):
                row = _row("E10", label, metric, stat, cmp[metric][stat], bounds[stat], rule)
                if metric == "raw":
                    e10.append(row)
                    revised[stat] = {"measured": row["measured"], "bound": row["bound"], "pass": row["pass"]}
                else:
                    context_rows.append(dict(row, gate="context E10", rule=f"{rule}; reported, not gated"))
        revisions["E10_fp32"] = {
            "original": {
                "rule": "per-pair raw rel-L2 p99 <= 1e-5",
                "bound": E10_FP32_P99_MAX,
                "measured": original["measured"],
                "pass": original["pass"],
            },
            "revised": {
                "rule": "per-pair raw rel-L2 p99 <= max(3 F_hh(0.99), 1e-4) and median <= max(2 F_hh(0.5), 1e-5)",
                "F_hh_raw": None if hh is None else {k: hh["raw"][k] for k in ("q50", "q99", "max")},
                "checks": revised,
            },
            "measured_raw": {k: cmp["raw"][k] for k in ("q50", "q99", "max")},
        }
        if truth_ready:
            for metric in ("raw", "hyp"):
                d_s8 = runs.compare("T2_S8", TRUTH_RUN, "truth")[metric]
                d_s4 = runs.compare("T2", TRUTH_RUN, "truth")[metric]
                for stat in ("q50", "max"):
                    context_rows.append(
                        _row(
                            "context E10 accuracy",
                            "d(T2_S8,T64) vs 1.2 d(T2,T64)",
                            metric,
                            stat,
                            d_s8[stat],
                            ACCURACY_FACTOR * d_s4[stat],
                            "E7(a)'s form; not a tie-breaker for E10",
                        )
                    )
    gates["E10"] = e10

    # context: our own rerun, the literal R run on the truth peaks, distances to T64, deltas per run
    comparisons = {}
    for a, b in (("T1p", "T1"), ("T2", "R1"), ("T3", "T1"), ("T1", "R1p"), ("T2", "R2p")):
        if need(a, b):
            comparisons[f"{a} vs {b}"] = _public(runs.compare(a, b))
    if need("R1_8", "R1"):
        comparisons["R1_8 vs R1 (truth peaks)"] = _public(runs.compare("R1_8", "R1", "truth"))
    to_truth = {}
    if truth_ready:
        for name in [n for n in [*ORACLE_RUNS, *OURS_RUNS] if runs.has(n) and runs.record(n)["spec"]["peaks"] != "all"]:
            to_truth[name] = _public(runs.compare(name, TRUTH_RUN, "truth"))
    report["comparisons"] = comparisons
    report["distance_to_T64"] = to_truth
    report["probes"] = _probe_summary(args.probe_json) if args.probe_json else None
    report["deltas"] = {n: runs.record(n).get("deltas") for n in present}
    for name, cmp in comparisons.items():
        for metric in ("raw", "hyp"):
            for stat in ("q50", "q99", "max"):
                context_rows.append(_row("context", name, metric, stat, cmp[metric][stat], None, "reported"))
    for name, cmp in to_truth.items():
        for metric in ("raw", "hyp"):
            for stat in ("q50", "max"):
                context_rows.append(
                    _row("context d(.,T64)", f"{name} vs T64", metric, stat, cmp[metric][stat], None, "reported")
                )

    # verdicts: the plan's gates only. A gate that reads a planned run that is missing is not run (None).
    summary = {}
    not_run = {}
    for gate in plan["gates"]:
        gate_rows = gates.get(gate, [])
        absent = [n for n in GATE_RUNS[gate] if n in missing]
        if absent:
            verdict = None
            not_run[gate] = absent
        elif gate == "E2":
            verdict = len(floors) == 3
        elif gate in ("E3", "E4"):
            v = verdicts.get(gate, {})
            verdict = v.get("direct") if v.get("direct") is not False else (v.get("via_E7a") and "via E7(a)")
        else:
            verdict = _gate_pass(gate_rows)
        summary[gate] = verdict
        rows += gate_rows
    rows += gates.get("E7", [])
    rows += original_rows
    rows += context_rows
    report["summary"] = summary
    report["not_run"] = not_run
    report["gates"] = gates
    report["revisions"] = revisions
    report["original_bounds"] = original_rows
    report["all_gates_pass"] = all(v not in (None, False) for v in summary.values())
    report["deviations"] = _deviations(runs, report)
    report["open_issues"] = _open_issues(runs, summary, report)

    _write_json(out_dir / "report.json", report)
    _write_tsv(out_dir / "report.tsv", rows)
    (out_dir / "report.md").write_text(_markdown(report, rows))
    for gate, verdict in summary.items():
        logger.info("%s: %s", gate, verdict)
    logger.info("wrote %s, report.tsv and report.md", out_dir / "report.json")
    return 0 if report["all_gates_pass"] else 1


def _probe_summary(path: str) -> dict:
    """The cuDNN classes probe P3 found per convolution and row count (dls_probe.py's dls_probes.json)."""
    rep = json.loads(Path(path).read_text())
    p3 = rep.get("conclusions", {}).get("P3", {})
    return {
        "file": str(path),
        "reference_rows": p3.get("reference_rows"),
        "n_runs": p3.get("n_runs"),
        "heads": p3.get("heads"),
        "ops": {
            op: {
                "state": v.get("state"),
                "class_counts_by_rows": v.get("class_counts_by_rows"),
                "same_bits_across_runs_by_rows": v.get("same_bits_across_runs_by_rows"),
                "deterministic_in_process_by_rows": v.get("deterministic_in_process_by_rows"),
            }
            for op, v in p3.get("ops", {}).items()
        },
    }


def _run_summary(runs: Runs, name: str) -> dict:
    rec = runs.record(name)
    out = {
        "kind": rec["kind"],
        "spec": rec["spec"],
        "started_et": rec.get("started_et"),
        "finished_et": rec.get("finished_et"),
        "total_s": rec.get("timings_s", {}).get("total"),
        "gpu_mem_cap_gb": rec.get("gpu_mem_cap_gb"),
        "cuda_visible_devices": (rec.get("env") or {}).get("CUDA_VISIBLE_DEVICES"),
        "deltas": rec.get("deltas"),
    }
    if rec["kind"] == "oracle":
        out.update(
            batch_size=int(rec["args"]["batch_size"]),
            deep_lift_shap_s=rec["timings_s"].get("deep_lift_shap"),
            first_batch_s=rec.get("first_batch_s"),
            steady_pairs_per_s=rec.get("steady_pairs_per_s"),
            gpu_max_allocated_gb=rec["gpu_memory_attribution"]["max_allocated_bytes"] / 1e9,
            gpu_max_reserved_gb=rec["gpu_memory_attribution"]["max_reserved_bytes"] / 1e9,
            flags=_flag_brief(rec["precision_flags"]),
        )
    elif rec["kind"] == "ours":
        meta = rec["meta"]
        out.update(
            seqs_per_step=meta["seqs_per_step"],
            run_s=rec["timings_s"].get("run"),
            compile_s=meta.get("compile_s"),
            steady_pairs_per_s=meta.get("steady_pairs_per_s"),
            step_peak_gb=(meta.get("step_peak_bytes") or 0) / 1e9,
            gpu_max_allocated_gb=rec["gpu_memory"]["max_allocated_bytes"] / 1e9,
            gpu_max_reserved_gb=rec["gpu_memory"]["max_reserved_bytes"] / 1e9,
            oom=meta.get("oom"),
            flags=_flag_brief(meta["precision_flags"]),
            run_vs_raw=rec.get("run_vs_raw"),
        )
    return out


def _flag_brief(flags: dict) -> dict:
    keys = (
        "torch.get_float32_matmul_precision()",
        "torch.backends.cuda.matmul.allow_tf32",
        "torch.backends.cudnn.allow_tf32",
        "torch.backends.cudnn.benchmark",
        "torch.backends.cudnn.deterministic",
    )
    return {k.replace("torch.backends.", "").replace("torch.", ""): flags.get(k) for k in keys}


def _deviations(runs: Runs, report: dict) -> list[str]:
    out = []
    plan = report["plan"]
    if plan["name"] != "full":
        out.append(
            f"Plan {plan['name']!r}: {plan['about']}. Out of its scope, and not in the verdict: "
            + ", ".join(plan["out_of_scope"])
            + ". The plan has no eager run T3 (E6), no P5000 runs (E8) and no T2_S8 (E10's fp32 mode). "
            "inputs/x_tokens.npy still holds every kept peak as tokens (2,114 bytes each), as prepare writes it for "
            "any plan."
        )
    if runs.has("R1"):
        bs = int(runs.record("R1")["args"]["batch_size"])
        if bs != 64:
            out.append(
                f"The oracle ran at batch_size {bs}, not the pilot's 64: stock deep_lift_shap at batch_size 64 allocates "
                "15.5 GB on this model, above the 8 GB our processes may use on the shared GPUs. The floors are measured "
                f"at batch_size {bs} too, so every floor-relative gate compares like with like."
            )
    out.append("E10 compares S = 4 with S = 8, not 16: step_bytes estimates 14.9 GB for S = 16, above the 8 GB budget.")
    out.append(
        "The 8 truth peaks are a subset of P256, so d(R, T64) is read from R1/R2's own rows for those peaks (rows never "
        "mix); R1_8 is the literal extra R1 run on the 8 peaks alone, reported as context."
    )
    out.append(
        "Counts head only: the profile head is not implemented in cheri_dls (attribute_fast.py hands output 'profile' to "
        "the stock command). E9, end to end against `cherimoya attribute`, ran separately on the pilot model "
        "(attr_pilot/e9)."
    )
    out.append(
        f"E1 in reference mode and E10 in fp32 mode are gated as revised after round 2 ({REVISIONS_DOC}): their "
        "original fixed bounds sat below the stock reference's own rerun noise. The original bounds are still computed "
        "and listed under 'Revised criteria'; they do not gate."
    )
    if (runs.root.parent / "superseded").exists():
        out.append(
            "superseded/ keeps a first set of oracle runs whose E1 forward used plain batches of 40 rows. The rows of "
            "their last batch, 16 rows, differed from the engine's outputs by up to 1.2e-3, 2.6 to 2.9 times as much as "
            "any row of a 40-row batch: cuDNN and Triton pick kernels per shape. The oracle now forwards "
            "deep_lift_shap's own batches, and every oracle run was repeated."
        )
    if runs.has("R2"):
        getter = runs.record("R2")["precision_flags"].get("torch.get_float32_matmul_precision()")
        if isinstance(getter, str) and getter.startswith("<"):
            out.append(
                "R2's flags are checked on the CUDA getters (cuda.matmul.allow_tf32 False, cuda.matmul.fp32_precision "
                "ieee, cudnn.allow_tf32 False). In torch 2.14 the global torch.get_float32_matmul_precision() raises "
                "after the legacy allow_tf32 setter, because `import cherimoya` left the CPU backend at tf32."
            )
    return out


def _open_issues(runs: Runs, summary: dict, report: dict) -> list[str]:
    """One line per gate that failed or was not run, with the numbers that explain it where the report has them."""
    out = []
    floors = report.get("floors", {})
    e1 = report.get("E1_context", {})
    for gate, verdict in summary.items():
        if verdict is None:
            absent = report.get("not_run", {}).get(gate) or report["missing_runs"]
            out.append(f"{gate} was not evaluated (missing runs: {absent})")
            continue
        failed = [r for r in report["gates"].get(gate, []) if r["pass"] is False]
        what = "; ".join(
            f"{r['comparison']} {r['metric']} {r['statistic']} {_fmt(r['measured'])}, bound {r['op']} {_fmt(r['bound'])}"
            for r in failed
        )
        policy = report.get("E3_policy") or {}
        policy_note = (
            " E3's policy margins (median / 0.3 F_tf(0.5)): "
            + "; ".join(
                f"{m} ours {_ratio(v['ours_ratio'])}, stock rerun {_ratio(v['stock_rerun_ratio'])}"
                for m, v in policy.items()
            )
            + "."
            if gate == "E3" and policy
            else ""
        )
        if verdict == "via E7(a)":
            out.append(
                f"{gate} failed its floor-relative limits ({what}) and passed only through the E7(a) accuracy "
                f"tie-breaker (design §15.3).{policy_note}"
            )
            continue
        if verdict is not False:
            continue
        line = f"{gate} failed: {what}.{policy_note}"
        if gate == "E1":
            noise = e1.get("max_abs_between_runs", {})
            spread = [v for v in e1.get("module_y_x_copy_spread", {}).values() if v is not None]
            acc = e1.get("accuracy", {})
            line += (
                f" Context: R1' vs R1 {_fmt(noise.get('R1p vs R1'))}; identical copies of one sequence inside one "
                f"stock batch differ by up to {_fmt(max(spread) if spread else None)}. Against the fp64 truth on the 8 "
                "truth peaks: "
                + ", ".join(f"{k} {_fmt(v['ours_to_T64'])} vs {_fmt(v['reference_to_T64'])}" for k, v in acc.items())
                + " (ours vs the reference, max |y - y_T64|)."
            )
        if gate == "E10":
            hh = floors.get("F_hh", {}).get("raw", {})
            line += (
                f" Context: F_hh per-pair raw q50 {_fmt(hh.get('q50'))}, q99 {_fmt(hh.get('q99'))}, max "
                f"{_fmt(hh.get('max'))}; d(T2,T64) and d(T2_S8,T64) raw q50 "
                f"{_fmt(report['distance_to_T64'].get('T2', {}).get('raw', {}).get('q50'))} and "
                f"{_fmt(report['distance_to_T64'].get('T2_S8', {}).get('raw', {}).get('q50'))}."
            )
        out.append(line)
    if report["E8_reference"] is not None and not report["E8_reference"]["stock_p5000_run"]:
        out.append(
            "No stock P5000 run exists (attr_pilot/stock_run/ is empty), so E8 is read against the pilot's logged numbers "
            "only."
        )
    return out


def _ratio(value) -> str:
    """A margin such as measured / bound, to 3 decimals."""
    return "" if value is None else f"{value:.3f}"


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value == 0 or math.isnan(value) or math.isinf(value):
            return str(value)
        if 0.99 <= value < 1.0:  # correlations: keep the digits that matter
            return f"{value:.8f}"
        return f"{value:.4g}" if 1e-3 <= abs(value) < 1e4 else f"{value:.3e}"
    return str(value)


def _write_tsv(path: Path, rows: list[dict]) -> None:
    cols = ("gate", "comparison", "metric", "statistic", "measured", "op", "bound", "rule", "pass", "note")
    lines = ["\t".join(cols)]
    for r in rows:
        lines.append(
            "\t".join(_fmt(r[c]) if c in ("measured", "bound") else str(r[c] if r[c] is not None else "") for c in cols)
        )
    path.write_text("\n".join(lines) + "\n")


def _revisions_markdown(report: dict) -> list[str]:
    """The report's "Revised criteria" section: each revised criterion's original bound, reported only, and the
    revised bounds that gate, with the measured values."""
    rev = report.get("revisions") or {}
    e1, e10 = rev.get("E1_reference"), rev.get("E10_fp32")
    if not (e1 and e1.get("runs")) and not e10:
        return []
    lines = [
        "",
        "## Revised criteria",
        "",
        f"E1 in reference mode and E10 in fp32 mode gate with the revised bounds of `{rev.get('document')}`, in the "
        "table above. Their original fixed bounds sat below the stock reference's own rerun noise. They are still "
        "computed, here, and do not gate.",
        "",
        "| criterion | comparison | statistic | measured | original bound | passes original | revised bound | "
        "passes revised |",
        "|---|---|---|---|---|---|---|---|",
    ]
    if e1 and e1.get("runs"):
        r = e1["revised"]
        floor_note = f"2 x R1' vs R1 {_fmt(r['floor_R1p_vs_R1'])} = {_fmt(r['floor_bound'])}"
        accuracy_note = f"1.2 x R1 to T64 {_fmt(r['R1_to_T64'])} = {_fmt(r['accuracy_bound'])}"
        for name, v in e1["runs"].items():
            lines.append(
                f"| E1 reference | {name} y vs R1 module y | max abs | {_fmt(v['max_abs_vs_R1'])} | "
                f"{_fmt(e1['original']['bound'])} | {v['original_pass']} | {floor_note} | {v['floor_pass']} |"
            )
            lines.append(
                f"| E1 reference | {name} y vs T64 y (truth peaks) | max abs | {_fmt(v['max_abs_to_T64'])} | (none) | "
                f"- | {accuracy_note} | {v['accuracy_pass']} |"
            )
    if e10:
        checks = e10["revised"]["checks"]
        hh = e10["revised"]["F_hh_raw"] or {}
        for stat, factor, name in (("q50", "2", "0.5"), ("q99", "3", "0.99")):
            check = checks.get(stat, {})
            original = (
                (_fmt(e10["original"]["bound"]), str(e10["original"]["pass"])) if stat == "q99" else ("(none)", "-")
            )
            lines.append(
                f"| E10 fp32 | T2_S8 vs T2, raw | {stat} | {_fmt(check.get('measured'))} | {original[0]} | "
                f"{original[1]} | max({factor} x F_hh({name}) {_fmt(hh.get(stat))}, "
                f"{'1e-5' if stat == 'q50' else '1e-4'}) = {_fmt(check.get('bound'))} | {check.get('pass')} |"
            )
    return lines


def _verdict(verdict) -> str:
    """A gate's verdict as the reports print it."""
    if verdict is True:
        return "PASS"
    if verdict is False:
        return "FAIL"
    if verdict is None:
        return "not run"
    return f"PASS {verdict}"  # "via E7(a)"


def _margins_markdown(report: dict) -> list[str]:
    """E3's precision-policy margins and E7(a)'s accuracy ratios, both context: computed whatever E3 and E4 give."""
    lines = []
    policy = report.get("E3_policy") or {}
    if policy:
        lines += [
            "",
            "## E3 precision policy: margins",
            "",
            "E3's policy check is median <= 0.3 F_tf(0.5). Ours is T1 vs R1; the stock rerun is R1' vs R1 (F_rr), "
            "against the same bound.",
            "",
            "| metric | bound 0.3 F_tf(0.5) | T1 vs R1 median | ours / bound | R1' vs R1 median | stock rerun / bound | "
            "pass |",
            "|---|---|---|---|---|---|---|",
        ]
        for metric, v in policy.items():
            lines.append(
                f"| {metric} | {_fmt(v['bound'])} | {_fmt(v['ours_q50'])} | {_ratio(v['ours_ratio'])} | "
                f"{_fmt(v['stock_rerun_q50'])} | {_ratio(v['stock_rerun_ratio'])} | {v['pass']} |"
            )
    e7 = report.get("E7_context") or {}
    if e7:
        lines += [
            "",
            "## E7(a) context: distance to the fp64 truth",
            "",
            "d(a, T64) is a's rel-L2 to T64 on the 8 truth peaks (raw per pair, hyp per peak). E7(a) passes when "
            "d(ours, T64) <= 1.2 d(R, T64) at the median and the max; it decides E3 or E4 only when they fail.",
            "",
            "| gate | ours vs reference | metric | d(ours) q50 | d(R) q50 | ratio q50 | d(ours) max | d(R) max | "
            "ratio max | within 1.2 |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for gate, g in e7.items():
            for metric, v in g["metrics"].items():
                lines.append(
                    f"| {gate} | {g['ours']} vs {g['reference']} | {metric} | {_fmt(v['d_ours']['q50'])} | "
                    f"{_fmt(v['d_ref']['q50'])} | {_ratio(v['ratio']['q50'])} | {_fmt(v['d_ours']['max'])} | "
                    f"{_fmt(v['d_ref']['max'])} | {_ratio(v['ratio']['max'])} | {v['within']} |"
                )
    return lines


def _markdown(report: dict, rows: list[dict]) -> str:
    inp = report["inputs"]
    lines = [
        "# cheri_dls GPU validation on the real model, counts head (design §15.3)",
        "",
        f"Generated {report['generated_et']} (ET) by `workflows/attribution/dls_validate.py gates`. Every number is in "
        "`report.json`; every check, with its bound and rule, in `report.tsv`; each run's record and arrays under `runs/`.",
        "",
        f"Model: `{inp.get('model')}`. Peaks: `{', '.join(inp.get('loci') or [])}`.",
        "",
        f"Gates computed by `{report['harness']['file']}` at commit {report['harness']['commit']} (uncommitted changes "
        f"in workflows/attribution: {report['harness']['dirty']}).",
        "",
        f"Inputs: {inp['n_kept']} kept peaks of length {inp['length']}, window {inp['window']}; P256 = "
        f"{inp['p256_rule']}; truth peaks = {inp['truth_rule']} (kept indices {inp['truth_kept_indices']}). "
        f"G0: {inp['G0']['bitwise_equal']}/{inp['G0']['pairs']} reference pairs bitwise equal to tangermeme's.",
        "",
        f"Plan {report['plan']['name']!r}: {report['plan']['about']}.",
        "",
        "## Verdict",
        "",
        "| gate | result |",
        "|---|---|",
    ]
    for gate, verdict in report["summary"].items():
        lines.append(f"| {gate} | {_verdict(verdict)} |")
    lines += ["", f"All gates pass: **{report['all_gates_pass']}**."]
    if report["plan"]["out_of_scope"]:
        lines += [
            "",
            f"Out of the scope of plan {report['plan']['name']!r}, not in the verdict: "
            f"{', '.join(report['plan']['out_of_scope'])}.",
        ]
    lines += ["", "## Floors (E2), counts head", ""]
    lines += [
        "| floor | comparison | raw q50 | raw q99 | raw max | hyp q50 | hyp q99 | hyp max |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, f in report["floors"].items():
        lines.append(
            f"| {name} | {f['a']} vs {f['b']} | "
            + " | ".join(_fmt(f[m][s]) for m in ("raw", "hyp") for s in ("q50", "q99", "max"))
            + " |"
        )
    lines += [
        "",
        "raw = per-pair rel-L2 of the multipliers over 4 x 2114; hyp = per-peak rel-L2 of the windowed hypothetical "
        "attributions over 4 x 400; a vs b = ||a - b|| / ||b||.",
        "",
        "## Gate checks",
        "",
        "| gate | comparison | metric | statistic | measured | bound | rule | pass |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r["gate"].startswith(("context", "E8 fp32")) or r["gate"] == "E2" or "original" in r["gate"]:
            continue
        lines.append(
            f"| {r['gate']} | {r['comparison']} | {r['metric']} | {r['statistic']} | {_fmt(r['measured'])} | "
            f"{r['op']} {_fmt(r['bound'])} | {r['rule']} | {r['pass']} |"
        )
    lines += _revisions_markdown(report)
    lines += _margins_markdown(report)
    lines += [
        "",
        "## Runs",
        "",
        "| run | kind | config | seconds | steady pairs/s | GPU max alloc GB | deltas > thr | max delta |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, s in report["runs"].items():
        spec = s["spec"]
        config = ", ".join(f"{k}={v}" for k, v in spec.items() if k != "about")
        if s["kind"] == "oracle":
            config += f", batch_size={s['batch_size']}"
        d = s.get("deltas") or {}
        lines.append(
            f"| {name} | {s['kind']} | {config} | {_fmt(s.get('total_s'))} | {_fmt(s.get('steady_pairs_per_s'))} | "
            f"{_fmt(s.get('gpu_max_allocated_gb'))} | {d.get('above')}/{d.get('pairs')} | {_fmt(d.get('max'))} |"
        )
    e1 = report.get("E1_context") or {}
    if e1.get("max_abs_between_runs") or e1.get("max_abs_to_T64_truth_peaks"):
        lines += ["", "## Forward outputs (E1 context): max |y_a - y_b| over every sequence and reference row", ""]
        lines += ["| comparison | max abs |", "|---|---|"]
        lines += [f"| {k} | {_fmt(v)} |" for k, v in e1.get("max_abs_between_runs", {}).items()]
        lines += [
            f"| {k} vs T64 (8 truth peaks, 168 rows) | {_fmt(v)} |"
            for k, v in e1.get("max_abs_to_T64_truth_peaks", {}).items()
        ]
        spread = {k: v for k, v in e1.get("module_y_x_copy_spread", {}).items() if v is not None}
        if spread:
            lines += [
                "",
                "Inside one stock process, the K copies of a sequence in one batch give outputs that differ by up to: "
                + ", ".join(f"{k} {_fmt(v)}" for k, v in spread.items())
                + ".",
            ]
    if report.get("distance_to_T64"):
        lines += [
            "",
            "## Distance to the fp64 truth T64 (8 peaks, 160 pairs)",
            "",
            "| run | raw q50 | raw max | hyp q50 | hyp max |",
            "|---|---|---|---|---|",
        ]
        for name, c in report["distance_to_T64"].items():
            lines.append(
                f"| {name} | {_fmt(c['raw']['q50'])} | {_fmt(c['raw']['max'])} | {_fmt(c['hyp']['q50'])} | {_fmt(c['hyp']['max'])} |"
            )
    if report.get("comparisons"):
        lines += [
            "",
            "## Other comparisons (context)",
            "",
            "| comparison | raw q50 | raw q99 | raw max | hyp q50 | hyp max | min Pearson |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, c in report["comparisons"].items():
            lines.append(
                f"| {name} | {_fmt(c['raw']['q50'])} | {_fmt(c['raw']['q99'])} | {_fmt(c['raw']['max'])} | "
                f"{_fmt(c['hyp']['q50'])} | {_fmt(c['hyp']['max'])} | {_fmt(c['pearson']['min'])} |"
            )
    probes = report.get("probes")
    if probes:
        lines += [
            "",
            f"## Probe P3 at these row counts ({probes['n_runs']} fresh processes; the reference's rows "
            f"{probes['reference_rows']})",
            "",
            "| op | state | classes by rows (processes) |",
            "|---|---|---|",
        ]
        for op, v in probes["ops"].items():
            classes = "; ".join(
                f"{rows}: " + ", ".join(f"{c} x{n}" for c, n in counts.items())
                for rows, counts in (v.get("class_counts_by_rows") or {}).items()
            )
            lines.append(f"| {op} | {v['state']} | {classes} |")
        heads = probes.get("heads") or {}
        if "counts" in heads:
            lines += ["", f"conv_class needed for the counts head: **{heads['counts'].get('conv_class_needed')}**."]
    lines += ["", "## Deviations", ""] + [f"- {d}" for d in report["deviations"]]
    lines += ["", "## Open issues", ""] + ([f"- {d}" for d in report["open_issues"]] or ["- none"])
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------------------------
# summarize: several validations side by side


def _summary_entry(out_dir: Path) -> dict:
    """What summarize reads from one validation directory: report.json, inputs/meta.json and the runs' records.

    Everything is derived from fields every report.json has, so a report written before plans existed (the pilot's,
    plan "full") reads the same way.
    """
    import numpy as np

    out_dir = Path(out_dir)
    report = json.loads((out_dir / "report.json").read_text())
    meta = json.loads((out_dir / "inputs" / "meta.json").read_text())
    model = Path(meta["parameters"]["model"])
    plan = report.get("plan") or {"name": "full", "out_of_scope": []}
    devices: dict = {}
    for name in report.get("runs", {}):
        rec = json.loads((out_dir / "runs" / name / "record.json").read_text())
        device = (rec.get("env") or {}).get("CUDA_VISIBLE_DEVICES")
        if device:
            gpu = f"{device} ({(rec.get('versions') or {}).get('gpu', '?')})"
            devices.setdefault(gpu, []).append(name)
    gates = report["gates"]
    return {
        "dir": str(out_dir),
        "label": f"{model.parent.parent.name} {model.parent.name}",
        "model": str(model),
        "loci": meta["parameters"]["loci"],
        "n_loci_read": int(np.load(out_dir / "inputs" / "idx.npy", mmap_mode="r").shape[0]),
        "n_kept": meta["n_kept"],
        "p256_first_last": meta["p256_first_last"],
        "truth_kept_indices": meta["truth_kept_indices"],
        "G0": meta["G0"],
        "plan": plan["name"],
        "out_of_scope": plan.get("out_of_scope", []),
        "gpu_runs": devices,
        "summary": report["summary"],
        "all_gates_pass": report["all_gates_pass"],
        "floors": report["floors"],
        "E3_policy": e3_policy(report["floors"], gates.get("E3", [])),
        "E7_context": e7_context(report.get("E7_computed") or {}),
        "rows": [r for g in report["summary"] if g != "E2" for r in gates.get(g, [])] + gates.get("E7", []),
        "open_issues": report.get("open_issues", []),
        "harness": report.get("harness"),
    }


def _check_cell(row: dict | None) -> str:
    """One gate check in the summary's matrix: measured against the bound, the margin for an upper bound, the verdict."""
    if row is None:
        return ""
    verdict = {True: "pass", False: "FAIL", None: "not run"}[row["pass"]]
    measured, bound, op = row["measured"], row["bound"], row["op"]
    if bound is None:
        return f"{_fmt(measured)}, {verdict}"
    if op == "<=" and isinstance(bound, float) and bound > 0 and isinstance(measured, int | float):
        return f"{_fmt(measured)} / {_fmt(bound)} = {measured / bound:.3f}, {verdict}"
    return f"{_fmt(measured)} {op} {_fmt(bound)}, {verdict}"


def _summary_markdown(entries: list[dict], reference: dict | None, notes: list[str]) -> str:
    cols = entries + ([reference] if reference else [])
    names = [e["label"] + (" (reference)" if e is reference else "") for e in cols]
    lines = [
        "# cheri_dls validation on more checkpoints, counts head",
        "",
        f"Generated {_now_et()} (ET) by `workflows/attribution/dls_validate.py summarize`, from each directory's "
        "report.json and inputs/meta.json. Each checkpoint's report.md has the full report.",
        "",
    ]
    for name in sorted({e["plan"] for e in entries}):
        lines.append(f"Plan {name!r}: {PLANS[name]['about'] if name in PLANS else '?'}.")
    harness = sorted({f"{h['commit']} (dirty: {h['dirty']})" for e in entries if (h := e["harness"])})
    if harness:
        lines += ["", "The checkpoints' gates were computed at commit " + ", ".join(harness) + "."]
    if reference:
        lines += [
            "",
            f"The last column is the {reference['plan']!r} validation of {reference['label']} ({reference['dir']}), "
            "shown for comparison. It is not part of this summary's verdict.",
        ]
    verdict = all(e["all_gates_pass"] for e in entries)
    lines += ["", f"Every gate in scope passes on every checkpoint: **{verdict}**."]
    for metric in ("raw", "hyp"):
        ours = [e["E3_policy"][metric]["ours_ratio"] for e in entries if metric in e["E3_policy"]]
        rerun = [e["E3_policy"][metric]["stock_rerun_ratio"] for e in entries if metric in e["E3_policy"]]
        if ours and None not in rerun:
            line = (
                f"E3 policy margin, {metric}: ours / bound {min(ours):.3f} to {max(ours):.3f}; stock rerun / bound "
                f"{min(rerun):.3f} to {max(rerun):.3f}"
            )
            ref = (reference or {}).get("E3_policy", {}).get(metric)
            if ref:
                line += f" (reference: {_ratio(ref['ours_ratio'])} and {_ratio(ref['stock_rerun_ratio'])})"
            lines.append("")
            lines.append(line + ".")

    lines += [
        "",
        "## Checkpoints",
        "",
        "| checkpoint | model | peaks file | loci read | kept peaks | P256 (kept indices) | truth peaks (kept indices) | "
        "G0 | GPU (runs) | plan |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, e in zip(names, cols):
        gpus = "; ".join(f"{g}: {len(r)}" for g, r in e["gpu_runs"].items())
        first, last = e["p256_first_last"]
        lines.append(
            f"| {name} | `{Path(e['model']).name}` | `{', '.join(Path(p).name for p in e['loci'])}` | "
            f"{e['n_loci_read']} | {e['n_kept']} | 256, {first}..{last} | {e['truth_kept_indices']} | "
            f"{e['G0']['bitwise_equal']}/{e['G0']['pairs']} | {gpus} | {e['plan']} |"
        )

    gates = [g for g in GATES if any(g in e["summary"] for e in cols)]
    lines += [
        "",
        "## Verdicts",
        "",
        "| checkpoint | " + " | ".join(gates) + " | all gates in scope |",
        "|---|" + "---|" * (len(gates) + 1),
    ]
    for name, e in zip(names, cols):
        cells = [
            _verdict(e["summary"][g]) if g in e["summary"] else "out of scope" if g in e["out_of_scope"] else "-"
            for g in gates
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {e['all_gates_pass']} |")

    lines += [
        "",
        "## Floors (E2)",
        "",
        "raw = per-pair rel-L2 of the multipliers over 4 x 2114; hyp = per-peak rel-L2 of the windowed hypothetical "
        "attributions over 4 x 400. F_rr = R1' vs R1, F_hh = R2' vs R2, F_tf = R2 vs R1.",
        "",
        "| checkpoint | floor | raw q50 | raw q99 | raw max | hyp q50 | hyp q99 | hyp max |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, e in zip(names, cols):
        for floor, f in e["floors"].items():
            lines.append(
                f"| {name} | {floor} | "
                + " | ".join(_fmt(f[m][s]) for m in ("raw", "hyp") for s in ("q50", "q99", "max"))
                + " |"
            )

    lines += [
        "",
        "## E3 precision policy: margins",
        "",
        "The policy check is median <= 0.3 F_tf(0.5). Ours is T1 vs R1's median; the stock rerun's is F_rr(0.5), "
        "R1' vs R1, against the same bound. A ratio above 1 fails the check.",
        "",
        "| checkpoint | metric | bound 0.3 F_tf(0.5) | T1 vs R1 median | ours / bound | R1' vs R1 median | "
        "stock rerun / bound | pass |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, e in zip(names, cols):
        for metric, v in e["E3_policy"].items():
            lines.append(
                f"| {name} | {metric} | {_fmt(v['bound'])} | {_fmt(v['ours_q50'])} | {_ratio(v['ours_ratio'])} | "
                f"{_fmt(v['stock_rerun_q50'])} | {_ratio(v['stock_rerun_ratio'])} | {v['pass']} |"
            )

    lines += [
        "",
        "## E7(a) context: distance to the fp64 truth",
        "",
        "d(a, T64): rel-L2 to the fp64 truth on the 8 truth peaks (raw per pair, hyp per peak). E7(a) holds when "
        "d(ours, T64) <= 1.2 d(R, T64) at the median and the max; it decides E3 or E4 only when they fail.",
        "",
        "| checkpoint | for | ours vs R | metric | d(ours) q50 | d(R) q50 | ratio q50 | ratio max | within 1.2 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, e in zip(names, cols):
        for gate, g in e["E7_context"].items():
            for metric, v in g["metrics"].items():
                lines.append(
                    f"| {name} | {gate} | {g['ours']} vs {g['reference']} | {metric} | {_fmt(v['d_ours']['q50'])} | "
                    f"{_fmt(v['d_ref']['q50'])} | {_ratio(v['ratio']['q50'])} | {_ratio(v['ratio']['max'])} | "
                    f"{v['within']} |"
                )

    keys: list = []
    for e in entries:
        for r in e["rows"]:
            key = (r["gate"], r["comparison"], r["metric"], r["statistic"])
            if key not in keys:
                keys.append(key)
    lookup = [{(r["gate"], r["comparison"], r["metric"], r["statistic"]): r for r in e["rows"]} for e in cols]
    lines += [
        "",
        "## Gate checks: measured against the bound",
        "",
        "Each cell is measured / bound = margin for an upper bound, or measured, the operator and the bound otherwise. "
        "The rule of each check is in its row.",
        "",
        "| gate | comparison | metric | statistic | rule | " + " | ".join(names) + " |",
        "|---|---|---|---|---|" + "---|" * len(names),
    ]
    for key in keys:
        rule = next(lk[key]["rule"] for lk in lookup if key in lk)
        cells = [_check_cell(lk.get(key)) for lk in lookup]
        lines.append(f"| {' | '.join(key)} | {rule} | " + " | ".join(cells) + " |")

    issues = [(name, issue) for name, e in zip(names, cols) for issue in e["open_issues"]]
    lines += ["", "## Open issues, from each report", ""]
    lines += [f"- {name}: {issue}" for name, issue in issues] or ["- none"]
    if notes:
        lines += ["", "## Notes", ""] + [f"- {n}" for n in notes]
    return "\n".join(lines) + "\n"


def cmd_summarize(args: argparse.Namespace) -> int:
    """summary.md (and summary.json) over several validation directories; exit 1 unless every one passes."""
    entries = [_summary_entry(Path(d)) for d in args.dirs]
    reference = _summary_entry(Path(args.reference_dir)) if args.reference_dir else None
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_summary_markdown(entries, reference, args.note or []))
    _write_json(
        out.with_suffix(".json"),
        {"generated_et": _now_et(), "checkpoints": entries, "reference": reference, "notes": args.note or []},
    )
    for e in entries:
        logger.info("%s: %s (all gates in scope pass: %s)", e["label"], e["summary"], e["all_gates_pass"])
    logger.info("wrote %s and %s", out, out.with_suffix(".json"))
    return 0 if all(e["all_gates_pass"] for e in entries) else 1


# ---------------------------------------------------------------------------------------------------------------------
# command line


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--out-dir", type=Path, required=True, help="the validation directory (inputs/, runs/, reports)")

    p = sub.add_parser("checkpoint-json", help="the attribute.json of another trained checkpoint, from a base JSON")
    common(p)
    p.add_argument("--base", required=True, help="the base attribute.json (the pilot's), listing all 24 chromosomes")
    p.add_argument("--fold-dir", type=Path, required=True, help="<dataset>/<fold>, with fit.json and the model")

    p = sub.add_parser("prepare", help="extract the peaks, P256, the truth peaks and P256's references; check G0")
    common(p)
    p.add_argument("-p", "--parameters", required=True, help="the attribute.json of `cherimoya attribute`")
    p.add_argument("--n-peaks", type=int, default=256)
    p.add_argument("--n-truth", type=int, default=8)
    p.add_argument("--ref-workers", type=int, default=2, help="RefProducer worker processes (default 2)")
    p.add_argument("--chunk-peaks", type=int, default=16, help="peaks per RefProducer chunk (default 16)")
    p.add_argument("--threads", type=int, default=4)

    p = sub.add_parser("p256-table", help="write inputs/p256.tsv (P256's indices and coordinates) for a prepared dir")
    common(p)

    for kind, table in (("oracle", ORACLE_RUNS), ("ours", OURS_RUNS)):
        for name in (kind, f"_{kind}"):
            p = sub.add_parser(
                name, help=f"the {kind} runs, each in a fresh subprocess" if name == kind else argparse.SUPPRESS
            )
            common(p)
            if name == kind:
                p.add_argument(
                    "--runs", default=None, help=f"comma-separated, of {', '.join(table)}; default: the plan's"
                )
                p.add_argument("--plan", choices=list(PLANS), default="full", help="whose runs --runs defaults to")
                p.add_argument("--timeout", type=int, default=2400, help="seconds before a child is killed")
            else:
                p.add_argument("--run", required=True, choices=list(table))
            p.add_argument(
                "--gpu-mem-cap-gb",
                type=float,
                default=7.3 if kind == "oracle" else 7.5,
                help="cap of each child's CUDA caching allocator, GB (1e9 bytes)",
            )
            if kind == "oracle":
                p.add_argument(
                    "--batch-size",
                    type=int,
                    default=64,
                    help="deep_lift_shap's batch_size (the JSON's is 64, which needs about 15.5 GB)",
                )
            else:
                p.add_argument("--mem-budget-gb", type=float, default=8.0)
                p.add_argument("--ref-workers", type=int, default=2, help="RefProducer workers for P5000 (default 2)")

    p = sub.add_parser("truth", help="T64: the engine on the CPU in float64 on the 8 truth peaks")
    common(p)
    p.add_argument("--seqs-per-step", type=int, default=2)
    p.add_argument("--threads", type=int, default=4)

    p = sub.add_parser("gates", help="floors, gates and context; writes report.json, report.tsv and report.md")
    common(p)
    p.add_argument("--plan", choices=list(PLANS), default="full", help="the runs and gates of this validation")
    p.add_argument("--stock-deltas", help="deltas (.npy, (N, K)) of a stock P5000 run, to read E8 against as well")
    p.add_argument("--probe-json", help="dls_probes.json of a P3 probe run at the rows these runs used, to report")

    p = sub.add_parser("summarize", help="summary.md (and .json) over several validation directories")
    p.add_argument("dirs", nargs="+", type=Path, help="validation directories, each with report.json")
    p.add_argument("--out", type=Path, required=True, help="the summary's markdown path; .json is written next to it")
    p.add_argument("--reference-dir", type=Path, help="another validation to show last, for comparison (not gated)")
    p.add_argument("--note", action="append", help="a line for the summary's Notes section (repeatable)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    commands = {
        "checkpoint-json": cmd_checkpoint_json,
        "prepare": cmd_prepare,
        "p256-table": cmd_p256_table,
        "oracle": lambda a: _drive("oracle", a),
        "ours": lambda a: _drive("ours", a),
        "_oracle": child_oracle,
        "_ours": child_ours,
        "truth": cmd_truth,
        "gates": cmd_gates,
        "summarize": cmd_summarize,
    }
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
