#!/usr/bin/env python3
"""One Mímir project: its project.yaml, the paths every stage writes, and whether each stage is done.

run_project.sh runs a project's stages in one box; this module is what it asks. A project is one directory (root) and
one project.yaml (kept in it; project.example.yaml in this folder documents every key). Every path below is derived
here, once, from root, prefix and the sample stems:

  <root>/fragments/<stem>.fragments.tsv.gz       the inputs (fetch_igvf.py writes them; igvf_tf_chrombpnet's config
                                                 check accepts this suffix, not .bed.gz)
  <reference_root>/hg38/, motif/                 the genome, chrom sizes, blacklist and motif database, read by every
                                                 step (setup.sh --references downloads them)
  <root>/portal/                                 other portal files and manifest.tsv (fetch_igvf.py)
  <root>/peaks/<ds>/<ds>.narrowPeak.gz           00.0.call_peaks           (prep)
  <root>/chrombpnet/configs/<ds>/config.yaml     make_configs.py           (prep)
  <root>/chrombpnet/<ds>/preprocessing/          00.0 signal, 00.1 peaks, 01.0 fold negatives (prep)
  <root>/chrombpnet/<ds>/plots/signal_qc/        02.0                      (prep)
  <root>/cherimoya/<ds>/fold_<k>/                make_jobs.py, the fit, the attribution (train, attribute)
  <root>/cherimoya/configs/<ds>/config.yaml      make_downstream.py: 06.0-11.0 on the converted scores
  <root>/cherimoya/<ds>/downstream/              06.0-08.0 (.done/ markers), finemo_<tag>/<ds>_all/ (downstream, motifs)
  <root>/compendia/<tag>/                        09.0's compendium; compendia/inputs/ the lists (motifs)
  <root>/cherimoya/cooperativity/                motifs.tsv, args/, groups.tsv, comparisons.tsv, series.tsv, <ds>/,
                                                 combined/, official/ (cooperativity)
  <root>/cherimoya/cooperativity/matrix/         <sites ds>/<model ds>/, combined/ (coop_matrix, optional)
  <root>/box/<job>/                              each box's logs and stages.tsv

with <ds> = prefix + stem. A stage is done when its outputs are there (`check`), never because a command exited 0:
the runners it calls report per task, and some of them always exit 0.

    project.py <project.yaml> show                 the validated project, as JSON
    project.py <project.yaml> env                  shell assignments for run_project.sh (eval)
    project.py <project.yaml> check <stage>        exit 0 when the stage's outputs are complete, else 1 and what is
                                                   missing (stages: fetch prep train attribute downstream motifs
                                                   cooperativity coop_matrix)
    project.py <project.yaml> status               every stage's state, one line each
    project.py <project.yaml> coop-inputs          write the cooperativity args files, groups.tsv, comparisons.tsv
    project.py <project.yaml> prune [--apply <manifest>]
                                                   the dry run writes <root>/prune/manifest.<stamp>.tsv, the
                                                   derived intermediates a finished project can lose; --apply
                                                   deletes exactly that manifest's files (prune.py)
    project.py <project.yaml> share --out <file.tar> [--datasets a,b] [--with-signal] [--list]
                                                   the results, packed for someone without cluster access (share.py)

    project.py <project.yaml> sbatch-args          submit.sh's sbatch options from the slurm: block, one per line

The steps after Cherimoya's own (00.0-02.0, 06.0-11.0) are igvf_tf_chrombpnet's, copied unchanged into
vendor/igvf_tf_chrombpnet/ (VENDOR); their folds are the copy's folds/, and every reference file, the genome
included, comes from reference_root through the copy's own references.layout(), so the steps and this module cannot
name different files.

`driver: false` in project.yaml marks a project the driver must not run (one set up by hand before the driver
existed, kept for its results): env and coop-inputs refuse it, so run_project.sh and submit.sh do too, while
show, check, status, prune and share still work, without reference_root.

Needs only the standard library and PyYAML (the default pixi env), so it runs on the login node.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "attribution"))
REPO = HERE.parent.parent
VENDOR = REPO / "vendor" / "igvf_tf_chrombpnet"
"""igvf_tf_chrombpnet's steps, copied unchanged (VENDORED.md); MANIFEST.tsv names the commit."""

STAGES = ("fetch", "prep", "train", "attribute", "downstream", "motifs", "cooperativity", "coop_matrix")
"""In order. fetch runs on the login node (fetch_igvf.py); run_project.sh runs the others in a box."""
OPTIONAL_STAGES = ("coop_matrix",)
"""Box stages a project runs only when its project.yaml lists them in stages."""
BOX_STAGES = tuple(s for s in STAGES[1:] if s not in OPTIONAL_STAGES)
"""The box stages a project runs by default."""
NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")
ACCESSION_RE = re.compile(r"^IGVFFI[0-9]{4}[A-Z]{4}$")
FRAGMENTS_SUFFIX = ".fragments.tsv.gz"
DEFAULT_FOLDS = ("0", "1", "2", "3", "4")
SLOTS = {
    "train_per_gpu": 4,  # Cherimoya fits per H100: CPU/launch-bound, so several share a GPU
    "train_cores": 6,
    "attribute_per_gpu": 4,  # `cherimoya attribute`'s fast path plans at most 12 GB per process
    "attribute_cores": 4,
    "modisco_max": 5,  # 08.0 runs at once
    "modisco_threads": 24,  # Leiden, most of an 08.0, is single-threaded
    "finemo_slots": 5,  # 10.0 runs at once, one GPU each
    "finemo_batch": 200,
    "coop_cores": 16,
    "prep_max": 0,  # prep chains at once; 0 = every dataset
}
"""The most each slot gets: what a whole Gefion node (8 H100, 216 CPUs, ~2 TB) runs. resolve_slots() scales them
down to a smaller machine; a value project.yaml gives is used as it is."""
MODISCO_GB = 60  # an 08.0's peak memory, with room (docs/pipeline.md: ~55 GB)
PREP_GB = 64  # a prep chain's peak memory, with room (~63 GB)
SLURM_KEYS = ("partition", "account", "gpus", "cpus", "time", "extra")
RETIRED = {
    "igvf": "igvf_tf_chrombpnet's steps are in vendor/igvf_tf_chrombpnet/ now",
    "chrombpnet": "chrombpnet is the pixi env `chrombpnet` now (setup.sh installs it)",
    "chrombpnet_env": "chrombpnet is the pixi env `chrombpnet` now (setup.sh installs it)",
    "genome": "the genome comes from reference_root",
    "folds_dir": "the folds are vendor/igvf_tf_chrombpnet/folds/",
}
"""Keys project.yaml no longer takes, and why."""
DEFAULTS = {
    "reference_root": None,
    "slurm": None,
    "stages": list(BOX_STAGES),
    "hold_minutes": 60,
    "shift_every": 100,
    "driver": True,
}
DRIVER_REFUSED = 3
"""project.py's exit code when a driver command meets a project marked driver: false."""


class ProjectError(ValueError):
    pass


# ---------------------------------------------------------------------------------------------------------------------
# the project


def load(path: Path) -> dict:
    """project.yaml, validated, with defaults filled and every derived path added."""
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ProjectError(f"{path}: not a mapping")
    retired = sorted(set(raw) & set(RETIRED))
    if retired:
        why = "; ".join(f"{k}: {RETIRED[k]}" for k in retired)
        raise ProjectError(f"{path}: {retired} are no longer project.yaml keys ({why}); give reference_root instead "
                           "(docs/install.md)")  # fmt: skip
    known = {"project", "root", "prefix", "samples", "compendium", "cooperativity", "slots", "folds", "prune",
             *DEFAULTS}  # fmt: skip
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ProjectError(f"{path}: unknown keys {unknown}")
    p = {**DEFAULTS, **raw, "file": str(path)}
    for key in ("project", "root", "prefix", "samples", "compendium"):
        if not p.get(key):
            raise ProjectError(f"{path}: {key} is required")
    if not NAME_RE.match(p["project"]) or not NAME_RE.match(p["compendium"]):
        raise ProjectError(f"{path}: project and compendium must match {NAME_RE.pattern}")
    if p["prefix"] and not NAME_RE.match(p["prefix"]):
        raise ProjectError(f"{path}: prefix must match {NAME_RE.pattern}")
    if not isinstance(p["driver"], bool):
        raise ProjectError(f"{path}: driver must be true or false")
    root = Path(p["root"])
    if not root.is_absolute():
        raise ProjectError(f"{path}: root must be an absolute path")
    p["root"] = str(root)
    p["references"] = None
    p["genome"] = None
    if p["reference_root"] is not None:
        ref = Path(str(p["reference_root"]))
        if not ref.is_absolute():
            raise ProjectError(f"{path}: reference_root must be an absolute path")
        p["reference_root"] = str(ref)
        p["references"] = reference_files(ref)
        p["genome"] = p["references"]["genome_fa"]
    p["folds_dir"] = str(VENDOR / "folds")
    p["folds"] = [str(f) for f in p.get("folds") or DEFAULT_FOLDS]
    stages = list(p["stages"])
    bad = [s for s in stages if s not in STAGES[1:]]
    if bad:
        raise ProjectError(f"{path}: unknown stages {bad}; the box stages are {list(STAGES[1:])}")
    p["stages"] = [s for s in STAGES[1:] if s in stages]  # always in pipeline order
    p["slots"] = dict(p.get("slots") or {})  # what project.yaml gives; resolve_slots() fills the rest in the box
    bad = sorted(set(p["slots"]) - set(SLOTS))
    if bad:
        raise ProjectError(f"{path}: unknown slots {bad}")
    for k, v in p["slots"].items():
        if not isinstance(v, int) or isinstance(v, bool) or v < (0 if k == "prep_max" else 1):
            raise ProjectError(f"{path}: slots {k} must be a positive integer, got {v!r}")
    slurm = p["slurm"] or {}
    if not isinstance(slurm, dict) or sorted(set(slurm) - set(SLURM_KEYS)):
        raise ProjectError(f"{path}: slurm takes only {list(SLURM_KEYS)}, got {slurm!r}")
    p["slurm"] = slurm

    samples, seen = [], set()
    for s in p["samples"]:
        if not isinstance(s, dict) or "stem" not in s or "fragments" not in s:
            raise ProjectError(f"{path}: each sample needs stem and fragments, got {s!r}")
        stem = str(s["stem"])
        if not NAME_RE.match(stem) or stem in seen:
            raise ProjectError(f"{path}: sample stem {stem!r} is not unique or not {NAME_RE.pattern}")
        seen.add(stem)
        frag = str(s["fragments"])
        accession = frag if ACCESSION_RE.match(frag) else None
        if accession is None and not Path(frag).is_absolute():
            raise ProjectError(
                f"{path}: sample {stem}: fragments is neither an IGVF file accession nor an absolute path"
            )
        extra = [str(x) for x in s.get("extra") or []]
        for acc in ([s["index"]] if s.get("index") else []) + extra:
            if not ACCESSION_RE.match(str(acc)):
                raise ProjectError(f"{path}: sample {stem}: {acc!r} is not an IGVF file accession")
        samples.append({
            "stem": stem,
            "dataset": f"{p['prefix']}{stem}",
            "fragments": frag,
            "accession": accession,
            "md5": s.get("md5"),
            "index": s.get("index"),
            "extra": extra,
            "label": s.get("label", stem),
        })  # fmt: skip
    p["samples"] = samples
    p["datasets"] = [s["dataset"] for s in samples]
    p["cooperativity"] = _cooperativity(path, p.get("cooperativity") or {}, samples)
    return p


def _cooperativity(path: Path, raw: dict, samples: list[dict]) -> dict:
    """groups (name -> datasets) and comparisons ([name, group_a, group_b]). Groups are given by stem; by default one
    group per sample, named by its stem."""
    by_stem = {s["stem"]: s["dataset"] for s in samples}
    groups_raw = raw.get("groups") or {s["stem"]: [s["stem"]] for s in samples}
    groups = {}
    for name, stems in groups_raw.items():
        stems = [str(x) for x in (stems if isinstance(stems, list) else [stems])]
        missing = [x for x in stems if x not in by_stem]
        if missing:
            raise ProjectError(f"{path}: cooperativity group {name}: unknown stems {missing}")
        groups[str(name)] = [by_stem[x] for x in stems]
    comparisons = []
    for c in raw.get("comparisons") or []:
        if not (isinstance(c, list) and len(c) == 3):
            raise ProjectError(f"{path}: a comparison is [name, group_a, group_b], got {c!r}")
        name, a, b = (str(x) for x in c)
        for g in (a, b):
            if g not in groups:
                raise ProjectError(f"{path}: comparison {name}: no group {g}")
        comparisons.append([name, a, b])
    names = [c[0] for c in comparisons]
    if len(set(names)) != len(names):
        raise ProjectError(f"{path}: comparison names repeat: {names}")
    series = [str(x) for x in raw.get("series") or []]
    if series and (len(series) < 3 or len(set(series)) != len(series) or not set(series) <= set(groups)):
        raise ProjectError(f"{path}: cooperativity series must be 3 or more distinct groups, got {series}")
    return {"groups": groups, "comparisons": comparisons, "series": series,
            **{k: v for k, v in raw.items() if k not in ("groups", "comparisons", "series")}}  # fmt: skip


def _vendored_references():
    """The vendored igvf_tf_chrombpnet's lib/python/utils/references.py (standard library only), the module its steps
    resolve every reference path with."""
    spec = importlib.util.spec_from_file_location("igvf_references", VENDOR / "lib/python/utils/references.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


REFERENCE_FILES = ("genome_fa", "chrom_sizes", "chrom_sizes_main", "blacklist", "ref_db_meme")
"""The reference files the steps the driver runs read (the .fai beside genome_fa too); tss_bed is optional (02.0's TSS
enrichment)."""


def reference_files(reference_root: Path) -> dict[str, str]:
    lay = _vendored_references().layout(str(reference_root))
    return {k: lay[k] for k in (*REFERENCE_FILES, "tss_bed")}


# ---------------------------------------------------------------------------------------------------------------------
# the machine the box runs on, and the slots it can hold


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def machine() -> dict:
    """The CPUs this process may use, the GPUs (CUDA_VISIBLE_DEVICES when it lists device numbers, else every GPU
    nvidia-smi lists), and the memory in GB (the cgroup's limit when there is one, else MemTotal)."""
    cpus = len(os.sched_getaffinity(0))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and re.fullmatch(r"\d+(,\d+)*", visible.strip()):
        gpus = visible.strip().split(",")
    else:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True,
                                 text=True, check=False, timeout=60).stdout  # fmt: skip
        except (OSError, subprocess.SubprocessError):
            out = ""
        gpus = [x.strip() for x in out.splitlines() if x.strip().isdigit()]
    mem_bytes = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                mem_bytes = int(line.split()[1]) * 1024
    except OSError:
        pass
    for limit in (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")):
        try:
            text = limit.read_text().strip()
        except OSError:
            continue
        if text.isdigit() and (mem_bytes is None or int(text) < mem_bytes):
            mem_bytes = int(text)
    return {"cpus": cpus, "gpus": gpus, "mem_gb": (mem_bytes or 0) // 2**30}


def resolve_slots(given: dict, m: dict, n_datasets: int) -> dict[str, int]:
    """The slots for machine m: SLOTS' values, each scaled down so the stages fit the machine's CPUs and memory; a
    value project.yaml gives is kept as it is. On a whole Gefion node (8 GPUs, 216 CPUs, ~2 TB) this is SLOTS itself."""
    c, g, mem = max(1, m["cpus"]), max(1, len(m["gpus"])), m["mem_gb"]
    s = dict(given)

    def pick(key, value):
        s.setdefault(key, value)

    pick("train_per_gpu", SLOTS["train_per_gpu"])
    pick("train_cores", _clamp(c // (g * s["train_per_gpu"]), 1, SLOTS["train_cores"]))
    pick("attribute_per_gpu", SLOTS["attribute_per_gpu"])
    pick("attribute_cores", _clamp(c // (g * s["attribute_per_gpu"]), 1, SLOTS["attribute_cores"]))
    pick("modisco_max", _clamp(min(mem // MODISCO_GB, c // SLOTS["modisco_threads"]), 1, SLOTS["modisco_max"]))
    pick("modisco_threads", _clamp(c // s["modisco_max"], 1, SLOTS["modisco_threads"]))
    pick("finemo_slots", _clamp(g, 1, SLOTS["finemo_slots"]))
    pick("finemo_batch", SLOTS["finemo_batch"])
    pick("coop_cores", _clamp(c // g, 1, SLOTS["coop_cores"]))
    pick("prep_max", _clamp(min(mem // PREP_GB, c // 8), 1, max(1, n_datasets)))
    if s["prep_max"] == 0:
        s["prep_max"] = max(1, n_datasets)
    return {k: int(s[k]) for k in SLOTS}


def sbatch_args(p: dict) -> list[str]:
    """The slurm: block as sbatch options (submit.sh passes them on the command line, which beats the #SBATCH lines
    of run_project.sbatch). Every key is optional: what is not given is the cluster's default."""
    s = p["slurm"]
    out = []
    if s.get("partition"):
        out.append(f"--partition={s['partition']}")
    if s.get("account"):
        out.append(f"--account={s['account']}")
    if s.get("gpus"):
        out.append(f"--gres=gpu:{int(s['gpus'])}")
    if s.get("cpus"):
        out.append(f"--cpus-per-task={int(s['cpus'])}")
    if s.get("time"):
        out.append(f"--time={s['time']}")
    extra = s.get("extra") or []
    out += shlex.split(extra) if isinstance(extra, str) else [str(x) for x in extra]
    return out


# ---------------------------------------------------------------------------------------------------------------------
# paths


def paths(p: dict, ds: str | None = None) -> dict[str, Path]:
    root = Path(p["root"])
    out = {
        "root": root,
        "fragments_dir": root / "fragments",
        "portal": root / "portal",
        "configs_dir": root / "chrombpnet" / "configs",
        "shifts": root / "chrombpnet" / "shifts.json",
        "cheri_root": root / "cherimoya",
        "shared": root / "cherimoya" / "shared.txt",
        "compendia": root / "compendia",
        "compendium": root / "compendia" / p["compendium"],
        "inputs": root / "compendia" / "inputs",
        "coop": root / "cherimoya" / "cooperativity",
        "coop_matrix": root / "cherimoya" / "cooperativity" / "matrix",
        "box": root / "box",
    }
    out["motif_list"] = out["inputs"] / f"cheri{len(p['datasets'])}.tsv"  # make_downstream.py's cheri<D>.tsv
    out["dataset_list"] = out["inputs"] / f"rest{len(p['datasets'])}.txt"  # every dataset (no shared ones)
    if ds is not None:
        out.update(
            config=out["configs_dir"] / ds / "config.yaml",
            narrowpeak=root / "peaks" / ds / f"{ds}.narrowPeak.gz",
            output_dir=root / "chrombpnet" / ds,
            bigwig=root / "chrombpnet" / ds / "preprocessing" / "signal" / "data_unstranded.bw",
            peaks=root / "chrombpnet" / ds / "preprocessing" / "peaks" / f"{ds}_all_peaks_no_blacklist.narrowPeak",
            signal_qc=root / "chrombpnet" / ds / "plots" / "signal_qc" / f"{ds}_signal_qc.json",
            cheri=out["cheri_root"] / ds,
            downstream=out["cheri_root"] / ds / "downstream",
            finemo=out["cheri_root"] / ds / "downstream" / f"finemo_{p['compendium']}" / f"{ds}_all",
            coop_args=out["coop"] / "args" / f"{ds}.args",
            coop_run=out["coop"] / ds / "run.json",
        )
    return out


def fragments_file(p: dict, sample: dict) -> Path:
    return paths(p)["fragments_dir"] / f"{sample['stem']}{FRAGMENTS_SUFFIX}"


def negatives(p: dict, ds: str, fold: str) -> Path:
    return Path(p["root"]) / "chrombpnet" / ds / "preprocessing" / ds / f"output_all_fold_{fold}_negatives.bed"


# ---------------------------------------------------------------------------------------------------------------------
# stage checks: (done, what is missing)


def _missing(files) -> list[str]:
    out = []
    for f in files:
        f = Path(f)
        if not (f.is_file() and f.stat().st_size > 0) and not (f.is_dir() and any(f.iterdir())):
            out.append(str(f))
    return out


def needed(p: dict, stage: str) -> list[Path]:
    """The files and directories the stage's check reads. For attribute, check also has attr_status read each fold's
    attribution files through its attribute.json; prune.needed adds those. prune.py deletes none of them."""
    if stage not in STAGES:
        raise ProjectError(f"unknown stage {stage}; stages are {list(STAGES)}")
    need: list = []
    if stage == "fetch":
        need = [fragments_file(p, s) for s in p["samples"]]
    for ds in p["datasets"]:
        q = paths(p, ds)
        if stage == "prep":
            need += [q["config"], q["narrowpeak"], q["bigwig"], q["peaks"], q["signal_qc"]]
            need += [negatives(p, ds, k) for k in p["folds"]]
        elif stage == "train":
            for k in p["folds"]:
                fold = q["cheri"] / f"fold_{k}"
                need += [fold / f"cherimoya.fold_{k}.torch", fold / f"cherimoya.fold_{k}.test.performance.tsv"]
        elif stage == "attribute":
            need += [q["cheri"] / f"fold_{k}" / "attribute.json" for k in p["folds"]]
        elif stage == "downstream":
            need += [q["downstream"] / ".done" / m for m in ("06.0", "v2", "07.0", "08.0")]
        elif stage == "motifs":
            need += [q["finemo"] / f for f in ("hits_unique.tsv", "peaks_qc.tsv", "compendium.md5")]
            need += [q["finemo"] / "finemo_report"]
        elif stage == "cooperativity":
            need.append(q["coop_run"])
    if stage == "motifs":
        need.insert(0, paths(p)["compendium"] / "compendium.json")
    if stage == "cooperativity":
        need += [paths(p)["coop"] / "combined" / "summary.json", paths(p)["coop"] / "official" / "summary.json"]
    if stage == "coop_matrix":
        need.append(paths(p)["coop_matrix"] / "combined" / "summary.json")
    return [Path(x) for x in need]


def check(p: dict, stage: str) -> list[str]:
    """What the stage has not produced yet ([] when done)."""
    missing = _missing(needed(p, stage))
    if stage == "coop_matrix" and not missing and p["cooperativity"]["series"]:
        summary = json.loads((paths(p)["coop"] / "combined" / "summary.json").read_text())
        if "trend_pairs" not in summary:
            missing.append(f"{paths(p)['coop'] / 'combined' / 'summary.json'}: no trends (combine.py --series)")
    if stage == "attribute" and not missing:
        import attr_status  # stdlib only: npy/npz headers, no numpy

        for ds in p["datasets"]:
            for k in p["folds"]:
                fold = paths(p, ds)["cheri"] / f"fold_{k}"
                state, reasons = attr_status.check_fold(fold)
                if state != "done":
                    missing.append(f"{fold}: {state}: {'; '.join(reasons)[:300]}")
    return missing


# ---------------------------------------------------------------------------------------------------------------------
# outputs for the box


def env_lines(p: dict, m: dict | None = None) -> list[str]:
    """Shell assignments for run_project.sh, with the slots for machine m (default: this one)."""
    if not p["reference_root"]:
        raise ProjectError(f"{p['file']}: reference_root is required to run the project (docs/install.md)")
    m = machine() if m is None else m
    q = paths(p)
    refs = p["references"]
    values = {
        "PROJECT": p["project"],
        "PROJECT_FILE": p["file"],
        "PROJECT_ROOT": p["root"],
        "PROJECT_PREFIX": p["prefix"],
        "PROJECT_DATASETS": " ".join(p["datasets"]),
        "PROJECT_N_DATASETS": str(len(p["datasets"])),
        "PROJECT_FOLDS": " ".join(p["folds"]),
        "PROJECT_STAGES": " ".join(p["stages"]),
        "PROJECT_GENOME": p["genome"],
        "PROJECT_FOLDS_DIR": p["folds_dir"],
        "PROJECT_REFERENCE_ROOT": p["reference_root"],
        "PROJECT_REFERENCE_FILES": " ".join([refs[k] for k in REFERENCE_FILES] + [refs["genome_fa"] + ".fai"]),
        "PROJECT_VENDOR": str(VENDOR),
        "PROJECT_GPUS": ",".join(m["gpus"]),
        "PROJECT_CPUS": str(m["cpus"]),
        "PROJECT_MEM_GB": str(m["mem_gb"]),
        "PROJECT_COMPENDIUM": p["compendium"],
        "PROJECT_HOLD_MIN": str(int(p["hold_minutes"])),
        "PROJECT_SHIFT_EVERY": str(int(p["shift_every"])),
        **{f"PROJECT_{k.upper()}": str(v) for k, v in q.items()},
        **{f"SLOT_{k.upper()}": str(v) for k, v in resolve_slots(p["slots"], m, len(p["datasets"])).items()},
    }
    return [f"{k}={shlex.quote(v)}" for k, v in values.items()]


def write_coop_inputs(p: dict) -> list[Path]:
    """Per dataset coop_isa.py's args file (lane.sh reads one argument per line), and combine.py's groups.tsv,
    comparisons.tsv and, with a series, series.tsv."""
    if not p["genome"]:
        raise ProjectError(f"{p['file']}: reference_root is required to write the cooperativity inputs")
    q = paths(p)
    written = []
    (q["coop"] / "args").mkdir(parents=True, exist_ok=True)
    for ds in p["datasets"]:
        d = paths(p, ds)
        args = ["--dataset", ds,
                "--models", str(d["cheri"] / "fold_{fold}" / "cherimoya.fold_{fold}.torch"),
                "--finemo-dir", str(d["finemo"]),
                "--motifs", str(q["coop"] / "motifs.tsv"),
                "--fasta", p["genome"],
                "--out-dir", str(q["coop"]),
                "--folds", *p["folds"]]  # fmt: skip
        d["coop_args"].write_text("\n".join(args) + "\n")
        written.append(d["coop_args"])
    groups = ["group\tdatasets"] + [f"{g}\t{','.join(ds)}" for g, ds in p["cooperativity"]["groups"].items()]
    (q["coop"] / "groups.tsv").write_text("\n".join(groups) + "\n")
    comparisons = ["name\tgroup_a\tgroup_b"] + ["\t".join(c) for c in p["cooperativity"]["comparisons"]]
    (q["coop"] / "comparisons.tsv").write_text("\n".join(comparisons) + "\n")
    written += [q["coop"] / "groups.tsv", q["coop"] / "comparisons.tsv"]
    if p["cooperativity"]["series"]:  # the groups in order, for combine.py's and paired.py's --series
        (q["coop"] / "series.tsv").write_text("\n".join(["group", *p["cooperativity"]["series"]]) + "\n")
        written.append(q["coop"] / "series.tsv")
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project", type=Path, help="the project's project.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("show")
    sub.add_parser("env")
    c = sub.add_parser("check")
    c.add_argument("stage", choices=STAGES)
    sub.add_parser("status")
    sub.add_parser("coop-inputs")
    sub.add_parser("sbatch-args")
    pr = sub.add_parser("prune", help="dry run: the manifest of what would be deleted; --apply deletes it")
    pr.add_argument("--apply", type=Path, metavar="MANIFEST", help="a manifest a dry run wrote, read by a person")
    sh = sub.add_parser("share", help="pack the results into one tar")
    sh.add_argument("--out", type=Path, help="the tar to write (required unless --list)")
    sh.add_argument("--datasets", help="comma-separated (default: every dataset)")
    sh.add_argument("--with-signal", action="store_true", help="add each dataset's observed signal bigWig")
    sh.add_argument("--list", action="store_true", help="print what would go in, and write nothing")
    args = ap.parse_args(argv)
    try:
        p = load(args.project)
    except (ProjectError, OSError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.cmd in ("env", "coop-inputs", "sbatch-args") and not p["driver"]:
        print(f"ERROR: {p['file']} says driver: false: this project is not run by the driver (run_project.sh, "
              "submit.sh); its README says how it was made", file=sys.stderr)  # fmt: skip
        return DRIVER_REFUSED
    if args.cmd == "prune":
        import prune

        try:
            if args.apply:
                c = prune.apply(p, args.apply)
                print(f"deleted {c['deleted']:,} files ({c['bytes'] / 2**30:,.1f} GiB) and {c['dirs']:,} empty "
                      f"directories; gone already {c['gone']}, changed since the dry run {c['changed']}, refused "
                      f"{c['refused']}; log {c['log']}")  # fmt: skip
                return 0 if not (c["changed"] or c["refused"]) else 1
            pl = prune.plan(p)
            prune.print_plan(pl, prune.write_plan(p, pl))
        except (prune.PruneError, OSError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        return 0
    if args.cmd == "share":
        import share

        datasets = p["datasets"] if not args.datasets else args.datasets.split(",")
        unknown = sorted(set(datasets) - set(p["datasets"]))
        if unknown:
            print(f"ERROR: not datasets of this project: {unknown}", file=sys.stderr)
            return 2
        files = share.share_set(p, datasets, args.with_signal)
        if args.list:
            root = Path(p["root"])
            size = 0
            for f in files:
                there = f.is_file()
                size += f.stat().st_size if there else 0
                print(f"{'' if there else 'MISSING '}{f.relative_to(root)}")
            print(f"{sum(f.is_file() for f in files)} files, {size / 2**30:,.2f} GiB")
            return 0
        if not args.out:
            print("ERROR: share needs --out <file.tar> (or --list)", file=sys.stderr)
            return 2
        r = share.write_share(p, files, args.out)
        print(f"wrote {r['out']}: {r['files']} files, {r['bytes'] / 2**30:,.2f} GiB; missing {len(r['missing'])}")
        for m in r["missing"][:20]:
            print(f"  missing: {m}")
        return 0
    if args.cmd == "show":
        print(json.dumps(p, indent=1))
    elif args.cmd == "env":
        try:
            print("\n".join(env_lines(p)))
        except ProjectError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
    elif args.cmd == "sbatch-args":
        print("\n".join(sbatch_args(p)))
    elif args.cmd == "check":
        missing = check(p, args.stage)
        for m in missing[:50]:
            print(f"missing: {m}")
        if len(missing) > 50:
            print(f"... and {len(missing) - 50} more")
        return 0 if not missing else 1
    elif args.cmd == "status":
        for stage in STAGES:
            missing = check(p, stage)
            print(f"{stage}\t{'done' if not missing else f'{len(missing)} missing'}")
    elif args.cmd == "coop-inputs":
        try:
            written = write_coop_inputs(p)
        except ProjectError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        for f in written:
            print(f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
