#!/usr/bin/env python3
"""One Mímir project: its project.yaml, the paths every stage writes, and whether each stage is done.

run_project.sh runs a project's stages in one box; this module is what it asks. A project is one directory (root) and
one project.yaml (kept in it; project.example.yaml in this folder documents every key). Every path below is derived
here, once, from root, prefix and the sample stems:

  <root>/fragments/<stem>.fragments.tsv.gz       the inputs (fetch_igvf.py writes them; igvf_tf_chrombpnet's config
                                                 check accepts this suffix, not .bed.gz)
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

Needs only the standard library and PyYAML (the default pixi env), so it runs on the login node.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "attribution"))

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
}
DEFAULTS = {
    "genome": "/dcai/projects/iu_0109/annotations/hg38/Sequence/IGVFFI0653VCGH.fasta",
    "igvf": "/dcai/users/mateug/git/igvf_tf_chrombpnet",
    "chrombpnet": "/dcai/users/mateug/git/chrombpnet",
    "chrombpnet_env": "cuda13",
    "stages": list(BOX_STAGES),
    "hold_minutes": 60,
    "shift_every": 100,
}


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
    known = {"project", "root", "prefix", "samples", "compendium", "cooperativity", "slots", "folds", "folds_dir",
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
    root = Path(p["root"])
    if not root.is_absolute():
        raise ProjectError(f"{path}: root must be an absolute path")
    p["root"] = str(root)
    p.setdefault("folds_dir", str(Path(p["igvf"]) / "folds"))
    p["folds"] = [str(f) for f in p.get("folds") or DEFAULT_FOLDS]
    stages = list(p["stages"])
    bad = [s for s in stages if s not in STAGES[1:]]
    if bad:
        raise ProjectError(f"{path}: unknown stages {bad}; the box stages are {list(STAGES[1:])}")
    p["stages"] = [s for s in STAGES[1:] if s in stages]  # always in pipeline order
    p["slots"] = {**SLOTS, **(p.get("slots") or {})}
    bad = sorted(set(p["slots"]) - set(SLOTS))
    if bad:
        raise ProjectError(f"{path}: unknown slots {bad}")

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


def check(p: dict, stage: str) -> list[str]:
    """What the stage has not produced yet ([] when done)."""
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
    missing = _missing(need)
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


def env_lines(p: dict) -> list[str]:
    q = paths(p)
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
        "PROJECT_IGVF": p["igvf"],
        "PROJECT_CHROMBPNET": p["chrombpnet"],
        "PROJECT_CHROMBPNET_ENV": p["chrombpnet_env"],
        "PROJECT_COMPENDIUM": p["compendium"],
        "PROJECT_HOLD_MIN": str(int(p["hold_minutes"])),
        "PROJECT_SHIFT_EVERY": str(int(p["shift_every"])),
        **{f"PROJECT_{k.upper()}": str(v) for k, v in q.items()},
        **{f"SLOT_{k.upper()}": str(int(v)) for k, v in p["slots"].items()},
    }
    return [f"{k}={shlex.quote(v)}" for k, v in values.items()]


def write_coop_inputs(p: dict) -> list[Path]:
    """Per dataset coop_isa.py's args file (lane.sh reads one argument per line), and combine.py's groups.tsv,
    comparisons.tsv and, with a series, series.tsv."""
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
    args = ap.parse_args(argv)
    try:
        p = load(args.project)
    except (ProjectError, OSError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.cmd == "show":
        print(json.dumps(p, indent=1))
    elif args.cmd == "env":
        print("\n".join(env_lines(p)))
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
        for f in write_coop_inputs(p):
            print(f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
