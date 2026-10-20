#!/usr/bin/env python3
"""Read make_manifest.py's manifest.json for the shell runners: datasets, folds, model files, peaks, genome.

The manifest (make_manifest.py) holds, per dataset, `peaks.path` and, per fold, `chroms.test` and
`models.{cherimoya,cbp_nobias,cbp_bias,cbp_full}.path`, plus `genome.path` at the top, `problems` and `ok`.

  check    the manifest is `ok`, lists no problems, and every model, peaks and genome file it names exists
           (exit 1 otherwise, with the reasons)
  tasks    one TSV row per dataset and fold:
             dataset  fold  cherimoya  cbp_nobias  cbp_bias  cbp_full  test_chroms (comma-separated)
  peaks    one TSV row per dataset: dataset  peaks
  genome   the genome FASTA's path

    python workflows/variant_scoring/read_manifest.py check <root>/manifest.json
    python workflows/variant_scoring/read_manifest.py tasks <root>/manifest.json [--datasets A B] [--folds 0 1]

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MODELS = ("cherimoya", "cbp_nobias", "cbp_bias", "cbp_full")
SCHEMA = """expected (make_manifest.py):
  {"genome": {"path": ...}, "ok": true, "problems": [],
   "datasets": {"<dataset>": {"peaks": {"path": ...},
                              "folds": {"<k>": {"chroms": {"test": [...]},
                                                "models": {"cherimoya": {"path": ...}, "cbp_nobias": {"path": ...},
                                                           "cbp_bias": {"path": ...}, "cbp_full": {"path": ...}}}}}}}"""


class ManifestError(ValueError):
    pass


def load(path: Path) -> dict:
    try:
        manifest = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as err:
        raise ManifestError(f"{path}: {err}") from err
    if not isinstance(manifest.get("datasets"), dict) or not manifest["datasets"]:
        raise ManifestError(f"{path}: no datasets\n{SCHEMA}")
    return manifest


def _path(entry, where: str) -> str:
    if not isinstance(entry, dict) or not entry.get("path"):
        raise ManifestError(f"{where}: no path\n{SCHEMA}")
    return str(entry["path"])


def rows(manifest: dict, datasets: list[str] | None = None, folds: list[str] | None = None) -> list[dict]:
    """One dict per (dataset, fold): dataset, fold, the four model paths, peaks, test_chroms; in manifest order."""
    out = []
    unknown = sorted(set(datasets or []) - set(manifest["datasets"]))
    if unknown:
        raise ManifestError(f"datasets not in the manifest: {', '.join(unknown)}")
    for ds, entry in manifest["datasets"].items():
        if datasets and ds not in datasets:
            continue
        peaks = _path(entry.get("peaks"), f"{ds}: peaks")
        fold_entries = entry.get("folds")
        if not isinstance(fold_entries, dict) or not fold_entries:
            raise ManifestError(f"{ds}: no folds\n{SCHEMA}")
        for fold, f in sorted(fold_entries.items(), key=lambda kv: int(kv[0])):
            if folds and str(fold) not in folds:
                continue
            models = f.get("models") or {}
            test = (f.get("chroms") or {}).get("test")
            if not test:
                raise ManifestError(f"{ds} fold {fold}: no chroms.test\n{SCHEMA}")
            out.append({"dataset": ds, "fold": str(fold),
                        **{m: _path(models.get(m), f"{ds} fold {fold}: models.{m}") for m in MODELS},
                        "peaks": peaks, "test_chroms": ",".join(test)})  # fmt: skip
    return out


def check(manifest: dict) -> list[str]:
    problems = [f"manifest problem: {p}" for p in manifest.get("problems", [])]
    if manifest.get("ok") is not True:
        problems.append("manifest is not ok")
    try:
        genome = _path(manifest.get("genome"), "genome")
        if not Path(genome).is_file() or not Path(f"{genome}.fai").is_file():
            problems.append(f"genome or its .fai missing: {genome}")
        seen = set()
        for row in rows(manifest):
            for key in (*MODELS, "peaks"):
                if row[key] not in seen and not Path(row[key]).is_file():
                    problems.append(f"{row['dataset']} fold {row['fold']}: {key} missing: {row[key]}")
                seen.add(row[key])
    except ManifestError as err:
        problems.append(str(err))
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("check", "tasks", "peaks", "genome"))
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--datasets", nargs="+", help="only these datasets (tasks, peaks)")
    ap.add_argument("--folds", nargs="+", help="only these folds (tasks)")
    args = ap.parse_args(argv)
    try:
        manifest = load(args.manifest)
        if args.command == "check":
            problems = check(manifest)
            for p in problems:
                print(f"ERROR: {p}", file=sys.stderr)
            if not problems:
                n = len(rows(manifest))
                print(f"{args.manifest}: ok, {len(manifest['datasets'])} dataset(s), {n} fold(s)")
            return 1 if problems else 0
        if args.command == "genome":
            print(_path(manifest.get("genome"), "genome"))
            return 0
        selected = rows(manifest, args.datasets, args.folds if args.command == "tasks" else None)
        if args.command == "peaks":
            seen = []
            for r in selected:
                if r["dataset"] not in seen:
                    seen.append(r["dataset"])
                    print(f"{r['dataset']}\t{r['peaks']}")
            return 0
        for r in selected:
            print("\t".join([r["dataset"], r["fold"], *(r[m] for m in MODELS), r["test_chroms"]]))
        return 0
    except ManifestError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
