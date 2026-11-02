#!/usr/bin/env python3
"""A project's STATUS.md: where every stage stands, and each dataset's headline numbers, read from its outputs.

run_project.sh writes it as its last stage; it can be run any time, also mid-run, also on the login node (standard
library and PyYAML only; it reads files, never imports numpy). Per dataset:

  shift        the Tn5 shift detect_shifts.py found in the fragments (plus/minus; 10x files: +4/-5)
  peaks        rows of the called narrowPeak (the igvf pseudobulking recipe)
  counts r     the mean over folds of Cherimoya's test count Pearson (cherimoya.fold_<k>.test.performance.tsv)
  profile JSD  the same for the median profile JSD
  attribution  folds attributed / folds, and how many took `cherimoya attribute`'s fast path
  hits         Fi-NeMo hits (hits_unique.tsv rows) against the project's compendium
  DeepISA      site pairs, our synergy score sum(i > 0.1) / sum(|i| > 0.1) over all of them, and the official
               package's coop score sum(i)/sum(|i|) over its qualified pairs

    status.py <project.yaml> [--out <root>/STATUS.md]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import project as pj  # noqa: E402


def et() -> str:
    os.environ["TZ"] = "America/New_York"
    time.tzset()
    return time.strftime("%Y-%m-%d %H:%M %Z")


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def count_rows(path: Path, header: bool = False) -> int | None:
    try:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as fh:
            n = sum(1 for line in fh if line.strip())
        return n - 1 if header else n
    except OSError:
        return None


def performance(path: Path) -> dict | None:
    try:
        with open(path) as fh:
            rows = list(csv.DictReader(fh, delimiter="\t"))
        return {k: float(v) for k, v in rows[0].items() if v not in ("", "nan")}
    except (OSError, IndexError, ValueError):
        return None


def fmt(x, spec: str = ".3f") -> str:
    return "–" if x is None else format(x, spec)


def latest_box(p: dict) -> Path | None:
    boxes = [b for b in pj.paths(p)["box"].glob("*") if (b / "stages.tsv").is_file()]
    return max(boxes, key=lambda b: (b / "stages.tsv").stat().st_mtime) if boxes else None


def build(p: dict) -> str:
    q = pj.paths(p)
    lines = [f"# {p['project']}: status", "", f"Written {et()} by regulatory-variant-interpretation "
             "workflows/pipeline/status.py, from the outputs under this directory.", ""]  # fmt: skip
    lines += ["## Stages", "", "| stage | state |", "|---|---|"]
    for stage in pj.STAGES:
        missing = pj.check(p, stage)
        lines.append(f"| {stage} | {'done' if not missing else f'{len(missing)} output(s) missing'} |")
    box = latest_box(p)
    if box is not None:
        lines += ["", f"Last box: `{box.name}` ({box / 'stages.tsv'})", "", "| stage | state | minutes | started |",
                  "|---|---|---|---|"]  # fmt: skip
        with open(box / "stages.tsv") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                lines.append(f"| {r['stage']} | {r['state']} | {int(r['seconds']) / 60:.0f} | {r['started_et']} |")
        tests = {}
        for name, log in (("preflight: pixi run check", "preflight_check.log"),
                          ("preflight: test-pipeline (required)", "preflight_test_pipeline.log"),
                          ("every test in the cherimoya env (reported, not required)", "test_cherimoya_full.log")):  # fmt: skip
            path = box / "logs" / log
            if path.is_file():
                tail = [x for x in path.read_text(errors="replace").splitlines() if x.strip()][-2:]
                tests[name] = " / ".join(tail) if tail else "(empty)"
        if tests:
            lines += ["", "Tests in that box:", ""] + [f"- {k}: `{v}`" for k, v in tests.items()]

    shifts = read_json(q["shifts"]) or {}
    off_summary = read_json(q["coop"] / "official" / "summary.json") or {}
    lines += ["", "## Datasets", "",
              "| dataset | shift | peaks | counts r | profile JSD | attribution (fast) | hits | DeepISA pairs | "
              "synergy (ours) | coop (official) |", "|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for s in p["samples"]:
        ds = s["dataset"]
        d = pj.paths(p, ds)
        sh = shifts.get(s["stem"]) or {}
        shift = f"{sh['plus_shift']:+d}/{sh['minus_shift']:+d}" if "plus_shift" in sh else "–"
        perf = [performance(d["cheri"] / f"fold_{k}" / f"cherimoya.fold_{k}.test.performance.tsv") for k in p["folds"]]
        perf = [x for x in perf if x]
        r = sum(x.get("count_pearson", 0) for x in perf) / len(perf) if perf else None
        jsd = sum(x.get("profile_jsd", 0) for x in perf) / len(perf) if perf else None
        stock = [read_json(d["cheri"] / f"fold_{k}" / "attr.attr.stock.json") for k in p["folds"]]
        stock = [x for x in stock if x]
        fast = sum(1 for x in stock if x.get("fast_path"))
        hits = count_rows(d["finemo"] / "hits_unique.tsv", header=True)
        run = read_json(d["coop_run"]) or {}
        off = (off_summary.get(ds) or {}).get("coop_score")
        lines.append(
            f"| {ds} | {shift} | {fmt(count_rows(d['narrowpeak']), ',')} | {fmt(r)} | {fmt(jsd)} | "
            f"{len(stock)}/{len(p['folds'])} ({fast}) | {fmt(hits, ',')} | {fmt(run.get('pairs'), ',')} | "
            f"{fmt(run.get('synergy_all'), '.2f')} | {fmt(off, '.2f')} |"
        )

    cj = read_json(q["compendium"] / "compendium.json")
    motifs = q["coop"] / "motifs.tsv"
    if cj is not None or motifs.is_file():
        n = count_rows(motifs, header=True)
        named = None
        if motifs.is_file():
            with open(motifs) as fh:
                named = sum(1 for r in csv.DictReader(fh, delimiter="\t") if r.get("cluster_annotation"))
        lines += ["", "## Compendium", "", f"`{q['compendium']}`: {fmt(n, ',')} motifs ({fmt(named, ',')} annotated)."]

    comb = read_json(q["coop"] / "combined" / "summary.json") or {}
    if comb or off_summary:
        lines += ["", "## DeepISA: changes between groups", "",
                  "TF pairs (and TFs) whose synergy changes, with a bootstrap 95% CI over peaks that excludes 0. "
                  "Ours: synergy = sum(i > 0.1) / sum(|i| > 0.1); official: sum(i) / sum(|i|) over the package's "
                  "qualified pairs (workflows/cooperativity/README.md).", "",
                  "| comparison | ours: pairs | ours: TFs | official: pairs | official: TFs |", "|---|---|---|---|---|"]  # fmt: skip
        for name, _, _ in p["cooperativity"]["comparisons"]:
            cells = []
            for src in (comb, off_summary):
                for level in ("changes_pairs", "changes_tfs"):
                    c = (src.get(level) or {}).get(name)
                    cells.append(f"{c['significant']}/{c['tested']}" if c else "–")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project", type=Path)
    ap.add_argument("--out", type=Path, help="default: print")
    args = ap.parse_args(argv)
    p = pj.load(args.project)
    text = build(p)
    if args.out:
        tmp = args.out.with_name(args.out.name + ".tmp")
        tmp.write_text(text)
        tmp.replace(args.out)
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
