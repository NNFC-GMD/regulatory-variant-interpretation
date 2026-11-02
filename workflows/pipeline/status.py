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
               package's coop score sum(i)/sum(|i|) over its qualified pairs; beside the model's accuracy, the
               effect sizes (median intact prediction a, median |e| and |i|), since a weaker model predicts smaller
               effects and fewer pairs pass 0.1
Then the changes between groups (BH q < 0.05; robust: every fold agrees too), the trends along the series, and the
coop_matrix stage's paired results (every dataset's sites under every dataset's models).

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
    for stage in ("fetch", *p["stages"]):
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
    matrix = read_json(q["coop_matrix"] / "combined" / "summary.json") or {}
    if comb:
        open_ = matrix.get("open") or {}
        lines += ["", "## DeepISA effect sizes and model accuracy", "",
                  "A weaker model predicts smaller effects, so fewer of its pairs pass |i| > 0.1: read the synergy "
                  "beside these. Log2 counts; open windows: those whose intact prediction reaches the model's own "
                  "10th percentile, over every dataset's sites (coop_matrix).", "",
                  "| dataset | counts r | median a | median abs(e) | median abs(i) | synergy (ours) | open windows |",
                  "|---|---|---|---|---|---|---|"]  # fmt: skip
        for s in p["samples"]:
            ds = s["dataset"]
            d = pj.paths(p, ds)
            perf = [
                performance(d["cheri"] / f"fold_{k}" / f"cherimoya.fold_{k}.test.performance.tsv") for k in p["folds"]
            ]
            perf = [x for x in perf if x]
            r = sum(x.get("count_pearson", 0) for x in perf) / len(perf) if perf else None
            c, o = comb.get(ds) or {}, open_.get(ds) or {}
            lines.append(f"| {ds} | {fmt(r)} | {fmt(c.get('median_a'), '.2f')} | {fmt(c.get('median_abs_e'))} | "
                         f"{fmt(c.get('median_abs_i'))} | {fmt(c.get('synergy'), '.2f')} | "
                         f"{fmt(o.get('open'), ',')} / {fmt(o.get('windows'), ',')} |")  # fmt: skip
    if comb or off_summary:
        lines += ["", "## DeepISA: changes between groups, each dataset's own sites", "",
                  "TF pairs (and TFs) whose score changes: significant (robust) / tested, significant = BH q < 0.05 "
                  "within the comparison (bootstrap SE over peaks), robust = significant and the change has the same "
                  "sign under every fold's models. Ours: synergy = sum(i > 0.1) / sum(|i| > 0.1); official: sum(i) / "
                  "sum(|i|) over the package's qualified pairs (workflows/cooperativity/README.md).", "",
                  "| comparison | ours: pairs | ours: TFs | official: pairs | official: TFs |", "|---|---|---|---|---|"]  # fmt: skip
        for name, _, _ in p["cooperativity"]["comparisons"]:
            cells = []
            for src in (comb, off_summary):
                for level in ("changes_pairs", "changes_tfs"):
                    cells.append(sig_cell((src.get(level) or {}).get(name)))
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
    series = p["cooperativity"]["series"]
    if series and (comb.get("trend_pairs") or off_summary.get("trend_pairs") or matrix.get("series")):
        ms = matrix.get("series") or {}
        lines += ["", f"## DeepISA: trends along {' -> '.join(series)}", "",
                  "TF pairs (and TFs) with a slope over the series: significant (robust) / tested (BH q < 0.05; "
                  "robust: every fold's slope has the same sign). Own sites: each dataset's sites under its models; "
                  "same sites: every dataset's sites, open under every model, paired bootstrap; mean i: the slope "
                  "of the mean interaction, no threshold.", "",
                  "| level | ours, own sites | official, own sites | ours, same sites | mean i, same sites |",
                  "|---|---|---|---|---|"]  # fmt: skip
        for level in ("pairs", "tfs"):
            m = ms.get(level) or {}
            cells = [sig_cell(comb.get(f"trend_{level}")), sig_cell(off_summary.get(f"trend_{level}")),
                     sig_cell(m.get("synergy")), sig_cell(m.get("slope_i"), "slope_i_significant", "slope_i_robust")]  # fmt: skip
            lines.append(f"| {'TF pairs' if level == 'pairs' else 'TFs'} | " + " | ".join(cells) + " |")
    if matrix.get("comparisons"):
        lines += ["", "## DeepISA: changes between groups, the same sites under both groups' models", "",
                  "Every dataset's sites, the pairs whose window is open under both groups' models, a paired "
                  "bootstrap (coop_matrix; workflows/cooperativity/paired.py). Synergy as above; delta i: the mean "
                  "change of the interaction itself. significant (robust) / tested.", "",
                  "| comparison | pairs open | synergy: pairs | synergy: TFs | delta i: pairs | delta i: TFs |",
                  "|---|---|---|---|---|---|"]  # fmt: skip
        for name, _, _ in p["cooperativity"]["comparisons"]:
            c = matrix["comparisons"].get(name) or {}
            cells = [fmt(c.get("pairs_open"), ",")]
            cells += [sig_cell((c.get(level) or {}).get("synergy")) for level in ("pairs", "tfs")]
            cells += [sig_cell((c.get(level) or {}).get("delta_i"), "delta_i_significant", "delta_i_robust")
                      for level in ("pairs", "tfs")]  # fmt: skip
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def sig_cell(c: dict | None, sig: str = "significant", robust: str = "robust") -> str:
    """significant (robust) / tested, from a summary.json count."""
    if not c or "tested" not in c:
        return "–"
    rob = f" ({c[robust]})" if robust in c else ""
    return f"{c.get(sig, 0)}{rob}/{c['tested']}"


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
