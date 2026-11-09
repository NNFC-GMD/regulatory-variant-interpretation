#!/usr/bin/env python3
"""Delete a project's derived intermediates by rule, from a manifest a person has read first.

A finished project keeps its models, its canonical attributions and its results, and the files that let it be
rerun; the large intermediates that a later step derived and nothing reads again can go. Which is which is written
down here, once, as rules (docs/results.md, "What prune removes", explains each and how to regenerate it):

    project.py <project.yaml> prune                       dry run: writes <root>/prune/manifest.<stamp>.tsv (the
                                                          files it would delete: path, bytes, rule),
                                                          summary.<stamp>.tsv and unclassified.<stamp>.tsv, and
                                                          prints the totals per rule; deletes nothing
    project.py <project.yaml> prune --apply <manifest>    deletes the manifest's files, and only those, then the
                                                          directories left empty; logs deleted.<stamp>.tsv

Each file under <root> gets the first of:
  1. needed   a path some stage's `project.py check` reads (the fold attributions attr_status checks included):
              never deleted, whatever the rules say, so a pruned project still checks as done;
  2. keep     a keep rule matches;
  3. delete   a delete rule matches;
  4. unclassified: kept, and listed so a person can decide.
Rule sets: `pipeline` (every project), and for projects that have them `chrombpnet` (trained ChromBPNet models in
igvf_tf_chrombpnet's layout) and `variant_scoring` (workflows/variant_scoring/pilot_steps.sh's runs). project.yaml's
`prune:` block picks them and adds the project's own rules:

    prune:
      rulesets: [pipeline, chrombpnet, variant_scoring]   # default [pipeline]
      shared: compendia/inputs/shared10.txt                # datasets <shared> stands for (a list, or a file of names)
      keep: ["cherimoya/analyses/**/report*"]
      delete: [{pattern: "variant_scoring/tools/**", why: "a pixi env, rebuilt from its pixi.toml"}]

Patterns are relative to <root>: `*` and `?` stay within a path segment, `**` spans segments, `{a,b}` is either,
`<ds>` is any of the project's datasets and `<shared>` any of prune.shared's. A directory a `keep` rule ending in
`/**` covers is kept whole without being walked.

Safety: nothing outside <root> is touched; symlinks are never followed (a matched link is removed as a link); the
apply step refuses a manifest of another root, skips a file whose size changed since the dry run, and checks every
row against rule 1 again. Stat and unlink only: it runs on the login node (under nice and ionice: it walks the
whole tree).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "attribution"))

PRUNE_DIR = "prune"
ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------------------------------------------------
# the rules


@dataclass(frozen=True)
class Rule:
    action: str  # "keep" or "delete"
    pattern: str
    why: str
    ruleset: str

    @property
    def name(self) -> str:
        return f"{self.ruleset}: {self.pattern}"


def _k(pattern: str, why: str) -> tuple[str, str, str]:
    return ("keep", pattern, why)


def _d(pattern: str, why: str) -> tuple[str, str, str]:
    return ("delete", pattern, why)


ALWAYS = [
    _k("project.yaml", "the project"),
    _k("{NOTES,STATUS,README}.md", "the project's record"),
    _k("**/*.md", "notes, reports and READMEs, wherever they are"),
    _k("prune/**", "this tool's manifests and logs"),
    _k("fragments/**", "the inputs"),
    _k("portal/**", "the portal's other files"),
    _k("compendia/**", "the motif compendia (hits are tied to them by md5)"),
    _k("box/**", "each box's logs and stage table"),
    _k("bigwigs/**", "symlinks to the signal tracks"),
    _k("analyses/**", "the project's own analyses"),
]

RULESETS: dict[str, list[tuple[str, str, str]]] = {
    "pipeline": [
        # Cherimoya: the models and the float32 fold attributions are the canonical copy
        _k("cherimoya/<ds>/fold_*/**", "the fold's model, its performance and its float32 attribution (attr.*)"),
        _k("cherimoya/<ds>/downstream/contrib_scores/**", "06.0 fold average, 07.0 bigWig, 08.0 TF-MoDISco"),
        _k("cherimoya/<ds>/downstream/{.done,logs,metadata,validation}/**", "downstream.sh's markers and records"),
        _k(
            "cherimoya/<ds>/downstream/full_models/*/interpretation/interpretation.{interpreted_regions.bed,"
            "provenance.json}",
            "05.0's row index and provenance (07.0 and 10.0 read fold 0's bed)",
        ),
        _k("cherimoya/<ds>/downstream/full_models/*/{auxiliary,evaluation}/**", "per-fold motif QC"),
        _k("cherimoya/<ds>/downstream/finemo_*/*/finemo_report/**", "11.0's report"),
        _k(
            "cherimoya/<ds>/downstream/finemo_*/*/{hits.tsv,hits_unique.tsv,hits.bed.gz,hits.bed.gz.tbi,peaks_qc.tsv,"
            "motif_cwms.npy,motif_data.tsv,parameters.json,compendium.md5}",
            "10.0's hits",
        ),
        _k("cherimoya/{configs,cooperativity,box}/**", "downstream configs, DeepISA, the runners' records"),
        _k("cherimoya/*", "the runners' task lists, shared.txt, the engine pin, notes"),
        _d(
            "cherimoya/<ds>/downstream/full_models/*/interpretation/interpretation.counts_scores.h5",
            "05.0: a float16 copy of the fold's attr.attr.npz in ChromBPNet's layout; "
            "workflows/downstream/to_chrombpnet_scores.py rewrites it, CPU, ~2 min per fold",
        ),
        _d(
            "cherimoya/<ds>/downstream/finemo_*/*/intermediate_inputs.npz",
            "10.0's input arrays (the 06.0 average and the sequence); only 11.0 reads it, and 11.0 is done",
        ),
        _d("cherimoya/<ds>/downstream/finemo_*/*/hits.bed", "the uncompressed copy of hits.bed.gz"),
        # prep (igvf_tf_chrombpnet 00.0-02.0)
        _k("peaks/<ds>/{<ds>.narrowPeak.gz,call_peaks.json}", "the called peaks"),
        _k("chrombpnet/{configs/**,shifts.json}", "the igvf configs and the pinned Tn5 shifts"),
        _k("chrombpnet/<ds>/preprocessing/{signal,peaks}/**", "the signal bigWig and the summit windows"),
        _k("chrombpnet/<ds>/preprocessing/<ds>/output_all_fold_*_negatives.bed", "01.0's background regions"),
        _k("chrombpnet/<ds>/{plots,metadata,logs}/**", "02.0's QC, igvf's metadata and logs"),
        _d("peaks/<ds>/reps/**", "MACS3's per-replicate calls; only 00.0 reads them"),
        _d(
            "chrombpnet/<ds>/preprocessing/<ds>/output_all_fold_*_auxiliary/**",
            "01.0's genome-wide GC and candidate beds; only 01.0 reads them",
        ),
    ],
    "chrombpnet": [
        _k("chrombpnet/<ds>/full_models/*/models/*.h5", "the trained models (chrombpnet, nobias, scaled bias)"),
        _k("chrombpnet/<ds>/full_models/*/logs/**", "training logs and parameters (variant scoring reads them)"),
        _k("chrombpnet/<ds>/full_models/*/evaluation/*.{json,txt,png,pdf,html}", "metrics and reports"),
        _k(
            "chrombpnet/<ds>/full_models/*/evaluation/chrombpnet_predictions.h5",
            "test-peak predictions: train/compare_chrombpnet.py derives compare.tsv (level 1) from them",
        ),
        _k("chrombpnet/<ds>/full_models/*/evaluation/motif_qc_counts/**", "per-fold motif QC (comparison level 3)"),
        _k("chrombpnet/<ds>/full_models/*/auxiliary/*.{bed,tsv}", "filtered peaks and background, the subsample"),
        _k("chrombpnet/<ds>/full_models/*/auxiliary/interpret_subsample/*.{bed,json}", "the QC subsample's rows"),
        _k("chrombpnet/<ds>/full_models/*/interpretation/*.{bed,json}", "05.0's row index and arguments"),
        _k(
            "chrombpnet/<shared>/full_models/*/interpretation/interpretation.counts_scores.h5",
            "05.0 per fold, counts head: comparison level 2 reads every fold",
        ),
        _k("chrombpnet/<ds>/contrib_scores/<ds>/<ds>_average_shaps.counts.h5", "06.0 fold average (counts)"),
        _k("chrombpnet/<ds>/contrib_scores/<ds>/modisco/{modisco_counts_results.h5,counts_report/**}", "08.0"),
        _k("chrombpnet/<ds>/finemo_*/*/finemo_report/**", "11.0's report"),
        _k(
            "chrombpnet/<ds>/finemo_*/*/{hits.tsv,hits_unique.tsv,hits.bed.gz,hits.bed.gz.tbi,peaks_qc.tsv,"
            "motif_cwms.npy,motif_data.tsv,parameters.json,compendium.md5}",
            "10.0's hits",
        ),
        _k("chrombpnet/<ds>/bias_models/**/models/*.h5", "every bias model trained, the rejected ones too"),
        _k("chrombpnet/<ds>/bias_models/**/*.{json,tsv,txt,pdf,png,html,log}", "the bias models' QC and logs"),
        _k("chrombpnet/<ds>/bias_models/**/*bias_modisco_results.h5", "what each bias model learned (Tn5 or not)"),
        _k("chrombpnet/<ds>/full_models_*/**/evaluation/chrombpnet_metrics.json", "the A/B arms' metrics"),
        _d(
            "chrombpnet/<ds>/full_models/*/interpretation/*.{h5,partial}",
            "05.0 profile-head scores, and counts scores nothing compares; stopped runs' partial files",
        ),
        _d("chrombpnet/<ds>/full_models/*/interpretation/*.bw", "05.0 per-fold bigWigs"),
        _d(
            "chrombpnet/<ds>/full_models/*/auxiliary/interpret_subsample/*.h5",
            "04.4's QC-subsample scores; their TF-MoDISco (motif_qc_counts) is kept",
        ),
        _d("chrombpnet/<ds>/full_models/*/evaluation/bias_predictions.h5", "the bias model's test predictions"),
        _d("chrombpnet/<ds>/full_models/*/evaluation/motif_qc_profile/**", "per-fold motif QC of the profile head"),
        _d("chrombpnet/<ds>/contrib_scores/<ds>/*.bw", "07.0 bigWigs"),
        _d("chrombpnet/<ds>/contrib_scores/<ds>/*average_shaps.profile.h5", "06.0 fold average, profile head"),
        _d(
            "chrombpnet/<ds>/contrib_scores/<ds>/modisco/{modisco_profile_results.h5,profile_report/**}",
            "08.0 on the profile head",
        ),
        _d("chrombpnet/<ds>/contrib_scores/**/*.failed_*/**", "a failed report's leftovers"),
        _d("chrombpnet/<ds>/finemo_*/*/intermediate_inputs.npz", "10.0's input arrays; 11.0 is done"),
        _d("chrombpnet/<ds>/finemo_*/*/hits.bed", "the uncompressed copy of hits.bed.gz"),
        _d("chrombpnet/<ds>/predictions/**", "04.3's genome-wide predictions"),
        _d(
            "chrombpnet/<ds>/bias_models/**/*.{h5,bed,batch}",
            "the bias models' predictions, subsample scores and training beds",
        ),
        _d("chrombpnet/<ds>/full_models_*/**", "A/B training arms (their metrics are kept)"),
    ],
    "variant_scoring": [
        _k(
            "variant_scoring/*/scores/*/*.{summary.tsv.gz,summary.npz,json}",
            "each model's per-dataset summary: fold mean and held-out logFC, the null's p (qtl_compare.py reads it)",
        ),
        _k("variant_scoring/*/{lists,compare,compare_*,analysis,checks,nulls,box,logs,deferred}/**", "results"),
        _k("variant_scoring/*/{manifest*,*.env,*.sbatch,*.sh}", "the run's manifest and scripts"),
        _d("variant_scoring/*/preds/**", "the models' raw predictions per window"),
        _d("variant_scoring/*/windows/**", "the sequence windows, rebuilt from the list and the genome"),
        _d("variant_scoring/*/scores/*/fold_*/**", "per-fold variant and null scores; the summaries are kept"),
        _d("variant_scoring/*/scores_genomic_null/**", "the scores against the genomic null"),
        _d("variant_scoring/*/snapshots/**", "the pilot's intermediate snapshots"),
    ],
}


# ---------------------------------------------------------------------------------------------------------------------
# patterns


def _alternation(names) -> str:
    names = sorted(set(names), key=lambda n: (-len(n), n))
    return "(?:" + "|".join(re.escape(n) for n in names) + ")" if names else "(?!)"


def translate(pattern: str, datasets, shared=()) -> str:
    """A pattern (see the module docstring) as a regex over a POSIX path relative to <root>."""
    out, i, n = [], 0, len(pattern)
    while i < n:
        c = pattern[i]
        if pattern.startswith("**/", i):
            out.append("(?:[^/]+/)*")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif c == "*":
            out.append("[^/]*")
            i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        elif pattern.startswith("<ds>", i):
            out.append(_alternation(datasets))
            i += 4
        elif pattern.startswith("<shared>", i):
            out.append(_alternation(shared))
            i += 8
        elif c == "{":
            j = pattern.index("}", i)
            parts = pattern[i + 1 : j].split(",")
            out.append("(?:" + "|".join(translate(x, datasets, shared) for x in parts) + ")")
            i = j + 1
        else:
            out.append(re.escape(c))
            i += 1
    return "".join(out)


@dataclass
class Compiled:
    rule: Rule
    regex: re.Pattern
    whole: re.Pattern | None  # a keep rule ending in /**: the directories it keeps whole


def compile_rules(rules: list[Rule], datasets, shared=()) -> list[Compiled]:
    out = []
    for r in rules:
        whole = None
        if r.action == "keep" and r.pattern.endswith("/**"):
            whole = re.compile(translate(r.pattern[:-3], datasets, shared))
        out.append(Compiled(r, re.compile(translate(r.pattern, datasets, shared)), whole))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# the project's rules and what its checks need


class PruneError(ValueError):
    pass


def project_rules(p: dict) -> tuple[list[Rule], list[str]]:
    """(rules in order: every keep before every delete, so a keep always wins; shared datasets)."""
    block = p.get("prune") or {}
    if not isinstance(block, dict):
        raise PruneError("prune: must be a mapping")
    unknown = sorted(set(block) - {"rulesets", "shared", "keep", "delete"})
    if unknown:
        raise PruneError(f"prune: unknown keys {unknown}")
    sets = list(block.get("rulesets", ["pipeline"]) or [])  # an empty list: no rule set, only ALWAYS and the project's
    bad = [s for s in sets if s not in RULESETS]
    if bad:
        raise PruneError(f"prune: unknown rulesets {bad}; there are {sorted(RULESETS)}")
    raw = [("always", a, pat, why) for a, pat, why in ALWAYS]
    for s in sets:
        raw += [(s, a, pat, why) for a, pat, why in RULESETS[s]]
    for x in block.get("keep") or []:
        pat, why = (x, "") if isinstance(x, str) else (x["pattern"], x.get("why", ""))
        raw.append(("project", "keep", pat, why))
    for x in block.get("delete") or []:
        pat, why = (x, "") if isinstance(x, str) else (x["pattern"], x.get("why", ""))
        raw.append(("project", "delete", pat, why))
    for _, _, pat, _ in raw:
        if pat.startswith("/") or ".." in Path(pat).parts:
            raise PruneError(f"prune: pattern {pat!r} must be relative to the root, without ..")
    rules = [Rule(a, pat, why, s) for s, a, pat, why in raw]
    rules.sort(key=lambda r: r.action != "keep")  # stable: keeps first, each group in its written order
    shared = block.get("shared") or []
    if isinstance(shared, str):
        f = Path(shared) if Path(shared).is_absolute() else Path(p["root"]) / shared
        shared = [x.strip() for x in f.read_text().splitlines() if x.strip() and not x.startswith("#")]
    unknown = sorted(set(shared) - set(p["datasets"]))
    if unknown:
        raise PruneError(f"prune: shared datasets {unknown} are not the project's")
    return rules, list(shared)


def needed(p: dict) -> tuple[set[str], set[str]]:
    """(files, directories) that some stage's check reads, relative to <root>."""
    import attr_status
    import project as pj  # here, not at the top: project.py imports this module for its prune command

    root = Path(p["root"])
    files, dirs = set(), set()

    def add(path: Path) -> None:
        try:
            rel = Path(path).relative_to(root).as_posix()
        except ValueError:
            return  # outside the root: never ours to delete anyway
        (dirs.add(rel) if Path(path).is_dir() else files.add(rel))

    for stage in pj.STAGES:
        for path in pj.needed(p, stage):
            add(path)
    for ds in p["datasets"]:
        for k in p["folds"]:
            fold = pj.paths(p, ds)["cheri"] / f"fold_{k}"
            try:
                params = json.loads((fold / "attribute.json").read_text())
            except (OSError, ValueError):
                continue
            for path in attr_status.fold_files(params).values():
                add(path)
            for path in attr_status.model_files(params):
                add(path)
    return files, dirs


# ---------------------------------------------------------------------------------------------------------------------
# the dry run


def _under(rel: str, dirs: set[str]) -> bool:
    """rel is one of dirs or inside one."""
    parts = rel.split("/")
    return any("/".join(parts[:i]) in dirs for i in range(1, len(parts) + 1))


@dataclass
class Plan:
    root: Path
    delete: list[tuple[str, int, Rule]] = field(default_factory=list)
    unclassified: list[tuple[str, int]] = field(default_factory=list)
    whole: list[tuple[str, Rule]] = field(default_factory=list)  # directories kept without walking
    counts: dict[str, list] = field(default_factory=dict)  # rule name -> [action, why, files, bytes]
    needed_kept: int = 0

    def count(self, key: str, action: str, why: str, size: int) -> None:
        c = self.counts.setdefault(key, [action, why, 0, 0])
        c[2] += 1
        c[3] += size


def plan(p: dict) -> Plan:
    rules, shared = project_rules(p)
    compiled = compile_rules(rules, p["datasets"], shared)
    keeps = [c for c in compiled if c.rule.action == "keep"]
    deletes = [c for c in compiled if c.rule.action == "delete"]
    need_files, need_dirs = needed(p)
    root = Path(p["root"])
    out = Plan(root=root)

    def under_needed_dir(rel: str) -> bool:
        return _under(rel, need_dirs)

    def walk(dir_path: Path, rel_dir: str) -> None:
        try:
            entries = sorted(os.scandir(dir_path), key=lambda e: e.name)
        except OSError as exc:
            print(f"WARNING: cannot read {dir_path}: {exc}", file=sys.stderr)
            return
        for e in entries:
            rel = f"{rel_dir}/{e.name}" if rel_dir else e.name
            if e.is_dir(follow_symlinks=False):
                if rel == PRUNE_DIR:
                    continue
                if under_needed_dir(rel):
                    out.whole.append((rel, Rule("keep", rel, "a stage check reads it", "needed")))
                    continue
                kept = next((c for c in keeps if c.whole is not None and c.whole.fullmatch(rel)), None)
                if kept is not None:
                    out.whole.append((rel, kept.rule))
                    continue
                walk(Path(e.path), rel)
                continue
            size = e.stat(follow_symlinks=False).st_size
            if rel in need_files or under_needed_dir(rel):
                out.needed_kept += 1
                out.count("needed: a stage check reads it", "keep", "", size)
                continue
            c = next((c for c in keeps if c.regex.fullmatch(rel)), None)
            if c is not None:
                out.count(c.rule.name, "keep", c.rule.why, size)
                continue
            c = next((c for c in deletes if c.regex.fullmatch(rel)), None)
            if c is not None:
                out.delete.append((rel, size, c.rule))
                out.count(c.rule.name, "delete", c.rule.why, size)
                continue
            out.unclassified.append((rel, size))
            out.count("unclassified", "keep", "no rule matched: kept, listed in unclassified.<stamp>.tsv", size)

    walk(root, "")
    return out


def _stamp() -> str:
    return datetime.now(ET).strftime("%Y%m%d-%H%M%S")


def _gib(n: int) -> str:
    return f"{n / 2**30:,.1f} GiB"


def _commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_plan(p: dict, pl: Plan, stamp: str | None = None) -> dict[str, Path]:
    stamp = stamp or _stamp()
    d = pl.root / PRUNE_DIR
    d.mkdir(exist_ok=True)
    header = (f"# prune of {pl.root}, {datetime.now(ET):%Y-%m-%d %H:%M %Z}, by workflows/pipeline/prune.py "
              f"(RVI {_commit()}) from {p['file']}\n")  # fmt: skip
    files = {
        "manifest": d / f"manifest.{stamp}.tsv",
        "summary": d / f"summary.{stamp}.tsv",
        "unclassified": d / f"unclassified.{stamp}.tsv",
    }
    with files["manifest"].open("w") as fh:
        fh.write(header + f"# root\t{pl.root}\n" + "path\tbytes\trule\n")
        for rel, size, rule in pl.delete:
            fh.write(f"{rel}\t{size}\t{rule.name}\n")
    with files["summary"].open("w") as fh:
        fh.write(header + "action\trule\tfiles\tbytes\twhy\n")
        for name, (action, why, n, size) in sorted(pl.counts.items(), key=lambda kv: (kv[1][0], -kv[1][3])):
            fh.write(f"{action}\t{name}\t{n}\t{size}\t{why}\n")
        for rel, rule in pl.whole:
            fh.write(f"keep (whole, not walked)\t{rule.name}\t\t\t{rel}\n")
    with files["unclassified"].open("w") as fh:
        fh.write(header + "path\tbytes\n")
        for rel, size in sorted(pl.unclassified, key=lambda x: -x[1]):
            fh.write(f"{rel}\t{size}\n")
    return files


def print_plan(pl: Plan, files: dict[str, Path]) -> None:
    dele = sum(s for _, s, _ in pl.delete)
    print(f"prune of {pl.root} (dry run; nothing deleted)")
    print(f"  would delete {len(pl.delete):,} files, {_gib(dele)}:")
    for name, (action, _why, n, size) in sorted(pl.counts.items(), key=lambda kv: -kv[1][3]):
        if action == "delete":
            print(f"    {_gib(size):>12}  {n:>9,}  {name}")
    unc = sum(s for _, s in pl.unclassified)
    print(f"  unclassified (kept): {len(pl.unclassified):,} files, {_gib(unc)}; the largest directories:")
    by_dir: dict[str, int] = {}
    for rel, size in pl.unclassified:
        key = "/".join(rel.split("/")[:4])
        by_dir[key] = by_dir.get(key, 0) + size
    for key, size in sorted(by_dir.items(), key=lambda kv: -kv[1])[:15]:
        print(f"    {_gib(size):>12}  {key}")
    print(f"  kept whole without walking: {len(pl.whole):,} directories")
    for k, f in files.items():
        print(f"  {k}: {f}")
    print(f"  to delete: project.py <project.yaml> prune --apply {files['manifest']}")


# ---------------------------------------------------------------------------------------------------------------------
# apply


def read_manifest(path: Path) -> tuple[Path, list[tuple[str, int, str]]]:
    root, rows = None, []
    for line in Path(path).read_text().splitlines():
        if line.startswith("# root\t"):
            root = Path(line.split("\t", 1)[1])
        elif line.startswith("#") or line == "path\tbytes\trule":
            continue
        elif line:
            rel, size, rule = line.split("\t", 2)
            rows.append((rel, int(size), rule))
    if root is None:
        raise PruneError(f"{path}: not a prune manifest (no root line)")
    return root, rows


def _inside(root: Path, path: Path) -> bool:
    """path's directory, resolved, is root or below it (path itself may be a symlink: it is not followed)."""
    real_root = Path(os.path.realpath(root))
    real_dir = Path(os.path.realpath(path.parent))
    return real_dir == real_root or real_root in real_dir.parents


def apply(p: dict, manifest: Path, stamp: str | None = None) -> dict[str, int]:
    root = Path(p["root"])
    mroot, rows = read_manifest(manifest)
    if os.path.realpath(mroot) != os.path.realpath(root):
        raise PruneError(f"{manifest} is a manifest of {mroot}, not of this project's root {root}")
    need_files, need_dirs = needed(p)
    stamp = stamp or _stamp()
    log = root / PRUNE_DIR / f"deleted.{stamp}.tsv"
    counts = {"deleted": 0, "bytes": 0, "gone": 0, "changed": 0, "refused": 0, "dirs": 0}
    parents: set[Path] = set()
    with log.open("w") as fh:
        fh.write(f"# prune --apply {manifest}, {datetime.now(ET):%Y-%m-%d %H:%M %Z}, RVI {_commit()}\n")
        fh.write("path\tbytes\tstatus\n")
        for rel, size, _rule in rows:
            path = root / rel
            if (rel.startswith("/") or ".." in Path(rel).parts or not _inside(root, path) or rel in need_files
                    or _under(rel, need_dirs)):  # fmt: skip
                status = "refused"
            else:
                try:
                    st = os.lstat(path)
                except FileNotFoundError:
                    status = "gone"
                else:
                    if not (os.path.islink(path) or os.path.isfile(path)):
                        status = "refused"
                    elif st.st_size != size:
                        status = "changed"
                    else:
                        os.unlink(path)
                        status = "deleted"
                        counts["bytes"] += size
                        parents.add(path.parent)
            counts[status] += 1
            fh.write(f"{rel}\t{size}\t{status}\n")
        # the directories left empty, deepest first, never the root
        todo = set()
        for d in parents:
            while d != root and root in d.parents:
                todo.add(d)
                d = d.parent
        for d in sorted(todo, key=lambda x: -len(x.parts)):
            try:
                d.rmdir()
            except OSError:
                continue
            counts["dirs"] += 1
            fh.write(f"{d.relative_to(root).as_posix()}/\t0\tdirectory removed\n")
    counts["log"] = str(log)  # type: ignore[assignment]
    return counts
