#!/usr/bin/env python3
"""Split single-cell fragments into one pseudobulk per group of cells (e.g. a cell type), from a table of labelled
cells. Each pseudobulk becomes a project sample: project.yaml's fragments is the absolute path of the file written
here.

    pseudobulk.py --cells <cells.csv|.tsv> --group-column <col> --wells <wells.tsv> --out <root>/fragments
                  [--id-column barcode] [--sep _] [--jobs N] [--tmp DIR] [--bin DIR] [--min-found 0.9] [--dry-run]

Inputs:
  --cells   a table with a header, comma-separated if it ends in .csv, else tab-separated. --id-column is each cell's
            ID, <well><sep><barcode> (split at the last <sep>); --group-column its group. Every ID is unique, every
            group non-empty, and every well is in --wells.
  --wells   a TSV with a header and the columns well and path: each well's 10x fragments file (BGZF, with its .tbi),
            whose column 4 is the cell barcode. A well with no labelled cell is skipped (and listed).

Output, per group, in --out (project.py's fragments directory, <root>/fragments):
  <stem>.fragments.tsv.gz (BGZF) and its .tbi: every fragment of the group's cells from every well, column 4
            renamed to <well><sep><barcode> (unique across wells), the other columns as the wells have them; sorted
            by contig (C locale) and start. <stem> is the group's name lower-cased, each run of other characters
            than a-z and 0-9 replaced by _.
  pseudobulk.tsv  group, stem, cells, cells_found (with at least one fragment), fragments (rows), wells, md5, bytes
  pseudobulk.json the inputs (paths, md5s, columns), the wells used and skipped with their counts, the tools'
            versions, the RVI commit, start and end (ET)
  and on stdout, the samples block for project.yaml (stem, fragments, md5, label = the group's name).

How:
  1. Per well, --jobs at a time: `tabix <well> <contig>` for each contig, in C-locale order (10x writes its
     non-primary contigs out of that order, and step 2 needs every input in one order), into mawk, which keeps the
     rows of the well's labelled cells, renames their barcode and writes them to one BGZF temporary file per
     (well, group). A well where fewer than --min-found of its labelled cells have a fragment stops the run: the
     IDs do not match the fragments' barcodes.
  2. Per group, all groups at once: `sort -m` of the group's temporary files (each already sorted), through bgzip,
     to <stem>.fragments.tsv.gz.partial; tabix; the rows counted again and compared with step 1's; the md5; then
     renamed into place. The temporary files are deleted when every group is done.
Prep does not need sorted fragments (its readers stream them), but sorted and indexed is the 10x/ENCODE format, so
any tool can read the files by region.

bgzip, tabix and mawk come from --bin (by default the vendored peaks env); sort is the system's GNU sort. Standard
library only. --dry-run reads the two tables and checks the inputs and outputs, so it runs on the login node; the
split reads every well's fragments, so it runs in a box (docs/pipeline.md, "Single-cell data").
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from pathlib import Path

HERE = Path(__file__).resolve().parent
RVI = HERE.parents[1]
DEFAULT_BIN = RVI / "vendor" / "igvf_tf_chrombpnet" / ".pixi" / "envs" / "peaks" / "bin"
FRAGMENTS_SUFFIX = ".fragments.tsv.gz"  # project.py's FRAGMENTS_SUFFIX
NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")  # project.py's NAME_RE, for stems
WELL_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
TMP_MARK = ".pseudobulk_tmp"
CHUNK = 8 << 20

# Read by mawk: first the commands file (f == 1: "stats" or a group index, TAB, a path or a shell command), then the
# well's cells (f == 2: barcode TAB group index), then the fragments on stdin (f == 3). Per kept row: the barcode is
# renamed and the row goes to its group's bgzip, opened on its first row; a group with no row gets no file.
SPLIT_AWK = r"""
BEGIN { FS = OFS = "\t"; f = 0 }
FNR == 1 { f++ }
f == 1 { if ($1 == "stats") stats = $2; else cmd[$1] = $2; next }
f == 2 { grp[$1] = $2; next }
/^#/ { next }
{ rows++ }
($4 in grp) {
    g = grp[$4]
    if (!($4 in seen)) { seen[$4] = 1; found[g]++ }
    $4 = well sep $4
    print | cmd[g]
    kept[g]++
}
END {
    bad = 0
    for (g in kept) if (close(cmd[g]) != 0) bad = 1
    printf "rows\t%d\n", rows > stats
    for (g in kept) printf "%s\t%d\t%d\n", g, kept[g], found[g] > stats
    close(stats)
    exit bad
}
"""


class PseudobulkError(Exception):
    pass


def et() -> str:
    os.environ.setdefault("TZ", "America/New_York")
    time.tzset()
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def stem_of(group: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", group.lower()).strip("_")


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def read_wells(path: Path) -> dict[str, Path]:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if not rows or not {"well", "path"} <= set(rows[0]):
        raise PseudobulkError(f"{path}: needs a header with the columns well and path")
    wells: dict[str, Path] = {}
    for r in rows:
        w = r["well"]
        if not WELL_RE.match(w) or w in wells:
            raise PseudobulkError(f"{path}: well {w!r} is not unique or not {WELL_RE.pattern}")
        wells[w] = Path(r["path"])
    return wells


def read_cells(path: Path, id_column: str, group_column: str, sep: str, wells: dict[str, Path]) -> dict:
    """{well: {barcode: group}} and the groups in order of size (largest first, then by name)."""
    delimiter = "," if path.name.endswith(".csv") else "\t"
    by_well: dict[str, dict[str, str]] = {}
    sizes: dict[str, int] = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        missing = [c for c in (id_column, group_column) if c not in (reader.fieldnames or [])]
        if missing:
            raise PseudobulkError(f"{path}: no column {missing}; it has {reader.fieldnames}")
        for i, r in enumerate(reader, start=2):
            cid, group = r[id_column], r[group_column]
            well, s, barcode = cid.rpartition(sep)
            if not s or not well or not barcode:
                raise PseudobulkError(f"{path}, line {i}: {cid!r} is not <well>{sep}<barcode>")
            if well not in wells:
                raise PseudobulkError(f"{path}, line {i}: {cid!r}: well {well!r} is not in the wells table")
            if not group.strip():
                raise PseudobulkError(f"{path}, line {i}: {cid!r} has no {group_column}")
            cells = by_well.setdefault(well, {})
            if barcode in cells:
                raise PseudobulkError(f"{path}, line {i}: {cid!r} is there twice")
            cells[barcode] = group
            sizes[group] = sizes.get(group, 0) + 1
    if not sizes:
        raise PseudobulkError(f"{path}: no cells")
    groups = sorted(sizes, key=lambda g: (-sizes[g], g))
    stems: dict[str, str] = {}
    for g in groups:
        st = stem_of(g)
        if not NAME_RE.match(st) or st in stems.values():
            other = [k for k, v in stems.items() if v == st]
            raise PseudobulkError(f"group {g!r} gives the stem {st!r}, which is empty or also {other}'s")
        stems[g] = st
    return {"by_well": by_well, "groups": groups, "sizes": sizes, "stems": stems}


def plan(args) -> dict:
    wells = read_wells(args.wells)
    cells = read_cells(args.cells, args.id_column, args.group_column, args.sep, wells)
    used = [w for w in wells if w in cells["by_well"]]
    skipped = [w for w in wells if w not in cells["by_well"]]
    for w in used:
        for f in (wells[w], Path(f"{wells[w]}.tbi")):
            if not f.is_file():
                raise PseudobulkError(f"well {w}: {f} is missing")
    out = {g: args.out / f"{cells['stems'][g]}{FRAGMENTS_SUFFIX}" for g in cells["groups"]}
    for f in out.values():
        for x in (f, Path(f"{f}.tbi")):
            if x.exists():
                raise PseudobulkError(f"{x} exists: delete it to make the pseudobulks again")
    tools = {t: args.bin / t for t in ("bgzip", "tabix", "mawk")}
    tools["sort"] = Path(shutil.which("sort") or "sort")
    for t, p in tools.items():
        if not (p.is_file() and os.access(p, os.X_OK)):
            raise PseudobulkError(f"{t}: {p} is not there (give --bin, or run setup.sh)")
    in_well = [set(cells["by_well"][w].values()) for w in used]
    wells_of = {g: sum(g in s for s in in_well) for g in cells["groups"]}
    return {**cells, "wells": wells, "used": used, "skipped": skipped, "tools": tools, "out": out,
            "wells_of": wells_of}  # fmt: skip


def _tmp_dir(path: Path) -> None:
    if path.exists():
        if (path / TMP_MARK).exists():
            shutil.rmtree(path)  # a failed run's, by its mark
        elif any(path.iterdir()):
            raise PseudobulkError(f"{path} exists and is not this tool's temporary directory")
    path.mkdir(parents=True, exist_ok=True)
    (path / TMP_MARK).write_text("pseudobulk.py's temporary files, deleted when it succeeds\n")


def split_well(pl: dict, args, well: str) -> dict:
    """Step 1 for one well: its labelled cells' rows, one temporary BGZF file per group."""
    tools, gidx = pl["tools"], {g: str(i) for i, g in enumerate(pl["groups"])}
    d = args.tmp / "wells" / well
    d.mkdir(parents=True, exist_ok=True)
    cmds, cells_f, stats_f = d / "cmds.tsv", d / "cells.tsv", d / "stats.tsv"
    lines = [f"stats\t{stats_f}"]
    for i in gidx.values():
        tmp = args.tmp / "groups" / i / f"{well}.tsv.gz"
        lines.append(f"{i}\t{shlex.quote(str(tools['bgzip']))} -l 1 -c > {shlex.quote(str(tmp))}")
    cmds.write_text("\n".join(lines) + "\n")
    cells = pl["by_well"][well]
    cells_f.write_text("".join(f"{b}\t{gidx[g]}\n" for b, g in cells.items()))
    frag = pl["wells"][well]
    listed = subprocess.run([str(tools["tabix"]), "-l", str(frag)], capture_output=True, text=True)
    if listed.returncode:
        raise PseudobulkError(f"well {well}: tabix -l {frag}: {listed.stderr.strip()}")
    contigs = sorted(listed.stdout.split())  # code-point order = the C locale's, for sort -m -k1,1
    for c in contigs:
        if ":" in c:
            raise PseudobulkError(f"well {well}: contig {c!r} has a ':', which tabix would read as a region")
    env = {**os.environ, "LC_ALL": "C"}
    err_f = d / "mawk.err"  # a file, not a pipe: a pipe left unread could block mawk and so tabix
    with open(err_f, "wb") as err:
        awk = subprocess.Popen([str(tools["mawk"]), "-v", f"well={well}", "-v", f"sep={args.sep}", "-f",
                                str(args.tmp / "split.awk"), str(cmds), str(cells_f), "-"],
                               stdin=subprocess.PIPE, stderr=err, env=env)  # fmt: skip
        try:
            for c in contigs:
                q = subprocess.run([str(tools["tabix"]), str(frag), c], stdout=awk.stdin, stderr=subprocess.PIPE)
                if q.returncode:
                    raise PseudobulkError(f"well {well}: tabix {frag} {c}: {q.stderr.decode().strip()}")
        finally:
            awk.stdin.close()
            rc = awk.wait()
    if rc:
        raise PseudobulkError(f"well {well}: mawk or bgzip failed ({rc}): {err_f.read_text().strip()}")
    rows, kept, found = 0, {}, {}
    for line in stats_f.read_text().splitlines():
        k, *v = line.split("\t")
        if k == "rows":
            rows = int(v[0])
        else:
            g = pl["groups"][int(k)]
            kept[g], found[g] = int(v[0]), int(v[1])
    n_found = sum(found.values())
    if n_found < args.min_found * len(cells):
        raise PseudobulkError(f"well {well}: only {n_found} of its {len(cells)} labelled cells have a fragment in "
                              f"{frag}: do the cell IDs match the fragments' barcodes (column 4)?")  # fmt: skip
    return {"well": well, "path": str(frag), "cells": len(cells), "cells_found": n_found, "rows": rows,
            "rows_kept": sum(kept.values()), "kept": kept, "found": found}  # fmt: skip


def merge_group(pl: dict, args, group: str, expected: int, threads: int) -> dict:
    """Step 2 for one group: sort -m of its temporary files, bgzip, tabix, count, md5, rename."""
    tools, i = pl["tools"], str(pl["groups"].index(group))
    parts = sorted((args.tmp / "groups" / i).glob("*.tsv.gz"))
    if not parts:
        raise PseudobulkError(f"group {group}: no fragments in any well")
    out = pl["out"][group]
    partial = Path(f"{out}.partial")
    q = shlex.quote
    inputs = " ".join(f"<({q(str(tools['bgzip']))} -dc {q(str(p))})" for p in parts)
    merge = (f"set -o pipefail; LC_ALL=C {q(str(tools['sort']))} -m -t $'\\t' -k1,1 -k2,2n "
             f"--batch-size={len(parts) + 1} {inputs} "
             f"| {q(str(tools['bgzip']))} -@ {threads} -c > {q(str(partial))}")  # fmt: skip
    for cmd in (["bash", "-c", merge], [str(tools["tabix"]), "-f", "-p", "bed", str(partial)]):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            raise PseudobulkError(f"group {group}: {cmd[0]} failed ({r.returncode}): {r.stderr.strip()[-2000:]}")
    count = subprocess.run(["bash", "-c", f"set -o pipefail; {q(str(tools['bgzip']))} -dc -@ {threads} "
                            f"{q(str(partial))} | wc -l"], capture_output=True, text=True)  # fmt: skip
    if count.returncode or int(count.stdout.split()[0]) != expected:
        raise PseudobulkError(f"group {group}: {partial} has {count.stdout.strip()} rows, step 1 kept {expected}")
    md5 = md5_of(partial)
    os.replace(f"{partial}.tbi", f"{out}.tbi")
    os.replace(partial, out)
    return {"md5": md5, "bytes": out.stat().st_size, "parts": len(parts)}


def _version(cmd: list[str]) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return (r.stdout or r.stderr).strip().splitlines()[0]
    except (OSError, IndexError, subprocess.TimeoutExpired):
        return "unknown"


def _commit() -> str:
    r = subprocess.run(["git", "-C", str(RVI), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() or "unknown"


def _parallel(fn, items, jobs: int) -> list:
    """fn over items on jobs threads (each waits on its subprocesses); the first error cancels the rest."""
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        futures = [ex.submit(fn, x) for x in items]
        done, _ = wait(futures, return_when=FIRST_EXCEPTION)
        for f in done:
            if f.exception():
                for other in futures:
                    other.cancel()
                raise f.exception()
        return [f.result() for f in futures]


def samples_block(pl: dict, md5s: dict[str, str] | None) -> str:
    lines = ["samples:"]
    for g in pl["groups"]:
        md5 = f", md5: {md5s[g]}" if md5s else ""
        lines.append(f"  - {{stem: {pl['stems'][g]}, fragments: {pl['out'][g]}{md5}, label: {json.dumps(g)}}}")
    return "\n".join(lines)


def run(args) -> dict:
    started = et()
    pl = plan(args)
    print(f"{len(pl['groups'])} groups, {sum(pl['sizes'].values()):,} cells, {len(pl['used'])} wells used, "
          f"{len(pl['skipped'])} skipped (no labelled cell): {' '.join(pl['skipped']) or '-'}")  # fmt: skip
    print("group\tstem\tcells\twells\toutput")
    for g in pl["groups"]:
        print(f"{g}\t{pl['stems'][g]}\t{pl['sizes'][g]}\t{pl['wells_of'][g]}\t{pl['out'][g]}")
    if args.dry_run:
        print(samples_block(pl, None))
        return pl
    args.out.mkdir(parents=True, exist_ok=True)
    _tmp_dir(args.tmp)
    (args.tmp / "split.awk").write_text(SPLIT_AWK)
    for i in range(len(pl["groups"])):
        (args.tmp / "groups" / str(i)).mkdir(parents=True)
    t0 = time.time()
    per_well = _parallel(lambda w: split_well(pl, args, w), pl["used"], args.jobs)
    print(f"step 1: {len(per_well)} wells split in {time.time() - t0:.0f} s", flush=True)
    expected = {g: sum(w["kept"].get(g, 0) for w in per_well) for g in pl["groups"]}
    threads = max(1, args.jobs // len(pl["groups"]))
    t1 = time.time()
    merged = _parallel(lambda g: merge_group(pl, args, g, expected[g], threads), pl["groups"], len(pl["groups"]))
    print(f"step 2: {len(merged)} groups merged in {time.time() - t1:.0f} s", flush=True)
    rows = []
    for g, m in zip(pl["groups"], merged):
        found = sum(w["found"].get(g, 0) for w in per_well)
        rows.append({"group": g, "stem": pl["stems"][g], "cells": pl["sizes"][g], "cells_found": found,
                     "fragments": expected[g], "wells": m["parts"], "md5": m["md5"], "bytes": m["bytes"],
                     "path": str(pl["out"][g])})  # fmt: skip
    cols = ["group", "stem", "cells", "cells_found", "fragments", "wells", "md5", "bytes"]
    (args.out / "pseudobulk.tsv").write_text("\n".join(["\t".join(cols)] + ["\t".join(str(r[c]) for c in cols)
                                                                             for r in rows]) + "\n")  # fmt: skip
    t = pl["tools"]
    record = {
        "cells": {"path": str(args.cells), "md5": md5_of(args.cells), "id_column": args.id_column,
                  "group_column": args.group_column, "sep": args.sep},
        "wells": {"path": str(args.wells), "md5": md5_of(args.wells), "used": len(pl["used"]),
                  "skipped": pl["skipped"]},
        "min_found": args.min_found,
        "groups": rows,
        "per_well": [{k: w[k] for k in ("well", "path", "cells", "cells_found", "rows", "rows_kept")}
                     for w in per_well],
        "tools": {"bgzip": _version([str(t["bgzip"]), "--version"]), "tabix": _version([str(t["tabix"]), "--version"]),
                  "mawk": _version([str(t["mawk"]), "-W", "version"]), "sort": _version([str(t["sort"]), "--version"])},
        "rvi": _commit(),
        "started": started,
        "finished": et(),
    }  # fmt: skip
    (args.out / "pseudobulk.json").write_text(json.dumps(record, indent=1) + "\n")
    shutil.rmtree(args.tmp)
    md5s = {r["group"]: r["md5"] for r in rows}
    print("group\tcells\tcells_found\tfragments\tbytes")
    for r in rows:
        print(f"{r['group']}\t{r['cells']}\t{r['cells_found']}\t{r['fragments']}\t{r['bytes']}")
    print(samples_block(pl, md5s))
    return {**pl, "rows": rows, "record": record}


def parse(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cells", type=Path, required=True)
    ap.add_argument("--group-column", required=True)
    ap.add_argument("--id-column", default="barcode")
    ap.add_argument("--wells", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sep", default="_")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--tmp", type=Path, help="temporary files (default <out>/.pseudobulk_tmp)")
    ap.add_argument("--bin", type=Path, default=DEFAULT_BIN, help="bgzip, tabix and mawk (default: %(default)s)")
    ap.add_argument("--min-found", type=float, default=0.9)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    args.out = args.out.absolute()
    args.tmp = (args.tmp or args.out / ".pseudobulk_tmp").absolute()
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)
    try:
        run(args)
    except PseudobulkError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
