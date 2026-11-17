"""workflows/pipeline/pseudobulk.py: single-cell fragments of many wells split into one pseudobulk per group of cells.

The split runs the vendored peaks env's bgzip, tabix and mawk (setup.sh installs it); the refusals need none of them.
"""

from __future__ import annotations

import csv
import gzip
import json
import subprocess
import sys
from pathlib import Path

import pytest

PIPELINE = Path(__file__).resolve().parents[1] / "workflows" / "pipeline"
sys.path.insert(0, str(PIPELINE))

import pseudobulk as pb  # noqa: E402

BIN = pb.DEFAULT_BIN
needs_tools = pytest.mark.skipif(
    not all((BIN / t).is_file() for t in ("bgzip", "tabix", "mawk")),
    reason=f"no bgzip, tabix or mawk in {BIN}: run workflows/pipeline/setup.sh",
)

# 10x's contig order is not the C locale's: chr2 before chr10, KI... before GL...
ORDER = ["chr2", "chr10", "KI270728.1", "GL000009.2"]
HEADER = "# id=toy\n# pipeline_name=cellranger-arc\n#\n"


def _rows(barcodes: list[str], contigs: list[str], seed: int) -> list[list[str]]:
    """6-column rows (chrom, start, end, barcode, count, strand) for each barcode on each contig, with ties on start
    within and across wells."""
    rows = []
    for ci, c in enumerate(contigs):
        for bi, b in enumerate(barcodes):
            for k in range(3):
                start = 100 * (k + 1) + 10 * ((bi + seed) % 2)
                rows.append([c, str(start), str(start + 50 + bi + ci), b, str(k + 1), "+-"[k % 2]])
    return sorted(rows, key=lambda r: (ORDER.index(r[0]), int(r[1]), int(r[2])))


def _bgzip_indexed(path: Path, rows: list[list[str]]) -> None:
    text = HEADER + "".join("\t".join(r) + "\n" for r in rows)
    with open(path, "wb") as fh:
        subprocess.run([str(BIN / "bgzip"), "-c"], input=text.encode(), stdout=fh, check=True)
    subprocess.run([str(BIN / "tabix"), "-p", "bed", str(path)], check=True)


def _write_wells(tmp_path: Path, wells: dict[str, list[list[str]]]) -> Path:
    table = tmp_path / "wells.tsv"
    lines = ["well\tpath"]
    for w, rows in wells.items():
        f = tmp_path / "frags" / f"{w}.atac_fragments.tsv.gz"
        f.parent.mkdir(exist_ok=True)
        _bgzip_indexed(f, rows)
        lines.append(f"{w}\t{f}")
    table.write_text("\n".join(lines) + "\n")
    return table


def _write_cells(tmp_path: Path, cells: list[tuple[str, str]]) -> Path:
    path = tmp_path / "cells.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["barcode", "umap_1", "cell_type"])
        for cid, group in cells:
            w.writerow([cid, "0.1", group])
    return path


def _args(tmp_path: Path, cells: Path, wells: Path, *extra: str) -> list[str]:
    return ["--cells", str(cells), "--group-column", "cell_type", "--wells", str(wells),
            "--out", str(tmp_path / "root" / "fragments"), "--jobs", "2", "--bin", str(BIN), *extra]  # fmt: skip


def _read(path: Path) -> list[list[str]]:
    with gzip.open(path, "rt") as fh:
        return [line.rstrip("\n").split("\t") for line in fh]


@pytest.fixture
def toy(tmp_path):
    """w1: two labelled cells of two groups, one unlabelled barcode, every contig; w2: two cells of one group (so the
    other group has no row there), two contigs only; w3: fragments but no labelled cell. One labelled cell of w1
    (NNN-1) has no fragment."""
    w1 = _rows(["AAA-1", "CCC-1", "GGG-1"], ORDER, 0)
    w2 = _rows(["AAA-1", "TTT-1"], ORDER[:2], 1)
    w3 = _rows(["AAA-1"], ORDER[:1], 0)
    wells = _write_wells(tmp_path, {"w1": w1, "w2": w2, "w3": w3})
    cells = _write_cells(tmp_path, [("w1_AAA-1", "T cell"), ("w1_CCC-1", "B/plasma"), ("w1_NNN-1", "T cell"),
                                    ("w2_AAA-1", "T cell"), ("w2_TTT-1", "T cell")])  # fmt: skip
    return {"wells": wells, "cells": cells, "rows": {"w1": w1, "w2": w2, "w3": w3}}


def test_stems_are_lowercase_with_underscores():
    assert pb.stem_of("T cell") == "t_cell"
    assert pb.stem_of("B/plasma") == "b_plasma"
    assert pb.stem_of("  NK (CD56+) ") == "nk_cd56"


@needs_tools
def test_each_labelled_row_lands_once_in_its_group_renamed_and_sorted(tmp_path, toy, capsys):
    rc = pb.main(_args(tmp_path, toy["cells"], toy["wells"], "--min-found", "0.5"))
    out_dir = tmp_path / "root" / "fragments"
    assert rc == 0, capsys.readouterr().err
    labelled = {("w1", "AAA-1"): "T cell", ("w1", "CCC-1"): "B/plasma", ("w2", "AAA-1"): "T cell",
                ("w2", "TTT-1"): "T cell"}  # fmt: skip
    for group, stem in (("T cell", "t_cell"), ("B/plasma", "b_plasma")):
        f = out_dir / f"{stem}.fragments.tsv.gz"
        got = _read(f)
        want = [[r[0], r[1], r[2], f"{w}_{r[3]}", r[4], r[5]] for w, rows in toy["rows"].items() for r in rows
                if labelled.get((w, r[3])) == group]  # fmt: skip
        assert sorted(got) == sorted(want)  # every kept row once, renamed, all six columns
        keys = [(r[0], int(r[1])) for r in got]
        assert keys == sorted(keys)  # C-locale contig order, then start
        assert (out_dir / f"{stem}.fragments.tsv.gz.tbi").is_file()
        q = subprocess.run([str(BIN / "tabix"), str(f), "chr10"], capture_output=True, text=True, check=True)
        assert sorted(line.split("\t") for line in q.stdout.splitlines()) == sorted(r for r in want
                                                                                    if r[0] == "chr10")  # fmt: skip
    # the KI and GL contigs exist in w1 only; the B/plasma group has no row in w2
    assert {r[0] for r in _read(out_dir / "t_cell.fragments.tsv.gz")} == set(ORDER)
    table = {r["stem"]: r for r in csv.DictReader(open(out_dir / "pseudobulk.tsv"), delimiter="\t")}
    assert table["t_cell"]["cells"] == "4" and table["t_cell"]["cells_found"] == "3"
    assert table["t_cell"]["wells"] == "2" and table["b_plasma"]["wells"] == "1"
    assert int(table["b_plasma"]["fragments"]) == len(_read(out_dir / "b_plasma.fragments.tsv.gz"))
    record = json.loads((out_dir / "pseudobulk.json").read_text())
    assert record["wells"]["skipped"] == ["w3"]
    assert record["groups"][0]["group"] == "T cell"  # largest first
    assert not (out_dir / ".pseudobulk_tmp").exists()
    printed = capsys.readouterr().out
    md5 = table["t_cell"]["md5"]
    assert f'{{stem: t_cell, fragments: {out_dir}/t_cell.fragments.tsv.gz, md5: {md5}, label: "T cell"}}' in printed
    assert md5 == pb.md5_of(out_dir / "t_cell.fragments.tsv.gz")


@needs_tools
def test_ids_that_are_not_the_fragments_barcodes_stop_the_run(tmp_path, toy, capsys):
    cells = _write_cells(tmp_path, [("w1_AAA-2", "T cell"), ("w1_CCC-2", "B/plasma")])  # -2: not in the fragments
    assert pb.main(_args(tmp_path, cells, toy["wells"])) == 1
    assert "do the cell IDs match the fragments' barcodes" in capsys.readouterr().err
    assert not list((tmp_path / "root" / "fragments").glob("*.fragments.tsv.gz"))


def test_dry_run_reads_the_tables_and_writes_nothing(tmp_path, capsys):
    wells = tmp_path / "wells.tsv"
    wells.write_text(f"well\tpath\nw1\t{tmp_path / 'w1.tsv.gz'}\nw2\t{tmp_path / 'w2.tsv.gz'}\n")
    for w in ("w1", "w2"):
        (tmp_path / f"{w}.tsv.gz").write_bytes(b"")
        (tmp_path / f"{w}.tsv.gz.tbi").write_bytes(b"")
    cells = _write_cells(tmp_path, [("w1_A-1", "T cell"), ("w1_B-1", "B/plasma"), ("w1_C-1", "T cell")])
    tools = tmp_path / "bin"
    tools.mkdir()
    for t in ("bgzip", "tabix", "mawk"):
        (tools / t).write_text("#!/bin/sh\n")
        (tools / t).chmod(0o755)
    args = _args(tmp_path, cells, wells, "--dry-run")
    args[args.index("--bin") + 1] = str(tools)
    assert pb.main(args) == 0
    printed = capsys.readouterr().out
    assert "2 groups, 3 cells, 1 wells used, 1 skipped (no labelled cell): w2" in printed
    assert "T cell\tt_cell\t2\t1\t" in printed
    assert not (tmp_path / "root").exists()


@pytest.mark.parametrize(
    "cells, message",
    [
        ([("w1_A-1", "T cell"), ("w1_A-1", "B/plasma")], "is there twice"),
        ([("w9_A-1", "T cell")], "is not in the wells table"),
        ([("A-1", "T cell")], "is not <well>_<barcode>"),
        ([("w1_A-1", " ")], "has no cell_type"),
        ([("w1_A-1", "B/plasma"), ("w1_B-1", "B plasma")], "gives the stem 'b_plasma'"),
        ([("w1_A-1", "T cell")], "exists: delete it"),
    ],
)
def test_bad_tables_and_existing_outputs_are_refused(tmp_path, capsys, cells, message):
    wells = tmp_path / "wells.tsv"
    wells.write_text(f"well\tpath\nw1\t{tmp_path / 'w1.tsv.gz'}\n")
    (tmp_path / "w1.tsv.gz").write_bytes(b"")
    (tmp_path / "w1.tsv.gz.tbi").write_bytes(b"")
    out = tmp_path / "root" / "fragments"
    out.mkdir(parents=True)
    (out / "t_cell.fragments.tsv.gz").write_bytes(b"")  # only the T cell group's output exists
    assert pb.main(_args(tmp_path, _write_cells(tmp_path, cells), wells)) == 1
    assert message in capsys.readouterr().err
