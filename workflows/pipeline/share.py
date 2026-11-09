#!/usr/bin/env python3
"""Pack a project's results for someone without cluster access: one tar, with an md5 of every file in it.

    project.py <project.yaml> share --out <file.tar> [--datasets <ds>,<ds>] [--with-signal] [--list]

What goes in (docs/results.md explains each file; the tar holds that page too, as docs/results.md):
  per dataset    the 5 fold models with their fit.json and performance tables; the fold-averaged attribution
                 (06.0 h5) and its bigWig (07.0); TF-MoDISco (08.0) and its report; the Fi-NeMo hits on the
                 project's compendium and their report (10.0, 11.0); the peaks and the summit windows the
                 attributions are indexed by; the signal QC; the dataset's DeepISA tables; with --with-signal the
                 observed signal bigWig too
  per project    README.md, STATUS.md, project.yaml, the compendium, the DeepISA tables across datasets
Everything sits under <project>/ in the tar, at its path under <root>. MANIFEST.md5 (md5sum -c reads it) and
SHARE.txt (what, when, from which commit, and what was missing) are the last two members. Files are hashed as
they are written, so each is read once. The tar is written to <out>.partial and renamed when complete.

Ask before sending a project's files outside the group: a dataset may be unpublished (its README says).
Standard library only; it reads every file it packs, so run it on the login node only for a few datasets.
"""

from __future__ import annotations

import hashlib
import io
import os
import subprocess
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_DOC = HERE.parents[1] / "docs" / "results.md"
FINEMO_FILES = ("hits_unique.tsv", "hits.bed.gz", "hits.bed.gz.tbi", "peaks_qc.tsv", "motif_data.tsv",
                "motif_cwms.npy", "parameters.json", "compendium.md5")  # fmt: skip


def _tree(d: Path) -> list[Path]:
    return sorted(f for f in d.rglob("*") if f.is_file()) if d.is_dir() else [d]


def share_set(p: dict, datasets: list[str], with_signal: bool = False) -> list[Path]:
    """Every path the share holds, project files first; a missing one is listed as it would be."""
    import project as pj  # here: project.py imports this module for its share command

    root = Path(p["root"])
    coop = pj.paths(p)["coop"]
    out = [root / "README.md", root / "STATUS.md", root / "project.yaml"]
    out += _tree(pj.paths(p)["compendium"])
    out += [coop / f for f in ("motifs.tsv", "groups.tsv", "comparisons.tsv", "series.tsv") if (coop / f).exists()]
    for d in ("combined", "official", "matrix/combined"):
        if (coop / d).is_dir():
            out += _tree(coop / d)
    for ds in datasets:
        q = pj.paths(p, ds)
        for k in p["folds"]:
            fold = q["cheri"] / f"fold_{k}"
            out += [fold / f"cherimoya.fold_{k}.torch", fold / "fit.json",
                    fold / f"cherimoya.fold_{k}.test.performance.tsv",
                    fold / f"cherimoya.fold_{k}.validation.performance.tsv"]  # fmt: skip
        contrib = q["downstream"] / "contrib_scores" / ds
        out += [contrib / f"{ds}_average_shaps.counts.h5", contrib / f"{ds}_average_shaps.counts.bw",
                contrib / "modisco" / "modisco_counts_results.h5"]  # fmt: skip
        out += _tree(contrib / "modisco" / "counts_report")
        out += [q["finemo"] / f for f in FINEMO_FILES]
        out += _tree(q["finemo"] / "finemo_report")
        out += [q["narrowpeak"], q["peaks"], q["signal_qc"]]
        if (coop / ds).is_dir():
            out += _tree(coop / ds)
        if with_signal:
            out.append(q["bigwig"])
    seen, unique = set(), []
    for f in out:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique


class _Hashing(io.RawIOBase):
    """A file read through, md5 of what was read."""

    def __init__(self, fh):
        self.fh, self.md5 = fh, hashlib.md5()

    def readable(self) -> bool:
        return True

    def read(self, n: int = -1) -> bytes:
        b = self.fh.read(n)
        self.md5.update(b)
        return b


def _commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size, info.mtime, info.mode = len(data), int(time.time()), 0o644
    tar.addfile(info, io.BytesIO(data))


def write_share(p: dict, files: list[Path], out: Path) -> dict:
    root, top = Path(p["root"]), p["project"]
    present = [f for f in files if f.is_file()]
    missing = [f for f in files if not f.is_file()]
    sums, total = [], 0
    partial = out.with_name(out.name + ".partial")
    with tarfile.open(partial, "w", format=tarfile.PAX_FORMAT) as tar:
        members = [(f, f"{top}/{f.relative_to(root).as_posix()}") for f in present]
        if RESULTS_DOC.is_file():
            members.append((RESULTS_DOC, f"{top}/docs/results.md"))
        for src, arc in members:
            info = tar.gettarinfo(str(src), arcname=arc)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with open(src, "rb") as fh:
                h = _Hashing(fh)
                tar.addfile(info, h)
            sums.append(f"{h.md5.hexdigest()}  {arc}")
            total += info.size
        _add_bytes(tar, f"{top}/MANIFEST.md5", ("\n".join(sums) + "\n").encode())
        lines = [
            f"{top}: results packed by regulatory-variant-interpretation workflows/pipeline/share.py (RVI {_commit()})",
            f"from {root}, {time.strftime('%Y-%m-%d %H:%M %Z')}",
            f"{len(present)} files, {total:,} bytes; check them with: md5sum -c {top}/MANIFEST.md5 (from the "
            "directory the tar was unpacked in)",
            "how to read them: docs/results.md",
            "",
            f"missing at packing time ({len(missing)}):",
            *[f"  {f.relative_to(root).as_posix()}" for f in missing],
        ]
        _add_bytes(tar, f"{top}/SHARE.txt", ("\n".join(lines) + "\n").encode())
    os.replace(partial, out)
    return {"files": len(present), "bytes": total, "missing": [str(f) for f in missing], "out": str(out)}
