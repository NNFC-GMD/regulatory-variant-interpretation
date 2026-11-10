"""BGZF (bgzip) output, so the pipeline's .gz files can be tabix-indexed.

Everything this pipeline writes compressed is written with bgzip rather than
plain gzip. **A bgzip file is a valid gzip file** -- a concatenation of gzip
members with a BGZF extra field -- so `zcat`, `gzip.open`, pandas and
chrombpnet read it unchanged. What it adds is block boundaries, which is what
tabix needs to random-access a region instead of decompressing from the start.

That matters here because a filtered fragments file is several GB and the
per-chromosome work downstream would otherwise stream all of it. Step 10 already
did this by hand (`bgzip -c hits.bed | tabix -p bed`); this module is the same
thing for the Python steps.

Writing goes through `pysam.BGZFile`, so bgzip is a library call rather than a
shell-out to htslib -- no `ml biology samtools` needed just to compress a file.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pysam


def open_write(path, bgzip: bool = True):
    """Open ``path`` for binary writing, bgzip-compressed when it ends in ``.gz``.

    Pass ``bgzip=False`` to fall back to plain gzip; the only reason to do that
    is a consumer that rejects multi-member gzip, which nothing here does.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix != ".gz":
        return open(path, "wb")
    return pysam.BGZFile(str(path), "wb") if bgzip else gzip.open(path, "wb")


def write_tsv(df, path, bgzip: bool = True) -> None:
    """Write a DataFrame as headerless TSV, bgzip-compressed when ``path`` is ``.gz``."""
    payload = df.to_csv(sep="\t", header=False, index=False).encode()
    with open_write(path, bgzip=bgzip) as fh:
        fh.write(payload)


def tabix_index(path, preset: str = "bed") -> Path:
    """Build a ``.tbi`` next to a bgzipped ``path``. Requires bgzip, not plain gzip.

    The file must be coordinate-sorted, which is why callers sort first.
    """
    pysam.tabix_index(str(path), preset=preset, force=True)
    return Path(f"{path}.tbi")


def is_bgzf(path) -> bool:
    """True if ``path`` is BGZF (block-compressed), not merely gzip.

    Checks for the ``BC`` extra subfield in the first gzip member's header.
    """
    with open(path, "rb") as fh:
        head = fh.read(18)
    return len(head) >= 14 and head[:2] == b"\x1f\x8b" and head[12:14] == b"BC"
