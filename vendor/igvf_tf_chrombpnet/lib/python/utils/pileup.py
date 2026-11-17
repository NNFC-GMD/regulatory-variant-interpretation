"""Tn5 insertion pileup to bigWig, without chrombpnet's two external sorts.

chrombpnet builds its training bigwig like this (reads_to_bigwig.generate_bigwig,
verbatim from 1.0.1)::

    awk (apply shift) | sort -k1,1 | bedtools genomecov -bg -5 -i stdin -g cs
                      | LC_COLLATE=C sort -k1,1 -k2,2n            -> bedGraph
    bedGraphToBigWig bedGraph cs out.bw

Two full external sorts of a text stream with two lines per fragment. Counting
cut sites into an array needs no sort at all: ``np.bincount`` is O(n), and the
bedGraph runs fall out of a single ``np.diff``. The bigWig is then written by
``pybigtools`` (Rust) rather than by piping through a UCSC binary.

**The output is required to be identical, not merely similar.**
``tests/test_pileup.py`` runs chrombpnet's own command string -- extracted from
the installed wheel, not paraphrased -- over the same input and asserts every
interval matches. These conventions were derived from that command's actual
output, not from reading it:

===========================  ==================================================
``+`` strand cut             ``start + plus_delta``
``-`` strand cut             ``end + minus_delta - 1``   (genomecov -5 on a
                             minus feature reports the last covered base)
cut position < 0             **error** -- bedtools rejects the record outright,
                             so silently clamping would diverge
cut position >= chrom length **dropped silently**, as bedtools does
adjacent equal values        **merged** into one interval, as ``-bg`` does
zero coverage                omitted, as ``-bg`` does
chromosome with no reads     contributes no intervals
===========================  ==================================================

This module does not detect the Tn5 shift. That stays chrombpnet's
``auto_shift_detect.compute_shift``, so only the pileup changes.
"""

from __future__ import annotations

import numpy as np

__all__ = ["CutSiteError", "cut_positions", "figwig_bigwig", "pileup_runs", "write_bigwig"]

# Fragment files can be huge; read them in chunks this size (rows).
CHUNK_ROWS = 5_000_000


class CutSiteError(ValueError):
    """A cut site fell off the start of a chromosome, as bedtools would reject."""


def cut_positions(
    starts: np.ndarray, ends: np.ndarray, plus_delta: int, minus_delta: int
) -> np.ndarray:
    """Tn5 cut sites for fragments, matching ``genomecov -5`` after the shift.

    Each fragment contributes two cuts, because
    ``auto_shift_detect.fragment_to_tagalign_stream`` emits it twice, once per
    strand. The ``-1`` on the minus side is the half-open BED convention: the 5'
    end of a minus-strand feature is its last covered base.
    """
    plus = starts.astype(np.int64) + plus_delta
    minus = ends.astype(np.int64) + minus_delta - 1
    return np.concatenate([plus, minus])


def cut_positions_stranded(starts, ends, is_reverse, plus_delta, minus_delta):
    """Cut sites for reads that each carry their OWN strand (a BAM).

    Fragments and BAMs differ here, and it is easy to get wrong.
    ``fragment_to_tagalign_stream`` emits every fragment TWICE, once as ``+``
    and once as ``-``, so a fragment yields two cuts. ``bedtools bamtobed``
    emits each read ONCE with its actual strand, so a read yields one:

        + read -> start + plus_delta
        - read -> end + minus_delta - 1
    """
    starts = np.asarray(starts, dtype=np.int64)
    ends = np.asarray(ends, dtype=np.int64)
    is_reverse = np.asarray(is_reverse, dtype=bool)
    return np.where(is_reverse, ends + minus_delta - 1, starts + plus_delta)


def pileup_runs(cuts: np.ndarray, chrom_length: int):
    """Count cuts per base and return the non-zero runs as (starts, ends, values).

    Adjacent bases with equal counts are merged and zero runs omitted, which is
    what ``bedtools genomecov -bg`` emits.
    """
    if cuts.size and cuts.min() < 0:
        bad = int(cuts.min())
        raise CutSiteError(
            f"cut site at {bad} is before the start of the chromosome. bedtools "
            "rejects such a record outright, so this is an error rather than a "
            "clamp -- the shift is probably wrong for this data."
        )
    # Off the far end is dropped silently, which is what bedtools does.
    cuts = cuts[cuts < chrom_length]
    if cuts.size == 0:
        empty_i = np.empty(0, dtype=np.int64)
        return empty_i, empty_i.copy(), np.empty(0, dtype=np.float32)

    counts = np.bincount(cuts, minlength=chrom_length)

    # Run boundaries: every index where the value changes, plus the two ends.
    change = np.flatnonzero(np.diff(counts)) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [counts.size]))
    values = counts[starts]

    keep = values > 0
    return starts[keep], ends[keep], values[keep].astype(np.float32)


def _iter_fragment_chunks(path, all_columns: bool, chunk_rows: int = CHUNK_ROWS):
    """Yield chunks of a fragments TSV, gz or plain.

    ``all_columns`` keeps barcode/count, which matters only when the filtered
    rows are written back out; the pileup needs the first three. Parsing three
    columns rather than five is meaningfully cheaper over hundreds of millions
    of rows.
    """
    import pandas as pd

    kwargs = {
        "sep": "\t",
        "header": None,
        "comment": "#",
        "chunksize": chunk_rows,
        "dtype": {0: str, 1: np.int64, 2: np.int64},
    }
    if not all_columns:
        kwargs["usecols"] = [0, 1, 2]
    yield from pd.read_csv(path, **kwargs)


def collect_cuts_bam(
    bam_path,
    chrom_sizes: dict[str, int],
    plus_delta,
    minus_delta,
):
    """Cut sites straight from a BAM, replacing `bedtools bamtobed`.

    One cut per mapped read, at its own strand's 5' end -- matching what
    bamtobed emits and what chrombpnet would therefore have piled up. Unmapped
    reads are skipped, as bamtobed skips them.
    """
    import pysam  # noqa: PLC0415  # only needed for BAM input

    per_chrom: dict[str, list[np.ndarray]] = {}
    skipped: dict[str, int] = {}
    kept = 0
    buf: dict[str, list[tuple[int, int, bool]]] = {}

    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        for read in bam.fetch(until_eof=True):
            if read.is_unmapped:
                continue
            chrom = read.reference_name
            if chrom not in chrom_sizes:
                skipped[chrom] = skipped.get(chrom, 0) + 1
                continue
            buf.setdefault(chrom, []).append(
                (read.reference_start, read.reference_end, read.is_reverse)
            )
            kept += 1

    for chrom, rows in buf.items():
        arr = np.array(rows, dtype=object)
        per_chrom[chrom] = cut_positions_stranded(
            np.array([r[0] for r in rows], dtype=np.int64),
            np.array([r[1] for r in rows], dtype=np.int64),
            np.array([r[2] for r in rows], dtype=bool),
            plus_delta,
            minus_delta,
        )
        del arr
    return per_chrom, skipped, kept


def collect_cuts(
    fragments_path,
    chrom_sizes: dict[str, int],
    plus_delta,
    minus_delta,
    write_filtered=None,
):
    """Filter to the main chromosomes AND count cut sites in a SINGLE pass.

    Dropping rows on contigs absent from ``chrom_sizes`` *is* the
    main-chromosome filter, and the pileup has to look at every row anyway, so
    running the filter as a separate pass over a multi-GB file buys nothing.

    ``write_filtered`` additionally writes the kept rows out (bgzipped when the
    path ends in .gz), preserving every column so barcodes survive. That file is
    only needed for the fallback where chrombpnet reads the reads itself; omit
    it and nothing is rewritten at all.

    Returns ``(cuts_by_chrom, skipped_by_contig, kept_row_count)``. Memory is
    proportional to the number of cut sites, not to the genome.
    """
    per_chrom: dict[str, list[np.ndarray]] = {}
    skipped: dict[str, int] = {}
    kept_rows = 0

    out_fh = None
    if write_filtered is not None:
        from utils import compression

        out_fh = compression.open_write(write_filtered)
    try:
        for chunk in _iter_fragment_chunks(fragments_path, all_columns=out_fh is not None):
            keep = chunk[0].isin(chrom_sizes.keys())
            if not keep.all():
                for contig, n in chunk.loc[~keep, 0].value_counts().items():
                    skipped[contig] = skipped.get(contig, 0) + int(n)
            kept = chunk[keep]
            if kept.empty:
                continue
            kept_rows += len(kept)

            if out_fh is not None:
                out_fh.write(kept.to_csv(sep="\t", header=False, index=False).encode())

            for chrom, grp in kept.groupby(0, sort=False):
                per_chrom.setdefault(chrom, []).append(
                    cut_positions(grp[1].to_numpy(), grp[2].to_numpy(), plus_delta, minus_delta)
                )
    finally:
        if out_fh is not None:
            out_fh.close()

    return {c: np.concatenate(v) for c, v in per_chrom.items()}, skipped, kept_rows


def write_bigwig(out_path, chrom_sizes: dict[str, int], cuts_by_chrom: dict[str, np.ndarray]):
    """Write the pileup as a bigWig via pybigtools.

    Chromosomes are emitted in ``chrom_sizes`` order and positions ascending,
    which is what the writer requires; getting it wrong raises rather than
    producing a subtly broken file.
    """
    import pybigtools

    def intervals():
        for chrom, length in chrom_sizes.items():
            cuts = cuts_by_chrom.get(chrom)
            if cuts is None or cuts.size == 0:
                continue
            starts, ends, values = pileup_runs(cuts, length)
            for s, e, v in zip(starts.tolist(), ends.tolist(), values.tolist(), strict=True):
                yield chrom, s, e, v

    writer = pybigtools.open(str(out_path), "w")
    try:
        writer.write(dict(chrom_sizes), intervals())
    finally:
        try:
            writer.close()
        except Exception:  # noqa: BLE001  # already closed by write() in some versions
            pass
    return out_path


def figwig_bigwig(
    fragments_path,
    chrom_sizes_path,
    out_path,
    plus_delta: int,
    minus_delta: int,
    threads: int = 8,
):
    """The same per-base cut counts as ``collect_cuts`` + ``write_bigwig``, via figwig.

    ``figwig bam2bw -f -u`` counts both ends of every fragment, at
    ``start + ps`` and ``end + ns - 1`` -- exactly ``cut_positions`` with
    ``ps = plus_delta`` and ``ns = minus_delta`` -- into one unstranded bigWig.
    Contigs absent from ``chrom_sizes_path`` are dropped, which makes the
    main-chromosome filter implicit, as it is for ``collect_cuts``. It reads a
    BGZF fragments file on ``threads`` cores (the numpy path parses it with
    pandas on one), so on 10x files of several hundred million fragments it is
    the fast path.

    Two differences from ``write_bigwig``, neither of them in the values:
    figwig writes single-base (varStep) entries where pybigtools writes merged
    runs, so the files differ in bytes but decode to the same per-base values
    (``tests/test_pileup.py`` compares values, never checksums); and a cut that
    a negative delta would push before base 0 is dropped and reported rather
    than raised, which no ATAC/DNASE delta from ``shift.shift_deltas`` can
    produce for well-formed fragments.

    Runs ``python -m figwig`` from the CURRENT interpreter, so it is the
    figwig of whichever environment the step activated. Returns ``out_path``.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    out_path = Path(out_path)
    # figwig names its output <name>.bw; write beside the target under a
    # temporary name and rename, so a killed run never leaves a truncated
    # data_unstranded.bw that the next run would take as finished.
    partial = out_path.with_name(f".{out_path.stem}.partial")
    produced = partial.with_name(partial.name + ".bw")
    produced.unlink(missing_ok=True)
    cmd = [
        sys.executable, "-m", "figwig", "bam2bw", str(fragments_path),
        "-s", str(chrom_sizes_path),
        "-n", str(partial),
        "-f", "-u",
        "-ps", str(int(plus_delta)),
        "-ns", str(int(minus_delta)),
        "-p", str(max(1, int(threads))),
    ]  # fmt: skip
    # numpy's OpenBLAS otherwise starts a thread per host CPU on import, which
    # figwig's docs flag for many concurrent bam2bw processes on a large node.
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1"}
    subprocess.run(cmd, check=True, env=env)
    if not produced.is_file():
        raise RuntimeError(f"figwig bam2bw wrote no {produced}")
    os.replace(produced, out_path)
    return out_path
