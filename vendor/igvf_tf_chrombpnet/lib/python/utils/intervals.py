"""Genomic interval operations, replacing the pipeline's bedtools calls.

**The distribution is `pyranges1` and so is the module: `import pyranges1 as pr`.**
`import pyranges` gets you the unrelated 0.x line (PyPI `pyranges` stops at 0.1.4).
PyRanges 1 is backed by `ruranges`, a Rust extension, and a `PyRanges` is a
`pandas.DataFrame` subclass — there is no `.df` attribute, you index it directly.

**This is the only module in `utils` allowed to import `pyranges1`.** It needs
Python >= 3.12 and is installed only in the pixi `preprocess` (and `qc`)
environment. Keeping the import here means `folds`, `palettes`, `plotting` and
`regions` stay importable from the chrombpnet env, where the QC scripts run.

Each function documents the bedtools command it replaces. The equivalences are
pinned by `tests/test_intervals.py`; three of them are not obvious:

- `bedtools intersect -v` requires >= 1bp of overlap, so book-ended intervals are
  kept. `overlap(invert=True)` agrees at its default `slack=0`.
- `bedtools slop` clamps to `[0, chromlen]` and never drops an interval.
  `clip_ranges(chromsizes)` agrees (`remove=False` is the default).
- narrowPeak's summit is an offset from `Start`, floor-divided.

Two ways in: a plain BED of regions, summit = midpoint (`read_bed`), or a
peak caller's narrowPeak re-centred on its OWN summit (`read_narrowpeak` +
`summit_windows`), which is what a summit-anchored model wants.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pyranges1 as pr

from utils import references

# The 10-column narrowPeak layout chrombpnet expects, in PyRanges' column names.
# Kept separate from regions.NARROWPEAK_SCHEMA, which names the same columns the
# way pandas reads the file (chr/start/end); PyRanges requires Chromosome/Start/End.
NARROWPEAK_OUT_COLUMNS = [
    "Chromosome",
    "Start",
    "End",
    "name",
    "score",
    "strand",
    "signal",
    "pvalue",
    "qvalue",
    "summit",
]


def read_bed(path) -> pr.PyRanges:
    """Read a local BED (optionally gzipped) into a PyRanges. Extra columns are kept."""
    return pr.read_bed(str(path))


def read_narrowpeak(path) -> pr.PyRanges:
    """Read a 10-column narrowPeak (e.g. MACS2) with its columns named.

    ``read_bed`` would name columns 7-10 after BED12's thickStart/thickEnd/
    itemRgb/blockCount, so the summit would be read as a "blockCount". This
    names them for what narrowPeak says they are and fails on any other shape.
    """
    df = pd.read_csv(path, sep="\t", header=None, comment="#")
    if df.shape[1] != 10:
        raise ValueError(f"{path}: expected a 10-column narrowPeak, found {df.shape[1]} columns")
    df.columns = NARROWPEAK_OUT_COLUMNS
    return pr.PyRanges(df)


def summit_windows(peaks: pr.PyRanges, window: int, max_qvalue: float | None = None):
    """Re-centre narrowPeak rows on their summit: ``[summit - w//2, summit + w - w//2)``.

    For ``window=1000`` that is 500 bp either side, i.e. bases summit-500 ..
    summit+499 inclusive. Every row is a window of its own, so a region MACS2
    called with several summits yields several (overlapping) windows -- one per
    summit, as chrombpnet's reference preprocessing does. Afterwards the
    midpoint IS the summit, so ``summit_offsets`` and ``to_narrowpeak`` need no
    special case.

    ``max_qvalue`` keeps rows with q <= it, reading column 9 as -log10(q) the
    way MACS2 writes it; a file with q = -1 (not computed) cannot be filtered
    and raises. Windows that would start before base 0 are dropped.

    Returns ``(windows, {"qvalue": n_dropped, "before_chrom_start": n_dropped})``.
    """
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    dropped = {"qvalue": 0, "before_chrom_start": 0}
    if max_qvalue is not None:
        if not 0.0 < max_qvalue <= 1.0:
            raise ValueError(f"max_qvalue must be in (0, 1], got {max_qvalue}")
        if (peaks["qvalue"] < 0).all():
            raise ValueError("no q-values in this narrowPeak (column 9 is -1); cannot filter on q")
        keep = (peaks["qvalue"] >= -math.log10(max_qvalue)).to_numpy()
        dropped["qvalue"] = int((~keep).sum())
        peaks = peaks[keep]
    summit = (peaks["Start"] + peaks["summit"]).to_numpy()
    start = summit - window // 2
    ok = start >= 0
    dropped["before_chrom_start"] = int((~ok).sum())
    out = peaks[ok].copy()
    out["Start"] = start[ok]
    out["End"] = start[ok] + window
    out["summit"] = window // 2
    return pr.PyRanges(out), dropped


def read_bed3(source) -> pr.PyRanges:
    """Read the first three BED columns from a path, URL, or ENCODE accession.

    Goes through pandas rather than ``pr.read_bed`` so that a URL works, and
    takes only chrom/start/end because that is all an interval filter needs --
    the ENCODE blacklist's 4th column is a reason string ("Low Mappability").
    """
    src = references.resolve_bed_source(source)
    try:
        df = pd.read_csv(
            src,
            sep="\t",
            header=None,
            comment="#",
            usecols=[0, 1, 2],
            names=["Chromosome", "Start", "End"],
        )
    except OSError as exc:  # network failure, or a path that is not there
        raise OSError(
            f"could not read {references.describe_source(source)}: {exc}\n"
            "If this is an accession and the job runs on a compute node without "
            "outbound network access, pass a local path instead."
        ) from exc
    return pr.PyRanges(df)


def remove_blacklisted(peaks: pr.PyRanges, blacklist: pr.PyRanges) -> pr.PyRanges:
    """Drop every peak overlapping ``blacklist``.

    Replaces ``bedtools intersect -v -a peaks -b blacklist``. Book-ended
    intervals do not overlap and are kept, matching bedtools.
    """
    return peaks.overlap(blacklist, invert=True)


def overlaps_either_fraction(a, b, fraction: float | None) -> np.ndarray:
    """Which rows of ``a`` some row of ``b`` overlaps, as a boolean mask over ``a``.

    Replaces ``bedtools intersect -u -a A -b B -f F -F F -e``: a row of A is
    reported (once) when any B row overlaps it by at least F of A's length OR
    at least F of B's length -- ``-e`` is the "or". ``fraction=None`` is plain
    ``-u``, at least 1 bp, so ``~mask`` is ``intersect -v``; book-ended
    intervals do not overlap in either case. ``a`` and ``b`` are anything with
    Chromosome/Start/End columns (a DataFrame or a PyRanges); the mask follows
    ``a``'s row order, whatever its index.
    """
    left = pr.PyRanges(
        pd.DataFrame(
            {
                "Chromosome": np.asarray(a["Chromosome"], dtype=str),
                "Start": np.asarray(a["Start"], dtype=np.int64),
                "End": np.asarray(a["End"], dtype=np.int64),
                "_row": np.arange(len(a)),
            }
        )
    )
    right = pr.PyRanges(
        pd.DataFrame(
            {
                "Chromosome": np.asarray(b["Chromosome"], dtype=str),
                "Start": np.asarray(b["Start"], dtype=np.int64),
                "End": np.asarray(b["End"], dtype=np.int64),
            }
        )
    )
    mask = np.zeros(len(a), dtype=bool)
    if len(left) == 0 or len(right) == 0:
        return mask
    hits = left.join_overlaps(right, suffix="_b")
    if len(hits) == 0:
        return mask
    rows = hits["_row"].to_numpy()
    if fraction is not None:
        start, end = hits["Start"].to_numpy(), hits["End"].to_numpy()
        start_b, end_b = hits["Start_b"].to_numpy(), hits["End_b"].to_numpy()
        overlap = np.minimum(end, end_b) - np.maximum(start, start_b)
        ok = (overlap >= fraction * (end - start)) | (overlap >= fraction * (end_b - start_b))
        rows = rows[ok]
    mask[rows] = True
    return mask


def slop(ranges: pr.PyRanges, bp: int, chromsizes: dict[str, int]) -> pr.PyRanges:
    """Extend every interval by ``bp`` on both sides, clamped to the chromosome.

    Replaces ``bedtools slop -b <bp> -g <chrom.sizes>``. Intervals running off
    either end are clamped to ``[0, chromlen]``, never dropped.
    """
    return ranges.extend_ranges(ext=bp).clip_ranges(chromsizes=chromsizes)


def summit_offsets(peaks: pr.PyRanges):
    """narrowPeak summit: the midpoint, as a floored offset from Start.

    One definition, used both to write the summit column and to decide whether
    a peak's model window fits on the chromosome -- those two must agree or the
    window check tests a different position from the one chrombpnet reads.
    """
    return (peaks["End"] - peaks["Start"]) // 2


def restrict_to_chromosomes(peaks: pr.PyRanges, chromsizes: dict[str, int]):
    """Drop peaks on contigs absent from ``chromsizes``.

    Handed the same main-chromosome chrom.sizes the signal pileup uses, this
    keeps peaks and signal on the same contigs. Without it a peak can sit on a
    scaffold the bigwig does not cover, and chrombpnet trains on a region with
    no data under it.
    """
    keep = peaks["Chromosome"].astype(str).isin(set(chromsizes)).to_numpy()
    return peaks[keep], int((~keep).sum())


def drop_windows_off_chromosome(peaks: pr.PyRanges, chromsizes: dict[str, int], input_window: int):
    """Drop peaks whose model input window runs past either chromosome end.

    chrombpnet extracts ``input_window`` bases centred on ``start + summit``.
    When that window overhangs, ``chrombpnet contribs_bw`` silently drops the
    peak later -- which is why interpretation.interpreted_regions.bed has fewer
    rows than the peak file it was given, and why steps 07/10 have to take
    their regions from that file instead. Dropping them here instead makes the
    peak set stable from step 01 onward, and counts them out loud.
    """
    half = input_window // 2
    centre = (peaks["Start"] + summit_offsets(peaks)).to_numpy()
    # PyRanges keeps Chromosome as a pandas Categorical, and mapping one yields
    # a Categorical that cannot be compared with <=. Go through plain arrays.
    lengths = peaks["Chromosome"].astype(str).map(chromsizes).to_numpy(dtype="int64")
    fits = (centre - half >= 0) & (centre + half <= lengths)
    return peaks[fits], int((~fits).sum())


def to_narrowpeak(peaks: pr.PyRanges) -> pd.DataFrame:
    """Build the 10-column narrowPeak chrombpnet consumes, summit at the midpoint.

    (After ``summit_windows`` the midpoint is the caller's summit.)

    Replaces the awk in the old ``00.1.preprocess_peaks.sh``:

        summit = int(($3-$2)/2); print $1,$2,$3,"peak_"NR,0,".",0,-1,-1,summit

    ``NR`` numbered the rows *after* blacklist filtering, so names are assigned
    over the filtered set, 1-based. The summit is an offset from ``Start``, and
    the division floors.
    """
    out = pd.DataFrame(
        {
            "Chromosome": peaks["Chromosome"].to_numpy(),
            "Start": peaks["Start"].to_numpy(),
            "End": peaks["End"].to_numpy(),
        }
    )
    out["name"] = [f"peak_{i}" for i in range(1, len(out) + 1)]
    out["score"] = 0
    out["strand"] = "."
    # A caller's narrowPeak (read_narrowpeak) keeps its own signal/p/q; a
    # plain BED has none, and gets the awk's placeholders.
    for col, placeholder in (("signal", 0), ("pvalue", -1), ("qvalue", -1)):
        out[col] = peaks[col].to_numpy() if col in peaks.columns else placeholder
    out["summit"] = (out["End"] - out["Start"]) // 2  # see summit_offsets()
    return out[NARROWPEAK_OUT_COLUMNS]


def read_chromsizes(path) -> dict[str, int]:
    """Read a 2-column chrom.sizes TSV into the dict ``slop`` wants."""
    df = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1], names=["chrom", "size"])
    return dict(zip(df["chrom"], df["size"], strict=True))


def count_in_peaks(fragments: pr.PyRanges, peaks: pr.PyRanges) -> int:
    """Number of fragments overlapping at least one peak (the numerator of FRIP).

    Replaces ``bedtools intersect -a fragments -b peaks -u | wc -l``. ``-u``
    reports each fragment at most once however many peaks it hits, which is what
    ``overlap`` (not ``join_overlaps``) does.
    """
    return len(fragments.overlap(peaks))
