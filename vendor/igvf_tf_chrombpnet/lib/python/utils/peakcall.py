"""Peak calling the igvf_pseudobulking_pipeline way, on MACS3 PR #756.

The recipe, as kundajelab/igvf_pseudobulking_pipeline v2.0.1 (e8774b2) runs it
(modules/MACS3.nf, modules/CALL_PEAKS.nf, pseudobulk/src/pseudobulk/fragment.py,
pseudobulk/src/pseudobulk/tools/split_fragments.py):

1. Every fragment becomes its two Tn5 insertions, and each fragment goes to
   pseudoreplicate 1 or 2 with probability 1/2. repT is the two pooled.
2. ``macs3 callpeak -f BED -g hs -p 0.01 --shift -75 --extsize 150 --nomodel
   --keep-dup all --call-summits`` on each of rep1, rep2 and repT, given 1-bp
   insertion records (tag size 1, so max-gap 1 and min-length = d = 150).
3. Per rep, the 300k strongest summit rows by column 8 (-log10 p):
   ``sort -k 8g,8g | tail -n 300000``. (Before v2.0.1 this was ``sort
   --reverse -k 8gr,8gr | tail``, which kept the 300k WEAKEST rows -- a key
   with its own flags ignores the global --reverse, in GNU sort and in the
   image's uutils sort alike. PR #7 fixed it.)
4. repT rows overlapping a rep1 row AND a rep2 row, ``bedtools intersect -u
   -f 0.5 -F 0.5 -e``.
5. Drop rows touching the blacklist; score capped at 1000, End at chromsize.

What this module changes, deliberately -- each is recorded in call_peaks.json:

- **Insertions come from the shift actually in the fragments.** The pipeline
  adds +4/-5 unconditionally, which double-shifts 10x fragments (already
  +4/-5). Here the insertion is where chrombpnet's bigwig puts it: ``start +
  plus_delta`` and ``end + minus_delta - 1`` with the deltas from
  ``shift.shift_deltas`` (+4/-4 convention), so peaks and signal agree.
- **Deterministic pseudoreplicates**: one stream, one seeded RNG (the original
  fills its output from several threads).
- **No ``-B --SPMR``**: they only write the bedGraphs behind the p-value track.
- **MACS input defaults to the original 1-bp BED records.** MACS3 PR #756
  parses BAM/BAMPE/FRAG in C on threads but leaves the BED parser as it was;
  its C pileup, scoring and per-chromosome fork pools apply to BED input all
  the same. ``macs_input="frag"`` is a faster-parsing alternative that is NOT
  equivalent, and is kept only as an option: plain ``-f FRAG`` on fragments
  piles up whole fragments (paired-end mode zeroes --shift and replaces
  --extsize), so instead each insertion is written as a 150-bp "fragment"
  ``[cut-75, cut+75)`` with ``--max-gap 1 --min-length 150``. The treatment
  pileup then matches, but paired-end mode builds the local-lambda pileup
  around each fragment's two ENDS (PairedEndTrack.pileup_a_chromosome_c over
  ``locations['l'/'r']``), i.e. at insertion +-75 rather than at the
  insertion, and takes lambda_bg from total fragment length. On synthetic data
  peak coordinates and summits agreed while columns 7-9 differed by 0.1-1% and
  one rep gained a peak -- so it is not the pipeline's call.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pandas as pd

from utils import intervals

#: MACS3 flags shared by both input routes (modules/MACS3.nf:31-44, minus -B
#: --SPMR, which only write the bedGraphs behind a p-value track).
MACS_COMMON = ["-g", "hs", "-p", "0.01", "--nomodel", "--keep-dup", "all", "--call-summits"]
#: The original route: 1-bp records, single-end, tag size 1.
MACS_BED = ["-f", "BED", "--shift", "-75", "--extsize", "150"]
#: The fast route: 150-bp insertion windows as fragments; tag size 150 there,
#: so the max-gap/min-length that tag size 1 implied are given explicitly.
MACS_FRAG = ["-f", "FRAG", "--max-gap", "1", "--min-length", "150"]
#: Half of --extsize: an insertion's window is [cut - HALF_WINDOW, cut + HALF_WINDOW).
HALF_WINDOW = 75

TOP_N = 300_000
MIN_OVERLAP = 0.5
SCORE_CAP = 1000
SEED = 42

NARROWPEAK_COLUMNS = intervals.NARROWPEAK_OUT_COLUMNS

# Reads chrom.sizes first (FNR == NR), then the fragments on stdin. Per kept
# fragment: one rand() draw picks the pseudoreplicate, then both insertions go
# to that file -- a fragment is never split between reps, as in the original.
# mawk's RNG is deterministic for a given seed and mawk build; the version is
# recorded.
_SPLIT_AWK = r"""
BEGIN { FS = OFS = "\t"; srand(seed); kept = 0; skipped = 0 }
FNR == NR { keep[$1] = 1; next }
/^#/ { next }
!($1 in keep) { skipped++; next }
{
    out = (rand() < 0.5) ? out1 : out2
    p = $2 + dp
    m = $3 + dm - 1
    if (fmt == "frag") {
        s = p - h; if (s < 0) s = 0
        print $1, s, p + h, ".", 1 > out
        s = m - h; if (s < 0) s = 0
        print $1, s, m + h, ".", 1 > out
    } else {
        print $1, p, p + 1 > out
        print $1, m, m + 1 > out
    }
    kept++
}
END { printf "%d\t%d\n", kept, skipped > "/dev/stderr" }
"""


def split_insertions(
    fragments,
    chrom_sizes,
    rep1,
    rep2,
    plus_delta: int,
    minus_delta: int,
    macs_input: str = "bed",
    seed: int = SEED,
    threads: int = 4,
):
    """Stream fragments into two pseudoreplicate files of insertions, in one pass.

    ``bgzip -dc -@threads <fragments> | mawk``: contigs absent from
    ``chrom_sizes`` are dropped (the original keeps exactly those in its
    chr_sizes), ``#`` header lines are skipped. ``macs_input`` "frag" writes
    ``chrom, max(0, cut-75), cut+75, ".", 1`` per insertion; "bed" writes the
    original 1-bp ``chrom, cut, cut+1``.

    Returns ``(fragments_kept, fragments_skipped)``.
    """
    if macs_input not in ("frag", "bed"):
        raise ValueError(f"macs_input must be 'frag' or 'bed', got {macs_input!r}")
    decompress = subprocess.Popen(
        ["bgzip", "-dc", "-@", str(max(1, int(threads))), str(fragments)],
        stdout=subprocess.PIPE,
    )
    split = subprocess.run(
        [
            "mawk",
            "-v",
            f"seed={int(seed)}",
            "-v",
            f"dp={int(plus_delta)}",
            "-v",
            f"dm={int(minus_delta)}",
            "-v",
            f"h={HALF_WINDOW}",
            "-v",
            f"fmt={macs_input}",
            "-v",
            f"out1={rep1}",
            "-v",
            f"out2={rep2}",
            _SPLIT_AWK,
            str(chrom_sizes),
            "-",
        ],  # fmt: skip
        stdin=decompress.stdout,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    decompress.stdout.close()
    if decompress.wait() != 0:
        raise RuntimeError(f"bgzip -dc {fragments} exited {decompress.returncode}")
    if split.returncode != 0:
        raise RuntimeError(f"mawk split exited {split.returncode}: {split.stderr.strip()}")
    kept, skipped = (int(x) for x in split.stderr.strip().splitlines()[-1].split("\t"))
    return kept, skipped


def macs_command(name: str, treatment: list, outdir, macs_input: str = "bed", bedgraph=False):
    """The ``macs3 callpeak`` argv for one pseudoreplicate (or repT: two files)."""
    route = MACS_FRAG if macs_input == "frag" else MACS_BED
    cmd = ["macs3", "callpeak", "-t", *map(str, treatment), "-n", name, "--outdir", str(outdir)]
    cmd += route + MACS_COMMON
    if bedgraph:
        cmd += ["-B", "--SPMR"]
    return cmd


def run_concurrently(commands: dict, log_dir) -> dict:
    """Start every command at once, wait for all; return per-name wall time and max RSS.

    MACS3 PR #756 fork-pools per chromosome inside each call, so the three
    calls overlap their single-threaded parsing with each other's pools.
    ``os.wait4`` gives each child's own peak RSS. Raises if any call failed,
    after all have finished, naming every failure.
    """
    import time

    log_dir = Path(log_dir)
    procs = {}
    started = time.monotonic()
    for name, cmd in commands.items():
        log = open(log_dir / f"{name}.macs3.log", "w")  # noqa: SIM115  # closed below
        procs[name] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log)
    stats, failed = {}, []
    for name, (proc, log) in procs.items():
        _, status, usage = os.wait4(proc.pid, 0)
        proc.returncode = os.waitstatus_to_exitcode(status)
        log.close()
        stats[name] = {
            "exit": proc.returncode,
            "wall_s": round(time.monotonic() - started, 1),
            "max_rss_gb": round(usage.ru_maxrss / 1024**2, 2),
        }
        if proc.returncode != 0:
            failed.append(f"{name} (exit {proc.returncode}, see {log.name})")
    if failed:
        raise RuntimeError("macs3 callpeak failed: " + ", ".join(failed))
    return stats


def read_narrowpeak(path) -> pd.DataFrame:
    """A MACS3 narrowPeak as a DataFrame with chrombpnet's column names.

    ``_line`` keeps each row's text as MACS wrote it, which is what GNU sort
    compares when column 8 ties (``top_n_by_pvalue``); ``finalize`` drops it.
    """
    lines = [ln for ln in Path(path).read_bytes().splitlines() if ln and not ln.startswith(b"#")]
    df = pd.read_csv(path, sep="\t", header=None, names=NARROWPEAK_COLUMNS, comment="#")
    if len(df) != len(lines):
        raise ValueError(f"{path}: {len(df)} rows parsed but {len(lines)} lines read")
    df["Chromosome"] = df["Chromosome"].astype(str)
    df["_line"] = lines
    return df


def top_n_by_pvalue(peaks: pd.DataFrame, n: int = TOP_N) -> pd.DataFrame:
    """The ``n`` rows with the largest column 8, as ``sort -k 8g,8g | tail -n n``.

    Rows tied on column 8 are ordered by sort's last-resort comparison, the
    whole line ascending, bytewise (the containers set no locale), so at the
    cut the tail keeps the tied rows whose lines sort LAST. Reproduced from
    ``_line``, so the cut is the pipeline's to the row, not merely the same
    size. Rows come back in that ascending order.
    """
    key = pd.DataFrame(
        {"p": peaks["pvalue"].to_numpy(dtype=float), "line": peaks["_line"].to_numpy()}
    )
    order = key.sort_values(["p", "line"], kind="mergesort").index.to_numpy()
    return peaks.iloc[order[max(0, len(order) - n) :]]


def reproducible(rep_t: pd.DataFrame, rep1: pd.DataFrame, rep2: pd.DataFrame, fraction=MIN_OVERLAP):
    """repT rows overlapping some rep1 row and some rep2 row, either-fraction.

    ``bedtools intersect -u -a repT -b rep1 -f F -F F -e | bedtools intersect
    -u -a stdin -b rep2 -f F -F F -e``. Order and columns of repT are kept.
    """
    keep = intervals.overlaps_either_fraction(rep_t, rep1, fraction)
    keep &= intervals.overlaps_either_fraction(rep_t, rep2, fraction)
    return rep_t[keep]


def finalize(peaks: pd.DataFrame, blacklist, chrom_sizes: dict[str, int]) -> pd.DataFrame:
    """Blacklist, caps and order, as CALL_PEAKS.nf (v2.0.1) writes peaks.narrowPeak.

    Any overlap with the blacklist drops a row (``bedtools intersect -v``).
    Score is capped at 1000 and End at the chromosome length, as its awk does
    (``max_end[$1]=$2``). Rows come out in chrom.sizes order, then Start, then
    End (the order sort-bed.sh gives repT).
    """
    hit = intervals.overlaps_either_fraction(peaks, blacklist, fraction=None)
    out = peaks[~hit].copy()
    out["score"] = out["score"].clip(upper=SCORE_CAP)
    limit = out["Chromosome"].map(chrom_sizes)
    over = limit.notna() & (out["End"] > limit)
    out.loc[over, "End"] = limit[over].astype(out["End"].dtype)
    rank = {c: i for i, c in enumerate(chrom_sizes)}
    out["_rank"] = out["Chromosome"].map(rank).fillna(len(rank))
    out = out.sort_values(["_rank", "Start", "End"], kind="mergesort").drop(columns="_rank")
    return out[NARROWPEAK_COLUMNS]
