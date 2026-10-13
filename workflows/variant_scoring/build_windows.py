#!/usr/bin/env python3
"""Build the 2114-bp windows every model scores, once, as uint8 base codes.

Every model of every fold reads the arrays written here, so "same windows, same null, same AAQ peaks" holds by
construction. Codes are A=0, C=1, G=2, T=3 and 4 for anything else (N, IUPAC codes). One-hot them as eye(5)[:, :4],
which gives code 4 an all-zero column, as VS's one_hot.dna_to_one_hot does (src/utils/one_hot.py:16, 31).

VS is kundajelab/variant-scorer, main @0e1e341; "VS file:line" refers to that commit.

Variant windows (default mode), from a variant list
  For each row, the allele1 (REF) and allele2 (ALT) windows of VS's VariantGenerator.__get_allele_seq__
  (src/generators/variant_generator.py:28-76), ported line for line:
    pos0 = pos - 1 (:39); the allele starts at index 1057 of the 2114-bp window (output position 500 of both models).
    SNVs and MNVs: flank = genome[pos0 - 1057, pos0 + 1057); each allele overwrites flank[1057, 1057 + len) (:41-47).
    Indels: the left flank genome[pos0 - 1057, pos0) is kept; a deletion reads len(ref) - len(alt) extra bases on the
      right and an insertion truncates the right flank, so both windows stay 2114 bp (:52-72).
  Where this departs from VS, it refuses or drops rows that VS would score wrongly or fail on:
    REF is checked against the FASTA for every row. VS checks it for indels only (:55), so a wrong or swapped REF on an
      SNV would silently flip its logFC. A mismatch stops the run before anything is written, unless --drop-invalid,
      which drops the row and counts it. The same applies to alleles that are not ACGT, REF = ALT, and chromosomes
      the FASTA lacks. VS's io.py:22-37 also allows "-", which this pipeline's anchored VCF alleles never use.
    Windows that run off a chromosome are dropped and counted. The test is VS's get_valid_variants
      (src/utils/helpers.py:28-42: pos - 1057 > 0 and pos + 1057 <= length), plus the deletion's extra right-flank
      bases, which VS does not check; its length assert (variant_generator.py:74-75) would fail instead.
    Alleles longer than 1057 bp are dropped and counted: VS's construction cannot keep their windows at 2114 bp.
    Windows that hold a base other than ACGT are kept and flagged (`nonacgt`), as VS keeps them (as all-zero columns).

Null windows (--null), from the same variant list
  VS's null takes rows of the list itself, each with its own seed (create_shuffle_table, helpers.py:293-317). For each
  row, the flank, including a deletion's extra bases, is dinucleotide-shuffled, and BOTH alleles are then written back
  at 1057 exactly as above (VariantGenerator with shuf=True, variant_generator.py:43-47, 62-72). A null row therefore
  has an allele1 and an allele2 window, and its logFC is log2(c2 / c1) in the shuffled background
  (variant_scoring.py:77-89, 150-163). The npz holds both windows, and both must be scored.
  How this builder does it:
    The pool is the rows of --null-strata (default summit100,flank500) that pass every check above and whose whole
      flank is ACGT. tangermeme's shuffle reads an all-zero column as A (tangermeme ersatz.py:650-656).
    --n rows are drawn with numpy's default_rng(--seed): without replacement when the pool is larger than --n, with
      replacement otherwise, VS's rule (helpers.py:294-304). VS draws with pandas' sampler, which is not reproduced.
    Row r is shuffled in its own call, dinucleotide_shuffle(X, n=1, random_state=--seed + r). tangermeme seeds the
      i-th sequence of a call with random_state + i (ersatz.py:701-703), so one row per call keeps each row's shuffle
      independent of batching. Every shuffle is checked to keep the flank's dinucleotide counts and its first and last
      base, and the .json counts the shuffles that returned the flank unchanged.
    VS shuffles with deeplift's dinuc_shuffle (variant_generator.py:7, 45, 64), which is in neither environment.
      tangermeme's is adapted from it (ersatz.py:643-645), but its random draws differ, so this null matches VS's in
      distribution, not row for row.
    VS also shuffles the flank as pyfaidx returns it, soft-masked (pyfaidx.Fasta keeps case, variant_generator.py:22),
      and dinuc_shuffle takes each distinct character as its own token, so a lowercase run keeps case-specific
      dinucleotide counts; the case is dropped only at one-hot time (one_hot.py:22). This builder upper-cases the
      flank before shuffling. That too changes the null in distribution only, so p-values can be compared with VS's
      as distributions, never row for row (the pilot's V2 runs VS without a null).
  Needs torch and tangermeme (the pixi `cherimoya` environment).

Peak windows (--peaks), from a narrowPeak, the AAQ reference
  Summit-centred windows genome[summit - 1057, summit + 1057): ChromBPNet's training window (chrombpnet
  training/utils/data_utils.py:37). summit = start + column 10, or start + (end - start) // 2 when the column is
  absent (VS io.py:88-89) or -1.
  VS's PeakGenerator instead reads pyfaidx's 1-based, closed get_seq(chrom, summit - 1057, summit + 1056)
  (peak_generator.py:25-32), which starts 1 bp further left. --vs-peak-window reproduces that, for a check against VS.
  The rest follows VS (variant_scoring.py:106-108, 120-123): duplicate (chrom, start, end, summit) rows are dropped,
  get_valid_peaks' bounds test is applied (helpers.py:16-26), and when more peaks remain than --n, --n of them are
  drawn without replacement with default_rng(--seed). Peak windows have allele1 only.

Outputs, for --out <dir>/<name>.npz
  <name>.npz          a1, and a2 except for peaks: uint8 [N, 2114]; ids: str [N]; nonacgt: bool [N], true when a
                      window holds a base other than ACGT. --null adds src_ids (the list row each null row came from)
                      and seeds (int64). --peaks adds chrom and summit (0-based).
  <name>.json         settings; counts (input, kept, dropped by reason, non-ACGT); the FASTA's path, size and md5;
                      the VS commit followed.
  <name>.dropped.tsv  every dropped row: id, reason, detail.
  The .npz is written last, under a temporary name that is then renamed, so its presence means the build finished.

Variant list: a TSV with a header row. It needs at least variant_id, chrom, pos (1-based, as in a VCF), ref and alt
(anchored VCF alleles, as `bcftools norm -m -any` writes them), plus stratum for --null; other columns are ignored.
The FASTA needs its .fai (samtools faidx).

    python workflows/variant_scoring/build_windows.py --list <list.tsv> --fasta <genome.fa> --out <dir>/<name>.npz
    python workflows/variant_scoring/build_windows.py --null --list <list.tsv> --fasta <genome.fa> \\
        --n 10000 --seed 1234 --out <dir>/null.npz
    python workflows/variant_scoring/build_windows.py --peaks <peaks.narrowPeak[.gz]> --fasta <genome.fa> \\
        --n 10000 --seed 1234 --out <dir>/peaks.<dataset>.npz

Needs numpy only, except --null, which needs torch and tangermeme. Single-threaded. The arrays take about
2 x 2114 bytes of RAM per row; reading the FASTA's md5 takes a few seconds per GB.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np

logger = logging.getLogger("build_windows")

INPUT_LEN = 2114
FLANK = INPUT_LEN // 2  # 1057, the index the allele starts at (VS variant_generator.py:24, 46-47)
VS_COMMIT = "kundajelab/variant-scorer@0e1e34199e63112aa618748bb79a206fc491300a"
REQUIRED = ("variant_id", "chrom", "pos", "ref", "alt")
NULL_STRATA = ("summit100", "flank500")
# Refused unless --drop-invalid: the list is wrong for this FASTA.
INVALID = ("bad_allele", "unknown_chrom", "ref_mismatch")
# Always dropped and counted: VS cannot build these windows.
UNBUILDABLE = ("allele_too_long", "off_chrom_end")

_ALLELE = re.compile(r"[ACGT]+")

# A, C, G, T in either case -> 0-3; any other byte -> 4.
CODE = np.full(256, 4, dtype=np.uint8)
for _i, _b in enumerate(b"ACGT"):
    CODE[_b] = CODE[_b + 32] = _i
BASES = np.frombuffer(b"ACGTN", dtype=np.uint8)

# A shuffler takes one flank's codes (all < 4) and a seed, and returns the shuffled codes.
Shuffler = Callable[[np.ndarray, int], np.ndarray]


def encode(seq: bytes) -> np.ndarray:
    return CODE[np.frombuffer(seq, dtype=np.uint8)]


def decode(codes: np.ndarray) -> str:
    return BASES[np.asarray(codes)].tobytes().decode()


class IndexedFasta:
    """[start, end) of a sequence (0-based, half-open) from a FASTA and its .fai, upper-cased.

    A region outside the sequence raises an error. pyfaidx's slicing instead wraps a negative start around to the
    end of the sequence and cuts an end that runs past it short."""

    def __init__(self, path: Path):
        self.path = Path(path)
        fai = Path(f"{self.path}.fai")
        if not fai.exists():
            raise FileNotFoundError(f"{fai} not found; index the FASTA with `samtools faidx`")
        self.index: dict[str, tuple[int, int, int, int]] = {}
        with open(fai) as fh:
            for line in fh:
                name, length, offset, line_bases, line_width = line.rstrip("\n").split("\t")[:5]
                self.index[name] = (int(length), int(offset), int(line_bases), int(line_width))
        self._fh = open(self.path, "rb")

    def __contains__(self, chrom: str) -> bool:
        return chrom in self.index

    def length(self, chrom: str) -> int:
        return self.index[chrom][0]

    def fetch(self, chrom: str, start: int, end: int) -> bytes:
        length, offset, line_bases, line_width = self.index[chrom]
        if not 0 <= start <= end <= length:
            raise ValueError(f"{chrom}:{start}-{end} is not inside {chrom} (length {length})")
        first = offset + (start // line_bases) * line_width + start % line_bases
        last = offset + (end // line_bases) * line_width + end % line_bases
        self._fh.seek(first)
        seq = self._fh.read(last - first).replace(b"\n", b"").replace(b"\r", b"").upper()
        if len(seq) != end - start:
            raise ValueError(f"read {len(seq)} bases for {chrom}:{start}-{end}; does {self.path}.fai match it?")
        return seq

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> IndexedFasta:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def file_md5(path: Path, block: int = 1 << 24) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        while chunk := fh.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def _open_text(path: Path):
    return gzip.open(path, "rt", newline="") if path.suffix == ".gz" else open(path, newline="")


def read_list(path: Path, required: tuple[str, ...] = REQUIRED) -> list[dict]:
    """The variant list's rows, as dicts keyed by its header. variant_id must be unique."""
    with _open_text(path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        missing = [c for c in required if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: no column(s) {', '.join(missing)} in its header {reader.fieldnames}")
        rows = list(reader)
    repeated = [vid for vid, count in Counter(r["variant_id"] for r in rows).items() if count > 1]
    if repeated:
        raise ValueError(f"{path}: {len(repeated)} variant_id(s) appear more than once, e.g. {repeated[0]}")
    return rows


@dataclass(frozen=True)
class Entry:
    """A row that passed every check, with the genome VS reads for it."""

    vid: str
    chrom: str
    pos: int
    ref: bytes
    alt: bytes
    stratum: str
    flank: bytes  # genome[flank_span(...)], upper-cased


def flank_span(pos: int, ref: bytes, alt: bytes) -> tuple[int, int]:
    """0-based [start, end) of the genome VS reads for a variant (variant_generator.py:39-42, 56-60)."""
    pos0 = pos - 1
    return pos0 - FLANK, pos0 + FLANK + max(len(ref) - len(alt), 0)


def allele_windows(flank: bytes, ref: bytes, alt: bytes) -> tuple[bytes, bytes]:
    """VS's __get_allele_seq__ once the flank is read, and for the null shuffled (variant_generator.py:41-75)."""
    if len(ref) == len(alt):
        a1 = flank[:FLANK] + ref + flank[FLANK + len(ref) :]
        a2 = flank[:FLANK] + alt + flank[FLANK + len(alt) :]
    else:
        mismatch = len(ref) - len(alt)
        left = flank[:FLANK]
        a1 = left + ref + flank[FLANK + len(ref) : 2 * FLANK]
        a2 = left + alt + flank[FLANK + len(ref) : 2 * FLANK + mismatch]
    if len(a1) != INPUT_LEN or len(a2) != INPUT_LEN:
        raise ValueError(f"windows of {len(a1)} and {len(a2)} bp for {ref!r}>{alt!r}; expected {INPUT_LEN}")
    return a1, a2


def prepare(rows: list[dict], fasta: IndexedFasta) -> tuple[list[Entry], list[tuple[str, str, str]]]:
    """Check every row; return the rows that can be built and (id, reason, detail) for the others."""
    kept: list[Entry] = []
    dropped: list[tuple[str, str, str]] = []
    for row in rows:
        vid, chrom = row["variant_id"], row["chrom"]
        try:
            pos = int(row["pos"])
        except (TypeError, ValueError):
            raise ValueError(f"{vid}: pos {row['pos']!r} is not an integer") from None
        if pos < 1:
            raise ValueError(f"{vid}: pos {pos} is not 1-based")
        ref_s, alt_s = (row["ref"] or "").strip().upper(), (row["alt"] or "").strip().upper()
        ref, alt = ref_s.encode(), alt_s.encode()
        reason = detail = ""
        if not (_ALLELE.fullmatch(ref_s) and _ALLELE.fullmatch(alt_s)) or ref_s == alt_s:
            reason, detail = "bad_allele", f"{ref_s or '(empty)'}>{alt_s or '(empty)'}"
        elif chrom not in fasta:
            reason, detail = "unknown_chrom", str(chrom)
        elif max(len(ref), len(alt)) > FLANK:
            reason, detail = "allele_too_long", f"REF {len(ref)} bp, ALT {len(alt)} bp"
        else:
            start, end = flank_span(pos, ref, alt)
            length = fasta.length(chrom)
            # VS's get_valid_variants (helpers.py:32-33), plus a deletion's extra bases on the right.
            if not (pos - FLANK > 0 and pos + FLANK <= length and end <= length):
                reason, detail = "off_chrom_end", f"needs {chrom}:{start}-{end}, length {length}"
            else:
                flank = fasta.fetch(chrom, start, end)
                in_genome = flank[FLANK : FLANK + len(ref)]
                if in_genome != ref:
                    reason, detail = "ref_mismatch", f"REF {ref_s}, FASTA {in_genome.decode()}"
        if reason:
            dropped.append((vid, reason, detail))
        else:
            kept.append(Entry(vid, chrom, pos, ref, alt, row.get("stratum") or "", flank))
    return kept, dropped


def build_variants(kept: list[Entry]) -> dict[str, np.ndarray]:
    a1 = np.empty((len(kept), INPUT_LEN), dtype=np.uint8)
    a2 = np.empty((len(kept), INPUT_LEN), dtype=np.uint8)
    for i, e in enumerate(kept):
        s1, s2 = allele_windows(e.flank, e.ref, e.alt)
        a1[i], a2[i] = encode(s1), encode(s2)
    nonacgt = (a1 == 4).any(axis=1) | (a2 == 4).any(axis=1)
    return {"a1": a1, "a2": a2, "ids": np.array([e.vid for e in kept], dtype=str), "nonacgt": nonacgt}


def dinucleotide_counts(codes: np.ndarray) -> np.ndarray:
    """How often each ordered pair of adjacent codes occurs (25 counts)."""
    c = np.asarray(codes, dtype=np.int64)
    return np.bincount(5 * c[:-1] + c[1:], minlength=25)


def tangermeme_shuffler() -> Shuffler:
    """tangermeme.ersatz.dinucleotide_shuffle on one sequence per call (see the module docstring). Every shuffle is
    checked to keep the sequence's dinucleotide counts and its first and last base, as the algorithm must."""
    import torch
    from tangermeme.ersatz import dinucleotide_shuffle

    def shuffle(codes: np.ndarray, seed: int) -> np.ndarray:
        x = torch.zeros((1, 4, len(codes)), dtype=torch.float32)
        x[0, torch.from_numpy(codes.astype(np.int64)), torch.arange(len(codes))] = 1.0
        out = dinucleotide_shuffle(x, n=1, random_state=int(seed))  # (1, n, 4, L)
        shuffled = out[0, 0].argmax(dim=0).numpy().astype(np.uint8)
        kept = shuffled[0] == codes[0] and shuffled[-1] == codes[-1]
        if not kept or not np.array_equal(dinucleotide_counts(shuffled), dinucleotide_counts(codes)):
            raise ValueError(f"tangermeme's shuffle with seed {seed} changed the dinucleotide counts or the end bases")
        return shuffled

    return shuffle


def build_null(
    kept: list[Entry], n: int, seed: int, strata: tuple[str, ...], shuffle: Shuffler
) -> tuple[dict[str, np.ndarray], dict]:
    """VS's null rows (module docstring): n rows of the ACGT-only pool, row r shuffled with seed + r."""
    in_strata = [e for e in kept if e.stratum in strata]
    pool = [e for e in in_strata if bool((encode(e.flank) < 4).all())]
    if not pool:
        raise ValueError(f"no row of strata {', '.join(strata)} has an ACGT-only window to shuffle")
    replace = not len(pool) > n  # VS helpers.py:294-304
    picks = np.random.default_rng(seed).choice(len(pool), size=n, replace=replace)
    a1 = np.empty((n, INPUT_LEN), dtype=np.uint8)
    a2 = np.empty((n, INPUT_LEN), dtype=np.uint8)
    seeds = seed + np.arange(n, dtype=np.int64)
    unchanged = 0
    for r, i in enumerate(picks):
        e = pool[i]
        codes = encode(e.flank)
        shuffled = np.asarray(shuffle(codes, int(seeds[r])), dtype=np.uint8)
        same_bases = shuffled.shape == codes.shape and np.array_equal(
            np.bincount(shuffled, minlength=5), np.bincount(codes, minlength=5)
        )
        if not same_bases:
            raise ValueError(f"the shuffle of {e.vid} (seed {seeds[r]}) did not keep the flank's base composition")
        unchanged += int(np.array_equal(shuffled, codes))
        s1, s2 = allele_windows(BASES[shuffled].tobytes(), e.ref, e.alt)
        a1[r], a2[r] = encode(s1), encode(s2)
    src_ids = np.array([pool[i].vid for i in picks], dtype=str)
    arrays = {
        "a1": a1,
        "a2": a2,
        "ids": np.array([f"{vid}|shuf{s}" for vid, s in zip(src_ids, seeds)], dtype=str),
        "nonacgt": (a1 == 4).any(axis=1) | (a2 == 4).any(axis=1),
        "src_ids": src_ids,
        "seeds": seeds,
    }
    stats = {
        "strata": list(strata),
        "rows_in_strata": len(in_strata),
        "excluded_not_acgt": len(in_strata) - len(pool),
        "pool": len(pool),
        "pool_by_stratum": dict(Counter(e.stratum for e in pool)),
        "drawn": n,
        "with_replacement": replace,
        "distinct_rows_drawn": len(set(picks.tolist())),
        "shuffles_returning_the_flank_unchanged": unchanged,
        "drawn_by_stratum": dict(Counter(pool[i].stratum for i in picks)),
        "seeds": f"{seed} + row, row = 0..{n - 1}",
    }
    return arrays, stats


def read_narrowpeak(path: Path) -> list[tuple[str, int, int, int]]:
    """(chrom, start, end, summit offset) per peak; the offset defaults to (end - start) // 2 (VS io.py:88-89)."""
    peaks = []
    with _open_text(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            f = line.rstrip("\n").split("\t")
            chrom, start, end = f[0], int(f[1]), int(f[2])
            summit = int(f[9]) if len(f) >= 10 and f[9] not in ("", ".") else -1
            peaks.append((chrom, start, end, summit if summit >= 0 else (end - start) // 2))
    return peaks


def build_peaks(
    peaks: list[tuple[str, int, int, int]], fasta: IndexedFasta, n: int, seed: int, vs_window: bool
) -> tuple[dict[str, np.ndarray], dict, list[tuple[str, str, str]]]:
    seen: set[tuple[str, int, int, int]] = set()
    valid, dropped = [], []
    for p in peaks:
        chrom, start, end, offset = p
        pid = f"{chrom}:{start}-{end}:{offset}"
        summit = start + offset
        if p in seen:  # VS variant_scoring.py:106-107
            dropped.append((pid, "duplicate", ""))
        elif chrom not in fasta:
            dropped.append((pid, "unknown_chrom", chrom))
        elif not (summit - FLANK > 0 and summit + FLANK <= fasta.length(chrom)):  # VS helpers.py:16-26
            dropped.append((pid, "off_chrom_end", f"summit {summit}, length {fasta.length(chrom)}"))
        else:
            valid.append((pid, chrom, summit))
        seen.add(p)
    picks = np.arange(len(valid))
    if len(valid) > n:  # VS variant_scoring.py:120-123
        picks = np.sort(np.random.default_rng(seed).choice(len(valid), size=n, replace=False))
    a1 = np.empty((len(picks), INPUT_LEN), dtype=np.uint8)
    shift = 1 if vs_window else 0
    for r, i in enumerate(picks):
        _, chrom, summit = valid[i]
        a1[r] = encode(fasta.fetch(chrom, summit - FLANK - shift, summit + FLANK - shift))
    arrays = {
        "a1": a1,
        "ids": np.array([valid[i][0] for i in picks], dtype=str),
        "nonacgt": (a1 == 4).any(axis=1),
        "chrom": np.array([valid[i][1] for i in picks], dtype=str),
        "summit": np.array([valid[i][2] for i in picks], dtype=np.int64),
    }
    stats = {
        "valid_peaks": len(valid),
        "drawn": len(picks),
        "window": "VS PeakGenerator: [summit - 1058, summit + 1056)"
        if vs_window
        else "training: [summit - 1057, summit + 1057)",
    }
    return arrays, stats, dropped


def now_et() -> str:
    """Now, in US Eastern time (the cluster's clock runs 6 h ahead of it)."""
    try:
        return datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S %Z")
    except ZoneInfoNotFoundError:  # no time zone database
        return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


def code_version() -> str | None:
    """The repository commit this script ran from, when git can tell."""
    try:
        out = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=30)  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def write_outputs(out: Path, arrays: dict[str, np.ndarray], meta: dict, dropped: list[tuple[str, str, str]]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".dropped.tsv"), "w") as fh:
        fh.write("id\treason\tdetail\n")
        fh.writelines(f"{i}\t{r}\t{d}\n" for i, r, d in dropped)
    meta["arrays"] = {k: [list(v.shape), str(v.dtype)] for k, v in arrays.items()}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    partial = out.with_name(out.name + ".partial")
    with open(partial, "wb") as fh:
        np.savez(fh, **arrays)
    os.replace(partial, out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--null", action="store_true", help="build the shuffled null from --list")
    mode.add_argument("--peaks", type=Path, metavar="NARROWPEAK", help="build summit-centred windows of these peaks")
    ap.add_argument("--list", type=Path, help="variant list TSV (variant and --null modes)")
    ap.add_argument("--fasta", type=Path, required=True, help="genome FASTA, with its .fai")
    ap.add_argument("--out", type=Path, required=True, help="output .npz; its .json and .dropped.tsv go beside it")
    ap.add_argument("--n", type=int, default=10000, help="rows to draw for --null and --peaks (default 10000)")
    ap.add_argument("--seed", type=int, default=1234,
                    help="seed of the draw; --null also shuffles row r with seed + r (default 1234)")  # fmt: skip
    ap.add_argument("--null-strata", default=",".join(NULL_STRATA),
                    help=f"strata the null is drawn from (default {','.join(NULL_STRATA)})")  # fmt: skip
    ap.add_argument("--vs-peak-window", action="store_true",
                    help="--peaks: VS's window, 1 bp left of the training window (for a check against VS)")  # fmt: skip
    ap.add_argument("--drop-invalid", action="store_true",
                    help="drop and count rows whose REF does not match the FASTA, whose alleles are not ACGT or "
                         "whose chromosome the FASTA lacks, instead of refusing the list")  # fmt: skip
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%F %T")  # fmt: skip

    if args.out.suffix != ".npz":
        ap.error("--out must end in .npz")
    if args.peaks is None and args.list is None:
        ap.error("--list is needed unless --peaks is given")
    if args.n < 1:
        ap.error("--n must be at least 1")
    started = time.time()
    meta: dict = {
        "mode": "null" if args.null else "peaks" if args.peaks else "variants",
        "created": now_et(),
        "argv": sys.argv[1:] if argv is None else list(argv),
        "code_commit": code_version(),
        "follows": f"{VS_COMMIT}: src/generators/variant_generator.py:28-76, src/utils/helpers.py:16-42, 293-317",
        "input_len": INPUT_LEN,
        "allele_index": FLANK,
        "codes": {"A": 0, "C": 1, "G": 2, "T": 3, "other": 4},
        "fasta": str(args.fasta.resolve()),
        "fasta_size": args.fasta.stat().st_size,
    }
    with IndexedFasta(args.fasta) as fasta:
        built = build(args, fasta, meta)
    if built is None:
        return 1
    arrays, dropped = built
    meta["dropped"] = dict(Counter(d[1] for d in dropped))
    meta["n_nonacgt"] = int(arrays["nonacgt"].sum())
    meta["fasta_md5"] = file_md5(args.fasta)
    meta["seconds"] = round(time.time() - started, 1)
    write_outputs(args.out, arrays, meta, dropped)
    logger.info("%s: %d rows from %d (dropped: %s; %d with a non-ACGT base) in %.0f s", args.out, meta["n_kept"],
                meta["n_input"], ", ".join(f"{k} {v}" for k, v in sorted(meta["dropped"].items())) or "none",
                meta["n_nonacgt"], meta["seconds"])  # fmt: skip
    return 0


def build(args, fasta: IndexedFasta, meta: dict) -> tuple[dict[str, np.ndarray], list[tuple[str, str, str]]] | None:
    """The arrays and the dropped rows of the mode asked for, filling meta in; None when the list is refused."""
    if args.peaks:
        peaks = read_narrowpeak(args.peaks)
        arrays, stats, dropped = build_peaks(peaks, fasta, args.n, args.seed, args.vs_peak_window)
        meta.update(peaks=str(args.peaks.resolve()), peaks_md5=file_md5(args.peaks), n_input=len(peaks),
                    n_kept=len(arrays["ids"]), seed=args.seed, peak_draw=stats)  # fmt: skip
        return arrays, dropped
    rows = read_list(args.list, REQUIRED + (("stratum",) if args.null else ()))
    kept, dropped = prepare(rows, fasta)
    invalid = [d for d in dropped if d[1] in INVALID]
    if invalid and not args.drop_invalid:
        counts = Counter(d[1] for d in invalid)
        logger.error("%d row(s) do not fit %s (%s); nothing written. First ones:\n  %s\nFix the list, or pass "
                     "--drop-invalid to drop and count them.", len(invalid), args.fasta,
                     ", ".join(f"{k} {v}" for k, v in sorted(counts.items())),
                     "\n  ".join("\t".join(d) for d in invalid[:10]))  # fmt: skip
        return None
    meta.update(list=str(args.list.resolve()), list_md5=file_md5(args.list), n_input=len(rows))
    if args.null:
        strata = tuple(s for s in args.null_strata.split(",") if s)
        arrays, stats = build_null(kept, args.n, args.seed, strata, tangermeme_shuffler())
        meta.update(n_kept=len(arrays["ids"]), seed=args.seed, null=stats,
                    shuffler="tangermeme.ersatz.dinucleotide_shuffle, one row per call, random_state = seed + row; "
                             "each shuffle checked to keep the dinucleotide counts and the first and last base",
                    null_alleles="a1 and a2: the shuffled flank with each allele written back at 1057, as VS")  # fmt: skip
    else:
        arrays = build_variants(kept)
        meta["n_kept"] = len(arrays["ids"])
    return arrays, dropped


if __name__ == "__main__":
    sys.exit(main())
