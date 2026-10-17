#!/usr/bin/env python3
"""The 1-bp alignment gate between a per-base Tn5 bias track and an ATAC dataset's Tn5 cut sites, strand by strand.

A bias track is a control for a model only if its base i is the bias of a cut at base i. The h5 (one float32 dataset
per contig, v = 10^(2x - 1) - 0.01 of the bias CNN's output x) has its own frame and the dataset's cut sites have
theirs; a 1-bp slip between the two, on one strand or on both, does not show in a model's metrics and leaves a Tn5
signal for the model to learn. This measures it. With b = log10(v + 0.01) = 2x - 1, over --n-windows windows sampled
(--seed) from the negatives (background regions, where the cuts are mostly Tn5's own preference) on --chroms:

    m_s(d) = sum_i c_s[i] * b[i + d] / sum_i c_s[i] - mean_i b[i]      for d in [-max_offset, max_offset]
    r_s(d) = Pearson(c_s[i], b[i + d])                                 pooled over the window bases

for s = plus (fragment starts), minus (fragment ends) and both (their sum). i runs over the union of the sampled
windows' bases, each base once, and mean_i b[i] is over the same bases unshifted. The cuts are the pipeline's
(igvf_tf_chrombpnet lib/python/utils/pileup.py, cut_positions): every fragment cuts at start + plus_delta on the plus
strand and at end + minus_delta - 1 on the minus strand. A cut off its chromosome is dropped and counted, as figwig
(which piled up the signal bigWig) drops it. That these are the cuts the model was trained on is checked: plus +
minus must equal the signal bigWig (--signal, NaN read as 0) at >= AGREEMENT_MIN of the window bases. The shift that
the signal's prepared_bigwig.json (beside it, when there is one) says the fragments carry is recorded, with whether
the deltas given are the ones it implies (ATAC: 4 - plus_shift, -4 - minus_shift); a mismatch is a warning only.

The gate (gate.json) is CLEAN iff argmax_d m_plus = argmax_d m_minus = argmax_d m_both =: d*, |d*| <= 1,
m_both(d*) > 0 and the agreement holds. Then offset = d*, and the tracks are written as t[i] = f(h5[i + d*])
(tn5_tracks.py). Otherwise it is AMBIGUOUS, offset null, with every clause that failed in the reason. The case this
is for is a split, one strand at 0 and the other at +-1: a +4/-5 frame against a +4/-4 one moves the minus strand
only, and no single offset of the track fixes that.

The fragments are read once: decompressed by `bgzip -dc -@<threads>` (--bgzip, else bgzip on PATH or beside the
interpreter; without one `gzip -dc`, a single thread and the slow part, else Python's gzip; align.json records which)
and filtered to --chroms by mawk (or awk), which passes on a chromosome code, the start and the end of each row and
drops the '#' header; the cuts are kept per chromosome and strand as int32 and counted with np.bincount, one
chromosome at a time. Every row is one fragment, whatever its fifth (count) column says, as figwig counts it.
Outputs, each written to <name>.partial and renamed:

    --out        align.json: inputs (sizes, mtimes), parameters, the stream's counts and commands, the windows, m, r,
                 argmax and margin per strand, the same per chromosome, the agreement, the shift check, the gate,
                 code and versions, timings, the time in ET
    --gate-out   gate.json: {"status": "clean"|"ambiguous", "offset": int|null, "reason": str, "argmax": {plus,
                 minus, both}, "m": {strand: {d: m}}, "agreement", "rule", "align"}

    python tn5_align.py --h5 <bias.h5> --fragments <fragments.tsv.gz> --negatives <negatives.bed>
        --signal <data_unstranded.bw> --plus-delta 0 --minus-delta 1 [--chroms chr2 chr4 ...] [--n-windows 50000]
        [--max-offset 5] [--seed 1234] [--threads 8] [--bgzip <bgzip>] --out <align.json> --gate-out <gate.json>
    python tn5_align.py --selftest

Exit 0 whenever a gate was computed, clean or ambiguous; 1 on an error (no gate.json is written then). Run it in the
cherimoya environment (numpy, h5py, pyBigWig), on CPU, inside a box: it holds the cut positions of the chosen
chromosomes (8 bytes per fragment) and one chromosome's h5 dataset and counts at a time, a few GB.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import re
import resource
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
logger = logging.getLogger("tn5_align")

FORMAT = 1
DEFAULT_CHROMS = ("chr2", "chr4", "chr5", "chr7", "chr9", "chr11", "chr12")
STRANDS = ("plus", "minus", "both")
PSEUDO = 0.01
"""scPrinter's: v = 10^((x - 0.5) * 2) - PSEUDO, so log10(v + PSEUDO) = 2x - 1."""
GATE_MAX_OFFSET = 1
AGREEMENT_MIN = 0.999
BLOCK_BYTES = 1 << 26
"""Bytes of the filtered stream parsed at a time (about 3 million fragments)."""
CHROM_NAME = re.compile(r"^[A-Za-z0-9_.\-]+$")
"""Names that pass into the awk program as they are."""
AWK_PROGRAM = r"""
BEGIN { n = split(chroms, names, ","); for (i = 1; i <= n; i++) code[names[i]] = i - 1 }
/^#/ { header++; next }
($1 in code) { print code[$1], $2, $3 }
END { print NR, header + 0 > countfile }
"""
PY_GUNZIP = "import gzip, shutil, sys; shutil.copyfileobj(gzip.open(sys.argv[1], 'rb'), sys.stdout.buffer, 1 << 22)"
GATE_RULE = (
    f"clean iff argmax_d m_plus == argmax_d m_minus == argmax_d m_both =: d*, |d*| <= {GATE_MAX_OFFSET}, "
    f"m_both(d*) > 0 and plus + minus reproduce the signal at >= {AGREEMENT_MIN} of the window bases; offset = d*"
)
ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------------------------------------------------
# files


def partial(path: Path) -> Path:
    return Path(f"{path}.partial")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = partial(path)
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


def stat_file(path: Path) -> dict:
    st = os.stat(path)
    return {
        "path": str(path),
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "mtime_et": datetime.fromtimestamp(st.st_mtime, ET).isoformat(timespec="seconds"),
    }


def finite(x: float) -> float | None:
    """x for JSON: None when it is not finite."""
    return float(x) if np.isfinite(x) else None


def h5_lengths(h5_path: Path, chroms: list[str]) -> dict[str, int]:
    with h5py.File(h5_path, "r") as f:
        missing = [c for c in chroms if c not in f]
        if missing:
            raise ValueError(f"{h5_path} has no dataset {', '.join(missing)}")
        return {c: int(f[c].shape[0]) for c in chroms}


def shift_check(signal: Path, plus_delta: int, minus_delta: int) -> dict:
    """The shift prepared_bigwig.json (beside the signal) records, and whether the deltas are the ones it implies."""
    path = signal.parent / "prepared_bigwig.json"
    try:
        rec = json.loads(path.read_text())
    except FileNotFoundError:
        return {"path": str(path), "found": False}
    except (OSError, ValueError) as exc:
        return {"path": str(path), "found": False, "error": str(exc)}
    out = {"path": str(path), "found": True}
    out.update({k: rec.get(k) for k in ("assay", "plus_shift", "minus_shift", "signal_path", "pileup")})
    plus, minus, assay = rec.get("plus_shift"), rec.get("minus_shift"), str(rec.get("assay") or "ATAC").upper()
    if isinstance(plus, int) and isinstance(minus, int) and assay in ("ATAC", "DNASE"):
        expected = (4 - plus, -4 - minus) if assay == "ATAC" else (-plus, 1 - minus)
        out["expected_deltas"] = list(expected)
        out["match"] = expected == (plus_delta, minus_delta)
        if not out["match"]:
            logger.warning("%s records the shift %+d/%+d (%s), which needs the deltas %s, not (%d, %d)", path, plus,
                           minus, assay, expected, plus_delta, minus_delta)  # fmt: skip
    return out


# ---------------------------------------------------------------------------------------------------------------------
# the cuts


def decompressor(fragments: Path, threads: int, bgzip: str | None) -> list[str]:
    """The command that writes the fragments file to stdout, without its path: bgzip as given, else on PATH, else
    beside this interpreter (an environment with htslib has one); else gzip -dc (one thread), else Python's gzip."""
    with open(fragments, "rb") as fh:
        if fh.read(2) != b"\x1f\x8b":
            return ["cat"]
    beside = Path(sys.executable).parent / "bgzip"
    bgzip = bgzip or shutil.which("bgzip") or (str(beside) if os.access(beside, os.X_OK) else None)
    if bgzip:
        return [bgzip, "-dc", "-@", str(max(1, threads))]
    gz = shutil.which("gzip")
    if gz:
        return [gz, "-dc"]
    return [sys.executable, "-I", "-c", PY_GUNZIP]


def awk_binary() -> str:
    for name in ("mawk", "gawk", "awk"):
        found = shutil.which(name)
        if found:
            return found
    raise RuntimeError("no mawk, gawk or awk on PATH to filter the fragments with")


def parse_block(data: bytes, n_chroms: int) -> np.ndarray:
    """Lines of '<code> <start> <end>' as an (n, 3) int64 array; refuses anything else."""
    n_lines = data.count(b"\n")
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)  # numpy warns, not raises, at a token it cannot read
        arr = np.fromstring(data.decode("ascii"), dtype=np.int64, sep=" ")
    if arr.size != 3 * n_lines:
        raise ValueError(f"the filtered stream has {arr.size} numbers on {n_lines} lines, not 3 per line")
    arr = arr.reshape(n_lines, 3)
    if n_lines and (arr[:, 0].min() < 0 or arr[:, 0].max() >= n_chroms):
        raise ValueError("the filtered stream has a chromosome code out of range")
    return arr


def stream_cuts(fragments: Path, chroms: list[str], lengths: dict[str, int], plus_delta: int, minus_delta: int,
                decompress: list[str], block_bytes: int = BLOCK_BYTES) -> tuple[dict, dict]:  # fmt: skip
    """Every fragment's two cuts on the chosen chromosomes: ({chrom: {"plus": int32, "minus": int32}}, record)."""
    for chrom in chroms:
        if not CHROM_NAME.match(chrom):
            raise ValueError(f"chromosome name {chrom!r} has a character the awk filter would misread")
    began = time.perf_counter()
    awk = awk_binary()
    parts: dict[str, dict[str, list]] = {c: {"plus": [], "minus": []} for c in chroms}
    rows = np.zeros(len(chroms), dtype=np.int64)
    dropped = {f"{s}_{w}": 0 for s in ("plus", "minus") for w in ("before_start", "past_end")}
    length_of = np.array([lengths[c] for c in chroms], dtype=np.int64)

    def take(block: bytes) -> None:
        arr = parse_block(block, len(chroms))
        codes = arr[:, 0]
        cuts = {"plus": arr[:, 1] + plus_delta, "minus": arr[:, 2] + (minus_delta - 1)}
        # A sorted fragments file gives blocks of one chromosome but at a chromosome's end; spare those the sort.
        one = codes.size > 0 and bool(codes[0] == codes[-1]) and bool((codes == codes[0]).all())
        present = [int(codes[0])] if one else np.unique(codes).tolist()
        for k in present:
            sel = None if len(present) == 1 else codes == k
            rows[k] += codes.size if sel is None else int(sel.sum())
            for strand, pos in cuts.items():
                pos = pos if sel is None else pos[sel]
                before, past = pos < 0, pos >= length_of[k]
                n_before, n_past = int(before.sum()), int(past.sum())
                if n_before or n_past:
                    dropped[f"{strand}_before_start"] += n_before
                    dropped[f"{strand}_past_end"] += n_past
                    pos = pos[~(before | past)]
                parts[chroms[k]][strand].append(pos.astype(np.int32))

    with tempfile.TemporaryDirectory(prefix="tn5_align.") as tmp:
        countfile = Path(tmp) / "lines"
        dec_cmd = [*decompress, str(fragments)]
        awk_cmd = [awk, "-v", f"chroms={','.join(chroms)}", "-v", f"countfile={countfile}", AWK_PROGRAM]
        logger.info("streaming %s through %s | %s", fragments, " ".join(decompress), awk)
        dec = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE)
        try:
            # LC_ALL=C: bytes, not characters, which gawk otherwise decodes on every line.
            flt = subprocess.Popen(awk_cmd, stdin=dec.stdout, stdout=subprocess.PIPE, env={**os.environ, "LC_ALL": "C"})
        except BaseException:
            dec.kill()
            dec.wait()
            raise
        dec.stdout.close()  # awk holds the pipe now, so the decompressor stops if awk does
        try:
            rest = b""
            while True:
                chunk = flt.stdout.read(block_bytes)
                if not chunk:
                    break
                data = rest + chunk
                cut = data.rfind(b"\n") + 1
                rest = data[cut:]
                if cut:
                    take(data[:cut])
            if rest.strip():
                take(rest + b"\n")
        except BaseException:
            for proc in (flt, dec):
                proc.kill()
                proc.wait()
            raise
        flt.stdout.close()
        rc_flt, rc_dec = flt.wait(), dec.wait()
        if rc_dec != 0 or rc_flt != 0:
            raise RuntimeError(f"reading {fragments} failed: {' '.join(dec_cmd)} exited {rc_dec}, awk {rc_flt}")
        try:
            n_lines, n_header = (int(x) for x in countfile.read_text().split())
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"awk left no line count: {exc}") from exc

    cuts = {}
    for chrom in chroms:
        cuts[chrom] = {
            s: np.concatenate(p) if p else np.empty(0, dtype=np.int32) for s, p in parts.pop(chrom).items()
        }  # fmt: skip
    record = {
        "decompress": dec_cmd,
        "filter": [awk, "-v", f"chroms={','.join(chroms)}", "<program>"],
        "lines": n_lines,
        "header_lines": n_header,
        "rows_kept": int(rows.sum()),
        "rows_per_chrom": {c: int(rows[k]) for k, c in enumerate(chroms)},
        "cuts_dropped": dropped,
        "seconds": time.perf_counter() - began,
    }
    if dropped["plus_before_start"] or dropped["minus_before_start"]:
        logger.warning("cuts before a chromosome's start, dropped: %s (pileup.py would refuse them)", dropped)
    return cuts, record


# ---------------------------------------------------------------------------------------------------------------------
# the windows


def sample_windows(negatives: Path, chroms: list[str], lengths: dict[str, int], n: int, max_offset: int,
                   seed: int) -> tuple[dict[str, list[tuple[int, int]]], dict]:  # fmt: skip
    """n negatives on the chromosomes, at least max_offset from either end, drawn without replacement and merged
    into disjoint intervals per chromosome (sorted)."""
    order = {c: i for i, c in enumerate(chroms)}
    rows, off_chroms, too_close = [], 0, 0
    for n_line, line in enumerate(Path(negatives).read_text().splitlines(), 1):
        fields = line.split()
        if not fields or fields[0].startswith(("#", "track", "browser")):
            continue
        if fields[0] not in order:
            off_chroms += 1
            continue
        try:
            start, end = int(fields[1]), int(fields[2])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"{negatives}, line {n_line}: not a BED line: {line!r}") from exc
        if start - max_offset < 0 or end + max_offset > lengths[fields[0]] or end <= start:
            too_close += 1
            continue
        rows.append((order[fields[0]], start, end))
    if not rows:
        raise ValueError(f"{negatives} has no usable window on {', '.join(chroms)}")
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(len(rows), size=min(n, len(rows)), replace=False))
    chosen = sorted(rows[i] for i in pick.tolist())
    merged: dict[str, list[list[int]]] = {c: [] for c in chroms}
    for k, start, end in chosen:
        runs = merged[chroms[k]]
        if runs and start <= runs[-1][1]:
            runs[-1][1] = max(runs[-1][1], end)
        else:
            runs.append([start, end])
    windows = {c: [(s, e) for s, e in runs] for c, runs in merged.items()}
    record = {
        "on_chroms": len(rows) + too_close,
        "off_chroms": off_chroms,
        "too_close_to_an_end": too_close,
        "requested": n,
        "sampled": len(chosen),
        "per_chrom": {c: sum(1 for k, _, _ in chosen if k == order[c]) for c in chroms},
        "bases": int(sum(e - s for runs in windows.values() for s, e in runs)),
        "seed": seed,
    }
    return windows, record


# ---------------------------------------------------------------------------------------------------------------------
# the statistics


class Sums:
    """The sums m and r are made of, over window bases, for each strand and offset."""

    def __init__(self, offsets: list[int]):
        k = len(offsets)
        self.offsets = offsets
        self.n = 0
        self.b0 = 0.0
        self.b = np.zeros(k)
        self.bb = np.zeros(k)
        self.c = {s: 0.0 for s in STRANDS}
        self.cc = {s: 0.0 for s in STRANDS}
        self.cb = {s: np.zeros(k) for s in STRANDS}

    def add(self, other: Sums) -> None:
        self.n += other.n
        self.b0 += other.b0
        self.b += other.b
        self.bb += other.bb
        for s in STRANDS:
            self.c[s] += other.c[s]
            self.cc[s] += other.cc[s]
            self.cb[s] += other.cb[s]

    def m(self) -> dict[str, np.ndarray]:
        mean_b = self.b0 / self.n if self.n else np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            return {s: self.cb[s] / self.c[s] - mean_b for s in STRANDS}

    def r(self) -> dict[str, np.ndarray]:
        out = {}
        n = float(self.n)
        with np.errstate(divide="ignore", invalid="ignore"):
            for s in STRANDS:
                cov = self.cb[s] / n - (self.c[s] / n) * (self.b / n)
                var_c = self.cc[s] / n - (self.c[s] / n) ** 2
                var_b = self.bb / n - (self.b / n) ** 2
                out[s] = cov / np.sqrt(var_c * var_b)
        return out


def argmax_of(values: np.ndarray, offsets: list[int]) -> tuple[int | None, float | None]:
    """(offset of the maximum, its margin over the next best), or (None, None) when a value is not finite."""
    if not np.isfinite(values).all():
        return None, None
    i = int(np.argmax(values))
    others = np.delete(values, i)
    return offsets[i], float(values[i] - others.max()) if others.size else None


def chrom_sums(h5_path: Path, chrom: str, length: int, windows: list[tuple[int, int]], cuts: dict[str, np.ndarray],
               signal_bw, offsets: list[int]) -> tuple[Sums, dict]:  # fmt: skip
    """One chromosome's sums, and how well its cut counts reproduce the signal bigWig on the window bases."""
    pos = np.concatenate([np.arange(s, e, dtype=np.int64) for s, e in windows])
    counts = {strand: np.bincount(cuts[strand], minlength=length)[pos].astype(np.float64) for strand in cuts}
    counts["both"] = counts["plus"] + counts["minus"]
    lo, hi = pos[0] + min(offsets), pos[-1] + 1 + max(offsets)
    with h5py.File(h5_path, "r") as f:
        v = f[chrom][lo:hi]
    sums = Sums(offsets)
    sums.n = int(pos.size)
    for s in STRANDS:
        sums.c[s] = float(counts[s].sum())
        sums.cc[s] = float(np.dot(counts[s], counts[s]))
    for j, d in enumerate(offsets):
        b = np.log10(v[pos - lo + d].astype(np.float64) + PSEUDO)
        if not np.isfinite(b).all():
            raise ValueError(f"log10(v + 0.01) is not finite at {int((~np.isfinite(b)).sum())} bases of {chrom}")
        sums.b[j], sums.bb[j] = float(b.sum()), float(np.dot(b, b))
        if d == 0:
            sums.b0 = sums.b[j]
        for s in STRANDS:
            sums.cb[s][j] = float(np.dot(counts[s], b))
    sig = np.concatenate([bw_values(signal_bw, chrom, s, e) for s, e in windows])
    sig = np.nan_to_num(sig, nan=0.0)
    diff = np.abs(sig - counts["both"])
    agree = {
        "bases": int(pos.size),
        "equal": int((diff == 0).sum()),
        "max_abs_diff": float(diff.max(initial=0.0)),
        "signal_sum": float(sig.sum()),
        "counts_sum": float(counts["both"].sum()),
    }
    return sums, agree


def bw_values(bw, chrom: str, start: int, end: int) -> np.ndarray:
    import pyBigWig

    if getattr(pyBigWig, "numpy", 0):
        return np.asarray(bw.values(chrom, start, end, numpy=True), dtype=np.float64)
    return np.array(bw.values(chrom, start, end), dtype=np.float64)


def table(values: dict[str, np.ndarray], offsets: list[int]) -> dict[str, dict[str, float | None]]:
    return {s: {str(d): finite(x) for d, x in zip(offsets, values[s])} for s in STRANDS}


def decide(m: dict[str, np.ndarray], agreement: float, offsets: list[int]) -> dict:
    """The gate from m and the agreement (GATE_RULE)."""
    argmax = {s: argmax_of(m[s], offsets)[0] for s in STRANDS}
    failures = []
    d_star = None
    if any(a is None for a in argmax.values()):
        failures.append("m is not finite on " + ", ".join(s for s in STRANDS if argmax[s] is None) + " (no cuts?)")
    elif len(set(argmax.values())) != 1:
        failures.append(
            "argmax_d m differs between the strands: " + ", ".join(f"{s} {argmax[s]:+d}" for s in STRANDS)
        )  # fmt: skip
    else:
        d_star = argmax["both"]
        if abs(d_star) > GATE_MAX_OFFSET:
            failures.append(f"argmax_d m = {d_star:+d} on every strand, beyond +-{GATE_MAX_OFFSET}")
        m_star = float(m["both"][offsets.index(d_star)])
        if not m_star > 0:
            failures.append(f"m_both({d_star:+d}) = {m_star:.6g} is not positive")
    if not agreement >= AGREEMENT_MIN:
        failures.append(f"plus + minus reproduce the signal at {agreement:.6f} of the window bases, < {AGREEMENT_MIN}")
    clean = not failures
    if clean:
        reason = (
            f"argmax_d m = {d_star:+d} on plus, minus and both; m_both({d_star:+d}) = "
            f"{float(m['both'][offsets.index(d_star)]):.6g} > 0; plus + minus reproduce the signal at {agreement:.6f}"
        )
    else:
        reason = "; ".join(failures)
    return {
        "status": "clean" if clean else "ambiguous",
        "offset": d_star if clean else None,
        "reason": reason,
        "argmax": argmax,
        "m": table(m, offsets),
        "agreement": agreement,
        "rule": GATE_RULE,
    }


# ---------------------------------------------------------------------------------------------------------------------
# a run


def code_identity() -> dict:
    me = Path(__file__).resolve()

    def git(*args: str) -> str | None:
        try:
            run = subprocess.run(["git", "-C", str(HERE), *args], capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return None
        return run.stdout.strip() if run.returncode == 0 else None

    dirty = git("status", "--porcelain", "--", me.name)
    return {
        "path": str(me),
        "sha256": hashlib.sha256(me.read_bytes()).hexdigest(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": None if dirty is None else bool(dirty),
    }


def run(h5_path: Path, fragments: Path, negatives: Path, signal: Path, plus_delta: int, minus_delta: int, *,
        chroms: list[str] | None = None, n_windows: int = 50000, max_offset: int = 5, seed: int = 1234,
        threads: int = 8, bgzip: str | None = None, decompress: list[str] | None = None) -> dict:  # fmt: skip
    """Measure m and r and decide the gate. Returns the align record, whose "gate" is gate.json."""
    import pyBigWig

    began = time.perf_counter()
    h5_path, fragments = Path(os.path.abspath(h5_path)), Path(os.path.abspath(fragments))
    negatives, signal = Path(os.path.abspath(negatives)), Path(os.path.abspath(signal))
    chroms = list(chroms or DEFAULT_CHROMS)
    if len(set(chroms)) != len(chroms):
        raise ValueError(f"--chroms repeats a chromosome: {chroms}")
    offsets = list(range(-max_offset, max_offset + 1))
    sources = {"h5": h5_path, "fragments": fragments, "negatives": negatives, "signal": signal}
    inputs = {role: stat_file(path) for role, path in sources.items()}
    lengths = h5_lengths(h5_path, chroms)
    bw = pyBigWig.open(str(signal))
    if bw is None:
        raise ValueError(f"cannot open the signal bigWig {signal}")
    try:
        bw_chroms = bw.chroms()
        wrong = {c: (lengths[c], bw_chroms.get(c)) for c in chroms if bw_chroms.get(c) != lengths[c]}
        if wrong:
            raise ValueError(f"the h5 and {signal} disagree on (or lack) chromosomes: {wrong}")
        shift = shift_check(signal, plus_delta, minus_delta)
        windows, window_record = sample_windows(negatives, chroms, lengths, n_windows, max_offset, seed)
        logger.info("%d windows (%d bases) on %s", window_record["sampled"], window_record["bases"], ", ".join(chroms))

        decompress = decompress or decompressor(fragments, threads, bgzip)
        cuts, stream = stream_cuts(fragments, chroms, lengths, plus_delta, minus_delta, decompress)
        logger.info("%d fragments on %d chromosomes, of %d lines (%.1f s)", stream["rows_kept"], len(chroms),
                    stream["lines"], stream["seconds"])  # fmt: skip

        t = time.perf_counter()
        total = Sums(offsets)
        per_chrom, agree_total = {}, {"bases": 0, "equal": 0, "max_abs_diff": 0.0, "signal_sum": 0.0, "counts_sum": 0.0}
        for chrom in chroms:
            chrom_cuts = cuts.pop(chrom)
            n_cuts = {s: int(a.size) for s, a in chrom_cuts.items()}
            if not windows[chrom]:
                per_chrom[chrom] = {"windows": 0, "cuts": n_cuts}
                continue
            sums, agree = chrom_sums(h5_path, chrom, lengths[chrom], windows[chrom], chrom_cuts, bw, offsets)
            del chrom_cuts
            total.add(sums)
            for k in ("bases", "equal", "signal_sum", "counts_sum"):
                agree_total[k] += agree[k]
            agree_total["max_abs_diff"] = max(agree_total["max_abs_diff"], agree["max_abs_diff"])
            m_chrom = sums.m()
            per_chrom[chrom] = {
                "windows": window_record["per_chrom"][chrom],
                "bases": sums.n,
                "cuts": n_cuts,
                "cuts_in_windows": {s: sums.c[s] for s in ("plus", "minus")},
                "argmax": {s: argmax_of(m_chrom[s], offsets)[0] for s in STRANDS},
                "m_both": {str(d): finite(x) for d, x in zip(offsets, m_chrom["both"])},
                "agreement": agree["equal"] / agree["bases"] if agree["bases"] else None,
                "signal": agree,
            }
            amax = per_chrom[chrom]["argmax"]
            logger.info("%s: argmax_d m plus %s, minus %s, both %s; agreement %.6f", chrom, amax["plus"],
                        amax["minus"], amax["both"], per_chrom[chrom]["agreement"] or 0.0)  # fmt: skip
        stats_s = time.perf_counter() - t
    finally:
        bw.close()

    m, r = total.m(), total.r()
    agreement = agree_total["equal"] / agree_total["bases"] if agree_total["bases"] else 0.0
    gate = decide(m, agreement, offsets)
    now = datetime.now().astimezone()
    gate["written_et"] = now.astimezone(ET).isoformat(timespec="seconds")
    margins = {s: argmax_of(m[s], offsets)[1] for s in STRANDS}
    return {
        "format": FORMAT,
        "tool": "tn5_align.py",
        "written": now.isoformat(timespec="seconds"),
        "written_et": now.astimezone(ET).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "argv": list(sys.argv),
        "code": code_identity(),
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "h5py": h5py.__version__,
            "pyBigWig": getattr(pyBigWig, "__version__", None),
        },
        "inputs": inputs,
        "parameters": {
            "plus_delta": plus_delta,
            "minus_delta": minus_delta,
            "cut_rule": "plus = start + plus_delta; minus = end + minus_delta - 1 (pileup.py cut_positions)",
            "chroms": chroms,
            "n_windows": n_windows,
            "max_offset": max_offset,
            "seed": seed,
            "threads": threads,
            "b": "log10(v + 0.01) = 2x - 1 of the h5's v",
            "m": "m_s(d) = sum_i c_s[i] b[i + d] / sum_i c_s[i] - mean_i b[i], i over the union of the window bases",
            "r": "Pearson of c_s[i] and b[i + d], pooled over the window bases",
        },
        "shift_check": shift,
        "stream": stream,
        "windows": window_record,
        "offsets": offsets,
        "n_bases": total.n,
        "cuts_in_windows": {s: total.c[s] for s in STRANDS},
        "mean_b": finite(total.b0 / total.n) if total.n else None,
        "m": table(m, offsets),
        "r": table(r, offsets),
        "argmax": gate["argmax"],
        "argmax_r": {s: argmax_of(r[s], offsets)[0] for s in STRANDS},
        "margin": margins,
        "signal_check": {
            **agree_total,
            "agreement": agreement,
            "min": AGREEMENT_MIN,
            "pass": agreement >= AGREEMENT_MIN,
        },
        "per_chrom": per_chrom,
        "gate": gate,
        "timings_s": {"stream": stream["seconds"], "stats": stats_s, "total": time.perf_counter() - began},
        "rss_max_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }


def write_outputs(record: dict, out: Path, gate_out: Path) -> None:
    gate = {**record["gate"], "align": str(Path(os.path.abspath(out)))}
    write_json(Path(os.path.abspath(out)), record)
    write_json(Path(os.path.abspath(gate_out)), gate)


def summary(record: dict, out: Path) -> str:
    gate = record["gate"]
    a = gate["argmax"]
    return (
        f"tn5_align: gate {gate['status']}, offset {gate['offset']} (argmax_d m: plus {a['plus']}, minus "
        f"{a['minus']}, both {a['both']}; agreement {gate['agreement']:.6f}); {record['windows']['sampled']} "
        f"windows, {record['n_bases']} bases, {record['stream']['rows_kept']} fragments; {out}"
        + ("" if gate["status"] == "clean" else f"; {gate['reason']}")
    )


# ---------------------------------------------------------------------------------------------------------------------
# selftest


def selftest() -> int:
    """Synthetic h5, negatives, fragments and signal: a 1-bp offset planted on both strands must come out clean at
    +1, a strand split, a signal from other deltas and an offset of 3 ambiguous; the fallback decompressor must give
    the same numbers."""
    import pyBigWig

    rng = np.random.default_rng(11)
    lengths = {"chrA": 300_000, "chrB": 250_000, "chrC": 120_000}
    chosen = ["chrA", "chrB"]  # chrC is in every file but not chosen: its rows must be filtered out
    plus_delta, minus_delta = 0, 1
    v = {c: (10.0 ** ((rng.uniform(0.0, 1.0, size=n) - 0.5) * 2) - PSEUDO).astype(np.float32) for c, n in
         lengths.items()}  # fmt: skip
    b = {c: np.log10(x.astype(np.float64) + PSEUDO) for c, x in v.items()}
    failures = []

    def check(cond: bool, what: str) -> None:
        if not cond:
            failures.append(what)
            logger.error("selftest: FAILED %s", what)

    def weights(chrom: str, plant: int) -> np.ndarray:
        n = lengths[chrom]
        return 10.0 ** (2.0 * b[chrom][np.clip(np.arange(n) + plant, 0, n - 1)])

    def plant_cuts(plant_plus: int, plant_minus: int, n: int = 40_000) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Plus cuts drawn where b[i + plant_plus] is high, minus cuts where b[j + plant_minus] is: (plus, minus)."""
        out = {}
        for chrom, length in lengths.items():
            w = weights(chrom, plant_plus)[: length - 700]
            plus = rng.choice(w.size, size=n, p=w / w.sum())
            cand = plus[:, None] + rng.integers(40, 600, size=(n, 8))  # fragment ends, picked by the minus weights
            wc = weights(chrom, plant_minus)[cand]
            pick = (rng.random(n)[:, None] * wc.sum(axis=1, keepdims=True) < np.cumsum(wc, axis=1)).argmax(axis=1)
            out[chrom] = (plus, cand[np.arange(n), pick])
        return out

    def write_case(tmp: Path, name: str, cuts: dict, signal_minus_shift: int = 0) -> tuple[Path, Path]:
        """The fragments (starts and ends that the deltas turn back into the cuts) and the signal bigWig."""
        lines = ["# id=selftest\n", "# primary_contig=chrA\n"]
        for chrom, (plus, minus) in cuts.items():
            start, end = plus - plus_delta, minus + 1 - minus_delta
            order = np.lexsort((end, start))
            lines += [f"{chrom}\t{s}\t{e}\tAAAC-1\t1\n" for s, e in zip(start[order].tolist(), end[order].tolist())]
        text = tmp / f"{name}.fragments.tsv"
        text.write_text("".join(lines))
        frag = tmp / f"{name}.fragments.tsv.gz"
        bgzip = shutil.which("bgzip")
        if bgzip:
            with open(text, "rb") as src, open(frag, "wb") as dst:
                subprocess.run([bgzip, "-c"], stdin=src, stdout=dst, check=True)
        else:
            with open(text, "rb") as src, gzip.open(frag, "wb") as dst:
                shutil.copyfileobj(src, dst)
        sig = tmp / f"{name}.signal.bw"
        bw = pyBigWig.open(str(sig), "w")
        bw.addHeader(list(lengths.items()), maxZooms=0)
        for chrom, (plus, minus) in cuts.items():
            counts = np.bincount(plus, minlength=lengths[chrom]) + np.bincount(
                np.clip(minus + signal_minus_shift, 0, lengths[chrom] - 1), minlength=lengths[chrom]
            )
            where = np.flatnonzero(counts)
            bw.addEntries([chrom] * where.size, where.tolist(), ends=(where + 1).tolist(),
                          values=counts[where].astype(float).tolist())  # fmt: skip
        bw.close()
        return frag, sig

    with tempfile.TemporaryDirectory(prefix="tn5_align_selftest.") as tmp:
        tmp = Path(tmp)
        h5_path = tmp / "bias.h5"
        with h5py.File(h5_path, "w") as f:
            for chrom, x in v.items():
                f.create_dataset(chrom, data=x)
        neg = tmp / "negatives.bed"
        neg_lines = []
        for chrom, length in lengths.items():
            for s in rng.integers(10, length - 2114 - 10, size=60).tolist():
                neg_lines.append(f"{chrom}\t{s}\t{s + 2114}\t.\t.\t.\t.\t.\t.\t1057\n")
        neg.write_text("".join(neg_lines))
        common = {"chroms": chosen, "n_windows": 80, "max_offset": 5, "seed": 3, "threads": 2}

        cases = {
            "offset1": (plant_cuts(1, 1), 0, "clean", 1),
            "split": (plant_cuts(0, 1), 0, "ambiguous", None),
            "other_deltas": (plant_cuts(0, 0), 1, "ambiguous", None),
            "offset3": (plant_cuts(3, 3), 0, "ambiguous", None),
        }
        results = {}
        for name, (cuts, sig_shift, status, offset) in cases.items():
            frag, sig = write_case(tmp, name, cuts, sig_shift)
            rec = run(h5_path, frag, neg, sig, plus_delta, minus_delta, **common)
            write_outputs(rec, tmp / f"{name}.align.json", tmp / f"{name}.gate.json")
            results[name] = rec
            gate = json.loads((tmp / f"{name}.gate.json").read_text())
            check(gate["status"] == status and gate["offset"] == offset, f"{name}: {status}, offset {offset} "
                  f"(got {gate['status']}, {gate['offset']}: {gate['reason']})")  # fmt: skip
            n_rows = sum(len(cuts[c][0]) for c in chosen)
            check(rec["stream"]["rows_kept"] == n_rows, f"{name}: {n_rows} rows kept, chrC filtered out")
            check(rec["stream"]["header_lines"] == 2, f"{name}: the 2 header lines are skipped")
            logger.info("selftest %s: %s", name, summary(rec, tmp / f"{name}.align.json"))

        a = results["offset1"]
        check(a["argmax"] == {"plus": 1, "minus": 1, "both": 1}, "offset1: every strand at +1")
        check(a["signal_check"]["agreement"] == 1.0, "offset1: the counts reproduce the signal exactly")
        check(a["argmax_r"]["both"] == 1, "offset1: r peaks at +1 too")
        s = results["split"]
        check(s["argmax"]["plus"] == 0 and s["argmax"]["minus"] == 1, "split: plus at 0, minus at +1")
        check("differs between the strands" in s["gate"]["reason"], "split: the reason names the split")
        o = results["other_deltas"]
        check(o["argmax"] == {"plus": 0, "minus": 0, "both": 0}, "other_deltas: every strand at 0")
        check(o["signal_check"]["agreement"] < AGREEMENT_MIN and "reproduce" in o["gate"]["reason"],
              "other_deltas: refused for not reproducing the signal")  # fmt: skip
        check(results["offset3"]["argmax"]["both"] == 3, "offset3: found at +3, refused for being beyond +-1")

        frag = tmp / "offset1.fragments.tsv.gz"
        py = run(h5_path, frag, neg, tmp / "offset1.signal.bw", plus_delta, minus_delta, **common,
                 decompress=[sys.executable, "-I", "-c", PY_GUNZIP])  # fmt: skip
        check(py["m"] == a["m"] and py["stream"]["rows_kept"] == a["stream"]["rows_kept"],
              "Python's gzip gives what the default decompressor gives")  # fmt: skip

    if failures:
        print(f"tn5_align selftest: FAILED {len(failures)} check(s): {'; '.join(failures)}")
        return 1
    print("tn5_align selftest: all checks passed")
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# command line


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", type=Path, help="the bias h5: one float32 dataset per contig")
    ap.add_argument("--fragments", type=Path, help="the dataset's fragments (.tsv.gz, BGZF or gzip, '#' header)")
    ap.add_argument("--negatives", type=Path, help="background windows (BED)")
    ap.add_argument("--signal", type=Path, help="the dataset's unstranded cut-count bigWig the model is trained on")
    ap.add_argument("--plus-delta", type=int, help="the pipeline's plus-strand delta (0 for 10x fragments)")
    ap.add_argument("--minus-delta", type=int, help="the pipeline's minus-strand delta (+1 for 10x fragments)")
    ap.add_argument("--chroms", nargs="+", default=list(DEFAULT_CHROMS), help=f"({' '.join(DEFAULT_CHROMS)})")
    ap.add_argument("--n-windows", type=int, default=50000, help="negatives sampled (50000)")
    ap.add_argument("--max-offset", type=int, default=5, help="offsets d in [-max, max] (5)")
    ap.add_argument("--seed", type=int, default=1234, help="for the windows (1234)")
    ap.add_argument("--threads", type=int, default=8, help="bgzip's decompressing threads (8)")
    ap.add_argument("--bgzip", help="htslib's bgzip (default: on PATH, else beside the interpreter; else gzip -dc)")
    ap.add_argument("--out", type=Path, help="align.json")
    ap.add_argument("--gate-out", type=Path, help="gate.json")
    ap.add_argument("--selftest", action="store_true", help="a tiny synthetic run in a temporary directory")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%F %T")  # fmt: skip
    if args.selftest:
        return selftest()
    need = ("h5", "fragments", "negatives", "signal", "plus_delta", "minus_delta", "out", "gate_out")
    missing = [f"--{k.replace('_', '-')}" for k in need if getattr(args, k) is None]
    if missing:
        ap.error(f"{', '.join(missing)} required (or --selftest)")
    if args.n_windows < 1 or args.max_offset < GATE_MAX_OFFSET or args.threads < 1:
        ap.error(f"--n-windows and --threads must be positive and --max-offset at least {GATE_MAX_OFFSET}")
    if args.bgzip and not (shutil.which(args.bgzip) or os.access(args.bgzip, os.X_OK)):
        ap.error(f"--bgzip {args.bgzip} is not an executable")
    try:
        record = run(args.h5, args.fragments, args.negatives, args.signal, args.plus_delta, args.minus_delta,
                     chroms=args.chroms, n_windows=args.n_windows, max_offset=args.max_offset, seed=args.seed,
                     threads=args.threads, bgzip=args.bgzip)  # fmt: skip
        write_outputs(record, args.out, args.gate_out)
    except (ValueError, OSError, RuntimeError) as exc:
        logger.error("no gate: %s", exc)
        return 1
    print(summary(record, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
