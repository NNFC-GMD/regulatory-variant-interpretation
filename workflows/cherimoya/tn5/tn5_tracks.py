#!/usr/bin/env python3
"""scPrinter's per-base Tn5 bias, an h5 of one float32 dataset per contig, as two bigWig control tracks for Cherimoya.

The h5 holds v = 10^((x - 0.5) * 2) - 0.01 at every base, x being the bias CNN's output there, so log10(v + 0.01) =
2x - 1. Cherimoya concatenates a control track to the features its profile head's last convolution (fconv) reads, and
feeds log(sum + 1) of it to the counts head, so a control must be >= 0 everywhere. For every base of every chromosome
of --chrom-sizes, in that file's order, this writes:

    <prefix>.log.bw          y - ymin, y = log10(v + 0.01) and ymin its minimum over those chromosomes (>= 0): the bias
                             on the log scale, the one a profile head that adds the control in logit space wants
    <prefix>.linear.bw       v + 0.01 = 10^(2x - 1) (> 0)
    <prefix>.tracks.json     the sidecar: inputs (with the h5's size and mtime), transforms, ymin, statistics, the
                             offset and the gate it came from, the read-back check, versions, times (ET)

Both transforms are computed in float64 and stored as float32, every base: figwig's BigWigWriter with missing=NaN, so
that the log track's 0.0 is stored too, and no zoom levels. Track base i holds f(h5[i + offset]). The offset is the
alignment gate's (tn5_align.py's gate.json) when its status is "clean", and 0 when it is "ambiguous"; a missing or
unreadable gate is an error, not an offset of 0. A base whose i + offset falls off its chromosome takes f of the N
constant, the CNN's output on an all-N window, which is read from the h5 itself: the value of the runs of
N_EDGE_BASES equal values that the chromosomes start and end with (hg38's are telomeric N), see n_constant().

Pass 1 reads every chromosome, BLOCK_BASES at a time, in a pool of processes and collects y's minimum and maximum, a
histogram of y in bins of BIN_WIDTH (quantiles 0, .001, .01, .5, .99, .999, 1 from it, to a bin), and the bases equal
to the N constant, with v < 0, or where v + 0.01 is not positive or not finite (an error: the transforms are undefined
there). Pass 2 writes the two tracks at once, one process each, with n_jobs = threads // 2 compressing threads each.
Then --verify-n random 1-kb windows, and the first and last kb of every chromosome (where the offset shows), are read
back with pyBigWig and compared with the same float32 transform of the h5: no base may come back NaN, every value
must be equal (or within 1e-6 relative), and the header's chromosomes must be the sizes' (its count of bases covered
is recorded beside the sizes' total, a warning when they differ, not a fail). Only then are the two
tracks renamed into place and the sidecar written; a failed check leaves them as <name>.partial beside
<prefix>.tracks.failed.json. Outputs being replaced (an older offset or h5) are removed first, so a sidecar never
describes tracks it did not check.

    python tn5_tracks.py --h5 <bias.h5> --chrom-sizes <chrom.sizes> --gate <gate.json> --out-dir <dir>
        [--prefix scprinter_v2] [--threads 32] [--verify-n 10000] [--seed 1234]
    python tn5_tracks.py --selftest

Nothing is written when the sidecar records the same offset, the same h5 (size and mtime) and the two tracks as they
are on disk (exit 0, "up to date"; a changed gate status or reason is copied into the sidecar). Exit 0 when written
or up to date, 1 on an error or a failed check. Run it in an environment with figwig, h5py, numpy and pyBigWig,
inside a box: it holds blocks, not chromosomes (a few GB in all), and each track is ~12-16 GB on disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing
import os
import resource
import socket
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
logger = logging.getLogger("tn5_tracks")

FORMAT = 1
"""The layout of the sidecar; a sidecar of another format is written again."""
TRACKS = ("log", "linear")
PSEUDO = 0.01
"""scPrinter's: v = 10^((x - 0.5) * 2) - PSEUDO, so log10(v + PSEUDO) = 2x - 1."""
N_CONSTANT_HINT = 0.251753
"""scPrinter v2's value on N runs, as printed: the fallback when no chromosome edge holds a run, and a sanity check."""
N_EDGE_BASES = 1000
"""A chromosome edge whose first (last) this many values are all equal holds the N constant."""
BLOCK_BASES = 1 << 24
"""Bases read, transformed and written at a time: 64 MB of float32."""
BIN_WIDTH = 1e-4
Y_RANGE = (-12.0, 6.0)
"""The histogram's range of y; values outside it are counted in an under- and an overflow bin."""
QUANTILES = (0.0, 0.001, 0.01, 0.5, 0.99, 0.999, 1.0)
VERIFY_WIDTH = 1000
VERIFY_RTOL = 1e-6
PASS1_WORKERS = 8
"""Pass 1 is bound by reading the h5, so more processes than this buy little."""
ET = ZoneInfo("America/New_York")


class TracksFailed(RuntimeError):
    """The written tracks did not read back as the transform of the h5."""


# ---------------------------------------------------------------------------------------------------------------------
# files


def partial(path: Path) -> Path:
    return Path(f"{path}.partial")


def remove(path: Path) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def write_json(path: Path, data: dict) -> None:
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


def now_et() -> str:
    return datetime.now(ET).isoformat(timespec="seconds")


def read_chrom_sizes(path: Path) -> list[tuple[str, int]]:
    """(name, length) pairs in the file's order; the first two columns of each line, blank and # lines skipped."""
    sizes: list[tuple[str, int]] = []
    seen = set()
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        try:
            name, length = fields[0], int(fields[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"{path}, line {n}: not '<chrom> <length>': {line!r}") from exc
        if name in seen or length <= 0:
            raise ValueError(f"{path}, line {n}: {name} is repeated or has length {length}")
        seen.add(name)
        sizes.append((name, length))
    if not sizes:
        raise ValueError(f"{path} names no chromosome")
    return sizes


def read_gate(path: Path) -> tuple[dict, int]:
    """(gate, offset): offset = gate["offset"] when the status is "clean", else 0."""
    try:
        gate = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read the gate {path}: {exc}") from exc
    status = gate.get("status")
    if status == "clean":
        offset = gate.get("offset")
        if isinstance(offset, bool) or not isinstance(offset, int):
            raise ValueError(f"{path}: status clean with offset {offset!r}, not an integer")
    elif status == "ambiguous":
        offset = 0
    else:
        raise ValueError(f"{path}: status {status!r}, not 'clean' or 'ambiguous'")
    if abs(offset) >= N_EDGE_BASES:
        raise ValueError(f"{path}: offset {offset} is not a few bases")
    return gate, offset


def check_h5(h5_path: Path, sizes: list[tuple[str, int]]) -> None:
    """Every chromosome of the sizes is a 1-d float32 dataset of the h5, of exactly its length."""
    with h5py.File(h5_path, "r") as f:
        for chrom, length in sizes:
            if chrom not in f:
                raise ValueError(f"{h5_path} has no dataset {chrom}")
            ds = f[chrom]
            if ds.ndim != 1 or ds.shape[0] != length or ds.dtype != np.float32:
                raise ValueError(f"{h5_path}:{chrom} is {ds.dtype} {ds.shape}, the sizes say float32 ({length},)")


# ---------------------------------------------------------------------------------------------------------------------
# the transforms


def transform(v32: np.ndarray, kind: str, ymin: float) -> np.ndarray:
    """The track's float32 values of h5 values v32: in float64, then rounded once. The writer and the check both call
    this, so a base that reads back right reads back equal."""
    v = v32.astype(np.float64)
    if kind == "log":
        return (np.log10(v + PSEUDO) - ymin).astype(np.float32)
    if kind == "linear":
        return (v + PSEUDO).astype(np.float32)
    raise ValueError(f"no track {kind!r}")


def read_shifted(ds, length: int, start: int, end: int, offset: int, fill: np.float32) -> np.ndarray:
    """h5 values at [start + offset, end + offset) of a chromosome of `length` bases, as float32; the bases off the
    chromosome hold `fill`."""
    out = np.full(end - start, fill, dtype=np.float32)
    lo, hi = max(start + offset, 0), min(end + offset, length)
    if hi > lo:
        out[lo - start - offset : hi - start - offset] = ds[lo:hi]
    return out


def n_constant(h5_path: Path, sizes: list[tuple[str, int]]) -> tuple[np.float32, dict]:
    """The h5's N constant: the most common value of the chromosome edges that hold a run of N_EDGE_BASES equal
    values; N_CONSTANT_HINT when none does."""
    runs: dict[float, list[str]] = {}
    with h5py.File(h5_path, "r") as f:
        for chrom, length in sizes:
            width = min(N_EDGE_BASES, length)
            for edge, a in (("start", 0), ("end", length - width)):
                vals = f[chrom][a : a + width]
                if vals.size and np.isfinite(vals[0]) and (vals == vals[0]).all():
                    runs.setdefault(float(vals[0]), []).append(f"{chrom}:{edge}")
    if not runs:
        logger.warning("no chromosome starts or ends with %d equal values; N constant taken as %s", N_EDGE_BASES,
                       N_CONSTANT_HINT)  # fmt: skip
        return np.float32(N_CONSTANT_HINT), {
            "value": float(np.float32(N_CONSTANT_HINT)),
            "source": f"default (no chromosome edge holds a run of {N_EDGE_BASES} equal values)",
            "edges": {},
        }
    value = max(runs, key=lambda k: len(runs[k]))
    if abs(value - N_CONSTANT_HINT) > 1e-3:
        logger.warning("the h5's N constant is %r, scPrinter v2's is %s", value, N_CONSTANT_HINT)
    return np.float32(value), {
        "value": value,
        "source": f"the value of the {len(runs[value])} chromosome edges that start or end with {N_EDGE_BASES} equal "
        "values",
        "edges": {repr(k): sorted(e) for k, e in runs.items()},
        "differs_from_hint": abs(value - N_CONSTANT_HINT) > 1e-3,
    }


# ---------------------------------------------------------------------------------------------------------------------
# pass 1: statistics


def n_bins() -> int:
    return int(round((Y_RANGE[1] - Y_RANGE[0]) / BIN_WIDTH))


def chrom_stats(h5_path: str, chrom: str, length: int, nconst: float, block_bases: int) -> dict:
    """y = log10(v + 0.01)'s statistics over one chromosome, in float64 (a pool worker)."""
    hist = np.zeros(n_bins() + 2, dtype=np.int64)  # [underflow, bins..., overflow]
    st = {"chrom": chrom, "bases": length, "n_constant": 0, "v_negative": 0, "nonpositive": 0, "nonfinite": 0}
    v_min, v_max, y_min, y_max, y_sum = np.inf, -np.inf, np.inf, -np.inf, 0.0
    with h5py.File(h5_path, "r") as f:
        ds = f[chrom]
        for a in range(0, length, block_bases):
            v32 = ds[a : min(a + block_bases, length)]
            finite = np.isfinite(v32)
            if not finite.all():
                st["nonfinite"] += int((~finite).sum())
                v32 = v32[finite]
            st["n_constant"] += int((v32 == np.float32(nconst)).sum())
            st["v_negative"] += int((v32 < 0).sum())
            if v32.size:
                v_min, v_max = min(v_min, float(v32.min())), max(v_max, float(v32.max()))
            v = v32.astype(np.float64) + PSEUDO
            positive = v > 0
            if not positive.all():
                st["nonpositive"] += int((~positive).sum())
                v = v[positive]
            if v.size == 0:
                continue
            y = np.log10(v)
            y_min, y_max, y_sum = min(y_min, float(y.min())), max(y_max, float(y.max())), y_sum + float(y.sum())
            idx = np.floor((y - Y_RANGE[0]) / BIN_WIDTH).astype(np.int64) + 1
            np.clip(idx, 0, hist.size - 1, out=idx)
            hist += np.bincount(idx, minlength=hist.size)
    st.update(v_min=v_min, v_max=v_max, y_min=y_min, y_max=y_max, y_sum=y_sum, hist=hist)
    return st


def quantiles(hist: np.ndarray, n: int, y_min: float, y_max: float) -> dict[str, float]:
    """Quantiles of y from its histogram: the centre of the bin holding the value of rank q * (n - 1), within the
    exact minimum and maximum; q = 0 and 1 are those exactly."""
    cum = np.cumsum(hist)
    out = {}
    for q in QUANTILES:
        if q == 0.0:
            out[str(q)] = y_min
        elif q == 1.0:
            out[str(q)] = y_max
        else:
            k = int(np.searchsorted(cum, int(q * (n - 1)) + 1))
            centre = Y_RANGE[0] + (k - 0.5) * BIN_WIDTH
            out[str(q)] = float(min(max(centre, y_min), y_max))
    return out


def genome_stats(per_chrom: list[dict]) -> dict:
    """Pooled pass-1 statistics; raises where the transforms are undefined."""
    bad = {s["chrom"]: (s["nonfinite"], s["nonpositive"]) for s in per_chrom if s["nonfinite"] or s["nonpositive"]}
    if bad:
        raise ValueError(
            "v is not finite, or v + 0.01 is not positive, at bases of "
            + ", ".join(f"{c} ({a} non-finite, {b} non-positive)" for c, (a, b) in bad.items())
        )
    hist = np.sum([s["hist"] for s in per_chrom], axis=0)
    n = int(sum(s["bases"] for s in per_chrom))
    y_min, y_max = min(s["y_min"] for s in per_chrom), max(s["y_max"] for s in per_chrom)
    q = quantiles(hist, n, y_min, y_max)
    return {
        "bases": n,
        "y_min": y_min,
        "y_max": y_max,
        "y_mean": sum(s["y_sum"] for s in per_chrom) / n,
        "v_min": min(s["v_min"] for s in per_chrom),
        "v_max": max(s["v_max"] for s in per_chrom),
        "y_quantiles": q,
        "v_quantiles": {k: 10.0**y - PSEUDO for k, y in q.items()},
        "quantile_resolution": f"bins of {BIN_WIDTH} in y over {list(Y_RANGE)}; q = 0 and 1 exact",
        "histogram_outside_range": {"under": int(hist[0]), "over": int(hist[-1])},
        "n_constant_bases": int(sum(s["n_constant"] for s in per_chrom)),
        "v_negative_bases": int(sum(s["v_negative"] for s in per_chrom)),
        "per_chrom": {
            s["chrom"]: {
                k: s[k] for k in ("bases", "y_min", "y_max", "v_min", "v_max", "n_constant", "v_negative")
            }  # fmt: skip
            for s in per_chrom
        },
    }


# ---------------------------------------------------------------------------------------------------------------------
# pass 2: the tracks


def write_track(kind: str, h5_path: str, sizes: list[tuple[str, int]], offset: int, ymin: float, fill: float,
                path: str, n_jobs: int, block_bases: int) -> dict:  # fmt: skip
    """Write one track, every base, chromosome by chromosome in the sizes' order (a pool worker)."""
    import figwig

    began = time.perf_counter()
    fill32 = np.float32(fill)
    with h5py.File(h5_path, "r") as f, figwig.BigWigWriter(path, sizes, zooms=0, n_jobs=n_jobs) as writer:
        engine, level = writer.engine, writer.level
        for chrom, length in sizes:
            ds = f[chrom]
            for a in range(0, length, block_bases):
                b = min(a + block_bases, length)
                values = transform(read_shifted(ds, length, a, b, offset, fill32), kind, ymin)
                writer.write(chrom, np.array([a], dtype=np.int64), values[None, :], missing=np.nan)
    return {
        "kind": kind,
        "partial": path,
        "seconds": time.perf_counter() - began,
        "engine": engine,
        "level": level,
        "n_jobs": n_jobs,
        "zooms": 0,
    }


def bw_values(bw, chrom: str, start: int, end: int) -> np.ndarray:
    import pyBigWig

    if getattr(pyBigWig, "numpy", 0):
        return np.asarray(bw.values(chrom, start, end, numpy=True), dtype=np.float64)
    return np.array(bw.values(chrom, start, end), dtype=np.float64)


def verify(paths: dict[str, Path], h5_path: Path, sizes: list[tuple[str, int]], offset: int, ymin: float,
           fill: np.float32, n: int, seed: int) -> dict:  # fmt: skip
    """Read the tracks back with pyBigWig over n random windows and every chromosome's first and last kb, against the
    float32 transform of the h5; and their headers against the sizes."""
    import pyBigWig

    rng = np.random.default_rng(seed)
    lengths = np.array([length for _, length in sizes], dtype=np.int64)
    total = int(lengths.sum())
    windows = []
    for chrom, length in sizes:
        width = min(VERIFY_WIDTH, length)
        windows += [(chrom, 0, width), (chrom, length - width, length)]
    for i in rng.choice(len(sizes), size=n, p=lengths / total).tolist():
        chrom, length = sizes[i]
        width = min(VERIFY_WIDTH, length)
        start = int(rng.integers(0, length - width + 1))
        windows.append((chrom, start, start + width))
    order = {chrom: i for i, (chrom, _) in enumerate(sizes)}
    windows.sort(key=lambda w: (order[w[0]], w[1]))
    length_of = dict(sizes)

    out: dict = {"windows": len(windows), "random_windows": n, "edge_windows": 2 * len(sizes), "seed": seed}
    bws = {kind: pyBigWig.open(str(path)) for kind, path in paths.items()}
    try:
        for kind, bw in bws.items():
            header = bw.header()
            out[kind] = {
                "chroms_match": dict(bw.chroms()) == dict(sizes),
                "bases_covered": int(header["nBasesCovered"]),
                "bases_expected": total,
                "bases": 0,
                "nan": 0,
                "unequal": 0,
                "max_abs_diff": 0.0,
                "max_rel_diff": 0.0,
            }
        with h5py.File(h5_path, "r") as f:
            for chrom, start, end in windows:
                raw = read_shifted(f[chrom], length_of[chrom], start, end, offset, fill)
                for kind, bw in bws.items():
                    want = transform(raw, kind, ymin).astype(np.float64)
                    got = bw_values(bw, chrom, start, end)
                    rec = out[kind]
                    nan = np.isnan(got)
                    diff = np.abs(np.where(nan, want, got) - want)
                    rec["bases"] += int(got.size)
                    rec["nan"] += int(nan.sum())
                    rec["unequal"] += int((diff > 0).sum())
                    rec["max_abs_diff"] = max(rec["max_abs_diff"], float(diff.max(initial=0.0)))
                    rel = diff / np.maximum(np.abs(want), np.finfo(np.float32).tiny)
                    rec["max_rel_diff"] = max(rec["max_rel_diff"], float(rel.max(initial=0.0)))
    finally:
        for bw in bws.values():
            bw.close()
    for kind in paths:
        rec = out[kind]
        rec["pass"] = bool(
            rec["chroms_match"]
            and rec["nan"] == 0
            and (rec["max_abs_diff"] == 0.0 or rec["max_rel_diff"] <= VERIFY_RTOL)
        )
        # The header's total summary is figwig's bookkeeping, not the values: recorded and warned about, not a fail.
        rec["bases_covered_match"] = rec["bases_covered"] == total
        if not rec["bases_covered_match"]:
            logger.warning("%s: the header counts %d bases covered, the sizes %d", paths[kind], rec["bases_covered"],
                           total)  # fmt: skip
    out["rule"] = (
        f"every base read back (no NaN), equal or within {VERIFY_RTOL} relative, and the header's chromosomes the "
        "sizes'; its bases covered against their sum is recorded, not a condition"
    )
    out["pass"] = all(out[kind]["pass"] for kind in paths)
    return out


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


def versions() -> dict:
    import figwig
    import pyBigWig

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "h5py": h5py.__version__,
        "hdf5": h5py.version.hdf5_version,
        "figwig": figwig.__version__,
        "pyBigWig": getattr(pyBigWig, "__version__", None),
        "pyBigWig_numpy": bool(getattr(pyBigWig, "numpy", 0)),
    }


def gate_record(gate_path: Path, gate: dict) -> dict:
    return {
        "path": str(gate_path),
        "status": gate.get("status"),
        "reason": gate.get("reason"),
        "offset": gate.get("offset"),
    }


def up_to_date(sidecar: Path, paths: dict[str, Path], offset: int, h5: dict) -> tuple[str | None, dict | None]:
    """(None, sidecar) when the sidecar describes these tracks for this offset and this h5, else (why not, None)."""
    try:
        rec = json.loads(sidecar.read_text())
    except FileNotFoundError:
        return "not written yet", None
    except (OSError, ValueError) as exc:
        return f"unreadable {sidecar}: {exc}", None
    if rec.get("format") != FORMAT:
        return f"sidecar format {rec.get('format')!r}, this writes {FORMAT}", None
    if rec.get("offset") != offset:
        return f"the sidecar's offset is {rec.get('offset')!r}, the gate's {offset}", None
    old = (rec.get("inputs") or {}).get("h5") or {}
    if (old.get("size"), old.get("mtime_ns")) != (h5["size"], h5["mtime_ns"]):
        return "the h5 is not the one the sidecar records", None
    if not (rec.get("verification") or {}).get("pass"):
        return "the sidecar records a failed check", None
    for kind, path in paths.items():
        written = ((rec.get("outputs") or {}).get(kind)) or {}
        try:
            st = os.stat(path)
        except FileNotFoundError:
            return f"no {path}", None
        if (written.get("size"), written.get("mtime_ns")) != (st.st_size, st.st_mtime_ns):
            return f"{path} is not the file the sidecar records", None
    return None, rec


def run(h5_path: Path, chrom_sizes: Path, gate_path: Path, out_dir: Path, *, prefix: str = "scprinter_v2",
        threads: int = 32, verify_n: int = 10000, seed: int = 1234,
        block_bases: int = BLOCK_BASES) -> tuple[str, dict]:  # fmt: skip
    """Write the two tracks and the sidecar. Returns ("written", sidecar) or ("up to date", sidecar); raises
    TracksFailed when they do not read back right and ValueError on bad inputs."""
    began = time.perf_counter()
    h5_path, chrom_sizes = Path(os.path.abspath(h5_path)), Path(os.path.abspath(chrom_sizes))
    gate_path, out_dir = Path(os.path.abspath(gate_path)), Path(os.path.abspath(out_dir))
    sizes = read_chrom_sizes(chrom_sizes)
    gate, offset = read_gate(gate_path)
    paths = {kind: out_dir / f"{prefix}.{kind}.bw" for kind in TRACKS}
    sidecar = out_dir / f"{prefix}.tracks.json"
    failed = out_dir / f"{prefix}.tracks.failed.json"
    h5_stat = stat_file(h5_path)

    reason, rec = up_to_date(sidecar, paths, offset, h5_stat)
    if reason is None:
        if rec.get("gate") != gate_record(gate_path, gate):
            rec["gate"] = gate_record(gate_path, gate)
            rec["gate_refreshed_et"] = now_et()
            write_json(sidecar, rec)
        return "up to date", rec
    logger.info("writing the tracks with offset %d (%s)", offset, reason)

    check_h5(h5_path, sizes)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in (sidecar, failed, *paths.values(), *(partial(p) for p in paths.values())):
        remove(path)  # from here on, no sidecar describes tracks that are being replaced
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # numpy in each worker would otherwise start a thread per CPU
    ctx = multiprocessing.get_context("spawn")  # no h5py or numba state forked into the workers

    t = time.perf_counter()
    nconst, nconst_record = n_constant(h5_path, sizes)
    workers = max(1, min(len(sizes), threads, PASS1_WORKERS))
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        per_chrom = list(
            pool.map(
                chrom_stats,
                [str(h5_path)] * len(sizes),
                [c for c, _ in sizes],
                [n for _, n in sizes],
                [float(nconst)] * len(sizes),
                [block_bases] * len(sizes),
            )
        )
    stats = genome_stats(per_chrom)
    ymin = stats["y_min"]
    pass1_s = time.perf_counter() - t
    logger.info("pass 1: %d bases, y in [%.6f, %.6f], N constant %r on %d bases, v < 0 on %d (%.1f s)",
                stats["bases"], ymin, stats["y_max"], float(nconst), stats["n_constant_bases"],
                stats["v_negative_bases"], pass1_s)  # fmt: skip

    t = time.perf_counter()
    n_jobs = max(1, threads // len(TRACKS))
    with ProcessPoolExecutor(max_workers=len(TRACKS), mp_context=ctx) as pool:
        futures = {}
        for kind, path in paths.items():
            args = (kind, str(h5_path), sizes, offset, ymin, float(nconst), str(partial(path)), n_jobs, block_bases)
            futures[kind] = pool.submit(write_track, *args)
        written = {kind: future.result() for kind, future in futures.items()}
    pass2_s = time.perf_counter() - t
    logger.info("pass 2: wrote both tracks in %.1f s", pass2_s)

    t = time.perf_counter()
    check = verify({k: partial(p) for k, p in paths.items()}, h5_path, sizes, offset, ymin, nconst, verify_n, seed)
    verify_s = time.perf_counter() - t

    now = datetime.now().astimezone()
    record = {
        "format": FORMAT,
        "tool": "tn5_tracks.py",
        "written": now.isoformat(timespec="seconds"),
        "written_et": now.astimezone(ET).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "argv": list(sys.argv),
        "code": code_identity(),
        "versions": versions(),
        "inputs": {"h5": h5_stat, "chrom_sizes": stat_file(chrom_sizes), "gate": stat_file(gate_path)},
        "gate": gate_record(gate_path, gate),
        "offset": offset,
        "offset_rule": "t[i] = f(h5[i + offset]); offset = the gate's when its status is clean, else 0",
        "chroms": [c for c, _ in sizes],
        "transforms": {
            "log": "float32(log10(float64(v) + 0.01) - ymin), ymin = min over the chromosomes of log10(v + 0.01)",
            "linear": "float32(float64(v) + 0.01)",
            "off_chromosome": "v = the N constant",
        },
        "ymin": ymin,
        "n_constant": nconst_record,
        "stats": stats,
        "writer": {kind: {k: w[k] for k in ("engine", "level", "n_jobs", "zooms")} for kind, w in written.items()},
        "verification": check,
        "timings_s": {"pass1": pass1_s, "pass2": pass2_s, "verify": verify_s, "total": time.perf_counter() - began},
        "rss_max_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
    if not check["pass"]:
        record["outputs"] = {kind: stat_file(partial(p)) for kind, p in paths.items()}
        write_json(failed, record)
        raise TracksFailed(f"the tracks did not read back as written; see {failed}")
    for path in paths.values():
        os.replace(partial(path), path)
    record["outputs"] = {kind: stat_file(p) for kind, p in paths.items()}
    write_json(sidecar, record)
    return "written", record


# ---------------------------------------------------------------------------------------------------------------------
# selftest


def selftest() -> int:
    """Tiny synthetic h5 and sizes: write with offsets +1, 0 (an ambiguous gate) and -1, read every base back, and
    check the edges, the statistics, the skip and the refusals."""
    import pyBigWig

    rng = np.random.default_rng(7)
    nconst = np.float32(N_CONSTANT_HINT)
    block = 1 << 20  # several blocks per chromosome, the last one short
    lengths = {"chrA": 2 * block + 456_789, "chrB": 1_234_567, "chrUn_extra": 5_000}
    data = {}
    for chrom, length in lengths.items():
        x = rng.uniform(-0.8, 1.6, size=length)  # v < 0 below x = -0.5
        data[chrom] = (10.0 ** ((x - 0.5) * 2) - PSEUDO).astype(np.float32)
    data["chrA"][:3000] = nconst
    data["chrA"][1_000_000:1_050_000] = nconst
    data["chrA"][-2000:] = nconst
    # chrB's ends are not N, so a base an offset pushes off it shows whether it took the fill or the h5's edge value.
    main = ("chrB", "chrA")  # the sizes' order, not the h5's
    ymin = min(float(np.log10(data[c].astype(np.float64) + PSEUDO).min()) for c in main)

    def expected(kind: str, chrom: str, offset: int) -> np.ndarray:
        v = data[chrom]
        shifted = np.full(v.size, nconst, dtype=np.float32)
        if offset >= 0:
            shifted[: v.size - offset] = v[offset:]
        else:
            shifted[-offset:] = v[: v.size + offset]
        return transform(shifted, kind, ymin)

    failures = []

    def check(cond: bool, what: str) -> None:
        if not cond:
            failures.append(what)
            logger.error("selftest: FAILED %s", what)

    with tempfile.TemporaryDirectory(prefix="tn5_tracks_selftest.") as tmp:
        tmp = Path(tmp)
        h5_path = tmp / "bias.h5"
        with h5py.File(h5_path, "w") as f:
            for chrom, v in data.items():
                f.create_dataset(chrom, data=v)
        sizes = tmp / "chrom.sizes"
        sizes.write_text("".join(f"{c}\t{lengths[c]}\n" for c in main))
        gates = {
            "plus1": {"status": "clean", "offset": 1, "reason": "test"},
            "zero": {"status": "ambiguous", "offset": None, "reason": "test: strands disagree"},
            "minus1": {"status": "clean", "offset": -1, "reason": "test"},
        }
        for name, gate in gates.items():
            (tmp / f"{name}.json").write_text(json.dumps(gate))

        for name, offset in (("plus1", 1), ("zero", 0), ("minus1", -1)):
            out = tmp / name
            status, rec = run(h5_path, sizes, tmp / f"{name}.json", out, threads=4, verify_n=200, block_bases=block)
            check(status == "written" and rec["offset"] == offset, f"{name}: written with offset {offset}")
            check(rec["verification"]["pass"], f"{name}: the read-back check passes")
            check(rec["ymin"] == ymin, f"{name}: ymin {rec['ymin']} == {ymin}")
            check(rec["n_constant"]["value"] == float(nconst), f"{name}: the N constant is found")
            n_const = sum(int((data[c] == nconst).sum()) for c in main)
            n_neg = sum(int((data[c] < 0).sum()) for c in main)
            check(rec["stats"]["n_constant_bases"] == n_const, f"{name}: N-constant bases {n_const}")
            check(rec["stats"]["v_negative_bases"] == n_neg > 0, f"{name}: v < 0 bases {n_neg}")
            check(rec["stats"]["bases"] == sum(lengths[c] for c in main), f"{name}: only the sizes' chromosomes")
            for kind in TRACKS:
                bw = pyBigWig.open(str(out / f"scprinter_v2.{kind}.bw"))
                try:
                    check(dict(bw.chroms()) == {c: lengths[c] for c in main}, f"{name} {kind}: the header")
                    for chrom in main:
                        got = bw_values(bw, chrom, 0, lengths[chrom])
                        want = expected(kind, chrom, offset).astype(np.float64)
                        check(not np.isnan(got).any(), f"{name} {kind} {chrom}: every base stored")
                        check(np.array_equal(got, want), f"{name} {kind} {chrom}: every base equal")
                        edge = transform(np.array([nconst]), kind, ymin)[0]
                        if offset > 0:
                            check(got[-1] == edge and got[0] == want[0], f"{name} {kind} {chrom}: the end is N")
                        if offset < 0:
                            check(got[0] == edge, f"{name} {kind} {chrom}: the start is N")
                        if offset != 0 and chrom == "chrB":
                            own = transform(data[chrom][[-1 if offset > 0 else 0]], kind, ymin)[0]
                            check(own != edge, f"{name} {kind} chrB: its edge value is not the fill, so the fill shows")
                    if kind == "log":
                        lows = [float(bw_values(bw, c, 0, lengths[c]).min()) for c in main]
                        check(min(lows) >= 0.0, f"{name} log: >= 0")
                        if offset == 0:
                            check(min(lows) == 0.0, f"{name} log: its minimum, 0.0, is stored")
                finally:
                    bw.close()

        status, _ = run(h5_path, sizes, tmp / "plus1.json", tmp / "plus1", threads=4, verify_n=200, block_bases=block)
        check(status == "up to date", "a second run with the same gate and h5 is a skip")

        bad_sizes = tmp / "bad.sizes"
        bad_sizes.write_text(f"chrA\t{lengths['chrA'] + 1}\n")
        try:
            run(h5_path, bad_sizes, tmp / "plus1.json", tmp / "bad", threads=2, verify_n=10, block_bases=block)
            check(False, "a length the h5 disagrees with is refused")
        except ValueError:
            pass
        bad_gate = tmp / "bad_gate.json"
        bad_gate.write_text(json.dumps({"status": "clean", "offset": None}))
        try:
            run(h5_path, sizes, bad_gate, tmp / "bad", threads=2, verify_n=10, block_bases=block)
            check(False, "a clean gate without an offset is refused")
        except ValueError:
            pass
        try:
            run(h5_path, sizes, tmp / "no_such_gate.json", tmp / "bad", threads=2, verify_n=10, block_bases=block)
            check(False, "a missing gate is refused")
        except ValueError:
            pass

    if failures:
        print(f"tn5_tracks selftest: FAILED {len(failures)} check(s): {'; '.join(failures)}")
        return 1
    print("tn5_tracks selftest: all checks passed")
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# command line


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", type=Path, help="the bias h5: one float32 dataset per contig")
    ap.add_argument("--chrom-sizes", type=Path, help="the chromosomes to write, and their order")
    ap.add_argument("--gate", type=Path, help="tn5_align.py's gate.json")
    ap.add_argument("--out-dir", type=Path, help="where the tracks and the sidecar go")
    ap.add_argument("--prefix", default="scprinter_v2", help="file names: <prefix>.{log,linear}.bw (scprinter_v2)")
    ap.add_argument("--threads", type=int, default=32, help="pass 1 processes (at most 8) and compressing threads")
    ap.add_argument("--verify-n", type=int, default=10000, help="random 1-kb windows read back (10000)")
    ap.add_argument("--seed", type=int, default=1234, help="for the read-back windows (1234)")
    ap.add_argument("--selftest", action="store_true", help="a tiny synthetic run in a temporary directory")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%F %T")  # fmt: skip
    if args.selftest:
        return selftest()
    missing = [f"--{k.replace('_', '-')}" for k in ("h5", "chrom_sizes", "gate", "out_dir") if getattr(args, k) is None]
    if missing:
        ap.error(f"{', '.join(missing)} required (or --selftest)")
    if args.threads < 1 or args.verify_n < 0:
        ap.error("--threads must be positive and --verify-n not negative")
    try:
        status, rec = run(args.h5, args.chrom_sizes, args.gate, args.out_dir, prefix=args.prefix, threads=args.threads,
                          verify_n=args.verify_n, seed=args.seed)  # fmt: skip
    except (ValueError, OSError, TracksFailed) as exc:
        logger.error("no tracks: %s", exc)
        return 1
    check = rec["verification"]
    print(
        f"tn5_tracks: {status}: {', '.join(rec['outputs'][k]['path'] for k in TRACKS)}; offset {rec['offset']} "
        f"(gate {rec['gate']['status']}); ymin {rec['ymin']:.6f}; {rec['stats']['bases']} bases; read-back "
        f"{'pass' if check['pass'] else 'FAIL'} on {check['windows']} windows"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
