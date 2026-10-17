#!/usr/bin/env python3
"""The Tn5 bias CNN against the genome-wide bias h5 it made: output length, registration, strand symmetry.

The CNN is a TorchScript module: a (B, 4, L) one-hot of A, C, G, T in (that order), (B, L - 100) out, each output base
seeing +-50 bp (valid convolutions, no padding). The h5 holds v = 10^((x - 0.5) * 2) - 0.01 of its output x at every
base, so log10(v + 0.01) = 2x - 1. Before the CNN is run on sequences the h5 does not cover (an alternative allele, a
shuffled sequence), three things are checked on --n windows of L = 1100 bp centred on negatives (--seed; on
--chroms, default every chromosome of the negatives), skipping a window with a base other than A, C, G or T
(soft-masked bases count as what they are):

    output length   the CNN maps L bases to L - 100
    registration    2x - 1 against log10(h5[s + 50 + k : s + L - 50 + k] + 0.01) of the window [s, s + L): at k = 0
                    the max |difference| must be <= 1e-4 (its q99, median and mean, and the pooled Pearson, are
                    reported too); k in -3..+3 is searched and the k of the smallest median reported, so that a
                    misregistration shows as its size
    symmetry        y = 2x - 1 of m(x) against yr = 2x - 1 of m(rc(x)).flip(-1), rc = x.flip(1, 2) (the channel flip
                    is the complement in ACGT order), y[j] against yr[j + k] for k in -2..+2: pooled Pearson, median
                    and q99 |difference|. Tn5 inserts as a symmetric dimer, so the bias of a cut should not depend
                    on the strand it is read from; pass iff r >= 0.98 and median |difference| <= 0.05 at k = 0

Informational, not a gate: every flag is recorded, and the exit is 0 whether they pass or not. The CNN runs in
float64 (on --device, cuda by default, else the CPU): TF32 cannot touch float64, which matters here because the
traced graph records allow_tf32=True in each of its convolutions (torch._convolution's last argument, from the
flags at trace time), so the global switches, which are also turned off, might not reach them. The h5's own float32
rounding is ~1e-7 on 2x - 1, far inside the tolerance; but the h5 may have been made at a lower precision (TF32 or
half on a GPU puts 2x - 1 ~1e-3 off a float64 recompute). So a registration fail with best k = 0 and a small median
is that precision, not a slip; a slip shows as a best k other than 0, its median far below the others'.

    python tn5_cnn_check.py --pt <Tn5_NN_model.pt> --h5 <bias.h5> --fasta <genome.fa> --negatives <negatives.bed>
        [--n 20000] [--chroms ...] [--seed 1234] [--device cuda] [--batch 512] --out <cnn.json>
    python tn5_cnn_check.py --selftest

--out (written to <name>.partial and renamed) records the inputs (sizes, mtimes), the windows, the device, dtype and
precision flags, the three checks with their numbers and pass flags, versions and timings, the time in ET. Exit 0
when the checks ran, 1 on an error. Run it in the cherimoya environment (torch, numpy, h5py, pyfaidx), with a GPU if
there is one; it holds the windows' outputs, about 0.5 GB at --n 20000.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import resource
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
import torch

HERE = Path(__file__).resolve().parent
logger = logging.getLogger("tn5_cnn_check")

FORMAT = 1
WINDOW = 1100
FLANK = 50
"""The CNN's receptive field is 2 * FLANK + 1: L bases in, L - 2 * FLANK out."""
PSEUDO = 0.01
REG_OFFSETS = tuple(range(-3, 4))
SYM_OFFSETS = tuple(range(-2, 3))
REG_TOL = 1e-4
SYM_R_MIN = 0.98
SYM_MEDIAN_MAX = 0.05
DTYPE = torch.float64
ET = ZoneInfo("America/New_York")

_CODES = np.full(256, 255, dtype=np.uint8)
for _i, _base in enumerate(b"ACGT"):
    _CODES[_base] = _i


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


# ---------------------------------------------------------------------------------------------------------------------
# the model and the windows


def precision_off() -> dict:
    """Turn TF32 off where the global switches reach (cheri_dls.py's "fp32" mode: matmul precision 'highest',
    cudnn.allow_tf32 False), and return them as set; a getter this torch lacks is recorded as its error."""
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    flags: dict = {"dtype": str(DTYPE)}
    getters = {
        "torch.get_float32_matmul_precision()": torch.get_float32_matmul_precision,
        "torch.backends.cudnn.allow_tf32": lambda: torch.backends.cudnn.allow_tf32,
    }
    for name, get in getters.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                flags[name] = get()
            except Exception as exc:  # recorded, not fatal: the record is for the report
                flags[name] = f"<{type(exc).__name__}: {exc}>"
    return flags


def pick_device(requested: str) -> tuple[torch.device, str | None]:
    """The device to run on, and a note when it is not the one asked for."""
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu"), f"{requested} asked for, but no GPU is visible: ran on the CPU"
    return torch.device(requested), None


def load_model(pt: Path, device: torch.device):
    model = torch.jit.load(str(pt), map_location=device)
    model.eval()
    return model.to(dtype=DTYPE)


def one_hot(codes: np.ndarray, device: torch.device) -> torch.Tensor:
    """(B, L) codes 0-3 as a (B, 4, L) one-hot in DTYPE."""
    x = torch.from_numpy(codes.astype(np.int64)).to(device)
    return torch.nn.functional.one_hot(x, 4).permute(0, 2, 1).to(DTYPE).contiguous()


def candidate_windows(negatives: Path, chroms: list[str] | None, lengths: dict[str, int],
                      seed: int) -> tuple[list[tuple[str, int]], dict]:  # fmt: skip
    """(chrom, start) of every WINDOW-bp window centred on a negative on the chromosomes, in a random order."""
    rows, off_chroms, off_ends = [], 0, 0
    for n_line, line in enumerate(Path(negatives).read_text().splitlines(), 1):
        fields = line.split()
        if not fields or fields[0].startswith(("#", "track", "browser")):
            continue
        chrom = fields[0]
        if (chroms is not None and chrom not in chroms) or chrom not in lengths:
            off_chroms += 1
            continue
        try:
            start, end = int(fields[1]), int(fields[2])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"{negatives}, line {n_line}: not a BED line: {line!r}") from exc
        s = start + (end - start) // 2 - WINDOW // 2
        if s < 0 or s + WINDOW > lengths[chrom]:
            off_ends += 1
            continue
        rows.append((chrom, s))
    if not rows:
        raise ValueError(f"{negatives} has no window on {', '.join(chroms) if chroms else 'the h5 and FASTA'}")
    order = np.random.default_rng(seed).permutation(len(rows))
    return [rows[i] for i in order.tolist()], {
        "negatives_used": len(rows),
        "off_chroms": off_chroms,
        "too_close_to_an_end": off_ends,
    }


def take_windows(fasta: Path, candidates: list[tuple[str, int]], n: int) -> tuple[list[tuple[str, int]], np.ndarray,
                                                                                  int]:  # fmt: skip
    """The first n candidates whose sequence is A, C, G and T only: (windows, (n, WINDOW) codes, rejected)."""
    import pyfaidx

    fa = pyfaidx.Fasta(str(fasta), as_raw=True, sequence_always_upper=True, rebuild=False)
    windows, codes, rejected = [], [], 0
    try:
        for chrom, s in candidates:
            if len(windows) == n:
                break
            seq = fa[chrom][s : s + WINDOW]
            c = _CODES[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]
            if c.size != WINDOW or (c == 255).any():
                rejected += 1
                continue
            windows.append((chrom, s))
            codes.append(c)
    finally:
        fa.close()
    if not windows:
        raise ValueError(f"no window of A, C, G and T only among {len(candidates)} candidates")
    return windows, np.stack(codes), rejected


def h5_targets(h5_path: Path, windows: list[tuple[str, int]], margin: int) -> np.ndarray:
    """log10(h5 + 0.01) over [s + FLANK - margin, s + WINDOW - FLANK + margin) of each window, float64."""
    width = WINDOW - 2 * FLANK + 2 * margin
    out = np.empty((len(windows), width), dtype=np.float64)
    order = sorted(range(len(windows)), key=lambda i: windows[i])
    with h5py.File(h5_path, "r") as f:
        for i in order:
            chrom, s = windows[i]
            a = s + FLANK - margin
            out[i] = np.log10(f[chrom][a : a + width].astype(np.float64) + PSEUDO)
    return out


def predict(model, codes: np.ndarray, device: torch.device, batch: int) -> tuple[np.ndarray, np.ndarray]:
    """2x - 1 of the CNN on the windows, and of it on their reverse complements flipped back: (y, yr)."""
    ys, yrs = [], []
    with torch.inference_mode():
        for a in range(0, len(codes), batch):
            x = one_hot(codes[a : a + batch], device)
            y = model(x)
            yr = model(x.flip(1).flip(2)).flip(-1)
            ys.append((2 * y - 1).double().cpu().numpy())
            yrs.append((2 * yr - 1).double().cpu().numpy())
    return np.concatenate(ys), np.concatenate(yrs)


# ---------------------------------------------------------------------------------------------------------------------
# the checks


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    return float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)))


def registration(y: np.ndarray, target: np.ndarray, margin: int) -> dict:
    """y against the h5's log10(v + 0.01) shifted by k, for k in REG_OFFSETS (target has `margin` extra bases each
    side)."""
    width = y.shape[1]
    per = {}
    for k in REG_OFFSETS:
        t = target[:, margin + k : margin + k + width]
        diff = np.abs(y - t)
        per[str(k)] = {
            "max": float(diff.max()),
            "q99": float(np.quantile(diff, 0.99)),
            "median": float(np.median(diff)),
            "mean": float(diff.mean()),
            "pearson": pearson(y, t),
        }
    best = min(REG_OFFSETS, key=lambda k: (per[str(k)]["median"], abs(k)))
    at0 = per["0"]
    return {
        "per_offset": per,
        "best_offset": best,
        "tolerance": REG_TOL,
        "pass": at0["max"] <= REG_TOL,
        "pass_q99": at0["q99"] <= REG_TOL,
        "rule": f"max |2x - 1 - log10(h5[s + {FLANK} + k ...] + 0.01)| <= {REG_TOL} at k = 0; best k by median",
    }


def symmetry(y: np.ndarray, yr: np.ndarray) -> dict:
    """y[j] against yr[j + k], for k in SYM_OFFSETS."""
    width = y.shape[1]
    per = {}
    for k in SYM_OFFSETS:
        a = y[:, max(0, -k) : min(width, width - k)]
        b = yr[:, max(0, k) : min(width, width + k)]
        diff = np.abs(a - b)
        per[str(k)] = {
            "pearson": pearson(a, b),
            "median": float(np.median(diff)),
            "q99": float(np.quantile(diff, 0.99)),
        }
    best = max(SYM_OFFSETS, key=lambda k: (per[str(k)]["pearson"], -abs(k)))
    at0 = per["0"]
    return {
        "per_offset": per,
        "best_offset": best,
        "pass": at0["pearson"] >= SYM_R_MIN and at0["median"] <= SYM_MEDIAN_MAX,
        "rule": f"Pearson >= {SYM_R_MIN} and median |y - yr| <= {SYM_MEDIAN_MAX} at k = 0 (y[j] against yr[j + k]); "
        "best k by Pearson",
    }


# ---------------------------------------------------------------------------------------------------------------------
# a run


def run(pt: Path, h5_path: Path, fasta: Path, negatives: Path, *, n: int = 20000, chroms: list[str] | None = None,
        seed: int = 1234, device: str = "cuda", batch: int = 512) -> dict:  # fmt: skip
    """The three checks; returns the record --out holds."""
    import pyfaidx

    began = time.perf_counter()
    pt, h5_path = Path(os.path.abspath(pt)), Path(os.path.abspath(h5_path))
    fasta, negatives = Path(os.path.abspath(fasta)), Path(os.path.abspath(negatives))
    inputs = {
        role: stat_file(p) for role, p in (("pt", pt), ("h5", h5_path), ("fasta", fasta), ("negatives", negatives))
    }
    flags = precision_off()
    dev, device_note = pick_device(device)
    if device_note:
        logger.warning("%s", device_note)

    fa = pyfaidx.Fasta(str(fasta), as_raw=True, rebuild=False)
    fa_lengths = {name: len(fa[name]) for name in fa.keys()}
    fa.close()
    with h5py.File(h5_path, "r") as f:
        h5_len = {name: int(f[name].shape[0]) for name in f.keys()}
    lengths = {c: n_fa for c, n_fa in fa_lengths.items() if c in h5_len}
    wrong = {c: (lengths[c], h5_len[c]) for c in lengths if lengths[c] != h5_len[c]}
    if wrong:
        raise ValueError(f"the FASTA and the h5 disagree on chromosome lengths: {wrong}")
    if chroms:
        lacking = [c for c in chroms if c not in lengths]
        if lacking:
            raise ValueError(f"not in both the FASTA and the h5: {', '.join(lacking)}")

    t = time.perf_counter()
    candidates, window_record = candidate_windows(negatives, chroms, lengths, seed)
    windows, codes, rejected = take_windows(fasta, candidates, n)
    margin = max(abs(k) for k in REG_OFFSETS)
    target = h5_targets(h5_path, windows, margin)
    window_record.update(
        requested=n,
        used=len(windows),
        rejected_not_acgt=rejected,
        per_chrom={c: sum(1 for w in windows if w[0] == c) for c in sorted({w[0] for w in windows})},
        window=WINDOW,
        seed=seed,
    )
    read_s = time.perf_counter() - t
    logger.info("%d windows of %d bp (%d rejected for a base other than ACGT) in %.1f s", len(windows), WINDOW,
                rejected, read_s)  # fmt: skip

    t = time.perf_counter()
    model = load_model(pt, dev)
    y, yr = predict(model, codes, dev, batch)
    model_s = time.perf_counter() - t
    expected = WINDOW - 2 * FLANK
    length = {"expected": expected, "got": int(y.shape[1]), "pass": y.shape[1] == expected}
    record_checks: dict = {"output_length": length}
    if length["pass"]:
        record_checks["registration"] = registration(y, target, margin)
        record_checks["symmetry"] = symmetry(y, yr)
    else:
        logger.error("the CNN maps %d bases to %d, not %d: registration and symmetry not checked", WINDOW,
                     y.shape[1], expected)  # fmt: skip
        record_checks["registration"] = record_checks["symmetry"] = {"pass": False, "skipped": "output length"}
    passed = {name: bool(check["pass"]) for name, check in record_checks.items()}

    now = datetime.now().astimezone()
    return {
        "format": FORMAT,
        "tool": "tn5_cnn_check.py",
        "written": now.isoformat(timespec="seconds"),
        "written_et": now.astimezone(ET).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "argv": list(sys.argv),
        "code": code_identity(),
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "numpy": np.__version__,
            "h5py": h5py.__version__,
            "pyfaidx": getattr(pyfaidx, "__version__", None),
        },
        "inputs": inputs,
        "device": str(dev),
        "device_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
        "device_note": device_note,
        "precision": flags,
        "parameters": {"n": n, "chroms": chroms, "seed": seed, "batch": batch, "window": WINDOW, "flank": FLANK},
        "windows": window_record,
        **record_checks,
        "informational": "not a gate: the box goes on whatever these say",
        "pass": passed,
        "all_pass": all(passed.values()),
        "timings_s": {"windows": read_s, "model": model_s, "total": time.perf_counter() - began},
        "rss_max_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }


def summary(record: dict, out: Path) -> str:
    reg, sym = record["registration"], record["symmetry"]
    if "per_offset" not in reg:
        return f"tn5_cnn_check: output length {record['output_length']['got']}, not {WINDOW - 2 * FLANK}; {out}"
    r0, s0 = reg["per_offset"]["0"], sym["per_offset"]["0"]
    return (
        f"tn5_cnn_check: {record['windows']['used']} windows on {record['device']}; registration "
        f"{'pass' if reg['pass'] else 'FAIL'} (max {r0['max']:.3g}, q99 {r0['q99']:.3g}, best k "
        f"{reg['best_offset']:+d}); symmetry {'pass' if sym['pass'] else 'FAIL'} (r {s0['pearson']:.4f}, median "
        f"{s0['median']:.4f}, best k {sym['best_offset']:+d}); {out}"
    )


# ---------------------------------------------------------------------------------------------------------------------
# selftest


class _TinyCNN(torch.nn.Module):
    """The CNN's contract with random weights: (B, 4, L) in, (B, L - 100) out, receptive field 101. The output is
    kept within (0.25, 0.75), where v + 0.01 is far from 0 and float32 rounds 2x - 1 to ~1e-7."""

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(4, 8, 51)
        self.conv2 = torch.nn.Conv1d(8, 1, 51)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv1(x))
        return 0.25 + 0.5 * torch.sigmoid(self.conv2(h))[:, 0, :]


class _Symmetrized(torch.nn.Module):
    """A model averaged with its own reverse complement: symmetric at offset 0 by construction."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (self.model(x) + self.model(x.flip(1).flip(2)).flip(-1))


def selftest() -> int:
    """A tiny random CNN, traced and saved as the real one is, over a synthetic genome and the h5 it makes: the
    registration must pass at 0 and find a planted 1-bp shift, a symmetrized CNN must pass the symmetry check and
    the raw one fail it."""
    import pyfaidx

    torch.manual_seed(5)
    rng = np.random.default_rng(5)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    failures = []

    def check(cond: bool, what: str) -> None:
        if not cond:
            failures.append(what)
            logger.error("selftest: FAILED %s", what)

    lengths = {"chrA": 40_000, "chrB": 30_000}
    genome = {c: np.frombuffer(b"ACGT", dtype=np.uint8)[rng.integers(0, 4, size=n)].copy() for c, n in lengths.items()}
    genome["chrB"][10_000:12_000] = ord("N")
    genome["chrA"][5_000:6_000] += ord("a") - ord("A")  # soft-masked: must read as what it is

    with tempfile.TemporaryDirectory(prefix="tn5_cnn_check_selftest.") as tmp:
        tmp = Path(tmp)
        fasta = tmp / "genome.fa"
        with open(fasta, "w") as fh:
            for chrom, seq in genome.items():
                text = seq.tobytes().decode("ascii")
                fh.write(f">{chrom}\n" + "".join(text[i : i + 60] + "\n" for i in range(0, len(text), 60)))
        pyfaidx.Fasta(str(fasta)).close()  # writes the .fai

        sym = _Symmetrized(_TinyCNN()).eval().double()
        raw = sym.model
        example = torch.zeros(1, 4, 300, dtype=torch.float64)
        for name, module in (("sym", sym), ("raw", raw)):
            torch.jit.trace(module, example).save(str(tmp / f"{name}.pt"))

        # The h5 the symmetrized CNN makes: its output at base p is h5[p]; the 50 bases at each end hold an N value.
        h5_path, shifted = tmp / "bias.h5", tmp / "shifted.h5"
        with torch.inference_mode(), h5py.File(h5_path, "w") as f, h5py.File(shifted, "w") as g:
            for chrom, seq in genome.items():
                codes = _CODES[np.frombuffer(seq.tobytes().upper(), dtype=np.uint8)]
                x = torch.nn.functional.one_hot(torch.from_numpy(np.minimum(codes, 4).astype(np.int64)), 5)
                x = x[:, :4].T[None].to(torch.float64)  # N: all zeros
                out = sym(x)[0].numpy()
                v = np.full(seq.size, 0.251753, dtype=np.float32)
                v[FLANK : seq.size - FLANK] = (10.0 ** ((out - 0.5) * 2) - PSEUDO).astype(np.float32)
                f.create_dataset(chrom, data=v)
                g.create_dataset(chrom, data=np.concatenate([v[:1], v[:-1]]))  # h5'[p + 1] = h5[p]

        neg = tmp / "negatives.bed"
        lines = []
        for chrom, n in lengths.items():
            for s in rng.integers(0, n - 2114, size=40).tolist():
                lines.append(f"{chrom}\t{s}\t{s + 2114}\t.\t.\t.\t.\t.\t.\t1057\n")
        lines.append(f"chrB\t{10_500 - 1057}\t{10_500 + 1057}\t.\t.\t.\t.\t.\t.\t1057\n")  # inside the N run
        neg.write_text("".join(lines))
        common = {"n": 30, "seed": 2, "device": device, "batch": 8}

        rec = run(tmp / "sym.pt", h5_path, fasta, neg, **common)
        write_json(tmp / "cnn.json", rec)
        logger.info("selftest symmetrized: %s", summary(rec, tmp / "cnn.json"))
        check(rec["output_length"]["pass"] and rec["output_length"]["got"] == 1000, "the output is L - 100 long")
        check(rec["windows"]["used"] == 30, "30 windows used")
        check(rec["registration"]["pass"] and rec["registration"]["best_offset"] == 0, "registration passes at 0")
        check(rec["symmetry"]["pass"] and rec["symmetry"]["best_offset"] == 0, "the symmetrized CNN is symmetric")
        check(rec["all_pass"], "every check passes for the symmetrized CNN on its own h5")

        rec = run(tmp / "sym.pt", shifted, fasta, neg, **common)
        check(rec["registration"]["best_offset"] == 1, "a 1-bp shift of the h5 is found at +1")
        check(not rec["registration"]["pass"], "and fails the registration at 0")

        rec = run(tmp / "raw.pt", h5_path, fasta, neg, **common)
        check(not rec["symmetry"]["pass"], "a random CNN is not symmetric")

        n_rej = run(tmp / "sym.pt", h5_path, fasta, neg, n=10_000, seed=2, device=device, batch=64)["windows"]
        check(n_rej["rejected_not_acgt"] >= 1, "a window over the N run is rejected")
        check(n_rej["used"] + n_rej["rejected_not_acgt"] == n_rej["negatives_used"], "every candidate is accounted for")

    if failures:
        print(f"tn5_cnn_check selftest: FAILED {len(failures)} check(s): {'; '.join(failures)}")
        return 1
    print(f"tn5_cnn_check selftest: all checks passed (on {device})")
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# command line


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pt", type=Path, help="the bias CNN (TorchScript)")
    ap.add_argument("--h5", type=Path, help="the bias h5 it made: one float32 dataset per contig")
    ap.add_argument("--fasta", type=Path, help="the genome (indexed)")
    ap.add_argument("--negatives", type=Path, help="background windows (BED); each centres one window")
    ap.add_argument("--n", type=int, default=20000, help="windows (20000)")
    ap.add_argument("--chroms", nargs="+", help="only these chromosomes (default: all of the negatives')")
    ap.add_argument("--seed", type=int, default=1234, help="for the windows (1234)")
    ap.add_argument("--device", default="cuda", help="cuda (the default, if a GPU is visible) or cpu")
    ap.add_argument("--batch", type=int, default=512, help="windows per forward pass (512)")
    ap.add_argument("--out", type=Path, help="cnn.json")
    ap.add_argument("--selftest", action="store_true", help="a tiny synthetic run in a temporary directory")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%F %T")  # fmt: skip
    if args.selftest:
        return selftest()
    missing = [f"--{k}" for k in ("pt", "h5", "fasta", "negatives", "out") if getattr(args, k) is None]
    if missing:
        ap.error(f"{', '.join(missing)} required (or --selftest)")
    if args.n < 1 or args.batch < 1:
        ap.error("--n and --batch must be positive")
    try:
        record = run(args.pt, args.h5, args.fasta, args.negatives, n=args.n, chroms=args.chroms, seed=args.seed,
                     device=args.device, batch=args.batch)  # fmt: skip
        write_json(Path(os.path.abspath(args.out)), record)
    except (ValueError, OSError, RuntimeError) as exc:
        logger.error("no CNN check: %s", exc)
        return 1
    print(summary(record, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
