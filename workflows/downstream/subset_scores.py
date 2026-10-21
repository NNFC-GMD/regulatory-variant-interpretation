#!/usr/bin/env python3
"""One Cherimoya fold's attribution on ChromBPNet's 30K-peak QC subsample of the same fold, for TF-MoDISco at 04.5's
scale.

igvf_tf_chrombpnet's per-fold motif QC attributes a 30,000-peak subsample of each fold (04.4's
auxiliary/30K_subsample_peaks.bed: pandas' sample(30000, random_state=1234) of the fold's filtered.peaks.bed, so each
fold has its own) and runs TF-MoDISco on it with a budget of 5,000 seqlets (04.5, src/motif_qc.py). For the same
comparison fold by fold, this takes the same peaks, in the same order, from Cherimoya's attribution of the fold (the
ohe and attr npz files its attribute.json names) and lays them out as ChromBPNet's QC scores are, for motif_qc.py:

    <output_dir>/full_models/<dataset>_<peak_type>_fold_<k>/auxiliary/interpret_subsample/
        cherimoya.<head>_scores.h5         /raw/seq int8, /shap/seq float16, /projected_shap/seq float16, each
                                           (M, 4, W); gzip level 4 in chunks of 16 rows. Cast as to_chrombpnet_scores.py
                                           casts (projected = ohe x attr in float32, then float16), so a row is bit for
                                           bit the converted fold's row of the same peak.
        cherimoya.interpreted_regions.bed  the subsample's lines whose peaks Cherimoya kept, byte for byte and in the
                                           subsample's order: h5 row j is line j
        cherimoya.provenance.json          the sources (paths, sizes, mtimes), the peaks dropped, the check against
                                           ChromBPNet, float16 statistics, the outputs, the code and the timings

output_dir and peak_type come from the dataset's Cherimoya config (make_downstream.py writes it to
<root>/configs/<dataset>/config.yaml; --config names another), the head from attribute.json's `output`; --out-dir
names the directory instead. The subsample is the one in the ChromBPNet fold that the fold's compare.json names (the
auxiliary/ directory beside its evaluation/ one), or --subsample.

The join is by peak name (column 4), never by row. pandas rewrote the subsample from ChromBPNet's filtered peaks, so
for each name the chrom, start, end and summit (columns 1-3 and 10) must equal, as integers, those of the peaks-file
line the fold's idx mask refers to; and the window centres must agree (Cherimoya centres on start + (end - start) // 2,
ChromBPNet on start + summit). A name the peaks file lacks, a repeated name or a disagreement is refused. A subsample
peak the idx mask dropped (its window holds a base other than A, C, G or T) is left out and listed.

Checked against ChromBPNet when its QC scores of the fold are there (auxiliary/interpret_subsample/
chrombpnet_nobias.<head>_scores.h5 and chrombpnet_nobias.interpreted_regions.bed beside the subsample): on every row
kept here, the ohe equals ChromBPNet's raw/seq[:, :, c0:c0 + W] (c0 = L/2 - W/2, 557 at L = 2114 and W = 1000), and
every peak dropped here has a column of zeros in ChromBPNet's L-bp raw/seq. A failure refuses the subset. Every ohe row
must be one-hot, and the attributions finite and within float16's range.

Atomic and idempotent, as the converter is: each output is written to <name>.partial and renamed, the provenance last,
after the sources have been stat'ed again and found unchanged. When the provenance matches the current sources
(attribute.json, the peaks file, the ohe, attr and idx files, the meta or stock record, the subsample and ChromBPNet's
QC files) and the outputs it records, nothing is done; --force writes them again.

    python workflows/downstream/subset_scores.py --attribute-json <root>/<dataset>/fold_<k>/attribute.json
        [--config <config.yaml> | --out-dir <dir>] [--subsample <30K_subsample_peaks.bed>] [--pin <pin>] [--force]

Exit 0 when written or up to date; 3 when the fold is not DONE by attr_status.py's rule (not attributed yet, or being
attributed again), so a caller waits; 1 when refused or failed. Run it in the cherimoya environment, inside a box: it
reads each npz once, in blocks, and holds the M rows, about 2 GB of RAM for 30,000 rows at W = 1000.
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
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "attribution"), str(HERE.parent / "train")]

import attr_status  # noqa: E402
import to_chrombpnet_scores as tcs  # noqa: E402

logger = logging.getLogger("subset_scores")

FORMAT = 1
"""The layout and provenance this writes; a provenance of another format is written again."""
PREFIX = "cherimoya"
SUBSAMPLE_NAME = "30K_subsample_peaks.bed"
CHROMBPNET_PREFIX = "chrombpnet_nobias"
"""04.4's file prefix: the bias-free model's QC scores."""
GZIP_LEVEL = 4
NOT_READY_EXIT = 3
CODE_FILES = ("subset_scores.py", "to_chrombpnet_scores.py", "attr_status.py")
CODE_DIRS = {"attr_status.py": HERE.parent / "attribution"}


class NotReady(Exception):
    """The fold's attribution is not DONE: nothing to take yet."""


# ---------------------------------------------------------------------------------------------------------------------
# where things are


def chrombpnet_fold(fold_dir: Path) -> Path:
    """ChromBPNet's directory of the same dataset and fold: the parent of the evaluation/ directory compare.json names
    (as to_chrombpnet_scores.chrombpnet_interpretation finds it)."""
    try:
        compare = json.loads((Path(fold_dir) / "compare.json").read_text())
        return Path(compare["chrombpnet_predictions"]).parent.parent
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise tcs.Refused(f"cannot find ChromBPNet's fold from {Path(fold_dir) / 'compare.json'}: {exc}") from exc


def locate(params: dict, attribute_json: Path, config: Path | None, out_dir: Path | None,
           subsample: Path | None) -> dict:  # fmt: skip
    """The fold, its outputs, the subsample and ChromBPNet's QC files beside it."""
    where = tcs.resolve(params, attribute_json, config, None)
    head = where["head"]
    out = Path(os.path.abspath(out_dir)) if out_dir else where["out_dir"].parent / "auxiliary" / "interpret_subsample"
    if subsample is None:
        subsample = chrombpnet_fold(where["fold_dir"]) / "auxiliary" / SUBSAMPLE_NAME
    subsample = Path(os.path.abspath(subsample))
    cbp_dir = subsample.parent / "interpret_subsample"
    return {
        **where,
        "subset_dir": out,
        "h5": out / f"{PREFIX}.{head}_scores.h5",
        "bed": out / f"{PREFIX}.interpreted_regions.bed",
        "provenance": out / f"{PREFIX}.provenance.json",
        "subsample": subsample,
        "chrombpnet_h5": cbp_dir / f"{CHROMBPNET_PREFIX}.{head}_scores.h5",
        "chrombpnet_bed": cbp_dir / f"{CHROMBPNET_PREFIX}.interpreted_regions.bed",
    }


def up_to_date(where: dict, sources: dict) -> str | None:
    """Why the subset must be written (again), or None when its provenance describes these sources and outputs."""
    try:
        prov = json.loads(where["provenance"].read_text())
    except FileNotFoundError:
        return "not written yet"
    except (OSError, ValueError) as exc:
        return f"unreadable {where['provenance']}: {exc}"
    if prov.get("format") != FORMAT:
        return f"provenance format {prov.get('format')!r}, this writes {FORMAT}"
    recorded = prov.get("sources") or {}
    if recorded != sources:
        changed = sorted(role for role in set(sources) | set(recorded) if recorded.get(role) != sources.get(role))
        return f"sources changed since it was written: {', '.join(changed)}"
    for name in ("h5", "bed"):
        written = {k: ((prov.get("outputs") or {}).get(name) or {}).get(k) for k in ("path", "size", "mtime_ns")}
        try:
            current = tcs.stat_output(where[name])
        except FileNotFoundError:
            return f"no {where[name]}"
        if written != current:
            return f"{where[name]} is not the file the provenance records"
    return None


# ---------------------------------------------------------------------------------------------------------------------
# the rows


def coordinates(line: bytes, path: Path, row: int) -> tuple[str, int, int, int]:
    """(chrom, start, end, summit) of a narrowPeak line, as integers."""
    fields = tcs.bed_fields(line)
    try:
        return fields[0], int(fields[1]), int(fields[2]), int(fields[9])
    except (IndexError, ValueError) as exc:
        raise tcs.Refused(f"{path}: line {row + 1} is not a 10-column narrowPeak line ({exc})") from exc


def join(sub_lines: list[bytes], peak_lines: list[bytes], idx: np.ndarray, subsample: Path, peaks: Path) -> dict:
    """Each subsample line's peaks-file row, joined by name and checked (see the module docstring); which lines are
    kept and, for those, the Cherimoya row (peaks-file row flatnonzero(idx)[r] is Cherimoya row r)."""
    row_of: dict[str, int] = {}
    for i, line in enumerate(peak_lines):
        name = tcs.peak_name(line)
        if name in row_of:
            raise tcs.Refused(f"{peaks}: peak name {name} repeats (rows {row_of[name] + 1} and {i + 1}); the join "
                              "needs unique names")  # fmt: skip
        row_of[name] = i
    rank = np.cumsum(idx, dtype=np.int64) - 1
    seen: set[str] = set()
    kept_lines: list[int] = []
    cheri_rows: list[int] = []
    dropped: list[dict] = []
    for j, line in enumerate(sub_lines):
        name = tcs.peak_name(line)
        if name in seen:
            raise tcs.Refused(f"{subsample}: peak name {name} repeats (line {j + 1})")
        seen.add(name)
        i = row_of.get(name)
        if i is None:
            raise tcs.Refused(f"{subsample}: {name} (line {j + 1}) is not in {peaks}; the subsample is of other peaks")
        theirs, mine = coordinates(line, subsample, j), coordinates(peak_lines[i], peaks, i)
        if theirs != mine:
            raise tcs.Refused(f"{name}: chrom, start, end, summit {theirs} in {subsample}, {mine} in {peaks}")
        _chrom, start, end, summit = mine
        if (end - start) // 2 != summit:
            raise tcs.Refused(
                f"{name}: Cherimoya's window centre (start + (end - start) // 2 = "
                f"{start + (end - start) // 2}) is not ChromBPNet's (start + summit = {start + summit})"
            )
        if idx[i]:
            kept_lines.append(j)
            cheri_rows.append(int(rank[i]))
        else:
            dropped.append({"name": name, "subsample_line": j + 1, "peaks_row": i + 1})
    return {
        "kept_lines": np.asarray(kept_lines, dtype=np.int64),
        "cheri_rows": np.asarray(cheri_rows, dtype=np.int64),
        "dropped": dropped,
    }


def take_rows(path: Path, rows: np.ndarray, block_rows: int) -> np.ndarray:
    """arr_0[rows] of an npz, in the order of `rows` (distinct), read in one pass."""
    order = np.argsort(rows, kind="stable")
    block = tcs.take_npz_rows(path, rows[order], block_rows)
    out = np.empty_like(block)
    out[order] = block
    return out


def check_against_chrombpnet(where: dict, sub_lines: list[bytes], joined: dict, ohe: np.ndarray, width: int,
                             in_window: int) -> dict:  # fmt: skip
    """On every kept row, the ohe is ChromBPNet's QC raw/seq[:, :, c0:c0 + W]; every dropped peak has a column of
    zeros in ChromBPNet's raw/seq. Returns the check's record (its failures refuse the subset)."""
    out: dict = {"h5": str(where["chrombpnet_h5"]), "bed": str(where["chrombpnet_bed"])}
    cbp_lines = tcs.read_peak_lines(where["chrombpnet_bed"])
    cbp_row = {tcs.peak_name(line): r for r, line in enumerate(cbp_lines)}
    if len(cbp_row) != len(cbp_lines):
        raise tcs.Refused(f"{where['chrombpnet_bed']}: repeated peak names; the join needs unique names")
    names = [tcs.peak_name(sub_lines[j]) for j in joined["kept_lines"]]
    present = np.array([name in cbp_row for name in names], dtype=bool)
    theirs_rows = np.array([cbp_row[name] for name, here in zip(names, present) if here], dtype=np.int64)
    dropped_rows = np.array([cbp_row[d["name"]] for d in joined["dropped"] if d["name"] in cbp_row], dtype=np.int64)
    with h5py.File(where["chrombpnet_h5"], "r") as fh:
        raw = fh["raw/seq"]
        if raw.ndim != 3 or raw.shape[:2] != (len(cbp_lines), 4):
            raise tcs.Refused(f"{where['chrombpnet_h5']}: raw/seq is {raw.shape}, not ({len(cbp_lines)}, 4, L)")
        length = raw.shape[2]
        if length != in_window:
            raise tcs.Refused(f"{where['chrombpnet_h5']}: raw/seq is {length} bp wide, the attribution's in_window "
                              f"{in_window}")  # fmt: skip
        c0 = length // 2 - width // 2
        mismatch = 0
        if theirs_rows.size:
            order = np.argsort(theirs_rows, kind="stable")
            theirs = np.empty((theirs_rows.size, 4, width), dtype=raw.dtype)
            theirs[order] = tcs.take_h5_rows(raw, theirs_rows[order])[:, :, c0 : c0 + width]
            mismatch = int((theirs != ohe[present]).any(axis=(1, 2)).sum())
        zero = (
            tcs.zero_column_rows(tcs.take_h5_rows(raw, np.sort(dropped_rows)))
            if dropped_rows.size
            else np.zeros(0, dtype=bool)
        )
    out.update(
        window_columns=[c0, c0 + width],
        rows_compared=int(theirs_rows.size),
        kept_not_in_chrombpnet=[name for name, here in zip(names, present) if not here][:20],
        ohe_mismatch_rows=mismatch,
        dropped_in_chrombpnet=int(dropped_rows.size),
        dropped_with_zero_column=int(zero.sum()),
    )
    failures = []
    if mismatch:
        failures.append(f"{mismatch} of {theirs_rows.size} rows: the ohe is not ChromBPNet's QC raw/seq[:, :, "
                        f"{c0}:{c0 + width}]")  # fmt: skip
    if not zero.all():
        failures.append(f"{int((~zero).sum())} peaks dropped here have no column of zeros in ChromBPNet's raw/seq")
    out["failures"] = failures
    out["pass"] = not failures
    return out


# ---------------------------------------------------------------------------------------------------------------------
# writing


def write_h5(path: Path, ohe: np.ndarray, shap: np.ndarray, projected: np.ndarray) -> None:
    n, _, width = ohe.shape
    chunks = (max(1, min(n, tcs.CHUNK_ROWS)), 4, width)
    with h5py.File(path, "w") as fh:
        for key, data in (("raw", ohe), ("shap", shap), ("projected_shap", projected)):
            fh.create_dataset(f"{key}/seq", data=data, chunks=chunks, compression="gzip", compression_opts=GZIP_LEVEL)


def code_identity() -> dict:
    hashes = {}
    for name in CODE_FILES:
        try:
            hashes[name] = hashlib.sha256((CODE_DIRS.get(name, HERE) / name).read_bytes()).hexdigest()
        except OSError:
            hashes[name] = None

    def git(*args: str) -> str | None:
        try:
            run = subprocess.run(["git", "-C", str(HERE), *args], capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return None
        return run.stdout.strip() if run.returncode == 0 else None

    dirty = git("status", "--porcelain", "--", *(str(CODE_DIRS.get(n, HERE) / n) for n in CODE_FILES))
    return {
        "path": str(HERE / "subset_scores.py"),
        "files_sha256": hashes,
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": None if dirty is None else bool(dirty),
        "versions": {"numpy": np.__version__, "h5py": h5py.__version__, "hdf5": h5py.version.hdf5_version},
    }


def subset(
    attribute_json: Path,
    *,
    config: Path | None = None,
    out_dir: Path | None = None,
    subsample: Path | None = None,
    pin: Path | None = None,
    force: bool = False,
    block_rows: int = tcs.BLOCK_ROWS,
) -> tuple[str, dict]:
    """Write one fold's subset. Returns ("written", provenance) or ("up to date", {"reason": ...}); raises NotReady
    when the fold is not DONE and tcs.Refused when it cannot be written."""
    began = time.perf_counter()
    attribute_json = Path(os.path.abspath(attribute_json))
    try:
        params = json.loads(attribute_json.read_text())
    except (OSError, ValueError) as exc:
        raise tcs.Refused(f"cannot read {attribute_json}: {exc}") from exc
    where = locate(params, attribute_json, config, out_dir, subsample)
    state, reasons = attr_status.check_fold(where["fold_dir"], pin)
    if state != "done":
        raise NotReady(f"{where['fold_dir']} is {state}, not DONE: {'; '.join(reasons)}")
    if not where["subsample"].is_file():
        raise tcs.Refused(f"no subsample {where['subsample']} (ChromBPNet's 04.4 of the fold)")
    if len(params["loci"]) != 1:
        raise tcs.Refused(f"{attribute_json} names {len(params['loci'])} peaks files; the join needs exactly one")
    files = attr_status.fold_files(params)
    source_paths = {**tcs.fold_sources(params, attribute_json), "subsample": where["subsample"]}
    against = where["chrombpnet_h5"].is_file() and where["chrombpnet_bed"].is_file()
    if against:
        source_paths.update(chrombpnet_h5=where["chrombpnet_h5"], chrombpnet_bed=where["chrombpnet_bed"])
    sources = tcs.stat_sources(source_paths)
    if not force:
        reason = up_to_date(where, sources)
        if reason is None:
            return "up to date", {"reason": f"{where['provenance']} matches the sources and outputs"}
        logger.info("%s fold %s: writing the subset (%s)", where["dataset"], where["fold"], reason)

    width, in_window = params["attr_window"], params["in_window"]
    if not isinstance(width, int) or width <= 0 or width % 2:
        raise tcs.Refused(f"attr_window {width!r} is not a positive even number of bases")
    idx = np.load(files["idx"], allow_pickle=False)
    if idx.dtype != np.bool_ or idx.ndim != 1:
        raise tcs.Refused(f"{files['idx']} is {idx.dtype} {idx.shape}, not a 1-d boolean mask")
    peak_lines = tcs.read_peak_lines(source_paths["peaks"])
    if len(peak_lines) != idx.size:
        raise tcs.Refused(f"{source_paths['peaks']} has {len(peak_lines)} rows, the idx mask {idx.size} entries")
    n_kept = int(idx.sum())
    for path, descr in ((files["ohe"], "|i1"), (files["attr"], "<f4")):
        header = attr_status.npz_header(path)
        if header["descr"] != descr or tuple(header["shape"]) != (n_kept, 4, width) or header["fortran_order"]:
            raise tcs.Refused(f"{path} holds {header['descr']} {tuple(header['shape'])}, not {descr} "
                              f"{(n_kept, 4, width)} in C order")  # fmt: skip
    sub_lines = tcs.read_peak_lines(where["subsample"])
    joined = join(sub_lines, peak_lines, idx, where["subsample"], source_paths["peaks"])
    del peak_lines
    rows = joined["cheri_rows"]
    if rows.size == 0:
        raise tcs.Refused(f"none of the {len(sub_lines)} subsample peaks is among the fold's kept rows")

    t = time.perf_counter()
    ohe = take_rows(files["ohe"], rows, block_rows)
    attr = take_rows(files["attr"], rows, block_rows)
    read_s = time.perf_counter() - t
    tcs.check_block(ohe, attr, 0)
    chrombpnet = (
        check_against_chrombpnet(where, sub_lines, joined, ohe, width, in_window)
        if against
        else {"skipped": f"no {where['chrombpnet_h5']} and {where['chrombpnet_bed']}"}
    )
    if chrombpnet.get("failures"):
        raise tcs.Refused("against ChromBPNet's QC scores: " + "; ".join(chrombpnet["failures"]))
    stats = {"shap": tcs.Float16Stats(), "projected_shap": tcs.Float16Stats()}
    shap = attr.astype(np.float16)
    stats["shap"].update(attr, shap)
    projected32 = ohe.astype(np.float32) * attr
    del attr
    projected = projected32.astype(np.float16)
    stats["projected_shap"].update(projected32, projected)
    del projected32
    bed_bytes = b"".join(sub_lines[j] for j in joined["kept_lines"])

    targets = [where["h5"], where["bed"], where["provenance"]]
    where["subset_dir"].mkdir(parents=True, exist_ok=True)
    tcs.remove(where["provenance"])  # from here on, no provenance describes outputs that are being replaced
    try:
        tcs.partial(where["bed"]).write_bytes(bed_bytes)
        write_h5(tcs.partial(where["h5"]), ohe, shap, projected)
        changed = [role for role, st in tcs.stat_sources(source_paths).items() if st != sources[role]]
        if changed:
            raise tcs.Refused(f"{', '.join(changed)} changed while the subset was taken; run it again")
        os.replace(tcs.partial(where["h5"]), where["h5"])
        os.replace(tcs.partial(where["bed"]), where["bed"])
    except BaseException:
        for path in targets:
            tcs.remove(tcs.partial(path))
        raise

    now = datetime.now().astimezone()
    record_kind, record_file = tcs.record_path(files)
    provenance = {
        "format": FORMAT,
        "tool": "subset_scores.py",
        "written": now.isoformat(timespec="seconds"),
        "written_et": now.astimezone(ZoneInfo("America/New_York")).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "argv": list(sys.argv),
        "code": code_identity(),
        "dataset": where["dataset"],
        "fold": where["fold"],
        "head": where["head"],
        "config": where["config"]["config"] if where["config"] else None,
        "pin": str(pin if pin is not None else attr_status.default_pin(where["fold_dir"])),
        "sources": sources,
        "attribution": {
            "record": record_kind,
            "record_path": str(record_file),
            "random_state": params["random_state"],
            "attr_window": width,
            "in_window": in_window,
        },
        "rows": {
            "subsample_lines": len(sub_lines),
            "kept": int(rows.size),
            "dropped": joined["dropped"],
            "cheri_rows_sha256": hashlib.sha256(rows.tobytes()).hexdigest(),
            "cheri_rows_sha256_of": "the fold's attribution row of each h5 row, int64, in h5 row order",
        },
        "chrombpnet_check": chrombpnet,
        "outputs": {
            "h5": {**tcs.stat_output(where["h5"]), "datasets": tcs.describe_h5(where["h5"])},
            "bed": {
                **tcs.stat_output(where["bed"]),
                "sha256": hashlib.sha256(bed_bytes).hexdigest(),
                "rows": int(rows.size),
            },  # fmt: skip
        },
        "float16": {key: s.record() for key, s in stats.items()},
        "timings_s": {"read": read_s, "total": time.perf_counter() - began},
        "rss_max_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
    try:
        tcs.write_json(where["provenance"], provenance)
    except BaseException:
        tcs.remove(tcs.partial(where["provenance"]))
        raise
    return "written", provenance


# ---------------------------------------------------------------------------------------------------------------------
# command line


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attribute-json", type=Path, required=True,
                    help="the fold: <root>/<dataset>/fold_<k>/attribute.json")  # fmt: skip
    ap.add_argument("--config", type=Path, help="the Cherimoya config.yaml (default <root>/configs/<dataset>/)")
    ap.add_argument("--out-dir", type=Path, help="where the subset goes (default: from the config, see above)")
    ap.add_argument("--subsample", type=Path, help="ChromBPNet's 30K_subsample_peaks.bed (default: from compare.json)")
    ap.add_argument("--pin", type=Path, help=f"the engine pin (default <root>/{attr_status.PIN_NAME})")
    ap.add_argument("--force", action="store_true", help="write even when the provenance says up to date")
    ap.add_argument("--block-rows", type=int, default=tcs.BLOCK_ROWS, help=f"npz rows per block ({tcs.BLOCK_ROWS})")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%F %T")  # fmt: skip
    if args.block_rows <= 0:
        ap.error("--block-rows must be positive")
    fold_dir = Path(os.path.abspath(args.attribute_json)).parent
    try:
        status, info = subset(args.attribute_json, config=args.config, out_dir=args.out_dir, subsample=args.subsample,
                              pin=args.pin, force=args.force, block_rows=args.block_rows)  # fmt: skip
    except NotReady as exc:
        logger.warning("%s: not ready: %s", fold_dir, exc)
        return NOT_READY_EXIT
    except tcs.Refused as exc:
        logger.error("%s: no subset: %s", fold_dir, exc)
        return 1
    if status == "up to date":
        logger.info("%s: up to date (%s)", fold_dir, info["reason"])
        return 0
    rows, check = info["rows"], info["chrombpnet_check"]
    logger.info(
        "%s fold %s: wrote %s and its bed (%d of %d subsample peaks; dropped %s) in %.1f s; against ChromBPNet's QC "
        "scores: %s. float16: shap underflow %.3g of non-zero values, max |attr| %.4g",
        info["dataset"],
        info["fold"],
        info["outputs"]["h5"]["path"],
        rows["kept"],
        rows["subsample_lines"],
        ", ".join(d["name"] for d in rows["dropped"][:5]) + (" ..." if len(rows["dropped"]) > 5 else "") or "none",
        info["timings_s"]["total"],
        check.get("skipped")
        or f"{check['rows_compared']} rows equal, {check['dropped_with_zero_column']} dropped "
        "peaks with a column of zeros",
        info["float16"]["shap"]["underflow_fraction"],
        info["float16"]["shap"]["max_abs"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
