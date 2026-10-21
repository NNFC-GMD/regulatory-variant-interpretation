#!/usr/bin/env python3
"""Convert one Cherimoya fold's attribution into ChromBPNet's 05.0 per-fold layout, for igvf_tf_chrombpnet's 06.0-11.0.

attribute_fast.py (run_task.sh's `attribute` stage) leaves three files per fold, named by the fold's attribute.json:
the ohe npz (arr_0, int8, (N, 4, W), one-hot, ACGT, channels first), the attr npz (arr_0, float32, (N, 4, W),
hypothetical scores over [summit - W/2, summit + W/2)) and the idx npy (bool, one entry per row of the peaks file,
True for the N rows attributed; windows holding a base other than A, C, G or T are dropped). igvf_tf_chrombpnet's 06.0
averages per-fold HDF5 files in ChromBPNet's 05.0 layout, and 07.0 and 10.0 read fold 0's interpreted_regions.bed.
This writes both, plus a provenance record, into the directory 06.0 reads:

    <output_dir>/full_models/<dataset>_<peak_type>_fold_<k>/interpretation/
        interpretation.<head>_scores.h5         /raw/seq int8, /shap/seq float16, /projected_shap/seq float16,
                                                each (N, 4, W); chunks (min(N, 16), 4, W); HDF5 filter 32001 (Blosc,
                                                blosclz level 9, byte shuffle), as chrombpnet's scores_io.py writes
                                                05.0's. shap is the attr cast to float16; projected_shap is ohe x attr
                                                computed in float32, then cast. There is no hyp_scores key: modisco
                                                reads that one first.
        interpretation.interpreted_regions.bed  the peaks-file lines where idx is True, byte for byte and in file
                                                order, so h5 row j is peaks-file row flatnonzero(idx)[j]
        interpretation.provenance.json          the sources (paths, sizes, mtimes), the idx sha256, the meta's audit
                                                and code blocks, random_state, W, the row counts and the dropped peaks,
                                                float16 statistics, the converter's commit and the time

output_dir, peak_type and the fold list come from the dataset's Cherimoya config.yaml (make_downstream.py writes it to
<root>/configs/<dataset>/config.yaml, where the fold is <root>/<dataset>/fold_<k>); the head is attribute.json's
`output`. --out-dir names the interpretation directory instead, and must still end in <dataset>_*_fold_<k>/
interpretation. Paths of the attribution's files come from attribute.json, as attr_status.fold_files() derives them.

Before anything is written:
  - the fold is DONE by attr_status.py's rule (--pin names another pin than <root>/attribution_engine.json; a fold the
    stock command attributed is accepted through its stock record, as attr_status accepts it);
  - attribute.json names exactly one peaks file, and it has one non-blank line per idx entry (blank lines are refused);
  - the ohe and attr npz hold (idx.sum(), 4, W) of int8 and float32 in C order, with W attribute.json's attr_window,
    even;
  - the idx mask, and the bed it selects, equal those recorded by every sibling fold of the dataset already converted
    (06.0 checks only shapes, so rows that differ between folds would be averaged silently).
While the scores are written, in blocks of --block-rows rows (default 16,384, whole chunks), each ohe column must be
one-hot, each attr value finite, and no |attr| above 65504, the largest float16 (it would become inf); any of these
fails the conversion. The float16 statistics record, for shap and for projected_shap, the non-zero float32 values
that become 0, the float16 subnormals, max |value| and the count above 65504; an underflow fraction above 1e-3 of the
non-zero values is flagged (the downstream plan's check V5), not refused.

Atomic and idempotent: the h5, the bed and the provenance are each written to <name>.partial and renamed, the
provenance last, after every source has been stat'ed again and found unchanged. A provenance that is there therefore
describes the outputs beside it; an earlier one is removed before a conversion replaces its outputs, and a failed
conversion removes its .partial files. A fold whose provenance matches the current sources (path, size and mtime of
attribute.json, the peaks file, the ohe, attr and idx files and the meta or stock record) and whose outputs still have
the recorded size and mtime is left alone ("up to date"); --force converts it again. A source changed by a later
attribution therefore makes the next run convert again.

    python workflows/downstream/to_chrombpnet_scores.py --attribute-json <root>/<dataset>/fold_<k>/attribute.json
        [--config <root>/configs/<dataset>/config.yaml | --out-dir <...>/<dataset>_<peak_type>_fold_<k>/interpretation]
        [--pin <attribution_engine.json>] [--force] [--lock] [--block-rows 16384]

Run it under the fold's lock, `flock -n <fold>/.task.lock python ...` (run_task.sh's lock: no attribution can replace
the files while they are read), or pass --lock to take it here; a lock held elsewhere then exits 4. Exit 0 when
converted or up to date, 1 when refused or failed. Run it inside a box, never on the login node: it needs h5py and
hdf5plugin (the cherimoya environment has both) and reads the npz files block by block, about 1-2 GB of RAM at the
default block size and W = 1000.

Two checks of a whole dataset, for downstream.sh: each writes a JSON report (--report; default
<output_dir>/validation/check_dataset.json or check_average.json) and exits 0 only when everything passes.

    --check-dataset <dataset> --root <root> [--against auto|chrombpnet|fasta] [--sample-rows 2000] [--seed 0]
        Before 06.0. Every fold of the config is converted from its current attribution (its provenance matches the
        sources and the outputs), the idx mask hashes to what every provenance recorded, and the masks and the beds
        (byte for byte) are one across the folds. Then V1 of the downstream plan, on each fold: against ChromBPNet's
        own 05.0 files of the same dataset and fold (the interpretation/ directory beside the evaluation/ one the
        fold's compare.json names), joined by peak name, the ohe equals ChromBPNet's raw/seq[:, :, c0:c0 + W] (c0 =
        L/2 - W/2, 557 at W = 1000) and so does the converted raw/seq, and the rows only ChromBPNet has are exactly
        those whose L-bp raw/seq holds a column of zeros: every row of the first fold, --sample-rows rows of the
        others. Or, for a dataset ChromBPNet did not interpret, against the genome: on --sample-rows rows of every
        fold the L-bp window around the peak's centre is A, C, G and T only and its central W bases' one-hot is the
        ohe, and every dropped row's window holds another letter. auto picks ChromBPNet's files when every fold has
        them.
    --check-average <dataset> --root <root> [--sample-rows 1000] [--seed 0] [--threads 5]
        After 06.0 (V2). The averaged h5 has the keys, dtypes and shape (N, 4, W) of the first fold's bed, its raw/seq
        is the first fold's ohe on every row, and on --sample-rows rows its shap and projected_shap equal bit for bit
        the average recomputed from the npz files with 06.0's float16 arithmetic, in the config's fold order.
"""

from __future__ import annotations

import argparse
import fcntl
import glob
import hashlib
import json
import logging
import os
import re
import resource
import socket
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import h5py
import hdf5plugin
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "attribution"), str(HERE.parent / "train")]

import attr_status  # noqa: E402
import make_jobs  # noqa: E402

logger = logging.getLogger("to_chrombpnet_scores")

FORMAT = 1
"""The layout and provenance this writes; a provenance of another format is converted again."""
KEYS = ("raw", "shap", "projected_shap")
DTYPES = {"raw": np.int8, "shap": np.float16, "projected_shap": np.float16}
CHUNK_ROWS = 16
"""chrombpnet scores_io.py's CHUNK_ROWS: chunks are (min(N, 16), 4, L)."""
BLOCK_ROWS = 16384
FLOAT16_MAX = 65504.0
FLOAT16_TINY = 2.0**-14
"""The smallest normal float16; non-zero values below it are subnormal."""
UNDERFLOW_FLAG = 1e-3
"""Validation V5: a float16 underflow fraction above this, of the non-zero values, is flagged."""
TAGGING_KEYS = ("full_model_precision", "full_model_optimizer", "full_model_patience")
"""Config keys that rename full_models/ (config.sh:160-173); a Cherimoya config has none of them."""
LOCK_BUSY_EXIT = 4
CODE_FILES = ("to_chrombpnet_scores.py", "attr_status.py", "make_jobs.py")
CODE_DIRS = {"attr_status.py": HERE.parent / "attribution", "make_jobs.py": HERE.parent / "train"}
OUTPUT_NAMES = ("h5", "bed", "provenance")
CHECK_SAMPLE_ROWS = 2000
"""--check-dataset: rows compared per fold (all of them for the first fold against ChromBPNet)."""
AVERAGE_SAMPLE_ROWS = 1000
"""--check-average: rows of 06.0's average recomputed from the npz files."""
ACGT = b"ACGT"


class Refused(Exception):
    """The fold cannot be converted as it is; nothing is left behind."""


def blosc_filter() -> hdf5plugin.Blosc:
    """chrombpnet scores_io.py's filter: Blosc, blosclz at level 9 with byte shuffle (h5ls: blosc-32001 {...,9,1,0})."""
    return hdf5plugin.Blosc(cname="blosclz", clevel=9, shuffle=hdf5plugin.Blosc.SHUFFLE)


# ---------------------------------------------------------------------------------------------------------------------
# where things are


def read_cheri_config(path: Path) -> dict:
    """The dataset's Cherimoya config.yaml: make_jobs.read_config's fields, and contrib_heads when it lists them."""
    import yaml

    try:
        cfg = make_jobs.read_config(path)
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise Refused(f"cannot read the Cherimoya config {path}: {exc}") from exc
    tagging = [key for key in TAGGING_KEYS if raw.get(key) is not None]
    if tagging:
        raise Refused(f"{path} sets {', '.join(tagging)}, which rename full_models/; a Cherimoya config has none")
    heads = raw.get("contrib_heads")
    cfg["contrib_heads"] = [str(h) for h in heads] if isinstance(heads, list) else None
    cfg["config"] = str(path)
    return cfg


def resolve(params: dict, attribute_json: Path, config: Path | None, out_dir: Path | None) -> dict:
    """The fold (dataset, k), the converted files and where sibling folds keep theirs."""
    fold_dir = attribute_json.parent
    dataset = fold_dir.parent.name
    match = re.fullmatch(r"fold_(\d+)", fold_dir.name)
    if match is None:
        raise Refused(f"{attribute_json} is not in a fold_<k> directory")
    fold = match.group(1)
    model = Path(params["model"]).name
    if model != f"cherimoya.fold_{fold}.torch":
        raise Refused(f"{attribute_json} attributes {model}, not cherimoya.fold_{fold}.torch")
    head = params.get("output")
    cfg = None
    if out_dir is None:
        config = config or fold_dir.parent.parent / "configs" / dataset / "config.yaml"
        cfg = read_cheri_config(config)
        if cfg["dataset_name"] != dataset:
            raise Refused(f"{config} is for {cfg['dataset_name']}, the fold for {dataset}")
        if fold not in cfg["folds"]:
            raise Refused(f"{config} lists folds {cfg['folds']}, not {fold}")
        if cfg["contrib_heads"] is not None and head not in cfg["contrib_heads"]:
            raise Refused(f"{config} lists contrib_heads {cfg['contrib_heads']}, not {head!r}")
        prefix = f"{dataset}_{cfg['peak_type']}_fold_{fold}"
        out_dir = Path(cfg["output_dir"]) / "full_models" / prefix / "interpretation"
    out_dir = Path(os.path.abspath(out_dir))
    parent = out_dir.parent.name
    if out_dir.name != "interpretation" or not (parent.startswith(f"{dataset}_") and parent.endswith(f"_fold_{fold}")):
        raise Refused(
            f"{out_dir} is not a {dataset}_<peak_type>_fold_{fold}/interpretation directory (06.0 reads those)"
        )
    stem = out_dir / "interpretation"
    return {
        "dataset": dataset,
        "fold": fold,
        "fold_dir": fold_dir,
        "head": head,
        "config": cfg,
        "out_dir": out_dir,
        "h5": Path(f"{stem}.{head}_scores.h5"),
        "bed": Path(f"{stem}.interpreted_regions.bed"),
        "provenance": Path(f"{stem}.provenance.json"),
        "siblings": (out_dir.parent.parent, parent[: -len(f"_fold_{fold}")]),
    }


def partial(path: Path) -> Path:
    return Path(f"{path}.partial")


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def record_path(files: dict) -> tuple[str, Path]:
    """("meta" or "stock", path): the record attr_status judges the fold by (the stock record when it is newer)."""
    meta, stock = _mtime(files["meta"]), _mtime(files["stock"])
    if stock is not None and (meta is None or stock > meta):
        return "stock", files["stock"]
    return "meta", files["meta"]


def fold_sources(params: dict, attribute_json: Path) -> dict[str, Path]:
    """{role: path} of every file a fold's conversion depends on: what its provenance's `sources` fingerprint."""
    files = attr_status.fold_files(params)
    return {
        "attribute_json": Path(attribute_json),
        "peaks": Path(params["loci"][0]),
        "ohe": files["ohe"],
        "attr": files["attr"],
        "idx": files["idx"],
        "record": record_path(files)[1],
    }


def stat_output(path: Path) -> dict:
    st = os.stat(path)
    return {"path": str(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def stat_sources(paths: dict[str, Path]) -> dict[str, dict]:
    """{role: {path, size, mtime_ns}}: the fingerprint a provenance records and a rerun compares."""
    try:
        return {role: stat_output(path) for role, path in paths.items()}
    except FileNotFoundError as exc:
        raise Refused(f"no {exc.filename}") from exc


def sources_current(recorded: dict) -> bool:
    """Whether every source a provenance recorded still has its recorded size and mtime."""
    for src in recorded.values():
        try:
            if stat_output(Path(src["path"])) != src:
                return False
        except (OSError, KeyError, TypeError):
            return False
    return bool(recorded)


def up_to_date(where: dict, sources: dict) -> str | None:
    """Why the fold must be converted (again), or None when its provenance describes these sources and outputs."""
    try:
        prov = json.loads(where["provenance"].read_text())
    except FileNotFoundError:
        return "not converted yet"
    except (OSError, ValueError) as exc:
        return f"unreadable {where['provenance']}: {exc}"
    if prov.get("format") != FORMAT:
        return f"provenance format {prov.get('format')!r}, this converter writes {FORMAT}"
    recorded = prov.get("sources") or {}
    if recorded != sources:
        changed = sorted(role for role in set(sources) | set(recorded) if recorded.get(role) != sources.get(role))
        return f"sources changed since the conversion: {', '.join(changed)}"
    for name in ("h5", "bed"):
        written = {k: ((prov.get("outputs") or {}).get(name) or {}).get(k) for k in ("path", "size", "mtime_ns")}
        try:
            current = stat_output(where[name])
        except FileNotFoundError:
            return f"no {where[name]}"
        if written != current:
            return f"{where[name]} is not the file the provenance records"
    return None


# ---------------------------------------------------------------------------------------------------------------------
# reading


def read_peak_lines(path: Path) -> list[bytes]:
    """The peaks file's lines, each with its terminator (a last line without one gets b"\\n"); blank lines are refused,
    so line i is row i of tangermeme's (and attr_status's) count."""
    data = Path(path).read_bytes()
    lines = data.split(b"\n")
    tail = lines.pop()
    out = [line + b"\n" for line in lines]
    if tail:
        out.append(tail + b"\n")
    for i, line in enumerate(out):
        if not line.strip():
            raise Refused(f"{path}: blank line at row {i + 1}; the idx mask cannot be mapped onto its rows")
    return out


def peak_name(line: bytes) -> str:
    fields = line.rstrip(b"\r\n").decode("utf-8", "replace").split("\t")
    return fields[3] if len(fields) > 3 else ":".join(fields[:3])


class NpzRows:
    """The rows of an .npz's arr_0, read in order from the compressed member: one block in memory at a time."""

    def __init__(self, path: Path, member: str = "arr_0.npy"):
        self.path = path
        self.zip = zipfile.ZipFile(path)
        try:
            self.fh = self.zip.open(member)
            header = attr_status.read_npy_header(self.fh)
        except BaseException:
            self.zip.close()
            raise
        if header["fortran_order"]:
            self.close()
            raise Refused(f"{path}: {member} is in Fortran order")
        self.dtype = np.dtype(header["descr"])
        self.descr = header["descr"]
        self.shape = header["shape"]
        self.row_shape = self.shape[1:]
        self.row_bytes = int(np.prod(self.row_shape, dtype=np.int64)) * self.dtype.itemsize
        self.rows_read = 0

    def read(self, n: int) -> np.ndarray:
        want = n * self.row_bytes
        parts, got = [], 0
        while got < want:
            chunk = self.fh.read(want - got)
            if not chunk:
                raise Refused(f"{self.path}: the array ends after {self.rows_read} rows of {self.shape[0]}")
            parts.append(chunk)
            got += len(chunk)
        self.rows_read += n
        buf = parts[0] if len(parts) == 1 else b"".join(parts)
        return np.frombuffer(buf, dtype=self.dtype).reshape((n, *self.row_shape))

    def finish(self) -> None:
        """Read to the end of the member (zipfile checks its CRC there); refuse trailing data."""
        if self.fh.read(1):
            raise Refused(f"{self.path}: data after the {self.shape[0]} rows its header declares")

    def close(self) -> None:
        try:
            if hasattr(self, "fh"):
                self.fh.close()
        finally:
            self.zip.close()

    def __enter__(self) -> NpzRows:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ---------------------------------------------------------------------------------------------------------------------
# writing


class Float16Stats:
    """What casting a float32 array to float16 does to it, accumulated over blocks."""

    def __init__(self) -> None:
        self.values = self.nonzero = self.underflow = self.subnormal = self.overflow = 0
        self.max_abs = 0.0

    def update(self, x32: np.ndarray, x16: np.ndarray) -> None:
        a32 = np.abs(x32)
        a16 = np.abs(x16)
        nonzero = x32 != 0
        self.values += x32.size
        self.nonzero += int(np.count_nonzero(nonzero))
        self.underflow += int(np.count_nonzero(nonzero & (x16 == 0)))
        self.subnormal += int(np.count_nonzero((a16 > 0) & (a16 < FLOAT16_TINY)))
        self.overflow += int(np.count_nonzero(a32 > FLOAT16_MAX))
        if x32.size:
            self.max_abs = max(self.max_abs, float(a32.max()))

    def record(self) -> dict:
        fraction = self.underflow / self.nonzero if self.nonzero else 0.0
        return {
            "values": self.values,
            "nonzero": self.nonzero,
            "underflow_to_zero": self.underflow,
            "underflow_fraction": fraction,
            "underflow_flagged": fraction > UNDERFLOW_FLAG,
            "underflow_flag_above": UNDERFLOW_FLAG,
            "subnormal": self.subnormal,
            "max_abs": self.max_abs,
            "above_float16_max": self.overflow,
        }


def check_block(ohe: np.ndarray, attr: np.ndarray, first_row: int) -> None:
    """Refuse a block whose ohe is not one-hot at every base, or whose attr is not finite or exceeds float16."""
    bad_value = (ohe.view(np.uint8) > 1).any(axis=(1, 2))
    bad_column = (ohe.sum(axis=1, dtype=np.int32) != 1).any(axis=1)
    bad = np.flatnonzero(bad_value | bad_column)
    if bad.size:
        raise Refused(
            f"ohe row {first_row + int(bad[0])} is not one-hot at every base ({bad.size} such rows in this block); "
            "attribute_fast.py keeps only windows of A, C, G and T"
        )
    finite = np.isfinite(attr)
    if not finite.all():
        row = first_row + int(np.flatnonzero(~finite.all(axis=(1, 2)))[0])
        raise Refused(f"attr row {row} holds a non-finite value")
    over = np.abs(attr) > FLOAT16_MAX
    if over.any():
        row = first_row + int(np.flatnonzero(over.any(axis=(1, 2)))[0])
        raise Refused(
            f"attr row {row} holds |value| {float(np.abs(attr[row - first_row]).max()):g} > {FLOAT16_MAX:g}, "
            f"the largest float16 ({int(np.count_nonzero(over))} such values in this block); the fold fails"
        )


def write_block(datasets: dict, start: int, ohe: np.ndarray, attr: np.ndarray, stats: dict) -> None:
    """Rows [start, start + len(ohe)) of the three datasets, as scores_io.score_arrays casts them."""
    stop = start + len(ohe)
    shap = attr.astype(np.float16)
    projected32 = ohe.astype(np.float32) * attr
    projected = projected32.astype(np.float16)
    stats["shap"].update(attr, shap)
    stats["projected_shap"].update(projected32, projected)
    datasets["raw"][start:stop] = ohe
    datasets["shap"][start:stop] = shap
    datasets["projected_shap"][start:stop] = projected


def write_scores(path: Path, ohe_rows: NpzRows, attr_rows: NpzRows, n: int, width: int, block_rows: int) -> dict:
    """The scores h5 at `path` (truncated if it exists), streamed from the two npz readers; returns the statistics."""
    stats = {"shap": Float16Stats(), "projected_shap": Float16Stats()}
    chunks = (max(1, min(n, CHUNK_ROWS)), 4, width)
    with h5py.File(path, "w", rdcc_nbytes=64 * 1024 * 1024) as fh:
        datasets = {
            key: fh.create_group(key).create_dataset(
                "seq", shape=(n, 4, width), dtype=DTYPES[key], chunks=chunks, **blosc_filter()
            )
            for key in KEYS
        }
        for start in range(0, n, block_rows):
            rows = min(block_rows, n - start)
            ohe, attr = ohe_rows.read(rows), attr_rows.read(rows)
            check_block(ohe, attr, start)
            write_block(datasets, start, ohe, attr, stats)
    ohe_rows.finish()
    attr_rows.finish()
    return {key: s.record() for key, s in stats.items()}


def describe_h5(path: Path) -> dict:
    """Datasets, dtypes, shapes, chunks and filters of the written h5, as read back."""
    out = {}
    with h5py.File(path, "r") as fh:
        for key in KEYS:
            ds = fh[key]["seq"]
            plist = ds.id.get_create_plist()
            filters = []
            for i in range(plist.get_nfilters()):
                code, _flags, values, name = plist.get_filter(i)
                filters.append({"id": int(code), "name": name.decode() if isinstance(name, bytes) else name,
                                "cd_values": [int(v) for v in values]})  # fmt: skip
            out[f"{key}/seq"] = {
                "dtype": str(ds.dtype),
                "shape": list(ds.shape),
                "chunks": list(ds.chunks) if ds.chunks else None,
                "filters": filters,
            }
    return out


def _git(*args: str) -> str | None:
    try:
        run = subprocess.run(["git", "-C", str(HERE), *args], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return run.stdout.strip() if run.returncode == 0 else None


def converter_identity() -> dict:
    hashes = {}
    for name in CODE_FILES:
        try:
            hashes[name] = hashlib.sha256((CODE_DIRS.get(name, HERE) / name).read_bytes()).hexdigest()
        except OSError:
            hashes[name] = None
    dirty = _git("status", "--porcelain", "--", *(str(CODE_DIRS.get(n, HERE) / n) for n in CODE_FILES))
    return {
        "path": str(HERE / "to_chrombpnet_scores.py"),
        "files_sha256": hashes,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if dirty is None else bool(dirty),
        "versions": {
            "numpy": np.__version__,
            "h5py": h5py.__version__,
            "hdf5plugin": hdf5plugin.version,
            "hdf5": h5py.version.hdf5_version,
        },
    }


def write_json(path: Path, data: dict) -> None:
    tmp = partial(path)
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


def remove(path: Path) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------------------------------------------------
# a fold


def convert(
    attribute_json: Path,
    *,
    config: Path | None = None,
    out_dir: Path | None = None,
    pin: Path | None = None,
    force: bool = False,
    block_rows: int = BLOCK_ROWS,
) -> tuple[str, dict]:
    """Convert one fold. Returns ("converted", provenance) or ("up to date", {"reason": ...}); raises Refused."""
    began = time.perf_counter()
    attribute_json = Path(os.path.abspath(attribute_json))
    try:
        params = json.loads(attribute_json.read_text())
    except (OSError, ValueError) as exc:
        raise Refused(f"cannot read {attribute_json}: {exc}") from exc
    where = resolve(params, attribute_json, config, out_dir)
    state, reasons = attr_status.check_fold(where["fold_dir"], pin)
    if state != "done":
        raise Refused(f"{where['fold_dir']} is {state}, not DONE: {'; '.join(reasons)}")
    if len(params["loci"]) != 1:
        raise Refused(f"{attribute_json} names {len(params['loci'])} peaks files; the bed needs exactly one")
    files = attr_status.fold_files(params)
    kind, record_file = record_path(files)
    source_paths = fold_sources(params, attribute_json)
    sources = stat_sources(source_paths)
    if not force:
        reason = up_to_date(where, sources)
        if reason is None:
            return "up to date", {"reason": f"{where['provenance']} matches the sources and outputs"}
        logger.info("%s fold %s: converting (%s)", where["dataset"], where["fold"], reason)
    else:
        logger.info("%s fold %s: converting (--force)", where["dataset"], where["fold"])

    # what the rows are
    width, in_window = params["attr_window"], params["in_window"]
    if not isinstance(width, int) or width <= 0 or width % 2:
        raise Refused(f"attr_window {width!r} is not a positive even number of bases")
    record = json.loads(record_file.read_text())
    idx = np.load(files["idx"], allow_pickle=False)
    if idx.dtype != np.bool_ or idx.ndim != 1:
        raise Refused(f"{files['idx']} is {idx.dtype} {idx.shape}, not a 1-d boolean mask")
    lines = read_peak_lines(source_paths["peaks"])
    if len(lines) != idx.size:
        raise Refused(f"{source_paths['peaks']} has {len(lines)} rows, the idx mask {idx.size} entries")
    kept = np.flatnonzero(idx)
    n = int(kept.size)
    if n == 0:
        raise Refused(f"{files['idx']} keeps no rows")
    bed_bytes = b"".join(lines[i] for i in kept)
    idx_sha = hashlib.sha256(idx.astype(np.uint8).tobytes()).hexdigest()
    bed_sha = hashlib.sha256(bed_bytes).hexdigest()
    dropped = [{"row": int(i), "name": peak_name(lines[i])} for i in np.flatnonzero(~idx)]
    del lines

    siblings = check_siblings(where, idx_sha, bed_sha)

    t = time.perf_counter()
    targets = [where[name] for name in OUTPUT_NAMES]
    with NpzRows(files["ohe"]) as ohe_rows, NpzRows(files["attr"]) as attr_rows:
        for rows, descr in ((ohe_rows, "|i1"), (attr_rows, "<f4")):
            if rows.descr != descr or rows.shape != (n, 4, width):
                raise Refused(f"{rows.path} holds {rows.descr} {rows.shape}, not {descr} {(n, 4, width)}")
        where["out_dir"].mkdir(parents=True, exist_ok=True)
        remove(where["provenance"])  # from here on, no provenance describes outputs that are being replaced
        try:
            partial(where["bed"]).write_bytes(bed_bytes)
            stats = write_scores(partial(where["h5"]), ohe_rows, attr_rows, n, width, block_rows)
            changed = [role for role, st in stat_sources(source_paths).items() if st != sources[role]]
            if changed:
                raise Refused(f"{', '.join(changed)} changed during the conversion; run it under the fold's lock")
            os.replace(partial(where["h5"]), where["h5"])
            os.replace(partial(where["bed"]), where["bed"])
        except BaseException:
            for path in targets:
                remove(partial(path))
            raise
    written_s = time.perf_counter() - t

    start = in_window // 2 - width // 2
    attribution = {
        "record": kind,
        "record_path": str(record_file),
        "random_state": params["random_state"],
        "n_shuffles": params["n_shuffles"],
        "attr_window": width,
        "in_window": in_window,
        "window": [start, start + width],
        "summit_column": width // 2,
    }
    if kind == "meta":
        attribution.update(engine=record.get("tool"), audit=record.get("audit"), code=record.get("code"))
    else:
        attribution.update(engine=record.get("engine"), exit=record.get("exit"), versions=record.get("versions"))
    now = datetime.now().astimezone()
    provenance = {
        "format": FORMAT,
        "tool": "to_chrombpnet_scores.py",
        "written": now.isoformat(timespec="seconds"),
        "written_et": now.astimezone(ZoneInfo("America/New_York")).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "argv": list(sys.argv),
        "converter": converter_identity(),
        "dataset": where["dataset"],
        "fold": where["fold"],
        "head": where["head"],
        "config": where["config"]["config"] if where["config"] else None,
        "pin": str(pin if pin is not None else attr_status.default_pin(where["fold_dir"])),
        "sources": sources,
        "attribution": attribution,
        "rows": {
            "n_rows": int(idx.size),
            "n_kept": n,
            "n_dropped": len(dropped),
            "idx_sha256": idx_sha,
            "idx_sha256_of": "the mask, one byte (0 or 1) per peaks-file row",
            "dropped": dropped,
        },
        "outputs": {
            "h5": {**stat_output(where["h5"]), "datasets": describe_h5(where["h5"])},
            "bed": {**stat_output(where["bed"]), "sha256": bed_sha, "rows": n},
        },
        "float16": stats,
        "siblings": siblings,
        "timings_s": {"write": written_s, "total": time.perf_counter() - began},
        "rss_max_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
    try:
        write_json(where["provenance"], provenance)
    except BaseException:
        remove(partial(where["provenance"]))
        raise
    return "converted", provenance


def check_siblings(where: dict, idx_sha: str, bed_sha: str) -> list[dict]:
    """Compare the mask and the bed with every sibling fold already converted; refuse if any differs.

    A sibling whose recorded sources have changed since (re-attributed, or a new peaks file) is not compared: it is
    converted again before 06.0 may use it, and is compared then. Without that, a new peaks file could never be
    converted, as every sibling would still hold the old rows."""
    parent, prefix = where["siblings"]
    found = []
    for path in sorted(parent.glob(f"{glob.escape(prefix)}_fold_*/interpretation/interpretation.provenance.json")):
        if path == where["provenance"]:
            continue
        try:
            prov = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise Refused(f"unreadable sibling provenance {path}: {exc}") from exc
        entry = {"fold": prov.get("fold"), "provenance": str(path)}
        if not sources_current(prov.get("sources") or {}):
            found.append({**entry, "compared": False, "reason": "its sources changed since its conversion"})
            continue
        rows, bed = prov.get("rows") or {}, (prov.get("outputs") or {}).get("bed") or {}
        if (rows.get("idx_sha256"), bed.get("sha256")) != (idx_sha, bed_sha):
            raise Refused(
                f"fold {prov.get('fold')} ({path}) kept other rows: idx/bed sha256 {rows.get('idx_sha256')}/"
                f"{bed.get('sha256')}, this fold's {idx_sha}/{bed_sha}; 06.0 would average misaligned rows"
            )
        found.append({**entry, "compared": True})
    return found


def take_lock(fold_dir: Path) -> int | None:
    """The fold's .task.lock, held until exit (run_task.sh's `flock -n`); None when another run holds it."""
    fd = os.open(Path(fold_dir) / attr_status.LOCK_NAME, os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    return fd


# ---------------------------------------------------------------------------------------------------------------------
# a dataset: its converted folds (--check-dataset) and 06.0's average of them (--check-average)


def dataset_folds(root: Path, dataset: str, config: Path | None = None) -> tuple[dict, list[dict]]:
    """The dataset's Cherimoya config and its folds in the config's order, each with its attribute.json, parameters,
    attribution files and converted files (resolve())."""
    config = Path(config or Path(root) / "configs" / dataset / "config.yaml")
    cfg = read_cheri_config(config)
    if cfg["dataset_name"] != dataset:
        raise Refused(f"{config} is for {cfg['dataset_name']}, not {dataset}")
    folds = []
    for k in cfg["folds"]:
        attribute_json = Path(os.path.abspath(Path(root) / dataset / f"fold_{k}" / "attribute.json"))
        try:
            params = json.loads(attribute_json.read_text())
        except (OSError, ValueError) as exc:
            raise Refused(f"cannot read {attribute_json}: {exc}") from exc
        folds.append({
            "fold": k,
            "attribute_json": attribute_json,
            "params": params,
            "files": attr_status.fold_files(params),
            "where": resolve(params, attribute_json, config, None),
        })  # fmt: skip
    return cfg, folds


def bed_fields(line: bytes) -> list[str]:
    return line.rstrip(b"\r\n").decode("utf-8", "replace").split("\t")


def sample_rows(n: int, size: int, seed: int) -> np.ndarray:
    """`size` distinct rows of n (all of them if size >= n), sorted; the same for the same seed."""
    if size >= n:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size=size, replace=False))


def take_npz_rows(path: Path, rows: np.ndarray, block_rows: int = BLOCK_ROWS) -> np.ndarray:
    """arr_0[rows] of an npz (rows sorted and distinct), read in one pass that stops after the last row wanted."""
    with NpzRows(path) as reader:
        out = np.empty((rows.size, *reader.row_shape), dtype=reader.dtype)
        n, i = reader.shape[0], 0
        if rows.size and rows[-1] >= n:
            raise Refused(f"{path} has {n} rows; row {int(rows[-1])} was asked for")
        for start in range(0, n, block_rows):
            if i >= rows.size:
                break
            stop = min(n, start + block_rows)
            block = reader.read(stop - start)
            j = int(np.searchsorted(rows, stop))
            out[i:j] = block[rows[i:j] - start]
            i = j
    return out


def take_h5_rows(ds: h5py.Dataset, rows: np.ndarray) -> np.ndarray:
    """ds[rows] for sorted, distinct rows: one read of whole chunk rows per chunk touched."""
    out = np.empty((rows.size, *ds.shape[1:]), dtype=ds.dtype)
    step = ds.chunks[0] if ds.chunks else CHUNK_ROWS
    i = 0
    while i < rows.size:
        first = int(rows[i]) // step * step
        stop = min(first + step, ds.shape[0])
        j = int(np.searchsorted(rows, stop))
        out[i:j] = ds[first:stop][rows[i:j] - first]
        i = j
    return out


def zero_column_rows(raw: np.ndarray) -> np.ndarray:
    """Which rows of a (B, 4, L) one-hot hold a column of zeros: a base other than A, C, G or T in ChromBPNet's
    encoding, the windows Cherimoya drops."""
    return (raw.astype(np.int16).sum(axis=1) == 0).any(axis=1)


def check_conversions(folds: list[dict]) -> tuple[dict, dict | None]:
    """Every fold converted from its current attribution (its provenance describes the current sources and outputs),
    the idx mask recorded by every provenance, and one mask and one bed (byte for byte) across the folds.
    Returns the report and the first fold's mask and bed lines (None when the folds disagree)."""
    report: dict = {"folds": {}, "failures": []}
    fail = report["failures"].append
    first: dict | None = None
    for f in folds:
        k, where = f["fold"], f["where"]
        entry: dict = {"provenance": str(where["provenance"])}
        report["folds"][k] = entry
        try:
            prov = json.loads(where["provenance"].read_text())
        except FileNotFoundError:
            fail(f"fold {k}: not converted (no {where['provenance']})")
            continue
        except (OSError, ValueError) as exc:
            fail(f"fold {k}: unreadable {where['provenance']}: {exc}")
            continue
        try:
            reason = up_to_date(where, stat_sources(fold_sources(f["params"], f["attribute_json"])))
        except Refused as exc:
            reason = str(exc)
        if reason is not None:
            fail(f"fold {k}: the conversion is not current: {reason}")
        idx = np.load(f["files"]["idx"], allow_pickle=False)
        bed = where["bed"].read_bytes() if where["bed"].exists() else b""
        entry.update(
            n_rows=int(idx.size),
            n_kept=int(idx.sum()),
            idx_sha256=hashlib.sha256(idx.astype(np.uint8).tobytes()).hexdigest(),
            bed_sha256=hashlib.sha256(bed).hexdigest(),
        )
        recorded = (prov.get("rows") or {}).get("idx_sha256")
        if entry["idx_sha256"] != recorded:
            fail(f"fold {k}: the idx mask hashes to {entry['idx_sha256'][:12]}, its provenance has {recorded}")
        recorded = ((prov.get("outputs") or {}).get("bed") or {}).get("sha256")
        if entry["bed_sha256"] != recorded:
            fail(f"fold {k}: the bed hashes to {entry['bed_sha256'][:12]}, its provenance has {recorded}")
        if first is None:
            first = {"fold": k, "idx": idx, "bed": bed}
            continue
        if not np.array_equal(idx, first["idx"]):
            fail(f"fold {k} keeps other rows than fold {first['fold']} ({int((idx != first['idx']).sum())} differ)")
        if bed != first["bed"]:
            fail(f"fold {k}'s bed differs from fold {first['fold']}'s (cmp)")
    report["pass"] = not report["failures"]
    if first is None or not report["pass"]:
        return report, None
    lines = first["bed"].split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    return report, {"idx": first["idx"], "bed_lines": [line + b"\n" for line in lines]}


def chrombpnet_interpretation(fold_dir: Path, head: str) -> dict[str, Path]:
    """ChromBPNet's 05.0 files for the same dataset and fold: the interpretation/ directory beside the evaluation/
    directory the fold's compare.json (make_jobs.py) names."""
    try:
        compare = json.loads((Path(fold_dir) / "compare.json").read_text())
        model_dir = Path(compare["chrombpnet_predictions"]).parent.parent
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise Refused(f"cannot find ChromBPNet's fold from {Path(fold_dir) / 'compare.json'}: {exc}") from exc
    interpretation = model_dir / "interpretation"
    return {
        "h5": interpretation / f"interpretation.{head}_scores.h5",
        "bed": interpretation / "interpretation.interpreted_regions.bed",
    }


def v1_fold(f: dict, rows: np.ndarray | None, kept_lines: list[bytes], block_rows: int) -> dict:
    """V1 on one fold against ChromBPNet's own 05.0 output: joined by peak name, Cherimoya's ohe equals ChromBPNet's
    raw/seq[:, :, c0:c0 + W] (c0 = L/2 - W/2, 557 at L = 2114 and W = 1000), and so does the converted raw/seq; the
    rows only ChromBPNet has are exactly those whose L-bp raw/seq holds a column of zeros. `rows` are the converted
    rows to compare (None: every row, and every ChromBPNet row is scanned for zero columns; else only the sampled rows
    and the ChromBPNet-only rows are read)."""
    params, where, files = f["params"], f["where"], f["files"]
    width, length = int(params["attr_window"]), int(params["in_window"])
    c0 = length // 2 - width // 2
    cbp = chrombpnet_interpretation(where["fold_dir"], where["head"])
    out: dict = {"chrombpnet": {k: str(v) for k, v in cbp.items()}, "window_columns": [c0, c0 + width]}
    failures: list[str] = []
    out["failures"] = failures
    missing = [str(p) for p in cbp.values() if not p.is_file()]
    if missing:
        failures.append(f"no {', '.join(missing)}")
        out["pass"] = False
        return out

    cbp_lines = read_peak_lines(cbp["bed"])
    cbp_fields = [bed_fields(line) for line in cbp_lines]
    names = [peak_name(line) for line in cbp_lines]
    row_of = {name: r for r, name in enumerate(names)}
    if len(row_of) != len(names):
        failures.append(f"{cbp['bed']}: {len(names) - len(row_of)} repeated peak names; the join needs unique names")
    joined = np.full(len(kept_lines), -1, dtype=np.int64)
    coordinate_mismatches = 0
    for j, line in enumerate(kept_lines):
        mine = bed_fields(line)
        r = row_of.get(peak_name(line), -1)
        joined[j] = r
        if r >= 0 and (mine[:3] + mine[9:10]) != (cbp_fields[r][:3] + cbp_fields[r][9:10]):
            coordinate_mismatches += 1
    only_cbp = np.setdiff1d(np.arange(len(cbp_lines)), joined[joined >= 0])
    out.update(
        cbp_rows=len(cbp_lines),
        joined=int((joined >= 0).sum()),
        unjoined=int((joined < 0).sum()),
        join_order_kept=bool(np.all(np.diff(joined) > 0)),
        coordinate_mismatches=coordinate_mismatches,
        cbp_only_rows=int(only_cbp.size),
    )
    if out["unjoined"]:
        failures.append(f"{out['unjoined']} converted rows have no peak of that name in {cbp['bed']}")
    if coordinate_mismatches:
        failures.append(f"{coordinate_mismatches} joined rows differ in chrom, start, end or summit")
    if not out["join_order_kept"]:
        failures.append("joined by name, the converted rows are not in ChromBPNet's row order")
    if failures:
        out["pass"] = False
        return out

    with h5py.File(cbp["h5"], "r") as cbp_h5, h5py.File(where["h5"], "r") as conv_h5:
        cbp_raw, conv_raw = cbp_h5["raw/seq"], conv_h5["raw/seq"]
        if cbp_raw.shape != (len(cbp_lines), 4, length):
            failures.append(f"ChromBPNet's raw/seq is {cbp_raw.shape}, not {(len(cbp_lines), 4, length)}")
            out["pass"] = False
            return out
        mismatch = conv_mismatch = 0
        if rows is None:
            # every row: the ohe in memory, ChromBPNet's raw and the converted raw streamed
            with np.load(files["ohe"], allow_pickle=False) as z:
                ohe = z["arr_0"]
            zero = np.zeros(len(cbp_lines), dtype=bool)
            for b0 in range(0, len(cbp_lines), block_rows):
                b1 = min(len(cbp_lines), b0 + block_rows)
                block = cbp_raw[b0:b1]
                zero[b0:b1] = zero_column_rows(block)
                j0, j1 = np.searchsorted(joined, b0), np.searchsorted(joined, b1)  # joined is increasing (above)
                theirs = block[joined[j0:j1] - b0, :, c0 : c0 + width]
                mismatch += int((theirs != ohe[j0:j1]).any(axis=(1, 2)).sum())
            for b0 in range(0, len(kept_lines), block_rows):
                b1 = min(len(kept_lines), b0 + block_rows)
                conv_mismatch += int((conv_raw[b0:b1] != ohe[b0:b1]).any(axis=(1, 2)).sum())
            compared = len(kept_lines)
            zero_rows = np.flatnonzero(zero)
            out["zero_column_rows_equal_cbp_only"] = bool(np.array_equal(zero_rows, only_cbp))
            out["zero_column_rows"] = int(zero_rows.size)
        else:
            ohe = take_npz_rows(files["ohe"], rows, block_rows)
            order = np.argsort(joined[rows])
            theirs = np.empty_like(ohe)
            theirs[order] = take_h5_rows(cbp_raw, joined[rows][order])[:, :, c0 : c0 + width]
            mismatch = int((theirs != ohe).any(axis=(1, 2)).sum())
            conv_mismatch = int((take_h5_rows(conv_raw, rows) != ohe).any(axis=(1, 2)).sum())
            compared = int(rows.size)
            sampled_zero = int(zero_column_rows(take_h5_rows(cbp_raw, np.sort(joined[rows]))).sum())
            only_zero = zero_column_rows(take_h5_rows(cbp_raw, only_cbp)) if only_cbp.size else np.zeros(0, bool)
            out["sampled_rows_with_zero_column"] = sampled_zero
            out["cbp_only_rows_with_zero_column"] = int(only_zero.sum())
            out["zero_column_rows_equal_cbp_only"] = sampled_zero == 0 and bool(only_zero.all())
    out.update(rows_compared=compared, ohe_mismatch_rows=mismatch, converted_raw_mismatch_rows=conv_mismatch)
    if mismatch:
        failures.append(f"{mismatch} of {compared} rows: the ohe is not ChromBPNet's raw/seq[:, :, {c0}:{c0 + width}]")
    if conv_mismatch:
        failures.append(f"{conv_mismatch} of {compared} rows: the converted raw/seq is not the ohe")
    if not out["zero_column_rows_equal_cbp_only"]:
        failures.append("the rows only ChromBPNet has are not exactly those with a column of zeros in its raw/seq")
    out["pass"] = not failures
    return out


def fasta_fold(f: dict, rows: np.ndarray, idx: np.ndarray, block_rows: int) -> dict:
    """Against the genome alone: on the sampled converted rows, the L-bp window [mid - L/2, mid + L/2) (mid = start +
    (end - start) // 2, tangermeme's centre), upper-cased, is all A, C, G and T, and the one-hot (ACGT, channels
    first) of its central W bases is the ohe row; every dropped row's window holds another letter."""
    import pyfaidx

    params, files = f["params"], f["files"]
    width, length = int(params["attr_window"]), int(params["in_window"])
    c0 = length // 2 - width // 2
    lines = read_peak_lines(Path(params["loci"][0]))
    kept, dropped = np.flatnonzero(idx), np.flatnonzero(~idx)
    lut = np.zeros((256, 4), dtype=np.int8)
    for k, base in enumerate(ACGT):
        lut[base, k] = 1
    acgt = np.zeros(256, dtype=bool)
    acgt[list(ACGT)] = True
    fasta = pyfaidx.Fasta(str(params["sequences"]), as_raw=True, sequence_always_upper=False, rebuild=False)

    def window(i: int) -> np.ndarray | None:
        chrom, start, end = bed_fields(lines[i])[:3]
        mid = int(start) + (int(end) - int(start)) // 2
        w0 = mid - length // 2
        if chrom not in fasta or w0 < 0 or w0 + length > len(fasta[chrom]):
            return None
        return np.frombuffer(fasta[chrom][w0 : w0 + length].upper().encode("ascii"), dtype=np.uint8)

    ohe = take_npz_rows(files["ohe"], rows, block_rows)
    out_of_bounds, not_acgt, mismatch, dropped_acgt = [], [], [], []
    for t, j in enumerate(rows):
        seq = window(int(kept[j]))
        if seq is None:
            out_of_bounds.append(int(kept[j]))
        elif not acgt[seq].all():
            not_acgt.append(int(kept[j]))
        elif not np.array_equal(lut[seq[c0 : c0 + width]].T, ohe[t]):
            mismatch.append(int(j))
    for i in dropped:
        seq = window(int(i))
        if seq is not None and acgt[seq].all():
            dropped_acgt.append(int(i))
    fasta.close()
    out = {
        "fasta": str(params["sequences"]),
        "window": f"[mid - {length // 2}, mid + {length // 2}), central [{c0}, {c0 + width})",
        "rows_compared": int(rows.size),
        "kept_out_of_bounds": out_of_bounds,
        "kept_with_other_letters": not_acgt[:20],
        "onehot_mismatch_rows": len(mismatch),
        "onehot_mismatch_examples": mismatch[:10],
        "dropped_rows": int(dropped.size),
        "dropped_all_acgt": dropped_acgt[:20],
    }
    failures = []
    if out_of_bounds:
        failures.append(f"{len(out_of_bounds)} sampled windows run off their chromosome")
    if not_acgt:
        failures.append(f"{len(not_acgt)} sampled kept windows hold a letter other than A, C, G or T")
    if mismatch:
        failures.append(f"{len(mismatch)} of {rows.size} sampled rows: the ohe is not the window's one-hot")
    if dropped_acgt:
        failures.append(f"{len(dropped_acgt)} dropped rows have windows of A, C, G and T only")
    out["failures"] = failures
    out["pass"] = not failures
    return out


def check_dataset(
    root: Path,
    dataset: str,
    *,
    config: Path | None = None,
    against: str = "auto",
    n_sample: int = CHECK_SAMPLE_ROWS,
    seed: int = 0,
    block_rows: int = BLOCK_ROWS,
) -> dict:
    """--check-dataset: the folds agree with each other (check_conversions), then V1 against ChromBPNet (the first
    fold on every row, the others on n_sample rows) or the genome check on n_sample rows of every fold."""
    began = time.perf_counter()
    cfg, folds = dataset_folds(root, dataset, config)
    report: dict = {"mode": "check-dataset", "dataset": dataset, "config": cfg["config"], "folds": cfg["folds"]}
    report["conversions"], first = check_conversions(folds)
    if against == "auto":
        cbp_files = [chrombpnet_interpretation(f["where"]["fold_dir"], f["where"]["head"]) for f in folds]
        against = "chrombpnet" if all(p.is_file() for files in cbp_files for p in files.values()) else "fasta"
    report["against"] = against
    report["v1"] = {}
    if first is not None:
        n_kept = len(first["bed_lines"])
        for i, f in enumerate(folds):
            seed_k = seed + 1000 * i
            if against == "chrombpnet":
                rows = None if i == 0 else sample_rows(n_kept, n_sample, seed_k)
                result = v1_fold(f, rows, first["bed_lines"], block_rows)
            else:
                result = fasta_fold(f, sample_rows(n_kept, n_sample, seed_k), first["idx"], block_rows)
            result["rows"] = "all" if against == "chrombpnet" and i == 0 else f"{min(n_sample, n_kept)} sampled"
            report["v1"][f["fold"]] = result
            logger.info("%s fold %s: V1 against %s, %s rows: %s", dataset, f["fold"], against, result["rows"],
                        "pass" if result["pass"] else "FAIL: " + "; ".join(result["failures"]))  # fmt: skip
    report["pass"] = (
        report["conversions"]["pass"] and bool(report["v1"]) and all(r["pass"] for r in report["v1"].values())
    )
    report["timings_s"] = {"total": time.perf_counter() - began}
    report["rss_max_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return report


def average_path(cfg: dict, dataset: str, head: str) -> Path:
    """06.0's output: <output_dir>/contrib_scores/<dataset>/<dataset>_average_shaps.<head>.h5 (config.sh's
    averaged_dir)."""
    return Path(cfg["output_dir"]) / "contrib_scores" / dataset / f"{dataset}_average_shaps.{head}.h5"


def check_average(
    root: Path,
    dataset: str,
    *,
    config: Path | None = None,
    n_sample: int = AVERAGE_SAMPLE_ROWS,
    seed: int = 0,
    threads: int = 5,
    block_rows: int = BLOCK_ROWS,
) -> dict:
    """--check-average (V2): 06.0's averaged h5 has the keys, dtypes and shape (N, 4, W) of the first fold's bed, its
    raw/seq is the first fold's ohe on every row, and on n_sample rows its shap and projected_shap equal, bit for bit,
    the average recomputed from the attribution npz files with 06.0's arithmetic (average_contrib_scores.py): each
    fold cast to float16 as the converter casts it, summed in float16 in place in the config's fold order, then
    divided in place by the number of folds. float16 sums are not associative, so the order is part of the check."""
    began = time.perf_counter()
    cfg, folds = dataset_folds(root, dataset, config)
    head = folds[0]["params"]["output"]
    avg = average_path(cfg, dataset, head)
    width = int(folds[0]["params"]["attr_window"])
    n_bed = len(read_peak_lines(folds[0]["where"]["bed"]))
    report: dict = {"mode": "check-average", "dataset": dataset, "config": cfg["config"], "average": str(avg),
                    "folds": cfg["folds"], "bed_rows": n_bed}  # fmt: skip
    failures: list[str] = []
    report["failures"] = failures
    if not avg.is_file():
        failures.append(f"no {avg}")
    else:
        with h5py.File(avg, "r") as fh:
            layout = {key: (str(fh[key]["seq"].dtype), list(fh[key]["seq"].shape)) for key in KEYS if key in fh}
            report["layout"] = layout
            for key in KEYS:
                want = (str(np.dtype(DTYPES[key])), [n_bed, 4, width])
                if layout.get(key) != want:
                    failures.append(f"{key}/seq is {layout.get(key)}, not {want}")
            if not failures:
                raw, mismatch = fh["raw/seq"], 0
                with NpzRows(folds[0]["files"]["ohe"]) as ohe_rows:
                    for b0 in range(0, n_bed, block_rows):
                        b1 = min(n_bed, b0 + block_rows)
                        mismatch += int((raw[b0:b1] != ohe_rows.read(b1 - b0)).any(axis=(1, 2)).sum())
                report["raw_mismatch_rows"] = mismatch
                if mismatch:
                    failures.append(f"raw/seq differs from fold {folds[0]['fold']}'s ohe on {mismatch} rows")
                rows = sample_rows(n_bed, n_sample, seed)
                got = {key: take_h5_rows(fh[key]["seq"], rows) for key in KEYS[1:]}
    if not failures:

        def recompute(f: dict) -> tuple[np.ndarray, np.ndarray]:
            ohe = take_npz_rows(f["files"]["ohe"], rows, block_rows)
            attr = take_npz_rows(f["files"]["attr"], rows, block_rows)
            return attr.astype(np.float16), (ohe.astype(np.float32) * attr).astype(np.float16)

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
            parts = list(pool.map(recompute, folds))  # zlib and h5py release the GIL while they read
        want: dict[str, np.ndarray] = {}
        for shap, projected in parts:  # average_contrib_scores.py: copy the first fold, += the others, /= n
            if not want:
                want = {"shap": shap.copy(), "projected_shap": projected.copy()}
            else:
                want["shap"] += shap
                want["projected_shap"] += projected
        for key in want:
            want[key] /= len(parts)
            bad = int((want[key].view(np.uint16) != got[key].view(np.uint16)).any(axis=(1, 2)).sum())
            report[f"{key}_rows_not_bitwise_equal"] = bad
            if bad:
                failures.append(f"{key}/seq: {bad} of {rows.size} sampled rows differ from the recomputed average")
        report["rows_compared"] = int(rows.size)
    report["pass"] = not failures
    report["timings_s"] = {"total": time.perf_counter() - began}
    report["rss_max_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return report


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, report)


# ---------------------------------------------------------------------------------------------------------------------
# command line


def run_check(args: argparse.Namespace) -> int:
    """--check-dataset or --check-average: write the report, log the verdict; exit 0 when every check passes."""
    dataset = args.check_dataset or args.check_average
    name = "check_dataset" if args.check_dataset else "check_average"
    try:
        if args.check_dataset:
            report = check_dataset(args.root, dataset, config=args.config, against=args.against,
                                   n_sample=args.sample_rows or CHECK_SAMPLE_ROWS, seed=args.seed,
                                   block_rows=args.block_rows)  # fmt: skip
        else:
            report = check_average(args.root, dataset, config=args.config,
                                   n_sample=args.sample_rows or AVERAGE_SAMPLE_ROWS, seed=args.seed,
                                   threads=args.threads, block_rows=args.block_rows)  # fmt: skip
        cfg = read_cheri_config(Path(report["config"]))
    except Refused as exc:
        logger.error("%s: %s refused: %s", dataset, name, exc)
        return 1
    path = args.report or Path(cfg["output_dir"]) / "validation" / f"{name}.json"
    write_report(path, report)
    failures = report.get("failures") or []
    if args.check_dataset:
        failures = report["conversions"]["failures"] + [
            f"fold {k}: {msg}" for k, r in report["v1"].items() for msg in r["failures"]
        ]
    for line in failures:
        logger.error("%s: %s", dataset, line)
    logger.info("%s: %s %s in %.1f s (max RSS %.2f GB); report %s", dataset, name,
                "passed" if report["pass"] else "FAILED", report["timings_s"]["total"],
                report["rss_max_bytes"] / 1e9, path)  # fmt: skip
    return 0 if report["pass"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--attribute-json", type=Path, help="convert this fold: <root>/<dataset>/fold_<k>/attribute.json")
    mode.add_argument("--check-dataset", metavar="DATASET", help="check the dataset's converted folds (with --root)")
    mode.add_argument("--check-average", metavar="DATASET", help="V2: check 06.0's average of them (with --root)")
    where = ap.add_mutually_exclusive_group()
    where.add_argument("--config", type=Path, help="the Cherimoya config.yaml (default <root>/configs/<dataset>/)")
    where.add_argument("--out-dir", type=Path, help="the interpretation directory (default: from the config)")
    ap.add_argument("--pin", type=Path, help=f"the engine pin (default <root>/{attr_status.PIN_NAME})")
    ap.add_argument("--force", action="store_true", help="convert even when the provenance says up to date")
    ap.add_argument("--lock", action="store_true", help="take the fold's .task.lock here (exit 4 when it is held)")
    ap.add_argument("--block-rows", type=int, default=BLOCK_ROWS, help=f"rows per block (default {BLOCK_ROWS})")
    ap.add_argument("--root", type=Path, help="checks: make_jobs.py's --out-root, holding <dataset>/fold_<k>/")
    ap.add_argument("--against", choices=("auto", "chrombpnet", "fasta"), default="auto",
                    help="--check-dataset: V1 against ChromBPNet's 05.0 files or the genome (auto: ChromBPNet's when "
                         "every fold has them)")  # fmt: skip
    ap.add_argument("--sample-rows", type=int, help=f"checks: rows per fold (default {CHECK_SAMPLE_ROWS} for "
                    f"--check-dataset, {AVERAGE_SAMPLE_ROWS} for --check-average)")  # fmt: skip
    ap.add_argument("--seed", type=int, default=0, help="checks: seed of the row samples (default 0)")
    ap.add_argument("--threads", type=int, default=5, help="--check-average: folds read at once (default 5)")
    ap.add_argument(
        "--report", type=Path, help="checks: the JSON report (default <output_dir>/validation/<check>.json)"
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%F %T")  # fmt: skip
    if args.block_rows <= 0:
        ap.error("--block-rows must be positive")
    if args.check_dataset or args.check_average:
        if args.root is None or args.out_dir is not None:
            ap.error("--check-dataset and --check-average take --root (and --config), not --out-dir")
        return run_check(args)

    fold_dir = Path(os.path.abspath(args.attribute_json)).parent
    lock = None
    if args.lock:
        lock = take_lock(fold_dir)
        if lock is None:
            logger.error("%s: another run holds %s; not converted", fold_dir, attr_status.LOCK_NAME)
            return LOCK_BUSY_EXIT
    try:
        status, info = convert(
            args.attribute_json,
            config=args.config,
            out_dir=args.out_dir,
            pin=args.pin,
            force=args.force,
            block_rows=args.block_rows,
        )
    except Refused as exc:
        logger.error("%s: not converted: %s", fold_dir, exc)
        return 1
    finally:
        if lock is not None:
            os.close(lock)
    if status == "up to date":
        logger.info("%s: up to date (%s)", fold_dir, info["reason"])
        return 0
    rows, f16 = info["rows"], info["float16"]
    logger.info(
        "%s fold %s: wrote %s and %s (%d of %d rows; dropped %s), then %s in %.1f s. float16: shap underflow %.3g of "
        "non-zero values%s, max |attr| %.4g; projected underflow %.3g",
        info["dataset"],
        info["fold"],
        info["outputs"]["h5"]["path"],
        info["outputs"]["bed"]["path"],
        rows["n_kept"],
        rows["n_rows"],
        ", ".join(d["name"] for d in rows["dropped"][:5]) + (" ..." if rows["n_dropped"] > 5 else "") or "none",
        Path(info["outputs"]["h5"]["path"]).with_name("interpretation.provenance.json"),
        info["timings_s"]["total"],
        f16["shap"]["underflow_fraction"],
        " (flagged: above 1e-3)" if f16["shap"]["underflow_flagged"] else "",
        f16["shap"]["max_abs"],
        f16["projected_shap"]["underflow_fraction"],
    )
    for key in KEYS[1:]:
        if f16[key]["underflow_flagged"]:
            logger.warning("%s: %.3g of the non-zero values became 0 in float16 (V5 flags above %g)",
                           key, f16[key]["underflow_fraction"], UNDERFLOW_FLAG)  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
