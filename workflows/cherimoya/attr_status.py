#!/usr/bin/env python3
"""Whether each Cherimoya fold's attribution is done, and the pin of the engine it must have been made with.

A fold's attribution is attribute_fast.py run on its attribute.json (make_downstream.py; run_task.sh's `attribute`
stage). attribute_fast.py writes the ohe, attr and idx files the JSON names, then two sidecars beside the attr file,
<attr minus .npz>.deltas.npy and <attr minus .npz>.meta.json, then audits the run against the stock deep_lift_shap and
rewrites the meta with the audit. It never skips a finished fold and has no done marker of its own, and an idx file
proves nothing: it is written before the deltas, the meta and the audit. So "done" is defined here, once, and every
caller (the stage, the converter, the runners' loops) asks this script.

A fold is DONE when all of these hold:
  - the ohe, attr, idx and deltas files are non-empty, and the meta exists;
  - the meta is newer than the model (<name>.torch; a refit makes the attribution stale) and not older than any of
    the four outputs (an interrupted rerun beside an old meta);
  - audit.pass is true (an audit that failed, never finished, or was turned off is not done);
  - outputs.attr_finite and deltas.finite are true;
  - outputs.attr_shape is [inputs.kept, 4, W], and the files agree: the attr and ohe npz hold (kept, 4, W) float32 and
    int8, the deltas (kept, n_shuffles), and the idx mask has inputs.loci entries, kept of them true;
  - inputs.loci equals the rows of the JSON's peaks file(s);
  - the meta's random_state and n_shuffles, and the parameters that decide the result (model, sequences, loci,
    chroms, in_window, attr_window, exclusion_lists, algorithm, output, group, dtype), equal attribute.json's, and so
    do its output paths;
  - code.files_sha256 equals the pin's: the sha256 of attribute_fast.py, cheri_dls.py and dls_refs.py at sign-off.
    The pin compares file hashes, not the git commit, which moves with every commit to the checkout.

A fold the stock `cherimoya attribute` attributed instead (the fallback after an audit fails twice) is accepted through
a record beside the meta, <attr minus .npz>.stock.json, when it is newer than any meta. The stock command writes no
deltas and no meta, so whoever runs it writes the record: {"engine": "stock", "exit": 0, "attr_finite": true,
"parameters": <the JSON as run>, "versions": {...}}. The same clauses apply, without the deltas, the audit and the
pin; batch_size may differ (20 for stock).

States, for listing: done; running (the fold's .task.lock is held: a fit, compare or attribution); waiting (the fit is
not done: no <name>.torch or no <name>.test.performance.tsv); stale (outputs from other parameters, another engine, an
older model or other peaks); failed (the audit failed or never finished, non-finite or misshapen outputs); missing (no
attribute.json, never run, or interrupted). --check does not look at the lock, so a caller that holds it can use it.

    attr_status.py --check <fold>                exit 0 if DONE, else 1, with the state and every failing clause
    attr_status.py <fold> [<fold> ...]           one line per fold: dataset, fold, state, reason
    attr_status.py --root <dir> [--only "ds ..."] [--summary] [--settled]
                                                 every fold with a fit.json under <dir>/<dataset>/fold_<k>/; --summary
                                                 counts states per dataset; --settled exits 0 only when every fold
                                                 listed is done or failed
    attr_status.py --check-engine <engine dir>   exit 0 if the engine files hash to the pin
    attr_status.py --files <fold>                the fold's attribution files (outputs, sidecars, stock record)
    attr_status.py --write-pin <engine dir>      sign off: write the pin (commit and sha256 of the engine files)

The pin is <root>/attribution_engine.json, where <root> is the fold's grandparent (make_jobs.py's --out-root) or
--root; --pin names another. Stdlib only: it reads the .npy and .npz headers itself and loads no arrays but the idx.
"""

from __future__ import annotations

import argparse
import ast
import errno
import fcntl
import hashlib
import json
import os
import struct
import subprocess
import sys
import time
import zipfile
from pathlib import Path

CODE_FILES = ("attribute_fast.py", "cheri_dls.py", "dls_refs.py")
"""attribute_fast.py's CODE_FILES: the engine's code, whose sha256 its meta records."""
PIN_NAME = "attribution_engine.json"
LOCK_NAME = ".task.lock"
STATES = ("done", "running", "waiting", "stale", "failed", "missing")
RESULT_KEYS = (
    "model",
    "sequences",
    "loci",
    "chroms",
    "in_window",
    "attr_window",
    "exclusion_lists",
    "algorithm",
    "output",
    "group",
    "n_shuffles",
    "random_state",
    "dtype",
)
"""The attribute.json parameters an attribution's numbers depend on; the meta's must equal them."""
NPY_MAGIC = b"\x93NUMPY"

_rows_cache: dict[tuple, int] = {}


# ---------------------------------------------------------------------------------------------------------------------
# files


def sidecar_base(attr_filename: str) -> str:
    """<attr minus .npz>, as attribute_fast.py names its sidecars."""
    return attr_filename[: -len(".npz")] if attr_filename.endswith(".npz") else attr_filename


def npy_path(name: str) -> str:
    """numpy.save appends .npy unless the name has it."""
    return name if name.endswith(".npy") else name + ".npy"


def npz_path(name: str) -> str:
    """numpy.savez_compressed appends .npz unless the name has it."""
    return name if name.endswith(".npz") else name + ".npz"


def fold_files(params: dict) -> dict[str, Path]:
    """Every file of a fold's attribution, from its attribute.json."""
    base = sidecar_base(params["attr_filename"])
    return {
        "ohe": Path(npz_path(params["ohe_filename"])),
        "attr": Path(npz_path(params["attr_filename"])),
        "idx": Path(npy_path(params["idx_filename"])),
        "deltas": Path(base + ".deltas.npy"),
        "meta": Path(base + ".meta.json"),
        "stock": Path(base + ".stock.json"),
    }


def model_files(params: dict) -> tuple[Path, Path]:
    """The model and its test-performance file, the fit's done marker (run_task.sh)."""
    model = Path(params["model"])
    name = model.name[: -len(".torch")] if model.name.endswith(".torch") else model.name
    return model, model.with_name(f"{name}.test.performance.tsv")


def _nonempty(path: Path) -> bool:
    try:
        return path.stat().st_size > 0
    except OSError:
        return False


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def read_npy_header(fh) -> dict:
    """{"shape", "descr", "fortran_order"} of an .npy stream positioned at its start; the stream is left at the data."""
    if fh.read(6) != NPY_MAGIC:
        raise ValueError("not an .npy stream")
    major = fh.read(2)[0]
    size = struct.unpack("<H", fh.read(2))[0] if major == 1 else struct.unpack("<I", fh.read(4))[0]
    header = ast.literal_eval(fh.read(size).decode("latin1"))
    return {"shape": tuple(header["shape"]), "descr": header["descr"], "fortran_order": header["fortran_order"]}


def npz_header(path: Path, member: str = "arr_0.npy") -> dict:
    """The header of one array of an .npz (numpy.savez_compressed's first array is arr_0)."""
    with zipfile.ZipFile(path) as zf, zf.open(member) as fh:
        return read_npy_header(fh)


def npy_header(path: Path) -> dict:
    with open(path, "rb") as fh:
        return read_npy_header(fh)


def idx_counts(path: Path) -> tuple[int, int]:
    """(entries, true entries) of a boolean .npy mask."""
    with open(path, "rb") as fh:
        header = read_npy_header(fh)
        data = fh.read()
    if header["descr"] != "|b1" or len(header["shape"]) != 1:
        raise ValueError(f"{path}: not a 1-d boolean mask ({header['descr']}, {header['shape']})")
    if len(data) != header["shape"][0]:
        raise ValueError(f"{path}: {len(data)} bytes of data for {header['shape'][0]} entries (truncated?)")
    return len(data), len(data) - data.count(0)


def loci_rows(paths: list[str]) -> int:
    """Rows of the peaks files (non-blank lines), cached by path, size and mtime."""
    total = 0
    for path in paths:
        st = os.stat(path)
        key = (path, st.st_size, st.st_mtime_ns)
        if key not in _rows_cache:
            with open(path, "rb") as fh:
                _rows_cache[key] = sum(1 for line in fh if line.strip())
        total += _rows_cache[key]
    return total


# ---------------------------------------------------------------------------------------------------------------------
# the engine pin


def engine_hashes(engine_dir: Path) -> dict[str, str | None]:
    """sha256 of each engine file, as attribute_fast.py's meta records it (None for a missing file)."""
    out: dict[str, str | None] = {}
    for name in CODE_FILES:
        try:
            out[name] = hashlib.sha256((Path(engine_dir) / name).read_bytes()).hexdigest()
        except OSError:
            out[name] = None
    return out


def read_pin(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return None


def _git(engine_dir: Path, *args: str) -> str | None:
    try:
        run = subprocess.run(["git", "-C", str(engine_dir), *args], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return run.stdout.strip() if run.returncode == 0 else None


def write_pin(engine_dir: Path, pin_path: Path, replace: bool = False) -> tuple[str, str]:
    """Sign off the engine in engine_dir: write its files' sha256 and git commit to pin_path.

    Refused when a file is missing, when the engine files differ from the checkout's commit (sign off committed code),
    and when another pin is there already, unless `replace`: a new pin makes every fold made with the old one stale.
    Returns (status, message), status "written", "unchanged" or "refused".
    """
    engine_dir = Path(os.path.abspath(engine_dir))
    hashes = engine_hashes(engine_dir)
    missing = [name for name, h in hashes.items() if h is None]
    if missing:
        return "refused", f"no {', '.join(missing)} in {engine_dir}"
    dirty = _git(engine_dir, "status", "--porcelain", "--", *CODE_FILES)
    if dirty:
        return (
            "refused",
            f"the engine files differ from the checkout's commit:\n{dirty}\ncommit them before signing off",
        )
    old = read_pin(pin_path)
    if old is not None:
        if old.get("files_sha256") == hashes:
            return "unchanged", f"{pin_path} already pins these files (commit {old.get('git_commit')})"
        if not replace:
            return "refused", (
                f"{pin_path} pins other engine files (commit {old.get('git_commit')}); a new pin makes every fold made "
                "with the old engine stale. Pass --replace-pin to write it anyway"
            )
    pin = {
        "engine": "attribute_fast.py",
        "files_sha256": hashes,
        "engine_dir": str(engine_dir),
        "git_commit": _git(engine_dir, "rev-parse", "HEAD"),
        "git_branch": _git(engine_dir, "rev-parse", "--abbrev-ref", "HEAD"),
        "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "written_by": "attr_status.py --write-pin",
    }
    pin_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = pin_path.with_name(f".{pin_path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(pin, indent=2) + "\n")
    os.replace(tmp, pin_path)
    return "written", f"wrote {pin_path}: {engine_dir} at {pin['git_commit']}"


def check_engine(engine_dir: Path, pin_path: Path) -> list[str]:
    """Why the engine in engine_dir does not match the pin; empty when it does."""
    pin = read_pin(pin_path)
    if pin is None:
        return [f"no engine pin at {pin_path}; sign the engine off with attr_status.py --write-pin <engine dir>"]
    hashes = engine_hashes(engine_dir)
    return [
        f"{name} in {engine_dir}: sha256 {(hashes.get(name) or 'missing')[:12]}, the pin has {(want or '?')[:12]}"
        for name, want in (pin.get("files_sha256") or {}).items()
        if hashes.get(name) != want
    ] or ([] if pin.get("files_sha256") else [f"{pin_path} has no files_sha256"])


# ---------------------------------------------------------------------------------------------------------------------
# a fold


def default_pin(fold_dir: Path) -> Path:
    """<root>/attribution_engine.json for a fold at <root>/<dataset>/fold_<k>."""
    return Path(os.path.abspath(fold_dir)).parent.parent / PIN_NAME


def check_fold(fold_dir: Path, pin_path: Path | None = None) -> tuple[str, list[str]]:
    """(state, reasons) by the DONE rule, without looking at the lock: "done" with no reasons, or the first of
    missing, stale, failed that applies, with every failing clause."""
    fold_dir = Path(fold_dir)
    json_path = fold_dir / "attribute.json"
    try:
        params = json.loads(json_path.read_text())
    except FileNotFoundError:
        return "missing", [f"no {json_path}"]
    files = fold_files(params)
    model, _ = model_files(params)
    model_mtime = _mtime(model)
    if model_mtime is None:
        return "missing", [f"no model {model}"]
    meta_mtime, stock_mtime = _mtime(files["meta"]), _mtime(files["stock"])
    use_stock = stock_mtime is not None and (meta_mtime is None or stock_mtime > meta_mtime)
    kind = "stock" if use_stock else "meta"
    record_path = files[kind]
    record_mtime = stock_mtime if use_stock else meta_mtime
    needed = [files["ohe"], files["attr"], files["idx"]] + ([] if use_stock else [files["deltas"]])
    reasons = [f"missing or empty: {p}" for p in needed if not _nonempty(p)]
    if record_mtime is None:
        reasons.append(f"no {record_path}")
    if reasons:
        return "missing", reasons
    try:
        record = json.loads(record_path.read_text())
    except (OSError, ValueError) as exc:
        return "missing", [f"unreadable {record_path}: {exc}"]

    stale, failed = [], []
    # freshness: a refit, or outputs rewritten after the record
    if record_mtime <= model_mtime:
        stale.append(f"the model {model.name} is newer than {record_path.name} (refit since the attribution)")
    newer = [p.name for p in needed if (_mtime(p) or 0) > record_mtime]
    if newer:
        stale.append(f"{', '.join(newer)} newer than {record_path.name} (an interrupted rerun?)")
    # the parameters the numbers depend on
    run_params = record.get("parameters") or {}
    for key in RESULT_KEYS:
        if run_params.get(key) != params.get(key):
            stale.append(f"{kind} parameters.{key} is {run_params.get(key)!r}, attribute.json has {params.get(key)!r}")
    # the files themselves
    width, n_shuffles = params.get("attr_window"), params.get("n_shuffles")
    try:
        n_rows = loci_rows(list(params["loci"]))
    except OSError as exc:
        return "failed", [*stale, f"cannot read the peaks file: {exc}"]
    try:
        n_loci, n_kept = idx_counts(files["idx"])
        attr, ohe = npz_header(files["attr"]), npz_header(files["ohe"])
        deltas = None if use_stock else npy_header(files["deltas"])
    except (OSError, ValueError, KeyError, zipfile.BadZipFile, SyntaxError, IndexError, struct.error) as exc:
        return "failed", [*stale, f"unreadable output: {exc}"]
    if n_loci != n_rows:
        stale.append(f"the idx mask has {n_loci} entries, the peaks file {n_rows} rows")
    if attr["shape"] != (n_kept, 4, width) or attr["descr"] != "<f4":
        failed.append(f"the attr npz holds {attr['descr']} {attr['shape']}, not <f4 {(n_kept, 4, width)}")
    if ohe["shape"] != (n_kept, 4, width) or ohe["descr"] != "|i1":
        failed.append(f"the ohe npz holds {ohe['descr']} {ohe['shape']}, not |i1 {(n_kept, 4, width)}")

    if use_stock:
        if record.get("engine") != "stock" or record.get("exit") != 0:
            failed.append(f"{record_path.name}: engine {record.get('engine')!r}, exit {record.get('exit')!r}")
        if record.get("attr_finite") is not True:
            failed.append(f"{record_path.name}: attr_finite is {record.get('attr_finite')!r}")
    else:
        for key in ("random_state", "n_shuffles"):
            if record.get(key) != params.get(key):
                stale.append(f"meta {key} is {record.get(key)!r}, attribute.json has {params.get(key)!r}")
        outs, inputs = record.get("outputs") or {}, record.get("inputs") or {}
        for key, name in (("ohe", "ohe_filename"), ("attr", "attr_filename"), ("idx", "idx_filename")):
            if outs.get(key) != params.get(name):
                stale.append(f"meta outputs.{key} is {outs.get(key)!r}, attribute.json has {params.get(name)!r}")
        if inputs.get("loci") != n_rows:
            stale.append(f"inputs.loci is {inputs.get('loci')!r}, the peaks file has {n_rows} rows")
        stale += _pin_clause(record, pin_path if pin_path is not None else default_pin(fold_dir))
        audit = record.get("audit") or {}
        if audit.get("pass") is not True:
            failed.append(f"audit.pass is {audit.get('pass')!r} ({audit.get('status', 'failed')})")
        if outs.get("attr_finite") is not True:
            failed.append(f"outputs.attr_finite is {outs.get('attr_finite')!r}")
        if (record.get("deltas") or {}).get("finite") is not True:
            failed.append(f"deltas.finite is {(record.get('deltas') or {}).get('finite')!r}")
        kept = inputs.get("kept")
        if outs.get("attr_shape") != [kept, 4, width]:
            failed.append(
                f"outputs.attr_shape is {outs.get('attr_shape')!r}, not [inputs.kept, 4, W] = [{kept}, 4, {width}]"
            )
        if n_kept != kept:
            failed.append(f"the idx mask keeps {n_kept}, inputs.kept is {kept!r}")
        if deltas["shape"] != (n_kept, n_shuffles):
            failed.append(f"the deltas hold {deltas['shape']}, not {(n_kept, n_shuffles)}")
    if stale:
        return "stale", stale + failed
    if failed:
        return "failed", failed
    return "done", []


def _pin_clause(meta: dict, pin_path: Path) -> list[str]:
    pin = read_pin(pin_path)
    if pin is None:
        return [f"no engine pin at {pin_path} (attr_status.py --write-pin <engine dir>)"]
    ran = (meta.get("code") or {}).get("files_sha256")
    if not pin.get("files_sha256") or ran != pin["files_sha256"]:
        return [f"code.files_sha256 differs from the pin {pin_path} (commit {pin.get('git_commit')})"]
    return []


def lock_held(fold_dir: Path) -> bool:
    """Whether a run holds the fold's .task.lock (run_task.sh's flock). The probe takes a shared lock for an instant;
    a runner's `flock -n` in that instant skips the fold (exit 4) and a later pass takes it."""
    path = Path(fold_dir) / LOCK_NAME
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
            return True
        raise
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def fold_state(fold_dir: Path, pin_path: Path | None = None, locks: bool = True) -> tuple[str, list[str]]:
    """The fold's state for listing: done, else running (lock held), else waiting (fit not done), else check_fold's."""
    state, reasons = check_fold(fold_dir, pin_path)
    if state == "done" or not (Path(fold_dir) / "attribute.json").is_file():
        return state, reasons
    if locks and lock_held(fold_dir):
        return "running", [f"{LOCK_NAME} is held"]
    params = json.loads((Path(fold_dir) / "attribute.json").read_text())
    model, performance = model_files(params)
    if not (_nonempty(model) and _nonempty(performance)):
        return "waiting", [f"the fit is not done ({model.name} and {performance.name})"]
    return state, reasons


def list_folds(root: Path, only: list[str] | None) -> list[Path]:
    """Every <root>/<dataset>/fold_<k> with a fit.json (make_jobs.py's folds), optionally of some datasets only."""
    folds = sorted(p.parent for p in Path(root).glob("*/fold_*/fit.json"))
    if only:
        folds = [f for f in folds if f.parent.name in only]
    return folds


def _label(fold_dir: Path) -> tuple[str, str]:
    fold_dir = Path(os.path.abspath(fold_dir))
    return fold_dir.parent.name, fold_dir.name.removeprefix("fold_")


# ---------------------------------------------------------------------------------------------------------------------
# command line


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folds", nargs="*", type=Path, help="fold directories (with attribute.json)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", type=Path, metavar="FOLD", help="exit 0 if the fold is DONE, else 1")
    mode.add_argument("--check-engine", type=Path, metavar="DIR", help="exit 0 if DIR's engine matches the pin")
    mode.add_argument("--write-pin", type=Path, metavar="DIR", help="pin the engine in DIR (sign-off)")
    mode.add_argument("--files", type=Path, metavar="FOLD", help="print the fold's attribution files, one per line")
    ap.add_argument("--root", type=Path, help="make_jobs.py's --out-root: list its folds, and hold the pin")
    ap.add_argument("--only", action="append", default=[], help="datasets to list (space-separated; repeatable)")
    ap.add_argument("--pin", type=Path, help=f"the pin (default: <root>/{PIN_NAME})")
    ap.add_argument("--replace-pin", action="store_true", help="with --write-pin: replace a pin of other files")
    ap.add_argument("--summary", action="store_true", help="count the states per dataset")
    ap.add_argument("--settled", action="store_true", help="exit 0 only if every fold listed is done or failed")
    ap.add_argument("--no-locks", action="store_true", help="do not probe the folds' locks")
    args = ap.parse_args(argv)

    if args.write_pin or args.check_engine:
        if args.pin is None and args.root is None:
            ap.error("--write-pin and --check-engine need --pin or --root")
        pin_path = args.pin or Path(args.root) / PIN_NAME
        if args.write_pin:
            status, message = write_pin(args.write_pin, pin_path, args.replace_pin)
            print(f"{status}: {message}", file=sys.stderr if status == "refused" else sys.stdout)
            return 1 if status == "refused" else 0
        problems = check_engine(args.check_engine, pin_path)
        for line in problems:
            print(line, file=sys.stderr)
        return 1 if problems else 0

    if args.files:
        params = json.loads((args.files / "attribute.json").read_text())
        print("\n".join(str(f) for f in fold_files(params).values()))
        return 0

    if args.check:
        state, reasons = check_fold(args.check, args.pin)
        dataset, fold = _label(args.check)
        print(f"{dataset}\tfold_{fold}\t{state}\t{'; '.join(reasons)}")
        return 0 if state == "done" else 1

    only = [name for item in args.only for name in item.split()]
    folds = list(args.folds)
    if args.root:
        folds += list_folds(args.root, only)
    if not folds:
        ap.error("name fold directories, --root, or --check")
    pin = args.pin or (Path(args.root) / PIN_NAME if args.root else None)
    rows = []
    for fold_dir in folds:
        state, reasons = fold_state(fold_dir, pin, locks=not args.no_locks)
        rows.append((*_label(fold_dir), state, "; ".join(reasons)))
    if args.summary:
        per: dict[str, dict[str, int]] = {}
        for dataset, _, state, _ in rows:
            per.setdefault(dataset, dict.fromkeys(STATES, 0))[state] += 1
        print("dataset\tfolds\t" + "\t".join(STATES))
        for dataset, counts in per.items():
            print(f"{dataset}\t{sum(counts.values())}\t" + "\t".join(str(counts[s]) for s in STATES))
        totals = {s: sum(c[s] for c in per.values()) for s in STATES}
        print(f"total\t{len(rows)}\t" + "\t".join(str(totals[s]) for s in STATES))
    else:
        for dataset, fold, state, reason in rows:
            print(f"{dataset}\tfold_{fold}\t{state}\t{reason}")
    if args.settled:
        return 0 if all(state in ("done", "failed") for _, _, state, _ in rows) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
