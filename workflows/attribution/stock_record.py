#!/usr/bin/env python3
"""Write the record attr_status.py reads for a fold that `cherimoya attribute` attributed.

`cherimoya attribute` (Cherimoya >= 0.3.1, whose fast DeepLIFT/SHAP path is this repo's engine upstream) writes the
ohe, attr and idx files attribute.json names, and nothing that says it finished, what ran, or whether its fast path
held. attr_status.py accepts such a fold through a record beside the attr file, <attr minus .npz>.stock.json, and
checks the files against attribute.json itself; this script writes that record once the run has exited 0:

  {"engine": "stock", "tool": "cherimoya attribute", "exit": 0, "attr_finite": true,
   "parameters": <attribute.json as written>, "versions": {"cherimoya": ..., "tangermeme": ..., "torch": ...},
   "fast_path": true, "fallback": null, "convergence_warnings": 0, "written": "<UTC time>"}

- parameters is the JSON file's own text, parsed. Cherimoya rewrites "counts" to "logcounts" in its merged
  parameters; attr_status.py compares the record's parameters with attribute.json, and the same file must match.
- fast_path is false when the log has Cherimoya's fallback warning ("deep_lift_shap is using the default path because
  the fast path ..."): the fast path disagreed with tangermeme's on the first sequences, or failed, and tangermeme's
  path attributed the rest. Those attributions are still valid (slower, not different), so the record is written;
  the reason is kept in fallback. Any other RuntimeWarning (e.g. convergence deltas over the threshold) is counted,
  not treated as a fallback.
- attr_finite is checked on the attr npz itself; a non-finite attribution writes no record and exits 1.

Usage: stock_record.py --fold <fold dir> --log <fold dir>/attribute.log --exit <cherimoya attribute's exit status>
Input: the fold's attribute.json and the files it names; the run's log
Output: <attr minus .npz>.stock.json (attr_status.fold_files()["stock"]); exit 0 when written, 1 when refused
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import attr_status  # noqa: E402

FALLBACK = "deep_lift_shap is using the default path because the fast path"
PACKAGES = ("cherimoya", "tangermeme", "torch", "numpy")


def read_log(path: Path | None) -> tuple[str | None, int]:
    """(the fallback warning's line, or None; the number of other RuntimeWarning lines) in the run's log."""
    if path is None or not path.exists():
        return None, 0
    fallback, other = None, 0
    for line in path.read_text(errors="replace").splitlines():
        if FALLBACK in line:
            fallback = line.strip()
        elif "RuntimeWarning" in line:
            other += 1
    return fallback, other


def attr_is_finite(path: Path, block: int = 4096) -> bool:
    import numpy as np

    with np.load(path) as z:
        a = z["arr_0"]
    return all(bool(np.isfinite(a[i : i + block]).all()) for i in range(0, len(a), block))


def versions() -> dict[str, str | None]:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def write_record(fold: Path, log: Path | None, exit_status: int) -> tuple[Path | None, str]:
    """Write the fold's stock record; (its path, "") when written, (None, why) when refused."""
    json_path = fold / "attribute.json"
    params = json.loads(json_path.read_text())
    if exit_status != 0:
        return None, f"cherimoya attribute exited {exit_status}"
    cwd = os.getcwd()
    os.chdir(fold)  # attribute.json's paths are relative to the fold, as cherimoya ran them
    try:
        files = attr_status.fold_files(params)
        missing = [str(files[k]) for k in ("ohe", "attr", "idx") if not attr_status._nonempty(files[k])]
        if missing:
            return None, f"missing or empty: {', '.join(missing)}"
        if not attr_is_finite(files["attr"]):
            return None, f"{files['attr']} holds non-finite attributions"
        fallback, other = read_log(log)
        record = {
            "engine": "stock",
            "tool": "cherimoya attribute",
            "exit": 0,
            "attr_finite": True,
            "parameters": params,
            "versions": versions(),
            "fast_path": fallback is None,
            "fallback": fallback,
            "convergence_warnings": other,
            "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        out = files["stock"]
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(json.dumps(record, indent=1) + "\n")
        os.replace(tmp, out)
        return (fold / out) if not out.is_absolute() else out, ""
    finally:
        os.chdir(cwd)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", type=Path, required=True, help="the fold directory, holding attribute.json")
    ap.add_argument("--log", type=Path, help="the run's log (attribute.log), read for the fallback warning")
    ap.add_argument("--exit", type=int, required=True, dest="exit_status", help="cherimoya attribute's exit status")
    args = ap.parse_args(argv)
    out, why = write_record(args.fold.resolve(), args.log.resolve() if args.log else None, args.exit_status)
    if out is None:
        print(f"ERROR: {args.fold}: no stock record written: {why}", file=sys.stderr)
        return 1
    record = json.loads(out.read_text())
    note = "fast path" if record["fast_path"] else f"FELL BACK to tangermeme's path: {record['fallback']}"
    print(f"{args.fold}: wrote {out.name} ({note}; cherimoya {record['versions']['cherimoya']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
