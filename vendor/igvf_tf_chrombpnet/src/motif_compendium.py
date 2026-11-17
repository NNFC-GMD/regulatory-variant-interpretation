#!/usr/bin/env python3
"""
motif_compendium.py
Build a non-redundant motif compendium from per-dataset MoDISco H5 files
using MotifCompendium (Kundaje lab).

Called by 09.0.cross_dataset_compendium.sh, which pools the datasets it is
given (a list, COMPENDIUM_INPUTS, or the four collaboration datasets) and
whose modisco_compiled.h5 is what 10.0 calls hits with, and by the superseded
deprecated/motif_compendium.sh (per dataset).

The clustering algorithm is passed explicitly (--algorithm, default
cpm_leiden), because MotifCompendium v1.0.19 changed cluster()'s default from
"cpm_leiden" to ["cpm_leiden", "k_centroids"]. The k-centroids pass then
reassigns every motif to its nearest cluster average without looking at
similarity_threshold, and at its default n_iterations=-1 repeats until no
motif moves, with no iteration cap. Taking the new default would silently
change what --threshold means; cpm_leiden alone is what v1.0.16 ran.

The input table is checked before anything is loaded, with the rules 09.0
applies to its list (lib/python/utils/compendium_inputs.py). A label may hold
only letters, digits and '_': MotifCompendium names each motif
<label>-<pos|neg>.pattern_<n> and assumes no '-' or '.' in a label. Labels
and paths must be unique; a repeated label used to replace the earlier row
silently. A missing H5 is an error, unless --allow-missing, which drops it
with a warning as this script always did.

--modisco-region-width is the window TF-MoDISco ran with (`modisco motifs
-w`). MotifCompendium uses it for one metadata column only,
avg_dist_from_summit. Unset, MotifCompendium's own default (400) applies, as
before. "auto" reads the window_size attribute that modisco 2.5 writes at the
root of each results file (08.0 runs -w 500): the inputs must agree, and
when none records one, MotifCompendium's default applies. A number is used
as given and must agree with every input that records a window.

Input:  2-column TSV (-c): label <TAB> /path/to/modisco_counts_results.h5
        ('#' lines and blank lines skipped)
Output (inside --out-dir):
  modisco_compiled.h5          - cluster-averaged motifs in MoDISco format, for Fi-NeMo (10.0)
  modisco_compendium.meme      - same cluster-averaged motifs in MEME format
  modisco_compendium.mc        - pickled MotifCompendium object (for inspection)
  modisco_compendium_meta.tsv  - per-motif metadata with TF annotations + cluster IDs
  compendium.json              - provenance, written last: per input its label, path, size,
                                 mtime, md5, MoDISco window and pattern counts; the settings;
                                 the reference database and its md5; MotifCompendium's
                                 version and commit; this checkout's commit; the motif and
                                 cluster counts; each output's md5

Usage:
  python src/motif_compendium.py \\
      -c  results/compendium/modisco_compiled/modisco_config.tsv \\
      -o  results/compendium/modisco_compiled \\
      --ref-db /path/to/MotifCompendium-1.0.19-Database-Human.meme.txt \\
      --threshold 0.95 \\
      --algorithm cpm_leiden \\
      --modisco-region-width auto \\
      --cpus 16

Prerequisites:
  This repo's motif-compendium pixi environment (MotifCompendium v1.0.19,
  CPU only): `pixi install -e motif-compendium`. The steps enter it with
  activate_env "${motif_compendium_env}".
"""

import argparse
import inspect
import json
import os
import pickle
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make lib/python importable without an install step (works under the cluster
# conda envs, under pixi, and under a bare python).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib" / "python"))

# MotifCompendium itself is imported in main(), after the inputs are checked,
# so the checks (and tests/test_compendium_steps.py) need only h5py.
from utils import compendium_inputs, log, metadata  # noqa: E402

logger = log.get_logger(__name__)

# What a finished build holds, besides the provenance file written last.
OUTPUTS = {
    "compiled_h5": "modisco_compiled.h5",
    "meme": "modisco_compendium.meme",
    "mc": "modisco_compendium.mc",
    "meta_tsv": "modisco_compendium_meta.tsv",
}
PROVENANCE = "compendium.json"


def region_width_arg(text):
    """--modisco-region-width: 'auto' or a positive integer."""
    if text == "auto":
        return text
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value <= 0:
        raise argparse.ArgumentTypeError(f"expected 'auto' or a positive integer, got {text!r}")
    return value


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Build a non-redundant motif compendium using MotifCompendium."
    )
    p.add_argument(
        "-c",
        "--config",
        required=True,
        help="2-column TSV: label <TAB> modisco_h5_path",
    )
    p.add_argument(
        "-o",
        "--out-dir",
        required=True,
        help="Output directory (modisco_compiled.h5 written here for FiNeMo)",
    )
    p.add_argument(
        "--ref-db",
        required=True,
        help="Reference motif database in MEME format for annotation (the steps pass ${ref_db_meme}, MotifCompendium-1.0.19-Database-Human.meme.txt)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Similarity threshold for Leiden clustering (default: 0.95)",
    )
    p.add_argument(
        "--algorithm",
        nargs="+",
        default=["cpm_leiden"],
        metavar="NAME",
        help=(
            "MotifCompendium clustering algorithm(s), run in order, each starting "
            "from the previous one's clusters (default: cpm_leiden, as in "
            "MotifCompendium v1.0.16). v1.0.19's own default, cpm_leiden "
            "k_centroids, adds a k-centroids pass that ignores --threshold."
        ),
    )
    p.add_argument(
        "--cpus",
        type=int,
        default=8,
        help="CPUs for pairwise similarity computation (default: 8)",
    )
    p.add_argument(
        "--min-annotation-score",
        type=float,
        default=0.7,
        help="Min similarity to reference database for annotation match (default: 0.7)",
    )
    p.add_argument(
        "--allow-missing",
        action="store_true",
        help="Drop an input whose H5 does not exist, with a warning, instead of stopping",
    )
    p.add_argument(
        "--modisco-region-width",
        type=region_width_arg,
        default=None,
        metavar="auto|N",
        help=(
            "The window TF-MoDISco ran with (modisco motifs -w), for MotifCompendium's "
            "avg_dist_from_summit: 'auto' reads each input's window_size attribute, N is "
            "used as given (default: MotifCompendium's own default)"
        ),
    )
    log.add_logging_args(p)
    return p.parse_args(argv)


def load_config(config_path, allow_missing=False):
    """(label -> H5 path in the table's order, the rows dropped as missing).

    Stops naming every problem: a bad or repeated label, a repeated path, a
    row that is not an H5, and a missing H5 unless allow_missing.
    """
    try:
        rows = compendium_inputs.read(config_path, kinds=("h5",))
    except compendium_inputs.InputListError as exc:
        sys.exit(f"ERROR: {exc}")
    present, missing = compendium_inputs.split_missing(rows)
    if missing and not allow_missing:
        lines = "\n".join(f"  {r.label}: {r.path}" for r in missing)
        sys.exit(
            f"ERROR: {len(missing)} MoDISco H5(s) listed in {config_path} not found:\n{lines}\n"
            "(--allow-missing drops them with a warning instead)"
        )
    for row in missing:
        logger.warning(f"  [WARN] {row.label}: H5 not found, skipping: {row.path}")
    return {r.label: r.path for r in present}, missing


def _utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


def describe_input(label, path) -> dict:
    """compendium.json's record of one input: the file, its MoDISco window and pattern counts."""
    import h5py

    p = Path(path)
    st = p.stat()
    with h5py.File(p, "r") as f:
        window = f.attrs.get("window_size")
        n_pos = len(f["pos_patterns"]) if "pos_patterns" in f else 0
        n_neg = len(f["neg_patterns"]) if "neg_patterns" in f else 0
    try:
        window = None if window is None else int(window)
    except (TypeError, ValueError):
        sys.exit(
            f"ERROR: {label}: {path} has a window_size attribute that is not a number: {window!r}"
        )
    return {
        "label": label,
        "filepath": os.path.abspath(path),
        "file_size": st.st_size,
        "mtime": _utc(st.st_mtime),
        "md5sum": metadata.md5sum(p),
        "window_size": window,
        "n_pos_patterns": int(n_pos),
        "n_neg_patterns": int(n_neg),
    }


def region_width(requested, inputs, default):
    """(the width to pass to build_from_modisco, or None for its default; how it was chosen).

    requested is None, "auto" or an int; inputs are describe_input() records;
    default is MotifCompendium's own default, for the log.
    """
    recorded = {d["label"]: d["window_size"] for d in inputs if d["window_size"] is not None}
    unrecorded = [d["label"] for d in inputs if d["window_size"] is None]
    if requested is None:
        return None, "MotifCompendium's default"
    if requested == "auto":
        values = sorted(set(recorded.values()))
        if len(values) > 1:
            by_width = "; ".join(
                f"{w}: {', '.join(lbl for lbl, x in recorded.items() if x == w)}" for w in values
            )
            sys.exit(
                "ERROR: the inputs were made with different MoDISco windows, and MotifCompendium "
                f"takes one per build ({by_width})"
            )
        if not values:
            logger.warning(
                f"  [WARN] no input records its MoDISco window (window_size); using "
                f"MotifCompendium's default, {default}"
            )
            return None, "MotifCompendium's default (no input records window_size)"
        if unrecorded:
            logger.warning(
                f"  [WARN] {', '.join(unrecorded)} record(s) no window_size; assuming {values[0]}, "
                "the window the others record"
            )
        return values[0], "the inputs' window_size"
    disagree = {lbl: w for lbl, w in recorded.items() if w != requested}
    if disagree:
        listed = ", ".join(f"{lbl} ({w})" for lbl, w in disagree.items())
        sys.exit(
            f"ERROR: --modisco-region-width {requested}, but these inputs were made with another "
            f"window: {listed}"
        )
    return requested, "--modisco-region-width"


def _write_json_atomic(path: Path, record: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(tmp, path)


def main(argv=None):
    args = parse_args(argv)
    log.setup_from_args(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {key: out_dir / name for key, name in OUTPUTS.items()}
    h5_out, meme_out, mc_out, meta_out = (
        str(paths[k]) for k in ("compiled_h5", "meme", "mc", "meta_tsv")
    )

    # 1. Load MoDISco H5s: the table's rules, then each input's file, window and patterns
    modisco_dict, dropped = load_config(args.config, args.allow_missing)
    logger.info(f"@ loaded {len(modisco_dict)} MoDISco H5 paths: {list(modisco_dict.keys())}")
    if not modisco_dict:
        sys.exit("ERROR: no valid MoDISco H5 files found in config.")
    inputs = [describe_input(label, path) for label, path in modisco_dict.items()]

    import MotifCompendium
    import MotifCompendium.utils.analysis as utils_analysis

    width_param = inspect.signature(MotifCompendium.build_from_modisco).parameters.get(
        "modisco_region_width"
    )
    default_width = None if width_param is None else width_param.default
    width, width_source = region_width(args.modisco_region_width, inputs, default_width)
    if width is not None and width_param is None:
        sys.exit(
            "ERROR: this MotifCompendium's build_from_modisco takes no modisco_region_width; "
            "run without --modisco-region-width"
        )
    build_kwargs = {} if width is None else {"modisco_region_width": width}
    logger.info(
        f"@ MoDISco region width {width if width is not None else default_width} ({width_source})"
    )

    # 2. Build MotifCompendium (reads pos_patterns + neg_patterns from each H5)
    MotifCompendium.set_compute_options(max_cpus=args.cpus, progress_bar=True)
    logger.info("@ building MotifCompendium from MoDISco H5s...")
    mc = MotifCompendium.build_from_modisco(modisco_dict, **build_kwargs)
    logger.info(f"@ {len(mc.metadata)} motifs loaded across all datasets")

    # 3. Annotate individual motifs against reference database before clustering
    logger.info(f"@ annotating against reference database ({args.ref_db})...")
    utils_analysis.assign_label_from_pfms(
        mc,
        pfm_file=args.ref_db,
        save_col_prefix="annotation",
        min_score=args.min_annotation_score,
        save_images=False,
    )
    logger.info("@ annotation done")

    # 4. Cluster at similarity threshold (Leiden community detection)
    cluster_col = "cluster_id"
    logger.info(
        f"@ clustering with {' -> '.join(args.algorithm)} at similarity threshold "
        f"{args.threshold}..."
    )
    mc.cluster(
        algorithm=args.algorithm,
        similarity_threshold=args.threshold,
        save_name=cluster_col,
    )
    n_clusters = mc.metadata[cluster_col].nunique()
    logger.info(f"@ {len(mc.metadata)} motifs → {n_clusters} clusters (threshold={args.threshold})")

    # 5. Export clustered modisco H5 for Fi-NeMo (10.0)
    #    One averaged pattern per cluster, written to modisco_compiled.h5, the
    #    name 10.0.run_finemo_unified.sh reads.
    logger.info(f"@ exporting clustered modisco H5 → {h5_out}")
    utils_analysis.export_compendium_clustered_modisco(mc, cluster_col, h5_out)
    logger.info(f"@ modisco_compiled.h5 written ({n_clusters} patterns)")

    # 6. Export MEME format — cluster averages, matching the H5 content
    logger.info(f"@ exporting MEME → {meme_out}")
    mc_avg = mc.cluster_averages(clustering=cluster_col, aggregations=[], weight_col=None)
    utils_analysis.export_compendium_meme(mc_avg, meme_out, name_col="source_cluster")

    # 7. Save MotifCompendium object (for interactive inspection / re-analysis)
    logger.info(f"@ saving MotifCompendium object → {mc_out}")
    with open(mc_out, "wb") as fh:
        pickle.dump(mc, fh)

    # 8. Save full metadata TSV (motif names, source dataset, TF annotations,
    #    cluster IDs)
    mc.metadata.to_csv(meta_out, sep="\t", index=False)
    logger.info(f"@ metadata TSV → {meta_out} ({len(mc.metadata)} rows)")

    # 9. Provenance, written last: a compendium.json says the build finished
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as dist_version

    try:
        mc_version = dist_version("MotifCompendium")
    except PackageNotFoundError:
        mc_version = None
    git = metadata.git_info()
    per_label = mc.metadata["model"].value_counts()
    record = {
        "schema_version": 1,
        "date_created": _utc(datetime.now(timezone.utc).timestamp()),
        "inputs": inputs,
        "dropped_missing": [
            {"label": r.label, "filepath": os.path.abspath(r.path)} for r in dropped
        ],
        "settings": {
            "threshold": args.threshold,
            "algorithm": list(args.algorithm),
            "min_annotation_score": args.min_annotation_score,
            "modisco_region_width": width if width is not None else default_width,
            "modisco_region_width_requested": args.modisco_region_width,
            "modisco_region_width_source": width_source,
            "cpus": args.cpus,
        },
        "ref_db": {
            "filepath": os.path.abspath(args.ref_db),
            "md5sum": metadata.md5sum(args.ref_db),
        },
        "software": {
            "python": platform.python_version(),
            "MotifCompendium": mc_version,
            "MotifCompendium_commit": metadata.source_revision("MotifCompendium"),
        },
        "pipeline": {k: git.get(k) for k in ("commit", "branch", "dirty", "remote_url")},
        "results": {
            "n_motifs": int(len(mc.metadata)),
            "n_clusters": int(n_clusters),
            "n_motifs_per_label": {str(k): int(v) for k, v in per_label.items()},
            "outputs": {
                key: {
                    "file": path.name,
                    "file_size": path.stat().st_size,
                    "md5sum": metadata.md5sum(path),
                }
                for key, path in paths.items()
            },
        },
    }
    _write_json_atomic(out_dir / PROVENANCE, record)
    logger.info(f"@ provenance → {out_dir / PROVENANCE}")

    logger.info(
        f"\n@ done: {n_clusters} distinct motifs from "
        f"{len(mc.metadata)} input patterns across "
        f"{len(modisco_dict)} datasets"
    )
    logger.info(f"  FiNeMo input : {h5_out}")
    logger.info(f"  Annotations  : {meta_out}")


if __name__ == "__main__":
    main()
