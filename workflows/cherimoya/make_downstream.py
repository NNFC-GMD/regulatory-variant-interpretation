#!/usr/bin/env python3
"""Write the inputs of Cherimoya's downstream steps: per-fold attribution JSONs, per-dataset pipeline configs and the
compendium input lists.

After training, Cherimoya's folds go through the same downstream tools as ChromBPNet's: DeepLIFT/SHAP on the counts
head over every peak, the fold average, TF-MoDISco, the cross-dataset compendium and Fi-NeMo. The attributions come
from attribute_fast.py (`cherimoya attribute`, computed by cheri_dls and audited); the rest runs igvf_tf_chrombpnet's
own 06.0-11.0 step files on the converted scores. This script writes what those runs read; it computes nothing.

For every dataset and fold under --cheri-root (make_jobs.py's --out-root):

  <dataset>/fold_<k>/attribute.json   `attribute_fast.py -p attribute.json` (or run_task.sh's `attribute` stage),
                                      run from the fold's directory. Built from the fold's fit.json:
      model      <fold>/cherimoya.fold_<k>.torch, the best-validation checkpoint compare.json scores;
      loci       fit.json's one peaks file, the 00.1 narrowPeak ChromBPNet's 05.0 also scores;
      chroms     the union of fit.json's training, validation and test chromosomes. Every peak's chromosome
                 must be in it: extract_loci drops rows on unlisted chromosomes before it builds the mask, so
                 the idx file then has one entry per narrowPeak row. Cherimoya's default list is the training
                 and validation chromosomes only.
      window     in_window 2114 and attr_window W (default 1000): [1057 - W/2, 1057 + W/2) of the input,
                 [summit - W/2, summit + W/2). 1000 is the narrowest window whose crops hold the bases 08.0
                 (modisco -w 500) and 10.0 (Fi-NeMo --region-width 1000) read from ChromBPNet's scores.
      seeds      random_state --seed-base + --seed-stride x fold. tangermeme seeds reference k of every peak with
                 random_state + k, so folds whose seeds are closer than n_shuffles would share references and
                 the fold average would stop averaging out reference noise. The stride must be >= n_shuffles.
      n_shuffles 20, checked against ChromBPNet's interpretation.interpret.args.json for the fold where it
                 exists (num_shuffles).
      batch_size 64, the stock command's default. attribute_fast.py uses it only for the engine's head_rows
                 (2 x batch_size, the 128 rows the validated runs used) and caps its audit's stock batch at 20.
                 A stock `cherimoya attribute` rerun of this JSON (the fallback after a failed audit) needs
                 batch_size 20 instead: stock at 64 allocates about 15.5 GB on these models, 8 GB is the limit
                 on the shared GPUs.
      outputs    attr.ohe.npz, attr.attr.npz and attr.idx.npy in the fold directory; attribute_fast.py adds
                 attr.attr.deltas.npy and attr.attr.meta.json. A refit's cleanup (cherimoya.fold_<k>.*) does not
                 reach them; attr_status.py calls them stale once the model is newer than the meta.
      "fast"     precision reference, mem_budget_gb 7.5 (about 7 GB of GPU memory at 8 peaks per step),
                 ref_workers 3 (4 cores per process), audit 8 (each run re-checked against the stock
                 deep_lift_shap on 8 peaks). The stock command ignores the block.
  configs/<dataset>/config.yaml       an igvf_tf_chrombpnet dataset config for 06.0-11.0 on the converted scores
                                      (DATASET_CONFIG=). output_dir is <cheri-root>/<dataset>/downstream, a tree of
                                      its own, so no step skips on ChromBPNet's outputs. signal_path, assay,
                                      peak_type, folds and modisco_max_seqlets are the ChromBPNet config's;
                                      config.sh requires signal_path and regions, which 06-11 never read.
                                      contrib_heads is ["counts"], the head attributed. There is no
                                      fold_bias_suffix, so 03.x-05.0 fail fast if run with it, and no
                                      full_model_precision/optimizer/patience, which would rename full_models/.

and under --compendia/inputs/, the lists the runners, the compendia and Fi-NeMo read. The number in each name is
its row count: with S datasets in --shared and D configs,

  shared<S>.txt  the datasets of --shared, one per line, in its order (both models' comparison set)
  rest<D-S>.txt  every other dataset with a ChromBPNet config, sorted
  cbp<S>.tsv     cbp_<dataset><TAB><ChromBPNet config.yaml>, for the --shared datasets
  cheri<S>.tsv   cheri_<dataset><TAB><Cherimoya config.yaml>, for the --shared datasets
  cheri<D>.tsv   cheri<S>'s rows, then the other datasets' (every dataset)
  joint<2S>.tsv  cbp<S>'s rows, then cheri<S>'s

Labels match ^[A-Za-z0-9_]+$ and are unique within a list. Paths are absolute, as given (symlinks are kept).

Every file is written through a temporary file and os.replace, and only when its content changes. fit.json,
compare.json and tasks.tsv are never touched. An attribute.json that would change while its outputs exist (an
attribution ran with it) is refused unless --replace-used is given.

    python workflows/cherimoya/make_downstream.py \\
        --cheri-root <make_jobs.py --out-root> \\
        --chrombpnet-configs <configs>/*/config.yaml \\
        --shared <datasets.txt> --compendia <dir> \\
        [--attr-window 1000 --seed-base 1234 --seed-stride 100]

Stdlib and PyYAML (make_jobs.py's config reader); light enough for any node, but it reads every peaks file once.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_jobs  # noqa: E402

logger = logging.getLogger("make_downstream")

IN_WINDOW = 2114
OUTPUT_NAMES = {"ohe_filename": "attr.ohe.npz", "attr_filename": "attr.attr.npz", "idx_filename": "attr.idx.npy"}
"""The outputs of attribute_fast.py in each fold directory; its sidecars are attr.attr.deltas.npy and .meta.json."""
FOLD_LISTS = ("training_chroms", "validation_chroms", "test_chroms")
LABEL_RE = re.compile(r"^[A-Za-z0-9_]+$")
CONFIG_HEADER = (
    "# Generated by regulatory-variant-interpretation workflows/cherimoya/make_downstream.py; edit that, not this file.\n"
    "# Cherimoya's fold attributions, converted to ChromBPNet's 05.0 layout under output_dir, for igvf_tf_chrombpnet's\n"
    "# 06.0-11.0 (DATASET_CONFIG=<this file>). Never run 00.0-05.0 with it.\n"
)


def chrom_key(name: str) -> tuple:
    """chr1 < chr2 < ... < chr22 < chrX < chrY < others, by name."""
    tail = name[3:] if name.startswith("chr") else name
    return (0, int(tail), "") if tail.isdigit() else (1, 0, tail)


def read_chrombpnet_config(path: Path) -> dict:
    """make_jobs.read_config's fields (dataset_name, output_dir, peak_type, folds), plus the keys the Cherimoya config
    copies: signal_path (variables expanded), assay and modisco_max_seqlets."""
    cfg = make_jobs.read_config(path)
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    signal = os.path.expandvars(str(raw.get("signal_path") or ""))
    if not signal or "$" in signal:
        raise ValueError(f"{path}: no usable signal_path ({raw.get('signal_path')!r})")
    cfg["signal_path"] = signal
    cfg["assay"] = str(raw.get("assay") or "ATAC")
    cfg["modisco_max_seqlets"] = raw.get("modisco_max_seqlets")
    cfg["config"] = os.path.abspath(path)
    return cfg


def scan_peaks(path: Path) -> dict:
    """Row count and chromosome set of a narrowPeak (column 1); blank lines are refused."""
    chroms: set[str] = set()
    n = 0
    with open(path) as fh:
        for line in fh:
            chrom = line.split("\t", 1)[0].strip()
            if not chrom:
                raise ValueError(f"{path}: blank line or empty chromosome at row {n + 1}")
            chroms.add(chrom)
            n += 1
    if n == 0:
        raise ValueError(f"{path}: no rows")
    return {"rows": n, "chroms": chroms}


def fold_chroms(fit: dict, fit_path: Path) -> list[str]:
    """The union of fit.json's training, validation and test chromosomes, in chromosome order; overlaps are refused."""
    seen: dict[str, str] = {}
    for key in FOLD_LISTS:
        values = fit.get(key)
        if not isinstance(values, list) or not values:
            raise ValueError(f"{fit_path}: no {key}")
        for c in values:
            if c in seen:
                raise ValueError(f"{fit_path}: {c} is in both {seen[c]} and {key}")
            seen[c] = key
    return sorted(seen, key=chrom_key)


def chrombpnet_interpret_args(cfg: dict, fold: str, models_subdir: str) -> Path:
    """ChromBPNet's 05.0 record of the fold's attribution, interpretation.interpret.args.json."""
    prefix = f"{cfg['dataset_name']}_{cfg['peak_type']}_fold_{fold}"
    return Path(cfg["output_dir"]) / models_subdir / prefix / "interpretation" / "interpretation.interpret.args.json"


def attribute_parameters(fold_dir: Path, fold: str, fit: dict, chroms: list[str], args: argparse.Namespace) -> dict:
    """The fold's attribute.json, in the stock command's key names (cherimoya_cli/defaults.py)."""
    return {
        "model": str(fold_dir / f"{fit['name']}.torch"),
        "sequences": fit["sequences"],
        "loci": list(fit["loci"]),
        "chroms": chroms,
        "in_window": IN_WINDOW,
        "attr_window": args.attr_window,
        "exclusion_lists": None,
        "algorithm": "deep_lift_shap",
        "output": args.output,
        "group": 0,
        "n_shuffles": args.n_shuffles,
        "batch_size": args.batch_size,
        "warning_threshold": args.warning_threshold,
        "random_state": args.seed_base + args.seed_stride * int(fold),
        "dtype": "float32",
        "device": "cuda",
        "compile": False,
        "verbose": False,
        **{key: str(fold_dir / name) for key, name in OUTPUT_NAMES.items()},
        "fast": {
            "precision": args.precision,
            "mem_budget_gb": args.mem_budget_gb,
            "ref_workers": args.ref_workers,
            "audit": args.audit,
        },
    }


def output_files(params: dict) -> list[Path]:
    """Every file an attribution with these parameters writes: the three outputs and attribute_fast.py's sidecars."""
    attr = params["attr_filename"]
    base = attr[: -len(".npz")] if attr.endswith(".npz") else attr
    return [Path(params[k]) for k in OUTPUT_NAMES] + [Path(base + ".deltas.npy"), Path(base + ".meta.json")]


def config_text(cfg: dict, regions: str, output_dir: Path, output: str) -> str:
    """The Cherimoya dataset's igvf_tf_chrombpnet config.yaml (see the module doc)."""
    lines = [
        f"dataset_name: {cfg['dataset_name']}",
        f"assay: {cfg['assay']}",
        f"signal_path: {json.dumps(cfg['signal_path'])}   # required and suffix-checked by config.sh; 06-11 never read it",
        f"regions: {json.dumps(regions)}   # 06-11 never read it",
        f"output_dir: {json.dumps(str(output_dir))}",
        f"peak_type: {cfg['peak_type']}",
        f"folds: {json.dumps(cfg['folds'])}",
        f"contrib_heads: {json.dumps([output])}",
    ]
    if cfg.get("modisco_max_seqlets") is not None:
        lines.append(f"modisco_max_seqlets: {int(cfg['modisco_max_seqlets'])}")
    return CONFIG_HEADER + "\n".join(lines) + "\n"


def write_if_changed(path: Path, text: str) -> bool:
    """Write `text` to `path` through a temporary file and os.replace, unless the file already holds it."""
    if path.is_file() and path.read_text() == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)
    return True


def read_list(path: Path) -> list[str]:
    """Dataset names, one per line; blank lines and # comments are skipped, duplicates refused."""
    names = []
    for line in path.read_text().splitlines():
        name = line.split("#", 1)[0].strip()
        if name:
            if name in names:
                raise ValueError(f"{path}: {name} is listed twice")
            names.append(name)
    return names


def compendium_rows(prefix: str, datasets: list[str], configs: dict[str, str]) -> list[tuple[str, str]]:
    rows = [(f"{prefix}_{ds}", configs[ds]) for ds in datasets]
    for label, _ in rows:
        if not LABEL_RE.match(label):
            raise ValueError(f"label {label!r} does not match {LABEL_RE.pattern}")
    return rows


def write_lists(inputs_dir: Path, shared: list[str], rest: list[str], cbp: dict, cheri: dict) -> dict[str, bool]:
    """The dataset and compendium lists under inputs_dir; returns {file name: written}."""
    cbp_shared = compendium_rows("cbp", shared, cbp)
    cheri_shared = compendium_rows("cheri", shared, cheri)
    cheri_all = cheri_shared + compendium_rows("cheri", rest, cheri)
    joint = cbp_shared + cheri_shared
    for name, rows in (("cheri", cheri_all), ("joint", joint)):
        labels = [label for label, _ in rows]
        if len(set(labels)) != len(labels):
            raise ValueError(f"duplicate labels in the {name} list")

    def tsv(rows):
        return "".join(f"{label}\t{path}\n" for label, path in rows)

    files = {
        f"shared{len(shared)}.txt": "".join(f"{ds}\n" for ds in shared),
        f"rest{len(rest)}.txt": "".join(f"{ds}\n" for ds in rest),
        f"cbp{len(shared)}.tsv": tsv(cbp_shared),
        f"cheri{len(shared)}.tsv": tsv(cheri_shared),
        f"cheri{len(cheri_all)}.tsv": tsv(cheri_all),
        f"joint{len(joint)}.tsv": tsv(joint),
    }
    return {name: write_if_changed(inputs_dir / name, text) for name, text in files.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cheri-root", type=Path, required=True, help="make_jobs.py's --out-root")
    ap.add_argument("--chrombpnet-configs", type=Path, nargs="+", required=True,
                    help="igvf_tf_chrombpnet config.yaml files, one per dataset")  # fmt: skip
    ap.add_argument("--shared", type=Path, required=True,
                    help="the datasets both models are compared on, one per line")  # fmt: skip
    ap.add_argument("--compendia", type=Path, required=True, help="the compendia root; lists go to <dir>/inputs/")
    ap.add_argument("--models-subdir", default="full_models",
                    help="ChromBPNet's full-model directory under output_dir (default full_models)")  # fmt: skip
    ap.add_argument("--attr-window", type=int, default=1000, help="attributed bases around the summit (default 1000)")
    ap.add_argument("--seed-base", type=int, default=1234, help="fold k's random_state is base + stride x k")
    ap.add_argument("--seed-stride", type=int, default=100, help="must be >= --n-shuffles (default 100)")
    ap.add_argument("--n-shuffles", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64, help="sets the engine's head_rows (2 x); default 64")
    ap.add_argument("--warning-threshold", type=float, default=0.001)
    ap.add_argument("--output", choices=("counts",), default="counts", help="the head attributed (counts)")
    ap.add_argument("--precision", choices=("reference", "fp32"), default="reference")
    ap.add_argument("--mem-budget-gb", type=float, default=7.5)
    ap.add_argument("--ref-workers", type=int, default=3)
    ap.add_argument("--audit", type=int, default=8, help="peaks the stock deep_lift_shap re-checks per fold")
    ap.add_argument("--replace-used", action="store_true",
                    help="rewrite an attribute.json that changes although its outputs exist")  # fmt: skip
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%F %T")  # fmt: skip

    if args.attr_window <= 0 or args.attr_window % 2 or args.attr_window > IN_WINDOW:
        ap.error(f"--attr-window must be even and in (0, {IN_WINDOW}], got {args.attr_window}")
    if args.n_shuffles <= 0 or args.seed_stride < args.n_shuffles:
        ap.error(f"--seed-stride ({args.seed_stride}) must be >= --n-shuffles ({args.n_shuffles}) > 0, or folds would "
                 "share references")  # fmt: skip
    if args.batch_size <= 0 or args.audit < 0 or args.ref_workers < 0 or args.mem_budget_gb <= 0:
        ap.error("--batch-size and --mem-budget-gb must be positive, --audit and --ref-workers >= 0")

    cfgs = {}
    for path in sorted(args.chrombpnet_configs):
        cfg = read_chrombpnet_config(path)
        ds = cfg["dataset_name"]
        if ds in cfgs:
            raise ValueError(f"{ds} has two configs: {cfgs[ds]['config']} and {cfg['config']}")
        cfgs[ds] = cfg
    shared = read_list(args.shared)
    unknown = [ds for ds in shared if ds not in cfgs]
    if unknown:
        raise ValueError(f"{args.shared} names datasets without a ChromBPNet config: {unknown}")
    rest = sorted(ds for ds in cfgs if ds not in shared)

    cheri_root = Path(os.path.abspath(args.cheri_root))
    peaks_cache: dict[str, dict] = {}
    counts = {"attribute.json written": 0, "attribute.json unchanged": 0, "config written": 0, "config unchanged": 0}
    n_checked = 0
    cheri_configs: dict[str, str] = {}
    for ds, cfg in cfgs.items():
        regions = None
        for fold in cfg["folds"]:
            fold_dir = cheri_root / ds / f"fold_{fold}"
            fit_path = fold_dir / "fit.json"
            fit = json.loads(fit_path.read_text())
            if fit.get("name") != f"cherimoya.fold_{fold}":
                raise ValueError(f"{fit_path}: name is {fit.get('name')!r}, not cherimoya.fold_{fold}")
            loci = fit.get("loci")
            if not isinstance(loci, list) or len(loci) != 1:
                raise ValueError(f"{fit_path}: expected exactly one loci file, got {loci!r}")
            if regions is None:
                regions = loci[0]
            elif loci[0] != regions:
                raise ValueError(f"{fit_path}: loci {loci[0]} differs from the dataset's other folds ({regions})")
            chroms = fold_chroms(fit, fit_path)
            if loci[0] not in peaks_cache:
                peaks_cache[loci[0]] = scan_peaks(Path(loci[0]))
            peaks = peaks_cache[loci[0]]
            unlisted = sorted(peaks["chroms"] - set(chroms), key=chrom_key)
            if unlisted:
                raise ValueError(f"{loci[0]} has peaks on {unlisted}, which {fit_path} does not list")
            absent = sorted(set(chroms) - peaks["chroms"], key=chrom_key)
            if absent:
                logger.warning("%s fold %s: no peaks on %s (listed by fit.json; harmless)", ds, fold, absent)
            args_json = chrombpnet_interpret_args(cfg, fold, args.models_subdir)
            if args_json.is_file():
                cbp = json.loads(args_json.read_text())
                if cbp.get("num_shuffles") != args.n_shuffles:
                    raise ValueError(f"{args_json}: num_shuffles {cbp.get('num_shuffles')}, not {args.n_shuffles}")
                if cbp.get("regions") != loci[0]:
                    raise ValueError(f"{args_json}: ChromBPNet scored {cbp.get('regions')}, not {loci[0]}")
                n_checked += 1

            params = attribute_parameters(fold_dir, fold, fit, chroms, args)
            text = json.dumps(params, indent=4) + "\n"
            target = fold_dir / "attribute.json"
            if target.is_file() and target.read_text() != text:
                used = [str(p) for p in output_files(json.loads(target.read_text())) if p.exists()]
                if used and not args.replace_used:
                    raise RuntimeError(
                        f"{target} would change, but an attribution ran with it ({', '.join(used)}); "
                        "rerun with --replace-used to rewrite it (attr_status.py then calls those outputs stale)"
                    )
            written = write_if_changed(target, text)
            counts["attribute.json written" if written else "attribute.json unchanged"] += 1
            logger.debug("%s fold %s: %s (%d peaks, %d chromosomes)", ds, fold, target, peaks["rows"], len(chroms))

        out_dir = cheri_root / ds / "downstream"
        config = cheri_root / "configs" / ds / "config.yaml"
        written = write_if_changed(config, config_text(cfg, regions, out_dir, args.output))
        counts["config written" if written else "config unchanged"] += 1
        cheri_configs[ds] = str(config)

    inputs_dir = Path(os.path.abspath(args.compendia)) / "inputs"
    lists = write_lists(inputs_dir, shared, rest, {ds: c["config"] for ds, c in cfgs.items()}, cheri_configs)
    n_folds = counts["attribute.json written"] + counts["attribute.json unchanged"]
    logger.info(
        "%d dataset(s), %d fold(s): attribute.json %d written, %d unchanged; configs %d written, %d unchanged; "
        "n_shuffles checked against ChromBPNet's interpret args on %d fold(s)",
        len(cfgs),
        n_folds,
        counts["attribute.json written"],
        counts["attribute.json unchanged"],
        counts["config written"],
        counts["config unchanged"],
        n_checked,
    )
    logger.info(
        "lists in %s: %s",
        inputs_dir,
        ", ".join(f"{name} ({'written' if w else 'unchanged'})" for name, w in lists.items()),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
