"""Synthetic inputs for the comparison's tests (workflows/cherimoya/compare/): helpers, not tests.

make_performance_root  a Cherimoya results root whose folds hold compare.tsv, compare_counts.npz and fit.time, as
                       compare_chrombpnet.py and run_task.sh leave them, with known differences between the models
fake_bin               a directory with a fake `pixi` that runs `pixi run ... python <script>` with this test's
                       interpreter (or sleeps, for the signal test), for compare.sh
compare_env            the environment compare.sh wants, rooted in a test's tmp_path
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

TESTS = Path(__file__).resolve().parent
REPO = TESTS.parent
CHERIMOYA = REPO / "workflows" / "cherimoya"
COMPARE = CHERIMOYA / "compare"
for _p in (str(CHERIMOYA), str(COMPARE), str(TESTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FOLDS = ("0", "1", "2", "3", "4")
MEASURES = ("counts_pearson", "counts_spearman", "counts_mse", "median_jsd", "median_norm_jsd")
BASE = {"counts_pearson": 0.70, "counts_spearman": 0.66, "counts_mse": 0.55, "median_jsd": 0.25,
        "median_norm_jsd": 0.58}  # fmt: skip
N_PEAKS = 200


def write_config(root: Path, ds: str, folds=FOLDS) -> Path:
    config = root / "configs" / ds / "config.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        f'dataset_name: {ds}\nassay: ATAC\noutput_dir: "{root}/{ds}/downstream"\npeak_type: all\n'
        f'folds: {json.dumps(list(folds))}\ncontrib_heads: ["counts"]\n'
    )
    return config


def diff_of(i_ds: int, k: str) -> float:
    """Cherimoya minus ChromBPNet on every measure of dataset i, fold k: positive for all but the last dataset's fold
    0, which ChromBPNet wins."""
    return 0.01 * (i_ds + 1) + 0.002 * int(k) if (i_ds, k) != (2, "0") else -0.005


def make_performance_root(tmp_path: Path, datasets=("ds_a", "ds_b", "ds_c"), extra=("ds_other",), folds=FOLDS,
                          seed: int = 0) -> dict:  # fmt: skip
    """Every fold scored: compare.tsv (the five measures; Cherimoya = ChromBPNet + diff_of()), compare_counts.npz
    (observed log counts 0..N_PEAKS-1 shuffled; ChromBPNet predicts observed + 1, Cherimoya observed - 0.5) and
    fit.time. `extra` datasets exist on disk but are not in the list."""
    root = tmp_path / "cheri"
    rng = np.random.default_rng(seed)
    for i, ds in enumerate(list(datasets) + list(extra)):
        write_config(root, ds, folds)
        for k in folds:
            d = root / ds / f"fold_{k}"
            d.mkdir(parents=True, exist_ok=True)
            lines = ["\t".join(["dataset", "fold", "model", "n_regions", *MEASURES])]
            cbp = {m: BASE[m] + 0.001 * i + 0.0003 * int(k) for m in MEASURES}
            delta = diff_of(i, k)
            che = {m: cbp[m] + delta for m in MEASURES}
            for model, vals in (("chrombpnet", cbp), ("cherimoya", che)):
                lines.append("\t".join([ds, k, model, str(N_PEAKS), *(f"{vals[m]:.6g}" for m in MEASURES)]))
            (d / "compare.tsv").write_text("\n".join(lines) + "\n")
            observed = rng.permutation(np.arange(N_PEAKS, dtype=np.float64)) / 10.0
            np.savez_compressed(d / "compare_counts.npz", chrom=np.array(["chr1"] * N_PEAKS),
                                center=np.arange(N_PEAKS), observed=observed, chrombpnet=observed + 1.0,
                                cherimoya=observed - 0.5)  # fmt: skip
            (d / "fit.time").write_text("100 2097152\n")
    listing = tmp_path / "shared.txt"
    listing.write_text("# the shared datasets\n" + "\n".join(datasets) + "\n\n")
    return {"root": root, "list": listing, "datasets": list(datasets), "folds": list(folds)}


FAKE_PIXI = """#!/bin/bash
# pixi run --frozen --manifest-path M -e ENV python <script> ... -> the test's interpreter (or a sleep)
echo "pixi $*" >> "${FAKE_LOG:-/dev/null}"
shift
while (( $# )); do
    case "$1" in --frozen) shift ;; --manifest-path|-e) shift 2 ;; *) break ;; esac
done
if [[ -n "${FAKE_PIXI_SLEEP:-}" ]]; then exec sleep "${FAKE_PIXI_SLEEP}"; fi
if [[ "$1" == python ]]; then shift; exec "${FAKE_PYTHON}" "$@"; fi
exec "$@"
"""


def fake_bin(tmp_path: Path) -> Path:
    b = tmp_path / "bin"
    b.mkdir(exist_ok=True)
    (b / "pixi").write_text(FAKE_PIXI)
    (b / "pixi").chmod(0o755)
    return b


def compare_env(tmp_path: Path, cheri_root: Path, **extra) -> dict:
    """compare.sh's environment: the roots under tmp_path, a fake pixi first on PATH, its own SLURM_JOB_ID."""
    env = dict(os.environ)
    box = tmp_path / "box"
    box.mkdir(exist_ok=True)
    env.update(
        PATH=f"{fake_bin(tmp_path)}:{env.get('PATH', '/usr/bin:/bin')}",
        FAKE_PYTHON=sys.executable,
        FAKE_LOG=str(tmp_path / "fake.log"),
        CHERI_ROOT=str(cheri_root),
        CHROMBPNET_ROOT=str(tmp_path / "chrombpnet"),
        COMPENDIA_ROOT=str(tmp_path / "compendia"),
        COMPARISON_ROOT=str(tmp_path / "comparison"),
        BOX_DIR=str(box),
        REPO_ROOT=str(tmp_path / "igvf"),
        SLURM_JOB_ID=f"test{os.getpid()}{abs(hash(str(tmp_path))) % 100000}",
        COMPARE_WORKERS="2",
        COMPARE_NICE="0",
    )
    env.pop("SLURM_STEP_ID", None)
    env.update({k: str(v) for k, v in extra.items()})
    return env


def write_folds_dir(path: Path) -> Path:
    """igvf_tf_chrombpnet's folds/ for a genome of chr1-chr5: fold k tests chr<k+1>, validates the next one."""
    path.mkdir(parents=True, exist_ok=True)
    chroms = [f"chr{i}" for i in range(1, 6)]
    for k in range(5):
        test, valid = chroms[k], chroms[(k + 1) % 5]
        train = [c for c in chroms if c not in (test, valid)]
        (path / f"fold_{k}.json").write_text(json.dumps({"test": [test], "valid": [valid], "train": train}))
    return path


def make_attribution_dataset(tmp_path: Path, ds: str = "ds_a", *, seed: int = 0, n_chroms: int = 3) -> dict:
    """test_check_dataset.make_dataset (W = 12, 24 peaks, 2 dropped for an N) completed into what level 2 reads.

    The peaks are spread over chr1 .. chr<n_chroms> (row i on chr<1 + i % n_chroms>), in the peaks file and in
    ChromBPNet's beds, before the folds are converted. ChromBPNet's per-fold h5 gets shap/seq (float16, normal, sd
    0.01) and projected_shap/seq; its average is 06.0's float16 mean; Cherimoya's is average_as_06_0 of the converted
    folds, with downstream.sh's V2 marker. Returns make_dataset's dict plus cbp_shap {fold: (N, 4, L)}, cbp_average.
    """
    import shutil

    import h5py
    import to_chrombpnet_scores as tcs
    from test_check_dataset import average_as_06_0, make_dataset

    data = make_dataset(tmp_path, ds, convert=False, seed=seed)
    rows = [line.split("\t") for line in data["peaks"].read_text().splitlines()]
    for i, fields in enumerate(rows):
        fields[0] = f"chr{1 + i % n_chroms}"
    data["peaks"].write_text("".join("\t".join(fields) + "\n" for fields in rows))
    rng = np.random.default_rng(1000 + seed)
    cbp_shap = {}
    raw = None
    for k in FOLDS:
        interp = data["chrombpnet"] / f"{ds}_all_fold_{k}" / "interpretation"
        shutil.copy(data["peaks"], interp / "interpretation.interpreted_regions.bed")
        with h5py.File(interp / "interpretation.counts_scores.h5", "a") as fh:
            raw = fh["raw/seq"][()]
            shap = rng.normal(0.0, 0.01, size=raw.shape).astype(np.float16)
            chunks = (min(16, len(raw)), 4, raw.shape[2])
            fh.create_dataset("shap/seq", data=shap, chunks=chunks)
            fh.create_dataset("projected_shap/seq", data=(shap * raw).astype(np.float16), chunks=chunks)
        cbp_shap[k] = shap
        assert tcs.convert(data["folds"][k]["json"])[0] == "converted"
    average_as_06_0(data)
    (data["out"] / ".done").mkdir(parents=True, exist_ok=True)
    (data["out"] / ".done" / "v2").write_text("2026-10-08T23:13:13+02:00\tok\n")
    avg = None
    for k in FOLDS:
        if avg is None:
            avg = cbp_shap[k].copy()
        else:
            avg += cbp_shap[k]
    avg /= len(FOLDS)
    out = data["chrombpnet"].parent / "contrib_scores" / ds / f"{ds}_average_shaps.counts.h5"
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as fh:
        fh.create_dataset("raw/seq", data=raw, compression="gzip")
        fh.create_dataset("shap/seq", data=avg, compression="gzip")
        fh.create_dataset("projected_shap/seq", data=(avg * raw).astype(np.float16), compression="gzip")
    data.update(cbp_shap=cbp_shap, cbp_average=avg, cbp_average_path=out)
    return data
