"""Synthetic inputs for the comparison's tests (workflows/comparison/): helpers, not tests.

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
WORKFLOWS = REPO / "workflows"
COMPARE = WORKFLOWS / "comparison"
for _p in (*(str(WORKFLOWS / d) for d in ("train", "attribution", "downstream")), str(COMPARE), str(TESTS)):
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


# ---------------------------------------------------------------------------------------------------------------------
# level 3: motif databases, TF-MoDISco results, beds and the joint compendium


def random_motifs(names: list[str], width: int = 12, seed: int = 0) -> dict[str, np.ndarray]:
    """{name: (width, 4) PPM}: sharp, distinct motifs (Dirichlet 0.15 per position)."""
    rng = np.random.default_rng(seed)
    return {n: rng.dirichlet([0.15] * 4, size=width) for n in names}


def write_meme(path: Path, motifs: dict[str, np.ndarray]) -> Path:
    lines = ["MEME version 4", "", "ALPHABET= ACGT", "", "strands: + -", "", "Background letter frequencies:",
             "A 0.25 C 0.25 G 0.25 T 0.25", ""]  # fmt: skip
    for name, ppm in motifs.items():
        lines += [f"MOTIF {name}", f"letter-probability matrix: alength= 4 w= {len(ppm)} nsites= 20 E= 0"]
        lines += [" ".join(f"{v:.6f}" for v in row) for row in ppm]
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


DB_NAMES = [f"TF{i}_0" for i in range(40)] + ["TN5_1", "DNASE_1"]


def motif_db(tmp_path: Path) -> tuple[Path, dict[str, np.ndarray]]:
    motifs = random_motifs(DB_NAMES, seed=7)
    return write_meme(tmp_path / "db" / "motifs.meme", motifs), motifs


def pattern_arrays(ppm: np.ndarray, width: int = 30, sign: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """A pattern's (width, 4) PPM and CWM with the motif in its middle: uniform flanks with near-zero contributions,
    so modisco-lite's trimming keeps the motif."""
    core = len(ppm)
    a = (width - core) // 2
    seq = np.full((width, 4), 0.25)
    seq[a : a + core] = ppm
    cwm = seq * 0.002
    cwm[a : a + core] = ppm * sign
    return seq, cwm


def write_modisco(path: Path, patterns: dict[str, list[tuple[np.ndarray, dict]]], window: int) -> Path:
    """patterns: {"pos"|"neg": [(ppm, seqlets {example_idx, start, end, is_revcomp})]} as a modisco-lite results h5."""
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as fh:
        fh.attrs["window_size"] = window
        for posneg, items in patterns.items():
            g = fh.create_group(f"{posneg}_patterns")
            for i, (ppm, seqlets) in enumerate(items):
                p = g.create_group(f"pattern_{i}")
                seq, cwm = pattern_arrays(ppm, sign=1.0 if posneg == "pos" else -1.0)
                p.create_dataset("sequence", data=seq)
                p.create_dataset("contrib_scores", data=cwm)
                p.create_dataset("hypothetical_contribs", data=cwm)
                s = p.create_group("seqlets")
                n = len(seqlets["start"])
                s.create_dataset("n_seqlets", data=np.array([n]))
                for key in ("example_idx", "start", "end", "is_revcomp"):
                    s.create_dataset(key, data=np.asarray(seqlets[key]))
    return path


def peaks_bed(n: int, drop: tuple[int, ...] = ()) -> list[str]:
    """n narrowPeak lines on chr1 and chr2 (1,000 bp, summit 500), the rows `drop` left out."""
    out = []
    for i in range(n):
        chrom = f"chr{1 + i % 2}"
        start = 10_000 + 5_000 * i
        out.append(f"{chrom}\t{start}\t{start + 1000}\tpeak_{i}\t0\t.\t{i % 7 + 1}.5\t-1\t-1\t500\n")
    return [line for i, line in enumerate(out) if i not in drop]


def seqlets_at(peaks: list[int], rows_of: dict[int, int], window: int, offset: int = 0) -> dict:
    """Seqlets 30 wide at the window's centre (+ offset) of the given peaks, as rows of a model's bed."""
    start = window // 2 - 15 + offset
    return {"example_idx": np.array([rows_of[p] for p in peaks]), "start": np.full(len(peaks), start),
            "end": np.full(len(peaks), start + 30), "is_revcomp": np.zeros(len(peaks), bool)}  # fmt: skip


N_PEAKS3 = 40
DROPPED3 = (5, 17)
"""The peaks of the level-3 fixture that Cherimoya drops (windows with an N)."""
AVERAGE_PATTERNS = {
    # model: {posneg: [(motif, peaks of its seqlets, offset of the seqlets from the centre)]}
    "chrombpnet": {"pos": [("TF0_0", range(0, 10), 0), ("TF1_0", range(10, 20), 0), ("TF2_0", range(20, 25), 0),
                           ("TF3_0", range(25, 30), 0)],
                   "neg": [("TF4_0", range(30, 35), 0)]},
    "cherimoya": {"pos": [("TF0_0", range(0, 10), 0), ("TF1_0", range(10, 20), 20), ("TF2_0", range(20, 25), 10),
                          ("TN5_1", range(30, 40), 0)],
                  "neg": [("TF5_0", range(35, 40), 0)]},
}  # fmt: skip
JOINT_CLUSTERS = {("chrombpnet", "pos", 0): "0", ("cherimoya", "pos", 0): "0", ("chrombpnet", "pos", 1): "1",
                  ("cherimoya", "pos", 1): "1", ("chrombpnet", "pos", 2): "2", ("cherimoya", "pos", 2): "2",
                  ("chrombpnet", "pos", 3): "3", ("chrombpnet", "neg", 0): "4", ("cherimoya", "pos", 3): "5",
                  ("cherimoya", "neg", 0): "6"}  # fmt: skip
QC_PATTERNS = {"chrombpnet": ["TF0_0", "TF1_0", "TF2_0"], "cherimoya": ["TF0_0", "TF1_0", "TN5_1"]}


def make_modisco_dataset(tmp_path: Path, ds: str, db: dict[str, np.ndarray], *, folds=("0", "1", "2"),
                         seed: int = 0) -> dict:  # fmt: skip
    """Both models' fold-average (08.0, window 500) and per-fold QC (window 400) TF-MoDISco results of one dataset,
    in the layout compare_modisco.py reads: 40 peaks on chr1 and chr2, Cherimoya's beds without DROPPED3; the
    patterns of AVERAGE_PATTERNS (their PPMs DB motifs, their seqlets at the given peaks) and, per fold, of
    QC_PATTERNS on a 30-peak subsample in a fold-specific order (seqlets at its first 10, 20 and 30 peaks)."""
    import json as _json

    rng = np.random.default_rng(seed)
    cheri_root, cbp_out = tmp_path / "cheri", tmp_path / "chrombpnet" / ds
    cheri_out = cheri_root / ds / "downstream"
    write_config(cheri_root, ds, folds)
    lines = peaks_bed(N_PEAKS3)
    kept = [i for i in range(N_PEAKS3) if i not in DROPPED3]
    rows = {"chrombpnet": {i: i for i in range(N_PEAKS3)}, "cherimoya": {p: j for j, p in enumerate(kept)}}
    for k in folds:
        fold_dir = cheri_root / ds / f"fold_{k}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        cbp_fold = cbp_out / "full_models" / f"{ds}_all_fold_{k}"
        (fold_dir / "compare.json").write_text(_json.dumps(
            {"chrombpnet_predictions": str(cbp_fold / "evaluation" / "chrombpnet_predictions.h5")}))  # fmt: skip
        for path, text in ((cbp_fold / "interpretation" / "interpretation.interpreted_regions.bed", lines),
                           (cheri_out / "full_models" / f"{ds}_all_fold_{k}" / "interpretation" /
                            "interpretation.interpreted_regions.bed", [lines[i] for i in kept])):  # fmt: skip
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(text))
        # the QC subsample: 30 peaks in a fold's own order; Cherimoya's without the dropped ones
        order = [int(i) for i in rng.permutation(N_PEAKS3)[:30]]
        qc_rows = {"chrombpnet": {p: j for j, p in enumerate(order)},
                   "cherimoya": {p: j for j, p in enumerate(i for i in order if i not in DROPPED3)}}  # fmt: skip
        for model, fold_root, name in (
            ("chrombpnet", cbp_fold, "chrombpnet_nobias"),
            ("cherimoya", cheri_out / "full_models" / f"{ds}_all_fold_{k}", "cherimoya"),
        ):
            bed = fold_root / "auxiliary" / "interpret_subsample" / f"{name}.interpreted_regions.bed"
            bed.parent.mkdir(parents=True, exist_ok=True)
            bed.write_text("".join(lines[i] for i in qc_rows[model]))
            qc_dir = fold_root / "evaluation" / "motif_qc_counts"
            items = []
            for t, motif in enumerate(QC_PATTERNS[model]):
                peaks = [p for p in order[: 10 * (t + 1)] if p in qc_rows[model]][-10:]
                items.append((db[motif], seqlets_at(peaks, qc_rows[model], 400)))
            write_modisco(qc_dir / f"{name}_modisco_results.h5", {"pos": items}, 400)
            if model == "chrombpnet":
                (qc_dir / f"{name}_motif_qc.json").write_text(
                    _json.dumps({"motif_db": str(tmp_path / "db" / "motifs.meme")})
                )
            else:
                (qc_dir / "cherimoya_qc.done.json").write_text("{}")  # fmt: skip
    for model, out in (("chrombpnet", cbp_out), ("cherimoya", cheri_out)):
        patterns = {}
        for posneg, items in AVERAGE_PATTERNS[model].items():
            patterns[posneg] = []
            for motif, peaks, offset in items:
                peaks = [p for p in peaks if p in rows[model]]
                patterns[posneg].append((db[motif], seqlets_at(peaks, rows[model], 500, offset)))
        write_modisco(out / "contrib_scores" / ds / "modisco" / "modisco_counts_results.h5", patterns, 500)
    old = tmp_path.stat().st_mtime - 3600
    h5 = cbp_out / "contrib_scores" / ds / "modisco" / "modisco_counts_results.h5"
    os.utime(h5, (old, old))
    (cheri_out / ".done").mkdir(parents=True, exist_ok=True)
    (cheri_out / ".done" / "08.0").write_text("ok\n")
    return {"cheri_root": cheri_root, "cbp_out": cbp_out, "cheri_out": cheri_out, "ds": ds, "folds": list(folds)}


def write_joint(compendia: Path, datasets: list[str], db_path: Path, tag: str = "joint20") -> Path:
    """The joint compendium's modisco_compendium_meta.tsv (JOINT_CLUSTERS for each dataset, cluster ids offset by
    dataset) and a compendium.json naming the reference database."""
    import json as _json

    d = compendia / tag
    d.mkdir(parents=True, exist_ok=True)
    lines = ["name\tnum_seqlets\tmodel\tposneg\tavg_dist_from_summit\tannotation_score0\tannotation_name0\tcluster_id"]
    for n, ds in enumerate(datasets):
        for (model, posneg, i), cid in JOINT_CLUSTERS.items():
            label = ("cbp_" if model == "chrombpnet" else "cheri_") + ds
            motif = AVERAGE_PATTERNS[model][posneg][i][0]
            lines.append(f"{label}-{posneg}.pattern_{i}\t10\t{label}\t{posneg}\t0\t0.9\t{motif}\t{int(cid) + 100 * n}")
    (d / "modisco_compendium_meta.tsv").write_text("\n".join(lines) + "\n")
    (d / "compendium.json").write_text(_json.dumps({"ref_db": {"filepath": str(db_path)}}))
    return d


# ---------------------------------------------------------------------------------------------------------------------
# level 4: compendia


def make_compendium(root: Path, tag: str, clusters: list[tuple[str, str, np.ndarray, float]], meta_rows: list[tuple],
                    min_score: float = 0.7) -> Path:  # fmt: skip
    """A 09.0 compendium directory: modisco_compiled.h5 (each cluster's CWM, from a PPM times a scale, negative for
    neg), modisco_compendium_meta.tsv (meta_rows: name, num_seqlets, label, posneg, score, annotation, cluster) and
    compendium.json (min_annotation_score)."""
    import json as _json

    import h5py

    d = root / tag
    d.mkdir(parents=True, exist_ok=True)
    with h5py.File(d / "modisco_compiled.h5", "w") as fh:
        for cid, posneg, ppm, scale in clusters:
            _, cwm = pattern_arrays(ppm, sign=scale if posneg == "pos" else -scale)
            fh.create_dataset(f"{posneg}_patterns/{cid}/contrib_scores", data=cwm)
    lines = ["name\tnum_seqlets\tmodel\tposneg\tavg_dist_from_summit\tannotation_score0\tannotation_name0\tcluster_id"]
    for name, n, label, posneg, score, annotation, cid in meta_rows:
        lines.append(f"{name}\t{n}\t{label}\t{posneg}\t0\t{score}\t{annotation}\t{cid}")
    (d / "modisco_compendium_meta.tsv").write_text("\n".join(lines) + "\n")
    (d / "compendium.json").write_text(_json.dumps({"settings": {"min_annotation_score": min_score}}))
    return d


# ---------------------------------------------------------------------------------------------------------------------
# level 5: Fi-NeMo hits


def write_hits(directory: Path, bed_lines: list[str], hits: list[tuple], md5: str = "abc123",
               report: bool = True) -> Path:  # fmt: skip
    """A 10.0 output directory: hits.tsv (the columns Fi-NeMo 0.41 writes) from (peak index in bed_lines, motif,
    strand, offset of the hit centre from the summit, coefficient), hits.bed.gz, compendium.md5 and 11.0's
    finemo_report/motif_report.tsv."""
    directory.mkdir(parents=True, exist_ok=True)
    rows = [line.split("\t") for line in bed_lines]
    out = [
        "chr\tstart\tend\tstart_untrimmed\tend_untrimmed\tmotif_name\thit_coefficient\thit_coefficient_global\t"
        "hit_similarity\thit_correlation\thit_importance\thit_importance_sq\tstrand\tpeak_name\tpeak_id"
    ]
    for row, motif, strand, offset, coef in hits:
        f = rows[row]
        centre = int(f[1]) + int(f[9]) + offset
        start, end = centre - 5, centre + 5
        out.append(f"{f[0]}\t{start}\t{end}\t{start - 10}\t{end + 10}\t{motif}\t{coef}\t{coef}\t0.9\t0.9\t1\t1\t"
                   f"{strand}\t{f[3]}\t{row}")  # fmt: skip
    (directory / "hits.tsv").write_text("\n".join(out) + "\n")
    (directory / "hits.bed.gz").write_bytes(b"")
    (directory / "compendium.md5").write_text(f"{md5}  /compendia/x/modisco_compiled.h5\n")
    if report:
        motifs = sorted({h[1] for h in hits})
        (directory / "finemo_report").mkdir(exist_ok=True)
        (directory / "finemo_report" / "motif_report.tsv").write_text(
            "motif_name\tnum_hits_total\n" + "".join(f"{m}\t{sum(h[1] == m for h in hits)}\n" for m in motifs)
        )
    return directory
