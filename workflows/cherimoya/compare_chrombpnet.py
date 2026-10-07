#!/usr/bin/env python3
"""Score a Cherimoya model and ChromBPNet on the same test regions, with the same code.

The two tools report different numbers about different region sets: ChromBPNet's
`chrombpnet_metrics.json` takes the median JSD over its filtered test peaks,
while Cherimoya's `{name}.test.performance.tsv` takes the mean JSD over every
peak on the test chromosomes, after its own outlier filter. Comparing those
would compare the definitions as much as the models. This puts both models
through one definition on one region set:

  regions    ChromBPNet's test-set predictions file (chrombpnet_predictions.h5,
             written by its training), i.e. its filtered peaks on the fold's
             test chromosomes, at the centre it predicted them.
  observed   the bigwig both models were trained on, over [centre - 500,
             centre + 500), NaN read as 0 -- ChromBPNet's own extraction.
  ChromBPNet the profile probabilities and log counts in that file.
  Cherimoya  the model's prediction for [centre - 1057, centre + 1057) from the
             genome; its 1000-bp output covers the same window as observed.
  measures   ChromBPNet's definitions (chrombpnet/training/metrics.py):
             counts -- Pearson, Spearman and MSE of predicted log counts
             against log(1 + observed total); profile -- per region, the
             Jensen-Shannon distance (natural log) between observed/total and
             the predicted probabilities, then the median, and the median of
             its min-max normalised form (0 = as far as a uniform profile,
             1 = identical).

ChromBPNet's own numbers are recomputed here from its predictions and checked
against its `chrombpnet_metrics.json`: if they disagree, the observed signal
or the regions are not what it was scored on, and the run fails rather than
writing a comparison.

Output (the paths come from compare.json, written by make_jobs.py):
  compare.tsv          one row per model: dataset, fold, model, n_regions and
                       the five measures
  compare_counts.npz   observed and both models' log counts per region, for
                       plotting

    python workflows/cherimoya/compare_chrombpnet.py -p <dataset>/fold_<k>/compare.json

Runs in the pixi `cherimoya` environment, on a GPU by default (--device cpu works).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.special import softmax
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger("compare_chrombpnet")

IN_WINDOW, OUT_WINDOW = 2114, 1000
MEASURES = ("counts_pearson", "counts_spearman", "counts_mse", "median_jsd", "median_norm_jsd")
# How far a recomputed ChromBPNet measure may sit from its own report.
TOLERANCE = 1e-6


def counts_measures(true_logcounts: np.ndarray, pred_logcounts: np.ndarray) -> dict:
    """ChromBPNet's counts_metrics: Pearson, Spearman, MSE on log counts."""
    return {
        "counts_pearson": float(pearsonr(true_logcounts, pred_logcounts)[0]),
        "counts_spearman": float(spearmanr(true_logcounts, pred_logcounts)[0]),
        "counts_mse": float(((true_logcounts - pred_logcounts) ** 2).mean()),
    }


def profile_measures(observed: np.ndarray, pred_probs: np.ndarray, pseudocount: float = 0.001) -> dict:
    """ChromBPNet's profile_metrics, vectorised: the median Jensen-Shannon
    distance between observed and predicted profiles, and the median of its
    min-max normalised form. Regions with no observed reads give NaN, which
    the medians skip, as ChromBPNet's nanmedian does."""
    observed = np.asarray(observed, dtype=np.float64)
    totals = np.nansum(observed, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        jsd = jensenshannon(observed / (pseudocount + totals), pred_probs, axis=1)
        uniform = np.full(observed.shape[1], 1.0 / observed.shape[1])
        max_jsd = jensenshannon(observed / totals, np.broadcast_to(uniform, observed.shape), axis=1)
        norm = np.clip((jsd - max_jsd) / (0.0 - max_jsd), 0.0, 1.0)
    # get_min_max_normalized_value returns its clipped value, and a NaN fails
    # both of its comparisons, so NaN stays NaN there as here.
    return {"median_jsd": float(np.nanmedian(jsd)), "median_norm_jsd": float(np.nanmedian(norm))}


def measures(observed: np.ndarray, pred_probs: np.ndarray, pred_logcounts: np.ndarray) -> dict:
    true_logcounts = np.log(1 + observed.sum(axis=1))
    return {**counts_measures(true_logcounts, pred_logcounts), **profile_measures(observed, pred_probs)}


_LOOKUP = np.full(256, 4, dtype=np.int64)
for _i, _b in enumerate("ACGT"):
    _LOOKUP[ord(_b)] = _LOOKUP[ord(_b.lower())] = _i


def one_hot(seqs: list[str]) -> np.ndarray:
    """(n, 4, length) float32 one-hot in A, C, G, T order; other bases are all zero."""
    codes = _LOOKUP[np.frombuffer("".join(seqs).encode(), dtype=np.uint8)].reshape(len(seqs), -1)
    eye = np.vstack([np.eye(4, dtype=np.float32), np.zeros((1, 4), dtype=np.float32)])
    return np.ascontiguousarray(eye[codes].transpose(0, 2, 1))


def read_chrombpnet(path: Path):
    import h5py

    with h5py.File(path, "r") as h5:
        chroms = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in h5["coords/coords_chrom"][()]])
        centers = h5["coords/coords_center"][()].astype(np.int64)
        is_peak = h5["coords/coords_peak"][()].astype(np.int64)
        probs = h5["predictions/profs"][()]
        logcounts = h5["predictions/logcounts"][()]
    keep = is_peak == 1
    return chroms[keep], centers[keep], probs[keep], logcounts[keep]


def read_observed(bigwig: str, chroms, centers) -> np.ndarray:
    import pyBigWig

    out = np.zeros((len(centers), OUT_WINDOW), dtype=np.float64)
    with pyBigWig.open(bigwig) as bw:
        for i, (c, m) in enumerate(zip(chroms, centers)):
            out[i] = np.nan_to_num(
                np.asarray(bw.values(c, int(m) - OUT_WINDOW // 2, int(m) + OUT_WINDOW // 2), dtype=np.float64)
            )
    return out


def read_sequences(fasta: str, chroms, centers) -> np.ndarray:
    import pyfaidx

    genome = pyfaidx.Fasta(fasta, sequence_always_upper=True)
    seqs = [str(genome[c][int(m) - IN_WINDOW // 2 : int(m) + IN_WINDOW // 2]) for c, m in zip(chroms, centers)]
    short = [i for i, s in enumerate(seqs) if len(s) != IN_WINDOW]
    if short:
        raise ValueError(
            f"{len(short)} region(s) run off a chromosome end, e.g. {chroms[short[0]]}:{centers[short[0]]}"
        )
    return one_hot(seqs)


def predict_cherimoya(model_path: str, X: np.ndarray, batch_size: int, device: str, dtype: str):
    import torch
    from cherimoya import Cherimoya
    from tangermeme.predict import predict

    model = Cherimoya.load(model_path, device=device, compile=False).eval()
    if model.n_control_tracks:
        raise ValueError("the model was trained with controls; this comparison has none")
    logits, logcounts = predict(model, torch.from_numpy(X), batch_size=batch_size, device=device,
                                dtype=dtype, verbose=False)  # fmt: skip
    logits = logits.float().numpy()
    logcounts = logcounts.float().numpy()
    if logits.shape[1:] != (1, OUT_WINDOW) or logcounts.shape[1:] != (1,):
        raise ValueError(f"expected one unstranded {OUT_WINDOW}-bp track, got {logits.shape} / {logcounts.shape}")
    return softmax(logits[:, 0].astype(np.float64), axis=1), logcounts[:, 0].astype(np.float64)


def check_against_report(ours: dict, report_path: Path) -> list[str]:
    """Where our recomputation of ChromBPNet's measures departs from its own report."""
    with open(report_path) as fh:
        report = json.load(fh)
    reported = {
        "counts_pearson": report["counts_metrics"]["peaks"]["pearsonr"],
        "counts_spearman": report["counts_metrics"]["peaks"]["spearmanr"],
        "counts_mse": report["counts_metrics"]["peaks"]["mse"],
        "median_jsd": report["profile_metrics"]["peaks"]["median_jsd"],
        "median_norm_jsd": report["profile_metrics"]["peaks"]["median_norm_jsd"],
    }
    return [
        f"{k}: recomputed {ours[k]:.8g}, reported {v:.8g}"
        for k, v in reported.items()
        if not abs(ours[k] - v) <= TOLERANCE
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-p", "--parameters", type=Path, required=True, help="compare.json from make_jobs.py")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", help="Cherimoya's prediction dtype (default float32)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%F %T")  # fmt: skip

    p = json.loads(args.parameters.read_text())
    for key in ("model", "chrombpnet_predictions", "chrombpnet_metrics", "signal", "sequences"):
        if not Path(p[key]).exists():
            logger.error("%s fold %s: no %s at %s", p["dataset"], p["fold"], key, p[key])
            return 1

    chroms, centers, cbp_probs, cbp_logcounts = read_chrombpnet(Path(p["chrombpnet_predictions"]))
    logger.info("%s fold %s: %d ChromBPNet test peaks on %s", p["dataset"], p["fold"], len(centers),
                ", ".join(sorted(set(chroms), key=lambda c: (len(c), c))))  # fmt: skip
    observed = read_observed(p["signal"], chroms, centers)

    rows = {"chrombpnet": measures(observed, cbp_probs, cbp_logcounts)}
    problems = check_against_report(rows["chrombpnet"], Path(p["chrombpnet_metrics"]))
    if problems:
        logger.error("recomputed ChromBPNet measures do not match its report, so the regions or the "
                     "observed signal differ from what it was scored on:\n  %s", "\n  ".join(problems))  # fmt: skip
        return 1

    X = read_sequences(p["sequences"], chroms, centers)
    che_probs, che_logcounts = predict_cherimoya(p["model"], X, args.batch_size, args.device, args.dtype)
    rows["cherimoya"] = measures(observed, che_probs, che_logcounts)

    out = Path(p["output"])
    with open(out.with_suffix(".tsv.partial"), "w") as fh:
        fh.write("\t".join(["dataset", "fold", "model", "n_regions", *MEASURES]) + "\n")
        for model, m in rows.items():
            fh.write("\t".join([p["dataset"], str(p["fold"]), model, str(len(centers)),
                                *(f"{m[k]:.6g}" for k in MEASURES)]) + "\n")  # fmt: skip
    np.savez_compressed(out.with_name("compare_counts.npz"), chrom=chroms, center=centers,
                        observed=np.log(1 + observed.sum(axis=1)), chrombpnet=cbp_logcounts,
                        cherimoya=che_logcounts)  # fmt: skip
    out.with_suffix(".tsv.partial").rename(out)
    for model, m in rows.items():
        logger.info("%-10s %s", model, "  ".join(f"{k} {m[k]:.4f}" for k in MEASURES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
