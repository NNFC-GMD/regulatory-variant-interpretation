#!/usr/bin/env python3
"""The pilot's checks on predictions: V2 (against VS main), V3 (the derived full model), V4 (the noise floor) and V5
(forward against reverse strand).

VS is kundajelab/variant-scorer, main @0e1e341; "VS file:line" refers to that commit.

  pair        Two prediction files of the same rows (same ids, same order). Per array (a1/a2 log counts and each
              strand) and for logFC = (a2_logcounts - a1_logcounts) / ln 2: the absolute differences (median, 99th
              percentile, max, RMS) and the Pearson correlation. With --tol it passes when the max |difference| of
              every log-count array is within the tolerance.
              V3: cbp_full (derived from the two parts) against cbp_full_direct (chrombpnet.h5 run as it is); the
                  fork's own parity bound is 1e-4 at matmul precision 'highest' (fork VALIDATION.md:268-269).
              V4: one model on the same rows run twice: as it is; reordered and re-batched; TF32 against strict fp32;
                  Cherimoya's megakernel against its grad path; bf16 against the default. The 99th percentile of
                  |logFC difference| is the noise floor below which effects are not interpretable.
  strands     V5: every prediction file under --preds-root, forward against reverse complement: Pearson and Spearman
              of the per-strand log counts and their |difference|; for files with allele 2, the logFC from each strand
              alone, against the other strand and against the strand-averaged logFC (what forward-only scoring would
              change).
  vs-list     V2's inputs for VS main: --n rows of the variant list, among those build_windows.py kept, in VS's
              headerless `chrombpnet` schema (chr, pos, allele1, allele2, variant_id: VS src/utils/io.py:12, 42-48),
              plus chrom sizes from the FASTA's .fai.
  vs-compare  V2: VS's <prefix>.variant_scores.tsv (variant_scoring.py:330) against our predictions of the same model,
              joined on variant_id, since VS drops rows whose window leaves the chromosome without saying so
              (helpers.py:28-42; variant_scoring.py:44): VS's logfc against ours, and the log of VS's strand-averaged
              allele counts (allele1_pred_counts, allele2_pred_counts) against our a1/a2_logcounts. It passes when the
              99th percentile of |logfc difference| is within --tol (default 1e-3: TF-Keras and JAX numerics differ).
  summary     Every check JSON given, flattened to one TSV: check, label, quantity, n, p50, p99, max, pearson, tol,
              pass.

Exit status: 0 = ran and passed (or had nothing to pass), 3 = ran but outside its tolerance, 1 = could not run
(argparse's own usage errors exit 2, so 2 never means a tolerance).

    python workflows/variant_scoring/vchecks.py pair --a A.npz --b B.npz --label rerun --out pair.json [--tol 1e-4]
    python workflows/variant_scoring/vchecks.py strands --preds-root <root>/preds --out <root>/checks/v5.json
    python workflows/variant_scoring/vchecks.py vs-list --list <list.tsv> --windows <root>/windows/variants.npz \\
        --fai <genome.fasta>.fai --n 2000 --out-dir <root>/checks/v2
    python workflows/variant_scoring/vchecks.py vs-compare --vs-scores <prefix>.variant_scores.tsv \\
        --preds <root>/preds/<dataset>/fold_0/cbp_nobias.variants.npz --out <root>/checks/v2.json
    python workflows/variant_scoring/vchecks.py summary --inputs <root>/checks/*.json --out <root>/checks/summary.tsv

numpy only.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import predict_common as pc  # noqa: E402

logger = logging.getLogger("vchecks")

ARRAYS = ("a1_logcounts", "a2_logcounts", "a1_fwd", "a1_rev", "a2_fwd", "a2_rev")
VS_SCORE_COLUMNS = ("variant_id", "allele1_pred_counts", "allele2_pred_counts", "logfc")
SUMMARY_COLUMNS = ("check", "label", "quantity", "n", "p50", "p99", "max", "pearson", "tol", "pass")
# The exit status of a check that ran but is outside its tolerance (pilot_steps.sh reads it). Not 2, which is
# argparse's exit status for a usage error.
OUTSIDE_TOLERANCE = 3


def _f(x) -> float:
    return float(x) if x is not None and np.isfinite(x) else float("nan")


def pearson(a, b) -> float:
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2:
        return float("nan")
    a = a[ok] - a[ok].mean()
    b = b[ok] - b[ok].mean()
    den = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else float("nan")


def ranks(x) -> np.ndarray:
    """Ranks 0..n-1 (ties broken by position; the inputs here are continuous)."""
    x = np.asarray(x, np.float64)
    r = np.empty(len(x), np.float64)
    r[np.argsort(x, kind="stable")] = np.arange(len(x))
    return r


def spearman(a, b) -> float:
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    return pearson(ranks(a[ok]), ranks(b[ok])) if ok.sum() > 1 else float("nan")


def diff_stats(a, b) -> dict:
    """|a - b| quantiles, max and RMS over the finite pairs, the number of non-finite ones, and Pearson's r."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shapes differ: {a.shape} and {b.shape}")
    d = np.abs(a - b)
    finite = np.isfinite(d)
    out = {"n": int(d.size), "n_nonfinite": int((~finite).sum())}
    d = d[finite]
    if d.size == 0:
        return out | {"p50": float("nan"), "p99": float("nan"), "max": float("nan"), "rms": float("nan"),
                      "pearson": float("nan")}  # fmt: skip
    return out | {"p50": float(np.quantile(d, 0.5)), "p99": float(np.quantile(d, 0.99)), "max": float(d.max()),
                  "rms": float(np.sqrt(np.mean(d * d))), "pearson": pearson(a, b)}  # fmt: skip


def logfc(preds: dict) -> np.ndarray | None:
    if "a1_logcounts" in preds and "a2_logcounts" in preds:
        return (preds["a2_logcounts"].astype(np.float64) - preds["a1_logcounts"].astype(np.float64)) / pc.LN2
    return None


def compare_pair(a: dict, b: dict) -> dict:
    """diff_stats per shared array and for logFC; the rows must be the same ids in the same order."""
    if "ids" in a and "ids" in b and not np.array_equal(a["ids"], b["ids"]):
        raise ValueError("the two files hold different rows (ids differ); compare runs of the same --sample")
    quantities = {k: diff_stats(a[k], b[k]) for k in ARRAYS if k in a and k in b}
    if not quantities:
        raise ValueError("no prediction arrays in common")
    la, lb = logfc(a), logfc(b)
    out = {"quantities": quantities}
    if la is not None and lb is not None:
        quantities["logfc"] = diff_stats(la, lb)
        out["abs_logfc_a"] = {"p50": float(np.quantile(np.abs(la), 0.5)), "p99": float(np.quantile(np.abs(la), 0.99))}
    return out


def tolerance_pass(quantities: dict, tol: float | None, keys=ARRAYS) -> bool | None:
    if tol is None:
        return None
    worst = [q["max"] for k, q in quantities.items() if k in keys]
    return bool(worst) and all(np.isfinite(w) and w <= tol for w in worst) and all(
        quantities[k]["n_nonfinite"] == 0 for k in quantities if k in keys)  # fmt: skip


# ---------------------------------------------------------------------------------------------------- V5
def parse_pred_path(path: Path) -> dict:
    """<root>/<dataset>/fold_<k>/<model>.<set>.npz -> dataset, fold, model, set."""
    model, _, name = path.name.removesuffix(".npz").partition(".")
    return {"dataset": path.parent.parent.name, "fold": path.parent.name.removeprefix("fold_"), "model": model,
            "set": name}  # fmt: skip


def strand_rows(path: Path) -> list[dict]:
    preds = pc.read_predictions(path)
    where = parse_pred_path(path)
    out = []
    for allele in pc.ALLELES:
        fwd, rev = preds.get(f"{allele}_fwd"), preds.get(f"{allele}_rev")
        if fwd is None or rev is None:
            continue
        s = diff_stats(fwd, rev)
        out.append({**where, "quantity": f"{allele}_fwd_vs_rev", "n": s["n"], "pearson": s["pearson"],
                    "spearman": spearman(fwd, rev), "p50": s["p50"], "p99": s["p99"], "max": s["max"]})  # fmt: skip
    if all(f"{a}_{s}" in preds for a in pc.ALLELES for s in pc.STRANDS):
        lf = (preds["a2_fwd"].astype(np.float64) - preds["a1_fwd"]) / pc.LN2
        lr = (preds["a2_rev"].astype(np.float64) - preds["a1_rev"]) / pc.LN2
        la = logfc(preds)
        for quantity, x, y in (("logfc_fwd_vs_rev", lf, lr), ("logfc_fwd_vs_average", lf, la)):
            s = diff_stats(x, y)
            out.append({**where, "quantity": quantity, "n": s["n"], "pearson": s["pearson"],
                        "spearman": spearman(x, y), "p50": s["p50"], "p99": s["p99"], "max": s["max"]})  # fmt: skip
        out.append({**where, "quantity": "abs_logfc_average", "n": len(la), "pearson": float("nan"),
                    "spearman": float("nan"), "p50": float(np.quantile(np.abs(la), 0.5)),
                    "p99": float(np.quantile(np.abs(la), 0.99)), "max": float(np.abs(la).max(initial=0.0))})  # fmt: skip
    return out


def summarise_strands(rows: list[dict]) -> dict:
    """Per model and quantity: the files, and the median / worst Pearson and p99 over them."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["model"], r["quantity"]), []).append(r)
    out = {}
    for (model, quantity), rs in sorted(groups.items()):
        p = np.array([r["pearson"] for r in rs], np.float64)
        q = np.array([r["p99"] for r in rs], np.float64)
        out.setdefault(model, {})[quantity] = {
            "files": len(rs), "pearson_median": _f(np.nanmedian(p)) if np.isfinite(p).any() else float("nan"),
            "pearson_min": _f(np.nanmin(p)) if np.isfinite(p).any() else float("nan"),
            "p99_median": _f(np.nanmedian(q)), "p99_max": _f(np.nanmax(q))}  # fmt: skip
    return out


# ---------------------------------------------------------------------------------------------------- V2
def read_tsv(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def vs_list(list_path: Path, windows: Path, fai: Path, n: int, seed: int, out_dir: Path) -> dict:
    with np.load(windows, allow_pickle=False) as z:
        kept = set(np.asarray(z["ids"]).astype(str).tolist())
    header, rows = read_tsv(list_path)
    missing = [c for c in ("variant_id", "chrom", "pos", "ref", "alt") if c not in header]
    if missing:
        raise ValueError(f"{list_path}: no column {', '.join(missing)}")
    candidates = [r for r in rows if r["variant_id"] in kept]
    if not candidates:
        raise ValueError(f"no row of {list_path} is in {windows}")
    pick = np.sort(np.random.default_rng(seed).choice(len(candidates), size=min(n, len(candidates)), replace=False))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "vs_variants.tsv", "w") as fh:
        for i in pick.tolist():
            r = candidates[i]
            fh.write("\t".join([r["chrom"], r["pos"], r["ref"], r["alt"], r["variant_id"]]) + "\n")
    with open(fai) as src, open(out_dir / "chrom.sizes", "w") as dst:
        for line in src:
            name, length = line.split("\t")[:2]
            dst.write(f"{name}\t{length}\n")
    picked = [candidates[i] for i in pick.tolist()]
    n_indel = sum(len(r["ref"]) != len(r["alt"]) for r in picked)
    info = {"list": str(list_path), "windows": str(windows), "rows_in_list": len(rows), "rows_built": len(candidates),
            "picked": len(picked), "indels": n_indel, "seed": seed, "schema": "chr pos allele1 allele2 variant_id",
            "files": [str(out_dir / "vs_variants.tsv"), str(out_dir / "chrom.sizes")]}  # fmt: skip
    pc.write_json(out_dir / "vs_variants.json", info)
    return info


def vs_compare(vs_scores: Path, preds_path: Path, tol: float) -> dict:
    header, rows = read_tsv(vs_scores)
    missing = [c for c in VS_SCORE_COLUMNS if c not in header]
    if missing:
        raise ValueError(f"{vs_scores}: no column {', '.join(missing)}")
    preds = pc.read_predictions(preds_path)
    index = {v: i for i, v in enumerate(np.asarray(preds["ids"]).astype(str).tolist())}
    joined = [(index[r["variant_id"]], r) for r in rows if r["variant_id"] in index]
    if not joined:
        raise ValueError(f"no variant of {vs_scores} is in {preds_path}")
    idx = np.array([i for i, _ in joined])
    vs = {c: np.array([float(r[c]) for _, r in joined]) for c in VS_SCORE_COLUMNS[1:]}
    ours_logfc = logfc({k: preds[k][idx] for k in ("a1_logcounts", "a2_logcounts")})
    with np.errstate(divide="ignore", invalid="ignore"):
        quantities = {
            "logfc": diff_stats(vs["logfc"], ours_logfc),
            "a1_logcounts": diff_stats(np.log(vs["allele1_pred_counts"]), preds["a1_logcounts"][idx]),
            "a2_logcounts": diff_stats(np.log(vs["allele2_pred_counts"]), preds["a2_logcounts"][idx]),
        }
    p99 = quantities["logfc"]["p99"]
    return {"vs_scores": str(vs_scores), "preds": str(preds_path), "n_vs": len(rows), "n_ours": len(index),
            "n_joined": len(joined), "quantities": quantities, "tol": tol,
            "pass": bool(np.isfinite(p99) and p99 <= tol), "pass_max": bool(quantities["logfc"]["max"] <= tol)}  # fmt: skip


# ---------------------------------------------------------------------------------------------------- summary
def summary_rows(payload: dict) -> list[dict]:
    check = payload.get("check", "?")
    out = []
    if check == "strands":
        # One row per model and quantity: n = files, p99 = the worst file's p99, pearson = the lowest file's.
        for model, quantities in payload.get("by_model", {}).items():
            for quantity, s in quantities.items():
                out.append({"check": "V5", "label": f"{model} (worst of {s['files']} files)", "quantity": quantity,
                            "n": s["files"], "p50": None, "p99": s["p99_max"], "max": None,
                            "pearson": s["pearson_min"], "tol": None, "pass": None})  # fmt: skip
        return out
    for quantity, s in payload.get("quantities", {}).items():
        out.append({"check": check, "label": payload.get("label", ""), "quantity": quantity, "n": s["n"],
                    "p50": s["p50"], "p99": s["p99"], "max": s["max"], "pearson": s["pearson"],
                    "tol": payload.get("tol"), "pass": payload.get("pass")})  # fmt: skip
    return out


def write_rows(path: Path, columns, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with open(partial, "w") as fh:
        fh.write("\t".join(columns) + "\n")
        for r in rows:
            fh.write("\t".join(_cell(r.get(c)) for c in columns) + "\n")
    os.replace(partial, path)


def _cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return "nan" if not np.isfinite(v) else f"{v:.6g}"
    return str(v)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pair", help="V3 / V4: two prediction files of the same rows")
    p.add_argument("--a", type=Path, required=True)
    p.add_argument("--b", type=Path, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--check", default="pair", help="name in the summary (e.g. V3, V4)")
    p.add_argument("--tol", type=float, help="pass when every log-count array's max |a - b| is within this")
    p.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("strands", help="V5: forward vs reverse strand of every prediction file")
    s.add_argument("--preds-root", type=Path, required=True, help="holds <dataset>/fold_<k>/<model>.<set>.npz")
    s.add_argument("--out", type=Path, required=True, help="JSON; a .tsv with one row per file goes beside it")
    v = sub.add_parser("vs-list", help="V2: VS main's input files")
    v.add_argument("--list", type=Path, required=True, help="the variant list TSV (header row)")
    v.add_argument("--windows", type=Path, required=True, help="build_windows.py's variant npz (rows it kept)")
    v.add_argument("--fai", type=Path, required=True)
    v.add_argument("--n", type=int, default=2000)
    v.add_argument("--seed", type=int, default=1)
    v.add_argument("--out-dir", type=Path, required=True)
    c = sub.add_parser("vs-compare", help="V2: VS main's scores against ours")
    c.add_argument("--vs-scores", type=Path, required=True)
    c.add_argument("--preds", type=Path, required=True)
    c.add_argument("--tol", type=float, default=1e-3)
    c.add_argument("--label", default="VS main vs ours")
    c.add_argument("--out", type=Path, required=True)
    m = sub.add_parser("summary", help="flatten check JSONs into one TSV")
    m.add_argument("--inputs", type=Path, nargs="+", required=True)
    m.add_argument("--out", type=Path, required=True)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    pc.setup_logging(args.verbose)
    try:
        if args.command == "pair":
            result = compare_pair(pc.read_predictions(args.a), pc.read_predictions(args.b))
            payload = {"check": args.check, "label": args.label, "a": str(args.a), "b": str(args.b), **result,
                       "tol": args.tol, "pass": tolerance_pass(result["quantities"], args.tol)}  # fmt: skip
            pc.write_json(args.out, payload)
            lf = result["quantities"].get("logfc")
            logger.info("%s: max |d| %s; logFC p99 %s%s", args.label,
                        ", ".join(f"{k} {q['max']:.3g}" for k, q in result["quantities"].items() if k in ARRAYS),
                        f"{lf['p99']:.3g}" if lf else "-",
                        "" if args.tol is None else f"; tol {args.tol:g}: {'pass' if payload['pass'] else 'FAIL'}")  # fmt: skip
            return OUTSIDE_TOLERANCE if payload["pass"] is False else 0
        if args.command == "strands":
            files = sorted(args.preds_root.glob("*/fold_*/*.npz"))
            if not files:
                raise ValueError(f"no prediction files under {args.preds_root}")
            rows = [r for f in files for r in strand_rows(f)]
            write_rows(args.out.with_suffix(".tsv"),
                       ("dataset", "fold", "model", "set", "quantity", "n", "pearson", "spearman", "p50", "p99", "max"),
                       rows)  # fmt: skip
            pc.write_json(args.out, {"check": "strands", "label": "V5", "preds_root": str(args.preds_root),
                                     "files": len(files), "by_model": summarise_strands(rows)})  # fmt: skip
            logger.info("V5: %d files, %d rows -> %s", len(files), len(rows), args.out.with_suffix(".tsv"))
            return 0
        if args.command == "vs-list":
            info = vs_list(args.list, args.windows, args.fai, args.n, args.seed, args.out_dir)
            logger.info("V2 inputs: %d variants (%d indels) of %d built -> %s", info["picked"], info["indels"],
                        info["rows_built"], args.out_dir)  # fmt: skip
            return 0
        if args.command == "vs-compare":
            payload = {"check": "V2", "label": args.label, **vs_compare(args.vs_scores, args.preds, args.tol)}
            pc.write_json(args.out, payload)
            q = payload["quantities"]["logfc"]
            logger.info("V2: %d joined of %d VS rows; |logfc diff| p99 %.3g, max %.3g (tol %g): %s",
                        payload["n_joined"], payload["n_vs"], q["p99"], q["max"], args.tol,
                        "pass" if payload["pass"] else "FAIL")  # fmt: skip
            return 0 if payload["pass"] else OUTSIDE_TOLERANCE
        rows = []
        for path in args.inputs:
            try:
                rows.extend(summary_rows(json.loads(path.read_text())))
            except (OSError, json.JSONDecodeError) as err:
                logger.warning("%s: skipped (%s)", path, err)
        write_rows(args.out, SUMMARY_COLUMNS, rows)
        logger.info("%d rows -> %s", len(rows), args.out)
        return 0
    except (OSError, ValueError, KeyError) as err:
        logger.error("%s: %s", args.command, err)
        return 1


if __name__ == "__main__":
    sys.exit(main())
