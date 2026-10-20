#!/usr/bin/env python3
"""Cherimoya's predicted effect of QTL lead variants, scored in a window centred on the QTL's peak or on the variant.

scores.py, like kundajelab/variant-scorer, centres every window on the variant, so the model reports the variant's
effect on accessibility at the variant's own site. A QTL's effect is measured at its phenotype (a peak), which for
most leads lies elsewhere. Centred on the peak, with the variant written in at its real offset, the model's
counts are the peak's, and any variant inside the model's input can change them.

Windows (in_window bp, read from the model; uint8 codes A=0, C=1, G=2, T=3, anything else 4, as build_windows.py)
  The window's anchor is a 1-based base: the variant's position for --centre variant, the phenotype's middle base for
  --centre peak, floor((start + end) / 2) + 1 of its chrom:start-end (start 0-based, end exclusive; the 251st base of
  a 501-bp peak). The window is genome[anchor - 1 - half, anchor - 1 - half + in_window), half = in_window // 2, the
  layout of build_windows.py (the allele at index 1057 of a 2114-bp variant window). The variant's REF and ALT are
  written at index pos - anchor + half, after checking that the genome base there is REF. Rows whose variant is not
  inside the window (|pos - anchor| > half - 1) are left out and counted, as are windows that run off a chromosome.
  Leads are SNVs (qtl_compare.py's loader drops multi-base alleles and puts each variant on the genome's alleles).

Scores (as scores.py; only the counts)
  Each model predicts both strands of the REF and ALT windows; the strand-averaged log counts are
  log((exp(fwd) + exp(rev)) / 2), and logFC = log2 of the ALT over the REF counts. A QTL row's prediction is the mean,
  over the --datasets of its group (--groups), of the fold mean of logFC (`mean`), or of fold 0 alone (`fold0`).

Checks
  --check-summaries ROOT: for --centre variant, each dataset's fold-mean logFC is compared with scores.py's
  ROOT/<dataset>/cherimoya.variants.summary logfc.mean for the same variant; the largest difference is recorded.
  --selftest: windows, offsets, REF checks and the reach on a toy genome (no model).

Outputs, under --out-dir
  preds.tsv.gz   variant_id, phenotype_id, dataset, centre, fold, logfc
  rows.tsv.gz    one row per QTL row: variant_id, group, phenotype_id, dist_to_centre, effect, and pred.<centre>.<set>
  bins.tsv       per centre, fold set and distance bin: rows with a prediction, and the share whose sign matches
  run.json       settings, counts, models, checks and times

    python workflows/variant_scoring/peak_scores.py --qtl <table.csv> --fasta <genome.fa> --groups <groups.tsv> \\
        --datasets <d1> <d2> ... --model-root <root> --folds 0 1 2 3 4 --centre peak variant --out-dir <dir>
    python workflows/variant_scoring/peak_scores.py --selftest
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qtl_compare as qc  # noqa: E402

logger = logging.getLogger("peak_scores")
CODE = np.full(256, 4, dtype=np.uint8)
for _i, _b in enumerate("ACGT"):
    CODE[ord(_b)] = CODE[ord(_b.lower())] = _i
BINS = ((250, "<=250"), (500, "250-500"), (1000, "500-1000"), (2000, "1000-2000"), (5000, "2000-5000"),
        (math.inf, ">5000"))  # fmt: skip
LN2 = math.log(2.0)


class Genome:
    """Base codes of genome intervals, by seeking in an indexed FASTA (its .fai)."""

    def __init__(self, path: Path):
        self.index: dict[str, tuple[int, int, int, int]] = {}
        with open(f"{path}.fai") as fh:
            for line in fh:
                name, length, offset, bases, width = line.split("\t")[:5]
                self.index[name] = (int(length), int(offset), int(bases), int(width))
        self.fh = open(path, "rb")

    def codes(self, chrom: str, start0: int, end0: int) -> np.ndarray | None:
        """uint8 codes of genome[start0, end0) (0-based, end exclusive), or None off the chromosome."""
        if chrom not in self.index:
            return None
        length, offset, bases, width = self.index[chrom]
        if start0 < 0 or end0 > length:
            return None
        first, last = start0 // bases, (end0 - 1) // bases
        self.fh.seek(offset + first * width)
        raw = self.fh.read((last - first + 1) * width)
        seq = raw.replace(b"\n", b"").replace(b"\r", b"")
        lo = start0 - first * bases
        return CODE[np.frombuffer(seq[lo : lo + end0 - start0], dtype=np.uint8)]


def anchor_of(phenotype: str) -> tuple[str, int]:
    """chrom:start-end -> (chrom, its middle base, 1-based)."""
    chrom, span = phenotype.strip().rsplit(":", 1)
    start, end = (int(float(x)) for x in span.split("-"))
    return chrom, (start + end) // 2 + 1


def build_windows(genome: Genome, keys: list[tuple[str, str]], centre: str, in_window: int):
    """REF and ALT windows for (variant_id, phenotype_id) keys. Returns (kept indexes, ref codes, alt codes, counts)."""
    half = in_window // 2
    kept, ref_rows, alt_rows = [], [], []
    counts = {"keys": len(keys), "kept": 0, "out_of_window": 0, "off_chromosome": 0, "ref_mismatch": 0,
              "other_chrom": 0}  # fmt: skip
    for i, (variant, phenotype) in enumerate(keys):
        chrom, pos, ref, alt = qc.parse_variant(variant)
        if centre == "peak":
            pchrom, anchor = anchor_of(phenotype)
            if pchrom != chrom:
                counts["other_chrom"] += 1
                continue
        else:
            anchor = pos
        offset = pos - anchor + half
        if not 0 <= offset < in_window:
            counts["out_of_window"] += 1
            continue
        w = genome.codes(chrom, anchor - 1 - half, anchor - 1 - half + in_window)
        if w is None:
            counts["off_chromosome"] += 1
            continue
        if w[offset] != CODE[ord(ref)]:
            counts["ref_mismatch"] += 1
            continue
        a = w.copy()
        a[offset] = CODE[ord(alt)]
        kept.append(i)
        ref_rows.append(w)
        alt_rows.append(a)
    counts["kept"] = len(kept)
    shape = (0, in_window)
    ref = np.stack(ref_rows) if ref_rows else np.zeros(shape, np.uint8)
    alt = np.stack(alt_rows) if alt_rows else np.zeros(shape, np.uint8)
    return np.array(kept, dtype=np.int64), ref, alt, counts


def strand_mean(fwd: np.ndarray, rev: np.ndarray) -> np.ndarray:
    """log((exp(fwd) + exp(rev)) / 2), as scores.py averages the strands."""
    return np.logaddexp(fwd.astype(np.float64), rev.astype(np.float64)) - LN2


def bin_of(d: float) -> str:
    for edge, name in BINS:
        if d <= edge:
            return name
    return BINS[-1][1]


def concordance(pred: np.ndarray, effect: np.ndarray) -> tuple[int, float]:
    ok = np.isfinite(pred) & np.isfinite(effect) & (pred != 0) & (effect != 0)
    return int(ok.sum()), (float(np.mean(np.sign(pred[ok]) == np.sign(effect[ok]))) if ok.any() else math.nan)


def run(args) -> int:
    t0 = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fasta = qc.Fasta(args.fasta)
    qargs = qc.build_parser().parse_args(["--qtl", str(args.qtl), "--fasta", str(args.fasta), "--groups",
                                          str(args.groups), "--out-dir", str(out)])  # fmt: skip
    qtl = qc.load_qtl(qargs, fasta)
    groups = qc.read_groups(args.groups)
    datasets = list(args.datasets)
    genome = Genome(args.fasta)
    info: dict = {"started_et": qc.now_et(), "argv": sys.argv, "code": qc.code_commit(), "qtl_counts": qtl.counts,
                  "datasets": datasets, "folds": args.folds, "centres": args.centre, "models": {}, "windows": {},
                  "checks": {}}  # fmt: skip
    # which rows each dataset predicts: rows of the groups it stands for; one window set per unique (variant, peak)
    by_ds: dict[str, list[tuple[str, str]]] = {}
    for ds in datasets:
        gs = {g for g, dl in groups.items() if ds in dl}
        sel = np.isin(qtl.group, list(gs))
        by_ds[ds] = sorted(set(zip(qtl.variant[sel], qtl.phenotype[sel])))
    preds: dict[tuple[str, str, str, int], float] = {}  # (variant, phenotype, dataset, fold, centre) -> logFC
    import predict_cherimoya as pcm

    pcm.set_precision("default")
    for ds in datasets:
        keys = by_ds[ds]
        if not keys:
            logger.info("%s: no QTL rows of its groups", ds)
            continue
        for fold in args.folds:
            path = Path(args.model_root) / ds / f"fold_{fold}" / f"cherimoya.fold_{fold}.torch"
            if not path.is_file():
                raise FileNotFoundError(f"no model {path}")
            model = pcm.load_model(path, args.device)
            in_window = int(model.trimming) * 2 + int(args.out_window)
            info["models"][f"{ds}/fold_{fold}"] = {"path": str(path), "n_layers": int(model.n_layers),
                                                   "in_window": in_window}  # fmt: skip
            predictor = pcm.CountPredictor(model, args.device)
            for centre in args.centre:
                tag = f"{ds}/{centre}/{in_window}"
                kept, ref, alt, counts = build_windows(genome, keys, centre, in_window)
                info["windows"][tag] = counts
                if counts["ref_mismatch"]:
                    raise ValueError(f"{tag}: {counts['ref_mismatch']} windows whose genome base is not REF")
                t1 = time.time()
                r = predictor.predict(ref, args.batch)
                a = predictor.predict(alt, args.batch)
                lfc = (strand_mean(a["fwd"], a["rev"]) - strand_mean(r["fwd"], r["rev"])) / LN2
                for j, v in zip(kept, lfc):
                    preds[(keys[j][0], keys[j][1], ds, centre, fold)] = float(v)
                logger.info("%s fold %d: %d windows in %.0f s", tag, fold, len(kept), time.time() - t1)
            del predictor, model
    # long table
    with gzip.open(out / "preds.tsv.gz", "wt") as fh:
        fh.write("variant_id\tphenotype_id\tdataset\tcentre\tfold\tlogfc\n")
        for (v, p, ds, c, f), x in sorted(preds.items()):
            fh.write(f"{v}\t{p}\t{ds}\t{c}\t{f}\t{x:.6g}\n")
    # per dataset: fold mean, and fold 0
    sets = {"mean": list(args.folds)}
    if 0 in args.folds and len(args.folds) > 1:
        sets["fold0"] = [0]
    per_ds: dict[tuple[str, str, str, str, str], float] = {}
    for (v, p, ds, c, f), x in preds.items():
        for s, fl in sets.items():
            if f in fl:
                per_ds.setdefault((v, p, ds, c, s), []).append(x)
    per_ds = {k: float(np.mean(v)) for k, v in per_ds.items() if len(v) == len(sets[k[4]])}
    # the check against scores.py's summaries (variant-centred, every fold)
    if args.check_summaries and "variant" in args.centre:
        import scores as sc

        worst, n = 0.0, 0
        for ds in datasets:
            p = qc.summary_path([Path(args.check_summaries)], ds, "cherimoya", "variants")
            if p is None:
                continue
            tab = sc.read_table(p, ("variant_id", "logfc.mean"))
            ref = dict(zip(tab["variant_id"], tab["logfc.mean"]))
            for (v, _ph, d, c, s), x in per_ds.items():
                if d == ds and c == "variant" and s == "mean" and v in ref:
                    worst, n = max(worst, abs(x - float(ref[v]))), n + 1
        info["checks"]["variant_centred_vs_summaries"] = {"compared": n, "max_abs_diff": worst}
        logger.info("check: %d variant-centred fold means against scores.py's, max |diff| %.3g", n, worst)
    # per QTL row: mean over the group's datasets
    cols = [f"pred.{c}.{s}" for c in args.centre for s in sets]
    row_pred = {c: np.full(len(qtl.variant), np.nan) for c in cols}
    for i in range(len(qtl.variant)):
        dsl = [d for d in groups.get(qtl.group[i], []) if d in datasets]
        for c in args.centre:
            for s in sets:
                vals = [per_ds[k] for d in dsl if (k := (qtl.variant[i], qtl.phenotype[i], d, c, s)) in per_ds]
                if vals and len(vals) == len(dsl):
                    row_pred[f"pred.{c}.{s}"][i] = float(np.mean(vals))
    with gzip.open(out / "rows.tsv.gz", "wt") as fh:
        fh.write("\t".join(["variant_id", "group", "phenotype_id", "dist_to_centre", "effect"] + cols) + "\n")
        for i in range(len(qtl.variant)):
            vals = [qtl.variant[i], qtl.group[i], qtl.phenotype[i], qc.fmt(float(qtl.dist[i])),
                    qc.fmt(float(qtl.effect[i]))] + [qc.fmt(float(row_pred[c][i])) for c in cols]  # fmt: skip
            fh.write("\t".join(vals) + "\n")
    labels = np.array([bin_of(d) for d in qtl.dist])
    with open(out / "bins.tsv", "w") as fh:
        fh.write("centre\tset\tbin\tleads\tscored\tconcordance\n")
        for c in args.centre:
            for s in sets:
                pred = row_pred[f"pred.{c}.{s}"]
                for _, b in BINS:
                    m = labels == b
                    n, conc = concordance(pred[m], qtl.effect[m])
                    fh.write(f"{c}\t{s}\t{b}\t{int(m.sum())}\t{n}\t{qc.fmt(conc)}\n")
    info["seconds"] = round(time.time() - t0, 1)
    info["finished_et"] = qc.now_et()
    (out / "run.json").write_text(json.dumps(info, indent=1, default=str) + "\n")
    logger.info("wrote %s in %.0f s", out, time.time() - t0)
    return 0


def selftest() -> int:
    """Windows, offsets, REF checks and the reach on a toy genome; no model."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        rng = np.random.default_rng(0)
        seq = "".join(rng.choice(list("ACGT"), 6000))
        with open(d / "g.fa", "w") as fh:
            fh.write(">chr1\n" + "\n".join(seq[i : i + 60] for i in range(0, len(seq), 60)) + "\n")
        (d / "g.fa.fai").write_text(f"chr1\t{len(seq)}\t6\t60\t61\n")
        g = Genome(d / "g.fa")
        assert "".join("ACGT"[c] for c in g.codes("chr1", 100, 250)) == seq[100:250]
        assert g.codes("chr1", -1, 10) is None and g.codes("chr1", 5990, 6001) is None
        half, L = 1057, 2114
        pos = 3000
        ref = seq[pos - 1]
        alt = "A" if ref != "A" else "C"
        vid = f"chr1:{pos}:{ref}:{alt}"
        # variant-centred: the allele at index 1057, the window genome[pos - 1 - 1057, pos - 1 + 1057)
        kept, r, a, cnt = build_windows(g, [(vid, "chr1:0-10")], "variant", L)
        assert cnt["kept"] == 1 and "".join("ACGT"[c] for c in r[0]) == seq[pos - 1 - half : pos - 1 + half]
        assert a[0][half] == CODE[ord(alt)] and (a[0] != r[0]).sum() == 1
        # peak-centred: peak chr1:3399-3900 (501 bp) has anchor (3399 + 3900) // 2 + 1 = 3650; offset pos - 3650 + 1057
        kept, r, a, cnt = build_windows(g, [(vid, "chr1:3399-3900")], "peak", L)
        assert anchor_of("chr1:3399-3900") == ("chr1", 3650)
        assert cnt["kept"] == 1 and a[0][pos - 3650 + half] == CODE[ord(alt)] and (a[0] != r[0]).sum() == 1
        assert "".join("ACGT"[c] for c in r[0]) == seq[3650 - 1 - half : 3650 - 1 + half]
        # a variant at the anchor gives the variant-centred window exactly
        vid2 = f"chr1:3650:{seq[3649]}:{'A' if seq[3649] != 'A' else 'C'}"
        _, r1, a1, _ = build_windows(g, [(vid2, "chr1:3399-3900")], "peak", L)
        _, r2, a2, _ = build_windows(g, [(vid2, "x")], "variant", L)
        assert (r1 == r2).all() and (a1 == a2).all()
        # out of the window, and a wrong REF
        far = f"chr1:{3650 + half}:{seq[3650 + half - 1]}:{'A' if seq[3650 + half - 1] != 'A' else 'C'}"
        _, _, _, cnt = build_windows(g, [(far, "chr1:3399-3900")], "peak", L)
        assert cnt["out_of_window"] == 1, cnt
        bad = f"chr1:{pos}:{alt}:{ref}"
        _, _, _, cnt = build_windows(g, [(bad, "x")], "variant", L)
        assert cnt["ref_mismatch"] == 1, cnt
        # strand mean and logFC as scores.py: log of the mean of the counts
        assert abs(strand_mean(np.array([math.log(2.0)]), np.array([math.log(4.0)]))[0] - math.log(3.0)) < 1e-12
        assert bin_of(250) == "<=250" and bin_of(250.5) == "250-500" and bin_of(9000) == ">5000"
    print("peak_scores selftest: all checks passed")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--qtl", type=Path)
    ap.add_argument("--fasta", type=Path)
    ap.add_argument("--groups", type=Path)
    ap.add_argument("--datasets", nargs="+", default=[])
    ap.add_argument("--model-root", type=Path, help="<root>/<dataset>/fold_<k>/cherimoya.fold_<k>.torch")
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--centre", nargs="+", choices=("peak", "variant"), default=["peak", "variant"])
    ap.add_argument("--out-window", type=int, default=1000, help="the models' output window (in = out + 2 x trim)")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--check-summaries", type=Path, help="scores.py's root, to check the variant-centred fold means")
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--selftest", action="store_true")
    return ap


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    if args.selftest:
        return selftest()
    for name in ("qtl", "fasta", "groups", "model_root", "out_dir"):
        if getattr(args, name) is None:
            build_parser().error(f"--{name.replace('_', '-')} is required")
    if not args.datasets:
        build_parser().error("--datasets is required")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
