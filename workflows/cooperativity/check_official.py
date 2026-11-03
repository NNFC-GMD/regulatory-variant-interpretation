#!/usr/bin/env python3
"""Check combine.py --official: the official deepISA package's aggregate stage run again by hand on the tables
combine.py wrote, and its rules recomputed here from coop_isa.py's tables.

Per dataset (default: every <root>/official/<dataset>/):
  package  copies motif_single_isa.csv and motif_combi_isa.csv to a temporary directory (calc_interaction rewrites the
           combi table in place), runs calc_interaction(isa_thresh_quantile=0.1, normalize=False) and
           calc_coop_score(level="tf_pair" and "tf", min_count=10, q_val_thresh=0.1) as QuickStart's aggregate stage
           does, and compares both tables with the ones combine.py kept (<root>/official/coop_tf_pairs_by_dataset.tsv,
           coop_tfs_by_dataset.tsv): the same rows and labels, the numbers within --rtol;
  inputs   the two CSVs against coop_isa.py's <root>/<dataset>/pairs.tsv.gz and sites.tsv.gz: one single row per site;
           one combi row per pair with 1 <= gap <= --receptive-field, tf1 the TF of motif1, isa1_t0, isa2_t0 and
           isa_both_t0 the pair's e_a, e_b and isa_both, distance its gap;
  rules    the package's rules recomputed here from the pairs: the 10th percentile of the non-negative e over the
           sites; the qualified pairs (both e at least that, isa_both - e_b >= 0, isa_both - e_a >= 0); i = e_a + e_b -
           isa_both (written with 6 decimals, as calc_interaction writes it); per TF pair and TF with >= 10 pairs: n,
           coop_score = sum(i) / sum(|i|), the two-sided Mann-Whitney p against the qualified pairs more than 150 bp
           apart, BH, and the labels (Synergistic above the 70th percentile of the significant scores, Redundant below
           the 30th, Intermediate between, Independent where q >= 0.1, its score then NaN). Compared with the package's
           tables from the first check;
  groups   each group's coop_a and coop_b in <root>/official/changes_pairs.tsv and changes_tfs.tsv, recomputed as
           sum(i) / sum(|i|) over its datasets' qualified pairs.
Prints one line per check and dataset, and exits 1 if any fails.

    check_official.py --root <coop_isa.py --out-dir> --motifs <motifs.tsv> --groups <groups.tsv> [--datasets ...]
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import combine as cb  # noqa: E402

NUMBERS = ("n", "abs_i_sum", "coop_score", "p_val", "median_distance", "q_val")


def package_tables(official: Path, tmp: Path) -> tuple[dict, pd.DataFrame]:
    """The package's aggregate stage on copies of one dataset's two CSVs: ({level: table}, the scored combi table)."""
    from deepISA.score import aggregate_isa as dag

    single, combi = tmp / "motif_single_isa.csv", tmp / "motif_combi_isa.csv"
    shutil.copy(official / single.name, single)
    shutil.copy(official / combi.name, combi)
    with cb._object_strings():
        dag.calc_interaction(combi_isa_path=str(combi), tracks=[0], tau_map=None,
                             isa_thresh_quantile=cb.OFFICIAL["isa_thresh_quantile"], single_isa_path=str(single),
                             normalize=False)  # fmt: skip
        tables = {level: dag.calc_coop_score(str(combi), outpath=str(tmp / f"coop_{level}.csv"), level=level,
                                             track_idx=0, min_count=cb.OFFICIAL["min_count"],
                                             q_val_thresh=cb.OFFICIAL["q_val_thresh"])
                  for level in ("tf_pair", "tf")}  # fmt: skip
    return tables, pd.read_csv(combi)


def same_table(got: pd.DataFrame, want: pd.DataFrame, key: str, rtol: float, atol: float) -> list[str]:
    """What differs between two coop tables (rows by `key`): missing rows, labels, numbers beyond the tolerance."""
    got, want = got.set_index(key).sort_index(), want.set_index(key).sort_index()
    errors = []
    if list(got.index) != list(want.index):
        errors.append(f"rows differ: {sorted(set(got.index) ^ set(want.index))[:5]}")
        common = got.index.intersection(want.index)
        got, want = got.loc[common], want.loc[common]
    bad = got["cooperativity"] != want["cooperativity"]
    if bad.any():
        errors.append(f"{int(bad.sum())} labels differ, e.g. {list(got.index[bad][:3])}")
    for c in NUMBERS:
        a, b = got[c].to_numpy(dtype=float), want[c].to_numpy(dtype=float)
        off = ~np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
        if off.any():
            k = int(np.argmax(off))
            errors.append(f"{c}: {int(off.sum())} differ, e.g. {got.index[k]}: {a[k]:.6g} vs {b[k]:.6g}")
    return errors


def rules(pairs: pd.DataFrame, sites: pd.DataFrame, names: dict, receptive_field: int):
    """The package's rules recomputed from coop_isa.py's tables: (the pairs within reach, with i where qualified and
    NaN elsewhere; {level: the coop table})."""
    from scipy.stats import mannwhitneyu

    e = sites["e"].to_numpy()
    thresh = np.quantile(e[e >= 0], cb.OFFICIAL["isa_thresh_quantile"])
    p = pairs[(pairs["gap"] >= 1) & (pairs["gap"] <= receptive_field)].copy()
    for x in ("a", "b"):
        p[f"tf_{x}"] = p[f"motif_{x}"].map(names).fillna(p[f"motif_{x}"])
    ea, eb, both = (p[c].to_numpy() for c in ("e_a", "e_b", "isa_both"))
    q = (ea >= thresh) & (eb >= thresh) & (both - eb >= 0) & (both - ea >= 0)
    num = ea + eb - both  # the package's isa1 + isa2 - both, which it writes with %.6f and reads back
    p["i"] = [float(f"{v:.6f}") if ok else np.nan for v, ok in zip(num, q)]
    a, b = p["tf_a"].to_numpy(dtype=object), p["tf_b"].to_numpy(dtype=object)
    lo, hi = np.where(a <= b, a, b), np.where(a <= b, b, a)
    null = p.loc[p["gap"] > 150, "i"].dropna().to_numpy()
    out = {}
    for level, labels in (("tf_pair", [lo + "|" + hi]), ("tf", [a, b])):
        long = pd.concat([pd.DataFrame({"name": lab, "i": p["i"].to_numpy(), "distance": p["gap"].to_numpy()})
                          for lab in labels])  # fmt: skip
        rows = []
        for name, g in long.groupby("name"):
            i = g["i"].dropna().to_numpy()
            if len(i) < cb.OFFICIAL["min_count"]:
                continue
            rows.append({level: name, "n": len(i), "abs_i_sum": np.abs(i).sum(), "coop_score": i.sum() / np.abs(i).sum(),
                         "p_val": mannwhitneyu(i, null, alternative="two-sided")[1],
                         "median_distance": g["distance"].median()})  # fmt: skip
        t = pd.DataFrame(rows)
        t["q_val"] = cb.bh(t["p_val"])
        sig = t["q_val"] < cb.OFFICIAL["q_val_thresh"]
        top, bottom = t.loc[sig, "coop_score"].quantile(0.7), t.loc[sig, "coop_score"].quantile(0.3)
        t["cooperativity"] = np.select(
            [sig & (t["coop_score"] > top), sig & (t["coop_score"] < bottom), sig], ["Synergistic", "Redundant",
             "Intermediate"], "Independent")  # fmt: skip
        t.loc[~sig, "coop_score"] = np.nan
        out[level] = t
    return p, thresh, out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="coop_isa.py's --out-dir, with official/ (combine.py)")
    ap.add_argument("--motifs", type=Path, required=True, help="coop_isa.py's --motifs (motif, cluster_annotation)")
    ap.add_argument("--groups", type=Path, required=True, help="TSV: group, datasets (comma-separated)")
    ap.add_argument("--datasets", nargs="+", help="default: every <root>/official/<dataset>/")
    ap.add_argument("--receptive-field", type=int, default=255, help="combine.py's --receptive-field")
    ap.add_argument("--rtol", type=float, default=1e-4, help="combine.py keeps 5 significant digits")
    ap.add_argument("--atol", type=float, default=1e-6)
    args = ap.parse_args(argv)
    official = args.root / "official"
    datasets = args.datasets or sorted(d.name for d in official.iterdir() if (d / "motif_combi_isa.csv").is_file())
    names = pd.read_csv(args.motifs, sep="\t", usecols=["motif", "cluster_annotation"]).drop_duplicates("motif")
    names = dict(zip(names["motif"], names["cluster_annotation"].str.replace(r"_\d+$", "", regex=True)))
    kept = {lvl: pd.read_csv(official / f, sep="\t") for lvl, f in (("tf_pair", "coop_tf_pairs_by_dataset.tsv"),
                                                                     ("tf", "coop_tfs_by_dataset.tsv"))}  # fmt: skip
    failed, qualified = 0, {}

    def report(check: str, ds: str, errors: list[str], note: str = "") -> None:
        nonlocal failed
        failed += bool(errors)
        print(f"{'FAIL' if errors else 'ok  '} {check:8s} {ds}: {'; '.join(errors) or note}", flush=True)

    for ds in datasets:
        pairs = pd.read_csv(args.root / ds / "pairs.tsv.gz", sep="\t")
        sites = pd.read_csv(args.root / ds / "sites.tsv.gz", sep="\t")
        with tempfile.TemporaryDirectory() as tmp:
            pkg, scored = package_tables(official / ds, Path(tmp))
        errors = []
        for level in ("tf_pair", "tf"):
            want = kept[level][kept[level]["dataset"] == ds].drop(columns="dataset")
            errors += [f"{level}: {e}" for e in same_table(pkg[level], want, level, args.rtol, args.atol)]
        report("package", ds, errors, f"{len(pkg['tf_pair'])} TF pairs, {len(pkg['tf'])} TFs, the same tables")

        mine, thresh, recomputed = rules(pairs, sites, names, args.receptive_field)
        single = pd.read_csv(official / ds / "motif_single_isa.csv")
        errors = []
        if len(single) != len(sites):
            errors.append(f"{len(single)} single rows for {len(sites)} sites")
        if len(scored) != len(mine):
            errors.append(f"{len(scored)} combi rows for {len(mine)} pairs within reach")
        else:
            tf1 = scored["motif1"].map(names).fillna(scored["motif1"])
            if (tf1 != scored["tf1"]).any():
                errors.append(f"{int((tf1 != scored['tf1']).sum())} rows whose tf1 is not motif1's TF")
            for col, src in (("isa1_t0", "e_a"), ("isa2_t0", "e_b"), ("isa_both_t0", "isa_both"), ("distance", "gap")):
                if not np.allclose(scored[col].to_numpy(dtype=float), mine[src].to_numpy(dtype=float), atol=1e-6):
                    errors.append(f"{col} is not the pairs' {src}")
            same_i = np.isclose(scored["interaction_t0"], mine["i"], atol=1e-6, equal_nan=True)
            if not same_i.all():
                errors.append(f"interaction_t0: {int((~same_i).sum())} rows differ from e_a + e_b - isa_both")
        report("inputs", ds, errors, f"{len(scored):,} pairs within {args.receptive_field} bp, "
               f"{int(scored['interaction_t0'].notna().sum()):,} qualified (isa_thresh {thresh:.4g})")  # fmt: skip

        errors = []
        for level in ("tf_pair", "tf"):
            errors += [f"{level}: {e}" for e in same_table(recomputed[level], pkg[level], level, 1e-9, 1e-12)]
        report("rules", ds, errors, "the package's rules give the package's tables")
        q = mine[np.isfinite(mine["i"])].copy()
        a, b = q["tf_a"].to_numpy(dtype=object), q["tf_b"].to_numpy(dtype=object)
        q["pair"] = np.where(a <= b, a, b) + " + " + np.where(a <= b, b, a)
        qualified[ds] = q

    groups = {r["group"]: [d.strip() for d in r["datasets"].split(",")] for r in cb.read_tsv(args.groups)}
    for level, key in (("pairs", "pair"), ("tfs", "tf")):
        f = official / f"changes_{level}.tsv"
        if not f.is_file():
            continue
        ch, errors, n = pd.read_csv(f, sep="\t"), [], 0
        for side in ("a", "b"):
            for g, rows in ch.groupby(f"group_{side}"):
                if not all(d in qualified for d in groups[g]):
                    continue
                q = pd.concat([qualified[d] for d in groups[g]])
                if key == "tf":
                    q = pd.concat([q.assign(tf=q["tf_a"]), q.assign(tf=q["tf_b"])])
                s = q.groupby(key)["i"].agg(lambda i: i.sum() / np.abs(i).sum())
                got = rows[key].map(s).to_numpy()
                off = ~np.isclose(got, rows[f"coop_{side}"].to_numpy(), rtol=args.rtol, atol=args.atol)
                n += len(rows)
                if off.any():
                    errors.append(f"group {g}: {int(off.sum())} of {len(rows)} coop_{side} differ")
        report("groups", level, errors, f"{n} coop_a and coop_b values")
    print(f"{'FAILED' if failed else 'passed'}: {failed} checks failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
