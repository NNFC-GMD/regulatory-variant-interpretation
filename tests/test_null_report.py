"""null_report.py: the two nulls side by side, on a toy pilot whose numbers can be checked by hand.

The toy: 12 variants on three chromosomes (two folds), a shuffled null of 8 rows (6 from SNVs, 2 from indels) and a
genomic null of 6 SNVs, scored by the four models. The list holds one more variant, first, that the tables lack (as
when build_windows.py drops a row), and the genomic null's windows are stored in reverse order: both are joined on
their ids. The shuffled null's per-fold tables are .npz only (scores.py --no-fold-tsv) and its summaries both; the
genomic null's tables are TSVs only, so both ways of reading a table are taken. |logFC| lies on a grid of 1/64, exact
in float32 and in the tables' 7 digits. Cherimoya's shuffled null is twice as wide as that of ChromBPNet uncorrected;
its genomic null and its variants are as wide as the uncorrected model's, so the rule's branch is the null artefact.
"""

from __future__ import annotations

import csv
import itertools
import math
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflows" / "variant_scoring"))

import null_report as nr  # noqa: E402
import scores as sc  # noqa: E402

FOLDS = (0, 1)
LIST_HEADER = ("variant_id", "chrom", "pos", "ref", "alt", "source", "stratum", "af", "dist_to_summit", "fold_test",
               "mut_class", "most_severe_consequence", "AF_nfe", "AF_afr", "AF_eas", "AF_sas", "AF_amr")  # fmt: skip
# chrom, pos, ref, alt, stratum, mut_class, AF_nfe
VARIANTS = (
    ("chr1", 1000, "C", "T", "summit100", "CpG_ti", "0.00"),
    ("chr1", 1010, "A", "G", "summit100", "ti", "0.01"),
    ("chr1", 1020, "A", "C", "flank500", "tv", "0.30"),
    ("chr1", 1030, "G", "T", "summit100", "CpG_tv", "0.96"),
    ("chr1", 1040, "AT", "A", "summit100", "indel", "NaN"),
    ("chr2", 2000, "C", "T", "flank500", "ti", "0.05"),
    ("chr2", 2010, "G", "A", "summit100", "CpG_ti", "0.50"),
    ("chr2", 2020, "T", "G", "summit100", "tv", "0.00"),
    ("chr2", 2030, "A", "AT", "flank500", "indel", "0.02"),
    ("chr3", 3000, "C", "G", "summit100", "tv", "0.10"),
    ("chr3", 3010, "AC", "GT", "summit100", "mnv", "0.00"),
    ("chr3", 3020, "T", "C", "flank500", "ti", "0.99"),
)
FOLD_OF = {"chr1": 0, "chr2": 1, "chr3": 0}
NULL_DIRS = (("null", "scores"), ("genomic_null", "scores_genomic_null"))
IDS = [f"{c}:{p}:{r}:{a}" for c, p, r, a, *_ in VARIANTS]
SNVS = [i for i, v in enumerate(VARIANTS) if len(v[2]) == len(v[3]) == 1]  # 0, 1, 2, 3, 5, 6, 7, 9, 11
SHUFFLED_SRC = (0, 1, 2, 3, 4, 5, 6, 8)  # null rows 4 and 7 come from indels
SHUFFLED_SNV_ROWS = [0, 1, 2, 3, 5, 6]
SHUFFLED_CPG = (0, 20, 35, 60, 100, 130, 5, 16)
GENOMIC = ("chr1:5000:A:G", "chr1:5100:C:A", "chr2:6000:G:A", "chr2:6100:T:G", "chr9:7000:C:T", "chr3:8000:A:T")
GENOMIC_CPG = (3, 3, 18, 40, 40, 90)
SCALE = {"null": {"cherimoya": 2.0, "cbp_full": 1.0, "cbp_nobias": 1.5, "cbp_bias": 0.25},
         "genomic_null": {"cherimoya": 1.0, "cbp_full": 1.0, "cbp_nobias": 1.5, "cbp_bias": 0.25},
         "variants": {"cherimoya": 1.0, "cbp_full": 1.0, "cbp_nobias": 1.5}}  # fmt: skip
# The variants each model calls (p 0.001; the others 0.5) under each null.
CALLS = {"null": {"cherimoya": {0, 1, 6}, "cbp_full": {0, 6, 7, 9}, "cbp_nobias": {0}},
         "genomic_null": {"cherimoya": {0, 1, 6, 7}, "cbp_full": {0, 6, 7}, "cbp_nobias": {0}}}  # fmt: skip


def abs_logfc(scale: float, n: int, fold: int) -> np.ndarray:
    return scale * (np.arange(n) + 1 + fold) / 64


def signed(values: np.ndarray) -> np.ndarray:
    return values * np.where(np.arange(len(values)) % 2 == 0, 1.0, -1.0)


def windows(cpg: tuple[int, ...], length: int = 260) -> np.ndarray:
    """Base codes with `c` CpGs per row: CGCG...CG, then As."""
    a1 = np.zeros((len(cpg), length), dtype=np.uint8)
    for r, c in enumerate(cpg):
        a1[r, 0 : 2 * c : 2] = 1
        a1[r, 1 : 2 * c : 2] = 2
    return a1


def save(path: Path, columns: dict, tsv: bool, npz: bool) -> None:
    """A scores table as scores.py stores it: the TSV (write_table) and the .npz beside it (floats as float32)."""
    if tsv:
        sc.write_table(path, columns)
    if npz:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(nr.npz_of(path), **{k: v.astype(np.float32) if v.dtype.kind == "f" else v for k, v in columns.items()})


def write_root(root: Path, datasets: tuple[str, ...] = ("dsA",)) -> None:
    rows = ["\t".join(LIST_HEADER), "chr1:900:A:G\tchr1\t900\tA\tG\ttoy\tsummit100\t0.20\t5\t0\tti\tx\t0.20\t\t\t\t"]
    for vid, (chrom, pos, ref, alt, stratum, mut_class, af) in zip(IDS, VARIANTS):
        rows.append("\t".join([vid, chrom, str(pos), ref, alt, "toy", stratum, "" if af == "NaN" else af, "10",
                               str(FOLD_OF[chrom]), mut_class, "intergenic_variant", af, "0.1", "0.1", "0.1",
                               "0.1"]))  # fmt: skip
    (root / "lists").mkdir(parents=True)
    (root / "lists" / "toy.tsv").write_text("\n".join(rows) + "\n")
    src_ids = np.array([IDS[s] for s in SHUFFLED_SRC])
    shuffled_ids = np.array([f"{v}|shuf{1234 + r}" for r, v in enumerate(src_ids.tolist())])
    (root / "windows").mkdir()
    np.savez(root / "windows" / "null.npz", ids=shuffled_ids, src_ids=src_ids, a1=windows(SHUFFLED_CPG))
    np.savez(root / "windows" / "genomic_null.npz", ids=np.array(GENOMIC[::-1]), a1=windows(GENOMIC_CPG[::-1]))
    for (null, scores_dir), ds in itertools.product(NULL_DIRS, datasets):
        base = root / scores_dir / ds
        ids = {"id": shuffled_ids, "src_id": src_ids} if null == "null" else {"id": np.array(GENOMIC)}
        n = len(ids["id"])
        npz = null == "null"
        for model in nr.WIDTH_MODELS:
            for k in FOLDS:
                a = abs_logfc(SCALE[null][model], n, k)
                columns = {**ids, "allele1_logcounts": (np.arange(n) + 2) / 4, "logfc": signed(a), "abs_logfc": a}
                save(base / f"fold_{k}" / f"{model}.{null}.tsv.gz", columns, tsv=not npz, npz=npz)
            if model == "cbp_bias":
                continue
            p = np.where(np.isin(np.arange(12), sorted(CALLS[null][model])), 0.001, 0.5)
            logfc = []
            for k in FOLDS:
                a = abs_logfc(SCALE["variants"][model], 12, k)
                logfc.append(signed(a))
                save(base / f"fold_{k}" / f"{model}.variants.tsv.gz",
                     {"variant_id": np.array(IDS), "allele1_logcounts": (np.arange(12) + 4) / 4, "logfc": signed(a),
                      "abs_logfc": a, "abs_logfc.pval": p}, tsv=not npz, npz=npz)  # fmt: skip
            summary = {"variant_id": np.array(IDS), "logfc.mean": np.mean(logfc, axis=0), "abs_logfc.mean.pval": p}
            save(base / f"{model}.variants.summary.tsv.gz", summary, tsv=True, npz=npz)


def run(root: Path, datasets: tuple[str, ...] = ("dsA",)) -> int:
    return nr.main(["--root", str(root), "--list", str(root / "lists" / "toy.tsv"), "--datasets", *datasets,
                    "--variants", "variants", "--out-dir", str(root / "analysis"), "--folds", "0", "1"])  # fmt: skip


def table(root: Path, name: str) -> list[dict]:
    with open(root / "analysis" / name, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def pick(rows: list[dict], **keys) -> dict:
    got = [r for r in rows if all(r[k] == str(v) for k, v in keys.items())]
    assert len(got) == 1, (keys, len(got))
    return got[0]


def close(text: str, value: float) -> bool:
    """A number read back from an output table (6 significant digits) against its value."""
    return float(text) == pytest.approx(value, rel=1e-5)


def drop_table(path: Path) -> None:
    path.unlink(missing_ok=True)
    nr.npz_of(path).unlink(missing_ok=True)


# --- helpers ----------------------------------------------------------------------------------------------------


def test_maf_bins_take_hundredths():
    cases = {"0.00": (0, 0.0), "1.00": (0, 0.0), "0.01": (1, 0.01), "0.96": (1, 0.04), "0.99": (1, 0.01),
             "0.05": (2, 0.05), "0.95": (2, 0.05), "0.30": (2, 0.3)}  # fmt: skip
    for text, (b, maf) in cases.items():
        got_bin, got_maf = nr.maf_bin(text)
        assert got_bin == b and got_maf == pytest.approx(maf, abs=1e-12), text
    for text in ("NaN", "", "abc", "1.5"):
        b, maf = nr.maf_bin(text)
        assert nr.MAF_BINS[b] == "empty" and math.isnan(maf), text


def test_ids_are_typed_by_their_alleles():
    assert nr.parse_id("chr2:100:A:G") == ("chr2", nr.SNV, True)
    assert nr.parse_id("chr2:100:A:C") == ("chr2", nr.SNV, False)
    assert nr.parse_id("chrX:5:AC:GT") == ("chrX", nr.MNV, False)
    assert nr.parse_id("chr1:5:12>1:0123456789") == ("chr1", nr.INDEL, False)  # the list's compact form
    with pytest.raises(ValueError):
        nr.parse_id("chr1:5:A")
    with pytest.raises(ValueError):  # a shuffled-null row's own id is not typed as an indel: refused
        nr.parse_id("chr1:100:A:G|shuf1234")


def test_decision_branches():
    branch, removed = nr.decide(1.5, 1.1, 1.08)
    assert branch == "null artefact" and removed == pytest.approx(0.4 / 0.42)
    branch, removed = nr.decide(1.5, 1.3, 1.1)
    assert branch == "partly both" and removed == pytest.approx(0.5)
    assert nr.decide(1.5, 1.45, 1.08)[0] == "model property"
    assert nr.decide(1.5, math.nan, 1.1)[0] == "not evaluable"
    assert math.isnan(nr.decide(1.05, 1.3, 1.1)[1])  # no excess over the variants to remove


def test_cpg_counts_and_bins():
    a1 = np.array([[1, 2, 1, 2], [2, 1, 2, 0], [0, 0, 0, 0], [1, 1, 2, 2]], dtype=np.uint8)  # CGCG GCGA AAAA CCGG
    assert nr.cpg_counts(a1, chunk=3).tolist() == [2, 1, 0, 1]
    assert nr.cpg_counts(windows(SHUFFLED_CPG)).tolist() == list(SHUFFLED_CPG)
    assert nr.cpg_bins([0, 14, 15, 29, 30, 119, 120, 500]).tolist() == [0, 0, 1, 1, 2, 4, 5, 5]


def test_weighted_quantile():
    assert nr.weighted_quantile([4, 1, 3, 2], [1, 1, 1, 1], 0.5) == 2
    assert nr.weighted_quantile([4, 1, 3, 2], [1, 1, 1, 1], 0.99) == 4
    assert nr.weighted_quantile([1, 2, 3, 4], [0, 0, 1, 1], 0.5) == 3  # rows of no weight never count
    assert nr.weighted_quantile([1, 2, 3, 4], [3, 1, 0, 0], 0.5) == 1
    assert math.isnan(nr.weighted_quantile([1, 2], [0, 0], 0.5))


def test_ratio_bootstrap_is_seeded_and_brackets_its_point():
    """R: one draw of the SNV rows per replicate, for both models and every fold; the interval holds the point."""
    rng = random.Random(5)
    n = 600
    base = [abs(rng.gauss(0, 1)) for _ in range(n)]
    x = np.array([[v * (1 + 0.2 * rng.random()) for v in base] for _ in range(2)])
    y = np.array([[v * (1 + 0.6 * rng.random()) for v in base] for _ in range(2)])
    snv = np.array([i % 5 != 0 for i in range(n)])
    ti = np.array([i % 3 == 0 for i in range(n)])
    got = nr.ratio_draws(x, y, snv, ti, reps=40, seed=3)
    again = nr.ratio_draws(x, y, snv, ti, reps=40, seed=3)
    for subset in ("SNV", "SNV_Ti", "SNV_Tv"):
        d = got[subset]
        rows = [i for i in range(n) if snv[i] and (subset == "SNV" or bool(ti[i]) == (subset == "SNV_Ti"))]
        assert d["point"] == pytest.approx(nr.mean_q99(x, np.array(rows)) / nr.mean_q99(y, np.array(rows)))
        s = nr.spread(d["draws"])
        assert s["reps"] == 40 and s["lo"] < s["hi"] and s["lo"] <= d["point"] <= s["hi"] and s["sd"] > 0
        assert d["draws"].tolist() == again[subset]["draws"].tolist()
    assert nr.ratio_draws(x, y, snv, ti, reps=40, seed=4)["SNV"]["draws"].tolist() != got["SNV"]["draws"].tolist()
    assert math.isnan(nr.spread(np.array([math.nan, math.nan]))["lo"])


def test_pair_overlap():
    p_a = np.array([0.001, 0.002, 0.5, 0.004, 0.5])
    p_b = np.array([0.001, 0.5, 0.003, 0.5, 0.5])
    out = nr.pair_overlap(p_a, p_b, np.array([0.1, 0.2, 0.9, 0.3, 0.4]))
    # Cherimoya calls 0, 1 and 3, ChromBPNet uncorrected 0 and 2; at two calls Cherimoya's are 0 and 1.
    assert (out["both"], out["cherimoya_only"], out["cbp_full_only"]) == (1, 2, 1)
    assert out["share_of_cherimoya"] == pytest.approx(1 / 3) and out["share_of_cbp_full"] == 0.5
    assert out["equal_count_shared"] == 0.5 and out["equal_count_p"] == 0.002
    # Tied p-values: the larger |fold-mean logFC| first.
    tied = nr.pair_overlap(np.full(3, 0.001), np.array([0.5, 0.001, 0.5]), np.array([0.1, 0.3, 0.2]))
    assert tied["equal_count_shared"] == 1.0


def test_read_columns_prefers_the_npz(tmp_path):
    path = tmp_path / "t.tsv.gz"
    sc.write_table(path, {"variant_id": np.array(["a", "b"]), "logfc": np.array([0.5, -0.25])})
    np.savez(tmp_path / "t.npz", variant_id=np.array(["a", "b"]), logfc=np.array([1.5, 2.5], dtype=np.float32))
    got = nr.read_columns(path, ("logfc",))
    assert list(got) == ["logfc"] and got["logfc"].tolist() == [1.5, 2.5]
    (tmp_path / "t.npz").unlink()
    assert nr.read_columns(path, ("logfc",), ("src_id",))["logfc"].tolist() == [0.5, -0.25]
    with pytest.raises(ValueError, match="no column"):
        nr.read_columns(path, ("abs_logfc",))
    with pytest.raises(FileNotFoundError):
        nr.read_columns(tmp_path / "missing.tsv.gz", ("logfc",))


# --- the command line, end to end -------------------------------------------------------------------------------


def test_end_to_end(tmp_path, monkeypatch):
    # p < 0.5 for a call, so that the 6 SNV rows of the shuffled null can make one; the toy's own p-values are 0.001
    # and 0.5, called and not called at either threshold.
    monkeypatch.setattr(nr, "ALPHA", 0.5)
    write_root(tmp_path)
    assert run(tmp_path) == 0
    out = tmp_path / "analysis"
    assert all((out / name).exists() for name in (*nr.OUTPUTS.values(), "report.md"))
    assert not list(out.glob("*.partial"))

    a = table(tmp_path, "A_widths.tsv")
    row = pick(a, null="null", subset="SNV", model="cherimoya")
    q99 = np.mean([np.quantile(abs_logfc(2.0, 8, k)[SHUFFLED_SNV_ROWS], 0.99) for k in FOLDS])
    assert int(row["n"]) == 6 and close(row["q99"], q99)
    assert close(row["q99_vs_cbp_full"], 2.0)
    assert [int(pick(a, null="null", subset=s, model="cbp_bias")["n"]) for s in ("all", "SNV_Ti", "SNV_Tv", "indel")] \
        == [8, 4, 2, 2]  # fmt: skip
    assert [int(pick(a, null="genomic_null", subset=s, model="cbp_full")["n"]) for s in ("all", "SNV", "SNV_Ti")] \
        == [6, 6, 3]  # fmt: skip
    assert close(pick(a, null="genomic_null", subset="SNV", model="cherimoya")["q99_vs_cbp_full"], 1.0)
    assert int(pick(a, null="variants", subset="summit100_SNV", model="cherimoya")["n"]) == 6

    b = table(tmp_path, "B_calls.tsv")

    def calls(null, view, model, measure):
        return float(pick(b, null=null, view=view, model=model, measure=measure)["value"])

    assert calls("null", "all", "cherimoya", "fold_mean_calls") == 3
    assert calls("null", "all", "cherimoya", "per_fold_share") == pytest.approx(3 / 12)
    assert calls("null", "SNV", "cbp_full", "fold_mean_calls") == 4
    assert calls("genomic_null", "all", "cherimoya", "fold_mean_calls") == 4
    assert calls("null", "all", "cherimoya+cbp_full", "both") == 2
    assert calls("null", "all", "cherimoya+cbp_full", "share_of_cherimoya") == pytest.approx(2 / 3, rel=1e-5)
    assert calls("null", "all", "cherimoya+cbp_full", "equal_count_shared") == 0.5  # 6, 1, 0, then 11 (p 0.5)
    assert calls("null+genomic_null", "all", "cherimoya", "both") == 3
    # The SNVs against the shuffled null's 6 SNV rows: p = (rank + 1) / 7 per fold, 3 / 7 at best (variants 9, 11).
    obs = [abs_logfc(1.0, 12, k)[SNVS] for k in FOLDS]
    bg = [abs_logfc(2.0, 8, k)[SHUFFLED_SNV_ROWS] for k in FOLDS]
    p = sc.geo_mean(np.stack([sc.get_pvals(o, g, "right") for o, g in zip(obs, bg)]))
    assert calls("null", "SNV_vs_SNV_rows", "cherimoya", "fold_mean_calls") == int((p < 0.5).sum()) == 2
    assert "SNV_vs_SNV_rows" not in {r["view"] for r in b if r["null"] == "genomic_null"}

    c = pick(table(tmp_path, "C_fixed_effect_p.tsv"), null="null", subset="all", model="cbp_full")
    assert all(close(c[f"p_{x:g}"], 1 / 17) for x in nr.FIXED)  # all 16 rows lie below 0.25

    d = table(tmp_path, "D_cpg.tsv")
    assert [int(pick(d, null="null", model="cherimoya", cpg_bin=name)["rows"]) for name in nr.cpg_bin_names()] \
        == [2, 1, 1, 1, 0, 1]  # fmt: skip
    assert [int(pick(d, null="genomic_null", model="cherimoya", cpg_bin=name)["rows"]) for name in nr.cpg_bin_names()] \
        == [2, 1, 2, 0, 1, 0]  # fmt: skip
    assert close(pick(d, null="null", model="cherimoya", cpg_bin="all")["cpg_mean"], 250 / 6)
    reweighted = pick(d, null="genomic_null", model="cherimoya", cpg_bin="reweighted")
    # Bin weights 1, 1, 0.5, 0, 0, 0: rows 0-4 count, row 5 (90 CpGs) does not; the weighted q99 is their largest.
    assert close(reweighted["max_weight"], 1.0) and close(reweighted["uncovered_share"], 1 / 3)
    assert close(reweighted["q99"], 6 / 64)

    for name in ("null", "genomic_null", "variants"):
        e = pick(table(tmp_path, "E_agreement.tsv"), set=name, subset="all")
        assert close(e["r_mean"], 1.0) and close(e["top_mean"], 1.0)

    f = table(tmp_path, "F_heldout.tsv")
    row = pick(f, set="null", model="cbp_full")
    # Held out: rows 0-3 (chr1) in fold 0, rows 5 and 6 (chr2) in fold 1.
    q99 = (np.quantile(abs_logfc(1.0, 8, 0)[[0, 1, 2, 3]], 0.99) + np.quantile(abs_logfc(1.0, 8, 1)[[5, 6]], 0.99)) / 2
    assert (int(row["rows_min"]), int(row["rows_max"])) == (2, 4) and close(row["q99_heldout"], q99)
    row = pick(f, set="genomic_null", model="cbp_full")
    assert (int(row["rows_min"]), int(row["rows_max"])) == (2, 3)  # the chr9 row has no fold

    g = table(tmp_path, "G_count_quintiles.tsv")
    for group, n in (("null", 12), ("genomic_null", 12), ("variants", 18)):  # SNV rows x 2 folds
        assert sum(int(r["n"]) for r in g if r["model"] == "cherimoya" and r["group"] == group) == n

    h = table(tmp_path, "H_frequency.tsv")

    def freq(**keys):
        return float(pick(h, **keys)["value"])

    effect = {"part": "effect", "model": "cherimoya", "group": "", "stratum": "all"}
    assert [freq(**effect, null="", mut_class="all", maf_bin=b, measure="n") for b in nr.MAF_BINS] == [3, 4, 4, 1]
    assert freq(**effect, null="", mut_class="CpG_ti", maf_bin="0.00", measure="n") == 1
    assert freq(**effect, null="null", mut_class="all", maf_bin="all", measure="share_called") == 3 / 12
    pair = {"part": "calls", "null": "null", "model": "cherimoya+cbp_full", "mut_class": "all", "stratum": "all"}
    assert freq(**pair, group="both", maf_bin="all", measure="n") == 2  # variants 0 and 6
    assert freq(**pair, group="both", maf_bin="all", measure="mean_maf") == 0.25
    assert freq(**pair, group="cbp_full_only", maf_bin="all", measure="mean_maf") == pytest.approx(0.05)  # 7, 9

    r = table(tmp_path, "R_rule_bootstrap.tsv")
    # Cherimoya's |logFC| is exactly twice the uncorrected model's on the shuffled null and equal on the genomic one,
    # so every replicate gives the point ratio; the variants' ratio is 1, so the genomic null removes all of the excess.
    for null, point in (("null", 2.0), ("genomic_null", 1.0)):
        row = pick(r, measure="q99_ratio", null=null, subset="SNV")
        assert close(row["point"], point) and close(row["lo"], point) and close(row["hi"], point)
        assert int(row["reps"]) == nr.BOOTSTRAP
    assert close(pick(r, measure="q99_ratio", null="variants", subset="summit100_SNV")["point"], 1.0)
    removed = pick(r, measure="excess_removed", null="null+genomic_null")
    assert close(removed["point"], 1.0) and close(removed["lo"], 1.0) and close(removed["hi"], 1.0)
    shares = {b: float(pick(r, measure="branch_share", subset=b)["point"]) for b in nr.BRANCHES}
    assert shares == {"null artefact": 1.0, "partly both": 0.0, "model property": 0.0}

    report = (out / "report.md").read_text()
    assert report.index("## Decision rule") < report.index("## A.")
    assert "**null artefact** (bootstrap: 100 % null artefact)" in report and "Bootstrap intervals" in report
    assert "Purifying selection" in report and "CpG transitions" in report
    assert "1 of 6 rows lie on chromosomes without a fold" in report
    # The rule as written (its ChromBPNet-full), then which label that is; the glossary names all four models read.
    assert all(f"> {text}" in report for text in nr.RULE)
    assert "> Decision rule, written before the genomic null's numbers were seen (2026-10-13 15:37 ET)." in report
    assert "Cherimoya / ChromBPNet-full q99 ratio on like-for-like rows" in report
    assert "In this report's labels, the rule's ChromBPNet full is ChromBPNet uncorrected." in report
    assert "- ChromBPNet corrected: chrombpnet_nobias.h5" in report and "ChromBPNet bias model: the Tn5" in report
    assert "per-fold share Cherimoya / uncorrected / corrected" in report and report.count("ChromBPNet full") == 1


def test_two_datasets(tmp_path):
    write_root(tmp_path, ("dsA", "dsB"))
    assert run(tmp_path, ("dsA", "dsB")) == 0
    a = table(tmp_path, "A_widths.tsv")
    assert pick(a, dataset="dsA", null="null", subset="SNV", model="cherimoya")["q99"] == \
        pick(a, dataset="dsB", null="null", subset="SNV", model="cherimoya")["q99"]  # fmt: skip
    report = (tmp_path / "analysis" / "report.md").read_text()
    assert "2 of 2 datasets." in report and "null artefact in 2 of 2 datasets" in report


def test_a_missing_null_is_left_out_with_a_note(tmp_path):
    write_root(tmp_path)
    shutil.rmtree(tmp_path / "scores_genomic_null")
    assert run(tmp_path) == 0
    report = (tmp_path / "analysis" / "report.md").read_text()
    assert "scores_genomic_null; left out" in report and "**not evaluable**" in report
    assert {r["null"] for r in table(tmp_path, "A_widths.tsv")} == {"null", "variants"}


def test_missing_tables_and_nulls_fail(tmp_path):
    write_root(tmp_path)
    drop_table(tmp_path / "scores" / "dsA" / "fold_1" / "cbp_nobias.variants.tsv.gz")
    assert run(tmp_path) == 1
    a = table(tmp_path, "A_widths.tsv")  # the others are still read
    assert pick(a, null="null", subset="all", model="cherimoya")
    assert pick(a, null="genomic_null", subset="all", model="cbp_nobias")
    assert not [r for r in a if r["null"] == "null" and r["model"] == "cbp_nobias"]
    shutil.rmtree(tmp_path / "scores")
    shutil.rmtree(tmp_path / "scores_genomic_null")
    assert run(tmp_path) == 1
