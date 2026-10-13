"""table_list.py: selection by summit distance, strata, every drop reason, compact ids, mut_class and pass-through.

The tests run the builder on a small genome written to tmp_path (a FASTA, partly soft-masked, and its .fai, two
datasets' narrowPeaks, a manifest and a variant table). Every toy row carries the outcome it should have (kept with
its stratum and mut_class, dropped with its reason, or not selected), and the end-to-end test checks the list, the
dropped rows and the meta's counts against those, with distances recomputed by brute force from the toy's summits.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflows" / "variant_scoring"))

import gnomad_list as gl  # noqa: E402
import table_list as tl  # noqa: E402

LINE = 60
LENGTH = 12_000
FLANK = 1057
HEADER = ("chr", "pos", "ref", "alt", "most_severe_consequence", "AF_nfe", "AF_afr")
LIST_HEADER = ["variant_id", "chrom", "pos", "ref", "alt", "source", "stratum", "af", "dist_to_summit", "fold_test",
               "mut_class", "most_severe_consequence", "AF_nfe", "AF_afr"]  # fmt: skip
SUMMITS = {"ds_a": [("chr1", 3000), ("chr1", 5200)],
           "ds_b": [("chr1", 10_900), ("chr1", 11_500), ("chr2", 3000), ("chr3", 3000), ("chr4", 3300)]}  # fmt: skip
FOLDS = {"chr1": "0", "chr3": "1", "chr4": "2"}  # chr2 has no test fold
CHROMS = "chr1,chr2,chr4"  # chr3 is not among them; chr5 has no summit


# --- toy inputs ------------------------------------------------------------------------------------------------


def write_fasta(path: Path, seqs: dict[str, str]) -> Path:
    """A FASTA with LINE-wide lines and its .fai, as samtools faidx writes it."""
    fai, offset = [], 0
    with open(path, "w") as fh:
        for name, seq in seqs.items():
            header = f">{name}\n"
            fh.write(header)
            offset += len(header)
            fai.append(f"{name}\t{len(seq)}\t{offset}\t{LINE}\t{LINE + 1}\n")
            for i in range(0, len(seq), LINE):
                chunk = seq[i : i + LINE] + "\n"
                fh.write(chunk)
                offset += len(chunk)
    Path(f"{path}.fai").write_text("".join(fai))
    return path


def random_seq(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    return [rng.choice("ACGT") for _ in range(n)]


def other(base: str, choices: str = "ACGT") -> str:
    return next(b for b in choices if b != base)


def write_table(path: Path, rows: list[dict], header=HEADER) -> Path:
    lines = ["\t".join(header) + "\n"]
    for r in rows:
        lines.append("\t".join([r["chrom"], str(r["pos0"] + 1), r["ref"], r["alt"], r["csq"], r["af_nfe"],
                                r["af_afr"]]) + "\n")  # fmt: skip
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt") as fh:
        fh.writelines(lines)
    return path


def toy_rows(seqs: dict[str, list[str]]) -> list[dict]:
    """The table's rows in its order, each with its expected outcome: kept (with mut_class), a reason, or far."""
    s = seqs["chr1"]
    rows = []

    def add(pos0, ref, alt, expect, mc=None, chrom="chr1", af_nfe="0.05", af_afr="0.10", csq="intron_variant"):
        rows.append({"chrom": chrom, "pos0": pos0, "ref": ref, "alt": alt, "expect": expect, "mut_class": mc,
                     "af_nfe": af_nfe, "af_afr": af_afr, "csq": csq})  # fmt: skip

    def snv(pos0, expect="kept", chrom="chr1", **kw):
        ref = seqs[chrom][pos0]
        alt = other(ref)
        add(pos0, ref, alt, expect, tl.mut_class(ref, alt, seqs[chrom][pos0 - 1], seqs[chrom][pos0 + 1]), chrom, **kw)

    snv(2499, "far")  # 501 bp from the summit at 3000
    snv(2500)  # 500 bp: selected, flank500
    snv(2899)  # 101 bp: flank500
    snv(2900)  # 100 bp: summit100
    snv(3000, af_nfe="NaN", csq="regulatory region variant")  # af empty, AF_nfe passed through as NaN
    add(3010, "C", "T", "kept", "CpG_ti", af_nfe="1.00")  # the C of the CpG at 3010-3011
    add(3010, "C", "A", "kept", "CpG_tv")
    add(3011, "G", "A", "kept", "CpG_ti")  # the G of that CpG
    add(3011, "G", "T", "kept", "CpG_tv")
    add(3020, "C", "T", "kept", "ti")  # C followed by A
    add(3020, "C", "G", "kept", "tv")
    add(3030, "G", "A", "kept", "ti")  # G preceded by A
    add(3030, "G", "C", "kept", "tv")
    add(3040, "G", "A", "kept", "ti")  # the G of a GpC is no CpG
    add(3041, "C", "T", "kept", "ti")  # nor its C
    add(3050, "A", "G", "kept", "ti")
    add(3050, "A", "C", "kept", "tv")
    add(3050, "A", "T", "kept", "tv")  # A<>T and T<>A are transversions
    add(3060, "T", "C", "kept", "ti")
    add(3060, "T", "G", "kept", "tv")
    add(3060, "T", "A", "kept", "tv")
    mnv_ref = "".join(s[3070:3072])
    add(3070, mnv_ref, "".join(other(b) for b in mnv_ref), "kept", "mnv")  # MNVs are kept
    add(3079, s[3079], s[3079] + "ACGTACGTACGTACGTACGTACGTACG", "kept", "indel")  # id of exactly 40 characters
    add(3079, s[3079], s[3079] + "ACGTACGTACGTACGTACGTACGTACGT", "kept", "indel")  # 41: compacted
    add(3079, s[3079], s[3079] + "TTTTACGTACGTACGTACGTACGTACGT", "kept", "indel")  # same lengths, other hash
    add(3089, "".join(s[3089:3149]), s[3089], "kept", "indel")  # a 59-bp deletion: compacted
    add(3095, "".join(s[3095:3098]), s[3095], "kept", "indel")
    snv(3100)  # 100 bp: summit100
    rows.append(dict(rows[-1], expect="duplicate", mut_class=None))
    snv(3101)  # 101 bp: flank500
    add(3110, s[3110], "A,C", "multiallelic")
    add(3120, s[3120], "N", "non_acgt_allele")
    add(3121, s[3121], "*", "non_acgt_allele")
    add(3130, other(s[3130]), s[3130], "ref_mismatch")
    add(3140, "".join(s[3140 : 3140 + FLANK + 1]), s[3140], "window_off_chrom")  # REF longer than the half window
    snv(3500, csq=" upstream gene variant ")  # 500 bp: flank500; the spaces of a passed-through field are kept
    snv(3501, "far")
    snv(5200, "window_non_acgt")  # its window holds the N run at 6000-6010
    last = LENGTH - FLANK - 1  # pos0 of the last pos VS's bound allows: pos + 1057 <= length
    snv(last)
    add(last, "".join(s[last : last + 2]), s[last], "kept", "indel")  # a 1-bp deletion reads 1 more base: fits
    add(last, "".join(s[last : last + 3]), s[last], "window_off_chrom")  # 2 more bases: past the end
    snv(last + 1, "window_off_chrom")
    snv(11_400, "window_off_chrom")
    snv(3000, "no_test_fold", chrom="chr2")
    snv(3000, "other_chrom", chrom="chr3")
    add(3050, "A", "G", "kept", "ti", chrom="chr4")  # chr4's test fold is 2; chr1 has a row at this pos, REF and ALT
    snv(3000, "far", chrom="chr5")  # no summit on chr5
    return rows


@pytest.fixture
def toy(tmp_path):
    seqs = {c: random_seq(LENGTH, i) for i, c in enumerate(("chr1", "chr2", "chr3", "chr4", "chr5"), start=1)}
    s = seqs["chr1"]
    s[6000:6010] = "N" * 10
    for pos0, base in {3010: "C", 3011: "G", 3019: "T", 3020: "C", 3021: "A", 3029: "A", 3030: "G", 3039: "T",
                       3040: "G", 3041: "C", 3042: "A", 3050: "A", 3060: "T"}.items():  # fmt: skip
        s[pos0] = base
    seqs["chr4"][3050] = "A"
    text = {c: "".join(v) for c, v in seqs.items()}
    # Soft-masked (lower case) around the CpG at 3010-3011: the FASTA is read upper-cased, so REF, the window and the
    # CpG context see upper case.
    text["chr1"] = text["chr1"][:3005] + text["chr1"][3005:3015].lower() + text["chr1"][3015:]
    fasta = write_fasta(tmp_path / "g.fa", text)
    peaks = {}
    for ds, summits in SUMMITS.items():
        peaks[ds] = tmp_path / f"{ds}.narrowPeak"
        peaks[ds].write_text("".join(f"{c}\t{x - 250}\t{x + 250}\tp{i}\t0\t.\t5\t10\t8\t250\n"
                                     for i, (c, x) in enumerate(summits)))  # fmt: skip
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"genome": {"path": str(fasta), "md5": "toy"}, "ok": True, "problems": [],
                                    "datasets": {ds: {"peaks": {"path": str(p)}} for ds, p in peaks.items()},
                                    "test_chrom_to_fold": FOLDS}))  # fmt: skip
    rows = toy_rows(seqs)
    return {"seqs": seqs, "fasta": fasta, "manifest": manifest, "rows": rows,
            "table": write_table(tmp_path / "table.tsv.gz", rows)}  # fmt: skip


def run(toy: dict, out: Path, table: Path | None = None, extra: tuple = ()) -> int:
    return tl.main(["--table", str(table or toy["table"]), "--manifest", str(toy["manifest"]), "--chroms", CHROMS,
                    "--inner", "100", "--outer", "500", "--source", "toy_common", "--af-column", "AF_nfe",
                    "--out", str(out), *extra])  # fmt: skip


def read_tsv(path: Path) -> tuple[list[str], list[dict]]:
    lines = path.read_text().splitlines()
    header = lines[0].split("\t")
    return header, [dict(zip(header, x.split("\t"))) for x in lines[1:]]


def distance(chrom: str, pos0: int) -> int | None:
    xs = [x for summits in SUMMITS.values() for c, x in summits if c == chrom]
    return min(abs(pos0 - x) for x in xs) if xs else None


# --- units -----------------------------------------------------------------------------------------------------


def test_variant_id_compacts_past_40_characters():
    ref, alt = "A", "A" + "C" * 27
    vid, compacted = tl.variant_id("chr1", 3080, ref, alt)
    assert vid == f"chr1:3080:{ref}:{alt}" and len(vid) == 40 and not compacted  # exactly 40: kept as is
    alt = "A" + "C" * 28
    vid, compacted = tl.variant_id("chr1", 3080, ref, alt)
    digest = hashlib.md5(f"{ref}>{alt}".encode()).hexdigest()[:10]
    assert compacted and vid == f"chr1:3080:1>29:{digest}" and len(vid) <= 40  # 41: compacted
    assert tl.variant_id("chr1", 3080, ref, alt) == (vid, True)  # deterministic
    assert tl.variant_id("chr1", 3080, ref, "A" + "G" * 28)[0] != vid  # same lengths, other alleles: other id
    long_ref = "ACGT" * 200
    vid, compacted = tl.variant_id("chr22", 250_000_000, long_ref, "A")
    assert compacted and vid.startswith("chr22:250000000:800>1:") and len(vid) <= 40


def test_mut_class_cpg_both_bases_and_off_cpg():
    assert tl.mut_class("C", "T", "A", "G") == "CpG_ti"  # the C of a CpG
    assert tl.mut_class("C", "A", "A", "G") == "CpG_tv"
    assert tl.mut_class("G", "A", "C", "T") == "CpG_ti"  # the G of a CpG
    assert tl.mut_class("G", "T", "C", "T") == "CpG_tv"
    assert tl.mut_class("C", "T", "G", "A") == "ti"  # the C of a GpC
    assert tl.mut_class("G", "A", "T", "C") == "ti"  # the G of a GpC
    assert tl.mut_class("A", "G", "C", "G") == "ti" and tl.mut_class("T", "C", "C", "G") == "ti"
    assert tl.mut_class("A", "C", "C", "G") == "tv" and tl.mut_class("T", "G", "C", "G") == "tv"
    assert tl.mut_class("AC", "GT") == "mnv"
    assert tl.mut_class("AC", "A") == "indel" and tl.mut_class("A", "AC") == "indel"
    assert tl.mut_class("A", "T", "C", "G") == "tv" and tl.mut_class("T", "A", "C", "G") == "tv"  # not transitions


def test_af_text_keeps_numbers_as_written():
    assert [tl.af_text(x) for x in ("0.05", "1.00", "0", "1e-3")] == ["0.05", "1.00", "0", "1e-3"]
    assert [tl.af_text(x) for x in ("NaN", "nan", "NA", "", ".", "inf")] == [""] * 6


def test_checker_keeps_mnvs_only_when_asked(toy):
    s = toy["seqs"]["chr1"]
    ref = "".join(s[3070:3072])
    alt = "".join(other(b) for b in ref)
    assert gl.Checker(gl.Fasta(toy["fasta"]), ["chr1"], 2114).reason("chr1", 3071, ref, alt) == "mnv"
    assert gl.Checker(gl.Fasta(toy["fasta"]), ["chr1"], 2114, keep_mnv=True).reason("chr1", 3071, ref, alt) is None
    assert gl.Checker(tl.ChromFasta(toy["fasta"]), ["chr1"], 2114, keep_mnv=True).reason("chr1", 3071, alt, ref) == (
        "ref_mismatch"
    )


def test_chrom_fasta_matches_fasta(toy):
    fa, cached = gl.Fasta(toy["fasta"]), tl.ChromFasta(toy["fasta"])
    for chrom, start, end in [("chr1", 0, 1), ("chr1", 59, 61), ("chr1", 5990, 6020), ("chr2", 11_990, 12_000),
                              ("chr1", 100, 200), ("chr1", 3000, 3020)]:  # fmt: skip
        assert cached.fetch(chrom, start, end) == fa.fetch(chrom, start, end)
    assert cached.loads == 3  # chr1, chr2, chr1 again
    assert cached.fetch("chr1", 3000, 3020).isupper()  # its soft-masked bases, upper-cased


# --- end to end ------------------------------------------------------------------------------------------------


def test_list_from_table_end_to_end(toy, tmp_path):
    out = tmp_path / "lists" / "toy.tsv"
    assert run(toy, out) == 0
    header, rows = read_tsv(out)
    assert header == LIST_HEADER
    expected = []
    for line, r in enumerate(toy["rows"], start=2):
        r["line"] = line
        if r["expect"] == "kept":
            expected.append(r)
    assert len(rows) == len(expected)
    for got, want in zip(rows, expected):
        d = distance(want["chrom"], want["pos0"])
        vid, _ = tl.variant_id(want["chrom"], want["pos0"] + 1, want["ref"], want["alt"])
        assert got == {
            "variant_id": vid,
            "chrom": want["chrom"],
            "pos": str(want["pos0"] + 1),
            "ref": want["ref"],
            "alt": want["alt"],
            "source": "toy_common",
            "stratum": "summit100" if d <= 100 else "flank500",
            "af": "" if want["af_nfe"] == "NaN" else want["af_nfe"],
            "dist_to_summit": str(d),
            "fold_test": FOLDS[want["chrom"]],
            "mut_class": want["mut_class"],
            "most_severe_consequence": want["csq"],
            "AF_nfe": want["af_nfe"],
            "AF_afr": want["af_afr"],
        }
    by_pos = {int(r["pos"]) - 1: r for r in rows if r["alt"] == other(r["ref"])}
    # the distance and stratum edges
    assert by_pos[2500]["dist_to_summit"] == "500" and by_pos[3500]["dist_to_summit"] == "500"
    assert all(int(r["pos"]) - 1 not in (2499, 3501) for r in rows)
    assert by_pos[2900]["stratum"] == "summit100" and by_pos[2899]["stratum"] == "flank500"
    assert by_pos[3100]["stratum"] == "summit100" and by_pos[3101]["stratum"] == "flank500"
    # NaN gives an empty af; the pass-through columns keep their text
    nan_row = by_pos[3000]
    assert nan_row["af"] == "" and nan_row["AF_nfe"] == "NaN" and nan_row["AF_afr"] == "0.10"
    assert nan_row["most_severe_consequence"] == "regulatory region variant"
    # compact ids: at most 40 characters, unique, the 40-character one kept as is
    ids = [r["variant_id"] for r in rows]
    assert len(set(ids)) == len(ids) and max(map(len, ids)) <= 40
    s = toy["seqs"]["chr1"]
    assert f"chr1:3080:{s[3079]}:{s[3079]}ACGTACGTACGTACGTACGTACGTACG" in ids
    compact = [v for v in ids if ">" in v]
    assert len(compact) == 3 and sorted(v.split(":")[2] for v in compact) == ["1>29", "1>29", "60>1"]
    # the mut_class of every SNV is that of its reference context
    for r in rows:
        if len(r["ref"]) == len(r["alt"]) == 1:
            p, seq = int(r["pos"]) - 1, toy["seqs"][r["chrom"]]
            assert r["mut_class"] == tl.mut_class(r["ref"], r["alt"], seq[p - 1], seq[p + 1])
    # a second test chromosome; the same pos, REF and ALT kept on two chromosomes; passed-through text as it is
    chr4 = [r for r in rows if r["chrom"] == "chr4"]
    assert [(r["variant_id"], r["fold_test"], r["stratum"]) for r in chr4] == [("chr4:3051:A:G", "2", "flank500")]
    assert "chr1:3051:A:G" in [r["variant_id"] for r in rows]
    assert by_pos[3500]["most_severe_consequence"] == " upstream gene variant "

    dropped_header, dropped = read_tsv(out.with_name("toy.dropped.tsv"))
    assert dropped_header == ["reason", "detail", "line", "variant_id", *HEADER]
    want_drops = [r for r in toy["rows"] if r["expect"] not in ("kept", "far")]
    assert [(d["reason"], d["line"], d["chr"], d["pos"], d["ref"], d["alt"]) for d in dropped] == [
        (r["expect"], str(r["line"]), r["chrom"], str(r["pos0"] + 1), r["ref"], r["alt"]) for r in want_drops
    ]
    assert {d["reason"] for d in dropped} == set(tl.REASONS)  # every reason the builder can give
    duplicate = next(d for d in dropped if d["reason"] == "duplicate")
    first = next(r for r in toy["rows"] if r["pos0"] == 3100 and r["expect"] == "kept")
    assert duplicate["detail"] == f"line {first['line']}"
    mismatch = next(d for d in dropped if d["reason"] == "ref_mismatch")
    assert mismatch["detail"] == f"fasta {s[3130]}"

    meta = json.loads(out.with_name("toy.meta.json").read_text())
    c = meta["counts"]
    far = sum(r["expect"] == "far" for r in toy["rows"])
    assert c["rows"] == len(toy["rows"]) and c["selected"] == len(toy["rows"]) - far
    assert c["rows_on_chroms_with_summits"] == sum(r["chrom"] != "chr5" for r in toy["rows"])
    assert c["kept"] == len(rows) and c["selected"] == c["kept"] + len(dropped)
    assert c["dropped"] == {reason: sum(r["expect"] == reason for r in toy["rows"]) for reason in tl.REASONS}
    assert c["funnel"][-1]["left"] == c["kept"]
    assert c["by_stratum"] == {s: sum(r["stratum"] == s for r in rows) for s in ("summit100", "flank500")}
    assert c["by_mut_class"] == {m: sum(r["mut_class"] == m for r in rows) for m in tl.MUT_CLASSES}
    assert c["by_fold_test"] == {"0": len(rows) - 1, "2": 1}
    assert c["by_chrom"]["chr4"] == {"rows": 1, "selected": 1, "kept": 1, "dropped": 0}
    assert c["by_chrom"]["chr5"] == {"rows": 1, "selected": 0, "kept": 0, "dropped": 0}
    assert c["af_empty"] == 1
    longest = f"chr1:3090:{''.join(s[3089:3149])}:{s[3089]}"  # the 59-bp deletion's chrom:pos:ref:alt
    assert meta["ids"]["compacted"] == 3 and meta["ids"]["longest_plain"] == len(longest) == 72
    assert meta["ids"]["longest_written"] <= 40
    assert meta["list_columns"] == LIST_HEADER
    assert meta["inputs"]["fasta"]["md5_in_manifest"] == "toy"
    assert [p["dataset"] for p in meta["inputs"]["peaks"]] == ["ds_a", "ds_b"]
    assert meta["chromosome_loads"] == 3  # chr1, chr2 for the row the fold check drops, chr4; never chr3 or chr5
    assert meta["table_order"]["position_sorted_within_chrom"] and meta["table_order"]["chromosomes_in_fasta_order"]
    assert not list(out.parent.glob("*.tmp"))


def test_list_is_deterministic_and_reads_plain_tsv(toy, tmp_path):
    a, b = tmp_path / "a" / "x.tsv", tmp_path / "b" / "x.tsv"
    plain = write_table(tmp_path / "table.tsv", toy["rows"])
    assert run(toy, a) == 0 and run(toy, b, table=plain) == 0
    assert a.read_text() == b.read_text()
    assert a.with_name("x.dropped.tsv").read_text() == b.with_name("x.dropped.tsv").read_text()


def test_chrom_header_and_bad_tables(toy, tmp_path):
    rows = [r for r in toy["rows"] if r["chrom"] == "chr1"][:5]
    chrom_header = write_table(tmp_path / "chrom.tsv", rows, header=("chrom", *HEADER[1:]))
    assert run(toy, tmp_path / "ok.tsv", table=chrom_header) == 0
    assert read_tsv(tmp_path / "ok.tsv")[0] == LIST_HEADER

    regrouped = [toy["rows"][1], next(r for r in toy["rows"] if r["chrom"] == "chr2"), toy["rows"][2]]
    out = tmp_path / "bad" / "regrouped.tsv"
    assert run(toy, out, table=write_table(tmp_path / "regrouped.tsv", regrouped)) == 1
    assert not out.parent.exists() or not list(out.parent.iterdir())  # no list, no dropped rows, no temporary files

    no_alt = tmp_path / "no_alt.tsv"
    no_alt.write_text("chr\tpos\tref\tAF_nfe\nchr1\t3001\tA\t0.1\n")
    assert run(toy, tmp_path / "x.tsv", table=no_alt) == 1
    clash = tmp_path / "clash.tsv"
    clash.write_text("chr\tpos\tref\talt\tAF_nfe\tsource\n")
    assert run(toy, tmp_path / "x.tsv", table=clash) == 1
    assert run(toy, tmp_path / "x.tsv", table=chrom_header, extra=("--af-column", "AF_xyz")) == 1
    bad_pos = tmp_path / "bad_pos.tsv"
    bad_pos.write_text("chr\tpos\tref\talt\tAF_nfe\nchr1\t3001.5\tA\tC\t0.1\n")
    assert run(toy, tmp_path / "x.tsv", table=bad_pos) == 1
