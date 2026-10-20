"""genomic_null.py: exclusions, the bound, REF and ALT, GC matching and borrowing, determinism and the list format.

The tests run the generator on a small synthetic genome written to tmp_path (a FASTA and its .fai, narrowPeaks, a
gzipped blacklist, a target list and a manifest) and check the output against the toy by brute force; two call its bin,
largest-remainder and offset functions directly. Every run but the last test's uses a 100-bp window (--flank 50), so a
window's GC count is its own 1% bin and a few kilobases hold many tiles. GC is controlled exactly with W-periodic
chromosomes: every 100-bp window of a sequence that repeats a 100-bp unit holds each base of the unit once, at any
offset. The last test builds build_windows.py's windows from a list drawn with the default 2114-bp window.
"""

from __future__ import annotations

import gzip
import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflows" / "variant_scoring"))

import build_windows as bw  # noqa: E402
import genomic_null as gn  # noqa: E402

HEADER = "variant_id\tchrom\tpos\tref\talt\tsource\tstratum\taf\tdist_to_summit\tfold_test"
FLANK = 50
W = 2 * FLANK
SEED = 1234


# --- toy inputs ------------------------------------------------------------------------------------------------


def write_fasta(path: Path, seqs: dict[str, str], width: int = 60) -> Path:
    """A FASTA and the .fai samtools faidx writes for it."""
    fai, offset = [], 0
    with open(path, "w") as fh:
        for name, seq in seqs.items():
            header = f">{name} toy\n"
            fh.write(header)
            offset += len(header)
            fai.append(f"{name}\t{len(seq)}\t{offset}\t{width}\t{width + 1}\n")
            for i in range(0, len(seq), width):
                line = seq[i : i + width] + "\n"
                fh.write(line)
                offset += len(line)
    Path(f"{path}.fai").write_text("".join(fai))
    return path


def random_seq(n: int, seed: int, gc: float = 0.5) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice("GC") if rng.random() < gc else rng.choice("AT") for _ in range(n))


def periodic(gc: int, length: int, seed: int) -> str:
    """length bases repeating one W-bp unit with exactly gc G or C bases: every W-bp window holds gc of them."""
    rng = random.Random(seed)
    unit = [rng.choice("GC") for _ in range(gc)] + [rng.choice("AT") for _ in range(W - gc)]
    rng.shuffle(unit)
    return ("".join(unit) * (length // W + 1))[:length]


def write_peaks(path: Path, peaks: list[tuple[str, int, int]]) -> Path:
    """narrowPeak rows with the summit (column 10) at the middle."""
    path.write_text("".join(f"{c}\t{s}\t{e}\tpeak_{i}\t0\t.\t5.0\t10.0\t8.0\t{(e - s) // 2}\n"
                            for i, (c, s, e) in enumerate(peaks)))  # fmt: skip
    return path


def write_bed_gz(path: Path, rows: list[tuple[str, int, int]]) -> Path:
    with gzip.open(path, "wt") as fh:
        fh.writelines(f"{c}\t{s}\t{e}\n" for c, s, e in rows)
    return path


def write_targets(path: Path, rows: list[tuple[str, int, str]]) -> Path:
    """A variant list with the target rows (chrom, 1-based pos, stratum); the alleles are placeholders."""
    lines = [f"{c}:{p}:A:C:{i}\t{c}\t{p}\tA\tC\tgenome\t{s}\t0.001\t0\t0\n" for i, (c, p, s) in enumerate(rows)]
    path.write_text(HEADER + "\n" + "".join(lines))
    return path


def write_manifest(path: Path, folds: dict[str, str]) -> Path:
    path.write_text(json.dumps({"test_chrom_to_fold": folds}))
    return path


def summit_lists(peaks: list[tuple[str, int, int]]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for c, s, e in peaks:
        out.setdefault(c, []).append(s + (e - s) // 2)
    return out


def run(inputs: dict, out: Path, n: int, seed: int = SEED, flank: int = FLANK) -> tuple[list[dict], dict]:
    """genomic_null.main on the toy; the rows (dicts keyed by the header) and the meta."""
    argv = ["--fasta", str(inputs["fasta"]), "--peaks", *map(str, inputs["peaks"]), "--blacklist",
            str(inputs["blacklist"]), "--target-list", str(inputs["targets"]), "--manifest", str(inputs["manifest"]),
            "--chroms", ",".join(inputs["chroms"]), "--n", str(n), "--seed", str(seed), "--flank", str(flank),
            "--out", str(out)]  # fmt: skip
    assert gn.main(argv) == 0
    lines = out.read_text().splitlines()
    rows = [dict(zip(lines[0].split("\t"), line.split("\t"))) for line in lines[1:]]
    assert all(len(line.split("\t")) == 10 for line in lines)
    return rows, json.loads(out.with_name(out.name[: -len(".tsv")] + ".meta.json").read_text())


def expected_tiles(seq: str, offset: int, peaks, blacklist, flank: int = FLANK) -> tuple[dict, list[int]]:
    """Tile counts after each filter, and the eligible tile starts, by brute force."""
    w = 2 * flank
    counts = dict.fromkeys(("all", "acgt", "not_in_peak", "not_blacklisted", "in_bounds"), 0)
    starts = []
    for t in range(offset, len(seq) - w + 1, w):
        counts["all"] += 1
        if set(seq[t : t + w].upper()) - set("ACGT"):
            continue
        counts["acgt"] += 1
        if any(s < t + w and e > t for s, e in peaks):
            continue
        counts["not_in_peak"] += 1
        if any(s < t + w and e > t for s, e in blacklist):
            continue
        counts["not_blacklisted"] += 1
        pos = t + flank + 1
        if not (pos - flank > 0 and pos + flank <= len(seq)):
            continue
        counts["in_bounds"] += 1
        starts.append(t)
    return counts, starts


# --- a mixed toy: N runs, an IUPAC base, soft-masking, peaks in two files, a blacklist ----------------------------


@pytest.fixture
def toy(tmp_path):
    a = list(random_seq(30_000, 1))
    a[4000:4600] = [x.lower() for x in a[4000:4600]]
    a[9000:9150] = ["N"] * 150
    a[17_777] = "R"
    b = random_seq(24_000, 2, gc=0.42)
    b = b[:1000] + b[1000:3000].lower() + b[3000:]
    # chrB is written first, so the .fai order is not sorted order.
    seqs = {"chrB": b, "chrA": "".join(a), "chrM": random_seq(700, 3)}
    peaks1 = [("chrA", 2000, 2200), ("chrA", 12_000, 12_200), ("chrB", 5000, 5200)]
    peaks2 = [("chrA", 12_100, 12_300), ("chrB", 20_000, 20_200), ("chrM", 0, 200)]
    black = [("chrB", 10_000, 10_500), ("chrA", 25_000, 25_001)]
    targets = [("chrA", 2101, "summit100"), ("chrA", 12_151, "summit100"), ("chrB", 5101, "flank500"),
               ("chrB", 20_401, "flank500"), ("chrA", 9051, "summit100"), ("chrA", 27_000, "outside")]  # fmt: skip
    folds = {"chrA": "3", "chrB": "1", "chrM": "4"}
    return {
        "seqs": seqs,
        "fasta": write_fasta(tmp_path / "toy.fa", seqs),
        "peaks": [write_peaks(tmp_path / "p1.narrowPeak", peaks1), write_peaks(tmp_path / "p2.narrowPeak", peaks2)],
        "peak_rows": peaks1 + peaks2,
        "blacklist": write_bed_gz(tmp_path / "blacklist.bed.gz", black),
        "black_rows": black,
        "targets": write_targets(tmp_path / "targets.tsv", targets),
        "manifest": write_manifest(tmp_path / "manifest.json", folds),
        "folds": folds,
        "chroms": ["chrA", "chrB", "chrM"],
    }


@pytest.fixture
def every_tile(toy, tmp_path):
    """The toy drawn with --n equal to its eligible tiles, so every eligible tile is a row."""
    _, meta = run(toy, tmp_path / "probe" / "null.tsv", n=1)
    rows, meta = run(toy, tmp_path / "all" / "null.tsv", n=meta["tiles"]["total"]["in_bounds"])
    return rows, meta


def test_every_eligible_tile_is_drawn_and_no_window_touches_a_peak_blacklist_or_non_acgt_base(toy, every_tile):
    rows, meta = every_tile
    for chrom in toy["chroms"]:
        offset = meta["offsets"][chrom]
        assert offset == gn.tile_offset(SEED, chrom, W) and 0 <= offset < W
        peaks = [(s, e) for c, s, e in toy["peak_rows"] if c == chrom]
        black = [(s, e) for c, s, e in toy["black_rows"] if c == chrom]
        counts, starts = expected_tiles(toy["seqs"][chrom], offset, peaks, black)
        assert meta["tiles"]["per_chrom"][chrom] == counts
        assert sorted(int(r["pos"]) - 1 - FLANK for r in rows if r["chrom"] == chrom) == starts
    assert meta["tiles"]["per_chrom"]["chrA"]["acgt"] < meta["tiles"]["per_chrom"]["chrA"]["all"]  # N and R bite
    assert meta["tiles"]["per_chrom"]["chrB"]["not_blacklisted"] < meta["tiles"]["per_chrom"]["chrB"]["not_in_peak"]
    for r in rows:
        seq, pos0 = toy["seqs"][r["chrom"]], int(r["pos"]) - 1
        a, b = pos0 - FLANK, pos0 + FLANK
        assert set(seq[a:b].upper()) <= set("ACGT")
        assert not any(c == r["chrom"] and s < b and e > a for c, s, e in toy["peak_rows"])
        assert not any(c == r["chrom"] and s < b and e > a for c, s, e in toy["black_rows"])
    # The target row inside the N run is skipped; the 'outside' row is not a target.
    assert meta["target"]["skipped"] == {"window_non_acgt": 1}
    assert meta["target"]["rows_in_strata"] == 5 and meta["target"]["windows"] == 4


def test_rows_carry_the_reference_base_a_different_alt_ids_folds_and_summit_distances(toy, every_tile):
    rows, meta = every_tile
    summits = summit_lists(toy["peak_rows"])
    soft_masked = 0
    for r in rows:
        pos = int(r["pos"])
        base = toy["seqs"][r["chrom"]][pos - 1]
        soft_masked += base.islower()
        assert r["ref"] == base.upper() and r["ref"] in "ACGT"
        assert len(r["alt"]) == 1 and r["alt"] in "ACGT" and r["alt"] != r["ref"]
        assert r["variant_id"] == f"{r['chrom']}:{pos}:{r['ref']}:{r['alt']}"
        assert r["source"] == r["stratum"] == "genomic_null" and r["af"] == ""
        assert r["fold_test"] == toy["folds"][r["chrom"]]
        assert int(r["dist_to_summit"]) == min(abs(pos - 1 - s) for s in summits[r["chrom"]])
    assert soft_masked > 0  # REF is upper-cased where the FASTA is soft-masked
    assert meta["n_rows"] == len(rows) == sum(meta["ref_counts"].values()) == sum(meta["alt_counts"].values())
    assert all(meta["ref_alt_counts"].get(f"{x}>{x}", 0) == 0 for x in "ACGT")


def test_rows_are_in_fai_order_then_pos_with_the_exact_header(toy, tmp_path):
    out = tmp_path / "order" / "null.tsv"
    rows, _ = run(toy, out, n=120)
    assert out.read_text().splitlines()[0] == HEADER == "\t".join(gn.LIST_COLUMNS)
    fai = [line.split("\t")[0] for line in Path(f"{toy['fasta']}.fai").read_text().splitlines()]
    keys = [(fai.index(r["chrom"]), int(r["pos"])) for r in rows]
    assert keys == sorted(keys) and len(set(keys)) == len(keys) == 120


def test_the_bound_drops_a_tile_that_ends_at_the_chromosome_end(tmp_path):
    off = {c: gn.tile_offset(SEED, c, W) for c in ("chrE", "chrF", "chrG")}
    seqs = {
        "chrE": random_seq(off["chrE"] + 5 * W, 11),  # the 5th tile ends at the end: out (pos + flank > length)
        "chrF": random_seq(W, 12),  # at most one tile, ending at the end: no rows
        "chrG": random_seq(off["chrG"] + 3 * W + 1, 13),  # the 3rd tile ends 1 bp before the end: in
    }
    inputs = {
        "fasta": write_fasta(tmp_path / "bound.fa", seqs),
        "peaks": [write_peaks(tmp_path / "p.narrowPeak", [("chrX", 0, 10)])],
        "blacklist": write_bed_gz(tmp_path / "bl.bed.gz", [("chrX", 0, 10)]),
        "targets": write_targets(tmp_path / "t.tsv", [("chrE", 151, "summit100"), ("chrG", 120, "flank500")]),
        "manifest": write_manifest(tmp_path / "m.json", {"chrE": "0", "chrF": "1", "chrG": "2"}),
        "chroms": ["chrE", "chrF", "chrG"],
    }
    rows, meta = run(inputs, tmp_path / "bound" / "null.tsv", n=7)
    per = meta["tiles"]["per_chrom"]
    assert (per["chrE"]["all"], per["chrE"]["not_blacklisted"], per["chrE"]["in_bounds"]) == (5, 5, 4)
    assert per["chrF"]["all"] == (1 if off["chrF"] == 0 else 0) and per["chrF"]["in_bounds"] == 0
    assert (per["chrG"]["all"], per["chrG"]["in_bounds"]) == (3, 3)
    assert meta["counts_by_chrom"] == {"chrE": 4, "chrF": 0, "chrG": 3}
    for r in rows:
        pos, length = int(r["pos"]), len(seqs[r["chrom"]])
        assert pos - FLANK > 0 and pos + FLANK <= length  # VS's get_valid_variants
    assert max(int(r["pos"]) for r in rows if r["chrom"] == "chrG") + FLANK == len(seqs["chrG"])
    assert {int(r["dist_to_summit"]) for r in rows} == {gn.NO_SUMMIT}  # no summit on these chromosomes


def test_the_same_seed_gives_byte_identical_output(toy, tmp_path):
    out1, out2, out3 = (tmp_path / d / "null.tsv" for d in ("a", "b", "c"))
    _, m1 = run(toy, out1, n=150)
    _, m2 = run(toy, out2, n=150)
    assert out1.read_bytes() == out2.read_bytes()
    for key in ("offsets", "tiles", "target", "gc", "bins", "shortfall", "ref_alt_counts"):
        assert m1[key] == m2[key]
    run(toy, out3, n=150, seed=SEED + 1)
    assert out3.read_bytes() != out1.read_bytes()


# --- GC matching on W-periodic chromosomes -------------------------------------------------------------------------


def gc_toy(tmp_path: Path, available: dict[int, int], targets: dict[int, int]) -> dict:
    """Chromosome chrG<g> repeats a unit with g G/C bases and holds exactly available[g] tiles, all in bounds;
    targets[g] summit100 rows lie on it, and 25 'outside' rows on the first chromosome must not count."""
    seqs, rows = {}, []
    for g, k in available.items():
        name = f"chrG{g}"
        seqs[name] = periodic(g, gn.tile_offset(SEED, name, W) + k * W + 1, seed=g)
        span = len(seqs[name]) - W  # window starts that fit
        rows += [(name, FLANK + 1 + (i * 37) % span, "summit100") for i in range(targets.get(g, 0))]
    first = next(iter(available))
    rows += [(f"chrG{first}", FLANK + 1 + i, "outside") for i in range(25)]
    peak = [("chrZ", 0, 100)]
    return {
        "seqs": seqs,
        "fasta": write_fasta(tmp_path / "gc.fa", seqs),
        "peaks": [write_peaks(tmp_path / "gc.narrowPeak", peak)],
        "blacklist": write_bed_gz(tmp_path / "gc.bed.gz", peak),
        "targets": write_targets(tmp_path / "gc_targets.tsv", rows),
        "manifest": write_manifest(tmp_path / "gc_manifest.json", {c: "0" for c in seqs}),
        "chroms": list(seqs),
    }


def gc_of_rows(inputs: dict, rows: list[dict]) -> dict[int, int]:
    """Rows per GC count of their window, from the FASTA's sequence."""
    out: dict[int, int] = {}
    for r in rows:
        pos0 = int(r["pos"]) - 1
        window = inputs["seqs"][r["chrom"]][pos0 - FLANK : pos0 + FLANK]
        g = window.count("G") + window.count("C")
        out[g] = out.get(g, 0) + 1
    return out


def check_bin_invariants(meta: dict, n: int) -> None:
    bins = meta["bins"]
    assert sum(b["target"] for b in bins) == n == meta["n_rows"]
    assert sum(b["taken"] + b["borrowed"] for b in bins) == n == sum(b["achieved"] for b in bins)
    assert sum(b["borrowed"] for b in bins) == sum(b["lent"] for b in bins) == meta["shortfall"]["rows_borrowed"]
    for b in bins:
        assert b["taken"] == min(b["target"], b["available"]) and b["shortfall"] == b["target"] - b["taken"]
        assert b["borrowed"] == b["shortfall"] == sum(b["borrowed_from"].values())
        assert b["achieved"] == b["taken"] + b["lent"] <= b["available"]


def test_gc_matching_reproduces_the_target_histogram(tmp_path):
    inputs = gc_toy(tmp_path, available={30: 40, 40: 40, 50: 40, 60: 40}, targets={30: 10, 40: 20, 50: 30, 60: 40})
    rows, meta = run(inputs, tmp_path / "match" / "null.tsv", n=50)
    by_bin = {b["bin"]: b for b in meta["bins"]}
    assert {k: b["target_windows"] for k, b in by_bin.items()} == {30: 10, 40: 20, 50: 30, 60: 40}
    assert {k: b["target"] for k, b in by_bin.items()} == {30: 5, 40: 10, 50: 15, 60: 20}
    assert {k: b["achieved"] for k, b in by_bin.items()} == {30: 5, 40: 10, 50: 15, 60: 20}
    assert gc_of_rows(inputs, rows) == {30: 5, 40: 10, 50: 15, 60: 20}
    assert meta["shortfall"] == {"bins_short": [], "rows_borrowed": 0, "by_bin": {}}
    assert meta["gc"]["ks_drawn_vs_target"] == 0
    # Type-7 quantiles of 5/10/15/20 rows at 0.3/0.4/0.5/0.6 and of the target's 10/20/30/40 windows agree.
    quantiles = ("q01", "q10", "q25", "q50", "q75", "q90", "q99")
    assert [meta["gc"]["drawn"][q] for q in quantiles] == [0.3, 0.39, 0.4, 0.5, 0.6, 0.6, 0.6]
    assert [meta["gc"]["target"][q] for q in quantiles] == [0.3, 0.39, 0.4, 0.5, 0.6, 0.6, 0.6]
    check_bin_invariants(meta, 50)


def test_a_short_bin_borrows_from_the_nearest_bins_below_first_and_the_meta_records_it(tmp_path):
    # Bin 60 wants 20 rows and has 12 tiles. At distance 1 it takes bin 59's 3 tiles, then 5 of bin 61's;
    # bin 58 (distance 2) is never reached.
    available = {30: 40, 40: 40, 50: 40, 58: 40, 59: 3, 60: 12, 61: 40}
    inputs = gc_toy(tmp_path, available=available, targets={30: 10, 40: 20, 50: 30, 60: 40})
    rows, meta = run(inputs, tmp_path / "short" / "null.tsv", n=50)
    by_bin = {b["bin"]: b for b in meta["bins"]}
    assert by_bin[60] == {"bin": 60, "gc_from": 0.6, "gc_to": 0.61, "target_windows": 40, "target": 20,
                          "acgt_tiles": 12, "available": 12, "taken": 12, "shortfall": 8, "borrowed": 8,
                          "borrowed_from": {"59": 3, "61": 5}, "lent": 0, "achieved": 12}  # fmt: skip
    assert (by_bin[59]["lent"], by_bin[61]["lent"], by_bin[58]["lent"]) == (3, 5, 0)
    assert meta["shortfall"] == {
        "bins_short": [60],
        "rows_borrowed": 8,
        "by_bin": {
            "60": {"target": 20, "available": 12, "taken": 12, "borrowed": 8, "borrowed_from": {"59": 3, "61": 5}}
        },
    }
    assert gc_of_rows(inputs, rows) == {30: 5, 40: 10, 50: 15, 59: 3, 60: 12, 61: 5}
    assert meta["gc"]["ks_drawn_vs_target"] > 0
    check_bin_invariants(meta, 50)


def test_too_few_tiles_or_a_chromosome_without_a_fold_writes_nothing(tmp_path):
    inputs = gc_toy(tmp_path, available={40: 5, 50: 5}, targets={40: 3, 50: 3})
    out = tmp_path / "few" / "null.tsv"
    argv = ["--fasta", str(inputs["fasta"]), "--peaks", str(inputs["peaks"][0]), "--blacklist",
            str(inputs["blacklist"]), "--target-list", str(inputs["targets"]), "--manifest", str(inputs["manifest"]),
            "--chroms", ",".join(inputs["chroms"]), "--flank", str(FLANK), "--out", str(out)]  # fmt: skip
    assert gn.main([*argv, "--n", "11"]) == 1
    write_manifest(inputs["manifest"], {"chrG40": "0"})
    assert gn.main([*argv, "--n", "10"]) == 1
    assert not out.exists()


# --- edges, the in-bin draw, the bin and offset formulas, the target window ----------------------------------------


def test_a_window_touching_a_peak_or_blacklist_by_one_base_is_excluded_and_an_adjacent_one_is_not(tmp_path):
    o = gn.tile_offset(SEED, "chrE", W)

    def tile(i: int) -> int:
        return o + i * W

    peaks = [("chrE", tile(10) - 1, tile(10)), ("chrE", tile(20), tile(20) + 1), ("chrE", tile(30) - 3, tile(30))]
    black = [("chrE", tile(35) - 1, tile(35) + 1)]
    inputs = {
        "fasta": write_fasta(tmp_path / "edge.fa", {"chrE": random_seq(tile(40) + 1, 31)}),
        "peaks": [write_peaks(tmp_path / "edge.narrowPeak", peaks)],
        "blacklist": write_bed_gz(tmp_path / "edge.bed.gz", black),
        "targets": write_targets(tmp_path / "edge.tsv", [("chrE", tile(5) + FLANK + 1, "summit100")]),
        "manifest": write_manifest(tmp_path / "edge.json", {"chrE": "0"}),
        "chroms": ["chrE"],
    }
    rows, meta = run(inputs, tmp_path / "edge" / "null.tsv", n=35)
    # Tiles 9 and 20 touch a peak by one base, 29 ends where a peak ends, 34 and 35 share a base each with the
    # blacklist; tile 30 starts where a peak ends (half-open: no overlap) and stays.
    assert {(int(r["pos"]) - 1 - FLANK - o) // W for r in rows} == set(range(40)) - {9, 20, 29, 34, 35}
    per = meta["tiles"]["per_chrom"]["chrE"]
    assert (per["all"], per["not_in_peak"], per["not_blacklisted"], per["in_bounds"]) == (40, 37, 35, 35)


def test_each_bin_is_a_seeded_random_draw_not_its_first_tiles_in_genome_order(tmp_path):
    inputs = gc_toy(tmp_path, available={40: 200}, targets={40: 10})
    picked = {}
    for seed in (SEED, SEED + 1):
        rows, meta = run(inputs, tmp_path / f"seed{seed}" / "null.tsv", n=20, seed=seed)
        offset = meta["offsets"]["chrG40"]
        picked[seed] = sorted((int(r["pos"]) - 1 - FLANK - offset) // W for r in rows)
        assert len(picked[seed]) == 20 and picked[seed] != list(range(20))
    assert picked[SEED] != picked[SEED + 1]  # the seed moves the draw itself, not only the offsets


def test_gc_bins_at_the_model_window_and_largest_remainder_ties():
    assert [gn.gc_bin_of(g, 2114, 100) for g in (0, 21, 22, 1057, 2092, 2093, 2114)] == [0, 0, 1, 50, 98, 99, 99]
    assert gn.largest_remainder([1, 1, 1], 2) == [1, 1, 0]
    assert gn.largest_remainder([3, 1], 2) == [2, 0]
    assert gn.largest_remainder([5, 3, 2], 7) == [4, 2, 1]


def test_tile_offsets_follow_the_documented_stream():
    offsets = {c: gn.tile_offset(SEED, c, W) for c in ("chrA", "chrB", "chrM")}
    assert offsets == {c: random.Random(f"{SEED}:offset:{c}").randrange(W) for c in offsets}
    assert len(set(offsets.values())) > 1


def test_the_target_window_is_the_model_input_around_pos0(toy, tmp_path):
    # chrA has Ns at [9000, 9150): the windows ending at 9000 and starting at 9150 are ACGT-only; 1 bp off is not.
    write_targets(toy["targets"], [("chrA", 9000 - FLANK + 1, "summit100"), ("chrA", 9150 + FLANK + 1, "flank500")])
    _, meta = run(toy, tmp_path / "window" / "null.tsv", n=10)
    assert meta["target"]["windows"] == 2 and meta["target"]["skipped"] == {}


# --- the list goes through build_windows.py ------------------------------------------------------------------------


def test_build_windows_reads_the_list_and_builds_every_row(tmp_path):
    seqs = {"chr1": random_seq(40_000, 21), "chr2": random_seq(30_000, 22, gc=0.4)}
    inputs = {
        "fasta": write_fasta(tmp_path / "bw.fa", seqs),
        "peaks": [write_peaks(tmp_path / "bw.narrowPeak", [("chr1", 10_000, 11_000), ("chr2", 22_000, 23_000)])],
        "blacklist": write_bed_gz(tmp_path / "bw.bed.gz", [("chr2", 5_000, 6_000)]),
        "targets": write_targets(
            tmp_path / "bw_targets.tsv", [("chr1", 10_501, "summit100"), ("chr2", 22_400, "flank500")]
        ),
        "manifest": write_manifest(tmp_path / "bw_manifest.json", {"chr1": "0", "chr2": "1"}),
        "chroms": ["chr1", "chr2"],
    }
    out = tmp_path / "bw" / "genomic_null.tsv"
    rows, meta = run(inputs, out, n=12, flank=bw.FLANK)
    assert meta["params"]["window_len"] == bw.INPUT_LEN
    listed = bw.read_list(out)
    with bw.IndexedFasta(inputs["fasta"]) as fasta:
        kept, dropped = bw.prepare(listed, fasta)
    assert dropped == [] and len(kept) == len(rows) == 12
    arrays = bw.build_variants(kept)
    assert not any(arrays["nonacgt"].tolist())
    assert bw.decode(arrays["a1"][:, bw.FLANK]) == "".join(r["ref"] for r in rows)
    assert bw.decode(arrays["a2"][:, bw.FLANK]) == "".join(r["alt"] for r in rows)
    assert all(int(r["dist_to_summit"]) >= bw.FLANK + 500 for r in rows)  # 1000-bp peaks, summit in the middle
