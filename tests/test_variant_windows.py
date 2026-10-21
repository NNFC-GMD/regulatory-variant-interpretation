"""build_windows.py builds VS's allele windows, refuses what VS would score wrongly, and builds VS's null.

The reference is a line-by-line transcription of kundajelab/variant-scorer's VariantGenerator.__get_allele_seq__ and
PeakGenerator.__get_seq__ (main @0e1e341: src/generators/variant_generator.py:28-76, peak_generator.py:25-32), run on
a small synthetic genome. Each window is compared string by string with what VS's one_hot.dna_to_one_hot would keep:
the upper-cased sequence, with anything other than ACGT as N (src/utils/one_hot.py:22-31). VS's own code imports
TensorFlow and deeplift, so it is transcribed rather than imported.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "workflows" / d) for d in ("variant_scoring", "nulls", "qtl")]

import build_windows as bw  # noqa: E402

F = 1057
LONG_A = 9000  # chrA: lowercase (soft-masked) at [2000, 2300), N at 7000, R (IUPAC) at 7100
SHORT_B = 2400  # chrB: SNVs fit only at 1-based positions 1058-1343


def _random_seq(rng, n):
    return "".join(rng.choice(list("ACGT"), size=n))


def _genome():
    rng = np.random.default_rng(7)
    a = list(_random_seq(rng, LONG_A))
    a[2000:2300] = [c.lower() for c in a[2000:2300]]
    a[7000], a[7100] = "N", "R"
    return {"chrA": "".join(a), "chrB": _random_seq(rng, SHORT_B)}


GENOME = _genome()


def write_fasta(path: Path, records: dict[str, str], width: int = 60, newline: bytes = b"\n") -> Path:
    """A FASTA and the .fai samtools faidx would write for it."""
    fai = []
    with open(path, "wb") as fh:
        for name, seq in records.items():
            fh.write(f">{name} synthetic".encode() + newline)
            offset = fh.tell()
            for i in range(0, len(seq), width):
                fh.write(seq[i : i + width].encode() + newline)
            fai.append(f"{name}\t{len(seq)}\t{offset}\t{width}\t{width + len(newline)}\n")
    Path(f"{path}.fai").write_text("".join(fai))
    return path


@pytest.fixture
def fasta(tmp_path):
    return write_fasta(tmp_path / "genome.fa", GENOME)


@pytest.fixture
def reader(fasta):
    fa = bw.IndexedFasta(fasta)
    yield fa
    fa.close()


# --- VS, transcribed -------------------------------------------------------------------------------------------


class _Seq:
    def __init__(self, seq):
        self.seq = seq

    def __str__(self):
        return self.seq


class _Record:
    def __init__(self, seq):
        self._seq = seq

    def __getitem__(self, s):
        return _Seq(self._seq[s])


class _Genome:
    """What VS uses of pyfaidx.Fasta: genome[chrom][a:b], whose str() and .seq are the bases as stored (case kept).
    Inside the sequence, pyfaidx's slices are Python's (0-based, half-open); past its end both cut short."""

    def __init__(self, records):
        self.records = records

    def __getitem__(self, chrom):
        return _Record(self.records[chrom])

    def get_seq(self, chrom, start, end):
        """pyfaidx.Fasta.get_seq: 1-based, closed."""
        return _Seq(self.records[chrom][start - 1 : end])


def vs_get_allele_seq(genome, chrom, pos, allele1, allele2, flank_size=F, shuf=None):
    """variant_generator.py:28-76; `shuf` stands in for dinuc_shuffle(flank, rng=RandomState(seed)) (:43-45, 62-64)."""
    chrom = str(chrom)
    pos = int(pos)
    allele1 = str(allele1)
    allele2 = str(allele2)

    if allele1 == "-":
        allele1 = ""
    if allele2 == "-":
        allele2 = ""
    ### 1 - indexed position
    pos = pos - 1

    if len(allele1) == len(allele2):
        flank = str(genome[chrom][pos - flank_size : pos + flank_size])
        if shuf:
            flank = shuf(flank)
        allele1_seq = flank[:flank_size] + allele1 + flank[flank_size + len(allele1) :]
        allele2_seq = flank[:flank_size] + allele2 + flank[flank_size + len(allele2) :]

    else:
        assert len(allele1) != len(allele2)
        assert genome[chrom][pos : pos + len(allele1)].seq.upper() == allele1
        mismatch_length = len(allele1) - len(allele2)
        if mismatch_length > 0:
            flank = str(genome[chrom][pos - flank_size : pos + flank_size + mismatch_length])
        else:
            flank = str(genome[chrom][pos - flank_size : pos + flank_size])

        if shuf:
            flank = shuf(flank)

        left_flank = flank[:flank_size]

        allele1_right_flank = flank[flank_size + len(allele1) : flank_size * 2]
        allele2_right_flank = flank[flank_size + len(allele1) : flank_size * 2 + mismatch_length]

        allele1_seq = left_flank + allele1 + allele1_right_flank
        allele2_seq = left_flank + allele2 + allele2_right_flank

    assert len(allele1_seq) == flank_size * 2
    assert len(allele2_seq) == flank_size * 2
    return allele1_seq, allele2_seq


def vs_peak_seq(genome, chrom, start, summit, flank_size=F):
    """peak_generator.py:25-32."""
    chrom = str(chrom)
    start = int(start)
    summit = int(start) + int(summit)
    flank_start = int(summit - flank_size)
    flank_end = int(summit + (flank_size - 1))
    flank = str(genome.get_seq(chrom, flank_start, flank_end))
    return flank


def vs_view(seq: str) -> str:
    """What one_hot.dna_to_one_hot keeps of a sequence: upper-cased, anything but ACGT as N."""
    return "".join(c if c in "ACGT" else "N" for c in seq.upper())


# --- cases -----------------------------------------------------------------------------------------------------


def _g(chrom, pos, n=1):
    """n reference bases from the 1-based pos, upper-cased."""
    return GENOME[chrom][pos - 1 : pos - 1 + n].upper()


def _other(base):
    return {"A": "C", "C": "G", "G": "T", "T": "A"}[base]


_rng = np.random.default_rng(11)
BUILT = {  # name: (chrom, pos, ref, alt)
    "snv": ("chrA", 3000, _g("chrA", 3000), _other(_g("chrA", 3000))),
    "mnv": ("chrA", 3100, _g("chrA", 3100, 3), "".join(_other(b) for b in _g("chrA", 3100, 3))),
    "deletion": ("chrA", 3200, _g("chrA", 3200, 6), _g("chrA", 3200)),
    "insertion": ("chrA", 3300, _g("chrA", 3300), _g("chrA", 3300) + "ACGTTG"),
    "complex": ("chrA", 3400, _g("chrA", 3400, 3), _other(_g("chrA", 3400)) + "T"),
    "soft_masked": ("chrA", 2100, _g("chrA", 2100), _other(_g("chrA", 2100))),
    "near_n": ("chrA", 6500, _g("chrA", 6500), _other(_g("chrA", 6500))),
    "first_fit": ("chrB", F + 1, _g("chrB", F + 1), _other(_g("chrB", F + 1))),
    "last_fit": ("chrB", SHORT_B - F, _g("chrB", SHORT_B - F), _other(_g("chrB", SHORT_B - F))),
    "insertion_1057": ("chrA", 4000, _g("chrA", 4000), _g("chrA", 4000) + _random_seq(_rng, F - 1)),
    "deletion_1057": ("chrA", 4500, _g("chrA", 4500, F), _g("chrA", 4500)),
}
UNBUILDABLE = {  # name: (chrom, pos, ref, alt, reason)
    "off_left": ("chrB", F, _g("chrB", F), _other(_g("chrB", F)), "off_chrom_end"),
    "off_right": (
        "chrB",
        SHORT_B - F + 1,
        _g("chrB", SHORT_B - F + 1),
        _other(_g("chrB", SHORT_B - F + 1)),
        "off_chrom_end",
    ),
    # VS's own bounds test passes this deletion (pos + 1057 <= length) and its length assert then fails.
    "deletion_off_right": (
        "chrB",
        SHORT_B - F - 3,
        _g("chrB", SHORT_B - F - 3, 6),
        _g("chrB", SHORT_B - F - 3),
        "off_chrom_end",
    ),
    "insertion_1058": ("chrA", 5000, _g("chrA", 5000), _g("chrA", 5000) + _random_seq(_rng, F), "allele_too_long"),
}
INVALID = {  # name: (chrom, pos, ref, alt, reason)
    "ref_mismatch": ("chrA", 3000, _other(_g("chrA", 3000)), _g("chrA", 3000), "ref_mismatch"),
    "ref_mismatch_indel": ("chrA", 3200, _g("chrA", 3200) + _other(_g("chrA", 3201)), _g("chrA", 3200), "ref_mismatch"),
    "n_allele": ("chrA", 3500, _g("chrA", 3500), "N", "bad_allele"),
    "star_allele": ("chrA", 3510, _g("chrA", 3510), "*", "bad_allele"),
    "dash_allele": ("chrA", 3520, _g("chrA", 3520), "-", "bad_allele"),
    "ref_is_alt": ("chrA", 3530, _g("chrA", 3530), _g("chrA", 3530), "bad_allele"),
    "unknown_chrom": ("chrZ", 3000, "A", "C", "unknown_chrom"),
}


def _row(name, chrom, pos, ref, alt, stratum="summit100", fold_test="0"):
    return {"variant_id": name, "chrom": chrom, "pos": str(pos), "ref": ref, "alt": alt, "source": "genome",
            "stratum": stratum, "af": "0.01", "dist_to_summit": "0", "fold_test": fold_test}  # fmt: skip


def write_list(path: Path, rows: list[dict]) -> Path:
    cols = list(rows[0])
    path.write_text("\t".join(cols) + "\n" + "".join("\t".join(r[c] for c in cols) + "\n" for r in rows))
    return path


def test_known_cases_are_what_they_claim():
    assert GENOME["chrA"][2000:2300].islower() and GENOME["chrA"][7000] == "N"
    chrom, pos, ref, alt = BUILT["deletion_1057"]
    assert len(ref) == F and len(alt) == 1
    assert len(BUILT["insertion_1057"][3]) == F and len(UNBUILDABLE["insertion_1058"][3]) == F + 1
    chrom, pos, ref, alt, _ = UNBUILDABLE["deletion_off_right"]
    assert pos + F <= SHORT_B  # VS's get_valid_variants keeps it ...
    with pytest.raises(AssertionError):  # ... and its construction then fails
        vs_get_allele_seq(_Genome(GENOME), chrom, pos, ref, alt)


@pytest.mark.parametrize("name", sorted(BUILT))
def test_windows_match_vs_string_by_string(reader, name):
    chrom, pos, ref, alt = BUILT[name]
    kept, dropped = bw.prepare([_row(name, chrom, pos, ref, alt)], reader)
    assert not dropped and len(kept) == 1
    arrays = bw.build_variants(kept)
    assert arrays["a1"].shape == arrays["a2"].shape == (1, bw.INPUT_LEN) and arrays["a1"].dtype == np.uint8
    vs1, vs2 = vs_get_allele_seq(_Genome(GENOME), chrom, pos, ref, alt)
    assert bw.decode(arrays["a1"][0]) == vs_view(vs1)
    assert bw.decode(arrays["a2"][0]) == vs_view(vs2)
    # Both windows hold their allele at index 1057, so both models see it at output position 500.
    assert bw.decode(arrays["a1"][0][F : F + len(ref)]) == ref
    assert bw.decode(arrays["a2"][0][F : F + len(alt)]) == alt
    assert bool(arrays["nonacgt"][0]) == (name == "near_n")


def test_snv_window_is_the_reference_with_the_allele_written_in(reader):
    chrom, pos, ref, alt = BUILT["snv"]
    arrays = bw.build_variants(bw.prepare([_row("snv", chrom, pos, ref, alt)], reader)[0])
    window = GENOME[chrom][pos - 1 - F : pos - 1 + F].upper()
    assert bw.decode(arrays["a1"][0]) == window
    diff = np.flatnonzero(arrays["a1"][0] != arrays["a2"][0])
    assert diff.tolist() == [F]


@pytest.mark.parametrize("name", sorted(UNBUILDABLE))
def test_windows_vs_cannot_build_are_dropped(reader, name):
    chrom, pos, ref, alt, reason = UNBUILDABLE[name]
    kept, dropped = bw.prepare([_row(name, chrom, pos, ref, alt)], reader)
    assert not kept and [(d[0], d[1]) for d in dropped] == [(name, reason)]


@pytest.mark.parametrize("name", sorted(INVALID))
def test_invalid_rows_are_named(reader, name):
    chrom, pos, ref, alt, reason = INVALID[name]
    kept, dropped = bw.prepare([_row(name, chrom, pos, ref, alt)], reader)
    assert not kept and [(d[0], d[1]) for d in dropped] == [(name, reason)]


def _all_rows():
    rows = [_row(n, *BUILT[n]) for n in BUILT]
    rows += [_row(n, *UNBUILDABLE[n][:4]) for n in UNBUILDABLE]
    return rows


def test_cli_refuses_a_ref_mismatch_and_writes_nothing(tmp_path, fasta):
    rows = _all_rows() + [_row("ref_mismatch", *INVALID["ref_mismatch"][:4])]
    out = tmp_path / "w" / "variants.npz"
    assert (
        bw.main(["--list", str(write_list(tmp_path / "list.tsv", rows)), "--fasta", str(fasta), "--out", str(out)]) == 1
    )
    assert not out.parent.exists() or not any(out.parent.iterdir())


def test_cli_variants_end_to_end(tmp_path, fasta):
    rows = _all_rows() + [_row(n, *INVALID[n][:4]) for n in INVALID]
    listed = write_list(tmp_path / "list.tsv", rows)
    out = tmp_path / "w" / "variants.npz"
    assert bw.main(["--list", str(listed), "--fasta", str(fasta), "--out", str(out), "--drop-invalid"]) == 0
    with np.load(out, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    assert arrays["ids"].tolist() == list(BUILT)  # the list's order
    genome = _Genome(GENOME)
    for i, name in enumerate(arrays["ids"].tolist()):
        vs1, vs2 = vs_get_allele_seq(genome, *BUILT[name])
        assert bw.decode(arrays["a1"][i]) == vs_view(vs1) and bw.decode(arrays["a2"][i]) == vs_view(vs2)
    assert arrays["nonacgt"].tolist() == [n == "near_n" for n in arrays["ids"].tolist()]
    meta = json.loads(out.with_suffix(".json").read_text())
    assert meta["n_input"] == len(rows) and meta["n_kept"] == len(BUILT) and meta["n_nonacgt"] == 1
    expected = {}
    for spec in list(UNBUILDABLE.values()) + list(INVALID.values()):
        expected[spec[4]] = expected.get(spec[4], 0) + 1
    assert meta["dropped"] == expected
    assert meta["fasta_md5"] == hashlib.md5(fasta.read_bytes()).hexdigest()
    assert meta["allele_index"] == F and meta["input_len"] == 2 * F
    dropped = out.with_suffix(".dropped.tsv").read_text().splitlines()
    assert dropped[0] == "id\treason\tdetail" and len(dropped) == 1 + len(UNBUILDABLE) + len(INVALID)


def test_list_needs_its_columns_and_unique_ids(tmp_path):
    rows = [_row("v1", *BUILT["snv"]), _row("v1", *BUILT["mnv"])]
    with pytest.raises(ValueError, match="more than once"):
        bw.read_list(write_list(tmp_path / "dup.tsv", rows))
    (tmp_path / "bad.tsv").write_text("variant_id\tchrom\tpos\tref\nv1\tchrA\t3000\tA\n")
    with pytest.raises(ValueError, match="alt"):
        bw.read_list(tmp_path / "bad.tsv")


# --- the null --------------------------------------------------------------------------------------------------


def _null_rows():
    rows = [_row(n, *BUILT[n], stratum="summit100") for n in ("snv", "mnv", "deletion", "insertion", "complex")]
    rows += [_row("soft_masked", *BUILT["soft_masked"], stratum="flank500")]
    rows += [_row("near_n", *BUILT["near_n"], stratum="summit100")]  # not ACGT: never in the null
    rows += [_row("outside", "chrA", 5500, _g("chrA", 5500), _other(_g("chrA", 5500)), stratum="outside")]
    return rows


def _reverse(codes, seed):
    _reverse.calls.append(int(seed))
    return codes[::-1].copy()


def test_null_rows_are_vs_null_rows(reader):
    kept, dropped = bw.prepare(_null_rows(), reader)
    assert not dropped
    _reverse.calls = []
    arrays, stats = bw.build_null(kept, 4, 100, ("summit100", "flank500"), _reverse)
    assert stats["pool"] == 6 and stats["excluded_not_acgt"] == 1 and not stats["with_replacement"]
    assert _reverse.calls == [100, 101, 102, 103] and arrays["seeds"].tolist() == [100, 101, 102, 103]
    assert len(set(arrays["src_ids"].tolist())) == 4  # drawn without replacement
    assert not set(arrays["src_ids"].tolist()) & {"near_n", "outside"}
    genome = _Genome(GENOME)
    specs = {r["variant_id"]: (r["chrom"], int(r["pos"]), r["ref"], r["alt"]) for r in _null_rows()}
    for i, src in enumerate(arrays["src_ids"].tolist()):
        # VS shuffles the whole flank (with a deletion's extra bases), then writes both alleles back at 1057.
        vs1, vs2 = vs_get_allele_seq(genome, *specs[src], shuf=lambda s: s[::-1])
        assert bw.decode(arrays["a1"][i]) == vs_view(vs1) and bw.decode(arrays["a2"][i]) == vs_view(vs2)
        assert arrays["ids"][i] == f"{src}|shuf{100 + i}"
    assert not arrays["nonacgt"].any()


def test_null_with_an_identity_shuffle_is_the_variant_windows(reader):
    kept, _ = bw.prepare(_null_rows(), reader)
    arrays, stats = bw.build_null(kept, 3, 0, ("summit100",), lambda codes, seed: codes.copy())
    assert stats["shuffles_returning_the_flank_unchanged"] == 3
    built = bw.build_variants(kept)
    row = {vid: i for i, vid in enumerate(built["ids"].tolist())}
    for i, src in enumerate(arrays["src_ids"].tolist()):
        assert np.array_equal(arrays["a1"][i], built["a1"][row[src]])
        assert np.array_equal(arrays["a2"][i], built["a2"][row[src]])


def test_null_draws_with_replacement_from_a_small_pool(reader):
    kept, _ = bw.prepare(_null_rows(), reader)
    _reverse.calls = []
    arrays, stats = bw.build_null(kept, 20, 5, ("summit100",), _reverse)
    assert stats["pool"] == 5 and stats["with_replacement"] and len(arrays["ids"]) == 20
    assert set(arrays["src_ids"].tolist()) <= {"snv", "mnv", "deletion", "insertion", "complex"}
    again, _ = bw.build_null(kept, 20, 5, ("summit100",), _reverse)
    assert np.array_equal(again["a1"], arrays["a1"]) and np.array_equal(again["src_ids"], arrays["src_ids"])


def test_null_refuses_a_shuffle_that_changes_the_bases(reader):
    kept, _ = bw.prepare(_null_rows(), reader)
    with pytest.raises(ValueError, match="composition"):
        bw.build_null(kept, 2, 0, ("summit100",), lambda codes, seed: np.zeros_like(codes))


def test_tangermeme_shuffle_keeps_dinucleotides():
    pytest.importorskip("torch")
    pytest.importorskip("tangermeme")
    shuffle = bw.tangermeme_shuffler()
    codes = bw.encode(GENOME["chrB"][:2119].encode())
    out = shuffle(codes, 1234)
    assert out.dtype == np.uint8 and out.shape == codes.shape and not np.array_equal(out, codes)
    assert out[0] == codes[0] and out[-1] == codes[-1]

    def pairs(c):
        return np.bincount(4 * c[:-1].astype(np.int64) + c[1:], minlength=16)

    assert np.array_equal(pairs(out), pairs(codes))
    assert np.array_equal(shuffle(codes, 1234), out) and not np.array_equal(shuffle(codes, 1235), out)


def test_tangermeme_null_end_to_end(reader):
    pytest.importorskip("torch")
    pytest.importorskip("tangermeme")
    kept, _ = bw.prepare(_null_rows(), reader)
    arrays, _ = bw.build_null(kept, 3, 9, ("summit100",), bw.tangermeme_shuffler())
    specs = {r["variant_id"]: (r["ref"], r["alt"]) for r in _null_rows()}
    for i, src in enumerate(arrays["src_ids"].tolist()):
        ref, alt = specs[src]
        assert bw.decode(arrays["a1"][i][F : F + len(ref)]) == ref
        assert bw.decode(arrays["a2"][i][F : F + len(alt)]) == alt
        assert bw.decode(arrays["a1"][i][:F]) == bw.decode(arrays["a2"][i][:F])  # the same shuffled left flank


# --- peaks -----------------------------------------------------------------------------------------------------


def _write_peaks(path: Path) -> Path:
    lines = [
        "chrA\t2500\t3500\tp1\t0\t.\t5\t5\t5\t500",
        "chrA\t2500\t3500\tp1dup\t0\t.\t5\t5\t5\t500",  # duplicate (chrom, start, end, summit)
        "chrA\t2500\t3500\tp1b\t0\t.\t5\t5\t5\t480",  # same peak, another summit: kept, as VS
        "chrA\t4000\t5000\tp2\t0\t.\t5\t5\t5\t-1",  # no summit: the centre
        "chrA\t0\t1000\tp3\t0\t.\t5\t5\t5\t500",  # runs off the start
        "chrZ\t2500\t3500\tp4\t0\t.\t5\t5\t5\t500",
        "chrB\t700\t1500\tp5\t0\t.\t5\t5\t5\t400",  # summit 1100: [43, 2157) fits; VS's bounds keep it too
    ]
    with gzip.open(path, "wt") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


@pytest.mark.parametrize("vs_window", [False, True])
def test_peak_windows(tmp_path, fasta, vs_window):
    peaks = _write_peaks(tmp_path / "peaks.narrowPeak.gz")
    out = tmp_path / "peaks.npz"
    args = ["--peaks", str(peaks), "--fasta", str(fasta), "--out", str(out)] + (["--vs-peak-window"] * vs_window)
    assert bw.main(args) == 0
    with np.load(out, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    assert "a2" not in arrays
    assert arrays["ids"].tolist() == ["chrA:2500-3500:500", "chrA:2500-3500:480", "chrA:4000-5000:500",
                                      "chrB:700-1500:400"]  # fmt: skip
    genome = _Genome(GENOME)
    for i, (chrom, start, offset) in enumerate([("chrA", 2500, 500), ("chrA", 2500, 480), ("chrA", 4000, 500),
                                                ("chrB", 700, 400)]):  # fmt: skip
        summit = start + offset
        assert arrays["summit"][i] == summit and arrays["chrom"][i] == chrom
        if vs_window:
            expected = vs_view(vs_peak_seq(genome, chrom, start, offset))
        else:  # ChromBPNet's training window
            expected = GENOME[chrom][summit - F : summit + F].upper()
        assert bw.decode(arrays["a1"][i]) == expected
    meta = json.loads(out.with_suffix(".json").read_text())
    assert meta["dropped"] == {"duplicate": 1, "off_chrom_end": 1, "unknown_chrom": 1} and meta["n_kept"] == 4


def test_cli_refuses_to_write_through_a_link(tmp_path, fasta):
    # A set linked from another directory (a shared one) is never rebuilt through the link: the .json would be
    # written over the other directory's.
    peaks = _write_peaks(tmp_path / "peaks.narrowPeak.gz")
    shared = tmp_path / "shared" / "peaks.npz"
    assert bw.main(["--peaks", str(peaks), "--fasta", str(fasta), "--out", str(shared)]) == 0
    before = {suffix: shared.with_suffix(suffix).read_bytes() for suffix in (".npz", ".json", ".dropped.tsv")}
    here = tmp_path / "here"
    here.mkdir()
    for suffix in (".npz", ".json"):
        (here / f"peaks{suffix}").symlink_to(shared.with_suffix(suffix))
    assert bw.main(["--peaks", str(peaks), "--fasta", str(fasta), "--out", str(here / "peaks.npz"), "--n", "2"]) == 1
    assert {suffix: shared.with_suffix(suffix).read_bytes() for suffix in before} == before
    assert (here / "peaks.npz").is_symlink() and (here / "peaks.json").is_symlink()
    assert not (here / "peaks.dropped.tsv").exists()


def test_peak_subsample_is_fixed_by_the_seed(tmp_path, fasta):
    peaks = _write_peaks(tmp_path / "peaks.narrowPeak.gz")
    ids = []
    for name in ("a", "b"):
        out = tmp_path / f"{name}.npz"
        assert (
            bw.main(["--peaks", str(peaks), "--fasta", str(fasta), "--out", str(out), "--n", "2", "--seed", "3"]) == 0
        )
        with np.load(out, allow_pickle=False) as z:
            ids.append(z["ids"].tolist())
    assert ids[0] == ids[1] and len(ids[0]) == 2


# --- the FASTA reader ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("width, newline", [(60, b"\n"), (50, b"\r\n"), (2114, b"\n")])
def test_indexed_fasta_reads_any_region(tmp_path, width, newline):
    fa = bw.IndexedFasta(write_fasta(tmp_path / "g.fa", GENOME, width=width, newline=newline))
    rng = np.random.default_rng(0)
    for chrom, seq in GENOME.items():
        bounds = [(0, 0), (0, 1), (0, len(seq)), (len(seq) - 1, len(seq)), (width - 1, width + 1), (width, 2 * width)]
        bounds += [tuple(sorted(rng.integers(0, len(seq) + 1, size=2))) for _ in range(50)]
        for start, end in bounds:
            if end <= len(seq):
                assert fa.fetch(chrom, int(start), int(end)) == seq[start:end].upper().encode()
    for start, end in ((-1, 10), (10, SHORT_B + 1), (5, 4)):
        with pytest.raises(ValueError):
            fa.fetch("chrB", start, end)


def test_the_genome_stand_in_slices_as_pyfaidx_does(fasta):
    pyfaidx = pytest.importorskip("pyfaidx")
    real = pyfaidx.Fasta(str(fasta))
    stand_in = _Genome(GENOME)
    for chrom, start, end in (("chrA", 1000, 3114), ("chrA", 1990, 2310), ("chrB", 0, 2114), ("chrB", 286, 2400)):
        assert str(real[chrom][start:end]) == str(stand_in[chrom][start:end])
        assert str(real.get_seq(chrom, start + 1, end)) == str(stand_in.get_seq(chrom, start + 1, end))
