"""gnomad_list.py: windows, checks, strata and draws of the variant-list builder (no network, no bcftools)."""

from __future__ import annotations

import argparse
import gzip
import http.server
import json
import random
import re
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "workflows" / d) for d in ("variant_scoring", "nulls", "qtl")]

import gnomad_list as gl  # noqa: E402

LINE = 60


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


def random_seq(n: int, seed: int) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(n))


@pytest.fixture
def genome(tmp_path):
    seq = random_seq(20_000, 1)
    seq = seq[:15_000] + "N" * 100 + seq[15_100:]
    return write_fasta(tmp_path / "g.fa", {"chr1": seq, "chr2": random_seq(12_000, 2).lower()}), seq


def test_fasta_fetch_matches_the_sequence(genome):
    path, seq = genome
    fa = gl.Fasta(path)
    for start, end in [(0, 1), (59, 61), (60, 120), (1234, 5678), (19_990, 20_000)]:
        assert fa.fetch("chr1", start, end) == seq[start:end]
    assert fa.fetch("chr2", 0, 5).isupper()
    with pytest.raises(ValueError):
        fa.fetch("chr1", 19_999, 20_001)


def test_intervals_merge_find_and_overlap():
    assert gl.merge([(5, 10), (0, 3), (3, 4), (9, 12)]) == [(0, 4), (5, 12)]
    iv = gl.IntervalSet([(10, 20), (30, 40)])
    assert [iv.find(p) for p in (9, 10, 19, 20, 35, 40)] == [-1, 0, 0, -1, 1, -1]
    assert iv.overlaps(19, 25) and iv.overlaps(0, 11) and not iv.overlaps(20, 30) and not iv.overlaps(40, 50)


def test_summit_distance_and_outside_test():
    summits = [100, 1000]
    assert gl.nearest_distance(summits, 100) == 0
    assert gl.nearest_distance(summits, 550) == 450
    assert gl.nearest_distance(summits, 551) == 449
    assert gl.nearest_distance([], 5) is None
    # every base of [start, end) must be more than 500 bp from every summit
    assert gl.far_from_summits(summits, 1501, 1600, 500)
    assert not gl.far_from_summits(summits, 1500, 1600, 500)
    assert not gl.far_from_summits(summits, 0, 10, 500)
    assert gl.far_from_summits([], 0, 10, 500)


def test_strata_edges():
    assert [gl.stratum_of(d, 100, 500) for d in (0, 100, 101, 500, 501)] == [0, 0, 1, 1, 2]
    assert gl.strata_names(100, 500) == ("summit100", "flank500", "outside")


def test_checker_reasons(genome):
    path, seq = genome
    ck = gl.Checker(gl.Fasta(path), ["chr1", "chr2"], 2114)
    pos = 5000  # 1-based
    ref = seq[pos - 1]
    alt = "A" if ref != "A" else "C"
    assert ck.reason("chr1", pos, ref, alt) is None
    assert ck.reason("chr3", pos, ref, alt) == "other_chrom"
    assert ck.reason("chr1", pos, ref, "A,C") == "multiallelic"
    assert ck.reason("chr1", pos, ref, "*") == "non_acgt_allele"
    assert ck.reason("chr1", pos, seq[pos - 1 : pos + 1], "AA") == "mnv"
    assert ck.reason("chr1", pos, alt, ref) == "ref_mismatch"
    assert ck.reason("chr1", 1057, seq[1056], alt) == "window_off_chrom"
    assert ck.reason("chr1", 1058, seq[1057], "A" if seq[1057] != "A" else "C") is None
    # a deletion reads len(ref) - len(alt) extra right-flank bases: the SNV's window fits, the deletion's does not
    pos0 = 20_000 - 1057 - 2
    ref6 = seq[pos0 : pos0 + 6]
    assert ck.reason("chr1", pos0 + 1, seq[pos0], "A" if seq[pos0] != "A" else "C") is None
    assert ck.reason("chr1", pos0 + 1, ref6[:3], ref6[0]) is None
    assert ck.reason("chr1", pos0 + 1, ref6, ref6[0]) == "window_off_chrom"
    # VS's get_valid_variants needs pos + 1057 <= length, 1 bp more than the SNV's window [pos0 - 1057, pos0 + 1057)
    last = 20_000 - 1057
    assert ck.reason("chr1", last, *snv(seq, last)) is None
    assert ck.reason("chr1", last + 1, *snv(seq, last + 1)) == "window_off_chrom"
    # N block at 15,000-15,100 lies inside the window of a variant at 14,500
    assert ck.reason("chr1", 14_500, seq[14_499], "A" if seq[14_499] != "A" else "C") == "window_non_acgt"
    # soft-masked FASTA bases compare case-insensitively
    s2 = random_seq(12_000, 2)
    assert ck.reason("chr2", 6000, s2[5999], "A" if s2[5999] != "A" else "C") is None


def test_read_summits_needs_narrowpeak_columns(tmp_path):
    peaks = tmp_path / "p.narrowPeak"
    peaks.write_text("chr1\t100\t1100\tp\t0\t.\t1\t1\t1\t500\nchr1\t2000\t3000\tq\t0\t.\t1\t1\t1\t-1\n")
    summits, rows = gl.read_summits([peaks])
    assert summits == {"chr1": [600, 2500]} and rows == {str(peaks): 2}  # start + column 10; -1: the midpoint
    bed = tmp_path / "p.bed"
    bed.write_text("chr1\t100\t1100\n")
    with pytest.raises(ValueError, match="3 columns"):
        gl.read_summits([bed])


def test_parse_budget_and_norm_log():
    assert gl.parse_budget("exome:summit=10,outside=3") == ("exome", {"summit": 10, "outside": 3})
    with pytest.raises(argparse.ArgumentTypeError):
        gl.parse_budget("exome:inside=1")
    log = (
        "Reference allele mismatch at chr1:5 .. REF_SEQ:'A' vs VCF:'C'\n"
        "Lines   total/split/joined/realigned/mismatch_removed/dup_removed/skipped:\t12/1/0/2/0/0/0\n"
    )
    out = gl.parse_norm_log(log)
    assert out["ref_mismatch_warnings"] == 1
    assert out["total"] == 12 and out["split"] == 1 and out["realigned"] == 2


def write_vcf(path: Path, records):
    with gzip.open(path, "wt") as fh:
        fh.write("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for chrom, pos, ref, alt, af, filt in records:
            fh.write(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\t{filt}\tAF={af};allele_type=snv\n")


def fake_fetch(tmp_path: Path, name: str, bed_lines: list[str], records) -> Path:
    """A fetch directory as `fetch` leaves it: one job (BED and VCF) per chromosome, and fetch.json."""
    d = tmp_path / name
    d.mkdir()
    jobs = []
    for chrom in sorted({x.split("\t")[0] for x in bed_lines}):
        bed = d / f"{chrom}.bed"
        bed.write_text("".join(x for x in bed_lines if x.split("\t")[0] == chrom))
        vcf = d / f"{chrom}.000.vcf.gz"
        write_vcf(vcf, sorted((r for r in records if r[0] == chrom), key=lambda r: r[1]))
        jobs.append({"job": f"{chrom}.000", "chrom": chrom, "bed": str(bed), "regions": 1, "bp": 1,
                     "vcf": str(vcf), "norm": {}})  # fmt: skip
    fetch = {
        "name": name,
        "url_template": "unused/{chrom}",
        "bcftools": "bcftools",
        "bcftools_version": ["test"],
        "filters": "PASS",
        "keep_info": ["AF", "allele_type", "variant_type"],
        "started_et": "t0",
        "finished_et": "t1",
        "seconds": 0,
        "out_bytes": 0,
        "failed": [],
        "per_job": jobs,
    }
    (d / "fetch.json").write_text(json.dumps(fetch))
    return d


def snv(seq, pos):
    ref = seq[pos - 1]
    return ref, "A" if ref != "A" else "G"


def test_sample_end_to_end(genome, tmp_path):
    path, seq = genome
    summit = 8000  # 0-based
    peaks = tmp_path / "p.narrowPeak"
    peaks.write_text(f"chr1\t{summit - 500}\t{summit + 500}\tp\t0\t.\t1\t1\t1\t500\n")
    folds = tmp_path / "folds"
    folds.mkdir()
    (folds / "fold_0.json").write_text(json.dumps({"train": [], "valid": [], "test": ["chr1"]}))
    (folds / "fold_1.json").write_text(json.dumps({"train": [], "valid": [], "test": ["chr2"]}))
    bed = [f"chr1\t{summit - 500}\t{summit + 501}\n", "chr1\t11000\t12000\n"]
    near = [summit + 1 + d for d in (-100, -50, 0, 30, 100)]  # 1-based pos with pos-1 within 100 of the summit
    flank = [summit + 1 + d for d in (-400, 101, 300, 500)]
    outside = [11_100, 11_500, 11_900]
    shared = {near[0], flank[0], outside[0]}
    exome, genome_recs = [], []
    for p in near + flank + outside:
        ref, alt = snv(seq, p)
        exome.append(("chr1", p, ref, alt, "0.01", "PASS"))
        if p in shared:
            genome_recs.append(("chr1", p, ref, alt, "0.02", "PASS"))
    ref, alt = snv(seq, near[1])
    exome.append(("chr1", near[1], alt, ref, "0.5", "PASS"))  # REF mismatch, dropped
    ref, alt = snv(seq, near[2])
    exome.append(("chr1", near[2] + 1, seq[near[2]], "*", "0.5", "PASS"))  # spanning deletion allele, dropped
    exome.append(("chr1", near[3], *snv(seq, near[3]), "0.5", "AC0"))  # not PASS
    for p in (summit + 1 + 20, summit + 1 + 200, 11_300):
        ref, alt = snv(seq, p)
        genome_recs.append(("chr1", p, ref, alt, "0.03", "PASS"))
    e_dir = fake_fetch(tmp_path, "exome", bed, sorted(exome, key=lambda r: r[1]))
    g_dir = fake_fetch(tmp_path, "genome", bed, sorted(genome_recs, key=lambda r: r[1]))
    out = tmp_path / "lists" / "t.tsv"
    argv = [
        "sample",
        "--callset", f"exome={e_dir}",
        "--callset", f"genome={g_dir}",
        "--peaks", str(peaks),
        "--genome", str(path),
        "--folds-dir", str(folds),
        "--chroms", "chr1,chr2",
        "--target", "10",
        "--fractions", "0.6,0.2,0.2",
        "--cap-per-region", "100",
        "--af-priority", "genome,exome",
        "--out", str(out),
    ]  # fmt: skip
    assert gl.main(argv) == 0
    lines = out.read_text().splitlines()
    assert lines[0].split("\t") == list(gl.LIST_COLUMNS)
    rows = [dict(zip(gl.LIST_COLUMNS, x.split("\t"))) for x in lines[1:]]
    by_id = {r["variant_id"]: r for r in rows}
    assert len(by_id) == len(rows)
    for r in rows:
        d = int(r["dist_to_summit"])
        assert d == abs(int(r["pos"]) - 1 - summit) or r["stratum"] == "outside"
        assert r["stratum"] == ("summit100" if d <= 100 else "flank500" if d <= 500 else "outside")
        assert r["fold_test"] == "0"
        key = (r["chrom"], int(r["pos"]), r["ref"], r["alt"])
        if int(r["pos"]) in shared and key[2] == snv(seq, int(r["pos"]))[0]:
            assert r["source"] == "both" and r["af"] == "0.02"
    meta = json.loads(out.with_name("t.meta.json").read_text())
    assert meta["drops"]["exome"]["ref_mismatch"] == 1
    assert meta["drops"]["exome"]["non_acgt_allele"] == 1
    assert meta["drops"]["exome"]["not_pass"] == 1
    draws = meta["draws"]
    # 6 summit100 wanted: exome holds 5 valid near-summit SNVs (the swapped REF, the `*` allele and the AC0 copy
    # are dropped); genome adds its 1 not yet drawn
    assert draws["exome"]["summit100"]["drawn"] == 5 and draws["exome"]["summit100"]["shortfall"] == 1
    assert draws["exome"]["outside"]["drawn"] == 2
    assert draws["genome"]["summit100"]["drawn"] == 1
    assert sum(meta["counts_by_stratum"].values()) == len(rows)
    # deterministic
    first = out.read_text()
    assert gl.main(argv) == 0
    assert out.read_text() == first


def test_cap_per_region_limits_draws(genome, tmp_path):
    """The cap counts per merged region; regions with the same index on two chromosomes are different regions."""
    path, seq = genome
    seq2 = random_seq(12_000, 2)
    peaks = tmp_path / "p.narrowPeak"
    peaks.write_text("chr1\t2500\t3500\tp\t0\t.\t1\t1\t1\t500\n")
    folds = tmp_path / "folds"
    folds.mkdir()
    (folds / "fold_0.json").write_text(json.dumps({"train": [], "valid": [], "test": ["chr1", "chr2"]}))
    recs = [("chr1", p, *snv(seq, p), "0.1", "PASS") for p in range(11_001, 11_050)]
    recs += [("chr2", p, *snv(seq2, p), "0.1", "PASS") for p in range(5_001, 5_050)]
    e_dir = fake_fetch(tmp_path, "exome", ["chr1\t11000\t12000\n", "chr2\t5000\t6000\n"], recs)
    out = tmp_path / "c.tsv"
    argv = ["sample", "--callset", f"exome={e_dir}", "--peaks", str(peaks), "--genome", str(path),
            "--folds-dir", str(folds), "--chroms", "chr1,chr2", "--target", "100", "--cap-per-region", "7",
            "--out", str(out)]  # fmt: skip
    assert gl.main(argv) == 0
    meta = json.loads(out.with_name("c.meta.json").read_text())
    assert meta["draws"]["exome"]["outside"]["pool"] == 98
    assert meta["draws"]["exome"]["outside"]["pool_regions"] == 2
    assert meta["draws"]["exome"]["outside"]["drawn"] == 14
    assert meta["counts_by_chrom"] == {"chr1": 7, "chr2": 7}


def test_regions_windows_respect_the_rules(genome, tmp_path):
    path, seq = genome
    peaks = tmp_path / "p.narrowPeak"
    peaks.write_text(
        "".join(f"chr1\t{s - 500}\t{s + 500}\tp\t0\t.\t1\t1\t1\t500\n" for s in (3000, 6000, 9000))
        + "chr2\t5500\t6500\tp\t0\t.\t1\t1\t1\t500\n"
    )
    blacklist = tmp_path / "bl.bed"
    blacklist.write_text("chr2\t9000\t9100\n")
    out = tmp_path / "regions"
    argv = ["regions", "--peaks", str(peaks), "--genome", str(path), "--blacklist", str(blacklist),
            "--blacklist-slop", "100", "--chroms", "chr1,chr2", "--n-summit", "4", "--n-outside", "6",
            "--outside-len", "200", "--gap-pad", "300", "--budget", "a:summit=4,outside=6",
            "--budget", "b:summit=2,outside=3", "--out-dir", str(out)]  # fmt: skip
    assert gl.main(argv) == 0
    windows = [x.split("\t") for x in (out / "windows.tsv").read_text().splitlines()[1:]]
    summits = {"chr1": [3000, 6000, 9000], "chr2": [6000]}
    outside = [w for w in windows if w[4] == "outside"]
    assert len(outside) == 6 and len([w for w in windows if w[4] == "summit"]) == 4
    for _, chrom, start, end, *_ in outside:
        start, end = int(start), int(end)
        assert gl.far_from_summits(summits[chrom], start, end, 500)
        assert "N" not in gl.Fasta(path).fetch(chrom, start - 300, end + 300)
        assert not (chrom == "chr2" and start < 9200 and end > 8900)
    # budget b is a subset of budget a
    a = {tuple(x.split("\t")) for p in (out / "a").glob("chr*.bed") for x in p.read_text().splitlines()}
    b = {tuple(x.split("\t")) for p in (out / "b").glob("chr*.bed") for x in p.read_text().splitlines()}
    covered_a = {c: gl.IntervalSet([(int(s), int(e)) for cc, s, e in a if cc == c]) for c in ("chr1", "chr2")}
    for c, s, e in b:
        assert covered_a[c].find(int(s)) >= 0 and covered_a[c].find(int(e) - 1) >= 0


def test_cluster_positions():
    assert gl.cluster_positions([5, 1, 3, 2000, 2999, 4100], 1000) == [(1, 6), (2000, 3000), (4100, 4101)]
    assert gl.cluster_positions([], 1000) == []


class _Origin(http.server.BaseHTTPRequestHandler):
    """A tiny range-serving origin; with `flaky`, every other ranged response stops halfway."""

    protocol_version = "HTTP/1.1"
    data = bytes(range(256)) * 400
    flaky = False
    served = 0

    def log_message(self, format, *args):  # noqa: A002
        pass

    def do_HEAD(self):
        if self.path != "/f.bin":
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.data)))
        self.end_headers()

    def do_GET(self):
        m = re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers.get("Range", ""))
        if self.path != "/f.bin" or not m:
            self.send_response(400)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        a, b = int(m.group(1)), min(int(m.group(2)), len(self.data) - 1)
        body = self.data[a : b + 1]
        type(self).served += 1
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {a}-{b}/{len(self.data)}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.flaky and type(self).served % 2 == 0 and len(body) > 1:
            self.wfile.write(body[: len(body) // 2])
            self.close_connection = True
            return
        self.wfile.write(body)


@pytest.fixture(params=[False, True], ids=["steady", "flaky"])
def origin(request):
    handler = type("Origin", (_Origin,), {"flaky": request.param, "served": 0})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", handler
    server.shutdown()
    server.server_close()


def _open(request):
    """urlopen that ignores any proxy settings of the environment (the forwarder is on the loopback)."""
    return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request)


def test_forwarder_reassembles_ranges(origin):
    base, handler = origin
    fw = gl.RangeForwarder(base, resumes=20, recv_buffer=4096)
    try:
        url = fw.local_url(base + "/f.bin")
        data = handler.data
        with _open(url) as r:
            assert r.status == 200 and r.read() == data
        req = urllib.request.Request(url, headers={"Range": "bytes=12345-"})
        with _open(req) as r:
            assert r.status == 206
            assert r.headers["Content-Range"] == f"bytes 12345-{len(data) - 1}/{len(data)}"
            assert r.read() == data[12345:]
        req = urllib.request.Request(url, headers={"Range": "bytes=100-199"})
        with _open(req) as r:
            assert r.read() == data[100:200]
        with _open(urllib.request.Request(url, method="HEAD")) as r:
            assert int(r.headers["Content-Length"]) == len(data)
        with pytest.raises(urllib.error.HTTPError) as err:
            _open(fw.local_url(base + "/missing.bin"))
        assert err.value.code == 404
        with pytest.raises(urllib.error.HTTPError) as err:
            _open(urllib.request.Request(url, headers={"Range": f"bytes={len(data)}-"}))
        assert err.value.code == 416
        # a client that reads a little and hangs up, as htslib does at every seek
        with _open(urllib.request.Request(url, headers={"Range": "bytes=0-"})) as r:
            assert r.read(10) == data[:10]
    finally:
        stats = fw.close()
    assert stats["client_bytes"] <= stats["upstream_bytes"]
    assert stats["paths"] == {"/f.bin": 200, "/missing.bin": 404}
    if handler.flaky:
        assert stats["upstream_retries"] > 0
    with pytest.raises(ValueError):
        fw.local_url("https://elsewhere/f.bin")
