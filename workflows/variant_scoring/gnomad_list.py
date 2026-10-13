#!/usr/bin/env python3
"""Build a variant list from gnomAD-style sites VCFs, stratified by distance to peak summits.

Python standard library only (plus the bcftools binary for `fetch`), so it runs
in a small bcftools/htslib environment and never loads numpy. Three steps, each
a subcommand writing into the directory it is given:

  regions  The union of the summits (start + column 10) of the given narrowPeak
           files, and two kinds of ranked random windows (seeded):
             summit   summit +/- --summit-half bp (closed), summits drawn
                      uniformly from the union on --chroms;
             outside  --outside-len bp windows whose every base is more than
                      --outer bp from every summit, with no non-ACGT base within
                      --gap-pad bp, clear of --blacklist +/- --blacklist-slop.
           Each --budget NAME:summit=K,outside=K takes the first K windows of
           each kind by rank, so a smaller budget is a subset of a larger one.
           Writes windows.tsv, regions.json and NAME/<chrom>.bed (merged, sorted).

  fetch    One callset over one budget's BED files, in jobs of --chunk-regions
           regions of one chromosome each:
             bcftools view -f PASS -R <job>.bed --regions-overlap 0 <url>  (POS in a region)
             | bcftools annotate -x ^INFO/AF,INFO/allele_type,INFO/variant_type
             | bcftools norm -m -any -f <genome> -c w
           then `bcftools index -t`. --url-template has a {chrom} field. An https
           URL is read through RangeForwarder (below): bcftools gets
           http://127.0.0.1:<port>/<path>, and the forwarder fetches the origin
           over HTTPS with HTTP/1.1. The .tbi files are downloaded first into
           <out-dir>/index_cache, bcftools's working directory, where htslib looks
           for a remote file's index, so parallel jobs never race to write one.
           Writes <job>.vcf.gz, job_beds/, logs/ (appended per attempt) and
           fetch.json (commands, per-job timings and sizes, forwarder byte and
           request counts, the remote files' HTTP headers).
           Volume: htslib reads a region from the first record of the 16-kb
           linear-index window that holds its start (an empty window inherits the
           previous window's offset). With gnomAD's 5-8 KB records a 1-kb region
           therefore costs megabytes of transfer, which suits thousands of regions,
           not millions.

  sample   Reads every --callset NAME=<fetch out-dir> (draw order = order given).
           Each record is checked, in this order, and dropped and counted if:
             not_pass, other_chrom, multiallelic (ALT still has a comma after norm),
             non_acgt_allele (includes `*`), mnv (equal-length multi-base),
             window_off_chrom (the --input-len window, plus the extra right flank
             a deletion reads, must fit, with VS's get_valid_variants bounds,
             which are 1 bp stricter on the right; alleles must fit in half the
             window),
             ref_mismatch (REF against the FASTA, case-insensitive),
             window_non_acgt (any non-ACGT base in that window), duplicate.
           Stratum, from the distance d between POS-1 (the 0-based base placed at
           window index input_len // 2) and the nearest summit of the union:
             summit<inner>  d <= --inner;  flank<outer>  --inner < d <= --outer;
             outside  d > --outer.
           Each callset then draws --target * fraction variants per stratum
           (random.Random("<seed>:<callset>:<stratum>")) from the variants not yet
           drawn, at most --cap-per-region from one merged fetch region (per
           chromosome); shortfalls are reported, never filled from another stratum.
           Callset membership of every drawn variant is then completed: positions
           the other callset's fetch did not cover are queried there directly
           (<fetch out-dir>/complete/; positions closer than --complete-gap share a
           query). `source` is the callsets the variant is a PASS record of: one
           name, or `both`. `af` is AF from the first callset in --af-priority that
           has the variant.

Outputs of `sample` (--out <list>.tsv):
  <list>.tsv            variant_id chrom pos ref alt source stratum af dist_to_summit fold_test
                        (header row; pos 1-based VCF; anchored VCF alleles; variant_id
                        chrom:pos:ref:alt; fold_test = the fold whose test set holds chrom)
  <list>.callsets.tsv   per variant: the draw, membership and AF in every callset, allele_type, ID
  <list>.meta.json      inputs, filters, definitions, every count, drops, timings (ET)

    python -I gnomad_list.py regions --peaks a.narrowPeak b.narrowPeak --genome g.fa \\
        --blacklist bl.bed.gz --n-summit 60000 --n-outside 30000 \\
        --budget exome:summit=3000,outside=1500 --budget genome:summit=800,outside=300 \\
        --out-dir lists/regions
    python -I gnomad_list.py fetch --name exome --url-template 'https://.../exomes.{chrom}.vcf.bgz' \\
        --bed-dir lists/regions/exome --genome g.fa --out-dir downloads/exome
    python -I gnomad_list.py sample --callset exome=downloads/exome --callset genome=downloads/genome \\
        --peaks a.narrowPeak b.narrowPeak --genome g.fa --folds-dir folds --out lists/list.tsv
"""

from __future__ import annotations

import argparse
import bisect
import collections
import datetime as dt
import gzip
import hashlib
import http.client
import http.server
import itertools
import json
import logging
import queue
import random
import re
import shutil
import socket
import ssl
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zoneinfo import ZoneInfo

logger = logging.getLogger("gnomad_list")

ET = ZoneInfo("America/New_York")
NON_ACGT = re.compile(r"[^ACGT]")
DEFAULT_CHROMS = ",".join([f"chr{i}" for i in range(1, 23)] + ["chrX"])
KINDS = ("summit", "outside")
LIST_COLUMNS = ("variant_id", "chrom", "pos", "ref", "alt", "source", "stratum", "af", "dist_to_summit", "fold_test")
DROP_REASONS = (
    "not_pass",
    "other_chrom",
    "multiallelic",
    "non_acgt_allele",
    "mnv",
    "window_off_chrom",
    "ref_mismatch",
    "window_non_acgt",
    "duplicate",
)
NORM_FIELDS = ("total", "split", "joined", "realigned", "mismatch_removed", "dup_removed", "skipped")
ATTEMPT_MARK = "### attempt"


def now_et() -> str:
    return dt.datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S %Z")


def write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)


def open_text(path):
    return gzip.open(path, "rt") if str(path).endswith((".gz", ".bgz")) else open(path)


class Fasta:
    """Random access to an uncompressed FASTA with a .fai beside it; sequences come back upper-case."""

    def __init__(self, path):
        self.path = Path(path)
        self.index: dict[str, tuple[int, int, int, int]] = {}
        self.order: list[str] = []
        with open(f"{path}.fai") as fh:
            for line in fh:
                name, length, offset, line_bases, line_width = line.rstrip("\n").split("\t")[:5]
                self.index[name] = (int(length), int(offset), int(line_bases), int(line_width))
                self.order.append(name)
        self.fh = open(path, "rb")

    def length(self, chrom: str) -> int:
        return self.index[chrom][0]

    def fetch(self, chrom: str, start: int, end: int) -> str:
        """Bases [start, end), 0-based half-open."""
        length, offset, lb, lw = self.index[chrom]
        if not 0 <= start <= end <= length:
            raise ValueError(f"{chrom}:{start}-{end} is outside 0-{length}")
        a = offset + (start // lb) * lw + start % lb
        b = offset + (end // lb) * lw + end % lb
        self.fh.seek(a)
        return self.fh.read(b - a).translate(None, b"\r\n").decode("ascii").upper()


def merge(intervals) -> list[tuple[int, int]]:
    """Sorted, merged [start, end) intervals; touching intervals are joined."""
    out: list[list[int]] = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


class IntervalSet:
    """Disjoint sorted [start, end) intervals on one chromosome."""

    def __init__(self, intervals):
        merged = merge(intervals)
        self.starts = [s for s, _ in merged]
        self.ends = [e for _, e in merged]

    def __len__(self) -> int:
        return len(self.starts)

    def find(self, pos: int) -> int:
        """Index of the interval holding pos, or -1."""
        i = bisect.bisect_right(self.starts, pos) - 1
        return i if i >= 0 and pos < self.ends[i] else -1

    def overlaps(self, start: int, end: int) -> bool:
        i = bisect.bisect_right(self.starts, end - 1)
        return i > 0 and self.ends[i - 1] > start


def read_bed(path, slop: int = 0) -> dict[str, IntervalSet]:
    per: dict[str, list] = collections.defaultdict(list)
    with open_text(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            f = line.split("\t")
            per[f[0]].append((max(0, int(f[1]) - slop), int(f[2]) + slop))
    return {c: IntervalSet(v) for c, v in per.items()}


def read_summits(paths) -> tuple[dict[str, list[int]], dict[str, int]]:
    """Union of 0-based summits (start + column 10; the midpoint if column 10 is -1), sorted per chromosome."""
    per: dict[str, set] = collections.defaultdict(set)
    rows = {}
    for path in paths:
        n = 0
        with open_text(path) as fh:
            for line in fh:
                if not line.strip() or line.startswith(("#", "track", "browser")):
                    continue
                f = line.rstrip("\n").split("\t")
                if len(f) < 10:
                    raise ValueError(f"{path}: a row with {len(f)} columns; a narrowPeak has 10 (summit offset last)")
                start, end, offset = int(f[1]), int(f[2]), int(f[9])
                per[f[0]].add(start + offset if offset >= 0 else (start + end) // 2)
                n += 1
        rows[str(path)] = n
    return {c: sorted(v) for c, v in per.items()}, rows


def nearest_distance(summits: list[int], x: int) -> int | None:
    """|x - s| for the summit s nearest to x; None without summits."""
    i = bisect.bisect_left(summits, x)
    best = None
    if i < len(summits):
        best = summits[i] - x
    if i > 0 and (best is None or x - summits[i - 1] < best):
        best = x - summits[i - 1]
    return best


def far_from_summits(summits: list[int], start: int, end: int, outer: int) -> bool:
    """True if every base of [start, end) is more than `outer` bp from every summit."""
    i = bisect.bisect_left(summits, start - outer)
    return i == len(summits) or summits[i] > end - 1 + outer


def parse_chroms(text: str) -> list[str]:
    return [c for c in text.split(",") if c]


def parse_budget(text: str) -> tuple[str, dict[str, int]]:
    """NAME:summit=K,outside=K"""
    name, sep, rest = text.partition(":")
    if not sep or not name:
        raise argparse.ArgumentTypeError(f"--budget wants NAME:summit=K,outside=K, got {text!r}")
    budget = {k: 0 for k in KINDS}
    for item in rest.split(","):
        key, sep, value = item.partition("=")
        if key not in KINDS or not sep:
            raise argparse.ArgumentTypeError(f"--budget {text!r}: unknown item {item!r}")
        budget[key] = int(value)
    return name, budget


def parse_fractions(text: str) -> list[float]:
    values = [float(x) for x in text.split(",")]
    if len(values) != 3 or any(v < 0 for v in values):
        raise argparse.ArgumentTypeError("--fractions wants three non-negative numbers: in-summit,flank,outside")
    return values


def strata_names(inner: int, outer: int) -> tuple[str, str, str]:
    return f"summit{inner}", f"flank{outer}", "outside"


def stratum_of(d: int, inner: int, outer: int) -> int:
    return 0 if d <= inner else 1 if d <= outer else 2


# --------------------------------------------------------------------------- regions


def cmd_regions(args) -> int:
    started = now_et()
    fasta = Fasta(args.genome)
    chroms = parse_chroms(args.chroms)
    missing = [c for c in chroms if c not in fasta.index]
    if missing:
        logger.error("not in the FASTA: %s", missing)
        return 1
    summits, peak_rows = read_summits(args.peaks)
    blacklist = read_bed(args.blacklist, args.blacklist_slop) if args.blacklist else {}
    budgets = dict(args.budget)
    for name, budget in budgets.items():
        if budget["summit"] > args.n_summit or budget["outside"] > args.n_outside:
            logger.error("budget %s asks for more windows than --n-summit/--n-outside rank", name)
            return 1

    windows = []
    flat = [(c, s) for c in chroms for s in summits.get(c, [])]
    rng = random.Random(f"{args.seed}:summit_windows")
    picks = rng.sample(range(len(flat)), min(args.n_summit, len(flat)))
    for rank, i in enumerate(picks):
        c, s = flat[i]
        start, end = max(0, s - args.summit_half), min(fasta.length(c), s + args.summit_half + 1)
        windows.append((f"S{rank:06d}", c, start, end, "summit", rank, str(s)))

    rng = random.Random(f"{args.seed}:outside_windows")
    cumulative = list(itertools.accumulate(fasta.length(c) for c in chroms))
    accepted: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
    rejected: collections.Counter = collections.Counter()
    tries, rank, pad, length = 0, 0, args.gap_pad, args.outside_len
    while rank < args.n_outside:
        tries += 1
        if tries > args.max_tries:
            logger.error("only %d outside windows after %d tries", rank, args.max_tries)
            return 1
        x = rng.randrange(cumulative[-1])
        ci = bisect.bisect_right(cumulative, x)
        c = chroms[ci]
        start = x - (cumulative[ci - 1] if ci else 0)
        end = start + length
        if start - pad < 0 or end + pad > fasta.length(c):
            rejected["chrom_end"] += 1
            continue
        if not far_from_summits(summits.get(c, []), start, end, args.outer):
            rejected["near_summit"] += 1
            continue
        if c in blacklist and blacklist[c].overlaps(start, end):
            rejected["blacklist"] += 1
            continue
        taken = accepted[c]
        j = bisect.bisect_left(taken, (start, end))
        if (j < len(taken) and taken[j][0] < end) or (j > 0 and taken[j - 1][1] > start):
            rejected["overlaps_window"] += 1
            continue
        if NON_ACGT.search(fasta.fetch(c, start - pad, end + pad)):
            rejected["non_acgt_near"] += 1
            continue
        taken.insert(j, (start, end))
        windows.append((f"O{rank:06d}", c, start, end, "outside", rank, "."))
        rank += 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    order = {c: i for i, c in enumerate(fasta.order)}
    windows.sort(key=lambda w: (order[w[1]], w[2], w[3]))
    with open(args.out_dir / "windows.tsv", "w") as fh:
        fh.write("window_id\tchrom\tstart\tend\tkind\trank\tsummit\n")
        for w in windows:
            fh.write("\t".join(map(str, w)) + "\n")

    budget_info = {}
    for name, budget in budgets.items():
        chosen: dict[str, list] = collections.defaultdict(list)
        n_kind = collections.Counter()
        for _, c, start, end, kind, r, _ in windows:
            if r < budget[kind]:
                chosen[c].append((start, end))
                n_kind[kind] += 1
        bed_dir = args.out_dir / name
        bed_dir.mkdir(exist_ok=True)
        per_chrom = {}
        with open(bed_dir / "regions.bed", "w") as all_fh:
            for c in sorted(chosen, key=order.get):
                merged = merge(chosen[c])
                with open(bed_dir / f"{c}.bed", "w") as fh:
                    for s, e in merged:
                        fh.write(f"{c}\t{s}\t{e}\n")
                        all_fh.write(f"{c}\t{s}\t{e}\n")
                per_chrom[c] = {"windows": len(chosen[c]), "regions": len(merged), "bp": sum(e - s for s, e in merged)}
        budget_info[name] = {
            "budget": budget,
            "windows": dict(n_kind),
            "regions": sum(v["regions"] for v in per_chrom.values()),
            "bp": sum(v["bp"] for v in per_chrom.values()),
            "bed_dir": str(bed_dir),
            "per_chrom": per_chrom,
        }

    write_json(
        args.out_dir / "regions.json",
        {
            "created_et": started,
            "finished_et": now_et(),
            "argv": sys.argv[1:],
            "genome": str(args.genome),
            "peaks": peak_rows,
            "chroms": chroms,
            "union_summits": {c: len(summits.get(c, [])) for c in chroms},
            "union_summits_all_chroms": sum(len(v) for v in summits.values()),
            "seed": args.seed,
            "rng_streams": {"summit": f"{args.seed}:summit_windows", "outside": f"{args.seed}:outside_windows"},
            "summit_window": f"summit +/- {args.summit_half} bp, closed (BED [s-{args.summit_half}, s+{args.summit_half + 1}))",
            "outside_window": {
                "length": args.outside_len,
                "every_base_more_than_bp_from_any_summit": args.outer,
                "no_non_acgt_within_bp": args.gap_pad,
                "blacklist": str(args.blacklist) if args.blacklist else None,
                "blacklist_slop": args.blacklist_slop,
                "tries": tries,
                "rejected": dict(rejected),
            },
            "windows": {k: sum(1 for w in windows if w[4] == k) for k in KINDS},
            "budgets": budget_info,
        },
    )
    logger.info("%d summit and %d outside windows ranked; budgets: %s", len(picks), rank,
                {n: (b["windows"], b["regions"], b["bp"]) for n, b in budget_info.items()})  # fmt: skip
    return 0


# --------------------------------------------------------------------------- fetch


def find_bcftools(given: str | None) -> str:
    if given:
        return given
    beside = Path(sys.executable).parent / "bcftools"
    if beside.exists():
        return str(beside)
    found = shutil.which("bcftools")
    if not found:
        raise SystemExit("bcftools not found; pass --bcftools")
    return found


def http_head(url: str) -> dict:
    """Size, date and hashes of a remote file, from a HEAD request; errors are recorded, not raised."""
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30) as r:
            h = r.headers
            return {
                "status": r.status,
                "content_length": int(h.get("Content-Length", -1)),
                "last_modified": h.get("Last-Modified"),
                "x_goog_hash": h.get_all("x-goog-hash") or [],
            }
    except Exception as exc:  # noqa: BLE001 - provenance only
        return {"error": repr(exc)}


def node_rx_bytes() -> int | None:
    """Bytes received by this node's non-loopback interfaces (all processes, so only indicative)."""
    try:
        total = 0
        for line in Path("/proc/net/dev").read_text().splitlines()[2:]:
            name, data = line.split(":", 1)
            if name.strip() != "lo":
                total += int(data.split()[0])
        return total
    except OSError:
        return None


class ClientGone(Exception):
    """The local client (htslib) closed its connection: it seeked elsewhere, or finished."""


class UpstreamError(Exception):
    """The origin failed a piece more often than the forwarder retries."""


class RangeForwarder:
    """Serve http://127.0.0.1:<port>/<path> from <origin>/<path> over HTTP/1.1, for htslib.

    Why: htslib 1.24's libcurl (8.21 in the tools environment) negotiates HTTP/2 with storage.googleapis.com,
    and region queries of the gnomAD genome files then fail with "Error in the HTTP2 framing layer". bcftools
    instead talks plain HTTP/1.1 to this forwarder on the loopback interface, and the forwarder fetches each
    requested byte range from the origin over HTTPS with http.client, which speaks only HTTP/1.1.

    htslib asks for `Range: bytes=N-` at every seek and hangs up when it has read what it needs, so every
    origin connection has its receive buffer capped at `recv_buffer` bytes: the data in flight when htslib hangs
    up, which is thrown away, stays below about twice that. A response cut short by the origin is resumed from
    the first byte not yet sent, at most `resumes` times, so no byte reaches htslib twice. Origin connections
    whose response was read to its end are reused.
    """

    def __init__(self, origin: str, timeout: int = 60, resumes: int = 6, recv_buffer: int = 1 << 20,
                 send_buffer: int = 0):  # fmt: skip
        parts = urllib.parse.urlsplit(origin)
        if parts.scheme not in ("https", "http") or not parts.netloc:
            raise ValueError(f"origin must be https://host or http://host, got {origin!r}")
        self.scheme, self.host = parts.scheme, parts.netloc
        self.timeout, self.resumes, self.recv_buffer, self.send_buffer = timeout, resumes, recv_buffer, send_buffer
        self.context = ssl.create_default_context()
        self.pool: queue.LifoQueue = queue.LifoQueue()
        self.heads: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.stats: collections.Counter = collections.Counter()
        forwarder = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            timeout = 600

            def setup(self):
                if forwarder.send_buffer:
                    self.request.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, forwarder.send_buffer)
                super().setup()

            def log_message(self, format, *args):  # noqa: A002 - the base class's signature
                pass

            def do_HEAD(self):
                forwarder.serve(self, body=False)

            def do_GET(self):
                forwarder.serve(self, body=True)

        class Server(http.server.ThreadingHTTPServer):
            daemon_threads = True
            request_queue_size = 128

        self.server = Server(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, name="range-forwarder", daemon=True)
        self.thread.start()

    def settings(self) -> dict:
        return {
            "origin": f"{self.scheme}://{self.host}",
            "recv_buffer": self.recv_buffer,
            "send_buffer": self.send_buffer,
            "resumes": self.resumes,
        }

    def local_url(self, url: str) -> str:
        parts = urllib.parse.urlsplit(url)
        if parts.scheme != self.scheme or parts.netloc != self.host:
            raise ValueError(f"{url} is not {self.scheme}://{self.host}/...")
        return f"http://127.0.0.1:{self.port}{parts.path}"

    def close(self) -> dict:
        """Stop serving; the counters, and the status of every origin path a client asked for."""
        self.server.shutdown()
        self.server.server_close()
        while not self.pool.empty():
            self.pool.get_nowait().close()
        with self.lock:
            return {**self.stats, "paths": {path: info["status"] for path, info in self.heads.items()}}

    def count(self, key: str, n: int = 1) -> None:
        with self.lock:
            self.stats[key] += n

    def connection(self) -> http.client.HTTPConnection:
        try:
            return self.pool.get_nowait()
        except queue.Empty:
            pass
        self.count("upstream_connections")
        if self.scheme == "http":
            conn = http.client.HTTPConnection(self.host, timeout=self.timeout)
        else:
            conn = http.client.HTTPSConnection(self.host, timeout=self.timeout, context=self.context)
        conn.connect()
        conn.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.recv_buffer)
        return conn

    def head(self, path: str) -> dict:
        """Status, size and headers of an origin path, from one HEAD request (cached)."""
        with self.lock:
            if path in self.heads:
                return self.heads[path]
        for attempt in range(self.resumes + 1):
            conn = None
            try:
                conn = self.connection()
                conn.request("HEAD", path)
                resp = conn.getresponse()
                resp.read()
                if resp.status in (429, 500, 502, 503, 504):
                    raise OSError(f"HTTP {resp.status}")
                keep = ("Content-Type", "Last-Modified", "ETag")
                info = {
                    "status": resp.status,
                    "size": int(resp.getheader("Content-Length", "0")),
                    "headers": [(k, resp.getheader(k)) for k in keep if resp.getheader(k)],
                }
                self.pool.put(conn)
                self.count("upstream_heads")
                with self.lock:
                    self.heads[path] = info
                return info
            except (OSError, http.client.HTTPException):
                if conn is not None:
                    conn.close()
                self.count("upstream_retries")
                time.sleep(min(30, 0.5 * 2**attempt))
        raise UpstreamError(f"HEAD {path} failed")

    def serve(self, handler, body: bool) -> None:
        self.count("client_requests")
        handler.close_connection = True
        try:
            info = self.head(handler.path)
        except UpstreamError:
            self.count("upstream_fatal")
            handler.send_error(502)
            return
        if info["status"] != 200:
            handler.send_response(info["status"])
            handler.send_header("Content-Length", "0")
            handler.send_header("Connection", "close")
            handler.end_headers()
            return
        size, status = info["size"], 200
        start, end = 0, info["size"] - 1
        ranged = handler.headers.get("Range")
        if ranged:
            m = re.fullmatch(r"\s*bytes=(\d+)-(\d*)\s*", ranged)
            if not m or int(m.group(1)) >= size:
                handler.send_response(416)
                handler.send_header("Content-Range", f"bytes */{size}")
                handler.send_header("Content-Length", "0")
                handler.send_header("Connection", "close")
                handler.end_headers()
                return
            start, status = int(m.group(1)), 206
            end = min(size - 1, int(m.group(2))) if m.group(2) else size - 1
        handler.send_response(status)
        handler.send_header("Content-Length", str(end - start + 1))
        if status == 206:
            handler.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        handler.send_header("Accept-Ranges", "bytes")
        for key, value in info["headers"]:
            handler.send_header(key, value)
        handler.send_header("Connection", "close")
        handler.end_headers()
        if not body or end < start:
            return
        try:
            self.copy(handler.wfile, handler.path, start, end)
        except ClientGone:
            self.count("client_closed_early")
        except UpstreamError:
            self.count("upstream_fatal")

    def copy(self, wfile, path: str, first: int, last: int) -> None:
        """Copy origin bytes [first, last] to wfile, resuming a cut-short response from where it stopped."""
        pos, failures = first, 0
        while pos <= last:
            conn = None
            try:
                conn = self.connection()
                conn.request("GET", path, headers={"Range": f"bytes={pos}-{last}"})
                resp = conn.getresponse()
                if resp.status != 206:
                    resp.read()
                    raise OSError(f"HTTP {resp.status} for bytes={pos}-{last}")
                self.count("upstream_requests")
                while pos <= last:
                    chunk = resp.read(min(1 << 16, last - pos + 1))
                    if not chunk:
                        raise http.client.IncompleteRead(b"", last - pos + 1)
                    self.count("upstream_bytes", len(chunk))
                    try:
                        wfile.write(chunk)
                    except OSError as exc:
                        raise ClientGone from exc
                    pos += len(chunk)
                    self.count("client_bytes", len(chunk))
                resp.read()
                self.pool.put(conn)
            except ClientGone:
                conn.close()
                raise
            except (OSError, http.client.HTTPException):
                if conn is not None:
                    conn.close()
                failures += 1
                self.count("upstream_retries")
                if failures > self.resumes:
                    raise UpstreamError(f"{path} bytes={pos}-{last}") from None
                if failures > 1:  # the first retry is often only a pooled connection the origin had closed
                    time.sleep(min(30, 0.5 * 2**failures))


def parse_norm_log(text: str) -> dict:
    """norm's summary line and its REF-mismatch warnings, from the last attempt in an appended log."""
    text = text.rsplit(ATTEMPT_MARK, 1)[-1]
    out: dict = {
        "ref_mismatch_warnings": sum(1 for x in text.splitlines() if "mismatch" in x.lower() and "Lines" not in x)
    }
    found = re.findall(r"Lines\s+total/split/joined/realigned/mismatch_removed/dup_removed/skipped:\s+(\S+)", text)
    if found:
        out.update(dict(zip(NORM_FIELDS, (int(v) for v in found[-1].split("/")))))
    return out


def fetch_pipeline(bcftools, url, bed, out_vcf: Path, genome, cwd: Path, log_prefix: Path, filters, keep_info,
                   attempt: int = 1):  # fmt: skip
    """bcftools view | annotate | norm into out_vcf, then index it. Returns norm's summary and the commands."""
    part = out_vcf.with_name(out_vcf.name + ".part")
    view = [bcftools, "view", "--no-version", "-f", filters, "-R", str(bed), "--regions-overlap", "0", "-Ou", url]
    keep = "^" + ",".join(f"INFO/{k}" for k in keep_info)
    annotate = [bcftools, "annotate", "--no-version", "-x", keep, "-Ou"]
    norm = [bcftools, "norm", "--no-version", "-m", "-any", "-f", str(genome), "-c", "w", "-Oz", "-o", str(part)]
    logs = [Path(f"{log_prefix}.{step}.log") for step in ("view", "annotate", "norm")]
    for log in logs:
        with open(log, "a") as fh:
            fh.write(f"{ATTEMPT_MARK} {attempt} {now_et()}\n")
    with open(logs[0], "a") as l1, open(logs[1], "a") as l2, open(logs[2], "a") as l3:
        p1 = subprocess.Popen(view, stdout=subprocess.PIPE, stderr=l1, cwd=cwd)
        p2 = subprocess.Popen(annotate, stdin=p1.stdout, stdout=subprocess.PIPE, stderr=l2, cwd=cwd)
        p1.stdout.close()
        p3 = subprocess.Popen(norm, stdin=p2.stdout, stderr=l3, cwd=cwd)
        p2.stdout.close()
        codes = (p1.wait(), p2.wait(), p3.wait())
    if any(codes):
        raise RuntimeError(f"bcftools exit codes view/annotate/norm {codes}; see {log_prefix}.*.log")
    part.replace(out_vcf)
    subprocess.run([bcftools, "index", "-f", "-t", str(out_vcf)], check=True, cwd=cwd)
    return parse_norm_log(logs[2].read_text()), {"view": view, "annotate": annotate, "norm": norm}


def fetch_index(url: str, cache: Path) -> dict:
    """Download url's .tbi into cache once, before any job starts, so parallel jobs never race to write it.

    htslib looks for a remote file's index under its basename in the working directory first."""
    idx_url = url + ".tbi"
    local = cache / Path(urllib.parse.urlsplit(idx_url).path).name
    if local.exists():
        return {"url": idx_url, "path": str(local), "size": local.stat().st_size, "downloaded": False}
    tmp = local.with_name(local.name + ".part")
    for attempt in range(1, 6):
        try:
            with urllib.request.urlopen(idx_url, timeout=60) as r, open(tmp, "wb") as fh:
                expected = int(r.headers.get("Content-Length", -1))
                shutil.copyfileobj(r, fh)
            if expected >= 0 and tmp.stat().st_size != expected:
                raise OSError(f"{idx_url}: {tmp.stat().st_size} bytes of {expected}")
            tmp.replace(local)
            return {"url": idx_url, "path": str(local), "size": local.stat().st_size, "downloaded": True}
        except OSError as exc:
            logger.warning("index %s attempt %d: %s", idx_url, attempt, exc)
            time.sleep(5 * attempt)
    raise SystemExit(f"cannot download {idx_url}")


def split_bed(bed: Path, chunk: int, out_dir: Path) -> list[Path]:
    """bed cut into files of at most `chunk` regions, <stem>.<k>.bed in out_dir."""
    lines = [x for x in bed.read_text().splitlines(keepends=True) if x.strip()]
    parts = []
    for k in range(0, len(lines), chunk):
        part = out_dir / f"{bed.stem}.{k // chunk:03d}.bed"
        part.write_text("".join(lines[k : k + chunk]))
        parts.append(part)
    return parts


def run_fetch(name, url_template, beds: dict[str, Path], genome, out_dir: Path, bcftools, jobs, retries,
              filters="PASS", keep_info=("AF", "allele_type", "variant_type"), chunk_regions=250,
              transport="forwarder", recv_buffer=1 << 20, send_buffer=0) -> dict:  # fmt: skip
    """Fetch every chromosome's BED from one callset into out_dir, in jobs of at most chunk_regions regions."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("logs", "index_cache", "job_beds"):
        (out_dir / sub).mkdir(exist_ok=True)
    cwd = out_dir / "index_cache"
    started, t0, rx0 = now_et(), time.time(), node_rx_bytes()
    indexes = {}
    for chrom in beds:
        url = url_template.format(chrom=chrom)
        if url.startswith(("http://", "https://")):
            indexes[chrom] = fetch_index(url, cwd)
    forwarder = None
    if transport == "forwarder" and url_template.startswith("https://"):
        parts = urllib.parse.urlsplit(url_template)
        forwarder = RangeForwarder(f"{parts.scheme}://{parts.netloc}", recv_buffer=recv_buffer, send_buffer=send_buffer)
    todo = []
    for chrom, bed in beds.items():
        for part in split_bed(bed, chunk_regions, out_dir / "job_beds"):
            todo.append((chrom, part))

    def one(item) -> dict:
        chrom, bed = item
        rows = [x.split("\t") for x in bed.read_text().splitlines() if x.strip()]
        url = url_template.format(chrom=chrom)
        url_used = forwarder.local_url(url) if forwarder else url
        job = bed.stem
        out_vcf = out_dir / f"{job}.vcf.gz"
        rec = {"job": job, "chrom": chrom, "url": url, "url_used": url_used, "bed": str(bed), "regions": len(rows),
               "bp": sum(int(r[2]) - int(r[1]) for r in rows), "vcf": str(out_vcf)}  # fmt: skip
        for attempt in range(1, retries + 1):
            rec["attempts"] = attempt
            rec["started_et"], t = now_et(), time.time()
            try:
                rec["norm"], rec["commands"] = fetch_pipeline(bcftools, url_used, bed, out_vcf, genome, cwd,
                                                              out_dir / "logs" / job, filters, keep_info, attempt)  # fmt: skip
                rec["seconds"] = round(time.time() - t, 1)
                rec["finished_et"] = now_et()
                rec["size"] = out_vcf.stat().st_size
                logger.info("%s %s: %d regions, %d records, %.0f s", name, job, len(rows),
                            rec["norm"].get("total", -1), rec["seconds"])  # fmt: skip
                return rec
            except Exception as exc:  # noqa: BLE001 - retried, then reported
                rec.setdefault("errors", []).append(repr(exc))
                logger.warning("%s %s attempt %d failed: %s", name, job, attempt, exc)
                time.sleep(10 * attempt)
        rec["failed"] = True
        return rec

    try:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(one, sorted(todo, key=lambda x: -x[1].stat().st_size)))
    finally:
        forwarder_stats = forwarder.close() if forwarder else None
    results.sort(key=lambda r: r["job"])
    heads = {}
    for chrom in beds:
        url = url_template.format(chrom=chrom)
        if url.startswith(("http://", "https://")):
            heads[url] = http_head(url)
            heads[url + ".tbi"] = http_head(url + ".tbi")
    rx1 = node_rx_bytes()
    version = subprocess.run([bcftools, "--version"], capture_output=True, text=True).stdout.splitlines()[:2]
    per_chrom: dict = {}
    for r in results:
        agg = per_chrom.setdefault(r["chrom"], {"jobs": 0, "regions": 0, "bp": 0, "records": 0, "job_seconds": 0.0})
        agg["jobs"] += 1
        agg["regions"] += r["regions"]
        agg["bp"] += r["bp"]
        agg["records"] += r.get("norm", {}).get("total", 0)
        agg["job_seconds"] = round(agg["job_seconds"] + r.get("seconds", 0.0), 1)
    return {
        "name": name,
        "url_template": url_template,
        "transport": (
            "bcftools reads http://127.0.0.1 from an in-process forwarder that fetches bounded HTTPS (HTTP/1.1) "
            "ranges from the origin: htslib 1.24 + libcurl over HTTP/2 failed on these files with 'Error in the "
            "HTTP2 framing layer'"
            if forwarder
            else "direct"
        ),
        "transport_mode": transport,
        "forwarder": forwarder_stats,
        "forwarder_settings": forwarder.settings() if forwarder else None,
        "genome": str(genome),
        "beds": {c: str(p) for c, p in beds.items()},
        "chunk_regions": chunk_regions,
        "bcftools": bcftools,
        "bcftools_version": version,
        "filters": filters,
        "keep_info": list(keep_info),
        "regions_overlap": "0: POS in a region",
        "started_et": started,
        "finished_et": now_et(),
        "seconds": round(time.time() - t0, 1),
        "jobs": jobs,
        "out_bytes": sum(r.get("size", 0) for r in results),
        "node_rx_bytes_delta_indicative": (rx1 - rx0) if rx0 is not None and rx1 is not None else None,
        "failed": [r["job"] for r in results if r.get("failed")],
        "per_chrom": per_chrom,
        "per_job": results,
        "indexes": indexes,
        "remote_files": heads,
    }


def cmd_fetch(args) -> int:
    beds = {p.stem: p for p in sorted(args.bed_dir.glob("chr*.bed")) if p.stat().st_size > 0}
    if args.chroms:
        wanted = set(parse_chroms(args.chroms))
        beds = {c: p for c, p in beds.items() if c in wanted}
    if not beds:
        logger.error("no non-empty <chrom>.bed in %s", args.bed_dir)
        return 1
    order = {c: i for i, c in enumerate(Fasta(args.genome).order)}
    beds = dict(sorted(beds.items(), key=lambda kv: order[kv[0]]))
    record = run_fetch(args.name, args.url_template, beds, args.genome, args.out_dir, find_bcftools(args.bcftools),
                       args.jobs, args.retries, args.filters, tuple(parse_chroms(args.keep_info)),
                       args.chunk_regions, args.transport, args.recv_buffer_kb << 10,
                       args.send_buffer_kb << 10)  # fmt: skip
    record["argv"] = sys.argv[1:]
    record["bed_dir"] = str(args.bed_dir)
    write_json(args.out_dir / "fetch.json", record)
    logger.info("%s: %d jobs over %d chromosomes in %.0f s, %d bytes written, forwarder %s, failed: %s", args.name,
                len(record["per_job"]), len(beds), record["seconds"], record["out_bytes"], record["forwarder"],
                record["failed"])  # fmt: skip
    return 1 if record["failed"] else 0


# --------------------------------------------------------------------------- sample


def read_vcf(path: Path):
    """(chrom, pos, id, ref, alt, filter, info dict) per record of a sites VCF."""
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            info = {}
            for item in f[7].split(";"):
                key, _, value = item.partition("=")
                info[key] = value
            yield f[0], int(f[1]), f[2], f[3], f[4], f[6], info


def variant_type(ref: str, alt: str) -> str:
    if len(ref) == len(alt) == 1:
        return "snv"
    return "insertion" if len(alt) > len(ref) else "deletion"


class Checker:
    """The per-record checks of `sample`, against one FASTA."""

    def __init__(self, fasta: Fasta, chroms: list[str], input_len: int):
        self.fasta = fasta
        self.chroms = set(chroms)
        self.flank = input_len // 2
        self.input_len = input_len

    def reason(self, chrom: str, pos: int, ref: str, alt: str) -> str | None:
        """Why a record is dropped, or None to keep it."""
        if chrom not in self.chroms:
            return "other_chrom"
        if "," in alt:
            return "multiallelic"
        if NON_ACGT.search(ref) or NON_ACGT.search(alt) or not ref or not alt:
            return "non_acgt_allele"
        if len(ref) == len(alt) and len(ref) > 1:
            return "mnv"
        pos0, extra = pos - 1, max(0, len(ref) - len(alt))
        length = self.fasta.length(chrom)
        # VS's get_valid_variants (pos - flank > 0 and pos + flank <= length; 1 bp stricter on the right than the
        # window needs), as build_windows.py applies it, plus the extra right-flank bases a deletion reads.
        if (
            pos0 - self.flank < 0
            or pos + self.flank > length
            or pos0 + self.flank + extra > length
            or max(len(ref), len(alt)) > self.flank
        ):
            return "window_off_chrom"
        if self.fasta.fetch(chrom, pos0, pos0 + len(ref)) != ref.upper():
            return "ref_mismatch"
        if NON_ACGT.search(self.fasta.fetch(chrom, pos0 - self.flank, pos0 + self.flank + extra)):
            return "window_non_acgt"
        return None


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while block := fh.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def load_fetch(fetch_dir: Path) -> dict:
    record = json.loads((fetch_dir / "fetch.json").read_text())
    if record.get("failed"):
        raise SystemExit(f"{fetch_dir}: fetch failed for {record['failed']}")
    return record


def covered_sets(fetch: dict) -> dict[str, IntervalSet]:
    """Per chromosome, the union of the regions the jobs of one fetch queried."""
    per: dict[str, list] = collections.defaultdict(list)
    for job in fetch["per_job"]:
        for c, iv in read_bed(job["bed"]).items():
            per[c].extend(zip(iv.starts, iv.ends))
    return {c: IntervalSet(v) for c, v in per.items()}


def cluster_positions(positions, gap: int) -> list[tuple[int, int]]:
    """[start, end) regions covering 0-based positions; a position less than `gap` bp past a region joins it."""
    out: list[list[int]] = []
    for p in sorted(positions):
        if out and p - out[-1][1] < gap:
            out[-1][1] = p + 1
        else:
            out.append([p, p + 1])
    return [(a, b) for a, b in out]


def cmd_sample(args) -> int:
    started = now_et()
    fasta = Fasta(args.genome)
    chroms = parse_chroms(args.chroms)
    order = {c: i for i, c in enumerate(fasta.order)}
    checker = Checker(fasta, chroms, args.input_len)
    summits, peak_rows = read_summits(args.peaks)
    names = strata_names(args.inner, args.outer)
    test_fold = {}
    for path in sorted(args.folds_dir.glob("fold_*.json")):
        fold = path.stem.split("_", 1)[1]
        for c in json.loads(path.read_text())["test"]:
            test_fold[c] = fold
    no_fold = [c for c in chroms if c not in test_fold]
    if no_fold:
        logger.error("no fold holds out %s", no_fold)
        return 1

    callsets = [c for c, _ in args.callset]
    fetches = {c: load_fetch(d) for c, d in args.callset}
    fetch_dirs = dict(args.callset)
    covered = {c: covered_sets(f) for c, f in fetches.items()}

    # Every PASS record fetched, keyed (chrom, pos, ref, alt): membership. Kept ones also go to the pool.
    fetched: dict[str, dict] = {}
    pools: dict[str, dict] = {}
    drops = {c: collections.Counter() for c in callsets}
    mismatch_examples = {c: [] for c in callsets}
    norm_totals = {c: collections.Counter() for c in callsets}
    for c in callsets:
        seen: dict = {}
        pool: dict = {}
        for job in fetches[c]["per_job"]:
            norm_totals[c].update({k: v for k, v in job["norm"].items() if isinstance(v, int)})
            for chrom, pos, vid, ref, alt, filt, info in read_vcf(Path(job["vcf"])):
                if filt != "PASS":
                    drops[c]["not_pass"] += 1
                    continue
                key = (chrom, pos, ref, alt)
                if key in seen:
                    drops[c]["duplicate"] += 1
                    continue
                seen[key] = (info.get("AF", "NA"), info.get("allele_type", "NA"), vid)
                why = checker.reason(chrom, pos, ref, alt)
                if why:
                    drops[c][why] += 1
                    if why == "ref_mismatch" and len(mismatch_examples[c]) < 20:
                        mismatch_examples[c].append(f"{chrom}:{pos}:{ref}:{alt}")
                    continue
                d = nearest_distance(summits.get(chrom, []), pos - 1)
                d = 10**12 if d is None else d
                region = (chrom, covered[c][chrom].find(pos - 1))  # the cap's unit: one merged region
                pool[key] = (stratum_of(d, args.inner, args.outer), d, region)
        fetched[c], pools[c] = seen, pool
        logger.info("%s: %d records, %d kept, drops %s", c, len(seen) + drops[c]["duplicate"], len(pool),
                    dict(drops[c]))  # fmt: skip

    # Draws.
    quotas = [round(args.target * f) for f in args.fractions]
    selected: dict[tuple, str] = {}
    draws = {}

    def sort_key(k):
        return (order[k[0]], k[1], k[2], k[3])

    for c in callsets:
        draws[c] = {}
        for si, sname in enumerate(names):
            candidates = sorted((k for k, v in pools[c].items() if v[0] == si and k not in selected), key=sort_key)
            in_pool = sum(1 for v in pools[c].values() if v[0] == si)
            per_region = collections.Counter(pools[c][k][2] for k in candidates)
            capped = sum(min(n, args.cap_per_region) for n in per_region.values())
            rng = random.Random(f"{args.seed}:{c}:{sname}")
            rng.shuffle(candidates)
            taken: collections.Counter = collections.Counter()
            n = 0
            for k in candidates:
                if n == quotas[si]:
                    break
                region = pools[c][k][2]
                if taken[region] >= args.cap_per_region:
                    continue
                taken[region] += 1
                selected[k] = c
                n += 1
            draws[c][sname] = {
                "quota": quotas[si],
                "drawn": n,
                "shortfall": quotas[si] - n,
                "pool": in_pool,
                "pool_not_yet_drawn": len(candidates),
                "pool_regions": len(per_region),
                "pool_within_cap": capped,
                "regions_drawn": len(taken),
                "rng": f"{args.seed}:{c}:{sname}",
            }
            logger.info("%s %s: drew %d of %d (pool %d in %d regions, %d within the cap)", c, sname, n, quotas[si],
                        len(candidates), len(per_region), capped)  # fmt: skip

    # Membership of the drawn variants in every callset; query positions a callset's fetch did not cover.
    completion = {}
    for c in callsets:
        todo: dict[str, set] = collections.defaultdict(set)
        for chrom, pos, _ref, _alt in selected:
            iv = covered[c].get(chrom)
            if iv is None or iv.find(pos - 1) < 0:
                todo[chrom].add(pos - 1)
        completion[c] = {"positions": sum(len(v) for v in todo.values())}
        if not todo:
            continue
        out_dir = fetch_dirs[c] / "complete"
        bed_dir = out_dir / "regions"
        bed_dir.mkdir(parents=True, exist_ok=True)
        beds = {}
        for chrom in sorted(todo, key=order.get):
            beds[chrom] = bed_dir / f"{chrom}.bed"
            with open(beds[chrom], "w") as fh:
                for s, e in cluster_positions(todo[chrom], args.complete_gap):
                    fh.write(f"{chrom}\t{s}\t{e}\n")
        rec = run_fetch(f"{c}-complete", fetches[c]["url_template"], beds, args.genome, out_dir,
                        args.bcftools or fetches[c]["bcftools"], args.jobs, args.retries,
                        fetches[c]["filters"], tuple(fetches[c]["keep_info"]), args.chunk_regions,
                        fetches[c].get("transport_mode", "forwarder"),
                        (fetches[c].get("forwarder_settings") or {}).get("recv_buffer", 1 << 20),
                        (fetches[c].get("forwarder_settings") or {}).get("send_buffer", 0))  # fmt: skip
        write_json(out_dir / "fetch.json", rec)
        if rec["failed"]:
            logger.error("%s completion failed for %s", c, rec["failed"])
            return 1
        found = 0
        for job in rec["per_job"]:
            for chrom, pos, vid, ref, alt, filt, info in read_vcf(Path(job["vcf"])):
                key = (chrom, pos, ref, alt)
                if filt == "PASS" and key not in fetched[c]:
                    fetched[c][key] = (info.get("AF", "NA"), info.get("allele_type", "NA"), vid)
                    found += key in selected
        completion[c].update(
            {
                "regions": sum(r["regions"] for r in rec["per_job"]),
                "gap": args.complete_gap,
                "found_drawn_variants": found,
                "seconds": rec["seconds"],
                "forwarder": rec["forwarder"],
                "fetch_json": str(out_dir / "fetch.json"),
            }
        )

    # Write the list, the side table and the meta.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stem = args.out.name[: -len(".tsv")] if args.out.name.endswith(".tsv") else args.out.name
    side_path = args.out.with_name(f"{stem}.callsets.tsv")
    meta_path = args.out.with_name(f"{stem}.meta.json")
    priority = parse_chroms(args.af_priority) if args.af_priority else list(reversed(callsets))
    counts: collections.Counter = collections.Counter()
    per_chrom: collections.Counter = collections.Counter()
    per_fold: collections.Counter = collections.Counter()
    per_draw: collections.Counter = collections.Counter()
    rows = []
    for key in sorted(selected, key=sort_key):
        chrom, pos, ref, alt = key
        draw = selected[key]
        members = [c for c in callsets if key in fetched[c]]
        source = "both" if len(members) == len(callsets) == 2 else "+".join(members)
        af = next((fetched[c][key][0] for c in priority if key in fetched[c]), "NA")
        si, d, region = pools[draw][key]
        vtype = variant_type(ref, alt)
        rows.append((f"{chrom}:{pos}:{ref}:{alt}", chrom, pos, ref, alt, source, names[si], af, d, test_fold[chrom]))
        counts[(names[si], source, "snv" if vtype == "snv" else "indel")] += 1
        per_chrom[chrom] += 1
        per_fold[test_fold[chrom]] += 1
        per_draw[(draw, names[si], source)] += 1
    tmp = args.out.with_name(args.out.name + ".tmp")
    with open(tmp, "w") as fh:
        fh.write("\t".join(LIST_COLUMNS) + "\n")
        for row in rows:
            fh.write("\t".join(map(str, row)) + "\n")
    tmp.replace(args.out)
    with open(side_path, "w") as fh:
        fh.write("\t".join(["variant_id", "draw", "allele_type", "vcf_id"] + [f"in_{c}" for c in callsets]
                           + [f"af_{c}" for c in callsets]) + "\n")  # fmt: skip
        for key in sorted(selected, key=sort_key):
            draw = selected[key]
            af_type, vid = fetched[draw][key][1], fetched[draw][key][2]
            fh.write("\t".join([":".join(map(str, key)), draw, af_type, vid]
                               + ["1" if key in fetched[c] else "0" for c in callsets]
                               + [fetched[c][key][0] if key in fetched[c] else "NA" for c in callsets]) + "\n")  # fmt: skip

    in_peak = sum(1 for r in rows if r[6] != names[2])
    sources = sorted({r[5] for r in rows})
    norm_changed = {
        c: {k: norm_totals[c].get(k, 0) for k in ("split", "joined", "realigned", "ref_mismatch_warnings")}
        for c in callsets
    }
    table = {s: {src: {t: counts[(s, src, t)] for t in ("snv", "indel")} for src in sources} for s in names}
    meta = {
        "created_et": started,
        "finished_et": now_et(),
        "argv": sys.argv[1:],
        "release": args.release,
        "notes": dict(args.note),
        "outputs": {"list": str(args.out), "callsets": str(side_path), "meta": str(meta_path)},
        "output_files": {
            "list": {"size": args.out.stat().st_size, "md5": md5_file(args.out)},
            "callsets": {"size": side_path.stat().st_size, "md5": md5_file(side_path)},
        },
        "list_columns": list(LIST_COLUMNS),
        "n_variants": len(rows),
        "counts_stratum_source_type": table,
        "counts_by_type": {t: sum(v for (_, _, tt), v in counts.items() if tt == t) for t in ("snv", "indel")},
        "counts_by_source": {src: sum(v for (_, s2, _), v in counts.items() if s2 == src) for src in sources},
        "counts_by_stratum": {s: sum(v for (s1, _, _), v in counts.items() if s1 == s) for s in names},
        "counts_by_draw": {c: {s: {src: per_draw[(c, s, src)] for src in sources} for s in names} for c in callsets},
        "counts_by_chrom": {c: per_chrom[c] for c in chroms},
        "counts_by_fold_test": dict(sorted(per_fold.items())),
        "in_peak_rows": in_peak,
        "min_in_peak_rows": args.min_in_peak,
        "in_peak_rows_ok": in_peak >= args.min_in_peak,
        "draws": draws,
        "drops": {c: {r: drops[c][r] for r in DROP_REASONS} for c in callsets},
        "ref_mismatch_examples": mismatch_examples,
        "bcftools_norm_totals": {c: dict(norm_totals[c]) for c in callsets},
        "bcftools_norm_was_only_a_check": {c: not any(v.values()) for c, v in norm_changed.items()},
        "records_fetched": {c: len(fetched[c]) for c in callsets},
        "pool_kept": {c: len(pools[c]) for c in callsets},
        "membership_completion": completion,
        "filters": [
            "FILTER == PASS (bcftools view -f PASS; checked again when reading)",
            "POS inside a fetched region (bcftools view -R with --regions-overlap 0)",
            "INFO reduced to " + ",".join(fetches[callsets[0]]["keep_info"]) + " (bcftools annotate -x ^...)",
            "multiallelic sites split and indels left-aligned against the FASTA (bcftools norm -m -any -f; -c w)",
            "chromosome in --chroms; REF and ALT ACGT only (drops '*' and N); SNVs and indels only (MNVs dropped)",
            "the model input window inside the chromosome; REF equal to the FASTA; the window ACGT only",
            "first copy of a duplicate chrom:pos:ref:alt kept",
        ],
        "definitions": {
            "variant_id": "chrom:pos:ref:alt, pos 1-based (VCF), anchored VCF alleles after bcftools norm",
            "anchor": "dist_to_summit is |(pos - 1) - s| for the nearest union summit s (0-based start + column 10)",
            "strata": {
                names[0]: f"dist_to_summit <= {args.inner}",
                names[1]: f"{args.inner} < dist_to_summit <= {args.outer}",
                names[2]: f"dist_to_summit > {args.outer}, i.e. outside every summit +/- {args.outer} window",
            },
            "input_window_note": (
                f"an {names[2]} variant is outside every peak window, but its {args.input_len}-bp model input "
                f"still overlaps a peak window when dist_to_summit < {args.input_len // 2 + args.outer}"
            ),
            "source": "the callsets holding the variant as a PASS record after norm: one name, or 'both'",
            "af": f"INFO/AF of the first callset in {priority} that holds the variant",
            "fold_test": "the fold whose test chromosomes include chrom (fold_*.json)",
            "checks": list(DROP_REASONS),
            "window": f"[pos-1-{args.input_len // 2}, pos-1+{args.input_len // 2}+max(0, len(ref)-len(alt)))",
        },
        "params": {
            "seed": args.seed,
            "target_per_callset": args.target,
            "fractions": args.fractions,
            "quotas": dict(zip(names, quotas)),
            "cap_per_region": args.cap_per_region,
            "inner": args.inner,
            "outer": args.outer,
            "input_len": args.input_len,
            "chroms": chroms,
            "draw_order": callsets,
        },
        "peaks": peak_rows,
        "union_summits": sum(len(v) for v in summits.values()),
        "genome": str(args.genome),
        "folds_dir": str(args.folds_dir),
        "fetches": {
            c: {
                "fetch_json": str(fetch_dirs[c] / "fetch.json"),
                "bed_dir": fetches[c].get("bed_dir"),
                "regions_json": str(Path(fetches[c]["bed_dir"]).parent / "regions.json")
                if fetches[c].get("bed_dir")
                else None,
                "url_template": fetches[c]["url_template"],
                "filters": fetches[c]["filters"],
                "keep_info": fetches[c]["keep_info"],
                "bcftools_version": fetches[c]["bcftools_version"],
                "started_et": fetches[c]["started_et"],
                "finished_et": fetches[c]["finished_et"],
                "seconds": fetches[c]["seconds"],
                "out_bytes": fetches[c]["out_bytes"],
                "regions": sum(r["regions"] for r in fetches[c]["per_job"]),
                "bp": sum(r["bp"] for r in fetches[c]["per_job"]),
                "jobs": len(fetches[c]["per_job"]),
                "transport": fetches[c].get("transport"),
                "forwarder_settings": fetches[c].get("forwarder_settings"),
                "forwarder": fetches[c].get("forwarder"),
                "node_rx_bytes_delta_indicative": fetches[c].get("node_rx_bytes_delta_indicative"),
                "indexes": fetches[c].get("indexes"),
                "remote_files": fetches[c].get("remote_files"),
            }
            for c in callsets
        },
    }
    write_json(meta_path, meta)
    logger.info("wrote %s: %d variants %s; in-peak rows %d", args.out, len(rows), meta["counts_by_stratum"], in_peak)
    if in_peak < args.min_in_peak:
        logger.warning("only %d in-peak rows, fewer than --min-in-peak %d", in_peak, args.min_in_peak)
    return 0


# --------------------------------------------------------------------------- main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="command", required=True)

    r = sub.add_parser("regions", help="rank summit and outside windows; write per-budget BEDs")
    r.add_argument("--peaks", nargs="+", type=Path, required=True, help="narrowPeak files; their summits' union")
    r.add_argument("--genome", type=Path, required=True, help="uncompressed FASTA with .fai")
    r.add_argument("--blacklist", type=Path, default=None, help="BED(.gz) the outside windows must avoid")
    r.add_argument("--blacklist-slop", type=int, default=1057, help="bp added to each blacklist side (default 1057)")
    r.add_argument("--chroms", default=DEFAULT_CHROMS, help="comma-separated (default chr1-22,chrX)")
    r.add_argument("--n-summit", type=int, required=True, help="summit windows to rank")
    r.add_argument("--n-outside", type=int, required=True, help="outside windows to rank")
    r.add_argument("--summit-half", type=int, default=500, help="summit window half-width (default 500)")
    r.add_argument("--outside-len", type=int, default=1000, help="outside window length (default 1000)")
    r.add_argument("--outer", type=int, default=500, help="outside windows stay more than this from summits")
    r.add_argument("--gap-pad", type=int, default=5000, help="no non-ACGT base within this of an outside window")
    r.add_argument("--max-tries", type=int, default=10_000_000)
    r.add_argument("--seed", type=int, default=1234)
    r.add_argument("--budget", type=parse_budget, action="append", default=[], metavar="NAME:summit=K,outside=K")
    r.add_argument("--out-dir", type=Path, required=True)
    r.set_defaults(func=cmd_regions)

    f = sub.add_parser("fetch", help="query one callset over one budget's BEDs with bcftools")
    f.add_argument("--name", required=True)
    f.add_argument("--url-template", required=True, help="VCF URL or path with a {chrom} field")
    f.add_argument("--bed-dir", type=Path, required=True, help="<chrom>.bed files from `regions`")
    f.add_argument("--genome", type=Path, required=True, help="FASTA for bcftools norm (contig names must match)")
    f.add_argument("--out-dir", type=Path, required=True, help="a new, empty directory")
    f.add_argument("--chroms", default=None, help="only these chromosomes (comma-separated)")
    f.add_argument("--bcftools", default=None, help="default: beside this python, else on PATH")
    f.add_argument("--jobs", type=int, default=4)
    f.add_argument("--retries", type=int, default=3)
    f.add_argument("--filters", default="PASS", help="bcftools view -f (default PASS)")
    f.add_argument("--keep-info", default="AF,allele_type,variant_type", help="INFO fields to keep")
    f.add_argument("--chunk-regions", type=int, default=250, help="regions per bcftools job (default 250)")
    f.add_argument("--recv-buffer-kb", type=int, default=1024,
                   help="receive buffer of each forwarder connection to the origin, in KiB (default 1024)")  # fmt: skip
    f.add_argument("--send-buffer-kb", type=int, default=0,
                   help="send buffer of each forwarder connection to bcftools, in KiB (default 0: the kernel's)")  # fmt: skip
    f.add_argument("--transport", choices=("forwarder", "direct"), default="forwarder",
                   help="https URLs through the local HTTP/1.1 forwarder (default) or straight to htslib")  # fmt: skip
    f.set_defaults(func=cmd_fetch)

    s = sub.add_parser("sample", help="check, classify and draw the list")
    s.add_argument("--callset", action="append", required=True, metavar="NAME=FETCH_DIR",
                   type=lambda x: (x.partition("=")[0], Path(x.partition("=")[2])))  # fmt: skip
    s.add_argument("--peaks", nargs="+", type=Path, required=True, help="the narrowPeak files given to `regions`")
    s.add_argument("--genome", type=Path, required=True)
    s.add_argument("--folds-dir", type=Path, required=True, help="fold_<k>.json files with a `test` list")
    s.add_argument("--chroms", default=DEFAULT_CHROMS)
    s.add_argument("--target", type=int, default=10000, help="variants to draw per callset (default 10000)")
    s.add_argument("--fractions", type=parse_fractions, default=[0.6, 0.2, 0.2], help="per stratum (0.6,0.2,0.2)")
    s.add_argument("--inner", type=int, default=100)
    s.add_argument("--outer", type=int, default=500)
    s.add_argument("--input-len", type=int, default=2114)
    s.add_argument("--cap-per-region", type=int, default=10, help="max draws from one merged fetch region")
    s.add_argument(
        "--af-priority", default=None, help="callsets to take AF from, in order (default: reverse draw order)"
    )
    s.add_argument("--min-in-peak", type=int, default=0, help="warn if fewer non-outside rows")
    s.add_argument("--release", default=None, help="free text naming the callsets' release, for the meta")
    s.add_argument("--note", action="append", default=[], metavar="KEY=TEXT",
                   type=lambda x: tuple(x.split("=", 1)) if "=" in x else (x, ""),
                   help="free-text notes copied into the meta")  # fmt: skip
    s.add_argument("--seed", type=int, default=1234)
    s.add_argument("--bcftools", default=None)
    s.add_argument("--jobs", type=int, default=4)
    s.add_argument("--retries", type=int, default=3)
    s.add_argument("--chunk-regions", type=int, default=250, help="regions per completion job (default 250)")
    s.add_argument("--complete-gap", type=int, default=1000,
                   help="completion queries join positions closer than this (default 1000)")  # fmt: skip
    s.add_argument("--out", type=Path, required=True, help="<list>.tsv; .callsets.tsv and .meta.json go beside it")
    s.set_defaults(func=cmd_sample)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%F %T")  # fmt: skip
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
