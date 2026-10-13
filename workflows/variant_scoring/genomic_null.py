#!/usr/bin/env python3
"""A genomic null for variant scoring: random SNVs at real genomic positions away from peaks, GC-matched.

The shuffled null of build_windows.py --null dinucleotide-shuffles the windows of in-peak variants. That keeps each
window's composition but not its order, so the models also score sequence unlike any they were trained on. This null
keeps the order real: every row is a single-base substitution at the centre of a reference window that touches no
peak, and the windows are matched in GC content to the shuffled null's source windows. The two nulls then differ only
in whether the sequence order is real. The output is a variant list; build_windows.py builds its windows like any
other list's (variant mode: a1 and a2).

Windows
  window_len = 2 * --flank (2114 bp by default, the models' input). A row at 1-based pos has pos0 = pos - 1 and the
  window [pos0 - flank, pos0 + flank). The substituted base sits at window index flank (output position 500 of both
  models), where build_windows.py places every variant.

Tiles
  One sequential pass reads the FASTA, one chromosome in memory at a time, upper-cased (soft-masked bases count).
  Each chromosome of --chroms is cut into non-overlapping tiles [t, t + window_len), t = offset + k * window_len, from
  offset = random.Random("<seed>:offset:<chrom>").randrange(window_len). Every tile is one candidate window. These
  filters apply in this order, and the meta counts the tiles left after each, in total and per chromosome:
    all              every tile that fits in the chromosome;
    acgt             no base other than A, C, G or T;
    not_in_peak      overlaps no interval (columns 1-3) of any --peaks file. With peaks of summit +/- 500 bp, every
                     row has dist_to_summit >= flank + 500 (1557 by default);
    not_blacklisted  overlaps no --blacklist interval (no slop);
    in_bounds        VS's get_valid_variants bound, pos - flank > 0 and pos + flank <= length (kundajelab/
                     variant-scorer@0e1e341 src/utils/helpers.py:28-42), which build_windows.py applies too. It is
                     1 bp stricter on the right than the window, so a tile that ends at the chromosome's end is
                     dropped; build_windows.py would drop its row and the null would shrink.
  Tiles do not overlap, so no two rows share a window base.

GC target
  The GC fraction of the window [pos0 - flank, pos0 + flank) of every --target-list row in --target-strata (default
  summit100,flank500: the rows build_windows.py --null shuffles; a dinucleotide shuffle keeps composition), read in
  the same pass. A target window holding a non-ACGT base is skipped and counted, as the shuffled null's pool skips it.
  Bins have width --gc-bin: bin k holds GC fractions in [k * w, (k + 1) * w), the last one also 1, computed in
  integers as k = min(nbins - 1, gc_count * nbins // window_len). The target histogram is scaled to --n rows by
  largest remainder (ties to the lower bin).

Draw
  Each bin's eligible tiles, in genome order, are permuted with random.Random("<seed>:bin:<k>"). Every bin first takes
  its own tiles, up to its target (taken). A bin left short then borrows the surplus of other bins, nearest first: at
  distance 1 every short bin, in ascending order, takes from the bin below and then from the bin above; then the same
  at distance 2, and so on (borrowed, and the lender's lent). Peaks are GC-rich and the rest of the genome is not, so
  the top bins run short; the meta records by how much and from which bins. Fewer eligible tiles than --n is an error.
  A drawn tile at t gives pos = t + flank + 1 and ref = the FASTA base at pos0 (upper case); alt is one of the other
  three bases, drawn uniformly with random.Random("<seed>:alt"), one draw per row in output order.

Outputs, for --out <name>.tsv
  <name>.tsv        the variant-list columns (gnomad_list.py), one row per drawn tile, in .fai order and then pos:
                      variant_id      chrom:pos:ref:alt
                      chrom, pos      pos 1-based
                      ref, alt        one base each
                      source, stratum genomic_null
                      af              empty
                      dist_to_summit  |pos0 - s| for the nearest summit s (start + column 10) of any --peaks file;
                                      10**12 on a chromosome without one, as gnomad_list.py writes it
                      fold_test       the fold whose test set holds chrom (the manifest's test_chrom_to_fold)
  <name>.meta.json  argv; the inputs (path, size, mtime ET; md5 except for the FASTA); the seed, the RNG streams and
                    the per-chromosome offsets; the tile counts per filter; the per-bin table (target, available,
                    taken, shortfall, borrowed and from which bins, lent, achieved); the mean and quantiles (q01 to
                    q99) of the GC fraction of the target windows, the eligible tiles and the drawn rows, and their KS
                    distances to the target; REF and ALT base counts; rows per chromosome and per fold; the run time,
                    peak RSS and created and finished times (ET).

    python -I workflows/variant_scoring/genomic_null.py --fasta genome.fa --peaks a.narrowPeak b.narrowPeak \\
        --blacklist blacklist.bed.gz --target-list lists/list.tsv --manifest manifest.json --n 10000 --seed 1234 \\
        --out nulls/genomic_null.tsv
    python workflows/variant_scoring/build_windows.py --list nulls/genomic_null.tsv --fasta genome.fa \\
        --out windows/genomic_null.npz

Keep the list out of the pilot's lists/ directory: pilot_steps.sh takes the one TSV there as the variant list.

Python standard library only (with gnomad_list.py's helpers); it never loads numpy. Memory: the longest chromosome
once (about 250 MB for hg38's chr1), the peak intervals, and a few tens of bytes per eligible tile (its arrays and the
draw's per-bin indices: about 35 MB per million tiles).
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import json
import logging
import random
import resource
import statistics
import subprocess
import sys
import time
from array import array
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gnomad_list import (  # noqa: E402
    DEFAULT_CHROMS,
    ET,
    LIST_COLUMNS,
    Fasta,
    IntervalSet,
    md5_file,
    nearest_distance,
    now_et,
    open_text,
    parse_chroms,
    read_bed,
    read_summits,
    write_json,
)

logger = logging.getLogger("genomic_null")

NAME = "genomic_null"  # source and stratum of every row
FILTERS = ("all", "acgt", "not_in_peak", "not_blacklisted", "in_bounds")
QUANTILES = {"q01": 1, "q10": 10, "q25": 25, "q50": 50, "q75": 75, "q90": 90, "q99": 99}
NO_SUMMIT = 10**12  # dist_to_summit on a chromosome without a summit (gnomad_list.py's value)
UPPER = bytes(range(256)).upper()  # a bytes.translate table: a-z to A-Z, every other byte unchanged
BASES = "ACGT"


def tile_offset(seed: int, chrom: str, window_len: int) -> int:
    """Where chrom's first tile starts: a draw in [0, window_len) from the chromosome's own stream."""
    return random.Random(f"{seed}:offset:{chrom}").randrange(window_len)


def bin_count(gc_bin: float) -> int:
    """The number of GC bins of width gc_bin, which must divide 1 into whole bins."""
    if not 0 < gc_bin <= 1:
        raise ValueError(f"--gc-bin {gc_bin} is not in (0, 1]")
    n = round(1 / gc_bin)
    if abs(n * gc_bin - 1) > 1e-9:
        raise ValueError(f"--gc-bin {gc_bin} does not divide 1 into whole bins")
    return n


def gc_bin_of(gc: int, window_len: int, nbins: int) -> int:
    """The bin of a window with gc G or C bases: fraction in [k / nbins, (k + 1) / nbins), the last bin closed."""
    return min(nbins - 1, gc * nbins // window_len)


def largest_remainder(counts: list[int], n: int) -> list[int]:
    """counts scaled to sum to n: the floors, plus one for each of the largest remainders (ties to the lower index)."""
    total = sum(counts)
    scaled = [c * n // total for c in counts]
    order = sorted(range(len(counts)), key=lambda k: (-(counts[k] * n % total), k))
    for k in order[: n - sum(scaled)]:
        scaled[k] += 1
    return scaled


def read_chrom(fasta: Fasta, chrom: str, block: int = 1 << 24) -> bytearray:
    """chrom's whole sequence, upper-cased, read in blocks straight into one bytearray (no second full copy)."""
    length, offset, line_bases, line_width = fasta.index[chrom]
    seq = bytearray(length)
    if length == 0:
        return seq
    remaining = (length // line_bases) * line_width + length % line_bases
    step = max(line_width, block - block % line_width)
    fasta.fh.seek(offset)
    got = 0
    while remaining > 0:
        raw = fasta.fh.read(min(step, remaining))
        if not raw:
            break
        remaining -= len(raw)
        bases = raw.translate(UPPER, b"\r\n")
        if got + len(bases) <= length:
            seq[got : got + len(bases)] = bases
        got += len(bases)
    if got != length:
        raise ValueError(f"read {got} bases of {chrom} where the .fai says {length}; does {fasta.path}.fai match it?")
    return seq


def read_targets(path: Path, strata: tuple[str, ...]) -> tuple[dict[str, list[int]], dict[str, int]]:
    """pos0 (pos - 1) of every row in the given strata, per chromosome, and the list's row count per stratum."""
    per: dict[str, list[int]] = collections.defaultdict(list)
    rows: collections.Counter = collections.Counter()
    with open_text(path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        missing = [c for c in ("chrom", "pos", "stratum") if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: no column(s) {', '.join(missing)} in its header {reader.fieldnames}")
        for row in reader:
            rows[row["stratum"]] += 1
            if row["stratum"] in strata:
                per[row["chrom"]].append(int(row["pos"]) - 1)
    return dict(per), dict(sorted(rows.items()))


def read_peak_union(paths) -> dict[str, IntervalSet]:
    """The union of the intervals (columns 1-3) of every narrowPeak, per chromosome."""
    per: dict[str, list] = collections.defaultdict(list)
    for path in paths:
        for chrom, iv in read_bed(path).items():
            per[chrom].extend(zip(iv.starts, iv.ends))
    return {c: IntervalSet(v) for c, v in per.items()}


class Pool:
    """The eligible tiles in genome order, in compact parallel arrays."""

    def __init__(self):
        self.chrom = array("H")  # index into the --chroms list
        self.start = array("L")  # tile start t, 0-based
        self.gc = array("L")  # G and C bases in the tile
        self.ref = array("B")  # the base at t + flank, as a byte

    def __len__(self) -> int:
        return len(self.start)


def scan(fasta: Fasta, chroms: list[str], targets: dict[str, list[int]], peaks: dict[str, IntervalSet],
         blacklist: dict[str, IntervalSet], flank: int, seed: int, nbins: int) -> dict:  # fmt: skip
    """One pass over the FASTA: the target windows' GC counts and the eligible tiles (module docstring)."""
    width = 2 * flank
    index = {c: i for i, c in enumerate(chroms)}
    wanted = set(chroms) | set(targets)
    pool = Pool()
    counts = {c: dict.fromkeys(FILTERS, 0) for c in chroms}
    offsets: dict[str, int] = {}
    acgt_bins = [0] * nbins
    target_gc: list[int] = []
    skipped: collections.Counter = collections.Counter()
    for chrom in (c for c in fasta.order if c in wanted):
        t0 = time.time()
        seq = read_chrom(fasta, chrom)
        length = len(seq)
        for pos0 in targets.get(chrom, ()):
            a, b = pos0 - flank, pos0 + flank
            if a < 0 or b > length:
                skipped["window_off_chrom"] += 1
                continue
            gc = seq.count(b"G", a, b) + seq.count(b"C", a, b)
            if gc + seq.count(b"A", a, b) + seq.count(b"T", a, b) != width:
                skipped["window_non_acgt"] += 1
                continue
            target_gc.append(gc)
        if chrom in index:
            ci, n = index[chrom], counts[chrom]
            offsets[chrom] = offset = tile_offset(seed, chrom, width)
            in_peaks, in_blacklist = peaks.get(chrom), blacklist.get(chrom)
            for t in range(offset, length - width + 1, width):
                e = t + width
                n["all"] += 1
                gc = seq.count(b"G", t, e) + seq.count(b"C", t, e)
                if gc + seq.count(b"A", t, e) + seq.count(b"T", t, e) != width:
                    continue
                n["acgt"] += 1
                acgt_bins[gc_bin_of(gc, width, nbins)] += 1
                if in_peaks is not None and in_peaks.overlaps(t, e):
                    continue
                n["not_in_peak"] += 1
                if in_blacklist is not None and in_blacklist.overlaps(t, e):
                    continue
                n["not_blacklisted"] += 1
                pos = t + flank + 1
                if not (pos - flank > 0 and pos + flank <= length):
                    continue
                n["in_bounds"] += 1
                pool.chrom.append(ci)
                pool.start.append(t)
                pool.gc.append(gc)
                pool.ref.append(seq[t + flank])
            logger.info("%s: %d tiles, %d ACGT-only, %d eligible; %d target windows; %.1f s", chrom, n["all"],
                        n["acgt"], n["in_bounds"], len(targets.get(chrom, ())), time.time() - t0)  # fmt: skip
        elif chrom in targets:
            logger.info("%s: %d target windows only; %.1f s", chrom, len(targets[chrom]), time.time() - t0)
        del seq
    return {"pool": pool, "counts": counts, "offsets": offsets, "acgt_bins": acgt_bins, "target_gc": target_gc,
            "target_skipped": dict(sorted(skipped.items()))}  # fmt: skip


def draw(bins: list[array], target: list[int], seed: int) -> tuple[list[int], dict]:
    """Fill each bin to its target from its own tiles, then every shortfall from the nearest surplus (docstring).

    bins[k] holds the pool indices of bin k's eligible tiles in genome order. Returns the drawn pool indices, sorted,
    and per bin: available, taken (own tiles for its own target), borrowed_from ({bin: rows}) and lent."""
    nbins = len(target)
    queues = []
    for k, tiles in enumerate(bins):
        queue = array(tiles.typecode, tiles)
        random.Random(f"{seed}:bin:{k}").shuffle(queue)
        queues.append(queue)
    available = [len(q) for q in queues]
    if sum(available) < sum(target):
        raise ValueError(f"{sum(available)} eligible tiles for {sum(target)} rows")
    taken = [min(t, a) for t, a in zip(target, available)]
    used = list(taken)
    need = [t - k for t, k in zip(target, taken)]
    borrowed_from: list[collections.Counter] = [collections.Counter() for _ in range(nbins)]
    drawn = [i for k in range(nbins) for i in queues[k][: taken[k]]]
    for d in range(1, nbins):
        if not any(need):
            break
        for k in range(nbins):
            for j in (k - d, k + d):  # below, then above
                if need[k] and 0 <= j < nbins and used[j] < available[j]:
                    m = min(need[k], available[j] - used[j])
                    drawn.extend(queues[j][used[j] : used[j] + m])
                    used[j] += m
                    need[k] -= m
                    borrowed_from[k][j] += m
    if any(need):  # unreachable: the total was checked above, and every bin is within nbins - 1 of every other
        raise RuntimeError(f"bins still short after borrowing: {need}")
    return sorted(drawn), {
        "available": available,
        "taken": taken,
        "borrowed_from": [dict(sorted(c.items())) for c in borrowed_from],
        "lent": [u - k for u, k in zip(used, taken)],
    }


def gc_summary(gc_counts, window_len: int) -> dict:
    """n, mean and quantiles of GC fractions; quantiles are numpy's default ('linear', R type 7)."""
    xs = sorted(g / window_len for g in gc_counts)
    out: dict = {"n": len(xs), "mean": round(statistics.fmean(xs), 6) if xs else None}
    if len(xs) >= 2:
        cuts = statistics.quantiles(xs, n=100, method="inclusive")
        out.update({name: round(cuts[p - 1], 6) for name, p in QUANTILES.items()})
    return out


def ks_distance(a, b, window_len: int) -> float:
    """Two-sample Kolmogorov-Smirnov distance, sup |F_a - F_b|, of two samples of GC counts in [0, window_len]."""
    ha, hb = [0] * (window_len + 1), [0] * (window_len + 1)
    for g in a:
        ha[g] += 1
    for g in b:
        hb[g] += 1
    ca = cb = 0
    d = 0.0
    for g in range(window_len + 1):
        ca += ha[g]
        cb += hb[g]
        d = max(d, abs(ca / len(a) - cb / len(b)))
    return d


def file_info(path: Path, md5: bool = True) -> dict:
    st = path.stat()
    info = {"path": str(path.resolve()), "size": st.st_size,
            "mtime_et": dt.datetime.fromtimestamp(st.st_mtime, ET).strftime("%Y-%m-%d %H:%M:%S %Z")}  # fmt: skip
    if md5:
        info["md5"] = md5_file(path)
    return info


def code_info(script: Path | None = None) -> dict:
    """A script's md5 (this one's by default), and the commit of the checkout it ran from, when git can tell."""
    script = Path(script or __file__).resolve()
    info: dict = {"script": str(script), "md5": md5_file(script), "commit": None, "script_differs_from_commit": None}
    git = ["git", "-C", str(script.parent)]
    try:
        head = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
        status = subprocess.run(git + ["status", "--porcelain", "--", script.name], capture_output=True, text=True,
                                timeout=30)  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return info
    info["commit"] = head.stdout.strip() or None
    if status.returncode == 0:
        info["script_differs_from_commit"] = bool(status.stdout.strip())
    return info


def bin_table(nbins: int, target_windows: list[int], target: list[int], acgt_bins: list[int], drawn: dict) -> list:
    """Per GC bin with anything in it: the target, the tiles, and how the draw filled it (meta definitions)."""
    table = []
    for k in range(nbins):
        row = {
            "bin": k,
            "gc_from": round(k / nbins, 6),
            "gc_to": round((k + 1) / nbins, 6),
            "target_windows": target_windows[k],
            "target": target[k],
            "acgt_tiles": acgt_bins[k],
            "available": drawn["available"][k],
            "taken": drawn["taken"][k],
            "shortfall": target[k] - drawn["taken"][k],
            "borrowed": sum(drawn["borrowed_from"][k].values()),
            "borrowed_from": {str(j): m for j, m in drawn["borrowed_from"][k].items()},
            "lent": drawn["lent"][k],
            "achieved": drawn["taken"][k] + drawn["lent"][k],
        }
        if row["target_windows"] or row["acgt_tiles"] or row["achieved"]:
            table.append(row)
    return table


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fasta", type=Path, required=True, help="uncompressed FASTA with its .fai")
    ap.add_argument("--peaks", type=Path, nargs="+", required=True,
                    help="narrowPeaks: windows touch none of their intervals; summits give dist_to_summit")  # fmt: skip
    ap.add_argument("--blacklist", type=Path, required=True, help="BED(.gz) no window touches (no slop)")
    ap.add_argument("--target-list", type=Path, required=True,
                    help="variant list TSV whose --target-strata rows' windows set the GC target")  # fmt: skip
    ap.add_argument("--target-strata", default="summit100,flank500",
                    help="comma-separated (default summit100,flank500: build_windows.py --null's strata)")  # fmt: skip
    ap.add_argument(
        "--manifest", type=Path, required=True, help="make_manifest.py's manifest.json (test_chrom_to_fold)"
    )
    ap.add_argument("--chroms", default=DEFAULT_CHROMS, help="comma-separated (default chr1-22,chrX)")
    ap.add_argument("--n", type=int, default=10000, help="rows (default 10000)")
    ap.add_argument("--seed", type=int, default=1234, help="seed of the offsets, the draw and ALT (default 1234)")
    ap.add_argument("--flank", type=int, default=1057,
                    help="window = 2 * flank bp, the variant at index flank (default 1057: 2114 bp)")  # fmt: skip
    ap.add_argument("--gc-bin", type=float, default=0.01, help="GC bin width, a whole fraction of 1 (default 0.01)")
    ap.add_argument("--out", type=Path, required=True, help="<name>.tsv; <name>.meta.json goes beside it")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%F %T")  # fmt: skip

    if args.out.suffix != ".tsv":
        ap.error("--out must end in .tsv")
    if args.n < 1 or args.flank < 1:
        ap.error("--n and --flank must be at least 1")
    try:
        nbins = bin_count(args.gc_bin)
    except ValueError as err:
        ap.error(str(err))
    fasta = Fasta(args.fasta)
    try:
        return generate(args, fasta, nbins, sys.argv[1:] if argv is None else list(argv))
    finally:
        fasta.fh.close()


def generate(args, fasta: Fasta, nbins: int, argv: list[str]) -> int:
    """Check the inputs, scan, draw, and write the list and its meta; 0, or 1 when the inputs cannot give the list."""
    started, t0 = now_et(), time.time()
    width = 2 * args.flank
    chroms = parse_chroms(args.chroms)
    strata = tuple(parse_chroms(args.target_strata))
    order = {c: i for i, c in enumerate(fasta.order)}
    missing = [c for c in chroms if c not in order]
    repeated = sorted(c for c, k in collections.Counter(chroms).items() if k > 1)
    if missing or repeated or len(chroms) > 65535:
        logger.error("--chroms must name at most 65535 distinct sequences of the FASTA; it names %d, not in the FASTA: "
                     "%s, repeated: %s", len(chroms), missing, repeated)  # fmt: skip
        return 1
    test_fold = json.loads(args.manifest.read_text()).get("test_chrom_to_fold") or {}
    no_fold = [c for c in chroms if c not in test_fold]
    if no_fold:
        logger.error("%s: test_chrom_to_fold has no fold for %s", args.manifest, no_fold)
        return 1
    targets, target_rows = read_targets(args.target_list, strata)
    unknown = sorted(c for c in targets if c not in order)
    if unknown or not targets:
        logger.error("%s: no rows of strata %s, or rows on sequences the FASTA lacks: %s", args.target_list,
                     ",".join(strata), unknown)  # fmt: skip
        return 1
    peaks = read_peak_union(args.peaks)
    summits, peak_rows = read_summits(args.peaks)
    blacklist = read_bed(args.blacklist)

    found = scan(fasta, chroms, targets, peaks, blacklist, args.flank, args.seed, nbins)
    pool, target_gc = found["pool"], found["target_gc"]
    if not target_gc:
        logger.error("no target window is inside its chromosome and ACGT-only: %s", found["target_skipped"])
        return 1
    if len(pool) < args.n:
        logger.error("only %d eligible tiles for --n %d", len(pool), args.n)
        return 1
    target_windows = [0] * nbins
    for g in target_gc:
        target_windows[gc_bin_of(g, width, nbins)] += 1
    target = largest_remainder(target_windows, args.n)
    bins = [array("L") for _ in range(nbins)]
    for i, g in enumerate(pool.gc):
        bins[gc_bin_of(g, width, nbins)].append(i)
    picks, filled = draw(bins, target, args.seed)
    del bins

    # Rows in .fai order, then pos; ALT drawn in that order.
    tiles = sorted(((chroms[pool.chrom[i]], pool.start[i] + args.flank, chr(pool.ref[i]), pool.gc[i]) for i in picks),
                   key=lambda x: (order[x[0]], x[1]))  # fmt: skip
    if len({(c, p) for c, p, _, _ in tiles}) != len(tiles):
        raise RuntimeError("two drawn tiles share a position")
    rng = random.Random(f"{args.seed}:alt")
    rows = []
    for chrom, pos0, ref, _ in tiles:
        alt = rng.choice([b for b in BASES if b != ref])
        d = nearest_distance(summits.get(chrom, []), pos0)
        rows.append((f"{chrom}:{pos0 + 1}:{ref}:{alt}", chrom, pos0 + 1, ref, alt, NAME, NAME, "",
                     NO_SUMMIT if d is None else d, test_fold[chrom]))  # fmt: skip
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".tmp")
    with open(tmp, "w") as fh:
        fh.write("\t".join(LIST_COLUMNS) + "\n")
        for row in rows:
            fh.write("\t".join(map(str, row)) + "\n")
    tmp.replace(args.out)

    drawn_gc = [g for _, _, _, g in tiles]
    table = bin_table(nbins, target_windows, target, found["acgt_bins"], filled)
    short = [r for r in table if r["shortfall"]]
    totals = {f: sum(c[f] for c in found["counts"].values()) for f in FILTERS}
    dists = sorted(r[8] for r in rows)
    meta_path = args.out.with_name(args.out.name[: -len(".tsv")] + ".meta.json")
    meta = {
        "created_et": started,
        "finished_et": now_et(),
        "argv": argv,
        "python": {"executable": sys.executable, "version": sys.version.split()[0]},
        "code": code_info(),
        "inputs": {
            "fasta": file_info(args.fasta, md5=False),
            "fasta_fai": file_info(Path(f"{args.fasta}.fai")),
            "peaks": [{**file_info(p), "rows": peak_rows[str(p)]} for p in args.peaks],
            "blacklist": file_info(args.blacklist),
            "target_list": file_info(args.target_list),
            "manifest": file_info(args.manifest),
        },
        "outputs": {"list": {**file_info(args.out), "rows": len(rows)}, "meta": str(meta_path.resolve())},
        "list_columns": list(LIST_COLUMNS),
        "params": {
            "n": args.n,
            "seed": args.seed,
            "flank": args.flank,
            "window_len": width,
            "gc_bin": args.gc_bin,
            "n_bins": nbins,
            "chroms": chroms,
            "target_strata": list(strata),
        },
        "rng_streams": {
            "offset": f"random.Random('{args.seed}:offset:<chrom>').randrange({width})",
            "bin": f"random.Random('{args.seed}:bin:<k>').shuffle(the bin's eligible tiles in genome order)",
            "alt": f"random.Random('{args.seed}:alt').choice(the 3 other bases), one call per row in output order",
        },
        "offsets": found["offsets"],
        "tiles": {"filters": list(FILTERS), "total": totals, "per_chrom": found["counts"]},
        "target": {
            "rows_by_stratum": target_rows,
            "rows_in_strata": sum(len(v) for v in targets.values()),
            "windows": len(target_gc),
            "skipped": found["target_skipped"],
        },
        "gc": {
            "target": gc_summary(target_gc, width),
            "eligible": gc_summary(pool.gc, width),
            "drawn": gc_summary(drawn_gc, width),
            "ks_drawn_vs_target": round(ks_distance(drawn_gc, target_gc, width), 6),
            "ks_eligible_vs_target": round(ks_distance(pool.gc, target_gc, width), 6),
        },
        "shortfall": {
            "bins_short": [r["bin"] for r in short],
            "rows_borrowed": sum(r["borrowed"] for r in short),
            "by_bin": {
                str(r["bin"]): {k: r[k] for k in ("target", "available", "taken", "borrowed", "borrowed_from")}
                for r in short
            },
        },
        "bins": table,
        "n_rows": len(rows),
        "counts_by_chrom": {c: sum(1 for r in rows if r[1] == c) for c in chroms},
        "counts_by_fold_test": dict(sorted(collections.Counter(r[9] for r in rows).items())),
        "ref_counts": {b: sum(1 for r in rows if r[3] == b) for b in BASES},
        "alt_counts": {b: sum(1 for r in rows if r[4] == b) for b in BASES},
        "ref_alt_counts": dict(sorted(collections.Counter(f"{r[3]}>{r[4]}" for r in rows).items())),
        "dist_to_summit": {"min": dists[0], "median": dists[len(dists) // 2], "no_summit_rows": dists.count(NO_SUMMIT)},
        "union_peaks_on_chroms": {
            "intervals": sum(len(peaks[c]) for c in chroms if c in peaks),
            "bp": sum(e - s for c in chroms if c in peaks for s, e in zip(peaks[c].starts, peaks[c].ends)),
        },
        "definitions": {
            "window": f"[pos0 - {args.flank}, pos0 + {args.flank}), pos0 = pos - 1; the variant at index {args.flank}",
            "tiles": "non-overlapping windows from each chromosome's offset; counts are the tiles left after each "
            "filter, in the order given (module docstring)",
            "in_bounds": f"pos - {args.flank} > 0 and pos + {args.flank} <= length (VS get_valid_variants)",
            "bin": f"k = min({nbins - 1}, gc_count * {nbins} // {width}); GC fraction in [gc_from, gc_to)",
            "target_windows": "target-list windows in the bin; target: those scaled to n by largest remainder",
            "acgt_tiles": "ACGT-only tiles in the bin before the peak, blacklist and bound filters",
            "available": "eligible tiles in the bin (after every filter)",
            "taken": "the bin's own tiles drawn for its own target",
            "shortfall": "target - taken, filled by borrowing",
            "borrowed": "rows drawn from other bins for this bin's shortfall; borrowed_from: {bin: rows}",
            "lent": "this bin's tiles drawn for other bins' shortfalls",
            "achieved": "rows whose GC is in this bin (taken + lent)",
            "borrow_order": "distance 1, 2, ...; at each distance every short bin in ascending order, below first",
            "quantiles": "of the GC fraction; numpy's default linear interpolation (R type 7)",
            "ks": "two-sample Kolmogorov-Smirnov distance of the GC fractions",
            "dist_to_summit": "|pos0 - s|, s the nearest summit (start + column 10) of any --peaks file",
        },
        "seconds": round(time.time() - t0, 1),
        "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    }
    write_json(meta_path, meta)
    logger.info("wrote %s: %d rows from %d eligible tiles; GC median %.4f drawn, %.4f target; KS %.4f (eligible "
                "%.4f); %d rows borrowed for %d short bins; %.0f s, max RSS %.0f MB", args.out, len(rows), len(pool),
                meta["gc"]["drawn"].get("q50", float("nan")), meta["gc"]["target"].get("q50", float("nan")),
                meta["gc"]["ks_drawn_vs_target"], meta["gc"]["ks_eligible_vs_target"],
                meta["shortfall"]["rows_borrowed"], len(short), meta["seconds"], meta["max_rss_mb"])  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
