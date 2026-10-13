#!/usr/bin/env python3
"""Turn a variant table into a variant list: its rows near the summits of a manifest's datasets, checked and classified.

The table is a TSV (or TSV.gz) with a header row and at least the columns chr (or chrom), pos (1-based, as in a VCF),
ref and alt (anchored VCF alleles); every other column is passed through, its text unchanged. Its rows must be grouped
by chromosome: the table is read once, as a stream, and the FASTA one chromosome at a time.

From the manifest (make_manifest.py's manifest.json)
  genome.path              the FASTA, with its .fai (unless --genome names another)
  datasets.*.peaks.path    the narrowPeaks whose summits (start + column 10; the midpoint if it is -1) are united
  test_chrom_to_fold       fold_test

Selection
  pos0 = pos - 1 is the base build_windows.py places at window index input_len // 2, and dist_to_summit = |pos0 - s|
  for the nearest summit s of the union. A row is selected when dist_to_summit <= --outer. Rows farther away, or on a
  chromosome without a summit, are only counted.

Checks
  A selected row is dropped, written to <list>.dropped.tsv and counted, for the first of these it fails:
    duplicate          the chrom, pos, ref and alt of an earlier row (that first copy is checked on its own)
    other_chrom        chrom not in --chroms
    multiallelic       ALT holds a comma
    non_acgt_allele    REF or ALT empty, or holding anything but A, C, G and T (`*`, N)
    window_off_chrom   the --input-len window plus the extra right-flank bases a deletion reads must lie inside the
                       chromosome, with VS's get_valid_variants bounds (1 bp stricter on the right); both alleles must
                       fit in half the window
    ref_mismatch       REF differs from the FASTA (case-insensitive)
    window_non_acgt    the window holds a base other than A, C, G and T
    no_test_fold       test_chrom_to_fold has no fold for chrom
  other_chrom to window_non_acgt are gnomad_list.py's Checker, in its order. Unlike gnomad_list.py's lists, MNVs (REF
  and ALT of one length above 1) are kept: build_windows.py builds their windows as VS does.

Outputs, for --out <list>.tsv
  <list>.tsv          one row per kept row, in the table's order:
                        variant_id         chrom:pos:ref:alt; when that is longer than 40 characters,
                                           chrom:pos:<len(ref)>><len(alt)>:<the first 10 hex digits of
                                           md5(ref + ">" + alt)>. numpy stores ids at the width of the longest one,
                                           so one long insertion would otherwise widen every id array downstream
                        chrom, pos         pos 1-based
                        ref, alt           as in the table
                        source             --source
                        stratum            summit<inner> (dist_to_summit <= --inner) or flank<outer>
                        af                 the --af-column text when it is a finite number, else empty
                        dist_to_summit     as above
                        fold_test          the fold whose test set holds chrom
                        mut_class          SNVs: CpG_ti, CpG_tv, ti or tv (CpG_ when REF is the C or the G of a
                                           reference CpG); mnv; indel
                        then the table's other columns, in its order
  <list>.dropped.tsv  every dropped row: reason; detail (ref_mismatch: the FASTA's bases; duplicate: the line of the
                      first copy); line (in the table, the header being line 1); variant_id; the table's columns
  <list>.meta.json    the inputs (path, size, mtime ET, md5; the FASTA's md5 as the manifest records it), params,
                      definitions, the rows left after every filter, the kept rows by stratum, mut_class, chromosome
                      and fold, the ids compacted and the longest, the table's order, run time, peak RSS, the code
  The list and the dropped rows are written under temporary names and renamed at the end; the meta is written last.

    python -I workflows/variant_scoring/table_list.py --table variants.tsv.gz --manifest manifest.json \\
        --source gnomad_v4_common --af-column AF_nfe --out lists/list.tsv

Python standard library only (with gnomad_list.py's and genomic_null.py's helpers); it never loads numpy. Memory: the
longest chromosome (about 250 MB for hg38's chr1), the summits, and the keys of one chromosome's selected rows.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import logging
import math
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from genomic_null import code_info, file_info, read_chrom  # noqa: E402
from gnomad_list import (  # noqa: E402
    DEFAULT_CHROMS,
    DROP_REASONS,
    LIST_COLUMNS,
    Checker,
    Fasta,
    nearest_distance,
    now_et,
    open_text,
    parse_chroms,
    read_summits,
    strata_names,
    stratum_of,
    write_json,
)
from read_manifest import ManifestError  # noqa: E402
from read_manifest import load as load_manifest  # noqa: E402

logger = logging.getLogger("table_list")

MAX_ID_LEN = 40
ID_HASH_CHARS = 10
OUTPUT_COLUMNS = (*LIST_COLUMNS, "mut_class")
DROPPED_COLUMNS = ("reason", "detail", "line", "variant_id")
# In the order they are tested: duplicate, then the Checker's (MNVs are kept, so never `mnv`), then the fold.
REASONS = ("duplicate", *(r for r in DROP_REASONS if r not in ("not_pass", "mnv", "duplicate")), "no_test_fold")
MUT_CLASSES = ("CpG_ti", "CpG_tv", "ti", "tv", "mnv", "indel")
TRANSITIONS = {("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")}


class TableError(ValueError):
    """The table cannot be read as the module docstring describes."""


class ChromFasta(Fasta):
    """A Fasta whose fetch serves the chromosome asked for last from memory, read whole with genomic_null.read_chrom.

    A table grouped by chromosome then reads each chromosome once, and holds one at a time."""

    def __init__(self, path):
        super().__init__(path)
        self.chrom: str | None = None
        self.seq = bytearray()
        self.loads = 0

    def fetch(self, chrom: str, start: int, end: int) -> str:
        """Bases [start, end), 0-based half-open, upper-case."""
        if chrom != self.chrom:
            self.chrom, self.seq = None, bytearray()  # free the last chromosome before reading the next
            self.seq = read_chrom(self, chrom)
            self.chrom = chrom
            self.loads += 1
        if not 0 <= start <= end <= len(self.seq):
            raise ValueError(f"{chrom}:{start}-{end} is outside 0-{len(self.seq)}")
        return self.seq[start:end].decode("ascii")


def variant_id(chrom: str, pos: int, ref: str, alt: str) -> tuple[str, bool]:
    """The list's id, and whether it was compacted: chrom:pos:ref:alt, or, when that is longer than MAX_ID_LEN,
    chrom:pos:<len(ref)>><len(alt)>:<the first ID_HASH_CHARS hex digits of md5(ref + ">" + alt)>."""
    vid = f"{chrom}:{pos}:{ref}:{alt}"
    if len(vid) <= MAX_ID_LEN:
        return vid, False
    digest = hashlib.md5(f"{ref}>{alt}".encode()).hexdigest()[:ID_HASH_CHARS]
    return f"{chrom}:{pos}:{len(ref)}>{len(alt)}:{digest}", True


def mut_class(ref: str, alt: str, before: str = "", after: str = "") -> str:
    """indel, mnv, or an SNV's ti (A<>G, C<>T) or tv, as CpG_ti or CpG_tv when REF is the C or the G of a reference
    CpG: REF C with `after` (the next reference base) G, or REF G with `before` (the previous one) C."""
    if len(ref) != len(alt):
        return "indel"
    if len(ref) > 1:
        return "mnv"
    cpg = (ref == "C" and after == "G") or (ref == "G" and before == "C")
    return ("CpG_" if cpg else "") + ("ti" if (ref, alt) in TRANSITIONS else "tv")


def af_text(text: str) -> str:
    """text, unchanged, when it is a finite number; else empty (NaN, NA, inf, '')."""
    try:
        return text if math.isfinite(float(text)) else ""
    except ValueError:
        return ""


def table_columns(header: list[str], af_column: str | None) -> dict:
    """Where chrom, pos, ref, alt and --af-column are in the header, and the indices of the pass-through columns."""
    chrom = [c for c in ("chr", "chrom") if c in header]
    missing = [c for c in ("pos", "ref", "alt") if c not in header]
    if len(chrom) != 1 or missing:
        raise TableError(f"the header needs one of chr or chrom (it has {chrom or 'neither'}) and pos, ref and alt "
                         f"(it lacks {missing or 'none'}): {header}")  # fmt: skip
    repeated = sorted(c for c, n in collections.Counter(header).items() if n > 1)
    if repeated:
        raise TableError(f"the header repeats {repeated}")
    own = (chrom[0], "pos", "ref", "alt")
    passthrough = [i for i, c in enumerate(header) if c not in own]
    clash = [header[i] for i in passthrough if header[i] in OUTPUT_COLUMNS]
    if clash:
        raise TableError(f"the table's columns {clash} would repeat the list's own")
    if af_column is not None and af_column not in [header[i] for i in passthrough]:
        raise TableError(f"--af-column {af_column} is not one of the table's other columns")
    return {
        "chrom": header.index(chrom[0]),
        "pos": header.index("pos"),
        "ref": header.index("ref"),
        "alt": header.index("alt"),
        "af": header.index(af_column) if af_column is not None else None,
        "passthrough": passthrough,
    }


def manifest_inputs(path: Path) -> tuple[dict, dict[str, Path], dict[str, str]]:
    """The manifest, each dataset's peaks file and test_chrom_to_fold."""
    manifest = load_manifest(path)
    peaks = {}
    for ds, entry in manifest["datasets"].items():
        p = (entry.get("peaks") or {}).get("path")
        if not p:
            raise ManifestError(f"{path}: dataset {ds} has no peaks.path")
        peaks[ds] = Path(p)
    return manifest, peaks, dict(manifest.get("test_chrom_to_fold") or {})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", type=Path, required=True,
                    help="TSV(.gz) with a header: chr or chrom, pos (1-based), ref, alt; other columns pass "
                    "through")  # fmt: skip
    ap.add_argument("--manifest", type=Path, required=True,
                    help="make_manifest.py's manifest.json: the genome, the datasets' peaks, "
                    "test_chrom_to_fold")  # fmt: skip
    ap.add_argument("--genome", type=Path, default=None, help="FASTA with .fai (default: the manifest's genome)")
    ap.add_argument("--chroms", default=DEFAULT_CHROMS, help="comma-separated (default chr1-22,chrX)")
    ap.add_argument("--inner", type=int, default=100, help="summit<inner>: dist_to_summit <= this (default 100)")
    ap.add_argument("--outer", type=int, default=500, help="rows with dist_to_summit <= this are selected (500)")
    ap.add_argument("--input-len", type=int, default=2114, help="the models' input window (default 2114)")
    ap.add_argument("--source", required=True, help="the source column's value for every row")
    ap.add_argument("--af-column", default=None, help="the table column af takes its text from (default: none)")
    ap.add_argument("--out", type=Path, required=True, help="<list>.tsv; .dropped.tsv and .meta.json go beside it")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%F %T")  # fmt: skip

    if args.out.suffix != ".tsv":
        ap.error("--out must end in .tsv")
    if not 0 <= args.inner <= args.outer:
        ap.error("--inner and --outer must satisfy 0 <= inner <= outer")
    if args.input_len < 2:
        ap.error("--input-len must be at least 2")
    if not args.source or "\t" in args.source:
        ap.error("--source must be non-empty text without tabs")
    try:
        return build(args, sys.argv[1:] if argv is None else list(argv))
    except (TableError, ManifestError) as err:
        logger.error("%s", err)
        return 1


def build(args, argv: list[str]) -> int:
    """Read the manifest, stream the table into the list and the dropped rows, and write the meta (docstring)."""
    started, t0 = now_et(), time.time()
    manifest, peaks, test_fold = manifest_inputs(args.manifest)
    genome = args.genome or Path((manifest.get("genome") or {}).get("path") or "")
    if not genome.is_file() or not Path(f"{genome}.fai").is_file():
        raise TableError(f"no FASTA with a .fai at {genome} (--genome, or the manifest's genome.path)")
    fasta = ChromFasta(genome)
    try:
        chroms = parse_chroms(args.chroms)
        missing = [c for c in chroms if c not in fasta.index]
        if missing:
            raise TableError(f"--chroms names sequences the FASTA lacks: {missing}")
        summits, peak_rows = read_summits(list(peaks.values()))
        stem = args.out.name[: -len(".tsv")]
        paths = {"list": args.out, "dropped": args.out.with_name(f"{stem}.dropped.tsv")}
        tmps = {k: p.with_name(p.name + ".tmp") for k, p in paths.items()}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        try:
            found = stream(args, fasta, chroms, summits, test_fold, tmps)
        except BaseException:
            for tmp in tmps.values():
                tmp.unlink(missing_ok=True)
            raise
        for key, tmp in tmps.items():
            tmp.replace(paths[key])
    finally:
        fasta.fh.close()

    c = found["counts"]
    left, funnel = c["selected"], []
    for reason in REASONS:
        left -= c["dropped"][reason]
        funnel.append({"filter": reason, "dropped": c["dropped"][reason], "left": left})
    if left != c["kept"]:
        raise RuntimeError(f"{left} rows left after the checks, {c['kept']} written")
    names = strata_names(args.inner, args.outer)[:2]
    cross = c["stratum_mut_class"]
    genome_entry = manifest.get("genome") or {}
    same_genome = Path(genome_entry.get("path") or "").resolve() == genome.resolve()
    fasta_md5 = genome_entry.get("md5") if same_genome else None  # the manifest's, for its own genome only
    meta_path = args.out.with_name(f"{args.out.name[: -len('.tsv')]}.meta.json")
    meta = {
        "created_et": started,
        "finished_et": now_et(),
        "argv": argv,
        "python": {"executable": sys.executable, "version": sys.version.split()[0]},
        "code": code_info(Path(__file__)),
        "inputs": {
            "table": {**file_info(args.table), "rows": c["rows"], "header": found["header"]},
            "manifest": file_info(args.manifest),
            "fasta": {**file_info(genome, md5=False), "md5_in_manifest": fasta_md5},
            "fasta_fai": file_info(Path(f"{genome}.fai")),
            "peaks": [{"dataset": ds, **file_info(p), "rows": peak_rows[str(p)]} for ds, p in peaks.items()],
        },
        "outputs": {
            "list": {**file_info(paths["list"]), "rows": c["kept"]},
            "dropped": {**file_info(paths["dropped"]), "rows": sum(c["dropped"].values())},
            "meta": str(meta_path.resolve()),
        },
        "list_columns": found["columns"],
        "params": {
            "inner": args.inner,
            "outer": args.outer,
            "input_len": args.input_len,
            "chroms": chroms,
            "source": args.source,
            "af_column": args.af_column,
            "max_id_len": MAX_ID_LEN,
            "id_hash_chars": ID_HASH_CHARS,
            "datasets": list(peaks),
        },
        "definitions": {
            "selection": f"dist_to_summit = |pos0 - s|, pos0 = pos - 1 and s the nearest summit (start + column 10) "
            f"of the datasets' peaks; selected when dist_to_summit <= {args.outer}",
            "strata": {
                names[0]: f"dist_to_summit <= {args.inner}",
                names[1]: f"{args.inner} < dist_to_summit <= {args.outer}",
            },
            "checks": list(REASONS),
            "check_order": "the first failed check names the reason; duplicate is tested first, then gnomad_list.py's "
            "Checker (MNVs kept), then the test fold",
            "window": f"[pos0 - {args.input_len // 2}, pos0 + {args.input_len // 2} + max(0, len(ref) - len(alt)))",
            "variant_id": f"chrom:pos:ref:alt; longer than {MAX_ID_LEN} characters: "
            f"chrom:pos:<len(ref)>><len(alt)>:<md5(ref + '>' + alt), first {ID_HASH_CHARS} hex digits>",
            "af": f"the {args.af_column} text when it is a finite number, else empty" if args.af_column else "empty",
            "fold_test": "the manifest's test_chrom_to_fold",
            "mut_class": "SNVs: ti (A<>G, C<>T) or tv, CpG_ti or CpG_tv when REF is the C (next base G) or the G "
            "(previous base C) of a reference CpG; mnv: REF and ALT of one length above 1; indel: the rest",
            "passthrough": "the table's other columns, text unchanged",
            "order": "the table's",
        },
        "union_summits": {
            "total": sum(len(v) for v in summits.values()),
            "per_chrom": {ch: len(summits.get(ch, [])) for ch in chroms},
        },
        "counts": {
            "rows": c["rows"],
            "rows_on_chroms_with_summits": c["with_summits"],
            "selected": c["selected"],
            "funnel": funnel,
            "dropped": c["dropped"],
            "kept": c["kept"],
            "by_stratum": {s: sum(cross[s].values()) for s in names},
            "by_mut_class": {m: sum(cross[s][m] for s in names) for m in MUT_CLASSES},
            "by_stratum_mut_class": cross,
            "by_fold_test": dict(sorted(c["fold"].items())),
            "by_chrom": found["per_chrom"],
            "af_empty": c["af_empty"],
        },
        "ids": found["ids"],
        "table_order": found["order"],
        "chromosome_loads": fasta.loads,
        "seconds": round(time.time() - t0, 1),
        "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    }
    write_json(meta_path, meta)
    logger.info("wrote %s: %d of %d rows selected (dist_to_summit <= %d) kept, %s; dropped %s; %d ids compacted, "
                "longest %d; %.0f s, max RSS %.0f MB", args.out, c["kept"], c["selected"], args.outer,
                meta["counts"]["by_stratum"], {r: n for r, n in c["dropped"].items() if n}, found["ids"]["compacted"],
                found["ids"]["longest_written"], meta["seconds"], meta["max_rss_mb"])  # fmt: skip
    return 0


def stream(args, fasta: ChromFasta, chroms: list[str], summits: dict[str, list[int]], test_fold: dict[str, str],
           tmps: dict[str, Path]) -> dict:  # fmt: skip
    """One pass over the table: select, check and write every row; the counts for the meta."""
    checker = Checker(fasta, chroms, args.input_len, keep_mnv=True)
    names = strata_names(args.inner, args.outer)
    counts = {"rows": 0, "with_summits": 0, "selected": 0, "kept": 0, "af_empty": 0,
              "dropped": dict.fromkeys(REASONS, 0), "fold": collections.Counter(),
              "stratum_mut_class": {s: dict.fromkeys(MUT_CLASSES, 0) for s in names[:2]}}  # fmt: skip
    per_chrom: dict[str, dict] = {}
    ids = {"compacted": 0, "longest_plain": 0, "longest_plain_written_as": None, "longest_written": 0}
    compact_ids: set[str] = set()
    order = {"chromosomes": [], "position_steps_back": 0}
    with open_text(args.table) as fh, open(tmps["list"], "w") as out, open(tmps["dropped"], "w") as dropped:
        header = fh.readline().rstrip("\r\n").split("\t")
        cols = table_columns(header, args.af_column)
        ci, pi, ri, ai = (cols[k] for k in ("chrom", "pos", "ref", "alt"))
        fi, extra = cols["af"], cols["passthrough"]
        columns = [*OUTPUT_COLUMNS, *(header[i] for i in extra)]
        out.write("\t".join(columns) + "\n")
        dropped.write("\t".join([*DROPPED_COLUMNS, *header]) + "\n")
        current, seen, stats, summ, last_pos, t_chrom = None, {}, {}, [], 0, time.time()
        for lineno, line in enumerate(fh, start=2):
            f = line.rstrip("\r\n").split("\t")
            if len(f) != len(header):
                if not line.strip():
                    continue
                raise TableError(f"{args.table} line {lineno}: {len(f)} fields where the header has {len(header)}")
            chrom = f[ci]
            if chrom != current:
                if chrom in per_chrom:
                    raise TableError(f"{args.table} line {lineno}: {chrom} again after other chromosomes; the table "
                                     "must be grouped by chromosome")  # fmt: skip
                if current is not None:
                    log_chrom(current, stats, t_chrom)
                current, seen, summ, last_pos, t_chrom = chrom, {}, summits.get(chrom, []), 0, time.time()
                stats = per_chrom[chrom] = {"rows": 0, "selected": 0, "kept": 0, "dropped": 0}
                order["chromosomes"].append(chrom)
            try:
                pos = int(f[pi])
            except ValueError:
                raise TableError(f"{args.table} line {lineno}: pos {f[pi]!r} is not an integer") from None
            if pos < 1:
                raise TableError(f"{args.table} line {lineno}: pos {pos} is not 1-based")
            if pos < last_pos:
                order["position_steps_back"] += 1
            last_pos = pos
            stats["rows"] += 1
            if not summ:
                continue
            d = nearest_distance(summ, pos - 1)
            if d > args.outer:
                continue
            stats["selected"] += 1
            ref, alt = f[ri], f[ai]
            vid, compacted = variant_id(chrom, pos, ref, alt)
            key = (pos, ref, alt)
            first = seen.get(key)
            detail = ""
            if first is not None:
                why, detail = "duplicate", f"line {first}"
            else:
                seen[key] = lineno
                why = checker.reason(chrom, pos, ref, alt)
                if why == "ref_mismatch":
                    detail = "fasta " + fasta.fetch(chrom, pos - 1, pos - 1 + len(ref))
                elif why is None and chrom not in test_fold:
                    why = "no_test_fold"
            if why:
                counts["dropped"][why] += 1
                stats["dropped"] += 1
                dropped.write("\t".join([why, detail, str(lineno), vid, *f]) + "\n")
                continue

            if len(ref) == len(alt) == 1:
                context = fasta.fetch(chrom, pos - 2, pos + 1)  # the bases before, at and after pos0
                mc = mut_class(ref, alt, context[0], context[2])
            else:
                mc = mut_class(ref, alt)
            stratum = names[stratum_of(d, args.inner, args.outer)]
            af = af_text(f[fi]) if fi is not None else ""
            if compacted:
                if len(vid) > MAX_ID_LEN or vid in compact_ids:
                    raise RuntimeError(f"line {lineno}: compact id {vid} is longer than {MAX_ID_LEN} or not unique")
                compact_ids.add(vid)
                ids["compacted"] += 1
            plain_len = len(chrom) + len(str(pos)) + len(ref) + len(alt) + 3  # of chrom:pos:ref:alt
            if plain_len > ids["longest_plain"]:
                ids["longest_plain"], ids["longest_plain_written_as"] = plain_len, vid
            ids["longest_written"] = max(ids["longest_written"], len(vid))
            out.write("\t".join([vid, chrom, str(pos), ref, alt, args.source, stratum, af, str(d), test_fold[chrom],
                                 mc, *(f[i] for i in extra)]) + "\n")  # fmt: skip
            stats["kept"] += 1
            counts["kept"] += 1
            counts["fold"][test_fold[chrom]] += 1
            counts["stratum_mut_class"][stratum][mc] += 1
            counts["af_empty"] += not af
        if current is not None:
            log_chrom(current, stats, t_chrom)
    counts["rows"] = sum(s["rows"] for s in per_chrom.values())
    counts["with_summits"] = sum(s["rows"] for ch, s in per_chrom.items() if summits.get(ch))
    counts["selected"] = sum(s["selected"] for s in per_chrom.values())
    order["position_sorted_within_chrom"] = order["position_steps_back"] == 0
    index = {ch: i for i, ch in enumerate(fasta.order)}
    ranks = [index.get(ch, len(index)) for ch in order["chromosomes"]]
    order["chromosomes_in_fasta_order"] = ranks == sorted(ranks)
    return {"header": header, "columns": columns, "counts": counts, "per_chrom": per_chrom, "ids": ids, "order": order}


def log_chrom(chrom: str, stats: dict, t0: float) -> None:
    logger.info("%s: %d rows, %d selected, %d kept, %d dropped; %.1f s", chrom, stats["rows"], stats["selected"],
                stats["kept"], stats["dropped"], time.time() - t0)  # fmt: skip


if __name__ == "__main__":
    sys.exit(main())
