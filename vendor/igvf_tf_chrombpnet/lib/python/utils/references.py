"""The shared reference files: where they live, what they are, and fetching them.

This is the single definition. ``lib/bash/references.sh`` is a thin wrapper that
evals ``python3 references.py export``, the same arrangement ``config.py`` has
with ``config.yaml`` -- so the paths the pipeline reads and the paths the
downloader writes cannot drift, because there is only one set.

Everything hangs off ``$REFERENCE_ROOT`` (default: the Engreitz lab copy). Set it
to install or read from somewhere else.

Like ``config.py`` this must run on the bare system python before any conda env
exists, so it imports only the standard library at module level.

Accepting an accession means a step can state *which* blacklist it used rather
than a path that says nothing about provenance. The cost is a network call from
wherever the step runs -- on a SLURM compute node that may be firewalled, so a
local path stays supported and is what `lib/bash/references.sh` passes by default.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shlex
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

# ENCODE file accessions are ENCFF + 6 alphanumerics, e.g. ENCFF356LFX.
ENCODE_ACCESSION_RE = re.compile(r"^ENCFF[0-9A-Z]{6}$")

ENCODE_FILE_URL = "https://www.encodeproject.org/files/{accession}/@@download/{accession}.bed.gz"


def is_encode_accession(spec: str) -> bool:
    """True if ``spec`` looks like an ENCODE file accession rather than a path."""
    return bool(ENCODE_ACCESSION_RE.match(str(spec).strip()))


def encode_file_url(accession: str) -> str:
    """The ENCODE download URL for a BED file accession."""
    accession = str(accession).strip()
    if not is_encode_accession(accession):
        raise ValueError(f"not an ENCODE file accession: {accession!r}")
    return ENCODE_FILE_URL.format(accession=accession)


def resolve_bed_source(spec: str) -> str:
    """Turn an accession into a URL; leave URLs and local paths untouched."""
    return encode_file_url(spec) if is_encode_accession(spec) else str(spec)


def describe_source(spec: str) -> str:
    """Human-readable provenance for a log line."""
    spec = str(spec).strip()
    if is_encode_accession(spec):
        return f"ENCODE {spec} ({encode_file_url(spec)})"
    if spec.startswith(("http://", "https://", "ftp://")):
        return f"URL {spec}"
    return f"file {spec}"


# ── the registry ──────────────────────────────────────────────────────────────

DEFAULT_ROOT = "/oak/stanford/groups/engreitz/Data"

#: GRCh38 no-alt analysis set with UCSC ids (chr-prefixed), from the IGVF
#: portal. This exact file is what the GENCODE 43 GTF below was renamed
#: against -- the portal lists both as inputs to the same kallisto index --
#: so genome, annotation and chrom.sizes all agree on contig names. Do not
#: swap one without the other.
GENOME_ACCESSION = "IGVFFI0653VCGH"
#: md5 as published by the portal. Pinned so a corrupted or swapped download is
#: caught even when the metadata API cannot be reached; the API is still
#: consulted, and a disagreement between the two is itself worth failing on.
GENOME_MD5 = "a08035b6a6e31780e96a34008ff21bd6"
BLACKLIST_ACCESSION = "ENCFF356LFX"
#: ChromBPNet's input window. The blacklist slop is half of it.
CHROMBPNET_INPUT_WINDOW = 2114
#: GENCODE 43 annotation, used to derive a TSS list for QC.
#:
#: Taken from the IGVF portal rather than from EBI, for a reason that matters:
#: IGVFFI9573KOZR is GENCODE 43 with its chromosome names converted to the
#: UCSC style "as used in genome reference fasta file", i.e. it is guaranteed
#: to match IGVFFI0653VCGH. A GTF whose contigs disagree with the genome and
#: chrom.sizes produces an empty TSS list and a silently meaningless
#: enrichment number. It also means the md5 can be verified the same way the
#: genome's is, from the same metadata API.
GENCODE_RELEASE = "43"
GENCODE_ACCESSION = "IGVFFI9573KOZR"
GENCODE_MD5 = "230008ce6a9bbc0320ac3b7d7e8fbf23"

#: MotifCompendium's human reference database, which 08.0's modisco report and
#: 09.0's annotation match against. Pinned to the commit of MotifCompendium
#: v1.0.19, the version the motif-compendium environment installs, rather than
#: to `main`: the file changed at v1.0.17 (v1.0.16's copy has md5
#: aa2314af189c1cbddc543e5d5c09a4f8), so a floating URL meant two installs of
#: "the same" pipeline could annotate against different databases. The local
#: file name carries the version for the same reason chrombpnet's does below:
#: an install made from `main` before the pin keeps its old unversioned name,
#: so it is neither overwritten nor mistaken for the pinned file.
MOTIF_DB_VERSION = "1.0.19"
MOTIF_DB_URL = (
    "https://raw.githubusercontent.com/kundajelab/MotifCompendium/"
    "7e9d1c2f932ea7f53fd36e174affef26eaf086c4/"
    "pipeline/data/MotifCompendium-Database-Human.meme.txt"
)
MOTIF_DB_MD5 = "b54441e0bfb9623345b6802763472d3c"

#: chrombpnet's own motif database -- the one its TF-MoDISco report matches
#: against. Unlike MotifCompendium it carries the assay-bias motifs (TN5_1..8,
#: DNASE_*) next to the TF ones, which is what lets the per-fold motif QC
#: (03.3/04.5, src/motif_qc.py) say how many seqlets are Tn5 at all:
#: annotated against MotifCompendium, a bias model's Tn5 seqlets are forced
#: onto their nearest TF motif. Fetched from the 1.0.1 release tag, which
#: cannot move, and pinned by md5. The file has not changed since: it is
#: byte-identical to chrombpnet/data/motifs.meme.txt as packaged by the
#: chrombpnet 2.x the pipeline runs (NNFC-GMD/chrombpnet at 7dfb285, compared
#: with cmp, 2026-09-24; unchanged at c913b65 and at the current pin 05f5748).
CHROMBPNET_MOTIFS_URL = (
    "https://raw.githubusercontent.com/kundajelab/chrombpnet/v1.0.1/chrombpnet/data/motifs.meme.txt"
)
CHROMBPNET_MOTIFS_MD5 = "30d5f17c169eb2ea0588c7546acfbf4d"


def main_chromosomes(folds_dir=None) -> list[str]:
    """The chromosomes the pipeline actually trains on.

    Derived from folds/*.json rather than hardcoded, because those files are
    what chrombpnet is given as -fl: a chromosome absent from every fold is
    never trained, validated or tested on, so signal there is dead weight.
    Today that is chr1-22, X, Y -- note chrM is NOT included, precisely because
    no fold mentions it.
    """
    import json as _json  # noqa: PLC0415

    d = Path(folds_dir) if folds_dir else Path(__file__).resolve().parents[3] / "folds"
    seen: set[str] = set()
    for f in sorted(d.glob("fold_*.json")):
        split = _json.loads(f.read_text())
        for key in ("train", "valid", "test"):
            seen.update(split.get(key, []))
    if not seen:
        raise FileNotFoundError(f"no fold definitions in {d}")

    def order(c):
        bare = c[3:] if c.startswith("chr") else c
        return (0, int(bare), "") if bare.isdigit() else (1, 0, bare)

    return sorted(seen, key=order)


def root() -> Path:
    return Path(os.environ.get("REFERENCE_ROOT") or DEFAULT_ROOT)


def layout(reference_root=None) -> dict:
    """Every reference path and URL, derived from one root."""
    r = Path(reference_root) if reference_root else root()
    genome_path = r / "hg38"
    sequence_dir = genome_path / "Sequence"
    blacklist_dir = genome_path / "blacklist"
    motif_dir = r / "motif"
    slop = CHROMBPNET_INPUT_WINDOW // 2
    return {
        "REFERENCE_ROOT": str(r),
        "genome_build": "hg38",
        "genome_path": str(genome_path),
        "sequence_dir": str(sequence_dir),
        "chrom_sizes_dir": str(sequence_dir / "chrom_sizes"),
        "blacklist_dir": str(blacklist_dir),
        "motif_dir": str(motif_dir),
        "genome_accession": GENOME_ACCESSION,
        "genome_md5": GENOME_MD5,
        "genome_url": (
            f"https://api.data.igvf.org/reference-files/{GENOME_ACCESSION}"
            f"/@@download/{GENOME_ACCESSION}.fasta.gz"
        ),
        "genome_metadata_url": (
            f"https://api.data.igvf.org/reference-files/{GENOME_ACCESSION}/?format=json"
        ),
        "genome_fa_gz": str(sequence_dir / f"{GENOME_ACCESSION}.fasta.gz"),
        "genome_fa": str(sequence_dir / f"{GENOME_ACCESSION}.fasta"),
        "genome_fa_alias": str(sequence_dir / "GCA_000001405.15_GRCh38_no_alt_analysis_set.fna"),
        "chrom_sizes": str(sequence_dir / "chrom_sizes" / "IGVF.DACC.GRCh38.chrom.sizes.tsv"),
        # The same file restricted to the chromosomes the folds use. Handing
        # this to the pileup makes the main-chromosome filter implicit: a cut
        # on a contig that is not listed is dropped because it is not listed.
        "chrom_sizes_main": str(
            sequence_dir / "chrom_sizes" / "IGVF.DACC.GRCh38.chrom.sizes.main.tsv"
        ),
        "blacklist_accession": BLACKLIST_ACCESSION,
        "blacklist_url": encode_file_url(BLACKLIST_ACCESSION),
        "blacklist_raw": str(blacklist_dir / f"{BLACKLIST_ACCESSION}.bed.gz"),
        "blacklist": str(blacklist_dir / "blacklist.bed.gz"),
        "chrombpnet_input_window": CHROMBPNET_INPUT_WINDOW,
        "blacklist_slop_bp": slop,
        "blacklist_slop": str(blacklist_dir / "blacklist_slop.bed.gz"),
        "ref_db_meme": str(
            motif_dir / f"MotifCompendium-{MOTIF_DB_VERSION}-Database-Human.meme.txt"
        ),
        "ref_db_meme_url": MOTIF_DB_URL,
        "ref_db_meme_md5": MOTIF_DB_MD5,
        "chrombpnet_motifs_meme": str(motif_dir / "chrombpnet-1.0.1.motifs.meme.txt"),
        "chrombpnet_motifs_url": CHROMBPNET_MOTIFS_URL,
        "chrombpnet_motifs_md5": CHROMBPNET_MOTIFS_MD5,
        "gencode_release": GENCODE_RELEASE,
        "gencode_accession": GENCODE_ACCESSION,
        "gencode_md5": GENCODE_MD5,
        "gencode_gtf_url": (
            f"https://api.data.igvf.org/reference-files/{GENCODE_ACCESSION}"
            f"/@@download/{GENCODE_ACCESSION}.gtf.gz"
        ),
        "gencode_metadata_url": (
            f"https://api.data.igvf.org/reference-files/{GENCODE_ACCESSION}/?format=json"
        ),
        "gencode_gtf": str(genome_path / "annotation" / f"{GENCODE_ACCESSION}.gtf.gz"),
        # Unique TSS positions, for the TSS-enrichment QC. Derived, not downloaded.
        "tss_bed": str(genome_path / "annotation" / f"gencode.v{GENCODE_RELEASE}.tss_unique.bed"),
        "annotation_dir": str(genome_path / "annotation"),
    }


def to_shell(values: dict | None = None) -> str:
    """Render the layout as bash assignments for lib/bash/references.sh to eval."""
    values = values if values is not None else layout()
    out = []
    for key, value in values.items():
        if key == "REFERENCE_ROOT":
            # Preserve an inherited value; this is the one knob.
            out.append(f'REFERENCE_ROOT="${{REFERENCE_ROOT:-{value}}}"')
        else:
            out.append(f"{key}={shlex.quote(str(value))}")
    return "\n".join(out)


# ── fetching ──────────────────────────────────────────────────────────────────


def _md5(path, chunk=8 << 20) -> str:
    h = hashlib.md5()  # noqa: S324  # provenance/integrity check, not a secret
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest, retries: int = 3, log=print) -> Path:
    """Fetch a URL to a path, atomically, with retries. Replaces `curl -fL --retry 3`."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:  # noqa: S310
                shutil.copyfileobj(resp, out, 8 << 20)
            tmp.replace(dest)  # only appears complete once it is
            return dest
        except (urllib.error.URLError, OSError) as exc:
            last = exc
            log(f"    attempt {attempt}/{retries} failed: {exc}")
    tmp.unlink(missing_ok=True)
    raise RuntimeError(f"could not download {url}: {last}")


def published_md5(metadata_url: str) -> str | None:
    """The md5 an IGVF metadata record advertises, or None if unreachable."""
    try:
        with urllib.request.urlopen(  # noqa: S310
            urllib.request.Request(metadata_url, headers={"Accept": "application/json"})
        ) as resp:
            return json.load(resp).get("md5sum")
    except Exception:  # noqa: BLE001  # best effort, as the shell version was
        return None


def tss_sites_from_gtf(gtf_path, wanted_chroms=None, feature="gene"):
    """Unique TSS positions from a GENCODE GTF, as 0-based (chrom, pos, strand, name).

    GTF is 1-based inclusive and BED is 0-based half-open, so a ``+`` feature's
    TSS is ``start - 1`` and a ``-`` feature's is ``end - 1``. Getting that
    wrong shifts every TSS by one base and quietly flattens the enrichment
    profile, so it is pinned by a test.

    Deduplicated: many genes share a TSS, and counting one twice would weight
    it twice in the aggregate profile.
    """
    seen: dict[tuple[str, int, str], str] = {}
    opener = gzip.open if str(gtf_path).endswith(".gz") else open
    with opener(gtf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != feature:
                continue
            chrom, start, end, strand, attrs = f[0], int(f[3]), int(f[4]), f[6], f[8]
            if wanted_chroms is not None and chrom not in wanted_chroms:
                continue
            pos = start - 1 if strand == "+" else end - 1
            key = (chrom, pos, strand)
            if key not in seen:
                m = re.search(r'gene_name "([^"]+)"', attrs)
                seen[key] = m.group(1) if m else "."
    return [(c, p, st, seen[(c, p, st)]) for c, p, st in sorted(seen, key=lambda t: (t[0], t[1]))]


def _faidx(fasta, log=print) -> Path:
    """Write <fasta>.fai. Replaces `samtools faidx`."""
    import pysam  # noqa: PLC0415  # only needed when actually indexing

    pysam.faidx(str(fasta))
    return Path(f"{fasta}.fai")


def _relative_symlink(target, link, log=print) -> None:
    """Link -> target by basename, so the tree stays relocatable."""
    link = Path(link)
    link.unlink(missing_ok=True)
    link.symlink_to(Path(target).name)


def _check_pinned_md5(path, pinned: str, what: str) -> None:
    """Raise unless ``path`` has the md5 this module pins for it."""
    actual = _md5(path)
    if actual != pinned:
        raise RuntimeError(
            f"{what} md5 {actual} does not match the pinned {pinned} ({path}). "
            "If it is an old or damaged copy, remove it and re-run "
            "`cli.py download-references` to fetch the pinned file."
        )


def _verify(path, pinned: str, metadata_url: str, what: str, log=print) -> None:
    """Check a download against the pinned md5, and against the portal's.

    The pinned value means an offline or firewalled machine still gets a real
    integrity check. The portal is consulted too: if it now advertises a
    different md5 the file has been revised upstream, and silently accepting
    that would mean two people running "the same" pipeline on different data.
    """
    if pinned:
        _check_pinned_md5(path, pinned, what)
    published = published_md5(metadata_url)
    if published is None:
        log(f"  {what} md5 matches the pinned value (portal unreachable)")
    elif published != pinned:
        raise RuntimeError(
            f"{what}: the portal now advertises md5 {published}, but this pipeline "
            f"pins {pinned}. The reference has been revised upstream -- update "
            "utils/references.py deliberately rather than silently changing data."
        )
    else:
        log(f"  {what} md5 verified (pinned and portal agree)")


def _fetch_pinned(url: str, dest, pinned: str, what: str, log=print) -> Path:
    """Download ``url`` to ``dest`` unless it is present, then check its md5.

    For files with no metadata API behind them (the GitHub-hosted motif
    databases), so the pinned md5 is the whole check. A file that is already
    present is checked as well: a copy fetched before the pin, or a damaged
    one, then fails here, once, rather than in whichever step reads it.
    """
    dest = Path(dest)
    if dest.is_file() and dest.stat().st_size:
        log(f"  {what} present")
    else:
        log(f"  downloading {what}")
        download(url, dest, log=log)
    _check_pinned_md5(dest, pinned, what)
    log(f"  {what} md5 verified (pinned)")
    return dest


def fetch_all(reference_root=None, log=print) -> dict:
    """Install every shared reference. Idempotent: present files are left alone."""
    ref = layout(reference_root)
    for key in ("sequence_dir", "chrom_sizes_dir", "blacklist_dir", "motif_dir", "annotation_dir"):
        Path(ref[key]).mkdir(parents=True, exist_ok=True)
    log(f"installing references under {ref['REFERENCE_ROOT']}")

    # 1. genome
    gz = Path(ref["genome_fa_gz"])
    if gz.is_file() and gz.stat().st_size:
        log("  genome fasta.gz present")
    else:
        log(f"  downloading genome ({ref['genome_accession']})")
        download(ref["genome_url"], gz, log=log)

    _verify(gz, ref["genome_md5"], ref["genome_metadata_url"], "genome", log)

    fa = Path(ref["genome_fa"])
    if not (fa.is_file() and fa.stat().st_size):
        log("  decompressing genome")
        with gzip.open(gz, "rb") as src, open(fa, "wb") as dst:
            shutil.copyfileobj(src, dst, 8 << 20)
    fai = Path(f"{fa}.fai")
    if not (fai.is_file() and fai.stat().st_size):
        log("  indexing genome")
        _faidx(fa, log=log)

    alias = Path(ref["genome_fa_alias"])
    _relative_symlink(fa, alias)
    _relative_symlink(fai, Path(f"{alias}.fai"))
    _relative_symlink(gz, Path(f"{alias}.gz"))

    # 2. chrom.sizes -- same contigs and lengths as the genome, from the .fai
    cs = Path(ref["chrom_sizes"])
    if cs.is_file() and cs.stat().st_size:
        log("  chrom.sizes present")
    else:
        log("  deriving chrom.sizes from the genome index")
        cs.write_text(
            "".join("\t".join(line.split("\t")[:2]) + "\n" for line in fai.read_text().splitlines())
        )

    # 2b. chrom.sizes restricted to the chromosomes the folds use
    cs_main = Path(ref["chrom_sizes_main"])
    if cs_main.is_file() and cs_main.stat().st_size:
        log("  main-chromosome chrom.sizes present")
    else:
        wanted = main_chromosomes()
        rows = [line for line in cs.read_text().splitlines() if line.split("\t")[0] in set(wanted)]
        missing = set(wanted) - {r.split("\t")[0] for r in rows}
        if missing:
            raise RuntimeError(
                f"chrom.sizes is missing chromosomes the folds require: {sorted(missing)}"
            )
        cs_main.write_text("\n".join(rows) + "\n")
        log(
            f"  wrote main-chromosome chrom.sizes ({len(rows)} of "
            f"{len(cs.read_text().splitlines())} contigs)"
        )
        # Provenance beside the file, since it is derived rather than downloaded.
        sidecar = cs_main.with_suffix(cs_main.suffix + ".json")
        sidecar.write_text(
            json.dumps(
                {
                    "derived_from": str(cs),
                    "derived_from_md5": _md5(cs),
                    "chromosomes": wanted,
                    "n_chromosomes": len(wanted),
                    "rule": "union of train/valid/test across folds/fold_*.json",
                    "md5": _md5(cs_main),
                },
                indent=2,
            )
            + "\n"
        )

    # 3. blacklist
    raw = Path(ref["blacklist_raw"])
    if raw.is_file() and raw.stat().st_size:
        log("  blacklist present")
    else:
        log(f"  downloading ENCODE blacklist ({ref['blacklist_accession']})")
        download(ref["blacklist_url"], raw, log=log)
    _relative_symlink(raw, ref["blacklist"])

    # 4. motif databases
    _fetch_pinned(
        ref["ref_db_meme_url"],
        ref["ref_db_meme"],
        ref["ref_db_meme_md5"],
        f"MotifCompendium {MOTIF_DB_VERSION} reference DB",
        log,
    )
    _fetch_pinned(
        ref["chrombpnet_motifs_url"],
        ref["chrombpnet_motifs_meme"],
        ref["chrombpnet_motifs_md5"],
        "chrombpnet motif DB (Tn5/DNase bias + TF motifs)",
        log,
    )

    # 5. TSS list for QC, derived from the GENCODE annotation
    tss = Path(ref["tss_bed"])
    if tss.is_file() and tss.stat().st_size:
        log("  TSS list present")
    else:
        gtf = Path(ref["gencode_gtf"])
        if not (gtf.is_file() and gtf.stat().st_size):
            log(
                f"  downloading GENCODE {ref['gencode_release']} annotation "
                f"({ref['gencode_accession']})"
            )
            download(ref["gencode_gtf_url"], gtf, log=log)
        _verify(gtf, ref["gencode_md5"], ref["gencode_metadata_url"], "GENCODE GTF", log)
        log("  deriving unique TSS positions")
        sites = tss_sites_from_gtf(gtf, set(main_chromosomes()))
        if not sites:
            raise RuntimeError(f"no TSS parsed from {gtf}")
        tss.write_text("".join(f"{c}\t{p}\t{p + 1}\t{g}\t0\t{st}\n" for c, p, st, g in sites))
        log(f"  wrote {len(sites)} unique TSS positions")

    log(f"done. references under {ref['REFERENCE_ROOT']}")
    return ref


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1 or argv[0] not in {"export", "json"}:
        print(f"usage: {Path(__file__).name} {{export|json}}", file=sys.stderr)
        return 2
    print(to_shell() if argv[0] == "export" else json.dumps(layout(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
