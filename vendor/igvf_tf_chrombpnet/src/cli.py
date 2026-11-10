#!/usr/bin/env python3
"""``igvf-tf`` — one command, one subcommand per pipeline tool.

    python src/cli.py --help
    python src/cli.py preprocess-peaks --help

The workflow steps in ``workflows/SLURM/`` call these subcommands. The CLI never
reads the pipeline config: a step sources ``lib/bash/config.sh`` first and passes
the resolved paths as options, so the same command works by hand, from sbatch,
and (later) from Nextflow without a second config reader.

Conversion is partial and deliberately so. Natively Click: the tools written
during the bedtools -> pyranges port. Still standalone argparse scripts, listed
by ``legacy`` below: select_bias_model.py, qc_full_model.py, qc_datasets.py,
average_contrib_scores.py, contribs_to_bigwig.py, predict_and_avg.py,
predict_bias_metrics.py, run_bias_qc.py, motif_compendium.py. They work exactly
as before; converting select_bias_model.py alone is 1152 lines and buys nothing
until someone needs it.
"""

import os
import sys
from pathlib import Path

# Make lib/python importable without an install step (works under the cluster
# conda envs, under pixi, and under a bare python).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib" / "python"))

import json as _json  # noqa: E402

import click  # noqa: E402
import numpy as np  # noqa: E402

from utils import config as cfg  # noqa: E402
from utils import (  # noqa: E402
    intervals,
    log,
    metadata,
    palettes,
    peakcall,
    pileup,
    plotting,
    qc,
    references,
    shift,
)

logger = log.get_logger(__name__)


def _optional_version(dist: str):
    """Version of an installed distribution, or None. Never raises."""
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _v

    try:
        return _v(dist)
    except PackageNotFoundError:
        return None


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"

LEGACY_SCRIPTS = [
    "average_contrib_scores.py",
    "contribs_to_bigwig.py",
    "motif_compendium.py",
    "predict_and_avg.py",
    "predict_bias_metrics.py",
    "qc_datasets.py",
    "qc_full_model.py",
    "run_bias_qc.py",
    "select_bias_model.py",
]

DEFAULT_INPUT_WINDOW = 2114

verbose_opt = click.option("-v", "--verbose", is_flag=True, help="Debug-level logging.")
quiet_opt = click.option("-q", "--quiet", is_flag=True, help="Warnings and errors only.")


def _setup_logging(verbose, quiet):
    log.setup(level="WARNING" if quiet else None, verbose=verbose)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name=None, version="0.1.0", prog_name="igvf-tf")
def cli():
    """ChromBPNet pipeline tools for the IGVF TF collaboration."""


# ── preprocess-peaks ──────────────────────────────────────────────────────────


@cli.command("preprocess-peaks")
@click.option(
    "--peaks",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Input peak BED (summit = midpoint), or with --summit-window a 10-column narrowPeak.",
)
@click.option(
    "--summit-window",
    type=int,
    default=None,
    help="Read --peaks as a narrowPeak and re-centre every row on its OWN summit: "
    "[summit - w/2, summit + w/2). One window per row, so multi-summit regions give "
    "several. Off by default (plain BED, midpoint summit).",
)
@click.option(
    "--max-qvalue",
    type=float,
    default=None,
    help="With --summit-window: keep rows with q <= this (column 9 read as -log10 q, "
    "as MACS2 writes it). Off by default.",
)
@click.option(
    "--blacklist",
    required=True,
    help="ENCODE accession (e.g. ENCFF356LFX), URL, or local BED(.gz).",
)
@click.option(
    "--chrom-sizes",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="2-column chrom.sizes TSV; the slop is clamped to these lengths.",
)
@click.option(
    "--input-window",
    type=int,
    default=DEFAULT_INPUT_WINDOW,
    show_default=True,
    help="ChromBPNet input window (bp). The blacklist is extended by HALF this "
    "on each side, so a peak is dropped when the window the model reads "
    "overlaps a blacklist region.",
)
@click.option(
    "--signal",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Prepared bigwig from 00.0. Required for --min-signal-quantile.",
)
@click.option(
    "--min-signal-quantile",
    type=float,
    default=None,
    help="Drop peaks whose signal falls below this quantile of the experiment's "
    "own genome-wide windows. Depth-independent, because both sides are the "
    "same signal. Off by default.",
)
@click.option(
    "--compare-window",
    type=int,
    default=1000,
    show_default=True,
    help="Window for the background quantile and the peak signal compared "
    "against it. SAME NAME AND SAME CONFIG VALUE (qc_compare_window) as 02.0's, "
    "on purpose: both measure 'peak signal', and two independently settable "
    "knobs would let them disagree. Defaults to chrombpnet's outputlen, which "
    "is the window the bias threshold is computed over.",
)
@click.option(
    "--background-sample",
    type=int,
    default=50_000,
    show_default=True,
    help="Genome windows sampled to estimate the background quantile.",
)
@click.option(
    "--out-dir", required=True, type=click.Path(file_okay=False), help="Directory for both outputs."
)
@click.option("--prefix", required=True, help="Output basename stem, e.g. igvf3_cardiomyocyte_all.")
@click.option(
    "--metadata-dir",
    default=None,
    type=click.Path(file_okay=False),
    help="Run-metadata directory  [default: <out-dir>/../metadata]",
)
@verbose_opt
@quiet_opt
def preprocess_peaks(
    peaks,
    summit_window,
    max_qvalue,
    blacklist,
    chrom_sizes,
    input_window,
    signal,
    min_signal_quantile,
    compare_window,
    background_sample,
    out_dir,
    prefix,
    metadata_dir,
    verbose,
    quiet,
):
    """Blacklist-filter peaks and write chrombpnet's narrowPeak.

    Replaces `bedtools intersect -v` + awk. The blacklist is slopped by half the
    input window first; equivalence to bedtools is pinned by tests/test_intervals.py.
    """
    _setup_logging(verbose, quiet)
    if input_window <= 0:
        raise click.BadParameter("must be positive", param_hint="--input-window")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = metadata_dir or (out_dir.parent / "metadata")

    with metadata.record("preprocess_peaks", dataset=prefix, out_dir=meta_dir) as md:
        md.add_param("input_window", input_window)
        md.add_param("slop_bp", input_window // 2)
        md.add_param("blacklist_source", references.describe_source(blacklist))
        md.add_input("peaks", peaks)
        md.add_input("chrom_sizes", chrom_sizes)
        if Path(blacklist).exists():
            md.add_input("blacklist", blacklist)

        summit_dropped = {"qvalue": 0, "before_chrom_start": 0}
        if summit_window is None:
            if max_qvalue is not None:
                raise click.UsageError("--max-qvalue needs --summit-window")
            peaks_pr = intervals.read_bed(peaks)
            n_in = len(peaks_pr)
            logger.info(f"{n_in} peaks in from {peaks}")
        else:
            try:
                peaks_pr = intervals.read_narrowpeak(peaks)
                n_in = len(peaks_pr)
                peaks_pr, summit_dropped = intervals.summit_windows(
                    peaks_pr, summit_window, max_qvalue
                )
            except ValueError as exc:
                raise click.ClickException(str(exc)) from exc
            md.add_param("summit_window", summit_window)
            md.add_param("max_qvalue", max_qvalue)
            logger.info(
                f"{n_in} narrowPeak rows in from {peaks}; {summit_dropped['qvalue']} above "
                f"q={max_qvalue}, {summit_dropped['before_chrom_start']} too close to a "
                f"chromosome start; {len(peaks_pr)} {summit_window}bp summit windows"
            )
            md.add_metric("peaks_above_max_qvalue", summit_dropped["qvalue"])
        logger.info(f"blacklist: {references.describe_source(blacklist)}")
        try:
            bl = intervals.read_bed3(blacklist)
        except OSError as exc:
            raise click.ClickException(str(exc)) from exc
        logger.info(f"{len(bl)} blacklist regions")

        slop_bp = input_window // 2
        chromsizes = intervals.read_chromsizes(chrom_sizes)

        # bedtools slop warns and skips intervals on contigs the genome file does
        # not list; clip_ranges would raise instead, so drop them here with a count.
        known = bl["Chromosome"].isin(chromsizes.keys())
        if not known.all():
            missing = sorted(set(bl.loc[~known, "Chromosome"]))
            logger.warning(
                f"skipping {(~known).sum()} blacklist interval(s) on {len(missing)} "
                f"contig(s) absent from chrom.sizes: {', '.join(missing[:5])}"
                + (" ..." if len(missing) > 5 else "")
            )
            bl = bl[known]

        bl_raw_intervals = list(
            zip(bl["Chromosome"], bl["Start"], bl["End"])
        )  # un-slopped, for the background sampler
        bl = intervals.slop(bl, slop_bp, chromsizes)
        logger.info(f"blacklist extended +/-{slop_bp}bp (half the {input_window}bp window)")

        # Keep peaks on the same contigs the signal covers. With the
        # main-chromosome chrom.sizes this drops scaffolds and chrM, which the
        # bigwig has no signal for.
        peaks_pr, off_contig = intervals.restrict_to_chromosomes(peaks_pr, chromsizes)
        if off_contig:
            logger.warning("%d peak(s) dropped: on contigs absent from %s", off_contig, chrom_sizes)
        md.add_metric("peaks_off_contig", off_contig)

        n_after_contig = len(peaks_pr)
        kept = intervals.remove_blacklisted(peaks_pr, bl)
        n_after_blacklist = len(kept)
        md.add_metric("peaks_in", len(peaks_pr))
        md.add_metric("peaks_kept", len(kept))
        md.add_metric("peaks_dropped", len(peaks_pr) - len(kept))
        logger.info(
            f"{len(peaks_pr) - len(kept)} peak(s) hit the slopped blacklist, {len(kept)} kept"
        )

        # chrombpnet would drop these silently at contribs time; do it here so
        # the peak set is stable from this step onward.
        kept, overhanging = intervals.drop_windows_off_chromosome(kept, chromsizes, input_window)
        if overhanging:
            logger.warning(
                "%d peak(s) dropped: their %dbp model window runs off a chromosome end",
                overhanging,
                input_window,
            )
        md.add_metric("peaks_window_overhang", overhanging)

        if len(kept) == 0:
            raise click.ClickException(
                "every peak was filtered out. Check that the peak file and the "
                "blacklist use the same chromosome naming (chr1 vs 1)."
            )

        # ── signal floor ─────────────────────────────────────────────────
        # A "peak" whose signal is below the level ordinary genome windows
        # reach is not a peak. It matters far beyond its own row: chrombpnet
        # anchors EVERY bias threshold to quantile(peak_counts, 0.01), so a
        # handful of background-level peaks drag the whole sweep down. On d0
        # the weakest 1% of peaks sat at the 40.6th percentile of genome
        # windows -- weaker than most of the genome -- and dropping 1.8% of
        # peaks moved q01 from 4 to 17.
        n_before_floor = len(kept)
        dropped_pr = None
        floor_diag = {}
        signal_floor = None
        peak_signal = None
        if min_signal_quantile is not None:
            if not signal:
                raise click.ClickException("--min-signal-quantile needs --signal")
            if not 0.0 <= min_signal_quantile < 1.0:
                raise click.BadParameter("must be in [0, 1)", param_hint="--min-signal-quantile")
            md.add_input("signal", signal)
            md.add_param("min_signal_quantile", min_signal_quantile)
            md.add_param("compare_window", compare_window)

            signal_floor, floor_diag, bg = qc.background_signal_quantile(
                signal,
                min_signal_quantile,
                compare_window,
                blacklist_intervals=bl_raw_intervals,
                n_sample=background_sample,
            )
            logger.info(
                "background q%g over %dbp = %.1f insertions (%d windows sampled)",
                min_signal_quantile * 100,
                compare_window,
                signal_floor,
                floor_diag["background_n_sampled"],
            )
            peak_signal = qc.window_totals(
                signal,
                intervals.to_narrowpeak(kept).values.tolist(),
                window=compare_window,
                max_regions=10_000_000,
            )
            keep_mask = np.asarray(peak_signal >= signal_floor)
            dropped_pr = kept[~keep_mask]
            kept = kept[keep_mask]
            logger.info(
                "%d peak(s) dropped below the floor, %d kept",
                n_before_floor - len(kept),
                len(kept),
            )
            md.add_metric("peaks_below_signal_floor", n_before_floor - len(kept))
            md.add_metric("signal_floor", signal_floor)
            for k, v in floor_diag.items():
                md.add_metric(k, v)
            if len(kept) == 0:
                raise click.ClickException(
                    f"every peak fell below the signal floor ({signal_floor}). "
                    "Lower --min-signal-quantile, or check that --signal is this "
                    "dataset's bigwig."
                )

        # Only the narrowPeak is written. A plain 3-column BED used to be
        # written beside it, but nothing ever read it -- every downstream step
        # takes the narrowPeak -- and narrowPeak IS a BED6+4, so `cut -f1-3`
        # reproduces the BED exactly. On d0 that was 3.7 MB duplicated per
        # dataset, and one more artifact to explain.
        # peaks/ of its own, beside signal/. The two are the step-00 products
        # and each gets a directory and a sidecar, so a later step can read
        # what it is being given without re-deriving it.
        peaks_dir = out_dir / "peaks"
        peaks_dir.mkdir(parents=True, exist_ok=True)
        np_out = peaks_dir / f"{prefix}_peaks_no_blacklist.narrowPeak"
        intervals.to_narrowpeak(kept).to_csv(np_out, sep="\t", header=False, index=False)
        md.add_output("peaks", np_out)
        logger.info(f"-> {np_out}")

        # Sidecar: every filter this peak set went through, and what each one
        # cost. Without it the only record of "why are there 153,347 peaks and
        # not 156,236" is in a log that nothing keeps.
        sidecar = {
            "peaks_source": str(Path(peaks).resolve()),
            "input_window": input_window,
            "blacklist_source": references.describe_source(blacklist),
            "blacklist_slop_bp": slop_bp,
            "summit_window": summit_window,
            "filters": [
                {"filter": "input", "kept": n_in},
                {"filter": "qvalue", "dropped": summit_dropped["qvalue"], "max_qvalue": max_qvalue},
                {"filter": "before_chrom_start", "dropped": summit_dropped["before_chrom_start"]},
                {"filter": "off_contig", "dropped": off_contig},
                {"filter": "blacklist_slopped", "dropped": n_after_contig - n_after_blacklist},
                {"filter": "window_overhang", "dropped": overhanging},
                {
                    "filter": "signal_floor",
                    "dropped": n_before_floor - len(kept),
                    "quantile": min_signal_quantile,
                    "threshold": signal_floor,
                    "window": compare_window if min_signal_quantile is not None else None,
                },
            ],
            "peaks_final": len(kept),
            **floor_diag,
        }
        # The dropped peaks are kept as an artifact, not just a count. They do
        # not vanish from the analysis: dropping a peak removes it from the
        # exclusion list 01.0 builds, so the region becomes ELIGIBLE to be
        # sampled as GC-matched background. Some of it comes back as a
        # negative and the bias model then trains on regions we just judged
        # too weak to be peaks. 02.0 measures how much; it needs this file to
        # do so.
        if dropped_pr is not None and len(dropped_pr):
            dropped_bed = peaks_dir / f"{prefix}_peaks_dropped.bed"
            dropped_pr[["Chromosome", "Start", "End"]].to_csv(
                dropped_bed, sep="\t", header=False, index=False
            )
            md.add_output("peaks", dropped_bed)
            logger.info(f"-> {dropped_bed} ({len(dropped_pr)} dropped)")

        sidecar_path = peaks_dir / "peaks.json"
        sidecar_path.write_text(_json.dumps(sidecar, indent=2) + "\n")
        md.add_output("peaks", sidecar_path)
        logger.info(f"-> {sidecar_path}")

        # QC plot, in plots/ with the other figures rather than beside the data.
        if peak_signal is not None:
            import matplotlib.pyplot as plt

            plots_dir = out_dir.parent / "plots" / "peaks_qc"
            plots_dir.mkdir(parents=True, exist_ok=True)
            plotting.apply_style(font_size=10)
            _sc = palettes.BIAS_SCAN_COLORS
            fig, ax = plt.subplots(figsize=(5, 3.2))
            # Clip to the peak q90: the peak tail runs to ~800 and squashes
            # the part that matters -- the background mass, the floor, and
            # where the two distributions separate -- into the first 5% of
            # the axis.
            hi = float(np.quantile(peak_signal, 0.90))
            hi = max(hi, signal_floor * 4)
            bins = np.linspace(0, hi, 70)
            ax.hist(
                bg,
                bins=bins,
                density=True,
                alpha=0.55,
                color=_sc["nonpeaks"],
                label="genome windows (sampled)",
            )
            ax.hist(
                peak_signal, bins=bins, density=True, alpha=0.55, color=_sc["peaks"], label="peaks"
            )
            ax.axvline(
                signal_floor,
                color=_sc["fail"],
                lw=1.6,
                label=f"floor = background q{min_signal_quantile * 100:g} ({signal_floor:.0f})",
            )
            ax.set_xlim(0, hi)
            ax.set_xlabel(
                f"insertions per {compare_window}bp window  "
                f"(upper {100 * float((peak_signal > hi).mean()):.0f}% of peaks clipped)"
            )
            ax.set_ylabel("density")
            ax.set_title(
                f"{prefix}: {n_before_floor - len(kept)} of {n_before_floor} peaks "
                f"below background q{min_signal_quantile * 100:g}",
                fontsize=8.5,
            )
            ax.legend(frameon=False, fontsize=7)
            fig.tight_layout()
            plotting.save_fig(fig, plots_dir / f"{prefix}_peak_signal_floor")
            plt.close(fig)
            md.add_output("qc", plots_dir / f"{prefix}_peak_signal_floor.pdf")


# ── filter-fragments ──────────────────────────────────────────────────────────

MAIN_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"]


@cli.command("filter-fragments")
@click.option(
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Fragments TSV(.gz).",
)
@click.option("--output", "output_path", required=True, help="Filtered fragments TSV.gz (bgzip).")
@click.option(
    "--chroms",
    multiple=True,
    default=tuple(MAIN_CHROMS),
    help="Chromosomes to keep  [default: chr1-22, chrX, chrY, chrM]",
)
@click.option("--index", is_flag=True, help="Also write a tabix .tbi (input must be sorted).")
@click.option(
    "--metadata-dir",
    default=None,
    type=click.Path(file_okay=False),
    help="Run-metadata directory  [default: <output dir>/metadata]",
)
@verbose_opt
@quiet_opt
def filter_fragments(input_path, output_path, chroms, index, metadata_dir, verbose, quiet):
    """Filter an ATAC fragments TSV to the main chromosomes, bgzipped."""
    _setup_logging(verbose, quiet)
    import gzip

    from utils import compression

    meta_dir = metadata_dir or (Path(output_path).parent / "metadata")
    with metadata.record("filter_fragments", out_dir=meta_dir) as md:
        md.add_input("fragments", input_path)
        md.add_param("chroms", ",".join(chroms))
        md.add_param("index", index)

        keep = {c.encode() for c in chroms}
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        opener = gzip.open if str(input_path).endswith(".gz") else open

        kept = total = 0
        with opener(input_path, "rb") as fin, compression.open_write(output_path) as fout:
            while chunk := fin.readlines(1 << 22):
                out = []
                for line in chunk:
                    total += 1
                    if line[:1] != b"#" and line.split(b"\t", 1)[0] in keep:
                        out.append(line)
                kept += len(out)
                fout.write(b"".join(out))  # BGZFile has no writelines()

        md.add_metric("fragments_in", total)
        md.add_metric("fragments_kept", kept)
        md.add_output("fragments", output_path)

        if kept == 0:
            raise click.ClickException(
                f"no fragments kept from {total} line(s). Check the chromosome "
                f"naming in {input_path} (chr1 vs 1)."
            )
        logger.info(f"kept {kept} of {total} fragment(s) -> {output_path}")

        if index:
            tbi = compression.tabix_index(output_path, preset="bed")
            md.add_output("fragments", tbi)
            logger.info(f"-> {tbi}")


# ── prepare-bigwig (B) ────────────────────────────────────────────────────────


@cli.command("prepare-bigwig")
@click.option("--signal-path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--signal-type", required=True, type=click.Choice(["fragments", "bam", "tagalign"]))
@click.option("--assay", default="ATAC", type=click.Choice(["ATAC", "DNASE"]))
@click.option(
    "--genome",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Reference FASTA. Needed ONLY to detect the Tn5 shift, which reads "
    "sequence around cut sites. Not needed at all if you give "
    "--plus-shift/--minus-shift: the pileup itself never uses it.",
)
@click.option("--chrom-sizes", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--out-dir",
    required=True,
    type=click.Path(file_okay=False),
    help="Prepared-bigwig directory; 03.0/04.0 pass its data_unstranded.bw to chrombpnet as -bw.",
)
@click.option(
    "--write-filtered",
    default=None,
    type=click.Path(),
    help="Also write the main-chromosome rows here, in the SAME pass (bgzipped "
    "if .gz). Only needed for the fallback where chrombpnet reads the reads "
    "itself; omit it and nothing is rewritten.",
)
@click.option(
    "--plus-shift",
    type=int,
    default=None,
    help="Tn5 shift already present in the reads, plus strand. Give this with "
    "--minus-shift to skip detection entirely -- then no FASTA is needed, "
    "because the pileup does not read sequence.",
)
@click.option(
    "--minus-shift",
    type=int,
    default=None,
    help="As --plus-shift, minus strand.",
)
@click.option(
    "--num-samples",
    type=int,
    default=10000,
    show_default=True,
    help="Reads sampled for Tn5 shift detection (only used when detecting).",
)
@click.option(
    "--engine",
    type=click.Choice(["numpy", "figwig"]),
    default="numpy",
    show_default=True,
    help="Who counts the cut sites. numpy: utils/pileup.py (pandas reader, "
    "pybigtools writer). figwig: `figwig bam2bw -f -u`, threaded BGZF reading; "
    "fragments only, same per-base values, no --write-filtered.",
)
@click.option(
    "--threads",
    type=int,
    default=8,
    show_default=True,
    help="Cores for --engine figwig (its gains end near 8).",
)
@click.option("--metadata-dir", default=None, type=click.Path(file_okay=False))
@verbose_opt
@quiet_opt
def prepare_bigwig(
    signal_path,
    signal_type,
    assay,
    genome,
    chrom_sizes,
    out_dir,
    write_filtered,
    plus_shift,
    minus_shift,
    num_samples,
    engine,
    threads,
    metadata_dir,
    verbose,
    quiet,
):
    """Do chrombpnet's reads->bigwig conversion once, on CPU.

    Every `chrombpnet train` / `bias train` starts with this conversion plus an
    enzyme-shift detection pass, unconditionally and on the GPU node. A 5-fold x
    4-factor sweep repeats it 20 times. Run this once on CPU; 03.0/04.0 hand the
    bigwig to chrombpnet with -bw, which then skips its own conversion.

    Writes data_unstranded.bw plus a sidecar recording the signal file, its md5
    and the assay. src/chrombpnet_train.py refuses to train when the sidecar does
    not match the configured signal (the recorded chrombpnet version is
    informational; nothing compares it).
    """
    _setup_logging(verbose, quiet)
    import json as _json

    if engine == "figwig":
        # figwig reads a BED/tsv as chrom/start/end with no strand, which is a
        # fragments file's meaning but not a tagAlign's; a BAM it would count
        # per read, but the numpy path already covers BAMs.
        if signal_type != "fragments":
            raise click.UsageError("--engine figwig takes fragments only")
        if write_filtered:
            raise click.UsageError(
                "--engine figwig does not write filtered rows; drop --write-filtered "
                "(set filter_main_chroms: false in the config)"
            )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta_dir = metadata_dir or (out / "metadata")

    with metadata.record("prepare_bigwig", out_dir=meta_dir) as md:
        # The parameter_name is the data type itself -- fragments, bam or
        # tagalign -- not a generic "reads". signal_type is validated to be
        # one of those three, and it is what the file actually holds.
        md.add_input(signal_type, signal_path)
        if genome:
            md.add_input("genome", genome)
        md.add_param("signal_type", signal_type)
        md.add_param("assay", assay)
        md.add_param("num_samples", num_samples)

        # ── 1. shift detection: ours, no chrombpnet ─────────────────────
        # utils.shift (vendored scPrinter) reports the shift ALREADY PRESENT in
        # the reads; shift_deltas then adjusts to chrombpnet's +4/-4 target.
        if plus_shift is None or minus_shift is None:
            if genome is None:
                raise click.UsageError(
                    "--genome is required to detect the Tn5 shift (it reads sequence "
                    "around cut sites). Either pass it, or pass --plus-shift and "
                    "--minus-shift and no FASTA is needed."
                )
            logger.info("detecting the Tn5 shift already present in the reads")
            # No signal_type here: detect_shift_raw infers single- vs paired-end
            # itself, from whether the 6th column of the sampled reads is a
            # strand. Passing it was a TypeError -- shift.py has no such
            # parameter, so this path had never been executed.
            plus_shift, minus_shift = shift.detect_shift_raw(signal_path, genome)
        logger.info("shift in the reads: %+d/%+d", plus_shift, minus_shift)

        # Adjust from the detected shift to chrombpnet's target: +4/-4 (ATAC),
        # 0/+1 (DNASE). NOT scPrinter's +4/-5 -- see utils.shift.
        plus_delta, minus_delta = shift.shift_deltas(plus_shift, minus_shift, assay)
        logger.info("applying delta %+d/%+d", plus_delta, minus_delta)
        md.add_param("plus_shift", plus_shift)
        md.add_param("minus_shift", minus_shift)
        md.add_param("plus_delta", plus_delta)
        md.add_param("minus_delta", minus_delta)

        # ── 2. pileup: ours, not chrombpnet's two external sorts ─────────────
        md.add_param("engine", engine)
        chromsizes = intervals.read_chromsizes(chrom_sizes)
        if engine == "figwig":
            # Contigs absent from chrom.sizes are dropped by figwig itself, so
            # the main-chromosome filter is implicit here too.
            md.add_param("threads", threads)
            logger.info("counting cut sites with figwig bam2bw on %d core(s)", threads)
            pileup.figwig_bigwig(
                signal_path,
                chrom_sizes,
                out / "data_unstranded.bw",
                plus_delta,
                minus_delta,
                threads,
            )
            # Every fragment on a listed contig gives two cuts, so this total is
            # 2 x fragments kept: the number to check nothing was dropped.
            import pyBigWig

            handle = pyBigWig.open(str(out / "data_unstranded.bw"))
            try:
                cuts_total = int(round(handle.header()["sumData"]))
            finally:
                handle.close()
            md.add_metric("cuts_total", cuts_total)
            logger.info("%d cut sites written", cuts_total)
        else:
            logger.info("filtering to the main chromosomes and counting cut sites (one pass)")
            if signal_type == "bam":
                # One cut per mapped read at its own strand's 5' end, as
                # `bedtools bamtobed` would emit; --write-filtered does not apply.
                if write_filtered:
                    raise click.UsageError(
                        "--write-filtered applies to fragments/tagalign, not BAM"
                    )
                cuts, skipped, kept = pileup.collect_cuts_bam(
                    signal_path, chromsizes, plus_delta, minus_delta
                )
            else:
                cuts, skipped, kept = pileup.collect_cuts(
                    signal_path, chromsizes, plus_delta, minus_delta, write_filtered=write_filtered
                )
            md.add_metric("reads_kept", kept)
            if write_filtered:
                # Filtering does not change what the data IS.
                md.add_output(signal_type, write_filtered)
                logger.info("filtered reads -> %s", write_filtered)
            if skipped:
                logger.warning(
                    "skipped %d read(s) on %d contig(s) absent from chrom.sizes: %s",
                    sum(skipped.values()),
                    len(skipped),
                    ", ".join(sorted(skipped)[:5]),
                )
            logger.info("writing %s", out / "data_unstranded.bw")
            pileup.write_bigwig(out / "data_unstranded.bw", chromsizes, cuts)

        bw = out / "data_unstranded.bw"
        if not bw.is_file():
            raise click.ClickException(f"conversion produced no {bw}")

        sidecar = {
            "signal_path": str(Path(signal_path).resolve()),
            "signal_md5": metadata.md5sum(signal_path),
            "signal_type": signal_type,
            "assay": assay,
            # Recorded when chrombpnet happens to be installed, but preprocessing
            # does not import it, so its absence is not an error.
            "chrombpnet_version": _optional_version("chrombpnet"),
            # Which code produced the pileup, so a record says so.
            "pileup": engine,
            "figwig_version": _optional_version("figwig") if engine == "figwig" else None,
            "shift_detection": "utils.shift (scPrinter)",
            "plus_shift": int(plus_shift),
            "minus_shift": int(minus_shift),
        }
        (out / "prepared_bigwig.json").write_text(_json.dumps(sidecar, indent=2) + "\n")
        md.add_output("signal", bw)
        md.add_output("signal", out / "prepared_bigwig.json")
        logger.info("-> %s  (03.0/04.0 pass it to chrombpnet as -bw)", bw)


# ── call-peaks ────────────────────────────────────────────────────────────────


@cli.command("call-peaks")
@click.option("--fragments", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--chrom-sizes",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="FULL chrom.sizes: fragments on contigs it lists go to MACS (chrM included, "
    "as in the pseudobulking pipeline); 00.1 later keeps the main chromosomes.",
)
@click.option("--blacklist", required=True, help="Blacklist BED(.gz), path or ENCODE accession.")
@click.option("--output", required=True, help="Final narrowPeak (.gz -> bgzipped).")
@click.option("--assay", default="ATAC", type=click.Choice(["ATAC", "DNASE"]))
@click.option(
    "--plus-shift",
    type=int,
    default=None,
    help="Tn5 shift already present in the fragments, plus strand (as 00.0).",
)
@click.option("--minus-shift", type=int, default=None, help="As --plus-shift, minus strand.")
@click.option(
    "--genome",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="FASTA, only to detect the shift when --plus-shift/--minus-shift are not given.",
)
@click.option(
    "--macs-input",
    type=click.Choice(["bed", "frag"]),
    default="bed",
    show_default=True,
    help="bed: the pseudobulking pipeline's 1-bp insertion records (-f BED). "
    "frag: 150-bp insertion windows through MACS3 PR #756's fast FRAG parser -- "
    "faster to read, but NOT the same call (paired-end local lambda); see "
    "utils/peakcall.py.",
)
@click.option("--top-n", type=int, default=peakcall.TOP_N, show_default=True)
@click.option("--min-overlap", type=float, default=peakcall.MIN_OVERLAP, show_default=True)
@click.option("--seed", type=int, default=peakcall.SEED, show_default=True)
@click.option("--bedgraph", is_flag=True, help="Also pass -B --SPMR (bedGraphs; peaks unchanged).")
@click.option(
    "--tmp-dir",
    required=True,
    type=click.Path(file_okay=False),
    help="Scratch for the pseudoreplicate files (tens of GB per dataset); removed after.",
)
@click.option("--threads", type=int, default=4, show_default=True, help="bgzip -dc threads.")
@click.option("--metadata-dir", default=None, type=click.Path(file_okay=False))
@verbose_opt
@quiet_opt
def call_peaks(
    fragments,
    chrom_sizes,
    blacklist,
    output,
    assay,
    plus_shift,
    minus_shift,
    genome,
    macs_input,
    top_n,
    min_overlap,
    seed,
    bedgraph,
    tmp_dir,
    threads,
    metadata_dir,
    verbose,
    quiet,
):
    """Call peaks with the igvf_pseudobulking_pipeline recipe on MACS3 PR #756.

    Pseudoreplicates -> 3 MACS3 calls -> top-N per rep -> repT rows reproduced
    in rep1 and rep2 -> blacklist. See utils/peakcall.py for the recipe and
    the deliberate departures from the original, all recorded in
    call_peaks.json beside --output.
    """
    _setup_logging(verbose, quiet)
    import shutil
    import time

    from utils import compression

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    reps_dir = out.parent / "reps"
    reps_dir.mkdir(exist_ok=True)
    prefix = out.name.split(".")[0]
    meta_dir = metadata_dir or (out.parent / "metadata")

    with metadata.record("call_peaks", dataset=prefix, out_dir=meta_dir) as md:
        md.add_input("fragments", fragments)
        md.add_input("chrom_sizes", chrom_sizes)
        md.add_param("blacklist_source", references.describe_source(blacklist))
        for name, value in (
            ("macs_input", macs_input),
            ("top_n", top_n),
            ("min_overlap", min_overlap),
            ("seed", seed),
            ("bedgraph", bedgraph),
        ):
            md.add_param(name, value)

        # Insertions sit where chrombpnet's bigwig puts them (+4/-4), from the
        # shift actually in the fragments -- not a fixed +4/-5 on top of it.
        if plus_shift is None or minus_shift is None:
            if genome is None:
                raise click.UsageError(
                    "give --plus-shift/--minus-shift, or --genome to detect the shift"
                )
            logger.info("detecting the Tn5 shift already present in the fragments")
            plus_shift, minus_shift = shift.detect_shift_raw(fragments, genome)
        plus_delta, minus_delta = shift.shift_deltas(plus_shift, minus_shift, assay)
        logger.info(
            "shift in the fragments %+d/%+d -> insertions at start%+d, end%+d-1",
            plus_shift, minus_shift, plus_delta, minus_delta,
        )  # fmt: skip
        md.add_param("plus_shift", plus_shift)
        md.add_param("minus_shift", minus_shift)

        scratch = Path(tmp_dir) / f"call_peaks.{prefix}.{os.getpid()}"
        scratch.mkdir(parents=True, exist_ok=False)
        try:
            ext = "frag" if macs_input == "frag" else "bed"
            rep1, rep2 = scratch / f"rep1.{ext}", scratch / f"rep2.{ext}"
            t0 = time.monotonic()
            kept, skipped = peakcall.split_insertions(
                fragments, chrom_sizes, rep1, rep2, plus_delta, minus_delta,
                macs_input=macs_input, seed=seed, threads=threads,
            )  # fmt: skip
            split_s = round(time.monotonic() - t0, 1)
            logger.info(
                "%d fragments -> pseudoreplicates in %.0fs (%d on unlisted contigs dropped)",
                kept, split_s, skipped,
            )  # fmt: skip
            md.add_metric("fragments_kept", kept)
            md.add_metric("fragments_skipped_contig", skipped)
            md.add_metric("split_seconds", split_s)

            names = {"rep1": [rep1], "rep2": [rep2], "repT": [rep1, rep2]}
            commands = {
                rep: peakcall.macs_command(f"{prefix}.{rep}", files, reps_dir, macs_input, bedgraph)
                for rep, files in names.items()
            }
            logger.info("running 3 macs3 callpeak at once: %s", " ".join(commands["repT"]))
            macs_stats = peakcall.run_concurrently(commands, reps_dir)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        for rep, st in macs_stats.items():
            md.add_metric(f"macs_{rep}_seconds", st["wall_s"])
            md.add_metric(f"macs_{rep}_max_rss_gb", st["max_rss_gb"])

        counts = {}
        tops = {}
        for rep in names:
            raw = peakcall.read_narrowpeak(reps_dir / f"{prefix}.{rep}_peaks.narrowPeak")
            tops[rep] = peakcall.top_n_by_pvalue(raw, top_n)
            counts[f"{rep}_raw"] = len(raw)
            counts[f"{rep}_top"] = len(tops[rep])
        kept_peaks = peakcall.reproducible(tops["repT"], tops["rep1"], tops["rep2"], min_overlap)
        counts["reproducible"] = len(kept_peaks)
        bl = intervals.read_bed3(blacklist)
        final = peakcall.finalize(kept_peaks, bl, intervals.read_chromsizes(chrom_sizes))
        counts["final"] = len(final)
        for name, n in counts.items():
            md.add_metric(f"peaks_{name}", n)
        logger.info("peaks: %s", ", ".join(f"{k}={v}" for k, v in counts.items()))
        if final.empty:
            raise click.ClickException("no peaks survived; see the counts above")

        # Written under a temporary name that keeps the suffix (write_tsv
        # bgzips by it), then renamed: a killed run leaves no file that the
        # step's skip check would take as finished.
        partial = out.with_name(f".partial.{out.name}")
        compression.write_tsv(final, partial)
        os.replace(partial, out)
        md.add_output("peaks", out)

        sidecar = {
            "fragments": str(Path(fragments).resolve()),
            "fragments_md5": metadata.md5sum(fragments),
            "macs3_version": _optional_version("macs3"),
            "macs3_source": "jmschrei/MACS@fa52988 (macs3-project/MACS PR #756)",
            "macs_input": macs_input,
            "macs_flags": {rep: cmd[cmd.index("-n") + 2 :] for rep, cmd in commands.items()},
            "plus_shift": int(plus_shift),
            "minus_shift": int(minus_shift),
            "insertion": f"start{plus_delta:+d}, end{minus_delta:+d}-1 (chrombpnet +4/-4)",
            "pseudoreplicate_seed": seed,
            "top_n": top_n,
            "min_overlap": min_overlap,
            "blacklist": references.describe_source(blacklist),
            "counts": counts,
            "macs": macs_stats,
            "recipe": "kundajelab/igvf_pseudobulking_pipeline v2.0.1 (e8774b2)",
            "deviations_from_igvf_pseudobulking_pipeline": [
                "insertions follow the shift present in the fragments (+4/-4 like the "
                "bigwig); the original adds +4/-5 to already-shifted fragments",
                "pseudoreplicates from one seeded stream (deterministic)",
                "no -B --SPMR (bedGraph tracks only; peaks unaffected)",
            ]
            + (
                [
                    "insertions given to MACS as 150-bp FRAG windows with --max-gap 1 "
                    "--min-length 150, in place of 1-bp -f BED records"
                ]
                if macs_input == "frag"
                else []
            ),  # fmt: skip
        }
        sidecar_path = out.parent / "call_peaks.json"
        sidecar_path.write_text(_json.dumps(sidecar, indent=2) + "\n")
        md.add_output("peaks", sidecar_path)
        logger.info("-> %s (%d peaks)", out, len(final))


# ── download-references ───────────────────────────────────────────────────────


@cli.command("download-references")
@click.option("--dataset", default=None, help="Dataset name under config/.")
@click.option("--path", default=None, type=click.Path(), help="Explicit config.yaml.")
@click.option(
    "--reference-root",
    default=None,
    type=click.Path(file_okay=False),
    help="Where to install  [default: $REFERENCE_ROOT]",
)
@click.option("--metadata-dir", default=None, type=click.Path(file_okay=False))
@verbose_opt
@quiet_opt
def download_references(dataset, path, reference_root, metadata_dir, verbose, quiet):
    """Fetch the shared genome, chrom.sizes, blacklist and motif DB.

    Run once per cluster. Idempotent: files already present are left alone, and
    the genome is checked against the md5 IGVF publishes.

    Replaces `cli.py download-references`. Needs no curl and no samtools:
    downloads go through urllib and the FASTA index through pysam. The paths it
    writes come from utils.references, the same module the pipeline reads them
    from, so the two cannot disagree.
    """
    _setup_logging(verbose, quiet)
    if reference_root is None and (dataset or path):
        reference_root = cfg.load(_config_path(dataset, path)).get("reference_root")
    ref = references.layout(reference_root)
    meta_dir = metadata_dir or (Path(ref["REFERENCE_ROOT"]) / "metadata")

    with metadata.record("download_references", out_dir=meta_dir) as md:
        md.add_param("reference_root", ref["REFERENCE_ROOT"])
        md.add_param("genome_accession", ref["genome_accession"])
        md.add_param("blacklist_accession", ref["blacklist_accession"])
        try:
            references.fetch_all(reference_root, log=logger.info)
        except RuntimeError as exc:
            raise click.ClickException(str(exc)) from exc
        for role in (
            "genome_fa",
            "chrom_sizes",
            "chrom_sizes_main",
            "blacklist",
            "ref_db_meme",
            "chrombpnet_motifs_meme",
        ):
            md.add_output(role, ref[role])
        md.add_output("chrom_sizes", ref["chrom_sizes_main"] + ".json")
        md.add_param("main_chromosomes", ",".join(references.main_chromosomes()))


# ── qc-signal ─────────────────────────────────────────────────────────────────


@cli.command("qc-signal")
@click.option(
    "--bigwig",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Prepared signal bigwig (00.0 output).",
)
@click.option(
    "--peaks",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Filtered narrowPeak (00.1 output).",
)
@click.option(
    "--negatives",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="GC-matched negatives BED (01.0 output). Enables the comparative QC.",
)
@click.option(
    "--compare-window",
    type=int,
    default=1000,
    show_default=True,
    help="Fixed window, centred on each summit, for comparing peaks to negatives. "
    "Defaults to ChromBPNet's output window, the span its counts head predicts.",
)
@click.option(
    "--tss",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="TSS BED for the enrichment metric. Skipped if not given.",
)
@click.option(
    "--input-window",
    type=int,
    default=2114,
    show_default=True,
    help="Profile spans +/- half this around peak summits.",
)
@click.option(
    "--min-insertions",
    type=int,
    default=20_000_000,
    show_default=True,
    help="Depth below which ChromBPNet is unlikely to train well.",
)
@click.option("--out-dir", required=True, type=click.Path(file_okay=False))
@click.option("--prefix", required=True)
@click.option("--metadata-dir", default=None, type=click.Path(file_okay=False))
@verbose_opt
@quiet_opt
def qc_signal(
    bigwig,
    peaks,
    negatives,
    compare_window,
    tss,
    input_window,
    min_insertions,
    out_dir,
    prefix,
    metadata_dir,
    verbose,
    quiet,
):
    """QC the signal and peaks about to be handed to ChromBPNet.

    Reads the two artifacts training will actually use -- the prepared bigwig
    and the filtered narrowPeak -- so every number describes what the model
    sees, after all filtering.

    With --negatives it also runs the comparative half: peaks against their own
    GC-matched background, over a fixed window so the two are measured on equal
    footing. That is the question training actually poses, asked before any GPU
    time is spent on it.

    Writes <prefix>_signal_qc.json (all metrics), <prefix>_signal_qc.tsv (the
    flat ones, for DuckDB alongside the run metadata) and the profile plots.
    """
    _setup_logging(verbose, quiet)
    import json as _json

    import pandas as pd

    from utils import plotting, qc

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta_dir = metadata_dir or (out.parent / "metadata")

    with metadata.record("qc_signal", dataset=prefix, out_dir=meta_dir) as md:
        md.add_input("signal", bigwig)
        md.add_input("peaks", peaks)
        if negatives:
            md.add_input("negatives", negatives)
        if tss:
            md.add_input("tss", tss)

        peak_rows = pd.read_csv(peaks, sep="\t", header=None, dtype={0: str}).values.tolist()
        logger.info("%d peaks", len(peak_rows))

        metrics = {"dataset": prefix}
        metrics |= qc.signal_summary(bigwig)
        # signal_summary first: it supplies genome_bases, which turns the peak
        # coverage into a fraction.
        metrics |= qc.peak_width_summary(
            peak_rows, input_window=input_window, genome_bases=metrics.get("genome_bases")
        )
        metrics |= qc.peak_signal_distribution(bigwig, peak_rows)

        # Fraction of all insertions that land in peaks -- the FRiP of the
        # signal chrombpnet sees, rather than of the original fragments.
        if metrics.get("total_insertions"):
            metrics["frac_insertions_in_peaks"] = (
                metrics["insertions_in_peaks"] / metrics["total_insertions"]
            )
        metrics["enough_depth_for_chrombpnet"] = bool(
            metrics.get("total_insertions", 0) >= min_insertions
        )

        offsets, profile, used = qc.aggregate_profile(bigwig, peak_rows, flank=input_window // 2)
        metrics["n_peaks_profiled"] = used
        metrics["peak_profile_centre_over_flank"] = (
            float(profile[len(profile) // 2] / profile[:100].mean())
            if used and profile[:100].mean() > 0
            else None
        )

        # ── comparative: peaks vs their GC-matched background ─────────────
        neg_rows, neg_profile, cmp_metrics = None, None, {}
        if negatives:
            neg_rows = pd.read_csv(negatives, sep="\t", header=None, dtype={0: str}).values.tolist()
            logger.info("%d negatives", len(neg_rows))
            cmp_metrics, pos_totals, neg_totals = qc.peak_vs_nonpeak_signal(
                bigwig, peak_rows, neg_rows, window=compare_window
            )
            metrics |= cmp_metrics
            _o, neg_profile, neg_used = qc.aggregate_profile(
                bigwig, neg_rows, flank=input_window // 2
            )
            metrics["n_nonpeaks_profiled"] = neg_used

            # Did what 00.1 discarded come back as background?
            # Derive it from the peaks FILENAME, not from --prefix: 00.1 is
            # called with prefix "<dataset>_<peak_type>" and 02.0 with
            # "<dataset>", so composing the name here silently missed the file
            # and reported nothing.
            _dropped_bed = Path(
                str(peaks).replace("_peaks_no_blacklist.narrowPeak", "_peaks_dropped.bed")
            )
            metrics |= qc.dropped_peaks_resampled(_dropped_bed, neg_rows, compare_window)
            if metrics.get("n_dropped_peaks_resampled"):
                logger.info(
                    "%d of %d peaks dropped by the signal floor were re-sampled as "
                    "background (%d negatives, %.3f%% of the background)",
                    metrics["n_dropped_peaks_resampled"],
                    metrics["n_peaks_dropped_by_floor"],
                    metrics["n_negatives_in_dropped_peaks"],
                    100 * metrics["frac_negatives_in_dropped_peaks"],
                )

            # Which bias_threshold_factor values 03.0 can actually train on.
            # Everything needed is already loaded here, and the answer costs
            # CPU minutes instead of one failed GPU job per bad factor.
            bias_metrics, bias_rows, _bias_pk, _bias_ng = qc.bias_threshold_viability(
                bigwig, peak_rows, neg_rows, outputlen=compare_window
            )
            metrics |= bias_metrics
            if bias_rows:
                _bias_tsv = out / f"{prefix}_bias_threshold_scan.tsv"
                pd.DataFrame(bias_rows).to_csv(_bias_tsv, sep="\t", index=False)
                md.add_output("qc", _bias_tsv)
                logger.info(
                    "bias_threshold_factor: %d viable, %d distinct training sets",
                    bias_metrics.get("n_bias_factors_viable", 0),
                    bias_metrics.get("n_bias_factors_distinct", 0),
                )
                logger.info(
                    "  distinct factors worth sweeping: %s",
                    bias_metrics.get("bias_factors_distinct"),
                )
                if bias_metrics.get("bias_factor_recommended") is not None:
                    logger.info(
                        "  RECOMMENDED: %s (%s) -- the largest factor whose background "
                        "stays at or below q01=%g, the weakest 1%% of peaks",
                        bias_metrics["bias_factor_recommended"],
                        bias_metrics["bias_factor_recommended_suffix"],
                        bias_metrics["peak_signal_q01"],
                    )
        else:
            logger.warning("no --negatives given; skipping the peak vs background QC")

        tss_metrics = {}
        if tss:
            tss_rows = pd.read_csv(tss, sep="\t", header=None, dtype={0: str}).values.tolist()
            tss_metrics = qc.tss_enrichment(bigwig, tss_rows)
            metrics["tss_enrichment"] = tss_metrics.get("tss_enrichment")
            metrics["n_tss_used"] = tss_metrics.get("n_tss_used")
        else:
            logger.warning("no --tss given; skipping TSS enrichment")

        # ── report ────────────────────────────────────────────────────────
        for key in (
            "total_insertions",
            "n_peaks",
            "frac_insertions_in_peaks",
            "frac_peaks_zero_signal",
            "peak_profile_centre_over_flank",
            "tss_enrichment",
            "auroc_peaks_vs_nonpeaks",
            "signal_enrichment_peak_over_nonpeak",
            "frac_nonpeaks_above_peak_median",
        ):
            if metrics.get(key) is not None:
                logger.info("  %-32s %s", key, metrics[key])
        if not metrics["enough_depth_for_chrombpnet"]:
            logger.warning(
                "only %.1fM insertions; ChromBPNet usually needs >= %.0fM",
                metrics.get("total_insertions", 0) / 1e6,
                min_insertions / 1e6,
            )
        auc = metrics.get("auroc_peaks_vs_nonpeaks")
        if auc is not None and auc < 0.80:
            logger.warning(
                "signal separates peaks from GC-matched background with AUROC %.3f. "
                "Below ~0.8 the peaks and the signal disagree, and ChromBPNet has "
                "little to learn -- check they come from the same sample before "
                "spending GPU time.",
                auc,
            )
        if metrics.get("frac_peaks_zero_signal", 0) > 0.01:
            logger.warning(
                "%.1f%% of peaks have NO signal under them -- peaks and signal may "
                "not come from the same sample",
                100 * metrics["frac_peaks_zero_signal"],
            )

        json_out = out / f"{prefix}_signal_qc.json"
        json_out.write_text(
            _json.dumps({**metrics, "tss": tss_metrics, "peaks_vs_nonpeaks": cmp_metrics}, indent=2)
            + "\n"
        )
        tsv_out = out / f"{prefix}_signal_qc.tsv"
        flat = {k: v for k, v in metrics.items() if not isinstance(v, list | dict)}
        pd.DataFrame([flat]).to_csv(tsv_out, sep="\t", index=False)
        md.add_output("qc", json_out)
        md.add_output("qc", tsv_out)
        # Most of what qc-signal writes is MEASURED, but not all of it: the
        # flat dict also carries the window the comparison used (a setting)
        # and the dataset name (already a top-level field on every record).
        # Classifying the whole dict as metrics put a setting and an identifier
        # in the measurement bag.
        QC_SETTINGS = {"compare_window"}
        QC_NOT_A_PARAMETER = {"dataset"}
        for k, v in flat.items():
            if k in QC_NOT_A_PARAMETER:
                continue
            if k in QC_SETTINGS:
                md.add_param(k, v)
            else:
                md.add_metric(k, v)

        # ── plots ─────────────────────────────────────────────────────────
        import matplotlib.pyplot as plt

        plotting.apply_style(font_size=10)
        from utils.palettes import OKABE_ITO

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(offsets, profile, color=OKABE_ITO["blue"], lw=1.2, label=f"peaks (n={used})")
        if neg_profile is not None:
            # Same axes on purpose: the gap between the two curves IS the
            # signal ChromBPNet has to work with.
            ax.plot(
                offsets,
                neg_profile,
                color=OKABE_ITO["orange"],
                lw=1.2,
                label=f"GC-matched background (n={metrics['n_nonpeaks_profiled']})",
            )
            ax.legend(frameon=False, fontsize=7)
        ax.set_xlabel("distance from summit (bp)")
        ax.set_ylabel("mean insertions per base")
        ax.set_title(f"{prefix}: signal at peaks (n={used})")
        plotting.save_fig(fig, out / f"{prefix}_profile_peaks")
        plt.close(fig)
        md.add_output("profile_peaks", out / f"{prefix}_profile_peaks.pdf")

        if cmp_metrics.get("auroc_peaks_vs_nonpeaks") is not None:
            import numpy as _np

            fig, ax = plt.subplots(figsize=(4, 3))
            bins = _np.histogram_bin_edges(
                _np.log1p(_np.concatenate([pos_totals, neg_totals])), bins=60
            )
            ax.hist(
                _np.log1p(neg_totals),
                bins=bins,
                color=OKABE_ITO["orange"],
                alpha=0.65,
                label="GC-matched background",
            )
            ax.hist(
                _np.log1p(pos_totals),
                bins=bins,
                color=OKABE_ITO["blue"],
                alpha=0.65,
                label="peaks",
            )
            ax.set_xlabel(f"log1p(insertions in {compare_window}bp window)")
            ax.set_ylabel("regions")
            ax.set_title(
                f"{prefix}: AUROC = {cmp_metrics['auroc_peaks_vs_nonpeaks']:.3f}",
            )
            ax.legend(frameon=False, fontsize=7)
            plotting.save_fig(fig, out / f"{prefix}_peaks_vs_background")
            plt.close(fig)
            md.add_output("peaks_vs_background", out / f"{prefix}_peaks_vs_background.pdf")

        # ── how the bias threshold reshapes the background ────────────────
        if bias_rows:
            _sc = palettes.BIAS_SCAN_COLORS
            fig, (axl, axr) = plt.subplots(1, 2, figsize=(8.4, 3.2))

            # LEFT: WHY the staircase exists. Non-peak counts are integers,
            # so the bars sit at 0, 1, 2 ... and a cutoff landing anywhere in
            # the gap between two bars keeps exactly the same set. Only the
            # cutoffs that actually change the training set are drawn -- one
            # per distinct set -- because drawing all forty is a picket fence.
            _hi = 10
            _bins = np.arange(0, _hi + 1)
            _frac = np.array([(np.asarray(_bias_ng) == b).mean() for b in _bins])
            axl.bar(_bins, _frac, width=0.8, color=_sc["nonpeaks"], label="non-peak regions")

            _seen, _marks = set(), []
            for _r in bias_rows:
                if _r["n_nonpeaks"] and _r["n_nonpeaks"] not in _seen:
                    _seen.add(_r["n_nonpeaks"])
                    _marks.append(_r)
            for _r in _marks:
                if _r["counts_threshold"] > _hi:
                    continue
                axl.axvline(_r["counts_threshold"], color=_sc["ok"], lw=1.1, ls="--")
                axl.text(
                    _r["counts_threshold"],
                    max(_frac) * 1.02,
                    f"{_r['factor']:g}",
                    fontsize=6.5,
                    rotation=90,
                    color=_sc["ok"],
                    va="bottom",
                    ha="center",
                )
            # Every cutoff is q01 * factor, so this is the anchor the whole
            # sweep is scaled from.
            axl.axvline(
                bias_metrics["peak_signal_q01"],
                color=_sc["peaks"],
                lw=1.6,
                label=f"peak q01 = {bias_metrics['peak_signal_q01']:g}  (x factor)",
            )

            axl.set_xticks(_bins)
            axl.set_xlabel(f"insertions per non-peak ({compare_window}bp window)")
            axl.set_ylabel("fraction of non-peaks")
            axl.set_title("counts are integers, so cutoffs land in the gaps", fontsize=8.5)
            axl.set_ylim(0, max(_frac) * 1.22)
            axl.legend(frameon=False, fontsize=6.5, loc="upper right")

            # RIGHT: the landscape itself -- how many non-peaks survive.
            _f = [r["factor"] for r in bias_rows]
            _n = [r["n_nonpeaks"] for r in bias_rows]
            _c = [
                _sc["fail"]
                if r["n_nonpeaks"] == 0
                else _sc["risky"]
                if r["n_nonpeaks"] < 1000
                else _sc["ok"]
                for r in bias_rows
            ]
            axr.step(_f, _n, where="post", color=_sc["cutoff"], lw=0.8, zorder=1)
            axr.scatter(_f, _n, c=_c, s=18, zorder=2)
            axr.set_xlabel("bias_threshold_factor")
            axr.set_ylabel("non-peaks left to train on")
            axr.set_title("staircase, not a ramp", fontsize=9)
            for _x in (0.5, 0.8):
                axr.axvline(_x, color=OKABE_ITO["black"], lw=0.5, ls=":", alpha=0.6)
            fig.tight_layout()
            plotting.save_fig(fig, out / f"{prefix}_bias_threshold_scan")
            plt.close(fig)
            md.add_output("qc", out / f"{prefix}_bias_threshold_scan.pdf")

        if tss_metrics.get("profile"):
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.plot(
                tss_metrics["profile_offsets"],
                tss_metrics["profile"],
                color=OKABE_ITO["vermillion"],
                lw=1.2,
            )
            ax.axhline(1.0, color=OKABE_ITO["black"], lw=0.6, ls="--")
            ax.set_xlabel("distance from TSS (bp)")
            ax.set_ylabel("enrichment over flanks")
            ax.set_title(f"{prefix}: TSS enrichment = {tss_metrics['tss_enrichment']:.1f}")
            plotting.save_fig(fig, out / f"{prefix}_profile_tss")
            plt.close(fig)
            md.add_output("profile_tss", out / f"{prefix}_profile_tss.pdf")


# ── config ────────────────────────────────────────────────────────────────────


@cli.group("config")
def config_group():
    """Inspect a dataset's config.yaml."""


def _config_path(dataset, path):
    if path:
        return Path(path)
    if not dataset:
        raise click.UsageError("give --dataset NAME or --path config.yaml")
    return REPO_ROOT / "config" / dataset / "config.yaml"


@config_group.command("show")
@click.option("--dataset", default=None, help="Dataset name under config/.")
@click.option("--path", default=None, type=click.Path(), help="Explicit config.yaml.")
def config_show(dataset, path):
    """Print the parsed config as JSON."""
    import json as _json

    click.echo(_json.dumps(cfg.load(_config_path(dataset, path)), indent=2))


@config_group.command("export")
@click.option("--dataset", default=None, help="Dataset name under config/.")
@click.option("--path", default=None, type=click.Path(), help="Explicit config.yaml.")
def config_export(dataset, path):
    """Print the config as shell assignments (what lib/bash/config.sh evals)."""
    click.echo(cfg.to_shell(cfg.load(_config_path(dataset, path))))


@config_group.command("validate")
@click.option("--dataset", default=None, help="Dataset name under config/.")
@click.option("--path", default=None, type=click.Path(), help="Explicit config.yaml.")
def config_validate(dataset, path):
    """Check a config parses, renders to shell, and has the required keys."""
    target = _config_path(dataset, path)
    try:
        parsed = cfg.load(target)
        cfg.to_shell(parsed)
    except cfg.ConfigError as exc:
        raise click.ClickException(str(exc)) from exc

    required = [
        "datasets",
        "bias_dataset",
        "peak_type",
        "folds",
        "regions",
        "signal_path",
        "signal_type",
        "output_dir",
    ]
    missing = [k for k in required if not parsed.get(k)]
    if missing:
        raise click.ClickException(f"{target}: missing required key(s): {', '.join(missing)}")

    if len(parsed.get("bias_factors") or []) != len(parsed.get("bias_suffixes_sweep") or []):
        raise click.ClickException(
            f"{target}: bias_factors and bias_suffixes_sweep must be the same length "
            "(they are positional)"
        )
    if parsed.get("signal_type") not in {"fragments", "bam", "tagalign"}:
        raise click.ClickException(
            f"{target}: signal_type must be fragments, bam or tagalign "
            f"(got {parsed.get('signal_type')!r}). chrombpnet cannot train from a bigwig."
        )
    if parsed.get("assay", "ATAC") not in {"ATAC", "DNASE"}:
        raise click.ClickException(f"{target}: assay must be ATAC or DNASE")

    sweep_folds = parsed.get("bias_sweep_folds") or parsed["folds"]
    n_tasks = len(sweep_folds) * len(parsed.get("bias_factors") or [1])
    click.echo(f"{target}: OK")
    click.echo(f"  signal     : {parsed['signal_type']} -> {parsed['signal_path']}")
    click.echo(f"  output     : {parsed['output_dir']}")
    click.echo(f"  folds      : {len(parsed['folds'])}")
    click.echo(
        f"  03.0 sweep : {len(sweep_folds)} fold(s) x "
        f"{len(parsed.get('bias_factors') or [])} factor(s)  ->  --array=0-{n_tasks - 1}"
    )
    if parsed.get("signal_type") == "bam":
        click.echo(
            "  note       : run 00.0.prepare_signal.sh to normalise the BAM to tagAlign once on CPU"
        )


# ── legacy ────────────────────────────────────────────────────────────────────


@cli.command("legacy")
def legacy():
    """List the tools that are still standalone argparse scripts."""
    click.echo("Not yet Click subcommands — run these directly:\n")
    for name in LEGACY_SCRIPTS:
        click.echo(f"  python {SRC.relative_to(REPO_ROOT)}/{name} --help")


if __name__ == "__main__":
    cli()
