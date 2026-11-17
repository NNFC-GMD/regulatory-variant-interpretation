#!/bin/bash
# shellcheck disable=SC2218  # false positive in shellcheck 0.11.0: helpers come from common.sh
#SBATCH --job-name=preprocess_nonpeaks
#SBATCH --mem=100G
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --partition=normal,engreitz
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 01.0.preprocess_nonpeaks.sh
# Purpose: GC-matched background regions, one set per fold, via
#          `chrombpnet prep nonpeaks`.
#
# This one step deliberately runs ChromBPNet's own code rather than a port.
# The negatives ARE training data: they are half of what every model in 03 and
# 04 sees. A reimplementation that differed -- by a rounding step, or by the
# order it consumes its RNG -- would silently change every model trained
# afterwards, and the difference would surface as unexplained metric drift
# months later. The bigwig conversion in 00.0 is a different case: its output
# is verifiable byte-for-byte against ChromBPNet's, so a faster path there is
# free. Here it is not worth the risk.
#
# What ChromBPNet does, per fold (all inside ${out_prefix}_auxiliary/):
#   1. bin the genome into `inputlen` windows every `stride` bp and score GC
#   2. slop peaks and blacklist by inputlen/2, sort, merge -> exclude.bed
#   3. bedtools intersect -v -> candidates.bed
#   4. sample, for each peak, unused candidates from the same fold pool whose
#      GC matches, walking +/-0.01 until a bucket has one free
#
# Step 1 is a Python loop over ~3M windows and dominates the runtime, and it is
# redone for every fold because the auxiliary directory is per-prefix. That is
# the cost of using their code unchanged; it is CPU-only and runs once per
# dataset, so it is paid here rather than on a GPU node.
#
# Seeded: `-s 1234` by default, recorded in the metadata. Same inputs and same
# seed give the same negatives.
#
# Output (per fold, under ${data_path}/${dataset}/):
#   output_${peak_type}_fold_${fold}_negatives.bed   10-col, summit at col 10
#   output_${peak_type}_fold_${fold}_auxiliary/      intermediates, kept
#
# Usage:
#   export DATASET=<name>
#   cd workflows/SLURM && sbatch 01.0.preprocess_nonpeaks.sh
#
# Prerequisites: 00.1.preprocess_peaks.sh, and references installed
#   (`cli.py download-references`) for the genome, chrom.sizes and blacklist.

# --- bootstrap: locate the repo root (identical block in every workflow step) --
# sbatch copies the submitted script to a node-local spool dir, so BASH_SOURCE
# does not point into the repo; SLURM_SUBMIT_DIR (the cwd at submit time) does.
# Walk up from whichever is usable until lib/bash/common.sh turns up. Override
# with REPO_ROOT if you submit from outside the checkout.
# An inherited REPO_ROOT is only trusted if it really is this repo -- the name
# is generic enough that another project may have exported it.
[[ -n "${REPO_ROOT}" && -f "${REPO_ROOT}/lib/bash/common.sh" ]] || REPO_ROOT=""
if [[ -z "${REPO_ROOT}" ]]; then
    for _d in "${SLURM_SUBMIT_DIR}" "$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"; do
        while [[ -n "${_d}" && "${_d}" != "/" ]]; do
            [[ -f "${_d}/lib/bash/common.sh" ]] && { REPO_ROOT="${_d}"; break 2; }
            _d="$(dirname "${_d}")"
        done
    done
fi
[[ -n "${REPO_ROOT}" ]] || {
    echo "ERROR: cannot locate the repo root. Submit from inside the checkout, or:" >&2
    echo "  export REPO_ROOT=/path/to/this/checkout" >&2
    exit 1
}
export REPO_ROOT
# --- end bootstrap -------------------------------------------------------------

# shellcheck source=lib/bash/config.sh
source "${REPO_ROOT}/lib/bash/config.sh" || exit 1

set -euo pipefail

dataset="${datasets[0]}"
peaks_np="${peaks_dir}/${dataset}_${peak_type}_peaks_no_blacklist.narrowPeak"
out_dir="${data_path}/${dataset}"

metadata_start "01.0.preprocess_nonpeaks"
metadata_inputs+=( "peaks=${peaks_np}" "genome=${genome_fa}" )
metadata_inputs+=( "chrom_sizes=${chrom_sizes}" "blacklist=${blacklist}" )
metadata_params+=( "peak_type=${peak_type}" "folds=${folds[*]}" )
# Recorded even though they are ChromBPNet's defaults: these four values, with
# the inputs, are what reproduces a negatives file. A default that changes in a
# future ChromBPNet release would otherwise be invisible in the run record.
metadata_params+=( "seed=${nonpeak_seed}" "neg_to_pos_ratio_train=${neg_to_pos_ratio}" )
metadata_params+=( "inputlen=${chrombpnet_input_window}" "stride=${nonpeak_stride}" )

require_input "${peaks_np}" 00.1.preprocess_peaks.sh
require_input "${genome_fa}"   "cli.py download-references"
# The FULL chrom.sizes, not the main-chromosome one: the blacklist carries
# non-main contigs, `bedtools slop` errors on a contig it cannot find, and the
# fold JSON already confines the sampling to main chromosomes.
require_input "${chrom_sizes}" "cli.py download-references"
require_input "${blacklist}"   "cli.py download-references"
for fold in "${folds[@]}"; do
    require_input "${folds_dir}/fold_${fold}.json" ""
    metadata_inputs+=( "fold=${folds_dir}/fold_${fold}.json" )
    metadata_outputs+=( "negatives=${out_dir}/output_${peak_type}_fold_${fold}_negatives.bed" )
done
preflight_check

# ChromBPNet's env, not preprocess: `prep nonpeaks` shells out to bedtools
# slop/sort/merge/intersect, which that env pins.
activate_env "${chrombpnet_env}"

# The negatives are reproducible only from the seed AND the chrombpnet that
# consumed it. No probe is needed: the run record is written from this step's
# environment, so it lists chrombpnet's version and, because 2.x reports
# 2.0.0.dev0 at every commit, the commit its checkout is at
# (software_versions: chrombpnet, chrombpnet_commit). The probe that used to be
# here called `chrombpnet --version`, which no chrombpnet has, and always
# recorded "unknown".

# `-o` is a path PREFIX whose directory ChromBPNet does not create -- its own
# help says "make sure it exists".
mkdir -p "${out_dir}"

for fold in "${folds[@]}"; do
    out_prefix="${out_dir}/output_${peak_type}_fold_${fold}"
    negatives_file="${out_prefix}_negatives.bed"

    if [[ -f "${negatives_file}" ]]; then
        echo "[$(date)] fold_${fold}: negatives already exist, skipping"
        continue
    fi

    # ChromBPNet calls os.makedirs(prefix + "_auxiliary/", exist_ok=False), so a
    # job killed mid-run (the GC scan is long) leaves the directory behind and
    # every retry then dies on FileExistsError before doing any work. Clear it:
    # we only get here when the negatives themselves are absent, so nothing
    # anyone wants is being discarded.
    if [[ -d "${out_prefix}_auxiliary" ]]; then
        echo "[$(date)] fold_${fold}: clearing stale ${out_prefix}_auxiliary from a previous run"
        rm -rf "${out_prefix}_auxiliary"
    fi

    echo "[$(date)] fold_${fold}: generating GC-matched negatives (genome-wide GC scan is slow)"
    chrombpnet prep nonpeaks \
        -g  "${genome_fa}" \
        -p  "${peaks_np}" \
        -c  "${chrom_sizes}" \
        -fl "${folds_dir}/fold_${fold}.json" \
        -br "${blacklist}" \
        -il "${chrombpnet_input_window}" \
        -st "${nonpeak_stride}" \
        -npr "${neg_to_pos_ratio}" \
        -s  "${nonpeak_seed}" \
        -o  "${out_prefix}"

    echo "[$(date)] fold_${fold}: $(wc -l < "${negatives_file}") negatives -> ${negatives_file}"
done

echo "[$(date)] Done. Next: 02.0.qc_training_data.sh compares the signal at these"
echo "           background regions against the signal at the peaks."
