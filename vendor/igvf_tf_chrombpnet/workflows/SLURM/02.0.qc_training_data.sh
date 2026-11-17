#!/bin/bash
# shellcheck disable=SC2218  # false positive in shellcheck 0.11.0: helpers come from common.sh
#SBATCH --job-name=qc_training_data
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=2:00:00
#SBATCH --partition=normal,engreitz
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 02.0.qc_training_data.sh
# Purpose: QC the assembled TRAINING SET before committing GPU time.
#
# Named for what only this step can do. 00.1 already QCs the peaks -- it has
# to, since it filters them -- so a second step called "qc_signal_peaks" read
# as redundant. What is NOT available until here is the BACKGROUND: the
# GC-matched negatives come from 01.0, and the TSS list from
# download-references. Every distinctive check below needs one of those:
# auroc_peaks_vs_nonpeaks, the GC-match comparison, the bias-threshold scan
# and tss_enrichment. This is the first point at which signal, peaks and
# background all exist, which is the only point where the training set can be
# judged as a whole.
#
# Runs on the three artifacts training will actually use -- the prepared bigwig
# from 00.0, the filtered narrowPeak from 00.1 and the GC-matched negatives
# from 01.0 -- so the numbers describe what ChromBPNet will see, after every
# filter. Two halves:
#
# Individual, "is this good data?"
#   TSS enrichment          is this accessibility data at all?
#   profile at peak summits does the signal agree with the peaks?
#   peaks, widths, coverage how much is there
#   signal per peak         quantiles, and how many peaks have NO signal
#   fraction in peaks       signal-to-background for the model's regions
#   total insertions        against the depth ChromBPNet needs
#
# Comparative, "is there anything to learn?" -- peaks against their own
# GC-matched background, measured over one fixed window so the two sides get
# the same number of bases:
#   AUROC                   how well signal alone separates the two classes
#   enrichment              mean signal at peaks over mean at background
#   frac background >= peak median   background that looks as open as a peak
#   overlaid profile + distribution plots
#
# The comparative half is why this step runs AFTER 01 rather than before it.
# Separating peaks from GC-matched background is exactly the task ChromBPNet
# is being trained on, so measuring it first says whether the dataset is worth
# training on at all -- an AUROC near 0.5 means the peaks and the signal do not
# agree, and no amount of GPU time will fix that.
#
# Nothing here fails the pipeline: it is advisory, and the point is to read it
# before the GPU steps in 03 and 04.
#
# Usage:
#   export DATASET=<name>
#   cd workflows/SLURM && sbatch 02.0.qc_training_data.sh
#
# Prerequisites: 00.0.prepare_signal.sh, 00.1.preprocess_peaks.sh and
#   01.0.preprocess_nonpeaks.sh.

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

dataset="${datasets[0]}"
signal_bw="${data_path}/signal/data_unstranded.bw"
peaks_np="${peaks_dir}/${dataset}_${peak_type}_peaks_no_blacklist.narrowPeak"
qc_dir="${results_path}/plots/signal_qc"

# One fold's negatives, not all five. Every fold draws from the same candidate
# pool with the same GC matching, so the first is representative of the set;
# QC'ing all five would repeat the same measurement on overlapping regions.
qc_fold="${folds[0]}"
negatives_bed="${data_path}/${dataset}/output_${peak_type}_fold_${qc_fold}_negatives.bed"

metadata_start "02.0.qc_training_data"
metadata_inputs+=( "signal=${signal_bw}" "peaks=${peaks_np}" "negatives=${negatives_bed}" )
metadata_outputs+=( "qc=${qc_dir}/${dataset}_signal_qc.json" )
metadata_outputs+=( "qc=${qc_dir}/${dataset}_signal_qc.tsv" )
metadata_outputs+=( "profile_peaks=${qc_dir}/${dataset}_profile_peaks.pdf" )
metadata_outputs+=( "profile_tss=${qc_dir}/${dataset}_profile_tss.pdf" )
metadata_outputs+=( "peaks_vs_background=${qc_dir}/${dataset}_peaks_vs_background.pdf" )
metadata_params+=( "peak_type=${peak_type}" "input_window=${chrombpnet_input_window}" )
metadata_params+=( "qc_fold=${qc_fold}" "compare_window=${qc_compare_window}" )
[[ -n "${tss_bed}" && -f "${tss_bed}" ]] && metadata_inputs+=( "tss=${tss_bed}" )
require_input "${signal_bw}"     00.0.prepare_signal.sh
require_input "${peaks_np}"      00.1.preprocess_peaks.sh
require_input "${negatives_bed}" 01.0.preprocess_nonpeaks.sh
preflight_check

activate_env "${preprocess_env}"

# TSS enrichment needs a TSS list. `cli.py download-references` derives one from
# refGene; without it the rest of the QC still runs.
tss_args=()
if [[ -n "${tss_bed}" && -f "${tss_bed}" ]]; then
    tss_args+=( --tss "${tss_bed}" )
else
    echo "[$(date)] no TSS list at ${tss_bed:-<unset>}; skipping TSS enrichment"
    echo "           (re-run: cli.py download-references)"
fi

mkdir -p "${qc_dir}"

# This step has no `set -e` (despite what an older note claimed), so the guard
# below is what stops a traceback out of cli.py from printing "QC written to"
# and exiting 0. This QC is advisory and never blocks the pipeline -- which is
# exactly why a silent failure here is dangerous: nobody is waiting on it, so
# an empty plots/signal_qc/ reads as "QC was fine" rather than "QC never ran".
python "${src_dir}/cli.py" qc-signal \
    --bigwig         "${signal_bw}" \
    --peaks          "${peaks_np}" \
    --negatives      "${negatives_bed}" \
    "${tss_args[@]+"${tss_args[@]}"}" \
    --input-window   "${chrombpnet_input_window}" \
    --compare-window "${qc_compare_window}" \
    --out-dir        "${qc_dir}" \
    --prefix         "${dataset}" \
    --metadata-dir   "${metadata_dir}"
if [[ $? -ne 0 || ! -f "${qc_dir}/${dataset}_signal_qc.tsv" ]]; then
    echo "ERROR: qc-signal failed; ${qc_dir}/${dataset}_signal_qc.tsv was not written." >&2
    exit 1
fi

echo "[$(date)] QC written to ${qc_dir}/ — read it before the GPU steps in 03 and 04."
echo "           Start with auroc_peaks_vs_nonpeaks in ${dataset}_signal_qc.tsv."
