#!/bin/bash
#SBATCH --job-name=preprocess_peaks
# bedtools intersect streams the peaks and loads only the small blacklist, so
# memory is tiny and peak-count-independent (~22 MB observed).
#SBATCH --mem=2G
#SBATCH --time=6:00:00
#SBATCH --partition=normal,engreitz
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 00.1.preprocess_peaks.sh
# Purpose: Remove blacklisted regions from peak files and reformat to
#          narrowPeak for chrombpnet (summit = midpoint of peak).
#          With peak_summit_window set, `regions` is instead a peak caller's
#          10-column narrowPeak, and every row becomes a window of that width
#          centred on its OWN summit (one per summit; optional q filter via
#          peak_max_qvalue) -- the summit is then the midpoint again.
#
# The bedtools+awk pipeline this used to run inline now lives in
# src/preprocess_peaks.py (pyranges1), so there is no bedtools dependency and
# the bedtools equivalence is covered by tests/test_intervals.py.

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

mkdir -p "${data_path}"

# The SAME chrom.sizes the signal pileup used, so peaks and signal live on
# the same contigs. A peak on a scaffold the bigwig does not cover would be
# a training region with no data under it.
peak_chrom_sizes="${chrom_sizes_main:-${chrom_sizes}}"

metadata_start "00.1.preprocess_peaks"
metadata_inputs+=( "peaks=${regions}" )
metadata_inputs+=( "blacklist=${blacklist}" )
metadata_inputs+=( "chrom_sizes=${peak_chrom_sizes}" )
metadata_params+=( "peak_type=${peak_type}" "input_window=${chrombpnet_input_window}" )
metadata_params+=( "peak_summit_window=${peak_summit_window:-}" "peak_max_qvalue=${peak_max_qvalue:-}" )
for dataset in "${datasets[@]}"; do
    metadata_outputs+=( "peaks=${peaks_dir}/${dataset}_${peak_type}_peaks_no_blacklist.narrowPeak" )
done

# With call_peaks: true, 00.0.call_peaks.sh writes `regions`; otherwise it is
# an external input.
regions_producer=""
[[ "${call_peaks:-false}" == "true" ]] && regions_producer="00.0.call_peaks.sh"
require_input "${regions}" "${regions_producer}"
require_input "${blacklist}" "cli.py download-references"
require_input "${peak_chrom_sizes}" "cli.py download-references"
preflight_check

activate_env "${preprocess_env}"

for dataset in "${datasets[@]}"; do
    echo "Processing peaks for ${dataset}..."
    # --blacklist also accepts the ENCODE accession (${blacklist_accession}),
    # which fetches it directly; the local copy is the default because compute
    # nodes may have no outbound network.
    # The signal floor is optional and OFF unless peak_min_signal_quantile is
    # set. It needs 00.0's bigwig, which is why 00.0 runs first: a peak whose
    # signal is below what ordinary genome windows reach is not a peak, and it
    # does damage out of all proportion to its own row -- chrombpnet anchors
    # every bias threshold to quantile(peak_counts, 0.01).
    floor_args=()
    if [[ -n "${peak_min_signal_quantile:-}" ]]; then
        prepared_bw="${data_path}/signal/data_unstranded.bw"
        require_input "${prepared_bw}" 00.0.prepare_signal.sh
        preflight_check
        floor_args+=( --signal "${prepared_bw}" )
        floor_args+=( --min-signal-quantile "${peak_min_signal_quantile}" )
        floor_args+=( --compare-window "${qc_compare_window}" )
    fi

    # Optional: read `regions` as a caller's narrowPeak and re-centre each row
    # on its own summit (peak_summit_window), keeping q <= peak_max_qvalue.
    summit_args=()
    if [[ -n "${peak_summit_window:-}" ]]; then
        summit_args+=( --summit-window "${peak_summit_window}" )
        [[ -n "${peak_max_qvalue:-}" ]] && summit_args+=( --max-qvalue "${peak_max_qvalue}" )
    fi

    python "${src_dir}/cli.py" preprocess-peaks \
        --peaks        "${regions}" \
        ${summit_args[@]+"${summit_args[@]}"} \
        --blacklist    "${blacklist}" \
        --chrom-sizes  "${peak_chrom_sizes}" \
        --input-window "${chrombpnet_input_window}" \
        ${floor_args[@]+"${floor_args[@]}"} \
        --out-dir      "${data_path}" \
        --prefix       "${dataset}_${peak_type}"
done

echo "Done: 00.1.preprocess_peaks.sh"
