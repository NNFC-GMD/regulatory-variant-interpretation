#!/bin/bash
#SBATCH --job-name=call_peaks
# Three MACS3 callpeak runs at once (rep1, rep2, repT) on a few hundred million
# to over a billion insertion records, plus the pseudoreplicate files in
# ${PEAKS_TMPDIR}. Not yet measured on this pipeline; see workflows/dcai/ for the
# whole-node runner that sizes this from the node instead.
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 00.0.call_peaks.sh
# Purpose: call this dataset's peaks from its fragments, the way
#          kundajelab/igvf_pseudobulking_pipeline v2.0.1 does, on MACS3 PR #756
#          (jmschrei/MACS@fa52988: callpeak in C loops with per-chromosome fork
#          pools, byte-identical outputs). Writes the narrowPeak that the
#          config's `regions` names, which 00.1 then turns into training peaks.
#
# Runs only when the config sets `call_peaks: true`. Otherwise `regions` is an
# external input (e.g. scE2G candidate regions) and this step does nothing.
#
# The recipe (utils/peakcall.py has it with file:line references):
#   fragments -> Tn5 insertions, each fragment to pseudoreplicate 1 or 2
#   macs3 callpeak -f BED -g hs -p 0.01 --shift -75 --extsize 150 --nomodel
#                  --keep-dup all --call-summits     on rep1, rep2 and repT
#   top 300k rows per rep by -log10 p -> repT rows reproduced in rep1 AND rep2
#   (overlap >= 0.5 of either) -> ENCODE blacklist removed.
#
# (Before v2.0.1 its top-300k cut, `sort --reverse -k 8gr,8gr | tail`, kept the
# WEAKEST rows; this follows the fixed `sort -k 8g,8g | tail`.)
#
# Departures from v2.0.1, each recorded in call_peaks.json:
#   - insertions sit where 00.0's bigwig puts them (+4/-4), from the shift in
#     the fragments (plus_shift/minus_shift); the original adds +4/-5 to
#     fragments that 10x has already shifted
#   - deterministic pseudoreplicates (one seeded stream); no -B --SPMR
#
# Sibling of 00.0.prepare_signal.sh: both read only the fragments, so they run
# side by side; 00.1 needs both.
#
# Input:  signal_path (fragments), chrom_sizes (FULL: contigs it lists go to
#         MACS, chrM included, as in the original), blacklist
# Output: ${regions}                 the final narrowPeak (bgzipped if .gz)
#         <dir of regions>/call_peaks.json   recipe, shift, counts per filter,
#                                            MACS wall time and peak RSS
#         <dir of regions>/reps/     MACS3 outputs and logs per pseudoreplicate
#
# Usage:
#   export DATASET=<name>              # config with call_peaks: true
#   cd workflows/SLURM && sbatch 00.0.call_peaks.sh
#
# Scratch: the pseudoreplicate files go under ${PEAKS_TMPDIR:-${TMPDIR:-/tmp}}
#   (tens of GB per dataset) and are removed when MACS finishes.
#
# Prerequisites: `cli.py download-references`; `pixi install -e peaks`.

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

if [[ "${call_peaks:-false}" != "true" ]]; then
    echo "[$(date)] call_peaks is not set in ${dataset_config}: regions is an external"
    echo "           input (${regions}). Nothing to do."
    exit 0
fi

peaks_sidecar="$(dirname "${regions}")/call_peaks.json"

metadata_start "00.0.call_peaks"
metadata_inputs+=( "fragments=${signal_path}" "chrom_sizes=${chrom_sizes}" "blacklist=${blacklist}" )
metadata_outputs+=( "peaks=${regions}" "peaks=${peaks_sidecar}" )
metadata_params+=( "macs_input=${macs_input:-bed}" "assay=${assay}" )

if [[ "${signal_type}" != "fragments" ]]; then
    echo "ERROR: call_peaks needs a fragments signal_path; this one is ${signal_type}." >&2
    exit 1
fi
require_input "${signal_path}" ""
require_input "${chrom_sizes}" "cli.py download-references"
require_input "${blacklist}" "cli.py download-references"
# The genome is needed only to detect the shift; with plus_shift/minus_shift in
# the config nothing reads it (same rule as 00.0.prepare_signal).
if [[ -z "${plus_shift}" || -z "${minus_shift}" ]]; then
    require_input "${genome_fa}" "cli.py download-references"
fi
preflight_check

activate_env "${peaks_env}"

if [[ -f "${regions}" && -f "${peaks_sidecar}" ]]; then
    echo "[$(date)] Peaks already called, skipping: ${regions}"
    exit 0
fi

shift_args=()
if [[ -n "${plus_shift}" && -n "${minus_shift}" ]]; then
    shift_args+=( --plus-shift "${plus_shift}" --minus-shift "${minus_shift}" )
else
    echo "[$(date)] plus_shift/minus_shift not set; detecting (needs ${genome_fa})"
    shift_args+=( --genome "${genome_fa}" )
fi

scratch_root="${PEAKS_TMPDIR:-${TMPDIR:-/tmp}}"
mkdir -p "${scratch_root}"
echo "[$(date)] Calling peaks for ${dataset_name} (scratch under ${scratch_root})"

# No `set -e` in this step: the guard below is what turns a traceback out of
# cli.py into a failed step instead of "Done" and exit 0.
python "${src_dir}/cli.py" call-peaks \
    "${shift_args[@]}" \
    --fragments    "${signal_path}" \
    --chrom-sizes  "${chrom_sizes}" \
    --blacklist    "${blacklist}" \
    --output       "${regions}" \
    --assay        "${assay}" \
    --macs-input   "${macs_input:-bed}" \
    --tmp-dir      "${scratch_root}" \
    --threads      4 \
    --metadata-dir "${metadata_dir}"
if [[ $? -ne 0 || ! -f "${regions}" ]]; then
    echo "ERROR: call-peaks failed; ${regions} was not written." >&2
    exit 1
fi

echo "[$(date)] Done. 00.1 reads ${regions}"
