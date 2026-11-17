#!/bin/bash
#SBATCH --job-name=avg_contribs_bw
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --time=2:00:00
#SBATCH --partition=normal,engreitz
#SBATCH --array=0
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 07.0.contribs_to_bigwig.sh
# Purpose: Convert the fold-averaged contribution score H5s (from step 06)
#          into bigwigs, one per head, for each dataset. One SLURM array job
#          per dataset. src/contribs_to_bigwig.py uses chrombpnet's own
#          bigwig_helper, so it runs in the chrombpnet 2.x environment.
#
# score_types: contrib_heads (config.sh), mirroring 06.0. contribs_to_bigwig.py skips a head
#   whose bigwig already exists, so a rerun only fills in what is missing.
#
# Regions: interpretation.interpreted_regions.bed from fold ${folds[0]} (the
#   same fold 10.0 reads), not the narrowPeak -- 05.0's contribs_bw drops peaks
#   whose window runs off a chromosome end, so only its regions file matches
#   the H5 row for row. Every fold scores the same peaks against the same
#   genome, so every fold drops the same ones.
#
# Input:  ${averaged_dir}/{dataset}/{dataset}_average_shaps.{counts,profile}.h5  (06.0)
#         ${full_model_dir}/{dataset}_{peak_type}_fold_${folds[0]}/interpretation/
#             interpretation.interpreted_regions.bed                          (05.0)
# Output: ${averaged_dir}/{dataset}/{dataset}_average_shaps.{counts,profile}.bw
#
# Usage:
#   export DATASET=<name>        # or DATASET_CONFIG=/path/to/config.yaml
#   sbatch 07.0.contribs_to_bigwig.sh              # dataset 0 (the default --array=0)
#   sbatch --array=1 07.0.contribs_to_bigwig.sh    # dataset 1 of ${datasets[@]}
#
# Prerequisites: 06.0.average_contrib_scores.sh must have completed.

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

dataset="${datasets[${SLURM_ARRAY_TASK_ID}]}"
[[ -z "${dataset}" ]] && { echo "No dataset at array index ${SLURM_ARRAY_TASK_ID}, exiting."; exit 0; }

score_types=( "${contrib_heads[@]}" )  # see the header; contrib_heads, config.sh


metadata_start "07.0.contribs_to_bigwig"
metadata_params+=( "dataset=${dataset}" )

# The first configured fold's interpreted regions -- see the header.
regions_file="${full_model_dir}/${dataset}_${peak_type}_fold_${folds[0]}/interpretation/interpretation.interpreted_regions.bed"

metadata_inputs+=( "peaks=${regions_file}" "chrom_sizes=${chrom_sizes}" )
require_input "${regions_file}" 05.0.get_contrib_scores.sh
require_input "${chrom_sizes}" "cli.py download-references"
for score_type in "${score_types[@]}"; do
    metadata_inputs+=( "contributions=${averaged_dir}/${dataset}/${dataset}_average_shaps.${score_type}.h5" )
    metadata_outputs+=( "contributions=${averaged_dir}/${dataset}/${dataset}_average_shaps.${score_type}.bw" )
    require_input "${averaged_dir}/${dataset}/${dataset}_average_shaps.${score_type}.h5" 06.0.average_contrib_scores.sh
done
preflight_check

activate_env "${chrombpnet_env}"

for score_type in "${score_types[@]}"; do
    h5_file="${averaged_dir}/${dataset}/${dataset}_average_shaps.${score_type}.h5"
    out_bw="${averaged_dir}/${dataset}/${dataset}_average_shaps.${score_type}.bw"

    echo "[$(date)] [${dataset} ${score_type}] Writing averaged contribution score bigwig..."

    python3 "${src_dir}/contribs_to_bigwig.py" \
        --h5 "${h5_file}" \
        --regions "${regions_file}" \
        --chrom-sizes "${chrom_sizes}" \
        --output-bw "${out_bw}"
    _rc=$?
    # No `set -e` in this step: guard the call and its output explicitly.
    if [[ ${_rc} -ne 0 || ! -f "${out_bw}" ]]; then
        echo "ERROR: contribs_to_bigwig.py failed for ${dataset} ${score_type} (exit ${_rc}); ${out_bw} was not written." >&2
        exit 1
    fi

    echo "[$(date)] [${dataset} ${score_type}] Done: ${out_bw}"
done
