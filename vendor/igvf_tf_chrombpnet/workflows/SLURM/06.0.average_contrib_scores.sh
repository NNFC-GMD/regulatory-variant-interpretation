#!/bin/bash
#SBATCH --job-name=avg_contribs
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --partition=normal,engreitz
#SBATCH --array=0
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# =============================================================================
# 06.0.average_contrib_scores.sh
# Purpose: Average DeepLIFT contribution scores across all 5 folds for
#          each dataset. One SLURM array job per dataset.
#
# Why: Each fold's model was trained on a different 80/20 data split, so
#      its contribution scores carry fold-specific noise. Averaging reduces
#      this noise and gives a more robust signal for motif discovery.
#      This follows the Greenleaf lab approach (their step 05-average_deepshaps).
#      05.0 also gives each fold its own DeepSHAP reference seed, so the
#      average smooths over reference noise too.
#
# score_types: contrib_heads (config.sh), both heads by default -- whatever
#   05.0 scored. 09.0 and 10.0 consume the counts average only;
#   profile feeds 07.0's profile bigwig and 08.0's profile TF-MoDISco. (Counts
#   used to be left out here because it had already been averaged in the 1.x
#   results tree; a chrombpnet 2.x results tree starts empty.)
#   average_contrib_scores.py skips a head whose output already exists, so a
#   rerun only fills in what is missing; delete an output to recompute it.
#   It averages whichever folds have scores and warns about the rest.
#
# Input:  ${full_model_dir}/{dataset}_{peak_type}_fold_{fold}/interpretation/
#             interpretation.{counts,profile}_scores.h5  (05.0), for each fold in ${folds[@]}
# Output: ${averaged_dir}/{dataset}/{dataset}_average_shaps.{counts,profile}.h5
#
# Usage:
#   export DATASET=<name>        # or DATASET_CONFIG=/path/to/config.yaml
#   sbatch 06.0.average_contrib_scores.sh              # dataset 0 (the default --array=0)
#   sbatch --array=1 06.0.average_contrib_scores.sh    # dataset 1 of ${datasets[@]}
#
# Prerequisites: 05.0.get_contrib_scores.sh must have completed for all folds.
# =============================================================================

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

out_dir="${averaged_dir}/${dataset}"
mkdir -p "${out_dir}" "${log_dir}"

metadata_start "06.0.average_contrib_scores"
metadata_params+=( "dataset=${dataset}" "folds=${folds[*]}" )
# Declared up front (hashed at exit), so a failed run still records what it
# meant to write. A fold with no scores is recorded with file_exists false.
for score_type in "${score_types[@]}"; do
    for _fold in "${folds[@]}"; do
        metadata_inputs+=( "contributions=${full_model_dir_selected}/${dataset}_${peak_type}_fold_${_fold}/interpretation/interpretation.${score_type}_scores.h5" )
    done
    metadata_outputs+=( "contributions=${out_dir}/${dataset}_average_shaps.${score_type}.h5" )
done
unset _fold

activate_env "${chrombpnet_env}"

for score_type in "${score_types[@]}"; do
    out_h5="${out_dir}/${dataset}_average_shaps.${score_type}.h5"
    echo "[$(date)] [${dataset} ${score_type}] Averaging contribution scores..."
    python "${src_dir}/average_contrib_scores.py" \
        --dataset "${dataset}" \
        --folds "${folds[@]}" \
        --full-model-dir "${full_model_dir_selected}" \
        --peak-type "${peak_type}" \
        --score-type "${score_type}" \
        --out-dir "${out_dir}"
    _rc=$?
    # No `set -e` in this step: guard the call and its output explicitly.
    if [[ ${_rc} -ne 0 || ! -f "${out_h5}" ]]; then
        echo "ERROR: average_contrib_scores.py failed for ${dataset} ${score_type} (exit ${_rc}); ${out_h5} was not written." >&2
        exit 1
    fi
done

echo "Done"
