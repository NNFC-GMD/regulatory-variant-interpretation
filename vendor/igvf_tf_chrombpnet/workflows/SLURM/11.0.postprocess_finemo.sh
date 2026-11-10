#!/bin/bash
#SBATCH --job-name=finemo_postprocess
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --partition=normal,engreitz
#SBATCH --array=0
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 11.0.postprocess_finemo.sh
# Purpose: Generate the Fi-NeMo report for a dataset's hit calls against one
#          compendium:
#            finemo report - compute per-motif instance-CWM vs input-CWM
#                             similarity (cwm_similarity in motif_report.tsv)
#
# The hits are 10.0's for the same compendium (lib/bash/finemo_compendium.sh):
# FINEMO_COMPENDIUM_DIR and FINEMO_COMPENDIUM_TAG pick the directory,
# ${results_path}/finemo_<tag>/<dataset>_<peak_type>/, finemo_unified/ by
# default. When 10.0 recorded the compendium beside the hits (compendium.md5),
# the step stops if it is not the compendium FINEMO_COMPENDIUM_DIR names --
# before its skip check, so a report made from another compendium's hits is
# not taken as this one's either.
#
# CPU only, in ${finemo_env} (this repo's finemo pixi environment, Fi-NeMo
# 0.41). `finemo report` needs no GPU, so unlike 10.0 there is no require_gpu.
# `import finemo` still imports torch and compiles a numba function with
# cache=True, so NUMBA_CACHE_DIR (default ${log_dir}/numba_cache) gives numba
# a writable cache whatever the environment's or HOME's permissions; see 10.0.
#
# Input (from step 10):
#   {finemo_hits_dir}/{dataset}_{peak_type}/hits.tsv
#   {finemo_hits_dir}/{dataset}_{peak_type}/intermediate_inputs.npz
#   (finemo report also reads call-hits' motif_data.tsv, motif_cwms.npy and
#   parameters.json from the same directory)
#   {finemo_hits_dir}/{dataset}_{peak_type}/compendium.md5, when 10.0 wrote one,
#   against ${finemo_compendium_dir}/modisco_compiled.h5
#
# Output:
#   {finemo_hits_dir}/{dataset}_{peak_type}/finemo_report/motif_report.tsv
#   {finemo_hits_dir}/{dataset}_{peak_type}/finemo_report/report.html
#
# Usage:
#   cd workflows/SLURM && DATASET=<name> sbatch 11.0.postprocess_finemo.sh
#   FINEMO_COMPENDIUM_DIR=/path/compendia/<tag> DATASET_CONFIG=/path/config.yaml \
#       sbatch 11.0.postprocess_finemo.sh
#
# Prerequisites: 10.0.run_finemo_unified.sh must have completed for this
#   dataset, and `pixi install -e finemo` in this checkout.

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
# The compendium and the hits directory (see the header).
# shellcheck source=lib/bash/finemo_compendium.sh
source "${REPO_ROOT}/lib/bash/finemo_compendium.sh" || exit 1

dataset="${datasets[${SLURM_ARRAY_TASK_ID}]}"
[[ -z "${dataset}" ]] && { echo "No dataset at array index ${SLURM_ARRAY_TASK_ID}, exiting."; exit 0; }

out_dir="${finemo_hits_dir}/${dataset}_${peak_type}"
hits_tsv="${out_dir}/hits.tsv"
finemo_npz="${out_dir}/intermediate_inputs.npz"
report_dir="${out_dir}/finemo_report"

# The hits must be the named compendium's (see the header).
md5_file="${out_dir}/compendium.md5"
compiled_h5="${finemo_compendium_dir}/modisco_compiled.h5"
stored_md5=""
[[ -f "${md5_file}" ]] && read -r stored_md5 _ < "${md5_file}"
if [[ -n "${stored_md5}" ]]; then
    if [[ ! -f "${compiled_h5}" ]]; then
        echo "ERROR: [${dataset}] cannot check which compendium made the hits in ${out_dir}:" >&2
        echo "  ${compiled_h5} does not exist. Point FINEMO_COMPENDIUM_DIR at the compendium 10.0 used." >&2
        exit 1
    fi
    compendium_md5="$(md5sum < "${compiled_h5}")" || { echo "ERROR: cannot read ${compiled_h5}" >&2; exit 1; }
    compendium_md5="${compendium_md5%% *}"
    if [[ "${stored_md5}" != "${compendium_md5}" ]]; then
        echo "ERROR: [${dataset}] the hits in ${out_dir} came from another compendium:" >&2
        echo "  ${md5_file} records ${stored_md5}, but ${compiled_h5} is ${compendium_md5}." >&2
        exit 1
    fi
fi

if [[ -f "${report_dir}/motif_report.tsv" ]]; then
    echo "[${dataset}] Already post-processed, skipping."
    exit 0
fi

if [[ ! -f "${hits_tsv}" ]]; then
    echo "ERROR: ${hits_tsv} not found. Run 10.0.run_finemo_unified.sh first." >&2
    exit 1
fi

if [[ ! -f "${finemo_npz}" ]]; then
    echo "ERROR: ${finemo_npz} not found. Run 10.0.run_finemo_unified.sh first." >&2
    exit 1
fi

metadata_start "11.0.postprocess_finemo"
metadata_inputs+=( "hits=${hits_tsv}" "npz=${finemo_npz}" )
require_input "${hits_tsv}" 10.0.run_finemo_unified.sh
require_input "${finemo_npz}" 10.0.run_finemo_unified.sh
preflight_check

activate_env "${finemo_env}"
# See the header: numba's cache must be writable before finemo is imported.
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${log_dir}/numba_cache}"
mkdir -p "${NUMBA_CACHE_DIR}"
metadata_outputs+=( "motif_report=${report_dir}/motif_report.tsv" )
metadata_params+=( "dataset=${dataset}" "compendium_tag=${finemo_compendium_tag}" )


# Compute cwm_similarity for each motif by comparing the average CWM
# reconstructed from called instances against the input (modisco) CWM.
echo "[$(date)] [${dataset}] Running finemo report..."
finemo report \
    -r "${finemo_npz}" \
    -H "${out_dir}" \
    -o "${report_dir}"
rc=$?

# The exit status too: motif_report.tsv is also this step's skip marker, so a
# report that failed after an earlier run wrote one would otherwise pass.
motif_report="${report_dir}/motif_report.tsv"
if [[ ${rc} -ne 0 || ! -f "${motif_report}" ]]; then
    echo "ERROR: finemo report failed (exit ${rc}) or did not write ${motif_report}. See its output above." >&2
    exit 1
fi
echo "[$(date)] [${dataset}] Report done: ${motif_report}"
