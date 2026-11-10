#!/bin/bash
#SBATCH --job-name=finemo_unified
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
# The finemo environment's torch 2.14 is the default Linux wheel, i.e. the
# CUDA 13.0 build: kernels for sm_75 and up, none for GPU_CC 7.0 (V100,
# TITAN_V), and it needs NVIDIA driver >= 580. With no cuda module loaded any
# more (the wheel brings its own CUDA), Ada (8.9) and Hopper (9.0) are fine.
# Every listed CC has kernels in that build. For the finemo-cu126 fallback,
# see Prerequisites below.
#SBATCH --constraint="GPU_CC:7.5|GPU_CC:8.0|GPU_CC:8.6|GPU_CC:8.9|GPU_CC:9.0"
#SBATCH --time=24:00:00
#SBATCH --partition=gpu,owners
#SBATCH --array=0
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 10.0.run_finemo_unified.sh
# Purpose: Call motif hits with Fi-NeMo against a cross-dataset compendium.
#          Uses the modisco_compiled.h5 built in step 09 and the
#          fold-averaged counts contribution scores from step 06, so that a
#          single hit set per dataset is produced, enabling direct
#          cross-dataset comparisons.
#
#          Array index = index into ${datasets[@]}, which a dataset config
#          sets to its one dataset name: index 0.
#
# Which compendium (lib/bash/finemo_compendium.sh): FINEMO_COMPENDIUM_DIR,
# default ${REPO_ROOT}/results/compendium/modisco_compiled (09.0's default
# mode). The hits go to ${results_path}/finemo_<tag>/<dataset>_<peak_type>/,
# the tag being FINEMO_COMPENDIUM_TAG, else the compendium directory's name,
# else "unified" for the default directory (finemo_unified/, as before). One
# directory per compendium, so the hits of one never pass for another's.
#
# Fi-NeMo 0.41 on torch 2.14 (this repo's finemo pixi environment). finemo
# picks the GPU when torch sees one and otherwise runs on the CPU without a
# word, so the step stops with `require_gpu torch` instead of spending its time
# limit on the CPU. bgzip and tabix come from the same environment (htslib).
#
# hits.bed.gz is this step's done-marker, so it is written under a temporary
# name and moved into place only after bgzip and tabix have both exited 0. It
# used to be written by a redirect, which creates the file even when bgzip is
# missing or fails -- and the empty hits.bed.gz left behind made every rerun
# skip the dataset as finished.
#
# compendium.md5 (`<md5>  <modisco_compiled.h5>`, md5sum's format) records
# which compendium made the hits. It is written after tabix and before the
# index and hits.bed.gz are moved into place, so hits.bed.gz never exists
# without it. The skip check runs before the environment is entered:
#   - hits.bed.gz and a compendium.md5 that matches the compendium: skip;
#   - a compendium.md5 that does not match: stop (exit 1), hits.bed.gz or
#     not -- the directory belongs to another compendium. Use another tag,
#     or delete the directory to redo it;
#   - hits.bed.gz with no compendium.md5 (made before this rule): skip, with
#     a warning.
#
# Region width: finemo_region_width in the config or FINEMO_REGION_WIDTH
# (wins), default 1000, the window Fi-NeMo crops around each summit. Fi-NeMo
# crops the stored window's centre without checking it is wide enough (a
# negative start silently gives a shorter, misplaced crop), so the step stops
# unless the width is even and fits the scores h5: at most the width of its
# shap/seq (2114 for chrombpnet's), with an even difference so the crop is
# centred on whole bases.
#
# Batch size: finemo_batch_size in the config or FINEMO_BATCH_SIZE (wins),
# default 200 (the -b this step has always passed; Fi-NeMo's own default is
# 2000), a positive integer: how many regions `finemo call-hits` optimises at
# once on the GPU. Each region is optimised on its own, with its own step size
# and convergence test, so B is how many share the GPU at a time -- and the
# GPU memory grows with B x M, M being the motifs Fi-NeMo fits: every pattern
# in modisco_compiled.h5 (pos and neg), forward and reverse-complemented, so
# M = 2 x the compendium's patterns. Measured with Fi-NeMo 0.41 on an H100 at
# the default region width (1000), M = 156: about 622 MiB + 0.038 x B x M MiB,
# and ~80 min a dataset at B = 200 on a shared GPU. Keep B <= ~180000 / M,
# which holds one 10.0 under 7.5 GiB (622 + 0.038 x 180000 = 7462 MiB): the
# default fits M <= ~900, a compendium of up to ~450 patterns, and a larger
# one needs a smaller B. The slope is for a 1000-bp window: Fi-NeMo's working
# buffers are B x M x (region width - motif length + 1) float32 values, so a
# wider FINEMO_REGION_WIDTH needs a proportionally smaller B.
#
# `import finemo` compiles a numba function with cache=True. numba writes that
# cache beside finemo's source, else under the user's cache directory, and
# raises if neither is writable. NUMBA_CACHE_DIR (default ${log_dir}/numba_cache)
# is tried before both, so a read-only or shared environment, or an unwritable
# HOME, does not stop the step.
#
# Input per dataset:
#   {averaged_dir}/{dataset}/{dataset}_average_shaps.counts.h5 - averaged DeepLIFT counts scores (step 06)
#   fold 0's interpretation/interpretation.interpreted_regions.bed - the regions they cover (step 05)
#   ${finemo_compendium_dir}/modisco_compiled.h5 - the compendium's patterns (step 09)
#
# Output (inside ${results_path}/finemo_<tag>/{dataset}_{peak_type}/):
#   intermediate_inputs.npz        - regions extracted for Fi-NeMo (11.0 reads it too)
#   hits.tsv, hits_unique.tsv      - full hit tables
#   hits.bed.gz + hits.bed.gz.tbi  - tabix-indexed hit calls (the done-marker)
#   compendium.md5                 - which compendium the hits came from
#   hits.bed, motif_data.tsv, motif_cwms.npy, parameters.json, peaks_qc.tsv
#                                  - the rest of `finemo call-hits` output
#   The HTML report is 11.0's.
#
# Usage:
#   cd workflows/SLURM && DATASET=<name> sbatch 10.0.run_finemo_unified.sh
#   FINEMO_COMPENDIUM_DIR=/path/compendia/<tag> DATASET_CONFIG=/path/config.yaml \
#       sbatch 10.0.run_finemo_unified.sh
#   FINEMO_BATCH_SIZE=100 DATASET=<name> sbatch 10.0.run_finemo_unified.sh   # M up to ~1800
#
# Prerequisites: steps 05, 06 and 09 must have completed, and
#   `pixi install -e finemo` in this checkout. On nodes whose NVIDIA driver is
#   older than 580, use the CUDA 12.6 build of the same environment instead:
#     pixi install -e finemo-cu126
#     export FINEMO_ENV="pixi:${REPO_ROOT}/pixi.toml#finemo-cu126"
#   It has kernels for sm_50 to sm_90, so it needs no lower GPU_CC bound (7.0
#   works) but must not land on Blackwell; submit it with
#     sbatch --constraint="GPU_CC:7.0|GPU_CC:7.5|GPU_CC:8.0|GPU_CC:8.6|GPU_CC:8.9|GPU_CC:9.0" ...

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

# Use the cross-dataset compendium built by 09.0.cross_dataset_compendium.sh
# (finemo_compendium_dir), not the per-dataset one in ${modisco_compiled_dir}.
compiled_h5="${finemo_compendium_dir}/modisco_compiled.h5"
if [[ ! -f "${compiled_h5}" ]]; then
    echo "ERROR: ${compiled_h5} not found. Run 09.0.cross_dataset_compendium.sh first." >&2
    exit 1
fi

# The window Fi-NeMo crops (see the header); checked against the scores once
# h5py is available.
finemo_region_width="${FINEMO_REGION_WIDTH:-${finemo_region_width:-1000}}"
if [[ ! "${finemo_region_width}" =~ ^[1-9][0-9]*$ ]] || (( finemo_region_width % 2 )); then
    echo "ERROR: finemo_region_width must be a positive even number, got '${finemo_region_width}'." >&2
    exit 1
fi
# Fi-NeMo's batch size (see the header for its GPU memory).
finemo_batch_size="${FINEMO_BATCH_SIZE:-${finemo_batch_size:-200}}"
if [[ ! "${finemo_batch_size}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: finemo_batch_size must be a positive integer, got '${finemo_batch_size}'." >&2
    exit 1
fi


metadata_start "10.0.run_finemo_unified"



counts_h5="${averaged_dir}/${dataset}/${dataset}_average_shaps.counts.h5"

# Use the filtered peak list chrombpnet itself wrote during step 05
# (interpretation.interpreted_regions.bed), not the raw *_peaks_no_blacklist.narrowPeak.
# chrombpnet's contribs_bw silently drops peaks whose input window runs off a
# chromosome end (e.g. chrM), so the averaged H5 in step 06/07 has fewer regions
# than the raw peaks file. All folds filter identically (same peaks + genome), so
# fold 0's interpreted_regions.bed matches the H5 row-for-row.
peaks_file="${full_model_dir_selected}/${dataset}_${peak_type}_fold_${folds[0]}/interpretation/interpretation.interpreted_regions.bed"

if [[ ! -f "${counts_h5}" ]]; then
    echo "ERROR: ${counts_h5} not found. Run 06.0.average_contrib_scores.sh first." >&2
    exit 1
fi

if [[ ! -f "${peaks_file}" ]]; then
    echo "ERROR: ${peaks_file} not found. Run 05.0.get_contrib_scores.sh first." >&2
    exit 1
fi

out_dir="${finemo_hits_dir}/${dataset}_${peak_type}"
hits_file="${out_dir}/hits.bed.gz"
md5_file="${out_dir}/compendium.md5"


metadata_inputs+=( "contributions=${counts_h5}" "compiled_h5=${compiled_h5}" "peaks=${peaks_file}" )
require_input "${counts_h5}" 06.0.average_contrib_scores.sh
require_input "${compiled_h5}" 09.0.cross_dataset_compendium.sh
require_input "${peaks_file}" 05.0.get_contrib_scores.sh
preflight_check

compendium_md5="$(md5sum < "${compiled_h5}")" || { echo "ERROR: cannot read ${compiled_h5}" >&2; exit 1; }
compendium_md5="${compendium_md5%% *}"
metadata_outputs+=( "hits=${hits_file}" "compendium_md5=${md5_file}" )
metadata_params+=( "alpha=${finemo_alpha}" "dataset=${dataset}" )
metadata_params+=( "compendium_dir=${finemo_compendium_dir}" "compendium_md5=${compendium_md5}" )
metadata_params+=( "compendium_tag=${finemo_compendium_tag}" "region_width=${finemo_region_width}" )
metadata_params+=( "batch_size=${finemo_batch_size}" )

# Skip rule, keyed to the compendium (see the header), before the environment.
stored_md5=""
[[ -f "${md5_file}" ]] && read -r stored_md5 _ < "${md5_file}"
if [[ -n "${stored_md5}" && "${stored_md5}" != "${compendium_md5}" ]]; then
    echo "ERROR: [${dataset}] the hits in ${out_dir} came from another compendium:" >&2
    echo "  ${md5_file} records ${stored_md5}, but ${compiled_h5} is ${compendium_md5}." >&2
    echo "  Give this compendium its own FINEMO_COMPENDIUM_TAG, or delete ${out_dir} to redo it." >&2
    exit 1
fi
if [[ -f "${hits_file}" ]]; then
    if [[ -z "${stored_md5}" ]]; then
        echo "WARNING: [${dataset}] ${out_dir} has no compendium.md5 (hits made before 10.0 recorded" >&2
        echo "  one): which compendium made them cannot be checked." >&2
    fi
    echo "[${dataset}] Hit calls already exist, skipping."
    exit 0
fi

activate_env "${finemo_env}"
gpu_env
# See the header: numba's cache must be writable before finemo is imported.
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${log_dir}/numba_cache}"
mkdir -p "${NUMBA_CACHE_DIR}"

# The region width against the scores' stored window (see the header).
# Fi-NeMo takes the window from raw/seq and crops shap/seq with it, so both
# are read.
scores_width="$(python -c '
import sys, h5py
with h5py.File(sys.argv[1], "r") as f:
    widths = {f[k].shape[-1] for k in ("raw/seq", "shap/seq")}
print(widths.pop() if len(widths) == 1 else "mismatch")
' "${counts_h5}")"
if [[ ! "${scores_width}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: [${dataset}] cannot read one window width from raw/seq and shap/seq in ${counts_h5} (got '${scores_width}')." >&2
    exit 1
fi
if (( finemo_region_width > scores_width || (scores_width - finemo_region_width) % 2 )); then
    echo "ERROR: [${dataset}] finemo_region_width ${finemo_region_width} does not fit ${counts_h5}," >&2
    echo "  whose scores are ${scores_width} bp wide: it must be at most that, with an even difference." >&2
    exit 1
fi
metadata_params+=( "scores_width=${scores_width}" )

# finemo would otherwise fall back to the CPU silently.
require_gpu torch

finemo_npz="${out_dir}/intermediate_inputs.npz"
mkdir -p "${out_dir}"

echo "[$(date)] [${dataset}] Extracting regions from averaged scores..."

finemo extract-regions-chrombpnet-h5 \
    --h5s          "${counts_h5}" \
    --peaks        "${peaks_file}" \
    --out-path     "${finemo_npz}" \
    --region-width "${finemo_region_width}"
if [[ $? -ne 0 ]]; then
    echo "ERROR: [${dataset}] extract-regions-chrombpnet-h5 failed." >&2
    exit 1
fi

echo "[$(date)] [${dataset}] Calling hits (unified modisco, batch size ${finemo_batch_size})..."

# -l is --global-lambda (Fi-NeMo 0.41 deprecates -a/--alpha); finemo_alpha
# keeps the old name, and so does its metadata parameter. -b is the batch
# size (see the header).
finemo call-hits \
    -r "${finemo_npz}" \
    -m "${compiled_h5}" \
    -l "${finemo_alpha}" \
    -o "${out_dir}" \
    -b "${finemo_batch_size}"
if [[ $? -ne 0 ]]; then
    echo "ERROR: [${dataset}] call-hits failed." >&2
    exit 1
fi

# call-hits always writes hits.bed; its absence after a zero exit means the
# run did not produce what the done-marker below would claim.
if [[ ! -f "${out_dir}/hits.bed" ]]; then
    echo "ERROR: [${dataset}] call-hits exited 0 but wrote no ${out_dir}/hits.bed." >&2
    exit 1
fi

echo "[$(date)] [${dataset}] Compressing and indexing hits..."
hits_tmp="${out_dir}/hits.tmp.bed.gz"
rm -f "${hits_tmp}" "${hits_tmp}.tbi"
if ! bgzip -c "${out_dir}/hits.bed" > "${hits_tmp}"; then
    echo "ERROR: [${dataset}] bgzip failed on ${out_dir}/hits.bed." >&2
    rm -f "${hits_tmp}"
    exit 1
fi
if ! tabix -p bed "${hits_tmp}"; then
    echo "ERROR: [${dataset}] tabix failed on ${hits_tmp}." >&2
    rm -f "${hits_tmp}" "${hits_tmp}.tbi"
    exit 1
fi
# Which compendium made the hits, before the hits take their name (see the header).
md5_tmp="${md5_file}.tmp"
if ! printf '%s  %s\n' "${compendium_md5}" "${compiled_h5}" > "${md5_tmp}" || ! mv -f "${md5_tmp}" "${md5_file}"; then
    echo "ERROR: [${dataset}] could not write ${md5_file}." >&2
    rm -f "${md5_tmp}"
    exit 1
fi
# The index first, so the done-marker never exists without it.
if ! mv -f "${hits_tmp}.tbi" "${hits_file}.tbi" || ! mv -f "${hits_tmp}" "${hits_file}"; then
    echo "ERROR: [${dataset}] could not move the compressed hits into ${out_dir}." >&2
    exit 1
fi

echo "[$(date)] [${dataset}] Fi-NeMo (${finemo_compendium_tag}) complete."
echo "  Hits: ${hits_file}"
