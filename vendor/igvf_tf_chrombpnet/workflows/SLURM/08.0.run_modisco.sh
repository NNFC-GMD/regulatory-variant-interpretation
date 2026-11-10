#!/bin/bash
#SBATCH --job-name=modisco_avg
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --time=4-0
#SBATCH --partition=engreitz
#SBATCH --qos=high_p
#SBATCH --array=0
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 08.0.run_modisco.sh
# Purpose: Run TF-MoDISco on fold-averaged contribution scores (step 06/07).
#          One SLURM array job per dataset; produces ONE modisco result per
#          dataset per score type (rather than one per fold), which is then
#          used for the unified motif compendium.
#
# Why run on averaged scores:
#   Averaging contribution scores across folds before MoDISco improves
#   signal-to-noise ratio, so the discovered patterns are more reproducible
#   and biologically meaningful. This is the standard approach in the
#   Greenleaf lab ChromBPNet pipeline.
#
# score_types: contrib_heads (config.sh; both heads by default), one after the
#   other. 09.0 builds the compendium
#   from the counts results only (modisco_counts_results.h5); the profile
#   results are for reading. (Counts used to be left out here because it had
#   already been run in the 1.x results tree; a chrombpnet 2.x results tree
#   starts empty.)
#
# Settings: TF-MoDISco 2.5.2 (the `modisco` package, tfmodisco-lite), with
#   chrombpnet 1.x's pattern settings passed explicitly:
#   `-l 2 -z 20 -f 5 -t 20 -g 5 -j 0`. modisco-lite 2.0.7, which
#   1.x ran, fixed seqlet core 20 / flank 5 / final flank 0 and used trim 20 /
#   initial flank 5; 2.5.2 defaults to -t 30 -g 10, which makes every pattern
#   50 bp wide. These are the values chrombpnet 2.x's own
#   evaluation/modisco/run.py passes. -w 500, and -n 500000 by default: the
#   seqlet budget per metacluster, modisco_max_seqlets in config.yaml or
#   MODISCO_MAX_SEQLETS (wins). Runtime grows faster than linearly with it; a
#   lower budget keeps fewer instances of each pattern, so rare patterns are
#   the first to go (the per-fold QC in 03.3/04.5 uses 5000).
#
# Environment: this repo's `modisco` pixi environment, not chrombpnet's. It is
#   2.5.2 from NNFC-GMD/tfmodisco's parallel-leiden-seeds branch, which adds
#   `--n_leiden_jobs` and `--n_merge_threads` and replaces 2.5.2's all-pairs
#   neighbour search with an inverted index, all with bit-identical output
#   (91367a9: -n 25000 on 291,194 regions of AMSC scores, 42.9 -> 17.6 min,
#   results h5 identical dataset by dataset), with every package that computes (numpy, numba, scipy,
#   scikit-learn, h5py, igraph, leidenalg, memelite) pinned to the version in
#   chrombpnet's cuda13 environment, where 03.3/04.5 run stock 2.5.2.
#
# Leiden restarts: 2.5.2 runs the `-l` restarts of every clustering call one
#   after another on one core (leidenalg holds the GIL), and at this step's
#   budget they are most of a call: on a 20,000-seqlet affinity with 500
#   neighbours each restart took ~75 s, against 37 s for the rest of the call.
#   `--n_leiden_jobs 2` runs the two restarts in two processes. The clustering
#   is unchanged, since each restart is seeded and the best is still taken in
#   seed order: the branch tests it against 2.5.2's function, and `modisco
#   motifs` with this step's flags on a synthetic 8,000-region input wrote a
#   results file identical to stock 2.5.2's in chrombpnet's environment (1,221
#   datasets, 17 patterns). The 20,000-seqlet call took 84 s instead of 151 s. Each worker is a fresh process holding its own
#   copy of the graph, and reads the affinity matrix from TMPDIR (12-16 bytes
#   per stored entry), so TMPDIR wants node-local space. Calls on fewer than
#   1,000,000 stored entries stay in process.
#
# Report: `modisco report-simple` writes {head}_report/motifs.html with
#   trimmed_logos/ and the matched motifs' logos -- the report 2.0.7's
#   `modisco report` wrote; 2.5.2's `modisco report` is a different,
#   descriptive one. Matches come from MEME's `tomtom` (q-values), which
#   the modisco environment ships; the step checks for it up front
#   because it is only needed after days of `modisco motifs`. No -s:
#   report-simple pastes it straight in front of each match-logo file name, so
#   an absolute path without a trailing slash breaks those links, while the
#   default ./ keeps every link relative to motifs.html.
#
# Threads: numba (modisco, memelite) sizes its thread pool to the cores the
#   kernel reports -- the host's, not the allocation's -- so NUMBA_NUM_THREADS,
#   OMP_NUM_THREADS, MKL_NUM_THREADS and OPENBLAS_NUM_THREADS are pinned to
#   SLURM_CPUS_PER_TASK. memelite compiles with numba's on-disk cache, which
#   needs a writable directory: NUMBA_CACHE_DIR, default ${log_dir}/numba_cache.
#
# Skip rule, per head: `modisco motifs` is skipped when
#   modisco_{head}_results.h5 exists, and the report when {head}_report/
#   motifs.html does. The two heads run back to back and the 4-day --time was
#   set when this ran one, so a job that times out during profile resumes
#   there when resubmitted. modisco writes its h5 in place at the very end, so
#   a job killed in that moment leaves a truncated h5 that the report then
#   fails on: delete it to redo that head. Delete any output to redo it.
#
# Partition/GPU/QOS: modisco motifs (tfmodisco-lite) is CPU-only, so this
# runs on the engreitz partition (no GPU request) with --qos=high_p. The
# default QOS on any partition caps walltime at 2 days for this account
# regardless of what sh_part's per-partition ceiling shows; --qos=high_p
# (7-day MaxWall, granted to the engreitz account) is required to actually
# get more than 2 days, which some datasets need.
#
# Input:  ${averaged_dir}/{dataset}/{dataset}_average_shaps.{counts,profile}.h5  (06.0)
# Output: ${averaged_dir}/{dataset}/modisco/
#             modisco_{counts,profile}_results.h5
#             {counts,profile}_report/motifs.html  (+ trimmed_logos/, match logos)
#
# Usage:
#   export DATASET=<name>        # or DATASET_CONFIG=/path/to/config.yaml
#   sbatch 08.0.run_modisco.sh              # dataset 0 (the default --array=0)
#   sbatch --array=1 08.0.run_modisco.sh    # dataset 1 of ${datasets[@]}
#
# Prerequisites: 06.0.average_contrib_scores.sh must have completed; the
#   modisco environment (`pixi install -e modisco`, linux-64 only), which
#   includes MEME's tomtom; the MotifCompendium database (cli.py
#   download-references).

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

# TF-MoDISco settings -- see the header. The pattern flags reproduce what
# chrombpnet 1.x's modisco-lite 2.0.7 did; 2.5.2's defaults differ.
modisco_max_seqlets="${MODISCO_MAX_SEQLETS:-${modisco_max_seqlets:-500000}}"
[[ "${modisco_max_seqlets}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: modisco_max_seqlets must be a positive integer, got '${modisco_max_seqlets}'" >&2; exit 1; }
modisco_window=500
modisco_pattern_args=( -l 2 -z 20 -f 5 -t 20 -g 5 -j 0 )
# One process per -l restart; the clustering does not depend on it (see the header).
modisco_leiden_jobs=2
# Threads for pattern merging's AUROCs (the merges do not depend on it); config
# modisco_merge_threads or MODISCO_MERGE_THREADS.
modisco_merge_threads="${MODISCO_MERGE_THREADS:-${modisco_merge_threads:-16}}"
modisco_dir="${averaged_dir}/${dataset}/modisco"

metadata_start "08.0.run_modisco"
metadata_params+=( "dataset=${dataset}" "max_seqlets=${modisco_max_seqlets}" "window=${modisco_window}" )
metadata_params+=( "pattern_args=${modisco_pattern_args[*]}" "threads=${SLURM_CPUS_PER_TASK:-4}" )
metadata_params+=( "leiden_jobs=${modisco_leiden_jobs}" "merge_threads=${modisco_merge_threads}" )
for score_type in "${score_types[@]}"; do
    metadata_inputs+=( "contributions=${averaged_dir}/${dataset}/${dataset}_average_shaps.${score_type}.h5" )
    metadata_outputs+=( "motifs=${modisco_dir}/modisco_${score_type}_results.h5" )
    metadata_outputs+=( "motif_report=${modisco_dir}/${score_type}_report/motifs.html" )
    require_input "${averaged_dir}/${dataset}/${dataset}_average_shaps.${score_type}.h5" 06.0.average_contrib_scores.sh
done
metadata_inputs+=( "motif_db=${ref_db_meme}" )
require_input "${ref_db_meme}" "cli.py download-references"
preflight_check

mkdir -p "${log_dir}"
for score_type in "${score_types[@]}"; do
    mkdir -p "${modisco_dir}/${score_type}_report"
done

activate_env "${modisco_env}"

# report-simple shells out to MEME's tomtom; find out now, not after days of
# `modisco motifs`.
if ! command -v tomtom >/dev/null 2>&1; then
    echo "ERROR: MEME's tomtom is not on PATH in ${modisco_env}." >&2
    echo "  modisco report-simple needs it to match patterns against ${ref_db_meme}." >&2
    echo "  The modisco pixi environment includes MEME; a conda env needs 'meme' installed." >&2
    exit 1
fi

# numba threads and cache -- see the header. Pinned, not defaulted: numba
# would otherwise size its pool to the host's cores, not the allocation.
_threads="${SLURM_CPUS_PER_TASK:-4}"
export NUMBA_NUM_THREADS="${_threads}" OMP_NUM_THREADS="${_threads}"
export MKL_NUM_THREADS="${_threads}" OPENBLAS_NUM_THREADS="${_threads}"
NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${log_dir}/numba_cache}"
mkdir -p "${NUMBA_CACHE_DIR}"
export NUMBA_CACHE_DIR
echo "[$(date)] modisco: ${_threads} numba threads, NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR}"

for score_type in "${score_types[@]}"; do
    scores_h5="${averaged_dir}/${dataset}/${dataset}_average_shaps.${score_type}.h5"
    results_h5="${modisco_dir}/modisco_${score_type}_results.h5"
    report_dir="${modisco_dir}/${score_type}_report"

    if [[ -f "${results_h5}" ]]; then
        echo "[$(date)] Dataset ${dataset}: ${results_h5} exists, skipping modisco motifs."
    else
        echo "[$(date)] Dataset ${dataset}: running MoDISco on averaged ${score_type} scores..."
        modisco motifs \
            -i "${scores_h5}" \
            -n "${modisco_max_seqlets}" \
            -o "${results_h5}" \
            -w "${modisco_window}" \
            "${modisco_pattern_args[@]}" \
            --n_leiden_jobs "${modisco_leiden_jobs}" \
            --n_merge_threads "${modisco_merge_threads}" \
            -v
        _rc=$?
        # No `set -e` in this step: guard each call and its output explicitly.
        if [[ ${_rc} -ne 0 || ! -f "${results_h5}" ]]; then
            echo "ERROR: modisco motifs failed for ${dataset} ${score_type} (exit ${_rc}); ${results_h5} was not written." >&2
            exit 1
        fi
    fi

    if [[ -f "${report_dir}/motifs.html" ]]; then
        echo "[$(date)] Dataset ${dataset}: ${report_dir}/motifs.html exists, skipping the report."
    else
        modisco report-simple \
            -i "${results_h5}" \
            -o "${report_dir}/" \
            -m "${ref_db_meme}"
        _rc=$?
        if [[ ${_rc} -ne 0 || ! -f "${report_dir}/motifs.html" ]]; then
            echo "ERROR: modisco report-simple failed for ${dataset} ${score_type} (exit ${_rc}); ${report_dir}/motifs.html was not written." >&2
            echo "  If modisco motifs was killed while saving, ${results_h5} is truncated: delete it and resubmit." >&2
            exit 1
        fi
    fi

    echo "[$(date)] Dataset ${dataset}: ${score_type} MoDISco complete"
done
