#!/bin/bash
# motif_chain.sh
# Purpose: one motif campaign inside a running box: the compendium of a list
#   of TF-MoDISco results (09.0 in list mode), then Fi-NeMo hits against it
#   and their report (10.0 -> 11.0) for each dataset of a second list.
#
#   wait until every input H5 is in place
#   -> 09.0, unless <compendium_dir>/compendium.json exists (a finished build)
#   -> per hits row, up to CHAIN_FINEMO_SLOTS (default 4) rows at once:
#      10.0 on a GPU, then 11.0
#
# Every step file is run unchanged, as full_chain.sh runs them, so the steps'
# own skip rules make a rerun resume: 10.0 and 11.0 skip a dataset whose hits
# and report exist for this compendium, and stop on another compendium's.
#
# Lists: `label<TAB>path` rows, 09.0's list format (lib/python/utils/
#   compendium_inputs.py). An inputs row names a dataset config.yaml or a
#   MoDISco H5; a hits row must name a config, since 10.0 and 11.0 are
#   per-dataset steps. `-` for no hits: the compendium only. The tag is the
#   compendium directory's name, and the hits of a dataset go to its
#   ${results_path}/finemo_<tag>/ (lib/bash/finemo_compendium.sh).
#
# Waiting: 08.0 writes its results H5 in place at the very end, so an input
#   is ready once it exists and has not changed for MOTIF_MIN_AGE_S seconds
#   (default 600), checked every MOTIF_POLL_S seconds (default 600).
#   MOTIF_WAIT_FOR_HOLD=<file> names the hold file of whatever makes the
#   inputs. Once it is gone, an input that does not exist is not coming: the
#   chain releases the box and exits 1. An input that exists but is younger
#   than MOTIF_MIN_AGE_S is still waited for, since a producer releases its
#   hold right after writing its last file.
#
# Keeping the box: holds <box>/.hold.motifs.<tag> while it runs and, on exit,
#   puts <box>/.chains_done back if no other .hold* file is left and
#   bias_qc_box.py is not running, as full_chain.sh does. One chain per tag.
#
# GPUs: each 10.0 runs on a GPU claimed with full_chain.sh's protocol, so the
#   two pickers see each other: under flock on <scratch>/gpu_claims/.lock, the
#   GPU with the fewest claim files plus unclaimed GPU steps, ties broken by
#   the least memory in use; a claim file <gpu>.<pid>.<rand>.claim while the
#   step runs; GPU_CLAIM=1 and CUDA_VISIBLE_DEVICES=<gpu> for the step.
#   full_chain.sh's picker does not look for 10.0 processes, so without a
#   claim it could not see one. Its list of unclaimed GPU steps is copied
#   below, with 10.0 added, so that a 10.0 started by hand counts here.
#
# Threads: every step runs under a CPU mask (taskset) cut from the CPUs this
#   script may use, so start it under taskset to confine it: 09.0 gets the
#   first CHAIN_09_THREADS (default 32), Fi-NeMo slot k four CPUs from offset
#   4k for its 10.0 and 11.0, with the thread pools (OMP, numba, MKL,
#   OpenBLAS) pinned to the mask's size.
#
# Environment: REPO_ROOT is set to this checkout whatever the caller
#   exported, and workflows/dcai/env.sh is sourced. METADATA_DIR is unset:
#   09.0's list mode writes its run record beside the compendium
#   (<compendium_dir>/../metadata), 10.0 and 11.0 theirs with each dataset's
#   results. The COMPENDIUM_*, FINEMO_COMPENDIUM_* and DATASET* variables are
#   unset too, so each step gets only what this chain passes it.
#   FINEMO_REGION_WIDTH and FINEMO_BATCH_SIZE, when set, reach 10.0; a
#   compendium of more than ~450 patterns needs a batch size below the
#   default 200 (10.0's header has the GPU memory rule).
#
# Input:  $1 = the inputs list; $2 = the compendium directory, named by its
#         tag; $3 = the hits list, or -. BOX_DIR = the running box's directory.
# Output: the steps' outputs; <box>/motif_chain.tsv (tag, label, step, exit,
#         seconds); logs at <box>/logs/<label>/<step>.<tag>.log, keyed by
#         label because two models' trees can share dataset names (09.0's
#         under <box>/logs/compendium/). A log is appended to, not
#         truncated: each run of the step sits between a start line and an
#         exit line, so a rerun keeps the log of the attempt before it.
# Usage:  BOX_DIR=<box dir> srun --jobid <box> --overlap -N1 -n1 taskset -c <cpus> \
#             bash workflows/dcai/motif_chain.sh <inputs.tsv> <compendia>/<tag> <hits.tsv|->
# Prerequisites: this checkout's motif-compendium and finemo pixi
#   environments, installed before the first run (several first activations
#   at once would each try to install them).

set -uo pipefail
usage="usage: motif_chain.sh <inputs.tsv> <compendium_dir> <hits.tsv|->"
inputs="${1:?${usage}}"
compendium_dir="${2:?${usage}}"
hits="${3:?${usage}}"
: "${BOX_DIR:?set BOX_DIR to the directory of the running box}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export REPO_ROOT
# shellcheck source=workflows/dcai/env.sh
source "${REPO_ROOT}/workflows/dcai/env.sh"
unset METADATA_DIR COMPENDIUM_INPUTS COMPENDIUM_DIR COMPENDIUM_DRY_RUN COMPENDIUM_OVERWRITE \
    COMPENDIUM_ALLOW_MISSING FINEMO_COMPENDIUM_DIR FINEMO_COMPENDIUM_TAG DATASET DATASET_CONFIG DATASET_DIR
steps_dir="${REPO_ROOT}/workflows/SLURM"

abspath() { if [[ "$1" == /* ]]; then printf '%s\n' "$1"; else printf '%s\n' "${PWD}/$1"; fi; }
inputs="$(abspath "${inputs}")"
compendium_dir="$(abspath "${compendium_dir%/}")"
[[ "${hits}" == "-" ]] || hits="$(abspath "${hits}")"
tag="$(basename "${compendium_dir}")"
if [[ ! "${tag}" =~ ^[A-Za-z0-9_]+$ ]]; then
    echo "ERROR: the compendium directory's name is its tag, and '${tag}' is not one: letters, digits and '_' only." >&2
    exit 1
fi
[[ -f "${inputs}" ]] || { echo "ERROR: no inputs list at ${inputs}" >&2; exit 1; }
[[ "${hits}" == "-" || -f "${hits}" ]] || { echo "ERROR: no hits list at ${hits}" >&2; exit 1; }
min_age="${MOTIF_MIN_AGE_S:-600}"
poll="${MOTIF_POLL_S:-600}"
slots="${CHAIN_FINEMO_SLOTS:-4}"
n09="${CHAIN_09_THREADS:-32}"
for _v in min_age poll slots n09; do
    [[ "${!_v}" =~ ^[0-9]+$ ]] || { echo "ERROR: ${_v} must be a whole number, got '${!_v}'" >&2; exit 1; }
done
(( slots >= 1 && n09 >= 1 )) || { echo "ERROR: CHAIN_FINEMO_SLOTS and CHAIN_09_THREADS must be at least 1" >&2; exit 1; }

# The hits rows, checked now rather than after the compendium: configs only.
hit_rows=()
if [[ "${hits}" != "-" ]]; then
    _rows="$(python3 "${REPO_ROOT}/lib/python/utils/compendium_inputs.py" parse "${hits}")" || exit 1
    mapfile -t hit_rows <<< "${_rows}"
    for _row in "${hit_rows[@]}"; do
        IFS=$'\t' read -r _label _kind _path <<< "${_row}"
        if [[ "${_kind}" != "config" ]]; then
            echo "ERROR: ${hits}: ${_label} names ${_path}; a hits row must name a dataset config.yaml" >&2
            exit 1
        fi
    done
fi

record="${BOX_DIR}/motif_chain.tsv"
mkdir -p "${BOX_DIR}/logs"

# Node-local temp files and numba cache, as full_chain.sh keeps its scratch.
scratch="/dev/shm/${USER}/box.${SLURM_JOB_ID:-local}"
export TMPDIR="${scratch}/tmp" TMP="${scratch}/tmp" TEMP="${scratch}/tmp"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${scratch}/numba_cache}"
mkdir -p "${TMPDIR}" "${NUMBA_CACHE_DIR}"

hold="${BOX_DIR}/.hold.motifs.${tag}"
touch "${hold}"
rm -f "${BOX_DIR}/.chains_done"
release() {
    rm -f "${hold}"
    if ! compgen -G "${BOX_DIR}/.hold*" > /dev/null && ! pgrep -f "bias_qc_box.py --box-dir ${BOX_DIR}" > /dev/null; then
        touch "${BOX_DIR}/.chains_done"
        echo "[$(date)] no other holder: released the box (.chains_done)"
    fi
}
trap release EXIT

# CPUs this chain may use, in order; slices of them become each step's mask.
mapfile -t allowed < <(python3 -c 'import os; print("\n".join(map(str, sorted(os.sched_getaffinity(0)))))')
# cpu_slice <offset> <n> -- n of the allowed CPUs from offset (wrapping), comma-separated.
cpu_slice() {
    local i out=()
    for (( i = 0; i < $2; i++ )); do out+=( "${allowed[$(( ($1 + i) % ${#allowed[@]} ))]}" ); done
    local IFS=,
    echo "${out[*]}"
}

claims="${scratch}/gpu_claims"
mkdir -p "${claims}"
n_gpus_node="$(nvidia-smi -L 2>/dev/null | wc -l)"
# full_chain.sh's unclaimed GPU steps, plus 10.0.
gpu_steps_re='SLURM/(03\.0\.train_bias_model|03\.2\.qc_selected_bias|04\.0\.train_full_model|04\.3\.generate_predictions|04\.4\.qc_full_model_interpret|05\.0\.get_contrib_scores|10\.0\.run_finemo_unified)\.sh'
# claim_gpu -- under the lock, the least-loaded GPU (ties: least memory in use);
# prints the claim file made for it.
claim_gpu() {
    (
        flock 8
        local g p env_g best=0 file
        local -a load=() mem=()
        for (( g = 0; g < n_gpus_node; g++ )); do load[g]=0; mem[g]=0; done
        for file in "${claims}"/*.claim; do
            [[ -e "${file}" ]] || continue
            g="${file##*/}"; g="${g%%.*}"
            load[g]=$(( load[g] + 1 ))
        done
        for p in $(pgrep -f "${gpu_steps_re}"); do
            tr '\0' '\n' < "/proc/${p}/environ" 2>/dev/null | grep -qx 'GPU_CLAIM=1' && continue
            env_g="$(tr '\0' '\n' < "/proc/${p}/environ" 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p')"
            [[ "${env_g}" =~ ^[0-9]+$ ]] && (( env_g < n_gpus_node )) && load[env_g]=$(( load[env_g] + 1 ))
        done
        while IFS=', ' read -r g p; do
            [[ "${g}" =~ ^[0-9]+$ && "${p}" =~ ^[0-9]+$ ]] && (( g < n_gpus_node )) && mem[g]="${p}"
        done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null)
        for (( g = 1; g < n_gpus_node; g++ )); do
            if (( load[g] < load[best] || (load[g] == load[best] && mem[g] < mem[best]) )); then
                best="${g}"
            fi
        done
        file="${claims}/${best}.${BASHPID}.${RANDOM}.claim"
        touch "${file}"
        echo "${file}"
    ) 8> "${claims}/.lock"
}

declare -A step_script=(
    [09.0]=09.0.cross_dataset_compendium
    [10.0]=10.0.run_finemo_unified
    [11.0]=11.0.postprocess_finemo
)
# run_step <label> <step> <threads> <cpus> <gpu: 1 or ""> [VAR=value ...] -- one step, recorded.
run_step() {
    local label="$1" step="$2" threads="$3" cpus="$4" need_gpu="$5" t0 rc claim="" gpu="" log
    shift 5
    log="${BOX_DIR}/logs/${label}/${step}.${tag}.log"
    mkdir -p "${BOX_DIR}/logs/${label}"
    if [[ -n "${need_gpu}" && "${n_gpus_node}" -gt 0 ]]; then
        claim="$(claim_gpu)"
        gpu="${claim##*/}"; gpu="${gpu%%.*}"
    fi
    t0="$(date +%s)"
    # The start and exit lines go to this script's output and into the step's
    # log, which is appended to (see the header).
    echo "[$(date)] ${tag} ${label} ${step}: start${claim:+ on GPU ${gpu}}" | tee -a "${log}"
    env "$@" SLURM_ARRAY_TASK_ID=0 SLURM_CPUS_PER_TASK="${threads}" \
        OMP_NUM_THREADS="${threads}" NUMBA_NUM_THREADS="${threads}" \
        MKL_NUM_THREADS="${threads}" OPENBLAS_NUM_THREADS="${threads}" \
        CUDA_VISIBLE_DEVICES="${gpu}" GPU_CLAIM="${claim:+1}" \
        taskset -c "${cpus}" bash "${steps_dir}/${step_script[${step}]}.sh" \
        >> "${log}" 2>&1
    rc=$?
    [[ -n "${claim}" ]] && rm -f "${claim}"
    printf '%s\t%s\t%s\t%d\t%d\n' "${tag}" "${label}" "${step}" "${rc}" "$(( $(date +%s) - t0 ))" >> "${record}"
    echo "[$(date)] ${tag} ${label} ${step}: exit ${rc}" | tee -a "${log}"
    return "${rc}"
}

echo "[$(date)] ${tag}: start (inputs ${inputs}, hits ${hits}, ${#hit_rows[@]} hits row(s))"

# ── The compendium ────────────────────────────────────────────────────────────
if [[ -f "${compendium_dir}/compendium.json" ]]; then
    echo "[$(date)] ${tag}: ${compendium_dir}/compendium.json exists; 09.0 not run"
else
    # The H5 each input resolves to: 09.0's own resolution, in its dry run.
    resolved="$(COMPENDIUM_INPUTS="${inputs}" COMPENDIUM_DIR="${compendium_dir}" COMPENDIUM_DRY_RUN=1 \
        bash "${steps_dir}/09.0.cross_dataset_compendium.sh")" || {
        echo "ERROR: ${tag}: 09.0 could not resolve ${inputs} (see above)" >&2
        exit 1
    }
    mapfile -t input_h5s < <(cut -f2 <<< "${resolved}")
    # not_ready -- one line per input that is not ready: "missing <h5>" or "young <h5>".
    not_ready() {
        local h5 now
        now="$(date +%s)"
        for h5 in "${input_h5s[@]}"; do
            if [[ ! -f "${h5}" ]]; then
                echo "missing ${h5}"
            elif (( now - $(stat -c %Y "${h5}") < min_age )); then
                echo "young ${h5}"
            fi
        done
    }
    while :; do
        mapfile -t waiting < <(not_ready)
        (( ${#waiting[@]} == 0 )) && break
        n_missing="$(printf '%s\n' "${waiting[@]}" | grep -c '^missing ')"
        if [[ -n "${MOTIF_WAIT_FOR_HOLD:-}" && ! -e "${MOTIF_WAIT_FOR_HOLD}" ]] && (( n_missing > 0 )); then
            echo "ERROR: ${tag}: ${MOTIF_WAIT_FOR_HOLD} is gone, and ${n_missing} input(s) do not exist:" >&2
            printf '  %s\n' "${waiting[@]}" | grep '^  missing ' >&2
            exit 1
        fi
        echo "[$(date)] ${tag}: waiting for ${#waiting[@]} of ${#input_h5s[@]} input(s), ${n_missing} not there yet (${waiting[0]})"
        sleep "${poll}"
    done
    if ! run_step compendium 09.0 "${n09}" "$(cpu_slice 0 "${n09}")" "" \
        COMPENDIUM_INPUTS="${inputs}" COMPENDIUM_DIR="${compendium_dir}"; then
        echo "ERROR: ${tag}: 09.0 failed; see ${BOX_DIR}/logs/compendium/09.0.${tag}.log" >&2
        exit 1
    fi
fi

# ── Fi-NeMo, per hits row ─────────────────────────────────────────────────────
# finemo_worker <k> -- hits rows k, k + slots, ...: 10.0 then 11.0, on four CPUs of its own.
finemo_worker() {
    local k="$1" i label cfg cpus rc=0
    cpus="$(cpu_slice $(( 4 * k )) 4)"
    for (( i = k; i < ${#hit_rows[@]}; i += slots )); do
        IFS=$'\t' read -r label _ cfg <<< "${hit_rows[i]}"
        local -a env_args=( DATASET_CONFIG="${cfg}" FINEMO_COMPENDIUM_DIR="${compendium_dir}" FINEMO_COMPENDIUM_TAG="${tag}" )
        if run_step "${label}" 10.0 4 "${cpus}" 1 "${env_args[@]}"; then
            run_step "${label}" 11.0 4 "${cpus}" "" "${env_args[@]}" || rc=1
        else
            rc=1
        fi
    done
    return "${rc}"
}
rc=0
pids=()
for (( k = 0; k < slots && k < ${#hit_rows[@]}; k++ )); do
    finemo_worker "${k}" &
    pids+=( $! )
done
for p in "${pids[@]+"${pids[@]}"}"; do
    wait "${p}" || rc=1
done
echo "[$(date)] ${tag}: chain finished (exit ${rc})"
exit "${rc}"
