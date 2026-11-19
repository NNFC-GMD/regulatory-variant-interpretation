#!/bin/bash
# run_box.sh
# Purpose: keep the GPUs of one node busy training and scoring Cherimoya models,
#   one (dataset, fold) task at a time per slot, from tasks.tsv (make_jobs.py).
#
# DCAI rents whole nodes (208 cores, ~2 TB RAM, 8x H100) by node-time, so the
# tasks run inside one allocation: GPUs x CHERI_SLOTS_PER_GPU workers, each
# pinned to its own block of CHERI_CORES_PER_SLOT cores (taskset), take the
# next task from a queue under a lock and run run_task.sh on it. The GPU is
# chosen per task, when it is taken: the one with the fewest tasks running
# (CUDA_VISIBLE_DEVICES), so a short queue, or the tail of a long one, spreads
# over every GPU instead of piling onto the GPUs whose earlier tasks ended
# first (claims.tsv records each choice and the loads it saw). The default 9-layer, 128-filter model
# is small for an H100, so two trainings share a GPU by default. The same
# script runs inside an allocation that is already busy with other work
# (`srun --jobid <job> --overlap`): CHERI_GPUS and CHERI_CORE_OFFSET then keep
# it to the GPUs and cores given to it.
#
# A task whose ChromBPNet model is not trained yet trains, then leaves its
# comparison waiting (run_task.sh exit 3); a later run with CHERI_STAGES=compare
# picks those up. CHERI_STAGES=attribute waits the same way for folds whose fit
# is not done. Tasks skip finished stages, so resubmitting resumes.
#
# Knobs (environment):
#   CHERI_ROOT            make_jobs.py's --out-root (required)
#   CHERI_STAGES          fit,compare (default), fit, compare, or attribute
#                         (run_task.sh's stages; attribute's own knobs are in run_task.sh)
#   CHERI_GPUS            GPU indices to use, comma-separated (default: all)
#   CHERI_SLOTS_PER_GPU   concurrent tasks per GPU (default 2)
#   CHERI_CORES_PER_SLOT  cores pinned per task (default 8)
#   CHERI_CORE_OFFSET     cores to skip, counted in the allowed set (default 0)
#   CHERI_ONLY            only these datasets, space-separated (default: every ready row)
#   CHERI_MAX_TASKS       stop after this many tasks have started (default: no limit)
#   CHERI_ORDER           forward (default) or reverse: the order the queue is taken in
#                         (make_jobs.py lists the datasets with the most peaks first)
#   CHERI_TASK_CMD        the task runner (default workflows/train/run_task.sh; for tests)
#   CHERI_SCRATCH_ROOT    node-local scratch (default /raid if writable, else /dev/shm)
#
# Output: under ${CHERI_ROOT}/box/<job id>/: logs/<dataset>.fold_<k>.log,
#         status.tsv (dataset, fold, exit, seconds, gpu), claims.tsv (dataset,
#         fold, gpu, the running tasks per GPU when it was chosen), resources.tsv; and
#         each task's outputs in its fold directory.
# Usage:  CHERI_ROOT=<dir> sbatch workflows/train/run_box.sbatch
#         or, inside a running allocation:
#         CHERI_ROOT=<dir> CHERI_GPUS=7 CHERI_SLOTS_PER_GPU=1 CHERI_CORE_OFFSET=200 \
#             srun --jobid <job> --overlap bash workflows/train/run_box.sh
# Prerequisites: `pixi install -e cherimoya` in this repo; make_jobs.py.

set -uo pipefail

: "${CHERI_ROOT:?set CHERI_ROOT to the make_jobs.py --out-root}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export RVI_ROOT="${repo_root}"
case ":${PATH}:" in
    *":${HOME}/.pixi/bin:"*) ;;
    *) export PATH="${HOME}/.pixi/bin:${PATH}" ;;
esac
tasks_tsv="${CHERI_ROOT}/tasks.tsv"
[[ -s "${tasks_tsv}" ]] || { echo "ERROR: no ${tasks_tsv}; run make_jobs.py" >&2; exit 1; }

stages="${CHERI_STAGES:-fit,compare}"
slots_per_gpu="${CHERI_SLOTS_PER_GPU:-2}"
cores_per_slot="${CHERI_CORES_PER_SLOT:-8}"
core_offset="${CHERI_CORE_OFFSET:-0}"
max_tasks="${CHERI_MAX_TASKS:-0}"
task_cmd="${CHERI_TASK_CMD:-${repo_root}/workflows/train/run_task.sh}"
if [[ -n "${CHERI_GPUS:-}" ]]; then
    IFS=, read -r -a gpus <<< "${CHERI_GPUS}"
else
    mapfile -t gpus < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)
fi
(( ${#gpus[@]} > 0 )) || { echo "ERROR: no GPU found (nvidia-smi) and CHERI_GPUS unset" >&2; exit 1; }
n_slots=$(( ${#gpus[@]} * slots_per_gpu ))
# The cores this process may use (an srun step can be given fewer than the
# node has); CHERI_CORE_OFFSET counts into this list, not from core 0.
allowed=()
IFS=, read -r -a ranges <<< "$(awk '/^Cpus_allowed_list:/{print $2}' /proc/self/status)"
for r in "${ranges[@]}"; do
    if [[ "${r}" == *-* ]]; then mapfile -t -O "${#allowed[@]}" allowed < <(seq "${r%-*}" "${r#*-}"); else allowed+=( "${r}" ); fi
done
if (( core_offset + n_slots * cores_per_slot > ${#allowed[@]} )); then
    echo "ERROR: ${n_slots} slots x ${cores_per_slot} cores from core ${core_offset} exceed the ${#allowed[@]} allowed" >&2
    exit 1
fi

box_id="${SLURM_JOB_ID:-local}.${SLURM_STEP_ID:-$$}"
box_dir="${CHERI_ROOT}/box/${box_id}"
mkdir -p "${box_dir}/logs"
status="${box_dir}/status.tsv"
printf 'dataset\tfold\tstages\texit\tseconds\tgpu\n' > "${status}"

# Node-local scratch for temp files and the shared compile caches. A killed job
# skips the EXIT trap, so clear what earlier jobs of ours left first -- but not
# what another run in this same job is still using.
scratch_root=/dev/shm
[[ -d /raid && -w /raid ]] && scratch_root=/raid
scratch_root="${CHERI_SCRATCH_ROOT:-${scratch_root}}"
for old in "${scratch_root}/${USER}/cheri."*; do
    [[ -d "${old}" ]] || continue
    old_job="${old##*/cheri.}"
    [[ "${old_job%%.*}" != "${SLURM_JOB_ID:-local}" ]] && rm -rf "${old}"
done
scratch="${scratch_root}/${USER}/cheri.${box_id}"
export TMPDIR="${scratch}/tmp" TORCHINDUCTOR_CACHE_DIR="${scratch}/inductor" TRITON_CACHE_DIR="${scratch}/triton"
mkdir -p "${TMPDIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

# The queue: ready rows of tasks.tsv, optionally limited to CHERI_ONLY.
queue="${box_dir}/queue.tsv"
awk -F'\t' -v only=" ${CHERI_ONLY:-} " -v stages="${stages}" '
    NR == 1 { next }
    only != "  " && index(only, " " $1 " ") == 0 { next }
    stages == "compare" || $4 == "yes" { print $1 "\t" $2 "\t" $3 }
' "${tasks_tsv}" > "${queue}"
# A second runner beside a first can take the list from the other end
# (CHERI_ORDER=reverse), so the two meet in the middle instead of contending
# for the same next fold (the fold lock would make one of them skip it).
if [[ "${CHERI_ORDER:-forward}" == "reverse" ]]; then
    tac "${queue}" > "${queue}.rev" && mv "${queue}.rev" "${queue}"
fi
n_tasks="$(wc -l < "${queue}")"
echo "[$(date)] box ${box_id} on $(hostname): ${n_tasks} task(s), stages ${stages}," \
     "GPUs ${gpus[*]} x ${slots_per_gpu} slot(s), ${cores_per_slot} cores each from core ${core_offset}"
echo "           logs ${box_dir}/logs"

# Tasks running per GPU, one file each, read and written under the queue's lock.
load_dir="${box_dir}/.gpu_load"
mkdir -p "${load_dir}"
for gpu in "${gpus[@]}"; do echo 0 > "${load_dir}/${gpu}"; done
claims="${box_dir}/claims.tsv"
printf 'dataset\tfold\tgpu\tloads\ttime\n' > "${claims}"

# claim -- print the next queued task and the GPU it runs on, and drop the task from the queue, under a lock. The
# GPU is the one with the fewest tasks running (the first of them in CHERI_GPUS' order on a tie); its count goes up
# by one until release. A free worker always finds one below CHERI_SLOTS_PER_GPU, since workers = GPUs x slots.
claim() {
    (
        flock -x 9
        local line gpu best="" min=-1 n loads=()
        line="$(head -n 1 "${queue}")"
        [[ -z "${line}" ]] && exit 1
        if (( max_tasks > 0 )); then
            local started
            started="$(cat "${box_dir}/.started" 2>/dev/null || echo 0)"
            (( started >= max_tasks )) && exit 1
            echo $(( started + 1 )) > "${box_dir}/.started"
        fi
        for gpu in "${gpus[@]}"; do
            n="$(cat "${load_dir}/${gpu}")"
            loads+=( "${gpu}:${n}" )
            if (( min < 0 || n < min )); then min="${n}"; best="${gpu}"; fi
        done
        echo $(( min + 1 )) > "${load_dir}/${best}"
        sed -i 1d "${queue}"
        printf '%s\t%s\t%s\t%s\n' "$(cut -f1,2 <<< "${line}")" "${best}" "$(IFS=,; echo "${loads[*]}")" \
            "$(date +%FT%T)" >> "${claims}"
        printf '%s\t%s\n' "${line}" "${best}"
    ) 9> "${queue}.lock"
}

# release <gpu> -- one task fewer running on the GPU.
release() {
    (
        flock -x 9
        echo $(( $(cat "${load_dir}/$1") - 1 )) > "${load_dir}/$1"
    ) 9> "${queue}.lock"
}

worker() {
    local cores="$1" task dataset fold workdir gpu rc t0
    while task="$(claim)"; do
        IFS=$'\t' read -r dataset fold workdir gpu <<< "${task}"
        t0=${SECONDS}
        CUDA_VISIBLE_DEVICES="${gpu}" CHERI_THREADS="${cores_per_slot}" \
            taskset -c "${cores}" bash "${task_cmd}" "${workdir}" "${stages}" \
            >> "${box_dir}/logs/${dataset}.fold_${fold}.log" 2>&1
        rc=$?
        release "${gpu}"
        (( rc == 4 )) && continue   # another run has this fold
        (
            flock -x 8
            printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${dataset}" "${fold}" "${stages}" "${rc}" "$(( SECONDS - t0 ))" "${gpu}" >> "${status}"
        ) 8> "${status}.lock"
    done
}

sampler() {
    printf 'time\tmem_used_gb\tload1\tgpus(util%%,mem_gb)\n' > "${box_dir}/resources.tsv"
    while true; do
        printf '%s\t%s\t%s\t%s\n' "$(date +%FT%T)" \
            "$(awk '/^MemTotal:/{t=$2} /^MemAvailable:/{a=$2} END{printf "%d", (t-a)/1048576}' /proc/meminfo)" \
            "$(cut -d' ' -f1 /proc/loadavg)" \
            "$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null \
                | awk -F', ' '{printf "%s%d,%.0f", (NR>1?" ":""), $1, $2/1024}')" >> "${box_dir}/resources.tsv"
        sleep 30
    done
}
sampler &
sampler_pid=$!
trap 'kill "${sampler_pid}" 2>/dev/null; rm -rf "${scratch}"' EXIT

# One worker per slot, each with its own block of cores; each task's GPU is chosen when it is taken (claim).
pids=()
for (( slot = 0; slot < n_slots; slot++ )); do
    first=$(( core_offset + slot * cores_per_slot ))
    worker "$(IFS=,; echo "${allowed[*]:${first}:${cores_per_slot}}")" &
    pids+=( $! )
    # Let the first task compile and fill the cache before the others need it.
    (( slot == 0 && n_tasks > 1 )) && sleep "${CHERI_STAGGER:-120}"
done
wait "${pids[@]}"

awk -F'\t' 'NR > 1 { n[$4 == 0 ? "done" : $4 == 3 ? "waiting" : "failed"]++ }
    END { for (k in n) printf "  %s: %d\n", k, n[k] }' "${status}"
echo "[$(date)] box ${box_id} finished; status ${status}"
