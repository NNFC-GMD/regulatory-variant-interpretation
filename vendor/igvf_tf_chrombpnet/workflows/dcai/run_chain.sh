#!/bin/bash
# run_chain.sh
# Purpose: one dataset's CPU preprocessing, in order, inside a whole-node job.
#   run_box.sh starts one of these per dataset, several at once.
#
#   00.0.call_peaks  ||  00.0.prepare_signal      both only read the fragments
#   -> 00.1.preprocess_peaks -> 01.0.preprocess_nonpeaks -> 02.0.qc_training_data
#
# With CHAIN_STAGES containing "bias", the dataset's 03.0 bias-sweep tasks are
# then queued for run_box.sh's GPU workers (one line per array index, as many
# as 03.0's own load_bias_sweep implies), so GPU training starts as soon as
# this dataset is ready instead of when every dataset is.
#
# Each step is the workflows/SLURM script itself, run with `bash` as molab
# does: nothing here re-implements a step, and each step still skips work
# whose outputs exist, so a rerun resumes. A step's failure stops this
# dataset's chain only.
#
# Input:  $1 = the dataset's config.yaml
#         BOX_LOG_DIR, BOX_STATUS, BOX_SCRATCH, BOX_GPU_QUEUE (from run_box.sh)
#         CHAIN_STAGES: "prep" (default) or "prep,bias"
# Output: the steps' own outputs; <BOX_LOG_DIR>/<dataset>/<step>.log; one
#         "dataset step exit seconds" line per step in BOX_STATUS; a symlink
#         <DATASET_ROOT>/bigwigs/<dataset>.unstranded.bw to the prepared bigwig;
#         with bias, "<config>\t<array index>" lines appended to BOX_GPU_QUEUE
# Usage:  bash workflows/dcai/run_chain.sh <config.yaml>   (normally via run_box.sh)

set -uo pipefail

cfg="${1:?usage: run_chain.sh <config.yaml>}"
name="$(basename "$(dirname "${cfg}")")"
steps_dir="${REPO_ROOT:?source workflows/dcai/env.sh first}/workflows/SLURM"
logdir="${BOX_LOG_DIR:?}/${name}"
mkdir -p "${logdir}"

export DATASET_CONFIG="${cfg}"
export PEAKS_TMPDIR="${BOX_SCRATCH:?}"
# Single-threaded numerics unless a step asks for more: many chains share the node.
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 NUMEXPR_MAX_THREADS=2 NUMBA_NUM_THREADS=2

# run_step <step> [cpus] — run one step, log it, record its exit status and time.
run_step() {
    local step="$1" cpus="${2:-4}" t0 rc
    t0="$(date +%s)"
    SLURM_CPUS_PER_TASK="${cpus}" bash "${steps_dir}/${step}" > "${logdir}/${step%.sh}.log" 2>&1
    rc=$?
    printf '%s\t%s\t%d\t%d\n' "${name}" "${step%.sh}" "${rc}" "$(( $(date +%s) - t0 ))" >> "${BOX_STATUS:?}"
    return "${rc}"
}

echo "[$(date)] ${name}: start"
run_step 00.0.call_peaks.sh 4 &
peaks_pid=$!
# figwig bam2bw's gains end near 8 cores.
run_step 00.0.prepare_signal.sh 8 &
signal_pid=$!
wait "${peaks_pid}"; peaks_rc=$?
wait "${signal_pid}"; signal_rc=$?

bw="${DATASET_ROOT}/chrombpnet/${name}/preprocessing/signal/data_unstranded.bw"
if [[ -f "${bw}" ]]; then
    mkdir -p "${DATASET_ROOT}/bigwigs"
    ln -sfn "${bw}" "${DATASET_ROOT}/bigwigs/${name}.unstranded.bw"
fi
if (( peaks_rc || signal_rc )); then
    echo "[$(date)] ${name}: stopped (call_peaks exit ${peaks_rc}, prepare_signal exit ${signal_rc})"
    exit 1
fi

for step in 00.1.preprocess_peaks.sh 01.0.preprocess_nonpeaks.sh 02.0.qc_training_data.sh; do
    if ! run_step "${step}" 2; then
        echo "[$(date)] ${name}: stopped at ${step}"
        exit 1
    fi
done

if [[ ",${CHAIN_STAGES:-prep}," == *",bias,"* ]]; then
    # The same arithmetic 03.0 does: len(bias_sweep_folds) x the factors
    # load_bias_sweep picks (02.0's scan when it exists, else the config).
    n_tasks="$(
        set +u    # config.sh, like the steps that source it, is not written for nounset
        # shellcheck source=lib/bash/config.sh
        source "${REPO_ROOT}/lib/bash/config.sh" > /dev/null 2>&1 || exit 1
        load_bias_sweep > /dev/null 2>&1 || exit 1
        echo $(( ${#bias_sweep_folds[@]} * ${#bias_factors[@]} ))
    )"
    if [[ ! "${n_tasks}" =~ ^[1-9][0-9]*$ ]]; then
        # e.g. 02.0's scan found no viable bias factor; 03.0 would refuse too.
        echo "[$(date)] ${name}: could not size the 03.0 sweep (see 02.0's bias scan); not queued"
        printf '%s\t%s\t%d\t%d\n' "${name}" "03.0.queue" 1 0 >> "${BOX_STATUS}"
        exit 1
    fi
    (
        flock 9
        for (( i = 0; i < n_tasks; i++ )); do printf '%s\t%d\n' "${cfg}" "${i}"; done >> "${BOX_GPU_QUEUE:?}"
    ) 9> "${BOX_GPU_QUEUE}.lock"
    echo "[$(date)] ${name}: queued ${n_tasks} 03.0 bias-sweep task(s) for the GPUs"
fi
echo "[$(date)] ${name}: done"
