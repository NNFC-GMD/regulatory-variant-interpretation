#!/bin/bash
# run_project.sh
# Purpose: one Mímir project, end to end, inside one whole-node box: from fragment files to Cherimoya models,
#   attributions, TF-MoDISco motifs, a compendium, Fi-NeMo hits and DeepISA cooperativity. docs/pipeline.md
#   describes every step and why it is set up as it is; docs/new_project.md how to start a project.
#
#   preflight     both checkouts clean, igvf_tf_chrombpnet at workflows/pipeline/igvf.pin, every environment and
#                 input present, `pixi run check` and `pixi run -e cherimoya test-cherimoya` pass
#   prep          igvf_tf_chrombpnet: detect_shifts.py (the Tn5 shift in each file, pinned in the configs),
#                 make_configs.py, then run_chain.sh per dataset, all at once:
#                 00.0 call_peaks || 00.0 prepare_signal -> 00.1 -> 01.0 (fold negatives) -> 02.0 (QC)
#   train         make_jobs.py, then train/run_box.sh: `cherimoya fit` for every dataset x fold
#   attribute     make_downstream.py --engine cherimoya, then run_box.sh: `cherimoya attribute` on every peak
#   downstream    downstream/downstream.sh: convert -> check -> 06.0 -> V2 -> 07.0 -> 08.0 (TF-MoDISco)
#   motifs        igvf_tf_chrombpnet motif_chain.sh: 09.0 (the project's compendium) -> 10.0 (Fi-NeMo) -> 11.0
#   cooperativity motif_names.py, coop_isa.py per dataset (one GPU each, lane.sh), combine.py (ours + --official)
#   summary       status.py: <root>/STATUS.md
#
# A stage is skipped when project.py check <stage> finds its outputs complete, and is done only when it does after
# the stage ran: the runners report per task and some always exit 0, so exit codes are not trusted. A resubmit
# therefore resumes where the last box stopped. A stage that fails holds the box for hold_minutes (project.yaml,
# default 60) for a fix through `srun --jobid <job> --overlap`: `touch <box>/retry` runs the stage again, `touch
# <box>/release` ends the box. Every stage's row goes to <root>/box/<job>/stages.tsv; logs to <root>/box/<job>/logs/.
#
# Input:  $1 = the project's project.yaml (workflows/pipeline/project.example.yaml documents it)
#         --stages "<stage> ...": run only these (default: project.yaml's stages); --dry-run: print what would run
# Output: everything under <root> (project.py's docstring lists the layout)
# Usage:  bash workflows/pipeline/submit.sh <project.yaml>       (submits run_project.sbatch, which runs this)
#         bash workflows/pipeline/run_project.sh <project.yaml> --dry-run     (safe on the login node)
# Never edit this file while a box runs it: bash reads it as it goes. Commit, and resubmit.

set -uo pipefail

usage="usage: run_project.sh <project.yaml> [--stages \"<stage> ...\"] [--dry-run]"
project_file="${1:?${usage}}"
shift
dry=0
only=""
while (( $# )); do
    case "$1" in
        --dry-run) dry=1; shift ;;
        --stages) only="${2:?${usage}}"; shift 2 ;;
        *) echo "${usage}" >&2; exit 2 ;;
    esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
here="${repo_root}/workflows/pipeline"
# shellcheck source=workflows/pipeline/lib.sh
source "${here}/lib.sh"
case ":${PATH}:" in
    *":${HOME}/.pixi/bin:"*) ;;
    *) export PATH="${HOME}/.pixi/bin:${PATH}" ;;
esac
export CONDA_OVERRIDE_CUDA="${CONDA_OVERRIDE_CUDA:-13.0}"
rvi_run() { pixi run --frozen --manifest-path "${repo_root}/pixi.toml" -e "$@"; }

project_env="$(rvi_run default python "${here}/project.py" "${project_file}" env)" || exit 2
eval "${project_env}"
stages="${only:-${PROJECT_STAGES}}"

# run <cmd...> -- the command, or with --dry-run its line, on the terminal (fd 3) even when the caller redirects
exec 3>&1
run() {
    if (( dry )); then
        { printf '    $'; printf ' %q' "$@"; printf '\n'; } >&3
        return 0
    fi
    "$@"
}

job="${SLURM_JOB_ID:-local.$$}"
box="${PROJECT_BOX}/${job}"
scratch_root=/dev/shm
[[ -d /raid && -w /raid ]] && scratch_root=/raid
scratch="${scratch_root}/${USER}/pipeline.${job}"
if (( dry )); then
    box="$(mktemp -d)"    # a dry run writes nothing under the project
    trap 'rm -rf "${box}"' EXIT
else
    mkdir -p "${scratch}/prep"
    trap 'rm -rf "${scratch}"' EXIT
fi
logs="${box}/logs"
record="${box}/stages.tsv"
mkdir -p "${logs}"

# igvf_tf_chrombpnet's steps read their environment from its workflows/dcai/env.sh; the project overrides where
# the data and the chrombpnet environment are. env.sh sets REPO_ROOT to the igvf checkout.
export DATASET_ROOT="${PROJECT_ROOT}"
export CHROMBPNET_REPO="${PROJECT_CHROMBPNET}" CHROMBPNET_PIXI_ENV="${PROJECT_CHROMBPNET_ENV}"
# shellcheck source=/dev/null
source "${PROJECT_IGVF}/workflows/dcai/env.sh"
unset METADATA_DIR
igvf_run() { pixi run --frozen --manifest-path "${PROJECT_IGVF}/pixi.toml" -e "$@"; }

# ── stages ───────────────────────────────────────────────────────────────────────────────────────────────────────

stage_preflight() {
    local want have bad=0 e d k
    log "RVI $(code_state "${repo_root}") (${repo_root}); igvf_tf_chrombpnet $(code_state "${PROJECT_IGVF}")" \
        "(${PROJECT_IGVF}); chrombpnet $(code_state "${PROJECT_CHROMBPNET}")"
    if [[ "$(code_state "${repo_root}")" == *+changes* && "${PIPE_ALLOW_DIRTY:-0}" != 1 ]]; then
        log "ERROR: ${repo_root} has uncommitted changes: commit them, so the run is reproducible"; bad=1
    fi
    want="$(awk 'NF && $1 !~ /^#/ { print $1; exit }' "${here}/igvf.pin")"
    have="$(git -C "${PROJECT_IGVF}" rev-parse HEAD 2> /dev/null)"
    if [[ "${have}" != "${want}" ]]; then
        log "ERROR: ${PROJECT_IGVF} is at ${have:-?}, igvf.pin wants ${want}"; bad=1
    elif [[ "$(code_state "${PROJECT_IGVF}")" == *+changes* ]]; then
        log "ERROR: ${PROJECT_IGVF} has uncommitted changes"; bad=1
    fi
    for e in default cherimoya; do [[ -d "${repo_root}/.pixi/envs/${e}" ]] || { log "ERROR: no RVI env ${e}: pixi install -e ${e}"; bad=1; }; done
    for e in peaks preprocess qc modisco motif-compendium finemo; do
        [[ -d "${PROJECT_IGVF}/.pixi/envs/${e}" ]] || { log "ERROR: no igvf env ${e}"; bad=1; }
    done
    [[ -x "${PROJECT_CHROMBPNET}/.pixi/envs/${PROJECT_CHROMBPNET_ENV}/bin/python" ]] \
        || { log "ERROR: no chrombpnet env ${PROJECT_CHROMBPNET_ENV} in ${PROJECT_CHROMBPNET}"; bad=1; }
    [[ -s "${PROJECT_GENOME}" && -s "${PROJECT_GENOME}.fai" ]] || { log "ERROR: no genome or .fai ${PROJECT_GENOME}"; bad=1; }
    for k in ${PROJECT_FOLDS}; do
        [[ -s "${PROJECT_FOLDS_DIR}/fold_${k}.json" ]] || { log "ERROR: no ${PROJECT_FOLDS_DIR}/fold_${k}.json"; bad=1; }
    done
    if ! rvi_run default python "${here}/project.py" "${project_file}" check fetch; then
        log "ERROR: inputs missing: run fetch_igvf.py on the login node first"; bad=1
    fi
    d="$(nvidia-smi -L 2> /dev/null | wc -l)"
    log "GPUs: ${d}; CPUs allowed: $(allowed_cpus | wc -l); scratch ${scratch}"
    (( bad )) && return 1
    (( dry )) && { run pixi run check; run pixi run -e cherimoya test-pipeline; run pixi run -e cherimoya test-cherimoya; return 0; }
    log "tests: default env (lint, shellcheck, pytest), then the pipeline's tests in the cherimoya env"
    (cd "${repo_root}" && taskset -c "$(cpu_block 0 32)" pixi run --frozen check) > "${logs}/preflight_check.log" 2>&1 \
        || { log "ERROR: pixi run check failed: ${logs}/preflight_check.log"; return 1; }
    (cd "${repo_root}" && CUDA_VISIBLE_DEVICES="" taskset -c "$(cpu_block 0 32)" pixi run --frozen -e cherimoya test-pipeline) \
        > "${logs}/preflight_test_pipeline.log" 2>&1 \
        || { log "ERROR: test-pipeline failed: ${logs}/preflight_test_pipeline.log"; return 1; }
    log "tests pass: $(tail -n 1 "${logs}/preflight_check.log"); $(tail -n 1 "${logs}/preflight_test_pipeline.log")"
    # Every other test in the cherimoya env (Cherimoya 0.3.1), beside the stages on a few CPUs: reported, not required,
    # since only test-pipeline's cover what this driver runs.
    (cd "${repo_root}" && CUDA_VISIBLE_DEVICES="" nice -n 10 taskset -c "$(cpu_block 1 8)" \
        pixi run --frozen -e cherimoya test-cherimoya > "${logs}/test_cherimoya_full.log" 2>&1; \
        echo "exit $?" >> "${logs}/test_cherimoya_full.log") &
}

stage_prep() {
    if [[ ! -s "${PROJECT_SHIFTS}" ]]; then
        log "prep: the Tn5 shift of every fragments file (every ${PROJECT_SHIFT_EVERY}th fragment, genome-wide)"
        mkdir -p "$(dirname "${PROJECT_SHIFTS}")" 2> /dev/null
        (cd "${PROJECT_IGVF}" && run igvf_run peaks python workflows/dcai/detect_shifts.py \
            --fragments-dir "${PROJECT_FRAGMENTS_DIR}" --genome "${PROJECT_GENOME}" \
            --out "${PROJECT_SHIFTS}.partial" --jobs "${PROJECT_N_DATASETS}" --every "${PROJECT_SHIFT_EVERY}" \
            --tmp-dir "${scratch}" \
            --chrombpnet-python "${PROJECT_CHROMBPNET}/.pixi/envs/${PROJECT_CHROMBPNET_ENV}/bin/python") \
            > "${logs}/detect_shifts.log" 2>&1 \
            || { log "ERROR: the shift is not trustworthy or detection failed: ${logs}/detect_shifts.log"; return 1; }
        (( dry )) || mv "${PROJECT_SHIFTS}.partial" "${PROJECT_SHIFTS}"
    fi
    run igvf_run preprocess python "${PROJECT_IGVF}/workflows/dcai/make_configs.py" \
        --fragments-dir "${PROJECT_FRAGMENTS_DIR}" --dataset-root "${PROJECT_ROOT}" --shifts "${PROJECT_SHIFTS}" \
        --prefix "${PROJECT_PREFIX}" > "${box}/configs.txt" 2> "${logs}/make_configs.log" \
        || { log "ERROR: make_configs.py: ${logs}/make_configs.log"; return 1; }
    (( dry )) && { run xargs -P "${PROJECT_N_DATASETS}" -I{} bash "${PROJECT_IGVF}/workflows/dcai/run_chain.sh" {}; return 0; }
    log "prep: $(wc -l < "${box}/configs.txt") chains at once (00.0 call_peaks || prepare_signal -> 00.1 -> 01.0 -> 02.0)"
    BOX_LOG_DIR="${logs}/prep" BOX_STATUS="${box}/prep_status.tsv" BOX_SCRATCH="${scratch}/prep" CHAIN_STAGES=prep \
        xargs -P "${PROJECT_N_DATASETS}" -I{} bash "${PROJECT_IGVF}/workflows/dcai/run_chain.sh" {} \
        < "${box}/configs.txt" > "${logs}/prep.log" 2>&1
}

stage_train() {
    run rvi_run default python "${repo_root}/workflows/train/make_jobs.py" \
        --chrombpnet-configs "${PROJECT_CONFIGS_DIR}"/*/config.yaml --folds-dir "${PROJECT_FOLDS_DIR}" \
        --genome "${PROJECT_GENOME}" --out-root "${PROJECT_CHERI_ROOT}" > "${logs}/make_jobs.log" 2>&1 \
        || { log "ERROR: make_jobs.py: ${logs}/make_jobs.log"; return 1; }
    log "train: cherimoya fit, ${SLOT_TRAIN_PER_GPU} per GPU, ${SLOT_TRAIN_CORES} cores each"
    run env CHERI_ROOT="${PROJECT_CHERI_ROOT}" CHERI_STAGES=fit CHERI_SLOTS_PER_GPU="${SLOT_TRAIN_PER_GPU}" \
        CHERI_CORES_PER_SLOT="${SLOT_TRAIN_CORES}" CHERI_PIXI_MANIFEST="${repo_root}/pixi.toml" \
        bash "${repo_root}/workflows/train/run_box.sh" > "${logs}/train.log" 2>&1
}

stage_attribute() {
    (( dry )) || { [[ -e "${PROJECT_SHARED}" ]] || : > "${PROJECT_SHARED}"; }
    run rvi_run default python "${repo_root}/workflows/downstream/make_downstream.py" \
        --cheri-root "${PROJECT_CHERI_ROOT}" --chrombpnet-configs "${PROJECT_CONFIGS_DIR}"/*/config.yaml \
        --shared "${PROJECT_SHARED}" --compendia "${PROJECT_COMPENDIA}" --engine cherimoya \
        > "${logs}/make_downstream.log" 2>&1 \
        || { log "ERROR: make_downstream.py: ${logs}/make_downstream.log"; return 1; }
    log "attribute: cherimoya attribute (fast path), ${SLOT_ATTRIBUTE_PER_GPU} per GPU, ${SLOT_ATTRIBUTE_CORES} cores each"
    run env CHERI_ROOT="${PROJECT_CHERI_ROOT}" CHERI_STAGES=attribute CHERI_ATTR_ENGINE=cherimoya \
        CHERI_SLOTS_PER_GPU="${SLOT_ATTRIBUTE_PER_GPU}" CHERI_CORES_PER_SLOT="${SLOT_ATTRIBUTE_CORES}" \
        CHERI_PIXI_MANIFEST="${repo_root}/pixi.toml" \
        bash "${repo_root}/workflows/train/run_box.sh" > "${logs}/attribute.log" 2>&1
    (( dry )) || grep -h "FELL BACK" "${PROJECT_CHERI_ROOT}"/box/*/logs/*.log 2> /dev/null | sed 's/^/    /' | head -n 25
}

stage_downstream() {
    log "downstream: convert -> check -> 06.0 -> V2 -> 07.0 -> 08.0 (${SLOT_MODISCO_MAX} TF-MoDISco at once," \
        "${SLOT_MODISCO_THREADS} threads each)"
    run env CHERI_ROOT="${PROJECT_CHERI_ROOT}" BOX_DIR="${box}" CHERI_SHARED_LIST="${PROJECT_SHARED}" \
        CHERI_08_MAX="${SLOT_MODISCO_MAX}" CHERI_08_THREADS="${SLOT_MODISCO_THREADS}" CHERI_DOWNSTREAM_PRODUCERS="" \
        CHERI_PIXI_MANIFEST="${repo_root}/pixi.toml" \
        bash "${repo_root}/workflows/downstream/downstream.sh" "${PROJECT_DATASET_LIST}" > "${logs}/downstream.log" 2>&1
}

stage_motifs() {
    log "motifs: compendium ${PROJECT_COMPENDIUM##*/}, then Fi-NeMo (${SLOT_FINEMO_SLOTS} at once) and its report"
    run env BOX_DIR="${box}" CHAIN_FINEMO_SLOTS="${SLOT_FINEMO_SLOTS}" FINEMO_BATCH_SIZE="${SLOT_FINEMO_BATCH}" \
        MOTIF_MIN_AGE_S=0 MOTIF_POLL_S=60 \
        bash "${PROJECT_IGVF}/workflows/dcai/motif_chain.sh" "${PROJECT_MOTIF_LIST}" "${PROJECT_COMPENDIUM}" \
        "${PROJECT_MOTIF_LIST}" > "${logs}/motifs.log" 2>&1
}

stage_cooperativity() {
    local i=0 ds gpu cpus pids=() n_gpus
    run rvi_run cherimoya python "${repo_root}/workflows/cooperativity/motif_names.py" \
        --compendium "${PROJECT_COMPENDIUM}" --out "${PROJECT_COOP}/motifs.tsv" > "${logs}/motif_names.log" 2>&1 \
        || { log "ERROR: motif_names.py: ${logs}/motif_names.log"; return 1; }
    run rvi_run default python "${here}/project.py" "${project_file}" coop-inputs > "${logs}/coop_inputs.log" 2>&1 \
        || { log "ERROR: project.py coop-inputs: ${logs}/coop_inputs.log"; return 1; }
    n_gpus="$(nvidia-smi -L 2> /dev/null | wc -l)"
    (( n_gpus > 0 )) || n_gpus=1
    log "cooperativity: coop_isa.py for ${PROJECT_N_DATASETS} datasets, one lane per GPU"
    for ds in ${PROJECT_DATASETS}; do
        if [[ -s "${PROJECT_COOP}/${ds}/run.json" ]]; then
            log "  ${ds}: done, skipped"
            continue
        fi
        gpu=$(( i % n_gpus ))
        cpus="$(cpu_block "${i}" "${SLOT_COOP_CORES}")"
        [[ -n "${cpus}" ]] || cpus="$(cpu_block 0 "${SLOT_COOP_CORES}")"
        if (( dry )); then
            run bash "${repo_root}/workflows/cooperativity/lane.sh" "${gpu}" "${cpus}" "${PROJECT_COOP}/logs" \
                "${PROJECT_COOP}/args/${ds}.args"
        else
            mkdir -p "${PROJECT_COOP}/logs"
            COOP_PYTHON="${repo_root}/.pixi/envs/cherimoya/bin/python" \
                bash "${repo_root}/workflows/cooperativity/lane.sh" "${gpu}" "${cpus}" "${PROJECT_COOP}/logs" \
                "${PROJECT_COOP}/args/${ds}.args" > "${PROJECT_COOP}/logs/lane_${ds}.log" 2>&1 &
            pids+=( $! )
        fi
        i=$(( i + 1 ))
    done
    (( ${#pids[@]} )) && wait "${pids[@]}"
    run rvi_run cherimoya python "${repo_root}/workflows/cooperativity/combine.py" --root "${PROJECT_COOP}" \
        --motifs "${PROJECT_COOP}/motifs.tsv" --groups "${PROJECT_COOP}/groups.tsv" \
        --comparisons "${PROJECT_COOP}/comparisons.tsv" --official > "${logs}/combine.log" 2>&1 \
        || { log "ERROR: combine.py: ${logs}/combine.log"; return 1; }
}

stage_summary() {
    run rvi_run default python "${here}/status.py" "${project_file}" --out "${PROJECT_ROOT}/STATUS.md"
}

# ── the run ──────────────────────────────────────────────────────────────────────────────────────────────────────

checked() {  # checked <stage>: 0 when project.py finds its outputs complete
    rvi_run default python "${here}/project.py" "${project_file}" check "$1" > "${logs}/check.$1.log" 2>&1
}

log "project ${PROJECT} (${PROJECT_FILE}), box ${job} on $(hostname), stages: ${stages}$( (( dry )) && echo ' (dry run)')"
(( dry )) && rvi_run default python "${here}/project.py" "${project_file}" status | sed 's/^/    /'
for stage in preflight ${stages} summary; do
    while true; do
        if [[ "${stage}" != preflight && "${stage}" != summary ]] && checked "${stage}"; then
            log "${stage}: outputs complete, skipped"
            (( dry )) || run_rec "${record}" "${stage}" skipped 0 0 "$(et)" "outputs complete"
            break
        fi
        t0="$(date +%s)"; started="$(et)"
        log "${stage}: start"
        "stage_${stage}"
        rc=$?
        secs=$(( $(date +%s) - t0 ))
        if (( dry )); then
            break
        fi
        if [[ "${stage}" == preflight || "${stage}" == summary ]]; then
            (( rc == 0 )) && ok=1 || ok=0
        else
            checked "${stage}" && ok=1 || ok=0
        fi
        if (( ok )); then
            log "${stage}: done in $(( secs / 60 )) min"
            run_rec "${record}" "${stage}" "done" "${rc}" "${secs}" "${started}" ""
            break
        fi
        log "${stage}: NOT done (exit ${rc}); missing:"
        sed 's/^/    /' "${logs}/check.${stage}.log" 2> /dev/null | head -n 20
        run_rec "${record}" "${stage}" failed "${rc}" "${secs}" "${started}" "$(head -n 1 "${logs}/check.${stage}.log" 2> /dev/null)"
        if hold "${box}" "${PROJECT_HOLD_MIN}" "${stage} failed (exit ${rc})"; then
            continue
        fi
        log "ending the box: ${stage} is not done (see ${record})"
        exit 1
    done
done
log "project ${PROJECT}: every stage done$( (( dry )) && echo ' (dry run: nothing ran)')"
