#!/bin/bash
# run_project.sh
# Purpose: one Mímir project, end to end, inside one whole-node box: from fragment files to Cherimoya models,
#   attributions, TF-MoDISco motifs, a compendium, Fi-NeMo hits and DeepISA cooperativity. docs/pipeline.md
#   describes every step and why it is set up as it is; docs/new_project.md how to start a project.
#
#   preflight     the checkout clean, vendor/igvf_tf_chrombpnet as its MANIFEST.tsv says, every environment,
#                 reference file, tool and input present, a GPU on an NVIDIA driver >= 580, the slots within the
#                 machine; `pixi run check` and `pixi run -e cherimoya test-pipeline` pass
#   prep          igvf_tf_chrombpnet (vendored): detect_shifts.py (the Tn5 shift in each file, pinned in the
#                 configs), make_configs.py, then run_chain.sh per dataset, slots.prep_max at once:
#                 00.0 call_peaks || 00.0 prepare_signal -> 00.1 -> 01.0 (fold negatives) -> 02.0 (QC)
#   train         make_jobs.py, then train/run_box.sh: `cherimoya fit` for every dataset x fold
#   attribute     make_downstream.py --engine cherimoya, then run_box.sh: `cherimoya attribute` on every peak
#   downstream    downstream/downstream.sh: convert -> check -> 06.0 -> V2 -> 07.0 -> 08.0 (TF-MoDISco)
#   motifs        igvf_tf_chrombpnet (vendored) motif_chain.sh: 09.0 (the project's compendium) -> 10.0 (Fi-NeMo)
#                 -> 11.0
#   cooperativity motif_names.py, coop_isa.py per dataset (one GPU each, lane.sh), combine.py (ours + --official)
#   coop_matrix   (optional: only when project.yaml lists it) cooperativity/matrix.py: every dataset's sites under
#                 every dataset's models, the trends along the series, the paired changes (paired.py)
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
# Usage:  bash workflows/pipeline/submit.sh <project.yaml>       (Slurm: submits run_project.sbatch, which runs this)
#         bash workflows/pipeline/run_project.sh <project.yaml>   (no Slurm: on the machine itself, under tmux/nohup)
#         bash workflows/pipeline/run_project.sh <project.yaml> --dry-run     (safe on a login node)
# The machine: every GPU nvidia-smi lists, or those CUDA_VISIBLE_DEVICES names; the CPUs this process may use; the
# memory (cgroup limit or MemTotal). The slots project.yaml does not give are scaled to it (project.py
# resolve_slots). docs/install.md has what a machine needs.
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
# The GPUs every stage uses: run_box.sh reads CHERI_GPUS, the cooperativity lanes this list.
IFS=, read -r -a gpu_list <<< "${PROJECT_GPUS}"
[[ -n "${PROJECT_GPUS}" ]] && export CHERI_GPUS="${PROJECT_GPUS}"

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

# igvf_tf_chrombpnet's steps (vendor/igvf_tf_chrombpnet/, copied unchanged) read their environment from its
# workflows/dcai/env.sh, whose defaults are Gefion's; everything they use is set here first, so none of those applies:
# the copy as REPO_ROOT, the project's data, the reference files (the same reference_root project.py takes the genome
# from), chrombpnet as this repository's `chrombpnet` env, and a python for their stdlib-only helpers.
export REPO_ROOT="${PROJECT_VENDOR}" DATASET_ROOT="${PROJECT_ROOT}" REFERENCE_ROOT="${PROJECT_REFERENCE_ROOT}"
export CHROMBPNET_ENV="pixi:${repo_root}/pixi.toml#chrombpnet"
export BOOTSTRAP_PYTHON="${repo_root}/.pixi/envs/default/bin/python"
chrombpnet_python="${repo_root}/.pixi/envs/chrombpnet/bin/python"
# shellcheck source=/dev/null
source "${PROJECT_VENDOR}/workflows/dcai/env.sh"
unset METADATA_DIR
igvf_run() { pixi run --frozen --manifest-path "${PROJECT_VENDOR}/pixi.toml" -e "$@"; }
# The bare `python3` of motif_chain.sh and downstream.sh (stdlib-only helpers) is the default env's, so the machine
# needs no python of its own.
shims="${scratch}/bin"
(( dry )) && shims="${box}/bin"
mkdir -p "${shims}"
ln -sfn "${BOOTSTRAP_PYTHON}" "${shims}/python3"
export PATH="${shims}:${PATH}"

# ── stages ───────────────────────────────────────────────────────────────────────────────────────────────────────

stage_preflight() {
    local bad=0 e f k t n_gpus driver
    log "RVI $(code_state "${repo_root}") (${repo_root}); igvf_tf_chrombpnet" \
        "$(awk -F'\t' '$1 == "# commit" { print substr($2, 1, 7) }' "${PROJECT_VENDOR}/MANIFEST.tsv"), vendored"
    if [[ "$(code_state "${repo_root}")" == *+changes* && "${PIPE_ALLOW_DIRTY:-0}" != 1 ]]; then
        log "ERROR: ${repo_root} has uncommitted changes: commit them, so the run is reproducible"; bad=1
    fi
    if ! (cd "${PROJECT_VENDOR}" && awk -F'\t' '$1 !~ /^#/ && $1 != "path" { print $2 "  " $1 }' MANIFEST.tsv \
            | sha256sum --quiet -c -) > "${logs}/vendor_check.log" 2>&1; then
        log "ERROR: vendor/igvf_tf_chrombpnet differs from its MANIFEST.tsv: ${logs}/vendor_check.log"; bad=1
    fi
    for e in default cherimoya chrombpnet; do
        [[ -d "${repo_root}/.pixi/envs/${e}" ]] || { log "ERROR: no env ${e}: bash workflows/pipeline/setup.sh"; bad=1; }
    done
    for e in peaks preprocess modisco motif-compendium finemo; do
        [[ -d "${PROJECT_VENDOR}/.pixi/envs/${e}" ]] \
            || { log "ERROR: no vendored igvf env ${e}: bash workflows/pipeline/setup.sh"; bad=1; }
    done
    for f in ${PROJECT_REFERENCE_FILES}; do
        [[ -s "${f}" ]] || { log "ERROR: no reference file ${f}: bash workflows/pipeline/setup.sh --references" \
            "${PROJECT_REFERENCE_ROOT}"; bad=1; }
    done
    for k in ${PROJECT_FOLDS}; do
        [[ -s "${PROJECT_FOLDS_DIR}/fold_${k}.json" ]] || { log "ERROR: no ${PROJECT_FOLDS_DIR}/fold_${k}.json"; bad=1; }
    done
    [[ -x /usr/bin/time ]] || { log "ERROR: no /usr/bin/time (GNU time, the package 'time'), which downstream.sh needs"; bad=1; }
    for t in taskset flock pgrep nvidia-smi awk gzip sha256sum; do
        [[ "${t}" == nvidia-smi ]] && (( dry )) && continue
        command -v "${t}" > /dev/null || { log "ERROR: ${t} is not on PATH"; bad=1; }
    done
    [[ -d /dev/shm && -w /dev/shm ]] || { log "ERROR: /dev/shm is not writable (node-local scratch of the steps)"; bad=1; }
    n_gpus="${#gpu_list[@]}"
    if (( n_gpus == 0 )); then
        if (( dry )); then
            log "no GPU here: a dry run needs none, a run at least one"
        else
            log "ERROR: no GPU (nvidia-smi lists none, or CUDA_VISIBLE_DEVICES names none)"; bad=1
        fi
    else
        driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2> /dev/null | head -n 1)"
        if (( ${driver%%.*} < 580 )) 2> /dev/null || [[ -z "${driver}" ]]; then
            log "ERROR: NVIDIA driver ${driver:-?} is below 580: the cherimoya and finemo envs are CUDA 13 builds"; bad=1
        fi
        for k in train attribute; do
            t="SLOT_${k^^}_PER_GPU"; e="SLOT_${k^^}_CORES"
            if (( n_gpus * ${!t} * ${!e} > PROJECT_CPUS )); then
                log "ERROR: ${k}: ${n_gpus} GPUs x ${!t} per GPU x ${!e} cores is more than the ${PROJECT_CPUS} CPUs:" \
                    "lower slots.${k}_per_gpu or slots.${k}_cores in project.yaml"; bad=1
            fi
        done
    fi
    if ! rvi_run default python "${here}/project.py" "${project_file}" check fetch > /dev/null; then
        log "ERROR: inputs missing: run fetch_igvf.py first"; bad=1
    fi
    log "machine: GPUs ${PROJECT_GPUS:-none} (driver ${driver:-?}); CPUs ${PROJECT_CPUS}; memory ${PROJECT_MEM_GB} GB;" \
        "scratch ${scratch}"
    log "slots: train ${SLOT_TRAIN_PER_GPU}/GPU x ${SLOT_TRAIN_CORES} cores; attribute ${SLOT_ATTRIBUTE_PER_GPU}/GPU x" \
        "${SLOT_ATTRIBUTE_CORES}; prep ${SLOT_PREP_MAX} at once; TF-MoDISco ${SLOT_MODISCO_MAX} x" \
        "${SLOT_MODISCO_THREADS} threads; Fi-NeMo ${SLOT_FINEMO_SLOTS} (batch ${SLOT_FINEMO_BATCH}); DeepISA" \
        "${SLOT_COOP_CORES} cores per lane"
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
    # since only test-pipeline's cover what this driver runs. Not on a machine of fewer than 16 CPUs.
    if [[ -z "$(cpu_block 1 8)" ]] || (( PROJECT_CPUS < 16 )); then
        log "test-cherimoya: skipped (${PROJECT_CPUS} CPUs)"
        return 0
    fi
    (cd "${repo_root}" && CUDA_VISIBLE_DEVICES="" nice -n 10 taskset -c "$(cpu_block 1 8)" \
        pixi run --frozen -e cherimoya test-cherimoya > "${logs}/test_cherimoya_full.log" 2>&1; \
        echo "exit $?" >> "${logs}/test_cherimoya_full.log") &
}

stage_prep() {
    if [[ ! -s "${PROJECT_SHIFTS}" ]]; then
        log "prep: the Tn5 shift of every fragments file (every ${PROJECT_SHIFT_EVERY}th fragment, genome-wide)"
        mkdir -p "$(dirname "${PROJECT_SHIFTS}")" 2> /dev/null
        (cd "${PROJECT_VENDOR}" && run igvf_run peaks python workflows/dcai/detect_shifts.py \
            --fragments-dir "${PROJECT_FRAGMENTS_DIR}" --genome "${PROJECT_GENOME}" \
            --out "${PROJECT_SHIFTS}.partial" --jobs "${PROJECT_N_DATASETS}" --every "${PROJECT_SHIFT_EVERY}" \
            --tmp-dir "${scratch}" \
            --chrombpnet-python "${chrombpnet_python}") \
            > "${logs}/detect_shifts.log" 2>&1 \
            || { log "ERROR: the shift is not trustworthy or detection failed: ${logs}/detect_shifts.log"; return 1; }
        (( dry )) || mv "${PROJECT_SHIFTS}.partial" "${PROJECT_SHIFTS}"
    fi
    run igvf_run preprocess python "${PROJECT_VENDOR}/workflows/dcai/make_configs.py" \
        --fragments-dir "${PROJECT_FRAGMENTS_DIR}" --dataset-root "${PROJECT_ROOT}" --shifts "${PROJECT_SHIFTS}" \
        --prefix "${PROJECT_PREFIX}" > "${box}/configs.txt" 2> "${logs}/make_configs.log" \
        || { log "ERROR: make_configs.py: ${logs}/make_configs.log"; return 1; }
    (( dry )) && { run xargs -P "${SLOT_PREP_MAX}" -I{} bash "${PROJECT_VENDOR}/workflows/dcai/run_chain.sh" {}; return 0; }
    log "prep: $(wc -l < "${box}/configs.txt") chains, ${SLOT_PREP_MAX} at once (00.0 call_peaks || prepare_signal ->" \
        "00.1 -> 01.0 -> 02.0)"
    BOX_LOG_DIR="${logs}/prep" BOX_STATUS="${box}/prep_status.tsv" BOX_SCRATCH="${scratch}/prep" CHAIN_STAGES=prep \
        xargs -P "${SLOT_PREP_MAX}" -I{} bash "${PROJECT_VENDOR}/workflows/dcai/run_chain.sh" {} \
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
    # The tasks that fell back to the default path, if any (none is the expected case, not a failure).
    (( dry )) || grep -h "FELL BACK" "${PROJECT_CHERI_ROOT}"/box/*/logs/*.log 2> /dev/null | sed 's/^/    /' | head -n 25
    return 0
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
        bash "${PROJECT_VENDOR}/workflows/dcai/motif_chain.sh" "${PROJECT_MOTIF_LIST}" "${PROJECT_COMPENDIUM}" \
        "${PROJECT_MOTIF_LIST}" > "${logs}/motifs.log" 2>&1
}

stage_cooperativity() {
    local i=0 ds gpu cpus pids=() lanes=("${gpu_list[@]}")
    run rvi_run cherimoya python "${repo_root}/workflows/cooperativity/motif_names.py" \
        --compendium "${PROJECT_COMPENDIUM}" --out "${PROJECT_COOP}/motifs.tsv" > "${logs}/motif_names.log" 2>&1 \
        || { log "ERROR: motif_names.py: ${logs}/motif_names.log"; return 1; }
    run rvi_run default python "${here}/project.py" "${project_file}" coop-inputs > "${logs}/coop_inputs.log" 2>&1 \
        || { log "ERROR: project.py coop-inputs: ${logs}/coop_inputs.log"; return 1; }
    (( ${#lanes[@]} )) || lanes=(0)
    log "cooperativity: coop_isa.py for ${PROJECT_N_DATASETS} datasets, one lane per GPU (${lanes[*]})"
    for ds in ${PROJECT_DATASETS}; do
        if [[ -s "${PROJECT_COOP}/${ds}/run.json" ]]; then
            log "  ${ds}: done, skipped"
            continue
        fi
        gpu="${lanes[$(( i % ${#lanes[@]} ))]}"
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
    # STATUS.md now, so the results are summarised even if a later stage holds the box
    run rvi_run default python "${here}/status.py" "${project_file}" --out "${PROJECT_ROOT}/STATUS.md" \
        > "${logs}/status_cooperativity.log" 2>&1 || true
}

stage_coop_matrix() {
    # One call: everything else is in matrix.py, which can be fixed while the box runs (this file cannot).
    log "coop_matrix: every dataset's sites under every dataset's models, then trends and paired changes;" \
        "RVI $(code_state "${repo_root}")"
    run rvi_run cherimoya python "${repo_root}/workflows/cooperativity/matrix.py" --root "${PROJECT_COOP}" \
        --cores "${SLOT_COOP_CORES}" --official > "${logs}/coop_matrix.log" 2>&1 \
        || { log "ERROR: matrix.py: ${logs}/coop_matrix.log"; return 1; }
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
