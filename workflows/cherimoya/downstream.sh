#!/bin/bash
# downstream.sh
# Purpose: Cherimoya's fold attributions through igvf_tf_chrombpnet's own 06.0,
#   07.0 and 08.0, dataset by dataset, inside a box that runs other work too.
#
# For each dataset of the lists, in list order (the order is the priority, but a
#   dataset whose attribution is not done yet does not hold back the ones after
#   it), once attr_status.py calls every fold of its Cherimoya config DONE:
#   convert  to_chrombpnet_scores.py on every fold at once (2 CPUs each, under
#            the fold's .task.lock; a lock held elsewhere is retried)
#   check    to_chrombpnet_scores.py --check-dataset: the folds' masks and beds
#            agree, then V1 against ChromBPNet's own 05.0 files for a dataset in
#            the shared list, against the genome for the others
#   06.0     the igvf checkout's 06.0.average_contrib_scores.sh (8 threads)
#   v2       to_chrombpnet_scores.py --check-average: 06.0's average recomputed
#   07.0     07.0.contribs_to_bigwig.sh (8 threads), beside
#   08.0     08.0.run_modisco.sh (CHERI_08_THREADS threads), at most CHERI_08_MAX
#            at once on the node; a dataset waiting for a run goes in list order
# convert, check, 06.0 and v2 run in this script one dataset at a time; 07.0 and
#   08.0 then run in the background while the next datasets go ahead.
#
# The igvf steps run unchanged, as workflows/dcai/full_chain.sh's step() runs
#   them: DATASET_CONFIG=<root>/configs/<dataset>/config.yaml (make_downstream.py),
#   SLURM_ARRAY_TASK_ID=0, SLURM_CPUS_PER_TASK and the OMP/NUMBA/MKL/OPENBLAS thread
#   counts set to the step's threads, no GPU (CUDA_VISIBLE_DEVICES empty), under a
#   taskset mask of as many CPUs. The CPUs are picked when the step starts by
#   pick_cores.py: the least busy of those this script may use, avoiding CPUs
#   other processes are pinned to and CPUs other steps of this script hold.
#
# Done markers, in <output_dir>/.done/ (output_dir from the dataset's config):
#   convert_<k>, check, 06.0, v2, 07.0, 08.0. Each is written after the step
#   exits 0 with its output present; a step is done when its marker and its
#   output are both there. On start, an output without its marker (06.0's h5,
#   07.0's bigwig, 08.0's results h5 and report) is renamed
#   <name>.incomplete.<time> and the step runs again, and so do the steps after
#   it. Nothing is deleted. convert and check run again until 06.0 is done
#   (both are quick when nothing changed); once it is, they are not run again,
#   and a fold attributed again since its conversion stops the dataset (its
#   provenance no longer matches the attribution files): move its downstream
#   outputs and .done/ aside to redo it.
# 08.0's report: when 08.0 exits non-zero but its results h5 opens and has
#   pos_patterns (report-simple failed after modisco motifs wrote it), the run
#   is recorded as report_failed and marked done.
#
# Locks (node-local, in /dev/shm/$USER/box.<job>/cheri_downstream/): one per
#   dataset, taken here and inherited by its 07.0 and 08.0, so two runs of this
#   script never work on the same dataset (a dataset another run holds is
#   skipped and looked at again later); and the 08.0 slots, shared by every run
#   on the node. Before 08.0 starts, a `modisco motifs` already reading the
#   dataset's average also makes it wait.
#
# Waiting: a dataset whose folds are not all DONE waits while a producer's hold
#   file exists (CHERI_DOWNSTREAM_PRODUCERS, default the attribution runners'
#   ${BOX_DIR}/.hold.cherimoya.attr.*), polling every CHERI_DOWNSTREAM_POLL
#   seconds. With no producer left it is given up, and so is a dataset whose
#   conversion, check, 06.0, V2, 07.0 or 08.0 fails. This exits once every
#   dataset is done or given up: 0 if all are done, 1 otherwise.
#
# SIGTERM, SIGINT or SIGHUP stops every step this run started and exits 143;
#   the steps stopped write no markers, so the next run redoes them.
#
# Input:  dataset lists, one dataset per line (blank lines and # comments
#   skipped). Environment: CHERI_ROOT (make_jobs.py's --out-root, with the
#   configs make_downstream.py wrote), BOX_DIR (the box's directory: the record,
#   the producers' holds), REPO_ROOT (the igvf_tf_chrombpnet checkout whose
#   workflows/SLURM steps run, with what its steps need from the environment:
#   workflows/dcai/env.sh, CHROMBPNET_REPO, CHROMBPNET_PIXI_ENV).
# Knobs:  CHERI_PIXI_MANIFEST  pixi.toml whose cherimoya env runs the converter
#                              and the checks (default: this checkout's)
#         CHERI_08_MAX         08.0 runs at once on the node (default 6); the
#                              integer in CHERI_08_MAX_FILE (default
#                              ${BOX_DIR}/cheri_downstream.08_max), when that file
#                              exists, wins and is read again before every 08.0
#         CHERI_08_THREADS     threads of each 08.0 (default 24)
#         CHERI_06_THREADS, CHERI_07_THREADS  (default 8 each)
#         CHERI_SHARED_LIST    datasets ChromBPNet interpreted, for V1 against its
#                              files (default ${BOX_DIR}/attribution_datasets.txt)
#         CHERI_DOWNSTREAM_PRODUCERS  glob of the producers' hold files (above)
#         CHERI_DOWNSTREAM_POLL       seconds between looks (default 300)
#         CHERI_DOWNSTREAM_RETRY      seconds between tries for an 08.0 slot or a
#                                     fold lock (default 30)
#         CHERI_DROP_FOLD_H5   1: once 06.0, V2 and 07.0 are done, delete the
#                              converted per-fold h5 of a dataset not in the
#                              shared list (bed and provenance stay; they rebuild
#                              in minutes). Default 0: keep them.
# Output: the steps' outputs under each output_dir; logs in <output_dir>/logs/
#   (<step>.log, appended to on every run); the checks' reports in
#   <output_dir>/validation/; ${BOX_DIR}/cheri_downstream.tsv, one row per step
#   run: dataset, step, exit, seconds, max RSS (kB, /usr/bin/time %M), started,
#   note.
# Usage:  downstream.sh <list.txt> [<list.txt> ...]   (inside the box; the box's
#         cheri_downstream.sh sets the environment and holds the box)

set -uo pipefail

usage="usage: downstream.sh <list.txt> [<list.txt> ...]"
(( $# > 0 )) || { echo "${usage}" >&2; exit 2; }
: "${CHERI_ROOT:?set CHERI_ROOT to the make_jobs.py --out-root}"
: "${BOX_DIR:?set BOX_DIR to the box directory}"
: "${REPO_ROOT:?set REPO_ROOT to the igvf_tf_chrombpnet checkout whose steps run}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
here="${repo_root}/workflows/cherimoya"
steps_dir="${REPO_ROOT}/workflows/SLURM"
case ":${PATH}:" in
    *":${HOME}/.pixi/bin:"*) ;;
    *) export PATH="${HOME}/.pixi/bin:${PATH}" ;;
esac
# The box wrappers hold their one-run lock on fd 200. The steps must not inherit
# it: a step left running after a crash would lock the next run out, while the
# dataset and slot locks below already keep two runs from doing the same work.
exec 200>&-

pixi_run=( pixi run --frozen --manifest-path "${CHERI_PIXI_MANIFEST:-${repo_root}/pixi.toml}" -e cherimoya )
converter="${here}/to_chrombpnet_scores.py"
record="${BOX_DIR}/cheri_downstream.tsv"
max08="${CHERI_08_MAX:-6}"
max08_file="${CHERI_08_MAX_FILE-${BOX_DIR}/cheri_downstream.08_max}"
t08="${CHERI_08_THREADS:-24}"
t06="${CHERI_06_THREADS:-8}"
t07="${CHERI_07_THREADS:-8}"
poll="${CHERI_DOWNSTREAM_POLL:-300}"
retry="${CHERI_DOWNSTREAM_RETRY:-30}"
shared_list="${CHERI_SHARED_LIST:-${BOX_DIR}/attribution_datasets.txt}"
producers="${CHERI_DOWNSTREAM_PRODUCERS-${BOX_DIR}/.hold.cherimoya.attr.*}"
drop_fold_h5="${CHERI_DROP_FOLD_H5:-0}"
pick_sample="${CHERI_PICK_SAMPLE:-1}"
scratch="/dev/shm/${USER}/box.${SLURM_JOB_ID:-local}/cheri_downstream"

log() { printf '[%s] %s\n' "$(date)" "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

for n in "${max08}" "${t08}" "${t06}" "${t07}" "${poll}" "${retry}"; do
    [[ "${n}" =~ ^[1-9][0-9]*$ ]] || die "CHERI_08_MAX, the thread counts, the poll and the retry must be positive integers"
done
for s in 06.0.average_contrib_scores 07.0.contribs_to_bigwig 08.0.run_modisco; do
    [[ -f "${steps_dir}/${s}.sh" ]] || die "no ${steps_dir}/${s}.sh: REPO_ROOT is not an igvf_tf_chrombpnet checkout"
done
[[ -f "${shared_list}" ]] || die "no shared list ${shared_list} (CHERI_SHARED_LIST)"
[[ -x /usr/bin/time ]] || die "no /usr/bin/time, which measures each step's max RSS"
mkdir -p "${scratch}/locks" "${scratch}/slots" "${scratch}/queue" "${scratch}/cores" || die "cannot create ${scratch}"
[[ -s "${record}" ]] || printf 'dataset\tstep\texit\tseconds\tmax_rss_kb\tstarted\tnote\n' >> "${record}"

datasets=()
declare -A seen=()
for list in "$@"; do
    [[ -f "${list}" ]] || die "no dataset list ${list}"
    while IFS= read -r line || [[ -n "${line}" ]]; do
        line="${line%%#*}"
        line="${line//[[:space:]]/}"
        [[ -z "${line}" || -n "${seen[${line}]:-}" ]] && continue
        seen[${line}]=1
        datasets+=( "${line}" )
    done < "${list}"
done
(( ${#datasets[@]} > 0 )) || die "the lists name no dataset"
for ds in "${datasets[@]}"; do
    [[ -f "${CHERI_ROOT}/configs/${ds}/config.yaml" ]] || die "no ${CHERI_ROOT}/configs/${ds}/config.yaml (make_downstream.py)"
done

# ---------------------------------------------------------------------------
# one dataset: its paths (globals, set by context)

# yaml_scalar <file> <key> -- a top-level scalar, unquoted, without a trailing comment.
yaml_scalar() {
    awk -v key="$2" '
        index($0, key ":") == 1 {
            v = substr($0, length(key) + 2)
            sub(/^[ \t]+/, "", v)
            if (substr(v, 1, 1) == "\"") { v = substr(v, 2); sub(/".*/, "", v) }
            else { sub(/[ \t]+#.*/, "", v); sub(/[ \t]+$/, "", v) }
            print v
            exit
        }' "$1"
}

# yaml_list <file> <key> -- the items of a one-line flow list, one per line.
yaml_list() {
    sed -n "s/^$2: *\[\(.*\)\].*/\1/p" "$1" | head -n 1 | tr ',' '\n' | tr -d '" '
}

# context <dataset> -- set the dataset's paths; fails on a config this cannot run.
context() {
    ds="$1"
    cfg="${CHERI_ROOT}/configs/${ds}/config.yaml"
    out="$(yaml_scalar "${cfg}" output_dir)"
    pt="$(yaml_scalar "${cfg}" peak_type)"
    pt="${pt:-all}"
    mapfile -t folds < <(yaml_list "${cfg}" folds)
    mapfile -t heads < <(yaml_list "${cfg}" contrib_heads)
    [[ -n "${out}" && ${#folds[@]} -gt 0 ]] || { log "${ds}: ${cfg} has no output_dir or folds"; return 1; }
    [[ "${heads[*]}" == counts ]] || { log "${ds}: ${cfg} must list contrib_heads [\"counts\"], has ${heads[*]:-none}"; return 1; }
    head=counts
    fold_dirs=()
    provs=()
    for k in "${folds[@]}"; do
        fold_dirs+=( "${CHERI_ROOT}/${ds}/fold_${k}" )
        provs+=( "${out}/full_models/${ds}_${pt}_fold_${k}/interpretation/interpretation.provenance.json" )
    done
    avg_dir="${out}/contrib_scores/${ds}"
    avg_h5="${avg_dir}/${ds}_average_shaps.${head}.h5"
    bw="${avg_dir}/${ds}_average_shaps.${head}.bw"
    mod_h5="${avg_dir}/modisco/modisco_${head}_results.h5"
    report_dir="${avg_dir}/modisco/${head}_report"
    done_dir="${out}/.done"
    logs="${out}/logs"
    validation="${out}/validation"
    shared=0
    grep -qxF "${ds}" "${shared_list}" && shared=1
    mkdir -p "${done_dir}" "${logs}" "${validation}"
}

marked() { [[ -f "${done_dir}/$1" ]]; }
unmark() { rm -f "${done_dir}/$1"; }
# mark <step> <note> -- the step's done marker: when, and what it found.
mark() {
    local tmp="${done_dir}/.$1.tmp.${BASHPID}"
    printf '%s\t%s\n' "$(date -Iseconds)" "$2" > "${tmp}" && mv -f "${tmp}" "${done_dir}/$1"
}
# quarantine <path> -- an output without its marker is renamed, never deleted (an empty directory is left).
quarantine() {
    [[ -e "$1" ]] || return 0
    [[ -d "$1" && -z "$(ls -A "$1")" ]] && return 0
    local to
    to="$1.incomplete.$(date +%Y%m%dT%H%M%S)"
    [[ -e "${to}" ]] && to="${to}.${BASHPID}.${RANDOM}"
    mv -n "$1" "${to}"
    [[ -e "$1" ]] && { log "${ds}: cannot move ${1} aside to ${to}"; return 1; }
    log "${ds}: moved ${1} (no done marker) to ${to}"
}
step_done() {  # step_done <step> <output>: marker and output both there
    marked "$1" && [[ -e "$2" ]]
}

# record <step> <exit> <note> -- one row of the record, from run_step's last measurements.
record() {
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${ds}" "$1" "$2" "${step_seconds:-NA}" "${step_rss:-NA}" \
        "${step_started:-NA}" "$3" >> "${record}"
}

# run_step <step> <threads> <log> <cmd...> -- cmd under taskset on <threads>
# picked CPUs, with the thread counts set and no GPU, its max RSS measured by
# /usr/bin/time; sets step_seconds, step_rss and step_started; returns its exit.
run_step() {
    local step="$1" threads="$2" step_log="$3" t0 pid rc rss_file
    shift 3
    rss_file="${scratch}/rss.${BASHPID}.${RANDOM}"
    step_started="$(date -Iseconds)"
    t0="${SECONDS}"
    (
        owner="${BASHPID}"
        cpus="$(python3 "${here}/pick_cores.py" "${threads}" --reserve "${scratch}/cores" --owner "${owner}" \
                --sample "${pick_sample}")" || exit 125
        printf '==== [%s] %s %s on CPUs %s: %s\n' "$(date)" "${ds}" "${step}" "${cpus}" "$*" >> "${step_log}"
        exec env OMP_NUM_THREADS="${threads}" NUMBA_NUM_THREADS="${threads}" MKL_NUM_THREADS="${threads}" \
            OPENBLAS_NUM_THREADS="${threads}" SLURM_CPUS_PER_TASK="${threads}" CUDA_VISIBLE_DEVICES= \
            /usr/bin/time -f '%M' -o "${rss_file}" taskset -c "${cpus}" "$@" >> "${step_log}" 2>&1
    ) &
    pid=$!
    wait "${pid}"
    rc=$?
    step_seconds=$(( SECONDS - t0 ))
    step_rss="$(tail -n 1 "${rss_file}" 2>/dev/null)"
    [[ "${step_rss}" =~ ^[0-9]+$ ]] || step_rss=NA
    rm -f "${rss_file}" "${scratch}/cores/${pid}"
    return "${rc}"
}

# igvf <script> <step> <threads> -- one of the checkout's step files, as full_chain.sh runs it.
igvf() {
    run_step "$2" "$3" "${logs}/$2.log" env DATASET_CONFIG="${cfg}" SLURM_ARRAY_TASK_ID=0 \
        bash "${steps_dir}/$1.sh"
}

# ---------------------------------------------------------------------------
# convert, check, 06.0, v2 (in this script)

convert_fold() {  # convert_fold <k>: retried while another run holds the fold's lock (exit 4)
    local k="$1" tries=0 rc note
    local prov="${out}/full_models/${ds}_${pt}_fold_${k}/interpretation/interpretation.provenance.json"
    while :; do
        run_step "convert_${k}" 2 "${logs}/convert_${k}.log" "${pixi_run[@]}" python "${converter}" \
            --attribute-json "${CHERI_ROOT}/${ds}/fold_${k}/attribute.json" --config "${cfg}" --lock
        rc=$?
        if (( rc == 4 && tries < 40 )); then
            record "convert_${k}" "${rc}" "fold lock held; retried in ${retry} s"
            tries=$(( tries + 1 ))
            sleep "${retry}"
            continue
        fi
        break
    done
    if (( rc == 0 )) && [[ -f "${prov}" ]]; then
        note=converted
        tail -n 5 "${logs}/convert_${k}.log" | grep -q 'up to date' && note="up to date"
        record "convert_${k}" 0 "${note}"
        mark "convert_${k}" "${note}"
        return 0
    fi
    record "convert_${k}" "${rc}" "failed"
    return 1
}

# invalidate <step...> -- the outputs of these steps are stale: move them aside, drop their markers.
invalidate() {
    local s
    for s in "$@"; do
        case "${s}" in
            v2)   unmark v2 ;;
            07.0) unmark 07.0; quarantine "${bw}" || return 1 ;;
            08.0) unmark 08.0; quarantine "${mod_h5}" && quarantine "${report_dir}" || return 1 ;;
        esac
    done
}

# sources_current -- every fold's provenance still describes the attribution it was converted from.
sources_current() {
    python3 - "${provs[@]}" <<'EOF'
import json
import os
import sys

bad = []
for path in sys.argv[1:]:
    try:
        with open(path) as fh:
            sources = json.load(fh)["sources"]
    except (OSError, ValueError, KeyError) as exc:
        bad.append(f"{path}: {exc}")
        continue
    for role, src in sources.items():
        try:
            st = os.stat(src["path"])
        except OSError:
            bad.append(f"{path}: {role} {src['path']} is gone")
            continue
        if (st.st_size, st.st_mtime_ns) != (src["size"], src["mtime_ns"]):
            bad.append(f"{path}: {role} {src['path']} changed since the conversion")
print("\n".join(bad))
sys.exit(1 if bad else 0)
EOF
}

# head_stages -- convert, check, 06.0 and v2 of the dataset in context.
head_stages() {
    local k pids=() pid failed=0 against rc
    if ! step_done 06.0 "${avg_h5}"; then
        log "${ds}: converting folds ${folds[*]}"
        for k in "${folds[@]}"; do
            convert_fold "${k}" &
            pids+=( $! )
        done
        for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
        (( failed )) && { log "${ds}: a conversion failed (${logs}/convert_<k>.log)"; return 1; }

        against=fasta
        (( shared )) && against=chrombpnet
        log "${ds}: --check-dataset against ${against}"
        unmark check
        run_step check 2 "${logs}/check.log" "${pixi_run[@]}" python "${converter}" --check-dataset "${ds}" \
            --root "${CHERI_ROOT}" --config "${cfg}" --against "${against}" \
            --report "${validation}/check_dataset.json"
        rc=$?
        record check "${rc}" "against ${against}"
        (( rc == 0 )) || { log "${ds}: --check-dataset failed (${logs}/check.log, ${validation}/check_dataset.json)"; return 1; }
        mark check "against ${against}"

        unmark 06.0
        invalidate v2 07.0 08.0 && quarantine "${avg_h5}" || return 1
        log "${ds}: 06.0"
        igvf 06.0.average_contrib_scores 06.0 "${t06}"
        rc=$?
        if (( rc != 0 )) || [[ ! -f "${avg_h5}" ]]; then
            record 06.0 "${rc}" "failed"
            log "${ds}: 06.0 failed, exit ${rc} (${logs}/06.0.log)"
            return 1
        fi
        record 06.0 0 ""
        mark 06.0 "ok"
    elif ! marked v2; then
        log "${ds}: 06.0 done before; V2 next"
    fi
    if ! marked v2; then
        log "${ds}: V2 (--check-average)"
        run_step v2 6 "${logs}/v2.log" "${pixi_run[@]}" python "${converter}" --check-average "${ds}" \
            --root "${CHERI_ROOT}" --config "${cfg}" --threads 5 --report "${validation}/check_average.json"
        rc=$?
        record v2 "${rc}" ""
        (( rc == 0 )) || { log "${ds}: V2 failed (${logs}/v2.log, ${validation}/check_average.json)"; return 1; }
        mark v2 "ok"
    fi
    return 0
}

# ---------------------------------------------------------------------------
# 07.0 and 08.0 (in the background; the dataset's lock is inherited)

max_08() {
    local v=""
    [[ -n "${max08_file}" && -f "${max08_file}" ]] && v="$(tr -dc '0-9' < "${max08_file}")"
    [[ "${v}" =~ ^[1-9][0-9]*$ ]] || v="${max08}"
    echo "${v}"
}

# acquire_08_slot <priority> -- wait until this dataset is first in the queue and
# fewer than max_08 slots are taken, then take one: slot_fd holds it until closed.
acquire_08_slot() {
    local mine first f i busy free mx said=0
    mine="${scratch}/queue/$(printf '%06d' "$1").${BASHPID}"
    : > "${mine}"
    while :; do
        first=""
        for f in "${scratch}"/queue/*; do
            [[ -e "${f}" ]] || continue
            kill -0 "${f##*.}" 2> /dev/null || { rm -f "${f}"; continue; }
            first="${f}"
            break
        done
        if [[ "${first}" == "${mine}" ]]; then
            exec {mx}> "${scratch}/slots/.mutex"
            flock "${mx}"
            busy=0
            free=-1
            for (( i = 0; i < 64; i++ )); do
                if flock -n "${scratch}/slots/${i}.lock" true; then
                    (( free < 0 )) && free="${i}"
                else
                    busy=$(( busy + 1 ))
                fi
            done
            if (( busy < $(max_08) && free >= 0 )); then
                exec {slot_fd}> "${scratch}/slots/${free}.lock"
                if flock -n "${slot_fd}"; then
                    exec {mx}>&-
                    rm -f "${mine}"
                    log "${ds}: 08.0 slot ${free} ($(( busy + 1 )) of $(max_08) taken)"
                    return 0
                fi
                exec {slot_fd}>&-
            fi
            exec {mx}>&-
        fi
        (( said )) || { log "${ds}: 08.0 waits for a slot ($(max_08) at once)"; said=1; }
        sleep "${retry}"
    done
}

# modisco_summary <h5> -- "<pos_patterns> pos patterns, window_size <w>"; fails unless it opens with pos_patterns.
modisco_summary() {
    "${pixi_run[@]}" python -c '
import sys

import h5py

with h5py.File(sys.argv[1], "r") as fh:
    if "pos_patterns" not in fh:
        sys.exit(1)
    print(len(fh["pos_patterns"]), "pos patterns, window_size", fh.attrs.get("window_size", "NA"))
' "$1" 2> /dev/null
}

do_07() {
    local rc
    unmark 07.0
    quarantine "${bw}" || return 1
    log "${ds}: 07.0"
    igvf 07.0.contribs_to_bigwig 07.0 "${t07}"
    rc=$?
    if (( rc != 0 )) || [[ ! -f "${bw}" ]]; then
        record 07.0 "${rc}" "failed"
        log "${ds}: 07.0 failed, exit ${rc} (${logs}/07.0.log)"
        return 1
    fi
    record 07.0 0 ""
    mark 07.0 "ok"
    log "${ds}: 07.0 done (${step_seconds} s)"
}

do_08() {
    local rc summary note
    acquire_08_slot "$1"
    while pgrep -f -- "modisco motifs -i ${avg_h5}" > /dev/null; do
        log "${ds}: a modisco motifs on ${avg_h5} is running outside this run; 08.0 waits for it"
        sleep "${poll}"
    done
    if step_done 08.0 "${mod_h5}"; then  # that run finished it
        exec {slot_fd}>&-
        return 0
    fi
    unmark 08.0
    if ! { quarantine "${mod_h5}" && quarantine "${report_dir}"; }; then
        exec {slot_fd}>&-
        return 1
    fi
    log "${ds}: 08.0 (${t08} threads)"
    igvf 08.0.run_modisco 08.0 "${t08}"
    rc=$?
    exec {slot_fd}>&-
    summary="$(modisco_summary "${mod_h5}")"
    if [[ -f "${mod_h5}" && -n "${summary}" ]]; then
        note="${summary}"
        (( rc == 0 )) || note="report_failed: ${summary}"
        record 08.0 "${rc}" "${note}"
        mark 08.0 "${note}"
        log "${ds}: 08.0 done, exit ${rc} (${step_seconds} s): ${note}"
        return 0
    fi
    record 08.0 "${rc}" "failed"
    log "${ds}: 08.0 failed, exit ${rc} (${logs}/08.0.log)"
    return 1
}

# tail_stages <priority> -- 07.0 beside 08.0, then the optional drop of the per-fold h5.
tail_stages() {
    local p07="" rc07=0 rc08=0 k
    trap - TERM INT HUP
    if ! step_done 07.0 "${bw}"; then
        do_07 &
        p07=$!
    fi
    if ! step_done 08.0 "${mod_h5}"; then
        do_08 "$1"
        rc08=$?
    fi
    if [[ -n "${p07}" ]]; then
        wait "${p07}"
        rc07=$?
    fi
    if (( drop_fold_h5 == 1 && ! shared && rc07 == 0 )) && step_done 06.0 "${avg_h5}" && marked v2 \
        && step_done 07.0 "${bw}" && ! marked fold_h5_dropped; then
        for k in "${folds[@]}"; do
            rm -f "${out}/full_models/${ds}_${pt}_fold_${k}/interpretation/interpretation.${head}_scores.h5"
        done
        mark fold_h5_dropped "per-fold ${head} h5 deleted after 06.0, V2 and 07.0"
        log "${ds}: deleted the converted per-fold h5 (CHERI_DROP_FOLD_H5=1)"
    fi
    (( rc07 == 0 && rc08 == 0 ))
}

complete() {
    step_done 06.0 "${avg_h5}" && marked v2 && step_done 07.0 "${bw}" && step_done 08.0 "${mod_h5}"
}

# ---------------------------------------------------------------------------
# the loop

kill_tree() {  # kill_tree <pid>: its descendants, then itself
    local c
    for c in $(pgrep -P "$1"); do kill_tree "${c}"; done
    kill -TERM "$1" 2> /dev/null
}
on_signal() {
    trap - TERM INT HUP
    log "signal: stopping every step this run started (no markers are written for them)"
    local c
    for c in $(pgrep -P $$); do kill_tree "${c}"; done
    wait
    exit 143
}
trap on_signal TERM INT HUP

# attr_states -- "<done> <total> <summary>" of the dataset's folds by attr_status.py (no lock probes).
attr_states() {
    python3 "${here}/attr_status.py" --no-locks "${fold_dirs[@]}" 2> /dev/null \
        | awk -F'\t' -v n="${#fold_dirs[@]}" '{c[$3]++} $3 == "done" {d++}
            END {s = ""; for (k in c) s = s (s ? ", " : "") c[k] " " k; print d + 0, n, (s ? s : "unreadable")}'
}

producer_alive() {
    [[ -n "${producers}" ]] && compgen -G "${producers}" > /dev/null
}

log "downstream.sh on $(hostname), job ${SLURM_JOB_ID:-local} step ${SLURM_STEP_ID:-?}: ${#datasets[@]} dataset(s)" \
    "from $*; igvf steps from ${REPO_ROOT}; 08.0 x $(max_08) at ${t08} threads; shared list ${shared_list};" \
    "producers ${producers:-none}; poll ${poll} s; CPUs $(awk '/^Cpus_allowed_list:/{print $2}' /proc/self/status)"

declare -A tail_pid=() given_up=() finished=() said=()
while :; do
    open=0
    advanced=0
    for i in "${!datasets[@]}"; do
        ds="${datasets[${i}]}"
        [[ -n "${finished[${ds}]:-}" || -n "${given_up[${ds}]:-}" ]] && continue
        context "${ds}" || { given_up[${ds}]="its config"; continue; }
        if [[ -n "${tail_pid[${ds}]:-}" ]]; then
            if kill -0 "${tail_pid[${ds}]}" 2> /dev/null; then
                open=1
                continue
            fi
            wait "${tail_pid[${ds}]}"
            rc=$?
            unset "tail_pid[${ds}]"
            if (( rc != 0 )); then
                given_up[${ds}]="07.0 or 08.0 failed"
                log "${ds}: given up: 07.0 or 08.0 failed (${logs}/)"
                continue
            fi
        fi
        if complete; then
            finished[${ds}]=1
            log "${ds}: done ($(cut -f 2 "${done_dir}/08.0"))"
            continue
        fi
        exec {lock}> "${scratch}/locks/${ds}.lock"
        if ! flock -n "${lock}"; then
            exec {lock}>&-
            [[ "${said[${ds}]:-}" == busy ]] || log "${ds}: another run is working on it; looked at again later"
            said[${ds}]=busy
            open=1
            continue
        fi
        if step_done 06.0 "${avg_h5}" && marked v2; then
            if ! problems="$(sources_current)"; then
                exec {lock}>&-
                given_up[${ds}]="re-attributed after its conversion"
                log "${ds}: given up: a fold was attributed again after the conversion 06.0 averaged:" \
                    "${problems//$'\n'/; }. Move ${avg_dir} and ${done_dir} aside to redo it."
                continue
            fi
        else
            read -r n_done n_folds states <<< "$(attr_states)"
            if (( n_done != n_folds )); then
                exec {lock}>&-
                if producer_alive; then
                    [[ "${said[${ds}]:-}" == "${states}" ]] || log "${ds}: waiting for its attribution (${states})"
                    said[${ds}]="${states}"
                    open=1
                else
                    given_up[${ds}]="attribution not done (${states})"
                    log "${ds}: given up: its attribution is not done (${states}) and no producer holds" \
                        "${producers:-(none named)}"
                fi
                continue
            fi
            if ! head_stages; then
                exec {lock}>&-
                given_up[${ds}]="convert, check, 06.0 or V2 failed"
                log "${ds}: given up (see ${logs}/ and ${validation}/)"
                continue
            fi
            advanced=1
        fi
        tail_stages "${i}" &
        tail_pid[${ds}]=$!
        exec {lock}>&-
        open=1
        (( advanced )) && break  # a head took minutes: look at the list again from the top
    done
    (( open )) || break
    (( advanced )) && continue
    sleep "${poll}" &
    wait $!
done

log "downstream.sh: ${#finished[@]} of ${#datasets[@]} dataset(s) done, ${#given_up[@]} given up"
for ds in "${!given_up[@]}"; do log "  given up: ${ds}: ${given_up[${ds}]}"; done
(( ${#given_up[@]} == 0 ))
