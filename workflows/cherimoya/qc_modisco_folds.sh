#!/bin/bash
# qc_modisco_folds.sh
# Purpose: TF-MoDISco on each Cherimoya fold at the scale of igvf_tf_chrombpnet's per-fold motif QC (04.5), on the
#   peaks ChromBPNet's own QC of the same fold used, so the two models' per-fold motifs compare like for like (and
#   each model's fold-to-fold agreement gives the ceiling).
#
# For each dataset of the lists and each fold of its Cherimoya config (make_downstream.py), in that order (a fold
#   whose attribution is not done does not hold back the others), once attr_status.py calls the fold DONE:
#   subset   subset_scores.py (cherimoya env): the fold's attribution on the peaks of ChromBPNet's QC subsample of
#            the same fold (auxiliary/30K_subsample_peaks.bed of the ChromBPNet fold its compare.json names), in
#            that order, as <output_dir>/full_models/<dataset>_<peak_type>_fold_<k>/auxiliary/interpret_subsample/
#            cherimoya.counts_scores.h5 and its bed, checked against ChromBPNet's QC scores of the fold
#   modisco  the igvf checkout's src/motif_qc.py on it, as 04.5 runs it (04.5:85-106): the dataset's config
#            sourced through lib/bash/config.sh (the motif DB; motif_qc_window and motif_qc_max_seqlets, or
#            MOTIF_QC_WINDOW and MOTIF_QC_MAX_SEQLETS, default 400 and 5000), the chrombpnet env activated
#            (CHROMBPNET_REPO, CHROMBPNET_PIXI_ENV), --prefix cherimoya --threads CHERI_QC_THREADS, into
#            .../evaluation/motif_qc_counts/
# At most CHERI_QC_MAX folds run at once. Each step is pinned to CHERI_QC_THREADS of the least busy CPUs this script
#   may use (pick_cores.py, whose reservations are shared with downstream.sh's: CHERI_QC_CORES_DIR), with the thread
#   counts set, no GPU, and nice CHERI_QC_NICE: this is the lowest-priority work on the node.
#
# Done: <qc dir>/cherimoya_qc.done.json, written once motif_qc.py exits 0 with its results h5, report, MEME file and
#   summary; it records the scores h5 the run read (size, mtime), the pattern counts, and whether the settings
#   (window, seqlets, motif DB, modisco flags) are those of ChromBPNet's QC of the same fold. A fold is done when the
#   marker names the current scores h5, the h5's provenance still matches its sources (a fold attributed again since
#   makes it stale) and the outputs are there. motif_qc.py's own skip rule compares settings, not scores, so a QC
#   directory that is not done is renamed <dir>.incomplete.<time> (nothing is deleted) and the fold runs again from
#   its subset (which subset_scores.py leaves alone when it is up to date).
# Waiting: a fold that is not DONE waits while a producer's hold file exists (CHERI_QC_PRODUCERS, default the
#   attribution runners' ${BOX_DIR}/.hold.cherimoya.attr.*), looked at every CHERI_QC_POLL seconds; with no producer
#   left it is given up, and so is a fold whose subset or MoDISco fails. This exits once every fold is done or given
#   up: 0 if all are done, 1 otherwise. A fold another run is working on (its node-local lock in
#   /dev/shm/$USER/box.<job>/cheri_qc/locks/) is looked at again later.
# SIGTERM, SIGINT or SIGHUP stops every step this run started and exits 143; no marker is written for them.
#
# Input:  dataset lists, one dataset per line (blank lines and # comments skipped); with none, CHERI_SHARED_LIST
#   (default ${BOX_DIR}/attribution_datasets.txt: the datasets ChromBPNet interpreted, whose QC subsamples exist).
#   Environment: CHERI_ROOT (make_jobs.py's --out-root, with the configs make_downstream.py wrote), BOX_DIR (the
#   record, the producers' holds), REPO_ROOT (the igvf_tf_chrombpnet checkout whose src/motif_qc.py and
#   lib/bash/config.sh run, with what they need: workflows/dcai/env.sh, CHROMBPNET_REPO, CHROMBPNET_PIXI_ENV).
# Knobs:  CHERI_PIXI_MANIFEST  pixi.toml whose cherimoya env runs subset_scores.py (default: this checkout's)
#         CHERI_QC_MAX         folds at once (default 4)
#         CHERI_QC_THREADS     threads and CPUs of each step (default 16)
#         CHERI_QC_NICE        niceness of each step (default 19)
#         CHERI_QC_PRODUCERS   glob of the producers' hold files (above); empty: none
#         CHERI_QC_POLL        seconds between looks at folds waiting for their attribution (default 300)
#         CHERI_QC_RETRY       seconds between looks while folds run (default 30)
#         CHERI_QC_CORES_DIR   pick_cores.py's reservations (default downstream.sh's,
#                              /dev/shm/$USER/box.<job>/cheri_downstream/cores)
# Output: the files above; logs in <output_dir>/logs/qc_<k>.log (appended to); ${BOX_DIR}/cheri_qc.tsv, one row per
#   step run: dataset, step (subset_<k>, modisco_<k>), exit, seconds, max RSS (kB, /usr/bin/time %M), started, note.
# Usage:  qc_modisco_folds.sh [<list.txt> ...]   (inside the box; the box's cheri_qc.sh folds sets the environment
#         and holds the box)

set -uo pipefail

usage="usage: qc_modisco_folds.sh [<list.txt> ...]"
: "${CHERI_ROOT:?set CHERI_ROOT to the make_jobs.py --out-root}"
: "${BOX_DIR:?set BOX_DIR to the box directory}"
: "${REPO_ROOT:?set REPO_ROOT to the igvf_tf_chrombpnet checkout whose src/motif_qc.py runs}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
here="${repo_root}/workflows/cherimoya"
case ":${PATH}:" in
    *":${HOME}/.pixi/bin:"*) ;;
    *) export PATH="${HOME}/.pixi/bin:${PATH}" ;;
esac
# The box wrapper holds its one-run lock on fd 200; the steps must not inherit it (downstream.sh does the same).
exec 200>&-

pixi_run=( pixi run --frozen --manifest-path "${CHERI_PIXI_MANIFEST:-${repo_root}/pixi.toml}" -e cherimoya )
max_runs="${CHERI_QC_MAX:-4}"
threads="${CHERI_QC_THREADS:-16}"
niceness="${CHERI_QC_NICE:-19}"
poll="${CHERI_QC_POLL:-300}"
retry="${CHERI_QC_RETRY:-30}"
producers="${CHERI_QC_PRODUCERS-${BOX_DIR}/.hold.cherimoya.attr.*}"
pick_sample="${CHERI_PICK_SAMPLE:-1}"
scratch="/dev/shm/${USER}/box.${SLURM_JOB_ID:-local}/cheri_qc"
cores_dir="${CHERI_QC_CORES_DIR:-/dev/shm/${USER}/box.${SLURM_JOB_ID:-local}/cheri_downstream/cores}"
record_file="${BOX_DIR}/cheri_qc.tsv"
head=counts

log() { printf '[%s] %s\n' "$(date)" "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

for n in "${max_runs}" "${threads}" "${poll}" "${retry}"; do
    [[ "${n}" =~ ^[1-9][0-9]*$ ]] || die "CHERI_QC_MAX, CHERI_QC_THREADS, the poll and the retry must be positive integers"
done
[[ "${niceness}" =~ ^[0-9]+$ ]] || die "CHERI_QC_NICE must be a whole number"
for f in src/motif_qc.py lib/bash/config.sh; do
    [[ -f "${REPO_ROOT}/${f}" ]] || die "no ${REPO_ROOT}/${f}: REPO_ROOT is not an igvf_tf_chrombpnet checkout"
done
[[ -x /usr/bin/time ]] || die "no /usr/bin/time, which measures each step's max RSS"
if (( $# == 0 )); then
    shared_list="${CHERI_SHARED_LIST:-${BOX_DIR}/attribution_datasets.txt}"
    [[ -f "${shared_list}" ]] || die "no list given and no ${shared_list} (CHERI_SHARED_LIST). ${usage}"
    set -- "${shared_list}"
fi
mkdir -p "${scratch}/locks" "${cores_dir}" || die "cannot create ${scratch} or ${cores_dir}"
[[ -s "${record_file}" ]] || printf 'dataset\tstep\texit\tseconds\tmax_rss_kb\tstarted\tnote\n' >> "${record_file}"

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

# ---------------------------------------------------------------------------------------------------------------------
# a dataset's config, and one fold's paths (globals, set by context)

# yaml_scalar <file> <key> -- a top-level scalar, unquoted, without a trailing comment (as downstream.sh reads it).
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

tasks=()
declare -A task_cfg=() task_out=() task_pt=()
for ds in "${datasets[@]}"; do
    cfg="${CHERI_ROOT}/configs/${ds}/config.yaml"
    [[ -f "${cfg}" ]] || die "no ${cfg} (make_downstream.py)"
    out="$(yaml_scalar "${cfg}" output_dir)"
    pt="$(yaml_scalar "${cfg}" peak_type)"
    mapfile -t folds < <(yaml_list "${cfg}" folds)
    mapfile -t heads < <(yaml_list "${cfg}" contrib_heads)
    [[ -n "${out}" && ${#folds[@]} -gt 0 ]] || die "${cfg} has no output_dir or folds"
    [[ "${heads[*]}" == "${head}" ]] || die "${cfg} must list contrib_heads [\"${head}\"], has ${heads[*]:-none}"
    for k in "${folds[@]}"; do
        tasks+=( "${ds}/${k}" )
        task_cfg[${ds}/${k}]="${cfg}"
        task_out[${ds}/${k}]="${out}"
        task_pt[${ds}/${k}]="${pt:-all}"
    done
done

# context <ds/k> -- the fold's paths.
context() {
    task="$1"
    ds="${task%/*}"
    k="${task##*/}"
    cfg="${task_cfg[${task}]}"
    out="${task_out[${task}]}"
    fold_dir="${CHERI_ROOT}/${ds}/fold_${k}"
    model_dir="${out}/full_models/${ds}_${task_pt[${task}]}_fold_${k}"
    sub_dir="${model_dir}/auxiliary/interpret_subsample"
    h5="${sub_dir}/cherimoya.${head}_scores.h5"
    prov="${sub_dir}/cherimoya.provenance.json"
    qc_dir="${model_dir}/evaluation/motif_qc_${head}"
    marker="${qc_dir}/cherimoya_qc.done.json"
    logs="${out}/logs"
    step_log="${logs}/qc_${k}.log"
}

# record <step> <exit> <note> -- one row of the record, from run_step's last measurements.
record() {
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${ds}" "$1" "$2" "${step_seconds:-NA}" "${step_rss:-NA}" \
        "${step_started:-NA}" "$3" >> "${record_file}"
}

# run_step <step> <cmd...> -- cmd under nice and taskset on ${threads} picked CPUs, with the thread counts set and no
# GPU, its max RSS measured by /usr/bin/time, its output appended to the fold's log; sets step_seconds, step_rss and
# step_started; returns its exit.
run_step() {
    local step="$1" t0 pid rc rss_file
    shift
    rss_file="${scratch}/rss.${BASHPID}.${RANDOM}"
    step_started="$(date -Iseconds)"
    t0="${SECONDS}"
    (
        owner="${BASHPID}"
        cpus="$(python3 "${here}/pick_cores.py" "${threads}" --reserve "${cores_dir}" --owner "${owner}" \
                --sample "${pick_sample}")" || exit 125
        printf '==== [%s] %s fold %s %s on CPUs %s: %s\n' "$(date)" "${ds}" "${k}" "${step}" "${cpus}" \
            "${step_desc:-$*}" >> "${step_log}"
        exec env OMP_NUM_THREADS="${threads}" NUMBA_NUM_THREADS="${threads}" MKL_NUM_THREADS="${threads}" \
            OPENBLAS_NUM_THREADS="${threads}" SLURM_CPUS_PER_TASK="${threads}" CUDA_VISIBLE_DEVICES= \
            /usr/bin/time -f '%M' -o "${rss_file}" nice -n "${niceness}" taskset -c "${cpus}" "$@" >> "${step_log}" 2>&1
    ) &
    pid=$!
    wait "${pid}"
    rc=$?
    step_seconds=$(( SECONDS - t0 ))
    step_rss="$(tail -n 1 "${rss_file}" 2>/dev/null)"
    [[ "${step_rss}" =~ ^[0-9]+$ ]] || step_rss=NA
    rm -f "${rss_file}" "${cores_dir}/${pid}"
    return "${rc}"
}

# The MoDISco step, as 04.5:85-106 runs src/motif_qc.py, with this fold's scores and directory ($1, $2). No `set -u`
# here: config.sh reads variables it does not guard.
# shellcheck disable=SC2016  # expanded by the inner bash
qc_inner='source "${REPO_ROOT}/lib/bash/config.sh" || exit 1
window="${MOTIF_QC_WINDOW:-${motif_qc_window:-400}}"
max_seqlets="${MOTIF_QC_MAX_SEQLETS:-${motif_qc_max_seqlets:-5000}}"
activate_env "${chrombpnet_env}"
echo "[$(date)] motif_qc.py: window ${window}, ${max_seqlets} seqlets, motif DB ${chrombpnet_motifs_meme}"
exec python "${src_dir}/motif_qc.py" --scores "$1" --motif-db "${chrombpnet_motifs_meme}" --output-dir "$2" \
    --prefix cherimoya --window "${window}" --max-seqlets "${max_seqlets}" --threads "${SLURM_CPUS_PER_TASK:-4}"'

# qc_done -- exit 0 when the fold's QC is done (see the header); otherwise print why not.
qc_done() {
    python3 - "${prov}" "${h5}" "${qc_dir}" "${marker}" <<'EOF'
import json
import os
import sys
from pathlib import Path

prov_path, h5, qc_dir, marker = (Path(a) for a in sys.argv[1:5])
OUTPUTS = ("cherimoya_modisco_results.h5", "cherimoya_modisco_motifs.meme", "cherimoya_motif_qc.json",
           "report/report.html")


def stat(path):
    st = os.stat(path)
    return {"path": str(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def why():
    try:
        mark = json.loads(marker.read_text())
    except FileNotFoundError:
        return "no done marker"
    except (OSError, ValueError) as exc:
        return f"unreadable {marker}: {exc}"
    try:
        prov = json.loads(prov_path.read_text())
        scores = stat(h5)
    except (OSError, ValueError) as exc:
        return f"no subset to check the marker against ({exc})"
    if mark.get("scores") != scores:
        return f"{h5} changed since the QC read it"
    written = (prov.get("outputs") or {}).get("h5") or {}
    if {key: written.get(key) for key in scores} != scores:
        return f"{prov_path} does not describe {h5}"
    for role, src in (prov.get("sources") or {}).items():
        try:
            if stat(Path(src["path"])) != src:
                return f"its {role} changed since the subset was written"
        except (OSError, KeyError, TypeError):
            return f"its {role} is gone"
    for name in OUTPUTS:
        if not (qc_dir / name).is_file():
            return f"no {qc_dir / name}"
    return None


reason = why()
if reason:
    print(reason)
    sys.exit(1)
EOF
}

# write_marker -- the fold's done marker; prints its note.
write_marker() {
    python3 - "${prov}" "${h5}" "${qc_dir}" "${marker}" <<'EOF'
import json
import os
import sys
import time
from pathlib import Path

prov_path, h5, qc_dir, marker = (Path(a) for a in sys.argv[1:5])
SETTINGS = ("window", "max_seqlets", "motif_db", "modisco_flags")
summary = json.loads((qc_dir / "cherimoya_motif_qc.json").read_text())
st = os.stat(h5)
prov = json.loads(prov_path.read_text())
# ChromBPNet's QC of the same fold: evaluation/ beside the auxiliary/ directory that holds the subsample.
subsample = Path(prov["sources"]["subsample"]["path"])
theirs_path = subsample.parent.parent / "evaluation" / f"motif_qc_{prov['head']}" / "chrombpnet_nobias_motif_qc.json"
try:
    theirs = json.loads(theirs_path.read_text())
    differ = [key for key in SETTINGS if theirs.get(key) != summary.get(key)]
    chrombpnet = {"motif_qc_json": str(theirs_path), "settings_differ": differ,
                  "pos_patterns": theirs.get("pos_patterns"), "neg_patterns": theirs.get("neg_patterns")}
    same = "settings as ChromBPNet's QC" if not differ else f"settings differ from ChromBPNet's QC: {', '.join(differ)}"
except (OSError, ValueError) as exc:
    chrombpnet = {"motif_qc_json": str(theirs_path), "unreadable": str(exc)}
    same = "no ChromBPNet QC summary to compare settings with"
note = (f"{summary.get('pos_patterns')} pos, {summary.get('neg_patterns')} neg patterns from "
        f"{prov['rows']['kept']} peaks; {same}")
mark = {
    "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "scores": {"path": str(h5), "size": st.st_size, "mtime_ns": st.st_mtime_ns},
    "subset_provenance": str(prov_path),
    "summary": summary,
    "chrombpnet": chrombpnet,
    "note": note,
}
tmp = marker.with_name(marker.name + ".partial")
tmp.write_text(json.dumps(mark, indent=2) + "\n")
os.replace(tmp, marker)
print(note)
EOF
}

# quarantine <path> -- a QC directory that is not done is renamed, never deleted (an empty one is left).
quarantine() {
    [[ -e "$1" ]] || return 0
    [[ -d "$1" && -z "$(ls -A "$1")" ]] && return 0
    local to
    to="$1.incomplete.$(date +%Y%m%dT%H%M%S)"
    [[ -e "${to}" ]] && to="${to}.${BASHPID}.${RANDOM}"
    mv -n "$1" "${to}"
    [[ -e "$1" ]] && { log "${ds} fold ${k}: cannot move ${1} aside to ${to}"; return 1; }
    log "${ds} fold ${k}: moved ${1} (not done) to ${to}"
}

# run_fold -- the fold's QC from subset to marker, in the background. Exit 0 done, 3 not DONE after all (wait), 75
# another run holds the fold, 1 failed.
run_fold() {
    local rc note why_not lock
    trap - TERM INT HUP
    exec {lock}> "${scratch}/locks/${ds}.fold_${k}.lock"
    flock -n "${lock}" || return 75
    qc_done > /dev/null && return 0
    quarantine "${qc_dir}" || return 1
    mkdir -p "${logs}"
    step_desc=""
    run_step "subset_${k}" "${pixi_run[@]}" python "${here}/subset_scores.py" \
        --attribute-json "${fold_dir}/attribute.json" --config "${cfg}"
    rc=$?
    if (( rc == 3 )); then
        record "subset_${k}" 3 "fold not DONE; waits"
        return 3
    fi
    if (( rc != 0 )) || [[ ! -f "${h5}" || ! -f "${prov}" ]]; then
        record "subset_${k}" "${rc}" "failed"
        log "${ds} fold ${k}: subset_scores.py failed, exit ${rc} (${step_log})"
        return 1
    fi
    note=written
    tail -n 5 "${step_log}" | grep -q 'up to date' && note="up to date"
    record "subset_${k}" 0 "${note}"
    log "${ds} fold ${k}: MoDISco (${threads} threads)"
    step_desc="motif_qc.py (04.5's call) --scores ${h5} --output-dir ${qc_dir} --prefix cherimoya"
    run_step "modisco_${k}" env DATASET_CONFIG="${cfg}" bash -c "${qc_inner}" motif_qc "${h5}" "${qc_dir}"
    rc=$?
    if (( rc != 0 )); then
        record "modisco_${k}" "${rc}" "failed"
        log "${ds} fold ${k}: motif_qc.py failed, exit ${rc} (${step_log})"
        return 1
    fi
    if ! note="$(write_marker 2>> "${step_log}")"; then
        record "modisco_${k}" 0 "no done marker: the outputs could not be read"
        log "${ds} fold ${k}: motif_qc.py exited 0 but its outputs could not be read (${step_log})"
        return 1
    fi
    if ! why_not="$(qc_done)"; then
        record "modisco_${k}" 0 "not done: ${why_not}"
        log "${ds} fold ${k}: motif_qc.py exited 0 but the fold is not done: ${why_not}"
        return 1
    fi
    record "modisco_${k}" 0 "${note}"
    log "${ds} fold ${k}: done in $(( SECONDS - fold_t0 )) s: ${note}"
    return 0
}

# ---------------------------------------------------------------------------------------------------------------------
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

# attr_state -- the fold's state by attr_status.py (no lock probe): done, running, waiting, stale, failed, missing.
attr_state() {
    python3 "${here}/attr_status.py" --no-locks "${fold_dir}" 2> /dev/null | awk -F'\t' 'NR == 1 {print $3}'
}

producer_alive() {
    [[ -n "${producers}" ]] && compgen -G "${producers}" > /dev/null
}

log "qc_modisco_folds.sh on $(hostname), job ${SLURM_JOB_ID:-local} step ${SLURM_STEP_ID:-?}: ${#tasks[@]} fold(s) of" \
    "${#datasets[@]} dataset(s) from $*; ${max_runs} at once, ${threads} threads, nice ${niceness}; motif_qc.py from" \
    "${REPO_ROOT}; producers ${producers:-none}; poll ${poll} s; CPUs $(awk '/^Cpus_allowed_list:/{print $2}' /proc/self/status)"

declare -A state=() next_look=() pid_of=() why=() said=()
while :; do
    for task in "${!pid_of[@]}"; do
        kill -0 "${pid_of[${task}]}" 2> /dev/null && continue
        wait "${pid_of[${task}]}"
        rc=$?
        unset "pid_of[${task}]"
        case "${rc}" in
            0)  state[${task}]="done" ;;
            3)  state[${task}]=waiting; next_look[${task}]=$(( SECONDS + poll )) ;;
            75) state[${task}]=busy; next_look[${task}]=$(( SECONDS + retry ))
                log "${task%/*} fold ${task##*/}: another run is working on it; looked at again later" ;;
            *)  state[${task}]=gone; why[${task}]="subset or MoDISco failed (exit ${rc})" ;;
        esac
    done
    open=0
    for task in "${tasks[@]}"; do
        case "${state[${task}]:-new}" in done|gone) continue ;; esac
        open=1
        [[ -n "${pid_of[${task}]:-}" ]] && continue
        (( SECONDS >= ${next_look[${task}]:-0} )) || continue
        context "${task}"
        if [[ "${state[${task}]:-new}" == new ]]; then
            if qc_done > /dev/null; then
                state[${task}]="done"
                log "${ds} fold ${k}: done before ($(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["note"])' "${marker}" 2> /dev/null))"
                continue
            fi
            state[${task}]=todo
        fi
        (( ${#pid_of[@]} < max_runs )) || continue
        st="$(attr_state)"
        if [[ "${st}" != "done" ]]; then
            if producer_alive; then
                [[ "${said[${task}]:-}" == "${st}" ]] || log "${ds} fold ${k}: waiting for its attribution (${st:-unreadable})"
                said[${task}]="${st}"
                state[${task}]=waiting
                next_look[${task}]=$(( SECONDS + poll ))
            else
                state[${task}]=gone
                why[${task}]="attribution not done (${st:-unreadable}) and no producer holds ${producers:-(none named)}"
                log "${ds} fold ${k}: given up: ${why[${task}]}"
            fi
            continue
        fi
        log "${ds} fold ${k}: subset and MoDISco"
        fold_t0="${SECONDS}"
        run_fold &
        pid_of[${task}]=$!
        state[${task}]=running
    done
    (( open )) || break
    nap="${poll}"
    (( ${#pid_of[@]} > 0 )) && nap="${retry}"
    for task in "${tasks[@]}"; do  # a fold that may start sooner than the poll
        case "${state[${task}]:-new}" in todo|busy) nap="${retry}"; break ;; esac
    done
    sleep "${nap}" &
    wait $!
done

n_done=0
for task in "${tasks[@]}"; do [[ "${state[${task}]}" == "done" ]] && n_done=$(( n_done + 1 )); done
log "qc_modisco_folds.sh: ${n_done} of ${#tasks[@]} fold(s) done, ${#why[@]} given up"
for task in "${tasks[@]}"; do
    [[ -n "${why[${task}]:-}" ]] && log "  given up: ${task%/*} fold ${task##*/}: ${why[${task}]}"
done
(( ${#why[@]} == 0 ))
