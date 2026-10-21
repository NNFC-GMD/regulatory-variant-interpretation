#!/bin/bash
# compare.sh
# Purpose: compare ChromBPNet with Cherimoya on the datasets both were interpreted on (the downstream plan's section
#   11 and its "Comparison plan"), one level at a time, each level a script in this directory run in the cherimoya
#   environment:
#   1  performance    compare_performance.py: summarize.py's paired table of every fold's compare.tsv, with
#                     bootstrap CIs and Wilcoxon tests, residuals by observed-count decile and the plots
#                     -> <COMPARISON_ROOT>/performance/
#   2  attributions   compare_attributions.py, per dataset: per-peak agreement of the two models' attributions,
#                     joined by peak name, per fold, on the fold averages, against each model's leave-one-out
#                     ceiling and on held-out folds; then the float16 sensitivity on one dataset and the summary
#                     -> <COMPARISON_ROOT>/attributions/
#   3  TF-MoDISco     compare_modisco.py, per dataset two units: <ds> the fold-average patterns and seqlets
#                     (matched through the joint compendium's clusters and tomtom-lite), <ds>.qc the per-fold
#                     QC-scale runs with each model's fold-to-fold ceiling; then the bias control and the summary
#                     -> <COMPARISON_ROOT>/modisco/
#   4  compendia      compare_compendia.py: the like-for-like compendia (ChromBPNet's 10 against Cherimoya's 10) and
#                     the joint compendium's clusters by model -> <COMPARISON_ROOT>/compendium/
#   5  Fi-NeMo        compare_finemo.py, per dataset: both models' hits on the joint compendium, on the peaks both
#                     beds hold (and, with COMPARE_PRODUCTION=1, <ds>.production: the production hits, cbp10 against
#                     cheri39, through level 4's cluster matches); then the summary -> <COMPARISON_ROOT>/finemo/
#   The levels run in ascending order. Within a level the dataset units run COMPARE_WORKERS at a time, then the
#   level's summary.
#
# Units never wait for their inputs (the box wrapper holds the box while this runs): a unit whose inputs are not
#   there yet exits 3 and says what it lacks, the others go on, and running this again later picks up what has
#   appeared since. Each unit keeps a done marker (<level dir>/.done/<unit>.json) of its inputs, settings and
#   outputs, so a unit already done is left alone (--force runs it again).
# SIGTERM, SIGINT or SIGHUP stops every unit this run started and exits 143; a stopped unit writes no marker.
#
# Input:  <levels>: comma-separated digits 1-5. Environment: CHERI_ROOT, CHROMBPNET_ROOT, COMPENDIA_ROOT,
#   COMPARISON_ROOT, BOX_DIR (the record) and REPO_ROOT (the igvf_tf_chrombpnet checkout whose folds/ gives each
#   fold's test chromosomes); the box's compare.sh wrapper sets them all.
# Options: --datasets <list.txt>   the datasets (default ${COMPENDIA_ROOT}/inputs/shared10.txt)
#          --only <ds>[,<ds>...]   run only these datasets' units (the summaries still read the whole list)
#          --force                 run units whose done marker is current
#          --workers <n>           dataset units at once (default COMPARE_WORKERS, 10)
# Knobs:  CHERI_PIXI_MANIFEST     pixi.toml whose cherimoya env runs the scripts (default: this checkout's)
#         COMPARE_WORKERS         dataset units at once (default 10)
#         COMPARE_THREADS         threads per unit (OMP/MKL/OPENBLAS/NUMBA; default 2)
#         COMPARE_NICE            niceness of every unit (default 10)
#         COMPARE_BLOCK_ROWS      level 2: peaks per block (default 4096; RAM grows with it)
#         COMPARE_WINDOWS         level 2: half-widths around the summit (default 250,500)
#         COMPARE_FLOAT16_DATASET level 2: the dataset of the float16 sensitivity check (default the list's first;
#                                 "none" skips it)
#         COMPARE_JOINT           levels 3-5: the joint compendium's tag (default joint20)
#         COMPARE_MOTIF_DB        levels 3-4: the MEME database patterns are annotated with (default: the one a
#                                 compendium's compendium.json records, MotifCompendium's)
#         COMPARE_BIAS_DB         levels 3-4: the MEME database whose TN5_/DNASE_ entries flag bias (default: the one
#                                 ChromBPNet's per-fold QC used, chrombpnet's motifs.meme)
#         COMPARE_BIAS_MODISCO    level 3: bias references' TF-MoDISco h5 files, comma-separated (e.g. a bias model's
#                                 QC run), besides <COMPARISON_ROOT>/bias_control/*/
#         COMPARE_PRODUCTION      level 5: 1 adds the production-hits units (default 0: they wait for cheri39)
# Output: the level directories above; logs in <COMPARISON_ROOT>/logs/level<l>/<unit>.log (appended to);
#   ${BOX_DIR}/compare_steps.tsv, one row per unit run: level, unit, exit, seconds, max RSS (kB, /usr/bin/time %M),
#   started, note (the unit's RESULT line).
# Exit:   0 every unit done, 3 some waiting for their inputs (none failed), 1 a unit failed, 2 usage.
# Usage:  compare.sh <levels> [options]   (inside the box; the box's compare.sh wrapper sets the environment and
#         holds the box)

set -uo pipefail

usage="usage: compare.sh <levels, e.g. 1 or 1,2 or 3,4,5> [--datasets <list.txt>] [--only <ds,...>] [--force] [--workers <n>]"
(( $# >= 1 )) || { echo "${usage}" >&2; exit 2; }
levels_arg="$1"
shift
[[ "${levels_arg}" =~ ^[1-5](,[1-5])*$ ]] || { echo "ERROR: levels must be comma-separated digits 1-5, got '${levels_arg}'. ${usage}" >&2; exit 2; }
for v in CHERI_ROOT CHROMBPNET_ROOT COMPENDIA_ROOT COMPARISON_ROOT BOX_DIR; do
    [[ -n "${!v:-}" ]] || { echo "ERROR: set ${v} (the box's compare.sh wrapper sets it)" >&2; exit 2; }
done
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
here="${repo_root}/workflows/comparison"
case ":${PATH}:" in
    *":${HOME}/.pixi/bin:"*) ;;
    *) export PATH="${HOME}/.pixi/bin:${PATH}" ;;
esac
# The box wrapper holds its one-run lock on fd 200; the units must not inherit it.
exec 200>&-

list="${COMPENDIA_ROOT}/inputs/shared10.txt"
only=""
force=()
workers="${COMPARE_WORKERS:-10}"
while (( $# )); do
    case "$1" in
        --datasets) list="${2:?--datasets needs a file}"; shift 2 ;;
        --only) only="${2:?--only needs dataset names}"; shift 2 ;;
        --force) force=( --force ); shift ;;
        --workers) workers="${2:?--workers needs a number}"; shift 2 ;;
        -h|--help) sed -n '2,/^set -uo/p' "${BASH_SOURCE[0]}" | sed '$d'; exit 0 ;;
        *) echo "ERROR: unknown option '$1'. ${usage}" >&2; exit 2 ;;
    esac
done
threads="${COMPARE_THREADS:-2}"
niceness="${COMPARE_NICE:-10}"
for n in "${workers}" "${threads}"; do
    [[ "${n}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: the workers and COMPARE_THREADS must be positive integers" >&2; exit 2; }
done
[[ "${niceness}" =~ ^[0-9]+$ ]] || { echo "ERROR: COMPARE_NICE must be a whole number" >&2; exit 2; }
[[ -f "${list}" ]] || { echo "ERROR: no dataset list ${list}" >&2; exit 2; }
[[ -x /usr/bin/time ]] || { echo "ERROR: no /usr/bin/time, which measures each unit's max RSS" >&2; exit 2; }

datasets=()
while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line%%#*}"
    line="${line//[[:space:]]/}"
    [[ -n "${line}" ]] || continue
    [[ " ${datasets[*]} " == *" ${line} "* ]] || datasets+=( "${line}" )
done < "${list}"
(( ${#datasets[@]} > 0 )) || { echo "ERROR: ${list} names no dataset" >&2; exit 2; }
if [[ -n "${only}" ]]; then
    IFS=, read -r -a only_list <<< "${only}"
    for ds in "${only_list[@]}"; do
        [[ " ${datasets[*]} " == *" ${ds} "* ]] || { echo "ERROR: --only names ${ds}, which is not in ${list}" >&2; exit 2; }
    done
fi

pixi_run=( pixi run --frozen --manifest-path "${CHERI_PIXI_MANIFEST:-${repo_root}/pixi.toml}" -e cherimoya )
scratch="/dev/shm/${USER}/box.${SLURM_JOB_ID:-local}/compare.$$"
record_file="${BOX_DIR}/compare_steps.tsv"
status_file="${scratch}/status"
mkdir -p "${scratch}" "${COMPARISON_ROOT}/logs" || { echo "ERROR: cannot create ${scratch} or ${COMPARISON_ROOT}/logs" >&2; exit 1; }
: > "${status_file}"
[[ -s "${record_file}" ]] || printf 'level\tunit\texit\tseconds\tmax_rss_kb\tstarted\tnote\n' >> "${record_file}"
export OMP_NUM_THREADS="${threads}" MKL_NUM_THREADS="${threads}" OPENBLAS_NUM_THREADS="${threads}" \
       NUMBA_NUM_THREADS="${threads}" CUDA_VISIBLE_DEVICES=
dbs=()
[[ -n "${COMPARE_MOTIF_DB:-}" ]] && dbs+=( --motif-db "${COMPARE_MOTIF_DB}" )
[[ -n "${COMPARE_BIAS_DB:-}" ]] && dbs+=( --bias-db "${COMPARE_BIAS_DB}" )

log() { printf '[%s] %s\n' "$(date)" "$*"; }

# selected <ds> -- whether --only lets this dataset's units run.
selected() {
    [[ -z "${only}" || ",${only}," == *",$1,"* ]]
}

# unit <level> <name> <script> [args...] -- one unit, in the foreground: under nice, its max RSS measured, its
# output appended to its log; writes its record row and status line; returns its exit.
unit() {
    local level="$1" name="$2" script="$3" t0 rc rss started note logf rss_file first
    shift 3
    logf="${COMPARISON_ROOT}/logs/level${level}/${name}.log"
    mkdir -p "${logf%/*}"
    touch "${logf}"
    first=$(( $(wc -l < "${logf}") + 1 ))
    rss_file="${scratch}/rss.${BASHPID}.${RANDOM}"
    started="$(date -Iseconds)"
    t0="${SECONDS}"
    if [[ ! -f "${here}/${script}" ]]; then
        echo "==== [$(date)] level ${level} ${name}: ${here}/${script} does not exist" >> "${logf}"
        rc=1
        note="not built: no ${script}"
        rss=NA
    else
        printf '==== [%s] level %s %s on CPUs %s: %s %s\n' "$(date)" "${level}" "${name}" \
            "$(awk '/^Cpus_allowed_list:/{print $2}' /proc/self/status)" "${script}" "$*" >> "${logf}"
        /usr/bin/time -f '%M' -o "${rss_file}" nice -n "${niceness}" \
            "${pixi_run[@]}" python "${here}/${script}" "$@" >> "${logf}" 2>&1
        rc=$?
        rss="$(tail -n 1 "${rss_file}" 2> /dev/null)"
        [[ "${rss}" =~ ^[0-9]+$ ]] || rss=NA
        rm -f "${rss_file}"
        note="$(tail -n "+${first}" "${logf}" | grep '^RESULT: ' | tail -n 1 | cut -c9- | tr '\t' ' ')"
        [[ -n "${note}" ]] || note="no RESULT line (see ${logf})"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${level}" "${name}" "${rc}" "$(( SECONDS - t0 ))" "${rss}" "${started}" \
        "${note}" >> "${record_file}"
    printf '%s\t%s\t%s\n' "${level}" "${name}" "${rc}" >> "${status_file}"
    case "${rc}" in
        0) log "level ${level} ${name}: done in $(( SECONDS - t0 )) s: ${note}" ;;
        3) log "level ${level} ${name}: waiting: ${note#waiting: }" ;;
        *) log "level ${level} ${name}: FAILED, exit ${rc}: ${note} (${logf})" ;;
    esac
    return "${rc}"
}

# pool <level> <name> <script> [args...] -- the unit in the background, at most ${workers} at once.
pool() {
    while (( $(jobs -rp | wc -l) >= workers )); do
        wait -n
    done
    ( trap - TERM INT HUP; unit "$@" ) &
}

# one <level> <name> <script> [args...] -- the unit on its own, waited for in a way a signal interrupts (bash runs a
# trap only once a foreground child exits, but at once during `wait`).
one() {
    ( trap - TERM INT HUP; unit "$@" ) &
    wait "$!"
}

# shellcheck disable=SC2329  # kill_tree and on_signal run from the trap below
kill_tree() {  # kill_tree <pid>: its descendants, then itself
    local c
    for c in $(pgrep -P "$1"); do kill_tree "${c}"; done
    kill -TERM "$1" 2> /dev/null
}
# shellcheck disable=SC2329
on_signal() {
    trap - TERM INT HUP
    log "signal: stopping every unit this run started (no markers are written for them)"
    local c
    for c in $(pgrep -P $$); do kill_tree "${c}"; done
    wait
    rm -rf "${scratch}"
    exit 143
}
trap on_signal TERM INT HUP

# ---------------------------------------------------------------------------------------------------------------------
# the levels

level_1() {
    one 1 performance compare_performance.py --datasets "${list}" "${force[@]}"
}

level_2() {
    local ds windows="${COMPARE_WINDOWS:-250,500}" block="${COMPARE_BLOCK_ROWS:-4096}" f16
    for ds in "${datasets[@]}"; do
        selected "${ds}" || continue
        pool 2 "${ds}" compare_attributions.py --dataset "${ds}" --windows "${windows}" --block-rows "${block}" \
            "${force[@]}"
    done
    wait
    f16="${COMPARE_FLOAT16_DATASET:-${datasets[0]}}"
    if [[ "${f16}" != none ]] && selected "${f16}"; then
        one 2 "${f16}.float16" compare_attributions.py --dataset "${f16}" --cheri-source h5 --tag float16 \
            --windows "${windows}" --block-rows "${block}" "${force[@]}"
    fi
    one 2 summary compare_attributions.py --summarize --datasets "${list}" "${force[@]}"
}

level_3() {
    local ds joint="${COMPARE_JOINT:-joint20}" bias=() h5 refs=()
    if [[ -n "${COMPARE_BIAS_MODISCO:-}" ]]; then
        IFS=, read -r -a refs <<< "${COMPARE_BIAS_MODISCO}"
        for h5 in "${refs[@]}"; do bias+=( --bias-modisco "${h5}" ); done
    fi
    for ds in "${datasets[@]}"; do
        selected "${ds}" || continue
        pool 3 "${ds}" compare_modisco.py --dataset "${ds}" --part average --joint "${joint}" "${dbs[@]}" \
            "${force[@]}"
        pool 3 "${ds}.qc" compare_modisco.py --dataset "${ds}" --part qc --joint "${joint}" "${dbs[@]}" "${force[@]}"
    done
    wait
    one 3 bias compare_modisco.py --bias --datasets "${list}" --joint "${joint}" "${dbs[@]}" "${bias[@]}" \
        "${force[@]}"
    one 3 summary compare_modisco.py --summarize --datasets "${list}" --joint "${joint}" "${force[@]}"
}

level_4() {
    one 4 compendia compare_compendia.py --datasets "${list}" --joint "${COMPARE_JOINT:-joint20}" "${dbs[@]}" \
        "${force[@]}"
}

level_5() {
    local ds joint="${COMPARE_JOINT:-joint20}"
    for ds in "${datasets[@]}"; do
        selected "${ds}" || continue
        pool 5 "${ds}" compare_finemo.py --dataset "${ds}" --joint "${joint}" "${force[@]}"
        if [[ "${COMPARE_PRODUCTION:-0}" == 1 ]]; then
            pool 5 "${ds}.production" compare_finemo.py --dataset "${ds}" --production "${force[@]}"
        fi
    done
    wait
    one 5 summary compare_finemo.py --summarize --datasets "${list}" --joint "${joint}" "${force[@]}"
}

log "compare.sh ${levels_arg} on $(hostname), job ${SLURM_JOB_ID:-local} step ${SLURM_STEP_ID:-?}: ${#datasets[@]}" \
    "dataset(s) from ${list}${only:+ (only ${only})}; ${workers} unit(s) at once, ${threads} thread(s) each, nice" \
    "${niceness}; out ${COMPARISON_ROOT}; CPUs $(awk '/^Cpus_allowed_list:/{print $2}' /proc/self/status)"
for level in 1 2 3 4 5; do
    [[ ",${levels_arg}," == *",${level},"* ]] || continue
    log "level ${level}: start"
    case "${level}" in
        1) level_1 ;;
        2) level_2 ;;
        3) level_3 ;;
        4) level_4 ;;
        5) level_5 ;;
    esac
done

failed=$(awk -F'\t' '$3 != 0 && $3 != 3' "${status_file}" | wc -l)
waiting=$(awk -F'\t' '$3 == 3' "${status_file}" | wc -l)
done_n=$(awk -F'\t' '$3 == 0' "${status_file}" | wc -l)
rm -rf "${scratch}"
log "compare.sh ${levels_arg}: ${done_n} unit(s) done, ${waiting} waiting for inputs, ${failed} failed"
(( failed > 0 )) && exit 1
(( waiting > 0 )) && exit 3
exit 0
