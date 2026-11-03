# shellcheck shell=bash
# lib.sh -- helpers run_project.sh and submit.sh share. Sourced, never executed: it defines functions, sets no shell
#   option, starts nothing and writes nothing, so sourcing it is safe anywhere (under set -u, on the login node).
#   Most come from the AMSC tn5 box's lib.sh, which ran them for a whole campaign.
#
#   et                        the time now in ET (America/New_York), '2026-11-02 10:00:00 EST'
#   log <message...>          '[<ET>] <message>' on stdout
#   run_rec <file> <stage> <state> <exit> <seconds> <started_et> [<note>]
#                             one row of a box's stages.tsv, under flock on <file>.lock; the header first when the
#                             file is new (stage state exit seconds started_et note). Tabs and newlines become spaces.
#   code_state <checkout>     its short HEAD, '+changes' when a tracked file differs from it, '?' for no git checkout
#   allowed_cpus              the CPUs this process may use (Cpus_allowed_list), one per line
#   cpu_block <i> <n>         the i-th block of n allowed CPUs, as taskset takes them (a,b,c...); empty past the end
#   kill_tree <pid>           SIGTERM to its descendants, deepest first, then to it
#   hold <box dir> <minutes> <what>
#                             after a failure: holds <box>/.hold.pipeline for up to <minutes>, polling every 20 s.
#                             `touch <box>/retry` ends it with status 0 (the caller reruns the stage), `touch
#                             <box>/release` or the time running out with status 1 (the caller ends the box).

et() { TZ=America/New_York date '+%F %T %Z'; }
log() { printf '[%s] %s\n' "$(et)" "$*"; }

run_rec() {
    local f="$1" field row=""
    shift
    (( $# >= 5 )) || { echo "run_rec: usage: run_rec <file> <stage> <state> <exit> <seconds> <started_et> [<note>]" >&2; return 2; }
    for field in "$1" "$2" "$3" "$4" "$5" "${6:-}"; do
        field="${field//$'\t'/ }"
        row+="${row:+$'\t'}${field//$'\n'/ }"
    done
    mkdir -p "$(dirname "${f}")" || return 1
    (
        flock -x 8
        [[ -s "${f}" ]] || printf 'stage\tstate\texit\tseconds\tstarted_et\tnote\n' >> "${f}"
        printf '%s\n' "${row}" >> "${f}"
    ) 8> "${f}.lock"
}

code_state() {
    local h s
    h="$(git -C "$1" rev-parse --short HEAD 2> /dev/null)" || { echo "?"; return 0; }
    s="$(git --no-optional-locks -C "$1" status --porcelain --untracked-files=no 2> /dev/null)" || { echo "${h}"; return 0; }
    [[ -n "${s}" ]] && h+="+changes"
    echo "${h}"
}

allowed_cpus() {
    local r ranges
    IFS=, read -r -a ranges <<< "$(awk '/^Cpus_allowed_list:/{print $2}' /proc/self/status)"
    for r in "${ranges[@]}"; do
        if [[ "${r}" == *-* ]]; then seq "${r%-*}" "${r#*-}"; else echo "${r}"; fi
    done
}

cpu_block() {
    local i="$1" n="$2"
    allowed_cpus | awk -v lo=$(( i * n )) -v hi=$(( (i + 1) * n )) 'NR > lo && NR <= hi' | paste -sd,
}

kill_tree() {
    local c
    for c in $(pgrep -P "$1"); do kill_tree "${c}"; done
    kill -TERM "$1" 2> /dev/null
}

hold() {
    local box="$1" minutes="$2" what="$3" t_end
    touch "${box}/.hold.pipeline"
    rm -f "${box}/retry"
    log "HOLD: ${what}. For ${minutes} min the box waits: look at ${box}/logs, fix, then" \
        "\`touch ${box}/retry\` to run the stage again, or \`touch ${box}/release\` to end the box." \
        "Commands inside the box: srun --jobid ${SLURM_JOB_ID:-<job>} --overlap -N1 -n1 <cmd>"
    t_end=$(( SECONDS + minutes * 60 ))
    while (( SECONDS < t_end )); do
        if [[ -e "${box}/retry" ]]; then
            rm -f "${box}/retry" "${box}/.hold.pipeline"
            log "retry requested"
            return 0
        fi
        [[ -e "${box}/release" ]] && break
        sleep 20
    done
    rm -f "${box}/.hold.pipeline"
    return 1
}
