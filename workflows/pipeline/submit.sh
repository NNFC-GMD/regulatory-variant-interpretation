#!/bin/bash
# submit.sh -- submit one project's box (run_project.sbatch -> run_project.sh), after the checks that need no box.
#
#   1. this checkout has no uncommitted changes, and its HEAD is on a remote branch (the run is reproducible)
#   2. vendor/igvf_tf_chrombpnet is as its MANIFEST.tsv says (the copied igvf_tf_chrombpnet steps, unmodified)
#   3. the project's inputs are in place (project.py check fetch) and the dry run resolves every stage
#   4. no other box of this user is running or pending (DCAI: one box at a time)
# then sbatch, with the log at <root>/box/slurm_<job>.log, and the start Slurm expects, in ET. The partition,
# account, GPUs, CPUs and time limit come from project.yaml's slurm: block (project.py sbatch-args), as sbatch
# options, which beat run_project.sbatch's #SBATCH lines; what the block leaves out is the cluster's default.
#
# Usage: bash workflows/pipeline/submit.sh <project.yaml> [--stages "<stage> ..."]
#        SUBMIT_FORCE=1 skips check 1 (a test).
set -uo pipefail
project_file="$(realpath "${1:?usage: submit.sh <project.yaml> [--stages \"...\"]}")"
shift
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
here="${repo_root}/workflows/pipeline"
# shellcheck source=workflows/pipeline/lib.sh
source "${here}/lib.sh"
export PATH="${HOME}/.pixi/bin:${PATH}"
die() { echo "ERROR: $*" >&2; exit 1; }

state="$(code_state "${repo_root}")"
if [[ "${SUBMIT_FORCE:-0}" != 1 ]]; then
    [[ "${state}" != *+changes* ]] || die "${repo_root} has uncommitted changes; commit and push first"
    [[ -n "$(git -C "${repo_root}" branch -r --contains HEAD 2> /dev/null)" ]] \
        || die "HEAD ${state} is on no remote branch; push it first"
fi
# Captured first: `eval "$(cmd)" || die` would never die (eval of an empty string succeeds). project.py env refuses a
# project marked driver: false, and says why on stderr.
project_env="$(pixi run --frozen --manifest-path "${repo_root}/pixi.toml" -e default python "${here}/project.py" "${project_file}" env)" \
    || die "project.yaml does not load, or is not for the driver (driver: false)"
eval "${project_env}"
(cd "${PROJECT_VENDOR}" && awk -F'\t' '$1 !~ /^#/ && $1 != "path" { print $2 "  " $1 }' MANIFEST.tsv | sha256sum --quiet -c -) \
    || die "vendor/igvf_tf_chrombpnet differs from its MANIFEST.tsv (never edit it; workflows/pipeline/vendor_igvf.sh)"
sbatch_text="$(pixi run --frozen --manifest-path "${repo_root}/pixi.toml" -e default python "${here}/project.py" \
    "${project_file}" sbatch-args)" || die "project.py sbatch-args failed"
sbatch_opts=()
[[ -n "${sbatch_text}" ]] && mapfile -t sbatch_opts <<< "${sbatch_text}"
pixi run --frozen --manifest-path "${repo_root}/pixi.toml" -e default python "${here}/project.py" "${project_file}" check fetch \
    || die "inputs missing: run workflows/pipeline/fetch_igvf.py ${project_file} first"
bash "${here}/run_project.sh" "${project_file}" --dry-run "$@" > /dev/null 2>&1 || die "the dry run fails: bash ${here}/run_project.sh ${project_file} --dry-run"
busy="$(squeue -u "${USER}" -h -o '%i %j %T' 2> /dev/null)"
[[ -z "${busy}" ]] || die "a box is already running or pending (one at a time): ${busy}"

mkdir -p "${PROJECT_BOX}"
# No --export: on DCAI it gets the job cancelled by root as it starts (docs/cluster.md). The checkout goes as an argument.
out="$(sbatch --parsable --chdir "${PROJECT_BOX}" --output "${PROJECT_BOX}/slurm_%j.log" "${sbatch_opts[@]}" \
    "${here}/run_project.sbatch" "${repo_root}" "${project_file}" "$@")" || die "sbatch failed"
job="${out%%;*}"
log "submitted job ${job} for ${PROJECT} (RVI ${state}); log ${PROJECT_BOX}/slurm_${job}.log, box dir ${PROJECT_BOX}/${job}"
sleep 2
start="$(scontrol show job "${job}" 2> /dev/null | sed -n 's/.*StartTime=\([^ ]*\).*/\1/p' | head -n 1)"
if [[ "${start}" =~ ^[0-9]{4}- ]]; then
    # Slurm's time is the cluster's local time; through the epoch so TZ converts it instead of relabelling it.
    log "Slurm's expected start: $(TZ=America/New_York date -d "@$(date -d "${start}" +%s)" '+%F %H:%M %Z' 2> /dev/null || echo "${start}") (optimistic)"
fi
