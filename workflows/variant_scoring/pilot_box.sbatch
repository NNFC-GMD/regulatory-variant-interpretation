#!/bin/bash
#SBATCH --job-name=variant_pilot
#SBATCH --partition=defq
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=216
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=variant_pilot_%j.log
# pilot_box.sbatch -- one whole DCAI node for workflows/variant_scoring/pilot_steps.sh, the quick pilot of variant
# scoring: tests, windows, the throughput probe, the checks, the predictions, the scores.
# The job ends when the steps end (expected 30-60 min); --time is only the ceiling, set well above that because a
# node is billed for the time it is used, not for its limit (pass a longer --time to sbatch to override it).
# The log goes to <PILOT_ROOT>/logs/pilot_box_<job>.log. Pass --output into <PILOT_ROOT>/logs as below, or the
# #SBATCH --output stub (it only says where the log is) lands in the submit directory, the repository root.
# The steps run from a copy of pilot_steps.sh in <PILOT_ROOT>/logs, so the checkout's copy can be edited while the
# box runs (bash reads a script as it runs it); the Python scripts still run from the checkout.
# Usage: mkdir -p <PILOT_ROOT>/logs && PILOT_ROOT=<PILOT_ROOT> sbatch --output=<PILOT_ROOT>/logs/slurm_%j.out \
#            [--dependency=afterany:<job>] workflows/variant_scoring/pilot_box.sbatch
#        (submit from the repo root; SBATCH_ACCOUNT must be set; pilot_steps.sh reads <PILOT_ROOT>/pilot.env)

repo_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -f "${repo_root}/workflows/variant_scoring/pilot_steps.sh" ]] || {
    echo "ERROR: submit from the regulatory-variant-interpretation checkout" >&2; exit 1; }
: "${PILOT_ROOT:?set PILOT_ROOT to the pilot directory}"
mkdir -p "${PILOT_ROOT}/logs" || exit 1
log="${PILOT_ROOT}/logs/pilot_box_${SLURM_JOB_ID:-local}.log"
echo "[$(TZ=America/New_York date '+%F %T %Z')] job ${SLURM_JOB_ID:-local} on $(hostname): logging to ${log}"
exec >> "${log}" 2>&1
steps_copy="${PILOT_ROOT}/logs/pilot_steps_${SLURM_JOB_ID:-local}.sh"
cp "${repo_root}/workflows/variant_scoring/pilot_steps.sh" "${steps_copy}" || exit 1
PILOT_REPO_ROOT="${repo_root}" exec bash "${steps_copy}"
