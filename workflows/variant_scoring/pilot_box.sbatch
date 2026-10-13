#!/bin/bash
#SBATCH --job-name=variant_pilot
#SBATCH --partition=defq
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=216
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=variant_pilot_%j.log
# pilot_box.sbatch -- one whole DCAI node for workflows/variant_scoring/pilot_steps.sh, a pilot of variant scoring:
# tests, windows, the throughput probe, the checks, the predictions, the scores, the comparison and the analysis.
# The job ends when the steps (and a hold, below) end; --time is only the ceiling (a node is billed for the time it is
# used, not for its limit), and PILOT_TASK_MINUTES, not --time, bounds a hung prediction task. The 12 h below covers a
# small pilot (2 datasets x 20k variants: ~10 min) and a large one with a 2-h hold and a rerun in it.
# Expected for 3.2M variants x 10 datasets x 5 folds, a 100k-row shuffled null, a 100k-row genomic null and 50k peaks
# per dataset, 8 H100s:
#   1 tests ~2 min; 2 windows ~7-10 min (the 3.2M windows, ~13.5 GB, built row by row in one process, ~30 GB of RAM;
#   the null reads all 3.2M rows, then makes its 100k shuffles in parallel); 3 probe ~4 min; 4 checks ~3-4 min (12
#   processes each read all of windows/variants.npz: ~200 GB of RAM for a minute); 5 predict ~4.8-5.2 h (100 tasks,
#   ChromBPNet ~36 min, Cherimoya ~9.5 min, each after a ~13.5 GB window load; longest first over 8 slots; ~20 GB of
#   RAM per slot); 6 score ~25-45 min (20 scores.py processes ~5 GB each, 5-15 min of mostly I/O; two compare.py
#   ~50 GB each, ~8-10 min; the analysis ~10-30 min); about 6-6.5 h in all.
# With PILOT_HOLD_MINUTES=N, a run with a failed row that holds (pilot_steps.sh's Hold: a step the later steps or the
# report need, not V2 or tests that gate nothing) keeps the node up to N more minutes, and past them while a rerun
# started in it runs, for a rerun inside it: the box log prints the command, and
# `touch <PILOT_ROOT>/box/<job>.<step>/release` ends the job early. A run without such a row ends at once.
# The log goes to <PILOT_ROOT>/logs/pilot_box_<job>.log. Pass --output into <PILOT_ROOT>/logs as below, or the
# #SBATCH --output stub (it only says where the log is) lands in the submit directory, the repository root.
# The steps run from a copy of pilot_steps.sh in <PILOT_ROOT>/logs, so the checkout's copy can be edited while the
# box runs (bash reads a script as it runs it); the Python scripts still run from the checkout.
# Usage: mkdir -p <PILOT_ROOT>/logs && PILOT_ROOT=<PILOT_ROOT> sbatch --output=<PILOT_ROOT>/logs/slurm_%j.out \
#            [--time=12:00:00] [--dependency=afterany:<job>] workflows/variant_scoring/pilot_box.sbatch
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
