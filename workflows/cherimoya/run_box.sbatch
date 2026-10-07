#!/bin/bash
#SBATCH --job-name=cherimoya_box
#SBATCH --partition=defq
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=216
#SBATCH --mem=0
#SBATCH --time=3-00:00:00
#SBATCH --output=cherimoya_box_%j.log
# run_box.sbatch -- one whole DCAI node for workflows/cherimoya/run_box.sh.
# The --time is a ceiling, not an estimate: defq's default is 8 h, and a box
# killed at its limit loses the trainings in flight (they restart from scratch).
# Usage: CHERI_ROOT=<make_jobs out-root> sbatch [--dependency=afterany:<job>] run_box.sbatch
#        (submit from the repo root; SBATCH_ACCOUNT must be set)

repo_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -f "${repo_root}/workflows/cherimoya/run_box.sh" ]] || {
    echo "ERROR: submit from the regulatory-variant-interpretation checkout" >&2; exit 1; }
exec bash "${repo_root}/workflows/cherimoya/run_box.sh"
