#!/bin/bash
#SBATCH --job-name=mimir_project
#SBATCH --partition=defq
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=216
#SBATCH --mem=0
#SBATCH --time=2-00:00:00
# run_project.sbatch -- one whole DCAI node for workflows/pipeline/run_project.sh. Submit it with submit.sh, which
#   sets RVI_ROOT, the output log and the job's directory. The --time is a ceiling, not an estimate: the box is
#   billed for the time it runs, and a box killed at its limit loses the work in flight.
# Usage: bash workflows/pipeline/submit.sh <project.yaml>
exec bash "${RVI_ROOT:?submit with workflows/pipeline/submit.sh}/workflows/pipeline/run_project.sh" "$@"
