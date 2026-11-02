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
#   sets the output log and the job's directory, and passes the RVI checkout as the first argument: Slurm runs a copy
#   of this file, so it can't find the checkout itself, and on DCAI a job submitted with sbatch --export is cancelled
#   by root as it starts, with no log (docs/cluster.md). The --time is a ceiling, not an estimate: the box is billed
#   for the time it runs, and a box killed at its limit loses the work in flight.
# Usage: bash workflows/pipeline/submit.sh <project.yaml>   (it runs: sbatch run_project.sbatch <checkout> <project.yaml>)
rvi_root="${1:?submit with workflows/pipeline/submit.sh}"
shift
exec bash "${rvi_root}/workflows/pipeline/run_project.sh" "$@"
