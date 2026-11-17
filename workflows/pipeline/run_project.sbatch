#!/bin/bash
#SBATCH --job-name=mimir_project
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=0
# run_project.sbatch -- one whole node for workflows/pipeline/run_project.sh. Submit it with submit.sh, which sets
#   the output log and the job's directory, adds the partition, account, GPUs (--gres=gpu:<n>), CPUs and time limit
#   of project.yaml's slurm: block, and passes the RVI checkout as the first argument: Slurm runs a copy of this
#   file, so it can't find the checkout itself, and on DCAI a job submitted with sbatch --export is cancelled by root
#   as it starts, with no log (docs/cluster.md, which also has DCAI's slurm: block). The time limit is a ceiling, not
#   an estimate: a box is billed for the time it runs, and one killed at its limit loses the work in flight.
# Usage: bash workflows/pipeline/submit.sh <project.yaml>   (it runs: sbatch run_project.sbatch <checkout> <project.yaml>)
rvi_root="${1:?submit with workflows/pipeline/submit.sh}"
shift
exec bash "${rvi_root}/workflows/pipeline/run_project.sh" "$@"
