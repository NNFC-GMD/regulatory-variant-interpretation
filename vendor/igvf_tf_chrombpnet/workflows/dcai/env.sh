#!/bin/bash
# env.sh
# Purpose: the machine-specific environment for the DCAI cluster
#   (hpc.ite.dcai.dk), in one place. Source it before anything else:
#
#     source workflows/dcai/env.sh
#
# Like workflows/molab/env.sh, it sets only what differs from lib/bash/common.sh's
# defaults: where pixi, the chrombpnet checkout, the references and the data
# live. Every environment stays a pixi environment entered by activate_env.
#
# DCAI's policy is one whole node per job -- 216 usable cores, ~2 TB RAM,
# 8x H100 80GB -- billed by node-time, so work is packed into one exclusive
# allocation by workflows/dcai/run_box.sh rather than submitted step by step.
#
# Every value can be overridden by exporting it first.
#
# Input:  nothing
# Output: exported variables
# Usage:  source workflows/dcai/env.sh
# Prerequisites: pixi at ~/.pixi/bin; `pixi install -e preprocess` and
#   `-e peaks` from this checkout; the chrombpnet checkout's cuda13 env.

DCAI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="${REPO_ROOT:-$(cd "${DCAI_DIR}/../.." && pwd)}"

# pixi is installed per user and is not on the default PATH.
case ":${PATH}:" in
    *":${HOME}/.pixi/bin:"*) ;;
    *) export PATH="${HOME}/.pixi/bin:${PATH}" ;;
esac

# NNFC-GMD/chrombpnet, branch stable-modern-gpu-support, at CHROMBPNET_REV.
export CHROMBPNET_REPO="${CHROMBPNET_REPO:-/dcai/users/mateug/git/chrombpnet}"
# H100 nodes; cuda12 if `nvidia-smi` reports a driver below 580.
export CHROMBPNET_PIXI_ENV="${CHROMBPNET_PIXI_ENV:-cuda13}"
export CONDA_OVERRIDE_CUDA="${CONDA_OVERRIDE_CUDA:-13.0}"

# IGVF/ENCODE references, fetched by `cli.py download-references`.
export REFERENCE_ROOT="${REFERENCE_ROOT:-/dcai/projects/iu_0109/annotations}"
# The project's dataset tree: fragments/, peaks/, bigwigs/, chrombpnet/.
export DATASET_ROOT="${DATASET_ROOT:-/dcai/projects/iu_0109/datasets/amsc}"

# sbatch refuses a job without an account here; it reads this variable.
export SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-iu_0109}"
