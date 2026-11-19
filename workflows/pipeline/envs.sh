#!/bin/bash
# envs.sh -- source me. The environments the steps copied from igvf_tf_chrombpnet (vendor/igvf_tf_chrombpnet/) run
# in are this repository's: envs/pixi.toml and envs/pixi.lock (started as the copy's own, byte for byte; changes
# since are listed in envs/pixi.toml's header). The copied steps' lib/bash/common.sh reads PREPROCESS_ENV,
# PEAKS_ENV, MODISCO_ENV, FINEMO_ENV and MOTIF_COMPENDIUM_ENV (a value already set wins, here as there); the copy's
# own pixi.toml is not used. CHROMBPNET_ENV is not set here: run_project.sh sets it to this repository's `chrombpnet`
# env, and the comparison scripts keep the chrombpnet checkout's (CHROMBPNET_PIXI_ENV).
#   STEP_ENVS          the environments setup.sh installs from envs/pixi.toml and the preflight requires
#   step_run <env> ... a command in one of them (pixi run --frozen)
# shellcheck disable=SC2034  # STEP_ENVS is read by the scripts that source this
_envs_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RVI_ENVS_MANIFEST="${_envs_root}/envs/pixi.toml"
STEP_ENVS=(peaks preprocess modisco motif-compendium finemo)
export PREPROCESS_ENV="${PREPROCESS_ENV:-pixi:${RVI_ENVS_MANIFEST}#preprocess}"
export PEAKS_ENV="${PEAKS_ENV:-pixi:${RVI_ENVS_MANIFEST}#peaks}"
export MODISCO_ENV="${MODISCO_ENV:-pixi:${RVI_ENVS_MANIFEST}#modisco}"
export FINEMO_ENV="${FINEMO_ENV:-pixi:${RVI_ENVS_MANIFEST}#finemo}"
export MOTIF_COMPENDIUM_ENV="${MOTIF_COMPENDIUM_ENV:-pixi:${RVI_ENVS_MANIFEST}#motif-compendium}"

step_run() { pixi run --frozen --manifest-path "${RVI_ENVS_MANIFEST}" -e "$@"; }
