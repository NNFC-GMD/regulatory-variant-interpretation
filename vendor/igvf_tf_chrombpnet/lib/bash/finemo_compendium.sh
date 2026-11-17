#!/bin/bash
# shellcheck disable=SC2034  # everything here is consumed by the steps that source it
# lib/bash/finemo_compendium.sh - which compendium 10.0 and 11.0 call hits
# with, and where those hits go.
#
# Sourced right after lib/bash/config.sh by 10.0, 11.0 and status.sh, and by
# nothing else:
#
#   source "${REPO_ROOT}/lib/bash/config.sh" || exit 1
#   source "${REPO_ROOT}/lib/bash/finemo_compendium.sh" || exit 1
#
# Sets, each from the environment variable, else the config key of the same
# name in lower case, else the default:
#   finemo_compendium_dir  FINEMO_COMPENDIUM_DIR: the directory 09.0 built,
#                          holding modisco_compiled.h5. Default
#                          ${REPO_ROOT}/results/compendium/modisco_compiled,
#                          where 09.0's default mode builds it.
#   finemo_compendium_tag  FINEMO_COMPENDIUM_TAG: names the hits directory.
#                          Default the compendium directory's name, or
#                          "unified" for the default directory. Letters,
#                          digits and '_' only.
#   finemo_hits_dir        ${results_path}/finemo_<tag>. Tag "unified" gives
#                          config.sh's finemo_unified_dir, as before.
#
# One hits directory per compendium, because 10.0 skips a dataset whose
# hits.bed.gz exists: hits for a second compendium written into the first
# one's directory would be skipped as done. 10.0 also records the compendium's
# md5 beside the hits (compendium.md5), and 10.0 and 11.0 stop when a tag is
# reused for another compendium.
#
# A separate file rather than more of config.sh, which every step sources:
# only these three read it, and config.sh stays as every other step has it.
#
# This file is sourced, never executed: it uses `return`, not `exit`.

finemo_compendium_dir="${FINEMO_COMPENDIUM_DIR:-${finemo_compendium_dir:-}}"
if [[ -n "${finemo_compendium_dir}" ]]; then
    finemo_compendium_dir="${finemo_compendium_dir%/}"
    [[ "${finemo_compendium_dir}" == /* ]] || finemo_compendium_dir="${PWD}/${finemo_compendium_dir}"
    finemo_compendium_tag="${FINEMO_COMPENDIUM_TAG:-${finemo_compendium_tag:-$(basename "${finemo_compendium_dir}")}}"
else
    finemo_compendium_dir="${REPO_ROOT}/results/compendium/modisco_compiled"
    finemo_compendium_tag="${FINEMO_COMPENDIUM_TAG:-${finemo_compendium_tag:-unified}}"
fi
if [[ ! "${finemo_compendium_tag}" =~ ^[A-Za-z0-9_]+$ ]]; then
    echo "ERROR: bad Fi-NeMo compendium tag '${finemo_compendium_tag}': letters, digits and '_' only." >&2
    echo "  Set FINEMO_COMPENDIUM_TAG, or rename the compendium directory." >&2
    return 1
fi
finemo_hits_dir="${results_path}/finemo_${finemo_compendium_tag}"
