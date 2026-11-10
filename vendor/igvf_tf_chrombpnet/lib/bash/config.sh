#!/bin/bash
# shellcheck disable=SC2034  # everything here is consumed by the steps that source it
# lib/bash/config.sh - per-dataset pipeline configuration
#
# Source this at the top of any workflow step that operates on one dataset:
#
#   source "${REPO_ROOT}/lib/bash/config.sh" || exit 1
#
# It sources lib/bash/common.sh first, so a step needs only this one line.
# Cross-dataset steps (09, 04.2.qc_combined_boxplot, qc_datasets) have no
# DATASET_DIR and source common.sh directly instead.
#
# Select a dataset before submitting:
#
#   export DATASET=igvf3_cardiomyocyte     # -> config/igvf3_cardiomyocyte/config.yaml
#   cd workflows/SLURM && sbatch 04.0.train_full_model.sh
#
# DATASET is propagated automatically to SLURM jobs (--export=ALL default).
#
# This file is sourced, never executed: it uses `return`, not `exit`, so a
# failed `source lib/bash/config.sh` in an interactive shell does not kill the
# shell. Callers pair it with `|| exit 1`.

if [[ -z "${REPO_ROOT}" ]]; then
    echo "ERROR: REPO_ROOT is not set. Source this from a script with the standard bootstrap block." >&2
    return 1
fi

# shellcheck source=./common.sh
source "${REPO_ROOT}/lib/bash/common.sh" || return 1

# Pick the dataset config. Three equivalent ways, in precedence order:
#   DATASET_CONFIG=/any/path/config.yaml         explicit file
#   DATASET=<name>                               config/<name>/config.yaml
#   DATASET_DIR=/path/to/dataset                 legacy: config inside the data dir
if [[ -n "${DATASET_CONFIG}" ]]; then
    dataset_config="${DATASET_CONFIG}"
    DATASET="${DATASET:-$(basename "$(dirname "${dataset_config}")")}"
elif [[ -n "${DATASET}" ]]; then
    dataset_config="${REPO_ROOT}/config/${DATASET}/config.yaml"
elif [[ -n "${DATASET_DIR}" && -f "${DATASET_DIR}/config.yaml" ]]; then
    dataset_config="${DATASET_DIR}/config.yaml"
    DATASET="${DATASET:-$(basename "${DATASET_DIR}")}"
else
    echo "ERROR: no dataset selected. Pick one of:" >&2
    echo "  export DATASET=<name>                     # config/<name>/config.yaml" >&2
    echo "  export DATASET_CONFIG=/path/to/config.yaml" >&2
    echo "  export DATASET_DIR=/path/to/dataset       # legacy layout" >&2
    if [[ -d "${REPO_ROOT}/config" ]]; then
        echo "" >&2
        echo "Available:" >&2
        for d in "${REPO_ROOT}"/config/*/config.yaml; do
            [[ -f "${d}" ]] && echo "  $(basename "$(dirname "${d}")")" >&2
        done
    fi
    return 1
fi

if [[ ! -f "${dataset_config}" ]]; then
    echo "ERROR: no dataset config at ${dataset_config}" >&2
    echo "  Create one:  cp -r ${REPO_ROOT}/config/example_dataset ${REPO_ROOT}/config/${DATASET}" >&2
    return 1
fi

# Available to the config for interpolation; the config decides the real paths.
dataset_dir="${DATASET_DIR:-${data_root}/${DATASET}}"

# Dataset parameters come from YAML, rendered to shell assignments by a
# stdlib-only loader: it runs before any conda env is active, so it imports
# nothing third-party. It still needs python >= 3.9 to parse -- see
# bootstrap_python in common.sh. ${...} inside values is left for the shell to
# expand, which is why this is eval and not a pipe.
_config_python="$(bootstrap_python)" || {
    echo "ERROR: need python >= 3.9 on PATH to read ${dataset_config}" >&2
    echo "  Set BOOTSTRAP_PYTHON to one, or activate an environment first." >&2
    return 1
}
_config_shell="$("${_config_python}" "${REPO_ROOT}/lib/python/utils/config.py" export "${dataset_config}")" || {
    echo "ERROR: could not read ${dataset_config}" >&2
    return 1
}
eval "${_config_shell}" || return 1

# A config-supplied reference_root has to re-derive every reference path:
# lib/bash/references.sh already ran, from the environment's REFERENCE_ROOT.
# The config is then re-applied so that explicit per-file overrides in it
# (genome_fa, blacklist, ...) still win over the re-derived defaults.
if [[ -n "${reference_root}" && "${reference_root}" != "${REFERENCE_ROOT}" ]]; then
    REFERENCE_ROOT="${reference_root}"
    export REFERENCE_ROOT
    # shellcheck source=./references.sh
    source "${REPO_ROOT}/lib/bash/references.sh" || return 1
    eval "${_config_shell}" || return 1
fi
unset _config_shell _config_python

# ── Required inputs ───────────────────────────────────────────────────────────
# ChromBPNet needs a region file, one signal file and somewhere to write. These
# are explicit in the config rather than derived from a directory convention, so
# the data can live anywhere and be named anything.
for _required in dataset_name regions signal_path output_dir; do
    if [[ -z "${!_required}" ]]; then
        echo "ERROR: ${dataset_config} is missing required key: ${_required}" >&2
        return 1
    fi
done
unset _required

# signal_type, datasets, bias_dataset, bias_sweep_folds and the assay/peak_type
# defaults are all derived by lib/python/utils/config.py and arrive with the
# rest of the config, so there is one implementation of those rules, not two.
# DATASET picks the folder; dataset_name names the outputs. They can legitimately
# differ when a config is selected by path, but when they differ under DATASET=
# it is almost always a copied template that was not finished renaming.
# Only meaningful when DATASET= picked the folder. With DATASET_CONFIG= the
# path was given explicitly and the folder name carries no intent -- a config
# that lives beside its inputs is a legitimate layout, and warning about it
# every run trains people to ignore the warning.
_config_folder="$(basename "$(dirname "${dataset_config}")")"
if [[ -z "${DATASET_CONFIG:-}" && -n "${dataset_name}" && "${dataset_name}" != "${_config_folder}" ]]; then
    echo "WARNING: dataset_name is '${dataset_name}' but the config folder is" >&2
    echo "  '${_config_folder}' (${dataset_config})." >&2
    echo "  Outputs will be named '${dataset_name}'. Rename one to match if that" >&2
    echo "  is not what you meant." >&2
fi
unset _config_folder

case "${assay}" in
    ATAC|DNASE) ;;
    *) echo "ERROR: assay must be ATAC or DNASE (got '${assay}')" >&2; return 1 ;;
esac

# ── Output layout ─────────────────────────────────────────────────────────────
# Everything the pipeline writes hangs off output_dir.
results_path="${output_dir}"
data_path="${results_path}/preprocessing"
DATASET_DIR="${dataset_dir}"          # kept for the steps that reference it
export DATASET DATASET_DIR

# folds_dir is set by common.sh to ${REPO_ROOT}/folds. Older dataset_config.sh
# copies on the cluster still set it to "${SCRIPT_DIR}/../folds", which no longer
# resolves after the workflows/ split — catch that here rather than letting
# chrombpnet fail on a missing -fl argument several minutes into a GPU job.
if [[ ! -d "${folds_dir}" ]]; then
    echo "ERROR: folds_dir does not exist: ${folds_dir}" >&2
    echo "  Remove the 'folds_dir=' line from ${DATASET_DIR}/dataset_config.sh;" >&2
    echo "  lib/bash/common.sh now sets it to \${REPO_ROOT}/folds." >&2
    return 1
fi

# Output directories
# 04.0's training variant: precision, optimizer and early-stopping patience
# (config full_model_precision / full_model_optimizer / full_model_patience, or
# FULL_MODEL_PRECISION / _OPTIMIZER / _PATIENCE, which win). chrombpnet's
# defaults -- default (TF32), adam, 5 -- leave the directory as it was; anything
# else tags it (full_models_bf16_muon/), so a variant never overwrites the
# default model nor is skipped because the default one is finished, and every
# later step reads the variant the config names. Only this directory is
# tagged: run a dataset past 04.0 with one variant.
full_model_precision="${FULL_MODEL_PRECISION:-${full_model_precision:-}}"
full_model_optimizer="${FULL_MODEL_OPTIMIZER:-${full_model_optimizer:-}}"
full_model_patience="${FULL_MODEL_PATIENCE:-${full_model_patience:-}}"
full_model_tag=""
if [[ -n "${full_model_precision}" && "${full_model_precision}" != "default" ]]; then
    full_model_tag+="_${full_model_precision}"
fi
if [[ -n "${full_model_optimizer}" && "${full_model_optimizer}" != "adam" ]]; then
    full_model_tag+="_${full_model_optimizer}"
fi
if [[ -n "${full_model_patience}" && "${full_model_patience}" != "5" ]]; then
    full_model_tag+="_p${full_model_patience}"
fi
full_model_dir="${results_path}/full_models${full_model_tag}"
# Which heads 05.0 scores and 06.0-08.0 average, convert and cluster: config
# contrib_heads (a list) or CONTRIB_HEADS (space-separated, wins). Both by
# default; 09.0 and 10.0 read the counts head only, so ["counts"] halves 05.0
# (DeepSHAP on every peak, the longest GPU step) and drops the profile half of
# 06.0-08.0.
if [[ -n "${CONTRIB_HEADS:-}" ]]; then
    read -r -a contrib_heads <<< "${CONTRIB_HEADS}"
fi
contrib_heads=( ${contrib_heads[@]+"${contrib_heads[@]}"} )
if [[ ${#contrib_heads[@]} -eq 0 ]]; then
    contrib_heads=( counts profile )
fi
for _h in "${contrib_heads[@]}"; do
    if [[ "${_h}" != "counts" && "${_h}" != "profile" ]]; then
        echo "ERROR: contrib_heads may only list counts and profile, got '${_h}'" >&2
        return 1
    fi
done
unset _h
if [[ " ${contrib_heads[*]} " != *" counts "* ]]; then
    echo "ERROR: contrib_heads must include counts: 09.0 and 10.0 read the counts scores" >&2
    return 1
fi
full_model_dir_selected="${full_model_dir}" # alias kept for script compatibility
predictions_dir="${results_path}/predictions"
averaged_dir="${results_path}/contrib_scores"
compendium_dir="${results_path}/compendium"
log_dir="${results_path}/logs"
# 02.0 writes this; 03.0 reads it to decide which bias factors are worth a GPU
# job at all. See qc.bias_threshold_viability().
# 00.1 writes the filtered narrowPeak and its sidecar here, beside signal/.
peaks_dir="${data_path}/peaks"
signal_qc_dir="${results_path}/plots/signal_qc"
bias_scan_file="${signal_qc_dir}/${bias_dataset}_bias_threshold_scan.tsv"
# Per-dataset run metadata lives with that dataset's results; common.sh
# defaulted it to the collaboration root for cross-dataset steps.
metadata_dir="${METADATA_DIR:-${results_path}/metadata}"
modisco_compiled_dir="${compendium_dir}/modisco_compiled"
finemo_unified_dir="${results_path}/finemo_unified"
