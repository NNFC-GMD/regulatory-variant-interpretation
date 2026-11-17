#!/bin/bash
# shellcheck disable=SC2218  # false positive in shellcheck 0.11.0: activate_env and
# metadata_start both come from lib/bash/common.sh, sourced above.
#SBATCH --job-name=cross_dataset_compendium
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --partition=normal,engreitz
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 09.0.cross_dataset_compendium.sh
# Purpose: Build a single non-redundant motif compendium by pooling MoDISco
#          results across datasets: the inputs a list names (list mode), or
#          by default the four datasets of this collaboration:
#            igvf3_cardiomyocyte, igvf6_definitive_endoderm,
#            igvf11_h7_hesc, igvf_endothelial (d3 iPSC-EC)
#
# This script does NOT require DATASET_DIR; it operates at the collaboration
# root, over the datasets it is given.
#
# Default mode (COMPENDIUM_INPUTS unset), unchanged: the four datasets above.
# Each one's MoDISco H5 is where 08.0 wrote it according to that dataset's
# config/<ds>/config.yaml (${averaged_dir}/<ds>/modisco/), so a fresh results
# tree is followed; a dataset without a config falls back to a hardcoded
# legacy path under ${data_root} (DATASET_ROOT, default REPO_ROOT). A missing
# H5 is skipped with a [WARN]. The compendium is rebuilt in place under
# ${REPO_ROOT}/results/compendium/modisco_compiled, where 10.0 reads it by
# default, with MotifCompendium's default MoDISco window.
#
# List mode (COMPENDIUM_INPUTS=<list.tsv>): one `label<TAB>path` row per input
# ('#' lines and blank lines skipped), the rules in
# lib/python/utils/compendium_inputs.py. A path is either
#   - a dataset config.yaml, resolved the way 08.0 resolves it: config.sh in a
#     child bash, with that config's errors shown, gives
#     ${averaged_dir}/${datasets[0]}/modisco/modisco_counts_results.h5; or
#   - a MoDISco results H5, used as it is.
# A label may hold only letters, digits and '_' (MotifCompendium names each
# motif <label>-<pos|neg>.pattern_<n>), and labels are unique, so one dataset
# can enter twice under two labels (one per model, say) and stay apart.
#   - A missing H5 stops the step, with every missing one listed, unless
#     COMPENDIUM_ALLOW_MISSING=1, which drops each with a [WARN] instead.
#   - The compendium goes to COMPENDIUM_DIR (default ${REPO_ROOT}/results/
#     compendium/modisco_compiled) and is immutable: an existing directory
#     stops the step, unless COMPENDIUM_OVERWRITE=1, which moves it to
#     <dir>.replaced.<time> once the new one is built. Give a rebuild a new
#     directory instead: MotifCompendium writes modisco_compiled.h5 in place,
#     and a 10.0 reading the old one would see it half-written.
#   - The build goes into <dir>.partial.<pid>, which is moved to <dir> only
#     after motif_compendium.py exits 0 and all five of its outputs exist.
#     A failed build is left there for inspection.
#   - MotifCompendium gets the MoDISco window the inputs were made with
#     (motif_compendium.py --modisco-region-width auto: each results file's
#     window_size; 08.0 runs -w 500).
#   - Run metadata goes to METADATA_DIR, default <dir>/../metadata, beside the
#     compendia rather than into the checkout. Set it for this step alone: an
#     exported METADATA_DIR would also redirect every per-dataset step's.
#   - COMPENDIUM_DRY_RUN=1 resolves the list, prints `label<TAB>H5` per input
#     on stdout (everything else on stderr) and exits 0, missing H5s included,
#     without entering an environment or writing anything.
# COMPENDIUM_DIR and COMPENDIUM_DRY_RUN need a list: without one this step
# would build the four datasets above, in place.
#
# CPU only, in ${motif_compendium_env} (this repo's motif-compendium pixi
# environment, MotifCompendium v1.0.19). Clustering is cpm_leiden at
# ${motif_compendium_threshold}, passed to src/motif_compendium.py explicitly:
# v1.0.19's default appends a k-centroids pass that ignores the threshold.
#
# Input:  Per-dataset fold-averaged MoDISco H5s (step 08 for each dataset):
#           default mode: ${averaged_dir}/<dataset>/modisco/modisco_counts_results.h5
#           from each dataset's config, else the legacy
#           ${data_root}/<dataset>/results/contrib_scores/.../modisco_counts_results.h5
#           list mode: the H5 each row names, or its config resolves to
#         ${COMPENDIUM_INPUTS} (list mode)
#         ${ref_db_meme} (from `cli.py download-references`)
# Output (inside ${REPO_ROOT}/results/compendium/modisco_compiled/, or
#   ${COMPENDIUM_DIR} in list mode):
#   modisco_compiled.h5         - clustered motifs for FiNeMo (cross-dataset)
#   modisco_compendium.meme     - MEME format
#   modisco_compendium.mc       - pickled MotifCompendium object
#   modisco_compendium_meta.tsv - per-motif annotations + cluster IDs
#   modisco_config.tsv          - label -> H5 path mapping used for this run
#   compendium.json             - provenance, written last by motif_compendium.py:
#                                 inputs with md5s and MoDISco windows, settings,
#                                 versions, motif and cluster counts, output md5s
#
# Prerequisites: step 08 must have completed for the inputs, and
#   `pixi install -e motif-compendium` in this checkout.
#
# Usage:
#   cd workflows/SLURM && sbatch 09.0.cross_dataset_compendium.sh
#   COMPENDIUM_INPUTS=/path/inputs.tsv COMPENDIUM_DIR=/path/compendia/<tag> \
#       sbatch 09.0.cross_dataset_compendium.sh
#   COMPENDIUM_INPUTS=/path/inputs.tsv COMPENDIUM_DRY_RUN=1 bash 09.0.cross_dataset_compendium.sh

# --- bootstrap: locate the repo root (identical block in every workflow step) --
# sbatch copies the submitted script to a node-local spool dir, so BASH_SOURCE
# does not point into the repo; SLURM_SUBMIT_DIR (the cwd at submit time) does.
# Walk up from whichever is usable until lib/bash/common.sh turns up. Override
# with REPO_ROOT if you submit from outside the checkout.
# An inherited REPO_ROOT is only trusted if it really is this repo -- the name
# is generic enough that another project may have exported it.
[[ -n "${REPO_ROOT}" && -f "${REPO_ROOT}/lib/bash/common.sh" ]] || REPO_ROOT=""
if [[ -z "${REPO_ROOT}" ]]; then
    for _d in "${SLURM_SUBMIT_DIR}" "$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"; do
        while [[ -n "${_d}" && "${_d}" != "/" ]]; do
            [[ -f "${_d}/lib/bash/common.sh" ]] && { REPO_ROOT="${_d}"; break 2; }
            _d="$(dirname "${_d}")"
        done
    done
fi
[[ -n "${REPO_ROOT}" ]] || {
    echo "ERROR: cannot locate the repo root. Submit from inside the checkout, or:" >&2
    echo "  export REPO_ROOT=/path/to/this/checkout" >&2
    exit 1
}
export REPO_ROOT
# --- end bootstrap -------------------------------------------------------------

# The mode is settled before common.sh is sourced: common.sh sets metadata_dir
# from METADATA_DIR as it is sourced, and a list-mode compendium keeps its run
# records beside it rather than in the checkout.
list_mode=0
if [[ -n "${COMPENDIUM_INPUTS:-}" ]]; then
    list_mode=1
    out_dir="${COMPENDIUM_DIR:-${REPO_ROOT}/results/compendium/modisco_compiled}"
    out_dir="${out_dir%/}"
    [[ "${out_dir}" == /* ]] || out_dir="${PWD}/${out_dir}"
    METADATA_DIR="${METADATA_DIR:-$(dirname "${out_dir}")/metadata}"
elif [[ -n "${COMPENDIUM_DIR:-}" || "${COMPENDIUM_DRY_RUN:-}" == "1" ]]; then
    echo "ERROR: COMPENDIUM_DIR and COMPENDIUM_DRY_RUN need COMPENDIUM_INPUTS=<list.tsv>." >&2
    echo "  Without a list this step builds the four collaboration datasets, in place, in" >&2
    echo "  ${REPO_ROOT}/results/compendium/modisco_compiled." >&2
    exit 1
fi

# Cross-dataset step: no DATASET_DIR, so source common.sh directly rather than
# config.sh. It provides motif_compendium_env, ref_db_meme, data_root and
# motif_compendium_threshold.
# shellcheck source=lib/bash/common.sh
source "${REPO_ROOT}/lib/bash/common.sh" || exit 1

# MotifCompendium clustering algorithm: ${motif_compendium_algorithm} from
# common.sh, passed explicitly because v1.0.19 changed cluster()'s default to
# cpm_leiden followed by k_centroids, which reassigns motifs without looking at
# the similarity threshold; cpm_leiden alone is what v1.0.16 ran.

if (( list_mode )); then
    # ── List mode ────────────────────────────────────────────────────────────
    dry_run=0
    [[ "${COMPENDIUM_DRY_RUN:-}" == "1" ]] && dry_run=1

    # The list's own rules (labels, duplicates, what each path is), checked by
    # the module motif_compendium.py also reads lists with, on the bootstrap
    # python: nothing is active yet.
    _py="$(bootstrap_python)" || {
        echo "ERROR: need python >= 3.9 on PATH to read ${COMPENDIUM_INPUTS}" >&2
        echo "  Set BOOTSTRAP_PYTHON to one, or activate an environment first." >&2
        exit 1
    }
    _rows="$("${_py}" "${REPO_ROOT}/lib/python/utils/compendium_inputs.py" parse "${COMPENDIUM_INPUTS}")" || exit 1
    mapfile -t list_rows <<< "${_rows}"
    unset _rows _py

    if [[ -e "${out_dir}" ]]; then
        if (( dry_run )); then
            echo "[dry run] ${out_dir} exists; a build would need COMPENDIUM_OVERWRITE=1" >&2
        elif [[ "${COMPENDIUM_OVERWRITE:-}" != "1" ]]; then
            echo "ERROR: ${out_dir} exists. A compendium is not rebuilt in place: give the build" >&2
            echo "  a new COMPENDIUM_DIR, or set COMPENDIUM_OVERWRITE=1 to move this one to" >&2
            echo "  ${out_dir}.replaced.<time> once the new one is built." >&2
            exit 1
        fi
    fi

    # Where 08.0 writes the counts MoDISco H5 of the dataset a config names:
    # ${averaged_dir}/${datasets[0]}/modisco/ as that config's config.sh derives
    # it. Runs config.sh in a child bash, selected by DATASET_CONFIG the way
    # every per-dataset step is; its errors stay visible.
    modisco_h5_of_config() {
        # shellcheck disable=SC2016  # expanded by the child bash, after config.sh
        DATASET_CONFIG="$1" DATASET="" bash -c \
            'source "$1/lib/bash/config.sh" >/dev/null || exit 1
             printf "%s\n" "${averaged_dir}/${datasets[0]}/modisco/modisco_counts_results.h5"' \
            _ "${REPO_ROOT}"
    }

    labels=()
    h5s=()
    n_bad=0
    for _row in "${list_rows[@]}"; do
        IFS=$'\t' read -r _label _kind _path <<< "${_row}"
        if [[ "${_kind}" == "config" ]]; then
            if ! _h5="$(modisco_h5_of_config "${_path}")" || [[ -z "${_h5}" ]]; then
                echo "ERROR: ${_label}: cannot resolve the dataset config ${_path} (see above)" >&2
                n_bad=$(( n_bad + 1 ))
                continue
            fi
        else
            _h5="${_path}"
        fi
        labels+=( "${_label}" )
        h5s+=( "${_h5}" )
    done
    # Two rows that resolve to one H5 (two configs with one output_dir, or a
    # config and its own H5) would put the same motifs in twice.
    declare -A label_of_h5=()
    for i in "${!labels[@]}"; do
        if [[ -n "${label_of_h5[${h5s[i]}]:-}" ]]; then
            echo "ERROR: ${labels[i]} and ${label_of_h5[${h5s[i]}]} both resolve to ${h5s[i]}" >&2
            n_bad=$(( n_bad + 1 ))
        fi
        label_of_h5[${h5s[i]}]="${labels[i]}"
    done
    if (( n_bad )); then
        echo "ERROR: ${n_bad} problem(s) resolving the ${#list_rows[@]} input(s) in ${COMPENDIUM_INPUTS}." >&2
        exit 1
    fi

    if (( dry_run )); then
        n_missing=0
        for i in "${!labels[@]}"; do
            if [[ -f "${h5s[i]}" ]]; then
                echo "  [OK]      ${labels[i]}: ${h5s[i]}" >&2
            else
                echo "  [MISSING] ${labels[i]}: ${h5s[i]}" >&2
                n_missing=$(( n_missing + 1 ))
            fi
            printf '%s\t%s\n' "${labels[i]}" "${h5s[i]}"
        done
        echo "[dry run] ${#labels[@]} input(s), ${n_missing} H5(s) missing; would build ${out_dir}" >&2
        exit 0
    fi

    # Every missing H5 goes to the preflight below, which lists them all and
    # stops, unless COMPENDIUM_ALLOW_MISSING=1.
    config_rows=()
    for i in "${!labels[@]}"; do
        if [[ -f "${h5s[i]}" ]]; then
            config_rows+=( "${labels[i]}"$'\t'"${h5s[i]}" )
            echo "  [OK]   ${labels[i]}: ${h5s[i]}"
        elif [[ "${COMPENDIUM_ALLOW_MISSING:-}" == "1" ]]; then
            echo "  [WARN] ${labels[i]}: H5 not found, skipping: ${h5s[i]}" >&2
        else
            echo "  [MISSING] ${labels[i]}: ${h5s[i]}" >&2
            require_input "${h5s[i]}" 08.0.run_modisco.sh
        fi
    done
    n_found=${#config_rows[@]}
    if (( ${#preflight_missing[@]} == 0 && n_found == 0 )); then
        echo "ERROR: no MoDISco H5s found. Run 08.0.run_modisco.sh for the inputs first." >&2
        exit 1
    fi
    compendium_args=( --modisco-region-width auto )
    [[ "${COMPENDIUM_ALLOW_MISSING:-}" == "1" ]] && compendium_args+=( --allow-missing )
else
    # ── Default mode: the four collaboration datasets, unchanged ─────────────
    out_dir="${REPO_ROOT}/results/compendium/modisco_compiled"
    log_dir="${REPO_ROOT}/results/logs"
    mkdir -p "${out_dir}" "${log_dir}"

    # Where 08.0 wrote each dataset's counts MoDISco H5: ${averaged_dir}/<ds>/modisco/
    # as that dataset's own config.sh derives it, so a results tree moved to a new
    # output_dir (chrombpnet 2.x results go in a fresh one) is the one read -- the
    # legacy paths below would silently pick up the old tree's files instead. Runs
    # config.sh in a child bash, selected by DATASET, the way every step does.
    modisco_h5_from_config() {
        local ds="$1"
        [[ -f "${REPO_ROOT}/config/${ds}/config.yaml" ]] || return 1
        # shellcheck disable=SC2016  # expanded by the child bash, after config.sh
        DATASET="${ds}" DATASET_CONFIG="" bash -c \
            'source "$1/lib/bash/config.sh" >/dev/null 2>&1 || exit 1
             printf "%s\n" "${averaged_dir}/$2/modisco/modisco_counts_results.h5"' \
            _ "${REPO_ROOT}" "${ds}"
    }

    # Fallback for a dataset with no config/<ds>/config.yaml: the legacy layout
    # under the dataset data root. igvf_endothelial's lacks the <dataset>/ level the
    # other three have. Leave it unless the cluster layout says otherwise: a wrong
    # path only [WARN]s and drops the dataset (see CLAUDE.md, "The endothelial
    # dataset is named two ways").
    declare -A h5_map=(
        [igvf3_cardiomyocyte]="${data_root}/igvf3_cardiomyocyte/results/contrib_scores/igvf3_cardiomyocyte/modisco/modisco_counts_results.h5"
        [igvf6_definitive_endoderm]="${data_root}/igvf6_definitive_endoderm/results/contrib_scores/igvf6_definitive_endoderm/modisco/modisco_counts_results.h5"
        [igvf11_h7_hesc]="${data_root}/igvf11_h7_hesc/results/contrib_scores/igvf11_h7_hesc/modisco/modisco_counts_results.h5"
        [igvf_endothelial]="${data_root}/igvf_endothelial/results/contrib_scores/modisco/modisco_counts_results.h5"
    )

    # Build config TSV
    config_tsv="${out_dir}/modisco_config.tsv"
    echo "# dataset    modisco_h5" > "${config_tsv}"

    for dataset in igvf3_cardiomyocyte igvf6_definitive_endoderm igvf11_h7_hesc igvf_endothelial; do
        h5="$(modisco_h5_from_config "${dataset}")" || h5="${h5_map[$dataset]}"
        if [[ -f "${h5}" ]]; then
            echo -e "${dataset}\t${h5}" >> "${config_tsv}"
            echo "  [OK]   ${dataset}: ${h5}"
        else
            echo "  [WARN] ${dataset}: H5 not found, skipping: ${h5}" >&2
        fi
    done

    n_found=$(grep -c "^[^#]" "${config_tsv}" || true)
    echo "[$(date)] Config TSV: ${config_tsv} (${n_found}/4 datasets)"

    if [[ "${n_found}" -eq 0 ]]; then
        echo "ERROR: no MoDISco H5s found. Run 08.0.run_modisco.sh for each dataset first." >&2
        exit 1
    fi
    # The table holds only H5s that exist; --allow-missing keeps the old
    # warn-and-drop should one vanish before MotifCompendium reads it.
    compendium_args=( --allow-missing )
    tool_out_dir="${out_dir}"
fi

# The reference DB has a versioned name that `cli.py download-references`
# fetches, which an install made before the pin does not have yet. Check it
# here, before the environment and the compendium build, rather than meet it
# as a MotifCompendium traceback after both.
require_input "${ref_db_meme}" "cli.py download-references"
preflight_check

if (( list_mode )); then
    # The build directory, beside the final one so that the last mv is a rename.
    mkdir -p "$(dirname "${out_dir}")" || exit 1
    build_dir="${out_dir}.partial.$$"
    rm -rf "${build_dir}"
    mkdir -p "${build_dir}" || exit 1
    config_tsv="${build_dir}/modisco_config.tsv"
    { echo "# label    modisco_h5"; printf '%s\n' "${config_rows[@]}"; } > "${config_tsv}"
    echo "[$(date)] Config TSV: ${config_tsv} (${n_found}/${#labels[@]} inputs)"
    tool_out_dir="${build_dir}"
fi

# Run MotifCompendium clustering + annotation
activate_env "${motif_compendium_env}"

metadata_start "09.0.cross_dataset_compendium"
# Declared at their final paths: hashed at exit, after a list-mode build has
# been moved into place.
metadata_inputs+=( "config=${out_dir}/modisco_config.tsv" "ref_db=${ref_db_meme}" )
metadata_outputs+=( "motifs=${out_dir}/modisco_compiled.h5" "meme=${out_dir}/modisco_compendium.meme" "meta_tsv=${out_dir}/modisco_compendium_meta.tsv" )
metadata_params+=( "threshold=${motif_compendium_threshold}" "algorithm=${motif_compendium_algorithm}" )
if (( list_mode )); then
    metadata_inputs+=( "input_list=${COMPENDIUM_INPUTS}" )
    metadata_outputs+=( "provenance=${out_dir}/compendium.json" )
    metadata_params+=( "mode=list" "compendium_dir=${out_dir}" "n_inputs=${#labels[@]}" "n_found=${n_found}" )
    metadata_params+=( "modisco_region_width=auto" "allow_missing=${COMPENDIUM_ALLOW_MISSING:-0}" )
fi


echo "[$(date)] Running MotifCompendium across ${n_found} datasets (${motif_compendium_algorithm}, threshold=${motif_compendium_threshold})..."

python "${src_dir}/motif_compendium.py" \
    --config    "${config_tsv}" \
    --out-dir   "${tool_out_dir}" \
    --ref-db    "${ref_db_meme}" \
    --threshold "${motif_compendium_threshold}" \
    --algorithm "${motif_compendium_algorithm}" \
    --cpus      "${SLURM_CPUS_PER_TASK:-16}" \
    "${compendium_args[@]}"
rc=$?
# The exit status, not only the file: this step is rerun as more datasets reach
# 08.0, so an older modisco_compiled.h5 is usually already there and would let a
# failed rebuild exit 0.
if [[ ${rc} -ne 0 || ! -f "${tool_out_dir}/modisco_compiled.h5" ]]; then
    echo "ERROR: motif_compendium.py failed (exit ${rc}); see the log above." >&2
    (( list_mode )) && echo "  The partial build is left in ${build_dir} for inspection; delete it." >&2
    exit 1
fi

if (( list_mode )); then
    # All five outputs, compendium.json (written last) included, before the
    # directory takes its name.
    for _f in modisco_compiled.h5 modisco_compendium.meme modisco_compendium.mc \
              modisco_compendium_meta.tsv compendium.json; do
        if [[ ! -s "${build_dir}/${_f}" ]]; then
            echo "ERROR: motif_compendium.py exited 0 but wrote no ${build_dir}/${_f}." >&2
            echo "  The partial build is left in ${build_dir} for inspection; delete it." >&2
            exit 1
        fi
    done
    if [[ -e "${out_dir}" ]]; then
        # Only reachable with COMPENDIUM_OVERWRITE=1 (checked above).
        replaced="${out_dir}.replaced.$(date +%Y%m%dT%H%M%S)"
        if ! mv -T "${out_dir}" "${replaced}"; then
            echo "ERROR: could not move the old ${out_dir} aside; the new build is in ${build_dir}." >&2
            exit 1
        fi
        echo "[$(date)] Moved the old compendium to ${replaced}"
    fi
    # -T: fail rather than move the build INTO a directory another run made meanwhile.
    if ! mv -T "${build_dir}" "${out_dir}"; then
        echo "ERROR: could not move ${build_dir} to ${out_dir}." >&2
        exit 1
    fi
fi

echo "[$(date)] Cross-dataset compendium complete."
echo "  Compiled H5  : ${out_dir}/modisco_compiled.h5"
echo "  Annotations  : ${out_dir}/modisco_compendium_meta.tsv"
echo "  MEME         : ${out_dir}/modisco_compendium.meme"
