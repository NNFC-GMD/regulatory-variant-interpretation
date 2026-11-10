#!/bin/bash
# vendor_igvf.sh
# Purpose: copy the igvf_tf_chrombpnet files the driver runs into vendor/igvf_tf_chrombpnet/, byte for byte from one
#   commit, and write vendor/igvf_tf_chrombpnet/MANIFEST.tsv (the source commit, then path and sha256 of every file).
#   The preflight and tests/test_vendor.py refuse a vendored file that differs from its MANIFEST row, so the copy
#   changes only through this script. vendor/igvf_tf_chrombpnet/VENDORED.md says why these files and how to refresh.
# Input:  $1 = a clone of IGVF/igvf_tf_chrombpnet; $2 = the commit to copy (it must be on origin/chrombpnet2)
# Output: vendor/igvf_tf_chrombpnet/<the files below>, MANIFEST.tsv; every other file under it is removed, except
#   VENDORED.md and the untracked .pixi/
# Usage:  bash workflows/pipeline/vendor_igvf.sh <igvf clone> <commit>
#         then run the verification box (docs/pipeline.md, "Changing a pinned version") before merging
# Git and sha256sum only, so it runs on the login node.

set -euo pipefail

src="${1:?usage: vendor_igvf.sh <igvf clone> <commit>}"
rev="${2:?usage: vendor_igvf.sh <igvf clone> <commit>}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
dest="${repo_root}/vendor/igvf_tf_chrombpnet"

# The files the driver runs (run_project.sh, downstream.sh, motif_chain.sh), and nothing else: steps 00.0-02.0 and
# 06.0-11.0, what they source and call, the folds, and the manifest and lock of their environments.
FILES=(
    pixi.toml
    pixi.lock
    folds/fold_0.json
    folds/fold_1.json
    folds/fold_2.json
    folds/fold_3.json
    folds/fold_4.json
    lib/bash/common.sh
    lib/bash/config.sh
    lib/bash/finemo_compendium.sh
    lib/bash/references.sh
    lib/python/utils/__init__.py
    lib/python/utils/compendium_inputs.py
    lib/python/utils/compression.py
    lib/python/utils/config.py
    lib/python/utils/intervals.py
    lib/python/utils/log.py
    lib/python/utils/metadata.py
    lib/python/utils/palettes.py
    lib/python/utils/peakcall.py
    lib/python/utils/pileup.py
    lib/python/utils/plotting.py
    lib/python/utils/qc.py
    lib/python/utils/references.py
    lib/python/utils/shift.py
    src/average_contrib_scores.py
    src/cli.py
    src/contribs_to_bigwig.py
    src/emit_metadata.py
    src/motif_compendium.py
    workflows/SLURM/00.0.call_peaks.sh
    workflows/SLURM/00.0.prepare_signal.sh
    workflows/SLURM/00.1.preprocess_peaks.sh
    workflows/SLURM/01.0.preprocess_nonpeaks.sh
    workflows/SLURM/02.0.qc_training_data.sh
    workflows/SLURM/06.0.average_contrib_scores.sh
    workflows/SLURM/07.0.contribs_to_bigwig.sh
    workflows/SLURM/08.0.run_modisco.sh
    workflows/SLURM/09.0.cross_dataset_compendium.sh
    workflows/SLURM/10.0.run_finemo_unified.sh
    workflows/SLURM/11.0.postprocess_finemo.sh
    workflows/dcai/detect_shifts.py
    workflows/dcai/env.sh
    workflows/dcai/make_configs.py
    workflows/dcai/motif_chain.sh
    workflows/dcai/run_chain.sh
)

full="$(git -C "${src}" rev-parse --verify "${rev}^{commit}")"
git -C "${src}" merge-base --is-ancestor "${full}" origin/chrombpnet2 \
    || { echo "ERROR: ${full} is not on origin/chrombpnet2 of ${src}: push it first" >&2; exit 1; }

mkdir -p "${dest}"
# Remove the previous copy's files (a file dropped from FILES must not linger), keeping VENDORED.md and .pixi/.
find "${dest}" -mindepth 1 \( -path "${dest}/.pixi" -prune \) -o -type f ! -name VENDORED.md -print0 | xargs -0r rm -f
find "${dest}" -mindepth 1 -path "${dest}/.pixi" -prune -o -type d -empty -print0 | xargs -0r rmdir

manifest="${dest}/MANIFEST.tsv"
{
    echo "# source	https://github.com/IGVF/igvf_tf_chrombpnet"
    echo "# commit	${full}"
    echo "# copied by workflows/pipeline/vendor_igvf.sh; never edit these files here (VENDORED.md)"
    printf 'path\tsha256\n'
} > "${manifest}.partial"
for f in "${FILES[@]}"; do
    mode="$(git -C "${src}" ls-tree "${full}" -- "${f}" | awk '{ print $1 }')"
    [[ -n "${mode}" ]] || { echo "ERROR: ${f} is not in ${full}" >&2; exit 1; }
    mkdir -p "$(dirname "${dest}/${f}")"
    git -C "${src}" cat-file blob "${full}:${f}" > "${dest}/${f}"
    if [[ "${mode}" == 100755 ]]; then chmod 755 "${dest}/${f}"; else chmod 644 "${dest}/${f}"; fi
    printf '%s\t%s\n' "${f}" "$(sha256sum "${dest}/${f}" | awk '{ print $1 }')" >> "${manifest}.partial"
done
mv "${manifest}.partial" "${manifest}"
echo "vendored ${#FILES[@]} files from ${full} into ${dest}"
