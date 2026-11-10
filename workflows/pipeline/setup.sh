#!/bin/bash
# setup.sh
# Purpose: what a machine needs to run this repository's pipeline, from this checkout alone (docs/install.md):
#   - the pixi environments, each from its lock file: this repository's default, cherimoya and chrombpnet, and the
#     vendored igvf_tf_chrombpnet's peaks, preprocess, modisco, motif-compendium and finemo
#     (vendor/igvf_tf_chrombpnet/pixi.toml and pixi.lock, the ones its steps were run with);
#   - with --references, the reference files every step reads (project.yaml's reference_root): the IGVF GRCh38
#     genome and its .fai, the chrom sizes, the ENCODE blacklist, the MotifCompendium database and the GENCODE TSSs,
#     downloaded and md5-checked by the vendored steps' own `src/cli.py download-references`.
#   Safe to rerun: an installed environment is checked against its lock, a reference already in place is kept.
# Input:  --references <dir>   where the references go (an absolute path; created if missing)
#         --with-engine        also cherimoya-030 (only for the AMSC project's frozen attribution engine)
#         --dry-run            print the commands and what is missing, change nothing
# Output: .pixi/envs/<env>/ here and in vendor/igvf_tf_chrombpnet/; <dir>/hg38/, <dir>/motif/, <dir>/metadata/
# Usage:  bash workflows/pipeline/setup.sh --references /data/mimir/references
# Needs:  pixi (https://pixi.sh; `curl -fsSL https://pixi.sh/install.sh | bash` puts it in ~/.pixi/bin) and the
#   network: conda-forge, bioconda, PyPI and GitHub for the environments (about 20 GB), the IGVF portal, ENCODE and
#   GitHub for the references (about 4 GB). It installs and downloads only, so it runs on a login node.

set -euo pipefail

usage="usage: setup.sh [--references <dir>] [--with-engine] [--dry-run]"
refs=""
engine=0
dry=0
while (( $# )); do
    case "$1" in
        --references) refs="${2:?${usage}}"; shift 2 ;;
        --with-engine) engine=1; shift ;;
        --dry-run) dry=1; shift ;;
        -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "${usage}" >&2; exit 2 ;;
    esac
done
[[ -z "${refs}" || "${refs}" == /* ]] || { echo "ERROR: --references must be an absolute path" >&2; exit 2; }

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
vendor="${repo_root}/vendor/igvf_tf_chrombpnet"
case ":${PATH}:" in
    *":${HOME}/.pixi/bin:"*) ;;
    *) export PATH="${HOME}/.pixi/bin:${PATH}" ;;
esac
command -v pixi > /dev/null || { echo "ERROR: no pixi: curl -fsSL https://pixi.sh/install.sh | bash" >&2; exit 1; }
# The CUDA builds (cherimoya's torch, finemo's) install on a machine without a GPU too; they need an NVIDIA driver
# >= 580 only to run.
export CONDA_OVERRIDE_CUDA="${CONDA_OVERRIDE_CUDA:-13.0}"

run() {
    printf '+'; printf ' %q' "$@"; printf '\n'
    (( dry )) || "$@"
}

rvi_envs=(default cherimoya chrombpnet)
(( engine )) && rvi_envs+=(cherimoya-030)
igvf_envs=(peaks preprocess modisco motif-compendium finemo)

for e in "${rvi_envs[@]}"; do
    run pixi install --locked --manifest-path "${repo_root}/pixi.toml" -e "${e}"
done
for e in "${igvf_envs[@]}"; do
    run pixi install --locked --manifest-path "${vendor}/pixi.toml" -e "${e}"
done
if [[ -n "${refs}" ]]; then
    (( dry )) || mkdir -p "${refs}"
    run pixi run --frozen --manifest-path "${vendor}/pixi.toml" -e preprocess \
        python "${vendor}/src/cli.py" download-references --reference-root "${refs}"
fi

# What is still missing (a dry run: what would be).
missing=0
for e in "${rvi_envs[@]}"; do
    [[ -d "${repo_root}/.pixi/envs/${e}" ]] || { echo "missing: env ${e} (${repo_root})"; missing=1; }
done
for e in "${igvf_envs[@]}"; do
    [[ -d "${vendor}/.pixi/envs/${e}" ]] || { echo "missing: env ${e} (${vendor})"; missing=1; }
done
if [[ -n "${refs}" ]]; then
    python="${repo_root}/.pixi/envs/default/bin/python"
    [[ -x "${python}" ]] || python=python3
    # The files project.py's preflight list names, from the vendored layout (standard library only).
    while read -r f; do
        [[ -s "${f}" ]] || { echo "missing: reference ${f}"; missing=1; }
    done < <("${python}" - "${repo_root}" "${refs}" <<'EOF'
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]) / "workflows" / "pipeline"))
import project

refs = project.reference_files(Path(sys.argv[2]))
for k in project.REFERENCE_FILES:
    print(refs[k])
print(refs["genome_fa"] + ".fai")
EOF
    )
fi
if (( missing )); then
    echo "setup.sh: not complete (above)$( (( dry )) && echo ': a dry run installs nothing')"
    (( dry )) && exit 0
    exit 1
fi
echo "setup.sh: every environment${refs:+ and reference file} in place"
