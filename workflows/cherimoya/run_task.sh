#!/bin/bash
# run_task.sh
# Purpose: train and score one Cherimoya model -- one dataset, one fold -- in a
#   directory make_jobs.py prepared.
#
#   fit      `cherimoya fit -p fit.json`, run from the fold's directory, so the
#            model and its logs land there as cherimoya.fold_<k>.*. Cherimoya
#            evaluates the checkpoint on the validation and the test
#            chromosomes when training ends; the test performance file is the
#            last thing it writes, so it is the done marker. A directory with a
#            model but no test performance is an interrupted run, and its
#            outputs are cleared before training again.
#   compare  compare_chrombpnet.py -p compare.json: both models scored with
#            ChromBPNet's measures on ChromBPNet's test peaks for the fold. It
#            needs ChromBPNet's full model for that fold; until it exists the
#            stage exits 3 ("waiting"), not 1, and a later run picks it up.
#
# Input:  <workdir>/fit.json, <workdir>/compare.json (make_jobs.py)
# Output: <workdir>/cherimoya.fold_<k>.{torch,final.torch,log,detailed.log},
#         cherimoya.fold_<k>.{validation,test}.performance.tsv and .evaluate.json,
#         compare.tsv, compare_counts.npz; fit.time / compare.time (wall s, max RSS kB)
# Usage:  bash workflows/cherimoya/run_task.sh <workdir> [fit|compare|all]   (default all)
#         with CUDA_VISIBLE_DEVICES set to the GPU to use. run_box.sh calls it.
# Prerequisites: `pixi install -e cherimoya` in this repo; the ChromBPNet
#         pipeline's 00.0, 00.1 and 01.0 outputs for the dataset (04.0 for compare).

set -uo pipefail

workdir="${1:?usage: run_task.sh <workdir> [fit|compare|all]}"
stages="${2:-all}"
[[ "${stages}" == "all" ]] && stages="fit,compare"
repo_root="${RVI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${workdir}" || { echo "ERROR: no directory ${workdir}" >&2; exit 1; }
[[ -f fit.json && -f compare.json ]] || { echo "ERROR: ${workdir} has no fit.json/compare.json; run make_jobs.py" >&2; exit 1; }

fold="${workdir%/}"
fold="${fold##*/fold_}"
name="cherimoya.fold_${fold}"
pixi_run=( pixi run --frozen --manifest-path "${repo_root}/pixi.toml" -e cherimoya )

# Compiles and caches on the node, never on the shared filesystem: inductor's
# max-autotune benchmarks kernels in temp files and keeps an FX/Triton cache.
# run_box.sh points every task on a node at one cache, so only the first
# compile of each kernel autotunes.
export TMPDIR="${TMPDIR:-/tmp}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${TMPDIR}/inductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TMPDIR}/triton}"
threads="${CHERI_THREADS:-8}"
export OMP_NUM_THREADS="${threads}" MKL_NUM_THREADS="${threads}" OPENBLAS_NUM_THREADS="${threads}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-${threads}}"

# timed <label> <cmd...> -- run, writing "<seconds>\t<max RSS kB>" to <label>.time.
timed() {
    local label="$1"; shift
    local t0=${SECONDS} rc
    if [[ -x /usr/bin/time ]]; then
        /usr/bin/time -f '%M' -o "${label}.rss" "$@"; rc=$?
    else
        "$@"; rc=$?
    fi
    printf '%s\t%s\n' "$(( SECONDS - t0 ))" "$(tail -n 1 "${label}.rss" 2>/dev/null || echo NA)" > "${label}.time"
    rm -f "${label}.rss"
    return "${rc}"
}

rc=0
if [[ ",${stages}," == *",fit,"* ]]; then
    if [[ -s "${name}.torch" && -s "${name}.test.performance.tsv" ]]; then
        echo "[$(date)] ${workdir}: fit done, skipped"
    else
        rm -rf "${name}".* lightning_logs fit.time
        echo "[$(date)] ${workdir}: fit on GPU ${CUDA_VISIBLE_DEVICES:-?}"
        timed fit "${pixi_run[@]}" cherimoya fit -p fit.json
        rc=$?
        if (( rc != 0 )) || [[ ! -s "${name}.test.performance.tsv" ]]; then
            echo "ERROR: ${workdir}: cherimoya fit failed (exit ${rc})" >&2
            exit 1
        fi
    fi
fi

if [[ ",${stages}," == *",compare,"* ]]; then
    if [[ -s compare.tsv ]]; then
        echo "[$(date)] ${workdir}: compare done, skipped"
    elif [[ ! -s "${name}.torch" ]]; then
        echo "ERROR: ${workdir}: no ${name}.torch to compare" >&2
        exit 1
    else
        predictions="$(sed -n 's/^ *"chrombpnet_predictions": "\(.*\)",*$/\1/p' compare.json)"
        if [[ ! -s "${predictions}" ]]; then
            echo "[$(date)] ${workdir}: ChromBPNet's fold ${fold} model is not trained yet; compare waits"
            exit 3
        fi
        timed compare "${pixi_run[@]}" python "${repo_root}/workflows/cherimoya/compare_chrombpnet.py" -p compare.json
        rc=$?
        (( rc != 0 )) && { echo "ERROR: ${workdir}: compare failed (exit ${rc})" >&2; exit 1; }
    fi
fi
exit 0
