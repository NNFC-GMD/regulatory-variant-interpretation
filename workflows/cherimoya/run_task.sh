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
#   attribute  attribute_fast.py -p attribute.json (make_downstream.py):
#            DeepLIFT/SHAP on the counts head over every peak, computed by
#            cheri_dls and audited against the stock deep_lift_shap. Run only
#            when asked for (not part of `all`). It waits (exit 3) until the fit
#            is done, and skips a fold attr_status.py calls DONE. Before a run
#            it checks that the engine's files hash to the pin
#            (attr_status.py --write-pin), and removes the fold's earlier
#            attribution files. A failed audit (attribute_fast.py's exit 3)
#            exits 1 and keeps the outputs for inspection; a later run leaves
#            such a fold alone unless CHERI_ATTR_RERUN_FAILED=1. The outputs
#            are named by attribute.json, outside cherimoya.fold_<k>.*, so a
#            refit does not delete them: the refit makes them stale instead.
#   A fold directory another run is working on is skipped with exit 4.
#
# Knobs (environment):
#   CHERI_PIXI_MANIFEST   the pixi.toml whose cherimoya env runs everything
#                         (default: this checkout's; a worktree without
#                         installed environments points it at the main checkout)
#   CHERI_ATTRIBUTE_FAST  attribute_fast.py to run (default: this checkout's);
#                         its directory's cheri_dls.py and dls_refs.py are what
#                         the pin is checked against
#   CHERI_ATTR_ARGS       extra attribute_fast.py flags, e.g. "--seqs-per-step 4"
#   CHERI_ATTR_PIN        the engine pin (default: <workdir>/../../attribution_engine.json)
#   CHERI_ATTR_RERUN_FAILED  1: rerun a fold whose earlier attribution failed
#
# Input:  <workdir>/fit.json, <workdir>/compare.json (make_jobs.py);
#         <workdir>/attribute.json (make_downstream.py) for the attribute stage
# Output: <workdir>/cherimoya.fold_<k>.{torch,final.torch,log,detailed.log},
#         cherimoya.fold_<k>.{validation,test}.performance.tsv and .evaluate.json,
#         compare.tsv, compare_counts.npz; the files attribute.json names and
#         attribute_fast.py's sidecars; fit.time / compare.time / attribute.time
#         (wall s, max RSS kB)
# Usage:  bash workflows/cherimoya/run_task.sh <workdir> [fit|compare|attribute|all]
#         (default all = fit,compare; stages combine, e.g. fit,attribute)
#         with CUDA_VISIBLE_DEVICES set to the GPU to use. run_box.sh calls it.
# Prerequisites: `pixi install -e cherimoya` (here, or in CHERI_PIXI_MANIFEST's
#         checkout); the ChromBPNet pipeline's 00.0, 00.1 and 01.0 outputs for
#         the dataset (04.0 for compare).

set -uo pipefail

workdir="${1:?usage: run_task.sh <workdir> [fit|compare|attribute|all]}"
stages="${2:-all}"
[[ "${stages}" == "all" ]] && stages="fit,compare"
repo_root="${RVI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${workdir}" || { echo "ERROR: no directory ${workdir}" >&2; exit 1; }
[[ -f fit.json && -f compare.json ]] || { echo "ERROR: ${workdir} has no fit.json/compare.json; run make_jobs.py" >&2; exit 1; }

# One run per fold directory at a time: a second run would see no test
# performance file yet and clear the first one's outputs.
exec 9> .task.lock
flock -n 9 || { echo "[$(date)] ${workdir}: another run holds it, skipped"; exit 4; }

fold="${workdir%/}"
fold="${fold##*/fold_}"
name="cherimoya.fold_${fold}"
pixi_run=( pixi run --frozen --manifest-path "${CHERI_PIXI_MANIFEST:-${repo_root}/pixi.toml}" -e cherimoya )

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

if [[ ",${stages}," == *",attribute,"* ]]; then
    [[ -f attribute.json ]] || { echo "ERROR: ${workdir} has no attribute.json; run make_downstream.py" >&2; exit 1; }
    attribute_fast="${CHERI_ATTRIBUTE_FAST:-${repo_root}/workflows/cherimoya/attribute_fast.py}"
    attr_status=( "${pixi_run[@]}" python "${repo_root}/workflows/cherimoya/attr_status.py"
                  --pin "${CHERI_ATTR_PIN:-$(cd ../.. && pwd)/attribution_engine.json}" )
    if [[ ! -s "${name}.torch" || ! -s "${name}.test.performance.tsv" ]]; then
        echo "[$(date)] ${workdir}: the fit is not done; attribute waits"
        exit 3
    fi
    state_line="$("${attr_status[@]}" --check .)"
    state="$(cut -f3 <<< "${state_line}")"
    if [[ "${state}" == "done" ]]; then
        echo "[$(date)] ${workdir}: attribute done, skipped"
    elif [[ "${state}" == "failed" && "${CHERI_ATTR_RERUN_FAILED:-0}" != 1 ]]; then
        echo "ERROR: ${workdir}: an earlier attribution failed and is kept for inspection" \
             "(CHERI_ATTR_RERUN_FAILED=1 reruns it): $(cut -f4 <<< "${state_line}")" >&2
        exit 1
    elif [[ -z "${state}" ]]; then
        echo "ERROR: ${workdir}: attr_status.py --check gave no state" >&2
        exit 1
    else
        if ! "${attr_status[@]}" --check-engine "$(dirname "${attribute_fast}")"; then
            echo "ERROR: ${workdir}: refusing to attribute with an engine other than the pinned one" >&2
            exit 1
        fi
        mapfile -t attr_files < <("${attr_status[@]}" --files .)
        (( ${#attr_files[@]} > 0 )) || { echo "ERROR: ${workdir}: attr_status.py --files listed nothing" >&2; exit 1; }
        echo "[$(date)] ${workdir}: attribute (${state}: $(cut -f4 <<< "${state_line}")) on GPU ${CUDA_VISIBLE_DEVICES:-?}"
        rm -f -- "${attr_files[@]}" attribute.time
        for f in "${attr_files[@]}"; do mkdir -p "$(dirname "${f}")"; done
        read -r -a attr_args <<< "${CHERI_ATTR_ARGS:-}"
        # One compile thread: the process and its 3 reference workers are the 4 a slot has.
        TORCHINDUCTOR_COMPILE_THREADS=1 timed attribute "${pixi_run[@]}" python "${attribute_fast}" \
            -p attribute.json ${attr_args[@]+"${attr_args[@]}"}
        rc=$?
        if (( rc == 3 )); then
            echo "ERROR: ${workdir}: attribute_fast.py's audit failed; the outputs are kept for inspection" >&2
            exit 1
        elif (( rc != 0 )); then
            echo "ERROR: ${workdir}: attribute_fast.py failed (exit ${rc})" >&2
            exit 1
        fi
        if ! state_line="$("${attr_status[@]}" --check .)"; then
            echo "ERROR: ${workdir}: attribute_fast.py exited 0 but the fold is not DONE: $(cut -f3- <<< "${state_line}")" >&2
            exit 1
        fi
        echo "[$(date)] ${workdir}: attribute done"
    fi
fi
exit 0
