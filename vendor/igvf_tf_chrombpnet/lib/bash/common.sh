#!/bin/bash
# shellcheck disable=SC2034  # everything here is consumed by the steps that source it
# lib/bash/common.sh
# Shared settings and helpers that do NOT depend on a dataset.
#
# Source this directly from cross-dataset steps (09, 04.2.qc_combined_boxplot,
# qc_datasets) that have no DATASET_DIR. Per-dataset steps source
# lib/bash/config.sh instead, which sources this file first.
#
# Requires REPO_ROOT to be set by the caller's bootstrap block.
#
# This file is sourced, never executed: it uses `return`, not `exit`, so a
# failure in an interactive shell does not kill the shell.

if [[ -z "${REPO_ROOT}" ]]; then
    echo "ERROR: REPO_ROOT is not set. Source this from a script with the standard bootstrap block." >&2
    return 1
fi

# ── Machine-specific values ───────────────────────────────────────────────────
# There is no second config file for these. They are the same for every dataset
# and must not be committed, so they are environment variables with defaults —
# export them in your shell profile, or in the conda env's activate.d. The
# defaults below are the Engreitz-lab Sherlock install.

# ── Software environments ─────────────────────────────────────────────────────
# Every environment is a pixi environment, on the cluster and on molab alike,
# named as `pixi:<manifest>#<environment>` and entered by activate_env below.
#
#   chrombpnet 2.x (Keras 3 / JAX, CUDA from its own wheels) comes from a
#   separate chrombpnet checkout at a pinned commit, installed from ITS lock
#   file -- the one the port was validated against -- rather than re-solved
#   here. Its manifest is pyproject.toml; chrombpnet has no pixi.toml. Once:
#
#     git clone https://github.com/NNFC-GMD/chrombpnet "$CHROMBPNET_REPO"
#     (cd "$CHROMBPNET_REPO" && git checkout --detach "$CHROMBPNET_REV")
#     CONDA_OVERRIDE_CUDA=13.0 pixi install --locked \
#         --manifest-path "$CHROMBPNET_REPO/pyproject.toml" -e cuda13
#
#   (CONDA_OVERRIDE_CUDA lets a login node without a GPU install it; cuda13
#   needs NVIDIA driver >= 580 at run time, `cuda12` is the fallback.)
#
#   preprocess, peaks, modisco, finemo, finemo-cu126 and motif-compendium are this repo's
#   own pixi.toml environments: `pixi install -e <name>` from the checkout.
#
# Any of these can instead be a plain conda prefix, which activate_env enters
# with `conda activate` through CONDA_INIT.
CHROMBPNET_REPO="${CHROMBPNET_REPO:-}"
# The commit of NNFC-GMD/chrombpnet the pipeline runs: branch
# stable-modern-gpu-support (the fork's default branch), i.e. PR
# kundajelab/chrombpnet#284 plus -bw on the training commands, `pipeline
# --skip-interpretation`, the lookup-table one-hot encoder, and a leaner, faster
# training loader (identical batches) with asynchronous batch callbacks, validated
# on ENCODE K562, and a bias hyperparameter pass that reads only the counts it
# uses (identical hyperparameters, about half the set-up time of a bias
# training). Same pyproject.toml / pixi.lock as the previous pins (c913b65,
# 7dfb285). activate_env warns when the checkout is elsewhere.
CHROMBPNET_REV="${CHROMBPNET_REV:-05f5748f7fae32c30eb3ba4451d97668856627d1}"
CHROMBPNET_PIXI_ENV="${CHROMBPNET_PIXI_ENV:-cuda13}"
chrombpnet_env="${CHROMBPNET_ENV:-pixi:${CHROMBPNET_REPO}/pyproject.toml#${CHROMBPNET_PIXI_ENV}}"
preprocess_env="${PREPROCESS_ENV:-pixi:${REPO_ROOT}/pixi.toml#preprocess}"
# preprocess + MACS3 from PR #756, built from source (00.0.call_peaks only).
peaks_env="${PEAKS_ENV:-pixi:${REPO_ROOT}/pixi.toml#peaks}"
# TF-MoDISco with parallel Leiden restarts (08.0 only; 03.3/04.5 use chrombpnet's).
modisco_env="${MODISCO_ENV:-pixi:${REPO_ROOT}/pixi.toml#modisco}"
finemo_env="${FINEMO_ENV:-pixi:${REPO_ROOT}/pixi.toml#finemo}"
motif_compendium_env="${MOTIF_COMPENDIUM_ENV:-pixi:${REPO_ROOT}/pixi.toml#motif-compendium}"
# Only for an environment given as a conda prefix. `:-` would resolve an
# explicitly EMPTY CONDA_INIT back to this default, and empty is meaningful:
# "no conda, the tools are already on PATH". `-` keeps the empty value.
CONDA_INIT="${CONDA_INIT-/home/groups/engreitz/Software/anaconda3/etc/profile.d/conda.sh}"

# Where dataset data/ and results/ live. Defaults to the checkout, which is
# fine for one dataset but not for a shared collaboration tree. DATASET_ROOT
# itself gets the default too, not just data_root: the tracked configs
# interpolate "${DATASET_ROOT}/<dataset>/...", which with it unset expanded to
# "/<dataset>/...".
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}}"
data_root="${DATASET_ROOT}"

# ── Bootstrap interpreter ─────────────────────────────────────────────────────
# bootstrap_python — echo a python >= 3.9 usable BEFORE any conda env exists.
#
# config.sh and references.sh both render their settings by running a
# stdlib-only module with this. They used to default to a bare `python3`, and
# on Sherlock that is /usr/bin/python3 = 3.6.8, which cannot PARSE either
# module: references.py uses a walrus (3.8+), config.py uses
# `from __future__ import annotations` (3.7+) and emit_metadata.py uses
# `tuple[str, str]` (3.9+). The result was that no per-dataset step ran at all
# in the documented default configuration -- it died in references.sh, before
# reaching any preflight check.
#
# Defined here, above the references.sh source, because that file needs it too.
bootstrap_python() {
    # `:-` matters: steps that use `set -u` (00.1, 01.0, 03.1) abort here on an
    # unset BOOTSTRAP_PYTHON, and because this runs from the metadata EXIT trap
    # the only symptom was "no python >= 3.9 found; run metadata not written" --
    # a silently missing provenance record on an otherwise successful step.
    if [[ -n "${BOOTSTRAP_PYTHON:-}" ]]; then
        echo "${BOOTSTRAP_PYTHON}"
        return 0
    fi
    local _c
    # PATH first (a module-loaded or already-active interpreter wins), then this
    # repo's own pixi environments. Those are 3.13 and are the reason the
    # pipeline works on Sherlock at all, where /usr/bin/python3 is 3.6.8 and
    # there is no newer python on the bare PATH -- without this fallback every
    # step needs BOOTSTRAP_PYTHON exported by hand.
    for _c in python3.13 python3.12 python3.11 python3.10 python3.9 python3 python \
              "${REPO_ROOT}/.pixi/envs/preprocess/bin/python" \
              "${REPO_ROOT}/.pixi/envs/default/bin/python"; do
        [[ -n "${_c}" ]] || continue
        command -v "${_c}" >/dev/null 2>&1 || continue
        # < 3.14 too: running a utils/ file as a script puts utils/ first on
        # sys.path, where utils/compression.py shadows 3.14's new stdlib
        # `compression` package.
        "${_c}" -c 'import sys; sys.exit(0 if (3, 9) <= sys.version_info[:2] < (3, 14) else 1)' 2>/dev/null || continue
        echo "${_c}"
        return 0
    done
    return 1
}

# ── Shared references ─────────────────────────────────────────────────────────
# Genome, chrom.sizes, blacklist and the motif DB all come from one place, shared
# with `cli.py download-references` so the writer and the readers cannot
# disagree. Override the root with REFERENCE_ROOT; see that file.
# shellcheck source=./references.sh
source "${REPO_ROOT}/lib/bash/references.sh" || return 1

# ── Algorithm parameters ──────────────────────────────────────────────────────
finemo_alpha="0.8" # Fi-NeMo hit-calling threshold (lower = more hits)
# Leiden clustering similarity cutoff; MOTIF_COMPENDIUM_THRESHOLD overrides it
# for a sensitivity rebuild under its own compendium tag.
motif_compendium_threshold="${MOTIF_COMPENDIUM_THRESHOLD:-0.95}"
# cpm_leiden alone, as MotifCompendium v1.0.16 ran. v1.0.19 changed cluster()'s
# default to cpm_leiden followed by k_centroids, which reassigns motifs without
# looking at the similarity threshold.
motif_compendium_algorithm="cpm_leiden"

# ── Repo layout ───────────────────────────────────────────────────────────────
src_dir="${REPO_ROOT}/src"          # atomic Python scripts the workflows call
folds_dir="${REPO_ROOT}/folds"      # cross-validation splits shipped with the repo

# Run-metadata output. config.sh repoints this at ${results_path}/metadata for
# per-dataset steps; cross-dataset steps keep this collaboration-root default.
metadata_dir="${METADATA_DIR:-${REPO_ROOT}/results/metadata}"

# ── Helpers ───────────────────────────────────────────────────────────────────
#
# Only the ones below exist, because only they are used. The steps write their
# own progress lines with `echo "[$(date)] ..."` and their own input checks with
# `[[ -f ... ]] || { echo ...; exit 1; }`. Adding a log()/require_file() here
# means converting those ~40 call sites in the same change, not leaving a second
# way to do it.

# load_bias_sweep — set bias_factors and bias_suffixes_sweep from 02.0's scan.
#
# 02.0 replays chrombpnet's own background-selection arithmetic over a fine
# grid and writes ${bias_scan_file}, which knows two things a hand-written list
# cannot: which factors leave ZERO non-peaks (those jobs cannot succeed), and
# which are DUPLICATES of each other, because counts are integers and the
# training set only changes when the cutoff crosses one.
#
# Shared by 03.0 and 03.1 deliberately. When only 03.0 read the scan, 03.1 went
# on reading the config, the two disagreed, and 03.1 "selected" a winner from
# whichever single model happened to overlap -- silently, because a selection
# from one candidate looks exactly like a selection from six.
#
# Falls back to the config when there is no scan, so a cluster run that never
# executed 02.0 is unaffected. BIAS_FACTORS_FROM_SCAN=0 forces the config.
load_bias_sweep() {
    if [[ "${BIAS_FACTORS_FROM_SCAN:-1}" != "1" || ! -s "${bias_scan_file}" ]]; then
        # Falling back to the config, which may legitimately not list a sweep:
        # a dataset that always runs 02.0 has no reason to. Say so, rather than
        # letting n_factors=0 divide by zero three lines later.
        if [[ ${#bias_factors[@]} -eq 0 ]]; then
            echo "ERROR: no bias sweep to run." >&2
            echo "  No scan at ${bias_scan_file}, and the dataset config sets no" >&2
            echo "  bias_factors. Either run 02.0 to generate the scan, or set" >&2
            echo "  bias_factors / bias_suffixes_sweep in the config." >&2
            exit 1
        fi
        echo "[$(date)] bias sweep from the dataset config (no scan at ${bias_scan_file})"
        return 0
    fi
    local _f _sfx _thr _nafter _nnon _distinct _verdict
    local _factors=() _suffixes=()
    while IFS=$'\t' read -r _f _sfx _thr _nafter _nnon _distinct _verdict; do
        [[ "${_f}" == "factor" ]] && continue        # header
        [[ "${_distinct}" == "True" ]] || continue   # one per DISTINCT training set
        _factors+=( "${_f}" )
        _suffixes+=( "${_sfx}" )
    done < "${bias_scan_file}"

    if [[ ${#_factors[@]} -eq 0 ]]; then
        echo "ERROR: ${bias_scan_file} lists no viable bias factor." >&2
        echo "  Every candidate leaves zero non-peaks after chrombpnet's outlier" >&2
        echo "  filter, so no training can succeed. Widen the grid or raise the" >&2
        echo "  outlier threshold, and re-run 02.0." >&2
        exit 1
    fi
    bias_factors=( "${_factors[@]}" )
    bias_suffixes_sweep=( "${_suffixes[@]}" )
    echo "[$(date)] bias sweep from ${bias_scan_file}"
    echo "           ${#bias_factors[@]} distinct factor(s): ${bias_factors[*]}"
}

# activate_env <env> — enter a software environment for the rest of the step.
# <env> is `pixi:<manifest>#<environment>` (every default above) or a conda
# prefix. Deliberately does NOT set -euo pipefail. Activation scripts are not
# written against `set -u`, so the pixi branch lifts it around the eval; 00.1
# and 01.0 set it before this call, 03.1 after.
activate_env() {
    local env_path="${1:?activate_env: missing env path}"
    if [[ "${env_path}" == pixi:* ]]; then
        activate_pixi_env "${env_path#pixi:}"
        return 0
    fi
    # CONDA_INIT set to the empty string means "there is no conda here; the
    # tools this step needs are already on PATH". This is opt-in: an UNSET or
    # merely missing CONDA_INIT stays a hard error below, because carrying on
    # in whatever environment happened to be active is precisely the failure
    # this function exists to prevent.
    if [[ -z "${CONDA_INIT}" ]]; then
        echo "[$(date)] activate_env: CONDA_INIT empty, using tools on PATH (not activating ${env_path})"
        return 0
    fi
    if [[ ! -f "${CONDA_INIT}" ]]; then
        echo "ERROR: conda init script not found: ${CONDA_INIT}" >&2
        echo "  Export CONDA_INIT to your conda's etc/profile.d/conda.sh." >&2
        exit 1
    fi
    # Check the env exists before asking conda for it: `conda activate` on a
    # missing prefix prints its own error but the step used to carry on in
    # whatever env was already active, and then died much later on an import
    # that looked like a missing dependency rather than a missing env. The
    # defaults here point into a shared account, so an env that was never
    # created on this cluster is the normal way this goes wrong.
    if [[ ! -x "${env_path}/bin/python" ]]; then
        echo "ERROR: conda env not found (no ${env_path}/bin/python)." >&2
        echo "  Or point the pipeline at another with CHROMBPNET_ENV /" >&2
        echo "  PREPROCESS_ENV / FINEMO_ENV / MOTIF_COMPENDIUM_ENV." >&2
        exit 1
    fi
    # shellcheck disable=SC1090  # path is a cluster location, not resolvable here
    source "${CONDA_INIT}"
    if ! conda activate "${env_path}"; then
        echo "ERROR: conda activate failed for ${env_path}" >&2
        exit 1
    fi
}

# activate_pixi_env <manifest>#<environment> — activate_env's pixi branch.
#
# `pixi shell-hook` prints the activation script; eval'ing it here puts the
# environment on PATH for the rest of the step, the way `conda activate` did,
# so each step still names its environment in one place: its activate_env
# line. --frozen installs from the lock file if the environment is missing and
# never re-solves, so a job cannot rewrite a lock file or pick up new versions.
activate_pixi_env() {
    local spec="${1:?activate_pixi_env: missing <manifest>#<environment>}"
    local manifest="${spec%#*}" environment="${spec##*#}"
    if [[ "${manifest}" == "${spec}" || -z "${environment}" ]]; then
        echo "ERROR: pixi environment must be pixi:<manifest>#<environment>, got pixi:${spec}" >&2
        exit 1
    fi
    if ! command -v pixi >/dev/null 2>&1; then
        echo "ERROR: pixi is not on PATH (needed for pixi:${spec})." >&2
        echo "  Install it: curl -fsSL https://pixi.sh/install.sh | bash" >&2
        exit 1
    fi
    if [[ ! -f "${manifest}" ]]; then
        echo "ERROR: no pixi manifest at '${manifest}' (for environment ${environment})." >&2
        if [[ "${manifest}" == */pyproject.toml ]]; then
            echo "  This is the chrombpnet checkout. Set CHROMBPNET_REPO to it, or create it" >&2
            echo "  as described under 'Software environments' in lib/bash/common.sh:" >&2
            echo "    git clone https://github.com/NNFC-GMD/chrombpnet <dir>" >&2
            echo "    (cd <dir> && git checkout --detach ${CHROMBPNET_REV})" >&2
            echo "    CONDA_OVERRIDE_CUDA=13.0 pixi install --locked --manifest-path <dir>/pyproject.toml -e ${CHROMBPNET_PIXI_ENV}" >&2
        fi
        exit 1
    fi
    # Anything on these would shadow the environment: a system CUDA or cuDNN on
    # LD_LIBRARY_PATH (a cluster module, a login-shell profile) breaks JAX's and
    # torch's own CUDA wheels at start-up, and a notebook's PYTHONPATH/venv leaks
    # its packages into the environment's python.
    unset LD_LIBRARY_PATH PYTHONPATH PYTHONHOME PYTHONSAFEPATH VIRTUAL_ENV
    export PYTHONNOUSERSITE=1
    local hook
    # CUDA environments declare a __cuda virtual package; the override lets a
    # CPU-only node (01.0, 03.1, 06-08) enter one. GPU steps check for a GPU
    # themselves (require_gpu, --device gpu), so this cannot hide a missing one.
    if ! hook="$(CONDA_OVERRIDE_CUDA="${CONDA_OVERRIDE_CUDA:-13.0}" \
            pixi shell-hook --manifest-path "${manifest}" -e "${environment}" \
            --frozen --no-completions --shell bash)"; then
        echo "ERROR: pixi could not activate ${environment} from ${manifest}" >&2
        exit 1
    fi
    # Conda packages' activate.d scripts (glib, cuda, ...) are not written
    # against `set -u`; lift it for the eval in steps that set it (00.1, 01.0).
    local had_nounset=0
    [[ $- == *u* ]] && had_nounset=1
    set +u
    eval "${hook}"
    (( had_nounset )) && set -u
    echo "[$(date)] activate_env: pixi ${environment} (${manifest})"
    # The chrombpnet checkout is a separate repo: say so when it is not at the
    # commit the pipeline was tested against. A warning, not a stop, so a newer
    # chrombpnet can be tried on purpose; the run metadata records the commit.
    if [[ -n "${CHROMBPNET_REPO}" && "${manifest}" == "${CHROMBPNET_REPO}/pyproject.toml" ]]; then
        local head
        # cd, not `git -C`: Sherlock's git 1.8.3.1 predates -C.
        head="$(cd "${CHROMBPNET_REPO}" && git rev-parse HEAD 2>/dev/null || true)"
        if [[ -n "${head}" && "${head}" != "${CHROMBPNET_REV}" ]]; then
            echo "WARNING: ${CHROMBPNET_REPO} is at ${head}, not CHROMBPNET_REV=${CHROMBPNET_REV}" >&2
        fi
    fi
}

# gpu_env — settings shared by every GPU step.
gpu_env() {
    # SLURM sets this to the allocated device; default to the first one.
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
}

# require_gpu <jax|torch> — stop now if the framework cannot see a GPU.
#
# Both fall back to the CPU with at most a warning, and a GPU job on the CPU
# runs for its whole time limit before anyone notices. chrombpnet's training
# commands take --device gpu for this; interpretation, prediction and Fi-NeMo
# have no such flag, so those steps call this after activate_env.
require_gpu() {
    local framework="${1:?require_gpu: jax or torch}" probe
    case "${framework}" in
        jax)   probe='import jax, sys; print("JAX devices:", jax.devices()); sys.exit(jax.default_backend() != "gpu")' ;;
        torch) probe='import torch, sys; print("torch", torch.__version__, "CUDA", torch.version.cuda); sys.exit(not torch.cuda.is_available())' ;;
        *) echo "ERROR: require_gpu: unknown framework '${framework}'" >&2; exit 1 ;;
    esac
    if ! python -c "${probe}"; then
        echo "ERROR: ${framework} sees no GPU on $(hostname)." >&2
        echo "  Check nvidia-smi and the driver: CUDA 13 builds need driver >= 580." >&2
        exit 1
    fi
}

# ── Run metadata ──────────────────────────────────────────────────────────────
#
# Every step emits one JSON record: inputs, outputs, md5s, tool versions, the
# git commit and a GitHub permalink to the step script. Many runs load into
# DuckDB as one table -- see workflows/README.md for the queries.
#
# In a step:
#
#     metadata_start "00.1.preprocess_peaks"   # names the step, installs the EXIT trap
#     metadata_inputs+=( "peaks=${peaks_file}" )
#     metadata_params+=( "fold=${fold}" )
#     ... do the work ...
#     metadata_outputs+=( "narrowpeak=${out}" )
#
# The trap fires on success, on failure and on SIGTERM (SLURM cancellation or
# soft preemption), so a failed run still records what it meant to produce.
# SIGKILL -- OOM, hard preemption -- leaves no record; that absence is the
# signal. Emission never changes the step's exit status.

metadata_step="${metadata_step:-}"
metadata_inputs=()
metadata_outputs=()
metadata_params=()
metadata_metrics=()
metadata_tools=()

# metadata_start <step-name> — record the start time and arrange for emission
# on exit. Taking the name as an argument (rather than presetting a variable)
# keeps the step scripts free of assignments shellcheck cannot see a use for.
metadata_start() {
    metadata_step="${1:?metadata_start: missing step name}"
    _metadata_started_at="$(date +%s)"
    _metadata_script="${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}"
    trap 'metadata_emit $?' EXIT
    trap 'exit 143' TERM        # turn SIGTERM into a normal exit so EXIT runs
}

# metadata_emit <exit-status> — write the record. Never fails the step.
metadata_emit() {
    local rc="${1:-0}"
    [[ -n "${metadata_step}" ]] || return 0
    [[ -n "${metadata_dir}" ]]  || return 0

    local args=(
        --step       "${metadata_step}"
        --out-dir    "${metadata_dir}"
        --script     "${_metadata_script}"
        --started-at "${_metadata_started_at:-}"
        --exit-status "${rc}"
    )
    [[ -n "${dataset:-}" ]] && args+=( --dataset "${dataset}" )

    local item
    for item in "${metadata_inputs[@]+"${metadata_inputs[@]}"}";  do args+=( --input  "${item}" ); done
    for item in "${metadata_outputs[@]+"${metadata_outputs[@]}"}"; do args+=( --output "${item}" ); done
    for item in "${metadata_params[@]+"${metadata_params[@]}"}";  do args+=( --param  "${item}" ); done
    for item in "${metadata_metrics[@]+"${metadata_metrics[@]}"}"; do args+=( --metric "${item}" ); done
    for item in "${metadata_tools[@]+"${metadata_tools[@]}"}";    do args+=( --tool   "${item}" ); done

    # Same interpreter rule as config.sh and references.sh. The EXIT trap fires
    # wherever the step died, and preflight_check is deliberately ordered BEFORE
    # activate_env -- so on the early-failure path there is no conda env yet and
    # a bare `python` is Sherlock's /usr/bin/python 2.7.5, which cannot parse
    # emit_metadata.py's `tuple[str, str]` annotations. That is precisely the
    # path whose record matters most: the one reporting the outputs a failed run
    # never wrote. Note /usr/bin/python3 (3.6.8) is no good either -- it rejects
    # `from __future__ import annotations`, which 3.7 introduced.
    local _meta_python
    _meta_python="$(bootstrap_python)" || {
        echo "[metadata] no python >= 3.9 found; run metadata not written" >&2
        return 0
    }
    "${_meta_python}" "${src_dir}/emit_metadata.py" "${args[@]}" || \
        echo "[metadata] could not write run metadata (step still exited ${rc})" >&2
    return 0
}


# ── ChromBPNet inputs ─────────────────────────────────────────────────────────

# set_signal_args — point chrombpnet's training commands at the bigwig 00.0
# prepared. Sets:
#   ${prepared_bigwig}       ${data_path}/signal/data_unstranded.bw
#   ${prepared_bigwig_json}  its sidecar, prepared_bigwig.json, beside it
#   ${signal_args[@]}        ( -bw "${prepared_bigwig}" )
#
# Training reads that bigwig whatever signal_type is. chrombpnet 2.x takes
# -ibw/-bw/--bigwig on `pipeline`, `train`, `bias pipeline` and `bias train`
# in place of -ibam/-ifrag/-itag: it uses the file where it is (no copy in
# auxiliary/) and skips its reads-to-bigwig conversion and shift estimation.
# That conversion is identical for every fold and bias factor and needs no
# GPU, which is why 00.0 does it once on CPU. There is no path left on which
# chrombpnet converts reads itself, so a missing or stale prepared bigwig is
# an error -- re-run 00.0 -- never a slower fallback.
#
# The raw signal (${signal_path}) is still read by the training steps:
# src/chrombpnet_train.py md5s it to confirm that the sidecar records this
# signal file, this md5 and this assay before chrombpnet starts.
signal_args=()
set_signal_args() {
    prepared_bigwig="${data_path}/signal/data_unstranded.bw"
    prepared_bigwig_json="${data_path}/signal/prepared_bigwig.json"
    signal_args=( -bw "${prepared_bigwig}" )
}

# ── Preconditions ─────────────────────────────────────────────────────────────
#
# A step declares what it needs and which step makes it; if anything is absent
# the run stops before doing work and prints the command to fix it.
#
#     require_input "${peaks_file}"     00.1.preprocess_peaks.sh
#     require_input "${negatives_file}" 01.0.preprocess_nonpeaks.sh
#     preflight_check
#
# All missing inputs are reported together, not one per re-run: submitting a
# GPU job to be told about one missing file, fixing it, and being told about
# the next is the failure mode this exists to avoid.

preflight_missing=()

# require_input <path> <producing-step> — record a required input.
require_input() {
    local path="${1:?require_input: missing path}" producer="${2:-}"
    [[ -e "${path}" ]] || preflight_missing+=( "${path}|${producer}" )
}

# preflight_check — report every missing input and exit 1, or return quietly.
preflight_check() {
    (( ${#preflight_missing[@]} )) || return 0

    local entry path producer producers=()
    echo "" >&2
    echo "ERROR: ${metadata_step:-this step} cannot run yet — ${#preflight_missing[@]} missing input(s):" >&2
    echo "" >&2
    for entry in "${preflight_missing[@]}"; do
        path="${entry%%|*}"
        producer="${entry##*|}"
        echo "  missing : ${path}" >&2
        if [[ -n "${producer}" ]]; then
            echo "  produced by : ${producer}" >&2
            # shellcheck disable=SC2076  # literal match is intended
            [[ " ${producers[*]-} " =~ " ${producer} " ]] || producers+=( "${producer}" )
        else
            echo "  produced by : (an external input — stage it yourself)" >&2
        fi
        echo "" >&2
    done

    if (( ${#producers[@]} )); then
        echo "Run first, in this order:" >&2
        for producer in "${producers[@]}"; do
            if [[ "${producer}" == *.sh ]]; then
                echo "  cd ${REPO_ROOT}/workflows/SLURM && DATASET=${DATASET:-<dataset>} sbatch ${producer}" >&2
            else
                echo "  ${producer}" >&2
            fi
        done
        echo "" >&2
        echo "Check overall progress with:  bash ${REPO_ROOT}/workflows/SLURM/status.sh" >&2
    fi
    exit 1
}
