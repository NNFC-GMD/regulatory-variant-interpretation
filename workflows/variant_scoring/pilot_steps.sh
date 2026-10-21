#!/bin/bash
# pilot_steps.sh
# Purpose: a pilot of variant scoring in one whole-node box (8 GPUs): test the code, build the windows, measure
#   throughput, run the checks V2-V5, predict every dataset x fold x model, then score, compare and analyse.
#   Count scores only (logFC, |logFC|, AAQ, empirical p against the shuffled null and, when its windows are built,
#   the genomic null), as scores.py computes them. Sized for millions of variants: at 3.2M variants x 10 datasets
#   x 5 folds, step 5 takes ~4.8-5.2 h on 8 H100s and the whole run ~6-6.5 h (estimates per step in pilot_box.sbatch).
#
# Steps, in order. Each logs to logs/<n>.<name>.log and adds a row to status.tsv (step, name, state, exit, seconds,
# started ET, note). A step whose prerequisite failed or was skipped is skipped itself, so a failure stops only what
# depends on it. A step left out of PILOT_STEPS counts as done (its outputs from an earlier run are used as they are).
#   1 tests    the repository's tests on the CPU: pytest in the default environment, the predictors' backend tests
#              in the cherimoya and the chrombpnet interpreters (tiny random models), then the suite once more in the
#              cherimoya interpreter for the tests that need torch (pytest's own pure-Python packages from the
#              default environment, put on PYTHONPATH from a directory of links; skipped if that cannot import them).
#              Only the first run gates step 2. The second adds the tangermeme shuffle tests, whose property
#              build_windows.py checks on every null row anyway; it gets its own row (tests_cherimoya) in status.tsv.
#              Nothing is written into the checkout: no bytecode (PYTHONDONTWRITEBYTECODE, and PYTHONPYCACHEPREFIX on
#              scratch), no .pytest_cache (-p no:cacheprovider), and the tests write under pytest's tmp_path, in
#              TMPDIR on scratch. With PILOT_GATE_TESTS=0 a failure here fails nothing: see that knob
#   2 windows  build_windows.py: the variant windows, the null (in the cherimoya interpreter: it needs tangermeme)
#              and each dataset's peak windows; then, when the genomic null's list is there (PILOT_GENOMIC_NULL),
#              its variant windows (windows/genomic_null.npz). That build gates nothing: a failure gets its own row
#              (windows_genomic_null) in status.tsv. A set that exists is kept (PILOT_REBUILD=1 rebuilds it); a set
#              whose windows/<name>.npz, .json or .dropped.tsv is a symbolic link (another root's windows, shared) is
#              always kept, PILOT_REBUILD or not, since build_windows.py writes the .json and .dropped.tsv through
#              the link, into the other root; a link whose .npz is missing is a failed build, not rebuilt
#                                                                                needs 1 (unless PILOT_GATE_TESTS=0)
#   3 probe    probe_throughput.py: passes/s per model at batch 64-1024 (Cherimoya also 2048), with 1 and 2
#              processes per GPU                                                 needs 2
#   4 checks   on one dataset and fold (PILOT_V_DATASET, PILOT_V_FOLD):
#              V3  cbp_full derived from the two parts against chrombpnet.h5 itself, PILOT_V3_N windows, at the
#                  default precision (reported) and at 'highest' with deterministic XLA ops (must agree within 1e-4,
#                  the fork's own parity bound)
#              V4  the noise floor on PILOT_V_N windows: each model run twice; reordered and re-batched; TF32
#                  against strict fp32; Cherimoya's megakernel against its grad path, and bf16 autocast
#              V2  OFF unless PILOT_V2=1, which needs the user's explicit OK: it executes VS main, downloaded code.
#                  When on: in the background on the last GPU, VS main in the TF-Keras container on PILOT_V_N
#                  variants of the bias-free model (the export capped at 15 min, VS at 45 min; TensorFlow with
#                  TF_FORCE_GPU_ALLOW_GROWTH, so it takes GPU memory as it needs it, uncapped); skipped cleanly when
#                  the clone, the container or apptainer is missing; a failed run gets its own row (4 v2 failed) in
#                  status.tsv
#                                                                                needs 2
#   5 predict  every dataset x fold, for each kind of model in PILOT_KINDS: Cherimoya, and ChromBPNet's corrected
#              and bias models in one process (the uncorrected model derived), on the variant, null and peak windows
#              and, when built, the genomic null's; PILOT_SLOTS_PER_GPU processes per GPU, each pinned to its own CPUs.
#              The tasks go longest first into one queue (every ChromBPNet task, ~33 min per fold at 3.2M variants on
#              an idle H100, then every Cherimoya one, ~8 min), and each slot takes the next task when it is free, so
#              the GPUs end within about one Cherimoya task of each other. A failed task runs once more at half the
#              batch (task <name>.retry), keeping the window sets it finished. A task that runs past
#              PILOT_TASK_MINUTES is stopped (timeout, exit 124) and counts as failed, so a hung process cannot keep
#              its GPU until the job's time limit.
#              While V2's VS run is still going, the last GPU's slots wait for it to end before they take a task:
#              TensorFlow's memory there is uncapped, and a slot's cap (PILOT_GPU_FRACTION) is a ceiling, not a
#              reservation, so the two could otherwise run each other out of GPU memory. V2 is a few minutes on
#              PILOT_V_N variants, so that GPU starts that much later
#                                                                                needs 2
#   6 score    V5 (forward vs reverse strand, from the saved strands; in the background), then scores.py once per
#              dataset, all at once, each on its own CPUs, against the shuffled null (scores/) and, when its windows
#              were built, the genomic null (scores_genomic_null/) at the same time; then compare.py per null
#              (compare/, compare_genomic_null/; one worker process per dataset); then null_report.py over both
#              nulls (analysis/), which gates nothing and gets its own row (analysis) in status.tsv; then V2's
#              comparison and the summary of every check. A failure against the genomic null gets its own row,
#              score_genomic_null. PILOT_FOLD_TSV=0 leaves out the per-fold TSVs (scores.py --no-fold-tsv; the
#              .npz tables that compare.py and null_report.py read are always written). Runs when step 5 ran, even if
#              some of its tasks failed: scores.py leaves out (and exits 1 for) a model without predictions,
#              compare.py a model without scores. scores.py scores the models of PILOT_KINDS (--models: cherimoya;
#              cbp_nobias, cbp_bias, cbp_full). Unless PILOT_KINDS has both kinds, compare.py and null_report.py, which
#              compare Cherimoya with ChromBPNet, do not run: their rows (compare, analysis) are `skipped` in
#              status.tsv. V5 and the summary run as usual; without chrombpnet, V2's comparison (it reads ChromBPNet's
#              corrected predictions) is left out, and V2's own row still records a failed VS run
#                                                                                needs 5 (to have run)
#   Checks never gate the predictions: a check outside its tolerance is state `outside_tolerance` (exit 3; 2 is
#   argparse's usage error, so it means failed).
#   Hold: with PILOT_HOLD_MINUTES > 0, when a row that holds is `failed` after the last step, the job keeps its node
#   for up to that many minutes, so the failed step can be fixed and run again inside it (srun --jobid <job>
#   --overlap ...; the box log prints the command) instead of in a new box after the queue. A clean run, or one whose
#   only failures hold nothing, ends at once. The rows that hold: those of the steps the later steps or the report
#   need (windows, checks, predict, score; tests only when they gate step 2, PILOT_GATE_TESTS=1) and the genomic
#   null's and the analysis's rows (windows_genomic_null, score_genomic_null, analysis), which are cheap to rerun
#   inside the hold. v2 (VS main, downloaded code), tests_cherimoya and tests that gate nothing do not hold, and
#   neither does `outside_tolerance`: it is a result to read, with nothing to rerun. The hold ends early when
#   box/<job>.<step>/release exists (touch it; this ends the job and any step still running in it). At its deadline
#   it lasts while a step started with srun --jobid <job> (a rerun) is still running, up to the job's time limit, so
#   a rerun that outlasts the N minutes is not cut off; the exit status is still that of this run's rows.
#
# Every process is pinned to its own block of the CPUs this job may use (Cpus_allowed_list) and has its GPU memory
# capped: torch by PYTORCH_CUDA_ALLOC_CONF=per_process_memory_fraction, JAX by XLA_CLIENT_MEM_FRACTION with
# XLA_PYTHON_CLIENT_PREALLOCATE=false (jaxlib honours the fraction without preallocation, and refuses it together
# with the deprecated XLA_PYTHON_CLIENT_MEM_FRACTION, which is unset). Compile caches, Python's bytecode and numba's
# cache live on node-local scratch.
#
# Knobs (environment; also read from ${PILOT_ENV:-${PILOT_ROOT}/pilot.env} when that file exists):
#   PILOT_ROOT              the pilot directory: manifest.json (make_manifest.py) and lists/ in it; every output
#                           goes under it (required)
#   PILOT_MANIFEST          the manifest (default ${PILOT_ROOT}/manifest.json)
#   PILOT_STEPS             steps to run (default 1,2,3,4,5,6)
#   PILOT_LIST              the variant list TSV (default: the one lists/*.tsv, not counting *.callsets.tsv and
#                           *.dropped.tsv)
#   PILOT_VARIANTS_NAME     window-set name of the variants (default variants)
#   PILOT_GENOMIC_NULL      the genomic null's list, a TSV in the variant list's columns (default
#                           ${PILOT_ROOT}/nulls/genomic_null.tsv). Set empty, the genomic null is off in every step.
#                           Not under lists/, where the variant list is the one TSV. Step 2 builds its windows when
#                           the list is there and not empty (window set genomic_null; kept once built, as are its
#                           predictions, so a changed list needs windows/genomic_null.* and
#                           preds/*/fold_*/*.genomic_null.* deleted first); steps 5 and 6 predict and score against
#                           windows/genomic_null.npz whenever it exists and the knob is not empty
#   RVI_MAIN                the checkout whose .pixi/envs hold `default` and `cherimoya` (default: this
#                           repository's main worktree)
#   CHROMBPNET_REPO         the chrombpnet fork checkout whose .pixi/envs/cuda13 is used
#   DEFAULT_PY, CHERIMOYA_PY, CHROMBPNET_PY   the interpreters, if not those (no `pixi run`: a dozen processes
#                           start at once, and these environments need no activation)
#   PILOT_GPUS              GPU indices (default: every GPU nvidia-smi lists)
#   PILOT_KINDS             the models steps 5 and 6 run: chrombpnet, cherimoya or both (default "chrombpnet
#                           cherimoya"; space- or comma-separated, in any order: ChromBPNet's tasks still go first).
#                           Step 5 queues only these kinds' tasks, and step 6 scores only their models; compare.py and
#                           null_report.py need both (step 6). Steps 3 and 4 run both whatever it says (the manifest
#                           holds both), as do step 1's backend tests (CHROMBPNET_PY is needed either way)
#   PILOT_GPU_CLAIM_DIR     empty (default): off. A directory (made if missing): every process run_logged starts on a
#                           GPU (step 4's checks, step 5's tasks) has <dir>/<gpu>.<pid>.<random>.claim while it runs,
#                           touched before and removed after it, for a GPU picker of another job step on the node that
#                           counts such files per GPU (a motif chain's claim_gpu), so that it sees these tasks. Step 3's
#                           probe processes and step 4's V2 run take none; a claim of a task killed with this script
#                           (SIGKILL) stays until the directory is removed (the box's scratch, at its end)
#   PILOT_SLOTS_PER_GPU     prediction processes per GPU (default 3); PILOT_GPU_FRACTION each one's memory share
#                           (default 0.3)
#   PILOT_CHERIMOYA_BATCH, PILOT_CHROMBPNET_BATCH   prediction batch sizes (default 1024, 512)
#   PILOT_NULL_N, PILOT_PEAKS_N, PILOT_SEED   null rows, AAQ peak windows per dataset, their seed (10000, 10000,
#                           1234)
#   PILOT_V_DATASET, PILOT_V_FOLD   where V2-V4 run (default: the manifest's first dataset, fold 0)
#   PILOT_V_N, PILOT_V3_N   rows for V2/V4 and for V3 (default 2000, 500)
#   PILOT_PROBE_SECONDS     timed seconds per probe batch size (default 20)
#   PILOT_GATE_TESTS        1 (default): failing tests in the default environment skip every later step; 0: carry on,
#                           and step 1's rows (tests, tests_cherimoya) do not count in the exit status: a failed test is
#                           in status.tsv and the log, but this script still exits 0 when every other row is fine
#   PILOT_REBUILD           1: rebuild window sets that exist (default: keep them); never a linked set (step 2)
#   PILOT_V2                1: run V2 (default 0, off). V2 executes VS main, downloaded code, so it is set only after
#                           the user says so. It also needs VS_REPO (a kundajelab/variant-scorer checkout, main
#                           @0e1e341), VS_SIF (the chrombpnet 1.x TF-Keras container), and apptainer on PATH or from
#                           `module load Apptainer` (after `module use ${VS_MODULEPATH}` when that is set)
#   PILOT_REPO_ROOT         the checkout whose workflows/variant_scoring/*.py run (default: the one holding this
#                           script). pilot_box.sbatch runs a copy of this script from <PILOT_ROOT>/logs and sets
#                           it to the submit directory.
#   PILOT_FOLD_TSV          1 (default): scores.py writes every per-fold table as .tsv.gz too; 0: --no-fold-tsv, only
#                           their .npz (at 3.2M variants each TSV is ~9 s of a scores.py process and ~150 MB, 20 per
#                           dataset and null; the summary TSVs are written either way)
#   PILOT_HOLD_MINUTES      0 (default): end when the steps end. N: after a failed row that holds, keep the node up to
#                           N minutes for a rerun inside it (see Hold above); set sbatch --time to cover the steps plus
#                           N and the rerun. Worth it only when someone will watch the box log
#   PILOT_TASK_MINUTES      0 (default): no limit. N: a prediction task (step 5) is stopped after N minutes and counts
#                           as failed (its retry has N minutes too); set it well above the longest task
#
# Output, under PILOT_ROOT: windows/, probe/, checks/ (v2/, v3/, v4/, the check JSONs, summary.tsv, v5.tsv), preds/
#   (<dataset>/fold_<k>/<model>.<set>.npz), scores/, compare/, analysis/; and box/<job>.<step>/: status.tsv,
#   tasks.tsv (every process the steps launch: name, GPU, CPUs, exit, seconds, start ET), resources.tsv (RAM, load,
#   GPU use every 30 s), logs/. With the genomic null: windows/genomic_null.npz, its predictions
#   (<model>.genomic_null.npz) in preds/, and scores_genomic_null/ and compare_genomic_null/, scores.py's and
#   compare.py's outputs against it (scores/ and compare/ stay against the shuffled null).
# Usage: mkdir -p <dir>/logs && PILOT_ROOT=<dir> sbatch --output=<dir>/logs/slurm_%j.out \
#            workflows/variant_scoring/pilot_box.sbatch    (the whole box; submit from the repository root)
#        or, inside a running allocation: PILOT_ROOT=<dir> srun --jobid <job> --overlap bash workflows/variant_scoring/pilot_steps.sh
#        (bash reads a script as it runs it: never edit this file in place while a box runs it; pilot_box.sbatch
#        runs a copy for that reason)
# Prerequisites: `pixi install` and `pixi install -e cherimoya` in RVI_MAIN; the fork's cuda13 environment;
#   make_manifest.py's manifest.json and a variant list (gnomad_list.py's or table_list.py's) in PILOT_ROOT.

set -uo pipefail

: "${PILOT_ROOT:?set PILOT_ROOT to the pilot directory (manifest.json, lists/; every output goes under it)}"
PILOT_ROOT="$(cd "${PILOT_ROOT}" && pwd)" || exit 1
PILOT_ENV="${PILOT_ENV:-${PILOT_ROOT}/pilot.env}"
if [[ -f "${PILOT_ENV}" ]]; then
    # shellcheck source=/dev/null
    source "${PILOT_ENV}"
fi

repo_root="${PILOT_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
[[ -f "${repo_root}/workflows/variant_scoring/build_windows.py" ]] || {
    echo "ERROR: ${repo_root} holds no workflows/variant_scoring (set PILOT_REPO_ROOT)" >&2; exit 1; }
vs="${repo_root}/workflows/variant_scoring"
nulls_dir="${repo_root}/workflows/nulls"
manifest="${PILOT_MANIFEST:-${PILOT_ROOT}/manifest.json}"
steps=",${PILOT_STEPS:-1,2,3,4,5,6},"
rvi_main="${RVI_MAIN:-$(git -C "${repo_root}" worktree list --porcelain 2>/dev/null | awk 'NR == 1 { print $2 }')}"
DEFAULT_PY="${DEFAULT_PY:-${rvi_main}/.pixi/envs/default/bin/python}"
CHERIMOYA_PY="${CHERIMOYA_PY:-${rvi_main}/.pixi/envs/cherimoya/bin/python}"
CHROMBPNET_PY="${CHROMBPNET_PY:-${CHROMBPNET_REPO:-<CHROMBPNET_REPO unset>}/.pixi/envs/cuda13/bin/python}"
# The one variable `pixi run` would add: the fork's [tool.pixi.activation.env] (its environments' conda activation
# scripts only set glib/libxml2 paths, and the cherimoya environment has none).
export KERAS_BACKEND=jax

variants_name="${PILOT_VARIANTS_NAME:-variants}"
gpu_slots="${PILOT_SLOTS_PER_GPU:-3}"
gpu_fraction="${PILOT_GPU_FRACTION:-0.3}"
cheri_batch="${PILOT_CHERIMOYA_BATCH:-1024}"
cbp_batch="${PILOT_CHROMBPNET_BATCH:-512}"
null_n="${PILOT_NULL_N:-10000}"
peaks_n="${PILOT_PEAKS_N:-10000}"
seed="${PILOT_SEED:-1234}"
v_n="${PILOT_V_N:-2000}"
v3_n="${PILOT_V3_N:-500}"
v_fold="${PILOT_V_FOLD:-0}"
probe_seconds="${PILOT_PROBE_SECONDS:-20}"
fold_tsv="${PILOT_FOLD_TSV:-1}"
hold_minutes="${PILOT_HOLD_MINUTES:-0}"
task_minutes="${PILOT_TASK_MINUTES:-0}"
[[ "${fold_tsv}" == [01] ]] || { echo "ERROR: PILOT_FOLD_TSV must be 0 or 1, got ${fold_tsv}" >&2; exit 1; }
[[ "${hold_minutes}" =~ ^[0-9]+$ ]] || {
    echo "ERROR: PILOT_HOLD_MINUTES must be whole minutes, got ${hold_minutes}" >&2; exit 1; }
[[ "${task_minutes}" =~ ^[0-9]+$ ]] || {
    echo "ERROR: PILOT_TASK_MINUTES must be whole minutes, got ${task_minutes}" >&2; exit 1; }
# Base 10: bash arithmetic would read 08 as a bad octal number and 010 as 8.
hold_minutes=$(( 10#${hold_minutes} ))
task_minutes=$(( 10#${task_minutes} ))
# PILOT_KINDS: the kinds of model steps 5 and 6 run, in the queue's order (ChromBPNet's longer tasks first), whatever
# order the knob lists them in.
kinds_text="${PILOT_KINDS:-chrombpnet cherimoya}"
read -r -a kinds_given <<< "${kinds_text//,/ }"
kinds=()
for kind in "${kinds_given[@]}"; do
    [[ "${kind}" == chrombpnet || "${kind}" == cherimoya ]] || {
        echo "ERROR: PILOT_KINDS takes chrombpnet and cherimoya (space- or comma-separated), got ${kind}" >&2; exit 1; }
done
for kind in chrombpnet cherimoya; do
    [[ " ${kinds_given[*]} " == *" ${kind} "* ]] && kinds+=( "${kind}" )
done
(( ${#kinds[@]} > 0 )) || { echo "ERROR: PILOT_KINDS names no kind of model (${PILOT_KINDS:-})" >&2; exit 1; }
# has_kind <kind> -- PILOT_KINDS has it.
has_kind() { [[ " ${kinds[*]} " == *" $1 "* ]]; }
# both_kinds -- PILOT_KINDS has Cherimoya and ChromBPNet, which compare.py and null_report.py compare.
both_kinds() { has_kind chrombpnet && has_kind cherimoya; }
# The models step 6 scores, in scores.py's MODELS order: Cherimoya; ChromBPNet's corrected, bias and uncorrected
# (derived from the other two) models.
score_models=()
if has_kind cherimoya; then score_models+=( cherimoya ); fi
if has_kind chrombpnet; then score_models+=( cbp_nobias cbp_bias cbp_full ); fi
# PILOT_GPU_CLAIM_DIR: a claim file per GPU task, for another job step's GPU picker (run_logged); empty = off.
gpu_claim_dir="${PILOT_GPU_CLAIM_DIR:-}"
if [[ -n "${gpu_claim_dir}" ]]; then
    mkdir -p "${gpu_claim_dir}" || { echo "ERROR: cannot make PILOT_GPU_CLAIM_DIR ${gpu_claim_dir}" >&2; exit 1; }
fi
# The genomic null's list. `-`, not `:-`: an empty PILOT_GENOMIC_NULL turns the null off; only an unset one takes the
# default. step_windows builds it when the file is there and not empty.
genomic_null_list="${PILOT_GENOMIC_NULL-${PILOT_ROOT}/nulls/genomic_null.tsv}"

windows="${PILOT_ROOT}/windows"
preds="${PILOT_ROOT}/preds"
checks="${PILOT_ROOT}/checks"
# use_genomic_null -- steps 5 and 6 predict and score the genomic null: PILOT_GENOMIC_NULL is not set empty and step 2
# (of this box or an earlier one) built windows/genomic_null.npz.
use_genomic_null() { [[ -n "${genomic_null_list}" && -s "${windows}/genomic_null.npz" ]]; }
box_id="${SLURM_JOB_ID:-local}.${SLURM_STEP_ID:-$$}"
box_dir="${PILOT_ROOT}/box/${box_id}"
logs="${box_dir}/logs"
status="${box_dir}/status.tsv"
tasks_tsv="${box_dir}/tasks.tsv"
mkdir -p "${logs}/windows" "${logs}/checks" "${logs}/predict" "${logs}/score" "${windows}" "${preds}" "${checks}"

et() { TZ=America/New_York date '+%F %T %Z'; }
say() { echo "[$(et)] $*"; }

# ------------------------------------------------------------------------------------------------ CPUs and GPUs
# The CPUs this process may use (an srun step can be given fewer than the node has).
allowed=()
cpus_allowed_list="$(awk '/^Cpus_allowed_list:/ { print $2 }' /proc/self/status)"
IFS=, read -r -a ranges <<< "${cpus_allowed_list}"
for r in "${ranges[@]}"; do
    if [[ "${r}" == *-* ]]; then mapfile -t -O "${#allowed[@]}" allowed < <(seq "${r%-*}" "${r#*-}"); else allowed+=( "${r}" ); fi
done
# cpu_block <index> <size> -- the index-th block of <size> allowed CPUs, comma-separated (wraps around).
cpu_block() {
    local first=$(( ($1 * $2) % ${#allowed[@]} ))
    (( first + $2 > ${#allowed[@]} )) && first=0
    (IFS=,; echo "${allowed[*]:${first}:$2}")
}
if [[ -n "${PILOT_GPUS:-}" ]]; then
    IFS=, read -r -a gpus <<< "${PILOT_GPUS}"
else
    mapfile -t gpus < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)
fi

# ------------------------------------------------------------------------------------------------ scratch
# Node-local temp files and compile caches (Triton autotuning, the JAX compilation cache), shared by this job's
# processes. A killed job skips the EXIT trap, so clear what earlier jobs left first, but not this job's.
scratch_root=/dev/shm
[[ -d /raid && -w /raid ]] && scratch_root=/raid
for old in "${scratch_root}/${USER}/vs_pilot."*; do
    [[ -d "${old}" ]] || continue
    old_job="${old##*/vs_pilot.}"
    [[ "${old_job%%.*}" != "${SLURM_JOB_ID:-local}" ]] && rm -rf "${old}"
done
scratch="${scratch_root}/${USER}/vs_pilot.${box_id}"
export TMPDIR="${scratch}/tmp" TRITON_CACHE_DIR="${scratch}/triton" TORCHINDUCTOR_CACHE_DIR="${scratch}/inductor"
export JAX_COMPILATION_CACHE_DIR="${scratch}/jax"
# Python's bytecode and numba's cache (tangermeme's shuffle is numba cache=True) go to scratch too, not into this
# checkout, the fork checkout or the environments' site-packages.
export PYTHONPYCACHEPREFIX="${scratch}/pycache" NUMBA_CACHE_DIR="${scratch}/numba"
mkdir -p "${TMPDIR}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${JAX_COMPILATION_CACHE_DIR}" \
    "${PYTHONPYCACHEPREFIX}" "${NUMBA_CACHE_DIR}"
sampler_pid=""
trap '[[ -n "${sampler_pid}" ]] && kill "${sampler_pid}" 2>/dev/null; rm -rf "${scratch}"' EXIT

# ------------------------------------------------------------------------------------------------ helpers
manifest_cmd() { "${DEFAULT_PY}" "${vs}/read_manifest.py" "$@" "${manifest}"; }

# The variant list: PILOT_LIST, or the one TSV in lists/ that is not a .callsets.tsv or .dropped.tsv.
variant_list() {
    if [[ -n "${PILOT_LIST:-}" ]]; then echo "${PILOT_LIST}"; return 0; fi
    local found=()
    mapfile -t found < <(find "${PILOT_ROOT}/lists" -maxdepth 1 -name '*.tsv' ! -name '*.callsets.tsv' \
                         ! -name '*.dropped.tsv' 2>/dev/null | sort)
    if (( ${#found[@]} != 1 )); then
        echo "ERROR: set PILOT_LIST: ${#found[@]} candidate lists in ${PILOT_ROOT}/lists (${found[*]:-none})" >&2
        return 1
    fi
    echo "${found[0]}"
}

# The manifest row of the dataset and fold the checks run on: dataset fold cherimoya nobias bias full test_chroms
v_row() { "${DEFAULT_PY}" "${vs}/read_manifest.py" tasks "${manifest}" --datasets "${v_dataset}" --folds "${v_fold}"; }

# run_logged <name> <gpu|-> <cpus> <gpu fraction> <cmd...> -- run a command pinned to <cpus> with its GPU (- for none)
# and memory caps; log to logs/<name>.log, exit status to logs/<name>.exit, one row in tasks.tsv. Returns its status.
# With PILOT_GPU_CLAIM_DIR, a GPU command has its claim file there while it runs.
run_logged() {
    local name="$1" gpu="$2" cpus="$3" fraction="$4" started t0 rc n claim=""
    shift 4
    n="$(awk -F, '{ print NF }' <<< "${cpus}")"
    started="$(et)"
    t0=${SECONDS}
    # <gpu>.<pid>.<random>.claim: the GPU index first, which is how the other picker reads it (the part before the
    # first dot); this shell's PID and a random number keep two tasks' claims on one GPU apart.
    if [[ "${gpu}" != "-" && -n "${gpu_claim_dir}" ]]; then
        claim="${gpu_claim_dir}/${gpu}.${BASHPID}.${RANDOM}.claim"
        if ! touch "${claim}"; then
            say "WARNING: ${name}: no GPU claim (cannot write ${claim}); it runs unclaimed"
            claim=""
        fi
    fi
    (
        if [[ "${gpu}" == "-" ]]; then
            export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu
        else
            export CUDA_VISIBLE_DEVICES="${gpu}"
            export PYTORCH_CUDA_ALLOC_CONF="per_process_memory_fraction:${fraction}"
            export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_CLIENT_MEM_FRACTION="${fraction}"
            unset XLA_PYTHON_CLIENT_MEM_FRACTION
        fi
        export OMP_NUM_THREADS="${n}" MKL_NUM_THREADS="${n}" OPENBLAS_NUM_THREADS="${n}"
        echo "[${started}] GPU ${gpu}, CPUs ${cpus}: $*"
        exec taskset -c "${cpus}" "$@"
    ) > "${logs}/${name}.log" 2>&1
    rc=$?
    if [[ -n "${claim}" ]]; then rm -f "${claim}"; fi
    echo "${rc}" > "${logs}/${name}.exit"
    (
        flock -x 8
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${name}" "${gpu}" "${cpus}" "${rc}" "$(( SECONDS - t0 ))" "${started}" >> "${tasks_tsv}"
    ) 8> "${tasks_tsv}.lock"
    return "${rc}"
}

# launch <same arguments as run_logged> -- run_logged in the background; its PID is left in launched_pid.
launch() {
    run_logged "$@" &
    launched_pid=$!
}

# wait_all <pid>... -- wait for every PID; 1 if any failed, else 0.
wait_all() {
    local pid failed=0
    for pid in "$@"; do
        wait "${pid}" || failed=1
    done
    return "${failed}"
}

# pinned <cmd...> -- a command of the score step, pinned to the first 32 allowed CPUs like every other process.
pinned() { taskset -c "$(cpu_block 0 32)" "$@"; }

# The exit status of a check that ran but is outside its tolerance (vchecks.py's OUTSIDE_TOLERANCE). Not 2, which
# argparse returns for a usage error.
outside=3

# worst <status>... -- the status a group of checks reports: 1 if any could not run, else 3 if any was outside its
# tolerance, else 0.
worst() {
    local s out=0
    for s in "$@"; do
        if (( s != 0 && s != outside )); then out=1; elif (( s == outside && out == 0 )); then out=${outside}; fi
    done
    return "${out}"
}

declare -A state
record() {
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" "${step_started}" "$6" >> "${status}"
}

# run_step <n> <name> <prerequisite steps, space-separated> -- run step_<name>, record it, set state[<n>]. A step
# whose prerequisite failed or was skipped is skipped; a prerequisite written <d>~ only has to have run (a failed
# step d still lets this step use what it made; a skipped one does not).
run_step() {
    local n="$1" name="$2" deps="$3" d s rc t0 note=""
    if [[ "${steps}" != *",${n},"* ]]; then
        state[${n}]=not_run
        return 0
    fi
    step_started="$(et)"
    for d in ${deps}; do
        s="${state[${d%\~}]:-not_run}"
        if [[ "${s}" == skipped || ( "${s}" == failed && "${d}" != *\~ ) ]]; then
            say "step ${n} ${name}: skipped, step ${d%\~} did not finish"
            record "${n}" "${name}" skipped - 0 "step ${d%\~} did not finish"
            state[${n}]=skipped
            return 0
        fi
    done
    say "step ${n} ${name}: started; log ${logs}/${n}.${name}.log"
    t0=${SECONDS}
    "step_${name}" > "${logs}/${n}.${name}.log" 2>&1
    rc=$?
    case "${rc}" in
        0) state[${n}]="done" ;;
        "${outside}") state[${n}]=outside_tolerance; note="a check is outside its tolerance" ;;
        *) state[${n}]=failed; note="see ${logs}/${n}.${name}.log" ;;
    esac
    record "${n}" "${name}" "${state[${n}]}" "${rc}" "$(( SECONDS - t0 ))" "${note}"
    say "step ${n} ${name}: ${state[${n}]} (exit ${rc}) in $(( SECONDS - t0 )) s"
}

# ------------------------------------------------------------------------------------------------ 1 tests
step_tests() {
    local cpus rc=0 purelib shim pkg t0
    cpus="$(cpu_block 0 32)"
    say "pytest in the default environment; backend tests in ${CHERIMOYA_PY} and ${CHROMBPNET_PY}"
    env CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=32 \
        VS_CHERIMOYA_PYTHON="${CHERIMOYA_PY}" VS_CHROMBPNET_PYTHON="${CHROMBPNET_PY}" \
        taskset -c "${cpus}" "${DEFAULT_PY}" -m pytest -q -p no:cacheprovider -rs "${repo_root}/tests" || rc=1
    # The suite again in the cherimoya interpreter (torch, tangermeme), with pytest's pure-Python packages from the
    # default environment for those it lacks. It gates nothing (step 1's status is the run above's): it adds only the
    # tangermeme shuffle tests, and build_windows.py checks every null row's shuffle as it builds it. A failure gets
    # its own row in status.tsv.
    purelib="$("${DEFAULT_PY}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    shim="${box_dir}/pytest_shim"
    mkdir -p "${shim}"
    for pkg in pytest _pytest pluggy iniconfig pygments; do
        if ! "${CHERIMOYA_PY}" -c "import importlib.util, sys; sys.exit(importlib.util.find_spec('${pkg}') is None)"; then
            ln -sfn "${purelib}/${pkg}" "${shim}/${pkg}"
        fi
    done
    [[ -e "${purelib}/py.py" ]] && ln -sfn "${purelib}/py.py" "${shim}/py.py"
    if env PYTHONPATH="${shim}" PYTHONDONTWRITEBYTECODE=1 "${CHERIMOYA_PY}" -c 'import pytest' 2>/dev/null; then
        say "pytest in the cherimoya interpreter (pytest from ${shim})"
        t0=${SECONDS}
        if ! env CUDA_VISIBLE_DEVICES="" PYTHONPATH="${shim}" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=32 \
                taskset -c "${cpus}" "${CHERIMOYA_PY}" -m pytest -q -p no:cacheprovider -rs "${repo_root}/tests"; then
            say "the suite failed in the cherimoya interpreter: recorded as tests_cherimoya; it gates nothing"
            record 1 tests_cherimoya failed 1 "$(( SECONDS - t0 ))" "see ${logs}/1.tests.log; gates nothing"
        fi
    else
        say "the cherimoya interpreter cannot import pytest from ${shim}; its torch-only tests were skipped above"
    fi
    return "${rc}"
}

# ------------------------------------------------------------------------------------------------ 2 windows
# build_set <name> <python> <build_windows.py arguments...> -- build windows/<name>.npz in the background, unless
# it exists; uses step_windows's pids, slot, fasta. Returns 1 for a linked set without its .npz, else 0 (the build's
# own status is its PID's, in pids).
build_set() {
    local name="$1" py="$2" out
    shift 2
    # A linked set (another root's windows) is kept whatever PILOT_REBUILD says: build_windows.py writes <name>.json
    # and <name>.dropped.tsv with open(), which follows a link and would overwrite the other root's files (only the
    # .npz is renamed into place, which replaces a link instead).
    for out in "${windows}/${name}".{npz,json,dropped.tsv}; do
        [[ -L "${out}" ]] || continue
        if [[ -s "${windows}/${name}.npz" ]]; then
            say "windows ${name}: ${out} is a link (to $(readlink "${out}")); kept, never rebuilt here"
            return 0
        fi
        say "ERROR: windows ${name}: ${out} is a link (to $(readlink "${out}")), and ${windows}/${name}.npz is" \
            "missing or empty; not rebuilt through the link (fix or remove the links)"
        return 1
    done
    if [[ -s "${windows}/${name}.npz" && "${PILOT_REBUILD:-0}" != 1 ]]; then
        say "windows ${name}: ${windows}/${name}.npz exists; kept"
        return 0
    fi
    launch "windows/${name}" - "$(cpu_block "${slot}" 8)" 0 "${py}" "${vs}/build_windows.py" "$@" \
        --fasta "${fasta}" --out "${windows}/${name}.npz"
    pids+=( "${launched_pid}" )
    slot=$(( slot + 1 ))
}

step_windows() {
    local fasta list ds peaks slot=0 pids=() rc=0 t0 gn_rc
    fasta="$(manifest_cmd genome)" || return 1
    list="$(variant_list)" || return 1
    say "variant list ${list}; genome ${fasta}"
    build_set "${variants_name}" "${DEFAULT_PY}" --list "${list}" || rc=1
    build_set null "${CHERIMOYA_PY}" --null --list "${list}" --n "${null_n}" --seed "${seed}" || rc=1
    while IFS=$'\t' read -r ds peaks; do
        build_set "peaks.${ds}" "${DEFAULT_PY}" --peaks "${peaks}" --n "${peaks_n}" --seed "${seed}" || rc=1
    done < <(manifest_cmd peaks)
    wait_all "${pids[@]}" || rc=1
    # The genomic null: its list's variant windows, built as the variants' are (no shuffle, so the default
    # interpreter), after the sets the later steps need. It gates nothing: a failed build gets its own row
    # (windows_genomic_null) in status.tsv, and leaves no npz (build_windows.py writes it last), so steps 5 and 6
    # leave the genomic null out.
    if [[ -s "${genomic_null_list}" ]]; then
        say "genomic null list ${genomic_null_list}"
        if [[ -s "${windows}/genomic_null.npz" && "${PILOT_REBUILD:-0}" != 1 ]] \
                && [[ "${genomic_null_list}" -nt "${windows}/genomic_null.npz" ]]; then
            say "WARNING: genomic null: the list is newer than windows/genomic_null.npz, which is kept, as are its" \
                "predictions; if the list changed, delete windows/genomic_null.* and preds/*/fold_*/*.genomic_null.*"
        fi
        pids=()
        t0=${SECONDS}
        gn_rc=0
        build_set genomic_null "${DEFAULT_PY}" --list "${genomic_null_list}" || gn_rc=1
        wait_all "${pids[@]}" || gn_rc=1
        if (( gn_rc != 0 )); then
            say "genomic null: the build failed; recorded as windows_genomic_null; it gates nothing"
            record 2 windows_genomic_null failed 1 "$(( SECONDS - t0 ))" \
                "see ${logs}/windows/genomic_null.log; gates nothing"
        fi
    elif [[ -n "${genomic_null_list}" ]]; then
        say "genomic null: no list (${genomic_null_list} is missing or empty)"
    else
        say "genomic null: no list (PILOT_GENOMIC_NULL is empty)"
    fi
    return "${rc}"
}

# ------------------------------------------------------------------------------------------------ 3 probe
step_probe() {
    local cheri nobias bias per
    IFS=$'\t' read -r _ _ cheri nobias bias _ _ <<< "$(v_row)"
    per=$(( ${#allowed[@]} / (${#gpus[@]} * 2) ))
    (( per > 12 )) && per=12
    "${DEFAULT_PY}" "${vs}/probe_throughput.py" --out "${PILOT_ROOT}/probe/probe.json" \
        --cherimoya-python "${CHERIMOYA_PY}" --chrombpnet-python "${CHROMBPNET_PY}" \
        --cherimoya-model "${cheri}" --nobias-model "${nobias}" --bias-model "${bias}" \
        --gpus "$(IFS=,; echo "${gpus[*]}")" --cpus-per-proc "${per}" --seconds "${probe_seconds}"
}

# ------------------------------------------------------------------------------------------------ 4 checks
# pair <check> <slug> <label> <a.npz> <b.npz> [tol] -- vchecks.py pair, into checks/<check>.<slug>.json.
pair() {
    local tol=()
    [[ -n "${6:-}" ]] && tol=( --tol "$6" )
    "${DEFAULT_PY}" "${vs}/vchecks.py" pair --check "$1" --label "$3" --a "$4" --b "$5" "${tol[@]}" \
        --out "${checks}/$1.$2.json"
}

# v2_run -- V2's VS main run (best effort; off unless PILOT_V2=1). Writes checks/v2/vs/nobias.variant_scores.tsv, or
# checks/v2/skipped.txt with the reason. VS is kundajelab/variant-scorer main @0e1e341; it runs unchanged in the
# chrombpnet 1.x container. Every command that could stall is under `timeout`, so step 6's wait for it is bounded.
v2_run() {
    local out="${checks}/v2" vsrc="${VS_REPO:-}/src" sif="${VS_SIF:-}" fasta list nobias stub="" cpus py_bin
    cpus="$(cpu_block 6 16)"
    rm -f "${out}/skipped.txt"
    mkdir -p "${out}/vs"
    skip() { echo "$*" > "${out}/skipped.txt"; say "V2 skipped: $*"; }
    if [[ "${PILOT_V2:-0}" != 1 ]]; then
        skip "V2 is opt-in (PILOT_V2=1): it runs VS main, downloaded code, and needs the user's explicit OK"
        return 0
    fi
    [[ -n "${VS_REPO:-}" && -f "${vsrc}/variant_scoring.py" ]] || { skip "no VS checkout (VS_REPO=${VS_REPO:-unset})"; return 0; }
    [[ -n "${sif}" && -f "${sif}" ]] || { skip "no container (VS_SIF=${sif:-unset})"; return 0; }
    if ! command -v apptainer > /dev/null 2>&1; then
        if ! type module > /dev/null 2>&1 && [[ -f /etc/profile.d/z00_lmod.sh ]]; then
            # shellcheck source=/dev/null
            source /etc/profile.d/z00_lmod.sh
        fi
        if type module > /dev/null 2>&1; then
            [[ -n "${VS_MODULEPATH:-}" ]] && module use "${VS_MODULEPATH}" > /dev/null 2>&1
            module load Apptainer > /dev/null 2>&1
        fi
    fi
    command -v apptainer > /dev/null 2>&1 || { skip "no apptainer (module load Apptainer did not provide it)"; return 0; }
    fasta="$(manifest_cmd genome)" || return 1
    list="$(variant_list)" || return 1
    IFS=$'\t' read -r _ _ _ nobias _ _ _ <<< "$(v_row)"
    if ! timeout -k 30 600 apptainer exec --cleanenv "${sif}" python -c \
            'import tensorflow, deeplift, pyfaidx, h5py, pandas, scipy, tqdm' > /dev/null 2>&1; then
        skip "the container cannot import VS's requirements (tensorflow, deeplift, pyfaidx, h5py, pandas, scipy, tqdm)"
        return 0
    fi
    # The bias-free model in TF-Keras 2.x's layout: the Keras 3 file cannot be read there (fork
    # chrombpnet/helpers/postprocessing/README.md). A bias-free model has no Lambda layer, so the default export works.
    py_bin="$(dirname "${CHROMBPNET_PY}")"
    if [[ ! -s "${out}/chrombpnet_nobias.legacy.h5" ]]; then
        if ! CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu taskset -c "${cpus}" timeout -k 60 900 \
                "${py_bin}/chrombpnet" export -m "${nobias}" -o "${out}/chrombpnet_nobias.legacy.h5"; then
            skip "chrombpnet export failed or ran past 15 min"
            return 1
        fi
    fi
    "${DEFAULT_PY}" "${vs}/vchecks.py" vs-list --list "${list}" --windows "${windows}/${variants_name}.npz" \
        --fai "${fasta}.fai" --n "${v_n}" --seed 1 --out-dir "${out}" || return 1
    # VS's src/utils/io.py imports pybedtools (line 3), which chrombpnet's requirements do not list; only
    # load_genes/load_peaks use it, and variant_scoring.py calls neither. A stand-in keeps the import working.
    if ! timeout -k 30 600 apptainer exec --cleanenv "${sif}" python -c 'import pybedtools' > /dev/null 2>&1; then
        stub="${out}/pybedtools_stub"
        mkdir -p "${stub}"
        cat > "${stub}/pybedtools.py" << 'EOF'
"""Stand-in for pybedtools, written by workflows/variant_scoring/pilot_steps.sh for VS main's V2 run.

kundajelab/variant-scorer imports pybedtools in src/utils/io.py, but only load_genes and load_peaks use it, and
variant_scoring.py calls neither. Anything that does use it fails loudly here."""


class BedTool:
    def __init__(self, *args, **kwargs):
        raise ImportError("pybedtools is not installed in this container; this is the pilot's stand-in")

    @classmethod
    def from_dataframe(cls, *args, **kwargs):
        raise ImportError("pybedtools is not installed in this container; this is the pilot's stand-in")
EOF
    fi
    # VS main, unchanged: no null (-t 0) and no peaks, so only the variants are scored; no profile h5. --cleanenv
    # keeps the host's library paths out of the container's TensorFlow; what VS needs is passed explicitly, and only
    # the directories it reads and writes are bound.
    local binds=() bind
    for bind in "$(dirname "${fasta}")" "$(dirname "$(readlink -f "${fasta}")")" "${VS_REPO}" \
            "$(readlink -f "${VS_REPO}")" "${PILOT_ROOT}"; do
        [[ " ${binds[*]} " == *" ${bind} "* ]] || binds+=( "${bind}" )  # a link's directory and its target's
    done
    taskset -c "${cpus}" timeout -k 60 2700 \
        apptainer exec --nv --cleanenv --bind "$(IFS=,; echo "${binds[*]}")" "${sif}" \
        env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" PYTHONPATH="${stub}" TF_FORCE_GPU_ALLOW_GROWTH=true \
        OMP_NUM_THREADS=16 \
        python "${vsrc}/variant_scoring.py" -l "${out}/vs_variants.tsv" -g "${fasta}" -s "${out}/chrom.sizes" \
        -m "${out}/chrombpnet_nobias.legacy.h5" -o "${out}/vs/nobias" -t 0 --no_hdf5 -bs 256
}

step_checks() {
    local cheri nobias bias full vin="${windows}/${variants_name}.npz" g=0 pids=() runs=() s=() item name gpu m
    local v3="${checks}/v3" v4="${checks}/v4" sample=( --sample "${v_n}" --sample-seed 11 )
    IFS=$'\t' read -r _ _ cheri nobias bias full _ <<< "$(v_row)"
    say "checks on ${v_dataset} fold ${v_fold}"
    # V2's VS run (off unless PILOT_V2=1), in the background on the last GPU: it gates nothing, and step 6 compares it.
    (
        export CUDA_VISIBLE_DEVICES="${gpus[-1]}"
        v2_run > "${logs}/checks/v2.log" 2>&1
        echo $? > "${logs}/checks/v2.exit"
    ) &
    v2_pid=$!
    # launch_check <name> <cmd...> -- one check run on the next GPU and CPU block.
    launch_check() {
        gpu="${gpus[$(( g % ${#gpus[@]} ))]}"
        launch "checks/$1" "${gpu}" "$(cpu_block "${g}" 8)" 0.45 "${@:2}"
        pids+=( "${launched_pid}" )
        g=$(( g + 1 ))
    }
    # V3: the derived full model against chrombpnet.h5 itself (predict_chrombpnet.py --full), both strands
    launch_check v3.default "${CHROMBPNET_PY}" "${vs}/predict_chrombpnet.py" --nobias "${nobias}" --bias "${bias}" \
        --full "${full}" --windows "${vin}" --sample "${v3_n}" --sample-seed 3 --precision default \
        --batch "${cbp_batch}" --out-dir "${v3}/default" --overwrite
    launch_check v3.highest env XLA_FLAGS=--xla_gpu_deterministic_ops=true "${CHROMBPNET_PY}" \
        "${vs}/predict_chrombpnet.py" --nobias "${nobias}" --bias "${bias}" --full "${full}" --windows "${vin}" \
        --sample "${v3_n}" --sample-seed 3 --precision highest --batch "${cbp_batch}" --out-dir "${v3}/highest" \
        --overwrite
    # V4, Cherimoya: A the stack default; F A again; B reordered, batch 256; C the grad path; D strict fp32 (grad
    # path, TF32 off); E bf16 autocast. A row's own --batch comes after the common one, and argparse keeps the last.
    runs=( "A|" "F|" "B|--batch 256 --permute-seed 7" "C|--grad-path" "D|--precision strict-fp32"
           "E|--precision bf16" )
    for item in "${runs[@]}"; do
        name="${item%%|*}"
        read -r -a s <<< "${item#*|}"
        launch_check "v4.cherimoya.${name}" "${CHERIMOYA_PY}" "${vs}/predict_cherimoya.py" --model "${cheri}" \
            --windows "${vin}" "${sample[@]}" --batch "${cheri_batch}" "${s[@]}" --out-dir "${v4}/cherimoya.${name}" \
            --overwrite
    done
    # V4, ChromBPNet: A the default (TF32); F A again; B reordered, batch 128; C highest
    runs=( "A|" "F|" "B|--batch 128 --permute-seed 7" "C|--precision highest" )
    for item in "${runs[@]}"; do
        name="${item%%|*}"
        read -r -a s <<< "${item#*|}"
        launch_check "v4.chrombpnet.${name}" "${CHROMBPNET_PY}" "${vs}/predict_chrombpnet.py" --nobias "${nobias}" \
            --bias "${bias}" --windows "${vin}" "${sample[@]}" --batch "${cbp_batch}" "${s[@]}" \
            --out-dir "${v4}/chrombpnet.${name}" --overwrite
    done
    wait_all "${pids[@]}"
    local ran=$? st=()
    local vfile="${variants_name}.npz" c="${v4}/cherimoya" b="${v4}/chrombpnet"
    pair V3 default "cbp_full derived vs chrombpnet.h5, default precision" "${v3}/default/cbp_full.${vfile}" \
        "${v3}/default/cbp_full_direct.${vfile}"; st+=( $? )
    pair V3 highest "cbp_full derived vs chrombpnet.h5, precision highest" "${v3}/highest/cbp_full.${vfile}" \
        "${v3}/highest/cbp_full_direct.${vfile}" 1e-4; st+=( $? )
    pair V4 cherimoya.rerun "cherimoya: run twice" "${c}.A/cherimoya.${vfile}" "${c}.F/cherimoya.${vfile}"; st+=( $? )
    pair V4 cherimoya.reorder "cherimoya: reordered, batch 256" "${c}.A/cherimoya.${vfile}" \
        "${c}.B/cherimoya.${vfile}"; st+=( $? )
    pair V4 cherimoya.megakernel_vs_grad "cherimoya: megakernel vs grad path (TF32)" "${c}.A/cherimoya.${vfile}" \
        "${c}.C/cherimoya.${vfile}"; st+=( $? )
    pair V4 cherimoya.tf32_vs_fp32 "cherimoya: grad path, TF32 vs strict fp32" "${c}.C/cherimoya.${vfile}" \
        "${c}.D/cherimoya.${vfile}"; st+=( $? )
    pair V4 cherimoya.default_vs_fp32 "cherimoya: stack default vs strict fp32" "${c}.A/cherimoya.${vfile}" \
        "${c}.D/cherimoya.${vfile}"; st+=( $? )
    pair V4 cherimoya.bf16 "cherimoya: bf16 autocast vs stack default" "${c}.A/cherimoya.${vfile}" \
        "${c}.E/cherimoya.${vfile}"; st+=( $? )
    for m in cbp_nobias cbp_bias cbp_full; do
        pair V4 "${m}.rerun" "${m}: run twice" "${b}.A/${m}.${vfile}" "${b}.F/${m}.${vfile}"; st+=( $? )
        pair V4 "${m}.reorder" "${m}: reordered, batch 128" "${b}.A/${m}.${vfile}" "${b}.B/${m}.${vfile}"; st+=( $? )
        pair V4 "${m}.tf32_vs_fp32" "${m}: TF32 vs highest" "${b}.A/${m}.${vfile}" "${b}.C/${m}.${vfile}"; st+=( $? )
    done
    (( ran == 0 )) || return 1
    worst "${st[@]}"
}

# ------------------------------------------------------------------------------------------------ 5 predict
# v2_running -- V2's VS run (started by step 4 in the background on the last GPU) has not ended. Its subshell writes
# logs/checks/v2.exit when it ends, skipped or not; the PID test keeps a subshell that died without writing it from
# being waited for. A slot is a sibling of that subshell, not its parent, so it cannot `wait` for it.
v2_running() { [[ -n "${v2_pid:-}" && ! -e "${logs}/checks/v2.exit" ]] && kill -0 "${v2_pid}" 2> /dev/null; }

# claim <counter file> <n> -- the next task number (0-based) of the queue the slots share, under a lock; fails once
# all n have been handed out.
claim() {
    (
        flock -x 9
        next="$(cat "$1")"
        (( next < $2 )) || exit 1
        echo $(( next + 1 )) > "$1"
        echo "${next}"
    ) 9> "$1.lock"
}

# predict_task <gpu> <cpus> <task> -- one task, "<kind>\t<manifest row>". A task that fails (say, out of GPU memory at
# the fixed batch) runs once more at half the batch, as task <name>.retry: the window sets it finished are kept, since
# predict_common.run_window_sets skips a set whose outputs all exist.
predict_task() {
    local gpu="$1" cpus="$2" kind ds fold cheri nobias bias out batch name
    local cmd=() sets=() limit=()
    # PILOT_TASK_MINUTES: TERM at the limit, KILL a minute later; timeout exits 124, and the task counts as failed.
    if (( task_minutes > 0 )); then limit=( timeout -k 60 "${task_minutes}m" ); fi
    IFS=$'\t' read -r kind ds fold cheri nobias bias _ _ <<< "$3"
    out="${preds}/${ds}/fold_${fold}"
    sets=( --windows "${windows}/${variants_name}.npz" --windows "${windows}/null.npz"
           --windows "${windows}/peaks.${ds}.npz" )
    # The genomic null set, when step 2 (of this box or an earlier one) built it and the knob is not set empty.
    if use_genomic_null; then sets+=( --windows "${windows}/genomic_null.npz" ); fi
    if [[ "${kind}" == cherimoya ]]; then
        batch="${cheri_batch}"
        cmd=( "${limit[@]}" "${CHERIMOYA_PY}" "${vs}/predict_cherimoya.py" --model "${cheri}" "${sets[@]}"
              --out-dir "${out}" )
    else
        batch="${cbp_batch}"
        cmd=( "${limit[@]}" "${CHROMBPNET_PY}" "${vs}/predict_chrombpnet.py" --nobias "${nobias}" --bias "${bias}"
              "${sets[@]}" --out-dir "${out}" )
    fi
    name="predict/${ds}.fold_${fold}.${kind}"
    run_logged "${name}" "${gpu}" "${cpus}" "${gpu_fraction}" "${cmd[@]}" --batch "${batch}" && return 0
    run_logged "${name}.retry" "${gpu}" "${cpus}" "${gpu_fraction}" "${cmd[@]}" \
        --batch "$(( batch > 1 ? batch / 2 : 1 ))"
}

# predict_worker <gpu> <cpus> -- one slot: the next task of the shared queue (step_predict's order and queue), until
# none is left. On the last GPU it first waits for V2's VS run to end (see step 5 above).
predict_worker() {
    local gpu="$1" cpus="$2" i failed=0
    if [[ "${gpu}" == "${gpus[-1]}" ]] && v2_running; then
        say "slot on GPU ${gpu}: waiting for V2's VS run (PID ${v2_pid}) to end before its first task"
        while v2_running; do sleep 10; done
        say "slot on GPU ${gpu}: V2's VS run has ended; taking tasks"
    fi
    while i="$(claim "${queue}" "${#order[@]}")"; do
        predict_task "${gpu}" "${cpus}" "${order[i]}" || failed=1
    done
    return "${failed}"
}

step_predict() {
    local rows=() order=() row kind n_slots per i pids=() queue="${box_dir}/predict.queue"
    mapfile -t rows < <(manifest_cmd tasks)
    (( ${#rows[@]} > 0 )) || { echo "ERROR: no tasks in ${manifest}" >&2; return 1; }
    n_slots=$(( ${#gpus[@]} * gpu_slots ))
    per=$(( ${#allowed[@]} / n_slots ))
    (( per > 0 )) || { echo "ERROR: ${n_slots} slots but only ${#allowed[@]} CPUs" >&2; return 1; }
    # Longest first: every ChromBPNet task (both parts in one process, ~6,400 passes/s on an idle H100, so ~33 min a
    # fold at 3.2M variants x 2 alleles x 2 strands), then every Cherimoya task (~26,000 passes/s, ~8 min); only the
    # kinds of PILOT_KINDS. Each slot takes the next one when it is free (largest first, then whoever is free: the GPUs
    # end within about one Cherimoya task of each other), slot s on GPU s % n_gpus.
    for kind in "${kinds[@]}"; do
        for row in "${rows[@]}"; do order+=( "${kind}"$'\t'"${row}" ); done
    done
    echo 0 > "${queue}"
    say "${#order[@]} prediction tasks (${kinds[*]}, in that order) over ${n_slots} slots (${#gpus[@]} GPUs x" \
        "${gpu_slots}), ${per} CPUs each; next task from ${queue}"
    if v2_running; then say "V2's VS run is still on GPU ${gpus[-1]}: that GPU's slots start when it ends"; fi
    for (( i = 0; i < n_slots; i++ )); do
        predict_worker "${gpus[$(( i % ${#gpus[@]} ))]}" "$(cpu_block "${i}" "${per}")" &
        pids+=( "$!" )
    done
    wait_all "${pids[@]}"
}

# ------------------------------------------------------------------------------------------------ 6 score
# v2_compare -- wait for V2's VS run (bounded: its commands are under timeout) and compare its scores with ours. A V2
# run that failed gets its own row (4 v2 failed) in status.tsv and does not fail this step. Without chrombpnet in
# PILOT_KINDS there is nothing of ours to compare with.
v2_compare() {
    local vs_scores="${checks}/v2/vs/nobias.variant_scores.tsv" v2_exit
    if [[ -n "${v2_pid:-}" ]]; then
        say "V2: waiting for VS main (PID ${v2_pid})"
        wait "${v2_pid}"
    fi
    v2_exit="$(cat "${logs}/checks/v2.exit" 2>/dev/null || echo 0)"
    if [[ "${v2_exit}" != 0 ]]; then
        say "V2: VS main's run failed (exit ${v2_exit}); see ${logs}/checks/v2.log"
        record 4 v2 failed "${v2_exit}" 0 "see ${logs}/checks/v2.log"
    fi
    if [[ ! -s "${vs_scores}" ]]; then
        say "V2: no VS scores ($(cat "${checks}/v2/skipped.txt" 2>/dev/null || echo "see ${logs}/checks/v2.log"))"
        return 0
    fi
    if ! has_kind chrombpnet; then
        say "V2: no comparison: PILOT_KINDS (${kinds[*]}) has no chrombpnet, so step 5 predicted no cbp_nobias"
        return 0
    fi
    pinned "${DEFAULT_PY}" "${vs}/vchecks.py" vs-compare --vs-scores "${vs_scores}" \
        --preds "${preds}/${v_dataset}/fold_${v_fold}/cbp_nobias.${variants_name}.npz" --out "${checks}/V2.vs_main.json"
}

# scored_dir <null> <kind> -- where step 6 writes against a null: <kind>/ for the shuffled null (window set `null`),
# <kind>_<null>/ for another (kind: scores or compare).
scored_dir() { if [[ "$1" == null ]]; then echo "${PILOT_ROOT}/${2}"; else echo "${PILOT_ROOT}/${2}_${1}"; fi; }

step_score() {
    local list datasets=() nulls=(null) null ds st=() gn_rc=0 t0 slot=0 v5_pid rc
    local null_pids=() gn_pids=() compare_pids=() fold_args=()
    list="$(variant_list)" || return 1
    mapfile -t datasets < <(manifest_cmd peaks | cut -f1)
    # The genomic null, when step 2 built its windows and the knob is not set empty. --list stays the variant list (the
    # variants' fold_test and strata); the shuffled null's outputs stay in scores/ and compare/. Like its windows, it
    # gates nothing: a failure gets its own row (score_genomic_null) in status.tsv, and this step's state stays the
    # shuffled null's.
    if use_genomic_null; then nulls+=( genomic_null ); fi
    (( fold_tsv == 1 )) || fold_args=( --no-fold-tsv )
    t0=${SECONDS}
    # V5 reads only the predictions: in the background, on CPUs the scores do not use.
    launch score/v5 - "$(cpu_block 24 8)" 0 "${DEFAULT_PY}" "${vs}/vchecks.py" strands --preds-root "${preds}" \
        --out "${checks}/V5.strands.json"
    v5_pid=${launched_pid}
    # scores.py once per dataset and null, all at once, each on its own 8 CPUs (it is single-threaded; 20 processes at
    # 3.2M variants hold ~5 GB each). Each scores every model it has predictions for and exits 1 if any is missing or
    # refused; compare.py then compares the models that have scores. So both run whatever the other's status.
    for null in "${nulls[@]}"; do
        for ds in "${datasets[@]}"; do
            launch "score/scores.${null}.${ds}" - "$(cpu_block "${slot}" 8)" 0 "${DEFAULT_PY}" "${vs}/scores.py" \
                --datasets "${ds}" --preds-root "${preds}" --windows-root "${windows}" --list "${list}" \
                --variants "${variants_name}" --null "${null}" --peaks 'peaks.{dataset}' \
                --out-root "$(scored_dir "${null}" scores)" --models "${score_models[@]}" "${fold_args[@]}"
            if [[ "${null}" == null ]]; then null_pids+=( "${launched_pid}" ); else gn_pids+=( "${launched_pid}" ); fi
            slot=$(( slot + 1 ))
        done
    done
    wait_all "${null_pids[@]}"
    st+=( "$?" )
    wait_all "${gn_pids[@]}" || gn_rc=1
    say "scores.py: done for ${#datasets[@]} dataset(s) x ${#nulls[@]} null(s) in $(( SECONDS - t0 )) s;" \
        "logs ${logs}/score/scores.*.log"
    # compare.py per null, both at once, each with one worker process per dataset on its own 32 CPUs. Single-threaded
    # BLAS: the workers are the parallelism. It compares Cherimoya with ChromBPNet, so it needs both in PILOT_KINDS.
    if both_kinds; then
        slot=0
        for null in "${nulls[@]}"; do
            launch "score/compare.${null}" - "$(cpu_block "${slot}" 32)" 0 env OMP_NUM_THREADS=1 \
                OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "${CHERIMOYA_PY}" "${vs}/compare.py" \
                --scores-root "$(scored_dir "${null}" scores)" --datasets "${datasets[@]}" --list "${list}" \
                --variants "${variants_name}" --null "${null}" --out-dir "$(scored_dir "${null}" compare)" \
                --workers "${#datasets[@]}"
            compare_pids+=( "${launched_pid}" )
            slot=$(( slot + 1 ))
        done
        wait "${compare_pids[0]}"
        st+=( "$?" )
        if (( ${#compare_pids[@]} > 1 )); then wait "${compare_pids[1]}" || gn_rc=1; fi
    else
        say "compare: not run: it compares Cherimoya with ChromBPNet, and PILOT_KINDS is ${kinds[*]}"
        record 6 compare skipped - 0 "PILOT_KINDS=${kinds[*]}: compare.py compares Cherimoya with ChromBPNet"
    fi
    if (( gn_rc != 0 )); then
        say "genomic null: scores.py or compare.py failed; recorded as score_genomic_null"
        record 6 score_genomic_null failed 1 "$(( SECONDS - t0 ))" "see ${logs}/score/; gates nothing"
    fi
    # null_report.py over every null scored above; it gates nothing and gets its own row (analysis) in status.tsv.
    # Its rule is Cherimoya against ChromBPNet's uncorrected model, so it needs both in PILOT_KINDS.
    t0=${SECONDS}
    if ! both_kinds; then
        say "analysis: not run: null_report.py compares Cherimoya with ChromBPNet, and PILOT_KINDS is ${kinds[*]}"
        record 6 analysis skipped - 0 "PILOT_KINDS=${kinds[*]}: null_report.py compares Cherimoya with ChromBPNet"
    elif [[ ! -f "${nulls_dir}/null_report.py" ]]; then
        say "analysis: no ${nulls_dir}/null_report.py"
        record 6 analysis failed 1 0 "no ${nulls_dir}/null_report.py; gates nothing"
    else
        run_logged score/analysis - "$(cpu_block 0 32)" 0 "${DEFAULT_PY}" "${nulls_dir}/null_report.py" \
            --root "${PILOT_ROOT}" --list "${list}" --datasets "${datasets[@]}" --variants "${variants_name}" \
            --nulls "${nulls[@]}" --out-dir "${PILOT_ROOT}/analysis"
        rc=$?
        if (( rc == 0 )); then
            record 6 analysis "done" 0 "$(( SECONDS - t0 ))" "${PILOT_ROOT}/analysis; gates nothing"
        else
            say "analysis: null_report.py failed (exit ${rc}); recorded as analysis"
            record 6 analysis failed "${rc}" "$(( SECONDS - t0 ))" "see ${logs}/score/analysis.log; gates nothing"
        fi
    fi
    wait "${v5_pid}"
    st+=( "$?" )
    # Last, so that a slow V2 never holds up the scores and the report.
    v2_compare
    st+=( "$?" )
    "${DEFAULT_PY}" "${vs}/vchecks.py" summary --inputs "${checks}"/V*.json --out "${checks}/summary.tsv"
    st+=( "$?" )
    worst "${st[@]}"
}

# ------------------------------------------------------------------------------------------------ preflight
preflight() {
    local bad=0 py
    for py in "${DEFAULT_PY}" "${CHERIMOYA_PY}" "${CHROMBPNET_PY}"; do
        [[ -x "${py}" ]] || { echo "ERROR: no interpreter ${py} (set RVI_MAIN / CHROMBPNET_REPO or the *_PY knobs)" >&2; bad=1; }
    done
    (( bad == 0 )) || return 1
    manifest_cmd check || return 1
    (( ${#gpus[@]} > 0 )) || { echo "ERROR: no GPU found (nvidia-smi) and PILOT_GPUS unset" >&2; return 1; }
    variant_list > /dev/null || return 1
    v_dataset="${PILOT_V_DATASET:-$(manifest_cmd peaks | awk 'NR == 1 { print $1 }')}"
    [[ -n "$(v_row)" ]] || { echo "ERROR: no fold ${v_fold} of ${v_dataset} in ${manifest}" >&2; return 1; }
}

# failed_rows -- the rows of status.tsv in state `failed`, as "<step> <name>" lines; exit 1 when there are none.
failed_rows() { awk -F'\t' 'NR > 1 && $3 == "failed" { print $1, $2; found = 1 } END { exit !found }' "${status}"; }

# exit_rows -- the failed rows that make this script exit 1, as failed_rows prints them: every one, except step 1's
# when the tests gate nothing (gate empty: PILOT_GATE_TESTS=0); exit 1 when there are none.
exit_rows() { failed_rows | awk -v gate="${gate:-}" 'gate == "1" || $1 != 1 { print; found = 1 } END { exit !found }'; }

# hold_rows -- the failed rows that hold the node (see Hold above), as failed_rows prints them; exit 1 when there are
# none. step 1's tests hold only when they gate step 2 (gate, set in main).
hold_rows() {
    local holding='^(windows|windows_genomic_null|checks|predict|score|score_genomic_null|analysis)$'
    awk -F'\t' -v gate="${gate:-}" -v holding="${holding}" '
        NR > 1 && $3 == "failed" && ($2 ~ holding || ($2 == "tests" && gate == "1")) { print $1, $2; found = 1 }
        END { exit !found }' "${status}"
}

# steps_running -- a step of this job other than its batch script and extern step is running: a rerun or a command
# started with srun --jobid <job> --overlap. False outside Slurm, or when squeue cannot tell.
steps_running() {
    [[ -n "${SLURM_JOB_ID:-}" ]] || return 1
    squeue -h -s -j "${SLURM_JOB_ID}" -o %i 2> /dev/null | grep -qvE '\.(batch|extern)$'
}

# hold <minutes> -- after a failed row that holds: keep this allocation for up to <minutes>, and past them while a
# step started in it runs, so that the failed step can be fixed and run again inside it rather than in a new box;
# ends early when box/<job>.<step>/release exists.
hold() {
    local minutes="$1" release="${box_dir}/release" deadline first rerun polls=0 over=""
    local job="${SLURM_JOB_ID:-<job>}" script="${repo_root}/workflows/variant_scoring/pilot_steps.sh"
    deadline=$(( SECONDS + minutes * 60 ))
    first="$(hold_rows | sort -n | awk 'NR == 1 { print $1 }')"
    # This run's own steps from the first failed one on (a step it left out stays out).
    rerun="$(tr ',' '\n' <<< "${steps}" | awk -v first="${first:-6}" 'NF && $1 >= first' | paste -sd, -)"
    say "HOLD: failed: $(failed_rows | paste -sd, - | sed 's/,/, /g'); holding for: $(hold_rows | paste -sd, - |
        sed 's/,/, /g')"
    say "HOLD: keeping job ${job} on $(hostname) for up to ${minutes} min, until" \
        "$(TZ=America/New_York date -d "+${minutes} min" '+%F %T %Z') (longer while a step started in it runs) or its" \
        "time limit if that comes first; waiting for ${release}, checked every 30 s"
    say "HOLD: fix the step, then run it again inside this allocation (a rerun gets its own box/<job>.<step>/ and" \
        "scratch, so nothing of this box is overwritten; steps 2 and 5 keep the outputs they find complete, though" \
        "step 5 still loads each task's windows and model, ~10-15 min for all of them). From a copy of the" \
        "checkout's script, never the checkout's own file, with this run's CPUs and no second hold, e.g.:"
    say "HOLD:   f=${box_dir}/pilot_steps.rerun.\$(date +%s).sh; cp ${script} \"\$f\" &&" \
        "PILOT_ROOT=${PILOT_ROOT} PILOT_REPO_ROOT=${repo_root} PILOT_STEPS=${rerun:-${first:-6}} PILOT_HOLD_MINUTES=0" \
        "srun --jobid ${job} --overlap -N1 -n1 taskset -c ${cpus_allowed_list} bash \"\$f\""
    say "HOLD: or any one command of ${logs} the same way (srun --jobid ${job} --overlap -N1 -n1 taskset -c" \
        "${cpus_allowed_list} <command>)"
    say "HOLD: when done, release the node (this job ends, and with it any step still running in it): touch ${release}"
    while :; do
        if [[ -e "${release}" ]]; then
            say "HOLD: ${release} found; releasing the node"
            return 0
        fi
        if (( SECONDS >= deadline )); then
            if ! steps_running; then
                say "HOLD: ${minutes} min are up${over:+, and the steps started in this job have ended}; releasing" \
                    "the node"
                return 0
            fi
            [[ -n "${over}" ]] || say "HOLD: ${minutes} min are up, but a step started in this job is still running" \
                "(squeue -s -j ${job}); holding until it ends, ${release} exists or the time limit comes"
            over=1
        fi
        sleep 30
        polls=$(( polls + 1 ))
        if (( polls % 30 == 0 )); then
            if [[ -n "${over}" ]]; then
                say "HOLD: still waiting for the steps started in this job, or ${release}"
            else
                say "HOLD: still waiting for ${release}; $(( (deadline - SECONDS + 59) / 60 )) min left"
            fi
        fi
    done
}

sampler() {
    printf 'time_et\tmem_used_gb\tload1\tgpus(util%%,mem_gb)\n' > "${box_dir}/resources.tsv"
    while true; do
        printf '%s\t%s\t%s\t%s\n' "$(et)" \
            "$(awk '/^MemTotal:/{t=$2} /^MemAvailable:/{a=$2} END{printf "%d", (t-a)/1048576}' /proc/meminfo)" \
            "$(cut -d' ' -f1 /proc/loadavg)" \
            "$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null \
                | awk -F', ' '{printf "%s%d,%.0f", (NR>1?" ":""), $1, $2/1024}')" >> "${box_dir}/resources.tsv"
        sleep 30
    done
}

# ------------------------------------------------------------------------------------------------ main
printf 'step\tname\tstate\texit\tseconds\tstarted_et\tnote\n' > "${status}"
printf 'name\tgpu\tcpus\texit\tseconds\tstarted_et\n' > "${tasks_tsv}"
say "pilot box ${box_id} on $(hostname): ${#allowed[@]} CPUs, GPUs ${gpus[*]:-none}; steps ${PILOT_STEPS:-1,2,3,4,5,6}"
say "PILOT_ROOT ${PILOT_ROOT}; code ${repo_root} ($(git -C "${repo_root}" rev-parse --short HEAD 2>/dev/null || echo '?'))"
say "logs ${logs}; models ${kinds[*]}; GPU claims ${gpu_claim_dir:-off}"
step_started="$(et)"
if ! preflight > "${logs}/0.preflight.log" 2>&1; then
    cat "${logs}/0.preflight.log" >&2
    record 0 preflight failed 1 0 "see ${logs}/0.preflight.log"
    exit 1
fi
record 0 preflight "done" 0 0 ""
say "checks run on ${v_dataset} fold ${v_fold}"
sampler &
sampler_pid=$!
v2_pid=""

state[0]="done"
if [[ "${PILOT_GATE_TESTS:-1}" == 1 ]]; then gate=1; else gate=""; fi
run_step 1 tests ""
run_step 2 windows "${gate}"
run_step 3 probe "2"
run_step 4 checks "2"
run_step 5 predict "2"
run_step 6 score "5~"
if [[ -n "${v2_pid}" ]] && kill -0 "${v2_pid}" 2>/dev/null; then
    say "waiting for V2's VS run to end"
    wait "${v2_pid}"
fi

column -t -s $'\t' "${status}" 2>/dev/null || cat "${status}"
say "pilot box ${box_id} finished; status ${status}"
if (( hold_minutes > 0 )) && hold_rows > /dev/null; then
    hold "${hold_minutes}"
elif (( hold_minutes > 0 )) && failed_rows > /dev/null; then
    say "no hold: the failed rows ($(failed_rows | paste -sd, - | sed 's/,/, /g')) gate nothing and need no rerun"
fi
if exit_rows > /dev/null; then exit 1; fi
if failed_rows > /dev/null; then
    say "exit 0: the only failed rows are step 1's ($(failed_rows | paste -sd, - | sed 's/,/, /g')), which gate" \
        "nothing with PILOT_GATE_TESTS=0"
fi
