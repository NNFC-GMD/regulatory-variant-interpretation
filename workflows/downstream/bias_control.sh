#!/bin/bash
# bias_control.sh
# Purpose: the bias control of the ChromBPNet/Cherimoya comparison. ChromBPNet's scores come from its bias-free model
#   (chrombpnet_nobias.h5); Cherimoya has no bias model, so Tn5 bias can show up only on Cherimoya's side. This
#   attributes ChromBPNet's bias-included model (models/chrombpnet.h5) of one fold on that fold's 30K-peak QC
#   subsample with 05.0's command, then runs TF-MoDISco on the scores at 04.5's QC scale, as qc_modisco_folds.sh
#   runs it on Cherimoya's, so that patterns only Cherimoya finds can be looked up among the bias-included model's.
#
#   contribs  `chrombpnet contribs_bw -m <fold>/models/chrombpnet.h5 -r <fold>/auxiliary/30K_subsample_peaks.bed
#             -g <genome_fa> -c <chrom_sizes> -op <out_dir>/interpretation -pc <head> --shap-seed <seed>`, 05.0's
#             command (05.0:162-169) and the seed of 04.4's QC scores of the same peaks, in the chrombpnet env (the
#             config sourced through lib/bash/config.sh, activate_env, gpu_env, require_gpu jax), on one GPU claimed
#             as full_chain.sh claims one: under flock on /dev/shm/$USER/box.<job>/gpu_claims/.lock, the GPU with
#             the fewest claim files plus unclaimed GPU steps, ties broken by the least memory in use (motif_chain.sh's
#             picker); a claim file <gpu>.<pid>.<rand>.claim while it runs, GPU_CLAIM=1 and CUDA_VISIBLE_DEVICES for
#             the step. JAX's allocator grows on demand up to XLA_PYTHON_CLIENT_MEM_FRACTION of the GPU (default
#             0.09, 7.2 GB of 80 GB), so the process stays under 8 GB.
#   modisco   src/motif_qc.py on <out_dir>/interpretation.<head>_scores.h5, as 04.5 runs it (window 400, 5,000
#             seqlets), --prefix chrombpnet, into <out_dir>/motif_qc_<head>/, without a GPU
#
# A model contribs_bw refuses is the plan's fallback, not a failure to retry: chrombpnet's DeepSHAP has no rule for
#   the logsumexp that joins a bias-included model's bias and bias-free counts, so it refuses the counts head of
#   chrombpnet.h5 (explainer.check_model raises NotImplementedError before any peak is read; the profile head is
#   supported). Its reason is recorded in <out_dir>/status.json, MoDISco is skipped and this exits 3. A rerun tries
#   again (it takes about a minute), so a newer chrombpnet is picked up.
#
# Done: <out_dir>/.done/contribs.json once contribs_bw exits 0 with the h5, the bigwig and the bed (05.0's outputs),
#   recording the model, the regions and the h5 (size, mtime); <out_dir>/motif_qc_<head>/chrombpnet_qc.done.json once
#   motif_qc.py exits 0 with its outputs, recording the h5 it read. A step whose marker no longer matches is run again
#   (contribs_bw overwrites its outputs, as in 05.0; a MoDISco directory that is not done is renamed
#   <dir>.incomplete.<time> first). <out_dir>/status.json says where it stands: attributed, done, rejected, failed.
# SIGTERM, SIGINT or SIGHUP stops the step, releases the claim and exits 143 with no marker.
#
# Input:  $1 the ChromBPNet dataset's config.yaml (the model, the subsample, the genome and chrom sizes come from it
#         through config.sh); $2 the fold (as the config's folds list names it); $3 the output directory.
#         Environment: BOX_DIR (the record), REPO_ROOT (the igvf_tf_chrombpnet checkout whose config.sh and
#         src/motif_qc.py run, with workflows/dcai/env.sh, CHROMBPNET_REPO and CHROMBPNET_PIXI_ENV).
# Knobs:  CHERI_BIAS_HEAD       counts (default; the comparison's head) or profile
#         CHERI_BIAS_SHAP_SEED  contribs_bw --shap-seed (default 1234, 04.4's)
#         CHERI_BIAS_THREADS    threads of contribs_bw (default 8, on a mask of 16 CPUs, as full_chain.sh's GPU steps)
#         CHERI_QC_THREADS      threads and CPUs of MoDISco (default 16)
#         CHERI_QC_NICE         niceness of both steps (default 19)
#         CHERI_QC_CORES_DIR    pick_cores.py's reservations (default downstream.sh's)
# Output: the files above; logs in <out_dir>/logs/{contribs,modisco}.log (appended to); rows bias_contribs_<fold> and
#   bias_modisco_<fold> in ${BOX_DIR}/cheri_qc.tsv (dataset, step, exit, seconds, max RSS kB, started, note).
# Exit:   0 done, 3 contribs_bw refused the model (recorded), 1 failed, 2 usage.
# Usage:  bias_control.sh <chrombpnet config.yaml> <fold> <out_dir>   (inside the box; the box's cheri_qc.sh bias
#         sets the environment and holds the box)

set -uo pipefail

usage="usage: bias_control.sh <chrombpnet config.yaml> <fold> <out_dir>"
(( $# == 3 )) || { echo "${usage}" >&2; exit 2; }
: "${BOX_DIR:?set BOX_DIR to the box directory}"
: "${REPO_ROOT:?set REPO_ROOT to the igvf_tf_chrombpnet checkout whose config.sh and src/motif_qc.py run}"
abspath() { if [[ "$1" == /* ]]; then printf '%s\n' "$1"; else printf '%s\n' "${PWD}/$1"; fi; }
config="$(abspath "$1")"
fold="$2"
out_dir="$(abspath "${3%/}")"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
here="${repo_root}/workflows/downstream"
case ":${PATH}:" in
    *":${HOME}/.pixi/bin:"*) ;;
    *) export PATH="${HOME}/.pixi/bin:${PATH}" ;;
esac
exec 200>&-  # the box wrapper's one-run lock stays with the wrapper

head="${CHERI_BIAS_HEAD:-counts}"
seed="${CHERI_BIAS_SHAP_SEED:-1234}"
gpu_threads="${CHERI_BIAS_THREADS:-8}"
qc_threads="${CHERI_QC_THREADS:-16}"
niceness="${CHERI_QC_NICE:-19}"
pick_sample="${CHERI_PICK_SAMPLE:-1}"
box_scratch="/dev/shm/${USER}/box.${SLURM_JOB_ID:-local}"
cores_dir="${CHERI_QC_CORES_DIR:-${box_scratch}/cheri_downstream/cores}"
claims="${box_scratch}/gpu_claims"
record_file="${BOX_DIR}/cheri_qc.tsv"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.09}"

log() { printf '[%s] %s\n' "$(date)" "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

[[ "${head}" == counts || "${head}" == profile ]] || die "CHERI_BIAS_HEAD must be counts or profile, got '${head}'"
for n in "${seed}" "${gpu_threads}" "${qc_threads}" "${niceness}"; do
    [[ "${n}" =~ ^[0-9]+$ ]] || die "the seed, the thread counts and CHERI_QC_NICE must be whole numbers"
done
(( gpu_threads > 0 && qc_threads > 0 )) || die "the thread counts must be positive"
[[ -f "${config}" ]] || die "no config ${config}"
for f in src/motif_qc.py lib/bash/config.sh; do
    [[ -f "${REPO_ROOT}/${f}" ]] || die "no ${REPO_ROOT}/${f}: REPO_ROOT is not an igvf_tf_chrombpnet checkout"
done
[[ -x /usr/bin/time ]] || die "no /usr/bin/time, which measures each step's max RSS"

# What the config resolves to, through the checkout's own config.sh.
# shellcheck disable=SC2016  # expanded by the inner bash
resolved="$(env DATASET_CONFIG="${config}" bash -c 'source "${REPO_ROOT}/lib/bash/config.sh" > /dev/null || exit 1
    printf "%s\n" "${dataset_name}" "${peak_type}" "${full_model_dir}" "${genome_fa}" "${chrom_sizes}" "${folds[*]}"')" \
    || die "cannot read ${config} through ${REPO_ROOT}/lib/bash/config.sh"
mapfile -t cfg <<< "${resolved}"
(( ${#cfg[@]} == 6 )) || die "${config} resolved to ${#cfg[@]} values, not 6"
ds="${cfg[0]}"
read -r -a folds <<< "${cfg[5]}"
[[ " ${folds[*]} " == *" ${fold} "* ]] || die "${config} lists folds ${folds[*]}, not '${fold}'"
fold_model_dir="${cfg[2]}/${ds}_${cfg[1]}_fold_${fold}"
model="${fold_model_dir}/models/chrombpnet.h5"
regions="${fold_model_dir}/auxiliary/30K_subsample_peaks.bed"
for f in "${model}" "${regions}" "${cfg[3]}" "${cfg[4]}"; do
    [[ -f "${f}" ]] || die "no ${f}"
done
prefix="${out_dir}/interpretation"
h5="${prefix}.${head}_scores.h5"
bw="${prefix}.${head}_scores.bw"
bed="${prefix}.interpreted_regions.bed"
qc_dir="${out_dir}/motif_qc_${head}"
contribs_marker="${out_dir}/.done/contribs.json"
qc_marker="${qc_dir}/chrombpnet_qc.done.json"
status_file="${out_dir}/status.json"
logs="${out_dir}/logs"
mkdir -p "${out_dir}/.done" "${logs}" "${cores_dir}" "${claims}" || die "cannot create ${out_dir} or ${box_scratch}"
[[ -s "${record_file}" ]] || printf 'dataset\tstep\texit\tseconds\tmax_rss_kb\tstarted\tnote\n' >> "${record_file}"

# record <step> <exit> <note> -- one row of the record, from run_step's last measurements.
record() {
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${ds}" "$1" "$2" "${step_seconds:-NA}" "${step_rss:-NA}" \
        "${step_started:-NA}" "$3" >> "${record_file}"
}

# status <state> <reason> -- <out_dir>/status.json: where the control stands, and why.
status() {
    python3 - "${status_file}" "$1" "$2" "${ds}" "${fold}" "${head}" "${seed}" "${model}" "${regions}" \
        "${prefix}.interpret.args.json" "${gpu:-}" <<'EOF'
import json
import os
import sys
import time
from pathlib import Path

path, state, reason, ds, fold, head, seed, model, regions, args_json, gpu = sys.argv[1:12]
out = {"state": state, "reason": reason, "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "dataset": ds,
       "fold": fold, "head": head, "shap_seed": int(seed), "model": model, "regions": regions, "gpu": gpu or None}
try:  # what contribs_bw recorded before it looked at the model: the chrombpnet version, the JAX backend
    recorded = json.loads(Path(args_json).read_text())
    out["interpret_args"] = {k: recorded.get(k) for k in ("chrombpnet_version", "jax_backend", "jax_devices",
                                                          "seed", "num_shuffles", "precision_resolved")}
except (OSError, ValueError):
    pass
tmp = Path(path + ".partial")
tmp.write_text(json.dumps(out, indent=2) + "\n")
os.replace(tmp, path)
EOF
}

# marker_ok <marker> <file>... -- exit 0 when the marker records each file's current path, size and mtime (in order).
marker_ok() {
    python3 - "$@" <<'EOF'
import json
import os
import sys

try:
    recorded = json.load(open(sys.argv[1]))["files"]
    current = []
    for path in sys.argv[2:]:
        st = os.stat(path)
        current.append({"path": path, "size": st.st_size, "mtime_ns": st.st_mtime_ns})
except (OSError, ValueError, KeyError, TypeError):
    sys.exit(1)
sys.exit(0 if recorded == current else 1)
EOF
}

# write_marker <marker> <note> <file>... -- a done marker recording the files (path, size, mtime) and a note.
write_marker() {
    python3 - "$@" <<'EOF'
import json
import os
import sys
import time

marker, note, *paths = sys.argv[1:]
files = []
for path in paths:
    st = os.stat(path)
    files.append({"path": path, "size": st.st_size, "mtime_ns": st.st_mtime_ns})
tmp = marker + ".partial"
with open(tmp, "w") as fh:
    json.dump({"written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "note": note, "files": files}, fh, indent=2)
    fh.write("\n")
os.replace(tmp, marker)
EOF
}

n_gpus_node="$(nvidia-smi -L 2>/dev/null | wc -l)"
# full_chain.sh's unclaimed GPU steps, plus 10.0 (motif_chain.sh's list).
gpu_steps_re='SLURM/(03\.0\.train_bias_model|03\.2\.qc_selected_bias|04\.0\.train_full_model|04\.3\.generate_predictions|04\.4\.qc_full_model_interpret|05\.0\.get_contrib_scores|10\.0\.run_finemo_unified)\.sh'
# claim_gpu -- under the lock, the least-loaded GPU (ties: least memory in use); prints the claim file made for it.
claim_gpu() {
    (
        flock 8
        local g p env_g best=0 file
        local -a load=() mem=()
        for (( g = 0; g < n_gpus_node; g++ )); do load[g]=0; mem[g]=0; done
        for file in "${claims}"/*.claim; do
            [[ -e "${file}" ]] || continue
            g="${file##*/}"; g="${g%%.*}"
            [[ "${g}" =~ ^[0-9]+$ ]] && (( g < n_gpus_node )) && load[g]=$(( load[g] + 1 ))
        done
        for p in $(pgrep -f "${gpu_steps_re}"); do
            tr '\0' '\n' < "/proc/${p}/environ" 2>/dev/null | grep -qx 'GPU_CLAIM=1' && continue
            env_g="$(tr '\0' '\n' < "/proc/${p}/environ" 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p')"
            [[ "${env_g}" =~ ^[0-9]+$ ]] && (( env_g < n_gpus_node )) && load[env_g]=$(( load[env_g] + 1 ))
        done
        while IFS=', ' read -r g p; do
            [[ "${g}" =~ ^[0-9]+$ && "${p}" =~ ^[0-9]+$ ]] && (( g < n_gpus_node )) && mem[g]="${p}"
        done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null)
        for (( g = 1; g < n_gpus_node; g++ )); do
            if (( load[g] < load[best] || (load[g] == load[best] && mem[g] < mem[best]) )); then
                best="${g}"
            fi
        done
        file="${claims}/${best}.${BASHPID}.${RANDOM}.claim"
        touch "${file}"
        echo "${file}"
    ) 8> "${claims}/.lock"
}

# run_step <step> <log> <cpus> <threads> <gpu or ""> <cmd...> -- cmd under nice and taskset on <cpus> picked CPUs,
# with the thread counts set to <threads>, on that GPU or none, its max RSS measured by /usr/bin/time, its output
# appended to <log>; sets step_seconds, step_rss and step_started; returns its exit.
run_step() {
    local step="$1" step_log="$2" n_cpus="$3" threads="$4" gpu_id="$5" t0 rc rss_file
    shift 5
    rss_file="${box_scratch}/rss.bias.${BASHPID}.${RANDOM}"
    step_started="$(date -Iseconds)"
    t0="${SECONDS}"
    (
        owner="${BASHPID}"
        cpus="$(python3 "${here}/pick_cores.py" "${n_cpus}" --reserve "${cores_dir}" --owner "${owner}" \
                --sample "${pick_sample}")" || exit 125
        printf '==== [%s] %s fold %s %s on CPUs %s%s: %s\n' "$(date)" "${ds}" "${fold}" "${step}" "${cpus}" \
            "${gpu_id:+, GPU ${gpu_id}}" "${step_desc:-$*}" >> "${step_log}"
        exec env OMP_NUM_THREADS="${threads}" NUMBA_NUM_THREADS="${threads}" MKL_NUM_THREADS="${threads}" \
            OPENBLAS_NUM_THREADS="${threads}" SLURM_CPUS_PER_TASK="${threads}" CUDA_VISIBLE_DEVICES="${gpu_id}" \
            GPU_CLAIM="${gpu_id:+1}" \
            /usr/bin/time -f '%M' -o "${rss_file}" nice -n "${niceness}" taskset -c "${cpus}" "$@" >> "${step_log}" 2>&1
    ) &
    step_pid=$!
    wait "${step_pid}"
    rc=$?
    step_seconds=$(( SECONDS - t0 ))
    step_rss="$(tail -n 1 "${rss_file}" 2>/dev/null)"
    [[ "${step_rss}" =~ ^[0-9]+$ ]] || step_rss=NA
    rm -f "${rss_file}" "${cores_dir}/${step_pid}"
    step_pid=""
    return "${rc}"
}

claim=""
step_pid=""
step_desc=""
kill_tree() {  # kill_tree <pid>: its descendants, then itself
    local c
    for c in $(pgrep -P "$1"); do kill_tree "${c}"; done
    kill -TERM "$1" 2> /dev/null
}
on_signal() {
    trap - TERM INT HUP
    log "signal: stopping the step (no marker is written for it)"
    local c
    for c in $(pgrep -P $$); do kill_tree "${c}"; done
    wait
    exit 143
}
release_claim() {
    if [[ -n "${claim}" ]]; then rm -f "${claim}"; fi
}
trap release_claim EXIT
trap on_signal TERM INT HUP

# The steps, as 05.0:155-169 and 04.5:85-106 run them; no `set -u` there: config.sh reads variables it does not guard.
# shellcheck disable=SC2016  # expanded by the inner bash
contribs_inner='source "${REPO_ROOT}/lib/bash/config.sh" || exit 1
activate_env "${chrombpnet_env}"
gpu_env
require_gpu jax
echo "[$(date)] chrombpnet contribs_bw on GPU ${CUDA_VISIBLE_DEVICES}: -m $1 -r $2 -op $3 -pc $4 --shap-seed $5"
exec chrombpnet contribs_bw -m "$1" -r "$2" -g "${genome_fa}" -c "${chrom_sizes}" -op "$3" -pc "$4" --shap-seed "$5"'
# shellcheck disable=SC2016  # expanded by the inner bash
qc_inner='source "${REPO_ROOT}/lib/bash/config.sh" || exit 1
window="${MOTIF_QC_WINDOW:-${motif_qc_window:-400}}"
max_seqlets="${MOTIF_QC_MAX_SEQLETS:-${motif_qc_max_seqlets:-5000}}"
activate_env "${chrombpnet_env}"
echo "[$(date)] motif_qc.py: window ${window}, ${max_seqlets} seqlets, motif DB ${chrombpnet_motifs_meme}"
exec python "${src_dir}/motif_qc.py" --scores "$1" --motif-db "${chrombpnet_motifs_meme}" --output-dir "$2" \
    --prefix chrombpnet --window "${window}" --max-seqlets "${max_seqlets}" --threads "${SLURM_CPUS_PER_TASK:-4}"'

log "bias_control.sh on $(hostname), job ${SLURM_JOB_ID:-local} step ${SLURM_STEP_ID:-?}: ${ds} fold ${fold}, ${head}" \
    "head, --shap-seed ${seed}; ${model} on ${regions} -> ${out_dir}; JAX up to ${XLA_PYTHON_CLIENT_MEM_FRACTION} of" \
    "a GPU (preallocate ${XLA_PYTHON_CLIENT_PREALLOCATE}); nice ${niceness}"

# ---------------------------------------------------------------------------------------------------------------------
# contribs

if [[ -f "${h5}" && -f "${bw}" && -f "${bed}" ]] && marker_ok "${contribs_marker}" "${model}" "${regions}" "${h5}"; then
    log "contribs_bw done before ($(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["note"])' "${contribs_marker}"))"
else
    rm -f "${contribs_marker}"
    (( n_gpus_node > 0 )) || die "nvidia-smi lists no GPU"
    claim="$(claim_gpu)"
    gpu="${claim##*/}"
    gpu="${gpu%%.*}"
    contribs_log="${logs}/contribs.log"
    offset="$(stat -c %s "${contribs_log}" 2>/dev/null || echo 0)"
    log "contribs_bw on GPU ${gpu} (claim ${claim})"
    step_desc="contribs_bw (05.0's call) -m ${model} -r ${regions} -op ${prefix} -pc ${head} --shap-seed ${seed}"
    run_step "bias_contribs_${fold}" "${contribs_log}" "$(( gpu_threads > 16 ? gpu_threads : 16 ))" "${gpu_threads}" \
        "${gpu}" env DATASET_CONFIG="${config}" bash -c "${contribs_inner}" contribs_bw "${model}" "${regions}" \
        "${prefix}" "${head}" "${seed}"
    rc=$?
    rm -f "${claim}"
    claim=""
    if (( rc != 0 )) || [[ ! -f "${h5}" || ! -f "${bw}" || ! -f "${bed}" ]]; then
        reason="$(tail -c "+$(( offset + 1 ))" "${contribs_log}" | grep -E 'NotImplementedError' | tail -n 1)"
        if (( rc != 0 )) && [[ ! -f "${h5}" && -n "${reason}" ]]; then
            reason="${reason#*NotImplementedError: }"
            record "bias_contribs_${fold}" "${rc}" "rejected: ${reason}"
            status rejected "contribs_bw refused ${model} (${head} head): ${reason}"
            log "contribs_bw refused the model (${head} head): ${reason}"
            log "recorded in ${status_file}; MoDISco skipped (the plan's fallback). Exit 3."
            exit 3
        fi
        record "bias_contribs_${fold}" "${rc}" "failed"
        status failed "contribs_bw exit ${rc}; see ${contribs_log}"
        die "contribs_bw failed, exit ${rc} (${contribs_log})"
    fi
    note="${head} scores of $(wc -l < "${bed}") peaks on GPU ${gpu}"
    write_marker "${contribs_marker}" "${note}" "${model}" "${regions}" "${h5}" || die "cannot write ${contribs_marker}"
    record "bias_contribs_${fold}" 0 "${note}"
    status attributed "${note}"
    log "contribs_bw done in ${step_seconds} s: ${note}"
fi

# ---------------------------------------------------------------------------------------------------------------------
# modisco

qc_outputs=( "${qc_dir}/chrombpnet_modisco_results.h5" "${qc_dir}/chrombpnet_modisco_motifs.meme"
             "${qc_dir}/chrombpnet_motif_qc.json" "${qc_dir}/report/report.html" )
outputs_there() { local f; for f in "${qc_outputs[@]}"; do [[ -f "${f}" ]] || return 1; done; }
if outputs_there && marker_ok "${qc_marker}" "${h5}"; then
    log "MoDISco done before ($(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["note"])' "${qc_marker}"))"
else
    if [[ -e "${qc_dir}" && -n "$(ls -A "${qc_dir}" 2>/dev/null)" ]]; then
        to="${qc_dir}.incomplete.$(date +%Y%m%dT%H%M%S)"
        mv -n "${qc_dir}" "${to}" && log "moved ${qc_dir} (not done) to ${to}"
        [[ -e "${qc_dir}" ]] && die "cannot move ${qc_dir} aside"
    fi
    log "MoDISco (${qc_threads} threads)"
    step_desc="motif_qc.py (04.5's call) --scores ${h5} --output-dir ${qc_dir} --prefix chrombpnet"
    run_step "bias_modisco_${fold}" "${logs}/modisco.log" "${qc_threads}" "${qc_threads}" "" \
        env DATASET_CONFIG="${config}" bash -c "${qc_inner}" motif_qc "${h5}" "${qc_dir}"
    rc=$?
    if (( rc != 0 )) || ! outputs_there; then
        record "bias_modisco_${fold}" "${rc}" "failed"
        status failed "motif_qc.py exit ${rc}; see ${logs}/modisco.log"
        die "motif_qc.py failed, exit ${rc} (${logs}/modisco.log)"
    fi
    note="$(python3 -c 'import json, sys
s = json.load(open(sys.argv[1]))
print("%s pos, %s neg patterns" % (s.get("pos_patterns"), s.get("neg_patterns")))' "${qc_dir}/chrombpnet_motif_qc.json")"
    write_marker "${qc_marker}" "${note}" "${h5}" || die "cannot write ${qc_marker}"
    record "bias_modisco_${fold}" 0 "${note}"
    log "MoDISco done in ${step_seconds} s: ${note}"
fi
status "done" "$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["note"])' "${qc_marker}")"
log "bias control done: ${qc_dir}"
