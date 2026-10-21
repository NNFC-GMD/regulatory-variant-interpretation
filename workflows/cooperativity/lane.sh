#!/bin/bash
# lane.sh <gpu> <cpus> <log dir> <args file>...: coop_isa.py once per args file, in turn, on one GPU.
#
#   <gpu>        the CUDA device to use (CUDA_VISIBLE_DEVICES)
#   <cpus>       the CPUs to pin to, as taskset takes them (e.g. 32-63); OMP_NUM_THREADS is 8
#   <log dir>    each run's output goes to <log dir>/<name>.log, <name> being its args file's name without .args
#   <args file>  coop_isa.py's arguments for one dataset, one per line (blank lines and #-comments skipped)
#
#   COOP_PYTHON  the python to run (default: python); it needs torch and cherimoya, as pixi's cherimoya env has
#
# Several lanes run side by side, one per GPU, e.g. inside a running allocation:
#   srun --jobid <job> --overlap -N1 -n1 lane.sh 2 32-63 <log dir> a.args b.args > <log dir>/lane_2.log 2>&1 &
set -uo pipefail
(( $# >= 4 )) || { sed -n '2,11p' "$0" >&2; exit 2; }
gpu=$1; cpus=$2; logs=$3; shift 3
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
py="${COOP_PYTHON:-python}"
mkdir -p "${logs}"
rc_all=0
for f in "$@"; do
    name="$(basename "${f}" .args)"
    mapfile -t args < <(grep -v -e '^[[:space:]]*$' -e '^[[:space:]]*#' "${f}")
    echo "[$(date '+%F %T %Z')] ${name} on GPU ${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" OMP_NUM_THREADS=8 taskset -c "${cpus}" "${py}" "${here}/coop_isa.py" "${args[@]}" \
        > "${logs}/${name}.log" 2>&1
    rc=$?
    (( rc == 0 )) || rc_all=1
    echo "[$(date '+%F %T %Z')] ${name} exit ${rc} $(tail -1 "${logs}/${name}.log" | cut -c1-200)"
done
exit "${rc_all}"
