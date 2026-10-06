#!/usr/bin/env bash
# Nsight Systems capture of mdrun's steady state.
#   bench/profiling/nsys_capture.sh OUTDIR BUILD_DIR TPR NSTEPS RESETSTEP [mdrun args ...]
# Environment:
#   NSYS_OPTS   extra/override nsys options (default: full set below)
#   MDRUN_ENV   space-separated K=V pairs exported for mdrun only (e.g. "GMX_CUDA_GRAPH=1")
#   NSYS_LIGHT  =1: only cuda+nvtx tracing (overhead comparison)
#
# The capture range is mdrun's counter reset: at -resetstep GROMACS calls cudaProfilerStart()
# when it detects a CUDA profiler (NSYS_PROFILING_SESSION_ID), and cudaProfilerStop() at the end
# of the run, so exactly the steps RESETSTEP..NSTEPS are recorded (no start-up, no PME tuning).
set -euo pipefail
out=${1:?}; bld=${2:?}; tpr=${3:?}; nsteps=${4:?}; reset=${5:?}; shift 5
mkdir -p "$out"; out=$(realpath "$out"); tpr=$(realpath "$tpr"); bld=$(realpath "$bld")
if [ "${NSYS_LIGHT:-0}" = 1 ]; then
    opts=(--trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node)
else
    opts=(--trace=cuda,nvtx,osrt --sample=process-tree --sampling-period=500000 --backtrace=none --cpuctxsw=process-tree
          --cuda-graph-trace=node --gpu-metrics-devices=0 --gpu-metrics-frequency=20000)
fi
# shellcheck disable=SC2206
[ -n "${NSYS_OPTS:-}" ] && opts+=(${NSYS_OPTS})
envs=(GMX_MAXBACKUP=-1)
# shellcheck disable=SC2206
[ -n "${MDRUN_ENV:-}" ] && envs+=(${MDRUN_ENV})
cmd=("$bld/bin/gmx" -quiet mdrun -s "$tpr" -deffnm run -noconfout -nsteps "$nsteps" -resetstep "$reset" "$@")
{
    echo "# $(date -Is)"
    echo "env ${envs[*]}"
    echo "nsys profile ${opts[*]} --capture-range=cudaProfilerApi --capture-range-end=stop ${cmd[*]}"
} > "$out/command.txt"
cd "$out"
# scrub inherited GMX_/OMP_ variables like gmxbench does
for v in $(env | grep -oE '^(GMX_|OMP_|GOMP_|KMP_)[A-Za-z0-9_]*' || true); do unset "$v"; done
env "${envs[@]}" nsys profile "${opts[@]}" --capture-range=cudaProfilerApi --capture-range-end=stop \
    --force-overwrite=true -o prof "${cmd[@]}" > mdrun.out 2>&1
nsys export --type=sqlite --force-overwrite=true -o prof.sqlite prof.nsys-rep > /dev/null 2>&1
grep -A2 "(ns/day)" run.log | tail -2
