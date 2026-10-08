#!/usr/bin/env bash
# Nsight Compute metrics for selected mdrun kernels in steady state.
#   bench/profiling/ncu_capture.sh OUTDIR NAME BUILD_DIR TPR NSTEPS KERNEL_REGEX LAUNCH_SKIP LAUNCH_COUNT [mdrun args]
# KERNEL_REGEX matches the kernel's function base name (ncu --kernel-name-base function).
# LAUNCH_SKIP counts only matching launches, so e.g. 3000 for a once-per-step kernel skips ~3000 steps.
# Settings (recorded in OUTDIR/NAME.command.txt):
#   --set full, --clock-control base (ncu default: SM/memory clocks locked to base -> use for metrics,
#   not for durations), --cache-control all (caches flushed before each replay pass), kernel replay.
# Environment: NCU_OPTS (extra options), MDRUN_ENV (K=V pairs for mdrun).
set -euo pipefail
out=${1:?}; name=${2:?}; bld=${3:?}; tpr=${4:?}; nsteps=${5:?}; regex=${6:?}; skip=${7:?}; count=${8:?}; shift 8
mkdir -p "$out/$name.run"; out=$(realpath "$out"); tpr=$(realpath "$tpr"); bld=$(realpath "$bld")
opts=(--set full --clock-control base --cache-control all --replay-mode kernel --kernel-name-base function
      -k "regex:$regex" --launch-skip "$skip" --launch-count "$count" --target-processes all)
# shellcheck disable=SC2206
[ -n "${NCU_OPTS:-}" ] && opts+=(${NCU_OPTS})
envs=(GMX_MAXBACKUP=-1)
# shellcheck disable=SC2206
[ -n "${MDRUN_ENV:-}" ] && envs+=(${MDRUN_ENV})
cmd=("$bld/bin/gmx" -quiet mdrun -s "$tpr" -deffnm run -noconfout -nsteps "$nsteps" "$@")
echo "# $(date -Is)
env ${envs[*]} ncu ${opts[*]} -o $name ${cmd[*]}" > "$out/$name.command.txt"
for v in $(env | grep -oE '^(GMX_|OMP_|GOMP_|KMP_)[A-Za-z0-9_]*' || true); do unset "$v"; done
cd "$out/$name.run"
env "${envs[@]}" ncu "${opts[@]}" -f -o "../$name" "${cmd[@]}" > ncu.out 2>&1 || { tail -20 ncu.out; exit 1; }
grep -E "==PROF==|Profiling" ncu.out | tail -3
