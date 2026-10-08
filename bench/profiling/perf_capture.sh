#!/usr/bin/env bash
# CPU profile (Linux perf) of mdrun's steady state: start mdrun, wait for its counter reset, then
# attach `perf record` for SECONDS, then let mdrun finish.
#   bench/profiling/perf_capture.sh OUTDIR BUILD_DIR TPR NSTEPS RESETSTEP SECONDS [mdrun args]
# Environment: PERF_OPTS (default "-F 1999 --call-graph dwarf,16384"), MDRUN_ENV.
set -euo pipefail
out=${1:?}; bld=${2:?}; tpr=${3:?}; nsteps=${4:?}; reset=${5:?}; secs=${6:?}; shift 6
mkdir -p "$out"; out=$(realpath "$out"); tpr=$(realpath "$tpr"); bld=$(realpath "$bld")
cd "$out"
envs=(GMX_MAXBACKUP=-1)
# shellcheck disable=SC2206
[ -n "${MDRUN_ENV:-}" ] && envs+=(${MDRUN_ENV})
for v in $(env | grep -oE '^(GMX_|OMP_|GOMP_|KMP_)[A-Za-z0-9_]*' || true); do unset "$v"; done
popts=${PERF_OPTS:--F 1999 --call-graph dwarf,16384}
cmd=("$bld/bin/gmx" -quiet mdrun -s "$tpr" -deffnm run -noconfout -nsteps "$nsteps" -resetstep "$reset" "$@")
echo "# $(date -Is)
env ${envs[*]} ${cmd[*]}
perf record $popts -p <mdrun pid> -- sleep $secs   (attached after 'resetting all time and cycle counters')" > command.txt
env "${envs[@]}" "${cmd[@]}" > mdrun.out 2> mdrun.err &
pid=$!
until grep -q "resetting all time and cycle counters" mdrun.err 2>/dev/null; do
    kill -0 $pid 2>/dev/null || { echo "mdrun exited early"; tail mdrun.err; exit 1; }
    sleep 0.2
done
# shellcheck disable=SC2086
taskset -c 39 perf record $popts -o perf.data -p $pid -- sleep "$secs" > perf.out 2>&1
wait $pid
grep -A2 "(ns/day)" run.log | tail -2
