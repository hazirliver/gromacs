#!/usr/bin/env bash
# In-process replay of the NB force kernel and its variants (THROWAWAY experiment build only, see
# bench/profiling/experiments/gpukern-suite.toml): runs the production command with
# GMX_EXP_NB_REPLAY, optionally with the SM clock locked (sudo nvidia-smi -lgc; reset afterwards).
#   bench/profiling/nb_replay.sh OUTDIR BUILD_DIR TPR [CLOCK_MHZ|none] [first:every:count:reps]
# With SPREAD_REPLAY=first:every:count:reps also replays the PME spread kernel (GMX_EXP_SPREAD_REPLAY,
# output OUTDIR/spread_replay.txt); NB_REPLAY=0 disables the NB replay.
set -u
out=${1:?}; bld=${2:?}; tpr=${3:?}; clk=${4:-none}; spec=${5:-1100:150:8:30}
mkdir -p "$out"; out=$(realpath "$out"); tpr=$(realpath "$tpr")
rm -f "$out/replay.txt" "$out/spread_replay.txt"
if [ "$clk" != none ]; then sudo nvidia-smi -lgc "$clk,$clk" > "$out/clock.txt" 2>&1; fi
nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,clocks_throttle_reasons.active --format=csv -lms 200 \
    > "$out/telemetry.csv" 2>&1 &
smi=$!
envs=(GMX_MAXBACKUP=-1)
[ "${NB_REPLAY:-1}" != 0 ] && envs+=("GMX_EXP_NB_REPLAY=$spec:$out/replay.txt")
[ -n "${SPREAD_REPLAY:-}" ] && envs+=("GMX_EXP_SPREAD_REPLAY=$SPREAD_REPLAY:$out/spread_replay.txt")
( cd "$out" && env "${envs[@]}" "$bld/bin/gmx" -quiet mdrun -s "$tpr" \
    -deffnm run -noconfout -nsteps 2600 -resetstep 500 -ntmpi 1 -ntomp 16 -nb gpu -pme gpu -bonded gpu \
    -update gpu -nstlist 200 -pin on -notunepme > mdrun.out 2>&1 )
rc=$?
kill $smi 2>/dev/null
if [ "$clk" != none ]; then sudo nvidia-smi -rgc >> "$out/clock.txt" 2>&1; fi
echo "rc=$rc $(grep -c '^replay' "$out/replay.txt" 2>/dev/null) replay lines"
