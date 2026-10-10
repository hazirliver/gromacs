#!/usr/bin/env bash
# Run every capture that needs the GPU performance counters inside one short DCGM-pause window.
#   bench/profiling/counters_window.sh OUTDIR LINEINFO_BUILD NVTX_BUILD TPR [mdrun args...]
# DCGM holds the counters on the benchmark node; `dcgmi profile --pause` frees them (works from the user
# account). It pauses the node's DCGM GPU metrics, so ASK THE NODE OWNER FIRST. DCGM is resumed by an EXIT
# trap, also when a capture fails. The 2026-10-10 window took 2 minutes.
# Captures (ncu: --set full, kernel replay, --cache-control all, natural clocks unless noted):
#   ncu/plain-none     every plain-step kernel, ~2.5 steps after LAUNCH_SKIP matching launches (default 5000)
#   ncu/prune-fresh    the fresh-list prune of the 3rd search step (step 400)
#   ncu/nb-vf          the energy-step NB kernel (step 300)
#   ncu/main-base      NB + spread kernels at ncu's base clock
#   nsys/S-gpumetrics-100000  GPU-metrics timeline (10 us sampling) of 2000 steps, for gpumetrics_phase.py
# Note: --cache-control all flushes the caches before every replay pass: DRAM traffic and L2 hit rates of
# small kernels are cold-cache values (in production the 5 MB PME grid and the state stay in the 96 MB L2).
set -u
out=$(realpath -m "${1:?OUTDIR}"); L=$(realpath "${2:?LINEINFO_BUILD}"); S=$(realpath "${3:?NVTX_BUILD}"); TPR=$(realpath "${4:?TPR}")
shift 4
ARGS=("$@")
[ ${#ARGS[@]} -eq 0 ] && ARGS=(-ntmpi 1 -ntomp 16 -nb gpu -pme gpu -bonded gpu -update gpu -nstlist 200 -pin on -notunepme)
SKIP=${LAUNCH_SKIP:-5000}
mkdir -p "$out/ncu" "$out/nsys"
log=$out/counters_window.log
exec > >(tee -a "$log") 2>&1

resume() { echo "# $(date -Is) resuming DCGM profiling"; dcgmi profile --resume; }
trap resume EXIT
echo "# $(date -Is) pausing DCGM profiling"
dcgmi profile --pause || exit 1

PLAIN='nbnxn_kernel_ElecEw_VdwLJFsw_F_cuda|pme_spline_and_spread|pme_gather|pme_solve|regular_fft|bonded_kernel_gpu|lincsKernel|settleKernel|leapFrog|x_to_nbat|reduceKernel|nbnxn_kernel_prune_cuda'
run_ncu() {  # name clock regex skip count nsteps
    local name=$1 clock=$2 regex=$3 skip=$4 count=$5 nsteps=$6
    echo "# $(date -Is) ncu $name (clock $clock, skip $skip, count $count)"
    mkdir -p "$out/ncu/$name.run"
    ( cd "$out/ncu/$name.run" && env GMX_MAXBACKUP=-1 timeout 2400 ncu --set full --import-source yes \
        --clock-control "$clock" --cache-control all --replay-mode kernel --kernel-name-base mangled \
        -k "regex:$regex" --launch-skip "$skip" --launch-count "$count" -f -o "../$name" \
        "$L/bin/gmx" -quiet mdrun -s "$TPR" -deffnm run -noconfout -nsteps "$nsteps" "${ARGS[@]}" > ncu.out 2>&1 )
    echo "  rc=$? $(grep -c '==PROF== Profiling' "$out/ncu/$name.run/ncu.out") kernels profiled"
    rm -f "$out/ncu/$name.run"/*.trr "$out/ncu/$name.run"/*.xtc "$out/ncu/$name.run"/*.edr "$out/ncu/$name.run"/*.cpt
}

run_ncu plain-none none "$PLAIN" "$SKIP" 42 340
run_ncu prune-fresh none 'nbnxn_kernel_prune_cudaILb1E' 2 1 420
run_ncu nb-vf none 'VdwLJFsw_VF_cuda' 3 1 320
run_ncu main-base base 'nbnxn_kernel_ElecEw_VdwLJFsw_F_cuda|pme_spline_and_spread' 600 4 340

for f in 100000 20000; do
    d=$out/nsys/S-gpumetrics-$f; mkdir -p "$d"
    echo "# $(date -Is) nsys gpu metrics at $f Hz -> $d"
    ( cd "$d" && env GMX_MAXBACKUP=-1 timeout 1200 nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
        --gpu-metrics-devices=0 --gpu-metrics-frequency=$f \
        --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true -o prof \
        "$S/bin/gmx" -quiet mdrun -s "$TPR" -deffnm run -noconfout -nsteps 3000 -resetstep 1000 "${ARGS[@]}" \
        > mdrun.out 2>&1 ) && nsys export --type=sqlite --force-overwrite=true -o "$d/prof.sqlite" "$d/prof.nsys-rep" > /dev/null 2>&1 \
        && { echo "  ok"; break; }
    echo "  failed: $(tail -3 "$d/mdrun.out")"
done
echo "# $(date -Is) done"
