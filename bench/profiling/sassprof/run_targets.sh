#!/usr/bin/env bash
# Run one short mdrun per target kernel under libsassprof.so (one kernel per process, see sassprof.cpp).
#   run_targets.sh LIB GMX TPR OUTDIR [target ...]
# Targets are names from the table below; each maps to the launch-name sequence that precedes it in a
# production step (launch order per plain step: spread, x->nbat, bonded, NB F, r2c, fft x2, solve,
# fft x2, c2r, gather, reduce, [rolling prune every 2nd step], leap-frog, LINCS, SETTLE).
# The counter reset (cudaProfilerStart) is at step 151, a plain step (not search/energy), so the first
# match selects the plain-step variant of each kernel; 400-step window with two pair-list periods.
# leap-frog cannot be collected (CUPTI reports no records although it is the first launch after enabling).
set -u
lib=$1; gmx=$2; tpr=$3; out=$4; shift 4
declare -A AFTER=(
  [nb_f]="bonded_kernel_gpuILb0ELb0E"
  [r2c]="nbnxn_kernel_ElecEw_VdwLJFsw_F_cuda"
  [fft96]="regular_fft_r2c"
  [solve]="nbnxn_kernel_ElecEw_VdwLJFsw_F_cuda;regular_fft_r2c;regular_fftI;regular_fftI"
  [c2r]="pme_solve_kernel;regular_fftI;regular_fftI"
  [gather]="regular_fft_c2r"
  [reduce]="pme_gather_kernel"
  [prune]="reduceKernel;leapFrog;lincsKernel;settleKernel;pme_spline_and_spread;x_to_nbat;bonded_kernel_gpuILb0ELb0E;nbnxn_kernel_ElecEw_VdwLJFsw_F_cuda;regular_fft_r2c;regular_fftI;regular_fftI;pme_solve_kernel;regular_fftI;regular_fftI;regular_fft_c2r;pme_gather_kernel;reduceKernel"
  [leapfrog]="nbnxn_kernel_prune_cudaILb0E"
  [lincs]="leapFrogKernel"
  [settle]="lincsKernelILb1ELb0E"
  [spread]="settleKernelILb1ELb0E"
  [xnbat]="pme_spline_and_spread"
  [bonded]="x_to_nbat"
  [nb_vf]="bonded_kernel_gpuILb1ELb1E"
)
targets=("$@")
[ ${#targets[@]} -eq 0 ] && targets=(nb_f nb_vf r2c fft96 solve c2r gather reduce prune lincs settle spread xnbat bonded)
mkdir -p "$out"
for t in "${targets[@]}"; do
  d="$out/$t"; rm -rf "$d"; mkdir -p "$d/run"
  ( cd "$d/run" && CUDA_INJECTION64_PATH="$lib" SASSPROF_OUT="$d" NVPROF_ID=sassprof SASSPROF_AFTER="${AFTER[$t]}" \
      timeout 1800 "$gmx" mdrun -s "$tpr" -ntmpi 1 -ntomp 16 -nb gpu -pme gpu -bonded gpu -update gpu -nstlist 200 \
      -notunepme -nsteps ${SASSPROF_NSTEPS:-551} -resetstep ${SASSPROF_RESETSTEP:-151} -pin on > mdrun.out 2>&1 )
  rc=$?
  fn=$(awk -F'\t' 'NR>1{print $3}' "$d/sass_metrics.tsv" 2>/dev/null | sort -u | tr '\n' ' ')
  echo "$t rc=$rc collected: ${fn:-NOTHING}"
  rm -f "$d"/run/*.trr "$d"/run/*.xtc "$d"/run/*.cpt "$d"/run/*.edr "$d"/run/*.gro "$d"/run/\#*
done
