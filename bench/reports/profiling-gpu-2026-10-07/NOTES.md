# Lab notebook: GPU kernel deep dive (2026-10-07)

Continuation of `bench/results/profiling-2026-10-06/` ("day 1"). Same node, same inputs
(`profiling-2026-10-06/raw/inputs/prod/topol.tpr`, production args `-ntmpi 1 -ntomp 16 -nb gpu -pme gpu
-bonded gpu -update gpu -nstlist 200 -pin on -notunepme`). Times UTC.

## 08:38 Start, node state
- Node quiet (load 0.08, GPU idle P8, no compute apps). Power limit 325 W (enforced 325 W).
- DCGM still running (`nv-hostengine` pid 1514). Branch `feature/profiling` @ 9722b94a25, clean.
- Environment snapshot: `raw/env/` (bench/profiling/env_snapshot.sh, P build 3bf7713a0dd56c41).

## 08:40 Which counter-free kernel tools work under DCGM (scratch tests, CUPTI 2025.3 samples)
- Nsight Compute: still "driver resource unavailable" (day 1 `raw/ncu/ncu-permission-test.txt`).
- CUPTI PM sampling (`pm_sampling` sample): `CUPTI_ERROR_HARDWARE_BUSY`.
- CUPTI PC sampling (`pc_sampling_start_stop`): `cuptiPCSamplingStart` -> `CUPTI_ERROR_UNKNOWN`
  -> **no warp-stall reasons**.
- **CUPTI SASS metrics (binary patching) works** (`sass_metrics` sample): exact per-instruction counts
  of executed warp/thread instructions, predicated-on threads, branch divergence, global sectors vs
  ideal, L1 tag lookups, shared-memory wavefronts vs ideal, local-memory sectors.
- Policy (same rules as day 1): GPU counters blocked -> admin commands go into the report; no DCGM pause.

## 08:45 Line-info build
- Throwaway worktree `/home/asokolov/Projects/gromacs-exp-gpukern` (detached at `dev` 3df7585464;
  `git diff dev v2026.4 -- src` is empty).
- Profile `cuda-nosub-lineinfo` (`bench/profiling/experiments/gpukern-suite.toml`): P flags + `-lineinfo`,
  built from the unmodified worktree -> `~/.cache/gmxbench/builds/5d0e9c44d613eed1` (L).
- **SASS of L == SASS of P**: all 364,361 instructions of all 52 cubins identical (cuobjdump -sass,
  addresses/encodings stripped; md5 049f6598...). Counts measured on L apply to P.

## 08:50-09:05 sassprof (bench/profiling/sassprof/)
- CUPTI injection library (CUDA_INJECTION64_PATH) + `run_targets.sh` + `sassanalyze.py`.
- CUPTI 2025.3 quirks found (each tested, scratch logs):
  1. activity tracing cannot be enabled together with SASS patching (`CUPTI_ERROR_NOT_COMPATIBLE`) ->
     launches counted with API callbacks instead (driver-level cuLaunchKernel callbacks see all
     launches incl. cuFFT; runtime callbacks only GROMACS' own).
  2. Only the **first function launched after `cuptiSassMetricsEnable`** is collected (two-kernel test
     in the sample: second kernel never reported), and **nothing is collected after the first flush**.
  3. `cuptiSassMetricsSetConfig` must immediately precede `Enable`: configuring at cudaProfilerStart and
     enabling later collects nothing.
  4. Enabling at the API exit of the launch that precedes the target works (sync first).
  -> one mdrun per kernel: `run_targets.sh` enables after the launch sequence that precedes the target
     in a plain step; counter reset (cudaProfilerStart via NVPROF_ID) at step 151, window 400 steps
     (two pair-list periods), counts accumulated over all launches of the kernel and divided by the
     number of launches.
  5. leap-frog cannot be collected (no records although it is the first launch after enabling).
  6. mdrun segfaults at exit under the injection (after the output is written; harmless for counts).
- Raw: `raw/sass/<target>/{sass_metrics.tsv,launches.tsv,cubins/,sassprof.log,run/}`; the `nb_vf` target
  collected the fresh-list prune kernel instead (search step precedes) -> renamed `prune_fresh`.

## 09:06 Isolated vs concurrent kernel times at a locked 2100 MHz SM clock
- `sudo nvidia-smi -lgc 2100,2100`; telemetry `raw/nsys/telemetry-2100.csv` (58/59 samples at 2100 MHz);
  reset with `-rgc` (my pkill of the sampler killed the shell before -rgc; restored by hand at 09:08,
  verified 210 MHz idle / P8 afterwards).
- `raw/nsys/iso-2100-P`: `CUDA_LAUNCH_BLOCKING=1` (every kernel alone on the GPU); `raw/nsys/conc-2100-P`:
  normal concurrency. P build, 2000 steps each (`nsys_capture.sh`, NSYS_LIGHT=1).
- Median us alone / concurrent: NB F 459.1 / 494.7; spread 95.7 / 98.7; bonded 59.4 / 64.1; gather
  33.1 / 72.0; regular_fft 6.4 / 26.1 (x4); r2c 6.2 / 19.4; c2r 6.2 / 28.6; solve 5.2 / 31.2; rolling prune
  33.1 / 45.0; LINCS 14.9 / 15.8; SETTLE 13.7 / 13.7; leap-frog 6.0 / 6.8; x->nbat 4.6 / 29.7; reduce 4.0 / 4.0.
  Concurrent step: GPU span 815.9 us/step at 2100 MHz.

## 09:10 Dynamic instruction profiles (raw/sass/analysis-iso-2100.{json,txt})
- NB F kernel (`--regions bench/profiling/sassprof/regions-nbnxm-cuda.toml`): 433.4 M warp
  instructions per launch; **IPC 3.17 per SM = 79% of the 4/clk issue limit** (isolated, 2100 MHz);
  SIMT efficiency 0.689; the cut-off branch body runs at 0.51 active lanes; global sector efficiency
  0.82; no shared-memory bank conflicts (wavefront efficiency 1.00); no local memory.
  Regions: Ewald correction 16.6%, force switch 15.6%, i-loop load+distance 13.1%, force accumulation
  10.5%, exclusion/cut-off test 8.0%, rsqrt+LJ 7.8%, jm loop + j loads 6.6%, j-reduction 5.9%, Coulomb
  5.9%, LJ table fetch 4.9%, j-packed loop 3.5%, i-reduction 1.1%, setup 0.5%.
  Body executions 4.238 M warp-iterations per launch (= MUFU.RSQ count); i-iterations with mask bit set
  4.52 M (= LDS of xqib); j-iterations 1.53 M; mask-bit tests for empty (j,i) slots 63%.
  Pairs computed inside the cut-off: 4.238 M x 32 x 0.51 = 69 M per step (~ N rho 4/3 pi rc^3 / 2 = 67 M).
- One pair-interaction body = 61 SASS instructions (read from the line-info disassembly): force switch
  16 (2 constant MOVs, r-rsw, FSETP+FSEL for max(.,0), 2 FFMA, 6 FMUL, 1 FFMA), Ewald correction 17
  (4 constant MOVs, z4, 10 FFMA, RCP, ...), LJ fetch 5 (LDS, MOV 8, 2 IMAD, LDG.64), rsqrt+LJ 8,
  accumulation 9 + BSYNC.
- PME spread: 6.7 M warp instr, **IPC 0.24**, 95.7 us alone; 16 RED.ADD.F32 per atom-thread (64 per atom
  = 11.9 M atomics per step), 9.3 sectors per warp RED (ideal 4, efficiency 0.43).
- PME gather: IPC 1.04; same 9.3-sector pattern on the grid loads.
- Bonded: IPC 0.26, 59.4 us alone; iatoms loads 20 sectors per warp load (stride 20 B), RED forces 12.
- SETTLE: IPC 0.10; coordinate loads **8 sectors per warp load (efficiency 0.125)**: 3 atoms x float3 per
  thread = 36 B stride.
- LINCS: IPC 0.20, sector efficiency 0.33, SIMT 0.78. x->nbat 0.26, reduce 0.29 (gathers by index).
- solve IPC 1.85, cuFFT 0.15-0.89 (small grids, sub-wave).

## 09:30 Throwaway experiment build (worktree gromacs-exp-gpukern, profile cuda-nosub-gpukern)
All runtime-selectable (env), so one binary serves every comparison:
- `GMX_EXP_NB_VARIANT=k`: launch NB variant k (nbnxm_cuda_kernel_exp.cu) instead of the production
  ElecEw_VdwLJFsw_F kernel. Variants (NBEXP bits, nbnxm_cuda_kernel.cuh): 0 = same as production;
  1 factored force switch; 2 Ewald polynomials in Horner form; 4 LJ row pointer hoisted per j-cluster;
  7 = 1+2+4; 5 = 1+4; knock-outs (TIMING ONLY): 8 no force switch, 16 no Ewald correction, 32 no LJ.
- `GMX_EXP_NB_REPLAY=first:every:count:reps:file`: in-process replay of the production kernel and all
  variants on the live pair list/coordinates of selected NB calls (stream synchronised, i.e. isolated),
  into a scratch force buffer; CUDA-event timing (interleaved rounds) and force comparison with the
  production kernel (rel. RMS, max abs) incl. production-vs-production (atomic-order noise floor).
- `GMX_EXP_SKIP_PME_SPREAD`, `GMX_EXP_SKIP_PME_FFT` (FFTs + solve), `GMX_EXP_SKIP_PME_GATHER`,
  `GMX_EXP_SKIP_BONDED`: TIMING-ONLY kernel skips (upper bounds with real concurrency).
- `GMX_EXP_PME_ORDERSQ`: 16 threads per atom, splines stored (the small-system path).
- `GMX_EXP_SPREAD_INTERLEAVE`: spread block atoms assigned to warps with a stride (not with ORDERSQ).
- `GMX_EXP_CONCURRENT_SETTLE`: SETTLE in its own high-priority stream, forked after leap-frog, joined
  before the coordinates-ready event (LINCS and SETTLE constrain disjoint atoms).

## 09:24 NB replay (raw/replay/lock2100, raw/replay/unlocked; X build 9194ba626fe54212 incl. variants 9-13)
- 8 replay points x 30 reps, isolated; ratio vs production kernel (median [min,max] over points), 2100 MHz:
  fsw factored 0.9730, Ewald Horner 0.9652, both (exp3) 0.9394 [0.9361, 0.9424], +LJ hoist 0.9510,
  LJ hoist alone 1.0033 (slower), 12/10 blocks per SM 1.031/1.033 (slower), knock-outs: no fsw 0.886,
  no Ewald corr 0.853, no LJ 0.718. Unlocked (2520 MHz isolated) within 0.3% of these.
- Forces vs production: noise floor (prod again) 4.9e-8 rel RMS; fsw 5.0e-8; Horner 2.6e-7.
- Horner accuracy vs exact (raw/pmecorrf-accuracy.txt): same as production Estrin (RMS 1.010e-7 vs 1.015e-7).
- First variant-profile attempt failed (relative paths in run_targets.sh after cd) -> fixed (realpath), re-queued.

## 09:27-09:50 End-to-end (raw/exp/gpukern-e2e, 5 rounds, vs P)
- X-base 1.0020 [0.9979, 1.0060] (experiment build neutral); fsw 1.0109; Horner 1.0123; **both 1.0257
  [1.0211, 1.0303]** (238.63 ns/day, 109.3 kJ/ns = -3.6% energy); both+hoist 1.0201; +mb12 1.0005;
  prod mb12 0.9827; spread interleave 1.0007 (no change); PME OrderSquared 0.9615 (slower).
- settle-conc: all 5 runs segfaulted at teardown (rc -11 after the last step): SettleGpu outlived the
  stream it uses (member destruction order) -> fixed (settleGpu_.reset() first in ~Impl), re-run later.

## 09:50-09:57 Timing-only skips (raw/exp/gpukern-skips, 3 rounds, vs X-base 0.7441 ms/step)
- skip all PME 1.2235 (0.6082 ms/step: PME kernels cost 136 us/step = 18.3%); skip spread 1.1226
  [1.0999, 1.1457] (81 us/step: ~91% of the spread kernel's time is on the critical path); skip gather
  1.0361 (27 us); skip FFT+solve 1.0327 (24 us); NB without force switch (ko8) 1.0646 (45 us/step for a
  ~51 us kernel saving). skip-bonded: blew up ("badly or non-equilibrated" fatal error) in all 3 runs.

## 09:57 Rebuild X (same dir) with: SETTLE teardown fix, GMX_EXP_SETTLE_STAGED (coalesced smem staging),
  GMX_EXP_SPREAD_REPLAY (spread on live data: production order / interleaved / sorted by coarse spatial
  cell / sorted+interleaved), GMX_EXP_BONDED_STREAM (day-1 bonded-stream patch made a runtime switch).
  Count-only variant profiles ran concurrently with the build (counts are timing-independent).

## 10:08 Spread replay (raw/replay/spread-lock2100, spread-unlocked)
- vs production order (2100 MHz): interleave 0.9937, cell-sorted 0.9883, sorted+interleave 0.9563.
  Locality / same-address conflicts are not spread's limiter (hypothesis refuted).
## 10:10-10:28 Round 2 (raw/exp/gpukern-e2e2, vs P, 5 rounds); XO = worktree + -Xcompiler=-fopenmp
  (build 1093125cdb89aee7)
- X-base 0.9963 [0.9928, 0.9998] (round 1: 1.0020 -> X == P within +-0.4%); settle-conc 1.0017 (no
  change); settle-staged 1.0044 [1.0008, 1.0079]; bonded-stream 1.0048 [1.0010, 1.0086];
  nb3+settle-conc 1.0295; nb3+settle-conc+bstream 1.0370; XO-all (+OpenMP) 1.0484 [1.0444, 1.0524];
  **XO-all -ntomp 20 1.0570 [1.0532, 1.0607] = 0.7010 ms/step, 246.53 ns/day, 107.3 kJ/ns**; P-t20 1.0050.
## 10:29 nsys light captures (raw/nsys/{X-base,X-nb3,XO-all,X-skipspread}-light)
- NB kernel p50 449.9 (X-base) -> 429.6 us (nb3): -20.3 us in production (vs -6.1% isolated).
- XO-all: NB 441.3 (bonded kernel now concurrent in stream 24); LINCS p50 31.7 us (vs 14.3) and SETTLE
  16.0 (vs 12.5) when run concurrently -> they slow each other; explains settle-conc ~0.
- skip-spread: x->nbat 8.1 us (vs 26.9: no longer waits behind spread for SMs); GPU span -94 us/step.
## 10:29 Spread knock-out replay (raw/replay/spread-ko-lock2100; X rebuilt with GMX_EXP spread modes)
- splines only 0.0831 (8.2 us); stores instead of atomics 0.8164; stores + sorted 0.7011.
  -> 92% of spread is the grid update; atomics 18%, ordering small: cost = number of global updates.

## 10:30-10:38 Round 3 (raw/exp/gpukern-e2e3, vs P at -ntomp 16)
- XO-all-t20 1.0592 [1.0569, 1.0615] (reproduces round 2); XO-all+staged-t20 **1.0719 [1.0658, 1.0780]**
  (0.6930 ms/step, 249.36 ns/day, 107.06 kJ/ns); XO-nb3+bstream+staged-t20 1.0656.
## 10:38 Quality gate of the final stack (raw/gmxbench/quality-XO-final)
- First gate (without SETTLE staging) stopped after round 3 showed staging helps; restarted with all four
  GMX_EXP_* switches on XO: 63 cases / 546 jobs: IDENTICAL 78 (72/72 reprod-by-design), EQUIVALENT 67
  (f rel RMS 0.52-1.67x noise), SKIPPED 37 (gpu-resident unsupported for non-PME systems, both builds),
  grompp 63 IDENTICAL; no DIFFERENT/FAIL.
## 10:45 Staged SETTLE profile (raw/sass-variants/settle-staged): 0.37 M global sectors/launch vs 1.84 M
  (efficiency 0.69 vs 0.13), 0.83 M vs 0.40 M warp instructions.
## 10:45 Prior art (raw/prior-art-gpu.md, sub-agent, GitLab project id 17679574): factored force switch
  only in HIP (!4600), removed on main (!6195); spread smem tile: !6000 (open, HIP only); nothing on
  pmeCorrF cost, SETTLE coalescing, concurrent LINCS/SETTLE, cuFFT callbacks.
## 10:50 Deliverables written; summary.json/csv 1010 records (summarize.py collect now also labels
  gpukern-* records as throwaway + gate status). Throwaway worktree left in place:
  /home/asokolov/Projects/gromacs-exp-gpukern (patch bench/profiling/experiments/exp-gpukern.patch).
  GPU state: clocks unlocked (nvidia-smi -rgc after each locked run), power limit 325 W untouched.
