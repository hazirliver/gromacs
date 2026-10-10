# Performance optimisation of this fork: start here

For anyone (person or agent) continuing the performance work on this GROMACS fork. It says where things
stand, what was measured and rejected, what is still open, which tools exist and what to watch out for.
**Keep it current: add every new result, positive or negative, to the tables below.**

Last updated 2026-10-10 (iteration-3 profiling: hardware counters, hypotheses; nothing new implemented).

## 1. State

* `dev` = GROMACS 2026.4 release + gmxbench (`bench/`) + five GPU-path optimisations + the profiling tools
  (`bench/profiling/`) + the reports (`bench/reports/`).
* Target workload: **MAS1** (`bench/suites/target.toml`, input in `data/`, never committed), 185,486 atoms,
  membrane protein complex, CHARMM36 with LJ force switch, PME, 2 fs, h-bond constraints, one NVIDIA L40S.
  Production command (best by `gmxbench sweep`, re-checked after the changes):
  `gmx mdrun -ntmpi 1 -ntomp 16 -nb gpu -pme gpu -bonded gpu -update gpu -nstlist 200`
* MAS1 production: **233.0 -> 248.3 ns/day (+6.5% [6.3, 6.8])**, GPU energy per simulated ns -5.1%.
  Details and the other benchmark systems: [`reports/2026-10-gpu-kernel-opt.md`](reports/2026-10-gpu-kernel-opt.md).

| Commit | Change | MAS1 production | Results |
|---|---|---:|---|
| `2e3e86d3a1` | NB kernel: factored LJ force switch, Horner-form Ewald correction (`nbnxm/nbnxm_kernel_utils.h`) | +2.9% | FP order of GPU NB kernels changes (~1e-7 rel.) |
| `53f84025dd` | SETTLE: coalesced shared-memory staging (`mdlib/settle_gpu_internal.cu`) | +0.9% | bitwise identical |
| `e1971ae650` | CMake: OpenMP host flags for nvcc-compiled files with CMake < 3.31 | +1.3% | unchanged |
| `e2546053cf` | GPU bonded kernel in its own stream, joined after the NB launch | +1.0% | within GPU noise |
| `da0eed4161` | SETTLE concurrent with LINCS (when their atoms are disjoint) | +0.6% | bitwise identical |

## 2. Read in this order

1. This file.
2. [`TESTING.md`](TESTING.md) (the required validation procedure) and the repository's `CLAUDE.md`.
3. [`reports/2026-10-gpu-kernel-opt.md`](reports/2026-10-gpu-kernel-opt.md): results of the merged work.
4. [`reports/profiling-2026-10-10/`](reports/profiling-2026-10-10/): **profile of the current `dev`** (after
   the five commits) with the first hardware counters (Nsight Compute, GPU-metrics timeline). `HYPOTHESES.md`
   there is the current ranked to-do list with bounds and first experiments; `ANALYSIS.md` has the numbers;
   `NEXT-STEPS.md` is the hand-off plan for implementing and validating them.
5. [`reports/profiling-gpu-2026-10-07/`](reports/profiling-gpu-2026-10-07/): GPU kernel deep dive.
   `EXEC-SUMMARY.md` first, then `OPPORTUNITIES.md` (ranked, with bounds), `ANALYSIS.md` (per-kernel
   instruction profiles, critical path), `NOTES.md` (lab notebook), `NEXT-STEPS.md` (the hand-off plan that
   produced the five commits; section 9 is the final status).
6. [`reports/profiling-2026-10-06/`](reports/profiling-2026-10-06/): whole-step profile of the production
   run (CPU and GPU, pair search, CMAP, power cap, run settings), same file structure.
7. [`reports/phaseB-energy-2026-10-07/REPORT.md`](reports/phaseB-energy-2026-10-07/REPORT.md): NVE energy
   conservation before/after, and the `lincs-iter` finding.

The session reports are copies of write-ups made on the benchmark node. They refer to `raw/...` (logs, nsys
captures, tprs, summary.json) and to `bench/results/...`: those stay on that node (git-ignored; the tprs are
customer-derived). Where they say "branch `feature/profiling`", the tools are now in `bench/profiling/` on
`dev`. Numbers in them describe the code before the five commits unless stated otherwise.

## 3. Where MAS1 time goes (current `dev`, profiled 2026-10-10)

Source: [`reports/profiling-2026-10-10/ANALYSIS.md`](reports/profiling-2026-10-10/ANALYSIS.md). The numbers
before the five commits are in the 2026-10-06/07 reports.

* Mean step 709 µs (nsys); plain steps 87% (628 µs median), pair-search steps **10.3%**, energy steps 1%.
* Plain step on the GPU:

  | µs from step start | what runs | SM issue utilisation |
  |---|---|---|
  | 0-80 | PME spread alone, holding 94% of the warp slots | **5%** |
  | 80-140 | x→nbat, bonded, r2c; the NB kernel starts at 134 µs although its input is ready at 92 µs | 12-18% |
  | 140-540 | NB kernel | 80-87% |
  | 540-580 | NB tail | 48% → 19% |
  | 580-640 | update chain + rolling prune | 17-26% |

  DRAM traffic ~0: the working set lives in the 96 MB L2. CPU launches run ~400 µs ahead of the GPU (never
  launch-bound).
* **NB kernel** (442 µs/step, ~1:1 on the critical path): issue-bound. IPC 3.24/4, "not selected" the top
  stall, occupancy at the 64-register limit, L2 20%. It executes 406 M warp instructions per step = 95.6
  per pair body (body 53-54). 23% of the instructions are control flow (4 per i-cluster mask test,
  because of the uniform-register mask) and 8% are `MOV`s.
* **PME spread** (91 µs, 91% critical): **L2-throughput-bound (87% of peak)**, IPC 0.23, LSU queue full of
  `RED` atomics (11.9 M per step). Bonded (61 µs, overlapped): memory-instruction-bound. LINCS: 0.18 waves,
  17% occupancy.
* **Pair search**: 9.5 ms per search on 16 balanced threads, GPU idle 11.9 ms per search. 40% is a
  bounding-box test with 128-bit SIMD plus a scalar bit loop (the build is AVX-512), 14% exclusion masking
  by binary search.
* Energy/COM steps ~1%, CPU CMAP ~1% (hidden; it disables CUDA graphs).
* Still true: the GPU is power-capped at an enforced 325 W, so work reductions show up both as speed and
  as lower energy per ns.

## 4. Open opportunities (ranked by expected gain per effort)

Details, bounds and first experiments:
[`reports/profiling-2026-10-10/HYPOTHESES.md`](reports/profiling-2026-10-10/HYPOTHESES.md).

| Opportunity | Bound (measured) | Estimate | Effort |
|---|---|---|---|
| Fix the MAS1 input (interlocked rings) and use HMR at 4 fs | lysozyme: 1.7x ns/day at 4 fs | ~1.5-1.8x | input work + science decision, no code |
| **H1** NB pair loop: (a) mask test on a once-per-j-cluster shifted mask (bit-identical), (b) Ewald correction as a monic (5,4) rational in r² with host-scaled coefficients (2x more accurate than today), (c) force-switch coefficients pre-combined per type pair (`LDG.128`) | static SASS × measured counts: −4.1% / −4.2% / −6.2%, ~−14.5% of NB instructions together, ≤ 64 registers | +6-7% | S-M |
| **H2** PME spread with shared-memory accumulation in int32 fixed point (float shared atomics are CAS loops on sm_89), one global atomic per touched grid point | spread skip 12.3%; global updates −2.3x (current order) / −3.9x (spatial) / −4.6-5.7x (tiles) | +6-8% (spatial) / +3-4% (current order) | L / M |
| **H3a** Launch the NB kernel before the bonded kernel | NB start delay 42 µs | +1.5-3% | S |
| **H3b** Co-schedule spread with NB (NB first, spread residency capped) | shared with H2 | +3-6% | M |
| **H4** Overlap the CPU pair search with GPU steps (lagged, double-buffered list) | 8.4% (GPU idle per search) | 5-7% | L |
| **H5** AVX-512 bounding-box test → mask, exclusions without bisection | ~half of the search | +2-3% | M |
| H6 NB tail (SMs drain 77% → 48%) | ~3.4% | 1-2% | M-L |
| H7 Update chain: LINCS fuller launches, prune not in front of LINCS, fuse reduce + leap-frog | ~4% | 1-2% | M |
| H8 PME gather tile-staged; solve in cuFFT LTO callbacks | 3.5% + 3.2% | 1-2% | M-L |
| H9 Bonded: iatoms SoA + shared-memory force accumulation | ~1.6% | ~1% | M |
| `-ntomp 20` instead of 16 (run argument) | - | +0.5-0.6% (old code) | minutes |
| SETTLE staging only for larger systems (fixes -0.7% on a 5k-atom water box with CUDA graphs) | 0.7% on tiny systems | - | S |
| NB cluster-pair efficiency (14-18 of 32 lanes active in the pair body) | ~30% of NB kernel (theoretical) | research | L+ |

Not a speedup but recommended for the long MAS1 run: `lincs-iter = 2` cuts the NVE energy drift 3.3-fold for
0.2% speed (`phaseB-energy-2026-10-07/REPORT.md`).

## 5. Tried and rejected (measured; do not repeat without a new idea)

| Idea | Result | Source |
|---|---|---|
| NB kernel launch bounds for more occupancy (20 / 24 blocks per SM) | -4.0% / -16.4% (spills, lower clocks under the power cap) | 2026-10-06 |
| More registers for the NB kernel (12 / 10 blocks per SM) | kernel 1.031 / 1.033 slower; end to end -1.7% | 2026-10-07 |
| Hoisting the LJ parameter row pointer out of the i-loop | kernel 0.3% slower, +24 instructions | 2026-10-07 |
| Swapping stream priorities (NB local high, PME normal) | -0.57% | 2026-10-06 |
| Porting CMAP to the GPU alone (without CUDA graphs) | no change; CMAP + graphs ceiling +0.9% | 2026-10-06 |
| PME spread with 16 threads per atom (`OrderSquared`) | -3.9% end to end | 2026-10-07 |
| Interleaving spread atoms over warps | kernel time -0.5%, end to end no change | 2026-10-07 |
| Spatially sorting the spread atoms alone (without shared-memory accumulation) | kernel time only -1.2 to -1.6% | 2026-10-07 |
| LINCS concurrent with SETTLE without the SETTLE staging | no change (LINCS needs a second wave) | 2026-10-07 |
| nstlist other than 200 | 150: -0.5%, 300: -2.4%, 100: -3.2% | 2026-10-06, sweep 2026-10-08 |
| nstcalcenergy alone | no change | 2026-10-06 |
| Larger rc with a coarser PME grid | -6% (1.25) to -20% (1.40); the tuner picks 1.20 | 2026-10-06 |
| Tabulated Ewald, twin cut-off, no list splitting, other prune intervals | slower or no change | 2026-10-06 |
| Pinning / thread placement, hyperthreads (24/32 threads) | no effect / -0.7%, +0.2% | 2026-10-06 |
| Raising the GPU power limit | impossible: enforced 325 W | 2026-10-06 |
| Lower clocks for energy | -2.4% energy for -2.3% speed (trade-off) | 2026-10-06 |
| CUDA graphs / PME tuning / 2 or 4 simulations per GPU (sweep on the new code) | +0.05% (graphs unused: CMAP on the CPU) / -0.3% / -3.5% and -4.2% aggregate | sweep 2026-10-08 |
| Hand-written SASS/PTX for the NB kernel | analysis: with H1 the body has no wasted instruction; limits are issue slots and the 64-register cap; no supported sm_89 assembler | 2026-10-10 |
| Tensor cores (TF32/FP16) for NB distances, spread or FFT | analysis: 1e-3 precision fails the quality policy; 3xTF32 removes the gain | 2026-10-10 |
| Vector float atomics (`RED.F32x4`) for spread | not available on sm_89 (sm_90+ only; nvcc rejects float2/float4 `atomicAdd`) | 2026-10-10 |
| Lower-degree Ewald correction ((4,4), (3,4)) or a pure polynomial | 1.3-5x production's error relative to 1/r³; polynomial needs degree ≥ 8 | 2026-10-10 |

## 6. Tools

**gmxbench** (`bench/bin/gmxbench`, see [`README.md`](README.md)): the A/B gate for every source change
(quality, perf, micro, sweep, upstream ctest). Results go to `bench/results/<session>/` (git-ignored).

**Profiling tools** (`bench/profiling/`, Python tools run with the gmxbench venv after `bench/setup.sh`):

| Tool | Purpose |
|---|---|
| `env_snapshot.sh OUTDIR [BUILD...]` | record versions and machine state (read-only) |
| `mkinputs.py` | create the MAS1 profiling tprs (production and ablation variants) under the session's `raw/inputs/` |
| `prun EXPERIMENT.toml --out DIR [--repeats N] [--only GLOB]` | interleaved, seeded-random repeated mdrun of many variants (build x tpr x args x env), NVML telemetry; specs in `experiments/*.toml` (paths in them point to the node's build dirs and tprs) |
| `summarize.py exp DIR [--ref VARIANT]` / `summarize.py collect RESULTS_DIR` | statistics for a prun experiment (ratios vs a reference variant); session-wide `summary.json` / `.csv` |
| `nsys_capture.sh OUTDIR BUILD TPR NSTEPS RESETSTEP [args]` | Nsight Systems capture of the steady state (`NSYS_LIGHT=1` for cuda+nvtx only) |
| `nsys_steps.py` | per-kernel, per-copy, per-step and GPU-idle analysis of a capture; `budget.py` (step-time budget), `occupancy.py` (theoretical occupancy) build on it |
| `clockscale.py` | counter-free limiter classification from runs at locked SM clocks |
| `perf_capture.sh` | Linux perf CPU profile of the steady state |
| `sassprof/` (`build.sh`, `run_targets.sh`, `sassanalyze.py`) | CUPTI SASS-metric injection: exact per-instruction counts, divergence, sector efficiency per kernel; works while DCGM holds the counters; one kernel per mdrun process |
| `nb_replay.sh` | in-process replay of NB / spread kernel variants on live data; needs a throwaway build with the replay hooks from `experiments/exp-gpukern.patch` |
| `ncu_capture.sh` | Nsight Compute for chosen kernels; only works if DCGM profiling is paused (admin) |
| `pcie_probe.cu` | PCIe latency/bandwidth probe |
| `queue.sh FILE LOG` | run a list of commands sequentially with timestamps |
| `counters_window.sh OUTDIR LINEINFO_BUILD NVTX_BUILD TPR` | every counter-based capture in one DCGM-pause window (ncu `--set full` on all plain-step kernels, fresh prune, VF kernel, base-clock NB/spread; nsys GPU metrics at 10 µs); resumes DCGM on exit; **ask the node owner before each pause** |
| `ncu_summary.py REP` | per-kernel throughput, limiter, occupancy, cache, pipe and stall summary of an ncu report |
| `sass_hot.py SOURCE.csv` | per-SASS-instruction executed counts and stall samples (`ncu --page source --print-source sass --csv`), basic blocks by frequency, the pair-interaction body |
| `gpumetrics_phase.py PROF.sqlite` | nsys GPU metrics (SM active/issue, warps, DRAM) folded onto the step phase with the kernel schedule |
| `launch_gaps.py PROF.sqlite` | per kernel of a plain step: CPU launch call vs GPU start (launch-bound or not) |
| `nbvariants/` (`mkvariant.py`, `compile.sh`, `seg.py`, `weigh.py`) | static "what-if" SASS of NB kernel source variants with the production nvcc flags, weighted by measured execution counts; no GROMACS build or run |
| `spread_tiles.py GRO NX NY NZ` | exact global-update and sector counts of PME spread organisations (block-local / spatial / tiled shared-memory accumulation) for given coordinates |
| `pmecorr_fit.py`, `pmecorr_weighted.py` | minimax rational fits of the Ewald correction on the range a given `ewald-rtol` uses, fp32-emulated, in the kernel's r² form |
| `records.py`, `derived_records.py`, `gpu_records.py` | turn analysis outputs into `summary.json` records |

**Prototype patches** (`bench/profiling/experiments/`): `exp-throwaway.patch` (2026-10-06) and
`exp-gpukern.patch` (2026-10-07) contain every prototyped variant behind `GMX_EXP_*` runtime switches,
including the rejected ones and the kernel replay harness. They apply to commit `3df7585464` in a throwaway
worktree; never merge them.

## 7. Environment facts and gotchas

* **DCGM holds the GPU performance counters** on the node: Nsight Compute, nsys GPU metrics and CUPTI PM/PC
  sampling fail; CUPTI SASS metrics work (`sassprof`). `dcgmi profile --pause` / `--resume` **works from the
  user account** (2026-10-10). It leaves a gap in the node's DCGM metrics, so ask the node owner each time and
  keep the window short (`counters_window.sh`: 2 minutes, resume in an EXIT trap).
* **mdrun's performance line is wrong under an nsys capture range** (`--capture-range=cudaProfilerApi`): it
  includes nsys's flush at `cudaProfilerStop` (2-3 ms/step reported, "Rest" ~70%). Use the trace's step
  periods (`nsys_steps.py`).
* ncu's `--cache-control all` (used by `ncu_capture.sh` / `counters_window.sh`) flushes caches per replay
  pass: small kernels then look DRAM-bound. In production the whole step lives in the 96 MB L2.
* On sm_89, shared-memory `atomicAdd(float*)` is a CAS spin loop (`ATOMS.CAST.SPIN`); int32 is native
  (`ATOMS.ADD`). Global float `RED` is native; vector float atomics need sm_90.
* **Power cap**: enforced 325 W, cannot be raised; `sw_power_cap` is active all the time.
* **Re-running CMake in an nvcc build directory drops `-D_FORCE_INLINES`** (upstream
  `gmxManageNvccConfig.cmake` adds it only on the first configure). `gmxbench upstream` reconfigures in
  place. Rebuild from a fresh configure before a perf A/B after any CMake change.
* Builds whose source/build paths differ in length differed by ~4% in an A/A test: use gmxbench builds
  (fixed-length paths) for every comparison.
* Before commit `c651203516`, `-B WORKTREE` used one build directory for every checkout, so running
  gmxbench from a second checkout (git worktree) silently built the first checkout's sources. Results from
  the main checkout, git refs and `src:` specs were not affected. gmxbench now keys WORKTREE builds by path
  and refuses a build directory configured for another source tree.
* CPU `-reprod` runs are bitwise reproducible (also with DD and PME ranks); GPU runs never are and are judged
  against their own noise.
* mdrun keeps `nstlist` at 20 for NVE (`-nstlist` is ignored), so NVE runs are slower than production.
* MAS1: CMAP runs on the CPU, so CUDA graphs are not used. The input has interlocked aromatic rings: stable
  at 2 fs, but 4 fs / HMR blows up until fixed.
* `data/` and every tpr derived from it are customer data: never commit them.
* The node may be shared (other users or agents): check `nvidia-smi` and `uptime` before benchmarking.
  Background CPU load widens CPU-only intervals and can trip `--fail-on-slowdown`; re-run a single case with
  `--cases ... --configs ... --repeats 15` before believing a small slowdown.
* `pkill -f PATTERN` from a shell whose own command line contains PATTERN kills that shell; kill by PID.
* Git identity is not configured globally on the node; commit with
  `git -c user.name=... -c user.email=... commit`.

## 8. How the work was done (and how to continue)

1. Profile the production run (nsys, sassprof, knock-outs, clock scaling) and write down bounds.
2. Prototype in a throwaway worktree with runtime switches (`GMX_EXP_*`), measure variants interleaved
   with `prun` (5+ rounds; minimum detectable effect ~0.7% on MAS1, use 9+ rounds for small effects).
3. Port each winner cleanly on a feature branch from `dev`, one commit per change; validate every commit
   with `gmxbench all -A <parent> -B <branch> --tier quick --strict` (`--repeats 9` for effects < 1%).
4. Before merging into `dev`: `--tier full --strict` against `dev`, `gmxbench upstream`,
   `gmxbench sweep --target-only`, and for numerics-relevant changes an energy-conservation check.
5. Update this file and add a report under `reports/`.
