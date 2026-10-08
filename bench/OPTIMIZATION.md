# Performance optimisation of this fork: start here

For anyone (person or agent) continuing the performance work on this GROMACS fork. It says where things
stand, what was measured and rejected, what is still open, which tools exist and what to watch out for.
**Keep it current: add every new result, positive or negative, to the tables below.**

Last updated 2026-10-08.

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
4. [`reports/profiling-gpu-2026-10-07/`](reports/profiling-gpu-2026-10-07/): GPU kernel deep dive.
   `EXEC-SUMMARY.md` first, then `OPPORTUNITIES.md` (ranked, with bounds), `ANALYSIS.md` (per-kernel
   instruction profiles, critical path), `NOTES.md` (lab notebook), `NEXT-STEPS.md` (the hand-off plan that
   produced the five commits; section 9 is the final status).
5. [`reports/profiling-2026-10-06/`](reports/profiling-2026-10-06/): whole-step profile of the production
   run (CPU and GPU, pair search, CMAP, power cap, run settings), same file structure.
6. [`reports/phaseB-energy-2026-10-07/REPORT.md`](reports/phaseB-energy-2026-10-07/REPORT.md): NVE energy
   conservation before/after, and the `lincs-iter` finding.

The session reports are copies of write-ups made on the benchmark node. They refer to `raw/...` (logs, nsys
captures, tprs, summary.json) and to `bench/results/...`: those stay on that node (git-ignored; the tprs are
customer-derived). Where they say "branch `feature/profiling`", the tools are now in `bench/profiling/` on
`dev`. Numbers in them describe the code before the five commits unless stated otherwise.

## 3. Where MAS1 time goes (before the five commits; re-profile before acting)

* Plain steps (~87%) are GPU-bound. The NB force kernel was 60% of run time (445 us/step); it is
  issue-bound (IPC 3.17 of 4), memory is not the limit. After commit `2e3e86d3a1` it is ~6% cheaper.
* PME kernels cost 18.3% of the step although they overlap the NB kernel. PME spread alone: 81 us/step =
  **10.9%**, 91% of it on the critical path (high priority, holds the SMs at the start of the step). 92% of
  spread is 11.9 M scattered global grid updates per step; atomicity (18%) and atom order (2-5%) are not the
  limit. Gather and FFT+solve: ~3.5% and ~3.2% through SM sharing.
* Pair-search steps (every 200 steps): **10.3%** of run time; the GPU idled 12.7 ms per search while the CPU
  built the grid (1.7 ms), the pair list (8.6 ms) and the GPU bonded lists (2.1 ms, single-threaded until
  commit `e1971ae650`).
* Update chain: LINCS 21.6 us mean (2.9%; doubles when it shares SMs), SETTLE 12.5 us (1.7%; now staged and
  concurrent with LINCS). Bonded kernel 59 us, now overlapped with the NB kernel.
* Energy/COM steps ~1%, CPU CMAP path ~1% (CMAP on the CPU also disables CUDA graphs; ceiling +0.9%).
* The GPU is power-capped at an enforced 325 W (cannot be raised): clocks ~6% below max all the time, so
  work reductions show up both as speed and as lower energy per ns.

## 4. Open opportunities (ranked by expected gain per effort)

| Opportunity | Bound (measured) | Estimate (assumption) | Effort | Where described |
|---|---|---|---|---|
| Fix the MAS1 input (interlocked rings) and use HMR at 4 fs | lysozyme: 1.7x ns/day at 4 fs | ~1.5-1.8x | input work + science decision, no code | `profiling-2026-10-06/` (input defect), report |
| PME spread rewrite: atoms in spatial (nbnxm) order, accumulate in shared memory, one global atomic per grid point | 10.9% | 3-6% | days (`ewald/pme_spread.cu`, PME atom order) | `profiling-gpu-2026-10-07/OPPORTUNITIES.md` #5; upstream !6000 (HIP only) |
| Overlap the CPU pair search with GPU steps / faster search | 8.5-10% (before `e1971ae650`) | re-measure first | L | `profiling-2026-10-06/OPPORTUNITIES.md` |
| PME gather in sorted order; solve fused into cuFFT callbacks | 3.5% + 3.2% | 1-2% | M-L | `profiling-gpu-2026-10-07/OPPORTUNITIES.md` #6 |
| LINCS with more parallelism; rolling prune off the update path | 2.9% | 1-2% | M | #7 there; upstream !4998 (HIP) |
| Bonded kernel memory layout (iatoms SoA) | ~1.6% (overlapped) | 0.5-1% | M | #8 there |
| Energy/COM steps without blocking host copies | ~1% | 0.5-0.8% | M | `profiling-2026-10-06/OPPORTUNITIES.md` |
| `-ntomp 20` instead of 16 (run argument) | - | +0.5-0.6% (measured on old code) | minutes: add 20 to `[sweep.dims] gpu_threads` and re-sweep | both sessions |
| SETTLE staging only for larger systems (fixes -0.7% on a 5k-atom water box with CUDA graphs) | 0.7% on tiny systems | - | S | report, Limits |
| NB cluster-pair efficiency (52% of lanes idle in the pair loop) | ~30% of NB kernel (theoretical) | research | L+ | `profiling-gpu-2026-10-07/OPPORTUNITIES.md` #9 |

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
| `records.py`, `derived_records.py`, `gpu_records.py` | turn analysis outputs into `summary.json` records |

**Prototype patches** (`bench/profiling/experiments/`): `exp-throwaway.patch` (2026-10-06) and
`exp-gpukern.patch` (2026-10-07) contain every prototyped variant behind `GMX_EXP_*` runtime switches,
including the rejected ones and the kernel replay harness. They apply to commit `3df7585464` in a throwaway
worktree; never merge them.

## 7. Environment facts and gotchas

* **DCGM holds the GPU performance counters** on the node: Nsight Compute, nsys GPU metrics and CUPTI PM/PC
  sampling fail; CUPTI SASS metrics work (`sassprof`). Pausing DCGM (`dcgmi profile --pause` / `--resume`)
  needs an administrator.
* **Power cap**: enforced 325 W, cannot be raised; `sw_power_cap` is active all the time.
* **Re-running CMake in an nvcc build directory drops `-D_FORCE_INLINES`** (upstream
  `gmxManageNvccConfig.cmake` adds it only on the first configure). `gmxbench upstream` reconfigures in
  place. Rebuild from a fresh configure before a perf A/B after any CMake change.
* Builds whose source/build paths differ in length differed by ~4% in an A/A test: use gmxbench builds
  (fixed-length paths) for every comparison.
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
