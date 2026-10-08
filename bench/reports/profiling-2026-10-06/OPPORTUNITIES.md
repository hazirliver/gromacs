# Optimisation opportunities: MAS1 on one L40S

Baseline: production command on the production-like build, 0.7417 ms/step [0.7383, 0.7452] = 233.0
ns/day, 112.7 kJ GPU energy per simulated ns (`raw/exp/baseline`). Mean step 742 us; "share" = share of
the run time the mechanism lives in (from the step budget, ANALYSIS 2). Upper bounds are *measured*
(ablation, nsys exposed time or throwaway build); realistic estimates state their assumption.

Energy model (measured, ANALYSIS 6): the 325 W cap is enforced and binding, the GPU draws ~305 W whenever
it is busy, so for **work reductions** energy per ns follows time per ns (potential-shift -6.5% time /
-7.3% energy; reaction-field -19.4% / -18.7%); **removing GPU idle time** saves less energy than time and is
partly paid back by a lower clock (no-CMAP + graphs: -1.1% time, -0.6% energy, -10 MHz). For idle-gap items
the energy bound below assumes the GPU draws ~1/3 of its busy power while idle-waiting (not measurable at
NVML's 10 Hz; stated as an assumption).

Prior art was checked on GitLab (issues, MRs, 2025/2026 release notes; `raw/prior-art.md`).

## Ranked by realistic gain per unit effort

| rank | opportunity | mechanism | evidence | share of step time affected | upper bound: time / energy | realistic estimate (assumption) | effort (files) | risk (numerics / portability / upstreamability) | prior art upstream | gmxbench case that validates it |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Make the OpenMP loop in the GPU bonded-list update actually parallel | `#pragma omp parallel for` in listed_forces_gpu_impl_gpu.cpp is compiled by nvcc without -fopenmp, so a 2.05 ms index remap runs on one thread every search step while the GPU idles | `raw/nsys/prod-S-light` (NVTX "GPU Bonded list update" 2052 us/search), `raw/perf/prod-P-fp` (annotate), build.make:9979 + nm/objdump (no GOMP); **throwaway build: +0.85% [0.38, 1.32]**, GPU idle 78.1 -> 69.5 us/step (`raw/exp/exp-nbminblocks`, `raw/nsys/exp-nbminblocks-3136061f41d389dc-light`) | search steps (10.7% of time), 2.05 of 15.95 ms each | 1.4% / ~0.5% (idle-gap energy) | **+0.85% measured** | S: pass -Xcompiler=-fopenmp (or the host OpenMP flags) to nvcc for host code, e.g. in cmake/gmxManageNvccConfig.cmake, or move the loop to a host-compiled .cpp; also fixes lincs_gpu.cpp (start-up/DD only here) | none: bitwise identical (`quality --strict` passed, `raw/gmxbench/quality-ompcuda`); host-compiler flag handling for clang hosts; upstreamable bug fix | none found | `perf --configs perf-mas1-production` + `quality --strict` |
| 2 | Run the GPU bonded kernel in its own stream | the bonded kernel (59 us, 2.5 waves) sits in the NB local stream between x->nbat and the NB kernel, delaying the NB kernel (the step's critical path) | anatomy (`raw/nsys/prod-S-light`): NB starts 165 us after the previous SETTLE; **throwaway build: +0.91% [0.29, 1.53]**, GPU span 749.6 -> 739.2 us/step, NB kernel +13 us from sharing SMs | plain steps (87%) | 59 us/step = 8% if the bonded kernel were free; measured overlap recovers ~10 us | **+0.91% measured** | S-M (~40 lines): listed_forces/listed_forces_gpu.h, listed_forces_gpu_impl.h, listed_forces_gpu_impl_gpu.cpp, listed_forces_gpu_internal.cu, mdlib/sim_util.cpp; CUDA-graph capture must fork/join the extra stream; SYCL/HIP back-ends need the same | GPU force atomics reorder (EQUIVALENT: force rel-RMS 9.8e-8 vs noise 1.0e-7, `raw/gmxbench/quality-bondedstream`); upstreamable | none found | `perf` + `quality` |
| 3 | -ntomp 20 (one thread per physical core) | faster CPU pair search (8.9 vs 10.1 ms per search) | `raw/exp/threads` | search steps | +0.58% [0.15, 1.02] / -0.3% | **+0.6% measured** | S: `bench/suites/target.toml` [configs.perf-mas1-production] | none | - | `sweep --target-only` |
| 4 | Cheaper force-switch evaluation in the NB kernel | `ljForceSwitch` evaluates two cubic switch polynomials, each separately scaled by rSwitch^2/r, for every pair (branch-free); +232 SASS instructions (+13.7%) in an SM-clock-bound kernel (alpha 0.90) | potential-shift ablation: NB kernel 446 -> 406 us (`raw/nsys/potshift-S-light`); `raw/sass/` | NB kernel (60%) | 40 us/step = 5.4% / ~5.4% | 1-2% (assumption: factoring the polynomials and the r scaling removes 25-50% of the switch-specific instructions; time follows instructions as in the ablation, +13.7% -> +9.8%) | S-M: nbnxm/nbnxm_kernel_utils.h (shared CUDA/SYCL/HIP helper), nbnxm/cuda/nbnxm_cuda_kernel.cuh | FP reordering: EQUIVALENT not IDENTICAL; all GPU back-ends; upstreamable | none found for force-switch cost | `quality` + `perf` + nsys kernel table |
| 5 | Update chain: SETTLE concurrently with LINCS, fewer launches, rolling prune off the update path | 3 sub-wave latency-bound kernels in sequence (0.18-0.85 waves); LINCS and SETTLE constrain disjoint atoms but run back-to-back; the rolling prune (every 2nd step) slows LINCS 13 -> 30 us | `raw/nsys/prod-S-light` (anatomy, LINCS bimodality), occupancy table | plain steps: 51 us post-NB + ~8.5 us/step interference | avoidable part ~25 us/step = 3.4% / ~3.4% | 1.5-2.5% (assumption: LINCS||SETTLE saves ~12 us, prune moved off the path ~8 us, fewer gaps ~2 us) | M: mdlib/update_constrain_gpu_impl.cpp, lincs_gpu.cpp, settle_gpu.cpp, nbnxm GPU prune launch point | scheduling only: per-kernel arithmetic unchanged; virial accumulation; upstreamable | !5469 (update stream high priority, in 2026.0) | `perf` + `quality` |
| 6 | CUDA graphs despite CPU CMAP (stream memops) or CMAP on the GPU | graphs are disabled for the whole run while any CPU force exists; CMAP is the only one | `raw/exp/cmap`, `raw/exp/power`, `raw/nsys/nocmap-graph-S-full` | plain steps | ceiling +0.9% [0.3, 1.5] / -0.6% | +0.7-0.9% (assumption: the memops path recovers the graph part of the ceiling; a CMAP port additionally removes the copies, ~0.1%) | M (graphs with CPU forces, prototype exists) / M-L (CMAP kernel: listed_forces_gpu_internal.cu, `fTypesOnGpu`) | graphs still experimental upstream; CMAP port: FP order (EQUIVALENT); conflicts with the CMAP FEP refactors on main | prototype commit 0030776d "Prototype: allow CPU forces in CUDA graph via memops" (A. Gray, 2023-12, unmerged); MR !1730 "CUDA CMAP kernel" (P. Bauer, closed 2021-11) | `perf` with GMX_CUDA_GRAPH=1 + `quality` |
| 7 | Energy/COM-removal steps without blocking host round trips | with GPU update, COM removal copies x and v to the host, removes COM on the CPU and copies them back with blocking waits; 4 stream syncs per energy step | `raw/nsys/prod-S-light` energy-step anatomy; `raw/exp/nstcalc` (all intervals 1000: +1.17%) | energy steps (1.4% of time) | 0.95% (nsys) to 1.2% (ablation incl. fewer VF kernels) / ~0.6% | 0.5-0.8% (assumption: GPU-side COM removal and async energy transfer leave only the VF-kernel cost) | M: mdrun/md.cpp:1761-1842, mdlib/update_constrain_gpu_impl.cpp, mdlib/vcm.cpp | FP order of COM sums (EQUIVALENT); upstreamable | issues #3988, #4106 (cited in the code) | `perf` + `quality` |
| 8 | Overlap the CPU pair search with GPU steps | every 200 steps the GPU idles 12.7 ms while the CPU builds grid (1.7 ms), pair list (8.6 ms) and bonded lists (2.1 ms, see #1) | `raw/nsys/prod-S-light`, `raw/exp/nstlist`, `raw/exp/threads` | search steps (10.7%) | GPU idle 12.7 ms per search = 8.5% (10.3% excess incl. fresh prune and uploads) / ~3% | 4-6% (assumption: search on coordinates ~20 steps old with an outer buffer larger by the extra displacement; from the nstlist sweep +0.06 nm costs ~0.5%) | L: nbnxm pair search and grid, mdlib/sim_util.cpp, GPU-resident state reordering when switching lists | high: buffer correctness, atom-order switch on the GPU; upstreamability uncertain | none found (GROMACS pair search is CPU-only) | `quality` (must stay EQUIVALENT) + long-run energy drift + `perf` |

Combined S-effort result (`raw/exp/exp-combo`, one build with #1 + #2, quality `--strict` passed in
`raw/gmxbench/quality-combo`): **+2.03% [1.42, 2.65]**; with #3 (-ntomp 20) **+2.82% [2.19, 3.45]** =
0.7232 ms/step, 239.0 ns/day, 111.8 kJ/ns (-1.0% GPU energy per ns). The gains are additive.

Upper-bound order (largest bound first): reaction-field envelope of everything PME 24% (physics, not an
option) > search-step overlap 8.5-10.3% (#8) > bonded kernel fully hidden 8% (#2; 0.9% realised) > power cap
6.5% (not attainable: enforced 325 W) > force-switch kernel cost 5.4% (#4) > avoidable update-chain part
3.4% (#5) > bonded-list OpenMP 1.4% (#1) > energy/COM steps 0.95-1.2% (#7) > CUDA graphs/CMAP 0.9% (#6) >
-ntomp 20 0.6% (#3).

## Not worth pursuing (measured)

| idea | result | artifact |
|---|---|---|
| NB kernel launch bounds for Ada (20/24 blocks per SM) | -4.0% / -16.4%: more registers spilled and, under the power cap, lower clocks (2322 / 2110 MHz) | `raw/exp/exp-nbminblocks` |
| Swapping stream priorities (NB local High, PME Normal) | -0.57% [-1.05, -0.09]: the PME chain moves onto the critical path | `raw/exp/exp-nbminblocks` |
| Porting CMAP alone (without graphs) | 0.9992 [0.9963, 1.0022]: the copies and CPU work are hidden | `raw/exp/ablation` |
| -nstlist other than 200 | 150: -0.5%, 300: -2.4%, 100: -3.2% | `raw/exp/nstlist` |
| nstcalcenergy alone | no change (coupling/COM every 100 steps still need global steps) | `raw/exp/nstcalc` |
| PME: larger rc with coarser grid | -6% (1.25) to -20% (1.40); tuner already picks 1.20 | `raw/exp/pme` |
| Tabulated Ewald, twin cut-off, no list splitting, other prune intervals | all slower or no change | `raw/exp/knobs` |
| Pinning / thread placement | no effect unless threads share cores (-2.6%) | `raw/exp/pinning` |
| More threads via hyperthreading (24/32) | -0.7% / +0.2% | `raw/exp/threads` |
| Raising the power limit | not possible: enforced limit 325 W | `raw/exp/power` |
| Lower clocks for energy | -2.4% energy for -2.3% speed at 2250 MHz (a trade-off, not a gain) | `raw/exp/power` |
| Log/xtc output reduction | no measurable cost | `raw/exp/nstcalc` |
