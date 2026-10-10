# MAS1 on one L40S, iteration 3: measurements (2026-10-10)

Code: `dev` = `56a6b0fd5e` (2026.4 + commits A1-A5). Input: the day-1 production tpr
(`bench/results/profiling-2026-10-06/raw/inputs/prod/topol.tpr`, customer-derived, not committed).
Command: `-ntmpi 1 -ntomp 16 -nb gpu -pme gpu -bonded gpu -update gpu -nstlist 200 -pin on -notunepme`
(nstlist 200 → rlist 1.742 nm). Raw data: `bench/results/profiling-2026-10-10/` on the node (`raw/`,
`analysis/`). Node idle throughout (checked `uptime`, `nvidia-smi`).

## 0. Method

| layer | tool | builds |
|---|---|---|
| step budget, kernel table, anatomy | nsys light (`nsys_capture.sh`, `NSYS_LIGHT=1`) + `nsys_steps.py` + `budget.py` | S = `cuda` profile (NVTX), `7ec56764bc0b18c7` (same sources as dev) |
| CPU launch vs GPU start | `launch_gaps.py` on the same capture | S |
| hardware counters per kernel | **Nsight Compute `--set full`**, kernel replay, `--cache-control all`, natural clocks (`--clock-control none`) and base clock; `ncu_summary.py`, `sass_hot.py` | L = `cuda-nosub` + `-lineinfo`, `04055158c563c3d4`: SASS identical to the production build `496b74c76c06fb80` (all 377 functions, md5 of the disassembly) |
| GPU utilisation through the step | nsys `--gpu-metrics-devices=0 --gpu-metrics-frequency=100000` + `gpumetrics_phase.py` (samples folded onto the step phase, 1,970 plain steps) | S |
| CPU | `perf_capture.sh` (dwarf call graphs), production build and a `-g` build (`c8d3c847afdf0309`, profile in `raw/suite-dbg.toml`) for source lines | P, G |
| NB kernel what-ifs | `nbvariants/` (compile the NB TU of a patched source copy with the production nvcc flags; unmodified copy = production SASS) + `weigh.py` (per-copy body sizes × measured execution counts) | - |
| PME spread what-ifs | `spread_tiles.py` on the MAS1 coordinates (grid 96×96×144, order 4) | - |
| Ewald correction fits | `pmecorr_fit.py`, `pmecorr_weighted.py` (discrete minimax by LP, fp32 FMA emulation) | - |

**Counters**: DCGM was paused with `dcgmi profile --pause` (works from the user account) for 2 minutes
(17:28:43-17:30:39 UTC, `counters_window.sh`, approved by the node owner), resumed by an EXIT trap.

**Caveats**:
* `--cache-control all` flushes the caches before every replay pass. DRAM and L2-hit numbers for the small
  kernels (leap-frog, reduce, x→nbat, FFTs) are therefore cold-cache values; in production the working set
  is L2-resident (section 7: DRAM ~0).
* ncu serialises kernels: per-kernel numbers are "alone".

## 1. Baseline

| | value | source |
|---|---|---|
| mean step under nsys light | 709.3 µs (243.6 ns/day; nsys light ≈ +0.6%) | `raw/nsys/S-light/budget.txt` |
| plain step median / mean | 627.8 / 629.7 µs | same |
| plain mdrun, production build, 3,000 timed steps | 0.739 ms/step (234 ns/day); NVTX build 0.703 (246); production build with perf attached 0.703 (246) | `scratchpad` runs, `raw/perf/P/run.log` |
| gmxbench (merge report, long runs) | 248.3 ns/day | `../2026-10-gpu-kernel-opt.md` |

Short single runs vary by ~5%; claims below use trace-internal ratios, not these totals.

**Pitfall found**: mdrun's own performance line is wrong under an nsys capture range (2.1-3.1 ms/step,
"Rest" 68-77% of the time). The time nsys spends at `cudaProfilerStop` is counted. The 2026-10-07 captures
show the same. Use `nsys_steps.py` step periods.

## 2. Step budget and anatomy (`raw/nsys/S-light`)

| step type | share of run time | excess over a plain step |
|---|---|---|
| plain | 87.0% | - |
| pair search (+ energy) | **10.3%** | 72.8 µs per step (14.6 ms per search step) |
| energy/COM | 1.0% | 7.1 µs per step |

Pair-search step on the CPU (NVTX, per search step):

| work | time |
|---|---|
| NS search local | 9.46 ms |
| NS grid local | 1.71 ms |
| GPU bonded list update | 0.36 ms (2.05 ms before commit A3) |
| Wait GPU NB local | 1.26 ms |
| **GPU idle** | **11.9 ms** |

Kernel time per step (all steps):

| kernel | time per step | per launch / instances | share of kernel time |
|---|---|---|---|
| NB F | 441.6 µs | | 45.2% |
| 96-point FFT | 119.1 µs | 4 × 27 µs, concurrent | |
| spread | 90.9 µs | | |
| gather | 64.0 µs | | |
| bonded | 60.9 µs | | |
| x→nbat | 54.1 µs | 27.5 µs median, waiting for SM slots | |
| solve | 26.8 µs | | |
| c2r / r2c | 23.1 / 22.9 µs | | |
| rolling prune | 21.8 µs | 43.9 µs every 2nd step | |
| LINCS | 21.8 µs | | |
| SETTLE | 8.6 µs | | |
| leap-frog | 6.8 µs | | |
| VF (energy steps) | 6.3 µs | | |
| reduce | 3.7 µs | | |

Plain-step anatomy (median times in µs from the start of spread, `analysis/launch_gaps-S-light.txt`):

| kernel | stream | GPU start | GPU end | CPU launch call |
|---|---|---|---|---|
| spread | PME, high priority | 0 | 90.5 | −397 |
| D2H x for CPU CMAP | | 3 | 90 | |
| x→nbat | NB | 65.3 | 92.2 | −383 |
| bonded | own stream | 91.6 | 153.3 | −374 |
| r2c | PME | 104.8 | 130.8 | −359 |
| FFT | PME | 131.7 | 154.5 | |
| **NB** | NB | **134.3** | **579.8** | **−367** |
| FFT, solve, FFT, FFT, c2r | PME | 245-402 | | |
| gather | PME | 403 | 468 | |
| reduce | NB | 581.5 | 585 | |
| leap-frog | update | 587 | 595 | |
| prune | NB | 588.6 | 637 | |
| SETTLE | | 608.6 | 618 | |
| LINCS | | 610 | 638 | |

Every launch call returns ~400 µs before its kernel starts, so **plain steps are never launch-bound**. The
order of the launch calls decides SM access: the bonded kernel is launched before the NB kernel
(`sim_util.cpp`: `setPbcAndlaunchKernel` before `do_nb_verlet`), and r2c has high priority. The only copy
that starts as soon as it is issued is the 2.2 MB H2D of CPU CMAP forces (stream 25, hidden under the NB
kernel). All copies use pinned memory at 21-26 GB/s; a search step uploads ~9 MB (list, maps), ≈ 0.4 ms.

## 3. Nsight Compute: every plain-step kernel (`analysis/ncu_summary.txt`)

Natural clocks (~2.5 GHz, no power cap during replay), alone, cold caches.

| kernel | µs | IPC | SM% | L2% | DRAM% | L2 hit | occupancy achieved/theoretical | waves | top stalls (cycles per issued instruction) |
|---|---|---|---|---|---|---|---|---|---|
| NB F | 431.1 | **3.24** | 76 | 20 | 9 | 92 | 64/67 (registers) | 4.93 | not_selected 2.6, wait 2.0, long_scoreboard 1.6, short_scoreboard 0.9 |
| spread | 87.7 | 0.23 | 8 | **87** | 17 | 94 | 84/100 | 3.40 | **lg_throttle 73.9**, mio_throttle 46.9, barrier 24.3, long_scoreboard 12.9 |
| bonded | 57.6 | 0.25 | 6 | 65 | 26 | 87 | 85/100 | 2.50 | **lg_throttle 80.5**, long_scoreboard 35.8, mio_throttle 21.8 |
| rolling prune | 35.7 | 2.07 | 44 | 21 | 22 | 77 | 76/100 | 1.65 | long_scoreboard 8.2, wait 3.6 |
| gather | 33.2 | 0.96 | 30 | 64 | 42 | 90 | 87/100 | 3.40 | mio_throttle 16.3, long_scoreboard 10.2, barrier 5.8 |
| LINCS | 16.4 | 0.21 | 6 | 50 | 31 | 83 | **17**/100 | **0.18** | long_scoreboard 14.8, lg_throttle 8.3, barrier 4.5 |
| 96-point FFT | 13.6 | 0.31 | 10 | 26 | 61 (cold) | 59 | 35/100 | 0.39 | long_scoreboard 33.3, barrier 9.2 |
| leap-frog | 12.9 | 0.06 | 5 | 24 | 76 (cold) | 51 | 67/100 | 0.85 | long_scoreboard 354 (cold) |
| reduce | 12.6 | 0.05 | 4 | 21 | 81 (cold) | 40 | 70/100 | 0.85 | long_scoreboard 490 (cold) |
| c2r / r2c | 11.8 / 11.9 | 0.48 | 10 | 27 | 62 (cold) | 57 | 45/81 | 0.62 | long_scoreboard 24.8 |
| SETTLE | 10.0 | 0.29 | 10 | 16 | 64 (cold) | 42 | 19/50 | 0.37 | long_scoreboard 22.1 |
| solve | 9.7 | 1.08 | 23 | 29 | 72 (cold) | 53 | 74/88 | 3.61 | long_scoreboard 24.4 |
| x→nbat | 7.2 | 0.10 | 5 | 32 | 58 (cold) | 64 | 63/100 | 0.93 | long_scoreboard 259 (cold) |
| fresh prune (search step) | 404.4 | 3.23 | 79 | 14 | 7 | 94 | 97/100 | 13.2 | wait 3.1, not_selected 2.9 |
| NB VF (energy step) | 630.3 | 3.34 | 82 | 14 | 6 | 92 | 64/67 | 4.93 | not_selected 3.4, wait 1.6 |

At ncu's base clock the NB kernel takes 847 µs (IPC 3.33) and spread 125 µs (L2 86%). Spread scales less
with the SM clock than NB does (1.43x vs 1.97x), consistent with an L2-side limit.

Launch configurations:

| kernel | blocks × threads | registers | shared memory |
|---|---|---|---|
| NB | 11,212 × 64 | 61 | 0 |
| spread | 2,899 × 256 | 40 | 5.6 KB |
| bonded | 2,130 × 256 | 40 | |
| gather | 5,798 × 128 | | |
| LINCS | 150 × 256 | | |
| SETTLE | 156 × 256 | 61 | 27.6 KB |

## 4. NB kernel per SASS instruction (`raw/ncu/nb_f_source_sass.csv`, `analysis/nb_f_sass_hot.txt`)

* **406.1 M warp instructions per launch** (433 M before commit A1). 4.247 M pair bodies (16 unrolled
  copies, 53-54 instructions each, 14-18 active lanes), i.e. **95.6 executed instructions per body**.
* By class:

  | class | share of instructions | share of stall samples |
  |---|---|---|
  | FP32 arithmetic | 54.7% | 39.3% |
  | control / uniform / predicate | **22.8%** | **37.9%** |
  | integer / move | 16.0% | 15.4% |
  | memory / shuffle | 6.5% | 7.4% |

* Top opcodes: FFMA 27.3%, FMUL 15.8%, **MOV 8.0%**, FADD 5.2%, BRA 4.8%, PLOP3 3.5%, USHF 2.9%, BSYNC 2.6%.
* Stall samples: not_selected 26.3%, wait 20.9%, long_scoreboard 16.8%, selected 10.7%, short_scoreboard
  9.5%, **no_instruction 5.1%** (1824 instructions = 29 KB of unrolled code), dispatch 3.9%,
  branch_resolving 3.1%.
* **The `R2UR` that waits for the `imask` load at the head of each `jPacked` iteration holds 6.9% of all
  stall samples.** The cj → atom-index → coordinate load chain is the next largest.
* **i-slot test** (8.23 M per launch = 8 per j-cluster): `ULOP3.LUT UP1, UR(bit), UR(imask)`;
  `USHF.L.U32`; `PLOP3.LUT P1, …, UP1` (uniform → regular predicate); `@!P1 BRA`.
* **Body** (53, approximate attribution):
  * LJ parameter fetch 5 (`LDS` type, `IMAD`, `MOV`, `IMAD.WIDE`, `LDG.E.64.CONSTANT`).
  * `FMNMX` (min r²) + `MUFU.RSQ`.
  * Ewald correction 15: `FMUL` β²r², 2 `MOV` immediates, 10 `FFMA`, `MUFU.RCP`, `FMUL`. The β³ `FMUL` that
    follows is kept by every variant.
  * Force switch 11, including 2 `MOV`s from the constant bank (an FFMA takes one constant operand).
  * Then LJ r⁻⁶/r⁻¹² (~7), the exclusion mask (~3), qᵢqⱼ and the Coulomb combine, and 6 accumulation `FFMA`s.
* What-ifs (`nbvariants/`, section 0) are in `HYPOTHESES.md` H1: −4.1%, −4.2%, −6.2% and about −14.5%
  together, at ≤ 64 registers.

## 5. Ewald correction fits (`analysis/pmecorrf_fit.txt`, `pmecorr_weighted.txt`)

βr_c is set by `ewald-rtol` alone (erfc(βr_c) = rtol): 3.123 (1e-5), 3.459 (1e-6), 3.767 (1e-7).
Production's P6/Q4 has the same absolute error (8.3e-7) on [0, 16], so it is fitted for a wider range than
any default run uses. Discrete minimax rationals (LP bisection), fp32 coefficients, emulated fp32 Horner:

* With **uniform** error weighting, (4,4) looks 3x better than production (2.95e-7 vs 8.3e-7 absolute).
  But relative to the 1/r³ term it is 4.5x *worse* near r_c.
* With error weighted by max(1, z³), i.e. relative to the term the correction is added to, the cheapest
  forms that match production in the kernel's monic r² form at MAS1's β are (5,4) and (4,5): 11
  instructions vs 15, max 4.6e-7 vs 9.4e-7, RMS 1.1e-7 vs 3.3e-7.
* Pure polynomials are far worse: degree 7 gives 1.8e-4.

## 6. PME spread (`analysis/spread_tiles.txt`, `raw/ncu/pme_spline_and_spread_source_sass.csv`)

* Per instruction: `RED` 32% of stall samples, `LDG` 23% (spline-table and coordinate loads), `BRA` 13%,
  `STS`/`LDS`/`BAR` 17%.
* Model results are in `HYPOTHESES.md` H2: today 11.87 M updates and 3.47 M sectors (9.36 per warp
  instruction). A spatial order alone *raises* the sector touches of the unchanged kernel (3.91 M, 10.5 per
  instruction): ordering without accumulation does not help, as the 2026-10-07 replay found.
* SASS facts (sm_89, `nvcc -arch=sm_89`):

  | operation | SASS |
  |---|---|
  | shared fp32 `atomicAdd` | `ATOMS.CAST.SPIN` loop |
  | shared int32 `atomicAdd` | `ATOMS.ADD` |
  | shared int64 `atomicAdd` | CAS loop |
  | global fp32 `atomicAdd` | `RED.E.ADD.F32` |
  | float2/float4 `atomicAdd` | not available (sm_90: `REDG.E.ADD.F32x2/x4`) |

## 7. GPU metrics through the plain step (`analysis/gpumetrics_phase.txt`)

| µs from spread start | SMs active | warps in flight | SM issue | running |
|---|---|---|---|---|
| 0-60 | 100% | 94% | 4.6-6.3% | spread |
| 60-100 | 83-99% | 57-79% | 5-12% | spread, x→nbat, bonded |
| 100-140 | 97-100% | 71-78% | 16-18% | bonded, r2c, NB starting |
| 160-420 | 100% | 63-66% | 80-87% | NB (+ FFT/solve) |
| 420-480 | 100% | 67-70% | 64-74% | NB + gather |
| 480-540 | 99-100% | 61-65% | 69-80% | NB |
| 540-580 | 77% → 48% | 43% → 25% | 48% → 20% | NB tail |
| 580-640 | 53-75% | 32-45% | 17-26% | reduce, leap-frog, prune, LINCS, SETTLE |

* DRAM read bandwidth 0.0% and DRAM write ~0.8% throughout: **the step's working set lives in the 96 MB
  L2**.
* Means over plain steps: SMs active 93.5%, SM issue 55.8%.
* Lost issue capacity against the NB level (~85%): ≈ 124 µs-equivalents in 0-140 µs (H2/H3), ≈ 24 in the
  NB tail (H6), ≈ 50 in the update chain (H7).

## 8. CPU pair search (`raw/perf/P`, `raw/perf/G`)

* libgomp spin-waiting is 83% of all CPU samples; libgromacs is 10.6%.
* `nbnxn_make_pairlist_part<NbnxnPairlistGpu>` is 4.7% of all samples, **evenly over the 16 OpenMP threads
  (0.27-0.32% each)**. Grid sorting (`sortColumnsGpuGeometry`, `sort_atoms`), `convertIlistToNbnxnOrder`
  and `combine_nblists` are small and parallel.
* CPU CMAP (`cmap_dihs` + `accumulateCmapForces`) is 1.5% of samples, hidden under GPU work.
* Inside the search (`-g` build, source lines):
  * **`makeClusterListSupersub`** (`pairlist.cpp` 856-915): `clusterBoundingBoxDistance2_xxxx_simd4`
    (`xmmintrin.h`, 128-bit SSE, although the build is AVX_512), then the scalar loop `for ci < 8: if
    (d2l[ci] < rlist2) imask |= 1 << …` (lines 881, 906, 910 are the top three lines). About 40% of the
    search.
  * **`setExclusionsForIEntry`** (1590-1660) with the j-cluster bisection (1039-1055): about 14%.
* The single-threaded bonded-list remap of day 1 is gone (2.05 → 0.36 ms per search, commit A3).

## 9. Notes for the next session

* Run CPU-side analyses on cores 32-39 (`taskset`) while mdrun runs with `-ntomp 16 -pin on` (mdrun pins
  to CPUs 0, 2, …, 30). This is a precaution, not a measured effect: the "slow" runs seen first were the
  nsys reporting artefact of section 1.
* The DCGM pause is cheap (2 minutes covered 47 kernel profiles plus a 2,000-step GPU-metrics capture),
  but every pause needs the node owner's consent again.
* For realistic memory numbers of small kernels, rerun ncu with `--cache-control none`.
