# Hypotheses for the next optimisation round (MAS1, `dev` 56a6b0fd5e, 2026-10-10)

Every item comes with the evidence measured this session (`ANALYSIS.md` section numbers), an upper bound,
an estimate with its assumption, effort, numerical risk and the cheapest experiment that would confirm or
refute it. Nothing here is implemented. Step = 709 µs mean (nsys), plain step 628 µs median. "NB" = the
force-only kernel `nbnxn_kernel_ElecEw_VdwLJFsw_F_cuda`, 442 µs/step, on the critical path ~1:1 (92% of
its saving reached the step on 2026-10-07).

## Ranking

| # | hypothesis | bound | estimate | effort | numerics | first experiment |
|---|---|---|---|---|---|---|
| H1 | NB pair-loop instruction diet (a+b+c) | ~9% (all 14.5% of NB instructions) | **+6-7%** | S-M | (a) bit-identical; (b) more accurate; (c) FP order ~1e-7 | `nb_replay.sh` with the three variants, then `gmxbench --strict` |
| H2 | PME spread: shared-memory int32 accumulation, one global atomic per touched point | 12.3% (spread skip) | **+6-8%** spatial order / +3-4% current order | L / M | fixed point: deterministic sums, ~1e-7 rel. | spread replay harness (`exp-gpukern.patch`) with a block-local smem variant |
| H3a | Launch the NB kernel before the bonded kernel | ~6% (42 µs NB start delay) | +1.5-3% | S | none (scheduling) | swap two calls in `sim_util.cpp`, `prun` 9 rounds |
| H3b | Co-schedule spread (L2-bound) with NB (issue-bound): NB first, spread residency capped | 12.3% (shared with H2) | +3-6% | M | none (scheduling) | throwaway launch-order + `__launch_bounds__`/grid-stride spread, nsys phase metrics |
| H4 | Overlap the CPU pair search with GPU steps (lagged list) | 8.4% | 5-7% | L | list buffer for nstlist + lag (EQUIVALENT + energy drift) | prototype on a throwaway branch |
| H5 | Faster CPU pair search (AVX-512 bounding-box test → mask; exclusions without bisection) | ~4% (half of the search) | +2-3% | M | bit-identical list | micro-benchmark of `nbnxn_make_pairlist_part` (perf, `-g` build) |
| H6 | NB kernel tail (SMs drain from 77% to 48% active) | ~3.4% | 1-2% | M-L | none | sci-order / split experiment, phase metrics |
| H7 | Update chain: LINCS fuller launches, prune not in front of LINCS, reduce+leap-frog fused | ~4% (low-issue 580-640 µs) | 1-2% | M | FP order (virial) | nsys anatomy of the post-NB chain |
| H8 | PME gather tile-staged in the H2 order; solve fused into cuFFT LTO callbacks | 3.5% + 3.2% | 1-2% | M-L | FP order | after H2 |
| H9 | Bonded kernel: iatoms SoA + per-block shared-memory force accumulation | ~1.6% (overlapped) | ~1% | M | FP order | sassprof/ncu on a variant |

The NB kernel and spread are on the critical path, so H1 and H2 are the safest bets. H3a is the cheapest
experiment of all. H4 remains the largest non-kernel item.

---

## H1: NB pair-loop instruction diet (ANALYSIS 4, 5)

**Mechanism.** The kernel is issue-bound (IPC 3.24/4, "not selected" the top stall, occupancy at the
64-register cap), so its time follows the executed instruction count (measured 0.73-1.17 time per
instruction share on 2026-10-07). 406 M warp instructions per launch = 95.6 per pair body. Three
independent source-level changes, each checked by compiling the kernel TU with the production flags
(`bench/profiling/nbvariants/`, unmodified copy = production SASS) and weighting each of the 16 unrolled
body copies by its measured execution count (ncu source counters):

| change | where the instructions go today | static SASS | executed warp instructions | registers |
|---|---|---|---|---|
| (a) test `imask`/`wexcl` shifted once per j-cluster (`imaskJ = imask >> (jm*8)`, test `imaskJ & (1<<i)` with constant `i`) | each of the 8 i-slot tests is `USHF` + `ULOP3` + `PLOP3` + `BRA` because the mask lives in a uniform register and BRA needs a predicate | 1824 → 1792 | **−4.1%** (−16.5 M: 2 per i-slot test, 8.23 M tests) | 63 |
| (b) Ewald correction as a monic rational in r², coefficients pre-scaled by β on the host (constant bank), leading coefficients folded into the existing β³ multiply | today: `FMUL` β²r², 2 `MOV` constant re-materialisations, 10 `FFMA`, `MUFU.RCP`, `FMUL` = 15 per pair | (4,4) fit: 1824 → 1736 | (4,4): −5.2%; **(5,4): −4.2%** (−4 per body) | 60 |
| (c) force-switch polynomial pre-combined per atom-type pair: table entry (c6, c12, K2, K3) with K2 = c12·B2 − c6·A2, K3 = c12·B3 − c6·A3, one `LDG.128` instead of `LDG.64` | today 11 instructions per pair incl. 2 `MOV`s from the constant bank (an FFMA takes one constant operand) | 1824 → 1728 | **−6.2%** (−25 M, −6 per body) | 63 |
| a + b + c | body 53-54 → 42-44 instructions, mask test 4 → 2 | 1824 → 1608 | **−15.6% with (4,4), ~−14.5% with (5,4)** | 64 |

**Accuracy of (b)** (`bench/profiling/pmecorr_fit.py`, `pmecorr_weighted.py`; fp32 evaluation emulated
with single-rounding FMAs, MAS1's β = 2.603 nm⁻¹, r_c = 1.2 nm). The kernel only evaluates the correction
for z = βr < βr_c, and βr_c is fixed by `ewald-rtol` alone (3.12 for the default 1e-5). Production's P6/Q4
fit covers a wider range. Error is measured relative to the 1/r³ Coulomb term it is added to:

| form | instructions | max error | RMS error |
|---|---|---|---|
| production P6/Q4 | 15 | 9.4e-7 | 3.3e-7 |
| weighted-minimax **(5,4)** on [0, (βr_c)²] | 11 | **4.6e-7** | **1.1e-7** |
| (4,5) | 11 | 4.8e-7 | 1.0e-7 |
| (6,4) (same degrees as production) | 12 | 4.4e-7 | 0.7e-7 |
| (4,4) | 10 | 1.3e-6 (worse) | |
| (3,4) | 9 | 5.2e-6 (worse) | |

With `ewald-rtol` 1e-6, (5,4) gives 8.0e-7 vs production's 1.0e-6. Coefficients would be fitted once for
βr_c up to 3.46 (rtol ≥ 1e-6); for a smaller rtol the kernel would keep today's function. The β scaling is
done when the kernel parameters are set up (PME tuning changes β and r_c together).

**Table size for (c)**: MAS1 has 54 LJ types, so the table grows from ~24 KB to ~48 KB and stays
L1-resident (NB L1 throughput 42%, memory not limiting).

**Estimate.** −14.5% instructions → NB kernel −11 to −13% (≈ −50 µs/step) → **+6-7% ns/day**, plus lower
energy per ns under the 325 W cap.

**Risks.**
* (a) must keep the prune kernels' `mask_ji` logic, which writes `imask`.
* (b) and (c) also touch the SYCL/HIP kernels, which share `nbnxm_kernel_utils.h`; the CUDA FEP kernel uses
  `pmeCorrF` too.
* Code size grows little; `no_inst` stalls are already 5%.

**Validate.** In-process replay of each variant on live data (`nb_replay.sh`, 8 points × 30 repeats); then
one commit per change, each with `gmxbench all --tier quick` (`--strict` for (a)).

**Further small items seen in the SASS.**
* Hoist the per-body `wexclJ` shift to once per j-cluster (−1 per body).
* An exclusion-free fast path (−3 per body; doubles the unrolled code, i-cache risk).
* Prefetch the next `jPacked` entry's `imask`/`cj` loads: the `R2UR` waiting on the `imask` load holds 6.9%
  of all stall samples, and 19% of issue cycles are empty.

## H2: PME spread with shared-memory accumulation (ANALYSIS 3, 6)

**Mechanism.** Spread issues 11.87 M scattered fp32 `RED`s per step (64 per atom) touching 3.47 M 32-byte
sectors (9.4 per warp instruction). ncu: **L2 throughput 87% of peak** while the SMs issue 5% of their
capacity. The stalls are `lg_throttle` (74 cycles per instruction: the LSU queue is full of atomics) and
`RED` instructions (32% of stall samples). 91% of its 90 µs is critical path (skip on 2026-10-07: +12.3%).

**Proposal.** Each block accumulates its atoms' 4×4×4 stencils in shared memory and adds every touched
grid point to global memory once.

**Offline model on the MAS1 coordinates** (`bench/profiling/spread_tiles.py`, exact union counts):

| organisation | global updates | 32-byte sectors | shared memory |
|---|---|---|---|
| production | 11.87 M | 3.47 M | - |
| 64-atom blocks, current (topology) order, block-local hash | 5.14 M (2.3x fewer) | 1.56 M | up to 4096 entries |
| 64-atom blocks, spatial (nbnxm-like) order | 3.08 M (3.9x) | 0.70 M (5.0x) | ~1,060 points per block |
| 128-atom blocks, spatial order | 2.71 M (4.4x) | 0.51 M | |
| grid tiles 8×8×16 (+3 halo), atoms binned by tile | 2.60 M (4.6x) | 0.47 M (7.3x) | 9 KiB, 143 atoms per tile (p99 165) |
| grid tiles 16³ | 2.07 M (5.7x) | 0.37 M (9.5x) | 27 KiB |

**SASS facts that shape the design** (compiled for sm_89):
* `atomicAdd(float*)` on shared memory is a CAS spin loop (`ATOMS.CAST.SPIN` + branch).
* `atomicAdd(int*)` on shared memory is one native `ATOMS.ADD`, so accumulate in **int32 fixed point**: one
  instruction, and order-independent (deterministic) sums, then convert and `RED.F32` once per point.
  The scale can be per-grid from max|q|; a 64-atom partial sum needs ~8 integer bits of headroom.
* Global vector atomics (`RED.F32x4`, which would write the 4 z-points of an atom at once) exist only on
  sm_90+, not on the L40S.

**Estimate.** If spread time follows L2 sectors (it is L2-bound): the kernel goes from 90 µs to ~25-35 µs
(8 µs splines + ~10-20 µs global flush + shared-memory work) → **+6-8%** with a spatial order, +3-4% in
today's order.

**Effort.**
* Spatial order: L. The PME atom order must follow the nbnxm grid. Since `x_to_nbat` already writes
  coordinates in that order, spread could read the nbat coordinates. Gather must return forces in the same
  order.
* Topology order with a per-block hash: M, but it costs 32 KB of shared memory per block (occupancy).

**Prior art.** cuFINUFFT's "SM" (subproblem) spreading; upstream GROMACS !6000 (HIP only, per-wave
shared-memory tile).

**Validate.** Spread replay harness (`exp-gpukern.patch`: `GMX_EXP_NB_REPLAY` hooks), grid compared with
production.

## H3: fill the low-issue start of the step (ANALYSIS 2, 7)

**Evidence.**
* For the first 80 µs of every plain step the spread kernel holds 94% of all warp slots (and the register
  files) at 5% issue.
* From 80 to 140 µs the issue rate is still 12-18%.
* The NB kernel starts at 134 µs although x→nbat finished at 92 µs. In between, the bonded kernel (own
  stream since commit A4, but launched *before* the NB kernel: `sim_util.cpp` calls
  `setPbcAndlaunchKernel` before `do_nb_verlet`) and the high-priority r2c FFT take the SMs.
* CPU launches run ~400 µs ahead of the GPU, so the launch *order*, not CPU latency, decides who gets the
  SMs.

**(a) S.** Launch the NB kernel before the bonded kernel; bonded blocks then fill SMs as NB blocks retire
(its cost becomes SM sharing, ~12 µs, as measured for A4). Estimate +1.5-3%.

**(b) M.** Launch x→nbat and the NB kernel before spread, and cap spread's residency (e.g. a
persistent/grid-stride spread with fewer blocks, or a shared-memory reservation). The L2-bound spread
then shares SMs with the issue-bound NB kernel instead of idling them. Two cautions:
* Day 1's "NB stream high priority" was −0.57%, because the PME chain then started only at the NB tail.
  Here PME keeps its priority; only its footprint shrinks.
* Bound 81 µs, shared with H2. Estimate +3-6% alone.

**Validate.** Throwaway launch-order patch; `gpumetrics_phase.py` must show issue > 50% from ~10 µs into
the step; `prun` 9 rounds.

## H4: overlap the CPU pair search with GPU steps (ANALYSIS 2, 8)

Search steps are 10.3% of run time: 72.8 µs per step of excess, i.e. 14.6 ms per search step. The GPU
idles 11.9 ms of each while the CPU builds the grid (1.7 ms) and the list (9.5 ms).

**Proposal.** Build the list for step N from step N−k coordinates on the CPU while the GPU runs steps
N−k … N−1, with the outer buffer sized for nstlist + k. Double-buffer the nbnxm GPU data (list, atom index
maps, exclusions, nbat parameters) and swap it at step N. The GPU-resident state is in local order, so
only the nbnxm-ordered buffers and the GPU bonded lists switch.

**Bound** 8.4%; estimate 5-7% (a slightly larger outer list costs the prune kernels a little). L effort.
Validate with `gmxbench quality` (EQUIVALENT) and an NVE drift check (`phaseB-energy-2026-10-07`
procedure).

## H5: faster CPU pair search (ANALYSIS 8)

`nbnxn_make_pairlist_part` is the search: 16 threads at 0.27-0.32% of samples each, i.e. balanced, so
the cost is per-thread compute.

* About 40% is `makeClusterListSupersub` (`pairlist.cpp` 856-915). Bounding-box distances are computed four
  at a time with SIMD4 (`xmmintrin`, 128-bit), then a scalar loop over 8 i-clusters tests
  `d2 < rlist2` and ORs one `imask` bit at a time.
* About 14% is `setExclusionsForIEntry` (1590-1660), which finds the j-cluster of each excluded atom by
  binary search (1039-1055).

**Proposal.** With AVX-512 (the build's SIMD level), one 8-wide compare yields the 8-bit mask directly
(`_mm256_cmp_ps` + `movemask`, or a `__mmask8`). Exclusions could use a direct j-cluster → list-index map
built once per i-entry.

**Estimate.** Search −25-35% → +2-3% (overlaps with H4; still useful because H4 can only hide what fits in
k steps). M effort; bit-identical list.

## H6-H9 (smaller)

* **H6 NB tail.** At 540-580 µs the SMs drain: 77% → 48% active, issue 48% → 19%. That is ≈ 24 µs of lost
  issue capacity per step. The list is already sorted largest-first (`nbnxnKernelBucketSciSort`), 4.93
  waves of 11,212 blocks. Ideas: split the largest super-clusters further in the last wave, or a persistent
  kernel with an atomic work counter.
* **H7 Update chain.** 580-640 µs at 17-26% issue.
  * LINCS: 150 blocks (0.18 waves, 17% achieved occupancy).
  * The rolling prune (1,402 blocks) is launched before leap-frog/LINCS, so it takes the SMs first and LINCS
    doubles (day 2).
  * Options: launch the prune after SETTLE (it only needs the coordinates of this step); smaller LINCS
    blocks; fuse reduce + leap-frog.
* **H8 Gather and FFT/solve.**
  * Gather has spread's scattered pattern (L2 64%, `mio_throttle`); tile-staged reads in the H2 order would
    help.
  * The FFT kernels are sub-wave (0.39-0.62 waves).
  * The solve could move into cuFFT LTO callbacks (CUDA 12.6+, no static linking needed).
* **H9 Bonded.** `LDG` 36% / `RED` 23% of stall samples. iatoms in SoA layout (today 20-byte stride, 0.20
  sector efficiency) and per-block shared-memory force accumulation: molecules are contiguous, so a
  block's atoms overlap.

## Analysed and not recommended

| idea | why not |
|---|---|
| Hand-written SASS (or PTX) for the NB kernel | with (a)-(c) the body has no wasted instruction left (43 instructions: LJ fetch 4, rsqrt, Coulomb 12, switch 6, LJ 8, 7 accumulation FFMAs, exclusion mask 3); the limits are issue slots and the 64-register cap, which hand scheduling cannot lift; no supported assembler for sm_89 |
| Tensor cores (TF32/FP16/BF16) for cluster distances, spread outer products or the PME FFT | 1e-3 relative precision fails the quality policy (GPU noise floor ~1e-7); 3xTF32 removes the gain; a spread tile as a GEMM is ~20x sparse |
| `RED.F32x4` vector atomics for spread | sm_90+ only (nvcc rejects float2/float4 `atomicAdd` for sm_89) |
| Lower-degree Ewald correction ((4,4), (3,4)) or a pure polynomial | 1.3-5x the production error relative to 1/r³; a polynomial needs degree ≥ 8 |
| Float shared-memory atomics in a spread rewrite | CAS loop on sm_89; use int32 fixed point (H2) |
| More NB occupancy or registers | re-confirmed: occupancy is register-limited (64%/67%), and day 1/day 2 measured both directions slower |
| Memory-side NB optimisations (LJ table, coalescing) | L2 20%, DRAM 9%, L1 42%: memory is not a limiter |
