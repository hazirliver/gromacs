# MAS1 profiling, iteration 3: executive summary (2026-10-10)

**Scope**: the current `dev` (GROMACS 2026.4 + the five merged GPU optimisations, `56a6b0fd5e`) on MAS1
(185,486 atoms) on one L40S, production command `-ntmpi 1 -ntomp 16 -nb gpu -pme gpu -bonded gpu -update gpu
-nstlist 200`. Goal: find what is left and generate hypotheses. **No optimisation was implemented**; `src/`
is unchanged. Kernel "what-ifs" were compiled only to count SASS, never run.

**New this time: hardware counters.** DCGM's profiling was paused for a 2-minute window (with the node
owner's approval; `dcgmi profile --pause/--resume` works from the user account). That gave Nsight Compute
`--set full` on every kernel of a plain step (stall reasons, per-SASS-instruction warp sampling, L1/L2/DRAM
throughput) and an nsys GPU-metrics timeline at 10 µs. Also new: CPU perf with source lines for the pair
search, a static-SASS variant harness for the NB kernel, an offline PME-spread model on the MAS1
coordinates, and minimax fits for the Ewald correction.

**Where a step goes now** (nsys, 5,000 steps; mean 709 µs = ~244 ns/day under tracing): plain steps are
87% of the run time (628 µs median), pair-search steps 10.3%, energy steps 1%. Inside a plain step:

| phase (µs from step start) | what runs | SM issue utilisation |
|---|---|---|
| 0-80 | PME spread alone, holding 94% of all warp slots | **5%** |
| 80-140 | x→nbat, bonded, r2c FFT; NB kernel not yet started | 12-18% |
| 140-540 | NB kernel (+ PME FFT/solve/gather sharing SMs) | 80-87% |
| 540-580 | NB kernel tail (SMs draining) | 48% → 19% |
| 580-640 | reduce, leap-frog, rolling prune, LINCS, SETTLE | 17-26% |

DRAM traffic is ~0: the whole working set lives in the 96 MB L2.

**What the counters say**

* **NB kernel** (442 µs/step): issue-bound. IPC 3.24 of 4, and the top stall is "not selected": warps are
  ready but the issue port is busy. L2 20%, DRAM 9%, occupancy at its 64-register limit. **Only fewer
  instructions help.** It executes 95.6 warp instructions per computed pair body. The body itself is 53-54
  instructions, still including 5 constant re-materialisations. 23% of all instructions (38% of stall
  samples) are control flow: each of the 8 unrolled i-cluster mask tests costs 4 instructions, because the
  mask sits in a uniform register.
* **PME spread** (91 µs, on the critical path): **L2-throughput-bound at 87% of peak**, IPC 0.23. Stalls
  are the global-memory queue full (`lg_throttle`) and the atomics.
* **Bonded** (61 µs, overlapped): memory-instruction-queue-bound (strided index loads, force atomics).
  **LINCS** runs 0.18 waves at 17% occupancy.
* **CPU pair search**: 9.5 ms per search, evenly spread over 16 threads. It is 40% a bounding-box test with
  128-bit SIMD plus a scalar bit loop (the CPU has AVX-512) and 14% exclusion masking via binary search. The
  GPU idles 11.9 ms per search.

**Top hypotheses** (details, bounds and validation plans in `HYPOTHESES.md`):

| # | hypothesis | evidence | estimate | effort |
|---|---|---|---|---|
| H1 | **NB pair-loop instruction diet**: (a) mask test on a once-per-j-cluster shifted mask, (b) Ewald correction as a monic (5,4) rational in r² with host-scaled coefficients, (c) force-switch polynomial pre-combined per atom-type pair | static SASS of the variants weighted by measured execution counts: **−14.5% NB warp instructions**, 64 registers (occupancy kept); (b) is **2x more accurate** than production; (a) is bit-identical | **+6-7%** | S-M |
| H2 | **PME spread with shared-memory accumulation in int32 fixed point**, one global atomic per touched grid point | ncu: L2-bound; model on MAS1 coordinates: global updates −2.3x in today's atom order, −3.9x in spatial 64-atom blocks, −4.6-5.7x with grid tiles; float shared atomics are CAS loops on sm_89, int32 ones are native | **+6-8%** (spatial), +3-4% (current order) | L / M |
| H3 | **Fill the low-issue start of the step**: (a) launch the NB kernel before the bonded kernel; (b) start x→nbat + NB before spread and cap spread's residency so the L2-bound spread and the issue-bound NB share SMs | phase metrics; NB starts 42 µs after its input is ready; spread skip bound 81 µs | (a) +1.5-3%; (b) +3-6% (overlaps H2) | S / M |
| H4 | Overlap the CPU pair search with GPU steps (lagged, double-buffered list) | GPU idle 11.9 ms per 200 steps | ≤ +8.4% | L |
| H5 | AVX-512 bounding-box test with compare-to-mask, exclusions without binary search | perf with source lines | +2-3% (overlaps H4) | M |

Smaller items: the NB tail (~3% lost issue capacity), the update chain (1-2%), gather/FFT (1-2%) and the
bonded layout (~1%). Together the top items could plausibly give **+15-25%** over today's `dev`; the
gains are not all additive (H2 and H3b both target the same 81 µs).

**Analysed and not recommended**:

* Hand-written SASS for the NB kernel: once the source is fixed, ptxas's code has no waste, and there is no
  supported sm_89 assembler.
* Tensor cores: TF32/FP16 precision fails the quality policy.
* Vector float atomics: sm_90+ only.
* Lower-degree Ewald fits: they are less accurate than production.

**Gotchas found**:

* mdrun's own ns/day is wrong under an nsys capture range: it includes nsys's flush (2.1-3.1 ms/step
  reported; the 2026-10-07 captures show the same).
* ncu's `--cache-control all` makes small kernels look DRAM-bound.

Artifacts: `ANALYSIS.md` (all measurements), `HYPOTHESES.md` (ranked, with validation plans); raw data
in `bench/results/profiling-2026-10-10/` on the node; new tools in `bench/profiling/` (see
`../../OPTIMIZATION.md` section 6).
