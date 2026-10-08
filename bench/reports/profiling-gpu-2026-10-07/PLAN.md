# Plan: GPU kernel deep dive, MAS1 on one L40S (2026-10-07)

Continuation of `bench/results/profiling-2026-10-06/` (referred to below as **day 1**). Day 1 established that
plain steps (87% of run time) are GPU-bound and that the step's critical path is pre-NB work (165 us) ->
NB F kernel (445 us, 60% of all run time) -> reduction + update chain (51 us), with PME concurrent.
Question for today: **which GPU kernels and GPU-side code paths could be made faster by rewriting or
optimising code, how much each could gain for MAS1, and what the evidence is.**

Rules (unchanged): `src/` untouched except in the throwaway worktree
`/home/asokolov/Projects/gromacs-exp-gpukern` (detached at `dev`, every change behind a `GMX_EXP_*`
define); any build that is kept for a gain claim must pass `gmxbench quality --strict` (quick tier, all
cases, not only the target); every number traces to an artifact under `raw/` and to a record in
`summary.json`; ablation inputs are timing-only copies under day 1's `raw/inputs` (labelled).

## Tooling status (checked 08:40-09:00)

| tool | status under DCGM | use |
|---|---|---|
| Nsight Compute, CUPTI PM sampling | blocked (`CUPTI_ERROR_HARDWARE_BUSY` / driver resource unavailable) | - |
| CUPTI PC sampling (stall reasons) | blocked (`cuptiPCSamplingStart` -> CUPTI_ERROR_UNKNOWN) | - |
| **CUPTI SASS metrics (binary patching)** | **works** | exact per-instruction executed counts (warp/thread/predicated), branch divergence, global sectors vs ideal (coalescing), L1 tag lookups, shared wavefronts vs ideal (bank conflicts), local sectors (spills) |
| nsys (CUDA + NVTX trace) | works | kernel durations, overlap, step timeline |
| NVML telemetry, clock locks | works (sudo, as day 1) | clock-scaling, cycles at fixed clock |

New tool `bench/profiling/sassprof/` (CUPTI injection library). CUPTI 2025.3 quirks found and worked
around: data is collected only for the first kernel launched after `cuptiSassMetricsEnable`, nothing
after the first flush, and only if `SetConfig` immediately precedes `Enable`; so one mdrun per kernel,
enabled at the exit of the launch preceding the target (`run_targets.sh`). Leap-frog cannot be collected.
The line-info build (`-lineinfo`, profile `cuda-nosub-lineinfo`, `~/.cache/gmxbench/builds/5d0e9c44d613eed1`)
has SASS identical to the production-like P build (all 364,361 instructions), so its counts apply to P.

## Blocks

| # | block | method | output | machine time |
|---|---|---|---|---|
| A | dynamic instruction profile of every kernel | sassprof, 14 kernels x 400 plain steps; disassembly with line info | per-kernel instructions/launch, SIMT and predication efficiency, opcode-class mix, memory efficiency, divergence | 0.3 h |
| B | issue/pipe model | counts + nsys durations at known clock -> achieved IPC, per-pipe utilisation lower bounds (FP32, INT, MUFU, LSU/shared, TEX, issue) | which resource bounds each kernel; headroom | 0.2 h |
| C | NB kernel anatomy | line-info regions (setup, j-loads, distance/mask, LJ table fetch, LJ, force switch, Ewald, accumulation, j-reduction, i-reduction); useful-work fraction from thread-level counts of the cut-off branch | share of the kernel per code region; fraction of computed pairs inside the cut-off | 0.3 h |
| D | in-process kernel replay (throwaway) | mdrun hook replays the NB kernel on the live data at a chosen step, isolated, locked SM clock, K repeats per variant; knock-out variants (timing only) and rewrite candidates, forces compared with the reference launch | cost of each region in cycles; candidate speedups and max force error before any end-to-end run | 2 h |
| E | sensitivity: kernel us -> step us | (a) NB: end-to-end effect of measured kernel speedups; (b) PME/update/small kernels: "kernel skipped" throwaway variants (timing only) under nsys + prun -> exact ceilings with real concurrency | upper bound per kernel / kernel group | 1.5 h |
| F | prototypes end-to-end | best 2-3 rewrites as GMX_EXP builds: prun A/B (5+ rounds) + nsys + `quality --strict` | measured gains | 2 h |
| G | report | NOTES, ANALYSIS, OPPORTUNITIES, EXEC-SUMMARY, summary.json/csv | | 1 h |

Order: A -> B -> C (analysis only) -> D (needs a build, ~25 min) -> E (builds shared with D) -> F -> G.
Decision points: after C, pick the D variants from the region shares; after D, pick F prototypes by
(cycles saved x share on the critical path) / effort; drop E(b) items whose kernels day 1 already bounded.

## Hypotheses to test (from day 1 + code reading)

* K1: the NB F kernel is issue/FP32-bound (alpha 0.90, more occupancy slower) -> only fewer instructions
  per pair help.
* K2: a large share of computed pairs is outside the cut-off (8x4 warp tiles of 8-atom clusters, inner
  list buffer 1.203 nm) -> work efficiency, not arithmetic, dominates.
* K3: the force switch (r > 1.0 nm only) is evaluated for every pair; a warp-uniform skip plus factored
  polynomials removes most of its 40 us.
* K4: the per-pair LJ parameter texture fetch (NBFIX table, no combination rule) is a significant cost.
* K5: PME spread (alpha 0.64) is bound by global atomics / scattered grid access; gather by global loads.
* K6: the update chain kernels are latency-bound sub-wave launches; their cost is launch+tail, not work.

## Outcome (10:50)

| block | status |
|---|---|
| A dynamic instruction profiles | done: 14 kernels (leap-frog not collectable), `raw/sass/` |
| B issue/pipe model | done: `raw/sass/analysis-iso-2100.*`, isolated/concurrent times `raw/nsys/{iso,conc}-2100-P` |
| C NB anatomy | done: regions, 61-instruction body, pair efficiency 48% |
| D in-process replay | done: NB (14 variants) and PME spread (orderings + knock-outs) |
| E sensitivity | done: PME skips (spread 10.9%, gather 3.5%, FFT/solve 3.2%, all 18.3%); bonded skip impossible (blows up) |
| F prototypes end to end | done: NB rewrite +2.57%, SETTLE staging, concurrent SETTLE, bonded stream; stack +7.19% |
| quality gate | passed `--strict --tier quick` for the full stack (`raw/gmxbench/quality-XO-final`) |
| G report | done |

Hypotheses: K1 confirmed, K2 confirmed, K3 partly, K4 refuted, K5 refined (update count, not atomics or
order), K6 confirmed. Not done: spread rewrite (L effort), gather/FFT/LINCS/bonded rewrites (bounds only).
