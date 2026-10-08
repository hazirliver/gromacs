# Measurement plan (written 19:20 UTC, ~17 min into the session; budget ~8 h machine time)

Primary metric ms/step (also ns/day, Matom-steps/s). Production build `P` (cuda-nosub) for all timings;
sub-counter/NVTX build `S` (cuda) for cycle sub-counters and NVTX step ranges, with its overhead
measured against P. Steady state only: `-notunepme` (the tuner keeps rc 1.20 / 96x96x144 anyway, see
NOTES 19:13) and `-resetstep`; nsys captures are bounded to the steps after the counter reset
(`--capture-range=cudaProfilerApi`). Experiments run as interleaved rounds (bench/profiling/prun), 5
repeats unless stated, with NVML telemetry on every timed window.

| # | Block | What | Hypothesis | Est. machine time |
|---|---|---|---|---|
| 1 | Run level | `baseline` experiment (prod-P, prod-S, plain-P); gmxbench perf A/A (P vs P) and P vs S on `perf-mas1-production` | H1, tool overhead | 35 min |
| 2 | PCIe probe | 4 B and 2.2 MB H2D/D2H latency/bandwidth from vCPUs 0/10/20/30/39 (replaces GPU-local vs remote socket: VM has 1 socket/NUMA node) | H4 pinning | 3 min |
| 3 | Step timeline | nsys on prod-S (full: cuda, nvtx, osrt, CPU sampling, ctxsw, graph nodes, GPU metrics), prod-S light, prod-P light; nsys_steps.py per-step-type analysis | H2 critical path | 25 min |
| 4 | CMAP | `cmap` experiment: prod vs nocmap x GMX_CUDA_GRAPH on/off (P); nsys of nocmap+graphs; md.log "MD Graph" row | H2 | 20 min |
| 5 | Power | `power` experiment: 250/300/325(current)/350 W limits + one clock lock, perf + energy/ns; restore 325 W | H3 | 25 min |
| 6 | Sweeps | ntomp {4,8,12,16,24,32}; nstlist {100,200,300,400}; nstcalcenergy {100,500,1000} (+ all global intervals); pinning (stride 1/2, off); graphs on/off; PME rc 1.20-1.40 fixed points; kernel env knobs (ana/tab Ewald, twin-cut, GMX_NB_MIN_CI, dynamic pruning interval) | H4 | 90 min |
| 7 | Ablation ladder | nocmap, potshift, rf, nolincs, noconstr, nocmap-potshift (P, timing) + nsys light of each (S) | Amdahl numerators | 35 min |
| 8 | Kernels | ncu --set full on NB F and VF kernels, prune kernels, PME spread/gather/solve, cuFFT, bonded, x->nbat, F reduction, LINCS, SETTLE, leap-frog | limiters | 60 min |
| 9 | CPU | perf record (dwarf call graphs) attached after the counter reset; CMAP, pair search, waits, OpenMP spin; NS time vs threads from the ntomp sweep | CPU side | 20 min |
| 10 | Contrast modes | nsys light of bonded-on-CPU, NB+PME GPU with CPU update, NB-only, only where they explain something | - | 20 min (first to cut) |
| 11 | Synthesis | summary.json/csv, ANALYSIS.md, OPPORTUNITIES.md, EXEC-SUMMARY.md | - | (no machine time) |

Total ≈ 5.5-6 h of machine time, leaving slack for re-runs. Cut order if needed: 10, then parts of 6
(pinning, nstcalcenergy variants), never 8 on the production configuration.

Expected outcomes to test (from prior work, treated as hypotheses):
- H1: ~234 ns/day = ~0.738 ms/step; 1.38x plain mdrun.
- H2: per-step D2H x + H2D f (~2.2 MB each) for CPU CMAP; graphs disabled; exposed part unknown.
- H3: sw_power_cap throttling, SM clock a few % below 2520 MHz, ~112-116 kJ/ns.
- H4: defaults (nstlist, nstcalcenergy, threads, pinning, PME tuner) suboptimal.

## Revision at 19:56 UTC (first hour done)

Done: blocks 1-4 (baseline, gmxbench A/A and P-vs-S, PCIe probe, nsys light/full/graph captures,
`cmap` ablation, perf profiles). Findings so far change the priorities:
- H1 confirmed; H2 mechanism confirmed but its exposed cost is ~1% (ceiling of a CMAP port + graphs
  +0.9% [0.3, 1.5]); the large non-kernel costs are pair-search steps (~10% of run time, 12.7 ms GPU idle
  per search) and energy/COM-removal steps (~1-2%).
- **GPU performance counters are blocked by DCGM** (nv-hostengine holds the profiling resources):
  ncu and nsys GPU-metrics sampling fail. Pausing DCGM is not pre-authorised -> documented with the
  admin commands; kernel-level work uses substitutes: (a) clock-scaling of kernel durations at locked
  SM clocks 1500/2010/2520 MHz (nsys), (b) theoretical occupancy/limiter from launch records,
  (c) ablations. If the user pauses DCGM later, the ncu block (bench/profiling/ncu_capture.sh) is ready.

Queue2 (started 19:52, est. end ~22:30): perf profiles, power (8 variants incl. CMAP ceiling at
325/350 W), threads, nstlist, ablation ladder (+ nsys of each ablation), knobs (Ewald variants,
GMX_NB_MIN_CI, prune interval, alternating wait), nstcalcenergy, PME fixed points + tuner, pinning,
contrast-mode captures, clock-scaling captures, final GPU settings restore.
Analysis in parallel: step budget, critical path, CPU profile, occupancy, synthesis documents.

## Outcome at 22:50 UTC (3 h 47 min after start; ~3.5 h of GPU time used)

Done as planned: blocks 1-7, 9, 10 (contrast modes), all sweeps, the ablation ladder (with the RF run
repeated without -pme gpu after a failed first attempt), power, clock locks, PME points (re-run after a
TOML key bug). Added beyond the plan: clock-scaling captures (counter-free limiter evidence), static
SASS/occupancy analysis, and six throwaway builds (launch bounds 20/24, stream priority swap, OpenMP for
nvcc host code, bonded kernel stream, combination) with quality gates.
Not done: block 8 (Nsight Compute) - GPU counters held by DCGM; the user chose not to pause DCGM (22:53).
