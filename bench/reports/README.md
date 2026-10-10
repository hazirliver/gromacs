# Performance reports

Write-ups of the optimisation work on this fork. Start with [`../OPTIMIZATION.md`](../OPTIMIZATION.md),
which summarises the state, what was rejected, what is open and which tools exist.

| Report | Content |
|---|---|
| [`2026-10-gpu-kernel-opt.md`](2026-10-gpu-kernel-opt.md) | Results of the five GPU-path commits merged into `dev`: MAS1 +6.5%, other systems, validation |
| [`profiling-2026-10-06/`](profiling-2026-10-06/) | Whole-step profile of the MAS1 production run on one L40S: CPU/GPU critical path, pair search, CMAP, power cap, run settings, first prototypes |
| [`profiling-gpu-2026-10-07/`](profiling-gpu-2026-10-07/) | GPU kernel deep dive: per-kernel instruction profiles (sassprof), PME spread analysis, kernel prototypes, the hand-off plan (`NEXT-STEPS.md`) |
| [`phaseB-energy-2026-10-07/`](phaseB-energy-2026-10-07/) | NVE energy conservation of MAS1 before/after the changes; `lincs-iter` finding |
| [`profiling-2026-10-10/`](profiling-2026-10-10/) | Iteration 3 on the optimised `dev`: first hardware counters (Nsight Compute, GPU-metrics timeline), NB SASS anatomy and static what-ifs, PME spread model, CPU search with source lines, Ewald-correction fits; ranked hypotheses (`HYPOTHESES.md`), nothing implemented |

The session folders keep their original names so their cross-references work. Each has `EXEC-SUMMARY.md`
(read first), `OPPORTUNITIES.md`, `ANALYSIS.md`, `NOTES.md` (lab notebook) and `PLAN.md`; the 2026-10-10
folder has `EXEC-SUMMARY.md`, `HYPOTHESES.md` (in place of `OPPORTUNITIES.md`), `ANALYSIS.md` and
`NEXT-STEPS.md` (hand-off prompt for implementing the hypotheses). Their `raw/`
data (logs, nsys captures, tprs, `summary.json`) is not in git: it is large and partly customer-derived, and
stays in `bench/results/` on the benchmark node. Minor edits to the copies: two lines about node access were
reworded.
