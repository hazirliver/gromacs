# MAS1 single-GPU profiling: executive summary (2026-10-06)

**System and run**: MAS1 + Gi + 20E membrane (185,486 atoms, CHARMM36, 2 fs) on one NVIDIA L40S in a
20-core Xeon Gold 6338 VM, GROMACS 2026.4 (`dev`). Production command
`-ntmpi 1 -ntomp 16 -nb gpu -pme gpu -bonded gpu -update gpu -nstlist 200`.

**Where the time goes** (production-like build, 0.742 ms/step = 233.0 ns/day [231.9, 234.0], 1.37x plain
mdrun, 112.7 kJ GPU energy per ns):

* **87% plain steps, GPU-bound** (6.5 us GPU idle per 661 us step). Critical path: 165 us before the
  nonbonded kernel can start (the high-priority PME spread takes the SMs first, then x->nbat, then the
  bonded kernel serially), the **NB kernel itself 445 us (60% of all run time)**, then 51 us reduction +
  update/constraints. PME (340 us of kernel time per step) runs concurrently, off the critical path,
  but slows the NB kernel ~12%.
* **10.3% pair-search steps**: every 200 steps the GPU idles 12.7 ms while the CPU builds the grid
  (1.7 ms), the pair list (8.6 ms, 16 threads) and the GPU bonded lists (2.1 ms, **single-threaded
  because an OpenMP pragma is compiled by nvcc without -fopenmp**).
* **~1% energy/COM-removal steps** (blocking host round trips), **~1% the CPU CMAP path**.
* The GPU is **power-capped at an enforced 325 W** (cannot be raised on this node): SM clock 2362 MHz,
  6.3% below max, `sw_power_cap` 100% of the time; the cap costs ~6.5% vs an uncapped clock.

**Hypotheses**: H1 (234 ns/day, 1.38x) confirmed. H2 (CPU CMAP forces copies and disables CUDA graphs)
mechanism confirmed, but **refuted as a bottleneck**: the copies hide under the PME/NB kernels; removing
CMAP *and* enabling graphs gains only +0.9% [0.3, 1.5]. H3 (power throttling) confirmed and larger than
assumed. H4 (suboptimal runtime defaults) mostly refuted: nstlist 200, rc 1.20 / 96x96x144 (also the PME
tuner's choice), pinning and kernel-variant env vars are already optimal; only -ntomp 20 (+0.6%) helps.

**Top opportunities** (OPPORTUNITIES.md has bounds, effort, risk, prior art and validation per item):
1. **Fix the serialised OpenMP loop in nvcc-compiled host code** (S, bit-identical, passed `quality --strict`):
   measured **+0.85% [0.38, 1.32]** in a throwaway build (bound 1.4%).
2. **Run the GPU bonded kernel in its own stream** (S-M, within GPU noise, passed the quality gate):
   measured **+0.91% [0.29, 1.53]**.
3. **-ntomp 20** (config): +0.58% [0.15, 1.02].
   **Combined 1 + 2: +2.03% [1.42, 2.65]; with 3: +2.82% [2.19, 3.45] = 239.0 ns/day** (all three are
   S effort; the combined build passed `quality --strict`).
4. Cheaper force-switch arithmetic in the NB kernel (S-M): bound 5.4% (40 us/step measured), estimate 1-2%.
5. Update-chain scheduling (LINCS || SETTLE, rolling prune off the update path) (M): estimate 1.5-2.5%.
6. CUDA graphs despite CPU CMAP / CMAP on the GPU (M-L): ceiling +0.9%; prior art MR !1730 (closed),
   prototype commit 0030776d (unmerged).
7. Energy/COM steps without blocking host copies (M): 0.5-0.8%.
8. Overlapping the CPU pair search with GPU steps (L): bound 8.5-10%, the largest non-kernel item.
Measured and rejected: higher NB-kernel occupancy via launch bounds (-4% / -16%: spills and lower clocks
under the power cap), swapped stream priorities (-0.6%), other nstlist / PME grids / kernel env knobs /
pinning, raising the power limit (enforced at 325 W).
Envelope for context: everything PME costs 24% (reaction-field ablation), the CHARMM force switch 6.9%.

**Gaps**: GPU hardware counters were unavailable because DCGM holds them, so there are no Nsight Compute
metrics (occupancy achieved, stalls, cache hit rates, roofline) and no nsys GPU-metric timelines. Kernel
claims rest on clock scaling (the NB kernel scales with the SM clock, alpha 0.90, so it is not DRAM-bound),
launch records (NB kernel register-limited at 66.7% occupancy, yet more occupancy is measurably
slower), static SASS, ablations and throwaway builds. To enable the
prepared kernel-level block (`bench/profiling/ncu_capture.sh`), an admin would run:

```
dcgmi profile --pause          # pause DCGM profiling-metric collection (node monitoring gap!)
# ... ncu runs / nsys --gpu-metrics-devices=0 ...
dcgmi profile --resume
```
(heavier alternative: `sudo systemctl stop nvidia-dcgm` ... `sudo systemctl start nvidia-dcgm`). CPU energy
is not observable in the VM (no RAPL). No second socket is visible (NUMA test not possible).

Artifacts: `ANALYSIS.md` (layered analysis), `OPPORTUNITIES.md`, `NOTES.md` (lab notebook), `PLAN.md`,
`summary.json`/`summary.csv` (every measurement with source and uncertainty), `raw/` (logs, nsys, perf,
telemetry). Scripts: `bench/profiling/` on branch `feature/profiling`.
