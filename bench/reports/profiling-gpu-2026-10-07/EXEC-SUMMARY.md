# MAS1 GPU kernel deep dive: executive summary (2026-10-07)

**Scope**: the GPU work of the GPU-bound production run of MAS1 (185,486 atoms, CHARMM36, 2 fs) on one
NVIDIA L40S, GROMACS 2026.4 (`dev`), production command `-ntmpi 1 -ntomp 16 -nb gpu -pme gpu -bonded
gpu -update gpu -nstlist 200` (0.743 ms/step = 232.6 ns/day on the production-like build). Follows day 1
(`../profiling-2026-10-06/`). All code changes were made in a throwaway worktree; `src/` is untouched.

**How** (hardware counters are still held by DCGM): exact per-instruction executed counts, divergence,
and memory-sector efficiency for every kernel through a new CUPTI binary-patching profiler
(`bench/profiling/sassprof`, works under DCGM), mapped to source lines. Kernel times were taken alone
and concurrent at a locked clock. Kernel rewrites were timed by in-process replay on live data. Upper
bounds came from timing-only kernel skips, and end-to-end A/B used 5 interleaved rounds.

**Where the GPU time goes**
* **NB force kernel (60% of run time) is issue-bound**: 433 M warp instructions per step at 79% of the
  issue limit. Each pair-interaction body is 61 instructions, and the LJ force switch (16) and Ewald
  real-space correction (17) take more than half of that. 52% of the lanes sit idle in that body
  because of cluster geometry. Memory is not a factor: 0.82 sector efficiency, no bank conflicts.
* **PME kernels cost 18.3% of the step** although they run concurrently. Skipping PME spread alone saves
  81 us/step (10.9%), because 91% of its time is on the critical path: it holds every SM at the start of
  the step at an IPC of 0.24. 92% of spread's time is 11.9 M scattered grid updates. Atomicity (18%) and
  atom order (2-5%) are not the limiter; the number of global updates is. Gather and FFT/solve cost
  about 25 us each through SM sharing with the NB kernel.
* **Bonded, LINCS and SETTLE are latency-bound** (IPC 0.10-0.26). SETTLE reads coordinates at 1/8
  sector efficiency. LINCS doubles whenever it shares SMs.

**Measured gains** (throwaway build, vs production, 95% CI):

| change | end to end | effort |
|---|---|---|
| NB kernel: factored force switch + Ewald correction in Horner form (same coefficients and accuracy; 6.6% fewer instructions, kernel -6.1%) | **+2.57% [2.11, 3.03]**, -3.6% energy per ns | S (`nbnxm_kernel_utils.h`, two functions) |
| SETTLE with coalesced shared-memory staging (5x fewer sectors) | +0.44% alone, ~+1.2% in the stack | S (`settle_gpu_internal.cu`) |
| bonded kernel in own stream (day 1) / LINCS concurrent with SETTLE | +0.48% / no change alone, +0.5-0.7% in the stack | S |
| **all of the above + day 1's OpenMP fix + `-ntomp 20`** | **+7.19% [6.58, 7.80] = 249.4 ns/day, -5.7% GPU energy per ns** | S, ~5 files + build flag |

That stacked configuration **passed `gmxbench quality --strict` (quick tier, all 63 cases)**: all 72
bitwise-reproducible configurations IDENTICAL, GPU configurations EQUIVALENT within 0.5-1.7x their own
noise floor, no DIFFERENT and no FAIL. Of the +7.2%, about +4.3% comes from today's GPU-code changes and
about +2.8% from day 1's.

**Largest remaining opportunity**: a PME spread rewrite (atoms in spatial order, accumulation in
shared memory, one global atomic per touched grid point). The bound is 10.9%; my estimate of 3-6% is
not prototyped. Upstream has the same idea as an open HIP-only MR (!6000) with no CUDA port. Gather
and FFT/solve are bounded at 3.5% and 3.2%, LINCS restructuring at about 2.9%, and the bonded layout at
about 1.6%.

**Measured and rejected**: hoisting the LJ table fetch (slower), more registers for the NB kernel at
12 or 10 blocks per SM (-1.7% end to end), 16 threads per atom for PME (-3.9%), interleaving or
spatially sorting the spread atoms (no gain), and LINCS concurrent with SETTLE on its own (no gain;
day 1's 1.5-2.5% estimate refuted).

**Gaps**: there are still no stall reasons, cache hit rates or DRAM/L2 throughput. Nsight Compute, CUPTI
PM sampling and CUPTI PC sampling are all blocked by DCGM. The admin commands to free the counters are
in day 1's EXEC-SUMMARY (`dcgmi profile --pause` / `--resume`). Leap-frog could not be profiled (a CUPTI
limitation).

Artifacts: `ANALYSIS.md`, `OPPORTUNITIES.md`, `NOTES.md` (lab notebook), `PLAN.md`,
`summary.json`/`.csv` (1010 records), `raw/` (sass profiles, replays, nsys, experiments, quality gate,
prior art). Tools: `bench/profiling/sassprof/`, `nb_replay.sh`, `gpu_records.py`; throwaway patch
`bench/profiling/experiments/exp-gpukern.patch` (worktree `/home/asokolov/Projects/gromacs-exp-gpukern`).
