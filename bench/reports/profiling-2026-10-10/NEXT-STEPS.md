# Hand-off: implement and validate the iteration-3 performance hypotheses (prompt for a new agent)

Written 2026-10-10 at the end of the iteration-3 profiling session. You are an agent starting with no other
context. Your task: **implement the optimisations proposed in this directory, test each one end to end,
keep only what is measurably faster and passes the quality gate, and update the reports.** Read this
file fully before doing anything.

Repository: `/home/asokolov/Projects/gromacs` (a GROMACS 2026.4 fork). The rules in `CLAUDE.md` at the
repository root override anything here. Then read, in this order:

1. `bench/OPTIMIZATION.md`: state, tools, rejected ideas, gotchas.
2. `bench/TESTING.md`: the required validation procedure.
3. This directory: `EXEC-SUMMARY.md`, `HYPOTHESES.md` (the ideas, with bounds and evidence) and
   `ANALYSIS.md` (the measurements).
4. `../profiling-gpu-2026-10-07/NEXT-STEPS.md` section 9: how the previous hand-off was executed.

## 1. Situation

* `dev` = `56a6b0fd5e`: GROMACS 2026.4 + gmxbench + five GPU optimisations (A1-A5). MAS1 production
  233.0 → 248.3 ns/day.
* Target workload: MAS1 (185,486 atoms, CHARMM36, PME, LJ force switch, 2 fs, h-bond LINCS + SETTLE) on one
  NVIDIA L40S (sm_89, 142 SMs, 96 MB L2, power-capped at 325 W) in a 20-core Xeon 6338 VM (AVX-512).
  Production command, gmxbench config `perf-mas1-production`:
  `gmx mdrun -ntmpi 1 -ntomp 16 -nb gpu -pme gpu -bonded gpu -update gpu -nstlist 200`.
* Iteration 3 (this directory) profiled `dev` with hardware counters and found:
  * the NB kernel is issue-bound (only fewer instructions help);
  * PME spread is L2-atomic-bound and blocks the start of every step;
  * the NB kernel starts 42 µs late because the bonded kernel is launched before it;
  * the CPU pair search costs 10.3% of run time.
* **Nothing from `HYPOTHESES.md` is implemented yet.**
* The iteration-3 report and its tools (`bench/profiling/{counters_window.sh, ncu_summary.py, sass_hot.py,
  gpumetrics_phase.py, launch_gaps.py, spread_tiles.py, pmecorr_fit.py, pmecorr_weighted.py, nbvariants/}`)
  were written on branch `feature/profiling-2026-10-10` in the main checkout. Check `git log`/`git status`
  there: if they are committed, merge that branch into your feature branch first; if not, ask the user to
  commit them (or copy them) before you start. Your plan depends on them.

## 2. Goal and deliverables

On a new feature branch from `dev`:

* One clean commit per optimisation that is kept. No `GMX_EXP_*` switches or experiment code in the final
  commits; prototypes go into a separate throwaway worktree.
* Every kept commit validated per `bench/TESTING.md` against its parent: `gmxbench all --tier quick
  --strict`, MAS1 `perf-mas1-production` faster with the 95% CI excluding 1, no DIFFERENT/FAIL, no slower
  case.
* The before-merge checks of section 7 passed on the whole branch.
* Reports, if the branch is faster than `dev`:
  * a new results report `bench/reports/2026-10-perf-iter3.md` (model it on
    `bench/reports/2026-10-gpu-kernel-opt.md`);
  * `bench/OPTIMIZATION.md` updated (sections 1, 3, 4, 5);
  * a status section appended to this directory's `HYPOTHESES.md`;
  * `bench/reports/README.md` indexed.

  Negative results are recorded too (CLAUDE.md: "record every new result there, positive or negative"):
  rejected items go into `OPTIMIZATION.md` section 5 with their numbers, even if no code is kept.
* Do **not** merge into `dev` and do **not** push. Commit locally on the feature branch, then report to the
  user with the exact merge and push commands (the user pushes; see section 3).

## 3. Rules and environment (read before running anything)

* **Branches**: `git worktree add -b feature/perf-iter3 /home/asokolov/Projects/gromacs-perf-iter3 dev`.
  Develop and run gmxbench from inside that worktree with `-B WORKTREE`. Since `c651203516`, WORKTREE
  builds are keyed by path, so each worktree has its own build. Prototypes: a second throwaway worktree
  (e.g. `/home/asokolov/Projects/gromacs-exp-iter3`, detached at `dev`); never merge it.
* **Git**: no global identity. Commit with
  `git -c user.name=asokolov -c user.email=sokolov.ars.a@nebius.com commit ...`. Commit locally only; the
  user runs `git push` themselves (never push, never probe credentials). Commit messages name the gmxbench
  session, quality counts and the MAS1 speedup with CI, and say whether FP order changes.
* **Customer data**: `data/` and every tpr/gro derived from MAS1 (e.g.
  `bench/results/profiling-2026-10-06/raw/inputs/prod/topol.tpr`, `bench/results/profiling-2026-10-10/analysis/mas1.gro`)
  must never be committed. `bench/results/` is git-ignored; keep it that way.
* **Node**: check that it is quiet before benchmarking (`uptime`, `nvidia-smi`). Never run two gmxbench
  sessions at once. Run any CPU-heavy side work on cores 32-39 (`taskset -c 32-39`; mdrun `-ntomp 16 -pin on`
  uses CPUs 0, 2, …, 30). Do not change the GPU power limit. If you lock clocks (`sudo nvidia-smi -lgc`),
  reset with `-rgc` and verify.
* **Hardware counters**: DCGM holds them. `dcgmi profile --pause`/`--resume` works from the user account,
  but it gaps the node's monitoring: **ask the user (AskUserQuestion) before every pause**, keep the window
  short, and use `bench/profiling/counters_window.sh` (EXIT trap resumes DCGM). Counter-free tools usually
  suffice: nsys traces, `sassprof`, static SASS, gmxbench.
* **Gotchas** (from `OPTIMIZATION.md` section 7 and this session):
  * Re-running CMake in an nvcc build dir (incl. `gmxbench upstream`) drops `-D_FORCE_INLINES`; rebuild
    from a fresh configure before any perf A/B after that.
  * mdrun's own ns/day is wrong under an nsys capture range; use `nsys_steps.py`.
  * `pkill -f PATTERN` can kill your own shell; kill by PID.
  * ncu `--cache-control all` makes small kernels look DRAM-bound.
  * Short single mdrun timings vary by ~5%; only trust interleaved A/B (gmxbench, `prun`).
* **Measurement resolution**: gmxbench/`prun` A/A noise on MAS1 gives a minimum detectable effect of
  ~0.7% with 5 repeats. Use `--repeats 9` for expected effects below ~1.5%.

## 4. Work plan, in order (cheapest and safest first)

For every item: implement, then confirm the mechanism with the indicated check, then run the per-item gate
(section 6). Keep it only if it is faster; otherwise revert and record the negative result. Re-measure each
item **on top of the previously kept ones**, because gains interact (e.g. H3a and H2 both shorten the
start of the step).

### Phase 0: setup and baseline (≈ 1 h)

1. Read the files listed at the top. Create the worktree/branch (section 3) and bring in the iteration-3
   tools (section 1).
2. Build `dev` and the worktree: `bench/bin/gmxbench build dev --profile cuda-nosub` and `--profile cuda`.
   Confirm `gmxbench list --tier quick --target-only` shows the MAS1 cases (no "target unavailable"
   warning).
3. Optional mechanism baseline: `NSYS_LIGHT=1 bench/profiling/nsys_capture.sh` on the `cuda` (NVTX) build,
   then `nsys_steps.py`, `budget.py` and `bench/profiling/launch_gaps.py` (expect NB start ≈ 134 µs after
   spread start, NB kernel ≈ 442 µs/step, spread ≈ 91 µs).

### Phase 1, H3a: launch the NB kernel before the bonded kernel (S, expected +1.5-3%, bit-preserving)

* Where: `src/gromacs/mdlib/sim_util.cpp`, local GPU work block (~lines 1820-1840):
  `fr->listedForcesGpu->setPbcAndlaunchKernel(...)` is called before `do_nb_verlet(...)`, so the bonded
  kernel's 2,130 blocks are queued ahead of the NB kernel and take the SMs when spread finishes.
* **Trap**: commit A4 (`e2546053cf`) records `kernelCanStart_` in the NB local stream inside
  `ListedForcesGpu::Impl::launchKernel()` (`listed_forces_gpu_internal.cu`). If you only move the call
  after `do_nb_verlet`, that event is recorded *after* the NB kernel, and the bonded kernel then waits for
  the whole NB kernel (serialised, slower). Split the API: record the start event (and do the PBC/parameter
  setup) where the call is today, and enqueue the kernel launch in its own stream after `do_nb_verlet`,
  before `enqueueWaitForKernel()`.
* Handle the second call site (~line 1946, the non-local/DD branch) consistently, the CUDA-graph capture
  path (the fork/join must still be captured), and the SYCL/HIP builds (compile guards as in A4).
* Check: nsys anatomy shows the NB kernel starting right after x→nbat/r2c (~100 µs instead of 134) and the
  bonded kernel overlapping it. Gate with `--strict` (scheduling only).
* Alternative if (a) does not help: keep the order and give the bonded stream a lower priority than the NB
  stream (CUDA allows a range of priorities; `DeviceStreamPriority` has only Normal/High today).

### Phase 2, H1: NB pair-loop instruction diet (S-M, three commits, expected +6-7% together)

All three edit the CUDA NB kernel (`src/gromacs/nbnxm/cuda/nbnxm_cuda_kernel.cuh`) and its helpers. The
static what-if harness `bench/profiling/nbvariants/` (`NBVAR_SRC`, `NBVAR_BUILD`, `mkvariant.py`,
`compile.sh`, `seg.py`, `weigh.py`) reproduces the production SASS from a patched source copy. Use it to
check each real change before building: static instruction count, registers ≤ 64 (16 blocks/SM must stay;
`cuobjdump -res-usage`), no local memory. Measured weights per launch: 4.247 M pair bodies and 8.23 M
i-slot mask tests out of 406 M warp instructions.

**H1a, mask test on a shifted mask (bit-identical, −4.1% NB instructions).**
* In the j-cluster (`jm`) loop compute `imaskJ = imask >> (jm * c_superClusterSize)` and
  `wexclJ = wexcl >> (jm * c_superClusterSize)` once.
* Test `imaskJ & (1U << i)` (with `i` a compile-time constant after unrolling) instead of
  `imask & mask_ji`, and compute `int_bit` from `wexclJ`. Hoist the `wexclJ` shift out of the i-loop: in
  the variant the compiler re-did it per body.
* Keep the `PRUNE_NBL` variants correct. They clear bits of `imask` through `mask_ji` and write it back:
  either leave the prune path as it is (`#ifdef`) or shift consistently.
* Check: `seg.py` shows the i-slot test as `LOP3.LUT P, R, imm` + `BRA` (2 instructions, was 4); static
  1824 → ~1792. Gate `--strict`.

**H1c, force switch pre-combined per atom-type pair (−6.2%, FP order ~1e-7).**
* For the LJ force-switch flavours, upload a second table with
  `float4 {c6, c12, K2, K3}` per type pair, where `K2 = c12·repulsion_shift.c2 − c6·dispersion_shift.c2` and
  `K3 = c12·repulsion_shift.c3 − c6·dispersion_shift.c3`. Compute them in double on the host and round to
  float.
* Keep in mind that the GPU table holds 6·C6 and 12·C12, exactly what `ljForceSwitch` multiplies today, so
  use the same values.
* In the kernel, load it with one `LDG.128` (or a float4 texture fetch if textures are enabled; this build
  uses the `LDG` path) and compute the switch as `F_invr += (K3*s + K2) * (s*s*inv_r)` with
  `s = fmaxf(r2*inv_r − rvdw_switch, 0)`.
* Places:
  * table upload: `nbnxm_gpu_data_mgmt.cpp` (~line 491, `initParamLookupTable` for `nbfp`);
  * the struct: `NBParamGpu` (`nbnxm/gpu_types_common.h:222`);
  * `fetch_nbfp_c6_c12` (`cuda/nbnxm_cuda_kernel_utils.cuh:354`);
  * `calculate_force_switch_F` (same file, ~line 102).
* Keep the old float2 table for other flavours and kernels (prune, LJ-PME, FEP, SYCL/HIP). The energy (VF)
  kernel may keep computing the energy switch from c6/c12.
* MAS1 has 54 LJ types, so the extra table is ~48 KB.
* Check: static 1824 → ~1728, body 54 → 48.

**H1b, Ewald correction refitted (−3.1% with (5,5); more accurate than today).**
* `pmeCorrF(z²)` (`nbnxm/nbnxm_kernel_utils.h:219`) is a P6/Q4 fit valid to z² = 16. It costs 15
  instructions per pair: an `FMUL` for β²r², 2 `MOV`s, 10 `FFMA`, `RCP` and `FMUL`.
* Replace it **in the CUDA NB kernel only** by a monic rational in r² whose coefficients are scaled on the
  host: `a_k·β^{2k}`, divided by the leading coefficient. The quotient of the leading coefficients times β³
  becomes the existing β³ multiply. Pass the coefficients in `NBParamGpu` (constant bank; FFMA takes one
  constant operand, so the Horner steps need no `MOV`):

  ```
  n = r2 + N[m-1]; n = n*r2 + N[m-2]; … ;  d = r2 + D[n-1]; …;  corr = n * (1/d) * scale
  ```

* Coefficients: use `bench/profiling/pmecorr_weighted.py` (weighted minimax: error × max(1, z³), i.e.
  relative to the 1/r³ term). Store the t-basis (t = z²) coefficients as constants in the source; scale them
  on the host. Measured (fp32-emulated, weighted metric; production 8.8e-7):

  | form | fit range | instructions | weighted error, rtol 1e-5…1e-8 |
  |---|---|---|---|
  | **(5,5), recommended** | [0, 16] | 12 | 5.9-6.2e-7, better than production everywhere, no fallback needed |
  | (6,4) | [0, 16] | 12 | 8.4e-7 |
  | (5,4) | [0, 12] | 11 | 7.4e-7 for rtol ≥ 1e-6, but 1e-4…5e-4 for rtol ≤ 1e-7 |

  (5,4) needs a fallback, i.e. a kernel flavour switch; take it only if you can make the fallback clean.
  **Do not use (4,4) or (3,4)**: they look better under uniform weighting but are 1.3-5x worse relative to
  1/r³.
* Re-check the final coefficients in the kernel's exact r² form for β/r_c of MAS1 (`pmecorr_weighted.py`
  does this) and for rtol 1e-6, and put the numbers in the commit message.
* Set the coefficients wherever `ewald_beta` is set: `initNbparam` (`nbnxm_gpu_data_mgmt.cpp:457`, ~line
  217) **and** `gpu_pme_loadbal_update_param` (~line 690), because PME tuning changes β and r_c (βr_c stays
  fixed by `ewald-rtol`).
* Leave `pmeCorrF` itself for the CUDA FEP kernel (`cuda/nbfe_cuda_kernel.cuh:623`) and the SYCL/HIP kernels.
* The VF kernel's energy (`erff`) is unaffected.
* Check: body −3 instructions (−4 with (5,4)).

**Per-commit check for H1.** Optional mechanism check: nsys NB kernel p50 per step. Expected NB kernel
time ≈ −0.9 × the instruction reduction (the 2026-10-07 replay found 0.73-1.17).

### Phase 3, H2: PME spread with shared-memory accumulation (M-L, expected +6-8% spatial / +3-4% otherwise)

Evidence (`HYPOTHESES.md` H2, `ANALYSIS.md` 3 and 6):
* Spread does 11.87 M scattered fp32 `RED`s per step; ncu shows L2 at 87% of peak, IPC 0.23.
* Skipping it saves 81 µs/step (12.3%).
* Model on MAS1 coordinates: block-local accumulation cuts global updates 2.3x in today's atom order and
  3.9x in a spatial order with 64-atom blocks (sectors 5x).

Recommended design (self-contained in `ewald/`, single PP rank with PME on the same GPU):
1. **Spatial atom order without a new sort.** The nbnxm GPU data already holds `atomIndices` (device
   `nb->atomIndices`, `cuda/nbnxm_cuda_types.h:81`): the local atom index for every nbnxm grid slot, −1 for
   fillers, uploaded at each search step and used by `x_to_nbat`
   (`cuda/nbnxm_gpu_buffer_ops_internal.cu:101`).
   * Let the spread kernel walk the slots in that order: block b takes slots [64b, 64b+64) and reads
     `x[atomIndices[slot]]` and the charge, skipping −1. Coordinates come from the PME/state buffer in
     local order, which is ready at step start, so there is no dependency on x→nbat.
   * Make the PME stream wait for the search-step upload of `atomIndices` (an event).
   * Atom drift between searches is ~0.07 nm over 200 steps, small against a 0.11 nm grid spacing.
2. **Per-block tile.** Each block reduces the min/max stencil origin of its atoms, giving a bounding box of
   ~11³ grid points (with 3-point halo), ~5-11 KB.
   * If the box fits a fixed shared-memory budget, accumulate there, then flush every non-zero point with one
     global `RED.F32` (coalesced along z).
   * Otherwise (rare: periodic-boundary blocks, filler-heavy blocks) fall back to today's direct atomics.
     Handle the periodic wrap when mapping tile points to grid indices.
3. **Accumulate in int32 fixed point.** On sm_89, shared `atomicAdd(float*)` is a CAS spin loop and
   `atomicAdd(int*)` is a native `ATOMS.ADD`.
   * Use a per-block scale from the block's Σ|q| so that no partial sum can overflow: a point receives at
     most Σ|q| because the stencil weights are ≤ 1. Take the largest power of two that keeps
     Σ|q|·scale < 2³⁰.
   * Convert to float when flushing. The sums become order-independent (deterministic).
   * Verify that the grid's relative RMS error vs today's fp32 grid stays at the atomic-order noise level
     (4.1e-8 measured on 2026-10-07; target ≤ ~1e-7).
4. **Gather unchanged at first.** Production instantiates spread with `writeGlobal = false`, so gather
   recomputes its splines and does not depend on spread's atom order. A tile-staged gather in the same order
   (H8) is a later, separate commit.
5. **Keep the old kernel** for every unsupported case: PME on a separate rank, DD, `-nb cpu`, two grids
   (FEP), non-CUDA back-ends.

Process:
* Prototype first in the throwaway worktree. Time the kernel in isolation: a small in-process replay as
  in `bench/profiling/nb_replay.sh` / `experiments/exp-gpukern.patch` (`GMX_EXP_SPREAD_REPLAY`; the patch
  applies to `3df7585464`, so port only the replay idea), or nsys kernel time with the production command.
* Compare the grid with the production kernel's.
* Go/no-go: spread kernel time −40% or better in isolation and an end-to-end gain with the CI excluding 1.
* Time-box the prototype to ~1 day. If the spatial variant does not pay off, record the numbers and try the
  current-order variant (shared-memory hash, 2.3x fewer updates) only if the data suggest it can win
  despite its ~32 KB shared memory per block.

### Phase 4, H5: faster CPU pair search (M, expected +2-3%, bit-identical pair list)

* Where:
  * `src/gromacs/nbnxm/pairlist.cpp`: `makeClusterListSupersub` (~lines 820-960). It calls
    `clusterBoundingBoxDistance2_xxxx_simd4` (`boundingbox_simd.h`), then a scalar loop
    (lines 879-915) tests `d2l[ci] < rlist2` and ORs `imask` bits one at a time.
  * `setExclusionsForIEntry` (~1590-1660) and the bisection at ~1030-1060.
* Steps:
  1. Replace the scalar loop with SIMD compares and a mask extraction: two SIMD4 compares + movemask, or
     one 8-wide compare. Use the GROMACS SIMD layer, not raw intrinsics, so other architectures keep
     working. Derive `ci_last` (`PRUNE_LIST_CPU_ONE`) with a bit scan.
  2. Optionally widen the packed bounding-box distance to 8/16 lanes on AVX-512 (changes the
     packed-bounding-box layout in `Grid`; larger change).
  3. Replace the exclusion bisection with a direct j-cluster → list-index map per i-entry.
* Check:
  * the generated pair list must be identical. Compare `sci`, `cjPacked` (`cj`, `imei.imask`, `excl_ind`)
    and `excl` arrays before/after on a few search steps in a throwaway debug dump, or with the nbnxm unit
    tests (`gmxbench upstream`);
  * NVTX "NS search local" per search step (the NVTX build + `nsys_steps.py`) should drop from 9.46 ms;
    perf annotate with a `-g` build as in `ANALYSIS.md` 8 (profile `raw/suite-dbg.toml` style).
* Gate `--strict`.

### Phase 5 (optional, only if time remains)

* **H3b co-scheduling** (after H2): if the spread phase is still long, launch x→nbat and the NB kernel
  before spread and cap spread's residency (persistent grid-stride spread with fewer blocks).
  * Validate with `bench/profiling/gpumetrics_phase.py`: SM issue should be > 50% from ~10 µs into the step.
    This needs the GPU-metrics capture, i.e. a DCGM pause: **ask first**.
  * Day 1's "NB stream high priority" was −0.57%; PME must keep its priority.
* **H7 update chain**: the rolling prune (1,402 blocks, NB stream) is dispatched ahead of LINCS (150
  blocks), and LINCS doubles. Options:
  * make the prune depend on an event recorded after SETTLE, in its own stream joined before the next NB
    kernel;
  * smaller LINCS blocks;
  * fuse reduce + leap-frog.
  Measure each with `prun`.
* **A6** (`-ntomp 20`): extend the sweep or change `[configs.perf-mas1-production]`. This changes `bench/`,
  so run `bench/validation/run_validation.sh quick` afterwards.
* **H4** (lagged, double-buffered pair search, bound 8.4%) is too large for this hand-off. If you reach it,
  write a design note under `bench/reports/` instead of code.

## 5. Interleaved measurements outside gmxbench (optional, faster loop)

`bench/profiling/prun SPEC.toml --out DIR --repeats N` runs builds × args × env interleaved with NVML
telemetry; `summarize.py exp DIR --ref VARIANT` gives ratios with 95% CIs. Copy
`bench/profiling/experiments/gpukern-e2e3.toml` as a template:
* `[builds]` are gmxbench build dirs (`~/.cache/gmxbench/builds/<hash>`; `gmxbench build` prints the path);
* the tpr is `bench/results/profiling-2026-10-06/raw/inputs/prod/topol.tpr`;
* use 25,000 steps with `resetstep` 5,000.

## 6. Per-item gate (each commit, against its parent)

```bash
# inner loop (~5 min)
bench/bin/gmxbench all -A <parent> -B WORKTREE --tier smoke --target-only
bench/bin/gmxbench perf -A <parent> -B WORKTREE --target-only --configs perf-mas1-production --repeats 9
# before committing (~1 h)
bench/bin/gmxbench all -A <parent> -B WORKTREE --tier quick --strict
```

`<parent>` is the previous kept commit, or `dev` for the first one.

* All items here change only CUDA kernels or CPU code that must give bit-identical results. The CPU
  `-reprod` configurations must therefore stay IDENTICAL even for the FP-order changes (H1b, H1c, H2),
  which is why `--strict` is used throughout.
* GPU configurations must be EQUIVALENT at the level of their own noise (≈1e-7 relative force RMS).
* Exit status 0, no DIFFERENT/FAIL, MAS1 `perf-mas1-production` faster with the CI excluding 1, nothing
  slower (consider `--fail-on-slowdown`; re-run a suspicious CPU-only slowdown with `--cases … --configs …
  --repeats 15` before believing it).
* Energy per simulated ns must not get worse.
* Keep / drop rule: keep only if faster. If a change is neutral, drop it unless it is a prerequisite of a
  later item.

## 7. Before-merge checks (whole branch vs `dev`)

```bash
bench/bin/gmxbench all -A dev -B feature/perf-iter3 --tier full --strict   # hours
bench/bin/gmxbench upstream --build feature/perf-iter3                     # GROMACS unit + regression tests
bench/bin/gmxbench sweep --build feature/perf-iter3 --target-only          # best MAS1 run arguments
```

* All must pass. After `upstream`, rebuild from a fresh configure before any further perf run (the
  `_FORCE_INLINES` gotcha).
* If the sweep finds a different best configuration, update `[configs.perf-mas1-production]` in
  `bench/suites/target.toml` (that is a `bench/` change: run `bench/validation/run_validation.sh quick`).
* **Energy conservation** (needed because H1b, H1c and H2 change numerics): repeat the NVE check of
  `../phaseB-energy-2026-10-07/REPORT.md`. Use 2 ns NVE of MAS1, order dev / branch / branch / dev, and
  compare drift per atom, ⟨T⟩, ⟨Epot⟩ and ⟨P⟩. The branch must agree with `dev` within run-to-run scatter.
  Keep the inputs and outputs under `bench/results/` (customer-derived).

## 8. Reporting (when the branch is faster than `dev`)

1. `bench/reports/2026-10-perf-iter3.md`, written like `2026-10-gpu-kernel-opt.md`:
   * per-commit table (commit, change, MAS1 speedup with CI, numerics);
   * branch vs `dev` on all target configurations and other systems from the full run;
   * GPU energy per ns;
   * the NVE check;
   * limits and anything slower;
   * the session names of the gmxbench results.
2. `bench/OPTIMIZATION.md`:
   * section 1 (State: new ns/day, commit table);
   * section 3 (re-profile briefly with nsys light and update the step anatomy);
   * section 4 (mark done items, re-rank the rest);
   * section 5 (every rejected attempt with its number);
   * update "Last updated".
3. Append "Status YYYY-MM-DD" to this directory's `HYPOTHESES.md`: per hypothesis, done / rejected /
   open, with numbers.
4. `bench/reports/README.md`: index the new report.
5. Commit the docs in a separate commit on the feature branch.
6. Hand over to the user: the branch name, a summary, the merge command for `dev`
   (`git -C /home/asokolov/Projects/gromacs merge --ff-only feature/perf-iter3` from a `dev` checkout), and
   the push command. Do not merge or push yourself.

If nothing ends up faster, still do steps 2-4 for the negative results and report that to the user.

## 9. Expected outcome (for orientation, not a target to force)

| item | expected end to end | confidence |
|---|---|---|
| H3a | +1.5-3% | medium (scheduling effects interact) |
| H1a / H1b / H1c | ~+1.8% / +1.4% / +2.8% | high: instruction counts are measured and the kernel is issue-bound |
| H2 | +3-8% | medium: L2-bound kernel, update counts modelled; fixed-point details and the fallback rate decide it |
| H5 | +2-3% | medium |

A plausible total is +10-15% over `dev` (≈ 275-285 ns/day). Report what you measure, not these numbers.
