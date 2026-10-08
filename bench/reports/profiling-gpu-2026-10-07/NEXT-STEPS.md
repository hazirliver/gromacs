# Next steps: turn the measured GPU-kernel gains into real source changes (hand-off plan)

Written 2026-10-07 at the end of the GPU-kernel profiling session, for an agent starting with no other
context. Read this file fully first, then `EXEC-SUMMARY.md` and `OPPORTUNITIES.md` in this directory.
Repository: `/home/asokolov/Projects/gromacs` (a GROMACS 2026.4 fork). Project rules are in `CLAUDE.md`
at the repository root; they override anything here.

## 1. Where things stand

Two profiling sessions measured the single-GPU production run of the customer system MAS1 (185,486
atoms, CHARMM36, PME, LJ force-switch, h-bond LINCS + SETTLE, everything on the GPU; NVIDIA L40S,
Xeon 6338 VM). Production command:
`gmx mdrun -ntmpi 1 -ntomp 16 -nb gpu -pme gpu -bonded gpu -update gpu -nstlist 200` -> 0.743 ms/step =
232.6 ns/day on a production-like build.

Several small code changes were prototyped and measured. **None of them is in `src/` of any real
branch.** They exist only in a throwaway worktree `/home/asokolov/Projects/gromacs-exp-gpukern`
(detached at `dev`, diff saved as `bench/profiling/experiments/exp-gpukern.patch`, day-1 diff as
`bench/profiling/experiments/exp-throwaway.patch`), each behind a `GMX_EXP_*` runtime switch and mixed
with experiment-only code (kernel replay harnesses, kernel-skip flags, 14 extra NB kernel variants,
spread-order experiments). That build must not be used for production and must not be merged.

Measured gains (5 seeded interleaved rounds, end to end, vs the production-like build, 95% CI):

| # | change | end to end | where measured |
|---|---|---|---|
| 1 | NB kernel: factored LJ force switch + Ewald-correction polynomials in Horner form | **+2.57% [2.11, 3.03]**, -3.6% GPU energy/ns | `raw/exp/gpukern-e2e` (variant nb3-fsw-ewh) |
| 2 | SETTLE with coalesced shared-memory staging | +0.44% [0.08, 0.79] alone, ~+1.2% inside the stack | `raw/exp/gpukern-e2e2`, `-e2e3` |
| 3 | bonded GPU kernel in its own stream (day 1) | +0.48% [0.10, 0.86] today, +0.91% day 1 | `raw/exp/gpukern-e2e2`, day 1 `raw/exp/exp-combo` |
| 4 | SETTLE concurrent with LINCS (own stream) | no change alone (+0.17%), ~+0.4-0.6% inside the stack | `raw/exp/gpukern-e2e2`, `-e2e3` |
| 5 | OpenMP for nvcc-compiled host code (`-Xcompiler=-fopenmp`; a `#pragma omp` loop in `listed_forces_gpu_impl_gpu.cpp` runs serially today) | +0.85% (day 1) | day 1 `raw/exp/exp-nbminblocks` |
| 6 | `-ntomp 20` instead of 16 (run argument) | +0.5-0.6% | both days |
| all | 1-6 together | **+7.19% [6.58, 7.80] = 249.4 ns/day, -5.7% GPU energy/ns** | `raw/exp/gpukern-e2e3` (XO-all+staged-t20) |

That exact stacked configuration passed `gmxbench quality --strict --tier quick` on all 63 cases
(`raw/gmxbench/quality-XO-final`: 72/72 bitwise-reproducible configs IDENTICAL, GPU configs EQUIVALENT
at 0.5-1.7x their own noise, no DIFFERENT/FAIL). Caveat: in that test the NB change (#1) only replaced
the one kernel flavour MAS1 uses; as a source change it reaches every GPU kernel that uses the shared
helpers.

## 2. Goal of the next work

Phase A (main goal): re-implement changes 1-5 cleanly on a new feature branch from `dev`, without any
`GMX_EXP_*` switches or experiment code, validate them as `CLAUDE.md` / `bench/TESTING.md` require, and
get the branch to a state that can be merged into `dev`. Phase B: production-readiness checks for the
long MAS1 run. Phase C (larger, optional): the PME spread rewrite.

## 3. Rules and environment (read before doing anything)

* Follow `CLAUDE.md`: work happens on `dev`; feature branches from `dev`, merged back into `dev`; every
  `src/` change is tested with gmxbench following `bench/TESTING.md`.
* `data/` holds customer input (the MAS1 system). **Never commit it.** The same holds for every tpr
  derived from it, e.g. `bench/results/profiling-2026-10-06/raw/inputs/prod/topol.tpr` (the `bench/results/`
  directory is git-ignored; keep it that way).
* Do not run production workloads; benchmark and test runs on this node are fine. Check the node is
  quiet first (`nvidia-smi`, `uptime`); no other GPU jobs.
* GPU state to keep: power limit 325 W (it cannot be raised; do not change it), application clocks
  unlocked. If you lock clocks (`sudo nvidia-smi -lgc`), always reset with `sudo nvidia-smi -rgc` and
  verify. sudo is available on the node; use it only for that.
* DCGM holds the GPU performance counters: Nsight Compute and CUPTI PC/PM sampling fail. Use nsys
  (CUDA trace), and `bench/profiling/sassprof/` (CUPTI SASS metrics, works) if you need instruction
  counts. Do not pause DCGM (not authorised).
* Git identity is not configured globally; earlier sessions committed with
  `git -c user.name=asokolov -c user.email=sokolov.ars.a@nebius.com commit ...`.
* Pitfall seen twice: `pkill -f PATTERN` from a shell whose own command line contains PATTERN kills that
  shell. Kill by PID.
* The profiling tools (`bench/profiling/`: `prun`, `summarize.py`, `nsys_capture.sh`, `nsys_steps.py`,
  `sassprof/`, `nb_replay.sh`) live only on branch `feature/profiling`, which the main checkout has
  checked out. `bench/gmxbench` itself is the same as on `dev`.

Suggested layout: keep the main checkout on `feature/profiling` (tools), and create the work tree with
`git worktree add -b feature/gpu-kernel-opt /home/asokolov/Projects/gromacs-gpukern-port dev`. Run
gmxbench from the main checkout with `-A dev -B src:/home/asokolov/Projects/gromacs-gpukern-port`
(spec types: git ref, `WORKTREE`, `src:DIR`, `path:DIR`), or from inside the new worktree with
`-B WORKTREE` as in `bench/TESTING.md`.

## 4. Phase A: the changes, one commit each

Order: 1, 2, 5, 3, 4 (most valuable and simplest first). After each commit run the quick validation
(section 5). Reference implementations are in `bench/profiling/experiments/exp-gpukern.patch` (line
numbers below refer to that file); take only the parts named here.

### A1. NB pair arithmetic (`src/gromacs/nbnxm/nbnxm_kernel_utils.h`)

* `ljForceSwitch` (force part, both `calcFr` branches): compute `rSwitch = max(r - rVdwSwitch, 0)`
  once, then `*f += (c12 * (repuShiftV2 + repuShiftV3 * rSwitch) - c6 * (dispShiftV2 + dispShiftV3 *
  rSwitch)) * (rSwitch * rSwitch) * (calcFr ? r : rInv);`. The source currently multiplies term by
  term and nvcc does not re-associate even with `-use_fast_math` (16 SASS per pair; the factored form
  removed 3.8 instructions per pair and 2.7% of the kernel). The prototype used `fmaxf(r2 * inv_r -
  rsw, 0.0F)` instead of `gmxGpuFDim` (one instruction fewer); check `gmxGpuFDim` semantics
  (`gpu_utils/gpu_kernel_utils.h:100`) and that the helper still compiles for SYCL. Decide whether to
  factor the energy part too (it only runs on energy steps; not measured).
* `pmeCorrF`: evaluate both polynomials in Horner form with the same coefficients (prototype:
  `pmeCorrFHorner` in the patch, hunk at line 1404 ff.). Accuracy against the exact function is the
  same as the current Estrin form (RMS error 1.010e-7 vs 1.015e-7, max 7.96e-7 vs 8.30e-7 over the full
  cut-off range; `raw/pmecorrf-accuracy.txt`). Mention in the commit message that the FP order changes
  (forces differ 2.6e-7 relative RMS from the old kernel; the noise floor is ~5e-8 per kernel launch).
* Scope: these helpers are used by the CUDA NB kernels (all flavours with force switch / analytical
  Ewald), the CUDA free-energy kernel (`cuda/nbfe_cuda_kernel.cuh:623`, `pmeCorrF`) and the SYCL kernel.
  The HIP kernel has its own copies in `hip/nbnxm_hip_kernel_body.h` (optional to change; cannot be
  tested here).
* Check: with `-DCMAKE_CUDA_ARCHITECTURES=89`, `cuobjdump -sass` of
  `nbnxn_kernel_ElecEw_VdwLJFsw_F_cuda` should shrink from 1928 to about 1824 static instructions
  (prototype `exp3`), registers stay 61, no spills (`cuobjdump -res-usage`).
* Expected: about +2.5% on `perf-mas1-production` (prototype +2.57% [2.11, 3.03]).
* Do NOT include: the LJ-table hoist (variant 4, slower), the `NBEXP_MINBLOCKS` launch-bound variants
  (slower), anything from `nbnxm_cuda_kernel_exp.cu` or the replay code in `nbnxm_cuda.cu`.

### A2. SETTLE coalesced staging (`src/gromacs/mdlib/settle_gpu_internal.cu`)

* Prototype: `settleKernelStaged` in the patch (hunk at line 558 ff.). Each block checks with
  `__syncthreads_and` that its waters are consecutive atoms (O,H,H,O,H,H,...); if so, x, x' and v of
  the block are loaded/stored with coalesced loops through shared memory (3 x 9 floats x 256 threads =
  27.6 KB static shared memory), else the old per-thread gathers. Make this the only kernel (fold the
  fallback into it) instead of keeping two kernels and an env switch.
* It moved 5.0x fewer global sectors (0.37 M vs 1.84 M per launch; `raw/sass-variants/settle-staged`).
  Same arithmetic per water from identical inputs; expect bitwise-identical results (not separately
  verified; check with a `-reprod` style comparison or the quality suite's GPU-resident configs).
* Expected: +0.4% alone, more inside the full stack.

### A3. OpenMP for host code compiled by nvcc (CMake)

* Problem (day 1): `src/gromacs/listed_forces/listed_forces_gpu_impl_gpu.cpp` (and
  `mdlib/lincs_gpu.cpp`) are compiled by nvcc as CUDA sources, and the CUDA flags lack the host
  compiler's OpenMP flag, so their `#pragma omp parallel for` loops run on one thread (2 ms of GPU idle
  per pair-search step).
* The prototype only passed `-DCMAKE_CUDA_FLAGS=-Xcompiler=-fopenmp` (profile `cuda-nosub-gpukern-omp`
  in `bench/profiling/experiments/gpukern-suite.toml`). Do it properly in CMake (look at
  `cmake/gmxManageNvccConfig.cmake` / where CUDA host flags are assembled): add the host OpenMP flags
  (`OpenMP_CXX_FLAGS`) via `-Xcompiler` when GMX_OPENMP is on; keep clang-as-host and non-OpenMP builds
  working.
* Check: `nm`/`objdump` of the object shows GOMP calls; CPU-only reproducible configs stay IDENTICAL.
* Expected: +0.85% (day 1).

### A4. Bonded GPU kernel in its own stream

* Prototype: day-1 patch (`exp-throwaway.patch`, files `listed_forces/listed_forces_gpu.h`,
  `listed_forces_gpu_impl.h`, `listed_forces_gpu_impl_gpu.cpp`, `listed_forces_gpu_internal.cu`,
  `mdlib/sim_util.cpp`), runtime-switched version in `exp-gpukern.patch` (lines 410-557, 933-948). The
  kernel waits on an event marked in the NB local stream (after x->nbat etc.), runs in its own
  Normal-priority stream, and the NB local stream waits for it only after the NB kernel was launched
  (`enqueueWaitForKernel()` called in `do_force` after the local `do_nb_verlet`). The prototype skipped
  this with PP domain decomposition; keep that or make it correct for DD.
* Must check: CUDA graphs (`GMX_CUDA_GRAPH=1`) still capture correctly (the extra stream joins via
  events) on a system without CPU forces; SYCL/HIP paths compile; energy steps (bonded energies) still
  correct.
* Expected: +0.5-0.9%.

### A5. SETTLE concurrent with LINCS (`src/gromacs/mdlib/update_constrain_gpu_impl.{h,cpp}`)

* Prototype in the patch (lines 949-1021): a second High-priority `DeviceStream` for SETTLE, fork event
  after leap-frog, join event before `xUpdatedOnDeviceEvent_`. **Bug found in the prototype**: the
  stream was destroyed before `SettleGpu`, which uses it (segfault at teardown); own the stream so it
  outlives `settleGpu_` (declare it before, or reset `settleGpu_` first in the destructor).
* Alone it gives nothing measurable (LINCS needs a second wave when SETTLE occupies SMs: 14.3 -> 31.7
  us); it helped ~0.4-0.6% only together with A2. Keep it as the last, separate commit and drop it if
  it does not show a gain on top of A1-A4.

### A6. Run argument (not code)

`-ntomp 20` is ~+0.5% for MAS1. If it should become the reference production command, change
`[configs.perf-mas1-production]` in `bench/suites/target.toml` in a separate commit and, because that
changes `bench/`, run `bench/validation/run_validation.sh`. `gmxbench sweep --target-only` (section 5)
re-derives the best arguments anyway.

## 5. Validation (per `bench/TESTING.md`)

Per commit (inner loop; ~5 min for the first, longer when it rebuilds dev):

```
bench/bin/gmxbench all -A dev -B src:/home/asokolov/Projects/gromacs-gpukern-port --target-only --tier smoke
```

Before considering the branch done (about one hour):

```
bench/bin/gmxbench all -A dev -B src:/home/asokolov/Projects/gromacs-gpukern-port --tier quick --strict   # A2-A5 must keep CPU configs IDENTICAL
bench/bin/gmxbench all -A dev -B src:/home/asokolov/Projects/gromacs-gpukern-port --tier quick            # A1 changes FP order of GPU kernels
```

Acceptance: exit status 0; no DIFFERENT/FAIL; all bitwise-reproducible (CPU `-reprod`) configs IDENTICAL;
GPU configs EQUIVALENT at the level of their own noise; MAS1 target cases `faster` with the CI excluding
1; no case `slower` (consider `--fail-on-slowdown`). Expected total on `perf-mas1-production` (16
threads): roughly +4.5-5.5% for A1-A5 (prototype stack at 16 threads without the SETTLE staging: +4.84%).

Before merging into `dev` (hours):

```
bench/bin/gmxbench all -A dev -B <branch> --tier full --strict
bench/bin/gmxbench upstream --build <branch>        # GROMACS unit + regression tests
bench/bin/gmxbench sweep --build <branch> --target-only
```

Optional mechanism checks (from the main checkout, branch `feature/profiling`): nsys light captures
with `bench/profiling/nsys_capture.sh` + `nsys_steps.py` (expect the NB kernel p50 ~430 instead of ~450
us, bonded kernel in its own stream); the end-to-end interleaved A/B tool `bench/profiling/prun` with a
spec like `bench/profiling/experiments/gpukern-e2e3.toml` (builds given as `path:` gmxbench build dirs).
A/A noise on this node: minimum detectable effect ~0.74% with 5 repeats; use `--repeats 9` for the small
items (A2, A5).

Commit messages: name the gmxbench session, quality counts and the MAS1 speedup with CI (as
`bench/TESTING.md` asks); say which commits change FP order (A1) and which are bit-preserving.

## 6. Phase B: before the long MAS1 production run

* Energy conservation check of the new branch vs `dev` on MAS1: several-ns runs of each (NVE or the
  production ensemble), compare conserved-energy drift per ns and per atom and basic observables; A1
  changes FP order, so look for systematic differences, not bitwise ones. This is a benchmark run, not a
  production run; keep inputs under `bench/results/`.
* Independent of this work: the MAS1 input has interlocked aromatic rings (Gαi PHE147 / Gβ1 TRP104 in
  `npt2.gro`) that must be fixed before any long run (see the memory notes / day-1 report). Do not use
  4 fs or HMR with the current input.

## 7. Phase C (optional, larger): PME spread rewrite

* Why: skipping the spread kernel (timing only) saves 81 us/step = 10.9% of the step; 91% of its time is
  on the critical path. 92% of the kernel is 11.9 M scattered global grid updates per step; making them
  non-atomic saves only 18%, spatial atom order only 2-5% (`ANALYSIS.md` section 3).
* Idea: process atoms in spatial (nbnxm grid) order, accumulate each block's sub-grid in shared memory,
  then one global atomic per touched grid point (handle periodic wrap and tile size limits). Upstream
  has the same idea as an open HIP-only MR, !6000 (`raw/prior-art-gpu.md`); no CUDA port.
* Estimate 3-6% end to end (assumption: 2-3x fewer global updates halve the kernel). Effort: days.
* Tooling: the throwaway worktree has an in-process spread replay (`GMX_EXP_SPREAD_REPLAY`, run with
  `bench/profiling/nb_replay.sh` and `SPREAD_REPLAY=...`) that times spread variants on live data and
  compares the grids; reuse that approach for fast iteration before end-to-end runs.

## 8. Key artifacts

* Reports: `bench/results/profiling-gpu-2026-10-07/{EXEC-SUMMARY,OPPORTUNITIES,ANALYSIS,NOTES,PLAN}.md`,
  day 1: `bench/results/profiling-2026-10-06/`.
* Prototype diffs: `bench/profiling/experiments/exp-gpukern.patch` (today),
  `bench/profiling/experiments/exp-throwaway.patch` (day 1).
* Builds (gmxbench cache, may be overwritten by rebuilds): production-like P
  `~/.cache/gmxbench/builds/3bf7713a0dd56c41`; throwaway stack XO `~/.cache/gmxbench/builds/1093125cdb89aee7`
  (only meaningful with `GMX_EXP_NB_VARIANT=13 GMX_EXP_CONCURRENT_SETTLE=1 GMX_EXP_BONDED_STREAM=1
  GMX_EXP_SETTLE_STAGED=1`).
* MAS1 test input (customer-derived, never commit): `bench/results/profiling-2026-10-06/raw/inputs/prod/topol.tpr`;
  gmxbench reads the MAS1 system from `data/` itself (`bench/suites/target.toml`).
* When phase A is merged, the throwaway worktrees can be removed:
  `git worktree remove /home/asokolov/Projects/gromacs-exp-gpukern` and
  `git worktree remove /home/asokolov/Projects/gromacs-exp-nbminblocks`.

## 9. Status 2026-10-08 (Phase A done and merged into dev by fast-forward, dev = da0eed4161, not pushed; Phase B done)

Branch `feature/gpu-kernel-opt` (worktree `/home/asokolov/Projects/gromacs-gpukern-port`), five commits on
`dev`, each validated with `gmxbench all --tier quick --strict` against its parent (sessions named in the
commit messages); MAS1 perf-mas1-production gain per commit:

| commit | change | MAS1 production | notes |
|---|---|---|---|
| 2e3e86d3a1 | A1 NB force switch factored + Horner pmeCorrF | +2.90% [2.28, 3.53] | SASS 1920 -> 1811 (same opcode mix as prototype exp3); FP order of GPU NB kernels changes |
| 53f84025dd | A2 SETTLE coalesced staging | +0.90% [0.54, 1.25] | x'/v bitwise identical to the old kernel (harness, 72 input variants) |
| e1971ae650 | A3 OpenMP host flags for nvcc (CMake < 3.31) | +1.33% [0.89, 1.78] | MAS1 gpu-resident (more searches) +2.0% |
| e2546053cf | A4 bonded kernel in own stream | +1.03% [0.48, 1.58] | DD-correct; graphs + DD checked |
| da0eed4161 | A5 SETTLE || LINCS | +0.61% [0.29, 0.92] | only when SETTLE/constraint atoms are disjoint |

Before-merge checks (all passed):
* branch vs dev, quick --strict (20261007-174958-all): MAS1 production 1.0657 [1.0599, 1.0716].
* branch vs dev, full --strict (20261007-202242-all, exit 0): 73/73 bitwise-reproducible configurations
  IDENTICAL, no DIFFERENT/FAIL; MAS1 production 1.0654 [1.0626, 1.0683] (233.0 -> 248.3 ns/day, GPU energy/ns
  -5.1%), MAS1 gpu-resident 1.0762, lysozyme gpu-resident 1.0642, water-1m 1.0245; nothing slower except
  water-5k + CUDA graphs 0.9930 [0.9891, 0.9970] ("negligible", latency-bound 0.05 ms/step; likely the SETTLE
  staging barriers in a 7-block kernel).
* `gmxbench upstream` (20261007-184945-upstream): 96/96 ctest passed (incl. 40 GPU tests).
* `gmxbench sweep --target-only` (20261008-013323-sweep): best is still `-ntomp 16 ... -nstlist 200`
  (249.5 ns/day) -> target.toml unchanged. The sweep does not try -ntomp 20 (A6 untested).
* Phase B (`../phaseB-energy-2026-10-07/REPORT.md`): NVE energy drift identical for dev and the branch. Side
  finding: the production settings heat MAS1 by ~27 K/ns in NVE; ~70% of it from lincs-iter = 1
  (lincs-iter = 2: drift / 3.3, -0.2% speed), ~20% from the pair-list buffer.

Gotchas: re-running CMake in an nvcc build dir (incl. `gmxbench upstream`, which reconfigures for
REGRESSIONTEST_PATH) drops -D_FORCE_INLINES -> rebuild from a fresh configure before perf A/B. Another Claude
session (mdprecheck work on MAS1) shared the node and made CPU-only perf cases noisy; it uses the throwaway build
`~/.cache/gmxbench/builds/1093125cdb89aee7`, so keep that build until it is done.

Open: remove the throwaway worktrees (keep build 1093125cdb89aee7 while the mdprecheck session uses it); A6 (-ntomp 20); Phase C (spread).
