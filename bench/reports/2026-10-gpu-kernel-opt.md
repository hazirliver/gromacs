# GPU kernel speedups for GROMACS (October 2026)

Five changes to the CUDA code path of this fork (GROMACS 2026.4 base, branch `dev`, commits
`2e3e86d3a1`..`da0eed4161`) make the MAS1 production run **6.5% faster** (233.0 -> 248.3 ns/day, 95% CI
+6.3 to +6.8%) and cut GPU energy per simulated nanosecond by **5.1%**. Other GPU-resident simulations gain
up to 6.4%. Every configuration that is bitwise reproducible gives bitwise identical results, GPU runs stay
within their own run-to-run noise, and the GROMACS test suite passes (96 of 96).

Measured 7-8 October 2026 on one NVIDIA L40S with gmxbench (`bench/`, see `TESTING.md`).

| Headline | Before | After | Change |
|---|---:|---:|---|
| MAS1 production throughput | 233.0 ns/day | 248.3 ns/day | **+6.5%** [+6.3, +6.8] |
| MAS1 GPU energy per simulated ns | 112.9 kJ | 107.1 kJ | **-5.1%** |
| Bitwise-reproducible configurations identical | | 73 of 73 | all CPU `-reprod` runs, incl. domain decomposition |
| GROMACS ctest (unit, regression, MPI, GPU) | | 96 of 96 pass | |

## MAS1

185,486 atoms: a membrane protein complex, CHARMM36 with LJ force switch, PME, 2 fs time step with h-bond
constraints. Production command (unchanged; a fresh `gmxbench sweep` on the new code confirms it is still the
fastest):

```
gmx mdrun -ntmpi 1 -ntomp 16 -nb gpu -pme gpu -bonded gpu -update gpu -nstlist 200
```

| Configuration | Before (ns/day) | After (ns/day) | Change, 95% CI | GPU energy per ns | Verdict |
|---|---:|---:|---:|---:|---|
| Production: all on GPU, 16 threads, nstlist 200 | 233.0 | 248.3 | +6.5% [+6.3, +6.8] | -5.1% | faster |
| All on GPU, 8 threads, default nstlist | 204.4 | 220.0 | +7.6% [+7.2, +8.0] | -5.4% | faster |
| Nonbonded and PME on GPU, bondeds and update on CPU | 71.1 | 71.4 | +0.4% [-0.8, +1.7] | -1.9% | no change |
| Nonbonded on GPU only | 21.5 | 21.6 | +0.4% [-0.5, +1.2] | -0.9% | no change |
| CPU only, 5 ranks x 4 threads | 7.8 | 7.9 | +0.5% [-0.3, +1.4] | - | no change |

When bondeds and the update run on the CPU, the CPU sets the pace and the GPU savings do not show.

At 248 instead of 233 ns/day, one microsecond of MAS1 takes 4.0 instead of 4.3 GPU-days.

### Contribution of each change

Each change was measured against the commit before it, in the order applied (quick tier; 5 interleaved
repeats, 9 for the last one). The five gains multiply to +6.9%; measured directly against the original code,
all five together give +6.5%.

| Commit | Change | MAS1 production | MAS1 all on GPU, 8 threads |
|---|---|---:|---:|
| `2e3e86d3a1` | Cheaper pair arithmetic in the nonbonded kernel | +2.9% [+2.3, +3.5] | +2.2% [+1.9, +2.5] |
| `53f84025dd` | Coalesced memory access in SETTLE | +0.9% [+0.5, +1.3] | +1.1% [+0.5, +1.6] |
| `e1971ae650` | OpenMP for host code compiled by nvcc | +1.3% [+0.9, +1.8] | +2.0% [+1.2, +2.8] |
| `e2546053cf` | Bonded kernel in its own stream | +1.0% [+0.5, +1.6] | +0.8% [+0.5, +1.2] |
| `da0eed4161` | SETTLE concurrent with LINCS | +0.6% [+0.3, +0.9] | +1.1% [+0.9, +1.3] |
| all five | measured against the original code (full tier) | **+6.5% [+6.3, +6.8]** | **+7.6% [+7.2, +8.0]** |

## Other systems

The general benchmark suite: SPC/E water boxes from 5 thousand to 1.1 million atoms, lysozyme
(AMBER99SB-ILDN) in TIP3P water (also with hydrogen mass repartitioning at 4 fs and with virtual sites), a
TIP4P box and a free-energy water box. All five changes together vs the original code, full tier, 9
interleaved repeats each.

| System | Atoms | Configuration | Before (ns/day) | After (ns/day) | Change, 95% CI | Verdict |
|---|---:|---|---:|---:|---:|---|
| Lysozyme | 23,873 | all on GPU | 1301.6 | 1385.1 | +6.4% [+5.8, +7.0] | faster |
| Lysozyme, HMR 4 fs | 23,873 | all on GPU | 2249.6 | 2364.7 | +5.1% [+4.8, +5.5] | faster |
| Water | 331,776 | all on GPU | 195.0 | 200.3 | +2.7% [+2.4, +3.0] | faster |
| Water | 1,119,744 | all on GPU | 52.9 | 54.2 | +2.5% [+2.1, +2.8] | faster |
| Water | 41,472 | all on GPU | 1158.2 | 1169.6 | +1.0% [+0.6, +1.4] | negligible (< 1%) |
| Water | 5,184 | all on GPU | 2842.7 | 2851.2 | +0.3% [-0.6, +1.2] | no change |
| Water, CUDA graphs | 5,184 | all on GPU | 3284.3 | 3261.4 | -0.7% [-1.1, -0.3] | negligible (< 1%) |
| Water, free energy | 41,472 | all on GPU | 81.5 | 81.4 | -0.1% [-3.6, +3.4] | no change |
| Water | 331,776 | nonbonded + PME on GPU | 100.5 | 101.7 | +1.2% [+0.4, +1.9] | faster |
| Water | 41,472 | nonbonded + PME on GPU | 626.4 | 633.9 | +1.2% [+0.2, +2.2] | faster |
| Lysozyme | 23,873 | nonbonded + PME on GPU | 799.3 | 801.6 | +0.3% [-0.5, +1.1] | no change |
| TIP4P water | 23,328 | nonbonded + PME on GPU | 809.7 | 811.5 | +0.2% [-0.9, +1.3] | no change |
| Lysozyme, virtual sites 4 fs | 23,997 | nonbonded + PME on GPU | 1321.0 | 1320.5 | -0.0% [-0.8, +0.7] | no change |
| Lysozyme | 23,873 | nonbonded on GPU only | 191.2 | 192.2 | +0.5% [-1.0, +2.0] | no change |
| Water | 5,184 | CPU only | 554.1 | 554.3 | +0.0% [-0.9, +1.0] | no change |
| Water | 41,472 | CPU only | 101.9 | 102.6 | +0.7% [-0.5, +1.8] | no change |
| Water, reaction field | 41,472 | CPU only | 140.8 | 141.7 | +0.6% [-0.2, +1.5] | no change |
| Water, free energy | 41,472 | CPU only | 56.5 | 57.1 | +1.2% [-0.6, +2.9] | no change |
| TIP4P water | 23,328 | CPU only | 172.7 | 171.3 | -0.8% [-2.6, +1.1] | no change |
| Lysozyme | 23,873 | CPU only | 141.5 | 140.4 | -0.8% [-1.9, +0.3] | no change |

"All on GPU" is `-nb gpu -pme gpu -bonded gpu -update gpu` with 8 OpenMP threads; "nonbonded + PME on GPU"
keeps bondeds and the update on the CPU; CPU-only runs use 20 threads and do not execute the changed code.
GPU-resident runs gain most because the changed kernels are on their critical path.

## What changed

1. **Cheaper pair arithmetic in the nonbonded kernel** (`2e3e86d3a1`, `nbnxm/nbnxm_kernel_utils.h`). The LJ
   force switch is evaluated with its common factors pulled out (nvcc does not re-associate floating-point
   expressions), and the Ewald real-space correction polynomials in Horner instead of Estrin form, with the
   same coefficients and accuracy. The MAS1 kernel flavour (`ElecEw_VdwLJFsw_F`) shrinks from 1,920 to 1,811
   SASS instructions, 61 registers, no spills. Changes the floating-point order: forces differ by 1e-7 to
   3.4e-7 relative RMS. Helps every CUDA nonbonded kernel with analytical Ewald (and the CUDA free-energy
   kernel), most with force switch. The SYCL kernel shares these helpers; the HIP kernel has its own copies
   and is unchanged.
2. **Coalesced memory access in SETTLE** (`53f84025dd`, `mdlib/settle_gpu_internal.cu`). When a block's
   waters are consecutive atoms, x, x' and v of the block move through shared memory with coalesced loads and
   stores instead of 36-byte-stride gathers; other layouts use the old path. Bitwise identical results
   (checked against the old kernel on 72 input variants). Helps water-rich systems with GPU update.
3. **OpenMP for host code compiled by nvcc** (`e1971ae650`, `src/gromacs/CMakeLists.txt`). With CMake
   older than 3.31, files with LANGUAGE CUDA did not get the host compiler's OpenMP flag, so the GPU bonded
   and LINCS setup loops ran on one thread at every pair search while the GPU waited. Results unchanged.
   Helps GPU bondeds and GPU LINCS; more with frequent pair searches.
4. **Bonded kernel in its own stream** (`e2546053cf`, `listed_forces/`, `mdlib/sim_util.cpp`). The bonded
   kernel overlaps the nonbonded kernel instead of running before it; the nonbonded stream waits for it right
   after the nonbonded launch. Correct with PP domain decomposition; CUDA graph capture follows the event
   dependencies (both checked). Helps runs with `-bonded gpu`.
5. **SETTLE concurrent with LINCS** (`da0eed4161`, `mdlib/update_constrain_gpu_impl.{h,cpp}`). SETTLE runs in
   its own stream while LINCS runs; enabled only when no atom is constrained by both. Bitwise identical
   results. Helps GPU update with both water and other constraints.

## Method and validation

* **Hardware**: one NVIDIA L40S (enforced power limit 325 W), Intel Xeon Gold 6338 VM with 40 logical CPUs,
  CUDA 13.0, mixed precision. Both sides built identically by gmxbench from fixed-length paths.
* **Performance**: baseline and candidate run interleaved in seeded random order, 9 repeats of a 20 s timed
  window per configuration (full tier), PME tuning off. Speedup is the ratio of geometric means with a 95%
  Welch interval; significant changes under 1% are reported as negligible.
* **Correctness**: both builds run the same inputs (64 cases, 184 configurations incl. the upstream
  regressiontests inputs). CPU runs with `-reprod` must match to the bit (73 of 73 IDENTICAL); GPU runs are
  compared at step 0 against their own run-to-run noise (all EQUIVALENT, at most 9x the noise, limit 10x);
  no DIFFERENT or FAIL. Extra checks for the stream changes: CUDA graphs, 2 PP ranks, and 2 PP + 1 PME ranks
  with GPU update, all within noise.
* **Energy conservation**: 2 ns NVE runs of MAS1, two per build: drift 0.606 / 0.607 kJ/mol/ns per atom
  (original / changed); temperature, potential energy and pressure agree within run-to-run scatter.
* **GROMACS tests**: `gmxbench upstream`, 96 of 96 ctest tests pass (23 quick GPU, 17 slow GPU, MPI and
  regression tests included).

## Limits

* One small slowdown: the 5,184-atom water box with CUDA graphs runs 0.7% slower (95% CI -1.1 to -0.3%). At
  0.05 ms per step it is latency-bound; the most likely cause is the extra synchronisation in the SETTLE
  change.
* Measured on one GPU model (L40S, compute capability 8.9). Gains depend on how much of the step the changed
  kernels take.
* HIP is unchanged; SYCL shares the arithmetic of change 1 but was not built or tested here.
* Another job shared the node during part of the testing, which widened the intervals of some CPU-only
  cases; the GPU and MAS1 results have tight intervals.

## Note for the long MAS1 run

Without a thermostat the production integration settings heat MAS1 by about 27 K per ns (identically with
and without these changes). In 0.5 ns NVE tests, about 70% of that comes from LINCS with one iteration
(`lincs-iter = 1`): `lincs-iter = 2` cuts the drift 3.3-fold (0.536 -> 0.165 kJ/mol/ns per atom) for 0.2% less
speed. A 10x tighter `verlet-buffer-tolerance` removes another ~20% for 4.7% less speed. In production the
v-rescale thermostat removes the heat either way; the LINCS setting is worth considering for the long run.

## Reproduce

```bash
bench/bin/gmxbench all -A 3df7585464 -B dev --tier full --strict    # all changes vs the original code
bench/bin/gmxbench upstream --build dev
bench/bin/gmxbench sweep --build dev --target-only
```

Sessions behind these numbers (in the git-ignored `bench/results/` of the machine that ran them):
`20261007-202242-all` (full tier, all changes vs original), `20261007-174958-all` (quick tier, same
comparison), `20261007-113945-all`, `-124442-all`, `-135224-all`, `-145707-all`, `-160204-all` (one change
each), `20261007-184945-upstream`, `20261008-013323-sweep`, and the NVE check in
`phaseB-energy-2026-10-07/`.
