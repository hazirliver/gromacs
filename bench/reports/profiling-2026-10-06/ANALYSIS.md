# MAS1 on one L40S: where the time, energy and data movement go

Single-GPU GROMACS 2026.4 (this fork's `dev`), MAS1 + Gi + 20E membrane system (185,486 atoms, CHARMM36,
2 fs, PME 96x96x144, rc 1.2 nm, LJ force-switch 1.0-1.2 nm, h-bond LINCS + SETTLE, v-rescale + C-rescale
semi-isotropic, nstcalcenergy = nsttcouple = nstpcouple = nstcomm = 100). Node: NVIDIA L40S (Ada AD102,
142 SMs, driver 580.173.02, CUDA 13.0) in a KVM guest with 20 cores / 40 threads of a Xeon Gold 6338
(1 socket, 1 NUMA node visible). Production command:
`gmx mdrun -ntmpi 1 -ntomp 16 -nb gpu -pme gpu -bonded gpu -update gpu -nstlist 200` (+ `-pin on`).

Artifact IDs are paths under this directory (`raw/...`); every number in the tables also appears as a
record (with source, configuration, step type, uncertainty and overhead flag) in `summary.json` /
`summary.csv`. Lab notebook with every command, decision and dead end: `NOTES.md`.

## 0. Method, tools and their overhead

* **Builds** (both `-DCMAKE_CUDA_ARCHITECTURES=89`, sm_89 SASS for all 52 cubins plus compute_89 PTX,
  `-use_fast_math`, GCC 13.3, AVX-512, cuFFT; `raw/env/cmake-*.txt`, `raw/env/cuobjdump-*.txt`):
  **P** = production-like (gmxbench profile `cuda-nosub`: no cycle sub-counters, no NVTX), used for every
  timing; **S** = profile `cuda` (sub-counters + NVTX ranges), used for cycle sub-counters and NVTX step
  ranges. S vs P on the production configuration: 0.996 [0.990, 1.002] (`raw/exp/baseline`) and
  0.9976 [0.9940, 1.0011] (gmxbench, `raw/gmxbench/P-vs-S`); under nsys both builds give the same GPU
  timeline (749.6 vs 750.4 us/step, NB kernel p50 446.8 vs 446.2 us): S-build step analyses apply to P.
* **Steady state**: `-notunepme` (the tuner chooses rc 1.20 / 96x96x144 anyway, 6/6 runs), `-resetstep 5000`,
  20,000-step timed windows; nsys captures cover exactly the 2000 steps after the counter reset
  (`--capture-range=cudaProfilerApi`; GROMACS calls `cudaProfilerStart()` at the reset).
* **Statistics**: experiments are seeded interleaved rounds (5 repeats); ms/step is the geometric mean
  with a 95% t-interval on log values; ratios use a Welch 95% CI on logs (`bench/profiling/summarize.py`,
  same statistics as gmxbench). gmxbench A/A noise floor on the production configuration:
  1.0016 [0.9963, 1.0070], minimum detectable effect 0.74% (`raw/gmxbench/AA-P`). The production
  configuration was measured in 13 independent groups of 5 runs during the session: 0.7398-0.7450 ms/step.
* **Tool overhead** (every profiled configuration also ran unprofiled): nsys light (CUDA + NVTX) +0.6%
  per step (749.5 vs 744.7 us); nsys full (+ OS runtime, 4 kHz CPU sampling, context switches) +1.9%;
  an nsys run's own md.log is unusable (the capture start stalls ~4.5 s inside its timed window);
  perf with DWARF call graphs: 0.742 ms/step = unprofiled (GPU-bound step).
* **GPU performance counters were unavailable**: DCGM (nv-hostengine 4.7.0) holds the profiling
  resources. Nsight Compute fails ("Profiling failed because a driver resource was unavailable. Ensure
  that no other tool (like DCGM) is concurrently collecting profiling data", `raw/ncu/ncu-permission-test.txt`)
  and nsys GPU-metrics sampling reports "Already under profiling". perf_event_paranoid = 0 and
  RmProfilingAdminOnly = 0, so it is not a permission problem. Pausing DCGM affects node monitoring
  and was not pre-authorised (admin commands in EXEC-SUMMARY.md; `bench/profiling/ncu_capture.sh` is ready).
  **Consequently there are no counter-based statements in this report** (no achieved occupancy, stall
  reasons, cache hit rates, roofline position, dynamic instruction mix). Kernel-level statements rest on
  launch records (theoretical occupancy and its limiter), static SASS, the scaling of kernel durations
  with a locked SM clock, ablations and throwaway builds.

## 1. Run level (H1)

| configuration (P build, 5 interleaved runs) | ms/step [95% CI] | ns/day | Matom-steps/s | GPU W | SM MHz | kJ/ns |
|---|---|---|---|---|---|---|
| production (`raw/exp/baseline/runs/prod-P`) | 0.7417 [0.7383, 0.7452] | 233.0 | 250.1 | 304.2 | 2361 | 112.7 |
| plain `gmx mdrun` (`raw/exp/baseline/runs/plain-P`) | 1.0194 [1.0068, 1.0321] | 169.5 | 182.0 | 254.6 | 2514 | 129.3 |

**H1 confirmed**: 233.0 ns/day (CI 231.9-234.0) and 1.374x plain mdrun (CI 1.358-1.391); the earlier
234 ns/day / 1.38x are inside the intervals. Plain mdrun chooses 20 threads, nstlist 80 and keeps the
bondeds on the CPU (0.66 ms/step CPU force work). md.log cycle accounting of the production run: Wait GPU
state copy 71.5% (the CPU waits for the GPU), Neighbor search 6.9% (10.2 ms per search), Force 7.8% (CPU
CMAP and buffer clearing), Launch PP GPU ops 5.4%, PME GPU mesh (mostly launches) 4.4%.

## 2. Step timeline

Source: `raw/nsys/prod-S-light` (2001 steps, CUDA + NVTX), `bench/profiling/nsys_steps.py` (steps from the
NVTX "Step" ranges; every GPU activity is assigned to the step whose CPU code launched it; step type from
the step number, aligned on the search steps) and `budget.py`; cross-checked with `raw/nsys/prod-S-full`
and `raw/nsys/prod-P-light`.

### 2.1 Step types

| step type (step mod) | share of steps | GPU period p50 / p90 / max (us) | share of run time | excess over a plain step |
|---|---|---|---|---|
| plain | 98.0% | 661 / 696 / 1441 | 87.0% | - |
| pair search + energy (mod 200 = 0) | 0.5% | 15,686 / 16,302 / 19,394 | 10.7% | **76.5 us/step = 10.3% of run time** |
| energy/virial + COM removal (mod 100 = 0) | 0.5% | 2,063 / 2,161 / 2,237 | 1.4% | 7.1 us/step = 0.95% |
| after-energy (mod 100 = 1, T/P coupling applied) | 1.0% | 629 | 0.8% | -0.05% |

Output steps (log/energy every 1000, xtc every 500,000) coincide with search steps; removing log and xtc
output changes nothing measurable (0.9986 [0.9947, 1.0024], `raw/exp/nstcalc/runs/perfout`).

### 2.2 A plain step is GPU-bound; its critical path

GPU period 659.8 us: kernels busy 648.4 us, copy-only 4.8 us, idle 6.5 us. The CPU waits 455 us/step in
`cudaEventSynchronize` ("Wait GPU state copy") and runs ~0.5 ms ahead of the GPU (launch-to-start latency
p50 463 us). Median timeline (us from the first GPU activity of the step; s21 = NB local stream, Normal
priority; s22 = PME, High; s23 = update, High; s24 = copy-in stream):

```
s22  |spread 25-113|r2c+fft 121-173|  fft..solve..fft..c2r 278-440  |gather 441-507|
s23  |D2H x 26-113|                                                                |leap-frog 631-637|LINCS 643-662|SETTLE 662-675|
s21     |x->nbat 63-112|bonded 113-173|          NB F kernel 179-624              |reduce 626-629|prune 638-678 (every 2nd step)|
s24                          |H2D f 168-269|
```

Critical path per plain step: **SETTLE(k-1) end -> NB start = 165 us** (the PME spread, High priority,
3.4 waves, takes the SMs first; `x->nbat` - 4.2 us of work when alone, see 4.4 - waits behind it and takes
25-86 us; the bonded kernel runs serially in the NB stream for 59 us); **NB F kernel 445 us** (67%);
**NB end -> SETTLE(k) end = 51 us** (reduction 5 us, leap-frog + LINCS + SETTLE 46 us); 11.6 us gap
before the next spread. The PME chain ends ~95 us before the NB kernel: PME is off the critical path but
shares the SMs (the NB kernel is 446 us with concurrent PME vs 393 us alone, `raw/nsys/contrast-nbonly-S-light`).
Kernel time per step sums to 965 us in a 743 us step; two kernels overlap 41.6% of the time.
Two interference effects: LINCS alternates 13 us / 30 us because the rolling prune kernel (1400 x 256
threads, every 2nd step) runs concurrently with the update chain (~8.5 us/step on the critical path);
`x->nbat` is bimodal (25 / 86 us) depending on when the spread blocks retire.

### 2.3 Search steps: the largest non-kernel cost (10.3% of run time)

Every 200 steps the GPU idles 12.7 ms while the CPU works (`raw/nsys/prod-S-light`, means per search step):
NS grid 1.67 ms + NS search 8.57 ms (16 threads; md.log "Neighbor search" 10.1 ms) + **GPU bonded list
update 2.05 ms on a single thread** (section 5) + uploads (51 MB H2D/D2H per search step, 1.7 ms of copy
time). Then x is re-uploaded (sim_util.cpp:1700), the fresh-list prune kernel (366 us) and the VF kernel
run. Search cost per search grows slowly with the outer list (7.9 ms at rlist 1.372, 10.2 ms at 1.742,
18.2 ms at 2.634, `raw/exp/nstlist`): it has a large fixed part.

### 2.4 Energy / coupling steps (0.95%)

The VF kernel takes 633 us (vs 446 us), the bonded VF kernel 111-131 us (vs 59); the CPU blocks in 4
`cudaStreamSynchronize` (623 us), and COM-motion removal with GPU update makes a blocking round trip
(md.cpp:1761-1842, upstream issues #3988/#4106 cited in the code): D2H x and v, CPU `compute_globals` +
`process_and_stopcm_grp`, H2D x with a blocking wait, H2D v: 4 x 2.2 MB, 639 us copy-only per energy
step. Raising nstcalcenergy alone does nothing (coupling and COM removal still need global steps every 100);
raising all four intervals to 1000 gives +1.17% [0.75, 1.59] (`raw/exp/nstcalc`, timing only: tau_p = 5 ps is
then < 25 nstpcouple dt).

### 2.5 H2: CMAP on the CPU - mechanism confirmed, cost about 1%

* Confirmed: CMAP (1009 terms) is the only CPU force term (all other listed types are in `fTypesOnGpu`);
  every non-search step copies x D2H (2,225,832 B, 87 us, update stream) and the CPU forces H2D
  (2,225,832 B, 101 us, a dedicated copy-in stream): 2.354 MB D2H + 2.445 MB H2D per step at 25.6 / 21.8 GB/s
  (PCIe probe: 25.5 / 24.5 GB/s, 4.0 us small-copy round trip, identical from every vCPU, `raw/pcie/probe.txt`).
  With `GMX_CUDA_GRAPH=1` md.log has no "MD Graph" row: graphs never engage, and no message says so.
* In plain steps both copies overlap the PME and NB kernels (2.2); the CPU CMAP work (35 us "Bonded F",
  12 us clearing, 9 us buffer ops) is hidden behind the GPU. Copy-only GPU time in plain steps: 4.8 us/step.
* Ablation (`raw/exp/cmap`, `raw/exp/ablation`, `raw/exp/power`; CMAP removed = timing only): no CMAP
  1.0013 [0.9987, 1.0039] and 0.9992 [0.9963, 1.0022] (no change); no CMAP + CUDA graphs (which then engage:
  "MD Graph" cycle counter with 20,200 calls in the 20,000-step window) **1.0089 [1.0033, 1.0146]** and 1.0113 [1.0084, 1.0142]. Graphs
  shorten the plain-step GPU period by ~11 us (658 -> 648 us); the SM clock drops ~10 MHz because the GPU
  idles less under the power cap; energy -0.6 / -0.7% per ns.
* **Verdict**: the CMAP-on-CPU path (copies, CPU work, lost graphs) costs ~1% (CI 0.3-1.5%) on this system.
  The PCIe traffic (3-4 GB/s each way) is real but off the critical path. Refuted as the main bottleneck.

## 3. Stage level

Per-kernel GPU time, production (`raw/nsys/prod-S-light`; P build identical within +-2%):

| kernel (variant in use) | us/step | share of kernel time | p50 duration | on critical path? |
|---|---|---|---|---|
| `nbnxn_kernel_ElecEw_VdwLJFsw_F_cuda` (analytical Ewald, LJ force-switch, forces only, NBFIX table LJ) | 440.3 | 45.6% | 446.2 | yes |
| cuFFT `regular_fft<96,...>` x4, `regular_fft_r2c<144>`, `regular_fft_c2r<144>` | 95.8 + 20.9 + 35.2 | 15.7% | 23 / 18 / 26 | no |
| `pme_spline_and_spread_kernel<4,...>` | 89.2 | 9.2% | 89.0 | delays NB start |
| `pme_gather_kernel<4,...>` | 65.1 | 6.7% | 65.6 | no |
| `bonded_kernel_gpu<false,false>` | 58.8 | 6.1% | 59.1 | yes (serial before NB) |
| `nbnxn_gpu_x_to_nbat_x_kernel` | 54.6 | 5.7% | 27.1 (4.2 alone) | yes (waits for SMs) |
| `pme_solve_kernel` | 31.3 | 3.2% | 28.3 | no |
| `lincsKernel<true,false>` / `settleKernel` / `leapFrogKernel` | 21.4 / 12.4 / 6.2 | 4.1% | 14.3 / 12.5 / 6.0 | yes |
| `nbnxn_kernel_prune_cuda<false>` (rolling, every 2nd step) | 19.9 | 2.1% | 39.8 | partly (slows LINCS) |
| `nbnxn_kernel_ElecEw_VdwLJFsw_VF_cuda` (energy steps) | 6.7 | 0.7% | 633.1 | yes (energy steps) |
| `reduceKernel` | 3.6 | 0.4% | 3.6 | yes |
| `nbnxn_kernel_prune_cuda<true>` (fresh list, search steps) | 2.0 | 0.2% | 366 | yes (search steps) |
| memcpy D2H / H2D / memset | 91.8 / 112.0 / 25.8 | - | 87 / 101 / - | mostly overlapped |

Contrast modes (`raw/nsys/contrast-*-S-light`) only explain why full residency is needed: bondeds on the
CPU -> plain step 1032 us with 345 us GPU idle (+56%); CPU update (NB+PME on GPU) -> 1213 us, 416 us idle;
NB only -> 5245 us, 4625 us idle. Every non-resident mode is CPU-bound.

## 4. Kernel level (no GPU counters; see section 0)

### 4.1 Launch configuration and theoretical occupancy (`bench/profiling/occupancy.py`, sm_89 rules)

| kernel | block | regs | blocks/SM | theoretical occupancy | limiter | waves |
|---|---|---|---|---|---|---|
| NB F / VF | 64 (8x8) | 61 / 64 | 16 | 66.7% | registers (`__launch_bounds__(64,16)`) | 4.9 |
| PME spread | 256 | 40 | 6 | 100% | registers = warps | 3.4 |
| PME gather | 128 | 40 | 12 | 100% | registers = warps | 3.4 |
| cuFFT regular_fft / r2c / c2r | 192 / 96 / 96 | 40 / 47 / 48 | 8 / 14 / 14 | 100 / 87.5 / 87.5% | registers | 0.39 / 0.58 / 0.58 |
| bonded | 256 | 40 | 6 | 100% | registers = warps | 2.5 |
| LINCS / SETTLE / leap-frog | 256 | 40 / 61 / 26 | 6 / 4 / 6 | 100 / 66.7 / 100% | - | 0.18 / 0.27 / 0.85 |

The NB kernel's launch bounds `(64, 16)` (nbnxm_cuda_kernel.cuh:132-133, chosen for CC >= 5.0) give
16 blocks x 2 warps x 2048 registers = the whole 64K register file: on Ada (48 warps, 24 blocks per SM)
occupancy is capped at 66.7% and no other kernel can co-reside on an SM full of NB blocks. Raising the
occupancy with 20 or 24 blocks per SM makes the kernel slower (section 10), so the register limit is not
its performance limiter on this GPU. The update
kernels are sub-wave (LINCS uses 150 blocks on 142 SMs) and latency-bound. No register spills in any
of the NB variants (`raw/sass/resource-usage-P.txt`).

### 4.2 Clock scaling (`raw/nsys/clk*-S-light`, `raw/nsys/clockscale.json`)

With the SM clock locked at 1500 / 2010 / 2400 MHz (memory clock fixed at 9001 MHz), duration ~ f^-alpha:
NB F kernel alpha = 0.90, PME solve 0.98, c2r 0.92, regular_fft 0.86, gather 0.82, prune 0.85 -> these
scale with the graphics clock and are **not DRAM-bandwidth bound**; spread 0.64, bonded 0.66, r2c 0.61,
LINCS 0.73, SETTLE 0.68, leap-frog 0.66 -> partly limited by something outside the SM clock domain (DRAM
or atomic latency); copies alpha = 0 (PCIe). Plain step 0.85, mean step 0.75 (search steps are CPU-bound).
Whether the NB kernel is issue-, latency- or L1/shared-bound needs ncu.

### 4.3 Static SASS (`raw/sass/`)

`ElecEw_VdwLJFsw_F` has 1928 SASS instructions vs 1696 for `ElecEw_VdwLJ_F` (potential-shift): the force
switch adds 232 instructions (+13.7%: FFMA +80, FMUL +96, FSEL/FSETP +32, MOV +32). It is evaluated
branch-free for every pair (`ljForceSwitch`, nbnxm_kernel_utils.h:68-98). 32 MUFU (RSQ, RCP), 11
`RED.E.ADD.F32` global atomics, 55 SHFL; 54 `CALL.REL` to two tiny SHFL subroutines (the compiler's
non-convergent shuffle path) in the per-i-cluster force-reduction epilogue.

### 4.4 Ablations as Amdahl numerators (TIMING ONLY inputs, P build)

| ablation | vs production (95% CI) | energy/ns | kernel-level split (nsys) |
|---|---|---|---|
| LJ force-switch -> potential-shift | **1.0691 [1.0640, 1.0743]** | -7.3% | NB kernel 446 -> 406 us (**-40 us/step**, -8.9% of the NB kernel); rest of the gain from a smaller outer list (1.642 vs 1.742 nm: search 14.7 vs 15.7 ms) partly offset by pruning every 10 instead of 16 steps |
| PME -> reaction-field (eps_rf = inf) | **1.2404 [1.2354, 1.2454]** | -18.7% | NB kernel 446 -> 347 us (Ewald real-space cost -99 us), no PME kernels, `x->nbat` 4.2 us, plain step 661 -> 504 us; SM clock -92 MHz (busier GPU under the cap) |
| no CMAP | 0.9992 [0.9963, 1.0022] | 0% | copies and CPU work hidden (2.5) |
| no CMAP + CUDA graphs | 1.0089 [1.0033, 1.0146] | -0.6% | plain step -11 us |
| constraints off (LINCS / LINCS+SETTLE) | 0.73 / 0.64 (slower) | +31% / +51% | **dead end**: flexible bonds raise the Verlet buffer, outer rlist 1.742 -> 3.28 / 3.46 nm; LINCS/SETTLE/leap-frog cost taken from nsys (46 us/step on the critical path) |

### 4.5 Kernel-variant knobs (`raw/exp/knobs`, env vars verified in src/)

No knob beats the defaults: `GMX_GPU_NB_TAB_EWALD` 0.9849 [0.9761, 0.9938] (slower), `GMX_GPU_NB_EWALD_TWINCUT`
0.9936, `GMX_NB_MIN_CI=0` 0.9914, `GMX_NSTLIST_DYNAMICPRUNING=8` 0.9982 / `=32` 0.9896 (slower),
`GMX_DISABLE_ALTERNATING_GPU_WAIT` 1.0032 (all vs an in-experiment default with one slow run; the
non-significant ones are also below the 0.7398-0.7425 ms/step measured for production elsewhere),
`GMX_DISABLE_DYNAMICPRUNING` 0.7709 (dynamic pruning is worth 23%).

## 5. CPU side

* `raw/perf/prod-P-fp` (8 s after the reset, 16 threads x 4999 Hz): all 16 OpenMP threads are always
  on-CPU and 82.9% of their samples are libgomp spin-waiting (the CPU is idle-spinning; CPU energy is not
  observable in this VM: no RAPL/powercap). Real work: `nbnxn_make_pairlist_part<GPU>` 4.84% of samples
  (~115 CPU-ms per search, ~84% parallel efficiency), CMAP ~3% (cmap_dihs, accumulateCmapForces,
  dih_angle, libm atan/asin/acos), clearRVecs 0.9%, reduceThreadForceBuffers 0.5%, grid sorting 0.6%.
* **Serialised OpenMP in nvcc-compiled host code**: `ListedForcesGpu::Impl::updateInteractionListsAndDeviceBuffers`
  spends 1.9 CPU-ms per search (perf annotate: the index gather of `convertIlistToNbnxnOrder`, which has
  `#pragma omp parallel for`) on one thread. listed_forces_gpu_impl_gpu.cpp is compiled by nvcc (`-x cu`)
  with CUDA_FLAGS that lack -fopenmp (CXX_FLAGS have it); the object file has no GOMP call and no `_omp_fn`
  symbols. `mdlib/lincs_gpu.cpp` has the same problem but its loops run only at start-up/with DD here.
  Cost: 2.05 ms of GPU idle per search = 1.4% of run time.
* Search scaling (`raw/exp/threads`): 28.7 / 16.3 / 12.0 / 10.1 / 8.9 ms per search with 4 / 8 / 12 / 16 / 20
  threads; hyperthreads do not help (10.4 ms at 24 = 12 cores x 2). Plain steps are insensitive to the
  thread count (GPU-bound).
* Launch and wait paths: 10.5 `cudaLaunchKernel` (49 us) + 6 cuFFT `cuLaunchKernel` (21 us) + 3
  `cudaMemcpyAsync` per plain step on the main thread; it busy-spins in `cudaEventSynchronize` (no
  futex/poll waits in the OS-runtime trace) and is context-switched 13 times in 2000 steps on one CPU
  (`raw/nsys/prod-S-full`).
* NUMA/locality: one socket and one NUMA node are visible; copy latency and bandwidth are identical from
  vCPUs 0/1/10/20/30/39; pinning variants (`raw/exp/pinning`) are equal (pin off 1.0008 [0.9976, 1.0040],
  other cores 0.9982) unless threads share cores (16 threads on 8 cores: 0.9738, search 13.5 ms).

## 6. Power and energy (H3)

`raw/exp/power` (NVML at 10 Hz over the timed window): the **enforced power limit is 325 W and cannot be
raised** on this node (requesting 350 W sets `power.limit` 350 W but `enforced.power.limit` stays 325 W).
At 325 W `sw_power_cap` is active in 100% of samples; mean SM clock 2362 MHz (p05 2310) = 6.3% below the
2520 MHz maximum; mean power 307 W (1 s maxima 324 W).

| limit / lock | ms/step | vs 325 W | SM MHz | W | kJ/ns |
|---|---|---|---|---|---|
| 325 W (production) | 0.7415 | 1 | 2362 | 306.8 | 112.8 |
| 300 W | 0.7623 | 0.9727 [0.9565, 0.9891] | 2284 | 295.0 | 111.1 |
| 250 W | 0.8382 | 0.8846 [0.8818, 0.8875] | 1938 | 247.8 | 104.0 |
| lock 2250 MHz | 0.7586 | 0.9775 [0.9748, 0.9802] | 2250 | 290.6 | 110.1 |
| lock 1980 MHz | 0.8418 | 0.8809 [0.8783, 0.8834] | 1980 | 247.4 | 103.8 |

From the locked runs, T(f) = 0.148 ms + 1373 ms*MHz / f: ~80% of the step scales with the SM clock;
uncapped (2520 MHz) would give ~0.693 ms/step, i.e. **the power cap costs ~6.5%** (not attainable here).
Consequences for optimisation: (a) under the cap, energy per ns follows time per ns for work reductions
(potential-shift -6.5% time / -7.3% energy; RF -19.4% / -18.7%); (b) removing GPU idle time is partly
paid back by a lower clock (no-CMAP + graphs: -10 MHz, -1.1% time but only -0.6% energy); (c) lowering
the clock trades speed for energy almost 1:1 near the cap (2250 MHz: -2.3% speed, -2.4% energy).
**H3 confirmed** (throttling, 6% clock deficit; 112-113 kJ/ns, a little below the earlier 116 kJ/ns).

## 7. Configuration sweeps (H4)

| knob | best value | effect vs production (95% CI) | artifact |
|---|---|---|---|
| -ntomp (pinned, 1 thread/core) | 20 | +0.58% [0.15, 1.02] (16 is production; 4: -11%, 8: -4%, 12: -1.3%, 24/32 HT: -0.7% / +0.2%) | `raw/exp/threads` |
| -nstlist | 200 | 150: -0.46%, 300: -2.4%, 100: -3.2%, 80: -5.5%, 400: -7.3% | `raw/exp/nstlist` |
| nstcalcenergy | any | 500 / 1000 alone: no change (coupling still every 100) | `raw/exp/nstcalc` |
| all global intervals | 1000 | +1.17% [0.75, 1.59] (changes coupling; protocol decision) | `raw/exp/nstcalc` |
| pinning | on or off | no difference unless cores are shared (-2.6%) | `raw/exp/pinning` |
| PME rc / grid | 1.20 / 96x96x144 | 1.25: -5.9%, 1.30: -10.4%, 1.40: -19.5%; tuner picks 1.20; tuner left on: -0.47% | `raw/exp/pme` |
| GMX_CUDA_GRAPH | irrelevant | 0.9997 [0.9956, 1.0039] (cannot engage with CPU CMAP) | `raw/exp/cmap` |
| power limit | 325 W (max enforced) | lower limits trade speed for energy | `raw/exp/power` |
| kernel env knobs | defaults | see 4.5 | `raw/exp/knobs` |

**H4 mostly refuted**: the production command is already at the optimum of every runtime knob except
-ntomp 20 (+0.6%); mdrun's own defaults (plain mdrun) are far off only because of -bonded and -nstlist.

## 8. Hypothesis verdicts

| | claim | verdict | key evidence |
|---|---|---|---|
| H1 | production 234 ns/day, 1.38x plain | **confirmed**: 233.0 ns/day [231.9, 234.0], 1.374x [1.358, 1.391] | `raw/exp/baseline`, `raw/gmxbench/AA-P` |
| H2 | CPU CMAP forces copies every step and blocks graphs | mechanism **confirmed**; as a bottleneck **refuted** (~1%) | 2.5 |
| H3 | sw_power_cap throttling, clock ~3% below max, ~116 kJ/ns | **confirmed**, larger than thought: 6.3% clock deficit, 325 W enforced cap; 112.7 kJ/ns | 6 |
| H4 | defaults suboptimal (nstlist, tuner, nstcalcenergy, threads, pinning) | **mostly refuted** for the production command (only -ntomp 20, +0.6%) | 7 |

## 9. Connecting the layers

1. **Run level**: 0.742 ms/step, GPU-bound (CPU waits 71% of the time), power-capped at 325 W.
2. **Step timeline**: 87% of the time is plain steps (GPU-bound, 6.5 us idle each); 10.3% is the GPU
   idling during CPU pair searches every 200 steps; ~1% energy/COM steps; ~1% the CPU CMAP path.
3. **Stage**: in a plain step the critical path is pre-NB work (165 us: PME spread first by stream
   priority, `x->nbat` waiting for SMs, serial bonded kernel) -> NB F kernel (445 us) -> reduction +
   update chain (51 us). PME (340 us of kernel time) runs concurrently and is not on the path but slows
   the NB kernel by ~12% through SM sharing.
4. **Kernel**: the NB kernel runs at a register-limited 66.7% theoretical occupancy, but more occupancy
   is slower (section 10); its duration scales with the SM clock (alpha 0.90, not DRAM-bound); ~9% of its
   time is the CHARMM force switch and ~22% the Ewald real-space terms (vs RF). Update kernels are
   sub-wave, latency-bound kernels.
5. **Hardware limiter**: SM throughput at a power-capped clock (6.3% below max) for the plain steps;
   CPU pair search (single-socket VM, 16 cores) for 10% of the time.

## 10. Throwaway experiments (code changes outside src/)

Worktree `/home/asokolov/Projects/gromacs-exp-nbminblocks` (detached at `dev`), every change behind a
compile-time define, built as separate production-like builds (`bench/profiling/experiments/exp-nbminblocks-suite.toml`)
and measured interleaved against the upstream P build (`raw/exp/exp-nbminblocks`, 5 rounds) plus an nsys
light capture each (`raw/nsys/exp-nbminblocks-<build>-light`):

| change | vs upstream (95% CI) | mechanism check (nsys) | quality gate |
|---|---|---|---|
| OpenMP for nvcc-compiled host code (`-Xcompiler=-fopenmp`) | **1.0085 [1.0038, 1.0132]** | GPU idle 78.1 -> 69.5 us/step (bound 10.3) | `--strict` passed: CPU IDENTICAL, GPU EQUIVALENT (`raw/gmxbench/quality-ompcuda`) |
| GPU bonded kernel in its own stream (`GMX_EXP_BONDED_STREAM`) | **1.0091 [1.0029, 1.0153]** | NB kernel 446.8 -> 459.7 us (shares SMs) but starts earlier; GPU span 749.6 -> 739.2 us/step | passed (`raw/gmxbench/quality-bondedstream`) |
| stream priorities swapped (NB local High, PME Normal) | 0.9943 [0.9895, 0.9991] (slower) | PME chain pushed onto the critical path | - |
| NB launch bounds 20 blocks/SM (48 regs + 8 B stack) | 0.9596 [0.9547, 0.9645] (slower) | NB kernel 446.8 -> 473.8 us; SM clock 2322 MHz | - |
| NB launch bounds 24 blocks/SM (40 regs + 40 B stack) | 0.8358 [0.8331, 0.8386] (slower) | NB kernel 574.8 us; SM clock 2110 MHz (more resident warps draw more power under the cap) | - |

So the register-limited 66.7% occupancy is **not** the NB kernel's limiter (refuted by measurement), the
upstream stream priorities are right, and two small changes are real: +0.85% (OpenMP) and +0.91% (bonded
stream), both bit-identical on CPU paths and within GPU noise. The combined build: see `raw/exp/exp-combo`
vs upstream in `raw/exp/exp-combo` (5 rounds; the upstream reference ran at 0.7436 ms/step there): OpenMP fix
1.0107 [1.0047, 1.0168], bonded stream 1.0131 [1.0071, 1.0192], **both 1.0203 [1.0142, 1.0265]**, both with
-ntomp 20 **1.0282 [1.0219, 1.0345]** (0.7232 ms/step = 239.0 ns/day, 111.8 kJ/ns, -1.0% energy); upstream with
-ntomp 20 1.0110 [1.0044, 1.0175]. The three effects add up. The combined build also passed
`quality --strict` (`raw/gmxbench/quality-combo`).

## 11. Not measured / limitations

* No GPU hardware counters (DCGM), hence no achieved occupancy, warp-stall reasons, L1/L2 hit rates,
  roofline position or dynamic instruction mix; nsys GPU-metrics sampling (SM activity, DRAM and PCIe
  throughput timelines) likewise unavailable. NVML PCIe counters were recorded per run instead.
* CPU energy: no RAPL in the VM. GPU energy only (NVML hardware counter).
* Local vs remote socket: the VM exposes one socket.
* Power above 325 W: enforced limit. Clock locks above the cap ran at 2400 MHz.
* Constraint cost by ablation (confounded, 4.4); RF and potential-shift ablations change the pair-list
  buffer, so their kernel-level split comes from nsys.
