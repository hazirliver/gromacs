# Lab notebook: MAS1 single-GPU profiling (L40S + Xeon Gold 6338)

Session start 2026-10-06T19:03Z. Branch `feature/profiling` (from `dev` @ 3df7585464; GROMACS sources
identical to v2026.4 = 6c69f720aa, `git diff v2026.4 dev -- . ':!bench'` touches only CLAUDE.md).
All times UTC. Artifact IDs used in the reports are the paths under `raw/` relative to this directory.

## 19:03 Machine state (quiet-node check)
- GPU idle (P8, 34 W, 0 MiB used, no processes). Load average 0.08. Only background daemons
  (nv-hostengine/DCGM, nebius-observability-agent, atop at 60 s interval). Node is quiet.
- Host is a **KVM guest**: 1 socket, 1 NUMA node, 20 cores x 2 HT = 40 vCPUs (Xeon Gold 6338, 2.0 GHz
  base); `nvidia-smi topo -m`: GPU0 CPU affinity 0-39, NUMA 0. There is no second socket visible
  -> the "GPU-local vs remote socket" pinning experiment is not possible here; replaced by
  HT-sibling vs one-thread-per-core pinning and an H2D/D2H latency probe per vCPU.
- No cpufreq interface in the guest (governor not visible/controllable). THP = madvise.
- `kernel.perf_event_paranoid = 0`, `kptr_restrict = 0`, `nmi_watchdog = 0`;
  NVIDIA `RmProfilingAdminOnly: 0` -> perf and GPU counters usable without root.
- sudo available for clock/power queries. GPU power limit currently **325 W** (default and max 350 W, min 100 W);
  this is the production state and must be restored after the power experiments.
- Tools: nsys 2025.3.2.474, ncu 2025.3.1.0, perf 6.11.11, CUDA toolkit 13.0.88 (nvcc), driver
  580.173.02, kernel 6.11.0-1016-nvidia.

## 19:05 Builds
- `cuda` profile (sub-counters + NVTX): existing gmxbench build of v2026.4,
  `~/.cache/gmxbench/builds/780edef73168a96d` (CMAKE_CUDA_ARCHITECTURES=89, -use_fast_math).
  Used via `path:` so the binary stays fixed for the whole session.
- `cuda-nosub` (production-like, no sub-counters/NVTX): `gmxbench build v2026.4 --profile cuda-nosub`
  started 19:05 (background).

## 19:10 Inputs (bench/profiling/mkinputs.py -> raw/inputs/<name>/)
- `prod`: data/ md.mdp unchanged except continuation=yes, gen-vel=no, nsteps=-1 (output as in production:
  nstlog=nstenergy=1000, xtc every 500000, nstcalcenergy=100). gmx dump: nsttcouple=nstpcouple=100,
  nstcomm=100, PME grid 96x96x144 (fourierspacing 0.12), rlist 1.209 in the tpr.
- Only CPU-side listed interaction in the tpr is CMAP: 1009 terms (PROA 300, PROB 333, PROC 53, PROD 323);
  all other listed types (bonds, U-B, proper/improper dihedrals, LJ-14) are GPU-capable.
- Timing-only ablations (README in each dir says so): nocmap (1009 [cmap] lines removed), potshift
  (vdw-modifier Potential-shift), rf (Reaction-field, eps_rf=0), nolincs (constraints=none, SETTLE kept),
  noconstr (+ TIP3 SETTLE replaced by flexible bonds/angle), nocmap-potshift, nstglob500/1000
  (nstcalcenergy=nstcomm=nsttcouple=nstpcouple=N; grompp warns tau_p < 25*nstpcouple*dt).
  Physical variants: nstcalc500/1000 (only nstcalcenergy; T/P coupling stays at 100 so global
  communication still happens every 100 steps), pme-rc1.20..1.40 (rcoulomb and fourierspacing scaled
  together as the PME tuner does; grids 96x96x144 / 84x84x128 @1.30 / 80x80x120 @1.40).
- TIP3.itp has no FLEXIBLE ifdef (SETTLE is unconditional), hence the topology edit for `noconstr`.

## 19:12 Harness sanity run (scratch, not used): prod-P 0.735 ms/step, 235.1 ns/day, P_mean 304 W,
  SM clock mean 2393 MHz (p05 2355, max 2475 of 2520), sw_power_cap active in 148/148 samples (100%),
  PCIe tx 3.95 GB/s rx 4.73 GB/s (NVML), 111.2 kJ/ns. mdrun chose nstlist 200 -> rlist 1.742 nm.

## 19:13 Experiment `baseline` (raw/exp/baseline): prod-P, prod-S, plain-P; 5 interleaved rounds,
  25000 steps, -resetstep 5000 (plain: PME tuning is on, gmxbench's wrapper doubles resetstep until the
  tuner has finished: it needed resetstep 20000, 40000 steps).
- Plain mdrun's PME tuner scanned rc 1.20..1.64 and picked the original 96x96x144 / rc 1.200
  (raw/exp/baseline/runs/plain-P/rep0/run.log).

## 19:16 Upstream prior art -> raw/prior-art.md
- No CMAP GPU kernel upstream (2026.4 or main). Only attempt: MR !1730 "CUDA CMAP kernel" (P. Bauer),
  closed unmerged 2021-11-08 (verified via GitLab API). CUDA graphs still opt-in (`GMX_CUDA_GRAPH`),
  disabled for the whole domain when `haveCpuLocalForceWork` (CPU CMAP counts). Unmerged prototype
  "allow CPU forces in CUDA graph via memops" (A. Gray, commit 0030776d, authored 2023-12-12, verified).
- Env vars verified in src/: GMX_GPU_NB_ANA_EWALD / GMX_GPU_NB_TAB_EWALD / GMX_GPU_NB_EWALD_TWINCUT
  (not GMX_CUDA_NB_*), GMX_NB_MIN_CI, GMX_DISABLE_DYNAMICPRUNING, GMX_NSTLIST_DYNAMICPRUNING,
  GMX_DISABLE_ALTERNATING_GPU_WAIT, GMX_GPU_DISABLE_BUFFER_OPS, GMX_LISTED_FORCES_NUM_THREADS.
- Note: md.cpp only uses a graph on steps that are not NS/virial/T-coupling/PR/global-stat/output
  steps; with nstcalcenergy=nstcouple=100 and nstlist=200 at most ~98% of steps can use a graph.

## 19:33 Baseline results (raw/exp/baseline/summary.json; `summarize.py exp ... --ref prod-P`)
| variant | n | ms/step geomean [95% CI] | ns/day | min-max ms | P W | SM MHz | sw_power_cap frac | kJ/ns |
|---|---|---|---|---|---|---|---|---|
| prod-P | 5 | 0.7417 [0.7383, 0.7452] | 232.98 | 0.7381-0.7450 | 304.2 | 2361 | 1.00 | 112.66 |
| prod-S | 5 | 0.7447 [0.7405, 0.7488] | 232.06 | 0.7412-0.7496 | 304.7 | 2362 | 1.00 | 112.95 |
| plain-P | 5 | 1.0194 [1.0068, 1.0321] | 169.52 | 1.0099-1.0309 | 254.6 | 2514 | 0.31 | 129.32 |
- H1 confirmed: 233.0 ns/day (CI 231.9-234.0), 0.742 ms/step, 250.1 Matom-steps/s; production/plain =
  1.374x (CI 1.358-1.391). S build vs P: 0.996 [0.990, 1.002], no significant overhead.
- Plain mdrun = 1 rank x 20 OpenMP threads, nstlist 80 (rlist 1.372), NB+PME+update on GPU but
  bondeds on the CPU (log: "short-ranged interactions on the GPU", no "most bonded"): CPU Force
  0.662 ms/step = 64% of wall time.
- prod-P cycle table (rep0, 20000 steps): Neighbor search 1.032 s / 101 calls = **10.2 ms per search**
  (51.6 us/step amortised, 6.9%); Force (CPU: CMAP + clearing) 58 us/step (7.8%); Launch PP GPU ops
  40 us/step; PME GPU launch 29 us/step; Wait GPU state copy 531 us/step (71.5%, 2 waits/step).
- Interim wait-loop bug: `pgrep -f` matched the waiting shell itself; killed, queue1 started 19:32.

## 19:37 Code reading: CPU-force path in the GPU-resident step (v2026.4)
- sim_util.cpp:1677-1683: on non-search steps with GPU update and `haveCpuLocalForceWork` (CMAP), x is
  copied D2H every step (`copyCoordinatesFromGpu`, stream = update stream,
  state_propagator_data_gpu_impl_gpu.cpp:105); the CPU waits for it at sim_util.cpp:2052 before the
  CPU listed forces ("Wait GPU state copy").
- sim_util.cpp:2608-2610: the CPU forces are copied H2D with `copyForcesToGpu(..., Local)`, whose
  stream is `fCopyStreams_[Local] = localStream_` (the **nonbonded local stream**, line 113), i.e.
  stream-ordered after the NB kernel; then `gpuForceReduction[Local]->execute()` and the update.
  Source TODO at 2600: "CPU f H2D should be as soon as all CPU-side forces are done".
- Hypothesis H2b (to test with nsys): the ~2.2 MB H2D cannot start before the NB kernel ends, so it
  sits on the critical path (NB -> H2D f -> reduction -> update) instead of overlapping the NB kernel.
- 19:41 CORRECTION (refutes H2b by code reading): `copyForcesToGpu` does NOT use `fCopyStreams_`; it uses
  the module's own `copyInStream_` (state_propagator_data_gpu_impl_gpu.cpp:117/585, comment: "to allow
  overlap with GPU force calculations") and marks `fReadyOnDevice`, on which the local force
  reduction already depends (sim_util.cpp:1269). `fCopyStreams_[Local] = localStream_` is only used for
  the D2H force copy. So the H2D is not stream-ordered behind the NB kernel; whether it is exposed is
  a timing question for nsys. The throwaway worktree created for a stream change was removed unused.

## 19:36 gmxbench A/A noise floor (raw/gmxbench/AA-P; P build both sides, case mas1/charmm36 = perf-out
  mdp (nstlog 0, no xtc), config perf-mas1-production, 5 interleaved repeats, 10400-step window)
- A 234.24 ns/day (CI 232.97-235.52, CV 0.44%), B 234.62 (CI 234.02-235.22, CV 0.20%);
  B/A = 1.0016 [0.9963, 1.0070], p = 0.48, no-change; minimum detectable effect 0.74%.
- nsys kernel table (gmxbench whole-run capture, 3000 steps incl. start-up; us/step): NB F kernel 442,
  memcpy H2D 110, D2H 92, FFT 96+36+21, PME spread 89, gather 65, bonded 59, x->nbat 54, solve 32,
  memset 26, LINCS 21, prune 20, SETTLE 12. NVML: 304.9 W mean, 112.3 kJ/ns, util 91%.

## 19:45 nsys light capture of prod-S (raw/nsys/prod-S-light; analysis prof.analysis.json via
  `nsys_steps.py prof.sqlite --first-step 5000`), 2001 steps 5000-7000, cuda+nvtx tracing only
- Tool overhead: steady-state NVTX step mean 749.5 us vs unprofiled S 744.7 us (x1.006). The md.log of
  the nsys run reports 3.0 ms/step because only 2000 steps are timed and the capture start (at the
  counter reset) stalls ~4.5 s: md.log numbers of capture-range runs are NOT usable for timing.
- Step types (CPU step period from NVTX "Step" starts; base step 5000 from NS steps at mod 200 = 0):
  plain 98.0%: p50 653.7 us, p90 702 us, max 1441 us; GPU idle 6.5 us/step.
  energy (mod 100, not NS) n=10: p50 2063 us, GPU idle 478 us.
  ns+energy (mod 200) n=10: p50 15684 us, max 19394 us, GPU idle 12673 us (!).
  after-energy (mod 100 = 1) n=20: p50 252 us CPU period (CPU catches up; GPU period 629 us).
  -> mean 742.9 us vs plain p50 654 us: non-plain steps cost ~12% of the run.
- GPU timeline over the capture: kernels busy 87.0%, copy-only 2.5% (19 us/step), idle 10.4% (78 us/step);
  2 kernels concurrently 41.6% of the time (NB || PME). Copies 204 us/step total, 185 us overlapped.
- Plain-step anatomy (median offsets from the step's first GPU activity; s21 = NB local stream, s22 PME,
  s23 update, s24 copy-in): s21 x->nbat 63-112, bonded 113-173, NB F 179-624, reduce 626-629;
  s23 leapfrog 631-637, LINCS 643-662, SETTLE 662-675; s21 rolling prune 638-678 (every 2nd step);
  s22 spread 25-113, FFTs/solve 121-440, gather 441-507 (PME chain ends ~530, NOT critical);
  s23 D2H x 26-113 and s24 H2D f 168-269 overlap PME/NB kernels -> the CMAP copies are hidden in plain steps.
  Critical path = s21 chain (x->nbat, bonded, NB F, reduce) + update chain (s23).
- NS-step anatomy: D2H x+v (0-179 us), then CPU: NS grid local 1670 us, **GPU Bonded list update
  2052 us**, NS search local 8572 us ("Neighbor search" range total 10326 us); pair-list/bonded-list
  uploads (51 MB/NS step H2D, 1.7 ms) mostly 2.0-4.1 ms; GPU idle until ~13.0 ms; then prune<fresh>
  366 us + VF kernel 624 us.
- Energy-step anatomy: VF NB kernel 660 us (vs F 445), bonded VF 131 us; CPU "Wait GPU NB local" 587 us,
  "Launch PP GPU ops." 492 us; after SETTLE, s23 does D2H 2x2.2 MB + H2D 2x2.2 MB (1432-2027 us) +
  scaleCoordinatesKernel x2 (C-rescale coupling path) on the critical path.
- CUDA API (main thread): cudaEventSynchronize 449 us/step (2.04 calls); cudaLaunchKernel 10.5 calls,
  49.5 us/step; cuLaunchKernel (cuFFT) 6 calls, 21.3 us; cudaMemcpyAsync 3.25 calls; launch->start latency
  p50 463 us (CPU runs ~0.5 ms ahead of the GPU in plain steps: GPU-bound).

## 19:50 H2 ablation `cmap` (raw/exp/cmap; P build unless -S; 5 interleaved rounds, 20000-step windows)
| variant | ms/step [95% CI] | ns/day | vs prod (ns/day ratio, Welch 95% CI) | SM MHz | kJ/ns |
|---|---|---|---|---|---|
| prod | 0.7425 [0.7406, 0.7445] | 232.73 | 1 | 2360 | 113.24 |
| prod+graph (GMX_CUDA_GRAPH=1) | 0.7427 [0.7397, 0.7458] | 232.66 | 0.9997 [0.9956, 1.0039] | 2360 | 113.39 |
| nocmap (timing only) | 0.7415 [0.7405, 0.7425] | 233.04 | 1.0013 [0.9987, 1.0039] | 2359 | 113.17 |
| nocmap+graph (timing only) | 0.7359 [0.7318, 0.7401] | 234.82 | **1.0089 [1.0033, 1.0146]** | 2349 | 112.53 |
| nocmap+graph-S | 0.7361 [0.7337, 0.7385] | 234.76 | 1.0087 [1.0052, 1.0122] | 2353 | 112.68 |
- Graph engagement from md.log: prod+graph prints the env-var notes but has NO "MD Graph" row (graphs never
  used with CPU CMAP, as the code says); nocmap+graph: "MD Graph" row, 20200 calls (all non-NS/energy steps).
- H2 verdict: mechanism confirmed (per-step D2H x / H2D f 2.2 MB each, graphs disabled), but the exposed
  cost is small: removing CMAP and its copies alone gives no measurable change; the ceiling of a CMAP GPU
  port incl. CUDA graphs is +0.9% (CI 0.3-1.5%) at the 325 W limit. The clock drops ~10 MHz when the
  GPU idle gaps disappear (power-capped). Consistent with the plain-step anatomy (copies hidden under NB/PME).

## 19:53 GPU performance counters blocked (DCGM)
- `ncu --set basic` on a trivial kernel (raw/ncu/ncu-permission-test.txt): "==ERROR== Profiling failed
  because a driver resource was unavailable. Ensure that no other tool (like DCGM) is concurrently
  collecting profiling data." nsys `--gpu-metrics-devices=help`: "NVIDIA L40S ... Already under profiling".
- Cause: nv-hostengine (DCGM 4.7.0, `/usr/bin/nv-hostengine -n --service-account nvidia-dcgm`) holds the
  profiling counters (dcgmi profile -l lists sm_active/sm_occupancy fields; likely watched by
  nebius-observability-agent). Not a permission problem: perf_event_paranoid=0, RmProfilingAdminOnly=0.
- Not pre-authorised (affects node monitoring), so not done. Admin commands to unblock, for the report:
    dcgmi profile --pause        # pause DCGM profiling-metric collection
    ... ncu / nsys --gpu-metrics-devices=0 ...
    dcgmi profile --resume
  (heavier alternative: sudo systemctl stop nvidia-dcgm; ...; sudo systemctl start nvidia-dcgm)
- Substitute evidence planned: (a) kernel-duration scaling with locked SM clock (memory clock fixed at
  9001 MHz) from nsys captures -> SM-clock-bound vs memory-bound per kernel; (b) static resource usage
  (registers, shared memory, block size -> theoretical occupancy and its limiter) from nsys kernel
  records and cuobjdump; (c) static SASS instruction mix of the NB kernels.

## 19:58 CPU profile (perf) of prod-P, attached after the counter reset (raw/perf/prod-P-fp: -F 4999 -g, 8 s,
  643,681 samples; raw/perf/prod-P: --call-graph dwarf, 12 s, 6.1 GB, for call graphs). mdrun's own
  timing during the dwarf capture: 0.742 ms/step = unprofiled (perf does not perturb the GPU-bound step).
- 16 threads x 8 s x 4999 Hz = 640k: all 16 OpenMP threads are always on-CPU. 82.9% of samples are in
  libgomp (spin-waiting in barriers/idle loops; GOMP default spin policy) -> CPU is idle-spinning.
- Real work (share of all samples): nbnxn_make_pairlist_part<GPU> 4.84% (~115 CPU-ms per NS step,
  vs 8.57 ms wall x 16 threads = 137 -> ~84% parallel efficiency); CMAP (cmap_dihs 1.19% +
  accumulateCmapForces 0.36% + dih_angle 0.33% + libm atan/asin/acos ~1%); clearRVecs 0.90%;
  reduceThreadForceBuffers 0.51%; sortColumnsGpuGeometry 0.37%; sort_atoms 0.20%.
- ListedForcesGpu::Impl::updateInteractionListsAndDeviceBuffers 0.08% = 515 samples = 0.103 CPU-s / 54 NS
  steps = 1.9 ms per NS step on ONE thread -> serial; matches NVTX "GPU Bonded list update" 2.05 ms.

## 20:00 Step-time budget (bench/profiling/budget.py on raw/nsys/prod-S-light; GPU step periods)
- mean step 742.9 us = plain steps 87.0% of time (mean 659.8 us) + pair-search steps 10.7% of time
  (15.95 ms each; excess over plain 76.5 us/step = **10.3% of run time**) + energy steps 1.4% of time
  (2.07 ms each; excess 7.1 us/step = **0.95%**); after-energy steps -0.05%.
- Inside a search step (means): Neighbor search 10.33 ms = NS search local 8.57 + GPU Bonded list update
  2.05 (serial) + NS grid local 1.67 (ranges nest under "Neighbor search"); GPU: kernels 1.30 ms,
  copy-only 1.98 ms, idle 12.67 ms.
- Inside an energy step: CPU Wait GPU NB local 588 us (VF kernel), Launch PP GPU ops 492 us,
  Launch GPU update 351 us, Kinetic energy 75 us; GPU copy-only 639 us (COM removal round trip of x,v),
  idle 478 us.
- Plain step: GPU period 659.8, kernel busy 648.4, copy-only 4.8, idle 6.5 us -> GPU-bound; CPU spends
  455 us/step in "Wait GPU state copy".

## 20:01 Theoretical occupancy from launch records (bench/profiling/occupancy.py, sm_89 rules)
- NB F/VF kernel: block 64 (8x8), 61/64 regs/thread -> 16 blocks/SM, 66.7% theoretical occupancy,
  limiter = registers (16 x 2 warps x 2048 regs fill the 64K register file -> no other kernel can
  co-reside on an SM fully loaded with NB blocks); grid ~11,200 blocks = 4.9 waves on 142 SMs.
- PME spread (256 thr, 40 regs) 3.4 waves; gather 3.4 waves; cuFFT regular_fft 438 blocks = 0.39 waves,
  r2c/c2r 0.58 waves (low-parallelism kernels); bonded 2.5 waves; x->nbat 0.93 waves.
- Update chain is sub-wave and latency-bound: LINCS 150 blocks (0.18 waves), SETTLE 156 blocks
  (0.27 waves, 61 regs -> 66.7%), leap-frog 0.85 waves.
- Stream priorities (gpu_utils/device_stream_manager.cpp:97-118): PME stream High, NB local Normal,
  update High -> at each step start the spread kernel takes the SMs first; x->nbat starts only at
  ~63 us (duration 49 us in the step vs 27 us median when alone) and bonded+x->nbat delay the NB kernel
  start to ~179 us into the step.

## 20:04 THROWAWAY experiment queued: NB kernel launch bounds (not in src/)
- nbnxm_cuda_kernel.cuh:132-133 uses `__launch_bounds__(64, MIN_BLOCKS_PER_MP=16)` for all CC >= 5.0
  (comment: "16 blocks/multiproc ... fastest even though this setup gives low occupancy"). On sm_89
  (48 warps, 24 blocks per SM) that caps occupancy at 66.7% with <= 64 regs (kernel uses 61).
- Worktree /home/asokolov/Projects/gromacs-exp-nbminblocks (detached at dev): MIN_BLOCKS_PER_MP made a
  compile-time knob (-DGMX_EXP_NB_MIN_BLOCKS_PER_MP); profiles cuda-nosub-mb20 / -mb24 in
  bench/profiling/experiments/exp-nbminblocks-suite.toml (builds 16975953e72a9b65, f5b973e5f3b87e02).
  Queued at the end of queue2 (builds run alone so they do not perturb measurements), then
  exp-nbminblocks.toml (P vs mb20 vs mb24, 5 rounds) + nsys light captures. Results reported separately;
  the gmxbench quality suite would have to pass before mixing them into the main comparison.

## 20:12 More timeline detail (prod-S-light)
- Rolling prune interacts with the update chain: LINCS alternates 13 us / 30 us step by step
  (p10 13.1, p90 32.3); the rolling prune kernel (1400 blocks x 256 thr, ~40 us) runs every 2nd step
  concurrently with leap-frog/LINCS/SETTLE (it is launched on the NB stream right after the reduction),
  so ~17 us every other step (~8.5 us/step, ~1.2%) of pruning lands on the update critical path.
- x->nbat duration is bimodal (25 us in 54% of steps, ~86 us in 46%): it starts together with the PME
  spread (high-priority stream, 3.4 waves) and gets SMs only as spread blocks retire.
- CUDA API per step type (main thread): plain: 2 cudaEventSynchronize 454 us, 10.5 launches 49 us,
  3 cudaMemcpyAsync 13 us. energy: + 4 cudaStreamSynchronize 623 us, 16 cudaMemcpyAsync. NS+energy:
  6 cudaStreamSynchronize 1246 us, 33 cudaMemcpyAsync 183 us.
- NS step order on the CPU: grid 246->1916 us, GPU bonded list update 1976->4028 us (single thread;
  its H2D uploads of ~6.6 MB happen 2.0-4.1 ms), search 4035->12607 us, then x is re-uploaded (H2D
  2.2 MB at 13.0 ms, sim_util.cpp:1700 copies x at search steps) and PME/NB are launched; PME of the
  search step could in principle overlap the CPU search (bound ~0.33 ms per search = 0.2%).

## 20:11 H3 power experiment (raw/exp/power; P build, prod; post-hook restored -rgc / -pl 325 after each run)
| variant | ms/step [95% CI] | ns/day | vs pl325 | P mean / p95 / max W | SM MHz mean (p05) | sw_power_cap frac | kJ/ns |
|---|---|---|---|---|---|---|---|
| pl325 (production) | 0.7415 [0.7394, 0.7436] | 233.05 | 1 | 306.8 / 321.6 / 323.5 | 2362 (2310) | 1.00 | 112.80 |
| pl350 (requested; enforced stays 325) | 0.7406 [0.7389, 0.7422] | 233.34 | 1.0012 [0.9982, 1.0043] | 307.3 / 322.1 / 324.0 | 2363 (2310) | 1.00 | 113.05 |
| pl300 | 0.7623 [0.7496, 0.7753] | 226.69 | 0.9727 [0.9565, 0.9891] | 295.0 / 311.6 / 312.6 | 2284 (2220) | 1.00 | 111.10 |
| pl250 | 0.8382 [0.8360, 0.8404] | 206.16 | 0.8846 [0.8818, 0.8875] | 247.8 / 257.7 / 260.8 | 1938 (1685) | 1.00 | 103.99 |
| lock 2250 MHz @325 | 0.7586 [0.7576, 0.7596] | 227.80 | 0.9775 [0.9748, 0.9802] | 290.6 / 304.2 / 305.8 | 2250 | 0.00 | 110.08 |
| lock 1980 MHz @325 | 0.8418 [0.8401, 0.8435] | 205.28 | 0.8809 [0.8783, 0.8834] | 247.4 / 257.3 / 259.5 | 1980 | 0.00 | 103.76 |
| nocmap+graph @325 (timing only) | 0.7332 [0.7319, 0.7345] | 235.69 | 1.0113 [1.0084, 1.0142] | 306.7 / 319.5 / 322.0 | 2351 (2310) | 1.00 | 111.96 |
| nocmap+graph @350 req. (timing only) | 0.7336 [0.7319, 0.7353] | 235.56 | 1.0108 [1.0077, 1.0138] | 306.7 / 320.7 / 323.4 | 2353 (2310) | 1.00 | 112.12 |
- **Enforced power limit is 325 W regardless of the requested limit**: `sudo nvidia-smi -pl 350` ->
  power.limit 350 W but enforced.power.limit 325 W (checked 20:10:37, ~1 s during a `threads` run;
  restored to 325 immediately). So the GPU cannot be run above 325 W on this node; pl350 = pl325.
- H3 confirmed and quantified: sw_power_cap is active in 100% of samples at 325 W; mean SM clock 2362
  MHz = 6.3% below the 2520 MHz maximum (p05 2310); power mean 307 W, 1 s-averaged max 323.5 W.
- Clock sensitivity from the clock-locked runs (no throttling): T(f) = a + b/f with T(2250) = 0.7586 and
  T(1980) = 0.8418 ms -> b = 1373 ms*MHz, a = 0.148 ms: ~80% of the step time scales with the SM clock
  at 2250 MHz. Extrapolated to an uncapped 2520 MHz: 0.693 ms/step (-6.5% vs production) = the cost of
  the power cap (upper bound; not reachable on this node). The model predicts 0.730 ms at the mean
  capped clock 2362 vs 0.7415 measured: the effective clock during the power-heavy kernels is lower than
  the time-averaged clock (consistent with a power controller that lowers clocks in NB-heavy phases).
- Energy: locking at 2250 MHz costs 2.3% speed and saves 2.4% GPU energy per ns; 1980 MHz costs 11.9%
  and saves 8.0%; the 250 W cap is equivalent to the 1980 MHz lock.

## 20:16 Static kernel analysis (P build; raw/sass/)
- cuobjdump --dump-resource-usage (raw/sass/resource-usage-P.txt): NB F kernels: ElecEw_VdwLJFsw 61 regs,
  ElecEw_VdwLJ (potential-shift) 59, ElecRF_VdwLJFsw 61, ElecEwQSTab_VdwLJFsw 63, ElecEwTwinCut 63;
  no spills (LOCAL=0, STACK=0) -> all at 66.7% theoretical occupancy under (64, 16) launch bounds.
- Static SASS of ElecEw_VdwLJFsw_F (1928 instr) vs ElecEw_VdwLJ_F (1696): force-switch adds 232 static
  instructions (+13.7%; FFMA +80, FMUL +96, FSEL/FSETP +32, MOV +32). Static counts only (no dynamic mix
  without counters). SFU: 32 MUFU (RSQ, RCP); 11 RED.E.ADD.F32 (global force atomics); 55 SHFL; 54
  CALL.REL to 2 tiny SHFL.DOWN/UP subroutines (compiler's non-convergent shuffle path) in the
  force-reduction epilogue, executed once per i-cluster, not per pair.

## 20:26 H4 thread sweep (raw/exp/threads; P build, prod, -pin on with mdrun's auto stride: <=20 threads
  one per core, 24/32 threads use both hyperthreads of 12/16 cores)
| -ntomp | ms/step [95% CI] | ns/day | vs t16 | NS ms per search (md.log) | CPU Force us/step | kJ/ns |
|---|---|---|---|---|---|---|
| 4 | 0.8349 [0.8328, 0.8370] | 206.99 | 0.8881 [0.8851, 0.8912] | 28.69 | 127.0 | 117.80 |
| 8 | 0.7725 [0.7702, 0.7749] | 223.70 | 0.9598 [0.9563, 0.9634] | 16.34 | 77.3 | 115.02 |
| 12 | 0.7512 [0.7503, 0.7522] | 230.03 | 0.9870 [0.9838, 0.9902] | 11.98 | 64.1 | 113.71 |
| 16 (production) | 0.7415 [0.7390, 0.7439] | 233.06 | 1 | 10.07 | 57.6 | 113.52 |
| 20 | 0.7372 [0.7342, 0.7401] | 234.42 | 1.0058 [1.0015, 1.0102] | 8.89 | 54.3 | 113.20 |
| 24 (HT) | 0.7465 [0.7444, 0.7485] | 231.50 | 0.9933 [0.9898, 0.9969] | 10.44 | 65.4 | 113.50 |
| 32 (HT) | 0.7400 [0.7377, 0.7424] | 233.51 | 1.0020 [0.9982, 1.0058] | 9.10 | 59.8 | 113.19 |
- Thread count only matters through the CPU-bound search steps: NS cost per search 28.7 -> 10.1 -> 8.9 ms
  for 4 -> 16 -> 20 threads (4->16: 2.84x for 4x threads); hyperthreads do not help (24/32 slower than 20).
- -ntomp 20 is the best but by +0.6% (below the 1% practical threshold; significant). Search steps
  dominate the remaining CPU sensitivity: 1.2 ms less per search = ~6 us/step.

## 20:38 H4 nstlist sweep (raw/exp/nstlist; P build, t16; inner list always 1.203 nm / 16 steps)
| -nstlist | outer rlist | ms/step [95% CI] | ns/day | vs 200 | NS ms per search | kJ/ns |
|---|---|---|---|---|---|---|
| 80 | 1.372 | 0.7848 [0.7831, 0.7865] | 220.19 | 0.9451 [0.9427, 0.9474] | 7.93 | 113.86 |
| 100 | 1.420 | 0.7659 [0.7630, 0.7688] | 225.62 | 0.9684 [0.9647, 0.9721] | 8.31 | 113.18 |
| 150 | 1.564 | 0.7451 [0.7433, 0.7470] | 231.93 | 0.9954 [0.9927, 0.9981] | 8.89 | 112.49 |
| 200 (prod) | 1.742 | 0.7417 [0.7401, 0.7433] | 232.99 | 1 | 10.21 | 113.06 |
| 300 | 2.162 | 0.7602 [0.7580, 0.7624] | 227.33 | 0.9757 [0.9728, 0.9786] | 13.65 | 117.22 |
| 400 | 2.634 | 0.7999 [0.7972, 0.8025] | 216.04 | 0.9273 [0.9242, 0.9304] | 18.21 | 124.85 |
- 200 is the optimum (150 within 0.5%); search cost per search has a large fixed part (7.9 ms at
  rlist 1.372 vs 10.2 ms at 1.742 for 2x the pair volume), so fewer searches with a bigger list wins until
  the outer-list prune/search cost explodes (>= 300). Energy per ns rises with nstlist >= 300.

## 20:34 FINDING: OpenMP pragmas silently ignored in nvcc-compiled host code
- The hot loop of `ListedForcesGpu::Impl::updateInteractionListsAndDeviceBuffers` (perf annotate on
  raw/perf/prod-P-fp: the index gather `dest[i+1+a] = nbnxnAtomOrder[src[...]]` of
  `convertIlistToNbnxnOrder`, which has `#pragma omp parallel for`) runs on one thread.
- Cause: listed_forces_gpu_impl_gpu.cpp is compiled by nvcc (`-x cu`, build.make line 9979) with
  CUDA_FLAGS that contain no -fopenmp (CXX_FLAGS do); the object has no GOMP_parallel call and no
  _omp_fn symbols (objdump/nm). Same for src/gromacs/mdlib/lincs_gpu.cpp (4 omp pragmas).
  Of 52 files compiled with `-x cu`, these two contain OpenMP pragmas.
- Cost: 2.05 ms per search step on the critical path (GPU idle) = 10.3 us/step = 1.4% of run time.

## 20:52 Ablation ladder (raw/exp/ablation; TIMING ONLY inputs; P build, production args)
| variant | ms/step [95% CI] | ns/day | vs prod | kJ/ns | outer rlist / prune interval |
|---|---|---|---|---|---|
| prod | 0.7414 [0.7398, 0.7430] | 233.09 | 1 | 113.03 | 1.742 / 16 |
| nocmap | 0.7420 [0.7399, 0.7440] | 232.91 | 0.9992 [0.9963, 1.0022] | 113.10 | 1.742 / 16 |
| potshift | 0.6934 [0.6901, 0.6968] | 249.21 | **1.0691 [1.0640, 1.0743]** | 104.74 | 1.642 / 10 |
| nocmap-potshift | 0.6926 [0.6911, 0.6941] | 249.51 | 1.0705 [1.0678, 1.0732] | 104.14 | 1.642 / 10 |
| nolincs | 1.0118 [1.0062, 1.0175] | 170.79 | 0.7327 [0.7287, 0.7368] | 147.76 | **3.284** / 14 |
| noconstr | 1.1601 [1.1555, 1.1647] | 148.96 | 0.6391 [0.6366, 0.6416] | 170.56 | **3.459** / 10 |
| rf | failed: `-pme gpu` with no PME ("PME GPU does not support systems that do not use PME") -> re-queued as raw/exp/ablation-rf without -pme |
- DEAD END: constraint ablations are dominated by the Verlet-buffer estimate: with flexible H bonds the
  outer rlist at nstlist 200 grows to 3.28-3.46 nm, so the runs measure pair-list growth, not LINCS/SETTLE
  cost. LINCS/SETTLE/leap-frog cost is taken from nsys kernel timings instead.
- Force-switch -> potential-shift: 48 us/step (6.5% of step time), energy -7.3% per ns. Confound: the
  buffer estimate also changes (outer 1.642 vs 1.742 nm, prune every 10 vs 16 steps); inner list
  1.201 vs 1.203 nm (NB kernel works on the same pairs). Kernel-only split from raw/nsys/potshift-S-light.
- 20:55 raw/nsys/potshift-S-light vs prod-S-light: NB F kernel p50 406.4 vs 446.2 us (-39.8 us = -8.9% of
  the NB kernel = 5.4% of the mean step; static SASS +13.7% instructions for force-switch -> consistent with
  an instruction-issue-bound kernel); VF 522.5 vs 633.1 us; plain-step GPU period p50 631.3 vs 661.4 us;
  search step 14.69 vs 15.68 ms (smaller outer list); rolling prune 57.3 vs 39.8 us (every 10 vs 16 steps
  -> 2x per 2 steps). Kernel-only force-switch cost = ~40 us/step; the remaining ~8 us of the ablation
  gain is pair-list related. CHARMM36 requires force-switch, so this is an upper bound for making the
  switch cheaper, not an option to drop it.

## 21:00 Reaction-field ablation (raw/exp/ablation-rf, TIMING ONLY; run without -pme gpu)
| variant | ms/step [95% CI] | ns/day | vs prod | SM MHz | kJ/ns | outer rlist / prune interval |
|---|---|---|---|---|---|---|
| prod | 0.7419 [0.7403, 0.7436] | 232.92 | 1 | 2363 | 112.90 | 1.742 / 16 |
| rf (eps_rf = inf) | 0.5981 [0.5957, 0.6006] | 288.91 | **1.2404 [1.2354, 1.2454]** | 2271 | 91.75 | 1.993 / 4 |
- raw/nsys/rf-S-light: plain-step GPU period p50 504 us (prod 661); NB F kernel ElecRF 346.8 us vs ElecEw
  446.2 us (-99 us = Ewald real-space cost, -22% of the NB kernel); no PME kernels; **x->nbat 4.2 us**
  (prod: 25-86 us, i.e. its duration in production is waiting for SMs held by the PME spread); rolling
  prune 146.7 us every 2nd step (inner list every 4 steps); bonded 53 us; search step 17.9 ms (bigger
  outer list). SM clock drops 92 MHz (busier GPU under the 325 W cap).
- So "everything PME" (mesh + Ewald part of the NB kernel + SM contention) bounds at ~157 us per plain
  step / 24% of run time (time) and 18.7% of GPU energy per ns. Physics requires PME; this is the
  envelope for PME-related optimisations (scheduling, FFT, grid/rc balance), not an option.

## 21:14 Kernel-variant knobs (raw/exp/knobs; P build, production args; env vars verified in src/)
| variant (env) | ms/step [95% CI] | ns/day | vs default | kJ/ns |
|---|---|---|---|---|
| default | 0.7450 [0.7383, 0.7517] | 231.96 | 1 | 113.48 |
| GMX_DISABLE_ALTERNATING_GPU_WAIT | 0.7426 [0.7409, 0.7442] | 232.72 | 1.0032 [0.9944, 1.0122] | 113.19 |
| GMX_NSTLIST_DYNAMICPRUNING=8 (inner rlist 1.200) | 0.7463 [0.7452, 0.7475] | 231.54 | 0.9982 [0.9893, 1.0071] | 114.75 |
| GMX_GPU_NB_EWALD_TWINCUT | 0.7498 [0.7487, 0.7509] | 230.47 | 0.9936 [0.9847, 1.0025] | 114.79 |
| GMX_NB_MIN_CI=0 (no list splitting) | 0.7514 [0.7484, 0.7544] | 229.97 | 0.9914 [0.9827, 1.0003] | 114.20 |
| GMX_NSTLIST_DYNAMICPRUNING=32 (inner rlist 1.246) | 0.7528 [0.7520, 0.7536] | 229.55 | 0.9896 [0.9808, 0.9985] slower | 114.81 |
| GMX_GPU_NB_TAB_EWALD | 0.7564 [0.7555, 0.7573] | 228.46 | 0.9849 [0.9761, 0.9938] slower | 115.13 |
| GMX_DISABLE_DYNAMICPRUNING | 0.9663 [0.9647, 0.9679] | 178.83 | 0.7709 [0.7641, 0.7779] slower | 150.52 |
- The in-experiment default has one slow run (rep 4: 0.753 ms, search 59 vs 50 us/step amortised: CPU
  search jitter), widening every CI; the production config measured 0.7406-0.7425 in 8 other groups.
  Against those, twin-cut (0.7498), MIN_CI=0 (0.7514) and prune8 (0.7463) are also slower.
- Conclusion: no kernel-variant knob beats the defaults (analytical Ewald, single cut-off, list splitting,
  prune interval 16 chosen by mdrun). Dynamic pruning is worth 23% here.
- 21:17 Plain-step critical-path decomposition (prod-S-light anatomy, medians; GPU period 661 us):
  SETTLE(k-1) end -> NB start = 165 us (PME spread 88 us at High priority first, x->nbat (4 us of work)
  waiting for SMs, bonded 60 us serial in the NB stream, launch gaps); NB F kernel 445 us (67%);
  NB end -> SETTLE(k) end = 51 us (reduce 5, leap-frog/LINCS/SETTLE 45.6); 11.6 us gap before the next
  spread starts.

## 21:28 nstcalcenergy / global-communication intervals (raw/exp/nstcalc; P build, production args)
| tpr | ms/step [95% CI] | ns/day | vs prod | kJ/ns |
|---|---|---|---|---|
| prod (nstcalcenergy=nstcomm=nsttcouple=nstpcouple=100) | 0.7398 [0.7370, 0.7426] | 233.60 | 1 | 112.85 |
| nstcalc500 (only nstcalcenergy) | 0.7406 [0.7380, 0.7433] | 233.33 | 0.9988 [0.9946, 1.0031] | 112.49 |
| nstcalc1000 (only nstcalcenergy) | 0.7389 [0.7381, 0.7398] | 233.86 | 1.0011 [0.9975, 1.0048] | 113.08 |
| nstglob500 (all four = 500; timing only) | 0.7336 [0.7314, 0.7358] | 235.56 | 1.0084 [1.0044, 1.0124] | 112.45 |
| nstglob1000 (all four = 1000; timing only) | 0.7312 [0.7288, 0.7337] | 236.33 | **1.0117 [1.0075, 1.0159]** | 111.95 |
| perfout (nstlog 0, no xtc) | 0.7408 [0.7390, 0.7427] | 233.26 | 0.9986 [0.9947, 1.0024] | 113.44 |
- nstcalcenergy alone does nothing: T/P coupling (100) and COM removal (100) still need global steps.
  Raising all four together removes most of the energy/COM-step excess (+1.2% at 1000, ~= the 0.95%
  measured by nsys + the VF-kernel part). grompp warns that tau_p=5 ps < 25*nstpcouple*dt at 500/1000:
  a protocol decision, not a free knob.
- Log/energy output every 1000 steps costs nothing measurable (perfout 0.9986 [0.9947, 1.0024]); the 0.6%
  higher gmxbench numbers are between-session/run-length differences, not output cost.

## 21:40 Throwaway worktree extended (all behind compile-time defines; default path unchanged)
- (2) -DGMX_EXP_SWAP_STREAM_PRIO: NB local stream High, PME stream Normal (device_stream_manager.cpp).
- (3) build with -DCMAKE_CUDA_FLAGS=-Xcompiler=-fopenmp (OpenMP pragmas in nvcc-compiled host code active).
- (4) -DGMX_EXP_BONDED_STREAM: GPU bonded kernel in its own stream (waits on an event recorded in the NB
  local stream at launch, i.e. after x->nbat / force clearing / list uploads; the NB local stream waits
  for it right after the local NB kernel launch, before reduction, energy D2H and prune). Files:
  listed_forces_gpu.h, listed_forces_gpu_impl.h, listed_forces_gpu_impl_gpu.cpp,
  listed_forces_gpu_internal.cu, sim_util.cpp. Builds: prioswap cbb50c98ae37ae0d, ompcuda 3136061f41d389dc,
  bondedstream 44e346d549ee6c7d. Measured in raw/exp/exp-nbminblocks together with mb20/mb24.
- PME experiment had failed at 21:24 (TOML keys "rc1.20" parse as nested tables); fixed (rc120...) and
  re-queued after pinning. prun.py now records a missing binary as an error instead of crashing.

## 21:38 Pinning (raw/exp/pinning; P build; VM: 1 socket / 1 NUMA node, so no local-vs-remote socket test)
| variant | ms/step [95% CI] | ns/day | vs auto | NS ms/search |
|---|---|---|---|---|
| -pin on, auto stride 2 (cores 0-15, production) | 0.7420 [0.7397, 0.7444] | 232.88 | 1 | 10.03 |
| -pin off | 0.7415 [0.7405, 0.7425] | 233.06 | 1.0008 [0.9976, 1.0040] | 10.13 |
| -pinstride 2 -pinoffset 8 (cores 4-19) | 0.7434 [0.7403, 0.7465] | 232.46 | 0.9982 [0.9938, 1.0027] | 10.31 |
| -pinstride 1 (16 threads on 8 cores' HT pairs) | 0.7620 [0.7590, 0.7651] | 226.77 | 0.9738 [0.9695, 0.9780] | 13.46 |
| -ntomp 32 -pinstride 1 (16 cores x 2 HT) | 0.7401 [0.7369, 0.7432] | 233.50 | 1.0027 [0.9982, 1.0072] | 8.81 |
- Placement does not matter on this VM as long as each thread has its own core; sharing cores
  slows the search (13.5 vs 10.0 ms) and costs 2.6%. Consistent with the PCIe probe (no vCPU dependence).
- 21:40 P vs S under nsys light (raw/nsys/prod-P-light vs prod-S-light): GPU span 749.6 vs 750.4 us/step;
  kernel busy 86.98 vs 87.03%, idle 10.42% both; NB F kernel p50 446.8 vs 446.2 us; all large kernels within
  +-2% -> the NVTX/sub-counter build has the same GPU timeline; S-build step analyses apply to P.

## 21:51 PME fixed points vs tuner (raw/exp/pme; P build; -notunepme except "tunepme")
| variant (rcoulomb / grid) | ms/step [95% CI] | ns/day | vs rc 1.20 | kJ/ns |
|---|---|---|---|---|
| 1.20 / 96x96x144 (production) | 0.7408 [0.7402, 0.7414] | 233.27 | 1 | 112.90 |
| tuner on (steady state, reset at 25000) | 0.7444 [0.7426, 0.7461] | 232.16 | 0.9953 [0.9929, 0.9976] | 113.33 |
| 1.25 / fourierspacing 0.125 | 0.7869 [0.7846, 0.7893] | 219.59 | 0.9414 [0.9386, 0.9442] | 121.22 |
| 1.30 / 84x84x128 | 0.8266 [0.8240, 0.8293] | 209.06 | 0.8962 [0.8933, 0.8991] | 128.11 |
| 1.35 / fs 0.135 | 0.8681 [0.8664, 0.8698] | 199.07 | 0.8534 [0.8518, 0.8550] | 135.84 |
| 1.40 / 80x80x120 | 0.9203 [0.9168, 0.9237] | 187.78 | 0.8050 [0.8020, 0.8080] | 144.40 |
- The tuner chose 96x96x144 / rc 1.200 in all 5 runs (and in plain mdrun). Moving work from PME (off
  the critical path) into the NB kernel (on it) only hurts; rc < rvdw = 1.2 is not allowed with Verlet
  lists, so the default is the optimum. Leaving -tunepme on costs 0.47% in steady state (negligible).

## 22:00 Clock scaling (raw/nsys/clk*-S-light, bench/profiling/clockscale.py -> raw/nsys/clockscale.json)
- Locked SM clock 1500 / 2010 / "2520" MHz with memory clock fixed at 9001 MHz; the 2520 request ran at a
  median 2400 MHz (enforced 325 W cap). Power limit request 350 W during the lock, restored to 325 W and
  unlocked afterwards (queue log: 325.00 W / 2520 MHz application clock; enforced 325 W).
- Exponent alpha in duration ~ f^-alpha: NB F kernel 0.90; PME solve 0.98, c2r 0.92, regular_fft 0.86,
  gather 0.82, rolling prune 0.85; spread 0.64, bonded 0.66, r2c 0.61, LINCS 0.73, SETTLE 0.68, leap-frog
  0.66, reduce 0.73, memset 0.61; copies D2H 0.06, H2D 0.00 (PCIe); plain step 0.85, mean step 0.75.
  -> the NB kernel and most PME kernels scale with the graphics clock (not DRAM-bound); spread, bonded and
  the update kernels are partly limited by something outside the SM clock (DRAM or atomic latency).
  Without counters this cannot be split further (issue vs L1/shared vs L2).

## 22:00 Contrast modes (raw/nsys/contrast-*-S-light; S build, 16 threads)
| mode | plain-step GPU period p50 | GPU idle per plain step | NB F kernel p50 |
|---|---|---|---|
| production (GPU-resident, bonded GPU) | 661 us | 6.5 us | 446 us |
| GPU-resident, bondeds on CPU | 1032 us | 345 us | 453 us |
| NB+PME on GPU, update on CPU (nstlist 100) | 1213 us | 416 us | 431 us |
| NB only on GPU (nstlist 100) | 5245 us | 4625 us | 393 us |
- Every mode other than full GPU residency is CPU-bound (the GPU idles 30-88% of a plain step).
- NB kernel alone (NB-only mode, no concurrent PME) 393 us vs 446 us in production: concurrent PME work
  slows the NB kernel by ~12% (SM sharing), consistent with PME being off the critical path but not free.

## 22:20 THROWAWAY experiment results (raw/exp/exp-nbminblocks; NOT src/ changes; worktree
  /home/asokolov/Projects/gromacs-exp-nbminblocks; production args; 5 interleaved rounds)
| variant (build) | ms/step [95% CI] | ns/day | vs upstream P | SM MHz | kJ/ns |
|---|---|---|---|---|---|
| upstream (P) | 0.7409 [0.7384, 0.7434] | 233.23 | 1 | 2362 | 112.79 |
| bonded kernel in own stream (44e346d549ee6c7d) | 0.7343 [0.7298, 0.7388] | 235.35 | **1.0091 [1.0029, 1.0153]** | 2355 | 112.25 |
| OpenMP for nvcc host code (3136061f41d389dc) | 0.7347 [0.7314, 0.7380] | 235.22 | **1.0085 [1.0038, 1.0132]** | 2364 | 112.58 |
| stream priorities swapped (cbb50c98ae37ae0d) | 0.7452 [0.7418, 0.7486] | 231.91 | 0.9943 [0.9895, 0.9991] | 2361 | 112.95 |
| NB launch bounds 20 blocks/SM (16975953e72a9b65) | 0.7721 [0.7683, 0.7759] | 223.81 | 0.9596 [0.9547, 0.9645] | 2322 | 117.44 |
| NB launch bounds 24 blocks/SM (f5b973e5f3b87e02) | 0.8864 [0.8850, 0.8879] | 194.95 | 0.8358 [0.8331, 0.8386] | 2110 | 135.88 |
- nsys light of each (raw/nsys/exp-nbminblocks-<build>-light; P-type builds, no NVTX):
  NB F kernel p50: upstream 446.8, bondedstream 459.7 (shares SMs with bonded, which now runs on stream 24),
  ompcuda 446.4, prioswap 446.9, mb20 473.8 (48 regs + 8 B stack), mb24 574.8 (40 regs + 40 B stack).
  GPU idle: upstream 78.1, ompcuda 69.5 (-8.6 us/step vs the 10.3 us bound), bondedstream 75.3 us/step;
  GPU span/step: 749.6 / 742.2 / 739.2 us.
- Launch bounds REFUTED: higher occupancy makes the NB kernel slower even at equal work (spills, and under
  the power cap the clock drops to 2322 / 2110 MHz because more resident warps draw more power).
  The register-limited 66.7% occupancy is not this kernel's limiter.
- Priority swap REFUTED (-0.6%): giving the NB chain the SMs first delays the PME chain onto the critical path.
- `gmxbench quality --strict --target-only --tier quick` (raw/gmxbench/quality-ompcuda, quality-bondedstream):
  both rc=0; CPU configs IDENTICAL, GPU configs EQUIVALENT with force rel-RMS 8.5e-8..1.01e-7 vs the
  baseline's own noise 8.4e-8..1.0e-7; grompp identical. Both may enter the main comparison.
- 22:28 combined build (combo, 2f8c76e3571459cd = OpenMP + bonded stream) built; quality + exp-combo running.

## 22:44 Combined throwaway build (raw/exp/exp-combo; combo = OpenMP for nvcc host code + bonded stream,
  2f8c76e3571459cd; `quality --strict` passed: raw/gmxbench/quality-combo)
| variant | ms/step [95% CI] | ns/day | vs upstream P | kJ/ns |
|---|---|---|---|---|
| upstream P (t16) | 0.7436 [0.7391, 0.7481] | 232.40 | 1 | 112.95 |
| upstream P -ntomp 20 | 0.7355 [0.7320, 0.7391] | 234.94 | 1.0110 [1.0044, 1.0175] | 112.77 |
| ompcuda | 0.7357 [0.7346, 0.7369] | 234.89 | 1.0107 [1.0047, 1.0168] | 112.85 |
| bondedstream | 0.7339 [0.7319, 0.7360] | 235.45 | 1.0131 [1.0071, 1.0192] | 112.39 |
| combo | 0.7288 [0.7264, 0.7312] | 237.11 | **1.0203 [1.0142, 1.0265]** | 112.19 |
| combo -ntomp 20 | 0.7232 [0.7204, 0.7260] | 238.95 | **1.0282 [1.0219, 1.0345]** | 111.83 |
- Additive: 1.07% + 1.31% ~ 2.03%; + t20 1.1% ~ 2.8%. (This experiment's upstream reference ran 0.4% slower
  than the session mean 0.7417; vs 0.7417 the combo-t20 gain is +2.6%.)

## 22:50 Deliverables written
- ANALYSIS.md, OPPORTUNITIES.md, EXEC-SUMMARY.md, PLAN.md (with outcome), summary.json / summary.csv
  (1585 records: bench/profiling/records.py + derived_records.py + summarize.py collect).
- Scripts and experiment definitions committed on feature/profiling (bench/profiling/); results directory is
  git-ignored (it contains tprs of the customer system under raw/inputs; do not commit raw/).
- GPU state at the end: power limit 325 W (enforced 325 W), application clocks unlocked (2520 MHz).
- Open item: Nsight Compute block blocked by DCGM (see 19:53).

## 22:53 Decision: no Nsight Compute
- Asked whether to pause DCGM profiling (~45 min) for the ncu block; the user chose "Skip ncu, finish now".
  DCGM was not touched. Kernel-level statements stay counter-free (clock scaling, launch records, static
  SASS, ablations, throwaway builds); EXEC-SUMMARY.md lists the admin commands for a later run.
- Throwaway worktree /home/asokolov/Projects/gromacs-exp-nbminblocks and its builds are left in place for
  reproduction (patch: bench/profiling/experiments/exp-throwaway.patch); remove with
  `git worktree remove /home/asokolov/Projects/gromacs-exp-nbminblocks`.
