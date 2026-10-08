# MAS1 on one L40S: where the GPU time goes inside the kernels, and what rewriting them can gain

Continuation of `bench/results/profiling-2026-10-06/` ("day 1"; same node, system, inputs and production
command `gmx mdrun -ntmpi 1 -ntomp 16 -nb gpu -pme gpu -bonded gpu -update gpu -nstlist 200 -pin on`,
GROMACS 2026.4 = `dev`). Day 1 showed that plain steps (87% of run time) are GPU-bound with the critical
path pre-NB work (165 us) -> NB force kernel (445 us) -> reduction + update chain (51 us). This report
opens the GPU work itself. Artifact paths are relative to this directory; every number is also a record
in `summary.json` / `summary.csv`. Lab notebook: `NOTES.md`.

## 0. Method and tools

| layer | tool | what it gives | overhead / caveat |
|---|---|---|---|
| kernel durations, alone and concurrent | nsys (CUDA trace) with the SM clock locked at 2100 MHz (below the power-cap clock, so it holds: 58/59 telemetry samples at 2100 MHz); `CUDA_LAUNCH_BLOCKING=1` for "alone" | `raw/nsys/iso-2100-P`, `raw/nsys/conc-2100-P` (P build, 2000 steps each) | nsys light +0.6% (day 1); launch blocking serialises kernels (used only for per-kernel durations) |
| dynamic instruction profile | **`bench/profiling/sassprof`**: CUPTI SASS-metrics injection library (binary patching) | exact executed warp/thread instructions per SASS instruction, predication, branch divergence, global sectors vs ideal, L1 tag lookups, shared wavefronts vs ideal, local sectors; per source line via `-lineinfo` | counts exact; patched kernels run slower (no timing under patching) |
| attribution to code | `sassanalyze.py` + line-info disassembly (`nvdisasm -gi`) + region map `regions-nbnxm-cuda.toml` | instruction mix, pipe model, per-region shares | the build with `-lineinfo` has **SASS identical to the production-like P build** (all 364,361 instructions, `NOTES.md` 08:45) |
| kernel rewrites, isolated | throwaway build (worktree `gromacs-exp-gpukern`), **in-process replay** (`GMX_EXP_NB_REPLAY`, `bench/profiling/nb_replay.sh`) | each NB variant launched on the live pair list/coordinates of 8 plain steps, 30 interleaved repeats each, isolated, CUDA-event timed; forces compared with the production kernel | throwaway code; isolated timing (no concurrent PME) |
| end-to-end effect | same throwaway build, variants selected by `GMX_EXP_*` env (one binary), `bench/profiling/prun` (seeded interleaved rounds, 25,000 steps, 20,000 timed) vs the P build | ms/step with 95% CI, telemetry | throwaway; gains claimed only with `gmxbench quality --strict` |
| upper bounds | timing-only kernel skips in the same build (`GMX_EXP_SKIP_*`, NB knock-outs) | what the step would cost if a kernel were free | wrong physics, timing only |

**Hardware counters are still unavailable** (DCGM, see day 1). New today: CUPTI PM sampling fails with
`CUPTI_ERROR_HARDWARE_BUSY`, CUPTI PC sampling (warp-stall reasons) with `CUPTI_ERROR_UNKNOWN`; CUPTI
**SASS metrics work** because they use binary instrumentation, not the performance monitor. So this
report has exact instruction-level counts, but **no stall reasons, achieved occupancy, cache hit rates
or DRAM throughput**. Where a statement needs those (e.g. "latency-bound"), it is an inference from IPC,
memory-efficiency counts and clock scaling, and is marked as such. Pipe "utilisation" figures are a
model (documented Ada per-SM throughputs: 4 issue, 4 FP32, 2 ALU, 0.5 MUFU, 1 LSU instruction per
cycle) applied to exact counts: lower bounds on how busy a pipe must have been, not measurements.

sassprof needed work-arounds for CUPTI 2025.3 (documented in `bench/profiling/sassprof/sassprof.cpp`):
only the first kernel launched after enabling is collected and nothing after the first flush, so each
kernel is profiled in its own mdrun, enabled at the exit of the launch that precedes it in a plain step
(counter reset at step 151, 400-step window = two pair-list lifetimes). Leap-frog could not be collected.

## 1. The GPU work of one plain step

Per-kernel durations at a locked 2100 MHz (P build), alone (`CUDA_LAUNCH_BLOCKING=1`) and with the
normal concurrency; instruction profiles from sassprof (L build = P SASS):

| kernel | launches/step | alone p50 us | concurrent p50 us | warp inst/launch | IPC/SM alone (max 4) | SIMT eff. | global sector eff. | shared wavefront eff. | dominant classes |
|---|---|---|---|---|---|---|---|---|---|
| NB force `ElecEw_VdwLJFsw_F` | 0.99 | **459.1** | 494.7 | 433 M | **3.17** | 0.69 | 0.82 | 1.00 | FP32 49%, ALU 31%, control 9% |
| PME spline+spread | 1.00 | **95.7** | 98.7 | 6.75 M | **0.24** | 0.88 | **0.42** | 1.00 | ALU 33%, IMAD 22%, FP32 19% |
| bonded | 0.99 | **59.4** | 64.1 | 4.56 M | **0.26** | 0.99 | **0.47** | - | ALU 40%, FP32 33% |
| PME gather | 1.00 | 33.1 | 72.0 | 10.2 M | 1.04 | 0.89 | 0.44 | 1.00 | FP32 39%, ALU 22%, LSU 16% |
| rolling prune (every 2nd step) | 0.49 | 33.1 | 45.0 | 20.9 M | 2.11 | 1.00 | 1.00 | 1.00 | control 33%, ALU 30% |
| LINCS | 0.99 | 14.9 | 15.8 | 0.88 M | 0.20 | 0.78 | 0.33 | 0.89 | ALU 30%, control 19%, LSU 18% |
| SETTLE | 0.99 | 13.7 | 13.7 | 0.40 M | 0.10 | 1.00 | **0.13** | - | FP32 72%, LSU 15% |
| cuFFT 96-point (x4), r2c, c2r | 4+1+1 | 6.4 / 6.2 / 6.2 | 26.1 / 19.4 / 28.6 | 0.28 / 1.63 / 1.65 M | 0.15 / 0.89 / 0.88 | 1.00 | 0.73-0.82 | 0.67-0.80 | FP32 45-65% |
| PME solve | 0.99 | 5.2 | 31.2 | 2.88 M | 1.85 | 0.99 | 0.90 | - | ALU 49% |
| leap-frog | 0.99 | 6.0 | 6.8 | (not collectable) | | | | | |
| x -> nbat | 0.99 | 4.6 | 29.7 | 0.18 M | 0.13 | 1.00 | 0.26 | - | ALU 37%, LSU 29% |
| reduce | 0.99 | 4.0 | 4.0 | 0.19 M | 0.16 | 1.00 | 0.29 | - | LSU 39% |
| fresh-list prune (search steps) | 0.01 | 396.1 | 395.8 | 376 M | 3.18 | 1.00 | 1.00 | 1.00 | ALU 38%, control 25% |

Sources: `raw/nsys/{iso,conc}-2100-P/analysis.txt`, `raw/sass/analysis-iso-2100.{txt,json}`. Concurrent
step at 2100 MHz: 815.9 us GPU span; serialised kernel work ~750 us per plain step.

Two classes of kernels:

* **Issue-bound, the NB kernel** (and the two prune kernels): 3.17 warp instructions per cycle per SM
  = 79% of the issue limit, alone. With PME running concurrently it is 7.8% slower (494.7 vs 459.1 us),
  i.e. it loses issue slots and SM residency to the PME kernels. Day 1's clock scaling (alpha 0.90) and
  the failed occupancy increases point the same way: **only fewer instructions make it faster** (K1).
* **Low-IPC, memory/atomic-latency kernels: spread, bonded, LINCS, SETTLE, x->nbat, reduce** (IPC
  0.10-0.26, sector efficiencies 0.13-0.47). Their time is set by the latency of scattered memory
  accesses and atomics, not by arithmetic (inferred: no stall counters). Spread (95.7 us) and bonded
  (59.4 us) hold SM residency at the start of every step while issuing almost nothing: day 1's
  critical path has the NB kernel waiting 165 us for exactly these two (spread at high priority first,
  then the bonded kernel serially in the NB stream).
* PME FFT/solve/gather are small alone (sum 76 us) but 3-6x longer concurrently (NB blocks hold the
  SMs); they run off the critical path (day 1: the PME chain ends ~95 us before the NB kernel).

## 2. NB kernel anatomy (K1, K2, K3, K4)

### 2.1 Where the instructions go (`raw/sass/nb_f`, regions `bench/profiling/sassprof/regions-nbnxm-cuda.toml`)

433.4 M warp instructions per launch. Exact work counts (`raw/sass/nb_f_paircounts.txt`):

| quantity per step | count | from |
|---|---|---|
| j-cluster halves processed per warp (jm iterations) | 1.05 M | LDG.128 of j-atom x,q |
| i-cluster iterations with mask bit set (8x4 pair slots each) | 4.52 M = 144.6 M pair slots | LDS.128 of i-atom x,q |
| ... of which at least one pair inside the cut-off (body entered) | 4.24 M (93.8%) | MUFU.RSQ warp count |
| **pairs computed inside the cut-off** | **69.7 M** (16.4 active lanes per body) | MUFU.RSQ thread count |
| pair efficiency (computed / slots) | **0.482** | |
| instructions per computed pair | 199 lane-instructions | |
| atomic force adds (RED) | 1.20 M warp / 16.6 M thread | RED count |

The 69.7 M pairs match N rho (4/3) pi rc^3 / 2 ~ 67 M for 185,486 atoms at rc 1.2 nm, i.e. the
kernel computes each pair inside the cut-off once; the waste is the 52% of lanes that sit idle in the
interaction body (cluster geometry: 8 i-atoms x 4 j-atoms per warp; K2 confirmed: half the lanes do
nothing in the 61% of the instructions that form the body).

| region | share of warp instructions | SIMT eff. | warp inst per execution |
|---|---|---|---|
| Ewald real-space correction (`pmeCorrF`) | **16.6%** | 0.51 | 17.0 per body |
| LJ force switch (`ljForceSwitch`) | **15.6%** | 0.51 | 16.0 per body |
| i loop: i-atom load + distance | 13.1% | 1.00 | 12.5 per i-iteration |
| force accumulation (f_ij, i/j buffers) | 10.5% | 0.63 | 10.7 per body |
| exclusion + cut-off test | 8.0% | 0.88 | 7.7 per i-iteration |
| rsqrt + LJ r^-6 force | 7.8% | 0.51 | 8.0 per body |
| jm loop: mask test, cj, j-atom loads | 6.6% | 0.96 | |
| j-force reduction (shuffles + RED) | 5.9% | 0.81 | |
| Coulomb + Ewald (outside pmeCorrF) | 5.9% | 0.51 | 6.0 per body |
| LJ parameter table fetch | 4.9% | 0.51 | 5.0 per body |
| j-packed loop: imask/excl loads | 3.5% | 1.00 | |
| i-force reduction + shift forces | 1.1% | 0.88 | |
| setup + i-cluster preload | 0.5% | 1.00 | |

Memory is not the problem: global sector efficiency 0.82, no shared-memory bank conflicts (wavefront
efficiency 1.00), no local memory; the LJ parameters come from one `LDG.E.64.CONSTANT` per pair (textures
are not used on this build). By the pipe model, FP32 is 44% and ALU 49% busy (lower bounds), issue 79%.

### 2.2 One pair-interaction body = 61 SASS instructions

Read from the line-info disassembly (`raw/sass/nb_f/cubins/`, `NOTES.md` 09:10): the force switch
costs 16 (2 constant MOVs, `r - r_sw`, FSETP+FSEL for `max(.,0)`, 2 FFMA, 6 FMUL, 1 FFMA, merge) because
the source evaluates `-c6*(A2+A3 s)*s*s/r + c12*(B2+B3 s)*s*s/r` term by term and nvcc does not
re-associate floating-point products even with `-use_fast_math`. The Ewald correction costs 17 (4 MOVs
that re-materialise polynomial coefficients because the kernel is at its 64-register limit and the
Estrin-style split `c*z^4 + d` has two immediates per FFMA, plus z^4, 10 FFMA, MUFU.RCP, 2 FMUL).
Everything else in the body is close to minimal (rsqrt + LJ 8, Coulomb 6, accumulation 6 FFMA + BSYNC).

### 2.3 Cost of each part in time: knock-outs and rewrites (in-process replay)

Isolated NB kernel on the live data of 8 plain steps (NB calls 1100, 1250, ... 2150), 30 interleaved
repeats per variant and point; ratio = median over points of (variant / production kernel in the same
rounds), [min, max] over the 8 points; forces vs the production kernel's own launch
(`raw/replay/lock2100/replay.txt`, SM clock locked at 2100 MHz; `raw/replay/unlocked` at 2520 MHz agrees
within 0.3%). Throwaway build `9194ba626fe54212`; `exp0` is SASS-identical to the production kernel.

| variant | static SASS | kernel time vs production | force rel. RMS vs production | registers |
|---|---|---|---|---|
| production, launched again (noise floor) | 1928 | 0.9999 [0.9988, 1.0024] | 4.9e-8 | 61 |
| `exp0` = production source in the experiment file | 1928 (identical) | 1.0003 [0.9976, 1.0014] | 5.0e-8 | 61 |
| **1: factored force switch** `(c12 (B2+B3 s) - c6 (A2+A3 s)) s^2/r`, `s = max(r - r_sw, 0)` | 1872 | **0.9730 [0.9711, 0.9752]** | 5.0e-8 (= noise) | 62 |
| **2: Ewald correction polynomials in Horner form** (same coefficients) | 1880 | **0.9652 [0.9628, 0.9683]** | 2.6e-7 | 60 |
| **1 + 2** (`exp3`) | 1824 | **0.9394 [0.9361, 0.9424]** | 2.6e-7 | 61 |
| 4: LJ table row pointer hoisted per j-cluster | 1952 | 1.0033 [1.0019, 1.0081] (slower) | 4.9e-8 | 62 |
| 1 + 2 + 4 | 1848 | 0.9510 [0.9492, 0.9537] | 2.6e-7 | 61 |
| production at 12 / 10 blocks per SM (78/80 registers) | 1896 | 1.0310 / 1.0328 (slower) | 6.9e-8 | 78 / 80 |
| 1+2+4 at 12 / 10 blocks per SM | 1808 | 0.9977 / 0.9904 | 2.6e-7 | 76 / 78 |
| knock-out: no force switch (TIMING ONLY) | 1696 | 0.8859 [0.8810, 0.8915] | (wrong physics) | 59 |
| knock-out: no Ewald correction (TIMING ONLY) | 1624 | 0.8529 [0.8488, 0.8574] | (wrong physics) | 60 |
| knock-out: no LJ at all (TIMING ONLY) | 1496 | 0.7183 [0.7142, 0.7234] | (wrong physics) | 56 |

* **The force switch costs 11.4% and the Ewald correction 14.7% of the NB kernel**; LJ as a whole 28%.
  The two rewrites recover 2.7% and 3.5% of the kernel; together **6.1%** (additive), i.e. ~28 us per
  step of the 459 us kernel at 2100 MHz, ~27 us of the 446 us production kernel.
* The Horner form has the same accuracy as the production (Estrin) form against the exact function
  `-erf(z)/z^3 + 2 exp(-z^2)/(sqrt(pi) z^2)` over the whole cut-off range (`raw/pmecorrf-accuracy.txt`:
  RMS error 1.010e-7 vs 1.015e-7, max 7.96e-7 vs 8.30e-7; beta 2.6 nm^-1, rc 1.2 nm); its forces differ
  from production by 2.6e-7 relative RMS (rounding order), 5x the atomic-order noise floor and well
  inside single precision. The factored switch is at the noise floor.
* **More registers are slower** (K: the 64-register cap is not the problem): at 12 or 10 blocks per SM
  the compiler stops re-materialising constants (-32 static instructions) but the kernel is 3% slower
  (fewer warps to hide the MUFU/LDS/LDG latencies of an issue-bound loop). Together with day 1 (more
  blocks per SM: -4% / -16%), 16 blocks x 61 registers is the optimum on Ada.
* Hoisting the LJ row pointer adds instructions (pointer arithmetic per j-cluster, 64-bit register pair
  live in the i-loop) and is slower: the per-pair `IMAD + IMAD.WIDE + LDG.64` is already cheap.
* Time per removed instruction (dynamic counts of every variant, `raw/sass-variants/summary.txt`):

  | variant | executed warp instructions vs production | kernel time vs production | time / instruction share |
  |---|---|---|---|
  | factored force switch | 0.9628 (-16.1 M, -3.8 per body) | 0.9730 | 0.73 |
  | Ewald Horner | 0.9703 (-12.9 M, -3.0 per body) | 0.9652 | 1.17 |
  | both | **0.9338** (-28.7 M, -6.8 per body) | **0.9394** | 0.92 |
  | knock-out no force switch | 0.8558 | 0.8859 | 0.79 |
  | knock-out no Ewald correction | 0.8116 | 0.8529 | 0.78 |
  | production at 12 blocks/SM | 0.9671 | 1.0310 (slower) | - |

  In this issue-bound kernel time follows the instruction count of the pair-interaction body almost
  1:1 (0.73-1.17); the Horner form gains a little more than its instruction count (shorter
  register lifetimes, no constant re-materialisation). Fewer instructions with fewer resident warps
  (12 blocks/SM) is slower: latency hiding needs the 32 warps per SM.

### 2.4 End to end (`raw/exp/gpukern-e2e`, 5 seeded interleaved rounds, production command, vs the P build)

| variant (same throwaway binary, selected by env) | ms/step [95% CI] | ns/day | vs P [95% CI] | GPU kJ per ns (vs P) | SM MHz |
|---|---|---|---|---|---|
| P (production-like build) | 0.7428 [0.7399, 0.7457] | 232.65 | 1 | 113.4 | 2358 |
| X-base (experiment build, no GMX_EXP_* set) | 0.7413 [0.7393, 0.7433] | 233.11 | 1.0020 [0.9979, 1.0060] no change | 113.2 | 2359 |
| NB: factored force switch | 0.7348 [0.7314, 0.7381] | 235.19 | **1.0109 [1.0058, 1.0160]** | 111.0 (-2.1%) | 2362 |
| NB: Ewald correction in Horner form | 0.7338 [0.7321, 0.7355] | 235.51 | **1.0123 [1.0083, 1.0163]** | 111.2 (-2.0%) | 2351 |
| **NB: both** | **0.7242 [0.7215, 0.7268]** | **238.63** | **1.0257 [1.0211, 1.0303]** | **109.3 (-3.6%)** | 2358 |
| NB: both + LJ-fetch hoist | 0.7281 [0.7256, 0.7306] | 237.33 | 1.0201 [1.0157, 1.0246] | 109.9 | 2356 |
| NB: both + hoist at 12 blocks/SM | 0.7424 [0.7413, 0.7434] | 232.78 | 1.0005 no change | 114.0 | 2367 |
| NB: production at 12 blocks/SM | 0.7559 [0.7532, 0.7586] | 228.62 | 0.9827 [0.9784, 0.9870] slower | 116.6 | 2361 |

The kernel-level gains carry over in the same order and add up (1.09% + 1.23% = 2.32% vs 2.57% for
both). In production (nsys light, `raw/nsys/X-base-light` vs `raw/nsys/X-nb3-light`, natural clocks) the
rewritten kernel takes 429.6 us instead of 449.9 us (p50, -20.3 us = -4.5%; less than the -6.1% in
isolation because under concurrent PME part of the kernel's time is waiting for SM slots, which the
rewrite does not shorten), and the step gets 18.6 us shorter (prun), i.e. **~92% of the NB kernel time
saved in production reaches the step**: the NB kernel is on the critical path almost 1:1 (the
no-switch knock-out: ~51 us of kernel, 45 us of step, 3.1). Energy per simulated ns drops more than
time (-3.6%): under the binding 325 W cap, fewer instructions also mean less energy per step.

## 3. PME kernels: off the critical path on paper, 18% of the step in practice (K5)

### 3.1 What each PME kernel costs the step (timing-only skips, `raw/exp/gpukern-skips`, 3 rounds)

Kernel launches skipped in the experiment build (wrong physics, TIMING ONLY; the run stays stable over
the 25,000 steps), against the same build without skips (X-base 0.7441 ms/step [0.7385, 0.7497]):

| skipped | ms/step [95% CI] | vs X-base | step time saved | kernel time it removes (production, day 1) | SM MHz |
|---|---|---|---|---|---|
| all PME kernels (spread, FFTs, solve, gather) | 0.6082 [0.6055, 0.6108] | 1.2235 [1.2160, 1.2312] | **136 us (18.3%)** | 341 us | 2321 |
| **spline+spread only** | 0.6629 [0.6479, 0.6782] | **1.1226 [1.0999, 1.1457]** | **81 us (10.9%)** | 89 us | 2321 |
| gather only | 0.7182 [0.7153, 0.7211] | 1.0361 [1.0296, 1.0426] | 26 us (3.5%) | 65 us | 2381 |
| FFTs + solve | 0.7205 [0.7172, 0.7239] | 1.0327 [1.0263, 1.0391] | 24 us (3.2%) | 183 us | 2393 |
| NB knock-out: no force switch (for calibration) | 0.6989 [0.6941, 0.7038] | 1.0646 [1.0576, 1.0717] | 45 us (6.1%) | ~51 us of the NB kernel | 2377 |
| bonded kernel | blew up (no bonded forces) | - | - | 59 us | - |

* **91% of the spread kernel's time is on the step's critical path** (81 of 89 us): at high stream
  priority its 2,899 blocks (3.4 waves) take every SM at the start of the step, x->nbat and the bonded
  kernel wait for them, and the NB kernel cannot start (day 1: NB start 165 us after the previous
  SETTLE). With an IPC of 0.24 those SMs are almost idle while they wait.
* Gather and FFT+solve run concurrently with the NB kernel, but each still costs ~25 us of step time:
  they take SM residency from the issue-bound NB kernel (it is 7.8% slower with PME concurrent, 1),
  i.e. **every PME kernel costs the step roughly the issue slots its warps occupy, not its duration**.
* The separate skips add up to the all-PME skip (81 + 26 + 24 = 131 vs 136 us).
* Clock: removing spread (a low-power kernel) lowers the SM clock (2321 vs 2360 MHz) because the GPU
  then spends more time in the power-hungry NB kernel under the 325 W cap; the gains above are net of
  that.
* Day 1's reaction-field ablation (no PME, and a cheaper NB kernel) gave 24%; the difference to 18.3%
  is the Ewald real-space part of the NB kernel (2.3).

### 3.2 Why spread is slow (`raw/sass/spread`)

6.75 M warp instructions per launch; 64 `RED.E.ADD.F32` atomics per atom (4 threads x 16 grid points,
11.9 M per step) of which each warp instruction touches 9.3 sectors (ideal 4: the 8 atoms of a warp each
write 4 contiguous floats = one sector, mostly in different cache lines); x/q loads are staged through
shared memory (coalesced). IPC 0.24 at 2100 MHz with all 48 warps per SM resident, clock exponent 0.64
(day 1: partly outside the SM clock domain) and the atomics being the only traffic of note point to the
L2 atomic path (latency/throughput of 11.9 M scattered fp32 atomics) as the limiter (inferred; no
counters). Atoms are processed in topology order, so the waters (most atoms) of one block are spread
over the whole box after equilibration: no cache reuse between neighbouring threads.

### 3.3 What would make spread faster: replay of orderings and knock-outs (`raw/replay/spread-*`)

The spline+spread kernel replayed on the live coordinates of 8 plain steps (30 interleaved repeats
each, isolated, into a scratch grid; grids compared with the production order's grid):

| variant | time vs production (2100 MHz; production 98.2 us) | same at 2520 MHz | grid rel. RMS vs production |
|---|---|---|---|
| production order, launched again | 0.9993 [0.9980, 1.0007] | 1.0009 | 4.1e-8 (atomic-order noise) |
| warps take atoms with a stride (interleave) | 0.9953 [0.9922, 0.9971] | 0.9947 | 5.8e-8 |
| atoms sorted by coarse spatial cell (4 grid lines) | 0.9840 [0.9814, 0.9860] | 0.9872 | 7.9e-8 |
| sorted + interleave | 0.9486 [0.9456, 0.9517] | 0.9563 | 8.3e-8 |
| knock-out: splines only, no grid updates (TIMING ONLY) | **0.0831** (8.2 us) | | - |
| knock-out: plain stores instead of atomics (TIMING ONLY) | 0.8164 | | - |
| knock-out: plain stores, sorted atoms (TIMING ONLY) | 0.7011 | | - |

(`raw/replay/spread-lock2100`, `spread-unlocked`, `spread-ko-lock2100`.)

* **92% of the spread kernel is the grid update** (11.9 M scattered 4-byte updates per step); B-spline
  coefficients and grid indices cost 8 us.
* Making the updates non-atomic removes only 18% and perfect spatial order only 2-5% (30% together),
  so neither atomic conflicts nor cache locality of the update stream is the limiter: it is the number
  of individual global-memory updates (3.4 M sectors per step, 9.3 per warp instruction).
* Interleaving the atoms over warps (end to end 1.0007, no change, 2.4) and 16 threads per atom
  (`GMX_EXP_PME_ORDERSQ`, end to end 0.9615, slower) do not help.
* Therefore a faster spread needs **fewer global updates**: atoms in spatial (nbnxm) order so that a
  block's atoms share grid points, accumulation of the block's sub-grid in shared memory, and one
  global atomic per touched grid point. With 64 atoms per block covering a few hundred distinct grid
  points instead of 4,096 updates, the global traffic would drop several-fold. Not prototyped here
  (L effort: GPU-side atom order from the nbnxm grid, tile bounds, periodic wrap); the bounds are the
  skip result (81 us/step, 10.9%) and the knock-outs (the grid update is 90 of 98 us).

## 4. Bonded kernel, update chain and small kernels (K6)

### 4.1 Bonded kernel (`raw/sass/bonded`)

59.4 us alone for 4.56 M warp instructions (IPC 0.26): a latency-bound kernel. One thread per
interaction; the per-thread type dispatch (`listed_forces_gpu_internal.cu:115`, a loop over the
fType ranges) is 13.7% of the instructions; the interaction's atom indices are five 4-byte loads with a
20-byte stride between threads (**20 sectors per warp load, efficiency 0.20**), followed by dependent
coordinate gathers (efficient: float4) and three scattered `RED` force atomics per atom (12 sectors per
warp, efficiency 0.33); dihedral trigonometry dominates the arithmetic. In production it either sits
serially in the NB stream in front of the NB kernel (59 us of the pre-NB critical path, day 1) or, with
the bonded-stream change, overlaps the NB kernel and costs it SM residency (NB kernel 441.3 vs 429.6 us,
`raw/nsys/XO-all-light` vs `X-nb3-light`). The bonded-stream change measured +0.48% [0.10, 0.86] alone
today (day 1: +0.91%) and adds ~0.7% on top of the NB rewrite (`raw/exp/gpukern-e2e2`). A skip bound
is not available (the system blows up without bonded forces). A coalesced layout (structure of arrays
for iatoms, parameters gathered per interaction) would cut the load latency chain; not prototyped.

### 4.2 Update chain: leap-frog -> LINCS -> SETTLE (post-NB critical path)

| kernel | alone (2100 MHz) | production p50 / mean (`X-base-light`) | IPC | sector efficiency |
|---|---|---|---|---|
| leap-frog | 6.0 us | 6.2 / 6.5 | (not collectable) | |
| LINCS | 14.9 us | 14.3 / 21.6 (bimodal: 32.6 when the rolling prune runs) | 0.20 | 0.33 |
| SETTLE | 13.7 us | 12.5 / 12.5 | 0.10 | **0.125** |

All three are sub-wave (150-160 blocks of 256 threads on 142 SMs) and latency-bound. SETTLE moves 3
atoms x float3 per thread (36-byte stride): 8 sectors per 128 useful bytes.

| change (throwaway, `raw/exp/gpukern-e2e2`, 5 rounds) | vs P | mechanism (nsys) |
|---|---|---|
| SETTLE with coalesced staging through shared memory (`GMX_EXP_SETTLE_STAGED`; checked per block that the waters are consecutive atoms) | **1.0044 [1.0008, 1.0079]**; ~+1.2% inside the stacked configuration (`raw/exp/gpukern-e2e3`) | 5.0x fewer global sectors per launch (0.37 M vs 1.84 M, sector efficiency 0.69 vs 0.13) for 2x the (cheap) instructions (`raw/sass-variants/settle-staged`) |
| SETTLE concurrent with LINCS in its own stream (`GMX_EXP_CONCURRENT_SETTLE`) | 1.0017 [0.9978, 1.0056] (no change alone; +0.4% with the NB rewrite) | LINCS p50 14.3 -> 31.7 us and SETTLE 12.5 -> 16.0 us when concurrent (`XO-all-light`): with other blocks on some SMs the 150-block LINCS needs a second wave, so the overlap buys ~5 us of the expected 13 |

Day 1's estimate for "LINCS || SETTLE" (1.5-2.5%) is therefore refuted: LINCS is so sensitive to SM
availability (sub-wave, iterative, `__syncthreads`-bound) that sharing the GPU with SETTLE or the rolling
prune doubles it. What would help is fewer, fuller launches (one fused constraint kernel, or LINCS with
more parallelism), not concurrency.

### 4.3 Small and periodic kernels

* x -> nbat (4.6 us alone) waits 20-25 us for SM slots behind the spread blocks (26.9 us in production,
  8.1 us when spread is skipped, `raw/nsys/X-skipspread-light`): a symptom of spread, not a kernel
  problem.
* reduce (4.0 us, sector efficiency 0.29: gathers by index) and leap-frog (6 us): little to gain.
* Rolling prune (every 2nd step, 33.1 us alone / 40-45 us in production): IPC 2.11, 33% control
  instructions (warp votes and mask updates per cluster pair); it overlaps the update chain and makes
  LINCS bimodal (day 1). Fresh-list prune (396 us per search step, IPC 3.18): 0.3% of run time.
* cuFFT (4 x 96-point + r2c + c2r, 38 us alone, 0.15-0.89 IPC, sub-wave grids) and solve (5.2 us): small
  alone; their step cost (24 us together, 3.1) is SM sharing with the NB kernel.

## 5. From kernel microseconds to step microseconds

How much of a kernel's time the step actually pays, from removing it (timing-only skips) or shortening
it (rewrites), production command, natural clocks (742-744 us/step baseline):

| kernel (group) | time per step: alone (2100 MHz) / in production | step time saved when removed or shortened | share that reaches the step | source |
|---|---|---|---|---|
| NB force kernel, rewrite (force switch + Horner) | -28 us alone / -20.3 us in production | 18.6 us | ~92% of the production saving | `raw/replay/lock2100`, `raw/nsys/X-*-light`, `raw/exp/gpukern-e2e` |
| NB force kernel, no force switch (knock-out) | ~-51 us | 45 us | ~88% | `raw/exp/gpukern-skips` |
| PME spline+spread | 95.7 / 89 us | 81 us | **91%** | skips |
| PME gather | 33.1 / 66 us | 26 us | 79% of alone, 39% of production | skips |
| PME FFTs + solve | 43.3 / 183 us | 24 us | 55% of alone | skips |
| all PME kernels | 172 / 341 us | 136 us | 79% of alone | skips |
| bonded kernel moved to its own stream | 59 us serial -> concurrent | 3.6 us (+0.48%) today, 10 us day 1 | 6-17% | `raw/exp/gpukern-e2e2`, day 1 |
| SETTLE staged | (not separately timed) | 3.3 us (+0.44%) | | `raw/exp/gpukern-e2e2` |

Rule of thumb for this system: **the NB kernel and spread are on the critical path ~1:1; gather and
FFT/solve cost the step about their isolated time minus 20-45% (SM sharing with the NB kernel);
LINCS/SETTLE/bonded are latency-bound and gain little from overlap.**

## 6. Hypotheses

| | claim | verdict | evidence |
|---|---|---|---|
| K1 | the NB kernel is issue/FP32-bound; only fewer instructions help | **confirmed** (issue-bound: 79% of issue slots; FP32 pipe >= 44%, ALU >= 49% busy by the model) | IPC 3.17; time follows the body instruction count 0.73-1.17:1; more occupancy (day 1) and more registers (2.3) both slower |
| K2 | a large share of computed pair slots is outside the cut-off | **confirmed**: 48.2% pair efficiency, the interaction body runs at 16.4 of 32 lanes | `raw/sass/nb_f_paircounts.txt` |
| K3 | the force switch is evaluated for every pair; a better form removes most of its cost | **partly**: it costs 11.4% of the kernel; factoring removes 3.8 of 16 instructions per pair and 2.7% of the kernel (+1.09% end to end); the rest is inherent unless pairs below r_sw are skipped (a warp-uniform test pays only for warps with no pair beyond 1.0 nm, estimated ~12% of bodies; not tested) | 2.3, 2.4 |
| K4 | the per-pair LJ parameter fetch is significant | **refuted**: 4.9% of instructions, one `LDG.E.64` per pair; hoisting is slower | 2.1, 2.3 |
| K5 | spread is bound by global atomics / scattered grid access | **refined**: bound by the number of scattered grid updates (92% of the kernel); atomicity 18%, ordering 2-5% | 3.3 |
| K6 | the update-chain kernels are latency-bound sub-wave launches | **confirmed**: IPC 0.10-0.20, SETTLE sector efficiency 0.125; coalescing SETTLE +0.44%; overlapping LINCS with SETTLE barely helps (LINCS doubles when sharing SMs) | 4.2 |

## 7. Connecting the layers (what changed relative to day 1)

1. Day 1: plain steps GPU-bound; NB kernel 60% of run time; PME "off the critical path".
2. Today, inside the NB kernel: 433 M warp instructions per step at 79% of the issue limit; 61
   instructions per pair-interaction body, of which 33 are the force switch and the Ewald correction;
   48% of the lanes idle in the body by cluster geometry. Rewriting two helper functions (same
   coefficients, same accuracy) removes 6.6% of the instructions and **+2.57% [2.11, 3.03] end to end**.
3. PME is not free: its kernels cost 18.3% of the step (skips). Spread alone is 10.9%, 91% of its
   time is critical path, and it is bound by 11.9 M scattered grid updates per step; ordering and
   non-atomic stores do not fix it, fewer global updates (sorted atoms + shared-memory accumulation)
   would.
4. The latency-bound tail (bonded, LINCS, SETTLE) is cheap to improve a little (+0.4-0.5% each) but
   resists concurrency.
5. Stacked with day 1's two scheduling fixes and -ntomp 20: +5.70% [5.32, 6.07] without and
   **+7.19% [6.58, 7.80] = 249.4 ns/day, -5.7% GPU energy per ns with the SETTLE staging**
   (`raw/exp/gpukern-e2e2`, `raw/exp/gpukern-e2e3`), quality gate passed (8).

## 8. Quality gate of the kept configuration

The stacked configuration that gives +7.19% (throwaway build `1093125cdb89aee7` = worktree + day-1
OpenMP fix; `GMX_EXP_NB_VARIANT=13 GMX_EXP_CONCURRENT_SETTLE=1 GMX_EXP_BONDED_STREAM=1
GMX_EXP_SETTLE_STAGED=1`) passed `gmxbench quality --strict --tier quick` over the whole default suite
plus the target (`raw/gmxbench/quality-XO-final`, 63 cases, 546 mdrun jobs, P build as baseline):
78 IDENTICAL (all 72 bitwise-reproducible configurations among them), 67 EQUIVALENT (GPU
configurations; force rel. RMS 0.52-1.67x the configuration's own run-to-run noise, tolerance 10x;
MAS1 production 1.05e-7 vs noise 1.01e-7), 37 SKIPPED (non-PME systems in the `gpu-resident`
configuration, unsupported for baseline and candidate alike), grompp 63/63 IDENTICAL, no DIFFERENT, no
FAIL. Caveat: the NB rewrite is selected by swapping the ElecEw_VdwLJFsw force-only kernel, so it is
exercised only by suite cases with that flavour (the MAS1 cases); as a source change in
`nbnxm_kernel_utils.h` it would reach every force-switch / analytical-Ewald GPU kernel (also SYCL and the
CUDA free-energy kernel) and needs the full tier.

## 9. Not measured / limitations

* No hardware counters (DCGM): no stall reasons, achieved occupancy, cache hit rates, DRAM or L2
  throughput, roofline; "latency-bound" and "atomic/update-bound" are inferences from exact
  instruction/sector counts, IPC, clock scaling and knock-outs. Pipe utilisations are a model.
* sassprof could not collect leap-frog, and collects one kernel per run (CUPTI 2025.3); the counts are
  exact but come from a 400-step window (two pair-list lifetimes).
* Replay timings are isolated (no concurrent PME); end-to-end numbers are the ones that count.
* Throwaway code: the variants are runtime switches in one binary (`X-base` = P within +-0.4% over two
  rounds; production NB kernels SASS-identical to P; the spread kernel differs by the inactive
  experiment branches). The XO binary was built before the spread knock-out modes were added (patch
  hunk `expSpreadMode`), which are only used by the spread replay.
* Not prototyped: the spread rewrite (#5), gather/FFT changes (#6), LINCS restructuring (#7), bonded
  layout (#8); their gains are bounds and stated assumptions.
* Upstream note (from reading code, not run; `raw/prior-art-gpu.md`): on upstream main the HIP kernel
  now calls `ljForceSwitch<doCalcEnergies, true>` (F*r) into an F/r accumulator after !6195; this does
  not affect the 2026.4-based `dev` used here.
