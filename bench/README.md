# gmxbench — reproducible quality & performance benchmarking for GROMACS

`gmxbench` is the foundation for optimisation work on this GROMACS fork. It answers three questions for
any code change, on many different simulation setups at once:

1. **Quality** — *did the change alter the results?* Bitwise A/B comparison of full-precision outputs,
   with a calibrated numeric tolerance for configurations that are not bitwise reproducible.
2. **Performance** — *is it faster, where, and is that real?* Statistically controlled end-to-end A/B
   runs, per-stage timings (mdrun cycle sub-counters), per-kernel GPU timings (Nsight Systems) and
   isolated kernel micro-benchmarks.
3. **Usage** — *which mdrun arguments should I use for this kind of system?* A per-system sweep over
   offload modes, threading, `nstlist`, CUDA graphs, PME tuning and multi-simulation throughput.

Everything is recorded in one uniform format (`records.jsonl` + `session.json`) and rendered into a
self-contained HTML report (no network needed to view it), plus a flat CSV.

It does not modify GROMACS sources: baseline and candidate are built out-of-tree from git refs or the
working tree.

**How changes must be tested in this fork: see [`TESTING.md`](TESTING.md).** The system the fork is
optimised for (currently the MAS1 membrane complex from `data/`) is defined in `suites/target.toml` and
runs on every invocation.

## Quick start

```bash
bench/setup.sh                       # once: creates bench/.venv (numpy, scipy, pyedr, nvidia-ml-py)

# Everything (quality + perf + kernel micro-benchmarks) for your uncommitted changes vs dev:
bench/bin/gmxbench all -A dev -B WORKTREE --tier quick
# -> bench/results/<timestamp>-all/report.html

# Individual suites
bench/bin/gmxbench quality -A dev -B WORKTREE              # bitwise/tolerance regression check
bench/bin/gmxbench perf    -A dev -B my-branch --repeats 9 # statistical performance A/B
bench/bin/gmxbench micro   -A dev -B WORKTREE              # nonbonded kernel micro-benchmarks
bench/bin/gmxbench sweep   --build WORKTREE                # best mdrun arguments per system
bench/bin/gmxbench upstream --build WORKTREE               # GROMACS' own ctest unit + regression tests
bench/bin/gmxbench report bench/results/run1 bench/results/run2   # combined report + history table

# Only the target system (MAS1), fast loop while optimising for it:
bench/bin/gmxbench all -A dev -B WORKTREE --tier smoke --target-only
```

Build specs (`-A`, `-B`, `--build`):

| spec | meaning |
|---|---|
| `WORKTREE` | the current checkout incl. uncommitted changes; one persistent build dir per checkout, rebuilt incrementally |
| `<git ref>` | branch / tag / commit, checked out into a detached `git worktree` and built separately |
| `src:/dir` | another source tree (e.g. a second clone or a patched copy); built incrementally like `WORKTREE` |
| `path:/dir` | an existing build directory containing `bin/gmx` |

Useful options: `--tier smoke|quick|full`, `--cases 'water-*' 'lysozyme/*'`, `--configs 'gpu-*'`,
`--profile cuda|cuda-nosub|cpu|cpu-double`, `--target-only` / `--no-target`, `--candidate-env KEY=VAL` (e.g. try `GMX_CUDA_GRAPH=1`
without changing code), `--repeats N`, `--no-nsys`, `--parallel N` (quality jobs), `--no-reuse`.

Heavy state (worktrees, builds, prepared systems, downloads, cached baseline runs) lives in
`$GMXBENCH_WORKDIR` (default `~/.cache/gmxbench`); results go to `bench/results/` (git-ignored).

## Tiers

| tier | purpose | typical wall time on 1 GPU + 20 cores |
|---|---|---|
| `smoke` | check the harness itself | ~2–5 min |
| `quick` | every optimisation change | quality ~5 min, perf ~40 min, micro ~10 min |
| `full` | before merging / release; more systems, longer runs, more repeats | hours |

## 1. Quality suite

For each *case* (system × mdp variant) and *configuration* (mdrun offload/parallelisation), the
**same `.tpr`** (made by the baseline's grompp) is run three times: baseline `A`, baseline again `A2`,
candidate `B`.

* **Bitwise level.** SHA-256 of every TRR frame array (box, x, v, f at full precision), every EDR
  frame (all energy terms) and the data lines of any `.xvg` output. `B == A` → **IDENTICAL**.
* **Determinism check.** `A == A2`? CPU runs with `-reprod` are bitwise reproducible (also with
  domain decomposition and separate PME ranks), so for them *any* difference is caused by the change.
  GPU runs are not: GPU reductions use atomics and mdrun refuses `-reprod` with GPU non-bondeds.
  Such configurations are flagged `nd` and judged at the numeric level only.
* **Numeric level.** Step-0 forces and energies (identical input coordinates, so only arithmetic
  differs): force relative RMS difference, and the largest change of any energy term normalised by the
  summed magnitude of the component energy terms (robust against cancellation in sums such as Total Energy). **EQUIVALENT** when within tolerance
  (`[quality.tolerance]`, default 1e-5) or within 10× the measured A-vs-A2 noise of a non-deterministic
  configuration; otherwise **DIFFERENT**. Trajectory divergence curves, conserved-energy drift and
  ensemble averages are recorded as supporting evidence.
* **grompp check.** The candidate's grompp must produce a `.tpr` with identical content.
* **Coverage.** Suite cases (SPC/E water with PME / reaction-field / LJ-PME / switched vdW, md-vv NVE,
  SD, Nose-Hoover + Parrinello-Rahman, soft-core FEP, TIP4P virtual sites, lysozyme with
  h-bonds/all-bonds/HMR 4 fs/virtual-site hydrogens 4 fs, a 185k-atom CHARMM36 membrane–GPCR system),
  each under CPU (1 rank, 4-rank DD, separate PME rank) and GPU (NB, NB+PME, fully resident)
  configurations — plus every input of the version-matched upstream `regressiontests` package
  (`complex/*`, `essentialdynamics/*`), auto-downloaded and md5-verified.

For an optimisation that is *meant* to be bit-preserving (memory layout, scheduling, avoiding
redundant work), require IDENTICAL on all deterministic configurations. For one that legitimately
changes the floating-point order (e.g. new SIMD kernel), expect EQUIVALENT with differences at the
noise level shown in the report.

Baseline runs are cached by (baseline binary + env, tpr, configuration, host), so re-checking a series
of candidates against the same baseline only runs the candidate.

## 2. Performance suite

Per case × configuration:

1. **Calibration/warm-up** run per side; the baseline's ms/step sets `nsteps` so the timed window is
   ~`target_seconds` (identical `nsteps` for both sides).
2. **Interleaved repeats**: each round runs A and B in a seeded random order (cancels drift such as
   thermal throttling); threads pinned; PME tuning off (it is timing-driven, i.e. non-deterministic
   work); counters reset with `-resetstep` to exclude start-up.
3. **Metrics**: ns/day (primary); per-stage ms/step from the cycle accounting table (build profile
   `cuda` enables `GMX_CYCLE_SUBCOUNTERS`, giving NB kernel, pruning, PME spread/gather/FFT/solve,
   search, constraints, launch overheads, ...); per-kernel and per-copy GPU µs/step from extra runs
   under Nsight Systems; GPU temperature/clocks recorded with every run.
4. **Statistics** (`gmxbench/stats.py`): speedup = ratio of geometric means with a Welch 95% CI on log
   values (noise in timings is multiplicative); Mann–Whitney U and a bootstrap CI as distribution-free
   cross-checks; quantisation of log-printed times is folded into the variance; stage/kernel families
   are FDR-controlled (Benjamini–Hochberg). Verdicts: *faster* / *slower* (significant and > `min_effect`,
   default 1%), *negligible* (significant but < 1%), *no-change*. The minimum detectable effect for the
   observed noise is reported, so you know whether more repeats are needed.

What the A/A validation on this machine showed (quick tier, 5 repeats): 16/16 end-to-end comparisons
*no-change*, minimum detectable effect 0.5–4.6% (median ≈1.6%; tightest for GPU-resident runs, widest for
CPU-heavy FEP and GPU-NB+PME on the 185k-atom membrane). At the stage/kernel level ~380 comparisons in
31 FDR families produced one family with a (single, correlated) discovery at q = 0.04 — the expected rate
of false discoveries. Treat an isolated stage/kernel hit with q close to 0.05 as "re-run with more
repeats"; real effects (e.g. the CUDA-graph or Ewald-table checks below) come with q ≪ 0.01 and a
consistent end-to-end change.

### GPU telemetry

Every timed perf repeat (GPU configurations; `[perf] telemetry = "gpu" | "all" | "off"`) and every sweep
measurement is sampled with NVML (`nvidia-ml-py`, fallback `nvidia-smi`) every 0.1 s by a low-priority
thread pinned to the last logical CPU. mdrun's stderr is streamed, so the statistics cover exactly the
timed window (from the `-resetstep` counter reset to the end of the run):

| metric | meaning |
|---|---|
| `util_gpu_mean`, `util_mem_mean` | NVML GPU / memory-controller utilisation (%) |
| `vram_proc_max`, `vram_used_max` | peak VRAM of the mdrun process(es) / of the whole device (MiB) |
| `power_mean`, `power_max` | board power (W) |
| `energy_kj_per_ns` | GPU board energy (hardware energy counter) per simulated ns — the GPU's share of a campaign's cost (CPU/node power is not measured) |
| `ns_per_day_per_kw` | throughput per kW of GPU power |
| `sm_clock_mean`, `temp_max`, throttle reasons | clocks/thermals; throttling is flagged in the report |
| `pcie_tx_mean`, `pcie_rx_mean` | PCIe traffic GPU→host / host→GPU (MB/s) — shows offload-mode data movement |
| `cpu_cores_busy` | mdrun core-time / wall-time |

Each run also writes the raw samples to `gpu_telemetry.csv` in its run directory. Energy per ns, ns/day
per kW, power and VRAM are compared A vs B with the same statistics as ns/day (FDR-controlled); the other
quantities are shown as informational differences. The report adds an efficiency table, utilisation /
power / VRAM time series of the median repeat per side (with the start of the timed window marked), and a
*GPU load vs performance* section: Spearman/Pearson correlations of ns/day with utilisation, power, clock,
PCIe traffic, VRAM and CPU load across the sweep's configurations (with scatter plots) and across repeats
of each perf configuration (where a strong correlation with clock or temperature indicates throttling or
interference rather than code effects). The sweep's recommendations include the most energy-efficient
configuration per system next to the fastest one.

Micro-benchmarks (`gmxbench micro`) wrap `gmx nonbonded-benchmark`: all 36 kernel flavours × {F, VF}
for several sizes/thread counts, A/B interleaved, per flavour statistics plus a geometric-mean speedup.

## 3. Sweep: which arguments for which use case

`gmxbench sweep` measures, per system, plain `gmx mdrun` (the default users get), then searches in
stages: CPU rank×thread layouts and GPU offload modes × threads → `-nstlist` → CUDA graphs
(`GMX_CUDA_GRAPH=1`) → PME tuning → N simulations sharing the GPU. The report lists the best
single-simulation command line and the best throughput setup per system with the gain over default.

## Exit codes (scripts / CI)

`quality`, `perf`, `micro` and `all` exit with status 1 when a gate fails, 0 otherwise:

* quality: any **FAIL** or **DIFFERENT** result, or a candidate grompp whose `.tpr` content differs;
  with `--strict` also any non-IDENTICAL result on a configuration that is bitwise reproducible on the
  baseline (use this for optimisations that must not change a single bit);
* perf (`--fail-on-slowdown`): any case/configuration whose end-to-end verdict is *slower*.

```bash
bench/bin/gmxbench quality -A dev -B WORKTREE --strict && echo "bit-identical where it can be"
bench/bin/gmxbench perf -A dev -B WORKTREE --fail-on-slowdown --configs 'perf-gpu-*'
bench/bin/gmxbench list --tier full      # what a tier contains
```

## Results format

`records.jsonl` — one JSON object per measurement, same keys for every suite:

```json
{"v": 1, "session": "...", "suite": "perf", "kind": "stage", "case": "lysozyme/pme",
 "config": "perf-gpu-resident", "side": "B", "build": "6c69f720aa73...-cuda", "repeat": 3,
 "metric": "stage:PME mesh", "value": 0.0285, "unit": "ms/step", "better": "lower",
 "tags": {"pct": 45.5}}
```

`suite` ∈ quality | perf | micro | sweep | upstream; `side` A = baseline, B = candidate, S = single build.
`session.json` holds the command line, host fingerprint (CPU, GPUs, driver, CUDA), both builds
(git sha, dirty-diff hash, CMake flags, `gmx -version` info, runtime env), the full suite definition
used, and the computed summaries (verdicts, comparisons, recommendations). `report.csv` is a flat
export of all records.

## Suite definition (`bench/suites/*.toml`)

Files are merged in order: `default.toml` (generic coverage), `target.toml` (the systems you optimise
for; their cases always run, in every tier listed by the case and regardless of `--cases`), the
git-ignored `local.toml`, files listed in `$GMXBENCH_SUITE` (colon-separated), then `--suite` files.
Arrays of named tables (`[[quality.cases]]`, ...) merge by `name`, so an overlay can replace one case.
A perf case may set its own `repeats`; an `archive` system may name a `path_env` variable that
overrides its `path`.

* `[profiles.*]` — CMake build profiles (`cuda` default, `cuda-nosub` production-like, `cpu`, `cpu-double`).
* `[systems.*]` — reproducible system recipes: `water` (replicated SPC/E / TIP4P boxes, optional FEP
  subset), `pdb2gmx` (download → pdb2gmx → solvate → ions → EM → NVT/NPT with deterministic CPU MD),
  `archive` (ready inputs from a tarball; skipped if missing, e.g. `data/` is not in git).
* `[mdp.*]` — named mdp fragments; a case merges a list of them (`"@system"` = the system's own mdp).
  Seeds are fixed (`ld-seed`, `gen-seed`) so tpr files are reproducible.
* `[configs.*]` — mdrun argument sets (+ env, GPU requirement, CPU footprint).
* `[[quality.cases]]`, `[[perf.cases]]`, `[[micro.nbnxm]]`, `[[sweep.systems]]` — the suites.

Pass extra/override files with `--suite my.toml` (merged on top of the default). To add a new use
case, add a system and an mdp fragment, then reference them from quality/perf/sweep entries.

## Known properties of this machine/build (found while validating the suite)

* CPU `-reprod` runs are bitwise reproducible run-to-run and across independent builds of the same
  source — including 4-rank domain decomposition and a separate PME rank.
* All GPU-offloaded configurations are *not* run-to-run reproducible: step-0 force noise ≈1e-7 relative
  RMS, energies ≈1e-6.
* `GMX_ENABLE_GPU_TIMING` aborts with GPU update (`cudaErrorNotReady`); GPU kernel timings therefore come
  from Nsight Systems.
* `-bonded gpu` fails on systems without GPU-capable bonded types (pure water); gmxbench falls back to
  `-bonded cpu`, which is equivalent there, and notes it.
* With `-tunepme`, PME load balancing on fast systems can outlast `-resetstep`; gmxbench re-runs with a
  later reset point.
* **Code layout matters at the few-percent level.** Two builds of the *same* commit whose source/build
  directories had different path lengths produced different `.text` sections (the paths end up in
  assertion messages and data paths, changing instruction encodings and the alignment of everything
  after them). The A/A run flagged the CPU-bound soft-core FEP case as 4% slower (p = 0.008, B slower
  in every interleaved round). gmxbench therefore gives every build a fixed-length source path
  (`<workdir>/src/<12 chars>`, a symlink for `WORKTREE`/`src:`) and build path
  (`<workdir>/builds/<16 hex>`); identical sources now yield a byte-identical `.text`, and the same A/A
  comparison reads 0.998 [0.979, 1.018]. Keep in mind that a real code change also moves code: a CPU-kernel
  effect of a few percent should be confirmed (e.g. on more systems/configurations, or with a layout
  perturbation such as `-falign-functions=64`) before attributing it to the change itself.

## Layout

```
bench/
  bin/gmxbench          launcher (uses bench/.venv)
  setup.sh              creates bench/.venv
  TESTING.md            the testing procedure for changes in this fork
  suites/default.toml   generic systems, mdp fragments, run configurations, cases, tiers
  suites/target.toml    target systems (MAS1) - always run; local.toml = personal overrides (ignored)
  validation/           negative-control patches + run_validation.sh (validates gmxbench itself)
  gmxbench/             package: cli, builds, systems, gmx (grompp/mdrun/nsys), logparse, trr,
                        quality, perf, micro, sweep, upstream, stats, results, report
  results/              session directories (git-ignored)
```
