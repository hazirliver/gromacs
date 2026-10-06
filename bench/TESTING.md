# How to test a change (correctness + performance)

This is the required procedure for any change to GROMACS sources in this fork (optimisations in
particular). It uses `gmxbench` (see `README.md` for how the tool works). Baseline is always `dev`
(or the branch you started from); candidate is your working tree or branch.

The **target system** — currently MAS1 + Gi + 20-hydroxyecdysone in a membrane from `data/`
(`bench/suites/target.toml`) — is included in every run automatically. Its input archive is not in
git; keep it at `data/mas-20e-production-inputs.tar` or set `GMXBENCH_MAS1_ARCHIVE`. If gmxbench prints
`WARNING: target system ... unavailable`, fix that before trusting the results.

## 0. Once per machine

```bash
bench/setup.sh                                  # Python env (uv is fetched privately if missing)
bench/bin/gmxbench list --tier quick            # what will run; target cases are marked *
```

Benchmark on a quiet node: no other jobs on the GPU or CPU, never two gmxbench runs at the same time,
baseline and candidate on the same node in the same session (gmxbench interleaves them for you).

## 1. Inner loop while developing (minutes)

```bash
bench/bin/gmxbench all -A dev -B WORKTREE --tier smoke --target-only      # ~5 min, MAS1 only
bench/bin/gmxbench perf -A dev -B WORKTREE --target-only --configs perf-mas1-production   # one config
```

`WORKTREE` is rebuilt incrementally on every call, so this is the edit-build-measure loop.

## 2. Before committing a change to `src/` (about one hour)

```bash
bench/bin/gmxbench all -A dev -B WORKTREE --tier quick --strict           # bit-preserving changes
bench/bin/gmxbench all -A dev -B WORKTREE --tier quick                    # changes that may reorder FP math
```

The run must exit with status 0. In the report (`bench/results/<session>/report.html`):

* **Quality**: no `DIFFERENT` and no `FAIL`; candidate grompp output identical.
  * A change that is *meant* to keep results bit-identical (data layout, scheduling, removing redundant
    work, launch/synchronisation changes) must pass `--strict`: every configuration that is bitwise
    reproducible on the baseline (all CPU `-reprod` runs, incl. domain decomposition) stays `IDENTICAL`.
  * A change that legitimately alters floating-point order (new SIMD/GPU kernel, different summation)
    may turn CPU runs into `EQUIVALENT`; the force/energy differences in the report must stay at the
    level of the GPU runs' own run-to-run noise (≈1e-7 relative force RMS). Say so in the commit message.
  * GPU configurations are never bit-reproducible (`nd` in the report); they are judged against their
    measured noise floor.
* **Performance**: the claimed improvement shows as `faster` on the target (MAS1) cases, ideally on
  several configurations, with the 95% CI excluding 1; no case is `slower` (add `--fail-on-slowdown` to
  make that a hard gate). Check that the stage/kernel tables attribute the change to the code you
  touched. Effects smaller than the reported *minimum detectable effect* need more repeats
  (`--repeats 9`) or the full tier.
* **Telemetry**: energy per simulated ns should not get worse for a speedup claim; look for
  `sw_power_cap` / thermal throttling flags, which can hide or fake small effects on MAS1.
* Small CPU-kernel effects (a few %) can come from code layout alone; confirm them on more
  systems/configurations before attributing them to the change.

Mention the results in the commit or merge request: session name, quality counts, end-to-end speedups
with CIs for the target configurations.

## 3. Before merging into `dev`

```bash
bench/bin/gmxbench all -A dev -B <branch> --tier full --strict      # hours
bench/bin/gmxbench upstream --build <branch>                          # GROMACS' own unit + regression tests
bench/bin/gmxbench sweep --build <branch> --target-only               # re-derive the best MAS1 run arguments
```

All three must succeed. If the sweep finds a different best configuration for MAS1, update
`[configs.perf-mas1-production]` in `bench/suites/target.toml`.

## 4. Comparing over time

```bash
bench/bin/gmxbench report bench/results/<old> bench/results/<new> -o /tmp/history.html
```

Results directories are git-ignored; keep the ones that justify merged work (or copy the
`report.html` + `records.jsonl` somewhere permanent).

## 5. Customising

* Target systems and their cases: `bench/suites/target.toml` (committed).
* Personal or machine-specific overrides: `bench/suites/local.toml` (git-ignored), extra files via
  `GMXBENCH_SUITE=a.toml:b.toml` or `--suite file.toml`. Cases merge by `name`, so an override can
  replace a single case, e.g. change `repeats`, `nsteps` or `configs`.
* `--target-only` / `--no-target` select or drop the target cases; `--cases`/`--configs` take globs.
* Builds: `--profile cuda` (default, cycle sub-counters + NVTX), `cuda-nosub` (production-like),
  `cpu`, `cpu-double`.

## 6. After changing gmxbench itself

```bash
bench/validation/run_validation.sh quick
```

It runs an A/A comparison (must be clean), injects a bit-changing-but-equivalent code change and a
physics bug (both must be caught), and an environment change with known performance effects (must be
detected where it applies and nowhere else).

## Known caveats for the target system

* In `npt2.gro` the rings of PROA PHE147 and PROB TRP104 are interlocked (ring penetration). Runs at
  2 fs are stable; 4 fs with hydrogen mass repartitioning blows up, so those cases are disabled
  (`tiers = []`) in `target.toml` until the input is fixed.
* CHARMM CMAP runs on the CPU, so "GPU-resident" MAS1 runs still copy coordinates/forces every step and
  cannot use CUDA graphs.
