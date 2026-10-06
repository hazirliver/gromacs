# Notes for working on this GROMACS fork

## Branches
- Work happens on `dev` (based on the latest GROMACS release tag, currently v2026.4), not on `main`.
- Feature branches are created from `dev` and merged back into `dev`; new upstream releases are merged
  into `dev` as release tags.

## Testing changes
- Every change to GROMACS sources (especially optimisations) is tested with `gmxbench` following
  `bench/TESTING.md`. Minimum before committing to `src/`:
  `bench/bin/gmxbench all -A dev -B WORKTREE --tier quick` (add `--strict` for changes that must keep
  results bit-identical); it must exit 0 and the report must show the claimed speedup on the target
  system.
- The target system (MAS1 membrane complex, `bench/suites/target.toml`) runs on every gmxbench
  invocation; use `--target-only --tier smoke` for a ~5 min loop while developing.
- After changing `bench/` itself, run `bench/validation/run_validation.sh`.

## Data
- `data/` holds customer input (the MAS1 system). Never commit it; gmxbench reads it by path
  (`GMXBENCH_MAS1_ARCHIVE` overrides the location).
