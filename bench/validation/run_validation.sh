#!/usr/bin/env bash
# Re-validates gmxbench itself (not GROMACS): A/A runs must be clean, injected changes must be caught.
#   A/A        dev vs WORKTREE (identical source, independent builds) -> no DIFFERENT, no perf change
#   NC-reorder mathematically equivalent reordering of the leapfrog update -> bits change, EQUIVALENT,
#              --strict fails; md-vv/SD cases (other code path) stay IDENTICAL
#   NC-coulomb short-range Coulomb prefactor x1.0001 -> DIFFERENT wherever charges exist
#   NC-table   GMX_NBNXN_EWALD_TABLE=1 (env only) -> EQUIVALENT on CPU configurations
#   perf-known GMX_CUDA_GRAPH=1 + GMX_NBNXN_EWALD_TABLE=1 on the candidate -> faster where graphs apply,
#              slower on CPU-only runs, no-change elsewhere
# Usage: bench/validation/run_validation.sh [tier]   (default: quick; results in bench/results/validation-*)
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
tier="${1:-quick}"
G="$repo/bench/bin/gmxbench"
R="$repo/bench/results"
work="${GMXBENCH_WORKDIR:-$HOME/.cache/gmxbench}/negctl"
mkdir -p "$work"
for nc in reorder-leapfrog-fma coulomb-prefactor; do
    if [ ! -d "$work/$nc" ]; then
        git -C "$repo" worktree add --detach "$work/$nc" dev
        git -C "$work/$nc" apply "$here/negctl-$nc.patch"
    fi
done
cd "$repo"
$G all -A dev -B WORKTREE --tier "$tier" --strict --results "$R/validation-AA-$tier"; echo "A/A rc=$? (expect 0)"
$G quality -A dev -B "src:$work/reorder-leapfrog-fma" --tier "$tier" --results "$R/validation-NC-reorder"; echo "rc=$? (expect 0)"
$G quality -A dev -B "src:$work/reorder-leapfrog-fma" --tier "$tier" --strict --cases 'water-5k/*' --configs 'cpu-*' \
    --results "$R/validation-NC-reorder-strict"; echo "rc=$? (expect 1)"
$G quality -A dev -B "src:$work/coulomb-prefactor" --tier "$tier" --results "$R/validation-NC-coulomb"; echo "rc=$? (expect 1)"
$G quality -A dev -B dev --candidate-env GMX_NBNXN_EWALD_TABLE=1 --tier "$tier" --results "$R/validation-NC-ewaldtable"
echo "rc=$? (expect 0)"
$G perf -A dev -B dev --candidate-env GMX_CUDA_GRAPH=1 --candidate-env GMX_NBNXN_EWALD_TABLE=1 --tier "$tier" \
    --cases water-5k/pme lysozyme/pme mas1/charmm36 --results "$R/validation-perf-knownAB"
$G report "$R"/validation-* -o "$R/validation-summary.html"
