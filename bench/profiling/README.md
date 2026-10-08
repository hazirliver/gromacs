# Profiling and experiment tools

Tools used for the MAS1 performance work (nsys/perf/ncu capture and analysis, `sassprof`, kernel replay,
the interleaved experiment runner `prun`, experiment specs and the prototype patches). What each tool does,
how to run it and the environment caveats are described in
[`../OPTIMIZATION.md`](../OPTIMIZATION.md#6-tools); the reports that used them are in
[`../reports/`](../reports/).

Python tools run with the gmxbench virtualenv (`bench/setup.sh` once; `prun` is a launcher for it). Paths in
`experiments/*.toml` point to build directories and tprs on the benchmark node; adjust them before reuse.
