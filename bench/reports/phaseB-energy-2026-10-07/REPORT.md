# Phase B: energy conservation of feature/gpu-kernel-opt vs dev on MAS1 (2026-10-07)

Benchmark runs (not production). The run inputs and outputs are customer-derived and stay out of git; only this
report is committed (bench/reports/).

## Setup

* Input: the MAS1 system as prepared by gmxbench (`~/.cache/gmxbench/systems/mas1-20e-0581e770c8`), production
  mdp (`../profiling-2026-10-06/raw/inputs/prod/grompp.mdp`) switched to NVE: `tcoupl = no`, `pcoupl = no`,
  `nsteps = 1000000` (2 ns), no trajectory output (`nve.mdp`). One `nve.tpr` (dev grompp) for both builds.
* Builds: dev `~/.cache/gmxbench/builds/a538b705e1bacbff`, branch (`da0eed4161`, A1-A5)
  `~/.cache/gmxbench/builds/d3c6009d615519ff`, both gmxbench profile `cuda`.
* mdrun: `-ntmpi 1 -ntomp 16 -nb gpu -pme gpu -bonded gpu -update gpu -nstlist 200 -notunepme -pin on`.
  For NVE mdrun keeps `nstlist = 20` ("Can not increase nstlist because an NVE ensemble is used"), rlist 1.209 nm.
* Order dev, branch, branch, dev (`run_nve.sh`); analysis `analyze.py` (linear fit of the total energy, which
  is the conserved energy without coupling; block slopes over four 0.5 ns blocks).

## Result: no difference between the builds

| run | drift (kJ/mol/ns) | per atom (kJ/mol/ns) | <T> (K) | <Epot> (kJ/mol) | <P> (bar) |
|---|---|---|---|---|---|
| dev 1 | 112508 | 0.6066 | 336.18 | -1842412 | 454.0 |
| dev 4 | 112437 | 0.6062 | 336.20 | -1842476 | 452.7 |
| branch 2 | 112510 | 0.6066 | 336.23 | -1842421 | 454.5 |
| branch 3 | 112824 | 0.6083 | 336.29 | -1842157 | 462.2 |

Mean drift per atom: dev 0.6064, branch 0.6074 kJ/mol/ns (+0.16%, within the 0.28% spread between the two
branch runs). Temperature, potential energy and pressure agree within run-to-run scatter. The changes on the
branch do not affect energy conservation.

Throughput in these NVE runs (nstlist 20, search every 20 steps): dev 155.1 / 156.1 ns/day, branch 173.2 /
172.9 ns/day (+11.4%; the OpenMP fix for the GPU bonded/LINCS setup weighs more with frequent searches).

## Side finding: the production settings heat the system in NVE

Both builds drift by ~0.61 kJ/mol/ns per atom (without a thermostat the system heats from 310.5 K in the
first 10 ps to 365.7 K in the last 10 ps of the 2 ns, identically for both builds). That is within GROMACS' default pair-list tolerance (`verlet-buffer-tolerance` 0.005 kJ/mol/ps =
5 kJ/mol/ns per atom), and in production the v-rescale thermostat removes the heat, but it is large.
0.5 ns NVE diagnostics on the branch build (`run_diag.sh`):

| variant | drift per atom (kJ/mol/ns) | vs base | ns/day |
|---|---|---|---|
| base (production settings) | 0.536 | - | 173.3 |
| `lincs-iter = 2` | 0.165 | -69% | 172.9 |
| `verlet-buffer-tolerance = 0.0005` | 0.423 | -21% | 165.2 |

About 70% of the drift comes from LINCS with one iteration (`lincs-iter = 1`, CHARMM-GUI default; grompp
notes "For energy conservation with LINCS, lincs_iter should be 2 or larger"), ~20% from the pair-list buffer.
`lincs-iter = 2` costs 0.2% throughput here. Worth considering for the long MAS1 run (a decision for the
user; nothing was changed in the production inputs). The interlocked aromatic rings in the input (see memory
notes / day-1 report) still need fixing before any long run, independent of this.
