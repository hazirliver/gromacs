"""Create the run inputs (.tpr) for the MAS1 profiling session.

    bench/profiling/mkinputs.py --system DIR --gmx BUILD/bin/gmx --out RESULTS/raw/inputs [--only NAME ...]

DIR is a prepared gmxbench system (conf.gro, topol.top, toppar/, index.ndx, md.mdp), e.g.
~/.cache/gmxbench/systems/mas1-20e-*. Every input gets its own directory with the mdp, the (possibly
modified) topology, topol.tpr and the grompp log. Everything except `prod` and the PME/nstcalcenergy
points changes the physics and is labelled TIMING ONLY, NOT PHYSICAL in a README in its directory.
Never commit these directories: they contain the customer's system.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gmxbench.gmx import format_mdp, merge_mdp, parse_mdp  # noqa: E402

CONTINUE = {"continuation": "yes", "gen-vel": "no", "nsteps": -1}
TIMING_ONLY = "TIMING ONLY, NOT PHYSICAL: this input deliberately changes the force field / algorithms.\n"

# name -> (mdp overrides, topology transform or None, physical?)
INPUTS = {
    # production mdp exactly as in data/ (output every 1000 steps, xtc every 1 ns), continuing the run
    "prod": ({}, None, True),
    # the gmxbench perf output settings (nstlog 0, no xtc), to compare with gmxbench perf numbers
    "prod-perfout": ({"nstlog": 0, "nstxout-compressed": 0}, None, True),
    # energy/virial computation interval (with T/P coupling and COM removal left at their defaults)
    "nstcalc500": ({"nstcalcenergy": 500}, None, True),
    "nstcalc1000": ({"nstcalcenergy": 1000}, None, True),
    # ... and with every global-communication interval raised together (coupling every N steps:
    # outside the recommended tau/nst ratio, timing only)
    "nstglob500": ({"nstcalcenergy": 500, "nstcomm": 500, "nsttcouple": 500, "nstpcouple": 500}, None, False),
    "nstglob1000": ({"nstcalcenergy": 1000, "nstcomm": 1000, "nsttcouple": 1000, "nstpcouple": 1000}, None, False),
    # ablations (timing only)
    "nocmap": ({}, "strip_cmap", False),
    "potshift": ({"vdw-modifier": "Potential-shift"}, None, False),
    "rf": ({"coulombtype": "Reaction-field", "epsilon-rf": 0}, None, False),
    "nolincs": ({"constraints": "none"}, None, False),
    "noconstr": ({"constraints": "none"}, "flexible_water", False),
    "nocmap-potshift": ({"vdw-modifier": "Potential-shift"}, "strip_cmap", False),
}

# fixed PME points along the tuner's curve (rcoulomb, fourierspacing scaled together; rvdw stays 1.2)
for rc, fs in ((1.2, 0.12), (1.25, 0.125), (1.3, 0.13), (1.35, 0.135), (1.4, 0.14)):
    INPUTS[f"pme-rc{rc:.2f}"] = ({"rcoulomb": rc, "fourierspacing": fs}, None, True)


def strip_cmap(top_dir: Path) -> str:
    n = 0
    for itp in sorted((top_dir / "toppar").glob("PRO*.itp")):
        out, skip = [], False
        for line in itp.read_text().splitlines(keepends=True):
            if re.match(r"^\s*\[\s*cmap\s*\]", line):
                skip = True
                continue
            if skip and re.match(r"^\s*[\[#]", line):
                skip = False
            if skip:
                if re.match(r"^\s*\d", line):
                    n += 1
                continue
            out.append(line)
        itp.write_text("".join(out))
    return f"removed {n} [ cmap ] entries from toppar/PRO*.itp"


def flexible_water(top_dir: Path) -> str:
    """Replace TIP3 SETTLE by flexible bonds/angle (CHARMM TIP3P flexible parameters)."""
    p = top_dir / "toppar" / "TIP3.itp"
    txt = p.read_text()
    flex = ("[ bonds ]\n; i j funct b0 kb\n    1     2     1  0.09572  376560.0\n    1     3     1  0.09572  376560.0\n\n"
            "[ angles ]\n; i j k funct theta k\n    2     1     3     1  104.52  460.24\n")
    new = re.sub(r"\[\s*settles\s*\][^\[]*", flex + "\n", txt, count=1)
    if new == txt:
        raise RuntimeError("no [ settles ] in TIP3.itp")
    p.write_text(new)
    return "TIP3 SETTLE replaced by flexible bonds + angle"


TRANSFORMS = {"strip_cmap": strip_cmap, "flexible_water": flexible_water}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", type=Path, required=True)
    ap.add_argument("--gmx", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args(argv)
    base_mdp = parse_mdp((a.system / "md.mdp").read_text())
    for name, (over, transform, physical) in INPUTS.items():
        if a.only and name not in a.only:
            continue
        d = a.out / name
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        shutil.copy(a.system / "topol.top", d / "topol.top")
        shutil.copytree(a.system / "toppar", d / "toppar")
        note = ""
        if transform:
            note = TRANSFORMS[transform](d)
        mdp = merge_mdp(base_mdp, CONTINUE, over)
        (d / "grompp.mdp").write_text(format_mdp(mdp))
        cmd = [str(a.gmx), "-quiet", "grompp", "-f", "grompp.mdp", "-c", str(a.system / "conf.gro"),
               "-r", str(a.system / "conf.gro"), "-n", str(a.system / "index.ndx"), "-p", "topol.top",
               "-o", "topol.tpr", "-po", "mdout.mdp", "-maxwarn", "5"]
        res = subprocess.run(cmd, cwd=d, capture_output=True, text=True)
        (d / "grompp.log").write_text(f"$ {' '.join(cmd)}\n# rc={res.returncode}\n{res.stdout}\n{res.stderr}")
        warn = re.findall(r"^WARNING \d+.*?\n(.*?)\n\n", res.stderr, re.M | re.S)
        readme = (TIMING_ONLY if not physical else "Physical variant of the production input.\n")
        readme += f"mdp overrides: {over}\ntopology: {note or 'unchanged'}\ngrompp warnings: {len(warn)}\n"
        (d / "README").write_text(readme)
        # the topology copy is only needed for grompp; keep the directory small
        shutil.rmtree(d / "toppar")
        print(f"{name:<18} rc={res.returncode} {'' if physical else '[timing only] '}{note} warnings={len(warn)}")
        if res.returncode != 0:
            print(res.stderr[-2000:])


if __name__ == "__main__":
    main()
