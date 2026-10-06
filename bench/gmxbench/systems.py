"""Reproducible generation of benchmark/test systems.

Each [systems.<name>] entry in the suite names a recipe plus parameters. Prepared systems live in
<workdir>/systems/<name>-<hash of recipe parameters>/ and contain:

    conf.gro  topol.top  (+ itp files, index.ndx)   grompp inputs
    base.mdp                                         optional system-provided mdp (archive recipe)
    system.json                                      metadata (natoms, recipe, preparing build, ...)

Systems are prepared once and shared by both sides of an A/B comparison. Steps that run MD
(minimisation/equilibration) use deterministic CPU settings (-reprod), so preparation is
reproducible for a given build.

Recipes
  water            replicated SPC/E, TIP3P or TIP4P box (optionally with a decoupled FEP subset)
  pdb2gmx          PDB (downloaded, sha256-verified) -> pdb2gmx -> box -> solvate -> ions -> EM -> NVT/NPT
  archive          tarball with ready inputs (e.g. a CHARMM-GUI system); optional, skipped if absent
  regressiontests  the upstream regressiontests package matching the GROMACS version (quality only)
"""

from __future__ import annotations

import os
import re
import shutil
import tarfile
import urllib.request
from pathlib import Path

from .gmx import GromppInput, grompp, mdrun, parse_mdp
from .util import REPO_ROOT, clean_env, log, read_json, run, sha256_file, stable_hash, workdir, write_json

SEED = 1993


class SystemUnavailable(Exception):
    pass


def _natoms(gro: Path) -> int:
    with open(gro) as f:
        f.readline()
        return int(f.readline())


def download(url: str, sha256: str | None = None, md5: str | None = None) -> Path:
    d = workdir() / "downloads"
    d.mkdir(parents=True, exist_ok=True)
    dst = d / url.rstrip("/").split("/")[-1]
    if not dst.exists():
        log(f"downloading {url}")
        tmp = dst.with_suffix(dst.suffix + ".part")
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.replace(dst)
    if sha256 and sha256_file(dst) != sha256:
        raise SystemUnavailable(f"sha256 mismatch for {dst}")
    if md5:
        import hashlib
        if hashlib.md5(dst.read_bytes()).hexdigest() != md5:
            raise SystemUnavailable(f"md5 mismatch for {dst}")
    return dst


def _share_top(build) -> Path:
    for cand in (build.source_dir / "share" / "top", REPO_ROOT / "share" / "top"):
        if (cand / "spc216.gro").exists():
            return cand
    raise SystemUnavailable("cannot locate share/top")


def _gmx(build, *args, cwd: Path, stdin: str | None = None, name: str = "gmx"):
    res = run([build.gmx, "-quiet", *args], cwd=cwd, env=clean_env(), stdin=stdin,
              log_to=cwd / f"prep-{name}.log")
    if not res.ok:
        raise RuntimeError(f"gmx {args[0]} failed in {cwd}; see {cwd}/prep-{name}.log")
    return res


def _prep_threads() -> int:
    return max(1, min(16, (os.cpu_count() or 2) // 2))


def _cpu_md(build, d: Path, mdp: dict, conf: str, top: str, out: str, ref: str | None = None,
            maxwarn: int = 1):
    tpr = d / f"{out}.tpr"
    grompp(build, GromppInput(d / conf, d / top, mdp, ref=(d / ref) if ref else None, maxwarn=maxwarn), tpr, cwd=d)
    r = mdrun(build, tpr, d / f"{out}-run", ["-ntmpi", "1", "-ntomp", str(_prep_threads()), "-nb", "cpu", "-pme",
                                             "cpu", "-bonded", "cpu", "-update", "cpu", "-reprod"], timeout=7200)
    if not r.ok:
        raise RuntimeError(f"preparation MD '{out}' failed: {r.message}")
    shutil.copy(r.rundir / "run.gro", d / f"{out}.gro")
    return d / f"{out}.gro"


# ----------------------------------------------------------------------------------------------
# recipes
# ----------------------------------------------------------------------------------------------
def _recipe_water(build, d: Path, p: dict) -> dict:
    share = _share_top(build)
    model = p.get("model", "spce")
    ff = p.get("ff", "oplsaa")
    src = {"spce": "spc216.gro", "spc": "spc216.gro", "tip3p": "spc216.gro", "tip4p": "tip4p.gro"}[model]
    n = int(p["nbox"])
    _gmx(build, "genconf", "-f", share / src, "-nbox", n, n, n, "-o", "conf.gro", cwd=d, name="genconf")
    sites = 4 if model == "tip4p" else 3
    nmol = _natoms(d / "conf.gro") // sites
    top = [f'#include "{ff}.ff/forcefield.itp"', f'#include "{ff}.ff/{model}.itp"']
    mols = []
    nfep = int(p.get("fep_molecules", 0))
    if nfep:
        itp = (share / f"{ff}.ff" / f"{model}.itp").read_text()
        itp = re.sub(r"(\[ *moleculetype *\][^\n]*\n(?:;[^\n]*\n)*)\s*SOL\b", r"\1SOLF", itp, count=1)
        # grompp allows only one molecule block with [settles]: the perturbed copies use the flexible
        # variant (O-H bonds become LINCS constraints with constraints = h-bonds, the angle stays flexible).
        itp = re.sub(r"#ifndef FLEXIBLE.*?#else\n", "", itp, flags=re.S).replace("#endif", "")
        (d / "solf.itp").write_text(itp)
        top.append('#include "solf.itp"')
        mols.append(("SOLF", nfep))
    mols.append(("SOL", nmol - nfep))
    top += ["", "[ system ]", f"{model} water x{n ** 3}", "", "[ molecules ]"]
    top += [f"{m:<8}{c}" for m, c in mols]
    (d / "topol.top").write_text("\n".join(top) + "\n")
    # no velocities in the replicated box: generate them deterministically
    return {"base_mdp": {"gen-vel": "yes", "gen-temp": 300, "gen-seed": SEED}}


_EM_MDP = {"integrator": "steep", "emtol": 1000.0, "emstep": 0.01, "nsteps": 5000, "cutoff-scheme": "Verlet",
           "coulombtype": "PME", "rcoulomb": 1.0, "rvdw": 1.0, "nstlist": 10}


def _recipe_pdb2gmx(build, d: Path, p: dict) -> dict:
    share = _share_top(build)
    pdb = download(p["pdb_url"], sha256=p.get("pdb_sha256"))
    keep = [l for l in pdb.read_text().splitlines() if l.startswith(("ATOM", "TER", "END"))]
    (d / "protein.pdb").write_text("\n".join(keep) + "\n")
    args = ["pdb2gmx", "-f", "protein.pdb", "-o", "processed.gro", "-p", "topol.top", "-ff", p.get("ff", "amber99sb-ildn"),
            "-water", p.get("water", "tip3p"), "-ignh"]
    if p.get("vsite"):
        args += ["-vsite", p["vsite"]]
    _gmx(build, *args, cwd=d, name="pdb2gmx")
    _gmx(build, "editconf", "-f", "processed.gro", "-o", "box.gro", "-c", "-d", p.get("distance", 1.0), "-bt",
         p.get("boxtype", "dodecahedron"), cwd=d, name="editconf")
    _gmx(build, "solvate", "-cp", "box.gro", "-cs", share / "spc216.gro", "-o", "solv.gro", "-p", "topol.top",
         cwd=d, name="solvate")
    grompp(build, GromppInput(d / "solv.gro", d / "topol.top", _EM_MDP, maxwarn=2), d / "ions.tpr", cwd=d)
    _gmx(build, "genion", "-s", "ions.tpr", "-o", "ions.gro", "-p", "topol.top", "-pname", "NA", "-nname", "CL",
         "-neutral", "-conc", p.get("salt", 0.15), "-seed", SEED, cwd=d, stdin="SOL\n", name="genion")
    _cpu_md(build, d, _EM_MDP, "ions.gro", "topol.top", "em")
    dt = 0.002
    common = {"integrator": "md", "dt": dt, "cutoff-scheme": "Verlet", "coulombtype": "PME", "rcoulomb": 1.0,
              "rvdw": 1.0, "dispcorr": "EnerPres", "constraints": "h-bonds", "tcoupl": "v-rescale",
              "tc-grps": "Protein Non-Protein", "tau-t": "0.1 0.1", "ref-t": "300 300", "ld-seed": SEED,
              "define": "-DPOSRES", "nstenergy": 500, "nstlog": 500}
    nvt = {**common, "nsteps": int(p.get("nvt_ps", 20) / dt), "gen-vel": "yes", "gen-temp": 300, "gen-seed": SEED,
           "pcoupl": "no"}
    _cpu_md(build, d, nvt, "em.gro", "topol.top", "nvt", ref="em.gro")
    npt = {**common, "nsteps": int(p.get("npt_ps", 20) / dt), "continuation": "yes", "gen-vel": "no",
           "pcoupl": "C-rescale", "pcoupltype": "isotropic", "tau-p": 2.0, "ref-p": 1.0, "compressibility": 4.5e-5,
           "refcoord-scaling": "com"}
    _cpu_md(build, d, npt, "nvt.gro", "topol.top", "npt", ref="nvt.gro")
    shutil.copy(d / "npt.gro", d / "conf.gro")
    return {}


def archive_path(p: dict) -> Path:
    """Archive location; `path_env` names an environment variable that overrides `path`."""
    raw = os.environ.get(p["path_env"]) if p.get("path_env") else None
    path = Path(raw or p["path"]).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def unavailable_reason(suite, name: str) -> str | None:
    """Cheap pre-check (no preparation) of whether a system's inputs can be found."""
    spec = suite.systems.get(name)
    if spec is None:
        return f"system '{name}' is not defined in the suite"
    if spec.get("recipe") == "archive":
        path = archive_path(spec)
        if not path.exists():
            hint = f" (or set ${spec['path_env']})" if spec.get("path_env") else ""
            return f"input archive {path} not found{hint}"
    return None


def _recipe_archive(build, d: Path, p: dict) -> dict:
    path = archive_path(p)
    if not path.exists():
        raise SystemUnavailable(f"archive {path} not found")
    with tarfile.open(path) as t:
        t.extractall(d, filter="data")
    if p["gro"] != "conf.gro":
        shutil.copy(d / p["gro"], d / "conf.gro")
    if p.get("top", "topol.top") != "topol.top":
        shutil.copy(d / p["top"], d / "topol.top")
    meta = {}
    if p.get("ndx"):
        meta["ndx"] = p["ndx"]
    if p.get("mdp"):
        meta["base_mdp"] = parse_mdp((d / p["mdp"]).read_text())
    return meta


RECIPES = {"water": _recipe_water, "pdb2gmx": _recipe_pdb2gmx, "archive": _recipe_archive}
# Bump when a recipe's output changes, so cached prepared systems are regenerated.
RECIPE_VERSION = {"water": 2, "pdb2gmx": 1, "archive": 1}


# ----------------------------------------------------------------------------------------------
class PreparedSystem:
    def __init__(self, name: str, path: Path):
        self.name = name
        self.dir = path
        self.meta = read_json(path / "system.json")

    @property
    def conf(self) -> Path:
        return self.dir / "conf.gro"

    @property
    def top(self) -> Path:
        return self.dir / "topol.top"

    @property
    def ndx(self) -> Path | None:
        n = self.meta.get("ndx")
        return self.dir / n if n else None

    @property
    def base_mdp(self) -> dict:
        return self.meta.get("base_mdp", {})

    @property
    def natoms(self) -> int:
        return self.meta["natoms"]

    def __repr__(self):
        return f"{self.name}({self.natoms} atoms)"


def prepare(suite, build, name: str, force: bool = False) -> PreparedSystem:
    spec = suite.systems.get(name)
    if spec is None:
        raise SystemExit(f"unknown system '{name}'")
    recipe = spec["recipe"]
    if recipe not in RECIPES:
        raise SystemExit(f"system '{name}': unknown recipe '{recipe}'")
    d = workdir() / "systems" / f"{name}-{stable_hash({**spec, '_v': RECIPE_VERSION[recipe]}, 10)}"
    if (d / "system.json").exists() and not force:
        return PreparedSystem(name, d)
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    log(f"preparing system {name} ({recipe}) in {d}")
    try:
        meta = RECIPES[recipe](build, d, dict(spec)) or {}
    except Exception:
        shutil.rmtree(d, ignore_errors=True)
        raise
    meta.update({"name": name, "recipe": recipe, "spec": spec, "natoms": _natoms(d / "conf.gro"),
                 "prepared_by": build.id, "conf_sha256": sha256_file(d / "conf.gro")})
    write_json(d / "system.json", meta)
    return PreparedSystem(name, d)


def prepare_all(suite, build, select=None, force=False) -> dict:
    import fnmatch
    out = {}
    for name in suite.systems:
        if select and not any(fnmatch.fnmatch(name, s) for s in select):
            continue
        try:
            out[name] = prepare(suite, build, name, force).dir
        except SystemUnavailable as e:
            log(f"system {name} unavailable: {e}")
    return out


# ----------------------------------------------------------------------------------------------
# upstream regressiontests (used as broad-coverage quality cases)
# ----------------------------------------------------------------------------------------------
def regressiontests_dir(build) -> Path:
    ver = build.info().get("version", {}).get("GROMACS version", "")
    m = re.match(r"(\d{4}(?:\.\d+)?)", ver)
    if not m:
        raise SystemUnavailable(f"cannot determine GROMACS version from '{ver}'")
    version = m.group(1)
    md5 = None
    vinfo = build.source_dir / "cmake" / "gmxVersionInfo.cmake"
    if vinfo.exists():
        mm = re.search(r'set\(REGRESSIONTEST_MD5SUM "([0-9a-f]+)"', vinfo.read_text())
        md5 = mm.group(1) if mm else None
    dst = workdir() / "regressiontests" / f"regressiontests-{version}"
    if not dst.exists():
        tgz = download(f"https://ftp.gromacs.org/regressiontests/regressiontests-{version}.tar.gz", md5=md5)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tgz) as t:
            t.extractall(dst.parent, filter="data")
    return dst
