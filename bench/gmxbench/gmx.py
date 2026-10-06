"""Running gmx tools: mdp handling, grompp, mdrun (optionally under Nsight Systems), tpr comparison."""

from __future__ import annotations

import csv
import re
import shlex
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .logparse import classify_failure, fatal_message, parse_log
from .util import clean_env, run, sha256_file, stable_hash, workdir


# ----------------------------------------------------------------------------------------------
# mdp files
# ----------------------------------------------------------------------------------------------
def mdp_key(k: str) -> str:
    """grompp treats '-' and '_' as equivalent; normalise so merges cannot define a key twice."""
    return k.strip().lower().replace("_", "-")


def parse_mdp(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        line = line.split(";", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            if k.strip():
                out[mdp_key(k)] = v.strip()
    return out


def normalize_mdp(params: dict) -> dict:
    out = {}
    for k, v in params.items():
        out[mdp_key(k)] = v
    return out


def merge_mdp(*dicts: dict) -> dict:
    out: dict = {}
    for d in dicts:
        out.update(normalize_mdp(d or {}))
    return out


def format_mdp(params: dict) -> str:
    lines = []
    for k, v in params.items():
        if isinstance(v, bool):
            v = "yes" if v else "no"
        elif isinstance(v, (list, tuple)):
            v = " ".join(str(x) for x in v)
        lines.append(f"{k:<24} = {v}")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------------------------
# grompp
# ----------------------------------------------------------------------------------------------
@dataclass
class GromppInput:
    conf: Path
    top: Path
    mdp: dict
    ndx: Path | None = None
    ref: Path | None = None            # -r (position restraint reference)
    maxwarn: int = 0
    extra: list = field(default_factory=list)
    cpt: Path | None = None             # -t

    def key(self) -> str:
        return stable_hash({"conf": sha256_file(self.conf), "top": str(self.top), "topsha": sha256_file(self.top),
                            "mdp": self.mdp, "ndx": sha256_file(self.ndx) if self.ndx else None,
                            "ref": str(self.ref) if self.ref else None, "maxwarn": self.maxwarn,
                            "extra": self.extra}, 16)


def grompp(build, gi: GromppInput, out_tpr: Path, cwd: Path | None = None, env: dict | None = None):
    cwd = cwd or out_tpr.parent
    cwd.mkdir(parents=True, exist_ok=True)
    mdp_path = out_tpr.with_suffix(".mdp")
    mdp_path.write_text(format_mdp(gi.mdp))
    cmd = [build.gmx, "-quiet", "grompp", "-f", mdp_path, "-c", gi.conf, "-p", gi.top, "-o", out_tpr,
           "-po", out_tpr.with_name(out_tpr.stem + "-mdout.mdp"), "-maxwarn", str(gi.maxwarn)]
    if gi.ndx:
        cmd += ["-n", gi.ndx]
    if gi.ref:
        cmd += ["-r", gi.ref]
    if gi.cpt:
        cmd += ["-t", gi.cpt]
    cmd += list(gi.extra)
    res = run(cmd, cwd=cwd, env=clean_env(env), log_to=out_tpr.with_suffix(".grompp.log"))
    if not res.ok or not out_tpr.exists():
        raise RuntimeError(f"grompp failed for {out_tpr}: {fatal_message(res.stderr + res.stdout)}")
    nwarn = len(re.findall(r"^WARNING \d+", res.stderr, re.M))
    nnote = len(re.findall(r"^NOTE \d+", res.stderr, re.M))
    return {"warnings": nwarn, "notes": nnote}


def cached_tpr(build, gi: GromppInput, tag: str) -> Path:
    """grompp once per (inputs, build); both sides of an A/B comparison use the baseline's tpr."""
    d = workdir() / "tpr" / f"{tag}-{gi.key()}-{build.id}"
    tpr = d / "topol.tpr"
    if not tpr.exists():
        d.mkdir(parents=True, exist_ok=True)
        grompp(build, gi, tpr)
    return tpr


def _tpr_body(path: Path) -> bytes:
    """tpr bytes after the XDR-encoded "VERSION <gromacs version>" header string (which differs between
    builds, e.g. "-dirty" for a modified tree, without the content differing)."""
    data = Path(path).read_bytes()
    i = data.find(b"VERSION ", 0, 512)
    if i < 4:
        return data
    n = int.from_bytes(data[i - 4:i], "big")
    return data[i + (n + 3) // 4 * 4:]


def compare_tpr(build, tpr1: Path, tpr2: Path) -> dict:
    """Bitwise content comparison of two run input files, ignoring only the version header.
    On a difference, `gmx check` (zero tolerance) lists what differs."""
    if sha256_file(tpr1) == sha256_file(tpr2):
        return {"identical": True, "byte_identical": True, "differences": [], "n_differences": 0}
    if _tpr_body(tpr1) == _tpr_body(tpr2):
        return {"identical": True, "byte_identical": False, "differences": [], "n_differences": 0,
                "note": "identical apart from the GROMACS version string in the header"}
    res = run([build.gmx, "-quiet", "check", "-s1", tpr1, "-s2", tpr2, "-tol", "0", "-abstol", "0"],
              env=clean_env())
    text = res.stdout
    diffs = [l for l in text.splitlines()
             if l.strip() and not l.startswith(("Reading", "Note", "comparing", "Comparing", "WARNING"))
             and "precision" not in l and "VERSION" not in l and not l.startswith("Command line")]
    # the bodies differ, so the files differ even if gmx check cannot say where (e.g. pull parameters)
    return {"identical": False, "byte_identical": False, "differences": diffs[:50] or ["binary content differs"],
            "n_differences": len(diffs) or 1}


# ----------------------------------------------------------------------------------------------
# mdrun
# ----------------------------------------------------------------------------------------------
_BONDED_NA = "None of the bonded types are implemented on the GPU"


@dataclass
class MdrunResult:
    status: str                 # ok | unsupported | error | timeout
    rundir: Path
    wall: float
    message: str = ""
    log: dict = field(default_factory=dict)
    cmd: str = ""
    fallback: str | None = None
    marks: dict = field(default_factory=dict)       # monotonic times: start, reset (counter reset), end
    telemetry: dict = field(default_factory=dict)   # GPU telemetry summaries (window = timed part of the run)

    @property
    def ok(self) -> bool:
        return self.status == "ok"


_RESET_MSG = b"resetting all time and cycle counters"


def _run_live(cmd, cwd: Path, env: dict, timeout: float, log_to: Path, monitor=None):
    """Like util.run, but streams stderr to timestamp mdrun's counter reset as it happens, and
    registers the process with the GPU monitor (per-process VRAM)."""
    import subprocess
    import threading
    import time as _time
    from .util import ProcResult
    cmd = [str(c) for c in cmd]
    marks = {"start": _time.monotonic()}
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if monitor is not None:
        monitor.watch_pid(p.pid)
    out_b, err_b = [], []

    def rd_out():
        for chunk in iter(lambda: p.stdout.read1(65536), b""):
            out_b.append(chunk)

    def rd_err():
        tail = b""
        for chunk in iter(lambda: p.stderr.read1(65536), b""):
            err_b.append(chunk)
            tail = (tail + chunk)[-8192:]
            if "reset" not in marks and _RESET_MSG in tail:
                marks["reset"] = _time.monotonic()

    th = [threading.Thread(target=rd_out, daemon=True), threading.Thread(target=rd_err, daemon=True)]
    for t in th:
        t.start()
    timed_out = False
    try:
        rc = p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        rc, timed_out = -9, True
        p.wait()
    marks["end"] = _time.monotonic()
    for t in th:
        t.join(timeout=10)
    out = b"".join(out_b).decode(errors="replace")
    err = b"".join(err_b).decode(errors="replace")
    wall = marks["end"] - marks["start"]
    log_to.write_text(f"$ {' '.join(cmd)}\n# rc={rc} wall={wall:.2f}s timed_out={timed_out}\n"
                      f"--- stdout ---\n{out}\n--- stderr ---\n{err}\n")
    return ProcResult(cmd, rc, out, err, wall, timed_out), marks


def mdrun(build, tpr: Path, rundir: Path, args: list[str], env: dict | None = None, nsteps: int | None = None,
          resetstep: int | None = None, extra: list[str] | None = None, timeout: float = 3600,
          nsys: bool = False, telemetry=None) -> MdrunResult:
    """Run mdrun in `rundir`.

    telemetry: None (off), True (sample the GPUs during this run), or a running telemetry.GpuMonitor
    shared by several concurrent runs (the caller then reduces the samples itself).
    """
    rundir.mkdir(parents=True, exist_ok=True)
    own_monitor = None
    monitor = None
    if telemetry is not None and telemetry is not False and not nsys:
        from . import telemetry as tm
        if telemetry is True:
            if tm.available():
                own_monitor = monitor = tm.GpuMonitor().start()
        else:
            monitor = telemetry
    marks: dict = {}
    args = list(args)
    base = [build.gmx, "-quiet", "mdrun", "-s", tpr, "-deffnm", "run"]
    run_env = clean_env({**build.env, **(env or {})})
    fallback = None
    for attempt in range(5):
        tail = list(extra or [])
        if nsteps is not None:
            tail += ["-nsteps", str(nsteps)]
        if resetstep is not None:
            tail += ["-resetstep", str(resetstep)]
        cmd = base + args + tail
        if nsys:
            cmd = ["nsys", "profile", "-o", rundir / "prof", "--force-overwrite", "true", "--trace", "cuda,nvtx",
                   "--sample", "none", "--cpuctxsw", "none", "--cuda-graph-trace", "node", *cmd]
        if monitor is not None:
            res, marks = _run_live(cmd, rundir, run_env, timeout, rundir / "mdrun.out", monitor)
        else:
            res = run(cmd, cwd=rundir, env=run_env, timeout=timeout, log_to=rundir / "mdrun.out")
        text = res.stderr + "\n" + res.stdout
        # A system without any GPU-capable bonded interaction cannot use -bonded gpu; that is
        # physically equivalent to -bonded cpu, so retry once instead of reporting it unsupported.
        if not res.ok and _BONDED_NA in text and "-bonded" in args:
            i = args.index("-bonded")
            args[i + 1] = "cpu"
            fallback = "bonded=cpu (no GPU-capable bonded types)"
            continue
        # PME load balancing (-tunepme) can outlast the counter reset on fast systems: reset later,
        # keeping the length of the timed window.
        if not res.ok and "PME tuning was still active" in text and resetstep is not None and nsteps is not None:
            window = nsteps - resetstep
            resetstep *= 2
            nsteps = resetstep + window
            fallback = f"resetstep raised to {resetstep} (PME tuning still active)"
            continue
        break
    logp = rundir / "run.log"
    parsed = parse_log(logp) if logp.exists() else {}
    if res.timed_out:
        status, msg = "timeout", f"timed out after {timeout}s"
    elif res.ok:
        status, msg = "ok", ""
    else:
        status, msg = classify_failure(text), fatal_message(text)
    result = MdrunResult(status, rundir, res.wall, msg, parsed, " ".join(shlex.quote(str(c)) for c in cmd), fallback,
                         marks)
    if own_monitor is not None:
        own_monitor.stop()
        result.telemetry = window_telemetry(own_monitor, marks, parsed)
        own_monitor.write_csv(rundir / "gpu_telemetry.csv", marks.get("start"))
    return result


def window_telemetry(monitor, marks: dict, parsed: dict | None = None, t0: float | None = None,
                     t1: float | None = None) -> dict:
    """Telemetry over mdrun's timed window (counter reset -> end) and over the whole run, plus
    efficiency metrics derived from the run's own performance numbers."""
    w0 = t0 if t0 is not None else marks.get("reset", marks.get("start"))
    w1 = t1 if t1 is not None else marks.get("end")
    win = monitor.summary(w0, w1)
    tel = {"window": win, "run": monitor.summary(marks.get("start"), marks.get("end")),
           "window_from_reset": "reset" in marks or t0 is not None,
           "series": monitor.timeseries(marks.get("start", 0), marks.get("end", 0), marks.get("start", 0)),
           "reset_at_s": (marks["reset"] - marks["start"]) if "reset" in marks and "start" in marks else None}
    if parsed and win and parsed.get("ns_per_day") and win.get("duration_s"):
        nsd = parsed["ns_per_day"]
        ns_simulated = nsd * win["duration_s"] / 86400.0
        if win.get("energy_j") and ns_simulated > 0:
            tel["energy_kj_per_ns"] = win["energy_j"] / 1000.0 / ns_simulated
        if win.get("power_mean"):
            tel["ns_per_day_per_kw"] = nsd / (win["power_mean"] / 1000.0)
    if parsed and parsed.get("core_time_s") and parsed.get("wall_time_s"):
        tel["cpu_cores_busy"] = parsed["core_time_s"] / parsed["wall_time_s"]
    return tel


# ----------------------------------------------------------------------------------------------
# Nsight Systems kernel statistics
# ----------------------------------------------------------------------------------------------
def _short_kernel(name: str) -> str:
    n = name.strip().strip('"')
    if n.startswith("void "):
        n = n[5:]
    # drop the parameter list: last top-level '(...)'
    depth = 0
    for i in range(len(n) - 1, -1, -1):
        c = n[i]
        if c == ")":
            depth += 1
        elif c == "(":
            depth -= 1
            if depth == 0:
                n = n[:i]
                break
    n = n.replace("gmx::", "")
    n = re.sub(r"\((?:unsigned int|int|bool|padding_t|twiddle_t|loadstore_modifier_t|layout_t|ThreadsPerAtom|GridOrdering)\)", "", n)
    n = re.sub(r"\s+", " ", n)
    if len(n) > 70:
        n = n[:60] + "…#" + stable_hash(name, 6)
    return n


def nsys_kernel_stats(rundir: Path, keep_report: bool = False) -> dict:
    """Return {kernel: {instances, total_ns, avg_ns, med_ns, ...}} and memcpy stats; clean up big files."""
    rep = rundir / "prof.nsys-rep"
    out = {"kernels": {}, "memops": {}}
    if not rep.exists():
        return out
    run(["nsys", "stats", "--report", "cuda_gpu_kern_sum", "--report", "cuda_gpu_mem_time_sum", "--format", "csv",
         "--force-export", "true", "--output", rundir / "nsys", rep], cwd=rundir, timeout=600)
    for kind, fname in (("kernels", "nsys_cuda_gpu_kern_sum.csv"), ("memops", "nsys_cuda_gpu_mem_time_sum.csv")):
        p = rundir / fname
        if not p.exists():
            continue
        with open(p) as f:
            for row in csv.DictReader(f):
                key = _short_kernel(row.get("Name") or row.get("Operation") or "?")
                ent = out[kind].setdefault(key, {"instances": 0, "total_ns": 0.0, "med_ns": 0.0, "full_name": ""})
                # identical short names (template variants) are aggregated
                ent["instances"] += int(float(row["Instances"] if "Instances" in row else row.get("Count", 0)))
                ent["total_ns"] += float(row["Total Time (ns)"])
                ent["med_ns"] = max(ent["med_ns"], float(row["Med (ns)"]))
                ent["full_name"] = ent["full_name"] or (row.get("Name") or row.get("Operation") or "")
    if not keep_report:
        for p in (rep, rundir / "prof.sqlite"):
            if p.exists():
                p.unlink()
    return out


def nsys_available() -> bool:
    return shutil.which("nsys") is not None


def split_args(s) -> list[str]:
    if isinstance(s, list):
        return [str(x) for x in s]
    return shlex.split(s or "")
