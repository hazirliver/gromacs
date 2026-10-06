"""Interleaved repeated mdrun measurements for profiling experiments (single build or many variants).

    bench/profiling/prun EXPERIMENT.toml --out DIR [--repeats N] [--only GLOB ...] [--dry-run]

Unlike `gmxbench perf` (an A/B comparison of two builds), an experiment here is a list of *variants*
(build x tpr x mdrun arguments x environment) measured in rounds; in each round every variant runs once,
in a seeded random order, so slow drifts (thermals, background load) spread over all variants.
Each run uses gmxbench's mdrun wrapper (counter reset via -resetstep, NVML telemetry over the timed
window) and appends one JSON record to DIR/runs.jsonl; md.log, stderr and the telemetry CSV stay in
DIR/runs/<variant>/rep<k>/. `summarize.py` turns runs.jsonl into statistics.

Experiment file:
    [defaults]                       # applied to every variant (variant keys override)
    build = "P"                      # key into [builds]
    tpr = "prod"                     # key into [tprs]
    args = "-ntmpi 1 -ntomp 16 ..."
    nsteps = 20000                   # total steps (incl. resetstep)
    resetstep = 5000
    env = { }
    pre = ""                         # optional shell command run before each run of the variant
    post = ""                        # ... and after it (e.g. restore a power limit)
    [builds]  P = "/path/to/build"   # directories containing bin/gmx
    [tprs]    prod = "/path/to/topol.tpr"
    [[variant]] name = "t16" args = "..."
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import os
import random
import subprocess
import sys
import time
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gmxbench import builds as gb  # noqa: E402
from gmxbench.gmx import mdrun, split_args  # noqa: E402
from gmxbench.hostinfo import snapshot  # noqa: E402

SW_POWER_CAP = 0x4
HW_SLOWDOWN = 0x8 | 0x40 | 0x80
SW_THERMAL = 0x20


def window_from_csv(csv_path: Path, t0: float | None, t1: float | None) -> dict:
    """Telemetry statistics over [t0, t1] (seconds since run start) from gpu_telemetry.csv,
    including what fraction of samples had each throttle reason active (gmxbench only ORs them)."""
    import numpy as np
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            t = float(r["t"])
            if (t0 is None or t >= t0) and (t1 is None or t <= t1):
                rows.append(r)
    if len(rows) < 2:
        return {}

    def col(k):
        return np.array([float(r[k]) for r in rows if r.get(k) not in ("", None, "None")], dtype=float)

    out = {"n_samples": len(rows)}
    for k in ("sm_clock", "mem_clock", "power", "util_gpu", "util_mem", "temp", "pcie_tx", "pcie_rx"):
        v = col(k)
        if len(v):
            out[f"{k}_mean"] = float(v.mean())
            out[f"{k}_p05"] = float(np.percentile(v, 5))
            out[f"{k}_p50"] = float(np.percentile(v, 50))
            out[f"{k}_p95"] = float(np.percentile(v, 95))
            out[f"{k}_min"] = float(v.min())
            out[f"{k}_max"] = float(v.max())
    thr = np.array([int(float(r["throttle"] or 0)) for r in rows])
    out["frac_sw_power_cap"] = float(np.mean((thr & SW_POWER_CAP) != 0))
    out["frac_hw_slowdown"] = float(np.mean((thr & HW_SLOWDOWN) != 0))
    out["frac_sw_thermal"] = float(np.mean((thr & SW_THERMAL) != 0))
    e = col("energy")
    t = col("t")
    if len(e) >= 2:
        out["energy_j"] = float(e[-1] - e[0])
        out["duration_s"] = float(t[-1] - t[0])
    return out


def load_experiment(path: Path) -> dict:
    spec = tomllib.loads(path.read_text())
    defaults = spec.get("defaults", {})
    variants = []
    for v in spec.get("variant", []):
        merged = {**defaults, **v}
        merged["env"] = {**defaults.get("env", {}), **v.get("env", {})}
        variants.append(merged)
    spec["variants"] = variants
    return spec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--only", nargs="*", help="glob(s) selecting variant names")
    ap.add_argument("--seed", type=int, default=20261006)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-telemetry", action="store_true")
    a = ap.parse_args(argv)

    spec = load_experiment(a.experiment)
    variants = [v for v in spec["variants"] if not a.only or any(fnmatch.fnmatch(v["name"], p) for p in a.only)]
    repeats = a.repeats or int(spec.get("repeats", 5))
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "experiment.toml").write_text(a.experiment.read_text())
    blds = {k: gb.resolve("path:" + str(Path(p).expanduser()), "external", {}, label=k)
            for k, p in spec["builds"].items()}
    tprs = {k: Path(p).expanduser().resolve() for k, p in spec["tprs"].items()}
    rng = random.Random(a.seed)
    recf = open(out / "runs.jsonl", "a")
    t_start = time.time()
    for rep in range(repeats):
        order = list(variants)
        rng.shuffle(order)
        for v in order:
            name = v["name"]
            rundir = out / "runs" / name / f"rep{rep}"
            args = split_args(v.get("args", ""))
            if "-pin" not in args:
                args += ["-pin", "on"]
            if "-noconfout" not in args:
                args += ["-noconfout"]
            if v.get("tunepme", False) is False and "-tunepme" not in args and "-notunepme" not in args:
                args += ["-notunepme"]
            nsteps, resetstep = int(v["nsteps"]), int(v["resetstep"])
            bld, tpr = blds[v.get("build", "P")], tprs[v.get("tpr", "prod")]
            msg = f"[{time.time() - t_start:7.0f}s] rep {rep} {name}: {bld.gmx} {' '.join(args)} env={v['env']}"
            print(msg, file=sys.stderr, flush=True)
            if a.dry_run:
                continue
            if v.get("pre"):
                subprocess.run(v["pre"], shell=True, check=True)
            snap0 = snapshot()
            if not bld.gmx.exists():
                recf.write(json.dumps({"variant": name, "repeat": rep, "status": "error",
                                       "message": f"missing {bld.gmx}"}) + "\n")
                recf.flush()
                print(f"    -> error: missing {bld.gmx}", file=sys.stderr, flush=True)
                continue
            try:
                r = mdrun(bld, tpr, rundir, args, env=v["env"], nsteps=nsteps, resetstep=resetstep,
                          timeout=float(v.get("timeout", 1800)), telemetry=None if a.no_telemetry else True)
            finally:
                if v.get("post"):
                    subprocess.run(v["post"], shell=True, check=False)
            lg = r.log or {}
            rec = {"variant": name, "repeat": rep, "status": r.status, "message": r.message,
                   "rundir": str(rundir.relative_to(out)), "cmd": r.cmd, "env": v["env"],
                   "build": v.get("build", "P"), "tpr": v.get("tpr", "prod"), "nsteps": nsteps,
                   "resetstep": resetstep, "wall_total_s": r.wall, "snapshot": snap0,
                   "tags": v.get("tags", {})}
            for k in ("ns_per_day", "ms_per_step", "matom_steps_per_s", "core_time_s", "wall_time_s"):
                if k in lg:
                    rec[k] = lg[k]
            rec["setup"] = lg.get("setup", {})
            rec["stages"] = {k: {"ms_per_step": s["wall_s"] * 1000.0 / max(1, nsteps - resetstep), "pct": s["pct"],
                                 "count": s["count"]} for k, s in lg.get("stages", {}).items()}
            tel = r.telemetry or {}
            if tel:
                csvp = rundir / "gpu_telemetry.csv"
                t0 = tel.get("reset_at_s")
                rec["telemetry"] = window_from_csv(csvp, t0, None) if csvp.exists() else {}
                w = rec["telemetry"]
                if w.get("energy_j") and rec.get("ns_per_day") and w.get("duration_s"):
                    ns_sim = rec["ns_per_day"] * w["duration_s"] / 86400.0
                    w["energy_kj_per_ns"] = w["energy_j"] / 1000.0 / ns_sim
                rec["telemetry_window_from_reset"] = t0 is not None
            recf.write(json.dumps(rec) + "\n")
            recf.flush()
            extra = ""
            if rec.get("telemetry"):
                w = rec["telemetry"]
                extra = (f" P={w.get('power_mean', 0):.0f}W clk={w.get('sm_clock_mean', 0):.0f}MHz "
                         f"cap={w.get('frac_sw_power_cap', 0):.2f} E={w.get('energy_kj_per_ns', 0):.1f}kJ/ns")
            print(f"    -> {r.status} {rec.get('ms_per_step')} ms/step {rec.get('ns_per_day')} ns/day{extra} "
                  f"{r.message[:200]}", file=sys.stderr, flush=True)
    recf.close()


if __name__ == "__main__":
    main()
