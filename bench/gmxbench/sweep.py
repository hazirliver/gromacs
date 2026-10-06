"""Per-system search over mdrun run arguments: "which arguments should I use for this use case?"

The search is staged (a greedy coordinate descent), which keeps it to tens of runs per system:

  1. default      plain `gmx mdrun` (what users get without tuning)  -> reference
  2. offload      CPU-only rank/thread layouts and GPU offload modes x OpenMP threads per rank
  3. nstlist      pair-list update interval for the best GPU configuration
  4. graphs       CUDA graphs on/off (GPU-resident only)
  5. tunepme      PME load-balancing on/off
  6. throughput   N independent simulations sharing the node/GPU (aggregate ns/day)

Each point is measured `repeats` times with the run length calibrated to ~target_seconds.
The result is a ranked table per system plus a recommendation (best single-simulation and best
throughput configuration, and their gain over the default).
"""

from __future__ import annotations

import os
import threading

import numpy as np

from .gmx import mdrun
from .hostinfo import n_gpus, physical_cores
from .perf import TELEMETRY_METRICS, _round_steps, perf_case_inputs
from .systems import SystemUnavailable
from .util import log

OFFLOAD = {
    "cpu": ["-nb", "cpu", "-pme", "cpu", "-bonded", "cpu", "-update", "cpu"],
    "gpu-nb": ["-nb", "gpu", "-pme", "cpu", "-bonded", "cpu", "-update", "cpu"],
    "gpu-nb-pme": ["-nb", "gpu", "-pme", "gpu", "-bonded", "cpu", "-update", "cpu"],
    "gpu-nb-pme-bonded": ["-nb", "gpu", "-pme", "gpu", "-bonded", "gpu", "-update", "cpu"],
    "gpu-resident": ["-nb", "gpu", "-pme", "gpu", "-bonded", "gpu", "-update", "gpu"],
    "gpu-resident-bondedcpu": ["-nb", "gpu", "-pme", "gpu", "-bonded", "cpu", "-update", "gpu"],
}


class _Point:
    def __init__(self, label, stage, args, env=None, conc=1, tunepme=False):
        self.label, self.stage, self.args, self.env, self.conc, self.tunepme = label, stage, args, env or {}, conc, tunepme
        self.values: list = []       # aggregate ns/day per repeat
        self.per_sim: list = []      # ns/day of individual simulations
        self.status = "ok"
        self.message = ""
        self.setup = {}
        self.tel: list = []          # telemetry metric dicts per repeat

    @property
    def median(self):
        return float(np.median(self.values)) if self.values else 0.0

    def as_dict(self):
        v = np.asarray(self.values) if self.values else np.asarray([np.nan])
        return {"label": self.label, "stage": self.stage, "args": " ".join(self.args),
                "env": self.env, "concurrency": self.conc, "tunepme": self.tunepme, "status": self.status,
                "message": self.message, "median": self.median, "values": self.values,
                "cv": float(v.std(ddof=1) / v.mean()) if len(self.values) > 1 else None,
                "per_sim_median": float(np.median(self.per_sim)) if self.per_sim else None, "setup": self.setup,
                "telemetry": {k: float(np.median([t[k] for t in self.tel if t.get(k) is not None]))
                              for k in sorted({k for t in self.tel for k in t})
                              if any(t.get(k) is not None for t in self.tel)}}


def run_sweep(sess, suite, build, args) -> dict:
    sw = suite.section("sweep")
    repeats = args.repeats or int(suite.tv(sw.get("repeats", 2)))
    target_s = float(suite.tv(sw.get("target_seconds", 6)))
    timeout = float(sw.get("timeout", 1800))
    dims = sw.get("dims", {})
    cores = physical_cores()
    hw = os.cpu_count() or cores
    have_gpu = n_gpus() > 0
    summary = sess.meta["summary"].get("sweep", {"settings": {"repeats": repeats, "target_seconds": target_s},
                                                  "systems": []})
    selection = getattr(args, "systems", None)
    for ent in sw.get("systems", []):
        if not suite.selected(ent, selection):
            continue
        try:
            ps, tpr = perf_case_inputs(suite, build, ent)
        except (SystemUnavailable, RuntimeError) as e:
            log(f"sweep {ent['name']}: skipped ({e})")
            continue
        log(f"sweep {ent['name']} ({ps.natoms} atoms)")
        use_tel = have_gpu and str(sw.get("telemetry", "on")) != "off"
        base = sess.runs / "sweep" / ent["name"].replace("/", "_")
        calib = {}
        points: list[_Point] = []

        def measure(pt: _Point):
            # calibrate run length on first use of this argument set
            key = (tuple(pt.args), tuple(sorted(pt.env.items())), pt.conc, pt.tunepme)
            if key not in calib:
                r, _ = _launch(build, tpr, base / f"{len(points):03d}-calib", pt, int(sw.get("calib_steps", 1000)),
                               None, timeout, telemetry=False)
                if not all(x.ok for x in r):
                    bad = next(x for x in r if not x.ok)
                    pt.status, pt.message = bad.status, bad.message
                    log(f"  {pt.label:<44} {bad.status}: {bad.message[:90]}")
                    points.append(pt)
                    return pt
                ms = max(x.log.get("ms_per_step") or 1.0 for x in r)
                n = _round_steps(int(target_s * 1000 / max(ms, 1e-3)))
                n = min(max(n, int(sw.get("min_steps", 2000))), int(sw.get("max_steps", 1_000_000)))
                calib[key] = n
            n = calib[key]
            # PME tuning needs time to finish before the counters are reset
            reset = _round_steps(max(n // 4, 500)) if not pt.tunepme else _round_steps(max(n // 2, 3000))
            for rep in range(repeats):
                r, tel = _launch(build, tpr, base / f"{len(points):03d}-rep{rep}", pt, n + reset, reset, timeout,
                                 telemetry=use_tel)
                if not all(x.ok for x in r):
                    bad = next(x for x in r if not x.ok)
                    pt.status, pt.message = bad.status, bad.message
                    break
                sims = [x.log.get("ns_per_day") or 0.0 for x in r]
                pt.values.append(float(sum(sims)))
                pt.per_sim += sims
                pt.setup = r[0].log.get("setup", {})
                sess.record("sweep", "e2e", "ns_per_day", float(sum(sims)), case=ent["name"], config=pt.label,
                            side="S", repeat=rep, unit="ns/day", better="higher",
                            tags={"stage": pt.stage, "args": " ".join(pt.args), "env": pt.env,
                                  "concurrency": pt.conc, "per_sim": sims, "natoms": ps.natoms})
                if tel:
                    pt.tel.append(tel)
                    for key, unit, better in TELEMETRY_METRICS:
                        if tel.get(key) is not None:
                            sess.record("sweep", "telemetry", key, tel[key], case=ent["name"], config=pt.label,
                                        side="S", repeat=rep, unit=unit, better=better,
                                        tags={"ns_per_day": float(sum(sims)), "stage": pt.stage,
                                              "concurrency": pt.conc})
            log(f"  {pt.label:<44} {pt.median:10.2f} ns/day" + (f"  ({pt.status})" if pt.status != "ok" else ""))
            points.append(pt)
            return pt

        def best(stage_points):
            ok = [p for p in stage_points if p.status == "ok" and p.values]
            return max(ok, key=lambda p: p.median) if ok else None

        min_gain = float(sw.get("min_gain", 0.01))

        def refine(cur, stage_points):
            """Accept a refinement only if it beats the incumbent by min_gain; otherwise keep the
            simpler configuration (differences within noise must not add arguments)."""
            b = best([p for p in stage_points if p is not cur])
            return b if b is not None and b.median > cur.median * (1 + min_gain) else cur

        # 1. default
        default = measure(_Point("default", "default", [], tunepme=True))  # what users get: auto everything
        # 2. offload x threads
        stage2 = []
        for layout in dims.get("cpu_layouts", ["1x20", "1x40", "2x10", "4x5", "5x4", "10x2"]):
            nr, nt = (int(x) for x in layout.split("x"))
            if nr * nt > hw:
                continue
            npme = ["-npme", "0"] if nr > 1 else []
            stage2.append(measure(_Point(f"cpu {layout}", "offload",
                                         ["-ntmpi", str(nr), "-ntomp", str(nt), *npme, *OFFLOAD["cpu"], "-pin", "on"])))
        if have_gpu:
            for mode in dims.get("gpu_offload", ["gpu-nb", "gpu-nb-pme", "gpu-resident", "gpu-resident-bondedcpu"]):
                for nt in dims.get("gpu_threads", [4, 8, 16]):
                    if nt > hw:
                        continue
                    stage2.append(measure(_Point(f"{mode} t{nt}", "offload",
                                                 ["-ntmpi", "1", "-ntomp", str(nt), *OFFLOAD[mode], "-pin", "on"])))
        b2 = best(stage2) or default
        # 3. nstlist (GPU configurations only)
        cur = b2
        if "-nb" in cur.args and cur.args[cur.args.index("-nb") + 1] == "gpu":
            stage3 = [cur]
            for nl in dims.get("nstlist", [50, 80, 150, 200, 300]):
                stage3.append(measure(_Point(f"{cur.label} nstlist{nl}", "nstlist", [*cur.args, "-nstlist", str(nl)],
                                             cur.env)))
            cur = refine(cur, stage3)
            # 4. CUDA graphs (GPU-resident only)
            if "-update" in cur.args and cur.args[cur.args.index("-update") + 1] == "gpu":
                g = measure(_Point(f"{cur.label} +graphs", "graphs", cur.args, {**cur.env, "GMX_CUDA_GRAPH": "1"}))
                cur = refine(cur, [g])
        # 5. PME tuning
        if dims.get("tunepme", True):
            t = measure(_Point(f"{cur.label} +tunepme", "tunepme", cur.args, cur.env, tunepme=True))
            cur = refine(cur, [t])
        best_single = cur
        # 6. throughput: several independent simulations sharing the node
        best_tp = best_single
        if have_gpu and "-nb" in cur.args and cur.args[cur.args.index("-nb") + 1] == "gpu":
            nt = int(cur.args[cur.args.index("-ntomp") + 1]) if "-ntomp" in cur.args else 8
            tp_points = [best_single]
            for conc in dims.get("concurrency", [2, 4]):
                per = max(1, min(nt, hw // conc))
                a = list(cur.args)
                if "-ntomp" in a:
                    a[a.index("-ntomp") + 1] = str(per)
                tp_points.append(measure(_Point(f"{cur.label} x{conc} sims (t{per})", "throughput", a, cur.env,
                                                conc=conc, tunepme=cur.tunepme)))
            best_tp = refine(best_single, tp_points)
        rec = {"name": ent["name"], "system": ent["system"], "natoms": ps.natoms,
               "default": default.as_dict(), "best_single": best_single.as_dict(), "best_throughput": best_tp.as_dict(),
               "gain_single_vs_default": (best_single.median / default.median) if default.median else None,
               "gain_throughput_vs_default": (best_tp.median / default.median) if default.median else None,
               "points": [p.as_dict() for p in points]}
        summary["systems"] = [s for s in summary["systems"] if s["name"] != ent["name"]] + [rec]
        sess.set_summary("sweep", summary)
        log(f"  best single: {best_single.label} {best_single.median:.1f} ns/day "
            f"(x{rec['gain_single_vs_default'] or 0:.2f} vs default); best throughput: {best_tp.label} "
            f"{best_tp.median:.1f} ns/day aggregate")
    return summary


def _launch(build, tpr, rundir, pt: _Point, nsteps, resetstep, timeout, telemetry: bool = False):
    """Run `pt.conc` simulations simultaneously with disjoint thread pinning.

    Returns (results, telemetry metrics). With telemetry, one monitor samples the GPU(s) for the whole
    group; statistics cover the window in which all simulations were inside their timed part, and
    efficiency is computed on the aggregate throughput."""
    from . import telemetry as tm
    args = list(pt.args)
    if not pt.tunepme:
        args += ["-notunepme"]
    args += ["-noconfout"]
    monitor = tm.GpuMonitor().start() if telemetry and tm.available() else None
    try:
        results = _launch_runs(build, tpr, rundir, pt, args, nsteps, resetstep, timeout, monitor)
    finally:
        if monitor is not None:
            monitor.stop()
    if monitor is None or not all(r.ok for r in results):
        return results, {}
    t0 = max(r.marks.get("reset", r.marks.get("start", 0)) for r in results)
    t1 = min(r.marks.get("end", 0) for r in results)
    monitor.write_csv(rundir / "gpu_telemetry.csv" if pt.conc == 1 else rundir.parent / f"{rundir.name}-gpu_telemetry.csv",
                      min(r.marks.get("start", 0) for r in results))
    win = monitor.summary(t0, t1)
    out = {k: win[k] for k in win if isinstance(win[k], (int, float)) and not isinstance(win[k], bool)}
    agg_nsd = sum(r.log.get("ns_per_day") or 0.0 for r in results)
    if win.get("duration_s") and win.get("energy_j") and agg_nsd:
        out["energy_kj_per_ns"] = win["energy_j"] / 1000.0 / (agg_nsd * win["duration_s"] / 86400.0)
    if win.get("power_mean") and agg_nsd:
        out["ns_per_day_per_kw"] = agg_nsd / (win["power_mean"] / 1000.0)
    busy = [r.log["core_time_s"] / r.log["wall_time_s"] for r in results
            if r.log.get("core_time_s") and r.log.get("wall_time_s")]
    if busy:
        out["cpu_cores_busy"] = float(sum(busy))
    return results, out


def _launch_runs(build, tpr, rundir, pt: _Point, args, nsteps, resetstep, timeout, monitor):
    if pt.conc == 1:
        return [mdrun(build, tpr, rundir, args, env=pt.env, nsteps=nsteps, resetstep=resetstep, timeout=timeout,
                      telemetry=monitor)]
    nt = int(args[args.index("-ntomp") + 1]) if "-ntomp" in args else 1
    hw = os.cpu_count() or 1
    stride = 2 if pt.conc * nt * 2 <= hw else 1
    if "-pin" in args:
        i = args.index("-pin")
        del args[i:i + 2]
    results = [None] * pt.conc

    def one(i):
        a = args + ["-pin", "on", "-pinoffset", str(i * nt * stride), "-pinstride", str(stride)]
        results[i] = mdrun(build, tpr, rundir / f"sim{i}", a, env=pt.env, nsteps=nsteps, resetstep=resetstep,
                           timeout=timeout, telemetry=monitor)

    th = [threading.Thread(target=one, args=(i,)) for i in range(pt.conc)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    return results
