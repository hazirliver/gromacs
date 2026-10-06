"""End-to-end performance A/B with statistics, per-stage timings and GPU kernel timings.

Protocol per case x configuration
  1. calibration/warm-up: one untimed run per side; the baseline's ms/step sets nsteps so that the
     timed window lasts ~target_seconds (same nsteps for both sides)
  2. `repeats` rounds; in each round A and B run once, in a seeded random order (interleaving
     cancels slow drifts such as thermal throttling or background load)
  3. timing comes from mdrun's own counters after -resetstep (excludes start-up, pair-list and GPU
     warm-up); per-stage times come from the cycle accounting table (with sub-counters)
  4. optionally `nsys_repeats` extra runs per side under Nsight Systems for per-kernel GPU times
  5. comparisons: ns/day (primary), each stage and kernel; FDR control across stages/kernels
"""

from __future__ import annotations

import fnmatch
import math
import random

from .gmx import GromppInput, cached_tpr, merge_mdp, mdrun, nsys_available, nsys_kernel_stats, split_args
from .hostinfo import n_gpus, snapshot
from .stats import apply_fdr, compare
from .systems import SystemUnavailable, prepare
from .util import log

_SKIP_STAGES = ("Total", "Rest")

# GPU telemetry metrics recorded per timed run: (key in telemetry dict, unit, better or None)
TELEMETRY_METRICS = [
    ("energy_kj_per_ns", "kJ/ns", "lower"),
    ("ns_per_day_per_kw", "ns/day/kW", "higher"),
    ("power_mean", "W", "lower"),
    ("power_max", "W", None),
    ("util_gpu_mean", "%", None),
    ("util_mem_mean", "%", None),
    ("vram_proc_max", "MiB", "lower"),
    ("vram_used_max", "MiB", None),
    ("sm_clock_mean", "MHz", None),
    ("temp_max", "°C", None),
    ("pcie_tx_mean", "MB/s", None),
    ("pcie_rx_mean", "MB/s", None),
    ("cpu_cores_busy", "cores", None),
]


def telemetry_values(tel: dict) -> dict:
    """Flatten a run's telemetry (window statistics + derived efficiency) into metric -> value."""
    win = tel.get("window") or {}
    out = {}
    for key, _, _ in TELEMETRY_METRICS:
        v = tel.get(key, win.get(key))
        if v is not None:
            out[key] = float(v)
    return out


def _sel(name, pats):
    return not pats or any(fnmatch.fnmatch(name, p) for p in pats)


def perf_case_inputs(suite, build, case: dict):
    ps = prepare(suite, build, case["system"])
    mdp = merge_mdp(suite.mdp_params(case.get("mdp", ["common", "@system", "perf-out"]), ps.base_mdp),
                    {"nsteps": -1})
    gi = GromppInput(ps.conf, ps.top, mdp, ndx=ps.ndx, ref=ps.conf, maxwarn=int(case.get("maxwarn", 2)))
    return ps, cached_tpr(build, gi, "perf-" + case["name"].replace("/", "_"))


def _round_steps(n: int, mult: int = 100) -> int:
    return max(mult, int(math.ceil(n / mult)) * mult)


def run_args(cfg: dict, tunepme: bool) -> list:
    args = split_args(cfg.get("args", ""))
    if "-pin" not in args:
        args += ["-pin", "on"]
    args += ["-noconfout"]
    if not tunepme:
        args += ["-notunepme"]
    return args


def stage_metrics(parsed: dict, measured_steps: int) -> dict:
    out = {}
    for k, v in parsed.get("stages", {}).items():
        if k in _SKIP_STAGES or measured_steps <= 0:
            continue
        out[k] = (v["wall_s"] * 1000.0 / measured_steps, v["pct"])
    return out


def run_perf(sess, suite, A, B, args) -> dict:
    p = suite.section("perf")
    repeats = args.repeats or int(suite.tv(p.get("repeats", 3)))
    warmup = int(suite.tv(p.get("warmup", 1)))
    target_s = float(suite.tv(p.get("target_seconds", 10)))
    timeout = float(p.get("timeout", 1800))
    tunepme = bool(p.get("tunepme", False))
    alpha = float(p.get("alpha", 0.05))
    min_effect = float(p.get("min_effect", 0.01))
    stage_min_pct = float(p.get("stage_min_pct", 1.0))
    use_nsys = bool(suite.tv(p.get("nsys", True))) and not getattr(args, "no_nsys", False) and nsys_available()
    nsys_reps = int(suite.tv(p.get("nsys_repeats", 2)))
    tel_mode = str(p.get("telemetry", "gpu"))  # gpu: GPU configurations only | all | off
    rng = random.Random(getattr(args, "seed", 12345))
    have_gpu = n_gpus() > 0
    summary = {"settings": {"repeats": repeats, "warmup": warmup, "target_seconds": target_s, "tunepme": tunepme,
                            "alpha": alpha, "min_effect": min_effect, "nsys": use_nsys},
               "cases": []}
    sides = (("A", A), ("B", B))

    for case in suite.cases("perf", args.cases):
        try:
            ps, tpr = perf_case_inputs(suite, A, case)
        except SystemUnavailable as e:
            log(f"perf case {case['name']}: skipped ({e})")
            continue
        except RuntimeError as e:  # a broken case must not abort the rest of the suite
            log(f"perf case {case['name']}: input preparation failed: {e}")
            summary["cases"].append({"case": case["name"], "config": "*", "system": case["system"],
                                     "status": f"input preparation failed: {str(e)[:300]}"})
            sess.record("perf", "e2e", "status", "setup-error", case=case["name"], tags={"message": str(e)[:500]})
            continue
        for cfg_name in case["configs"]:
            if not _sel(cfg_name, args.configs):
                continue
            cfg = suite.configs[cfg_name]
            entry = {"case": case["name"], "config": cfg_name, "system": case["system"], "natoms": ps.natoms}
            if cfg.get("gpu") and not have_gpu:
                entry["status"] = "skipped: no GPU"
                summary["cases"].append(entry)
                continue
            rargs = run_args(cfg, tunepme)
            base = sess.runs / "perf" / case["name"].replace("/", "_") / cfg_name
            log(f"perf {case['name']} [{cfg_name}]")
            # -- calibration / warm-up --------------------------------------------------------
            calib_steps = int(suite.tv(case.get("calib_steps", p.get("calib_steps", 2000))))
            ms_step = None
            status = "ok"
            for w in range(max(1, warmup)):
                for side, bld in sides:
                    r = mdrun(bld, tpr, base / f"warmup{w}-{side}", rargs, env=cfg.get("env"), nsteps=calib_steps,
                              resetstep=calib_steps // 2, timeout=timeout)
                    if not r.ok:
                        status = f"{side} {r.status}: {r.message}"
                        break
                    if side == "A" and r.log.get("ms_per_step"):
                        ms_step = r.log["ms_per_step"]
                if status != "ok":
                    break
            if status != "ok":
                entry["status"] = status
                log(f"  -> {status}")
                sess.record("perf", "e2e", "status", status, case=case["name"], config=cfg_name)
                summary["cases"].append(entry)
                continue
            if "nsteps" in case:
                nsteps = int(suite.tv(case["nsteps"]))
            else:
                nsteps = _round_steps(int(target_s * 1000.0 / max(ms_step or 1.0, 1e-3)))
                nsteps = min(max(nsteps, int(p.get("min_steps", 2000))), int(p.get("max_steps", 2_000_000)))
            resetstep = _round_steps(max(int(nsteps * float(p.get("reset_fraction", 0.25))), 500))
            nsteps += resetstep
            measured = nsteps - resetstep
            entry.update({"nsteps": nsteps, "resetstep": resetstep, "calib_ms_per_step": ms_step})
            # -- timed repeats -----------------------------------------------------------------
            reps_case = int(suite.tv(case["repeats"])) if "repeats" in case and not args.repeats else repeats
            values = {"A": [], "B": []}
            stages = {"A": {}, "B": {}}
            setups = {}
            use_tel = have_gpu and (tel_mode == "all" or (tel_mode == "gpu" and cfg.get("gpu")))
            tels = {"A": [], "B": []}   # (ns/day, telemetry dict) per repeat
            for rep in range(reps_case):
                order = list(sides)
                rng.shuffle(order)
                for side, bld in order:
                    snap = snapshot()
                    r = mdrun(bld, tpr, base / f"rep{rep}-{side}", rargs, env=cfg.get("env"), nsteps=nsteps,
                              resetstep=resetstep, timeout=timeout, telemetry=True if use_tel else None)
                    if not r.ok:
                        log(f"  {side} rep {rep}: {r.status} {r.message}")
                        sess.record("perf", "e2e", "status", r.status, case=case["name"], config=cfg_name,
                                    side=side, repeat=rep, tags={"message": r.message})
                        continue
                    lg = r.log
                    setups[side] = lg.get("setup", {})
                    nsd = lg.get("ns_per_day")
                    values[side].append(nsd)
                    tags = {"order": [s for s, _ in order].index(side), "natoms": ps.natoms, **snap}
                    sess.record("perf", "e2e", "ns_per_day", nsd, case=case["name"], config=cfg_name, side=side,
                                repeat=rep, unit="ns/day", better="higher", tags=tags)
                    if lg.get("ms_per_step") is not None:
                        sess.record("perf", "e2e", "ms_per_step", lg["ms_per_step"], case=case["name"],
                                    config=cfg_name, side=side, repeat=rep, unit="ms/step", better="lower")
                    for k, (v, pct) in stage_metrics(lg, measured).items():
                        stages[side].setdefault(k, []).append((v, pct))
                        sess.record("perf", "stage", f"stage:{k}", v, case=case["name"], config=cfg_name, side=side,
                                    repeat=rep, unit="ms/step", better="lower", tags={"pct": pct})
                    if r.telemetry:
                        tvals = telemetry_values(r.telemetry)
                        tels[side].append((nsd, r.telemetry))
                        for key, unit, better in TELEMETRY_METRICS:
                            if key in tvals:
                                sess.record("perf", "telemetry", key, tvals[key], case=case["name"], config=cfg_name,
                                            side=side, repeat=rep, unit=unit, better=better,
                                            tags={"ns_per_day": nsd, "natoms": ps.natoms,
                                                  "throttle": (r.telemetry.get("window") or {}).get("throttle_reasons")})
                    tmsg = ""
                    if r.telemetry and r.telemetry.get("window"):
                        w = r.telemetry["window"]
                        tmsg = (f"  GPU {w.get('util_gpu_mean', 0):.0f}% {w.get('power_mean', 0):.0f} W "
                                f"{w.get('vram_proc_max', 0):.0f} MiB")
                    log(f"  rep {rep} {side}: {nsd:.2f} ns/day{tmsg}")
            entry["setup"] = setups
            if tels["A"] and tels["B"]:
                tfam = []
                for key, unit, better in TELEMETRY_METRICS:
                    va = [telemetry_values(t).get(key) for _, t in tels["A"]]
                    vb = [telemetry_values(t).get(key) for _, t in tels["B"]]
                    va, vb = [v for v in va if v is not None], [v for v in vb if v is not None]
                    if len(va) < 1 or len(vb) < 1:
                        continue
                    c = compare(va, vb, better or "higher", alpha, min_effect)
                    c.update(metric=key, unit=unit, informational=better is None)
                    tfam.append(c)
                apply_fdr([c for c in tfam if not c["informational"]], alpha)
                entry["telemetry"] = tfam
                # representative run per side: the repeat with the median ns/day
                rep_series = {}
                for side in ("A", "B"):
                    runs = sorted(tels[side], key=lambda x: x[0])
                    nsd_, t_ = runs[len(runs) // 2]
                    rep_series[side] = {"ns_per_day": nsd_, "series": t_.get("series"),
                                        "reset_at_s": t_.get("reset_at_s"),
                                        "throttle": (t_.get("window") or {}).get("throttle_reasons")}
                entry["telemetry_series"] = rep_series
            entry["e2e"] = compare(values["A"], values["B"], "higher", alpha, min_effect, resolution=0.001)
            # -- stage comparisons ---------------------------------------------------------------
            fam = []
            for k in sorted(set(stages["A"]) & set(stages["B"])):
                pcts = [pc for _, pc in stages["A"][k]]
                if not pcts or max(pcts) < stage_min_pct:
                    continue
                # md.log prints stage wall times with 1 ms resolution
                c = compare([v for v, _ in stages["A"][k]], [v for v, _ in stages["B"][k]], "lower", alpha,
                            min_effect, resolution=1.0 / measured)
                c["metric"] = k
                c["pct"] = sum(pcts) / len(pcts)
                fam.append(c)
            apply_fdr(fam, alpha)
            entry["stages"] = fam
            # -- GPU kernels (Nsight Systems) ----------------------------------------------------
            if use_nsys and cfg.get("gpu"):
                ksteps = int(suite.tv(p.get("nsys_steps", 3000)))
                kern = {"A": {}, "B": {}}
                for rep in range(nsys_reps):
                    order = list(sides)
                    rng.shuffle(order)
                    for side, bld in order:
                        rd = base / f"nsys{rep}-{side}"
                        r = mdrun(bld, tpr, rd, rargs, env=cfg.get("env"), nsteps=ksteps, timeout=timeout, nsys=True)
                        if not r.ok:
                            log(f"  nsys {side}: {r.status} {r.message}")
                            continue
                        ks = nsys_kernel_stats(rd)
                        for kind in ("kernels", "memops"):
                            for name, st_ in ks[kind].items():
                                v = st_["total_ns"] / 1000.0 / ksteps  # us per MD step
                                kern[side].setdefault(f"{kind[:-1]}:{name}", []).append(v)
                                sess.record("perf", "gpu-kernel", f"{kind[:-1]}:{name}", v, case=case["name"],
                                            config=cfg_name, side=side, repeat=rep, unit="us/step", better="lower",
                                            tags={"instances_per_step": st_["instances"] / ksteps,
                                                  "med_ns": st_["med_ns"], "full_name": st_["full_name"][:300]})
                kfam = []
                totals = {s: sum(sum(v) / len(v) for v in kern[s].values()) for s in ("A", "B") if kern[s]}
                for k in sorted(set(kern["A"]) & set(kern["B"])):
                    share = (sum(kern["A"][k]) / len(kern["A"][k])) / totals["A"] * 100 if totals.get("A") else 0
                    if share < stage_min_pct:
                        continue
                    c = compare(kern["A"][k], kern["B"][k], "lower", alpha, min_effect)
                    c["metric"] = k
                    c["pct"] = share
                    kfam.append(c)
                apply_fdr(kfam, alpha)
                entry["kernels"] = kfam
            entry["status"] = "ok"
            e = entry["e2e"]
            if "speedup" in e:
                log(f"  => speedup {e['speedup']:.4f} CI {e.get('ci')} p={e.get('p')} [{e['verdict']}]")
            summary["cases"].append(entry)
            sess.set_summary("perf", summary)
    sess.set_summary("perf", summary)
    return summary
