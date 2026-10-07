"""Statistics for prun experiments, and the session-wide summary.json / summary.csv.

    bench/profiling/summarize.py exp DIR [--ref VARIANT]        table for one experiment (stdout + DIR/summary.json)
    bench/profiling/summarize.py collect RESULTS_DIR            merge every raw/exp/*/summary.json and
                                                                 raw/**/records.json into summary.json/.csv

Per variant: mean of ms/step with a 95% t confidence interval (on log values, noise is multiplicative),
min/max, CV; ns/day; Matom-steps/s; GPU telemetry over the timed window (power, SM clock, fraction of
samples with sw_power_cap, energy per simulated ns); selected cycle-accounting stages.
With --ref, ratios to the reference variant with a Welch CI (gmxbench.stats.compare).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gmxbench.stats import compare  # noqa: E402

NATOMS = 185486


def ci_log(values):
    v = np.asarray([x for x in values if x is not None and x > 0], dtype=float)
    if len(v) == 0:
        return None
    m = float(np.exp(np.log(v).mean()))
    out = {"n": int(len(v)), "geomean": m, "mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}
    if len(v) > 1:
        s = float(np.log(v).std(ddof=1))
        h = st.t.ppf(0.975, len(v) - 1) * s / math.sqrt(len(v))
        out.update({"ci95": [m * math.exp(-h), m * math.exp(h)], "cv": float(v.std(ddof=1) / v.mean())})
    return out


def ci_lin(values):
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if len(v) == 0:
        return None
    out = {"n": int(len(v)), "mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}
    if len(v) > 1:
        h = st.t.ppf(0.975, len(v) - 1) * float(v.std(ddof=1)) / math.sqrt(len(v))
        out["ci95"] = [out["mean"] - h, out["mean"] + h]
    return out


STAGES = ["Neighbor search", "Launch PP GPU ops.", "Force", "Wait GPU state copy", "Wait GPU NB local",
          "Wait Bonded GPU", "Update", "Constraints", "Comm. energies", "Write traj.", "Rest", "PME GPU mesh",
          "NB X/F buffer ops.", "MD Graph", "Launch PME GPU ops.", "Comp. energies"]


def exp_summary(d: Path, ref: str | None = None) -> dict:
    recs = [json.loads(l) for l in open(d / "runs.jsonl")]
    by = defaultdict(list)
    for r in recs:
        if r.get("status") == "ok" and r.get("ms_per_step"):
            by[r["variant"]].append(r)
    failed = [(r["variant"], r["repeat"], r["status"], r.get("message", "")[:200]) for r in recs if r.get("status") != "ok"]
    out = {"experiment": d.name, "variants": {}, "failed": failed}
    for name, rs in by.items():
        ms = [r["wall_time_s"] * 1000 / (r["nsteps"] - r["resetstep"]) if r.get("wall_time_s") else r["ms_per_step"]
              for r in rs]
        ent = {"n": len(rs), "ms_per_step": ci_log(ms), "ns_per_day": ci_log([r["ns_per_day"] for r in rs]),
               "matom_steps_per_s": ci_log([NATOMS * (r["nsteps"] - r["resetstep"]) / r["wall_time_s"] / 1e6
                                            for r in rs if r.get("wall_time_s")]),
               "cmd": rs[0]["cmd"], "env": rs[0]["env"], "tpr": rs[0]["tpr"], "build": rs[0]["build"],
               "setup": rs[0].get("setup", {}), "rundirs": [r["rundir"] for r in rs]}
        tel = [r.get("telemetry") or {} for r in rs]
        for k in ("power_mean", "power_max", "sm_clock_mean", "sm_clock_p05", "sm_clock_max", "frac_sw_power_cap",
                  "energy_kj_per_ns", "util_gpu_mean", "temp_max", "pcie_tx_mean", "pcie_rx_mean"):
            vals = [t.get(k) for t in tel if t.get(k) is not None]
            if vals:
                ent[k] = ci_lin(vals)
        stg = defaultdict(list)
        for r in rs:
            for k, v in (r.get("stages") or {}).items():
                stg[k].append(v["ms_per_step"])
        ent["stages_ms_per_step"] = {k: float(np.mean(v)) for k, v in stg.items()}
        out["variants"][name] = ent
    if ref and ref in by:
        ra = [r["ns_per_day"] for r in by[ref]]
        for name, rs in by.items():
            c = compare(ra, [r["ns_per_day"] for r in rs], "higher", 0.05, 0.01)
            out["variants"][name]["vs_ref"] = {"ref": ref, "speedup": c.get("speedup"), "ci": c.get("ci"),
                                              "p": c.get("p"), "verdict": c.get("verdict")}
            ea = [(r.get("telemetry") or {}).get("energy_kj_per_ns") for r in by[ref]]
            eb = [(r.get("telemetry") or {}).get("energy_kj_per_ns") for r in rs]
            if all(x is not None for x in ea + eb) and ea and eb:
                ce = compare(ea, eb, "lower", 0.05, 0.01)
                out["variants"][name]["energy_vs_ref"] = {"ratio_lower_better": ce.get("speedup"), "ci": ce.get("ci"),
                                                         "verdict": ce.get("verdict")}
    (d / "summary.json").write_text(json.dumps(out, indent=1))
    return out


def fmt_ci(x, f="{:.3f}"):
    if not x:
        return "-"
    lo, hi = (x.get("ci95") or [x["min"], x["max"]])
    return f"{f.format(x.get('geomean', x.get('mean')))} [{f.format(lo)}, {f.format(hi)}]"


def print_exp(s: dict):
    print(f"experiment {s['experiment']}")
    hdr = f"{'variant':<26} {'n':>2} {'ms/step [95% CI]':>28} {'ns/day':>8} {'min-max ms':>15} {'P W':>6} {'clk':>6} {'cap':>5} {'kJ/ns':>7}  vs ref"
    print(hdr)
    for name, e in sorted(s["variants"].items(), key=lambda kv: kv[1]["ms_per_step"]["geomean"]):
        ms = e["ms_per_step"]
        vr = e.get("vs_ref")
        vs = f"{vr['speedup']:.4f} [{vr['ci'][0]:.4f},{vr['ci'][1]:.4f}] {vr['verdict']}" if vr and vr.get("ci") else ""
        print(f"{name:<26} {e['n']:>2} {fmt_ci(ms, '{:.4f}'):>28} {e['ns_per_day']['geomean']:8.2f} "
              f"{ms['min']:.4f}-{ms['max']:.4f} {e.get('power_mean', {}).get('mean', 0):6.1f} "
              f"{e.get('sm_clock_mean', {}).get('mean', 0):6.0f} {e.get('frac_sw_power_cap', {}).get('mean', 0):5.2f} "
              f"{e.get('energy_kj_per_ns', {}).get('mean', 0):7.2f}  {vs}")
    for f in s["failed"]:
        print(f"FAILED: {f}")


def collect(results: Path):
    """One record per measurement: experiment variants (raw/exp/*/summary.json) and hand-made records
    (raw/records/*.json, a list of record dicts written by the analysis steps)."""
    rows = []
    for sp in sorted((results / "raw" / "exp").glob("*/summary.json")):
        s = json.loads(sp.read_text())
        for name, e in s["variants"].items():
            base = {"source": str(sp.parent.relative_to(results)) + f"/runs/{name}/", "experiment": s["experiment"],
                    "config": name, "cmd": e["cmd"], "env": e["env"], "tpr": e["tpr"], "build": e["build"],
                    "step_type": "all (timed window)", "overhead": "none (unprofiled; telemetry sampler 10 Hz)"}
            for metric, unit, key in (("ms_per_step", "ms/step", "ms_per_step"), ("ns_per_day", "ns/day", "ns_per_day"),
                                      ("matom_steps_per_s", "Matom-steps/s", "matom_steps_per_s"),
                                      ("power_mean", "W", "power_mean"), ("sm_clock_mean", "MHz", "sm_clock_mean"),
                                      ("frac_sw_power_cap", "fraction", "frac_sw_power_cap"),
                                      ("energy_kj_per_ns", "kJ/ns", "energy_kj_per_ns")):
                x = e.get(key)
                if not x:
                    continue
                lo, hi = (x.get("ci95") or [None, None])
                rows.append({**base, "metric": metric, "unit": unit, "value": x.get("geomean", x.get("mean")),
                             "ci95_lo": lo, "ci95_hi": hi, "min": x["min"], "max": x["max"], "n": x["n"]})
            if e.get("vs_ref"):
                v = e["vs_ref"]
                rows.append({**base, "metric": f"speedup_vs_{v['ref']}", "unit": "ratio", "value": v["speedup"],
                             "ci95_lo": (v.get("ci") or [None])[0], "ci95_hi": (v.get("ci") or [None, None])[1],
                             "min": None, "max": None, "n": e["n"]})
    for rp in sorted((results / "raw" / "records").glob("*.json")):
        rows.extend(json.loads(rp.read_text()))
    for i, r in enumerate(rows):
        r.setdefault("id", f"M{i + 1:04d}")
        if str(r.get("experiment", "")).startswith("gpukern-") and not str(r.get("config", "")).startswith("P"):
            gated = r.get("config") == "XO-all+staged-t20"
            r["note"] = ((r.get("note") or "") + " THROWAWAY build outside src/ (worktree gromacs-exp-gpukern); "
                         + ("gmxbench quality --strict passed (raw/gmxbench/quality-XO-final)" if gated
                            else "quality gate run only for the full stacked configuration XO-all+staged-t20")).strip()
            r["overhead"] = r.get("overhead", "") + "; throwaway"
        if str(r.get("experiment", "")).startswith("exp-"):
            gated = r.get("config") in ("ompcuda", "bondedstream", "combo", "combo-t20", "mb16-P", "P", "P-t20")
            r["note"] = ((r.get("note") or "") + " THROWAWAY build outside src/ (worktree gromacs-exp-nbminblocks); "
                         + ("gmxbench quality --strict passed" if gated else "quality gate not run")).strip()
            r["overhead"] = r.get("overhead", "") + "; throwaway"
    (results / "summary.json").write_text(json.dumps(rows, indent=1, default=str))
    keys = ["id", "source", "experiment", "config", "step_type", "metric", "unit", "value", "ci95_lo", "ci95_hi",
            "min", "max", "n", "overhead", "build", "tpr", "env", "cmd", "note"]
    with open(results / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(r[k]) if isinstance(r.get(k), (dict, list)) else r.get(k)) for k in keys})
    print(f"{len(rows)} records -> {results / 'summary.json'}, summary.csv")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("exp")
    p.add_argument("dir", type=Path)
    p.add_argument("--ref")
    p = sub.add_parser("collect")
    p.add_argument("results", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "exp":
        print_exp(exp_summary(a.dir, a.ref))
    else:
        collect(a.results)


if __name__ == "__main__":
    main()
