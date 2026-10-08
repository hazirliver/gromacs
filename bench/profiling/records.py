"""Turn analysis artifacts into summary records (raw/records/*.json), merged by `summarize.py collect`.

    bench/profiling/records.py RESULTS_DIR

Sources:
  raw/nsys/*/prof.analysis.json   (nsys_steps.py)  -> per-capture step-type periods (p50/p90/max),
                                                      GPU busy/idle/copy shares, per-kernel us/step,
                                                      memops; overhead flag = the tracing mode
  raw/nsys/*/prof.analysis.budget.json (budget.py) -> run-time share and excess per step type
  raw/gmxbench/*/session.json                       -> A/B speedups with CIs, kernel tables
  raw/pcie/probe.txt                                -> copy latency/bandwidth per vCPU
  raw/records/manual.json (hand-written, optional)  -> numbers read from md.log/perf reports
"""
import json
import re
import sys
from pathlib import Path


def overhead_flag(capdir: Path) -> str:
    cmd = (capdir / "command.txt").read_text() if (capdir / "command.txt").exists() else ""
    if "--sample=process-tree" in cmd:
        return "nsys full (cuda,nvtx,osrt,4 kHz CPU sampling,ctxsw): ~+1.9% step time"
    return "nsys light (cuda,nvtx): ~+0.6% step time"


def nsys_records(res: Path):
    out = []
    for a in sorted((res / "raw" / "nsys").glob("*/prof.analysis.json")):
        cap = a.parent
        r = json.loads(a.read_text())
        src = str(cap.relative_to(res))
        base = {"source": src, "experiment": "nsys", "config": cap.name, "overhead": overhead_flag(cap),
                "build": "S" if "-S-" in cap.name else ("P" if "-P-" in cap.name else "?")}
        c = r["capture"]
        out.append({**base, "step_type": "all", "metric": "gpu_span_us_per_step", "unit": "us",
                    "value": c["ms_per_step_gpu_span"] * 1000, "n": c["steps"]})
        g = r["gpu_timeline"]
        for k in ("kernel_busy_pct", "copy_only_pct", "idle_pct"):
            out.append({**base, "step_type": "all", "metric": f"gpu_{k}", "unit": "%", "value": g[k], "n": c["steps"]})
        for k in ("idle_us_per_step", "copy_total_us_per_step", "copy_only_us_per_step"):
            out.append({**base, "step_type": "all", "metric": f"gpu_{k}", "unit": "us/step", "value": g[k], "n": c["steps"]})
        for t, e in (r.get("step_types") or {}).items():
            for per in ("cpu_period_us", "gpu_period_us", "gpu_idle_us"):
                d = e.get(per)
                if isinstance(d, dict) and d.get("n"):
                    out.append({**base, "step_type": t, "metric": per, "unit": "us", "value": d["p50"],
                                "min": d["min"], "max": d["max"], "n": d["n"],
                                "note": f"p50; p90={d['p90']:.1f}; mean={d['mean']:.1f}"})
        for k in r["kernels"][:25]:
            out.append({**base, "step_type": "all", "metric": f"kernel_us_per_step:{k['name'][:90]}", "unit": "us/step",
                        "value": k["us_per_step"], "min": k["dur_us"]["min"], "max": k["dur_us"]["max"],
                        "n": k["instances"], "note": f"instances/step={k['per_step']:.2f}; p50 duration={k['dur_us']['p50']:.1f} us"})
        for m in r["memops"]:
            out.append({**base, "step_type": "all", "metric": f"memop_us_per_step:{m['kind']}", "unit": "us/step",
                        "value": m["us_per_step"], "n": m["count"],
                        "note": f"{m['per_step']:.2f}/step, {m['bytes_per_step'] / 1e6:.3f} MB/step, {m['GBps_effective'] or 0:.1f} GB/s"})
        b = a.with_name("prof.analysis.budget.json")
        if b.exists():
            bb = json.loads(b.read_text())
            for t, e in bb["types"].items():
                out.append({**base, "step_type": t, "metric": "run_time_share_pct", "unit": "%", "value": e["share_pct"],
                            "n": e["n"]})
                out.append({**base, "step_type": t, "metric": "excess_over_plain_us_per_step", "unit": "us/step",
                            "value": e["excess_us_per_step"], "n": e["n"], "note": f"{e['excess_pct']:.2f}% of run time"})
    return out


def gmxbench_records(res: Path):
    out = []
    for s in sorted((res / "raw" / "gmxbench").glob("*/session.json")):
        j = json.loads(s.read_text())
        for c in (j.get("summary", {}).get("perf", {}) or {}).get("cases", []):
            e = c.get("e2e") or {}
            if "speedup" not in e:
                continue
            base = {"source": str(s.parent.relative_to(res)), "experiment": f"gmxbench:{s.parent.name}",
                    "config": c["config"], "step_type": "all (timed window)", "overhead": "none (unprofiled)",
                    "build": "A/B: " + " vs ".join(Path(b.get('build_dir', '?')).name for b in j.get("builds", {}).values())
                    if isinstance(j.get("builds"), dict) else ""}
            out.append({**base, "metric": "speedup_B_over_A (ns/day)", "unit": "ratio", "value": e["speedup"],
                        "ci95_lo": e["ci"][0], "ci95_hi": e["ci"][1], "n": e["n_a"], "note": f"p={e['p']:.3f}; MDE={e.get('mde', 0):.4f}; {e['verdict']}"})
            for side in ("a", "b"):
                x = e[side]
                out.append({**base, "metric": f"ns_per_day_{side.upper()}", "unit": "ns/day", "value": x["mean"],
                            "ci95_lo": x["ci95"][0], "ci95_hi": x["ci95"][1], "min": x["min"], "max": x["max"], "n": x["n"]})
    return out


def pcie_records(res: Path):
    p = res / "raw" / "pcie" / "probe.txt"
    out = []
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        m = re.match(r"cpu (\d+) (.*)", line)
        if not m:
            continue
        kv = dict(zip(m.group(2).split()[0::2], m.group(2).split()[1::2]))
        for k, unit in (("small_h2d_rt_us", "us"), ("small_d2h_rt_us", "us"), ("empty_kernel_rt_us", "us"),
                        ("h2d_dev_us", "us"), ("d2h_dev_us", "us"), ("h2d_GBps", "GB/s"), ("d2h_GBps", "GB/s")):
            out.append({"source": "raw/pcie/probe.txt", "experiment": "pcie-probe", "config": f"vCPU {m.group(1)}",
                        "step_type": "-", "metric": k, "unit": unit, "value": float(kv[k]), "n": 2000 if "small" in k or "empty" in k else 300,
                        "overhead": "none", "note": f"median; {kv['bytes']} bytes for the large copies"})
    return out


def main():
    res = Path(sys.argv[1])
    d = res / "raw" / "records"
    d.mkdir(parents=True, exist_ok=True)
    for name, fn in (("nsys", nsys_records), ("gmxbench", gmxbench_records), ("pcie", pcie_records)):
        recs = fn(res)
        (d / f"{name}.json").write_text(json.dumps(recs, indent=1))
        print(f"{name}: {len(recs)} records")


if __name__ == "__main__":
    main()
