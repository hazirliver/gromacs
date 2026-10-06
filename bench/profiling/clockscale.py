"""Counter-free limiter classification: how each kernel's duration scales with the locked SM clock.

    bench/profiling/clockscale.py DIR_f1 DIR_f2 [DIR_f3 ...]

Each DIR is an nsys capture (nsys_capture.sh) made with `nvidia-smi -lgc f,f` (memory clock unchanged)
containing prof.analysis.json (nsys_steps.py) and smi.csv (sampled clocks). For every kernel, the
median duration d(f) is fitted as d ~ f^-alpha over the clocks actually measured:
  alpha ~ 1   duration follows the SM clock: SM-bound (issue, latency, L1/shared, or L2 - L2 runs on
              the SM/graphics clock domain on NVIDIA GPUs)
  alpha ~ 0   independent of the SM clock: DRAM- or PCIe-bound (memory clock fixed) or launch-bound
Copies are reported the same way (expected alpha ~ 0).
"""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


def measured_clock(d: Path):
    p = d / "smi.csv"
    if not p.exists():
        return None
    vals = []
    with open(p) as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            try:
                sm = float(row[1].strip().split()[0])
                pw = float(row[3].strip().split()[0])
            except (ValueError, IndexError):
                continue
            if pw > 150:          # samples while mdrun is running
                vals.append(sm)
    return float(np.median(vals)) if vals else None


def main():
    dirs = [Path(x) for x in sys.argv[1:]]
    data = []
    for d in dirs:
        a = json.loads((d / "prof.analysis.json").read_text())
        f = measured_clock(d)
        k = {x["name"]: x["dur_us"]["p50"] for x in a["kernels"] if x["per_step"] >= 0.4}
        m = {f"[{x['kind']}]": x["dur_us"]["p50"] for x in a["memops"]}
        step = a.get("steps_all", {}).get("gpu_period_us", {}).get("p50")
        plain = a.get("step_types", {}).get("plain", {}).get("gpu_period_us", {}).get("p50")
        data.append({"dir": str(d), "clock": f, "kern": {**k, **m}, "mean_step": a["capture"]["ms_per_step_gpu_span"] * 1000,
                     "plain_p50": plain})
    data = [x for x in data if x["clock"]]
    data.sort(key=lambda x: x["clock"])
    clocks = np.array([x["clock"] for x in data])
    print("measured SM clocks (median while running):", clocks)
    out = []
    names = sorted(set.intersection(*[set(x["kern"]) for x in data]),
                   key=lambda n: -data[-1]["kern"][n])
    rows = [("mean step (GPU span/step)", [x["mean_step"] for x in data]),
            ("plain step GPU period p50", [x["plain_p50"] for x in data])] + \
           [(n, [x["kern"][n] for x in data]) for n in names]
    print(f"{'alpha':>6}  " + "  ".join(f"{c:7.0f}MHz" for c in clocks) + "  name")
    for n, v in rows:
        v = np.array(v, dtype=float)
        if np.any(v <= 0):
            continue
        alpha = -np.polyfit(np.log(clocks), np.log(v), 1)[0]
        out.append({"name": n, "alpha": float(alpha), "clocks_mhz": clocks.tolist(), "p50_us": v.tolist()})
        print(f"{alpha:6.2f}  " + "  ".join(f"{x:10.1f}" for x in v) + f"  {n[:80]}")
    Path(dirs[0]).parent.joinpath("clockscale.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
