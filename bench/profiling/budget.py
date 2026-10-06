"""Step-time budget (GPU step periods; the GPU is the bottleneck) from an nsys_steps.py analysis: share
of run time per step type, excess over a
plain step, and what the CPU does during pair-search / energy steps (NVTX ranges).

    bench/profiling/budget.py ANALYSIS.json
"""
import json
import sys

import numpy as np

r = json.load(open(sys.argv[1]))
rows = r["_rows"][:-1]
tot = sum(x.get("gpu_period_us", x["cpu_period_us"]) for x in rows)
n = len(rows)
plain = [x.get("gpu_period_us", x["cpu_period_us"]) for x in rows if x["type"] == "plain"]
bmean = float(np.mean(plain))
out = {"steps": n, "mean_us": tot / n, "plain_p50_us": float(np.median(plain)), "plain_mean_us": bmean, "types": {}}
print(f"steps {n}  mean {tot / n:.1f} us  plain median {np.median(plain):.1f} mean {bmean:.1f}")
for t in sorted(set(x["type"] for x in rows)):
    v = [x.get("gpu_period_us", x["cpu_period_us"]) for x in rows if x["type"] == t]
    exc = (sum(v) - len(v) * bmean) / n
    out["types"][t] = {"n": len(v), "mean_us": float(np.mean(v)), "share_pct": 100 * sum(v) / tot,
                       "excess_us_per_step": exc, "excess_pct": 100 * exc * n / tot}
    print(f"{t:14s} n={len(v):5d} mean {np.mean(v):9.1f} share {100 * sum(v) / tot:5.1f}%  "
          f"excess vs plain mean {exc:7.2f} us/step ({100 * exc * n / tot:5.2f}% of run time)")
for t in ("ns+energy", "energy", "plain"):
    ss = [x for x in rows if x["type"] == t]
    if not ss:
        continue
    keys = sorted({k for x in ss for k in x if k.startswith("nvtx:")}, key=lambda k: -np.mean([x.get(k, 0) for x in ss]))
    out["types"][t]["nvtx_us"] = {k[5:]: float(np.mean([x.get(k, 0) for x in ss])) for k in keys}
    out["types"][t]["gpu"] = {k: float(np.mean([x.get(k, 0) for x in ss])) for k in
                              ("gpu_period_us", "gpu_kernel_busy_us", "gpu_copy_only_us", "gpu_idle_us")}
    print(f"{t}: GPU " + " ".join(f"{k}={v:.1f}" for k, v in out["types"][t]["gpu"].items()))
    for k in keys[:12]:
        print(f"   {np.mean([x.get(k, 0) for x in ss]):9.1f} {k}")
json.dump(out, open(sys.argv[1].replace(".json", ".budget.json"), "w"), indent=1)
