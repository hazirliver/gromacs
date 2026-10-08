"""Derived measurements (clock scaling, occupancy, critical-path segments, power model, quality gates)
as summary records: writes RESULTS/raw/records/derived.json.

    bench/profiling/derived_records.py RESULTS_DIR
"""
import json
import sys
from pathlib import Path

res = Path(sys.argv[1])
raw = res / "raw"
out = []


def rec(**kw):
    kw.setdefault("overhead", "none")
    out.append(kw)


cs = raw / "nsys" / "clockscale.json"
if cs.exists():
    for x in json.loads(cs.read_text()):
        rec(source="raw/nsys/clk*-S-light, raw/nsys/clockscale.json", experiment="clock-scaling", config="production, SM clock locked",
            step_type="all", metric=f"clock_exponent_alpha:{x['name'][:90]}", unit="-", value=x["alpha"], n=len(x["clocks_mhz"]),
            note=f"d ~ f^-alpha over {x['clocks_mhz']} MHz; p50 us {[round(v, 1) for v in x['p50_us']]}",
            overhead="nsys light (~+0.6%)")
oc = raw / "nsys" / "prod-S-light" / "prof.analysis.occupancy.json"
if oc.exists():
    for x in json.loads(oc.read_text()):
        rec(source="raw/nsys/prod-S-light (launch records)", experiment="occupancy", config="production",
            step_type="all", metric=f"theoretical_occupancy:{x['kernel'][:90]}", unit="fraction", value=x["theoretical_occupancy"],
            note=f"block {x['block']}, regs {x['regs']}, smem {x['smem']} B, blocks/SM {x['blocks_per_sm']}, limiter {x['limiter']}, waves {x['waves']}")
a = json.loads((raw / "nsys" / "prod-S-light" / "prof.analysis.json").read_text())
an = {g["what"][:30]: g for g in a["anatomy"]["plain"]["gpu"]}
nb = next(v for k, v in an.items() if k.startswith("nbnxn_kernel_ElecEw"))
st = next(v for k, v in an.items() if k.startswith("settleKernel"))
per = a["step_types"]["plain"]["gpu_period_us"]["p50"]
for name, val in (("plain_step_pre_NB_segment_us", nb["start_us"] - (st["end_us"] - per)),
                  ("plain_step_NB_kernel_segment_us", nb["end_us"] - nb["start_us"]),
                  ("plain_step_post_NB_to_SETTLE_end_us", st["end_us"] - nb["end_us"])):
    rec(source="raw/nsys/prod-S-light (anatomy, medians)", experiment="nsys", config="prod-S-light", step_type="plain",
        metric=name, unit="us", value=val, n=a["anatomy"]["plain"]["steps"], overhead="nsys light (~+0.6%)")
# power model from the clock-locked runs
pw = json.loads((raw / "exp" / "power" / "summary.json").read_text())["variants"]
t1, t2 = pw["lock2250-pl325"]["ms_per_step"]["geomean"], pw["lock1980-pl325"]["ms_per_step"]["geomean"]
b = (t2 - t1) / (1 / 1980 - 1 / 2250)
a0 = t1 - b / 2250
rec(source="raw/exp/power (lock2250, lock1980)", experiment="power", config="T(f)=a+b/f fit", step_type="all (timed window)",
    metric="T_at_2520MHz_extrapolated", unit="ms/step", value=a0 + b / 2520, note=f"a={a0:.4f} ms, b={b:.1f} ms*MHz; 2 points, no CI")
rec(source="raw/exp/power", experiment="power", config="T(f)=a+b/f fit", step_type="all (timed window)",
    metric="clock_scaled_fraction_at_2250MHz", unit="fraction", value=(b / 2250) / t1)
rec(source="nvidia-smi --query-gpu=power.limit,enforced.power.limit (20:10:37)", experiment="power", config="-pl 350 requested",
    step_type="-", metric="enforced_power_limit", unit="W", value=325.0, note="power.limit 350 W, enforced.power.limit 325 W")
for q in ("quality-ompcuda", "quality-bondedstream", "quality-combo"):
    p = raw / "gmxbench" / q / "session.json"
    if not p.exists():
        continue
    s = json.loads(p.read_text())
    for r in s["summary"]["quality"]["results"]:
        rec(source=f"raw/gmxbench/{q}", experiment=f"gmxbench quality --strict ({q})", config=r["config"], step_type="steps 0-100",
            metric="quality_status", unit="-", value=r["status"],
            note=(f"f_rel_rms={r.get('f_rel_rms')}, noise={r.get('noise_f_rel_rms')}" if r["status"] == "EQUIVALENT" else ""))
rec(source="raw/perf/prod-P-fp, build.make:9979, nm/objdump of listed_forces_gpu_impl_gpu.cpp.o", experiment="perf",
    config="prod-P", step_type="ns+energy", metric="gpu_bonded_list_update_cpu_ms_per_search", unit="ms", value=1.9,
    note="single-threaded: nvcc-compiled host code without -fopenmp; 515 samples at 4999 Hz over 54 searches")
for k, v, src in (("prod_search_ms_per_search_t16", 10.07, "raw/exp/threads (md.log Neighbor search)"),):
    rec(source=src, experiment="threads", config="t16", step_type="ns", metric=k, unit="ms", value=v)
(raw / "records").mkdir(exist_ok=True)
(raw / "records" / "derived.json").write_text(json.dumps(out, indent=1, default=str))
print(len(out), "derived records")
