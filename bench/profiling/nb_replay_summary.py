"""Summarise nb_replay.sh output (NB kernel replay; with --file spread_replay.txt --ref prod-order the
PME spread replay): per variant, the median over replay points of the per-point median
kernel time, the ratio to the production kernel measured in the same rounds (with the spread over
replay points), and the force deviation from the production kernel.

    bench/profiling/nb_replay_summary.py OUTDIR [OUTDIR ...] [--json OUT]
"""
import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


def parse(path: Path):
    rows = []
    for line in open(path):
        if not line.startswith("replay"):
            continue
        d = dict(kv.split("=", 1) for kv in line.split()[1:])
        rows.append(d)
    return rows


def summarise(d: Path, fname: str = "replay.txt", ref: str = "prod"):
    rows = parse(d / fname)
    by_call = defaultdict(dict)
    for r in rows:
        by_call[r["call"]][r["variant"]] = r
    ratios = defaultdict(list)
    meds = defaultdict(list)
    frel = defaultdict(list)
    fmax = defaultdict(list)
    for call, vs in by_call.items():
        refv = float(vs[ref]["median_us"])
        for v, r in vs.items():
            ratios[v].append(float(r["median_us"]) / refv)
            meds[v].append(float(r["median_us"]))
            frel[v].append(float(r["f_rel_rms"]))
            fmax[v].append(float(r["f_max_abs"]))
    clk = []
    tel = d / "telemetry.csv"
    if tel.exists():
        for line in open(tel):
            m = re.match(r"[^,]+, (\d+) MHz, ([\d.]+) W", line)
            if m:
                clk.append(int(m.group(1)))
    out = {"dir": str(d), "replay_points": len(by_call), "sm_clock_mhz_median": statistics.median(clk) if clk else None,
           "variants": {}}
    for v in meds:
        out["variants"][v] = {
            "median_us": statistics.median(meds[v]),
            "ratio_to_prod_median": statistics.median(ratios[v]),
            "ratio_min": min(ratios[v]), "ratio_max": max(ratios[v]),
            "f_rel_rms_median": statistics.median(frel[v]), "f_rel_rms_max": max(frel[v]),
            "f_max_abs_max": max(fmax[v])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", type=Path, nargs="+")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--file", default="replay.txt", help="replay.txt (NB) or spread_replay.txt")
    ap.add_argument("--ref", default="prod", help="reference variant (prod / prod-order)")
    a = ap.parse_args()
    res = [summarise(d, a.file, a.ref) for d in a.dirs]
    for r in res:
        print(f"== {r['dir']}  points {r['replay_points']}  SM clock median {r['sm_clock_mhz_median']} MHz")
        print(f"{'variant':<22} {'median us':>10} {'vs prod':>8} {'[min, max]':>18} {'F rel RMS':>10} {'max':>9} {'F max abs':>10}")
        for v, s in sorted(r["variants"].items(), key=lambda kv: kv[1]["ratio_to_prod_median"]):
            print(f"{v:<22} {s['median_us']:10.2f} {s['ratio_to_prod_median']:8.4f} [{s['ratio_min']:.4f}, {s['ratio_max']:.4f}] "
                  f"{s['f_rel_rms_median']:10.2e} {s['f_rel_rms_max']:9.2e} {s['f_max_abs_max']:10.2e}")
    if a.json:
        a.json.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
