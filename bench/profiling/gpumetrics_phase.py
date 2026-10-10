#!/usr/bin/env python3
"""Phase-resolved GPU metrics of plain steps: nsys GPU-metrics samples (10 us period) folded onto the step
phase, with t = 0 at the start of the PME spread kernel (first kernel of each step). Prints, per 20 us bin
of the step, the mean of each metric and which kernels are running (median schedule).

    gpumetrics_phase.py PROF.sqlite [--bin 20] [--metrics REGEX]
"""
import argparse
import re
import sqlite3
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--bin", type=float, default=20.0)
    ap.add_argument("--metrics", default=r"SMs Active|SM Issue|Warp Occupancy|Unallocated Warps|DRAM Read|DRAM Write|GPC Clock|PCIe (Read|Write)|Compute Warps in Flight|Tensor")
    a = ap.parse_args()
    db = sqlite3.connect(a.sqlite)
    names = dict(db.execute("SELECT metricId, metricName FROM TARGET_INFO_GPU_METRICS"))
    sel = {mid: n for mid, n in names.items() if re.search(a.metrics, n)}
    sn = dict(db.execute("SELECT id, value FROM StringIds"))
    ks = db.execute("SELECT start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
    spread = [s for s, e, n in ks if "pme_spline_and_spread" in sn.get(n, "")]
    # plain steps: next spread within 1.2 ms
    steps = [(s0, s1) for s0, s1 in zip(spread[:-1], spread[1:]) if s1 - s0 < 1.0e6]
    print(f"{len(steps)} plain steps (spread-to-spread < 1 ms), median period {np.median([b - a for a, b in steps]) / 1e3:.1f} us")
    samples = db.execute(
        f"SELECT timestamp, metricId, value FROM GPU_METRICS WHERE metricId IN ({','.join(map(str, sel))}) ORDER BY timestamp").fetchall()
    ts = np.array([x[0] for x in samples])
    mids = np.array([x[1] for x in samples])
    vals = np.array([x[2] for x in samples], dtype=float)
    starts = np.array([a for a, b in steps])
    ends = np.array([b for a, b in steps])
    idx = np.searchsorted(starts, ts, side="right") - 1
    ok = (idx >= 0) & (ts < ends[np.clip(idx, 0, None)])
    phase = (ts[ok] - starts[idx[ok]]) / 1e3
    mids, vals = mids[ok], vals[ok]
    nb = int(np.ceil(700 / a.bin))
    acc = defaultdict(lambda: [np.zeros(nb), np.zeros(nb)])
    b = np.minimum((phase / a.bin).astype(int), nb - 1)
    for mid in sel:
        m = mids == mid
        s, c = acc[mid]
        np.add.at(s, b[m], vals[m])
        np.add.at(c, b[m], 1)
    # kernel schedule (median start/end relative to step start)
    sched = defaultdict(lambda: [[], []])
    j = 0
    for s0, s1 in steps[: min(len(steps), 3000)]:
        while j < len(ks) and ks[j][0] < s0:
            j += 1
        k = j
        seen = defaultdict(int)
        while k < len(ks) and ks[k][0] < s1:
            n = sn.get(ks[k][2], "?")
            for key in ("nbnxn_kernel_ElecEw_VdwLJFsw_F", "spline_and_spread", "x_to_nbat", "bonded", "fft_r2c", "fft_c2r",
                        "regular_fft", "pme_solve", "pme_gather", "reduceKernel", "leapFrog", "lincs", "settle", "prune"):
                if key in n:
                    seen[key] += 1
                    kk = key if seen[key] == 1 else f"{key}#{seen[key]}"
                    sched[kk][0].append((ks[k][0] - s0) / 1e3)
                    sched[kk][1].append((ks[k][1] - s0) / 1e3)
                    break
            k += 1
    spans = {k: (np.median(v[0]), np.median(v[1])) for k, v in sched.items() if len(v[0]) > 0.4 * min(len(steps), 3000)}
    order = sorted(sel, key=lambda m: names[m])
    short = {m: re.sub(r"\s*\[.*\]", "", names[m]).replace("Throughput", "").strip()[:14] for m in order}
    print("phase_us " + " ".join(f"{short[m]:>14s}" for m in order) + "  running kernels (median schedule)")
    for i in range(nb):
        row = []
        for m in order:
            s, c = acc[m]
            row.append(f"{s[i] / c[i]:14.1f}" if c[i] else " " * 14)
        lo, hi = i * a.bin, (i + 1) * a.bin
        run = [k for k, (x0, x1) in spans.items() if x0 < hi and x1 > lo]
        print(f"{lo:5.0f}-{hi:<4.0f}" + " ".join(row) + "  " + ",".join(run))
    print("\nmeans over plain steps: " + ", ".join(f"{names[m]} {acc[m][0].sum() / max(acc[m][1].sum(), 1):.1f}" for m in order))


if __name__ == "__main__":
    main()
