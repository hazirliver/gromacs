#!/usr/bin/env python3
"""Is the GPU waiting for the CPU to launch work in plain steps?

For every kernel of a plain step (steps anchored at the PME spread kernel, which is the first kernel of
each step), compare the CPU time of its launch API call with the GPU start of the kernel. A kernel whose
GPU start is within a few us of the end of its launch call was waiting for the CPU (launch-bound); one
that starts long after its launch was waiting for the GPU (dependencies or SM slots).

    launch_gaps.py PROF.sqlite [--skip-first 1000]
"""
import argparse
import sqlite3
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    a = ap.parse_args()
    db = sqlite3.connect(a.sqlite)
    names = dict(db.execute("SELECT id, value FROM StringIds"))
    k = db.execute("SELECT start, end, correlationId, shortName, streamId FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
    rt = {c: (s, e, names.get(n, "?")) for s, e, c, n in
          db.execute("SELECT start, end, correlationId, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME")}
    # copies too (they are on the critical path for CMAP)
    m = db.execute("SELECT start, end, correlationId, copyKind, bytes, streamId FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start").fetchall()

    def short(n):
        n = names.get(n, "?")
        for key in ("nbnxn_kernel_ElecEw_VdwLJFsw_F_cuda", "nbnxn_kernel_ElecEw_VdwLJFsw_VF_cuda",
                    "nbnxn_kernel_prune_cuda", "pme_spline_and_spread", "x_to_nbat", "bonded_kernel_gpu",
                    "regular_fft_r2c", "regular_fft_c2r", "regular_fft", "pme_solve", "pme_gather", "reduceKernel",
                    "leapFrog", "lincsKernel", "settleKernel"):
            if key in n:
                return key
        return n[:40]

    ev = [(s, e, c, short(n), st, "K") for s, e, c, n, st in k]
    ev += [(s, e, c, f"memcpy kind{ck} {b}B", st, "M") for s, e, c, ck, b, st in m]
    ev.sort()
    # split into steps at each spread kernel
    starts = [i for i, x in enumerate(ev) if x[3] == "pme_spline_and_spread"]
    per = defaultdict(lambda: defaultdict(list))
    nsteps = 0
    for a_i, b_i in zip(starts[:-1], starts[1:]):
        step = ev[a_i:b_i]
        names_in = [x[3] for x in step]
        if "nbnxn_kernel_ElecEw_VdwLJFsw_F_cuda" not in names_in or "nbnxn_kernel_prune_cuda" in names_in and False:
            continue
        if any("VF" in n or "kind1 5" in n for n in names_in):
            continue
        # plain step: duration below 1 ms
        if step[-1][1] - step[0][0] > 1.2e6:
            continue
        t0 = step[0][0]
        api0 = rt[step[0][2]][0] if step[0][2] in rt else None
        nsteps += 1
        seen = Counter()
        for s, e, c, n, st, kind in step:
            seen[n] += 1
            key = n if seen[n] == 1 else f"{n} #{seen[n]}"
            if c not in rt:
                continue
            a_s, a_e, an = rt[c]
            per[key]["gpu_start"].append((s - t0) / 1e3)
            per[key]["gpu_end"].append((e - t0) / 1e3)
            per[key]["api_start"].append((a_s - t0) / 1e3)
            per[key]["api_end"].append((a_e - t0) / 1e3)
            per[key]["wait"].append((s - a_e) / 1e3)
            per[key]["stream"].append(st)
    print(f"plain steps analysed: {nsteps}; times in us relative to the GPU start of the spread kernel")
    print(f"{'activity':40s} {'stream':>6s} {'API start':>9s} {'API end':>8s} {'GPU start':>9s} {'GPU end':>8s} "
          f"{'start-API end':>13s}  (medians; p10/p90 of start-API end)")
    rows = sorted(per.items(), key=lambda kv: np.median(kv[1]["gpu_start"]))
    for key, d in rows:
        if len(d["gpu_start"]) < nsteps * 0.5:
            continue
        w = np.array(d["wait"])
        print(f"{key[:40]:40s} {int(np.median(d['stream'])):6d} {np.median(d['api_start']):9.1f} {np.median(d['api_end']):8.1f} "
              f"{np.median(d['gpu_start']):9.1f} {np.median(d['gpu_end']):8.1f} {np.median(w):13.1f}  "
              f"[{np.percentile(w, 10):.1f}, {np.percentile(w, 90):.1f}]")


from collections import Counter  # noqa: E402

if __name__ == "__main__":
    main()
