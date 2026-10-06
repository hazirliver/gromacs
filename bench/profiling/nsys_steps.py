"""Step-level analysis of an Nsight Systems capture of GPU-resident mdrun.

    bench/profiling/nsys_steps.py PROF.sqlite --first-step N --nstlist 200 --nstcalcenergy 100 \
        [--nstlog 1000] [--out analysis.json] [--natoms 185486]

Works on the sqlite export of a capture made with nsys_capture.sh (capture range = counter reset).
Outputs (JSON + a text report on stdout):
  * kernels: per kernel name: instances/step, total and per-step time, p50/p90/max duration, streams,
    launch configuration, share of all kernel time
  * memops: per copy kind: count/step, bytes/step, time/step, effective GB/s
  * gpu timeline over the whole capture: kernel-busy / copy-only / idle fractions, idle-gap histogram,
    idle time split by what is in flight (copy vs nothing = waiting for the CPU)
  * steps (needs NVTX "Step" ranges, i.e. the GMX_USE_NVTX build): per step wall time on the main
    thread, classified by step type (pair search, energy/virial, coupling, output, plain), with
    p50/p90/max; per type the GPU work launched in the step, copies, idle gaps, sync waits on the CPU
  * cpu: CUDA API time per step on the main thread by call, NVTX sub-range times per step type,
    OS runtime waits and context switches per thread
  * gpu metrics (if sampled): mean of each metric over the capture
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict

import numpy as np


def q(db, sql, *args):
    return db.execute(sql, args).fetchall()


def has_table(db, name):
    return bool(q(db, "SELECT name FROM sqlite_master WHERE type='table' AND name=?", name))


def columns(db, table):
    return [r[1] for r in q(db, f"PRAGMA table_info({table})")]


def short_kernel(name: str) -> str:
    n = name.strip()
    if n.startswith("void "):
        n = n[5:]
    depth = 0
    for i in range(len(n) - 1, -1, -1):
        c = n[i]
        if c == ")":
            depth += 1
        elif c == "(":
            depth -= 1
            if depth == 0:
                n = n[:i]
                break
    n = n.replace("gmx::", "")
    return re.sub(r"\s+", " ", n)[:160]


def union_length(iv):
    """Total length of the union of intervals [(s, e), ...]."""
    if not iv:
        return 0
    iv = sorted(iv)
    tot, cs, ce = 0, iv[0][0], iv[0][1]
    for s, e in iv[1:]:
        if s > ce:
            tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return tot + ce - cs


def merge(iv):
    if not iv:
        return []
    iv = sorted(iv)
    out = [list(iv[0])]
    for s, e in iv[1:]:
        if s > out[-1][1]:
            out.append([s, e])
        else:
            out[-1][1] = max(out[-1][1], e)
    return [tuple(x) for x in out]


def subtract(a, b):
    """a minus b, both merged interval lists."""
    out = []
    j = 0
    for s, e in a:
        cur = s
        while j < len(b) and b[j][1] <= cur:
            j += 1
        k = j
        while k < len(b) and b[k][0] < e:
            if b[k][0] > cur:
                out.append((cur, b[k][0]))
            cur = max(cur, b[k][1])
            k += 1
        if cur < e:
            out.append((cur, e))
    return out


def clip(iv, lo, hi):
    return [(max(s, lo), min(e, hi)) for s, e in iv if e > lo and s < hi]


def pct(v, p):
    return float(np.percentile(v, p)) if len(v) else None


def dist(v):
    v = np.asarray(v, dtype=float)
    if not len(v):
        return {"n": 0}
    return {"n": int(len(v)), "mean": float(v.mean()), "p50": pct(v, 50), "p90": pct(v, 90), "p99": pct(v, 99),
            "max": float(v.max()), "min": float(v.min()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sqlite")
    ap.add_argument("--first-step", type=int, default=None, help="approximate first captured step (= resetstep)")
    ap.add_argument("--nstlist", type=int, default=200)
    ap.add_argument("--nstcalcenergy", type=int, default=100)
    ap.add_argument("--nstcouple", type=int, default=100)
    ap.add_argument("--nstlog", type=int, default=1000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    db = sqlite3.connect(a.sqlite)
    strings = dict(q(db, "SELECT id, value FROM StringIds"))
    res = {"source": a.sqlite}

    # ---------------------------------------------------------------- GPU activities
    kcols = columns(db, "CUPTI_ACTIVITY_KIND_KERNEL")
    extra = [c for c in ("gridX", "gridY", "gridZ", "blockX", "blockY", "blockZ", "registersPerThread",
                         "staticSharedMemory", "dynamicSharedMemory", "graphNodeId") if c in kcols]
    krows = q(db, f"SELECT start, end, streamId, correlationId, demangledName, {', '.join(extra)} "
                  f"FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start")
    kern = [{"s": r[0], "e": r[1], "stream": r[2], "corr": r[3], "name": short_kernel(strings.get(r[4], "?")),
             **dict(zip(extra, r[5:]))} for r in krows]
    mrows = q(db, "SELECT start, end, streamId, correlationId, bytes, copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY "
                  "ORDER BY start") if has_table(db, "CUPTI_ACTIVITY_KIND_MEMCPY") else []
    kinds = {1: "HtoD", 2: "DtoH", 3: "HtoA", 4: "AtoH", 8: "DtoD", 10: "PtoP"}
    if has_table(db, "ENUM_CUDA_MEMCPY_OPER"):
        kinds = {r[0]: r[1] for r in q(db, "SELECT id, name FROM ENUM_CUDA_MEMCPY_OPER")}
    mem = [{"s": r[0], "e": r[1], "stream": r[2], "corr": r[3], "bytes": r[4],
            "kind": re.sub(r"^CUDA_MEMCPY_OPER_", "", str(kinds.get(r[5], r[5])))} for r in mrows]
    srows = q(db, "SELECT start, end, streamId, correlationId, bytes FROM CUPTI_ACTIVITY_KIND_MEMSET") \
        if has_table(db, "CUPTI_ACTIVITY_KIND_MEMSET") else []
    mset = [{"s": r[0], "e": r[1], "stream": r[2], "corr": r[3], "bytes": r[4], "kind": "memset"} for r in srows]
    t0 = min([k["s"] for k in kern] + [m["s"] for m in mem])
    t1 = max([k["e"] for k in kern] + [m["e"] for m in mem])

    # ---------------------------------------------------------------- CPU: runtime API, NVTX
    rt = q(db, "SELECT start, end, globalTid, correlationId, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME ORDER BY start")
    rt = [(s, e, tid, c, re.sub(r"_v\d+$", "", strings.get(n, "?"))) for s, e, tid, c, n in rt]
    corr2api = {c: (s, e, tid, n) for s, e, tid, c, n in rt}
    tid_counts = Counter(r[2] for r in rt)
    main_tid = tid_counts.most_common(1)[0][0] if tid_counts else None
    nvtx = []
    if has_table(db, "NVTX_EVENTS"):
        ncols = columns(db, "NVTX_EVENTS")
        tcol = "COALESCE(text, (SELECT value FROM StringIds WHERE id = textId))" if "textId" in ncols else "text"
        nvtx = q(db, f"SELECT start, end, globalTid, {tcol} FROM NVTX_EVENTS WHERE end IS NOT NULL "
                     f"AND eventType IN (59, 60) ORDER BY start")
    steps_nvtx = [(s, e) for s, e, tid, t in nvtx if t == "Step" and tid == main_tid]
    if not steps_nvtx and nvtx:  # main thread by most Step ranges
        c = Counter(tid for s, e, tid, t in nvtx if t == "Step")
        if c:
            main_tid = c.most_common(1)[0][0]
            steps_nvtx = [(s, e) for s, e, tid, t in nvtx if t == "Step" and tid == main_tid]

    nsteps_cap = None
    if steps_nvtx:
        nsteps_cap = len(steps_nvtx)
    else:  # count update kernels (one leap-frog launch per step)
        nsteps_cap = sum(1 for k in kern if "leapfrog" in k["name"].lower() or "updateKernel" in k["name"])
    res["capture"] = {"t_span_ms": (t1 - t0) / 1e6, "steps": nsteps_cap, "main_tid": main_tid,
                      "ms_per_step_gpu_span": (t1 - t0) / 1e6 / max(1, nsteps_cap),
                      "kernels": len(kern), "memcpys": len(mem), "memsets": len(mset)}
    if steps_nvtx:
        res["capture"]["ms_per_step_nvtx"] = float(np.mean([e - s for s, e in steps_nvtx])) / 1e6

    # ---------------------------------------------------------------- kernel summary
    by = defaultdict(list)
    for k in kern:
        by[k["name"]].append(k)
    ktot = sum(k["e"] - k["s"] for k in kern)
    ksum = []
    for name, ks in by.items():
        d = np.array([k["e"] - k["s"] for k in ks], dtype=float) / 1000.0
        ent = {"name": name, "instances": len(ks), "per_step": len(ks) / nsteps_cap,
               "us_per_step": d.sum() / nsteps_cap, "share_pct": 100 * d.sum() * 1000 / ktot,
               "dur_us": dist(d), "streams": sorted({k["stream"] for k in ks})}
        for c in extra:
            vals = Counter(k.get(c) for k in ks)
            ent[c] = vals.most_common(1)[0][0] if len(vals) == 1 else dict(vals.most_common(3))
        ksum.append(ent)
    ksum.sort(key=lambda x: -x["us_per_step"])
    res["kernels"] = ksum

    # ---------------------------------------------------------------- memops summary
    msum = []
    for kind in sorted({m["kind"] for m in mem + mset}):
        ms = [m for m in mem + mset if m["kind"] == kind]
        d = np.array([m["e"] - m["s"] for m in ms], dtype=float)
        b = np.array([m["bytes"] for m in ms], dtype=float)
        sizes = Counter(m["bytes"] for m in ms)
        msum.append({"kind": kind, "count": len(ms), "per_step": len(ms) / nsteps_cap,
                     "bytes_per_step": b.sum() / nsteps_cap, "us_per_step": d.sum() / 1000 / nsteps_cap,
                     "dur_us": dist(d / 1000), "GBps_effective": float(b.sum() / d.sum()) if d.sum() else None,
                     "top_sizes": sizes.most_common(6), "streams": sorted({m["stream"] for m in ms})})
    res["memops"] = msum

    # ---------------------------------------------------------------- GPU timeline (whole capture)
    kiv = merge([(k["s"], k["e"]) for k in kern])
    civ = merge([(m["s"], m["e"]) for m in mem])
    siv = merge([(m["s"], m["e"]) for m in mset])
    span = t1 - t0
    kb = sum(e - s for s, e in kiv)
    copy_only = subtract(civ, kiv)
    co = sum(e - s for s, e in copy_only)
    busy_any = merge(kiv + civ + siv)
    idle = subtract([(t0, t1)], busy_any)
    gaps = np.array([e - s for s, e in idle], dtype=float) / 1000.0
    copy_overlapped = sum(e - s for s, e in civ) - co
    res["gpu_timeline"] = {
        "span_ms": span / 1e6, "kernel_busy_pct": 100 * kb / span, "copy_only_pct": 100 * co / span,
        "memset_only_pct": 100 * sum(e - s for s, e in subtract(siv, merge(kiv + civ))) / span,
        "idle_pct": 100 * sum(e - s for s, e in idle) / span,
        "copy_total_us_per_step": sum(e - s for s, e in civ) / 1000 / nsteps_cap,
        "copy_overlapped_with_kernels_us_per_step": copy_overlapped / 1000 / nsteps_cap,
        "copy_only_us_per_step": co / 1000 / nsteps_cap,
        "idle_us_per_step": float(gaps.sum()) / nsteps_cap,
        "idle_gaps_us": dist(gaps),
        "idle_gap_hist_us": {f"<{b}": int(((gaps < b) & (gaps >= lo)).sum())
                             for lo, b in ((0, 2), (2, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 1e9))},
    }
    # concurrency: time with >= 2 kernels running (multi-stream overlap)
    ev = sorted([(k["s"], 1) for k in kern] + [(k["e"], -1) for k in kern])
    conc, cur, last, conc_t = 0, 0, ev[0][0], defaultdict(int)
    for t, d in ev:
        conc_t[cur] += t - last
        cur += d
        last = t
    res["gpu_timeline"]["kernel_concurrency_pct"] = {str(k): 100 * v / span for k, v in sorted(conc_t.items())}

    # ---------------------------------------------------------------- steps
    if steps_nvtx:
        st = np.array(steps_nvtx, dtype=np.int64)
        starts = st[:, 0]
        n = len(st)
        # step type from the NVTX sub-ranges on the main thread
        main_ranges = [(s, e, t) for s, e, tid, t in nvtx if tid == main_tid and t != "Step"]
        idx_of = lambda t: int(np.searchsorted(starts, t, side="right") - 1)
        sub = defaultdict(lambda: defaultdict(int))
        names_in_step = defaultdict(set)
        for s, e, t in main_ranges:
            i = idx_of(s)
            if 0 <= i < n and s <= st[i, 1]:
                sub[i][t] += e - s
                names_in_step[i].add(t)
        ns_idx = [i for i in range(n) if "NS" in names_in_step[i] or "Neighbor search" in names_in_step[i]]
        # step numbering: NS steps are multiples of nstlist
        base = None
        if ns_idx and a.first_step is not None:
            i0 = ns_idx[0]
            cand = [b for b in range(a.first_step - a.nstlist, a.first_step + a.nstlist + 1)
                    if (b + i0) % a.nstlist == 0]
            base = min(cand, key=lambda b: abs(b - a.first_step)) if cand else None
        res["steps_meta"] = {"n": n, "ns_step_indices": ns_idx[:20], "base_step": base,
                             "range_names": sorted({t for _, _, t in main_ranges})}

        def stype(i):
            if base is None:
                return "ns" if i in ns_idx else "other"
            sn = base + i
            if sn % a.nstlist == 0:
                return "ns+energy" if sn % a.nstcalcenergy == 0 else "ns"
            if sn % a.nstlog == 0:
                return "energy+output"
            if sn % a.nstcalcenergy == 0:
                return "energy"
            if sn % a.nstcouple == 1:
                return "after-energy"
            if sn % a.nstlist == 1:
                return "after-ns"
            return "plain"

        types = [stype(i) for i in range(n)]
        # GPU activities by launching step (via correlation id -> runtime call on any thread)
        acts = [dict(k, cls="kernel") for k in kern] + [dict(m, cls=m["kind"]) for m in mem + mset]
        per_step = [defaultdict(float) for _ in range(n)]
        act_by_step = defaultdict(list)
        for x in acts:
            api = corr2api.get(x["corr"])
            if api is None:
                continue
            i = idx_of(api[0])
            if 0 <= i < n:
                act_by_step[i].append(x)
        # CPU runtime API calls per step (main thread)
        api_by_step = defaultdict(lambda: defaultdict(float))
        api_cnt = defaultdict(lambda: defaultdict(int))
        for s, e, tid, c, name in rt:
            if tid != main_tid:
                continue
            i = idx_of(s)
            if 0 <= i < n:
                api_by_step[i][name] += e - s
                api_cnt[i][name] += 1
        rows = []
        for i in range(n):
            xs = act_by_step.get(i, [])
            k_iv = merge([(x["s"], x["e"]) for x in xs if x["cls"] == "kernel"])
            c_iv = merge([(x["s"], x["e"]) for x in xs if x["cls"] in ("HtoD", "DtoH", "DtoD")])
            g0 = min((x["s"] for x in xs), default=None)
            nxt = act_by_step.get(i + 1, [])
            g1 = min((x["s"] for x in nxt), default=None)
            row = {"i": i, "type": types[i], "wall_us": (st[i, 1] - st[i, 0]) / 1000,
                   "cpu_period_us": ((starts[i + 1] - starts[i]) / 1000) if i + 1 < n else None,
                   "launches": sum(1 for x in xs if x["cls"] == "kernel"),
                   "copies": sum(1 for x in xs if x["cls"] in ("HtoD", "DtoH", "DtoD")),
                   "copy_bytes": sum(x["bytes"] for x in xs if x["cls"] in ("HtoD", "DtoH", "DtoD")),
                   "kernel_sum_us": sum(x["e"] - x["s"] for x in xs if x["cls"] == "kernel") / 1000,
                   "HtoD_us": sum(x["e"] - x["s"] for x in xs if x["cls"] == "HtoD") / 1000,
                   "DtoH_us": sum(x["e"] - x["s"] for x in xs if x["cls"] == "DtoH") / 1000}
            if g0 is not None and g1 is not None and g1 > g0:
                win = (g0, g1)
                kb_ = union_length(clip(kiv, *win))  # any kernel, also from neighbouring steps
                cb_ = union_length(clip(subtract(civ, kiv), *win))
                row.update({"gpu_period_us": (g1 - g0) / 1000, "gpu_kernel_busy_us": kb_ / 1000,
                            "gpu_copy_only_us": cb_ / 1000, "gpu_idle_us": (g1 - g0 - kb_ - cb_) / 1000})
            for name in ("cudaStreamSynchronize", "cudaEventSynchronize", "cudaLaunchKernel",
                         "cudaMemcpyAsync", "cudaEventRecord", "cudaStreamWaitEvent", "cudaEventQuery",
                         "cudaGraphLaunch"):
                if api_cnt[i].get(name):
                    row[f"api:{name}_us"] = api_by_step[i][name] / 1000
                    row[f"api:{name}_n"] = api_cnt[i][name]
            for t, v in sub[i].items():
                row[f"nvtx:{t}_us"] = v / 1000
            rows.append(row)
        # drop the last step (window to the next step unknown) and summarise per type
        rows_ok = rows[:-1]
        summ = {}
        for t in sorted(set(types)):
            rs = [r for r in rows_ok if r["type"] == t]
            if not rs:
                continue
            keys = sorted({k for r in rs for k in r if k not in ("i", "type")})
            ent = {"count": len(rs), "fraction": len(rs) / len(rows_ok)}
            for k in keys:
                v = [r.get(k, 0.0) or 0.0 for r in rs]
                ent[k] = dist(v) if k in ("wall_us", "cpu_period_us", "gpu_period_us", "gpu_idle_us") else float(np.mean(v))
            summ[t] = ent
        res["step_types"] = summ
        res["steps_all"] = {"cpu_period_us": dist([r["cpu_period_us"] for r in rows_ok if r["cpu_period_us"]]),
                            "gpu_period_us": dist([r["gpu_period_us"] for r in rows_ok if r.get("gpu_period_us")]),
                            "wall_us": dist([r["wall_us"] for r in rows_ok])}
        if base is not None:
            # residue classes mod nstlist: which step positions are expensive
            resid = defaultdict(list)
            for r in rows_ok:
                resid[(base + r["i"]) % a.nstlist].append(r["cpu_period_us"])
            res["residue_mod_nstlist_cpu_period_us"] = {str(k): float(np.mean(v)) for k, v in sorted(resid.items())}
        res["_rows"] = rows

    # ---------------------------------------------------------------- CPU API summary (main thread)
    api_tot = defaultdict(lambda: [0, 0])
    for s, e, tid, c, name in rt:
        if tid == main_tid:
            api_tot[name][0] += e - s
            api_tot[name][1] += 1
    res["api_main_thread"] = sorted([{"name": k, "us_per_step": v[0] / 1000 / nsteps_cap, "calls_per_step": v[1] / nsteps_cap,
                                      "mean_us": v[0] / 1000 / v[1]} for k, v in api_tot.items()],
                                    key=lambda x: -x["us_per_step"])
    # launch latency: kernel start - launch API start, for kernels launched by the main thread
    lat = []
    for k in kern:
        api = corr2api.get(k["corr"])
        if api and api[2] == main_tid and api[3].startswith("cudaLaunchKernel"):
            lat.append((k["s"] - api[0]) / 1000)
    res["launch_to_start_us"] = dist(lat)
    # NVTX ranges summary per thread
    if nvtx:
        nv = defaultdict(lambda: [0, 0])
        for s, e, tid, t in nvtx:
            nv[(tid == main_tid, t)][0] += e - s
            nv[(tid == main_tid, t)][1] += 1
        res["nvtx"] = sorted([{"main_thread": m, "name": t, "us_per_step": v[0] / 1000 / nsteps_cap,
                               "count_per_step": v[1] / nsteps_cap} for (m, t), v in nv.items()],
                             key=lambda x: -x["us_per_step"])[:60]
    # OS runtime and scheduling
    if has_table(db, "OSRT_API"):
        os_ = q(db, "SELECT globalTid, nameId, SUM(end-start), COUNT(*) FROM OSRT_API GROUP BY globalTid, nameId")
        res["osrt"] = sorted([{"tid": tid & 0xFFFFFF, "main": tid == main_tid, "name": strings.get(n, "?"),
                               "us_per_step": t / 1000 / nsteps_cap, "calls_per_step": c / nsteps_cap}
                              for tid, n, t, c in os_], key=lambda x: -x["us_per_step"])[:40]
    if has_table(db, "SCHED_EVENTS"):
        sc = q(db, "SELECT globalTid, COUNT(*), COUNT(DISTINCT cpu) FROM SCHED_EVENTS WHERE isSchedIn = 1 GROUP BY globalTid")
        res["sched"] = sorted([{"tid": tid & 0xFFFFFF, "main": tid == main_tid, "switch_ins_per_step": c / nsteps_cap,
                                "distinct_cpus": ncpu} for tid, c, ncpu in sc], key=lambda x: -x["switch_ins_per_step"])[:40]
    # GPU metrics
    if has_table(db, "GPU_METRICS") and has_table(db, "TARGET_INFO_GPU_METRICS"):
        names = dict(q(db, "SELECT DISTINCT metricId, metricName FROM TARGET_INFO_GPU_METRICS"))
        gm = q(db, "SELECT metricId, AVG(value), COUNT(*) FROM GPU_METRICS GROUP BY metricId")
        res["gpu_metrics_mean"] = {names.get(m, str(m)): {"mean": v, "samples": c} for m, v, c in gm}
    out = a.out or re.sub(r"\.sqlite$", "", a.sqlite) + ".analysis.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=1, default=str)
    report(res)
    print(f"\nwritten: {out}")


def report(res):
    c = res["capture"]
    print(f"capture: {c['steps']} steps, GPU span {c['t_span_ms']:.1f} ms = {c['ms_per_step_gpu_span']*1000:.1f} us/step"
          + (f", NVTX step mean {c['ms_per_step_nvtx']*1000:.1f} us" if 'ms_per_step_nvtx' in c else ""))
    g = res["gpu_timeline"]
    print(f"GPU timeline: kernel busy {g['kernel_busy_pct']:.1f}%  copy-only {g['copy_only_pct']:.1f}%  "
          f"memset-only {g['memset_only_pct']:.2f}%  idle {g['idle_pct']:.1f}%  "
          f"(idle {g['idle_us_per_step']:.1f} us/step; copies {g['copy_total_us_per_step']:.1f} us/step of which "
          f"{g['copy_only_us_per_step']:.1f} not overlapped with kernels)")
    print(f"  kernel concurrency (% of span): {g['kernel_concurrency_pct']}")
    print(f"  idle gaps: {g['idle_gap_hist_us']}")
    print("\nkernels (us/step, share of kernel time, instances/step, median us):")
    for k in res["kernels"][:25]:
        print(f"  {k['us_per_step']:8.1f} {k['share_pct']:5.1f}% {k['per_step']:6.2f}x {k['dur_us']['p50']:8.1f}  "
              f"s{k['streams']} {k['name'][:100]}")
    print("\nmemops:")
    for m in res["memops"]:
        print(f"  {m['kind']:<8} {m['per_step']:6.2f}/step {m['bytes_per_step']/1e6:8.3f} MB/step {m['us_per_step']:8.1f} us/step "
              f"{(m['GBps_effective'] or 0):6.2f} GB/s sizes={m['top_sizes'][:3]} s{m['streams']}")
    if "step_types" in res:
        print(f"\nsteps: {res['steps_meta']}")
        print(f"all: cpu period {res['steps_all']['cpu_period_us']}")
        print(f"     gpu period {res['steps_all']['gpu_period_us']}")
        for t, e in res["step_types"].items():
            cp, gp = e.get("cpu_period_us", {}), e.get("gpu_period_us", {})
            print(f"  {t:<14} n={e['count']:5d} ({100*e['fraction']:5.1f}%) cpu period p50 {cp.get('p50', 0):7.1f} "
                  f"p90 {cp.get('p90', 0):7.1f} max {cp.get('max', 0):8.1f} | gpu period p50 {gp.get('p50', 0):7.1f} "
                  f"idle mean {e.get('gpu_idle_us', {}).get('mean', 0):6.1f} | launches {e.get('launches', 0):5.1f} "
                  f"copies {e.get('copies', 0):4.1f} ({e.get('copy_bytes', 0)/1e6:5.2f} MB) "
                  f"H2D {e.get('HtoD_us', 0):6.1f} D2H {e.get('DtoH_us', 0):6.1f} us")
    print("\nCUDA API on main thread (us/step, calls/step):")
    for x in res["api_main_thread"][:12]:
        print(f"  {x['us_per_step']:8.1f} {x['calls_per_step']:6.2f} {x['name']}")
    print(f"launch->start latency (us): {res['launch_to_start_us']}")
    if "gpu_metrics_mean" in res:
        print("\nGPU metrics (mean over capture):")
        for k, v in res["gpu_metrics_mean"].items():
            print(f"  {v['mean']:10.2f}  {k}")


if __name__ == "__main__":
    main()
