#!/usr/bin/env python3
"""Hot-loop listing of one kernel from `ncu --page source --print-source sass --csv`: every SASS
instruction executed at least --min times, in address order, with executed warp instructions, average
active threads, warp-stall samples and the dominant stall reasons; plus a summary by execution count
(= basic-block frequency) and by opcode inside the pair-interaction body (the block of MUFU.RSQ).

    sass_hot.py SOURCE.csv [--min 100000] [--body-only]
"""
import argparse
import csv
import re
from collections import Counter, defaultdict

STALLS = ["stall_barrier", "stall_branch_resolving", "stall_dispatch", "stall_drain", "stall_imc", "stall_lg",
          "stall_long_sb", "stall_math", "stall_membar", "stall_mio", "stall_misc", "stall_no_inst",
          "stall_not_selected", "stall_selected", "stall_short_sb", "stall_sleep", "stall_tex", "stall_wait"]


def num(s):
    try:
        return float(s)
    except ValueError:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--min", type=float, default=1e5)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    rows = list(csv.reader(open(a.csv)))
    h = rows[1]
    ix = {k: i for i, k in enumerate(h)}
    ins = []
    for r in rows[2:]:
        if len(r) < len(h):
            continue
        src = r[ix["Source"]].strip()
        op = re.sub(r"^@!?U?P\w+\s+", "", src).split(" ")[0]
        ins.append(dict(addr=r[ix["Address"]], src=src, op=op, n=num(r[ix["Instructions Executed"]]),
                        thr=num(r[ix["Avg. Threads Executed"]]), pthr=num(r[ix["Avg. Predicated-On Threads Executed"]]),
                        samp=num(r[ix["Warp Stall Sampling (All Samples)"]]),
                        st={s: num(r[ix[s]]) for s in STALLS if s in ix}))
    tot_n = sum(x["n"] for x in ins)
    tot_s = sum(x["samp"] for x in ins)
    print(f"{len(ins)} SASS instructions, {tot_n / 1e6:.1f} M warp instructions executed, {tot_s:.0f} stall samples")
    # stall totals
    st = Counter()
    for x in ins:
        st.update(x["st"])
    print("stall samples by reason: " + ", ".join(f"{k[6:]} {100 * v / tot_s:.1f}%" for k, v in st.most_common(9)))
    # by execution count (basic-block frequency)
    groups = defaultdict(lambda: [0, 0.0, 0.0, Counter()])
    for x in ins:
        if x["n"] <= 0:
            continue
        g = groups[round(x["n"], -3)]
        g[0] += 1
        g[1] += x["n"]
        g[2] += x["samp"]
        g[3][x["op"]] += 1
    print("\nbasic blocks by execution count (static instr, share of executed warp instr, share of stall samples):")
    for k, (cnt, n, s, ops) in sorted(groups.items(), key=lambda kv: -kv[1][1])[:14]:
        print(f"  exec {k / 1e6:7.3f} M x {cnt:3d} instr = {100 * n / tot_n:5.1f}% of instr, {100 * s / tot_s:5.1f}% of samples; "
              + " ".join(f"{o}:{c}" for o, c in ops.most_common(8)))
    rsq = [x for x in ins if x["op"].startswith("MUFU.RSQ")]
    if rsq:
        body_n = max(x["n"] for x in rsq)
        body = [x for x in ins if abs(x["n"] - body_n) < 0.5]
        print(f"\npair-interaction body (exec count {body_n / 1e6:.3f} M, the MUFU.RSQ block): {len(body)} instructions, "
              f"{100 * sum(x['n'] for x in body) / tot_n:.1f}% of instr, {100 * sum(x['samp'] for x in body) / tot_s:.1f}% of samples, "
              f"avg active threads {body[0]['thr']:.1f}")
        print("  opcodes: " + " ".join(f"{o}:{c}" for o, c in Counter(x["op"] for x in body).most_common()))
    if a.list:
        print("\naddress  exec(M)  thr  samples  top stalls | SASS")
        for x in ins:
            if x["n"] < a.min:
                continue
            top = ", ".join(f"{k[6:]}:{int(v)}" for k, v in sorted(x["st"].items(), key=lambda kv: -kv[1])[:3] if v > 0)
            print(f"{x['addr'][-5:]} {x['n'] / 1e6:7.3f} {x['pthr']:4.1f} {x['samp']:7.0f}  {top:40s} | {x['src'][:90]}")


if __name__ == "__main__":
    main()
