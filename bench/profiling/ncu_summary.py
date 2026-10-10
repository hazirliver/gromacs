#!/usr/bin/env python3
"""Per-kernel summary of an Nsight Compute report (--set full): throughput, limiter, occupancy, cache and
warp-stall breakdown. Reads `ncu --import REP --page raw --csv` (all metrics).

    ncu_summary.py REPORT.ncu-rep [--json OUT] [--stalls N]
"""
import argparse
import csv
import io
import json
import re
import subprocess
from collections import OrderedDict, defaultdict

KEYS = OrderedDict([
    ("dur_us", "gpu__time_duration.sum"),
    ("sm_mhz", "sm__cycles_elapsed.avg.per_second"),
    ("ipc_active", "sm__inst_executed.avg.per_cycle_active"),
    ("issue_active_pct", "sm__inst_issued.avg.pct_of_peak_sustained_active"),
    ("sm_pct", "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("mem_pct", "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"),
    ("l1_pct", "l1tex__throughput.avg.pct_of_peak_sustained_active"),
    ("l2_pct", "lts__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("dram_pct", "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"),
    ("l2_hit_pct", "lts__t_sector_hit_rate.pct"),
    ("l1_hit_pct", "l1tex__t_sector_hit_rate.pct"),
    ("occ_achieved_pct", "sm__warps_active.avg.pct_of_peak_sustained_active"),
    ("occ_theor_pct", "sm__maximum_warps_per_active_cycle_pct"),
    ("regs", "launch__registers_per_thread"),
    ("grid", "launch__grid_size"),
    ("block", "launch__block_size"),
    ("waves", "launch__waves_per_multiprocessor"),
    ("smem_block", "launch__shared_mem_per_block_static"),
    ("eligible_warps", "smsp__warps_eligible.avg.per_cycle_active"),
    ("active_warps_sched", "smsp__warps_active.avg.per_cycle_active"),
    ("no_eligible_pct", "smsp__issue_inst0.avg.pct_of_peak_sustained_active"),
    ("fp32_pipe_pct", "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active"),
    ("alu_pipe_pct", "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active"),
    ("fmaheavy_pct", "sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_active"),
    ("xu_pipe_pct", "sm__pipe_xu_cycles_active.avg.pct_of_peak_sustained_active"),
    ("lsu_pipe_pct", "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active"),
    ("dram_bytes", "dram__bytes.sum"),
    ("l2_red_sectors", "lts__t_sectors_op_red.sum"),
    ("l2_atom_sectors", "lts__t_sectors_op_atom.sum"),
    ("l2_sectors", "lts__t_sectors.sum"),
    ("warp_cycles_per_inst", "smsp__average_warp_latency_per_inst_issued.ratio"),
])
STALL_RE = re.compile(r"^smsp__average_warps_issue_stalled_(\w+)_per_issue_active\.ratio$")


def load(rep):
    txt = subprocess.run(["ncu", "--import", rep, "--page", "raw", "--csv"], capture_output=True, text=True,
                         check=True).stdout
    rows = list(csv.reader(io.StringIO(txt)))
    hdr, units, data = rows[0], rows[1], rows[2:]
    return hdr, units, data


def num(s):
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def short(name):
    for k in ("nbnxn_kernel_ElecEw_VdwLJFsw_F_cuda", "nbnxn_kernel_ElecEw_VdwLJFsw_VF_cuda", "pme_spline_and_spread",
              "pme_gather", "pme_solve", "regular_fft_r2c", "regular_fft_c2r", "regular_fft", "bonded_kernel_gpu",
              "lincsKernel", "settleKernel", "leapFrogKernel", "x_to_nbat", "reduceKernel"):
        if k in name:
            return k
    if "nbnxn_kernel_prune_cuda" in name:
        return "prune_fresh" if "ILb1E" in name or "<(bool)1>" in name or "<true>" in name else "prune_rolling"
    return name[:40]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rep")
    ap.add_argument("--json")
    ap.add_argument("--stalls", type=int, default=6)
    a = ap.parse_args()
    hdr, units, data = load(a.rep)
    idx = {h: i for i, h in enumerate(hdr)}
    name_col = idx.get("Kernel Name")
    per = defaultdict(list)
    for r in data:
        k = short(r[name_col])
        rec = {}
        for key, m in KEYS.items():
            if m in idx:
                v = num(r[idx[m]])
                if v is not None and units[idx[m]] in ("nsecond",) and key == "dur_us":
                    v /= 1e3
                elif v is not None and key == "dur_us" and units[idx[m]] == "usecond":
                    pass
                elif v is not None and key == "dur_us" and units[idx[m]] == "msecond":
                    v *= 1e3
                if v is not None and key == "sm_mhz":
                    u = units[idx[m]]
                    v = v / 1e6 if u in ("cycle/second", "") else (v * 1e3 if u == "cycle/nsecond" else v)
                rec[key] = v
        st = {}
        for h, i in idx.items():
            mm = STALL_RE.match(h)
            if mm:
                v = num(r[i])
                if v is not None:
                    st[mm.group(1)] = v
        rec["stalls"] = st
        per[k].append(rec)
    out = {}
    print(f"{'kernel':34s} {'n':>2s} {'dur us':>7s} {'MHz':>5s} {'IPC':>5s} {'issue%':>6s} {'SM%':>5s} {'mem%':>5s} "
          f"{'L1%':>5s} {'L2%':>5s} {'DRAM%':>5s} {'L2hit':>5s} {'L1hit':>5s} {'occ%':>5s}/{'th%':>4s} {'elig':>5s} "
          f"{'cyc/inst':>8s} {'regs':>4s} {'waves':>5s}")
    for k, recs in sorted(per.items(), key=lambda kv: -max(x.get("dur_us") or 0 for x in kv[1])):
        agg = {}
        for key in KEYS:
            vals = [x[key] for x in recs if x.get(key) is not None]
            agg[key] = sorted(vals)[len(vals) // 2] if vals else None
        stall_keys = set().union(*[x["stalls"].keys() for x in recs])
        agg["stalls"] = {s: sorted(x["stalls"].get(s, 0) for x in recs)[len(recs) // 2] for s in stall_keys}
        out[k] = {"n": len(recs), **agg}

        def f(key, w=5, p=0):
            v = agg.get(key)
            return f"{v:{w}.{p}f}" if v is not None else " " * (w - 1) + "-"
        print(f"{k:34s} {len(recs):2d} {f('dur_us', 7, 1)} {f('sm_mhz', 5)} {f('ipc_active', 5, 2)} {f('issue_active_pct', 6, 1)} "
              f"{f('sm_pct')} {f('mem_pct')} {f('l1_pct')} {f('l2_pct')} {f('dram_pct')} {f('l2_hit_pct')} {f('l1_hit_pct')} "
              f"{f('occ_achieved_pct')}/{f('occ_theor_pct', 4)} {f('eligible_warps', 5, 2)} {f('warp_cycles_per_inst', 8, 1)} "
              f"{f('regs', 4)} {f('waves', 5, 2)}")
        tot = sum(v for s, v in agg["stalls"].items() if s not in ("selected",)) or 1
        top = sorted(((v, s) for s, v in agg["stalls"].items()), reverse=True)[: a.stalls]
        print(" " * 37 + "stalls (cycles/issued inst): " + ", ".join(f"{s} {v:.2f}" for v, s in top))
        print(" " * 37 + f"pipes: fma {f('fp32_pipe_pct', 4)}% fmaheavy {f('fmaheavy_pct', 4)}% alu {f('alu_pipe_pct', 4)}% "
              f"xu {f('xu_pipe_pct', 4)}% lsu {f('lsu_pipe_pct', 4)}%; L2 red sectors {f('l2_red_sectors', 9)} "
              f"atom {f('l2_atom_sectors', 9)} all {f('l2_sectors', 9)}; DRAM bytes {f('dram_bytes', 9)}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
