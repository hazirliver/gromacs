"""Theoretical occupancy and its limiter for the kernels in an nsys_steps.py analysis (no GPU counters needed).

    bench/profiling/occupancy.py ANALYSIS.json [--sm 89]

Uses the launch configuration recorded by Nsight Systems (block size, registers/thread, static + dynamic
shared memory) and the CUDA occupancy rules for the given SM (defaults: sm_89 / Ada: 1536 threads,
48 warps, 24 blocks, 64K registers, 100 KB shared memory per SM, 1 KB reserved per block, register
allocation in units of 256 per warp, warp allocation granularity 4 for registers).
Also reports waves = blocks / (blocks per SM x number of SMs).
"""
import argparse
import json
import math

ARCH = {89: dict(threads=1536, warps=48, blocks=24, regs=65536, smem=102400, smem_resv=1024,
                 reg_unit=256, smem_unit=128, max_regs_thread=255, sms=142)}


def occ(block, regs, smem, a):
    warps_b = math.ceil(block / 32)
    lim = {}
    lim["blocks"] = a["blocks"]
    lim["warps"] = a["warps"] // warps_b
    regs_w = math.ceil(max(regs, 1) * 32 / a["reg_unit"]) * a["reg_unit"]
    lim["registers"] = (a["regs"] // regs_w) // warps_b if regs_w else a["blocks"]
    smem_b = math.ceil((smem + a["smem_resv"]) / a["smem_unit"]) * a["smem_unit"]
    lim["shared_mem"] = a["smem"] // smem_b if smem > 0 else a["blocks"]
    bps = min(lim.values())
    limiter = sorted([k for k, v in lim.items() if v == bps])
    return bps, bps * warps_b / a["warps"], limiter, lim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis")
    ap.add_argument("--sm", type=int, default=89)
    ap.add_argument("--top", type=int, default=20)
    x = ap.parse_args()
    a = ARCH[x.sm]
    r = json.load(open(x.analysis))
    out = []
    print(f"{'us/step':>8} {'block':>5} {'regs':>4} {'smem B':>7} {'grid':>8} {'blk/SM':>6} {'occ%':>5} {'waves':>6}  limiter  kernel")
    for k in r["kernels"][:x.top]:
        if not isinstance(k.get("blockX"), int) or not isinstance(k.get("registersPerThread"), int):
            continue
        block = k["blockX"] * k.get("blockY", 1) * k.get("blockZ", 1)
        smem = (k.get("staticSharedMemory") or 0) + (k.get("dynamicSharedMemory") if isinstance(k.get("dynamicSharedMemory"), int) else 0)
        grid = k["gridX"] * k.get("gridY", 1) * k.get("gridZ", 1) if isinstance(k.get("gridX"), int) else None
        bps, o, lim, _ = occ(block, k["registersPerThread"], smem, a)
        waves = grid / (bps * a["sms"]) if grid and bps else None
        out.append({"kernel": k["name"], "us_per_step": k["us_per_step"], "block": block, "regs": k["registersPerThread"],
                    "smem": smem, "grid": grid, "blocks_per_sm": bps, "theoretical_occupancy": o, "limiter": lim, "waves": waves})
        print(f"{k['us_per_step']:8.1f} {block:5d} {k['registersPerThread']:4d} {smem:7d} {grid if grid else '-':>8} {bps:6d} "
              f"{100*o:5.1f} {waves if waves else 0:6.2f}  {','.join(lim):8s} {k['name'][:70]}")
    json.dump(out, open(x.analysis.replace(".json", ".occupancy.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
