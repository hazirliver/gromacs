"""Analyse sassprof output: per-kernel dynamic instruction profile, pipe-utilisation model, memory
efficiency and (with -lineinfo builds) attribution to source lines / code regions.

    sassanalyze.py kernel DIR [--regions REGIONS.toml] [--duration-us T --clock-mhz F] [--json OUT]
    sassanalyze.py table DIR... [--durations durations.json] [--json OUT]

DIR is one run_targets.sh output directory (sass_metrics.tsv, launches.tsv, cubins/). Counts are per
launch of the profiled kernel (sum over all its launches in the window / number of launches).

Pipe model (sm_89 / Ada, per SM per cycle; 4 SMSPs, 1 warp instruction issued per SMSP per cycle):
  issue  4.0 warp instructions
  fma    4.0 (FFMA/FMUL/FADD/HFMA2...; IMAD-type only on the "heavy" half: counted 2x)
  alu    2.0 (integer add/logic/shift/compare, FSETP/FSEL/FMNMX, SEL, PRMT, MOV-type)
  xu     0.5 (MUFU, conversions I2F/F2I/FRND)
  lsu    1.0 (shared/global/local memory instructions, SHFL, atomics) - plus wavefront/sector costs below
  tex    0.25 (TEX/TLD/TLD4: assumption, 4 texture lookups per SM per cycle)
  cbu    1.0 (branch/barrier/warp-sync instructions)
Utilisation = cycles the pipe needs / kernel cycles (duration x SM clock), i.e. a *lower bound* on that
pipe's busy fraction under perfect scheduling. These throughputs are the documented/commonly used Ada
values; they are a model, not counters (Nsight Compute is blocked by DCGM on this node).
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
import sys
from pathlib import Path

NSM = 142  # L40S

PIPE_RATE = {"issue": 4.0, "fma": 4.0, "alu": 2.0, "xu": 0.5, "lsu": 1.0, "tex": 0.25, "cbu": 1.0}

FMA_OPS = {"FFMA", "FMUL", "FADD", "HFMA2", "HMUL2", "HADD2", "FCHK"}
FMA_HEAVY_OPS = {"IMAD", "IMUL", "IDP", "IMNMX_HEAVY"}
ALU_OPS = {"IADD3", "IADD", "LOP3", "LOP", "SHF", "SHL", "SHR", "ISETP", "FSETP", "FSEL", "FMNMX", "IMNMX",
           "SEL", "PRMT", "LEA", "LEA.HI", "MOV", "IABS", "FSET", "ISET", "PLOP3", "P2R", "R2P", "FLO", "POPC",
           "BREV", "BMSK", "SGXT", "VOTE", "VOTEU", "CS2R", "S2R", "S2UR", "R2UR", "UMOV", "ULDC", "UIADD3",
           "UIMAD", "ULOP3", "USHF", "ULEA", "UISETP", "USEL", "UPRMT", "UFLO", "UPOPC", "UBMSK", "USGXT",
           "UP2UR", "UR2UP", "UPLOP3", "FRND", "F2FP", "IMNMX", "NOP"}
XU_OPS = {"MUFU", "I2F", "F2I", "F2F", "I2I", "I2FP", "F2IP"}
LSU_OPS = {"LDS", "STS", "LDG", "STG", "LD", "ST", "LDL", "STL", "ATOM", "ATOMS", "ATOMG", "RED", "SHFL", "LDSM",
           "LDGSTS", "LDC", "MATCH", "MEMBAR", "CCTL", "ERRBAR", "FENCE"}
TEX_OPS = {"TEX", "TLD", "TLD4", "TXQ", "TXD", "TMML"}
CBU_OPS = {"BRA", "BRX", "JMP", "JMX", "CALL", "RET", "EXIT", "BSSY", "BSYNC", "WARPSYNC", "BAR", "BPT", "YIELD",
           "NANOSLEEP", "KILL", "BMOV", "RPCMOV", "BREAK", "WARPGROUP"}


def opcode_base(op: str) -> str:
    return op.split(".")[0]


def classify(op: str) -> str:
    b = opcode_base(op)
    if b in FMA_OPS:
        return "fma"
    if b in FMA_HEAVY_OPS:
        return "fma_heavy"
    if b in XU_OPS:
        return "xu"
    if b in TEX_OPS:
        return "tex"
    if b in LSU_OPS:
        return "lsu"
    if b in CBU_OPS:
        return "cbu"
    if b in ALU_OPS or b.startswith("U"):
        return "alu"
    return "alu"


def opcode_family(op: str) -> str:
    """Finer grouping for the mix table."""
    b = opcode_base(op)
    if b in ("FFMA", "FMUL", "FADD"):
        return b
    if b == "MUFU":
        return "MUFU." + op.split(".")[1] if "." in op else "MUFU"
    if b in ("LDG", "STG", "LDS", "STS", "RED", "ATOM", "ATOMS", "SHFL", "TLD", "TEX", "LDL", "STL", "LDC"):
        return b
    if b in ("FSETP", "FSEL", "FMNMX"):
        return "FP compare/select"
    if b in ("ISETP", "IADD3", "LOP3", "SHF", "LEA", "IMAD", "SEL", "PRMT", "MOV", "IMNMX"):
        return "INT " + b
    if b.startswith("U"):
        return "uniform datapath"
    if b in CBU_OPS:
        return "control " + b
    return "other " + b


INSTR_RE = re.compile(r"^\s*/\*([0-9a-f]{4,})\*/\s+(@!?U?P[T0-9]+\s+)?([A-Z0-9_.]+)(.*?);")
LINE_RE = re.compile(r'//## File "([^"]+)", line (\d+)(?: inlined at "([^"]+)", line (\d+))?')


def disassemble(cubin: Path, function: str) -> dict[int, dict]:
    """pc -> {op, text, pred, loc: [(file, line), ...] innermost first}."""
    out = subprocess.run(["nvdisasm", "-c", "-gi", str(cubin)], capture_output=True, text=True, check=True).stdout
    res: dict[int, dict] = {}
    inside = False
    chain: list[tuple[str, int]] = []
    pending_reset = False
    for line in out.splitlines():
        if line.startswith("//--------------------- .text."):
            inside = line.split(".text.", 1)[1].split(" ", 1)[0] == function
            continue
        if not inside:
            continue
        m = LINE_RE.search(line)
        if m:
            if pending_reset:
                chain = []
                pending_reset = False
            chain.append((m.group(1), int(m.group(2))))
            continue
        m = INSTR_RE.match(line)
        if m:
            pc = int(m.group(1), 16)
            # the comment block lists the innermost location first and the outermost (kernel) location last
            res[pc] = {"op": m.group(3), "pred": (m.group(2) or "").strip(), "text": line.strip(), "loc": list(chain)}
            pending_reset = True
    return res


def short(path: str) -> str:
    p = path.split("/src/gromacs/", 1)
    return p[1] if len(p) == 2 else Path(path).name


def load_regions(path: Path | None) -> list[dict]:
    if not path:
        return []
    import tomllib
    return tomllib.loads(path.read_text()).get("region", [])


def region_of(loc: list[tuple[str, int]], regions: list[dict]) -> str:
    """First region (in file order of the TOML) that matches any frame of the inline chain, innermost first."""
    for fr_file, fr_line in loc:
        sf = short(fr_file)
        for r in regions:
            if sf.endswith(r["file"]) and r["lines"][0] <= fr_line <= r["lines"][1]:
                return r["name"]
    return "other"


def analyse_kernel(d: Path, regions: list[dict] | None = None, duration_us: float | None = None,
                   clock_mhz: float | None = None) -> dict:
    rows = [l.rstrip("\n").split("\t") for l in open(d / "sass_metrics.tsv")][1:]
    if not rows:
        return {"dir": str(d), "error": "no data"}
    fn = rows[0][2]
    crc = rows[0][0]
    per_pc: dict[int, dict[str, int]] = collections.defaultdict(dict)
    for crc_, fidx, f, pc, metric, value in rows:
        if f != fn:
            continue
        per_pc[int(pc)][metric] = per_pc[int(pc)].get(metric, 0) + int(value)
    launches = [l.rstrip("\n").split("\t") for l in open(d / "launches.tsv")][1:]
    nl = sum(1 for l in launches if l[0] == "driver" and l[2] == fn)
    grids = collections.Counter((l[3], l[4], l[5], l[6], l[7], l[8]) for l in launches if l[0] == "driver" and l[2] == fn)
    dis = disassemble(d / "cubins" / f"{crc}.cubin", fn)
    regions = regions or []

    tot = collections.Counter()
    by_class = collections.Counter()
    by_family = collections.Counter()
    by_region = collections.defaultdict(collections.Counter)
    by_line = collections.defaultdict(collections.Counter)
    mem_rows = []
    static_n = len(dis)
    for pc, m in per_pc.items():
        info = dis.get(pc, {"op": "?", "text": "?", "loc": []})
        w = m.get("smsp__sass_inst_executed", 0)
        t = m.get("smsp__sass_thread_inst_executed", 0)
        tp = m.get("smsp__sass_thread_inst_executed_pred_on", 0)
        for k, v in m.items():
            tot[k] += v
        cls = classify(info["op"])
        by_class[cls] += w
        by_family[opcode_family(info["op"])] += w
        reg = region_of(info["loc"], regions)
        for k, v in m.items():
            by_region[reg][k] += v
        by_region[reg]["class_" + cls] += w
        if info["loc"]:
            f0, l0 = info["loc"][0]
            fo, lo = info["loc"][-1]
            key = f"{short(f0)}:{l0}" + (f" <- {short(fo)}:{lo}" if (f0, l0) != (fo, lo) else "")
        else:
            key = "?"
        by_line[key]["w"] += w
        by_line[key]["t"] += t
        if any(k in m for k in ("smsp__sass_sectors_mem_global", "smsp__sass_l1tex_pipe_lsu_wavefronts_mem_shared")):
            mem_rows.append({"pc": pc, "op": info["op"], "text": info["text"], "region": reg, "loc": key, "warp_inst": w,
                             "sectors": m.get("smsp__sass_sectors_mem_global", 0),
                             "sectors_ideal": m.get("smsp__sass_sectors_mem_global_ideal", 0),
                             "l1_tags": m.get("smsp__sass_l1tex_tags_mem_global", 0),
                             "shared_wavefronts": m.get("smsp__sass_l1tex_pipe_lsu_wavefronts_mem_shared", 0),
                             "shared_wavefronts_ideal": m.get("smsp__sass_l1tex_pipe_lsu_wavefronts_mem_shared_ideal", 0)})
    n = max(nl, 1)
    w_tot = tot["smsp__sass_inst_executed"]
    res = {
        "dir": str(d), "function": fn, "cubin_crc": crc, "launches": nl,
        "launch_configs": {"x".join(k[:3]) + " / " + "x".join(k[3:]): c for k, c in grids.items()},
        "static_instructions": static_n, "patched_pcs": len(per_pc),
        "per_launch": {k: v / n for k, v in tot.items()},
        "simt_efficiency": tot["smsp__sass_thread_inst_executed"] / (32 * w_tot) if w_tot else None,
        "predicated_on_efficiency": tot["smsp__sass_thread_inst_executed_pred_on"] / (32 * w_tot) if w_tot else None,
        "branch_divergent_fraction": (tot["smsp__sass_branch_targets_threads_divergent"]
                                      / max(1, tot["smsp__sass_branch_targets_threads_divergent"]
                                            + tot["smsp__sass_branch_targets_threads_uniform"])),
        "global_sector_efficiency": (tot["smsp__sass_sectors_mem_global_ideal"] / tot["smsp__sass_sectors_mem_global"]
                                     if tot["smsp__sass_sectors_mem_global"] else None),
        "shared_wavefront_efficiency": (tot["smsp__sass_l1tex_pipe_lsu_wavefronts_mem_shared_ideal"]
                                        / tot["smsp__sass_l1tex_pipe_lsu_wavefronts_mem_shared"]
                                        if tot["smsp__sass_l1tex_pipe_lsu_wavefronts_mem_shared"] else None),
        "local_sectors_per_launch": tot["smsp__sass_sectors_mem_local"] / n,
        "class_per_launch": {k: v / n for k, v in by_class.items()},
        "family_per_launch": {k: v / n for k, v in sorted(by_family.items(), key=lambda kv: -kv[1])},
        "regions": {r: {"warp_inst": c["smsp__sass_inst_executed"] / n,
                        "share": c["smsp__sass_inst_executed"] / w_tot if w_tot else 0,
                        "simt_efficiency": (c["smsp__sass_thread_inst_executed"] / (32 * c["smsp__sass_inst_executed"])
                                            if c["smsp__sass_inst_executed"] else None),
                        "classes": {k[6:]: v / n for k, v in c.items() if k.startswith("class_")}}
                    for r, c in sorted(by_region.items(), key=lambda kv: -kv[1]["smsp__sass_inst_executed"])},
        "top_lines": [{"loc": k, "warp_inst": v["w"] / n, "share": v["w"] / w_tot if w_tot else 0,
                       "simt": v["t"] / (32 * v["w"]) if v["w"] else None}
                      for k, v in sorted(by_line.items(), key=lambda kv: -kv[1]["w"])[:40]],
        "memory_instructions": sorted(mem_rows, key=lambda r: -max(r["sectors"], r["shared_wavefronts"], r["warp_inst"]))[:40],
    }
    # pipe model
    pipes = collections.Counter()
    pipes["issue"] = w_tot
    pipes["fma"] = by_class["fma"] + 2 * by_class["fma_heavy"]
    pipes["alu"] = by_class["alu"]
    pipes["xu"] = by_class["xu"]
    pipes["lsu"] = by_class["lsu"]
    pipes["tex"] = by_class["tex"]
    pipes["cbu"] = by_class["cbu"]
    res["pipe_cycles_per_sm_per_launch"] = {p: pipes[p] / n / PIPE_RATE[p] / NSM for p in PIPE_RATE}
    # memory-side lower bounds: L1 handles 1 shared wavefront and 4 global sectors (128 B) per cycle per SM
    res["l1_cycles_per_sm_per_launch"] = {
        "shared_wavefronts": tot["smsp__sass_l1tex_pipe_lsu_wavefronts_mem_shared"] / n / NSM,
        "global_sectors_div4": tot["smsp__sass_sectors_mem_global"] / n / NSM / 4,
        "global_tag_lookups": tot["smsp__sass_l1tex_tags_mem_global"] / n / NSM}
    if duration_us and clock_mhz:
        cyc = duration_us * clock_mhz
        res["timing"] = {"duration_us": duration_us, "clock_mhz": clock_mhz, "cycles": cyc,
                         "ipc_per_sm": w_tot / n / NSM / cyc,
                         "pipe_utilisation": {p: c / cyc for p, c in res["pipe_cycles_per_sm_per_launch"].items()},
                         "l1_utilisation": {p: c / cyc for p, c in res["l1_cycles_per_sm_per_launch"].items()}}
    return res


def print_kernel(r: dict):
    if "error" in r:
        print(r["dir"], r["error"])
        return
    pl = r["per_launch"]
    print(f"== {r['function'][:100]}")
    print(f"launches {r['launches']}  configs {r['launch_configs']}  static {r['static_instructions']}  patched {r['patched_pcs']}")
    print(f"warp inst/launch {pl.get('smsp__sass_inst_executed', 0):,.0f}  thread inst/launch "
          f"{pl.get('smsp__sass_thread_inst_executed', 0):,.0f}  SIMT eff {r['simt_efficiency']:.3f}  pred-on eff "
          f"{r['predicated_on_efficiency']:.3f}  branch divergent {r['branch_divergent_fraction']:.3f}")
    ge = r["global_sector_efficiency"]
    se = r["shared_wavefront_efficiency"]
    print(f"global sectors/launch {pl.get('smsp__sass_sectors_mem_global', 0):,.0f} (eff {ge if ge is None else round(ge, 3)})  "
          f"shared wavefronts/launch {pl.get('smsp__sass_l1tex_pipe_lsu_wavefronts_mem_shared', 0):,.0f} (eff "
          f"{se if se is None else round(se, 3)})  local sectors/launch {r['local_sectors_per_launch']:,.0f}")
    print("class mix (warp inst/launch): " + ", ".join(f"{k} {v:,.0f}" for k, v in sorted(r["class_per_launch"].items(), key=lambda kv: -kv[1])))
    print("pipe cycles/SM/launch: " + ", ".join(f"{k} {v:,.0f}" for k, v in r["pipe_cycles_per_sm_per_launch"].items()))
    print("L1 cycles/SM/launch: " + ", ".join(f"{k} {v:,.0f}" for k, v in r["l1_cycles_per_sm_per_launch"].items()))
    if "timing" in r:
        t = r["timing"]
        print(f"timing {t['duration_us']} us @ {t['clock_mhz']} MHz = {t['cycles']:,.0f} cycles; IPC/SM {t['ipc_per_sm']:.2f}; "
              + "util " + ", ".join(f"{k} {v:.2f}" for k, v in t["pipe_utilisation"].items())
              + "; L1 " + ", ".join(f"{k} {v:.2f}" for k, v in t["l1_utilisation"].items()))
    if r["regions"] and list(r["regions"]) != ["other"]:
        print("regions:")
        for k, v in r["regions"].items():
            print(f"  {k:<28} {v['warp_inst']:>14,.0f} {100 * v['share']:6.1f}%  SIMT {v['simt_efficiency'] or 0:.2f}  "
                  + " ".join(f"{c}:{x:,.0f}" for c, x in sorted(v["classes"].items(), key=lambda kv: -kv[1])))
    print("top source lines:")
    for l in r["top_lines"][:15]:
        print(f"  {100 * l['share']:5.1f}%  SIMT {l['simt'] or 0:.2f}  {l['loc']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("kernel")
    p.add_argument("dir", type=Path)
    p.add_argument("--regions", type=Path)
    p.add_argument("--duration-us", type=float)
    p.add_argument("--clock-mhz", type=float)
    p.add_argument("--json", type=Path)
    p = sub.add_parser("table")
    p.add_argument("dirs", type=Path, nargs="+")
    p.add_argument("--durations", type=Path, help="json {target: [duration_us, clock_mhz]}")
    p.add_argument("--regions", type=Path)
    p.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "kernel":
        r = analyse_kernel(a.dir, load_regions(a.regions), a.duration_us, a.clock_mhz)
        print_kernel(r)
        if a.json:
            a.json.write_text(json.dumps(r, indent=1))
    else:
        dur = json.loads(a.durations.read_text()) if a.durations else {}
        out = {}
        for d in a.dirs:
            du = dur.get(d.name)
            r = analyse_kernel(d, load_regions(a.regions), *(du or (None, None)))
            out[d.name] = r
            print_kernel(r)
            print()
        if a.json:
            a.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    sys.exit(main())
