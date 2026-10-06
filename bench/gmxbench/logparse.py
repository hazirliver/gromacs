"""Parsing of mdrun md.log files: performance, cycle accounting (incl. sub-counters), GPU timings,
and the run setup mdrun actually chose (nstlist, offload, update location, PME tuning)."""

from __future__ import annotations

import re
from pathlib import Path

_NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
# " Name    ranks threads  count   wall  gcycles  %"  (ranks/threads/count optional, e.g. "Rest")
_CYCLE_ROW = re.compile(
    rf"^ (?P<name>\S.*?)\s{{2,}}(?:(?P<ranks>\d+)\s+(?P<threads>\d+)\s+)?(?:(?P<count>\d+)\s+)?"
    rf"(?P<wall>{_NUM})\s+(?P<gcyc>{_NUM})\s+(?P<pct>{_NUM})\s*$")
# GPU timings: " Name   count   wall   ms/step   %"
_GPU_ROW = re.compile(rf"^ (?P<name>\S.*?)\s{{2,}}(?:(?P<count>\d+)\s+)?(?P<wall>{_NUM})\s+(?P<msstep>{_NUM})\s+(?P<pct>{_NUM})\s*$")


def parse_log(path: Path) -> dict:
    text = Path(path).read_text(errors="replace")
    lines = text.splitlines()
    out: dict = {"stages": {}, "gpu_timings": {}, "setup": {}, "notes": 0, "warnings": 0}

    # -- performance summary ---------------------------------------------------------------------
    for i, line in enumerate(lines):
        if line.startswith("Performance:"):
            hdr = lines[i - 1]
            vals = [float(v) for v in line.split()[1:]]
            names = re.findall(r"\(([^)]+)\)", hdr)
            for n, v in zip(names, vals):
                key = {"ns/day": "ns_per_day", "hour/ns": "hour_per_ns", "ms/step": "ms_per_step",
                       "Matom*steps/s": "matom_steps_per_s"}.get(n, n)
                out[key] = v
        elif line.strip().startswith("Time:") and "Core t" in lines[i - 1]:
            parts = line.split()
            out["core_time_s"], out["wall_time_s"] = float(parts[1]), float(parts[2])

    # -- cycle accounting --------------------------------------------------------------------------
    try:
        start = next(i for i, l in enumerate(lines) if "R E A L   C Y C L E" in l)
    except StopIteration:
        start = None
    if start is not None:
        section = ""
        for line in lines[start + 1:]:
            if line.startswith("Breakdown of") or line.startswith(" Breakdown of"):
                section = line.strip().replace("Breakdown of ", "").replace(" activities", "")
                continue
            if line.strip().startswith(("Core t", "Time:", "Performance:")) or "GPU timings" in line:
                break
            m = _CYCLE_ROW.match(line)
            if not m or m.group("name").startswith(("Activity", "Computing")):
                continue
            name = m.group("name").strip()
            key = f"{section}: {name}" if section else name
            out["stages"][key] = {"wall_s": float(m.group("wall")), "pct": float(m.group("pct")),
                                  "count": int(m.group("count")) if m.group("count") else None,
                                  "gcycles": float(m.group("gcyc"))}

    # -- GPU timings (GMX_ENABLE_GPU_TIMING) --------------------------------------------------------
    for i, line in enumerate(lines):
        if line.strip() in ("GPU timings", "PME GPU timings"):
            prefix = "PME GPU" if "PME" in line else "NB GPU"
            for l2 in lines[i + 1:i + 40]:
                if not l2.strip():
                    break
                m = _GPU_ROW.match(l2)
                if m and not m.group("name").startswith("Computing"):
                    out["gpu_timings"][f"{prefix}: {m.group('name').strip()}"] = {
                        "wall_s": float(m.group("wall")), "ms_per_step": float(m.group("msstep")),
                        "count": int(m.group("count")) if m.group("count") else None}

    # -- setup chosen by mdrun ---------------------------------------------------------------------
    s = out["setup"]
    for line in lines:
        if m := re.search(r"Changing nstlist from (\d+) to (\d+), rlist from (\S+) to (\S+)", line):
            s["nstlist"], s["rlist"] = int(m.group(2)), float(m.group(4).rstrip(","))
        elif m := re.match(r"^Using (\d+) MPI (?:thread|process)", line):
            s["ranks"] = int(m.group(1))
        elif m := re.match(r"^Using (\d+) OpenMP threads?", line):
            s.setdefault("omp_threads", int(m.group(1)))
        elif line.startswith("Updating coordinates") and "on the GPU" in line:
            s["update"] = "gpu"
        elif m := re.match(r"^Using (?:a |)GPU .*?(nonbonded|PME)", line):
            s.setdefault("gpu_tasks", []).append(m.group(1))
        elif m := re.search(r"Mapping of GPU IDs to the (\d+) GPU tasks? in the (\d+) ranks?.*", line):
            s["gpu_task_count"] = int(m.group(1))
        elif "PP:" in line and ("PME:" in line or line.strip().startswith("PP:")):
            s.setdefault("gpu_mapping", line.strip())
        elif m := re.search(r"Using (\S+) non-bonded kernels?|Using SIMD (\S+) nonbonded short-range kernels", line):
            s["nb_kernel"] = (m.group(1) or m.group(2))
        elif m := re.search(r"optimal pme grid (\d+) (\d+) (\d+), coulomb cutoff (\S+)", line):
            s["pme_tuned_grid"] = [int(m.group(i)) for i in (1, 2, 3)]
            s["pme_tuned_rc"] = float(m.group(4))
        elif m := re.match(r"^\s+SIMD instructions:\s+(\S+)", line):
            s["simd"] = m.group(1)
        if line.startswith("NOTE"):
            out["notes"] += 1
        elif line.startswith("WARNING"):
            out["warnings"] += 1
    # "CUDA Graphs will be used, provided ..." is conditional; the "MD Graph" timer shows actual use
    s["cuda_graphs"] = any(k.startswith("MD Graph") for k in out["stages"])
    out["finished"] = "Finished mdrun" in text
    return out


_UNSUPPORTED = [
    "is not supported", "not implemented", "can not run on the GPU", "cannot run on the GPU",
    "Inconsistency in user input", "Cannot run", "requires", "not compatible", "No GPU was detected",
    "Domain decomposition does not support", "is too small", "There is no domain decomposition",
    "incompatible",
]


def classify_failure(text: str) -> str:
    """'unsupported' when mdrun rejected the configuration, 'error' otherwise."""
    m = re.search(r"Fatal error:\s*\n(.*?)(?:\n\s*\n|For more information)", text, re.S)
    msg = m.group(1) if m else text[-2000:]
    if any(k in msg for k in _UNSUPPORTED):
        return "unsupported"
    return "error"


def fatal_message(text: str) -> str:
    m = re.search(r"Fatal error:\s*\n(.*?)(?:\n\s*\n|For more information)", text, re.S)
    if m:
        return " ".join(m.group(1).split())[:500]
    tail = [l for l in text.splitlines() if l.strip()][-5:]
    return " | ".join(tail)[:500]
