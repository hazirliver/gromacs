"""Host hardware/software fingerprint and cheap dynamic state snapshots."""

from __future__ import annotations

import os
import platform
import re
import subprocess
from functools import lru_cache

from .util import stable_hash


def _cmd(args, timeout=20) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _read(path: str) -> str:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


GPU_STATIC_FIELDS = ["index", "name", "uuid", "driver_version", "memory.total", "compute_cap", "clocks.max.sm",
                     "clocks.max.mem", "power.limit", "persistence_mode", "pcie.link.gen.max", "pcie.link.width.max"]
GPU_DYNAMIC_FIELDS = ["index", "temperature.gpu", "clocks.sm", "clocks.mem", "power.draw", "utilization.gpu",
                      "memory.used", "clocks_throttle_reasons.active"]


def _gpu_query(fields: list[str]) -> list[dict]:
    out = _cmd(["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"])
    gpus = []
    for line in out.splitlines():
        vals = [v.strip() for v in line.split(",")]
        if len(vals) == len(fields):
            gpus.append(dict(zip(fields, vals)))
    return gpus


@lru_cache(maxsize=1)
def fingerprint() -> dict:
    lscpu = {}
    for line in _cmd(["lscpu"]).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            lscpu[k.strip()] = v.strip()
    mem_kb = re.search(r"MemTotal:\s+(\d+)", _read("/proc/meminfo"))
    nvcc = _cmd(["nvcc", "--version"])
    cuda = re.search(r"release ([\d.]+)", nvcc)
    fp = {
        "hostname": platform.node(),
        "os": _read("/etc/os-release").split("PRETTY_NAME=")[-1].split("\n")[0].strip('"') or platform.platform(),
        "kernel": platform.release(),
        "virtualization": _cmd(["systemd-detect-virt"]) or "unknown",
        "cpu": {
            "model": lscpu.get("Model name", platform.processor()),
            "logical_cpus": os.cpu_count(),
            "threads_per_core": lscpu.get("Thread(s) per core"),
            "cores_per_socket": lscpu.get("Core(s) per socket"),
            "sockets": lscpu.get("Socket(s)"),
            "numa_nodes": lscpu.get("NUMA node(s)"),
            "l3": lscpu.get("L3 cache"),
            "governor": _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor") or "n/a",
            "avx512": "avx512f" in lscpu.get("Flags", ""),
        },
        "memory_gb": round(int(mem_kb.group(1)) / 1024 / 1024, 1) if mem_kb else None,
        "gpus": _gpu_query(GPU_STATIC_FIELDS),
        "cuda_toolkit": cuda.group(1) if cuda else None,
        "python": platform.python_version(),
    }
    fp["key"] = stable_hash({"cpu": fp["cpu"]["model"], "n": fp["cpu"]["logical_cpus"],
                             "gpus": [(g.get("name"), g.get("driver_version")) for g in fp["gpus"]]}, 10)
    return fp


def n_gpus() -> int:
    return len(fingerprint()["gpus"])


def physical_cores() -> int:
    fp = fingerprint()["cpu"]
    try:
        return int(fp["cores_per_socket"]) * int(fp["sockets"])
    except (TypeError, ValueError):
        return os.cpu_count() or 1


def snapshot() -> dict:
    """Dynamic state recorded with every timed run (thermal/clock throttling diagnostics)."""
    snap = {"loadavg": os.getloadavg()[0]}
    for g in _gpu_query(GPU_DYNAMIC_FIELDS):
        i = g.pop("index")
        snap.update({f"gpu{i}.{k}": v for k, v in g.items()})
    return snap
