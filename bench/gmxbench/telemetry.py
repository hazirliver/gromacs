"""GPU telemetry sampled while mdrun runs.

A background thread polls NVML (via the `pynvml` module of nvidia-ml-py; falls back to
`nvidia-smi` if unavailable) every `interval` seconds and records, per GPU:

    util_gpu %   util_mem %   vram_used MiB (device)   vram_proc MiB (the watched processes)
    power W      energy J (hardware counter)          sm_clock / mem_clock MHz
    temp C       throttle (reason bit mask)            pcie_tx / pcie_rx MB/s

`summary(t0, t1)` reduces the samples of a time window (e.g. mdrun's timed window between the
counter reset and the end of the run) to means/percentiles/maxima; energy comes from the hardware
energy counter (exact) when available, else from integrating power.
"""

from __future__ import annotations

import csv
import subprocess
import threading
import time
from pathlib import Path

try:
    import pynvml as _nv
    _nv.nvmlInit()
    _NVML = True
except Exception:  # no driver, no GPU or no bindings: degrade gracefully
    _NVML = False

FIELDS = ["t", "gpu", "util_gpu", "util_mem", "vram_used", "vram_proc", "power", "energy", "sm_clock", "mem_clock",
          "temp", "throttle", "pcie_tx", "pcie_rx"]

# NVML throttle-reason bits that indicate the GPU was slowed down (idle/app-clock bits excluded)
THROTTLE_SLOWDOWN = {0x4: "sw_power_cap", 0x8: "hw_slowdown", 0x20: "sw_thermal", 0x40: "hw_thermal",
                     0x80: "hw_power_brake"}


def available() -> bool:
    if _NVML:
        return _nv.nvmlDeviceGetCount() > 0
    try:
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


class GpuMonitor:
    def __init__(self, interval: float = 0.1, gpus: list[int] | None = None):
        self.interval = interval
        self.samples: list[dict] = []
        self.pids: set[int] = set()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if _NVML:
            n = _nv.nvmlDeviceGetCount()
            self.gpus = gpus if gpus is not None else list(range(n))
            self._handles = {g: _nv.nvmlDeviceGetHandleByIndex(g) for g in self.gpus}
        else:
            self.gpus = gpus if gpus is not None else [0]
            self._handles = {}

    # -- lifecycle -------------------------------------------------------------------------------
    def start(self) -> "GpuMonitor":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def watch_pid(self, pid: int) -> None:
        self.pids.add(pid)

    # -- sampling --------------------------------------------------------------------------------
    def _loop(self) -> None:
        # Keep the sampler out of mdrun's way: lowest priority, on the last logical CPU (pinned mdrun
        # runs start at CPU 0). Both calls act on this thread only (Linux semantics for id 0 / tid).
        import os
        try:
            os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 19)
        except (AttributeError, OSError):
            pass
        nxt = time.monotonic()
        while not self._stop.is_set():
            try:
                self.samples.extend(self._sample_nvml() if _NVML else self._sample_smi())
            except Exception:
                pass  # a missed sample must never break a benchmark run
            nxt += self.interval
            self._stop.wait(max(0.0, nxt - time.monotonic()))

    def _sample_nvml(self) -> list[dict]:
        out = []
        t = time.monotonic()
        for g, h in self._handles.items():
            u = _nv.nvmlDeviceGetUtilizationRates(h)
            mem = _nv.nvmlDeviceGetMemoryInfo(h)
            proc = 0
            if self.pids:
                try:
                    for p in _nv.nvmlDeviceGetComputeRunningProcesses(h):
                        if p.pid in self.pids and p.usedGpuMemory:
                            proc += p.usedGpuMemory
                except _nv.NVMLError:
                    pass
            try:
                energy = _nv.nvmlDeviceGetTotalEnergyConsumption(h) / 1000.0
            except _nv.NVMLError:
                energy = None
            try:
                tx = _nv.nvmlDeviceGetPcieThroughput(h, _nv.NVML_PCIE_UTIL_TX_BYTES) / 1024.0
                rx = _nv.nvmlDeviceGetPcieThroughput(h, _nv.NVML_PCIE_UTIL_RX_BYTES) / 1024.0
            except _nv.NVMLError:
                tx = rx = None
            out.append({"t": t, "gpu": g, "util_gpu": u.gpu, "util_mem": u.memory,
                        "vram_used": mem.used / 2 ** 20, "vram_proc": proc / 2 ** 20,
                        "power": _nv.nvmlDeviceGetPowerUsage(h) / 1000.0, "energy": energy,
                        "sm_clock": _nv.nvmlDeviceGetClockInfo(h, _nv.NVML_CLOCK_SM),
                        "mem_clock": _nv.nvmlDeviceGetClockInfo(h, _nv.NVML_CLOCK_MEM),
                        "temp": _nv.nvmlDeviceGetTemperature(h, _nv.NVML_TEMPERATURE_GPU),
                        "throttle": _nv.nvmlDeviceGetCurrentClocksThrottleReasons(h), "pcie_tx": tx, "pcie_rx": rx})
        return out

    def _sample_smi(self) -> list[dict]:
        q = "index,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu"
        r = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"], capture_output=True,
                           text=True, timeout=5)
        t = time.monotonic()
        out = []
        for line in r.stdout.splitlines():
            v = [x.strip() for x in line.split(",")]
            if len(v) != 8 or int(v[0]) not in self.gpus:
                continue
            f = lambda x: float(x) if x not in ("", "[N/A]", "N/A") else None
            out.append({"t": t, "gpu": int(v[0]), "util_gpu": f(v[1]), "util_mem": f(v[2]), "vram_used": f(v[3]),
                        "vram_proc": None, "power": f(v[4]), "energy": None, "sm_clock": f(v[5]),
                        "mem_clock": f(v[6]), "temp": f(v[7]), "throttle": None, "pcie_tx": None, "pcie_rx": None})
        return out

    # -- reduction -------------------------------------------------------------------------------
    def summary(self, t0: float | None = None, t1: float | None = None) -> dict:
        """Per-window statistics, aggregated over the monitored GPUs (sum of power/VRAM, mean utilisation)."""
        import numpy as np
        win = [s for s in self.samples if (t0 is None or s["t"] >= t0) and (t1 is None or s["t"] <= t1)]
        if not win:
            return {}
        out: dict = {"samples": len(win), "duration_s": (win[-1]["t"] - win[0]["t"]) if len(win) > 1 else 0.0}
        by_t: dict = {}
        for s in win:
            by_t.setdefault(s["t"], []).append(s)
        ticks = [by_t[t] for t in sorted(by_t)]

        def series(key, agg):
            vals = []
            for tick in ticks:
                xs = [s[key] for s in tick if s.get(key) is not None]
                if xs:
                    vals.append(agg(xs))
            return np.asarray(vals, dtype=float)

        for key, agg in (("util_gpu", np.mean), ("util_mem", np.mean), ("sm_clock", np.mean), ("mem_clock", np.mean),
                         ("power", np.sum), ("vram_used", np.sum), ("vram_proc", np.sum), ("temp", np.max),
                         ("pcie_tx", np.sum), ("pcie_rx", np.sum)):
            v = series(key, agg)
            if len(v):
                out[f"{key}_mean"] = float(v.mean())
                out[f"{key}_p50"] = float(np.percentile(v, 50))
                out[f"{key}_p95"] = float(np.percentile(v, 95))
                out[f"{key}_max"] = float(v.max())
                out[f"{key}_min"] = float(v.min())
        # energy: exact hardware counter difference per GPU, else trapezoid integral of power
        e_total = 0.0
        exact = True
        for g in self.gpus:
            gs = [s for s in win if s["gpu"] == g]
            es = [s["energy"] for s in gs if s.get("energy") is not None]
            if len(es) >= 2:
                e_total += es[-1] - es[0]
            elif len(gs) >= 2:
                exact = False
                t = np.asarray([s["t"] for s in gs])
                p = np.asarray([s["power"] or 0.0 for s in gs])
                e_total += float(np.sum((p[1:] + p[:-1]) / 2 * np.diff(t)))
        out["energy_j"] = e_total
        out["energy_exact"] = exact
        thr = 0
        for s in win:
            thr |= int(s.get("throttle") or 0)
        out["throttle_reasons"] = [name for bit, name in THROTTLE_SLOWDOWN.items() if thr & bit]
        return out

    def write_csv(self, path: Path, t_origin: float | None = None) -> None:
        t0 = t_origin if t_origin is not None else (self.samples[0]["t"] if self.samples else 0.0)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for s in self.samples:
                w.writerow({**s, "t": round(s["t"] - t0, 3)})

    def timeseries(self, t0: float, t1: float, origin: float, max_points: int = 240) -> dict:
        """Downsampled (time, util, power, vram) series of GPU 0-aggregate for plotting."""
        by_t: dict = {}
        for s in self.samples:
            if t0 <= s["t"] <= t1:
                by_t.setdefault(s["t"], []).append(s)
        ts = sorted(by_t)
        step = max(1, len(ts) // max_points)
        out = {"t": [], "util_gpu": [], "power": [], "vram_used": []}
        for t in ts[::step]:
            tick = by_t[t]
            out["t"].append(round(t - origin, 2))
            out["util_gpu"].append(round(sum(s["util_gpu"] or 0 for s in tick) / len(tick), 1))
            out["power"].append(round(sum(s["power"] or 0 for s in tick), 1))
            out["vram_used"].append(round(sum(s["vram_used"] or 0 for s in tick), 0))
        return out
