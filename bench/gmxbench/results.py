"""Uniform results format.

Every invocation writes one *session* directory:

    session.json    metadata: command line, suite snapshot, host fingerprint, builds, timings,
                    and computed summaries (verdicts, statistical comparisons, recommendations)
    records.jsonl   one JSON object per measurement (schema below); append-only
    runs/           raw artefacts (md.log, mdrun stdout/stderr, nsys CSVs, ...)
    report.html     generated report

Record schema (v1), all keys always present:
    v        1
    session  session id
    suite    quality | perf | micro | sweep | upstream
    kind     sub-type, e.g. e2e, stage, gpu-kernel, nbnxm, bitwise, tolerance, ctest
    case     case name, e.g. "water-41k/pme"
    config   run configuration name, e.g. "gpu-resident"
    side     "A" (baseline), "B" (candidate), "S" (single build), or null
    build    build id of that side
    repeat   repeat index or null
    metric   metric name, e.g. "ns_per_day", "stage:PME mesh", "status"
    value    number, string or bool
    unit     unit string or null
    better   "higher" | "lower" | null  (direction of improvement for numeric metrics)
    tags     free-form dict with extra context (thread counts, kernel flavour, ...)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .hostinfo import fingerprint
from .util import BENCH_ROOT, log, read_json, write_json
from . import __version__


class Session:
    def __init__(self, path: Path, meta: dict):
        self.dir = path
        self.meta = meta
        self.runs = path / "runs"
        self.runs.mkdir(parents=True, exist_ok=True)
        self._rec = open(path / "records.jsonl", "a", buffering=1)
        self.builds: dict = {}

    @classmethod
    def create(cls, results: Path | None, command: str, suite, argv=None) -> "Session":
        sid = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{command}"
        path = Path(results).resolve() if results else BENCH_ROOT / "results" / sid
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "schema": "gmxbench/session/v1",
            "gmxbench_version": __version__,
            "id": path.name,
            "command": command,
            "argv": list(argv or []),
            "tier": suite.tier,
            "started": datetime.now().isoformat(timespec="seconds"),
            "finished": None,
            "host": fingerprint(),
            "builds": {},
            "suite": suite.snapshot(),
            "summary": {},
        }
        s = cls(path, meta)
        s.save()
        log(f"results -> {path}")
        return s

    def save(self) -> None:
        write_json(self.dir / "session.json", self.meta)

    def add_build(self, side: str, build) -> None:
        self.builds[side] = build
        info = build.info()
        self.meta["builds"][side] = info
        self.save()

    def record(self, suite: str, kind: str, metric: str, value, *, case=None, config=None, side=None,
               repeat=None, unit=None, better=None, tags=None) -> None:
        b = self.builds.get(side)
        rec = {"v": 1, "session": self.meta["id"], "suite": suite, "kind": kind, "case": case, "config": config,
               "side": side, "build": b.id if b is not None else None, "repeat": repeat, "metric": metric,
               "value": value, "unit": unit, "better": better, "tags": tags or {}}
        self._rec.write(json.dumps(rec, default=float) + "\n")

    def set_summary(self, section: str, obj) -> None:
        self.meta["summary"][section] = obj
        self.save()

    def close(self) -> None:
        self.meta["finished"] = datetime.now().isoformat(timespec="seconds")
        self.save()
        self._rec.close()


def load_session(path: Path) -> dict:
    path = Path(path)
    meta = read_json(path / "session.json")
    recs = []
    rp = path / "records.jsonl"
    if rp.exists():
        with open(rp) as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    meta["records"] = recs
    meta["dir"] = str(path)
    return meta
