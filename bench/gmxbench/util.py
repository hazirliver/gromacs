"""Small shared helpers: logging, subprocess execution, hashing, paths."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_ROOT = REPO_ROOT / "bench"

_t0 = time.time()
_verbose = 1


def set_verbosity(level: int) -> None:
    global _verbose
    _verbose = level


def log(msg: str, level: int = 1) -> None:
    if level <= _verbose:
        el = time.time() - _t0
        print(f"[gmxbench {el:7.1f}s] {msg}", file=sys.stderr, flush=True)


def die(msg: str) -> "None":
    print(f"gmxbench: error: {msg}", file=sys.stderr)
    sys.exit(2)


def workdir() -> Path:
    """Root for heavy, re-creatable state (worktrees, builds, prepared inputs, downloads)."""
    d = os.environ.get("GMXBENCH_WORKDIR")
    p = Path(d) if d else Path.home() / ".cache" / "gmxbench"
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_hash(obj, n: int = 12) -> str:
    """Short hash of a JSON-serialisable object, independent of dict ordering."""
    return sha256_bytes(json.dumps(obj, sort_keys=True, default=str).encode())[:n]


@dataclass
class ProcResult:
    cmd: list
    returncode: int
    stdout: str
    stderr: str
    wall: float
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


# Environment variables that silently change mdrun behaviour. They are removed
# from the inherited environment so that only explicitly requested ones apply.
_SCRUB_PREFIXES = ("GMX_", "OMP_", "GOMP_", "KMP_")


def clean_env(extra: dict | None = None) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(_SCRUB_PREFIXES)}
    env["GMX_MAXBACKUP"] = "-1"  # never create #backup# files
    if extra:
        env.update({k: str(v) for k, v in extra.items()})
    return env


def run(cmd, cwd: Path | None = None, env: dict | None = None, timeout: float | None = None,
        stdin: str | None = None, log_to: Path | None = None, check: bool = False) -> ProcResult:
    cmd = [str(c) for c in cmd]
    log(f"$ {' '.join(shlex.quote(c) for c in cmd)}" + (f"   (cwd={cwd})" if cwd else ""), 2)
    t = time.perf_counter()
    timed_out = False
    try:
        p = subprocess.run(cmd, cwd=cwd, env=env, input=stdin, capture_output=True, text=True,
                           timeout=timeout, errors="replace")
        rc, out, err = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        timed_out = True
        rc = -9
        out = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = (e.stderr or b"").decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
    wall = time.perf_counter() - t
    if log_to is not None:
        log_to.write_text(f"$ {' '.join(cmd)}\n# rc={rc} wall={wall:.2f}s timed_out={timed_out}\n"
                          f"--- stdout ---\n{out}\n--- stderr ---\n{err}\n")
    res = ProcResult(cmd, rc, out, err, wall, timed_out)
    if check and not res.ok:
        tail = "\n".join((err or out).splitlines()[-30:])
        raise RuntimeError(f"command failed (rc={rc}): {' '.join(cmd)}\n{tail}")
    return res


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=False, default=str))
    tmp.replace(path)


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def tier_value(v, tier: str):
    """Values in suite files may be scalars or {tier: value} tables."""
    if isinstance(v, dict) and tier in v:
        return v[tier]
    if isinstance(v, dict) and "default" in v:
        return v["default"]
    return v
