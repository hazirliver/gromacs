"""Minimal, exact reader for GROMACS .trr files (XDR, big-endian).

Arrays are returned as views of the on-disk bytes (dtype '>f4' or '>f8'), so hashing them is
bit-exact and nothing is lost to conversions.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_MAGIC = 1993


@dataclass
class TrrFrame:
    step: int
    time: float
    lam: float
    natoms: int
    double: bool
    box: np.ndarray | None
    x: np.ndarray | None
    v: np.ndarray | None
    f: np.ndarray | None

    def digest(self) -> dict:
        """Per-array sha256 digests of the raw bytes (only arrays present in the frame)."""
        out = {}
        for k in ("box", "x", "v", "f"):
            a = getattr(self, k)
            if a is not None:
                out[k] = hashlib.sha256(a.tobytes()).hexdigest()[:16]
        return out


def _i32(buf, off):
    return int.from_bytes(buf[off:off + 4], "big", signed=True), off + 4


def read_trr(path: Path):
    """Yield TrrFrame objects."""
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    buf = memoryview(raw)
    off, n = 0, len(raw)
    while off < n:
        magic, off = _i32(buf, off)
        if magic != _MAGIC:
            raise ValueError(f"{path}: bad trr magic {magic} at offset {off - 4}")
        _slen, off = _i32(buf, off)
        strlen, off = _i32(buf, off)
        off += (strlen + 3) // 4 * 4
        hdr = []
        for _ in range(13):
            v, off = _i32(buf, off)
            hdr.append(v)
        (ir_size, e_size, box_size, vir_size, pres_size, top_size, sym_size,
         x_size, v_size, f_size, natoms, step, nre) = hdr
        if box_size:
            fsize = box_size // 9
        elif x_size:
            fsize = x_size // (natoms * 3)
        elif v_size:
            fsize = v_size // (natoms * 3)
        elif f_size:
            fsize = f_size // (natoms * 3)
        else:
            raise ValueError(f"{path}: cannot determine precision")
        dt = np.dtype(">f8" if fsize == 8 else ">f4")
        t = float(np.frombuffer(buf[off:off + fsize], dtype=dt)[0])
        lam = float(np.frombuffer(buf[off + fsize:off + 2 * fsize], dtype=dt)[0])
        off += 2 * fsize
        off += ir_size + e_size  # always zero in practice

        def take(size, shape):
            nonlocal off
            if not size:
                return None
            a = np.frombuffer(buf[off:off + size], dtype=dt).reshape(shape)
            off += size
            return a

        box = take(box_size, (3, 3))
        take(vir_size, (3, 3))
        take(pres_size, (3, 3))
        off += top_size + sym_size
        x = take(x_size, (natoms, 3))
        v = take(v_size, (natoms, 3))
        f = take(f_size, (natoms, 3))
        yield TrrFrame(step, t, lam, natoms, fsize == 8, box, x, v, f)


def frame_digests(path: Path) -> list[dict]:
    return [{"step": fr.step, **fr.digest()} for fr in read_trr(path)]
