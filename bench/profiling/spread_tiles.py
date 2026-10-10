#!/usr/bin/env python3
"""Offline model of PME charge spreading for MAS1: how many global grid updates (and 32-byte sectors)
would a shared-memory-accumulating spread issue, compared with the production kernel?

Production kernel (ewald/pme_spread.cu, order 4, ThreadsPerAtom::Order): 64 atoms per block in the
PME atom order (= topology order with GPU update), one global atomicAdd per atom per stencil point
(4x4x4 = 64 per atom). Alternatives modelled:
  A. blocks of 64 atoms in a spatial order (nbnxm-like: xy columns, z-sorted, 64-atom cells) that
     accumulate their stencils in shared memory, then add each touched grid point once;
  B. grid tiles (cuFINUFFT "SM"/subproblem style): atoms binned by tile of their stencil origin; each
     tile accumulates (tile + 3 halo) points in shared memory, then adds the non-zero ones once.
Counts are exact for the given coordinates; time is not modelled here.

    spread_tiles.py GRO NX NY NZ [--order 4]
"""
import argparse
import math

import numpy as np


def read_gro(path):
    with open(path) as f:
        f.readline()
        n = int(f.readline())
        xyz = np.empty((n, 3))
        for i in range(n):
            line = f.readline()
            xyz[i] = (float(line[20:28]), float(line[28:36]), float(line[36:44]))
        box = np.array([float(x) for x in f.readline().split()[:3]])
    return xyz, box


def stencil_points(origin, n, order):
    """Flattened grid indices (x-major, z fastest) of each atom's order^3 stencil, wrapped."""
    nx, ny, nz = n
    k = np.arange(order)
    ix = (origin[:, 0, None] + k) % nx
    iy = (origin[:, 1, None] + k) % ny
    iz = (origin[:, 2, None] + k) % nz
    idx = (ix[:, :, None, None] * ny + iy[:, None, :, None]) * nz + iz[:, None, None, :]
    return idx.reshape(len(origin), -1)


def sectors_of(idx, nz, pnz):
    """32-byte sector ids of grid indices with the z dimension padded to pnz floats."""
    xy, z = np.divmod(idx, nz)
    return (xy * pnz + z) // 8


def per_block_union(points, block, nz, pnz):
    nb = math.ceil(len(points) / block)
    upd = sec = 0
    sizes = []
    for b in range(nb):
        p = np.unique(points[b * block:(b + 1) * block].ravel())
        upd += len(p)
        sec += len(np.unique(sectors_of(p, nz, pnz)))
        sizes.append(len(p))
    return upd, sec, np.array(sizes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gro")
    ap.add_argument("nx", type=int)
    ap.add_argument("ny", type=int)
    ap.add_argument("nz", type=int)
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--pnz", type=int, default=0, help="padded z size of the real grid (default nz)")
    a = ap.parse_args()
    xyz, box = read_gro(a.gro)
    n = np.array([a.nx, a.ny, a.nz])
    pnz = a.pnz or a.nz
    N = len(xyz)
    frac = (xyz / box) % 1.0
    origin = np.floor(frac * n).astype(np.int64)  # stencil origin (the exact shift does not matter)
    pts = stencil_points(origin, n, a.order)
    S = a.order ** 3
    print(f"atoms {N}, box {box}, grid {n} (spacing {box / n}), order {a.order}: {S} updates per atom")
    tot_upd = N * S
    tot_sec_prod = None

    # production: 64-atom blocks in topology order, every update global; sectors per warp instruction
    # (8 atoms x 4 z-points per warp instruction for a given (ithx, ithy))
    def warp_sectors(order_idx):
        p = pts[order_idx]
        nw = math.ceil(N / 8)
        total = 0
        for (ix, iy) in [(x, y) for x in range(a.order) for y in range(a.order)]:
            sub = p.reshape(N, a.order, a.order, a.order)[:, ix, iy, :]  # (N, 4 z-points)
            sec = sectors_of(sub, a.nz, pnz)
            pad = (-N) % 8
            if pad:
                sec = np.vstack([sec, np.full((pad, a.order), -1)])
            sec = sec.reshape(nw, 8 * a.order)
            s = np.sort(sec, axis=1)
            total += int(((s[:, 1:] != s[:, :-1]).sum(axis=1) + 1).sum())
        return total

    topo = np.arange(N)
    ws = warp_sectors(topo)
    print(f"\nproduction (topology order): {tot_upd / 1e6:.2f} M global updates, {ws / 1e6:.2f} M sector "
          f"touches by warp instructions ({ws / (tot_upd / 32):.2f} sectors per warp instruction)")

    # spatial order, nbnxm-like
    rho = N / box.prod()
    for cell_atoms in (64,):
        sxy = (cell_atoms / rho) ** (1 / 3)
        ncx, ncy = int(box[0] / sxy), int(box[1] / sxy)
        cx = np.minimum((frac[:, 0] * ncx).astype(int), ncx - 1)
        cy = np.minimum((frac[:, 1] * ncy).astype(int), ncy - 1)
        col = cx * ncy + cy
        order_sp = np.lexsort((xyz[:, 2], col))
        ws_sp = warp_sectors(order_sp)
        print(f"\nspatial order (xy columns {ncx}x{ncy}, z-sorted): sector touches {ws_sp / 1e6:.2f} M "
              f"({ws_sp / (tot_upd / 32):.2f} per warp instruction) with the production kernel")
        for block in (32, 64, 128, 256):
            for name, ordr in (("topology", topo), ("spatial", order_sp)):
                upd, sec, sizes = per_block_union(pts[ordr], block, a.nz, pnz)
                print(f"  A: blocks of {block:3d} atoms, {name:8s} order, smem accumulation: "
                      f"{upd / 1e6:6.2f} M global updates ({tot_upd / upd:5.2f}x fewer), {sec / 1e6:5.2f} M sectors; "
                      f"distinct points per block p50 {np.median(sizes):.0f} max {sizes.max()}")

    # grid tiles
    print("\nB: grid tiles (atoms binned by stencil origin), smem tile incl. halo")
    for t in ((4, 4, 4), (8, 8, 8), (8, 8, 16), (16, 16, 16), (16, 16, 32)):
        t = np.array(t)
        tid = origin // t
        nt = np.ceil(n / t).astype(int)
        key = (tid[:, 0] * nt[1] + tid[:, 1]) * nt[2] + tid[:, 2]
        o = np.argsort(key, kind="stable")
        ks = key[o]
        bounds = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1], True])
        upd = sec = 0
        counts = np.diff(bounds)
        for b0, b1 in zip(bounds[:-1], bounds[1:]):
            p = np.unique(pts[o[b0:b1]].ravel())
            upd += len(p)
            sec += len(np.unique(sectors_of(p, a.nz, pnz)))
        halo = t + a.order - 1
        print(f"  tile {tuple(t)}: {len(counts)} tiles (of {nt.prod()}), atoms/tile p50 {np.median(counts):.0f} "
              f"p99 {np.percentile(counts, 99):.0f} max {counts.max()}; smem {halo.prod() * 4 / 1024:.1f} KiB; "
              f"{upd / 1e6:.2f} M global updates ({tot_upd / upd:.1f}x fewer), {sec / 1e6:.2f} M sectors "
              f"({ws / sec:.1f}x fewer than production's sector touches)")


if __name__ == "__main__":
    main()
