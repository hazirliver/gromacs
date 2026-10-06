"""Isolated kernel micro-benchmarks (A/B).

Currently wraps `gmx nonbonded-benchmark`, which times the CPU Nbnxm pair kernels on synthetic
water systems for all 36 physics flavours (Coulomb type x LJ-on-half-atoms x combination rule x
interaction modifier), with and without energies. Each [[micro.nbnxm]] entry fixes size, threads
and cut-off; A and B runs are interleaved in seeded random order and every kernel flavour is
compared separately (FDR-controlled), plus a geometric-mean speedup over all flavours.

GPU kernels are covered per real system by the Nsight Systems pass of `gmxbench perf`, and CPU
stages (PME spread/gather/FFT, constraints, search, ...) by its cycle sub-counters.
"""

from __future__ import annotations

import csv
import fnmatch
import math
import random

import numpy as np

from .stats import apply_fdr, compare
from .util import clean_env, log, run


def _flavour(row: dict) -> str:
    return "/".join([row["Coulomb"], f"LJ-{row['LJ']}", row["comb"].rstrip("."), row["SIMD"], row["intmod"],
                     "VF" if row["compute energy"] == "yes" else "F"])


def nbnxm_run(build, outdir, size: int, threads: int, iters: int, warmup: int, cutoff: float, energy: bool,
              table: bool = False) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    csvp = outdir / "nb.csv"
    cmd = [build.gmx, "-quiet", "nonbonded-benchmark", "-size", size, "-nt", threads, "-iter", iters,
           "-warmup", warmup, "-cutoff", cutoff, "-all", "-o", csvp]
    if energy:
        cmd.append("-energy")
    if table:
        cmd.append("-table")
    env = clean_env({**build.env, "OMP_PROC_BIND": "close", "OMP_PLACES": "cores"})
    res = run(cmd, cwd=outdir, env=env, timeout=3600, log_to=outdir / "bench.out")
    if not res.ok or not csvp.exists():
        raise RuntimeError(f"nonbonded-benchmark failed: {res.stderr[-500:]}")
    out = {}
    with open(csvp) as f:
        for row in csv.DictReader(f):
            out[_flavour(row)] = {"mcycles_per_iter": float(row["Mcycles/it"]),
                                  "useful_pairs_per_cycle": float(row["useful pairs/cycle"])}
    csvp.unlink()
    return out


def run_micro(sess, suite, A, B, args) -> dict:
    m = suite.section("micro")
    repeats = args.repeats or int(suite.tv(m.get("repeats", 5)))
    alpha = float(m.get("alpha", 0.05))
    min_effect = float(m.get("min_effect", 0.01))
    rng = random.Random(getattr(args, "seed", 12345))
    summary = {"settings": {"repeats": repeats, "alpha": alpha, "min_effect": min_effect}, "nbnxm": []}
    if suite.target_mode == "only":  # synthetic kernels are not system-specific
        log("micro: skipped (--target-only)")
        return summary
    for ent in m.get("nbnxm", []):
        if not suite.in_tier(ent):
            continue
        if args.cases and not any(fnmatch.fnmatch(ent["name"], p) for p in args.cases):
            continue
        iters = int(suite.tv(ent.get("iter", 100)))
        log(f"micro {ent['name']}: size={ent['size']} nt={ent['threads']} iter={iters}")
        vals = {"A": {}, "B": {}}
        for energy in ent.get("energy", [False, True]):
            for rep in range(repeats):
                order = [("A", A), ("B", B)]
                rng.shuffle(order)
                for side, bld in order:
                    d = sess.runs / "micro" / ent["name"] / f"{'VF' if energy else 'F'}-rep{rep}-{side}"
                    try:
                        r = nbnxm_run(bld, d, int(ent["size"]), int(ent["threads"]), iters,
                                      int(ent.get("warmup", max(1, iters // 10))), float(ent.get("cutoff", 1.0)),
                                      energy)
                    except RuntimeError as e:  # keep going with the other entries
                        log(f"  {ent['name']} {side}: {str(e)[:300]}")
                        sess.record("micro", "nbnxm", "status", "error", case=ent["name"], side=side, repeat=rep,
                                    tags={"message": str(e)[:500]})
                        continue
                    for k, v in r.items():
                        vals[side].setdefault(k, []).append(v["mcycles_per_iter"])
                        sess.record("micro", "nbnxm", "mcycles_per_iter", v["mcycles_per_iter"], case=ent["name"],
                                    config=k, side=side, repeat=rep, unit="Mcycles/iter", better="lower",
                                    tags={"useful_pairs_per_cycle": v["useful_pairs_per_cycle"],
                                          "size_atoms": 3000 * int(ent["size"]), "threads": int(ent["threads"])})
        fam = []
        for k in sorted(set(vals["A"]) & set(vals["B"])):
            c = compare(vals["A"][k], vals["B"][k], "lower", alpha, min_effect, resolution=1e-4)
            c["metric"] = k
            fam.append(c)
        apply_fdr(fam, alpha)
        sps = [c["speedup"] for c in fam if "speedup" in c]
        geo = math.exp(float(np.mean(np.log(sps)))) if sps else None
        summary["nbnxm"].append({"name": ent["name"], "size_atoms": 3000 * int(ent["size"]),
                                 "threads": int(ent["threads"]), "iter": iters, "geomean_speedup": geo,
                                 "kernels": fam})
        if geo:
            n_f = sum(1 for c in fam if c.get("verdict_fdr") == "faster")
            n_s = sum(1 for c in fam if c.get("verdict_fdr") == "slower")
            log(f"  geomean speedup {geo:.4f} over {len(fam)} kernels ({n_f} faster, {n_s} slower after FDR)")
        sess.set_summary("micro", summary)
    sess.set_summary("micro", summary)
    return summary
