"""Quality (regression) suite.

For every case x configuration the same .tpr (made by the baseline grompp) is run by

    A   baseline build
    A2  baseline build again        -> is this configuration bitwise reproducible run-to-run?
    B   candidate build

and the outputs are compared at two levels:

1. Bitwise: sha256 of every TRR frame array (box, x, v, f at full precision), every EDR frame and the
   data lines of any .xvg output. If B == A bit for bit the verdict is IDENTICAL.
2. Numeric: if not identical, step-0 forces and energies (same input coordinates, so only the
   arithmetic differs) are compared against a tolerance. For configurations that are not
   reproducible run-to-run (e.g. GPU atomics), the tolerance is raised to a multiple of the measured
   A-vs-A2 noise floor. Verdict EQUIVALENT (within tolerance) or DIFFERENT. Trajectory divergence,
   conserved-energy drift and ensemble averages are recorded as supporting information.

The candidate's grompp is also checked: its .tpr must have the same content as the baseline's.

Baseline runs are cached by (baseline binary, runtime env, tpr, configuration, host), so repeated
checks against the same baseline only run the candidate.
"""

from __future__ import annotations

import fnmatch
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .gmx import GromppInput, cached_tpr, compare_tpr, grompp, mdrun, merge_mdp, parse_mdp, split_args
from .hostinfo import fingerprint, n_gpus
from .systems import SystemUnavailable, prepare, regressiontests_dir
from .trr import read_trr
from .util import log, read_json, sha256_bytes, sha256_file, stable_hash, workdir, write_json

STATUS_ORDER = ["FAIL", "DIFFERENT", "EQUIVALENT", "IDENTICAL", "SKIPPED"]


@dataclass
class QCase:
    name: str
    gi: GromppInput
    configs: list
    mdrun_extra: list = field(default_factory=list)
    natoms: int = 0
    tags: dict = field(default_factory=dict)
    tpr: Path | None = None


# ----------------------------------------------------------------------------------------------
# case expansion
# ----------------------------------------------------------------------------------------------
def _quality_output(nsteps: int, out_every: int, energy_every: int) -> dict:
    return {"nsteps": nsteps, "nstxout": out_every, "nstvout": out_every, "nstfout": out_every,
            "nstxout-compressed": 0, "nstlog": out_every, "nstenergy": energy_every,
            "nstcalcenergy": energy_every}


def _select(name: str, pats) -> bool:
    return not pats or any(fnmatch.fnmatch(name, p) for p in pats)


def expand_cases(suite, build, case_globs=None) -> list[QCase]:
    out = []
    for c in suite.cases("quality", case_globs):
        try:
            ps = prepare(suite, build, c["system"])
        except SystemUnavailable as e:
            log(f"quality case {c['name']}: skipped ({e})")
            continue
        except RuntimeError as e:
            log(f"quality case {c['name']}: system preparation failed: {e}")
            continue
        nsteps = int(suite.tv(c.get("nsteps", 200)))
        mdp = merge_mdp(suite.mdp_params(c.get("mdp", ["common", "@system"]), ps.base_mdp),
                        _quality_output(nsteps, int(suite.tv(c.get("out_every", 50))),
                                        int(suite.tv(c.get("energy_every", 10)))))
        gi = GromppInput(ps.conf, ps.top, mdp, ndx=ps.ndx, ref=ps.conf, maxwarn=int(c.get("maxwarn", 2)))
        out.append(QCase(c["name"], gi, list(c["configs"]), natoms=ps.natoms,
                         tags={"system": c["system"], "source": "suite"}))
    rt = suite.section("quality").get("regressiontests")
    if rt and suite.in_tier(rt) and suite.target_mode != "only":
        try:
            root = regressiontests_dir(build)
        except Exception as e:  # network or version problems must not block the rest of the suite
            log(f"regressiontests unavailable: {e}")
            return out
        skip = set(rt.get("skip", []))
        for set_name in rt.get("sets", ["complex"]):
            for d in sorted((root / set_name).iterdir()):
                name = f"rt/{set_name}/{d.name}"
                if not (d / "grompp.mdp").exists() or d.name in skip or not _select(name, case_globs):
                    continue
                conf = next((d / f for f in ("conf.gro", "conf.pdb", "conf.g96") if (d / f).exists()), None)
                if conf is None:
                    continue
                mdp = parse_mdp((d / "grompp.mdp").read_text())
                for k in ("ld-seed", "gen-seed"):
                    if mdp.get(k, "-1").strip() == "-1":
                        mdp[k] = "1993"
                extra = []
                if (d / "sam.edi").exists():
                    extra += ["-ei", str(d / "sam.edi")]
                configs = list(rt["configs"])
                if (d / "no-nb-gpu-support").exists():
                    configs = [x for x in configs if not suite.configs.get(x, {}).get("gpu")]
                maxranks = int((d / "max-mpi-ranks").read_text().split()[0]) if (d / "max-mpi-ranks").exists() else 99
                configs = [x for x in configs if _ranks(suite.configs.get(x, {})) <= maxranks]
                gi = GromppInput(conf, d / "topol.top", mdp, ndx=(d / "index.ndx") if (d / "index.ndx").exists() else None,
                                 ref=conf, maxwarn=10)
                natoms = 0
                if conf.suffix == ".gro":
                    with open(conf) as f:
                        f.readline()
                        natoms = int(f.readline())
                out.append(QCase(name, gi, configs, extra, natoms, {"system": name, "source": "regressiontests"}))
    return out


def _ranks(cfg: dict) -> int:
    a = split_args(cfg.get("args", ""))
    return int(a[a.index("-ntmpi") + 1]) if "-ntmpi" in a else 1


# ----------------------------------------------------------------------------------------------
# output analysis
# ----------------------------------------------------------------------------------------------
def _xvg_data_hash(p: Path) -> str:
    lines = [l for l in p.read_text(errors="replace").splitlines() if l and l[0] not in "#@"]
    return sha256_bytes("\n".join(lines).encode())[:16]


def _edr_frames(p: Path) -> dict:
    import pyedr
    return pyedr.edr_to_dict(str(p))


def digest_run(rundir: Path) -> dict:
    """Bitwise fingerprint of a run's outputs."""
    d = {"trr": [], "edr_sha": None, "edr_frames": [], "xvg": {}}
    trr = rundir / "run.trr"
    if trr.exists():
        for fr in read_trr(trr):
            d["trr"].append({"step": fr.step, **fr.digest()})
    edr = rundir / "run.edr"
    if edr.exists():
        d["edr_sha"] = sha256_file(edr)[:16]
        e = _edr_frames(edr)
        keys = sorted(k for k in e if k != "Time")
        mat = np.vstack([e[k] for k in keys]).T if keys else np.zeros((0, 0))
        d["edr_frames"] = [{"t": float(t), "h": sha256_bytes(row.tobytes())[:16]} for t, row in zip(e["Time"], mat)]
    for x in sorted(rundir.glob("*.xvg")):
        d["xvg"][x.name] = _xvg_data_hash(x)
    return d


def bitwise_compare(da: dict, db: dict) -> dict:
    res = {"identical": True, "first_diff_step": None, "diff_arrays": [], "frames_compared": 0}
    ta, tb = da["trr"], db["trr"]
    if len(ta) != len(tb):
        res["identical"] = False
        res["diff_arrays"].append("trr-frame-count")
    for fa, fb in zip(ta, tb):
        res["frames_compared"] += 1
        bad = [k for k in ("box", "x", "v", "f") if fa.get(k) != fb.get(k)]
        if bad or fa["step"] != fb["step"]:
            res["identical"] = False
            if res["first_diff_step"] is None:
                res["first_diff_step"] = fa["step"]
                res["diff_arrays"] += [f"trr:{k}" for k in bad]
    ea, eb = da["edr_frames"], db["edr_frames"]
    if len(ea) != len(eb):
        res["identical"] = False
        res["diff_arrays"].append("edr-frame-count")
    for fa, fb in zip(ea, eb):
        if fa["h"] != fb["h"]:
            res["identical"] = False
            res["first_diff_time_edr"] = fa["t"]
            res["diff_arrays"].append("edr")
            break
    for name in sorted(set(da["xvg"]) | set(db["xvg"])):
        if da["xvg"].get(name) != db["xvg"].get(name):
            res["identical"] = False
            res["diff_arrays"].append(f"xvg:{name}")
    res["diff_arrays"] = sorted(set(res["diff_arrays"]))
    return res


_NON_ENERGY_PREFIXES = ("Time", "Pres", "Vir", "Box", "Volume", "Density", "#Surf", "T-", "Lamb", "pV",
                        "Enthalpy", "Constr. rmsd", "Temperature", "Pressure")


_DERIVED_ENERGIES = {"Potential", "Total Energy", "Conserved En.", "Enthalpy", "pV", "Pres. DC (bar)",
                     "Constr. rmsd"}


def _min_image(d: np.ndarray, box: np.ndarray | None) -> np.ndarray:
    if box is None:
        return d
    diag = np.diag(box).astype(np.float64)
    diag[diag == 0] = np.inf
    return d - diag * np.round(d / diag)


def numeric_compare(ra: Path, rb: Path) -> dict:
    """Magnitude of differences between two runs of the same tpr."""
    out: dict = {}
    ta, tb = ra / "run.trr", rb / "run.trr"
    if ta.exists() and tb.exists():
        fa_list, fb_list = list(read_trr(ta)), list(read_trr(tb))
        f0a = next((f for f in fa_list if f.f is not None), None)
        f0b = next((f for f in fb_list if f.f is not None and f0a is not None and f.step == f0a.step), None)
        if f0a is not None and f0b is not None:
            fa, fb = f0a.f.astype(np.float64), f0b.f.astype(np.float64)
            rms = float(np.sqrt(np.mean(np.sum(fa * fa, axis=1)))) or 1.0
            df = np.linalg.norm(fb - fa, axis=1)
            out["f_step"] = int(f0a.step)
            out["f_rel_rms"] = float(np.sqrt(np.mean(df ** 2)) / rms)
            out["f_max_rel"] = float(df.max() / rms)
            out["f_rms"] = rms
        curve = []
        for fa, fb in zip(fa_list, fb_list):
            if fa.x is not None and fb.x is not None and fa.step == fb.step:
                dx = _min_image(fb.x.astype(np.float64) - fa.x.astype(np.float64),
                                fa.box.astype(np.float64) if fa.box is not None else None)
                curve.append([int(fa.step), float(np.sqrt(np.mean(np.sum(dx * dx, axis=1))))])
        out["x_rmsd_curve"] = curve
    ea, eb = ra / "run.edr", rb / "run.edr"
    if ea.exists() and eb.exists():
        A, B = _edr_frames(ea), _edr_frames(eb)
        n = min(len(A["Time"]), len(B["Time"]))
        terms = [k for k in A if k in B and not k.startswith(_NON_ENERGY_PREFIXES)]
        if n and terms:
            # Normalise by the summed magnitude of the component terms: derived sums such as Total
            # Energy or Conserved En. involve cancellation, so |dE|/|E| would be ill-conditioned.
            components = [k for k in terms if k not in _DERIVED_ENERGIES]
            scale = sum(abs(float(A[k][0])) for k in components) or 1.0
            worst, worst_term = 0.0, None
            for k in terms:
                rel = abs(float(B[k][0]) - float(A[k][0])) / scale
                if rel > worst:
                    worst, worst_term = rel, k
            out["e_scale"] = scale
            out["e_max_rel"] = worst
            out["e_max_rel_term"] = worst_term
            out["e_time"] = float(A["Time"][0])
            if "Potential" in A:
                dp = np.abs(B["Potential"][:n] - A["Potential"][:n])
                step = max(1, n // 200)
                out["epot_diff_curve"] = [[float(t), float(v)] for t, v in zip(A["Time"][:n:step], dp[::step])]
        cons = "Conserved En." if "Conserved En." in A else ("Total Energy" if "Total Energy" in A else None)
        if cons and n > 3:
            t = A["Time"][:n]
            out["drift_a"] = float(np.polyfit(t, A[cons][:n], 1)[0])  # kJ/mol/ps
            out["drift_b"] = float(np.polyfit(t, B[cons][:n], 1)[0])
        for k in ("Potential", "Temperature", "Pressure"):
            if k in A and k in B and n > 3:
                out[f"mean_{k}"] = [float(np.mean(A[k][:n])), float(np.mean(B[k][:n])), float(np.std(A[k][:n]))]
    return out


# ----------------------------------------------------------------------------------------------
# scheduling
# ----------------------------------------------------------------------------------------------
class _Resources:
    def __init__(self, cpus: int, gpu_slots: int):
        self.cpus, self.gpu = cpus, gpu_slots
        self.cv = threading.Condition()

    def acquire(self, cpus: int, gpu: bool):
        with self.cv:
            cpus = min(cpus, self.cpus_total)
            while self.cpus < cpus or (gpu and self.gpu < 1):
                self.cv.wait()
            self.cpus -= cpus
            self.gpu -= 1 if gpu else 0
        return cpus

    def release(self, cpus: int, gpu: bool):
        with self.cv:
            self.cpus += cpus
            self.gpu += 1 if gpu else 0
            self.cv.notify_all()


def _effective_args(case: QCase, cfg: dict) -> list:
    # mdrun rejects -reprod with GPU non-bondeds (GPU reductions use atomics); such configurations
    # are characterised by the A-vs-A2 determinism check and the noise-calibrated tolerance instead.
    reprod = ["-reprod"] if cfg_reprod(cfg) else []
    return split_args(cfg.get("args", "")) + reprod + ["-pin", "off"] + case.mdrun_extra


def _cache_dir(build, case: QCase, cfg_name: str, cfg: dict, rep: int) -> Path:
    key = stable_hash({"tpr": sha256_file(case.tpr), "args": _effective_args(case, cfg), "env": cfg.get("env"),
                       "host": fingerprint()["key"]}, 16)
    return workdir() / "runcache" / "quality" / build.run_id / f"{case.name.replace('/', '_')}-{cfg_name}-{key}" / f"run{rep}"


def _run_one(build, case: QCase, cfg: dict, rundir: Path, timeout: float) -> dict:
    done = rundir / "result.json"
    if done.exists():
        return read_json(done)
    if rundir.exists():
        shutil.rmtree(rundir)
    args = _effective_args(case, cfg)
    r = mdrun(build, case.tpr, rundir, args, env=cfg.get("env"), timeout=timeout)
    res = {"status": r.status, "message": r.message, "wall": r.wall, "cmd": r.cmd, "fallback": r.fallback,
           "setup": r.log.get("setup", {})}
    if r.ok:
        res["digest"] = digest_run(rundir)
    write_json(done, res)
    return res


# ----------------------------------------------------------------------------------------------
def run_quality(sess, suite, A, B, args) -> dict:
    q = suite.section("quality")
    tol = q.get("tolerance", {})
    tol_f = float(tol.get("force_rel_rms", 1e-5))
    tol_e = float(tol.get("energy_rel", 1e-5))
    noise_factor = float(tol.get("noise_factor", 10.0))
    timeout = float(q.get("timeout", 1800))
    have_gpu = n_gpus() > 0
    determinism = not getattr(args, "no_determinism_check", False)
    reuse = not getattr(args, "no_reuse", False)

    cases = expand_cases(suite, A, args.cases)
    log(f"quality: {len(cases)} cases")
    # tpr from the baseline grompp, candidate grompp checked against it
    grompp_checks = {}
    for c in cases:
        try:
            c.tpr = cached_tpr(A, c.gi, c.name.replace("/", "_"))
        except RuntimeError as e:
            log(f"quality case {c.name}: baseline grompp failed: {e}")
            grompp_checks[c.name] = {"status": "BASELINE_GROMPP_FAIL", "message": str(e)[:500]}
            continue
        tb = sess.runs / "quality" / c.name.replace("/", "_") / "grompp-B" / "topol.tpr"
        try:
            grompp(B, c.gi, tb)
            cmp = compare_tpr(A, c.tpr, tb)
            grompp_checks[c.name] = {"status": "IDENTICAL" if cmp["identical"] else "DIFFERENT", **cmp}
        except RuntimeError as e:
            grompp_checks[c.name] = {"status": "FAIL", "message": str(e)[:500]}
        sess.record("quality", "grompp", "tpr_status", grompp_checks[c.name]["status"], case=c.name, side="B",
                    tags={"n_differences": grompp_checks[c.name].get("n_differences")})
    cases = [c for c in cases if c.tpr is not None]

    # jobs
    jobs = []  # (case, cfg_name, side, build, rundir)
    skipped = []
    for c in cases:
        for cfg_name in c.configs:
            if not _select(cfg_name, args.configs):
                continue
            cfg = suite.configs.get(cfg_name)
            if cfg is None:
                raise SystemExit(f"case {c.name}: unknown config '{cfg_name}'")
            if cfg.get("gpu") and not have_gpu:
                skipped.append((c, cfg_name, "no GPU on this host"))
                continue
            base = sess.runs / "quality" / c.name.replace("/", "_") / cfg_name
            for side, bld, rep in (("A", A, 1), ("A2", A, 2), ("B", B, 1)):
                if side == "A2" and not determinism:
                    continue
                if side.startswith("A") and reuse:
                    rd = _cache_dir(bld, c, cfg_name, cfg, rep)
                else:
                    rd = base / side
                jobs.append((c, cfg_name, cfg, side, bld, rd))
    parallel = getattr(args, "parallel", 0) or max(1, (os.cpu_count() or 4) // 4)
    res_pool = _Resources(os.cpu_count() or 4, int(q.get("gpu_slots", 4)))
    res_pool.cpus_total = res_pool.cpus
    results: dict = {}
    # largest first to shorten the tail
    jobs.sort(key=lambda j: -(j[0].natoms or 1000) * int(j[0].gi.mdp.get("nsteps", 100)))
    t0 = time.time()

    def work(job):
        c, cfg_name, cfg, side, bld, rd = job
        need = int(cfg.get("cpus", 4))
        got = res_pool.acquire(need, bool(cfg.get("gpu")))
        try:
            return job, _run_one(bld, c, cfg, rd, timeout)
        finally:
            res_pool.release(got, bool(cfg.get("gpu")))

    log(f"quality: {len(jobs)} mdrun jobs, {parallel} concurrent")
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            (c, cfg_name, cfg, side, bld, rd), r = f.result()
            results[(c.name, cfg_name, side)] = (r, rd)
            if i % 20 == 0 or i == len(futs):
                log(f"quality: {i}/{len(futs)} runs done ({time.time() - t0:.0f}s)")

    # verdicts
    summary = {"tolerance": {"force_rel_rms": tol_f, "energy_rel": tol_e, "noise_factor": noise_factor},
               "grompp": grompp_checks, "results": [], "counts": {}}
    for c in cases:
        for cfg_name in c.configs:
            if not _select(cfg_name, args.configs):
                continue
            if any(s[0] is c and s[1] == cfg_name for s in skipped):
                row = {"case": c.name, "config": cfg_name, "status": "SKIPPED", "message": "no GPU on this host"}
            else:
                row = _verdict(c, cfg_name, cfg_reprod(suite.configs[cfg_name]), results, tol_f, tol_e,
                               noise_factor, determinism)
            row["natoms"] = c.natoms
            row["source"] = c.tags.get("source")
            row["grompp"] = grompp_checks.get(c.name, {}).get("status")
            summary["results"].append(row)
            tags = {k: row.get(k) for k in ("deterministic", "first_diff_step", "diff_arrays", "message", "source")}
            sess.record("quality", "verdict", "status", row["status"], case=c.name, config=cfg_name, side="B", tags=tags)
            for m in ("f_rel_rms", "f_max_rel", "e_max_rel", "noise_f_rel_rms", "noise_e_max_rel", "drift_a", "drift_b",
                      "wall_a", "wall_b"):
                if row.get(m) is not None:
                    sess.record("quality", "numeric", m, row[m], case=c.name, config=cfg_name, side="B")
    counts: dict = {}
    for r in summary["results"]:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    summary["counts"] = counts
    summary["grompp_counts"] = {}
    for g in grompp_checks.values():
        summary["grompp_counts"][g["status"]] = summary["grompp_counts"].get(g["status"], 0) + 1
    sess.set_summary("quality", summary)
    log(f"quality: {counts}; grompp: {summary['grompp_counts']}")
    return summary


def cfg_reprod(cfg: dict) -> bool:
    """Configurations run with -reprod (CPU) are bitwise reproducible by design."""
    return bool(cfg.get("reprod", not cfg.get("gpu")))


def _verdict(c: QCase, cfg_name: str, by_design: bool, results: dict, tol_f: float, tol_e: float,
             noise_factor: float, determinism: bool) -> dict:
    row = {"case": c.name, "config": cfg_name, "reprod_by_design": by_design}
    (ra, da), (rb, db) = results[(c.name, cfg_name, "A")], results[(c.name, cfg_name, "B")]
    ra2 = results.get((c.name, cfg_name, "A2"))
    row.update({"wall_a": ra.get("wall"), "wall_b": rb.get("wall"), "fallback": ra.get("fallback")})
    if ra["status"] != "ok":
        if rb["status"] == ra["status"]:
            row.update(status="SKIPPED" if ra["status"] == "unsupported" else "FAIL",
                       message=f"baseline and candidate: {ra['status']}: {ra['message']}")
        else:
            row.update(status="FAIL", message=f"baseline {ra['status']} ({ra['message']}), candidate {rb['status']}")
        return row
    if rb["status"] != "ok":
        row.update(status="FAIL", message=f"candidate {rb['status']}: {rb['message']}")
        return row
    bw = bitwise_compare(ra["digest"], rb["digest"])
    row.update(first_diff_step=bw["first_diff_step"], diff_arrays=bw["diff_arrays"])
    noise = {}
    if ra2 is not None and ra2[0]["status"] == "ok":
        same = bitwise_compare(ra["digest"], ra2[0]["digest"])["identical"]
        # Two identical GPU runs do not prove reproducibility (small systems can be lucky), so only
        # -reprod configurations are asserted deterministic; otherwise True becomes "not established".
        row["deterministic"] = same if by_design else (False if not same else None)
        if not same:
            noise = numeric_compare(da, ra2[1])
            row["noise_f_rel_rms"] = noise.get("f_rel_rms")
            row["noise_e_max_rel"] = noise.get("e_max_rel")
    else:
        row["deterministic"] = None
    if bw["identical"]:
        row["status"] = "IDENTICAL"
        return row
    num = numeric_compare(da, db)
    row.update({k: num.get(k) for k in ("f_rel_rms", "f_max_rel", "e_max_rel", "e_max_rel_term", "drift_a",
                                       "drift_b", "mean_Potential", "mean_Temperature")})
    row["curves"] = {"x_rmsd": num.get("x_rmsd_curve"), "epot_diff": num.get("epot_diff_curve")}
    tf = max(tol_f, noise_factor * (noise.get("f_rel_rms") or 0.0))
    te = max(tol_e, noise_factor * (noise.get("e_max_rel") or 0.0))
    row["tol_f"], row["tol_e"] = tf, te
    checks = []
    if num.get("f_rel_rms") is not None:
        checks.append(num["f_rel_rms"] <= tf)
    if num.get("e_max_rel") is not None:
        checks.append(num["e_max_rel"] <= te)
    if not checks:
        row.update(status="DIFFERENT", message="outputs differ and no step-0 forces/energies to bound the difference")
    else:
        row["status"] = "EQUIVALENT" if all(checks) else "DIFFERENT"
        if row["status"] == "DIFFERENT":
            row["message"] = (f"step-0 force rel-rms {num.get('f_rel_rms')} (tol {tf:.2e}), "
                              f"energy rel {num.get('e_max_rel')} [{num.get('e_max_rel_term')}] (tol {te:.2e})")
    return row
