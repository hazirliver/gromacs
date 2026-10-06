"""Command line interface.

    gmxbench build    <spec> [--profile P]               build a git ref / WORKTREE / path:DIR
    gmxbench prepare  --build SPEC [--systems ...]         generate test systems (grompp inputs)
    gmxbench quality  --baseline A --candidate B           bitwise + tolerance regression check
    gmxbench perf     --baseline A --candidate B           statistical end-to-end + stage + kernel A/B
    gmxbench micro    --baseline A --candidate B           isolated kernel micro-benchmarks A/B
    gmxbench sweep    --build A                            find best mdrun arguments per system
    gmxbench upstream --build A                            run GROMACS' own ctest unit/regression tests
    gmxbench all      --baseline A --candidate B           everything above + report
    gmxbench report   RESULTS_DIR [...]                    (re)generate the HTML report
    gmxbench env                                           print the host fingerprint
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import builds
from .suite import Suite, DEFAULT_SUITE
from .util import BENCH_ROOT, die, log, set_verbosity


def _env_pairs(items: list[str] | None) -> dict:
    out = {}
    for it in items or []:
        if "=" not in it:
            die(f"expected KEY=VALUE, got '{it}'")
        k, v = it.split("=", 1)
        out[k] = v
    return out


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--suite", action="append", type=Path,
                   help=f"suite TOML file(s); later ones override (default: {DEFAULT_SUITE.relative_to(BENCH_ROOT.parent)})")
    p.add_argument("--tier", default="quick", choices=["smoke", "quick", "full"],
                   help="case selection / run length (default: quick)")
    p.add_argument("--profile", default="cuda", help="build profile from the suite (default: cuda)")
    p.add_argument("--results", type=Path, default=None,
                   help="results directory (default: bench/results/<timestamp>-<command>)")
    p.add_argument("--jobs", type=int, default=0, help="build parallelism (default: ncpu-2)")
    tg = p.add_mutually_exclusive_group()
    tg.add_argument("--target-only", action="store_true",
                    help="run only the cases of the target systems ([target] in suites/target.toml)")
    tg.add_argument("--no-target", action="store_true",
                    help="do not add the target systems' cases (they run by default on every invocation)")
    p.add_argument("-v", "--verbose", action="count", default=1)
    p.add_argument("-q", "--quiet", action="store_true")


def _add_gates(p: argparse.ArgumentParser) -> None:
    p.add_argument("--strict", action="store_true",
                   help="quality: also fail when a bitwise-reproducible configuration is not IDENTICAL")
    p.add_argument("--fail-on-slowdown", action="store_true",
                   help="perf: exit non-zero if any case/config is significantly slower")


def _add_ab(p: argparse.ArgumentParser) -> None:
    p.add_argument("--baseline", "-A", required=True, help="baseline build spec (git ref, WORKTREE, src:DIR, path:DIR)")
    p.add_argument("--candidate", "-B", required=True, help="candidate build spec")
    p.add_argument("--baseline-env", action="append", metavar="K=V", help="runtime env for baseline runs")
    p.add_argument("--candidate-env", action="append", metavar="K=V", help="runtime env for candidate runs")
    p.add_argument("--baseline-profile", help="override build profile for the baseline")
    p.add_argument("--candidate-profile", help="override build profile for the candidate")
    p.add_argument("--cases", nargs="*", help="glob(s) selecting case names")
    p.add_argument("--configs", nargs="*", help="glob(s) selecting run configurations")


def _resolve_ab(args, suite: Suite):
    a = builds.resolve(args.baseline, args.baseline_profile or args.profile, suite.profiles, label="A:" + args.baseline,
                       env=_env_pairs(args.baseline_env))
    b = builds.resolve(args.candidate, args.candidate_profile or args.profile, suite.profiles,
                       label="B:" + args.candidate, env=_env_pairs(args.candidate_env))
    for x in (a, b):
        builds.ensure_built(x, jobs=args.jobs or None)
    return a, b


def _session(args, command: str, suite: Suite):
    from .results import Session
    return Session.create(args.results, command, suite, argv=sys.argv)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="gmxbench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build", help="build GROMACS for a spec")
    _add_common(p)
    p.add_argument("spec", nargs="+")
    p.add_argument("--target", action="append", default=None, help="cmake target(s) (default: gmx)")

    p = sub.add_parser("prepare", help="generate test systems")
    _add_common(p)
    p.add_argument("--build", default="WORKTREE")
    p.add_argument("--systems", nargs="*")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("quality", help="bitwise/tolerance A/B regression check")
    _add_common(p)
    _add_ab(p)
    p.add_argument("--no-determinism-check", action="store_true",
                   help="skip the second baseline run that measures run-to-run reproducibility")
    p.add_argument("--parallel", type=int, default=0, help="concurrent mdrun jobs (default: auto)")
    p.add_argument("--no-reuse", action="store_true", help="do not reuse cached baseline runs")
    _add_gates(p)

    p = sub.add_parser("perf", help="statistical end-to-end A/B performance comparison")
    _add_common(p)
    _add_ab(p)
    p.add_argument("--repeats", type=int, default=0)
    p.add_argument("--no-nsys", action="store_true", help="skip GPU kernel profiling runs")
    p.add_argument("--seed", type=int, default=12345, help="seed for run-order randomisation")
    _add_gates(p)

    p = sub.add_parser("micro", help="isolated kernel micro-benchmarks A/B")
    _add_common(p)
    _add_ab(p)
    p.add_argument("--repeats", type=int, default=0)

    p = sub.add_parser("sweep", help="search mdrun arguments per system")
    _add_common(p)
    p.add_argument("--build", default="WORKTREE")
    p.add_argument("--build-env", action="append", metavar="K=V")
    p.add_argument("--systems", nargs="*", help="glob(s) selecting sweep entries")
    p.add_argument("--repeats", type=int, default=0)

    p = sub.add_parser("upstream", help="run GROMACS ctest suites (unit + regressiontests) and record results")
    _add_common(p)
    p.add_argument("--build", default="WORKTREE")
    p.add_argument("--no-regressiontests", action="store_true")
    p.add_argument("--ctest-args", default="", help="extra ctest arguments, e.g. '-L QuickGpuTest'")

    p = sub.add_parser("all", help="quality + perf + micro (+ optional sweep/upstream) + report")
    _add_common(p)
    _add_ab(p)
    p.add_argument("--with-sweep", action="store_true")
    p.add_argument("--with-upstream", action="store_true")
    p.add_argument("--repeats", type=int, default=0)
    p.add_argument("--no-nsys", action="store_true")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--parallel", type=int, default=0)
    p.add_argument("--no-determinism-check", action="store_true")
    p.add_argument("--no-reuse", action="store_true")
    _add_gates(p)

    p = sub.add_parser("report", help="generate an HTML report from one or more results directories")
    p.add_argument("results", nargs="+", type=Path)
    p.add_argument("-o", "--output", type=Path)
    p.add_argument("-v", "--verbose", action="count", default=1)
    p.add_argument("-q", "--quiet", action="store_true")

    p = sub.add_parser("list", help="list systems, configurations and the cases of each suite for a tier")
    _add_common(p)

    p = sub.add_parser("env", help="print host/software fingerprint")
    p.add_argument("-v", "--verbose", action="count", default=1)
    p.add_argument("-q", "--quiet", action="store_true")

    args = ap.parse_args(argv)
    set_verbosity(0 if args.quiet else args.verbose)

    if args.cmd == "env":
        import json
        from .hostinfo import fingerprint
        print(json.dumps(fingerprint(), indent=2))
        return
    if args.cmd == "report":
        from .report import write_report
        out = write_report(args.results, args.output)
        print(out)
        return

    mode = "only" if args.target_only else ("exclude" if args.no_target else "include")
    suite = Suite(args.suite, args.tier, mode)
    if mode != "exclude" and args.cmd not in ("build", "list"):
        from .systems import unavailable_reason
        for name in suite.target_systems:
            why = unavailable_reason(suite, name)
            if why:
                log(f"WARNING: target system {name} unavailable - its cases will be skipped: {why}")

    if args.cmd == "list":
        _list(suite)
        return

    if args.cmd == "build":
        for spec in args.spec:
            b = builds.resolve(spec, args.profile, suite.profiles)
            builds.ensure_built(b, jobs=args.jobs or None, targets=tuple(args.target or ["gmx"]))
            print(f"{spec}\t{b.gmx}")
        return

    if args.cmd == "prepare":
        from .systems import prepare_all
        b = builds.ensure_built(builds.resolve(args.build, args.profile, suite.profiles), jobs=args.jobs or None)
        for name, sysd in prepare_all(suite, b, args.systems, force=args.force).items():
            print(f"{name}\t{sysd}")
        return

    if args.cmd == "sweep":
        from .sweep import run_sweep
        b = builds.ensure_built(builds.resolve(args.build, args.profile, suite.profiles,
                                               env=_env_pairs(args.build_env)), jobs=args.jobs or None)
        sess = _session(args, "sweep", suite)
        sess.add_build("S", b)
        run_sweep(sess, suite, b, args)
        _finish(sess)
        return

    if args.cmd == "upstream":
        from .upstream import run_upstream
        b = builds.resolve(args.build, args.profile, suite.profiles)
        sess = _session(args, "upstream", suite)
        run_upstream(sess, suite, b, args)
        _finish(sess)
        return

    a, b = _resolve_ab(args, suite)
    sess = _session(args, args.cmd, suite)
    sess.add_build("A", a)
    sess.add_build("B", b)
    if args.cmd in ("quality", "all"):
        from .quality import run_quality
        run_quality(sess, suite, a, b, args)
    if args.cmd in ("perf", "all"):
        from .perf import run_perf
        run_perf(sess, suite, a, b, args)
    if args.cmd in ("micro", "all"):
        from .micro import run_micro
        run_micro(sess, suite, a, b, args)
    if args.cmd == "all" and args.with_sweep:
        from .sweep import run_sweep
        sess.add_build("S", b)
        run_sweep(sess, suite, b, args)
    if args.cmd == "all" and args.with_upstream:
        from .upstream import run_upstream
        run_upstream(sess, suite, b, args)
    _finish(sess)
    sys.exit(_gate(sess, args))


def _gate(sess, args) -> int:
    """Exit status for scripts/CI: 1 if a quality or (optionally) performance gate failed."""
    rc = 0
    q = sess.meta["summary"].get("quality")
    if q:
        bad = [r for r in q["results"] if r["status"] in ("FAIL", "DIFFERENT")]
        bad += [{"case": k, "config": "grompp", "status": v["status"]} for k, v in q.get("grompp", {}).items()
                if v["status"] not in ("IDENTICAL",)]
        if getattr(args, "strict", False):
            bad += [r for r in q["results"] if r["status"] == "EQUIVALENT" and r.get("deterministic")]
        for r in bad:
            print(f"QUALITY GATE: {r['case']} [{r['config']}] {r['status']}", file=sys.stderr)
        rc |= 1 if bad else 0
    p = sess.meta["summary"].get("perf")
    if p and getattr(args, "fail_on_slowdown", False):
        slow = [c for c in p.get("cases", []) if (c.get("e2e") or {}).get("verdict") == "slower"]
        for c in slow:
            print(f"PERF GATE: {c['case']} [{c['config']}] slower: {c['e2e']['speedup']:.4f}", file=sys.stderr)
        rc |= 1 if slow else 0
    return rc


def _list(suite: Suite) -> None:
    from .systems import unavailable_reason
    print(f"tier: {suite.tier}   target mode: {suite.target_mode}   suite files: {', '.join(suite.paths)}")
    print(f"target systems (always run, marked *): {', '.join(suite.target_systems) or '-'}")
    for name in suite.target_systems:
        why = unavailable_reason(suite, name)
        print(f"  {name}: {'available' if why is None else 'UNAVAILABLE - ' + why}")
    print("\nsystems:")
    for n, s in suite.systems.items():
        mark = "*" if n in suite.target_systems else " "
        print(f" {mark}{n:<18} {s.get('recipe'):<9} {s.get('description', '')}")
    print("\nconfigurations:")
    for n, c in suite.configs.items():
        env = " ".join(f"{k}={v}" for k, v in (c.get("env") or {}).items())
        print(f"  {n:<26} {'GPU ' if c.get('gpu') else 'CPU '} {c.get('args', '')} {env}")
    for sec in ("quality", "perf"):
        print(f"\n{sec} cases:")
        for c in suite.cases(sec):
            mark = "*" if suite.is_target(c) else " "
            print(f" {mark}{c['name']:<28} {', '.join(c['configs'])}")
        rt = suite.section(sec).get("regressiontests")
        if rt and suite.in_tier(rt):
            print(f"  + upstream regressiontests {rt.get('sets')} with {', '.join(rt['configs'])}")
    print("\nmicro:" + (" (skipped with --target-only)" if suite.target_mode == "only" else ""))
    for m in suite.section("micro").get("nbnxm", []):
        if suite.in_tier(m) and suite.target_mode != "only":
            print(f"  {m['name']:<28} size x{m['size']} ({3000 * m['size']} atoms), {m['threads']} threads")
    print("\nsweep systems:")
    for m in suite.section("sweep").get("systems", []):
        if suite.selected(m):
            print(f" {'*' if suite.is_target(m) else ' '}{m['name']}")


def _finish(sess) -> None:
    sess.close()
    try:
        from .report import write_report
        out = write_report([sess.dir], sess.dir / "report.html")
        log(f"report: {out}")
    except Exception as e:  # the raw results are what matters; never lose them over a plotting issue
        log(f"report generation failed: {e!r}")
    print(sess.dir)
