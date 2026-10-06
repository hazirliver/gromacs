"""GROMACS' own test suites (ctest) recorded in the unified format.

Builds the `tests` target of the given build, points CMake at the version-matched upstream
regressiontests (unless --no-regressiontests), runs ctest with JUnit output and stores one record
per test. These are tolerance-based correctness tests that complement gmxbench's bitwise A/B
quality suite.
"""

from __future__ import annotations

import os
import shlex
import xml.etree.ElementTree as ET

from . import builds
from .systems import regressiontests_dir
from .util import log, run


def run_upstream(sess, suite, build, args) -> dict:
    jobs = getattr(args, "jobs", 0) or None
    build = builds.ensure_built(build, jobs=jobs)
    sess.add_build("U", build)
    if not getattr(args, "no_regressiontests", False):
        rt = regressiontests_dir(build)
        log(f"upstream: configuring REGRESSIONTEST_PATH={rt}")
        run(["cmake", "-S", build.source_dir, "-B", build.build_dir, f"-DREGRESSIONTEST_PATH={rt}"],
            log_to=build.build_dir / "gmxbench-configure-rt.log", check=True)
    log("upstream: building unit tests (target 'tests'); this can take a while")
    build = builds.ensure_built(build, jobs=jobs, targets=("gmx", "tests"), force=True)
    junit = sess.runs / "upstream" / "ctest-junit.xml"
    junit.parent.mkdir(parents=True, exist_ok=True)
    par = max(1, min(8, (os.cpu_count() or 2) // 4))
    cmd = ["ctest", "--test-dir", build.build_dir, "--output-junit", junit, "-j", str(par), "--timeout", "3600",
           *shlex.split(getattr(args, "ctest_args", "") or "")]
    log(f"upstream: running ctest -j{par}")
    res = run(cmd, log_to=sess.runs / "upstream" / "ctest.out", timeout=6 * 3600)
    summary = {"returncode": res.returncode, "counts": {}, "failed": [], "wall": res.wall}
    if junit.exists():
        root = ET.parse(junit).getroot()
        for tc in root.iter("testcase"):
            name = tc.get("name")
            # CTest's JUnit output marks executed tests status="run" and adds <failure> on failure
            if tc.find("failure") is not None or tc.get("status") == "fail":
                status = "failed"
            elif tc.find("skipped") is not None or tc.get("status") in ("notrun", "disabled"):
                status = "skipped"
            else:
                status = "passed"
            t = float(tc.get("time") or 0)
            summary["counts"][status] = summary["counts"].get(status, 0) + 1
            if status == "failed":
                summary["failed"].append(name)
            sess.record("upstream", "ctest", "status", status, case=name, side="U", tags={"time_s": t})
            sess.record("upstream", "ctest", "time_s", t, case=name, side="U", unit="s", better="lower")
    log(f"upstream: {summary['counts']}; failed: {summary['failed'][:10]}")
    sess.set_summary("upstream", summary)
    return summary
