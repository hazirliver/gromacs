"""Building GROMACS from a git ref or the working tree into isolated build directories.

A build spec is one of
  WORKTREE            the current checkout including uncommitted changes
  <git ref>           branch, tag or commit; checked out into a detached git worktree
  path:/some/dir      an existing build (or install) directory containing bin/gmx
Builds are keyed by source identity + build profile, so repeated invocations reuse them.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .util import REPO_ROOT, log, run, sha256_bytes, stable_hash, workdir, write_json, read_json


@dataclass
class Build:
    spec: str
    label: str
    profile: str
    source_dir: Path
    build_dir: Path
    sha: str | None
    dirty_hash: str | None
    cmake_args: list = field(default_factory=list)
    env: dict = field(default_factory=dict)  # runtime env applied to every gmx call of this side
    real_source: Path | None = None           # the actual tree behind a fixed-length source path

    @property
    def gmx(self) -> Path:
        return self.build_dir / "bin" / "gmx"

    @property
    def id(self) -> str:
        """Identity of the binary (not of the runtime env)."""
        src = self.sha or "nogit"
        if self.dirty_hash:
            src += f"+{self.dirty_hash}"
        return f"{src[:20]}-{self.profile}"

    @property
    def run_id(self) -> str:
        """Identity of binary + runtime env; used to cache run outputs."""
        return self.id + (f"-env{stable_hash(self.env, 8)}" if self.env else "")

    def info(self) -> dict:
        p = self.build_dir / "gmxbench-build.json"
        meta = read_json(p) if p.exists() else {}
        return {
            "spec": self.spec, "label": self.label, "profile": self.profile, "sha": self.sha,
            "dirty_hash": self.dirty_hash, "id": self.id, "build_dir": str(self.build_dir),
            "runtime_env": self.env, **meta,
        }


def _git(*args, cwd=REPO_ROOT) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _worktree_dirty_hash(root: Path = REPO_ROOT) -> str | None:
    """Hash of uncommitted changes to files that affect the GROMACS build (bench/ and data/ excluded)."""
    excl = [":(exclude)bench", ":(exclude)data"]
    diff = subprocess.run(["git", "diff", "HEAD", "--binary", "--", ".", *excl], cwd=root,
                          capture_output=True).stdout
    untracked = _git("ls-files", "--others", "--exclude-standard", "--", "src", "cmake", "CMakeLists.txt",
                     "share", "api", "tests", "python_packaging", cwd=root)
    blob = diff
    for f in sorted(untracked.splitlines()):
        p = root / f
        if p.is_file():
            blob += f.encode() + p.read_bytes()
    return sha256_bytes(blob)[:12] if blob else None


def cuda_arch() -> str | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout.split()
        caps = sorted({c.replace(".", "") for c in out if re.fullmatch(r"\d+\.\d+", c)})
        return ";".join(caps) if caps else None
    except (OSError, subprocess.SubprocessError):
        return None


def resolve(spec: str, profile: str, profiles: dict, label: str | None = None, env: dict | None = None) -> Build:
    if profile not in profiles and not spec.startswith("path:"):
        raise SystemExit(f"unknown build profile '{profile}' (have: {', '.join(profiles)})")
    prof = profiles.get(profile, {})
    cmake_args = list(prof.get("cmake", []))
    if prof.get("cuda_arch") == "auto":
        arch = cuda_arch()
        if arch:
            cmake_args.append(f"-DCMAKE_CUDA_ARCHITECTURES={arch}")
    root = workdir()
    if spec.startswith("path:"):
        bdir = Path(spec[5:]).resolve()
        return Build(spec, label or bdir.name, "external", bdir, bdir, None, None, [], env or {}, bdir)
    if spec == "WORKTREE":
        sha = _git("rev-parse", "HEAD")
        dirty = _worktree_dirty_hash()
        real_src, src_key = REPO_ROOT, "WORKTREE"
    elif spec.startswith("src:"):
        real_src = Path(spec[4:]).resolve()
        sha = _git("rev-parse", "HEAD", cwd=real_src)
        dirty = _worktree_dirty_hash(real_src)
        src_key = f"src:{real_src}"
    else:
        sha = _git("rev-parse", "--verify", f"{spec}^{{commit}}")
        dirty = None
        real_src, src_key = None, sha
    # Identical sources must give identical machine code. Source and build paths end up in the binary
    # (__FILE__ in assertions, data paths), and their *length* changes instruction encodings and hence the
    # alignment of everything after them - enough to move a hot kernel by several percent between two
    # builds of the same commit. So every build sees a source path and a build path of fixed length:
    #   <workdir>/src/<12 chars>     (a git worktree, or a symlink to the working tree / src: dir)
    #   <workdir>/builds/<16 hex>
    if real_src is None:
        src = root / "src" / sha[:12]
    else:
        src = root / "src" / (("w" if spec == "WORKTREE" else "s") + stable_hash(str(real_src), 11))
    b = Build(spec, label or spec, profile, src, Path(), sha, dirty, cmake_args, env or {})
    b.real_source = real_src or src
    b.build_dir = root / "builds" / stable_hash({"src": src_key, "profile": profile, "cmake": cmake_args}, 16)
    return b


def _ensure_source(b: Build) -> None:
    if b.spec == "WORKTREE" or b.spec.startswith("src:"):
        b.source_dir.parent.mkdir(parents=True, exist_ok=True)
        if not b.source_dir.is_symlink():
            b.source_dir.symlink_to(b.real_source, target_is_directory=True)
        return
    if b.source_dir.exists():
        return
    b.source_dir.parent.mkdir(parents=True, exist_ok=True)
    log(f"creating git worktree for {b.spec} ({b.sha[:12]}) at {b.source_dir}")
    _git("worktree", "add", "--detach", str(b.source_dir), b.sha)


def ensure_built(b: Build, jobs: int | None = None, targets=("gmx",), force: bool = False) -> Build:
    """Configure and build if needed. Rebuilds a WORKTREE build when its sources changed."""
    if b.profile == "external":
        if not b.gmx.exists():
            raise SystemExit(f"no bin/gmx in {b.build_dir}")
        return b
    stamp = b.build_dir / "gmxbench-build.json"
    if stamp.exists() and b.gmx.exists() and not force:
        st = read_json(stamp)
        if (st.get("sha"), st.get("dirty_hash")) == (b.sha, b.dirty_hash) and set(targets) <= set(st.get("targets", [])):
            log(f"build {b.label} up to date: {b.build_dir}")
            return b
    _ensure_source(b)
    b.build_dir.mkdir(parents=True, exist_ok=True)
    jobs = jobs or max(1, (os.cpu_count() or 2) - 2)
    if not (b.build_dir / "CMakeCache.txt").exists():
        log(f"configuring {b.label} [{b.profile}] in {b.build_dir}")
        run(["cmake", "-S", b.source_dir, "-B", b.build_dir, *b.cmake_args],
            log_to=b.build_dir / "gmxbench-configure.log", check=True)
    for t in targets:
        log(f"building target '{t}' for {b.label} (-j{jobs}); log: {b.build_dir}/gmxbench-build-{t}.log")
        run(["cmake", "--build", b.build_dir, "-j", str(jobs), "--target", t],
            log_to=b.build_dir / f"gmxbench-build-{t}.log", check=True)
    prev = []
    if stamp.exists():
        st = read_json(stamp)
        if (st.get("sha"), st.get("dirty_hash")) == (b.sha, b.dirty_hash):
            prev = st.get("targets", [])
    write_json(stamp, {"sha": b.sha, "dirty_hash": b.dirty_hash, "cmake_args": b.cmake_args,
                       "targets": sorted(set(prev) | set(targets)), "version": gmx_version(b)})
    return b


def gmx_version(b: Build) -> dict:
    """Parse `gmx -version` into a dict (precision, SIMD, GPU, FFT, compilers...)."""
    res = run([b.gmx, "-quiet", "-version"], env={**os.environ, "GMX_MAXBACKUP": "-1"})
    info = {}
    for line in (res.stdout + res.stderr).splitlines():
        m = re.match(r"^([A-Za-z][A-Za-z0-9 /()+-]*?):\s+(.*)$", line)
        if m:
            info[m.group(1).strip()] = m.group(2).strip()
    return info
