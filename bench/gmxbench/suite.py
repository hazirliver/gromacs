"""Loading and querying suite definition files (TOML).

Files are merged in this order (later ones override earlier ones):
  bench/suites/default.toml   generic coverage (committed)
  bench/suites/target.toml    the systems this fork is optimised for (committed)
  bench/suites/local.toml     personal overrides (git-ignored, optional)
  $GMXBENCH_SUITE             colon-separated list of extra files
  --suite FILE ...            command-line overlays
Tables merge recursively; arrays of named tables (e.g. [[quality.cases]]) merge by `name` (same name
replaces, new names are appended); any other value is replaced.

Target systems ([target] systems = [...]): their cases run on every invocation - in every tier listed
by the case and regardless of --cases - unless --no-target is given; --target-only runs nothing else.
"""

from __future__ import annotations

import fnmatch
import os
import tomllib
from pathlib import Path

from .util import BENCH_ROOT, tier_value

SUITES = BENCH_ROOT / "suites"
DEFAULT_SUITE = SUITES / "default.toml"
TARGET_SUITE = SUITES / "target.toml"
LOCAL_SUITE = SUITES / "local.toml"


def _merge(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        for k, v in b.items():
            out[k] = _merge(out[k], v) if k in out else v
        return out
    if (isinstance(a, list) and isinstance(b, list) and a and b
            and all(isinstance(x, dict) and "name" in x for x in a + b)):
        out = list(a)
        index = {x["name"]: i for i, x in enumerate(out)}
        for x in b:
            if x["name"] in index:
                out[index[x["name"]]] = x
            else:
                index[x["name"]] = len(out)
                out.append(x)
        return out
    return b


def default_paths() -> list[Path]:
    paths = [DEFAULT_SUITE]
    for p in (TARGET_SUITE, LOCAL_SUITE):
        if p.exists():
            paths.append(p)
    for p in os.environ.get("GMXBENCH_SUITE", "").split(":"):
        if p.strip():
            paths.append(Path(p.strip()))
    return paths


class Suite:
    def __init__(self, extra: list[Path] | None = None, tier: str = "quick", target_mode: str = "include"):
        paths = default_paths() + [Path(p) for p in (extra or [])]
        data: dict = {}
        for p in paths:
            with open(p, "rb") as f:
                data = _merge(data, tomllib.load(f))
        self.paths = [str(p) for p in paths]
        self.data = data
        self.tier = tier
        if target_mode not in ("include", "only", "exclude"):
            raise ValueError(target_mode)
        self.target_mode = target_mode

    # -- generic accessors -------------------------------------------------------------------
    @property
    def profiles(self) -> dict:
        return self.data.get("profiles", {})

    @property
    def systems(self) -> dict:
        return self.data.get("systems", {})

    @property
    def mdp_sets(self) -> dict:
        return self.data.get("mdp", {})

    @property
    def configs(self) -> dict:
        return self.data.get("configs", {})

    @property
    def target_systems(self) -> list[str]:
        return list(self.data.get("target", {}).get("systems", []))

    def section(self, name: str) -> dict:
        return self.data.get(name, {})

    def tv(self, value):
        return tier_value(value, self.tier)

    def in_tier(self, item: dict) -> bool:
        return self.tier in item.get("tiers", ["quick", "full"])

    def is_target(self, item: dict) -> bool:
        return item.get("system") in self.target_systems or bool(item.get("target"))

    def selected(self, item: dict, select: list[str] | None = None) -> bool:
        """Tier, target mode and --cases/--systems globs for one case-like entry."""
        tgt = self.is_target(item)
        if self.target_mode == "only" and not tgt:
            return False
        if self.target_mode == "exclude" and tgt:
            return False
        if not self.in_tier(item):
            return False
        if select and not any(fnmatch.fnmatch(item["name"], pat) for pat in select):
            # target cases are always run, unless the user asked for targets only (then globs narrow them)
            return tgt and self.target_mode == "include"
        return True

    def cases(self, section: str, select: list[str] | None = None) -> list[dict]:
        return [c for c in self.section(section).get("cases", []) if self.selected(c, select)]

    def mdp_params(self, names: list[str], system_mdp: dict | None = None) -> dict:
        """Merge named mdp sets in order. '@system' inserts the system's own base mdp."""
        params: dict = {}
        for n in names:
            if n == "@system":
                params.update(system_mdp or {})
            else:
                if n not in self.mdp_sets:
                    raise SystemExit(f"unknown mdp set '{n}'")
                params.update({k: self.tv(v) for k, v in self.mdp_sets[n].items()})
        return params

    def snapshot(self) -> dict:
        return {"paths": self.paths, "tier": self.tier, "target_mode": self.target_mode,
                "target_systems": self.target_systems, "data": self.data}
