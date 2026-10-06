"""Statistics for A/B performance comparisons.

Performance metrics are ratio-scale and their noise is roughly multiplicative, so comparisons are
done on log values: the effect is the ratio of geometric means B/A with a Welch t confidence
interval. A Mann-Whitney U test and a bootstrap CI of the ratio of medians are reported as
distribution-free cross-checks. Many stage/kernel metrics are tested at once, so their p-values are
also adjusted with Benjamini-Hochberg (q-values).

`speedup` is always oriented so that > 1 means the candidate (B) is better.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats as st


def summarize(values) -> dict:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    n = len(v)
    if n == 0:
        return {"n": 0}
    out = {"n": n, "mean": float(v.mean()), "median": float(np.median(v)), "min": float(v.min()),
           "max": float(v.max())}
    if n > 1:
        sd = float(v.std(ddof=1))
        out.update({"std": sd, "cv": sd / out["mean"] if out["mean"] else float("nan")})
        h = st.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
        out["ci95"] = [out["mean"] - h, out["mean"] + h]
    return out


def _orient(ratio: float, better: str) -> float:
    return ratio if better == "higher" else 1.0 / ratio


def compare(a, b, better: str = "higher", alpha: float = 0.05, min_effect: float = 0.01,
            n_boot: int = 5000, seed: int = 0, resolution: float = 0.0) -> dict:
    """Compare samples a (baseline) and b (candidate).

    `resolution` is the measurement quantum in the units of the values (e.g. a stage time printed
    with 1 ms resolution). Its uniform-rounding variance (q^2/12) is added so that identical,
    quantised repeats cannot produce a spuriously zero standard error.
    """
    a = np.asarray([x for x in a if x is not None and np.isfinite(x) and x > 0], dtype=float)
    b = np.asarray([x for x in b if x is not None and np.isfinite(x) and x > 0], dtype=float)
    res = {"n_a": len(a), "n_b": len(b), "better": better, "alpha": alpha, "min_effect": min_effect,
           "a": summarize(a), "b": summarize(b)}
    if len(a) < 2 or len(b) < 2:
        if len(a) and len(b):
            res["speedup"] = _orient(float(np.median(b) / np.median(a)), better)
        res["verdict"] = "insufficient-data"
        return res
    la, lb = np.log(a), np.log(b)
    d = lb.mean() - la.mean()
    va, vb = la.var(ddof=1) / len(a), lb.var(ddof=1) / len(b)
    if resolution > 0:
        va += (resolution / a.mean()) ** 2 / 12 / len(a)
        vb += (resolution / b.mean()) ** 2 / 12 / len(b)
    se = math.sqrt(va + vb)
    if se > 0:
        df = (va + vb) ** 2 / ((va ** 2) / (len(a) - 1) + (vb ** 2) / (len(b) - 1))
        tcrit = st.t.ppf(1 - alpha / 2, df)
        p = float(2 * st.t.sf(abs(d) / se, df))
    else:
        df, tcrit, p = float(len(a) + len(b) - 2), 0.0, (0.0 if d != 0 else 1.0)
    lo, hi = math.exp(d - tcrit * se), math.exp(d + tcrit * se)
    ratio = math.exp(d)
    if better == "higher":
        sp, sp_lo, sp_hi = ratio, lo, hi
    else:
        sp, sp_lo, sp_hi = 1 / ratio, 1 / hi, 1 / lo
    try:
        p_mwu = float(st.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except ValueError:
        p_mwu = 1.0
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(a), size=(n_boot, len(a)))
    ib = rng.integers(0, len(b), size=(n_boot, len(b)))
    br = np.median(b[ib], axis=1) / np.median(a[ia], axis=1)
    if better != "higher":
        br = 1 / br
    # minimum detectable effect (80% power, two-sided alpha) given the observed noise
    t_b = st.t.ppf(0.8, df)
    mde = math.exp((tcrit + t_b) * se) - 1 if se > 0 else 0.0
    res.update({
        "speedup": sp, "ci": [sp_lo, sp_hi], "p": p, "p_mwu": p_mwu, "df": df,
        "boot_ci_median": [float(np.percentile(br, 2.5)), float(np.percentile(br, 97.5))],
        "speedup_median": _orient(float(np.median(b) / np.median(a)), better),
        "mde": mde,
    })
    res["verdict"] = verdict(sp, sp_lo, sp_hi, p, alpha, min_effect)
    return res


def verdict(sp, lo, hi, p, alpha, min_effect) -> str:
    significant = p < alpha and (lo > 1 or hi < 1)
    if not significant:
        return "no-change"
    if abs(sp - 1) < min_effect:
        return "negligible"
    return "faster" if sp > 1 else "slower"


def bh_qvalues(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values (q-values), same order as the input."""
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [0.0] * m
    prev = 1.0
    for rank, i in reversed(list(enumerate(order, start=1))):
        prev = min(prev, pvals[i] * m / rank)
        q[i] = prev
    return q


def apply_fdr(comparisons: list[dict], alpha: float = 0.05) -> None:
    """Add q-values and FDR-controlled verdicts to a family of comparisons (in place)."""
    idx = [i for i, c in enumerate(comparisons) if "p" in c]
    qs = bh_qvalues([comparisons[i]["p"] for i in idx])
    for i, q in zip(idx, qs):
        c = comparisons[i]
        c["q"] = q
        c["verdict_fdr"] = verdict(c["speedup"], c["ci"][0], c["ci"][1], q, alpha, c["min_effect"])
