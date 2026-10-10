#!/usr/bin/env python3
"""How short can the GPU Ewald correction pmeCorrF(z2) be at the production accuracy?

pmeCorrF(t) approximates g(t) = -(erf(z)/z^3 - 2 exp(-z^2)/(sqrt(pi) z^2)), t = z^2 = (beta r)^2, as
P6(t)/Q4(t) (nbnxm_kernel_utils.h). The kernel only evaluates it for r < rcoulomb, i.e. t < (beta rc)^2,
and beta rc is fixed by ewald-rtol alone (erfc(beta rc) = rtol). This script

  1. measures the production approximation in emulated fp32 (FMA = one rounding) over [0, T],
  2. finds discrete minimax rationals P_m/Q_n on [0, T] by bisection + linear programming (differential
     correction), rounds the coefficients to fp32, evaluates them in emulated fp32 Horner form,
  3. prints the error (absolute, and relative to the 1/r^3 Coulomb term = z^3 |dg|) per (m, n) and the
     instruction count of each form.

Usage: pmecorrf_fit.py [--out FILE]
"""
import argparse
import itertools
import math
import sys

import numpy as np
from scipy.optimize import brentq, linprog
from scipy.special import erf, erfc

F32 = np.float32
SQRTPI = math.sqrt(math.pi)

PROD_N = [-0.75225204789749321333, 0.069670166153766424023, -0.019278317264888380590,
          0.0010054721316683106153, -0.000053401640219807709149, 1.4703624142580877519e-6,
          -1.7357322914161492954e-8]  # FN0..FN6
PROD_D = [1.0, 0.50736591960530292870, 0.11583842382862377919, 0.014866955030185295499,
          0.0011193462567257629232]  # FD0..FD4


def g_exact(t):
    t = np.asarray(t, dtype=np.float64)
    z = np.sqrt(t)
    out = np.empty_like(t)
    small = t < 1e-2
    # series: g = -(2/sqrt(pi)) * sum_k (-1)^k t^k (1/(2k+1) - 1/k!) ... computed directly below
    ts = t[small]
    # erf(z)/z^3 - 2 e^{-t}/(sqrt(pi) t) = (2/sqrt(pi)) sum_{k>=1} (-1)^k t^(k-1) [1/(k!(2k+1)) - 1/k!]
    acc = np.zeros_like(ts)
    for k in range(1, 12):
        acc += (-1) ** k * ts ** (k - 1) * (1.0 / (math.factorial(k) * (2 * k + 1)) - 1.0 / math.factorial(k))
    out[small] = -(2.0 / SQRTPI) * acc
    zl, tl = z[~small], t[~small]
    out[~small] = -(erf(zl) / zl ** 3 - 2.0 * np.exp(-tl) / (SQRTPI * tl))
    return out


def fma32(a, b, c):
    """fp32 fused multiply-add: exact product (fits in fp64), one rounding (fp64 add then fp32 round:
    double rounding is negligible here)."""
    return F32(np.float64(a) * np.float64(b) + np.float64(c))


def horner32(coef, t32):
    """Evaluate sum coef[k] t^k in fp32 Horner form with FMAs, as the GPU code does."""
    c = [F32(x) for x in coef]
    acc = np.full_like(t32, c[-1])
    for k in range(len(c) - 2, -1, -1):
        acc = fma32(acc, t32, c[k])
    return acc


def eval32(num, den, t):
    t32 = t.astype(F32)
    p = horner32(num, t32)
    if len(den) <= 1:
        return p.astype(np.float64) / (den[0] if den else 1.0)
    q = horner32(den, t32)
    rcp = (F32(1.0) / q).astype(F32)  # MUFU.RCP is ~1 ulp; correctly rounded here
    return (p * rcp).astype(F32).astype(np.float64)


def minimax_rational(m, n, T, npts=3000, weighted=False):
    """Discrete minimax P_m/Q_n (Q(0)=1, Q>0) on [0,T] by bisection on delta with an LP feasibility
    problem (Cheney-Loeb differential correction, linear form). Fit in s = t/T for conditioning."""
    # Chebyshev-distributed points (denser at both ends)
    s = 0.5 * (1 - np.cos(np.linspace(0, np.pi, npts)))
    t = s * T
    gt = g_exact(t)
    w = np.maximum(1.0, t ** 1.5) if weighted else np.ones_like(t)  # error relative to 1/r^3 (in beta^3 units)
    V = np.vander(s, max(m, n) + 1, increasing=True)
    P = V[:, : m + 1]
    Qv = V[:, 1 : n + 1]  # q1..qn (q0 = 1)

    def feasible(delta):
        # variables: p0..pm, q1..qn ; constraints:
        #   P(s) - g Q(s) <= delta Q(s)   ->  P - (g+delta) Qv q <= g + delta
        #  -P(s) + g Q(s) <= delta Q(s)   -> -P + (g-delta) Qv q <= -(g - delta)
        #   Q(s) >= 1e-3                  -> -Qv q <= 1 - 1e-3
        dw = delta / w
        A1 = np.hstack([P, -(gt + dw)[:, None] * Qv])
        b1 = gt + dw
        A2 = np.hstack([-P, (gt - dw)[:, None] * Qv])
        b2 = -(gt - dw)
        A3 = np.hstack([np.zeros((len(s), m + 1)), -Qv])
        b3 = np.full(len(s), 1 - 1e-3)
        A = np.vstack([A1, A2, A3]) if n else np.vstack([A1, A2])
        b = np.concatenate([b1, b2, b3]) if n else np.concatenate([b1, b2])
        res = linprog(np.zeros(m + 1 + n), A_ub=A, b_ub=b, bounds=[(None, None)] * (m + 1 + n), method="highs")
        return res.x if res.status == 0 else None

    lo, hi = 0.0, 1.0
    best = feasible(hi)
    if best is None:
        return None
    for _ in range(60):
        mid = math.sqrt(lo * hi) if lo > 0 else hi / 1e6 if hi > 1e-14 else hi / 2
        x = feasible(mid)
        if x is not None:
            hi, best = mid, x
        else:
            lo = mid
        if lo > 0 and hi / lo < 1.02:
            break
    p = best[: m + 1]
    q = np.concatenate([[1.0], best[m + 1 :]])
    # back to the t basis: coefficient k scales by T^-k
    num = [p[k] / T ** k for k in range(m + 1)]
    den = [q[k] / T ** k for k in range(n + 1)]
    return num, den, hi


def instr_count(m, n):
    """SASS instructions of the Horner form: m + n FFMA, + MUFU.RCP + FMUL for a rational. Constant
    re-materialisations (MOV) for the leading coefficients are not counted (0-2, the same for all)."""
    return m + n + (2 if n else 0)


def report(num, den, T, tag, grid):
    t = grid[grid <= T]
    ex = g_exact(t)
    ap = eval32(num, den, t)
    err = np.abs(ap - ex)
    z3 = t ** 1.5
    rel13 = err * z3  # error relative to the 1/r^3 Coulomb term (beta^3 dg vs 1/r^3)
    # relative to the full real-space force factor psi(z) = erfc(z) + 2/sqrt(pi) z e^{-z^2} (times 1/r^3)
    z = np.sqrt(t)
    psi = erfc(z) + 2 / SQRTPI * z * np.exp(-t)
    relf = rel13 / psi
    return (f"{tag:<34} max|dg| {err.max():.2e}  rms {np.sqrt((err**2).mean()):.2e}  "
            f"max z^3|dg| {rel13.max():.2e}  max dF/F {relf.max():.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    args = ap.parse_args()
    out = open(args.out, "w") if args.out else sys.stdout

    def pr(*a):
        print(*a, file=out, flush=True)

    rtols = [1e-5, 1e-6, 1e-7]
    Ts = {}
    for rt in rtols:
        x = brentq(lambda x: erfc(x) - rt, 0.5, 6)
        Ts[rt] = x * x
    grid = np.concatenate([np.linspace(1e-6, 0.01, 2000), np.linspace(0.01, 20, 200000)])
    pr("beta*rc per ewald-rtol (erfc(beta rc) = rtol): " +
       ", ".join(f"rtol {rt:g}: beta rc {math.sqrt(T):.4f}, T=(beta rc)^2 {T:.3f}" for rt, T in Ts.items()))
    pr("")
    pr("== production pmeCorrF (P6/Q4 Horner, 12 instructions) in emulated fp32 ==")
    for rt, T in Ts.items():
        pr(report(PROD_N, PROD_D, T, f"production on [0, {T:.2f}] (rtol {rt:g})", grid))
    pr(report(PROD_N, PROD_D, 16.0, "production on [0, 16]", grid))
    pr("")
    prod_err = {rt: np.abs(eval32(PROD_N, PROD_D, grid[grid <= T]) - g_exact(grid[grid <= T])).max()
                for rt, T in Ts.items()}

    pr("== minimax P_m/Q_n on [0, T], fp32 coefficients, emulated fp32 Horner ==")
    pr("(target: max|dg| <= production's on the same interval; instr = FFMA + RCP + FMUL)")
    results = []
    for rt in rtols[:2]:
        T = Ts[rt]
        pr(f"-- interval [0, {T:.3f}] (rtol {rt:g}), production max|dg| {prod_err[rt]:.2e}")
        for m, n in sorted(itertools.product(range(2, 8), range(0, 6)), key=lambda x: (instr_count(*x), x)):
            if instr_count(m, n) > 12 or (n == 0 and m < 5):
                continue
            r = minimax_rational(m, n, T)
            if r is None:
                pr(f"   ({m},{n}) LP failed")
                continue
            num, den, delta = r
            t = grid[grid <= T]
            err = np.abs(eval32(num, den, t) - g_exact(t)).max()
            ok = "OK " if err <= prod_err[rt] * 1.0001 else "   "
            line = report(num, den, T, f"   {ok}({m},{n}) instr {instr_count(m, n):2d} lp {delta:.1e}", grid)
            pr(line)
            results.append((rt, m, n, instr_count(m, n), err, num, den))
    pr("")
    pr("== cheapest forms that meet the production error ==")
    for rt in rtols[:2]:
        cand = [r for r in results if r[0] == rt and r[4] <= prod_err[rt] * 1.0001]
        cand.sort(key=lambda r: (r[3], r[4]))
        for r in cand[:3]:
            pr(f"rtol {rt:g}: ({r[1]},{r[2]}) {r[3]} instr, max|dg| {r[4]:.2e}")
            pr("   num " + ", ".join(f"{float(F32(c)):.9e}" for c in r[5]))
            pr("   den " + ", ".join(f"{float(F32(c)):.9e}" for c in r[6]))
        if cand:
            # robustness: evaluate the rtol 1e-5 fit beyond its interval
            pass


if __name__ == "__main__":
    main()
