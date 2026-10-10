#!/usr/bin/env python3
"""Weighted minimax (error x max(1, z^3), i.e. relative to the 1/r^3 Coulomb term) rational fits of the Ewald
correction on [0, (beta rc)^2], evaluated in the kernel's fp32 monic r^2 form at MAS1's beta/rc, vs production."""
import sys
import numpy as np
from scipy.optimize import brentq
from scipy.special import erfc
sys.path.insert(0, ".")
from pmecorr_fit import PROD_N, PROD_D, g_exact, eval32, minimax_rational, fma32, F32
rc = 1.2
for rtol in (1e-5, 1e-6):
    brc = brentq(lambda x: erfc(x) - rtol, 0.5, 6); beta = brc / rc; T = brc * brc
    r = np.linspace(0.02, rc, 400000); r2 = (r * r).astype(F32)
    exact = beta**3 * g_exact(beta**2 * r.astype(np.float64)**2)
    t = (F32(beta * beta) * r2).astype(F32)
    prod = (eval32(PROD_N, PROD_D, t.astype(np.float64)).astype(F32) * F32(beta**3)).astype(np.float64)
    inv_r3 = 1.0 / r**3
    def metric(val):
        e = np.abs(val - exact)
        return np.max(e / np.minimum(inv_r3, beta**3)), np.sqrt(np.mean((e / inv_r3)**2)), np.max(e / inv_r3)
    pm = metric(prod)
    print(f"rtol {rtol:g} beta {beta:.4f}: production P6/Q4 (12 instr incl. 2 MOV + FMUL beta^2): max err/min(1/r^3,beta^3) {pm[0]:.2e} max err/(1/r^3) {pm[2]:.2e} rms {pm[1]:.2e}")
    for (m, n) in ((3, 4), (4, 3), (4, 4), (5, 3), (3, 5), (5, 4), (4, 5), (6, 3), (6, 4)):
        res = minimax_rational(m, n, T, weighted=True)
        if res is None:
            print(f"   ({m},{n}) LP failed"); continue
        num, den, _ = res
        a = [num[k] * beta**(2 * k) for k in range(m + 1)]; b = [den[k] * beta**(2 * k) for k in range(n + 1)]
        scale = beta**3 * a[-1] / b[-1]; an = [F32(x / a[-1]) for x in a]; bn = [F32(x / b[-1]) for x in b]
        def hm(c, x):
            acc = (x + c[-2]).astype(F32)
            for k in range(len(c) - 3, -1, -1):
                acc = fma32(acc, x, c[k])
            return acc
        var = ((hm(an, r2) * (F32(1) / hm(bn, r2)).astype(F32)).astype(F32) * F32(scale)).astype(np.float64)
        vm = metric(var)
        instr = m + n + 2  # (m-1)+(n-1) FFMA + 2 FADD + RCP + FMUL ; the scale FMUL replaces production's *beta3
        ok = "OK" if vm[0] <= pm[0] * 1.05 else "  "
        print(f"   {ok} ({m},{n}) {instr:2d} instr (vs 15): max err/min(1/r^3,beta^3) {vm[0]:.2e} max err/(1/r^3) {vm[2]:.2e} rms {vm[1]:.2e}")
