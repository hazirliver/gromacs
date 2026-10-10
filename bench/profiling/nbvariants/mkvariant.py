#!/usr/bin/env python3
"""Make a patched copy of src/gromacs/nbnxm for static SASS counting (NOT for running: v3 reinterprets the
LJ table, the prune kernels are not patched).

    NBVAR_SRC=<gmxbench src worktree> mkvariant.py OUTDIR FLAGS   (FLAGS: any of 1 2 3, e.g. "123"; 0 = unmodified)
  1  i-cluster mask test on imask/wexcl shifted once per jm, bit test with a compile-time bit
  2  Ewald correction: monic (4,4) rational in r^2 with runtime coefficients (constant bank), beta folded
  3  force switch: per-type-pair coefficients K2,K3 loaded with c6,c12 as one float4 (LDG.128)
"""
import shutil
import sys
from pathlib import Path

import os
SRC = Path(os.environ["NBVAR_SRC"]).expanduser() / "src/gromacs/nbnxm"
out, flags = Path(sys.argv[1]), sys.argv[2]
dst = out / "src/gromacs/nbnxm"
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(SRC, dst)
k = dst / "cuda/nbnxm_cuda_kernel.cuh"
s = k.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old, s.count(old))
    s = s.replace(old, new)


if "1" in flags:
    rep("""                if (imask & (superClInteractionMask << (jm * c_superClusterSize)))
                {
                    mask_ji = (1U << (jm * c_superClusterSize));""",
        """                const unsigned int imaskJ = imask >> (jm * c_superClusterSize);
                const unsigned int wexclJ = wexcl >> (jm * c_superClusterSize);
                if (imaskJ & superClInteractionMask)
                {
                    mask_ji = (1U << (jm * c_superClusterSize));""")
    rep("""                        if (imask & mask_ji)
                        {""", """                        if (imaskJ & (1U << i))
                        {""")
    rep("int_bit = (wexcl & mask_ji) ? 1.0F : 0.0F;", "int_bit = (wexclJ & (1U << i)) ? 1.0F : 0.0F;")

if "2" in flags:
    rep("pmeCorrF(beta2 * r2) * beta3", "pmeCorrFR2(r2) * c_expPmeScale")
    s = s.replace("#include \"gromacs/pbcutil/ishift.h\"", """#include "gromacs/pbcutil/ishift.h"
#ifndef EXP_PMECORR_R2
#define EXP_PMECORR_R2
__constant__ float c_expPmeN[4];
__constant__ float c_expPmeD[4];
__constant__ float c_expPmeScale;
static __forceinline__ __device__ float pmeCorrFR2(const float r2)
{
    float n = r2 + c_expPmeN[3];
    n       = n * r2 + c_expPmeN[2];
    n       = n * r2 + c_expPmeN[1];
    n       = n * r2 + c_expPmeN[0];
    float d = r2 + c_expPmeD[3];
    d       = d * r2 + c_expPmeD[2];
    d       = d * r2 + c_expPmeD[1];
    d       = d * r2 + c_expPmeD[0];
    return n * (1.0F / d);
}
#endif""", 1)
    assert "pmeCorrFR2(const float r2)" in s

if "3" in flags:
    rep("""                                fetch_nbfp_c6_c12(c6, c12, nbparam, ntypes * typei + typej);""",
        """                                const float4 ljp = LDG(&reinterpret_cast<const float4*>(nbparam.nbfp)[ntypes * typei + typej]);
                                c6 = ljp.x;
                                c12 = ljp.y;
                                const float swK2 = ljp.z;
                                const float swK3 = ljp.w;""")
    rep("""                                calculate_force_switch_F(nbparam, c6, c12, inv_r, r2, &F_invr);""",
        """                                {
                                    const float rr = r2 * inv_r;
                                    const float sw = fmaxf(rr - nbparam.rvdw_switch, 0.0F);
                                    F_invr += (swK3 * sw + swK2) * (sw * sw * inv_r);
                                }""")
k.write_text(s)
print(f"variant {flags} -> {dst}")
