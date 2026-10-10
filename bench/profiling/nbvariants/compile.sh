#!/usr/bin/env bash
# compile.sh VARIANT_ROOT OUT_PREFIX : compile the NB F (noprune) kernel TU of a patched copy of src/gromacs/nbnxm
# with the production nvcc flags; writes OUT_PREFIX.cubin and OUT_PREFIX.sass (ElecEw_VdwLJFsw_F kernel only).
# Static "what-if" SASS for kernel ideas without building or running GROMACS. Environment:
#   NBVAR_SRC   gmxbench source worktree of the baseline (e.g. ~/.cache/gmxbench/src/<sha12>)
#   NBVAR_BUILD a configured gmxbench build of that source (cuda-nosub[-lineinfo]); its includes and flags are used
# Check first that an unmodified copy (mkvariant.py v0 0) gives SASS identical to the library's kernel.
set -eu
o=$(realpath -m "$2")
v=$(realpath "$1")
SRC=$(realpath "${NBVAR_SRC:?}")/src; B=$(realpath "${NBVAR_BUILD:?}")
inc=$(sed "s#-I$SRC #-I$v/src -I$SRC #" $B/src/gromacs/CMakeFiles/libgromacs.dir/includes_CUDA.rsp)
cd $B/src/gromacs
eval /usr/local/cuda-13.0/bin/nvcc -forward-unknown-to-host-compiler -ccbin=/usr/bin/c++ -DGMX_DOUBLE=0 -DHAVE_CONFIG_H -DTMPI_EXPORTS -DTMPI_USE_VISIBILITY -DUSE_STD_INTTYPES_H -Dlibgromacs_EXPORTS $inc \
  -lineinfo -D_FORCE_INLINES -O3 -DNDEBUG -std=c++17 -arch=sm_89 -cubin -use_fast_math -static-global-template-stub=false -Xptxas=-warn-double-usage -diag-suppress=177 ${EXTRA_NVCC:-} \
  -x cu $v/src/gromacs/nbnxm/cuda/nbnxm_cuda_kernel_F_noprune.cu -o $o.cubin
cuobjdump -sass $o.cubin | awk '/Function : .*nbnxn_kernel_ElecEw_VdwLJFsw_F_cuda/{on=1} /Function :/{if(!/nbnxn_kernel_ElecEw_VdwLJFsw_F_cuda/)on=0} on' | grep -E "^\s+/\*[0-9a-f]{4}\*/" | sed 's#/\*[0-9a-f]*\*/##; s#;.*##; s/^ *//' > $o.sass
echo "$o: $(wc -l < $o.sass) SASS instructions; regs: $(cuobjdump -res-usage $o.cubin 2>/dev/null | grep -A1 'ElecEw_VdwLJFsw_F_cuda' | grep -o 'REG:[0-9]*' | head -1)"
