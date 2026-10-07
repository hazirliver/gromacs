#!/usr/bin/env bash
# Build libsassprof.so (CUPTI injection library) into OUTDIR (default: this directory/build).
set -eu
here=$(cd "$(dirname "$0")" && pwd)
out=${1:-$here/build}
cuda=${CUDA_HOME:-/usr/local/cuda}
mkdir -p "$out"
g++ -O2 -std=c++17 -fPIC -shared -o "$out/libsassprof.so" "$here/sassprof.cpp" \
    -I"$cuda/include" -I"$cuda/extras/CUPTI/include" -L"$cuda/lib64" -lcupti -lcuda \
    -Wl,-rpath,"$cuda/lib64"
echo "$out/libsassprof.so"
