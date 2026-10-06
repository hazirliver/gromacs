#!/usr/bin/env bash
# Record versions and machine state for a profiling session.
#   bench/profiling/env_snapshot.sh OUTDIR [BUILD_DIR ...]
# Writes plain-text dumps into OUTDIR (created). Read-only: changes nothing on the machine.
set -u
out=${1:?usage: env_snapshot.sh OUTDIR [BUILD_DIR ...]}
shift
mkdir -p "$out"

snap() {  # snap NAME CMD...
    local name=$1; shift
    { echo "\$ $*"; "$@" 2>&1; echo "# rc=$?"; } > "$out/$name.txt"
}

date -Is > "$out/timestamp.txt"
snap uname uname -a
snap os-release cat /etc/os-release
snap cmdline cat /proc/cmdline
snap lscpu lscpu
snap lscpu-e lscpu -e
snap numactl numactl -H
snap meminfo cat /proc/meminfo
snap virt systemd-detect-virt
snap nvidia-smi nvidia-smi
snap nvidia-smi-q nvidia-smi -q
snap nvidia-smi-topo nvidia-smi topo -m
snap nvidia-params cat /proc/driver/nvidia/params
snap nvidia-version cat /proc/driver/nvidia/version
snap nvcc nvcc --version
snap nsys nsys --version
snap ncu ncu --version
snap perf perf --version
snap gcc gcc --version
snap cmake cmake --version
snap lspci-gpu lspci -vv -s "$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | head -1 | sed 's/^0000//; s/^0000://')"
{
    for f in /proc/sys/kernel/perf_event_paranoid /proc/sys/kernel/kptr_restrict /proc/sys/kernel/nmi_watchdog \
             /proc/sys/kernel/numa_balancing /proc/sys/kernel/sched_autogroup_enabled \
             /sys/kernel/mm/transparent_hugepage/enabled /sys/kernel/mm/transparent_hugepage/defrag \
             /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor /sys/devices/system/cpu/intel_pstate/status \
             /sys/devices/system/cpu/smt/control; do
        printf '%s = %s\n' "$f" "$(cat "$f" 2>/dev/null || echo n/a)"
    done
    printf 'sudo -n = %s\n' "$(sudo -n true 2>/dev/null && echo yes || echo no)"
} > "$out/sysctl.txt"
snap loadavg cat /proc/loadavg
snap top top -b -n 1 -w 200
for b in "$@"; do
    tag=$(basename "$b")
    grep -E '^(CMAKE_BUILD_TYPE|CMAKE_C_COMPILER|CMAKE_CXX_COMPILER|CMAKE_CUDA_COMPILER|CMAKE_CUDA_ARCHITECTURES|CMAKE_C_FLAGS|CMAKE_CXX_FLAGS|CMAKE_CUDA_FLAGS|GMX_[A-Z_]*):' \
        "$b/CMakeCache.txt" > "$out/cmake-$tag.txt" 2>&1
    "$b/bin/gmx" -quiet -version > "$out/gmx-version-$tag.txt" 2>&1
    { cuobjdump --list-elf "$b/lib/libgromacs.so"; cuobjdump --list-ptx "$b/lib/libgromacs.so"; } \
        > "$out/cuobjdump-$tag.txt" 2>&1
done
echo "$out"
