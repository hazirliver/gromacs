// PCIe copy latency/bandwidth probe from the calling CPU (pin it with taskset).
//   nvcc -O2 -o pcie_probe pcie_probe.cu && taskset -c 0 ./pcie_probe [bytes]
// Prints medians over many repetitions of:
//   small (4 B) H2D and D2H cudaMemcpyAsync + cudaStreamSynchronize round trip (wall clock, us)
//   kernel launch + sync round trip of an empty kernel (wall clock, us)
//   `bytes` (default 185486*3*4 = one rvec array of MAS1) H2D and D2H from pinned memory:
//   event-timed device duration and wall-clock incl. sync, and the resulting GB/s
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e)); exit(1);} } while (0)

__global__ void empty() {}

static double med(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
static double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    size_t bytes = argc > 1 ? strtoull(argv[1], nullptr, 10) : size_t(185486) * 3 * 4;
    cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    void *h, *d; CK(cudaMallocHost(&h, bytes)); CK(cudaMalloc(&d, bytes));
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    for (int i = 0; i < 200; i++) { CK(cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, s)); }
    CK(cudaStreamSynchronize(s));
    std::vector<double> sh, sd, kl, bh, bd, wh, wd;
    for (int i = 0; i < 2000; i++) {
        double t = now_us(); CK(cudaMemcpyAsync(d, h, 4, cudaMemcpyHostToDevice, s)); CK(cudaStreamSynchronize(s)); sh.push_back(now_us() - t);
        t = now_us(); CK(cudaMemcpyAsync(h, d, 4, cudaMemcpyDeviceToHost, s)); CK(cudaStreamSynchronize(s)); sd.push_back(now_us() - t);
        t = now_us(); empty<<<1, 32, 0, s>>>(); CK(cudaStreamSynchronize(s)); kl.push_back(now_us() - t);
    }
    for (int i = 0; i < 300; i++) {
        float ms;
        double t = now_us();
        CK(cudaEventRecord(a, s)); CK(cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, s)); CK(cudaEventRecord(b, s));
        CK(cudaStreamSynchronize(s)); wh.push_back(now_us() - t);
        CK(cudaEventElapsedTime(&ms, a, b)); bh.push_back(ms * 1000);
        t = now_us();
        CK(cudaEventRecord(a, s)); CK(cudaMemcpyAsync(h, d, bytes, cudaMemcpyDeviceToHost, s)); CK(cudaEventRecord(b, s));
        CK(cudaStreamSynchronize(s)); wd.push_back(now_us() - t);
        CK(cudaEventElapsedTime(&ms, a, b)); bd.push_back(ms * 1000);
    }
    printf("bytes %zu small_h2d_rt_us %.2f small_d2h_rt_us %.2f empty_kernel_rt_us %.2f "
           "h2d_dev_us %.1f h2d_wall_us %.1f h2d_GBps %.2f d2h_dev_us %.1f d2h_wall_us %.1f d2h_GBps %.2f\n",
           bytes, med(sh), med(sd), med(kl), med(bh), med(wh), bytes / med(bh) / 1e3, med(bd), med(wd), bytes / med(bd) / 1e3);
    return 0;
}
