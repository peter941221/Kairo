#include <cuda/ptx>
#include <cuda_runtime.h>

__global__ void tcgen05_fence_probe() {
    if (threadIdx.x == 0) {
        cuda::ptx::tcgen05_fence_before_thread_sync();
    }
}

int main() {
    tcgen05_fence_probe<<<1, 32>>>();
    return static_cast<int>(cudaDeviceSynchronize());
}
