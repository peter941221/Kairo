#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>

static void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(2);
    }
}

__global__ void copy_kernel(const uint4* input, uint4* output, size_t elements) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = input[index];
}

int main(int argc, char** argv) {
    const size_t bytes = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : (512ull << 20);
    const int iterations = argc > 2 ? std::atoi(argv[2]) : 20;
    if (bytes < 4096 || iterations < 1 || bytes % sizeof(uint4) != 0) {
        std::fprintf(stderr, "bytes must be >=4096 and divisible by 16; iterations must be positive\n");
        return 2;
    }
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
    uint4* input = nullptr;
    uint4* output = nullptr;
    const size_t elements = bytes / sizeof(uint4);
    check(cudaMalloc(&input, bytes), "cudaMalloc(input)");
    check(cudaMalloc(&output, bytes), "cudaMalloc(output)");
    check(cudaMemset(input, 0x5a, bytes), "cudaMemset(input)");
    check(cudaMemset(output, 0, bytes), "cudaMemset(output)");
    constexpr int threads = 256;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    copy_kernel<<<blocks, threads>>>(input, output, elements);
    check(cudaGetLastError(), "warmup launch");
    check(cudaDeviceSynchronize(), "warmup synchronize");
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check(cudaEventCreate(&start), "cudaEventCreate(start)");
    check(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    check(cudaEventRecord(start), "cudaEventRecord(start)");
    for (int iteration = 0; iteration < iterations; ++iteration) {
        copy_kernel<<<blocks, threads>>>(input, output, elements);
    }
    check(cudaEventRecord(stop), "cudaEventRecord(stop)");
    check(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
    float elapsed_ms = 0.0f;
    check(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");
    const double average_ms = elapsed_ms / iterations;
    const double moved_bytes = static_cast<double>(bytes) * 2.0;
    std::printf(
        "{\"gpu\":\"%s\",\"bytes\":%zu,\"iterations\":%d,\"average_ms\":%.6f,\"read_write_gbps\":%.3f}\n",
        properties.name, bytes, iterations, average_ms,
        moved_bytes / (average_ms * 1.0e6));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(input);
    cudaFree(output);
    return 0;
}
