#include <cuda_pipeline.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#ifndef KAIRO_TARGET_ARCH
#define KAIRO_TARGET_ARCH 0
#endif

__global__ void async_copy_checksum(const int* input, int* output) {
    __shared__ int tile[256];
    const int index = static_cast<int>(threadIdx.x);
    __pipeline_memcpy_async(tile + index, input + index, sizeof(int));
    __pipeline_commit();
    __pipeline_wait_prior(0);
    output[index] = tile[index] + 1;
}

static void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(2);
    }
}

int main() {
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
    int* input = nullptr;
    int* output = nullptr;
    check(cudaMallocManaged(&input, 256 * sizeof(int)), "cudaMallocManaged(input)");
    check(cudaMallocManaged(&output, 256 * sizeof(int)), "cudaMallocManaged(output)");
    for (int index = 0; index < 256; ++index) input[index] = index;
    async_copy_checksum<<<1, 256>>>(input, output);
    check(cudaGetLastError(), "async_copy_checksum launch");
    check(cudaDeviceSynchronize(), "async_copy_checksum synchronize");
    int checksum = 0;
    for (int index = 0; index < 256; ++index) checksum += output[index];
    const int expected = (256 * 257) / 2;
    std::printf("{\"gpu\":\"%s\",\"compute_capability\":\"%d.%d\","
                "\"runtime_version\":%d,\"compiled_arch\":\"sm_%d\","
                "\"async_copy_checksum\":%d,\"expected_checksum\":%d,"
                "\"async_copy_ok\":%s}\n",
                properties.name, properties.major, properties.minor,
                [] { int version = 0; cudaRuntimeGetVersion(&version); return version; }(),
                static_cast<int>(KAIRO_TARGET_ARCH), checksum, expected,
                checksum == expected ? "true" : "false");
    cudaFree(input);
    cudaFree(output);
    return checksum == expected ? 0 : 1;
}
