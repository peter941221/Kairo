#include <cuda.h>
#include <cuda/ptx>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

static void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(2);
    }
}

static void check_driver(CUresult status, const char* operation) {
    if (status != CUDA_SUCCESS) {
        const char* name = nullptr;
        const char* message = nullptr;
        cuGetErrorName(status, &name);
        cuGetErrorString(status, &message);
        std::fprintf(stderr, "%s: %s (%s)\n", operation, message, name);
        std::exit(2);
    }
}

__global__ void tma_copy_kernel(const CUtensorMap* tensor_map, float* output) {
    __shared__ alignas(128) float tile[16][16];
    __shared__ alignas(8) uint64_t barrier;
    if (threadIdx.x == 0) {
        cuda::ptx::mbarrier_init(&barrier, 1);
        cuda::ptx::fence_proxy_tensormap_generic(
            cuda::ptx::sem_release, cuda::ptx::scope_cta);
        const uint64_t state = cuda::ptx::mbarrier_arrive_expect_tx(
            cuda::ptx::sem_release, cuda::ptx::scope_cta,
            cuda::ptx::space_shared, &barrier, 16 * 16 * sizeof(float));
        const int32_t coordinates[2] = {0, 0};
        cuda::ptx::cp_async_bulk_tensor(
            cuda::ptx::space_shared, cuda::ptx::space_global, tile,
            tensor_map, coordinates, &barrier);
        while (!cuda::ptx::mbarrier_try_wait(&barrier, state)) {
        }
    }
    __syncthreads();
    const int index = static_cast<int>(threadIdx.x);
    if (index < 256) output[index] = tile[index / 16][index % 16];
}

int main() {
    check_cuda(cudaFree(nullptr), "cudaFree(0)");
    check_driver(cuInit(0), "cuInit");
    CUdevice device = 0;
    CUcontext context = nullptr;
    check_driver(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
    check_driver(cuCtxSetCurrent(context), "cuCtxSetCurrent");

    float* input = nullptr;
    float* output = nullptr;
    check_cuda(cudaMalloc(&input, 256 * sizeof(float)), "cudaMalloc(input)");
    check_cuda(cudaMalloc(&output, 256 * sizeof(float)), "cudaMalloc(output)");
    float host_input[256];
    for (int index = 0; index < 256; ++index) host_input[index] = static_cast<float>(index);
    check_cuda(cudaMemcpy(input, host_input, sizeof(host_input), cudaMemcpyHostToDevice), "copy(input)");

    CUtensorMap tensor_map{};
    const cuuint64_t global_dim[2] = {16, 16};
    const cuuint64_t global_stride[1] = {16 * sizeof(float)};
    const cuuint32_t box_dim[2] = {16, 16};
    const cuuint32_t element_stride[2] = {1, 1};
    check_driver(cuTensorMapEncodeTiled(
                     &tensor_map, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, input,
                     global_dim, global_stride, box_dim, element_stride,
                     CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                     CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
                 "cuTensorMapEncodeTiled");
    CUtensorMap* device_map = nullptr;
    check_cuda(cudaMalloc(&device_map, sizeof(CUtensorMap)), "cudaMalloc(tensor_map)");
    check_cuda(cudaMemcpy(device_map, &tensor_map, sizeof(CUtensorMap), cudaMemcpyHostToDevice), "copy(tensor_map)");
    tma_copy_kernel<<<1, 256>>>(device_map, output);
    check_cuda(cudaGetLastError(), "tma_copy_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "tma_copy_kernel synchronize");
    float host_output[256];
    check_cuda(cudaMemcpy(host_output, output, sizeof(host_output), cudaMemcpyDeviceToHost), "copy(output)");
    float max_error = 0.0f;
    for (int index = 0; index < 256; ++index) {
        max_error = fmaxf(max_error, fabsf(host_output[index] - host_input[index]));
    }
    std::printf("{\"operation\":\"tma_2d_copy\",\"shape\":[16,16],\"max_abs_error\":%.8f,\"tma_copy_ok\":%s}\n",
                max_error, max_error < 1.0e-6f ? "true" : "false");
    cudaFree(device_map);
    cudaFree(input);
    cudaFree(output);
    cuDevicePrimaryCtxRelease(device);
    return max_error < 1.0e-6f ? 0 : 1;
}
