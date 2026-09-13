#include <cuda.h>
#include <cuda/ptx>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

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
static void check_blas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s: cublas status %d\n", operation, static_cast<int>(status));
        std::exit(2);
    }
}
static float max_error(const std::vector<float>& lhs, const std::vector<float>& rhs) {
    float error = 0.0f;
    for (size_t index = 0; index < lhs.size(); ++index) {
        error = std::max(error, std::fabs(lhs[index] - rhs[index]));
    }
    return error;
}

__global__ void tma_wmma_gemm(const __half* a, const __half* b, float* c,
                              const CUtensorMap* a_map, const CUtensorMap* b_map,
                              int m, int n, int k) {
    using namespace nvcuda::wmma;
    constexpr int TILE = 16;
    __shared__ alignas(128) __half tile_a[64][TILE];
    __shared__ alignas(128) __half tile_b[TILE][TILE];
    __shared__ alignas(8) uint64_t barrier;
    const int warp = threadIdx.x / 32;
    const int block_row = blockIdx.y * 64;
    const int row = block_row + warp * TILE;
    const int col = blockIdx.x * TILE;
    if (threadIdx.x == 0) {
        cuda::ptx::mbarrier_init(&barrier, 1);
        cuda::ptx::fence_proxy_tensormap_generic(
            cuda::ptx::sem_release, cuda::ptx::scope_cta);
    }
    __syncthreads();
    fragment<accumulator, TILE, TILE, TILE, float> accumulator;
    fill_fragment(accumulator, 0.0f);
    for (int base = 0; base < k; base += TILE) {
        if (threadIdx.x == 0) {
            const int32_t a_coordinates[2] = {base, block_row};
            const int32_t b_coordinates[2] = {col, base};
            const uint64_t state = cuda::ptx::mbarrier_arrive_expect_tx(
                cuda::ptx::sem_release, cuda::ptx::scope_cta,
                cuda::ptx::space_shared, &barrier,
                static_cast<uint32_t>((64 * TILE + TILE * TILE) * sizeof(__half)));
            cuda::ptx::cp_async_bulk_tensor(
                cuda::ptx::space_shared, cuda::ptx::space_global, tile_a,
                a_map, a_coordinates, &barrier);
            cuda::ptx::cp_async_bulk_tensor(
                cuda::ptx::space_shared, cuda::ptx::space_global, tile_b,
                b_map, b_coordinates, &barrier);
            while (!cuda::ptx::mbarrier_try_wait(&barrier, state)) {
            }
        }
        __syncthreads();
        if (row < m && col < n) {
            fragment<matrix_a, TILE, TILE, TILE, __half, row_major> a_fragment;
            fragment<matrix_b, TILE, TILE, TILE, __half, row_major> b_fragment;
            load_matrix_sync(a_fragment, &tile_a[warp * TILE][0], TILE);
            load_matrix_sync(b_fragment, &tile_b[0][0], TILE);
            mma_sync(accumulator, a_fragment, b_fragment, accumulator);
        }
        __syncthreads();
    }
    if (row < m && col < n) store_matrix_sync(c + row * n + col, accumulator, n, mem_row_major);
}

template <int ROWS>
__global__ void tma_wmma_gemm_rows(const __half* a, const __half* b, float* c,
                                   const CUtensorMap* a_map, const CUtensorMap* b_map,
                                   int m, int n, int k) {
    using namespace nvcuda::wmma;
    constexpr int TILE = 16;
    __shared__ alignas(128) __half tile_a[ROWS][TILE];
    __shared__ alignas(128) __half tile_b[TILE][TILE];
    __shared__ alignas(8) uint64_t barrier;
    const int warp = threadIdx.x / 32;
    const int block_row = blockIdx.y * ROWS;
    const int row = block_row + warp * TILE;
    const int col = blockIdx.x * TILE;
    if (threadIdx.x == 0) {
        cuda::ptx::mbarrier_init(&barrier, 1);
        cuda::ptx::fence_proxy_tensormap_generic(
            cuda::ptx::sem_release, cuda::ptx::scope_cta);
    }
    __syncthreads();
    fragment<accumulator, TILE, TILE, TILE, float> accumulator;
    fill_fragment(accumulator, 0.0f);
    for (int base = 0; base < k; base += TILE) {
        if (threadIdx.x == 0) {
            const int32_t a_coordinates[2] = {base, block_row};
            const int32_t b_coordinates[2] = {col, base};
            const uint64_t state = cuda::ptx::mbarrier_arrive_expect_tx(
                cuda::ptx::sem_release, cuda::ptx::scope_cta,
                cuda::ptx::space_shared, &barrier,
                static_cast<uint32_t>((ROWS * TILE + TILE * TILE) * sizeof(__half)));
            cuda::ptx::cp_async_bulk_tensor(
                cuda::ptx::space_shared, cuda::ptx::space_global, tile_a,
                a_map, a_coordinates, &barrier);
            cuda::ptx::cp_async_bulk_tensor(
                cuda::ptx::space_shared, cuda::ptx::space_global, tile_b,
                b_map, b_coordinates, &barrier);
            while (!cuda::ptx::mbarrier_try_wait(&barrier, state)) {
            }
        }
        __syncthreads();
        if (row < m && col < n) {
            fragment<matrix_a, TILE, TILE, TILE, __half, row_major> a_fragment;
            fragment<matrix_b, TILE, TILE, TILE, __half, row_major> b_fragment;
            load_matrix_sync(a_fragment, &tile_a[warp * TILE][0], TILE);
            load_matrix_sync(b_fragment, &tile_b[0][0], TILE);
            mma_sync(accumulator, a_fragment, b_fragment, accumulator);
        }
        __syncthreads();
    }
    if (row < m && col < n) store_matrix_sync(c + row * n + col, accumulator, n, mem_row_major);
}

__global__ void tma_wmma_gemm_double(const __half* a, const __half* b, float* c,
                                     const CUtensorMap* a_map, const CUtensorMap* b_map,
                                     int m, int n, int k) {
    using namespace nvcuda::wmma;
    constexpr int TILE = 16;
    __shared__ alignas(128) __half tile_a[2][64][TILE];
    __shared__ alignas(128) __half tile_b[2][TILE][TILE];
    __shared__ alignas(8) uint64_t barriers[2];
    const int warp = threadIdx.x / 32;
    const int block_row = blockIdx.y * 64;
    const int row = block_row + warp * TILE;
    const int col = blockIdx.x * TILE;
    uint64_t states[2] = {0, 0};
    if (threadIdx.x == 0) {
        cuda::ptx::mbarrier_init(&barriers[0], 1);
        cuda::ptx::mbarrier_init(&barriers[1], 1);
        cuda::ptx::fence_proxy_tensormap_generic(
            cuda::ptx::sem_release, cuda::ptx::scope_cta);
        const int32_t a_coordinates[2] = {0, block_row};
        const int32_t b_coordinates[2] = {col, 0};
        states[0] = cuda::ptx::mbarrier_arrive_expect_tx(
            cuda::ptx::sem_release, cuda::ptx::scope_cta,
            cuda::ptx::space_shared, &barriers[0],
            static_cast<uint32_t>((64 * TILE + TILE * TILE) * sizeof(__half)));
        cuda::ptx::cp_async_bulk_tensor(
            cuda::ptx::space_shared, cuda::ptx::space_global, tile_a[0],
            a_map, a_coordinates, &barriers[0]);
        cuda::ptx::cp_async_bulk_tensor(
            cuda::ptx::space_shared, cuda::ptx::space_global, tile_b[0],
            b_map, b_coordinates, &barriers[0]);
        while (!cuda::ptx::mbarrier_try_wait(&barriers[0], states[0])) {
        }
    }
    __syncthreads();
    fragment<accumulator, TILE, TILE, TILE, float> accumulator;
    fill_fragment(accumulator, 0.0f);
    int current = 0;
    for (int base = 0; base < k; base += TILE) {
        const int next = current ^ 1;
        if (base + TILE < k && threadIdx.x == 0) {
            const int32_t a_coordinates[2] = {base + TILE, block_row};
            const int32_t b_coordinates[2] = {col, base + TILE};
            states[next] = cuda::ptx::mbarrier_arrive_expect_tx(
                cuda::ptx::sem_release, cuda::ptx::scope_cta,
                cuda::ptx::space_shared, &barriers[next],
                static_cast<uint32_t>((64 * TILE + TILE * TILE) * sizeof(__half)));
            cuda::ptx::cp_async_bulk_tensor(
                cuda::ptx::space_shared, cuda::ptx::space_global, tile_a[next],
                a_map, a_coordinates, &barriers[next]);
            cuda::ptx::cp_async_bulk_tensor(
                cuda::ptx::space_shared, cuda::ptx::space_global, tile_b[next],
                b_map, b_coordinates, &barriers[next]);
        }
        if (row < m && col < n) {
            fragment<matrix_a, TILE, TILE, TILE, __half, row_major> a_fragment;
            fragment<matrix_b, TILE, TILE, TILE, __half, row_major> b_fragment;
            load_matrix_sync(a_fragment, &tile_a[current][warp * TILE][0], TILE);
            load_matrix_sync(b_fragment, &tile_b[current][0][0], TILE);
            mma_sync(accumulator, a_fragment, b_fragment, accumulator);
        }
        __syncthreads();
        if (base + TILE < k && threadIdx.x == 0) {
            while (!cuda::ptx::mbarrier_try_wait(&barriers[next], states[next])) {
            }
        }
        __syncthreads();
        current = next;
    }
    if (row < m && col < n) store_matrix_sync(c + row * n + col, accumulator, n, mem_row_major);
}

static float event_ms(cudaEvent_t start, cudaEvent_t stop, int iterations) {
    float elapsed = 0.0f;
    check_cuda(cudaEventElapsedTime(&elapsed, start, stop), "cudaEventElapsedTime");
    return elapsed / static_cast<float>(iterations);
}

int main(int argc, char** argv) {
    const int m = argc > 1 ? std::atoi(argv[1]) : 1024;
    const int n = argc > 2 ? std::atoi(argv[2]) : 1024;
    const int k = argc > 3 ? std::atoi(argv[3]) : 1024;
    const int iterations = argc > 4 ? std::atoi(argv[4]) : 50;
    const char* requested_variant = argc > 5 ? argv[5] : "single";
    const bool double_buffered = std::strcmp(requested_variant, "double") == 0;
    const bool m128_variant = std::strcmp(requested_variant, "m128") == 0;
    const bool m256_variant = std::strcmp(requested_variant, "m256") == 0;
    const bool m256_swizzle32_variant = std::strcmp(requested_variant, "m256_s32") == 0;
    if (m < 64 || n < 16 || k < 16 || (m % 16) || (n % 16) || (k % 16) ||
        (m128_variant && (m % 128)) || ((m256_variant || m256_swizzle32_variant) && (m % 256))) {
        std::fprintf(stderr, "TMA-WMMA requires aligned dimensions (m128/m256 additionally require M divisible by 128/256)\n");
        return 2;
    }
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
    check_cuda(cudaFree(nullptr), "cudaFree(0)");
    check_driver(cuInit(0), "cuInit");
    CUcontext context = nullptr;
    check_driver(cuDevicePrimaryCtxRetain(&context, 0), "cuDevicePrimaryCtxRetain");
    check_driver(cuCtxSetCurrent(context), "cuCtxSetCurrent");

    std::mt19937 generator(5090);
    std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);
    std::vector<__half> host_a(static_cast<size_t>(m) * k);
    std::vector<__half> host_b(static_cast<size_t>(k) * n);
    for (auto& value : host_a) value = __float2half(distribution(generator));
    for (auto& value : host_b) value = __float2half(distribution(generator));
    std::vector<float> host_tma(static_cast<size_t>(m) * n);
    std::vector<float> host_blas(static_cast<size_t>(m) * n);
    __half* device_a = nullptr;
    __half* device_b = nullptr;
    float* device_tma = nullptr;
    float* device_blas = nullptr;
    check_cuda(cudaMalloc(&device_a, host_a.size() * sizeof(__half)), "cudaMalloc(a)");
    check_cuda(cudaMalloc(&device_b, host_b.size() * sizeof(__half)), "cudaMalloc(b)");
    check_cuda(cudaMalloc(&device_tma, host_tma.size() * sizeof(float)), "cudaMalloc(tma)");
    check_cuda(cudaMalloc(&device_blas, host_blas.size() * sizeof(float)), "cudaMalloc(blas)");
    check_cuda(cudaMemcpy(device_a, host_a.data(), host_a.size() * sizeof(__half), cudaMemcpyHostToDevice), "copy(a)");
    check_cuda(cudaMemcpy(device_b, host_b.data(), host_b.size() * sizeof(__half), cudaMemcpyHostToDevice), "copy(b)");

    CUtensorMap host_a_map{};
    CUtensorMap host_b_map{};
    const cuuint64_t a_dims[2] = {static_cast<cuuint64_t>(k), static_cast<cuuint64_t>(m)};
    const cuuint64_t b_dims[2] = {static_cast<cuuint64_t>(n), static_cast<cuuint64_t>(k)};
    const cuuint64_t a_strides[1] = {static_cast<cuuint64_t>(k * sizeof(__half))};
    const cuuint64_t b_strides[1] = {static_cast<cuuint64_t>(n * sizeof(__half))};
    const cuuint32_t a_box[2] = {16, (m256_variant || m256_swizzle32_variant) ? 256u : (m128_variant ? 128u : 64u)};
    const cuuint32_t b_box[2] = {16, 16};
    const cuuint32_t element_strides[2] = {1, 1};
    check_driver(cuTensorMapEncodeTiled(
                     &host_a_map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, device_a,
                     a_dims, a_strides, a_box, element_strides,
                     CU_TENSOR_MAP_INTERLEAVE_NONE,
                     m256_swizzle32_variant ? CU_TENSOR_MAP_SWIZZLE_32B : CU_TENSOR_MAP_SWIZZLE_NONE,
                     CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
                 "cuTensorMapEncodeTiled(A)");
    check_driver(cuTensorMapEncodeTiled(
                     &host_b_map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, device_b,
                     b_dims, b_strides, b_box, element_strides,
                     CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                     CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
                 "cuTensorMapEncodeTiled(B)");
    CUtensorMap* device_a_map = nullptr;
    CUtensorMap* device_b_map = nullptr;
    check_cuda(cudaMalloc(&device_a_map, sizeof(CUtensorMap)), "cudaMalloc(A map)");
    check_cuda(cudaMalloc(&device_b_map, sizeof(CUtensorMap)), "cudaMalloc(B map)");
    check_cuda(cudaMemcpy(device_a_map, &host_a_map, sizeof(CUtensorMap), cudaMemcpyHostToDevice), "copy(A map)");
    check_cuda(cudaMemcpy(device_b_map, &host_b_map, sizeof(CUtensorMap), cudaMemcpyHostToDevice), "copy(B map)");

    const int rows = (m256_variant || m256_swizzle32_variant) ? 256 : (m128_variant ? 128 : 64);
    const dim3 block(rows * 2, 1, 1);
    const dim3 grid((n + 15) / 16, (m + rows - 1) / rows);
    if (m256_variant || m256_swizzle32_variant) {
        tma_wmma_gemm_rows<256><<<grid, block>>>(device_a, device_b, device_tma, device_a_map, device_b_map, m, n, k);
    } else if (m128_variant) {
        tma_wmma_gemm_rows<128><<<grid, block>>>(device_a, device_b, device_tma, device_a_map, device_b_map, m, n, k);
    } else if (double_buffered) {
        tma_wmma_gemm_double<<<grid, block>>>(device_a, device_b, device_tma, device_a_map, device_b_map, m, n, k);
    } else {
        tma_wmma_gemm<<<grid, block>>>(device_a, device_b, device_tma, device_a_map, device_b_map, m, n, k);
    }
    check_cuda(cudaGetLastError(), "tma_wmma warmup launch");
    check_cuda(cudaDeviceSynchronize(), "tma_wmma warmup synchronize");

    cublasHandle_t handle = nullptr;
    check_blas(cublasCreate(&handle), "cublasCreate");
    check_blas(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH), "cublasSetMathMode");
    const float alpha = 1.0f;
    const float beta = 0.0f;
    check_blas(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                            &alpha, device_b, CUDA_R_16F, n,
                            device_a, CUDA_R_16F, k, &beta, device_blas,
                            CUDA_R_32F, n, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT), "cublas warmup");
    check_cuda(cudaDeviceSynchronize(), "cublas warmup synchronize");
    check_cuda(cudaMemcpy(host_tma.data(), device_tma, host_tma.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy(tma)");
    check_cuda(cudaMemcpy(host_blas.data(), device_blas, host_blas.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy(blas)");
    const float cross_error = max_error(host_tma, host_blas);
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    check_cuda(cudaEventRecord(start), "tma start");
    for (int iteration = 0; iteration < iterations; ++iteration) {
        if (m256_variant || m256_swizzle32_variant) {
            tma_wmma_gemm_rows<256><<<grid, block>>>(device_a, device_b, device_tma, device_a_map, device_b_map, m, n, k);
        } else if (m128_variant) {
            tma_wmma_gemm_rows<128><<<grid, block>>>(device_a, device_b, device_tma, device_a_map, device_b_map, m, n, k);
        } else if (double_buffered) {
            tma_wmma_gemm_double<<<grid, block>>>(device_a, device_b, device_tma, device_a_map, device_b_map, m, n, k);
        } else {
            tma_wmma_gemm<<<grid, block>>>(device_a, device_b, device_tma, device_a_map, device_b_map, m, n, k);
        }
    }
    check_cuda(cudaEventRecord(stop), "tma stop");
    check_cuda(cudaEventSynchronize(stop), "tma synchronize");
    const float tma_ms = event_ms(start, stop, iterations);
    check_cuda(cudaEventRecord(start), "blas start");
    for (int iteration = 0; iteration < iterations; ++iteration) {
        check_blas(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                                &alpha, device_b, CUDA_R_16F, n,
                                device_a, CUDA_R_16F, k, &beta, device_blas,
                                CUDA_R_32F, n, CUBLAS_COMPUTE_32F,
                                CUBLAS_GEMM_DEFAULT), "cublasGemmEx");
    }
    check_cuda(cudaEventRecord(stop), "blas stop");
    check_cuda(cudaEventSynchronize(stop), "blas synchronize");
    const float blas_ms = event_ms(start, stop, iterations);
    const double operations = 2.0 * static_cast<double>(m) * n * k;
    std::printf("{\"gpu\":\"%s\",\"shape\":[%d,%d,%d],\"iterations\":%d,\"variant\":\"%s\","
                "\"tma_ms\":%.6f,\"cublas_ms\":%.6f,\"tma_gflops\":%.3f,\"cublas_gflops\":%.3f,"
                "\"max_abs_error_vs_cublas\":%.8f,\"tma_ok\":%s}\n",
                properties.name, m, n, k, iterations,
                m256_swizzle32_variant ? "tma_wmma_fp16_m256_s32" : (m256_variant ? "tma_wmma_fp16_m256" : (m128_variant ? "tma_wmma_fp16_m128" : (double_buffered ? "tma_wmma_fp16_double" : "tma_wmma_fp16"))),
                tma_ms, blas_ms,
                operations / (tma_ms * 1.0e6), operations / (blas_ms * 1.0e6),
                cross_error, cross_error < 0.02f ? "true" : "false");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);
    cudaFree(device_a_map);
    cudaFree(device_b_map);
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_tma);
    cudaFree(device_blas);
    cuDevicePrimaryCtxRelease(0);
    return cross_error < 0.02f ? 0 : 1;
}
