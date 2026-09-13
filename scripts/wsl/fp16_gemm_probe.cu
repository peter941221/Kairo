#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

__global__ void kairo_fp16_tiled_gemm(const __half* a, const __half* b, float* c,
                                      int m, int n, int k) {
    constexpr int TILE = 16;
    __shared__ __half tile_a[TILE][TILE];
    __shared__ __half tile_b[TILE][TILE];
    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;
    float accumulator = 0.0f;
    for (int base = 0; base < k; base += TILE) {
        const int a_col = base + threadIdx.x;
        const int b_row = base + threadIdx.y;
        tile_a[threadIdx.y][threadIdx.x] =
            (row < m && a_col < k) ? a[row * k + a_col] : __float2half(0.0f);
        tile_b[threadIdx.y][threadIdx.x] =
            (b_row < k && col < n) ? b[b_row * n + col] : __float2half(0.0f);
        __syncthreads();
        for (int inner = 0; inner < TILE; ++inner) {
            accumulator += __half2float(tile_a[threadIdx.y][inner]) *
                           __half2float(tile_b[inner][threadIdx.x]);
        }
        __syncthreads();
    }
    if (row < m && col < n) c[row * n + col] = accumulator;
}

// A wider tile gives each thread a 2x2 output fragment, amortizing global
// loads and address arithmetic while retaining the simple correctness path.
__global__ void kairo_fp16_tiled_gemm_2x2(const __half* a, const __half* b, float* c,
                                          int m, int n, int k) {
    constexpr int TILE = 32;
    __shared__ __half tile_a[TILE][TILE];
    __shared__ __half tile_b[TILE][TILE];
    const int row = blockIdx.y * TILE + threadIdx.y * 2;
    const int col = blockIdx.x * TILE + threadIdx.x * 2;
    float accum00 = 0.0f;
    float accum01 = 0.0f;
    float accum10 = 0.0f;
    float accum11 = 0.0f;
    for (int base = 0; base < k; base += TILE) {
        const int a_col = base + threadIdx.x * 2;
        const int b_row = base + threadIdx.y * 2;
        for (int row_offset = 0; row_offset < 2; ++row_offset) {
            for (int col_offset = 0; col_offset < 2; ++col_offset) {
                const int shared_row = threadIdx.y * 2 + row_offset;
                const int shared_col = threadIdx.x * 2 + col_offset;
                const int global_a_row = row + row_offset;
                const int global_a_col = a_col + col_offset;
                const int global_b_row = b_row + row_offset;
                const int global_b_col = col + col_offset;
                tile_a[shared_row][shared_col] =
                    (global_a_row < m && global_a_col < k)
                        ? a[global_a_row * k + global_a_col]
                        : __float2half(0.0f);
                tile_b[shared_row][shared_col] =
                    (global_b_row < k && global_b_col < n)
                        ? b[global_b_row * n + global_b_col]
                        : __float2half(0.0f);
            }
        }
        __syncthreads();
        for (int inner = 0; inner < TILE; ++inner) {
            const float a0 = __half2float(tile_a[threadIdx.y * 2][inner]);
            const float a1 = __half2float(tile_a[threadIdx.y * 2 + 1][inner]);
            const float b0 = __half2float(tile_b[inner][threadIdx.x * 2]);
            const float b1 = __half2float(tile_b[inner][threadIdx.x * 2 + 1]);
            accum00 += a0 * b0;
            accum01 += a0 * b1;
            accum10 += a1 * b0;
            accum11 += a1 * b1;
        }
        __syncthreads();
    }
    if (row < m && col < n) c[row * n + col] = accum00;
    if (row < m && col + 1 < n) c[row * n + col + 1] = accum01;
    if (row + 1 < m && col < n) c[(row + 1) * n + col] = accum10;
    if (row + 1 < m && col + 1 < n) c[(row + 1) * n + col + 1] = accum11;
}

// The same fragment shape with 4-byte asynchronous global-to-shared copies.
// Boundary elements use ordinary stores so arbitrary dimensions retain the
// exact correctness behavior of the reference kernel.
__global__ void kairo_fp16_tiled_gemm_2x2_async(const __half* a, const __half* b, float* c,
                                                int m, int n, int k) {
    constexpr int TILE = 32;
    __shared__ __half tile_a[TILE][TILE];
    __shared__ __half tile_b[TILE][TILE];
    const int row = blockIdx.y * TILE + threadIdx.y * 2;
    const int col = blockIdx.x * TILE + threadIdx.x * 2;
    float accum00 = 0.0f;
    float accum01 = 0.0f;
    float accum10 = 0.0f;
    float accum11 = 0.0f;
    for (int base = 0; base < k; base += TILE) {
        const int a_col = base + threadIdx.x * 2;
        const int b_row = base + threadIdx.y * 2;
        for (int row_offset = 0; row_offset < 2; ++row_offset) {
            const int shared_row = threadIdx.y * 2 + row_offset;
            const int global_a_row = row + row_offset;
            const int global_b_row = b_row + row_offset;
            if (global_a_row < m && a_col + 1 < k &&
                ((global_a_row * k + a_col) & 1) == 0) {
                __pipeline_memcpy_async(&tile_a[shared_row][threadIdx.x * 2],
                                        &a[global_a_row * k + a_col],
                                        2 * sizeof(__half));
            } else {
                for (int col_offset = 0; col_offset < 2; ++col_offset) {
                    const int global_col = a_col + col_offset;
                    tile_a[shared_row][threadIdx.x * 2 + col_offset] =
                        (global_a_row < m && global_col < k)
                            ? a[global_a_row * k + global_col]
                            : __float2half(0.0f);
                }
            }
            if (global_b_row < k && col + 1 < n &&
                ((global_b_row * n + col) & 1) == 0) {
                __pipeline_memcpy_async(&tile_b[shared_row][threadIdx.x * 2],
                                        &b[global_b_row * n + col],
                                        2 * sizeof(__half));
            } else {
                for (int col_offset = 0; col_offset < 2; ++col_offset) {
                    const int global_col = col + col_offset;
                    tile_b[shared_row][threadIdx.x * 2 + col_offset] =
                        (global_b_row < k && global_col < n)
                            ? b[global_b_row * n + global_col]
                            : __float2half(0.0f);
                }
            }
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();
        for (int inner = 0; inner < TILE; ++inner) {
            const float a0 = __half2float(tile_a[threadIdx.y * 2][inner]);
            const float a1 = __half2float(tile_a[threadIdx.y * 2 + 1][inner]);
            const float b0 = __half2float(tile_b[inner][threadIdx.x * 2]);
            const float b1 = __half2float(tile_b[inner][threadIdx.x * 2 + 1]);
            accum00 += a0 * b0;
            accum01 += a0 * b1;
            accum10 += a1 * b0;
            accum11 += a1 * b1;
        }
        __syncthreads();
    }
    if (row < m && col < n) c[row * n + col] = accum00;
    if (row < m && col + 1 < n) c[row * n + col + 1] = accum01;
    if (row + 1 < m && col < n) c[(row + 1) * n + col] = accum10;
    if (row + 1 < m && col + 1 < n) c[(row + 1) * n + col + 1] = accum11;
}

// One warp owns a 16x16 output tile and uses FP16 tensor-core MMA. This path
// intentionally accepts only dimensions divisible by 16 and is guarded by
// the launcher; arbitrary-boundary inputs use the shared-memory fallback.
__global__ void kairo_fp16_wmma(const __half* a, const __half* b, float* c,
                                int m, int n, int k) {
    using namespace nvcuda::wmma;
    constexpr int TILE = 16;
    const int warp = threadIdx.x / 32;
    const int row = blockIdx.y * 64 + warp * TILE;
    const int col = blockIdx.x * TILE;
    if (row >= m || col >= n) return;
    fragment<accumulator, TILE, TILE, TILE, float> accumulator;
    fill_fragment(accumulator, 0.0f);
    for (int base = 0; base < k; base += TILE) {
        fragment<matrix_a, TILE, TILE, TILE, __half, row_major> a_fragment;
        fragment<matrix_b, TILE, TILE, TILE, __half, row_major> b_fragment;
        load_matrix_sync(a_fragment, a + row * k + base, k);
        load_matrix_sync(b_fragment, b + base * n + col, n);
        mma_sync(accumulator, a_fragment, b_fragment, accumulator);
    }
    store_matrix_sync(c + row * n + col, accumulator, n, mem_row_major);
}

// Stage one 64x16 A tile and one 16x16 B tile per block. Four warps reuse the
// B tile instead of issuing the same global loads independently.
__global__ void kairo_fp16_wmma_shared(const __half* a, const __half* b, float* c,
                                        int m, int n, int k) {
    using namespace nvcuda::wmma;
    constexpr int TILE = 16;
    __shared__ __half tile_a[64][TILE];
    __shared__ __half tile_b[TILE][TILE];
    const int warp = threadIdx.x / 32;
    const int block_row = blockIdx.y * 64;
    const int row = block_row + warp * TILE;
    const int col = blockIdx.x * TILE;
    fragment<accumulator, TILE, TILE, TILE, float> accumulator;
    fill_fragment(accumulator, 0.0f);
    for (int base = 0; base < k; base += TILE) {
        for (int index = threadIdx.x; index < 64 * TILE; index += 128) {
            const int local_row = index / TILE;
            const int local_col = index % TILE;
            const int global_row = block_row + local_row;
            const int global_col = base + local_col;
            tile_a[local_row][local_col] =
                (global_row < m && global_col < k)
                    ? a[global_row * k + global_col]
                    : __float2half(0.0f);
        }
        for (int index = threadIdx.x; index < TILE * TILE; index += 128) {
            const int local_row = index / TILE;
            const int local_col = index % TILE;
            const int global_row = base + local_row;
            const int global_col = col + local_col;
            tile_b[local_row][local_col] =
                (global_row < k && global_col < n)
                    ? b[global_row * n + global_col]
                    : __float2half(0.0f);
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

static void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
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

static float event_ms(cudaEvent_t start, cudaEvent_t stop, int iterations) {
    float elapsed = 0.0f;
    check(cudaEventElapsedTime(&elapsed, start, stop), "cudaEventElapsedTime");
    return elapsed / static_cast<float>(iterations);
}

int main(int argc, char** argv) {
    const int m = argc > 1 ? std::atoi(argv[1]) : 1024;
    const int n = argc > 2 ? std::atoi(argv[2]) : 1024;
    const int k = argc > 3 ? std::atoi(argv[3]) : 1024;
    const int iterations = argc > 4 ? std::atoi(argv[4]) : 50;
    const char* requested_variant = argc > 5 ? argv[5] : "tile32x32_output2x2";
    const bool optimized = std::strcmp(requested_variant, "tile16x16_output1x1") != 0;
    const bool async_variant = std::strcmp(requested_variant, "tile32x32_output2x2_async") == 0;
    const bool wmma_variant = std::strcmp(requested_variant, "wmma_fp16") == 0;
    const bool wmma_shared_variant = std::strcmp(requested_variant, "wmma_fp16_shared") == 0;
    if ((wmma_variant || wmma_shared_variant) &&
        ((m % 16) != 0 || (n % 16) != 0 || (k % 16) != 0)) {
        std::fprintf(stderr, "wmma_fp16 requires dimensions divisible by 16\n");
        return 2;
    }
    if (m < 1 || n < 1 || k < 1 || iterations < 1) {
        std::fprintf(stderr, "dimensions and iterations must be positive\n");
        return 2;
    }

    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
    std::mt19937 generator(5090);
    std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);
    std::vector<__half> host_a(static_cast<size_t>(m) * k);
    std::vector<__half> host_b(static_cast<size_t>(k) * n);
    for (auto& value : host_a) value = __float2half(distribution(generator));
    for (auto& value : host_b) value = __float2half(distribution(generator));
    std::vector<float> host_custom(static_cast<size_t>(m) * n);
    std::vector<float> host_blas(static_cast<size_t>(m) * n);

    __half* device_a = nullptr;
    __half* device_b = nullptr;
    float* device_custom = nullptr;
    float* device_blas = nullptr;
    check(cudaMalloc(&device_a, host_a.size() * sizeof(__half)), "cudaMalloc(a)");
    check(cudaMalloc(&device_b, host_b.size() * sizeof(__half)), "cudaMalloc(b)");
    check(cudaMalloc(&device_custom, host_custom.size() * sizeof(float)), "cudaMalloc(custom)");
    check(cudaMalloc(&device_blas, host_blas.size() * sizeof(float)), "cudaMalloc(blas)");
    check(cudaMemcpy(device_a, host_a.data(), host_a.size() * sizeof(__half), cudaMemcpyHostToDevice), "copy(a)");
    check(cudaMemcpy(device_b, host_b.data(), host_b.size() * sizeof(__half), cudaMemcpyHostToDevice), "copy(b)");

    const dim3 block = (wmma_variant || wmma_shared_variant) ? dim3(128, 1, 1) : dim3(16, 16, 1);
    const dim3 grid = (wmma_variant || wmma_shared_variant) ? dim3((n + 15) / 16, (m + 63) / 64)
                                   : (optimized ? dim3((n + 31) / 32, (m + 31) / 32)
                                                : dim3((n + 15) / 16, (m + 15) / 16));
    if (wmma_shared_variant) {
        kairo_fp16_wmma_shared<<<grid, block>>>(device_a, device_b, device_custom, m, n, k);
    } else if (wmma_variant) {
        kairo_fp16_wmma<<<grid, block>>>(device_a, device_b, device_custom, m, n, k);
    } else if (async_variant) {
        kairo_fp16_tiled_gemm_2x2_async<<<grid, block>>>(device_a, device_b, device_custom, m, n, k);
    } else if (optimized) {
        kairo_fp16_tiled_gemm_2x2<<<grid, block>>>(device_a, device_b, device_custom, m, n, k);
    } else {
        kairo_fp16_tiled_gemm<<<grid, block>>>(device_a, device_b, device_custom, m, n, k);
    }
    check(cudaGetLastError(), "custom warmup launch");
    check(cudaDeviceSynchronize(), "custom warmup synchronize");
    check(cudaMemcpy(host_custom.data(), device_custom, host_custom.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy(custom)");
    // A CPU O(MNK) reference is useful for boundary tests but becomes
    // impractical for production-sized matrices. Large shapes use the
    // independent cuBLAS result below as their correctness oracle.
    const bool cpu_reference_checked =
        static_cast<long double>(m) * n * k <= 50'000'000.0L;
    float custom_error = -1.0f;
    if (cpu_reference_checked) {
        std::vector<float> host_reference(host_custom.size(), 0.0f);
        for (int row = 0; row < m; ++row) {
            for (int col = 0; col < n; ++col) {
                float sum = 0.0f;
                for (int inner = 0; inner < k; ++inner) {
                    sum += __half2float(host_a[static_cast<size_t>(row) * k + inner]) *
                           __half2float(host_b[static_cast<size_t>(inner) * n + col]);
                }
                host_reference[static_cast<size_t>(row) * n + col] = sum;
            }
        }
        custom_error = max_error(host_custom, host_reference);
    }

    cublasHandle_t handle = nullptr;
    check_blas(cublasCreate(&handle), "cublasCreate");
    check_blas(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH), "cublasSetMathMode");
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // Row-major A(MxK), B(KxN) become column-major B^T(NxK), A^T(KxM).
    check_blas(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                            &alpha, device_b, CUDA_R_16F, n,
                            device_a, CUDA_R_16F, k, &beta, device_blas,
                            CUDA_R_32F, n, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT), "cublasGemmEx warmup");
    check(cudaDeviceSynchronize(), "cublas warmup synchronize");

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check(cudaEventCreate(&start), "cudaEventCreate(start)");
    check(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    check(cudaEventRecord(start), "custom start");
    for (int iteration = 0; iteration < iterations; ++iteration) {
        if (wmma_shared_variant) {
            kairo_fp16_wmma_shared<<<grid, block>>>(device_a, device_b, device_custom, m, n, k);
        } else if (wmma_variant) {
            kairo_fp16_wmma<<<grid, block>>>(device_a, device_b, device_custom, m, n, k);
        } else if (async_variant) {
            kairo_fp16_tiled_gemm_2x2_async<<<grid, block>>>(device_a, device_b, device_custom, m, n, k);
        } else if (optimized) {
            kairo_fp16_tiled_gemm_2x2<<<grid, block>>>(device_a, device_b, device_custom, m, n, k);
        } else {
            kairo_fp16_tiled_gemm<<<grid, block>>>(device_a, device_b, device_custom, m, n, k);
        }
    }
    check(cudaEventRecord(stop), "custom stop");
    check(cudaEventSynchronize(stop), "custom synchronize");
    const float custom_ms = event_ms(start, stop, iterations);

    check(cudaEventRecord(start), "blas start");
    for (int iteration = 0; iteration < iterations; ++iteration) {
        check_blas(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                                &alpha, device_b, CUDA_R_16F, n,
                                device_a, CUDA_R_16F, k, &beta, device_blas,
                                CUDA_R_32F, n, CUBLAS_COMPUTE_32F,
                                CUBLAS_GEMM_DEFAULT), "cublasGemmEx");
    }
    check(cudaEventRecord(stop), "blas stop");
    check(cudaEventSynchronize(stop), "blas synchronize");
    const float blas_ms = event_ms(start, stop, iterations);
    check(cudaMemcpy(host_blas.data(), device_blas, host_blas.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy(blas)");
    const float cross_error = max_error(host_custom, host_blas);
    const double operations = 2.0 * static_cast<double>(m) * n * k;
    std::printf("{\"gpu\":\"%s\",\"shape\":[%d,%d,%d],\"iterations\":%d,"
                "\"variant\":\"%s\",\"custom_ms\":%.6f,\"cublas_ms\":%.6f,"
                "\"custom_gflops\":%.3f,\"cublas_gflops\":%.3f,"
                "\"cpu_reference_checked\":%s,\"custom_max_abs_error\":%.8f,"
                "\"custom_vs_cublas_max_abs_error\":%.8f,\"correctness_ok\":%s}\n",
                properties.name, m, n, k, iterations,
                requested_variant,
                custom_ms, blas_ms,
                operations / (custom_ms * 1.0e6), operations / (blas_ms * 1.0e6),
                cpu_reference_checked ? "true" : "false", custom_error, cross_error,
                ((!cpu_reference_checked || custom_error < 0.02f) && cross_error < 0.02f)
                    ? "true"
                    : "false");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_custom);
    cudaFree(device_blas);
    return ((!cpu_reference_checked || custom_error < 0.02f) && cross_error < 0.02f) ? 0 : 1;
}
