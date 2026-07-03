#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            throw std::runtime_error(std::string("CUDA error: ") +          \
                                     cudaGetErrorString(err));               \
        }                                                                    \
    } while (0)

#define CHECK_CUBLAS(call)                                                   \
    do {                                                                     \
        cublasStatus_t st = (call);                                          \
        if (st != CUBLAS_STATUS_SUCCESS) {                                   \
            throw std::runtime_error("cuBLAS error: " + std::to_string(st)); \
        }                                                                    \
    } while (0)

__global__ void init_half(__half* x, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = __float2half(static_cast<float>((i % 17) - 8) * 0.03125f);
    }
}

__global__ void init_bf16(__nv_bfloat16* x, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = __float2bfloat16(static_cast<float>((i % 17) - 8) * 0.03125f);
    }
}

__global__ void init_float(float* x, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = static_cast<float>((i % 17) - 8) * 0.03125f;
    }
}

__global__ void triad(const float* a, const float* b, float* c, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + 0.5f * b[i];
    }
}

float elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

void print_device() {
    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    std::cout << "device_name=" << prop.name << "\n";
    std::cout << "compute_capability=" << prop.major << "." << prop.minor << "\n";
    std::cout << "global_mem_GiB=" << std::fixed << std::setprecision(2)
              << static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0)
              << "\n";
    std::cout << "multi_processor_count=" << prop.multiProcessorCount << "\n";
    std::cout << "memory_bus_width_bits=" << prop.memoryBusWidth << "\n";
    std::cout << "memory_clock_rate_khz=" << prop.memoryClockRate << "\n";
}

void bench_gemm(int n,
                const std::string& label,
                cudaDataType_t dtype,
                cublasComputeType_t compute_type,
                void (*init_kernel)(void*, size_t),
                size_t elem_size,
                int warmup,
                int iters) {
    const size_t elems = static_cast<size_t>(n) * static_cast<size_t>(n);
    void* a = nullptr;
    void* b = nullptr;
    void* c = nullptr;
    CHECK_CUDA(cudaMalloc(&a, elems * elem_size));
    CHECK_CUDA(cudaMalloc(&b, elems * elem_size));
    CHECK_CUDA(cudaMalloc(&c, elems * elem_size));

    init_kernel(a, elems);
    init_kernel(b, elems);
    init_kernel(c, elems);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cublasHandle_t handle{};
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    float alpha = 1.0f;
    float beta = 0.0f;

    for (int i = 0; i < warmup; ++i) {
        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                                  &alpha, a, dtype, n, b, dtype, n, &beta, c,
                                  dtype, n, compute_type,
                                  CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start{}, stop{};
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                                  &alpha, a, dtype, n, b, dtype, n, &beta, c,
                                  dtype, n, compute_type,
                                  CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    const double seconds = elapsed_ms(start, stop) / 1000.0;
    const double tflops = (2.0 * n * n * n * iters) / seconds / 1.0e12;
    std::cout << label << "_n=" << n << " iters=" << iters
              << " time_s=" << std::fixed << std::setprecision(4) << seconds
              << " tflops=" << std::setprecision(2) << tflops << "\n";

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(a));
    CHECK_CUDA(cudaFree(b));
    CHECK_CUDA(cudaFree(c));
}

void init_half_adapter(void* p, size_t n) {
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    init_half<<<blocks, threads>>>(static_cast<__half*>(p), n);
}

void init_bf16_adapter(void* p, size_t n) {
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    init_bf16<<<blocks, threads>>>(static_cast<__nv_bfloat16*>(p), n);
}

void init_float_adapter(void* p, size_t n) {
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    init_float<<<blocks, threads>>>(static_cast<float*>(p), n);
}

void bench_bandwidth(size_t elems, int warmup, int iters) {
    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;
    CHECK_CUDA(cudaMalloc(&a, elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&b, elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&c, elems * sizeof(float)));

    const int threads = 256;
    const int blocks = static_cast<int>((elems + threads - 1) / threads);
    init_float<<<blocks, threads>>>(a, elems);
    init_float<<<blocks, threads>>>(b, elems);
    init_float<<<blocks, threads>>>(c, elems);
    CHECK_CUDA(cudaDeviceSynchronize());

    for (int i = 0; i < warmup; ++i) {
        triad<<<blocks, threads>>>(a, b, c, elems);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start{}, stop{};
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        triad<<<blocks, threads>>>(a, b, c, elems);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    const double seconds = elapsed_ms(start, stop) / 1000.0;
    const double bytes = 3.0 * elems * sizeof(float) * iters;
    std::cout << "mem_triad_elems=" << elems << " iters=" << iters
              << " time_s=" << std::fixed << std::setprecision(4) << seconds
              << " bandwidth_TBps=" << std::setprecision(2) << bytes / seconds / 1.0e12
              << "\n";

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(a));
    CHECK_CUDA(cudaFree(b));
    CHECK_CUDA(cudaFree(c));
}

int main() {
    try {
        print_device();
        bench_gemm(8192, "fp16_tc", CUDA_R_16F, CUBLAS_COMPUTE_32F_FAST_16F,
                   init_half_adapter, sizeof(__half), 5, 20);
        bench_gemm(16384, "fp16_tc", CUDA_R_16F, CUBLAS_COMPUTE_32F_FAST_16F,
                   init_half_adapter, sizeof(__half), 3, 10);
        bench_gemm(8192, "bf16_tc", CUDA_R_16BF, CUBLAS_COMPUTE_32F_FAST_16BF,
                   init_bf16_adapter, sizeof(__nv_bfloat16), 5, 20);
        bench_gemm(16384, "bf16_tc", CUDA_R_16BF, CUBLAS_COMPUTE_32F_FAST_16BF,
                   init_bf16_adapter, sizeof(__nv_bfloat16), 3, 10);
        bench_gemm(8192, "tf32_tc", CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_TF32,
                   init_float_adapter, sizeof(float), 5, 20);
        bench_gemm(16384, "tf32_tc", CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_TF32,
                   init_float_adapter, sizeof(float), 3, 10);
        bench_bandwidth(268435456ULL, 10, 100);
    } catch (const std::exception& e) {
        std::cerr << "benchmark_failed=" << e.what() << "\n";
        return 1;
    }
    return 0;
}
