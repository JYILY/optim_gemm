#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <float.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#define CHECK(call)                                         \
    do {                                                    \
        const cudaError_t error_code = call;                \
        if (error_code != cudaSuccess) {                    \
            printf("CUDA Error:\n");                        \
            printf("    File:%s:%d\n", __FILE__, __LINE__); \
            printf("    Error code: %d\n", error_code);     \
            printf("    Error text: %s\n",                  \
                cudaGetErrorString(error_code));            \
            exit(1);                                        \
        }                                                   \
    } while (0)

#define BLOCK_SIZE 32

#define CEIL_DIV(a, b) ((a + b - 1) / b)
typedef struct {
    int height;
    int width;
    int stride;
    float* elements;
} Matrix;

typedef void(cublasGemmFunc)(cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, int, const float*, const float*, int, const float*, int, const float*, float*, int);

void cpuSgemm(int m, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc)
{
    // sgemm : C = alpha * A * B + beta * C
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int u = 0; u < k; ++u) {
                sum += A[i * lda + u] * B[u * ldb + j];
            }
            C[i * ldc + j] = *alpha * sum + *beta * C[i * ldc + j];
        }
    }
}

void checkOk(cublasGemmFunc gemm, std::string name)
{
    const int m = 128;
    const int n = 256;
    const int k = 512;

    const float alpha = 1.0f;
    const float beta = 0.0;

    float *h_A, *h_B, *h_C, *d2h_C;
    int bytes = sizeof(float);

    h_A = (float*)malloc(m * k * bytes);
    h_B = (float*)malloc(k * n * bytes);
    h_C = (float*)malloc(m * n * bytes);
    d2h_C = (float*)malloc(m * n * bytes);

    // Randomly initialize input matrices A and B
    for (int i = 0; i < m * k; ++i) {
        h_A[i] = 0.5f * (rand() % 100);
    }

    for (int i = 0; i < k * n; ++i) {
        h_B[i] = 0.5f * (rand() % 100);
    }

    cpuSgemm(m, n, k, &alpha, h_A, k, h_B, n, &beta, h_C, n);

    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc(&d_A, m * k * bytes));
    CHECK(cudaMalloc(&d_B, k * n * bytes));
    CHECK(cudaMalloc(&d_C, m * n * bytes));

    CHECK(cudaMemcpy(d_A, h_A, m * k * bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, k * n * bytes, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    cublasCreate_v2(&handle);

    gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, k, d_B, n, &beta, d_C, n);

    CHECK(cudaMemcpy(d2h_C, d_C, m * n * bytes, cudaMemcpyDeviceToHost));

    cudaDeviceSynchronize();
    bool is_correct = true;
    for (int i = 0; i < m * n; ++i) {
        if (fabs(h_C[i] - d2h_C[i]) > 1e-5) {
            is_correct = false;
            std::cout << "The result is wrong at index " << i << std::endl;
            std::cout << "Expected: " << d2h_C[i] << ", but got: " << h_C[i] << std::endl;
            break;
        }
    }

    if (is_correct) {
        std::cout << name << " is OK!" << std::endl;
    } else {
        std::cout << name << " is wrong!" << std::endl;
    }

    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    free(h_A);
    free(h_B);
    free(h_C);
    free(d2h_C);
    cublasDestroy(handle);
}

float testCudaGemmPerformance(cublasGemmFunc gemm, int M, int N, int K, int repeat = 1)
{
    float *d_A, *d_B, *d_C;
    int floatSize = sizeof(float);
    cudaMalloc(&d_A, M * K * floatSize);
    cudaMalloc(&d_B, K * N * floatSize);
    cudaMalloc(&d_C, M * N * floatSize);

    cublasHandle_t cublas_handle;
    cublasCreate_v2(&cublas_handle);
    float cublas_alpha = 1.0;
    float cublas_beta = 0;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    for (int i = 0; i < repeat; ++i) {
        gemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K,
            &cublas_alpha,
            d_A, K,
            d_B, N,
            &cublas_beta,
            d_C, N);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000 / repeat;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(cublas_handle);
    return sec;
}

void cublasSgemm_void(cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float* alpha,
    const float* A,
    int lda,
    const float* B,
    int ldb,
    const float* beta,
    float* C,
    int ldc)
{
    cublasSgemm(handle, transa, transb, n, m, k, alpha, B, n, A, k, beta, C, n);
}

double testGemmFunc(cublasGemmFunc gemm, int M, int N, int K)
{
    auto sec = testCudaGemmPerformance(gemm, M, N, K);
    return ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / sec;
}

double testGemm(cublasGemmFunc gemm, int M, int N, int K, bool log = true)
{
    auto gflops = testGemmFunc(gemm, M, N, K);
    if (log)
        printf("%d x %d x %d => %12.3f \n", M, N, K, gflops);
    return gflops;
}

__global__ void sgemm_v1(int m, int n, int k, const float alpha, const float* A, int lda, const float* B, int ldb, const float beta, float* C, int ldc)
{
    int t_row = blockIdx.x * blockDim.x + threadIdx.x;
    int t_col = blockIdx.y * blockDim.y + threadIdx.y;

    if (t_row < m && t_col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum = sum + A[t_row * lda + i] * B[i * ldb + t_col];
        }
        C[t_row * ldc + t_col] = alpha * sum + beta * C[t_row * ldc + t_col];
    }
}

void mySgemm_v1(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float* alpha,
    const float* A,
    int lda,
    const float* B,
    int ldb,
    const float* beta,
    float* C,
    int ldc)
{
    dim3 blocks(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grids(CEIL_DIV(m, blocks.x), CEIL_DIV(n, blocks.y));

    sgemm_v1<<<grids, blocks>>>(m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);

    cudaDeviceSynchronize();
}

__global__ void sgemm_v2(int m, int n, int k, const float alpha, const float* A, int lda, const float* B, int ldb, const float beta, float* C, int ldc)
{
    const uint tx = threadIdx.x + blockDim.x * blockIdx.x;
    const uint ty = threadIdx.y + blockDim.y * blockIdx.y;

    if (ty < m && tx < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum = sum + A[ty * lda + i] * B[i * ldb + tx];
        }
        C[ty * ldc + tx] = alpha * sum + beta * C[ty * ldc + tx];
    }
}

void mySgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float* alpha,
    const float* A,
    int lda,
    const float* B,
    int ldb,
    const float* beta,
    float* C,
    int ldc)
{
    dim3 blocks(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grids(CEIL_DIV(n, blocks.x), CEIL_DIV(m, blocks.y));

    sgemm_v2<<<grids, blocks>>>(m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);

    cudaDeviceSynchronize();
}

__global__ void sgemm_v3(int m, int n, int k, const float alpha, const float* A, int lda, const float* B, int ldb, const float beta, float* C, int ldc)
{
    __shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE * BLOCK_SIZE];

    const uint bx = blockIdx.x * blockDim.x;
    const uint by = blockIdx.y * blockDim.y;

    const uint tx = threadIdx.x;
    const uint ty = threadIdx.y;

    A += by * lda;
    B += bx;
    C += by * ldc + bx;

    float tmp = 0.0;
    for (int bk = 0; bk < k; bk += BLOCK_SIZE) {
        As[ty * BLOCK_SIZE + tx] = A[lda * ty + tx];
        Bs[ty * BLOCK_SIZE + tx] = B[ldb * ty + tx];

        __syncthreads();

        for (int ki = 0; ki < BLOCK_SIZE; ++ki) {
            tmp += As[ty * BLOCK_SIZE + ki] * Bs[ki * BLOCK_SIZE + tx];
        }

        __syncthreads();

        A += BLOCK_SIZE;
        B += BLOCK_SIZE * ldb;
    }

    C[ldc * ty + tx] = alpha * tmp + beta * C[ldc * ty + tx];
}

void mySgemm_v3(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float* alpha,
    const float* A,
    int lda,
    const float* B,
    int ldb,
    const float* beta,
    float* C,
    int ldc)
{
    dim3 blocks(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grids(CEIL_DIV(n, blocks.x), CEIL_DIV(m, blocks.y));

    sgemm_v3<<<grids, blocks>>>(m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);

    cudaDeviceSynchronize();
}

double test(cublasGemmFunc gemm, std::string name)
{
    int len = 3;
    int arr[] = { 512, 1024, 2048, 4096, 8192 };
    checkOk(gemm, name);

    double sum_gflpos = 0;
    for (int i = 0; i < len; ++i) {
        for (int j = 0; j < len; ++j) {
            for (int k = 0; k < len; ++k) {
                sum_gflpos += testGemm(gemm, arr[i], arr[j], arr[k], false);
            }
        }
    }

    sum_gflpos = sum_gflpos / len / len / len;
    printf("%s, Average GFLOPS: %12.3f\n", name.c_str(), sum_gflpos);
    return sum_gflpos;
}

#include <thread>

int main()
{
    auto cublas_gflopos = test(cublasSgemm_void, "cublasSgemm_void");

    auto v1_gflopos = test(mySgemm_v1, "mySgemm_v1");
    printf("V1 Achieve: %12.3f\n", v1_gflopos / cublas_gflopos);

    auto v2_gflopos = test(mySgemm_v2, "mySgemm_v2");
    printf("V2 Achieve: %12.3f\n", v2_gflopos / cublas_gflopos);

    auto v3_gflopos = test(mySgemm_v3, "mySgemm_v3");
    printf("V3 Achieve: %12.3f\n", v3_gflopos / cublas_gflopos);
    return 0;
}