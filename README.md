# V1: Naive Implementation
```c++
__global__ void sgemm_v1(int m, int n, int k, const float alpha, const float* A, int lda, const float* B, int ldb, const float beta, float* C, int ldc)
{
    int threadRow = blockIdx.x * blockDim.x + threadIdx.x;
    int threadCol = blockIdx.y * blockDim.y + threadIdx.y;

    if (threadRow < m && threadCol < n) {
        float s = 0.0f;
        for (int i = 0; i < k; ++i) {
            s += A[threadRow * lda + i] * B[i * ldb + threadCol];
        }
        C[threadRow * ldc + threadCol] = alpha * s + beta * C[threadRow * ldc + threadCol];
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
    dim3 blocks(32, 32);
    dim3 grids(CEIL_DIV(m, blocks.x), CEIL_DIV(n, blocks.y));

    sgemm_v1<<<grids, blocks>>>(m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);

    cudaDeviceSynchronize();
}
```