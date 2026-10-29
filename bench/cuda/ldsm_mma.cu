// IMMA fed by ldmatrix from smem: throughput vs IMMA per LDSM.x4 and warps per SM.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#define ITERS 1024
__device__ __forceinline__ uint32_t su(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
template <int MT>  // m-tiles sharing each B fragment: IMMA per LDSM.x4 = 2*MT
__global__ void __launch_bounds__(256, 1) k(int* out, int seed) {
  __shared__ __align__(128) uint8_t sm[16384];
  for (int i = threadIdx.x; i < 4096; i += blockDim.x) ((int*)sm)[i] = i * 2654435761u + seed;
  __syncthreads();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  uint32_t a[MT][4];
  for (int m = 0; m < MT; ++m) for (int j = 0; j < 4; ++j) a[m][j] = (lane + 1) * (m + 3) * (j + 7) * 0x9E3779B9u;
  int acc[MT][8][4] = {};
  for (int it = 0; it < ITERS; ++it) {
#pragma unroll
    for (int nb = 0; nb < 8; nb += 2) {
      uint32_t b0, b1, b2, b3;
      const int row = (nb + ((lane >> 3) >> 1)) * 8 + (lane & 7), ch = (it & 3) * 2 + ((lane >> 3) & 1);
      const uint8_t* p = sm + warp * 2048 % 8192 + row * 128 + ((ch ^ (row & 7)) << 4);
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(su(p)));
#pragma unroll
      for (int m = 0; m < MT; ++m) {
        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
            : "+r"(acc[m][nb][0]), "+r"(acc[m][nb][1]), "+r"(acc[m][nb][2]), "+r"(acc[m][nb][3])
            : "r"(a[m][0]), "r"(a[m][1]), "r"(a[m][2]), "r"(a[m][3]), "r"(b0), "r"(b1));
        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
            : "+r"(acc[m][nb + 1][0]), "+r"(acc[m][nb + 1][1]), "+r"(acc[m][nb + 1][2]), "+r"(acc[m][nb + 1][3])
            : "r"(a[m][0]), "r"(a[m][1]), "r"(a[m][2]), "r"(a[m][3]), "r"(b2), "r"(b3));
      }
    }
  }
  int s = 0;
  for (int m = 0; m < MT; ++m) for (int j = 0; j < 8; ++j) for (int i = 0; i < 4; ++i) s += acc[m][j][i];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
template <int MT>
void run(int* d, int warps) {
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  k<MT><<<48, 32 * warps>>>(d, 1);
  cudaEventRecord(e0); k<MT><<<48, 32 * warps>>>(d, 1); cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms; cudaEventElapsedTime(&ms, e0, e1);
  printf("MT=%d (IMMA per LDSM.x4 = %d) warps/SM=%d: %.1f TFLOPS\n", MT, 2 * MT, warps,
         48.0 * warps * ITERS * 8 * MT * 8192.0 / ms / 1e9);
}
int main() {
  int* d; cudaMalloc(&d, 1 << 22);
  for (int w : {4, 8}) { run<1>(d, w); run<2>(d, w); run<4>(d, w); }
}
