// IMMA m16n8k32 throughput vs resident warps per SM (one CTA per SM, register-only), and the cost of
// non-IMMA instructions mixed into the stream. Tells how many warps per sub-partition the tensor pipe
// needs to stay full, given that ptxas pads back-to-back IMMAs from one warp with a NOP.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#define ITERS 2048
template <int NACC, int EXTRA>  // NACC independent accumulators; EXTRA FFMAs after each IMMA
__global__ void __launch_bounds__(512, 1) k(int* out, int seed) {
  __shared__ int pad[24 * 1024];  // force one CTA per SM
  uint32_t a0 = seed + threadIdx.x, a1 = a0 * 3, a2 = a0 * 5, a3 = a0 * 7, b0 = a0 * 11, b1 = a0 * 13;
  int acc[NACC][4] = {};
  float f[8];
  for (int i = 0; i < 8; ++i) f[i] = a0 * 1e-9f;
  for (int it = 0; it < ITERS; ++it) {
#pragma unroll
    for (int j = 0; j < NACC; ++j) {
      asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                   : "+r"(acc[j][0]), "+r"(acc[j][1]), "+r"(acc[j][2]), "+r"(acc[j][3])
                   : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#pragma unroll
      for (int e = 0; e < EXTRA; ++e) asm volatile("fma.rn.f32 %0, %0, %0, %0;" : "+f"(f[(j * EXTRA + e) & 7]));
    }
  }
  int s = 0;
  for (int j = 0; j < NACC; ++j) for (int i = 0; i < 4; ++i) s += acc[j][i];
  for (int i = 0; i < 8; ++i) s += (int)f[i];
  if (seed == 12345) pad[threadIdx.x] = s;
  out[blockIdx.x * blockDim.x + threadIdx.x] = s + (seed == 7 ? pad[threadIdx.x ^ 1] : 0);
}
template <int NACC, int EXTRA>
void run(int* d, int sms, int warps) {
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  k<NACC, EXTRA><<<sms, 32 * warps>>>(d, 1);
  cudaEventRecord(e0); k<NACC, EXTRA><<<sms, 32 * warps>>>(d, 1); cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms; cudaEventElapsedTime(&ms, e0, e1);
  printf("acc=%2d extra=%d warps/SM=%2d: %6.1f TFLOPS\n", NACC, EXTRA, warps,
         (double)sms * warps * ITERS * NACC * 8192.0 / ms / 1e9);
}
int main() {
  int* d; cudaMalloc(&d, 1 << 24);
  int sms; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  for (int w : {4, 8, 12, 16}) { run<8, 0>(d, sms, w); run<16, 0>(d, sms, w); }
  for (int w : {8, 16}) { run<8, 1>(d, sms, w); run<8, 2>(d, sms, w); run<8, 4>(d, sms, w); run<8, 8>(d, sms, w); }
  printf("err=%s\n", cudaGetErrorString(cudaGetLastError()));
}
