// IMMA m16n8k32 throughput vs warps per SMSP and independent accumulators per warp.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#define ITERS 2048
template <int NACC, bool TOGGLE = false>
__global__ void __launch_bounds__(1024, 1) loop(int* out, int seed) {
  uint32_t a0 = seed + threadIdx.x * 0x9E3779B9u, a1 = a0 * 0x85EBCA6Bu, a2 = a1 * 0xC2B2AE35u, a3 = a2 * 0x27D4EB2Fu,
           b0 = a3 * 0x165667B1u, b1 = b0 * 0x9E3779B9u;
  int acc[NACC][4] = {};
  for (int it = 0; it < ITERS; ++it) {
    if (TOGGLE) { a0 = ~a0; a1 = ~a1; a2 = ~a2; a3 = ~a3; b0 = __byte_perm(b0, b0, 0x0321); b1 = ~b1; }
#pragma unroll
    for (int j = 0; j < NACC; ++j)
      asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                   : "+r"(acc[j][0]), "+r"(acc[j][1]), "+r"(acc[j][2]), "+r"(acc[j][3])
                   : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  }
  int s = 0;
  for (int j = 0; j < NACC; ++j) for (int i = 0; i < 4; ++i) s += acc[j][i];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
template <int NACC, bool TOGGLE = false>
void run(int warps_per_sm, int sms, int* d) {
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  loop<NACC, TOGGLE><<<sms, 32 * warps_per_sm>>>(d, 1);
  cudaEventRecord(e0); loop<NACC, TOGGLE><<<sms, 32 * warps_per_sm>>>(d, 1); cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms; cudaEventElapsedTime(&ms, e0, e1);
  double fl = (double)sms * warps_per_sm * ITERS * NACC * 8192.0;
  printf("%s warps/SM=%2d (per SMSP %.1f) acc=%2d : %6.1f TFLOPS\n", TOGGLE ? "random+toggle" : "random      ", warps_per_sm, warps_per_sm / 4.0, NACC, fl / ms / 1e9);
}
int main() {
  int* d; cudaMalloc(&d, 1 << 24);
  int sms = 48;
  for (int w : {8, 16}) { run<8>(w, sms, d); run<8, true>(w, sms, d); }
  // sustained: 20 launches back-to-back of the toggling kernel
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0);
  for (int i = 0; i < 200; ++i) loop<8, true><<<sms, 256>>>(d, i);
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms; cudaEventElapsedTime(&ms, e0, e1);
  printf("sustained toggling 200 launches: %.1f TFLOPS\n", 200.0 * sms * 8 * ITERS * 8 * 8192.0 / ms / 1e9);
}
