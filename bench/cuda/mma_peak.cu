// Raw mma.sync / MUFU throughput on GB10 (register-only loops).
#include <cstdio>
#include <cuda_runtime.h>
#include <cstdint>

#define ITERS 4096
template <int KIND>
__global__ void mma_loop(int* out, int seed) {
  uint32_t a0 = seed + threadIdx.x, a1 = a0 * 3, a2 = a0 * 5, a3 = a0 * 7, b0 = a0 * 11, b1 = a0 * 13;
  int acc[8][4] = {};
  float facc[8][4] = {};
  for (int it = 0; it < ITERS; ++it) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      if (KIND == 0)
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                     : "+r"(acc[j][0]), "+r"(acc[j][1]), "+r"(acc[j][2]), "+r"(acc[j][3])
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      else if (KIND == 1)
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                     : "+r"(acc[j][0]), "+r"(acc[j][1]), "+r"(acc[j][2]), "+r"(acc[j][3])
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      else if (KIND == 2)
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                     : "+f"(facc[j][0]), "+f"(facc[j][1]), "+f"(facc[j][2]), "+f"(facc[j][3])
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      else if (KIND == 3)
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                     : "+f"(facc[j][0]), "+f"(facc[j][1]), "+f"(facc[j][2]), "+f"(facc[j][3])
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      else if (KIND == 4)
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1};\n"
                     : "+r"(acc[j][0]), "+r"(acc[j][1])
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
  }
  int s = 0;
  for (int j = 0; j < 8; ++j) for (int i = 0; i < 4; ++i) s += acc[j][i] + (int)facc[j][i];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

__global__ void ex2_loop(float* out, float seed) {
  float x[8];
  for (int j = 0; j < 8; ++j) x[j] = seed * (threadIdx.x + j) * 1e-6f;
  for (int it = 0; it < ITERS; ++it)
#pragma unroll
    for (int j = 0; j < 8; ++j) asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+f"(x[j]));
  float s = 0; for (int j = 0; j < 8; ++j) s += x[j];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

int main() {
  int* d; cudaMalloc(&d, 1 << 26);
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
  int sms = p.multiProcessorCount;
  printf("%s SMs=%d smemPerBlockOptin=%zu smemPerSM=%zu regsPerSM=%d\n", p.name, sms, p.sharedMemPerBlockOptin, p.sharedMemPerMultiprocessor, p.regsPerMultiprocessor);
  const char* names[] = {"s8.s8 m16n8k32", "u8.s8 m16n8k32", "e4m3 f32acc m16n8k32", "bf16 f32acc m16n8k16", "e4m3 f16acc m16n8k32"};
  double flops_per[] = {16*8*32*2, 16*8*32*2, 16*8*32*2, 16*8*16*2, 16*8*32*2};
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  for (int warps : {4, 8, 16}) {
    for (int k = 0; k < 5; ++k) {
      auto launch = [&] {
        dim3 g(sms * 8), b(32 * warps);
        if (k == 0) mma_loop<0><<<g, b>>>(d, 1);
        if (k == 1) mma_loop<1><<<g, b>>>(d, 1);
        if (k == 2) mma_loop<2><<<g, b>>>(d, 1);
        if (k == 3) mma_loop<3><<<g, b>>>(d, 1);
        if (k == 4) mma_loop<4><<<g, b>>>(d, 1);
      };
      launch(); cudaEventRecord(e0); launch(); cudaEventRecord(e1); cudaEventSynchronize(e1);
      float ms; cudaEventElapsedTime(&ms, e0, e1);
      double fl = (double)sms * 8 * warps * ITERS * 8 * flops_per[k];
      printf("warps/blk=%2d %-22s %7.1f TFLOPS\n", warps, names[k], fl / ms / 1e9);
    }
  }
  {
    dim3 g(sms * 8), b(256);
    ex2_loop<<<g, b>>>((float*)d, 1.f); cudaEventRecord(e0); ex2_loop<<<g, b>>>((float*)d, 1.f); cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    double n = (double)sms * 8 * 256 * ITERS * 8;
    printf("ex2: %.1f Gop/s = %.1f per clk per SM @2.4GHz\n", n / ms / 1e6, n / ms / 1e6 / sms / 2.4);
  }
  printf("err=%s\n", cudaGetErrorString(cudaGetLastError()));
}
