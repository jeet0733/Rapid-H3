// rr_row.cuh: copies of attn_int8.cu's per-row q/k helpers for the qkv GEMM epilogue (fp8_rowwise.cu,
// VisitorQkvPrep, config 3xx): comfy-kitchen's RMSNorm + split-half RoPE (rms_rope_row), the normalized
// Hadamard-128 rotation and the per-row INT8 quantization of quant_qk_rr. They must stay op-for-op equal to
// attn_int8.cu's versions (the outputs are checked with torch.equal: bench/qkv_prep.py, config 4xx).
#pragma once
#include <cuda_bf16.h>
#include <cstdint>

namespace rapidh3_rr {

constexpr int D = 128;
constexpr float MAGIC_F = 12582912.f;

__device__ __forceinline__ float mul_ftz(float a, float b) { float r; asm("mul.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float add_ftz(float a, float b) { float r; asm("add.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float fma_ftz(float a, float b, float c) {
  float r; asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c)); return r;
}
__device__ __forceinline__ float fma_ftz_off(float a, float b, float c) {  // fma.rn without ftz (hadamard128 is non-ftz)
  float r; asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c)); return r;
}
__device__ __forceinline__ float div_approx_ftz(float a, float b) { float r; asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float rsqrt_approx_ftz(float a) { float r; asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }
__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ void bf4(uint2 raw, float* x) {
  float2 a = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&raw.x));
  float2 b = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&raw.y));
  x[0] = a.x; x[1] = a.y; x[2] = b.x; x[3] = b.y;
}

template <int RD>
struct RRWeights {
  float wa[4], wb[4];
  __device__ __forceinline__ void load(const __nv_bfloat16* __restrict__ w, int lane) {
    constexpr int HR = RD / 2;
    const int c = lane * 4;
    const int p = c < HR ? c : (c < RD ? c - HR : c);
    bf4(*reinterpret_cast<const uint2*>(w + p), wa);
    if (c < RD) bf4(*reinterpret_cast<const uint2*>(w + p + HR), wb);
  }
};

// attn_int8.cu's rms_rope_row; src and sm may be the same (shared-memory) row.
template <int RD>
__device__ __forceinline__ void rms_rope_row(const __nv_bfloat16* src, __nv_bfloat16* sm, const RRWeights<RD>& w,
                                             const __nv_bfloat16* __restrict__ tab, float eps, int lane, float* out) {
  constexpr int HR = RD / 2;
  static_assert(HR % 4 == 0 && RD <= D, "rot_dim must be a multiple of 8, <= 128");
  const uint2 raw = *reinterpret_cast<const uint2*>(src + lane * 4);
  __syncwarp();
  *reinterpret_cast<uint2*>(sm + lane * 4) = raw;
  __syncwarp();
  float acc = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float v = __bfloat162float(sm[lane + 32 * j]);
    acc = fma_ftz(v, v, acc);
  }
#pragma unroll
  for (int off = 16; off; off >>= 1) acc = add_ftz(acc, __shfl_down_sync(0xffffffff, acc, off));
  acc = __shfl_sync(0xffffffff, acc, 0);
  const float r = rsqrt_approx_ftz(add_ftz(eps, div_approx_ftz(acc, (float)D)));
  const int c = lane * 4;
  if (c < RD) {
    const int p = c < HR ? c : c - HR;
    float xa[4], xb[4];
    bf4(*reinterpret_cast<const uint2*>(sm + p), xa);
    bf4(*reinterpret_cast<const uint2*>(sm + p + HR), xb);
    alignas(16) __nv_bfloat16 t[16];
    *reinterpret_cast<uint4*>(t) = *reinterpret_cast<const uint4*>(tab + p * 4);
    *reinterpret_cast<uint4*>(t + 8) = *reinterpret_cast<const uint4*>(tab + p * 4 + 8);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float a = bf16r(mul_ftz(mul_ftz(r, xa[i]), w.wa[i]));
      const float b = bf16r(mul_ftz(mul_ftz(r, xb[i]), w.wb[i]));
      const float t00 = __bfloat162float(t[4 * i]), t01 = __bfloat162float(t[4 * i + 1]);
      const float t10 = __bfloat162float(t[4 * i + 2]), t11 = __bfloat162float(t[4 * i + 3]);
      out[i] = c < HR ? bf16r(fma_ftz(a, t00, mul_ftz(b, t01))) : bf16r(fma_ftz(a, t10, mul_ftz(b, t11)));
    }
  } else {
    float x[4];
    bf4(raw, x);
#pragma unroll
    for (int i = 0; i < 4; ++i) out[i] = bf16r(mul_ftz(mul_ftz(r, x[i]), w.wa[i]));
  }
}

// rms_rope_row on R rows at once (independent shuffle chains interleave), rows already in shared memory
// (row[i] = 128 bf16, no scratch copy needed), tab[i] = that row's table row. Same ops per row as rms_rope_row.
template <int RD, int R>
__device__ __forceinline__ void rms_rope_rows(const __nv_bfloat16* const* row, const RRWeights<RD>& w,
                                              const __nv_bfloat16* const* tab, float eps, int lane, float (*out)[4]) {
  constexpr int HR = RD / 2;
  const int c = lane * 4;
  const int p = c < HR ? c : c - HR;
  alignas(16) __nv_bfloat16 t[R][16];
  if (c < RD) {
#pragma unroll
    for (int i = 0; i < R; ++i) {
      *reinterpret_cast<uint4*>(t[i]) = *reinterpret_cast<const uint4*>(tab[i] + p * 4);
      *reinterpret_cast<uint4*>(t[i] + 8) = *reinterpret_cast<const uint4*>(tab[i] + p * 4 + 8);
    }
  }
  float acc[R];
#pragma unroll
  for (int i = 0; i < R; ++i) {
    acc[i] = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float v = __bfloat162float(row[i][lane + 32 * j]);
      acc[i] = fma_ftz(v, v, acc[i]);
    }
  }
#pragma unroll
  for (int off = 16; off; off >>= 1)
#pragma unroll
    for (int i = 0; i < R; ++i) acc[i] = add_ftz(acc[i], __shfl_down_sync(0xffffffff, acc[i], off));
#pragma unroll
  for (int i = 0; i < R; ++i) {
    acc[i] = __shfl_sync(0xffffffff, acc[i], 0);
    const float r = rsqrt_approx_ftz(add_ftz(eps, div_approx_ftz(acc[i], (float)D)));
    if (c < RD) {
      float xa[4], xb[4];
      bf4(*reinterpret_cast<const uint2*>(row[i] + p), xa);
      bf4(*reinterpret_cast<const uint2*>(row[i] + p + HR), xb);
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const float a = bf16r(mul_ftz(mul_ftz(r, xa[k]), w.wa[k]));
        const float b = bf16r(mul_ftz(mul_ftz(r, xb[k]), w.wb[k]));
        const float t00 = __bfloat162float(t[i][4 * k]), t01 = __bfloat162float(t[i][4 * k + 1]);
        const float t10 = __bfloat162float(t[i][4 * k + 2]), t11 = __bfloat162float(t[i][4 * k + 3]);
        out[i][k] = c < HR ? bf16r(fma_ftz(a, t00, mul_ftz(b, t01))) : bf16r(fma_ftz(a, t10, mul_ftz(b, t11)));
      }
    } else {
      float x[4];
      bf4(*reinterpret_cast<const uint2*>(row[i] + c), x);
#pragma unroll
      for (int k = 0; k < 4; ++k) out[i][k] = bf16r(mul_ftz(mul_ftz(r, x[k]), w.wa[k]));
    }
  }
}

__device__ __forceinline__ void hadamard128(float* x, int lane) {
  float a = x[0] + x[1], b = x[0] - x[1], c = x[2] + x[3], d = x[2] - x[3];
  x[0] = a + c; x[1] = b + d; x[2] = a - c; x[3] = b - d;
#pragma unroll
  for (int m = 1; m < 32; m <<= 1) {
    const bool hi = lane & m;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float y = __shfl_xor_sync(0xffffffff, x[i], m);
      x[i] = hi ? y - x[i] : x[i] + y;
    }
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) x[i] *= 0.08838834764831845f;  // 1/sqrt(128)
}

// quant_qk_rr's Q branch after rms_rope_row: rotate, per-row absmax scale, int8 (one warp, 4 values per lane)
__device__ __forceinline__ void quant_row(float* x, int lane, int rotate, int8_t* __restrict__ dst, float* __restrict__ sdst) {
  if (rotate) hadamard128(x, lane);
  float amax = fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fmaxf(fabsf(x[2]), fabsf(x[3])));
#pragma unroll
  for (int off = 16; off; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
  const float sc = fmaxf(amax, 1e-8f) / 127.f, inv = 1.f / sc;
  char4 qv;
  qv.x = (int8_t)__float2int_rn(x[0] * inv); qv.y = (int8_t)__float2int_rn(x[1] * inv);
  qv.z = (int8_t)__float2int_rn(x[2] * inv); qv.w = (int8_t)__float2int_rn(x[3] * inv);
  *reinterpret_cast<char4*>(dst + lane * 4) = qv;
  if (lane == 0) *sdst = sc;
}

// hadamard128 + quant_row on R rows at once (same ops per row; the R shuffle chains interleave)
template <int R>
__device__ __forceinline__ void quant_rows(float (*x)[4], int lane, int rotate, int8_t* const* dst, float* const* sdst) {
  if (rotate) {
#pragma unroll
    for (int i = 0; i < R; ++i) {
      float a = x[i][0] + x[i][1], b = x[i][0] - x[i][1], c = x[i][2] + x[i][3], d = x[i][2] - x[i][3];
      x[i][0] = a + c; x[i][1] = b + d; x[i][2] = a - c; x[i][3] = b - d;
    }
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
      // hi ? y - x : x + y as one exact fma: fma(-1, x, y) = round(y - x), fma(1, x, y) = round(x + y), signed zeros
      // included, so the result equals hadamard128's select bit for bit
      const float sg = (lane & m) ? -1.f : 1.f;
#pragma unroll
      for (int i = 0; i < R; ++i)
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const float y = __shfl_xor_sync(0xffffffff, x[i][k], m);
          x[i][k] = fma_ftz_off(sg, x[i][k], y);
        }
    }
#pragma unroll
    for (int i = 0; i < R; ++i)
#pragma unroll
      for (int k = 0; k < 4; ++k) x[i][k] *= 0.08838834764831845f;
  }
  float amax[R];
#pragma unroll
  for (int i = 0; i < R; ++i) amax[i] = fmaxf(fmaxf(fabsf(x[i][0]), fabsf(x[i][1])), fmaxf(fabsf(x[i][2]), fabsf(x[i][3])));
#pragma unroll
  for (int off = 16; off; off >>= 1)
#pragma unroll
    for (int i = 0; i < R; ++i) amax[i] = fmaxf(amax[i], __shfl_xor_sync(0xffffffff, amax[i], off));
#pragma unroll
  for (int i = 0; i < R; ++i) {
    if (!dst[i]) continue;
    const float sc = fmaxf(amax[i], 1e-8f) / 127.f, inv = 1.f / sc;
    char4 qv;
    qv.x = (int8_t)__float2int_rn(x[i][0] * inv); qv.y = (int8_t)__float2int_rn(x[i][1] * inv);
    qv.z = (int8_t)__float2int_rn(x[i][2] * inv); qv.w = (int8_t)__float2int_rn(x[i][3] * inv);
    *reinterpret_cast<char4*>(dst[i] + lane * 4) = qv;
    if (lane == 0) *sdst[i] = sc;
  }
}

}  // namespace rapidh3_rr
