// INT8 flash attention for GB10 (sm_121): non-causal, head_dim 128, batch 1, mma.sync only.
//
//   S = Q·Kᵀ   int8 x int8 -> int32   Q per-token scale, K per-token scale (K optionally mean-smoothed)
//   P = exp(S) u8 (0..255, relative to each 128-key tile's own row max)
//   O = P·V    u8 x int8 -> int32     V per-channel scale (optionally mean-smoothed), stored transposed
//
// Each tile's int32 P·V accumulator is folded into the fp32 output with a per-tile scale
// ts = 2^(tile max - running max) (quantizing P against the running max instead gave ≈4% error; the FP8 P·V
// configs 50-56 do use the running max). Row sums are taken from the quantized P, so numerator and denominator
// see the same weights. Configs and results: README.md; configs 94-99 / 90 / 89 / 79 are in attn_cake/attn_cake.cuh,
// and ts_h2 / ts_s6 run config 190 (= 90 with the q/k RMSNorm + RoPE fused into the pre-pass, int8_attention_rr).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace {

constexpr int D = 128;
constexpr int MAGIC_I = 0x4B400000;  // 1.5 * 2^23
constexpr float MAGIC_F = 12582912.f;
constexpr float LOG2_255 = 7.99435343685886f;  // 2^LOG2_255 = 255

// exact int -> float for |x| < 2^22 on the FMA pipe
__device__ __forceinline__ float i2f(int x) { return __int_as_float(x + MAGIC_I) - MAGIC_F; }

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_u32(dst)), "l"(src));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma_s8(int* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_u8s8(int* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_u8s8_c(int* d, const uint32_t* a, uint32_t b0, uint32_t b1, int c) {
  asm("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%10,%10,%10};\n"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(c));
}
__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
// low bytes of four ints -> one packed u32
__device__ __forceinline__ uint32_t pack4(int a, int b, int c, int d) {
  return __byte_perm(__byte_perm(a, b, 0x0040), __byte_perm(c, d, 0x0040), 0x5410);
}

// Row-major smem tile with RB bytes per row, 16 B chunks XOR-swizzled so 8 consecutive rows of
// one chunk column hit 8 distinct bank groups (ldmatrix conflict-free).
template <int RB>
__device__ __forceinline__ int swz(int row, int chunk) {
  if constexpr (RB == 128) return row * 128 + ((chunk ^ (row & 7)) << 4);
  else if constexpr (RB == 64) return row * 64 + ((chunk ^ ((row >> 1) & 3)) << 4);
  else return row * RB + (chunk << 4);
}

// ------------------------------------------------------------------------------------------------
// prep: statistics (deterministic two-level reduction), Q/K quantization, V quantize+transpose
// ------------------------------------------------------------------------------------------------
constexpr int STAT_CHUNK = 256;

__global__ void stats_partial(const __nv_bfloat16* __restrict__ K, const __nv_bfloat16* __restrict__ V,
                              long k_ss, long k_sh, long v_ss, long v_sh, int S, int nchunk,
                              float* __restrict__ part) {  // part [H][nchunk][4][D]
  const int h = blockIdx.y, c = blockIdx.x, d = threadIdx.x;
  const int s0 = c * STAT_CHUNK, s1 = min(S, s0 + STAT_CHUNK);
  float ks = 0.f, vs = 0.f, vmx = -INFINITY, vmn = INFINITY;
  for (int s = s0; s < s1; ++s) {
    float kv = __bfloat162float(K[s * k_ss + h * k_sh + d]);
    float vv = __bfloat162float(V[s * v_ss + h * v_sh + d]);
    ks += kv; vs += vv; vmx = fmaxf(vmx, vv); vmn = fminf(vmn, vv);
  }
  float* p = part + ((long)h * nchunk + c) * 4 * D;
  p[d] = ks; p[D + d] = vs; p[2 * D + d] = vmx; p[3 * D + d] = vmn;
}

__global__ void stats_final(const float* __restrict__ part, int nchunk, int S, int smooth_k, int smooth_v,
                            float* __restrict__ kmean, float* __restrict__ vmean, float* __restrict__ vscale) {
  const int h = blockIdx.x, d = threadIdx.x;
  float ks = 0.f, vs = 0.f, vmx = -INFINITY, vmn = INFINITY;
  for (int c = 0; c < nchunk; ++c) {
    const float* p = part + ((long)h * nchunk + c) * 4 * D;
    ks += p[d]; vs += p[D + d]; vmx = fmaxf(vmx, p[2 * D + d]); vmn = fminf(vmn, p[3 * D + d]);
  }
  float km = smooth_k ? ks / S : 0.f, vm = smooth_v ? vs / S : 0.f;
  kmean[h * D + d] = km;
  vmean[h * D + d] = vm;
  vscale[h * D + d] = fmaxf(fmaxf(vmx - vm, vm - vmn), 1e-6f) / 127.f;
}

// in-place normalized Walsh-Hadamard transform of a 128-vector held as 4 consecutive values per lane.
// Q and K both rotated leaves Q.K^T unchanged but spreads outlier channels before int8 rounding.
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

// one warp per (token, head) row: Q and K int8 with a per-row scale; rows in [S, Spad) are zeroed
__global__ void quant_qk(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
                         long q_ss, long q_sh, long k_ss, long k_sh, int S, int Spad,
                         const float* __restrict__ kmean, int8_t* __restrict__ Qo, float* __restrict__ qs,
                         int8_t* __restrict__ Ko, float* __restrict__ ks, float2* __restrict__ ksb, int rotate, int tiled) {
  const int h = blockIdx.y, lane = threadIdx.x & 31;
  const int s = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
  if (s >= Spad) return;
  const long o = ((long)h * Spad + s) * D + lane * 4;
  if (s >= S) {
    *reinterpret_cast<int*>(Qo + o) = 0;
    *reinterpret_cast<int*>(Ko + o) = 0;
    if (lane == 0) { qs[h * Spad + s] = 0.f; ks[h * Spad + s] = 0.f; ksb[h * Spad + s] = make_float2(0.f, -INFINITY); }
    return;
  }
#pragma unroll
  for (int which = 0; which < 2; ++which) {
    const __nv_bfloat16* src = which ? K + s * k_ss + h * k_sh : Q + s * q_ss + h * q_sh;
    uint2 raw = *reinterpret_cast<const uint2*>(src + lane * 4);
    float2 a = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&raw.x));
    float2 b = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&raw.y));
    float x[4] = {a.x, a.y, b.x, b.y};
    if (which) {
      float4 m = *reinterpret_cast<const float4*>(kmean + h * D + lane * 4);
      x[0] -= m.x; x[1] -= m.y; x[2] -= m.z; x[3] -= m.w;
    }
    if (rotate) hadamard128(x, lane);
    float amax = fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fmaxf(fabsf(x[2]), fabsf(x[3])));
#pragma unroll
    for (int off = 16; off; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
    const float sc = fmaxf(amax, 1e-8f) / 127.f, inv = 1.f / sc;
    char4 qv;
    qv.x = (int8_t)__float2int_rn(x[0] * inv); qv.y = (int8_t)__float2int_rn(x[1] * inv);
    qv.z = (int8_t)__float2int_rn(x[2] * inv); qv.w = (int8_t)__float2int_rn(x[3] * inv);
    // tiled: K rows stored with their 16 B chunks pre-swizzled (chunk c -> c ^ (row & 7)), i.e. the
    // smem image of a tile, so one bulk copy per tile lands conflict-free for ldmatrix
    const long ok = tiled && which ? ((long)h * Spad + s) * D + ((((lane >> 2) ^ (s & 7)) << 4) | ((lane & 3) << 2)) : o;
    *reinterpret_cast<char4*>((which ? Ko : Qo) + ok) = qv;
    if (lane == 0) (which ? ks : qs)[h * Spad + s] = sc;
    if (which && lane == 0) ksb[h * Spad + s] = make_float2(sc, -MAGIC_F * sc);
  }
}

// ------------------------------------------------------------------------------------------------
// per-head q/k RMSNorm + split-half partial RoPE fused into the pre-pass.
// rms_rope_row reproduces comfy-kitchen 0.2.37's (unchanged in 0.2.39) rope_kernel<bf16,bf16,bf16,1,1,1,1,1> (the in-place
// `rms_rope_split_half_` that ComfyUI's MiniMax H3 block calls) bit for bit; its PTX (sm_120f) was read:
//   * sum of squares: lane l accumulates fma(x,x,acc) over channels l, l+32, l+64, l+96 (from 0), then a
//     shfl.down 16/8/4/2/1 add tree, lane 0 broadcast;
//   * r = rsqrt.approx.ftz(eps + div.approx.ftz(sum, 128));
//   * y = bf16((r * x) * w)  (two mul.ftz, one rounding), for both halves of a rotary pair and for the
//     pass-through channels [rot, 128);
//   * rotary pair (i, i + rot/2), table row [rot/2][2][2] bf16 = (t00, t01, t10, t11):
//     out_i = bf16(fma(a, t00, b * t01)), out_{i+rot/2} = bf16(fma(a, t10, b * t11)), a = y_i, b = y_{i+rot/2}.
// All fp32 ops are the .ftz forms ck was built with (fast-math), so they are issued as inline PTX here.
// The result is the bf16 value ck would have written in place, returned as 4 floats for channels 4*lane..+3,
// i.e. exactly what quant_qk / stats_partial read back from the buffer today.
// ------------------------------------------------------------------------------------------------
__device__ __forceinline__ float mul_ftz(float a, float b) { float r; asm("mul.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float add_ftz(float a, float b) { float r; asm("add.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float fma_ftz(float a, float b, float c) {
  float r; asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c)); return r;
}
__device__ __forceinline__ float div_approx_ftz(float a, float b) { float r; asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float rsqrt_approx_ftz(float a) { float r; asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }
__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ void bf4(uint2 raw, float* x) {
  float2 a = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&raw.x));
  float2 b = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&raw.y));
  x[0] = a.x; x[1] = a.y; x[2] = b.x; x[3] = b.y;
}

// Per-lane constants of one norm weight for rot_dim RD (=96 for H3): wa = w at this lane's 4 channels (or at the
// first half of their rotary pairs), wb = w at the second half of the pairs.
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

// One warp, one (token, head) row of 128 bf16 at src: returns ck's RMSNorm+RoPE output for channels 4*lane..+3.
// sm: this warp's 128-element smem scratch. tab: this token's [RD/2][2][2] bf16 rotation table row.
template <int RD>
__device__ __forceinline__ void rms_rope_row(const __nv_bfloat16* __restrict__ src, __nv_bfloat16* sm,
                                             const RRWeights<RD>& w, const __nv_bfloat16* __restrict__ tab,
                                             float eps, int lane, float* out) {
  constexpr int HR = RD / 2;
  static_assert(HR % 4 == 0 && RD <= D, "rot_dim must be a multiple of 8, <= 128");
  const uint2 raw = *reinterpret_cast<const uint2*>(src + lane * 4);
  __syncwarp();  // previous row's readers are done with sm
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

// stats_partial with K = rms_rope(K): same per-channel summation order (tokens in order within a 256-token
// chunk, from 0), so stats_final gives bit-identical kmean/vmean/vscale. One warp per (head, chunk); a block's
// 8 warps are 8 heads of one chunk, so the chunk's rope table rows are shared through L1/L2.
template <int RD>
__global__ void __launch_bounds__(256) stats_partial_rr(
    const __nv_bfloat16* __restrict__ K, const __nv_bfloat16* __restrict__ V, long k_ss, long k_sh, long v_ss,
    long v_sh, int S, int H, int nchunk, const __nv_bfloat16* __restrict__ tab, const __nv_bfloat16* __restrict__ kw,
    float eps, float* __restrict__ part) {
  __shared__ __align__(16) __nv_bfloat16 smem[8][D];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int h = blockIdx.x * 8 + warp, c = blockIdx.y;
  if (h >= H) return;
  RRWeights<RD> w;
  w.load(kw, lane);
  const int s0 = c * STAT_CHUNK, s1 = min(S, s0 + STAT_CHUNK);
  float ks[4] = {0.f, 0.f, 0.f, 0.f}, vs[4] = {0.f, 0.f, 0.f, 0.f};
  float vmx[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY}, vmn[4] = {INFINITY, INFINITY, INFINITY, INFINITY};
  for (int s = s0; s < s1; ++s) {
    float kv[4], vv[4];
    bf4(*reinterpret_cast<const uint2*>(V + s * v_ss + h * v_sh + lane * 4), vv);
    rms_rope_row<RD>(K + s * k_ss + h * k_sh, smem[warp], w, tab + (long)s * (RD * 2), eps, lane, kv);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      ks[i] += kv[i]; vs[i] += vv[i]; vmx[i] = fmaxf(vmx[i], vv[i]); vmn[i] = fminf(vmn[i], vv[i]);
    }
  }
  float* p = part + ((long)h * nchunk + c) * 4 * D + lane * 4;
  *reinterpret_cast<float4*>(p) = make_float4(ks[0], ks[1], ks[2], ks[3]);
  *reinterpret_cast<float4*>(p + D) = make_float4(vs[0], vs[1], vs[2], vs[3]);
  *reinterpret_cast<float4*>(p + 2 * D) = make_float4(vmx[0], vmx[1], vmx[2], vmx[3]);
  *reinterpret_cast<float4*>(p + 3 * D) = make_float4(vmn[0], vmn[1], vmn[2], vmn[3]);
}

// quant_qk on raw (pre-norm) q/k: rms_rope_row, then exactly quant_qk's math. One warp per (token, head),
// heads fastest (contiguous in the qkv buffer; one token's table row serves all its heads).
template <int RD>
__global__ void __launch_bounds__(256) quant_qk_rr(
    const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K, long q_ss, long q_sh, long k_ss,
    long k_sh, int S, int Spad, int H, const __nv_bfloat16* __restrict__ tab, const __nv_bfloat16* __restrict__ qw,
    const __nv_bfloat16* __restrict__ kw, float eps, const float* __restrict__ kmean, int8_t* __restrict__ Qo,
    float* __restrict__ qs, int8_t* __restrict__ Ko, float* __restrict__ ks, float2* __restrict__ ksb, int rotate,
    int tiled) {
  __shared__ __align__(16) __nv_bfloat16 smem[8][D];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const long g = (long)blockIdx.x * 8 + warp;
  const int s = (int)(g / H), h = (int)(g % H);
  if (s >= Spad) return;
  const long o = ((long)h * Spad + s) * D + lane * 4;
  if (s >= S) {
    *reinterpret_cast<int*>(Qo + o) = 0;
    *reinterpret_cast<int*>(Ko + o) = 0;
    if (lane == 0) { qs[h * Spad + s] = 0.f; ks[h * Spad + s] = 0.f; ksb[h * Spad + s] = make_float2(0.f, -INFINITY); }
    return;
  }
  const __nv_bfloat16* trow = tab + (long)s * (RD * 2);
#pragma unroll
  for (int which = 0; which < 2; ++which) {
    RRWeights<RD> w;
    w.load(which ? kw : qw, lane);
    float x[4];
    rms_rope_row<RD>(which ? K + s * k_ss + h * k_sh : Q + s * q_ss + h * q_sh, smem[warp], w, trow, eps, lane, x);
    if (which) {
      float4 m = *reinterpret_cast<const float4*>(kmean + h * D + lane * 4);
      x[0] -= m.x; x[1] -= m.y; x[2] -= m.z; x[3] -= m.w;
    }
    if (rotate) hadamard128(x, lane);
    float amax = fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fmaxf(fabsf(x[2]), fabsf(x[3])));
#pragma unroll
    for (int off = 16; off; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
    const float sc = fmaxf(amax, 1e-8f) / 127.f, inv = 1.f / sc;
    char4 qv;
    qv.x = (int8_t)__float2int_rn(x[0] * inv); qv.y = (int8_t)__float2int_rn(x[1] * inv);
    qv.z = (int8_t)__float2int_rn(x[2] * inv); qv.w = (int8_t)__float2int_rn(x[3] * inv);
    const long ok = tiled && which ? ((long)h * Spad + s) * D + ((((lane >> 2) ^ (s & 7)) << 4) | ((lane & 3) << 2)) : o;
    *reinterpret_cast<char4*>((which ? Ko : Qo) + ok) = qv;
    if (lane == 0) (which ? ks : qs)[h * Spad + s] = sc;
    if (which && lane == 0) ksb[h * Spad + s] = make_float2(sc, -MAGIC_F * sc);
  }
}

// quant_qk_rr's K branch alone (config 3xx: q8/qs come from the qkv GEMM epilogue, fp8_rowwise.cu VisitorQkvPrep)
template <int RD>
__global__ void __launch_bounds__(256) quant_k_rr(
    const __nv_bfloat16* __restrict__ K, long k_ss, long k_sh, int S, int Spad, int H,
    const __nv_bfloat16* __restrict__ tab, const __nv_bfloat16* __restrict__ kw, float eps,
    const float* __restrict__ kmean, int8_t* __restrict__ Ko, float* __restrict__ ks, float2* __restrict__ ksb,
    int rotate, int tiled) {
  __shared__ __align__(16) __nv_bfloat16 smem[8][D];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const long g = (long)blockIdx.x * 8 + warp;
  const int s = (int)(g / H), h = (int)(g % H);
  if (s >= Spad) return;
  const long o = ((long)h * Spad + s) * D + lane * 4;
  if (s >= S) {
    *reinterpret_cast<int*>(Ko + o) = 0;
    if (lane == 0) { ks[h * Spad + s] = 0.f; ksb[h * Spad + s] = make_float2(0.f, -INFINITY); }
    return;
  }
  RRWeights<RD> w;
  w.load(kw, lane);
  float x[4];
  rms_rope_row<RD>(K + s * k_ss + h * k_sh, smem[warp], w, tab + (long)s * (RD * 2), eps, lane, x);
  {
    float4 m = *reinterpret_cast<const float4*>(kmean + h * D + lane * 4);
    x[0] -= m.x; x[1] -= m.y; x[2] -= m.z; x[3] -= m.w;
  }
  if (rotate) hadamard128(x, lane);
  float amax = fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fmaxf(fabsf(x[2]), fabsf(x[3])));
#pragma unroll
  for (int off = 16; off; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
  const float sc = fmaxf(amax, 1e-8f) / 127.f, inv = 1.f / sc;
  char4 qv;
  qv.x = (int8_t)__float2int_rn(x[0] * inv); qv.y = (int8_t)__float2int_rn(x[1] * inv);
  qv.z = (int8_t)__float2int_rn(x[2] * inv); qv.w = (int8_t)__float2int_rn(x[3] * inv);
  const long ok = tiled ? ((long)h * Spad + s) * D + ((((lane >> 2) ^ (s & 7)) << 4) | ((lane & 3) << 2)) : o;
  *reinterpret_cast<char4*>(Ko + ok) = qv;
  if (lane == 0) { ks[h * Spad + s] = sc; ksb[h * Spad + s] = make_float2(sc, -MAGIC_F * sc); }
}

// debug/verification: ck's in-place rms_rope_split_half_ result written out-of-place ([S, H, 128] bf16)
template <int RD>
__global__ void __launch_bounds__(256) rms_rope_only(
    const __nv_bfloat16* __restrict__ X, long x_ss, long x_sh, int S, int H, const __nv_bfloat16* __restrict__ tab,
    const __nv_bfloat16* __restrict__ w_, float eps, __nv_bfloat16* __restrict__ Y) {
  __shared__ __align__(16) __nv_bfloat16 smem[8][D];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const long g = (long)blockIdx.x * 8 + warp;
  const int s = (int)(g / H), h = (int)(g % H);
  if (s >= S) return;
  RRWeights<RD> w;
  w.load(w_, lane);
  float y[4];
  rms_rope_row<RD>(X + s * x_ss + h * x_sh, smem[warp], w, tab + (long)s * (RD * 2), eps, lane, y);
  __nv_bfloat162 a = __floats2bfloat162_rn(y[0], y[1]), b = __floats2bfloat162_rn(y[2], y[3]);
  uint2 st;
  st.x = *reinterpret_cast<uint32_t*>(&a); st.y = *reinterpret_cast<uint32_t*>(&b);
  *reinterpret_cast<uint2*>(Y + (g * D) + lane * 4) = st;
}


// K with one scale per 128-key block (for attn_fwd6): rows are smoothed + rotated like quant_qk, then
// quantized against the block's absmax. Block (128 keys, head), 8 warps x 16 rows.
__global__ void quant_k_block(const __nv_bfloat16* __restrict__ K, long k_ss, long k_sh, int S, int Spad,
                              const float* __restrict__ kmean, int rotate, int8_t* __restrict__ Ko,
                              float* __restrict__ kbs) {
  __shared__ float wmax[8];
  const int h = blockIdx.y, blk = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  float v[16][4];
  float amax = 0.f;
  const float4 m = *reinterpret_cast<const float4*>(kmean + h * D + lane * 4);
#pragma unroll
  for (int r = 0; r < 16; ++r) {
    const int s = blk * 128 + warp * 16 + r;
    if (s < S) {
      uint2 raw = *reinterpret_cast<const uint2*>(K + s * k_ss + h * k_sh + lane * 4);
      float2 a = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&raw.x));
      float2 b = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&raw.y));
      v[r][0] = a.x - m.x; v[r][1] = a.y - m.y; v[r][2] = b.x - m.z; v[r][3] = b.y - m.w;
      if (rotate) hadamard128(v[r], lane);
    } else {
      v[r][0] = v[r][1] = v[r][2] = v[r][3] = 0.f;
    }
    amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v[r][0]), fabsf(v[r][1])), fmaxf(fabsf(v[r][2]), fabsf(v[r][3]))));
  }
#pragma unroll
  for (int off = 16; off; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
  if (lane == 0) wmax[warp] = amax;
  __syncthreads();
  float bm = wmax[0];
#pragma unroll
  for (int w = 1; w < 8; ++w) bm = fmaxf(bm, wmax[w]);
  const float sc = fmaxf(bm, 1e-8f) / 127.f, inv = 1.f / sc;
  if (threadIdx.x == 0) kbs[h * (Spad / 128) + blk] = sc;
#pragma unroll
  for (int r = 0; r < 16; ++r) {
    const int s = blk * 128 + warp * 16 + r;
    char4 qv;
    qv.x = (int8_t)__float2int_rn(v[r][0] * inv); qv.y = (int8_t)__float2int_rn(v[r][1] * inv);
    qv.z = (int8_t)__float2int_rn(v[r][2] * inv); qv.w = (int8_t)__float2int_rn(v[r][3] * inv);
    *reinterpret_cast<char4*>(Ko + ((long)h * Spad + s) * D + lane * 4) = qv;
  }
}

// V -> int8 per-channel, transposed to [H][D][Spad] with keys permuted inside each 32-key group so
// that the QK accumulator fragment can be reused directly as the A operand of the PV mma.
// A column c of a 32-key group holds key  (c/16)*16 + ((c%4)/2)*8 + 2*((c%16)/4) + (c%2).
__device__ __forceinline__ int perm32(int c) {
  return (c >> 4) * 16 + ((c & 3) >> 1) * 8 + 2 * ((c & 15) >> 2) + (c & 1);
}

__global__ void quant_v(const __nv_bfloat16* __restrict__ V, long v_ss, long v_sh, int S, int Spad,
                        const float* __restrict__ vmean, const float* __restrict__ vscale,
                        int8_t* __restrict__ Vt, int tiled, int fp8) {
  constexpr int T = 64, P = D + 4;
  __shared__ int8_t tile[T * P];
  const int h = blockIdx.y, s0 = blockIdx.x * T, tid = threadIdx.x;
  // load + quantize: 64 tokens x 128 channels, 4 channels per thread-iteration
  for (int i = tid; i < T * D / 4; i += blockDim.x) {
    const int r = i / (D / 4), c4 = (i % (D / 4)) * 4, s = s0 + r;
    char4 qv = make_char4(0, 0, 0, 0);
    if (s < S) {
      uint2 raw = *reinterpret_cast<const uint2*>(V + s * v_ss + h * v_sh + c4);
      float2 a = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&raw.x));
      float2 b = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&raw.y));
      float4 m = *reinterpret_cast<const float4*>(vmean + h * D + c4);
      float4 sc = *reinterpret_cast<const float4*>(vscale + h * D + c4);
      if (fp8) {  // e4m3 with the per-channel scale stretched to 448
        constexpr float r = 448.f / 127.f;
        qv.x = (int8_t)__nv_cvt_float_to_fp8((a.x - m.x) / sc.x * r, __NV_SATFINITE, __NV_E4M3);
        qv.y = (int8_t)__nv_cvt_float_to_fp8((a.y - m.y) / sc.y * r, __NV_SATFINITE, __NV_E4M3);
        qv.z = (int8_t)__nv_cvt_float_to_fp8((b.x - m.z) / sc.z * r, __NV_SATFINITE, __NV_E4M3);
        qv.w = (int8_t)__nv_cvt_float_to_fp8((b.y - m.w) / sc.w * r, __NV_SATFINITE, __NV_E4M3);
      } else {
        qv.x = (int8_t)__float2int_rn((a.x - m.x) / sc.x); qv.y = (int8_t)__float2int_rn((a.y - m.y) / sc.y);
        qv.z = (int8_t)__float2int_rn((b.x - m.z) / sc.z); qv.w = (int8_t)__float2int_rn((b.y - m.w) / sc.w);
      }
    }
    *reinterpret_cast<char4*>(&tile[r * P + c4]) = qv;
  }
  __syncthreads();
  // write: row d, 16 consecutive (permuted) key columns per store
  for (int i = tid; i < D * (T / 16); i += blockDim.x) {
    const int d = i / (T / 16), cc = (i % (T / 16)) * 16;
    alignas(16) int8_t out[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const int c = cc + j, key = (c & ~31) + perm32(c & 31);
      out[j] = tile[key * P + d];
    }
    // tiled: [H][Spad/64][128][64] with chunk c of row d at c ^ ((d >> 1) & 3) (smem image of a tile)
    int8_t* dst = tiled ? Vt + (((long)h * (Spad / T) + blockIdx.x) * D + d) * T + (((cc >> 4) ^ ((d >> 1) & 3)) << 4)
                        : Vt + ((long)h * D + d) * Spad + s0 + cc;
    *reinterpret_cast<int4*>(dst) = *reinterpret_cast<int4*>(out);
  }
}

// ------------------------------------------------------------------------------------------------
// main kernel
// ------------------------------------------------------------------------------------------------
template <int BN, int STAGES, int NW, bool KSG>
struct Cfg {
  static constexpr int BM = NW * 16;
  static constexpr int K_BYTES = BN * D;
  static constexpr int V_BYTES = D * BN;
  static constexpr int KS_BYTES = KSG ? 0 : BN * 4;  // KSG: K scales read from global (L1) instead
  static constexpr int STAGE = K_BYTES + V_BYTES + KS_BYTES;
  static constexpr int SMEM = STAGE * STAGES;
};

template <int BN, int STAGES, int NW, int MINB, bool KSG>
__global__ void __launch_bounds__(NW * 32, MINB)
attn_fwd(const int8_t* __restrict__ Qg, const float* __restrict__ QS, const int8_t* __restrict__ Kg,
         const float* __restrict__ KS, const int8_t* __restrict__ Vt, const float* __restrict__ VS,
         const float* __restrict__ VM, __nv_bfloat16* __restrict__ O, long o_ss, long o_sh, int S, int Spad,
         float scale_log2) {
  using C = Cfg<BN, STAGES, NW, KSG>;
  constexpr int NB = BN / 8;    // n8 blocks of keys per tile
  constexpr int KC = BN / 32;   // 32-key chunks per tile
  constexpr int NT = NW * 32;
  extern __shared__ __align__(128) uint8_t smem[];

  const int h = blockIdx.y, m0 = blockIdx.x * C::BM;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tq = lane & 3;
  const int8_t* Kh = Kg + (long)h * Spad * D;
  const int8_t* Vh = Vt + (long)h * D * Spad;
  const float* KSh = KS + (long)h * Spad;
  const int ntiles = (S + BN - 1) / BN;

  auto load_tile = [&](int it) {
    uint8_t* st = smem + (it % STAGES) * C::STAGE;
    const int j0 = it * BN;
#pragma unroll
    for (int i = tid; i < BN * 8; i += NT) {  // K: BN rows x 8 chunks
      const int r = i >> 3, c = i & 7;
      cp_async16(st + swz<128>(r, c), Kh + (long)(j0 + r) * D + c * 16);
    }
    constexpr int CPR = BN / 16;
#pragma unroll
    for (int i = tid; i < D * CPR; i += NT) {  // Vt: 128 rows x BN bytes
      const int r = i / CPR, c = i % CPR;
      cp_async16(st + C::K_BYTES + swz<BN>(r, c), Vh + (long)r * Spad + j0 + c * 16);
    }
    if constexpr (!KSG)
      for (int i = tid; i < BN / 4; i += NT) cp_async16(st + C::K_BYTES + C::V_BYTES + i * 16, KSh + j0 + i * 4);
  };

#pragma unroll
  for (int i = 0; i < STAGES - 1; ++i) {
    if (i < ntiles) load_tile(i);
    cp_async_commit();
  }

  // Q fragments straight from global (int8 [Spad][128] per head): 4 k-steps x 4 regs
  const int r0 = m0 + warp * 16 + g;
  uint32_t qa[4][4];
  {
    const int8_t* q0 = Qg + ((long)h * Spad + r0) * D;
    const int8_t* q1 = q0 + 8 * D;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      qa[k][0] = *reinterpret_cast<const uint32_t*>(q0 + k * 32 + tq * 4);
      qa[k][1] = *reinterpret_cast<const uint32_t*>(q1 + k * 32 + tq * 4);
      qa[k][2] = *reinterpret_cast<const uint32_t*>(q0 + k * 32 + 16 + tq * 4);
      qa[k][3] = *reinterpret_cast<const uint32_t*>(q1 + k * 32 + 16 + tq * 4);
    }
  }
  const float c0 = QS[(long)h * Spad + r0] * scale_log2, c1 = QS[(long)h * Spad + r0 + 8] * scale_log2;

  float of[16][4];
#pragma unroll
  for (int i = 0; i < 16; ++i) of[i][0] = of[i][1] = of[i][2] = of[i][3] = 0.f;
  float mx0 = -INFINITY, mx1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  const int magic = MAGIC_I;

  for (int it = 0; it < ntiles; ++it) {
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    if (it + STAGES - 1 < ntiles) load_tile(it + STAGES - 1);
    cp_async_commit();

    const uint8_t* sK = smem + (it % STAGES) * C::STAGE;
    const uint8_t* sV = sK + C::K_BYTES;
    const float* sKS = KSG ? KSh + it * BN : reinterpret_cast<const float*>(sV + C::V_BYTES);

    // ---- S = Q K^T
    int sc[NB][4];
#pragma unroll
    for (int nb = 0; nb < NB; ++nb) sc[nb][0] = sc[nb][1] = sc[nb][2] = sc[nb][3] = 0;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
      for (int nb = 0; nb < NB; nb += 2) {  // x4 = (nb, nb+1) x (2 chunks of this k-step)
        uint32_t b0, b1, b2, b3;
        const int m = lane >> 3;
        ldsm_x4(b0, b1, b2, b3, sK + swz<128>((nb + (m >> 1)) * 8 + (lane & 7), kk * 2 + (m & 1)));
        mma_s8(sc[nb], qa[kk], b0, b1);
        mma_s8(sc[nb + 1], qa[kk], b2, b3);
      }
    }

    // ---- online softmax (t = q_int . k_real; logit*log2e = t * c). P is quantized to u8 relative to
    // this tile's own row max (full 0..255 range per tile); ts = 2^(tile max - running max) rescales it.
    float t[NB][4];
    float tm0 = -INFINITY, tm1 = -INFINITY;
    const bool tail = (it + 1) * BN > S;
#pragma unroll
    for (int nb = 0; nb < NB; ++nb) {
      const float2 kv = KSG ? __ldg(reinterpret_cast<const float2*>(sKS + nb * 8 + tq * 2))
                            : *reinterpret_cast<const float2*>(sKS + nb * 8 + tq * 2);
      t[nb][0] = i2f(sc[nb][0]) * kv.x; t[nb][1] = i2f(sc[nb][1]) * kv.y;
      t[nb][2] = i2f(sc[nb][2]) * kv.x; t[nb][3] = i2f(sc[nb][3]) * kv.y;
      if (tail) {
        const int j = it * BN + nb * 8 + tq * 2;
        if (j >= S) t[nb][0] = t[nb][2] = -INFINITY;
        if (j + 1 >= S) t[nb][1] = t[nb][3] = -INFINITY;
      }
      tm0 = fmaxf(tm0, fmaxf(t[nb][0], t[nb][1]));
      tm1 = fmaxf(tm1, fmaxf(t[nb][2], t[nb][3]));
    }
    tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffff, tm0, 1));
    tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffff, tm0, 2));
    tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffff, tm1, 1));
    tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffff, tm1, 2));
    if (__any_sync(0xffffffff, tm0 > mx0 || tm1 > mx1)) {
      const float n0 = fmaxf(mx0, tm0), n1 = fmaxf(mx1, tm1);
      const float a0 = ex2((mx0 - n0) * c0), a1 = ex2((mx1 - n1) * c1);
#pragma unroll
      for (int i = 0; i < 16; ++i) { of[i][0] *= a0; of[i][1] *= a0; of[i][2] *= a1; of[i][3] *= a1; }
      l0 *= a0; l1 *= a1;
      mx0 = n0; mx1 = n1;
    }
    const float ts0 = ex2((tm0 - mx0) * c0), ts1 = ex2((tm1 - mx1) * c1);
    const float b0 = fmaf(-tm0, c0, LOG2_255), b1 = fmaf(-tm1, c1, LOG2_255);
    uint32_t pa[KC][4];
    uint32_t rs0 = 0, rs1 = 0;
#pragma unroll
    for (int kc = 0; kc < KC; ++kc) {
      int q[4][4];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int nb = kc * 4 + u;
        q[u][0] = __float_as_int(ex2(fmaf(t[nb][0], c0, b0)) + MAGIC_F);
        q[u][1] = __float_as_int(ex2(fmaf(t[nb][1], c0, b0)) + MAGIC_F);
        q[u][2] = __float_as_int(ex2(fmaf(t[nb][2], c1, b1)) + MAGIC_F);
        q[u][3] = __float_as_int(ex2(fmaf(t[nb][3], c1, b1)) + MAGIC_F);
      }
      pa[kc][0] = pack4(q[0][0], q[0][1], q[1][0], q[1][1]);
      pa[kc][1] = pack4(q[0][2], q[0][3], q[1][2], q[1][3]);
      pa[kc][2] = pack4(q[2][0], q[2][1], q[3][0], q[3][1]);
      pa[kc][3] = pack4(q[2][2], q[2][3], q[3][2], q[3][3]);
      rs0 = __dp4a(pa[kc][0], 0x01010101u, rs0); rs0 = __dp4a(pa[kc][2], 0x01010101u, rs0);
      rs1 = __dp4a(pa[kc][1], 0x01010101u, rs1); rs1 = __dp4a(pa[kc][3], 0x01010101u, rs1);
    }
    l0 = fmaf(ts0, i2f(rs0), l0);
    l1 = fmaf(ts1, i2f(rs1), l1);

    // ---- O += ts * (P V), in two halves of D. The int32 accumulator starts at MAGIC so its bit
    // pattern is directly the float MAGIC_F + sum (|sum| <= BN*255*127 < 2^22).
    const float k0 = -ts0 * MAGIC_F, k1 = -ts1 * MAGIC_F;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      int acc[8][4];
#pragma unroll
      for (int kc = 0; kc < KC; ++kc) {
#pragma unroll
        for (int nd = 0; nd < 8; nd += 2) {
          uint32_t v0, v1, v2, v3;
          const int m = lane >> 3;
          ldsm_x4(v0, v1, v2, v3, sV + swz<BN>((half * 8 + nd + (m >> 1)) * 8 + (lane & 7), kc * 2 + (m & 1)));
          if (kc == 0) {
            mma_u8s8_c(acc[nd], pa[kc], v0, v1, magic);
            mma_u8s8_c(acc[nd + 1], pa[kc], v2, v3, magic);
          } else {
            mma_u8s8(acc[nd], pa[kc], v0, v1);
            mma_u8s8(acc[nd + 1], pa[kc], v2, v3);
          }
        }
      }
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        float* o = of[half * 8 + i];
        o[0] = fmaf(ts0, __int_as_float(acc[i][0]), o[0] + k0);
        o[1] = fmaf(ts0, __int_as_float(acc[i][1]), o[1] + k0);
        o[2] = fmaf(ts1, __int_as_float(acc[i][2]), o[2] + k1);
        o[3] = fmaf(ts1, __int_as_float(acc[i][3]), o[3] + k1);
      }
    }
  }
  cp_async_wait<0>();

  // ---- epilogue: normalize, undo V scale/shift, store bf16 in [S][H][D]
  l0 += __shfl_xor_sync(0xffffffff, l0, 1); l0 += __shfl_xor_sync(0xffffffff, l0, 2);
  l1 += __shfl_xor_sync(0xffffffff, l1, 1); l1 += __shfl_xor_sync(0xffffffff, l1, 2);
  const float il0 = 1.f / l0, il1 = 1.f / l1;
  const float* vs = VS + h * D;
  const float* vm = VM + h * D;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int d = i * 8 + tq * 2;
    const float2 s2 = *reinterpret_cast<const float2*>(vs + d);
    const float2 m2 = *reinterpret_cast<const float2*>(vm + d);
    if (r0 < S)
      *reinterpret_cast<__nv_bfloat162*>(O + r0 * o_ss + h * o_sh + d) =
          __floats2bfloat162_rn(fmaf(of[i][0] * il0, s2.x, m2.x), fmaf(of[i][1] * il0, s2.y, m2.y));
    if (r0 + 8 < S)
      *reinterpret_cast<__nv_bfloat162*>(O + (r0 + 8) * o_ss + h * o_sh + d) =
          __floats2bfloat162_rn(fmaf(of[i][2] * il1, s2.x, m2.x), fmaf(of[i][3] * il1, s2.y, m2.y));
  }
}

template <int BN, int STAGES, int NW, int MINB = 1, bool KSG = false>
void launch_fwd(const torch::Tensor& q8, const torch::Tensor& qs, const torch::Tensor& k8, const torch::Tensor& ks,
                const torch::Tensor& vt, const torch::Tensor& vs, const torch::Tensor& vm, torch::Tensor& out,
                int S, int Spad, float scale_log2, cudaStream_t stream) {
  using C = Cfg<BN, STAGES, NW, KSG>;
  const int H = q8.size(0);
  auto kern = attn_fwd<BN, STAGES, NW, MINB, KSG>;
  static bool once = [&] {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
    return true;
  }();
  (void)once;
  dim3 grid((S + C::BM - 1) / C::BM, H);
  kern<<<grid, NW * 32, C::SMEM, stream>>>(
      q8.data_ptr<int8_t>(), qs.data_ptr<float>(), k8.data_ptr<int8_t>(), ks.data_ptr<float>(),
      vt.data_ptr<int8_t>(), vs.data_ptr<float>(), vm.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), out.stride(0), out.stride(1), S, Spad, scale_log2);
}

// Software-pipelined variant: Q·K^T of tile i+1 is issued before the softmax of tile i, so the
// tensor pipe has independent work while the FMA/MUFU pipes run the softmax. Branch-free inner loop:
// per-key (scale, bias) folds dequantization and the tail mask into one FFMA, and the running-max
// rescale is folded into the per-tile output update.
template <int BN, int STAGES, int NW>
struct Cfg2 {
  static constexpr int BM = NW * 16;
  static constexpr int K_BYTES = BN * D;
  static constexpr int V_BYTES = D * BN;
  static constexpr int KS_BYTES = BN * 8;
  static constexpr int STAGE = K_BYTES + V_BYTES + KS_BYTES;
  static constexpr int SMEM = STAGE * STAGES;
};

// MODE (profiling only): 0 full, 1 no softmax (P = raw score bytes), 2 no output fold, 3 no P.V
template <int BN, int STAGES, int NW, int MODE = 0, int PVC = 2, bool SKEW = false>
__global__ void __launch_bounds__(NW * 32, 1)
attn_fwd2(const int8_t* __restrict__ Qg, const float* __restrict__ QS, const int8_t* __restrict__ Kg,
          const float2* __restrict__ KSB, const int8_t* __restrict__ Vt, const float* __restrict__ VS,
          const float* __restrict__ VM, __nv_bfloat16* __restrict__ O, long o_ss, long o_sh, int S, int Spad,
          float scale_log2) {
  static_assert(STAGES >= 3, "tiles i and i+1 must be resident while i+2 loads");
  using C = Cfg2<BN, STAGES, NW>;
  constexpr int NB = BN / 8, KC = BN / 32, NT = NW * 32;
  extern __shared__ __align__(128) uint8_t smem[];

  const int h = blockIdx.y, m0 = blockIdx.x * C::BM;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tq = lane & 3;
  const int8_t* Kh = Kg + (long)h * Spad * D;
  const int8_t* Vh = Vt + (long)h * D * Spad;
  const float2* KSh = KSB + (long)h * Spad;
  const int ntiles = (S + BN - 1) / BN;

  auto load_tile = [&](int it) {
    uint8_t* st = smem + (it % STAGES) * C::STAGE;
    const int j0 = it * BN;
#pragma unroll
    for (int i = tid; i < BN * 8; i += NT) {
      const int r = i >> 3, c = i & 7;
      cp_async16(st + swz<128>(r, c), Kh + (long)(j0 + r) * D + c * 16);
    }
    constexpr int CPR = BN / 16;
#pragma unroll
    for (int i = tid; i < D * CPR; i += NT) {
      const int r = i / CPR, c = i % CPR;
      cp_async16(st + C::K_BYTES + swz<BN>(r, c), Vh + (long)r * Spad + j0 + c * 16);
    }
    for (int i = tid; i < BN / 2; i += NT) cp_async16(st + C::K_BYTES + C::V_BYTES + i * 16, KSh + j0 + i * 2);
  };

  const int r0 = m0 + warp * 16 + g;
  uint32_t qa[4][4];
  auto qk = [&](int (&sc)[NB][4], const uint8_t* sK) {
#pragma unroll
    for (int nb = 0; nb < NB; ++nb) sc[nb][0] = sc[nb][1] = sc[nb][2] = sc[nb][3] = MAGIC_I;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
      for (int nb = 0; nb < NB; nb += 2) {
        uint32_t b0, b1, b2, b3;
        const int m = lane >> 3;
        ldsm_x4(b0, b1, b2, b3, sK + swz<128>((nb + (m >> 1)) * 8 + (lane & 7), kk * 2 + (m & 1)));
        mma_s8(sc[nb], qa[kk], b0, b1);
        mma_s8(sc[nb + 1], qa[kk], b2, b3);
      }
    }
  };

#pragma unroll
  for (int i = 0; i < STAGES - 1; ++i) {
    if (i < ntiles) load_tile(i);
    cp_async_commit();
  }
  {
    const int8_t* q0 = Qg + ((long)h * Spad + r0) * D;
    const int8_t* q1 = q0 + 8 * D;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      qa[k][0] = *reinterpret_cast<const uint32_t*>(q0 + k * 32 + tq * 4);
      qa[k][1] = *reinterpret_cast<const uint32_t*>(q1 + k * 32 + tq * 4);
      qa[k][2] = *reinterpret_cast<const uint32_t*>(q0 + k * 32 + 16 + tq * 4);
      qa[k][3] = *reinterpret_cast<const uint32_t*>(q1 + k * 32 + 16 + tq * 4);
    }
  }
  const float c0 = QS[(long)h * Spad + r0] * scale_log2, c1 = QS[(long)h * Spad + r0 + 8] * scale_log2;

  float of[16][4];
#pragma unroll
  for (int i = 0; i < 16; ++i) of[i][0] = of[i][1] = of[i][2] = of[i][3] = 0.f;
  float mx0 = -INFINITY, mx1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  const int magic = MAGIC_I;

  cp_async_wait<STAGES - 2>();
  __syncthreads();
  int sc[NB][4];  // accumulators start at MAGIC_I: bit pattern = float(MAGIC_F + s)
  qk(sc, smem);

  for (int it = 0; it < ntiles; ++it) {
    cp_async_wait<STAGES - 3>();
    __syncthreads();
    if (it + STAGES - 1 < ntiles) load_tile(it + STAGES - 1);
    cp_async_commit();
    const uint8_t* st = smem + (it % STAGES) * C::STAGE;
    const uint8_t* sV = st + C::K_BYTES;
    const float* sKS = reinterpret_cast<const float*>(sV + C::V_BYTES);

    // next tile's scores (garbage but harmless past the end: the slot holds stale data)
    int sn[NB][4];
    uint32_t pa[KC][4];
    float a0 = 1.f, a1 = 1.f, ts0 = 1.f, ts1 = 1.f;
    // ---- softmax of this tile
    auto softmax = [&]() {
    if constexpr (MODE == 1) {
#pragma unroll
      for (int kc = 0; kc < KC; ++kc)
#pragma unroll
        for (int j = 0; j < 4; ++j) pa[kc][j] = sc[kc * 4 + j][j];
    } else {
    float t[NB][4];
    float tm0 = -INFINITY, tm1 = -INFINITY;
#pragma unroll
    for (int nb = 0; nb < NB; ++nb) {
      const float4 kb = *reinterpret_cast<const float4*>(sKS + (nb * 8 + tq * 2) * 2);  // (s_j, b_j, s_j+1, b_j+1)
      t[nb][0] = fmaf(__int_as_float(sc[nb][0]), kb.x, kb.y);
      t[nb][1] = fmaf(__int_as_float(sc[nb][1]), kb.z, kb.w);
      t[nb][2] = fmaf(__int_as_float(sc[nb][2]), kb.x, kb.y);
      t[nb][3] = fmaf(__int_as_float(sc[nb][3]), kb.z, kb.w);
      tm0 = fmaxf(tm0, fmaxf(t[nb][0], t[nb][1]));
      tm1 = fmaxf(tm1, fmaxf(t[nb][2], t[nb][3]));
    }
    tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffff, tm0, 1));
    tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffff, tm0, 2));
    tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffff, tm1, 1));
    tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffff, tm1, 2));
    const float n0 = fmaxf(mx0, tm0), n1 = fmaxf(mx1, tm1);
    a0 = ex2((mx0 - n0) * c0); a1 = ex2((mx1 - n1) * c1);
    ts0 = ex2((tm0 - n0) * c0); ts1 = ex2((tm1 - n1) * c1);
    mx0 = n0; mx1 = n1;
    const float b0 = fmaf(-tm0, c0, LOG2_255), b1 = fmaf(-tm1, c1, LOG2_255);
    uint32_t rs0 = 0, rs1 = 0;
#pragma unroll
    for (int kc = 0; kc < KC; ++kc) {
      int q[4][4];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int nb = kc * 4 + u;
        q[u][0] = __float_as_int(ex2(fmaf(t[nb][0], c0, b0)) + MAGIC_F);
        q[u][1] = __float_as_int(ex2(fmaf(t[nb][1], c0, b0)) + MAGIC_F);
        q[u][2] = __float_as_int(ex2(fmaf(t[nb][2], c1, b1)) + MAGIC_F);
        q[u][3] = __float_as_int(ex2(fmaf(t[nb][3], c1, b1)) + MAGIC_F);
      }
      pa[kc][0] = pack4(q[0][0], q[0][1], q[1][0], q[1][1]);
      pa[kc][1] = pack4(q[0][2], q[0][3], q[1][2], q[1][3]);
      pa[kc][2] = pack4(q[2][0], q[2][1], q[3][0], q[3][1]);
      pa[kc][3] = pack4(q[2][2], q[2][3], q[3][2], q[3][3]);
      rs0 = __dp4a(pa[kc][0], 0x01010101u, rs0); rs0 = __dp4a(pa[kc][2], 0x01010101u, rs0);
      rs1 = __dp4a(pa[kc][1], 0x01010101u, rs1); rs1 = __dp4a(pa[kc][3], 0x01010101u, rs1);
    }
    l0 = fmaf(ts0, i2f(rs0), l0 * a0);
    l1 = fmaf(ts1, i2f(rs1), l1 * a1);
    }
    };
    // SKEW: warps 4-7 (second warp of each sub-partition) run the softmax before issuing the next
    // tile's Q.K^T, so the two warps sharing a tensor pipe are out of phase within each tile.
    if (SKEW && (warp & 4)) {
      softmax();
      qk(sn, smem + ((it + 1) % STAGES) * C::STAGE);
    } else {
      qk(sn, smem + ((it + 1) % STAGES) * C::STAGE);
      softmax();
    }

    // ---- O = a*O + ts*(P V)
    const float k0 = -ts0 * MAGIC_F, k1 = -ts1 * MAGIC_F;
#pragma unroll
    for (int half = 0; half < (MODE == 3 ? 0 : PVC); ++half) {
      constexpr int ND = 16 / PVC;
      int acc[ND][4];
#pragma unroll
      for (int kc = 0; kc < KC; ++kc) {
#pragma unroll
        for (int nd = 0; nd < ND; nd += 2) {
          uint32_t v0, v1, v2, v3;
          const int m = lane >> 3;
          ldsm_x4(v0, v1, v2, v3, sV + swz<BN>((half * ND + nd + (m >> 1)) * 8 + (lane & 7), kc * 2 + (m & 1)));
          if (kc == 0) {
            mma_u8s8_c(acc[nd], pa[kc], v0, v1, magic);
            mma_u8s8_c(acc[nd + 1], pa[kc], v2, v3, magic);
          } else {
            mma_u8s8(acc[nd], pa[kc], v0, v1);
            mma_u8s8(acc[nd + 1], pa[kc], v2, v3);
          }
        }
      }
#pragma unroll
      for (int i = 0; i < ND; ++i) {
        float* o = of[half * ND + i];
        if constexpr (MODE == 1 || MODE == 2) {
          o[0] += __int_as_float(acc[i][0]); o[1] += __int_as_float(acc[i][1]);
          o[2] += __int_as_float(acc[i][2]); o[3] += __int_as_float(acc[i][3]);
          continue;
        }
        o[0] = fmaf(ts0, __int_as_float(acc[i][0]), fmaf(o[0], a0, k0));
        o[1] = fmaf(ts0, __int_as_float(acc[i][1]), fmaf(o[1], a0, k0));
        o[2] = fmaf(ts1, __int_as_float(acc[i][2]), fmaf(o[2], a1, k1));
        o[3] = fmaf(ts1, __int_as_float(acc[i][3]), fmaf(o[3], a1, k1));
      }
    }
    if constexpr (MODE == 3) {
#pragma unroll
      for (int kc = 0; kc < KC; ++kc) of[kc][0] += __int_as_float(pa[kc][0] ^ pa[kc][1] ^ pa[kc][2] ^ pa[kc][3]);
    }
#pragma unroll
    for (int nb = 0; nb < NB; ++nb)
#pragma unroll
      for (int j = 0; j < 4; ++j) sc[nb][j] = sn[nb][j];
  }
  cp_async_wait<0>();

  l0 += __shfl_xor_sync(0xffffffff, l0, 1); l0 += __shfl_xor_sync(0xffffffff, l0, 2);
  l1 += __shfl_xor_sync(0xffffffff, l1, 1); l1 += __shfl_xor_sync(0xffffffff, l1, 2);
  const float il0 = 1.f / l0, il1 = 1.f / l1;
  const float* vs = VS + h * D;
  const float* vm = VM + h * D;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int d = i * 8 + tq * 2;
    const float2 s2 = *reinterpret_cast<const float2*>(vs + d);
    const float2 m2 = *reinterpret_cast<const float2*>(vm + d);
    if (r0 < S)
      *reinterpret_cast<__nv_bfloat162*>(O + r0 * o_ss + h * o_sh + d) =
          __floats2bfloat162_rn(fmaf(of[i][0] * il0, s2.x, m2.x), fmaf(of[i][1] * il0, s2.y, m2.y));
    if (r0 + 8 < S)
      *reinterpret_cast<__nv_bfloat162*>(O + (r0 + 8) * o_ss + h * o_sh + d) =
          __floats2bfloat162_rn(fmaf(of[i][2] * il1, s2.x, m2.x), fmaf(of[i][3] * il1, s2.y, m2.y));
  }
}

template <int BN, int STAGES, int NW, int MODE = 0, int PVC = 2, bool SKEW = false>
void launch_fwd2(const torch::Tensor& q8, const torch::Tensor& qs, const torch::Tensor& k8, const torch::Tensor& ksb,
                 const torch::Tensor& vt, const torch::Tensor& vs, const torch::Tensor& vm, torch::Tensor& out,
                 int S, int Spad, float scale_log2, cudaStream_t stream) {
  using C = Cfg2<BN, STAGES, NW>;
  const int H = q8.size(0);
  auto kern = attn_fwd2<BN, STAGES, NW, MODE, PVC, SKEW>;
  static bool once = [&] {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
    return true;
  }();
  (void)once;
  dim3 grid((S + C::BM - 1) / C::BM, H);
  kern<<<grid, NW * 32, C::SMEM, stream>>>(
      q8.data_ptr<int8_t>(), qs.data_ptr<float>(), k8.data_ptr<int8_t>(),
      reinterpret_cast<const float2*>(ksb.data_ptr<float>()), vt.data_ptr<int8_t>(), vs.data_ptr<float>(),
      vm.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), out.stride(0), out.stride(1), S, Spad,
      scale_log2);
}

// ------------------------------------------------------------------------------------------------
// Ping-pong kernel. Two groups of 4 warps share each SM sub-partition; named barriers hand the
// tensor pipe back and forth so one group's softmax runs while the other group's IMMAs issue
// (FlashAttention-3 style scheduling, on mma.sync). K/V tiles arrive by bulk copy (one thread,
// mbarrier tx-count), from global buffers that already hold the swizzled smem image.
// ------------------------------------------------------------------------------------------------
__device__ __forceinline__ void mbar_init(uint64_t* b, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(b)), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, int bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, int parity) {
  asm volatile(
      "{\n .reg .pred p;\n WAIT_%=:\n"
      " mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      " @!p bra WAIT_%=;\n}\n" ::"r"(smem_u32(b)), "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, int bytes, uint64_t* b) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void named_sync(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void named_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(n) : "memory"); }

template <int STAGES>
struct Cfg3 {
  static constexpr int BN = 64, NW = 8, BM = NW * 16;
  static constexpr int K_BYTES = BN * D, V_BYTES = D * BN, KS_BYTES = BN * 8;
  static constexpr int STAGE = K_BYTES + V_BYTES + KS_BYTES;
  static constexpr int SMEM = STAGE * STAGES + 2 * STAGES * 8;
};

template <int STAGES, bool PINGPONG>
__global__ void __launch_bounds__(256, 1)
attn_fwd3(const int8_t* __restrict__ Qg, const float* __restrict__ QS, const int8_t* __restrict__ Kg,
          const float2* __restrict__ KSB, const int8_t* __restrict__ Vt, const float* __restrict__ VS,
          const float* __restrict__ VM, __nv_bfloat16* __restrict__ O, long o_ss, long o_sh, int S, int Spad,
          float scale_log2) {
  using C = Cfg3<STAGES>;
  constexpr int BN = C::BN, NB = BN / 8, KC = BN / 32;
  extern __shared__ __align__(128) uint8_t smem[];
  uint64_t* full = reinterpret_cast<uint64_t*>(smem + C::STAGE * STAGES);
  uint64_t* empty = full + STAGES;

  const int h = blockIdx.y, m0 = blockIdx.x * C::BM;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tq = lane & 3;
  const int grp = warp >> 2;  // warps w and w+4 share sub-partition w%4
  const int ntiles = (S + BN - 1) / BN;
  const int8_t* Kh = Kg + (long)h * Spad * D;
  const int8_t* Vh = Vt + (long)h * D * Spad;  // tiles of 128 x 64, contiguous
  const float2* KSh = KSB + (long)h * Spad;

  auto issue = [&](int t) {  // thread 0 only
    const int sl = t % STAGES;
    if (t >= STAGES) mbar_wait(&empty[sl], ((t / STAGES) - 1) & 1);
    uint8_t* st = smem + sl * C::STAGE;
    mbar_expect_tx(&full[sl], C::STAGE);
    bulk_g2s(st, Kh + (long)t * BN * D, C::K_BYTES, &full[sl]);
    bulk_g2s(st + C::K_BYTES, Vh + (long)t * BN * D, C::V_BYTES, &full[sl]);
    bulk_g2s(st + C::K_BYTES + C::V_BYTES, KSh + (long)t * BN, C::KS_BYTES, &full[sl]);
  };

  if (tid == 0) {
    for (int i = 0; i < STAGES; ++i) { mbar_init(&full[i], 1); mbar_init(&empty[i], 8); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (tid == 0)
    for (int t = 0; t < STAGES - 1 && t < ntiles; ++t) issue(t);

  const int r0 = m0 + warp * 16 + g;
  uint32_t qa[4][4];
  {
    const int8_t* q0 = Qg + ((long)h * Spad + r0) * D;
    const int8_t* q1 = q0 + 8 * D;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      qa[k][0] = *reinterpret_cast<const uint32_t*>(q0 + k * 32 + tq * 4);
      qa[k][1] = *reinterpret_cast<const uint32_t*>(q1 + k * 32 + tq * 4);
      qa[k][2] = *reinterpret_cast<const uint32_t*>(q0 + k * 32 + 16 + tq * 4);
      qa[k][3] = *reinterpret_cast<const uint32_t*>(q1 + k * 32 + 16 + tq * 4);
    }
  }
  const float c0 = QS[(long)h * Spad + r0] * scale_log2, c1 = QS[(long)h * Spad + r0 + 8] * scale_log2;

  auto qk = [&](int (&sc)[NB][4], const uint8_t* sK) {
#pragma unroll
    for (int nb = 0; nb < NB; ++nb) sc[nb][0] = sc[nb][1] = sc[nb][2] = sc[nb][3] = MAGIC_I;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
      for (int nb = 0; nb < NB; nb += 2) {
        uint32_t b0, b1, b2, b3;
        const int m = lane >> 3;
        ldsm_x4(b0, b1, b2, b3, sK + swz<128>((nb + (m >> 1)) * 8 + (lane & 7), kk * 2 + (m & 1)));
        mma_s8(sc[nb], qa[kk], b0, b1);
        mma_s8(sc[nb + 1], qa[kk], b2, b3);
      }
    }
  };

  float of[16][4];
#pragma unroll
  for (int i = 0; i < 16; ++i) of[i][0] = of[i][1] = of[i][2] = of[i][3] = 0.f;
  float mx0 = -INFINITY, mx1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  float pa0 = 0.f, pa1 = 0.f, pts0 = 0.f, pts1 = 0.f;  // previous tile's rescale and tile scale
  int acc[16][4];
#pragma unroll
  for (int i = 0; i < 16; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = MAGIC_I;
  const int magic = MAGIC_I;

  int sc[NB][4];
  mbar_wait(&full[0], 0);
  qk(sc, smem);
  if (PINGPONG && grp == 1) named_arrive(1, 256);  // group 0 takes the tensor pipe first

  for (int it = 0; it < ntiles; ++it) {
    // ---------------- softmax phase: fold previous P.V, softmax of tile it
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      float* o = of[i];
      o[0] = fmaf(pts0, __int_as_float(acc[i][0]), fmaf(o[0], pa0, -pts0 * MAGIC_F));
      o[1] = fmaf(pts0, __int_as_float(acc[i][1]), fmaf(o[1], pa0, -pts0 * MAGIC_F));
      o[2] = fmaf(pts1, __int_as_float(acc[i][2]), fmaf(o[2], pa1, -pts1 * MAGIC_F));
      o[3] = fmaf(pts1, __int_as_float(acc[i][3]), fmaf(o[3], pa1, -pts1 * MAGIC_F));
    }
    const int sl = it % STAGES;
    const uint8_t* st = smem + sl * C::STAGE;
    const float* sKS = reinterpret_cast<const float*>(st + C::K_BYTES + C::V_BYTES);
    float t[NB][4];
    float tm0 = -INFINITY, tm1 = -INFINITY;
#pragma unroll
    for (int nb = 0; nb < NB; ++nb) {
      const float4 kb = *reinterpret_cast<const float4*>(sKS + (nb * 8 + tq * 2) * 2);
      t[nb][0] = fmaf(__int_as_float(sc[nb][0]), kb.x, kb.y);
      t[nb][1] = fmaf(__int_as_float(sc[nb][1]), kb.z, kb.w);
      t[nb][2] = fmaf(__int_as_float(sc[nb][2]), kb.x, kb.y);
      t[nb][3] = fmaf(__int_as_float(sc[nb][3]), kb.z, kb.w);
      tm0 = fmaxf(tm0, fmaxf(t[nb][0], t[nb][1]));
      tm1 = fmaxf(tm1, fmaxf(t[nb][2], t[nb][3]));
    }
    tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffff, tm0, 1));
    tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffff, tm0, 2));
    tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffff, tm1, 1));
    tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffff, tm1, 2));
    const float n0 = fmaxf(mx0, tm0), n1 = fmaxf(mx1, tm1);
    pa0 = ex2((mx0 - n0) * c0); pa1 = ex2((mx1 - n1) * c1);
    pts0 = ex2((tm0 - n0) * c0); pts1 = ex2((tm1 - n1) * c1);
    mx0 = n0; mx1 = n1;
    const float b0 = fmaf(-tm0, c0, LOG2_255), b1 = fmaf(-tm1, c1, LOG2_255);
    uint32_t pa[KC][4];
    uint32_t rs0 = 0, rs1 = 0;
#pragma unroll
    for (int kc = 0; kc < KC; ++kc) {
      int q[4][4];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int nb = kc * 4 + u;
        q[u][0] = __float_as_int(ex2(fmaf(t[nb][0], c0, b0)) + MAGIC_F);
        q[u][1] = __float_as_int(ex2(fmaf(t[nb][1], c0, b0)) + MAGIC_F);
        q[u][2] = __float_as_int(ex2(fmaf(t[nb][2], c1, b1)) + MAGIC_F);
        q[u][3] = __float_as_int(ex2(fmaf(t[nb][3], c1, b1)) + MAGIC_F);
      }
      pa[kc][0] = pack4(q[0][0], q[0][1], q[1][0], q[1][1]);
      pa[kc][1] = pack4(q[0][2], q[0][3], q[1][2], q[1][3]);
      pa[kc][2] = pack4(q[2][0], q[2][1], q[3][0], q[3][1]);
      pa[kc][3] = pack4(q[2][2], q[2][3], q[3][2], q[3][3]);
      rs0 = __dp4a(pa[kc][0], 0x01010101u, rs0); rs0 = __dp4a(pa[kc][2], 0x01010101u, rs0);
      rs1 = __dp4a(pa[kc][1], 0x01010101u, rs1); rs1 = __dp4a(pa[kc][3], 0x01010101u, rs1);
    }
    l0 = fmaf(pts0, i2f(rs0), l0 * pa0);
    l1 = fmaf(pts1, i2f(rs1), l1 * pa1);

    // ---------------- tensor phase: P.V of tile it, Q.K^T of tile it+1
    if (PINGPONG) named_sync(1 + grp, 256);
    if (tid == 0 && it + STAGES - 1 < ntiles) issue(it + STAGES - 1);
    const uint8_t* sV = st + C::K_BYTES;
#pragma unroll
    for (int kc = 0; kc < KC; ++kc) {
#pragma unroll
      for (int nd = 0; nd < 16; nd += 2) {
        uint32_t v0, v1, v2, v3;
        const int m = lane >> 3;
        ldsm_x4(v0, v1, v2, v3, sV + swz<BN>((nd + (m >> 1)) * 8 + (lane & 7), kc * 2 + (m & 1)));
        if (kc == 0) {
          mma_u8s8_c(acc[nd], pa[kc], v0, v1, magic);
          mma_u8s8_c(acc[nd + 1], pa[kc], v2, v3, magic);
        } else {
          mma_u8s8(acc[nd], pa[kc], v0, v1);
          mma_u8s8(acc[nd + 1], pa[kc], v2, v3);
        }
      }
    }
    if (it + 1 < ntiles) {
      const int sn = (it + 1) % STAGES;
      mbar_wait(&full[sn], ((it + 1) / STAGES) & 1);
      qk(sc, smem + sn * C::STAGE);
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(&empty[sl]);
    if (PINGPONG && !(grp == 1 && it == ntiles - 1)) named_arrive(2 - grp, 256);
  }
  // final fold
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    float* o = of[i];
    o[0] = fmaf(pts0, __int_as_float(acc[i][0]), fmaf(o[0], pa0, -pts0 * MAGIC_F));
    o[1] = fmaf(pts0, __int_as_float(acc[i][1]), fmaf(o[1], pa0, -pts0 * MAGIC_F));
    o[2] = fmaf(pts1, __int_as_float(acc[i][2]), fmaf(o[2], pa1, -pts1 * MAGIC_F));
    o[3] = fmaf(pts1, __int_as_float(acc[i][3]), fmaf(o[3], pa1, -pts1 * MAGIC_F));
  }

  l0 += __shfl_xor_sync(0xffffffff, l0, 1); l0 += __shfl_xor_sync(0xffffffff, l0, 2);
  l1 += __shfl_xor_sync(0xffffffff, l1, 1); l1 += __shfl_xor_sync(0xffffffff, l1, 2);
  const float il0 = 1.f / l0, il1 = 1.f / l1;
  const float* vs = VS + h * D;
  const float* vm = VM + h * D;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int d = i * 8 + tq * 2;
    const float2 s2 = *reinterpret_cast<const float2*>(vs + d);
    const float2 m2 = *reinterpret_cast<const float2*>(vm + d);
    if (r0 < S)
      *reinterpret_cast<__nv_bfloat162*>(O + r0 * o_ss + h * o_sh + d) =
          __floats2bfloat162_rn(fmaf(of[i][0] * il0, s2.x, m2.x), fmaf(of[i][1] * il0, s2.y, m2.y));
    if (r0 + 8 < S)
      *reinterpret_cast<__nv_bfloat162*>(O + (r0 + 8) * o_ss + h * o_sh + d) =
          __floats2bfloat162_rn(fmaf(of[i][2] * il1, s2.x, m2.x), fmaf(of[i][3] * il1, s2.y, m2.y));
  }
}

template <int STAGES, bool PINGPONG>
void launch_fwd3(const torch::Tensor& q8, const torch::Tensor& qs, const torch::Tensor& k8, const torch::Tensor& ksb,
                 const torch::Tensor& vt, const torch::Tensor& vs, const torch::Tensor& vm, torch::Tensor& out,
                 int S, int Spad, float scale_log2, cudaStream_t stream) {
  using C = Cfg3<STAGES>;
  const int H = q8.size(0);
  auto kern = attn_fwd3<STAGES, PINGPONG>;
  static bool once = [&] {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
    return true;
  }();
  (void)once;
  dim3 grid((S + C::BM - 1) / C::BM, H);
  kern<<<grid, 256, C::SMEM, stream>>>(
      q8.data_ptr<int8_t>(), qs.data_ptr<float>(), k8.data_ptr<int8_t>(),
      reinterpret_cast<const float2*>(ksb.data_ptr<float>()), vt.data_ptr<int8_t>(), vs.data_ptr<float>(),
      vm.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), out.stride(0), out.stride(1), S, Spad,
      scale_log2);
}

// ------------------------------------------------------------------------------------------------
// Warp-pair kernel. A pair of warps owns 32 query rows. For Q.K^T each warp takes half of the key
// tile (32 keys) for both 16-row m-tiles; for P.V each warp takes half of D (64 channels) over all
// keys. Every K/V fragment loaded from smem therefore feeds two m-tiles, halving smem traffic per
// IMMA versus the 16-rows-per-warp layout. The pair swaps per-row tile maxima and its half of P
// through smem (two 64-thread named barriers per tile). Tiles arrive by bulk copy + mbarriers.
// ------------------------------------------------------------------------------------------------
template <int STAGES>
struct Cfg4 {
  static constexpr int BN = 64, NW = 8, BM = 128;
  static constexpr int K_BYTES = BN * D, V_BYTES = D * BN, KS_BYTES = BN * 8;
  static constexpr int STAGE = K_BYTES + V_BYTES + KS_BYTES;
  static constexpr int P_BYTES = 4 * 32 * BN;          // per pair: 32 rows x 64 keys u8
  static constexpr int TM_BYTES = 4 * 2 * 32 * 4;      // per pair, per half: 32 row maxima
  static constexpr int SMEM = STAGE * STAGES + P_BYTES + TM_BYTES + 2 * STAGES * 8;
};

template <int STAGES, int MODE>
__global__ void __launch_bounds__(256, 1)
attn_fwd4(const int8_t* __restrict__ Qg, const float* __restrict__ QS, const int8_t* __restrict__ Kg,
          const float2* __restrict__ KSB, const int8_t* __restrict__ Vt, const float* __restrict__ VS,
          const float* __restrict__ VM, __nv_bfloat16* __restrict__ O, long o_ss, long o_sh, int S, int Spad,
          float scale_log2) {
  using C = Cfg4<STAGES>;
  constexpr int BN = C::BN;
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* sP = smem + C::STAGE * STAGES;
  float* sTM = reinterpret_cast<float*>(sP + C::P_BYTES);
  uint64_t* full = reinterpret_cast<uint64_t*>(sTM + C::TM_BYTES / 4);
  uint64_t* empty = full + STAGES;

  const int h = blockIdx.y, m0 = blockIdx.x * C::BM;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tq = lane & 3;
  const int pair = warp >> 1, hf = warp & 1;
  const int ntiles = (S + BN - 1) / BN;
  const int8_t* Kh = Kg + (long)h * Spad * D;
  const int8_t* Vh = Vt + (long)h * D * Spad;
  const float2* KSh = KSB + (long)h * Spad;
  uint8_t* myP = sP + pair * 32 * BN;                   // [32 rows][64] swizzled, A-column order
  float* tmMine = sTM + (pair * 2 + hf) * 32;
  const float* tmPart = sTM + (pair * 2 + (hf ^ 1)) * 32;
  const int barId = 1 + pair;

  auto issue = [&](int t) {
    const int sl = t % STAGES;
    if (t >= STAGES) mbar_wait(&empty[sl], ((t / STAGES) - 1) & 1);
    uint8_t* st = smem + sl * C::STAGE;
    mbar_expect_tx(&full[sl], C::STAGE);
    bulk_g2s(st, Kh + (long)t * BN * D, C::K_BYTES, &full[sl]);
    bulk_g2s(st + C::K_BYTES, Vh + (long)t * BN * D, C::V_BYTES, &full[sl]);
    bulk_g2s(st + C::K_BYTES + C::V_BYTES, KSh + (long)t * BN, C::KS_BYTES, &full[sl]);
  };
  if (tid == 0) {
    for (int i = 0; i < STAGES; ++i) { mbar_init(&full[i], 1); mbar_init(&empty[i], 8); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (tid == 0)
    for (int t = 0; t < STAGES - 1 && t < ntiles; ++t) issue(t);

  // Q fragments for both m-tiles (rows m0 + pair*32 + mt*16 + {g, g+8})
  const int rb = m0 + pair * 32;
  uint32_t qa[2][4][4];
  float cr[2][2];
#pragma unroll
  for (int mt = 0; mt < 2; ++mt) {
    const int8_t* q0 = Qg + ((long)h * Spad + rb + mt * 16 + g) * D;
    const int8_t* q1 = q0 + 8 * D;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      qa[mt][k][0] = *reinterpret_cast<const uint32_t*>(q0 + k * 32 + tq * 4);
      qa[mt][k][1] = *reinterpret_cast<const uint32_t*>(q1 + k * 32 + tq * 4);
      qa[mt][k][2] = *reinterpret_cast<const uint32_t*>(q0 + k * 32 + 16 + tq * 4);
      qa[mt][k][3] = *reinterpret_cast<const uint32_t*>(q1 + k * 32 + 16 + tq * 4);
    }
    cr[mt][0] = QS[(long)h * Spad + rb + mt * 16 + g] * scale_log2;
    cr[mt][1] = QS[(long)h * Spad + rb + mt * 16 + g + 8] * scale_log2;
  }

  float of[2][8][4];  // [m-tile][n8 block of my 64 channels][frag]
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int i = 0; i < 8; ++i) of[mt][i][0] = of[mt][i][1] = of[mt][i][2] = of[mt][i][3] = 0.f;
  float mx[2][2], ll[2][2];
#pragma unroll
  for (int mt = 0; mt < 2; ++mt) mx[mt][0] = mx[mt][1] = -INFINITY, ll[mt][0] = ll[mt][1] = 0.f;
  const int magic = MAGIC_I;

  for (int it = 0; it < ntiles; ++it) {
    const int sl = it % STAGES;
    const uint8_t* st = smem + sl * C::STAGE;
    if (tid == 0 && it + STAGES - 1 < ntiles) issue(it + STAGES - 1);
    mbar_wait(&full[sl], (it / STAGES) & 1);

    // ---- S = Q K^T for my 32 keys, both m-tiles
    int sc[2][4][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int nb = 0; nb < 4; ++nb) sc[mt][nb][0] = sc[mt][nb][1] = sc[mt][nb][2] = sc[mt][nb][3] = MAGIC_I;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
      for (int nb = 0; nb < 4; nb += 2) {
        uint32_t b0, b1, b2, b3;
        const int m = lane >> 3;
        ldsm_x4(b0, b1, b2, b3, st + swz<128>(hf * 32 + (nb + (m >> 1)) * 8 + (lane & 7), kk * 2 + (m & 1)));
        mma_s8(sc[0][nb], qa[0][kk], b0, b1);
        mma_s8(sc[0][nb + 1], qa[0][kk], b2, b3);
        mma_s8(sc[1][nb], qa[1][kk], b0, b1);
        mma_s8(sc[1][nb + 1], qa[1][kk], b2, b3);
      }
    }

    // ---- dequantize + my-half row maxima
    const float* sKS = reinterpret_cast<const float*>(st + C::K_BYTES + C::V_BYTES) + hf * 64;
    float t[2][4][4], tm[2][2];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      tm[mt][0] = tm[mt][1] = -INFINITY;
#pragma unroll
      for (int nb = 0; nb < 4; ++nb) {
        const float4 kb = *reinterpret_cast<const float4*>(sKS + (nb * 8 + tq * 2) * 2);
        t[mt][nb][0] = fmaf(__int_as_float(sc[mt][nb][0]), kb.x, kb.y);
        t[mt][nb][1] = fmaf(__int_as_float(sc[mt][nb][1]), kb.z, kb.w);
        t[mt][nb][2] = fmaf(__int_as_float(sc[mt][nb][2]), kb.x, kb.y);
        t[mt][nb][3] = fmaf(__int_as_float(sc[mt][nb][3]), kb.z, kb.w);
        tm[mt][0] = fmaxf(tm[mt][0], fmaxf(t[mt][nb][0], t[mt][nb][1]));
        tm[mt][1] = fmaxf(tm[mt][1], fmaxf(t[mt][nb][2], t[mt][nb][3]));
      }
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        tm[mt][r] = fmaxf(tm[mt][r], __shfl_xor_sync(0xffffffff, tm[mt][r], 1));
        tm[mt][r] = fmaxf(tm[mt][r], __shfl_xor_sync(0xffffffff, tm[mt][r], 2));
      }
    }
    if (tq == 0) {
      tmMine[g] = tm[0][0]; tmMine[g + 8] = tm[0][1]; tmMine[16 + g] = tm[1][0]; tmMine[24 + g] = tm[1][1];
    }
    named_sync(barId, 64);
    tm[0][0] = fmaxf(tm[0][0], tmPart[g]); tm[0][1] = fmaxf(tm[0][1], tmPart[g + 8]);
    tm[1][0] = fmaxf(tm[1][0], tmPart[16 + g]); tm[1][1] = fmaxf(tm[1][1], tmPart[24 + g]);

    // ---- P (u8, relative to the pair's common tile max) -> registers + pair buffer
    float al[2][2], ts[2][2];
    uint32_t pa[2][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const float n = fmaxf(mx[mt][r], tm[mt][r]);
        al[mt][r] = ex2((mx[mt][r] - n) * cr[mt][r]);
        ts[mt][r] = ex2((tm[mt][r] - n) * cr[mt][r]);
        mx[mt][r] = n;
      }
      const float b0 = fmaf(-tm[mt][0], cr[mt][0], LOG2_255), b1 = fmaf(-tm[mt][1], cr[mt][1], LOG2_255);
      int q[4][4];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        if constexpr (MODE == 1) {
          q[u][0] = sc[mt][u][0]; q[u][1] = sc[mt][u][1]; q[u][2] = sc[mt][u][2]; q[u][3] = sc[mt][u][3];
        } else {
          q[u][0] = __float_as_int(ex2(fmaf(t[mt][u][0], cr[mt][0], b0)) + MAGIC_F);
          q[u][1] = __float_as_int(ex2(fmaf(t[mt][u][1], cr[mt][0], b0)) + MAGIC_F);
          q[u][2] = __float_as_int(ex2(fmaf(t[mt][u][2], cr[mt][1], b1)) + MAGIC_F);
          q[u][3] = __float_as_int(ex2(fmaf(t[mt][u][3], cr[mt][1], b1)) + MAGIC_F);
        }
      }
      pa[mt][0] = pack4(q[0][0], q[0][1], q[1][0], q[1][1]);
      pa[mt][1] = pack4(q[0][2], q[0][3], q[1][2], q[1][3]);
      pa[mt][2] = pack4(q[2][0], q[2][1], q[3][0], q[3][1]);
      pa[mt][3] = pack4(q[2][2], q[2][3], q[3][2], q[3][3]);
      uint32_t rs0 = __dp4a(pa[mt][2], 0x01010101u, __dp4a(pa[mt][0], 0x01010101u, 0u));
      uint32_t rs1 = __dp4a(pa[mt][3], 0x01010101u, __dp4a(pa[mt][1], 0x01010101u, 0u));
      ll[mt][0] = fmaf(ts[mt][0], i2f(rs0), ll[mt][0] * al[mt][0]);
      ll[mt][1] = fmaf(ts[mt][1], i2f(rs1), ll[mt][1] * al[mt][1]);
      // A-column order: row (mt*16 + g [+8]), bytes hf*32 + {4tq, 16+4tq}
      const int ra = mt * 16 + g, rbw = ra + 8;
      *reinterpret_cast<uint32_t*>(myP + swz<64>(ra, hf * 2) + tq * 4) = pa[mt][0];
      *reinterpret_cast<uint32_t*>(myP + swz<64>(rbw, hf * 2) + tq * 4) = pa[mt][1];
      *reinterpret_cast<uint32_t*>(myP + swz<64>(ra, hf * 2 + 1) + tq * 4) = pa[mt][2];
      *reinterpret_cast<uint32_t*>(myP + swz<64>(rbw, hf * 2 + 1) + tq * 4) = pa[mt][3];
    }
    named_sync(barId, 64);
    // partner's P chunk via ldmatrix (A fragments for both m-tiles)
    uint32_t pb[2][4];
    {
      const int m = lane >> 3;  // matrices: (rows 0-7, c0) (rows 8-15, c0) (rows 0-7, c1) (rows 8-15, c1)
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
        ldsm_x4(pb[mt][0], pb[mt][1], pb[mt][2], pb[mt][3],
                myP + swz<64>(mt * 16 + (m & 1) * 8 + (lane & 7), (hf ^ 1) * 2 + (m >> 1)));
    }

    // ---- O[:, my 64 channels] = al*O + ts*(P V), two 32-channel quarters
    const uint8_t* sV = st + C::K_BYTES;
#pragma unroll
    for (int qd = 0; qd < 2; ++qd) {
      int acc[2][4][4];
#pragma unroll
      for (int kc = 0; kc < 2; ++kc) {
#pragma unroll
        for (int nd = 0; nd < 4; nd += 2) {
          uint32_t v0, v1, v2, v3;
          const int m = lane >> 3;
          const int drow = (hf * 8 + qd * 4 + nd + (m >> 1)) * 8 + (lane & 7);
          ldsm_x4(v0, v1, v2, v3, sV + swz<BN>(drow, kc * 2 + (m & 1)));
#pragma unroll
          for (int mt = 0; mt < 2; ++mt) {
            const uint32_t* a = (kc == hf) ? pa[mt] : pb[mt];
            if (kc == 0) {
              mma_u8s8_c(acc[mt][nd], a, v0, v1, magic);
              mma_u8s8_c(acc[mt][nd + 1], a, v2, v3, magic);
            } else {
              mma_u8s8(acc[mt][nd], a, v0, v1);
              mma_u8s8(acc[mt][nd + 1], a, v2, v3);
            }
          }
        }
      }
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        const float k0 = -ts[mt][0] * MAGIC_F, k1 = -ts[mt][1] * MAGIC_F;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          float* o = of[mt][qd * 4 + i];
          o[0] = fmaf(ts[mt][0], __int_as_float(acc[mt][i][0]), fmaf(o[0], al[mt][0], k0));
          o[1] = fmaf(ts[mt][0], __int_as_float(acc[mt][i][1]), fmaf(o[1], al[mt][0], k0));
          o[2] = fmaf(ts[mt][1], __int_as_float(acc[mt][i][2]), fmaf(o[2], al[mt][1], k1));
          o[3] = fmaf(ts[mt][1], __int_as_float(acc[mt][i][3]), fmaf(o[3], al[mt][1], k1));
        }
      }
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(&empty[sl]);
  }

  // ---- epilogue: row sums are per key-half; combine across the pair
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      ll[mt][r] += __shfl_xor_sync(0xffffffff, ll[mt][r], 1);
      ll[mt][r] += __shfl_xor_sync(0xffffffff, ll[mt][r], 2);
    }
  named_sync(barId, 64);  // partner done reading tile maxima
  if (tq == 0) {
    tmMine[g] = ll[0][0]; tmMine[g + 8] = ll[0][1]; tmMine[16 + g] = ll[1][0]; tmMine[24 + g] = ll[1][1];
  }
  named_sync(barId, 64);
  float il[2][2];
  il[0][0] = 1.f / (ll[0][0] + tmPart[g]); il[0][1] = 1.f / (ll[0][1] + tmPart[g + 8]);
  il[1][0] = 1.f / (ll[1][0] + tmPart[16 + g]); il[1][1] = 1.f / (ll[1][1] + tmPart[24 + g]);
  const float* vs = VS + h * D;
  const float* vm = VM + h * D;
#pragma unroll
  for (int mt = 0; mt < 2; ++mt) {
    const int ra = rb + mt * 16 + g;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int d = hf * 64 + i * 8 + tq * 2;
      const float2 s2 = *reinterpret_cast<const float2*>(vs + d);
      const float2 m2 = *reinterpret_cast<const float2*>(vm + d);
      if (ra < S)
        *reinterpret_cast<__nv_bfloat162*>(O + ra * o_ss + h * o_sh + d) = __floats2bfloat162_rn(
            fmaf(of[mt][i][0] * il[mt][0], s2.x, m2.x), fmaf(of[mt][i][1] * il[mt][0], s2.y, m2.y));
      if (ra + 8 < S)
        *reinterpret_cast<__nv_bfloat162*>(O + (ra + 8) * o_ss + h * o_sh + d) = __floats2bfloat162_rn(
            fmaf(of[mt][i][2] * il[mt][1], s2.x, m2.x), fmaf(of[mt][i][3] * il[mt][1], s2.y, m2.y));
    }
  }
}

template <int STAGES, int MODE>
void launch_fwd4(const torch::Tensor& q8, const torch::Tensor& qs, const torch::Tensor& k8, const torch::Tensor& ksb,
                 const torch::Tensor& vt, const torch::Tensor& vs, const torch::Tensor& vm, torch::Tensor& out,
                 int S, int Spad, float scale_log2, cudaStream_t stream) {
  using C = Cfg4<STAGES>;
  const int H = q8.size(0);
  auto kern = attn_fwd4<STAGES, MODE>;
  static bool once = [&] {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
    return true;
  }();
  (void)once;
  dim3 grid((S + C::BM - 1) / C::BM, H);
  kern<<<grid, 256, C::SMEM, stream>>>(
      q8.data_ptr<int8_t>(), qs.data_ptr<float>(), k8.data_ptr<int8_t>(),
      reinterpret_cast<const float2*>(ksb.data_ptr<float>()), vt.data_ptr<int8_t>(), vs.data_ptr<float>(),
      vm.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), out.stride(0), out.stride(1), S, Spad,
      scale_log2);
}

// ------------------------------------------------------------------------------------------------
// FP8 P.V variant: P (e4m3, relative to the running row max, scaled to 448) times V (e4m3,
// per-channel) accumulates straight into the fp32 output fragments, so there is no per-tile fold;
// the output is only rescaled (warp-uniform branch) when a row max grows. Q.K^T stays INT8.
// TWO: two-level accumulation (fresh accumulator per tile, then one FADD into O) in case the
// e4m3 MMA accumulates with reduced precision.
// ------------------------------------------------------------------------------------------------
__device__ __forceinline__ void mma_e4m3(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_e4m3x4(float a, float b, float c, float d) {
  uint32_t r;
  asm("{\n .reg .b16 lo, hi;\n"
      " cvt.rn.satfinite.e4m3x2.f32 lo, %2, %1;\n"
      " cvt.rn.satfinite.e4m3x2.f32 hi, %4, %3;\n"
      " mov.b32 %0, {lo, hi};\n}\n" : "=r"(r) : "f"(a), "f"(b), "f"(c), "f"(d));
  return r;
}
constexpr float LOG2_448 = 8.807354922057604f;

template <int BN, int STAGES, int NW, bool PIPE, bool TWO>
__global__ void __launch_bounds__(NW * 32, 1)
attn_fwd5(const int8_t* __restrict__ Qg, const float* __restrict__ QS, const int8_t* __restrict__ Kg,
          const float2* __restrict__ KSB, const int8_t* __restrict__ Vt, const float* __restrict__ VS,
          const float* __restrict__ VM, __nv_bfloat16* __restrict__ O, long o_ss, long o_sh, int S, int Spad,
          float scale_log2) {
  static_assert(!PIPE || STAGES >= 3, "pipelined Q.K^T needs tiles i and i+1 resident");
  using C = Cfg2<BN, STAGES, NW>;
  constexpr int NB = BN / 8, KC = BN / 32, NT = NW * 32;
  extern __shared__ __align__(128) uint8_t smem[];

  const int h = blockIdx.y, m0 = blockIdx.x * C::BM;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tq = lane & 3;
  const int8_t* Kh = Kg + (long)h * Spad * D;
  const int8_t* Vh = Vt + (long)h * D * Spad;
  const float2* KSh = KSB + (long)h * Spad;
  const int ntiles = (S + BN - 1) / BN;

  auto load_tile = [&](int it) {
    uint8_t* st = smem + (it % STAGES) * C::STAGE;
    const int j0 = it * BN;
#pragma unroll
    for (int i = tid; i < BN * 8; i += NT) {
      const int r = i >> 3, c = i & 7;
      cp_async16(st + swz<128>(r, c), Kh + (long)(j0 + r) * D + c * 16);
    }
    constexpr int CPR = BN / 16;
#pragma unroll
    for (int i = tid; i < D * CPR; i += NT) {
      const int r = i / CPR, c = i % CPR;
      cp_async16(st + C::K_BYTES + swz<BN>(r, c), Vh + (long)r * Spad + j0 + c * 16);
    }
    for (int i = tid; i < BN / 2; i += NT) cp_async16(st + C::K_BYTES + C::V_BYTES + i * 16, KSh + j0 + i * 2);
  };

  const int r0 = m0 + warp * 16 + g;
  uint32_t qa[4][4];
  auto qk = [&](int (&sc)[NB][4], const uint8_t* sK) {
#pragma unroll
    for (int nb = 0; nb < NB; ++nb) sc[nb][0] = sc[nb][1] = sc[nb][2] = sc[nb][3] = MAGIC_I;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
      for (int nb = 0; nb < NB; nb += 2) {
        uint32_t b0, b1, b2, b3;
        const int m = lane >> 3;
        ldsm_x4(b0, b1, b2, b3, sK + swz<128>((nb + (m >> 1)) * 8 + (lane & 7), kk * 2 + (m & 1)));
        mma_s8(sc[nb], qa[kk], b0, b1);
        mma_s8(sc[nb + 1], qa[kk], b2, b3);
      }
    }
  };

#pragma unroll
  for (int i = 0; i < STAGES - 1; ++i) {
    if (i < ntiles) load_tile(i);
    cp_async_commit();
  }
  {
    const int8_t* q0 = Qg + ((long)h * Spad + r0) * D;
    const int8_t* q1 = q0 + 8 * D;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      qa[k][0] = *reinterpret_cast<const uint32_t*>(q0 + k * 32 + tq * 4);
      qa[k][1] = *reinterpret_cast<const uint32_t*>(q1 + k * 32 + tq * 4);
      qa[k][2] = *reinterpret_cast<const uint32_t*>(q0 + k * 32 + 16 + tq * 4);
      qa[k][3] = *reinterpret_cast<const uint32_t*>(q1 + k * 32 + 16 + tq * 4);
    }
  }
  const float c0 = QS[(long)h * Spad + r0] * scale_log2, c1 = QS[(long)h * Spad + r0 + 8] * scale_log2;

  float of[16][4];
#pragma unroll
  for (int i = 0; i < 16; ++i) of[i][0] = of[i][1] = of[i][2] = of[i][3] = 0.f;
  float mx0 = -INFINITY, mx1 = -INFINITY, l0 = 0.f, l1 = 0.f;

  int sc[NB][4];
  if constexpr (PIPE) {
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    qk(sc, smem);
  }

  for (int it = 0; it < ntiles; ++it) {
    if constexpr (PIPE) cp_async_wait<STAGES - 3>();
    else cp_async_wait<STAGES - 2>();
    __syncthreads();
    if (it + STAGES - 1 < ntiles) load_tile(it + STAGES - 1);
    cp_async_commit();
    const uint8_t* st = smem + (it % STAGES) * C::STAGE;
    const uint8_t* sV = st + C::K_BYTES;
    const float* sKS = reinterpret_cast<const float*>(sV + C::V_BYTES);

    int sn[NB][4];
    if constexpr (PIPE) qk(sn, smem + ((it + 1) % STAGES) * C::STAGE);
    else qk(sc, st);

    float t[NB][4];
    float tm0 = -INFINITY, tm1 = -INFINITY;
#pragma unroll
    for (int nb = 0; nb < NB; ++nb) {
      const float4 kb = *reinterpret_cast<const float4*>(sKS + (nb * 8 + tq * 2) * 2);
      t[nb][0] = fmaf(__int_as_float(sc[nb][0]), kb.x, kb.y);
      t[nb][1] = fmaf(__int_as_float(sc[nb][1]), kb.z, kb.w);
      t[nb][2] = fmaf(__int_as_float(sc[nb][2]), kb.x, kb.y);
      t[nb][3] = fmaf(__int_as_float(sc[nb][3]), kb.z, kb.w);
      tm0 = fmaxf(tm0, fmaxf(t[nb][0], t[nb][1]));
      tm1 = fmaxf(tm1, fmaxf(t[nb][2], t[nb][3]));
    }
    tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffff, tm0, 1));
    tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffff, tm0, 2));
    tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffff, tm1, 1));
    tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffff, tm1, 2));
    if (__any_sync(0xffffffff, tm0 > mx0 || tm1 > mx1)) {
      const float n0 = fmaxf(mx0, tm0), n1 = fmaxf(mx1, tm1);
      const float a0 = ex2((mx0 - n0) * c0), a1 = ex2((mx1 - n1) * c1);
#pragma unroll
      for (int i = 0; i < 16; ++i) { of[i][0] *= a0; of[i][1] *= a0; of[i][2] *= a1; of[i][3] *= a1; }
      l0 *= a0; l1 *= a1;
      mx0 = n0; mx1 = n1;
    }
    const float b0 = fmaf(-mx0, c0, LOG2_448), b1 = fmaf(-mx1, c1, LOG2_448);
    uint32_t pa[KC][4];
#pragma unroll
    for (int kc = 0; kc < KC; ++kc) {
      float p[4][4];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int nb = kc * 4 + u;
        p[u][0] = ex2(fmaf(t[nb][0], c0, b0)); p[u][1] = ex2(fmaf(t[nb][1], c0, b0));
        p[u][2] = ex2(fmaf(t[nb][2], c1, b1)); p[u][3] = ex2(fmaf(t[nb][3], c1, b1));
        l0 += p[u][0] + p[u][1];
        l1 += p[u][2] + p[u][3];
      }
      pa[kc][0] = pack_e4m3x4(p[0][0], p[0][1], p[1][0], p[1][1]);
      pa[kc][1] = pack_e4m3x4(p[0][2], p[0][3], p[1][2], p[1][3]);
      pa[kc][2] = pack_e4m3x4(p[2][0], p[2][1], p[3][0], p[3][1]);
      pa[kc][3] = pack_e4m3x4(p[2][2], p[2][3], p[3][2], p[3][3]);
    }

    if constexpr (TWO) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        float acc[8][4];
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
#pragma unroll
        for (int kc = 0; kc < KC; ++kc)
#pragma unroll
          for (int nd = 0; nd < 8; nd += 2) {
            uint32_t v0, v1, v2, v3;
            const int m = lane >> 3;
            ldsm_x4(v0, v1, v2, v3, sV + swz<BN>((half * 8 + nd + (m >> 1)) * 8 + (lane & 7), kc * 2 + (m & 1)));
            mma_e4m3(acc[nd], pa[kc], v0, v1);
            mma_e4m3(acc[nd + 1], pa[kc], v2, v3);
          }
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
          for (int j = 0; j < 4; ++j) of[half * 8 + i][j] += acc[i][j];
      }
    } else {
#pragma unroll
      for (int kc = 0; kc < KC; ++kc)
#pragma unroll
        for (int nd = 0; nd < 16; nd += 2) {
          uint32_t v0, v1, v2, v3;
          const int m = lane >> 3;
          ldsm_x4(v0, v1, v2, v3, sV + swz<BN>((nd + (m >> 1)) * 8 + (lane & 7), kc * 2 + (m & 1)));
          mma_e4m3(of[nd], pa[kc], v0, v1);
          mma_e4m3(of[nd + 1], pa[kc], v2, v3);
        }
    }
    if constexpr (PIPE) {
#pragma unroll
      for (int nb = 0; nb < NB; ++nb)
#pragma unroll
        for (int j = 0; j < 4; ++j) sc[nb][j] = sn[nb][j];
    }
  }
  cp_async_wait<0>();

  l0 += __shfl_xor_sync(0xffffffff, l0, 1); l0 += __shfl_xor_sync(0xffffffff, l0, 2);
  l1 += __shfl_xor_sync(0xffffffff, l1, 1); l1 += __shfl_xor_sync(0xffffffff, l1, 2);
  const float il0 = 1.f / l0, il1 = 1.f / l1;
  const float* vs = VS + h * D;
  const float* vm = VM + h * D;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int d = i * 8 + tq * 2;
    const float2 s2 = *reinterpret_cast<const float2*>(vs + d);
    const float2 m2 = *reinterpret_cast<const float2*>(vm + d);
    if (r0 < S)
      *reinterpret_cast<__nv_bfloat162*>(O + r0 * o_ss + h * o_sh + d) =
          __floats2bfloat162_rn(fmaf(of[i][0] * il0, s2.x, m2.x), fmaf(of[i][1] * il0, s2.y, m2.y));
    if (r0 + 8 < S)
      *reinterpret_cast<__nv_bfloat162*>(O + (r0 + 8) * o_ss + h * o_sh + d) =
          __floats2bfloat162_rn(fmaf(of[i][2] * il1, s2.x, m2.x), fmaf(of[i][3] * il1, s2.y, m2.y));
  }
}

template <int BN, int STAGES, int NW, bool PIPE, bool TWO>
void launch_fwd5(const torch::Tensor& q8, const torch::Tensor& qs, const torch::Tensor& k8, const torch::Tensor& ksb,
                 const torch::Tensor& vt, const torch::Tensor& vs, const torch::Tensor& vm, torch::Tensor& out,
                 int S, int Spad, float scale_log2, cudaStream_t stream) {
  using C = Cfg2<BN, STAGES, NW>;
  const int H = q8.size(0);
  auto kern = attn_fwd5<BN, STAGES, NW, PIPE, TWO>;
  static bool once = [&] {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
    return true;
  }();
  (void)once;
  dim3 grid((S + C::BM - 1) / C::BM, H);
  kern<<<grid, NW * 32, C::SMEM, stream>>>(
      q8.data_ptr<int8_t>(), qs.data_ptr<float>(), k8.data_ptr<int8_t>(),
      reinterpret_cast<const float2*>(ksb.data_ptr<float>()), vt.data_ptr<int8_t>(), vs.data_ptr<float>(),
      vm.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), out.stride(0), out.stride(1), S, Spad,
      scale_log2);
}

// ------------------------------------------------------------------------------------------------
// attn_fwd6: attn_fwd's layout (8 warps x 16 rows, BN=128, cp.async) with comfy-kitchen-style lean
// softmax: one K scale per 128-key tile, so the row max is an integer max over the raw accumulators
// and each score costs one FFMA before exp2 (no per-key dequantization, no float max).
// ------------------------------------------------------------------------------------------------
template <int STAGES>
__global__ void __launch_bounds__(256, 1)
attn_fwd6(const int8_t* __restrict__ Qg, const float* __restrict__ QS, const int8_t* __restrict__ Kg,
          const float* __restrict__ KBS, const int8_t* __restrict__ Vt, const float* __restrict__ VS,
          const float* __restrict__ VM, __nv_bfloat16* __restrict__ O, long o_ss, long o_sh, int S, int Spad,
          float scale_log2) {
  constexpr int BN = 128, NW = 8, BM = 128, NB = BN / 8, KC = BN / 32, NT = NW * 32;
  constexpr int K_BYTES = BN * D, STAGE = K_BYTES + D * BN;
  extern __shared__ __align__(128) uint8_t smem[];

  const int h = blockIdx.y, m0 = blockIdx.x * BM;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tq = lane & 3;
  const int8_t* Kh = Kg + (long)h * Spad * D;
  const int8_t* Vh = Vt + (long)h * D * Spad;
  const float* KBh = KBS + (long)h * (Spad / 128);
  const int ntiles = (S + BN - 1) / BN;

  auto load_tile = [&](int it) {
    uint8_t* st = smem + (it % STAGES) * STAGE;
    const int j0 = it * BN;
#pragma unroll
    for (int i = tid; i < BN * 8; i += NT) {
      const int r = i >> 3, c = i & 7;
      cp_async16(st + swz<128>(r, c), Kh + (long)(j0 + r) * D + c * 16);
    }
#pragma unroll
    for (int i = tid; i < D * (BN / 16); i += NT) {
      const int r = i / (BN / 16), c = i % (BN / 16);
      cp_async16(st + K_BYTES + swz<BN>(r, c), Vh + (long)r * Spad + j0 + c * 16);
    }
  };
#pragma unroll
  for (int i = 0; i < STAGES - 1; ++i) {
    if (i < ntiles) load_tile(i);
    cp_async_commit();
  }

  const int r0 = m0 + warp * 16 + g;
  uint32_t qa[4][4];
  {
    const int8_t* q0 = Qg + ((long)h * Spad + r0) * D;
    const int8_t* q1 = q0 + 8 * D;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      qa[k][0] = *reinterpret_cast<const uint32_t*>(q0 + k * 32 + tq * 4);
      qa[k][1] = *reinterpret_cast<const uint32_t*>(q1 + k * 32 + tq * 4);
      qa[k][2] = *reinterpret_cast<const uint32_t*>(q0 + k * 32 + 16 + tq * 4);
      qa[k][3] = *reinterpret_cast<const uint32_t*>(q1 + k * 32 + 16 + tq * 4);
    }
  }
  const float c0 = QS[(long)h * Spad + r0] * scale_log2, c1 = QS[(long)h * Spad + r0 + 8] * scale_log2;

  float of[16][4];
#pragma unroll
  for (int i = 0; i < 16; ++i) of[i][0] = of[i][1] = of[i][2] = of[i][3] = 0.f;
  float mx0 = -INFINITY, mx1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  const int magic = MAGIC_I;

  for (int it = 0; it < ntiles; ++it) {
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    if (it + STAGES - 1 < ntiles) load_tile(it + STAGES - 1);
    cp_async_commit();
    const uint8_t* sK = smem + (it % STAGES) * STAGE;
    const uint8_t* sV = sK + K_BYTES;
    const float kb = __ldg(KBh + it);

    int sc[NB][4];
#pragma unroll
    for (int nb = 0; nb < NB; ++nb) sc[nb][0] = sc[nb][1] = sc[nb][2] = sc[nb][3] = MAGIC_I;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
      for (int nb = 0; nb < NB; nb += 2) {
        uint32_t b0, b1, b2, b3;
        const int m = lane >> 3;
        ldsm_x4(b0, b1, b2, b3, sK + swz<128>((nb + (m >> 1)) * 8 + (lane & 7), kk * 2 + (m & 1)));
        mma_s8(sc[nb], qa[kk], b0, b1);
        mma_s8(sc[nb + 1], qa[kk], b2, b3);
      }
    }
    if ((it + 1) * BN > S) {  // tail tile: push padded keys far below any real score
#pragma unroll
      for (int nb = 0; nb < NB; ++nb) {
        const int j = it * BN + nb * 8 + tq * 2;
        if (j >= S) sc[nb][0] = sc[nb][2] = MAGIC_I - (1 << 22);
        if (j + 1 >= S) sc[nb][1] = sc[nb][3] = MAGIC_I - (1 << 22);
      }
    }
    int ti0 = sc[0][0], ti1 = sc[0][2];
#pragma unroll
    for (int nb = 0; nb < NB; ++nb) {
      ti0 = max(ti0, max(sc[nb][0], sc[nb][1]));
      ti1 = max(ti1, max(sc[nb][2], sc[nb][3]));
    }
    ti0 = max(ti0, __shfl_xor_sync(0xffffffff, ti0, 1)); ti0 = max(ti0, __shfl_xor_sync(0xffffffff, ti0, 2));
    ti1 = max(ti1, __shfl_xor_sync(0xffffffff, ti1, 1)); ti1 = max(ti1, __shfl_xor_sync(0xffffffff, ti1, 2));
    const float cc0 = c0 * kb, cc1 = c1 * kb;                         // log2 units per integer score unit
    const float tv0 = (__int_as_float(ti0) - MAGIC_F) * cc0, tv1 = (__int_as_float(ti1) - MAGIC_F) * cc1;
    const float n0 = fmaxf(mx0, tv0), n1 = fmaxf(mx1, tv1);
    const float a0 = ex2(mx0 - n0), a1 = ex2(mx1 - n1);
    const float ts0 = ex2(tv0 - n0), ts1 = ex2(tv1 - n1);
    mx0 = n0; mx1 = n1;
    const float b0 = fmaf(-__int_as_float(ti0), cc0, LOG2_255), b1 = fmaf(-__int_as_float(ti1), cc1, LOG2_255);
    uint32_t pa[KC][4];
    uint32_t rs0 = 0, rs1 = 0;
#pragma unroll
    for (int kc = 0; kc < KC; ++kc) {
      int q[4][4];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int nb = kc * 4 + u;
        q[u][0] = __float_as_int(ex2(fmaf(__int_as_float(sc[nb][0]), cc0, b0)) + MAGIC_F);
        q[u][1] = __float_as_int(ex2(fmaf(__int_as_float(sc[nb][1]), cc0, b0)) + MAGIC_F);
        q[u][2] = __float_as_int(ex2(fmaf(__int_as_float(sc[nb][2]), cc1, b1)) + MAGIC_F);
        q[u][3] = __float_as_int(ex2(fmaf(__int_as_float(sc[nb][3]), cc1, b1)) + MAGIC_F);
      }
      pa[kc][0] = pack4(q[0][0], q[0][1], q[1][0], q[1][1]);
      pa[kc][1] = pack4(q[0][2], q[0][3], q[1][2], q[1][3]);
      pa[kc][2] = pack4(q[2][0], q[2][1], q[3][0], q[3][1]);
      pa[kc][3] = pack4(q[2][2], q[2][3], q[3][2], q[3][3]);
      rs0 = __dp4a(pa[kc][0], 0x01010101u, rs0); rs0 = __dp4a(pa[kc][2], 0x01010101u, rs0);
      rs1 = __dp4a(pa[kc][1], 0x01010101u, rs1); rs1 = __dp4a(pa[kc][3], 0x01010101u, rs1);
    }
    l0 = fmaf(ts0, i2f(rs0), l0 * a0);
    l1 = fmaf(ts1, i2f(rs1), l1 * a1);

    const float k0 = -ts0 * MAGIC_F, k1 = -ts1 * MAGIC_F;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      int acc[8][4];
#pragma unroll
      for (int kc = 0; kc < KC; ++kc) {
#pragma unroll
        for (int nd = 0; nd < 8; nd += 2) {
          uint32_t v0, v1, v2, v3;
          const int m = lane >> 3;
          ldsm_x4(v0, v1, v2, v3, sV + swz<BN>((half * 8 + nd + (m >> 1)) * 8 + (lane & 7), kc * 2 + (m & 1)));
          if (kc == 0) {
            mma_u8s8_c(acc[nd], pa[kc], v0, v1, magic);
            mma_u8s8_c(acc[nd + 1], pa[kc], v2, v3, magic);
          } else {
            mma_u8s8(acc[nd], pa[kc], v0, v1);
            mma_u8s8(acc[nd + 1], pa[kc], v2, v3);
          }
        }
      }
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        float* o = of[half * 8 + i];
        o[0] = fmaf(ts0, __int_as_float(acc[i][0]), fmaf(o[0], a0, k0));
        o[1] = fmaf(ts0, __int_as_float(acc[i][1]), fmaf(o[1], a0, k0));
        o[2] = fmaf(ts1, __int_as_float(acc[i][2]), fmaf(o[2], a1, k1));
        o[3] = fmaf(ts1, __int_as_float(acc[i][3]), fmaf(o[3], a1, k1));
      }
    }
  }
  cp_async_wait<0>();

  l0 += __shfl_xor_sync(0xffffffff, l0, 1); l0 += __shfl_xor_sync(0xffffffff, l0, 2);
  l1 += __shfl_xor_sync(0xffffffff, l1, 1); l1 += __shfl_xor_sync(0xffffffff, l1, 2);
  const float il0 = 1.f / l0, il1 = 1.f / l1;
  const float* vs = VS + h * D;
  const float* vm = VM + h * D;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int d = i * 8 + tq * 2;
    const float2 s2 = *reinterpret_cast<const float2*>(vs + d);
    const float2 m2 = *reinterpret_cast<const float2*>(vm + d);
    if (r0 < S)
      *reinterpret_cast<__nv_bfloat162*>(O + r0 * o_ss + h * o_sh + d) =
          __floats2bfloat162_rn(fmaf(of[i][0] * il0, s2.x, m2.x), fmaf(of[i][1] * il0, s2.y, m2.y));
    if (r0 + 8 < S)
      *reinterpret_cast<__nv_bfloat162*>(O + (r0 + 8) * o_ss + h * o_sh + d) =
          __floats2bfloat162_rn(fmaf(of[i][2] * il1, s2.x, m2.x), fmaf(of[i][3] * il1, s2.y, m2.y));
  }
}

template <int STAGES>
void launch_fwd6(const torch::Tensor& q8, const torch::Tensor& qs, const torch::Tensor& k8, const torch::Tensor& kbs,
                 const torch::Tensor& vt, const torch::Tensor& vs, const torch::Tensor& vm, torch::Tensor& out,
                 int S, int Spad, float scale_log2, cudaStream_t stream) {
  constexpr int SMEM = STAGES * (128 * D * 2);
  const int H = q8.size(0);
  auto kern = attn_fwd6<STAGES>;
  static bool once = [&] {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    return true;
  }();
  (void)once;
  dim3 grid((S + 127) / 128, H);
  kern<<<grid, 256, SMEM, stream>>>(
      q8.data_ptr<int8_t>(), qs.data_ptr<float>(), k8.data_ptr<int8_t>(), kbs.data_ptr<float>(),
      vt.data_ptr<int8_t>(), vs.data_ptr<float>(), vm.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), out.stride(0), out.stride(1), S, Spad, scale_log2);
}

// attn_fwd2 with the output fold software-pipelined (2026-10-24). In attn_fwd2 each tile ends with
// ~128 FFMAs (O = a*O + ts*PV) that have no IMMA to pair with, so the warp's tensor issue stops
// while it folds. Here half 0 of the fold is issued between the IMMAs of P.V half 1, and half 1 is
// carried (32 int32 regs) into the next tile, where it interleaves with that tile's Q.K^T and softmax.
// Same arithmetic in the same order per output element as attn_fwd2 (bit-identical output).
template <int BN, int STAGES, int NW>
__global__ void __launch_bounds__(NW * 32, 1)
attn_fwd7(const int8_t* __restrict__ Qg, const float* __restrict__ QS, const int8_t* __restrict__ Kg,
          const float2* __restrict__ KSB, const int8_t* __restrict__ Vt, const float* __restrict__ VS,
          const float* __restrict__ VM, __nv_bfloat16* __restrict__ O, long o_ss, long o_sh, int S, int Spad,
          float scale_log2) {
  static_assert(STAGES >= 3, "tiles i and i+1 must be resident while i+2 loads");
  using C = Cfg2<BN, STAGES, NW>;
  constexpr int NB = BN / 8, KC = BN / 32, NT = NW * 32;
  extern __shared__ __align__(128) uint8_t smem[];

  const int h = blockIdx.y, m0 = blockIdx.x * C::BM;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tq = lane & 3;
  const int8_t* Kh = Kg + (long)h * Spad * D;
  const int8_t* Vh = Vt + (long)h * D * Spad;
  const float2* KSh = KSB + (long)h * Spad;
  const int ntiles = (S + BN - 1) / BN;

  auto load_tile = [&](int it) {
    uint8_t* st = smem + (it % STAGES) * C::STAGE;
    const int j0 = it * BN;
#pragma unroll
    for (int i = tid; i < BN * 8; i += NT) {
      const int r = i >> 3, c = i & 7;
      cp_async16(st + swz<128>(r, c), Kh + (long)(j0 + r) * D + c * 16);
    }
    constexpr int CPR = BN / 16;
#pragma unroll
    for (int i = tid; i < D * CPR; i += NT) {
      const int r = i / CPR, c = i % CPR;
      cp_async16(st + C::K_BYTES + swz<BN>(r, c), Vh + (long)r * Spad + j0 + c * 16);
    }
    for (int i = tid; i < BN / 2; i += NT) cp_async16(st + C::K_BYTES + C::V_BYTES + i * 16, KSh + j0 + i * 2);
  };

  const int r0 = m0 + warp * 16 + g;
  uint32_t qa[4][4];
  auto qk = [&](int (&sc)[NB][4], const uint8_t* sK) {
#pragma unroll
    for (int nb = 0; nb < NB; ++nb) sc[nb][0] = sc[nb][1] = sc[nb][2] = sc[nb][3] = MAGIC_I;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
      for (int nb = 0; nb < NB; nb += 2) {
        uint32_t b0, b1, b2, b3;
        const int m = lane >> 3;
        ldsm_x4(b0, b1, b2, b3, sK + swz<128>((nb + (m >> 1)) * 8 + (lane & 7), kk * 2 + (m & 1)));
        mma_s8(sc[nb], qa[kk], b0, b1);
        mma_s8(sc[nb + 1], qa[kk], b2, b3);
      }
    }
  };

#pragma unroll
  for (int i = 0; i < STAGES - 1; ++i) {
    if (i < ntiles) load_tile(i);
    cp_async_commit();
  }
  {
    const int8_t* q0 = Qg + ((long)h * Spad + r0) * D;
    const int8_t* q1 = q0 + 8 * D;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      qa[k][0] = *reinterpret_cast<const uint32_t*>(q0 + k * 32 + tq * 4);
      qa[k][1] = *reinterpret_cast<const uint32_t*>(q1 + k * 32 + tq * 4);
      qa[k][2] = *reinterpret_cast<const uint32_t*>(q0 + k * 32 + 16 + tq * 4);
      qa[k][3] = *reinterpret_cast<const uint32_t*>(q1 + k * 32 + 16 + tq * 4);
    }
  }
  const float c0 = QS[(long)h * Spad + r0] * scale_log2, c1 = QS[(long)h * Spad + r0 + 8] * scale_log2;

  float of[16][4];
#pragma unroll
  for (int i = 0; i < 16; ++i) of[i][0] = of[i][1] = of[i][2] = of[i][3] = 0.f;
  float mx0 = -INFINITY, mx1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  const int magic = MAGIC_I;
  // pending P.V half 1 of the previous tile and its fold coefficients (a = 1, ts = 0: a no-op)
  int accp[8][4];
#pragma unroll
  for (int i = 0; i < 8; ++i) accp[i][0] = accp[i][1] = accp[i][2] = accp[i][3] = MAGIC_I;
  float pa0 = 1.f, pa1 = 1.f, pts0 = 0.f, pts1 = 0.f;

  auto fold = [&](float* o, const int* acc, float a0, float a1, float ts0, float ts1) {
    const float k0 = -ts0 * MAGIC_F, k1 = -ts1 * MAGIC_F;
    o[0] = fmaf(ts0, __int_as_float(acc[0]), fmaf(o[0], a0, k0));
    o[1] = fmaf(ts0, __int_as_float(acc[1]), fmaf(o[1], a0, k0));
    o[2] = fmaf(ts1, __int_as_float(acc[2]), fmaf(o[2], a1, k1));
    o[3] = fmaf(ts1, __int_as_float(acc[3]), fmaf(o[3], a1, k1));
  };

  cp_async_wait<STAGES - 2>();
  __syncthreads();
  int sc[NB][4];
  qk(sc, smem);

  for (int it = 0; it < ntiles; ++it) {
    cp_async_wait<STAGES - 3>();
    __syncthreads();
    if (it + STAGES - 1 < ntiles) load_tile(it + STAGES - 1);
    cp_async_commit();
    const uint8_t* st = smem + (it % STAGES) * C::STAGE;
    const uint8_t* sV = st + C::K_BYTES;
    const float* sKS = reinterpret_cast<const float*>(sV + C::V_BYTES);

    int sn[NB][4];
    uint32_t pa[KC][4];
    float a0, a1, ts0, ts1;
    qk(sn, smem + ((it + 1) % STAGES) * C::STAGE);
    // previous tile's P.V half 1 (independent of this tile's scores)
#pragma unroll
    for (int i = 0; i < 8; ++i) fold(of[8 + i], accp[i], pa0, pa1, pts0, pts1);
    {
      float t[NB][4];
      float tm0 = -INFINITY, tm1 = -INFINITY;
#pragma unroll
      for (int nb = 0; nb < NB; ++nb) {
        const float4 kb = *reinterpret_cast<const float4*>(sKS + (nb * 8 + tq * 2) * 2);
        t[nb][0] = fmaf(__int_as_float(sc[nb][0]), kb.x, kb.y);
        t[nb][1] = fmaf(__int_as_float(sc[nb][1]), kb.z, kb.w);
        t[nb][2] = fmaf(__int_as_float(sc[nb][2]), kb.x, kb.y);
        t[nb][3] = fmaf(__int_as_float(sc[nb][3]), kb.z, kb.w);
        tm0 = fmaxf(tm0, fmaxf(t[nb][0], t[nb][1]));
        tm1 = fmaxf(tm1, fmaxf(t[nb][2], t[nb][3]));
      }
      tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffff, tm0, 1));
      tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffff, tm0, 2));
      tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffff, tm1, 1));
      tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffff, tm1, 2));
      const float n0 = fmaxf(mx0, tm0), n1 = fmaxf(mx1, tm1);
      a0 = ex2((mx0 - n0) * c0); a1 = ex2((mx1 - n1) * c1);
      ts0 = ex2((tm0 - n0) * c0); ts1 = ex2((tm1 - n1) * c1);
      mx0 = n0; mx1 = n1;
      const float b0 = fmaf(-tm0, c0, LOG2_255), b1 = fmaf(-tm1, c1, LOG2_255);
      uint32_t rs0 = 0, rs1 = 0;
#pragma unroll
      for (int kc = 0; kc < KC; ++kc) {
        int q[4][4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          const int nb = kc * 4 + u;
          q[u][0] = __float_as_int(ex2(fmaf(t[nb][0], c0, b0)) + MAGIC_F);
          q[u][1] = __float_as_int(ex2(fmaf(t[nb][1], c0, b0)) + MAGIC_F);
          q[u][2] = __float_as_int(ex2(fmaf(t[nb][2], c1, b1)) + MAGIC_F);
          q[u][3] = __float_as_int(ex2(fmaf(t[nb][3], c1, b1)) + MAGIC_F);
        }
        pa[kc][0] = pack4(q[0][0], q[0][1], q[1][0], q[1][1]);
        pa[kc][1] = pack4(q[0][2], q[0][3], q[1][2], q[1][3]);
        pa[kc][2] = pack4(q[2][0], q[2][1], q[3][0], q[3][1]);
        pa[kc][3] = pack4(q[2][2], q[2][3], q[3][2], q[3][3]);
        rs0 = __dp4a(pa[kc][0], 0x01010101u, rs0); rs0 = __dp4a(pa[kc][2], 0x01010101u, rs0);
        rs1 = __dp4a(pa[kc][1], 0x01010101u, rs1); rs1 = __dp4a(pa[kc][3], 0x01010101u, rs1);
      }
      l0 = fmaf(ts0, i2f(rs0), l0 * a0);
      l1 = fmaf(ts1, i2f(rs1), l1 * a1);
    }

    // P.V half 0 is folded right away (between half 1's IMMAs); half 1 is carried to the next tile
    int acc0[8][4];
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      int (&acc)[8][4] = half ? accp : acc0;
#pragma unroll
      for (int kc = 0; kc < KC; ++kc) {
#pragma unroll
        for (int nd = 0; nd < 8; nd += 2) {
          uint32_t v0, v1, v2, v3;
          const int m = lane >> 3;
          ldsm_x4(v0, v1, v2, v3, sV + swz<BN>((half * 8 + nd + (m >> 1)) * 8 + (lane & 7), kc * 2 + (m & 1)));
          if (kc == 0) {
            mma_u8s8_c(acc[nd], pa[kc], v0, v1, magic);
            mma_u8s8_c(acc[nd + 1], pa[kc], v2, v3, magic);
          } else {
            mma_u8s8(acc[nd], pa[kc], v0, v1);
            mma_u8s8(acc[nd + 1], pa[kc], v2, v3);
          }
        }
      }
      if (half == 0) {
#pragma unroll
        for (int i = 0; i < 8; ++i) fold(of[i], acc0[i], a0, a1, ts0, ts1);
      }
    }
    pa0 = a0; pa1 = a1; pts0 = ts0; pts1 = ts1;
#pragma unroll
    for (int nb = 0; nb < NB; ++nb)
#pragma unroll
      for (int j = 0; j < 4; ++j) sc[nb][j] = sn[nb][j];
  }
  cp_async_wait<0>();
#pragma unroll
  for (int i = 0; i < 8; ++i) fold(of[8 + i], accp[i], pa0, pa1, pts0, pts1);

  l0 += __shfl_xor_sync(0xffffffff, l0, 1); l0 += __shfl_xor_sync(0xffffffff, l0, 2);
  l1 += __shfl_xor_sync(0xffffffff, l1, 1); l1 += __shfl_xor_sync(0xffffffff, l1, 2);
  const float il0 = 1.f / l0, il1 = 1.f / l1;
  const float* vs = VS + h * D;
  const float* vm = VM + h * D;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int d = i * 8 + tq * 2;
    const float2 s2 = *reinterpret_cast<const float2*>(vs + d);
    const float2 m2 = *reinterpret_cast<const float2*>(vm + d);
    if (r0 < S)
      *reinterpret_cast<__nv_bfloat162*>(O + r0 * o_ss + h * o_sh + d) =
          __floats2bfloat162_rn(fmaf(of[i][0] * il0, s2.x, m2.x), fmaf(of[i][1] * il0, s2.y, m2.y));
    if (r0 + 8 < S)
      *reinterpret_cast<__nv_bfloat162*>(O + (r0 + 8) * o_ss + h * o_sh + d) =
          __floats2bfloat162_rn(fmaf(of[i][2] * il1, s2.x, m2.x), fmaf(of[i][3] * il1, s2.y, m2.y));
  }
}

template <int BN, int STAGES, int NW>
void launch_fwd7(const torch::Tensor& q8, const torch::Tensor& qs, const torch::Tensor& k8, const torch::Tensor& ksb,
                 const torch::Tensor& vt, const torch::Tensor& vs, const torch::Tensor& vm, torch::Tensor& out,
                 int S, int Spad, float scale_log2, cudaStream_t stream) {
  using C = Cfg2<BN, STAGES, NW>;
  const int H = q8.size(0);
  auto kern = attn_fwd7<BN, STAGES, NW>;
  static bool once = [&] {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
    return true;
  }();
  (void)once;
  dim3 grid((S + C::BM - 1) / C::BM, H);
  kern<<<grid, NW * 32, C::SMEM, stream>>>(
      q8.data_ptr<int8_t>(), qs.data_ptr<float>(), k8.data_ptr<int8_t>(),
      reinterpret_cast<const float2*>(ksb.data_ptr<float>()), vt.data_ptr<int8_t>(), vs.data_ptr<float>(),
      vm.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), out.stride(0), out.stride(1), S, Spad,
      scale_log2);
}

// ------------------------------------------------------------------------------------------------
// attn_fwd8 (2026-10-24, round 4): attn_fwd's layout (8 warps x 16 rows, BN=128) with the work
// reordered so every phase of a tile has IMMAs to issue, as comfy-kitchen's kernel does:
//   - Q.K^T runs key-pair-outer, so each 16-key score block is dequantized (per-key (scale, bias)
//     FFMA, attn_fwd2's) and max-reduced while the next block's IMMAs issue;
//   - exp2 + u8 rounding of 32-key chunk kc interleaves with the P.V IMMAs of chunk kc-1; P is
//     rounded and packed by cvt.pack.sat.u8 (F2IP, 2 values per instruction) instead of a magic
//     FADD plus PRMTs;
//   - the output fold of D-half 0 interleaves with P.V of half 1; with DEFER, half 1's int32
//     accumulator (32 regs) is carried into the next tile and folded between its Q.K^T IMMAs.
// All ldmatrix addresses are one per-tile base plus 4 hoisted per-lane offsets and immediates.
// Same quantities as attn_fwd2 (per-token Q/K scales, Hadamard, smoothing, per-tile u8 P).
// ------------------------------------------------------------------------------------------------
__device__ __forceinline__ void ldsm_x4a(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ float4 lds128(uint32_t a) {
  float4 v;
  asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];\n" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(a));
  return v;
}
// bytes (x0, x1, x2, x3), each round-to-nearest and saturated to [0, 255] (F2IP.U8 on sm_12x)
__device__ __forceinline__ uint32_t pack_u8(float x0, float x1, float x2, float x3) {
  uint32_t hi, d;
  asm("cvt.pack.sat.u8.s32.b32 %0, %1, %2, 0;" : "=r"(hi) : "r"(__float2int_rn(x3)), "r"(__float2int_rn(x2)));
  asm("cvt.pack.sat.u8.s32.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(__float2int_rn(x1)), "r"(__float2int_rn(x0)), "r"(hi));
  return d;
}

// INC: load/read positions advance incrementally (no spills at 3 stages) instead of from the tile index
template <int STAGES, bool DEFER, bool INC, int VAR = 0>
__global__ void __launch_bounds__(256, 1)
attn_fwd8(const int8_t* __restrict__ Qg, const float* __restrict__ QS, const int8_t* __restrict__ Kg,
          const float2* __restrict__ KSB, const int8_t* __restrict__ Vt, const float* __restrict__ VS,
          const float* __restrict__ VM, __nv_bfloat16* __restrict__ O, long o_ss, long o_sh, int S, int Spad,
          float scale_log2) {
  static_assert(!(DEFER && (VAR & 1)), "the deferred fold rides on the key-pair-outer Q.K^T");
  constexpr int BN = 128, NB = 16, KC = 4, NT = 256, BM = 128;
  constexpr int K_BYTES = BN * D, V_BYTES = D * BN, STAGE = K_BYTES + V_BYTES + BN * 8;
  extern __shared__ __align__(128) uint8_t smem[];

  const int h = blockIdx.y, m0 = blockIdx.x * BM;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tq = lane & 3;
  const int8_t* Kh = Kg + (long)h * Spad * D;
  const int8_t* Vh = Vt + (long)h * D * Spad;
  const float2* KSh = KSB + (long)h * Spad;
  const int ntiles = (S + BN - 1) / BN;
  const uint32_t s0 = smem_u32(smem);

  // cp.async: thread covers rows tid/8 + 32j (j < 4) of K and of Vt, 16 B chunk tid%8; rows of both
  // are 128 B, so the swizzled smem offset is the same for K and V and steps by 4096 per j.
  const int lr = tid >> 3, lc = tid & 7;
  // load position (global pointers of the next tile to load, its smem stage) advances incrementally
  uint32_t ld_st = s0 + lr * 128 + ((lc ^ (lr & 7)) << 4);
  const int8_t* kg = Kh + (long)lr * D + lc * 16;
  const int8_t* vg = Vh + (long)lr * Spad + lc * 16;
  const float2* ksg = KSh + (tid & 63) * 2;
  const long vstep = 32L * Spad;
  auto load_tile = [&](int it) {  // !INC: addresses from the tile index
    const uint32_t st = s0 + (it % STAGES) * STAGE + lr * 128 + ((lc ^ (lr & 7)) << 4);
    const int j0 = it * BN;
    const int8_t* kp = Kh + (long)(j0 + lr) * D + lc * 16;
    const int8_t* vp = Vh + (long)lr * Spad + j0 + lc * 16;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(st + j * 4096), "l"(kp + j * 32 * D));
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(st + K_BYTES + j * 4096), "l"(vp + (long)j * 32 * Spad));
    }
    if (tid < BN / 2)
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s0 + (it % STAGES) * STAGE + K_BYTES + V_BYTES + tid * 16),
                   "l"(KSh + j0 + tid * 2));
  };
  auto load_next = [&]() {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(ld_st + j * 4096), "l"(kg + j * 32 * D));
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(ld_st + K_BYTES + j * 4096), "l"(vg + j * vstep));
    }
    if (tid < BN / 2)
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(ld_st - (lr * 128 + ((lc ^ (lr & 7)) << 4)) + K_BYTES + V_BYTES + tid * 16), "l"(ksg));
    kg += BN * D; vg += BN; ksg += BN;
    ld_st += STAGE;
    if (ld_st >= s0 + STAGES * STAGE) ld_st -= STAGES * STAGE;
  };

  // ldmatrix x4 lane address inside a 128 B-row swizzled tile: rows (m>>1)*8 + lane%8 (+ 16 per
  // n-pair), 16 B chunk 2*k + (m&1) for k-step k (32 keys of K / 32 keys of Vt's row).
  uint32_t off[4];
  {
    const int m = lane >> 3, r = (m >> 1) * 8 + (lane & 7);
#pragma unroll
    for (int k = 0; k < 4; ++k) off[k] = r * 128 + (((k * 2 + (m & 1)) ^ (lane & 7)) << 4);
  }

#pragma unroll
  for (int i = 0; i < STAGES - 1; ++i) {
    if (i < ntiles) { if constexpr (INC) load_next(); else load_tile(i); }
    cp_async_commit();
  }

  const int r0 = m0 + warp * 16 + g;
  uint32_t qa[4][4];
  {
    const int8_t* q0 = Qg + ((long)h * Spad + r0) * D;
    const int8_t* q1 = q0 + 8 * D;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      qa[k][0] = *reinterpret_cast<const uint32_t*>(q0 + k * 32 + tq * 4);
      qa[k][1] = *reinterpret_cast<const uint32_t*>(q1 + k * 32 + tq * 4);
      qa[k][2] = *reinterpret_cast<const uint32_t*>(q0 + k * 32 + 16 + tq * 4);
      qa[k][3] = *reinterpret_cast<const uint32_t*>(q1 + k * 32 + 16 + tq * 4);
    }
  }
  const float c0 = QS[(long)h * Spad + r0] * scale_log2, c1 = QS[(long)h * Spad + r0 + 8] * scale_log2;

  float of[16][4];
#pragma unroll
  for (int i = 0; i < 16; ++i) of[i][0] = of[i][1] = of[i][2] = of[i][3] = 0.f;
  float mx0 = -INFINITY, mx1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  const int magic = MAGIC_I;
  uint32_t sK = s0;  // INC: stage being read
  // carried half-1 accumulator and the scalars of the tile it belongs to (DEFER)
  int acc1[8][4];
  float pa0 = 1.f, pa1 = 1.f, pt0 = 0.f, pt1 = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) acc1[i][0] = acc1[i][1] = acc1[i][2] = acc1[i][3] = MAGIC_I;
  auto fold = [&](float* o, const int* a, float ts0, float ts1, float a0, float a1) {
    o[0] = fmaf(ts0, __int_as_float(a[0]), fmaf(o[0], a0, -ts0 * MAGIC_F));
    o[1] = fmaf(ts0, __int_as_float(a[1]), fmaf(o[1], a0, -ts0 * MAGIC_F));
    o[2] = fmaf(ts1, __int_as_float(a[2]), fmaf(o[2], a1, -ts1 * MAGIC_F));
    o[3] = fmaf(ts1, __int_as_float(a[3]), fmaf(o[3], a1, -ts1 * MAGIC_F));
  };

  for (int it = 0; it < ntiles; ++it) {
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    if (it + STAGES - 1 < ntiles) { if constexpr (INC) load_next(); else load_tile(it + STAGES - 1); }
    cp_async_commit();
    if constexpr (!INC) sK = s0 + (it % STAGES) * STAGE;
    const uint32_t sV = sK + K_BYTES, sKS = sV + V_BYTES;

    // ---- S = Q K^T, 16 keys at a time; dequantize + running tile max as each block completes
    // (VAR bit 0: all of Q.K^T first, k-step outer, as attn_fwd; bit 1: two max chains per row)
    float t[NB][4];
    float tm0 = -INFINITY, tm1 = -INFINITY, tm0b = -INFINITY, tm1b = -INFINITY;
    auto dequant = [&](int nb, const int* scv) {
      const float4 kb = lds128(sKS + nb * 64 + tq * 16);  // (s_j, b_j, s_j+1, b_j+1)
      t[nb][0] = fmaf(__int_as_float(scv[0]), kb.x, kb.y);
      t[nb][1] = fmaf(__int_as_float(scv[1]), kb.z, kb.w);
      t[nb][2] = fmaf(__int_as_float(scv[2]), kb.x, kb.y);
      t[nb][3] = fmaf(__int_as_float(scv[3]), kb.z, kb.w);
      if ((VAR & 2) && (nb & 1)) {
        tm0b = fmaxf(tm0b, fmaxf(t[nb][0], t[nb][1]));
        tm1b = fmaxf(tm1b, fmaxf(t[nb][2], t[nb][3]));
      } else {
        tm0 = fmaxf(tm0, fmaxf(t[nb][0], t[nb][1]));
        tm1 = fmaxf(tm1, fmaxf(t[nb][2], t[nb][3]));
      }
    };
    if constexpr (VAR & 1) {
      int sc[NB][4];
#pragma unroll
      for (int nb = 0; nb < NB; ++nb) sc[nb][0] = sc[nb][1] = sc[nb][2] = sc[nb][3] = MAGIC_I;
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
        for (int p = 0; p < NB / 2; ++p) {
          uint32_t b0, b1, b2, b3;
          ldsm_x4a(b0, b1, b2, b3, sK + off[kk] + p * 2048);
          mma_s8(sc[2 * p], qa[kk], b0, b1);
          mma_s8(sc[2 * p + 1], qa[kk], b2, b3);
        }
      }
#pragma unroll
      for (int nb = 0; nb < NB; ++nb) dequant(nb, sc[nb]);
    } else {
#pragma unroll
    for (int p = 0; p < NB / 2; ++p) {
      int sc[2][4];
#pragma unroll
      for (int j = 0; j < 2; ++j) sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = MAGIC_I;
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        uint32_t b0, b1, b2, b3;
        ldsm_x4a(b0, b1, b2, b3, sK + off[kk] + p * 2048);
        mma_s8(sc[0], qa[kk], b0, b1);
        mma_s8(sc[1], qa[kk], b2, b3);
      }
      if constexpr (DEFER) fold(of[8 + p], acc1[p], pt0, pt1, pa0, pa1);
      dequant(2 * p, sc[0]);
      dequant(2 * p + 1, sc[1]);
    }
    }
    if constexpr (VAR & 2) { tm0 = fmaxf(tm0, tm0b); tm1 = fmaxf(tm1, tm1b); }
    tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffff, tm0, 1));
    tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffff, tm0, 2));
    tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffff, tm1, 1));
    tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffff, tm1, 2));
    const float n0 = fmaxf(mx0, tm0), n1 = fmaxf(mx1, tm1);
    const float a0 = ex2((mx0 - n0) * c0), a1 = ex2((mx1 - n1) * c1);
    const float ts0 = ex2((tm0 - n0) * c0), ts1 = ex2((tm1 - n1) * c1);
    mx0 = n0; mx1 = n1;
    const float b0 = fmaf(-tm0, c0, LOG2_255), b1 = fmaf(-tm1, c1, LOG2_255);

    // ---- P (u8, relative to this tile's max) chunk by chunk, P.V half 0 right behind it
    uint32_t pa[KC][4];
    uint32_t rs0 = 0, rs1 = 0;
    int acc0[8][4];
#pragma unroll
    for (int kc = 0; kc < KC; ++kc) {
      float e[4][4];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int nb = kc * 4 + u;
        e[u][0] = ex2(fmaf(t[nb][0], c0, b0)); e[u][1] = ex2(fmaf(t[nb][1], c0, b0));
        e[u][2] = ex2(fmaf(t[nb][2], c1, b1)); e[u][3] = ex2(fmaf(t[nb][3], c1, b1));
      }
      pa[kc][0] = pack_u8(e[0][0], e[0][1], e[1][0], e[1][1]);
      pa[kc][1] = pack_u8(e[0][2], e[0][3], e[1][2], e[1][3]);
      pa[kc][2] = pack_u8(e[2][0], e[2][1], e[3][0], e[3][1]);
      pa[kc][3] = pack_u8(e[2][2], e[2][3], e[3][2], e[3][3]);
      rs0 = __dp4a(pa[kc][0], 0x01010101u, rs0); rs0 = __dp4a(pa[kc][2], 0x01010101u, rs0);
      rs1 = __dp4a(pa[kc][1], 0x01010101u, rs1); rs1 = __dp4a(pa[kc][3], 0x01010101u, rs1);
#pragma unroll
      for (int nd = 0; nd < 8; nd += 2) {
        uint32_t v0, v1, v2, v3;
        ldsm_x4a(v0, v1, v2, v3, sV + off[kc] + nd * 1024);
        if (kc == 0) {
          mma_u8s8_c(acc0[nd], pa[kc], v0, v1, magic);
          mma_u8s8_c(acc0[nd + 1], pa[kc], v2, v3, magic);
        } else {
          mma_u8s8(acc0[nd], pa[kc], v0, v1);
          mma_u8s8(acc0[nd + 1], pa[kc], v2, v3);
        }
      }
    }
    l0 = fmaf(ts0, i2f(rs0), l0 * a0);
    l1 = fmaf(ts1, i2f(rs1), l1 * a1);

    // ---- P.V half 1 (kc == 0 overwrites acc1), with half 0's fold between its IMMAs
#pragma unroll
    for (int kc = 0; kc < KC; ++kc) {
#pragma unroll
      for (int nd = 0; nd < 8; nd += 2) {
        uint32_t v0, v1, v2, v3;
        ldsm_x4a(v0, v1, v2, v3, sV + off[kc] + (8 + nd) * 1024);
        if (kc == 0) {
          mma_u8s8_c(acc1[nd], pa[kc], v0, v1, magic);
          mma_u8s8_c(acc1[nd + 1], pa[kc], v2, v3, magic);
        } else {
          mma_u8s8(acc1[nd], pa[kc], v0, v1);
          mma_u8s8(acc1[nd + 1], pa[kc], v2, v3);
        }
      }
      fold(of[2 * kc], acc0[2 * kc], ts0, ts1, a0, a1);
      fold(of[2 * kc + 1], acc0[2 * kc + 1], ts0, ts1, a0, a1);
    }
    if constexpr (DEFER) {
      pa0 = a0; pa1 = a1; pt0 = ts0; pt1 = ts1;
    } else {
#pragma unroll
      for (int i = 0; i < 8; ++i) fold(of[8 + i], acc1[i], ts0, ts1, a0, a1);
    }
    if constexpr (INC) {
      sK += STAGE;
      if (sK >= s0 + STAGES * STAGE) sK -= STAGES * STAGE;
    }
  }
  cp_async_wait<0>();
  if constexpr (DEFER) {
#pragma unroll
    for (int i = 0; i < 8; ++i) fold(of[8 + i], acc1[i], pt0, pt1, pa0, pa1);
  }

  l0 += __shfl_xor_sync(0xffffffff, l0, 1); l0 += __shfl_xor_sync(0xffffffff, l0, 2);
  l1 += __shfl_xor_sync(0xffffffff, l1, 1); l1 += __shfl_xor_sync(0xffffffff, l1, 2);
  const float il0 = 1.f / l0, il1 = 1.f / l1;
  const float* vs = VS + h * D;
  const float* vm = VM + h * D;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int d = i * 8 + tq * 2;
    const float2 s2 = *reinterpret_cast<const float2*>(vs + d);
    const float2 m2 = *reinterpret_cast<const float2*>(vm + d);
    if (r0 < S)
      *reinterpret_cast<__nv_bfloat162*>(O + r0 * o_ss + h * o_sh + d) =
          __floats2bfloat162_rn(fmaf(of[i][0] * il0, s2.x, m2.x), fmaf(of[i][1] * il0, s2.y, m2.y));
    if (r0 + 8 < S)
      *reinterpret_cast<__nv_bfloat162*>(O + (r0 + 8) * o_ss + h * o_sh + d) =
          __floats2bfloat162_rn(fmaf(of[i][2] * il1, s2.x, m2.x), fmaf(of[i][3] * il1, s2.y, m2.y));
  }
}

template <int STAGES, bool DEFER, bool INC = false, int VAR = 0>
void launch_fwd8(const torch::Tensor& q8, const torch::Tensor& qs, const torch::Tensor& k8, const torch::Tensor& ksb,
                 const torch::Tensor& vt, const torch::Tensor& vs, const torch::Tensor& vm, torch::Tensor& out,
                 int S, int Spad, float scale_log2, cudaStream_t stream) {
  constexpr int SMEM = STAGES * (128 * D * 2 + 128 * 8);
  const int H = q8.size(0);
  auto kern = attn_fwd8<STAGES, DEFER, INC, VAR>;
  static bool once = [&] {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    return true;
  }();
  (void)once;
  dim3 grid((S + 127) / 128, H);
  kern<<<grid, 256, SMEM, stream>>>(
      q8.data_ptr<int8_t>(), qs.data_ptr<float>(), k8.data_ptr<int8_t>(),
      reinterpret_cast<const float2*>(ksb.data_ptr<float>()), vt.data_ptr<int8_t>(), vs.data_ptr<float>(),
      vm.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), out.stride(0), out.stride(1), S, Spad,
      scale_log2);
}

}  // namespace

#include "attn_cake/attn_cake.cuh"  // configs 94-99: config 80's numerics, CAKE schedule

// Fused q/k RMSNorm + RoPE: when given, q and k are the raw qkv-projection outputs and the
// pre-pass applies comfy-kitchen's rms_rope_split_half_ itself (bit-identical), instead of reading q/k that a
// separate in-place pass already normalized and rotated.
struct RmsRope {
  torch::Tensor tab, qw, kw;  // [1, S, 1, rot/2, 2, 2] bf16, [128] bf16, [128] bf16
  float eps;
  int rot;
};

static RmsRope check_rr(const torch::Tensor& q, const torch::Tensor& tab, const torch::Tensor& qw,
                        const torch::Tensor& kw, double eps) {
  TORCH_CHECK(tab.dim() == 6 && tab.size(0) == 1 && tab.size(1) == q.size(0) && tab.size(2) == 1 &&
              tab.size(4) == 2 && tab.size(5) == 2, "rope table must be [1, S, 1, rot/2, 2, 2]");
  TORCH_CHECK(tab.scalar_type() == at::kBFloat16 && qw.scalar_type() == at::kBFloat16 &&
              kw.scalar_type() == at::kBFloat16, "rope table and norm weights must be bf16");
  TORCH_CHECK(qw.numel() == D && kw.numel() == D, "norm weights must have 128 elements");
  const int rot = tab.size(3) * 2;
  TORCH_CHECK(rot == 96 || rot == 128, "rot_dim 96 or 128 supported");
  return RmsRope{tab.contiguous(), qw.contiguous(), kw.contiguous(), static_cast<float>(eps), rot};
}

struct Prep {
  torch::Tensor q8, qs, k8, ks, ksb, kbs, vt, vscale, vmean;
};

// Config 3xx: q8, qs and the K/V statistics partials come from the qkv GEMM epilogue
// (fp8_rowwise.cu int8_gemm_qkv_prep, VisitorQkvPrep); the pre-pass then only runs stats_final, quant_k_rr, quant_v.
struct PreQ {
  torch::Tensor q8, qs, part;
};

static Prep prepass(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, int64_t config,
                    bool smooth_k, bool smooth_v, bool rotate, const RmsRope* rr, cudaStream_t stream,
                    const PreQ* pre = nullptr) {
  const int S = q.size(0), H = q.size(1);
  const int Spad = (S + 127) / 128 * 128;
  auto f32 = q.options().dtype(at::kFloat), i8 = q.options().dtype(at::kChar);
  const int tiled = config >= 30 && config < 50;
  const int fp8v = config >= 50 && config < 60;
  const auto* qp = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
  const auto* kp = reinterpret_cast<const __nv_bfloat16*>(k.data_ptr());
  const auto* vp = reinterpret_cast<const __nv_bfloat16*>(v.data_ptr());
  Prep r;

  const int nchunk = (S + STAT_CHUNK - 1) / STAT_CHUNK;
  auto part = pre ? pre->part : torch::empty({H, nchunk, 4, D}, f32);
  auto kmean = torch::empty({H, D}, f32);
  r.vmean = torch::empty({H, D}, f32);
  r.vscale = torch::empty({H, D}, f32);
  r.q8 = pre ? pre->q8 : torch::empty({H, Spad, D}, i8);
  r.k8 = torch::empty({H, Spad, D}, i8);
  r.qs = pre ? pre->qs : torch::empty({H, Spad}, f32);
  r.ks = torch::empty({H, Spad}, f32);
  r.ksb = torch::empty({H, Spad, 2}, f32);
  if (pre) {
    TORCH_CHECK(rr && rr->rot == 96 && !tiled && !fp8v && !(config >= 60 && config < 70), "config 3xx: rot_dim 96, configs 80-99");
    TORCH_CHECK(pre->part.dim() == 4 && pre->part.size(0) == H && pre->part.size(1) == nchunk && pre->part.size(2) == 4 &&
                pre->part.size(3) == D && pre->q8.size(0) == H && pre->q8.size(1) == Spad && pre->qs.size(1) == Spad,
                "int8_gemm_qkv_prep operand shapes");
    stats_final<<<H, D, 0, stream>>>(part.data_ptr<float>(), nchunk, S, smooth_k, smooth_v, kmean.data_ptr<float>(),
                                     r.vmean.data_ptr<float>(), r.vscale.data_ptr<float>());
    const long nwarp = (long)Spad * H;
    TORCH_CHECK(nwarp % 8 == 0);
    quant_k_rr<96><<<(unsigned)(nwarp / 8), 256, 0, stream>>>(
        kp, k.stride(0), k.stride(1), S, Spad, H, reinterpret_cast<const __nv_bfloat16*>(rr->tab.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(rr->kw.data_ptr()), rr->eps, kmean.data_ptr<float>(),
        r.k8.data_ptr<int8_t>(), r.ks.data_ptr<float>(), reinterpret_cast<float2*>(r.ksb.data_ptr<float>()), rotate,
        tiled);
  } else if (rr) {
    TORCH_CHECK(!(config >= 60 && config < 70), "configs 60-69 have no fused rms/rope path");
    const auto* tp = reinterpret_cast<const __nv_bfloat16*>(rr->tab.data_ptr());
    const auto* qwp = reinterpret_cast<const __nv_bfloat16*>(rr->qw.data_ptr());
    const auto* kwp = reinterpret_cast<const __nv_bfloat16*>(rr->kw.data_ptr());
    const dim3 sgrid((H + 7) / 8, nchunk);
    const long nwarp = (long)Spad * H;
    TORCH_CHECK(nwarp % 8 == 0);
    auto stats = rr->rot == 96 ? stats_partial_rr<96> : stats_partial_rr<128>;
    auto quant = rr->rot == 96 ? quant_qk_rr<96> : quant_qk_rr<128>;
    stats<<<sgrid, 256, 0, stream>>>(kp, vp, k.stride(0), k.stride(1), v.stride(0), v.stride(1), S, H, nchunk, tp, kwp,
                                     rr->eps, part.data_ptr<float>());
    stats_final<<<H, D, 0, stream>>>(part.data_ptr<float>(), nchunk, S, smooth_k, smooth_v, kmean.data_ptr<float>(),
                                     r.vmean.data_ptr<float>(), r.vscale.data_ptr<float>());
    quant<<<(unsigned)(nwarp / 8), 256, 0, stream>>>(
        qp, kp, q.stride(0), q.stride(1), k.stride(0), k.stride(1), S, Spad, H, tp, qwp, kwp, rr->eps,
        kmean.data_ptr<float>(), r.q8.data_ptr<int8_t>(), r.qs.data_ptr<float>(), r.k8.data_ptr<int8_t>(),
        r.ks.data_ptr<float>(), reinterpret_cast<float2*>(r.ksb.data_ptr<float>()), rotate, tiled);
  } else {
    stats_partial<<<dim3(nchunk, H), D, 0, stream>>>(kp, vp, k.stride(0), k.stride(1), v.stride(0), v.stride(1), S,
                                                    nchunk, part.data_ptr<float>());
    stats_final<<<H, D, 0, stream>>>(part.data_ptr<float>(), nchunk, S, smooth_k, smooth_v, kmean.data_ptr<float>(),
                                     r.vmean.data_ptr<float>(), r.vscale.data_ptr<float>());
    quant_qk<<<dim3(Spad / 8, H), 256, 0, stream>>>(
        qp, kp, q.stride(0), q.stride(1), k.stride(0), k.stride(1), S, Spad, kmean.data_ptr<float>(),
        r.q8.data_ptr<int8_t>(), r.qs.data_ptr<float>(), r.k8.data_ptr<int8_t>(), r.ks.data_ptr<float>(),
        reinterpret_cast<float2*>(r.ksb.data_ptr<float>()), rotate, tiled);
    if (config >= 60 && config < 70) {  // one K scale per 128-key block (attn_fwd6)
      r.kbs = torch::empty({H, Spad / 128}, f32);
      quant_k_block<<<dim3(Spad / 128, H), 256, 0, stream>>>(kp, k.stride(0), k.stride(1), S, Spad,
                                                             kmean.data_ptr<float>(), rotate, r.k8.data_ptr<int8_t>(),
                                                             r.kbs.data_ptr<float>());
    }
  }
  r.vt = torch::empty({H, D, Spad}, i8);
  quant_v<<<dim3(Spad / 64, H), 256, 0, stream>>>(vp, v.stride(0), v.stride(1), S, Spad, r.vmean.data_ptr<float>(),
                                                  r.vscale.data_ptr<float>(), r.vt.data_ptr<int8_t>(), tiled, fp8v);
  if (fp8v) r.vscale.mul_(127.f / 448.f);
  return r;
}

static void check_qkv(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v) {
  TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3, "q/k/v must be [S, H, D]");
  TORCH_CHECK(q.size(2) == D && k.size(2) == D && v.size(2) == D, "head_dim must be 128");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16 && v.scalar_type() == at::kBFloat16);
  TORCH_CHECK(q.stride(2) == 1 && k.stride(2) == 1 && v.stride(2) == 1, "last dim must be contiguous");
  TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(), "self-attention shapes only");
}

static torch::Tensor attention_impl(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
                                    double scale, int64_t config, bool smooth_k, bool smooth_v, bool rotate,
                                    c10::optional<torch::Tensor> out_opt, const RmsRope* rr, const PreQ* pre = nullptr,
                                    int64_t nbias = 0, double bias = 0.0);

// q, k, v: [S, H, 128] bf16 (any S/H strides, last dim contiguous). Returns [S, H, 128] bf16.
torch::Tensor int8_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale, int64_t config,
                             bool smooth_k, bool smooth_v, bool rotate, c10::optional<torch::Tensor> out_opt) {
  check_qkv(q, k, v);
  const at::cuda::CUDAGuard guard(q.device());
  return attention_impl(q, k, v, scale, config, smooth_k, smooth_v, rotate, out_opt, nullptr);
}

// Same, with q and k the raw (pre-RMSNorm, pre-RoPE) projections: applies ck's rms_rope_split_half_ inside the
// pre-pass. Output and INT8 operands are bit-identical to rms_rope_split_half_(q, k, ...) + int8_attention.
torch::Tensor int8_attention_rr(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor tab,
                                torch::Tensor qw, torch::Tensor kw, double eps, double scale, int64_t config,
                                bool smooth_k, bool smooth_v, bool rotate, c10::optional<torch::Tensor> out_opt) {
  check_qkv(q, k, v);
  const at::cuda::CUDAGuard guard(q.device());
  const RmsRope rr = check_rr(q, tab, qw, kw, eps);
  return attention_impl(q, k, v, scale, config, smooth_k, smooth_v, rotate, out_opt, &rr);
}

// int8_attention_rr plus the Jenga text-attention bias: every query's logit on keys
// [0, nbias) gets + bias (natural-log units, i.e. those keys' softmax weight x e^bias before renormalising).
// Config 94 or 90 only; nbias = 0 runs the unbiased kernel.
torch::Tensor int8_attention_rr_bias(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor tab,
                                     torch::Tensor qw, torch::Tensor kw, double eps, double scale, int64_t config,
                                     int64_t nbias, double bias, bool smooth_k, bool smooth_v, bool rotate,
                                     c10::optional<torch::Tensor> out_opt) {
  check_qkv(q, k, v);
  const at::cuda::CUDAGuard guard(q.device());
  const RmsRope rr = check_rr(q, tab, qw, kw, eps);
  return attention_impl(q, k, v, scale, config, smooth_k, smooth_v, rotate, out_opt, &rr, nullptr, nbias, bias);
}

// Verification: the pre-pass operands [q8, qs, k8, ks, ksb, vt, vscale, vmean], unfused (no tab) or fused.
std::vector<torch::Tensor> int8_prepass(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                        c10::optional<torch::Tensor> tab, c10::optional<torch::Tensor> qw,
                                        c10::optional<torch::Tensor> kw, double eps, int64_t config, bool smooth_k,
                                        bool smooth_v, bool rotate) {
  check_qkv(q, k, v);
  const at::cuda::CUDAGuard guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  RmsRope rr;
  if (tab.has_value()) rr = check_rr(q, *tab, *qw, *kw, eps);
  Prep p = prepass(q, k, v, config, smooth_k, smooth_v, rotate, tab.has_value() ? &rr : nullptr, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {p.q8, p.qs, p.k8, p.ks, p.ksb, p.vt, p.vscale, p.vmean};
}

// Config 3xx: attention_rr with q8/qs/part from int8_gemm_qkv_prep (the qkv GEMM epilogue); k, v are the raw
// (pre-RMSNorm) k and v columns of the same GEMM output. Bit-identical to int8_attention_rr on that GEMM's q/k/v.
torch::Tensor int8_attention_pre(torch::Tensor k, torch::Tensor v, torch::Tensor q8, torch::Tensor qs,
                                 torch::Tensor part, torch::Tensor tab, torch::Tensor kw, double eps, double scale,
                                 int64_t config, bool smooth_k, bool smooth_v, bool rotate,
                                 c10::optional<torch::Tensor> out_opt) {
  check_qkv(k, k, v);
  const at::cuda::CUDAGuard guard(k.device());
  const RmsRope rr = check_rr(k, tab, kw, kw, eps);
  const PreQ pre{q8, qs, part};
  return attention_impl(k, k, v, scale, config, smooth_k, smooth_v, rotate, out_opt, &rr, &pre);
}

// Verification: config 3xx pre-pass operands [q8, qs, k8, ks, ksb, vt, vscale, vmean] (as int8_prepass)
std::vector<torch::Tensor> int8_prepass_pre(torch::Tensor k, torch::Tensor v, torch::Tensor q8, torch::Tensor qs,
                                            torch::Tensor part, torch::Tensor tab, torch::Tensor kw, double eps,
                                            int64_t config, bool smooth_k, bool smooth_v, bool rotate) {
  check_qkv(k, k, v);
  const at::cuda::CUDAGuard guard(k.device());
  const RmsRope rr = check_rr(k, tab, kw, kw, eps);
  const PreQ pre{q8, qs, part};
  Prep p = prepass(k, k, v, config, smooth_k, smooth_v, rotate, &rr, at::cuda::getCurrentCUDAStream().stream(), &pre);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {p.q8, p.qs, p.k8, p.ks, p.ksb, p.vt, p.vscale, p.vmean};
}

// Verification: ck's rms_rope_split_half_ on one of q/k ([S, H, 128], strided), out of place -> [S, H, 128].
torch::Tensor rms_rope_ref(torch::Tensor x, torch::Tensor tab, torch::Tensor w, double eps) {
  TORCH_CHECK(x.dim() == 3 && x.size(2) == D && x.stride(2) == 1 && x.scalar_type() == at::kBFloat16);
  const at::cuda::CUDAGuard guard(x.device());
  const RmsRope rr = check_rr(x, tab, w, w, eps);
  const int S = x.size(0), H = x.size(1);
  auto y = torch::empty({S, H, D}, x.options());
  const long nwarp = (long)S * H;
  auto kern = rr.rot == 96 ? rms_rope_only<96> : rms_rope_only<128>;
  kern<<<(unsigned)((nwarp + 7) / 8), 256, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), x.stride(0), x.stride(1), S, H,
      reinterpret_cast<const __nv_bfloat16*>(rr.tab.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(rr.qw.data_ptr()),
      rr.eps, reinterpret_cast<__nv_bfloat16*>(y.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

static torch::Tensor attention_impl(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
                                    double scale, int64_t config, bool smooth_k, bool smooth_v, bool rotate,
                                    c10::optional<torch::Tensor> out_opt, const RmsRope* rr, const PreQ* pre,
                                    int64_t nbias, double bias) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  const int S = q.size(0), H = q.size(1);
  const int Spad = (S + 127) / 128 * 128;
  Prep p = prepass(q, k, v, config, smooth_k, smooth_v, rotate, rr, stream, pre);
  auto &q8 = p.q8, &qs = p.qs, &k8 = p.k8, &ks = p.ks, &ksb = p.ksb, &kbs = p.kbs, &vt = p.vt;
  auto &vscale = p.vscale, &vmean = p.vmean;

  torch::Tensor out = out_opt.has_value() ? *out_opt : torch::empty({S, H, D}, q.options());
  TORCH_CHECK(out.stride(2) == 1 && out.size(0) == S && out.size(1) == H);
  const float scale_log2 = static_cast<float>(scale) * 1.4426950408889634f;
  if (nbias > 0) {  // text bias: config 94's (or 90's) kernel with TB
    TORCH_CHECK(config == 94 || config == 90, "text bias: config 94 or 90 only");
    TORCH_CHECK(nbias <= S, "text bias: nbias > S");
    if (config == 90)
      cake::launch<7, 2, true, 4>(q8, qs, k8, ksb, vt, vscale, vmean, out, S, Spad, scale_log2, stream, 0, (int)nbias,
                                  static_cast<float>(bias * 1.4426950408889634));
    else
      cake::launch<7, 2, true>(q8, qs, k8, ksb, vt, vscale, vmean, out, S, Spad, scale_log2, stream, 0, (int)nbias,
                               static_cast<float>(bias * 1.4426950408889634));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
  }
  switch (config) {
    case 0: launch_fwd<64, 3, 8>(q8, qs, k8, ks, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 1: launch_fwd<64, 4, 8>(q8, qs, k8, ks, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 2: launch_fwd<128, 2, 8>(q8, qs, k8, ks, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 3: launch_fwd<128, 3, 8>(q8, qs, k8, ks, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 4: launch_fwd<64, 3, 4>(q8, qs, k8, ks, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 5: launch_fwd<64, 2, 4>(q8, qs, k8, ks, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 6: launch_fwd<64, 2, 4, 3, true>(q8, qs, k8, ks, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 7: launch_fwd<64, 2, 4, 2, true>(q8, qs, k8, ks, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 8: launch_fwd<64, 3, 4, 2, false>(q8, qs, k8, ks, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 9: launch_fwd<128, 2, 4, 2, false>(q8, qs, k8, ks, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 10: launch_fwd<64, 2, 8, 1, false>(q8, qs, k8, ks, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 20: launch_fwd2<64, 3, 8>(q8, qs, k8, ksb, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 21: launch_fwd2<64, 4, 8>(q8, qs, k8, ksb, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 22: launch_fwd2<64, 3, 4>(q8, qs, k8, ksb, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 23: launch_fwd2<64, 4, 4>(q8, qs, k8, ksb, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 28: launch_fwd2<64, 3, 8, 0, 2, true>(q8, qs, k8, ksb, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 29: launch_fwd2<64, 4, 8, 0, 2, true>(q8, qs, k8, ksb, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 24: launch_fwd2<64, 3, 12>(q8, qs, k8, ksb, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 25: launch_fwd2<64, 4, 12>(q8, qs, k8, ksb, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 26: launch_fwd2<64, 4, 12, 0, 4>(q8, qs, k8, ksb, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 27: launch_fwd2<64, 4, 8, 0, 4>(q8, qs, k8, ksb, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 11: launch_fwd<64, 3, 12>(q8, qs, k8, ks, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 12: launch_fwd<64, 4, 12>(q8, qs, k8, ks, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 30: launch_fwd3<4, true>(q8, qs, k8, ksb, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 31: launch_fwd3<5, true>(q8, qs, k8, ksb, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 32: launch_fwd3<4, false>(q8, qs, k8, ksb, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 40: launch_fwd4<4, 0>(q8, qs, k8, ksb, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 41: launch_fwd4<3, 0>(q8, qs, k8, ksb, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 49: launch_fwd4<4, 1>(q8, qs, k8, ksb, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 50: launch_fwd5<64, 3, 8, true, false>(q8, qs, k8, ksb, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 51: launch_fwd5<64, 3, 8, true, true>(q8, qs, k8, ksb, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 52: launch_fwd5<128, 3, 8, false, false>(q8, qs, k8, ksb, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 53: launch_fwd5<128, 3, 8, false, true>(q8, qs, k8, ksb, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 54: launch_fwd5<64, 4, 8, true, false>(q8, qs, k8, ksb, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 55: launch_fwd5<128, 2, 8, false, false>(q8, qs, k8, ksb, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 56: launch_fwd5<64, 3, 12, true, false>(q8, qs, k8, ksb, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 60: launch_fwd6<2>(q8, qs, k8, kbs, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 61: launch_fwd6<3>(q8, qs, k8, kbs, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 91: launch_fwd2<64, 3, 8, 1>(q8, qs, k8, ksb, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 92: launch_fwd2<64, 3, 8, 2>(q8, qs, k8, ksb, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 93: launch_fwd2<64, 3, 8, 3>(q8, qs, k8, ksb, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 70: launch_fwd7<64, 3, 8>(q8, qs, k8, ksb, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 71: launch_fwd7<64, 4, 8>(q8, qs, k8, ksb, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 80: launch_fwd8<2, false>(q8, qs, k8, ksb, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 81: launch_fwd8<2, true>(q8, qs, k8, ksb, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 82: launch_fwd8<3, false, true>(q8, qs, k8, ksb, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 83: launch_fwd8<3, true, true>(q8, qs, k8, ksb, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 84: launch_fwd8<2, false, true>(q8, qs, k8, ksb, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 86: launch_fwd8<2, false, false, 1>(q8, qs, k8, ksb, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 87: launch_fwd8<2, false, false, 2>(q8, qs, k8, ksb, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 88: launch_fwd8<2, false, false, 3>(q8, qs, k8, ksb, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 85: launch_fwd8<2, true, true>(q8, qs, k8, ksb, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    // see attn_cake/DESIGN.md; all bit-identical to config 80. cake::launch<VAR, SYNC>:
    // persistent 1 CTA/SM, TMA ring. VAR bit 0: exp/fold interleaved in P.V (80's body), bit 1: dequant
    // inside Q.K^T, bit 2: 2 key pairs per Q.K^T group, 16: merged P.V(n)+Q.K^T(n+1) phase.
    // SYNC 0: none, 1: CAKE/FA3 ping-pong of the tensor sections, 2: softmax mutex (94 = best of these; 90 below is faster still).
    case 94: cake::launch<7, 2>(q8, qs, k8, ksb, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 95: cake::launch<1, 1>(q8, qs, k8, ksb, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 96: cake::launch<3, 2>(q8, qs, k8, ksb, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 97: cake::launch<16, 1>(q8, qs, k8, ksb, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 98: cake::launch<7, 0>(q8, qs, k8, ksb, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 99: cake::launch<3, 0>(q8, qs, k8, ksb, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    // attn_cake/DESIGN.md section 7, config 94 with a softmax variant:
    // 90: the softmax mutex starts after the row max and scales (bit-identical to 94 and 80, faster);
    // 89: lazy rescale, the magic bias kept in O (not bit-identical, same error);
    // 79: f16x2 exps (not bit-identical, slower: sm_121 has no paired MUFU.EX2).
    case 90: cake::launch<7, 2, false, 4>(q8, qs, k8, ksb, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 89: cake::launch<7, 2, false, 2>(q8, qs, k8, ksb, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    case 79: cake::launch<7, 2, false, 1>(q8, qs, k8, ksb, vt, vscale, vmean, out, S, Spad, scale_log2, stream); break;
    default: TORCH_CHECK(false, "bad config");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("int8_attention", &int8_attention, "INT8 flash attention, D=128, non-causal (sm_121)",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale"), py::arg("config") = 0,
        py::arg("smooth_k") = true, py::arg("smooth_v") = true, py::arg("rotate") = true,
        py::arg("out") = py::none());
  m.def("int8_attention_rr", &int8_attention_rr,
        "INT8 flash attention on raw q/k with comfy-kitchen's rms_rope_split_half_ fused into the pre-pass",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("tab"), py::arg("qw"), py::arg("kw"), py::arg("eps"),
        py::arg("scale"), py::arg("config") = 94, py::arg("smooth_k") = true, py::arg("smooth_v") = true,
        py::arg("rotate") = true, py::arg("out") = py::none());
  m.def("int8_attention_rr_bias", &int8_attention_rr_bias,
        "int8_attention_rr with + bias (natural log) on every query's logits of keys [0, nbias) (config 94 or 90)",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("tab"), py::arg("qw"), py::arg("kw"), py::arg("eps"),
        py::arg("scale"), py::arg("config") = 94, py::arg("nbias") = 0, py::arg("bias") = 0.0,
        py::arg("smooth_k") = true, py::arg("smooth_v") = true, py::arg("rotate") = true, py::arg("out") = py::none());
  m.def("int8_prepass", &int8_prepass, "pre-pass operands (verification)", py::arg("q"), py::arg("k"), py::arg("v"),
        py::arg("tab") = py::none(), py::arg("qw") = py::none(), py::arg("kw") = py::none(), py::arg("eps") = 1e-5,
        py::arg("config") = 94, py::arg("smooth_k") = true, py::arg("smooth_v") = true, py::arg("rotate") = true);
  m.def("int8_attention_pre", &int8_attention_pre,
        "config 3xx: INT8 attention with q8/qs/stat partials from the qkv GEMM epilogue (int8_gemm_qkv_prep)",
        py::arg("k"), py::arg("v"), py::arg("q8"), py::arg("qs"), py::arg("part"), py::arg("tab"), py::arg("kw"),
        py::arg("eps"), py::arg("scale"), py::arg("config") = 94, py::arg("smooth_k") = true, py::arg("smooth_v") = true,
        py::arg("rotate") = true, py::arg("out") = py::none());
  m.def("int8_prepass_pre", &int8_prepass_pre, "config 3xx pre-pass operands (verification)", py::arg("k"),
        py::arg("v"), py::arg("q8"), py::arg("qs"), py::arg("part"), py::arg("tab"), py::arg("kw"), py::arg("eps"),
        py::arg("config") = 94, py::arg("smooth_k") = true, py::arg("smooth_v") = true, py::arg("rotate") = true);
  m.def("rms_rope_ref", &rms_rope_ref, "rms_rope_split_half_ re-implementation, out of place (verification)",
        py::arg("x"), py::arg("tab"), py::arg("w"), py::arg("eps"));
}
