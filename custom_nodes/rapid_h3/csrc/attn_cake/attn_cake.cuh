// attn_cake.cuh: config 80's INT8 attention numerics on a CAKE-style schedule.
//
// Included by ../attn_int8.cu after its anonymous namespace (it reuses D, MAGIC_*, ex2, pack_u8,
// ldsm_x4a, lds128, mma_s8, mma_u8s8(_c), i2f and the whole quantization pre-pass). See DESIGN.md.
//
//   * persistent: one 256-thread CTA per SM walks (head, 128-row Q tile) units bid, bid+grid, ...
//     (head-major, so the 48 resident CTAs share one head's K/V in L2; all units are equal work for
//     a single non-causal segment, so the LPT order of CAKE reduces to round robin);
//   * K / V^T tiles (128 x 128 B) arrive by TMA (cp.async.bulk.tensor, 128 B swizzle = the
//     chunk ^ (row & 7) layout config 80's ldmatrix offsets expect), the per-key (scale, bias)
//     pairs by a 1 KB bulk copy, into a 2-stage ring with full / empty mbarriers. K and V(+KSB)
//     have separate barriers, so a K stage is refilled as soon as both warp groups' Q.K^T are done;
//   * one producer thread (warp 4, lane 0, i.e. in the trailing warp group) issues the copies;
//   * warps 0-3 and 4-7 form two warp groups (one warp of each per sub-partition), ordered with
//     named barriers 1/2 (bar.sync by one group, bar.arrive by the other):
//       SYNC 1 (CAKE / FA3 ping-pong): the tensor sections alternate, 0.QK 1.QK 0.PV 1.PV ...;
//              measured 8-10% SLOWER than config 80 on GB10: one warp per sub-partition can only
//              drive ldmatrix-fed m16n8k32 IMMAs at ~204 of 242 TFLOPS (bench/cuda/ldsm_mma.cu);
//       SYNC >= 2 (softmax mutex, the kept design): the softmax sections alternate and the tensor
//              sections overlap freely, so both warps of a sub-partition still issue IMMAs but are
//              never both stuck in the max-shuffle / exp2 critical path at once;
//   * arithmetic per score, per tile and per output element is config 80's, op for op, so the
//     output is bit-identical to config 80 (and the error vs fp32 identical).
//
// VAR bit 0: config 80's in-warp interleave inside the P.V section (exp2 + u8 packing of chunk kc
//            between the IMMAs of chunk kc-1; half 0's fold between half 1's IMMAs).
//      bit 1: dequantize + max of each 16-key block between the Q.K^T IMMAs (key-pair-outer, as 80);
//             otherwise Q.K^T runs k-step-outer with 16 independent accumulators.
//      bit 2 / 3: 2 / 4 key pairs per Q.K^T group (4 / 8 independent accumulator chains).
//      >= 16: attn_cake_m, the FA3 merged phase [P.V(n) | Q.K^T(n+1)] with ping-pong (slow, kept
//             as a documented experiment).
#pragma once
#include <cuda.h>
#include <cudaTypedefs.h>

namespace cake {

constexpr int BN = 128, BM = 128, NT = 256;
constexpr int TILE = BN * D;                       // 16 KB: K tile or V^T tile
constexpr int SM_K = 1024;                         // [0, 1024): mbarriers
constexpr int SM_V = SM_K + 2 * TILE;
constexpr int SM_KS = SM_V + 2 * TILE;
constexpr int SM_TOTAL = SM_KS + 2 * BN * 8;       // 68608
constexpr int SMEM_ALLOC = SM_TOTAL + 1024;        // + alignment slack (swizzle atoms need 1 KB)

__device__ __forceinline__ void mbar_init(uint32_t b, int n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(b), "r"(n));
}
__device__ __forceinline__ void mbar_expect(uint32_t b, int bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(b) : "memory");
}
// try_wait with a suspend-time hint: a waiting warp sleeps until the phase completes (or the hint
// expires) instead of spinning, so waits do not steal issue slots from the other warp of the SMSP
__device__ __forceinline__ void mbar_wait(uint32_t b, int ph) {
  asm volatile(
      "{\n .reg .pred p;\n CAKE_W_%=:\n"
      " mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2;\n"
      " @!p bra CAKE_W_%=;\n}\n" ::"r"(b), "r"(ph), "r"(0x989680) : "memory");
}
__device__ __forceinline__ void tma3(uint32_t dst, const CUtensorMap* m, int x, int y, int z, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
      ::"r"(dst), "l"(reinterpret_cast<uint64_t>(m)), "r"(x), "r"(y), "r"(z), "r"(bar) : "memory");
}
__device__ __forceinline__ void bulk1(uint32_t dst, const void* src, int bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void bar_sync(int id) { asm volatile("bar.sync %0, 256;" ::"r"(id) : "memory"); }
__device__ __forceinline__ void bar_arrive(int id) { asm volatile("bar.arrive %0, 256;" ::"r"(id) : "memory"); }

// SMX bit 0 (config 79): half-precision exps. sm_121 has no paired MUFU op:
// ex2.approx.f16x2 assembles to two MUFU.EX2.F16 plus a PRMT, so this path costs more instructions than the fp32
// one (kept as the measured experiment, DESIGN.md section 7)
__device__ __forceinline__ uint32_t ex2h2(uint32_t x) {
  uint32_t y;
  asm("ex2.approx.f16x2 %0, %1;" : "=r"(y) : "r"(x));
  return y;
}
__device__ __forceinline__ uint32_t f2h2(float lo, float hi) {
  uint32_t y;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(y) : "f"(hi), "f"(lo));
  return y;
}
__device__ __forceinline__ uint32_t hadd2(uint32_t a, uint32_t b) {
  uint32_t y;
  asm("add.rn.f16x2 %0, %1, %2;" : "=r"(y) : "r"(a), "r"(b));
  return y;
}
// u8 target of the half-precision path: 2^LOG2_254 = 254, so that ex2.approx.f16's error (<= 2^-9.9 relative) plus
// the f16 rounding of the argument can never reach 255.5 (which the 1024 magic add would turn into byte 0)
constexpr float LOG2_254 = 7.988684686772166f;

// TB (Jenga text-attention bias): keys [0, nbias) get bias_log2 added to their
// exp2-domain logit for every query row. Folded into the dequantized score t (the units of the running max) as
// bias_log2 / c per row, so the max, the u8 P range and the row sums all see it. TB = false compiles it out.
//
// SMX (softmax variants, DESIGN.md section 7; VAR bit 0 and the mutex (SYNC >= 2) only):
//   bit 2: the mutex section starts after the row-max shuffles and the a / ts / b scalars, i.e. at the exps
//          (config 90; bit-identical to config 94, 2-3% faster main kernel);
//   bit 0: exps in f16x2 (ex2.approx.f16x2), u8 packing by a 1024 magic add + PRMT (config 79; not
//          bit-identical, slower);
//   bit 1: lazy rescale (config 89; not bit-identical). The running max m only moves when a tile max
//          exceeds it by more than tau (exp2 units); otherwise a = 1 and ts = 2^((tm - m) c) <= 2^tau (exact
//          in math: O and l share the stale m). P.V accumulators start at the magic 1.5 * 2^23, so the per-tile fold
//          O = a O + ts (acc_bits - MAGIC) was 2 FFMA per element. Here the fold is O' += ts acc_bits (1 FFMA)
//          and the bias MAGIC * B (B = sum of ts) stays in O'. It is removed by a separate pass
//          O' = a O' - a MAGIC B, run when a row of the warp moved its max (a != 1) or every nclean tiles
//          (bounds |O'| and so its rounding), and once in the epilogue.
template <int VAR, int SYNC, bool TB = false, int SMX = 0>
__global__ void __launch_bounds__(NT, 1)
attn_cake(const __grid_constant__ CUtensorMap Kmap, const __grid_constant__ CUtensorMap Vmap,
          const int8_t* __restrict__ Qg, const float* __restrict__ QS, const float2* __restrict__ KSB,
          const float* __restrict__ VS, const float* __restrict__ VM, __nv_bfloat16* __restrict__ O,
          long o_ss, long o_sh, int S, int Spad, int ntq, int nunits, float scale_log2, int nbias,
          float bias_log2, int nclean, float tau) {
  constexpr int NB = 16, KC = 4;
  constexpr bool H2 = SMX & 1, LZ = SMX & 2, EARLY = SMX & 4;
  static_assert(SMX == 0 || ((VAR & 1) && SYNC >= 2), "softmax variants: VAR bit 0 (P.V interleave), mutex only");
  constexpr bool PP = SYNC == 1;  // tensor sections alternate between the warp groups
  // SYNC >= 2: softmax sections alternate between the warp groups (mutex), tensor sections overlap.
  // The section starts at the row-max shuffles and ends (VAR bit 0) after chunk MXE's IMMAs of P.V
  // half 0 (SYNC 2: MXE = 3, the whole half 0; 3: 2; 4: 1; 5: 0) or (no bit 0) after the u8 packing.
  constexpr bool MX = SYNC >= 2;
  constexpr int MXE = SYNC >= 2 ? 5 - SYNC : -1;
  extern __shared__ uint8_t smem_raw[];
  const uint32_t s0 = (smem_u32(smem_raw) + 1023u) & ~1023u;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tq = lane & 3, wg = warp >> 2;
  const uint32_t kfull = s0, vfull = s0 + 16, kempty = s0 + 32, vempty = s0 + 48;  // 2 barriers each
  const int ntiles = Spad / BN;
  const int G = gridDim.x;

  if (tid == 0) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      mbar_init(kfull + 8 * i, 1); mbar_init(vfull + 8 * i, 1);
      mbar_init(kempty + 8 * i, 8); mbar_init(vempty + 8 * i, 8);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  // ---- producer (warp 4 lane 0): separate cursors for the K stream and the V(+KSB) stream
  int ku = blockIdx.x, kkt = 0, vu = blockIdx.x, vkt = 0;
  auto load_k = [&](int st) {
    const int h = ku / ntq;
    mbar_expect(kfull + 8 * st, TILE);
    tma3(s0 + SM_K + st * TILE, &Kmap, 0, kkt * BN, h, kfull + 8 * st);
    if (++kkt == ntiles) { kkt = 0; ku += G; }
  };
  auto load_v = [&](int st) {
    const int h = vu / ntq;
    mbar_expect(vfull + 8 * st, TILE + BN * 8);
    tma3(s0 + SM_V + st * TILE, &Vmap, vkt * BN, 0, h, vfull + 8 * st);
    bulk1(s0 + SM_KS + st * BN * 8, KSB + (long)h * Spad + vkt * BN, BN * 8, vfull + 8 * st);
    if (++vkt == ntiles) { vkt = 0; vu += G; }
  };
  const bool prod_warp = warp == 4;
  if (prod_warp && lane == 0) {
    for (int st = 0; st < 2; ++st) {
      if (ku < nunits) load_k(st);
      if (vu < nunits) load_v(st);
    }
  }

  // ldmatrix x4 lane offsets inside a 128 B-row swizzled tile (config 80's)
  uint32_t off[4];
  {
    const int m = lane >> 3, r = (m >> 1) * 8 + (lane & 7);
#pragma unroll
    for (int k = 0; k < 4; ++k) off[k] = r * 128 + (((k * 2 + (m & 1)) ^ (lane & 7)) << 4);
  }
  const int magic = MAGIC_I;
  auto fold = [&](float* o, const int* a, float ts0, float ts1, float a0, float a1) {
    o[0] = fmaf(ts0, __int_as_float(a[0]), fmaf(o[0], a0, -ts0 * MAGIC_F));
    o[1] = fmaf(ts0, __int_as_float(a[1]), fmaf(o[1], a0, -ts0 * MAGIC_F));
    o[2] = fmaf(ts1, __int_as_float(a[2]), fmaf(o[2], a1, -ts1 * MAGIC_F));
    o[3] = fmaf(ts1, __int_as_float(a[3]), fmaf(o[3], a1, -ts1 * MAGIC_F));
  };
  // LZ: O' += ts acc_bits; the magic bias ts * MAGIC_F stays in O' (tracked in B)
  auto fold1 = [&](float* o, const int* a, float ts0, float ts1) {
    o[0] = fmaf(ts0, __int_as_float(a[0]), o[0]);
    o[1] = fmaf(ts0, __int_as_float(a[1]), o[1]);
    o[2] = fmaf(ts1, __int_as_float(a[2]), o[2]);
    o[3] = fmaf(ts1, __int_as_float(a[3]), o[3]);
  };

  if ((PP || MX) && wg == 1) bar_arrive(1);  // group 0 goes first
  int n = 0;                          // tile counter of this CTA (ring position / mbarrier phase)
  for (int u = blockIdx.x; u < nunits; u += G) {
    const int h = u / ntq, m0 = (u % ntq) * BM;
    const bool last_unit = u + G >= nunits;
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
    // TB: the bias in this row's score units (padded rows have c = 0 and are never written)
    const float ib0 = TB && c0 > 0.f ? bias_log2 / c0 : 0.f, ib1 = TB && c1 > 0.f ? bias_log2 / c1 : 0.f;
    float of[16][4];
#pragma unroll
    for (int i = 0; i < 16; ++i) of[i][0] = of[i][1] = of[i][2] = of[i][3] = 0.f;
    float mx0 = -INFINITY, mx1 = -INFINITY, l0 = 0.f, l1 = 0.f;
    float B0 = 0.f, B1 = 0.f;  // LZ: sum of ts since the last bias removal (O' = O + MAGIC_F * B)
    int lzc = 0;               // LZ: tiles since the last bias removal (warp-uniform)

#pragma unroll 1
    for (int kt = 0; kt < ntiles; ++kt, ++n) {
      const int st = n & 1, ph = (n >> 1) & 1;
      const uint32_t sK = s0 + SM_K + st * TILE, sV = s0 + SM_V + st * TILE, sKS = s0 + SM_KS + st * BN * 8;
      const bool btile = TB && kt * BN < nbias;

      // =============== tensor section 1: S = Q K^T
      float t[NB][4];
      float tm0 = -INFINITY, tm1 = -INFINITY;
      auto dequant = [&](int nb, const int* scv) {
        const float4 kb = lds128(sKS + nb * 64 + tq * 16);  // (s_j, b_j, s_j+1, b_j+1)
        t[nb][0] = fmaf(__int_as_float(scv[0]), kb.x, kb.y);
        t[nb][1] = fmaf(__int_as_float(scv[1]), kb.z, kb.w);
        t[nb][2] = fmaf(__int_as_float(scv[2]), kb.x, kb.y);
        t[nb][3] = fmaf(__int_as_float(scv[3]), kb.z, kb.w);
        if constexpr (TB) {
          if (btile) {
            const int j = kt * BN + nb * 8 + tq * 2;
            if (j < nbias) { t[nb][0] += ib0; t[nb][2] += ib1; }
            if (j + 1 < nbias) { t[nb][1] += ib0; t[nb][3] += ib1; }
          }
        }
        tm0 = fmaxf(tm0, fmaxf(t[nb][0], t[nb][1]));
        tm1 = fmaxf(tm1, fmaxf(t[nb][2], t[nb][3]));
      };
      mbar_wait(kfull + 8 * st, ph);
      if constexpr (VAR & 2) mbar_wait(vfull + 8 * st, ph);  // KSB rides with V
      if (PP) bar_sync(1 + wg);
      if constexpr (VAR & 2) {
        constexpr int PQ = (VAR & 8) ? 4 : (VAR & 4) ? 2 : 1;  // key pairs per group: 2 * PQ IMMA chains
#pragma unroll
        for (int p = 0; p < NB / 2; p += PQ) {
          int sc[2 * PQ][4];
#pragma unroll
          for (int j = 0; j < 2 * PQ; ++j) sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = MAGIC_I;
#pragma unroll
          for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
            for (int q = 0; q < PQ; ++q) {
              uint32_t b0, b1, b2, b3;
              ldsm_x4a(b0, b1, b2, b3, sK + off[kk] + (p + q) * 2048);
              mma_s8(sc[2 * q], qa[kk], b0, b1);
              mma_s8(sc[2 * q + 1], qa[kk], b2, b3);
            }
          }
#pragma unroll
          for (int j = 0; j < 2 * PQ; ++j) dequant(2 * p + j, sc[j]);
        }
        if (PP) bar_arrive(2 - wg);
        __syncwarp();
        if (lane == 0) mbar_arrive(kempty + 8 * st);
      } else {
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
        if (PP) bar_arrive(2 - wg);
        __syncwarp();
        if (lane == 0) mbar_arrive(kempty + 8 * st);
        mbar_wait(vfull + 8 * st, ph);
#pragma unroll
        for (int nb = 0; nb < NB; ++nb) dequant(nb, sc[nb]);
      }
      // refill this K stage with tile n+2 once all 8 warps are past it
      if (prod_warp) {
        if (ku < nunits) {
          mbar_wait(kempty + 8 * st, ph);
          if (lane == 0) load_k(st);
        }
        __syncwarp();
      }

      // =============== softmax (fp32, exp2 domain), config 80's arithmetic
      const bool last = last_unit && kt == ntiles - 1;
      if (MX && !EARLY) bar_sync(1 + wg);
      tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffff, tm0, 1));
      tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffff, tm0, 2));
      tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffff, tm1, 1));
      tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffff, tm1, 2));
      float a0, a1, ts0, ts1;
      bool clean = true;  // LZ: remove the bias (and apply a) before this tile's fold
      if constexpr (LZ) {
        // move the running max only when the tile max exceeds it by more than tau (first tile: mx = -inf)
        const bool u0 = (tm0 - mx0) * c0 > tau, u1 = (tm1 - mx1) * c1 > tau;
        a0 = u0 ? ex2((mx0 - tm0) * c0) : 1.f; a1 = u1 ? ex2((mx1 - tm1) * c1) : 1.f;
        ts0 = u0 ? 1.f : ex2((tm0 - mx0) * c0); ts1 = u1 ? 1.f : ex2((tm1 - mx1) * c1);
        if (u0) mx0 = tm0;
        if (u1) mx1 = tm1;
        clean = __any_sync(0xffffffff, u0 || u1) || ++lzc >= nclean;
        if (clean) lzc = 0;
      } else {
        const float n0 = fmaxf(mx0, tm0), n1 = fmaxf(mx1, tm1);
        a0 = ex2((mx0 - n0) * c0); a1 = ex2((mx1 - n1) * c1);
        ts0 = ex2((tm0 - n0) * c0); ts1 = ex2((tm1 - n1) * c1);
        mx0 = n0; mx1 = n1;
      }
      const float b0 = fmaf(-tm0, c0, H2 ? LOG2_254 : LOG2_255), b1 = fmaf(-tm1, c1, H2 ? LOG2_254 : LOG2_255);

      uint32_t pa[KC][4];
      uint32_t rs0 = 0, rs1 = 0;
      auto make_p = [&](int kc) {
        if constexpr (H2) {
          // (lo, hi) argument pairs -> f16x2 -> ex2 -> + 1024 (the integer in the low mantissa byte) -> PRMT
          uint32_t h[4][2];
#pragma unroll
          for (int u2 = 0; u2 < 4; ++u2) {
            const int nb = kc * 4 + u2;
            h[u2][0] = hadd2(ex2h2(f2h2(fmaf(t[nb][0], c0, b0), fmaf(t[nb][1], c0, b0))), 0x64006400u);
            h[u2][1] = hadd2(ex2h2(f2h2(fmaf(t[nb][2], c1, b1), fmaf(t[nb][3], c1, b1))), 0x64006400u);
          }
          pa[kc][0] = __byte_perm(h[0][0], h[1][0], 0x6420);
          pa[kc][1] = __byte_perm(h[0][1], h[1][1], 0x6420);
          pa[kc][2] = __byte_perm(h[2][0], h[3][0], 0x6420);
          pa[kc][3] = __byte_perm(h[2][1], h[3][1], 0x6420);
          rs0 = __dp4a(pa[kc][0], 0x01010101u, rs0); rs0 = __dp4a(pa[kc][2], 0x01010101u, rs0);
          rs1 = __dp4a(pa[kc][1], 0x01010101u, rs1); rs1 = __dp4a(pa[kc][3], 0x01010101u, rs1);
          return;
        }
        float e[4][4];
#pragma unroll
        for (int u2 = 0; u2 < 4; ++u2) {
          const int nb = kc * 4 + u2;
          e[u2][0] = ex2(fmaf(t[nb][0], c0, b0)); e[u2][1] = ex2(fmaf(t[nb][1], c0, b0));
          e[u2][2] = ex2(fmaf(t[nb][2], c1, b1)); e[u2][3] = ex2(fmaf(t[nb][3], c1, b1));
        }
        pa[kc][0] = pack_u8(e[0][0], e[0][1], e[1][0], e[1][1]);
        pa[kc][1] = pack_u8(e[0][2], e[0][3], e[1][2], e[1][3]);
        pa[kc][2] = pack_u8(e[2][0], e[2][1], e[3][0], e[3][1]);
        pa[kc][3] = pack_u8(e[2][2], e[2][3], e[3][2], e[3][3]);
        rs0 = __dp4a(pa[kc][0], 0x01010101u, rs0); rs0 = __dp4a(pa[kc][2], 0x01010101u, rs0);
        rs1 = __dp4a(pa[kc][1], 0x01010101u, rs1); rs1 = __dp4a(pa[kc][3], 0x01010101u, rs1);
      };

      // =============== tensor section 2: O += P V (int32 per tile, folded into fp32)
      int acc[16][4];
      if constexpr (VAR & 1) {
        if (PP) bar_sync(1 + wg);
        if constexpr (MX && EARLY) bar_sync(1 + wg);  // config 90: the mutex starts at the exps
#pragma unroll
        for (int kc = 0; kc < KC; ++kc) {
          make_p(kc);
#pragma unroll
          for (int nd = 0; nd < 8; nd += 2) {
            uint32_t v0, v1, v2, v3;
            ldsm_x4a(v0, v1, v2, v3, sV + off[kc] + nd * 1024);
            if (kc == 0) {
              mma_u8s8_c(acc[nd], pa[kc], v0, v1, magic);
              mma_u8s8_c(acc[nd + 1], pa[kc], v2, v3, magic);
            } else {
              mma_u8s8(acc[nd], pa[kc], v0, v1);
              mma_u8s8(acc[nd + 1], pa[kc], v2, v3);
            }
          }
          if (MX && kc == MXE && !(wg == 1 && last)) bar_arrive(2 - wg);
        }
        if constexpr (LZ) {
          if (clean) {  // O' = a (O' - MAGIC_F B), B = 0 (warp-uniform branch)
            const float k0 = -MAGIC_F * a0 * B0, k1 = -MAGIC_F * a1 * B1;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
              of[i][0] = fmaf(of[i][0], a0, k0); of[i][1] = fmaf(of[i][1], a0, k0);
              of[i][2] = fmaf(of[i][2], a1, k1); of[i][3] = fmaf(of[i][3], a1, k1);
            }
            B0 = 0.f; B1 = 0.f;
          }
          B0 += ts0; B1 += ts1;
        }
        l0 = fmaf(ts0, i2f(rs0), l0 * a0);
        l1 = fmaf(ts1, i2f(rs1), l1 * a1);
#pragma unroll
        for (int kc = 0; kc < KC; ++kc) {
#pragma unroll
          for (int nd = 0; nd < 8; nd += 2) {
            uint32_t v0, v1, v2, v3;
            ldsm_x4a(v0, v1, v2, v3, sV + off[kc] + (8 + nd) * 1024);
            if (kc == 0) {
              mma_u8s8_c(acc[8 + nd], pa[kc], v0, v1, magic);
              mma_u8s8_c(acc[8 + nd + 1], pa[kc], v2, v3, magic);
            } else {
              mma_u8s8(acc[8 + nd], pa[kc], v0, v1);
              mma_u8s8(acc[8 + nd + 1], pa[kc], v2, v3);
            }
          }
          if constexpr (LZ) {
            fold1(of[2 * kc], acc[2 * kc], ts0, ts1);
            fold1(of[2 * kc + 1], acc[2 * kc + 1], ts0, ts1);
          } else {
            fold(of[2 * kc], acc[2 * kc], ts0, ts1, a0, a1);
            fold(of[2 * kc + 1], acc[2 * kc + 1], ts0, ts1, a0, a1);
          }
        }
        if (PP && !(wg == 1 && last)) bar_arrive(2 - wg);
        __syncwarp();
        if (lane == 0) mbar_arrive(vempty + 8 * st);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          if constexpr (LZ) fold1(of[8 + i], acc[8 + i], ts0, ts1);
          else fold(of[8 + i], acc[8 + i], ts0, ts1, a0, a1);
        }
      } else {
#pragma unroll
        for (int kc = 0; kc < KC; ++kc) make_p(kc);
        if (MX && !(wg == 1 && last)) bar_arrive(2 - wg);
        if (PP) bar_sync(1 + wg);
#pragma unroll
        for (int kc = 0; kc < KC; ++kc) {
#pragma unroll
          for (int nd = 0; nd < 16; nd += 2) {
            uint32_t v0, v1, v2, v3;
            ldsm_x4a(v0, v1, v2, v3, sV + off[kc] + nd * 1024);
            if (kc == 0) {
              mma_u8s8_c(acc[nd], pa[kc], v0, v1, magic);
              mma_u8s8_c(acc[nd + 1], pa[kc], v2, v3, magic);
            } else {
              mma_u8s8(acc[nd], pa[kc], v0, v1);
              mma_u8s8(acc[nd + 1], pa[kc], v2, v3);
            }
          }
        }
        if (PP && !(wg == 1 && last)) bar_arrive(2 - wg);
        __syncwarp();
        if (lane == 0) mbar_arrive(vempty + 8 * st);
        l0 = fmaf(ts0, i2f(rs0), l0 * a0);
        l1 = fmaf(ts1, i2f(rs1), l1 * a1);
#pragma unroll
        for (int i = 0; i < 16; ++i) fold(of[i], acc[i], ts0, ts1, a0, a1);
      }
      // refill this V / KSB stage with tile n+2
      if (prod_warp) {
        if (vu < nunits) {
          mbar_wait(vempty + 8 * st, ph);
          if (lane == 0) load_v(st);
        }
        __syncwarp();
      }
    }

    if constexpr (LZ) {  // remove the remaining bias
      const float k0 = -MAGIC_F * B0, k1 = -MAGIC_F * B1;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        of[i][0] += k0; of[i][1] += k0; of[i][2] += k1; of[i][3] += k1;
      }
    }
    // ---- epilogue (config 80's): O / l * v_scale + v_mean -> bf16, [S][H][D]
    float l0s = l0 + __shfl_xor_sync(0xffffffff, l0, 1); l0s += __shfl_xor_sync(0xffffffff, l0s, 2);
    float l1s = l1 + __shfl_xor_sync(0xffffffff, l1, 1); l1s += __shfl_xor_sync(0xffffffff, l1s, 2);
    const float il0 = 1.f / l0s, il1 = 1.f / l1s;
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
}

// ------------------------------------------------------------------------------------------------
// attn_cake_m: FA3-style merged tensor phase. Each warp group alternates
//   tensor phase  [ P.V(n) | Q.K^T(n+1) key-pair-outer, with dequant + max of (n+1) and the fold of
//                   P.V(n) interleaved between the IMMAs ]
//   softmax phase [ row max shuffles, exp2, u8 packing of tile n+1 ]
// so there are 2 hand-offs per tile instead of 4, the softmax phase is short (no dequant, no fold),
// and the int32 P.V accumulator is retired block by block while the scores of n+1 fill in
// (peak live set ~170 registers instead of ~225). Same arithmetic as config 80.
// ------------------------------------------------------------------------------------------------
template <bool PP>
__global__ void __launch_bounds__(NT, 1)
attn_cake_m(const __grid_constant__ CUtensorMap Kmap, const __grid_constant__ CUtensorMap Vmap,
            const int8_t* __restrict__ Qg, const float* __restrict__ QS, const float2* __restrict__ KSB,
            const float* __restrict__ VS, const float* __restrict__ VM, __nv_bfloat16* __restrict__ O,
            long o_ss, long o_sh, int S, int Spad, int ntq, int nunits, float scale_log2, int, float, int,
            float) {
  constexpr int NB = 16, KC = 4;
  extern __shared__ uint8_t smem_raw[];
  const uint32_t s0 = (smem_u32(smem_raw) + 1023u) & ~1023u;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tq = lane & 3, wg = warp >> 2;
  const uint32_t kfull = s0, vfull = s0 + 16, kempty = s0 + 32, vempty = s0 + 48;
  const int ntiles = Spad / BN;
  const int G = gridDim.x;

  if (tid == 0) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      mbar_init(kfull + 8 * i, 1); mbar_init(vfull + 8 * i, 1);
      mbar_init(kempty + 8 * i, 8); mbar_init(vempty + 8 * i, 8);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  int ku = blockIdx.x, kkt = 0, vu = blockIdx.x, vkt = 0;
  auto load_k = [&](int st) {
    const int h = ku / ntq;
    mbar_expect(kfull + 8 * st, TILE);
    tma3(s0 + SM_K + st * TILE, &Kmap, 0, kkt * BN, h, kfull + 8 * st);
    if (++kkt == ntiles) { kkt = 0; ku += G; }
  };
  auto load_v = [&](int st) {
    const int h = vu / ntq;
    mbar_expect(vfull + 8 * st, TILE + BN * 8);
    tma3(s0 + SM_V + st * TILE, &Vmap, vkt * BN, 0, h, vfull + 8 * st);
    bulk1(s0 + SM_KS + st * BN * 8, KSB + (long)h * Spad + vkt * BN, BN * 8, vfull + 8 * st);
    if (++vkt == ntiles) { vkt = 0; vu += G; }
  };
  const bool prod_warp = warp == 4;
  if (prod_warp && lane == 0) {
    for (int st = 0; st < 2; ++st) {
      if (ku < nunits) load_k(st);
      if (vu < nunits) load_v(st);
    }
  }
  // producer: refill the K stage of tile nk / the V stage of tile nv once all 8 warps released it
  auto refill_k = [&](int nk) {
    if (prod_warp) {
      if (ku < nunits) {
        mbar_wait(kempty + 8 * (nk & 1), (nk >> 1) & 1);
        if (lane == 0) load_k(nk & 1);
      }
      __syncwarp();
    }
  };
  auto refill_v = [&](int nv) {
    if (prod_warp) {
      if (vu < nunits) {
        mbar_wait(vempty + 8 * (nv & 1), (nv >> 1) & 1);
        if (lane == 0) load_v(nv & 1);
      }
      __syncwarp();
    }
  };

  uint32_t off[4];
  {
    const int m = lane >> 3, r = (m >> 1) * 8 + (lane & 7);
#pragma unroll
    for (int k = 0; k < 4; ++k) off[k] = r * 128 + (((k * 2 + (m & 1)) ^ (lane & 7)) << 4);
  }
  const int magic = MAGIC_I;
  auto fold = [&](float* o, const int* a, float ts0, float ts1, float a0, float a1) {
    o[0] = fmaf(ts0, __int_as_float(a[0]), fmaf(o[0], a0, -ts0 * MAGIC_F));
    o[1] = fmaf(ts0, __int_as_float(a[1]), fmaf(o[1], a0, -ts0 * MAGIC_F));
    o[2] = fmaf(ts1, __int_as_float(a[2]), fmaf(o[2], a1, -ts1 * MAGIC_F));
    o[3] = fmaf(ts1, __int_as_float(a[3]), fmaf(o[3], a1, -ts1 * MAGIC_F));
  };

  if (PP && wg == 1) bar_arrive(1);
  int n = 0;
  for (int u = blockIdx.x; u < nunits; u += G) {
    const int h = u / ntq, m0 = (u % ntq) * BM;
    const bool last_unit = u + G >= nunits;
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

    float t[NB][4];
    float tm0, tm1;
    int acc[16][4];
    float a0 = 1.f, a1 = 1.f, ts0 = 0.f, ts1 = 0.f;  // rescale / tile scale of the tile in acc
    // Q.K^T of tile nq (key-pair-outer) with dequant + running tile max per 16-key block; with
    // do_fold, the P.V accumulator of the previous tile is folded two blocks per key pair
    auto qk = [&](int nq, bool do_fold) {
      const uint32_t sK = s0 + SM_K + (nq & 1) * TILE, sKS = s0 + SM_KS + (nq & 1) * BN * 8;
      tm0 = -INFINITY; tm1 = -INFINITY;
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
        if (do_fold && p < 4) {  // half 1 of the previous tile's P.V (half 0 was folded in P.V)
          fold(of[8 + 2 * p], acc[8 + 2 * p], ts0, ts1, a0, a1);
          fold(of[8 + 2 * p + 1], acc[8 + 2 * p + 1], ts0, ts1, a0, a1);
        }
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int nb = 2 * p + j;
          const float4 kb = lds128(sKS + nb * 64 + tq * 16);
          t[nb][0] = fmaf(__int_as_float(sc[j][0]), kb.x, kb.y);
          t[nb][1] = fmaf(__int_as_float(sc[j][1]), kb.z, kb.w);
          t[nb][2] = fmaf(__int_as_float(sc[j][2]), kb.x, kb.y);
          t[nb][3] = fmaf(__int_as_float(sc[j][3]), kb.z, kb.w);
          tm0 = fmaxf(tm0, fmaxf(t[nb][0], t[nb][1]));
          tm1 = fmaxf(tm1, fmaxf(t[nb][2], t[nb][3]));
        }
      }
    };
    uint32_t pa[KC][4];
    // softmax of the scores in t: new running max, a / ts, P (u8, vs the tile max), row sums
    auto softmax = [&]() {
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
        float e[4][4];
#pragma unroll
        for (int u2 = 0; u2 < 4; ++u2) {
          const int nb = kc * 4 + u2;
          e[u2][0] = ex2(fmaf(t[nb][0], c0, b0)); e[u2][1] = ex2(fmaf(t[nb][1], c0, b0));
          e[u2][2] = ex2(fmaf(t[nb][2], c1, b1)); e[u2][3] = ex2(fmaf(t[nb][3], c1, b1));
        }
        pa[kc][0] = pack_u8(e[0][0], e[0][1], e[1][0], e[1][1]);
        pa[kc][1] = pack_u8(e[0][2], e[0][3], e[1][2], e[1][3]);
        pa[kc][2] = pack_u8(e[2][0], e[2][1], e[3][0], e[3][1]);
        pa[kc][3] = pack_u8(e[2][2], e[2][3], e[3][2], e[3][3]);
        rs0 = __dp4a(pa[kc][0], 0x01010101u, rs0); rs0 = __dp4a(pa[kc][2], 0x01010101u, rs0);
        rs1 = __dp4a(pa[kc][1], 0x01010101u, rs1); rs1 = __dp4a(pa[kc][3], 0x01010101u, rs1);
      }
      l0 = fmaf(ts0, i2f(rs0), l0 * a0);
      l1 = fmaf(ts1, i2f(rs1), l1 * a1);
    };

    // ---- first tensor phase of the unit: Q.K^T(n) alone
    mbar_wait(kfull + 8 * (n & 1), (n >> 1) & 1);
    mbar_wait(vfull + 8 * (n & 1), (n >> 1) & 1);
    if (PP) bar_sync(1 + wg);
    qk(n, false);
    if (PP) bar_arrive(2 - wg);
    __syncwarp();
    if (lane == 0) mbar_arrive(kempty + 8 * (n & 1));
    refill_k(n);
    softmax();

#pragma unroll 1
    for (int kt = 0; kt < ntiles; ++kt, ++n) {
      const bool more = kt + 1 < ntiles;
      const bool last = last_unit && !more;
      const uint32_t sV = s0 + SM_V + (n & 1) * TILE;
      if (more) {
        mbar_wait(kfull + 8 * ((n + 1) & 1), ((n + 1) >> 1) & 1);
        mbar_wait(vfull + 8 * ((n + 1) & 1), ((n + 1) >> 1) & 1);
      }
      // ---- tensor phase: P.V(n) [+ Q.K^T(n+1) with fold(n)]
      if (PP) bar_sync(1 + wg);
#pragma unroll
      for (int half = 0; half < 2; ++half) {
#pragma unroll
        for (int kc = 0; kc < KC; ++kc) {
#pragma unroll
          for (int nd = half * 8; nd < half * 8 + 8; nd += 2) {
            uint32_t v0, v1, v2, v3;
            ldsm_x4a(v0, v1, v2, v3, sV + off[kc] + nd * 1024);
            if (kc == 0) {
              mma_u8s8_c(acc[nd], pa[kc], v0, v1, magic);
              mma_u8s8_c(acc[nd + 1], pa[kc], v2, v3, magic);
            } else {
              mma_u8s8(acc[nd], pa[kc], v0, v1);
              mma_u8s8(acc[nd + 1], pa[kc], v2, v3);
            }
          }
          if (half == 1) {  // half 0's fold between half 1's IMMAs
            fold(of[2 * kc], acc[2 * kc], ts0, ts1, a0, a1);
            fold(of[2 * kc + 1], acc[2 * kc + 1], ts0, ts1, a0, a1);
          }
        }
      }
      if (more) qk(n + 1, true);
      if (PP && !(wg == 1 && last)) bar_arrive(2 - wg);
      __syncwarp();
      if (lane == 0) {
        mbar_arrive(vempty + 8 * (n & 1));
        if (more) mbar_arrive(kempty + 8 * ((n + 1) & 1));
      }
      refill_v(n);
      if (more) {
        refill_k(n + 1);
        softmax();
      } else {
#pragma unroll
        for (int i = 8; i < 16; ++i) fold(of[i], acc[i], ts0, ts1, a0, a1);
      }
    }

    float l0s = l0 + __shfl_xor_sync(0xffffffff, l0, 1); l0s += __shfl_xor_sync(0xffffffff, l0s, 2);
    float l1s = l1 + __shfl_xor_sync(0xffffffff, l1, 1); l1s += __shfl_xor_sync(0xffffffff, l1s, 2);
    const float il0 = 1.f / l0s, il1 = 1.f / l1s;
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
}

// ------------------------------------------------------------------------------------------------
// host side
// ------------------------------------------------------------------------------------------------
inline PFN_cuTensorMapEncodeTiled_v12000 encode_fn() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = [] {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q;
    TORCH_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q) ==
                        cudaSuccess && q == cudaDriverEntryPointSuccess && p,
                "cuTensorMapEncodeTiled not available");
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
  }();
  return fn;
}

// 3D uint8 map {inner, mid, H}, box {128, 128, 1}, 128 B swizzle
inline CUtensorMap make_map(const void* base, uint64_t inner, uint64_t mid, uint64_t H) {
  CUtensorMap m;
  const cuuint64_t dims[3] = {inner, mid, H};
  const cuuint64_t strides[2] = {inner, inner * mid};
  const cuuint32_t box[3] = {128, 128, 1}, es[3] = {1, 1, 1};
  const CUresult r = encode_fn()(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, const_cast<void*>(base), dims, strides, box, es,
                                 CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                 CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  return m;
}

// q8 [H][Spad][128], k8 [H][Spad][128], ksb [H][Spad][2], vt [H][128][Spad] (config 80's pre-pass)
// TB: keys [0, nbias) get bias_log2 (exp2 domain) on every query's logit (attn_cake only, VAR < 16)
// SMX: softmax variant (attn_cake); nclean / tau: its lazy-rescale parameters (SMX bit 1)
template <int VAR, int SYNC, bool TB = false, int SMX = 0>
void launch(const torch::Tensor& q8, const torch::Tensor& qs, const torch::Tensor& k8, const torch::Tensor& ksb,
            const torch::Tensor& vt, const torch::Tensor& vs, const torch::Tensor& vm, torch::Tensor& out, int S,
            int Spad, float scale_log2, cudaStream_t stream, int grid_override = 0, int nbias = 0,
            float bias_log2 = 0.f, int nclean = 8, float tau = 8.f) {
  static_assert(!TB || VAR < 16, "text bias: attn_cake only");
  static_assert(SMX == 0 || VAR < 16, "softmax variants: attn_cake only");
  const int H = q8.size(0);
  const CUtensorMap km = make_map(k8.data_ptr(), D, Spad, H);
  const CUtensorMap vmap = make_map(vt.data_ptr(), Spad, D, H);
  auto kern = VAR >= 16 ? attn_cake_m<SYNC == 1> : attn_cake<VAR & 15, SYNC, TB, SMX>;
  static bool once = [&] {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_ALLOC);
    return true;
  }();
  (void)once;
  const int ntq = Spad / BM, nunits = ntq * H;
  const int nsm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int grid = std::min(nunits, grid_override > 0 ? grid_override : nsm);
  kern<<<grid, NT, SMEM_ALLOC, stream>>>(
      km, vmap, q8.data_ptr<int8_t>(), qs.data_ptr<float>(), reinterpret_cast<const float2*>(ksb.data_ptr<float>()),
      vs.data_ptr<float>(), vm.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), out.stride(0),
      out.stride(1), S, Spad, ntq, nunits, scale_log2, nbias, bias_log2, nclean, tau);
}

}  // namespace cake
