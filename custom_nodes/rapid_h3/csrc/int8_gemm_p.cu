// Persistent INT8 row-scaled GEMM for GB10 (sm_121), experimental (2026-10-24):
//   out[M,N] = bf16(float(XQ[M,K] . WQ[N,K]^T) * xs[M] * ws[N])
// Same int32 accumulation and the same epilogue rounding as fp8_rowwise.cu's CUTLASS kernel
// (config 6, 128x256x64, 4 stages, 8 warps of 64x64), so outputs are bit-identical. What differs:
//   - one CTA per SM loops over output tiles, and the cp.async pipeline runs straight through tile
//     boundaries, so the next tile's first stages load while this tile finishes (no refill bubble);
//   - the epilogue writes bf16 pairs straight from the accumulator registers (no smem staging).
// Result (bench/gemm_persistent.py, docs/kernels.md "Lossless kernels, round 3"): bit-identical, but
// 185-191 TFLOPS against CUTLASS's 198-205. The main loop alone (grid < 0: no epilogue) runs at
// 205-213, so the loss is the epilogue, which all 8 warps run at once with the tensor pipe idle.
// Not wired in.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {

constexpr int BM = 128, BN = 256, BK = 64, STAGES = 4, NT = 256;
constexpr int A_BYTES = BM * BK, B_BYTES = BN * BK, STAGE = A_BYTES + B_BYTES, SMEM = STAGE * STAGES;
constexpr int GROUP = 8;  // M-tiles per rasterization group (CUTLASS GemmIdentityThreadblockSwizzle<8>-like)

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool pred) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src), "r"(pred ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldsm_x4(uint32_t* r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void mma_s8(int* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// 64-byte rows (one BK slice), 16 B chunks XOR-swizzled so any 8 consecutive rows of one chunk
// column land in 8 distinct 16 B bank groups (ldmatrix conflict-free)
__device__ __forceinline__ int swz64(int row, int chunk) { return row * 64 + ((chunk ^ ((row >> 1) & 3)) << 4); }

__global__ void __launch_bounds__(NT, 1)
gemm_i8_persistent(const int8_t* __restrict__ A, const int8_t* __restrict__ B, const float* __restrict__ xs,
                   const float* __restrict__ ws, __nv_bfloat16* __restrict__ out, int M, int N, int K, int epi) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tq = lane & 3;
  const int wm = warp >> 2, wn = warp & 3;  // 2 x 4 warps, 64 x 64 each
  const int mt = (M + BM - 1) / BM, nt = N / BN, KT = K / BK;
  const int ntiles = mt * nt;
  // tiles of this CTA: blockIdx.x, +gridDim.x, ...; the k-iterations are flattened over all of them
  const int my_tiles = (int)blockIdx.x < ntiles ? (ntiles - 1 - blockIdx.x) / gridDim.x + 1 : 0;
  const int total = my_tiles * KT;

  auto tile_mn = [&](int t, int& m0, int& n0) {
    const int grp = t / (GROUP * nt), rem = t % (GROUP * nt);
    const int gm = min(GROUP, mt - grp * GROUP);
    m0 = (grp * GROUP + rem % gm) * BM;
    n0 = (rem / gm) * BN;
  };

  // loader state: next k-iteration to load (tile lt, k-tile lkt, stage lst)
  const int lrow = tid >> 2, lch = tid & 3;
  const int a_s0 = swz64(lrow, lch), a_s1 = swz64(lrow + 64, lch);
  int b_s[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) b_s[i] = A_BYTES + swz64(lrow + 64 * i, lch);
  int lt = blockIdx.x, lkt = 0, lst = 0;
  const int8_t *pa0 = nullptr, *pa1 = nullptr, *pb = nullptr;
  bool va0 = false, va1 = false;
  auto set_tile = [&]() {
    int m0, n0;
    tile_mn(lt, m0, n0);
    const int r0 = m0 + lrow, r1 = r0 + 64;
    va0 = r0 < M; va1 = r1 < M;
    pa0 = A + (long)min(r0, M - 1) * K + lch * 16;
    pa1 = A + (long)min(r1, M - 1) * K + lch * 16;
    pb = B + (long)(n0 + lrow) * K + lch * 16;
  };
  auto load_next = [&]() {
    if (lkt == 0) set_tile();
    uint8_t* st = smem + lst * STAGE;
    const int k0 = lkt * BK;
    cp_async16(st + a_s0, pa0 + k0, va0);
    cp_async16(st + a_s1, pa1 + k0, va1);
#pragma unroll
    for (int i = 0; i < 4; ++i) cp_async16(st + b_s[i], pb + (long)64 * i * K + k0, true);
    if (++lkt == KT) { lkt = 0; lt += gridDim.x; }
    if (++lst == STAGES) lst = 0;
  };

  // every stage is filled up front; iteration it refills its own stage with iteration it + STAGES
#pragma unroll
  for (int s = 0; s < STAGES; ++s) {
    if (s < total) load_next();
    cp_async_commit();
  }

  int acc[4][8][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0;

  // ldmatrix lane offsets: A matrix lm -> rows (lm&1)*8 + lr, chunk (lm>>1); B matrix lm -> n8 tile
  // (lm>>1), chunk (lm&1). The swizzle term ((row >> 1) & 3) depends only on lr.
  const int lm = lane >> 3, lr = lane & 7, sx = (lr >> 1) & 3;
  uint32_t a_off[2], b_off[2];
#pragma unroll
  for (int ks = 0; ks < 2; ++ks) {
    a_off[ks] = (wm * 64 + (lm & 1) * 8 + lr) * 64 + (((ks * 2 + (lm >> 1)) ^ sx) << 4);
    b_off[ks] = A_BYTES + (wn * 64 + (lm >> 1) * 8 + lr) * 64 + (((ks * 2 + (lm & 1)) ^ sx) << 4);
  }
  const uint32_t sbase = smem_u32(smem);
  uint32_t af[2][4][4], bf[2][4][4];
  auto frag_load = [&](int buf, int stage, int ks) {
    const uint32_t s = sbase + stage * STAGE;
#pragma unroll
    for (int i = 0; i < 4; ++i) ldsm_x4(af[buf][i], s + a_off[ks] + i * 16 * 64);
#pragma unroll
    for (int j = 0; j < 4; ++j) ldsm_x4(bf[buf][j], s + b_off[ks] + j * 16 * 64);
  };
  auto mma_all = [&](int buf) {
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int j = 0; j < 8; ++j)
        mma_s8(acc[i][j], af[buf][i], bf[buf][j >> 1][(j & 1) * 2], bf[buf][j >> 1][(j & 1) * 2 + 1]);
  };

  cp_async_wait<STAGES - 1>();
  __syncthreads();
  if (total > 0) frag_load(0, 0, 0);
  int st = 0, kt = 0, tile = blockIdx.x;
  for (int it = 0; it < total; ++it) {
    frag_load(1, st, 1);
    mma_all(0);
    // stage `st` is fully in registers for every warp after this barrier: refill it
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    if (it + STAGES < total) load_next();
    cp_async_commit();
    const int nst = st + 1 == STAGES ? 0 : st + 1;
    if (it + 1 < total) frag_load(0, nst, 0);
    mma_all(1);
    st = nst;

    if (++kt == KT && epi) {  // tile done: epilogue straight from registers
      kt = 0;
      int m0, n0;
      tile_mn(tile, m0, n0);
      tile += gridDim.x;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int r = m0 + wm * 64 + i * 16 + g;
        const float x0 = r < M ? xs[r] : 0.f, x1 = r + 8 < M ? xs[r + 8] : 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int c = n0 + wn * 64 + j * 8 + tq * 2;
          const float2 w = *reinterpret_cast<const float2*>(ws + c);
          // CUTLASS EVT order: ((float(acc) * xs) * ws) + bias(0), round to bf16
          if (r < M)
            *reinterpret_cast<__nv_bfloat162*>(out + (long)r * N + c) = __floats2bfloat162_rn(
                (float(acc[i][j][0]) * x0) * w.x + 0.f, (float(acc[i][j][1]) * x0) * w.y + 0.f);
          if (r + 8 < M)
            *reinterpret_cast<__nv_bfloat162*>(out + (long)(r + 8) * N + c) = __floats2bfloat162_rn(
                (float(acc[i][j][2]) * x1) * w.x + 0.f, (float(acc[i][j][3]) * x1) * w.y + 0.f);
          acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0;
        }
      }
    }
  }
  cp_async_wait<0>();
}

}  // namespace

// grid: CTAs to launch (0 = one per SM). grid < 0 launches |grid| CTAs and skips the epilogue
// (main-loop-only timing; the output is left unwritten).
at::Tensor int8_gemm_persistent(at::Tensor xq, at::Tensor wq, at::Tensor xs, at::Tensor ws, int64_t grid) {
  TORCH_CHECK(xq.is_contiguous() && wq.is_contiguous() && xs.is_contiguous() && ws.is_contiguous());
  TORCH_CHECK(xq.scalar_type() == at::kChar && wq.scalar_type() == at::kChar, "int8 inputs");
  const int M = xq.size(0), K = xq.size(1), N = wq.size(0);
  TORCH_CHECK(wq.size(1) == K && K % BK == 0 && N % BN == 0, "K % 64 and N % 256 required");
  c10::cuda::CUDAGuard g(xq.device());
  auto out = at::empty({M, N}, xq.options().dtype(at::kBFloat16));
  static bool once = [] {
    cudaFuncSetAttribute(gemm_i8_persistent, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    return true;
  }();
  (void)once;
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, xq.device().index());
  const int ntiles = (M + BM - 1) / BM * (N / BN);
  const int nb = grid != 0 ? abs((int)grid) : std::min(sms, ntiles);
  gemm_i8_persistent<<<nb, NT, SMEM, at::cuda::getCurrentCUDAStream()>>>(
      xq.data_ptr<int8_t>(), wq.data_ptr<int8_t>(), xs.data_ptr<float>(), ws.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), M, N, K, grid >= 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("int8_gemm_persistent", &int8_gemm_persistent, "persistent INT8 row-scaled GEMM (sm_121, experimental)",
        py::arg("xq"), py::arg("wq"), py::arg("xs"), py::arg("ws"), py::arg("grid") = 0);
}
