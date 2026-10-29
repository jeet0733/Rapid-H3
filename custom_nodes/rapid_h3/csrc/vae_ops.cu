// Video-VAE decoder kernels (sm_121), bit-identical to the comfy-kitchen 0.2.37 path they replace
// (re-checked on 0.2.39, the version in the rapid-h3/h3-comfyui:v0.42.1 image), except the w1 + SwiGLU GEMM below
// (RAPIDH3_VAE_FAST >= 3, part of the default level 4; not bit-identical, see there).
//
// 1. vae_out_quant: the to_out INT8 ConvRot quantizer reading the int8-attention output in its
//    native head-major [B, H, S, 64] layout, with torch.nan_to_num applied on load. Replaces
//    `out.transpose(1, 2).reshape(...)` + `torch.nan_to_num(out)` + ck's quantizer (two fp16
//    round trips of the attention output). The per-row math is comfy-kitchen's
//    convrot64_quantize_row (Apache-2.0, Copyright (c) 2025 Comfy Org), copied unchanged.
// 2. vae_gemm: comfy-kitchen's fused INT8 GEMM epilogues (plain + bias, and the pre-norm block's
//    residual addcmul), same EVT ops in the same order, with our tile choice per shape.
// 3. vae_qk_rope_quant (RAPIDH3_VAE_FAST=4): the attention's q/k RMSNorm + split-half rope fused into comfy-kitchen's
//    SageAttention Q/K quantizer and K-anchor detector (see the round-5 section below).
//
// Built with --use_fast_math like comfy-kitchen, so the quantizer's division rounds identically.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cfloat>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cute/tensor.hpp>

namespace {

constexpr int kThreadsPerWarp = 32;
constexpr int kConvRotGroup = 256;

// ---- copied from comfy-kitchen ops/int8_linear.cu (fp16 specializations only) -------------------
__device__ __forceinline__ float to_float(half v) { return __half2float(v); }
__device__ __forceinline__ half from_float_h(float v) { return __float2half_rn(v); }
__device__ __forceinline__ float finite_absmax_for_int8_scale_h(float abs_max) { return fminf(abs_max, 65504.0f); }
__device__ __forceinline__ float quant_div_float_to_float_h(float val, float scale) {
  const float scale_t = to_float(from_float_h(scale));
  return to_float(from_float_h(to_float(from_float_h(val)) / scale_t));
}
__device__ __forceinline__ float warp_reduce_max(float v) {
  for (int offset = kThreadsPerWarp / 2; offset > 0; offset >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
  return v;
}
template <int NUM_WARPS>
__device__ __forceinline__ float block_reduce_max_t(float v, float* warp_smem, float* block_smem) {
  const int lane = threadIdx.x & (kThreadsPerWarp - 1);
  const int wid = threadIdx.x >> 5;
  v = warp_reduce_max(v);
  if (lane == 0) warp_smem[wid] = v;
  __syncthreads();
  if (wid == 0) {
    float total = lane < NUM_WARPS ? warp_smem[lane] : 0.0f;
    total = warp_reduce_max(total);
    if (lane == 0) *block_smem = total;
  }
  __syncthreads();
  return *block_smem;
}
template <int S>
__device__ __forceinline__ void convrot_fht_stage64(const float* __restrict__ src, float* __restrict__ dst, int lane) {
  const int base = (lane % S) + (lane / S) * (4 * S);
  const float x0 = src[base], x1 = src[base + S], x2 = src[base + 2 * S], x3 = src[base + 3 * S];
  dst[base] = 0.5f * (x0 + x1 + x2 - x3);
  dst[base + S] = 0.5f * (x0 + x1 - x2 + x3);
  dst[base + 2 * S] = 0.5f * (x0 - x1 + x2 + x3);
  dst[base + 3 * S] = 0.5f * (-x0 + x1 + x2 + x3);
}
template <int S>
__device__ __forceinline__ float convrot_fht_stage64_store_absmax(const float* __restrict__ src, float* __restrict__ output, int lane) {
  const int base = (lane % S) + (lane / S) * (4 * S);
  const float x0 = src[base], x1 = src[base + S], x2 = src[base + 2 * S], x3 = src[base + 3 * S];
  const float y0 = 0.5f * (x0 + x1 + x2 - x3);
  const float y1 = 0.5f * (x0 + x1 - x2 + x3);
  const float y2 = 0.5f * (x0 - x1 + x2 + x3);
  const float y3 = 0.5f * (-x0 + x1 + x2 + x3);
  output[base] = y0; output[base + S] = y1; output[base + 2 * S] = y2; output[base + 3 * S] = y3;
  return fmaxf(fmaxf(fabsf(y0), fabsf(y1)), fmaxf(fabsf(y2), fabsf(y3)));
}
// ---- end of copied helpers ----------------------------------------------------------------------

// torch.nan_to_num for fp16: NaN -> 0, +-inf -> +-65504
__device__ __forceinline__ float load_nan_to_num(const half* p) {
  const half h = *p;
  if (__hisnan(h)) return 0.0f;
  if (__hisinf(h)) return __hgt(h, __float2half(0.f)) ? 65504.0f : -65504.0f;
  return to_float(h);
}

// One block per output row (b, s). Row column c = head * D + d reads o[b, head, s, d].
template <int BLOCK_THREADS>
__global__ void vae_out_quant_kernel(const half* __restrict__ o, int8_t* __restrict__ q, float* __restrict__ scales,
                                     int H, int S, int D) {
  constexpr int kGroupThreads = 64;
  constexpr int kGroupsInFlight = BLOCK_THREADS / kGroupThreads;
  constexpr int kWarps = BLOCK_THREADS / kThreadsPerWarp;
  extern __shared__ float smem[];
  __shared__ float warp_smem[kWarps];
  __shared__ float block_smem;
  const int K = H * D;
  float* row_buf = smem;
  float* tmp = smem + K;

  const int row = blockIdx.x, b = row / S, s = row % S;
  const half* ob = o + ((int64_t)b * H * S + s) * D;  // + head * S * D + d
  const int tid = threadIdx.x, sub = tid / kGroupThreads, lane = tid % kGroupThreads;
  const int n_groups = K / kConvRotGroup;
  float* buf0 = tmp + sub * (2 * kConvRotGroup);
  float* buf1 = buf0 + kConvRotGroup;
  float abs_max = 0.0f;

  auto ld = [&](int col) { return load_nan_to_num(ob + (int64_t)(col / D) * S * D + (col % D)); };

  const int iters = (n_groups + kGroupsInFlight - 1) / kGroupsInFlight;
  for (int it = 0; it < iters; ++it) {
    const int group = it * kGroupsInFlight + sub;
    const bool active = group < n_groups;
    const int base = lane * 4;
    const int group_col = group * kConvRotGroup;
    const int col = group_col + base;
    const float x0 = active ? ld(col) : 0.0f;
    const float x1 = active ? ld(col + 1) : 0.0f;
    const float x2 = active ? ld(col + 2) : 0.0f;
    const float x3 = active ? ld(col + 3) : 0.0f;
    buf1[base] = 0.5f * (x0 + x1 + x2 - x3);
    buf1[base + 1] = 0.5f * (x0 + x1 - x2 + x3);
    buf1[base + 2] = 0.5f * (x0 - x1 + x2 + x3);
    buf1[base + 3] = 0.5f * (-x0 + x1 + x2 + x3);
    __syncthreads();
    convrot_fht_stage64<4>(buf1, buf0, lane);
    __syncthreads();
    convrot_fht_stage64<16>(buf0, buf1, lane);
    __syncthreads();
    if (active) abs_max = fmaxf(abs_max, convrot_fht_stage64_store_absmax<64>(buf1, row_buf + group_col, lane));
    __syncthreads();
  }
  abs_max = block_reduce_max_t<kWarps>(abs_max, warp_smem, &block_smem);
  const float scale = fmaxf(finite_absmax_for_int8_scale_h(abs_max) * (1.0f / 127.0f), 1.0e-30f);
  int8_t* q_row = q + (int64_t)row * K;
  for (int c = tid; c < K; c += BLOCK_THREADS) {
    float quantized = nearbyintf(quant_div_float_to_float_h(row_buf[c], scale));
    quantized = fminf(127.0f, fmaxf(-128.0f, quantized));
    q_row[c] = static_cast<int8_t>(quantized);
  }
  if (tid == 0) scales[row] = scale;
}

// ---- GEMMs: comfy-kitchen FusedInt8Gemm / FusedInt8GemmResidual epilogues, fp16 out -------------
using Out = cutlass::half_t;

template <class TB, class Warp, int Stages, bool Residual>
bool run_gemm(const at::Tensor& xq, const at::Tensor& wq, const at::Tensor& xs, const at::Tensor& ws,
              const at::Tensor& bias, const at::Tensor* rscale, const at::Tensor* resid, at::Tensor& out) {
  using namespace cutlass::epilogue::threadblock;
  using namespace cute;
  const int M = xq.size(0), K = xq.size(1), N = wq.size(0);
  constexpr int AlignC = 8;
  using Inst = cutlass::gemm::GemmShape<16, 8, 32>;
  using ThreadMap = OutputTileThreadLayout<TB, Warp, Out, AlignC, 1>;
  using Accum = VisitorAccFetch;
  using XScale = VisitorColBroadcast<ThreadMap, float, Stride<_1, _0, int32_t>>;
  using WScale = VisitorRowBroadcast<ThreadMap, float, Stride<_0, _1, int32_t>>;
  using Bias = VisitorRowBroadcast<ThreadMap, Out, Stride<_0, _1, int32_t>>;
  using Mul0 = VisitorCompute<cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
  using EVT0 = Sm80EVT<Mul0, Accum, XScale>;
  using EVT1 = Sm80EVT<Mul0, EVT0, WScale>;
  using Add2 = VisitorCompute<cutlass::plus, Out, float, cutlass::FloatRoundStyle::round_to_nearest>;
  using EVT2 = Sm80EVT<Add2, EVT1, Bias>;
  using StoreD = VisitorAuxStore<ThreadMap, Out, cutlass::FloatRoundStyle::round_to_nearest, Stride<int64_t, _1, int64_t>>;
  using RScale = VisitorRowBroadcast<ThreadMap, Out, Stride<_0, _1, int32_t>>;
  using Resid = VisitorAuxLoad<ThreadMap, Out, Stride<int64_t, _1, int64_t>>;
  using EVT3 = Sm80EVT<Mul0, EVT2, RScale>;
  using Add4 = VisitorCompute<cutlass::plus, Out, float, cutlass::FloatRoundStyle::round_to_nearest>;
  using EVT4 = Sm80EVT<Add4, EVT3, Resid>;
  using EVTD = std::conditional_t<Residual, Sm80EVT<StoreD, EVT4>, Sm80EVT<StoreD, EVT2>>;
  using Kernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      int8_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 16,
      int8_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 16,
      Out, cutlass::layout::RowMajor, AlignC, int32_t, float,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, TB, Warp, Inst, EVTD,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, Stages,
      cutlass::arch::OpMultiplyAddSaturate, 1>::GemmKernel;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

  typename EVT2::Arguments e2{
      {{{}, {(float*)xs.data_ptr(), 0.f, {_1{}, _0{}, M}}, {}}, {(float*)ws.data_ptr(), 0.f, {_0{}, _1{}, N}}, {}},
      {(Out*)bias.data_ptr(), Out(0), {_0{}, _1{}, N}}, {}};
  typename StoreD::Arguments st{(Out*)out.data_ptr(), {int64_t(N), _1{}, int64_t(M) * N}};
  typename EVTD::Arguments cb;
  if constexpr (Residual) {
    cb = typename EVTD::Arguments{
        {{e2, {(Out*)rscale->data_ptr(), Out(0), {_0{}, _1{}, N}}, {}},
         {(Out*)resid->data_ptr(), Out(0), {int64_t(N), _1{}, int64_t(M) * N}}, {}},
        st};
  } else {
    cb = typename EVTD::Arguments{e2, st};
  }
  typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K}, 1, cb,
                                (int8_t*)xq.data_ptr(), (int8_t*)wq.data_ptr(), nullptr, nullptr,
                                (int64_t)M * K, (int64_t)N * K, 0, 0, K, K, 0, 0);
  Gemm gemm;
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
  auto workspace = at::empty({(int64_t)Gemm::get_workspace_size(args)}, xq.options().dtype(at::kByte));
  if (gemm.initialize(args, workspace.data_ptr()) != cutlass::Status::kSuccess) return false;
  return gemm(at::cuda::getCurrentCUDAStream()) == cutlass::Status::kSuccess;
}

template <int M_, int N_, int K_> using Sh = cutlass::gemm::GemmShape<M_, N_, K_>;

}  // namespace

// ---- w1 + SwiGLU (levels 3-4, not bit-identical: h is rounded to fp16 before the w2 quantizer) -----
// Adapted from NVIDIA CUTLASS v4.2.1 VisitorAuxStore (include/cutlass/epilogue/threadblock/fusion/visitor_store.hpp):
// the Arguments / Params / Callbacks skeleton and the begin_step / visit / get_callbacks code. Copyright (c) 2017-2025
// NVIDIA CORPORATION & AFFILIATES, BSD-3-Clause (licenses/CUTLASS-BSD-3-Clause.txt). The end_step bodies are ours.
// Epilogue store for the FeedForward w1 GEMM with its weight rows interleaved [g 4j..4j+3 | u 4j..4j+3]:
// each 8-wide fp16 output vector (already fp16(acc*xs*ws + bias), exactly the plain GEMM's values) gives
// h = silu(g) * u for 4 columns, computed as comfy-kitchen's quantizer does ((g / (1 + expf(-g))) * u, fp32),
// then rounded to fp16 and stored at column n/2 of a [M, N/2] output. The w2 quantizer then reads h at half
// the width instead of reading the [M, N] gate|up tensor.
namespace cutlass::epilogue::threadblock {
template <class ThreadMap, class StrideMNL>
struct VisitorSwigluStoreH {
  using Element = cutlass::half_t;
  struct Arguments {
    Element* ptr_out = nullptr;
    int64_t ldo = 0;
    StrideMNL dAux = {};
  };
  using Params = Arguments;
  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(ProblemShape const&, Arguments const& args, void*) { return args; }
  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const&, Arguments const&) { return 0; }
  struct SharedStorage {};
  static int constexpr vec_bits = ThreadMap::kElementsPerAccess * sizeof_bits<Element>::value;
  using VecType = uint_bit_t<cute::min(128, vec_bits)>;
  static int constexpr VecLength = sizeof(VecType) / sizeof(Element);
  static_assert(VecLength == 8, "SwiGLU store expects 8-wide fp16 output vectors");

  CUTLASS_HOST_DEVICE VisitorSwigluStoreH() {}
  CUTLASS_HOST_DEVICE VisitorSwigluStoreH(Params const& params, SharedStorage const&) : params_ptr(&params) {}
  Params const* params_ptr;

  template <class RTensor, class CTensor, class ProblemShape>
  struct Callbacks : EmptyCallbacks {
    CUTLASS_DEVICE Callbacks(RTensor&& r, CTensor&& c, ProblemShape ps, Params const* pp)
        : tC_rAux(cute::forward<RTensor>(r)), tC_cAux(cute::forward<CTensor>(c)), problem_shape(ps), params_ptr(pp) {}
    RTensor tC_rAux;
    CTensor tC_cAux;
    ProblemShape problem_shape;
    Params const* params_ptr;

    CUTLASS_DEVICE void begin_step(int) { clear(tC_rAux); }

    template <class ElementAccumulator, class ElementInput, int FragmentSize>
    CUTLASS_DEVICE auto visit(int, int, int, int frg_idx, Array<ElementAccumulator, FragmentSize> const&,
                              Array<ElementInput, FragmentSize> const& frg_input) {
      NumericArrayConverter<Element, ElementInput, FragmentSize, FloatRoundStyle::round_to_nearest> cvt{};
      Tensor frg = recast<Array<Element, FragmentSize>>(coalesce(tC_rAux));
      frg(frg_idx) = cvt(frg_input);
      return frg_input;
    }

    CUTLASS_DEVICE void end_step(int step_idx) {
      auto src_v = filter(tC_rAux);
      auto coord_v = filter(tC_cAux(_, _, _, step_idx));
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(src_v); ++i) {
        auto coord = coord_v(i);
        if (!elem_less(coord, problem_shape)) continue;
        VecType raw = src_v(i);
        Element const* e = reinterpret_cast<Element const*>(&raw);
        __half2 h2[2];
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < 2; ++j) {
          const float g0 = float(e[2 * j]), g1 = float(e[2 * j + 1]);
          const float u0 = float(e[4 + 2 * j]), u1 = float(e[4 + 2 * j + 1]);
          h2[j] = __floats2half2_rn((g0 / (1.0f + expf(-g0))) * u0, (g1 / (1.0f + expf(-g1))) * u1);
        }
        Element* dst = params_ptr->ptr_out + int64_t(get<0>(coord)) * params_ptr->ldo + get<1>(coord) / 2;
        *reinterpret_cast<uint2*>(dst) = *reinterpret_cast<uint2*>(h2);
      }
    }
  };

  template <class ProblemShape>
  CUTLASS_DEVICE auto get_callbacks(gemm::GemmCoord threadblock_tile_offset, int thread_idx, ProblemShape problem_shape) {
    Tensor mAux = make_tensor(make_gmem_ptr(params_ptr->ptr_out), problem_shape, params_ptr->dAux);
    Tensor tC_gAux = recast<VecType>(group_modes<3, 6>(ThreadMap::partition(mAux, thread_idx, threadblock_tile_offset)));
    Tensor tC_rAux = make_tensor_like(take<0, 3>(tC_gAux));
    Tensor cAux = make_identity_tensor(mAux.shape());
    Tensor tC_cAux = outer_partition(
        group_modes<3, 6>(ThreadMap::partition(cAux, thread_idx, threadblock_tile_offset)), Shape<Int<VecLength>>{}, (_0{}));
    return Callbacks<decltype(tC_rAux), decltype(tC_cAux), ProblemShape>(
        cute::move(tC_rAux), cute::move(tC_cAux), problem_shape, params_ptr);
  }
};
}  // namespace cutlass::epilogue::threadblock

namespace {

template <class TB, class Warp, int Stages>
bool run_gemm_swiglu(const at::Tensor& xq, const at::Tensor& wq, const at::Tensor& xs, const at::Tensor& ws,
                     const at::Tensor& bias, at::Tensor& out) {
  using namespace cutlass::epilogue::threadblock;
  using namespace cute;
  const int M = xq.size(0), K = xq.size(1), N = wq.size(0);
  constexpr int AlignC = 8;
  using Inst = cutlass::gemm::GemmShape<16, 8, 32>;
  using ThreadMap = OutputTileThreadLayout<TB, Warp, Out, AlignC, 1>;
  using Accum = VisitorAccFetch;
  using XScale = VisitorColBroadcast<ThreadMap, float, Stride<_1, _0, int32_t>>;
  using WScale = VisitorRowBroadcast<ThreadMap, float, Stride<_0, _1, int32_t>>;
  using Bias = VisitorRowBroadcast<ThreadMap, Out, Stride<_0, _1, int32_t>>;
  using Mul0 = VisitorCompute<cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
  using EVT0 = Sm80EVT<Mul0, Accum, XScale>;
  using EVT1 = Sm80EVT<Mul0, EVT0, WScale>;
  using Add2 = VisitorCompute<cutlass::plus, Out, float, cutlass::FloatRoundStyle::round_to_nearest>;
  using EVT2 = Sm80EVT<Add2, EVT1, Bias>;
  using StrideO = Stride<int64_t, _1, int64_t>;
  using Store = VisitorSwigluStoreH<ThreadMap, StrideO>;
  using EVTD = Sm80EVT<Store, EVT2>;
  using Kernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      int8_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 16,
      int8_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 16,
      Out, cutlass::layout::RowMajor, AlignC, int32_t, float,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, TB, Warp, Inst, EVTD,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, Stages,
      cutlass::arch::OpMultiplyAddSaturate, 1>::GemmKernel;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

  typename EVT2::Arguments e2{
      {{{}, {(float*)xs.data_ptr(), 0.f, {_1{}, _0{}, M}}, {}}, {(float*)ws.data_ptr(), 0.f, {_0{}, _1{}, N}}, {}},
      {(Out*)bias.data_ptr(), Out(0), {_0{}, _1{}, N}}, {}};
  typename Store::Arguments st{(Out*)out.data_ptr(), out.stride(0), {int64_t(N), _1{}, int64_t(M) * N}};
  typename EVTD::Arguments cb{e2, st};
  typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K}, 1, cb,
                                (int8_t*)xq.data_ptr(), (int8_t*)wq.data_ptr(), nullptr, nullptr,
                                (int64_t)M * K, (int64_t)N * K, 0, 0, K, K, 0, 0);
  Gemm gemm;
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
  auto workspace = at::empty({(int64_t)Gemm::get_workspace_size(args)}, xq.options().dtype(at::kByte));
  if (gemm.initialize(args, workspace.data_ptr()) != cutlass::Status::kSuccess) return false;
  return gemm(at::cuda::getCurrentCUDAStream()) == cutlass::Status::kSuccess;
}

template <bool Residual>
bool dispatch(int config, const at::Tensor& xq, const at::Tensor& wq, const at::Tensor& xs, const at::Tensor& ws,
              const at::Tensor& bias, const at::Tensor* rs, const at::Tensor* rd, at::Tensor& out) {
  switch (config) {
    case 1: return run_gemm<Sh<128, 128, 64>, Sh<64, 64, 64>, 5, Residual>(xq, wq, xs, ws, bias, rs, rd, out);
    case 4: return run_gemm<Sh<128, 128, 128>, Sh<64, 64, 128>, 3, Residual>(xq, wq, xs, ws, bias, rs, rd, out);
    case 5: return run_gemm<Sh<256, 128, 64>, Sh<64, 64, 64>, 4, Residual>(xq, wq, xs, ws, bias, rs, rd, out);
    case 6: return run_gemm<Sh<128, 256, 64>, Sh<64, 64, 64>, 4, Residual>(xq, wq, xs, ws, bias, rs, rd, out);
    case 7: return run_gemm<Sh<128, 256, 64>, Sh<64, 64, 64>, 3, Residual>(xq, wq, xs, ws, bias, rs, rd, out);
    default: TORCH_CHECK(false, "vae_gemm: unknown config ", config);
  }
}

}  // namespace

// o [B, H, S, D] fp16 (contiguous) -> q [B*S, H*D] int8, scales [B*S] fp32
std::vector<at::Tensor> vae_out_quant(at::Tensor o) {
  TORCH_CHECK(o.dim() == 4 && o.is_contiguous() && o.scalar_type() == at::kHalf, "o must be contiguous fp16 [B,H,S,D]");
  const int B = o.size(0), H = o.size(1), S = o.size(2), D = o.size(3), K = H * D;
  TORCH_CHECK(K % kConvRotGroup == 0 && kConvRotGroup % D == 0, "row width must be a multiple of 256");
  c10::cuda::CUDAGuard g(o.device());
  auto q = at::empty({(int64_t)B * S, K}, o.options().dtype(at::kChar));
  auto s = at::empty({(int64_t)B * S}, o.options().dtype(at::kFloat));
  constexpr int kThreads = 128;  // comfy-kitchen's choice for 256 < K <= 3072
  const size_t smem = (size_t(K) + (kThreads / 64) * 2 * kConvRotGroup) * sizeof(float);
  static bool once = [&] {
    cudaFuncSetAttribute(vae_out_quant_kernel<kThreads>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024);
    return true;
  }();
  (void)once;
  vae_out_quant_kernel<kThreads><<<B * S, kThreads, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(o.data_ptr()), q.data_ptr<int8_t>(), s.data_ptr<float>(), H, S, D);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {q, s};
}

// out = fp16(acc*xs*ws + bias)  [ + residual addcmul: out = resid + rscale * that ]
at::Tensor vae_gemm(at::Tensor xq, at::Tensor wq, at::Tensor xs, at::Tensor ws, at::Tensor bias,
                    c10::optional<at::Tensor> rscale, c10::optional<at::Tensor> resid, int64_t config) {
  TORCH_CHECK(xq.is_contiguous() && wq.is_contiguous() && xq.scalar_type() == at::kChar, "int8 contiguous inputs");
  TORCH_CHECK(bias.scalar_type() == at::kHalf && xs.scalar_type() == at::kFloat && ws.scalar_type() == at::kFloat);
  c10::cuda::CUDAGuard g(xq.device());
  auto out = at::empty({xq.size(0), wq.size(0)}, xq.options().dtype(at::kHalf));
  bool ok;
  if (resid.has_value()) {
    TORCH_CHECK(resid->is_contiguous() && resid->scalar_type() == at::kHalf && rscale->scalar_type() == at::kHalf);
    ok = dispatch<true>((int)config, xq, wq, xs, ws, bias, &*rscale, &*resid, out);
  } else {
    ok = dispatch<false>((int)config, xq, wq, xs, ws, bias, nullptr, nullptr, out);
  }
  TORCH_CHECK(ok, "vae_gemm: launch failed");
  return out;
}

// h[M, N/2] = fp16(silu(g) * u) with g|u = fp16(acc*xs*ws + bias) from the interleaved w1 (see VisitorSwigluStoreH)
at::Tensor vae_gemm_swiglu(at::Tensor xq, at::Tensor wq, at::Tensor xs, at::Tensor ws, at::Tensor bias, int64_t config) {
  TORCH_CHECK(xq.is_contiguous() && wq.is_contiguous() && xq.scalar_type() == at::kChar, "int8 contiguous inputs");
  TORCH_CHECK(bias.scalar_type() == at::kHalf && xs.scalar_type() == at::kFloat && ws.scalar_type() == at::kFloat);
  TORCH_CHECK(wq.size(0) % 8 == 0, "N must be a multiple of 8");
  c10::cuda::CUDAGuard g(xq.device());
  auto out = at::empty({xq.size(0), wq.size(0) / 2}, xq.options().dtype(at::kHalf));
  bool ok;
  switch (config) {
    case 1: ok = run_gemm_swiglu<Sh<128, 128, 64>, Sh<64, 64, 64>, 5>(xq, wq, xs, ws, bias, out); break;
    case 5: ok = run_gemm_swiglu<Sh<256, 128, 64>, Sh<64, 64, 64>, 4>(xq, wq, xs, ws, bias, out); break;
    case 6: ok = run_gemm_swiglu<Sh<128, 256, 64>, Sh<64, 64, 64>, 4>(xq, wq, xs, ws, bias, out); break;
    default: TORCH_CHECK(false, "vae_gemm_swiglu: unknown config ", config);
  }
  TORCH_CHECK(ok, "vae_gemm_swiglu: launch failed");
  return out;
}

// ---- round 5: q/k RMSNorm + split-half rope fused into comfy-kitchen's SageAttention Q/K quantizer -------------
// The decoder's Attention ran ck.rms_rope_split_half_ (in place on q, k inside the qkv GEMM output) and then
// ck.int8_attention, whose first kernels (detect_k_anchor, quant_qk_fused) read q and k back. Here one kernel reads
// the raw q, k of the qkv buffer, applies ck's rms+rope per row into shared memory (the fp16 values ck would have
// written back), and quantizes them with ck's per-thread-group Q/K quantizer. The anchor detector computes rms+rope
// for its 9 sampled keys itself and keeps the chosen anchor key. V quantization and the attention kernel stay
// comfy-kitchen's own (_C.quant_v_int8, _C.sage_sdpa_prequantized), so the attention output is bit-identical.
// The per-element math below is copied from comfy-kitchen 0.2.37 (Apache-2.0, Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES): ops/rms_rope.cu + ops/rope_device.cuh (rope_kernel<half, half, half, HasRms,
// SplitHalf, HasK, InPlace, ContigHead>) and sage_attention/quant_qk_int8.cu (process_q / process_k /
// detect_k_anchor, C = 64: quant_qk_fused<half, 4, 8, 128, 32, 64, 64, 1, ROT, false>).
namespace {

constexpr int kHD = 64;   // head dim (q, k, v each 64 wide inside a 192-wide qkv head chunk)

struct RopeArgs {
  const half* qkv;          // [B, S, H, 3 * 64] fp16, q at +0, k at +64
  int64_t sb, sn, sh;       // element strides of batch, token, head
  const half* freqs;        // [Bf, Sf, 1, pairs, 2, 2] fp16
  int64_t f_s0, f_s1, f_s3, f_s4, f_s5;
  int freqs_batch, freqs_seq;
  const float* scale;       // [64] q/k RMSNorm scale (q and k share it in the decoder)
  int64_t scale_stride;
  float eps;
  int pairs;                // rot_dim / 2
  int B, H, L;
};

// ck rope::rotate (ComputeType float)
__device__ __forceinline__ void ck_rotate(float x0, float x1, float f00, float f01, float f10, float f11,
                                          float& y0, float& y1) {
  y0 = f00 * x0 + f01 * x1;
  y1 = f10 * x0 + f11 * x1;
}

// One warp: rms + split-half rope of one 64-wide row (ck rope_kernel, ContigHead, kPairsPerLane = 2),
// fp16 results to dst[0..63] (shared memory).
__device__ __forceinline__ void rope_row_warp(const RopeArgs& a, const half* row, int b, int n, half* dst, int lane) {
  float sum = 0.0f;
  for (int e = lane; e < kHD; e += 32) {
    const float v = __half2float(row[e]);
    sum = fmaf(v, v, sum);
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, offset);
  sum = __shfl_sync(0xffffffffu, sum, 0);
  const float rrms = rsqrtf(sum / static_cast<float>(kHD) + a.eps);

  const int64_t fb = a.freqs_batch == 1 ? 0 : b;
  const int64_t fs = a.freqs_seq == 1 ? 0 : n;
  const int64_t freq_row = fb * a.f_s0 + fs * a.f_s1;
  const int pairs = a.pairs;
  for (int pair_base = lane * 2; pair_base < pairs; pair_base += 64) {
    const half2 lo = *reinterpret_cast<const half2*>(row + pair_base);
    const half2 hi = *reinterpret_cast<const half2*>(row + pairs + pair_base);
    const half x0r[2] = {__low2half(lo), __high2half(lo)};
    const half x1r[2] = {__low2half(hi), __high2half(hi)};
    half y0r[2], y1r[2];
#pragma unroll
    for (int p = 0; p < 2; ++p) {
      const int pair = pair_base + p;
      const int first = pair, second = pair + pairs;
      float x0 = __half2float(x0r[p]);
      float x1 = __half2float(x1r[p]);
      x0 = __half2float(__float2half_rn(x0 * rrms * a.scale[first * a.scale_stride]));
      x1 = __half2float(__float2half_rn(x1 * rrms * a.scale[second * a.scale_stride]));
      const int64_t fbase = freq_row + static_cast<int64_t>(pair) * a.f_s3;
      const float f00 = __half2float(a.freqs[fbase]);
      const float f01 = __half2float(a.freqs[fbase + a.f_s5]);
      const float f10 = __half2float(a.freqs[fbase + a.f_s4]);
      const float f11 = __half2float(a.freqs[fbase + a.f_s4 + a.f_s5]);
      float y0, y1;
      ck_rotate(x0, x1, f00, f01, f10, f11, y0, y1);
      y0r[p] = __float2half_rn(y0);
      y1r[p] = __float2half_rn(y1);
    }
    *reinterpret_cast<half2*>(dst + pair_base) = __halves2half2(y0r[0], y0r[1]);
    *reinterpret_cast<half2*>(dst + pairs + pair_base) = __halves2half2(y1r[0], y1r[1]);
  }
  for (int d = 2 * pairs + lane; d < kHD; d += 32)
    dst[d] = __float2half_rn(__half2float(row[d]) * rrms * a.scale[d * a.scale_stride]);
}

__device__ __forceinline__ void ld4_smem(const half* p, float* out) {
  const float2 raw = *reinterpret_cast<const float2*>(p);
  const half* h = reinterpret_cast<const half*>(&raw);
  out[0] = __half2float(h[0]); out[1] = __half2float(h[1]); out[2] = __half2float(h[2]); out[3] = __half2float(h[3]);
}

// ---- copied from comfy-kitchen sage_attention/quant_qk_int8.cu --------------------------------------------------
__forceinline__ __device__ void convrot4(float* values) {
  const float x0 = values[0], x1 = values[1], x2 = values[2], x3 = values[3];
  const float a0 = x0 + x1, a1 = x0 - x1, a2 = x2 + x3, a3 = x2 - x3;
  values[0] = (a0 + a2) * 0.5f;
  values[1] = (a1 + a3) * 0.5f;
  values[2] = (a0 - a2) * 0.5f;
  values[3] = (a1 - a3) * 0.5f;
}
__forceinline__ __device__ void convrot64(float* values) {
  convrot4(values);
  const int half_lane = threadIdx.x & 15;
  const unsigned mask = (threadIdx.x & 16) ? 0xffff0000u : 0x0000ffffu;
#pragma unroll
  for (int bit = 1; bit < 16; bit <<= 1) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const float other = __shfl_xor_sync(mask, values[c], bit, 16);
      values[c] = (half_lane & bit) ? other - values[c] : values[c] + other;
    }
  }
#pragma unroll
  for (int c = 0; c < 4; ++c) values[c] *= 0.25f;
}
__forceinline__ __device__ float ck_warp_reduce_fmax(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
  return v;
}
__forceinline__ __device__ int8_t ck_quant_int8_rcp(float v, float inv_scale) {
  int32_t q;
  asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(q) : "f"(v * inv_scale));
  return static_cast<int8_t>(q);
}
__forceinline__ __device__ void ck_store4_i8(int8_t* ptr, int8_t a, int8_t b, int8_t c, int8_t d) {
  *reinterpret_cast<int32_t*>(ptr) = (uint32_t)(uint8_t)a | ((uint32_t)(uint8_t)b << 8) |
                                     ((uint32_t)(uint8_t)c << 16) | ((uint32_t)(uint8_t)d << 24);
}
template <int ROTATION>
__forceinline__ __device__ void ck_rotate4(float* v) {
  if constexpr (ROTATION == 64) convrot64(v);
  else convrot4(v);
}

// process_q<half, NR=4, BLKQ=128, WARPQ=32, CHANNEL_TILES=1, ROTATION, ALIGNED4=false>, C = 64, input from the
// block's shared-memory rows (row r of the block = token row0 + r, 64 halfs each).
template <int ROTATION>
__device__ __forceinline__ void process_q_smem(const half* s_rows, int row0, int8_t* __restrict__ out,
                                               float* __restrict__ sc_buf, const int oblk, const int L) {
  constexpr int NR = 4, BLKQ = 128, WARPQ = 32, NSUB = BLKQ / WARPQ, C = kHD;
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
#pragma unroll
  for (int g = 0; g < 2; ++g) {
    const int otld = wid * 2 + g;
    const int base = (oblk / NSUB) * BLKQ + (oblk % NSUB) * WARPQ + otld;
    float v[NR * 4];
    float mx = 0.f;
#pragma unroll
    for (int i = 0; i < NR * 4; ++i) v[i] = 0.f;
    const int ch = lane << 2;
    if (ch + 3 < C) {
#pragma unroll
      for (int j = 0; j < NR; ++j) {
        const int n = base + j * 8;
        const int vi = j * 4;
        if (n < L) {
          ld4_smem(s_rows + (n - row0) * kHD + ch, &v[vi]);
          mx = fmaxf(mx, fmaxf(fmaxf(fabsf(v[vi]), fabsf(v[vi + 1])), fmaxf(fabsf(v[vi + 2]), fabsf(v[vi + 3]))));
        }
      }
    }
    if constexpr (ROTATION != 0) {
#pragma unroll
      for (int j = 0; j < NR; ++j) ck_rotate4<ROTATION>(&v[j * 4]);
      mx = 0.f;
#pragma unroll
      for (int j = 0; j < NR * 4; ++j) mx = fmaxf(mx, fabsf(v[j]));
    }
    mx = ck_warp_reduce_fmax(mx);
    const float sc = mx / 127.f + 1e-7f;
    const float inv_sc = 1.f / sc;
    if (lane == 0) sc_buf[oblk * 8 + otld] = sc;
    if (ch + 3 < C) {
#pragma unroll
      for (int j = 0; j < NR; ++j) {
        const int n = base + j * 8;
        const int vi = j * 4;
        if (n < L)
          ck_store4_i8(&out[(int64_t)n * C + ch], ck_quant_int8_rcp(v[vi], inv_sc), ck_quant_int8_rcp(v[vi + 1], inv_sc),
                       ck_quant_int8_rcp(v[vi + 2], inv_sc), ck_quant_int8_rcp(v[vi + 3], inv_sc));
      }
    }
  }
}

// process_k<half, NL=8, WARPK=64, CHANNEL_TILES=1, ROTATION, ALIGNED4=false>, C = 64, anchor key given as floats.
template <int ROTATION>
__device__ __forceinline__ void process_k_smem(const half* s_rows, int row0, int8_t* __restrict__ out,
                                               float* __restrict__ sc_buf, const int oblk, const int L,
                                               const float* anchor /* nullptr = no centering */) {
  constexpr int NL = 8, WARPK = 64, C = kHD;
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  const int otld = wid;
  const int ch = lane << 2;
  float bias[4] = {0.f, 0.f, 0.f, 0.f};
  if (anchor != nullptr && ch + 3 < C) {
    bias[0] = anchor[ch]; bias[1] = anchor[ch + 1]; bias[2] = anchor[ch + 2]; bias[3] = anchor[ch + 3];
  }
  float v[2 * NL * 4];
  float mx = 0.f;
#pragma unroll
  for (int i = 0; i < 2 * NL * 4; ++i) v[i] = 0.f;
  if (ch + 3 < C) {
#pragma unroll
    for (int j = 0; j < NL; ++j) {
#pragma unroll
      for (int p = 0; p < 2; ++p) {
        const int n = oblk * WARPK + j * 8 + otld * 2 + p;
        const int vi = (j * 2 + p) * 4;
        if (n < L) {
          ld4_smem(s_rows + (n - row0) * kHD + ch, &v[vi]);
#pragma unroll
          for (int c = 0; c < 4; ++c) v[vi + c] -= bias[c];
          mx = fmaxf(mx, fmaxf(fmaxf(fabsf(v[vi]), fabsf(v[vi + 1])), fmaxf(fabsf(v[vi + 2]), fabsf(v[vi + 3]))));
        }
      }
    }
  }
  if constexpr (ROTATION != 0) {
#pragma unroll
    for (int j = 0; j < 2 * NL; ++j) ck_rotate4<ROTATION>(&v[j * 4]);
    mx = 0.f;
#pragma unroll
    for (int j = 0; j < 2 * NL * 4; ++j) mx = fmaxf(mx, fabsf(v[j]));
  }
  mx = ck_warp_reduce_fmax(mx);
  const float sc = mx / 127.f + 1e-7f;
  const float inv_sc = 1.f / sc;
  if (lane == 0) sc_buf[oblk * 4 + otld] = sc;
  if (ch + 3 < C) {
#pragma unroll
    for (int j = 0; j < NL; ++j) {
#pragma unroll
      for (int p = 0; p < 2; ++p) {
        const int n = oblk * WARPK + j * 8 + otld * 2 + p;
        const int vi = (j * 2 + p) * 4;
        if (n < L)
          ck_store4_i8(&out[(int64_t)n * C + ch], ck_quant_int8_rcp(v[vi], inv_sc), ck_quant_int8_rcp(v[vi + 1], inv_sc),
                       ck_quant_int8_rcp(v[vi + 2], inv_sc), ck_quant_int8_rcp(v[vi + 3], inv_sc));
      }
    }
  }
}

constexpr int CENTER_DETECT_THREADS = 128;
constexpr int CENTER_SAMPLES = 9;

// ck detect_k_anchor (C = 64) on rms+rope'd keys; also stores the chosen anchor key (as floats) for the K quantizer.
__global__ __launch_bounds__(CENTER_DETECT_THREADS) void vae_k_anchor_kernel(RopeArgs a, int* __restrict__ anchor_indices,
                                                                             float* __restrict__ anchor_vals) {
  constexpr int C = kHD;
  const int h = blockIdx.x;
  const int b = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int Lk = a.L;

  __shared__ float samples[CENTER_SAMPLES * C];
  __shared__ __align__(16) half s_rows[CENTER_SAMPLES * C];
  __shared__ float warp_original_energy[4];
  __shared__ float warp_original_max[4];
  __shared__ float warp_candidate_distance[CENTER_SAMPLES][4];
  __shared__ float warp_best_energy[4];
  __shared__ float warp_best_max[4];
  __shared__ int selected_candidate;

  for (int sample = warp; sample < CENTER_SAMPLES; sample += 4) {
    const int row = sample * (Lk - 1) / (CENTER_SAMPLES - 1);
    const half* src = a.qkv + (int64_t)b * a.sb + (int64_t)row * a.sn + (int64_t)h * a.sh + kHD;
    rope_row_warp(a, src, b, row, s_rows + sample * C, lane);
  }
  __syncthreads();
  for (int index = tid; index < CENTER_SAMPLES * C; index += CENTER_DETECT_THREADS)
    samples[index] = __half2float(s_rows[index]);
  __syncthreads();

  float original_energy = 0.f;
  float original_max = 0.f;
  float candidate_distance[CENTER_SAMPLES];
#pragma unroll
  for (int candidate = 0; candidate < CENTER_SAMPLES; ++candidate) candidate_distance[candidate] = 0.f;

  for (int channel = tid; channel < C; channel += CENTER_DETECT_THREADS) {
    float channel_sum = 0.f;
#pragma unroll
    for (int sample = 0; sample < CENTER_SAMPLES; ++sample) {
      const float value = samples[sample * C + channel];
      original_energy = fmaf(value, value, original_energy);
      original_max = fmaxf(original_max, fabsf(value));
      channel_sum += value;
    }
#pragma unroll
    for (int candidate = 0; candidate < CENTER_SAMPLES; ++candidate) {
      const float distance = CENTER_SAMPLES * samples[candidate * C + channel] - channel_sum;
      candidate_distance[candidate] = fmaf(distance, distance, candidate_distance[candidate]);
    }
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    original_energy += __shfl_down_sync(0xffffffffu, original_energy, offset);
    original_max = fmaxf(original_max, __shfl_down_sync(0xffffffffu, original_max, offset));
#pragma unroll
    for (int candidate = 0; candidate < CENTER_SAMPLES; ++candidate)
      candidate_distance[candidate] += __shfl_down_sync(0xffffffffu, candidate_distance[candidate], offset);
  }
  if (lane == 0) {
    warp_original_energy[warp] = original_energy;
    warp_original_max[warp] = original_max;
#pragma unroll
    for (int candidate = 0; candidate < CENTER_SAMPLES; ++candidate)
      warp_candidate_distance[candidate][warp] = candidate_distance[candidate];
  }
  __syncthreads();
  if (tid == 0) {
    int best_candidate = 0;
    float best_distance = 3.402823466e+38F;
#pragma unroll
    for (int candidate = 0; candidate < CENTER_SAMPLES; ++candidate) {
      float distance = 0.f;
#pragma unroll
      for (int w = 0; w < 4; ++w) distance += warp_candidate_distance[candidate][w];
      if (distance < best_distance) {
        best_candidate = candidate;
        best_distance = distance;
      }
    }
    selected_candidate = best_candidate;
  }
  __syncthreads();

  float best_energy = 0.f;
  float best_max = 0.f;
  for (int channel = tid; channel < C; channel += CENTER_DETECT_THREADS) {
    const float anchor = samples[selected_candidate * C + channel];
#pragma unroll
    for (int sample = 0; sample < CENTER_SAMPLES; ++sample) {
      const float residual = samples[sample * C + channel] - anchor;
      best_energy = fmaf(residual, residual, best_energy);
      best_max = fmaxf(best_max, fabsf(residual));
    }
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    best_energy += __shfl_down_sync(0xffffffffu, best_energy, offset);
    best_max = fmaxf(best_max, __shfl_down_sync(0xffffffffu, best_max, offset));
  }
  if (lane == 0) {
    warp_best_energy[warp] = best_energy;
    warp_best_max[warp] = best_max;
  }
  __syncthreads();
  if (tid < C) anchor_vals[((int64_t)b * a.H + h) * C + tid] = samples[selected_candidate * C + tid];
  if (tid == 0) {
    float total_original_energy = 0.f, total_original_max = 0.f, total_best_energy = 0.f, total_best_max = 0.f;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      total_original_energy += warp_original_energy[w];
      total_original_max = fmaxf(total_original_max, warp_original_max[w]);
      total_best_energy += warp_best_energy[w];
      total_best_max = fmaxf(total_best_max, warp_best_max[w]);
    }
    const bool improves_range = total_best_energy < total_original_energy && total_best_max <= total_original_max * 1.125f;
    anchor_indices[b * a.H + h] = improves_range ? selected_candidate * (Lk - 1) / (CENTER_SAMPLES - 1) : -1;
  }
}
// ---- end of copied comfy-kitchen code ---------------------------------------------------------------------------

// In-place rms + split-half rope of one 64-wide fp16 row in shared memory, by one warp: the same per-lane math as
// rope_row_warp (ck rope_kernel); f = the token's rotation table, [f00 f01 f10 f11] per pair; sc = scale (fp32).
__device__ __forceinline__ void rope_row_smem(half* row, const half* f, const float* sc, float eps, int pairs, int lane) {
  float sum = 0.0f;
  for (int e = lane; e < kHD; e += 32) {
    const float v = __half2float(row[e]);
    sum = fmaf(v, v, sum);
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, offset);
  sum = __shfl_sync(0xffffffffu, sum, 0);
  __syncwarp();
  const float rrms = rsqrtf(sum / static_cast<float>(kHD) + eps);
  for (int pair_base = lane * 2; pair_base < pairs; pair_base += 64) {
    const half2 lo = *reinterpret_cast<const half2*>(row + pair_base);
    const half2 hi = *reinterpret_cast<const half2*>(row + pairs + pair_base);
    const half x0r[2] = {__low2half(lo), __high2half(lo)};
    const half x1r[2] = {__low2half(hi), __high2half(hi)};
    half y0r[2], y1r[2];
#pragma unroll
    for (int p = 0; p < 2; ++p) {
      const int pair = pair_base + p;
      float x0 = __half2float(x0r[p]);
      float x1 = __half2float(x1r[p]);
      x0 = __half2float(__float2half_rn(x0 * rrms * sc[pair]));
      x1 = __half2float(__float2half_rn(x1 * rrms * sc[pair + pairs]));
      const float2 fa = __half22float2(*reinterpret_cast<const half2*>(f + 4 * pair));
      const float2 fb = __half22float2(*reinterpret_cast<const half2*>(f + 4 * pair + 2));
      float y0, y1;
      ck_rotate(x0, x1, fa.x, fa.y, fb.x, fb.y, y0, y1);
      y0r[p] = __float2half_rn(y0);
      y1r[p] = __float2half_rn(y1);
    }
    *reinterpret_cast<half2*>(row + pair_base) = __halves2half2(y0r[0], y0r[1]);
    *reinterpret_cast<half2*>(row + pairs + pair_base) = __halves2half2(y1r[0], y1r[1]);
  }
  for (int d = 2 * pairs + lane; d < kHD; d += 32) row[d] = __float2half_rn(__half2float(row[d]) * rrms * sc[d]);
}

// grid (q_oblk + k_oblk, H, B), 128 threads: blockIdx.x < q_oblk -> 32 Q rows, else 64 K rows.
// 1) the block's raw q (or k) rows and their rope tables -> shared memory (16-byte loads), 2) one warp per row:
// rms + rope in place, 3) ck's Q/K quantizer reading shared memory. Needs 16-byte aligned rows and a contiguous
// [pairs, 2, 2] table per token (checked by the launcher).
template <int ROTATION>
__global__ __launch_bounds__(128) void vae_qk_rope_quant_kernel(RopeArgs a, int q_oblk, int8_t* __restrict__ q_out,
                                                                float* __restrict__ q_sb, int8_t* __restrict__ k_out,
                                                                float* __restrict__ k_sb,
                                                                const int* __restrict__ anchor_indices,
                                                                const float* __restrict__ anchor_vals) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  half* s_rows = reinterpret_cast<half*>(smem_raw);                 // [rows][64]
  half* s_f = s_rows + 64 * kHD;                                     // [rows][4 * pairs]
  __shared__ float s_sc[kHD];
  const int h = blockIdx.y, b = blockIdx.z;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const bool is_q = blockIdx.x < (unsigned)q_oblk;
  const int oblk = is_q ? blockIdx.x : blockIdx.x - q_oblk;
  const int row0 = is_q ? oblk * 32 : oblk * 64;   // (oblk / 4) * 128 + (oblk % 4) * 32 == oblk * 32
  const int rows = min(is_q ? 32 : 64, a.L - row0);
  const int fw = 4 * a.pairs;                      // halfs of rope table per token
  const half* src = a.qkv + (int64_t)b * a.sb + (int64_t)h * a.sh + (is_q ? 0 : kHD);
  const half* ftab = a.freqs + (a.freqs_batch == 1 ? 0 : (int64_t)b * a.f_s0);
  if (tid < kHD) s_sc[tid] = a.scale[tid * a.scale_stride];
  for (int i = tid; i < rows * 8; i += 128) {
    const int r = i >> 3, c = i & 7;
    *reinterpret_cast<uint4*>(s_rows + r * kHD + c * 8) =
        __ldg(reinterpret_cast<const uint4*>(src + (int64_t)(row0 + r) * a.sn + c * 8));
  }
  const int fchunks = fw / 8;
  for (int i = tid; i < rows * fchunks; i += 128) {
    const int r = i / fchunks, c = i - r * fchunks;
    const int64_t fs = a.freqs_seq == 1 ? 0 : (int64_t)(row0 + r) * a.f_s1;
    *reinterpret_cast<uint4*>(s_f + r * fw + c * 8) = __ldg(reinterpret_cast<const uint4*>(ftab + fs + c * 8));
  }
  __syncthreads();
  for (int r = warp; r < rows; r += 4) rope_row_smem(s_rows + r * kHD, s_f + r * fw, s_sc, a.eps, a.pairs, lane);
  __syncthreads();
  const int64_t bh = (int64_t)b * a.H + h;
  if (is_q) {
    const int q_sc_per_h = q_oblk * 8;
    process_q_smem<ROTATION>(s_rows, row0, q_out + bh * a.L * kHD, q_sb + bh * q_sc_per_h, oblk, a.L);
  } else {
    const int k_oblk = (a.L + 63) / 64;
    const int anchor_index = anchor_indices[bh];
    process_k_smem<ROTATION>(s_rows, row0, k_out + bh * a.L * kHD, k_sb + bh * k_oblk * 4, oblk, a.L,
                             anchor_index >= 0 ? anchor_vals + bh * kHD : nullptr);
  }
}

}  // namespace

// qkv [B, S, H, 192] fp16 view (last dim contiguous), freqs [Bf, Sf, 1, pairs, 2, 2] fp16, scale [64] fp32
// -> q_int8 [B, H, S, 64], q_scale [B, H, ceil(S/128)*32], k_int8 [B, H, S, 64], k_scale [B, H, ceil(S/64)*4]
// (comfy-kitchen's SageAttention layouts for head_dim 64, cta_k 64), with ck's q/k rms+rope applied first.
std::vector<at::Tensor> vae_qk_rope_quant(at::Tensor qkv, at::Tensor freqs, at::Tensor scale, double eps) {
  TORCH_CHECK(qkv.dim() == 4 && qkv.size(3) == 3 * kHD && qkv.stride(3) == 1 && qkv.scalar_type() == at::kHalf,
              "qkv must be [B, S, H, 192] fp16 with a contiguous last dim");
  TORCH_CHECK(freqs.dim() == 6 && freqs.scalar_type() == at::kHalf && freqs.size(2) == 1 && freqs.size(4) == 2 &&
              freqs.size(5) == 2, "freqs must be [B|1, S|1, 1, pairs, 2, 2] fp16");
  TORCH_CHECK(scale.dim() == 1 && scale.size(0) == kHD && scale.scalar_type() == at::kFloat, "scale must be fp32 [64]");
  const int B = qkv.size(0), S = qkv.size(1), H = qkv.size(2);
  const int pairs = freqs.size(3);
  TORCH_CHECK(pairs % 2 == 0 && 2 * pairs <= kHD && pairs <= 64, "rot_dim must be a multiple of 4 and <= 64");
  TORCH_CHECK((freqs.size(0) == 1 || freqs.size(0) == B) && (freqs.size(1) == 1 || freqs.size(1) == S));
  TORCH_CHECK(qkv.stride(0) % 8 == 0 && qkv.stride(1) % 8 == 0 && qkv.stride(2) % 8 == 0 &&
              reinterpret_cast<uintptr_t>(qkv.data_ptr()) % 16 == 0, "qkv rows must be 16-byte aligned");
  TORCH_CHECK(freqs.stride(5) == 1 && freqs.stride(4) == 2 && freqs.stride(3) == 4 &&
              (freqs.size(0) == 1 || freqs.stride(0) % 8 == 0) && (freqs.size(1) == 1 || freqs.stride(1) % 8 == 0) &&
              reinterpret_cast<uintptr_t>(freqs.data_ptr()) % 16 == 0, "freqs must be a contiguous [pairs, 2, 2] table per token");
  c10::cuda::CUDAGuard g(qkv.device());
  RopeArgs a;
  a.qkv = reinterpret_cast<const half*>(qkv.data_ptr());
  a.sb = qkv.stride(0); a.sn = qkv.stride(1); a.sh = qkv.stride(2);
  a.freqs = reinterpret_cast<const half*>(freqs.data_ptr());
  a.f_s0 = freqs.stride(0); a.f_s1 = freqs.stride(1); a.f_s3 = freqs.stride(3); a.f_s4 = freqs.stride(4);
  a.f_s5 = freqs.stride(5);
  a.freqs_batch = freqs.size(0); a.freqs_seq = freqs.size(1);
  a.scale = scale.data_ptr<float>(); a.scale_stride = scale.stride(0);
  a.eps = static_cast<float>(eps);
  a.pairs = pairs;
  a.B = B; a.H = H; a.L = S;
  auto o8 = qkv.options().dtype(at::kChar), of = qkv.options().dtype(at::kFloat);
  const int q_oblk = (S + 127) / 128 * 4, k_oblk = (S + 63) / 64;
  auto q8 = at::empty({B, H, S, kHD}, o8), k8 = at::empty({B, H, S, kHD}, o8);
  auto qs = at::empty({B, H, (S + 127) / 128 * 32}, of), ks = at::empty({B, H, k_oblk * 4}, of);
  auto anchor = at::empty({B, H}, qkv.options().dtype(at::kInt));
  auto anchor_vals = at::empty({B, H, kHD}, of);
  auto stream = at::cuda::getCurrentCUDAStream();
  vae_k_anchor_kernel<<<dim3(H, B), CENTER_DETECT_THREADS, 0, stream>>>(a, anchor.data_ptr<int>(),
                                                                         anchor_vals.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const dim3 grid(q_oblk + k_oblk, H, B);
  const size_t smem = 64 * (kHD + 4 * pairs) * sizeof(half);   // <= 24 KB (pairs <= 32)
  if (S <= 256)
    vae_qk_rope_quant_kernel<4><<<grid, 128, smem, stream>>>(a, q_oblk, q8.data_ptr<int8_t>(), qs.data_ptr<float>(),
        k8.data_ptr<int8_t>(), ks.data_ptr<float>(), anchor.data_ptr<int>(), anchor_vals.data_ptr<float>());
  else
    vae_qk_rope_quant_kernel<64><<<grid, 128, smem, stream>>>(a, q_oblk, q8.data_ptr<int8_t>(), qs.data_ptr<float>(),
        k8.data_ptr<int8_t>(), ks.data_ptr<float>(), anchor.data_ptr<int>(), anchor_vals.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {q8, qs, k8, ks, anchor};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vae_qk_rope_quant", &vae_qk_rope_quant,
        "VAE attention: ck q/k rms+rope fused into ck's SageAttention Q/K quantizer (head_dim 64)");
  m.def("vae_gemm_swiglu", &vae_gemm_swiglu, "w1 INT8 GEMM with SwiGLU in the epilogue (interleaved rows, fp16 h)",
        py::arg("xq"), py::arg("wq"), py::arg("xs"), py::arg("ws"), py::arg("bias"), py::arg("config") = 6);
  m.def("vae_out_quant", &vae_out_quant, "VAE to_out ConvRot INT8 quantizer from head-major attention output");
  m.def("vae_gemm", &vae_gemm, "comfy-kitchen-identical fused INT8 GEMM (fp16 out)",
        py::arg("xq"), py::arg("wq"), py::arg("xs"), py::arg("ws"), py::arg("bias"),
        py::arg("rscale") = py::none(), py::arg("resid") = py::none(), py::arg("config") = 6);
}
