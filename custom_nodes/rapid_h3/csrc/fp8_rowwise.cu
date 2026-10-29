// Row-wise scaled FP8 / INT8 GEMMs for GB10 (sm_121): out[M,N] = (XQ[M,K] . WQ[N,K]^T) * xs[M] * ws[N] (+ bias)
//
// CUTLASS 2.x warp-level mma.sync (m16n8k32 e4m3) kernel with an epilogue visitor tree, the
// structure PyTorch uses for sm_89.  PyTorch routes sm_12x to an sm_120a-only CUTLASS 3 kernel
// that aborts on sm_121; mma.sync kernels built for sm_121 run natively.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cute/tensor.hpp>
#include "rr_row.cuh"


// Adapted from NVIDIA CUTLASS v4.2.1 VisitorAuxStore (include/cutlass/epilogue/threadblock/fusion/visitor_store.hpp):
// the Arguments / Params / Callbacks skeleton and the begin_step / visit / get_callbacks code. Copyright (c) 2017-2025
// NVIDIA CORPORATION & AFFILIATES, BSD-3-Clause (licenses/CUTLASS-BSD-3-Clause.txt). The end_step bodies are ours.
//
// Epilogue store for a SwiGLU fc1: the weight rows are interleaved so each 8-column output vector
// holds [gate j..j+3 | up j..j+3]. The vector is rounded to bf16 (as the plain GEMM output is), then
// h = silu(gate) * up is computed in fp32 and 4 bf16 values land at column n/2 of a [M, N/2] output.
// The fc1 activation never round-trips through memory at full width.
namespace cutlass::epilogue::threadblock {
template <class ThreadMap, class StrideMNL>
struct VisitorSwigluStore {
  using Element = cutlass::bfloat16_t;
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
  static_assert(VecLength == 8, "SwiGLU store expects 8-wide bf16 output vectors");

  CUTLASS_HOST_DEVICE VisitorSwigluStore() {}
  CUTLASS_HOST_DEVICE VisitorSwigluStore(Params const& params, SharedStorage const&) : params_ptr(&params) {}
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
        __nv_bfloat162 h2[2];
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < 2; ++j) {
          float g0 = float(e[2 * j]), g1 = float(e[2 * j + 1]);
          float u0 = float(e[4 + 2 * j]), u1 = float(e[4 + 2 * j + 1]);
          h2[j] = __floats2bfloat162_rn(g0 / (1.f + __expf(-g0)) * u0, g1 / (1.f + __expf(-g1)) * u1);
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
// Epilogue store for the DiT's qkv projection with our INT8 attention's pre-pass partly fused in (config 3xx).
// The GEMM runs 256x128 tiles (int8 config 5, bit-identical output to config 6), so
// one tile is one 256-token statistics chunk (STAT_CHUNK) of one head of q, k or v. Every output vector is
// rounded to bf16 exactly as VisitorAuxStore does and staged in shared memory; then, per tile:
//   q: comfy-kitchen RMSNorm + RoPE, Hadamard-128, per-row INT8 (= attn_int8.cu quant_qk_rr's Q branch) ->
//      q8 [H][Spad][128], qs [H][Spad]; rows [S, Spad) zeroed; the bf16 q is never written;
//   k: written to out as usual; RMSNorm + RoPE, then the per-channel token sum in token order from 0 (=
//      stats_partial_rr's chunk partial) -> part[h][chunk][0][:];
//   v: written to out as usual; per-channel sum (same order), max, min -> part[h][chunk][1..3][:].
// So stats_partial_rr's pass over K and V and quant_qk_rr's pass over Q disappear; stats_final, a K-only quant
// and quant_v remain (attn_int8.cu int8_attention_pre).
// Built on the same CUTLASS VisitorAuxStore skeleton as VisitorSwigluStore (see the BSD-3-Clause note above).
template <class ThreadMap, class StrideMNL, int TM, int TN, int RD>
struct VisitorQkvPrep {
  using Element = cutlass::bfloat16_t;
  static_assert(TN == 128, "one head (128 channels) per tile column");
  static_assert(ThreadMap::kThreads == 256, "8 warps");
  struct Arguments {
    Element* ptr_out = nullptr;
    int64_t ldo = 0;
    StrideMNL dAux = {};
    int S = 0, Spad = 0, H = 0, nchunk = 0, rotate = 1;
    float eps = 0.f;
    const __nv_bfloat16* tab = nullptr;  // [S][RD/2][2][2]
    const __nv_bfloat16* qw = nullptr;
    const __nv_bfloat16* kw = nullptr;
    int8_t* q8 = nullptr;
    float* qs = nullptr;
    float* part = nullptr;               // [H][nchunk][4][128]
  };
  using Params = Arguments;
  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(ProblemShape const&, Arguments const& args, void*) { return args; }
  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const&, Arguments const&) { return 0; }
  struct SharedStorage {
    __nv_bfloat16 tile[TM * TN];
  };
  static int constexpr vec_bits = ThreadMap::kElementsPerAccess * sizeof_bits<Element>::value;
  using VecType = uint_bit_t<cute::min(128, vec_bits)>;
  static int constexpr VecLength = sizeof(VecType) / sizeof(Element);
  static_assert(VecLength == 8, "8-wide bf16 output vectors");
  CUTLASS_HOST_DEVICE VisitorQkvPrep() {}
  CUTLASS_HOST_DEVICE VisitorQkvPrep(Params const& params, SharedStorage const& ss)
      : params_ptr(&params), smem(const_cast<SharedStorage*>(&ss)) {}
  Params const* params_ptr;
  SharedStorage* smem;

  template <class RTensor, class CTensor, class ProblemShape>
  struct Callbacks : EmptyCallbacks {
    CUTLASS_DEVICE Callbacks(RTensor&& r, CTensor&& c, ProblemShape ps, Params const* pp, SharedStorage* sm, int m0,
                             int n0, int tid)
        : tC_rAux(cute::forward<RTensor>(r)), tC_cAux(cute::forward<CTensor>(c)), problem_shape(ps), params_ptr(pp),
          smem(sm), m0(m0), n0(n0), tid(tid) {}
    RTensor tC_rAux;
    CTensor tC_cAux;
    ProblemShape problem_shape;
    Params const* params_ptr;
    SharedStorage* smem;
    int m0, n0, tid;
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
      Params const& p = *params_ptr;
      const bool is_q = n0 < p.H * TN;
      auto src_v = filter(tC_rAux);
      auto coord_v = filter(tC_cAux(_, _, _, step_idx));
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(src_v); ++i) {
        auto coord = coord_v(i);
        if (!elem_less(coord, problem_shape)) continue;
        const int m = get<0>(coord), n = get<1>(coord);
        VecType raw = src_v(i);
        if (!is_q) *reinterpret_cast<VecType*>(p.ptr_out + int64_t(m) * p.ldo + n) = raw;
        *reinterpret_cast<VecType*>(smem->tile + (m - m0) * TN + (n - n0)) = raw;
      }
    }
    CUTLASS_DEVICE void end_epilogue() {
      Params const& p = *params_ptr;
      const int role = n0 / (p.H * TN), h = (n0 - role * p.H * TN) / TN;
      const int warp = tid >> 5, lane = tid & 31, nw = ThreadMap::kThreads / 32;
      const int nrows = min(TM, p.S - m0), chunk = m0 / TM;
      __nv_bfloat16* tile = smem->tile;
      __syncthreads();
      constexpr int R = 4;  // rows per warp in flight (independent shuffle chains)
      if (role <= 1) {
        rapidh3_rr::RRWeights<RD> w;
        w.load(role == 0 ? p.qw : p.kw, lane);
        const int rend = role == 0 ? min(TM, p.Spad - m0) : nrows;
        for (int r0 = warp; r0 < rend; r0 += nw * R) {
          const __nv_bfloat16* rows[R];
          const __nv_bfloat16* tabs[R];
          int8_t* dst[R];
          float* sdst[R];
          float x[R][4];
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < R; ++i) {
            const int r = r0 + i * nw, s = m0 + r;
            const bool ok = r < nrows;
            rows[i] = tile + (ok ? r : 0) * TN;
            tabs[i] = p.tab + (int64_t)(ok ? s : m0) * (RD * 2);
            dst[i] = role == 0 && ok ? p.q8 + ((int64_t)h * p.Spad + s) * TN : nullptr;
            sdst[i] = p.qs + (int64_t)h * p.Spad + s;
            if (role == 0 && !ok && r < rend) {  // q rows [S, Spad): zero, as quant_qk_rr
              *reinterpret_cast<int*>(p.q8 + ((int64_t)h * p.Spad + s) * TN + lane * 4) = 0;
              if (lane == 0) p.qs[(int64_t)h * p.Spad + s] = 0.f;
            }
          }
          rapidh3_rr::rms_rope_rows<RD, R>(rows, w, tabs, p.eps, lane, x);
          if (role == 0) {
            rapidh3_rr::quant_rows<R>(x, lane, p.rotate, dst, sdst);
          } else {
            __syncwarp();
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < R; ++i) {
              if (r0 + i * nw >= nrows) continue;
              __nv_bfloat162 a = __floats2bfloat162_rn(x[i][0], x[i][1]), b = __floats2bfloat162_rn(x[i][2], x[i][3]);
              uint2 st;
              st.x = *reinterpret_cast<uint32_t*>(&a); st.y = *reinterpret_cast<uint32_t*>(&b);
              *reinterpret_cast<uint2*>(tile + (r0 + i * nw) * TN + lane * 4) = st;
            }
          }
        }
        if (role == 0) return;
        __syncthreads();
      }
      float* part = p.part + ((int64_t)h * p.nchunk + chunk) * 4 * TN;
      if (tid < TN) {  // k: rms/rope'd sum; v: sum (token order from 0, as stats_partial(_rr))
        float acc = 0.f;
#pragma unroll 16
        for (int r = 0; r < nrows; ++r) acc += __bfloat162float(tile[r * TN + tid]);
        part[(role == 1 ? 0 : 1) * TN + tid] = acc;
      } else if (role == 2 && tid < 2 * TN) {
        const int d = tid - TN;
        float mx = -INFINITY, mn = INFINITY;
#pragma unroll 16
        for (int r = 0; r < nrows; ++r) {
          const float v = __bfloat162float(tile[r * TN + d]);
          mx = fmaxf(mx, v); mn = fminf(mn, v);
        }
        part[2 * TN + d] = mx;
        part[3 * TN + d] = mn;
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
        cute::move(tC_rAux), cute::move(tC_cAux), problem_shape, params_ptr, smem,
        threadblock_tile_offset.m() * TM, threadblock_tile_offset.n() * TN, thread_idx);
  }
};
}  // namespace cutlass::epilogue::threadblock

namespace {

using DtypeOut = cutlass::bfloat16_t;
using DtypeEpi = float;
using DtypeScale = float;

// FP8: e4m3 x e4m3 -> fp32 accumulate (sm_89 mma.sync).  INT8: s8 x s8 -> s32 (sm_80 mma.sync).
template <typename In, typename DtypeAcc, typename Arch, typename Op,
          typename TB, typename Warp, int Stages, typename Swizzle>
void run_t(const at::Tensor& xq, const at::Tensor& wq, const at::Tensor& xs, const at::Tensor& ws,
           const c10::optional<at::Tensor>& bias, at::Tensor& out) {
  using DtypeA = In;
  using DtypeB = In;
  const int M = xq.size(0), K = xq.size(1), N = wq.size(0);
  constexpr int AlignA = 16, AlignB = 16, AlignOut = 8;
  using Instr = cutlass::gemm::GemmShape<16, 8, 32>;
  constexpr int EpiStages = 1;
  using TileMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<TB, Warp, DtypeOut, AlignOut, EpiStages>;

  using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;
  using XScale = cutlass::epilogue::threadblock::VisitorColBroadcast<TileMap, DtypeScale, cute::Stride<cute::_1, cute::_0, int64_t>>;
  using WScale = cutlass::epilogue::threadblock::VisitorRowBroadcast<TileMap, DtypeScale, cute::Stride<cute::_0, cute::_1, int64_t>>;
  using Bias = cutlass::epilogue::threadblock::VisitorRowBroadcast<TileMap, DtypeOut, cute::Stride<cute::_0, cute::_1, int64_t>>;
  using Mul = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, DtypeEpi, DtypeEpi, cutlass::FloatRoundStyle::round_to_nearest>;
  using Add = cutlass::epilogue::threadblock::VisitorCompute<cutlass::plus, DtypeEpi, DtypeEpi, cutlass::FloatRoundStyle::round_to_nearest>;
  using EVTx = cutlass::epilogue::threadblock::Sm80EVT<Mul, Accum, XScale>;
  using EVTw = cutlass::epilogue::threadblock::Sm80EVT<Mul, EVTx, WScale>;
  using EVTb = cutlass::epilogue::threadblock::Sm80EVT<Add, EVTw, Bias>;
  using Store = cutlass::epilogue::threadblock::VisitorAuxStore<TileMap, DtypeOut, cutlass::FloatRoundStyle::round_to_nearest, cute::Stride<int64_t, cute::_1, int64_t>>;
  using EVTout = cutlass::epilogue::threadblock::Sm80EVT<Store, EVTb>;

  using Kernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      DtypeA, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, AlignA,
      DtypeB, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, AlignB,
      DtypeOut, cutlass::layout::RowMajor, AlignOut, DtypeAcc, DtypeEpi,
      cutlass::arch::OpClassTensorOp, Arch, TB, Warp, Instr, EVTout, Swizzle,
      Stages, Op, EpiStages>::GemmKernel;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

  cutlass::gemm::GemmCoord ps(M, N, K);
  typename XScale::Arguments xa{(DtypeScale*)xs.data_ptr(), DtypeScale(1), {cute::_1{}, cute::_0{}, (int64_t)M}};
  typename WScale::Arguments wa{(DtypeScale*)ws.data_ptr(), DtypeScale(1), {cute::_0{}, cute::_1{}, (int64_t)N}};
  typename Bias::Arguments ba{bias.has_value() ? (DtypeOut*)bias->data_ptr() : nullptr, DtypeOut(0), {cute::_0{}, cute::_1{}, (int64_t)N}};
  typename Store::Arguments oa{(DtypeOut*)out.data_ptr(), {(int64_t)N, cute::_1{}, (int64_t)M * N}};
  typename EVTout::Arguments cb{{{{{}, xa, {}}, wa, {}}, ba, {}}, oa};

  typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, ps, 1, cb,
                                (DtypeA*)xq.data_ptr(), (DtypeB*)wq.data_ptr(), nullptr, nullptr,
                                (int64_t)M * K, (int64_t)N * K, 0, 0, K, K, 0, 0);
  Gemm gemm;
  auto ws_size = Gemm::get_workspace_size(args);
  auto workspace = at::empty({(int64_t)ws_size}, xq.options().dtype(at::kByte));
  TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess, "fp8_rowwise: cannot implement");
  TORCH_CHECK(gemm.initialize(args, workspace.data_ptr()) == cutlass::Status::kSuccess, "fp8_rowwise: init failed");
  auto st = gemm(at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(st == cutlass::Status::kSuccess, "fp8_rowwise: run failed ", cutlass::cutlassGetStatusString(st));
}

// INT8 GEMM with the DiT's gated residual in the epilogue, in place:
//   resid[M,N] = resid + bf16(acc * xs * ws) * gate[N]
// matching ComfyUI's `x.addcmul_(y, gate)` (y rounded to bf16 first, then one fp32 fma, one rounding),
// so the bf16 GEMM output never round-trips through memory.
template <typename TB, typename Warp, int Stages, typename Swizzle>
void run_gr(const at::Tensor& xq, const at::Tensor& wq, const at::Tensor& xs, const at::Tensor& ws,
            const at::Tensor& gate, at::Tensor& resid) {
  using DtypeA = int8_t;
  using DtypeB = int8_t;
  const int M = xq.size(0), K = xq.size(1), N = wq.size(0);
  constexpr int AlignA = 16, AlignB = 16, AlignOut = 8;
  using Instr = cutlass::gemm::GemmShape<16, 8, 32>;
  constexpr int EpiStages = 1;
  using TileMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<TB, Warp, DtypeOut, AlignOut, EpiStages>;

  using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;
  using XScale = cutlass::epilogue::threadblock::VisitorColBroadcast<TileMap, DtypeScale, cute::Stride<cute::_1, cute::_0, int64_t>>;
  using WScale = cutlass::epilogue::threadblock::VisitorRowBroadcast<TileMap, DtypeScale, cute::Stride<cute::_0, cute::_1, int64_t>>;
  using Gate = cutlass::epilogue::threadblock::VisitorRowBroadcast<TileMap, DtypeOut, cute::Stride<cute::_0, cute::_1, int64_t>>;
  using Resid = cutlass::epilogue::threadblock::VisitorAuxLoad<TileMap, DtypeOut, cute::Stride<int64_t, cute::_1, int64_t>>;
  using Mul = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, DtypeEpi, DtypeEpi, cutlass::FloatRoundStyle::round_to_nearest>;
  using MulBf = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, DtypeOut, DtypeEpi, cutlass::FloatRoundStyle::round_to_nearest>;
  using Fma = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiply_add, DtypeOut, DtypeEpi, cutlass::FloatRoundStyle::round_to_nearest>;
  using EVTx = cutlass::epilogue::threadblock::Sm80EVT<Mul, Accum, XScale>;
  using EVTy = cutlass::epilogue::threadblock::Sm80EVT<MulBf, EVTx, WScale>;
  using EVTg = cutlass::epilogue::threadblock::Sm80EVT<Fma, EVTy, Gate, Resid>;
  using Store = cutlass::epilogue::threadblock::VisitorAuxStore<TileMap, DtypeOut, cutlass::FloatRoundStyle::round_to_nearest, cute::Stride<int64_t, cute::_1, int64_t>>;
  using EVTout = cutlass::epilogue::threadblock::Sm80EVT<Store, EVTg>;

  using Kernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      DtypeA, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, AlignA,
      DtypeB, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, AlignB,
      DtypeOut, cutlass::layout::RowMajor, AlignOut, int32_t, DtypeEpi,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, TB, Warp, Instr, EVTout, Swizzle,
      Stages, cutlass::arch::OpMultiplyAddSaturate, EpiStages>::GemmKernel;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

  const int64_t ldr = resid.stride(0);
  cutlass::gemm::GemmCoord ps(M, N, K);
  typename XScale::Arguments xa{(DtypeScale*)xs.data_ptr(), DtypeScale(1), {cute::_1{}, cute::_0{}, (int64_t)M}};
  typename WScale::Arguments wa{(DtypeScale*)ws.data_ptr(), DtypeScale(1), {cute::_0{}, cute::_1{}, (int64_t)N}};
  typename Gate::Arguments ga{(DtypeOut*)gate.data_ptr(), DtypeOut(0), {cute::_0{}, cute::_1{}, (int64_t)N}};
  typename Resid::Arguments ra{(DtypeOut*)resid.data_ptr(), DtypeOut(0), {ldr, cute::_1{}, ldr * M}};
  typename Store::Arguments oa{(DtypeOut*)resid.data_ptr(), {ldr, cute::_1{}, ldr * M}};
  typename EVTout::Arguments cb{{{{{}, xa, {}}, wa, {}}, ga, ra, {}}, oa};

  typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, ps, 1, cb,
                                (DtypeA*)xq.data_ptr(), (DtypeB*)wq.data_ptr(), nullptr, nullptr,
                                (int64_t)M * K, (int64_t)N * K, 0, 0, K, K, 0, 0);
  Gemm gemm;
  auto ws_size = Gemm::get_workspace_size(args);
  auto workspace = at::empty({(int64_t)ws_size}, xq.options().dtype(at::kByte));
  TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess, "int8_gemm_gated: cannot implement");
  TORCH_CHECK(gemm.initialize(args, workspace.data_ptr()) == cutlass::Status::kSuccess, "int8_gemm_gated: init failed");
  auto st = gemm(at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(st == cutlass::Status::kSuccess, "int8_gemm_gated: run failed ", cutlass::cutlassGetStatusString(st));
}

// fc1 with SwiGLU in the epilogue: out[M, N/2] = bf16(silu(g) * u), g/u = bf16(acc * xs * ws) from the
// interleaved weight layout [g 4j..4j+3 | u 4j..4j+3] (see VisitorSwigluStore).
template <typename TB, typename Warp, int Stages, typename Swizzle>
void run_swiglu(const at::Tensor& xq, const at::Tensor& wq, const at::Tensor& xs, const at::Tensor& ws, at::Tensor& out) {
  const int M = xq.size(0), K = xq.size(1), N = wq.size(0);
  constexpr int AlignA = 16, AlignB = 16, AlignOut = 8;
  using Instr = cutlass::gemm::GemmShape<16, 8, 32>;
  constexpr int EpiStages = 1;
  using TileMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<TB, Warp, DtypeOut, AlignOut, EpiStages>;
  using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;
  using XScale = cutlass::epilogue::threadblock::VisitorColBroadcast<TileMap, DtypeScale, cute::Stride<cute::_1, cute::_0, int64_t>>;
  using WScale = cutlass::epilogue::threadblock::VisitorRowBroadcast<TileMap, DtypeScale, cute::Stride<cute::_0, cute::_1, int64_t>>;
  using Mul = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, DtypeEpi, DtypeEpi, cutlass::FloatRoundStyle::round_to_nearest>;
  using EVTx = cutlass::epilogue::threadblock::Sm80EVT<Mul, Accum, XScale>;
  using EVTw = cutlass::epilogue::threadblock::Sm80EVT<Mul, EVTx, WScale>;
  using StrideO = cute::Stride<int64_t, cute::_1, int64_t>;
  using Store = cutlass::epilogue::threadblock::VisitorSwigluStore<TileMap, StrideO>;
  using EVTout = cutlass::epilogue::threadblock::Sm80EVT<Store, EVTw>;
  using Kernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      int8_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, AlignA,
      int8_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, AlignB,
      DtypeOut, cutlass::layout::RowMajor, AlignOut, int32_t, DtypeEpi,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, TB, Warp, Instr, EVTout, Swizzle,
      Stages, cutlass::arch::OpMultiplyAddSaturate, EpiStages>::GemmKernel;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

  cutlass::gemm::GemmCoord ps(M, N, K);
  typename XScale::Arguments xa{(DtypeScale*)xs.data_ptr(), DtypeScale(1), {cute::_1{}, cute::_0{}, (int64_t)M}};
  typename WScale::Arguments wa{(DtypeScale*)ws.data_ptr(), DtypeScale(1), {cute::_0{}, cute::_1{}, (int64_t)N}};
  typename Store::Arguments oa{(DtypeOut*)out.data_ptr(), out.stride(0), {(int64_t)N, cute::_1{}, (int64_t)M * N}};
  typename EVTout::Arguments cb{{{{}, xa, {}}, wa, {}}, oa};
  typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, ps, 1, cb,
                                (int8_t*)xq.data_ptr(), (int8_t*)wq.data_ptr(), nullptr, nullptr,
                                (int64_t)M * K, (int64_t)N * K, 0, 0, K, K, 0, 0);
  Gemm gemm;
  auto ws_size = Gemm::get_workspace_size(args);
  auto workspace = at::empty({(int64_t)ws_size}, xq.options().dtype(at::kByte));
  TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess, "int8_gemm_swiglu: cannot implement");
  TORCH_CHECK(gemm.initialize(args, workspace.data_ptr()) == cutlass::Status::kSuccess, "int8_gemm_swiglu: init failed");
  auto st = gemm(at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(st == cutlass::Status::kSuccess, "int8_gemm_swiglu: run failed ", cutlass::cutlassGetStatusString(st));
}

// qkv projection with the attention pre-pass partly fused (VisitorQkvPrep). Same EVT as run_t's int8 path
// ((acc * xs) * ws + bias(0), bf16 RN), so k/v in `out` are bit-identical to int8_rowwise_gemm.
template <int RD, typename TB, typename Warp, int Stages, typename Swizzle>
void run_qkv_prep(const at::Tensor& xq, const at::Tensor& wq, const at::Tensor& xs, const at::Tensor& ws,
                  at::Tensor& out, const at::Tensor& tab, const at::Tensor& qw, const at::Tensor& kw, double eps,
                  int H, int Spad, bool rotate, at::Tensor& q8, at::Tensor& qs, at::Tensor& part) {
  const int M = xq.size(0), K = xq.size(1), N = wq.size(0);
  constexpr int AlignA = 16, AlignB = 16, AlignOut = 8;
  using Instr = cutlass::gemm::GemmShape<16, 8, 32>;
  constexpr int EpiStages = 1;
  using TileMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<TB, Warp, DtypeOut, AlignOut, EpiStages>;
  using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;
  using XScale = cutlass::epilogue::threadblock::VisitorColBroadcast<TileMap, DtypeScale, cute::Stride<cute::_1, cute::_0, int64_t>>;
  using WScale = cutlass::epilogue::threadblock::VisitorRowBroadcast<TileMap, DtypeScale, cute::Stride<cute::_0, cute::_1, int64_t>>;
  using Bias = cutlass::epilogue::threadblock::VisitorRowBroadcast<TileMap, DtypeOut, cute::Stride<cute::_0, cute::_1, int64_t>>;
  using Mul = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, DtypeEpi, DtypeEpi, cutlass::FloatRoundStyle::round_to_nearest>;
  using Add = cutlass::epilogue::threadblock::VisitorCompute<cutlass::plus, DtypeEpi, DtypeEpi, cutlass::FloatRoundStyle::round_to_nearest>;
  using EVTx = cutlass::epilogue::threadblock::Sm80EVT<Mul, Accum, XScale>;
  using EVTw = cutlass::epilogue::threadblock::Sm80EVT<Mul, EVTx, WScale>;
  using EVTb = cutlass::epilogue::threadblock::Sm80EVT<Add, EVTw, Bias>;
  using StrideO = cute::Stride<int64_t, cute::_1, int64_t>;
  using Store = cutlass::epilogue::threadblock::VisitorQkvPrep<TileMap, StrideO, TB::kM, TB::kN, RD>;
  using EVTout = cutlass::epilogue::threadblock::Sm80EVT<Store, EVTb>;
  using Kernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      int8_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, AlignA,
      int8_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, AlignB,
      DtypeOut, cutlass::layout::RowMajor, AlignOut, int32_t, DtypeEpi,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, TB, Warp, Instr, EVTout, Swizzle,
      Stages, cutlass::arch::OpMultiplyAddSaturate, EpiStages>::GemmKernel;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  cutlass::gemm::GemmCoord ps(M, N, K);
  typename XScale::Arguments xa{(DtypeScale*)xs.data_ptr(), DtypeScale(1), {cute::_1{}, cute::_0{}, (int64_t)M}};
  typename WScale::Arguments wa{(DtypeScale*)ws.data_ptr(), DtypeScale(1), {cute::_0{}, cute::_1{}, (int64_t)N}};
  typename Bias::Arguments ba{nullptr, DtypeOut(0), {cute::_0{}, cute::_1{}, (int64_t)N}};
  typename Store::Arguments oa;
  oa.ptr_out = (DtypeOut*)out.data_ptr();
  oa.ldo = out.stride(0);
  oa.dAux = {(int64_t)N, cute::_1{}, (int64_t)M * N};
  oa.S = M; oa.Spad = Spad; oa.H = H; oa.nchunk = part.size(1); oa.rotate = rotate;
  oa.eps = static_cast<float>(eps);
  oa.tab = reinterpret_cast<const __nv_bfloat16*>(tab.data_ptr());
  oa.qw = reinterpret_cast<const __nv_bfloat16*>(qw.data_ptr());
  oa.kw = reinterpret_cast<const __nv_bfloat16*>(kw.data_ptr());
  oa.q8 = q8.data_ptr<int8_t>(); oa.qs = qs.data_ptr<float>(); oa.part = part.data_ptr<float>();
  typename EVTout::Arguments cb{{{{{}, xa, {}}, wa, {}}, ba, {}}, oa};
  typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, ps, 1, cb,
                                (int8_t*)xq.data_ptr(), (int8_t*)wq.data_ptr(), nullptr, nullptr,
                                (int64_t)M * K, (int64_t)N * K, 0, 0, K, K, 0, 0);
  Gemm gemm;
  auto ws_size = Gemm::get_workspace_size(args);
  auto workspace = at::empty({(int64_t)ws_size}, xq.options().dtype(at::kByte));
  TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess, "int8_gemm_qkv_prep: cannot implement");
  TORCH_CHECK(gemm.initialize(args, workspace.data_ptr()) == cutlass::Status::kSuccess, "int8_gemm_qkv_prep: init failed");
  auto st = gemm(at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(st == cutlass::Status::kSuccess, "int8_gemm_qkv_prep: run failed ", cutlass::cutlassGetStatusString(st));
}

template <typename TB, typename Warp, int Stages, typename Swizzle, bool FastAccum>
void run(const at::Tensor& xq, const at::Tensor& wq, const at::Tensor& xs, const at::Tensor& ws,
         const c10::optional<at::Tensor>& bias, at::Tensor& out) {
  using Op = std::conditional_t<FastAccum, cutlass::arch::OpMultiplyAddFastAccum, cutlass::arch::OpMultiplyAdd>;
  run_t<cutlass::float_e4m3_t, float, cutlass::arch::Sm89, Op, TB, Warp, Stages, Swizzle>(xq, wq, xs, ws, bias, out);
}

template <typename TB, typename Warp, int Stages, typename Swizzle, bool Unused>
void run_i8(const at::Tensor& xq, const at::Tensor& wq, const at::Tensor& xs, const at::Tensor& ws,
            const c10::optional<at::Tensor>& bias, at::Tensor& out) {
  run_t<int8_t, int32_t, cutlass::arch::Sm80, cutlass::arch::OpMultiplyAddSaturate, TB, Warp, Stages, Swizzle>(
      xq, wq, xs, ws, bias, out);
}

using Ident = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;
using StreamK = cutlass::gemm::threadblock::ThreadblockSwizzleStreamK;
template <int M, int N, int K> using S = cutlass::gemm::GemmShape<M, N, K>;

}  // namespace

// config: index into the tile table below; fast_accum trades accumulation precision for speed
at::Tensor fp8_rowwise_gemm(at::Tensor xq, at::Tensor wq, at::Tensor xs, at::Tensor ws,
                            c10::optional<at::Tensor> bias, int64_t config, bool fast_accum) {
  TORCH_CHECK(xq.is_contiguous() && wq.is_contiguous(), "xq [M,K] and wq [N,K] must be contiguous");
  TORCH_CHECK(xq.size(1) == wq.size(1), "K mismatch");
  TORCH_CHECK(xs.numel() == xq.size(0) && ws.numel() == wq.size(0), "scale sizes");
  TORCH_CHECK(xs.scalar_type() == at::kFloat && ws.scalar_type() == at::kFloat, "scales must be fp32");
  c10::cuda::CUDAGuard g(xq.device());
  auto out = at::empty({xq.size(0), wq.size(0)}, xq.options().dtype(at::kBFloat16));
#define CASE(i, TB, W, ST, SW)                                                        \
  case i:                                                                            \
    fast_accum ? run<TB, W, ST, SW, true>(xq, wq, xs, ws, bias, out)                 \
               : run<TB, W, ST, SW, false>(xq, wq, xs, ws, bias, out);               \
    break;
  using TB0 = S<128, 128, 64>;  using W0 = S<64, 64, 64>;
  using TB2 = S<128, 256, 64>;
  using TB3 = S<256, 128, 64>;
  using TB4 = S<128, 128, 128>; using W4 = S<64, 64, 128>;
  using W5 = S<64, 128, 64>;
  using W6 = S<128, 64, 64>;
  switch (config) {
    CASE(0, TB0, W0, 5, StreamK)   // PyTorch's sm_89 large-M choice
    CASE(1, TB0, W0, 5, Ident)
    CASE(2, TB2, W0, 3, Ident)
    CASE(3, TB3, W0, 3, Ident)
    CASE(4, TB4, W4, 3, Ident)
    CASE(5, TB2, W5, 3, Ident)
    CASE(6, TB3, W6, 3, Ident)
    CASE(7, TB0, W0, 4, Ident)
    default: TORCH_CHECK(false, "unknown config ", config);
  }
#undef CASE
  return out;
}

at::Tensor int8_rowwise_gemm(at::Tensor xq, at::Tensor wq, at::Tensor xs, at::Tensor ws,
                             c10::optional<at::Tensor> bias, int64_t config) {
  TORCH_CHECK(xq.is_contiguous() && wq.is_contiguous(), "xq [M,K] and wq [N,K] must be contiguous");
  TORCH_CHECK(xq.scalar_type() == at::kChar && wq.scalar_type() == at::kChar, "int8 inputs");
  TORCH_CHECK(xq.size(1) == wq.size(1), "K mismatch");
  c10::cuda::CUDAGuard g(xq.device());
  auto out = at::empty({xq.size(0), wq.size(0)}, xq.options().dtype(at::kBFloat16));
  using TB0 = S<128, 128, 64>;  using W0 = S<64, 64, 64>;
  using TB2 = S<128, 256, 64>;
  using TB3 = S<256, 128, 64>;
  using TB4 = S<128, 128, 128>; using W4 = S<64, 64, 128>;
  switch (config) {
    case 1: run_i8<TB0, W0, 5, Ident, false>(xq, wq, xs, ws, bias, out); break;
    case 2: run_i8<TB2, W0, 3, Ident, false>(xq, wq, xs, ws, bias, out); break;
    case 3: run_i8<TB3, W0, 3, Ident, false>(xq, wq, xs, ws, bias, out); break;
    case 4: run_i8<TB4, W4, 3, Ident, false>(xq, wq, xs, ws, bias, out); break;
    // tuning sweep (2026-10-13)
    case 5: run_i8<TB3, W0, 4, Ident, false>(xq, wq, xs, ws, bias, out); break;
    case 6: run_i8<TB2, W0, 4, Ident, false>(xq, wq, xs, ws, bias, out); break;
    case 7: run_i8<TB3, S<128, 64, 64>, 4, Ident, false>(xq, wq, xs, ws, bias, out); break;
    case 8: run_i8<TB2, S<64, 128, 64>, 4, Ident, false>(xq, wq, xs, ws, bias, out); break;
    case 9: run_i8<TB0, W0, 6, Ident, false>(xq, wq, xs, ws, bias, out); break;
    case 10: run_i8<S<256, 128, 128>, W4, 2, Ident, false>(xq, wq, xs, ws, bias, out); break;
    case 11: run_i8<S<128, 256, 128>, W4, 2, Ident, false>(xq, wq, xs, ws, bias, out); break;
    case 12: run_i8<S<256, 256, 64>, W0, 3, Ident, false>(xq, wq, xs, ws, bias, out); break;
    case 13: run_i8<TB3, W0, 4, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, false>(xq, wq, xs, ws, bias, out); break;
    case 14: run_i8<TB3, W0, 4, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<16>, false>(xq, wq, xs, ws, bias, out); break;
    case 15: run_i8<TB3, W0, 4, StreamK, false>(xq, wq, xs, ws, bias, out); break;
    // round 3 (2026-10-24), none better than 6: 16/17/20 = two CTAs per SM (48 KB smem each, one CTA's
    // epilogue overlaps the other's main loop), 18/19 = config 6 with rasterization swizzle 16/32
    case 16: run_i8<TB0, W0, 3, Ident, false>(xq, wq, xs, ws, bias, out); break;
    case 17: run_i8<TB0, W0, 3, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, false>(xq, wq, xs, ws, bias, out); break;
    case 18: run_i8<TB2, W0, 4, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<16>, false>(xq, wq, xs, ws, bias, out); break;
    case 19: run_i8<TB2, W0, 4, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<32>, false>(xq, wq, xs, ws, bias, out); break;
    case 20: run_i8<TB0, W0, 3, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<16>, false>(xq, wq, xs, ws, bias, out); break;
    default: TORCH_CHECK(false, "unknown int8 config ", config);
  }
  return out;
}

// resid += bf16(xq.wq^T * xs * ws) * gate, in place (resid [M, N] bf16 with unit column stride)
void int8_gemm_gated_residual(at::Tensor xq, at::Tensor wq, at::Tensor xs, at::Tensor ws,
                              at::Tensor gate, at::Tensor resid, int64_t config) {
  TORCH_CHECK(xq.is_contiguous() && wq.is_contiguous(), "xq [M,K] and wq [N,K] must be contiguous");
  TORCH_CHECK(xq.scalar_type() == at::kChar && wq.scalar_type() == at::kChar, "int8 inputs");
  TORCH_CHECK(resid.scalar_type() == at::kBFloat16 && gate.scalar_type() == at::kBFloat16, "bf16 resid/gate");
  TORCH_CHECK(resid.size(0) == xq.size(0) && resid.size(1) == wq.size(0) && resid.stride(1) == 1, "resid shape");
  TORCH_CHECK(gate.numel() == wq.size(0) && gate.is_contiguous(), "gate [N]");
  TORCH_CHECK(xs.numel() == xq.size(0) && ws.numel() == wq.size(0), "scale sizes");
  if (xq.size(0) == 0) return;
  c10::cuda::CUDAGuard g(xq.device());
  using TB0 = S<128, 128, 64>;  using W0 = S<64, 64, 64>;
  using TB2 = S<128, 256, 64>;
  using TB3 = S<256, 128, 64>;
  using TB4 = S<128, 128, 128>; using W4 = S<64, 64, 128>;
  switch (config) {
    case 1: run_gr<TB0, W0, 5, Ident>(xq, wq, xs, ws, gate, resid); break;
    case 3: run_gr<TB3, W0, 3, Ident>(xq, wq, xs, ws, gate, resid); break;
    case 4: run_gr<TB4, W4, 3, Ident>(xq, wq, xs, ws, gate, resid); break;
    case 6: run_gr<TB2, W0, 4, Ident>(xq, wq, xs, ws, gate, resid); break;
    default: TORCH_CHECK(false, "unknown gated config ", config);
  }
}

// fc1 + SwiGLU: wq rows interleaved [g 4j..4j+3 | u 4j..4j+3]; returns h [M, N/2] bf16
at::Tensor int8_gemm_swiglu(at::Tensor xq, at::Tensor wq, at::Tensor xs, at::Tensor ws, int64_t config) {
  TORCH_CHECK(xq.is_contiguous() && wq.is_contiguous(), "xq [M,K] and wq [N,K] must be contiguous");
  TORCH_CHECK(xq.scalar_type() == at::kChar && wq.scalar_type() == at::kChar, "int8 inputs");
  TORCH_CHECK(xq.size(1) == wq.size(1) && wq.size(0) % 8 == 0, "shapes");
  TORCH_CHECK(xs.numel() == xq.size(0) && ws.numel() == wq.size(0), "scale sizes");
  c10::cuda::CUDAGuard g(xq.device());
  auto out = at::empty({xq.size(0), wq.size(0) / 2}, xq.options().dtype(at::kBFloat16));
  using TB0 = S<128, 128, 64>;  using W0 = S<64, 64, 64>;
  using TB2 = S<128, 256, 64>;
  using TB3 = S<256, 128, 64>;
  switch (config) {
    case 1: run_swiglu<TB0, W0, 5, Ident>(xq, wq, xs, ws, out); break;
    case 3: run_swiglu<TB3, W0, 3, Ident>(xq, wq, xs, ws, out); break;
    case 5: run_swiglu<TB3, W0, 4, Ident>(xq, wq, xs, ws, out); break;
    case 6: run_swiglu<TB2, W0, 4, Ident>(xq, wq, xs, ws, out); break;
    default: TORCH_CHECK(false, "unknown swiglu config ", config);
  }
  return out;
}

// qkv projection (xq [S, K] int8, wq [3*H*128, K] int8) with our INT8 attention's pre-pass partly fused into the
// epilogue (VisitorQkvPrep). Returns {qkv [S, 3*H*128] bf16 (q columns NOT written),
// q8 [H][Spad][128] int8, qs [H][Spad] fp32, part [H][nchunk][4][128] fp32}; k/v columns, q8, qs and part are
// bit-identical to int8_rowwise_gemm (config 5/6) followed by attn_int8.cu's fused rms/rope pre-pass.
std::vector<at::Tensor> int8_gemm_qkv_prep(at::Tensor xq, at::Tensor wq, at::Tensor xs, at::Tensor ws, at::Tensor tab,
                                           at::Tensor qw, at::Tensor kw, double eps, int64_t heads, bool rotate) {
  TORCH_CHECK(xq.is_contiguous() && wq.is_contiguous(), "xq [M,K] and wq [N,K] must be contiguous");
  TORCH_CHECK(xq.scalar_type() == at::kChar && wq.scalar_type() == at::kChar, "int8 inputs");
  TORCH_CHECK(xq.size(1) == wq.size(1) && wq.size(0) == 3 * heads * 128, "shapes");
  TORCH_CHECK(xs.numel() == xq.size(0) && ws.numel() == wq.size(0), "scale sizes");
  TORCH_CHECK(tab.is_contiguous() && tab.scalar_type() == at::kBFloat16 && tab.numel() == xq.size(0) * 192,
              "rope table must be [1, S, 1, 48, 2, 2] bf16 (rot_dim 96)");
  TORCH_CHECK(qw.is_contiguous() && kw.is_contiguous() && qw.numel() == 128 && kw.numel() == 128 &&
              qw.scalar_type() == at::kBFloat16 && kw.scalar_type() == at::kBFloat16, "norm weights [128] bf16");
  c10::cuda::CUDAGuard g(xq.device());
  const int64_t M = xq.size(0), H = heads, Spad = (M + 127) / 128 * 128, nchunk = (M + 255) / 256;
  auto out = at::empty({M, wq.size(0)}, xq.options().dtype(at::kBFloat16));
  auto q8 = at::empty({H, Spad, 128}, xq.options().dtype(at::kChar));
  auto qs = at::empty({H, Spad}, xq.options().dtype(at::kFloat));
  auto part = at::empty({H, nchunk, 4, 128}, xq.options().dtype(at::kFloat));
  run_qkv_prep<96, S<256, 128, 64>, S<64, 64, 64>, 4, Ident>(xq, wq, xs, ws, out, tab, qw, kw, eps, (int)H, (int)Spad,
                                                             rotate, q8, qs, part);
  return {out, q8, qs, part};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("int8_gemm_qkv_prep", &int8_gemm_qkv_prep, "qkv INT8 GEMM with the INT8 attention pre-pass partly fused",
        py::arg("xq"), py::arg("wq"), py::arg("xs"), py::arg("ws"), py::arg("tab"), py::arg("qw"), py::arg("kw"),
        py::arg("eps"), py::arg("heads"), py::arg("rotate") = true);
  m.def("int8_gemm_swiglu", &int8_gemm_swiglu, "INT8 fc1 GEMM with SwiGLU epilogue (sm_121)",
        py::arg("xq"), py::arg("wq"), py::arg("xs"), py::arg("ws"), py::arg("config") = 6);
  m.def("int8_gemm_gated_residual", &int8_gemm_gated_residual, "INT8 GEMM + in-place gated residual (sm_121)",
        py::arg("xq"), py::arg("wq"), py::arg("xs"), py::arg("ws"), py::arg("gate"), py::arg("resid"),
        py::arg("config") = 6);
  m.def("int8_rowwise_gemm", &int8_rowwise_gemm, "row-wise scaled INT8 GEMM (sm_121)",
        py::arg("xq"), py::arg("wq"), py::arg("xs"), py::arg("ws"), py::arg("bias") = py::none(),
        py::arg("config") = 2);
  m.def("fp8_rowwise_gemm", &fp8_rowwise_gemm, "row-wise scaled FP8 GEMM (sm_121)",
        py::arg("xq"), py::arg("wq"), py::arg("xs"), py::arg("ws"), py::arg("bias") = py::none(),
        py::arg("config") = 2, py::arg("fast_accum") = true);
}
