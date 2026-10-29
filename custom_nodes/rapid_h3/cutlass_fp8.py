"""JIT build of csrc/fp8_rowwise.cu for sm_121 (CUTLASS headers from $RAPIDH3_CUTLASS)."""
import os

import torch
from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))
_mod = None


def ext():
    global _mod
    if _mod is None:
        cutlass = os.environ.get("RAPIDH3_CUTLASS", "/work/third_party/cutlass")
        build = os.environ.get("RAPIDH3_BUILD", os.path.join(_HERE, "build"))
        os.makedirs(build, exist_ok=True)
        _mod = load(
            name="rapidh3_fp8_rowwise",
            sources=[os.path.join(_HERE, "csrc", "fp8_rowwise.cu")],
            extra_include_paths=[f"{cutlass}/include", f"{cutlass}/tools/util/include"],
            extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr",
                               "-gencode=arch=compute_121,code=sm_121",
                               "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1"],
            build_directory=build, verbose=False)
    return _mod


def fp8_gemm(xq, xs, wq, ws, bias=None, config=2, fast_accum=True):
    """xq [M,K] fp8, xs [M,1] fp32, wq [N,K] fp8, ws [1,N] fp32 -> bf16 [M,N]."""
    return ext().fp8_rowwise_gemm(xq, wq, xs.reshape(-1), ws.reshape(-1), bias, config, fast_accum)


def int8_gemm(xq, xs, wq, ws, bias=None, config=2):
    """xq [M,K] int8, xs [M,1] fp32, wq [N,K] int8, ws [1,N] fp32 -> bf16 [M,N]."""
    return ext().int8_rowwise_gemm(xq, wq, xs.reshape(-1), ws.reshape(-1), bias, config)


def int8_gemm_qkv_prep(xq, xs, wq, ws, tab, qw, kw, eps, heads, rotate=True):
    """qkv projection with our INT8 attention's pre-pass partly fused into the epilogue (config 3xx):
    returns (qkv [M, 3*heads*128] bf16 with the q columns left unwritten, q8, qs, part) for attn_int8.attention_pre.
    tab: rope table [1, M, 1, 48, 2, 2] bf16 (rot_dim 96); qw, kw: q/k RMSNorm weights [128] bf16."""
    return ext().int8_gemm_qkv_prep(xq, wq, xs.reshape(-1), ws.reshape(-1), tab, qw, kw, eps, heads, rotate)


def int8_gemm_gated_residual(xq, xs, wq, ws, gate, resid, config=6):
    """resid[M,N] += bf16(xq @ wq^T * xs * ws) * gate[N], in place (ComfyUI addcmul_ semantics)."""
    ext().int8_gemm_gated_residual(xq, wq, xs.reshape(-1), ws.reshape(-1), gate, resid, config)
    return resid


def swiglu_perm(n2, device):
    """Row order for an fc1 weight [gate (F rows) | up (F rows)] so each 8-row group is
    [gate 4j..4j+3 | up 4j..4j+3], the layout int8_gemm_swiglu's epilogue expects."""
    import torch
    f = n2 // 2
    j = torch.arange(0, f, 4, device=device)
    blk = torch.arange(4, device=device)
    return torch.stack([j[:, None] + blk, f + j[:, None] + blk], dim=1).reshape(-1)


def int8_gemm_swiglu(xq, xs, wq_perm, ws_perm, config=6):
    """h [M, F] = bf16(silu(g) * u) for an fc1 whose rows were permuted with swiglu_perm."""
    return ext().int8_gemm_swiglu(xq, wq_perm, xs.reshape(-1), ws_perm.reshape(-1), config)
