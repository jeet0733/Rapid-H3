"""INT8 flash attention for GB10 (csrc/attn_int8.cu), JIT-built for sm_121."""
import os

import torch
from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))
_mod = None
CONFIG = int(os.environ.get("RAPIDH3_ATTN_CONFIG", "0"))


def ext():
    global _mod
    if _mod is None:
        build = os.path.join(os.environ.get("RAPIDH3_BUILD", os.path.join(_HERE, "build")), "attn")
        os.makedirs(build, exist_ok=True)
        _mod = load(
            name="rapidh3_attn_int8",
            sources=[os.path.join(_HERE, "csrc", "attn_int8.cu")],
            extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_121a,code=sm_121a", "-lineinfo"],
            build_directory=build, verbose=False)
    return _mod


def attention(q, k, v, scale=None, config=None, smooth_k=True, smooth_v=True, rotate=True, out=None):
    """q, k, v: [S, H, 128] bf16 (strided ok, last dim contiguous) -> [S, H, 128] bf16."""
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    return ext().int8_attention(q, k, v, scale, CONFIG if config is None else config, smooth_k, smooth_v, rotate, out)


def attention_bhsd(q, k, v, scale=None, **kw):
    """[1, H, S, 128] layout (comfy's skip_reshape convention) -> [1, H, S, 128]."""
    o = attention(q[0].transpose(0, 1), k[0].transpose(0, 1), v[0].transpose(0, 1), scale, **kw)
    return o.transpose(0, 1).unsqueeze(0)


def attention_rr(q, k, v, tab, qw, kw, eps, scale=None, config=94, smooth_k=True, smooth_v=True, rotate=True, out=None):
    """attention() on the raw qkv-projection q, k ([S, H, 128] bf16, strided ok), with comfy-kitchen's
    rms_rope_split_half_(q, k, tab, qw, kw, eps, rot_dim=2*tab.shape[-3]) fused into the pre-pass.
    Bit-identical to running that in-place op first and then attention(q, k, v, config=config); q, k untouched."""
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    return ext().int8_attention_rr(q, k, v, tab, qw, kw, eps, scale, config, smooth_k, smooth_v, rotate, out)


def attention_pre(k, v, q8, qs, part, tab, kw, eps, scale=None, config=94, smooth_k=True, smooth_v=True, rotate=True,
                  out=None):
    """Config 3xx: attention_rr whose q quantization and K/V statistics partials were
    already produced by the qkv GEMM epilogue (cutlass_fp8.int8_gemm_qkv_prep -> q8, qs, part); k, v are that
    GEMM's raw k/v columns ([S, H, 128] bf16 views). Bit-identical to attention_rr on the same GEMM's q/k/v."""
    scale = k.shape[-1] ** -0.5 if scale is None else scale
    return ext().int8_attention_pre(k, v, q8, qs, part, tab, kw, eps, scale, config, smooth_k, smooth_v, rotate, out)


def attention_rr_bias(q, k, v, tab, qw, kw, eps, nbias, bias, scale=None, config=94, smooth_k=True, smooth_v=True,
                      rotate=True, out=None):
    """attention_rr with the Jenga text-attention bias: + bias (natural-log units) on
    every query's logits of keys [0, nbias). Config 94 or 90 only; nbias 0 is attention_rr."""
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    return ext().int8_attention_rr_bias(q, k, v, tab, qw, kw, eps, scale, config, nbias, bias, smooth_k, smooth_v,
                                        rotate, out)
