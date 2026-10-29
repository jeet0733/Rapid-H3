"""Round-5 VAE kernels (RAPIDH3_VAE_FAST=4) against the comfy-kitchen path they replace: bits and time.

vae_qk_rope_quant (q/k rms+rope fused into ck's SageAttention Q/K quantizer) + ck quant_v + sage_sdpa_prequantized
vs ck.rms_rope_split_half_ (in place) + ck.int8_attention, on the decoder Attention's real layout (qkv [B, S, H, 192]
from the qkv GEMM, rope table from the decoder's own RotaryEmbeddingND). Compares the int8 Q/K codes, their scales,
the anchor choice and the attention output.

Runs inside the ComfyUI image: python vae_r5_test.py   (RAPIDH3_NODE_DIR = a copy of custom_nodes/rapid_h3)
"""
import os
import sys
import time

import torch

sys.path.insert(0, "/opt/ComfyUI")
sys.path.insert(0, os.environ.get("RAPIDH3_NODE_DIR", "/work/build/vae_r5/rapid_h3"))
import vae_fast  # noqa: E402
from comfy_kitchen.backends import cuda as ckc  # noqa: E402
import comfy.quant_ops  # noqa: E402
from comfy.ldm.minimax.vae import RotaryEmbeddingND, create_token_ids  # noqa: E402

W = ckc._wrap_for_dlpack


def timed(fn, iters=20):
    fn(); torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e3


class _Attn:  # what _attn_ropeq reads from the decoder's Attention module
    dim_head = 64

    def __init__(self):
        self.qk_norm_scale = torch.ones(64, device="cuda", dtype=torch.float16)
        self.norm_q = type("n", (), {"eps": 1e-5})()


def make(B, T, Hh, Ww, H=32, seed=0, k_offset=3.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    S = T * Hh * Ww + 5
    qkv = torch.randn(B, S, H, 192, device="cuda", generator=g) * 2.0
    qkv[..., 64:128] += torch.randn(1, 1, H, 64, device="cuda", generator=g) * k_offset   # per-head K offset -> anchors
    qkv[..., :8] *= 6                                                                     # a few large channels
    qkv = qkv.half().reshape(B, S, H * 192)
    ids = create_token_ids((T, Hh, Ww), "cuda", torch.float16).expand(B, -1, -1)
    ids = torch.cat([ids, torch.zeros(B, 5, 3, device="cuda", dtype=torch.float16)], dim=1)
    rope = RotaryEmbeddingND(48, 100.0, 3).cuda()(ids)                                    # [B, S, 1, 24, 2, 2] fp16
    return qkv.view(B, S, H, 192), rope


def ck_ref(qkv, rope, attn):
    """The level-3 path: rms+rope in place, then ck int8_attention (also returns its Q/K quantization)."""
    qkv = qkv.clone()
    query, key, value = torch.chunk(qkv, 3, dim=-1)
    query, key = comfy.quant_ops.ck.rms_rope_split_half_(query, key, rope, attn.qk_norm_scale, epsilon=attn.norm_q.eps,
                                                         rot_dim=rope.shape[-3] * 2)
    q, k, v = (t.transpose(1, 2) for t in (query, key, value))
    out = comfy.quant_ops.ck.int8_attention(q, k, v)
    B, H, S, D = q.shape
    q8 = torch.empty(q.shape, dtype=torch.int8, device="cuda"); k8 = torch.empty(k.shape, dtype=torch.int8, device="cuda")
    qs = torch.empty(B, H, (S + 127) // 128 * 32, device="cuda"); ks = torch.empty(B, H, (S + 63) // 64 * 4, device="cuda")
    anchor = torch.empty(B, H, dtype=torch.int32, device="cuda")
    ckc._C._quant_qk_per_thread_int8(W(q), W(q8), W(qs), W(k), W(k8), W(ks), 128, 32, 64, 64, 1,
                                    torch.cuda.current_stream().cuda_stream, anchor.data_ptr())
    return out, q8, qs, k8, ks, anchor


def ck_time(qkv, rope, attn):
    query, key, value = torch.chunk(qkv, 3, dim=-1)
    query, key = comfy.quant_ops.ck.rms_rope_split_half_(query, key, rope, attn.qk_norm_scale, epsilon=attn.norm_q.eps,
                                                         rot_dim=rope.shape[-3] * 2)
    return comfy.quant_ops.ck.int8_attention(*(t.transpose(1, 2) for t in (query, key, value)))


ok_all = True
with torch.inference_mode():
    attn = _Attn()
    for (B, T, Hh, Ww) in [(4, 7, 16, 16), (28, 7, 16, 16), (2, 1, 16, 16), (3, 1, 8, 16), (2, 3, 16, 16)]:
        for seed, koff in ((0, 3.0), (1, 0.0)):
            qkv, rope = make(B, T, Hh, Ww, seed=seed, k_offset=koff)
            S = qkv.shape[1]
            ref, rq8, rqs, rk8, rks, ranchor = ck_ref(qkv, rope, attn)
            scale = attn.qk_norm_scale.float()
            q8, qs, k8, ks, anchor = vae_fast.ext().vae_qk_rope_quant(qkv, rope, scale, attn.norm_q.eps)
            out = vae_fast._attn_ropeq(attn, qkv, rope)
            res = {"q8": torch.equal(q8, rq8), "qs": torch.equal(qs, rqs), "k8": torch.equal(k8, rk8),
                   "ks": torch.equal(ks, rks), "anchor": torch.equal(anchor, ranchor), "out": torch.equal(out, ref)}
            ok = all(res.values())
            ok_all &= ok
            extra = ""
            if not ok:
                extra = (f" q8 diff {(q8 != rq8).sum().item()} k8 diff {(k8 != rk8).sum().item()} "
                         f"out maxabs {(out.float() - ref.float()).abs().max().item():.3g} anchors {anchor[0, :6].tolist()} "
                         f"vs {ranchor[0, :6].tolist()}")
            print(f"B={B:2d} S={S:5d} seed={seed} centered heads {(anchor >= 0).float().mean().item():.2f}: "
                  f"{'BIT-IDENTICAL' if ok else 'MISMATCH'} {res if not ok else ''}{extra}", flush=True)
            if seed == 0 and B == 28:
                t_ck = timed(lambda: ck_time(qkv.clone(), rope, attn), 10)
                t_clone = timed(lambda: qkv.clone(), 10)
                t_new = timed(lambda: vae_fast._attn_ropeq(attn, qkv, rope), 10)
                t_k = timed(lambda: vae_fast.ext().vae_qk_rope_quant(qkv, rope, scale, attn.norm_q.eps), 10)
                print(f"  time per call: ck rope+int8_attention {t_ck - t_clone:.2f} ms, fused {t_new:.2f} ms "
                      f"(of which rope+QK quant {t_k:.2f} ms)", flush=True)
print("ALL BIT-IDENTICAL" if ok_all else "SOME MISMATCH")
