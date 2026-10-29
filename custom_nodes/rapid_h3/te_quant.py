"""Faster Qwen3-VL-32B text(+image) encode for MiniMax-H3 (ts_h2 / ts_s6
use te_quant="int8_rot" since 2026-10-26).

ComfyUI runs the int8_convrot text encoder with full_precision_mm (sd1_clip: mixed_precision_ops(...,
full_precision_mm=True)) and an fp32 hidden stream (SDClipModel passes dtype=float32), the "stock" path below: every encode dequantizes all
350 LM weights to fp32 (inverse ConvRot included), runs fp32 GEMMs and fp32 attention, ≈2.9-4.3 s per clip on GB10
for only ≈0.6-1.2k tokens.

RapidH3TextEncoderQuant(clip, mode) returns a clone of the CLIP whose encodes run the 50 LM layers' linears and
attention differently; the residual stream, RMSNorms, RoPE, DeepStack injection and the vision tower stay as
ComfyUI runs them (fp32). Only while that clone encodes: the CLIPLoader's own CLIP (and every graph without the
node) is unchanged.

  int8_rot        W8A8: Comfy's int8_convrot weights adopted as they are (no copy), the activations rotated (H256)
                  and quantized per token (Triton), CUTLASS INT8 GEMM, as RapidH3LinearQuant(int8_rot) does for the DiT;
                  rows whose rotated amax / rms exceeds outlier_ratio (default 8: ≈6% of rows, mostly vision tokens
                  with massive activations) get a second INT8 pass on their quantization residual. bf16 attention
  int8_rot_plain  the same without the residual pass (fastest; ≈4x farther from bf16 than stock on text tokens, up
                  to 0.5 relative error on vision tokens: harness/te_compare.py)
  bf16_w8         weight-only: the stored (rotated) int8 weights scaled to bf16 per call, the activations rotated in
                  fp32 and rounded to bf16, bf16 GEMM; bf16 attention (no activation quantization)
  default         unchanged (control)

Measured 2026-10-26 (harness/te_compare.py, fast_test variants ts_h2_te8 / ts_s6_te8): in-server encode
ts_h2 2.87 -> 1.2-1.37 s, ts_s6 4.47 -> 2.06 s (standalone, weights resident: 2.75 -> 0.92-1.0 s, 4.19 -> 1.80 s).
int8_rot conditioning vs stock: cos >= 0.9991, rel 0.7-4.2%; vs the bf16 checkpoint (fp32 compute) text tokens 0.84%
(stock 0.40%), whole tensor about as far as stock (draft-size vision tokens are ≈15% from bf16 on every path).
"""
import contextlib

import torch
import torch.nn.functional as F

import comfy.ops
import comfy.sd
import comfy.text_encoders.llama as _llama

from . import kernels

_ROWS = [0, 0]  # rows given the residual pass, all rows (statistics for harness/te_compare.py)
_PATHS = {}  # linear calls per path in the current encode (logged once per encode)
LM_LINEARS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
              "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def _int8_rot_weight(w):
    """kernels.QWeight from a Comfy int8_convrot QuantizedTensor (H256, per-row scale), or None if it is not one."""
    p = getattr(w, "_params", None)
    if (p is None or not hasattr(w, "_qdata") or not getattr(p, "convrot", False)
            or getattr(p, "convrot_groupsize", 0) != kernels._tq.ROT_GROUP or getattr(p, "transposed", False)
            or w._qdata.dtype != torch.int8):
        return None
    return kernels.QWeight(w._qdata, p.scale.float().reshape(1, -1))


class _TELinear:
    """Replacement forward for one LM linear (fp32 in, fp32 out, like the module it replaces)."""

    def __init__(self, module, mode, outlier_ratio):
        self.module, self.mode, self.outlier_ratio = module, mode, outlier_ratio

    def __call__(self, x):
        mod = self.module
        ctx = None
        # Never ComfyUI's default cast for an int8 weight: it dequantizes to the input dtype (fp32). Asking for the
        # weight's own dtype with want_requant (as MixedPrecisionOps' weight-only path does) returns the
        # QuantizedTensor as is: resident, or staged through ComfyUI's dynamic-VRAM buffers when offloaded
        if len(getattr(mod, "weight_function", [])) > 0 or len(getattr(mod, "bias_function", [])) > 0:
            ctx = comfy.ops.CastBiasWeightContext(mod, x, offloadable=False)  # LoRA etc.: ComfyUI's numerics
            path = "patched"
        elif hasattr(mod, "_v") or mod.weight.device != x.device:
            ctx = comfy.ops.CastBiasWeightContext(mod, None, dtype=mod.weight.dtype, device=x.device,
                                                  bias_dtype=x.dtype, offloadable=True, want_requant=True)
            path = "cast"
        if ctx is not None:
            w, b = ctx.__enter__()
        else:
            w, b, path = mod.weight, mod.bias, "resident"
        try:
            qw = _int8_rot_weight(w)
            if qw is None or x.device.type != "cuda":  # not an int8_convrot weight: ComfyUI's numerics
                path += "_fallback"
                _PATHS[path] = _PATHS.get(path, 0) + 1
                wd = w.dequantize() if hasattr(w, "dequantize") else w
                return F.linear(x, wd.to(x.dtype), None if b is None else b.to(x.dtype))
            _PATHS[path] = _PATHS.get(path, 0) + 1
            shape = x.shape
            x2 = x.reshape(-1, shape[-1])
            if self.mode == "int8_rot_plain":
                y = kernels.linear(x2.to(torch.bfloat16).contiguous(), qw, "int8_rot")
            elif self.mode == "int8_rot":  # + a second INT8 pass on the quantization residual of outlier rows
                q1, s1 = kernels._tq.rot_quant_rows_i8(x2.to(torch.bfloat16).contiguous())
                s1 = s1.reshape(-1, 1)
                y = kernels._cf.int8_gemm(q1, s1, qw.q, qw.scale, config=6)
                # the rotation keeps the row norm: amax / rms of the rotated row from the scale and the input
                ratio = 127.0 * s1[:, 0] / x2.float().pow(2).mean(-1).sqrt().clamp_min(1e-12)
                rows = (ratio > self.outlier_ratio).nonzero()[:, 0]
                if rows.numel():
                    r = kernels._tq.convrot(x2[rows].float()) - q1[rows].float() * s1[rows]
                    s2 = (r.abs().amax(-1, keepdim=True) / 127.0).clamp_min(1e-12)
                    q2 = (r / s2).round().clamp(-127, 127).to(torch.int8)
                    y = y.float()
                    y[rows] += kernels._cf.int8_gemm(q2, s2, qw.q, qw.scale, config=6).float()
                    _ROWS[0] += rows.numel()
                _ROWS[1] += x2.shape[0]
            else:  # bf16_w8: y = rot(x) @ (q * s)^T, the rotation cancels against the stored rotated weight
                xr = kernels._tq.convrot(x2.float()).to(torch.bfloat16)
                y = F.linear(xr, qw.q.to(torch.bfloat16) * qw.scale.t().to(torch.bfloat16))
            y = y.to(x.dtype)
            if b is not None:
                y = y + b.to(y.dtype)
            return y.reshape(*shape[:-1], y.shape[-1])
        finally:
            if ctx is not None:
                ctx.__exit__(None, None, None)


def _bf16_attention(fn):
    def attn(q, k, v, heads, mask=None, **kwargs):
        dt = q.dtype
        if dt == torch.bfloat16:
            return fn(q, k, v, heads, mask=mask, **kwargs)
        m = mask.to(torch.bfloat16) if mask is not None and mask.is_floating_point() else mask
        return fn(q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16), heads, mask=m, **kwargs).to(dt)
    return attn


def _lm(cond_stage_model):
    """The Llama2_ (decoder stack) inside a MiniMax / Qwen3-VL TE model."""
    for m in cond_stage_model.modules():
        if isinstance(m, _llama.Llama2_):
            return m
    raise ValueError("RapidH3TextEncoderQuant: no Llama2_ decoder in this CLIP")


@contextlib.contextmanager
def patched(cond_stage_model, mode, outlier_ratio=8.0):
    if mode == "default":
        yield
        return
    lm = _lm(cond_stage_model)
    mods = [layer.get_submodule(n) for layer in lm.layers for n in LM_LINEARS]
    saved = [(m, m.__dict__.get("forward")) for m in mods]
    orig_attn = _llama.optimized_attention_for_device
    _PATHS.clear()
    try:
        for m in mods:
            m.forward = _TELinear(m, mode, outlier_ratio)
        _llama.optimized_attention_for_device = lambda *a, **k: _bf16_attention(orig_attn(*a, **k))
        yield
    finally:
        _llama.optimized_attention_for_device = orig_attn
        print(f"[rapid-h3] te_quant {mode}: linear calls {dict(_PATHS)}", flush=True)
        for m, prev in saved:
            if prev is None:
                m.__dict__.pop("forward", None)
            else:
                m.forward = prev


class RapidH3TextEncoderQuant:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clip": ("CLIP",),
                             "mode": (["int8_rot", "int8_rot_plain", "bf16_w8", "default"], {"default": "int8_rot"})},
                "optional": {"outlier_ratio": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 1e9, "step": 0.5})}}

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "patch"
    CATEGORY = "Rapid-H3"

    def patch(self, clip, mode, outlier_ratio=8.0):
        if mode == "default":
            return (clip,)
        c = clip.clone()

        def encode_from_tokens(*args, **kwargs):
            with patched(c.cond_stage_model, mode, outlier_ratio):
                return comfy.sd.CLIP.encode_from_tokens(c, *args, **kwargs)

        c.encode_from_tokens = encode_from_tokens  # encode_from_tokens_scheduled calls self.encode_from_tokens
        return (c,)


NODE_CLASS_MAPPINGS = {"RapidH3TextEncoderQuant": RapidH3TextEncoderQuant}
