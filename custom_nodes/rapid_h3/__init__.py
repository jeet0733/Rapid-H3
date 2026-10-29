"""Rapid-H3 research nodes for ComfyUI.

RapidH3LinearQuant swaps the forward of the 200 block linears of a MiniMax-H3 DiT
(blocks.N.attn.{qkv_proj,out_proj}, blocks.N.mlp.{fc1,fc2}) for a low-precision path,
only while the diffusion model runs.  Weights are quantized lazily on first use, after
ComfyUI has merged any LoRA, and cached per (weight pointer, version).

Modes
  bf16           untouched (control)
  fp8_tensor     FP8 e4m3, one scale per activation tensor and one per weight
  fp8_rowwise    FP8 e4m3, per-token activation scale x per-output-channel weight scale
                 (fast path: Triton quantizer + CUTLASS sm_121 GEMM with scales in the epilogue)
  fp8_rowwise_torch  same numerics through plain PyTorch ops (slow; for verification)
  int8_rowwise   INT8, per-token x per-channel (no rotation), plain PyTorch
  int8_fast      same numerics, Triton quantizer + CUTLASS INT8 GEMM
  int8_rot       INT8 after a block-256 Hadamard rotation of K on weights and activations
                 (ConvRot-style outlier spreading), CUTLASS INT8 GEMM

With measure_every > 0, every Nth call of each linear also runs the bf16 reference on
the same input and records ||y - y_ref|| / ||y_ref|| into RAPIDH3_STATS (json).
"""
import json
import os
import time
import re
import threading

import torch
import torch.nn.functional as F

import comfy.ops
import comfy.quant_ops
import comfy.patcher_extension

from . import kernels
from . import vae_fast
from .decode_save import RapidH3DecodeSaveVideo
from . import av_inputs
from . import encode_once
from . import trace
from . import upscale_lean
from . import te_quant
from . import sparse_attn

trace.install()  # opt-in: RAPIDH3_TRACE=path (wall-time attribution, harness/time_split.py)

# Faster INT8 video-VAE decoder blocks (vae_fast.py). RAPIDH3_VAE_FAST: 0 off; 1 bit-identical block
# kernels; 2 adds whole-chunk tile batching (all tiles of a temporal chunk per decoder call, bit-identical);
# 3 (default 2026-10-26, kept after a side-by-side review) adds SwiGLU in the w1 GEMM epilogue
# (h rounded to fp16 once, ~61 dB vs level 1, same distance to the fp16 VAE). 4 (default since 2026-10-27) adds the
# attention's q/k rms+rope fused into its Q/K quantizer: bit-identical to level 3, ≈1.0 s less per 141-frame decode.
_VAE_FAST = os.environ.get("RAPIDH3_VAE_FAST", "4")
if _VAE_FAST != "0":
    try:
        vae_fast.install()
        if _VAE_FAST in ("2", "3", "4"):
            vae_fast.install_tile_batch(int(os.environ.get("RAPIDH3_VAE_TILES", "28")))
        if _VAE_FAST in ("3", "4"):
            vae_fast.install_swiglu(True)
        if _VAE_FAST == "4":
            vae_fast.install_ropeq(True)
    except Exception as e:  # noqa: BLE001
        print(f"[rapid-h3] vae_fast not installed: {e}")

NAME_RE = re.compile(r"^blocks\.(\d+)\.(attn|mlp)\.(qkv_proj|out_proj|fc1|fc2)$")

_stats_lock = threading.Lock()
_SHADOW_REF = [False]   # True while RapidH3Shadow runs its bf16 reference pass
_APPROX_OFF = [False]   # True while RapidH3Shadow runs its unpatched (int8, no approximation) pass


def _parse_blocks(spec):
    out = set()
    for part in filter(None, (p.strip() for p in spec.split(","))):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


class _QuantLinear:
    """Replacement forward bound to one Linear.

    With ComfyUI's dynamic VRAM the bf16 weight may live off-GPU and only be materialized
    (with LoRA applied) inside a cast context per call, so the quantized copy is cached per
    patch set (ModelPatcher.patches_uuid) instead of per weight pointer.
    """

    def __init__(self, module, name, mode, measure_every, stats, patcher):
        self.module, self.name, self.mode = module, name, mode
        self.measure_every, self.stats, self.patcher = measure_every, stats, patcher
        self.cache_key, self.qweight = None, None
        self.calls = 0

    def _bf16_weight(self, x):
        """(weight, bias) on x's device with any LoRA patches applied, as ComfyUI computes them."""
        mod = self.module
        if (getattr(mod, "comfy_cast_weights", False) or len(getattr(mod, "weight_function", [])) > 0
                or mod.weight.device != x.device):
            ctx = comfy.ops.CastBiasWeightContext(mod, x, offloadable=False)
            w, b = ctx.__enter__()
            return w, b, ctx
        return mod.weight, mod.bias, None

    def _make_qweight(self, w):
        """Quantize a bf16 weight, or adopt an already-quantized Comfy int8_convrot (H256, per-row)
        weight as is, so stock int8_convrot checkpoints run without any bf16 copy."""
        p = getattr(w, "_params", None)
        if (p is not None and hasattr(w, "_qdata") and self.mode == "int8_rot"
                and getattr(p, "convrot", False) and getattr(p, "convrot_groupsize", 0) == kernels._tq.ROT_GROUP
                and not getattr(p, "transposed", False) and w._qdata.dtype == torch.int8):
            return kernels.QWeight(w._qdata.contiguous(), p.scale.float().reshape(1, -1).contiguous())
        if p is not None and hasattr(w, "dequantize"):
            w = w.dequantize()
        return kernels.quantize_weight(w.detach(), self.mode)

    def qweight_for(self, x):
        """The cached quantized weight for this patch set (quantizing it on first use)."""
        key = (self.patcher.patches_uuid, str(x.device))
        if key != self.cache_key:
            w, b, ctx = self._bf16_weight(x)
            try:
                self.qweight = self._make_qweight(w)
                self.cache_key = key
            finally:
                if ctx is not None:
                    ctx.__exit__(None, None, None)
        return self.qweight

    def __call__(self, x):
        if _SHADOW_REF[0]:
            w, b, ctx = self._bf16_weight(x)
            try:
                return F.linear(x, w.to(x.dtype), b)
            finally:
                if ctx is not None:
                    ctx.__exit__(None, None, None)
        key = (self.patcher.patches_uuid, str(x.device))
        need_ref = self.measure_every and (self.calls + 1) % self.measure_every == 0
        w = b = ctx = None
        if key != self.cache_key or need_ref or self.module.bias is not None:
            w, b, ctx = self._bf16_weight(x)
        try:
            if x.device.type != "cuda" or x.dtype not in (torch.bfloat16, torch.float16):
                return F.linear(x, w.to(x.dtype), b)
            if key != self.cache_key:
                self.qweight = self._make_qweight(w)
                self.cache_key = key
            shape = x.shape
            y = kernels.linear(x.reshape(-1, shape[-1]), self.qweight, self.mode)
            if getattr(self.qweight, "swiglu_perm", False):  # fc1 weight held in SwiGLU-epilogue order
                y = y[:, torch.argsort(kernels._cf.swiglu_perm(y.shape[-1], y.device))]
            if b is not None:
                y = y + b.to(y.dtype)
            y = y.reshape(*shape[:-1], y.shape[-1])
            self.calls += 1
            if need_ref:
                ref = F.linear(x, w.to(x.dtype), b)
                err = ((y.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-12)).item()
                with _stats_lock:
                    self.stats.setdefault(self.name, []).append(err)
            return y
        finally:
            if ctx is not None:
                ctx.__exit__(None, None, None)


def _fused_mlp(mlp, fc2q):
    orig = type(mlp).forward

    def forward(x):
        if _SHADOW_REF[0] or x.dtype != torch.bfloat16 or x.device.type != "cuda":
            return orig(mlp, x)
        hdn = mlp.fc1(x)
        shape = hdn.shape
        qw = fc2q.qweight_for(x)
        xq, sa = kernels._tq.swiglu_rot_quant_rows_i8(hdn.reshape(-1, shape[-1]).contiguous())
        y = kernels._cf.int8_gemm(xq, sa, qw.q, qw.scale,
                                  config=kernels.gemm_config(xq.shape[1], qw.q.shape[0]))
        return y.reshape(*shape[:-1], y.shape[-1])
    return forward


_IDX_CACHE = {}


def _row_index(segments, s, device):
    """Per-token modulation-row index (int32 [S]) from [(start, stop, row-or-LongTensor)]."""
    key = (id(segments), s, str(device))
    hit = _IDX_CACHE.get(key)
    if hit is not None and hit[0] is segments:
        return hit[1]
    idx = torch.empty(s, dtype=torch.int32, device=device)
    for a, b, row in segments:
        idx[a:b] = row.to(device=device, dtype=torch.int32) if torch.is_tensor(row) else int(row)
    _IDX_CACHE.clear()
    _IDX_CACHE[key] = (segments, idx)
    return idx


GATED_GEMM_CONFIG = 6  # 128x256x64 tile, 4 stages (bench/gemm_tune.py)
SWIGLU_GEMM_CONFIG = 5  # 256x128x64, 4 stages (bench/swiglu_epi_test.py)


def _fused_block(block, qkv_q, out_q, fc1_q, fc2_q):
    """DiTBlock.forward with norm+modulation fused into the INT8 quantizers (int8_rot recipe).

    Same graph as the stock block: rms_norm -> (1+scale)*h+shift -> qkv -> rms/rope -> attention
    -> out_proj -> gated residual, then the MLP likewise; only the activations entering the
    linears are never materialized in bf16."""
    import comfy.model_management as mm
    from comfy.ldm.minimax import model as h3
    from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention
    orig = type(block).forward
    tq, cf = kernels._tq, kernels._cf

    def norm_w(norm, x):
        w = getattr(norm, "weight", None)
        return None if w is None else mm.cast_to(w, dtype=torch.bfloat16, device=x.device).contiguous()

    def eps_of(norm):
        e = getattr(norm, "eps", None)
        return torch.finfo(torch.bfloat16).eps if e is None else e

    def gemm(xq, xs, q, x):
        qw = q.qweight_for(x)
        return cf.int8_gemm(xq, xs, qw.q, qw.scale, config=kernels.gemm_config(xq.shape[1], qw.q.shape[0]))

    def fc1_swiglu_weight(x):
        """fc1's INT8 weight with rows interleaved for the SwiGLU epilogue (cached in place of the
        plain copy; _QuantLinear.__call__ undoes the order if fc1 is ever called directly)."""
        qw = fc1_q.qweight_for(x)
        if not getattr(qw, "swiglu_perm", False):
            perm = cf.swiglu_perm(qw.q.shape[0], qw.q.device)
            qw = kernels.QWeight(qw.q[perm].contiguous(), qw.scale[:, perm].contiguous())
            qw.swiglu_perm = True
            fc1_q.qweight = qw
        return qw.q, qw.scale

    def gemm_gated(xq, xs, q, x, gate, segments):
        """x += gate * (xq @ W^T): ComfyUI's _mod_gate fused into the GEMM epilogue (bit-identical);
        one GEMM per modality segment, each with its own gate row."""
        qw = q.qweight_for(x)
        if any(torch.is_tensor(row) for _, _, row in segments):
            return h3._mod_gate(x, gate, gemm(xq, xs, q, x), segments)
        for a, b, row in segments:
            g = h3._mod_row(gate, row, x.dtype).reshape(-1).contiguous()
            cf.int8_gemm_gated_residual(xq[a:b], xs[a:b], qw.q, qw.scale, g, x[a:b], config=GATED_GEMM_CONFIG)
        return x

    def forward(x, t_emb, mod_segments, rope_freqs, transformer_options={}, attention=None):
        if (_SHADOW_REF[0] or attention is not None or rope_freqs is None
                or x.dtype != torch.bfloat16 or x.device.type != "cuda" or x.ndim != 2):
            return orig(block, x, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options,
                        attention=attention)
        tm = _TIMING is not None
        if tm:
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
            ev[0].record()
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(t_emb)
        if tm:
            ev[1].record()
        s = x.shape[0]
        idx = _row_index(mod_segments, s, x.device)
        a = block.attn
        xq, xs = tq.norm_mod_rot_quant_rows_i8(x, norm_w(block.norm1, x), eps_of(block.norm1), idx, shift_msa, scale_msa)
        q, k, v = gemm(xq, xs, qkv_q, x).split(a.heads * a.head_dim, dim=-1)
        v = v.view(s, a.heads, a.head_dim)
        q = q.view(1, s, a.heads, a.head_dim)
        k = k.view(1, s, a.heads, a.head_dim)
        qw = mm.cast_to(a.q_norm.weight, device=x.device)
        kw = mm.cast_to(a.k_norm.weight, device=x.device)
        # RapidH3Int8Attention config >= 100 tags its override; then q/k rms+rope runs inside our
        # attention pre-pass (bit-identical to the ck call below) instead of as a separate in-place pass
        rr_cfg = getattr(transformer_options.get("optimized_attention_override"), "rapidh3_rmsrope", None)
        if (rr_cfg is not None and s >= 1024 and a.head_dim == 128 and q.dtype == torch.bfloat16
                and rope_freqs.dtype == qw.dtype == kw.dtype == torch.bfloat16 and q.shape == k.shape):
            from . import attn_int8
            if tm:
                ev[2].record()
            o = torch.empty(1, a.heads, s, a.head_dim, dtype=x.dtype, device=x.device)
            tb = transformer_options.get("rapidh3_text_bias")  # RapidH3TextAttnBias
            if tb is not None:
                attn_int8.attention_rr_bias(q[0], k[0], v, rope_freqs, qw, kw, a.q_norm.eps,
                                            _text_rows(transformer_options), tb, config=rr_cfg,
                                            out=o[0].transpose(0, 1))
            else:
                attn_int8.attention_rr(q[0], k[0], v, rope_freqs, qw, kw, a.q_norm.eps, config=rr_cfg,
                                       out=o[0].transpose(0, 1))
            if getattr(transformer_options.get("optimized_attention_override"), "rapidh3_rrcheck", False):
                _rr_check(q, k, v, o, rope_freqs, qw, kw, a.q_norm.eps, rr_cfg, transformer_options)
        else:
            if transformer_options.get("rapidh3_text_bias") is not None:
                raise RuntimeError("RapidH3TextAttnBias: needs the fused INT8 attention path (RapidH3Int8Attention 190 or 194)")
            comfy.quant_ops.ck.rms_rope_split_half_(q, k, rope_freqs, qw, kw, epsilon=a.q_norm.eps,
                                                   rot_dim=rope_freqs.shape[-3] * 2)
            if tm:
                ev[2].record()
            q = AttentionTensorContainer(q[0].transpose(0, 1).unsqueeze(0))
            k = AttentionTensorContainer(k[0].transpose(0, 1).unsqueeze(0))
            v = AttentionTensorContainer(v.transpose(0, 1).unsqueeze(0))
            # head-major output [1, H, S, D]: the out_proj quantizer reads it in place (no transpose copy)
            o = optimized_attention(q, k, v, a.heads, preferred_attention=a.comfy_attention, mask=None,
                                    skip_reshape=True, skip_output_reshape=True, transformer_options=transformer_options)
        if tm:
            ev[3].record()
        oq, os_ = tq.rot_quant_heads_i8(o[0])
        del o
        x = gemm_gated(oq, os_, out_q, x, gate_msa, mod_segments)
        xq, xs = tq.norm_mod_rot_quant_rows_i8(x, norm_w(block.norm2, x), eps_of(block.norm2), idx, shift_mlp, scale_mlp)
        h = cf.int8_gemm_swiglu(xq, xs, *fc1_swiglu_weight(x), config=SWIGLU_GEMM_CONFIG)
        hq, hs = tq.rot_quant_rows_i8(h)
        del h
        out = gemm_gated(hq, hs, fc2_q, x, gate_mlp, mod_segments)
        if tm:
            ev[4].record()
            _TIMING.append(ev)
        return out
    return forward


def _text_rows(transformer_options):
    """Number of text rows at the start of H3's packed sequence (the model's PackedLayout, which _forward puts into
    transformer_options["minimax_h3_layout"]); the text span must be the first segment."""
    layout = transformer_options.get("minimax_h3_layout")
    seg = layout.segments[0] if layout is not None and layout.segments else None
    if seg is None or seg[2] != "text" or seg[0] != 0:
        raise RuntimeError("RapidH3TextAttnBias: no leading text segment in minimax_h3_layout")
    return seg[1]


def _rr_check(q, k, v, o, rope_freqs, qw, kw, eps, cfg, transformer_options):
    """RapidH3Int8Attention config 2xx: verify the fused rms+rope path on the real q/k/v of this call against the
    unfused one (ck rms_rope_split_half_ on copies, then attention config xx): output and every pre-pass operand
    torch.equal. One JSON line per call in $RAPIDH3_RRCHECK (default /data/output/rapidh3_rrcheck.jsonl)."""
    from . import attn_int8
    qc, kc = q.clone(), k.clone()
    comfy.quant_ops.ck.rms_rope_split_half_(qc, kc, rope_freqs, qw, kw, epsilon=eps, rot_dim=rope_freqs.shape[-3] * 2)
    ref = attn_int8.attention(qc[0], kc[0], v, config=cfg)
    pu = attn_int8.ext().int8_prepass(qc[0], kc[0], v, config=cfg)
    pf = attn_int8.ext().int8_prepass(q[0], k[0], v, rope_freqs, qw, kw, eps, config=cfg)
    rec = {"block": transformer_options.get("block_index"), "S": q.shape[1],
           "out_equal": torch.equal(ref, o[0].transpose(0, 1)),
           "operands_equal": all(torch.equal(x, y) for x, y in zip(pu, pf)),
           "max_abs_diff": float((ref.float() - o[0].transpose(0, 1).float()).abs().max())}
    with open(os.environ.get("RAPIDH3_RRCHECK", "/data/output/rapidh3_rrcheck.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


class RapidH3LinearQuant:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "mode": (["bf16", "fp8_tensor", "fp8_rowwise", "fp8_rowwise_torch", "int8_rowwise", "int8_fast", "int8_rot",
                      "nvfp4_fake", "nvfp4_rot_fake"], {"default": "fp8_rowwise"}),
            "targets": ("STRING", {"default": "qkv_proj,out_proj,fc1,fc2"}),
            "keep_bf16_blocks": ("STRING", {"default": ""}),
            "measure_every": ("INT", {"default": 0, "min": 0, "max": 1000}),
            "fuse_block": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Rapid-H3"

    def patch(self, model, mode, targets, keep_bf16_blocks, measure_every, fuse_block=True):
        if mode == "bf16":
            return (model,)
        m = model.clone()
        dm = m.model.diffusion_model
        want = {t.strip() for t in targets.split(",") if t.strip()}
        keep = _parse_blocks(keep_bf16_blocks)
        stats = {}
        wrappers = []
        by_name = {}
        for name, mod in dm.named_modules():
            hit = NAME_RE.match(name)
            if not hit or hit.group(3) not in want or int(hit.group(1)) in keep:
                continue
            q = _QuantLinear(mod, name, mode, measure_every, stats, m)
            by_name[name] = q
            wrappers.append((mod, q))
        if mode == "int8_rot" and not measure_every:
            # fused block where all four linears are quantized, else just the fused MLP
            for name, mod in dm.named_modules():
                if not re.match(r"^blocks\.\d+$", name):
                    continue
                parts = [by_name.get(f"{name}.{n}") for n in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")]
                if all(parts) and fuse_block:
                    wrappers.append((mod, _fused_block(mod, *parts)))
                elif parts[2] and parts[3]:
                    wrappers.append((mod.mlp, _fused_mlp(mod.mlp, parts[3])))
        stats_path = os.environ.get("RAPIDH3_STATS")

        def wrapper(executor, *args, **kwargs):
            saved = []
            for mod, q in wrappers:
                saved.append((mod, mod.__dict__.get("forward")))
                mod.forward = q
            try:
                return executor(*args, **kwargs)
            finally:
                for mod, prev in saved:
                    if prev is None:
                        mod.__dict__.pop("forward", None)
                    else:
                        mod.forward = prev
                if stats_path and stats:
                    with _stats_lock:
                        json.dump({"mode": mode, "errors": stats}, open(stats_path, "w"))

        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "rapidh3_linear_quant", wrapper)
        print(f"[rapid-h3] {mode}: {len(wrappers)} linears patched", flush=True)
        return (m,)


class RapidH3Shadow:
    """Single-step divergence: every denoising step runs the model twice on the same input,
    once as configured (Rapid-H3 quantized linears, chosen attention backend) and once as the
    bf16 / PyTorch-attention reference, and records ||v - v_ref|| / ||v_ref|| for the video
    and audio velocities.  follow=reference keeps the trajectory on the reference so every
    variant is measured on identical states.  Stats go to $RAPIDH3_SHADOW (json).

    reference (optional, default bf16 = the behaviour above):
      int8_unpatched  the reference is the same int8 model with the approximate-compute patches off
                      (RapidH3FrameSkip / RapidH3AsymRnR / any patches_replace such as BlockSparseAttention),
                      so "video"/"audio" is the error the patch adds on top of int8; both passes are timed
                      (synced, t_cand_ms / t_mid_ms).
      both            also the bf16 pass: int8_video/int8_audio = unpatched int8 vs bf16, total_* = candidate
                      vs bf16.
    mid_attention: attention for the unpatched pass; "comfy kitchen attention" replaces an attention override
    installed by a sparse node (BlockSparseAttention keeps its own on top) with the dense ck int8 backend.
    follow=unpatched keeps the trajectory on the unpatched int8 model (the real draft)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",),
                             "follow": (["reference", "candidate", "unpatched"], {"default": "reference"}),
                             "candidate_sdpa": (["default", "cudnn", "efficient", "perturb_ulp"], {"default": "default"})},
                "optional": {"reference": (["bf16", "int8_unpatched", "both"], {"default": "bf16"}),
                             "mid_attention": (["keep", "comfy kitchen attention"], {"default": "keep"})}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Rapid-H3"

    def patch(self, model, follow, candidate_sdpa="default", reference="bf16", mid_attention="keep"):
        if reference != "bf16":
            return (self._patch_mid(model, follow, reference, mid_attention),)
        from torch.nn.attention import SDPBackend, sdpa_kernel
        backend = {"cudnn": SDPBackend.CUDNN_ATTENTION, "efficient": SDPBackend.EFFICIENT_ATTENTION}.get(candidate_sdpa)
        m = model.clone()
        steps = []
        path = os.environ.get("RAPIDH3_SHADOW")

        def rel(a, b):
            return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()

        def wrapper(executor, x, timestep, context, transformer_options, *args, **kwargs):
            if candidate_sdpa == "perturb_ulp":
                # noise floor: bf16 model on inputs jittered by ~one bf16 ulp (2^-8 relative)
                g = torch.Generator(device=x[0].device).manual_seed(1234)
                xp = [t * (1 + 2.0 ** -8 * torch.randn(t.shape, generator=g, device=t.device, dtype=t.dtype)) for t in x]
                cand = executor(xp, timestep, context, transformer_options, *args, **kwargs)
            elif backend is not None:   # noise floor: same math, different attention kernel
                with sdpa_kernel(backend):
                    cand = executor(x, timestep, context, transformer_options, *args, **kwargs)
            else:
                cand = executor(x, timestep, context, transformer_options, *args, **kwargs)
            cand = [c.clone() for c in cand]
            # the reference is the stock bf16 model: no attention override, no block replacements
            # (e.g. ComfyUI's BlockSparseAttention node patches every DiT block through patches_replace)
            ref_opts = {k: v for k, v in transformer_options.items()
                        if k not in ("optimized_attention_override", "patches_replace")}
            _SHADOW_REF[0] = True
            try:
                ref = executor(x, timestep, context, ref_opts, *args, **kwargs)
            finally:
                _SHADOW_REF[0] = False
            steps.append({"sigma": float(timestep.flatten()[0]) / 1000.0,
                          "video": rel(cand[0], ref[0]), "audio": rel(cand[1], ref[1])})
            if path:
                json.dump({"steps": steps}, open(path, "w"))
            return ref if follow == "reference" else cand

        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "rapidh3_shadow", wrapper)
        return (m,)

    @staticmethod
    def _patch_mid(model, follow, reference, mid_attention):
        m = model.clone()
        steps = []
        path = os.environ.get("RAPIDH3_SHADOW")
        ck_override = None
        if mid_attention == "comfy kitchen attention":
            from comfy.ldm.modules import attention as cattn
            fn = cattn.get_attention_function("comfy_kitchen_int8")

            def ck_override(_, *a, **kw):
                return fn(*a, **kw)
            if getattr(fn, "container_function", None) is not None:
                ck_override.container_function = fn.container_function

        def rel(a, b):
            return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()

        def timed(fn):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = [c.clone() for c in fn()]
            torch.cuda.synchronize()
            return out, (time.perf_counter() - t0) * 1e3

        def wrapper(executor, x, timestep, context, transformer_options, *args, **kwargs):
            sigma = float(timestep.flatten()[0]) / 1000.0
            ss = transformer_options.get("sample_sigmas")
            if ss is not None and abs(float(ss.flatten()[0]) - sigma) < 1e-6:
                steps.clear()   # first step of a new sampling run (ComfyUI may reuse this node's clone)
            cand, t_cand = timed(lambda: executor(x, timestep, context, transformer_options, *args, **kwargs))
            mid_opts = {k: v for k, v in transformer_options.items() if k != "patches_replace"}
            if ck_override is not None:
                mid_opts["optimized_attention_override"] = ck_override
            _APPROX_OFF[0] = True
            try:
                mid, t_mid = timed(lambda: executor(x, timestep, context, mid_opts, *args, **kwargs))
            finally:
                _APPROX_OFF[0] = False
            rec = {"sigma": sigma, "video": rel(cand[0], mid[0]), "audio": rel(cand[1], mid[1]),
                   "t_cand_ms": round(t_cand, 1), "t_mid_ms": round(t_mid, 1)}
            ref = None
            if reference == "both":
                ref_opts = {k: v for k, v in transformer_options.items()
                            if k not in ("optimized_attention_override", "patches_replace")}
                _SHADOW_REF[0] = _APPROX_OFF[0] = True
                try:
                    ref = executor(x, timestep, context, ref_opts, *args, **kwargs)
                finally:
                    _SHADOW_REF[0] = _APPROX_OFF[0] = False
                rec.update(int8_video=rel(mid[0], ref[0]), int8_audio=rel(mid[1], ref[1]),
                           total_video=rel(cand[0], ref[0]), total_audio=rel(cand[1], ref[1]))
            steps.append(rec)
            if path:
                json.dump({"reference": reference, "steps": steps}, open(path, "w"))
            if follow == "candidate":
                return cand
            if follow == "reference" and ref is not None:
                return ref
            return mid

        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "rapidh3_shadow", wrapper)
        return m


def _sigma_gate(m, min_sigma, key):
    """Per-step on/off state for an approximate-compute patch: on while the step's sigma >= min_sigma
    (the draft's steps 1-5 at shift 12 run at 1 .. 0.857, its last at 0.706, ts_h2's refine at 0.72)."""
    state = {"on": False}

    def wrapper(executor, x, timestep, context, transformer_options, *args, **kwargs):
        state["on"] = float(timestep.flatten()[0]) / 1000.0 >= min_sigma
        return executor(x, timestep, context, transformer_options, *args, **kwargs)

    m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, key, wrapper)
    return state


def _video_span(layout, s):
    """(va, vb, latent_t, rows per frame, frame h, frame w) of the target video segment, or None."""
    if layout is None or layout.seq_len != s:
        return None
    va, vb, _ = next(seg for seg in layout.segments if seg[2] == "video")
    _, t, lh, lw, _ = layout.signature
    fh, fw = lh // 2, lw // 2
    if (vb - va) != t * fh * fw:
        return None
    return va, vb, t, fh * fw, fh, fw


class RapidH3FrameSkip:
    """FIS-DiT-style frame-interleaved sparsity (arXiv 2605.11869; no code released, own reading):
    in the chosen DiT blocks only every `stride`-th latent frame of the target video is computed, at an offset
    that rotates with the block index (block i computes frames i % stride, i % stride + stride, ...). The block
    runs on [text | cond | audio | kept frames] (attention included, so kept frames only see kept frames); the
    skipped frames get the block's residual update linearly interpolated in time from the nearest computed
    frames (before the first / after the last computed frame: copied from it). Non-video rows take the block's
    output as computed. Active only while sigma >= min_sigma (default: the draft's steps 1-5). Opt-in, research."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",),
                             "stride": ("INT", {"default": 3, "min": 2, "max": 16}),
                             "blocks": ("STRING", {"default": "10-39"}),
                             "min_sigma": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.001})}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Rapid-H3"

    def patch(self, model, stride, blocks, min_sigma):
        m = model.clone()
        state = _sigma_gate(m, min_sigma, "rapidh3_frame_skip")
        plans = {}

        def plan(s, va, t, f, off, device):
            key = (s, va, t, f, off, str(device))
            p = plans.get(key)
            if p is None:
                sel = list(range(off, t, stride)) or [t - 1]
                skip = [j for j in range(t) if j not in sel]
                lo, hi, w = [], [], []
                for j in skip:
                    a = max((i for i, fr in enumerate(sel) if fr < j), default=None)
                    b = min((i for i, fr in enumerate(sel) if fr > j), default=None)
                    a = b if a is None else a
                    b = a if b is None else b
                    lo.append(a)
                    hi.append(b)
                    w.append(0.0 if a == b else (j - sel[a]) / (sel[b] - sel[a]))
                L = lambda v: torch.tensor(v, dtype=torch.long, device=device)  # noqa: E731
                sel_t = L(sel)
                keep = torch.cat([torch.arange(va, device=device),
                                  (va + sel_t[:, None] * f + torch.arange(f, device=device)).reshape(-1)])
                p = dict(sel=sel_t, skip=L(skip), lo=L(lo), hi=L(hi),
                         w=torch.tensor(w, dtype=torch.float32, device=device).view(-1, 1, 1), keep=keep)
                plans[key] = p
            return p

        def replace(args, extra):
            orig = extra["original_block"]
            x = args["img"]
            opts = args["transformer_options"]
            if not state["on"] or _APPROX_OFF[0] or _SHADOW_REF[0] or x.ndim != 2:
                return orig(args)
            span = _video_span(args.get("layout") or opts.get("minimax_h3_layout"), x.shape[0])
            if span is None or span[1] != x.shape[0]:
                return orig(args)
            va, vb, t, f, _, _ = span
            p = plan(x.shape[0], va, t, f, opts["block_index"] % stride, x.device)
            keep = p["keep"]
            nsel = p["sel"].shape[0]
            segs = []
            for a, b, row in args["mod_segments"]:
                if a == va:
                    b = va + nsel * f
                    if torch.is_tensor(row):
                        row = row[keep[va:] - va]
                segs.append((a, b, row))
            sub = {"img": x[keep], "t_emb": args["t_emb"], "mod_segments": segs,
                   "rope_freqs": args["rope_freqs"][:, keep], "transformer_options": opts}
            if args.get("attention") is not None:
                sub["attention"] = args["attention"]
            out = orig(sub)["img"]
            d = x.shape[1]
            xv = x[va:vb].view(t, f, d)
            outv = out[va:].view(nsel, f, d)
            if p["skip"].numel():
                delta = outv.float() - xv[p["sel"]].float()
                upd = delta[p["lo"]] * (1.0 - p["w"]) + delta[p["hi"]] * p["w"]
                xv[p["skip"]] = (xv[p["skip"]].float() + upd).to(x.dtype)
                del delta, upd
            xv[p["sel"]] = outv
            x[:va] = out[:va]
            return {"img": x}

        for i in sorted(_parse_blocks(blocks)):
            m.set_model_patch_replace(replace, "dit", "double_block", i)
        print(f"[rapid-h3] frame skip: stride {stride}, blocks {blocks}, sigma >= {min_sigma}", flush=True)
        return (m,)


class RapidH3AsymRnR:
    """AsymRnR-style asymmetric token reduction in attention (arXiv 2412.11706; own implementation): in the chosen
    blocks, target-video tokens that are most similar to their window's anchor token (window t,h,w in latent
    tokens; anchor = the window's first token; cosine similarity over a strided subset of heads, after rope) are
    dropped, separately for queries (q_ratio of the video tokens, by q similarity) and for keys/values (kv_ratio,
    by k similarity). Attention runs on the reduced sets through the attention backend that was installed before
    this node; a dropped query's output is restored by copying its anchor's output. Text, cond and audio tokens
    are never dropped. Active only while sigma >= min_sigma. Opt-in, research."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",),
                             "q_ratio": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 0.75, "step": 0.01}),
                             "kv_ratio": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.75, "step": 0.01}),
                             "window": ("STRING", {"default": "1,2,2"}),
                             "blocks": ("STRING", {"default": "5-44"}),
                             "min_sigma": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.001}),
                             "sim_heads": ("INT", {"default": 8, "min": 1, "max": 64})}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Rapid-H3"

    def patch(self, model, q_ratio, kv_ratio, window, blocks, min_sigma, sim_heads):
        m = model.clone()
        state = _sigma_gate(m, min_sigma, "rapidh3_asymrnr")
        want = _parse_blocks(blocks)
        wt, wh, ww = (int(v) for v in window.split(","))
        prev = m.model_options["transformer_options"].get("optimized_attention_override")
        anchors = {}

        def anchor_of(t, fh, fw, device):
            key = (t, fh, fw, str(device))
            a = anchors.get(key)
            if a is None:
                ti, hi, wi = torch.meshgrid(torch.arange(t), torch.arange(fh), torch.arange(fw), indexing="ij")
                a = (((ti - ti % wt) * fh + (hi - hi % wh)) * fw + (wi - wi % ww)).reshape(-1).to(device)
                anchors[key] = a
            return a

        def reduce(feat, va, vb, anchor, ratio):
            """(keep index into the full sequence, restore index full -> kept position)"""
            s = feat.shape[2]
            nh = feat.shape[1]
            hs = torch.arange(0, nh, max(1, nh // sim_heads), device=feat.device)[:sim_heads]
            f = feat[0, hs, va:vb].float()
            f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            sim = (f * f[:, anchor]).sum(-1).mean(0)
            del f
            is_src = anchor != torch.arange(anchor.shape[0], device=anchor.device)
            sim = torch.where(is_src, sim, torch.full_like(sim, -2.0))
            r = min(int(ratio * (vb - va)), int(is_src.sum()))
            drop = torch.topk(sim, r).indices
            keep_mask = torch.ones(s, dtype=torch.bool, device=feat.device)
            keep_mask[va + drop] = False
            keep = keep_mask.nonzero().squeeze(1)
            pos = torch.cumsum(keep_mask, 0) - 1
            restore = pos.clone()
            restore[va + drop] = pos[va + anchor[drop]]
            return keep, restore

        def override(func, q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False,
                     skip_output_reshape=False, **kwargs):
            def attend(q, k, v, skip_out):
                kw = dict(mask=mask, attn_precision=attn_precision, skip_reshape=skip_reshape,
                          skip_output_reshape=skip_out, **kwargs)
                return func(q, k, v, heads, **kw) if prev is None else prev(func, q, k, v, heads, **kw)

            opts = kwargs.get("transformer_options") or {}
            if (not state["on"] or _APPROX_OFF[0] or _SHADOW_REF[0] or mask is not None or not skip_reshape
                    or opts.get("block_index") not in want):
                return attend(q, k, v, skip_output_reshape)
            span = _video_span(opts.get("minimax_h3_layout"), q.shape[2])
            if span is None:
                return attend(q, k, v, skip_output_reshape)
            va, vb, t, _, fh, fw = span
            anchor = anchor_of(t, fh, fw, q.device)
            restore = None
            if kv_ratio > 0:
                keep_k, _ = reduce(k, va, vb, anchor, kv_ratio)
                k, v = k.index_select(2, keep_k), v.index_select(2, keep_k)
            if q_ratio > 0:
                keep_q, restore = reduce(q, va, vb, anchor, q_ratio)
                q = q.index_select(2, keep_q)
            out = attend(q, k, v, True)          # [B, H, S', D]
            if restore is not None:
                out = out.index_select(2, restore)
            if skip_output_reshape:
                return out
            b, h, s, dh = out.shape
            return out.transpose(1, 2).reshape(b, s, h * dh)

        m.model_options["transformer_options"]["optimized_attention_override"] = override
        print(f"[rapid-h3] AsymRnR: q {q_ratio}, kv {kv_ratio}, window {window}, blocks {blocks}, sigma >= {min_sigma}",
              flush=True)
        return (m,)


class RapidH3DumpQKV:
    """Save the real q, k, v ([1, H, S, D] bf16) seen by attention in chosen blocks at chosen
    sampler steps to $RAPIDH3_DUMP/<step>_<block>.pt, for offline attention-kernel benchmarks."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "blocks": ("STRING", {"default": "5,25,45"}),
                             "steps": ("STRING", {"default": "1,5"})}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Rapid-H3"

    def patch(self, model, blocks, steps):
        m = model.clone()
        want_b = _parse_blocks(blocks)
        want_s = _parse_blocks(steps)
        out_dir = os.environ.get("RAPIDH3_DUMP", "/data/output/qkv")
        os.makedirs(out_dir, exist_ok=True)
        state = {"step": -1}
        prev = m.model_options["transformer_options"].get("optimized_attention_override")

        def override(func, *args, **kwargs):
            opts = kwargs.get("transformer_options", {})
            blk = opts.get("block_index")
            if state["step"] in want_s and blk in want_b:
                q, k, v = args[:3]
                lay = opts.get("minimax_h3_layout")  # the packed sequence's (start, stop, kind) segments
                torch.save({"q": q.detach().cpu(), "k": k.detach().cpu(), "v": v.detach().cpu(),
                            "segments": list(getattr(lay, "segments", None) or [])},
                           os.path.join(out_dir, f"s{state['step']}_b{blk:02d}.pt"))
            return prev(func, *args, **kwargs) if prev else func(*args, **kwargs)

        def wrapper(executor, *args, **kwargs):
            state["step"] += 1
            return executor(*args, **kwargs)

        m.model_options["transformer_options"]["optimized_attention_override"] = override
        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "rapidh3_dump", wrapper)
        return (m,)


_TIMING = [] if os.environ.get("RAPIDH3_TIMING") else None


def _timing_flush(path, wall_ms):
    """Per-step GPU time by phase of the fused blocks (CUDA events, no profiler)."""
    torch.cuda.synchronize()
    names = ("adaln", "pre_attn", "attention", "post")
    tot = dict.fromkeys(names, 0.0)
    gaps = 0.0
    prev = None
    for ev in _TIMING:
        for i, n in enumerate(names):
            tot[n] += ev[i].elapsed_time(ev[i + 1])
        if prev is not None:
            gaps += prev[4].elapsed_time(ev[0])
        prev = ev
    rec = {k: round(v) for k, v in tot.items()}
    rec.update(blocks=len(_TIMING), between_blocks=round(gaps), wall=round(wall_ms))
    _TIMING.clear()
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _probe(path):
    """Time fixed synthetic kernels (attention, INT8 GEMM, copy) inside the server process, to tell
    an environment slowdown (clocks, memory mapping) from a model-data one. Appends one line."""
    import comfy_kitchen as ck
    g = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (torch.randn(1, 8, 37728, 128, device="cuda", dtype=torch.bfloat16, generator=g) for _ in range(3))
    a = torch.randint(-127, 127, (37728, 5376), device="cuda", dtype=torch.int8, generator=g)
    w = torch.randint(-127, 127, (21504, 5376), device="cuda", dtype=torch.int8, generator=g)
    xs = torch.ones(37728, 1, device="cuda"); ws = torch.ones(1, 21504, device="cuda")
    big = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
    res = {}
    for name, fn in (("attn8h", lambda: ck.int8_attention(q, k, v)),
                     ("gemm_qkv", lambda: kernels._cf.int8_gemm(a, xs, w, ws, config=kernels.gemm_config(5376, 21504))),
                     ("copy512M", lambda: big.clone())):
        fn()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(3):
            fn()
        e1.record()
        torch.cuda.synchronize()
        res[name] = round(e0.elapsed_time(e1) / 3, 2)
    del q, k, v, a, w, big
    with open(path, "a") as f:
        f.write(json.dumps(res) + "\n")


class RapidH3Profile:
    """torch.profiler over one denoising step; CUDA kernel time grouped into classes, written to
    $RAPIDH3_PROFILE (default /data/output/profile.txt)."""

    CLASSES = [("attention", ("attn", "fmha", "flash", "int8_attention", "sage")),
               ("gemm", ("gemm", "nvjet", "cutlass", "mma", "Kernel2")),
               ("quant/rotate", ("quant", "rot_", "swiglu_rot")),
               ("rope/norm", ("rope", "norm", "rms")),
               ("elementwise", ("elementwise", "vectorized", "addcmul", "copy", "mul", "add"))]

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "step": ("INT", {"default": 3, "min": 0, "max": 100})}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Rapid-H3"

    def patch(self, model, step):
        m = model.clone()
        state = {"i": -1}
        path = os.environ.get("RAPIDH3_PROFILE", "/data/output/profile.txt")

        def wrapper(executor, *args, **kwargs):
            # count per sampling run: ComfyUI may reuse this cached clone across prompts
            state["i"] += 1
            if _TIMING is not None:
                _TIMING.clear()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = executor(*args, **kwargs)
                torch.cuda.synchronize()
                _timing_flush(path + ".timing", (time.perf_counter() - t0) * 1e3)
                return out
            if os.environ.get("RAPIDH3_PROBE") and state["i"] % 8 == step:
                _probe(path + ".probe")
            if state["i"] % int(os.environ.get("RAPIDH3_PROFILE_PERIOD", "8")) != step:
                return executor(*args, **kwargs)
            from torch.profiler import ProfilerActivity, profile
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
                out = executor(*args, **kwargs)
                torch.cuda.synchronize()
            wall = (time.perf_counter() - t0) * 1e3
            tot, by = 0.0, {}
            rows = []
            cpu_rows = sorted(((e.self_cpu_time_total / 1e3, e.count, e.key[:90]) for e in prof.key_averages()),
                              reverse=True)[:25]
            for e in prof.key_averages():
                t = e.self_device_time_total / 1e3  # ms
                if t <= 0:
                    continue
                tot += t
                cls = next((c for c, keys in self.CLASSES if any(k in e.key for k in keys)), "other")
                by[cls] = by.get(cls, 0.0) + t
                rows.append((t, e.count, cls, e.key[:110]))
            with open(path, "w") as f:
                f.write(f"step {step}: wall {wall:.0f} ms, total CUDA kernel time {tot:.0f} ms\n")
                for c, t in sorted(by.items(), key=lambda kv: -kv[1]):
                    f.write(f"  {c:14s} {t:9.0f} ms  {100*t/tot:5.1f}%\n")
                f.write("top kernels:\n")
                for t, n, c, k in sorted(rows, reverse=True)[:25]:
                    f.write(f"  {t:9.1f} ms  n={n:5d}  [{c}] {k}\n")
                f.write("top CPU ops (self time):\n")
                for t, n, k in cpu_rows:
                    f.write(f"  {t:9.1f} ms  n={n:6d}  {k}\n")
            return out

        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "rapidh3_profile", wrapper)
        return (m,)


class RapidH3StepTimer:
    """Wall time of every denoising step with a device sync on both sides, appended as JSON lines to
    $RAPIDH3_STEPS. ComfyUI's tqdm s/it is not a step time: nothing in the sampler syncs, so the host
    runs ahead of the GPU and the bar under-reports by up to a step (≈−12% on an 8-step render)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Rapid-H3"

    def patch(self, model):
        m = model.clone()
        path = os.environ.get("RAPIDH3_STEPS", "/data/output/rapidh3_steps.jsonl")

        def wrapper(executor, *args, **kwargs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = executor(*args, **kwargs)
            torch.cuda.synchronize()
            with open(path, "a") as f:
                f.write(json.dumps({"t": round(time.time(), 3), "step_ms": round((time.perf_counter() - t0) * 1e3, 1)}) + "\n")
            return out

        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "rapidh3_step_timer", wrapper)
        return (m,)


class RapidH3Int8Attention:
    """Route self-attention (q, k, v of one shape, head_dim 128, no mask, S >= 1024, [1, H, S, D] skip_reshape
    layout) to our INT8 kernel (attn_int8.py, csrc/attn_int8.cu) with the given config; every other call goes to the
    attention chosen before (ModelAttentionBackend: comfy-kitchen INT8 in the presets). ts_h2 / ts_s6 use config 190
    (harness/workflow.py PRESETS, since 2026-10-27; 194 before, bit-identical).
    config 100 + c (e.g. 194): config c, and in the fused DiT block (RapidH3LinearQuant fuse_block) the
    q/k RMSNorm + RoPE also moves into this kernel's pre-pass (attn_int8.attention_rr, bit-identical output);
    200 + c: the same, plus a per-call check against the unfused path (_rr_check; slow, verification only).
    Kernel configs 94 / 90 / 89 / 79 (csrc/attn_cake/DESIGN.md): 90 is 94 with the softmax mutex entered after the
    row max (bit-identical, faster); 89 (lazy rescale) and 79 (f16x2 exps) are measured experiments.
    exact_rows (optional: Ref2VA voice references): comma-separated packed-layout segment kinds
    (MiniMax PackedLayout: text, cond, cond_audio, ref_img, ref_audio, audio, video) whose query rows get bf16 SDPA
    attention over all keys instead of the INT8 kernel's output (e.g. "ref_audio,audio": the reference voice and the
    generated audio track, ≈2% of the rows). It needs q/k after RMSNorm + RoPE, so the 1xx fusion is off then
    (the block runs ck's rms_rope pass, then this override). Measured on Ref2VA voice references (ref2, 2026-10-26):
    one-step audio divergence 8.5% -> 7.5-8.0% only, the rest is the checkpoint's sensitivity (its bf16 1-ulp floor
    is 4.8-5.4% audio vs 1.3% on FL2VA); +12-20% per step, +160% with a reference video ("ref_img" rows): not a
    default."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "config": ("INT", {"default": 80, "min": 0, "max": 299})},
                "optional": {"exact_rows": ("STRING", {"default": ""})}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Rapid-H3"

    def patch(self, model, config, exact_rows=""):
        from . import attn_int8
        m = model.clone()
        prev = m.model_options["transformer_options"].get("optimized_attention_override")
        exact = {s.strip() for s in exact_rows.split(",") if s.strip()}
        rmsrope, rrcheck = config >= 100 and not exact, config >= 200
        config = config % 100
        spans_cache = {}

        def exact_spans(opts, s):
            """merged [a, b) row spans of the exact_rows segment kinds in this call's layout (cached per layout)."""
            lay = opts.get("minimax_h3_layout")
            if lay is None or getattr(lay, "seq_len", None) != s:
                return ()
            hit = spans_cache.get(id(lay))
            if hit is not None and hit[0] is lay:
                return hit[1]
            spans = []
            for a, b, kind in lay.segments:
                if kind in exact and b > a:
                    if spans and spans[-1][1] == a:
                        spans[-1][1] = b
                    else:
                        spans.append([a, b])
            spans_cache.clear()
            spans_cache[id(lay)] = (lay, [tuple(x) for x in spans])
            return spans_cache[id(lay)][1]

        def override(func, *args, **kwargs):
            q, k, v = args[:3]
            mask = kwargs.get("mask", args[4] if len(args) > 4 else None)
            if (kwargs.get("transformer_options") or {}).get("rapidh3_text_bias") is not None and q.shape[-2] >= 1024:
                raise RuntimeError("RapidH3TextAttnBias: DiT attention left the fused INT8 path (RapidH3LinearQuant "
                                   "fuse_block + RapidH3Int8Attention 190 or 194 needed)")
            if (mask is None and kwargs.get("skip_reshape") and q.dim() == 4 and q.shape[0] == 1
                    and q.shape == k.shape == v.shape and q.shape[-1] == 128 and q.shape[2] >= 1024
                    and q.dtype == torch.bfloat16):
                out = torch.empty_like(q)
                attn_int8.attention(q[0].transpose(0, 1), k[0].transpose(0, 1), v[0].transpose(0, 1),
                                    kwargs.get("scale"), config=config, out=out[0].transpose(0, 1))
                if exact:
                    for a, b in exact_spans(kwargs.get("transformer_options", {}), q.shape[2]):
                        out[:, :, a:b] = F.scaled_dot_product_attention(q[:, :, a:b], k, v, scale=kwargs.get("scale"))
                if kwargs.get("skip_output_reshape"):
                    return out
                return out.transpose(1, 2).reshape(1, q.shape[2], -1)
            return prev(func, *args, **kwargs) if prev else func(*args, **kwargs)

        if rmsrope:
            override.rapidh3_rmsrope = config  # read by _fused_block (only while this override is the active one)
            override.rapidh3_rrcheck = rrcheck
        m.model_options["transformer_options"]["optimized_attention_override"] = override
        return (m,)


class RapidH3TextAttnBias:
    """Jenga text-attention bias (arXiv 2505.16864, github.com/JIA-Lab-research/Jenga) for a
    reduced-size draft: every query's attention logits on the text span (rows [0, text_len) of H3's packed sequence,
    i.e. the Qwen3-VL states incl. their first-frame vision tokens) get + rho * ln(sqrt(ratio)), ratio = full-size /
    draft-size tokens per frame. That is Jenga's code: text_amp = -log2(sqrt(token_diff)) * scale_txt_amp added to
    the exp2-domain logits of the text key blocks, in every layer and head, for every query row, low-res stage only.
    Runs inside the fused INT8 attention (RapidH3LinearQuant fuse_block + RapidH3Int8Attention 194 or 190, kernel config 94
    or 90 with the bias folded into its dequantized scores); any other DiT attention path raises. rho 0: model
    unchanged. Opt-in research node: rejected as a draft lever on 2026-10-27."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",),
                             "rho": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.05}),
                             "ratio": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 64.0, "step": 0.001})}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Rapid-H3"

    def patch(self, model, rho, ratio):
        import math
        bias = rho * 0.5 * math.log(ratio)
        if bias == 0.0:
            return (model,)
        # a transformer_options entry, not a wrapper: ComfyUI's clone_has_same_weights compares wrapper counts, so a
        # wrapper on the draft only would make the draft and refine models look different (reload at the switch)
        m = model.clone()
        m.model_options["transformer_options"]["rapidh3_text_bias"] = bias
        print(f"[rapid-h3] text-attention bias {bias:.4f} (rho {rho}, ratio {ratio:.3f}) on the draft model")
        return (m,)


class RapidH3SaveLatent:
    """Save every component of a (possibly nested video+audio) latent as safetensors."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",), "filename_prefix": ("STRING", {"default": "rapid-h3"})}}

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "Rapid-H3"

    def save(self, samples, filename_prefix):
        import folder_paths
        from safetensors.torch import save_file
        s = samples["samples"]
        parts = s.tensors if getattr(s, "is_nested", False) else [s]
        names = ["video", "audio"] if len(parts) == 2 else [f"t{i}" for i in range(len(parts))]
        out_dir, name, counter, sub, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory())
        fname = f"{name}_{counter:05}_.latent"
        save_file({n: t.contiguous().cpu() for n, t in zip(names, parts)}, os.path.join(out_dir, fname))
        return {"ui": {"latents": [{"filename": fname, "subfolder": sub, "type": "output"}]}}


NODE_CLASS_MAPPINGS = {"RapidH3LinearQuant": RapidH3LinearQuant, "RapidH3SaveLatent": RapidH3SaveLatent, "RapidH3Shadow": RapidH3Shadow, "RapidH3DumpQKV": RapidH3DumpQKV, "RapidH3Profile": RapidH3Profile, "RapidH3StepTimer": RapidH3StepTimer,
                       "RapidH3DecodeSaveVideo": RapidH3DecodeSaveVideo, "RapidH3Int8Attention": RapidH3Int8Attention,
                       "RapidH3TextAttnBias": RapidH3TextAttnBias,
                       "RapidH3FrameSkip": RapidH3FrameSkip, "RapidH3AsymRnR": RapidH3AsymRnR}
NODE_CLASS_MAPPINGS.update(av_inputs.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(encode_once.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(upscale_lean.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(te_quant.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(sparse_attn.NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS = {"RapidH3LinearQuant": "Rapid-H3 Linear Quant (research)",
                              "RapidH3SaveLatent": "Rapid-H3 Save Latent"}
