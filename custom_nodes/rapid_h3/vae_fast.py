"""Faster INT8 video-VAE decoder blocks, bit-identical to ComfyUI's int8_convrot decoder (checked on ComfyUI 0.39 +
comfy-kitchen 0.2.37 and on 0.42.1 + 0.2.39, the release image) except where noted below.

- Attention.to_out: quantize the int8-attention output straight from its head-major layout with
  nan_to_num on load (csrc/vae_ops.cu), instead of transpose-copy + nan_to_num-copy + quantize.
- All four decoder linears: comfy-kitchen's ConvRot quantizer, then our CUTLASS GEMM carrying
  comfy-kitchen's exact epilogue (bias, residual addcmul) with a per-shape tile choice.

Levels (RAPIDH3_VAE_FAST, see __init__.py; default 4 since 2026-10-27, level 3 before; bench/vae_bm.py,
bench/vae_swiglu_test.py):
- install_tile_batch(n): all tiles of a temporal chunk (28 at 768x1344) go through the decoder together
  instead of <= 4 tiles of one row per call. Bit-identical; -0.3..0.6 s per decode.
- install_swiglu(): w1 rows interleaved [g|u] once (cached, +1.2 GB), SwiGLU in the w1 GEMM epilogue, fp16 h
  into a plain w2 quantizer. Not bit-identical (0.7% of w2 input codes move by 1; 61-62 dB vs level 1,
  60.6 dB vs the fp16 VAE for both); with tile batching ≈-1.5..1.9 s per decode.
- install_ropeq() (RAPIDH3_VAE_FAST=4, round 5): the attention's q/k RMSNorm + rope fused into comfy-kitchen's
  SageAttention Q/K quantizer (csrc vae_qk_rope_quant), then ck's own V quantizer and attention kernel. Bit-identical
  to level 3 (bench/vae_r5_test.py, full decodes); ≈-1.0 s per 141-frame decode.

Anything outside the measured fast path (non-INT8 weights, a LoRA dequant, tiny M, other dtypes)
falls back to the stock forward.
"""
import os
import weakref

import torch
from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))
_mod = None
_ORIG = {}


def ext():
    global _mod
    if _mod is None:
        cutlass = os.environ.get("RAPIDH3_CUTLASS", "/work/third_party/cutlass")
        build = os.path.join(os.environ.get("RAPIDH3_BUILD", os.path.join(_HERE, "build")), "vae")
        os.makedirs(build, exist_ok=True)
        _mod = load(
            name="rapidh3_vae_ops",
            sources=[os.path.join(_HERE, "csrc", "vae_ops.cu")],
            extra_include_paths=[f"{cutlass}/include", f"{cutlass}/tools/util/include"],
            extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr", "--use_fast_math",
                               "-gencode=arch=compute_121,code=sm_121"],
            build_directory=build, verbose=False)
    return _mod


def gemm_config(m, n, k):
    """Our tile for a decoder GEMM, or None where comfy-kitchen's own pick is as fast or faster
    (bench/vae_fast_test.py, bench/vae_cfg_sweep.py; M = 7188 / 5391 for 4 / 3 tiles per call).
    Both produce identical bits, so this is purely a speed choice."""
    if k >= 8192:
        return 4                       # w2: 1.33 vs 1.59 ms (M=7188), 1.04 vs 1.56 ms (M=5391)
    if m >= 6000 and n >= 6144:
        return 6                       # qkv 1.10 vs 1.27 ms, w1 2.78 vs 3.24 ms
    return None                        # to_out, and qkv/w1 at M=5391: comfy-kitchen is ahead


def _ck():
    from comfy_kitchen.backends import cuda as ckc
    return ckc


def _quantize(x2d, act, act_weight, eps):
    """comfy-kitchen's fused ConvRot quantizer, called exactly as int8_linear calls it."""
    ckc = _ck()
    k_act = x2d.shape[-1] // ckc._input_act_width(act)
    xq = torch.empty((x2d.shape[0], k_act), dtype=torch.int8, device=x2d.device)
    xs = torch.empty((x2d.shape[0], 1), dtype=torch.float32, device=x2d.device)
    w = ckc._wrap_for_dlpack
    ckc._C.quantize_int8_rowwise_convrot64(
        w(x2d), w(xq), w(xs), 256, False, ckc._input_act_code(act), 0,
        w(ckc._act_weight_arg(act, act_weight, x2d.device, x2d.dtype)), float(eps),
        torch.cuda.current_stream(x2d.device).cuda_stream)
    return xq, xs


_LOGGED = set()


def _why(tag, reason):
    if (tag, reason) not in _LOGGED:
        _LOGGED.add((tag, reason))
        print(f"[rapid-h3] vae_fast {tag}: {reason}", flush=True)
    return False


def _fast_ok(linear, x, act):
    import comfy.model_management
    from comfy.quant_ops import QuantizedTensor
    w = getattr(linear, "weight", None)
    if not isinstance(w, QuantizedTensor):
        return _why("fallback", f"weight is {type(w).__name__}")
    if not (x.is_cuda and x.dtype == torch.float16):
        return _why("fallback", f"input {x.dtype} on {x.device}")
    p = w._params
    if w._layout_cls != "TensorWiseINT8Layout" or getattr(p, "transposed", False) or not getattr(p, "convrot", False):
        return _why("fallback", f"layout {w._layout_cls} convrot={getattr(p, 'convrot', None)}")
    if getattr(p, "convrot_groupsize", 256) != 256 or getattr(linear, "_full_precision_mm", False):
        return _why("fallback", "groupsize/full precision")
    if comfy.model_management.in_training or len(getattr(linear, "weight_function", [])) > 0:
        return _why("fallback", "training or weight_function")
    k_act = x.shape[-1] // _ck()._input_act_width(act)
    m = x.numel() // x.shape[-1]
    if not (k_act % 256 == 0 and 256 <= k_act <= 16384 and m >= 256):
        return _why("fallback", f"shape m={m} k={k_act}")
    _why("active", "fast path in use")
    return True


def _linear(linear, x, act, act_weight=None, eps=0.0, residual=None, residual_scale=None, xq_xs=None):
    """linear_input_act(linear, x, act, ...) for an fp16 INT8-ConvRot linear, our GEMM."""
    import comfy.ops
    import comfy.quant_ops
    from comfy.quant_ops import QuantizedTensor
    ckc = _ck()
    weight, bias, stream = comfy.ops.cast_bias_weight(linear, x, offloadable=True, compute_dtype=x.dtype,
                                                      want_requant=True)
    try:
        if not isinstance(weight, QuantizedTensor):
            return None
        # ComfyUI >= 0.42 moved TensorWiseINT8Layout from comfy.ops to comfy.quant_ops.
        _L = getattr(comfy.quant_ops, "TensorWiseINT8Layout", None) or comfy.ops.TensorWiseINT8Layout
        qdata, scale = _L.get_plain_tensors(weight)
        lead = x.shape[:-1]
        if xq_xs is None:
            xq, xs = _quantize(x.reshape(-1, x.shape[-1]).contiguous(), act, act_weight, eps)
        else:
            xq, xs = xq_xs
        n = qdata.shape[0]
        qdata = qdata if qdata.is_contiguous() else qdata.contiguous()
        ws = ckc._int8_weight_scale_arg(scale, x.device)
        ws = ws if ws.numel() == n else ws.expand(n).contiguous()
        bias16 = (ckc._gemm_vector_arg(bias, x.device, torch.float16) if bias is not None
                  else torch.zeros(n, dtype=torch.float16, device=x.device))
        cfg = gemm_config(xq.shape[0], n, xq.shape[1])
        xs1, ws1 = xs.reshape(-1), ws.reshape(-1)
        rs = resid = None
        if residual is not None:
            rs = ckc._gemm_vector_arg(residual_scale, x.device, torch.float16)
            resid = residual.reshape(-1, n).contiguous()
        if cfg is not None:
            out = ext().vae_gemm(xq, qdata, xs1, ws1, bias16, rs, resid, cfg)
        else:  # comfy-kitchen's own fused GEMM (same epilogue, its tile heuristic)
            w = ckc._wrap_for_dlpack
            out = torch.empty((xq.shape[0], n), dtype=torch.float16, device=x.device)
            sp = torch.cuda.current_stream(x.device).cuda_stream
            ok = (ckc._C.cutlass_int8_dequant_residual(w(xq), w(qdata), w(xs1), w(ws1), w(bias16), w(rs), w(resid),
                                                        w(out), 1, sp) if resid is not None else
                  ckc._C.cutlass_int8_dequant(w(xq), w(qdata), w(xs1), w(ws1), w(bias16), w(out), 1, sp))
            if not ok:
                out = ext().vae_gemm(xq, qdata, xs1, ws1, bias16, rs, resid, 6)
        return out.reshape(*lead, n)
    finally:
        comfy.ops.uncast_bias_weight(linear, weight, bias, stream)


_ROPEQ = False
# comfy-kitchen versions whose rope / SageAttention Q/K quantizer / anchor / V quantizer code is the code
# vae_qk_rope_quant copies or calls (0.2.39 only adds paged-decode kernels next to it); checked bit-identical
_CK_VERIFIED = ("0.2.37", "0.2.39")
_ck_version = None


def _ck_ok():
    global _ck_version
    if _ck_version is None:
        try:
            from importlib.metadata import version
            _ck_version = version("comfy-kitchen")
        except Exception:  # noqa: BLE001
            _ck_version = "unknown"
    return _ck_version in _CK_VERIFIED


def _attn_ropeq(self, qkv, rope):
    """Level 4: comfy-kitchen's q/k rms+rope fused into its SageAttention Q/K quantizer (csrc vae_qk_rope_quant),
    then ck's own V quantizer and attention kernel on the prequantized tensors. Same bits as
    rms_rope_split_half_ + int8_attention (bench/vae_r5_test.py). Returns [B, H, S, 64] fp16, or None."""
    D = self.dim_head
    if not (D == 64 and qkv.dtype == torch.float16 and rope.dtype == torch.float16 and rope.dim() == 6
            and rope.shape[2] == 1 and rope.shape[-2:] == (2, 2) and rope.shape[-3] * 4 <= 2 * D
            and rope.shape[-3] % 2 == 0 and qkv.stride(-1) == 1 and rope.device == qkv.device
            and rope.stride()[-3:] == (4, 2, 1) and all(st % 8 == 0 for st in qkv.stride()[:3])
            and qkv.data_ptr() % 16 == 0 and rope.data_ptr() % 16 == 0
            and (rope.shape[0] == 1 or rope.stride(0) % 8 == 0) and (rope.shape[1] == 1 or rope.stride(1) % 8 == 0)):
        _why("attn fallback", f"ropeq shape D={D} {qkv.dtype} rope {tuple(rope.shape)} {rope.dtype}")
        return None
    if not _ck_ok():
        _why("attn fallback", f"comfy-kitchen {_ck_version}: level 4 is verified on {', '.join(_CK_VERIFIED)}, level 3 path used")
        return None
    ckc = _ck()
    if not all(hasattr(ckc._C, f) for f in ("_quant_v_int8", "sage_sdpa_prequantized")):
        _why("attn fallback", "comfy-kitchen without sage_sdpa_prequantized")
        return None
    b, s, h, _ = qkv.shape
    scale = self.qk_norm_scale.to(device=qkv.device, dtype=torch.float32)
    q8, qs, k8, ks, _ = ext().vae_qk_rope_quant(qkv, rope, scale, float(self.norm_q.eps))
    v = qkv[..., 2 * D:].transpose(1, 2)                     # [B, H, S, 64] view into qkv
    padded = (s + 63) // 64 * 64                              # cta_k 64 (head_dim 64)
    v8 = torch.empty(b * h * D, padded, dtype=torch.int8, device=qkv.device)
    vs = torch.empty(b * h * D, dtype=torch.float32, device=qkv.device)
    W = ckc._wrap_for_dlpack
    sp = torch.cuda.current_stream(qkv.device).cuda_stream
    ckc._C._quant_v_int8(W(v), W(v8), W(vs), padded, 1, sp)
    out = torch.empty(b, h, s, D, dtype=torch.float16, device=qkv.device)
    ckc._C.sage_sdpa_prequantized(W(q8), W(k8), W(v8), W(out), W(qs), W(ks), W(vs), 64, D ** -0.5, 1, sp, None)
    return out


def _attn_forward(self, x, rotary_pos_emb, pre_norm, residual, residual_scale):
    import comfy.ops
    import comfy.quant_ops
    if not (_fast_ok(self.to_qkv, x, "rms_norm") and _fast_ok(self.to_out, x, None) and rotary_pos_emb is not None
            and residual is not None and residual.dtype == torch.float16 and residual_scale.dtype == torch.float16):
        return _ORIG["attn"](self, x, rotary_pos_emb, pre_norm, residual, residual_scale)
    b, s, _ = x.shape
    with comfy.ops.CastBiasWeightContext(pre_norm if pre_norm.weight is not None else None, x,
                                         offloadable=True) as (weight, bias):
        qkv = _linear(self.to_qkv, x, "rms_norm", weight, pre_norm.eps)
    if qkv is None:
        return _ORIG["attn"](self, x, rotary_pos_emb, pre_norm, residual, residual_scale)
    qkv = qkv.view(b, s, -1, 3 * self.dim_head)
    out = _attn_ropeq(self, qkv, rotary_pos_emb) if _ROPEQ else None
    if out is not None:
        del qkv
        xq, xs = ext().vae_out_quant(out)
        del out
        r = _linear(self.to_out, residual, None, residual=residual, residual_scale=residual_scale, xq_xs=(xq, xs))
        if r is None:
            raise RuntimeError("Rapid-H3 vae_fast: to_out weight lost its INT8 form mid-block")
        return r
    query, key, value = torch.chunk(qkv, 3, dim=-1)
    query, key = comfy.quant_ops.ck.rms_rope_split_half_(
        query, key, rotary_pos_emb, self.qk_norm_scale.to(query.device),
        epsilon=self.norm_q.eps, rot_dim=rotary_pos_emb.shape[-3] * 2)
    query, key, value = (t.transpose(1, 2) for t in (query, key, value))
    out = comfy.quant_ops.ck.int8_attention(query, key, value)          # [B, H, S, D]
    del query, key, value, qkv
    xq, xs = ext().vae_out_quant(out.contiguous())
    del out
    r = _linear(self.to_out, residual, None, residual=residual, residual_scale=residual_scale, xq_xs=(xq, xs))
    if r is None:
        raise RuntimeError("Rapid-H3 vae_fast: to_out weight lost its INT8 form mid-block")
    return r


_SWIGLU = False
_W1_CACHE = weakref.WeakKeyDictionary()   # w1 module -> (source qdata ptr, interleaved qdata, interleaved weight scale, interleaved bias)


def _interleave_rows(t, n):
    """Rows [gate 0..n/2 | up n/2..n] -> [g 4j..4j+3 | u 4j..4j+3] for j = 0..n/8."""
    half = n // 2
    return torch.stack([t[:half].reshape(half // 4, 4, *t.shape[1:]),
                        t[half:].reshape(half // 4, 4, *t.shape[1:])], dim=1).reshape(n, *t.shape[1:])


def _w1_swiglu(linear, x, norm_weight, eps):
    """h = silu(gate) * up of w1(rms_norm(x)), SwiGLU in the GEMM epilogue, fp16 [M, N/2]; None = not possible."""
    import comfy.ops
    import comfy.quant_ops
    from comfy.quant_ops import QuantizedTensor
    ckc = _ck()
    weight, bias, stream = comfy.ops.cast_bias_weight(linear, x, offloadable=True, compute_dtype=x.dtype,
                                                      want_requant=True)
    try:
        if not isinstance(weight, QuantizedTensor):
            return None
        _L = getattr(comfy.quant_ops, "TensorWiseINT8Layout", None) or comfy.ops.TensorWiseINT8Layout
        qdata, scale = _L.get_plain_tensors(weight)
        n = qdata.shape[0]
        if n % 8:
            return None
        hit = _W1_CACHE.get(linear)
        if hit is None or hit[0] != qdata.data_ptr():
            ws = ckc._int8_weight_scale_arg(scale, x.device)
            ws = (ws if ws.numel() == n else ws.expand(n)).reshape(-1)
            b16 = (ckc._gemm_vector_arg(bias, x.device, torch.float16) if bias is not None
                   else torch.zeros(n, dtype=torch.float16, device=x.device))
            hit = (qdata.data_ptr(), _interleave_rows(qdata.contiguous(), n).contiguous(),
                   _interleave_rows(ws.float(), n).contiguous(), _interleave_rows(b16, n).contiguous())
            _W1_CACHE[linear] = hit
        xq, xs = _quantize(x.reshape(-1, x.shape[-1]).contiguous(), "rms_norm", norm_weight, eps)
        h = ext().vae_gemm_swiglu(xq, hit[1], xs.reshape(-1), hit[2], hit[3], 6)
        return h.reshape(*x.shape[:-1], n // 2)
    finally:
        comfy.ops.uncast_bias_weight(linear, weight, bias, stream)


def _ff_forward(self, x, pre_norm, residual, residual_scale):
    import comfy.ops
    if _SWIGLU and _fast_ok(self.w1, x, "rms_norm") and residual is not None \
            and residual.dtype == torch.float16 and residual_scale.dtype == torch.float16:
        with comfy.ops.CastBiasWeightContext(pre_norm if pre_norm.weight is not None else None, x,
                                             offloadable=True) as (weight, bias):
            h = _w1_swiglu(self.w1, x, weight, pre_norm.eps)
        if h is not None and _fast_ok(self.w2, h, None):
            r = _linear(self.w2, h, None, residual=residual, residual_scale=residual_scale)
            if r is not None:
                return r
    if not (_fast_ok(self.w1, x, "rms_norm") and residual is not None and residual.dtype == torch.float16
            and residual_scale.dtype == torch.float16):
        return _ORIG["ff"](self, x, pre_norm, residual, residual_scale)
    with comfy.ops.CastBiasWeightContext(pre_norm if pre_norm.weight is not None else None, x,
                                         offloadable=True) as (weight, bias):
        h = _linear(self.w1, x, "rms_norm", weight, pre_norm.eps)
    if h is None or not _fast_ok(self.w2, h, "swiglu"):
        return _ORIG["ff"](self, x, pre_norm, residual, residual_scale)
    r = _linear(self.w2, h, "swiglu", residual=residual, residual_scale=residual_scale)
    return r if r is not None else _ORIG["ff"](self, x, pre_norm, residual, residual_scale)


def _tile_groups(n, batch):
    """Sizes of the decoder calls for n tiles at about `batch` tiles per call: as equal as possible, never a lone
    tile next to bigger groups. A decoder call on a single tile does not give the same bits as that tile decoded
    together with others (maxabs 4-18 / 255 on er41; groups of >= 2 tiles all match), and batch follows the free
    memory, so the plain [batch, batch, ..., rest] split made the output depend on the memory state (e.g. 27 + 1
    tiles in the last chunks of a render: ~50 dB between two renders of the same latent, on 0.39 and 0.42.1)."""
    if n <= 1:
        return [n] if n else []
    groups = -(-n // max(1, batch))
    if n // groups < 2:  # batch 1 (very low memory): pairs (and one triple) instead of single tiles
        groups = n // 2
    base, extra = divmod(n, groups)
    return [base + 1] * extra + [base] * (groups - extra)


def _tiled_decode_batched(self, z, frames=None):
    """A replacement for MiniMaxH3VideoVAE.tiled_decode that decodes the tiles of the whole chunk `_TILE_BATCH` at
    a time, across tile rows, instead of <= 4 tiles of one row per decoder call. Same tiles, same blending, in the
    same order;
    every decoder op is per tile (row-wise quantizers, int32-exact GEMMs, per-sequence attention), so only the
    GEMM M dimension grows.

    frames: ComfyUI >= 0.42 calls tiled_decode(z, frames) and keeps only those output frames of every tile (a
    single still latent is decoded as a clip of copies, then one frame kept); 0.39 calls tiled_decode(z)."""
    import comfy.model_management
    r = self.vae_ratio
    height, width = (n * r for n in z.shape[-2:])      # output size in pixels
    y_idx, y_len, y_overlap = self.split_tiles(height)  # tile starts, sizes and overlaps per axis (ComfyUI's split)
    x_idx, x_len, x_overlap = self.split_tiles(width)

    def all_tiles():
        slices = [z[..., i_pos // r:(i_pos + i_len) // r, j_pos // r:(j_pos + j_len) // r]
                  for i_pos, i_len in zip(y_idx, y_len) for j_pos, j_len in zip(x_idx, x_len)]
        # each tile in flight holds ~110 MB of activations (5.2 GiB peak at 28 tiles of 1797 tokens)
        free = comfy.model_management.get_free_memory(z.device)
        batch = int(max(1, min(_TILE_BATCH, free // (192 * 2**20 * z.shape[0]))))
        k = 0
        for size in _tile_groups(len(slices), batch):
            group = slices[k:k + size]
            k += size
            out = self._decode_pixels(torch.cat(group))
            del group
            yield from out.chunk(size)
            del out

    # Stitch in tile order. Tile (i, j) covers rows y_idx[i] + [0, y_len[i]) and columns x_idx[j] + [0, x_len[j]);
    # consecutive tiles overlap by y_overlap / x_overlap, and the next tile starts where the kept part of this one
    # ends. Each tile is cross-faded (self.blend) first with the seam rows the tile row above left behind, then
    # with the seam columns its left neighbour left behind, both as already cross-faded. The kept part goes to the
    # canvas; the bottom seam rows are saved for the next tile row, the right seam columns for the next tile.
    rows, cols = len(y_idx), len(x_idx)
    tiles = all_tiles()
    canvas = above = None   # above: bottom seam rows of the previous tile row, full canvas width
    for i in range(rows):
        y0, more_rows = y_idx[i], i + 1 < rows
        below = left = None
        for j in range(cols):
            x0, more_cols = x_idx[j], j + 1 < cols
            t = next(tiles)
            if frames is not None:
                t = t[:, :, frames]
            if above is not None:
                t = self.blend(above[..., :, x0:x0 + x_len[j]], t, y_overlap[i - 1], dim=-2)
            if left is not None:
                t = self.blend(left, t, x_overlap[j - 1], dim=-1)
            h, w = t.shape[-2], t.shape[-1]
            if more_cols:
                w -= x_overlap[j]
                left = t[..., :, w:].clone()
            if canvas is None:
                canvas = t.new_empty(*t.shape[:-2], height, width)
            if more_rows:
                h -= y_overlap[i]
                if below is None:
                    below = t.new_empty(*t.shape[:-2], y_overlap[i], width)
                below[..., :, x0:x0 + w] = t[..., h:, :w]
            canvas[..., y0:y0 + h, x0:x0 + w].copy_(t[..., :h, :w])
            del t
        above = below
    return canvas


_TILE_BATCH = 0


def install_tile_batch(n):
    """Decode n tiles per decoder call across tile rows (RAPIDH3_VAE_TILES); n <= 0 restores ComfyUI's tiling."""
    global _TILE_BATCH
    import comfy.ldm.minimax.vae as mv
    if "tiled_decode" not in _ORIG:
        _ORIG["tiled_decode"] = mv.MiniMaxH3VideoVAE.tiled_decode
    _TILE_BATCH = int(n)
    mv.MiniMaxH3VideoVAE.tiled_decode = _tiled_decode_batched if _TILE_BATCH > 0 else _ORIG["tiled_decode"]


def install_swiglu(on=True):
    """SwiGLU in the w1 GEMM epilogue (RAPIDH3_VAE_FAST=3): not bit-identical, h is rounded to fp16 once."""
    global _SWIGLU
    _SWIGLU = bool(on)
    if not on:
        _W1_CACHE.clear()


def install_ropeq(on=True):
    """q/k rms+rope fused into the attention's Q/K quantizer (RAPIDH3_VAE_FAST=4): bit-identical to level 3."""
    global _ROPEQ
    _ROPEQ = bool(on)


def install():
    """Patch the MiniMax-H3 video VAE decoder blocks (idempotent)."""
    import comfy.ldm.minimax.vae as mv
    if "attn" not in _ORIG:
        _ORIG["attn"] = mv.Attention.forward
        _ORIG["ff"] = mv.FeedForward.forward
        mv.Attention.forward = _attn_forward
        mv.FeedForward.forward = _ff_forward


def uninstall():
    import comfy.ldm.minimax.vae as mv
    if "attn" in _ORIG:
        mv.Attention.forward = _ORIG.pop("attn")
        mv.FeedForward.forward = _ORIG.pop("ff")


def install_level(level, tiles=28):
    """RAPIDH3_VAE_FAST levels: 1 bit-identical blocks, 2 + tile batching, 3 + SwiGLU epilogue, 4 + round-5 kernels."""
    level = int(level)
    if level <= 0:
        uninstall()
        install_tile_batch(0)
        install_swiglu(False)
        install_ropeq(False)
        return
    install()
    install_tile_batch(tiles if level >= 2 else 0)
    install_swiglu(level >= 3)
    install_ropeq(level >= 4)
