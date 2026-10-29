"""Quantized linear kernels for GB10 (sm_121).

Reference path in plain PyTorch.  PyTorch's row-wise FP8 _scaled_mm is broken on sm_121
(aborts with "Arch conditional MMA instruction used without targeting appropriate compute
capability"), so row-wise variants run the working tensor-wise kernel with unit scales and
apply the per-token x per-channel scales afterwards.  That is numerically the same product
up to one bf16 rounding; a fused-epilogue kernel replaces it in the fast path.
"""
from dataclasses import dataclass

import torch

try:  # inside the ComfyUI package
    from . import cutlass_fp8 as _cf, triton_fp8 as _tq
except ImportError:  # standalone benches with custom_nodes/rapid_h3 on sys.path
    import cutlass_fp8 as _cf, triton_fp8 as _tq

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
_ONE = {}


def _one(device):
    if device not in _ONE:
        _ONE[device] = torch.ones((), device=device, dtype=torch.float32)
    return _ONE[device]


@dataclass
class QWeight:
    q: torch.Tensor        # [N, K] quantized
    scale: torch.Tensor    # [1, N] fp32 (row-wise) or [] fp32 (tensor-wise)


E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def fake_nvfp4(t):
    """NVFP4 round trip in fp32 (error check, no fast kernel): 16-element blocks along the last dim,
    e2m1 values, an e4m3 scale per block and one fp32 scale per tensor, as in NVIDIA's NVFP4 recipe."""
    shp = t.shape
    t = t.float().reshape(-1, 16)
    gs = (t.abs().amax() / (FP8_MAX * 6.0)).clamp_min(1e-12)
    s = (t.abs().amax(dim=1, keepdim=True) / 6.0 / gs).clamp(max=FP8_MAX).to(FP8).float() * gs
    s = torch.where(s == 0, torch.ones_like(s), s)
    q = t / s
    grid = torch.tensor(E2M1, device=t.device)
    idx = torch.bucketize(q.abs().clamp(max=6.0), (grid[1:] + grid[:-1]) / 2)
    return (grid[idx] * q.sign() * s).reshape(shp)


def quantize_weight(w, mode):
    if mode in ("nvfp4_fake", "nvfp4_rot_fake"):
        if mode == "nvfp4_rot_fake":
            w = _tq.convrot(w.float())
        return QWeight(fake_nvfp4(w.float()).to(torch.bfloat16), None)
    if mode == "int8_fast":
        mode = "int8_rowwise"
    if mode == "int8_rot":
        w = _tq.convrot(w.float())
        mode = "int8_rowwise"
    w = w.float()
    if mode == "fp8_tensor":
        s = (w.abs().amax() / FP8_MAX).clamp_min(1e-12)
        return QWeight((w / s).clamp(-FP8_MAX, FP8_MAX).to(FP8), s)
    if mode in ("fp8_rowwise", "fp8_rowwise_torch"):
        s = (w.abs().amax(dim=1, keepdim=True) / FP8_MAX).clamp_min(1e-12)
        return QWeight((w / s).clamp(-FP8_MAX, FP8_MAX).to(FP8), s.t().contiguous())
    if mode == "int8_rowwise":
        s = (w.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-12)
        return QWeight((w / s).round().clamp(-127, 127).to(torch.int8), s.t().contiguous())
    raise ValueError(mode)


def quantize_rows(x, mode):
    """Per-token (row) dynamic quantization -> (q, scale[M, 1] fp32)."""
    amax = x.abs().amax(dim=-1, keepdim=True).float()
    if mode == "int8_rowwise":
        s = (amax / 127.0).clamp_min(1e-12)
        return (x.float() / s).round().clamp(-127, 127).to(torch.int8), s
    s = (amax / FP8_MAX).clamp_min(1e-12)
    return (x.float() / s).clamp(-FP8_MAX, FP8_MAX).to(FP8), s


_FAST = None


def _fast():
    """Triton row quantizer + CUTLASS row-wise FP8 GEMM, or None if they cannot load."""
    global _FAST
    if _FAST is None:
        try:
            _cf.ext()
            _FAST = (_tq, _cf)
        except Exception as e:  # noqa: BLE001
            print(f"[rapid-h3] fast FP8 path unavailable, using torch path: {e}", flush=True)
            _FAST = False
    return _FAST or None


def gemm_config(k, n):
    """Best INT8 tile per H3 shape (bench/results/gemm_tune.json, M = 37.7k): 128x256x64 with 4 stages
    wins or ties every block GEMM (qkv 203, out_proj 197, fc1 200, fc2 180 TFLOPS)."""
    return 6


def linear(x, qw, mode):
    """y = x @ W^T for x [M, K] bf16 and a QWeight of W [N, K]; returns x.dtype."""
    if mode == "fp8_rowwise" and x.dtype == torch.bfloat16 and _fast() is not None:
        tq, cf = _fast()
        xq, sa = tq.quant_rows(x.contiguous())
        return cf.fp8_gemm(xq, sa, qw.q, qw.scale, config=2)
    if mode in ("int8_rot", "int8_fast") and x.dtype == torch.bfloat16 and _fast() is not None:
        tq, cf = _fast()
        if mode == "int8_rot":
            xq, sa = tq.rot_quant_rows_i8(x.contiguous())
        else:
            xq, sa = tq.quant_rows_i8(x.contiguous())
        return cf.int8_gemm(xq, sa, qw.q, qw.scale, config=gemm_config(x.shape[1], qw.q.shape[0]))
    if mode in ("nvfp4_fake", "nvfp4_rot_fake"):
        xr = _tq.convrot(x.float()) if mode == "nvfp4_rot_fake" else x
        return torch.nn.functional.linear(fake_nvfp4(xr).to(torch.bfloat16), qw.q).to(x.dtype)
    if mode == "fp8_rowwise_torch":
        mode = "fp8_rowwise"
    if mode == "fp8_tensor":
        s = (x.abs().amax().float() / FP8_MAX).clamp_min(1e-12)
        xq = (x.float() / s).clamp(-FP8_MAX, FP8_MAX).to(FP8)
        return torch._scaled_mm(xq, qw.q.t(), s, qw.scale, out_dtype=x.dtype)
    xq, sa = quantize_rows(x, mode)
    if mode == "fp8_rowwise":
        one = _one(x.device)
        acc = torch._scaled_mm(xq, qw.q.t(), one, one, out_dtype=torch.bfloat16)
    elif mode == "int8_rowwise":
        acc = torch._int_mm(xq, qw.q.t())
    else:
        raise ValueError(mode)
    return (acc.float() * sa * qw.scale).to(x.dtype)
