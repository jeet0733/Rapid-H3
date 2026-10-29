"""Triton kernels for row-wise FP8 linears on GB10 (sm_121).

quant_rows(x)          -> (q fp8 [M, K], s fp32 [M, 1]); s = amax(|x_row|) / 448
fp8_gemm(q, s, wq, ws) -> bf16 [M, N] = (q @ wq^T) * s * ws, scales applied in the epilogue

Both replace multi-pass PyTorch code with one pass each.  The GEMM exists because PyTorch's
own row-wise _scaled_mm aborts on sm_121.
"""
import torch
import triton
import triton.language as tl

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0


@triton.jit
def _quant_rows_kernel(x_ptr, q_ptr, s_ptr, K, stride_x, stride_q, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_row = x_ptr + row.to(tl.int64) * stride_x
    amax = tl.zeros([BLOCK], dtype=tl.float32)
    for k0 in range(0, K, BLOCK):
        offs = k0 + tl.arange(0, BLOCK)
        v = tl.load(x_row + offs, mask=offs < K, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.abs(v))
    scale = tl.maximum(tl.max(amax, axis=0) / 448.0, 1e-12)
    tl.store(s_ptr + row, scale)
    inv = 1.0 / scale
    q_row = q_ptr + row.to(tl.int64) * stride_q
    for k0 in range(0, K, BLOCK):
        offs = k0 + tl.arange(0, BLOCK)
        v = tl.load(x_row + offs, mask=offs < K, other=0.0).to(tl.float32)
        v = tl.minimum(tl.maximum(v * inv, -448.0), 448.0)
        tl.store(q_row + offs, v.to(tl.float8e4nv), mask=offs < K)


@triton.jit
def _quant_rows_i8_kernel(x_ptr, q_ptr, s_ptr, K, stride_x, stride_q, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_row = x_ptr + row.to(tl.int64) * stride_x
    amax = tl.zeros([BLOCK], dtype=tl.float32)
    for k0 in range(0, K, BLOCK):
        offs = k0 + tl.arange(0, BLOCK)
        v = tl.load(x_row + offs, mask=offs < K, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.abs(v))
    scale = tl.maximum(tl.max(amax, axis=0) / 127.0, 1e-12)
    tl.store(s_ptr + row, scale)
    inv = 1.0 / scale
    q_row = q_ptr + row.to(tl.int64) * stride_q
    for k0 in range(0, K, BLOCK):
        offs = k0 + tl.arange(0, BLOCK)
        v = tl.load(x_row + offs, mask=offs < K, other=0.0).to(tl.float32) * inv
        v = tl.extra.cuda.libdevice.rint(tl.minimum(tl.maximum(v, -127.0), 127.0))
        tl.store(q_row + offs, v.to(tl.int8), mask=offs < K)


def quant_rows_i8(x):
    """Per-row symmetric INT8: s = amax / 127, q = rint(x / s)."""
    assert x.ndim == 2 and x.stride(1) == 1
    m, k = x.shape
    q = torch.empty((m, k), device=x.device, dtype=torch.int8)
    s = torch.empty((m, 1), device=x.device, dtype=torch.float32)
    _quant_rows_i8_kernel[(m,)](x, q, s, k, x.stride(0), q.stride(0), BLOCK=1024, num_warps=4)
    return q, s


_RH = {}


def regular_hadamard(n, device, dtype=torch.float32):
    """ComfyUI / comfy-kitchen ConvRot matrix: Kronecker powers of the 4x4 regular Hadamard,
    normalized to be orthonormal.  Weights are stored as W @ blockdiag(H)^T (int8_convrot)."""
    key = (n, str(device), dtype)
    if key not in _RH:
        h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float64)
        h = h4
        while h.shape[0] < n:
            h = torch.kron(h, h4)
        _RH[key] = (h / n ** 0.5).to(device=device, dtype=dtype)
    return _RH[key]


ROT_GROUP = 256   # Comfy int8_convrot group size; H256 = H4 (x) H64, applied as A . X . B^T


def convrot(x, group=ROT_GROUP):
    """x @ blockdiag(H)^T in fp32 (reference / offline weight rotation)."""
    m, k = x.shape
    h = regular_hadamard(group, x.device, torch.float32)
    return (x.float().reshape(-1, group) @ h.T).reshape(m, k).to(x.dtype)


rotate = convrot


def _h64t(device):
    """B^T for H256 = (H4/2) (x) B, B = H64 (normalized); bf16 is exact for +-1/8."""
    return regular_hadamard(64, device, torch.float32).T.contiguous().to(torch.bfloat16)


@triton.jit
def _rot256(x0, x1, x2, x3, bt):
    """y = [x0|x1|x2|x3] @ H256^T for [BM, 64] fp32 tiles: Z_k = X_k B^T, Y_i = sum_k A[i,k] Z_k."""
    z0 = tl.dot(x0.to(tl.bfloat16), bt)
    z1 = tl.dot(x1.to(tl.bfloat16), bt)
    z2 = tl.dot(x2.to(tl.bfloat16), bt)
    z3 = tl.dot(x3.to(tl.bfloat16), bt)
    y0 = (z0 + z1 + z2 - z3) * 0.5
    y1 = (z0 + z1 - z2 + z3) * 0.5
    y2 = (z0 - z1 + z2 + z3) * 0.5
    y3 = (z1 + z2 + z3 - z0) * 0.5
    return y0, y1, y2, y3


@triton.jit
def _amax4(a, y0, y1, y2, y3):
    m = tl.maximum(tl.max(tl.abs(y0), axis=1), tl.max(tl.abs(y1), axis=1))
    m = tl.maximum(m, tl.maximum(tl.max(tl.abs(y2), axis=1), tl.max(tl.abs(y3), axis=1)))
    return tl.maximum(a, m)


@triton.jit
def _q8(y, inv):
    y = y * inv[:, None]
    return tl.extra.cuda.libdevice.rint(tl.minimum(tl.maximum(y, -127.0), 127.0)).to(tl.int8)


@triton.jit
def _load_act(base, k, rmask, KIND: tl.constexpr, F,
              rstd, w_ptr, t_off, shift_ptr, scale_ptr, cols, HAS_W: tl.constexpr, HSTRIDE):
    """One [BM, 64] activation tile at column k, after the fused prologue:
    KIND 0: x;  1: silu(x[:, k]) * x[:, F + k];  2: rms_norm(x) * w * (1 + scale[idx]) + shift[idx];
    KIND 3: x given head-major as [H, M, 128] (attention output), column k = head k//128, dim k%128."""
    if KIND == 3:
        v = tl.load(base + (k // 128) * HSTRIDE + (k % 128), mask=rmask[:, None], other=0.0).to(tl.float32)
    else:
        v = tl.load(base + k, mask=rmask[:, None], other=0.0).to(tl.float32)
    if KIND == 1:
        u = tl.load(base + F + k, mask=rmask[:, None], other=0.0).to(tl.float32)
        v = v * tl.sigmoid(v) * u
    if KIND == 2:
        v = v * rstd[:, None]
        if HAS_W:
            v = v * tl.load(w_ptr + k + cols).to(tl.float32)[None, :]
        sc = tl.load(scale_ptr + t_off + k, mask=rmask[:, None], other=0.0).to(tl.float32)
        sh = tl.load(shift_ptr + t_off + k, mask=rmask[:, None], other=0.0).to(tl.float32)
        v = v * (1.0 + sc) + sh
    return v


@triton.jit
def _rot_quant_kernel(x_ptr, bt_ptr, q_ptr, s_ptr, M, K, stride_x, stride_q, eps,
                      w_ptr, idx_ptr, shift_ptr, scale_ptr, stride_tab, HSTRIDE,
                      KIND: tl.constexpr, HAS_W: tl.constexpr, BM: tl.constexpr):
    """q = int8(prologue(x) @ blockdiag(H256)^T) with per-row scales; K is the quantized width.
    Two passes over the input (amax, then quantize); the rotation is recomputed, not stored."""
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM)
    rmask = rows < M
    cols = tl.arange(0, 64)
    bt = tl.load(bt_ptr + cols[:, None] * 64 + cols[None, :])
    base = x_ptr + rows[:, None].to(tl.int64) * stride_x + cols[None, :]
    rstd = tl.zeros([BM], dtype=tl.float32)
    t_off = cols[None, :]
    if KIND == 2:
        ss = tl.zeros([BM], dtype=tl.float32)
        for k0 in range(0, K, 64):
            v = tl.load(base + k0, mask=rmask[:, None], other=0.0).to(tl.float32)
            ss += tl.sum(v * v, axis=1)
        rstd = 1.0 / tl.sqrt(ss / K + eps)
        r = tl.load(idx_ptr + rows, mask=rmask, other=0).to(tl.int64)
        t_off = r[:, None] * stride_tab + cols[None, :]
    amax = tl.zeros([BM], dtype=tl.float32)
    for k0 in range(0, K, 256):
        x0 = _load_act(base, k0, rmask, KIND, K, rstd, w_ptr, t_off, shift_ptr, scale_ptr, cols, HAS_W, HSTRIDE)
        x1 = _load_act(base, k0 + 64, rmask, KIND, K, rstd, w_ptr, t_off, shift_ptr, scale_ptr, cols, HAS_W, HSTRIDE)
        x2 = _load_act(base, k0 + 128, rmask, KIND, K, rstd, w_ptr, t_off, shift_ptr, scale_ptr, cols, HAS_W, HSTRIDE)
        x3 = _load_act(base, k0 + 192, rmask, KIND, K, rstd, w_ptr, t_off, shift_ptr, scale_ptr, cols, HAS_W, HSTRIDE)
        y0, y1, y2, y3 = _rot256(x0, x1, x2, x3, bt)
        amax = _amax4(amax, y0, y1, y2, y3)
    qs = tl.maximum(amax / 127.0, 1e-12)
    tl.store(s_ptr + rows, qs, mask=rmask)
    inv = 1.0 / qs
    qb = q_ptr + rows[:, None].to(tl.int64) * stride_q + cols[None, :]
    for k0 in range(0, K, 256):
        x0 = _load_act(base, k0, rmask, KIND, K, rstd, w_ptr, t_off, shift_ptr, scale_ptr, cols, HAS_W, HSTRIDE)
        x1 = _load_act(base, k0 + 64, rmask, KIND, K, rstd, w_ptr, t_off, shift_ptr, scale_ptr, cols, HAS_W, HSTRIDE)
        x2 = _load_act(base, k0 + 128, rmask, KIND, K, rstd, w_ptr, t_off, shift_ptr, scale_ptr, cols, HAS_W, HSTRIDE)
        x3 = _load_act(base, k0 + 192, rmask, KIND, K, rstd, w_ptr, t_off, shift_ptr, scale_ptr, cols, HAS_W, HSTRIDE)
        y0, y1, y2, y3 = _rot256(x0, x1, x2, x3, bt)
        tl.store(qb + k0, _q8(y0, inv), mask=rmask[:, None])
        tl.store(qb + k0 + 64, _q8(y1, inv), mask=rmask[:, None])
        tl.store(qb + k0 + 128, _q8(y2, inv), mask=rmask[:, None])
        tl.store(qb + k0 + 192, _q8(y3, inv), mask=rmask[:, None])


ROT_CFG = {"bm": 16, "num_warps": 4, "num_stages": 2}


def _launch(x, k, kind, eps=0.0, weight=None, idx=None, shift=None, scale=None, cfg=None, hstride=0):
    assert x.ndim == 2 and x.stride(1) == 1 and k % ROT_GROUP == 0
    m = x.shape[0]
    q = torch.empty((m, k), device=x.device, dtype=torch.int8)
    s = torch.empty((m, 1), device=x.device, dtype=torch.float32)
    c = cfg or ROT_CFG
    dummy = s
    _rot_quant_kernel[(triton.cdiv(m, c["bm"]),)](
        x, _h64t(x.device), q, s, m, k, x.stride(0), q.stride(0), float(eps),
        weight if weight is not None else dummy, idx if idx is not None else dummy,
        shift if shift is not None else dummy, scale if scale is not None else dummy,
        shift.stride(0) if shift is not None else 0, hstride,
        KIND=kind, HAS_W=weight is not None, BM=c["bm"], num_warps=c["num_warps"], num_stages=c["num_stages"])
    return q, s


def rot_quant_rows_i8(x, cfg=None):
    """INT8 of x @ blockdiag(H256)^T with per-row scales (Comfy int8_convrot activation side)."""
    return _launch(x, x.shape[1], 0, cfg=cfg)


def rot_quant_heads_i8(o, cfg=None):
    """Same as rot_quant_rows_i8(o.transpose(0, 1).reshape(M, H * 128)) for a head-major attention
    output o [H, M, 128] (any head stride), without materializing the transpose."""
    h, m, d = o.shape
    assert d == 128 and o.stride(2) == 1
    return _launch(o[0], h * d, 3, cfg=cfg, hstride=o.stride(0))


def swiglu_rot_quant_rows_i8(x, cfg=None):
    """x [M, 2F] (gate | up) -> INT8 of rotate(silu(gate) * up): replaces eager SwiGLU + quantizer."""
    return _launch(x, x.shape[1] // 2, 1, cfg=cfg)


def norm_mod_rot_quant_rows_i8(x, weight, eps, idx, shift, scale, cfg=None):
    """INT8 of rotate(rms_norm(x) * w * (1 + scale[idx]) + shift[idx]); idx [M] int32 rows into
    shift/scale tables [R, K] (shared row stride, unit column stride)."""
    assert shift.stride(1) == 1 and scale.stride(1) == 1 and shift.stride(0) == scale.stride(0)
    return _launch(x, x.shape[1], 2, eps=eps, weight=weight, idx=idx, shift=shift, scale=scale, cfg=cfg)


def quant_rows(x):
    assert x.ndim == 2 and x.stride(1) == 1
    m, k = x.shape
    q = torch.empty((m, k), device=x.device, dtype=FP8)
    s = torch.empty((m, 1), device=x.device, dtype=torch.float32)
    _quant_rows_kernel[(m,)](x, q, s, k, x.stride(0), q.stride(0), BLOCK=1024, num_warps=4)
    return q, s


def _configs():
    out = []
    for bm, bn, bk, w, st in [(128, 128, 64, 4, 4), (128, 128, 128, 8, 3), (128, 256, 64, 8, 3),
                              (256, 128, 64, 8, 3), (128, 128, 64, 8, 4), (64, 256, 128, 4, 3),
                              (128, 256, 128, 8, 2), (256, 128, 128, 8, 2)]:
        out.append(triton.Config({"BM": bm, "BN": bn, "BK": bk, "GROUP": 8}, num_warps=w, num_stages=st))
    return out


@triton.autotune(configs=_configs(), key=["M", "N", "K"])
@triton.jit
def _fp8_gemm_kernel(a_ptr, b_ptr, c_ptr, sa_ptr, sb_ptr, M, N, K,
                     stride_am, stride_bn, stride_cm,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    group = GROUP * num_n
    gid = pid // group
    first_m = gid * GROUP
    gsize = tl.minimum(num_m - first_m, GROUP)
    pid_m = first_m + (pid % group) % gsize
    pid_n = (pid % group) // gsize

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_m[:, None].to(tl.int64) * stride_am + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :].to(tl.int64) * stride_bn + offs_k[:, None]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k0 < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_n[None, :] < N) & (offs_k[:, None] + k0 < K), other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK
        b_ptrs += BK
    sa = tl.load(sa_ptr + offs_m, mask=offs_m < M, other=0.0)
    sb = tl.load(sb_ptr + offs_n, mask=offs_n < N, other=0.0)
    c = acc * sa[:, None] * sb[None, :]
    c_ptrs = c_ptr + offs_m[:, None].to(tl.int64) * stride_cm + offs_n[None, :]
    tl.store(c_ptrs, c.to(tl.bfloat16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def fp8_gemm(q, sa, wq, sb):
    """q [M, K] fp8, sa [M, 1] fp32, wq [N, K] fp8, sb [1, N] fp32 -> bf16 [M, N]."""
    m, k = q.shape
    n = wq.shape[0]
    c = torch.empty((m, n), device=q.device, dtype=torch.bfloat16)
    grid = lambda meta: (triton.cdiv(m, meta["BM"]) * triton.cdiv(n, meta["BN"]),)
    _fp8_gemm_kernel[grid](q, wq, c, sa, sb, m, n, k, q.stride(0), wq.stride(0), c.stride(0))
    return c
