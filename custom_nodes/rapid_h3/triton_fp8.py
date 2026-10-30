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


@triton.jit
def _load_wide(xrow, c, m2, KIND: tl.constexpr, F, rstd, w_ptr, trow, shift_ptr, scale_ptr,
               HAS_W: tl.constexpr, HSTRIDE):
    """_load_act for a [BM, W] tile at absolute columns c (masked tail); same per-element arithmetic."""
    if KIND == 3:
        v = tl.load(xrow + ((c // 128) * HSTRIDE + (c % 128))[None, :], mask=m2, other=0.0).to(tl.float32)
    else:
        v = tl.load(xrow + c[None, :], mask=m2, other=0.0).to(tl.float32)
    if KIND == 1:
        u = tl.load(xrow + F + c[None, :], mask=m2, other=0.0).to(tl.float32)
        v = v * tl.sigmoid(v) * u
    if KIND == 2:
        v = v * rstd[:, None]
        if HAS_W:
            v = v * tl.load(w_ptr + c, mask=c < F, other=0.0).to(tl.float32)[None, :]
        sc = tl.load(scale_ptr + trow + c[None, :], mask=m2, other=0.0).to(tl.float32)
        sh = tl.load(shift_ptr + trow + c[None, :], mask=m2, other=0.0).to(tl.float32)
        v = v * (1.0 + sc) + sh
    return v


@triton.jit
def _rot_wide(v, bt, BM: tl.constexpr, NG: tl.constexpr):
    """[BM, NG*256] fp32 -> y0..y3 [BM, NG, 64]: the same Z_k = X_k B^T (one dot over all 64-chunks)
    and Y_i = sum_k A[i,k] Z_k as _rot256, element for element."""
    z = tl.dot(tl.reshape(v, (BM * NG * 4, 64)).to(tl.bfloat16), bt)
    z = tl.permute(tl.reshape(z, (BM, NG, 2, 2, 64)), (0, 1, 4, 2, 3))   # [.., c, i, j], chunk = 2i + j
    za, zb = tl.split(z)          # j = 0 / 1: [BM, NG, 64, 2(i)]
    z0, z2 = tl.split(za)
    z1, z3 = tl.split(zb)
    y0 = (z0 + z1 + z2 - z3) * 0.5
    y1 = (z0 + z1 - z2 + z3) * 0.5
    y2 = (z0 - z1 + z2 + z3) * 0.5
    y3 = (z1 + z2 + z3 - z0) * 0.5
    return y0, y1, y2, y3


@triton.jit
def _ss_exact(xrow, rmask, K, BM: tl.constexpr):
    """Row sum of squares with the exact fp32 operation order of _rot_quant_kernel's
    `ss += tl.sum(v * v, axis=1)` over [16, 64] tiles at num_warps=4 (blocked layout: 8 lanes x 8
    contiguous values per row; per lane t = v1*v1, then fma v0, v2..v7; butterfly xor 4, 2, 1;
    chunks added in column order), written as explicit element-wise ops so it does not depend on
    this kernel's BM / num_warps / layout."""
    c = tl.arange(0, 256)
    ss = tl.zeros([BM], dtype=tl.float32)
    for k0 in range(0, K, 256):
        v = tl.load(xrow + (k0 + c)[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
        v = tl.reshape(v, (BM, 4, 8, 2, 2, 2))   # [row, chunk, lane, b2, b1, b0], element = 4*b2 + 2*b1 + b0
        a, b = tl.split(v)
        a0, a1 = tl.split(a)
        b0, b1 = tl.split(b)
        e0, e4 = tl.split(a0)
        e2, e6 = tl.split(a1)
        e1, e5 = tl.split(b0)
        e3, e7 = tl.split(b1)
        t = e1 * e1
        t = tl.fma(e0, e0, t)
        t = tl.fma(e2, e2, t)
        t = tl.fma(e3, e3, t)
        t = tl.fma(e4, e4, t)
        t = tl.fma(e5, e5, t)
        t = tl.fma(e6, e6, t)
        t = tl.fma(e7, e7, t)
        t = tl.reshape(t, (BM, 4, 2, 2, 2))      # [row, chunk, l2, l1, l0], lane = 4*l2 + 2*l1 + l0
        p, q = tl.split(t)
        p0, p1 = tl.split(p)
        q0, q1 = tl.split(q)
        s0, s4 = tl.split(p0)
        s2, s6 = tl.split(p1)
        s1, s5 = tl.split(q0)
        s3, s7 = tl.split(q1)
        cs = tl.reshape(((s0 + s4) + (s2 + s6)) + ((s1 + s5) + (s3 + s7)), (BM, 2, 2))
        ca, cb = tl.split(cs)
        c0, c2 = tl.split(ca)
        c1, c3 = tl.split(cb)
        ss = (((ss + c0) + c1) + c2) + c3
    return ss


@triton.jit
def _rot_quant_wide_kernel(x_ptr, bt_ptr, q_ptr, s_ptr, M, K, stride_x, stride_q, eps,
                           w_ptr, idx_ptr, shift_ptr, scale_ptr, stride_tab, HSTRIDE, NPROG,
                           KIND: tl.constexpr, HAS_W: tl.constexpr, BM: tl.constexpr, NG: tl.constexpr,
                           REV_AMAX: tl.constexpr, REV_Q: tl.constexpr, SS_EXACT: tl.constexpr):
    """_rot_quant_kernel with NG 256-column groups per loop step ([BM, NG*256] tiles, more bytes in
    flight per thread) and a persistent grid (NPROG programs, so the second pass re-reads from L2).
    Bit-identical to it: same per-element prologue, same dot per 64-chunk, same Y combination,
    order-free amax; the KIND 2 sum of squares is _ss_exact."""
    pid = tl.program_id(0)
    cols = tl.arange(0, 64)
    bt = tl.load(bt_ptr + cols[:, None] * 64 + cols[None, :])
    W: tl.constexpr = NG * 256
    wc = tl.arange(0, W)
    gi = tl.arange(0, NG)
    nblk = tl.cdiv(M, BM)
    nch = tl.cdiv(K, W)
    for blk in range(pid, nblk, NPROG):
        rows = blk * BM + tl.arange(0, BM)
        rmask = rows < M
        xrow = x_ptr + rows[:, None].to(tl.int64) * stride_x
        rstd = tl.zeros([BM], dtype=tl.float32)
        trow = rows[:, None].to(tl.int64) * 0
        if KIND == 2:
            if SS_EXACT:
                ss = _ss_exact(xrow, rmask, K, BM)
            else:   # the original [BM, 64] reduction (bit-identical only at BM 16, num_warps 4), 4 tiles per step
                base = xrow + cols[None, :]
                ss = tl.zeros([BM], dtype=tl.float32)
                for k0 in range(0, K, 256):
                    va = tl.load(base + k0, mask=rmask[:, None], other=0.0).to(tl.float32)
                    vb = tl.load(base + k0 + 64, mask=rmask[:, None], other=0.0).to(tl.float32)
                    vc = tl.load(base + k0 + 128, mask=rmask[:, None], other=0.0).to(tl.float32)
                    vd = tl.load(base + k0 + 192, mask=rmask[:, None], other=0.0).to(tl.float32)
                    ss += tl.sum(va * va, axis=1)
                    ss += tl.sum(vb * vb, axis=1)
                    ss += tl.sum(vc * vc, axis=1)
                    ss += tl.sum(vd * vd, axis=1)
            rstd = 1.0 / tl.sqrt(ss / K + eps)
            r = tl.load(idx_ptr + rows, mask=rmask, other=0).to(tl.int64)
            trow = r[:, None] * stride_tab
        amax = tl.zeros([BM], dtype=tl.float32)
        for t in range(0, nch):
            if REV_AMAX:
                k0 = (nch - 1 - t) * W
            else:
                k0 = t * W
            c = k0 + wc
            m2 = rmask[:, None] & (c < K)[None, :]
            v = _load_wide(xrow, c, m2, KIND, K, rstd, w_ptr, trow, shift_ptr, scale_ptr, HAS_W, HSTRIDE)
            y0, y1, y2, y3 = _rot_wide(v, bt, BM, NG)
            m = tl.maximum(tl.max(tl.max(tl.abs(y0), axis=2), axis=1), tl.max(tl.max(tl.abs(y1), axis=2), axis=1))
            m = tl.maximum(m, tl.maximum(tl.max(tl.max(tl.abs(y2), axis=2), axis=1),
                                         tl.max(tl.max(tl.abs(y3), axis=2), axis=1)))
            amax = tl.maximum(amax, m)
        qs = tl.maximum(amax / 127.0, 1e-12)
        tl.store(s_ptr + rows, qs, mask=rmask)
        inv = (1.0 / qs)[:, None, None]
        qrow = q_ptr + rows[:, None, None].to(tl.int64) * stride_q
        for t in range(0, nch):
            if REV_Q:
                k0 = (nch - 1 - t) * W
            else:
                k0 = t * W
            c = k0 + wc
            m2 = rmask[:, None] & (c < K)[None, :]
            v = _load_wide(xrow, c, m2, KIND, K, rstd, w_ptr, trow, shift_ptr, scale_ptr, HAS_W, HSTRIDE)
            y0, y1, y2, y3 = _rot_wide(v, bt, BM, NG)
            oc = k0 + gi[None, :, None] * 256 + cols[None, None, :]
            sm = rmask[:, None, None] & (oc < K)
            qp = qrow + oc
            tl.store(qp, tl.extra.cuda.libdevice.rint(tl.minimum(tl.maximum(y0 * inv, -127.0), 127.0)).to(tl.int8), mask=sm)
            tl.store(qp + 64, tl.extra.cuda.libdevice.rint(tl.minimum(tl.maximum(y1 * inv, -127.0), 127.0)).to(tl.int8), mask=sm)
            tl.store(qp + 128, tl.extra.cuda.libdevice.rint(tl.minimum(tl.maximum(y2 * inv, -127.0), 127.0)).to(tl.int8), mask=sm)
            tl.store(qp + 192, tl.extra.cuda.libdevice.rint(tl.minimum(tl.maximum(y3 * inv, -127.0), 127.0)).to(tl.int8), mask=sm)


# Per-kind defaults for _rot_quant_wide_kernel, tuned on the real arguments of an H3 forward (GB10,
# M = 44,455 / 22,310; the modulation tables are fp32 [rows, 5376] views with row stride 32256).
# nprog is in units of SMs.  KIND 2 with ss_exact=False is bit-identical only at bm 16 / num_warps 4
# (the original [16, 64] sum-of-squares reduction depends on that layout); _ss_exact works at any bm.
ROT_CFGS = {
    0: {"bm": 4, "ng": 8, "num_warps": 4, "num_stages": 1, "nprog": 1, "rev_amax": False, "rev_q": True},
    1: {"bm": 4, "ng": 8, "num_warps": 4, "num_stages": 1, "nprog": 2, "rev_amax": False, "rev_q": True},
    2: {"bm": 8, "ng": 2, "num_warps": 4, "num_stages": 1, "nprog": 2, "rev_amax": True, "rev_q": True,
        "ss_exact": True},
    3: {"bm": 8, "ng": 4, "num_warps": 4, "num_stages": 1, "nprog": 1, "rev_amax": False, "rev_q": True},
}
ROT_CFG = {"bm": 16, "num_warps": 4, "num_stages": 2}   # the original one-row-block-per-program kernel
_H64T = {}
_NSM = {}
_STAGES_OK = {}


def _h64t_cached(device):
    key = str(device)
    if key not in _H64T:
        _H64T[key] = _h64t(device)
    return _H64T[key]


def _launch(x, k, kind, eps=0.0, weight=None, idx=None, shift=None, scale=None, cfg=None, hstride=0):
    """cfg None: the per-kind ROT_CFGS (wide, persistent kernel).  A cfg without "ng" runs the
    original _rot_quant_kernel ({"bm", "num_warps", "num_stages"}).  Both give identical bits."""
    assert x.ndim == 2 and x.stride(1) == 1 and k % ROT_GROUP == 0
    m = x.shape[0]
    q = torch.empty((m, k), device=x.device, dtype=torch.int8)
    s = torch.empty((m, 1), device=x.device, dtype=torch.float32)
    dummy = s
    args = (x, _h64t_cached(x.device), q, s, m, k, x.stride(0), q.stride(0), float(eps),
            weight if weight is not None else dummy, idx if idx is not None else dummy,
            shift if shift is not None else dummy, scale if scale is not None else dummy,
            shift.stride(0) if shift is not None else 0, hstride)
    if cfg is not None and not cfg.get("ng"):
        c = dict(ROT_CFG, **cfg)
        _rot_quant_kernel[(triton.cdiv(m, c["bm"]),)](
            *args, KIND=kind, HAS_W=weight is not None, BM=c["bm"],
            num_warps=c["num_warps"], num_stages=c["num_stages"])
        return q, s
    c = dict(ROT_CFGS[kind], **(cfg or {}))
    ss_exact = c.get("ss_exact", True) or not (c["bm"] == 16 and c["num_warps"] == 4)
    if _STAGES_OK.get((kind, k, weight is not None, x.dtype, None if shift is None else shift.dtype,
                       str(sorted(c.items())))) == 0:
        return _launch(x, k, kind, eps, weight, idx, shift, scale, cfg=dict(ROT_CFG), hstride=hstride)
    dkey = x.device.index if x.device.index is not None else torch.cuda.current_device()
    if dkey not in _NSM:
        _NSM[dkey] = torch.cuda.get_device_properties(dkey).multi_processor_count
    nblk = triton.cdiv(m, c["bm"])
    nprog = max(1, min(nblk, int(c["nprog"] * _NSM[dkey])))
    # shared memory depends on the argument dtypes (e.g. fp32 modulation tables): if a config does not
    # fit, retry with fewer pipeline stages, then the original kernel (all variants give the same bits)
    fkey = (kind, k, weight is not None, x.dtype, None if shift is None else shift.dtype, str(sorted(c.items())))
    st = _STAGES_OK.get(fkey, c["num_stages"])
    while st >= 1:
        try:
            _rot_quant_wide_kernel[(nprog,)](
                *args, nprog, KIND=kind, HAS_W=weight is not None, BM=c["bm"], NG=c["ng"],
                REV_AMAX=c["rev_amax"], REV_Q=c["rev_q"], SS_EXACT=ss_exact,
                num_warps=c["num_warps"], num_stages=st)
            _STAGES_OK[fkey] = st
            return q, s
        except triton.runtime.errors.OutOfResources:
            st -= 1
    _STAGES_OK[fkey] = 0
    return _launch(x, k, kind, eps, weight, idx, shift, scale, cfg=dict(ROT_CFG), hstride=hstride)


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
