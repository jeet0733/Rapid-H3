"""Triton INT8 GEMMs vs our CUTLASS sm_121 kernels at the H3 block shapes, with the recipe's epilogues.

Checks the claim of github.com/kanthai/gb10-comfyui-tuning (docs/02: a Triton fused-dequant INT8 GEMM at
"88% of peak" on GB10) against what RapidH3LinearQuant actually runs per block:

  qkv       K=5376  N=21504   plain dequant          out = bf16(acc * xs * ws)              CUTLASS cfg 6
  out_proj  K=7168  N=5376    gated residual         r = bf16(fma(bf16(acc*xs*ws), g, r))   CUTLASS gated cfg 6
  fc1       K=5376  N=28672   SwiGLU (half width)    h = bf16(silu(g) * u), g/u bf16          CUTLASS swiglu cfg 5
  fc2       K=14336 N=5376    gated residual                                                CUTLASS gated cfg 6

Variants per shape:
  cutlass  our kernel exactly as the node calls it
  kanthai  their kernel and config (128x128x64, 8 warps, 4 stages, GROUP_M 8, plain dequant epilogue);
           for the fused shapes it is followed by the separate torch op the fusion replaces
  triton   our own Triton kernel with the same epilogue fused, autotuned over tiles / warps / stages
  int_mm   torch._int_mm (cuBLAS, int32 out, no epilogue): a floor, not a usable path

Needs Triton >= 3.7 on sm_121 (the image's 3.5 cannot build int8 tl.dot for sm_121):
  PYTHONPATH=<pip --target dir with triton 3.7>:/work/custom_nodes/rapid_h3 TRITON_CACHE_DIR=<fresh> \
    python gemm_triton.py --tokens 22332 44992 --split 0.8 0.15 0.05 --out results/gemm_triton.json

Result (2026-10-26, results/gemm_triton.json): every Triton variant is bit-identical to CUTLASS,
SwiGLU and gated forms included, but none is faster where it matters. Per block, CUTLASS 92.4 / 179.2 ms
(S = 22,332 / 44,992), our tuned Triton 95.5 / 190.6, kanthai's kernel plus separate epilogue ops
177.1 / 341.6, torch._int_mm without any epilogue 106.0 / 207.6. Only fc2 at 22k is 3% faster in Triton.
"""
import argparse, json, sys, time

import torch
import triton
import triton.language as tl

sys.path.insert(0, "/work/custom_nodes/rapid_h3")
import cutlass_fp8 as cf  # noqa: E402

SHAPES = {"qkv": (5376, 21504), "out_proj": (7168, 5376), "fc1": (5376, 28672), "fc2": (14336, 5376)}
SMEM = 101376  # sm_120/121 max dynamic smem per block


# ---------------------------------------------------------------- kernels
@triton.jit
def _tile(pid, M, N, BM: tl.constexpr, BN: tl.constexpr, G: tl.constexpr):
    npm = tl.cdiv(M, BM)
    npn = tl.cdiv(N, BN)
    gid = pid // (G * npn)
    fm = gid * G
    gs = min(npm - fm, G)
    pm = fm + ((pid % (G * npn)) % gs)
    pn = (pid % (G * npn)) // gs
    return pm, pn


@triton.jit
def _mainloop(a, b, rm, rn, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rk = tl.arange(0, BK)
    rmc = tl.where(rm < M, rm, 0).to(tl.int64)  # clamp instead of masking the K loop (K % BK == 0)
    rnc = tl.where(rn < N, rn, 0).to(tl.int64)
    A = a + rmc[:, None] * K + rk[None, :]
    B = b + rnc[None, :] * K + rk[:, None]
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for _ in range(0, K, BK):
        acc = tl.dot(tl.load(A), tl.load(B), acc, out_dtype=tl.int32)
        A += BK
        B += BK
    return acc


def _cfgs(swiglu=False):
    out = []
    for bm, bn, bk, w, s in [(128, 128, 64, 4, 4), (128, 128, 64, 8, 4), (128, 128, 64, 8, 5), (128, 128, 64, 8, 6),
                             (128, 128, 128, 8, 3), (128, 256, 64, 8, 3), (128, 256, 64, 8, 4),
                             (256, 128, 64, 8, 3), (256, 128, 64, 8, 4), (128, 256, 128, 8, 2),
                             (256, 128, 128, 8, 2), (256, 256, 64, 8, 3), (128, 256, 64, 4, 4), (256, 128, 64, 4, 4)]:
        ncols = bn  # B tile width of the GEMM; the SwiGLU kernel loads two tiles of BN // 2
        if s * (bm + ncols) * bk > SMEM:
            continue
        for g in (8, 16):
            out.append(triton.Config({"BM": bm, "BN": bn if not swiglu else bn // 2, "BK": bk, "G": g},
                                      num_warps=w, num_stages=s))
    return out


@triton.jit
def _gemm_plain(a, b, xs, ws, c, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, G: tl.constexpr):
    pm, pn = _tile(tl.program_id(0), M, N, BM, BN, G)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    acc = _mainloop(a, b, rm, rn, M, N, K, BM, BN, BK)
    out = acc.to(tl.float32) * tl.load(xs + rm, mask=rm < M, other=0.0)[:, None]
    out = out * tl.load(ws + rn, mask=rn < N, other=0.0)[None, :]
    C = c + rm.to(tl.int64)[:, None] * N + rn[None, :]
    tl.store(C, out.to(tl.bfloat16), mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def _gemm_gated(a, b, xs, ws, gate, r, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                G: tl.constexpr):
    pm, pn = _tile(tl.program_id(0), M, N, BM, BN, G)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    acc = _mainloop(a, b, rm, rn, M, N, K, BM, BN, BK)
    y = acc.to(tl.float32) * tl.load(xs + rm, mask=rm < M, other=0.0)[:, None]
    y = (y * tl.load(ws + rn, mask=rn < N, other=0.0)[None, :]).to(tl.bfloat16).to(tl.float32)
    g = tl.load(gate + rn, mask=rn < N, other=0.0).to(tl.float32)
    R = r + rm.to(tl.int64)[:, None] * N + rn[None, :]
    msk = (rm[:, None] < M) & (rn[None, :] < N)
    res = tl.load(R, mask=msk, other=0.0).to(tl.float32)
    tl.store(R, tl.fma(y, g[None, :], res).to(tl.bfloat16), mask=msk)


@triton.jit
def _fast_exp(x):
    # CUDA __expf: ex2.approx.ftz(x * log2(e))
    return tl.inline_asm_elementwise(
        "{ .reg .f32 t; mul.ftz.f32 t, $1, 0f3FB8AA3B; ex2.approx.ftz.f32 $0, t; }",
        "=r,r", [x], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _gemm_swiglu(a, b, xs, ws, h, M, F, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, G: tl.constexpr):
    """h[M, F] = bf16(silu(g) * u); weight b = [gate rows 0..F-1 | up rows F..2F-1] (unpermuted)."""
    pm, pn = _tile(tl.program_id(0), M, F, BM, BN, G)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    rmc = tl.where(rm < M, rm, 0).to(tl.int64)
    rnc = tl.where(rn < F, rn, 0).to(tl.int64)
    A = a + rmc[:, None] * K + rk[None, :]
    Bg = b + rnc[None, :] * K + rk[:, None]
    Bu = Bg + F.to(tl.int64) * K
    ag = tl.zeros((BM, BN), dtype=tl.int32)
    au = tl.zeros((BM, BN), dtype=tl.int32)
    for _ in range(0, K, BK):
        x = tl.load(A)
        ag = tl.dot(x, tl.load(Bg), ag, out_dtype=tl.int32)
        au = tl.dot(x, tl.load(Bu), au, out_dtype=tl.int32)
        A += BK
        Bg += BK
        Bu += BK
    sx = tl.load(xs + rm, mask=rm < M, other=0.0)[:, None]
    gt = ((ag.to(tl.float32) * sx) * tl.load(ws + rn, mask=rn < F, other=0.0)[None, :]).to(tl.bfloat16).to(tl.float32)
    up = ((au.to(tl.float32) * sx) * tl.load(ws + F + rn, mask=rn < F, other=0.0)[None, :]).to(tl.bfloat16).to(tl.float32)
    o = tl.math.div_rn(gt, 1.0 + _fast_exp(-gt)) * up
    H = h + rm.to(tl.int64)[:, None] * F + rn[None, :]
    tl.store(H, o.to(tl.bfloat16), mask=(rm[:, None] < M) & (rn[None, :] < F))


@triton.jit
def _gemm_plain_persist(a, b, xs, ws, c, M, N, K, NSM: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                        BK: tl.constexpr, G: tl.constexpr):
    """Persistent version (one program per SM, tiles strided), flattened loop so the next tile's loads
    can be issued while this tile's epilogue runs (Triton's tutorial-09 pattern)."""
    start = tl.program_id(0)
    ntiles = tl.cdiv(M, BM) * tl.cdiv(N, BN)
    ksteps = K // BK
    tiles_here = (ntiles - start + NSM - 1) // NSM
    tile = start - NSM
    ki = -1
    rk = tl.arange(0, BK)
    pm = start * 0
    pn = start * 0
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for _ in tl.range(0, tiles_here * ksteps, flatten=True):
        ki = tl.where(ki == ksteps - 1, 0, ki + 1)
        if ki == 0:
            tile += NSM
            pm, pn = _tile(tile, M, N, BM, BN, G)
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        rmc = tl.where(rm < M, rm, 0).to(tl.int64)
        rnc = tl.where(rn < N, rn, 0).to(tl.int64)
        A = a + rmc[:, None] * K + (ki * BK + rk)[None, :]
        B = b + rnc[None, :] * K + (ki * BK + rk)[:, None]
        acc = tl.dot(tl.load(A), tl.load(B), acc, out_dtype=tl.int32)
        if ki == ksteps - 1:
            out = acc.to(tl.float32) * tl.load(xs + rm, mask=rm < M, other=0.0)[:, None]
            out = out * tl.load(ws + rn, mask=rn < N, other=0.0)[None, :]
            C = c + rm.to(tl.int64)[:, None] * N + rn[None, :]
            tl.store(C, out.to(tl.bfloat16), mask=(rm[:, None] < M) & (rn[None, :] < N))
            acc = tl.zeros((BM, BN), dtype=tl.int32)


_NSM = torch.cuda.get_device_properties(0).multi_processor_count if torch.cuda.is_available() else 48
_persist_at = triton.autotune(_cfgs(), key=["M", "N", "K"])(_gemm_plain_persist)


def tri_plain_persist(xq, xs, wq, ws):
    M, K = xq.shape
    N = wq.shape[0]
    c = torch.empty(M, N, device=xq.device, dtype=torch.bfloat16)
    _persist_at[(_NSM,)](xq, wq, xs, ws, c, M, N, K, NSM=_NSM)
    return c


_plain_at = triton.autotune(_cfgs(), key=["M", "N", "K"])(_gemm_plain)
_gated_at = triton.autotune(_cfgs(), key=["M", "N", "K"], restore_value=["r"])(_gemm_gated)
_swiglu_at = triton.autotune(_cfgs(swiglu=True), key=["M", "F", "K"])(_gemm_swiglu)


def _grid(M, N):
    return lambda m: (triton.cdiv(M, m["BM"]) * triton.cdiv(N, m["BN"]),)


def tri_plain(xq, xs, wq, ws, kern=_plain_at, **kw):
    M, K = xq.shape
    N = wq.shape[0]
    c = torch.empty(M, N, device=xq.device, dtype=torch.bfloat16)
    kern[_grid(M, N) if not kw else (triton.cdiv(M, kw["BM"]) * triton.cdiv(N, kw["BN"]),)](
        xq, wq, xs, ws, c, M, N, K, **kw)
    return c


def tri_gated(xq, xs, wq, ws, gate, resid):
    M, K = xq.shape
    N = wq.shape[0]
    _gated_at[_grid(M, N)](xq, wq, xs, ws, gate, resid, M, N, K)
    return resid


def tri_swiglu(xq, xs, wq, ws):
    M, K = xq.shape
    F = wq.shape[0] // 2
    h = torch.empty(M, F, device=xq.device, dtype=torch.bfloat16)
    _swiglu_at[_grid(M, F)](xq, wq, xs, ws, h, M, F, K)
    return h


KANTHAI = dict(BM=128, BN=128, BK=64, G=8, num_warps=8, num_stages=4)


def kanthai(xq, xs, wq, ws):
    return tri_plain(xq, xs, wq, ws, kern=_gemm_plain, **KANTHAI)


# ---------------------------------------------------------------- bench
def timed(fn, iters=10, rounds=3):
    """Median over rounds of the mean CUDA-event time (ms) of `iters` back-to-back calls."""
    fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(rounds):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e) / iters)
    return sorted(out)[len(out) // 2]


def cmp(y, ref):
    d = (y.float() - ref.float()).abs()
    return {"equal": bool(torch.equal(y, ref)), "max_abs": d.max().item(),
            "n_diff": int((y != ref).sum().item())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+", default=[22332, 44992])
    ap.add_argument("--shapes", nargs="+", default=list(SHAPES))
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--split", type=float, nargs="*", default=[],
                    help="also time the gated GEMMs split into row segments of these fractions (one call per modality)")
    ap.add_argument("--out")
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = "cuda"
    rows = []
    for m in a.tokens:
        for name in a.shapes:
            k, n = SHAPES[name]
            fl = 2 * m * k * n
            xq = torch.randint(-127, 128, (m, k), device=dev, dtype=torch.int8)
            wq = torch.randint(-127, 128, (n, k), device=dev, dtype=torch.int8)
            # realistic magnitudes: x ~ O(1) bf16 activations, int8 codes scaled per row/col
            xs = (torch.rand(m, 1, device=dev) + 0.5) * (4.0 / 127)
            ws = (torch.rand(1, n, device=dev) + 0.5) * (0.05 / 127)
            r = {"M": m, "gemm": name}
            t = {}
            if name == "qkv":
                ref = cf.int8_gemm(xq, xs, wq, ws, config=6)
                t["cutlass"] = timed(lambda: cf.int8_gemm(xq, xs, wq, ws, config=6), a.iters)
                y = kanthai(xq, xs, wq, ws)
                r["kanthai_vs_cutlass"] = cmp(y, ref)
                t["kanthai"] = timed(lambda: kanthai(xq, xs, wq, ws), a.iters)
                y = tri_plain(xq, xs, wq, ws)
                r["triton_vs_cutlass"] = cmp(y, ref)
                t["triton"] = timed(lambda: tri_plain(xq, xs, wq, ws), a.iters)
                r["triton_cfg"] = str(_plain_at.best_config)
                try:
                    y = tri_plain_persist(xq, xs, wq, ws)
                    r["triton_persist_vs_cutlass"] = cmp(y, ref)
                    t["triton_persist"] = timed(lambda: tri_plain_persist(xq, xs, wq, ws), a.iters)
                    r["triton_persist_cfg"] = str(_persist_at.best_config)
                except Exception as e:  # noqa: BLE001
                    r["triton_persist_error"] = str(e)[:200]
            elif name in ("out_proj", "fc2"):
                gate = (torch.randn(n, device=dev) * 0.5).to(torch.bfloat16)
                resid0 = torch.randn(m, n, device=dev).to(torch.bfloat16)
                ref = cf.int8_gemm_gated_residual(xq, xs, wq, ws, gate, resid0.clone(), config=6)
                rr = resid0.clone()
                t["cutlass"] = timed(lambda: cf.int8_gemm_gated_residual(xq, xs, wq, ws, gate, rr, config=6), a.iters)

                def kan_gated(res):
                    return res.addcmul_(kanthai(xq, xs, wq, ws), gate)
                r["kanthai_vs_cutlass"] = cmp(kan_gated(resid0.clone()), ref)
                t["kanthai"] = timed(lambda: kan_gated(rr), a.iters)
                t["kanthai_gemm_only"] = timed(lambda: kanthai(xq, xs, wq, ws), a.iters)
                tri_gated(xq, xs, wq, ws, gate, resid0.clone())  # autotune (restores r)
                r["triton_vs_cutlass"] = cmp(tri_gated(xq, xs, wq, ws, gate, resid0.clone()), ref)
                t["triton"] = timed(lambda: tri_gated(xq, xs, wq, ws, gate, rr), a.iters)
                r["triton_cfg"] = str(_gated_at.best_config)
                if a.split:
                    cuts = [0]
                    for f in a.split[:-1]:
                        cuts.append(cuts[-1] + int(round(f * m)))
                    cuts.append(m)
                    seg = list(zip(cuts[:-1], cuts[1:]))

                    def split_run(fn):
                        for s0, s1 in seg:
                            fn(xq[s0:s1], xs[s0:s1], wq, ws, gate, rr[s0:s1])
                    t["cutlass_split"] = timed(lambda: split_run(
                        lambda *z: cf.int8_gemm_gated_residual(*z, config=6)), a.iters)
                    split_run(tri_gated)  # autotune the segment sizes
                    t["triton_split"] = timed(lambda: split_run(tri_gated), a.iters)
                    r["split"] = seg
                del resid0, rr
            else:  # fc1 + SwiGLU
                perm = cf.swiglu_perm(n, dev)
                wqp, wsp = wq[perm].contiguous(), ws[:, perm].contiguous()
                ref = cf.int8_gemm_swiglu(xq, xs, wqp, wsp, config=5)
                t["cutlass"] = timed(lambda: cf.int8_gemm_swiglu(xq, xs, wqp, wsp, config=5), a.iters)

                def kan_swiglu():
                    y = kanthai(xq, xs, wq, ws)
                    f = y.shape[1] // 2
                    g, u = y[:, :f].float(), y[:, f:].float()
                    return (g * torch.sigmoid(g) * u).to(torch.bfloat16)
                r["kanthai_vs_cutlass"] = cmp(kan_swiglu(), ref)
                t["kanthai"] = timed(kan_swiglu, a.iters)
                t["kanthai_gemm_only"] = timed(lambda: kanthai(xq, xs, wq, ws), a.iters)
                y = tri_swiglu(xq, xs, wq, ws)
                r["triton_vs_cutlass"] = cmp(y, ref)
                t["triton"] = timed(lambda: tri_swiglu(xq, xs, wq, ws), a.iters)
                r["triton_cfg"] = str(_swiglu_at.best_config)
                del wqp
            t["int_mm"] = timed(lambda: torch._int_mm(xq, wq.t()), a.iters)
            r["ms"] = {k_: round(v, 3) for k_, v in t.items()}
            r["TOPS"] = {k_: round(fl / v / 1e9, 1) for k_, v in t.items()}
            print(json.dumps(r), flush=True)
            rows.append(r)
            del xq, wq, ref
            torch.cuda.empty_cache()
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
