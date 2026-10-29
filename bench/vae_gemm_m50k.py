"""Decoder GEMM tiles at the level-3 shape (28 tiles per call, M = 50316): ck vs our configs, exactness.

Runs inside the ComfyUI image (RAPIDH3_NODE_DIR = a copy of custom_nodes/rapid_h3; CFGS="1 4 5 6 7" picks the tiles).
"""
import os, sys, time
import torch
sys.path.insert(0, "/opt/ComfyUI")
sys.path.insert(0, os.environ.get("RAPIDH3_NODE_DIR", "/work/build/vae_r5/rapid_h3"))
import vae_fast  # noqa
from comfy_kitchen.backends import cuda as ckc
W = ckc._wrap_for_dlpack


def timed(fn, iters=10):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / iters * 1e3


CFGS = [int(c) for c in os.environ.get("CFGS", "1 4 5 6 7").split()]
torch.manual_seed(0)
with torch.inference_mode():
    for m in (50316,):
        for name, (k, n, resid) in {"qkv": (2048, 6144, False), "out": (2048, 2048, True),
                                    "w1": (2048, 16384, False), "w2": (8192, 2048, True)}.items():
            xq = torch.randint(-127, 128, (m, k), device="cuda", dtype=torch.int8)
            wq = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
            xs = torch.rand(m, device="cuda") * 1e-3; ws = torch.rand(n, device="cuda") * 1e-3
            b = (torch.randn(n, device="cuda") * 0.1).half(); rs = (torch.randn(n, device="cuda") * 0.5).half()
            r = torch.randn(m, n, device="cuda").half()
            ref = torch.empty(m, n, device="cuda", dtype=torch.float16)
            sp = torch.cuda.current_stream().cuda_stream
            if resid:
                f_ck = lambda: ckc._C.cutlass_int8_dequant_residual(W(xq), W(wq), W(xs), W(ws), W(b), W(rs), W(r), W(ref), 1, sp)
            else:
                f_ck = lambda: ckc._C.cutlass_int8_dequant(W(xq), W(wq), W(xs), W(ws), W(b), W(ref), 1, sp)
            f_ck(); torch.cuda.synchronize()
            res = {"ck": round(timed(f_ck), 3)}
            eq = {}
            for c in CFGS:
                f = lambda c=c: vae_fast.ext().vae_gemm(xq, wq, xs, ws, b, rs if resid else None, r if resid else None, c)
                eq[c] = torch.equal(f(), ref)
                res[c] = round(timed(f), 3)
            print(m, name, res, "equal", eq, "best", min(res, key=res.get),
                  "cur", vae_fast.gemm_config(m, n, k), flush=True)
            del xq, wq, r, ref
        # SwiGLU epilogue configs (level 3 uses 6)
        k, n = 2048, 16384
        xq = torch.randint(-127, 128, (m, k), device="cuda", dtype=torch.int8)
        wq = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
        xs = torch.rand(m, device="cuda") * 2e-4; ws = torch.full((n,), 3e-4, device="cuda")
        b = (torch.randn(n, device="cuda") * 0.3).half()
        ref = vae_fast.ext().vae_gemm_swiglu(xq, wq, xs, ws, b, 6)
        res = {}
        for c in (1, 5, 6):
            f = lambda c=c: vae_fast.ext().vae_gemm_swiglu(xq, wq, xs, ws, b, c)
            res[c] = (round(timed(f), 3), torch.equal(f(), ref))
        print(m, "w1_swiglu", res, flush=True)
