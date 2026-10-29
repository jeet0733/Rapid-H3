"""Does GB10 run FP16 mma with FP16 accumulation at 2x the FP32-accumulate rate (like GeForce parts)?

Measures cuBLAS fp16/bf16 GEMM with and without reduced-precision accumulation at an H3 shape.
"""
import time

import torch


def timed(fn, iters=5):
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e3


m, k, n = 40000, 5376, 16128
flop = 2 * m * k * n
for dt in (torch.float16, torch.bfloat16):
    x = torch.randn(m, k, device="cuda", dtype=dt)
    w = torch.randn(n, k, device="cuda", dtype=dt) / k ** 0.5
    ref = (x.float() @ w.float().t())
    for acc in (False, True):
        try:
            torch.backends.cuda.matmul.allow_fp16_accumulation = acc
        except AttributeError:
            if acc:
                continue
        ms = timed(lambda: x @ w.t())
        err = ((x @ w.t()).float() - ref).norm() / ref.norm()
        print(f"{str(dt):15s} fp16_accum={acc!s:5s} {ms:7.2f} ms {flop/ms/1e9:6.1f} TFLOPS rel_err {err:.2e}", flush=True)
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    del x, w, ref
    torch.cuda.empty_cache()
