"""comfy-kitchen's int8 GEMM (its own tile heuristic) vs ours at the VAE decoder shapes."""
import sys, time

import torch

sys.path.insert(0, "/work/custom_nodes/rapid_h3")
import cutlass_fp8 as cf  # noqa: E402
from comfy_kitchen.backends import cuda as ckc  # noqa: E402

W = ckc._wrap_for_dlpack


def timed(fn, iters=20):
    fn(); torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e3


SH = {"qkv": (2048, 6144), "out": (2048, 2048), "w1": (2048, 16384), "w2": (8192, 2048)}
tot_ck = tot_best = 0.0
for m in (7188, 5391):
    for name, (k, n) in SH.items():
        xq = torch.randint(-127, 128, (m, k), device="cuda", dtype=torch.int8)
        wq = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
        xs = torch.rand(m, 1, device="cuda") * 1e-3
        ws = torch.rand(n, device="cuda") * 1e-3
        bias = torch.zeros(n, device="cuda", dtype=torch.float16)
        out = torch.empty(m, n, device="cuda", dtype=torch.float16)
        stream = torch.cuda.current_stream().cuda_stream
        sel = ckc._C.cutlass_int8_selected_config(m, n, k)
        t_ck = timed(lambda: ckc._C.cutlass_int8_dequant(W(xq), W(wq), W(xs), W(ws), W(bias), W(out), 1, stream))
        ours = {c: timed(lambda: cf.int8_gemm(xq, xs, wq, ws.view(1, -1), config=c)) for c in (1, 3, 4, 5, 6)}
        best = min(ours, key=ours.get)
        fl = 2 * m * n * k / 1e9
        print(f"M={m} {name}: ck(cfg {sel}) {t_ck:.3f} ms {fl / t_ck:.0f} TF | ours best cfg{best} {ours[best]:.3f} ms {fl / ours[best]:.0f} TF")
        tot_ck += t_ck; tot_best += min(t_ck, ours[best])
print(f"per layer-call pair: ck {tot_ck:.2f} ms, best-of {tot_best:.2f} ms -> x504 layer-pairs = {(tot_ck - tot_best) * 504 / 1e3:.2f} s saved per decode")
