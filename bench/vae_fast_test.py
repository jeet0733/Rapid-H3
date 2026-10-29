"""Bit-exactness of csrc/vae_ops.cu against the comfy-kitchen kernels it replaces."""
import sys, time

import torch

sys.path.insert(0, "/opt/ComfyUI")
sys.path.insert(0, "/work/custom_nodes/rapid_h3")
import vae_fast  # noqa: E402
from comfy_kitchen.backends import cuda as ckc  # noqa: E402

W = ckc._wrap_for_dlpack
st = lambda: torch.cuda.current_stream().cuda_stream


def timed(fn, iters=20):
    fn(); torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e3


torch.manual_seed(0)
B, H, S, D = 4, 32, 1797, 64
o = (torch.randn(B, H, S, D, device="cuda") * 3).half()
o.view(-1)[torch.randint(0, o.numel(), (50,))] = float("nan")
o.view(-1)[torch.randint(0, o.numel(), (50,))] = float("inf")
o.view(-1)[torch.randint(0, o.numel(), (50,))] = float("-inf")


def ck_path():
    x = torch.nan_to_num(o.transpose(1, 2).reshape(B * S, H * D))
    return vae_fast._quantize(x, None, None, 0.0)


q_ref, s_ref = ck_path()
q, s = vae_fast.ext().vae_out_quant(o)
print("out_quant: q equal", torch.equal(q, q_ref), " scales equal", torch.equal(s, s_ref.view(-1)))
print(f"  ck transpose+nan_to_num+quant {timed(ck_path):.3f} ms, fused {timed(lambda: vae_fast.ext().vae_out_quant(o)):.3f} ms")

tot_ck = tot_ours = 0.0
for m in (7188, 5391):
    for name, (k, n, resid) in {"qkv": (2048, 6144, False), "out": (2048, 2048, True),
                                "w1": (2048, 16384, False), "w2": (8192, 2048, True)}.items():
        xq = torch.randint(-127, 128, (m, k), device="cuda", dtype=torch.int8)
        wq = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
        xs = torch.rand(m, device="cuda") * 1e-3
        ws = torch.rand(n, device="cuda") * 1e-3
        bias = (torch.randn(n, device="cuda") * 0.1).half()
        rs = (torch.randn(n, device="cuda") * 0.5).half()
        r = torch.randn(m, n, device="cuda").half()
        ref = torch.empty(m, n, device="cuda", dtype=torch.float16)
        if resid:
            f_ck = lambda: ckc._C.cutlass_int8_dequant_residual(W(xq), W(wq), W(xs), W(ws), W(bias), W(rs), W(r), W(ref), 1, st())
        else:
            f_ck = lambda: ckc._C.cutlass_int8_dequant(W(xq), W(wq), W(xs), W(ws), W(bias), W(ref), 1, st())
        f_ck()
        cfg = vae_fast.gemm_config(m, n, k)
        f_us = lambda: vae_fast.ext().vae_gemm(xq, wq, xs, ws, bias, rs if resid else None, r if resid else None, cfg)
        out = f_us()
        t_ck, t_us = timed(f_ck), timed(f_us)
        tot_ck += t_ck; tot_ours += t_us
        print(f"M={m} {name}: equal {torch.equal(out, ref)}  ck {t_ck:.3f} ms  ours(cfg{cfg}) {t_us:.3f} ms")
print(f"sum over the 8 shapes: ck {tot_ck:.2f} ms, ours {tot_ours:.2f} ms -> x1008 layer-calls each = {(tot_ck - tot_ours) * 1008 / 1e3:.2f} s per decode")
