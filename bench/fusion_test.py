"""Exactness + speed of the step-1 fusions against the eager path they replace."""
import sys, time

import torch

sys.path.insert(0, "/work/custom_nodes/rapid_h3")
import cutlass_fp8 as cf, triton_fp8 as tq  # noqa: E402


def timed(fn, iters=10):
    fn(); torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e3


torch.manual_seed(0)
M, H = 37728, 56
# 1) head-major rot-quant (attention output without the transpose copy)
o = torch.randn(H, M, 128, device="cuda", dtype=torch.bfloat16)
q1, s1 = tq.rot_quant_rows_i8(o.transpose(0, 1).reshape(M, H * 128))
q2, s2 = tq.rot_quant_heads_i8(o)
print("rot_quant_heads: q equal", torch.equal(q1, q2), "s equal", torch.equal(s1, s2))
print("  eager transpose+rot_quant %.2f ms, head-major %.2f ms" % (
    timed(lambda: tq.rot_quant_rows_i8(o.transpose(0, 1).reshape(M, H * 128))), timed(lambda: tq.rot_quant_heads_i8(o))))

# 2) gated residual in the GEMM epilogue
for name, K, N in (("out_proj", 7168, 5376), ("fc2", 14336, 5376)):
    xq = torch.randint(-127, 128, (M, K), device="cuda", dtype=torch.int8)
    wq = torch.randint(-127, 128, (N, K), device="cuda", dtype=torch.int8)
    xs = torch.rand(M, 1, device="cuda") * 1e-4
    ws = torch.rand(1, N, device="cuda") * 1e-3
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    gate = (torch.randn(N, device="cuda") * 0.3).to(torch.bfloat16)
    segs = [(0, 30000), (30000, 37000), (37000, M)]
    ref = x.clone()
    for cfg in (3, 4, 6):
        y = cf.int8_gemm(xq, xs, wq, ws, config=cfg)
        r = x.clone()
        for a, b in segs:
            r[a:b].addcmul_(y[a:b], gate)
        f = x.clone()
        for a, b in segs:
            cf.int8_gemm_gated_residual(xq[a:b], xs[a:b], wq, ws, gate, f[a:b], config=cfg)
        d = (r.float() - f.float()).abs()
        neq = (r != f).float().mean().item()
        print(f"{name} cfg{cfg}: mismatch frac {neq:.2e}, max abs diff {d.max().item():.3g}, "
              f"rel L2 {(d.norm() / r.float().norm()).item():.2e}")
    cfg = 6

    def eager():
        y = cf.int8_gemm(xq, xs, wq, ws, config=cfg)
        for a, b in segs:
            ref[a:b].addcmul_(y[a:b], gate)

    def fused():
        for a, b in segs:
            cf.int8_gemm_gated_residual(xq[a:b], xs[a:b], wq, ws, gate, ref[a:b], config=cfg)
    print(f"  {name}: eager gemm+addcmul {timed(eager):.2f} ms, fused {timed(fused):.2f} ms")
