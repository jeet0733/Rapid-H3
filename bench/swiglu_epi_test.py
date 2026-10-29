"""fc1 + SwiGLU epilogue vs the current path (fc1 GEMM -> swiglu_rot_quant)."""
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
M, K, N2 = 37728, 5376, 28672
xq = torch.randint(-127, 128, (M, K), device="cuda", dtype=torch.int8)
wq = torch.randint(-127, 128, (N2, K), device="cuda", dtype=torch.int8)
xs = torch.rand(M, 1, device="cuda") * 2e-4
ws = torch.rand(1, N2, device="cuda") * 2e-3
perm = cf.swiglu_perm(N2, "cuda")
wqp, wsp = wq[perm].contiguous(), ws[:, perm].contiguous()

y = cf.int8_gemm(xq, xs, wq, ws, config=6)                      # current: [M, 2F] bf16
g, u = y[:, : N2 // 2].float(), y[:, N2 // 2:].float()
h_ref = (g * torch.sigmoid(g) * u).to(torch.bfloat16)
for cfg in (6, 5, 1):
    h = cf.int8_gemm_swiglu(xq, xs, wqp, wsp, config=cfg)
    d = (h.float() - h_ref.float()).abs()
    print(f"cfg{cfg}: h vs bf16(silu(g)*u): mismatch {(h != h_ref).float().mean().item():.2e}, max abs {d.max().item():.3g}, "
          f"rel L2 {(d.norm() / h_ref.float().norm()).item():.2e}")
# downstream: the int8 activations fc2 sees
q_old, s_old = tq.swiglu_rot_quant_rows_i8(y)
q_new, s_new = tq.rot_quant_rows_i8(cf.int8_gemm_swiglu(xq, xs, wqp, wsp))
deq_old, deq_new = q_old.float() * s_old, q_new.float() * s_new
print("fc2 input (dequant) rel diff new vs old: %.2e; int8 codes differing: %.2e" % (
    ((deq_new - deq_old).norm() / deq_old.norm()).item(), (q_old != q_new).float().mean().item()))
t_old = timed(lambda: tq.swiglu_rot_quant_rows_i8(cf.int8_gemm(xq, xs, wq, ws, config=6)))
t_new = timed(lambda: tq.rot_quant_rows_i8(cf.int8_gemm_swiglu(xq, xs, wqp, wsp, config=6)))
print(f"fc1+swiglu+quant: current {t_old:.2f} ms, fused epilogue {t_new:.2f} ms")
for cfg in (6, 5, 1):
    print(f"  swiglu gemm cfg{cfg}: {timed(lambda: cf.int8_gemm_swiglu(xq, xs, wqp, wsp, config=cfg)):.2f} ms"
          f" (plain fc1 gemm cfg6 {timed(lambda: cf.int8_gemm(xq, xs, wq, ws, config=6)):.2f} ms)")
