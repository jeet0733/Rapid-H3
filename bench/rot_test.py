"""H256 ConvRot fused quantizers: agreement with an fp32 torch reference, GEMM error, speed (M=40k)."""
import sys, time
import torch
import torch.nn.functional as F
sys.path.insert(0, "/work/custom_nodes/rapid_h3")
import cutlass_fp8 as cf, kernels, triton_fp8 as tq  # noqa: E402


def timed(fn, iters=5):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / iters * 1e3


def ref_quant(a):
    r = tq.convrot(a.float())
    s = (r.abs().amax(dim=1, keepdim=True) / 127).clamp_min(1e-12)
    return (r / s).round().clamp(-127, 127).to(torch.int8), s


def agree(name, got, ref):
    (q, s), (q0, s0) = got, ref
    d = (q.int() - q0.int()).abs()
    print(f"{name:10s} mismatch {(d > 0).float().mean().item():.4f} max|dq| {d.max().item()} "
          f"scale rel {((s - s0).abs() / s0).max().item():.2e}", flush=True)


torch.manual_seed(0)
m = 40000
x = torch.randn(m, 5376, device="cuda", dtype=torch.bfloat16)
x[:, :8] *= 20
agree("plain", tq.rot_quant_rows_i8(x), ref_quant(x))
print("  plain ms", round(timed(lambda: tq.rot_quant_rows_i8(x)), 2))

h = torch.randn(m, 2 * 14336, device="cuda", dtype=torch.bfloat16)
g, u = h.float().chunk(2, dim=-1)
agree("swiglu", tq.swiglu_rot_quant_rows_i8(h), ref_quant(F.silu(g) * u))
print("  swiglu ms", round(timed(lambda: tq.swiglu_rot_quant_rows_i8(h)), 2))
del h, g, u

w = torch.rand(5376, device="cuda", dtype=torch.bfloat16) + 0.5
tab = torch.randn(9, 6 * 5376, device="cuda", dtype=torch.bfloat16) * 0.1
shift, scale = tab[:, :5376], tab[:, 5376:2 * 5376]
idx = torch.randint(0, 9, (m,), device="cuda", dtype=torch.int32)
eps = 1e-6
xf = x.float()
hn = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()
hn = hn * (1 + scale.float()[idx.long()]) + shift.float()[idx.long()]
agree("norm_mod", tq.norm_mod_rot_quant_rows_i8(x, w, eps, idx, shift, scale), ref_quant(hn))
print("  norm_mod ms", round(timed(lambda: tq.norm_mod_rot_quant_rows_i8(x, w, eps, idx, shift, scale)), 2))

wt = (torch.randn(21504, 5376, device="cuda") / 5376 ** 0.5).to(torch.bfloat16)
qw = kernels.quantize_weight(wt, "int8_rot")
y = kernels.linear(x, qw, "int8_rot")
ref = x.float() @ wt.float().t()
print("gemm rel err vs fp32", ((y.float() - ref).norm() / ref.norm()).item())
