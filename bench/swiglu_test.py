"""Fused SwiGLU+rotate+INT8 quantizer vs eager SwiGLU then rotate+quantize (fc2 input at 40k tokens)."""
import sys, time
import torch
import torch.nn.functional as F
sys.path.insert(0, "/work/custom_nodes/rapid_h3")
import triton_fp8 as tq  # noqa: E402


def timed(fn, iters=5):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / iters * 1e3


x = torch.randn(40000, 2 * 14336, device="cuda", dtype=torch.bfloat16)
def eager():
    g, u = x.chunk(2, dim=-1)
    return tq.rot_quant_rows_i8(F.silu(g).mul_(u))
q0, s0 = eager()
q1, s1 = tq.swiglu_rot_quant_rows_i8(x)
print("q mismatch", (q0 != q1).float().mean().item(), "max |dq|", (q0.int() - q1.int()).abs().max().item(),
      "scale rel", ((s0 - s1).abs() / s0).max().item())
print("eager+rotquant ms", timed(eager), "fused ms", timed(lambda: tq.swiglu_rot_quant_rows_i8(x)))

for bm in (16, 32, 64, 128):
    for nw in (4, 8):
        for st in (1, 2, 3):
            cfg = {"bm": bm, "num_warps": nw, "num_stages": st}
            try:
                print(cfg, round(timed(lambda: tq.swiglu_rot_quant_rows_i8(x, cfg=cfg)), 2), flush=True)
            except Exception as e:
                print(cfg, "fail", str(e)[:60])
