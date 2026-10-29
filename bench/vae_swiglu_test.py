"""VAE w1 SwiGLU epilogue (csrc/vae_ops.cu vae_gemm_swiglu) against the stock w1 GEMM + comfy-kitchen's
SwiGLU quantizer: h vs fp16(silu(g)*u), the w2 int8 codes it produces, and the time of w1 + w2-quantizer."""
import os, sys, time

import torch

sys.path.insert(0, "/opt/ComfyUI")
sys.path.insert(0, os.environ.get("RAPIDH3_NODE_DIR", "/work/build/vae_bm/rapid_h3"))
import vae_fast  # noqa: E402


def timed(fn, iters=20):
    fn(); torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e3


with torch.inference_mode():
    torch.manual_seed(0)
    M, K, N = 28 * 1797, 2048, 16384
    xq = torch.randint(-127, 128, (M, K), device="cuda", dtype=torch.int8)
    wq = torch.randint(-127, 128, (N, K), device="cuda", dtype=torch.int8)
    xs = torch.rand(M, device="cuda") * 2e-4
    ws = torch.full((N,), 3e-4, device="cuda")
    bias = (torch.randn(N, device="cuda") * 0.3).half()
    E = vae_fast.ext()
    gu = E.vae_gemm(xq, wq, xs, ws, bias, None, None, 6)                     # stock w1 output [M, N]
    g, u = gu[:, :N // 2].float(), gu[:, N // 2:].float()
    h_ref = ((g / (1.0 + torch.exp(-g))) * u).half()
    il = lambda t: vae_fast._interleave_rows(t, N).contiguous()
    h = E.vae_gemm_swiglu(xq, il(wq), xs, il(ws), il(bias), 6)
    d = (h.float() - h_ref.float()).abs()
    print(f"h vs fp16(silu(g)*u): equal {torch.equal(h, h_ref)}, mismatches {(d > 0).sum().item()} of {h.numel()}, "
          f"max abs {d.max().item():.3g}, |h| max {h.abs().max().item():.3g}")
    q_ref, s_ref = vae_fast._quantize(gu, "swiglu", None, 0.0)                # what ck feeds w2 today
    q, s = vae_fast._quantize(h, None, None, 0.0)
    dq = (q.int() - q_ref.int()).abs()
    print(f"w2 int8 codes: differ {(dq > 0).float().mean().item() * 100:.4f}% (max {dq.max().item()}), "
          f"scales max rel diff {((s - s_ref).abs() / s_ref).max().item():.3g}")
    t_old = timed(lambda: vae_fast._quantize(E.vae_gemm(xq, wq, xs, ws, bias, None, None, 6), "swiglu", None, 0.0))
    wqi, wsi, bi = il(wq), il(ws), il(bias)
    t_new = timed(lambda: vae_fast._quantize(E.vae_gemm_swiglu(xq, wqi, xs, wsi, bi, 6), None, None, 0.0))
    print(f"w1 GEMM + w2 quantizer, M={M}: stock {t_old:.2f} ms, SwiGLU epilogue {t_new:.2f} ms "
          f"({t_old / 28:.3f} -> {t_new / 28:.3f} ms per tile)")
