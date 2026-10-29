"""Phase 0 microbenchmarks at MiniMax-H3 shapes on one GB10.

GEMMs: bf16 (cuBLAS), FP8 e4m3 _scaled_mm (tensorwise and rowwise), INT8 _int_mm and
comfy-kitchen mm_int8.  Attention: non-causal, B=1, H=56, D=128, through each SDPA backend
and comfy-kitchen int8_attention, with output cosine against the bf16 flash reference.

Usage: python microbench.py [--tokens 40000] [--out results.json]
"""
import argparse, json, time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

HIDDEN, HEADS, HEAD_DIM, FFN = 5376, 56, 128, 14336


def timed(fn, iters, warmup=1, budget_s=20.0):
    # One timed probe first; a kernel that falls back to a slow path gets one sample, not a hang.
    torch.cuda.synchronize()
    t = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    probe = time.perf_counter() - t
    if probe * (iters + warmup) > budget_s:
        return probe
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters


def cosine(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def bench_gemms(m, iters):
    shapes = {
        "qkv": (HIDDEN, 3 * HIDDEN),
        "out_proj": (HIDDEN, HIDDEN),
        "fc1": (HIDDEN, 2 * FFN),
        "fc2": (FFN, HIDDEN),
    }
    rows = []
    for name, (k, n) in shapes.items():
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k**0.5
        flop = 2 * m * k * n
        ref = x @ w.t()
        row = {"gemm": name, "m": m, "k": k, "n": n}

        print("gemm", name, flush=True)
        row["bf16_ms"] = timed(lambda: x @ w.t(), iters) * 1e3

        xf, wf = x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)
        one = torch.ones((), device="cuda")
        try:
            row["fp8_tensorwise_ms"] = timed(
                lambda: torch._scaled_mm(xf, wf.t(), one, one, out_dtype=torch.bfloat16), iters) * 1e3
        except Exception as e:  # noqa: BLE001
            row["fp8_tensorwise_err"] = str(e)[:120]
        try:
            sa = torch.ones(m, 1, device="cuda")
            sb = torch.ones(1, n, device="cuda")
            row["fp8_rowwise_ms"] = timed(
                lambda: torch._scaled_mm(xf, wf.t(), sa, sb, out_dtype=torch.bfloat16), iters) * 1e3
        except Exception as e:  # noqa: BLE001
            row["fp8_rowwise_err"] = str(e)[:120]

        xi = torch.randint(-127, 127, (m, k), device="cuda", dtype=torch.int8)
        wi = torch.randint(-127, 127, (n, k), device="cuda", dtype=torch.int8)
        try:
            row["int8_torch_ms"] = timed(lambda: torch._int_mm(xi, wi.t()), iters) * 1e3
        except Exception as e:  # noqa: BLE001
            row["int8_torch_err"] = str(e)[:120]
        try:
            import comfy_kitchen as ck
            row["int8_ck_ms"] = timed(lambda: ck.mm_int8(xi, wi.t()), iters) * 1e3
        except Exception as e:  # noqa: BLE001
            row["int8_ck_err"] = str(e)[:120]

        for key in [k_ for k_ in row if k_.endswith("_ms")]:
            row[key.replace("_ms", "_tflops")] = flop / (row[key] * 1e-3) / 1e12
        print(json.dumps(row), flush=True)
        rows.append(row)
        del x, w, xf, wf, xi, wi, ref
        torch.cuda.empty_cache()
    return rows


def bench_attention(s, iters):
    q, k, v = (torch.randn(1, HEADS, s, HEAD_DIM, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    flop = 4 * s * s * HEAD_DIM * HEADS
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        ref = F.scaled_dot_product_attention(q, k, v)
    rows = []

    def add(name, fn):
        try:
            out = fn()
            ms = timed(fn, iters, warmup=1) * 1e3
            rows.append({"attn": name, "s": s, "ms": ms, "tflops": flop / (ms * 1e-3) / 1e12,
                         "cos_vs_flash": cosine(out, ref)})
            print(json.dumps(rows[-1]), flush=True)
        except Exception as e:  # noqa: BLE001
            rows.append({"attn": name, "s": s, "err": str(e)[:160]})
        torch.cuda.empty_cache()

    for name, be in [("sdpa_flash", SDPBackend.FLASH_ATTENTION),
                     ("sdpa_cudnn", SDPBackend.CUDNN_ATTENTION),
                     ("sdpa_efficient", SDPBackend.EFFICIENT_ATTENTION)]:
        def run(be=be):
            with sdpa_kernel(be):
                return F.scaled_dot_product_attention(q, k, v)
        add(name, run)
    try:
        import comfy_kitchen as ck
        add("ck_int8_attention", lambda: ck.int8_attention(q, k, v))
        add("ck_sage_attention", lambda: ck.sage_attention(q, k, v))
    except Exception as e:  # noqa: BLE001
        rows.append({"attn": "comfy_kitchen", "err": str(e)[:160]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=40000)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--out", default="microbench.json")
    a = ap.parse_args()
    torch.manual_seed(0)
    info = {"torch": torch.__version__, "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(), "cc": torch.cuda.get_device_capability()}
    res = {"info": info, "gemm": bench_gemms(a.tokens, a.iters),
           "attention": bench_attention(a.tokens, max(2, a.iters // 2))}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
