"""Correctness and speed of the Triton row-wise FP8 path against bf16 and cuBLAS FP8.

python fp8_gemm_bench.py --tokens 40000   (run with custom_nodes/rapid_h3 on PYTHONPATH)
"""
import argparse, json, time

import torch

import kernels
import triton_fp8 as tf8

SHAPES = {"qkv": (5376, 16128), "out_proj": (5376, 5376), "fc1": (5376, 28672), "fc2": (14336, 5376)}


def timed(fn, iters=5):
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e3


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=40000)
    ap.add_argument("--out")
    a = ap.parse_args()
    torch.manual_seed(0)
    m = a.tokens
    rows = []
    for name, (k, n) in SHAPES.items():
        # activations with a few outlier channels, like real DiT inputs
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        x[:, torch.randperm(k)[:16]] *= 20
        w = (torch.randn(n, k, device="cuda") / k ** 0.5).to(torch.bfloat16)
        ref = x @ w.t()
        qw = kernels.quantize_weight(w, "fp8_rowwise")

        y_torch = kernels.linear(x, qw, "fp8_rowwise")
        q, s = tf8.quant_rows(x)
        y_tri = tf8.fp8_gemm(q, s, qw.q, qw.scale)
        one = torch.ones((), device="cuda")
        xf, wf = x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)
        flop = 2 * m * k * n
        r = {"gemm": name,
             "err_torch_path": rel(y_torch, ref), "err_triton": rel(y_tri, ref),
             "triton_vs_torch_path": rel(y_tri, y_torch),
             "bf16_ms": timed(lambda: x @ w.t()),
             "cublas_fp8_tensorwise_ms": timed(lambda: torch._scaled_mm(xf, wf.t(), one, one, out_dtype=torch.bfloat16)),
             "torch_rowwise_path_ms": timed(lambda: kernels.linear(x, qw, "fp8_rowwise")),
             "triton_quant_ms": timed(lambda: tf8.quant_rows(x)),
             "triton_gemm_ms": timed(lambda: tf8.fp8_gemm(q, s, qw.q, qw.scale))}
        r["triton_total_ms"] = r["triton_quant_ms"] + r["triton_gemm_ms"]
        r["triton_gemm_tflops"] = flop / r["triton_gemm_ms"] / 1e9
        r["speedup_vs_bf16"] = r["bf16_ms"] / r["triton_total_ms"]
        r["best_config"] = str(tf8._fp8_gemm_kernel.best_config)
        print(json.dumps({k_: (round(v, 5) if isinstance(v, float) else v) for k_, v in r.items()}), flush=True)
        rows.append(r)
        del x, w, ref, qw, y_torch, q, s, y_tri, xf, wf
        torch.cuda.empty_cache()
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
