"""INT8 (plain and Hadamard-rotated) vs FP8 row-wise linears at H3 shapes: speed and error.

Activations get a few heavy outlier channels, as real DiT inputs do; error is relative
to the bf16 product.   python int8_bench.py --tokens 40000
"""
import argparse, json, sys, time

import torch

sys.path.insert(0, "/work/custom_nodes/rapid_h3")
import cutlass_fp8, kernels, triton_fp8 as tq  # noqa: E402

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


ap = argparse.ArgumentParser()
ap.add_argument("--tokens", type=int, default=40000)
ap.add_argument("--out")
a = ap.parse_args()
torch.manual_seed(0)
cutlass_fp8.ext()
rows = []
for name, (k, n) in SHAPES.items():
    m = a.tokens
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    x[:, torch.randperm(k)[:16]] *= 20
    w = (torch.randn(n, k, device="cuda") / k ** 0.5).to(torch.bfloat16)
    ref = x @ w.t()
    flop = 2 * m * k * n
    r = {"gemm": name, "bf16_ms": timed(lambda: x @ w.t())}
    for mode in ("fp8_rowwise", "int8_fast", "int8_rot"):
        qw = kernels.quantize_weight(w, mode)
        r[f"{mode}_err"] = rel(kernels.linear(x, qw, mode), ref)
        r[f"{mode}_ms"] = timed(lambda: kernels.linear(x, qw, mode))
    xr = tq.rotate(x)
    r["rotate_ms"] = timed(lambda: tq.rotate(x))
    r["quant_i8_ms"] = timed(lambda: tq.quant_rows_i8(xr))
    r["fused_rot_quant_ms"] = timed(lambda: tq.rot_quant_rows_i8(x))
    q1, s1 = tq.quant_rows_i8(xr)
    xq, xs = tq.rot_quant_rows_i8(x)
    r["fused_vs_unfused_q_mismatch"] = (xq != q1).float().mean().item()
    qw = kernels.quantize_weight(w, "int8_rot")
    for cfg in (1, 2, 3, 4):
        r[f"i8_c{cfg}_tflops"] = flop / timed(lambda: cutlass_fp8.int8_gemm(xq, xs, qw.q, qw.scale, config=cfg)) / 1e9
    print(json.dumps({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in r.items()}), flush=True)
    rows.append(r)
    del x, w, ref, xr, xq, xs, qw
    torch.cuda.empty_cache()
if a.out:
    json.dump(rows, open(a.out, "w"), indent=1)
