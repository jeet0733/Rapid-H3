"""Sweep the CUTLASS row-wise FP8 GEMM tile configs at H3 shapes; check against the torch path.

python cutlass_fp8_bench.py --tokens 40000 --out ../bench/results/cutlass_fp8_40k.json
"""
import argparse, json, time

import torch

import cutlass_fp8
import kernels
import triton_fp8

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
    ap.add_argument("--configs", type=int, nargs="*", default=list(range(8)))
    ap.add_argument("--out")
    a = ap.parse_args()
    torch.manual_seed(0)
    t = time.time()
    cutlass_fp8.ext()
    print(f"built in {time.time()-t:.0f}s", flush=True)
    m, rows = a.tokens, []
    for name, (k, n) in SHAPES.items():
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        w = (torch.randn(n, k, device="cuda") / k ** 0.5).to(torch.bfloat16)
        ref = x @ w.t()
        qw = kernels.quantize_weight(w, "fp8_rowwise")
        q, s = triton_fp8.quant_rows(x)
        y_torch = kernels.linear(x, qw, "fp8_rowwise")
        flop = 2 * m * k * n
        r = {"gemm": name, "bf16_ms": timed(lambda: x @ w.t()), "quant_ms": timed(lambda: triton_fp8.quant_rows(x)),
             "err_torch_path_vs_bf16": rel(y_torch, ref)}
        for cfg in a.configs:
            for fast in (True, False):
                key = f"c{cfg}{'f' if fast else ''}"
                try:
                    y = cutlass_fp8.fp8_gemm(q, s, qw.q, qw.scale, config=cfg, fast_accum=fast)
                    r[key + "_ms"] = timed(lambda: cutlass_fp8.fp8_gemm(q, s, qw.q, qw.scale, config=cfg, fast_accum=fast))
                    r[key + "_tflops"] = flop / r[key + "_ms"] / 1e9
                    r[key + "_err_vs_torch_path"] = rel(y, y_torch)
                except Exception as e:  # noqa: BLE001
                    r[key + "_err"] = str(e)[:100]
        best = min((v, kk) for kk, v in r.items() if kk.endswith("_ms") and kk.startswith("c"))
        r["best"] = best[1]
        r["speedup_vs_bf16_incl_quant"] = r["bf16_ms"] / (best[0] + r["quant_ms"])
        print(json.dumps({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in r.items()}), flush=True)
        rows.append(r)
        del x, w, ref, qw, q, s, y_torch
        torch.cuda.empty_cache()
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
