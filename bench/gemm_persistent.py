"""Persistent INT8 GEMM (csrc/int8_gemm_p.cu) vs the CUTLASS kernel (config 6): bit-exactness and TFLOPS
at the H3 block shapes.

python gemm_persistent.py --tokens 17700 36300 [--out results/gemm_persistent.json]
"""
import argparse, json, os, sys, time

import torch
from torch.utils.cpp_extension import load

sys.path.insert(0, "/work/custom_nodes/rapid_h3")
import cutlass_fp8  # noqa: E402

SHAPES = {"qkv": (5376, 21504), "out_proj": (7168, 5376), "fc1": (5376, 28672), "fc2": (14336, 5376)}
HERE = "/work/custom_nodes/rapid_h3"


def ext():
    build = os.path.join(os.environ.get("RAPIDH3_BUILD", "/tmp"), "gemm_p")
    os.makedirs(build, exist_ok=True)
    return load(name="rapidh3_int8_gemm_p", sources=[f"{HERE}/csrc/int8_gemm_p.cu"], build_directory=build,
                extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_121a,code=sm_121a", "-lineinfo"])


def timed(fn, iters=10):
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e3


ap = argparse.ArgumentParser()
ap.add_argument("--tokens", type=int, nargs="+", default=[17700, 36300])
ap.add_argument("--grid", type=int, nargs="+", default=[0])
ap.add_argument("--out")
a = ap.parse_args()
mod = ext()
torch.manual_seed(0)
rows = []
for m in a.tokens:
    for name, (k, n) in SHAPES.items():
        xq = torch.randint(-127, 128, (m, k), device="cuda", dtype=torch.int8)
        wq = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
        xs = torch.rand(m, 1, device="cuda") * 1e-3
        ws = torch.rand(1, n, device="cuda") * 1e-3
        fl = 2 * m * k * n
        ref = cutlass_fp8.int8_gemm(xq, xs, wq, ws, config=6)
        r = {"M": m, "gemm": name, "cutlass6": round(fl / timed(lambda: cutlass_fp8.int8_gemm(xq, xs, wq, ws, config=6)) / 1e9, 1)}
        for gd in a.grid:
            y = mod.int8_gemm_persistent(xq, wq, xs.reshape(-1), ws.reshape(-1), gd)
            r[f"persist_g{gd}"] = round(fl / timed(lambda: mod.int8_gemm_persistent(xq, wq, xs.reshape(-1), ws.reshape(-1), gd)) / 1e9, 1)
            if gd >= 0:  # grid < 0 skips the epilogue (main-loop-only timing)
                r[f"equal_g{gd}"] = bool(torch.equal(y, ref))
        print(json.dumps(r), flush=True)
        rows.append(r)
        del xq, wq, ref
if a.out:
    json.dump(rows, open(a.out, "w"), indent=1)
