"""INT8 row-wise GEMM tile sweep at the real H3 block shapes (M = tokens).

python gemm_tune.py --tokens 37728 --configs 2 3 4 5 ...
"""
import argparse, json, sys, time

import torch

sys.path.insert(0, "/work/custom_nodes/rapid_h3")
import cutlass_fp8  # noqa: E402

SHAPES = {"qkv": (5376, 21504), "out_proj": (7168, 5376), "fc1": (5376, 28672), "fc2": (14336, 5376)}
VAE_SHAPES = {"vae_qkv": (2048, 6144), "vae_out": (2048, 2048), "vae_w1": (2048, 16384), "vae_w2": (8192, 2048)}


def timed(fn, iters=10):
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e3


ap = argparse.ArgumentParser()
ap.add_argument("--tokens", type=int, default=37728)
ap.add_argument("--configs", type=int, nargs="+", default=list(range(1, 16)))
ap.add_argument("--out")
ap.add_argument("--vae", action="store_true")
a = ap.parse_args()
torch.manual_seed(0)
rows = []
for name, (k, n) in (VAE_SHAPES if a.vae else SHAPES).items():
    m = a.tokens
    xq = torch.randint(-127, 128, (m, k), device="cuda", dtype=torch.int8)
    wq = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
    xs = torch.rand(m, 1, device="cuda") * 1e-3
    ws = torch.rand(1, n, device="cuda") * 1e-3
    ref = None
    r = {"gemm": name}
    for c in a.configs:
        try:
            y = cutlass_fp8.int8_gemm(xq, xs, wq, ws, config=c)
            if ref is None:
                ref = y.float()
            ok = torch.equal(y.float(), ref)
            r[c] = round(2 * m * k * n / timed(lambda: cutlass_fp8.int8_gemm(xq, xs, wq, ws, config=c)) / 1e9, 1)
            if not ok:
                r[f"{c}_mismatch"] = True
        except Exception as e:  # noqa: BLE001
            r[c] = str(e)[:60]
    print(json.dumps(r), flush=True)
    rows.append(r)
if a.out:
    json.dump(rows, open(a.out, "w"), indent=1)
