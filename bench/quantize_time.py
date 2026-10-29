"""Time to quantize all 200 block linears of the H3 DiT (int8_rot) from bf16 on GPU."""
import sys, time
import torch
sys.path.insert(0, "/work/custom_nodes/rapid_h3")
import kernels  # noqa: E402
shapes = [(16128, 5376), (5376, 5376), (28672, 5376), (5376, 14336)]
ws = [torch.randn(n, k, device="cuda", dtype=torch.bfloat16) for n, k in shapes]
kernels.quantize_weight(ws[0], "int8_rot"); torch.cuda.synchronize()
t = time.perf_counter()
for _ in range(50):
    for w in ws:
        kernels.quantize_weight(w, "int8_rot")
torch.cuda.synchronize()
print(f"quantize 200 linears: {time.perf_counter()-t:.1f} s")
