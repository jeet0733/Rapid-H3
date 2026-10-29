"""How close our Triton rot-quant kernels run to memory bandwidth at the real block shapes."""
import sys, time

import torch

sys.path.insert(0, "/work/custom_nodes/rapid_h3")
import triton_fp8 as tq  # noqa: E402


def timed(fn, iters=10):
    fn(); torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e3


M = 37728
big = torch.empty(2 << 30, dtype=torch.uint8, device="cuda")
ms = timed(lambda: big.clone())
print(f"device copy: {2 * 2 / ms:.0f} GB/s effective (read+write {ms:.2f} ms for 2 GiB)")
x = torch.randn(M, 5376, device="cuda", dtype=torch.bfloat16)
w = torch.ones(5376, device="cuda", dtype=torch.bfloat16)
idx = torch.zeros(M, dtype=torch.int32, device="cuda")
sh = torch.zeros(3, 5376, device="cuda", dtype=torch.bfloat16); sc = sh.clone()
o = torch.randn(56, M, 128, device="cuda", dtype=torch.bfloat16)
h = torch.randn(M, 14336, device="cuda", dtype=torch.bfloat16)
cases = {
    "norm_mod_rot_quant (x 5376, read x3)": (lambda: tq.norm_mod_rot_quant_rows_i8(x, w, 1e-6, idx, sh, sc), 3 * x.numel() * 2 + x.numel()),
    "rot_quant_heads (attn out 7168, read x2)": (lambda: tq.rot_quant_heads_i8(o), 2 * o.numel() * 2 + o.numel()),
    "rot_quant_rows (h 14336, read x2)": (lambda: tq.rot_quant_rows_i8(h), 2 * h.numel() * 2 + h.numel()),
}
for name, (fn, bytes_) in cases.items():
    t = timed(fn)
    print(f"{name}: {t:.2f} ms, {bytes_ / t / 1e6:.0f} GB/s moved; ideal single-read {(bytes_ - (bytes_ // 3 if 'x3' in name else bytes_ // 2) * 0) / 273e9 * 1e3:.2f} ms at 273 GB/s")
