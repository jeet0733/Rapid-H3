import sys, time
import torch
sys.path.insert(0, "/work/custom_nodes/rapid_h3")
import vae_fast  # noqa
def timed(fn, iters=20):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / iters * 1e3
for m in (7188, 5391):
    for name, (k, n, resid) in {"qkv": (2048, 6144, False), "out": (2048, 2048, True), "w1": (2048, 16384, False), "w2": (8192, 2048, True)}.items():
        xq = torch.randint(-127, 128, (m, k), device="cuda", dtype=torch.int8); wq = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
        xs = torch.rand(m, device="cuda"); ws = torch.rand(n, device="cuda"); b = torch.zeros(n, device="cuda").half(); rs = b.clone(); r = torch.zeros(m, n, device="cuda").half()
        res = {c: round(timed(lambda: vae_fast.ext().vae_gemm(xq, wq, xs, ws, b, rs if resid else None, r if resid else None, c)), 3) for c in (1, 4, 5, 6, 7)}
        print(m, name, res, "best", min(res, key=res.get), flush=True)
