"""ms/call of comfy-kitchen INT8 attention vs our configs at given token counts (random q/k/v, [1,56,S,128]).

python attn_speed.py --tokens 17700 36300 --configs 2 20 52
"""
import argparse, json, time

import torch

import attn_int8


def timed(fn, iters=5):
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e3


ap = argparse.ArgumentParser()
ap.add_argument("--tokens", type=int, nargs="+", default=[17700, 36300])
ap.add_argument("--configs", type=int, nargs="+", default=[2])
ap.add_argument("--heads", type=int, default=56)
ap.add_argument("--out")
a = ap.parse_args()
import comfy_kitchen as ck
rows = []
torch.manual_seed(0)
for s in a.tokens:
    q, k, v = (torch.randn(1, a.heads, s, 128, device="cuda", dtype=torch.bfloat16) * 2 for _ in range(3))
    flop = 4 * s * s * 128 * a.heads
    r = {"S": s, "ck": round(timed(lambda: ck.int8_attention(q, k, v)), 2)}
    for c in a.configs:
        r[c] = round(timed(lambda: attn_int8.attention_bhsd(q, k, v, config=c)), 2)
    r["tflops_ck"] = round(flop / r["ck"] / 1e9, 1)
    print(json.dumps(r), flush=True)
    rows.append(r)
if a.out:
    json.dump(rows, open(a.out, "w"), indent=1)
