"""One call of comfy-kitchen INT8 attention and of each given config on random [1, H, S, 128] q/k/v (for ncu).

ncu -k regex:'attn_fwd|qk_int_sv' python attn_ncu1.py --tokens 36288 --heads 8 --configs 80
"""
import argparse

import torch

import attn_int8

ap = argparse.ArgumentParser()
ap.add_argument("--tokens", type=int, default=36288)
ap.add_argument("--heads", type=int, default=8)
ap.add_argument("--configs", type=int, nargs="+", default=[80])
ap.add_argument("--no-ck", action="store_true")
a = ap.parse_args()
torch.manual_seed(0)
q, k, v = (torch.randn(1, a.heads, a.tokens, 128, device="cuda", dtype=torch.bfloat16) * 2 for _ in range(3))
if not a.no_ck:
    import comfy_kitchen as ck
    ck.int8_attention(q, k, v)
for c in a.configs:
    attn_int8.attention_bhsd(q, k, v, config=c)
torch.cuda.synchronize()
