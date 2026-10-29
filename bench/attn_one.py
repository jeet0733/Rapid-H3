"""Run one attention call (for ncu): python attn_one.py file heads config|ck"""
import sys

import torch

import attn_int8

d = torch.load(sys.argv[1])
H = int(sys.argv[2])
q, k, v = (d[n][:, :H].cuda() for n in ("q", "k", "v"))
if sys.argv[3] == "ck":
    import comfy_kitchen as ck
    ck.int8_attention(q, k, v)
else:
    attn_int8.attention_bhsd(q, k, v, config=int(sys.argv[3]))
torch.cuda.synchronize()
