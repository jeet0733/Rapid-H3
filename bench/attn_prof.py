"""Per-kernel CUDA time of our INT8 attention vs ck on one real dump (torch profiler)."""
import sys

import torch
from torch.profiler import ProfilerActivity, profile

import attn_int8

d = torch.load(sys.argv[1])
H = int(sys.argv[2]) if len(sys.argv) > 2 else 8
configs = [int(c) for c in sys.argv[3].split(",")] if len(sys.argv) > 3 else [3]
q, k, v = (d[n][:, :H].cuda() for n in ("q", "k", "v"))
import comfy_kitchen as ck
fns = {"ck": lambda: ck.int8_attention(q, k, v)}
for c in configs:
    fns[f"ours{c}"] = (lambda c: lambda: attn_int8.attention_bhsd(q, k, v, config=c))(c)
for name, fn in fns.items():
    fn(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
    print("==", name)
    for e in sorted(p.key_averages(), key=lambda e: -e.device_time_total)[:8]:
        print(f"  {e.device_time_total / 3 / 1e3:8.2f} ms  {e.key[:90]}")
