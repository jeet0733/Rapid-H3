"""Does config 94 lose L2 reuse at long S? Runs config 94 twice per S for ncu.

Inside the image with custom_nodes/rapid_h3 on PYTHONPATH (the docker run line of bench/text_bias.py), under ncu:
ncu --kernel-name regex:attn_cake --launch-skip 1 --launch-count 1 --clock-control none \
    --metrics gpu__time_duration.sum,sm__cycles_elapsed.avg.per_second,lts__t_sector_hit_rate.pct,\
lts__t_sectors_srcunit_tex_lookup_miss.sum python bench/attn_l2probe.py 78592

Result (2026-10-26, synthetic q/k/v, 56 heads):
  S        L2 hit   tex misses (x32 B)   ms (clocks free)   SM clock   TFLOPS   TFLOPS per GHz
  22,332   98.97%   0.59 GB              78.7               2.46 GHz   181.8    73.9
  44,992   99.33%   2.73 GB              327.8              2.37 GHz   177.0    74.7
  78,592   90.77%   73.5 GB              1020               2.29 GHz   173.6    75.8
  With ncu's locked clocks: 157.7 / 159.6 / 161.0 TFLOPS. The hit rate does drop at 79k (one head's K+V^T ~20 MB
  vs 24 MB L2), but work per clock does not: the TFLOPS drop with S is the SM clock (power/thermal over longer
  kernels), so keeping a head's CTAs in lockstep cannot recover it.
"""
import sys

import torch

import attn_int8
from attn_cake import synth

S = int(sys.argv[1])
cfg = int(sys.argv[2]) if len(sys.argv) > 2 else 94
with torch.inference_mode():
    q, k, v = synth(S, 56)
    q, k, v = (t[0].transpose(0, 1) for t in (q, k, v))
    for _ in range(2):
        attn_int8.attention(q, k, v, config=cfg)
        torch.cuda.synchronize()
print("done", S, cfg)
