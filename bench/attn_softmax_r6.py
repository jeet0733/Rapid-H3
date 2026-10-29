"""Attention round 6 (attn_cake/DESIGN.md section 7): checks of configs 90 / 89 / 79
against config 94 on the call paths the presets use.

  1. attention_rr (q/k rms+rope fused into the pre-pass, the 1xx path of ts_h2 / ts_s6): config 90 == config 94
     (torch.equal), and the max difference of 89 / 79 vs 94;
  2. attention_rr_bias (text bias, 1xx path): config 90 == config 94 with and without the bias.
Speed and error vs fp32 on real q/k/v: bench/attn_cake.py --files ... --configs 94 90 89 79.

docker run --rm --gpus all --ipc host -v $PWD:/w -w /w -e PYTHONPATH=/w/custom_nodes/rapid_h3 -e RAPIDH3_BUILD=/w/build \
    rapid-h3/h3-comfyui:v0.42.1 python bench/attn_softmax_r6.py --tokens 1000 4096 22332 44992
"""
import argparse, json, os, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attn_int8  # noqa: E402
from rmsrope_fuse import EPS, qkv_buffer, rope_table, views, weights  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+", default=[1000, 4096, 22332])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    a = ap.parse_args()
    ok = True
    for S in a.tokens:
        for seed in a.seeds:
            buf, tab = qkv_buffer(S, seed=seed), rope_table(S, seed=seed)
            qw, kw = weights(seed=seed + 1)
            q, k, v = views(buf)
            outs = {c: attn_int8.attention_rr(q[0], k[0], v, tab, qw, kw, EPS, config=c) for c in (94, 90, 89, 79)}
            row = {"S": S, "seed": seed, "rr_equal_90": torch.equal(outs[90], outs[94])}
            for c in (89, 79):
                row[f"rr_maxdiff_{c}"] = (outs[c].float() - outs[94].float()).abs().max().item()
            nb = min(300, S // 4)
            for bias in (0.0, 0.7):
                b94 = attn_int8.attention_rr_bias(q[0], k[0], v, tab, qw, kw, EPS, nb, bias, config=94)
                b90 = attn_int8.attention_rr_bias(q[0], k[0], v, tab, qw, kw, EPS, nb, bias, config=90)
                row[f"bias{bias}_equal_90"] = torch.equal(b90, b94)
            ok &= row["rr_equal_90"] and row["bias0.0_equal_90"] and row["bias0.7_equal_90"]
            print(json.dumps(row), flush=True)
            del buf, tab, q, k, v, outs
            torch.cuda.empty_cache()
    print("ALL BIT-IDENTICAL (90 vs 94)" if ok else "MISMATCH")


if __name__ == "__main__":
    main()
