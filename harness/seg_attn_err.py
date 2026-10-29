#!/usr/bin/env python3
"""Per-segment attention error of our INT8 kernel on dumped q/k/v (RapidH3DumpQKV, which also saves the packed layout's
segments), for Ref2VA reference-audio work. Runs inside the Rapid-H3 container (needs the
attn_int8 extension; free ComfyUI's models first):

  docker exec -i rapid-h3-comfy python - '/data/output/qkv/s*_b*.pt' --mass < harness/seg_attn_err.py

For each dump: the fp32 reference (SDPA on fp32 q/k/v), the bf16 SDPA (the RapidH3Shadow reference's precision) and the
INT8 kernel at the given configs, and per query segment kind (text, ref_img, ref_audio, audio, video, ...) the
relative error ||o - o_ref|| / ||o_ref|| over that segment's rows (video rows sampled); with --mass also, from the
reference, the attention mass each query segment puts on each key segment (mean over rows and heads).
"""
import argparse, glob, json, os, sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/data/custom_nodes")
from rapid_h3 import attn_int8  # noqa: E402


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def sdpa_rows(q, k, v, rows, dtype):
    """attention output for query rows only: q, k, v [H, S, D] -> [H, len(rows), D] (fp32 result)."""
    out = []
    for i in range(0, len(rows), 4096):
        r = rows[i:i + 4096]
        out.append(F.scaled_dot_product_attention(q[:, r].to(dtype)[None], k.to(dtype)[None], v.to(dtype)[None])[0].float())
    return torch.cat(out, 1)


def mass_rows(q, k, segs, rows, scale):
    """mean attention mass of query rows on each key segment (fp32)."""
    acc = {kind: 0.0 for _, _, kind in segs}
    n = 0
    for h in range(q.shape[0]):
        for i in range(0, len(rows), 2048):
            r = rows[i:i + 2048]
            p = torch.softmax((q[h, r].float() @ k[h].float().T) * scale, -1)
            for a, b, kind in segs:
                acc[kind] += p[:, a:b].sum().item()
            n += len(r)
    return {kd: round(v / n, 4) for kd, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--configs", default="94")
    ap.add_argument("--video-rows", type=int, default=8192, help="video query rows sampled (all other kinds: all rows)")
    ap.add_argument("--mass", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = "cuda"
    results = []
    for path in sorted(sum((glob.glob(p) for p in a.dumps), [])):
        d = torch.load(path, map_location="cpu")
        q, k, v = (d[n][0].to(dev) for n in ("q", "k", "v"))  # [H, S, D] bf16
        segs = [tuple(s) for s in d["segments"]]
        H, S, D = q.shape
        scale = D ** -0.5
        g = torch.Generator().manual_seed(0)
        rows_by = {}
        for s0, s1, kind in segs:
            r = torch.arange(s0, s1)
            if kind == "video" and len(r) > a.video_rows:
                r = r[torch.randperm(len(r), generator=g)[:a.video_rows]].sort().values
            rows_by.setdefault(kind, []).append(r)
        rows_by = {kd: torch.cat(rs).to(dev) for kd, rs in rows_by.items()}
        allrows = torch.cat(list(rows_by.values()))
        ref = sdpa_rows(q, k, v, allrows, torch.float32)
        bf = sdpa_rows(q, k, v, allrows, torch.bfloat16)
        outs = {"bf16_sdpa": bf}
        qs, ks, vs = (t.transpose(0, 1) for t in (q, k, v))  # [S, H, D] views
        for c in a.configs.split(","):
            cfg = int(c)
            o = attn_int8.attention(qs, ks, vs, scale, config=cfg).transpose(0, 1)[:, allrows].float()
            outs[f"int8_{cfg}"] = o
        rec = {"dump": os.path.basename(path), "S": S, "segments": {kd: int(sum(b - a_ for a_, b, k_ in segs if k_ == kd))
                                                                     for kd in rows_by}}
        off = 0
        per = {}
        for kd, r in rows_by.items():
            n = len(r)
            per[kd] = {name: round(rel(o[:, off:off + n], ref[:, off:off + n]), 5) for name, o in outs.items()}
            off += n
        rec["err"] = per
        rec["err_all_sampled"] = {name: round(rel(o, ref), 5) for name, o in outs.items()}
        if a.mass:
            rec["mass"] = {kd: mass_rows(q, k, segs, r[:2048], scale) for kd, r in rows_by.items()}
        print(json.dumps(rec), flush=True)
        results.append(rec)
        del q, k, v, ref, bf, outs
        torch.cuda.empty_cache()
    if a.out:
        json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
