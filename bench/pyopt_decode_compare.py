"""Per-frame PSNR / SSIM between raw decoded frames saved by pyopt_decode_bench.py.

    python pyopt_decode_compare.py <dir> <latent stem> ...    # expects <dir>/<stem>.<variant>.pt

For every stem, every variant against fp16 and against int8: PSNR on float frames and on 8-bit
rounded frames (what the mp4 encoder sees), luma SSIM (harness/score.py), mean and worst frame.
"""
import glob, json, math, os, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import ssim  # harness/score.py, copied next to this file  # noqa: E402

DEV = "cuda"
d, stems = sys.argv[1], sys.argv[2:]   # variants: <stem>.{int8,fp16,pyopt,pyopt_int8}.pt, "f" suffix = decode_fusions
rows = []


def per_frame(a, b, q):
    ps, ss = [], []
    for i in range(0, a.shape[0], 8):
        x = a[i:i + 8].to(DEV)
        y = b[i:i + 8].to(DEV)
        if q:
            x, y = (x * 255).round() / 255, (y * 255).round() / 255
        mse = (x - y).pow(2).flatten(1).mean(1)
        ps += [99.0 if m == 0 else 10 * math.log10(1 / m) for m in mse.tolist()]
        if not q:
            ss += [ssim(x[j:j + 1], y[j:j + 1]) for j in range(x.shape[0])]
    return ps, ss


for stem in stems:
    fr = {os.path.basename(p).split(".")[-2]: p for p in glob.glob(os.path.join(d, f"{stem}.*.pt"))}
    cache = {k: torch.load(v) for k, v in fr.items()}
    for ref in ("fp16", "int8", "fp32"):
        if ref not in cache:
            continue
        for var in ("int8", "fp16", "pyopt", "pyopt_int8", "pyoptf", "pyopt_int8f"):
            if var == ref or var not in cache:
                continue
            p, s = per_frame(cache[var], cache[ref], False)
            p8, _ = per_frame(cache[var], cache[ref], True)
            mse = sum(10 ** (-x / 10) for x in p) / len(p)
            r = {"latent": stem, "variant": var, "ref": ref,
                 "psnr_mean": round(sum(p) / len(p), 2), "psnr_global": round(10 * math.log10(1 / mse), 2),
                 "psnr_min": round(min(p), 2), "psnr_min_frame": p.index(min(p)),
                 "psnr8_mean": round(sum(p8) / len(p8), 2), "psnr8_min": round(min(p8), 2),
                 "ssim_mean": round(sum(s) / len(s), 5), "ssim_min": round(min(s), 5),
                 "maxabs": round((cache[var] - cache[ref]).abs().max().item(), 4)}
            rows.append(r)
            print(json.dumps(r), flush=True)
    del cache
with open(os.path.join(d, "compare.json"), "w") as f:
    json.dump(rows, f, indent=1)
