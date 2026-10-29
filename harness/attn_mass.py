#!/usr/bin/env python3
"""Analyse RapidH3AttnMass dumps (phase 1).

    python3 harness/feature_test.py YOUR_JOBS.json --name mass1   # (your own jobs file) ts_s6 + workflow attn_mass=
    venv_eval/bin/python harness/attn_mass.py data/output/mass1/er41 [--theta 0.99 0.995] [--frozen 0 01]

Per (block, step): with the always-dense rules (every query/key block touching a non-video token, i.e. text, cond
(first-frame) and audio rows; optionally the first video latent frame; the diagonal), the per-query-block smallest
key-block set covering theta of the mass; its density (kept / all blocks, the attention cost fraction) and, for a mask
built on step(s) F and frozen, the mass it recalls on each later step (mean, 1st percentile, min over video query
rows x heads).
"""
import argparse, glob, os, re

import torch

BLK = 128


def load(d):
    out = {}
    for p in glob.glob(os.path.join(d, "s*_b*.pt")):
        s, b = map(int, re.match(r"s(\d+)_b(\d+)\.pt", os.path.basename(p)).groups())
        out[(s, b)] = torch.load(p)
    return out


def forced(rec, keep_frame0):
    """bool [nb] key/query blocks that stay dense: any non-video token (and the first video latent frame)."""
    nb = rec["mass"].shape[-1]
    va = next(a for a, b, k in rec["segments"] if k == "video")
    lim = va + (rec.get("frame_tokens", 1008) if keep_frame0 else 0)
    idx = torch.arange(nb)
    return idx * BLK < lim  # block starts before the limit -> it touches a forced token


def select(m, f, theta):
    """m [H, nb, nb] fp32 mass (rows sum to 1), f [nb] forced -> kept bool [H, nb, nb]."""
    H, nb, _ = m.shape
    keep = torch.zeros(H, nb, nb, dtype=torch.bool)
    keep[:, f, :] = True
    keep[:, :, f] = True
    keep[:, torch.arange(nb), torch.arange(nb)] = True
    opt = m.masked_fill(keep, -1.0)
    base = (m * keep).sum(-1, keepdim=True)  # forced mass per row
    v, order = opt.sort(-1, descending=True)
    cum = base + v.clamp(min=0).cumsum(-1)
    # number of optional blocks needed: first index where cum >= theta, +1
    need = (cum < theta).sum(-1, keepdim=True) + 1
    need = torch.where(base >= theta, torch.zeros_like(need), need)
    rank = torch.arange(nb).expand(H, nb, nb)
    sel = torch.zeros_like(keep)
    sel.scatter_(-1, order, rank < need)
    return keep | (sel & (opt >= 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--theta", type=float, nargs="+", default=[0.99, 0.995])
    ap.add_argument("--frozen", nargs="+", default=["0", "01"], help="mask source steps (digits), union of masks")
    ap.add_argument("--keep-frame0", action="store_true")
    a = ap.parse_args()
    data = load(a.dir)
    blocks = sorted({b for _, b in data})
    steps = sorted(s for s in {s for s, _ in data} if all((s, b) in data for b in blocks))
    r0 = data[(steps[0], blocks[0])]
    f = forced(r0, a.keep_frame0)
    nb = r0["mass"].shape[-1]
    print(f"{a.dir}: S={r0['S']} nb={nb} forced blocks={int(f.sum())} ({f.float().mean():.1%} of rows/cols) "
          f"segments={[(x, y, k) for x, y, k in r0['segments']]}")
    vid = ~f
    forced_density = 1 - (vid.float().mean() ** 2)
    print(f"density floor from forced rows/cols + diagonal: {forced_density + vid.float().sum() / nb**2:.3f}")
    for theta in a.theta:
        print(f"\n== theta {theta}")
        print("oracle density per step (mask built on that step), by block:")
        print("block " + " ".join(f"  s{s}  " for s in steps))
        dens = {}
        masks = {}
        for b in blocks:
            row = []
            for s in steps:
                m = data[(s, b)]["mass"].float()
                k = select(m, f, theta)
                masks[(s, b)] = k
                dens[(s, b)] = k.float().mean().item()
                row.append(f"{dens[(s, b)]:.3f}")
            print(f"b{b:02d}   " + "  ".join(row))
        print("mean  " + "  ".join(f"{sum(dens[(s, b)] for b in blocks) / len(blocks):.3f}" for s in steps))
        s_last = steps[-1]
        print(f"per-head oracle density on step {s_last} (quantiles over heads min/p25/p50/p75/max; heads <= 0.3):")
        for b in blocks:
            hd = masks[(s_last, b)].float().mean((1, 2))
            qs = torch.quantile(hd, torch.tensor([0, .25, .5, .75, 1.]))
            print(f"  b{b:02d} " + " ".join(f"{x:.2f}" for x in qs) + f"  {int((hd <= 0.3).sum())}/{len(hd)}")
        for fr in a.frozen:
            src = [int(c) for c in fr]
            print(f"frozen mask from steps {src}: density | recall mean / p1 / min on later steps (video rows)")
            for b in blocks:
                k = masks[(src[0], b)].clone()
                for s in src[1:]:
                    k |= masks[(s, b)]
                out = [f"d={k.float().mean():.3f}"]
                for s in steps:
                    if s <= max(src):
                        continue
                    m = data[(s, b)]["mass"].float()
                    rec = (m * k).sum(-1)[:, vid]  # [H, n_video_rows]
                    out.append(f"s{s} {rec.mean():.4f}/{rec.flatten().kthvalue(max(1, rec.numel() // 100)).values:.4f}"
                               f"/{rec.min():.4f}")
                print(f"  b{b:02d} " + "  ".join(out))


if __name__ == "__main__":
    main()
