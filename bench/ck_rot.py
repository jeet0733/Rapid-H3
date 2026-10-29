"""Does a Hadamard rotation of q and k (exact for q.k) lower comfy-kitchen INT8 attention error?"""
import sys

import torch
import torch.nn.functional as F

import comfy_kitchen as ck


def hadamard(n, device):
    h = torch.ones(1, 1, device=device)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / n ** 0.5


for f in sys.argv[1:]:
    d = torch.load(f)
    q, k, v = (d[n][:, :16].cuda() for n in ("q", "k", "v"))
    ref = torch.cat([F.scaled_dot_product_attention(q[:, h:h + 4].float(), k[:, h:h + 4].float(), v[:, h:h + 4].float())
                     for h in range(0, q.shape[1], 4)], dim=1)
    H = hadamard(128, q.device)
    qr, kr = (q.float() @ H).bfloat16(), (k.float() @ H).bfloat16()
    km = k.float().mean(2, keepdim=True)
    krs = ((k.float() - km) @ H).bfloat16()
    rel = lambda o: ((o.float() - ref).norm() / ref.norm()).item()
    print(f.split("/")[-1], "ck", round(rel(ck.int8_attention(q, k, v)) * 100, 3),
          "ck_rot", round(rel(ck.int8_attention(qr, kr, v)) * 100, 3),
          "ck_rot_smoothk", round(rel(ck.int8_attention(qr, krs, v)) * 100, 3), flush=True)
