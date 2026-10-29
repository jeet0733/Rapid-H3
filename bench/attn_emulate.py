"""Torch emulation of INT8 attention error sources on real q/k/v (a few heads), vs fp32 reference.

python attn_emulate.py /work/data/qkv_real/s1_b25.pt --heads 0 9 27 49
"""
import argparse

import torch


def q8(x, dim, group=None):
    """symmetric int8 fake-quant; absmax over `dim` (and over `group` rows along dim 0 if given)."""
    a = x.abs().amax(dim=dim, keepdim=True)
    if group:
        s = a.shape
        a = torch.nn.functional.pad(a, (0, 0, 0, (-a.shape[0]) % group))
        a = a.view(-1, group, *s[1:]).amax(1, keepdim=True).expand(-1, group, *s[1:]).reshape(-1, *s[1:])[: x.shape[0]]
    sc = a.clamp_min(1e-8) / 127
    return (x / sc).round().clamp(-127, 127) * sc


def f8(x, dim):
    sc = x.abs().amax(dim=dim, keepdim=True).clamp_min(1e-8) / 448
    return (x / sc).to(torch.float8_e4m3fn).float() * sc


def hadamard(n, device):
    h = torch.ones(1, 1, device=device)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / n ** 0.5


def attn(q, k, v, kq=None, pmode="exact", tile=128):
    """q,k,v [S,D] fp32 (already fake-quantized as desired). pmode: exact | tile_u8 | run_u8"""
    S = q.shape[0]
    out = torch.empty_like(q)
    for r in range(0, S, 2048):
        s = (q[r:r + 2048] @ k.T) * q.shape[1] ** -0.5
        if pmode == "exact":
            p = torch.softmax(s, -1)
            out[r:r + 2048] = p @ v
            continue
        m = s.amax(-1, keepdim=True).unsqueeze(-1)
        st = torch.nn.functional.pad(s, (0, (-S) % tile), value=-torch.inf).view(s.shape[0], -1, tile)
        tm = st.amax(-1, keepdim=True)
        if pmode == "tile_u8":
            pq = (torch.exp(st - tm) * 255).round()
            w = torch.exp(tm - m) / 255
        elif pmode == "run_u8_sr":  # stochastic rounding (uniform dither)
            pq = (torch.exp(st - m) * 255 + torch.rand_like(st)).floor()
            w = torch.ones_like(tm) / 255
        elif pmode == "run_u8_od":  # ordered dither by key index (golden-ratio sequence)
            j = torch.arange(st.shape[1] * st.shape[2], device=st.device).view(1, st.shape[1], st.shape[2])
            dith = torch.frac(j * 0.6180339887498949 + 0.5 * (torch.arange(st.shape[0], device=st.device).view(-1, 1, 1) * 0.7548776662466927 % 1))
            pq = (torch.exp(st - m) * 255 + dith).floor()
            w = torch.ones_like(tm) / 255
        elif pmode == "run_fp8":  # e4m3 relative to the running (here: final) row max, scaled to 448
            pq = (torch.exp(st - m) * 448).to(torch.float8_e4m3fn).float()
            w = torch.ones_like(tm) / 448
        elif pmode == "run_u8":  # relative to the final row max (upper bound on the online running max)
            pq = (torch.exp(st - m) * 255).round()
            w = torch.ones_like(tm) / 255
        p = (pq * w).view(s.shape[0], -1)[:, :S]
        out[r:r + 2048] = (p @ v) / p.sum(-1, keepdim=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--heads", type=int, nargs="+", default=[0, 9, 27, 49])
    a = ap.parse_args()
    d = torch.load(a.file)
    res = {}
    for h in a.heads:
        q, k, v = (d[n][0, h].cuda().float() for n in ("q", "k", "v"))
        ref = attn(q, k, v)
        km = k.mean(0, keepdim=True)
        vm = v.mean(0, keepdim=True)
        qq = q8(q, 1)
        H = hadamard(128, q.device)
        qr, kr = q @ H, (k - km) @ H
        qqr = q8(qr, 1)
        variants = {
            "qk_tok": lambda: attn(qq, q8(k, 1), v),
            "qk_tok_smooth": lambda: attn(qq, q8(k - km, 1), v),
            "qk_blk128_smooth": lambda: attn(qq, q8(k - km, 1, 128), v),
            "qk_blk64_smooth": lambda: attn(qq, q8(k - km, 1, 64), v),
            "qk_blk128": lambda: attn(qq, q8(k, 1, 128), v),
            "v_ch": lambda: attn(q, k, q8(v, 0)),
            "v_ch_smooth": lambda: attn(q, k, q8(v - vm, 0) + vm),
            "p_tile128": lambda: attn(q, k, v, pmode="tile_u8"),
            "p_tile64": lambda: attn(q, k, v, pmode="tile_u8", tile=64),
            "p_run": lambda: attn(q, k, v, pmode="run_u8"),
            "qk_tok_smooth_rot": lambda: attn(qqr, q8(kr, 1), v),
            "qk_blk128_smooth_rot": lambda: attn(qqr, q8(kr, 1, 128), v),
            "all_rot_tok_tile128": lambda: attn(qqr, q8(kr, 1), q8(v - vm, 0) + vm, pmode="tile_u8"),
            "p_run_sr": lambda: attn(q, k, v, pmode="run_u8_sr"),
            "p_run_od": lambda: attn(q, k, v, pmode="run_u8_od"),
            "all_rot_run_sr": lambda: attn(qqr, q8(kr, 1), q8(v - vm, 0) + vm, pmode="run_u8_sr"),
            "v_fp8_smooth": lambda: attn(q, k, f8(v - vm, 0) + vm),
            "p_fp8_run": lambda: attn(q, k, v, pmode="run_fp8"),
            "fp8pv_rotqk": lambda: attn(qqr, q8(kr, 1), f8(v - vm, 0) + vm, pmode="run_fp8"),
            "all_tok_tile128": lambda: attn(qq, q8(k - km, 1), q8(v - vm, 0) + vm, pmode="tile_u8"),
            "all_blk128_tile128": lambda: attn(qq, q8(k - km, 1, 128), q8(v - vm, 0) + vm, pmode="tile_u8"),
        }
        for name, fn in variants.items():
            o = fn()
            e = ((o - ref).norm() / ref.norm()).item()
            res.setdefault(name, []).append(e)
        print(h, {k: round(v[-1] * 100, 3) for k, v in res.items()}, flush=True)
    print("mean %:", {k: round(sum(v) / len(v) * 100, 3) for k, v in res.items()})


if __name__ == "__main__":
    main()
