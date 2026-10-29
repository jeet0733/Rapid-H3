"""q/k RMSNorm + RoPE fused into our INT8 attention pre-pass (attn_int8.attention_rr).

Checks, per S and seed, on a synthetic qkv-projection buffer [S, 3*56*128] bf16 with an H3-style rope table
([1, S, 1, 48, 2, 2] bf16, rot_dim 96) and bf16 norm weights:
  1. rms_rope_ref (our re-implementation) == comfy-kitchen rms_rope_split_half_ (in place), torch.equal on q and k;
  2. every pre-pass operand (q8, qs, k8, ks, ksb, vt, vscale, vmean) fused == unfused;
  3. attention output, config 94 fused == ck rms/rope + config 94; the raw buffer is left untouched.
--time adds ms per call (unfused = ck rms/rope + attention) and the pre-pass kernel breakdown (torch profiler).

docker run --rm --gpus all --ipc host -v $PWD:/w -w /w -e PYTHONPATH=/w/custom_nodes/rapid_h3 -e RAPIDH3_BUILD=/w/build \
    rapid-h3/h3-comfyui:v0.42.1 python bench/rmsrope_fuse.py --tokens 1000 4096 22332 44992 --time --seeds 2
"""
import argparse, json

import torch

import attn_int8

H, D, EPS = 56, 128, 1e-5


def rope_table(S, device="cuda", seed=0):
    """comfy/ldm/minimax/model.py rope_freqs + rope_rotation_table (bf16), on random (t, h, w) positions."""
    g = torch.Generator().manual_seed(seed)
    t = torch.randint(0, 40, (S,), generator=g).double() * (5.0 / 3.0)
    hh = torch.rand(S, generator=g, dtype=torch.float64) * 32
    ww = torch.rand(S, generator=g, dtype=torch.float64) * 32
    pos = torch.stack([t, hh, ww], 1).to(torch.float32).to(device)
    inv = 1.0 / (10000.0 ** (torch.arange(16, dtype=torch.float32, device=device) / 16))
    half = torch.cat((pos.unsqueeze(-1) * inv.view(1, 1, -1)).unbind(1), -1)
    a = torch.cat((half, half), -1)[:, :48]
    c, s = torch.cos(a), torch.sin(a)
    return torch.stack([c, -s, s, c], -1).reshape(1, S, 1, 48, 2, 2).to(torch.bfloat16)


def qkv_buffer(S, device="cuda", seed=0):
    """[S, 3*H*D] bf16 like the qkv GEMM output: per-channel scale spread, a per-channel K offset."""
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(S, 3, H, D, device=device, generator=g)
    chs = torch.exp(0.8 * torch.randn(1, 3, H, D, device=device, generator=g))
    off = torch.zeros(1, 3, H, D, device=device)
    off[:, 1] = 2.0 * torch.randn(1, H, D, device=device, generator=g)
    return (x * chs + off).reshape(S, 3 * H * D).to(torch.bfloat16)


def weights(device="cuda", seed=1):
    g = torch.Generator(device=device).manual_seed(seed)
    return tuple((1 + 0.3 * torch.randn(D, device=device, generator=g)).to(torch.bfloat16) for _ in range(2))


def views(buf):
    S = buf.shape[0]
    q, k, v = buf.split(H * D, dim=-1)
    return q.view(1, S, H, D), k.view(1, S, H, D), v.view(S, H, D)


def ck_rmsrope_(buf, tab, qw, kw):
    import comfy_kitchen as ck
    q, k, v = views(buf)
    ck.rms_rope_split_half_(q, k, tab, qw, kw, epsilon=EPS, rot_dim=tab.shape[-3] * 2)
    return q[0], k[0], v


def ev_time(fn, iters=5, warm=2):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); b.synchronize()
        ts.append(a.elapsed_time(b))
    return sorted(ts)[len(ts) // 2]


def prof_kernels(fn, n=3):
    from torch.profiler import ProfilerActivity, profile
    fn(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
    out = {e.key[:70]: round(e.device_time_total / n / 1e3, 3) for e in p.key_averages() if e.device_time_total > 0}
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+", default=[1000, 4096, 22332, 44992])
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--seeds", type=int, default=1)
    a = ap.parse_args()
    X = attn_int8.ext()
    names = ["q8", "qs", "k8", "ks", "ksb", "vt", "vscale", "vmean"]
    with torch.inference_mode():
        for S in a.tokens:
            for seed in range(a.seeds):
                buf, tab = qkv_buffer(S, seed=seed), rope_table(S, seed=seed)
                qw, kw = weights(seed=seed + 1)
                raw = buf.clone()
                q0, k0, v0 = views(raw)
                q0, k0 = q0[0], k0[0]
                yq, yk = X.rms_rope_ref(q0, tab, qw, EPS), X.rms_rope_ref(k0, tab, kw, EPS)
                ref = buf.clone()
                qr, kr, vr = ck_rmsrope_(ref, tab, qw, kw)
                r = dict(rr_q=torch.equal(yq, qr), rr_k=torch.equal(yk, kr))
                pu = X.int8_prepass(qr, kr, vr)
                pf = X.int8_prepass(q0, k0, v0, tab, qw, kw, EPS)
                r["prepass_equal"] = {n: torch.equal(x, y) for n, x, y in zip(names, pu, pf)}
                ou = attn_int8.attention(qr, kr, vr, config=94)
                of = attn_int8.attention_rr(q0, k0, v0, tab, qw, kw, EPS, config=94)
                r["out_equal"] = torch.equal(ou, of)
                r["raw_untouched"] = torch.equal(raw, buf)
                print(S, seed, json.dumps(r), flush=True)
                del pu, pf, ou, of, ref
            if a.time:
                out = torch.empty(S, H, D, device="cuda", dtype=torch.bfloat16)
                work = buf.clone()

                def unfused():
                    q, k, v = ck_rmsrope_(work, tab, qw, kw)
                    attn_int8.attention(q, k, v, config=94, out=out)

                def fused():
                    attn_int8.attention_rr(q0, k0, v0, tab, qw, kw, EPS, config=94, out=out)

                def pre_u():
                    X.int8_prepass(*ck_rmsrope_(work, tab, qw, kw))

                def pre_f():
                    X.int8_prepass(q0, k0, v0, tab, qw, kw, EPS)

                t = {}
                for _ in range(3):
                    for n, f in (("unfused", unfused), ("fused", fused), ("pre_unfused", pre_u), ("pre_fused", pre_f)):
                        t.setdefault(n, []).append(ev_time(f))
                t = {n: round(sorted(v)[1], 3) for n, v in t.items()}
                t["kernels_unfused"], t["kernels_fused"] = prof_kernels(pre_u), prof_kernels(pre_f)
                print(S, "TIME", json.dumps(t, indent=1), flush=True)


if __name__ == "__main__":
    main()
