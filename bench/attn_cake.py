"""CAKE-schedule INT8 attention (configs 94-99, csrc/attn_cake) vs config 80 vs comfy-kitchen.

Per case: bit-identity of each new config against config 80, error ||o - o_ref|| / ||o_ref|| (and max abs
error / max |o_ref|) against fp32 attention on the same bf16 inputs, ms per call (CUDA events, calls
interleaved round-robin, median), TFLOPS (4 S^2 D H / t) and optionally main-kernel ms (torch profiler).

python attn_cake.py --tokens 22332 44992 --configs 94 95 [--kernel]          # synthetic q/k/v, [1, 56, S, 128]
python attn_cake.py --files /data/qkv/*.pt --configs 94                       # real dumps (RapidH3DumpQKV)
Synthetic q/k/v mimic H3 after rms-norm + rope: unit-RMS rows, a few outlier channels with a large
per-head mean on K (what K smoothing removes), heavier-tailed V.
"""
import argparse, glob, json, statistics

import torch
import torch.nn.functional as F

import attn_int8


def synth(S, H, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    D = 128
    q = torch.randn(1, H, S, D, device="cuda", generator=g)
    k = torch.randn(1, H, S, D, device="cuda", generator=g)
    # outlier channels: 4 per head, shared offset on K (a per-head channel mean) and larger scale on Q/K
    idx = torch.randint(0, D, (H, 4), device="cuda", generator=g)
    kbias = torch.zeros(1, H, 1, D, device="cuda")
    kbias.scatter_(3, idx.view(1, H, 1, 4), 6.0 * torch.randn(1, H, 1, 4, device="cuda", generator=g))
    boost = torch.ones(1, H, 1, D, device="cuda")
    boost.scatter_(3, idx.view(1, H, 1, 4), 3.0)
    q = q * boost * 1.6
    k = k * boost * 1.6 + kbias
    v = torch.randn(1, H, S, D, device="cuda", generator=g) * (1 + torch.rand(1, H, 1, D, device="cuda", generator=g))
    v = v + 0.05 * torch.randn(1, H, S, D, device="cuda", generator=g).pow(3)
    return q.bfloat16(), k.bfloat16(), v.bfloat16()


def ref_fp32(q, k, v, chunk=4):
    return torch.cat([F.scaled_dot_product_attention(q[:, h:h + chunk].float(), k[:, h:h + chunk].float(),
                                                     v[:, h:h + chunk].float()) for h in range(0, q.shape[1], chunk)], 1)


def errs(o, ref):
    d = (o.float() - ref)
    return round((d.norm() / ref.norm()).item(), 6), round((d.abs().max() / ref.abs().max()).item(), 6)


def main_kernel_ms(fn, name_has):
    from torch.profiler import ProfilerActivity, profile
    fn(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
    ev = [e for e in p.key_averages() if any(s in e.key for s in name_has)]
    return round(sum(e.device_time_total for e in ev) / 3 / 1e3, 2)


def bench(fns, rounds, iters):
    times = {n: [] for n in fns}
    for fn in fns.values():
        fn()
    torch.cuda.synchronize()
    for _ in range(rounds):
        for n, fn in fns.items():
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(iters):
                fn()
            e.record()
            e.synchronize()
            times[n].append(s.elapsed_time(e) / iters)
    return {n: round(statistics.median(t), 2) for n, t in times.items()}


def run(q, k, v, label, a):
    fns = {}
    if not a.no_ck:
        import comfy_kitchen as ck
        fns["ck"] = lambda: ck.int8_attention(q, k, v)
    for c in [80] + a.configs:
        fns[str(c)] = (lambda c: lambda: attn_int8.attention_bhsd(q, k, v, config=c))(c)
    S, H = q.shape[2], q.shape[1]
    row = {"case": label, "S": S, "H": H}
    o80 = fns["80"]()
    for c in a.configs:
        o = fns[str(c)]()
        row[f"bitident_{c}"] = bool(torch.equal(o, o80))
        if not row[f"bitident_{c}"]:
            row[f"maxdiff_vs80_{c}"] = (o.float() - o80.float()).abs().max().item()
        del o
    if not a.no_err:
        ref = ref_fp32(q, k, v)
        for n, fn in fns.items():
            o = fn()
            row[f"err_{n}"], row[f"maxerr_{n}"] = errs(o, ref)
            del o
        del ref
    del o80
    torch.cuda.empty_cache()
    if not a.no_time:
        flop = 4.0 * S * S * 128 * H
        for n, t in bench(fns, a.rounds, a.iters).items():
            row[f"ms_{n}"] = t
            row[f"tflops_{n}"] = round(flop / t / 1e9, 1)
        if a.kernel:
            if "ck" in fns:
                row["main_ck"] = main_kernel_ms(fns["ck"], ["qk_int_sv_i8_attn_kernel"])
            row["main_80"] = main_kernel_ms(fns["80"], ["attn_fwd"])
            for c in a.configs:
                row[f"main_{c}"] = main_kernel_ms(fns[str(c)], ["attn_cake"])
            for n in [x for x in row if x.startswith("main_")]:
                row["tflops_" + n] = round(flop / row[n] / 1e9, 1)
    print(json.dumps(row), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="*", default=[])
    ap.add_argument("--files", nargs="*", default=[])
    ap.add_argument("--configs", type=int, nargs="+", default=[94])
    ap.add_argument("--heads", type=int, default=56)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--kernel", action="store_true", help="also main-kernel time (torch profiler)")
    ap.add_argument("--no-err", action="store_true")
    ap.add_argument("--no-time", action="store_true")
    ap.add_argument("--no-ck", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = []
    for s in a.tokens:
        q, k, v = synth(s, a.heads)
        rows.append(run(q, k, v, f"synth{s}", a))
        del q, k, v
        torch.cuda.empty_cache()
    for f in sorted(sum((glob.glob(p) for p in a.files), [])):
        d = torch.load(f)
        q, k, v = (d[n][:, :a.heads].cuda() for n in ("q", "k", "v"))
        rows.append(run(q, k, v, f.split("/")[-1], a))
        del q, k, v, d
        torch.cuda.empty_cache()
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
