"""Round 4: ms/call and main-kernel ms of comfy-kitchen INT8 attention vs our configs, plus output error
against an fp32 reference (exact attention on the bf16 inputs) and against bf16 SDPA.

python attn_r4.py --tokens 17748 36288 --configs 2 80 81            # random q/k/v, [1, 56, S, 128]
python attn_r4.py --files /data/output/qkv/*.pt --configs 2 80        # real dumps ([1, H, S, 128] bf16)
Timing: CUDA events, calls interleaved round-robin (--rounds x --iters), median per call.
"""
import argparse, glob, json, statistics

import torch
import torch.nn.functional as F

import attn_int8


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def ref_fp32(q, k, v, chunk=4):
    return torch.cat([F.scaled_dot_product_attention(q[:, h:h + chunk].float(), k[:, h:h + chunk].float(),
                                                     v[:, h:h + chunk].float()) for h in range(0, q.shape[1], chunk)], 1)


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
    import comfy_kitchen as ck
    fns = {"ck": lambda: ck.int8_attention(q, k, v)}
    for c in a.configs:
        fns[str(c)] = (lambda c: lambda: attn_int8.attention_bhsd(q, k, v, config=c))(c)
    row = {"case": label, "S": q.shape[2], "H": q.shape[1]}
    if not a.no_err:
        ref = ref_fp32(q, k, v)
        sd = torch.cat([F.scaled_dot_product_attention(q[:, h:h + 8], k[:, h:h + 8], v[:, h:h + 8])
                        for h in range(0, q.shape[1], 8)], 1)
        row["err_bf16sdpa"] = round(rel(sd, ref), 5)
        for n, fn in fns.items():
            o = fn()
            row[f"err_{n}"] = round(rel(o, ref), 5)
            row[f"errsd_{n}"] = round(rel(o, sd), 5)
            del o
        del ref, sd
        torch.cuda.empty_cache()
    if not a.no_time:
        row.update({f"ms_{n}": t for n, t in bench(fns, a.rounds, a.iters).items()})
        if a.kernel:
            row["main_ck"] = main_kernel_ms(fns["ck"], ["qk_int_sv_i8_attn_kernel"])
            for c in a.configs:
                row[f"main_{c}"] = main_kernel_ms(fns[str(c)], ["attn_fwd"])
    print(json.dumps(row), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="*", default=[])
    ap.add_argument("--files", nargs="*", default=[])
    ap.add_argument("--configs", type=int, nargs="+", default=[2])
    ap.add_argument("--heads", type=int, default=56)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--kernel", action="store_true", help="also main-kernel time (torch profiler)")
    ap.add_argument("--no-err", action="store_true")
    ap.add_argument("--no-time", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = []
    torch.manual_seed(0)
    for s in a.tokens:
        q, k, v = (torch.randn(1, a.heads, s, 128, device="cuda", dtype=torch.bfloat16) * 2 for _ in range(3))
        rows.append(run(q, k, v, f"rand{s}", a))
        del q, k, v
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
