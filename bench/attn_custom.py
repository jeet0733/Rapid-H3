"""Our INT8 attention vs comfy-kitchen INT8 vs bf16 flash on real H3 q/k/v dumps ([1,H,S,D] bf16).

python attn_custom.py /work/data/qkv_real/*.pt [--configs 0 1 2] [--heads 56]
"""
import argparse, glob, json, time

import torch
import torch.nn.functional as F

import attn_int8


def timed(fn, iters=3):
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e3


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--configs", type=int, nargs="+", default=[0])
    ap.add_argument("--heads", type=int, default=0, help="use only the first N heads (0 = all)")
    ap.add_argument("--ck", action="store_true")
    ap.add_argument("--variants", default="sk_sv_rot", help="comma list of smoothing variants: sk_sv,sv,sk,none")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import comfy_kitchen as ck
    rows = []
    for f in sorted(sum((glob.glob(p) for p in a.files), [])):
        d = torch.load(f)
        q, k, v = (d[n].cuda() for n in ("q", "k", "v"))
        if a.heads:
            q, k, v = q[:, :a.heads], k[:, :a.heads], v[:, :a.heads]
        H, S = q.shape[1], q.shape[2]
        flop = 4 * S * S * 128 * H
        ref = torch.cat([F.scaled_dot_product_attention(q[:, h:h + 4].float(), k[:, h:h + 4].float(), v[:, h:h + 4].float())
                         for h in range(0, H, 4)], dim=1)
        row = {"file": f.split("/")[-1], "S": S, "H": H}

        def rec(name, fn):
            try:
                o = fn()
                ms = timed(fn)
                row[name] = {"err": round(rel(o, ref), 5), "ms": round(ms, 1), "tflops": round(flop / ms / 1e9, 1)}
            except Exception as e:  # noqa: BLE001
                row[name] = {"fail": str(e)[:200]}
            print(name, row[name], flush=True)

        if a.ck:
            rec("ck_int8", lambda: ck.int8_attention(q, k, v))
        for c in a.configs:
            for var in a.variants.split(","):
                sk, sv, rot = "sk" in var, "sv" in var, "rot" in var
                rec(f"ours_c{c}_{var}", lambda: attn_int8.attention_bhsd(q, k, v, config=c, smooth_k=sk, smooth_v=sv, rotate=rot))
        print(json.dumps(row), flush=True)
        rows.append(row)
        del q, k, v, ref
        torch.cuda.empty_cache()
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
