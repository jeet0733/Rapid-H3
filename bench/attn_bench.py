"""Attention kernels on real H3 q/k/v (dumped by RapidH3DumpQKV): speed and error vs an fp32-softmax reference.

python attn_bench.py /data/output/qkv/*.pt
"""
import glob, json, sys, time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


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


def kernels():
    ks = {}

    def flash(q, k, v):
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(q, k, v)
    ks["sdpa_flash_bf16"] = flash
    try:
        import comfy_kitchen as ck
        ks["ck_int8"] = lambda q, k, v: ck.int8_attention(q, k, v)
    except Exception as e:  # noqa: BLE001
        print("no comfy_kitchen:", e)
    try:
        import sageattention as sa
        for acc in ("fp32+fp32", "fp32+fp16", "fp32"):
            ks[f"sage2_int8qk_fp8pv_{acc}"] = (lambda acc: lambda q, k, v: sa.sageattn_qk_int8_pv_fp8_cuda(
                q, k, v, tensor_layout="HND", is_causal=False, pv_accum_dtype=acc))(acc)
        ks["sage2_int8qk_fp8pv_smoothv"] = lambda q, k, v: sa.sageattn_qk_int8_pv_fp8_cuda(
            q, k, v, tensor_layout="HND", is_causal=False, pv_accum_dtype="fp32", smooth_v=True)
    except Exception as e:  # noqa: BLE001
        print("no sageattention:", e)
    try:
        from sageattn3 import sageattn3_blackwell
        ks["sage3_fp4"] = lambda q, k, v: sageattn3_blackwell(q, k, v, is_causal=False)
        ks["sage3_fp4_noblockmean"] = lambda q, k, v: sageattn3_blackwell(q, k, v, is_causal=False, per_block_mean=False)
    except Exception as e:  # noqa: BLE001
        print("no sageattn3:", e)
    return ks


def main():
    files = sorted(sum((glob.glob(p) for p in sys.argv[1:]), []))
    ks = kernels()
    rows = []
    for f in files:
        d = torch.load(f)
        q, k, v = (d[n].cuda() for n in ("q", "k", "v"))
        s = q.shape[2]
        flop = 4 * s * s * q.shape[3] * q.shape[1]
        # reference: fp32 math, heads in chunks to bound memory
        ref = torch.cat([F.scaled_dot_product_attention(q[:, h:h + 4].float(), k[:, h:h + 4].float(), v[:, h:h + 4].float())
                         for h in range(0, q.shape[1], 4)], dim=1)
        row = {"file": f.split("/")[-1], "S": s}
        for name, fn in ks.items():
            try:
                out = fn(q, k, v)
                row[name] = {"err": round(rel(out, ref), 5), "ms": round(timed(lambda: fn(q, k, v)), 1)}
                row[name]["tflops"] = round(flop / row[name]["ms"] / 1e9, 1)
            except Exception as e:  # noqa: BLE001
                row[name] = {"fail": str(e)[:120]}
        print(json.dumps(row), flush=True)
        rows.append(row)
        del q, k, v, ref
        torch.cuda.empty_cache()
    json.dump(rows, open("/work/bench/results/attn_real_h3.json", "w"), indent=1)


if __name__ == "__main__":
    main()
