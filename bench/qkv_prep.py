"""Config 3xx: our INT8 attention's pre-pass partly fused into the qkv GEMM epilogue.

int8_gemm_qkv_prep (csrc/fp8_rowwise.cu, VisitorQkvPrep; 256x128 tiles) writes q8/qs (q RMSNorm + RoPE +
Hadamard + INT8) and the per-256-token K/V statistics partials, so the pre-pass loses stats_partial_rr (a pass
over K and V) and the Q half of quant_qk_rr (and the GEMM no longer writes bf16 q).

Checks per S and seed, against the config-194 chain (int8_gemm config 6 -> attention_rr config 94):
  * k/v columns of the GEMM output torch.equal;
  * all 8 pre-pass operands (q8, qs, k8, ks, ksb, vt, vscale, vmean) torch.equal;
  * attention output torch.equal.
--time: ms per (GEMM + attention) for both chains, plus GEMM alone and pre-pass kernels (torch profiler).

docker run --rm --gpus all --ipc host -v $PWD:/w -w /w -e PYTHONPATH=/w/custom_nodes/rapid_h3 -e RAPIDH3_BUILD=/w/build \
    -e RAPIDH3_CUTLASS=/w/third_party/cutlass rapid-h3/h3-comfyui:v0.42.1 python bench/qkv_prep.py --tokens 1000 4096 22332 44992 --seeds 2 --time

Result (2026-10-26, results/qkv_prep.log): torch.equal on everything at S = 1000, 4096, 22,310,
22,332, 44,455, 44,992 (2 seeds each). Pre-pass 9.5 -> 4.7 ms (22k), 19.1 -> 9.2 ms (45k), but the GEMM goes
25.7 -> 29.1 ms and 51.2 -> 58.1 ms: the epilogue's per-row RMSNorm/RoPE (+ Hadamard/quant for q) is ~300
instructions per row per warp and runs with the tensor pipe idle (1 CTA/SM), while the standalone passes hid the
same work under memory traffic. Net -1.3 ms/call at 22k, -2.7 ms at 45k: ~0.5 s per ts_h2 clip, ~0.8 s per
ts_s6 clip (< 0.5%). Kept as an opt-in kernel pair, not wired into the DiT block.
"""
import argparse, json

import torch

import attn_int8
import cutlass_fp8 as cf
from rmsrope_fuse import rope_table, weights, ev_time, prof_kernels

H, D, EPS, HID = 56, 128, 1e-5, 5376


def operands(S, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    xq = torch.randint(-127, 128, (S, HID), dtype=torch.int8, device="cuda", generator=g)
    wq = torch.randint(-127, 128, (3 * H * D, HID), dtype=torch.int8, device="cuda", generator=g)
    xs = (0.5 + torch.rand(S, 1, device="cuda", generator=g)) * 2e-3
    ws = torch.exp(0.8 * torch.randn(1, 3 * H * D, device="cuda", generator=g)) * 2e-3
    return xq, xs, wq, ws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+", default=[1000, 4096, 22332, 44992])
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--time", action="store_true")
    a = ap.parse_args()
    X = attn_int8.ext()
    names = ["q8", "qs", "k8", "ks", "ksb", "vt", "vscale", "vmean"]
    with torch.inference_mode():
        for S in a.tokens:
            for seed in range(a.seeds):
                xq, xs, wq, ws = operands(S, seed)
                tab = rope_table(S, seed=seed)
                qw, kw = weights(seed=seed + 1)
                buf = cf.int8_gemm(xq, xs, wq, ws, config=6)
                q, k, v = (t.view(S, H, D) for t in buf.split(H * D, dim=-1))
                ref_ops = X.int8_prepass(q, k, v, tab, qw, kw, EPS, config=94)
                ref = attn_int8.attention_rr(q, k, v, tab, qw, kw, EPS, config=94)
                buf2, q8, qs, part = cf.int8_gemm_qkv_prep(xq, xs, wq, ws, tab, qw, kw, EPS, H)
                _, k2, v2 = (t.view(S, H, D) for t in buf2.split(H * D, dim=-1))
                ops = X.int8_prepass_pre(k2, v2, q8, qs, part, tab, kw, EPS, config=94)
                out = attn_int8.attention_pre(k2, v2, q8, qs, part, tab, kw, EPS, config=94)
                r = dict(kv_equal=torch.equal(buf[:, H * D:], buf2[:, H * D:]),
                         operands={n: torch.equal(x, y) for n, x, y in zip(names, ref_ops, ops)},
                         out_equal=torch.equal(ref, out),
                         max_abs_diff=float((ref.float() - out.float()).abs().max()))
                print(S, seed, json.dumps(r), flush=True)
                del ref_ops, ops, ref, out, buf2, q8, qs, part
            if a.time:
                o = torch.empty(S, H, D, device="cuda", dtype=torch.bfloat16)

                def chain194():
                    b = cf.int8_gemm(xq, xs, wq, ws, config=6)
                    q, k, v = (t.view(S, H, D) for t in b.split(H * D, dim=-1))
                    attn_int8.attention_rr(q, k, v, tab, qw, kw, EPS, config=94, out=o)

                def chain394():
                    b, q8, qs, part = cf.int8_gemm_qkv_prep(xq, xs, wq, ws, tab, qw, kw, EPS, H)
                    _, k, v = (t.view(S, H, D) for t in b.split(H * D, dim=-1))
                    attn_int8.attention_pre(k, v, q8, qs, part, tab, kw, EPS, config=94, out=o)

                def pre194():
                    X.int8_prepass(q, k, v, tab, qw, kw, EPS, config=94)

                b2, q8, qs, part = cf.int8_gemm_qkv_prep(xq, xs, wq, ws, tab, qw, kw, EPS, H)
                _, k2, v2 = (t.view(S, H, D) for t in b2.split(H * D, dim=-1))

                def pre394():
                    X.int8_prepass_pre(k2, v2, q8, qs, part, tab, kw, EPS, config=94)

                fns = {"chain194": chain194, "chain394": chain394,
                       "gemm6": lambda: cf.int8_gemm(xq, xs, wq, ws, config=6),
                       "gemm5": lambda: cf.int8_gemm(xq, xs, wq, ws, config=5),
                       "gemm_prep": lambda: cf.int8_gemm_qkv_prep(xq, xs, wq, ws, tab, qw, kw, EPS, H),
                       "pre194": pre194, "pre394": pre394}
                t = {}
                for _ in range(5):
                    for n, f in fns.items():
                        t.setdefault(n, []).append(ev_time(f, iters=3, warm=1))
                t = {n: round(sorted(v)[len(v) // 2], 3) for n, v in t.items()}
                t["saved_ms"] = round(t["chain194"] - t["chain394"], 3)
                t["kernels_pre194"], t["kernels_pre394"] = prof_kernels(pre194), prof_kernels(pre394)
                print(S, "TIME", json.dumps(t, indent=1), flush=True)


if __name__ == "__main__":
    main()
