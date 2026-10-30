"""Jenga text-attention bias: checks for attn_int8.attention_rr_bias (config 94 + per-key logit bias).

  save  (run with the build BEFORE the change): attention_rr config 94 outputs -> --ref file
  check (run with the new build):
    1. attention_rr == saved output (torch.equal: the unbiased kernel is unchanged);
    2. attention_rr_bias with nbias 0 == saved output;
    3. attention_rr_bias(nbias, bias) vs an fp32 reference with + bias on keys [0, nbias) (a sample of heads and
       rows), next to the unbiased kernel's error vs the unbiased fp32 reference; and the text keys' attention mass
       with and without the bias (should grow by ~e^bias at small mass);
    4. --time: ms per call, unbiased vs biased.

docker run --rm --gpus all --ipc host -v $PWD:/w -w /w -e PYTHONPATH=/w/custom_nodes/rapid_h3 -e RAPIDH3_BUILD=/w/build \
    rapid-h3/h3-comfyui:v0.42.1 python bench/text_bias.py check --ref /w/ref.pt --tokens 4096 25000 --time
"""
import argparse, math, os, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attn_int8  # noqa: E402
from rmsrope_fuse import D, EPS, H, ck_rmsrope_, ev_time, qkv_buffer, rope_table, views, weights  # noqa: E402


def case(S, seed):
    buf, tab = qkv_buffer(S, seed=seed), rope_table(S, seed=seed)
    qw, kw = weights()
    return buf, tab, qw, kw


def run_rr(buf, tab, qw, kw):
    q, k, v = views(buf)
    return attn_int8.attention_rr(q[0], k[0], v, tab, qw, kw, EPS, config=94)


def run_bias(buf, tab, qw, kw, nbias, bias):
    q, k, v = views(buf)
    return attn_int8.attention_rr_bias(q[0], k[0], v, tab, qw, kw, EPS, nbias, bias, config=94)


def reference(buf, tab, qw, kw, nbias, bias, heads, rows):
    """fp32 softmax(q k^T / sqrt(D) + bias on keys < nbias) v on ck's rms/rope'd q, k; + text mass per row."""
    q, k, v = ck_rmsrope_(buf.clone(), tab, qw, kw)
    outs, mass = [], []
    for h in heads:
        qh, kh, vh = q[rows, h].float(), k[:, h].float(), v[:, h].float()
        lg = qh @ kh.T / math.sqrt(D)
        lg[:, :nbias] += bias
        p = torch.softmax(lg, -1)
        outs.append(p @ vh)
        mass.append(p[:, :nbias].sum(-1))
    return torch.stack(outs, 1), torch.stack(mass, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["save", "check"])
    ap.add_argument("--ref", required=True)
    ap.add_argument("--tokens", type=int, nargs="+", default=[4096, 25000])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--nbias", type=int, default=640)
    ap.add_argument("--biases", type=float, nargs="+", default=[0.264, 0.529, 1.0])
    ap.add_argument("--time", action="store_true")
    a = ap.parse_args()
    if a.mode == "save":
        out = {}
        for S in a.tokens:
            for seed in range(a.seeds):
                out[(S, seed)] = run_rr(*case(S, seed)).cpu()
        torch.save(out, a.ref)
        print("saved", list(out))
        return
    ref = torch.load(a.ref)
    heads, ok = [0, 17, 41, 55], True
    for S in a.tokens:
        for seed in range(a.seeds):
            c = case(S, seed)
            o = run_rr(*c)
            o0 = run_bias(*c, 0, 0.0)
            e1, e2 = torch.equal(o.cpu(), ref[(S, seed)]), torch.equal(o0.cpu(), ref[(S, seed)])
            ok &= e1 and e2
            print(f"S={S} seed={seed}: attention_rr == before: {e1}; rr_bias(nbias 0) == before: {e2}")
            g = torch.Generator().manual_seed(seed)
            rows = torch.randperm(S, generator=g)[:1024].sort().values.cuda()
            r0, m0 = reference(*c, a.nbias, 0.0, heads, rows)
            err0 = ((o[rows][:, heads].float() - r0).norm() / r0.norm()).item()
            print(f"   bias 0     : rel err vs fp32 {err0:.4f}, text mass {m0.mean().item():.4f}")
            for b in a.biases:
                ob = run_bias(*c, a.nbias, b)
                rb, mb = reference(*c, a.nbias, b, heads, rows)
                errb = ((ob[rows][:, heads].float() - rb).norm() / rb.norm()).item()
                shift = ((rb - r0).norm() / r0.norm()).item()
                ok &= errb < 1.5 * err0 + 1e-3
                print(f"   bias {b:.3f}: rel err vs fp32 {errb:.4f} (the bias moves the output by {shift:.4f}), "
                      f"text mass {mb.mean().item():.4f} (x{(mb.mean() / m0.mean()).item():.2f}, e^b {math.exp(b):.2f})")
            if a.time:
                t0 = ev_time(lambda: run_rr(*c), iters=10)
                tb = ev_time(lambda: run_bias(*c, a.nbias, a.biases[0]), iters=10)
                print(f"   time: rr {t0:.2f} ms, rr_bias {tb:.2f} ms ({100 * (tb / t0 - 1):+.1f}%)")
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
