#!/usr/bin/env python3
"""Qwen3-VL text(+first frame) encode: time and conditioning error per RapidH3TextEncoderQuant mode (custom_nodes/rapid_h3/te_quant.py).

Runs standalone inside the rapid-h3-comfy container (free the server's models first: memory), one checkpoint per call,
after copying this script, harness/shots/eval1.json and custom_nodes/rapid_h3 (as nodes/rapid_h3) to data/te/ on the host:

  docker exec rapid-h3-comfy python /data/te/te_compare.py encode --nodes /data/te/nodes \
      --te /data/models/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors --tag int8 \
      --modes default,int8_rot,int8_rot_plain,bf16_w8 --shots er41,ef31 --out /data/output/te1
  docker exec rapid-h3-comfy python /data/te/te_compare.py compare --out /data/output/te1 [--ref bf16_default]

encode: for each shot (harness/shots/eval1.json entries, copied next to the script) at the ts_h2 draft size (0.7)
and the full size, tokenizes like MiniMaxH3ImageToVideo (first frame stretched to the canvas) and encodes with every
mode; saves <out>/<tag>_<mode>_<shot>_<size>.pt (cond [1, T, 5120] fp32 + minimax_token_tags) and appends timings
(the median of --reps warm encodes, synchronized) to <out>/times.jsonl.
compare: every saved tensor vs int8_default (ComfyUI's own path, te_quant off) and vs --ref (the bf16 checkpoint's
default path, if encoded): relative L2 error, cosine of the whole tensor, mean / min per-token cosine, and the same on text tokens.
"""
import argparse, glob, json, os, statistics, sys, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def draft_size(w, h, f=0.7):
    return round(w * f / 32) * 32, round(h * f / 32) * 32


def load_image(path):
    from PIL import Image, ImageOps
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return torch.from_numpy(np.array(img).astype(np.float32) / 255.0)[None]


def encode(a):
    sys.path.insert(0, "/opt/ComfyUI")
    os.chdir("/opt/ComfyUI")
    sys.path.insert(0, a.nodes)
    import comfy.sd
    from comfy_extras.nodes_minimax_h3 import _resize
    import rapid_h3.te_quant as tq

    shots = {s["id"]: s for s in json.load(open(a.shots_file))}
    clip = comfy.sd.load_clip(ckpt_paths=[a.te], embedding_directory=None, clip_type=comfy.sd.CLIPType.MINIMAX)
    os.makedirs(a.out, exist_ok=True)
    modes = a.modes.split(",")
    clips = {m: tq.RapidH3TextEncoderQuant().patch(clip, m)[0] for m in modes}
    for sid in a.shots.split(","):
        s = shots[sid]
        img = load_image(os.path.join(a.input_dir, s["first"]))
        for size in a.sizes.split(","):
            w, h = (s["width"], s["height"]) if size == "full" else draft_size(s["width"], s["height"])
            tokens = clip.tokenize(s["prompt"], images=[_resize(img[:1], w, h, "disabled")])
            for m in modes:
                c = clips[m]
                ts = []
                for r in range(a.reps + 1):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    cond = c.encode_from_tokens_scheduled(tokens)
                    torch.cuda.synchronize()
                    ts.append(time.perf_counter() - t0)
                t, extra = cond[0][0], cond[0][1]
                torch.save({"cond": t.float().cpu(), "tags": extra.get("minimax_token_tags")},
                           os.path.join(a.out, f"{a.tag}_{m}_{sid}_{size}.pt"))
                rec = {"tag": a.tag, "mode": m, "shot": sid, "size": size, "w": w, "h": h, "tokens": t.shape[1],
                       "first_s": round(ts[0], 3), "warm_s": round(statistics.median(ts[1:]), 3),
                       "all_s": [round(x, 3) for x in ts], "resid_rows": list(tq._ROWS)}
                tq._ROWS[:] = [0, 0]
                print(json.dumps(rec), flush=True)
                with open(os.path.join(a.out, "times.jsonl"), "a") as f:
                    f.write(json.dumps(rec) + "\n")


def metrics(x, ref, tags):
    x, ref = x[0].double(), ref[0].double()
    out = {"rel_l2": ((x - ref).norm() / ref.norm()).item(),
           "cos": torch.nn.functional.cosine_similarity(x.flatten(), ref.flatten(), dim=0).item()}
    tc = torch.nn.functional.cosine_similarity(x, ref, dim=-1)
    out["tok_cos_mean"], out["tok_cos_min"] = tc.mean().item(), tc.min().item()
    if tags is not None:
        txt = tags.bool()
        out["text_rel_l2"] = ((x[txt] - ref[txt]).norm() / ref[txt].norm()).item()
        out["text_tok_cos_min"] = tc[txt].min().item()
        out["vis_rel_l2"] = ((x[~txt] - ref[~txt]).norm() / ref[~txt].norm()).item() if (~txt).any() else None
    return out


def compare(a):
    files = sorted(glob.glob(os.path.join(a.out, "*.pt")))
    data = {os.path.basename(f)[:-3]: torch.load(f) for f in files}
    rows = []
    for name, d in data.items():
        for ref_prefix in ("int8_default", a.ref):
            if not ref_prefix:
                continue
            tail = "_".join(name.split("_")[-2:])  # shot_size
            ref_name = f"{ref_prefix}_{tail}"
            if ref_name == name or ref_name not in data:
                continue
            r = data[ref_name]
            if not torch.equal(d["tags"], r["tags"]):
                raise SystemExit(f"{name}: token tags differ from {ref_name}")
            m = metrics(d["cond"], r["cond"], r["tags"])
            rows.append({"x": name[: -len(tail) - 1], "ref": ref_prefix, "case": tail, **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in m.items()}})
    with open(os.path.join(a.out, "compare.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for r in rows:
        print(f"{r['x']:>18} vs {r['ref']:<14} {r['case']:<11} rel {r['rel_l2']:.5f}  cos {r['cos']:.6f}  "
              f"tok_cos mean {r['tok_cos_mean']:.6f} min {r['tok_cos_min']:.6f}  text rel {r.get('text_rel_l2', 0):.5f}  "
              f"vis rel {r.get('vis_rel_l2') or 0:.5f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["encode", "compare"])
    p.add_argument("--nodes", default=os.path.join(HERE, "nodes"))
    p.add_argument("--te")
    p.add_argument("--tag", default="int8")
    p.add_argument("--modes", default="default,int8_rot,int8_rot_plain,bf16_w8")
    p.add_argument("--shots", default="er41,ef31")
    p.add_argument("--sizes", default="draft,full")
    p.add_argument("--shots-file", default=os.path.join(HERE, "eval1.json"))
    p.add_argument("--input-dir", default="/data/input")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--out", required=True)
    p.add_argument("--ref", default="")
    a = p.parse_args()
    with torch.inference_mode():
        encode(a) if a.cmd == "encode" else compare(a)


if __name__ == "__main__":
    main()
