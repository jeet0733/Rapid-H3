#!/usr/bin/env python3
"""Step-count and two-stage speed/quality test on a set of first-frame (fl2va) shots.

Runs on the Spark host (stdlib only), against the rapid-h3-comfy server:

    python3 harness/fast_test.py SHOTS.json --variants ts_h2 ts_s6 [--shots er41 ef31] [--name fast1]

SHOTS.json: [{"id", "prompt", "first" (image name in data/input; omit for text-only), "width", "height", "length",
"seed" (optional, default 7)}], e.g. harness/shots/eval1.json; the first-frame images are your own (not in the repo).
Every render gets the full Rapid-H3 recipe plus the variant's settings (VARIANTS below). For each render it records
the wall time and its split from RapidH3StepTimer: before the first step (text encoding, conditioning, any model
load), each denoising step, and after the last step (latent upscale for two-stage is inside sampling, VAE decode
and save are after). Results append to data/output/<name>/results.jsonl; videos land in data/output/<name>/.
"""
import argparse, json, os, sys, time, uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render, workflow  # noqa: E402

RECIPE = dict(quant="int8_rot", attention="comfy kitchen attention", video_vae="int8", turbo8=False,
              sampler="res_multistep", scheduler="simple", seed=7)
PDD = "MiniMax-H3-FL2VA-Acc-8Step.safetensors"  # alibaba-pai/MiniMax-H3-Acc-LoRAs, in models/pdd_acc
TRT_VAE = "minimax_h3_vae_decoder.engine"  # fp16 TensorRT engine built from lihaoyun6/MiniMax-H3-VAE-ONNX
PDD_UNET = "minimax_h3_fl2va_pruned_pddbaked_int8_convrot.safetensors"  # PDD trunk baked into Comfy's int8_convrot DiT
# shift 12 schedule, 10 steps: 1, .991, .980, .966, .947, .923, .889, .837, .750, .571, 0
VARIANTS = {
    "base10": dict(steps=10),
    "base8": dict(steps=8),
    "base7": dict(steps=7),
    "base6": dict(steps=6),
    # two-stage: 10 base steps at the draft size, x2 latent upscale, a short refine at full size
    "ts_a": dict(steps=10, draft="half", refine_sigmas="0.889, 0.837, 0.75, 0.571, 0"),   # tail of the 10-step schedule
    "ts_b": dict(steps=10, draft="half", refine_sigmas="0.9035, 0.6316, 0.3158, 0"),     # a three-step refine tail
    "ts_c": dict(steps=10, draft="half", refine_sigmas="0.75, 0.571, 0"),                 # lighter refine
    "ts_d": dict(steps=10, draft="0.7", refine_sigmas="0.889, 0.837, 0.75, 0.571, 0"),    # bigger draft
    # refine from 0.78 with front-loaded small steps, the draft's audio kept (2026-10-23 research shortlist)
    "ts_e": dict(steps=10, draft="0.7", refine_sigmas="0.78, 0.55, 0.25, 0", freeze_audio=True),
    "ts_f": dict(steps=10, draft="0.7", refine_sigmas="0.78, 0.64, 0.45, 0.22, 0", freeze_audio=True),
    "ts_g": dict(steps=10, draft="half", refine_sigmas="0.78, 0.55, 0.25, 0", freeze_audio=True),
    # 544x928 draft with fewer draft steps and a 2-3 step refine, aimed at < 180 s
    "ts_h": dict(steps=6, draft="0.7", refine_sigmas="0.78, 0.45, 0", freeze_audio=True),
    "ts_i": dict(steps=7, draft="0.7", refine_sigmas="0.78, 0.55, 0.25, 0", freeze_audio=True),
    "base6_s8": dict(steps=6, shift=8.0),
    "base7_s8": dict(steps=7, shift=8.0),
    # cheaper drafts / refine against ts_h
    "h5": dict(steps=6, draft="0.7", draft_sigmas="1, 0.97, 0.93, 0.87, 0.78, 0", refine_sigmas="0.78, 0.45, 0", freeze_audio=True),
    "h4": dict(steps=6, draft="0.7", draft_sigmas="1, 0.96, 0.9, 0.8, 0", refine_sigmas="0.78, 0.45, 0", freeze_audio=True),
    "h_d6": dict(steps=6, draft="0.6", refine_sigmas="0.78, 0.45, 0", freeze_audio=True),
    "h5_d6": dict(steps=6, draft="0.6", draft_sigmas="1, 0.97, 0.93, 0.87, 0.78, 0", refine_sigmas="0.78, 0.45, 0", freeze_audio=True),
    "h_r1": dict(steps=6, draft="0.7", refine_sigmas="0.55, 0", freeze_audio=True),
    # shorter drafts that keep ts_h's low tail (0.857, 0.706, 0); 1-step refine combined with them
    "h5t": dict(steps=6, draft="0.7", draft_sigmas="1, 0.96, 0.923, 0.857, 0.706, 0", refine_sigmas="0.78, 0.45, 0", freeze_audio=True),
    "h4t": dict(steps=6, draft="0.7", draft_sigmas="1, 0.94, 0.857, 0.706, 0", refine_sigmas="0.78, 0.45, 0", freeze_audio=True),
    "h4t_r1": dict(steps=6, draft="0.7", draft_sigmas="1, 0.94, 0.857, 0.706, 0", refine_sigmas="0.55, 0", freeze_audio=True),
    # diagnostic: ts_h's own 6 draft sigmas (shift 12, simple) fed through ManualSigmas; should reproduce ts_h
    "h6m": dict(steps=6, draft="0.7", draft_sigmas="1, 0.98361, 0.96, 0.92308, 0.85714, 0.70588, 0", refine_sigmas="0.78, 0.45, 0", freeze_audio=True),
    # PDD Acc distilled draft (euler on the trained grid, CFG 1), base model refines as in h_r1
    "p4_r1": dict(steps=6, draft="0.7", draft_pdd=PDD, draft_pdd_nfe=4, refine_sigmas="0.55, 0", freeze_audio=True),
    "p6_r1": dict(steps=6, draft="0.7", draft_pdd=PDD, draft_pdd_nfe=6, refine_sigmas="0.55, 0", freeze_audio=True),
    "p8_r1": dict(steps=6, draft="0.7", draft_pdd=PDD, draft_pdd_nfe=8, refine_sigmas="0.55, 0", freeze_audio=True),
    # the same on Comfy's int8_convrot DiT (adopted as is, no bf16 copy): the bf16 DiT + two int8 copies + text
    # encoder leave ~5 GB free and the Spark re-reads the DiT from disk every step (draft 23 s/step instead of 10)
    "p4c_r1": dict(steps=6, draft="0.7", dit="int8_convrot", draft_pdd=PDD, draft_pdd_nfe=4, refine_sigmas="0.55, 0", freeze_audio=True),
    "p6c_r1": dict(steps=6, draft="0.7", dit="int8_convrot", draft_pdd=PDD, draft_pdd_nfe=6, refine_sigmas="0.55, 0", freeze_audio=True),
    # PDD trunk baked offline into an int8_convrot DiT (bake_pdd_trunk.py): both DiTs adopted as is, no LoRA at runtime
    "p4k_r1": dict(steps=6, draft="0.7", dit="int8_convrot", draft_pdd=PDD, draft_pdd_nfe=4, draft_pdd_unet=PDD_UNET,
                   refine_sigmas="0.55, 0", freeze_audio=True),
    "p6k_r1": dict(steps=6, draft="0.7", dit="int8_convrot", draft_pdd=PDD, draft_pdd_nfe=6, draft_pdd_unet=PDD_UNET,
                   refine_sigmas="0.55, 0", freeze_audio=True),
    "h_r1c": dict(steps=6, draft="0.7", dit="int8_convrot", refine_sigmas="0.55, 0", freeze_audio=True),
    # h_r1 with the mp4 write overlapping the VAE decode (same file, bit for bit), the same node
    # without overlap (timing control), and with a TensorRT fp16 decoder engine (ComfyUI-H3VAE_TRT)
    # ts_h2 as of 2026-10-24 (= h_r72_ds below; workflow.PRESETS["ts_h2_ck"]); ts_h above is the rollback
    "ts_h2_ck": dict(steps=6, draft="0.7", refine_sigmas="0.72, 0", freeze_audio=True, decode_save="overlap"),
    # = workflow.PRESETS["ts_h2"]: + encode once, our INT8 attention (config 190), INT8 text encode
    "ts_h2": dict(steps=6, draft="0.7", refine_sigmas="0.72, 0", freeze_audio=True, decode_save="overlap",
                  encode_once="draft", rapidh3_attn=190, te_quant="int8_rot"),
    # Qwen3-VL encode on our INT8 W8A8 path (RapidH3TextEncoderQuant int8_rot): conditioning changes slightly; with
    # attention config 194 these are the 2026-10-26 presets (= ts_h2 / ts_s6 with rapidh3_attn=194, bit-identical)
    "ts_h2_te8": dict(steps=6, draft="0.7", refine_sigmas="0.72, 0", freeze_audio=True, decode_save="overlap",
                      encode_once="draft", rapidh3_attn=194, te_quant="int8_rot"),
    "ts_s6_te8": dict(steps=6, decode_save="overlap", rapidh3_attn=194, te_quant="int8_rot"),
    "h_r1_ds": dict(steps=6, draft="0.7", refine_sigmas="0.55, 0", freeze_audio=True, decode_save="overlap"),
    # ts_h2 with a 1-step refine from a higher sigma (same cost): small faces in wide shots (eval1 er1*, ex5*)
    "h_r65_ds": dict(steps=6, draft="0.7", refine_sigmas="0.65, 0", freeze_audio=True, decode_save="overlap"),
    "h_r72_ds": dict(steps=6, draft="0.7", refine_sigmas="0.72, 0", freeze_audio=True, decode_save="overlap"),
    "h_r1_dss": dict(steps=6, draft="0.7", refine_sigmas="0.55, 0", freeze_audio=True, decode_save="serial"),
    "h_r1_trt": dict(steps=6, draft="0.7", refine_sigmas="0.55, 0", freeze_audio=True, trt_vae=TRT_VAE),
    # motion1: which part of the fast draft weakens camera push-in vs stock
    "m_d10": dict(steps=10, draft="0.7", refine_sigmas="0.72, 0", freeze_audio=True, decode_save="overlap"),
    "m_d8": dict(steps=8, draft="0.7", refine_sigmas="0.72, 0", freeze_audio=True, decode_save="overlap"),
    "m_d7h": dict(steps=6, draft="0.7", draft_sigmas="1, 0.991, 0.98, 0.96, 0.923, 0.857, 0.706, 0",
                  refine_sigmas="0.72, 0", freeze_audio=True, decode_save="overlap"),  # ts_h2 + one extra high step
    "m_hi6": dict(steps=6, draft="0.7", draft_sigmas="1, 0.99, 0.975, 0.95, 0.91, 0.857, 0",
                  refine_sigmas="0.72, 0", freeze_audio=True, decode_save="overlap"),  # 6 steps moved to high noise
    "m_sh16": dict(steps=6, draft="0.7", shift=16.0, refine_sigmas="0.72, 0", freeze_audio=True, decode_save="overlap"),
    "m_sh8": dict(steps=6, draft="0.7", shift=8.0, refine_sigmas="0.72, 0", freeze_audio=True, decode_save="overlap"),
    # motion2: a larger draft, or one stage with fewer steps (the 0.7 draft decides the camera move)
    "m2_d80": dict(steps=6, draft="0.8", refine_sigmas="0.72, 0", freeze_audio=True, decode_save="overlap"),
    "m2_d85": dict(steps=6, draft="0.85", refine_sigmas="0.72, 0", freeze_audio=True, decode_save="overlap"),
    "m2_b6": dict(steps=6, decode_save="overlap"),
    "ts_s6_ck": dict(steps=6, decode_save="overlap"),  # ts_s6 before 2026-10-26 (= m2_b6)
    "ts_s6": dict(steps=6, decode_save="overlap", rapidh3_attn=190, te_quant="int8_rot"),  # = workflow.PRESETS["ts_s6"]
    "m2_b8": dict(steps=8, decode_save="overlap"),
    "ts_s5": dict(steps=5, decode_save="overlap"),  # ts_s6 with 5 steps (s5 test): does the camera move survive?
    "m_rp": dict(steps=6, draft="0.7", refine_sigmas="0.72, 0", freeze_audio=True, decode_save="overlap",
                 refine_prompt="Photorealistic, natural film look, high detail."),  # timing only
    # round 4 kernel check (opt-in): ts_h2 with our INT8 attention (config 80) instead of comfy-kitchen's; same seed
    # as ts_h2 (the extra attention node changes the graph, so ComfyUI's cache cannot reuse a ts_h2 result)
    "ts_h2_a80": dict(steps=6, draft="0.7", refine_sigmas="0.72, 0", freeze_audio=True, decode_save="overlap",
                      rapidh3_attn=80),
    # config 80's math on a CAKE-style schedule (bit-identical to 80, ≈4% faster per call)
    "ts_h2_a94": dict(steps=6, draft="0.7", refine_sigmas="0.72, 0", freeze_audio=True, decode_save="overlap",
                      rapidh3_attn=94),
    "ts_s6_a94": dict(steps=6, decode_save="overlap", rapidh3_attn=94),
    # attention round 6 (attn_cake/DESIGN.md section 7): ts_h2 / ts_s6 with config 190 (= 194 with the softmax
    # mutex entered after the row max; bit-identical) and 189 (not bit-identical)
    "ts_h2_a190": dict(steps=6, draft="0.7", refine_sigmas="0.72, 0", freeze_audio=True, decode_save="overlap",
                       encode_once="draft", rapidh3_attn=190, te_quant="int8_rot"),
    "ts_s6_a190": dict(steps=6, decode_save="overlap", rapidh3_attn=190, te_quant="int8_rot"),
    "ts_h2_a189": dict(steps=6, draft="0.7", refine_sigmas="0.72, 0", freeze_audio=True, decode_save="overlap",
                       encode_once="draft", rapidh3_attn=189, te_quant="int8_rot"),
    "ts_s6_a189": dict(steps=6, decode_save="overlap", rapidh3_attn=189, te_quant="int8_rot"),
    # error check: per-linear error vs bf16 (measure_every) for int8_rot and NVFP4 fake-quant
    "q_int8m": dict(steps=6, draft="0.7", refine_sigmas="0.72, 0", freeze_audio=True, measure_every=3),
    "q_fp4m": dict(steps=6, draft="0.7", refine_sigmas="0.72, 0", freeze_audio=True, measure_every=3, quant="nvfp4_fake"),
    "q_fp4rm": dict(steps=6, draft="0.7", refine_sigmas="0.72, 0", freeze_audio=True, measure_every=3,
                    quant="nvfp4_rot_fake"),
    # references at the same 10 steps: stock ComfyUI bf16, and Comfy-Org's int8_convrot DiT + VAE on Comfy's own path
    "orig10": dict(steps=10, quant="bf16", attention="pytorch attention", video_vae="fp16"),
    "cr10": dict(steps=10, dit="int8_convrot", quant="bf16", attention="pytorch attention"),
}


def draft_size(spec, w, h):
    f = 0.5 if spec == "half" else float(spec)
    return f"{round(w * f / 32) * 32}x{round(h * f / 32) * 32}"


def run(graph, timeout):
    t0 = time.time()
    pid = render.api("/prompt", {"prompt": graph, "client_id": str(uuid.uuid4())})["prompt_id"]
    hist = None
    while time.time() - t0 < timeout:
        time.sleep(1)
        h = render.api(f"/history/{pid}")
        if pid in h and (h[pid].get("status", {}).get("completed") or h[pid].get("status", {}).get("status_str") == "error"):
            hist = h[pid]
            break
    return hist, t0, time.time()


def steps_since(t0):
    try:
        rows = [json.loads(l) for l in open(render.STEPS)]
    except OSError:
        return []
    return [r for r in rows if r["t"] >= t0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shots")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--shots", nargs="*", dest="only")
    ap.add_argument("--name", default="fast1")
    ap.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args()
    shots = [s for s in json.load(open(a.shots)) if not a.only or s["id"] in a.only]
    out_dir = os.path.join(render.DATA, "output", a.name)
    os.makedirs(out_dir, exist_ok=True)
    res_path = os.path.join(out_dir, "results.jsonl")
    done = set()
    if os.path.exists(res_path):
        done = {(r["shot"], r["variant"]) for r in map(json.loads, open(res_path)) if r["status"] == "success"}
    for s in shots:
        for v in a.variants:
            if (s["id"], v) in done:
                continue
            kw = dict(RECIPE, **VARIANTS[v])
            if "seed" in s:  # the shot's seed wins over a variant's
                kw["seed"] = s["seed"]
            if "draft" in kw:
                kw["draft"] = draft_size(kw["draft"], s["width"], s["height"])
            g = workflow.build(s["prompt"], width=s["width"], height=s["height"], length=s["length"],
                               first_frame=s.get("first"), prefix=f"{a.name}/{s['id']}_{v}", **kw)
            hist, t0, t1 = run(g, a.timeout)
            status = (hist or {}).get("status", {}).get("status_str", "timeout")
            st = steps_since(t0)
            rec = {"shot": s["id"], "variant": v, "status": status, "wall_s": round(t1 - t0, 1),
                   "settings": {k: kw[k] for k in ("steps", "dit", "draft", "draft_sigmas", "draft_pdd", "draft_pdd_nfe", "draft_pdd_unet", "refine_sigmas", "shift", "freeze_audio", "decode_save", "trt_vae") if k in kw},
                   "time": time.strftime("%Y-%m-%d %H:%M:%S")}
            if st:
                first_start = st[0]["t"] - st[0]["step_ms"] / 1e3
                rec.update(pre_s=round(first_start - t0, 1), post_s=round(t1 - st[-1]["t"], 1),
                           sampling_s=round(st[-1]["t"] - first_start, 1), steps_s=[round(r["step_ms"] / 1e3, 2) for r in st])
            if status != "success":
                rec["error"] = json.dumps((hist or {}).get("status", {}).get("messages", []))[-1500:]
            with open(res_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
