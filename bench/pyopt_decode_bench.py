"""Decode one saved latent with one video-VAE variant, time it, save the raw frames.

Runs inside a ComfyUI container (no server needed). Variants:
  int8        our current decode: minimax_h3_video_vae_int8_convrot + Rapid-H3 vae_fast (rapid-h3-comfy image)
  fp16        stock comfy VAE, minimax_h3_video_vae_fp16
  fp32        the same weights computed in fp32 (--fp32-vae): which of fp16 / pyopt is nearer full precision
  pyopt       ComfyUI-H3VAE-PyOpt fp16, compile_decoder, tile 256 / batch 2 (rapid-h3-pyopt container)
  pyopt_int8  same + int8_decode (needs comfy-kitchen 0.2.34 first on PYTHONPATH)

    python pyopt_decode_bench.py <x.latent> <variant> --out frames.pt [--iters 3]

Frames are saved as float32 [T, H, W, 3] in [0, 1] (what CreateVideo quantizes to 8 bit).
"""
import argparse, json, os, sys, time

sys.path.insert(0, "/opt/ComfyUI")
ap = argparse.ArgumentParser()
ap.add_argument("latent")
ap.add_argument("variant", choices=["int8", "fp16", "fp32", "pyopt", "pyopt_int8"])
ap.add_argument("--out", default="")
ap.add_argument("--iters", type=int, default=3)
ap.add_argument("--tile-batch", type=int, default=2)
ap.add_argument("--fusions", action="store_true", help="PyOpt decode_fusions (its fastest opt-in CUDA fusions)")
ap.add_argument("--comfy-args", default="--reserve-vram 12 --disable-pinned-memory",
                help="ComfyUI flags as serve.sh passes them (comfy ignores sys.argv unless args parsing is enabled)")
a = ap.parse_args()
sys.argv = sys.argv[:1] + a.comfy_args.split() + (["--fp32-vae"] if a.variant == "fp32" else [])
import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()

import torch  # noqa: E402
import comfy.sd, comfy.utils  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

VAE_DIR = "/data/models/vae"
lat = load_file(a.latent)
z = lat["video"] if "video" in lat else max(lat.values(), key=lambda t: t.numel())
z = z.cuda()
print("latent", tuple(z.shape), z.dtype, flush=True)

if a.variant in ("int8", "fp16", "fp32"):   # fp32: the fp16 file run in fp32 (--fp32-vae), a precision yardstick
    name = {"int8": "minimax_h3_video_vae_int8_convrot.safetensors",
            "fp16": "minimax_h3_video_vae_fp16.safetensors",
            "fp32": "minimax_h3_video_vae_fp16.safetensors"}[a.variant]
    path = os.path.join(VAE_DIR, name)
    if not os.path.exists(path):
        path = os.path.join("/work/models/vae", name)
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(path))
    if a.variant == "int8":
        sys.path.insert(0, "/data/custom_nodes/rapid_h3")
        import vae_fast
        vae_fast.install()                       # as the Rapid-H3 pack does at import (RAPIDH3_VAE_FAST=1)
else:
    sys.path.insert(0, "/data/custom_nodes/ComfyUI-H3VAE-PyOpt")
    import logging
    logging.basicConfig(level=logging.INFO)       # the runtime logs which INT8 / fused layers it installed
    from h3vae_runtime import H3VAEPyOptRuntime, build_comfy_vae
    torch.backends.cudnn.benchmark = False        # cudnn_benchmark=False, as the workflow would pass
    rt = H3VAEPyOptRuntime(model_code_dir=os.environ["H3_VAE_MODEL_CODE_DIR"],
                           weights_path=os.path.join(VAE_DIR, "minimax_h3_video_vae_fp16.safetensors"),
                           device="cuda", dtype=torch.float16, decoder_tile_size=256,
                           tile_batch=a.tile_batch, compile_decoder=True, compile_encoder=False,
                           int8_decode=a.variant == "pyopt_int8", decode_fusions=a.fusions, log_calls=True)
    vae = build_comfy_vae(rt)
print("vae dtype", vae.vae_dtype, flush=True)

times = []
for i in range(a.iters):
    out = None
    torch.cuda.synchronize(); t = time.perf_counter()
    with torch.inference_mode():   # as ComfyUI's executor runs nodes (else autograd keeps the fp16 activations: OOM)
        out = vae.decode(z)
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t)
    print(f"decode {i}: {times[-1]:.2f} s, out {tuple(out.shape)} {out.dtype}", flush=True)
out = out.reshape(-1, *out.shape[-3:]).float().clamp(0, 1).cpu()
print(json.dumps({"variant": a.variant, "latent": os.path.basename(a.latent), "times_s": [round(t, 3) for t in times],
                  "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "frames": tuple(out.shape)}))
if a.out:
    torch.save(out, a.out)
