"""Video-VAE decode: tiles per decoder call (block-major batching), timing and exactness.

Runs inside the ComfyUI image (no server needed):
  python vae_bm.py block  <vae.safetensors> [--tiles 4 7 14 28]        # per-op GPU time of one ViT block
  python vae_bm.py decode <latent> <vae.safetensors> --batch N [--save f.pt] [--iters 3]
  python vae_bm.py cmp a.pt b.pt                                        # per-frame PSNR / SSIM

--batch 0 = ComfyUI's own tiling (<= 4 tiles of one row per call); N > 0 = vae_fast.install_tile_batch(N),
N tiles per call across rows. --fast installs vae_fast (as the server does), --swiglu its w1 SwiGLU epilogue.
Frames are saved as raw float32. RAPIDH3_NODE_DIR points at a copy of custom_nodes/rapid_h3; run it in a throwaway
`docker run --rm --gpus all ... rapid-h3/h3-comfyui:v0.42.1` with the server stopped (a decode next to the
resident server can trip earlyoom).
"""
import argparse
import math
import os
import statistics
import sys
import time

sys.path.insert(0, "/opt/ComfyUI")
sys.path.insert(0, os.environ.get("RAPIDH3_NODE_DIR", "/work/build/vae_bm/rapid_h3"))
import torch  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("mode")
ap.add_argument("args", nargs="*")
ap.add_argument("--tiles", type=int, nargs="*", default=[4, 7, 14, 28])
ap.add_argument("--batch", type=int, default=0)
ap.add_argument("--iters", type=int, default=3)
ap.add_argument("--save", default="")
ap.add_argument("--fast", action="store_true")
ap.add_argument("--swiglu", action="store_true", help="vae_fast.install_swiglu() (w1 SwiGLU epilogue)")
ap.add_argument("--level", type=int, default=0, help="vae_fast level as RAPIDH3_VAE_FAST sets it (1-4; overrides --fast/--swiglu/--batch)")
ap.add_argument("--profile", action="store_true", help="decode: one extra profiled decode, GPU time per kernel")
a = ap.parse_args()
sys.argv = sys.argv[:1]


def load_vae(path):
    import comfy.sd
    import comfy.utils
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(path))
    if a.level:
        import vae_fast
        vae_fast.ext()
        vae_fast.install_level(a.level, int(os.environ.get("RAPIDH3_VAE_TILES", "28")))
        return vae
    if a.fast:
        import vae_fast
        vae_fast.ext()
        vae_fast.install()
    if a.swiglu:
        import vae_fast
        vae_fast.install_swiglu(True)
    if a.batch:
        import vae_fast
        vae_fast.install_tile_batch(a.batch)
    return vae


def block_profile(vae_path):
    """GPU time per op of one decoder block for N tiles in one call (S = 1797 tokens per tile)."""
    from torch.profiler import ProfilerActivity, profile
    import comfy.model_management as mm
    vae = load_vae(vae_path)
    mm.load_models_gpu([vae.patcher], force_full_load=True)
    dec = vae.first_stage_model.decoder
    blk = dec.transformer_blocks[10]
    S = 7 * 16 * 16 + 5
    for n in a.tiles:
        torch.manual_seed(0)
        x = (torch.randn(n, S, 2048, device="cuda") * 0.5).half()
        ids = torch.rand(n, S, 3, device="cuda").half()
        rot = dec.pos_embed(ids)
        for _ in range(3):
            blk(x, rot)
        torch.cuda.synchronize()
        reps = max(1, 28 // n)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(reps):
                blk(x, rot)
            torch.cuda.synchronize()
        rows = {}
        for e in prof.key_averages():
            if e.self_device_time_total > 0:
                k = e.key
                cat = ("gemm" if ("gemm" in k.lower() or "cutlass" in k.lower() or "Gemm" in k) else
                       "attn" if "attn" in k.lower() or "attention" in k.lower() or "fmha" in k.lower() else
                       "quant" if "quant" in k.lower() else "rope" if "rope" in k.lower() else "other")
                rows.setdefault(cat, 0.0)
                rows[cat] += e.self_device_time_total / 1e3 / reps
        t0 = time.perf_counter()
        for _ in range(reps * 3):
            blk(x, rot)
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) / (reps * 3) * 1e3
        per_tile = {k: v / n for k, v in rows.items()}
        print(f"tiles={n:3d} wall {wall:8.2f} ms/call {wall / n:6.3f} ms/tile | per tile: "
              + " ".join(f"{k} {v:.3f}" for k, v in sorted(per_tile.items())) +
              f" | sum {sum(per_tile.values()):.3f}", flush=True)
        if n == a.tiles[0] or n == a.tiles[-1]:
            for e in sorted(prof.key_averages(), key=lambda e: -e.self_device_time_total)[:8]:
                print(f"     {e.self_device_time_total / 1e3 / reps:8.2f} ms n={e.count // reps:3d} {e.key[:100]}")
        del x, rot


def decode(lat_path, vae_path):
    from safetensors.torch import load_file
    lat = load_file(lat_path)
    z = max(lat.values(), key=lambda t: t.numel()).cuda()
    print("latent", tuple(z.shape), flush=True)
    vae = load_vae(vae_path)
    ts = []
    for i in range(a.iters):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = vae.decode(z)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
        print(f"decode {i}: {ts[-1]:.2f} s, out {tuple(out.shape)}, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB",
              flush=True)
    warm = ts[1:] if len(ts) > 1 else ts
    print(f"RESULT level={a.level} batch={a.batch} fast={a.fast} swiglu={a.swiglu} median_warm {statistics.median(warm):.2f} s  all {[round(t, 2) for t in ts]}")
    if a.save:
        torch.save(out.cpu(), a.save)
    if a.profile:
        from torch.profiler import ProfilerActivity, profile
        del out
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            vae.decode(z)
            torch.cuda.synchronize()
        rows = sorted(((e.self_device_time_total / 1e3, e.count, e.key[:140]) for e in prof.key_averages()
                       if e.self_device_time_total > 0), reverse=True)
        print("PROFILE total GPU ms %.0f" % sum(r[0] for r in rows))
        for t, n, k in rows[:40]:
            print(f"  {t:9.1f} ms  n={n:6d}  {k}")


def cmp(pa, pb):
    sys.path.insert(0, os.environ.get("RAPIDH3_SCORE_DIR", "/work/build/h3vae_pyopt"))  # a copy of harness/score.py
    from score import ssim
    x, y = torch.load(pa), torch.load(pb)   # [1, T, H, W, 3] float in [0, 1]
    print("identical:", torch.equal(x, y), "max abs", (x - y).abs().max().item())
    x, y = x[0], y[0]
    ps, ss = [], []
    for i in range(x.shape[0]):
        u = x[i].cuda()[None]   # [1, H, W, 3]
        v = y[i].cuda()[None]
        m = (u - v).pow(2).mean().item()
        ps.append(99.0 if m == 0 else 10 * math.log10(1 / m))
        ss.append(ssim(u, v))
    print(f"PSNR mean {sum(ps) / len(ps):.2f} dB  min {min(ps):.2f} (frame {ps.index(min(ps))})  "
          f"SSIM mean {sum(ss) / len(ss):.6f} min {min(ss):.6f}  frames {len(ps)}")


with torch.inference_mode():
    if a.mode == "block":
        block_profile(*a.args)
    elif a.mode == "decode":
        decode(*a.args)
    elif a.mode == "cmp":
        cmp(*a.args)
