"""Profile the H3 video VAE decode on a saved latent (standalone, inside the ComfyUI image).

python vae_bench.py <latent.safetensors> <vae.safetensors> [--profile]
"""
import argparse, os, sys, time

sys.path.insert(0, "/opt/ComfyUI")
sys.argv = sys.argv[:1] + [a for a in sys.argv[1:]]
import torch  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("latent")
ap.add_argument("vae")
ap.add_argument("--profile", action="store_true")
ap.add_argument("--iters", type=int, default=2)
ap.add_argument("--batch", type=int, default=0, help="tiles per decoder call (0 = ComfyUI default, max 4)")
ap.add_argument("--save", default="")
ap.add_argument("--compare", default="")
ap.add_argument("--fast", action="store_true", help="install custom_nodes/rapid_h3/vae_fast.py")
a, _ = ap.parse_known_args()
sys.argv = sys.argv[:1]

import comfy.sd, comfy.utils  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

lat = load_file(a.latent)
print({k: tuple(v.shape) for k, v in lat.items()})
z = max(lat.values(), key=lambda t: t.numel()).cuda()  # the video part
vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(a.vae))
if a.fast:
    sys.path.insert(0, "/work/custom_nodes/rapid_h3")
    import vae_fast
    vae_fast.ext()
    vae_fast.install()
    print("vae_fast installed")
if a.batch:
    import comfy.ldm.minimax.vae as mv

    def _decode_tile_row(self, z_row, x_idx, x_len, _b=a.batch):
        slices = [z_row[..., j_pos // self.vae_ratio:(j_pos + j_len) // self.vae_ratio] for j_pos, j_len in zip(x_idx, x_len)]
        for k in range(0, len(slices), _b):
            group = slices[k:k + _b]
            yield from self._decode_pixels(torch.cat(group)).chunk(len(group))
    for cls in vars(mv).values():
        if isinstance(cls, type) and hasattr(cls, "_decode_tile_row"):
            cls._decode_tile_row = _decode_tile_row
            print("patched", cls.__name__, "batch", a.batch)
print("vae dtype", vae.vae_dtype, "device", vae.device)
for i in range(a.iters):
    torch.cuda.synchronize(); t = time.perf_counter()
    out = vae.decode(z)
    torch.cuda.synchronize()
    print(f"decode {i}: {time.perf_counter() - t:.2f} s, out {tuple(out.shape)}", flush=True)
if a.save:
    torch.save(out.cpu(), a.save)
if a.compare:
    ref = torch.load(a.compare)
    o = out.cpu()
    print("identical:", torch.equal(o, ref), "max abs diff", (o.float() - ref.float()).abs().max().item())
if a.profile:
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], record_shapes=True) as prof:
        vae.decode(z)
        torch.cuda.synchronize()
    shp = sorted(((e.device_time_total / 1e3, e.count, e.key[:50], str(e.input_shapes)[:160])
                  for e in prof.key_averages(group_by_input_shape=True) if e.device_time_total > 50e3 and not e.key.startswith("void")),
                 reverse=True)
    print("ops by input shape (device ms incl. children):")
    for t, n, k, sh in shp[:25]:
        print(f"  {t:8.1f} ms n={n:5d} {k} {sh}")
    rows = sorted(((e.self_device_time_total / 1e3, e.count, e.key[:110]) for e in prof.key_averages()
                   if e.self_device_time_total > 0), reverse=True)
    print("total CUDA ms %.0f" % sum(r[0] for r in rows))
    for t, n, k in rows[:30]:
        print(f"  {t:9.1f} ms  n={n:6d}  {k}")
