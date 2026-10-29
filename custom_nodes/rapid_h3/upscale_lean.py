"""RapidH3LatentUpscale3D: the H3 latent upscaler without its memory housekeeping.

MinimaxH3LatentUpscaler3D (the third-party Comfyui_Minimax_h3_latent_Upscaler node, used unmodified) ends every run with
torch.cuda.synchronize + empty_cache + ipc_collect (comfy soft_empty_cache) and a full gc.collect. In ts_h2 that sits
in the draft -> refine gap: the node is ≈1.7 s, of which gc.collect ≈0.18 s, and empty_cache hands the allocator's
cached blocks back to the driver just before the refine reallocates them.

This node runs the very same execute() of that node (same model cache, same arithmetic, same output tensor) with
comfy.model_management.soft_empty_cache and gc.collect turned into no-ops for the duration of the call, then waits
for the output's device -> host copy (the original launches it non_blocking and relied on the sync inside
soft_empty_cache). Bit-identical output by construction. Used by workflow.build(upscale_lean=True).
"""
import gc

import torch

import comfy.model_management as mm


class RapidH3LatentUpscale3D:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "latent": ("LATENT",),
            "model_name": ("STRING", {"default": "minimax_h3_latent_upscaler_3d_fp16.safetensors"}),
            "width": ("INT", {"default": 1280, "min": 64, "max": 8192, "step": 8}),
            "height": ("INT", {"default": 704, "min": 64, "max": 8192, "step": 8}),
            "align": ("INT", {"default": 32, "min": 1, "max": 512}),
            "enable_temporal_chunking": ("BOOLEAN", {"default": False}),
            "device": (["cuda", "rocm", "cpu"], {"default": "cuda"}),
            "precision": (["fp32", "fp16", "bf16"], {"default": "fp16"}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "run"
    CATEGORY = "Rapid-H3"

    def run(self, latent, model_name, width, height, align, enable_temporal_chunking, device, precision):
        import nodes
        up = nodes.NODE_CLASS_MAPPINGS.get("MinimaxH3LatentUpscaler3D")
        if up is None:
            raise RuntimeError("RapidH3LatentUpscale3D needs the Comfyui_Minimax_h3_latent_Upscaler node pack")
        saved = mm.soft_empty_cache, gc.collect
        mm.soft_empty_cache, gc.collect = (lambda *a, **k: None), (lambda *a, **k: 0)
        try:
            out = up.execute(latent=latent, model_name=model_name,
                             mode={"mode": "target dimensions", "width": width, "height": height}, align=align,
                             enable_temporal_chunking=enable_temporal_chunking, force_unload=False, device=device,
                             precision=precision)
        finally:
            mm.soft_empty_cache, gc.collect = saved
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # the non_blocking device -> host copy of the output
        return (out.args[0],)


NODE_CLASS_MAPPINGS = {"RapidH3LatentUpscale3D": RapidH3LatentUpscale3D}
