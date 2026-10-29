"""Two-stage conditioning without a second Qwen3-VL pass.

The two-stage graph (workflow.build draft=...) ran MiniMaxH3ImageToVideo twice: at the draft size for the draft
and at the full size for the refine; ComfyUI ran the full-size one only after the draft (in the draft -> refine gap).
Each runs the 32B Qwen3-VL encoder; with a first / last frame the image is resized to the canvas first, so the vision
tokens (canvas / 32 squared of them) differ between the two sizes.

  RapidH3DraftConditioning  the full-size node's conditioning -> the draft's: the keyframe latents re-encoded from the
                         same images at the draft size (the same resize and VAE encode as MiniMaxH3ImageToVideo), the
                         Qwen hidden states kept (full-size vision tokens), plus the draft-size empty AV latent.
                         Text-only (no frames): the conditioning passes through unchanged, which is exactly what the
                         draft-size node computes (the text tokens do not depend on the canvas), so it is bit-identical.
                         Size-generic: encode_once="draft" uses it the other way, the refine's from the draft's.
  RapidH3After              passes `value` through once `after` exists: forces ComfyUI to run a node earlier (both
                         MiniMaxH3ImageToVideo nodes before the draft, back to back), with no change to any tensor.
  RapidH3RefConditioning    Ref2VA (MiniMaxH3ReferenceToVideo) two-stage: one stage's conditioning reused for the other
                         stage when MiniMaxH3ReferenceToVideo would compute exactly the same there, else that stage's
                         own node (the lazy `other` input) runs. Only reference images depend on the canvas (sized
                         "match": scaled down to the canvas area; "max": 2048 short edge); reference videos (own
                         canvas), audios and the text do not. So the reuse is bit-identical whenever every reference
                         image gets the same size at both canvases (crops smaller than the draft area, or "max"),
                         and full-frame references fall back to the second encode (the pre-2026-10-26 two-node graph).
"""
import math

import node_helpers

from comfy_extras.nodes_minimax_h3 import (CANVAS_MULTIPLE, REF_IMAGE_SHORT_EDGE, _empty_av_latent,
                                           _resize)


class RapidH3DraftConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning": ("CONDITIONING",),
            "vae": ("VAE",),
            "width": ("INT", {"default": 576, "min": 32, "max": 8192, "step": 32}),
            "height": ("INT", {"default": 352, "min": 32, "max": 8192, "step": 32}),
            "length": ("INT", {"default": 124, "min": 5, "max": 3600}),
        }, "optional": {
            "first_frame": ("IMAGE",),
            "last_frame": ("IMAGE",),
        }}

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    FUNCTION = "apply"
    CATEGORY = "Rapid-H3"

    def apply(self, conditioning, vae, width, height, length, first_frame=None, last_frame=None):
        latent, frame_count = _empty_av_latent(width, height, length)
        keyframes = []
        if first_frame is not None:  # as MiniMaxH3ImageToVideo: plain stretch to the canvas
            keyframes.append({"resolved_frame_index": 0, "image": _resize(first_frame[:1], width, height, "disabled")})
        if last_frame is not None:  # aspect-preserving cover crop
            keyframes.append({"resolved_frame_index": frame_count - 1,
                              "image": _resize(last_frame[:1], width, height, "center")})
        had = [len(c[1].get("minimax_keyframes", ())) for c in conditioning]
        if any(n != len(keyframes) for n in had):
            raise ValueError(f"RapidH3DraftConditioning: the conditioning carries {had} keyframes, {len(keyframes)} images "
                             "given (wire the same first/last frames as the full-size node)")
        if not keyframes:
            return (conditioning, latent)
        for kf in keyframes:
            kf["latent"] = vae.encode(kf.pop("image"))
        return (node_helpers.conditioning_set_values(conditioning, {"minimax_keyframes": keyframes}), latent)


class RapidH3After:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("LATENT",), "after": ("CONDITIONING",)}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply"
    CATEGORY = "Rapid-H3"

    def apply(self, value, after):
        return (value,)


def _ref_image_size(img, width, height, ref_image_size):
    """The size MiniMaxH3ReferenceToVideo gives one reference image."""
    rh, rw = img.shape[1:3]
    # never upscaled: "match" keeps the canvas area, otherwise the short edge is capped
    limit = math.sqrt(width * height / (rw * rh)) if ref_image_size == "match" else REF_IMAGE_SHORT_EDGE / min(rw, rh)
    s = min(1.0, limit)

    def snap(v):  # nearest multiple of the canvas grid, at least one cell
        return max(CANVAS_MULTIPLE, CANVAS_MULTIPLE * round(v * s / CANVAS_MULTIPLE))
    return snap(rw), snap(rh)


class RapidH3RefConditioning:
    N_IMAGES = 9

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning": ("CONDITIONING",),
            "other": ("CONDITIONING", {"lazy": True}),
            "width": ("INT", {"default": 1344, "min": 32, "max": 8192, "step": 32}),
            "height": ("INT", {"default": 768, "min": 32, "max": 8192, "step": 32}),
            "length": ("INT", {"default": 124, "min": 5, "max": 3600}),
            "ref_image_size": (["match", "max"], {"default": "match"}),
        }, "optional": {f"ref_image_{i}": ("IMAGE",) for i in range(cls.N_IMAGES)}}

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    FUNCTION = "apply"
    CATEGORY = "Rapid-H3"

    @staticmethod
    def _same(conditioning, width, height, ref_image_size, images):
        for c in conditioning:
            blocks = [b for b in c[1].get("minimax_refs", ()) if b["kind"] == "image"]
            if len(blocks) != len(images):
                return False
            for b, img in zip(blocks, images):
                tw, th = _ref_image_size(img, width, height, ref_image_size)
                if (th // 16, tw // 16) != (b["latent_h"], b["latent_w"]):
                    return False
        return True

    @classmethod
    def _images(cls, kw):
        return [kw[f"ref_image_{i}"] for i in range(cls.N_IMAGES) if kw.get(f"ref_image_{i}") is not None]

    def check_lazy_status(self, conditioning, width, height, length, ref_image_size, other=None, **kw):
        if self._same(conditioning, width, height, ref_image_size, self._images(kw)):
            return []
        return ["other"]

    def apply(self, conditioning, width, height, length, ref_image_size, other=None, **kw):
        latent, _ = _empty_av_latent(width, height, length)
        same = self._same(conditioning, width, height, ref_image_size, self._images(kw))
        print(f"[rapid-h3] RapidH3RefConditioning {width}x{height}: "
              + ("reused (same reference sizes)" if same else "own encode (reference image sizes differ)"), flush=True)
        return (conditioning if same else other, latent)


NODE_CLASS_MAPPINGS = {"RapidH3DraftConditioning": RapidH3DraftConditioning, "RapidH3After": RapidH3After,
                       "RapidH3RefConditioning": RapidH3RefConditioning}
