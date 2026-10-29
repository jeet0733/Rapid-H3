"""frozen, mass-bounded block-sparse attention (LoSA-style) for MiniMax-H3.

RapidH3AttnMass (diagnostic): for chosen DiT blocks and sampler steps, the exact softmax attention mass per
(head, 128-query block, 128-key block) of the real (post RMSNorm + RoPE) q, k, in fp32, saved with the token
segment table to <output>/<out_dir>/s<step>_b<block>.pt. harness/attn_mass.py analyses them (theta coverage,
sparsity, step-to-step stability of a frozen mask).
"""
import math
import os

import torch

import comfy.patcher_extension

BLK = 128


def _parse(spec):
    out = set()
    for part in filter(None, (p.strip() for p in spec.split(","))):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


@torch.inference_mode()
def block_mass(q, k, scale, rows=4096):
    """q, k: [H, S, D] (any float dtype) -> [H, nb, nb] fp32, nb = ceil(S / 128): for each query block, the mean
    over its rows of the softmax probability falling in each key block (each row of the result sums to 1)."""
    H, S, D = q.shape
    nb = math.ceil(S / BLK)
    sp = nb * BLK
    out = torch.empty(H, nb, nb, device=q.device, dtype=torch.float32)
    kp = torch.zeros(sp, D, device=q.device, dtype=torch.float32)
    for h in range(H):
        kp[:S] = k[h].float()
        for r0 in range(0, S, rows):
            r1 = min(S, r0 + rows)
            s = torch.mm(q[h, r0:r1].float(), kp.t())
            s.mul_(scale)
            s[:, S:] = float("-inf")
            p = torch.softmax(s, dim=-1)
            del s
            bs = p.view(r1 - r0, nb, BLK).sum(-1)  # [r, nb]
            del p
            n = r1 - r0
            nq = math.ceil(n / BLK)
            if n % BLK:
                bs = torch.cat([bs, bs.new_zeros(nq * BLK - n, nb)])
            blk = bs.view(nq, BLK, nb).sum(1)
            cnt = torch.full((nq, 1), float(BLK), device=q.device)
            if n % BLK:
                cnt[-1] = n % BLK
            out[h, r0 // BLK:r0 // BLK + nq] = blk / cnt
    return out


class RapidH3AttnMass:
    """Diagnostic: per-head 128x128 block attention mass of chosen blocks/steps (phase 1). Wraps the
    attention override like RapidH3DumpQKV (the fused rms+rope path then falls back to the bit-identical unfused one,
    so the output is unchanged); costs a few seconds per measured block."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "blocks": ("STRING", {"default": "0,5,12,20,28,36,44,49"}),
                             "steps": ("STRING", {"default": "0-5"}),
                             "out_dir": ("STRING", {"default": "mass"})}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Rapid-H3"

    def patch(self, model, blocks, steps, out_dir):
        import folder_paths
        m = model.clone()
        want_b, want_s = _parse(blocks), _parse(steps)
        d = os.path.join(folder_paths.get_output_directory(), out_dir)
        os.makedirs(d, exist_ok=True)
        state = {"step": -1}
        prev = m.model_options["transformer_options"].get("optimized_attention_override")

        def override(func, *args, **kwargs):
            opts = kwargs.get("transformer_options", {})
            blk = opts.get("block_index")
            q, k = args[0], args[1]
            if (state["step"] in want_s and blk in want_b and q.dim() == 4 and q.shape[2] >= 1024
                    and kwargs.get("skip_reshape")):
                scale = kwargs.get("scale") or q.shape[-1] ** -0.5
                mass = block_mass(q[0], k[0], scale)
                lay = opts.get("minimax_h3_layout")
                torch.save({"mass": mass.half().cpu(), "S": q.shape[2],
                            "segments": list(lay.segments) if lay is not None else None,
                            "step": state["step"], "block": blk},
                           os.path.join(d, f"s{state['step']}_b{blk:02d}.pt"))
                del mass
            return prev(func, *args, **kwargs) if prev else func(*args, **kwargs)

        def wrapper(executor, *args, **kwargs):
            state["step"] += 1
            return executor(*args, **kwargs)

        m.model_options["transformer_options"]["optimized_attention_override"] = override
        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "rapidh3_mass", wrapper)
        return (m,)


NODE_CLASS_MAPPINGS = {"RapidH3AttnMass": RapidH3AttnMass}
