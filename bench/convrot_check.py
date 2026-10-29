"""Does Comfy's int8_convrot weight equal per-row INT8 of W @ blockdiag(H256_sylvester)?"""
import sys
import torch
from safetensors import safe_open
sys.path.insert(0, "/work/custom_nodes/rapid_h3")
import triton_fp8 as tq  # noqa: E402
M = "/models/diffusion_models/"
name = "blocks.30.attn.out_proj"
with safe_open(M + "minimax_h3_fl2va_pruned_bf16.safetensors", "pt") as f:
    w = f.get_tensor(name + ".weight").cuda()
with safe_open(M + "minimax_h3_fl2va_pruned_int8_convrot.safetensors", "pt") as f:
    q = f.get_tensor(name + ".weight").cuda().float()
    s = f.get_tensor(name + ".weight_scale").cuda().float()
deq = q * s
for g in (64, 256):
    r = tq.convrot(w, group=g).float()
    cos = torch.nn.functional.cosine_similarity(r.flatten(), deq.flatten(), dim=0).item()
    rel = ((r - deq).norm() / r.norm()).item()
    print(f"group {g}: cosine {cos:.6f}  rel {rel:.4f}")
