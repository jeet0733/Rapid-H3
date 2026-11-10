"""nvcc -gencode flag for the GPU this process runs on.

The JIT builds in this pack (attn_int8, cutlass_fp8, vae_fast, bench/gemm_persistent) call gencode() instead of
naming an arch. One branch then builds on GB10 (sm_121, DGX Spark) and on Blackwell desktop cards (sm_120, RTX PRO 6000).

    gencode()      -> "-gencode=arch=compute_121,code=sm_121" on GB10, "...compute_120,code=sm_120" on sm_120
    gencode("a")   -> the arch-specific target, "...compute_121a,code=sm_121a" (for kernels that use it)
"""
import torch


def gencode(suffix: str = "", device=None) -> str:
    major, minor = torch.cuda.get_device_capability(device)
    arch = f"{major}{minor}{suffix}"
    return f"-gencode=arch=compute_{arch},code=sm_{arch}"
