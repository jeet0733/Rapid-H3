# CUDA sources (sm_121 / GB10)

| File | What | Used by the recipe? |
|---|---|---|
| `fp8_rowwise.cu` | CUTLASS 4.2.1 row-scaled GEMMs: INT8 (`int8_rowwise_gemm`, configs 1–15, default 6 = 128×256×64 with 4 stages) and FP8 (`fp8_rowwise_gemm`), with an EVT epilogue `acc · x_scale · w_scale (+ bias) → bf16`. Also `int8_gemm_gated_residual`: `resid += bf16(acc·xs·ws) · gate`, in place, bit-identical to ComfyUI's `addcmul_`. And `int8_gemm_swiglu`: fc1 with interleaved gate/up rows and a custom store visitor (`VisitorSwigluStore`) that writes `h = bf16(silu(g)·u)` at half width. JIT-built by `../cutlass_fp8.py`. | **Yes.** All INT8-rot linears; out_proj and fc2 use the gated form, fc1 the SwiGLU form. |
| `attn_int8.cu` | Our own INT8 flash attention, D=128, non-causal, batch 1: the pre-pass, the config dispatch and the older kernels (configs 0–88). JIT-built by `../attn_int8.py`. | **Yes**, through `attn_cake/`: `ts_h2` / `ts_s6` run config 190 (`RapidH3Int8Attention`). The other configs are measured experiments, see below. |
| `attn_cake/attn_cake.cuh` | Configs 94–99 and 90 / 89 / 79: config 80's math on a persistent CAKE-style schedule with a TMA ring and the softmax mutex. `#include`d by `attn_int8.cu`. Design and numbers: `attn_cake/DESIGN.md`. | **Yes.** Config 90 (as 190, with the q/k RMSNorm + RoPE fused into the pre-pass) since 2026-10-27; 94 (as 194) before that. |
| `vae_ops.cu` | Video-VAE decoder kernels: the to_out quantizer, comfy-kitchen's GEMM epilogues on our tiles, the w1 + SwiGLU GEMM and the fused q/k rms+rope + Q/K quantizer (level 4). JIT-built by `../vae_fast.py`. | **Yes.** `RAPIDH3_VAE_FAST=4` (default). |
| `rr_row.cuh` | Copies of `attn_int8.cu`'s per-row rms+rope / rotation / quantization helpers for the qkv GEMM epilogue (`int8_gemm_qkv_prep`, attention config 3xx). | **No.** Bit-identical but not faster, code kept. |
| `int8_gemm_p.cu` | Persistent INT8 row-scaled GEMM (128×256×64, 4 stages, pipeline runs across tiles, register-direct epilogue). Bit-identical to `int8_rowwise_gemm` config 6. JIT-built by `bench/gemm_persistent.py`. | **No.** 6–10% slower than CUTLASS; its main loop alone is 3–8% faster (docs/kernels.md, "Lossless kernels, round 3"). |

---

## `attn_int8.cu`: custom INT8 attention (the default in `ts_h2` / `ts_s6`; config 190 since 2026-10-27)

**Since 2026-10-27:** the presets use config 190 (= 194 with the two warp groups' softmax sections overlapped; bit-identical to 194, 1.8–2.4% faster per call; `attn_cake/DESIGN.md` §7). 194 = 94 with the q/k RMSNorm + RoPE fused into its pre-pass.

**2026-10-26:** config 80 (round 4) and config 94 (`attn_cake/`, config 80's math on a persistent schedule, bit-identical to it) are faster than comfy-kitchen: config 94 is −6.6% per call at S = 22,332 and −6.7% at 44,992, with lower error. From 2026-10-26 to 2026-10-27 `ts_h2` and `ts_s6` used it through the `RapidH3Int8Attention` node (config 94; `rapidh3_attn=194` with the q/k RMSNorm + RoPE fused into its pre-pass). Design and numbers: `attn_cake/DESIGN.md`. The 2026-10-13 bottom line below is kept as history.

**Bottom line (2026-10-13):** this kernel has **20–25% lower output error** than comfy-kitchen's INT8 attention, but it is **0–7% slower**. The FP8-P·V variant is 1.5% faster but less accurate. The recipe keeps comfy-kitchen. Full write-up: docs/kernels.md, section "Custom INT8 attention kernel".

### API
```python
import attn_int8                          # custom_nodes/rapid_h3/attn_int8.py
o = attn_int8.attention(q, k, v)          # q,k,v: [S, H, 128] bf16 (strided ok, last dim contiguous) -> [S, H, 128] bf16
o = attn_int8.attention_bhsd(q, k, v)     # comfy's [1, H, S, 128] layout
# kwargs: config=<int>, smooth_k=True, smooth_v=True, rotate=True; env RAPIDH3_ATTN_CONFIG sets the default config
```
The first call JIT-compiles with `-gencode=arch=compute_121a,code=sm_121a` into `$RAPIDH3_BUILD/attn`. This takes about 1–2 min, because every config below is instantiated.

### What it computes
1. **Pre-pass** (`stats_partial`/`stats_final`, `quant_qk`, `quant_v`). Together these take about 2.4 ms per 8 heads at S = 37.7k.
   - **K, V statistics.** Per-head channel means of K and V, plus the min and max of V. The two-level reduction is deterministic, with no float atomics.
   - **Q, K → int8, per token.** Each token gets its own scale after an in-register **128-point Hadamard rotation** (`hadamard128`). The rotation is exact for Q·Kᵀ and spreads outlier channels. K is mean-smoothed first, which shifts each softmax row by a constant. Each key also stores a `(scale, −1.5·2²³·scale)` pair (`KSB`), so dequantization and the tail mask fold into one FFMA. Padded keys get bias −∞.
   - **V → int8, per channel, mean-smoothed.** V is written transposed as `[H][128][Spad]`. Keys are **permuted inside each 32-key group** (`perm32`) so the Q·Kᵀ accumulator fragment can be packed straight into the A operand of the P·V mma, with no shuffles.
2. **Main kernel.** Q·Kᵀ is `mma.m16n8k32.s8.s8`. The softmax runs in exp2, and P is quantized to **u8 relative to each tile's own row max**: `2^(t·c − tm·c + log2 255)`. A per-tile scale `ts = 2^(tm − m)` folds it into the fp32 output. Quantizing against the running max instead gives about 4% error. P·V is `mma.m16n8k32.u8.s8`. Row sums come from the *quantized* P via `dp4a`, so numerator and denominator agree.
3. **Integer tricks.**
   - Accumulators start at `0x4B400000`, i.e. 1.5·2²³. Their bit pattern is then directly the float `1.5·2²³ + sum`, which avoids I2F for |sum| < 2²², and BN=128 still fits.
   - P → u8 uses the same magic-add, and `pack4` is two `PRMT`s.
4. **Epilogue.** `O / l · v_scale + v_mean` is stored as bf16 directly in the `[S][H][D]` layout.

### Configs (`config=`)
| id | kernel | idea | result (8 heads, S=37.7k) |
|---|---|---|---|
| **2** | `attn_fwd<128,2,8>` | 8 warps × 16 rows, BN=128, cp.async 2-stage, plain loop | **best INT8**: +5% vs ck, error −20% |
| **20** | `attn_fwd2<64,3,8>` | next tile's Q·Kᵀ issued before this tile's softmax, branch-free | +7% vs ck, lowest error |
| 0–12, 21–27 | `attn_fwd` / `attn_fwd2` | sweeps of BN 64/128, 2–4 stages, 4/8/12 warps, P·V in halves or quarters, `__launch_bounds__` minBlocks | none faster |
| 28, 29 | `attn_fwd2<…,SKEW>` | warps 4–7 do softmax before their Q·Kᵀ, to de-phase the two warps of a sub-partition | slower |
| 30–32 | `attn_fwd3` | FA3-style ping-pong (named barriers) plus TMA bulk copies and mbarriers; needs the pre-swizzled "tiled" K/V layout (configs 30–49) | slower |
| 40, 41 | `attn_fwd4` | warp pairs: split keys for Q·Kᵀ and D for P·V. Every K/V fragment feeds two m-tiles, halving smem traffic | slower; smem was not the limit |
| **52** | `attn_fwd5<128,3,8,…>` | **FP8 e4m3 P·V** accumulating straight into the fp32 output, so no per-tile fold. P is relative to the running max, V is e4m3 per channel (configs 50–89) | **−1.5% vs ck**, but error 1.0–1.6% (worse than ck) |
| 50, 51, 53–56 | `attn_fwd5` | pipelined, two-level accumulation, 12 warps | not better |
| 91–93 | `attn_fwd2<…,MODE>` | **profiling only**: 1 = no softmax, 2 = no output fold, 3 = no P·V | core without softmax ≈ 190 TFLOPS |
| 49 | `attn_fwd4<…,1>` | profiling only (no exp) | — |
| 60, 61 | `attn_fwd6<2/3>` | config 2's layout with ck's lean softmax: one K scale per 128-key tile (`quant_k_block`), an integer row max, one FFMA per score | no faster than config 2 (35.9 vs 36.1 ms main kernel, 8 heads); error 1.01% |
| 70, 71 | `attn_fwd7<64,3/4,8>` | config 20 with the output fold software-pipelined: P·V half 0 folded between half 1's IMMAs, half 1 carried into the next tile (2026-10-24) | bit-identical to config 20 but spills past 255 registers: 12–15% slower |
| **80** | `attn_fwd8<2,false>` | **round 4 (2026-10-24):** config 2's layout, reordered like ck's SASS: exp2 + u8 packing of 32-key chunk kc issued between the P·V IMMAs (half 0, kc-outer), half 0's fold between half 1's IMMAs; P rounded/packed with `cvt.pack.sat.u8` (F2IP, 2 per instruction) instead of magic FADD + PRMTs; per-key `(scale, bias)` FFMA dequant (config 20's); ldmatrix addresses = per-tile base + 4 hoisted lane offsets + immediates | **faster than ck**: −2% per call at S=45.0k, −2.7% at 22.3k (real H3 q/k/v); same error as config 2 (8–20% below ck). Loop: 774–809 SASS instructions per 128 IMMA (config 2: ≈1100 on the common path, ck: 851) |
| 81, 83, 85 | `attn_fwd8<…,DEFER>` | 80 with half 1's fold carried into the next tile's Q·Kᵀ | spills a little; 1–4% slower than 80 |
| 82, 84 | `attn_fwd8<3/2,…,INC>` | 80 with load/read positions advanced incrementally (no spills at 3 stages) | 3–6% slower than 80 (ptxas schedule), despite fewer instructions |
| 86–88 | `attn_fwd8<2,false,false,VAR>` | 80 with k-step-outer Q·Kᵀ (bit 0) and/or two max chains per row (bit 1) | within ±0.5% of 80 |
| **94–99, 90, 89, 79** | `cake::launch<…>` (`attn_cake/attn_cake.cuh`) | config 80's math on a persistent schedule; 94 = softmax mutex, 90 = 94 with the mutex entered after the row max | 94: −3.9–4.0% vs 80 at the 5 s shapes; **90: −1.8–2.4% vs 94, bit-identical** (`attn_cake/DESIGN.md`) |
| 100 + c, 200 + c | config c | `RapidH3Int8Attention` only: the q/k RMSNorm + RoPE fused into the pre-pass (1xx; 190 and 194 in the presets), plus a per-call check against the unfused path (2xx, slow) | bit-identical to config c |

Full 56-head results on real H3 q/k/v (`bench/results/attn_custom_56h.json`). Each error column is one of the three captured q/k/v dumps: step 1 block 25, step 5 block 5, step 5 block 45.

| kernel | ms / call | error vs fp32 (s1_b25 / s5_b05 / s5_b45) |
|---|---|---|
| comfy-kitchen | 266 | 0.82 / 0.92 / 1.23% |
| config 2 | 279 | 0.74 / 0.84 / 0.97% |
| config 20 | 285 | 0.67 / 0.81 / 0.93% |
| config 52 (FP8 P·V) | 262 | 1.03 / 1.51 / 1.59% |

### Why it didn't beat comfy-kitchen before round 4 (what we measured, 2026-10-13)
- **The hardware isn't the limit:**
  - raw IMMA m16n8k32 peaks at **252 TFLOPS** (`bench/cuda/mma_peak.cu`);
  - fed by `ldmatrix` from smem it still reaches **242** (`bench/cuda/ldsm_mma.cu`);
  - real data doesn't hit a power or clock limit (`bench/cuda/mma_occ.cu`).
- **Our loop is.** The attention loop with the softmax removed (config 91) reaches about **190 TFLOPS**, with the tensor pipe 73% busy. With the softmax it reaches about 165; comfy-kitchen's main kernel reaches 169.
- **Why:** the 255-register cap leaves only 2 warps per sub-partition, so the softmax and the per-tile output fold can't hide behind the other warp's MMAs.
- **What ncu shows:** no local-memory spills. In the no-softmax loop the stalls are tensor-pipe cadence (the `wait`/`math` stalls on IMMA and the NOPs between them) and about 14% `ldmatrix` queue pressure (`mio_throttle`).
- **`setmaxnreg`** compiles on sm_121a (`bench/cuda/setmaxnreg_test.cu`, it emits `USETMAXREG`). It can't raise a warp above 255, though, and we have no separate load warp to starve.
- **Instruction count:** comfy-kitchen issues about 19% fewer instructions per call. It uses per-block K scales and an integer-domain max, while we pay extra for per-token K scales and the rotation.

### Was open TODO, done 2026-10-24/26 (configs 80, 94)
Close the 190-versus-242 gap in the multiply (time-boxed to about 1 day):
- hoist the smem address math out of the inner loop;
- trim `ldmatrix` issue (SageAttention-style instruction buffer);
- diff register and instruction counts against comfy-kitchen's SASS.

Ceiling is about −1.5 s/step at 5 s 768p and about −5.5 s/step at 10 s, with no quality risk. Done as config 80 (address hoisting, F2IP packing, ck-like ordering; round 4) and config 94 (persistent schedule, main kernel 170–175 TFLOPS): −3.9–4.0% per call vs config 80 at the 5 s shapes. Details are in docs/kernels.md under "Lossless kernels, round 4" and `attn_cake/DESIGN.md`.

### Benchmarks
Run inside the `rapid-h3/h3-comfyui:v0.42.1` image with `PYTHONPATH=/work/custom_nodes/rapid_h3`.

- `bench/attn_custom.py <dumps> --ck --configs 2 20 52 [--heads 8]`: speed and error against an fp32 reference.
- `bench/attn_prof.py <dump> 8 2,20`: per-kernel CUDA time, comfy-kitchen included.
- `bench/attn_emulate.py <dump>`: torch emulation of each error source (Q/K/V/P quantization choices).
- `bench/attn_one.py <dump> 2 <config|ck>`: a single call, for `ncu`.

The q/k/v dumps come from the `RapidH3DumpQKV` node: real H3 activations, `[1, 56, S, 128]` bf16.
