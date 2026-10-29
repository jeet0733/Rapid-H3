# Rapid-H3 kernels: faster H3 steps without losing quality

The kernel layer of Rapid-H3: what makes each MiniMax-H3 denoising step faster on a DGX Spark (GB10, sm_121) without changing what the model computes beyond 8-bit rounding. The render presets built on it are in `../README.md`.

## Current kernel stack (2026-10-27, used by `ts_h2` / `ts_s6`)

| Part | Kernel | Where |
|---|---|---|
| 200 block linears | INT8 W8A8, H256 ConvRot rotation, CUTLASS mma.sync for sm_121 at ≈200 TFLOPS; quantizers, SwiGLU and gated residual fused | `RapidH3LinearQuant` (`int8_rot`, `fuse_block`), `csrc/int8_gemm*.cu`, `kernels.py` |
| Attention | our INT8 attention config 190 (since 2026-10-27; was 194, bit-identical outputs): config 80's math on a CAKE-style persistent schedule (config 90 = config 94 with the softmax mutex entered after the row max) with the q/k RMSNorm + RoPE fused into its pre-pass; bit-identical final latents to config 94 | `RapidH3Int8Attention` (`rapidh3_attn=190`; `194` runs the previous kernel), `csrc/attn_cake/DESIGN.md` |
| Text encoder | Qwen3-VL-32B linears on the INT8 W8A8 path with a residual pass on outlier rows; conditioning cos ≥ 0.9991 vs fp32 | `RapidH3TextEncoderQuant` (`te_quant="int8_rot"`) |
| Video VAE decode | int8_convrot VAE, bit-identical trims, all 28 tiles of a temporal chunk per call, SwiGLU in the w1 GEMM epilogue (61–62 dB vs level 1); level 4 adds the q/k rms+rope fused into the attention quantizer (bit-identical to level 3, −1.0 s per decode) | `vae_fast.py`, `csrc/vae_ops.cu`, `RAPIDH3_VAE_FAST=4` (default since 2026-10-27) |
| Timing | synced per-step timer (ComfyUI's progress bar reads ≈12% low) | `RapidH3StepTimer` |
| Accuracy check | single-step divergence against bf16 and a 1-ulp noise floor | `RapidH3Shadow` |

Per full-size step at 141 frames 768p: 56.7 s stock bf16 → 29.6 s (1.9×; `ts_s6`, 2026-10-27 timing run in `runs/2026-10-27/release_timing/`). The sections below are the record from the first recipe (2026-10-12, 8-step Turbo renders, 1.76× per step) through each later round; later rounds supersede earlier numbers where they differ.

## At a glance: original H3 vs ComfyUI int8_convrot vs Rapid-H3

Same render for all three: FL2VA text → video + audio, 1344×768, 124 frames (5 s), 8-step Turbo LoRA, models kept loaded. Step times are synced (`RapidH3StepTimer`), not ComfyUI's progress bar. *(Historical: the first recipe, 2026-10-13 to 10-23, on ComfyUI 0.39. The `ts_h2` / `ts_s6` presets use no Turbo LoRA and run on ComfyUI 0.42.1; their per-step numbers are above the table.)*

| | Original H3 (stock ComfyUI 0.39) | ComfyUI int8_convrot (Comfy-Org's int8 release, stock ComfyUI) | **Rapid-H3** |
|---|---|---|---|
| DiT weights | pruned bf16, 40.2 GB | pruned int8_convrot, 21.0 GB | pruned bf16 quantized at load (≈4 s), or the int8_convrot file adopted as is |
| 200 block linears | bf16 cuBLAS, ≈100 TFLOPS | INT8 W8A8 with ConvRot rotation, Comfy's kernel (≈108 ms per GEMM, slower than bf16's ≈79 ms) | INT8 W8A8 with the same ConvRot rotation, our CUTLASS sm_121 kernel at ≈200 TFLOPS, quantizers and SwiGLU / gated residual fused into it |
| Attention | bf16 PyTorch flash, ≈90 TFLOPS | bf16 PyTorch flash | comfy-kitchen INT8, ≈150 TFLOPS (superseded: the ts_h2 / ts_s6 presets use our INT8 attention, config 94 from 2026-10-26 and config 190 since 2026-10-27, `custom_nodes/rapid_h3/csrc/attn_cake/DESIGN.md`) |
| Video VAE decode | fp16, ≈37 s | int8_convrot | int8_convrot plus bit-identical trims, ≈18 s (124 frames; ≈21.1 s at 141 frames, `bench/results/pyopt_decode.json`) |
| **s / step** | **42.2** | **45.1–45.4** | **24.0** |
| **Render, models loaded** | **378 s** | **387 s** (2 prompts, 387–396 s) | **219 s** |
| **vs original** | 1.00× | **0.93× per step, 0.98× per render** (slower) | **1.76× per step, 1.73× per render** |
| Per-layer output error | reference | 1.35% | 1.4% (same rotation) |
| Single-step divergence vs bf16 (× the 1-ulp noise floor, video / audio) | 1.0 / 1.0 by definition | not measured | 1.7 / 2.1 |
| Same seed, same picture as bf16? | — | no: any 8-bit change gives a different but equally plausible clip | no, same reason; side-by-side review of 6 prompts (videos not in the repository) |

- ComfyUI int8_convrot measured 2026-10-23 (`i8cr768`, prompts 1–2); at 832×480 it was likewise ≈12% slower per step than bf16.
- **Why Comfy's int8 file isn't faster:** its weights are int8, but its GEMM kernel is slow on sm_121 and attention stays bf16. Rapid-H3 can load that same file (`dit=int8_convrot`) and runs it at 25.0 s/step, the same as from bf16; the difference is entirely in the kernels.
- At 10 s (243 frames), Rapid-H3 is **1.78×** per step (125.5 → 70.5 s); at 832×480 it is 1.64×.
- Steps are a free choice and multiply through: at 10 base steps (no Turbo LoRA) a 768×1344, 141-frame clip takes ≈350 s with Rapid-H3.


---

## Results and kernel log (from 2026-10-12)

### The recipe
*(2026-10-12 to 10-13: the first recipe, with the 8-step Turbo LoRA. The current presets are in `../README.md` and the table at the top.)* Everything went through ComfyUI 0.39 (the release image runs ComfyUI 0.42.1, see the note below) with our node pack `custom_nodes/rapid_h3`, which needs no changes to ComfyUI itself:

> **ComfyUI ≥ 0.42 (2026-10-23):** `TensorWiseINT8Layout` moved from `comfy.ops` to `comfy.quant_ops`. `vae_fast.py`
> `_linear` now looks it up in `comfy.quant_ops` first and falls back to `comfy.ops`, so it runs on 0.39 and 0.42.
> Nothing else Rapid-H3 uses moved (comfy-kitchen 0.2.39). Validated (ComfyUI 0.42.1) on
> 2026-10-23: a 1344×768, 107-frame turbo8 fl2va clip rendered bit-identical to its 0.39 reference (same `framemd5`,
> SSIM 1.000, PSNR inf). It took 240–250 s from submit to download against 241.7 s on 0.39 (8 steps at about 21 s/it,
> `vae_fast active`), and at least 23.8 GB stayed available.

| Component | Stock | Ours |
|---|---|---|
| 200 block linears (qkv, out_proj, fc1, fc2) | bf16 cuBLAS, about 100 TFLOPS | **INT8 W8A8 with a block-128 Hadamard rotation**. Per-token × per-channel scales are applied in the epilogue of a **CUTLASS mma.sync kernel built for sm_121** (`csrc/fp8_rowwise.cu`), at about 195 TFLOPS. |
| Norm → modulation → quantize | 3 bf16 passes plus a quantizer | **One Triton kernel**: RMSNorm, (1+scale)·h+shift, rotation and per-row INT8 (`norm_mod_rot_quant_rows_i8`) |
| SwiGLU → fc2 input | eager SwiGLU, then a quantizer | **One Triton kernel** (`swiglu_rot_quant_rows_i8`) |
| Attention | PyTorch flash (sm80 FA2), about 90 TFLOPS | comfy-kitchen INT8 attention (`ModelAttentionBackend`), about 150 TFLOPS. Superseded: the ts_h2 / ts_s6 presets use our own INT8 attention (`RapidH3Int8Attention`, config 94 from 2026-10-26, config 190 since 2026-10-27, `custom_nodes/rapid_h3/csrc/attn_cake/DESIGN.md`) |
| Video VAE decode | fp16, about 37 s at 768p / 5 s | Comfy-Org `int8_convrot` VAE, about 22 s at 124 frames (≈18 s with the trims below; ≈21.1 s with them at 141 frames, `bench/results/pyopt_decode.json`). On the same latent it gives PSNR 42.7 dB and SSIM 0.988 against fp16. |

Weights are quantized on the fly from the stock pruned bf16 DiT, after any LoRA is merged (the Turbo LoRA in this first recipe; the presets use none), so any LoRA works. That takes about 4 s for all 200 linears.

The rotation is the exact regular-Hadamard H256 ConvRot that Comfy uses: W·Hᵀ in groups of 256, with per-row INT8. We checked it against the shipped `int8_convrot` weights (cosine 0.99995). So the node can also run **Comfy-Org's stock `int8_convrot` DiT directly**, adopting its INT8 weights and scales as they are with no bf16 copy: `dit=int8_convrot`. That is a 21 GB load instead of 40 GB, and resident memory drops from about 95 GB to about 76 GB.

### Speed
Measured on a DGX Spark (GB10). FL2VA text→video+audio, 1344×768, 124 frames (5 s), 8-step Turbo, seed 0. **s / step** is the median synced step time from `RapidH3StepTimer`. "Current" in this section means the recipe as of 2026-10-13.

| Configuration | s / step | Render, cold server | Render, models kept loaded | vs bf16 (step / cold / loaded) |
|---|---:|---:|---:|---:|
| Stock bf16 (ComfyUI 0.39) | 42.1 | 402 s | 375 s | 1.00× |
| **Current recipe** (INT8-rot linears + fused quantizers + INT8 attention + INT8 VAE) | **25.0** | **261 s** | **228 s** | **1.68× / 1.54× / 1.64×** |
| Same recipe from the stock `int8_convrot` checkpoint (`dit=int8_convrot`) | 25.0 | 255 s | — | 1.68× |
| + bit-identical fusions: gated residual in the GEMM epilogue, out_proj quantizer reading the attention output in place, retuned GEMM tiles | 24.3 | — | 221 s | 1.73× / — / 1.70× |
| **+ SwiGLU in the fc1 GEMM epilogue (current)** | **24.0** | — | **219 s** | **1.76× / — / 1.73×** (bf16 re-measured with models loaded: 42.2 s/step, 378 s) |

- **Models kept loaded**, i.e. `--cache-ram`, is now the `serve.sh` default. Step time is unchanged at 25.0 s. What's saved is the reload, and the first step drops from about 48 s to about 26 s, so a render takes 228 s instead of 261 s.
- A repeat measurement ran about 2% faster: bf16 41.3 s and recipe 24.8 s per step, 258 s cold and 225 s loaded.
- Run-to-run spread is about 0.5%. Prompt choice doesn't change step time.
- Per-step speedup for other sizes, synced, single prompt:
  - **832×480:** bf16 10.95 s → recipe 6.85 s per step (**1.60×**). The whole render goes from 126 s to 99 s. With the fusions it's **6.66 s per step (1.64×)**.
  - **10 s** (1344×768, 243 frames, about 75k tokens, 4-step timing run): bf16 125.5 s → recipe 72.8 s per step (**1.72×**). The render goes from 604 s to 373 s. With the fusions it's 71.2 s per step, and with the SwiGLU epilogue **70.5 s per step (1.78×)**.
  - At 10 s, attention is 51.5 of the recipe's 72.8 s, and the int8 attention kernel is now the limit.
- Intermediate recipes, from the earlier progress-bar timing, which reads about 12% low. Use them only as a ranking:
  - INT8 attention only: 29.7.
  - FP8 row-wise linears + INT8 attention: 24.0.
  - INT8-rot linears + INT8 attention + INT8 VAE: 23.4.
  - Current recipe: 21.4.
  - bf16: 37.9.
- About 55 s of a cold 261 s render is outside the sampler: text encoder, VAE, and model load and init in the first step. With models kept loaded that falls to about 27 s. So **50-step renders get close to the full 1.68× per step.**
- Side-by-side videos of all 6 prompts were reviewed (not in the repository).

#### How we time a step (and why the old numbers were off)
ComfyUI's tqdm `s/it` is not a step time. Nothing in the sampler synchronizes with the GPU, so the host enqueues work up to a full command buffer ahead. The bar then closes about one step before the GPU finishes, and the leftover work is billed to VAE decode. On an 8-step render it reads about 12% low: 21.8 s/it for a 25.0 s step. Two runs with and without a per-step sync took the same wall time (246 s against 243 s), while the progress bar showed 21.8 and 24.9 s/it.

`RapidH3StepTimer` syncs the device before and after each step. The harness adds it to every graph, and `render.py` reports its `step` block: `first_s`, `step_s` (median of steps 2..N) and `sampling_s`. The sync itself costs nothing measurable.

**Where a 25 s step goes** (CUDA events per block, `RAPIDH3_TIMING=1`):

| Phase | Time | Share | What's in it |
|---|---:|---:|---|
| Attention | 13.9 s | 56% | int8 attention, including its q/k/v quantization |
| Before attention | 3.1 s | 12% | norm + modulation quantizer, qkv GEMM, rms/rope |
| After attention | 7.7 s | 31% | out_proj, fc1, SwiGLU quantizer, fc2, gated residuals |

AdaLN and the work between blocks are negligible.

### Quality
**Same-seed final-video PSNR/SSIM is not usable here.** bf16 is bit-deterministic across runs. But *any* 8-bit change makes 8-step same-seed renders diverge chaotically. For example, the earlier team's "visually equal" INT8 attention lands at PSNR ≈ 22 dB and SSIM ≈ 0.77 against bf16.

So we measure **single-step divergence** instead (`RapidH3Shadow` node):
- Every denoising step runs twice on identical inputs: once with the candidate, once with bf16 / flash attention.
- We record ‖v − v_bf16‖ / ‖v_bf16‖ for the video and audio velocities.
- The trajectory follows bf16, so every variant is measured on the same states.
- The yardstick is a **noise floor**: bf16 evaluated on inputs jittered by one bf16 ulp.

Results at 832×480, mean over prompts and the 8 steps:

| Variant | Prompts | Video | Audio | × floor (video / audio) |
|---|---:|---:|---:|---|
| **Noise floor** (1-ulp input jitter) | 3 | 2.24% | 1.18% | 1.0 / 1.0 |
| INT8 attention only (earlier team: "visually equal") | 6 | 3.05% | 1.74% | 1.4 / 1.5 |
| INT8-rot linears only | 6 | 3.90% | 1.82% | 1.7 / 1.5 |
| INT8-rot linears + INT8 attention | 3 | 3.78% | 2.44% | 1.7 / 2.1 |
| **Current recipe** (fused, H256 ConvRot) | 3 | 3.92% | 2.49% | 1.7 / 2.1 |
| same with a block-128 Sylvester rotation (earlier) | 3 | 3.94% | 2.40% | 1.8 / 2.0 |
| FP8 row-wise linears + INT8 attention (rejected) | 1 | 4.34% | 3.68% | 1.9 / 3.1 |

- **Per-layer output error on real activations:** INT8-rot is ≈1.4%, against 3.2% for FP8 row-wise (e4m3 has only 3 mantissa bits). The stock Comfy int8_convrot checkpoint is 1.35%.
- **Ablation:** quantizing any single linear class alone already gives ≈3.0–3.5%. The error is not concentrated in one class, so keeping out_proj in bf16 buys nothing here.
- **Side-by-side review:** the review set had all 6 prompts at 768p, stock bf16 on the left and the recipe on the right, plus the full-resolution originals with their own audio. Same seed still gives a different composition after 8 steps, as any 8-bit change does. In a look at sampled frames, faces, wrinkles and fine texture, the legible "OPEN" neon text and motion all hold up equally well. At the time this still needed a viewing and listening check, especially on speech, before being called lossless, with a wider blind A/B (≈20 prompts, 10 s clips) as the remaining sign-off step. *(2026-10-12. The later presets were scored against stock bf16 on the eval1 / motion1 shot sets.)*

### How to run it (first recipe, 2026-10-12)
*Superseded: to render with the current presets, see `../README.md`. Starting from an empty Spark: `../REBUILD.md` (Docker image recipe, weight files with sha256, kernels, metrics venv).*

```bash
# on a Spark (CUTLASS v4.2.1 cloned into ~/rapid-h3/third_party/cutlass; the first run JIT-builds the extension, ~1 min)
~/rapid-h3/serve.sh start                          # ComfyUI on 127.0.0.1:8189, shared weights mounted read-only
python3 harness/render.py --variant demo --prompts harness/prompts.txt --only 1 \
  --set width=1344 height=768 quant=int8_rot attention="comfy kitchen attention" video_vae=int8
```

In a ComfyUI graph, the current recipe is:

```
UNETLoader(pruned bf16) → LoRA → MiniMaxH3SigmaShift → ModelAttentionBackend(comfy kitchen attention)
  → Rapid-H3 Linear Quant(mode=int8_rot, fuse_block=true)
```

Decode with the `int8_convrot` video VAE.

### Repo map
- `custom_nodes/rapid_h3/`: the ComfyUI node pack.
  - `RapidH3LinearQuant`: quantized linears and the fused block.
  - `RapidH3Shadow`: single-step divergence.
  - `RapidH3Profile`: per-step kernel profile.
  - `RapidH3DumpQKV`: saves real q/k/v for kernel benchmarks.
  - `RapidH3SaveLatent`: saves the nested video+audio latent.
  - `RapidH3Int8Attention`: our INT8 attention (configs 190 / 194 in the fused block).
  - Kernels: `kernels.py`, `triton_fp8.py`, `cutlass_fp8.py`, `csrc/fp8_rowwise.cu`; `attn_int8.py`, `csrc/attn_int8.cu`, `csrc/attn_cake/`; `vae_fast.py`, `csrc/vae_ops.cu`.
- `harness/`:
  - `workflow.py`: graph builder.
  - `render.py`: submit, time and record.
  - `score.py`: same-seed metrics.
  - `shadow_summary.py`, `layer_errors.py`.
  - `ablate.sh`.
- `harness/render.py` records `step` (synced step times from the `RapidH3StepTimer` node, added to every graph by `workflow.build(timer=True)`) next to tqdm's `sampler`. Quote `step.step_s`.
- `bench/`: microbenchmarks (GEMM, attention, accumulation, fused kernels) and `results/`.
  - `gemm_tune.py` (INT8 tile sweep), `fusion_test.py` and `swiglu_epi_test.py` (exactness of the epilogue fusions), `vae_bench.py` (VAE decode profile), `vae_bm.py` (VAE decode by level, `--profile`), `vae_r5_test.py` (level-4 exactness), `quant_bw.py` (quantizer bandwidth).
  - `attn_custom.py`, `attn_prof.py`, `attn_emulate.py` and `cuda/` cover the custom attention kernel: speed and error, per-kernel profile, an error-source emulation, and raw IMMA/ldmatrix peak tests.
- `custom_nodes/rapid_h3/csrc/attn_int8.cu` and `attn_int8.py` hold the custom INT8 attention kernel. It was experimental when this map was first written (2026-10-13); config 94 / 194 (from 2026-10-26) and config 190 (since 2026-10-27) are the attention in `ts_h2` / `ts_s6`.
- `serve.sh`: our own ComfyUI container (`rapid-h3-comfy`) from the `rapid-h3/h3-comfyui:v0.42.1` image built from `docker/` (`docker/README.md`).

### Things that did not pan out (so nobody re-tries them blindly)
- **FP8 row-wise linears**: just as fast as INT8, but 2.3× the per-layer error of rotated INT8 and worse single-step divergence. FP8 row-wise also needs our own kernel, because PyTorch's aborts on sm_121.
- **FP16 accumulation** (`allow_fp16_accumulation`): no speedup on GB10. Unlike GeForce Blackwell, fp16 and bf16 both run at ≈100 TFLOPS.
- **SageAttention 2.2** (built from source for sm_121): no faster than comfy-kitchen INT8 attention, and higher error on synthetic data.
- **Triton FP8 `tl.dot`** on sm_121: about 105 TFLOPS, i.e. it runs at the bf16 rate. Use CUTLASS mma.sync.
- **Two-Spark sequence parallel**: built, but dropped because the target is a single Spark. A direct 200G ConnectX-7 link measured ≈8.7 GB/s per direction (NCCL over RoCE).
- **SageAttention3** (FP4 microscaling, built for sm_121a) on real H3 q/k/v: 148–160 TFLOPS, so no faster than INT8 on GB10, with 8–27% output error against INT8's ≈1%. Not usable.
- **"Resident mode slows steps" was a measurement artifact, not a slowdown.** With synced timing, steps take 25.0 s whether models stay loaded or not:
  - cold 24.96 s and loaded 25.01 s on one Spark;
  - 24.6–24.8 s for three back-to-back loaded renders in a repeat run.

  The old 21 s versus 25 s gap came from the progress bar, which reads low only when the first step also pays for model load (see "How we time a step"). An in-process probe of fixed kernels (int8 attention, INT8 GEMM, a 512 MB copy) timed the same in both modes, and clocks and power matched. Keeping models loaded is now the default, at 228 s per render instead of 261 s. `--highvram` is still worse: 600 s for the first render, 320–370 s after that.
- **Fused-modulation node** from the earlier recipe: no longer exists in ComfyUI 0.39, because in-place modulation is now native. That is part of why stock bf16 is 38 s/step here against 43 s before.

### Custom INT8 attention kernel (2026-10-13): built, accurate, not faster
`custom_nodes/rapid_h3/csrc/attn_int8.cu` (`attn_int8.attention(q, k, v)`) is a hand-written sm_121 flash-attention kernel for D=128, non-causal. It uses INT8 Q·Kᵀ and u8·int8 P·V on `mma.sync`, with its own quantization pre-pass.

**Accuracy is better than comfy-kitchen's.** Three changes do it:
- Q and K get a 128-point Hadamard rotation (exact for Q·Kᵀ) plus per-token scales.
- K is mean-smoothed.
- P is quantized to u8 against each tile's own row maximum, with a per-tile rescale. This mirrors comfy-kitchen; quantizing against the running maximum gave 4% error.

**Speed matches it at best.** Real H3 q/k/v, all 56 heads, S = 37,728, three dumps (`bench/results/attn_custom_56h.json`):

| Kernel | ms / call | Output error vs fp32 |
|---|---|---|
| comfy-kitchen INT8 (current recipe) | 266 | 0.82 / 0.92 / 1.23% |
| ours, INT8 P·V, BN=128 | 279 (+5%) | 0.74 / 0.84 / 0.97% |
| ours, INT8 P·V, BN=64, software-pipelined | 285 (+7%) | 0.67 / 0.81 / 0.93% |
| ours, FP8 P·V (no per-tile fold) | 262 (−1.5%) | 1.03 / 1.51 / 1.59% |

**What we learned about the GB10 ceiling** (`bench/cuda/`, ncu):
- Raw IMMA m16n8k32 peaks at 252 TFLOPS.
- Fed by `ldmatrix` from shared memory it still reaches 242 TFLOPS.
- With real data it doesn't hit a power or clock limit.
- An attention-shaped loop with the softmax removed reaches only about 190 TFLOPS (tensor pipe about 73% busy). With the softmax it drops to about 165, and comfy-kitchen's main kernel runs at 169.
- The 255-register budget allows just 2 warps per sub-partition, so the softmax and the per-tile output fold can't be hidden behind the other warp's MMAs.

**Variants tried, none faster:**
- 8 or 12 warps, with BN 64 or 128 and 2 to 4 stages.
- Pipelining the next tile's Q·Kᵀ.
- Skewing the softmax between the two warps of a sub-partition.
- FA3-style ping-pong with named barriers plus TMA bulk copies and mbarriers.
- Warp pairs that split keys for Q·Kᵀ and D for P·V. This halves shared-memory traffic per IMMA, but shared memory was not the limit.
- FP8 P·V, which needs no output fold.

**Why it's not wired in:** it would trade about 0.5 s/step for lower attention error. That is the wrong direction for this project unless quality sign-off asks for it. *(2026-10-13; superseded: configs 80 and 94 are faster than ck and as accurate, config 94 was the default from 2026-10-26 and config 190 is since 2026-10-27, see "Lossless kernels, round 4", "Attention, round 6" and `custom_nodes/rapid_h3/csrc/attn_cake/DESIGN.md`.)*

### Sol-H3 (NVIDIA): what it does, and what's worth taking (2026-10-13)
Sol-H3 is NVIDIA Sana/Sol-Engine's H3 runtime ([code](https://github.com/NVlabs/Sana/tree/sol-engine/models/minimax_h3), [Spark page](https://nvlabs.github.io/Sana/Sol-Engine/Sol-H3-Spark/)). We are **not** adopting it. Nearly all of its speed comes from changes that alter the output:

| Sol-H3 ingredient | Lossless? | Us |
|---|---|---|
| Sol-Attn / VSA sparse attention (Spark draft: 90% sparsity; skipped blocks approximated from pooled K/V) | ✗ approximate | skip. A ComfyUI issue ([#16382](https://github.com/comfy-org/ComfyUI/issues/16382)) reports ripples and broken textures with sol-attn that disappear when it is turned off. |
| FirstBlockCache (reuse the previous step's residual for 49 of 50 blocks when block 0 barely changes) | ✗ skips compute | skip |
| FastH3 4-step adapter, plus a 384p H3 draft refined by **LTX-2.5** with a fixed generic prompt and a learned VAE adapter (Spark recipe: 56 s per 5 s 768p clip) | ✗ a different model renders the final pixels | skip. Its own docs say "no perceptual quality gate exists" and that cached generic conditioning is "not a guarantee of unchanged identity". |
| NVFP4 AWQ text encoder | ✗ lossy | skip. Our text encoding is only a few seconds per render once loaded. |
| FP8 W8A8 DiT (fc2 weight-only) | ≈ | ours is INT8-ConvRot, which is more accurate (1.4% vs 3.2% per layer) and as fast |
| Fused QKV/quant/AdaLN/RoPE/SwiGLU kernels, required to be bit-identical | ✓ | done the same way, `_fused_block` |
| Rank-8 AdaLN table | ✓ | already in the pruned checkpoint; AdaLN takes 3 ms per step |
| Batched VAE tile decode (bit-identical) | ✓ | ComfyUI 0.39 already batches up to 4 tiles per call |
| Models resident, no host syncs, NHWC VAE output | ✓ | resident is our default; we use the int8 VAE |

Their GB10 headline (480p, 50 steps, 710 s → 181 s, 3.9×) comes from sparse attention plus FirstBlockCache, measured against a baseline runtime of their own, and they publish no quality metrics. Sol-H3 has no lossless technique we don't already have. The one useful confirmation is their bit-identity discipline for fused kernels: an approximate `/` in Triton flips about 0.9% of FP8 rounding midpoints. Our INT8 quantizers compute `1/scale` with Triton's approximate division before `rint`. That's harmless here, because our gate is measured step error, not bit-identity to an eager path. Switching to `div_rn` would be an optional cleanup.

#### 2026-10-24 update: timings, requirements, user feedback
Sources: the code at `sol-engine` commit `41866cd` (2026-10-23), read locally; NVIDIA's pages; GitHub issues; NVIDIA's published Spark clips (not in the repository). **Not measured by us** (reasons below). Verdict unchanged: not adopted.

**What the Spark recipe is.** `Sol-H3-Spark/configs/default.json`, frozen ("changing this record alone is rejected"). The CLI (`infer.py`) has no resolution, frame, step or quality options, only `--task t2va|fl2va|ref2va`, `--first-frame/--last-frame`, `--reference`. Steps:
- H3 draft: 4 updates of the FastH3 VSA LoRA (W8A8 FP8, 90% VSA sparsity) at **672×384, 124 frames**.
- Learned ×2 upscaler, then an H3→LTX latent adapter.
- **LTX-2.5 dev** + distilled LoRA: 3 updates from σ 0.909, Sol-Attn, with a fixed generic prompt ("4K, refined, high quality, cinematic detail, clean textures, natural motion.").
- LTX VAE decode → **1344×768 landscape, 121 frames** (`stage1.py` hardcodes `--num-frames 124`, `stage2.py` hardcodes 124/768/1344). The original H3 audio is muxed.
- **141 frames and 768×1344 are not supported.** LTX needs 8k+1 frames (137 or 145). Only an unofficial fork does other lengths and 9:16, [slogonomo/lf-studio](https://github.com/slogonomo/lf-studio), and it warns that the "LoRAs, upscaler and adapter were trained on landscape".
- First frame (FL2VA): supported, but it reuses the T2VA-only VSA LoRA. NVIDIA calls first-frame quality "a separate validation target", and says "neither pixel-exact endpoint reproduction nor unchanged identity is guaranteed".

| Config | Time | Included | Source | Claimed / independent |
|---|---|---|---|---|
| Spark T2VA, 1344×768, 121 f, audio, warm | **56 s** (56.17) | Fresh Qwen encode → muxed MP4. Excludes model load, warmup and the offline prompt-cache build. Shares: draft 35%, LTX refine 44%, LTX decode 12.5%, Qwen 2.4% | [Spark page](https://nvlabs.github.io/Sana/Sol-Engine/Sol-H3-Spark/), Sep 2026; arXiv 2609.35110 | claimed |
| Same, through ComfyUI (Sol-H3 Studio, two-stage) | 64.3 s (mean of 3) | Request → MP4, Qwen included, model load excluded | [Sol-H3-ComfyUI README](https://github.com/NVlabs/Sana/tree/sol-engine/models/minimax_h3/Sol-H3-ComfyUI), Oct 2026 | claimed |
| Same, after the crash fix | 54.5–56.1 s warm; **first request ≈6.5 min** (load + warmup) | request → MP4 | [Sana #541](https://github.com/NVlabs/Sana/issues/541), 2026-10-16 | independent |
| lf-studio fork, launch → first clip | 5 s T2VA 99 s, 5 s FL2VA 115 s, 15 s T2VA 238 s | cold | lf-studio README, Oct 2026 | independent (fork) |
| Own H3+LTX two-stage build, ComfyUI 0.34, Spark, 124 f | 73.3 s warm / 118.8 s cold (2200 MHz cap); 66.1 s warm uncapped | submit → file | [ai-muninn](https://ai-muninn.com/en/blog/h3-ltx-two-stage-dgx-spark), 2026-09-14 | independent, not NVIDIA's package |
| GB10 one-stage H3, 832×480, 124 f, 50 steps, Sol-Attn + FirstBlockCache | 710.6 → 181.3 s | warm | [H3-OnDevice](https://nvlabs.github.io/Sana/Sol-Engine/H3-OnDevice/) | claimed |
| Same, with the crash fix | 145.1 s (warmup 170.3 s) | | [Sana PR #517](https://github.com/NVlabs/Sana/issues/517), open | independent |
| **Ours, `ts_h2`** (as of 2026-10-24): 1344×768 / 768×1344, **141 f**, audio, FL2VA | **≈118 s warm, ≈255 s cold** (`ts_h2` since 2026-10-27: ≈117 s warm) | prompt → MP4, everything | this repo | measured |

At our length the Sol-H3 recipe would need 145 frames (+20% tokens), which only the unofficial fork offers. Extrapolated from the 56 s, that is roughly 70–75 s warm: an estimate, never measured by anyone. Memory: 116.9 of 119.7 GiB used with everything resident (2.8 GiB margin); lf-studio asks for ≈110 GiB free and the whole machine. FL2VA and Ref2VA memory are "not covered".

**User feedback, by our criteria** (no Reddit or YouTube reviews found; all of the following is from GitHub or NVIDIA's docs):
- **Lip sync: worse than the draft.** [Sana #535](https://github.com/NVlabs/Sana/issues/535) (2026-10-12), confirmed by an NVIDIA maintainer. Stage 2 re-noises the H3 audio, denoises it with the video and then discards it, so the lips follow audio nobody hears:
  - SyncNet LSE-C drops from 8.2 (draft) to 4.4 (shipped), or 7.8 with the fix, for one speaker, and from 6.5 to 3.5 (fix 5.2) for the other;
  - lips stay within ±1 frame 55–67% of the time vs 93–96% for the draft;
  - lips "move while nobody speaks".
  - The fix ([PR #543](https://github.com/NVlabs/Sana/issues/543)) is open and not GPU-validated. The sample is about 4 clips.
- **Identity, reference fidelity, voice.** NVIDIA: the generic refine prompt is "not a guarantee of unchanged identity"; "No identity, accessory or voice fidelity guarantee"; "Smaller references may lose fine identity or accessory details". ai-muninn: an LTX refine starting at 0.909 (NVIDIA's value) "redraws the face into a different person", so they used 0.78. Ref2VA voice cloning through ComfyUI produced gibberish ([ComfyUI #16155](https://github.com/Comfy-Org/ComfyUI/issues/16155)).
- **Texture ripples from sparse attention.** [ComfyUI #16382](https://github.com/Comfy-Org/ComfyUI/issues/16382) (2026-09-17, open): sol-attn with a turbo LoRA gives "severe widespread ripples and broken textures", fixed by disabling it. That report is ComfyUI's node, not the Spark package. ai-muninn saw no visible difference.
- **Flicker, LTX colour/look shift, prompt adherence:** no reports found either way.
- **Reliability:**
  - ComfyUI two-stage crashed on Spark (#541, fixed in #542);
  - GB10 FirstBlockCache crashes at step 38 (#517);
  - host lockups on Spark ([ComfyUI #16587](https://github.com/Comfy-Org/ComfyUI/issues/16587); ai-muninn lost power twice uncapped);
  - NVIDIA has not validated a clean install ("fresh_environment_validated": false).

**Our look at NVIDIA's own published Spark clips** (7 showcase clips, all 1344×768, 121 f, 32 kHz stereo AAC; frames 0/60/120 at full resolution):
- They are good at sheet size, with no visible ripples in full-res crops.
- Skin is smooth and slightly waxy: the LTX look, softer than H3's own texture.
- **Identity drift in the English dialogue clip** (`seaside-coffee-english`): the woman's face at frame 0 (rounder, blue eyes, red lips) reads as a different, younger woman by frames 60 and 120, beyond what her head turn explains. The man stays consistent.
- In `wildflower-firefly-portrait` the flower crown rearranges over the clip.
- `radio-reunion-mandarin` holds identity well.
- These are curated showcase clips at NVIDIA's own prompts, not our eval set, so no same-prompt or same-seed comparison with ours was possible.

**Why we did not run it:**
1. The LTX-2.5 weights are gated. No local copy was available.
2. ≈224 GB of downloads (including 66 GB of H3 diffusers transformer shards, not reusable from our ComfyUI single-file checkpoints) at ≈11 MB/s measured, ≈5.5 h.
3. Three separate environments: patched FastVideo + fastvideo-kernel + cuDNN BSA; LTX with a compiled all2all extension; an NGC Qwen container. Even NVIDIA has not validated building them fresh.
4. It cannot render our 141-frame or portrait clips anyway.

**What we take (lossless):** nothing new; expected saving 0 s.
- Its lossless parts (resident models, fused kernels, rank-8 AdaLN table, batched VAE tiles, chunked writing) we already have; the table above stands.
- Its speed comes from the 384p draft + 4-step distilled LoRA + 90% sparse attention + LTX re-render. Each part is lossy, and our own tests already rejected the same class of change (distilled drafts, 0.6× draft, sparse attention, LTX as refiner).
- **One check it prompted:** the refined video must see the same audio that is output. Ours does. Every two-stage ts_h / ts_h2 variant sets `freeze_audio=True` (`LTXVFreezeLatent` on the draft's audio latent, `harness/workflow.py`), so the refine conditions the lips on exactly the audio we decode. That is the same fix as Sana PR #543, so #535's failure mode does not apply to us.

### Speed work, round 2 (2026-10-13)
**1. Fusions: all bit-identical.** Latents match the previous recipe exactly on 2 prompts, and per-op checks are in `bench/fusion_test.py`. Steps went from **25.0 to 24.2–24.4 s**.
- **Gated residual in the GEMM epilogue.** `x += gate · (xq·Wᵀ)` for out_proj and fc2 now happens inside the CUTLASS epilogue (`int8_gemm_gated_residual`, EVT: acc·xs·ws → bf16 → fma with the gate and the residual → in-place store). One GEMM runs per modality segment. This removes 300 `addcmul_` launches and a 0.4 GB round trip per GEMM:
  - out_proj: 19.7 → 15.4 ms;
  - fc2: 39.5 → 33.3 ms.
- **The out_proj quantizer reads the attention output in place.** Attention now returns head-major (`skip_output_reshape`), and the Triton rot-quant kernel gathers its 256-column groups across heads (`rot_quant_heads_i8`). That drops the 0.24 s/step transpose copy: 11.2 → 6.2 ms per block.

**3. GEMM tiles** (`bench/gemm_tune.py`, `bench/results/gemm_tune.json`). Sweeping 15 CUTLASS 2.x tilings at the real shapes found 128×256×64 with 4 stages to be best or tied everywhere:
- qkv 203, out_proj 197 (was 188), fc1 200 and fc2 180 TFLOPS.
- 256×256, K=128 tiles, stream-K and other swizzles were all slower.
- About 200 TFLOPS (80% of the 252 peak) looks like the ceiling for this kernel style.

**2. Video VAE: no worthwhile lossless gain.** `bench/vae_bench.py` measured the int8 VAE decode at about 20 s warm, made up of:
- INT8 GEMMs: 10.8 s at about 157 TFLOPS. Our GEMM reaches only 160–189 TFLOPS at these M = 7k shapes.
- Activation quantizers: 3.2 s.
- int8 attention (S = 1.8k, D = 64): 3.0 s at about 62 TFLOPS.
- rope: 1.3 s.
- a transpose copy and `nan_to_num`: 1.4 s.

ComfyUI already batches tiles. Bigger batches (7 or 14 tiles) were slower (27 s), smaller ones were the same speed, and the batch size changes the bits (≤0.05 on [0,1] pixels). So batching isn't a free lever either. Swapping in our GEMM would save about 1 s per render.

**SwiGLU in the fc1 GEMM epilogue.** fc1's weight rows are interleaved as [gate 4j..4j+3 | up 4j..4j+3], so every 8-wide output vector of a thread holds matching gate/up pairs. A custom CUTLASS store visitor (`VisitorSwigluStore`) rounds them to bf16 exactly as the plain GEMM output was, computes `silu(g)·u` in fp32 and writes 4 bf16 values to the half-width `h [M, 14336]`. The out_proj-style quantizer then reads h, instead of the old `swiglu_rot_quant` reading the full fc1 output twice.
- **Exactness** (`bench/swiglu_epi_test.py`): h equals `bf16(silu(g)·u)` bit for bit. The one change is that h is now rounded to bf16 before the rotation, which the bf16 model does too. 1.3·10⁻⁷ of fc2's int8 input codes change.
- **Shadow quality:** single-step divergence over 3 prompts is 3.60% video / 2.45% audio, against 3.92% / 2.49% for the previous recipe. That's within run-to-run variation, so no degradation.
- **Speed:** fc1+SwiGLU+quantizer goes from 78.8 to 70.9 ms per block. The step goes from 24.3 to 24.0 s at 5 s, and from 71.2 to 70.5 s at 10 s.
- **Memory:** for the stock `int8_convrot` checkpoint, the interleaved copy of fc1 adds about 7.7 GB. Our own quantized weights are replaced in place instead.

**VAE decode trims** (`custom_nodes/rapid_h3/vae_fast.py` + `csrc/vae_ops.cu`, on by default, `RAPIDH3_VAE_FAST=0` disables). Two changes:
- **to_out quantizer.** It now reads the int8-attention output in its head-major layout and applies `nan_to_num` on load. This replaces a transpose copy, a `nan_to_num` copy and ck's quantizer, going from 0.73 to 0.20 ms per call. The per-row math is comfy-kitchen's own (copied, Apache-2.0, built with the same `--use_fast_math`).
- **Decoder GEMMs.** Our CUTLASS tiles carry comfy-kitchen's exact epilogues (bias, residual addcmul) wherever they are faster: qkv and w1 at 4 tiles per call, and w2 always. comfy-kitchen's own pick stays where it wins.

**Exactness:** every kernel matches bit for bit (`bench/vae_fast_test.py`), the full decode matches (`bench/vae_bench.py --fast --compare`), and so do the rendered videos frame by frame and their audio packets.

**Speed:** decode goes from 20.1 to 18.2 s (124-frame clip; ≈21.1 s at 141 frames). In the pipeline that is ≈1.8 s per render: prompt time 217.7 → 215.9 s.

The earlier ≈4–5 s estimate also counted swapping ck's other quantizers and the VAE attention, which can't be done bit-identically. The rope step inside the VAE has the same ck-coupling caveat as the DiT one.

**Comfy compiler is now off by default** (`serve.sh` passes `--disable-comfy-compiler`). With the compiler on and our current node pack, a 10 s render that follows a 5 s render in the same resident server crawls: about 300 s/step instead of 70. The GPU sits at about 21 W, waiting. With the compiler off, the same sequence runs at 23.9 then 70.5 s/step, and step times are otherwise unchanged (bf16 42.2, recipe 24.0). The pre-round-2 node pack did not trigger this, but the compiler gives no measurable speed here, so it stays off.

**4. Attention multiply: tried once more, no gain.** `attn_fwd6` (configs 60/61) uses ck's lean softmax: one K scale per 128-key tile, an integer row max, and one FFMA per score. It cut softmax instructions by about a third, but the kernel time didn't move (35.9 against 36.1 ms per 8 heads; ck is 34.7). Its error is 1.01%, between ck and our per-token version. So the softmax instruction count isn't what limits it. The limit is the QK→softmax→PV dependency chain with 2 warps per sub-partition, as measured earlier. Closing it needs a different execution model (Blackwell tcgen05/TMEM, which GB10 lacks), so attention stays on comfy-kitchen. *(Superseded by round 4 and config 94, the default from 2026-10-26; config 190 since 2026-10-27.)*

### Lossless kernels, round 3 (2026-10-24): nothing wired in
Time-boxed (about 4 h) attempt at the on-hold items: the attention multiply loop and a faster INT8 GEMM. Neither beat what the recipe already uses, so the recipe is unchanged and no end-to-end render was needed. Shapes are the ts_h ones: 17.7k tokens (928×544 draft) and 36.3k (1344×768), 56 heads.

**Why the tensor pipe can't be kept full** (`bench/cuda/imma_warps.cu`, `bench/results/imma_warps.txt`). This is a new measurement: IMMA throughput, one CTA per SM, with independent FFMAs mixed into each warp's IMMA stream:

| non-IMMA instructions per IMMA | 0 | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|---:|
| TFLOPS, 8 warps / SM | 245–247 | 240 | 238 | 231 | 208 |
| TFLOPS, 16 warps / SM | 247–248 | 243 | 240 | 233 | 213 |

- One warp per sub-partition already reaches 225–231, and two reach 245, so occupancy isn't the problem.
- Every extra instruction costs tensor throughput, even when it is independent and more warps are resident.
- ptxas pads back-to-back IMMAs from one warp with a NOP (attention) or a dummy `UIADD3` (CUTLASS). Those padding slots are the `wait` stalls ncu reports.
- What this means:
  - Our attention issues about 9 instructions per IMMA (config 20: 3 FFMA, plus FMNMX, MUFU, FADD, LDSM, PRMT, MOV), so about 210 TFLOPS is its ceiling even with perfect overlap. That is why it lands at 165.
  - The CUTLASS GEMM issues about 1 per IMMA, so it isn't instruction-bound.

**1. Attention multiply loop: no gain.** At the new shapes, comfy-kitchen is still ahead (`bench/attn_speed.py`, random q/k/v, `bench/results/attn_speed_r3.json`):

| ms / call | ck | config 2 | config 20 | config 52 (FP8 P·V) | config 70 (new) | config 71 (new) |
|---|---:|---:|---:|---:|---:|---:|
| S = 17.7k | 65.4 | 66.0 | 67.7 | 62.6 | 76.8 | 73.7 |
| S = 36.3k | 248.9 | 261.9 | 265.9 | 245.8 | 298.6 | 293.1 |

- **Per-kernel split at 36.3k:**
  - ck: main kernel 232.4 ms, prepass 18.4 ms.
  - Config 2: main kernel 242.2 ms (+4%), prepass 15.7 ms.
  - Our prepass is already memory-bound (197–229 GB/s). Even dropping the K-statistics read would save only about 2 ms, which can't close the main-kernel gap.
- **New: `attn_fwd7` (configs 70/71)** is config 20 with the output fold software-pipelined:
  - In config 20's SASS the last ~110 FFMAs of every tile (O = a·O + ts·PV) have no IMMA to pair with.
  - fwd7 folds P·V half 0 between half 1's IMMAs and carries half 1 into the next tile's Q·Kᵀ/softmax.
  - It is **bit-identical** to config 20 (`torch.equal`), but ptxas needs more than 255 registers and spills (config 20 already uses 248), so it is 12–15% slower.
  - The register file is the wall. Any overlap inside one warp needs a second set of accumulators.
- **Verdict:** this confirms round 2. On mma.sync with 255 registers per thread, the softmax's instruction count and the register budget cap our kernel below ck's lean one. Config 52 (FP8 P·V) stays the only faster variant, and it fails the accuracy gate. Attention stays on comfy-kitchen. *(Superseded by round 4 and config 94, the default from 2026-10-26; config 190 since 2026-10-27.)*

**2. INT8 GEMM: no gain.**
- **More CUTLASS 2.x variants** (`bench/results/gemm_tune_r3_36k.json`, M = 36.3k):
  - 128×128 tiles at 3 stages, so two CTAs fit on an SM and one's epilogue can overlap the other's main loop: 188–196 TFLOPS against 198–204.
  - Rasterization swizzle 16 or 32: same as 8, within the ±3% run-to-run noise. Swizzle 4 drops to 157.
- **ncu on config 6** (qkv, locked clocks):
  - The tensor pipe is 84–91% active.
  - Stalls are almost all `math_pipe_throttle` and `wait`, i.e. the pipe itself plus the IMMA padding.
  - About 8% of the time is per-tile prologue and epilogue, from comparing cycles at K = 2688 and K = 5376.
- **New: `csrc/int8_gemm_p.cu`, a hand-written persistent INT8 GEMM** (`bench/gemm_persistent.py`, `bench/results/gemm_persistent.json`):
  - Design: same 128×256×64 tile, 4 stages, 2×4 warps of 64×64. The cp.async pipeline runs straight through tile boundaries, and the epilogue writes bf16 pairs straight from registers.
  - **Bit-identical** to config 6 at every shape and both token counts.
  - It is still slower:

| TFLOPS | qkv | out_proj | fc1 | fc2 |
|---|---:|---:|---:|---:|
| CUTLASS config 6, 17.7k / 36.3k | 205 / 204 | 200 / 202 | 203 / 204 | 182 / 183 |
| persistent, 17.7k / 36.3k | 187 / 184 | 185 / 184 | 189 / 190 | 175 / 156 |
| persistent main loop only (no epilogue), 17.7k / 36.3k | 197 / 214 | 208 / 210 | 211 / 207 | 199 / 193 |

  - With the epilogue removed, the main loop runs 3–8% faster than the full CUTLASS kernel. That is the most a fully hidden epilogue could buy.
  - The register-direct epilogue costs 10% instead. All 8 warps run it at once, so the tensor pipe idles, and it issues 4× more store instructions than CUTLASS's smem-staged 16 B stores (`lg_throttle`).
  - Hiding it properly needs warp specialization: a load warp plus two consumer groups taking turns on the epilogue. CUTLASS 4.2.1 has that only for F8F6F4 on sm_120 (its sm120 builder asserts on int8), so it would be a hand-written kernel of its own.
  - At best that is +5% on the GEMMs: about 0.35 s/step at 5 s 768p and about 1.5 s per ts_h clip. That isn't worth the risk inside this time box.

**3. rms+rope in the q/k quantizer: not attempted.** rms and rope are already one in-place ck kernel (`rms_rope_split_half_`). Folding them into the quantizer saves one read+write of q and k, about 0.35 s/step at 5 s. But ck's quantizer (`sage_sdpa_quantize`: signed H128, sampled K anchor, V quantizer) ships only as a binary. A bit-identical rewrite would track ck internals, the coupling we already declined.

**Net (round 3):** no lossless kernel lever is left that's worth its risk on GB10's mma.sync path (round 4 below then found one in attention). Both new kernels are kept as documented experiments (`attn_int8.cu` configs 70/71, `csrc/int8_gemm_p.cu`). Neither is loaded by the node pack.

### Lossless kernels, round 4 (2026-10-24): our attention now beats comfy-kitchen, opt-in then (default as config 94 from 2026-10-26, config 190 since 2026-10-27)
Time-boxed (about 6 h). **Config 80 (`attn_fwd8`) is faster than ck at both ts_h2 shapes and keeps our accuracy.** It was not wired in at the time; its bit-identical successors are the attention in ts_h2 / ts_s6: config 94 (as 194) from 2026-10-26, config 190 since 2026-10-27 (`custom_nodes/rapid_h3/csrc/attn_cake/DESIGN.md`).

**Real token counts.** Dumped from an er13 render (`RapidH3DumpQKV`): the attention sequence is **S = 22,332** for the 0.7 draft and **S = 44,992** for the full-size refine. Round 3's 17.7k / 36.3k were the video-only counts. Landscape and portrait have the same S.

**Real H3 q/k/v, 56 heads** (`bench/attn_r4.py`, `bench/results/attn_r4_real.json`; blocks 5/25/45; error is ‖o − o_ref‖/‖o_ref‖ against fp32 attention on the same bf16 inputs):

| | ck ms/call | config 80 ms/call | main kernel ck → 80 | error ck | error 80 | error bf16 SDPA |
|---|---:|---:|---:|---:|---:|---:|
| S = 44,992 (refine) | 385–389 | **378–380 (−2.0%)** | 362–367 → 357–363 | 0.90 / 0.95 / 1.23% | **0.81 / 0.87 / 0.98%** | 0.16% |
| S = 22,332 (draft) | 102.2–102.3 | **99.2–99.7 (−2.7%)** | 90–93 → 89–90 | 0.99 / 1.05 / 1.14% | **0.89 / 0.92 / 0.98%** | 0.16% |

Against bf16 SDPA's output the picture is the same (0.83–0.99% vs ck's 0.92–1.24%). On random q/k/v the result matches: at 36.3k it is 242.9 vs 249.8 ms, main kernel 227.0 vs 231.8.

**What changed: the SASS of ck's main loop, copied.**
- *Where ck gets its speed.* `qk_int_sv_i8_attn_kernel<128,128,16,128,128>` uses our layout: 8 warps × 16 rows, 128-key tiles, 255 registers. Its loop issues each 32-key chunk's exp2 and u8 conversion between the P·V IMMAs of the previous chunk, so the tensor pipe never waits for the whole softmax. Our config 2 computed every exp first, then ran P·V, then the fold.
- *fwd8 does the same.* P·V half 0 runs chunk-outer behind the exps. Half 0's fold sits between half 1's IMMAs.
- *Cheaper P conversion.* `cvt.pack.sat.u8.s32.b32` on `__float2int_rn` assembles to `F2IP.U8.F32` (2 values per instruction, same round-to-nearest). It replaces a magic FADD plus 0.75 PRMT per score.
- *Cheaper dequant.* Config 20's per-key `(scale, bias)` FFMA replaces config 2's IADD+FADD+FMUL.
- *Hoisted addresses.* All ldmatrix addresses are a per-tile base, 4 hoisted per-lane offsets and immediates.

**Loop size:** 809 SASS instructions per tile (128 IMMA), against ≈1100 for config 2 on the common path and 851 for ck. ncu at 8 heads and S = 36.3k (locked clocks):

| | ck | config 2 | config 80 |
|---|---:|---:|---:|
| time (ms) | 37.17 | 39.68 | **36.69** |
| instructions issued | 4.41·10⁹ | 5.40·10⁹ | **4.15·10⁹** |
| tensor pipe active | 69.0% | 66.5% | **70.9%** |

**What's left vs ck:** more `short_scoreboard` stalls (the 4-shuffle row-max chain) and more barrier stalls (one `__syncthreads` per tile).

**Variants tried and not kept:**

| config | change | result |
|---|---|---|
| 81, 83, 85 | half 1's fold deferred into the next tile's Q·Kᵀ | slight spill, 1–4% slower |
| 82, 84 | incremental load/read addressing (removes 3-stage spills) | 3–6% slower: ptxas schedules it worse, despite fewer instructions |
| 86–88 | k-step-outer Q·Kᵀ / two max chains per row | within ±0.5% |
| — | 3-input `max.f32` | assembles to 2 FMNMX on sm_121, no gain |
| — | packed `f32x2` FMA | assembles to 2 FFMA on sm_121, no gain |

**End to end** (ts_h2, first-frame shots er13 1344×768 and er43 768×1344, `fast_test.py` variants `ts_h2` (seed 7) vs opt-in `ts_h2_a80` (seed 8), `bench/results/attn_r4_e2e.jsonl`). Steady-state steps:

| | draft step | refine step |
|---|---:|---:|
| ts_h2 (ck) | 11.28–11.32 s | 32.63 / 32.05 s |
| ts_h2_a80 | 11.14–11.33 s, mean 11.21 | **31.52 / 31.63 s** |

- That is **≈ −1.4 s of sampling per clip, about 1%** (6 × 0.10 s + 0.75 s).
- Wall times (130.6–135.2 s) don't show it. The first step after switching between the two variants pays a one-off 5–9 s re-patch, and that swamps the difference.

**How to use it (opt-in):**
- Node `RapidH3Int8Attention` (config 80). It takes self-attention calls with D = 128, no mask, S ≥ 1024 and the `[1, H, S, D]` layout; every other call falls through to ck.
- `workflow.build(rapidh3_attn=80)` adds the node.
- `fast_test.py` variant `ts_h2_a80`.

**Verdict:** a real but small win (−2 to −3% per attention call, ≈1% per clip) with 8–20% lower attention error than ck. It is not bit-identical to ck, so a render-level check against stock comes before it becomes a default. That is a separate decision; no quality campaign was run. **Update 2026-10-26:** checked on eval1 and adopted as config 94, the default in ts_h2 / ts_s6 (config 190 since 2026-10-27, see "Attention, round 6").

### Video VAE, round 5 (2026-10-27): q/k rms+rope fused into the attention quantizer, level 4 (opt-in when built, default since 2026-10-27)
int8_convrot VAE, ComfyUI 0.39 + comfy-kitchen 0.2.37, 141-frame latents of the 6 screening shots (ts_h2 renders), decoded standalone with `bench/vae_bm.py decode --level N` (models loaded, 2 warm decodes after a cold one).

**Fresh profile of level 3** (`vae_bm.py decode … --level 3 --profile`, er41): 19.2 s warm, 18.8 s of GPU kernels over 288 decoder-layer calls (36 layers × 8 temporal chunks, 28 tiles = M 50,316 rows per call):

| kernel | GPU s | per call | note |
|---|---:|---:|---|
| INT8 GEMMs (w1+SwiGLU 5.35, w2 2.66, qkv 2.07, to_out 0.86) | 10.94 | 38.0 ms | 142–183 TFLOPS |
| ck attention main kernel (`qk_int_sv_i8_attn_kernel<128,64,32,64,64>`) | 1.77 | 6.1 ms | ≈120 TFLOPS |
| w2 quantizer (ck ConvRot, K 8192) | 1.54 | 5.4 ms | ≈230 GB/s, at bandwidth |
| ck `rms_rope_split_half_` (in place on q, k) | 1.37 | 4.8 ms | **removed in level 4** |
| ck attention Q/K quantizer (`quant_qk_fused` + `detect_k_anchor`) | 0.90 | 3.1 ms | **replaced in level 4** |
| qkv / w1 rms_norm quantizers (ck) | 0.76 | 1.3 ms | at bandwidth |
| ck V quantizer | 0.72 | 2.5 ms | ≈205 GB/s |
| to_out quantizer (ours, `vae_out_quant`) | 0.41 | 1.4 ms | at bandwidth |

**Built: `RAPIDH3_VAE_FAST=4`** (`vae_fast.install_ropeq`, `csrc/vae_ops.cu` `vae_qk_rope_quant`). Level 3 ran the rope kernel over q and k in place inside the qkv GEMM output, then `int8_attention` read them back to quantize. Level 4 does it in one pass:
- One kernel per 32 Q rows or 64 K rows of a head: 16-byte loads of the raw q (or k) rows and their rope tables into shared memory, ck's rms + split-half rope per row (one warp per row, ck's exact per-lane arithmetic, fp16 results), then ck's per-thread-group Q/K quantizer (H64 ConvRot, anchor-centred K) on those rows. The rope'd q/k never go to memory.
- The K-anchor detector ropes its 9 sampled keys itself and keeps the chosen anchor key for the K quantizer.
- ck's own V quantizer and attention kernel run on the result (`_C._quant_v_int8`, `_C.sage_sdpa_prequantized`, the same launches `sage_sdpa` makes).
- The copied per-element code is comfy-kitchen's (`ops/rms_rope.cu`, `sage_attention/quant_qk_int8.cu`, Apache-2.0), built with the same `--use_fast_math`.

**Exactness: bit-identical to level 3.**
- `bench/vae_r5_test.py`: int8 Q/K codes, their scales, the anchor index of every head and the attention output equal ck's on 10 shapes (B 2–28, S 133–1797, with and without anchor centring, both Q/K rotation paths).
- Full decodes of all 6 screening latents: `torch.equal` with level 3 on every frame (`vae_bm.py cmp`: identical, max abs 0).

**Speed.**
- Per decoder-layer call (M 50,316): rope + `int8_attention` 16.2–17.0 ms → 12.5 ms; the fused rope + Q/K quantizer is 3.9 ms against 4.8 + 3.1 ms. In the profile: 1.13 s less GPU time per decode.
- Decode at 141 frames, warm median of 2 per shot:

| shot | er11 | er41 | er43 | ef31 | ef41 | ex51 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| level 3 | 19.23 | 19.53 | 19.51 | 19.65 | 19.41 | 19.55 | **19.48 s** |
| level 4 | 18.35 | 18.43 | 18.47 | 18.53 | 18.58 | 18.43 | **18.47 s** |

  That is **−1.0 s per decode (−5.2%)**, with no memory change (peak 5.5 GiB either way).
- End to end (ts_h2 through `harness/fast_test.py`, the 6 shots after one warm-up render per level, models loaded):

| | level 3 | level 4 |
|---|---:|---:|
| decode (`RapidH3DecodeSaveVideo`, synced) | 19.57–19.71 s, mean 19.65 | 18.59–18.74 s, mean **18.67** |
| after the last step (decode + write tail) | 21.1–21.3 s | 20.2–20.5 s |
| clip wall time, median of 6 | 118.6 s | **117.6 s** |

  er11's level-4 render had a slower sampling pass (100.0 s against 95.5–96.9 s for every other render), which has nothing to do with the decode; its decode was 18.68 s. The mp4 video and audio packets are identical between the two levels on all 6 shots (`ffmpeg -f framemd5`).

**Looked at and not taken:**
- **(a) One norm + rotate + quantize kernel per linear input:** the decoder already has it. Its quantizers are comfy-kitchen's fused ConvRot kernels with the RMSNorm folded in (act `rms_norm`); there is no modulation, and the residual scale is in the GEMM epilogue. All of them run at 205–235 GB/s, which is GB10's memory bandwidth, so only removing a pass could help. The two big remaining passes can't be removed without changing the numerics. The w2 quantizer needs the row absmax of 8,192 rotated values before it can write any int8 code, and the to_out quantizer needs all 32 heads of a row; no GEMM or attention CTA holds a full row.
- **(c) A faster D = 64 attention:** comfy-kitchen's kernel compiled from its source with other query tilings (CTA_Q 64 with WARP_Q 32 or 16) is bit-identical to its CTA_Q 128 launch but no faster: 5.97–6.31 ms against 6.09–6.22 ms per call. A new kernel in the style of our D = 128 config 94 would not be bit-identical, and at D = 64 the softmax work per IMMA doubles. The whole attention kernel is 1.8 s per decode, so even 20% would be ≈0.35 s. Not attempted; the vendored build was dropped.
- **GEMM tiles at the batched shape** (M 50,316, `bench/vae_gemm_m50k.py`): the best tile per shape is within 1–2% of the current pick (to_out 2.93 vs 2.98 ms, w2 9.02 vs 9.12 ms), ≈0.05 s per decode. Not changed. The GEMMs run at 176–183 TFLOPS, the ≈200 ceiling of this kernel style (rounds 2–3).
- **V quantizer** (2.5 ms per call, ≈205 GB/s): a better-coalesced rewrite or folding it into the anchor/QK kernels would save ≈0.1 s per decode. Not done.

**comfy-kitchen versions.** Level 4 copies or calls comfy-kitchen code: the rms+rope kernel, the SageAttention Q/K quantizer, the K-anchor detector, `_quant_v_int8` and `sage_sdpa_prequantized`. It therefore runs only on the versions where that code was checked: **0.2.37** (ComfyUI 0.39 image) and **0.2.39** (the ComfyUI 0.42.1 image that `serve.sh` runs). On any other version it logs a fallback and takes the level-3 path.
- Between 0.2.37 and 0.2.39, `rms_rope.cu`, `rope_device.cuh`, `sage_attention.py` and the bindings of these calls are unchanged. `quant_qk_int8.cu` and `quant_v_int8.cu` only gain paged-decode kernels: `process_q` / `process_k` / `detect_k_anchor` are untouched, and the non-paged V kernel compiles to the same code. The launcher changes touch only D = 128 tilings on Ampere.
- Checked on 0.42.1 / 0.2.39: `bench/vae_r5_test.py` gives the same bits on all 10 shapes, and the full decodes of the 6 screening latents are `torch.equal` to level 3. Level 3 decodes in 19.16–19.67 s (mean 19.52), level 4 in 18.37–18.54 s (mean 18.48). In the pipeline (ts_h2 er41 rendered twice per level; ComfyUI's cache reused the sampling on the second render, so its wall time is the decode + save), level 3 decodes in 19.32 s and level 4 in 18.59 s, and the render takes 21.2 s against 20.1 s. The mp4 video and audio packets are identical between the levels (`ffmpeg -f framemd5`), and also identical to the ComfyUI 0.39 renders of er41.

**Status:** the `serve.sh` default since 2026-10-27 (bit-identical to level 3 on both verified comfy-kitchen versions, re-checked after the switch); `RAPIDH3_VAE_FAST=3 ./serve.sh restart` runs level 3; levels 0–3 are unchanged.

### Attention, round 6 (2026-10-27): config 190, bit-identical to 194 and 1–3% faster; two other softmax variants do not pay
Paired half-precision exps and a lazy softmax rescale were tried as softmax variants of config 94's kernel (`attn_cake<7, 2>` with a new `SMX` switch; details and numbers in `custom_nodes/rapid_h3/csrc/attn_cake/DESIGN.md` section 7). Real q/k/v dumped from an er13 render (S = 22,332 and 44,992, 3 dumps each, 56 heads), `bench/attn_cake.py`, `bench/results/attn_softmax_r6.json`:

| config | what | ms/call vs 94 (22k / 45k / 78.6k synth.) | error vs fp32 (real) |
|---|---|---|---|
| 94 (194) | default until 2026-10-27 | 93.9 / 363.1 / 1087.6 ms | 0.80–1.02% |
| **90 (190)** | mutex entered after the row max | **−1.8% / −2.4% / −2.2%** | bit-identical |
| 89 (189) | lazy rescale + magic bias kept in O | −0.1% / −0.6% / −0.7% | same (4 digits) |
| 79 (179) | `ex2.approx.f16x2` exps | +2.4% / +2.1% / +1.8% | +0.001–0.003 pt |

- **Row 52 does not exist on this GPU.** sm_121 has no paired MUFU.EX2: ptxas assembles `ex2.approx.f16x2` to two `MUFU.EX2.F16` plus a PRMT (also on sm_90a / 100a / 120a; bf16x2 likewise). With the f32→f16x2 packing and the magic-add u8 conversion it is 15 instructions per 4 scores instead of 10.
- **Row 53 is moot in our per-tile-max scheme.** Every tile needs its own `ts` multiply, and the rescale `a` already rides in the FFMA that removes the P·V accumulator's 1.5·2²³ bias, so a lazy max alone saves nothing. Keeping that bias in O as well (1 FFMA fold, bias removed every 8 tiles or when a row's max moves by > 8 in exp2 units) removes ≈56 of ≈680 non-IMMA instructions per tile, but those FFMAs sit between half 1's IMMAs, off the softmax critical path: ≤0.7% faster. Shadow (ts_h2, 3 shots): 1.84% video / 4.77% audio vs 194's 1.88% / 4.76%.
- **Config 90 is the win.** Config 94's mutex section began with the row-max shuffles; 90 enters it after the shuffles and the per-row scalars, so each warp group runs its latency-bound shuffle → MUFU chain while the other holds the section. Same operations in the same order: `torch.equal` to 94 on all tested shapes, the fused rms/rope path and the text-bias path (`bench/attn_softmax_r6.py`). Ending the section earlier, starting P before it, or adding config 89's fold to it were all slower.
- **End to end:** on the 6 screening shots, ts_h2 / ts_s6 with 190 give identical latents and mp4 frames to 194. Steady-state steps: ts_h2 draft 10.52 → 10.44 s, refine 29.30 → 28.82 s (≈ −0.9 s per clip); ts_s6 29.74 → 29.30 s (≈ −2.6 s per clip). Contiguous blocks on 3 more shots: ts_h2 123.4 → 120.3 s, ts_s6 ≈206 → ≈204.5 s.
- **Status:** the default in `ts_h2` / `ts_s6` since 2026-10-27 (bit-identical to 194, so no quality check was needed; the preset graphs differ only in the attention node's `config`); `rapidh3_attn=194` runs the previous kernel. 189 and 179 stay opt-in for reference.

### Next levers (single Spark)
1. **Attention kernel**: tried, see "Custom INT8 attention kernel" above. It was as fast as comfy-kitchen at best and 20–25% more accurate, so it was not wired in then (2026-10-13; see round 4, config 94 and config 190 below).
   - **Round 3 (2026-10-24): tried, no gain**, see "Lossless kernels, round 3" above.
   - **Round 4 (2026-10-24): done, small win, opt-in.** Config 80 is 2–2.7% faster than ck per call and more accurate, ≈−1.4 s per ts_h2 clip. Not a default then. See "Lossless kernels, round 4" above.
   - **Config 94 (2026-10-26): default in ts_h2 / ts_s6 (as 194) until 2026-10-27.** Config 80's math on a persistent schedule, bit-identical to it, −3.9–4.0% per call vs config 80 at 5 s (−6.7% vs ck). See `custom_nodes/rapid_h3/csrc/attn_cake/DESIGN.md`.
   - **Config 90 (2026-10-27): the default in ts_h2 / ts_s6, as 190.** Bit-identical to 94, −1.8–2.4% per call. See "Attention, round 6" above.
   - *Was:* **TODO, time-boxed to about 1 day:** close the gap inside the multiply. With the softmax removed, our loop reaches about 190 of 242 TFLOPS. ncu shows no spills; the stalls are tensor-pipe cadence plus about 14% `ldmatrix` queue pressure. Hoist the smem address math out of the inner loop, cut ldmatrix issue (SageAttention-style instruction buffer), then compare register and instruction counts against comfy-kitchen's SASS. Ceiling is about −1.5 s/step at 5 s and about −5.5 s/step at 10 s, with no quality risk.
2. **Resident serving**: done. There was no real slowdown, and `--cache-ram` is the default. Shipping our own INT8 checkpoint in Comfy's format (with any LoRA baked in) would remove the remaining first-step init (about 1 s once loaded).
3. **Fusions, GEMM tiles, VAE, SwiGLU epilogue: done 2026-10-13.** Details are in "Speed work, round 2" above. One optional item is left: fusing rms+rope into the q/k quantizer, worth ≈0.45 s/step at 5 s and ≈0.9 s/step at 10 s. *Was:* it would have meant a rewritten copy of comfy-kitchen's quantizer, tying us to ck 0.2.37 internals (its quantizer, K-anchor detection and V quantizer). Once config 94 was the default we owned the attention pre-pass, so that objection no longer applied. **Done 2026-10-26** as config 194 (rms+rope fused into our pre-pass, bit-identical; `custom_nodes/rapid_h3/csrc/attn_cake/DESIGN.md` §6), carried into config 190.
4. **What's left for lossless speed (researched 2026-10-13):** little. The step is tensor-core bound at 67–80% of peak and its memory-bound parts run at bandwidth.
   - **GPU clocks:** under load the chip holds ≈2.33 GHz. Locking it at the 2418 MHz applications clock made it *slower* (233 vs 248 TFLOPS sustained); there is no settable power limit on GB10. Dead end.
   - **Our quantizers:** 215–338 GB/s moved against ≈220 GB/s for a plain device copy (`bench/quant_bw.py`), so they are memory-bound. Fewer passes would save ≈0.3 s/step at most.
   - **INT8 GEMM:** 200 of 252 TFLOPS. A hand-written kernel might reach ≈230, worth ≈1 s/step (4%). The best FP8 CUTLASS number others report on GB10 is 188. *Round 3:* a persistent hand-written kernel's main loop alone reaches only 193–214, and with its epilogue it is slower than CUTLASS. See "Lossless kernels, round 3".
   - **Attention:** 169 of 252 TFLOPS, limited by the QK→softmax→PV chain at 2 warps per sub-partition (measured above). No lossless lever left on mma.sync hardware. *(2026-10-13; superseded: config 94's main kernel reaches 170–175 TFLOPS against ck's 162–164 on the same benchmark; it was the default from 2026-10-26, and config 190, 1.8–2.4% faster per call, is since 2026-10-27.)*
   - **VAE decode:** ≈4–5 s per render from removing the `nan_to_num` guard and two layout copies and using our GEMM/quantizers inside the VAE. ComfyUI's September VAE speed-up is already in 0.39; its fp16-accumulation part only touches the encoder.
   - **Community levers** (step caching, distillation, forecasting, draft+upscale) all change the output.
   - **Sol-Attn sparse attention: measured 2026-10-13, lossy.** We tested ComfyUI's `BlockSparseAttention` node (sol-attn, τ = 1.3, the node's defaults: dense for the first 20% of the schedule and for the conditioning rows) on top of our recipe:

     | | Recipe | + sol-attn |
     |---|---|---|
     | 5 s 768p, s/step (steps 3–8) | 24.0 | **17.5–18.0** |
     | 5 s 768p, render (models loaded) | ≈216 s | **186 s** |
     | 10 s 768p, s/step | 70.5 | **39.3** |
     | Shadow error once sparse is on, 768p (1 prompt) | 1.3–2.5% per step | **4.8–14.0%** |
     | Shadow error, 480p (3 prompts), mean video / audio | 3.60% / 2.45% | **11.56% / 4.88%** (max 30.6%) |

     It's fast, but per-step divergence from bf16 is 3–6× our recipe's, far above the 1-ulp noise floor. So it is not lossless, whatever one guide claims. It could serve as an opt-in fast mode after a visual review (`harness/workflow.py sparse=sol-attn`). *(Closed: sparse attention was rejected, including Sol-Attn on the refine step only.)* `RapidH3Shadow` now strips `patches_replace` from its bf16 reference so block-patching nodes can be measured.
   - **NVFP4 GEMMs:** others measured 356 TFLOPS dense on GB10 (≈1.8× our INT8), worth ≈12% of a step, but 4-bit activations are lossy. Out of scope (rejected).
   - **Throughput:** two concurrent requests could overlap one clip's VAE decode with the next clip's sampling (≈5% more clips per hour); per-clip latency unchanged.
5. **Perceptual sign-off**: blind A/B on ~20 prompts at 768p / 10 s, plus audio-sync checks. *(The presets were later scored on the eval1 / motion1 shot sets.)*


---

## 0. Starting point: a DGX Spark, 2026-10-12

### Starting point
- **Software:** stock ComfyUI with the Comfy-Org pruned bf16 DiT (the un-pruned 66 GB bf16 DiT is only needed to check that AdaLN pruning is exact).
- **Docker image:** the ComfyUI 0.39 base image (ComfyUI 0.39, torch 2.10 cu130, comfy-kitchen 0.2.37, Triton 3.5). *(No longer used: the release runs `rapid-h3/h3-comfyui:v0.42.1` built from `docker/`: ComfyUI 0.42.1, comfy-kitchen 0.2.39, NGC torch 2.10 nv25.11.)*
- **Memory:** running out of unified memory on GB10 can freeze the machine; keep an eye on free memory (the video decoder sizes its tile batches to the free memory).

### Where a bf16 step goes (prior profile, 1344×768, 124 frames ≈ 40k tokens)

| Class | ms/step | Share | Kernel |
|---|---:|---:|---|
| Attention | 23,839 | 55% | `pytorch_flash` sm80 FA2 kernel, ~458 ms × 52 calls |
| Linear GEMMs | 15,800 | 37% | cuBLAS `nvjet_sm121` bf16, ~79 ms per call |
| Modulation / elementwise | 2,735 | 6% | unfused PyTorch elementwise ops (2,400 kernels per step) |
| RoPE + norm | 850 | 2% | |

### Our microbenchmarks at H3 shapes (`bench/microbench.py`, M = 40k tokens)

Results are in `bench/results/microbench_40k.json`.

| Op | bf16 | FP8 e4m3 (tensorwise) | INT8 | Notes |
|---|---:|---:|---:|---|
| qkv GEMM 5376→16128 | 67.7 ms (102 TF) | **35.4 ms (196 TF)** | 41.5 ms (167 TF) | |
| out_proj 5376→5376 | 23.2 ms (100 TF) | **11.5 ms (201 TF)** | 14.0 ms (165 TF) | |
| fc1 5376→28672 | 118.3 ms (104 TF) | **62.2 ms (198 TF)** | 73.8 ms (167 TF) | |
| fc2 14336→5376 | 65.7 ms (94 TF) | **31.9 ms (193 TF)** | 38.4 ms (161 TF) | |
| Attention S=40k, 56×128, non-causal | flash 502 ms (91 TF), cuDNN 498 ms, mem-efficient 1,285 ms | — (no kernel yet) | comfy-kitchen int8 **311 ms (148 TF)**, cos 0.99986 | |

**What this settles:**
1. **GB10 dense rates are about 100 TFLOPS for bf16, about 200 for FP8 and about 165 for INT8**, on real H3 shapes. So FP8 is the fastest format: 2× bf16 and 1.2× INT8.
2. **PyTorch row-wise FP8 `_scaled_mm` is broken on sm_121.** It hits "Arch conditional MMA instruction used without targeting appropriate compute capability" and takes 20–106 s per GEMM. Only *per-tensor*-scaled FP8 works. That explains why the stock `fp8_scaled` checkpoint uses a single scalar scale and why its quality was worse: the FP8 variant was not adopted, with 5.7% video divergence against 2.8% for INT8.
3. **INT8 GEMM itself is fine at 165 TF.** The 108 ms per GEMM seen in the ComfyUI int8_convrot path comes from comfy's CUTLASS kernel with a per-row-scale epilogue, not from the hardware.
4. **bf16 attention can't be beaten by switching backend:** cuDNN equals flash. Only lower precision helps. INT8 is 1.6×, and an FP8 attention kernel should be about 2× (not yet built).

### Step-time model: 1344×768, 5 s, per denoising step
These are our estimates from the measured kernel times above.

| Configuration | Attention | GEMM | Other | **Step** | vs bf16 |
|---|---:|---:|---:|---:|---:|
| bf16 (measured in profile) | 23.8 | 15.8 | 3.6 | **43.2 s** | 1.00× |
| INT8 attention + fused modulation (measured) | ~16 | ~15.8 | ~1.5 | **32.9 s** | 1.31× |
| + FP8 GEMM with per-token × per-channel scales (**needs a kernel**) | 16.2 | 7.1 | 1.5 | ≈ 25 s | ≈ 1.7× |
| + FP8 attention (**needs a kernel**) | ≈ 12 | 7.1 | 1.0 | ≈ 20 s | ≈ 2.1× |

Those two "needs a kernel" rows were the plan on 2026-10-12. What was built instead: INT8 W8A8 with ConvRot rotation on our own CUTLASS sm_121 kernel (as fast as FP8 row-wise and more accurate), and our own INT8 attention (no FP8 attention kernel). See the table at the top.

---
