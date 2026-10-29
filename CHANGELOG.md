# Changelog

Times are warm per-clip times for a 768×1344, 141-frame (≈5.9 s) first-frame clip with audio on one DGX Spark.
Stock ComfyUI bf16 at 10 steps: 590–630 s. Kernel details: [`docs/kernels.md`](docs/kernels.md).

## 2026-10-27

- **Attention config 190** is the default in `ts_h2` and `ts_s6`: the two warp groups' softmax sections overlap.
  Bit-identical to config 194, 1.8–2.4 % faster per attention call (≈−1 s `ts_h2`, ≈−2.6 s `ts_s6`). `rapidh3_attn=194`
  runs the previous kernel.
- **Video decoder level 4** (`RAPIDH3_VAE_FAST=4`) is the `serve.sh` default: the decoder attention's q/k RMSNorm + RoPE
  fused into its Q/K quantizer. Bit-identical to level 3, −1.0 s per clip.
- **Docker image `rapid-h3/h3-comfyui:v0.42.1`** built from our own recipe (`docker/`): NGC PyTorch base, ComfyUI 0.42.1,
  comfy-kitchen 0.2.39, torchaudio from source, versions held by `docker/constraints.txt`. Renders bit-identical to the
  ComfyUI 0.39 image at the same speed.
- **Ref2VA** on the current defaults: `encode_once` now works with reference images; usage rules (full-frame
  references → `ts_s6` with `steps=10`; `ts_h2` only with cropped references).
- Documentation consolidated into `README.md` and `docs/`; `LICENSE`, `NOTICE`, `CREDITS.md`.
- Tested and not adopted: paired half-precision softmax exponentials (slower on sm_121), lazy softmax rescale (<1 %),
  Jenga text-attention bias for smaller drafts (still crops in), smaller drafts (0.6× / 0.5×).
- `ts_h2` ≈117 s (6-shot median 116.5 s), `ts_s6` ≈201 s (median 200.7 s); `runs/2026-10-27/release_timing/`.

## 2026-10-26

- `ts_h2` / `ts_s6` defaults: conditioning encoded once (`encode_once="draft"`), our INT8 attention kernel (config 94,
  then 194 with the q/k RMSNorm + RoPE fused into its pre-pass), the Qwen3-VL text encoder on our INT8 path
  (`te_quant="int8_rot"`), video decoder level 3 (all tiles of a chunk per call, SwiGLU in the GEMM epilogue).
  Kept after side-by-side review. The previous settings are the `ts_h2_ck` / `ts_s6_ck` presets.
- `ts_h2` 131.6 → ≈119 s, `ts_s6` ≈215 → ≈205 s.
- ComfyUI 0.42.1 support (`vae_fast.py`), validated bit-identical to 0.39.
- Tested and not adopted: GB10 clock tuning, an external fused VAE decoder, Triton INT8 GEMM, sparse attention on the
  refine, 4–5-step `ts_s6`, high-low-high sampling, distilled 4-step LoRAs (DMAD, PDMD), a 0.8× draft with a two-step
  refine.

## 2026-10-25

- Optional inputs for both presets: `last_frame`, fixed voice / music track (`audio_input`, `audio_pad="floor"`),
  frame and region pins (`pin_latent`, `pin_region`), mid-clip anchors (`anchors`), Ref2VA reference images and voices
  (`ref_images`, `ref_audios`, `ref_videos`).

## 2026-10-24

- **`ts_s6`**: one stage at full size, 6 steps; stock level on every scored criterion, for camera moves and close-up
  faces (≈217 s then).
- **`ts_h2`** becomes the default: 6-step draft at 0.7× size, learned ×2 latent upscale, one refine step from σ 0.72,
  mp4 written during the decode (≈131 s then). `ts_h` (two refine steps) kept as a preset for rollback.
- Quality evaluation harness (`harness/eval_clips.py`, `harness/audio_quality.py`) and the 15-shot `eval1` set.

## 2026-10-12 – 2026-10-23

- The Rapid-H3 kernel recipe: INT8 W8A8 linears with the ConvRot rotation on our CUTLASS sm_121 kernel, fused
  norm/modulation/quantize and SwiGLU kernels, INT8 attention, the int8 video VAE, models kept loaded, synced step
  timing. 1.76× faster per step than stock bf16, with single-step error within ≈2× of bf16's own rounding noise.
