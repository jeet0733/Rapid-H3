# Credits

Rapid-H3 stands on other people's work. Licenses of the code included in this repository are in [`NOTICE`](NOTICE).

## Model

- **[MiniMax](https://huggingface.co/MiniMaxAI)**: MiniMax-H3, the text / image → video + audio model these presets run
  (MiniMax H3 Community License), and its Ref2VA reference-to-video checkpoint.
- **[Comfy-Org](https://huggingface.co/Comfy-Org/MiniMax-H3)**: the ComfyUI packaging of H3 (pruned bf16 and
  `int8_convrot` DiTs, the int8 video VAE, the Qwen3-VL text encoder) and the ConvRot rotation our INT8 linears reuse.
- **[LBH-123-AI](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)**: the learned ×2 H3 latent
  upscaler node and weights that `ts_h2` uses between its draft and refine (node MIT).

## Software

- **[ComfyUI](https://github.com/Comfy-Org/ComfyUI)** (GPL-3.0): the server, the MiniMax H3 nodes and the sampling
  stack the presets are built from.
- **[comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen)** by Comfy Org and NVIDIA (Apache-2.0): INT8 attention,
  quantizers and kernels; parts are copied into our video-decoder kernels (`custom_nodes/rapid_h3/csrc/vae_ops.cu`, see
  `NOTICE`).
- **[NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass)** (BSD-3-Clause): the INT8 GEMM templates and epilogue visitors
  our sm_121 kernels are built on.
- **[PyTorch](https://github.com/pytorch/pytorch)**, **[torchaudio](https://github.com/pytorch/audio)** and the
  **NVIDIA PyTorch container**: the GB10 (sm_121) build of the stack; **[Triton](https://github.com/triton-lang/triton)**:
  the fused quantizer kernels.

## Inspiration

- **[Mia's AI Lab](https://mia-ai.net)** ([MiaAI-Lab on GitHub](https://github.com/MiaAI-Lab)): the motivation for
  this project. Her DGX Spark recipes showed how much speed careful, verified engineering can get out of a single
  Spark and made me want to do the same for video.

## Ideas we built on

- **[SageAttention](https://github.com/thu-ml/SageAttention)**: INT8 Q·Kᵀ with smoothed K and 8-bit P, the numerics our
  attention kernel follows (via comfy-kitchen).
- **FlashInfer CAKE** ([flashinfer-ai/flashinfer#5539](https://github.com/flashinfer-ai/flashinfer/pull/5539)): the
  persistent, ping-pong attention schedule our config 94 / 190 kernel adapts to GB10.
- **ConvRot / Hadamard-rotated INT8** (Comfy-Org's `int8_convrot` release): the weight rotation of our INT8 linears.
- **[TensorFold](https://github.com/ashhart/TensorFold)**: the verification discipline (every change gated against a
  reference and a measured noise floor) the project started from.

## Evaluation tools

Quality numbers in the README come from these open models and tools, run in a separate Python environment:
InsightFace (ArcFace `buffalo_l`), DINOv2, SyncNet (`joonson/syncnet_python`) with S3FD, SpeechBrain ECAPA-TDNN,
faster-whisper (Whisper large-v3-turbo), RAFT optical flow, CLIP ViT-L/14, pyiqa (MUSIQ, CLIP-IQA), DOVER, Demucs (htdemucs vocal stems), DNSMOS
(microsoft/DNS-Challenge) and UTMOS22 (tarepan/SpeechMOS).
