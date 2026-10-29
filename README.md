# Rapid-H3: Super fast H3 video + audio in about 2 minutes on one DGX Spark

Rapid-H3 makes MiniMax-H3 (text / image → video with stereo audio) render **3–5× faster on a single NVIDIA DGX Spark (GB10)** than stock ComfyUI, with no human-noticeable loss in picture, motion, voice or lip sync. It is a ComfyUI node pack plus two ready-made render presets:

- **`ts_h2`** (default): **≈117 s** for a 768×1344, 141-frame (≈5.9 s) first-frame clip with audio.
- **`ts_s6`** (camera moves, close-up faces): **≈201 s**, at stock level on every criterion we score.
- Stock ComfyUI bf16 at the 10-step reference: **590–630 s**.
- The first render is slower on a fresh install, while the CUDA kernels are built (once) and the models load.

Rapid-H3 is an independent community project. It is not affiliated with or endorsed by MiniMax, NVIDIA or Comfy Org; it works with the openly released MiniMax-H3 weights, which keep their own licence (see [Licence](#licence)).

It gets there in two layers:
1. **Faster steps** (the Rapid-H3 kernels, `docs/kernels.md`): INT8 matrix multiplies and attention with our own sm_121 kernels, fused quantizers, an INT8 text-encoder path and a faster video decoder. **1.9× per full-size step** (56.7 → 29.6 s) with per-step error within ≈2× of bf16's own rounding noise.
2. **Fewer, smaller steps** (the ts_h recipe): 6 steps instead of 10; for `ts_h2`, a draft at 0.7× size, a learned ×2 latent upscale and one full-size refine step.

Status: 2026-10-29. Measured on DGX Spark (GB10, 128 GB) with ComfyUI 0.39; the default image is now ComfyUI 0.42.1, validated bit-identical (same latents, frames and audio at the same speed). The build-up, every experiment and the rejected ideas are in `docs/kernels.md`.

## Results

All numbers: DGX Spark, 141 frames (times: 2026-10-27, `runs/2026-10-27/release_timing/`) (≈5.9 s), first frame + prompt, speech in every clip, models kept loaded. Same seed does not give the same picture as stock (any 8-bit change lands on a different, equally plausible clip), so quality is compared as statistics over shot sets.

| | Stock bf16, 10 steps | **`ts_h2`** (default) | **`ts_s6`** (motion, close-ups) |
|---|---|---|---|
| **Speed** (768×1344, warm server) | | | |
| Steps | 10 at full size | 6 at 0.7× size + 1 at full size | 6 at full size |
| s / step (draft / full size) | — / 56.7 | 10.5 / 29.1–29.5 | — / 29.4–29.9 |
| **Time per clip** | **590–630 s** | **≈117 s** (6 shots: 115.5–118.4 s, median 116.5 s) | **≈201 s** (6 shots: 199.8–201.9 s, median 200.7 s) |
| Speed-up | 1.0× | ≈5× | ≈3× |
| **Motion set** (16 clips: 8 shots × 2 seeds; camera push-ins, close-up faces) | | | |
| Camera push-ins completed | 16/16 | 13/16 | **16/16** |
| Motion (optical flow vs stock) | ×1.00 | ×0.88 | ×0.95 |
| Jerkiness (frame-to-frame acceleration, lower is smoother) | 0.48 | 0.63 | 0.50 |
| Identity (ArcFace vs reference face) | 0.794 | 0.802 | 0.814 |
| Lip sync (SyncNet LSE-C on speech, higher is better) | 9.77 | 9.53 | 9.97 |
| Close-up face sharpness | 644 | 566 (×0.88) | 613 (×0.95) |
| Flicker (shimmer on still pixels, 8-bit levels) | 0.437 | 0.456 | 0.467 |
| Overall video quality (DOVER) | 0.769 | 0.748 | 0.757 |
| Voice naturalness (UTMOS) | 3.94 | 3.84 | 3.93 |
| **Dialogue set `eval1`** (15 shots: 5 characters × 3 lines; ts_s6 not run on it) | | | |
| Identity (ArcFace vs reference face) | 0.776 | 0.767 | — |
| Identity drift (worst frame vs first) | 0.711 | 0.709 | — |
| Lip sync (LSE-C on speech) | 7.71 | 7.22 | — |
| Dialogue word error (vs the scripted line) | 0.6 % | 0.6 % | — |
| Reference fidelity (first-frame SSIM) | 0.970 | 0.968 | — |
| Prompt match (CLIP score) | 30.6 | 30.5 | — |
| Camera push-ins completed | 15/15 | 12/15 | — |
| Motion (optical flow vs stock) | ×1.00 | ×0.86 | — |
| Face sharpness | 555 | ×0.92 | — |
| Flicker | 0.349 | ×1.18 | — |
| Voice naturalness (UTMOS) | 3.85 | 3.72 | — |

How to read it: `ts_s6` is at stock level on every criterion. `ts_h2` matches stock on identity, dialogue, reference fidelity and prompt, and gives up some camera motion (push-ins mis-scaled on about 1 clip in 5), close-up face detail and a little lip-sync confidence on small faces. Flicker differences of this size (a tenth of one 8-bit level) were not visible in side-by-side review. The motion-set numbers were measured with the settings before 2026-10-26; the later speed-ups were checked to stay within run-to-run noise (encode once on the 15-shot set, INT8 attention on the screening shots, the INT8 text encoder on screening renders and a side-by-side review) or are bit-identical (decoder level 3 → 4, attention config 194 → 190, the Docker image). The evaluation shots' first-frame images are not in the repository; `harness/eval_clips.py` and `harness/audio_quality.py` re-run the scoring on your own clips.

The first clip after a server start takes longer: ≈0.5–1 min more to load the models, and ≈3 min more on a fresh install, where the CUDA kernels are built once (cached in `~/rapid-h3/build` afterwards). Clip length scales the time: 260 frames with `ts_h2` took 303.8 s; a 209-frame clip took 227 s (`ts_h2`) and 388 s (`ts_s6`) against 1,130 s stock.

**Which preset:** `ts_h2` for most shots (static or gently moving camera, medium and wide shots, dialogue). `ts_s6` for shots where a camera move or a close-up face matters (push-ins, dolly moves, beauty close-ups, lip-sync-critical dialogue), and with reference images (Ref2VA). Rule of thumb: camera move → `ts_s6`, static → `ts_h2`.

## Quick start

Requirements: one NVIDIA DGX Spark (GB10, 128 GB unified memory) with DGX OS and Docker, about 72 GB of disk for the model files (plus ≈40 GB for the optional Ref2VA checkpoint) and ≈21 GB for the image. Setting it up from scratch (Docker image, weights with sha256, kernels, evaluation environment): **`REBUILD.md`**.

Starting the server needs the Docker image (`REBUILD.md` section 2), `third_party/cutlass` (section 1) and the weights (section 3). By default `serve.sh` looks for the weights in `~/h3/models` (set `RAPIDH3_MODELS` otherwise) and for the int8 video VAE in `~/rapid-h3/models/vae/`.

```bash
# on the Spark, from ~/rapid-h3: ComfyUI with the Rapid-H3 node pack on 127.0.0.1:8189, models kept loaded
./serve.sh start                 # stop | restart | logs | status; RAPIDH3_MODELS=... if the weights live elsewhere
```

```python
import sys; sys.path.insert(0, "harness")
import render, workflow

# first frame + prompt -> video + audio, portrait 768x1344, 141 frames (≈5.9 s at 24 fps)
g = workflow.build(prompt, width=768, height=1344, length=141, first_frame="shot.png", seed=7,
                   prefix="myrun/shot01", **workflow.PRESETS["ts_h2"])
render.api("/prompt", {"prompt": g})          # mp4 in data/output/myrun/

# camera move or close-up: one stage at full size
g = workflow.build(prompt, width=1344, height=768, length=141, first_frame="shot.png", seed=7,
                   prefix="myrun/pushin01", **workflow.PRESETS["ts_s6"])

# text only: leave out first_frame
```

- `first_frame` is a file name in the server's input folder (`~/rapid-h3/data/input`, subfolders allowed); `prefix` is the output path under `~/rapid-h3/data/output`. The server runs as root inside its container, so the files it writes there are owned by root (remove them with `sudo`).
- **In the ComfyUI UI:** `workflows/ts_h2_portrait_api.json`, `ts_h2_landscape_api.json`, `ts_s6_portrait_api.json`, `ts_s6_landscape_api.json` (API format, 141 frames, first frame `first_frame.png`, seed 7). Open `http://127.0.0.1:8189` on the Spark (from another machine, tunnel first: `ssh -L 8189:127.0.0.1:8189 <spark>`), then drag one onto the page or POST it to `/prompt`; edit the prompt, image, length and seed nodes. Regenerate them after changing a preset (`workflow.build(...)` + `json.dump`).
- **The first render is slow:** on a fresh install it builds the CUDA kernels and loads the models (≈5 min for a `ts_h2` clip instead of ≈2); after that the kernels are cached, and with the server left running the models stay loaded.
- Every graph includes `RapidH3StepTimer`; synced per-step times land in `data/output/rapidh3_steps.jsonl` (ComfyUI's progress bar reads ≈12% low).
- ComfyUI caches identical nodes: two submissions with the same prompt, seed and draft reuse the draft. Change the seed when timing variants.
- Batch testing: `python3 harness/fast_test.py SHOTS.json --variants ts_h2 ts_s6 --name NAME`.

## How it works

### 1. Faster steps: the Rapid-H3 kernels

| Part | Stock | Rapid-H3 | Gain |
|---|---|---|---|
| 200 block linears (qkv, out_proj, fc1, fc2) | bf16 cuBLAS, ≈100 TFLOPS | INT8 W8A8 with the ConvRot Hadamard rotation (the same rotation as Comfy-Org's `int8_convrot` release), our CUTLASS sm_121 kernel at ≈200 TFLOPS; quantizers, SwiGLU and gated residual fused into it | the bulk of the 1.9× |
| Norm → modulation → quantize, SwiGLU → quantize | separate bf16 passes | one Triton kernel each | part of the above |
| Attention | bf16 PyTorch flash, ≈90 TFLOPS | our INT8 attention, **config 190** (`RapidH3Int8Attention`, `custom_nodes/rapid_h3/csrc/attn_cake/`): persistent schedule, the two warp groups' softmax sections overlapped, q/k RMSNorm + RoPE fused into its pre-pass; 0.81–0.98% error vs fp32 (comfy-kitchen INT8: 0.90–1.23%) | ≈170–175 TFLOPS |
| Text encoder (Qwen3-VL-32B) | fp32 path | INT8 W8A8 with a residual pass on outlier rows (`te_quant="int8_rot"`), conditioning cos ≥ 0.9991 | −1.5 s / −2.4 s per clip |
| Video VAE decode | fp16, ≈37 s | int8_convrot VAE plus bit-identical trims, all tiles of a chunk per call, SwiGLU in the GEMM epilogue, q/k rms+rope fused into the attention quantizer (`RAPIDH3_VAE_FAST=4`) | ≈18.5 s at 141 frames |
| Weights | — | quantized at load from the stock pruned bf16 DiT (≈4 s, after any LoRA merge), or Comfy-Org's int8_convrot file adopted as is | |

Accuracy gate: every step is checked by **single-step divergence** (`RapidH3Shadow`): each denoising step runs twice on identical inputs, candidate vs bf16, against a noise floor of bf16 evaluated on inputs jittered by one bf16 ulp. The recipe sits at 1.7× (video) / 2.1× (audio) that floor. Kernel changes made since the first recipe are bit-identical to what they replace, or checked on the eval set and in side-by-side review. Details, measurements and what did not work: `docs/kernels.md`.

### 2. Fewer, smaller steps: the ts_h recipe

| Part | What it does | Speed | Quality |
|---|---|---|---|
| 6 steps instead of 10 (`res_multistep`, `simple`, shift 12 / 3) | | 1.7× fewer steps | `ts_s6`: stock level on every scored criterion |
| **Draft at 0.7× size** (`ts_h2`) | the whole 6-step schedule at ≈half the tokens; this fixes composition, motion and audio | 10.5 s per step instead of 29.6 s | framing, people and identity match the first frame. Smaller drafts (0.6×, 0.5×) crop in to close-ups and are rejected |
| **×2 latent upscale** | `MinimaxH3LatentUpscaler3D` (community node, MIT) maps the draft's video latent to full size without a decode | ≈3 s | none visible |
| **1 refine step at full size from σ 0.72** | re-renders skin, hair and texture at full resolution, low enough that the layout stays | 29.3 s | identity at stock level; lip sync −0.5 LSE-C, flicker ×1.18 |
| **Conditioning encoded once** (`encode_once="draft"`) | the refine reuses the draft's text encode and re-encodes only the first frame at full size | −5.5 s | audio bit-identical; refine latent moves 1.5–2.0% (int8 vs bf16: 3–8%) |
| **Audio kept from the draft** | `LTXVFreezeLatent` on the audio latent, so the refine cannot change it | | lip sync checked on eval1 |
| **mp4 written during the decode** | `RapidH3DecodeSaveVideo` encodes each decoded chunk while the next decodes | −2 s | bit-identical mp4 |

## Optional inputs

All off by default; the preset graphs are unchanged unless you pass them (`workflow.build` keyword arguments; they combine).

| feature | option | `ts_s6` | `ts_h2` | cost per step |
|---|---|---|---|---|
| end on a given frame | `last_frame` | ✓ exact (32.5–38.8 dB) | ✓ content only | +7–8.5% |
| fixed voice / music track | `audio_input` + `audio_pad="floor"` | ✓ | ✓ | ±0–2% |
| hold frames of an earlier render | `pin_latent`, `pin_frames`, `pin_audio` | ✓ (≈42 dB) | ✗ (raises) | +5% |
| inpainting (regenerate a masked area) | `pin_region` | ✓ (no visible seam) | ✗ (raises) | +5% |
| mid-clip anchor: image / clip / audio | `anchors=[...]` | ✓ (add `last_frame` on camera moves) | ⚠ content only, can pop before the anchor | image +2–4%, clip +26%, audio +1% |
| character look and voice (Ref2VA) | `ref_images`, `ref_audios`, `ref_videos` | ✓ (`steps=10` with full-frame references) | ⚠ only with cropped references | ≈+3% per image; a reference video ≈2.7× |

```python
# end on a given frame
g = workflow.build(prompt, ..., first_frame="a.png", last_frame="b.png", **workflow.PRESETS["ts_s6"])

# keep a given voice/music track (top level of data/input; starts at pixel frame 12 = 0.5 s);
# audio_pad="floor" holds the whole clip's audio, so no extra lines are invented around it
g = workflow.build(prompt, ..., audio_input="line01.wav", audio_start=12, audio_pad="floor", **workflow.PRESETS["ts_h2"])

# hold frames 0-21 of an earlier render; or regenerate only the white part of a mask
g = workflow.build(prompt, ..., pin_latent="run/shot_00001_.latent", pin_frames="0-21", pin_audio="same", **workflow.PRESETS["ts_s6"])
g = workflow.build(prompt, ..., pin_latent="run/shot_00001_.latent", pin_frames="0-140", pin_audio="same",
                   pin_region="masks/right_half.png", **workflow.PRESETS["ts_s6"])

# anchors at any frame: an image, a 5/22/39-frame clip, or audio placed from that frame
g = workflow.build(prompt, ..., last_frame="end.png",
                   anchors=[{"image": "mid.png", "frame": 70}, {"video": "clip.mp4", "frame": 90},
                            {"audio": "line.wav", "audio_s": (2.0, 2.0), "frame": 48}],
                   **workflow.PRESETS["ts_s6"])
```

Notes from the measurements (4 eval1 shots each):
- `audio_input`: the track is kept (correlation 0.97–0.98, ≤12 ms), lips in sync, but SyncNet confidence is 1.3–2.0 lower than with H3's own voice and motion 15–45% lower. Mix any ambience into the track first: nothing else is generated with `audio_pad="floor"`.
- `anchors` image: `ts_s6` hits the anchored frame (34–38 dB) and moves smoothly through it; the camera stops at the anchor's framing, so on a moving shot give `last_frame` too. Clip anchors keep the move but raise flicker up to +50%.

### Reference images and voices (Ref2VA)

A character's look (≤9 images) and voice (≤3 clips of 2–15 s) carried into a new scene, with the Ref2VA checkpoint mounted (opt-in) and the references named in the prompt as `<Picture i>` / `<Audio j>`:

```bash
RAPIDH3_REF2VA=/path/to/minimax_h3_ref2va_pruned_bf16.safetensors ./serve.sh restart
```

```python
prompt = ("... The fisherman from <Picture 1>, in his faded blue knit cap, sits in a harbour tavern at night ... "
          'Audio: in the voice of <Audio 1>, low and gravelly, the character says: "Nobody goes out past the lighthouse."')
g = workflow.build(prompt, width=1344, height=768, length=141, seed=7, prefix="run/shot",
                   ref_images=["chars/fisherman.png"], ref_audios=["voice_fisherman.wav"], **workflow.PRESETS["ts_s6"])
```

`dit` switches to the Ref2VA checkpoint by itself (quantized to our INT8 at load; mount it with `RAPIDH3_REF2VA`, or put it in the models folder's `diffusion_models/`); no first or last frame with references. Measured on 7 shots against stock bf16 at 10 steps (2026-10-27): identity (ArcFace to the reference) 0.673 `ts_s6` / 0.707 `ts_h2` vs 0.678 stock, voice similarity 0.633 / 0.626 vs 0.632 (unrelated voices ≈0.24), lip sync and voice naturalness at stock level; `ts_s6` ≈205 s, `ts_h2` ≈122 s per clip (measured before attention config 190 and decoder level 4, which save a few seconds more).

- **Full-frame reference images:** use `ts_s6` with `steps=10`. At 6 steps a portrait close-up can break and a two-person frame can duplicate a person (stock bf16 does the same).
- **`ts_h2`:** only with cropped references (a face or character crop); with full-frame references its small draft pastes the reference shot into the clip.
- **Reference videos** work but are expensive (`ts_s6` ≈81 s/step, ≈550 s per clip), and a short portrait reference video tends to be continued rather than re-staged.
- `encode_once` works with references; the INT8 path is as accurate here as on FL2VA (relative to the checkpoint's own bf16 noise).

## Using it in another ComfyUI

`ts_h2` and `ts_s6` are ordinary ComfyUI graphs; `workflow.build` only writes them. Node chain:
- **ts_s6:** UNETLoader (`minimax_h3_fl2va_pruned_bf16`) → MiniMaxH3SigmaShift (12 / 3) → ModelAttentionBackend (`comfy kitchen attention`) → **RapidH3Int8Attention** (config 190) → **RapidH3LinearQuant** (`int8_rot`) → [RapidH3StepTimer] → BasicGuider ← MiniMaxH3ImageToVideo (full size; text encoder through **RapidH3TextEncoderQuant**) → SamplerCustomAdvanced (`res_multistep`, BasicScheduler `simple`, 6 steps) → **RapidH3DecodeSaveVideo** (int8 video VAE) + VAEDecodeAudio.
- **ts_h2:** the same model chain; MiniMaxH3ImageToVideo at 0.7× size → SamplerCustomAdvanced (6 steps) → LTXVSeparateAVLatent → **MinimaxH3LatentUpscaler3D** (video, to full size) + LTXVFreezeLatent (audio) → LTXVConcatAVLatent → SamplerCustomAdvanced with ManualSigmas `0.72, 0` and the full-size conditioning from **RapidH3DraftConditioning** → RapidH3DecodeSaveVideo.

What it needs:
- ComfyUI 0.42.1 with comfy-kitchen 0.2.39 (the default image) or ComfyUI 0.39 with comfy-kitchen 0.2.37, with the MiniMax H3 nodes. Start it with `--disable-comfy-compiler`.
- This repo's node pack `custom_nodes/rapid_h3` with its compiled kernels (CUTLASS INT8 GEMM, Triton quantizers, attention, VAE ops; `REBUILD.md`). Importing the pack also installs the faster video decoder (`RAPIDH3_VAE_FAST=0` turns it off).
- For `ts_h2`: the `Comfyui_Minimax_h3_latent_Upscaler` node pack and its weights (below).
- Models: FL2VA DiT (bf16, quantized at load, or int8_convrot), int8 video VAE, fp32 audio VAE, Qwen3-VL int8 text encoder.
- Optional: RapidH3StepTimer and RapidH3SaveLatent can be removed; RapidH3DecodeSaveVideo can be replaced by VAEDecode → CreateVideo → SaveVideo (+≈2 s, same frames).

## Rolling back

- The settings before 2026-10-26 (comfy-kitchen attention, two encodes, fp32 text encoder, decoder level 1) are the presets `ts_h2_ck` / `ts_s6_ck`; `RAPIDH3_VAE_FAST=1 ./serve.sh restart` restores the previous decoder.
- The earlier two-step refine is the preset `ts_h` (`refine_sigmas="0.78, 0.45, 0"`).
- The previous attention kernel is `rapidh3_attn=194` (bit-identical output), the previous decoder level `RAPIDH3_VAE_FAST=3` (bit-identical).

## Licence

Copyright 2026 The Rapid-H3 authors. This repository's code (node pack, kernels, harness, scripts) is licensed under the **Apache License 2.0** (`LICENSE`). A few parts are adapted from third-party code (comfy-kitchen, NVIDIA CUTLASS, syncnet_python, DNSMOS) under their Apache-2.0 / BSD-3-Clause / MIT licences; `NOTICE` lists them and `licenses/` holds their texts. The model weights and the components below keep their own licences. The node pack runs inside ComfyUI, which is GPL-3.0.

Rapid-H3 is an independent project, not affiliated with or endorsed by MiniMax, NVIDIA or Comfy Org.

### Dependencies

| Item | Source | Licence |
|---|---|---|
| MiniMax-H3 weights | Comfy-Org/MiniMax-H3 on Hugging Face | MiniMax H3 Community License: excludes the USA, EU, UK and South Korea unless you apply separately (https://platform.minimax.io/h3-license). Check it before distributing anything derived from the weights |
| ComfyUI 0.42.1 (or 0.39), comfy-kitchen 0.2.39 (or 0.2.37) | Comfy-Org | GPL-3.0 / Apache-2.0 |
| `MinimaxH3LatentUpscaler3D` node | github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler, commit `40316cf` | MIT |
| `minimax_h3_latent_upscaler_3d_fp16.safetensors` (690.6 MB, sha256 `043e5a48…12eb55c2`) | huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler | see its model card |
| CUTLASS v4.2.1 | NVIDIA | BSD-3-Clause |
| Quality-evaluation tools (`harness/eval_setup.sh` only) | InsightFace `buffalo_l`, pyiqa, DOVER, SyncNet / S3FD weights, … | some are research / non-commercial only; not needed to render |

`serve.sh` mounts the upscaler node and weights from `~/rapid-h3/upscaler/` (`custom_nodes/`, `models/latent_upscale_models/`); `RAPIDH3_UPSCALER=/path/to/folder` points it at another copy with the same layout. `LTXVSeparateAVLatent`, `LTXVConcatAVLatent`, `LTXVFreezeLatent` and `ManualSigmas` are stock ComfyUI nodes.

## Repository

- `custom_nodes/rapid_h3/`: the ComfyUI node pack (quantized linears, attention, text-encoder path, decoder, draft conditioning, decode-and-save, timers, diagnostics). Kernel sources in `csrc/`.
- `harness/`: `workflow.py` (graph builder and `PRESETS`), `render.py` (submit, time, record), `fast_test.py` (shot × variant runs), `eval_clips.py` / `audio_quality.py` (quality metrics), `shots/` (test sets).
- `bench/`: microbenchmarks and their results.
- `workflows/`: the ready-made API graphs.
- `serve.sh`: runs the ComfyUI container (`rapid-h3-comfy`). `REBUILD.md`: building it all from an empty Spark.
- `docs/kernels.md`: the kernel layer in depth (design, speed, accuracy, what did not work).
- `runs/`: raw results of the end-to-end checks (timings, bit-identity checks). `bench/results/`: kernel microbenchmarks.
- `CHANGELOG.md`, `CREDITS.md`, `NOTICE` (third-party code and licenses, texts in `licenses/`), `LICENSE` (Apache-2.0).
