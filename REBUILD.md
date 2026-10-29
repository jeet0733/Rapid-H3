# Rebuilding the setup on an empty DGX Spark

Everything needed to run Rapid-H3 on a DGX Spark (GB10, 128 GB) that has only DGX OS, the NVIDIA driver and Docker: the repository, the image, the weights and, optionally, the evaluation environment. Run the steps in this order; paths are under your home folder unless absolute. Your user must be able to run `docker` (member of the `docker` group, as on a default DGX OS install).

## 1. The repository (`~/rapid-h3`)

```bash
git clone https://github.com/jeet0733/Rapid-H3.git ~/rapid-h3
cd ~/rapid-h3 && git clone --depth 1 --branch v4.2.1 https://github.com/NVIDIA/cutlass third_party/cutlass
```

The CUDA kernels (`custom_nodes/rapid_h3/csrc/`) are JIT-built by torch on the first render into `~/rapid-h3/build` (mounted as `/work/build`), with the CUTLASS headers from `third_party/cutlass`. The first render after a fresh build takes a few minutes longer; nothing to do by hand.

## 2. Docker image `rapid-h3/h3-comfyui:v0.42.1`

`serve.sh` runs **`rapid-h3/h3-comfyui:v0.42.1`**, built from `docker/` (recipe and reasons in `docker/README.md`): the NGC base `nvcr.io/nvidia/pytorch:25.11-py3`, ComfyUI 0.42.1 from the official tag tarball with its requirements (versions held by `docker/constraints.txt`), torchaudio built from source, and apt ffmpeg. About 2.5 min on a Spark once the base image is pulled:

```bash
cd ~/rapid-h3/docker
./fetch_sources.sh                                       # comfyui-src/ and audio-src/ (on the host: the docker bridge is slow)
docker build --network host -t rapid-h3/h3-comfyui:v0.42.1 .
docker run --rm rapid-h3/h3-comfyui:v0.42.1 sh -c "tail -1 comfyui_version.py; pip show comfy-kitchen | grep Version"   # 0.42.1, 0.2.39
```

| Package | Version |
|---|---|
| Base | `nvcr.io/nvidia/pytorch:25.11-py3` |
| ComfyUI | v0.42.1 (official tag tarball) |
| torch | 2.10.0a0+b558c986e8.nv25.11 (from the NGC base, never a PyPI wheel; the build fails if it moves) |
| torchaudio | 2.8.0a0, built from pytorch/audio `87ff22e49ed0` |
| comfy-kitchen | 0.2.39 (our video-decoder kernels copy parts of it and check the version: 0.2.37 and 0.2.39 are verified, any other falls back to the previous decoder path) |
| comfy-aimdo | 0.5.6 |
| numpy | 2.1.0 |

Full package list: `docker/image-v0.42.1-packages.txt`. A build from this recipe renders `er41_ts_h2` and `er41_ts_s6` (`harness/shots/cu421/`) bit-identically to the reference renders in `runs/2026-10-27/image_check/` (`ffmpeg -f framemd5` of every video and audio frame).

`RAPIDH3_IMAGE=<tag> ./serve.sh restart` runs another image (e.g. a test build); `./serve.sh restart` without it goes back to the default.

## 3. Weights

All from public Hugging Face repos; the sha256 of every file we ran with matches the Hugging Face LFS hash. `serve.sh` expects them under `$RAPIDH3_MODELS/<folder>/` (default `~/h3/models`; set `RAPIDH3_MODELS` to wherever you put them), except the int8 video VAE, which goes to `~/rapid-h3/models/vae/`. About 72 GB for the required files.

| File (folder) | Repo | Size | sha256 | Needed for |
|---|---|---|---|---|
| `diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors` | Comfy-Org/MiniMax-H3 | 40.23 GB | `a32572fb…6535d541b4` | **required**: loaded in bf16 and quantized to INT8 at load (`RapidH3LinearQuant`) |
| `text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | Comfy-Org/MiniMax-H3 | 27.14 GB | `bc2ced0f…be283923` | **required** |
| `vae/minimax_h3_video_vae_int8_convrot.safetensors` | Comfy-Org/MiniMax-H3 | 2.81 GB | `52a2c8c7…5c8e60c0e` | **required** (in `~/rapid-h3/models/vae/`) |
| `vae/minimax_h3_audio_vae_fp32.safetensors` | Comfy-Org/MiniMax-H3 | 0.61 GB | `8e505d95…cba778d76ae4db48` | **required** |
| `minimax_h3_latent_upscaler_3d_fp16.safetensors` | LBH-123-AI/Minimax_h3_latent_Upscaler (file `minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors`, saved under this name) | 0.69 GB | `043e5a48…12eb55c2` | **required for `ts_h2`** (in `~/rapid-h3/upscaler/models/latent_upscale_models/`) |
| `diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors` | Comfy-Org/MiniMax-H3 | 40.23 GB | `37c0da79…a736ed6fe` | optional: Ref2VA (reference images / voices) |
| `vae/minimax_h3_video_vae_fp16.safetensors` | Comfy-Org/MiniMax-H3 | 5.21 GB | `7c1f1314…cdce5e522` | optional: the stock bf16 comparison (`orig10`) |
| `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` | Comfy-Org/MiniMax-H3 | 20.97 GB | `e889202c…615d03c47a` | optional: Comfy-Org's int8 checkpoint, for comparisons (`dit=int8_convrot`) |

DGX OS does not allow `pip install` into the system Python, so the Hugging Face CLI goes into its own environment. `~/h3/models` is `serve.sh`'s default weights folder; if you choose another one, set `RAPIDH3_MODELS` to it again in every shell that runs `serve.sh`.

```bash
python3 -m venv ~/hfenv && ~/hfenv/bin/pip install -U huggingface_hub
export PATH=~/hfenv/bin:$PATH RAPIDH3_MODELS=~/h3/models
for f in diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors \
         text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors \
         vae/minimax_h3_audio_vae_fp32.safetensors; do
  hf download Comfy-Org/MiniMax-H3 "$f" --local-dir "$RAPIDH3_MODELS"
done
hf download Comfy-Org/MiniMax-H3 vae/minimax_h3_video_vae_int8_convrot.safetensors --local-dir ~/rapid-h3/models

# latent upscaler (ts_h2): node at commit 40316cf, weights under the file name the presets load
U=~/rapid-h3/upscaler
git clone https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler $U/custom_nodes/Comfyui_Minimax_h3_latent_Upscaler
git -C $U/custom_nodes/Comfyui_Minimax_h3_latent_Upscaler checkout 40316cf
hf download LBH-123-AI/Minimax_h3_latent_Upscaler \
    minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors --local-dir /tmp/h3up
mkdir -p $U/models/latent_upscale_models
mv /tmp/h3up/minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors \
   $U/models/latent_upscale_models/minimax_h3_latent_upscaler_3d_fp16.safetensors
rm -rf /tmp/h3up
```

## 4. Start and smoke test

```bash
cd ~/rapid-h3
./serve.sh start                 # rapid-h3-comfy on 127.0.0.1:8189; RAPIDH3_MODELS=... if the weights are not in ~/h3/models
python3 - <<'EOF'                # text-only ts_h2 clip -> ~/rapid-h3/data/output/smoke/
import sys; sys.path.insert(0, "harness")
import workflow
from fast_test import run
g = workflow.build("A red fox trots through fresh snow at dawn, soft golden light, its breath steaming.",
                   width=768, height=1344, length=141, seed=7, prefix="smoke/fox", **workflow.PRESETS["ts_h2"])
hist, t0, t1 = run(g, 3600)
print(hist["status"]["status_str"] if hist else "timeout", round(t1 - t0), "s")   # expect: success
EOF
```

If it does not print `success`, `./serve.sh logs` shows the server's error. The first render builds the kernels and loads the models (≈3–4 min more than a warm render); a second render of the same size takes ≈2 min. Files the server writes under `~/rapid-h3/data` are owned by root (it runs as root in its container). For your own clips, see `README.md` ("Quick start").

**Evaluation environment** (optional, only to reproduce the quality numbers: `harness/eval_clips.py`, `harness/audio_quality.py`): `bash harness/eval_setup.sh` creates `~/rapid-h3/venv_eval` with the face, lip-sync, speech, flow and video-quality models (≈35–45 min, ≈7 GB venv + ≈4.5 GB models). The older side-by-side tool `harness/fast_compare.py` uses a lighter `venv_metrics` instead: `cd ~/rapid-h3 && python3 -m venv venv_metrics && venv_metrics/bin/pip install -r config/venv_metrics.requirements.txt`.

## 5. Results kept in the repository

- `runs/`: per-check end-to-end results (timings, bit-identity checks), see `runs/README.md`.
- `bench/results/*.json`: raw microbenchmarks behind the attention, GEMM and decoder conclusions in `docs/kernels.md`. Nsight Compute profiles and q/k/v dumps are not kept; `bench/attn_prof.py` and `RapidH3DumpQKV` regenerate them.
- The evaluation shots' first-frame images and the rendered clips are not part of the repository, so the quality numbers can be re-run only with your own reference images (`harness/shots/eval1.json` lists the prompts and lines).
