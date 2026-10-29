# docker/

The image `serve.sh` runs, `rapid-h3/h3-comfyui:v0.42.1`, and how to build it from scratch on a DGX Spark.

| File | What |
|---|---|
| `Dockerfile` | NGC PyTorch base + ComfyUI tag tarball + its requirements + torchaudio from source + ffmpeg |
| `fetch_sources.sh` | downloads the ComfyUI tarball and the torchaudio source next to the Dockerfile (on the host: the docker bridge is slow) |
| `constraints.txt` | versions of ComfyUI's unpinned dependencies, held at the validated set (review when changing `COMFYUI_REF`) |
| `image-v0.42.1-packages.txt` | `pip list --format=freeze` of the built image |

```bash
cd docker
./fetch_sources.sh                                      # comfyui-src/, audio-src/
docker build --network host -t rapid-h3/h3-comfyui:v0.42.1 .
docker run --rm rapid-h3/h3-comfyui:v0.42.1 sh -c "tail -1 comfyui_version.py; pip show comfy-kitchen | grep Version"   # 0.42.1, 0.2.39
```

Build args: `COMFYUI_REF` (default `v0.42.1`), `TORCHAUDIO_REF` (`87ff22e49ed0`), `BASE_IMAGE` (`nvcr.io/nvidia/pytorch:25.11-py3`). Pass the same refs to `fetch_sources.sh` as environment variables. A from-scratch build takes about 2.5 min on a Spark once the base image (~20 GB) is pulled; half of it is the torchaudio compile.

What is in it and why:

- **torch** `2.10.0a0+b558c986e8.nv25.11` from the NGC base, the only build with sm_121 kernels. It is never replaced by a PyPI wheel: pip runs with the torch stack and numpy pinned in a constraints file, and the build fails if torch's version changes.
- **ComfyUI** from the official tag tarball, with its `requirements.txt` (which pins comfy-kitchen and comfy-aimdo) minus the torch stack and the UI workflow-template and docs bundles; the server is driven over its HTTP API. The rest of its dependencies are held by `constraints.txt`: left free, pip picks newer av (whose bundled FFmpeg writes the mp4s), transformers and huggingface_hub.
- **torchaudio** built from pytorch/audio `87ff22e49ed0` against the NGC torch (no matching wheel exists); the H3 audio VAE imports it.
- **ffmpeg** (apt) for scripts that call the `ffmpeg`/`ffprobe` CLI inside the image.

Nothing else: no entrypoint script, no model downloader, and none of the extra packages the previous image carried (diffusers, peft, accelerate, rotary-embedding-torch, omegaconf, gguf, opencv-python-headless), which nothing of ours imports inside the container; `harness/h3vae_pyopt_setup.sh` installs the diffusers its opt-in VAE needs into its own `--target` dir. `serve.sh` starts `python /opt/ComfyUI/main.py` with its own flags and mounts, and JIT-builds our extension at first use. `RAPIDH3_IMAGE=<tag> ./serve.sh restart` runs another image.

Renders from this recipe are bit-identical to the image it replaced (latents and decoded frames; `REBUILD.md` section 2).
