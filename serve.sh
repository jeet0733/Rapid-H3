#!/usr/bin/env bash
# Run our own ComfyUI on the Spark from our image (rapid-h3/h3-comfyui:v0.42.1, docker/README.md).
#   ./serve.sh start|stop|restart|logs|status
# Weights are mounted read-only; everything the server writes is under $RAPIDH3_DIR/data (default ~/rapid-h3/data).
# RAPIDH3_CACHE overrides the cache flag. Default --cache-ram keeps models loaded between renders (same
# step time, no reload: ≈30 s less per 768p render when measured on an earlier recipe); --cache-none reloads
# every render.
# RAPIDH3_EXTRA adds ComfyUI flags (default --disable-comfy-compiler: with the compiler on, a 10 s render that
# follows a 5 s render in the same server crawls at ~300 s/step instead of ~70; it gives no speed otherwise).
set -euo pipefail
# RAPIDH3_NAME / RAPIDH3_DIR / RAPIDH3_PORT run a second copy (own container, tree and host port)
NAME=${RAPIDH3_NAME:-rapid-h3-comfy}
IMAGE=${RAPIDH3_IMAGE:-rapid-h3/h3-comfyui:v0.42.1}   # built from docker/ (REBUILD.md); RAPIDH3_IMAGE=<tag> runs another image
ROOT=${RAPIDH3_DIR:-$HOME/rapid-h3}
PORT=${RAPIDH3_PORT:-8189}
M=${RAPIDH3_MODELS:-$HOME/h3/models}            # H3 weights (any other folder: RAPIDH3_MODELS=/path/to/models)
TE=${RAPIDH3_TEXT_ENCODERS:-$M/text_encoders}
mkdir -p "$ROOT/build" "$ROOT/models"

start() {
  mkdir -p "$ROOT/data"/{output,input,temp,models,custom_nodes}
  # H3 latent upscaler (two-stage tests): node and weights, read-only, from RAPIDH3_UPSCALER (default
  # $ROOT/upscaler; layout custom_nodes/Comfyui_Minimax_h3_latent_Upscaler + models/latent_upscale_models)
  local up=() U=${RAPIDH3_UPSCALER:-$ROOT/upscaler}
  [ -d "$U/custom_nodes/Comfyui_Minimax_h3_latent_Upscaler" ] && up+=(
    -v "$U/custom_nodes/Comfyui_Minimax_h3_latent_Upscaler:/data/custom_nodes/Comfyui_Minimax_h3_latent_Upscaler:ro"
    -v "$U/models/latent_upscale_models:/data/models/latent_upscale_models:ro")
  # RAPIDH3_REF2VA (opt-in): space-separated Ref2VA DiT files, each mounted read-only into /work/ref2va
  # (a diffusion_models folder in config/extra_model_paths.yaml), e.g.
  # RAPIDH3_REF2VA=/path/to/minimax_h3_ref2va_pruned_bf16.safetensors
  local f
  for f in ${RAPIDH3_REF2VA:-}; do
    [ -f "$f" ] || { echo "RAPIDH3_REF2VA: no file $f"; exit 1; }
    up+=(-v "$f:/work/ref2va/$(basename "$f"):ro")
  done
  # RAPIDH3_H3VAE_PYOPT=1 (opt-in): ComfyUI-H3VAE-PyOpt node pack + the MiniMax FL2VA/video_vae model
  # code it imports, Triton 3.7.1 ahead of the image's 3.5.0, own compile caches (harness/h3vae_pyopt_setup.sh).
  # Run it as a second copy: RAPIDH3_H3VAE_PYOPT=1 RAPIDH3_NAME=rapid-h3-pyopt RAPIDH3_PORT=8190 ./serve.sh start
  if [ "${RAPIDH3_H3VAE_PYOPT:-0}" = 1 ]; then
    local C=${RAPIDH3_H3VAE_CODE:-$ROOT/models/h3vae_code}/FL2VA/video_vae
    [ -f "$C/klvae.py" ] && [ -d "$ROOT/build/h3vae_pyopt/site/triton" ] || { echo "run harness/h3vae_pyopt_setup.sh first"; exit 1; }
    up+=(-v "$ROOT/custom_nodes/h3vae_pyopt:/data/custom_nodes/ComfyUI-H3VAE-PyOpt:ro"
         -v "$C:/work/h3_video_vae/video_vae:ro"
         -e H3_VAE_MODEL_CODE_DIR=/work/h3_video_vae/video_vae
         -e H3_VAE_WEIGHTS_PATH=/data/models/vae/minimax_h3_video_vae_fp16.safetensors
         -e PYTHONPATH=/work/build/h3vae_pyopt/site
         -e TRITON_CACHE_DIR=/work/build/h3vae_pyopt/triton_cache
         -e TORCHINDUCTOR_CACHE_DIR=/work/build/h3vae_pyopt/inductor_cache)
  fi
  docker run -d --name "$NAME" --gpus all --ipc host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -p 127.0.0.1:$PORT:8188 \
    -e RAPIDH3_STATS=/data/output/rapidh3_stats.json -e RAPIDH3_SHADOW=/data/output/rapidh3_shadow.json \
    -e RAPIDH3_PROBE="${RAPIDH3_PROBE:-}" -e RAPIDH3_TRACE="${RAPIDH3_TRACE:-}" -e RAPIDH3_TRACE_CLIP="${RAPIDH3_TRACE_CLIP:-}" -e RAPIDH3_TIMING="${RAPIDH3_TIMING:-}" -e RAPIDH3_VAE_FAST="${RAPIDH3_VAE_FAST:-4}" \
    -e RAPIDH3_CUTLASS=/work/third_party/cutlass -e RAPIDH3_BUILD=/work/build/fp8 \
    -v "$ROOT/third_party:/work/third_party:ro" -v "$ROOT/build:/work/build" \
    -v "$ROOT/models:/work/models:ro" -v "$ROOT/config:/work/config:ro" \
    -v "$ROOT/data:/data" \
    -v "$ROOT/custom_nodes/rapid_h3:/data/custom_nodes/rapid_h3:ro" \
    -v "$M/diffusion_models:/data/models/diffusion_models:ro" \
    -v "$TE:/data/models/text_encoders:ro" \
    -v "$M/vae:/data/models/vae:ro" \
    -v "$M/loras:/data/models/loras:ro" "${up[@]}" \
    --entrypoint python "$IMAGE" /opt/ComfyUI/main.py \
      --listen 0.0.0.0 --port 8188 --base-directory /data \
      --output-directory /data/output --input-directory /data/input --temp-directory /data/temp \
      --extra-model-paths-config /work/config/extra_model_paths.yaml \
      --reserve-vram 12 ${RAPIDH3_CACHE:---cache-ram} --disable-pinned-memory ${RAPIDH3_EXTRA---disable-comfy-compiler} >/dev/null
  local url=http://127.0.0.1:$PORT
  for _ in $(seq 120); do
    curl -sf $url/system_stats >/dev/null && { echo "$NAME up"; return; }
    sleep 2
  done
  echo "$NAME did not come up"; docker logs --tail 50 "$NAME"; exit 1
}

case "${1:-status}" in
  start) start ;;
  stop) docker rm -f "$NAME" >/dev/null 2>&1 || true ;;
  restart) docker rm -f "$NAME" >/dev/null 2>&1 || true; start ;;
  logs) docker logs --tail "${2:-80}" "$NAME" ;;
  status) docker ps -f name="$NAME" --format '{{.Names}} {{.Status}}' ;;
esac
