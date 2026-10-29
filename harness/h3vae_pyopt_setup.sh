#!/usr/bin/env bash
# Prepare ComfyUI-H3VAE-PyOpt for our rapid-h3-comfy container on a DGX Spark.
# Run on the Spark host from ~/rapid-h3 while the container is up (`./serve.sh status`).  Nothing here edits
# serve.sh or the running server; the last step prints how to start the opt-in server and verify it.
#
#   harness/h3vae_pyopt_setup.sh            # all steps
#   SKIP_CODE=1 SKIP_PIP=1 harness/h3vae_pyopt_setup.sh   # clone + patch only
#
# What it does
#   1. clones github.com/fishelegs/ComfyUI-H3VAE-PyOpt at a pinned commit into custom_nodes/h3vae_pyopt
#      (serve.sh mounts our packs from this repo's custom_nodes/, read-only)
#   2. applies harness/patches/h3vae_pyopt_sm121.patch: the three `capability == (12, 0)` gates become
#      `capability[0] == 12`, so GB10 (sm_121) takes the same INT8 kernel selection as sm_120.  The FP16
#      default path has no SM gate (only `>= (8, 0)`), so the patch only matters for int8_decode/int8_encode.
#   3. downloads the MiniMax H3 `FL2VA/video_vae` model-code bundle (config + Python, NOT the 10 GB fp32
#      source weights) that the loader imports as the package `video_vae`; weights are our existing
#      $M/vae/minimax_h3_video_vae_fp16.safetensors (sha256 7c1f1314..., the file the node was tested on)
#   4. installs Triton 3.7.1 into a persistent `pip --target` dir under ~/rapid-h3/build (mounted at /work/build),
#      so it survives `serve.sh restart` and is only seen by a server started with PYTHONPATH set.
#      The image's pytorch-triton 3.5.0 stays in site-packages and is broken on sm_121; 3.7.1 is known to work.
#   5. prints how to start it (serve.sh RAPIDH3_H3VAE_PYOPT=1 opt-in) and verify it against our int8 decode.
#
# Not done on purpose: `pip install comfy-kitchen==0.2.34`.  int8_decode needs exactly 0.2.34 and our node
# pack (RapidH3LinearQuant, vae_fast) needs the image's 0.2.39, so the INT8 decoder can only be tried in a second
# container (RAPIDH3_NAME=rapid-h3-pyopt RAPIDH3_PORT=8190 ./serve.sh start, then pip install 0.2.34 inside it).
set -euo pipefail

ROOT=${RAPIDH3_DIR:-$HOME/rapid-h3}
NAME=${RAPIDH3_NAME:-rapid-h3-comfy}
M=${RAPIDH3_MODELS:-$HOME/h3/models}
CODE=${RAPIDH3_H3VAE_CODE:-$ROOT/models/h3vae_code}        # bundle lands in $CODE/FL2VA/video_vae
REPO=https://github.com/fishelegs/ComfyUI-H3VAE-PyOpt
PIN=5face9697f0170c4e67a39f3728cc8da3f6607e7         # 2026-10-08 "perf: adopt exact SM120 INT8 norm1 decoder fusion"
DEST=$ROOT/custom_nodes/h3vae_pyopt
PATCH=$ROOT/harness/patches/h3vae_pyopt_sm121.patch
BUILD=$ROOT/build/h3vae_pyopt                           # /work/build/h3vae_pyopt in the container
SITE=$BUILD/site
TRITON_VER=3.7.1

say() { printf '\n== %s\n' "$*"; }

# ---------------------------------------------------------------- 1. clone at the pinned commit
say "node pack -> $DEST @ ${PIN:0:7}"
if [ -d "$DEST/.git" ]; then
  git -C "$DEST" fetch -q origin
else
  git clone -q "$REPO" "$DEST"
fi
git -C "$DEST" -c advice.detachedHead=false checkout -q "$PIN"
[ "$(git -C "$DEST" rev-parse HEAD)" = "$PIN" ] || { echo "checkout did not land on $PIN"; exit 1; }
# ComfyUI treats the folder as the pack name; the upstream name is kept out of the mount so the on-disk
# folder (h3vae_pyopt) and the mounted one (ComfyUI-H3VAE-PyOpt) are both unambiguous.

# ---------------------------------------------------------------- 2. sm_121 gate patch
say "sm_121 gates ($PATCH)"
if git -C "$DEST" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "already applied"
else
  git -C "$DEST" apply --check "$PATCH"
  git -C "$DEST" apply "$PATCH"
  echo "applied"
fi
git -C "$DEST" --no-pager diff --stat

# ---------------------------------------------------------------- 3. model code bundle (FL2VA/video_vae)
if [ -z "${SKIP_CODE:-}" ]; then
  say "model code -> $CODE/FL2VA/video_vae (MiniMaxAI/MiniMax-H3)"
  command -v hf >/dev/null || { echo "need the hf CLI: pip install -U huggingface_hub (see REBUILD.md)"; exit 1; }
  mkdir -p "$CODE"
  # the loader needs config.json, the *.py modules and source/config.json; source/model.safetensors (10.4 GB
  # fp32) is NOT needed, the weights come from the Comfy-Org fp16 export we already have
  hf download MiniMaxAI/MiniMax-H3 --include 'FL2VA/video_vae/*' --exclude 'FL2VA/video_vae/source/model.safetensors' \
    --local-dir "$CODE"
  for f in config.json klvae.py parallel.py flash.py attention.py vae_vit.py vae_cnn.py source/config.json; do
    [ -f "$CODE/FL2VA/video_vae/$f" ] || { echo "bundle incomplete: $f missing"; exit 1; }
  done
  [ -f "$CODE/FL2VA/video_vae/source/model.safetensors" ] && echo "note: source/model.safetensors present (10 GB, unused by the loader)"
  echo "bundle ok"
fi
[ -f "$M/vae/minimax_h3_video_vae_fp16.safetensors" ] || echo "WARNING: $M/vae/minimax_h3_video_vae_fp16.safetensors missing (REBUILD.md section 2)"

# ---------------------------------------------------------------- 4. Python deps inside the container
if [ -z "${SKIP_PIP:-}" ]; then
  say "Triton $TRITON_VER -> $SITE (container: /work/build/h3vae_pyopt/site)"
  docker ps --format '{{.Names}}' | grep -qx "$NAME" || { echo "container $NAME is not running (./serve.sh start)"; exit 1; }
  mkdir -p "$SITE"
  # --target keeps the image's pytorch-triton 3.5.0 untouched; PYTHONPATH (set by serve.sh when
  # RAPIDH3_H3VAE_PYOPT=1) puts 3.7.1 first.  --no-deps: triton 3.7.1 has no runtime deps on py3.12, and
  # diffusers (imported by the klvae bundle; 0.36+ needs a newer torchao than the image's) only needs
  # packages the image already has.
  docker exec "$NAME" python -m pip install -q --no-deps --no-cache-dir --upgrade \
    --target /work/build/h3vae_pyopt/site "triton==$TRITON_VER" "diffusers==0.35.2"
  docker exec -e PYTHONPATH=/work/build/h3vae_pyopt/site "$NAME" python - <<'PY'
import importlib.metadata as md, torch, triton
cap = torch.cuda.get_device_capability()
print(f"triton {triton.__version__} from {triton.__file__}")
print(f"torch {torch.__version__} cuda {torch.version.cuda} capability {cap}")
for p in ("safetensors", "diffusers", "comfy-kitchen"):
    print(p, md.version(p))
assert triton.__version__.startswith("3.7."), "PYTHONPATH did not take precedence"
assert cap[0] == 12, "not a GB10/Blackwell device"
PY
  # Triton's kernel cache is keyed by version but the sm_120 -> sm_121 trap bit people before: use a fresh,
  # persistent cache dir for this server (TRITON_CACHE_DIR below) and wipe it whenever Triton changes.
  rm -rf "$BUILD/triton_cache" "$BUILD/inductor_cache"
  mkdir -p "$BUILD/triton_cache" "$BUILD/inductor_cache"
  echo "WARNING: if you ever run Triton $TRITON_VER with the default cache, clear ~/.triton/cache (host and container) first:"
  echo "         kernels cached under sm_120 are not valid for sm_121."
fi

# ---------------------------------------------------------------- 5. what to change and how to verify
cat <<EOF

== serve.sh: the opt-in is already there (RAPIDH3_H3VAE_PYOPT=1, next to the RAPIDH3_REF2VA block in start());
it mounts this pack, the model-code bundle (RAPIDH3_H3VAE_CODE, default \$ROOT/models/h3vae_code) as .../video_vae,
puts /work/build/h3vae_pyopt/site first on PYTHONPATH and uses its own Triton / inductor caches.

Notes on the mount: the bundle dir must be mounted under a folder literally named video_vae (the runtime adds
its parent to sys.path and imports video_vae.klvae).  PYTHONPATH switches the WHOLE server to Triton 3.7.1,
including our Rapid-H3 Triton quantizers; run it as a second copy:
  RAPIDH3_H3VAE_PYOPT=1 RAPIDH3_NAME=rapid-h3-pyopt RAPIDH3_PORT=8190 ./serve.sh start
  RAPIDH3_NAME=rapid-h3-pyopt ./serve.sh logs 200 | grep -iE "h3vae|triton|Import times|IMPORT FAILED"

== workflow: harness/workflow.py build(pyopt_vae="fp16" | "int8") puts H3VAEPyOptLoader in front of the final
  decode (the first-frame encode keeps the int8 VAE). decode_save="overlap" also works: RapidH3DecodeSaveVideo falls
  back to overlap=False because the PyOpt runtime has no _finalize_pixels; the mp4 is the same.  The node runs with
  cudnn_benchmark=False so it does not flip the process-wide torch.backends.cudnn.benchmark under the DiT.  First
  decode pays the inductor compile (minutes); compile_decoder=False gives the eager fused path if torch.compile
  misbehaves on torch 2.10 + Triton 3.7.1.

== verify: same latent, our int8 decode vs PyOpt, PSNR with the existing harness/score.py
  # on the Spark host, one prompt, same seed; the reference is ts_h2 as of 2026-10-24 without decode_save
  python3 harness/render.py --variant pyopt_ref --prompts harness/prompts.txt --only 2 \\
      --set steps=6 draft=0.7 refine_sigmas="0.72, 0" freeze_audio=true quant=int8_rot \\
            attention="comfy kitchen attention" video_vae=int8 turbo8=false sampler=res_multistep
  python3 harness/render.py --variant pyopt_fp16 --prompts harness/prompts.txt --only 2 --set <same> pyopt_vae=fp16
  # pyopt_ref/p02_*.latent and pyopt_fp16/p02_*.latent must be identical (relL2 0): only the decoder differs
  docker cp harness/score.py $NAME:/tmp/score.py
  docker exec $NAME python /tmp/score.py --out /data/output --ref pyopt_ref --cand pyopt_fp16
  # columns: latent_video_relL2 (expect 0), psnr / psnr_last_quarter / ssim of the mp4s = decoder difference.
  # Yardstick: our int8_convrot VAE vs the fp16 VAE on one latent is 42.7 dB / SSIM 0.988 (docs/kernels.md);
  # the gate is >= 50 dB against our int8 decode for an INT8 PyOpt path.
  # Time: "[H3VAE-PyOpt] decode #n: ... ms" in \`./serve.sh logs\`, against "[rapid-h3] decode_save: decode" of the ref.
  # Codec-free alternative for the reference side only: bench/vae_bench.py <latent> <int8 vae> --fast --save ref.pt
EOF
