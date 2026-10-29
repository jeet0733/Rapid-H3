#!/usr/bin/env bash
# One-time setup of the clip-evaluation venv for harness/eval_clips.py (DGX Spark, aarch64, GB10 sm_121, CUDA 13).
#
#   bash harness/eval_setup.sh            # from ~/rapid-h3; ~35-45 min on the Spark's network, ~7 GB venv + ~4.5 GB models
#
# Creates ~/rapid-h3/venv_eval, clones SyncNet and DOVER into ~/rapid-h3/third_party with their official weights, and
# pre-downloads every model into ~/rapid-h3/eval_cache (HF_HOME, TORCH_HOME), so other users' caches are untouched.
# Versions are the ones validated on 2026-10-24.
set -euo pipefail
ROOT=${RAPIDH3_DIR:-$HOME/rapid-h3}
VENV=$ROOT/venv_eval
TP=$ROOT/third_party
export RAPIDH3_EVAL_CACHE=${RAPIDH3_EVAL_CACHE:-$ROOT/eval_cache}
export HF_HOME=$RAPIDH3_EVAL_CACHE/hf TORCH_HOME=$RAPIDH3_EVAL_CACHE/torch
mkdir -p "$TP" "$RAPIDH3_EVAL_CACHE"
cd "$ROOT"

[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q -U pip==26.2.2 wheel==0.48.0
# torch for CUDA 13 (aarch64 wheels run on GB10 / sm_121)
"$VENV/bin/pip" install -q --index-url https://download.pytorch.org/whl/cu130 \
    torch==2.14.1+cu130 torchvision==0.29.1+cu130 torchaudio==2.11.0+cu130
cat > "$RAPIDH3_EVAL_CACHE/constraints.txt" <<'EOF'
torch==2.14.1+cu130
torchvision==0.29.1+cu130
torchaudio==2.11.0+cu130
EOF
# demucs 4.0.1: newer demucs needs sphn, which does not build on aarch64.
# setuptools < 81: openai-clip (pyiqa's CLIP-IQA) imports pkg_resources.
"$VENV/bin/pip" install -q -c "$RAPIDH3_EVAL_CACHE/constraints.txt" --extra-index-url https://download.pytorch.org/whl/cu130 \
    numpy==2.5.2 scipy==1.18.1 scikit-image==0.26.0 PyWavelets==1.10.0 opencv-python-headless==5.0.0.93 av==19.0.2 \
    insightface==2.1 onnxruntime==1.31.0 transformers==5.20.2 huggingface_hub==2.3.0 timm==1.0.30 \
    pyiqa==0.1.16 demucs==4.0.1 speechbrain==1.1.2 faster-whisper==1.2.1 ctranslate2==4.8.3 librosa==1.0.0 \
    python_speech_features==0.6 PyYAML==6.0.3 einops==0.8.2 "setuptools==80.10.2"

# SyncNet (joonson/syncnet_python, MIT) + S3FD face detector, official weights from robots.ox.ac.uk
if [ ! -d "$TP/syncnet_python" ]; then
    git clone -q https://github.com/joonson/syncnet_python.git "$TP/syncnet_python"
    git -C "$TP/syncnet_python" checkout -q 907c0b579c2e2d83f0eae1b2ac9e720cde4e5623
fi
mkdir -p "$TP/syncnet_python/data" "$TP/syncnet_python/detectors/s3fd/weights"
[ -s "$TP/syncnet_python/data/syncnet_v2.model" ] || \
    wget -q http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model -O "$TP/syncnet_python/data/syncnet_v2.model"
[ -s "$TP/syncnet_python/detectors/s3fd/weights/sfd_face.pth" ] || \
    wget -q https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/sfd_face.pth -O "$TP/syncnet_python/detectors/s3fd/weights/sfd_face.pth"
[ -s "$TP/syncnet_python/data/example.avi" ] || \
    wget -q http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/example.avi -O "$TP/syncnet_python/data/example.avi"

# DOVER (VQAssessment/DOVER) + DOVER.pth from its GitHub release
if [ ! -d "$TP/DOVER" ]; then
    git clone -q https://github.com/VQAssessment/DOVER.git "$TP/DOVER"
    git -C "$TP/DOVER" checkout -q f1ddc96215bc7fbcf8f315c65d47905f339c3419
fi
mkdir -p "$TP/DOVER/pretrained_weights"
[ -s "$TP/DOVER/pretrained_weights/DOVER.pth" ] || \
    wget -q https://github.com/QualityAssessment/DOVER/releases/download/v0.1.0/DOVER.pth -O "$TP/DOVER/pretrained_weights/DOVER.pth"

# Pre-download every model (DINOv2, CLIP L/14, RAFT, LPIPS, MUSIQ, CLIP-IQA, DOVER, htdemucs, ECAPA, Whisper,
# InsightFace buffalo_l) and check SyncNet against its reference result on example.avi (offset 3, dist 5.35, conf 10.0).
HERE=$(cd "$(dirname "$0")" && pwd)
"$VENV/bin/python" - "$HERE" "$TP" <<'EOF'
import sys, types, numpy as np, cv2
sys.path.insert(0, sys.argv[1])
import eval_clips as E
E.ARGS = types.SimpleNamespace(syncnet_dir=sys.argv[2] + "/syncnet_python", dover_dir=sys.argv[2] + "/DOVER",
                               cache=E._C, threads=8, whisper="large-v3-turbo")
E.face_app(); E.dino(); E.clip_model(); E.raft(); E.iqa("lpips"); E.iqa("musiq"); E.iqa("clipiqa"); E.dover()
E.demucs_model(); E.ecapa(); E.whisper(); E.normalizer()
fr, fps, audio, sr = E.load_video(sys.argv[2] + "/syncnet_python/data/example.avi")
a16 = np.clip(E.resample(audio.mean(0), sr, 16000) * 32768, -32768, 32767).astype(np.int16)
r = E.syncnet_score([cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in fr], a16)
print("SyncNet example.avi:", r, "(reference: offset 3, min dist 5.353, confidence 10.021)")
assert r["av_offset"] == 3 and abs(r["lse_d"] - 5.35) < 0.2 and abs(r["lse_c"] - 10.0) < 0.3
print("eval setup OK")
EOF
