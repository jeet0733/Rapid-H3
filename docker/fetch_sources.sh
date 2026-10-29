#!/usr/bin/env bash
# Download the two source trees the Dockerfile copies in: the ComfyUI tag tarball and pytorch/audio at
# the commit that builds against the NGC torch. Done on the host because the docker bridge is slow.
#   COMFYUI_REF=v0.42.1 TORCHAUDIO_REF=87ff22e49ed0 ./fetch_sources.sh
set -euo pipefail
cd "$(dirname "$0")"
COMFYUI_REF=${COMFYUI_REF:-v0.42.1}
TORCHAUDIO_REF=${TORCHAUDIO_REF:-87ff22e49ed0}

fetch() {  # url dest
  rm -rf "$2" && mkdir -p "$2"
  curl -fsSL --retry 5 "$1" | tar xz -C "$2" --strip-components=1
}
fetch "https://codeload.github.com/Comfy-Org/ComfyUI/tar.gz/refs/tags/${COMFYUI_REF}" comfyui-src
fetch "https://codeload.github.com/pytorch/audio/tar.gz/${TORCHAUDIO_REF}" audio-src
echo "ComfyUI $(sed -n 's/.*"\(.*\)".*/\1/p' comfyui-src/comfyui_version.py) | torchaudio ${TORCHAUDIO_REF}"
