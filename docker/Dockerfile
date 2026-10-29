# ComfyUI image for MiniMax-H3 on DGX Spark (GB10, sm_121, aarch64).
#
#   ./fetch_sources.sh                                   # comfyui-src/ and audio-src/ next to this file
#   docker build --network host -t rapid-h3/h3-comfyui:v0.42.1 .
#
# The NGC base carries the only torch built for sm_121. Every pip step runs under constraints that
# pin the torch stack (read from the base) and ComfyUI's other dependencies (constraints.txt), and
# the build fails if torch changes anyway.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.11-py3
FROM ${BASE_IMAGE}
ARG COMFYUI_REF=v0.42.1
ARG TORCHAUDIO_REF=87ff22e49ed0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    COMFYUI_ROOT=/opt/ComfyUI

# ffmpeg/ffprobe CLI for scripts run inside the image
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

COPY constraints.txt /tmp/constraints.txt
RUN python - <<'EOF'
import importlib.metadata as m
with open('/tmp/constraints.txt', 'a') as f:
    for p in ('torch', 'torchvision', 'torchaudio', 'torchsde', 'numpy'):
        try: f.write(f'{p}=={m.version(p)}\n')
        except m.PackageNotFoundError: pass
EOF

# ComfyUI from the tag tarball (fetched on the host). Its requirements without the torch stack and
# without the UI workflow-template and docs bundles (the server is driven over its HTTP API).
# The build checks that the tree's version matches COMFYUI_REF.
COPY comfyui-src/ /opt/ComfyUI/
RUN set -eux; \
    grep -q "\"${COMFYUI_REF#v}\"" /opt/ComfyUI/comfyui_version.py; \
    python -c "import torch; print(torch.__version__)" > /tmp/torch.before; \
    grep -viE '^(torch|torchvision|torchaudio|torchsde)([^a-z]|$)|workflow.templates|embedded.docs' \
        /opt/ComfyUI/requirements.txt > /tmp/req.txt; \
    python -m pip install --no-cache-dir --timeout 60 --retries 10 -c /tmp/constraints.txt -r /tmp/req.txt; \
    python -m pip install --no-cache-dir --no-deps -c /tmp/constraints.txt torchsde trampoline; \
    python -c "import torch; print(torch.__version__)" > /tmp/torch.after; \
    cmp /tmp/torch.before /tmp/torch.after

# torchaudio from source: no wheel matches the NGC torch, and the H3 audio VAE imports it.
COPY audio-src/ /tmp/audio-src/
RUN set -eux; \
    python -m pip install --no-cache-dir --no-build-isolation --no-deps /tmp/audio-src; \
    rm -rf /tmp/audio-src; \
    python -c "import torch; print(torch.__version__)" > /tmp/torch.after; \
    cmp /tmp/torch.before /tmp/torch.after; \
    python -c "import torchaudio, comfy_kitchen; print('torchaudio', torchaudio.__version__)"

LABEL rapid-h3.comfyui-ref=${COMFYUI_REF} rapid-h3.torchaudio-ref=${TORCHAUDIO_REF}
WORKDIR /opt/ComfyUI
EXPOSE 8188
# serve.sh replaces this with its own flags and mounts
CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188"]
