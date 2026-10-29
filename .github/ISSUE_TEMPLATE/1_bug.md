---
name: Bug report
about: A render fails, is slow, or looks wrong
labels: bug
---

**What happened** (error text, or what looks wrong in the clip):

**Preset and options** (`ts_h2` / `ts_s6`, and any `workflow.build` keywords such as `last_frame`, `audio_input`, `ref_images`):

**Clip:** size, frame count, first frame or text only, seed:

**Setup:** DGX Spark? Docker image tag (`docker images | grep h3-comfyui`), ComfyUI version, `RAPIDH3_VAE_FAST` if set:

**Timing** (from `data/output/rapidh3_steps.jsonl` or `harness/fast_test.py`, not ComfyUI's progress bar):

**Server log** (`./serve.sh logs`, the last ~50 lines):
