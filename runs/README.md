# runs/

Raw results of the end-to-end checks behind the README's claims, one folder per date and check, each run on one DGX
Spark. Kernel microbenchmarks live in `bench/results/`.

The `results.jsonl` files are `harness/feature_test.py` results for the jobs in
`harness/shots/cu421/jobs.json` (`er41` on `ts_h2` and `ts_s6`, 141 frames, seed 7): `wall_s` is submit → mp4,
`steps_s` the synced per-step times (`RapidH3StepTimer`), `pre_s` / `post_s` the time before the first and after the last
step. The first render after a server restart includes model loading and one-time kernel builds (`steps_s[0]`, `pre_s`).
The `release_timing/*.jsonl` files are `harness/fast_test.py` results (`shot`, `variant`, same timing fields) on
`harness/shots/eval1.json`. `framemd5.txt` files hold the md5 over `ffmpeg -f framemd5` of each mp4 (every video and
audio frame).

## 2026-10-29

| Folder | What it checks | Result |
|---|---|---|
| `quantizer/` | The faster rotation + INT8 quantizers (`_rot_quant_wide_kernel`, now the default) against the previous kernel: the 6 screening shots on `ts_h2` (`ts_h2_old.jsonl` / `ts_h2_new.jsonl`) and on `ts_s6` (`ts_s6_*.jsonl`), each preset on its own Spark, same server settings otherwise | `framemd5` identical for all 12 clips (`framemd5.txt`). Warm (first render excluded) median `ts_h2` 116.5 → 114.4 s (draft step ≈10.5 → ≈10.2 s), `ts_s6` 197.1 → 193.7 s (step ≈29.2 → ≈28.5 s) |

## 2026-10-27

| Folder | What it checks | Result |
|---|---|---|
| `image_check/` | The `rapid-h3/h3-comfyui:v0.42.1` image built from `docker/`, first render after a restart | final latents identical to the reference render (md5 of the video + audio latents: `ts_h2` `5ec5c86822ab`, `ts_s6` `6ae15cee0391`) |
| `decoder_level4/` | Video decoder level 4 (`RAPIDH3_VAE_FAST=4`, the new default) against level 3 | identical `framemd5` for `ts_h2` and `ts_s6` (`framemd5.txt`) |
| `attention_190/` | Attention config 190 (the new default in both presets) against config 194 | identical `framemd5` for `ts_h2` and `ts_s6`; `ts_s6` 198.9 s, steps 29.2–29.4 s. The `ts_h2` render was the first with config 190 on that server (its first draft step includes the kernel build) |
| `release_timing/` | Today's defaults (image built from `docker/`, attention config 190, decoder level 4): warm server, one preset block at a time, warm-up render dropped (`warm*.jsonl`), screening shots er11, er41, er43, ef31, ef41, ex51 | `ts_h2` 115.5–118.4 s (median 116.5 s), `ts_s6` 199.8–201.9 s (median 200.7 s). er11's `ts_h2` render is in `h2b.jsonl` (re-rendered after its own warm-up; the first attempt was interrupted) |
