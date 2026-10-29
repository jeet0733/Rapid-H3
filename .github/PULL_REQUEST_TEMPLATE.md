**What changes:**

**Checks** (tick what applies):
- [ ] Preset graphs unchanged: `json.dumps(workflow.build(...), sort_keys=True)` identical for every `PRESETS` entry, or the intended difference is described above
- [ ] Bit-identical output (latents / `ffmpeg -f framemd5`), or quality measured on the screening shots with `harness/eval_clips.py` and `harness/audio_quality.py`
- [ ] Timing from synced step times (`RapidH3StepTimer`), warm server
- [ ] `CHANGELOG.md` updated (and `docs/kernels.md` if a kernel changed)
