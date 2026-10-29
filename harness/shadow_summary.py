#!/usr/bin/env python3
"""Summarize RapidH3Shadow single-step divergence: per variant, mean / max over prompts and steps of
||v - v_bf16|| / ||v_bf16|| for the video and audio velocities.

    shadow_summary.py /data/output sh_ck sh_int8_rot ...
"""
import glob, json, os, statistics, sys

out, variants = sys.argv[1], sys.argv[2:]
print(f"{'variant':18s} {'n':>3s} {'video mean':>10s} {'video max':>9s} {'audio mean':>10s} {'audio max':>9s}  per-step video mean")
for v in variants:
    steps = [json.load(open(f))["steps"] for f in sorted(glob.glob(os.path.join(out, v, "p*_shadow.json")))]
    if not steps:
        print(f"{v:18s} no data")
        continue
    vid = [s["video"] for run in steps for s in run]
    aud = [s["audio"] for run in steps for s in run]
    per_step = [statistics.mean(run[i]["video"] for run in steps) for i in range(min(map(len, steps)))]
    print(f"{v:18s} {len(steps):3d} {100*statistics.mean(vid):9.2f}% {100*max(vid):8.2f}% "
          f"{100*statistics.mean(aud):9.2f}% {100*max(aud):8.2f}%  " + " ".join(f"{100*x:.1f}" for x in per_step))
