#!/usr/bin/env bash
# Which linear class drives int8_rot's single-step divergence?  Shadow runs on prompt 2 only.
set -u
cd ~/rapid-h3
R="python3 harness/render.py --prompts harness/prompts.txt --only 2"
run() { echo "=== $1 $(date +%T)"; $R --variant "$@" || echo "!!! $1 failed"; }
run ab_qkv   --set shadow=true quant=int8_rot quant_targets=qkv_proj
run ab_out   --set shadow=true quant=int8_rot quant_targets=out_proj
run ab_fc1   --set shadow=true quant=int8_rot quant_targets=fc1
run ab_fc2   --set shadow=true quant=int8_rot quant_targets=fc2
run ab_noout --set shadow=true quant=int8_rot quant_targets=qkv_proj,fc1,fc2
echo "=== done $(date +%T)"
