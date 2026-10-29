#!/usr/bin/env python3
"""Summarize per-linear relative output error ||y_q - y_bf16|| / ||y_bf16|| recorded by
RapidH3LinearQuant(measure_every>0) across all prompts of each variant.

    layer_errors.py /data/output fp8_rowwise fp8_tensor int8_rowwise
"""
import collections, glob, json, os, re, statistics, sys

out, variants = sys.argv[1], sys.argv[2:]
NAME = re.compile(r"blocks\.(\d+)\.\w+\.(\w+)$")
for v in variants:
    per_class = collections.defaultdict(list)
    per_block = collections.defaultdict(list)
    worst = []
    for path in glob.glob(os.path.join(out, v, "p*_stats.json")):
        for name, errs in json.load(open(path))["errors"].items():
            blk, cls = NAME.search(name).groups()
            per_class[cls] += errs
            per_block[int(blk)] += errs
            worst.append((statistics.mean(errs), name))
    if not per_class:
        print(f"{v}: no stats")
        continue
    allv = [e for es in per_class.values() for e in es]
    print(f"\n{v}: {len(allv)} samples, mean {100*statistics.mean(allv):.2f}%  "
          f"median {100*statistics.median(allv):.2f}%")
    for cls in ("qkv_proj", "out_proj", "fc1", "fc2"):
        es = per_class.get(cls, [])
        if es:
            print(f"  {cls:9s} mean {100*statistics.mean(es):.2f}%  p95 {100*sorted(es)[int(.95*len(es))]:.2f}%")
    bands = [(0, 9), (10, 29), (30, 49)]
    print("  by block: " + "  ".join(
        f"{a}-{b}: {100*statistics.mean([e for k in range(a, b+1) for e in per_block.get(k, [])] or [0]):.2f}%"
        for a, b in bands))
    print("  worst: " + ", ".join(f"{n.replace('blocks.', 'b')} {100*e:.1f}%" for e, n in sorted(worst, reverse=True)[:6]))
