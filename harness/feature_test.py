#!/usr/bin/env python3
"""Render feature tests: each job = an eval shot + a preset + extra build options.

    python3 harness/feature_test.py JOBS.json --name feat1 [--only id1 id2]

JOBS.json: [{"id", "shot" (id in --shots, default harness/shots/eval1.json), "preset" (workflow.PRESETS key or a fast_test variant such as orig10),
"opts" (extra workflow.build kwargs, e.g. last_frame / audio_input / pin_latent / ref_images), "seed", "group" (optional),
"bust" (optional token: the prompt goes through a StringReplace that removes it again, so ComfyUI re-runs the text
encode and everything after it with the same text, for re-timing a shot)}].
Output and the per-job record (wall time, synced step split, settings) go to data/output/<name>/ like fast_test.py;
the clip is <name>/<id>_00001_.mp4 and its latent <name>/<id>_00001_.latent. exec_s: ComfyUI's own execution start ->
success span (history timestamps, no polling granularity). With a server started with RAPIDH3_TRACE (serve.sh), the
trace and the graph go to <name>/<id>_trace.jsonl and <id>_graph.json (harness/time_split.py reads them).
"""
import argparse, json, os, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render, workflow  # noqa: E402
from fast_test import RECIPE, VARIANTS, run, steps_since  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--name", required=True)
    ap.add_argument("--shots", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "shots", "eval1.json"))
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args()
    shots = {s["id"]: s for s in json.load(open(a.shots))}
    out_dir = os.path.join(render.DATA, "output", a.name)
    os.makedirs(out_dir, exist_ok=True)
    res_path = os.path.join(out_dir, "results.jsonl")
    done = set()
    if os.path.exists(res_path):
        done = {r["id"] for r in map(json.loads, open(res_path)) if r["status"] == "success"}
    for j in json.load(open(a.jobs)):
        if (a.only and j["id"] not in a.only) or j["id"] in done:
            continue
        s = shots[j["shot"]]
        # a preset, or a fast_test variant such as the stock reference orig10 (its draft scale is kept as a scale)
        base = workflow.PRESETS.get(j["preset"]) or {k: v for k, v in dict(RECIPE, **VARIANTS[j["preset"]]).items()
                                                     if k != "seed"}
        kw = dict(base, **j.get("opts", {}))
        g = workflow.build(s["prompt"], width=s["width"], height=s["height"], length=s["length"],
                           first_frame=s.get("first"), seed=j.get("seed", s.get("seed", 7)),
                           prefix=f"{a.name}/{j['id']}", **kw)
        if j.get("bust"):
            g["0b"] = {"class_type": "StringReplace",
                       "inputs": {"string": s["prompt"] + j["bust"], "find": j["bust"], "replace": ""}}
            for n in g.values():
                if n["inputs"].get("prompt") == s["prompt"]:
                    n["inputs"]["prompt"] = ["0b", 0]
        trace = os.path.join(render.DATA, "output", "rapidh3_trace.jsonl")
        if os.path.exists(trace):  # only this job's events
            os.remove(trace)
        hist, t0, t1 = run(g, a.timeout)
        status = (hist or {}).get("status", {}).get("status_str", "timeout")
        st = steps_since(t0)
        # per-linear errors (measure_every) / shadow divergence (shadow=True), if any -> <name>/<group>/p_<id>_*.json
        # (layer_errors.py / shadow_summary.py DATA/output/<name> <group>)
        grp = f"/data/output/{a.name}/{j.get('group', j['id'])}"
        subprocess.run(["docker", "exec", render.CONTAINER, "sh", "-c",
                        f"for k in stats shadow; do [ -f /data/output/rapidh3_$k.json ] && mkdir -p {grp} && "
                        f"mv /data/output/rapidh3_$k.json {grp}/p_{j['id']}_$k.json; done; true"])
        rec ={"id": j["id"], "shot": j["shot"], "preset": j["preset"], "opts": j.get("opts", {}), "status": status,
               "wall_s": round(t1 - t0, 1), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        ts = {m[0]: m[1].get("timestamp") for m in (hist or {}).get("status", {}).get("messages", [])
              if isinstance(m, list) and len(m) == 2 and isinstance(m[1], dict)}
        if ts.get("execution_start") and ts.get("execution_success"):
            rec.update(exec_s=round((ts["execution_success"] - ts["execution_start"]) / 1e3, 2),
                       exec_t0=ts["execution_start"] / 1e3, submit_t0=round(t0, 3))
        if os.path.exists(trace):
            os.replace(trace, os.path.join(out_dir, f"{j['id']}_trace.jsonl"))
            json.dump(g, open(os.path.join(out_dir, f"{j['id']}_graph.json"), "w"))
        if st:
            first_start = st[0]["t"] - st[0]["step_ms"] / 1e3
            rec.update(pre_s=round(first_start - t0, 1), post_s=round(t1 - st[-1]["t"], 1),
                       sampling_s=round(st[-1]["t"] - first_start, 1), steps_s=[round(r["step_ms"] / 1e3, 2) for r in st],
                       # idle time before each step after the first (two-stage: the largest is the draft -> refine gap)
                       gaps_s=[round(b["t"] - b["step_ms"] / 1e3 - a["t"], 2) for a, b in zip(st, st[1:])])
        if status != "success":
            rec["error"] = json.dumps((hist or {}).get("status", {}).get("messages", []))[-1500:]
        with open(res_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
