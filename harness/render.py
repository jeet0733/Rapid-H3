#!/usr/bin/env python3
"""Render the eval prompts for one variant through the rapid-h3-comfy server.

Runs on the Spark host (stdlib only).  For each prompt: build the graph, submit, wait,
then record wall time, synced per-step times (RapidH3StepTimer), tqdm s/it and the output files.
`step_s` (median of steps 2..N) is the per-step number to quote; tqdm's s/it under-reports.

    render.py --variant ref --prompts prompts.txt [--set width=1344 height=768 quant=fp8_rowwise]
"""
import argparse, json, os, re, subprocess, sys, time, urllib.request, uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import workflow  # noqa: E402

# RAPIDH3_HOST / RAPIDH3_NAME / RAPIDH3_DIR point the harness at a second server copy (default: the main one)
HOSTS = [os.environ.get("RAPIDH3_HOST", "127.0.0.1:8189")]
CONTAINER = os.environ.get("RAPIDH3_NAME", "rapid-h3-comfy")
DATA = os.path.join(os.path.expanduser(os.environ.get("RAPIDH3_DIR", "~/rapid-h3")), "data")


def api(path, payload=None, timeout=60, host=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(f"http://{host or HOSTS[0]}{path}", data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def coerce(v):
    for t in (int, float):
        try:
            return t(v)
        except ValueError:
            pass
    return {"true": True, "false": False}.get(v.lower(), v)


def sampler_rate(since):
    log = subprocess.run(["docker", "logs", "--since", since, CONTAINER],
                         capture_output=True, text=True).stderr
    its = re.findall(r"(\d+)/(\d+) \[([\d:]+)<[^,]*,\s*([\d.]+)(s/it|it/s)\]", log)
    if not its:
        return None
    n, total, elapsed, rate, unit = its[-1]
    rate = float(rate)
    return {"steps": int(total), "s_per_it": rate if unit == "s/it" else 1 / rate}


STEPS = os.path.join(DATA, "output", "rapidh3_steps.jsonl")


def step_times(t0):
    """Synced step times (s) written by RapidH3StepTimer since t0."""
    try:
        rows = [json.loads(l) for l in open(STEPS)]
    except OSError:
        return None
    ms = [r["step_ms"] / 1e3 for r in rows if r["t"] >= t0]
    if not ms:
        return None
    rest = sorted(ms[1:]) or ms
    return {"steps": len(ms), "first_s": round(ms[0], 2), "step_s": round(rest[len(rest) // 2], 2),
            "sampling_s": round(sum(ms), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--only", type=int, nargs="*", help="prompt indices (1-based)")
    ap.add_argument("--set", nargs="*", default=[], help="workflow.build kwargs as key=value")
    ap.add_argument("--timeout", type=int, default=7200)
    a = ap.parse_args()

    kw = {k: coerce(v) for k, v in (s.split("=", 1) for s in a.set)}
    prompts = [l.strip() for l in open(a.prompts) if l.strip() and not l.startswith("#")]
    results = os.path.join(DATA, "results.jsonl")
    for i, prompt in enumerate(prompts, 1):
        if a.only and i not in a.only:
            continue
        tag = f"{a.variant}/p{i:02d}"
        graph = workflow.build(prompt, prefix=tag, **kw)
        since = time.strftime("%Y-%m-%dT%H:%M:%S")
        t0 = time.time()
        pids = [api("/prompt", {"prompt": graph, "client_id": str(uuid.uuid4())}, host=hh)["prompt_id"]
                for hh in HOSTS]
        pid = pids[0]
        print(f"[{tag}] submitted {pids}", flush=True)
        hist = None
        done = {}
        while time.time() - t0 < a.timeout and len(done) < len(HOSTS):
            time.sleep(3)
            for hh, p in zip(HOSTS, pids):
                if hh in done:
                    continue
                h = api(f"/history/{p}", host=hh)
                if p in h and (h[p].get("status", {}).get("completed")
                               or h[p].get("status", {}).get("status_str") == "error"):
                    done[hh] = h[p]
        hist = done.get(HOSTS[0])
        wall = time.time() - t0
        status = (hist or {}).get("status", {}).get("status_str", "timeout")
        files = []
        for out in (hist or {}).get("outputs", {}).values():
            for v in out.values():
                if isinstance(v, list):
                    files += [os.path.join(x.get("subfolder", ""), x["filename"])
                              for x in v if isinstance(x, dict) and "filename" in x]
        # per-layer error stats written by RapidH3LinearQuant (if measuring) -> next to the render
        subprocess.run(["docker", "exec", CONTAINER, "sh", "-c",
                        f"[ -f /data/output/rapidh3_stats.json ] && mv /data/output/rapidh3_stats.json "
                        f"/data/output/{a.variant}/p{i:02d}_stats.json; "
                        f"[ -f /data/output/rapidh3_shadow.json ] && mv /data/output/rapidh3_shadow.json "
                        f"/data/output/{a.variant}/p{i:02d}_shadow.json; true"])
        rec = {"variant": a.variant, "prompt_idx": i, "status": status, "wall_s": round(wall, 1),
               "step": step_times(t0), "sampler": sampler_rate(since), "files": files, "settings": kw,
               "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        if status != "success":
            msgs = (hist or {}).get("status", {}).get("messages", [])
            rec["error"] = json.dumps(msgs)[-1500:]
        with open(results, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
        if status != "success":
            sys.exit(1)


if __name__ == "__main__":
    main()
