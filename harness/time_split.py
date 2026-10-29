#!/usr/bin/env python3
"""Where every second of one render goes, from a RAPIDH3_TRACE trace.

    python3 harness/time_split.py OUT_DIR JOB_ID [--steps data/output/rapidh3_steps.jsonl] [--json]

OUT_DIR is a feature_test.py output folder of a run on a server started with RAPIDH3_TRACE=/data/output/rapidh3_trace.jsonl
(serve.sh); it holds <id>_trace.jsonl, <id>_graph.json and results.jsonl. Prints, in execution order, every node's
wall span (synced at node boundaries) and what ran inside it: denoising steps (RapidH3StepTimer), the time before the
first / after the last step and between steps, and the wrapped calls (model loads / unloads, soft_empty_cache,
gc.collect, VAE / CLIP calls). The spans tile ComfyUI's execution from its first node to its end, so their sum
is exec_s; submit -> first node and last node -> history are listed separately.
"""
import argparse, collections, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render  # noqa: E402


def load(out_dir, job, steps_path):
    rec = [r for r in map(json.loads, open(os.path.join(out_dir, "results.jsonl"))) if r["id"] == job][-1]
    graph = json.load(open(os.path.join(out_dir, f"{job}_graph.json")))
    ev = [json.loads(l) for l in open(os.path.join(out_dir, f"{job}_trace.jsonl"))]
    nodes = [e for e in ev if e["ev"] == "node"]
    t_lo, t_hi = nodes[0]["t"], nodes[-1]["t"]
    steps = []
    for l in open(steps_path):
        r = json.loads(l)
        if t_lo <= r["t"] <= t_hi + 0.01:
            steps.append((r["t"] - r["step_ms"] / 1e3, r["t"]))
    return rec, graph, ev, nodes, steps


def split(rec, graph, ev, nodes, steps):
    calls = [e for e in ev if e["ev"] == "call"]
    logs = [e for e in ev if e["ev"] == "log"]
    rows = []
    for a, b in zip(nodes, nodes[1:]):
        t0, t1 = a["t"], b["t"]
        inside = lambda x0, x1: x0 >= t0 - 1e-3 and x1 <= t1 + 1e-3  # noqa: E731
        st = [s for s in steps if inside(*s)]
        row = {"node": a["node"], "class_type": graph.get(a["node"], {}).get("class_type", "?"), "s": round(t1 - t0, 3)}
        if st:
            row["steps"] = [round(e - s, 2) for s, e in st]
            row["before_first_step"] = round(st[0][0] - t0, 3)
            row["between_steps"] = [round(n[0] - p[1], 3) for p, n in zip(st, st[1:])]
            row["after_last_step"] = round(t1 - st[-1][1], 3)
        # depth-0 calls (outermost wrapped call on the main thread), and every call by name for detail
        by = collections.defaultdict(lambda: [0, 0.0])
        top = collections.defaultdict(lambda: [0, 0.0])
        for c in calls:
            if inside(c["t0"], c["t1"]):
                d = by[c["name"]]
                d[0] += 1
                d[1] += c["t1"] - c["t0"]
                if c["depth"] == 0:
                    d = top[c["name"]]
                    d[0] += 1
                    d[1] += c["t1"] - c["t0"]
        row["top_calls"] = {k: [n, round(v, 3)] for k, (n, v) in sorted(top.items(), key=lambda kv: -kv[1][1])}
        row["all_calls"] = {k: [n, round(v, 3)] for k, (n, v) in sorted(by.items(), key=lambda kv: -kv[1][1])}
        row["logs"] = [l["msg"][:90] for l in logs if t0 <= l["t"] <= t1]
        rows.append(row)
    head = {"exec_s": rec.get("exec_s"), "wall_s": rec.get("wall_s"),
            "nodes_s": round(nodes[-1]["t"] - nodes[0]["t"], 2),
            "exec_start_to_first_node": round(nodes[0]["t"] - rec["exec_t0"], 3) if rec.get("exec_t0") else None,
            "submit_to_exec_start": round(rec["exec_t0"] - rec["submit_t0"], 3) if rec.get("exec_t0") else None,
            "steps": len(steps), "steps_sum": round(sum(e - s for s, e in steps), 2)}
    return head, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("job")
    ap.add_argument("--steps", default=render.STEPS)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--min", type=float, default=0.02, help="hide nodes shorter than this (s)")
    a = ap.parse_args()
    head, rows = split(*load(a.out_dir, a.job, a.steps))
    if a.json:
        print(json.dumps({"head": head, "nodes": rows}, indent=1))
        return
    print(json.dumps(head))
    for r in rows:
        if r["s"] < a.min:
            continue
        print(f"{r['s']:8.2f}  {r['node']:>5}  {r['class_type']}")
        if "steps" in r:
            print(f"{'':16}steps {r['steps']} sum {sum(r['steps']):.2f}; before first {r['before_first_step']:.2f}, "
                  f"between {r['between_steps']}, after last {r['after_last_step']:.2f}")
        for k, (n, v) in r["top_calls"].items():
            if v >= 0.01:
                print(f"{'':16}{v:7.2f}  {k} x{n}")
        nested = {k: nv for k, nv in r["all_calls"].items() if k not in r["top_calls"] or nv != r["top_calls"][k]}
        for k, (n, v) in nested.items():
            if v >= 0.01:
                print(f"{'':18}(all) {v:7.2f}  {k} x{n}")
        for l in r["logs"]:
            print(f"{'':16}log: {l}")


if __name__ == "__main__":
    main()
