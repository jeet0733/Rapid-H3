"""Per-node wall time of one render, from ComfyUI's websocket "executing" events.

Runs INSIDE the rapid-h3-comfy container (needs aiohttp, which ComfyUI ships):

    docker exec -i rapid-h3-comfy python - < harness/node_times.py GRAPH.json     # graph path inside /data

ComfyUI runs nodes one after another, so a node's time is the gap until the next node starts (the GPU is synced at
the node boundaries that matter here: samplers via RapidH3StepTimer, VAE decode returns CPU tensors). Prints JSON:
[{"node", "class_type", "s"}] in execution order plus the total.
"""
import asyncio, json, sys, time, uuid

import aiohttp

URL = "http://127.0.0.1:8188"


async def main(path):
    graph = json.load(open(path))
    cid = str(uuid.uuid4())
    events = []
    async with aiohttp.ClientSession() as s:
        async with s.ws_connect(f"{URL}/ws?clientId={cid}") as ws:
            async with s.post(f"{URL}/prompt", json={"prompt": graph, "client_id": cid}) as r:
                pid = (await r.json())["prompt_id"]
            t0 = time.time()
            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                m = json.loads(msg.data)
                d = m.get("data", {})
                if d.get("prompt_id") != pid:
                    continue
                if m["type"] == "executing":
                    events.append((time.time(), d.get("node")))
                    if d.get("node") is None:
                        break
                elif m["type"] == "execution_error":
                    print(json.dumps(d)[-2000:])
                    return
    out = []
    for (t, n), (t2, _) in zip(events, events[1:]):
        if n is not None:
            out.append({"node": n, "class_type": graph[n]["class_type"], "s": round(t2 - t, 2)})
    print(json.dumps({"total_s": round(events[-1][0] - t0, 1), "nodes": out}, indent=1))


asyncio.run(main(sys.argv[1]))
