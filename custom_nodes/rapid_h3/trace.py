"""Opt-in wall-time trace of one ComfyUI prompt: RAPIDH3_TRACE=/data/output/rapidh3_trace.jsonl.

Installed at import when RAPIDH3_TRACE is set (serve.sh passes it through); off by default, no effect on outputs.
Appends JSON lines (wall clock time.time(), the same clock as RapidH3StepTimer and the harness):

  {"ev": "node", "t", "node", "prompt_id"}    ComfyUI's "executing" event (node None = prompt done), after a
                                             torch.cuda.synchronize(), so a node's span ends when its GPU work does
  {"ev": "call", "name", "t0", "t1", "depth"} a wrapped function: model management (load_models_gpu, free_memory,
                                             soft_empty_cache, cleanup_models_gc), gc.collect, ModelPatcher
                                             (load / partially_load / partially_unload / unpatch_model), VAE
                                             encode / decode, CLIP tokenize / encode, the sampler (CFGGuider.sample);
                                             depth = nesting level within wrapped calls on the thread
  {"ev": "log", "t", "msg"}                  ComfyUI's model load / unload log lines
  {"ev": "call", "name": "gc.genN", ...}     a Python garbage collection (gc.callbacks, automatic ones included)
                                             taking > 2 ms or of generation 2; depth 99
  {"ev": "executed", "t", "node"}            an output node's "executed" event

harness/time_split.py turns it (plus RapidH3StepTimer's steps) into a per-node / per-cause table.
"""
import functools
import gc
import json
import logging
import os
import threading
import time

import torch

_PATH = os.environ.get("RAPIDH3_TRACE")
_LOCK = threading.Lock()
_TLS = threading.local()


def _emit(rec):
    with _LOCK, open(_PATH, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _wrap(owner, attr, name, sync=False):
    fn = getattr(owner, attr, None)
    if fn is None or getattr(fn, "_rapidh3_traced", False):
        return
    is_static = isinstance(owner, type) and isinstance(owner.__dict__.get(attr), staticmethod)

    @functools.wraps(fn)
    def w(*args, **kwargs):
        depth = getattr(_TLS, "depth", 0)
        _TLS.depth = depth + 1
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        try:
            return fn(*args, **kwargs)
        finally:
            if sync and torch.cuda.is_available():
                torch.cuda.synchronize()
            _TLS.depth = depth
            _emit({"ev": "call", "name": name, "t0": round(t0, 4), "t1": round(time.time(), 4), "depth": depth,
                   "thread": threading.current_thread().name})

    w._rapidh3_traced = True
    setattr(owner, attr, staticmethod(w) if is_static else w)


class _LogTap(logging.Handler):
    KEYS = ("Requested to load", "loaded partially", "loaded completely", "Unloaded partially", "Unloading",
            "memory leak")

    def emit(self, record):
        msg = record.getMessage()
        if any(k in msg for k in self.KEYS):
            _emit({"ev": "log", "t": round(time.time(), 4), "msg": msg[:200]})


def _profile_clip(cls):
    """RAPIDH3_TRACE_CLIP=1: torch.profiler over every text encode, kernel / op summary to <RAPIDH3_TRACE>.clip.txt."""
    fn = cls.encode_from_tokens_scheduled

    @functools.wraps(fn)
    def w(*args, **kwargs):
        from torch.profiler import ProfilerActivity, profile
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            out = fn(*args, **kwargs)
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        with open(_PATH + ".clip.txt", "a") as f:
            f.write(f"encode wall {wall:.3f} s (profiled)\n")
            f.write(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30, max_name_column_width=90))
            f.write(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20, max_name_column_width=90))
        return out

    cls.encode_from_tokens_scheduled = w


def install():
    if not _PATH:
        return False
    import comfy.model_management as mm
    import comfy.model_patcher as mp
    import comfy.samplers
    import comfy.sd
    import server

    for f in ("load_models_gpu", "free_memory", "soft_empty_cache", "cleanup_models_gc", "unload_all_models"):
        _wrap(mm, f, f"mm.{f}")
    _wrap(gc, "collect", "gc.collect")
    for cls in (getattr(mp, n) for n in dir(mp)):
        if isinstance(cls, type) and issubclass(cls, getattr(mp, "ModelPatcher", object)):
            for f in ("load", "partially_load", "partially_unload", "unpatch_model"):
                if f in cls.__dict__:
                    _wrap(cls, f, f"{cls.__name__}.{f}")
    for f in ("encode", "decode", "decode_tiled"):
        _wrap(comfy.sd.VAE, f, f"VAE.{f}", sync=True)
    for f in ("tokenize", "encode_from_tokens_scheduled"):
        _wrap(comfy.sd.CLIP, f, f"CLIP.{f}", sync=True)
    _wrap(comfy.samplers.CFGGuider, "sample", "CFGGuider.sample", sync=True)
    _wrap(comfy.samplers.CFGGuider, "inner_sample", "CFGGuider.inner_sample", sync=True)
    import comfy.model_base
    import comfy.sampler_helpers
    for f in ("process_latent_in", "process_latent_out"):
        _wrap(comfy.model_base.BaseModel, f, f"BaseModel.{f}", sync=True)
    for f in ("prepare_sampling", "cleanup_models", "cleanup_additional_models"):
        _wrap(comfy.sampler_helpers, f, f"sampler_helpers.{f}")

    gc_t0 = {}

    def gc_cb(phase, info):  # Python's automatic (and explicit) collections, by generation
        if phase == "start":
            gc_t0["t"] = time.time()
        elif "t" in gc_t0:
            t0 = gc_t0.pop("t")
            if time.time() - t0 > 0.002 or info.get("generation") == 2:
                _emit({"ev": "call", "name": f"gc.gen{info.get('generation')}", "t0": round(t0, 4),
                       "t1": round(time.time(), 4), "depth": 99, "thread": threading.current_thread().name,
                       "collected": info.get("collected")})

    gc.callbacks.append(gc_cb)
    if os.environ.get("RAPIDH3_TRACE_CLIP"):
        _profile_clip(comfy.sd.CLIP)
    logging.getLogger().addHandler(_LogTap(level=logging.INFO))
    srv = getattr(server.PromptServer, "instance", None)
    if srv is None:
        print("[rapid-h3] trace: no PromptServer instance, node events off", flush=True)
        return True
    orig_send = srv.send_sync

    def send_sync(event, data, sid=None):
        if event == "executed" and isinstance(data, dict):
            _emit({"ev": "executed", "t": round(time.time(), 4), "node": data.get("node")})
        if event == "executing" and isinstance(data, dict):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _emit({"ev": "node", "t": round(time.time(), 4), "node": data.get("node"),
                   "prompt_id": data.get("prompt_id")})
        return orig_send(event, data, sid)

    srv.send_sync = send_sync
    print(f"[rapid-h3] trace on -> {_PATH}", flush=True)
    return True
