#!/usr/bin/env python3
"""Compare fast variants of the fast1 test against base10 (the 10-step quality reference).

fast1 was an early fast_test.py run on first-frame shots (ids s05, s07, ...) whose shot file and first-frame images
(<in-dir>/fast1_<shot>.png) are not in the repo; point --out-dir / --in-dir / --ref-prefix at your own run.

Runs on the Spark host, CPU only, in the metrics venv (config/venv_metrics.requirements.txt: InsightFace
buffalo_l, av, opencv, scikit-image):

  cd ~/rapid-h3 && OMP_NUM_THREADS=4 nice -n 19 venv_metrics/bin/python harness/fast_compare.py
      [--out-dir data/output/fast1] [--in-dir data/input] [--cmp-dir data/output/fast1/cmp] [--shots s05 s07]

For every shot that has a base10 clip it writes into --cmp-dir:
  <shot>_sheet.jpg   rows = variants (base10 first), columns = frames at 0/25/50/75/100 %
  <shot>.json        per-variant metrics (see METRICS below)
  summary.md / summary.json   all shots x variants in one table
Per-clip raw measurements are cached in --cmp-dir/cache keyed by file size+mtime, so re-running
only processes new clips (idempotent; safe to run again as more clips land).

METRICS (faces = largest face per sampled frame, every --every frames, near-frontal |yaw| <= 45 only)
  id_med / id_min   cosine similarity of the face embedding to the first-frame image's face
  fsize_med/max     face bbox height / frame height, as a ratio vs the first frame (>1.3 = closer framing)
  sharp_face        Laplacian variance of the face crop (resized to 128x128), median, ratio vs base10
  sharp_frame       Laplacian variance of the whole grayscale frame, mean, ratio vs base10
  motion            mean abs difference of consecutive frames (all frames, 1/4 res), ratio vs base10
  comp_ssim         SSIM of 64x112 grayscale thumbnails vs base10 at the same frame, mean over samples
  cut               adjacent sampled-thumbnail SSIM drops below median - --cut-drop (min reported as adj_min)
"""
import argparse, glob, json, os, re, sys

import av
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as ssim

REF = "base10"  # the reference variant every other is compared with (--ref)
ORDER = ["base10", "base8", "base7", "base6", "ts_a", "ts_b", "ts_c", "ts_d"]
CACHE_VER = 2
THUMB = (64, 112)  # w, h (portrait)
_APP = None
NO_FACE = False  # --no-face: skip InsightFace (identity / face-size / face-sharpness columns stay empty)


def app():
    global _APP
    if _APP is None:
        from insightface.app import FaceAnalysis
        _APP = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        _APP.prepare(ctx_id=-1, det_size=(640, 640))
    return _APP


def yaw_of(f):
    if getattr(f, "pose", None) is not None:
        return float(f.pose[1])
    k = f.kps
    eye_mid = (k[0] + k[1]) / 2
    return float((k[2][0] - eye_mid[0]) / (np.linalg.norm(k[1] - k[0]) + 1e-6) * 90)


def largest(fs):
    return max(fs, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1])) if fs else None


def lapvar(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def face_crop(bgr, f, size=128):
    x0, y0, x1, y1 = f.bbox
    cx, cy, s = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) / 2
    H, W = bgr.shape[:2]
    a, b = int(max(0, cx - s)), int(max(0, cy - s))
    c, d = int(min(W, cx + s)), int(min(H, cy + s))
    if c - a < 8 or d - b < 8:
        return None
    return cv2.resize(cv2.cvtColor(bgr[b:d, a:c], cv2.COLOR_BGR2GRAY), (size, size), interpolation=cv2.INTER_AREA)


def face_info(bgr, max_yaw):
    if NO_FACE:
        return {"n": None}, None
    fs = app().get(bgr)
    f = largest(fs)
    r = {"n": len(fs)}
    if f is None:
        return r, None
    y = yaw_of(f)
    r.update(yaw=round(y, 1), h=float((f.bbox[3] - f.bbox[1]) / bgr.shape[0]), frontal=abs(y) <= max_yaw)
    crop = face_crop(bgr, f)
    if crop is not None:
        r["lap_face"] = lapvar(crop)
    return r, f.normed_embedding


def analyze_clip(mp4, every, max_yaw, pct):
    """Decode once; return per-sample rows, embeddings, thumbnails, motion, and sheet frames."""
    c = av.open(mp4)
    rows, embs, thumbs, sheet = [], [], [], {}
    prev, diffs = None, []
    frames = []
    for i, fr in enumerate(c.decode(video=0)):
        bgr = fr.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (gray.shape[1] // 4, gray.shape[0] // 4), interpolation=cv2.INTER_AREA).astype(np.float32)
        if prev is not None:
            diffs.append(float(np.abs(small - prev).mean()))
        prev = small
        frames.append(cv2.resize(bgr, (200, int(round(200 * bgr.shape[0] / bgr.shape[1]))), interpolation=cv2.INTER_AREA))
        if i % every:
            continue
        r, e = face_info(bgr, max_yaw)
        r.update(frame=i, lap_frame=lapvar(gray))
        rows.append(r)
        embs.append(e if e is not None else np.zeros(512, np.float32))
        thumbs.append(cv2.resize(gray, THUMB, interpolation=cv2.INTER_AREA))
    c.close()
    n = len(frames)
    for p in pct:
        sheet[p] = frames[min(n - 1, int(round(p / 100 * (n - 1))))]
    return dict(rows=rows, embs=np.array(embs), thumbs=np.array(thumbs), diffs=np.array(diffs), n_frames=n,
                sheet=np.array([sheet[p] for p in pct]), shape=frames[0].shape if frames else None)


def load_cached(mp4, cache_dir, every, max_yaw, pct):
    st = os.stat(mp4)
    key = f"v{CACHE_VER}_{st.st_size}_{int(st.st_mtime)}_{every}_{max_yaw}_{'nf' if NO_FACE else 'f'}"
    base = os.path.join(cache_dir, os.path.basename(mp4)[:-4])
    if os.path.exists(base + ".json") and os.path.exists(base + ".npz"):
        meta = json.load(open(base + ".json"))
        if meta.get("key") == key:
            z = np.load(base + ".npz")
            return dict(rows=meta["rows"], n_frames=meta["n_frames"], embs=z["embs"], thumbs=z["thumbs"],
                        diffs=z["diffs"], sheet=z["sheet"]), False
    d = analyze_clip(mp4, every, max_yaw, pct)
    np.savez_compressed(base + ".npz", embs=d["embs"], thumbs=d["thumbs"], diffs=d["diffs"], sheet=d["sheet"])
    with open(base + ".json.tmp", "w") as fh:
        json.dump({"key": key, "rows": d["rows"], "n_frames": d["n_frames"]}, fh)
    os.replace(base + ".json.tmp", base + ".json")
    return d, True


def find_clips(out_dir):
    """{shot: {variant: path}}, newest counter per (shot, variant)."""
    pat = re.compile(r"^([A-Za-z0-9]+)_(.+)_(\d{5})_\.mp4$")
    best = {}
    for p in glob.glob(os.path.join(out_dir, "*.mp4")):
        m = pat.match(os.path.basename(p))
        if not m:
            continue
        k = (m.group(1), m.group(2))
        if k not in best or int(m.group(3)) > best[k][0]:
            best[k] = (int(m.group(3)), p)
    out = {}
    for (s, v), (_, p) in best.items():
        out.setdefault(s, {})[v] = p
    return out


def load_results(out_dir):
    res = {}
    p = os.path.join(out_dir, "results.jsonl")
    if os.path.exists(p):
        for line in open(p):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("status") == "success":
                res[(r["shot"], r["variant"])] = r
    return res


def first_frame(mp4):
    c = av.open(mp4)
    try:
        return next(c.decode(video=0)).to_ndarray(format="bgr24")
    finally:
        c.close()


def ref_embedding(png, max_yaw, fallback_mp4=None):
    """Face of the first-frame image; for text-only shots (no image) the first frame of the base10 clip."""
    if NO_FACE:
        return None, None
    bgr = cv2.imread(png) if os.path.exists(png) else first_frame(fallback_mp4) if fallback_mp4 else None
    if bgr is None:
        return None, None
    fs = app().get(bgr)
    f = largest(fs)
    if f is None:
        return None, None
    return f.normed_embedding, float((f.bbox[3] - f.bbox[1]) / bgr.shape[0])


def rnd(x, k=3):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), k)


def clip_metrics(d, ref_emb, ref_h, base, cut_drop):
    rows = d["rows"]
    fr = [(i, r) for i, r in enumerate(rows) if r.get("frontal")]
    sims = [float(np.dot(d["embs"][i], ref_emb)) for i, _ in fr] if ref_emb is not None else []
    hs = [r["h"] for r in rows if "h" in r]
    lf = [r["lap_face"] for _, r in fr if "lap_face" in r]
    lfr = [r["lap_frame"] for r in rows]
    th = d["thumbs"].astype(np.float32)
    adj = [ssim(th[k], th[k + 1], data_range=255) for k in range(len(th) - 1)]
    adj_med = float(np.median(adj)) if adj else None
    cut_at = [rows[k + 1]["frame"] for k, a in enumerate(adj) if adj_med is not None and a < adj_med - cut_drop]
    m = {
        "frames": d["n_frames"],
        "sampled": len(rows),
        "faces": len(hs),
        "frontal": len(fr),
        "max_people": max((r["n"] or 0 for r in rows), default=0),
        "id_med": rnd(np.median(sims)) if sims else None,
        "id_min": rnd(min(sims)) if sims else None,
        "id_min_frame": fr[int(np.argmin(sims))][1]["frame"] if sims else None,
        "fsize_med": rnd(np.median(hs) / ref_h, 2) if hs and ref_h else None,
        "fsize_max": rnd(max(hs) / ref_h, 2) if hs and ref_h else None,
        "_lap_face": float(np.median(lf)) if lf else None,
        "_lap_frame": float(np.mean(lfr)) if lfr else None,
        "_motion": float(d["diffs"].mean()) if len(d["diffs"]) else None,
        "adj_med": rnd(adj_med),
        "adj_min": rnd(min(adj)) if adj else None,
        "cut": bool(cut_at),
        "cut_frames": cut_at,
    }
    if base is not None:
        bt = base["thumbs"].astype(np.float32)
        k = min(len(bt), len(th))
        m["comp_ssim"] = rnd(np.mean([ssim(th[j], bt[j], data_range=255) for j in range(k)])) if k else None
    return m


def add_ratios(ms):
    b = dict(ms.get(REF, {}))  # copy: the reference's own keys are popped below
    for v, m in ms.items():
        for key, name in (("_lap_face", "sharp_face"), ("_lap_frame", "sharp_frame"), ("_motion", "motion")):
            m[name] = rnd(m[key] / b[key], 2) if m.get(key) is not None and b.get(key) else None
        m["lap_face_abs"] = rnd(m.pop("_lap_face"), 1)
        m["lap_frame_abs"] = rnd(m.pop("_lap_frame"), 1)
        m["motion_abs"] = rnd(m.pop("_motion"), 2)


def font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def contact_sheet(shot, variants, data, ms, res, pct, path):
    tile = data[variants[0]]["sheet"][0]
    th, tw = tile.shape[:2]
    lw, hh = 190, 26
    W, H = lw + tw * len(pct), hh + th * len(variants)
    img = Image.new("RGB", (W, H), (20, 20, 20))
    dr = ImageDraw.Draw(img)
    f1, f2 = font(17), font(13)
    for j, p in enumerate(pct):
        dr.text((lw + j * tw + 6, 5), f"{p}%", fill=(230, 230, 230), font=f2)
    dr.text((6, 5), shot, fill=(255, 220, 120), font=f1)
    for i, v in enumerate(variants):
        y = hh + i * th
        for j in range(len(pct)):
            fr = data[v]["sheet"][j]
            if fr.shape[:2] != (th, tw):
                fr = cv2.resize(fr, (tw, th))
            img.paste(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)), (lw + j * tw, y))
        r, m = res.get((shot, v), {}), ms[v]
        lines = [f"wall {r['wall_s']:.0f}s" if r.get("wall_s") else "wall ?"]
        if r.get("settings"):
            s = r["settings"]
            lines.append(f"steps {s.get('steps', '?')}" + (f" d{s['draft']}" if s.get("draft") else ""))
        if not NO_FACE:
            lines += [f"id {m['id_med']} / min {m['id_min']}", f"face x{m['fsize_med']} (max {m['fsize_max']})"]
        if v != REF:
            lines += [f"sharp F{m['sharp_face']} W{m['sharp_frame']}" if not NO_FACE else f"sharp W{m['sharp_frame']}", f"motion {m['motion']}", f"comp {m.get('comp_ssim')}"]
        if m["cut"]:
            cf = m["cut_frames"]
            lines.append(f"CUT? @{cf[0]}" + (f" (+{len(cf) - 1} more)" if len(cf) > 1 else ""))
        if m["max_people"] > 1:
            lines.append(f"people<= {m['max_people']}")
        dr.text((8, y + 8), v, fill=(255, 220, 120), font=f1)
        for k, t in enumerate(lines):
            dr.text((8, y + 34 + 17 * k), t, fill=(220, 220, 220), font=f2)
        dr.line([(0, y), (W, y)], fill=(90, 90, 90), width=2)
    img.save(path, quality=88)


COLS = ["wall_s", "id_med", "id_min", "fsize_med", "fsize_max", "sharp_face", "sharp_frame", "motion", "comp_ssim",
        "adj_min", "cut", "max_people", "frontal"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/output/fast1")
    ap.add_argument("--in-dir", default="data/input")
    ap.add_argument("--cmp-dir", default=None)
    ap.add_argument("--shots", nargs="*")
    ap.add_argument("--ref", default="base10", help="reference variant (default base10)")
    ap.add_argument("--ref-prefix", default="fast1_", help="first-frame images are <in-dir>/<prefix><shot>.png")
    ap.add_argument("--every", type=int, default=6)
    ap.add_argument("--max-yaw", type=float, default=45.0)
    ap.add_argument("--cut-drop", type=float, default=0.25)
    ap.add_argument("--no-face", action="store_true", help="skip InsightFace metrics")
    a = ap.parse_args()
    global NO_FACE, REF
    NO_FACE, REF = a.no_face, a.ref
    cmp_dir = a.cmp_dir or os.path.join(a.out_dir, "cmp")
    cache = os.path.join(cmp_dir, "cache")
    os.makedirs(cache, exist_ok=True)
    pct = [0, 25, 50, 75, 100]
    clips, res = find_clips(a.out_dir), load_results(a.out_dir)
    summary = {}
    for shot in sorted(clips):
        if a.shots and shot not in a.shots:
            continue
        vs = clips[shot]
        if REF not in vs:
            print(f"{shot}: no {REF} yet, skipped", flush=True)
            continue
        variants = [REF] + [v for v in ORDER if v in vs and v != REF] + sorted(v for v in vs if v not in ORDER and v != REF)
        ref_emb, ref_h = ref_embedding(os.path.join(a.in_dir, f"{a.ref_prefix}{shot}.png"), a.max_yaw, vs[REF])
        if ref_emb is None:
            print(f"{shot}: no face in first-frame image; identity/face-size metrics empty", flush=True)
        data = {}
        for v in variants:
            data[v], fresh = load_cached(vs[v], cache, a.every, a.max_yaw, pct)
            print(f"{shot} {v}: {'analyzed' if fresh else 'cached'}", flush=True)
        ms = {v: clip_metrics(data[v], ref_emb, ref_h, data[REF] if v != REF else None, a.cut_drop) for v in variants}
        add_ratios(ms)
        for v in variants:
            r = res.get((shot, v), {})
            ms[v] = {"clip": os.path.basename(vs[v]), "wall_s": r.get("wall_s"), "settings": r.get("settings"), **ms[v]}
        summary[shot] = ms
        sheet = os.path.join(cmp_dir, f"{shot}_sheet.jpg")
        contact_sheet(shot, variants, data, ms, res, pct, sheet)  # cheap; always redrawn so labels track results.jsonl
        with open(os.path.join(cmp_dir, f"{shot}.json"), "w") as fh:
            json.dump(ms, fh, indent=1)
    with open(os.path.join(cmp_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    md = [f"# {os.path.basename(os.path.normpath(a.out_dir))}: variants vs {REF}", "",
          "id = face cosine sim to first frame (frontal, largest face); fsize = face height vs first frame; "
          f"sharp/motion = ratio vs {REF}; comp = thumbnail SSIM vs {REF}; adj_min = lowest adjacent-sample SSIM", "",
          "| shot | variant | " + " | ".join(COLS) + " |", "|" + "---|" * (len(COLS) + 2)]
    for shot, ms in summary.items():
        for v, m in ms.items():
            md.append(f"| {shot} | {v} | " + " | ".join("" if m.get(c) is None else str(m.get(c)) for c in COLS) + " |")
    open(os.path.join(cmp_dir, "summary.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
