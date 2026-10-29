#!/usr/bin/env python3
"""Mid-video anchor tests (`anchor1`); run with venv_eval.

    venv_eval/bin/python harness/anchor_check.py prep --stock data/output/anchor1/ref/eval1 --shots er41 er43 ef41 ef31
    venv_eval/bin/python harness/anchor_check.py check --clips data/output/anchor1 --stock data/output/anchor1/ref/eval1 \
        --plain data/output/anchor1/ref/feat1 --out data/output/anchor1/anchor_check.json

prep: from each stock clip <shot>_orig10_*.mp4 writes the targets: frame --mid as data/input/anchor1/<shot>_f<mid>.png,
frames [--clip-at, --clip-at + 22) as data/input/anchor1_<shot>_clip<at>.mp4 (h264 crf 8) and the stock soundtrack as
data/input/anchor1_<shot>_stock.wav (top level: LoadVideo / LoadAudio list only that).
check: per clip <shot>_<variant>_*.mp4, and the same for the plain renders of that shot and preset (<shot>_s6 / _h2 in
--plain, never anchored) and stock itself:
- hit: PSNR / gray SSIM / DINOv2 of frame --mid (and of the last frame, and the mean over the clip-anchor frames) vs
  the stock frames;
- smoothness around the anchor: mean |frame diff| per frame pair, the max over frames mid-8..mid+8 divided by the
  clip's median (spike; 1.0-1.5 is normal), and the clip's median frame diff (motion amount) vs stock;
sheet: a contact sheet of --frames for --files (one row each), for looking at the frames around the anchor.
- audio: correlation of the 16 kHz mono track and of its 40 ms loudness envelope with stock's, over --audio-span
  (the audio anchor's span) and over the rest.
"""
import argparse, glob, json, math, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_clips as E  # noqa: E402

DATA = os.path.expanduser("~/rapid-h3/data")


def newest(d, pat):
    p = sorted(glob.glob(os.path.join(d, pat)))
    return p[-1] if p else None


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return 99.0 if mse == 0 else 10 * math.log10(255 ** 2 / mse)


def sim(frame, target):
    e = E.dino_emb([frame, target])
    return {"psnr": round(psnr(frame, target), 2), "ssim": round(float(E.ssim_gray(frame, target)), 3),
            "dino": round(float(E.cos(e[0], e[1])), 3)}


def diffs(fr):
    return np.array([np.abs(fr[i + 1].astype(np.float32) - fr[i]).mean() for i in range(len(fr) - 1)])


def env(x, hop=640):
    """log RMS loudness per 40 ms (16 kHz)"""
    k = len(x) // hop
    return 20 * np.log10(np.sqrt((x[:k * hop].reshape(k, hop) ** 2).mean(1)) + 1e-5)


def prep(a):
    import av, cv2, soundfile as sf
    os.makedirs(os.path.join(DATA, "input", "anchor1"), exist_ok=True)
    for shot in a.shots:
        p = newest(a.stock, f"{shot}_orig10_*.mp4")
        fr, fps, audio, sr = E.load_video(p)
        cv2.imwrite(os.path.join(DATA, "input", "anchor1", f"{shot}_f{a.mid}.png"), cv2.cvtColor(fr[a.mid], cv2.COLOR_RGB2BGR))
        out = av.open(os.path.join(DATA, "input", f"anchor1_{shot}_clip{a.clip_at}.mp4"), "w")
        st = out.add_stream("libx264", rate=24, options={"crf": "8"})
        st.width, st.height, st.pix_fmt = fr[0].shape[1], fr[0].shape[0], "yuv420p"
        for f in fr[a.clip_at:a.clip_at + 22]:
            for pk in st.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                out.mux(pk)
        for pk in st.encode():
            out.mux(pk)
        out.close()
        sf.write(os.path.join(DATA, "input", f"anchor1_{shot}_stock.wav"), audio.T, sr)
        print(shot, len(fr), fps, sr, flush=True)


def check(a):
    E.ARGS = type("A", (), {"cache": E._C, "threads": 8})()
    E.DEV = a.device
    pat = "*_*_[0-9][0-9][0-9][0-9][0-9]_.mp4"
    clips = {}
    for d in (a.clips, a.plain):
        for p in sorted(glob.glob(os.path.join(d, pat))):
            clips[os.path.basename(p).rsplit("_0", 1)[0]] = p
    res, stock_cache = {}, {}
    for cid, p in sorted(clips.items()):
        shot, var = cid.split("_", 1)
        if shot not in stock_cache:
            sp = newest(a.stock, f"{shot}_orig10_*.mp4")
            if not sp:
                continue
            sfr, _, sau, ssr = E.load_video(sp)
            sd = diffs(sfr)
            stock_cache[shot] = (sfr, E.resample(sau.mean(0), ssr, 16000), sd)
            res[f"{shot}_orig10"] = {"median_diff": round(float(np.median(sd)), 2),
                                     "spike": round(float(sd[a.mid - 8:a.mid + 8].max() / np.median(sd)), 2)}
        sfr, sau, sd = stock_cache[shot]
        fr, _, au, sr = E.load_video(p)
        d = diffs(fr)
        r = {"mid": sim(fr[a.mid], sfr[a.mid]), "last": sim(fr[-1], sfr[-1]),
             "clip_psnr": round(float(np.mean([psnr(fr[i], sfr[i]) for i in range(a.clip_at, a.clip_at + 22)])), 2),
             "median_diff": round(float(np.median(d)), 2), "motion_vs_stock": round(float(np.median(d) / np.median(sd)), 2),
             "spike": round(float(d[a.mid - 8:a.mid + 8].max() / np.median(d)), 2),
             "diff_around_mid": [round(float(v), 2) for v in d[a.mid - 4:a.mid + 4]]}
        y = E.resample(au.mean(0), sr, 16000)
        s0, s1 = int(a.audio_span[0] * 16000), int(a.audio_span[1] * 16000)
        n = min(len(y), len(sau))
        r["audio_corr_span"] = round(float(np.corrcoef(y[s0:s1], sau[s0:s1])[0, 1]), 3)
        r["audio_corr_rest"] = round(float(np.corrcoef(np.r_[y[:s0], y[s1:n]], np.r_[sau[:s0], sau[s1:n]])[0, 1]), 3)
        ey, es = env(y[:n]), env(sau[:n])
        k0, k1 = s0 // 640, s1 // 640
        r["env_corr_span"] = round(float(np.corrcoef(ey[k0:k1], es[k0:k1])[0, 1]), 3)
        r["env_corr_rest"] = round(float(np.corrcoef(np.r_[ey[:k0], ey[k1:]], np.r_[es[:k0], es[k1:]])[0, 1]), 3)
        res[cid] = r
        print(cid, json.dumps(r), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


def sheet(a):
    """one row per clip (--files), frames --frames, scaled to 160 px wide, labelled; written to --out (jpg)"""
    import cv2
    rows = []
    for p in a.files:
        fr, _, _, _ = E.load_video(p)
        tiles = []
        for i in a.frames:
            f = cv2.resize(fr[min(i, len(fr) - 1)], (a.tile, int(a.tile * fr[0].shape[0] / fr[0].shape[1])))
            cv2.putText(f, str(i), (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            tiles.append(f)
        row = np.concatenate(tiles, 1)
        lab = np.zeros((18, row.shape[1], 3), np.uint8)
        cv2.putText(lab, os.path.basename(p), (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        rows.append(np.concatenate([lab, row], 0))
    cv2.imwrite(a.out, cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prep", "check", "sheet"])
    ap.add_argument("--stock")
    ap.add_argument("--files", nargs="*")
    ap.add_argument("--frames", type=int, nargs="*", default=[0, 40, 62, 66, 68, 69, 70, 71, 72, 74, 78, 100, 140])
    ap.add_argument("--tile", type=int, default=160)
    ap.add_argument("--shots", nargs="*", default=["er41", "er43", "ef41", "ef31"])
    ap.add_argument("--clips")
    ap.add_argument("--plain")
    ap.add_argument("--out")
    ap.add_argument("--mid", type=int, default=70)
    ap.add_argument("--clip-at", type=int, default=60)
    ap.add_argument("--audio-span", type=float, nargs=2, default=[2.0, 4.0])
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    {"prep": prep, "check": check, "sheet": sheet}[a.mode](a)


if __name__ == "__main__":
    main()
