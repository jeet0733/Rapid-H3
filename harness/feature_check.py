#!/usr/bin/env python3
"""Feature-specific checks for the feat1 tests; run with venv_eval.

    venv_eval/bin/python harness/feature_check.py --clips data/output/feat1 --targets data/input/feat1 \
        --audio-dir data/input --out data/output/feat1/feature_check.json

The target images (data/input/feat1) and input tracks (feat1_tts_<shot>.wav) are your own; they are not in the repo.

- last frame (`*_last` clips): the clip's final frame vs the target image (PSNR, gray SSIM, DINOv2 cosine), and the
  same for the plain clip of that shot and preset (`<shot>_s6`, `<shot>_h2`), which never saw the target.
- input audio (`*_a2v*` clips, track feat1_tts_<shot>.wav placed at frame 12): the clip's audio vs the input track,
  16 kHz mono: best normalized cross-correlation and its lag (ms), SNR after least-squares gain over the track,
  and how much louder the clip is outside the track (generated sound around the held line).
- pins (`*_pin_head`: frames 0-21 + audio from `<shot>_s6`; `*_pin_inpaint`: all frames, right half regenerated):
  PSNR of pinned pixels vs the source clip, the regenerated part's PSNR (low = really regenerated), and the
  seam: mean |diff| between frames 21 and 22 (head) or across the half boundary (inpaint) vs the clip's typical value.
"""
import argparse, glob, json, math, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_clips as E  # noqa: E402

PLAIN = {"h2": "h2", "s6": "s6"}


def frames_audio(path):
    fr, fps, audio, sr = E.load_video(path)
    return np.stack(fr), audio, sr


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return 99.0 if mse == 0 else 10 * math.log10(255 ** 2 / mse)


def sim(frame, target):
    import cv2
    t = cv2.resize(target, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA)
    e = E.dino_emb([frame, t])
    return {"psnr": round(psnr(frame, t), 2), "ssim": round(float(E.ssim_gray(frame, t)), 3),
            "dino": round(float(E.cos(e[0], e[1])), 3)}


def mono16(audio, sr):
    return E.resample(audio.mean(0), sr, 16000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--audio-start", type=int, default=12)
    ap.add_argument("--device", default="cpu", help="DINOv2 device (cpu leaves the GPU to renders and eval_clips)")
    a = ap.parse_args()
    E.ARGS = type("A", (), {"cache": E._C, "threads": 8})()
    E.DEV = a.device
    clips = {os.path.basename(p).rsplit("_0", 1)[0]: p for p in sorted(glob.glob(os.path.join(a.clips, "*.mp4")))}
    res = {}
    import cv2
    for cid, p in sorted(clips.items()):
        shot = cid.split("_")[0]
        r = {}
        if cid.endswith("_last") or cid in (f"{shot}_s6", f"{shot}_h2"):
            tp = os.path.join(a.targets, f"{shot}_last.png")
            if os.path.exists(tp):
                fr, _, _ = frames_audio(p)
                target = cv2.cvtColor(cv2.imread(tp), cv2.COLOR_BGR2RGB)
                r["last_vs_target"] = sim(fr[-1], target)
                r["frames"] = len(fr)
        if "_a2v" in cid:
            fr, audio, sr = frames_audio(p)
            import soundfile as sf
            w, wsr = sf.read(os.path.join(a.audio_dir, f"feat1_tts_{shot}.wav"), dtype="float32")
            x = E.resample(w if w.ndim == 1 else w.mean(1), wsr, 16000)
            y = mono16(audio, sr)
            off = int(round(a.audio_start / 24 * 16000))
            seg = y[max(0, off - 1600): off + len(x) + 1600]
            c = np.correlate(seg - seg.mean(), x - x.mean(), mode="valid")
            k = int(np.argmax(np.abs(c)))
            ncc = float(c[k] / (np.linalg.norm(x - x.mean()) * np.linalg.norm(seg[k:k + len(x)] - seg[k:k + len(x)].mean()) + 1e-9))
            yy = seg[k:k + len(x)]
            gain = float(np.dot(yy, x) / (np.dot(x, x) + 1e-12))
            snr = float(10 * np.log10(np.sum((gain * x) ** 2) / (np.sum((yy - gain * x) ** 2) + 1e-12)))
            start = max(0, off - 1600) + k
            outside = np.concatenate([y[:start], y[start + len(x):]])
            r["audio_vs_input"] = {"ncc": round(ncc, 3), "lag_ms": round((start - off) / 16, 1), "snr_db": round(snr, 1),
                                   "gain": round(gain, 3),
                                   "rms_in_track_db": round(20 * np.log10(np.sqrt(np.mean(yy ** 2)) + 1e-9), 1),
                                   "rms_outside_db": round(20 * np.log10(np.sqrt(np.mean(outside ** 2)) + 1e-9), 1)}
        if "_pin_" in cid and f"{shot}_s6" in clips:
            fr, audio, sr = frames_audio(p)
            src, saudio, ssr = frames_audio(clips[f"{shot}_s6"])
            n = min(len(fr), len(src))
            adj = float(np.mean([np.abs(fr[i + 1].astype(np.float32) - fr[i]).mean() for i in range(n - 1)]))
            if cid.endswith("_pin_head"):
                r["pinned_psnr"] = round(float(np.mean([psnr(fr[i], src[i]) for i in range(0, 22)])), 2)
                r["after_psnr"] = round(float(np.mean([psnr(fr[i], src[i]) for i in range(30, n)])), 2)
                r["seam_diff"] = round(float(np.abs(fr[22].astype(np.float32) - fr[21]).mean()), 2)
                r["typical_adjacent_diff"] = round(adj, 2)
                y, ys = mono16(audio, sr), mono16(saudio, ssr)
                m = int(22 / 24 * 16000)
                r["audio_head_corr"] = round(float(np.corrcoef(y[:m], ys[:m])[0, 1]), 3)
            else:
                w = fr.shape[2]
                keep = slice(0, w // 2 - 16)
                regen = slice(w // 2 + 16, w)
                r["kept_psnr"] = round(float(np.mean([psnr(fr[i][:, keep], src[i][:, keep]) for i in range(0, n, 10)])), 2)
                r["regen_psnr"] = round(float(np.mean([psnr(fr[i][:, regen], src[i][:, regen]) for i in range(0, n, 10)])), 2)
                col = [np.abs(fr[i][:, w // 2].astype(np.float32) - fr[i][:, w // 2 - 1]).mean() for i in range(0, n, 10)]
                ref_col = [np.abs(src[i][:, w // 2].astype(np.float32) - src[i][:, w // 2 - 1]).mean() for i in range(0, n, 10)]
                r["seam_col_diff"] = round(float(np.mean(col)), 2)
                r["source_col_diff"] = round(float(np.mean(ref_col)), 2)
        if r:
            res[cid] = r
            print(cid, json.dumps(r, default=float), flush=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
