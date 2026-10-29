#!/usr/bin/env python3
"""Audio quality of rendered clips: no-reference speech MOS predictors, per shot and variant.

    venv_eval/bin/python harness/audio_quality.py --clips data/output/eval1 --ref-variant orig10 [--device cpu]

Per clip, on the 16 kHz mono mix and on the htdemucs vocals stem:
- DNSMOS P.835 (SIG speech quality, BAK background intrusiveness, OVRL overall; 1-5) and P.808 MOS, from
  microsoft/DNS-Challenge DNSMOS/DNSMOS/{sig_bak_ovr,model_v8}.onnx in models_eval/dnsmos (scoring as dnsmos_local.py:
  the clip is tiled to 9.01 s, non-personalized polynomial mapping);
- UTMOS22 strong (speech naturalness MOS, 1-5) on the vocals, tarepan/SpeechMOS (ed25eac, MIT) in third_party/SpeechMOS;
- level: RMS and peak dBFS, clipped-sample fraction, and the vocals' 95 % spectral rolloff on speech frames (muffled
  speech has a low rolloff).
Clips are named <shot>_<variant>_NNNNN_.mp4 (eval_clips.find_clips). Results are cached per clip in <out>/clips/;
<out>/audio_quality.json holds shot -> variant -> metrics and <out>/audio_quality.md the tables. The DNSMOS BAK score
also penalises background sound the prompt asked for (rain, gulls), so compare variants of the same shot, not shots.
"""
import argparse, hashlib, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_clips import find_clips, resample  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DNSMOS_DIR = os.path.join(ROOT, "models_eval", "dnsmos")
SPEECHMOS = os.path.join(ROOT, "third_party", "SpeechMOS")
SR = 16000
METRICS = ["mos_ovrl", "mos_sig", "mos_bak", "mos_p808", "voc_ovrl", "voc_sig", "utmos", "rolloff_hz",
           "rms_db", "peak_db", "clip_frac"]
_m = {}


def load_audio(path):
    import av
    c = av.open(path)
    st = c.streams.audio[0]
    sr = st.rate
    rs = av.AudioResampler(format="fltp", layout="stereo", rate=sr)
    ch = []
    for fr in c.decode(st):
        ch += [o.to_ndarray() for o in rs.resample(fr)]
    ch += [o.to_ndarray() for o in rs.resample(None)]
    c.close()
    return np.concatenate(ch, axis=1).astype(np.float32), sr


def dnsmos(x):
    # Scoring adapted from microsoft/DNS-Challenge DNSMOS/dnsmos_local.py (MIT, Copyright (c) Microsoft Corporation;
    # licenses/DNSMOS-MIT.txt): the input tiling, the P.808 mel features and the polynomial score mapping.
    import onnxruntime as ort, librosa
    if "dns" not in _m:
        _m["dns"] = (ort.InferenceSession(os.path.join(DNSMOS_DIR, "sig_bak_ovr.onnx")),
                     ort.InferenceSession(os.path.join(DNSMOS_DIR, "model_v8.onnx")))
    s835, s808 = _m["dns"]
    n = int(9.01 * SR)
    while len(x) < n:
        x = np.append(x, x)
    rows = []
    for i in range(int(np.floor(len(x) / SR) - 9.01) + 1):
        seg = x[i * SR: i * SR + n].astype(np.float32)
        if len(seg) < n:
            continue
        mel = librosa.feature.melspectrogram(y=seg[:-160], sr=SR, n_fft=321, hop_length=160, n_mels=120)
        mel = ((librosa.power_to_db(mel, ref=np.max) + 40) / 40).T.astype(np.float32)
        p808 = s808.run(None, {"input_1": mel[None]})[0][0][0]
        sig, bak, ovr = s835.run(None, {"input_1": seg[None]})[0][0]
        rows.append((np.poly1d([-0.08397278, 1.22083953, 0.0052439])(sig),
                     np.poly1d([-0.13166888, 1.60915514, -0.39604546])(bak),
                     np.poly1d([-0.06766283, 1.11546468, 0.04602535])(ovr), p808))
    sig, bak, ovr, p808 = np.mean(rows, 0)
    return dict(sig=float(sig), bak=float(bak), ovrl=float(ovr), p808=float(p808))


def utmos(x, dev):
    import torch
    if "utmos" not in _m:
        _m["utmos"] = torch.hub.load(SPEECHMOS, "utmos22_strong", source="local", trust_repo=True).to(dev)
    with torch.no_grad():
        return float(_m["utmos"](torch.from_numpy(x)[None].to(dev), SR).item())


def vocals(audio, sr, dev):
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    if "demucs" not in _m:
        _m["demucs"] = get_model("htdemucs").to(dev).eval()
    m = _m["demucs"]
    w = torch.from_numpy(resample(audio, sr, m.samplerate))
    ref = w.mean(0)
    mu, sd = ref.mean(), ref.std() + 1e-8
    with torch.no_grad():
        s = apply_model(m, ((w - mu) / sd)[None].to(dev), device=dev, split=True, overlap=0.25, progress=False)[0]
    s = (s * sd + mu).cpu().numpy()
    return resample(s[m.sources.index("vocals")].mean(0), m.samplerate, SR)


def rolloff(v):
    import librosa
    hop = 800
    rms = np.array([np.sqrt(np.mean(v[i:i + hop] ** 2)) for i in range(0, len(v) - hop + 1, hop)])
    act = np.nonzero(rms > max(0.1 * rms.max(), 1e-4))[0] if len(rms) else []
    if len(act) < 10:
        return None
    seg = np.concatenate([v[i * hop:(i + 1) * hop] for i in act])
    return float(np.median(librosa.feature.spectral_rolloff(y=seg, sr=SR, roll_percent=0.95)))


def measure(path, dev):
    audio, sr = load_audio(path)
    mono = audio.mean(0)
    m16 = resample(mono, sr, SR)
    v16 = vocals(audio, sr, dev)
    d, dv = dnsmos(m16), dnsmos(v16)
    return {"mos_ovrl": d["ovrl"], "mos_sig": d["sig"], "mos_bak": d["bak"], "mos_p808": d["p808"],
            "voc_ovrl": dv["ovrl"], "voc_sig": dv["sig"], "utmos": utmos(v16, dev), "rolloff_hz": rolloff(v16),
            "rms_db": float(20 * np.log10(np.sqrt(np.mean(mono ** 2)) + 1e-9)),
            "peak_db": float(20 * np.log10(np.abs(audio).max() + 1e-9)),
            "clip_frac": float(np.mean(np.abs(audio) >= 0.999)),
            "audio_md5": hashlib.md5(np.round(mono, 4).tobytes()).hexdigest()[:12]}


def fmt(x, k=2):
    return "" if x is None else f"{x:.{k}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--out", default=None, help="default <clips>/audio")
    ap.add_argument("--ref-variant", default="orig10")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    out = a.out or os.path.join(a.clips, "audio")
    os.makedirs(os.path.join(out, "clips"), exist_ok=True)
    clips = find_clips(a.clips)
    res = {}
    for s in sorted(clips):
        for v, p in sorted(clips[s].items()):
            cp = os.path.join(out, "clips", os.path.basename(p) + ".json")
            if os.path.exists(cp) and os.path.getmtime(cp) >= os.path.getmtime(p):
                r = json.load(open(cp))
            else:
                r = measure(p, a.device)
                json.dump(r, open(cp, "w"))
                print(s, v, {k: round(r[k], 2) for k in ("mos_ovrl", "voc_sig", "utmos") if r[k] is not None}, flush=True)
            res.setdefault(s, {})[v] = r
    json.dump(res, open(os.path.join(out, "audio_quality.json"), "w"), indent=1)
    variants = sorted({v for s in res for v in res[s]}, key=lambda v: (v != a.ref_variant, v))
    L = [f"# Audio quality: {a.clips}, reference {a.ref_variant}", "",
         "DNSMOS P.835 (mix: OVRL/SIG/BAK, P.808; vocals stem: OVRL/SIG), UTMOS22 on the vocals (1-5, higher is "
         "better); rolloff = 95 % spectral rolloff of the vocals on speech frames (Hz); level in dBFS. "
         "Brackets: mean change vs the reference over the shots both have.", "",
         "## Means", "", "| variant | n | " + " | ".join(METRICS) + " |", "|---|---|" + "---|" * len(METRICS)]
    for v in variants:
        sh = [s for s in res if v in res[s]]
        cells = []
        for k in METRICS:
            vals = [res[s][v][k] for s in sh if res[s][v][k] is not None]
            c = fmt(np.mean(vals), 3 if k == "clip_frac" else 2) if vals else ""
            if v != a.ref_variant and vals:
                dd = [res[s][v][k] - res[s][a.ref_variant][k] for s in sh
                      if a.ref_variant in res[s] and res[s][v][k] is not None and res[s][a.ref_variant][k] is not None]
                if dd:
                    c += f" ({np.mean(dd):+.{3 if k == 'clip_frac' else 2}f})"
            cells.append(c)
        L.append(f"| {v} | {len(sh)} | " + " | ".join(cells) + " |")
    L += ["", "## Per shot", "", "| shot | variant | " + " | ".join(METRICS) + " | audio md5 |",
          "|---|---|" + "---|" * (len(METRICS) + 1)]
    for s in sorted(res):
        for v in variants:
            if v in res[s]:
                r = res[s][v]
                L.append(f"| {s} | {v} | " + " | ".join(fmt(r[k], 3 if k == "clip_frac" else 2) for k in METRICS)
                         + f" | {r['audio_md5']} |")
    open(os.path.join(out, "audio_quality.md"), "w").write("\n".join(L) + "\n")
    print("wrote", os.path.join(out, "audio_quality.md"))


if __name__ == "__main__":
    main()
