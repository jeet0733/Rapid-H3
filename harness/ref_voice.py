#!/usr/bin/env python3
"""Ref2VA voice references, in the eval venv:

  venv_eval/bin/python harness/ref_voice.py make CLIP.mp4 OUT.wav
      the clip's speech as a reference audio: demucs htdemucs vocals stem (ambience removed), 44.1 kHz mono wav
  venv_eval/bin/python harness/ref_voice.py score --clips data/output/ref1 --shots harness/shots/ref2.json
      after eval_clips.py on the same clips: ECAPA cosine of each clip's speech (eval cache spk_emb) to its shot's
      reference voice (shot "ref_audio", in data/input), embedded the same way (vocals, active 50 ms windows);
      (v_other: the same clip vs the other shots' reference voices, the unrelated-voice baseline); also
      ArcFace id_ref / DINOv2 person_ref and mid_dino to the reference image from the eval -> table on stdout
"""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_clips as ec  # noqa: E402


def vocals(path):
    import av, torch
    c = av.open(path)
    st = c.streams.audio[0]
    sr = st.rate
    rs = av.AudioResampler(format="fltp", layout="stereo", rate=sr)
    ch = []
    for fr in c.decode(st):
        ch += [o.to_ndarray() for o in rs.resample(fr)]
    ch += [o.to_ndarray() for o in rs.resample(None)]
    c.close()
    audio = np.concatenate(ch, axis=1).astype(np.float32)
    m = ec.demucs_model()
    from demucs.apply import apply_model
    w = torch.from_numpy(ec.resample(audio, sr, m.samplerate))
    ref = w.mean(0)
    mu, sd = ref.mean(), ref.std() + 1e-8
    with torch.no_grad():
        s = apply_model(m, ((w - mu) / sd)[None].to(ec.DEV), device=ec.DEV, split=True, overlap=0.25, progress=False)[0]
    s = (s * sd + mu).cpu().numpy()
    return s[m.sources.index("vocals")].mean(0), m.samplerate


def speaker_emb(wav, sr):
    import torch
    v16 = ec.resample(wav, sr, 16000)
    hop = 800
    rms = np.array([np.sqrt(np.mean(v16[i:i + hop] ** 2)) for i in range(0, len(v16) - hop + 1, hop)])
    act = rms > max(0.1 * rms.max(), 1e-4)
    seg = np.concatenate([v16[i * hop:(i + 1) * hop] for i in np.nonzero(act)[0]])
    with torch.no_grad():
        e = ec.ecapa().encode_batch(torch.from_numpy(seg)[None].to(ec.DEV)).squeeze().float().cpu().numpy()
    return e / np.linalg.norm(e)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make")
    mk.add_argument("clip")
    mk.add_argument("out")
    sc = sub.add_parser("score")
    sc.add_argument("--clips", required=True)
    sc.add_argument("--shots", required=True)
    sc.add_argument("--input", default=os.path.join(ec.__file__.rsplit("/harness/", 1)[0], "data", "input"))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    ec.DEV = a.device
    ec.ARGS = argparse.Namespace(cache=ec._C)  # model cache dir, as eval_clips' --cache default
    if a.cmd == "make":
        import soundfile as sf
        v, sr = vocals(a.clip)
        v = v / max(np.abs(v).max(), 1e-6) * 0.9
        sf.write(a.out, v, sr, subtype="PCM_16")
        print(a.out, f"{len(v) / sr:.2f} s")
        return
    shots = {s["id"]: s for s in json.load(open(a.shots))}
    evals = json.load(open(os.path.join(a.clips, "eval", "shots.json")))
    import soundfile as sf
    refs = {}
    for s in shots.values():
        ra = s.get("ref_audio")
        if ra and ra not in refs:
            w, sr = sf.read(os.path.join(a.input, ra), dtype="float32")
            refs[ra] = speaker_emb(w if w.ndim == 1 else w.mean(1), sr)
    rows = []
    for sid, s in sorted(shots.items()):
        ra = s.get("ref_audio")
        for v, r in sorted(evals.get(sid, {}).items()):
            npz = os.path.join(a.clips, "eval", "clips", os.path.basename(r["clip"]).rsplit(".", 1)[0] + ".npz")
            spk = other = None
            if ra and os.path.exists(npz):
                z = np.load(npz)
                if "spk_emb" in z:
                    spk = round(ec.cos(z["spk_emb"], refs[ra]), 3)
                    # baseline: the same clip against the other characters' reference voices
                    other = round(float(np.mean([ec.cos(z["spk_emb"], e) for k, e in refs.items() if k != ra])), 3)
            rows.append((sid, v, r.get("id_ref"), r.get("person_ref"), r.get("mid_dino"), spk, other, r.get("wer"),
                         r.get("pitch_med")))
    cols = ("id_ref", "person", "middino", "voice", "v_other", "wer", "f0")
    print(f"{'shot':6s} {'variant':10s} " + " ".join(f"{c:>7s}" for c in cols))
    for row in rows:
        print(f"{row[0]:6s} {row[1]:10s} " + " ".join(f"{'-' if x is None else x:>7}" for x in row[2:]))
    by_v = {}
    for row in rows:
        by_v.setdefault(row[1], []).append(row)
    print("\nmeans (mid_dino: middle frame vs the reference image, high = the reference is pasted)")
    for v, rs in sorted(by_v.items()):
        m = [np.mean([r[k] for r in rs if r[k] is not None]) if any(r[k] is not None for r in rs) else float("nan")
             for k in range(2, 8)]
        print(f"{v:10s} " + "  ".join(f"{c} {x:.3f}" for c, x in zip(cols, m)) + f"  (n={len(rs)})")

if __name__ == "__main__":
    main()
