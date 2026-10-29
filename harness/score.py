#!/usr/bin/env python3
"""Same-seed parity of candidate renders against a reference variant.

Run inside the Rapid-H3 container (needs torch, av, safetensors).  For every prompt present
in both variants it reports:
  latent  rel-L2 and cosine of the final denoised latent (codec-free, most sensitive)
  video   PSNR, windowed SSIM (luma, 11x11 gaussian), and PSNR of the last quarter of the clip
  audio   log-mel cosine and log-mel L1 (dB) over the stereo track

These measure divergence from the reference, not quality on their own; the noise floor
(two references that differ only in a numerically-equivalent kernel) is the yardstick.

    score.py --out /data/output --ref ref --cand ref_cudnn fp8_rowwise --json scores.json
"""
import argparse, glob, json, math, os

import torch
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(4)


def find(out, variant, idx, ext):
    hits = sorted(glob.glob(os.path.join(out, variant, f"p{idx:02d}_*.{ext}")))
    return hits[-1] if hits else None


def load_latent(path):
    from safetensors.torch import load_file
    d = load_file(path)
    return {k: v.to(DEV).float() for k, v in d.items() if v.is_floating_point()}


def load_av(path):
    import av
    frames, audio, sr = [], [], None
    with av.open(path) as c:
        for f in c.decode(video=0):
            frames.append(torch.from_numpy(f.to_ndarray(format="rgb24")))
    with av.open(path) as c:
        if c.streams.audio:
            for f in c.decode(audio=0):
                audio.append(torch.from_numpy(f.to_ndarray()).float())
                sr = f.sample_rate
    v = torch.stack(frames).to(DEV).float().div(255) if frames else None
    a = torch.cat(audio, dim=-1).to(DEV) if audio else None
    return v, a, sr


def luma(x):
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]


def psnr(a, b):
    mse = (a - b).pow(2).mean().item()
    return float("inf") if mse == 0 else 10 * math.log10(1 / mse)


def ssim(a, b, win=11, sigma=1.5):
    ya, yb = luma(a).unsqueeze(1), luma(b).unsqueeze(1)
    g = torch.exp(-(torch.arange(win) - win // 2).float() ** 2 / (2 * sigma ** 2))
    g = (g / g.sum()).to(ya)
    k = (g[:, None] * g[None, :])[None, None]
    blur = lambda t: F.conv2d(t, k)
    mu_a, mu_b = blur(ya), blur(yb)
    saa = blur(ya * ya) - mu_a ** 2
    sbb = blur(yb * yb) - mu_b ** 2
    sab = blur(ya * yb) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2))
    return m.mean().item()


def logmel(a, sr, n_fft=2048, hop=512, n_mels=128):
    import torchaudio.functional as AF
    win = torch.hann_window(n_fft, device=a.device)
    spec = torch.stft(a, n_fft, hop, window=win, return_complex=True).abs() ** 2
    fb = AF.melscale_fbanks(n_fft // 2 + 1, 0.0, sr / 2, n_mels, sr).to(a.device)
    mel = torch.einsum("cft,fm->cmt", spec, fb)
    return 10 * torch.log10(mel.clamp_min(1e-10))


def compare(ref, cand, out, ref_out=None):
    rec = {}
    ref_out = ref_out or out
    lr, lc = find(ref_out, ref, cand[1], "latent"), find(out, cand[0], cand[1], "latent")
    if lr and lc:
        a, b = load_latent(lr), load_latent(lc)
        for k in a:
            if k in b and a[k].shape == b[k].shape:
                x, y = a[k].flatten(), b[k].flatten()
                rec[f"latent_{k}_relL2"] = ((x - y).norm() / x.norm()).item()
                rec[f"latent_{k}_cos"] = F.cosine_similarity(x, y, dim=0).item()
    vr, vc = find(ref_out, ref, cand[1], "mp4"), find(out, cand[0], cand[1], "mp4")
    if vr and vc:
        (va, aa, sr), (vb, ab, _) = load_av(vr), load_av(vc)
        n = min(len(va), len(vb))
        va, vb = va[:n], vb[:n]
        rec["psnr"] = psnr(va, vb)
        rec["psnr_last_quarter"] = psnr(va[-n // 4:], vb[-n // 4:])
        rec["ssim"] = sum(ssim(va[i:i + 16], vb[i:i + 16]) * len(va[i:i + 16]) for i in range(0, n, 16)) / n
        if aa is not None and ab is not None:
            m = min(aa.shape[-1], ab.shape[-1])
            ma, mb = logmel(aa[..., :m], sr), logmel(ab[..., :m], sr)
            rec["audio_logmel_cos"] = F.cosine_similarity(ma.flatten(), mb.flatten(), dim=0).item()
            rec["audio_logmel_l1_db"] = (ma - mb).abs().mean().item()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/output")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ref-out", help="output root holding the reference (default: --out)")
    ap.add_argument("--cand", nargs="+", required=True)
    ap.add_argument("--json")
    a = ap.parse_args()
    table = {}
    for variant in a.cand:
        idxs = sorted({int(os.path.basename(p)[1:3]) for p in glob.glob(os.path.join(a.out, variant, "p*_*.mp4"))})
        for i in idxs:
            table.setdefault(variant, {})[i] = compare(a.ref, (variant, i), a.out, a.ref_out)
    keys = ["latent_video_relL2", "latent_audio_relL2", "psnr", "psnr_last_quarter", "ssim", "audio_logmel_cos", "audio_logmel_l1_db"]
    print(f"{'variant':22s} {'p':>3s} " + " ".join(f"{k[:18]:>18s}" for k in keys))
    for variant, rows in table.items():
        for i, r in rows.items():
            print(f"{variant:22s} {i:3d} " + " ".join(f"{r.get(k, float('nan')):18.4f}" for k in keys))
        means = {k: sum(r[k] for r in rows.values() if k in r) / max(1, sum(k in r for r in rows.values())) for k in keys}
        print(f"{variant:22s} {'avg':>3s} " + " ".join(f"{means[k]:18.4f}" for k in keys))
    if a.json:
        json.dump(table, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
