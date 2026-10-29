"""Input-audio and per-token pin nodes for MiniMax H3.

Both write known content into the target's own AV latent and hold it with a per-token denoise mask
(ComfyUI PR #15375): masked rows run at the conditioning timestep and are re-injected from the latent
every step, so they come out exactly as given and cost no extra tokens.

  RapidH3InputAudio   a voice/music track -> the latent's audio, held fixed (a2v: the video follows the audio);
                   pad=silence/floor holds the whole clip's audio (the track padded), not just the track's span
  RapidH3LoadAVLatent .latent (safetensors with keys video/audio, as RapidH3SaveLatent writes) -> AV LATENT
  RapidH3PinLatent    copy a frame range (and optionally its audio) from a source AV latent and hold it;
                   an optional region mask (white = regenerate) turns the pin into inpainting
"""
import math
import os

import torch

import comfy.audio
import comfy.nested_tensor
import folder_paths

FPS = 24
AUDIO_HZ = 40
try:
    from comfy.ldm.minimax.model import FRAME_PER_TOKEN  # (1, 4, 4, 4, 4): pixel frames per video latent frame
except ImportError:
    FRAME_PER_TOKEN = (1, 4, 4, 4, 4)


def _av(latent):
    s = latent["samples"]
    if not getattr(s, "is_nested", False) or len(s.tensors) != 2 or s.tensors[0].shape[1] != 24:
        raise ValueError("expected a MiniMax H3 AV latent (nested video [B,24,T,H,W] + audio [B,32,2,A])")
    return s.tensors[0], s.tensors[1]


def _masks(latent, video, audio):
    """The latent's current per-stream masks (1 = generate), or all ones."""
    m = latent.get("noise_mask")
    if m is not None and getattr(m, "is_nested", False):
        return m.tensors[0].clone().float(), m.tensors[1].clone().float()
    return (torch.ones((1, 1) + tuple(video.shape[2:]), dtype=torch.float32),
            torch.ones((1, 1) + tuple(audio.shape[2:]), dtype=torch.float32))


def _out(latent, video, audio, vmask, amask):
    out = dict(latent)
    out["samples"] = comfy.nested_tensor.NestedTensor((video, audio))
    out["noise_mask"] = comfy.nested_tensor.NestedTensor((vmask, amask))
    return out


def latent_range(start, end):
    """Video latent frames [a, b) covering pixel frames [start, end] (inclusive)."""
    a = b = None
    f = 0
    k = 0
    while True:
        n = FRAME_PER_TOKEN[k % 5]
        if a is None and f + n > start:
            a = k
        if f + n > end:
            b = k + 1
            break
        f += n
        k += 1
    return a, b


def audio_t(frame):
    return int(round(frame / FPS * AUDIO_HZ))


class RapidH3InputAudio:
    """Put an audio track into the AV latent and hold it (a fixed voice line drives the video)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "latent": ("LATENT",),
            "audio_vae": ("VAE",),
            "audio": ("AUDIO",),
            "start_frame": ("INT", {"default": 0, "min": 0, "max": 9999,
                                    "tooltip": "pixel frame (24 fps) where the track starts"}),
            "hold": ("BOOLEAN", {"default": True, "tooltip": "mask the track's rows (kept exactly); off = only a starting point"}),
        }, "optional": {
            "pad": (["none", "silence", "floor"], {"default": "none", "tooltip":
                    "none: only the track's span is set (and held); silence / floor: the whole clip's audio is the track "
                    "padded with digital silence / a pink-noise floor at floor_db (under the track too), all held"}),
            "floor_db": ("FLOAT", {"default": -55.0, "min": -90.0, "max": -20.0, "step": 1.0,
                                   "tooltip": "RMS of the pink-noise floor (dBFS), pad=floor only"}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply"
    CATEGORY = "Rapid-H3"

    def apply(self, latent, audio_vae, audio, start_frame, hold=True, pad="none", floor_db=-55.0):
        video, alat = _av(latent)
        wave, sr = audio["waveform"], audio["sample_rate"]
        vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
        if sr != vae_sr:
            wave = comfy.audio.resample(wave, sr, vae_sr)
        if wave.shape[1] == 1:  # the H3 audio VAE is stereo
            wave = wave.repeat(1, 2, 1)
        if pad != "none":
            return (self._whole(latent, video, alat, audio_vae, wave[:1, :2].float(), vae_sr, start_frame, hold, pad, floor_db),)
        z = audio_vae.encode(wave[:1].movedim(1, -1)).to(alat.device, alat.dtype)  # [1, 32, 2, T]
        a0 = audio_t(start_frame)
        n = min(z.shape[-1], alat.shape[-1] - a0)
        if n < 1:
            raise ValueError(f"start_frame {start_frame} is past the end of the audio latent ({alat.shape[-1]} frames)")
        alat = alat.clone()
        alat[..., a0:a0 + n] = z[..., :n]
        vmask, amask = _masks(latent, video, alat)
        if hold:
            amask[..., a0:a0 + n] = 0.0
        print(f"[rapid-h3] input audio: {z.shape[-1] / AUDIO_HZ:.2f} s track, {n} of {alat.shape[-1]} audio frames "
              f"from frame {start_frame}{' held' if hold else ''}", flush=True)
        return (_out(latent, video, alat, vmask, amask),)

    @staticmethod
    def _whole(latent, video, alat, audio_vae, wave, vae_sr, start_frame, hold, pad, floor_db):
        """The whole clip's audio: the track at start_frame in a clip-long buffer (silence or a noise floor), encoded in
        one piece (no seams at the track's edges) and held everywhere, so the model cannot add sound before or after."""
        A = alat.shape[-1]
        hop = vae_sr // AUDIO_HZ  # 800 samples per audio latent frame
        buf = torch.zeros((1, 2, A * hop), dtype=torch.float32)
        o = int(round(start_frame / FPS * vae_sr))
        n = min(wave.shape[-1], buf.shape[-1] - o)
        if n < 1:
            raise ValueError(f"start_frame {start_frame} is past the end of the clip ({A / AUDIO_HZ:.2f} s)")
        buf[..., o:o + n] = wave[..., :n].cpu()
        if pad == "floor":  # stereo pink noise (1/f power), fixed seed, at floor_db RMS over the whole clip
            g = torch.Generator().manual_seed(0)
            spec = torch.fft.rfft(torch.randn((1, 2, buf.shape[-1]), generator=g), dim=-1)
            f = torch.fft.rfftfreq(buf.shape[-1], 1.0 / vae_sr)
            noise = torch.fft.irfft(spec / f.clamp(min=20.0).sqrt(), n=buf.shape[-1], dim=-1)
            buf += noise / noise.pow(2).mean().sqrt() * 10 ** (floor_db / 20)
        z = audio_vae.encode(buf.movedim(1, -1)).to(alat.device, alat.dtype)  # [1, 32, 2, A]
        if z.shape[-1] < A:  # never seen (A * hop samples encode to A frames); repeat the last frame to be safe
            z = torch.cat([z, z[..., -1:].expand(*z.shape[:-1], A - z.shape[-1])], dim=-1)
        alat = alat.clone()
        alat[...] = z[..., :A]
        vmask, amask = _masks(latent, video, alat)
        if hold:
            amask[...] = 0.0
        print(f"[rapid-h3] input audio: {n / vae_sr:.2f} s track from frame {start_frame}, whole clip ({A} audio frames) "
              f"padded with {pad}{f' at {floor_db:.0f} dB' if pad == 'floor' else ''}{' held' if hold else ''}", flush=True)
        return _out(latent, video, alat, vmask, amask)


class RapidH3LoadAVLatent:
    """Load a .latent written by RapidH3SaveLatent (safetensors with `video` and `audio`)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"path": ("STRING", {"default": "", "tooltip": "relative to the output directory, or absolute"})}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "load"
    CATEGORY = "Rapid-H3"

    @staticmethod
    def _path(path):
        return path if os.path.isabs(path) else os.path.join(folder_paths.get_output_directory(), path)

    @classmethod
    def IS_CHANGED(cls, path):
        p = cls._path(path)
        return os.path.getmtime(p) if os.path.exists(p) else float("nan")

    def load(self, path):
        from safetensors.torch import load_file
        d = load_file(self._path(path))
        if "video" not in d or "audio" not in d:
            raise ValueError(f"{path}: not an AV latent (keys {list(d)})")
        return ({"samples": comfy.nested_tensor.NestedTensor((d["video"].float(), d["audio"].float()))},)


class RapidH3PinLatent:
    """Copy pixel frames [start, end] of a source AV latent into the target and hold them."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "latent": ("LATENT",),
            "source": ("LATENT",),
            "start_frame": ("INT", {"default": 0, "min": 0, "max": 9999}),
            "end_frame": ("INT", {"default": 21, "min": 0, "max": 9999, "tooltip": "inclusive"}),
            "audio": (["none", "same"], {"default": "none", "tooltip": "same: also pin the audio under those frames"}),
        }, "optional": {
            "region": ("MASK", {"tooltip": "white = regenerate inside the pinned frames (inpainting); black = keep"}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply"
    CATEGORY = "Rapid-H3"

    def apply(self, latent, source, start_frame, end_frame, audio, region=None):
        video, alat = _av(latent)
        sv, sa = _av(source)
        if sv.shape[-2:] != video.shape[-2:]:
            raise ValueError(f"source latent is {tuple(sv.shape[-2:])}, target {tuple(video.shape[-2:])}: "
                             "pin at the size the stage samples at (ts_s6 / one stage: full size)")
        a, b = latent_range(start_frame, end_frame)
        b = min(b, video.shape[2], sv.shape[2])
        if a >= b:
            raise ValueError(f"frames {start_frame}-{end_frame} are outside the latents")
        video = video.clone()
        video[:, :, a:b] = sv[:1, :, a:b].to(video.device, video.dtype)
        vmask, amask = _masks(latent, video, alat)
        keep = torch.zeros((1, 1, b - a) + tuple(video.shape[-2:]), dtype=torch.float32)
        if region is not None:  # MASK [B, H, W] in pixels, 1 = regenerate
            r = region[:1, None].float()
            r = torch.nn.functional.interpolate(r, size=tuple(video.shape[-2:]), mode="area").clamp(0, 1)
            keep = (r > 0.5).float()[:, :, None].expand(1, 1, b - a, *video.shape[-2:]).clone()
        vmask[:, :, a:b] = torch.minimum(vmask[:, :, a:b], keep)
        if audio == "same":
            a0, a1 = audio_t(start_frame), min(audio_t(end_frame + 1), alat.shape[-1], sa.shape[-1])
            alat = alat.clone()
            alat[..., a0:a1] = sa[:1, ..., a0:a1].to(alat.device, alat.dtype)
            amask[..., a0:a1] = 0.0
        print(f"[rapid-h3] pin: latent frames {a}-{b - 1} (pixel {start_frame}-{end_frame})"
              f"{', region mask' if region is not None else ''}{', audio' if audio == 'same' else ''}", flush=True)
        return (_out(latent, video, alat, vmask, amask),)


NODE_CLASS_MAPPINGS = {"RapidH3InputAudio": RapidH3InputAudio, "RapidH3LoadAVLatent": RapidH3LoadAVLatent,
                       "RapidH3PinLatent": RapidH3PinLatent}
