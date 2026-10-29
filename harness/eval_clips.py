#!/usr/bin/env python3
"""Automatic evaluation of MiniMax-H3 video+audio clips, every variant against a reference variant (ts_h).

Run on the Spark host in the eval venv (harness/eval_setup.sh):

  cd ~/rapid-h3 && OMP_NUM_THREADS=8 venv_eval/bin/python harness/eval_clips.py --workers 2 \
      --clips data/output/eval1 --shots harness/shots/eval1.json --ref-dir data/input --ref-variant ts_h

Clips are named <shot>_<variant>_<nnnnn>_.mp4 (newest counter per shot and variant wins). Output in --out
(default <clips>/eval): clips/<clip>.json + .npz (per-clip measurements, cached by file size + mtime, so a re-run
only scores new clips), shots.json (per-clip metrics, comparisons vs the reference variant, flags) and summary.md.
About 1-2 min per clip on a GB10 (faces on CPU, the rest on the GPU); --workers 2 runs two in parallel.

Criteria and metrics (flag thresholds and their reasons: THRESH below):
 1 identity   id_ref: ArcFace (InsightFace buffalo_l) cosine of each reference-image face to its best-matching
              near-frontal face per frame, median over frames (p05 too); id_xclip: cosine between this clip's
              mean face embedding and the other clips of the same character (same variant); person_ref: DINOv2
              cosine of a head-and-torso crop (face box widened 3x, 3.5x tall) to the same crop of the reference
              image (hair, clothing); id_vs_ref / person_vs_ref: the same embeddings against the ts_h clip.
 2 voice      spk_vs_ref: ECAPA-TDNN (speechbrain spkrec-ecapa-voxceleb) cosine between this clip's speech and the
              ts_h clip of the same shot; spk_xclip: against the other clips of the same character; pitch_med / pitch_sd:
              median F0 (Hz) and F0 spread (semitones, std) from librosa pyin. Speech is the demucs htdemucs vocals
              stem (ambience removed), trimmed to its active part. A two-speaker clip gives one mixed embedding.
 3 lip sync   lse_c / lse_d / av_offset: SyncNet (joonson/syncnet_python, syncnet_v2.model) confidence, min
              distance and best offset (frames at 25 fps) on S3FD face tracks of the 25-fps-resampled clip, best
              track (the speaker); lse_c_sp / lse_d_sp / av_offset_sp: the same over the windows with speech
              (vocals stem active) only; lsp_r: mouth-opening vs loudness correlation (a simple lip-sync proxy).
 4 dialogue   wer: Whisper large-v3-turbo (faster-whisper) transcript vs the scripted line after Whisper's English
              normaliser; missing / inserted / substituted words listed. Clips scripted without speech report the
              number of words heard instead.
 5 temporal   id_first_min / id_first_med: ArcFace of the main face to the clip's own first face; flicker: mean
              |dY| between consecutive frames on static pixels (RAFT flow < 0.25 px); tflick: temporal
              high-frequency energy |Y_t - (Y_t-1 + Y_t+1)/2| on static pixels; warp_err: RAFT
              occlusion-masked warping error between consecutive frames (0-255 RGB); cut: thumbnail-SSIM drop.
 6 motion     flow_mag: mean RAFT flow (px/frame at full res); dyn_degree: VBench dynamic degree (share of 8-fps
              frame pairs whose top-5% flow exceeds 6 px * min(H,W)/256) and is_dynamic; accel: mean
              |F(t->t+1) + F(t->t-1)| / flow_mag (0 = constant velocity); frozen: still clip or a >1 s freeze.
 7 ref image  f0_ssim / f0_lpips / f0_dino / f0_hist: frame 0 vs the reference image (SSIM, LPIPS-alex, DINOv2
              global cosine, HSV histogram Bhattacharyya distance); the same for the middle and last frames
              (mid_*, last_*); ArcFace at those frames is id_ref above.
 8 prompt     clip_score: 100 x cosine of CLIP ViT-L/14 text (prompt without camera/audio boilerplate) and 8 frames;
              clip_margin: own prompt minus the mean over the other prompts in --shots (does the clip match its
              own prompt better than the others?); zoom: cumulative scale change over the clip from a fit of the
              RAFT flow field, and face_growth: last/first face height; checked against the camera in the prompt.
 9 quality    sharp_face / sharp_frame: Laplacian variance of the face crop (128 px) / whole frame; musiq, clipiqa:
              pyiqa no-reference IQA on 8 frames; noise: wavelet noise sigma (skimage estimate_sigma, 0-255).
              dover: DOVER fused video quality (0-1; technical + aesthetic), DOVER's own view sampling on our frames.
"""
import argparse, functools, glob, json, math, os, re, sys, time, warnings

import av
import cv2
import numpy as np

warnings.filterwarnings("ignore")
_C = os.path.expanduser(os.environ.get("RAPIDH3_EVAL_CACHE", "~/rapid-h3/eval_cache"))
os.environ.setdefault("HF_HOME", os.path.join(_C, "hf"))
os.environ.setdefault("TORCH_HOME", os.path.join(_C, "torch"))
HERE = os.path.dirname(os.path.abspath(__file__))
CODE_VER = 2
DEV = "cuda"
ARGS = None
THUMB = 64

# --------------------------------------------------------------------------------------------- thresholds
# kind: min = flag if value < x, max = flag if value > x, range = flag outside [lo, hi];
# delta_* thresholds compare against the reference variant (ts_h) of the same shot.
# Calibration: "r2 spread" = h_r1 vs ts_h on 48 r2 shots (same seed, visually indistinguishable by eye), the noise
# floor of a re-render that changes only the last refine step; "different" = ts_h clips of different shots.
THRESH = {
    # 1 identity. buffalo_l's verification threshold is ~0.3-0.4. id_ref < 0.40 = a lost face. Two renders of the same
    # shot score id_vs_ref 0.66-0.98 (r2), different people ~0.04 (p95 0.20), so < 0.60 = a different face.
    # id_ref / person_ref drop: r2 spread p10 -0.10 / -0.03 (text-only shots, see below), so -0.10 / -0.05.
    "id_ref": ("min", 0.40), "id_ref_d": ("delta_min", -0.10), "id_vs_ref": ("min", 0.60),
    "id_xclip": ("min", 0.45), "person_ref": ("min", 0.55), "person_ref_d": ("delta_min", -0.05),
    # 2 voice. ECAPA (VoxCeleb EER ~1 %; its own verification threshold 0.25). Same shot re-rendered: 0.83-1.0;
    # different shots: median 0.14, but similar voice descriptions reach 0.7. So < 0.70 vs ts_h = a different voice;
    # across a character's clips (different lines and emotions) < 0.40. Median F0 shift > 2 semitones is clearly
    # audible (an octave is 12); only computed when both clips are >= 25 % voiced (pyin fails on whispers).
    "spk_vs_ref": ("min", 0.70), "spk_xclip": ("min", 0.40), "pitch_shift_st": ("absmax", 2.0),
    # 3 lip sync. SyncNet on real talking heads: LSE-C ~7-8, LSE-D ~6.5-7; unsynced: LSE-C < 2, LSE-D > 10. Our clips
    # are only ~40 % speech, which drags whole-clip LSE-C down (r2 ts_h median 4.6 whole clip, 7.1 on speech), so
    # the absolute flags use the speech-gated numbers: LSE-C_sp < 3 or LSE-D_sp > 10. Drops vs ts_h: r2 spread min
    # -1.2 (whole) / -1.8 (speech), so -1.5 / -2.0. An offset beyond 3 frames (120 ms) is visible.
    "lse_c_sp": ("min", 3.0), "lse_d_sp": ("max", 10.0), "lse_c_d": ("delta_min", -1.5),
    "lse_c_sp_d": ("delta_min", -2.0), "av_offset_sp": ("absmax", 3),
    # 4 dialogue. Whisper large-v3 WER on clean read speech ~2-5 %; 0.15 = more than one wrong word in a 7-word line.
    "wer": ("max", 0.15), "wer_d": ("delta_max", 0.10),
    # 5 temporal. Identity to the clip's own first face < 0.5 = the face changed mid-clip. Flicker / temporal
    # high-frequency / warping error: r2 spread 0.82-1.14x, so > 1.25x.
    "id_first_min": ("min", 0.50), "flicker_r": ("max", 1.25), "tflick_r": ("max", 1.25), "warp_err_r": ("max", 1.25),
    "cut": ("true", None),
    # 6 motion. r2 spread of flow 0.85-1.04x, acceleration 0.97-1.13x; outside 0.67-1.5x is a visibly different amount
    # of motion, acceleration +30 % = jerkier. frozen: flow < 0.08 px/frame (the calmest r2 talking head is 0.14) or
    # more than 1 s of near-identical frames (r2 max 0.3 s).
    "flow_mag_r": ("range", (0.67, 1.5)), "accel_r": ("max", 1.3), "frozen": ("true", None),
    # 7 reference image. Frame 0 is conditioned on the reference: SSIM < 0.75 / LPIPS > 0.25 = not kept.
    # Mid-clip DINOv2 similarity to the reference: r2 spread +-0.03, so -0.05.
    "f0_ssim": ("min", 0.75), "f0_lpips": ("max", 0.25), "mid_dino_d": ("delta_min", -0.05),
    # 8 prompt. CLIP score (100 x cosine): r2 spread -1.2..+1.6, so -2.0. Negative margin = the clip matches another
    # prompt better than its own (r2: +20..+28). Camera: static wants zoom within +-7 % and face height 0.85-1.18x;
    # push-in wants zoom > 1.07 or face > 1.15x.
    "clip_score_d": ("delta_min", -2.0), "clip_margin": ("min", 0.0), "camera_ok": ("false", None),
    # 9 quality. Sharpness 0.8-1.25x ts_h (the acceptance band; r2 spread 0.78-1.19). MUSIQ (0-100) r2 spread
    # +-1.7, so -3; CLIP-IQA (0-1) +-0.05, so -0.08; DOVER (0-1) -0.05..+0.01, so -0.08; noise r2 0.93-1.05x, so 1.3x.
    "sharp_face_r": ("range", (0.75, 1.3)), "sharp_frame_r": ("range", (0.8, 1.25)), "musiq_d": ("delta_min", -3.0),
    "clipiqa_d": ("delta_min", -0.08), "dover_d": ("delta_min", -0.08), "noise_r": ("max", 1.3),
}
# Deltas of metrics measured against the reference image are biased towards ts_h when there is no reference image
# (text-only shots use ts_h's own frame 0): not flagged then.
REF_BIASED = {"id_ref_d", "person_ref_d", "mid_dino_d", "f0_ssim", "f0_lpips", "id_ref", "person_ref"}

CRITERIA = [
    ("1 identity", ["id_ref", "id_ref_p05", "id_vs_ref", "id_xclip", "person_ref", "person_vs_ref"]),
    ("2 voice", ["spk_vs_ref", "spk_xclip", "pitch_med", "pitch_sd", "pitch_shift_st", "speech_s"]),
    ("3 lip sync", ["lse_c", "lse_d", "av_offset", "lse_c_sp", "lse_d_sp", "av_offset_sp", "sync_tracks", "lsp_r"]),
    ("4 dialogue", ["wer", "transcript", "errors"]),
    ("5 temporal", ["id_first_min", "id_first_med", "flicker", "tflick", "warp_err", "cut"]),
    ("6 motion", ["flow_mag", "dyn_degree", "accel", "frozen"]),
    ("7 ref image", ["f0_ssim", "f0_lpips", "f0_dino", "f0_hist", "mid_dino", "last_dino", "last_ssim"]),
    ("8 prompt", ["clip_score", "clip_margin", "camera", "zoom", "face_growth", "camera_ok"]),
    ("9 quality", ["sharp_face", "sharp_frame", "musiq", "clipiqa", "dover", "noise"]),
]
FLAG_CRIT = {}
for _c, _ks in CRITERIA:
    for _k in _ks:
        FLAG_CRIT[_k] = _c
for _k in THRESH:  # deltas (_d) and ratios (_r) belong to the criterion of their base metric
    if _k not in FLAG_CRIT and _k[-2:] in ("_d", "_r"):
        FLAG_CRIT[_k] = FLAG_CRIT.get(_k[:-2], "7 ref image" if _k.startswith("mid_dino") else None)
assert all(FLAG_CRIT.get(k) for k in THRESH), [k for k in THRESH if not FLAG_CRIT.get(k)]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def rnd(x, k=3):
    if x is None:
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    x = float(x)
    return round(x, k) if math.isfinite(x) else None


def cos(a, b):
    if a is None or b is None:
        return None
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else None


# --------------------------------------------------------------------------------------------- input
def find_clips(d):
    pat = re.compile(r"^([A-Za-z0-9]+)_(.+)_(\d{5})_\.mp4$")
    best = {}
    for p in glob.glob(os.path.join(d, "*.mp4")):
        m = pat.match(os.path.basename(p))
        if m:
            k = (m.group(1), m.group(2))
            if k not in best or int(m.group(3)) > best[k][0]:
                best[k] = (int(m.group(3)), p)
    out = {}
    for (s, v), (_, p) in best.items():
        out.setdefault(s, {})[v] = p
    return out


def load_video(path):
    c = av.open(path)
    vs = c.streams.video[0]
    fps = float(vs.average_rate)
    frames = [f.to_ndarray(format="rgb24") for f in c.decode(vs)]
    c.close()
    audio, sr = None, None
    c = av.open(path)
    if c.streams.audio:
        st = c.streams.audio[0]
        sr = st.rate
        rs = av.AudioResampler(format="fltp", layout="stereo", rate=sr)
        ch = []
        for fr in c.decode(st):
            ch += [o.to_ndarray() for o in rs.resample(fr)]
        ch += [o.to_ndarray() for o in rs.resample(None)]
        audio = np.concatenate(ch, axis=1).astype(np.float32) if ch else None
    c.close()
    return frames, fps, audio, sr


def resample(x, sr, to):
    from scipy.signal import resample_poly
    if sr == to:
        return x
    g = math.gcd(int(sr), int(to))
    return resample_poly(x, to // g, sr // g, axis=-1).astype(np.float32)


def expected_line(shot):
    """The scripted dialogue: shot['line'], else every quoted string in the prompt's Audio section."""
    if shot.get("line"):
        return shot["line"]
    p = shot.get("prompt", "")
    i = p.find("Audio:")
    if i < 0:
        return None
    q = re.findall(r"[\"“]([^\"”]+)[\"”]", p[i:])
    return " ".join(q) if q else None


def expected_camera(shot, source="prompt"):
    """push-in / static from the prompt's Camera: section (that is what the model saw); shot['camera'] when the
    prompt has none or with --camera-source field."""
    if source == "field" and shot.get("camera"):
        return shot["camera"]
    p = shot.get("prompt", "")
    m = re.search(r"Camera:([^.]*(?:\.[^A-Z][^.]*)*)", p)
    if m:
        t = m.group(1).lower()
        if re.search(r"push-in toward|pushes in|dolly in|zooms? in toward|slow(?:, smooth)? push-in", t) and "no push-in" not in t:
            return "push-in"
        if "static" in t or "holds its framing" in t:
            return "static"
    return shot.get("camera")


def clean_prompt(p):
    p = re.sub(r"Single continuous unbroken shot, no cuts\.\s*", "", p)
    p = re.sub(r"Camera:.*?(?=Audio:|$)", "", p, flags=re.S)
    p = re.sub(r"Audio:.*$", "", p, flags=re.S)
    p = re.sub(r"Timeline:\s*", "", p)
    p = re.sub(r"\[[0-9.s\- ]+\]\s*", "", p)
    return re.sub(r"\s+", " ", p).strip()


# --------------------------------------------------------------------------------------------- models (lazy)
@functools.lru_cache(None)
def face_app():
    from insightface.app import FaceAnalysis
    a = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition", "landmark_3d_68"],
                     providers=["CPUExecutionProvider"])
    a.prepare(ctx_id=-1, det_size=(640, 640))
    return a


@functools.lru_cache(None)
def dino():
    import torch
    from transformers import AutoImageProcessor, AutoModel
    p = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    m = AutoModel.from_pretrained("facebook/dinov2-base").to(DEV).eval()
    return p, m


def dino_emb(imgs):
    """imgs: list of RGB uint8 -> (n, 768) normalised CLS embeddings."""
    import torch
    p, m = dino()
    with torch.no_grad():
        x = p(images=imgs, return_tensors="pt")["pixel_values"].to(DEV)
        e = m(pixel_values=x).pooler_output.float()
        e = torch.nn.functional.normalize(e, dim=-1)
    return e.cpu().numpy()


@functools.lru_cache(None)
def clip_model():
    from transformers import CLIPModel, CLIPProcessor
    name = "openai/clip-vit-large-patch14"
    return CLIPProcessor.from_pretrained(name), CLIPModel.from_pretrained(name).to(DEV).eval()


def clip_text(texts):
    import torch
    p, m = clip_model()
    with torch.no_grad():
        t = p.tokenizer(texts, padding=True, truncation=True, max_length=77, return_tensors="pt").to(DEV)
        e = m.get_text_features(**t)
        e = e if torch.is_tensor(e) else e.pooler_output
        return torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy()


def clip_image(imgs):
    import torch
    p, m = clip_model()
    with torch.no_grad():
        x = p.image_processor(images=imgs, return_tensors="pt")["pixel_values"].to(DEV)
        e = m.get_image_features(pixel_values=x)
        e = e if torch.is_tensor(e) else e.pooler_output
        return torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy()


@functools.lru_cache(None)
def raft():
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    w = Raft_Large_Weights.DEFAULT
    return raft_large(weights=w, progress=False).to(DEV).eval(), w.transforms()


@functools.lru_cache(None)
def iqa(name):
    import pyiqa
    return pyiqa.create_metric(name, device=DEV)


@functools.lru_cache(None)
def dover():
    """DOVER (VQAssessment/DOVER, dover.yml, DOVER.pth). Its dataset module imports decord / skvideo only to read
    files; both are stubbed and the views are built from our decoded frames with DOVER's own samplers."""
    import types, torch, yaml
    for mod in ("decord", "skvideo", "skvideo.io"):
        if mod not in sys.modules:
            m = types.ModuleType(mod)
            m.VideoReader = m.cpu = m.gpu = None
            m.bridge = types.SimpleNamespace(set_bridge=lambda *a: None)
            sys.modules[mod] = m
    sys.modules["skvideo"].io = sys.modules["skvideo.io"]
    if ARGS.dover_dir not in sys.path:
        sys.path.insert(0, ARGS.dover_dir)
    from dover.datasets import UnifiedFrameSampler, get_single_view
    from dover.models import DOVER
    opt = yaml.safe_load(open(os.path.join(ARGS.dover_dir, "dover.yml")))
    net = DOVER(**opt["model"]["args"]).to(DEV).eval()
    net.load_state_dict(torch.load(os.path.join(ARGS.dover_dir, "pretrained_weights/DOVER.pth"), map_location=DEV))
    # the overall score: DOVER's own fuse_results (its score normalization and weights)
    import importlib.util
    one = os.path.join(ARGS.dover_dir, "evaluate_one_video.py")
    spec = importlib.util.spec_from_file_location("dover_eval_one", one)
    fuse = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fuse)
    dopt = opt["data"]["val-l1080p"]["args"]["sample_types"]
    samplers = {}
    for st, so in dopt.items():
        samplers[st] = (UnifiedFrameSampler(so["clip_len"], so["num_clips"], so["frame_interval"]) if "t_frag" not in so else
                        UnifiedFrameSampler(so["clip_len"] // so["t_frag"], so["t_frag"], so["frame_interval"], so["num_clips"]))
    return net, dopt, samplers, get_single_view, fuse.fuse_results


def dover_score(frames):
    """As DOVER's evaluate_one_video.py: technical and aesthetic raw scores, and its fused overall score in [0, 1]."""
    import torch
    import contextlib, io
    net, dopt, samplers, get_single_view, fuse_results = dover()
    mean = torch.FloatTensor([123.675, 116.28, 103.53])
    std = torch.FloatTensor([58.395, 57.12, 57.375])
    views = {}
    np.random.seed(0)
    for st, so in dopt.items():
        idx = samplers[st](len(frames), False)
        v = get_single_view(torch.stack([torch.from_numpy(frames[i]) for i in idx], 0).permute(3, 0, 1, 2), st, **so)
        nc = so.get("num_clips", 1)
        views[st] = (((v.permute(1, 2, 3, 0) - mean) / std).permute(3, 0, 1, 2)
                     .reshape(v.shape[0], nc, -1, *v.shape[2:]).transpose(0, 1).to(DEV))
    with torch.no_grad():
        r = [x.mean().item() for x in net(views)]
    with contextlib.redirect_stdout(io.StringIO()):  # fuse_results prints its logit
        overall = float(fuse_results(r))
    return {"dover": rnd(overall), "dover_tech": rnd(r[0]), "dover_aes": rnd(r[1])}


@functools.lru_cache(None)
def demucs_model():
    from demucs.pretrained import get_model
    m = get_model("htdemucs")
    m.to(DEV).eval()
    return m


@functools.lru_cache(None)
def ecapa():
    from speechbrain.inference.speaker import EncoderClassifier
    return EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                          savedir=os.path.join(ARGS.cache, "spkrec-ecapa-voxceleb"),
                                          run_opts={"device": "cuda:0" if DEV == "cuda" else DEV})


@functools.lru_cache(None)
def whisper():
    from faster_whisper import WhisperModel
    return WhisperModel(ARGS.whisper, device="cpu", compute_type="int8", cpu_threads=ARGS.threads)


@functools.lru_cache(None)
def normalizer():
    try:
        from huggingface_hub import hf_hub_download
        from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
        mapping = json.load(open(hf_hub_download("openai/whisper-large-v3", "normalizer.json")))
        return EnglishTextNormalizer(mapping)
    except Exception as e:  # basic fallback
        log("whisper normaliser unavailable, basic normalisation:", e)
        return lambda s: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]", " ", s.lower())).strip()


@functools.lru_cache(None)
def syncnet():
    import torch
    sp = os.path.join(ARGS.syncnet_dir)
    if sp not in sys.path:
        sys.path.insert(0, sp)
    from SyncNetModel import S
    import detectors.s3fd as s3fd
    s3fd.PATH_WEIGHT = os.path.join(sp, "detectors/s3fd/weights/sfd_face.pth")
    det = s3fd.S3FD(device=DEV)
    net = S(num_layers_in_fc_layers=1024).to(DEV).eval()
    st = torch.load(os.path.join(sp, "data/syncnet_v2.model"), map_location="cpu", weights_only=True)
    own = net.state_dict()
    for k, v in st.items():
        own[k].copy_(v)
    return det, net


# --------------------------------------------------------------------------------------------- faces
def faces_of(rgb):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    out = []
    # at most --max-faces faces (largest and most central, InsightFace's own ranking): a 20-person crowd otherwise
    # costs ~12 min of CPU per clip, and the reference faces are among the largest
    for f in face_app().get(bgr, max_num=ARGS.max_faces):
        lm = f.landmark_3d_68[:, :2]
        yaw = float(f.pose[1]) if getattr(f, "pose", None) is not None else 0.0
        out.append(dict(bbox=f.bbox.astype(np.float32), emb=f.normed_embedding.astype(np.float32), yaw=yaw,
                        mouth=float(np.linalg.norm(lm[66] - lm[62]) / max(np.linalg.norm(lm[8] - lm[27]), 1e-6))))
    return out


def ref_faces(rgb, k=3):
    """Faces of the reference image at least half the height of the largest, largest first (max k)."""
    fs = faces_of(rgb)
    if not fs:
        return []
    hs = [f["bbox"][3] - f["bbox"][1] for f in fs]
    fs = [f for f, h in sorted(zip(fs, hs), key=lambda z: -z[1]) if h >= 0.5 * max(hs)]
    return fs[:k]


def person_crop(rgb, bbox):
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    cx = (x0 + x1) / 2
    H, W = rgb.shape[:2]
    a, b, c, d = int(max(0, cx - 1.5 * w)), int(max(0, y0 - 0.5 * h)), int(min(W, cx + 1.5 * w)), int(min(H, y1 + 2.5 * h))
    return rgb[b:d, a:c] if c - a > 16 and d - b > 16 else None


def lapvar(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def face_gray(rgb, bbox, size=128):
    x0, y0, x1, y1 = bbox
    cx, cy, s = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) / 2
    H, W = rgb.shape[:2]
    a, b, c, d = int(max(0, cx - s)), int(max(0, cy - s)), int(min(W, cx + s)), int(min(H, cy + s))
    if c - a < 8 or d - b < 8:
        return None
    return cv2.resize(cv2.cvtColor(rgb[b:d, a:c], cv2.COLOR_RGB2GRAY), (size, size), interpolation=cv2.INTER_AREA)


def best_corr(m, a, max_lag=8):
    best = (-2.0, 0)
    for lag in range(-max_lag, max_lag + 1):
        x, y = (m[:len(m) - lag], a[lag:]) if lag >= 0 else (m[-lag:], a[:len(a) + lag])
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 20 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
            continue
        best = max(best, (float(np.corrcoef(x[ok], y[ok])[0, 1]), lag))
    return best


def identity_block(frames, ref_rgb, fps, mono, sr):
    """Criterion 1 + face parts of 5, 8, 9 and the lip-sync proxy."""
    n = len(frames)
    rf = ref_faces(ref_rgb)
    per = [faces_of(frames[i]) for i in range(n)]
    R = len(rf)
    out, npz = {"ref_faces": R, "frames_with_face": int(sum(1 for p in per if p))}, {}
    out["max_people"] = int(max((len(p) for p in per), default=0))
    if R == 0:
        return out, npz, per, None
    H = frames[0].shape[0]
    sims = np.full((R, n), np.nan, np.float32)
    match = [[None] * n for _ in range(R)]
    for i, fs in enumerate(per):
        fr = [f for f in fs if abs(f["yaw"]) <= ARGS.max_yaw]
        for r in range(R):
            if not fr:
                continue
            s = [float(f["emb"] @ rf[r]["emb"]) for f in fr]
            j = int(np.argmax(s))
            sims[r, i] = s[j]
            if s[j] > 0.2:
                match[r][i] = fr[j]
    ok = np.isfinite(sims)
    out["id_ref"] = rnd(np.mean([np.nanmedian(sims[r]) for r in range(R) if ok[r].any()])) if ok.any() else None
    out["id_ref_p05"] = rnd(np.mean([np.nanpercentile(sims[r], 5) for r in range(R) if ok[r].any()])) if ok.any() else None
    out["id_ref_frames"] = [rnd(np.nanmean(sims[:, i]) if ok[:, i].any() else None) for i in (0, n // 2, n - 1)]
    # clip-level identity embedding per reference face
    ide = np.zeros((R, 512), np.float32)
    for r in range(R):
        es = [m["emb"] for m in match[r] if m is not None]
        if es:
            e = np.mean(es, 0)
            ide[r] = e / np.linalg.norm(e)
    npz["id_emb"] = ide
    # drift vs the clip's own first matched face
    dr = []
    for r in range(R):
        idx = [i for i in range(n) if match[r][i] is not None]
        if len(idx) < 2:
            continue
        e0 = match[r][idx[0]]["emb"]
        dr.append([float(match[r][i]["emb"] @ e0) for i in idx[1:]])
    if dr:
        out["id_first_min"] = rnd(np.mean([np.percentile(d, 2) for d in dr]))
        out["id_first_med"] = rnd(np.mean([np.median(d) for d in dr]))
    # person crops (hair / clothing): 6 frames
    pe = np.zeros((R, 768), np.float32)
    for r in range(R):
        rc = person_crop(ref_rgb, rf[r]["bbox"])
        idx = [i for i in range(n) if match[r][i] is not None]
        if rc is None or not idx:
            continue
        pick = [idx[int(k)] for k in np.linspace(0, len(idx) - 1, min(6, len(idx)))]
        crops = [c for c in (person_crop(frames[i], match[r][i]["bbox"]) for i in pick) if c is not None]
        if not crops:
            continue
        e = dino_emb([rc] + crops)
        m = e[1:].mean(0)
        pe[r] = m / np.linalg.norm(m)
        out.setdefault("_person", []).append(float(pe[r] @ e[0]))
    if "_person" in out:
        out["person_ref"] = rnd(np.mean(out.pop("_person")))
    npz["person_emb"] = pe
    # face size (camera check), face sharpness: main reference face
    hs = np.array([(m["bbox"][3] - m["bbox"][1]) / H if m is not None else np.nan for m in match[0]])
    k = max(3, n // 7)
    a, b = np.nanmedian(hs[:k]) if np.isfinite(hs[:k]).any() else np.nan, np.nanmedian(hs[-k:]) if np.isfinite(hs[-k:]).any() else np.nan
    out["face_h"] = rnd(np.nanmedian(hs)) if np.isfinite(hs).any() else None
    out["face_growth"] = rnd(b / a) if np.isfinite(a) and np.isfinite(b) and a > 0 else None
    lf = [lapvar(g) for i in range(0, n, 3) if match[0][i] is not None
          for g in [face_gray(frames[i], match[0][i]["bbox"])] if g is not None]
    out["sharp_face"] = rnd(np.median(lf), 1) if lf else None
    # lip-sync proxy: each reference face, best
    if mono is not None:
        hop = sr / fps
        env = np.array([np.sqrt(np.mean(mono[int(i * hop):int((i + 1) * hop)] ** 2) + 1e-12) for i in range(n)])
        best = None
        for r in range(R):
            ms = np.array([m["mouth"] if m is not None else np.nan for m in match[r]])
            if np.isfinite(ms).mean() < 0.3:
                continue
            c, lag = best_corr(ms, env)
            if c > -2 and (best is None or c > best[0]):
                best = (c, lag, best_corr(ms, env, 0)[0])
        if best:
            out["lsp_r"], out["lsp_lag"], out["lsp_r0"] = rnd(best[0]), best[1], rnd(best[2])
    return out, npz, per, match


# --------------------------------------------------------------------------------------------- SyncNet
def iou(a, b):
    xa, ya, xb, yb = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    return inter / float((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def syncnet_score(crops, aud, vs=15, speech=None):
    """SyncNetInstance.evaluate: crops = 224x224 BGR uint8 frames at 25 fps, aud = int16 mono at 16 kHz.
    speech: optional bool per 25-fps frame of the crop; adds lse_c_sp / lse_d_sp over windows centred on speech."""
    import torch
    import python_speech_features
    det, net = syncnet()
    mf = python_speech_features.mfcc(aud, 16000).T  # (13, T)
    im = torch.from_numpy(np.stack(crops, 3)[None].transpose(0, 3, 4, 1, 2).astype(np.float32))
    cc = torch.from_numpy(mf[None, None].astype(np.float32))
    L = min(len(crops), len(aud) // 640)
    last = L - 5
    if last < 10:
        return None
    imf, ccf = [], []
    with torch.no_grad():
        for i in range(0, last, 20):
            ks = range(i, min(last, i + 20))
            imf.append(net.forward_lip(torch.cat([im[:, :, k:k + 5] for k in ks]).to(DEV)).cpu())
            ccf.append(net.forward_aud(torch.cat([cc[:, :, :, k * 4:k * 4 + 20] for k in ks]).to(DEV)).cpu())
    imf, ccf = torch.cat(imf), torch.cat(ccf)
    p2 = torch.nn.functional.pad(ccf, (0, 0, vs, vs))
    d = torch.stack([torch.nn.functional.pairwise_distance(imf[[i]].repeat(2 * vs + 1, 1), p2[i:i + 2 * vs + 1])
                     for i in range(len(imf))], 1)
    md = d.mean(1)
    mv, mi = torch.min(md, 0)
    out = dict(lse_c=float(torch.median(md) - mv), lse_d=float(mv), av_offset=int(vs - mi))
    if speech is not None:
        sp = np.array([bool(speech[min(len(speech) - 1, i + 2)]) for i in range(d.shape[1])])
        if sp.sum() >= 10:
            ms = d[:, torch.from_numpy(sp)].mean(1)
            v2, i2 = torch.min(ms, 0)
            out.update(lse_c_sp=float(torch.median(ms) - v2), lse_d_sp=float(v2), av_offset_sp=int(vs - i2),
                       speech_frames=int(sp.sum()))
    return out


def syncnet_block(frames, fps, mono, sr, v16=None):
    """Adapted from joonson/syncnet_python (MIT, Copyright (c) 2016-present Joon Son Chung; licenses/
    syncnet_python-MIT.txt), as are iou() and syncnet_score() above.
    syncnet_python's run_pipeline + run_syncnet, in memory: 25 fps resample, S3FD (conf 0.9, scale 0.25),
    IoU tracking, median-smoothed 224 px crops (crop_scale 0.4), MFCC at 16 kHz, vshift 15.
    Deviations: no scene detection (single-shot clips), no jpeg/xvid round trip, min track 50 frames, min face 50 px."""
    import torch
    import python_speech_features
    from scipy import signal
    from scipy.interpolate import interp1d
    if mono is None:
        return {}
    det, net = syncnet()
    dur = len(frames) / fps
    n25 = int(math.floor(dur * 25 + 1e-6))
    src = [min(len(frames) - 1, int(math.floor(k * fps / 25 + 1e-6))) for k in range(n25)]
    f25 = [frames[i] for i in src]
    dets = []
    for k, im in enumerate(f25):
        bb = det.detect_faces(im, conf_th=0.9, scales=[0.25])
        dets.append([{"frame": k, "bbox": b[:-1].tolist()} for b in bb])
    tracks, pool = [], [list(d) for d in dets]
    while True:
        tr = []
        for ff in pool:
            for face in list(ff):
                if not tr:
                    tr.append(face); ff.remove(face)
                elif face["frame"] - tr[-1]["frame"] <= 25:
                    if iou(face["bbox"], tr[-1]["bbox"]) > 0.5:
                        tr.append(face); ff.remove(face)
                        continue
                else:
                    break
        if not tr:
            break
        if len(tr) > 50:
            fn = np.array([f["frame"] for f in tr])
            bb = np.array([f["bbox"] for f in tr])
            fi = np.arange(fn[0], fn[-1] + 1)
            bi = np.stack([interp1d(fn, bb[:, j])(fi) for j in range(4)], 1)
            if max(np.mean(bi[:, 2] - bi[:, 0]), np.mean(bi[:, 3] - bi[:, 1])) > 50:
                tracks.append((fi, bi))
    # score only the --max-faces largest tracks (crowds)
    tracks = sorted(tracks, key=lambda t: -np.mean(np.maximum(t[1][:, 2] - t[1][:, 0], t[1][:, 3] - t[1][:, 1])))
    tracks = tracks[:ARGS.max_faces]
    a16 = resample(mono, sr, 16000)
    a16 = np.clip(a16 * 32768.0, -32768, 32767).astype(np.int16)  # syncnet reads int16 wav
    act = None
    if v16 is not None:  # speech activity per 25-fps frame from the vocals stem
        r = np.array([np.sqrt(np.mean(v16[k * 640:(k + 1) * 640] ** 2)) if k * 640 < len(v16) else 0.0 for k in range(n25)])
        act = r > max(0.1 * r.max(), 1e-4)
    res = []
    for fi, bi in tracks:
        s = signal.medfilt(np.maximum(bi[:, 3] - bi[:, 1], bi[:, 2] - bi[:, 0]) / 2, 13)
        y = signal.medfilt((bi[:, 1] + bi[:, 3]) / 2, 13)
        x = signal.medfilt((bi[:, 0] + bi[:, 2]) / 2, 13)
        crops = []
        for j, k in enumerate(fi):
            bs = s[j]
            bsi = int(bs * 1.8)
            bgr = cv2.cvtColor(f25[k], cv2.COLOR_RGB2BGR)
            fr = np.pad(bgr, ((bsi, bsi), (bsi, bsi), (0, 0)), "constant", constant_values=110)
            my, mx = y[j] + bsi, x[j] + bsi
            face = fr[int(my - bs):int(my + bs * 1.8), int(mx - bs * 1.4):int(mx + bs * 1.4)]
            crops.append(cv2.resize(face, (224, 224)))
        aud = a16[int(fi[0] / 25 * 16000):int((fi[-1] + 1) / 25 * 16000)]
        r = syncnet_score(crops, aud, speech=act[fi] if act is not None else None)
        if r:
            r.update(frames=int(len(fi)), face_px=float(np.mean(2 * s)))
            res.append(r)
    out = {"sync_tracks": len(res)}
    if res:
        b = max(res, key=lambda r: r["lse_c"])
        out.update(lse_c=rnd(b["lse_c"]), lse_d=rnd(b["lse_d"]), av_offset=b["av_offset"],
                   lse_c_sp=rnd(b.get("lse_c_sp")), lse_d_sp=rnd(b.get("lse_d_sp")), av_offset_sp=b.get("av_offset_sp"),
                   sync_all=[{k: rnd(v) if isinstance(v, float) else v for k, v in r.items()} for r in res])
    return out


# --------------------------------------------------------------------------------------------- audio
def voice_block(audio, sr, line):
    import torch
    out, npz = {}, {}
    if audio is None:
        return {"audio": False}, npz, None
    mono = audio.mean(0)
    out["audio_rms_db"] = rnd(20 * np.log10(np.sqrt(np.mean(mono ** 2)) + 1e-9), 1)
    # vocals stem
    vocals = None
    try:
        m = demucs_model()
        from demucs.apply import apply_model
        w = torch.from_numpy(resample(audio, sr, m.samplerate))
        ref = w.mean(0)
        mu, sd = ref.mean(), ref.std() + 1e-8
        with torch.no_grad():
            s = apply_model(m, ((w - mu) / sd)[None].to(DEV), device=DEV, split=True, overlap=0.25, progress=False)[0]
        s = (s * sd + mu).cpu().numpy()
        vocals = resample(s[m.sources.index("vocals")].mean(0), m.samplerate, 16000)
        out["sep"] = "htdemucs"
    except Exception as e:
        log("demucs failed, using the mix:", repr(e)[:200])
        out["sep"] = "none"
    m16 = resample(mono, sr, 16000)
    v16 = vocals if vocals is not None else m16
    # active speech: 50 ms RMS above 10 % of the loudest window
    hop = 800
    rms = np.array([np.sqrt(np.mean(v16[i:i + hop] ** 2)) for i in range(0, len(v16) - hop + 1, hop)])
    act = rms > max(0.1 * rms.max(), 1e-4) if len(rms) else np.zeros(0, bool)
    out["speech_s"] = rnd(act.sum() * hop / 16000, 2)
    out["vocal_ratio_db"] = rnd(10 * np.log10((np.mean(v16 ** 2) + 1e-12) / (np.mean(m16 ** 2) + 1e-12)), 1)
    if act.sum() * hop >= 8000:
        seg = np.concatenate([v16[i * hop:(i + 1) * hop] for i in np.nonzero(act)[0]])
        enc = ecapa()
        with torch.no_grad():
            e = enc.encode_batch(torch.from_numpy(seg)[None].to(DEV)).squeeze().float().cpu().numpy()
            em = enc.encode_batch(torch.from_numpy(m16)[None].to(DEV)).squeeze().float().cpu().numpy()
        npz["spk_emb"], npz["spk_emb_mix"] = e / np.linalg.norm(e), em / np.linalg.norm(em)
        import librosa
        f0, vf, _ = librosa.pyin(seg, fmin=60, fmax=600, sr=16000, frame_length=1024)
        f0 = f0[np.isfinite(f0)]
        if len(f0) > 10:
            out["pitch_med"] = rnd(np.median(f0), 1)
            out["pitch_sd"] = rnd(np.std(12 * np.log2(f0 / np.median(f0))), 2)
            out["voiced_frac"] = rnd(len(f0) / max(len(vf), 1), 2)
    # transcript (on the mix: Whisper is robust to ambience; the stem can drop consonants)
    segs, _ = whisper().transcribe(m16, language="en", beam_size=5, condition_on_previous_text=False, vad_filter=False)
    txt = " ".join(s.text.strip() for s in segs).strip()
    out["transcript"] = txt
    out["line"] = line
    norm = normalizer()
    hyp = norm(txt).split()
    if line:
        refw = norm(line).split()
        w, ops = word_align(refw, hyp)
        out["wer"] = rnd(w)
        out["errors"] = ops
    else:
        out["words_heard"] = len(hyp)
    return out, npz, (vocals if vocals is not None else None)


def word_align(ref, hyp):
    n, m = len(ref), len(hyp)
    D = np.zeros((n + 1, m + 1), int)
    D[:, 0], D[0, :] = np.arange(n + 1), np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, ops = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            if ref[i - 1] != hyp[j - 1]:
                ops.append(f"sub {ref[i - 1]}->{hyp[j - 1]}")
            i, j = i - 1, j - 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            ops.append(f"missing {ref[i - 1]}"); i -= 1
        else:
            ops.append(f"inserted {hyp[j - 1]}"); j -= 1
    return D[n, m] / max(n, 1), ops[::-1]


# --------------------------------------------------------------------------------------------- flow
def flow_block(frames, fps):
    """RAFT-large at half resolution: forward/backward flow of every consecutive pair, plus 8-fps pairs."""
    import torch
    import torch.nn.functional as F
    model, tf = raft()
    H, W = frames[0].shape[:2]
    h, w = (H // 2) // 8 * 8, (W // 2) // 8 * 8
    sc = W / w
    small = torch.from_numpy(np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in frames])).permute(0, 3, 1, 2)
    n = len(frames)

    def flows(i1, i2):
        res = []
        with torch.no_grad():
            for k in range(0, len(i1), 8):
                a = small[i1[k:k + 8]].to(DEV).float() / 255
                b = small[i2[k:k + 8]].to(DEV).float() / 255
                a, b = tf(a, b)
                res.append(model(a, b, num_flow_updates=12)[-1].half())
        return torch.cat(res)

    fw = flows(list(range(n - 1)), list(range(1, n)))   # F(t -> t+1), at pixels of t
    bw = flows(list(range(1, n)), list(range(n - 1)))   # F(t+1 -> t), at pixels of t+1
    out = {}
    mag = fw.float().norm(dim=1)                        # (n-1, h, w)
    out["flow_mag"] = rnd(mag.mean().item() * sc)
    per = mag.mean((1, 2)).cpu().numpy() * sc
    # acceleration at the same pixel: F(t->t+1) + F(t->t-1) = 0 for constant velocity
    acc = (fw[1:].float() + bw[:-1].float()).norm(dim=1).mean().item() * sc
    out["accel_px"] = rnd(acc)
    out["accel"] = rnd(acc / max(out["flow_mag"], 1e-3))
    # VBench dynamic degree on ~8 fps pairs
    st = max(1, int(round(fps / 8)))
    idx = list(range(0, n, st))
    f8 = flows(idx[:-1], idx[1:]).float().norm(dim=1) * sc
    top = [torch.topk(m.flatten(), max(1, int(m.numel() * 0.05))).values.mean().item() for m in f8]
    thr = 6.0 * (min(H, W) / 256.0)
    cnt = int(round(4 * (len(idx) / 16.0)))
    out["dyn_degree"] = rnd(np.mean([t > thr for t in top]))
    out["is_dynamic"] = bool(sum(t > thr for t in top) >= cnt)
    # warping error: warp frame t into t+1 with F(t+1 -> t), occlusions masked by forward-backward consistency
    yy, xx = torch.meshgrid(torch.arange(h, device=DEV), torch.arange(w, device=DEV), indexing="ij")
    errs, flick, still = [], [], []
    lum = (0.299 * small[:, 0].float() + 0.587 * small[:, 1].float() + 0.114 * small[:, 2].float())
    for k in range(0, n - 1, 16):
        sl = slice(k, min(n - 1, k + 16))
        b = bw[sl].float()
        f = fw[sl].float()
        gx, gy = xx + b[:, 0], yy + b[:, 1]
        grid = torch.stack([2 * gx / (w - 1) - 1, 2 * gy / (h - 1) - 1], -1)
        src = small[k:k + b.shape[0]].to(DEV).float()
        warped = F.grid_sample(src, grid, align_corners=True, padding_mode="border")
        fwarp = F.grid_sample(f, grid, align_corners=True, padding_mode="border")
        dd = (b + fwarp).pow(2).sum(1)
        occ_ok = dd < 0.01 * (b.pow(2).sum(1) + fwarp.pow(2).sum(1)) + 0.5
        tgt = small[k + 1:k + 1 + b.shape[0]].to(DEV).float()
        e = (tgt - warped).abs().mean(1)
        errs.append(((e * occ_ok).sum((1, 2)) / occ_ok.sum((1, 2)).clamp(min=1)).cpu())
        # flicker on static pixels (|flow| < 0.25 px at analysis res)
        stat = f.norm(dim=1) < 0.25
        dy = (lum[k + 1:k + 1 + b.shape[0]] - lum[k:k + b.shape[0]]).abs().to(DEV)
        flick.append(((dy * stat).sum((1, 2)) / stat.sum((1, 2)).clamp(min=1)).cpu())
        still.append(stat.float().mean((1, 2)).cpu())
    out["warp_err"] = rnd(torch.cat(errs).mean().item())
    sf = torch.cat(still).numpy()
    out["static_frac"] = rnd(sf.mean())
    out["flicker"] = rnd(torch.cat(flick).mean().item()) if sf.mean() > 0.02 else None
    # temporal high-frequency energy on pixels static both ways
    hf = []
    for t in range(1, n - 1):
        m = (fw[t].float().norm(dim=0) < 0.25) & (bw[t - 1].float().norm(dim=0) < 0.25)
        if m.float().mean() > 0.02:
            v = (lum[t] - 0.5 * (lum[t - 1] + lum[t + 1])).abs().to(DEV)
            hf.append(((v * m).sum() / m.sum()).item())
    out["tflick"] = rnd(np.mean(hf)) if hf else None
    # camera zoom: fit F = a (p - c) + t per pair; cumulative scale
    cy, cx = (h - 1) / 2, (w - 1) / 2
    px, py = (xx - cx).float(), (yy - cy).float()
    den = (px ** 2 + py ** 2).sum()
    scale = 1.0
    for k in range(n - 1):
        f = fw[k].float()
        u, v = f[0] - f[0].mean(), f[1] - f[1].mean()
        a = ((u * px).sum() + (v * py).sum()) / den
        scale *= 1 + a.item()
    out["zoom"] = rnd(scale)
    # frozen: little motion or a still stretch longer than 1 s
    dif = np.array([(lum[t + 1] - lum[t]).abs().mean().item() for t in range(n - 1)])
    run, longest = 0, 0
    for d in dif:
        run = run + 1 if d < 0.3 else 0
        longest = max(longest, run)
    out["still_run_s"] = rnd(longest / fps, 2)
    out["frozen"] = bool(out["flow_mag"] < 0.15 or longest / fps > 1.0)
    out["_per_pair_mag"] = per
    return out


# --------------------------------------------------------------------------------------------- per-clip
def fit_ref(ref_rgb, H, W):
    """Centre-crop the reference to the clip's aspect, then resize."""
    h, w = ref_rgb.shape[:2]
    if abs(w / h - W / H) > 0.01:
        if w / h > W / H:
            nw = int(h * W / H); x = (w - nw) // 2; ref_rgb = ref_rgb[:, x:x + nw]
        else:
            nh = int(w * H / W); y = (h - nh) // 2; ref_rgb = ref_rgb[y:y + nh]
    return cv2.resize(ref_rgb, (W, H), interpolation=cv2.INTER_AREA)


def ssim_gray(a, b):
    from skimage.metrics import structural_similarity
    ga = cv2.resize(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    gb = cv2.resize(cv2.cvtColor(b, cv2.COLOR_RGB2GRAY), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    return float(structural_similarity(ga, gb, data_range=255))


def hist_dist(a, b):
    ha = cv2.calcHist([cv2.cvtColor(a, cv2.COLOR_RGB2HSV)], [0, 1, 2], None, [16, 8, 8], [0, 180, 0, 256, 0, 256])
    hb = cv2.calcHist([cv2.cvtColor(b, cv2.COLOR_RGB2HSV)], [0, 1, 2], None, [16, 8, 8], [0, 180, 0, 256, 0, 256])
    return float(cv2.compareHist(cv2.normalize(ha, None), cv2.normalize(hb, None), cv2.HISTCMP_BHATTACHARYYA))


def to_t(imgs):
    import torch
    return torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float().div(255).to(DEV)


def measure_clip(path, shot, ref_rgb, ref_src, prompt_emb, other_embs):
    import torch
    t0 = time.time()
    frames, fps, audio, sr = load_video(path)
    n = len(frames)
    H, W = frames[0].shape[:2]
    rec = {"clip": os.path.basename(path), "frames": n, "fps": rnd(fps, 2), "size": [W, H], "ref_src": ref_src}
    npz = {}
    tm = {}
    mono = audio.mean(0) if audio is not None else None
    ref_fit = fit_ref(ref_rgb, H, W)
    # 1 / 5 / 8 / 9 faces
    t = time.time()
    fb, fnpz, per, match = identity_block(frames, ref_fit, fps, mono, sr)
    rec.update(fb); npz.update(fnpz); tm["faces"] = time.time() - t
    # 2 / 4 audio
    t = time.time()
    vb = voice_block(audio, sr, expected_line(shot))
    rec.update(vb[0]); npz.update(vb[1]); tm["audio"] = time.time() - t
    # 3 SyncNet
    t = time.time()
    try:
        rec.update(syncnet_block(frames, fps, mono, sr, vb[2]))
    except Exception as e:
        log("syncnet failed:", repr(e)[:300]); rec["sync_error"] = repr(e)[:200]
    tm["syncnet"] = time.time() - t
    # 5 / 6 flow
    t = time.time()
    fl = flow_block(frames, fps)
    fl.pop("_per_pair_mag")
    rec.update(fl); tm["flow"] = time.time() - t
    # cut detection (fast_compare: adjacent thumbnail SSIM, every 6th frame)
    from skimage.metrics import structural_similarity
    th = np.array([cv2.resize(cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY), (THUMB, int(THUMB * H / W)), interpolation=cv2.INTER_AREA)
                   for i in range(n)])
    s6 = th[::6].astype(np.float32)
    adj = [structural_similarity(s6[k], s6[k + 1], data_range=255) for k in range(len(s6) - 1)]
    med = float(np.median(adj))
    rec["adj_min"] = rnd(min(adj))
    rec["cut_frames"] = [6 * (k + 1) for k, a in enumerate(adj) if a < med - 0.25]
    rec["cut"] = bool(rec["cut_frames"])
    npz["thumbs"] = th
    # 7 reference fidelity
    t = time.time()
    sel = {"f0": 0, "mid": n // 2, "last": n - 1}
    lp = iqa("lpips")
    de = dino_emb([ref_fit] + [frames[i] for i in sel.values()])
    for j, (k, i) in enumerate(sel.items()):
        rec[f"{k}_ssim"] = rnd(ssim_gray(frames[i], ref_fit))
        with torch.no_grad():
            rec[f"{k}_lpips"] = rnd(lp(to_t([frames[i]]), to_t([ref_fit])).item())
        rec[f"{k}_dino"] = rnd(float(de[j + 1] @ de[0]))
        rec[f"{k}_hist"] = rnd(hist_dist(frames[i], ref_fit))
    npz["dino_frames"] = de[1:]
    npz["key_frames"] = np.stack([cv2.resize(frames[i], (W // 2, H // 2), interpolation=cv2.INTER_AREA) for i in sel.values()])
    # 8 prompt adherence
    pick = [int(round(x)) for x in np.linspace(0, n - 1, 8)]
    ce = clip_image([frames[i] for i in pick])
    rec["clip_score"] = rnd(100 * float((ce @ prompt_emb).mean()), 2)
    if other_embs is not None and len(other_embs):
        rec["clip_margin"] = rnd(rec["clip_score"] - 100 * float((ce @ other_embs.T).mean()), 2)
    # 9 quality
    with torch.no_grad():
        x = to_t([frames[i] for i in pick])
        rec["musiq"] = rnd(iqa("musiq")(x).mean().item(), 2)
        rec["clipiqa"] = rnd(iqa("clipiqa")(x).mean().item())
    if not ARGS.no_dover:
        try:
            rec.update(dover_score(frames))
        except Exception as e:
            log("dover failed:", repr(e)[:300])
    from skimage.restoration import estimate_sigma
    rec["sharp_frame"] = rnd(np.mean([lapvar(cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)) for i in pick]), 1)
    rec["noise"] = rnd(np.mean([estimate_sigma(cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY).astype(np.float32)) for i in pick]))
    tm["rest"] = time.time() - t
    rec["timing_s"] = {k: rnd(v, 1) for k, v in tm.items()}
    rec["timing_s"]["total"] = rnd(time.time() - t0, 1)
    return rec, npz


def cached_measure(path, shot, ref_rgb, ref_src, prompt_emb, other_embs, cache_dir):
    st = os.stat(path)
    key = f"v{CODE_VER}_{st.st_size}_{int(st.st_mtime)}_{ref_src}"
    base = os.path.join(cache_dir, os.path.basename(path)[:-4])
    if not ARGS.force and os.path.exists(base + ".json") and os.path.exists(base + ".npz"):
        rec = json.load(open(base + ".json"))
        if rec.get("_key") == key:
            if not ARGS.no_dover and "dover" not in rec:  # added after a clip was cached: fill it in
                try:
                    rec.update(dover_score(load_video(path)[0]))
                    with open(base + ".json.tmp", "w") as fh:
                        json.dump(rec, fh, indent=1)
                    os.replace(base + ".json.tmp", base + ".json")
                except Exception as e:
                    log("dover failed:", repr(e)[:300])
            return rec, dict(np.load(base + ".npz")), False
    rec, npz = measure_clip(path, shot, ref_rgb, ref_src, prompt_emb, other_embs)
    rec["_key"] = key
    np.savez_compressed(base + ".npz", **npz)
    with open(base + ".json.tmp", "w") as fh:
        json.dump(rec, fh, indent=1)
    os.replace(base + ".json.tmp", base + ".json")
    return rec, npz, True


# --------------------------------------------------------------------------------------------- comparisons
def compare(shots_data, chars):
    """Adds vs-reference-variant metrics, deltas, ratios and cross-clip (same character) similarities."""
    REF = ARGS.ref_variant
    for sid, vs in shots_data.items():
        ref = vs.get(REF)
        for v, (rec, npz) in vs.items():
            if ref is None or v == REF:
                continue
            rr, rn = ref
            def pair(key):
                a, b = npz.get(key), rn.get(key)
                if a is None or b is None or a.shape != b.shape:
                    return None
                ss = [cos(a[i], b[i]) for i in range(len(a))] if a.ndim == 2 else [cos(a, b)]
                ss = [s for s in ss if s is not None and abs(s) > 0]
                return rnd(np.mean(ss)) if ss else None
            rec["id_vs_ref"] = pair("id_emb")
            rec["person_vs_ref"] = pair("person_emb")
            rec["spk_vs_ref"] = pair("spk_emb")
            rec["spk_mix_vs_ref"] = pair("spk_emb_mix")
            if rec.get("pitch_med") and rr.get("pitch_med") and min(rec.get("voiced_frac") or 0, rr.get("voiced_frac") or 0) >= 0.25:
                rec["pitch_shift_st"] = rnd(12 * math.log2(rec["pitch_med"] / rr["pitch_med"]), 2)
            if "dino_frames" in npz and "dino_frames" in rn:
                rec["mid_dino_vs_ref"] = rnd(float(npz["dino_frames"][1] @ rn["dino_frames"][1]))
                rec["last_dino_vs_ref"] = rnd(float(npz["dino_frames"][2] @ rn["dino_frames"][2]))
            if "thumbs" in npz and "thumbs" in rn:
                from skimage.metrics import structural_similarity
                a, b = npz["thumbs"].astype(np.float32), rn["thumbs"].astype(np.float32)
                k = min(len(a), len(b))
                rec["comp_ssim"] = rnd(np.mean([structural_similarity(a[j], b[j], data_range=255) for j in range(0, k, 6)]))
            for key in ("id_ref", "person_ref", "lse_c", "lse_d", "lse_c_sp", "lse_d_sp", "wer", "clip_score", "musiq", "clipiqa", "dover", "mid_dino",
                        "f0_ssim", "id_first_min"):
                if rec.get(key) is not None and rr.get(key) is not None:
                    rec[key + "_d"] = rnd(rec[key] - rr[key])
            for key in ("flicker", "tflick", "warp_err", "flow_mag", "accel", "sharp_face", "sharp_frame", "noise"):
                if rec.get(key) is not None and rr.get(key):
                    rec[key + "_r"] = rnd(rec[key] / rr[key])
    # cross-clip, same character, same variant
    groups = {}
    for sid, vs in shots_data.items():
        c = chars.get(sid)
        if c is None:
            continue
        for v, (rec, npz) in vs.items():
            groups.setdefault((c, v), []).append((rec, npz))
    for (c, v), items in groups.items():
        for i, (rec, npz) in enumerate(items):
            for key, name in (("id_emb", "id_xclip"), ("person_emb", "person_xclip"), ("spk_emb", "spk_xclip")):
                ss = []
                for j, (_, o) in enumerate(items):
                    if j == i or key not in npz or key not in o:
                        continue
                    a, b = npz[key], o[key]
                    s = cos(a[0], b[0]) if a.ndim == 2 else cos(a, b)
                    if s is not None and abs(s) > 0:
                        ss.append(s)
                rec[name] = rnd(np.mean(ss)) if ss else None
            rec["char"] = c


def camera_check(rec, cam):
    """static: flow zoom within +-7 % and face height within 0.85-1.18x; push-in: zoom > 1.07 or face > 1.15x."""
    rec["camera"] = cam
    rec.pop("camera_ok", None)
    if cam:
        z, fg = rec.get("zoom"), rec.get("face_growth")
        if cam == "static":
            rec["camera_ok"] = bool(z is not None and 0.93 <= z <= 1.07 and (fg is None or 0.85 <= fg <= 1.18))
        else:
            rec["camera_ok"] = bool((z is not None and z > 1.07) or (fg is not None and fg > 1.15))


def flags_for(rec, is_ref, has_line):
    fl = []
    rec["frozen"] = bool((rec.get("flow_mag") is not None and rec["flow_mag"] < 0.08) or (rec.get("still_run_s") or 0) > 1.0)
    no_ref_img = str(rec.get("ref_src", "")).endswith("_frame0")
    for key, (kind, th) in THRESH.items():
        if is_ref and (key.endswith("_d") or key.endswith("_r") or key.endswith("vs_ref") or key == "pitch_shift_st"):
            continue
        if no_ref_img and key in REF_BIASED:
            continue
        if key.startswith(("wer", "lse", "av_offset", "spk", "pitch")) and not has_line:
            continue  # no scripted speech: speech metrics informative only
        v = rec.get(key)
        if v is None:
            continue
        bad = ((kind in ("min", "delta_min") and v < th) or (kind in ("max", "delta_max") and v > th)
               or (kind == "absmax" and abs(v) > th) or (kind == "range" and not th[0] <= v <= th[1])
               or (kind == "true" and v is True) or (kind == "false" and v is False))
        if bad:
            fl.append(key)
    return fl


# --------------------------------------------------------------------------------------------- report
def fmt(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.3g}" if abs(v) < 100 else f"{v:.0f}"
    if isinstance(v, list):
        return "; ".join(map(str, v))[:80]
    return str(v).replace("|", "/")[:80]


DELTA_OF = {k: k + "_d" for k in ("id_ref", "person_ref", "lse_c", "lse_d", "lse_c_sp", "lse_d_sp", "wer", "clip_score", "musiq", "clipiqa",
                                  "dover", "mid_dino", "f0_ssim", "id_first_min")}
DELTA_OF.update({k: k + "_r" for k in ("flicker", "tflick", "warp_err", "flow_mag", "accel", "sharp_face", "sharp_frame", "noise")})


def report(shots_data, out_dir, title):
    REF = ARGS.ref_variant
    variants = sorted({v for vs in shots_data.values() for v in vs}, key=lambda v: (v != REF, v))
    md = [f"# {title}: {len(shots_data)} shots, variants {', '.join(variants)} (reference {REF})", "",
          "Cells: value, and for non-reference variants the change vs the reference in brackets "
          "(difference `+x`, or ratio `x1.05` for scale-free metrics). Flags use the thresholds in "
          "`harness/eval_clips.py` THRESH.", ""]
    # overview
    md += ["## Overview: flagged clips per criterion", "",
           "| criterion | " + " | ".join(variants) + " |", "|---|" + "---|" * len(variants)]
    for crit, _ in CRITERIA:
        row = []
        for v in variants:
            recs = [vs[v][0] for vs in shots_data.values() if v in vs]
            nf = sum(1 for r in recs if any(FLAG_CRIT.get(f) == crit for f in r.get("flags", [])))
            row.append(f"{nf}/{len(recs)}")
        md.append(f"| {crit} | " + " | ".join(row) + " |")
    md.append("")
    # means
    md += ["## Means over shots (non-reference: mean change vs reference)", ""]
    for crit, keys in CRITERIA:
        num = [k for k in keys if k not in ("transcript", "errors", "camera", "sync_tracks")]
        md += [f"**{crit}**", "", "| variant | " + " | ".join(num) + " |", "|---|" + "---|" * len(num)]
        for v in variants:
            cells = []
            for k in num:
                vals = [vs[v][0].get(k) for vs in shots_data.values() if v in vs]
                vals = [float(x) for x in vals if isinstance(x, (int, float, bool)) and x is not None]
                c = fmt(float(np.mean(vals))) if vals else ""
                dk = DELTA_OF.get(k)
                if v != REF and dk:
                    dv = [vs[v][0].get(dk) for vs in shots_data.values() if v in vs]
                    dv = [x for x in dv if x is not None]
                    if dv:
                        c += f" ({np.mean(dv):+.3g})" if dk.endswith("_d") else f" (x{np.mean(dv):.3g})"
                cells.append(c)
            md.append(f"| {v} | " + " | ".join(cells) + " |")
        md.append("")
    # per criterion, per shot
    for crit, keys in CRITERIA:
        md += [f"## {crit}", "", "| shot | variant | " + " | ".join(keys) + " | flags |", "|---|---|" + "---|" * (len(keys) + 1)]
        for sid in sorted(shots_data):
            vs = shots_data[sid]
            for v in variants:
                if v not in vs:
                    continue
                r = vs[v][0]
                cells = []
                for k in keys:
                    c = fmt(r.get(k))
                    dk = DELTA_OF.get(k)
                    if v != REF and dk and r.get(dk) is not None:
                        c += f" (x{r[dk]:.2f})" if dk.endswith("_r") else f" ({r[dk]:+.3g})"
                    cells.append(c)
                f = [x for x in r.get("flags", []) if FLAG_CRIT.get(x) == crit]
                md.append(f"| {sid} | {v} | " + " | ".join(cells) + f" | {'FLAG ' + ', '.join(f) if f else 'pass'} |")
        md.append("")
    md += ["## Thresholds", "", "| metric | flag when | value |", "|---|---|---|"]
    for k, (kind, th) in THRESH.items():
        md.append(f"| {k} | {kind} | {th} |")
    open(os.path.join(out_dir, "summary.md"), "w").write("\n".join(md) + "\n")


# --------------------------------------------------------------------------------------------- main
def main():
    global ARGS, DEV
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", required=True, help="directory of <shot>_<variant>_<nnnnn>_.mp4")
    ap.add_argument("--shots", required=True, help="shots JSON (id, prompt, first, line, camera, char)")
    ap.add_argument("--ref-dir", default="data/input", help="reference first-frame images (shot['first'])")
    ap.add_argument("--ref-variant", default="ts_h")
    ap.add_argument("--variants", nargs="*", help="only these variants (the reference is always included)")
    ap.add_argument("--only", nargs="*", help="only these shot ids")
    ap.add_argument("--out", default=None, help="output dir (default <clips>/eval)")
    ap.add_argument("--char-regex", default=None, help="character group from the shot id when 'char' is missing")
    ap.add_argument("--whisper", default="large-v3-turbo")
    ap.add_argument("--syncnet-dir", default=os.path.expanduser("~/rapid-h3/third_party/syncnet_python"))
    ap.add_argument("--dover-dir", default=os.path.expanduser("~/rapid-h3/third_party/DOVER"))
    ap.add_argument("--no-dover", action="store_true")
    ap.add_argument("--cache", default=_C)
    ap.add_argument("--camera-source", choices=["prompt", "field"], default="prompt",
                    help="expected camera move: parsed from the prompt (default) or the shots' 'camera' field")
    ap.add_argument("--max-yaw", type=float, default=45.0)
    ap.add_argument("--max-faces", type=int, default=4, help="faces analysed per frame (largest / most central)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true", help="re-measure cached clips")
    ap.add_argument("--workers", type=int, default=1, help="parallel measuring processes (2 suits a shared Spark)")
    ap.add_argument("--measure-only", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--title", default=None)
    ARGS = a = ap.parse_args()
    DEV = a.device
    out = a.out or os.path.join(a.clips, "eval")
    cdir = os.path.join(out, "clips")
    os.makedirs(cdir, exist_ok=True)
    shots = {s["id"]: s for s in json.load(open(a.shots))}
    clips = find_clips(a.clips)
    chars = {}
    for sid, s in shots.items():
        if s.get("char"):
            chars[sid] = s["char"]
        elif a.char_regex:
            m = re.match(a.char_regex, sid)
            if m:
                chars[sid] = m.group(1)
    mism = [sid for sid, s in shots.items() if s.get("camera") and expected_camera(s) and s["camera"] != expected_camera(s)]
    if mism:
        log(f"note: 'camera' field disagrees with the prompt's Camera: section for {', '.join(mism)}; "
            f"using the {a.camera_source} (--camera-source)")
    # prompt embeddings
    ids = [sid for sid in shots if sid in clips and (not a.only or sid in a.only)]
    if a.workers > 1 and not a.measure_only:
        # measure in parallel subprocesses (disjoint shot subsets, shared cache), then aggregate here from the cache
        import subprocess
        argv = [x for x in sys.argv[1:]]
        if "--only" in argv:  # drop the parent's --only and its values
            i = argv.index("--only"); j = i + 1
            while j < len(argv) and not argv[j].startswith("--"):
                j += 1
            argv = argv[:i] + argv[j:]
        procs = []
        for w in range(a.workers):
            part = sorted(ids)[w::a.workers]
            if part:
                lf = open(os.path.join(out, f"worker{w}.log"), "w")
                procs.append(subprocess.Popen([sys.executable, os.path.abspath(__file__)] + argv +
                                              ["--measure-only", "--only"] + part, stdout=lf, stderr=subprocess.STDOUT))
                time.sleep(20)  # stagger model loading
        rc = [p.wait() for p in procs]
        log(f"workers done, exit codes {rc}; logs in {out}/worker*.log")
        a.force = False  # the workers re-measured; aggregate from their cache
    texts = {sid: clean_prompt(shots[sid]["prompt"]) for sid in shots}
    uniq = sorted(set(texts.values()))
    temb = dict(zip(uniq, clip_text(uniq)))
    data = {}
    for sid in sorted(ids):
        s = shots[sid]
        vs = clips[sid]
        if a.variants:
            vs = {v: p for v, p in vs.items() if v in a.variants or v == a.ref_variant}
        ref_img = s.get("first") or s.get("ref_image")  # Ref2VA shots: the reference image (f0_* then mean little)
        ref_png = os.path.join(a.ref_dir, ref_img) if ref_img else None
        if ref_png and os.path.exists(ref_png):
            ref_rgb, ref_src = cv2.cvtColor(cv2.imread(ref_png), cv2.COLOR_BGR2RGB), os.path.basename(ref_png)
        elif a.ref_variant in vs:
            f, *_ = load_video(vs[a.ref_variant])
            ref_rgb, ref_src = f[0], f"{a.ref_variant}_frame0"
            if ref_png:
                log(f"{sid}: {ref_png} missing, using {ref_src}")
        else:
            log(f"{sid}: no reference image and no {a.ref_variant} clip, skipped")
            continue
        pe = temb[texts[sid]]
        others = np.array([temb[t] for t in uniq if t != texts[sid]])
        for v in sorted(vs, key=lambda v: (v != a.ref_variant, v)):
            rec, npz, fresh = cached_measure(vs[v], s, ref_rgb, ref_src, pe, others, cdir)
            rec.update(shot=sid, variant=v)
            camera_check(rec, expected_camera(s, a.camera_source))
            data.setdefault(sid, {})[v] = (rec, npz)
            log(f"{sid} {v}: {'measured in ' + str(rec['timing_s']['total']) + ' s' if fresh else 'cached'}")
    if a.measure_only:
        return
    compare(data, chars)
    for sid, vs in data.items():
        for v, (rec, _) in vs.items():
            rec["flags"] = flags_for(rec, v == a.ref_variant, bool(expected_line(shots[sid])))
    with open(os.path.join(out, "shots.json"), "w") as fh:
        json.dump({sid: {v: {k: x for k, x in r.items() if k != "_key"} for v, (r, _) in vs.items()}
                   for sid, vs in data.items()}, fh, indent=1)
    report(data, out, a.title or os.path.basename(os.path.normpath(a.clips)))
    log("wrote", os.path.join(out, "summary.md"), "and shots.json")


if __name__ == "__main__":
    main()
