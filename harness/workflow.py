"""Build a ComfyUI API graph for one MiniMax-H3 render: FL2VA (text only, first / last frame, input audio, anchors)
or Ref2VA (reference images / audios / videos).

The graph is the stock Comfy H3 chain plus optional knobs; every knob defaults to the
reference (pruned bf16 DiT, PyTorch attention, no fused modulation, no quantization).
The release presets are PRESETS["ts_h2"] and PRESETS["ts_s6"] below.
"""

DIT = {
    "bf16": "minimax_h3_fl2va_pruned_bf16.safetensors",
    "fp8_scaled": "minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
    "int8_convrot": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    # Ref2VA (reference images / videos / audio): in the models folder's diffusion_models/ if present there,
    # otherwise the bf16 file mounted by serve.sh RAPIDH3_REF2VA
    "ref2va_bf16": "minimax_h3_ref2va_pruned_bf16.safetensors",
    "ref2va_int8_convrot": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
}
REF2VA = {"bf16": "ref2va_bf16", "int8_convrot": "ref2va_int8_convrot"}  # dit when references are given
TURBO8 = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
TEXT_ENCODER = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"


VIDEO_VAE = {"fp16": "minimax_h3_video_vae_fp16.safetensors",
             "int8": "minimax_h3_video_vae_int8_convrot.safetensors"}


def _flow_sigma(shift, percent):
    """ComfyUI's ModelSamplingDiscreteFlow.percent_to_sigma (comfy/model_sampling.py, H3 = ModelSamplingAV): the sigma
    at a schedule percent for the model's video shift; BlockSparseAttention's start/end_percent go through it."""
    if percent <= 0.0:
        return 1.0
    if percent >= 1.0:
        return 0.0
    t = 1.0 - percent
    return shift * t / (1 + (shift - 1) * t)


# Named render settings (add prompt, size, length, seed, first_frame): build(prompt, ..., **PRESETS["ts_h2"]).
# ts_h2 is the default since 2026-10-24 (README.md); ts_s6 is the camera/close-up-safe preset (slower);
# ts_h is the 2026-10-24 default before ts_h2 (two refine steps), kept for rollback.
_RAPIDH3 = dict(quant="int8_rot", attention="comfy kitchen attention", video_vae="int8", turbo8=False,
             sampler="res_multistep", scheduler="simple")
# Since 2026-10-26 ts_h2 and ts_s6 use our INT8 attention kernel config 94 (bit-identical to config 80, more accurate
# than comfy-kitchen's) and ts_h2 encodes the conditioning once (encode_once="draft"); both checked against ts_h2 on
# eval1 (draft4: h2eo, h2a80). Config 194 = 94 with the q/k RMSNorm + RoPE fused into its pre-pass
# (bit-identical final latents). Since 2026-10-26 (side-by-side review: "exactly same") both presets also run the
# Qwen3-VL text encode on our INT8 path (te_quant="int8_rot"). Since 2026-10-27 both use config 190 (= 194 with the
# softmax mutex entered after the row max: bit-identical outputs, 1.8–2.4% faster per call; attn_cake/DESIGN.md §7);
# rapidh3_attn=194 runs the previous kernel. The *_ck presets are the settings before 2026-10-26.
_H2 = dict(_RAPIDH3, steps=6, draft="0.7", refine_sigmas="0.72, 0", freeze_audio=True, decode_save="overlap")
PRESETS = {
    "ts_h2": dict(_H2, encode_once="draft", rapidh3_attn=190, te_quant="int8_rot"),
    "ts_h": dict(_RAPIDH3, steps=6, draft="0.7", refine_sigmas="0.78, 0.45, 0", freeze_audio=True),
    # one stage at full size, 6 steps: stock-level camera moves, motion and close-up faces (motion2), ≈201 s per clip
    "ts_s6": dict(_RAPIDH3, steps=6, decode_save="overlap", rapidh3_attn=190, te_quant="int8_rot"),
    "ts_h2_ck": _H2,
    "ts_s6_ck": dict(_RAPIDH3, steps=6, decode_save="overlap"),
}

def build(prompt, *, seed=0, width=832, height=480, length=124, turbo8=True, steps=None, video_vae="fp16",
          dit="bf16", attention="pytorch attention", fused_mod=False, quant="bf16",
          quant_targets="qkv_proj,out_proj,fc1,fc2", keep_bf16_blocks="", measure_every=0, fuse_block=True,
          shadow=False, shadow_sdpa="default", dump_qkv=False, profile=False, vae2=None, prefix="rapid-h3",
          timer=True, sparse="", sparse_tau=1.3, sparse_start=0.2, sampler="euler", scheduler="simple",
          first_frame=None, draft=None, refine_sigmas="", shift=None, shift_audio=3.0, freeze_audio=False, draft_sigmas="",
          draft_pdd="", draft_pdd_nfe=4, draft_pdd_unet="", decode_save="", trt_vae="", pyopt_vae="",
          draft_lora=None, draft_lora_strength=1.0, draft_lora_steps=None, draft_lora_sigmas=None,
          draft_lora_shift=None, draft_lora_shift_audio=None, draft_lora_sampler=None, draft_lora_scheduler=None,
          draft_lora_stack_turbo=False, lora=None, lora_strength=1.0,
          sparse_stage="all",
          refine_prompt="", rapidh3_attn=None, rapidh3_attn_exact="", last_frame=None, audio_input=None, audio_start=0, audio_hold=True,
          audio_pad="none", audio_floor_db=-55.0, encode_once=False, upscale_lean=False, te_quant="",
          pin_latent=None, pin_frames="0-21", pin_audio="none", pin_region=None, anchors=None,
          anchors_refine=True,
          shadow_reference="bf16", shadow_follow="reference", shadow_mid_attention="keep",
          ref_images=(), ref_audios=(), ref_videos=(), ref_video_audio=True, ref_image_size="match",
          fis_stride=0, fis_blocks="10-39", fis_min_sigma=0.8,
          asym_q=0.0, asym_kv=0.0, asym_window="1,2,2", asym_blocks="5-44", asym_min_sigma=0.8,
          bottleneck=None, bneck_full=1, bneck_steps=None, bneck_shift=2.0, bneck_refine="", bneck_up="learned",
          bneck_down="area", attn_mass="", attn_mass_blocks="0,5,12,20,28,36,44,49", attn_mass_steps="0-5",
          draft_text_bias=None):
    """first_frame / last_frame: input image names (fl2va: first, last, or both).
    audio_input: an audio file name in the input folder's top level (LoadAudio lists only that); it is encoded with the
    H3 audio VAE into the latent's audio from pixel frame audio_start and held by a per-token mask (RapidH3InputAudio),
    so the clip keeps exactly that track and the video is generated to it (a2v). Two-stage: put into the draft (the
    refine keeps the draft's audio with freeze_audio). audio_pad="silence" / "floor": hold the whole clip's audio,
    the track padded with silence / a pink-noise floor at audio_floor_db, so nothing is generated around the line.
    pin_latent: a .latent (RapidH3SaveLatent) whose pixel frames pin_frames ("a-b", inclusive) are copied in and held
    (RapidH3PinLatent); pin_audio="same" also holds their audio; pin_region (a mask image, white = regenerate) turns the
    pin into inpainting. One stage only (the pinned latent must be at the sampled size).
    anchors: guides at any pixel frame via stock MiniMaxH3AddGuide (re-injected condition tokens, not held latent;
    no strength knob): a list of {"frame": idx (negative = from the end), and "image": input image, or "video": input
    video ("start"/"frames" pick frames of it; the node keeps 1 or 5/22/39... frames), and/or "audio": input audio
    ("audio_s": (start_s, duration_s) slice), placed from that frame}. Two-stage: added to the draft and the refine
    conditioning (like last_frame); anchors_refine=False: the draft only. Video/audio files must sit at the input
    folder's top level (LoadVideo / LoadAudio list only that).
    draft="WxH": two-stage render, the full schedule at the draft size,
    the video latent x2 by the H3 latent upscaler, then refine_sigmas ("s0, s1, ..., 0") at width x height;
    freeze_audio keeps the draft's audio as it is through the refine; draft_sigmas ("1, ..., 0") replaces the draft's
    scheduler. shift overrides the video sigma shift.
    draft_pdd (two-stage only): a PDD Acc file (MiniMaxH3PDDAccApply, models/pdd_acc) distils the draft only:
    the draft model is UNET -> PDD apply -> shift 12/3 -> attention -> quant, sampled with euler on the node's
    trained sigma grid (draft_pdd_nfe = 4, 6 or 8 steps, CFG 1); the refine keeps the base model.
    draft_pdd_unet: a DiT with the PDD trunk LoRA already baked in (the pack's bake_pdd_trunk.py) for the draft;
    the apply node then runs at lora_strength 0 (heads only), so no LoRA is re-applied on each model switch.
    draft_lora (two-stage only): a LoRA file (models/loras) for the draft only, e.g. a 4-step distillation (PDMD, DMAD):
    the draft model is UNET -> LoraLoaderModelOnly(draft_lora, draft_lora_strength) -> shift -> attention -> quant,
    REPLACING the Turbo LoRA on the draft (draft_lora_stack_turbo=True puts it on top of Turbo instead); the refine
    keeps the main model (base, or base + Turbo with turbo8). The draft's schedule defaults to the preset's;
    draft_lora_steps / draft_lora_scheduler (BasicScheduler on the draft model), draft_lora_sigmas ("1, ..., 0",
    ManualSigmas, wins), draft_lora_shift / draft_lora_shift_audio (the draft model's shifts; DMAD wants audio 2) and
    draft_lora_sampler override it.
    Both models are clones of the one UNETLoader, so ComfyUI holds the base weights once and re-patches them with the
    other LoRA at each draft <-> refine switch (p4c_r1 measured +7.5 s on the first draft step); the draft's own
    RapidH3LinearQuant node keeps a second int8 copy of the DiT (~21 GB), so run it with dit="int8_convrot" (p4c_r1: peak
    87 GB) rather than the bf16 DiT (p4_r1: ~5 GB free, the DiT re-read from disk every step).
    lora: a LoRA file (models/loras) on the main model, e.g. DMAD as a one-stage full-size preset (dmfull1): UNET ->
    [Turbo] -> LoraLoaderModelOnly(lora, lora_strength) -> shift -> attention -> quant; sampler / scheduler / steps /
    shift / shift_audio set its schedule (DMAD: lcm, simple, 4, 12, 2). Run it with dit="int8_convrot" (the LoRA is
    patched into the dequantized weights at load time; the quant node keeps its int8 copy of the patched model).
    sparse_stage ("all" | "draft" | "refine"): which model carries the BlockSparseAttention patch of sparse.
    "all": every sampled step (one patch node before quant as before). Two-stage "draft" / "refine": the chosen
    stage's model only (a patch node after the main chain, so the one quant node and its int8 copy serve both stages;
    a draft_pdd / draft_lora draft carries it in its own chain). The node turns sparse_start / end_percent into a sigma
    window with the model's shift (percent_to_sigma: shift * (1 - p) / (1 + (shift - 1) * (1 - p)); shift 12: 0.2 ->
    0.98, 0.5 -> 0.92, 0.8 -> 0.75) and compares each step's sigma to it, not its position in the sampler's list,
    so the refine (first sigma 0.72) is sparse from its first step at any sparse_start <= 0.82; "refine" raises if
    the window would leave the refine's first step dense. One stage: "all" and "refine" apply, "draft" does nothing.
    decode_save: replace VAEDecode -> CreateVideo -> SaveVideo by RapidH3DecodeSaveVideo, which writes the mp4 while
    the VAE still decodes: "overlap" (the same file as SaveVideo, bit for bit), "serial" (same node, no overlap,
    for timing).
    refine_prompt (two-stage only): a different text for the full-size refine's conditioning (timing check).
    encode_once (two-stage only): the draft's conditioning without its own MiniMaxH3ImageToVideo, which
    otherwise means two Qwen3-VL passes, the full-size one deferred by ComfyUI into the draft -> refine gap.
    True: RapidH3DraftConditioning derives it from the full-size node "10": text-only, the same conditioning (the text
    tokens do not depend on the canvas: bit-identical clip); with first / last frame, the keyframe latents re-encoded
    at the draft size but the Qwen hidden states of the full-size image (more vision tokens: the draft changes).
    "order": both nodes kept, RapidH3After makes ComfyUI run the full-size one before the draft (bit-identical clip; the
    encodes run back to back instead of one in the gap). "draft": the reverse of True, the refine's conditioning
    derived from the draft-size node (keyframes re-encoded at full size, the draft's Qwen hidden states; the draft
    unchanged, the refine changes). Not with refine_prompt (True / "draft"). With Ref2VA references (True / "draft"):
    RapidH3RefConditioning reuses the other stage's MiniMaxH3ReferenceToVideo conditioning when it is exactly what this
    stage's own node computes (every reference image sized the same at both canvases: crops smaller than the draft
    area, ref_image_size="max", or audio / video references only), else runs this stage's own node (a lazy input):
    the clip is always the same as the two-encode graph's.
    te_quant ("int8_rot" in ts_h2 / ts_s6 since 2026-10-26): "int8_rot" / "int8_rot_plain" / "bf16_w8" puts RapidH3TextEncoderQuant after the CLIPLoader: the
    Qwen3-VL LM linears on our INT8 W8A8 ConvRot GEMM (or bf16 on the int8 weights) and bf16 attention instead of
    ComfyUI's fp32 path (custom_nodes/rapid_h3/te_quant.py); changes the conditioning slightly (harness/te_compare.py).
    upscale_lean (two-stage only): RapidH3LatentUpscale3D instead of MinimaxH3LatentUpscaler3D, the same
    upscaler call without its soft_empty_cache / gc.collect (bit-identical latent, shorter draft -> refine gap).
    trt_vae: decode the video with a TensorRT engine (ComfyUI-H3VAE_TRT's MiniMaxH3TRTVAELoader, file in models/vae)
    instead of the int8 VAE; encoding the first frame still uses the int8 VAE.
    pyopt_vae: decode the video with ComfyUI-H3VAE-PyOpt's H3VAEPyOptLoader instead of the int8 VAE:
    "fp16" (fp16 weights, whole-decoder torch.compile, tile 256 / batch 2) or "int8" (+ its int8_decode FFNs, which
    need comfy-kitchen 0.2.34, i.e. not our Rapid-H3 server). Needs a server started with RAPIDH3_H3VAE_PYOPT=1 (serve.sh);
    the first-frame encode keeps the int8 VAE. With decode_save the node falls back to overlap=False (the runtime
    has no _finalize_pixels), the mp4 is the same.
    Approximate-compute research patches (off by default, applied to the one
    model, active only while sigma >= *_min_sigma, i.e. the draft's steps 1-5 at the default 0.8):
    fis_stride > 0: RapidH3FrameSkip (FIS-DiT-style: every fis_stride-th latent frame computed in fis_blocks);
    asym_q / asym_kv > 0: RapidH3AsymRnR (query / key-value token reduction in asym_blocks' attention).
    shadow_reference / shadow_follow / shadow_mid_attention: RapidH3Shadow's optional modes (see the node).
    ref_images / ref_audios / ref_videos (Ref2VA): lists of input file names (<= 9 images, <= 3 audios,
    <= 3 videos of 2-15 s at 24 fps); the prompt calls them <Picture i> / <Audio j> / <Video k>, ordinals per type in
    list order. The conditioning node becomes MiniMaxH3ReferenceToVideo (draft and refine both get the references)
    and dit bf16 / int8_convrot switches to the Ref2VA checkpoint. ref_video_audio: each reference video's soundtrack
    goes in too (its <Audio j> label comes before its <Video k>, before the standalone ref_audios). ref_image_size:
    "match" (scaled down to the sampled area, so a ts_h2 draft sees smaller references than the refine) or "max"
    (2048 short edge, slower). No first/last frame with references. Measured (ref1 / ref2, 2026-10-26): with
    full-frame reference images (a whole eval shot, also a two-person frame) 6 steps can break the subject (garbled
    face, a duplicated person) in bf16 too, steps=10 fixes it; ts_h2 pastes full-frame references (ref_image_size
    "max" does not help, a crop smaller than the draft area does). A reference video (adapt_canvas size, e.g. 5 s at
    768x1344) costs ≈2.6x per step (ts_s6 81 s/step); a half-size file ≈1.3x. A close-up portrait reference video
    makes the clip continue the reference shot (stock bf16 too).
    rapidh3_attn_exact (with rapidh3_attn): RapidH3Int8Attention exact_rows, e.g. "ref_audio,audio": those packed-layout
    segments' query rows get bf16 SDPA attention (Ref2VA voice references; turns the 1xx rms+rope fusion off).
    bottleneck (one-stage presets such as ts_s6): high-low-high sampling (Bottleneck
    Sampling, arXiv 2503.18940) on the preset's step grid t_i = 1 - i/steps: bneck_full full-size steps (sigmas at
    the video shift, "15a"), their last x0 estimate (denoised_output) area-downscaled to bottleneck x the size
    ("0.75": the draft rounding) and re-noised by the sampler at the next grid point, bneck_steps (default: the rest
    but one) steps at that size on the model with video shift + bneck_shift ("15b"; the audio shift is kept, so the
    audio sees the same base grid), their last x0 upscaled back (bneck_up "learned": MinimaxH3LatentUpscaler3D, or an
    interpolation mode of LatentUpscale such as "bicubic") and re-noised to bneck_refine ("s0, ..., 0"; default: the
    grid's remaining sigmas at the video shift, ts_s6: "0.706, 0") for the final full-size step(s) ("15"). Re-noising
    uses the seed's noise (at full size the same noise as step 1). The audio rides along and is re-noised from its x0
    at each boundary like the video. The reduced stage's conditioning is RapidH3DraftConditioning of "10" (no second
    Qwen pass; keyframes re-encoded at the reduced size).
    draft_text_bias (two-stage only): rho of the Jenga text-attention bias on the draft
    model only (RapidH3TextAttnBias "6j" in front of the draft guider; the refine is untouched): the draft's attention
    logits on the text span get + rho * ln(sqrt(full / draft tokens per frame)), against the crop-in to close-ups of
    small drafts. Needs the fused INT8 attention (rapidh3_attn >= 100, quant int8_rot, fuse_block), as in ts_h2."""
    refs = bool(ref_images or ref_audios or ref_videos)
    if refs:
        if first_frame or last_frame:
            raise ValueError("Ref2VA takes no first/last frame (MiniMaxH3ReferenceToVideo has no keyframe input)")
        dit = REF2VA.get(dit, dit)
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": DIT[dit], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": TEXT_ENCODER, "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE[video_vae]}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
    }
    model = ["1", 0]
    if turbo8:
        g["5"] = {"class_type": "LoraLoaderModelOnly",
                  "inputs": {"model": model, "lora_name": TURBO8, "strength_model": 1.0}}
        model = ["5", 0]
    if lora:  # one-stage LoRA on the main model (docstring)
        g["5m"] = {"class_type": "LoraLoaderModelOnly",
                   "inputs": {"model": model, "lora_name": lora, "strength_model": lora_strength}}
        model = ["5m", 0]
    # shift_audio: the audio stream rides the video sigma through its own flow shift (ModelSamplingAV), so a lower
    # audio shift gives the audio lower noise levels, i.e. finer late steps, at the same video schedule
    shift_video = shift or (6.0 if turbo8 else 12.0)
    g["6"] = {"class_type": "MiniMaxH3SigmaShift",
              "inputs": {"model": model, "shift_video": shift_video, "shift_audio": shift_audio}}
    model = ["6", 0]
    if attention != "pytorch attention":
        g["6a"] = {"class_type": "ModelAttentionBackend", "inputs": {"model": model, "attention": attention}}
        model = ["6a", 0]
    if rapidh3_attn is not None:  # opt-in: our INT8 attention kernel (config rapidh3_attn) for self-attention
        g["6i"] = {"class_type": "RapidH3Int8Attention", "inputs": {"model": model, "config": rapidh3_attn}}
        if rapidh3_attn_exact:  # segment kinds whose query rows get bf16 attention (docstring)
            g["6i"]["inputs"]["exact_rows"] = rapidh3_attn_exact
        model = ["6i", 0]
    if sparse_stage not in ("all", "draft", "refine"):
        raise ValueError(f"sparse_stage {sparse_stage!r}: all, draft or refine")

    def sparse_node(m):  # ComfyUI's BlockSparseAttention (sol-attn / sla / vsa); approximate attention
        return {"class_type": "BlockSparseAttention",
                "inputs": {"model": m, "selection": sparse, "selection.tau": sparse_tau,
                           "start_percent": sparse_start, "end_percent": 1.0, "dense_blocks": "",
                           "min_tokens": 12288, "extra_tokens": 256,
                           "sink_conditioning": "exact_kv_and_rows", "verbose": True}}

    # one stage: "refine" is the one stage; two-stage "draft" / "refine" patch one stage's model further down
    sparse_main = sparse and (sparse_stage == "all" or (sparse_stage == "refine" and not draft))
    if sparse_main:
        g["6s"] = sparse_node(model)
        model = ["6s", 0]
    if fused_mod:
        g["6b"] = {"class_type": "MiniMaxH3FusedModulation", "inputs": {"model": model, "enabled": True}}
        model = ["6b", 0]
    if quant != "bf16":
        g["6c"] = {"class_type": "RapidH3LinearQuant",
                   "inputs": {"model": model, "mode": quant, "targets": quant_targets,
                              "keep_bf16_blocks": keep_bf16_blocks, "measure_every": measure_every,
                              "fuse_block": fuse_block}}
        model = ["6c", 0]
    if fis_stride:
        g["6x"] = {"class_type": "RapidH3FrameSkip",
                   "inputs": {"model": model, "stride": fis_stride, "blocks": fis_blocks, "min_sigma": fis_min_sigma}}
        model = ["6x", 0]
    if asym_q or asym_kv:
        g["6y"] = {"class_type": "RapidH3AsymRnR",
                   "inputs": {"model": model, "q_ratio": asym_q, "kv_ratio": asym_kv, "window": asym_window,
                              "blocks": asym_blocks, "min_sigma": asym_min_sigma, "sim_heads": 8}}
        model = ["6y", 0]
    if timer:  # synced per-step wall time (render.py reads it; tqdm s/it under-reports)
        g["6t"] = {"class_type": "RapidH3StepTimer", "inputs": {"model": model}}
        model = ["6t", 0]
    if profile:
        g["6g"] = {"class_type": "RapidH3Profile", "inputs": {"model": model, "step": 3}}
        model = ["6g", 0]
    if attn_mass:  # per-head block attention mass -> output/<attn_mass>/ (sparse_attn.RapidH3AttnMass)
        g["6m"] = {"class_type": "RapidH3AttnMass", "inputs": {"model": model, "blocks": attn_mass_blocks,
                                                             "steps": attn_mass_steps, "out_dir": attn_mass}}
        model = ["6m", 0]
    if dump_qkv:
        g["6f"] = {"class_type": "RapidH3DumpQKV", "inputs": {"model": model, "blocks": "5,25,45", "steps": "1,5"}}
        model = ["6f", 0]
    if shadow:
        g["6e"] = {"class_type": "RapidH3Shadow", "inputs": {"model": model, "follow": shadow_follow, "candidate_sdpa": shadow_sdpa}}
        if shadow_reference != "bf16":
            g["6e"]["inputs"].update(reference=shadow_reference, mid_attention=shadow_mid_attention)
        model = ["6e", 0]
    # two-stage, one stage sparse: the patch goes on top of the finished chain (a block patch on a clone; its place in
    # the chain does not matter), so the quant node and its int8 copy serve both stages; the chosen guider takes it
    sparse_one = model
    if sparse and draft and not sparse_main and not (sparse_stage == "draft" and (draft_pdd or draft_lora)):
        sparse_one = ["6s", 0]
        g["6s"] = sparse_node(model)
        if sparse_stage == "refine":  # the window is in sigma: the refine's first step must sit inside it
            s0, s_start = float(refine_sigmas.split(",")[0]), _flow_sigma(shift_video, sparse_start)
            if s0 > s_start:
                raise ValueError(f"sparse_stage=refine: sparse_start {sparse_start} is sigma {s_start:.3f} at shift "
                                 f"{shift_video}, above the refine's first sigma {s0} -> the refine would run dense")

    def draft_chain(dm, nid, shift_v, shift_a=3.0):
        """The alternate draft model's chain after its LoRA / PDD node: shift -> attention -> [sparse] -> quant -> timer
        (the main chain's optional research patches are left out), node ids nid, nid+a, nid+s, nid+c, nid+t."""
        g[nid] = {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": dm, "shift_video": shift_v, "shift_audio": shift_a}}
        dm = [nid, 0]
        if attention != "pytorch attention":
            g[nid + "a"] = {"class_type": "ModelAttentionBackend", "inputs": {"model": dm, "attention": attention}}
            dm = [nid + "a", 0]
        if sparse and sparse_stage in ("all", "draft"):
            g[nid + "s"] = sparse_node(dm)
            dm = [nid + "s", 0]
        if quant != "bf16":
            g[nid + "c"] = {"class_type": "RapidH3LinearQuant",
                            "inputs": {"model": dm, "mode": quant, "targets": quant_targets,
                                       "keep_bf16_blocks": keep_bf16_blocks, "measure_every": measure_every,
                                       "fuse_block": fuse_block}}
            dm = [nid + "c", 0]
        if timer:
            g[nid + "t"] = {"class_type": "RapidH3StepTimer", "inputs": {"model": dm}}
            dm = [nid + "t", 0]
        return dm

    first = {} if not first_frame else {"first_frame": ["9", 0]}
    if first_frame:
        g["9"] = {"class_type": "LoadImage", "inputs": {"image": first_frame}}
    if last_frame:
        g["9l"] = {"class_type": "LoadImage", "inputs": {"image": last_frame}}
        first["last_frame"] = ["9l", 0]
    refs_in = {}
    for i, name in enumerate(ref_images):
        g[f"9i{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        refs_in[f"ref_images.ref_image_{i}"] = [f"9i{i}", 0]
    for i, name in enumerate(ref_videos):
        g[f"9v{i}"] = {"class_type": "LoadVideo", "inputs": {"file": name}}
        g[f"9w{i}"] = {"class_type": "GetVideoComponents", "inputs": {"video": [f"9v{i}", 0]}}
        refs_in[f"ref_videos.ref_video_{i}"] = [f"9w{i}", 0]
        if ref_video_audio:
            refs_in[f"ref_video_audios.ref_video_audio_{i}"] = [f"9w{i}", 1]
    for i, name in enumerate(ref_audios):
        g[f"9ra{i}"] = {"class_type": "LoadAudio", "inputs": {"audio": name}}
        refs_in[f"ref_audios.ref_audio_{i}"] = [f"9ra{i}", 0]

    def cond_node(w, h):
        if not refs:
            return {"class_type": "MiniMaxH3ImageToVideo",
                    "inputs": {"clip": ["2", 0], "vae": ["3", 0], "prompt": prompt,
                               "width": w, "height": h, "length": length, **first}}
        return {"class_type": "MiniMaxH3ReferenceToVideo",
                "inputs": {"clip": ["2", 0], "vae": ["3", 0], "audio_vae": ["4", 0], "prompt": prompt,
                           "width": w, "height": h, "length": length, "ref_image_size": ref_image_size, **refs_in}}

    g.update({
        "10": cond_node(width, height),
        "11": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "12": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": sampler}},
        "13": {"class_type": "BasicScheduler",
               "inputs": {"model": model, "scheduler": scheduler,
                          "steps": steps or (8 if turbo8 else 50), "denoise": 1.0}},
        "14": {"class_type": "BasicGuider", "inputs": {"model": sparse_one if sparse_stage == "refine" else model,
                                                      "conditioning": ["10", 0]}},
        "15": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["11", 0], "guider": ["14", 0], "sampler": ["12", 0],
                          "sigmas": ["13", 0], "latent_image": ["10", 1]}},
        "16": {"class_type": "VAEDecode", "inputs": {"samples": ["15", 0], "vae": ["3", 0]}},
        "17": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["15", 0], "vae": ["4", 0]}},
        "22": {"class_type": "CreateVideo",
               "inputs": {"images": ["16", 0], "fps": 24.0, "audio": ["17", 0],
                          "bit_depth": "auto", "color_space": "sRGB"}},
        "23": {"class_type": "SaveVideo",
               "inputs": {"video": ["22", 0], "filename_prefix": prefix, "format": "mp4"}},
        **({} if not vae2 else {
            # the same latent decoded by a second video VAE, saved as variant "<variant>_vae2"
            "3b": {"class_type": "VAELoader", "inputs": {"vae_name": vae2}},
            "16b": {"class_type": "VAEDecode", "inputs": {"samples": ["15", 0], "vae": ["3b", 0]}},
            "22b": {"class_type": "CreateVideo", "inputs": {"images": ["16b", 0], "fps": 24.0, "audio": ["17", 0],
                                                           "bit_depth": "auto", "color_space": "sRGB"}},
            "23b": {"class_type": "SaveVideo", "inputs": {"video": ["22b", 0], "format": "mp4",
                                                          "filename_prefix": prefix.replace("/", "_vae2/", 1)}},
        }),
        # codec-free copy of the final latent for exact comparisons
        "24": {"class_type": "RapidH3SaveLatent", "inputs": {"samples": ["15", 0], "filename_prefix": prefix}},
    })
    if trt_vae:
        g["3t"] = {"class_type": "MiniMaxH3TRTVAELoader", "inputs": {"decoder": trt_vae, "encoder": "None"}}
        g["16"]["inputs"]["vae"] = ["3t", 0]
    elif pyopt_vae:
        g["3t"] = {"class_type": "H3VAEPyOptLoader", "inputs": {
            "vae_name": VIDEO_VAE["fp16"], "dtype": "fp16", "decoder_tile_size": 256, "tile_batch": 2,
            "compile_decoder": True, "compile_encoder": False, "encoder_staged_batch": 4,
            "cudnn_benchmark": False,   # the node would otherwise flip torch.backends.cudnn.benchmark for the DiT too
            "warmup": "none", "warmup_frames": 124, "warmup_width": 1344, "warmup_height": 768, "log_calls": True,
            "encoder_tile_size": 0, "fast_linear": False, "decode_fusions": False,
            "int8_decode": {"fp16": False, "int8": True}[pyopt_vae], "int8_encode": False,
            "model_code_dir": "/work/h3_video_vae/video_vae", "weights_path": ""}}
        g["16"]["inputs"]["vae"] = ["3t", 0]
    if decode_save:
        overlap = {"overlap": True, "serial": False}[decode_save]
        for k in ("16", "22", "23"):
            g.pop(k)
        g["23"] = {"class_type": "RapidH3DecodeSaveVideo",
                   "inputs": {"samples": ["15", 0], "vae": ["3t", 0] if trt_vae or pyopt_vae else ["3", 0], "fps": 24.0,
                              "filename_prefix": prefix, "overlap": overlap, "audio": ["17", 0]}}
    if draft:  # two-stage: "15" becomes the full-size refine of the upscaled draft
        if "x" not in draft:  # a scale such as "0.7" (PRESETS): the same rounding as fast_test.draft_size
            f = 0.5 if draft == "half" else float(draft)
            draft = f"{round(width * f / 32) * 32}x{round(height * f / 32) * 32}"
        dw, dh = (int(v) for v in draft.split("x"))
        if encode_once not in (False, True, "order", "draft"):
            raise ValueError(f"encode_once {encode_once!r}: False, True, 'order' or 'draft'")
        if encode_once in (True, "draft") and refine_prompt:
            raise ValueError(f"encode_once={encode_once!r}: one Qwen pass for both prompts; use 'order'")
        g.update({
            "10d": cond_node(dw, dh) if encode_once is not True or refs else {
                "class_type": "RapidH3DraftConditioning",
                "inputs": {"conditioning": ["10", 0], "vae": ["3", 0], "width": dw, "height": dh, "length": length,
                           **first}},
            "14d": {"class_type": "BasicGuider", "inputs": {"model": sparse_one if sparse_stage == "draft" else model,
                                                           "conditioning": ["10d", 0]}},
            "15d": {"class_type": "SamplerCustomAdvanced",
                    "inputs": {"noise": ["11", 0], "guider": ["14d", 0], "sampler": ["12", 0],
                               "sigmas": ["13d", 0] if draft_sigmas else ["13", 0], "latent_image": ["10d", 1]}},
            "15s": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["15d", 1]}},
            "15u": {"class_type": "MinimaxH3LatentUpscaler3D",
                    "inputs": {"latent": ["15s", 0], "model_name": "minimax_h3_latent_upscaler_3d_fp16.safetensors",
                               "mode": "target dimensions", "mode.width": width, "mode.height": height,
                               "align": 32, "enable_temporal_chunking": False, "force_unload": False,
                               "device": "cuda", "precision": "fp16"}},
            "15j": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["15u", 0],
                                                                  "audio_latent": ["15f", 0] if freeze_audio else ["15s", 1]}},
            "13r": {"class_type": "ManualSigmas", "inputs": {"sigmas": refine_sigmas}},
        })
        if upscale_lean:
            u = g["15u"]["inputs"]
            g["15u"] = {"class_type": "RapidH3LatentUpscale3D",
                        "inputs": {"latent": u["latent"], "model_name": u["model_name"], "width": width, "height": height,
                                   "align": u["align"], "enable_temporal_chunking": u["enable_temporal_chunking"],
                                   "device": u["device"], "precision": u["precision"]}}
        def ref_reuse(src, other, w, h):  # Ref2VA: src's conditioning if identical at w x h, else other's (lazy)
            return {"class_type": "RapidH3RefConditioning",
                    "inputs": {"conditioning": [src, 0], "other": [other, 0], "width": w, "height": h,
                               "length": length, "ref_image_size": ref_image_size,
                               **{k.split(".", 1)[1]: v for k, v in refs_in.items() if k.startswith("ref_images.")}}}

        if refs and encode_once is True:
            g["10e"] = g["10d"]
            g["10d"] = ref_reuse("10", "10e", dw, dh)
        elif refs and encode_once == "draft":
            g["10f"] = g["10"]
            g["10"] = ref_reuse("10d", "10f", width, height)
        elif encode_once == "draft":  # the refine's conditioning from the draft-size node's
            g["10"] = {"class_type": "RapidH3DraftConditioning",
                       "inputs": {"conditioning": ["10d", 0], "vae": ["3", 0], "width": width, "height": height,
                                  "length": length, **first}}
        if encode_once == "order":  # the draft waits for the full-size conditioning (no tensor changes)
            g["10o"] = {"class_type": "RapidH3After", "inputs": {"value": ["10d", 1], "after": ["10", 0]}}
            g["15d"]["inputs"]["latent_image"] = ["10o", 0]
        if draft_sigmas:
            g["13d"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": draft_sigmas}}
        if freeze_audio:
            g["15f"] = {"class_type": "LTXVFreezeLatent", "inputs": {"latent": ["15s", 1]}}
        g["15"]["inputs"].update({"sigmas": ["13r", 0], "latent_image": ["15j", 0]})
        if refine_prompt:
            g["10"]["inputs"]["prompt"] = refine_prompt
        if draft_pdd and draft_lora:
            raise ValueError("draft_pdd and draft_lora both replace the draft model; give one")
        if draft_pdd:
            if draft_pdd_unet:
                g["1p"] = {"class_type": "UNETLoader", "inputs": {"unet_name": draft_pdd_unet, "weight_dtype": "default"}}
            g["5p"] = {"class_type": "MiniMaxH3PDDAccApply",
                       "inputs": {"model": ["1p", 0] if draft_pdd_unet else ["1", 0], "pdd_file": draft_pdd,
                                  "nfe": str(draft_pdd_nfe), "lora_strength": 0.0 if draft_pdd_unet else 1.0,
                                  "head_strength": 1.0, "on_off_grid": "error"}}
            dm = draft_chain(["5p", 0], "6p", 12.0)
            g["12p"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
            g["14d"]["inputs"]["model"] = dm
            g["15d"]["inputs"].update({"sampler": ["12p", 0], "sigmas": ["5p", 1]})
        if draft_lora:
            # the draft's LoRA on the one UNETLoader ("1", or its Turbo clone "5" when stacking): a second ModelPatcher
            # clone of the same model, so the base weights load once; ComfyUI unpatches / re-patches the LoRA weights
            # at each draft <-> refine switch and the draft's quant node ("6lc") caches its own int8 copy (docstring)
            g["5l"] = {"class_type": "LoraLoaderModelOnly",
                       "inputs": {"model": ["5", 0] if turbo8 and draft_lora_stack_turbo else ["1", 0],
                                  "lora_name": draft_lora, "strength_model": draft_lora_strength}}
            dm = draft_chain(["5l", 0], "6l", draft_lora_shift or shift_video, draft_lora_shift_audio or 3.0)
            g["14d"]["inputs"]["model"] = dm
            if draft_lora_sigmas:
                g["13l"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": draft_lora_sigmas}}
                g["15d"]["inputs"]["sigmas"] = ["13l", 0]
            elif (draft_lora_steps or draft_lora_scheduler or draft_lora_shift) and not draft_sigmas:
                # the schedule follows the draft model (its shift), not "13" on the main model
                g["13l"] = {"class_type": "BasicScheduler",
                            "inputs": {"model": dm, "scheduler": draft_lora_scheduler or scheduler,
                                       "steps": draft_lora_steps or steps or (8 if turbo8 else 50), "denoise": 1.0}}
                g["15d"]["inputs"]["sigmas"] = ["13l", 0]
            if draft_lora_sampler:
                g["12l"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": draft_lora_sampler}}
                g["15d"]["inputs"]["sampler"] = ["12l", 0]
        if draft_text_bias:
            if not (rapidh3_attn is not None and rapidh3_attn >= 100 and quant == "int8_rot" and fuse_block):
                raise ValueError("draft_text_bias: needs rapidh3_attn >= 100, quant int8_rot and fuse_block (ts_h2)")
            g["6j"] = {"class_type": "RapidH3TextAttnBias",
                       "inputs": {"model": g["14d"]["inputs"]["model"], "rho": float(draft_text_bias),
                                  "ratio": round(width * height / (dw * dh), 6)}}
            g["14d"]["inputs"]["model"] = ["6j", 0]
    elif draft_text_bias:
        raise ValueError("draft_text_bias: two-stage (draft) graphs only")
    if bottleneck:  # high-low-high: "15a" full size -> "15b" reduced -> "15" full size (docstring)
        if draft or refs or anchors or audio_input or pin_latent or sparse:
            raise ValueError("bottleneck: one-stage graphs without draft / refs / anchors / audio_input / pin / sparse")
        f = float(bottleneck)
        dw, dh = round(width * f / 32) * 32, round(height * f / 32) * 32
        n = steps or (8 if turbo8 else 50)
        k, m = bneck_full, bneck_steps or n - bneck_full - 1
        if k < 1 or m < 1 or k + m >= n:
            raise ValueError(f"bottleneck: {k} full + {m} reduced steps leave no final step of {n}")
        grid = [1.0 - i / n for i in range(n + 1)]

        def sig(shift_v, ts):
            return ", ".join(f"{shift_v * t / (1 + (shift_v - 1) * t):.6f}" for t in ts)

        refine = bneck_refine or sig(shift_video, grid[k + m:])
        g.update({
            "13a": {"class_type": "ManualSigmas", "inputs": {"sigmas": sig(shift_video, grid[:k + 1])}},
            "15a": {"class_type": "SamplerCustomAdvanced",
                    "inputs": {"noise": ["11", 0], "guider": ["14", 0], "sampler": ["12", 0],
                               "sigmas": ["13a", 0], "latent_image": ["10", 1]}},
            "15as": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["15a", 1]}},
            # LatentUpscale divides its pixel size by 8; the H3 video latent is /16
            "15ad": {"class_type": "LatentUpscale", "inputs": {"samples": ["15as", 0], "upscale_method": bneck_down,
                                                               "width": dw // 2, "height": dh // 2, "crop": "disabled"}},
            "15aj": {"class_type": "LTXVConcatAVLatent",
                     "inputs": {"video_latent": ["15ad", 0], "audio_latent": ["15as", 1]}},
            "10d": {"class_type": "RapidH3DraftConditioning",
                    "inputs": {"conditioning": ["10", 0], "vae": ["3", 0], "width": dw, "height": dh,
                               "length": length, **first}},
            "6r": {"class_type": "MiniMaxH3SigmaShift",
                   "inputs": {"model": model, "shift_video": shift_video + bneck_shift, "shift_audio": shift_audio}},
            "14b": {"class_type": "BasicGuider", "inputs": {"model": ["6r", 0], "conditioning": ["10d", 0]}},
            # the last value only sets the (unused) noisy output; x0 of the last evaluated step goes on
            "13b": {"class_type": "ManualSigmas",
                    "inputs": {"sigmas": sig(shift_video + bneck_shift, grid[k:k + m]) + ", " + refine.split(",")[0].strip()}},
            "15b": {"class_type": "SamplerCustomAdvanced",
                    "inputs": {"noise": ["11", 0], "guider": ["14b", 0], "sampler": ["12", 0],
                               "sigmas": ["13b", 0], "latent_image": ["15aj", 0]}},
            "15bs": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["15b", 1]}},
            "15bu": {"class_type": "MinimaxH3LatentUpscaler3D",
                     "inputs": {"latent": ["15bs", 0], "model_name": "minimax_h3_latent_upscaler_3d_fp16.safetensors",
                                "mode": "target dimensions", "mode.width": width, "mode.height": height,
                                "align": 32, "enable_temporal_chunking": False, "force_unload": False,
                                "device": "cuda", "precision": "fp16"}} if bneck_up == "learned" else
                    {"class_type": "LatentUpscale", "inputs": {"samples": ["15bs", 0], "upscale_method": bneck_up,
                                                               "width": width // 2, "height": height // 2,
                                                               "crop": "disabled"}},
            "15bj": {"class_type": "LTXVConcatAVLatent",
                     "inputs": {"video_latent": ["15bu", 0], "audio_latent": ["15bs", 1]}},
            "13r": {"class_type": "ManualSigmas", "inputs": {"sigmas": refine}},
        })
        g["15"]["inputs"].update({"sigmas": ["13r", 0], "latent_image": ["15bj", 0]})
    if anchors:  # MiniMaxH3AddGuide chain on every conditioning (two-stage: draft and refine, like last_frame)
        guides = []
        for i, a in enumerate(anchors):
            src = {}
            if a.get("image"):
                g[f"9g{i}"] = {"class_type": "LoadImage", "inputs": {"image": a["image"]}}
                src["image"] = [f"9g{i}", 0]
            elif a.get("video"):
                g[f"9g{i}"] = {"class_type": "LoadVideo", "inputs": {"file": a["video"]}}
                g[f"9gc{i}"] = {"class_type": "GetVideoComponents", "inputs": {"video": [f"9g{i}", 0]}}
                src["image"] = [f"9gc{i}", 0]
                if "frames" in a:  # take frames [start, start + frames) of the file; the node crops to 5/22/39...
                    g[f"9gb{i}"] = {"class_type": "ImageFromBatch",
                                    "inputs": {"image": [f"9gc{i}", 0], "batch_index": a.get("start", 0),
                                               "length": a["frames"]}}
                    src["image"] = [f"9gb{i}", 0]
            if a.get("audio"):
                g[f"9ga{i}"] = {"class_type": "LoadAudio", "inputs": {"audio": a["audio"]}}
                src["audio"] = [f"9ga{i}", 0]
                if "audio_s" in a:  # (start_s, duration_s) slice of the file
                    g[f"9gt{i}"] = {"class_type": "TrimAudioDuration",
                                    "inputs": {"audio": [f"9ga{i}", 0], "start_index": a["audio_s"][0],
                                               "duration": a["audio_s"][1]}}
                    src["audio"] = [f"9gt{i}", 0]
            guides.append((a["frame"], src))
        stages = (("10", "14"),) if not draft else (("10", "14"), ("10d", "14d")) if anchors_refine else (("10d", "14d"),)
        for cond, guider in stages:
            pos = [cond, 0]
            for i, (frame, src) in enumerate(guides):
                nid = f"{cond}g{i}"
                g[nid] = {"class_type": "MiniMaxH3AddGuide",
                          "inputs": {"positive": pos, "latent": [cond, 1], "frame_idx": frame, "vae": ["3", 0],
                                     "audio_vae": ["4", 0], **src}}
                pos = [nid, 0]
            g[guider]["inputs"]["conditioning"] = pos
    if audio_input or pin_latent:
        if pin_latent and draft:
            raise ValueError("pin_latent works on one-stage graphs (ts_s6); the draft samples at another size")
        sampler_node = "15d" if draft else "15"
        src = g[sampler_node]["inputs"]["latent_image"]
        if audio_input:
            g["9a"] = {"class_type": "LoadAudio", "inputs": {"audio": audio_input}}
            g["10a"] = {"class_type": "RapidH3InputAudio", "inputs": {"latent": src, "audio_vae": ["4", 0], "audio": ["9a", 0],
                                                                "start_frame": audio_start, "hold": audio_hold}}
            if audio_pad != "none":
                g["10a"]["inputs"]["pad"] = audio_pad
                if audio_pad == "floor":
                    g["10a"]["inputs"]["floor_db"] = audio_floor_db
            src = ["10a", 0]
        if pin_latent:
            a, b = (int(v) for v in pin_frames.split("-"))
            g["9p"] = {"class_type": "RapidH3LoadAVLatent", "inputs": {"path": pin_latent}}
            g["10p"] = {"class_type": "RapidH3PinLatent", "inputs": {"latent": src, "source": ["9p", 0], "start_frame": a,
                                                               "end_frame": b, "audio": pin_audio}}
            if pin_region:
                g["9r"] = {"class_type": "LoadImageMask", "inputs": {"image": pin_region, "channel": "red"}}
                g["10p"]["inputs"]["region"] = ["9r", 0]
            src = ["10p", 0]
        g[sampler_node]["inputs"]["latent_image"] = src
    if te_quant:
        g["2q"] = {"class_type": "RapidH3TextEncoderQuant", "inputs": {"clip": ["2", 0], "mode": te_quant}}
        for nid, node in g.items():
            if nid != "2q" and node["inputs"].get("clip") == ["2", 0]:
                node["inputs"]["clip"] = ["2q", 0]
    return g
