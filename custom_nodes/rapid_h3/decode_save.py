"""RapidH3DecodeSaveVideo: video VAE decode + mp4 write in one node, the write overlapping the decode.

Stock graph: VAEDecode -> CreateVideo -> SaveVideo. The mp4 write (≈3.75 s for 141 frames at 1344x768) starts
only after the whole decode (≈20 s). The MiniMax-H3 VAE decodes in 8 temporal chunks and copies each finished
chunk into a CPU buffer (VAE.decode's pre-allocated output, `comfy_has_chunked_io`). This node watches that
buffer and encodes every finished frame on a worker thread while the GPU decodes the next chunk.

The file is the one SaveVideo writes: the same pixels (same decode call, the same (x*255).clamp(0,255).byte()
rounding), and the same PyAV calls and settings as ComfyUI's VideoFromComponents.save_to (written against 0.39; the 0.42.1
image gives the same mp4 packets) (libx264 crf 18,
yuv420p BT.709, AAC audio, faststart, prompt metadata), so the video and audio packets match bit for bit
(checked with ffmpeg -f framemd5 on r1/r4). NVENC is not offered: the image's PyAV build has no h264_nvenc
(only the ffmpeg binary does), and with the overlap only ≈0.7 s of encoding is left after the decode.
"""
import json
import math
import os
import queue
import threading
import time
from fractions import Fraction

import av



class _Writer:
    """Encodes frames [0, n) of a [3, T, H, W] float buffer in [0, 1] as they become ready."""

    def __init__(self, path, fps, height, width, audio, metadata):
        from comfy_api.latest._input_impl import video_types as vt
        self.vt = vt
        self.path, self.fps, self.h, self.w = path, fps, height, width
        self.audio, self.metadata = audio, metadata
        self.q = queue.Queue()
        self.err = None
        self.abort = False
        self.busy_s = 0.0
        self.phases = []  # (frames, s) per batch, then flush / audio / close: where the write tail goes
        self.t = threading.Thread(target=self._run, daemon=True)

    def start(self, buf):
        self.buf = buf  # [3, T, H, W] float32, CPU
        self.t.start()

    def ready(self, n):
        self.q.put(n)

    def finish(self):
        self.q.put(None)
        self.t.join()
        if self.err is not None:
            raise self.err

    def cancel(self):
        """The decode failed: stop the worker without encoding the rest."""
        self.abort = True
        if self.t.is_alive():
            self.q.put(None)
            self.t.join()

    def _run(self):
        try:
            self._encode()
        except BaseException as e:  # noqa: BLE001
            self.err = e
            while not self.abort and self.q.get() is not None:  # wait for finish()'s end marker
                pass

    def _encode(self):
        vt = self.vt
        open_kwargs, fmt, codec = vt.video_output_config(self.path, vt.VideoContainer.MP4, vt.VideoCodec.AUTO)
        with av.open(self.path, **open_kwargs) as output:
            if self.metadata is not None:
                for key, value in self.metadata.items():
                    output.metadata[key] = json.dumps(value)
            frame_rate = Fraction(round(self.fps * 1000), 1000)
            pix_fmt = "yuv420p"
            stream = output.add_stream(vt.VIDEO_ENCODERS[codec], rate=frame_rate)
            stream.width, stream.height, stream.pix_fmt = self.w, self.h, pix_fmt
            stream.options = vt.video_encoder_options(codec, None, None)
            vt.set_video_color_properties(stream.codec_context, "sRGB")

            audio_stream = None
            if self.audio:
                sr = int(self.audio["sample_rate"])
                waveform = self.audio["waveform"]
                waveform = waveform[0, :, :math.ceil((sr / frame_rate) * self.buf.shape[1])]
                layout = {1: "mono", 2: "stereo", 6: "5.1"}.get(waveform.shape[0], "stereo")
                audio_stream = output.add_stream("aac", rate=sr, layout=layout)

            done = 0
            while True:
                n = self.q.get()
                if self.abort:
                    raise RuntimeError("decode failed")
                last = n
                n = self.buf.shape[1] if n is None else min(n, self.buf.shape[1])
                t0 = time.perf_counter()
                for i in range(done, n):
                    img = (self.buf[:, i].permute(1, 2, 0) * 255).clamp(0, 255).byte().cpu().contiguous().numpy()
                    frame = av.VideoFrame.from_ndarray(img, format="rgb24")
                    frame = frame.reformat(format=pix_fmt, dst_colorspace=vt.BT709_NCL)
                    vt.set_video_color_properties(frame, "sRGB")
                    output.mux(stream.encode(frame))
                self.phases.append((n - done, round(time.perf_counter() - t0, 3)))
                done = max(done, n)
                self.busy_s += time.perf_counter() - t0
                if done >= self.buf.shape[1]:
                    break
            t0 = time.perf_counter()
            if last is not None:  # all frames came in before the end marker
                while self.q.get() is not None:
                    pass
            t1 = time.perf_counter()
            output.mux(stream.encode(None))
            t2 = time.perf_counter()
            if audio_stream is not None:
                frame = av.AudioFrame.from_ndarray(waveform.float().cpu().contiguous().numpy(), format="fltp",
                                                   layout=layout)
                frame.sample_rate = sr
                frame.pts = 0
                output.mux(audio_stream.encode(frame))
                output.mux(audio_stream.encode(None))
            self.busy_s += time.perf_counter() - t0
            self.phases.append(("flush", round(t2 - t1, 3)))
            self.phases.append(("audio", round(time.perf_counter() - t2, 3)))
            t0 = time.perf_counter()
        self.phases.append(("close", round(time.perf_counter() - t0, 3)))


class RapidH3DecodeSaveVideo:
    """VAEDecode + CreateVideo(fps, audio, 8 bit, sRGB) + SaveVideo(mp4, h264) with the write overlapping the
    decode. overlap=False decodes everything first (same output, for timing comparisons)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",), "vae": ("VAE",),
                             "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0}),
                             "filename_prefix": ("STRING", {"default": "rapid-h3"}),
                             "overlap": ("BOOLEAN", {"default": True})},
                "optional": {"audio": ("AUDIO",)},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}}

    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "Rapid-H3"

    def run(self, samples, vae, fps, filename_prefix, overlap, audio=None, prompt=None,
            extra_pnginfo=None):
        import folder_paths
        from comfy.cli_args import args
        latent = samples["samples"]
        if latent.is_nested:
            latent = latent.unbind()[0]
        if latent.ndim != 5 or latent.shape[0] != 1:
            raise ValueError(f"RapidH3DecodeSaveVideo: expects one video latent [1, C, T, H, W], got {tuple(latent.shape)}")
        fsm = vae.first_stage_model
        out_shape = fsm.decode_output_shape(latent.shape)          # (1, 3, T, H, W)
        height, width = out_shape[3], out_shape[4]
        out_dir, name, counter, sub, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), width, height)
        metadata = None
        if not args.disable_metadata:
            metadata = {}
            if extra_pnginfo is not None:
                metadata.update(extra_pnginfo)
            if prompt is not None:
                metadata["prompt"] = prompt
            metadata = metadata or None
        fname = f"{name}_{counter:05}_.mp4"
        writer = _Writer(os.path.join(out_dir, fname), fps, height, width, audio, metadata)

        stream = overlap and getattr(fsm, "comfy_has_chunked_io", False) and hasattr(fsm, "_finalize_pixels")
        t0 = time.perf_counter()
        try:
            images = self._decode(vae, fsm, latent, writer, stream)
        except BaseException:
            writer.cancel()
            raise
        t1 = time.perf_counter()
        writer.finish()
        t2 = time.perf_counter()
        print(f"[rapid-h3] decode_save: decode {t1 - t0:.2f} s, write tail {t2 - t1:.2f} s, encoder busy "
              f"{writer.busy_s:.2f} s (overlap={bool(stream)}); last batches {writer.phases[-5:]}", flush=True)
        return {"ui": {"images": [{"filename": fname, "subfolder": sub, "type": "output"}], "animated": (True,)}}

    @staticmethod
    def _decode(vae, fsm, latent, writer, stream):
        if stream:
            # VAE.decode hands the H3 decoder a pre-allocated CPU buffer and the decoder copies each finalized
            # chunk into it in order (decode_temporal.write_part). When the next part is finalized, all earlier
            # parts are already in the buffer, so those frames can be encoded.
            state = {"ready": 0, "pending": 0}
            orig_decode, orig_final = fsm.decode, fsm._finalize_pixels

            def decode(z, output_buffer=None):
                writer.start(output_buffer[0])
                return orig_decode(z, output_buffer=output_buffer)

            def finalize(part):
                state["ready"] += state["pending"]
                if state["ready"]:
                    writer.ready(state["ready"])
                state["pending"] = part.shape[2]
                return orig_final(part)

            fsm.decode, fsm._finalize_pixels = decode, finalize
            try:
                images = vae.decode(latent)
            finally:
                del fsm.decode, fsm._finalize_pixels   # back to the class methods
            if not hasattr(writer, "buf"):  # the decoder took another path (e.g. a single frame)
                writer.start(images[0].permute(3, 0, 1, 2))
        else:
            images = vae.decode(latent)                               # [1, T, H, W, 3] view
            writer.start(images[0].permute(3, 0, 1, 2))
        return images
