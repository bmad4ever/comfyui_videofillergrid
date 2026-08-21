"""Assemble a VideoFillerGrid timeline in one pass, straight to a file.

The workflow produces N source clips, N*N generated fillers, and an order file
listing "clipIndex,fillerIndex" per unit. Joining those inside a normal graph is
not possible: a dynamic-length join needs a list-reduce, the only one for images
(`RebatchImages`) caps at 4096 frames, and for audio none exists at all. Worse,
ComfyUI materialises the whole expanded list at every node, so building the cut
as tensors costs memory proportional to its duration.

This node sidesteps all of that. It takes every clip and every filler ONCE, then
walks the order writing segments directly into an encoder. Only the N + N*N
distinct sources are ever resident, so memory is flat in duration, and nothing
intermediate is written to disk.

Writing frames and samples directly also removes the failure mode that container
stitching has: a demuxer joins by whole-file duration, so an audio track even one
AAC frame longer than its video leaves a gap that gets filled with duplicated
frames. Here the audio cut for each segment is derived from the running video
frame index, so the two can never drift apart.
"""
from __future__ import annotations

import logging
import os
from fractions import Fraction

import av
import numpy as np
import torch

import comfy.utils
import folder_paths

OUT_SAMPLE_RATE = 48000
OUT_CHANNELS = 2


def _resample(waveform: torch.Tensor, src_rate: int) -> torch.Tensor:
    """[C, L] at src_rate -> [OUT_CHANNELS, L'] at OUT_SAMPLE_RATE."""
    if src_rate != OUT_SAMPLE_RATE:
        import torchaudio

        waveform = torchaudio.functional.resample(waveform, src_rate, OUT_SAMPLE_RATE)
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(OUT_CHANNELS, 1)
    elif waveform.shape[0] > OUT_CHANNELS:
        waveform = waveform[:OUT_CHANNELS]
    elif waveform.shape[0] < OUT_CHANNELS:
        waveform = torch.cat(
            [waveform, waveform[-1:].repeat(OUT_CHANNELS - waveform.shape[0], 1)], dim=0
        )
    return waveform.to(torch.float32).clamp(-1.0, 1.0)


def _audio_of(components, frames: int, fps: Fraction) -> torch.Tensor:
    """Segment audio as [OUT_CHANNELS, L], silent when the source carries none."""
    want = int(round(frames * OUT_SAMPLE_RATE / float(fps)))
    audio = getattr(components, "audio", None)
    if not audio:
        return torch.zeros((OUT_CHANNELS, want), dtype=torch.float32)
    wf = audio["waveform"]
    if wf.ndim == 3:  # [B, C, L]
        wf = wf[0]
    wf = _resample(wf, int(audio["sample_rate"]))
    if wf.shape[1] < want:
        wf = torch.nn.functional.pad(wf, (0, want - wf.shape[1]))
    return wf[:, :want]


class VideoFillerGridAssemble:
    """Walk an order list over clips + fillers and encode the result."""

    # every clip and every filler must arrive together, not one expansion at a time
    INPUT_IS_LIST = True

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "assemble"
    CATEGORY = "video"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Assemble clips and generated fillers into one video, following an "
        "order list of 'clipIndex,fillerIndex' lines. Streams straight to the "
        "encoder: memory is flat in duration and no intermediate files are written."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clips": ("VIDEO", {"tooltip": "The source clips, in loader order."}),
                "fillers": ("VIDEO", {"tooltip": "The N*N fillers, in cartesian order."}),
                "order": ("STRING", {"forceInput": True,
                                     "tooltip": "One 'clipIndex,fillerIndex' per line."}),
                "seconds": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 86400.0,
                                      "step": 0.1,
                                      "tooltip": "Length of the final cut; the walk is "
                                                 "truncated to exactly this. Set 0 to "
                                                 "write every unit whole instead - "
                                                 "required for a looping cut, where "
                                                 "clipping the closing filler would "
                                                 "break the join back to the start."}),
                "filename_prefix": ("STRING", {
                    "default": "video/VideoFillerGrid",
                    "tooltip": "Path prefix under the output folder. A counter and "
                               ".mp4 are appended, so 'video/Foo' becomes "
                               "output/video/Foo_00001_.mp4."}),
            },
            "optional": {
                "crf": ("INT", {
                    "default": 18, "min": 0, "max": 51,
                    "tooltip": "H.264 quality. Lower is better and bigger; each +6 "
                               "roughly halves the file size. 0 is lossless, 18 is "
                               "near-transparent, 23 is x264's own default, above ~28 "
                               "gets visibly soft. Only affects the final video."}),
                "preset": (["veryfast", "fast", "medium", "slow"], {
                    "default": "veryfast",
                    "tooltip": "How hard x264 works to hit the CRF quality. Slower "
                               "presets give the same look in a smaller file, but take "
                               "longer to encode; quality is set by CRF, not by this. "
                               "veryfast suits long cuts, slow is for a final master."}),
                "save_sources": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Also write the N*N fillers and the order file into a "
                               "<name>_sources folder beside the video. They are not "
                               "needed for the cut, only to re-cut it to another length "
                               "without regenerating. Turn off to write the video alone."}),
            },
        }

    @staticmethod
    def _first(value, default=None):
        """INPUT_IS_LIST wraps every scalar input in a one-item list."""
        if isinstance(value, list):
            return value[0] if value else default
        return value if value is not None else default

    def assemble(self, clips, fillers, order, seconds, filename_prefix,
                 crf=None, preset=None, save_sources=None):
        order = self._first(order, "")
        seconds = float(self._first(seconds, 20.0))
        filename_prefix = self._first(filename_prefix, "video/VideoFillerGrid")
        crf = int(self._first(crf, 18))
        preset = self._first(preset, "veryfast")
        save_sources = bool(self._first(save_sources, True))

        units = []
        for lineno, raw in enumerate(order.splitlines(), 1):
            line = raw.strip()
            if not line:
                continue
            try:
                s, f = (int(x) for x in line.split(","))
            except ValueError:
                raise ValueError(
                    f"order line {lineno}: expected 'clipIndex,fillerIndex', got {raw!r}")
            if not 0 <= s < len(clips):
                raise ValueError(
                    f"order line {lineno}: clip {s} out of range ({len(clips)} supplied)")
            if not 0 <= f < len(fillers):
                raise ValueError(
                    f"order line {lineno}: filler {f} out of range ({len(fillers)} supplied)")
            units.append((s, f))
        if not units:
            raise ValueError("order list is empty")

        # Decode each distinct source exactly once. This is the whole memory
        # story: N + N*N segments resident, regardless of how long the cut is.
        clip_parts = [v.get_components() for v in clips]
        fill_parts = [v.get_components() for v in fillers]

        fps = Fraction(fill_parts[0].frame_rate)
        height, width = clip_parts[0].images.shape[1], clip_parts[0].images.shape[2]
        # seconds == 0 means "no truncation": every unit is written whole. A
        # looping cut needs this, since trimming would clip the closing filler
        # and the join back to the first clip would no longer land.
        target_frames = int(round(seconds * float(fps))) if seconds > 0 else None

        full_output_folder, filename, counter, subfolder, _ = \
            folder_paths.get_save_image_path(
                filename_prefix, folder_paths.get_output_directory(), width, height)
        file = f"{filename}_{counter:05}_.mp4"
        path = os.path.join(full_output_folder, file)

        # Walk the order once up front to decide how much of each segment
        # survives the truncation, so the video and audio passes agree exactly.
        plan = []  # (components, take, resize)
        frame_idx = 0
        for s, f in units:
            for parts, resize in ((clip_parts[s], False), (fill_parts[f], True)):
                take = int(parts.images.shape[0])
                if target_frames is not None:
                    take = min(take, target_frames - frame_idx)
                if take <= 0:
                    break
                plan.append((parts, take, resize))
                frame_idx += take
            if target_frames is not None and frame_idx >= target_frames:
                break
        total_frames = frame_idx

        logging.info(
            "VideoFillerGrid: %d units, %d clips + %d fillers, %dx%d @ %s fps -> %d frames%s",
            len(units), len(clips), len(fillers), width, height, float(fps), total_frames,
            "" if target_frames is not None else " (untrimmed)")

        pix_fmt = "yuv420p"
        # movflags mirrors what ComfyUI's own writer passes; without it the mp4
        # muxer rejects the header (EINVAL) on the first mux.
        with av.open(path, mode="w",
                     options={"movflags": "use_metadata_tags+faststart"}) as output:
            vstream = output.add_stream("h264", rate=fps)
            vstream.width, vstream.height = width, height
            vstream.pix_fmt = pix_fmt
            vstream.options = {"crf": str(crf), "preset": preset}

            # layout has to be given at creation; a stream with no channel
            # layout also makes the header invalid
            astream = output.add_stream("aac", rate=OUT_SAMPLE_RATE, layout="stereo")

            # Pass 1: video. Frames are encoded one at a time and released, so
            # only the decoded sources stay resident.
            for parts, take, resize in plan:
                images = parts.images[:take]
                if resize and (images.shape[1] != height or images.shape[2] != width):
                    images = comfy.utils.common_upscale(
                        images.movedim(-1, 1), width, height, "lanczos", "center"
                    ).movedim(1, -1)
                rgb = (images[..., :3].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
                for i in range(take):
                    vf = av.VideoFrame.from_ndarray(rgb[i], format="rgb24")
                    output.mux(vstream.encode(vf.reformat(format=pix_fmt)))
            output.mux(vstream.encode(None))

            # Pass 2: audio, in the same order. Each segment's sample count is
            # derived from the running FRAME index rather than accumulated per
            # segment, so rounding can never let the streams drift apart.
            fifo = av.audio.fifo.AudioFifo()
            size = astream.frame_size or 1024
            emitted = 0
            frame_idx = 0

            def drain(final: bool = False):
                nonlocal emitted
                while True:
                    chunk = fifo.read(size, partial=final)
                    if chunk is None:
                        break
                    chunk.pts = emitted
                    chunk.sample_rate = OUT_SAMPLE_RATE
                    chunk.time_base = Fraction(1, OUT_SAMPLE_RATE)
                    emitted += chunk.samples
                    output.mux(astream.encode(chunk))

            for parts, take, _ in plan:
                want = (int(round((frame_idx + take) * OUT_SAMPLE_RATE / float(fps)))
                        - int(round(frame_idx * OUT_SAMPLE_RATE / float(fps))))
                seg = _audio_of(parts, take, fps)
                if seg.shape[1] < want:
                    seg = torch.nn.functional.pad(seg, (0, want - seg.shape[1]))
                seg = seg[:, :want]
                if seg.shape[1]:
                    af = av.AudioFrame.from_ndarray(
                        np.ascontiguousarray(seg.numpy()), format="fltp", layout="stereo")
                    af.sample_rate = OUT_SAMPLE_RATE
                    fifo.write(af)
                    drain()
                frame_idx += take

            drain(final=True)
            output.mux(astream.encode(None))

        frame_idx = total_frames

        logging.info("VideoFillerGrid: wrote %s (%d frames)", path, frame_idx)

        # Optional sidecars. Folding this in is what makes it switchable at all:
        # a SaveVideo node is an execution ROOT (execution.py:1163 collects every
        # OUTPUT_NODE), so it runs no matter what feeds it -- an upstream switch
        # cannot stop it. A boolean on our own output node can, and it is what
        # ExecutionBlocker's own docstring recommends over blocking.
        # Each run gets its own folder, so stale files from a previous run can
        # never be picked up.
        if save_sources:
            src_dir = os.path.join(full_output_folder, f"{filename}_{counter:05}_sources")
            os.makedirs(src_dir, exist_ok=True)
            for i, video in enumerate(fillers):
                video.save_to(os.path.join(src_dir, f"f_{i:05}.mp4"))
            with open(os.path.join(src_dir, "order.txt"), "w", encoding="utf-8") as fh:
                fh.write("\n".join(f"{s},{f}" for s, f in units))
            logging.info("VideoFillerGrid: wrote %d fillers + order.txt to %s",
                         len(fillers), src_dir)

        return {
            "ui": {"images": []},
            "result": (path,),
        }


NODE_CLASS_MAPPINGS = {"VideoFillerGridAssemble": VideoFillerGridAssemble}
NODE_DISPLAY_NAME_MAPPINGS = {"VideoFillerGridAssemble": "VideoFillerGrid Assemble"}
