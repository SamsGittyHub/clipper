"""Clip cutting, vertical reframing and rendering (PRD 8.7).

One FFmpeg pass per clip: seek, reframe to 9:16, burn captions, encode. A
hardware encoder is used when one is available, with an automatic software
retry so a driver problem degrades speed rather than failing the job.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from ..config import settings
from ..errors import ErrorCode, PipelineError
from . import face_tracking
from .media import (
    FFmpegError,
    encoder_quality_args,
    encoder_slot,
    run_ffmpeg,
    select_encoder,
    video_info,
)

log = logging.getLogger("clipz.clipper")


@dataclass
class RenderRequest:
    source: Path
    start: float
    end: float
    destination: Path
    subtitle_path: Path | None = None
    job_id: str = "no-job"


def _scaled_dimensions(src_w: int, src_h: int, target_w: int, target_h: int) -> tuple[int, int]:
    """What `scale=W:H:force_original_aspect_ratio=increase` actually produces.

    Matches FFmpeg's own arithmetic (verified empirically against real
    output), which is what lets a Python-computed crop offset land in the
    right place without needing to touch a single decoded frame.
    """
    scale_factor = max(target_w / src_w, target_h / src_h)
    return round(src_w * scale_factor), round(src_h * scale_factor)


def build_filter(
    *,
    has_subtitles: bool,
    subtitle_name: str | None,
    crop_plan: face_tracking.CropPlan | None = None,
    sendcmd_name: str | None = None,
) -> str:
    """Reframe to 9:16 and optionally burn in captions.

    `crop` scales the source to cover the vertical frame and takes a fixed
    centre. `blur_pad` keeps the whole frame over a blurred fill, which is
    kinder to wide two-shots. `smart` (the default) is `crop` with the
    horizontal offset following the detected speaker instead of sitting
    still -- `crop_plan` carries that track; passing none reproduces the
    plain centred crop exactly.
    """
    width, height = settings.output_width, settings.output_height

    if settings.reframe_mode == "blur_pad":
        chain = (
            f"[0:v]split=2[bg][fg];"
            f"[bg]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},gblur=sigma=26,eq=brightness=-0.10[bgb];"
            f"[fg]scale={width}:{height}:force_original_aspect_ratio=decrease[fgs];"
            f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,"
            f"fps={settings.output_fps},setsar=1"
        )
    elif crop_plan is not None and crop_plan.is_animated and sendcmd_name:
        x0 = crop_plan.keyframes[0].x
        chain = (
            f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"sendcmd=f={sendcmd_name},"
            f"crop@pan=w={width}:h={height}:x={x0}:y={crop_plan.anchor_y},"
            f"fps={settings.output_fps},setsar=1"
        )
    elif crop_plan is not None:
        # A plan exists but has nothing to animate (no face found, or
        # smoothing collapsed to one point): a numerically identical static
        # crop, just computed by face_tracking instead of FFmpeg's default.
        x0 = crop_plan.keyframes[0].x if crop_plan.keyframes else "(in_w-out_w)/2"
        chain = (
            f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height}:x={x0}:y={crop_plan.anchor_y},"
            f"fps={settings.output_fps},setsar=1"
        )
    else:
        # No plan at all: the original v1 behaviour, letting FFmpeg centre it.
        chain = (
            f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},fps={settings.output_fps},setsar=1"
        )

    if has_subtitles and subtitle_name:
        # Bare filename: FFmpeg runs with cwd set to the subtitle's directory.
        chain += f",ass={subtitle_name}"
    return chain + "[v]"


def _plan_crop_for(request: RenderRequest, work_dir: Path) -> tuple[face_tracking.CropPlan | None, str | None]:
    """Build a face-tracking plan for this clip, and its sendcmd file if animated.

    Returns (None, None) on any failure to probe the source -- the caller
    then renders with the plain, always-safe default crop.
    """
    try:
        info = video_info(request.source)
        src_w, src_h = info.get("width"), info.get("height")
        if not src_w or not src_h:
            return None, None
    except FFmpegError as exc:
        log.warning("Could not probe %s for smart crop (%s); using a plain crop", request.source.name, exc)
        return None, None

    width, height = settings.output_width, settings.output_height
    scaled_w, scaled_h = _scaled_dimensions(src_w, src_h, width, height)
    plan = face_tracking.plan_crop(
        request.source,
        start=request.start,
        end=request.end,
        scaled_w=scaled_w,
        scaled_h=scaled_h,
        target_w=width,
        target_h=height,
    )

    sendcmd_name = None
    if plan.is_animated:
        cmds_path = work_dir / f"{request.destination.stem}.cmds"
        face_tracking.write_sendcmd_file(plan, cmds_path)
        sendcmd_name = cmds_path.name
    return plan, sendcmd_name


def render_clip(request: RenderRequest) -> Path:
    """Cut, reframe, caption and encode one clip."""
    duration = request.end - request.start
    if duration <= 0:
        raise PipelineError(
            ErrorCode.RENDER_FAILED,
            "Clip end must be after its start",
            detail=f"{request.start}-{request.end}",
        )

    request.destination.parent.mkdir(parents=True, exist_ok=True)

    # libass (and sendcmd, when used) read their file relative to cwd, so
    # they all have to live together.
    work_dir = request.subtitle_path.parent if request.subtitle_path else request.destination.parent
    work_dir.mkdir(parents=True, exist_ok=True)
    subtitle_name = request.subtitle_path.name if request.subtitle_path else None

    crop_plan: face_tracking.CropPlan | None = None
    sendcmd_name: str | None = None
    if settings.reframe_mode == "smart":
        crop_plan, sendcmd_name = _plan_crop_for(request, work_dir)

    encoders = [select_encoder()]
    if encoders[0] != "libx264":
        encoders.append("libx264")  # software fallback for the retry pass

    attempts = max(1, settings.render_retries + 1)
    last_error: Exception | None = None

    for attempt in range(attempts):
        encoder = encoders[min(attempt, len(encoders) - 1)]
        args = [
            "-ss", f"{request.start:.3f}",
            "-i", str(request.source),
            "-t", f"{duration:.3f}",
            "-filter_complex", build_filter(
                has_subtitles=bool(subtitle_name),
                subtitle_name=subtitle_name,
                crop_plan=crop_plan,
                sendcmd_name=sendcmd_name,
            ),
            "-map", "[v]",
            "-map", "0:a?",
            "-c:v", encoder,
            *encoder_quality_args(encoder),
            "-c:a", "aac",
            "-b:a", settings.audio_bitrate,
            "-ar", "48000",
            "-ac", "2",
            "-movflags", "+faststart",
            "-shortest",
            str(request.destination.resolve()),
        ]
        try:
            # One of the encoder's concurrent sessions, so parallel clips
            # queue for hardware rather than silently falling back to x264.
            with encoder_slot(encoder):
                run_ffmpeg(
                    args,
                    job_id=request.job_id,
                    label=f"render_{request.destination.stem}",
                    cwd=work_dir,
                )
            if not request.destination.exists() or request.destination.stat().st_size == 0:
                raise FFmpegError("FFmpeg exited cleanly but wrote no output")
            log.info(
                "Rendered %s (%.1fs) with %s%s",
                request.destination.name,
                duration,
                encoder,
                f", {crop_plan.reason}" if crop_plan else "",
            )
            return request.destination
        except FFmpegError as exc:
            last_error = exc
            log.warning(
                "Render attempt %d/%d failed with %s: %s",
                attempt + 1,
                attempts,
                encoder,
                exc,
            )
            # Deliberately NOT clearing crop_plan/sendcmd_name here. Most
            # failures at this point are encoder trouble (GPU contention, a
            # driver hiccup) that the encoder switch on the next attempt
            # already handles -- they have nothing to do with the crop
            # filter. The sendcmd/crop syntax is entirely self-generated and
            # covered by tests (crop also clamps its own bounds), so a real
            # filter-syntax bug is both unlikely and, if it ever happened,
            # would fail identically on retry and surface as a normal
            # RENDER_FAILED with the FFmpeg log attached -- which is more
            # useful than silently discarding the requested smart crop.

    detail = getattr(last_error, "tail", "") or str(last_error)
    raise PipelineError(
        ErrorCode.RENDER_FAILED,
        "FFmpeg could not render the clip",
        detail=detail,
    ) from last_error


def make_thumbnail(clip_path: Path, destination: Path, *, at: float = 1.0) -> Path | None:
    """Grab a poster frame for the review grid. Never fatal."""
    try:
        run_ffmpeg(
            [
                "-ss", f"{at:.2f}",
                "-i", str(clip_path),
                "-frames:v", "1",
                "-vf", "scale=360:-2",
                str(destination),
            ],
            label="thumbnail",
        )
        return destination
    except FFmpegError as exc:
        log.warning("Thumbnail generation failed for %s: %s", clip_path.name, exc)
        return None


def cleanup_workspace(job_id: str) -> None:
    """Remove intermediates once a job reaches a terminal state (PRD 8.3)."""
    if settings.keep_intermediate_files:
        return
    for directory in (settings.downloads_dir / job_id, settings.work_dir / job_id):
        if directory.exists():
            shutil.rmtree(directory, ignore_errors=True)
            log.info("Cleaned %s", directory)
