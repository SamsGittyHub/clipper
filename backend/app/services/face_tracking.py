"""Face-aware crop planning (PRD phase 4: detection/tracking, primary-subject
selection, smooth crop movement).

The rest of the pipeline never touches OpenCV directly. This module's only
public entry point, `plan_crop`, always returns a usable `CropPlan` -- it
never raises. Anything that goes wrong (no OpenCV, no model file, a corrupt
frame, nothing to detect, a slow video) degrades to a plain centred crop, the
same result v1 always produced. Smart cropping can only make output better or
identical; it can never make a render fail.

Approach: sample frames across the clip at a fixed interval, run a small ONNX
face detector on each (YuNet -- OpenCV's own, MIT-licensed, ~230KB, no GPU
needed), track the horizontal centre of the largest face found, smooth that
track, and emit it as a short list of (time, x-offset) keyframes that the
renderer feeds to FFmpeg's `crop` filter via `sendcmd`.

Frames come from one FFmpeg process that decodes the clip's span once
(multithreaded) and emits already-downscaled frames at the sample cadence
over a pipe. The earlier version seeked OpenCV's VideoCapture to every
sample time, which decodes from the previous keyframe each time -- on a
long-GOP YouTube source that is tens of thousands of decoded frames per
clip instead of a few thousand. Vertical framing is a
single fixed anchor for the whole clip rather than animated -- speakers bob
and gesture more than they change seats, and a still vertical position reads
as far less jittery than one that chases every head movement.

There is no active-speaker detection: on multiple faces, the largest
(closest to camera) is treated as the subject. That is a real limitation for
wide two-shots where neither person dominates the frame; it is still a large
improvement over a fixed centre crop for the common single-speaker case.
"""

from __future__ import annotations

import logging
import math
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..config import settings

log = logging.getLogger("clipz.face_tracking")


@dataclass(frozen=True)
class PanKeyframe:
    """One point on the horizontal pan track, in clip-relative seconds."""

    t: float
    x: int  # crop x-offset in scaled-frame pixel space


@dataclass
class CropPlan:
    """What the renderer needs to build the crop (+ optional pan) filter."""

    mode: str  # "smart" | "center"
    anchor_y: int
    keyframes: list[PanKeyframe] = field(default_factory=list)
    reason: str = ""

    @property
    def is_animated(self) -> bool:
        return self.mode == "smart" and len(self.keyframes) > 1


def _static(target_w: int, target_h: int, scaled_w: int, scaled_h: int, *, reason: str) -> CropPlan:
    """The v1 behaviour: dead centre, no motion."""
    x0 = max(0, (scaled_w - target_w) // 2)
    y0 = max(0, (scaled_h - target_h) // 2)
    return CropPlan(mode="center", anchor_y=y0, keyframes=[PanKeyframe(0.0, x0)], reason=reason)


def face_tracking_available() -> bool:
    """Whether smart cropping can actually run: OpenCV importable, model on disk.

    Cheap and side-effect-free (no detector construction) -- meant for a
    health check, not the hot path.
    """
    try:
        import cv2  # noqa: F401
    except ImportError:
        return False
    return settings.face_model_path.exists()


# Clips render in parallel and a YuNet detector is not safe to share between
# threads, so each worker thread keeps its own.
_thread_state = threading.local()


def _load_detector(width: int, height: int):
    """A YuNet detector sized for one (width, height); cached per thread."""
    import cv2

    # OpenCV logs an informational (harmless) line about setPreferableTarget
    # on every detector construction; it is not a Python warning and would
    # otherwise spam server logs once per clip rendered.
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

    key = (str(settings.face_model_path), width, height)
    cache: dict = getattr(_thread_state, "detectors", None) or {}
    cached = cache.get(key)
    if cached is not None:
        return cached

    if not settings.face_model_path.exists():
        raise FileNotFoundError(f"Face model not found at {settings.face_model_path}")

    detector = cv2.FaceDetectorYN_create(
        str(settings.face_model_path),
        "",
        (width, height),
        score_threshold=settings.face_detect_confidence,
    )
    cache.clear()  # one size in flight at a time keeps memory bounded
    cache[key] = detector
    _thread_state.detectors = cache
    return detector


@dataclass
class _Sample:
    t: float
    cx: float | None  # fraction of frame width, or None if no face found
    cy: float | None


def _source_geometry(source: Path) -> tuple[int, int]:
    from .media import video_info

    info = video_info(source)
    width, height = info.get("width"), info.get("height")
    if not width or not height:
        raise RuntimeError(f"Could not read the frame size of {source.name}")
    return int(width), int(height)


def _sample_faces(source: Path, *, start: float, end: float) -> list[_Sample]:
    """Sample the source at a fixed cadence, returning the largest face's
    centre (as a fraction of frame size) at each sampled time, or None.

    One FFmpeg process decodes [start, end) once and writes downscaled
    BGR frames at `1/interval` fps to a pipe; each frame is read as it
    arrives and detected on immediately, so detection overlaps decoding.
    """
    import numpy as np

    from .media import FFMPEG

    interval = max(0.1, settings.face_sample_interval_s)
    span = max(0.0, end - start)
    expected = max(1, int(math.ceil(span / interval)))

    src_w, src_h = _source_geometry(source)
    scale = min(1.0, settings.face_detect_max_side / max(src_w, src_h))
    # Even dimensions keep every pixel format happy.
    width = max(2, int(src_w * scale) // 2 * 2)
    height = max(2, int(src_h * scale) // 2 * 2)
    frame_bytes = width * height * 3

    command = [
        FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-i", str(source), "-t", f"{span:.3f}",
        "-an", "-sn",
        "-vf", f"fps=1/{interval:.4f},scale={width}:{height}",
        "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1",
    ]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert proc.stdout is not None

    samples: list[_Sample] = []
    deadline = time.monotonic() + settings.face_detect_time_budget_s
    detector = None
    try:
        for index in range(expected):
            if time.monotonic() > deadline:
                log.warning(
                    "Face sampling for %s hit its time budget after %d sample(s); "
                    "using what was gathered",
                    source.name, len(samples),
                )
                break
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break  # source ended early
            t = index * interval
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))

            if detector is None:
                detector = _load_detector(width, height)
                detector.setInputSize((width, height))
            _, faces = detector.detect(frame)

            if faces is None or len(faces) == 0:
                samples.append(_Sample(t, None, None))
            else:
                # The largest box is treated as the on-camera subject; see the
                # module docstring for why (no active-speaker detection).
                best = max(faces, key=lambda face: face[2] * face[3])
                cx = (best[0] + best[2] / 2) / width
                cy = (best[1] + best[3] / 2) / height
                samples.append(_Sample(t, float(cx), float(cy)))
    finally:
        try:
            proc.stdout.close()
        except OSError:  # pragma: no cover
            pass
        if proc.poll() is None:
            proc.kill()
        proc.wait()

    if not samples:
        raise RuntimeError(f"FFmpeg produced no frames for {source.name}")
    return samples


def _fill_and_smooth(samples: list[_Sample]) -> list[tuple[float, float, float]] | None:
    """Forward/back-fill missing samples, then EMA-smooth the horizontal track.

    Returns None if not a single sample found a face -- there is nothing to
    track, and the caller should fall back to a static crop entirely.
    """
    detected = [s for s in samples if s.cx is not None]
    if not detected:
        return None

    # Back-fill any leading gap with the first real detection, so the pan
    # does not start at an arbitrary point before sliding to where the
    # subject actually is.
    first = detected[0]
    filled: list[tuple[float, float, float]] = []
    last_cx, last_cy = first.cx, first.cy
    for sample in samples:
        if sample.cx is not None:
            last_cx, last_cy = sample.cx, sample.cy
        filled.append((sample.t, last_cx, last_cy))

    alpha = max(0.01, min(1.0, settings.face_pan_ema_alpha))
    smoothed: list[tuple[float, float, float]] = []
    ema_x = filled[0][1]
    for t, cx, cy in filled:
        ema_x = ema_x + alpha * (cx - ema_x)
        smoothed.append((t, ema_x, cy))
    return smoothed


def _to_keyframes(
    smoothed: list[tuple[float, float, float]],
    *,
    scaled_w: int,
    target_w: int,
) -> list[PanKeyframe]:
    """Turn a smoothed fractional track into pixel-space crop keyframes.

    A dead zone and a maximum pan speed keep the result calm: real speakers
    move constantly at the scale a face detector sees, and reproducing every
    twitch would look worse than the fixed crop this replaces.
    """
    max_x = max(0, scaled_w - target_w)
    dead_zone_px = settings.face_pan_dead_zone * scaled_w
    max_speed_px_s = settings.face_pan_max_speed * scaled_w

    keyframes: list[PanKeyframe] = []
    last_x: float | None = None
    last_t = 0.0

    for t, cx, _cy in smoothed:
        target_x = max(0.0, min(max_x, cx * scaled_w - target_w / 2))
        if last_x is None:
            last_x = target_x
            keyframes.append(PanKeyframe(t, round(last_x)))
            last_t = t
            continue

        if abs(target_x - last_x) < dead_zone_px:
            continue  # not worth moving for

        elapsed = max(1e-3, t - last_t)
        step = max(-max_speed_px_s * elapsed, min(max_speed_px_s * elapsed, target_x - last_x))
        last_x += step
        last_x = max(0.0, min(max_x, last_x))
        last_t = t
        rounded = round(last_x)
        if not keyframes or rounded != keyframes[-1].x:
            keyframes.append(PanKeyframe(t, rounded))

    return keyframes


def plan_crop(
    source: Path,
    *,
    start: float,
    end: float,
    scaled_w: int,
    scaled_h: int,
    target_w: int,
    target_h: int,
) -> CropPlan:
    """Best-effort face-aware crop. Never raises; always returns something
    the renderer can use immediately."""
    if settings.reframe_mode != "smart":
        return _static(target_w, target_h, scaled_w, scaled_h, reason="reframe_mode is not smart")

    try:
        import cv2  # noqa: F401  -- import here so the app runs without it if reframe_mode != smart
    except ImportError:
        log.warning("OpenCV is not installed; falling back to a centred crop")
        return _static(target_w, target_h, scaled_w, scaled_h, reason="opencv not installed")

    try:
        samples = _sample_faces(source, start=start, end=end)
        smoothed = _fill_and_smooth(samples)
        if smoothed is None:
            return _static(target_w, target_h, scaled_w, scaled_h, reason="no face detected")

        keyframes = _to_keyframes(smoothed, scaled_w=scaled_w, target_w=target_w)
        cy_values = [cy for _, _, cy in smoothed]
        anchor_fraction = sorted(cy_values)[len(cy_values) // 2]  # median: resists outliers
        # Bias toward the configured vertical anchor rather than the raw
        # detected position -- a face dead-centre in the source frame should
        # not necessarily sit dead-centre in a 9:16 crop.
        blended_fraction = (anchor_fraction + settings.face_vertical_anchor) / 2
        anchor_y = max(0, min(scaled_h - target_h, round(blended_fraction * scaled_h - target_h / 2)))

        log.info(
            "Smart crop for %s: %d sample(s), %d keyframe(s), anchor_y=%d",
            source.name, len(samples), len(keyframes), anchor_y,
        )
        return CropPlan(mode="smart", anchor_y=anchor_y, keyframes=keyframes, reason="face tracked")
    except Exception as exc:  # a rendering feature must never take the job down
        log.warning("Smart crop failed for %s (%s); falling back to a centred crop", source.name, exc)
        return _static(target_w, target_h, scaled_w, scaled_h, reason=f"error: {exc}")


def write_sendcmd_file(plan: CropPlan, destination: Path) -> Path:
    """An FFmpeg `sendcmd` script that drives the `crop` filter's `x` over time."""
    lines = [f"{kf.t:.3f} crop@pan x '{kf.x}';" for kf in plan.keyframes]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination
