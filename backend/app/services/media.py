"""FFmpeg/FFprobe wrappers.

All media work funnels through here so every invocation is logged the same way
and FFmpeg output is preserved for diagnosis (PRD 12).
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

from ..config import settings
from ..errors import ErrorCode, PipelineError

log = logging.getLogger("clipz.media")

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"


class FFmpegError(RuntimeError):
    def __init__(self, message: str, *, log_path: Path | None = None, tail: str = ""):
        super().__init__(message)
        self.log_path = log_path
        self.tail = tail


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


# Live FFmpeg processes per job, so a cancel can stop an encode that is in
# flight instead of waiting for it to finish.
_active: dict[str, set[subprocess.Popen]] = {}
_active_lock = threading.Lock()


def _track(job_id: str, proc: subprocess.Popen) -> None:
    with _active_lock:
        _active.setdefault(job_id, set()).add(proc)


def _untrack(job_id: str, proc: subprocess.Popen) -> None:
    with _active_lock:
        procs = _active.get(job_id)
        if procs:
            procs.discard(proc)
            if not procs:
                _active.pop(job_id, None)


def terminate_job_processes(job_id: str) -> int:
    """Kill every FFmpeg process running for a job. Returns how many."""
    with _active_lock:
        procs = list(_active.get(job_id, ()))
    for proc in procs:
        try:
            proc.terminate()
        except OSError:  # pragma: no cover - already gone
            pass
    if procs:
        log.info("Terminated %d FFmpeg process(es) for job %s", len(procs), job_id)
    return len(procs)


def run_ffmpeg(
    args: Sequence[str],
    *,
    job_id: str = "no-job",
    label: str = "ffmpeg",
    timeout: int | None = None,
    cwd: Path | None = None,
) -> str:
    """Run FFmpeg, persisting its log; raise `FFmpegError` on failure.

    `cwd` lets callers pass bare filenames into filtergraphs, which avoids the
    quoting minefield of absolute paths inside `subtitles=`/`ass=` options.
    """
    command = [FFMPEG, "-hide_banner", "-nostdin", "-y", *args]
    log_path = settings.logs_dir / f"{job_id}_{label}_{int(time.time() * 1000)}.log"
    started = time.time()
    log.debug("ffmpeg %s", " ".join(command))

    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(cwd) if cwd else None,
    )
    _track(job_id, proc)
    try:
        stdout, stderr = proc.communicate(timeout=timeout or settings.ffmpeg_timeout_s)
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - environment dependent
        proc.kill()
        proc.communicate()
        raise FFmpegError(f"{label} timed out after {timeout or settings.ffmpeg_timeout_s}s") from exc
    finally:
        _untrack(job_id, proc)

    output = (stderr or "") + (stdout or "")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"$ {' '.join(command)}\n\n{output}\n\nexit={proc.returncode} "
        f"elapsed={time.time() - started:.1f}s\n",
        encoding="utf-8",
    )

    if proc.returncode != 0:
        tail = "\n".join(output.strip().splitlines()[-15:])
        raise FFmpegError(
            f"{label} failed with exit code {proc.returncode}",
            log_path=log_path,
            tail=tail,
        )
    return output


_probe_cache: dict[tuple[str, int, int], dict[str, Any]] = {}
_probe_lock = threading.Lock()


def probe(path: Path) -> dict[str, Any]:
    """ffprobe a file into a dict of format/stream data.

    Cached on (path, size, mtime): the renderer probes the same source once
    per clip, and with clips rendering in parallel that was a burst of
    identical subprocesses.
    """
    path = Path(path)
    try:
        stat = path.stat()
        key = (str(path), stat.st_size, int(stat.st_mtime))
    except OSError:
        key = None
    if key is not None:
        with _probe_lock:
            cached = _probe_cache.get(key)
        if cached is not None:
            return cached

    command = [
        FFPROBE,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise FFmpegError(f"ffprobe failed for {path.name}: {proc.stderr.strip()}")
    data = json.loads(proc.stdout or "{}")
    if key is not None:
        with _probe_lock:
            if len(_probe_cache) > 64:
                _probe_cache.clear()
            _probe_cache[key] = data
    return data


def video_info(path: Path) -> dict[str, Any]:
    """Duration plus the first video stream's geometry."""
    data = probe(path)
    info: dict[str, Any] = {
        "duration": float(data.get("format", {}).get("duration", 0.0) or 0.0),
        "width": None,
        "height": None,
        "fps": None,
        "has_audio": False,
    }
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video" and info["width"] is None:
            info["width"] = stream.get("width")
            info["height"] = stream.get("height")
            info["fps"] = _parse_fraction(stream.get("avg_frame_rate"))
            if not info["duration"]:
                info["duration"] = float(stream.get("duration", 0.0) or 0.0)
        elif stream.get("codec_type") == "audio":
            info["has_audio"] = True
    return info


def _parse_fraction(value: str | None) -> float | None:
    if not value or "/" not in value:
        return None
    numerator, denominator = value.split("/", 1)
    try:
        denominator_f = float(denominator)
        if denominator_f == 0:
            return None
        return float(numerator) / denominator_f
    except ValueError:
        return None


def extract_audio(source: Path, destination: Path, *, job_id: str = "no-job") -> Path:
    """16 kHz mono WAV -- what Whisper wants, and small enough to keep around."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_ffmpeg(
            [
                "-i",
                str(source),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                str(destination),
            ],
            job_id=job_id,
            label="extract_audio",
        )
    except FFmpegError as exc:
        raise PipelineError(
            ErrorCode.AUDIO_EXTRACTION_FAILED,
            "Could not extract audio from the downloaded video",
            detail=exc.tail or str(exc),
        ) from exc
    return destination


_ENCODER_CACHE: dict[str, str] = {}


def select_encoder() -> str:
    """Pick a hardware encoder when one is actually usable, else libx264."""
    configured = settings.video_encoder
    if configured != "auto":
        return configured
    if "auto" in _ENCODER_CACHE:
        return _ENCODER_CACHE["auto"]

    chosen = "libx264"
    for candidate in ("h264_nvenc", "h264_vaapi"):
        if _encoder_works(candidate):
            chosen = candidate
            break
    _ENCODER_CACHE["auto"] = chosen
    log.info("Selected video encoder: %s", chosen)
    return chosen


def _encoder_works(name: str) -> bool:
    """Listed is not the same as working -- encode two frames to be sure."""
    listing = subprocess.run([FFMPEG, "-hide_banner", "-encoders"], capture_output=True, text=True)
    if name not in listing.stdout:
        return False
    if name == "h264_vaapi":
        # VAAPI needs a device and a hwupload chain; skip the probe and let the
        # operator opt in explicitly via VIDEO_ENCODER.
        return False
    probe_cmd = [
        FFMPEG,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=256x256:d=0.1",
        "-c:v",
        name,
        "-frames:v",
        "2",
        "-f",
        "null",
        "-",
    ]
    try:
        return subprocess.run(probe_cmd, capture_output=True, timeout=60).returncode == 0
    except (subprocess.TimeoutExpired, OSError):  # pragma: no cover - environment dependent
        return False


# --------------------------------------------------------------------------
# Concurrent encoder sessions
# --------------------------------------------------------------------------
#
# Rendering several clips at once is the difference between five serial
# encodes and one wall-clock pass, but a hardware encoder will only open so
# many sessions at a time. Two separate ceilings apply, and either can bind
# first:
#
#   * a driver session cap -- consumer NVIDIA cards were long limited to two
#     or three concurrent encodes regardless of the card's capacity;
#   * free VRAM -- each session allocates buffers sized by the output
#     resolution, so a card shared with something else (a resident language
#     model, another encode, a game) may allow fewer than the driver would.
#
# Exceeding either fails the encode outright. That is survivable, since the
# renderer retries in software, but a silent drop to libx264 is slower than
# waiting a moment for a session and makes output quality depend on
# scheduling luck. So the limit is measured and then enforced.
#
# The measurement encodes at the configured output resolution rather than a
# token frame size: a 256x256 probe fits where a 1080x1920 session does not,
# and would happily report a limit the real workload cannot sustain.

_session_limit: dict[str, int] = {}
_session_semaphores: dict[str, threading.Semaphore] = {}
_session_lock = threading.Lock()

# Beyond this there is nothing to gain from more parallel encodes anyway.
MAX_PROBED_SESSIONS = 6
UNLIMITED_SESSIONS = 32


def _concurrent_encodes_work(encoder: str, count: int) -> bool:
    """Can `count` sessions of `encoder` run at once, at output resolution?"""
    size = f"{settings.output_width}x{settings.output_height}"

    def encode(_index: int) -> int:
        command = [
            FFMPEG, "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"color=c=black:s={size}:d=1:r=10",
            "-c:v", encoder, *encoder_quality_args(encoder),
            "-frames:v", "8", "-f", "null", "-",
        ]
        try:
            return subprocess.run(command, capture_output=True, timeout=90).returncode
        except (subprocess.TimeoutExpired, OSError):  # pragma: no cover
            return 1

    # Started together, so they genuinely contend for sessions and memory.
    with ThreadPoolExecutor(max_workers=count) as pool:
        return all(code == 0 for code in pool.map(encode, range(count)))


def hardware_session_limit(encoder: str) -> int:
    """How many clips this encoder can really encode at once.

    Software encoding has no session limit -- it is bounded by CPU, which the
    caller already accounts for. For a hardware encoder the answer depends on
    the card, the driver and how much VRAM is free right now, so it is
    measured by opening sessions at the real output resolution until one is
    refused. Measured once per process, before any render is in flight.
    """
    if encoder == "libx264":
        return UNLIMITED_SESSIONS
    with _session_lock:
        cached = _session_limit.get(encoder)
    if cached is not None:
        return cached

    limit = 1
    for count in range(2, MAX_PROBED_SESSIONS + 1):
        if not _concurrent_encodes_work(encoder, count):
            break
        limit = count

    with _session_lock:
        _session_limit[encoder] = limit
    log.info("%s supports %d concurrent session(s)", encoder, limit)
    return limit


@contextmanager
def encoder_slot(encoder: str) -> Iterator[None]:
    """Hold one of the encoder's sessions for the duration of an encode.

    Guards against every source of concurrency at once -- clips inside a job,
    and jobs running alongside each other -- so an extra encode waits its turn
    instead of failing over to software.
    """
    # Resolved before taking the lock: `hardware_session_limit` takes the
    # same lock, and it is not reentrant.
    limit = hardware_session_limit(encoder)
    with _session_lock:
        semaphore = _session_semaphores.get(encoder)
        if semaphore is None:
            semaphore = threading.Semaphore(limit)
            _session_semaphores[encoder] = semaphore
    with semaphore:
        yield


def reset_session_limits() -> None:
    """Forget the measured limits (used by tests)."""
    with _session_lock:
        _session_limit.clear()
        _session_semaphores.clear()


def encoder_quality_args(encoder: str) -> list[str]:
    """Map the configured quality onto whichever encoder was selected."""
    if encoder == "h264_nvenc":
        return [
            "-preset", "p5",
            "-rc", "vbr",
            "-cq", str(settings.video_crf),
            "-b:v", "0",
            "-pix_fmt", "yuv420p",
        ]
    if encoder == "h264_vaapi":
        return ["-qp", str(settings.video_crf)]
    return [
        "-preset", settings.video_preset,
        "-crf", str(settings.video_crf),
        "-pix_fmt", "yuv420p",
    ]
