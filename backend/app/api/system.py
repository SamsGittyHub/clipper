"""Health and non-secret configuration for the frontend."""

from __future__ import annotations

import os
import shutil
import signal

from fastapi import APIRouter, HTTPException

from .. import __version__
from ..config import settings
from ..services.ai_highlights import get_provider
from ..services.captions import CAPTION_STYLES
from ..services.downloader import detect_js_runtimes
from ..services.face_tracking import face_tracking_available
from ..services.media import ffmpeg_available, select_encoder
from ..services.tiktok import is_configured as tiktok_configured
from ..workers.queue import get_queue
from . import auth

router = APIRouter(prefix="/api", tags=["system"])


def _storage_report() -> dict:
    """Free space and what this install is using, in bytes.

    Rendering fills a disk quietly, and the first sign is usually a
    truncated MP4. Surfacing it lets the UI warn before that happens.
    """
    report: dict = {"path": str(settings.storage_dir), "free_bytes": None, "used_bytes": None}
    try:
        report["free_bytes"] = shutil.disk_usage(settings.storage_dir).free
    except OSError:  # pragma: no cover - unreadable mount
        return report
    used = 0
    for directory in (settings.clips_dir, settings.downloads_dir, settings.work_dir):
        for path in directory.rglob("*"):
            try:
                if path.is_file():
                    used += path.stat().st_size
            except OSError:  # pragma: no cover - raced with cleanup
                continue
    report["used_bytes"] = used
    return report


@router.get("/health")
def health() -> dict:
    """Whether this install can actually run a job, and with what."""
    provider = get_provider()
    runtimes = detect_js_runtimes()
    queue = get_queue()
    return {
        "status": "ok" if ffmpeg_available() else "degraded",
        "version": __version__,
        "ffmpeg": ffmpeg_available(),
        # Without one of these, yt-dlp falls back to a limited YouTube player
        # client and some downloads fail with "Requested format is not available".
        "js_runtime": runtimes[0] if runtimes else None,
        "encoder": select_encoder(),
        "whisper_model": settings.whisper_model,
        "ai_provider": provider.name,
        "ai_model": getattr(provider, "model", "unknown"),
        "queue_depth": queue.depth,
        "active_jobs": queue.active,
        "packaged": bool(os.environ.get("CLIPZ_PACKAGED")),
        "reframe_mode": settings.reframe_mode,
        "face_tracking_available": face_tracking_available(),
        "auth_required": auth.required(),
        # Whether publishing is available at all, so the UI can say why the
        # connect button will not work rather than failing on the click.
        "tiktok_configured": tiktok_configured(),
        "storage": _storage_report(),
    }


@router.post("/quit", status_code=202)
def quit_app() -> dict:
    """Stop the server from the UI.

    Only available in the packaged desktop build: launched from a menu entry
    there is no terminal to interrupt, so the window needs a way to shut the
    process down. A source checkout is run from a terminal and ignores this.
    """
    if not os.environ.get("CLIPZ_PACKAGED"):
        raise HTTPException(
            status_code=404,
            detail="Quit is only available in the desktop build; press Ctrl+C instead.",
        )
    # Raise SIGINT in this process: uvicorn treats it as a normal shutdown and
    # runs lifespan teardown, so in-flight work stops cleanly.
    os.kill(os.getpid(), signal.SIGINT)
    return {"stopping": True}


@router.get("/config")
def config() -> dict:
    """Defaults and limits the UI needs. Never includes credentials."""
    return {
        "caption_styles": list(CAPTION_STYLES),
        "default_caption_style": settings.default_caption_style,
        "default_clip_count": 5,
        "default_max_duration": 60,
        "min_clip_duration": settings.min_clip_duration,
        "max_clip_duration": settings.max_clip_duration,
        "recommended_duration": [settings.ideal_duration_low, settings.ideal_duration_high],
        "output_resolution": f"{settings.output_width}x{settings.output_height}",
        "reframe_mode": settings.reframe_mode,
        "version": __version__,
        "ranking_weights": {
            "ai_score": settings.weight_ai_score,
            "hook": settings.weight_hook,
            "emotion": settings.weight_emotion,
            "speech_density": settings.weight_speech_density,
            "duration_fit": settings.weight_duration_fit,
            "confidence": settings.weight_confidence,
            "novelty": settings.weight_novelty,
        },
    }
