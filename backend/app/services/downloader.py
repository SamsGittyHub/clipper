"""Source acquisition via yt-dlp (PRD 8.3)."""

from __future__ import annotations

import functools
import logging
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..config import settings
from ..errors import ErrorCode, JobCancelled, PipelineError

log = logging.getLogger("clipz.downloader")

ProgressFn = Callable[[float, str], None]

# Recognised for the MVP. Anything else is rejected up front with an actionable
# code rather than failing deep inside yt-dlp.
SUPPORTED_HOSTS = (
    "youtube.com",
    "youtu.be",
    "m.youtube.com",
    "www.youtube.com",
    "music.youtube.com",
)


# yt-dlp colours its error strings; those escapes are meaningless in a JSON
# error field and were showing up verbatim in the UI.
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# Failures no retry can fix. Each maps to a message that says what to do.
PERMANENT = (
    ("private video", "This video is private."),
    ("sign in to confirm your age", "This video is age-restricted and needs a signed-in account."),
    ("confirm your age", "This video is age-restricted and needs a signed-in account."),
    ("members-only", "This video is members-only."),
    ("join this channel", "This video is members-only."),
    # yt-dlp phrases this two ways depending on the extractor path -- a bare
    # "Video unavailable" and "This video is unavailable" -- and matching only
    # the first meant the commonest permanent failure of all was retried three
    # times before reporting a generic error.
    ("video unavailable", "This video is unavailable."),
    ("video is unavailable", "This video is unavailable."),
    ("has been removed", "This video has been removed."),
    ("removed by the uploader", "This video has been removed by its uploader."),
    ("not available in your country", "This video is blocked in your region."),
    ("is not a valid url", "That is not a valid video URL."),
    ("live event will begin", "This is a scheduled livestream that has not started."),
    ("sign in to confirm you're not a bot", "YouTube is asking this machine to prove it is not a bot; cookies are needed."),
)


# yt-dlp needs a JavaScript runtime to decipher YouTube's stream URLs. Without
# one it falls back to a limited player client, which is the usual reason a
# download fails with "Requested format is not available". Only deno is enabled
# by default, so anything else present is enabled explicitly. Node below 20 is
# rejected by yt-dlp itself, so it is not offered.
JS_RUNTIMES: tuple[tuple[str, str, int | None], ...] = (
    ("deno", "deno", None),
    ("bun", "bun", None),
    ("node", "node", 20),
    ("quickjs", "qjs", None),
)


def _binary_major_version(binary: str) -> int | None:
    try:
        result = subprocess.run(
            [binary, "--version"], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"(\d+)\.\d+", result.stdout or result.stderr or "")
    return int(match.group(1)) if match else None


@functools.lru_cache(maxsize=1)
def detect_js_runtimes() -> tuple[str, ...]:
    """JavaScript runtimes on this machine that yt-dlp can actually use."""
    found: list[str] = []
    for name, binary, minimum_major in JS_RUNTIMES:
        if not shutil.which(binary):
            continue
        if minimum_major is not None:
            major = _binary_major_version(binary)
            if major is None or major < minimum_major:
                log.info("Ignoring %s: yt-dlp needs %s %d or newer", binary, name, minimum_major)
                continue
        found.append(name)
    if not found:
        log.warning(
            "No JavaScript runtime found (deno, bun, node 20+, quickjs). yt-dlp will "
            "use a limited YouTube player client, and some downloads will fail. "
            "See https://github.com/yt-dlp/yt-dlp/wiki/EJS"
        )
    return tuple(found)


def clean_error(text: str) -> str:
    return ANSI.sub("", str(text)).strip()


def _diagnose(exc: Exception) -> tuple[str, bool]:
    """Return a human-readable reason and whether retrying could help."""
    message = clean_error(exc).lower()
    for needle, reason in PERMANENT:
        if needle in message:
            return reason, False
    if "requested format is not available" in message:
        return (
            "YouTube offered no usable video/audio combination on this attempt. "
            "This is usually transient; if it persists, update yt-dlp or install "
            "a JavaScript runtime (deno, or Node 20+) so yt-dlp can read every format.",
            True,
        )
    return "Failed to download the source video", True


@dataclass
class SourceMedia:
    path: Path
    title: str | None
    duration: float | None
    channel: str | None
    thumbnail_url: str | None
    upload_date: str | None
    source_url: str
    width: int | None = None
    height: int | None = None


def is_supported(url: str) -> bool:
    lowered = url.lower()
    return any(f"//{host}/" in lowered or f".{host}/" in lowered for host in SUPPORTED_HOSTS)


def download(
    url: str,
    *,
    job_id: str,
    progress: ProgressFn | None = None,
) -> SourceMedia:
    """Fetch the best available rendition into `storage/downloads/<job_id>/`."""
    import yt_dlp

    if not is_supported(url):
        raise PipelineError(
            ErrorCode.UNSUPPORTED_SOURCE,
            "Only YouTube URLs are supported in this release",
            detail=url,
        )

    target_dir = settings.downloads_dir / job_id
    target_dir.mkdir(parents=True, exist_ok=True)

    def hook(status: dict[str, Any]) -> None:
        if progress is None:
            return
        if status.get("status") == "downloading":
            total = status.get("total_bytes") or status.get("total_bytes_estimate") or 0
            done = status.get("downloaded_bytes") or 0
            fraction = (done / total) if total else 0.0
            progress(fraction, f"Downloading source ({fraction * 100:.0f}%)")
        elif status.get("status") == "finished":
            progress(1.0, "Download complete, preparing media")

    base_options: dict[str, Any] = {
        "format": settings.ytdlp_format,
        "outtmpl": str(target_dir / "source.%(ext)s"),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 3,
        "fragment_retries": 3,
        "extractor_retries": 3,
        "socket_timeout": settings.download_socket_timeout_s,
        "concurrent_fragment_downloads": max(1, settings.ytdlp_concurrent_fragments),
        "progress_hooks": [hook],
        "restrictfilenames": True,
        "overwrites": True,
    }

    runtimes = detect_js_runtimes()
    if runtimes:
        base_options["js_runtimes"] = {name: {} for name in runtimes}

    # A signed-in session is the only way through age gates, members-only
    # videos and "confirm you're not a bot" walls.
    if settings.ytdlp_cookies_file:
        base_options["cookiefile"] = str(settings.ytdlp_cookies_file)
    elif settings.ytdlp_cookies_from_browser:
        base_options["cookiesfrombrowser"] = (settings.ytdlp_cookies_from_browser,)

    info = _extract_with_retries(yt_dlp, url, base_options, progress=progress)

    if info.get("_type") == "playlist":  # noplaylist still yields this for mixes
        entries = [entry for entry in info.get("entries", []) if entry]
        if not entries:
            raise PipelineError(ErrorCode.DOWNLOAD_FAILED, "The source contained no playable video")
        info = entries[0]

    path = _resolve_downloaded_path(info, target_dir)
    if path is None:
        raise PipelineError(
            ErrorCode.DOWNLOAD_FAILED,
            "yt-dlp reported success but produced no output file",
            detail=str(target_dir),
        )

    log.info("Downloaded %s (%.0fs) to %s", info.get("title"), info.get("duration") or 0, path)
    return SourceMedia(
        path=path,
        title=info.get("title"),
        duration=float(info["duration"]) if info.get("duration") else None,
        channel=info.get("uploader") or info.get("channel"),
        thumbnail_url=info.get("thumbnail"),
        upload_date=info.get("upload_date"),
        source_url=info.get("webpage_url") or url,
        width=info.get("width"),
        height=info.get("height"),
    )


def _extract_with_retries(
    yt_dlp,
    url: str,
    base_options: dict[str, Any],
    *,
    progress: ProgressFn | None,
) -> dict[str, Any]:
    """Download, retrying transient extraction failures.

    Which formats YouTube exposes varies between attempts — it depends on the
    player client yt-dlp lands on, and without a JavaScript runtime some of
    them come back missing streams. A retry usually lands on a working client,
    so each attempt loosens things a little: first as configured, then with the
    player cache bypassed, then with every format constraint dropped.
    """
    attempts = max(1, settings.download_attempts)
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        options = dict(base_options)
        if attempt >= 2:
            options["cachedir"] = False  # a stale cached player can pin the failure
        if attempt >= 3:
            options["format"] = "bv*+ba/b"  # anything playable at all
            log.warning("Retrying %s with unconstrained format selection", url)

        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                return ydl.extract_info(url, download=True)
        except JobCancelled:
            raise  # the progress hook raised it: stop, do not retry
        except Exception as exc:  # yt-dlp raises a wide variety of error types
            last_error = exc
            reason, retryable = _diagnose(exc)
            if not retryable:
                raise PipelineError(
                    ErrorCode.DOWNLOAD_FAILED, reason, detail=clean_error(exc)[:500]
                ) from exc
            if attempt == attempts:
                break
            wait = 2.0 * attempt
            log.warning(
                "Download attempt %d/%d failed (%s); retrying in %.0fs",
                attempt, attempts, clean_error(exc)[:160], wait,
            )
            if progress:
                progress(0.0, f"Download failed, retrying ({attempt + 1} of {attempts})")
            time.sleep(wait)

    reason, _ = _diagnose(last_error) if last_error else ("Failed to download the source video", True)
    raise PipelineError(
        ErrorCode.DOWNLOAD_FAILED, reason, detail=clean_error(last_error)[:500]
    ) from last_error


def _resolve_downloaded_path(info: dict[str, Any], target_dir: Path) -> Path | None:
    """yt-dlp reports the pre-merge name, so confirm against the filesystem."""
    candidates: list[Path] = []
    for downloaded in info.get("requested_downloads") or []:
        if downloaded.get("filepath"):
            candidates.append(Path(downloaded["filepath"]))
    if info.get("filename"):
        candidates.append(Path(info["filename"]))

    for candidate in candidates:
        if candidate.exists():
            return candidate
        merged = candidate.with_suffix(".mp4")
        if merged.exists():
            return merged

    files = sorted(
        (p for p in target_dir.glob("source.*") if p.is_file() and not p.name.endswith(".part")),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    return files[0] if files else None
