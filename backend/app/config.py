"""Application configuration.

Every tunable in the PRD lives here so the pipeline stays inspectable and the
defaults are visible in one place. Values come from the environment (or a
`.env` file at the repository root); secrets are never sent to the frontend.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
# Bundled alongside the code (not REPO_ROOT) so it resolves correctly both
# in a source checkout and inside the packaged AppImage.
ASSETS_DIR = Path(__file__).resolve().parent / "assets"


# The app was called ClipForge AI before this release. Anyone upgrading has a
# populated data directory and database under the old name, so both are
# adopted on first run rather than left behind -- silently starting empty
# would strand finished clips and, worse, the connected TikTok accounts whose
# refresh tokens live in that database.
LEGACY_DATA_DIR_NAME = "ClipForgeAI"
LEGACY_CONFIG_DIR_NAME = "clipforge"
LEGACY_DATABASE_NAME = "clipforge.db"
DATABASE_NAME = "clipz.db"

# Set when a directory was adopted, so the database can repair the absolute
# paths it recorded under the old location. None when nothing was migrated.
adopted_from: Path | None = None

log = logging.getLogger("clipz.config")


def _adopt(legacy: Path, current: Path) -> bool:
    """Move a pre-rename directory into place. Never overwrites."""
    if current.exists() or not legacy.is_dir():
        return False
    try:
        current.parent.mkdir(parents=True, exist_ok=True)
        legacy.rename(current)
    except OSError as exc:  # pragma: no cover - permissions, or across devices
        log.warning("Could not move %s to %s: %s", legacy, current, exc)
        return False
    log.info("Adopted data from the previous name: %s -> %s", legacy, current)
    return True


def _config_dir() -> Path:
    """XDG config location, used by the packaged build."""
    base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    current = Path(base) / "clipz"
    _adopt(Path(base) / LEGACY_CONFIG_DIR_NAME, current)
    return current


def default_storage_dir() -> Path:
    """Where jobs, clips and the database live.

    A source checkout keeps everything under `storage/`. The packaged build
    runs from a read-only mount, so it falls back to the XDG data directory.
    """
    if os.environ.get("CLIPZ_PACKAGED"):
        base = os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share"
        current = Path(base) / "CLIPZ"
        global adopted_from
        legacy = Path(base) / LEGACY_DATA_DIR_NAME
        if _adopt(legacy, current):
            adopted_from = legacy
        return current
    return REPO_ROOT / "storage"


def adopt_legacy_database(storage_dir: Path) -> bool:
    """Rename a pre-rename database file, checkpointing its WAL first.

    Moving the main file alone would orphan `-wal`, and any committed
    transaction still sitting in it would be lost.
    """
    legacy = storage_dir / LEGACY_DATABASE_NAME
    current = storage_dir / DATABASE_NAME
    if current.exists() or not legacy.is_file():
        return False
    try:
        import sqlite3

        with sqlite3.connect(legacy) as connection:
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        legacy.rename(current)
        for suffix in ("-wal", "-shm"):
            legacy.with_name(legacy.name + suffix).unlink(missing_ok=True)
    except (OSError, Exception) as exc:  # pragma: no cover - corrupt or locked
        log.warning("Could not adopt %s: %s", legacy, exc)
        return False
    log.info("Adopted the database from the previous name: %s", legacy.name)
    return True


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # Later files win, so a user config overrides one shipped in a checkout.
        env_file=(REPO_ROOT / ".env", _config_dir() / "config.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- storage ---------------------------------------------------------
    storage_dir: Path = default_storage_dir()
    database_url: str = ""  # derived from storage_dir unless set explicitly
    keep_intermediate_files: bool = False

    # --- job queue -------------------------------------------------------
    worker_count: int = 1
    # Clips within one job are encoded in parallel. 0 = auto, and whatever
    # it resolves to is capped by the encoder's measured concurrent session
    # limit (see services/media.py), so this only expresses what is worth
    # attempting rather than what the hardware will actually grant.
    render_concurrency: int = 0
    # Jobs that were mid-flight when the server last stopped are marked
    # FAILED on startup; QUEUED ones are picked up again.
    recover_jobs_on_startup: bool = True
    # A rotating log file in the storage tree, so the packaged desktop build
    # (which has no terminal) still leaves a trail to diagnose from.
    log_to_file: bool = True
    log_file_max_bytes: int = 5 * 1024 * 1024
    log_file_backups: int = 3
    # FFmpeg writes one log per invocation. They are the first thing to read
    # when a render misbehaves, and worthless a week later.
    log_retention_days: int = 7

    # --- access control --------------------------------------------------
    # Unset (the default) means the API is open, which is right for a
    # single user on localhost. Set it when the server is reachable from
    # other machines: every /api/* call then needs the token, as a bearer
    # header, an X-API-Key header, or the session cookie the UI sets after
    # the user enters it once.
    api_token: str | None = None

    # --- download (PRD 8.3) ----------------------------------------------
    # Prefer separate streams so yt-dlp can choose the best video and audio
    # independently. ``bv*`` also accepts formats that yt-dlp labels as
    # progressive or otherwise does not expose under the older ``bestvideo``
    # alias. Fall back to a combined stream when separate streams are absent.
    #
    # `<=?` makes the height filter non-strict: a format whose height YouTube
    # did not report is kept rather than discarded, which is what leaves the
    # selector unsatisfiable on YouTube's fallback player. No codec preference
    # -- forcing H.264 can mean downloading 75% more data (1.19 GB vs 680 MB on
    # a 33-minute 1080p60 source) to save decode time on clips of under a
    # minute. The last branch drops every constraint so an unusual source with
    # no 1080p rendition still downloads.
    ytdlp_format: str = (
        "bv*[height<=?1080]+ba/"
        "b[height<=?1080]/"
        "bv*+ba/b"
    )
    download_socket_timeout_s: int = 30
    download_attempts: int = 3
    # DASH sources come as many small fragments; fetching several at once
    # is the single biggest download speed-up yt-dlp offers.
    ytdlp_concurrent_fragments: int = 4
    # Age-restricted and members-only videos need a signed-in session.
    # Either a Netscape-format cookies file, or a browser name to read the
    # cookies from directly (chrome, firefox, chromium, brave, edge, ...).
    ytdlp_cookies_file: Path | None = None
    ytdlp_cookies_from_browser: str | None = None

    # --- transcription (PRD 8.4) -----------------------------------------
    whisper_model: str = "base"
    whisper_device: str = "auto"          # auto | cpu | cuda
    whisper_compute_type: str = "default"  # e.g. float16, int8_float16, int8
    whisper_beam_size: int = 5
    whisper_language: str | None = None    # None => autodetect
    # Batched inference runs several VAD-split chunks through the model at
    # once and is several times faster on a GPU. 0 = auto (16 on CUDA, off
    # on CPU); any failure falls back to the sequential path automatically.
    whisper_batch_size: int = 0
    # CTranslate2 defaults to four intra-op threads; on a many-core CPU box
    # that leaves most of the machine idle. 0 = auto.
    whisper_cpu_threads: int = 0

    # --- AI highlight model (PRD 8.5) ------------------------------------
    # Provider-agnostic: any OpenAI-compatible endpoint works.
    ai_provider: str = "openai_compatible"  # openai_compatible | heuristic
    ox_api_key: str | None = None
    ox_base_url: str = "https://api.ox.ai/v1"
    ox_model: str = "ox-alpha"
    ai_temperature: float = 0.4
    ai_timeout_s: int = 180
    ai_max_retries: int = 2               # retries on transport/5xx errors
    ai_repair_attempts: int = 1           # extra passes to fix invalid JSON
    ai_candidate_multiplier: float = 3.0  # ask for more candidates than needed
    # A transcript longer than this is analysed in overlapping chunks (each
    # its own model call, run concurrently) instead of being truncated,
    # which is what used to happen to the tail of a three-hour source.
    ai_max_prompt_chars: int = 120_000
    ai_chunk_concurrency: int = 3

    # --- validation (PRD 8.6) --------------------------------------------
    min_clip_duration: float = 15.0
    max_clip_duration: float = 90.0
    ideal_duration_low: float = 25.0
    ideal_duration_high: float = 60.0
    overlap_threshold: float = 0.70
    clip_padding_s: float = 0.25  # snap outward to word boundaries by this much

    # --- ranking weights (PRD 8.6); normalised at use time ---------------
    weight_ai_score: float = 0.50
    weight_hook: float = 0.15
    weight_emotion: float = 0.10
    weight_speech_density: float = 0.10
    weight_duration_fit: float = 0.05
    weight_confidence: float = 0.05
    weight_novelty: float = 0.05

    # --- rendering (PRD 8.7) ---------------------------------------------
    output_width: int = 1080
    output_height: int = 1920
    output_fps: int = 30
    video_encoder: str = "auto"  # auto | libx264 | h264_nvenc | h264_vaapi
    video_crf: int = 20          # x264 quality; mapped to -cq for nvenc
    video_preset: str = "medium"
    audio_bitrate: str = "160k"
    reframe_mode: str = "smart"  # smart | crop | blur_pad
    render_retries: int = 1
    ffmpeg_timeout_s: int = 1800

    # --- smart reframing (PRD phase 4: face detection/tracking) ----------
    # "smart" tracks the most prominent detected face horizontally and pans
    # the crop to follow it, falling back to a plain centred crop (silently,
    # never failing the render) whenever no face is found or anything about
    # detection goes wrong.
    face_model_path: Path = ASSETS_DIR / "face_detection_yunet_2023mar.onnx"
    face_detect_confidence: float = 0.6
    face_sample_interval_s: float = 0.5     # how often to sample the source
    face_detect_max_side: int = 480         # downscale before detection, for speed
    face_pan_ema_alpha: float = 0.25        # smoothing: higher = snappier, lower = smoother
    face_pan_dead_zone: float = 0.03        # ignore moves smaller than this (fraction of width)
    face_pan_max_speed: float = 0.6         # cap on pan speed, fraction of width per second
    face_vertical_anchor: float = 0.42      # where the face sits vertically, 0=top 1=bottom
    face_detect_time_budget_s: float = 15.0  # abort sampling past this and use what we have

    # --- publishing (PRD 8.10) -------------------------------------------
    tiktok_client_key: str | None = None
    tiktok_client_secret: str | None = None
    tiktok_redirect_uri: str = "http://localhost:8000/api/accounts/tiktok/callback"
    # Only scopes your TikTok app has actually been granted may be requested;
    # asking for one it does not have fails the login with an error that names
    # `client_key` rather than the scope, which is a confusing place to start
    # debugging. Narrow this to `user.info.basic` to prove the connection
    # works, then widen it once Content Posting is approved.
    tiktok_scopes: str = "user.info.basic,video.publish,video.upload"
    tiktok_auth_base: str = "https://www.tiktok.com"
    tiktok_api_base: str = "https://open.tiktokapis.com"
    tiktok_use_pkce: bool = True
    tiktok_timeout_s: int = 60
    tiktok_upload_timeout_s: int = 900
    # Unaudited apps can only post privately; SELF_ONLY always works.
    default_privacy_level: str = "PUBLIC_TO_EVERYONE"
    publish_concurrency: int = 4
    publish_max_attempts: int = 3
    publish_status_poll_s: int = 5
    publish_status_timeout_s: int = 300
    # PRD 8.10 mode 4: post every rendered clip to every connected account the
    # moment a job completes, with no button press. The Settings panel toggle
    # (runtime_config) overrides this; it applies to jobs completing while it
    # is on, never to jobs that finished earlier.
    auto_publish: bool = False
    # 0 posts every rendered clip. A cap bounds the burst a batch paste turns
    # into: 5 clips x 5 accounts is 25 posts against a ~15/day per-account
    # limit on an unaudited app.
    auto_publish_max_clips: int = 0
    # Upload to each account's TikTok drafts instead of direct posting. This
    # works before the app passes TikTok's audit (direct public posts are
    # rejected until then); the creator finishes each post from the TikTok
    # app with whatever privacy they choose.
    publish_as_draft: bool = False

    # --- captions (PRD 8.8) ----------------------------------------------
    default_caption_style: str = "dynamic"
    caption_font: str = "DejaVu Sans"
    caption_font_size: int = 96
    caption_max_chars: int = 22
    caption_words_per_chunk: int = 4

    @model_validator(mode="after")
    def _derive_database_url(self) -> "Settings":
        """Keep the database beside the rest of the storage tree by default.

        Without this, moving STORAGE_DIR (as the packaged build does) would
        leave the database behind in the installation directory.
        """
        if not self.database_url:
            adopt_legacy_database(self.storage_dir)
            self.database_url = f"sqlite:///{self.storage_dir / DATABASE_NAME}"
        return self

    @property
    def downloads_dir(self) -> Path:
        return self.storage_dir / "downloads"

    @property
    def clips_dir(self) -> Path:
        return self.storage_dir / "clips"

    @property
    def work_dir(self) -> Path:
        return self.storage_dir / "work"

    @property
    def logs_dir(self) -> Path:
        return self.storage_dir / "logs"

    @property
    def transcripts_dir(self) -> Path:
        """Transcripts are tiny and useful after the job, so they are kept
        here rather than in the per-job work directory that gets cleaned."""
        return self.storage_dir / "transcripts"

    def ensure_dirs(self) -> None:
        for path in (
            self.storage_dir,
            self.downloads_dir,
            self.clips_dir,
            self.work_dir,
            self.logs_dir,
            self.transcripts_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def effective_render_concurrency(self, encoder: str) -> int:
        """How many clips to encode at once, before the hardware cap.

        The caller narrows this further with the encoder's measured session
        limit, so this only has to express what is worth attempting: a
        hardware encoder is fast enough that a few in flight saturate it,
        while x264 already spreads one encode across several threads.
        """
        if self.render_concurrency > 0:
            return self.render_concurrency
        if encoder != "libx264":
            return 4
        return max(1, min(3, (os.cpu_count() or 2) // 8))


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings


settings = get_settings()
