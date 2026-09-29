"""Persistence models: Video, Job, Clip, PublishingJob (PRD 10)."""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Base(DeclarativeBase):
    pass


class JobStatus(str, enum.Enum):
    """Pipeline states from PRD 8.2, in execution order."""

    QUEUED = "QUEUED"
    DOWNLOADING = "DOWNLOADING"
    EXTRACTING_AUDIO = "EXTRACTING_AUDIO"
    TRANSCRIBING = "TRANSCRIBING"
    ANALYZING = "ANALYZING"
    SELECTING_CLIPS = "SELECTING_CLIPS"
    PROCESSING_VIDEO = "PROCESSING_VIDEO"
    GENERATING_CAPTIONS = "GENERATING_CAPTIONS"
    RENDERING = "RENDERING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    @property
    def is_terminal(self) -> bool:
        return self in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)


class ClipStatus(str, enum.Enum):
    PENDING = "PENDING"
    RENDERING = "RENDERING"
    READY = "READY"
    FAILED = "FAILED"


class PublishStatus(str, enum.Enum):
    PENDING = "PENDING"
    SCHEDULED = "SCHEDULED"
    PUBLISHING = "PUBLISHING"
    UPLOADED = "UPLOADED"      # TikTok has the bytes; it is still processing
    PUBLISHED = "PUBLISHED"
    FAILED = "FAILED"

    @property
    def is_terminal(self) -> bool:
        return self in (PublishStatus.PUBLISHED, PublishStatus.FAILED)


class Account(Base):
    """A social account the user has connected via OAuth."""

    __tablename__ = "accounts"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: _id("acct"))
    platform: Mapped[str] = mapped_column(String(32), default="tiktok")
    # The platform's stable id for this user; unique per platform.
    external_id: Mapped[str] = mapped_column(String(128))
    display_name: Mapped[str | None] = mapped_column(Text, default=None)
    username: Mapped[str | None] = mapped_column(Text, default=None)
    avatar_url: Mapped[str | None] = mapped_column(Text, default=None)

    # Ciphertext (see services/secrets.py) -- never plain tokens.
    access_token_enc: Mapped[str] = mapped_column(Text)
    refresh_token_enc: Mapped[str | None] = mapped_column(Text, default=None)
    scopes: Mapped[str | None] = mapped_column(Text, default=None)
    token_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    refresh_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)

    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    last_error: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    publishing_jobs: Mapped[list["PublishingJob"]] = relationship(back_populates="account")

    __table_args__ = (UniqueConstraint("platform", "external_id", name="uq_account_platform_external"),)


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: _id("vid"))
    source_url: Mapped[str] = mapped_column(Text)
    title: Mapped[str | None] = mapped_column(Text, default=None)
    duration: Mapped[float | None] = mapped_column(Float, default=None)
    channel: Mapped[str | None] = mapped_column(Text, default=None)
    thumbnail_url: Mapped[str | None] = mapped_column(Text, default=None)
    upload_date: Mapped[str | None] = mapped_column(String(16), default=None)
    local_path: Mapped[str | None] = mapped_column(Text, default=None)
    audio_path: Mapped[str | None] = mapped_column(Text, default=None)
    width: Mapped[int | None] = mapped_column(Integer, default=None)
    height: Mapped[int | None] = mapped_column(Integer, default=None)
    language: Mapped[str | None] = mapped_column(String(16), default=None)
    transcript_path: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    jobs: Mapped[list["Job"]] = relationship(back_populates="video")


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: _id("job"))
    video_id: Mapped[str | None] = mapped_column(ForeignKey("videos.id"), default=None, index=True)

    source_url: Mapped[str] = mapped_column(Text)
    clip_count: Mapped[int] = mapped_column(Integer, default=5)
    max_duration: Mapped[float] = mapped_column(Float, default=60.0)
    caption_style: Mapped[str] = mapped_column(String(32), default="dynamic")

    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, native_enum=False, length=32), default=JobStatus.QUEUED, index=True
    )
    progress: Mapped[int] = mapped_column(Integer, default=0)
    message: Mapped[str] = mapped_column(Text, default="Queued")
    error_code: Mapped[str | None] = mapped_column(String(64), default=None)
    error: Mapped[str | None] = mapped_column(Text, default=None)
    # Set by the cancel endpoint; the worker checks it between stages, between
    # clips, and on every progress callback, and stops at the next check.
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)

    video: Mapped[Video | None] = relationship(back_populates="jobs")
    clips: Mapped[list["Clip"]] = relationship(
        back_populates="job", cascade="all, delete-orphan", order_by="Clip.rank"
    )
    stage_logs: Mapped[list["StageLog"]] = relationship(
        back_populates="job", cascade="all, delete-orphan", order_by="StageLog.created_at"
    )


class Clip(Base):
    __tablename__ = "clips"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: _id("clip"))
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), index=True)

    rank: Mapped[int] = mapped_column(Integer, default=0)
    start_time: Mapped[float] = mapped_column(Float)
    end_time: Mapped[float] = mapped_column(Float)
    duration: Mapped[float] = mapped_column(Float)

    # AI-supplied semantics (PRD 8.5)
    ai_score: Mapped[float] = mapped_column(Float, default=0.0)
    hook_score: Mapped[float] = mapped_column(Float, default=0.0)
    emotion_score: Mapped[float] = mapped_column(Float, default=0.0)
    reason: Mapped[str | None] = mapped_column(Text, default=None)

    # Deterministic signals + explainable final score (PRD 8.6)
    speech_density: Mapped[float] = mapped_column(Float, default=0.0)
    duration_fit: Mapped[float] = mapped_column(Float, default=0.0)
    transcript_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    novelty: Mapped[float] = mapped_column(Float, default=0.0)
    final_score: Mapped[float] = mapped_column(Float, default=0.0)
    score_breakdown: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    # Editable metadata (PRD 8.9)
    title: Mapped[str | None] = mapped_column(Text, default=None)
    hook: Mapped[str | None] = mapped_column(Text, default=None)
    description: Mapped[str | None] = mapped_column(Text, default=None)
    hashtags: Mapped[list[str]] = mapped_column(JSON, default=list)
    transcript_text: Mapped[str | None] = mapped_column(Text, default=None)

    caption_style: Mapped[str] = mapped_column(String(32), default="dynamic")
    output_path: Mapped[str | None] = mapped_column(Text, default=None)
    subtitle_path: Mapped[str | None] = mapped_column(Text, default=None)
    status: Mapped[ClipStatus] = mapped_column(
        Enum(ClipStatus, native_enum=False, length=16), default=ClipStatus.PENDING
    )
    error: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    job: Mapped[Job] = relationship(back_populates="clips")
    publishing_jobs: Mapped[list["PublishingJob"]] = relationship(
        back_populates="clip", cascade="all, delete-orphan"
    )


class PublishingJob(Base):
    """Reserved for the post-MVP publishing phase (PRD 8.10)."""

    __tablename__ = "publishing_jobs"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: _id("pub"))
    clip_id: Mapped[str] = mapped_column(ForeignKey("clips.id"), index=True)
    account_id: Mapped[str | None] = mapped_column(ForeignKey("accounts.id"), default=None, index=True)
    platform: Mapped[str] = mapped_column(String(32))
    scheduled_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    status: Mapped[PublishStatus] = mapped_column(
        Enum(PublishStatus, native_enum=False, length=16), default=PublishStatus.PENDING
    )
    # Idempotency key so a retried publish cannot double-post (PRD 12).
    idempotency_key: Mapped[str] = mapped_column(String(64), unique=True, default=lambda: uuid.uuid4().hex)
    external_post_id: Mapped[str | None] = mapped_column(String(128), default=None)
    published_url: Mapped[str | None] = mapped_column(Text, default=None)
    caption: Mapped[str | None] = mapped_column(Text, default=None)
    privacy_level: Mapped[str | None] = mapped_column(String(32), default=None)
    # Draft uploads land in the account's TikTok drafts instead of posting:
    # the path that works before the app passes TikTok's audit (PRD 8.10).
    draft: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)

    clip: Mapped[Clip] = relationship(back_populates="publishing_jobs")
    account: Mapped["Account | None"] = relationship(back_populates="publishing_jobs")

    __table_args__ = (
        # One publish per clip per account: the natural idempotency key, so a
        # retried fan-out cannot double-post the same clip to the same account.
        UniqueConstraint("clip_id", "account_id", name="uq_publish_clip_account"),
    )


class StageLog(Base):
    """One row per pipeline stage: timestamp, job, stage, status, duration, error."""

    __tablename__ = "stage_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), index=True)
    stage: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16))  # started | ok | failed
    duration_s: Mapped[float | None] = mapped_column(Float, default=None)
    detail: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    job: Mapped[Job] = relationship(back_populates="stage_logs")
