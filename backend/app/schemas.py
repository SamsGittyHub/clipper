"""Pydantic request/response models for the public API (PRD 10)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config import settings

CaptionStyle = Literal["classic", "highlight", "dynamic", "chunks", "karaoke"]


class JobCreate(BaseModel):
    url: str = Field(..., description="Source video URL (YouTube for the MVP)")
    clip_count: int = Field(default=5, ge=1, le=20)
    max_duration: float = Field(default=60.0, ge=5.0)
    caption_style: CaptionStyle = settings.default_caption_style  # type: ignore[assignment]

    @field_validator("url")
    @classmethod
    def _url_must_be_http(cls, value: str) -> str:
        value = value.strip()
        if not value.startswith(("http://", "https://")):
            raise ValueError("url must be an http(s) link")
        return value

    @field_validator("max_duration")
    @classmethod
    def _cap_duration(cls, value: float) -> float:
        return min(value, settings.max_clip_duration)


class BatchJobCreate(BaseModel):
    """Queue several source URLs at once -- one job per line pasted in."""

    urls: list[str] = Field(..., min_length=1, max_length=50)
    clip_count: int = Field(default=5, ge=1, le=20)
    max_duration: float = Field(default=60.0, ge=5.0)
    caption_style: CaptionStyle = settings.default_caption_style  # type: ignore[assignment]

    @field_validator("max_duration")
    @classmethod
    def _cap_duration(cls, value: float) -> float:
        return min(value, settings.max_clip_duration)


class VideoOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    source_url: str
    title: str | None = None
    duration: float | None = None
    channel: str | None = None
    thumbnail_url: str | None = None
    upload_date: str | None = None
    language: str | None = None
    created_at: datetime


class JobOut(BaseModel):
    """The status payload from PRD 8.2, plus context useful to the UI."""

    model_config = ConfigDict(from_attributes=True)

    job_id: str = Field(validation_alias="id", serialization_alias="job_id")
    status: str
    progress: int
    message: str
    error_code: str | None = None
    error: str | None = None
    source_url: str
    clip_count: int
    max_duration: float
    caption_style: str
    video: VideoOut | None = None
    clip_ready_count: int = 0
    clip_total_count: int = 0
    cancel_requested: bool = False
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    eta_seconds: float | None = None

    @field_validator("status", mode="before")
    @classmethod
    def _enum_to_str(cls, value: Any) -> str:
        return getattr(value, "value", value)


class BatchJobResult(BaseModel):
    """One line of a batch submission: either a queued job, or why not."""

    url: str
    job: JobOut | None = None
    error: str | None = None


class ClipOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    job_id: str
    rank: int
    start_time: float
    end_time: float
    duration: float
    ai_score: float
    hook_score: float
    emotion_score: float
    final_score: float
    score_breakdown: dict[str, Any] = {}
    reason: str | None = None
    title: str | None = None
    hook: str | None = None
    description: str | None = None
    hashtags: list[str] = []
    transcript_text: str | None = None
    caption_style: str
    status: str
    error: str | None = None
    has_output: bool = False
    download_url: str | None = None
    preview_url: str | None = None

    @field_validator("status", mode="before")
    @classmethod
    def _enum_to_str(cls, value: Any) -> str:
        return getattr(value, "value", value)


class ClipUpdate(BaseModel):
    """Metadata edits from the review screen (PRD 8.9)."""

    title: str | None = Field(default=None, max_length=300)
    hook: str | None = Field(default=None, max_length=500)
    description: str | None = Field(default=None, max_length=5000)
    hashtags: list[str] | None = Field(default=None, max_length=30)

    @field_validator("hashtags")
    @classmethod
    def _normalise_tags(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        tags: list[str] = []
        for raw in value:
            tag = raw.strip()
            if not tag:
                continue
            tags.append(tag if tag.startswith("#") else f"#{tag}")
        return tags


class PublishRequest(BaseModel):
    platform: Literal["tiktok", "youtube", "instagram"]
    scheduled_time: datetime | None = None


class PublishOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    clip_id: str
    platform: str
    status: str
    scheduled_time: datetime | None = None
    external_post_id: str | None = None
    published_url: str | None = None
    error: str | None = None

    @field_validator("status", mode="before")
    @classmethod
    def _enum_to_str(cls, value: Any) -> str:
        return getattr(value, "value", value)


class StageLogOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    stage: str
    status: str
    duration_s: float | None = None
    detail: str | None = None
    created_at: datetime
