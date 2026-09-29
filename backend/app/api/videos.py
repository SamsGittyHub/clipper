"""Source-video endpoints (PRD 8.1: retained source metadata)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_session
from ..models import Video
from ..schemas import VideoOut

router = APIRouter(prefix="/api/videos", tags=["videos"])


@router.get("", response_model=list[VideoOut])
def list_videos(
    limit: int = Query(default=25, ge=1, le=100),
    session: Session = Depends(get_session),
) -> list[VideoOut]:
    videos = session.scalars(select(Video).order_by(Video.created_at.desc()).limit(limit)).all()
    return [VideoOut.model_validate(video) for video in videos]


@router.get("/{video_id}", response_model=VideoOut)
def get_video(video_id: str, session: Session = Depends(get_session)) -> VideoOut:
    video = session.get(Video, video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    return VideoOut.model_validate(video)
