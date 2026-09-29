"""Clip endpoints: review, metadata editing, download, publish stub (PRD 10)."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException, Response
from sqlalchemy.orm import Session

from ..db import get_session
from ..models import Clip, ClipStatus
from ..schemas import ClipOut, ClipUpdate, PublishRequest
from .streaming import range_response

router = APIRouter(prefix="/api/clips", tags=["clips"])


def serialise_clip(clip: Clip) -> ClipOut:
    payload = ClipOut.model_validate(clip)
    ready = clip.status == ClipStatus.READY and bool(clip.output_path)
    payload.has_output = ready and Path(clip.output_path).exists()
    if payload.has_output:
        payload.download_url = f"/api/clips/{clip.id}/download"
        payload.preview_url = f"/api/clips/{clip.id}/preview"
    return payload


def _get_clip(clip_id: str, session: Session) -> Clip:
    clip = session.get(Clip, clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="Clip not found")
    return clip


def _output_path(clip: Clip) -> Path:
    if not clip.output_path:
        raise HTTPException(status_code=409, detail="This clip has not been rendered yet")
    path = Path(clip.output_path)
    if not path.exists():
        raise HTTPException(status_code=410, detail="The rendered file is no longer on disk")
    return path


@router.get("/{clip_id}", response_model=ClipOut)
def get_clip(clip_id: str, session: Session = Depends(get_session)) -> ClipOut:
    return serialise_clip(_get_clip(clip_id, session))


@router.patch("/{clip_id}", response_model=ClipOut)
def update_clip(
    clip_id: str, payload: ClipUpdate, session: Session = Depends(get_session)
) -> ClipOut:
    """Save metadata edits from the review screen (PRD 8.9)."""
    clip = _get_clip(clip_id, session)
    for field, value in payload.model_dump(exclude_unset=True).items():
        if value is not None:
            setattr(clip, field, value)
    session.commit()
    session.refresh(clip)
    return serialise_clip(clip)


@router.get("/{clip_id}/download")
def download_clip(
    clip_id: str,
    range: str | None = Header(default=None),
    session: Session = Depends(get_session),
):
    clip = _get_clip(clip_id, session)
    path = _output_path(clip)
    stem = (clip.title or "clip").strip().replace("/", "-")[:60] or "clip"
    return range_response(
        path,
        range_header=range,
        filename=f"{stem}.mp4",
        download=True,
    )


@router.get("/{clip_id}/preview")
def preview_clip(
    clip_id: str,
    range: str | None = Header(default=None),
    session: Session = Depends(get_session),
):
    """Same bytes as download, served inline so the player can seek."""
    clip = _get_clip(clip_id, session)
    return range_response(_output_path(clip), range_header=range)


@router.get("/{clip_id}/thumbnail")
def clip_thumbnail(clip_id: str, session: Session = Depends(get_session)):
    clip = _get_clip(clip_id, session)
    path = _output_path(clip).with_suffix(".jpg")
    if not path.exists():
        raise HTTPException(status_code=404, detail="No thumbnail for this clip")
    return range_response(path, range_header=None, media_type="image/jpeg")


@router.get("/{clip_id}/captions")
def clip_captions(clip_id: str, session: Session = Depends(get_session)) -> Response:
    """The ASS file used for this clip, for anyone who wants to restyle it."""
    clip = _get_clip(clip_id, session)
    if not clip.subtitle_path or not Path(clip.subtitle_path).exists():
        raise HTTPException(status_code=404, detail="No caption file for this clip")
    return Response(
        Path(clip.subtitle_path).read_text(encoding="utf-8"),
        media_type="text/plain; charset=utf-8",
    )


@router.post("/{clip_id}/publish", status_code=501)
def publish_clip(
    clip_id: str, payload: PublishRequest, session: Session = Depends(get_session)
) -> None:
    """Reserved for phase 6 (PRD 8.10).

    The data model, idempotency key and endpoint shape exist so the UI can be
    built against them, but nothing is posted anywhere yet -- returning 501 is
    the honest answer rather than silently queueing a post that never happens.
    """
    _get_clip(clip_id, session)
    raise HTTPException(
        status_code=501,
        detail=(
            f"Publishing to {payload.platform} is not implemented yet. "
            "Download the clip and post it manually for now."
        ),
    )
