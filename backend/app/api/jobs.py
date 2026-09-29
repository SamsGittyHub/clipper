"""Job endpoints (PRD 10)."""

from __future__ import annotations

import logging
import shutil
from datetime import timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..config import settings
from ..db import get_session
from ..models import Clip, ClipStatus, Job, JobStatus, utcnow
from ..schemas import BatchJobCreate, BatchJobResult, ClipOut, JobCreate, JobOut, StageLogOut
from ..services.captions import resolve_style
from ..services.downloader import is_supported
from ..workers.queue import get_queue
from .clips import serialise_clip

log = logging.getLogger("clipz.jobs")

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


def _estimate_remaining(job: Job) -> float | None:
    """Seconds left, extrapolated from how long the job has taken so far.

    Crude by construction -- stages are not linear in time -- but a moving
    "about four minutes left" is far more useful while watching a bar than
    no estimate at all, and it is only ever shown as an approximation.
    """
    if job.status.is_terminal or job.started_at is None or job.progress <= 3:
        return None
    started = job.started_at
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    elapsed = (utcnow() - started).total_seconds()
    if elapsed < 5:
        return None
    remaining = elapsed * (100 - job.progress) / job.progress
    return round(min(remaining, 6 * 3600), 1)


def serialise_job(job: Job, session: Session) -> JobOut:
    counts = dict(
        session.execute(
            select(Clip.status, func.count(Clip.id)).where(Clip.job_id == job.id).group_by(Clip.status)
        ).all()
    )
    payload = JobOut.model_validate(job)
    payload.clip_ready_count = int(counts.get(ClipStatus.READY, 0))
    payload.clip_total_count = int(sum(counts.values()))
    payload.eta_seconds = _estimate_remaining(job)
    return payload


def _new_job(session: Session, *, url: str, clip_count: int, max_duration: float, caption_style: str) -> Job:
    """Shared by the single and batch endpoints so both queue jobs identically."""
    job = Job(
        source_url=url,
        clip_count=clip_count,
        max_duration=max_duration,
        caption_style=resolve_style(caption_style),
        status=JobStatus.QUEUED,
        message="Queued, waiting for a worker",
    )
    session.add(job)
    session.flush()
    get_queue().enqueue(job.id)
    return job


@router.post("", response_model=JobOut, status_code=201)
def create_job(payload: JobCreate, session: Session = Depends(get_session)) -> JobOut:
    """Accept a clip-generation request and hand it to the worker."""
    if not is_supported(payload.url):
        raise HTTPException(
            status_code=422,
            detail="Only YouTube URLs are supported in this release",
        )

    job = _new_job(
        session,
        url=payload.url,
        clip_count=payload.clip_count,
        max_duration=payload.max_duration,
        caption_style=payload.caption_style,
    )
    session.commit()
    session.refresh(job)
    return serialise_job(job, session)


@router.post("/batch", response_model=list[BatchJobResult], status_code=201)
def create_jobs_batch(payload: BatchJobCreate, session: Session = Depends(get_session)) -> list[BatchJobResult]:
    """Queue several source URLs at once (one job per valid line).

    A bad line does not sink the batch: it comes back as a result with an
    `error` instead of a `job`, so pasting 10 URLs where one is a typo still
    queues the other 9. Blank lines are dropped silently; duplicates within
    the same paste are flagged rather than double-queued.
    """
    results: list[BatchJobResult] = []
    seen: set[str] = set()

    for raw in payload.urls:
        url = raw.strip()
        if not url:
            continue
        if url in seen:
            results.append(BatchJobResult(url=url, error="Duplicate in this batch; skipped"))
            continue
        seen.add(url)

        if not url.startswith(("http://", "https://")):
            results.append(BatchJobResult(url=url, error="Not a valid http(s) URL"))
            continue
        if not is_supported(url):
            results.append(
                BatchJobResult(url=url, error="Only YouTube URLs are supported in this release")
            )
            continue

        job = _new_job(
            session,
            url=url,
            clip_count=payload.clip_count,
            max_duration=payload.max_duration,
            caption_style=payload.caption_style,
        )
        results.append(BatchJobResult(url=url, job=serialise_job(job, session)))

    if not any(result.job for result in results):
        session.rollback()
        raise HTTPException(status_code=422, detail="No valid YouTube URLs were found in that list")

    session.commit()
    return results


@router.get("", response_model=list[JobOut])
def list_jobs(
    limit: int = Query(default=25, ge=1, le=100),
    session: Session = Depends(get_session),
) -> list[JobOut]:
    jobs = session.scalars(select(Job).order_by(Job.created_at.desc()).limit(limit)).all()
    return [serialise_job(job, session) for job in jobs]


@router.get("/{job_id}", response_model=JobOut)
def get_job(job_id: str, session: Session = Depends(get_session)) -> JobOut:
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return serialise_job(job, session)


@router.get("/{job_id}/clips", response_model=list[ClipOut])
def list_job_clips(job_id: str, session: Session = Depends(get_session)) -> list[ClipOut]:
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    clips = session.scalars(
        select(Clip).where(Clip.job_id == job_id).order_by(Clip.rank)
    ).all()
    return [serialise_clip(clip) for clip in clips]


@router.post("/{job_id}/cancel", response_model=JobOut)
def cancel_job(job_id: str, session: Session = Depends(get_session)) -> JobOut:
    """Stop a queued or running job.

    Sets the flag the worker polls, and kills any FFmpeg process the job has
    running so a cancel during a long encode takes effect immediately rather
    than whenever that encode would have finished. A job that never reached a
    worker is marked cancelled here and skipped when one picks it up.
    """
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status.is_terminal:
        raise HTTPException(
            status_code=409, detail=f"This job already finished ({job.status.value})"
        )

    job.cancel_requested = True
    if job.status == JobStatus.QUEUED:
        job.status = JobStatus.CANCELLED
        job.message = "Cancelled before it started"
        job.completed_at = utcnow()
    else:
        job.message = "Cancelling..."
    session.commit()

    from ..workers.pipeline import request_cancel

    request_cancel(job_id)
    session.refresh(job)
    log.info("Cancel requested for job %s", job_id)
    return serialise_job(job, session)


@router.post("/{job_id}/retry", response_model=JobOut, status_code=201)
def retry_job(job_id: str, session: Session = Depends(get_session)) -> JobOut:
    """Queue a fresh job with the same settings as a finished one."""
    original = session.get(Job, job_id)
    if original is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if not original.status.is_terminal:
        raise HTTPException(status_code=409, detail="This job is still running")

    job = _new_job(
        session,
        url=original.source_url,
        clip_count=original.clip_count,
        max_duration=original.max_duration,
        caption_style=original.caption_style,
    )
    session.commit()
    session.refresh(job)
    return serialise_job(job, session)


@router.delete("/{job_id}", status_code=200)
def delete_job(job_id: str, session: Session = Depends(get_session)) -> dict:
    """Remove a job, its clips and everything they wrote to disk."""
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.status.is_terminal:
        raise HTTPException(
            status_code=409, detail="Cancel this job before deleting it"
        )

    removed = 0
    for clip in job.clips:
        for path in (clip.output_path, clip.subtitle_path):
            if not path:
                continue
            candidate = Path(path)
            for target in (candidate, candidate.with_suffix(".jpg")):
                try:
                    if target.is_file():
                        target.unlink()
                        removed += 1
                except OSError as exc:  # pragma: no cover - permissions
                    log.warning("Could not delete %s: %s", target, exc)

    for directory in (
        settings.clips_dir / job_id,
        settings.work_dir / job_id,
        settings.downloads_dir / job_id,
    ):
        shutil.rmtree(directory, ignore_errors=True)
    transcript = settings.transcripts_dir / f"{job_id}.json"
    transcript.unlink(missing_ok=True)

    session.delete(job)
    session.commit()
    log.info("Deleted job %s and %d file(s)", job_id, removed)
    return {"deleted": job_id, "files_removed": removed}


@router.get("/{job_id}/logs", response_model=list[StageLogOut])
def job_logs(job_id: str, session: Session = Depends(get_session)) -> list[StageLogOut]:
    """Per-stage timings and errors -- the observability record from PRD 12."""
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return [StageLogOut.model_validate(entry) for entry in job.stage_logs]
