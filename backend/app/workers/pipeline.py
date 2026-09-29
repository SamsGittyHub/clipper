"""The clip-generation pipeline (PRD 8.2-8.9).

Runs one job start to finish, writing state to the database as it goes so the
API can report progress without talking to the worker. Every stage is logged
with its duration and outcome; a failure records an actionable error code and
leaves the job in FAILED rather than raising into the queue.

Two things happen concurrently inside a job:

* clips are encoded on a small thread pool, because one FFmpeg process does
  not saturate a modern machine and a five-clip job is otherwise five serial
  encodes;
* the user can cancel at any point -- every stage boundary, every progress
  callback and every clip checks the job's cancel flag, and an in-flight
  FFmpeg process is terminated outright.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select

from ..config import settings
from ..db import session_scope
from ..errors import ErrorCode, JobCancelled, PipelineError
from ..models import Clip, ClipStatus, Job, JobStatus, StageLog, Video, utcnow
from ..services import ai_highlights, captions, clipper, downloader, media, metadata, selection
from ..services.media import extract_audio, select_encoder, video_info
from ..services.transcript import Transcript

log = logging.getLogger("clipz.pipeline")

# Under this much free space a job cannot realistically finish: a long source
# plus its audio, work files and rendered clips. Better to say so up front
# than to fail with a truncated file half an hour in.
MIN_FREE_BYTES = 1_500_000_000
LOW_FREE_BYTES = 6_000_000_000


@dataclass(frozen=True)
class StageSpan:
    """Progress band owned by a stage, so sub-progress maps onto the whole job."""

    status: JobStatus
    start: int
    end: int

    def at(self, fraction: float) -> int:
        fraction = max(0.0, min(1.0, fraction))
        return int(self.start + (self.end - self.start) * fraction)


STAGES = {
    "download": StageSpan(JobStatus.DOWNLOADING, 3, 22),
    "audio": StageSpan(JobStatus.EXTRACTING_AUDIO, 22, 28),
    "transcribe": StageSpan(JobStatus.TRANSCRIBING, 28, 55),
    "analyze": StageSpan(JobStatus.ANALYZING, 55, 70),
    "select": StageSpan(JobStatus.SELECTING_CLIPS, 70, 75),
    "render": StageSpan(JobStatus.PROCESSING_VIDEO, 75, 98),
}

# Captions are cheap next to encoding, so they take a thin slice of the band.
CAPTION_FRACTION = 0.12


class JobReporter:
    """Writes status, progress and stage logs for one job.

    Also the one place that notices a cancellation. The flag lives in the
    database (the API sets it from a different thread), and it is read at
    most once a second so a chatty progress callback cannot turn into a
    stream of queries.
    """

    CANCEL_POLL_S = 1.0

    def __init__(self, job_id: str):
        self.job_id = job_id
        self._lock = threading.Lock()
        self._cancelled = False
        self._checked_at = 0.0

    # ------------------------------------------------------------- state
    def update(self, *, status: JobStatus | None = None, progress: int | None = None, message: str | None = None) -> None:
        with session_scope() as session:
            job = session.get(Job, self.job_id)
            if job is None:  # pragma: no cover - job deleted mid-flight
                return
            if status is not None:
                job.status = status
            if progress is not None:
                job.progress = max(job.progress, min(100, progress))
            if message is not None:
                job.message = message

    def stage_event(self, stage: str, status: str, *, duration: float | None = None, detail: str | None = None) -> None:
        with session_scope() as session:
            session.add(
                StageLog(
                    job_id=self.job_id,
                    stage=stage,
                    status=status,
                    duration_s=duration,
                    detail=detail,
                )
            )

    # -------------------------------------------------------- cancelling
    def is_cancelled(self, *, force: bool = False) -> bool:
        with self._lock:
            if self._cancelled:
                return True
            now = time.monotonic()
            if not force and now - self._checked_at < self.CANCEL_POLL_S:
                return False
            self._checked_at = now
        with session_scope() as session:
            job = session.get(Job, self.job_id)
            flagged = bool(job and job.cancel_requested)
        if flagged:
            with self._lock:
                self._cancelled = True
        return flagged

    def raise_if_cancelled(self, *, force: bool = False) -> None:
        if self.is_cancelled(force=force):
            raise JobCancelled()

    # ----------------------------------------------------------- stages
    @contextmanager
    def stage(self, name: str, message: str):
        """Enter a stage: set its status, log start/finish, own its progress band."""
        span = STAGES[name]
        self.raise_if_cancelled(force=True)
        started = time.time()
        self.update(status=span.status, progress=span.start, message=message)
        self.stage_event(name, "started")
        log.info("[%s] %s started", self.job_id, name)
        try:
            yield span
        except JobCancelled:
            self.stage_event(name, "cancelled", duration=time.time() - started)
            log.info("[%s] %s cancelled", self.job_id, name)
            raise
        except Exception as exc:
            elapsed = time.time() - started
            # Cancelling kills the stage's FFmpeg process, so the stage
            # surfaces a perfectly ordinary failure -- "could not extract
            # audio" -- for something the user deliberately asked for. Check
            # the flag before believing the error.
            if self.is_cancelled(force=True):
                self.stage_event(name, "cancelled", duration=elapsed)
                log.info("[%s] %s cancelled", self.job_id, name)
                raise JobCancelled() from exc
            self.stage_event(name, "failed", duration=elapsed, detail=str(exc)[:1000])
            log.error("[%s] %s failed after %.1fs: %s", self.job_id, name, elapsed, exc)
            raise
        elapsed = time.time() - started
        self.stage_event(name, "ok", duration=elapsed)
        self.update(progress=span.end)
        log.info("[%s] %s finished in %.1fs", self.job_id, name, elapsed)

    def progress_fn(self, span: StageSpan, label: str):
        def report(fraction: float, message: str | None = None) -> None:
            # A long download or transcription is exactly where a cancel has
            # to land, and this is the only code that runs during one.
            self.raise_if_cancelled()
            self.update(progress=span.at(fraction), message=message or label)

        return report


def run_job(job_id: str) -> None:
    """Execute a job. Never raises: terminal state is always recorded."""
    reporter = JobReporter(job_id)
    started = time.time()

    with session_scope() as session:
        job = session.get(Job, job_id)
        if job is None:
            log.error("Job %s vanished before it started", job_id)
            return
        if job.status != JobStatus.QUEUED:
            log.warning("Job %s is %s, not QUEUED; skipping", job_id, job.status)
            return
        if job.cancel_requested:
            job.status = JobStatus.CANCELLED
            job.message = "Cancelled before it started"
            job.completed_at = utcnow()
            log.info("Job %s was cancelled before a worker picked it up", job_id)
            return
        job.started_at = utcnow()
        params = {
            "url": job.source_url,
            "clip_count": job.clip_count,
            "max_duration": job.max_duration,
            "caption_style": job.caption_style,
        }

    try:
        _execute(job_id, params, reporter)
    except JobCancelled as exc:
        _terminate(job_id, JobStatus.CANCELLED, ErrorCode.CANCELLED, exc.message, None)
    except PipelineError as exc:
        if reporter.is_cancelled(force=True):
            _terminate(job_id, JobStatus.CANCELLED, ErrorCode.CANCELLED, "Cancelled by the user", None)
        else:
            _terminate(job_id, JobStatus.FAILED, exc.code, exc.message, exc.detail)
    except Exception as exc:  # unexpected: still a recoverable error record
        if reporter.is_cancelled(force=True):
            _terminate(job_id, JobStatus.CANCELLED, ErrorCode.CANCELLED, "Cancelled by the user", None)
            return
        log.exception("Unhandled error in job %s", job_id)
        _terminate(
            job_id,
            JobStatus.FAILED,
            ErrorCode.INTERNAL,
            "An unexpected error stopped the job",
            f"{type(exc).__name__}: {exc}",
        )
    else:
        with session_scope() as session:
            job = session.get(Job, job_id)
            if job:
                ready = sum(1 for clip in job.clips if clip.status == ClipStatus.READY)
                job.status = JobStatus.COMPLETED
                job.progress = 100
                job.message = f"{ready} clip{'s' if ready != 1 else ''} ready"
                job.completed_at = utcnow()
        log.info("[%s] completed in %.1fs", job_id, time.time() - started)
        _auto_publish_if_enabled(job_id, reporter)
    finally:
        clipper.cleanup_workspace(job_id)


def _auto_publish_if_enabled(job_id: str, reporter: JobReporter) -> None:
    """Fan a completed job's clips out to every connected account (PRD 8.10).

    Fires only when the global toggle is on, and only at completion time --
    jobs that finished while it was off are never back-filled. The toggle is
    read fresh so flipping it on applies from the next completion without a
    restart. A posting problem must never fail a job whose clips already
    rendered: every failure lands in the stage log, and `fan_out` itself is
    idempotent per (clip, account), so a retry of a failed run cannot
    double-post.
    """
    from ..services import publishing as publishing_service
    from ..services import runtime_config

    try:
        publish = runtime_config.publishing()
        if not publish.auto_publish:
            return

        limit = settings.auto_publish_max_clips
        with session_scope() as session:
            query = (
                select(Clip.id)
                .where(Clip.job_id == job_id, Clip.status == ClipStatus.READY)
                .order_by(Clip.rank)
            )
            if limit > 0:
                query = query.limit(limit)
            clip_ids = list(session.scalars(query))
        if not clip_ids:
            reporter.stage_event("autopost", "ok", detail="enabled but nothing rendered to post")
            return

        reporter.stage_event(
            "autopost",
            "started",
            detail=f"clips={len(clip_ids)} privacy={publish.privacy_level}",
        )
        result = publishing_service.fan_out(clip_ids=clip_ids, privacy_level=publish.privacy_level)
        reporter.stage_event(
            "autopost", "ok", detail=f"queued={result.total} skipped={len(result.skipped)}"
        )
        log.info("[%s] auto-posting %d clip(s)", job_id, result.total)
    except Exception:
        # No connected accounts, or the fan-out could not even start: the job
        # itself is done and stays COMPLETED.
        log.exception("[%s] auto-posting failed", job_id)
        reporter.stage_event("autopost", "failed", detail="see the server log")


def _terminate(job_id: str, status: JobStatus, code: str, message: str, detail: str | None) -> None:
    with session_scope() as session:
        job = session.get(Job, job_id)
        if job is None:
            return
        job.status = status
        job.error_code = code
        job.error = f"{message}: {detail}" if detail else message
        job.message = message
        job.completed_at = utcnow()
    if status == JobStatus.CANCELLED:
        log.info("[%s] cancelled", job_id)
    else:
        log.error("[%s] failed %s: %s (%s)", job_id, code, message, detail)


def _check_disk_space() -> None:
    """Refuse to start a job that cannot fit, and warn when space is tight."""
    try:
        free = shutil.disk_usage(settings.storage_dir).free
    except OSError:  # pragma: no cover - unreadable mount
        return
    if free < MIN_FREE_BYTES:
        raise PipelineError(
            ErrorCode.DOWNLOAD_FAILED,
            "Not enough free disk space to process a video",
            detail=(
                f"{free / 1e9:.1f} GB free at {settings.storage_dir}; "
                f"at least {MIN_FREE_BYTES / 1e9:.1f} GB is needed"
            ),
        )
    if free < LOW_FREE_BYTES:
        log.warning("Only %.1f GB free at %s", free / 1e9, settings.storage_dir)


def _execute(job_id: str, params: dict, reporter: JobReporter) -> None:
    work_dir = settings.work_dir / job_id
    work_dir.mkdir(parents=True, exist_ok=True)
    _check_disk_space()

    # ---------------------------------------------------------- download
    with reporter.stage("download", "Downloading the source video") as span:
        source_media = downloader.download(
            params["url"], job_id=job_id, progress=reporter.progress_fn(span, "Downloading")
        )
        info = video_info(source_media.path)
        source_duration = source_media.duration or info.get("duration") or 0.0

        with session_scope() as session:
            video = Video(
                source_url=source_media.source_url,
                title=source_media.title,
                duration=source_duration,
                channel=source_media.channel,
                thumbnail_url=source_media.thumbnail_url,
                upload_date=source_media.upload_date,
                local_path=str(source_media.path),
                width=info.get("width") or source_media.width,
                height=info.get("height") or source_media.height,
            )
            session.add(video)
            session.flush()
            video_id = video.id
            job = session.get(Job, job_id)
            if job:
                job.video_id = video_id

    # ------------------------------------------------------ audio + text
    with reporter.stage("audio", "Extracting audio for transcription"):
        audio_path = extract_audio(source_media.path, work_dir / "audio.wav", job_id=job_id)
        with session_scope() as session:
            video = session.get(Video, video_id)
            if video:
                video.audio_path = str(audio_path)

    with reporter.stage("transcribe", "Generating timestamped transcript") as span:
        from ..services import transcription

        transcript = transcription.transcribe(
            audio_path,
            duration_hint=source_duration,
            progress=reporter.progress_fn(span, "Transcribing"),
        )
        # Kept outside the per-job work directory, which is cleaned when the
        # job ends: the transcript is small and worth keeping afterwards.
        settings.transcripts_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = transcript.save(settings.transcripts_dir / f"{job_id}.json")
        with session_scope() as session:
            video = session.get(Video, video_id)
            if video:
                video.language = transcript.language
                video.transcript_path = str(transcript_path)

    # ------------------------------------------------------------- AI
    with reporter.stage("analyze", "Finding the strongest moments"):
        result = ai_highlights.find_highlights(
            ai_highlights.HighlightRequest(
                title=source_media.title or "Untitled",
                duration=source_duration or transcript.duration,
                clip_count=params["clip_count"],
                max_duration=params["max_duration"],
                transcript=transcript,
            )
        )
        reporter.stage_event(
            "analyze",
            "ok",
            detail=f"provider={result.provider} model={result.model} candidates={len(result.candidates)}",
        )

    # -------------------------------------------------- validate + rank
    with reporter.stage("select", "Validating and ranking candidates"):
        chosen, rejected = selection.select(
            result.candidates,
            transcript=transcript,
            source_duration=source_duration or transcript.duration,
            max_duration=params["max_duration"],
            clip_count=params["clip_count"],
        )
        for clip in chosen:
            metadata.complete(clip, source_title=source_media.title)
        reporter.stage_event(
            "select",
            "ok",
            detail=f"selected={len(chosen)} rejected={len(rejected)}",
        )

        with session_scope() as session:
            for index, scored in enumerate(chosen):
                session.add(
                    Clip(
                        job_id=job_id,
                        rank=index + 1,
                        start_time=scored.start,
                        end_time=scored.end,
                        duration=scored.duration,
                        ai_score=scored.ai_score,
                        hook_score=scored.hook_score,
                        emotion_score=scored.emotion_score,
                        speech_density=scored.speech_density,
                        duration_fit=scored.duration_fit,
                        transcript_confidence=scored.transcript_confidence,
                        novelty=scored.novelty,
                        final_score=scored.final_score,
                        score_breakdown=scored.breakdown,
                        reason=scored.reason,
                        title=scored.title,
                        hook=scored.hook,
                        description=scored.description,
                        hashtags=scored.hashtags,
                        transcript_text=scored.transcript_text,
                        caption_style=params["caption_style"],
                    )
                )

    # ----------------------------------------------------------- render
    _render_clips(job_id, transcript, source_media.path, work_dir, params["caption_style"], reporter)


@dataclass
class _ClipPlan:
    """Everything one clip needs to render, resolved before the pool starts."""

    clip_id: str
    rank: int
    start: float
    end: float
    subtitle_path: Path | None
    destination: Path


def _render_clips(
    job_id: str,
    transcript: Transcript,
    source: Path,
    work_dir: Path,
    caption_style: str,
    reporter: JobReporter,
) -> None:
    span = STAGES["render"]
    with session_scope() as session:
        rows = list(
            session.execute(
                select(Clip.id, Clip.rank, Clip.start_time, Clip.end_time)
                .where(Clip.job_id == job_id)
                .order_by(Clip.rank)
            )
        )

    total = len(rows)
    if total == 0:
        raise PipelineError(ErrorCode.NO_VALID_CANDIDATES, "No clips were selected to render")

    output_dir = settings.clips_dir / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    reporter.stage_event("render", "started")

    # ---------------------------------------------------------- captions
    caption_end = span.at(CAPTION_FRACTION)
    reporter.update(
        status=JobStatus.GENERATING_CAPTIONS,
        progress=span.start,
        message=f"Building captions for {total} clip{'s' if total != 1 else ''}",
    )
    plans: list[_ClipPlan] = []
    for index, (clip_id, rank, start, end) in enumerate(rows):
        reporter.raise_if_cancelled()
        subtitle_path = None
        try:
            subtitle_path = captions.build_ass(
                transcript,
                start=start,
                end=end,
                style=caption_style,
                destination=work_dir / f"clip_{rank:02d}.ass",
            )
        except Exception as exc:  # a caption problem must not lose the clip
            log.warning("[%s] captions failed for clip %s: %s", job_id, clip_id, exc)
        plans.append(
            _ClipPlan(
                clip_id=clip_id,
                rank=rank,
                start=start,
                end=end,
                subtitle_path=subtitle_path,
                destination=output_dir / f"clip_{rank:02d}.mp4",
            )
        )
        reporter.update(progress=span.start + int((caption_end - span.start) * (index + 1) / total))

    # ----------------------------------------------------------- encode
    encoder = select_encoder()
    workers = min(
        total,
        settings.effective_render_concurrency(encoder),
        # Measured, not assumed: a consumer GPU may grant as few as two.
        media.hardware_session_limit(encoder),
    )
    reporter.update(
        status=JobStatus.RENDERING,
        progress=caption_end,
        message=f"Rendering {total} clip{'s' if total != 1 else ''}"
        + (f", {workers} at a time" if workers > 1 else ""),
    )
    log.info("[%s] rendering %d clip(s) with %d worker(s)", job_id, total, workers)

    finished = 0
    rendered = 0
    counter_lock = threading.Lock()

    def render_one(plan: _ClipPlan) -> bool:
        nonlocal finished, rendered
        try:
            reporter.raise_if_cancelled()
            with session_scope() as session:
                clip = session.get(Clip, plan.clip_id)
                if clip is None:
                    return False
                clip.status = ClipStatus.RENDERING

            clipper.render_clip(
                clipper.RenderRequest(
                    source=source,
                    start=plan.start,
                    end=plan.end,
                    destination=plan.destination,
                    subtitle_path=plan.subtitle_path,
                    job_id=job_id,
                )
            )
            clipper.make_thumbnail(
                plan.destination,
                plan.destination.with_suffix(".jpg"),
                at=min(1.0, (plan.end - plan.start) / 2),
            )

            with session_scope() as session:
                clip = session.get(Clip, plan.clip_id)
                if clip:
                    clip.output_path = str(plan.destination)
                    clip.subtitle_path = str(plan.subtitle_path) if plan.subtitle_path else None
                    clip.status = ClipStatus.READY
                    clip.error = None
            with counter_lock:
                rendered += 1
            return True
        except JobCancelled:
            with session_scope() as session:
                clip = session.get(Clip, plan.clip_id)
                if clip and clip.status == ClipStatus.RENDERING:
                    clip.status = ClipStatus.PENDING
            raise
        except PipelineError as exc:
            # One bad clip must not sink a job that produced others.
            log.error("[%s] clip %s failed: %s", job_id, plan.clip_id, exc)
            with session_scope() as session:
                clip = session.get(Clip, plan.clip_id)
                if clip:
                    clip.status = ClipStatus.FAILED
                    clip.error = f"{exc.code}: {exc.message}"
            return False
        finally:
            with counter_lock:
                finished += 1
                done = finished
            reporter.update(
                progress=caption_end + int((span.end - caption_end) * done / total),
                message=f"Rendered {done} of {total} clips",
            )

    cancelled = False
    if workers <= 1:
        for plan in plans:
            try:
                render_one(plan)
            except JobCancelled:
                cancelled = True
                break
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="render") as pool:
            futures = [pool.submit(render_one, plan) for plan in plans]
            for future in futures:
                try:
                    future.result()
                except JobCancelled:
                    cancelled = True
                except Exception:  # pragma: no cover - render_one handles its own
                    log.exception("[%s] unexpected renderer error", job_id)

    elapsed = time.time() - started
    if cancelled or reporter.is_cancelled(force=True):
        reporter.stage_event("render", "cancelled", duration=elapsed, detail=f"rendered={rendered}/{total}")
        raise JobCancelled()

    reporter.stage_event(
        "render", "ok" if rendered else "failed", duration=elapsed, detail=f"rendered={rendered}/{total}"
    )
    if rendered == 0:
        raise PipelineError(
            ErrorCode.RENDER_FAILED,
            "Every clip failed to render",
            detail=f"{total} attempted",
        )
    log.info("[%s] rendered %d/%d clip(s) in %.1fs", job_id, rendered, total, elapsed)
    reporter.update(progress=span.end, message=f"{rendered} of {total} clips rendered")


def request_cancel(job_id: str) -> None:
    """Stop a running job as fast as the current stage allows.

    The flag is what the worker polls; killing the FFmpeg process is what
    makes a cancel during a long encode feel immediate rather than taking
    however long the encode had left.
    """
    media.terminate_job_processes(job_id)
