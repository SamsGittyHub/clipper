"""In-process job queue.

The PRD starts on FastAPI `BackgroundTasks`; this is that idea with the one
change the workload demands. Transcription and FFmpeg are blocking and
CPU/GPU-bound, so they run on dedicated worker threads instead of the event
loop, and jobs are processed in submission order rather than all at once.

`enqueue` is the only entry point, so swapping in Celery/Dramatiq/RQ later
means reimplementing this module and nothing else.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Callable

from ..config import settings

log = logging.getLogger("clipz.queue")


class JobQueue:
    def __init__(self, handler: Callable[[str], None], workers: int = 1):
        self._handler = handler
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._workers = max(1, workers)
        self._started = False
        self._lock = threading.Lock()
        self._active: set[str] = set()

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            for index in range(self._workers):
                thread = threading.Thread(
                    target=self._run, name=f"clipz-worker-{index}", daemon=True
                )
                thread.start()
                self._threads.append(thread)
            self._started = True
            log.info("Job queue started with %d worker(s)", self._workers)

    def stop(self, *, timeout: float = 5.0) -> None:
        with self._lock:
            if not self._started:
                return
            for _ in self._threads:
                self._queue.put(None)
            threads = list(self._threads)
            self._threads.clear()
            self._started = False
        for thread in threads:
            thread.join(timeout=timeout)
        log.info("Job queue stopped")

    def enqueue(self, job_id: str) -> None:
        self.start()  # tolerate enqueue before startup completes
        self._queue.put(job_id)
        log.info("Queued job %s (%d waiting)", job_id, self._queue.qsize())

    @property
    def depth(self) -> int:
        return self._queue.qsize()

    @property
    def active(self) -> list[str]:
        """Jobs a worker is currently executing."""
        with self._lock:
            return sorted(self._active)

    def _run(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                if job_id is None:
                    return
                log.info("Worker picked up job %s", job_id)
                with self._lock:
                    self._active.add(job_id)
                self._handler(job_id)
            except Exception:  # a worker thread must never die
                log.exception("Worker crashed handling job %s", job_id)
            finally:
                if job_id is not None:
                    with self._lock:
                        self._active.discard(job_id)
                self._queue.task_done()


_job_queue: JobQueue | None = None


def get_queue() -> JobQueue:
    global _job_queue
    if _job_queue is None:
        from .pipeline import run_job

        _job_queue = JobQueue(run_job, workers=settings.worker_count)
    return _job_queue


def recover_interrupted_jobs() -> tuple[int, int]:
    """Reconcile jobs left mid-flight by a crash or a restart.

    Nothing survives a process exit: the worker thread, its FFmpeg children
    and any in-memory progress are all gone. Without this, those rows sit at
    `TRANSCRIBING` forever and the UI polls a job that will never move.

    A job that never started is simply queued again; one that was running is
    marked failed with a code that says why, since its intermediate files
    were cleaned up and resuming mid-pipeline is not something the stages
    support. Returns `(requeued, failed)`.
    """
    from sqlalchemy import select

    from ..db import session_scope
    from ..errors import ErrorCode
    from ..models import Job, JobStatus, utcnow

    running = [
        status
        for status in JobStatus
        if not status.is_terminal and status != JobStatus.QUEUED
    ]

    requeue: list[str] = []
    failed = 0
    with session_scope() as session:
        for job in session.scalars(select(Job).where(Job.status.in_(running))):
            job.status = JobStatus.FAILED
            job.error_code = ErrorCode.INTERRUPTED
            job.error = "The server stopped while this job was running"
            job.message = "Interrupted by a restart"
            job.completed_at = utcnow()
            failed += 1
        for job in session.scalars(select(Job).where(Job.status == JobStatus.QUEUED)):
            if job.cancel_requested:
                job.status = JobStatus.CANCELLED
                job.message = "Cancelled"
                job.completed_at = utcnow()
                continue
            requeue.append(job.id)

    if failed:
        log.warning("Marked %d interrupted job(s) as failed after restart", failed)
    for job_id in requeue:
        get_queue().enqueue(job_id)
    if requeue:
        log.info("Re-queued %d job(s) that had not started", len(requeue))
    return len(requeue), failed
