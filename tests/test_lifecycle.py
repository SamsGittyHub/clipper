"""Job lifecycle: cancelling, retrying, deleting, and surviving a restart.

These cover the parts of the system that only matter once it is left running
for a while -- a job the user changed their mind about, a crash halfway
through, an upgrade that adds a database column.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from app.db import session_scope
from app.errors import ErrorCode
from app.models import Clip, ClipStatus, Job, JobStatus


@pytest.fixture(autouse=True)
def no_real_worker(monkeypatch):
    """Accept jobs without running the pipeline."""
    enqueued: list[str] = []
    monkeypatch.setattr(
        "app.api.jobs.get_queue",
        lambda: type("Q", (), {"enqueue": staticmethod(enqueued.append), "depth": 0})(),
    )
    return enqueued


def create_job(client, **overrides):
    payload = {
        "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "clip_count": 3,
        "max_duration": 45,
        "caption_style": "dynamic",
        **overrides,
    }
    return client.post("/api/jobs", json=payload).json()


# ------------------------------------------------------------------ cancel


def test_cancelling_a_queued_job_stops_it_before_a_worker_sees_it(client):
    job_id = create_job(client)["job_id"]
    body = client.post(f"/api/jobs/{job_id}/cancel").json()

    assert body["status"] == "CANCELLED"
    assert body["cancel_requested"] is True
    assert body["completed_at"] is not None


def test_a_worker_skips_a_job_cancelled_before_pickup(client):
    """The queue holds the id, so the worker must re-check on pickup."""
    from app.workers.pipeline import run_job

    job_id = create_job(client)["job_id"]
    client.post(f"/api/jobs/{job_id}/cancel")

    run_job(job_id)  # must not start the pipeline

    with session_scope() as session:
        job = session.get(Job, job_id)
        assert job.status == JobStatus.CANCELLED
        assert job.stage_logs == [], "no stage should have run"


def test_cancelling_a_running_job_marks_it_cancelled_not_failed(client, monkeypatch):
    """A cancel is a user decision, not a fault, and must read as one."""
    from app.workers.pipeline import JobReporter, run_job

    # The worker reads the cancel flag at most once a second so a chatty
    # progress hook cannot hammer the database; that throttle is what this
    # test has to step around to stay deterministic.
    monkeypatch.setattr(JobReporter, "CANCEL_POLL_S", 0.0)

    job_id = create_job(client)["job_id"]

    def cancel_midway(url, *, job_id, progress=None):
        # Stand in for a long download: the flag is set while the stage runs,
        # and the progress callback is where the worker notices.
        client.post(f"/api/jobs/{job_id}/cancel")
        progress(0.5, "Downloading")
        raise AssertionError("the progress callback should have raised")

    monkeypatch.setattr("app.services.downloader.download", cancel_midway)
    run_job(job_id)

    with session_scope() as session:
        job = session.get(Job, job_id)
        assert job.status == JobStatus.CANCELLED
        assert job.error_code == ErrorCode.CANCELLED
        cancelled = [entry for entry in job.stage_logs if entry.status == "cancelled"]
        assert cancelled and cancelled[0].stage == "download"


def test_a_stage_killed_by_the_cancel_reads_as_cancelled_not_failed(client, monkeypatch):
    """Cancelling kills the stage's FFmpeg, which fails the stage.

    Regression: pressing Cancel during audio extraction reported
    ERROR_AUDIO_EXTRACTION_FAILED -- "Could not extract audio from the
    downloaded video" -- which reads as a fault the user has to diagnose,
    for something they deliberately asked for.
    """
    from app.errors import PipelineError
    from app.services.downloader import SourceMedia
    from app.workers.pipeline import run_job

    job_id = create_job(client)["job_id"]

    def fake_download(url, *, job_id, progress=None):
        return SourceMedia(
            path=Path("/nonexistent/source.mp4"), title="t", duration=60.0, channel="c",
            thumbnail_url=None, upload_date=None, source_url=url, width=1280, height=720,
        )

    def killed_extract(source, destination, *, job_id="no-job"):
        # Exactly what happens when the cancel terminates the FFmpeg process.
        client.post(f"/api/jobs/{job_id}/cancel")
        raise PipelineError(
            ErrorCode.AUDIO_EXTRACTION_FAILED,
            "Could not extract audio from the downloaded video",
        )

    monkeypatch.setattr("app.services.downloader.download", fake_download)
    monkeypatch.setattr("app.workers.pipeline.video_info", lambda path: {"duration": 60.0, "width": 1280, "height": 720})
    monkeypatch.setattr("app.workers.pipeline.extract_audio", killed_extract)

    run_job(job_id)

    with session_scope() as session:
        job = session.get(Job, job_id)
        assert job.status == JobStatus.CANCELLED
        assert job.error_code == ErrorCode.CANCELLED
        assert "audio" not in (job.error or "").lower()
        audio_stage = [entry for entry in job.stage_logs if entry.stage == "audio"]
        assert audio_stage[-1].status == "cancelled"


def test_cancelling_a_finished_job_is_a_conflict(client):
    job_id = create_job(client)["job_id"]
    client.post(f"/api/jobs/{job_id}/cancel")
    assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 409


def test_cancel_of_an_unknown_job_is_a_404(client):
    assert client.post("/api/jobs/job_missing/cancel").status_code == 404


# ------------------------------------------------------------ retry, delete


def test_retry_queues_a_new_job_with_the_same_settings(client, no_real_worker):
    original = create_job(client, clip_count=7, max_duration=35, caption_style="karaoke")
    client.post(f"/api/jobs/{original['job_id']}/cancel")

    retried = client.post(f"/api/jobs/{original['job_id']}/retry").json()

    assert retried["job_id"] != original["job_id"]
    assert retried["status"] == "QUEUED"
    assert (retried["clip_count"], retried["max_duration"]) == (7, 35.0)
    assert retried["caption_style"] == "karaoke"
    assert retried["job_id"] in no_real_worker


def test_a_running_job_cannot_be_retried_or_deleted(client):
    job_id = create_job(client)["job_id"]
    with session_scope() as session:
        session.get(Job, job_id).status = JobStatus.TRANSCRIBING

    assert client.post(f"/api/jobs/{job_id}/retry").status_code == 409
    assert client.delete(f"/api/jobs/{job_id}").status_code == 409


def test_delete_removes_the_job_its_clips_and_their_files(client, tmp_path):
    job_id = create_job(client)["job_id"]
    output = tmp_path / "clip_01.mp4"
    output.write_bytes(b"\x00" * 512)
    thumbnail = output.with_suffix(".jpg")
    thumbnail.write_bytes(b"\x00" * 64)

    with session_scope() as session:
        session.add(
            Clip(
                job_id=job_id,
                rank=1,
                start_time=0.0,
                end_time=20.0,
                duration=20.0,
                status=ClipStatus.READY,
                output_path=str(output),
            )
        )
        session.get(Job, job_id).status = JobStatus.COMPLETED

    body = client.delete(f"/api/jobs/{job_id}").json()

    assert body["deleted"] == job_id
    assert not output.exists() and not thumbnail.exists()
    assert client.get(f"/api/jobs/{job_id}").status_code == 404
    with session_scope() as session:
        assert session.query(Clip).filter_by(job_id=job_id).count() == 0


# ----------------------------------------------------------------- restart


def test_a_restart_fails_running_jobs_and_requeues_waiting_ones(client, monkeypatch):
    """Nothing survives the process exiting, so the rows must be reconciled.

    Without this a job sits at TRANSCRIBING forever and the UI polls something
    that will never move again.
    """
    from app.workers import queue as queue_module

    waiting = create_job(client)["job_id"]
    running = create_job(client)["job_id"]
    cancelled = create_job(client)["job_id"]

    with session_scope() as session:
        session.get(Job, running).status = JobStatus.RENDERING
        session.get(Job, cancelled).cancel_requested = True

    requeued: list[str] = []
    monkeypatch.setattr(
        queue_module,
        "get_queue",
        lambda: type("Q", (), {"enqueue": staticmethod(requeued.append)})(),
    )

    queued_count, failed_count = queue_module.recover_interrupted_jobs()

    assert failed_count >= 1 and queued_count >= 1
    assert waiting in requeued
    with session_scope() as session:
        assert session.get(Job, running).status == JobStatus.FAILED
        assert session.get(Job, running).error_code == ErrorCode.INTERRUPTED
        assert session.get(Job, waiting).status == JobStatus.QUEUED
        # A job cancelled while queued must not come back from the dead.
        assert session.get(Job, cancelled).status == JobStatus.CANCELLED
        assert cancelled not in requeued


# --------------------------------------------------------------- migration


def test_an_older_database_gains_new_columns_instead_of_breaking(tmp_path):
    """`create_all` never alters an existing table, so upgrades need this.

    Simulates a database written by an earlier version: the jobs table exists
    but predates the columns added since.
    """
    from app.db import add_missing_columns
    from app.models import Base

    database = tmp_path / "old.db"
    engine = create_engine(f"sqlite:///{database}")
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE jobs ("
                "id VARCHAR(40) NOT NULL PRIMARY KEY, "
                "source_url TEXT, "
                "status VARCHAR(32))"
            )
        )
        connection.execute(
            text("INSERT INTO jobs (id, source_url, status) VALUES ('job_old', 'u', 'COMPLETED')")
        )

    Base.metadata.create_all(engine)  # creates the other tables, skips `jobs`
    applied = add_missing_columns(Base.metadata, engine)

    assert "jobs.cancel_requested" in applied
    with engine.begin() as connection:
        row = connection.execute(text("SELECT cancel_requested FROM jobs")).scalar_one()
    assert row in (0, None), "the existing row survived the migration"
    # Running it a second time must be a no-op, not an error.
    assert add_missing_columns(Base.metadata, engine) == []
    engine.dispose()


# -------------------------------------------------------------------- eta


def test_progress_carries_an_estimate_once_a_job_is_underway(client):
    from datetime import timedelta

    from app.models import utcnow

    job_id = create_job(client)["job_id"]
    with session_scope() as session:
        job = session.get(Job, job_id)
        job.status = JobStatus.TRANSCRIBING
        job.progress = 25
        job.started_at = utcnow() - timedelta(seconds=60)

    body = client.get(f"/api/jobs/{job_id}").json()
    # A quarter done after a minute implies roughly three more.
    assert body["eta_seconds"] == pytest.approx(180, rel=0.2)

    with session_scope() as session:
        session.get(Job, job_id).status = JobStatus.COMPLETED
    assert client.get(f"/api/jobs/{job_id}").json()["eta_seconds"] is None
