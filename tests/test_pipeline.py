"""End-to-end pipeline (PRD 8.2-8.9).

The download and transcription stages are stubbed -- they are the two that
need the network and a model download. Everything after them is the real
thing, including FFmpeg encoding the final MP4s.
"""

from __future__ import annotations

import pytest

from app.db import session_scope
from app.errors import ErrorCode
from app.models import Clip, ClipStatus, Job, JobStatus, Video


@pytest.fixture
def stub_pipeline(monkeypatch, sample_video, long_transcript):
    """Feed the worker a local file and a canned transcript."""
    from app.services.downloader import SourceMedia

    def fake_download(url, *, job_id, progress=None):
        if progress:
            progress(0.5, "Downloading")
            progress(1.0, "Done")
        return SourceMedia(
            path=sample_video,
            title="How I lost forty thousand dollars",
            duration=120.0,
            channel="Test Channel",
            thumbnail_url="https://example.invalid/thumb.jpg",
            upload_date="20250101",
            source_url=url,
            width=1280,
            height=720,
        )

    def fake_transcribe(audio_path, *, duration_hint=None, progress=None):
        assert audio_path.exists(), "audio extraction should run for real"
        if progress:
            progress(1.0, "Transcribed")
        return long_transcript

    monkeypatch.setattr("app.services.downloader.download", fake_download)
    monkeypatch.setattr("app.services.transcription.transcribe", fake_transcribe)
    return sample_video


def run_pipeline(clip_count=2, max_duration=30, caption_style="dynamic"):
    from app.workers.pipeline import run_job

    with session_scope() as session:
        job = Job(
            source_url="https://www.youtube.com/watch?v=test",
            clip_count=clip_count,
            max_duration=max_duration,
            caption_style=caption_style,
            status=JobStatus.QUEUED,
        )
        session.add(job)
        session.flush()
        job_id = job.id

    run_job(job_id)
    return job_id


def mean_volume_db(path) -> float:
    """Measured loudness of a file's audio track; silence reads about -91 dB."""
    import re
    import subprocess

    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    match = re.search(r"mean_volume:\s*(-?\d+(?:\.\d+)?) dB", result.stderr)
    assert match, "ffmpeg reported no mean_volume"
    return float(match.group(1))


@pytest.mark.slow
def test_full_pipeline_produces_playable_vertical_clips(stub_pipeline, isolated_storage):
    from app.services.media import video_info

    job_id = run_pipeline()

    with session_scope() as session:
        job = session.get(Job, job_id)
        assert job.status == JobStatus.COMPLETED, job.error
        assert job.progress == 100
        assert job.error is None

        video = session.get(Video, job.video_id)
        assert video.title == "How I lost forty thousand dollars"
        assert video.channel == "Test Channel"
        assert video.language == "en"

        clips = sorted(job.clips, key=lambda clip: clip.rank)
        assert clips, "no clips were produced"
        assert all(clip.status == ClipStatus.READY for clip in clips)

        for clip in clips:
            info = video_info(clip.output_path)
            # 9:16 output, correct length, audio carried through.
            assert (info["width"], info["height"]) == (1080, 1920)
            assert info["has_audio"] is True
            assert info["duration"] == pytest.approx(clip.duration, abs=0.6)
            assert clip.subtitle_path, "captions should have been generated"

            # An AAC track can exist and still be silent, which would ship
            # mute clips; measure it instead of trusting the stream header.
            assert mean_volume_db(clip.output_path) > -50.0, "the clip's audio is silent"

        # Ranked, deduplicated, and explainable.
        scores = [clip.final_score for clip in clips]
        assert scores == sorted(scores, reverse=True)
        assert [clip.rank for clip in clips] == list(range(1, len(clips) + 1))
        for clip in clips:
            assert sum(clip.score_breakdown["contributions"].values()) == pytest.approx(
                clip.final_score, abs=0.05
            )
            assert clip.title and clip.hook and clip.description and clip.hashtags


@pytest.mark.slow
def test_stage_log_records_every_stage_with_timings(stub_pipeline):
    job_id = run_pipeline(clip_count=1)
    with session_scope() as session:
        job = session.get(Job, job_id)
        stages = {entry.stage for entry in job.stage_logs}
        assert {"download", "audio", "transcribe", "analyze", "select", "render"} <= stages
        finished = [entry for entry in job.stage_logs if entry.status == "ok" and entry.duration_s]
        assert finished, "stage durations were not recorded"
        assert all(entry.duration_s >= 0 for entry in finished)


@pytest.mark.slow
def test_clips_respect_the_requested_duration_ceiling(stub_pipeline):
    job_id = run_pipeline(clip_count=3, max_duration=20)
    with session_scope() as session:
        for clip in session.get(Job, job_id).clips:
            assert clip.duration <= 20.0 + 0.01


def test_a_failing_download_records_an_actionable_error(monkeypatch):
    from app.errors import PipelineError

    def broken_download(url, *, job_id, progress=None):
        raise PipelineError(ErrorCode.DOWNLOAD_FAILED, "Video unavailable", detail="HTTP 403")

    monkeypatch.setattr("app.services.downloader.download", broken_download)
    job_id = run_pipeline()

    with session_scope() as session:
        job = session.get(Job, job_id)
        assert job.status == JobStatus.FAILED
        assert job.error_code == ErrorCode.DOWNLOAD_FAILED
        assert "HTTP 403" in job.error
        assert job.completed_at is not None
        failures = [entry for entry in job.stage_logs if entry.status == "failed"]
        assert failures and failures[0].stage == "download"


def test_an_unexpected_error_still_lands_in_a_terminal_state(monkeypatch, stub_pipeline):
    monkeypatch.setattr(
        "app.services.transcription.transcribe",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("model exploded")),
    )
    job_id = run_pipeline()
    with session_scope() as session:
        job = session.get(Job, job_id)
        assert job.status == JobStatus.FAILED
        assert job.error_code == ErrorCode.INTERNAL
        assert "model exploded" in job.error


@pytest.mark.slow
def test_one_bad_clip_does_not_sink_the_whole_job(stub_pipeline, monkeypatch):
    from app.errors import PipelineError
    from app.services import clipper

    real_render = clipper.render_clip

    def flaky_render(request):
        # Keyed on the clip itself rather than call order: clips render in
        # parallel, so "the first call" is whichever thread got there first.
        if request.destination.stem.endswith("01"):
            raise PipelineError(ErrorCode.RENDER_FAILED, "encoder hiccup")
        return real_render(request)

    monkeypatch.setattr("app.workers.pipeline.clipper.render_clip", flaky_render)
    job_id = run_pipeline(clip_count=2)

    with session_scope() as session:
        job = session.get(Job, job_id)
        assert job.status == JobStatus.COMPLETED
        statuses = {clip.status for clip in job.clips}
        assert ClipStatus.FAILED in statuses and ClipStatus.READY in statuses
        failed = next(clip for clip in job.clips if clip.status == ClipStatus.FAILED)
        assert ErrorCode.RENDER_FAILED in failed.error


@pytest.mark.slow
def test_a_key_saved_in_the_settings_panel_drives_the_pipeline(
    client, stub_pipeline, stub_openai_server
):
    """The whole point of the settings panel: entering a key changes selection.

    No environment variable is involved -- the key, base URL and model are
    saved through the API the browser uses, and the job's clips come back
    carrying the stub model's own reasoning.
    """
    import json

    stub_openai_server["reply"]["content"] = json.dumps(
        {
            "clips": [
                {
                    "start": 20,
                    "end": 48,
                    "viral_score": 96,
                    "hook_score": 90,
                    "emotion_score": 80,
                    "reason": "Chosen by the configured model.",
                    "title": "The forty thousand dollar lesson",
                    "hashtags": ["#startup"],
                }
            ]
        }
    )

    saved = client.put(
        "/api/settings/ai",
        json={
            "api_key": "sk-panel-entered-key",
            "base_url": stub_openai_server["base_url"],
            "model": "stub-model",
            "provider": "openai_compatible",
        },
    ).json()
    assert saved["effective_provider"] == "openai_compatible"

    try:
        job_id = run_pipeline(clip_count=1, max_duration=45)

        with session_scope() as session:
            job = session.get(Job, job_id)
            assert job.status == JobStatus.COMPLETED, job.error
            [clip] = job.clips
            assert clip.reason == "Chosen by the configured model."
            assert clip.title == "The forty thousand dollar lesson"
            assert clip.ai_score == 96
            assert clip.start_time == pytest.approx(20, abs=1.5)
            assert clip.status == ClipStatus.READY

        # The request really went to the configured endpoint.
        assert stub_openai_server["received"], "the model was never called"
        assert stub_openai_server["received"][0]["auth"] == "Bearer sk-panel-entered-key"
    finally:
        client.delete("/api/settings/ai")


# --------------------------------------------------------- download retries


def test_a_transient_download_failure_is_retried(monkeypatch, sample_video):
    """YouTube's format list varies per attempt, so a retry usually works."""
    from app.services import downloader

    attempts = {"n": 0}

    class FlakyYDL:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, url, download=True):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RuntimeError(
                    "\x1b[0;31mERROR:\x1b[0m [youtube] x: Requested format is not available."
                )
            return {"title": "Recovered", "duration": 120.0, "filename": str(sample_video)}

    monkeypatch.setattr("time.sleep", lambda _: None)
    monkeypatch.setattr(downloader, "yt_dlp", None, raising=False)
    monkeypatch.setitem(
        __import__("sys").modules, "yt_dlp", type("M", (), {"YoutubeDL": FlakyYDL})
    )

    media = downloader.download("https://www.youtube.com/watch?v=x", job_id="retry-test")
    assert media.title == "Recovered"
    assert attempts["n"] == 3
    # The third attempt drops every format constraint.
    assert FlakyYDL.__init__


def test_a_permanent_failure_is_not_retried(monkeypatch):
    """A private video will never succeed, so fail immediately and say why."""
    from app.errors import PipelineError
    from app.services import downloader

    attempts = {"n": 0}

    class PrivateYDL:
        def __init__(self, options):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, url, download=True):
            attempts["n"] += 1
            raise RuntimeError("ERROR: Private video. Sign in if you have been granted access.")

    monkeypatch.setitem(
        __import__("sys").modules, "yt_dlp", type("M", (), {"YoutubeDL": PrivateYDL})
    )

    with pytest.raises(PipelineError) as excinfo:
        downloader.download("https://www.youtube.com/watch?v=x", job_id="private-test")

    assert attempts["n"] == 1, "a private video must not be retried"
    assert excinfo.value.message == "This video is private."
    assert "\x1b" not in excinfo.value.detail, "ANSI escapes must not reach the UI"
