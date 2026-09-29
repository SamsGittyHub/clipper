"""Downloader configuration and output handling tests."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


def test_download_uses_a_current_format_fallback(monkeypatch):
    captured: dict = {}

    class FakeYoutubeDL:
        def __init__(self, options):
            captured.update(options)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, url, download):
            assert url == "https://www.youtube.com/watch?v=test"
            assert download is True
            output = Path(captured["outtmpl"].replace("%(ext)s", "mp4"))
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"fake video")
            return {
                "title": "Test video",
                "duration": 12,
                "requested_downloads": [{"filepath": str(output)}],
            }

    monkeypatch.setitem(sys.modules, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYoutubeDL))

    from app.services.downloader import download

    media = download("https://www.youtube.com/watch?v=test", job_id="format-test")

    from app.config import settings

    assert media.path.suffix == ".mp4"
    assert captured["format"] == settings.ytdlp_format
    assert captured["merge_output_format"] == "mp4"
    assert captured["noplaylist"] is True

    # The properties the selector has to keep, whatever it is tuned to:
    # non-strict height filters (so a format with no reported height survives)
    # and a final branch with no constraints at all.
    assert "height<=?" in captured["format"], "height filters must be non-strict"
    assert captured["format"].split("/")[-1] == "b"


def test_permanent_failures_are_recognised_in_the_words_yt_dlp_actually_uses():
    """Regression: the commonest permanent failure of all was being retried.

    We matched "video unavailable", but yt-dlp writes "This video is
    unavailable" on the usual extractor path. The needle never fired, so an
    unavailable video was retried three times -- with backoff sleeps between
    them -- and then reported as a generic download failure instead of saying
    what was actually wrong.
    """
    from app.services.downloader import _diagnose

    permanent = [
        "ERROR: [youtube] aaa: This video is unavailable",
        "ERROR: [youtube] aaa: Video unavailable",
        "ERROR: [youtube] aaa: Private video. Sign in if you have been granted access.",
        "ERROR: [youtube] aaa: Join this channel to get access to members-only content",
        "ERROR: [youtube] aaa: This video has been removed by the uploader",
        "ERROR: [youtube] aaa: Sign in to confirm your age",
    ]
    for message in permanent:
        reason, retryable = _diagnose(RuntimeError(message))
        assert retryable is False, f"should not be retried: {message}"
        assert reason != "Failed to download the source video", f"no reason given for: {message}"


def test_a_genuinely_transient_failure_is_still_retried():
    """The permanent list must not swallow failures a retry actually fixes.

    "Requested format is not available" reads like a permanent refusal and is
    not one: which formats YouTube exposes varies per attempt, which is why
    the downloader loosens its selector and tries again.
    """
    from app.services.downloader import _diagnose

    for message in (
        "ERROR: unable to download video data: HTTP Error 503",
        "ERROR: [youtube] aaa: Requested format is not available",
    ):
        _reason, retryable = _diagnose(RuntimeError(message))
        assert retryable is True, f"should be retried: {message}"
