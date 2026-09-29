"""Filter construction and smart-crop wiring in the renderer (PRD 8.7)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import settings
from app.services import clipper, face_tracking

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE = FIXTURES / "face_sample.mp4"
WIDE = FIXTURES / "face_wide.mp4"


@pytest.fixture(autouse=True)
def restore_reframe_mode():
    original = settings.reframe_mode
    yield
    settings.reframe_mode = original


# ------------------------------------------------------------- build_filter


def test_no_plan_reproduces_the_original_v1_filter_exactly():
    """The legacy path (no crop_plan at all) must be byte-identical to
    what shipped before smart cropping existed -- nothing should change
    for anyone who has not touched reframe_mode."""
    chain = clipper.build_filter(has_subtitles=False, subtitle_name=None)
    assert chain == (
        f"[0:v]scale={settings.output_width}:{settings.output_height}:"
        f"force_original_aspect_ratio=increase,"
        f"crop={settings.output_width}:{settings.output_height},"
        f"fps={settings.output_fps},setsar=1[v]"
    )


def test_blur_pad_mode_is_unaffected_by_a_crop_plan():
    settings.reframe_mode = "blur_pad"
    plan = face_tracking.CropPlan(mode="smart", anchor_y=50, keyframes=[
        face_tracking.PanKeyframe(0.0, 10), face_tracking.PanKeyframe(1.0, 200)
    ])
    chain = clipper.build_filter(has_subtitles=False, subtitle_name=None, crop_plan=plan, sendcmd_name="x.cmds")
    assert "sendcmd" not in chain
    assert "split=2" in chain  # still takes the blur_pad path, untouched


def test_a_static_plan_produces_a_plain_crop_with_computed_offsets():
    plan = face_tracking.CropPlan(mode="center", anchor_y=77, keyframes=[face_tracking.PanKeyframe(0.0, 33)])
    chain = clipper.build_filter(has_subtitles=False, subtitle_name=None, crop_plan=plan)
    assert "sendcmd" not in chain
    assert "crop=1080:1920:x=33:y=77" in chain


def test_an_animated_plan_inserts_sendcmd_before_a_named_crop_instance():
    plan = face_tracking.CropPlan(mode="smart", anchor_y=12, keyframes=[
        face_tracking.PanKeyframe(0.0, 5), face_tracking.PanKeyframe(1.0, 400)
    ])
    chain = clipper.build_filter(has_subtitles=False, subtitle_name=None, crop_plan=plan, sendcmd_name="clip.cmds")
    assert "sendcmd=f=clip.cmds" in chain
    assert "crop@pan=w=1080:h=1920:x=5:y=12" in chain
    assert chain.index("sendcmd") < chain.index("crop@pan"), "sendcmd must precede the filter it commands"


def test_captions_still_append_after_an_animated_crop():
    plan = face_tracking.CropPlan(mode="smart", anchor_y=0, keyframes=[
        face_tracking.PanKeyframe(0.0, 0), face_tracking.PanKeyframe(1.0, 50)
    ])
    chain = clipper.build_filter(
        has_subtitles=True, subtitle_name="c.ass", crop_plan=plan, sendcmd_name="c.cmds"
    )
    assert chain.endswith("ass=c.ass[v]")
    assert chain.index("crop@pan") < chain.index("ass=c.ass")


# --------------------------------------------------------- _scaled_dimensions


def test_scaled_dimensions_match_ffmpegs_own_arithmetic():
    """Verified empirically against a real ffmpeg scale filter output."""
    assert clipper._scaled_dimensions(1280, 720, 1080, 1920) == (3413, 1920)
    assert clipper._scaled_dimensions(480, 854, 1080, 1920) == (1080, 1922)


# ------------------------------------------------------------- render_clip


@pytest.mark.slow
def test_render_clip_with_smart_mode_actually_frames_an_off_centre_subject(tmp_path):
    """The headline proof: on a wide source with the subject off to one
    side, smart mode must produce meaningfully brighter (non-black) output
    than a plain centre crop, which lands in empty padding."""
    import subprocess

    settings.reframe_mode = "smart"
    dest = tmp_path / "smart.mp4"
    clipper.render_clip(clipper.RenderRequest(source=WIDE.resolve(), start=0.0, end=4.0, destination=dest, job_id="t"))

    settings.reframe_mode = "crop"
    dest_plain = tmp_path / "plain.mp4"
    clipper.render_clip(clipper.RenderRequest(source=WIDE.resolve(), start=0.0, end=4.0, destination=dest_plain, job_id="t"))

    def mean_brightness(path: Path) -> float:
        frame = tmp_path / f"{path.stem}.png"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", "2.0",
             "-i", str(path), "-frames:v", "1", str(frame)],
            check=True,
        )
        import cv2
        return float(cv2.imread(str(frame)).mean())

    assert mean_brightness(dest) > mean_brightness(dest_plain) + 30


@pytest.mark.slow
def test_smart_crop_survives_an_encoder_retry(tmp_path, monkeypatch):
    """A transient encoder failure (GPU contention, a driver hiccup) must not
    silently discard the smart-crop plan on the next attempt -- only the
    encoder should change. Regression test for a real bug: the first cut of
    this feature cleared crop_plan on ANY first-attempt failure, so smart
    crop would quietly vanish under ordinary NVENC contention, taking a
    genuinely well-framed clip back down to a naive centre crop with no
    error and no way to notice."""
    settings.reframe_mode = "smart"
    calls: list[str] = []
    real_run_ffmpeg = clipper.run_ffmpeg

    def flaky_first_attempt(args, **kwargs):
        calls.append(kwargs.get("label", ""))
        if len(calls) == 1:
            from app.services.media import FFmpegError

            raise FFmpegError("simulated transient encoder failure")
        return real_run_ffmpeg(args, **kwargs)

    monkeypatch.setattr(clipper, "run_ffmpeg", flaky_first_attempt)
    dest = tmp_path / "retry.mp4"
    clipper.render_clip(clipper.RenderRequest(source=WIDE.resolve(), start=0.0, end=4.0, destination=dest, job_id="t"))

    assert len(calls) == 2, "should have retried exactly once"
    assert dest.exists() and dest.stat().st_size > 0

    from app.services.media import video_info

    def mean_brightness(path):
        import subprocess

        import cv2

        frame = tmp_path / "retry_frame.png"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", "2.0",
             "-i", str(path), "-frames:v", "1", str(frame)],
            check=True,
        )
        return float(cv2.imread(str(frame)).mean())

    # The retried render must still be the smart, well-framed crop -- not the
    # plain centre crop the old buggy fallback would have quietly produced.
    assert mean_brightness(dest) > 60
    assert (video_info(dest)["width"], video_info(dest)["height"]) == (
        settings.output_width, settings.output_height
    )


@pytest.mark.slow
def test_render_clip_falls_back_safely_when_the_model_is_missing(tmp_path, monkeypatch):
    """Smart mode with a broken model must still produce a normal clip."""
    settings.reframe_mode = "smart"
    monkeypatch.setattr(settings, "face_model_path", Path("/nonexistent/model.onnx"))
    dest = tmp_path / "fallback.mp4"
    clipper.render_clip(clipper.RenderRequest(source=SAMPLE.resolve(), start=0.0, end=2.0, destination=dest, job_id="t"))
    assert dest.exists() and dest.stat().st_size > 0


@pytest.mark.slow
def test_render_clip_produces_the_configured_output_resolution_in_smart_mode(tmp_path):
    settings.reframe_mode = "smart"
    dest = tmp_path / "res.mp4"
    clipper.render_clip(clipper.RenderRequest(source=WIDE.resolve(), start=0.0, end=3.0, destination=dest, job_id="t"))
    from app.services.media import video_info

    info = video_info(dest)
    assert (info["width"], info["height"]) == (settings.output_width, settings.output_height)
