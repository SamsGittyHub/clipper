"""Face-aware crop planning (PRD phase 4).

Two fixtures under tests/fixtures/ are small real video clips (a few hundred
KB, trimmed and downscaled from a real recording) so the detector is
exercised against actual footage, not a synthetic pattern it could never
plausibly find a face in.

  face_sample.mp4  -- near enough to 9:16 already that there is no room to
                      pan; used for "does detection work at all" checks.
  face_wide.mp4    -- a 16:9 frame with the face content deliberately placed
                      off-centre (left third); used to prove the crop
                      actually follows the subject rather than sitting still.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import settings
from app.services import face_tracking as ft

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE = FIXTURES / "face_sample.mp4"
WIDE = FIXTURES / "face_wide.mp4"


@pytest.fixture(autouse=True)
def restore_reframe_mode():
    original = settings.reframe_mode
    yield
    settings.reframe_mode = original


# -------------------------------------------------------------- availability


def test_face_tracking_reports_available_when_opencv_and_model_are_present():
    assert ft.face_tracking_available() is True


def test_face_tracking_reports_unavailable_without_the_model(monkeypatch):
    monkeypatch.setattr(settings, "face_model_path", Path("/nonexistent/model.onnx"))
    assert ft.face_tracking_available() is False


# ------------------------------------------------------------------ plan_crop


def test_non_smart_reframe_mode_skips_detection_entirely(monkeypatch):
    """crop and blur_pad must never touch OpenCV -- cheap to prove via a spy."""
    settings.reframe_mode = "crop"
    monkeypatch.setattr(ft, "_sample_faces", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
    plan = ft.plan_crop(SAMPLE, start=0, end=4, scaled_w=1080, scaled_h=1920, target_w=1080, target_h=1920)
    assert plan.mode == "center"
    assert not plan.is_animated


def test_missing_opencv_falls_back_cleanly(monkeypatch):
    settings.reframe_mode = "smart"
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "cv2":
            raise ImportError("no cv2 here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    plan = ft.plan_crop(SAMPLE, start=0, end=4, scaled_w=1080, scaled_h=1920, target_w=1080, target_h=1920)
    assert plan.mode == "center"
    assert plan.reason == "opencv not installed"


def test_a_video_with_no_face_falls_back_to_a_static_centred_crop(monkeypatch):
    settings.reframe_mode = "smart"
    monkeypatch.setattr(ft, "_sample_faces", lambda *a, **k: [ft._Sample(t, None, None) for t in (0, 1, 2)])
    plan = ft.plan_crop(SAMPLE, start=0, end=3, scaled_w=1200, scaled_h=1920, target_w=1080, target_h=1920)
    assert plan.mode == "center"
    assert plan.reason == "no face detected"
    assert plan.keyframes == [ft.PanKeyframe(0.0, 60)]  # (1200-1080)/2, same as FFmpeg's own default


def test_a_detection_crash_never_propagates(monkeypatch):
    """A rendering feature must never be able to take a job down."""
    settings.reframe_mode = "smart"
    monkeypatch.setattr(ft, "_sample_faces", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("decoder exploded")))
    plan = ft.plan_crop(SAMPLE, start=0, end=3, scaled_w=1200, scaled_h=1920, target_w=1080, target_h=1920)
    assert plan.mode == "center"
    assert "decoder exploded" in plan.reason


def test_a_real_face_is_actually_detected_in_the_narrow_fixture():
    settings.reframe_mode = "smart"
    plan = ft.plan_crop(SAMPLE, start=0, end=4, scaled_w=1080, scaled_h=1920, target_w=1080, target_h=1920)
    assert plan.mode == "smart"
    assert plan.reason == "face tracked"


def test_the_crop_follows_a_face_placed_off_centre_in_a_wide_frame():
    """The headline case: a naive centre crop on a wide source misses the
    subject entirely; the plan must move toward wherever the face actually
    is instead of sitting at the geometric centre."""
    settings.reframe_mode = "smart"
    # Matches a 960x540 source scaled to cover a 1080x1920 target.
    scaled_w, scaled_h = 3413, 1920
    target_w, target_h = 1080, 1920
    plan = ft.plan_crop(WIDE, start=0, end=5, scaled_w=scaled_w, scaled_h=scaled_h, target_w=target_w, target_h=target_h)

    assert plan.mode == "smart"
    assert plan.is_animated, "there is ample room to pan in a 3413-wide scaled frame"

    naive_center_x = (scaled_w - target_w) // 2
    # The face sits in the left third of the source; every keyframe should
    # land well to the left of dead centre, in the region that actually
    # contains the subject.
    assert all(kf.x < naive_center_x - 500 for kf in plan.keyframes)
    assert all(0 <= kf.x <= scaled_w - target_w for kf in plan.keyframes)


def test_keyframes_are_monotonically_increasing_in_time():
    settings.reframe_mode = "smart"
    plan = ft.plan_crop(WIDE, start=0, end=5, scaled_w=3413, scaled_h=1920, target_w=1080, target_h=1920)
    times = [kf.t for kf in plan.keyframes]
    assert times == sorted(times)
    assert len(set(times)) == len(times)


def test_time_budget_stops_sampling_rather_than_hanging(monkeypatch):
    """A pathological source must not be able to stall a render indefinitely."""
    settings.reframe_mode = "smart"
    monkeypatch.setattr(settings, "face_detect_time_budget_s", 0.0)
    # Even with zero budget, at least the loop must terminate and produce a
    # usable (if minimal) result rather than hanging or raising.
    plan = ft.plan_crop(WIDE, start=0, end=30, scaled_w=3413, scaled_h=1920, target_w=1080, target_h=1920)
    assert plan.mode in ("smart", "center")


# --------------------------------------------------------------- smoothing


def test_smoothing_forward_and_back_fills_missing_samples():
    samples = [
        ft._Sample(0.0, None, None),
        ft._Sample(0.5, 0.2, 0.5),
        ft._Sample(1.0, None, None),
        ft._Sample(1.5, 0.8, 0.5),
    ]
    smoothed = ft._fill_and_smooth(samples)
    assert smoothed is not None
    assert len(smoothed) == 4
    # t=0 has no detection yet: back-filled from the first real one (0.2),
    # not left undefined or defaulted to the frame centre.
    assert smoothed[0][1] == pytest.approx(0.2)


def test_smoothing_returns_none_when_nothing_was_ever_detected():
    samples = [ft._Sample(t, None, None) for t in (0.0, 0.5, 1.0)]
    assert ft._fill_and_smooth(samples) is None


def test_ema_smoothing_moves_gradually_not_instantly(monkeypatch):
    """A hard cut in detected position should not appear as a hard cut in
    the crop -- that is the whole point of smoothing it."""
    monkeypatch.setattr(settings, "face_pan_ema_alpha", 0.3)
    samples = [ft._Sample(0.0, 0.1, 0.5)] + [ft._Sample(i * 0.5, 0.9, 0.5) for i in range(1, 6)]
    smoothed = ft._fill_and_smooth(samples)
    xs = [cx for _, cx, _ in smoothed]
    assert xs[0] == pytest.approx(0.1)
    assert xs[-1] < 0.9, "should not have snapped instantly to the new position"
    assert xs == sorted(xs), "should move monotonically toward the target, not overshoot"


def test_dead_zone_ignores_sub_threshold_movement(monkeypatch):
    monkeypatch.setattr(settings, "face_pan_dead_zone", 0.5)  # deliberately huge
    monkeypatch.setattr(settings, "face_pan_max_speed", 10.0)
    smoothed = [(0.0, 0.40, 0.5), (1.0, 0.41, 0.5), (2.0, 0.42, 0.5)]
    keyframes = ft._to_keyframes(smoothed, scaled_w=1000, target_w=500)
    assert len(keyframes) == 1, "tiny movements inside the dead zone should not emit new keyframes"


def test_max_speed_caps_how_fast_the_crop_can_move(monkeypatch):
    monkeypatch.setattr(settings, "face_pan_dead_zone", 0.0)
    monkeypatch.setattr(settings, "face_pan_max_speed", 0.1)  # 10% of width per second, deliberately slow
    smoothed = [(0.0, 0.0, 0.5), (1.0, 1.0, 0.5)]  # a full-width jump in one second
    keyframes = ft._to_keyframes(smoothed, scaled_w=1000, target_w=0)
    moved = keyframes[-1].x - keyframes[0].x
    assert moved <= 100 + 1, "must not move faster than the configured cap allows"


def test_keyframes_never_exceed_the_valid_crop_range():
    smoothed = [(0.0, -0.5, 0.5), (1.0, 1.5, 0.5)]  # deliberately out-of-range fractions
    keyframes = ft._to_keyframes(smoothed, scaled_w=1000, target_w=400)
    assert all(0 <= kf.x <= 600 for kf in keyframes)


# ------------------------------------------------------------ sendcmd file


def test_sendcmd_file_format(tmp_path):
    plan = ft.CropPlan(mode="smart", anchor_y=100, keyframes=[ft.PanKeyframe(0.0, 10), ft.PanKeyframe(1.5, 200)])
    path = ft.write_sendcmd_file(plan, tmp_path / "clip.cmds")
    content = path.read_text()
    assert content == "0.000 crop@pan x '10';\n1.500 crop@pan x '200';\n"
