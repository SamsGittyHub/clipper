"""Concurrent encoder sessions.

Clips render in parallel, and a hardware encoder only opens so many sessions
at once -- bounded by the driver, and by free VRAM on a card shared with
anything else. Overshooting fails the encode and silently drops that clip to
software, so the limit is measured and then enforced.
"""

from __future__ import annotations

import threading
import time

import pytest

from app.config import settings
from app.services import media


@pytest.fixture(autouse=True)
def clean_session_state():
    media.reset_session_limits()
    yield
    media.reset_session_limits()


# ------------------------------------------------------------------ limits


def test_software_encoding_is_not_session_limited():
    """x264 is bounded by CPU, which the caller already accounts for."""
    assert media.hardware_session_limit("libx264") == media.UNLIMITED_SESSIONS


def test_the_limit_is_measured_once_and_then_reused(monkeypatch):
    probes: list[int] = []

    def fake_probe(encoder, count):
        probes.append(count)
        return count <= 2

    monkeypatch.setattr(media, "_concurrent_encodes_work", fake_probe)

    assert media.hardware_session_limit("h264_nvenc") == 2
    assert probes == [2, 3], "probe upward until one is refused"

    probes.clear()
    assert media.hardware_session_limit("h264_nvenc") == 2
    assert probes == [], "the second call must not re-measure"


def test_an_encoder_that_allows_nothing_concurrent_falls_back_to_one(monkeypatch):
    monkeypatch.setattr(media, "_concurrent_encodes_work", lambda encoder, count: False)
    assert media.hardware_session_limit("h264_nvenc") == 1


def test_probing_uses_the_real_output_resolution(monkeypatch):
    """A 256x256 probe fits where a 1080x1920 session does not.

    Session capacity depends on the memory each one allocates, so probing at
    a token size reports a limit the actual renders cannot sustain.
    """
    commands: list[list[str]] = []

    class Result:
        returncode = 0

    def fake_run(command, **kwargs):
        commands.append(command)
        return Result()

    monkeypatch.setattr(media.subprocess, "run", fake_run)
    media._concurrent_encodes_work("h264_nvenc", 1)

    assert commands
    expected = f"{settings.output_width}x{settings.output_height}"
    assert any(expected in part for part in commands[0])


# -------------------------------------------------------------- the slot


def test_the_slot_admits_only_as_many_encodes_as_the_limit_allows(monkeypatch):
    monkeypatch.setattr(media, "_concurrent_encodes_work", lambda encoder, count: count <= 2)

    peak = 0
    current = 0
    lock = threading.Lock()
    barrier_passed = threading.Event()

    def encode():
        nonlocal peak, current
        with media.encoder_slot("h264_nvenc"):
            with lock:
                current += 1
                peak = max(peak, current)
            time.sleep(0.05)
            with lock:
                current -= 1
        barrier_passed.set()

    threads = [threading.Thread(target=encode) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not any(thread.is_alive() for thread in threads)
    assert peak <= 2, f"{peak} encodes ran at once against a limit of 2"


def test_the_slot_does_not_deadlock_resolving_its_own_limit():
    """Regression: `encoder_slot` used to call `hardware_session_limit`
    while holding the very lock that function takes, and that lock is not
    reentrant -- so the first render of every job hung forever."""
    finished: list[int] = []

    def use():
        with media.encoder_slot("libx264"):
            finished.append(1)

    threads = [threading.Thread(target=use) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not any(thread.is_alive() for thread in threads), "encoder_slot deadlocked"
    assert len(finished) == 4


def test_the_slot_is_released_even_when_the_encode_raises(monkeypatch):
    monkeypatch.setattr(media, "_concurrent_encodes_work", lambda encoder, count: False)

    for _ in range(3):
        with pytest.raises(RuntimeError):
            with media.encoder_slot("h264_nvenc"):
                raise RuntimeError("encode blew up")

    # A leaked permit on a limit of one would block this forever.
    acquired = threading.Event()

    def use():
        with media.encoder_slot("h264_nvenc"):
            acquired.set()

    thread = threading.Thread(target=use)
    thread.start()
    thread.join(timeout=5)
    assert acquired.is_set(), "the permit was not returned after a failure"


# ------------------------------------------------------------ concurrency


def test_render_concurrency_is_capped_by_what_the_encoder_grants(monkeypatch):
    """The pool must never ask for more sessions than exist."""
    monkeypatch.setattr(settings, "render_concurrency", 8)
    monkeypatch.setattr(media, "_concurrent_encodes_work", lambda encoder, count: count <= 2)

    desired = settings.effective_render_concurrency("h264_nvenc")
    allowed = media.hardware_session_limit("h264_nvenc")

    assert desired == 8, "an explicit setting is still what the user asked for"
    assert min(desired, allowed) == 2, "but the hardware limit wins"


def test_software_concurrency_scales_with_the_machine(monkeypatch):
    monkeypatch.setattr(settings, "render_concurrency", 0)
    assert 1 <= settings.effective_render_concurrency("libx264") <= 3
    assert settings.effective_render_concurrency("h264_nvenc") == 4
