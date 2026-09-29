"""Choosing a transcription backend.

Whisper is the slowest stage, so it should use the GPU whenever the GPU can
run it at all -- and quietly stop trying when the GPU is busy with something
else. Both halves of that are easy to get wrong in ways that cost minutes per
job and never raise.
"""

from __future__ import annotations

import pytest

from app.config import settings
from app.services import transcription


@pytest.fixture(autouse=True)
def clear_model_cache():
    transcription._model_cache.clear()
    yield
    transcription._model_cache.clear()


class FakeWhisper:
    """Stands in for WhisperModel, accepting only some device/compute pairs."""

    def __init__(self, supported):
        self.supported = supported
        self.built: list[tuple[str, str]] = []

    def __call__(self, model, *, device, compute_type, cpu_threads=0):
        self.built.append((device, compute_type))
        if (device, compute_type) not in self.supported:
            raise ValueError(
                f"Requested {compute_type} compute type, but the target device "
                "or backend do not support it"
            )
        return f"model:{device}:{compute_type}"


@pytest.fixture
def whisper(monkeypatch):
    def install(supported):
        fake = FakeWhisper(supported)
        import faster_whisper

        monkeypatch.setattr(faster_whisper, "WhisperModel", fake)
        return fake

    return install


# ------------------------------------------------------------- precision


def test_cuda_tries_faster_precisions_before_slower_ones():
    order = transcription._compute_candidates("cuda", "default")
    assert order[0] == "float16"
    assert set(order) == {"float16", "int8_float16", "float32", "int8"}


def test_an_explicit_compute_type_is_tried_first():
    order = transcription._compute_candidates("cuda", "int8")
    assert order[0] == "int8"


def test_a_pascal_card_keeps_the_gpu_instead_of_dropping_to_the_cpu(monkeypatch, whisper):
    """Regression: one refusal used to mean the whole job ran on the CPU.

    A GTX 1080 Ti has no efficient float16 path and CTranslate2 refuses
    rather than downgrading, so float16 fails -- but float32 runs on that
    card perfectly well, and is far faster than any CPU fallback.
    """
    monkeypatch.setattr(settings, "whisper_device", "cuda")
    monkeypatch.setattr(settings, "whisper_compute_type", "default")
    fake = whisper({("cuda", "float32")})

    model, device = transcription._get_model()

    assert device == "cuda", "gave up the GPU when a supported precision existed"
    assert model == "model:cuda:float32"
    assert ("cuda", "float16") in fake.built, "should still prefer float16 first"
    assert ("cpu", "int8") not in fake.built


def test_the_cpu_is_still_the_last_resort(monkeypatch, whisper):
    monkeypatch.setattr(settings, "whisper_device", "cuda")
    monkeypatch.setattr(settings, "whisper_compute_type", "default")
    whisper({("cpu", "int8")})

    model, device = transcription._get_model()

    assert (model, device) == ("model:cpu:int8", "cpu")


def test_a_model_that_loads_nowhere_raises(monkeypatch, whisper):
    monkeypatch.setattr(settings, "whisper_device", "cuda")
    whisper(set())
    with pytest.raises(Exception):
        transcription._get_model()


# ------------------------------------------------------------ batch sizing


def test_batching_is_skipped_when_the_gpu_is_nearly_full(monkeypatch):
    """A CUDA OOM wedges the context, so the cheap fix is not to cause one."""
    monkeypatch.setattr(settings, "whisper_batch_size", 0)
    monkeypatch.setattr(transcription, "_free_vram_mib", lambda: 676)
    assert transcription._batch_size("cuda") == 0


@pytest.mark.parametrize(
    "free,expected", [(2000, 4), (4000, 8), (10000, 16), (None, 16)]
)
def test_batch_size_follows_available_memory(monkeypatch, free, expected):
    monkeypatch.setattr(settings, "whisper_batch_size", 0)
    monkeypatch.setattr(transcription, "_free_vram_mib", lambda: free)
    assert transcription._batch_size("cuda") == expected


def test_an_explicit_batch_size_is_honoured(monkeypatch):
    monkeypatch.setattr(settings, "whisper_batch_size", 32)
    monkeypatch.setattr(transcription, "_free_vram_mib", lambda: 500)
    assert transcription._batch_size("cuda") == 32


def test_the_cpu_never_batches(monkeypatch):
    monkeypatch.setattr(settings, "whisper_batch_size", 0)
    assert transcription._batch_size("cpu") == 0


# ---------------------------------------------------------------- fallback


def test_a_failed_batch_falls_back_to_sequential_exactly_once(monkeypatch, tmp_path):
    """Not a step-down: after an OOM every smaller batch fails too."""
    monkeypatch.setattr(transcription, "_get_model", lambda: ("model", "cuda"))
    monkeypatch.setattr(transcription, "_batch_size", lambda device: 16)

    attempts: list[int] = []

    def fake_run(model, audio_path, duration_hint, progress, *, batch_size):
        attempts.append(batch_size)
        if batch_size:
            raise RuntimeError("CUDA failed with error out of memory")
        return "transcript"

    monkeypatch.setattr(transcription, "_run", fake_run)

    assert transcription.transcribe(tmp_path / "a.wav") == "transcript"
    assert attempts == [16, 0], "one batched attempt, then straight to sequential"


def test_a_pipeline_error_from_transcription_is_not_swallowed(monkeypatch, tmp_path):
    from app.errors import ErrorCode, PipelineError

    monkeypatch.setattr(transcription, "_get_model", lambda: ("model", "cuda"))
    monkeypatch.setattr(transcription, "_batch_size", lambda device: 16)

    def fake_run(*args, **kwargs):
        raise PipelineError(ErrorCode.TRANSCRIPTION_FAILED, "no speech")

    monkeypatch.setattr(transcription, "_run", fake_run)

    with pytest.raises(PipelineError):
        transcription.transcribe(tmp_path / "a.wav")
