"""Transcription via faster-whisper (PRD 8.4).

Word-level timing is mandatory: the caption engine and boundary snapping both
depend on it, so `word_timestamps=True` is not optional here.

Two execution paths share one loaded model:

* batched -- `BatchedInferencePipeline` splits the audio on VAD silences and
  runs several chunks through the model per forward pass. On a GPU it is
  several times faster than sequential decoding with no quality change that
  matters for clip selection. It is the default on CUDA.
* sequential -- the classic path, used on CPU (where batching gains little)
  and as the automatic fallback if the batched path raises for any reason.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
from pathlib import Path
from typing import Callable

from ..config import settings
from ..errors import ErrorCode, PipelineError
from .transcript import Segment, Transcript, Word

log = logging.getLogger("clipz.transcription")

ProgressFn = Callable[[float, str], None]

_model_cache: dict[tuple[str, str, str], object] = {}
_model_lock = threading.Lock()


def _resolve_device() -> tuple[str, str]:
    """Choose device/compute type, preferring CUDA when it is actually usable."""
    device = settings.whisper_device
    compute = settings.whisper_compute_type

    if device == "auto":
        device = "cuda" if _cuda_available() else "cpu"
    if compute == "default":
        compute = "float16" if device == "cuda" else "int8"
    return device, compute


def _compute_candidates(device: str, configured: str) -> list[str]:
    """Compute types to try on `device`, best first.

    float16 is the right default on modern NVIDIA hardware, but Pascal cards
    (a GTX 1080 Ti, say) have no efficient float16 path and CTranslate2
    refuses outright rather than falling back. Treating that single refusal
    as "no GPU here" gives up the GPU for the whole job, when int8_float16 or
    float32 would have run on it perfectly well.
    """
    if configured != "default":
        # An explicit choice is honoured first, then the same safety net.
        preferred = [configured]
    elif device == "cuda":
        preferred = ["float16"]
    else:
        preferred = ["int8"]

    fallbacks = ["int8_float16", "float32", "int8"] if device == "cuda" else ["float32"]
    ordered: list[str] = []
    for candidate in preferred + fallbacks:
        if candidate not in ordered:
            ordered.append(candidate)
    return ordered


def _cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:  # pragma: no cover - depends on local CUDA install
        return False


def _cpu_threads() -> int:
    if settings.whisper_cpu_threads > 0:
        return settings.whisper_cpu_threads
    return max(1, min(8, (os.cpu_count() or 4) // 2))


def _free_vram_mib() -> int | None:
    """Free VRAM on the first GPU, or None if it cannot be determined."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - no nvidia-smi
        return None
    if result.returncode != 0:
        return None
    first = (result.stdout or "").strip().splitlines()
    try:
        return int(first[0].strip())
    except (IndexError, ValueError):  # pragma: no cover - unexpected format
        return None


def _batch_size(device: str) -> int:
    """How many chunks to decode per forward pass.

    Batching is the single biggest transcription speed-up on a GPU, and the
    first thing to run out of memory on a card shared with anything else --
    another encode, a game, a resident language model. Recovering from that
    OOM is expensive rather than cheap: CUDA leaves the context wedged, so
    the next attempt fails with `invalid device ordinal` whatever size it
    asks for. Sizing the batch to the memory actually free avoids paying for
    a failure at all.
    """
    if settings.whisper_batch_size > 0:
        return settings.whisper_batch_size
    if device != "cuda":
        return 0

    free = _free_vram_mib()
    if free is None:
        return 16  # cannot tell; the single fallback still covers a failure
    if free < 1500:
        log.info("Only %d MiB of VRAM free; transcribing without batching", free)
        return 0
    if free < 3000:
        return 4
    if free < 6000:
        return 8
    return 16


def _get_model():
    """Load the transcription model, keeping the GPU if any precision works."""
    device, compute = _resolve_device()
    key = (settings.whisper_model, device, compute)
    with _model_lock:
        if key in _model_cache:
            return _model_cache[key], key[1]

        from faster_whisper import WhisperModel

        attempts: list[tuple[str, str]] = [
            (device, candidate)
            for candidate in _compute_candidates(device, settings.whisper_compute_type)
        ]
        if device != "cpu":
            attempts.append(("cpu", "int8"))  # last resort: always available

        last_error: Exception | None = None
        for attempt_device, attempt_compute in attempts:
            cached_key = (settings.whisper_model, attempt_device, attempt_compute)
            if cached_key in _model_cache:
                return _model_cache[cached_key], attempt_device
            try:
                log.info("Loading whisper model=%s device=%s compute=%s", *cached_key)
                model = WhisperModel(
                    settings.whisper_model,
                    device=attempt_device,
                    compute_type=attempt_compute,
                    cpu_threads=_cpu_threads(),
                )
            except Exception as exc:
                last_error = exc
                log.warning(
                    "Whisper %s/%s unavailable (%s)", attempt_device, attempt_compute, exc
                )
                continue
            if (attempt_device, attempt_compute) != (device, compute):
                log.info(
                    "Transcribing on %s/%s instead of the requested %s/%s",
                    attempt_device, attempt_compute, device, compute,
                )
            _model_cache[cached_key] = model
            return model, attempt_device

        raise last_error or RuntimeError("No usable transcription backend")


def _decode_options() -> dict:
    return {
        "language": settings.whisper_language,
        "beam_size": settings.whisper_beam_size,
        "word_timestamps": True,
        "vad_filter": True,
        "vad_parameters": {"min_silence_duration_ms": 500},
    }


def transcribe(
    audio_path: Path,
    *,
    duration_hint: float | None = None,
    progress: ProgressFn | None = None,
) -> Transcript:
    """Produce a word-timed transcript for `audio_path`."""
    try:
        model, device = _get_model()
    except Exception as exc:
        raise PipelineError(
            ErrorCode.TRANSCRIPTION_FAILED,
            "Could not load the transcription model",
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc

    # Batching is an optimisation, never a requirement. One fallback, not a
    # step-down: a CUDA OOM wedges the context, so every smaller retry fails
    # too and only adds delay before the sequential path that always works.
    batch = _batch_size(device)
    if batch > 0:
        try:
            return _run(model, audio_path, duration_hint, progress, batch_size=batch)
        except PipelineError:
            raise
        except Exception as exc:
            log.warning("Batched transcription failed (%s); using the sequential path", exc)
    return _run(model, audio_path, duration_hint, progress, batch_size=0)


def _run(
    model,
    audio_path: Path,
    duration_hint: float | None,
    progress: ProgressFn | None,
    *,
    batch_size: int,
) -> Transcript:
    try:
        if batch_size > 0:
            from faster_whisper import BatchedInferencePipeline

            raw_segments, info = BatchedInferencePipeline(model=model).transcribe(
                str(audio_path), batch_size=batch_size, **_decode_options()
            )
        else:
            raw_segments, info = model.transcribe(str(audio_path), **_decode_options())

        total = duration_hint or getattr(info, "duration", 0.0) or 0.0
        segments: list[Segment] = []
        texts: list[str] = []

        # faster-whisper streams segments lazily; consuming the generator is
        # what actually performs the work, so progress is reported here. The
        # progress callback is also where a cancel surfaces (it raises).
        for raw in raw_segments:
            words = [
                Word(
                    start=float(word.start),
                    end=float(word.end),
                    text=word.word,
                    probability=float(getattr(word, "probability", 1.0) or 0.0),
                )
                for word in (raw.words or [])
                if word.start is not None and word.end is not None
            ]
            segments.append(
                Segment(
                    start=float(raw.start),
                    end=float(raw.end),
                    text=raw.text.strip(),
                    words=words,
                    confidence=_confidence_of(raw),
                )
            )
            texts.append(raw.text.strip())
            if progress and total:
                progress(min(0.99, raw.end / total), f"Transcribing ({raw.end / total * 100:.0f}%)")
    except PipelineError:
        raise
    except Exception as exc:
        if batch_size > 0:
            raise  # let the caller fall back to the sequential path
        raise PipelineError(
            ErrorCode.TRANSCRIPTION_FAILED,
            "Transcription failed",
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc

    if not segments:
        raise PipelineError(
            ErrorCode.TRANSCRIPTION_FAILED,
            "No speech was detected in the source audio",
        )

    transcript = Transcript(
        text=" ".join(texts).strip(),
        language=getattr(info, "language", None) or "en",
        language_probability=float(getattr(info, "language_probability", 1.0) or 1.0),
        duration=float(duration_hint or getattr(info, "duration", 0.0) or segments[-1].end),
        segments=segments,
    )
    log.info(
        "Transcribed %s: %d segments, %d words, language=%s (%s)",
        audio_path.name,
        len(transcript.segments),
        len(transcript.words),
        transcript.language,
        f"batched x{batch_size}" if batch_size else "sequential",
    )
    return transcript


def _confidence_of(raw) -> float:
    """Convert whisper's mean log-probability into a 0-1 confidence."""
    import math

    logprob = getattr(raw, "avg_logprob", None)
    if logprob is None:
        return 1.0
    return max(0.0, min(1.0, math.exp(float(logprob))))
