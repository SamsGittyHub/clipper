"""Actionable error codes (PRD 12) and the exception the worker records."""

from __future__ import annotations


class ErrorCode:
    DOWNLOAD_FAILED = "ERROR_DOWNLOAD_FAILED"
    UNSUPPORTED_SOURCE = "ERROR_UNSUPPORTED_SOURCE"
    AUDIO_EXTRACTION_FAILED = "ERROR_AUDIO_EXTRACTION_FAILED"
    TRANSCRIPTION_FAILED = "ERROR_TRANSCRIPTION_FAILED"
    AI_ANALYSIS_FAILED = "ERROR_AI_ANALYSIS_FAILED"
    AI_INVALID_OUTPUT = "ERROR_AI_INVALID_OUTPUT"
    NO_VALID_CANDIDATES = "ERROR_NO_VALID_CANDIDATES"
    RENDER_FAILED = "ERROR_RENDER_FAILED"
    CAPTION_FAILED = "ERROR_CAPTION_FAILED"
    CANCELLED = "ERROR_CANCELLED"
    INTERRUPTED = "ERROR_INTERRUPTED"
    INTERNAL = "ERROR_INTERNAL"


class PipelineError(Exception):
    """A stage failure carrying a stable, actionable code."""

    def __init__(self, code: str, message: str, *, detail: str | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return f"[{self.code}] {self.message}"


class JobCancelled(PipelineError):
    """Raised inside the pipeline when the user cancels the job.

    A subclass of PipelineError so every `except PipelineError: raise` in the
    stages lets it through untouched, and the worker can tell it apart from a
    real failure when it records the terminal state.
    """

    def __init__(self, message: str = "Cancelled by the user"):
        super().__init__(ErrorCode.CANCELLED, message)
