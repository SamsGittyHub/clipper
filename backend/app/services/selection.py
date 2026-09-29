"""Validation, deduplication and ranking (PRD 8.6).

AI output is never trusted. Every candidate is checked against the real
transcript, clamped to the source, snapped to word boundaries, deduplicated by
overlap, and scored with a breakdown that explains the result.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from ..config import settings
from ..errors import ErrorCode, PipelineError
from .ai_highlights import HighlightCandidate
from .transcript import Transcript

log = logging.getLogger("clipz.selection")


@dataclass
class ScoredClip:
    """A validated candidate with an explainable final score."""

    start: float
    end: float
    ai_score: float
    hook_score: float
    emotion_score: float
    speech_density: float
    duration_fit: float
    transcript_confidence: float
    novelty: float
    final_score: float
    breakdown: dict[str, Any]
    reason: str | None
    title: str | None
    hook: str | None
    description: str | None
    hashtags: list[str] = field(default_factory=list)
    transcript_text: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class RejectedCandidate:
    start: float
    end: float
    rule: str


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def validate(
    candidates: list[HighlightCandidate],
    *,
    transcript: Transcript,
    source_duration: float,
    max_duration: float,
) -> tuple[list[HighlightCandidate], list[RejectedCandidate]]:
    """Drop or repair candidates that do not describe a real, usable window."""
    kept: list[HighlightCandidate] = []
    rejected: list[RejectedCandidate] = []
    limit = source_duration if source_duration > 0 else transcript.duration
    max_allowed = min(max_duration, settings.max_clip_duration)

    for candidate in candidates:
        start, end = float(candidate.start), float(candidate.end)

        if start != start or end != end:  # NaN
            rejected.append(RejectedCandidate(start, end, "timestamps were not numbers"))
            continue
        if end <= start:
            rejected.append(RejectedCandidate(start, end, "end was not after start"))
            continue

        start = max(0.0, start)
        if limit:
            end = min(end, limit)
        if end - start < settings.min_clip_duration:
            rejected.append(
                RejectedCandidate(start, end, f"shorter than {settings.min_clip_duration:.0f}s")
            )
            continue

        # Trim an over-long window rather than discarding a good moment.
        if end - start > max_allowed:
            end = start + max_allowed

        # Snap outward to word boundaries so speech is never clipped mid-word.
        snapped_start = transcript.nearest_word_boundary(start, side="start")
        snapped_end = transcript.nearest_word_boundary(end, side="end")
        if snapped_end - snapped_start >= settings.min_clip_duration:
            start, end = snapped_start, snapped_end
        start = max(0.0, start - settings.clip_padding_s)
        end = min(limit or end + settings.clip_padding_s, end + settings.clip_padding_s)

        if end - start > max_allowed:
            end = start + max_allowed
        if end - start < settings.min_clip_duration:
            rejected.append(RejectedCandidate(start, end, "too short after boundary snapping"))
            continue

        text = transcript.text_between(start, end)
        if len(text.split()) < 8:
            rejected.append(RejectedCandidate(start, end, "almost no speech in the window"))
            continue

        candidate.start, candidate.end = start, end
        kept.append(candidate)

    log.info("Validation kept %d of %d candidates", len(kept), len(candidates))
    for item in rejected:
        log.info("Rejected %.1f-%.1f: %s", item.start, item.end, item.rule)
    return kept, rejected


# --------------------------------------------------------------------------
# Deduplication
# --------------------------------------------------------------------------


def overlap_ratio(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Overlapping duration divided by the shorter clip's duration (PRD 8.6)."""
    overlap = min(a[1], b[1]) - max(a[0], b[0])
    if overlap <= 0:
        return 0.0
    shorter = min(a[1] - a[0], b[1] - b[0])
    if shorter <= 0:
        return 0.0
    return overlap / shorter


def deduplicate(clips: list[ScoredClip], *, threshold: float | None = None) -> list[ScoredClip]:
    """Keep the higher-ranked clip of any overlapping pair."""
    limit = settings.overlap_threshold if threshold is None else threshold
    ordered = sorted(clips, key=lambda clip: clip.final_score, reverse=True)
    kept: list[ScoredClip] = []
    for clip in ordered:
        duplicate_of = next(
            (
                other
                for other in kept
                if overlap_ratio((clip.start, clip.end), (other.start, other.end)) > limit
            ),
            None,
        )
        if duplicate_of is None:
            kept.append(clip)
        else:
            log.info(
                "Dropped %.1f-%.1f as %.0f%% duplicate of %.1f-%.1f",
                clip.start,
                clip.end,
                overlap_ratio((clip.start, clip.end), (duplicate_of.start, duplicate_of.end)) * 100,
                duplicate_of.start,
                duplicate_of.end,
            )
    return kept


# --------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------


def _weights() -> dict[str, float]:
    raw = {
        "ai_score": settings.weight_ai_score,
        "hook": settings.weight_hook,
        "emotion": settings.weight_emotion,
        "speech_density": settings.weight_speech_density,
        "duration_fit": settings.weight_duration_fit,
        "confidence": settings.weight_confidence,
        "novelty": settings.weight_novelty,
    }
    total = sum(raw.values()) or 1.0
    return {key: value / total for key, value in raw.items()}


def duration_fit(duration: float) -> float:
    """1.0 inside the recommended band, tapering to 0 at the hard limits."""
    low, high = settings.ideal_duration_low, settings.ideal_duration_high
    if low <= duration <= high:
        return 1.0
    if duration < low:
        floor = settings.min_clip_duration
        return max(0.0, (duration - floor) / max(1e-6, low - floor))
    ceiling = settings.max_clip_duration
    return max(0.0, (ceiling - duration) / max(1e-6, ceiling - high))


def speech_density(transcript: Transcript, start: float, end: float) -> float:
    """Share of the window that is actual speech."""
    span = max(1e-6, end - start)
    return max(0.0, min(1.0, transcript.speech_seconds_between(start, end) / span))


_TOKEN = re.compile(r"[a-z0-9']+")
_STOPWORDS = frozenset(
    """a an and are as at be but by for from has have he her his i if in is it its me my
    not of on or our she so that the their them there they this to was we were what when
    who will with you your just like really very can do does did been am are'nt""".split()
)


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS and len(t) > 2}


def novelty(text: str, previous: list[set[str]]) -> float:
    """1.0 when a clip shares no vocabulary with higher-ranked clips."""
    if not previous:
        return 1.0
    tokens = _tokens(text)
    if not tokens:
        return 0.0
    worst_overlap = max(
        (len(tokens & other) / len(tokens | other) if (tokens | other) else 0.0)
        for other in previous
    )
    return max(0.0, 1.0 - worst_overlap)


def rank(
    candidates: list[HighlightCandidate],
    *,
    transcript: Transcript,
) -> list[ScoredClip]:
    """Combine AI judgement with measured signals into an explainable score."""
    weights = _weights()

    # Novelty is relative to stronger clips, so seed the pass in AI order.
    ordered = sorted(candidates, key=lambda c: c.viral_score, reverse=True)
    seen_tokens: list[set[str]] = []
    scored: list[ScoredClip] = []

    for candidate in ordered:
        text = transcript.text_between(candidate.start, candidate.end)
        signals = {
            "ai_score": candidate.viral_score / 100.0,
            "hook": candidate.hook_score / 100.0,
            "emotion": candidate.emotion_score / 100.0,
            "speech_density": speech_density(transcript, candidate.start, candidate.end),
            "duration_fit": duration_fit(candidate.end - candidate.start),
            "confidence": transcript.confidence_between(candidate.start, candidate.end),
            "novelty": novelty(text, seen_tokens),
        }
        seen_tokens.append(_tokens(text))

        contributions = {key: signals[key] * weights[key] for key in weights}
        final = round(sum(contributions.values()) * 100.0, 2)

        scored.append(
            ScoredClip(
                start=candidate.start,
                end=candidate.end,
                ai_score=candidate.viral_score,
                hook_score=candidate.hook_score,
                emotion_score=candidate.emotion_score,
                speech_density=round(signals["speech_density"], 4),
                duration_fit=round(signals["duration_fit"], 4),
                transcript_confidence=round(signals["confidence"], 4),
                novelty=round(signals["novelty"], 4),
                final_score=final,
                breakdown={
                    "weights": {k: round(v, 4) for k, v in weights.items()},
                    "signals": {k: round(v, 4) for k, v in signals.items()},
                    "contributions": {k: round(v * 100, 2) for k, v in contributions.items()},
                    "final_score": final,
                },
                reason=candidate.reason,
                title=candidate.title,
                hook=candidate.hook,
                description=candidate.description,
                hashtags=list(candidate.hashtags),
                transcript_text=text,
            )
        )

    scored.sort(key=lambda clip: clip.final_score, reverse=True)
    return scored


def select(
    candidates: list[HighlightCandidate],
    *,
    transcript: Transcript,
    source_duration: float,
    max_duration: float,
    clip_count: int,
) -> tuple[list[ScoredClip], list[RejectedCandidate]]:
    """Validate, rank, deduplicate, and take the top `clip_count`."""
    valid, rejected = validate(
        candidates,
        transcript=transcript,
        source_duration=source_duration,
        max_duration=max_duration,
    )
    if not valid:
        raise PipelineError(
            ErrorCode.NO_VALID_CANDIDATES,
            "No highlight candidate survived validation",
            detail="; ".join(f"{r.start:.0f}-{r.end:.0f}s: {r.rule}" for r in rejected[:10]),
        )

    ranked = rank(valid, transcript=transcript)
    deduplicated = deduplicate(ranked)
    return deduplicated[:clip_count], rejected
