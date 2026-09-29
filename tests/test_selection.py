"""Validation, deduplication and ranking (PRD 8.6)."""

from __future__ import annotations

import pytest

from app.config import settings
from app.errors import ErrorCode, PipelineError
from app.services.ai_highlights import HighlightCandidate
from app.services.selection import (
    deduplicate,
    duration_fit,
    novelty,
    overlap_ratio,
    rank,
    select,
    validate,
)


def candidate(start, end, **kwargs):
    return HighlightCandidate(start=start, end=end, **kwargs)


# ----------------------------------------------------------------- validate


def test_rejects_impossible_windows(transcript):
    bad = [
        candidate(50, 20),          # end before start
        candidate(5, 9),            # under the 15s minimum
        candidate(float("nan"), 5), # not a number
    ]
    kept, rejected = validate(bad, transcript=transcript, source_duration=60, max_duration=60)
    assert kept == []
    assert len(rejected) == 3


def test_clamps_and_trims_rather_than_discarding(transcript):
    # Overlong and past the end of the source, but built on a real moment.
    kept, _ = validate(
        [candidate(4, 400)], transcript=transcript, source_duration=25.0, max_duration=60
    )
    assert len(kept) == 1
    assert kept[0].start >= 0
    assert kept[0].end <= 25.0


def test_rejects_windows_with_almost_no_speech(transcript):
    silent_tail = transcript.duration + 5
    kept, rejected = validate(
        [candidate(silent_tail, silent_tail + 30)],
        transcript=transcript,
        source_duration=silent_tail + 60,
        max_duration=60,
    )
    assert kept == []
    assert "speech" in rejected[0].rule


def test_snaps_to_word_boundaries(transcript):
    word = transcript.words[3]
    kept, _ = validate(
        [candidate(word.start + 0.12, word.start + 20)],
        transcript=transcript,
        source_duration=transcript.duration,
        max_duration=60,
    )
    assert kept
    # Snapped to the word edge, then padded outward by the configured amount.
    assert kept[0].start == pytest.approx(word.start - settings.clip_padding_s, abs=0.01)


# ------------------------------------------------------------------ overlap


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ((10, 40), (20, 50), 20 / 30),   # overlap / shorter duration
        ((10, 70), (20, 40), 1.0),       # fully contained
        ((0, 10), (20, 30), 0.0),        # disjoint
        ((0, 10), (10, 20), 0.0),        # touching, not overlapping
    ],
)
def test_overlap_ratio(a, b, expected):
    assert overlap_ratio(a, b) == pytest.approx(expected)


def test_deduplicate_keeps_the_higher_ranked_clip(transcript):
    ranked = rank(
        [
            candidate(5, 35, viral_score=90),
            candidate(8, 38, viral_score=60),  # ~90% overlap with the first
            candidate(60, 90, viral_score=70),
        ],
        transcript=transcript,
    )
    kept = deduplicate(ranked, threshold=0.7)
    starts = sorted(clip.start for clip in kept)
    assert starts == [5, 60]


# ------------------------------------------------------------------ ranking


def test_duration_fit_peaks_in_the_recommended_band():
    assert duration_fit(40) == 1.0
    assert duration_fit(settings.ideal_duration_low) == 1.0
    assert duration_fit(settings.ideal_duration_high) == 1.0
    assert duration_fit(settings.min_clip_duration) == 0.0
    assert duration_fit(settings.max_clip_duration) == 0.0
    assert 0 < duration_fit(20) < 1


def test_novelty_penalises_repeated_vocabulary():
    assert novelty("alpha beta gamma", []) == 1.0
    assert novelty("alpha beta gamma", [{"alpha", "beta", "gamma"}]) == 0.0
    assert 0 < novelty("alpha beta delta", [{"alpha", "beta", "gamma"}]) < 1


def test_final_score_is_the_weighted_sum_of_its_parts(transcript):
    [clip] = rank([candidate(5, 35, viral_score=80, hook_score=70, emotion_score=60)], transcript=transcript)
    breakdown = clip.breakdown
    assert sum(breakdown["weights"].values()) == pytest.approx(1.0)
    assert sum(breakdown["contributions"].values()) == pytest.approx(clip.final_score, abs=0.05)
    assert breakdown["signals"]["ai_score"] == pytest.approx(0.80)
    assert 0 <= clip.final_score <= 100


def test_ranking_is_ordered_by_final_score(transcript):
    ranked = rank(
        [candidate(60, 90, viral_score=40), candidate(5, 35, viral_score=95)],
        transcript=transcript,
    )
    assert [clip.final_score for clip in ranked] == sorted(
        (clip.final_score for clip in ranked), reverse=True
    )


# ------------------------------------------------------------------ select


def test_select_returns_at_most_the_requested_count(transcript):
    chosen, _ = select(
        [candidate(5, 35, viral_score=90), candidate(60, 95, viral_score=80)],
        transcript=transcript,
        source_duration=transcript.duration,
        max_duration=60,
        clip_count=1,
    )
    assert len(chosen) == 1
    # Padded outward from the requested start by `clip_padding_s`.
    assert chosen[0].start == pytest.approx(5 - settings.clip_padding_s)


def test_select_raises_an_actionable_error_when_nothing_survives(transcript):
    with pytest.raises(PipelineError) as excinfo:
        select(
            [candidate(10, 11)],
            transcript=transcript,
            source_duration=transcript.duration,
            max_duration=60,
            clip_count=3,
        )
    assert excinfo.value.code == ErrorCode.NO_VALID_CANDIDATES
