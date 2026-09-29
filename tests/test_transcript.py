"""Transcript normalisation and lookup (PRD 8.4)."""

from __future__ import annotations

from app.services.transcript import Transcript, format_timestamp, parse_timestamp


def test_sentences_preserve_source_timing(transcript):
    sentences = transcript.sentences()
    assert len(sentences) == 5
    assert sentences[0].text.startswith("Nobody talks about this")
    # Sentence timing must match the words it was built from.
    assert sentences[0].start == transcript.segments[0].words[0].start
    assert sentences[0].end == transcript.segments[0].words[-1].end


def test_prompt_text_is_timestamped_and_paragraphed(transcript):
    prompt = transcript.to_prompt_text()
    assert prompt.startswith("[00:05] Nobody talks about this")
    assert prompt.count("[") == 5
    # The 0.8s inter-sentence pause is below the paragraph threshold.
    assert "\n\n" not in prompt


def test_prompt_text_truncates_on_whole_lines(transcript):
    prompt = transcript.to_prompt_text(max_chars=120)
    assert "[transcript truncated]" in prompt
    for line in prompt.splitlines():
        assert not line or line.startswith("[")


def test_text_between_uses_word_midpoints(transcript):
    first = transcript.segments[0]
    text = transcript.text_between(first.start, first.end)
    assert "Nobody" in text and "business." in text
    assert "truth" not in text


def test_speech_density_and_confidence(transcript):
    first = transcript.segments[0]
    density = transcript.speech_seconds_between(first.start, first.end)
    assert 0 < density <= (first.end - first.start)
    assert 0.9 <= transcript.confidence_between(first.start, first.end) <= 0.95


def test_boundary_snapping_moves_to_real_word_edges(transcript):
    word = transcript.words[4]
    # A timestamp landing mid-word snaps out to that word's own boundary.
    snapped = transcript.nearest_word_boundary(word.start + 0.1, side="start")
    assert snapped == word.start
    far = transcript.nearest_word_boundary(9999.0, side="end")
    assert far == 9999.0  # nothing within the window: leave it alone


def test_round_trip_serialisation(transcript, tmp_path):
    path = transcript.save(tmp_path / "t.json")
    restored = Transcript.load(path)
    assert restored.text == transcript.text
    assert len(restored.words) == len(transcript.words)
    assert restored.words[0].probability == transcript.words[0].probability


def test_timestamp_helpers():
    assert format_timestamp(65) == "01:05"
    assert format_timestamp(3725) == "1:02:05"
    assert parse_timestamp("1:02:05") == 3725
    assert parse_timestamp(12.5) == 12.5
    assert parse_timestamp("garbage") is None
    assert parse_timestamp(None) is None
