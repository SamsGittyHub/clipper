"""ASS caption generation (PRD 8.8)."""

from __future__ import annotations

import pytest

from app.services.captions import (
    CAPTION_STYLES,
    build_ass,
    build_lines,
    display_ends,
    resolve_style,
)
from app.services.transcript import Word


def parse_events(path):
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Dialogue:"):
            _, start, end, *rest = line.split(",", 3)
            events.append((to_seconds(start), to_seconds(end), rest[-1]))
    return events


def to_seconds(stamp: str) -> float:
    hours, minutes, seconds = stamp.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


@pytest.mark.parametrize("style", CAPTION_STYLES)
def test_every_style_produces_a_valid_timed_file(transcript, tmp_path, style):
    path = build_ass(transcript, start=5.0, end=20.0, style=style, destination=tmp_path / "c.ass")
    content = path.read_text(encoding="utf-8")
    assert "[Script Info]" in content and "[V4+ Styles]" in content and "[Events]" in content
    assert "PlayResX: 1080" in content and "PlayResY: 1920" in content

    events = parse_events(path)
    assert events, "no dialogue events were written"
    for start, end, _ in events:
        assert 0 <= start < end <= 15.0 + 0.5  # relative to the clip, inside its span


@pytest.mark.parametrize("style", CAPTION_STYLES)
def test_no_visual_gaps_between_phrases(transcript, tmp_path, style):
    """Captions hold until the next phrase, except across a long silence."""
    from app.services.captions import MAX_HOLD

    path = build_ass(transcript, start=5.0, end=25.0, style=style, destination=tmp_path / "c.ass")
    events = parse_events(path)
    for (start, end, _), (next_start, _, _) in zip(events, events[1:]):
        gap = next_start - end
        assert gap <= 1e-6 or gap >= MAX_HOLD - 1e-6, "captions flicker off mid-speech"


def test_window_with_no_words_returns_none(transcript, tmp_path):
    assert build_ass(
        transcript, start=900.0, end=930.0, style="dynamic", destination=tmp_path / "c.ass"
    ) is None


def test_line_grouping_respects_the_character_budget(transcript):
    lines = build_lines(transcript.words, style="dynamic")
    assert lines
    for line in lines:
        assert len(line.text) <= 40  # budget plus one trailing word
        assert line.start <= line.end


def test_chunks_style_uses_shorter_bursts(transcript):
    dynamic = build_lines(transcript.words, style="dynamic")
    chunks = build_lines(transcript.words, style="chunks")
    assert len(chunks) > len(dynamic)
    assert all(len(line.words) <= 3 for line in chunks)


def test_display_ends_hold_until_the_next_phrase(transcript):
    lines = build_lines(transcript.words, style="classic")
    ends = display_ends(lines)
    for index, (line, end) in enumerate(zip(lines, ends)):
        assert end >= line.end
        if index + 1 < len(lines):
            assert end <= lines[index + 1].start + 1e-9


def test_override_syntax_in_speech_is_neutralised(tmp_path, transcript):
    from app.services.transcript import Segment, Transcript

    hostile = "{\\an8}drop shadow \\N injection"
    words = [Word(start=1.0 + i * 0.3, end=1.25 + i * 0.3, text=token)
             for i, token in enumerate(hostile.split())]
    doctored = Transcript(
        text=hostile,
        language="en",
        segments=[Segment(start=1.0, end=3.0, text=hostile, words=words)],
        duration=10.0,
    )
    path = build_ass(doctored, start=0.0, end=10.0, style="classic", destination=tmp_path / "c.ass")
    body = "\n".join(line for line in path.read_text().splitlines() if line.startswith("Dialogue"))
    assert "{\\an8}" not in body
    assert "(" in body  # braces were rewritten to parentheses


def test_style_aliases_resolve():
    assert resolve_style("dynamic pop") == "dynamic"
    assert resolve_style("TikTok") == "chunks"
    assert resolve_style("highlight word") == "highlight"
    assert resolve_style(None) == "dynamic"
    assert resolve_style("nonsense") == "dynamic"
