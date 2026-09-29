"""Highlight providers and the parsing of untrusted model output (PRD 8.5)."""

from __future__ import annotations

import json

import pytest

from app.errors import ErrorCode, PipelineError
from app.services.ai_highlights import (
    HeuristicProvider,
    HighlightRequest,
    OpenAICompatibleProvider,
    _candidates_from,
    _parse_response,
    build_user_prompt,
    find_highlights,
)


@pytest.fixture
def request_for(transcript):
    return HighlightRequest(
        title="How I lost forty thousand dollars",
        duration=transcript.duration,
        clip_count=2,
        max_duration=60,
        transcript=transcript,
    )


# ------------------------------------------------------------------ parsing


@pytest.mark.parametrize(
    "payload",
    [
        '{"clips":[{"start":10,"end":40}]}',
        'Here you go:\n```json\n{"clips":[{"start":10,"end":40}]}\n```',
        'Sure! {"clips":[{"start":10,"end":40}]} Let me know if you want more.',
        '{"highlights":[{"start":10,"end":40}]}',
        '[{"start":10,"end":40}]',
    ],
)
def test_parses_the_shapes_models_actually_return(payload):
    parsed, error = _parse_response(payload)
    assert error is None
    [candidate] = _candidates_from(parsed)
    assert (candidate.start, candidate.end) == (10.0, 40.0)


@pytest.mark.parametrize("payload", ["", "   ", "no json here", "{broken"])
def test_reports_unparseable_output(payload):
    parsed, error = _parse_response(payload)
    assert parsed is None and error


def test_normalises_scores_and_timestamps():
    parsed, _ = _parse_response(
        json.dumps(
            {
                "clips": [
                    {"start": "1:02", "end": "1:45", "viral_score": 0.92, "hashtags": "a, b"},
                    {"start": 5, "end": 25, "score": 300},
                ]
            }
        )
    )
    first, second = _candidates_from(parsed)
    assert (first.start, first.end) == (62.0, 105.0)
    assert first.viral_score == 92.0        # 0-1 scale accepted
    assert first.hashtags == ["#a", "#b"]   # bare tags get their hash
    assert second.viral_score == 100.0      # clamped


def test_entries_without_usable_timestamps_are_skipped():
    parsed, _ = _parse_response('{"clips":[{"reason":"no times"},{"start":1,"end":9}]}')
    assert len(_candidates_from(parsed)) == 1


# ---------------------------------------------------------------- prompting


def test_prompt_carries_the_constraints_and_transcript(request_for):
    prompt = build_user_prompt(request_for, wanted=6)
    assert "Find the 6 best" in prompt
    assert "between 15 and 60 seconds" in prompt
    assert "[00:05] Nobody talks about this" in prompt
    assert "How I lost forty thousand dollars" in prompt


# --------------------------------------------------------------- heuristic


def test_heuristic_provider_returns_usable_windows(request_for):
    result = HeuristicProvider().find_highlights(request_for)
    assert result.candidates
    for candidate in result.candidates:
        assert candidate.end > candidate.start
        assert candidate.title and candidate.hook
        assert 0 <= candidate.viral_score <= 100


def test_heuristic_provider_scores_hooks_above_filler(transcript):
    from app.services.transcript import Segment, Transcript, Word

    def build(text: str) -> Transcript:
        words, clock = [], 0.0
        for token in text.split():
            words.append(Word(start=clock, end=clock + 0.3, text=token, probability=0.9))
            clock += 0.35
        return Transcript(
            text=text,
            language="en",
            segments=[Segment(start=0, end=clock, text=text, words=words)],
            duration=clock,
        )

    hooky = "The truth is nobody talks about the biggest mistake that costs 90 percent of people everything they own and it is shocking how many still do it every single day of their lives right now."
    filler = "um so yeah I mean we were kind of just sort of hanging out over there by the thing near the other thing for a while and then we left and that was it and then we came back again"

    def top_score(text: str) -> float:
        request = HighlightRequest(title="t", duration=60, clip_count=1, max_duration=60, transcript=build(text))
        return HeuristicProvider().find_highlights(request).candidates[0].viral_score

    assert top_score(hooky) > top_score(filler)


# ------------------------------------------------------- provider fallback


def test_failing_ai_provider_degrades_to_heuristics(request_for, monkeypatch):
    class Broken:
        name = "openai_compatible"
        model = "ox-alpha"

        def find_highlights(self, request):
            raise PipelineError(ErrorCode.AI_ANALYSIS_FAILED, "endpoint down")

    monkeypatch.setattr("app.services.ai_highlights.get_provider", lambda: Broken())
    result = find_highlights(request_for)
    assert result.provider == "heuristic (fallback)"
    assert result.candidates


def test_openai_provider_repairs_invalid_json(request_for, monkeypatch):
    """A malformed first reply triggers one repair pass, not a failed job."""
    replies = iter(["not json at all", '{"clips":[{"start":6,"end":26,"viral_score":88}]}'])
    sent: list[list[dict]] = []

    provider = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    provider.model = "ox-alpha"
    provider._complete = lambda messages: (sent.append(messages), next(replies))[1]

    result = provider.find_highlights(request_for)
    assert len(sent) == 2, "expected exactly one repair attempt"
    assert "could not be parsed" in sent[1][-1]["content"]
    assert result.candidates[0].viral_score == 88


def test_openai_provider_raises_when_repair_also_fails(request_for):
    provider = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    provider.model = "ox-alpha"
    provider._complete = lambda messages: "still not json"

    with pytest.raises(PipelineError) as excinfo:
        provider.find_highlights(request_for)
    assert excinfo.value.code == ErrorCode.AI_INVALID_OUTPUT


# ------------------------------------------- live client against a stub API


def test_real_client_round_trip_against_an_openai_compatible_server(
    request_for, stub_openai_server
):
    provider = OpenAICompatibleProvider(
        api_key="sk-stub-key", base_url=stub_openai_server["base_url"], model="stub-model"
    )
    result = provider.find_highlights(request_for)

    [candidate] = result.candidates
    assert (candidate.start, candidate.end) == (6.0, 30.0)
    assert candidate.viral_score == 91
    assert candidate.hashtags == ["#podcast", "#mindset"]

    [call] = stub_openai_server["received"]
    assert call["path"].endswith("/chat/completions")
    assert call["auth"] == "Bearer sk-stub-key"
    assert call["body"]["model"] == "stub-model"
    assert call["body"]["response_format"] == {"type": "json_object"}
    # The transcript really is in the prompt the server received.
    assert "Nobody talks about this" in call["body"]["messages"][1]["content"]


def test_connection_check_succeeds_against_a_live_endpoint(stub_openai_server):
    from app.services.ai_highlights import check_connection

    ok, message = check_connection(
        api_key="sk-stub-key", base_url=stub_openai_server["base_url"], model="stub-model"
    )
    assert ok is True
    assert "stub-model" in message


# ------------------------------------------------- long transcripts (sections)


def _long_transcript(sentence_count: int):
    """A transcript far too long to fit one prompt at a small budget."""
    from app.services.transcript import Segment, Transcript, Word

    segments = []
    clock = 0.0
    for index in range(sentence_count):
        text = (
            f"Point number {index} is the one where the whole argument finally "
            f"turns around completely and lands."
        )
        words = []
        for token in text.split():
            words.append(Word(start=clock, end=clock + 0.3, text=token, probability=0.9))
            clock += 0.4
        segments.append(
            Segment(start=words[0].start, end=words[-1].end, text=text, words=words, confidence=0.9)
        )
        clock += 1.0
    return Transcript(
        text=" ".join(segment.text for segment in segments),
        language="en",
        segments=segments,
        duration=clock,
    )


def test_chunking_respects_the_budget_and_overlaps_the_seam():
    from app.services.ai_highlights import _chunk_sentences
    from app.services.transcript import render_prompt_lines

    sentences = _long_transcript(60).sentences()
    chunks = _chunk_sentences(sentences, budget=1200)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk, "no empty sections"
        # One sentence may exceed the budget on its own; a run of them may not.
        assert len(render_prompt_lines(chunk)) <= 1200 + len(chunk[-1].text) + 12
    # Consecutive sections share a seam, so a moment lying across a boundary
    # is still visible whole to at least one call.
    first_texts = {sentence.text for sentence in chunks[0]}
    assert first_texts & {sentence.text for sentence in chunks[1]}
    # Every sentence is covered by some section.
    covered = {sentence.text for chunk in chunks for sentence in chunk}
    assert covered == {sentence.text for sentence in sentences}


def test_a_transcript_too_long_for_one_prompt_is_analysed_end_to_end(monkeypatch):
    """The tail of a long source must still be able to produce clips.

    Truncating at the prompt limit silently made everything after roughly the
    first hour of a long video unclippable -- no error, just no candidates
    from there ever.
    """
    import re

    from app.config import settings
    from app.services.ai_highlights import HighlightRequest, OpenAICompatibleProvider

    monkeypatch.setattr(settings, "ai_max_prompt_chars", 1500)
    transcript = _long_transcript(80)
    request = HighlightRequest(
        title="A very long conversation",
        duration=transcript.duration,
        clip_count=3,
        max_duration=60,
        transcript=transcript,
    )

    prompts: list[str] = []

    def fake_complete(messages):
        prompt = messages[-1]["content"]
        prompts.append(prompt)
        # Answer inside whatever window this section was given, so the
        # timestamps returned are the section's own.
        stamps = re.findall(r"\[(\d+):(\d\d)\]", prompt)
        first = int(stamps[0][0]) * 60 + int(stamps[0][1])
        return json.dumps({"clips": [{"start": first, "end": first + 30, "viral_score": 80}]})

    provider = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    provider.model = "stub"
    provider._complete = fake_complete

    result = provider.find_highlights(request)

    assert len(prompts) > 1, "a long transcript should be analysed in sections"
    assert any("one section of a longer source" in prompt for prompt in prompts)
    # The decisive assertion: a candidate from the back half of the source.
    latest = max(candidate.start for candidate in result.candidates)
    assert latest > transcript.duration / 2


def test_one_failing_section_costs_only_its_own_clips(monkeypatch):
    from app.config import settings
    from app.services.ai_highlights import HighlightRequest, OpenAICompatibleProvider

    monkeypatch.setattr(settings, "ai_max_prompt_chars", 1500)
    monkeypatch.setattr(settings, "ai_repair_attempts", 0)
    transcript = _long_transcript(80)
    request = HighlightRequest(
        title="A very long conversation",
        duration=transcript.duration,
        clip_count=3,
        max_duration=60,
        transcript=transcript,
    )

    calls = {"n": 0}

    def flaky_complete(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("that section's request timed out")
        return json.dumps(
            {"clips": [{"start": calls["n"] * 40, "end": calls["n"] * 40 + 25, "viral_score": 70}]}
        )

    provider = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    provider.model = "stub"
    provider._complete = flaky_complete

    result = provider.find_highlights(request)

    assert calls["n"] > 1
    assert result.candidates, "the surviving sections should still produce clips"


def test_a_short_transcript_still_makes_exactly_one_call(request_for):
    """Sectioning must not add a round trip to the ordinary case."""
    from app.services.ai_highlights import OpenAICompatibleProvider

    prompts: list[str] = []

    provider = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    provider.model = "stub"
    provider._complete = lambda messages: (
        prompts.append(messages[-1]["content"]),
        json.dumps({"clips": [{"start": 6, "end": 30, "viral_score": 88}]}),
    )[1]

    provider.find_highlights(request_for)

    assert len(prompts) == 1
    assert "one section of a longer source" not in prompts[0]
