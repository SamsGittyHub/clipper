"""AI highlight detection (PRD 8.5).

The model makes semantic judgements only -- which moments are compelling and
how to describe them. It never sees a file path, never chooses an encoder, and
never controls a media operation. Its answer is untrusted input that the
validation stage checks before anything is rendered.

Two providers ship:

* `openai_compatible` -- any OpenAI-compatible endpoint. Ox Alpha is the
  default; OpenAI, Claude-via-proxy, Gemini-via-proxy, Qwen, or a local
  llama.cpp server work by changing two environment variables.
* `heuristic` -- no network. Scores transcript windows on speech density,
  hook phrasing, question/emotion markers and sentence completeness. It keeps
  the pipeline runnable (and testable) with no key configured, and it is the
  automatic fallback so a model outage degrades rather than fails.
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..config import settings
from ..errors import ErrorCode, PipelineError
from . import runtime_config
from .transcript import Sentence, Transcript, format_timestamp, parse_timestamp, render_prompt_lines

log = logging.getLogger("clipz.ai")

# Kept as the fallback for `settings.ai_max_prompt_chars`, which is what the
# code actually reads.
MAX_PROMPT_CHARS = 120_000


def _prompt_budget() -> int:
    return max(4_000, int(getattr(settings, "ai_max_prompt_chars", MAX_PROMPT_CHARS)))


@dataclass
class HighlightRequest:
    title: str
    duration: float
    clip_count: int
    max_duration: float
    transcript: Transcript


@dataclass
class HighlightCandidate:
    """One AI suggestion, before validation."""

    start: float
    end: float
    viral_score: float = 50.0
    hook_score: float = 50.0
    emotion_score: float = 50.0
    reason: str | None = None
    title: str | None = None
    hook: str | None = None
    description: str | None = None
    hashtags: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class HighlightResult:
    candidates: list[HighlightCandidate]
    provider: str
    model: str
    raw: dict[str, Any] | None = None


class HighlightProvider(Protocol):
    name: str

    def find_highlights(self, request: HighlightRequest) -> HighlightResult: ...


# --------------------------------------------------------------------------
# Prompting
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a short-form video editor who finds the moments in a long video that \
work as standalone TikTok, Reels and Shorts clips.

You judge content only. You never describe edits, crops, transitions or \
encoding -- separate software handles all of that.

Look for: strong hooks in the first seconds, insights, emotional beats, \
humour, surprising facts, strong or contrarian opinions, complete short \
stories, and unexpected conclusions.

Rules:
- A clip must be understandable with no other context.
- Start on the beginning of a thought, end after it resolves. Never cut \
mid-sentence.
- Use only timestamps that exist in the transcript.
- Prefer a strong first line: the opening 3 seconds decide whether a viewer stays.
- Do not overlap your suggestions; each should cover a distinct moment.
- Reply with JSON only, matching the requested schema exactly."""

RESPONSE_SCHEMA_HINT = """\
{
  "clips": [
    {
      "start": 124.2,
      "end": 174.5,
      "viral_score": 92,
      "hook_score": 95,
      "emotion_score": 82,
      "reason": "Why this works as a standalone clip.",
      "title": "Punchy title under 60 characters",
      "hook": "The spoken or written line that stops the scroll",
      "description": "One or two sentences for the post caption.",
      "hashtags": ["#podcast", "#mindset", "#success"]
    }
  ]
}"""


def build_user_prompt(
    request: HighlightRequest,
    *,
    wanted: int,
    body: str | None = None,
    window: tuple[float, float] | None = None,
) -> str:
    """The analysis prompt.

    `body` and `window` are set when the transcript is too long for one call
    and is being analysed in sections; the window is stated explicitly so the
    model returns timestamps on the source's own clock, not the section's.
    """
    if body is None:
        body = request.transcript.to_prompt_text(max_chars=_prompt_budget())
    scope = ""
    if window is not None:
        scope = (
            f"\nYou are reading one section of a longer source, covering "
            f"{format_timestamp(window[0])} to {format_timestamp(window[1])}. "
            f"Timestamps below are already on the full source's clock -- use them as they appear.\n"
        )
    return f"""\
Source title: {request.title}
Source duration: {format_timestamp(request.duration)} ({request.duration:.0f} seconds)
Language: {request.transcript.language}
{scope}
Find the {wanted} best standalone clips.

Constraints:
- Each clip must last between {settings.min_clip_duration:.0f} and {request.max_duration:.0f} seconds.
- {settings.ideal_duration_low:.0f}-{settings.ideal_duration_high:.0f} seconds performs best; aim there unless the moment needs otherwise.
- `start` and `end` are seconds from the beginning of the source, as numbers.
- Scores are integers from 0 to 100.
- Return between 1 and {wanted} clips, best first.

Respond with JSON in exactly this shape:
{RESPONSE_SCHEMA_HINT}

Transcript (each line is prefixed with its start time):
{body}
"""


# --------------------------------------------------------------------------
# OpenAI-compatible provider
# --------------------------------------------------------------------------


class OpenAICompatibleProvider:
    """Any OpenAI-compatible chat-completions endpoint."""

    name = "openai_compatible"

    def __init__(self, *, api_key: str, base_url: str, model: str):
        from openai import OpenAI

        self.model = model
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=settings.ai_timeout_s,
            max_retries=settings.ai_max_retries,
        )

    def find_highlights(self, request: HighlightRequest) -> HighlightResult:
        wanted = max(
            request.clip_count,
            int(request.clip_count * settings.ai_candidate_multiplier),
        )
        sentences = request.transcript.sentences()
        budget = _prompt_budget()
        full_body = render_prompt_lines(sentences)

        if len(full_body) <= budget or not sentences:
            candidates = self._analyse(request, wanted=wanted, body=full_body)
        else:
            candidates = self._analyse_in_sections(request, sentences, wanted=wanted, budget=budget)

        if not candidates:
            raise PipelineError(
                ErrorCode.AI_INVALID_OUTPUT,
                "The highlight model returned no usable clips",
            )
        return HighlightResult(candidates, self.name, self.model)

    def _analyse_in_sections(
        self,
        request: HighlightRequest,
        sentences: list[Sentence],
        *,
        wanted: int,
        budget: int,
    ) -> list[HighlightCandidate]:
        """Analyse a transcript too long for one call, section by section.

        Truncating a three-hour source at the prompt limit threw away
        everything after roughly the first hour -- no clip could ever be
        found there. Each section is its own call, they run concurrently,
        and one failing section costs its own clips rather than the job.
        """
        sections = _chunk_sentences(sentences, budget=budget)
        total_span = sum(max(1e-6, chunk[-1].end - chunk[0].start) for chunk in sections)
        log.info(
            "Transcript needs %d section(s) to analyse (%d sentences, budget %d chars)",
            len(sections), len(sentences), budget,
        )

        def analyse(chunk: list[Sentence]) -> list[HighlightCandidate]:
            window = (chunk[0].start, chunk[-1].end)
            share = max(1e-6, window[1] - window[0]) / total_span
            share_wanted = max(2, min(wanted, round(wanted * share) or 1))
            return self._analyse(
                request,
                wanted=share_wanted,
                body=render_prompt_lines(chunk),
                window=window,
            )

        workers = max(1, min(settings.ai_chunk_concurrency, len(sections)))
        collected: list[HighlightCandidate] = []
        failures = 0
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ai-section") as pool:
            for chunk, result in zip(sections, pool.map(_swallow(analyse), sections)):
                if result is None:
                    failures += 1
                    log.warning(
                        "Section %.0f-%.0fs produced no candidates", chunk[0].start, chunk[-1].end
                    )
                    continue
                collected.extend(result)

        if failures and not collected:
            raise PipelineError(
                ErrorCode.AI_ANALYSIS_FAILED,
                "Every section of the transcript failed to analyse",
                detail=f"{failures} of {len(sections)} sections",
            )
        return _merge_candidates(collected)

    def _analyse(
        self,
        request: HighlightRequest,
        *,
        wanted: int,
        body: str | None = None,
        window: tuple[float, float] | None = None,
    ) -> list[HighlightCandidate]:
        """One model call (plus repair passes), parsed into candidates."""
        prompt = build_user_prompt(request, wanted=wanted, body=body, window=window)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        content = self._complete(messages)
        parsed, error = _parse_response(content)

        # Repair pass: hand the model its own broken output and the error.
        attempts = 0
        while parsed is None and attempts < settings.ai_repair_attempts:
            attempts += 1
            log.warning("AI returned unparseable JSON (%s); repair attempt %d", error, attempts)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": content[:4000]},
                {
                    "role": "user",
                    "content": (
                        f"That response could not be parsed: {error}. "
                        f"Reply again with JSON only, matching this schema exactly:\n"
                        f"{RESPONSE_SCHEMA_HINT}"
                    ),
                },
            ]
            content = self._complete(messages)
            parsed, error = _parse_response(content)

        if parsed is None:
            raise PipelineError(
                ErrorCode.AI_INVALID_OUTPUT,
                "The highlight model did not return valid JSON",
                detail=f"{error}; last response: {content[:500]}",
            )

        candidates = _candidates_from(parsed)
        if not candidates and window is None:
            raise PipelineError(
                ErrorCode.AI_INVALID_OUTPUT,
                "The highlight model returned no usable clips",
                detail=json.dumps(parsed)[:500],
            )
        return candidates

    def _complete(self, messages: list[dict[str, str]]) -> str:
        started = time.time()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=settings.ai_temperature,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            # Not every OpenAI-compatible server implements response_format.
            if _is_unsupported_param(exc):
                log.info("Endpoint rejected response_format; retrying without it")
                try:
                    response = self._client.chat.completions.create(
                        model=self.model,
                        messages=messages,  # type: ignore[arg-type]
                        temperature=settings.ai_temperature,
                    )
                except Exception as inner:
                    raise PipelineError(
                        ErrorCode.AI_ANALYSIS_FAILED,
                        "The highlight model request failed",
                        detail=f"{type(inner).__name__}: {inner}",
                    ) from inner
            else:
                raise PipelineError(
                    ErrorCode.AI_ANALYSIS_FAILED,
                    "The highlight model request failed",
                    detail=f"{type(exc).__name__}: {exc}",
                ) from exc

        log.info("AI analysis call finished in %.1fs", time.time() - started)
        choice = response.choices[0] if response.choices else None
        return (choice.message.content if choice and choice.message else "") or ""


def _is_unsupported_param(exc: Exception) -> bool:
    text = str(exc).lower()
    return "response_format" in text and ("unsupported" in text or "invalid" in text or "not support" in text)


# --------------------------------------------------------------------------
# Heuristic provider (no network)
# --------------------------------------------------------------------------

HOOK_PHRASES = (
    "the truth is", "nobody talks about", "here's the thing", "most people",
    "what if", "the problem is", "i realised", "i realized", "the secret",
    "let me tell you", "the biggest mistake", "you have to", "never", "always",
    "the reason", "actually", "honestly", "believe it or not", "turns out",
)
EMOTION_WORDS = (
    "love", "hate", "afraid", "scared", "amazing", "incredible", "insane",
    "crazy", "terrible", "devastating", "hilarious", "shocking", "unbelievable",
    "furious", "excited", "proud", "ashamed", "heartbreaking", "beautiful",
)


class HeuristicProvider:
    """Deterministic window scoring; the offline fallback."""

    name = "heuristic"
    model = "heuristic-v1"

    def find_highlights(self, request: HighlightRequest) -> HighlightResult:
        sentences = request.transcript.sentences()
        if not sentences:
            raise PipelineError(
                ErrorCode.AI_ANALYSIS_FAILED,
                "The transcript contained no usable sentences to analyse",
            )

        target = max(
            settings.min_clip_duration + 5.0,
            min(request.max_duration, settings.ideal_duration_high),
        )
        wanted = max(request.clip_count, int(request.clip_count * settings.ai_candidate_multiplier))

        scored: list[HighlightCandidate] = []
        for index, sentence in enumerate(sentences):
            window = [sentence]
            end = sentence.end
            for follower in sentences[index + 1 :]:
                if follower.end - sentence.start > target:
                    break
                window.append(follower)
                end = follower.end
            span = end - sentence.start
            if span < settings.min_clip_duration or span > request.max_duration:
                continue

            text = " ".join(s.text for s in window)
            scored.append(self._score(sentence.start, end, text, request.transcript))

        if not scored:
            # The source is shorter than one clip, or one long unbroken take.
            end = min(request.duration, request.max_duration)
            text = request.transcript.text_between(0.0, end)
            scored = [self._score(0.0, end, text, request.transcript)]

        scored.sort(key=lambda candidate: candidate.viral_score, reverse=True)

        # Thin out near-duplicates so the ranker receives distinct moments.
        selected: list[HighlightCandidate] = []
        for candidate in scored:
            if all(_overlap_ratio(candidate, kept) < 0.4 for kept in selected):
                selected.append(candidate)
            if len(selected) >= wanted:
                break

        return HighlightResult(selected, self.name, self.model)

    def _score(
        self, start: float, end: float, text: str, transcript: Transcript
    ) -> HighlightCandidate:
        lowered = text.lower()
        words = len(text.split())
        span = max(1e-6, end - start)

        density = min(1.0, (words / span) / 3.0)  # ~3 words/second is dense speech
        hook = sum(1 for phrase in HOOK_PHRASES if phrase in lowered[:220])
        emotion = sum(1 for word in EMOTION_WORDS if word in lowered)
        question = 1.0 if "?" in text else 0.0
        complete = 1.0 if text.strip().endswith((".", "!", "?")) else 0.4
        numbers = 1.0 if re.search(r"\b\d+([.,]\d+)?\s*(%|percent|x|times|million|billion)\b", lowered) else 0.0

        hook_score = min(100.0, 35.0 + hook * 22.0 + question * 15.0 + numbers * 10.0)
        emotion_score = min(100.0, 30.0 + emotion * 18.0 + question * 8.0)
        viral = min(
            100.0,
            density * 34.0 + hook_score * 0.28 + emotion_score * 0.18 + complete * 14.0 + numbers * 6.0,
        )

        sentence_text = text.strip()
        first_sentence = re.split(r"(?<=[.!?])\s+", sentence_text)[0][:120]
        return HighlightCandidate(
            start=start,
            end=end,
            viral_score=round(viral, 1),
            hook_score=round(hook_score, 1),
            emotion_score=round(emotion_score, 1),
            reason=(
                f"Dense speech ({words / span:.1f} words/s)"
                + (f", {hook} hook phrase(s)" if hook else "")
                + (f", {emotion} emotional cue(s)" if emotion else "")
                + (", opens with a question" if question else "")
                + "."
            ),
            title=_title_from(first_sentence),
            hook=first_sentence,
            description=sentence_text[:280],
            hashtags=[],
        )


def _swallow(function):
    """Run `function`, returning None instead of raising.

    Used for per-section analysis: one section erroring out must cost only
    its own candidates.
    """

    def wrapped(argument):
        try:
            return function(argument)
        except Exception as exc:  # noqa: BLE001 - deliberately broad; logged
            log.warning("Transcript section analysis failed: %s", exc)
            return None

    return wrapped


def _chunk_sentences(
    sentences: list[Sentence], *, budget: int, overlap_fraction: float = 0.06
) -> list[list[Sentence]]:
    """Group sentences into rendered-length-bounded, slightly overlapping runs.

    The overlap means a moment sitting across a section boundary is still
    visible whole to at least one call.
    """
    overlap_chars = int(budget * overlap_fraction)
    chunks: list[list[Sentence]] = []
    current: list[Sentence] = []
    length = 0

    for sentence in sentences:
        # 12 covers the "[mm:ss] " prefix and the newline.
        line_length = len(sentence.text) + 12
        if current and length + line_length > budget:
            chunks.append(current)
            tail: list[Sentence] = []
            tail_length = 0
            for previous in reversed(current):
                if tail_length >= overlap_chars:
                    break
                tail.insert(0, previous)
                tail_length += len(previous.text) + 12
            current, length = tail, tail_length
        current.append(sentence)
        length += line_length

    if current:
        chunks.append(current)
    return chunks


def _merge_candidates(
    candidates: list[HighlightCandidate], *, overlap_limit: float = 0.6
) -> list[HighlightCandidate]:
    """Best-first merge, dropping near-duplicates from overlapping sections."""
    merged: list[HighlightCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.viral_score, reverse=True):
        if all(_overlap_ratio(candidate, kept) <= overlap_limit for kept in merged):
            merged.append(candidate)
    return merged


def _title_from(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip(" .,-")
    if len(cleaned) > 60:
        cleaned = cleaned[:57].rsplit(" ", 1)[0] + "..."
    return cleaned or "Untitled clip"


def _overlap_ratio(a: HighlightCandidate, b: HighlightCandidate) -> float:
    overlap = min(a.end, b.end) - max(a.start, b.start)
    if overlap <= 0:
        return 0.0
    return overlap / max(1e-6, min(a.duration, b.duration))


# --------------------------------------------------------------------------
# Response parsing
# --------------------------------------------------------------------------


def _parse_response(content: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse a model reply that should be JSON but may be wrapped in prose."""
    if not content or not content.strip():
        return None, "empty response"

    text = content.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        extracted = _extract_json_object(text)
        if extracted is None:
            return None, f"invalid JSON: {exc.msg}"
        data = extracted

    if isinstance(data, list):
        data = {"clips": data}
    if not isinstance(data, dict):
        return None, "top-level JSON value was not an object"
    if "clips" not in data:
        for key in ("highlights", "results", "moments", "segments"):
            if key in data:
                data = {"clips": data[key]}
                break
    if not isinstance(data.get("clips"), list):
        return None, "missing a `clips` array"
    return data, None


def _extract_json_object(text: str) -> Any | None:
    """Recover the outermost balanced {...} from a reply padded with prose."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _candidates_from(parsed: dict[str, Any]) -> list[HighlightCandidate]:
    """Coerce model output into candidates; skip entries that make no sense."""
    candidates: list[HighlightCandidate] = []
    for entry in parsed.get("clips", []):
        if not isinstance(entry, dict):
            continue
        start = parse_timestamp(entry.get("start", entry.get("start_time")))
        end = parse_timestamp(entry.get("end", entry.get("end_time")))
        if start is None or end is None:
            continue
        candidates.append(
            HighlightCandidate(
                start=start,
                end=end,
                viral_score=_score(entry, "viral_score", "score"),
                hook_score=_score(entry, "hook_score", "hook"),
                emotion_score=_score(entry, "emotion_score", "emotion"),
                reason=_text(entry.get("reason")),
                title=_text(entry.get("title")),
                hook=_text(entry.get("hook")),
                description=_text(entry.get("description")),
                hashtags=_hashtags(entry.get("hashtags")),
            )
        )
    return candidates


def _score(entry: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = entry.get(key)
        if isinstance(value, (int, float)):
            # Accept 0-1 as well as 0-100; some models normalise.
            number = float(value)
            return max(0.0, min(100.0, number * 100.0 if number <= 1.0 else number))
        if isinstance(value, str):
            try:
                return max(0.0, min(100.0, float(value)))
            except ValueError:
                continue
    return 50.0


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _hashtags(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        value = re.split(r"[,\s]+", value)
    tags: list[str] = []
    for item in value:
        tag = str(item).strip()
        if not tag:
            continue
        tags.append(tag if tag.startswith("#") else f"#{tag}")
    return tags[:10]


# --------------------------------------------------------------------------
# Provider selection
# --------------------------------------------------------------------------


def get_provider() -> HighlightProvider:
    """Resolve the active provider: runtime settings, then environment.

    A key entered in the settings panel wins over `OX_API_KEY`; with neither,
    the heuristic provider keeps the pipeline working.
    """
    config = runtime_config.current()
    if config.effective_provider == "heuristic":
        if config.provider != "heuristic":
            log.warning(
                "No AI API key configured; using the heuristic highlight provider. "
                "Add a key in the settings panel, or set OX_API_KEY."
            )
        return HeuristicProvider()
    return OpenAICompatibleProvider(
        api_key=config.api_key,
        base_url=config.base_url,
        model=config.model,
    )


def check_connection(
    *, api_key: str, base_url: str, model: str, timeout: float = 30.0
) -> tuple[bool, str]:
    """Send the smallest possible request to verify a key/URL/model combination.

    Returns `(ok, message)`. The key is scrubbed from any error text before it
    is returned, since this message is rendered in the browser.
    """
    from openai import OpenAI

    try:
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with the word: ok"}],
            max_tokens=8,
        )
    except Exception as exc:
        detail = _scrub(f"{type(exc).__name__}: {exc}", api_key)
        return False, detail[:400]

    served = getattr(response, "model", None) or model
    return True, f"Connected to {served}."


def _scrub(text: str, secret: str | None) -> str:
    if secret and len(secret) > 6:
        return text.replace(secret, "***")
    return text


def find_highlights(request: HighlightRequest) -> HighlightResult:
    """Run the configured provider; degrade to heuristics rather than fail."""
    provider = get_provider()
    try:
        return provider.find_highlights(request)
    except PipelineError:
        if provider.name == "heuristic":
            raise
        log.exception("Highlight provider %s failed; falling back to heuristics", provider.name)
        result = HeuristicProvider().find_highlights(request)
        result.provider = "heuristic (fallback)"
        return result
