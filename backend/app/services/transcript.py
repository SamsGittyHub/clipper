"""Transcript structures and normalisation (PRD 8.4).

The transcript is the single source of truth downstream: the AI reads a
normalised, timestamp-preserving rendering of it; validation snaps candidate
timestamps to real word boundaries; captions are built from word timings.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

# Whisper emits these when a stretch is non-speech; they carry no meaning for
# the highlight model, and dropping them keeps the prompt clean. Source timing
# is untouched -- only the text of a fully-filler segment is removed.
FILLER_PATTERN = re.compile(
    r"^\W*(\[|\()?\s*(music|applause|laughter|silence|blank_audio|inaudible|"
    r"foreign|sound effects?|noise)\s*(\]|\))?\W*$",
    re.IGNORECASE,
)

SENTENCE_END = re.compile(r"[.!?…]+[\"')\]]*$")


@dataclass
class Word:
    start: float
    end: float
    text: str
    probability: float = 1.0


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    confidence: float = 1.0
    speaker: str | None = None


@dataclass
class Sentence:
    """A normalised, timestamp-preserving unit handed to the AI."""

    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Transcript:
    text: str
    language: str
    segments: list[Segment]
    duration: float
    language_probability: float = 1.0

    # ---------------------------------------------------------------- words
    @property
    def words(self) -> list[Word]:
        out: list[Word] = []
        for segment in self.segments:
            out.extend(segment.words)
        return out

    def words_between(self, start: float, end: float) -> list[Word]:
        """Words whose midpoint falls inside [start, end)."""
        result = []
        for word in self.words:
            midpoint = (word.start + word.end) / 2
            if start <= midpoint < end:
                result.append(word)
        return result

    def text_between(self, start: float, end: float) -> str:
        words = self.words_between(start, end)
        if words:
            return " ".join(w.text.strip() for w in words if w.text.strip()).strip()
        # Fall back to segment text when a model gave us no word timings.
        parts = [
            seg.text.strip()
            for seg in self.segments
            if seg.end > start and seg.start < end and seg.text.strip()
        ]
        return " ".join(parts).strip()

    def confidence_between(self, start: float, end: float) -> float:
        words = self.words_between(start, end)
        if words:
            return sum(w.probability for w in words) / len(words)
        overlapping = [s for s in self.segments if s.end > start and s.start < end]
        if not overlapping:
            return 0.0
        return sum(s.confidence for s in overlapping) / len(overlapping)

    def speech_seconds_between(self, start: float, end: float) -> float:
        """Seconds of actual speech inside the window, from word spans."""
        total = 0.0
        for word in self.words:
            overlap = min(word.end, end) - max(word.start, start)
            if overlap > 0:
                total += overlap
        if total > 0:
            return total
        for segment in self.segments:
            overlap = min(segment.end, end) - max(segment.start, start)
            if overlap > 0:
                total += overlap
        return total

    def nearest_word_boundary(self, time: float, *, side: str, window: float = 2.0) -> float:
        """Snap a timestamp to the nearest silence between words.

        `side="start"` moves to the beginning of the word that follows the gap;
        `side="end"` moves to the end of the word that precedes it. This keeps
        clips from opening or closing mid-syllable.
        """
        words = self.words
        if not words:
            return time
        best = time
        best_distance = math.inf
        for word in words:
            boundary = word.start if side == "start" else word.end
            distance = abs(boundary - time)
            if distance < best_distance and distance <= window:
                best_distance = distance
                best = boundary
        return best

    # ------------------------------------------------------- normalisation
    def sentences(self) -> list[Sentence]:
        """Group words into sentences, preserving source timing.

        Falls back to whole segments when word timings are unavailable.
        """
        sentences: list[Sentence] = []
        for segment in self.segments:
            text = segment.text.strip()
            if not text or FILLER_PATTERN.match(text):
                continue
            if not segment.words:
                sentences.append(Sentence(segment.start, segment.end, text))
                continue

            buffer: list[Word] = []
            for word in segment.words:
                if not word.text.strip():
                    continue
                buffer.append(word)
                if SENTENCE_END.search(word.text.strip()):
                    sentences.append(_sentence_from(buffer))
                    buffer = []
            if buffer:
                sentences.append(_sentence_from(buffer))
        return [s for s in sentences if s.text]

    def to_prompt_text(self, *, max_chars: int | None = None) -> str:
        """Render `[mm:ss] sentence` lines, grouped into paragraphs.

        A gap longer than `PARAGRAPH_GAP` starts a new paragraph, which gives
        the model a cheap topic-boundary signal without a diarisation model.
        """
        rendered = render_prompt_lines(self.sentences())
        if max_chars is not None and len(rendered) > max_chars:
            rendered = _trim_to_chars(rendered, max_chars)
        return rendered

    # ------------------------------------------------------ serialisation
    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "language": self.language,
            "language_probability": self.language_probability,
            "duration": self.duration,
            "segments": [asdict(segment) for segment in self.segments],
        }

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "Transcript":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Transcript":
        segments = [
            Segment(
                start=float(seg["start"]),
                end=float(seg["end"]),
                text=seg.get("text", ""),
                confidence=float(seg.get("confidence", 1.0)),
                speaker=seg.get("speaker"),
                words=[
                    Word(
                        start=float(w["start"]),
                        end=float(w["end"]),
                        text=w.get("text", ""),
                        probability=float(w.get("probability", 1.0)),
                    )
                    for w in seg.get("words", [])
                ],
            )
            for seg in raw.get("segments", [])
        ]
        return cls(
            text=raw.get("text", ""),
            language=raw.get("language", "en"),
            language_probability=float(raw.get("language_probability", 1.0)),
            duration=float(raw.get("duration", 0.0)),
            segments=segments,
        )


# A gap longer than this starts a new paragraph in the rendered prompt, which
# gives the model a cheap topic-boundary signal without a diarisation model.
PARAGRAPH_GAP = 2.5


def render_prompt_lines(sentences: list[Sentence]) -> str:
    """`[mm:ss] sentence` lines for a run of sentences, paragraphed on pauses."""
    lines: list[str] = []
    previous_end: float | None = None
    for sentence in sentences:
        if previous_end is not None and sentence.start - previous_end > PARAGRAPH_GAP:
            lines.append("")
        lines.append(f"[{format_timestamp(sentence.start)}] {sentence.text}")
        previous_end = sentence.end
    return "\n".join(lines).strip()


def _sentence_from(words: Iterable[Word]) -> Sentence:
    words = list(words)
    text = " ".join(w.text.strip() for w in words).strip()
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return Sentence(start=words[0].start, end=words[-1].end, text=text)


def _trim_to_chars(text: str, max_chars: int) -> str:
    """Keep whole lines only, so every retained timestamp stays parseable."""
    kept: list[str] = []
    used = 0
    for line in text.split("\n"):
        if used + len(line) + 1 > max_chars:
            break
        kept.append(line)
        used += len(line) + 1
    kept.append("\n[transcript truncated]")
    return "\n".join(kept)


def format_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def parse_timestamp(value: Any) -> float | None:
    """Accept seconds, or `mm:ss` / `hh:mm:ss` strings the model may return."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    if ":" in text:
        parts = text.split(":")
        try:
            numbers = [float(p) for p in parts]
        except ValueError:
            return None
        seconds = 0.0
        for number in numbers:
            seconds = seconds * 60 + number
        return seconds
    try:
        return float(text)
    except ValueError:
        return None
