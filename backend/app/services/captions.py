"""Caption generation (PRD 8.8).

Word-level timings become an ASS subtitle file, which FFmpeg burns in with
libass. Everything is computed from real word times -- no style invents timing.

Styles:
  classic   - a static phrase per line
  highlight - the phrase stays up, the spoken word changes colour
  dynamic   - highlight plus a scale pop on the spoken word
  chunks    - TikTok-style 3-5 word bursts
  karaoke   - a left-to-right \\k sweep across the phrase
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ..config import settings
from .transcript import Transcript, Word

log = logging.getLogger("clipz.captions")

CAPTION_STYLES = ("classic", "highlight", "dynamic", "chunks", "karaoke")

# ASS colours are &HAABBGGRR (alpha, blue, green, red) -- not RGB.
WHITE = "&H00FFFFFF"
BLACK = "&H00000000"
ACCENT = "&H0000E5FF"   # amber
ACCENT_ALT = "&H00B0FF00"  # spring green
SHADOW = "&H96000000"


@dataclass
class CaptionLine:
    """One on-screen phrase, with the words that make it up."""

    words: list[Word]

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end

    @property
    def text(self) -> str:
        return " ".join(w.text.strip() for w in self.words if w.text.strip())


def resolve_style(name: str | None) -> str:
    candidate = (name or settings.default_caption_style).lower()
    if candidate in CAPTION_STYLES:
        return candidate
    if candidate in ("dynamic pop", "dynamic_pop", "pop"):
        return "dynamic"
    if candidate in ("tiktok", "tiktok_chunks"):
        return "chunks"
    if candidate in ("highlight_word", "highlight word"):
        return "highlight"
    log.warning("Unknown caption style %r; using %s", name, settings.default_caption_style)
    return settings.default_caption_style


def build_lines(words: list[Word], *, style: str) -> list[CaptionLine]:
    """Group words into on-screen phrases.

    A phrase breaks on a character budget, a word budget, sentence-final
    punctuation, or a pause -- whichever comes first -- so captions track the
    rhythm of the speech instead of a fixed cadence.
    """
    max_words = 3 if style == "chunks" else settings.caption_words_per_chunk
    max_chars = 18 if style == "chunks" else settings.caption_max_chars
    PAUSE = 0.6

    lines: list[CaptionLine] = []
    current: list[Word] = []
    length = 0

    for index, word in enumerate(words):
        text = word.text.strip()
        if not text:
            continue
        gap = word.start - current[-1].end if current else 0.0
        would_be = length + len(text) + (1 if current else 0)

        if current and (
            len(current) >= max_words
            or would_be > max_chars
            or gap > PAUSE
        ):
            lines.append(CaptionLine(current))
            current, length = [], 0
            would_be = len(text)

        current.append(word)
        length = would_be

        # Break after sentence-final punctuation unless it is the last word.
        if text.endswith((".", "!", "?")) and index != len(words) - 1:
            lines.append(CaptionLine(current))
            current, length = [], 0

    if current:
        lines.append(CaptionLine(current))
    return [line for line in lines if line.text]


# A phrase is held until the next one starts, so captions do not flicker off
# during the pause between phrases. Past this much silence the caption clears
# instead, so a stale line never sits over an unrelated shot.
MAX_HOLD = 2.0


def display_ends(lines: list[CaptionLine], *, limit: float | None = None) -> list[float]:
    """End time to draw each phrase until, avoiding gaps at phrase changes."""
    ends: list[float] = []
    for index, line in enumerate(lines):
        end = line.end + MAX_HOLD
        if index + 1 < len(lines):
            end = min(end, lines[index + 1].start)
        if limit is not None:
            end = min(end, limit)
        ends.append(max(end, line.end))
    return ends


def _escape(text: str) -> str:
    """Neutralise ASS override syntax in spoken text."""
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")").replace("\n", " ")


def _ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    centiseconds = int(round((secs - int(secs)) * 100))
    secs = int(secs)
    if centiseconds == 100:  # rounding carry
        centiseconds = 0
        secs += 1
    return f"{int(hours)}:{int(minutes):02d}:{secs:02d}.{centiseconds:02d}"


def _header(style: str) -> str:
    font = settings.caption_font
    size = settings.caption_font_size
    if style == "chunks":
        size = int(size * 1.15)

    # Bottom-centre, lifted clear of platform UI in the lower ~20% of frame.
    margin_v = int(settings.output_height * 0.24)
    outline = 5 if style in ("dynamic", "chunks") else 4

    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {settings.output_width}
PlayResY: {settings.output_height}
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{size},{WHITE},{ACCENT},{BLACK},{SHADOW},-1,0,0,0,100,100,0,0,1,{outline},2,2,80,80,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _dialogue(start: float, end: float, text: str) -> str:
    return f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{text}\n"


def _events(lines: list[CaptionLine], style: str, limit: float | None = None) -> str:
    ends = display_ends(lines, limit=limit)
    if style == "classic":
        return _events_classic(lines, ends)
    if style == "chunks":
        return _events_chunks(lines, ends)
    if style == "karaoke":
        return _events_karaoke(lines, ends)
    return _events_word_highlight(lines, ends, pop=(style == "dynamic"))


def _events_classic(lines: list[CaptionLine], ends: list[float]) -> str:
    out = ""
    for line, end in zip(lines, ends):
        out += _dialogue(line.start, end, _escape(line.text))
    return out


def _events_chunks(lines: list[CaptionLine], ends: list[float]) -> str:
    """Whole chunk at once, with a quick scale-in so it lands with the voice."""
    out = ""
    for line, end in zip(lines, ends):
        effect = r"{\fad(60,60)\t(0,90,\fscx108\fscy108)\t(90,170,\fscx100\fscy100)}"
        out += _dialogue(line.start, end, effect + _escape(line.text.upper()))
    return out


def _events_word_highlight(lines: list[CaptionLine], ends: list[float], *, pop: bool) -> str:
    """One event per word: the phrase persists, the active word is accented."""
    out = ""
    for line, line_end in zip(lines, ends):
        words = line.words
        for index, word in enumerate(words):
            start = word.start
            # Hold the last word until the phrase gives way to the next one.
            end = words[index + 1].start if index + 1 < len(words) else line_end
            if end <= start:
                end = start + 0.08

            parts: list[str] = []
            for position, other in enumerate(words):
                text = _escape(other.text.strip())
                if not text:
                    continue
                if position == index:
                    tags = rf"{{\c{ACCENT}"
                    if pop:
                        tags += r"\fscx100\fscy100\t(0,110,\fscx116\fscy116)\t(110,200,\fscx100\fscy100)"
                    tags += "}"
                    parts.append(f"{tags}{text}{{\\c{WHITE}}}")
                else:
                    parts.append(text)
            out += _dialogue(start, end, " ".join(parts))
    return out


def _events_karaoke(lines: list[CaptionLine], ends: list[float]) -> str:
    r"""A single event per phrase using \kf sweep timings."""
    out = ""
    for line, line_end in zip(lines, ends):
        parts: list[str] = []
        cursor = line.start
        for word in line.words:
            text = _escape(word.text.strip())
            if not text:
                continue
            # Absorb any gap before the word into its own sweep duration.
            centiseconds = max(1, int(round((word.end - cursor) * 100)))
            parts.append(rf"{{\kf{centiseconds}}}{text}")
            cursor = word.end
        prefix = rf"{{\c{WHITE}\2c{ACCENT_ALT}}}"
        out += _dialogue(line.start, line_end, prefix + " ".join(parts))
    return out


def build_ass(
    transcript: Transcript,
    *,
    start: float,
    end: float,
    style: str,
    destination: Path,
) -> Path | None:
    """Write an ASS file for [start, end), timed relative to the clip.

    Returns None when the window has no usable word timings -- the renderer
    then produces an uncaptioned clip rather than failing the job.
    """
    style = resolve_style(style)
    words = [
        Word(
            start=max(0.0, word.start - start),
            end=max(0.0, word.end - start),
            text=word.text,
            probability=word.probability,
        )
        for word in transcript.words_between(start, end)
        if word.text.strip()
    ]
    if not words:
        log.warning("No word timings for %.1f-%.1f; clip will be uncaptioned", start, end)
        return None

    span = end - start
    for word in words:  # never draw past the end of the clip
        word.end = min(word.end, span)
    words = [w for w in words if w.end > w.start]
    if not words:
        return None

    lines = build_lines(words, style=style)
    content = _header(style) + _events(lines, style, limit=span)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
    log.info("Wrote %s captions (%d lines) to %s", style, len(lines), destination.name)
    return destination
