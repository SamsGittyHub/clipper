"""Clip metadata completion (PRD 8.9).

The highlight model supplies titles and hashtags. It sometimes omits them, and
the user is going to edit them anyway, so anything missing is filled in
deterministically from the clip's own transcript rather than by a second model
call. Every field stays editable.
"""

from __future__ import annotations

import re
from collections import Counter

from .selection import ScoredClip

_STOPWORDS = frozenset(
    """a about after all also am an and any are as at be because been before being but by
    can could did do does doing down during each few for from further had has have having
    he her here hers him his how i if in into is it its just like me more most my no nor
    not now of off on once only or other our out over own same she should so some such than
    that the their them then there these they this those through to too under until up very
    was we were what when where which while who whom why will with you your yeah okay right
    know think going really thing things want got get one two""".split()
)

_GENERIC_TAGS = ("#shorts", "#reels", "#fyp")


def _keywords(text: str, limit: int) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z'-]{2,}", text.lower())
    counts = Counter(word for word in words if word not in _STOPWORDS)
    return [word for word, _ in counts.most_common(limit)]


def _first_sentence(text: str, limit: int = 120) -> str:
    sentence = re.split(r"(?<=[.!?])\s+", text.strip())[0] if text.strip() else ""
    sentence = re.sub(r"\s+", " ", sentence).strip()
    return sentence[:limit].strip()


def title_from(text: str) -> str:
    sentence = _first_sentence(text, 80)
    if not sentence:
        return "Untitled clip"
    sentence = sentence.rstrip(".,;: ")
    if len(sentence) > 60:
        sentence = sentence[:57].rsplit(" ", 1)[0].rstrip(",;: ") + "..."
    return sentence[0].upper() + sentence[1:] if sentence else "Untitled clip"


def hashtags_from(text: str, *, source_title: str | None = None) -> list[str]:
    tags = [f"#{word}" for word in _keywords(text, 4) if len(word) > 3]
    if source_title:
        channel_words = _keywords(source_title, 1)
        tags.extend(f"#{word}" for word in channel_words if f"#{word}" not in tags)
    for generic in _GENERIC_TAGS:
        if len(tags) >= 6:
            break
        if generic not in tags:
            tags.append(generic)
    return tags[:6]


def complete(clip: ScoredClip, *, source_title: str | None = None) -> ScoredClip:
    """Fill any metadata the model left out; never overwrite what it gave."""
    text = clip.transcript_text or ""

    if not clip.title:
        clip.title = title_from(text)
    if not clip.hook:
        clip.hook = _first_sentence(text, 100) or clip.title
    if not clip.description:
        body = re.sub(r"\s+", " ", text).strip()
        clip.description = (body[:277] + "...") if len(body) > 280 else body
    if not clip.hashtags:
        clip.hashtags = hashtags_from(text, source_title=source_title)
    return clip
