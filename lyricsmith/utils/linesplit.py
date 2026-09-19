"""Reflow of transcribed lyrics into readable verses.

HeartTranscriptor often returns the whole text on a single line: the model does
not emit usable segment timestamps (a single ``0.0/0.0`` segment) and word level
timestamps, when present, are contiguous with no detectable pause. Rhythm is
therefore not a usable signal.

This module splits an overly long line using three textual signals, by order of
priority:

1. Capital letters, since the model usually capitalizes the first word of a verse.
2. Function words, used in fully lowercase passages where the first signal is
   silent: the line is cut before a common conjunction or pronoun such as "and",
   "but", "et" or "si", once it is long enough, rather than mid sentence.
3. Strong punctuation, cutting after ``.``, ``!`` or ``?``.

A length guard prevents endless verses. Commas are deliberately ignored because
in lyrics they usually sit inside a verse rather than end one.
"""

from __future__ import annotations

# Capitalized words that almost never open a verse: the English pronoun "I" and
# its contractions, always written uppercase in the middle of a sentence.
DEFAULT_STOPWORDS = frozenset({"I", "I'll", "I'm", "I've", "I'd"})

# English and French conjunctions or pronouns that naturally open a verse. They
# serve as fallback cut points in sections without any capital letter.
DEFAULT_BREAKERS = frozenset(
    {
        # English
        "and", "but", "or", "so", "then", "if", "when", "while", "cause",
        "because", "wish", "oh", "i", "you", "we", "they", "he", "she",
        "my", "your",
        # French
        "et", "mais", "ou", "donc", "si", "quand", "que", "car", "je", "tu",
        "il", "elle", "on", "dans", "pour", "mon", "ton",
    }
)

_CAPITALS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞ")
_STRIP = ".,!?;:\"'()[]"


def _starts_new_line(word: str, stopwords: frozenset[str]) -> bool:
    """Tell whether a capitalized word likely opens a new verse."""
    if not word or word[0] not in _CAPITALS:
        return False
    if word in stopwords:
        return False
    stripped = word.strip(_STRIP)
    if len(stripped) > 1 and stripped.isupper():
        return False  # An all caps word is an acronym, not a verse start.
    return True


def _is_breaker(word: str, breakers: frozenset[str]) -> bool:
    """Tell whether a word is a conjunction or pronoun able to open a verse."""
    return word.strip(_STRIP).lower() in breakers


def split_line(
    words: list[str],
    *,
    max_words: int,
    hard_max_words: int,
    stopwords: frozenset[str],
    breakers: frozenset[str],
) -> list[str]:
    """Split a word sequence into verses.

    Args:
        words: Words of a single source line.
        max_words: Length above which a cut on a function word is allowed.
        hard_max_words: Absolute maximum length before a forced cut.
        stopwords: Capitalized words that must not open a verse.
        breakers: Function words usable as fallback cut points.

    Returns:
        The resulting verses.
    """
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        if current and _starts_new_line(word, stopwords):
            lines.append(" ".join(current))
            current = []
        elif current and len(current) >= max_words and _is_breaker(word, breakers):
            lines.append(" ".join(current))
            current = []
        current.append(word)
        if word[-1:] in ".!?":
            lines.append(" ".join(current))
            current = []
        elif len(current) >= hard_max_words:
            lines.append(" ".join(current))
            current = []
    if current:
        lines.append(" ".join(current))
    return lines


def relineate(
    text: str,
    *,
    max_words: int = 10,
    hard_max_words: int = 16,
    min_words_to_split: int = 12,
    stopwords: frozenset[str] = DEFAULT_STOPWORDS,
    breakers: frozenset[str] = DEFAULT_BREAKERS,
) -> str:
    """Reflow a text, often a single line, into readable verses.

    An existing line is only split when it exceeds ``min_words_to_split`` words,
    so segments that are already short and correct stay untouched.

    Args:
        text: Lyrics to reflow.
        max_words: Length above which a cut on a function word is allowed.
        hard_max_words: Absolute maximum length before a forced cut.
        min_words_to_split: Lines at or below this length are left as they are.
        stopwords: Capitalized words that must not open a verse.
        breakers: Function words usable as fallback cut points.

    Returns:
        The text with one verse per line.
    """
    out: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        words = line.split()
        if len(words) <= min_words_to_split:
            out.append(line)
        else:
            out.extend(
                split_line(
                    words,
                    max_words=max_words,
                    hard_max_words=hard_max_words,
                    stopwords=stopwords,
                    breakers=breakers,
                )
            )
    return "\n".join(out)
