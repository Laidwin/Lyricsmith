"""Post transcription text cleaning, meaning character artifact removal.

HeartTranscriptor, which is based on Whisper, sometimes leaves residues behind:
byte order marks, zero width spaces, fullwidth characters, CJK punctuation,
repeated letters such as "iiii", or stray non ASCII characters at the end of a
segment. This module removes them and records every removal in an
`ArtifactReport` so the cleaning stays auditable.
"""

from __future__ import annotations

import re

from ..domain import ArtifactReport, Segment
from .logger import get_logger

logger = get_logger("utils.cleaner")

# Known artifact patterns; each one deletes the text it matches.
_PATTERN_BOM = r"[﻿​‌‍]"          # BOM and zero width
_PATTERN_FULLWIDTH = r"[＀-￯]"               # Fullwidth range
_PATTERN_CJK_PUNCT = r"[　-〿]"               # CJK punctuation
_PATTERN_FW_PARENS = r"[（）｟｠]"    # Fullwidth parentheses
_PATTERN_REPEATED = r"\b([a-zA-Z])\1{3,}\b"           # A letter repeated four times or more
# Non ASCII outside basic and extended Latin, so accents are preserved.
_PATTERN_NON_LATIN = r"[^\x00-\x7fÀ-ɏḀ-ỿ\s]"

ARTIFACT_PATTERNS = [
    _PATTERN_BOM,
    _PATTERN_FULLWIDTH,
    _PATTERN_CJK_PUNCT,
    _PATTERN_FW_PARENS,
    _PATTERN_REPEATED,
    _PATTERN_NON_LATIN,
]


def clean_segment_text(
    text: str, patterns: list[str] | None = None
) -> tuple[str, list[str]]:
    """Clean the text of a single segment.

    Args:
        text: Raw segment text.
        patterns: Regular expressions to apply. Defaults to `ARTIFACT_PATTERNS`.

    Returns:
        A tuple ``(cleaned_text, removed_artifacts)``.
    """
    active = patterns if patterns is not None else ARTIFACT_PATTERNS
    removed: list[str] = []
    cleaned = text

    for pattern in active:
        def _capture(match: re.Match[str]) -> str:
            removed.append(match.group(0))
            return ""

        cleaned = re.sub(pattern, _capture, cleaned)

    # Collapse the whitespace left behind by the removals.
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned, removed


def _build_patterns(
    *,
    remove_fullwidth: bool,
    remove_repeated_chars: bool,
    custom_patterns: list[str] | None,
) -> list[str]:
    """Build the active pattern list from the configuration flags."""
    patterns = [_PATTERN_BOM, _PATTERN_CJK_PUNCT, _PATTERN_FW_PARENS, _PATTERN_NON_LATIN]
    if remove_fullwidth:
        patterns.insert(1, _PATTERN_FULLWIDTH)
    if remove_repeated_chars:
        patterns.append(_PATTERN_REPEATED)
    if custom_patterns:
        patterns.extend(custom_patterns)
    return patterns


def clean_transcription(
    segments: list[Segment],
    *,
    remove_fullwidth: bool = True,
    remove_repeated_chars: bool = True,
    min_segment_words: int = 1,
    custom_patterns: list[str] | None = None,
) -> tuple[list[Segment], list[ArtifactReport]]:
    """Clean every segment and drop the ones emptied by the cleaning.

    Segments left with fewer than `min_segment_words` words are flagged as
    ``was_emptied`` and excluded from the result, while still being kept in the
    reports for inspection.

    Args:
        segments: Segments to clean in place.
        remove_fullwidth: Whether to strip fullwidth characters.
        remove_repeated_chars: Whether to strip letters repeated four times or more.
        min_segment_words: Minimum word count a cleaned segment must keep.
        custom_patterns: Extra regular expressions read from the configuration.

    Returns:
        A tuple ``(kept_segments, artifact_reports)``.
    """
    patterns = _build_patterns(
        remove_fullwidth=remove_fullwidth,
        remove_repeated_chars=remove_repeated_chars,
        custom_patterns=custom_patterns,
    )

    kept: list[Segment] = []
    reports: list[ArtifactReport] = []
    emptied = 0

    for index, seg in enumerate(segments):
        original = seg.text
        cleaned, removed = clean_segment_text(original, patterns)
        was_emptied = len(cleaned.split()) < min_segment_words

        if removed or was_emptied:
            reports.append(
                ArtifactReport(
                    source_segment_index=index,
                    timestamp_start=float(seg.start),
                    original_text=original,
                    cleaned_text=cleaned,
                    artifacts_found=removed,
                    was_emptied=was_emptied,
                )
            )

        if was_emptied:
            emptied += 1
            continue

        seg.text = cleaned
        kept.append(seg)

    total_artifacts = sum(len(r.artifacts_found) for r in reports)
    if total_artifacts or emptied:
        logger.warning(
            "Cleaning: %d artifact(s) removed, %d segment(s) emptied.",
            total_artifacts,
            emptied,
        )
    return kept, reports
