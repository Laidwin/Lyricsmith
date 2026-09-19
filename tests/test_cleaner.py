"""Tests of the character artifact cleaning."""

from __future__ import annotations

from lyricsmith.domain import ArtifactReport, Segment
from lyricsmith.utils.cleaner import clean_segment_text, clean_transcription


def test_fullwidth_chars_removed() -> None:
    """Fullwidth characters and fullwidth parentheses are removed."""
    cleaned, removed = clean_segment_text("hello （） ＡＢＣ world")
    assert "（" not in cleaned and "）" not in cleaned
    assert "Ａ" not in cleaned and "Ｂ" not in cleaned
    assert "hello" in cleaned and "world" in cleaned
    assert removed


def test_repeated_chars_removed() -> None:
    """Words made of a letter repeated four times or more are removed."""
    cleaned, removed = clean_segment_text("oh iiii yeah")
    assert "iiii" not in cleaned
    assert "oh" in cleaned and "yeah" in cleaned
    assert "iiii" in removed


def test_bom_removed() -> None:
    """Byte order marks and zero width characters are removed."""
    cleaned, removed = clean_segment_text("﻿hello​world")
    assert "﻿" not in cleaned and "​" not in cleaned
    assert "hello" in cleaned and "world" in cleaned
    assert removed


def test_clean_segment_preserved() -> None:
    """A clean text, Latin accents included, is left untouched."""
    text = "Je n'oublierai jamais ça"
    cleaned, removed = clean_segment_text(text)
    assert cleaned == text
    assert removed == []


def test_artifact_report_populated() -> None:
    """clean_transcription fills a report and drops emptied segments."""
    segments = [
        Segment(0.0, 1.0, "clean line here"),
        Segment(1.0, 2.0, "﻿（）"),  # Becomes empty once cleaned.
        Segment(2.0, 3.0, "another good line"),
    ]
    kept, reports = clean_transcription(segments, min_segment_words=1)

    # The emptied segment is excluded but kept in the report.
    assert len(kept) == 2
    assert all(isinstance(r, ArtifactReport) for r in reports)
    assert any(r.was_emptied for r in reports)
    emptied = [r for r in reports if r.was_emptied][0]
    assert emptied.source_segment_index == 1
    assert emptied.timestamp_start == 1.0
