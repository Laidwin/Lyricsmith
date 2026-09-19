"""Tests of the text post processing chain applied to a raw transcription."""

from __future__ import annotations

import pytest

from lyricsmith.config import (
    CleaningConfig,
    DeduplicationConfig,
    LineSplittingConfig,
    QualityConfig,
)
from lyricsmith.domain import RawTranscription, Segment
from lyricsmith.refine import refine_transcription
from lyricsmith.utils.dedup import remove_repetition_loops
from lyricsmith.utils.quality import assess_transcription_quality, quality_report


@pytest.fixture()
def steps() -> dict:
    """Default step settings, so each test overrides only what it exercises."""
    return {
        "cleaning": CleaningConfig(),
        "deduplication": DeduplicationConfig(),
        "line_splitting": LineSplittingConfig(),
        "quality": QualityConfig(),
    }


def _raw(*texts: str) -> RawTranscription:
    segments = [Segment(float(i), float(i) + 1, text) for i, text in enumerate(texts)]
    return RawTranscription(
        lyrics="\n".join(texts), segments=segments, language="unknown"
    )


def test_refine_applies_the_whole_chain(steps) -> None:
    """Cleaning, deduplication and quality scoring all run on the raw output."""
    raw = _raw("hello world", "hello world", "hello world", "hello world")

    result = refine_transcription(raw, **steps)

    assert result.duplicates_removed == 2
    assert len(result.segments) == 2
    assert result.quality_report["quality_grade"] in {"A", "B", "C", "D"}
    assert result.language == "unknown"


def test_refine_cleans_artifacts(steps) -> None:
    """Artifacts are removed and recorded in the reports."""
    raw = _raw("oh iiii yeah", "another good line")

    result = refine_transcription(raw, **steps)

    assert "iiii" not in result.lyrics
    assert any(r.artifacts_found for r in result.artifact_reports)


def test_disabled_steps_are_skipped(steps) -> None:
    """A disabled step leaves the segments untouched."""
    steps["cleaning"] = CleaningConfig(enabled=False)
    steps["deduplication"] = DeduplicationConfig(enabled=False)
    raw = _raw("same line", "same line", "same line", "same line")

    result = refine_transcription(raw, **steps)

    assert len(result.segments) == 4
    assert result.duplicates_removed == 0


def test_refine_keeps_raw_lyrics_without_segments(steps) -> None:
    """Without segments, the raw lyrics are preserved."""
    raw = RawTranscription(lyrics="just a short line", segments=[])

    result = refine_transcription(raw, **steps)

    assert result.lyrics == "just a short line"
    assert result.segments == []


def test_degraded_segments_are_flagged(steps) -> None:
    """Low confidence segments are listed and carry their reason."""
    raw = RawTranscription(
        lyrics="x",
        segments=[Segment(0.0, 2.0, "some words here", confidence=0.1)],
    )

    result = refine_transcription(raw, **steps)

    assert result.degraded_segments == [0]
    assert result.segments[0].degraded_reason == "low_confidence"


def test_repetition_loop_detection() -> None:
    """Seven repetitions of the same sentence keep only two."""
    segments = [
        Segment(start=float(i), end=float(i) + 1, text="give it a thumbs up")
        for i in range(7)
    ]
    cleaned, removed = remove_repetition_loops(
        segments, similarity_threshold=0.85, max_repeats=2
    )
    assert len(cleaned) == 2
    assert removed == 5


def test_repetition_keeps_distinct_segments() -> None:
    """Distinct segments are never removed."""
    segments = [
        Segment(0, 1, "first line"),
        Segment(1, 2, "second line"),
        Segment(2, 3, "third line"),
    ]
    cleaned, removed = remove_repetition_loops(segments)
    assert len(cleaned) == 3
    assert removed == 0


def test_quality_assessment_degraded() -> None:
    """Low confidence segments yield a D grade."""
    segments = [
        Segment(0, 2, "some words here", confidence=0.3),
        Segment(2, 4, "more words here", confidence=0.25),
    ]
    qualities = assess_transcription_quality(segments, low_confidence_threshold=0.4)
    assert all(q.is_degraded for q in qualities)
    assert all(q.reason == "low_confidence" for q in qualities)
    assert quality_report(qualities)["quality_grade"] == "D"


def test_quality_assessment_clean() -> None:
    """Clean and confident segments yield an A or B grade."""
    segments = [
        Segment(0, 3, "they say you will get used to it", confidence=0.95),
        Segment(3, 6, "but it never goes away", confidence=0.92),
        Segment(6, 9, "if i get through this", confidence=0.9),
    ]
    qualities = assess_transcription_quality(segments)
    assert not any(q.is_degraded for q in qualities)
    assert quality_report(qualities)["quality_grade"] in {"A", "B"}
