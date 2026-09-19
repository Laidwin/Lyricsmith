"""Transcription quality control.

Analyses the segments produced by HeartTranscriptor to spot likely artifacts
(low confidence, gibberish, repetitions) and produces a summary report graded
from A to D.

The Whisper ASR pipeline shipped with `transformers` does not return a per
segment confidence score. When `Segment.confidence` is missing, a heuristic
confidence is derived from the text itself, using lexical diversity, the share
of single character tokens and the share of unusual characters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..domain import Segment

# Characters considered normal inside lyrics.
_ALLOWED_PUNCT = set(".,!?'\"()-:;")


@dataclass
class SegmentQuality:
    """Quality assessment of a single transcription segment."""

    segment: Segment
    confidence: float
    is_degraded: bool
    reason: str | None  # "low_confidence", "repetition" or "noise"


def _heuristic_confidence(text: str) -> float:
    """Estimate a confidence between 0 and 1 from the text features alone."""
    t = text.strip()
    words = t.split()
    if not words:
        return 0.0
    normal = sum(1 for c in t if c.isalnum() or c.isspace() or c in _ALLOWED_PUNCT)
    char_score = normal / len(t)
    unique_ratio = len({w.lower() for w in words}) / len(words)
    single_ratio = sum(1 for w in words if len(w) == 1) / len(words)
    score = char_score * 0.4 + unique_ratio * 0.4 + (1.0 - single_ratio) * 0.2
    return max(0.0, min(1.0, score))


def _assess_one(seg: Segment, low_confidence_threshold: float) -> SegmentQuality:
    """Assess one segment and derive its degradation reason, if any."""
    text = seg.text or ""
    confidence = (
        float(seg.confidence) if seg.confidence is not None else _heuristic_confidence(text)
    )

    words = text.split()
    n = len(words)
    unique_ratio = len({w.lower() for w in words}) / n if n else 0.0
    single_ratio = sum(1 for w in words if len(w) == 1) / n if n else 1.0
    weird_ratio = (
        sum(
            1
            for c in text
            if not (c.isalnum() or c.isspace() or c in _ALLOWED_PUNCT)
        )
        / len(text)
        if text
        else 1.0
    )

    reason: str | None = None
    if confidence < low_confidence_threshold:
        reason = "low_confidence"
    elif single_ratio > 0.5 or unique_ratio < 0.3:
        reason = "repetition"
    elif weird_ratio > 0.15:
        reason = "noise"
    elif n < 3 and confidence < 0.6:
        reason = "low_confidence"

    return SegmentQuality(
        segment=seg,
        confidence=round(confidence, 3),
        is_degraded=reason is not None,
        reason=reason,
    )


def assess_transcription_quality(
    segments: list[Segment],
    low_confidence_threshold: float = 0.4,
) -> list[SegmentQuality]:
    """Assess every segment and flag the suspicious ones.

    A segment is marked as degraded when its confidence is below
    ``low_confidence_threshold``, when it is repetitive (many single character
    tokens or few distinct words), when it holds an abnormal share of unusual
    characters, or when it is very short with an average confidence.

    Args:
        segments: Segments to assess.
        low_confidence_threshold: Confidence below which a segment is degraded.

    Returns:
        One `SegmentQuality` per input segment, in the same order.
    """
    return [_assess_one(seg, low_confidence_threshold) for seg in segments]


def _grade(overall_confidence: float, degraded_ratio: float) -> str:
    """Compute the overall grade from confidence and degraded ratio."""
    if overall_confidence >= 0.80 and degraded_ratio < 0.05:
        return "A"
    if overall_confidence >= 0.65 and degraded_ratio < 0.15:
        return "B"
    if overall_confidence >= 0.50 and degraded_ratio < 0.30:
        return "C"
    return "D"


def quality_report(qualities: list[SegmentQuality]) -> dict[str, Any]:
    """Aggregate per segment assessments into a global report.

    Args:
        qualities: Per segment assessments.

    Returns:
        A dictionary with ``overall_confidence`` (duration weighted average),
        ``degraded_segments``, ``degraded_ratio`` and ``quality_grade`` from A to D.
    """
    if not qualities:
        return {
            "overall_confidence": 0.0,
            "degraded_segments": 0,
            "degraded_ratio": 0.0,
            "quality_grade": "D",
        }

    def _duration(q: SegmentQuality) -> float:
        return max(q.segment.end - q.segment.start, 0.0)

    total_duration = sum(_duration(q) for q in qualities)
    if total_duration > 0:
        overall = sum(q.confidence * _duration(q) for q in qualities) / total_duration
    else:
        overall = sum(q.confidence for q in qualities) / len(qualities)

    degraded = sum(1 for q in qualities if q.is_degraded)
    ratio = degraded / len(qualities)

    return {
        "overall_confidence": round(overall, 3),
        "degraded_segments": degraded,
        "degraded_ratio": round(ratio, 3),
        "quality_grade": _grade(overall, ratio),
    }
