"""Core entities of the pipeline.

This module is the innermost layer: it holds the data the whole project is
about and depends on nothing else, neither on the configuration, the ASR
engine, nor any input or output concern. Every other module may import it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Segment:
    """A timestamped chunk of lyrics."""

    start: float
    end: float
    text: str
    confidence: float | None = None
    degraded_reason: str | None = None


@dataclass
class ArtifactReport:
    """Record of what the cleaning removed from one segment.

    ``source_segment_index`` refers to the raw segment list as returned by the
    ASR engine, before cleaning and deduplication dropped any segment. It is
    therefore not an index into the segments of the final `TranscriptionResult`;
    use ``timestamp_start`` to locate the segment in the finished transcript.
    """

    source_segment_index: int
    timestamp_start: float
    original_text: str
    cleaned_text: str
    artifacts_found: list[str]
    was_emptied: bool


@dataclass
class RawTranscription:
    """Unrefined output of the ASR engine, before any text post processing."""

    lyrics: str
    segments: list[Segment] = field(default_factory=list)
    language: str = "unknown"
    processing_time_seconds: float = 0.0


@dataclass
class TranscriptionResult:
    """Refined outcome of a lyrics transcription."""

    lyrics: str
    segments: list[Segment] = field(default_factory=list)
    language: str = "unknown"
    processing_time_seconds: float = 0.0
    duplicates_removed: int = 0
    ghost_blocks_removed: int = 0
    quality_report: dict[str, Any] = field(default_factory=dict)
    degraded_segments: list[int] = field(default_factory=list)
    artifact_reports: list[ArtifactReport] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result, segments included, for JSON export."""
        return {
            "lyrics": self.lyrics,
            "language": self.language,
            "processing_time_seconds": self.processing_time_seconds,
            "duplicates_removed": self.duplicates_removed,
            "ghost_blocks_removed": self.ghost_blocks_removed,
            "quality_report": self.quality_report,
            "degraded_segments": self.degraded_segments,
            "artifact_reports": [asdict(r) for r in self.artifact_reports],
            "segments": [asdict(seg) for seg in self.segments],
        }
