"""Rendering of the pipeline results: annotated lyrics and JSON reports.

Presentation lives here rather than in the modules that produce the data, so
that changing the layout of a report never touches the transcription or the
separation code.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from .domain import Segment, TranscriptionResult

_DEGRADED_REASON_LABELS = {
    "low_confidence": "low confidence",
    "repetition": "repetition",
    "noise": "noise or artifact",
}

_ARTIFACT_SAMPLE_SIZE = 5


def format_duration(seconds: float) -> str:
    """Format a number of seconds as ``m:ss``."""
    total = int(max(seconds, 0.0))
    return f"{total // 60}:{total % 60:02d}"


def _format_timestamp(seconds: float) -> str:
    """Format a number of seconds as ``[mm:ss]``."""
    total = int(max(seconds, 0.0))
    return f"[{total // 60:02d}:{total % 60:02d}]"


def _reason_for(seg: Segment) -> str:
    """Return the human readable label of a segment degradation reason."""
    return _DEGRADED_REASON_LABELS.get(seg.degraded_reason or "", "low confidence")


def describe_preprocess(pre: Any) -> str:
    """Describe the applied audio pre processing in one readable line."""
    if pre is None:
        return "disabled"
    parts = []
    if pre.loudness_normalized:
        parts.append(f"normalized ({pre.target_lufs:g} LUFS)")
    if pre.dc_removed:
        parts.append("DC offset removed")
    if pre.resampled:
        parts.append(f"resampled from {pre.original_sr} Hz")
    return ", ".join(parts) if parts else "no transformation"


def describe_postprocess(post: Any, aggressiveness: float) -> str:
    """Describe the applied vocal post processing in one readable line."""
    if post is None:
        return "disabled"
    parts = []
    if post.reverb_reduced:
        parts.append(f"reverb reduced (aggressiveness {aggressiveness:g})")
    if post.clarity_enhanced:
        parts.append("clarity enhanced")
    return ", ".join(parts) if parts else "no transformation"


def _artifact_summary(result: TranscriptionResult) -> tuple[int, int, list[str]]:
    """Summarize the cleaning reports.

    Args:
        result: Transcription result holding the artifact reports.

    Returns:
        A tuple ``(artifact_count, emptied_segment_count, artifact_sample)``,
        where the sample holds at most five distinct removed artifacts.
    """
    total = sum(len(r.artifacts_found) for r in result.artifact_reports)
    emptied = sum(1 for r in result.artifact_reports if r.was_emptied)
    sample: list[str] = []
    for report in result.artifact_reports:
        for artifact in report.artifacts_found:
            artifact = artifact.strip()
            if artifact and artifact not in sample:
                sample.append(artifact)
            if len(sample) >= _ARTIFACT_SAMPLE_SIZE:
                return total, emptied, sample
    return total, emptied, sample


def format_annotated_lyrics(
    result: TranscriptionResult, pipeline_info: dict[str, Any] | None = None
) -> str:
    """Build the annotated lyrics text followed by a detailed report.

    Args:
        result: Refined transcription result.
        pipeline_info: Optional metadata (separation, pre and post processing,
            duration, elapsed time) only known at pipeline level.

    Returns:
        The full annotated document, ending with a trailing newline.
    """
    degraded = set(result.degraded_segments)
    lines: list[str] = []
    for i, seg in enumerate(result.segments):
        timestamp = _format_timestamp(seg.start)
        if i in degraded:
            confidence = (
                f", confidence {seg.confidence:.2f}" if seg.confidence is not None else ""
            )
            lines.append(f"{timestamp} [degraded segment: {_reason_for(seg)}{confidence}]")
        else:
            lines.append(f"{timestamp} {seg.text}")

    report = result.quality_report or {}
    total = len(result.segments)
    degraded_n = report.get("degraded_segments", len(degraded))
    ratio_pct = (degraded_n / total * 100) if total else 0.0
    artifacts_n, emptied_n, sample = _artifact_summary(result)

    info = pipeline_info or {}
    title = info.get("input_name")
    footer: list[str] = ["", "---", f"Pipeline report{': ' + title if title else ''}"]

    if info.get("separation"):
        footer.append(f"  Vocal separation   : {info['separation']}")
    if info.get("preprocessing"):
        footer.append(f"  Pre processing     : {info['preprocessing']}")
    if info.get("postprocessing"):
        footer.append(f"  Post processing    : {info['postprocessing']}")
    footer.append(f"  Detected language  : {result.language}")
    if info.get("duration"):
        footer.append(f"  Audio duration     : {info['duration']}")
    if info.get("processing_time"):
        footer.append(f"  Processing time    : {info['processing_time']}")

    footer += [
        "",
        "Quality",
        f"  Grade              : {report.get('quality_grade', '?')}",
        f"  Overall confidence : {report.get('overall_confidence', 0.0):.2f}",
        f"  Degraded segments  : {degraded_n} / {total} ({ratio_pct:.1f}%)",
        "",
        "Cleaning",
        f"  Artifacts removed  : {artifacts_n}" + (f"  ({', '.join(sample)})" if sample else ""),
        f"  Segments emptied   : {emptied_n}",
        "",
        "Deduplication",
        f"  Consecutive repetitions removed : {result.duplicates_removed}",
        f"  Ghost blocks removed            : {result.ghost_blocks_removed}",
    ]
    return "\n".join(lines + footer) + "\n"


def build_metadata(
    input_path: Path,
    transcription: TranscriptionResult,
    *,
    duration_seconds: float,
    separation_used: bool,
    separation_time_seconds: float,
    total_time_seconds: float,
    stems_kept: bool,
) -> dict[str, Any]:
    """Build the flat metadata document written next to the lyrics."""
    return {
        "input_file": str(input_path),
        "duration_seconds": duration_seconds,
        "language": transcription.language,
        "separation_used": separation_used,
        "separation_time_seconds": round(separation_time_seconds, 2),
        "transcription_time_seconds": round(transcription.processing_time_seconds, 2),
        "total_time_seconds": round(total_time_seconds, 2),
        "num_segments": len(transcription.segments),
        "duplicates_removed": transcription.duplicates_removed,
        "ghost_blocks_removed": transcription.ghost_blocks_removed,
        "quality_grade": transcription.quality_report.get("quality_grade", "?"),
        "overall_confidence": transcription.quality_report.get("overall_confidence", 0.0),
        "degraded_segments": len(transcription.degraded_segments),
        "artifacts_removed": sum(
            len(r.artifacts_found) for r in transcription.artifact_reports
        ),
        "stems_kept": stems_kept,
    }


def build_pipeline_info(
    input_path: Path,
    separation: str,
    preprocess: Any,
    postprocess: Any,
    *,
    reverb_aggressiveness: float,
    duration_seconds: float,
    total_time_seconds: float,
) -> dict[str, Any]:
    """Build the textual metadata used by the annotated lyrics footer."""
    return {
        "input_name": input_path.name,
        "separation": separation,
        "preprocessing": describe_preprocess(preprocess),
        "postprocessing": describe_postprocess(postprocess, reverb_aggressiveness),
        "duration": format_duration(duration_seconds),
        "processing_time": f"{total_time_seconds:.0f}s",
    }


def build_pipeline_report(
    input_path: Path,
    transcription: TranscriptionResult,
    metadata: dict[str, Any],
    *,
    separation: dict[str, Any],
    preprocess: Any,
    postprocess: Any,
) -> dict[str, Any]:
    """Build the structured JSON pipeline report."""
    return {
        "input_file": str(input_path),
        "metadata": metadata,
        "separation": separation,
        "preprocessing": asdict(preprocess) if preprocess is not None else None,
        "postprocessing": asdict(postprocess) if postprocess is not None else None,
        "quality": transcription.quality_report,
        "cleaning": {
            "artifacts_removed": sum(
                len(r.artifacts_found) for r in transcription.artifact_reports
            ),
            "segments_emptied": sum(
                1 for r in transcription.artifact_reports if r.was_emptied
            ),
            "reports": [asdict(r) for r in transcription.artifact_reports],
        },
        "deduplication": {
            "consecutive_removed": transcription.duplicates_removed,
            "ghost_blocks_removed": transcription.ghost_blocks_removed,
        },
    }
