"""Text post processing applied to a raw transcription.

This is the pipeline policy that turns the raw ASR output into the final
lyrics, in a strict order: cleaning, then deduplication, then quality scoring,
then verse reflow. It is deliberately separate from the ASR adapter so the
order of the steps can be read, changed and tested without a model.
"""

from __future__ import annotations

from .config import (
    CleaningConfig,
    DeduplicationConfig,
    LineSplittingConfig,
    QualityConfig,
)
from .domain import ArtifactReport, RawTranscription, Segment, TranscriptionResult
from .utils.cleaner import clean_transcription
from .utils.dedup import detect_ghost_blocks, remove_repetition_loops
from .utils.linesplit import relineate
from .utils.logger import get_logger
from .utils.quality import assess_transcription_quality, quality_report

logger = get_logger("refine")


def _deduplicate(
    segments: list[Segment], config: DeduplicationConfig
) -> tuple[list[Segment], int, int]:
    """Remove consecutive repetitions, then non consecutive ghost blocks.

    Args:
        segments: Segments to filter.
        config: Deduplication settings.

    Returns:
        A tuple ``(kept_segments, consecutive_removed, ghost_segments_removed)``.
    """
    segments, duplicates_removed = remove_repetition_loops(
        segments,
        similarity_threshold=config.consecutive_similarity,
        max_repeats=config.consecutive_max_repeats,
    )
    blocks = detect_ghost_blocks(
        segments,
        block_min_size=config.ghost_block_min_size,
        similarity_threshold=config.ghost_block_similarity,
        time_window_seconds=config.ghost_block_time_window,
    )
    ghost_indices = {idx for block in blocks for idx in block.ghost_indices}
    if ghost_indices:
        segments = [s for i, s in enumerate(segments) if i not in ghost_indices]
    return segments, duplicates_removed, len(ghost_indices)


def refine_transcription(
    raw: RawTranscription,
    *,
    cleaning: CleaningConfig,
    deduplication: DeduplicationConfig,
    line_splitting: LineSplittingConfig,
    quality: QualityConfig,
) -> TranscriptionResult:
    """Turn a raw transcription into the final, refined result.

    Args:
        raw: Output of the ASR engine.
        cleaning: Character artifact cleaning settings.
        deduplication: Repetition removal settings.
        line_splitting: Verse reflow settings.
        quality: Quality control settings.

    Returns:
        The refined `TranscriptionResult`.
    """
    segments = list(raw.segments)
    artifact_reports: list[ArtifactReport] = []
    duplicates_removed = 0
    ghost_blocks_removed = 0

    if cleaning.enabled:
        segments, artifact_reports = clean_transcription(
            segments,
            remove_fullwidth=cleaning.remove_fullwidth,
            remove_repeated_chars=cleaning.remove_repeated_chars,
            min_segment_words=cleaning.min_segment_words,
            custom_patterns=list(cleaning.custom_patterns),
        )

    if deduplication.enabled:
        segments, duplicates_removed, ghost_blocks_removed = _deduplicate(
            segments, deduplication
        )

    qualities = assess_transcription_quality(
        segments, low_confidence_threshold=quality.low_confidence_threshold
    )
    for seg, assessment in zip(segments, qualities):
        if seg.confidence is None:
            seg.confidence = assessment.confidence
        if assessment.is_degraded:
            seg.degraded_reason = assessment.reason
    degraded_indices = [i for i, q in enumerate(qualities) if q.is_degraded]
    report = quality_report(qualities)

    lyrics = "\n".join(s.text for s in segments) if segments else raw.lyrics

    if line_splitting.enabled:
        lyrics = relineate(
            lyrics,
            max_words=line_splitting.max_words,
            hard_max_words=line_splitting.hard_max_words,
            min_words_to_split=line_splitting.min_words_to_split,
        )

    if degraded_indices:
        logger.warning(
            "%d transcription segment(s) marked as degraded.", len(degraded_indices)
        )
    logger.info(
        "Transcript refined: %d segment(s) kept, grade=%s",
        len(segments),
        report.get("quality_grade", "?"),
    )
    return TranscriptionResult(
        lyrics=lyrics,
        segments=segments,
        language=raw.language,
        processing_time_seconds=raw.processing_time_seconds,
        duplicates_removed=duplicates_removed,
        ghost_blocks_removed=ghost_blocks_removed,
        quality_report=report,
        degraded_segments=degraded_indices,
        artifact_reports=artifact_reports,
    )
