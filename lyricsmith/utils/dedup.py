"""Removal of repeated segments produced by the ASR engine.

Two distinct repetitions are handled: hallucination loops, where the same
sentence is emitted several times in a row, and ghost blocks, where a group of
segments reappears a few seconds later without being adjacent.

These are text algorithms working on `Segment` entities; they involve no audio.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass

from ..domain import Segment
from .logger import get_logger

logger = get_logger("utils.dedup")


def _norm_text(value: str) -> str:
    """Normalize a text before similarity comparison."""
    return (value or "").strip().lower()


def remove_repetition_loops(
    segments: list[Segment],
    similarity_threshold: float = 0.85,
    max_repeats: int = 2,
) -> tuple[list[Segment], int]:
    """Remove hallucination loops, meaning nearly identical repeated segments.

    Whisper sometimes repeats the same sentence several times in a row. Each
    segment is compared to the last kept one with `difflib.SequenceMatcher`;
    beyond `max_repeats` consecutive similar occurrences, the extra ones are
    dropped.

    Args:
        segments: Segments to filter.
        similarity_threshold: Similarity ratio between 0 and 1 above which two
            segments are considered identical.
        max_repeats: Number of consecutive similar occurrences to keep.

    Returns:
        A tuple ``(kept_segments, removed_count)``.
    """
    if not segments:
        return [], 0

    cleaned: list[Segment] = []
    removed = 0
    run_count = 0
    last_text: str | None = None

    for seg in segments:
        text = _norm_text(seg.text)
        if last_text is not None:
            ratio = difflib.SequenceMatcher(None, last_text, text).ratio()
            similar = ratio >= similarity_threshold
        else:
            similar = False

        if similar:
            run_count += 1
            if run_count > max_repeats:
                removed += 1
                continue
        else:
            run_count = 1
            last_text = text

        cleaned.append(seg)

    if removed:
        logger.warning(
            "%d repeated segment(s) removed (hallucination loop).", removed
        )
    return cleaned, removed


@dataclass
class GhostBlock:
    """A block of segments duplicated within a short time window."""

    original_indices: list[int]
    ghost_indices: list[int]
    similarity: float
    time_delta_seconds: float


def detect_ghost_blocks(
    segments: list[Segment],
    block_min_size: int = 3,
    similarity_threshold: float = 0.85,
    time_window_seconds: float = 15.0,
) -> list[GhostBlock]:
    """Detect blocks of segments repeated within a short time window.

    Unlike `remove_repetition_loops`, which only handles adjacent repetitions,
    this looks for a block of at least `block_min_size` segments reappearing
    later without being adjacent, typically a pre chorus duplicated by the model
    at the same place. A genuine structural repeat, such as a chorus coming back
    a minute later, exceeds `time_window_seconds` and is therefore ignored.

    Args:
        segments: Segments to scan, ordered by start time.
        block_min_size: Minimum number of segments forming a block.
        similarity_threshold: Similarity ratio above which two segments match.
        time_window_seconds: Maximum time gap between a block and its duplicate.

    Returns:
        The detected `GhostBlock` instances, whose ``ghost_indices`` list the
        segments to remove.
    """
    n = len(segments)
    if n < 2 * block_min_size:
        return []

    texts = [_norm_text(s.text) for s in segments]
    starts = [float(s.start) for s in segments]

    blocks: list[GhostBlock] = []
    consumed: set[int] = set()

    for i in range(n):
        if i in consumed:
            continue
        for j in range(i + block_min_size, n):
            if j in consumed:
                continue
            delta = starts[j] - starts[i]
            if delta > time_window_seconds:
                break  # Start times are sorted, nothing further fits the window.

            ratios: list[float] = []
            length = 0
            while (
                i + length < j
                and j + length < n
                and (j + length) not in consumed
            ):
                ratio = difflib.SequenceMatcher(
                    None, texts[i + length], texts[j + length]
                ).ratio()
                if ratio < similarity_threshold:
                    break
                ratios.append(ratio)
                length += 1

            if length >= block_min_size:
                ghost = list(range(j, j + length))
                blocks.append(
                    GhostBlock(
                        original_indices=list(range(i, i + length)),
                        ghost_indices=ghost,
                        similarity=round(sum(ratios) / length, 3),
                        time_delta_seconds=round(delta, 2),
                    )
                )
                consumed.update(ghost)
                break

    if blocks:
        total = sum(len(b.ghost_indices) for b in blocks)
        logger.warning(
            "%d ghost block(s) detected (%d duplicated segment(s)).", len(blocks), total
        )
    return blocks
