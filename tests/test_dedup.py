"""Tests of the non consecutive deduplication, meaning ghost blocks."""

from __future__ import annotations

from lyricsmith.domain import Segment
from lyricsmith.utils.dedup import detect_ghost_blocks


def _block(start: float, texts: list[str]) -> list[Segment]:
    return [Segment(start + i, start + i + 1, t) for i, t in enumerate(texts)]


def test_ghost_block_detected() -> None:
    """A three line block repeated inside the time window is detected."""
    lines = ["take my hand", "hold me close", "never let go"]
    segments = _block(0.0, lines) + _block(5.0, lines)

    blocks = detect_ghost_blocks(segments, block_min_size=3, time_window_seconds=15.0)

    assert len(blocks) == 1
    assert blocks[0].ghost_indices == [3, 4, 5]
    assert blocks[0].original_indices == [0, 1, 2]


def test_ghost_block_not_detected_far_apart() -> None:
    """The same block repeated beyond the window is not a ghost block."""
    lines = ["take my hand", "hold me close", "never let go"]
    segments = _block(0.0, lines) + _block(60.0, lines)

    blocks = detect_ghost_blocks(segments, block_min_size=3, time_window_seconds=15.0)

    assert blocks == []


def test_ghost_block_min_size_respected() -> None:
    """A repetition shorter than block_min_size is not detected."""
    lines = ["take my hand", "hold me close"]
    segments = _block(0.0, lines) + _block(5.0, lines)

    blocks = detect_ghost_blocks(segments, block_min_size=3, time_window_seconds=15.0)

    assert blocks == []
