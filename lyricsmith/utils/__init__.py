"""Cross cutting services: audio, text cleaning, deduplication, quality, GPU."""

from __future__ import annotations

from .audio import convert_to_wav, get_audio_duration, validate_audio_file
from .cleaner import clean_segment_text, clean_transcription
from .dedup import GhostBlock, detect_ghost_blocks, remove_repetition_loops
from .gpu import assert_cuda, check_gpu_status, resolve_device
from .linesplit import relineate, split_line
from .logger import console, get_logger, set_level, setup_logging
from .preprocess import (
    PostprocessResult,
    PreprocessResult,
    postprocess_vocals,
    preprocess_audio,
)
from .quality import (
    SegmentQuality,
    assess_transcription_quality,
    quality_report,
)

__all__ = [
    "GhostBlock",
    "PostprocessResult",
    "PreprocessResult",
    "SegmentQuality",
    "assert_cuda",
    "assess_transcription_quality",
    "check_gpu_status",
    "clean_segment_text",
    "clean_transcription",
    "console",
    "convert_to_wav",
    "detect_ghost_blocks",
    "get_audio_duration",
    "get_logger",
    "postprocess_vocals",
    "preprocess_audio",
    "quality_report",
    "relineate",
    "remove_repetition_loops",
    "resolve_device",
    "set_level",
    "setup_logging",
    "split_line",
    "validate_audio_file",
]
