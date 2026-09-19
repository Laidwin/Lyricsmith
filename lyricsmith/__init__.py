"""Music lyrics transcription pipeline (Demucs + HeartTranscriptor).

Exposes the high level public API for library usage:

    from lyricsmith import Config, LyricsPipeline

    config = Config.load()
    result = LyricsPipeline(config).run(Path("song.mp3"))
    print(result.transcription.lyrics)

The package is layered: `domain` holds the entities and depends on nothing,
`config` parses the settings, `utils` holds the audio and text services,
`separator` and `transcriptor` adapt the two models, `refine` applies the text
post processing policy, `report` renders the results, and `pipeline`
orchestrates the whole run.
"""

from __future__ import annotations

from .config import Config
from .domain import (
    ArtifactReport,
    RawTranscription,
    Segment,
    TranscriptionResult,
)
from .exceptions import (
    AudioFormatError,
    ConfigError,
    GPUError,
    ModelNotFoundError,
    PipelineError,
    SeparationError,
    TranscriptionError,
)
from .pipeline import BatchProgress, LyricsPipeline, PipelineResult
from .refine import refine_transcription
from .report import format_annotated_lyrics
from .separator import SeparationResult, VocalSeparator
from .transcriptor import LyricsTranscriptor

__version__ = "0.1.0"

__all__ = [
    "ArtifactReport",
    "AudioFormatError",
    "BatchProgress",
    "Config",
    "ConfigError",
    "GPUError",
    "LyricsPipeline",
    "LyricsTranscriptor",
    "ModelNotFoundError",
    "PipelineError",
    "PipelineResult",
    "RawTranscription",
    "Segment",
    "SeparationError",
    "SeparationResult",
    "TranscriptionError",
    "TranscriptionResult",
    "VocalSeparator",
    "format_annotated_lyrics",
    "refine_transcription",
]
