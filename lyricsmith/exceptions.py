"""Domain exceptions raised by the pipeline.

Every module raises explicit exceptions so that callers can diagnose failures
and handle each error category separately.
"""

from __future__ import annotations


class PipelineError(Exception):
    """Base class for every pipeline error."""


class ConfigError(PipelineError):
    """Raised when the configuration file holds invalid or unknown settings."""


class SeparationError(PipelineError):
    """Raised when vocal separation (Demucs) fails."""


class TranscriptionError(PipelineError):
    """Raised when lyrics transcription (HeartTranscriptor) fails."""


class ModelNotFoundError(PipelineError):
    """Raised when a model or checkpoint is missing from disk."""


class GPUError(PipelineError):
    """Raised when CUDA is unavailable or VRAM is insufficient."""


class AudioFormatError(PipelineError, ValueError):
    """Raised when an audio file is invalid or its format unsupported."""
