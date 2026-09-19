"""Typed configuration loaded from `config.yaml`.

Every setting is parsed once into dataclasses, so each default is declared in
exactly one place and consumers read plain typed attributes instead of
dictionary lookups carrying their own fallback value. Unknown keys are
rejected rather than silently ignored, which turns a typo into an error
instead of a surprising default.

The configuration is a value object: it is loaded explicitly by the entry
points (the CLI and the setup script) and injected into the pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import UnionType
from typing import Any, Mapping, TypeVar, Union, get_args, get_origin, get_type_hints

import yaml

from .exceptions import ConfigError

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

_T = TypeVar("_T")


@dataclass
class DemucsConfig:
    """Vocal separation settings."""

    name: str = "htdemucs"
    device: str = "cuda"
    shifts: int = 1
    overlap: float = 0.25
    jobs: int = 0
    quality_preset: str | None = None


@dataclass
class TranscriptorConfig:
    """HeartTranscriptor model and generation settings.

    The generation options default to ``None``, meaning "not set": they are
    then left out of the generation call so the model defaults apply.
    """

    model_path: str = "./models/ckpt"
    device: str = "cuda"
    lazy_load: bool = True
    dtype: str = "bf16"
    chunk_length_s: int = 30
    batch_size: int = 16
    condition_on_previous_text: bool | None = None
    no_speech_threshold: float | None = None
    logprob_threshold: float | None = None
    compression_ratio_threshold: float | None = None
    temperature: Any = None
    beam_size: int | None = None


@dataclass
class ModelsConfig:
    """Model section of the configuration."""

    demucs: DemucsConfig = field(default_factory=DemucsConfig)
    hearttranscriptor: TranscriptorConfig = field(default_factory=TranscriptorConfig)


@dataclass
class PreprocessingConfig:
    """Audio pre processing applied before separation."""

    enabled: bool = True
    resample: bool = True
    resample_target_sr: int = 44100
    remove_dc_offset: bool = True
    normalize_loudness: bool = True
    target_lufs: float = -14.0


@dataclass
class PostprocessingConfig:
    """Vocal post processing applied after separation."""

    enabled: bool = True
    reduce_reverb: bool = True
    reverb_aggressiveness: float = 0.45
    enhance_clarity: bool = True


@dataclass
class CleaningConfig:
    """Character artifact cleaning settings."""

    enabled: bool = True
    remove_fullwidth: bool = True
    remove_repeated_chars: bool = True
    min_segment_words: int = 1
    custom_patterns: tuple[str, ...] = ()


@dataclass
class DeduplicationConfig:
    """Repetition removal settings."""

    enabled: bool = True
    consecutive_max_repeats: int = 2
    consecutive_similarity: float = 0.85
    ghost_block_min_size: int = 3
    ghost_block_time_window: float = 15.0
    ghost_block_similarity: float = 0.85


@dataclass
class LineSplittingConfig:
    """Verse reflow settings."""

    enabled: bool = True
    max_words: int = 10
    hard_max_words: int = 16
    min_words_to_split: int = 12


@dataclass
class QualityConfig:
    """Transcription quality control settings."""

    low_confidence_threshold: float = 0.4


@dataclass
class ReportConfig:
    """Report generation settings."""

    json: bool = True


@dataclass
class PipelineConfig:
    """Pipeline section of the configuration."""

    output_dir: str = "./outputs"
    keep_stems: bool = False
    target_sample_rate: int = 16000
    supported_formats: tuple[str, ...] = (".mp3", ".wav", ".flac", ".m4a", ".ogg")
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    postprocessing: PostprocessingConfig = field(default_factory=PostprocessingConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    deduplication: DeduplicationConfig = field(default_factory=DeduplicationConfig)
    line_splitting: LineSplittingConfig = field(default_factory=LineSplittingConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    report: ReportConfig = field(default_factory=ReportConfig)


def _coerce(value: Any, hint: Any) -> Any:
    """Convert a YAML value to the type declared by a dataclass field."""
    if value is None:
        return None
    if get_origin(hint) in (Union, UnionType):
        members = [arg for arg in get_args(hint) if arg is not type(None)]
        return _coerce(value, members[0]) if len(members) == 1 else value
    if get_origin(hint) is tuple:
        return tuple(value)
    if hint is bool:
        return bool(value)
    if hint in (int, float, str):
        return hint(value)
    return value


def _build(cls: type[_T], data: Mapping[str, Any], section: str) -> _T:
    """Build a leaf configuration dataclass from a raw mapping.

    Args:
        cls: Target dataclass.
        data: Raw mapping read from the YAML file.
        section: Dotted section name, used in error messages.

    Returns:
        The populated dataclass instance.

    Raises:
        ConfigError: When the mapping holds a key the dataclass does not declare.
    """
    if not isinstance(data, Mapping):
        raise ConfigError(f"Section '{section}' must be a mapping, got {type(data).__name__}.")

    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(
            f"Unknown key(s) in section '{section}': {', '.join(unknown)}. "
            f"Known keys: {', '.join(sorted(known))}."
        )
    return cls(**{key: _coerce(value, hints[key]) for key, value in data.items()})


def _section(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Return a subsection of the raw configuration, defaulting to empty."""
    return data.get(key) or {}


@dataclass
class Config:
    """Configuration of a pipeline run, loaded from a YAML file."""

    path: Path
    models: ModelsConfig = field(default_factory=ModelsConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        """Load and validate the configuration file.

        Args:
            path: Path to an alternative `config.yaml`. When omitted, the
                ``LYRICSMITH_CONFIG`` environment variable is used, then
                the default file at the project root.

        Returns:
            The parsed configuration.

        Raises:
            FileNotFoundError: When the resolved file does not exist.
            ConfigError: When a section or a key is invalid.
        """
        resolved = cls._resolve_path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"Configuration file not found: {resolved}")

        with resolved.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, Mapping):
            raise ConfigError(f"Configuration file must hold a mapping: {resolved}")

        return cls(
            path=resolved,
            models=cls._build_models(_section(raw, "models")),
            pipeline=cls._build_pipeline(_section(raw, "pipeline")),
        )

    @staticmethod
    def _build_models(raw: Mapping[str, Any]) -> ModelsConfig:
        """Build the models section."""
        unknown = sorted(set(raw) - {"demucs", "hearttranscriptor"})
        if unknown:
            raise ConfigError(f"Unknown section(s) under 'models': {', '.join(unknown)}.")
        return ModelsConfig(
            demucs=_build(DemucsConfig, _section(raw, "demucs"), "models.demucs"),
            hearttranscriptor=_build(
                TranscriptorConfig,
                _section(raw, "hearttranscriptor"),
                "models.hearttranscriptor",
            ),
        )

    @staticmethod
    def _build_pipeline(raw: Mapping[str, Any]) -> PipelineConfig:
        """Build the pipeline section, including every nested step."""
        nested = {
            "preprocessing": PreprocessingConfig,
            "postprocessing": PostprocessingConfig,
            "cleaning": CleaningConfig,
            "deduplication": DeduplicationConfig,
            "line_splitting": LineSplittingConfig,
            "quality": QualityConfig,
            "report": ReportConfig,
        }
        scalar_keys = {"output_dir", "keep_stems", "target_sample_rate", "supported_formats"}
        unknown = sorted(set(raw) - set(nested) - scalar_keys)
        if unknown:
            raise ConfigError(f"Unknown key(s) in section 'pipeline': {', '.join(unknown)}.")

        formats = raw.get("supported_formats")
        config = PipelineConfig(
            **{
                name: _build(step_cls, _section(raw, name), f"pipeline.{name}")
                for name, step_cls in nested.items()
            }
        )
        config.output_dir = str(raw.get("output_dir", config.output_dir))
        config.keep_stems = bool(raw.get("keep_stems", config.keep_stems))
        config.target_sample_rate = int(
            raw.get("target_sample_rate", config.target_sample_rate)
        )
        if formats is not None:
            config.supported_formats = tuple(str(fmt).lower() for fmt in formats)
        return config

    @staticmethod
    def _resolve_path(path: str | Path | None) -> Path:
        """Resolve the configuration path from the argument, env or default."""
        if path is not None:
            return Path(path).expanduser().resolve()
        env_path = os.environ.get("LYRICSMITH_CONFIG")
        if env_path:
            return Path(env_path).expanduser().resolve()
        return _DEFAULT_CONFIG_PATH

    @property
    def project_root(self) -> Path:
        """Project root, meaning the directory holding `config.yaml`."""
        return self.path.parent

    def resolve_path(self, value: str | Path) -> Path:
        """Resolve a configured path relatively to the project root."""
        candidate = Path(value).expanduser()
        if candidate.is_absolute():
            return candidate
        return (self.project_root / candidate).resolve()
