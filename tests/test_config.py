"""Tests of the typed configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from lyricsmith.config import Config
from lyricsmith.exceptions import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_project_config_is_valid() -> None:
    """The config.yaml shipped with the project parses without error."""
    config = Config.load(PROJECT_ROOT / "config.yaml")
    assert config.models.demucs.name
    assert config.pipeline.supported_formats


def test_defaults_apply_to_missing_keys(tmp_path: Path) -> None:
    """Sections left out fall back to the declared defaults."""
    config = Config.load(_write(tmp_path / "c.yaml", "models:\n  demucs:\n    shifts: 2\n"))

    assert config.models.demucs.shifts == 2
    assert config.models.demucs.name == "htdemucs"
    assert config.pipeline.postprocessing.reverb_aggressiveness == 0.45
    assert config.pipeline.quality.low_confidence_threshold == 0.4


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    """A typo raises instead of silently falling back to a default."""
    source = _write(tmp_path / "c.yaml", "pipeline:\n  cleaning:\n    enabledd: false\n")

    with pytest.raises(ConfigError, match="enabledd"):
        Config.load(source)


def test_unknown_section_is_rejected(tmp_path: Path) -> None:
    """An unknown section raises as well."""
    source = _write(tmp_path / "c.yaml", "pipeline:\n  cleanup:\n    enabled: false\n")

    with pytest.raises(ConfigError, match="cleanup"):
        Config.load(source)


def test_values_are_coerced(tmp_path: Path) -> None:
    """Numbers written as integers are coerced to the declared type."""
    config = Config.load(
        _write(tmp_path / "c.yaml", "models:\n  demucs:\n    overlap: 1\n")
    )

    assert isinstance(config.models.demucs.overlap, float)
    assert config.models.demucs.overlap == 1.0


def test_supported_formats_are_normalized(tmp_path: Path) -> None:
    """Extensions are lowercased once, at load time."""
    config = Config.load(
        _write(tmp_path / "c.yaml", "pipeline:\n  supported_formats: ['.MP3', '.Wav']\n")
    )

    assert config.pipeline.supported_formats == (".mp3", ".wav")


def test_missing_file_raises(tmp_path: Path) -> None:
    """A missing configuration file is reported clearly."""
    with pytest.raises(FileNotFoundError):
        Config.load(tmp_path / "nope.yaml")


def test_relative_paths_resolve_against_the_project_root(tmp_path: Path) -> None:
    """Configured paths are resolved relatively to the configuration file."""
    config = Config.load(_write(tmp_path / "c.yaml", "pipeline:\n  output_dir: ./out\n"))

    assert config.resolve_path(config.pipeline.output_dir) == (tmp_path / "out").resolve()
