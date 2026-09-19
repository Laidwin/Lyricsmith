"""Tests of the vocal separation module, with Demucs mocked."""

from __future__ import annotations

from pathlib import Path

import pytest

from lyricsmith.config import Config
from lyricsmith.separator import SeparationResult, VocalSeparator


def test_separate_creates_output_paths(
    config: Config, fake_demucs, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`separate` writes both stems and returns a consistent result."""
    monkeypatch.setattr("lyricsmith.separator.get_audio_duration", lambda _p: 5.0)

    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"fake")
    out_dir = tmp_path / "stems"

    separator = VocalSeparator(config)
    result = separator.separate(audio, out_dir)

    assert isinstance(result, SeparationResult)
    assert result.vocals_path == out_dir / "song_vocals.wav"
    assert result.no_vocals_path == out_dir / "song_no_vocals.wav"
    assert result.vocals_path.exists()
    assert result.no_vocals_path.exists()
    assert result.duration_seconds == 5.0
    assert result.processing_time_seconds >= 0.0


def test_separator_resolves_cpu_without_cuda(config: Config) -> None:
    """Without CUDA the device falls back to CPU."""
    separator = VocalSeparator(config)
    assert separator.device in {"cpu", "cuda"}


def test_quality_preset_fast(config: Config) -> None:
    """The 'fast' preset selects htdemucs with one shift."""
    config.models.demucs.quality_preset = "fast"
    separator = VocalSeparator(config)
    assert separator.model_name == "htdemucs"
    assert separator.shifts == 1


def test_quality_preset_best(config: Config) -> None:
    """The 'best' preset selects htdemucs_ft with four shifts."""
    config.models.demucs.quality_preset = "best"
    separator = VocalSeparator(config)
    assert separator.model_name == "htdemucs_ft"
    assert separator.shifts == 4


def test_preset_overrides_config(config: Config) -> None:
    """A preset takes precedence over 'name' and 'shifts' from config.yaml."""
    config.models.demucs.name = "mdx_extra"
    config.models.demucs.shifts = 99
    config.models.demucs.quality_preset = "best"
    separator = VocalSeparator(config)
    assert separator.model_name == "htdemucs_ft"
    assert separator.shifts == 4


def test_no_preset_uses_config(config: Config) -> None:
    """Without a preset the config.yaml values are used."""
    config.models.demucs.quality_preset = None
    config.models.demucs.name = "htdemucs"
    config.models.demucs.shifts = 3
    separator = VocalSeparator(config)
    assert separator.model_name == "htdemucs"
    assert separator.shifts == 3


def test_separate_missing_demucs_raises(
    config: Config, monkeypatch: pytest.MonkeyPatch, wav_factory, tmp_path: Path
) -> None:
    """Without Demucs installed or mocked, an explicit SeparationError is raised."""
    import sys

    monkeypatch.setitem(sys.modules, "demucs", None)
    monkeypatch.setitem(sys.modules, "demucs.api", None)

    from lyricsmith.exceptions import SeparationError

    audio = wav_factory(tmp_path / "x.wav")
    separator = VocalSeparator(config)
    with pytest.raises(SeparationError):
        separator.separate(audio, tmp_path / "out")
