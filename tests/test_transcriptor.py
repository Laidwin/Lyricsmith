"""Tests of the ASR adapter, with heartlib mocked."""

from __future__ import annotations

from pathlib import Path

import pytest

from lyricsmith.config import Config
from lyricsmith.domain import RawTranscription, Segment
from lyricsmith.exceptions import ModelNotFoundError
from lyricsmith.transcriptor import LyricsTranscriptor


@pytest.fixture()
def transcriptor(config: Config, tmp_path: Path) -> LyricsTranscriptor:
    """Transcriptor pointing at an existing fake checkpoint."""
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(b"fake-weights")
    config.models.hearttranscriptor.model_path = str(ckpt)
    return LyricsTranscriptor(config)


def test_transcribe_returns_raw_transcription(
    transcriptor: LyricsTranscriptor, fake_heartlib, wav_factory, tmp_path: Path
) -> None:
    """`transcribe` returns the raw segments, without any refinement."""
    audio = wav_factory(tmp_path / "vocals.wav")

    result = transcriptor.transcribe(audio)

    assert isinstance(result, RawTranscription)
    assert result.lyrics == "hello world\nsecond line"
    assert len(result.segments) == 2
    assert all(isinstance(s, Segment) for s in result.segments)
    assert result.segments[0].start == 0.0
    assert result.segments[0].end == 1.5
    assert result.segments[0].text == "hello world"
    assert result.processing_time_seconds >= 0.0


def test_missing_checkpoint_raises(config: Config, tmp_path: Path) -> None:
    """A missing checkpoint raises ModelNotFoundError."""
    config.models.hearttranscriptor.model_path = str(tmp_path / "absent")
    audio = tmp_path / "any.wav"
    audio.write_bytes(b"fake")
    transcriptor = LyricsTranscriptor(config)
    with pytest.raises(ModelNotFoundError):
        transcriptor.transcribe(audio)


def test_normalize_output_from_plain_string() -> None:
    """The normalization accepts a plain lyrics string."""
    normalized = LyricsTranscriptor._normalize_output("just lyrics")
    assert normalized.lyrics == "just lyrics"
    assert normalized.segments == []
    assert normalized.language == "unknown"


def test_no_speech_segments_filtered(
    config: Config, tmp_path: Path, wav_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Segments without lyrics are excluded from the raw transcription."""
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(b"fake")
    config.models.hearttranscriptor.model_path = str(ckpt)

    class FakePipeline:
        def __call__(self, inputs, **kwargs) -> dict:
            return {
                "text": "real words",
                "chunks": [
                    {"timestamp": (0.0, 1.0), "text": "real words"},
                    {"timestamp": (1.0, 2.0), "text": "   "},
                    {"timestamp": (2.0, 3.0), "text": ""},
                ],
            }

    monkeypatch.setattr(LyricsTranscriptor, "_load_model", lambda self: FakePipeline())
    transcriptor = LyricsTranscriptor(config)
    result = transcriptor.transcribe(wav_factory(tmp_path / "a.wav"))

    assert len(result.segments) == 1
    assert result.segments[0].text == "real words"


def test_generate_kwargs_omit_unset_options(
    config: Config, tmp_path: Path
) -> None:
    """Options left unset in the configuration are not forwarded to the model."""
    settings = config.models.hearttranscriptor
    settings.model_path = str(tmp_path)
    settings.beam_size = 5
    settings.temperature = [0.0, 0.2]
    settings.no_speech_threshold = None
    settings.compression_ratio_threshold = None

    kwargs = LyricsTranscriptor(config)._build_generate_kwargs()

    assert kwargs["num_beams"] == 5
    assert kwargs["temperature"] == (0.0, 0.2)
    assert "no_speech_threshold" not in kwargs
    assert "compression_ratio_threshold" not in kwargs


def test_release_model_clears_the_cached_model(
    config: Config, tmp_path: Path
) -> None:
    """Releasing drops the cached model so its memory can be reclaimed."""
    config.models.hearttranscriptor.model_path = str(tmp_path)
    config.models.hearttranscriptor.lazy_load = True
    transcriptor = LyricsTranscriptor(config)
    transcriptor._model = object()

    transcriptor._release_model()

    assert transcriptor._model is None
