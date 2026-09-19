"""End to end integration test of the orchestrator.

A synthetic five second WAV file is generated and pushed through the full
pipeline with Demucs and heartlib doubles, so no real model and no GPU are
needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lyricsmith.config import Config
from lyricsmith.pipeline import (
    SEPARATION_PHASE,
    TRANSCRIPTION_PHASE,
    BatchProgress,
    LyricsPipeline,
    PipelineResult,
)


@pytest.fixture()
def synthetic_wav(tmp_path: Path) -> Path:
    """Create a five second mono 16 kHz WAV holding a 440 Hz sine wave."""
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")

    sample_rate = 16000
    t = np.linspace(0, 5.0, int(sample_rate * 5.0), endpoint=False)
    waveform = 0.2 * np.sin(2 * np.pi * 440 * t).astype("float32")

    path = tmp_path / "song.wav"
    sf.write(str(path), waveform, sample_rate, subtype="PCM_16")
    return path


@pytest.fixture()
def ready_config(config: Config, tmp_path: Path) -> Config:
    """Test configuration pointing at an existing fake checkpoint."""
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(b"fake-weights")
    config.models.hearttranscriptor.model_path = str(ckpt)
    return config


def test_pipeline_end_to_end(
    ready_config: Config,
    synthetic_wav: Path,
    fake_demucs,
    fake_heartlib,
    wav_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The full pipeline produces the three expected output files."""
    # The WAV conversion is stubbed to avoid librosa and ffmpeg, while still
    # writing a real WAV because the transcription reads it back with soundfile.
    monkeypatch.setattr(
        "lyricsmith.pipeline.convert_to_wav", lambda src, dst, **kw: wav_factory(dst)
    )

    pipeline = LyricsPipeline(ready_config)
    result = pipeline.run(synthetic_wav)

    assert isinstance(result, PipelineResult)

    assert result.lyrics_txt_path.exists()
    assert result.lyrics_json_path.exists()
    assert result.metadata_path.exists()
    assert result.output_dir.name == "song"

    assert result.lyrics_txt_path.read_text(encoding="utf-8") == "hello world\nsecond line"

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["num_segments"] == 2
    assert metadata["separation_used"] is True

    # keep_stems is false, so no stem is kept in the output directory.
    assert result.vocals_path is None


def test_pipeline_no_separation(
    ready_config: Config,
    synthetic_wav: Path,
    fake_heartlib,
    wav_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With separate=False, Demucs is skipped but transcription still runs."""
    monkeypatch.setattr(
        "lyricsmith.pipeline.convert_to_wav", lambda src, dst, **kw: wav_factory(dst)
    )

    pipeline = LyricsPipeline(ready_config)
    result = pipeline.run(synthetic_wav, separate=False)

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["separation_used"] is False
    assert result.lyrics_txt_path.exists()


def test_run_batch_handles_failures(
    ready_config: Config,
    synthetic_wav: Path,
    fake_heartlib,
    wav_factory,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """run_batch keeps going when one file is invalid."""
    monkeypatch.setattr(
        "lyricsmith.pipeline.convert_to_wav", lambda src, dst, **kw: wav_factory(dst)
    )

    bad = tmp_path / "broken.xyz"
    bad.write_bytes(b"nope")

    pipeline = LyricsPipeline(ready_config)
    results = pipeline.run_batch([synthetic_wav, bad], separate=False)

    assert len(results) == 1


def test_run_batch_reports_progress(
    ready_config: Config,
    synthetic_wav: Path,
    fake_heartlib,
    wav_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The batch reports progress through the callback, rendering nothing."""
    monkeypatch.setattr(
        "lyricsmith.pipeline.convert_to_wav", lambda src, dst, **kw: wav_factory(dst)
    )
    events: list[BatchProgress] = []

    pipeline = LyricsPipeline(ready_config)
    pipeline.run_batch([synthetic_wav], separate=False, on_progress=events.append)

    phases = {event.phase for event in events}
    assert phases == {SEPARATION_PHASE, TRANSCRIPTION_PHASE}
    assert events[-1].completed == events[-1].total == 1
    assert any(event.current == synthetic_wav.name for event in events)
