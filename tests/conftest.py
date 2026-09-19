"""Shared fixtures: fake module injection and test configuration.

The tests must not depend on a GPU, on real Demucs or heartlib installs, or on
downloaded weights. The device is therefore forced to CPU and controllable test
doubles are provided.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from lyricsmith.config import Config

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def config(tmp_path: Path) -> Config:
    """Real config.yaml, forced on CPU with a temporary output directory."""
    cfg = Config.load(PROJECT_ROOT / "config.yaml")
    cfg.models.demucs.device = "cpu"
    cfg.models.hearttranscriptor.device = "cpu"
    cfg.pipeline.output_dir = str(tmp_path / "outputs")
    cfg.pipeline.keep_stems = False
    # Audio pre and post processing are tested separately, and disabling them
    # here avoids heavy dependencies and real audio.
    cfg.pipeline.preprocessing.enabled = False
    cfg.pipeline.postprocessing.enabled = False
    return cfg


@pytest.fixture()
def fake_demucs(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Inject a fake `demucs.api` module exposing Separator and save_audio."""

    class FakeSeparator:
        samplerate = 44100

        def __init__(self, *args, **kwargs) -> None:
            self.kwargs = kwargs

        def separate_audio_file(self, path: str):
            # Scalar values are enough for the summing logic in
            # VocalSeparator.separate.
            stems = {"vocals": 1.0, "drums": 0.5, "bass": 0.3, "other": 0.2}
            return None, stems

    def fake_save_audio(tensor, path: str, samplerate: int = 44100) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"RIFF\x00\x00\x00\x00WAVEfake")

    module = types.ModuleType("demucs")
    api = types.ModuleType("demucs.api")
    api.Separator = FakeSeparator
    api.save_audio = fake_save_audio
    module.api = api

    monkeypatch.setitem(sys.modules, "demucs", module)
    monkeypatch.setitem(sys.modules, "demucs.api", api)
    return api


@pytest.fixture()
def fake_heartlib(monkeypatch: pytest.MonkeyPatch):
    """Replace the model loading with a fake ASR pipeline.

    Instead of depending on heartlib, transformers and real weights,
    `LyricsTranscriptor._load_model` is patched to return a callable double
    mimicking the output of a transformers ASR pipeline.
    """
    from lyricsmith.transcriptor import LyricsTranscriptor

    class FakePipeline:
        def __call__(self, inputs, **kwargs) -> dict:
            return {
                "text": "hello world second line",
                "chunks": [
                    {"timestamp": (0.0, 1.5), "text": "hello world"},
                    {"timestamp": (1.5, 3.0), "text": "second line"},
                ],
            }

    monkeypatch.setattr(LyricsTranscriptor, "_load_model", lambda self: FakePipeline())
    return FakePipeline


def write_wav(path, *, seconds: float = 1.0, sample_rate: int = 16000) -> "Path":
    """Write a short real mono WAV file, used as a test helper."""
    import numpy as np
    import soundfile as sf

    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    waveform = (0.1 * np.sin(2 * np.pi * 440 * t)).astype("float32")
    sf.write(str(path), waveform, sample_rate, subtype="PCM_16")
    return Path(path)


@pytest.fixture()
def wav_factory():
    """Expose `write_wav` to generate test WAV files."""
    pytest.importorskip("numpy")
    pytest.importorskip("soundfile")
    return write_wav
