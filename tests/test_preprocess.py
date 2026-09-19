"""Tests of the audio pre and post processing, using the real dependencies."""

from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")


def _write(path: Path, data, sr: int) -> Path:
    sf.write(str(path), data.astype("float32"), sr, subtype="PCM_16")
    return path


def _sine(freq: float, seconds: float, sr: int, amp: float = 0.3):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype("float32")


def _noise(seconds: float, sr: int, amp: float = 0.2):
    rng = np.random.default_rng(0)
    return (amp * rng.standard_normal(int(sr * seconds))).astype("float32")


def test_resample_44100_unchanged(tmp_path: Path) -> None:
    """A file already at 44100 Hz is returned untouched."""
    from lyricsmith.utils.preprocess import resample_if_needed

    src = _write(tmp_path / "a.wav", _sine(440, 1.0, 44100), 44100)
    out = resample_if_needed(src, target_sr=44100)
    assert out == src


def test_resample_48000_converted(tmp_path: Path) -> None:
    """A file at 48000 Hz is converted to 44100 Hz."""
    from lyricsmith.utils.preprocess import resample_if_needed

    src = _write(tmp_path / "a.wav", _sine(440, 1.0, 48000), 48000)
    out = resample_if_needed(src, target_sr=44100, output_path=tmp_path / "out.wav")
    assert sf.info(str(out)).samplerate == 44100


def test_loudness_normalization_applied(tmp_path: Path) -> None:
    """The target loudness is reached within 1 dB."""
    pytest.importorskip("pyloudnorm")
    from lyricsmith.utils.preprocess import _measure_lufs, normalize_loudness

    sr = 44100
    src = _write(tmp_path / "a.wav", _noise(3.0, sr, amp=0.05), sr)
    out = normalize_loudness(src, target_lufs=-14.0, output_path=tmp_path / "n.wav")
    assert abs(_measure_lufs(out) - (-14.0)) <= 1.0


def test_dc_offset_removed(tmp_path: Path) -> None:
    """A DC offset falls below the threshold after filtering."""
    from lyricsmith.utils.preprocess import remove_dc_offset

    sr = 44100
    signal = _sine(440, 2.0, sr) + 0.3
    src = _write(tmp_path / "a.wav", signal, sr)
    out = remove_dc_offset(src, output_path=tmp_path / "dc.wav")
    data, _ = sf.read(str(out), dtype="float32")
    assert abs(float(np.mean(data))) < 0.01


def test_reverb_reduction_does_not_clip(tmp_path: Path) -> None:
    """The output of reduce_reverb does not clip."""
    pytest.importorskip("noisereduce")
    from lyricsmith.utils.preprocess import reduce_reverb

    sr = 44100
    src = _write(tmp_path / "v.wav", _noise(2.0, sr, amp=0.9), sr)
    out = reduce_reverb(src, output_path=tmp_path / "r.wav", aggressiveness=0.3)
    data, _ = sf.read(str(out), dtype="float32")
    assert float(np.max(np.abs(data))) < 1.0


def test_reverb_aggressiveness_clamped(tmp_path: Path) -> None:
    """An aggressiveness above 0.6 is clamped instead of raising."""
    pytest.importorskip("noisereduce")
    from lyricsmith.utils.preprocess import reduce_reverb

    sr = 44100
    src = _write(tmp_path / "v.wav", _noise(1.0, sr, amp=0.5), sr)
    out = reduce_reverb(src, output_path=tmp_path / "r.wav", aggressiveness=0.95)
    assert Path(out).exists()


def test_preprocess_idempotent(tmp_path: Path) -> None:
    """Running the result through the chain a second time keeps it valid."""
    pytest.importorskip("pyloudnorm")
    from lyricsmith.utils.preprocess import preprocess_audio

    sr = 48000
    src = _write(tmp_path / "song.wav", _noise(2.0, sr, amp=0.1), sr)
    first = preprocess_audio(src, tmp_path / "p1")
    second = preprocess_audio(first.output_path, tmp_path / "p2")

    data, _ = sf.read(str(second.output_path), dtype="float32")
    assert data.size > 0
    assert float(np.max(np.abs(data))) <= 1.0
