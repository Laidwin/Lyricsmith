"""Audio pre and post processing around vocal separation.

Pre processing, applied before Demucs: resampling to 44.1 kHz, DC offset
removal and LUFS loudness normalization. Post processing, applied after Demucs
on the isolated vocals: reverb reduction and clarity enhancement.

Heavy dependencies (`pyloudnorm`, `noisereduce`, `scipy`) are imported lazily,
and the orchestrating functions (`preprocess_audio`, `postprocess_vocals`)
degrade gracefully when one of them is missing. No network access is performed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .logger import get_logger

logger = get_logger("utils.preprocess")

_DEFAULT_TARGET_SR = 44100
_DEFAULT_TARGET_LUFS = -14.0
_MAX_REVERB_AGGRESSIVENESS = 0.6


@dataclass
class PreprocessResult:
    """Summary of the pre processing applied before separation."""

    output_path: Path
    original_sr: int
    resampled: bool
    dc_removed: bool
    loudness_normalized: bool
    original_lufs: float
    target_lufs: float


@dataclass
class PostprocessResult:
    """Summary of the post processing applied to the isolated vocals."""

    output_path: Path
    reverb_reduced: bool
    clarity_enhanced: bool


def _load(path: Path) -> tuple[Any, int]:
    """Load an audio file as float32, shaped ``(frames,)`` or ``(frames, channels)``."""
    import soundfile as sf

    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        return data, int(sr)
    except (RuntimeError, ValueError):
        import librosa

        y, sr = librosa.load(str(path), sr=None, mono=False)
        if getattr(y, "ndim", 1) == 2:
            y = y.T  # librosa returns (channels, frames), soundfile wants the transpose.
        return y.astype("float32"), int(sr)


def _save(path: Path, data: Any, sr: int) -> Path:
    """Write audio data as a 16 bit PCM WAV file."""
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sr, subtype="PCM_16")
    return path


def _samplerate(path: Path) -> int:
    """Return the sample rate of an audio file."""
    import soundfile as sf

    try:
        return int(sf.info(str(path)).samplerate)
    except (RuntimeError, ValueError):
        import librosa

        return int(librosa.get_samplerate(str(path)))


def _measure_lufs(path: Path) -> float:
    """Return the integrated loudness of an audio file, in LUFS."""
    import pyloudnorm as pyln

    data, sr = _load(path)
    return float(pyln.Meter(sr).integrated_loudness(data))


def _prevent_clip(data: Any, ceiling: float = 0.99) -> Any:
    """Scale the signal down so that its peak stays below `ceiling`."""
    import numpy as np

    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak > ceiling:
        data = data * (ceiling / peak)
    return data


def _default_out(audio_path: Path, suffix: str, output_dir: Path | None = None) -> Path:
    """Build a default output path by appending `suffix` to the file stem."""
    audio_path = Path(audio_path)
    name = f"{audio_path.stem}{suffix}.wav"
    return (output_dir or audio_path.parent) / name


def resample_if_needed(
    audio_path: Path, target_sr: int = _DEFAULT_TARGET_SR, output_path: Path | None = None
) -> Path:
    """Resample the file to `target_sr`, or return the input path untouched.

    Args:
        audio_path: Source file.
        target_sr: Target sample rate.
        output_path: Destination file. Defaults to a sibling `_resampled.wav`.

    Returns:
        The path of the resampled file, or the input path when it already has
        the target sample rate.
    """
    audio_path = Path(audio_path)
    if _samplerate(audio_path) == target_sr:
        return audio_path

    import librosa

    data, sr = _load(audio_path)
    resampled = librosa.resample(data, orig_sr=sr, target_sr=target_sr, axis=0)
    out = output_path or _default_out(audio_path, "_resampled")
    return _save(out, resampled, target_sr)


def remove_dc_offset(audio_path: Path, output_path: Path | None = None) -> Path:
    """Remove the DC offset with a 20 Hz Butterworth high pass filter.

    Args:
        audio_path: Source file.
        output_path: Destination file. Defaults to a sibling `_dc.wav`.

    Returns:
        The path of the filtered file.
    """
    from scipy.signal import butter, sosfilt

    audio_path = Path(audio_path)
    data, sr = _load(audio_path)
    sos = butter(2, 20.0, btype="highpass", fs=sr, output="sos")
    filtered = sosfilt(sos, data, axis=0).astype("float32")
    out = output_path or _default_out(audio_path, "_dc")
    return _save(out, filtered, sr)


def normalize_loudness(
    audio_path: Path,
    target_lufs: float = _DEFAULT_TARGET_LUFS,
    output_path: Path | None = None,
) -> Path:
    """Normalize loudness to `target_lufs` with pyloudnorm.

    Args:
        audio_path: Source file.
        target_lufs: Target integrated loudness, in LUFS.
        output_path: Destination file. Defaults to a sibling `_norm.wav`.

    Returns:
        The path of the normalized file.
    """
    import pyloudnorm as pyln

    audio_path = Path(audio_path)
    data, sr = _load(audio_path)
    loudness = pyln.Meter(sr).integrated_loudness(data)
    normalized = pyln.normalize.loudness(data, loudness, target_lufs)
    normalized = _prevent_clip(normalized).astype("float32")
    out = output_path or _default_out(audio_path, "_norm")
    return _save(out, normalized, sr)


def preprocess_audio(
    audio_path: Path,
    output_dir: Path,
    *,
    resample: bool = True,
    target_sr: int = _DEFAULT_TARGET_SR,
    remove_dc: bool = True,
    normalize: bool = True,
    target_lufs: float = _DEFAULT_TARGET_LUFS,
) -> PreprocessResult:
    """Chain resampling, DC removal and loudness normalization.

    Every step is guarded: a missing dependency or a failure is logged and the
    step is skipped rather than aborting the pipeline.

    Args:
        audio_path: Source file.
        output_dir: Directory receiving the intermediate files.
        resample: Whether to resample to `target_sr`.
        target_sr: Target sample rate.
        remove_dc: Whether to remove the DC offset.
        normalize: Whether to normalize loudness.
        target_lufs: Target integrated loudness, in LUFS.

    Returns:
        A `PreprocessResult` describing what was actually applied.
    """
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = audio_path.stem

    original_sr = _samplerate(audio_path)
    original_lufs = 0.0
    if normalize:
        try:
            original_lufs = _measure_lufs(audio_path)
        except Exception as exc:
            logger.warning("Could not measure loudness (%s).", exc)

    current = audio_path
    resampled = dc_removed = normalized_done = False

    if resample:
        try:
            new = resample_if_needed(
                current, target_sr, output_dir / f"{stem}_resampled.wav"
            )
            resampled = new != current
            current = new
        except Exception as exc:
            logger.warning("Resampling skipped (%s).", exc)

    if remove_dc:
        try:
            current = remove_dc_offset(current, output_dir / f"{stem}_dc.wav")
            dc_removed = True
        except Exception as exc:
            logger.warning("DC offset removal skipped (%s).", exc)

    if normalize:
        try:
            current = normalize_loudness(
                current, target_lufs, output_dir / f"{stem}_norm.wav"
            )
            normalized_done = True
        except Exception as exc:
            logger.warning("Loudness normalization skipped (%s).", exc)

    return PreprocessResult(
        output_path=current,
        original_sr=original_sr,
        resampled=resampled,
        dc_removed=dc_removed,
        loudness_normalized=normalized_done,
        original_lufs=round(original_lufs, 2),
        target_lufs=target_lufs,
    )


def reduce_reverb(
    vocals_path: Path, output_path: Path | None = None, aggressiveness: float = 0.3
) -> Path:
    """Reduce residual reverb through spectral gating with noisereduce.

    `aggressiveness` is clamped to the range 0.0 to 0.6; higher values risk
    metallic artifacts, and a warning is emitted when the requested value is
    above that ceiling.

    Args:
        vocals_path: Isolated vocals file.
        output_path: Destination file. Defaults to a sibling `_dereverb.wav`.
        aggressiveness: Noise reduction strength.

    Returns:
        The path of the processed file.
    """
    import noisereduce as nr

    if aggressiveness > _MAX_REVERB_AGGRESSIVENESS:
        logger.warning(
            "reverb_aggressiveness=%.2f is above %.1f and was clamped to %.1f "
            "to avoid metallic artifacts.",
            aggressiveness,
            _MAX_REVERB_AGGRESSIVENESS,
            _MAX_REVERB_AGGRESSIVENESS,
        )
    agg = max(0.0, min(aggressiveness, _MAX_REVERB_AGGRESSIVENESS))

    vocals_path = Path(vocals_path)
    data, sr = _load(vocals_path)
    multichannel = getattr(data, "ndim", 1) == 2
    y = data.T if multichannel else data  # noisereduce expects (channels, frames).
    reduced = nr.reduce_noise(y=y, sr=sr, prop_decrease=agg, stationary=False)
    if multichannel:
        reduced = reduced.T
    reduced = _prevent_clip(reduced).astype("float32")
    out = output_path or _default_out(vocals_path, "_dereverb")
    return _save(out, reduced, sr)


def enhance_vocal_clarity(vocals_path: Path, output_path: Path | None = None) -> Path:
    """Apply a light EQ: 80 Hz high pass plus a 2 to 4 kHz presence boost.

    The presence band sharpens consonants and makes the vocals easier to
    transcribe.

    Args:
        vocals_path: Isolated vocals file.
        output_path: Destination file. Defaults to a sibling `_clarity.wav`.

    Returns:
        The path of the processed file.
    """
    from scipy.signal import butter, sosfilt

    vocals_path = Path(vocals_path)
    data, sr = _load(vocals_path)

    high_pass = butter(2, 80.0, btype="highpass", fs=sr, output="sos")
    cleaned = sosfilt(high_pass, data, axis=0)

    nyquist = sr / 2.0
    high = min(4000.0, nyquist * 0.99)
    band = butter(2, [2000.0, high], btype="bandpass", fs=sr, output="sos")
    presence = sosfilt(band, cleaned, axis=0)
    enhanced = (cleaned + 0.3 * presence).astype("float32")
    enhanced = _prevent_clip(enhanced).astype("float32")

    out = output_path or _default_out(vocals_path, "_clarity")
    return _save(out, enhanced, sr)


def postprocess_vocals(
    vocals_path: Path,
    output_dir: Path,
    *,
    reduce_reverb_: bool = True,
    reverb_aggressiveness: float = 0.3,
    enhance: bool = True,
) -> PostprocessResult:
    """Chain reverb reduction and clarity enhancement.

    Every step is guarded: a missing dependency does not interrupt the pipeline.

    Args:
        vocals_path: Isolated vocals file.
        output_dir: Directory receiving the intermediate files.
        reduce_reverb_: Whether to run reverb reduction.
        reverb_aggressiveness: Reverb reduction strength, clamped to 0.0 to 0.6.
        enhance: Whether to run the clarity enhancement.

    Returns:
        A `PostprocessResult` describing what was actually applied.
    """
    vocals_path = Path(vocals_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = vocals_path.stem

    current = vocals_path
    reverb_reduced = clarity_enhanced = False

    if reduce_reverb_:
        try:
            current = reduce_reverb(
                current,
                output_dir / f"{stem}_dereverb.wav",
                aggressiveness=reverb_aggressiveness,
            )
            reverb_reduced = True
        except Exception as exc:
            logger.warning("Reverb reduction skipped (%s).", exc)

    if enhance:
        try:
            current = enhance_vocal_clarity(current, output_dir / f"{stem}_clarity.wav")
            clarity_enhanced = True
        except Exception as exc:
            logger.warning("Clarity enhancement skipped (%s).", exc)

    return PostprocessResult(
        output_path=current,
        reverb_reduced=reverb_reduced,
        clarity_enhanced=clarity_enhanced,
    )
