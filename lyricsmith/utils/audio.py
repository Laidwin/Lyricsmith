"""Audio file helpers: validation, conversion and metadata.

Heavy dependencies (`soundfile`, `librosa`) are imported lazily so that
importing this module stays cheap. It depends on no other project module
besides the exceptions and the logger.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ..exceptions import AudioFormatError
from .logger import get_logger

logger = get_logger("utils.audio")


def validate_audio_file(path: Path, supported_formats: Sequence[str]) -> None:
    """Check that the file exists and that its extension is supported.

    Args:
        path: Audio file to validate.
        supported_formats: Accepted extensions, lowercased, dot included.

    Raises:
        AudioFormatError: When the file is missing, is not a regular file, or
            has an extension outside `supported_formats`.
    """
    path = Path(path)
    if not path.exists():
        raise AudioFormatError(f"Audio file not found: {path}")
    if not path.is_file():
        raise AudioFormatError(f"Path is not a file: {path}")

    if path.suffix.lower() not in supported_formats:
        raise AudioFormatError(
            f"Unsupported format: '{path.suffix}'. "
            f"Accepted formats: {', '.join(supported_formats)}"
        )


def convert_to_wav(
    input_path: Path,
    output_path: Path,
    *,
    sample_rate: int = 16000,
    mono: bool = True,
) -> Path:
    """Convert a supported audio file to WAV, 16 kHz mono by default.

    This is the input format expected by HeartTranscriptor. Decoding relies on
    `librosa`, which handles mp3, m4a and ogg through audioread or ffmpeg, and
    writing relies on `soundfile`.

    Args:
        input_path: Source file.
        output_path: Destination `.wav` file.
        sample_rate: Target sample rate.
        mono: Whether to downmix to a single channel.

    Returns:
        The path of the written WAV file.
    """
    import librosa
    import soundfile as sf

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.debug(
        "Converting %s to %s (%d Hz, mono=%s)", input_path, output_path, sample_rate, mono
    )
    waveform, _ = librosa.load(str(input_path), sr=sample_rate, mono=mono)
    sf.write(str(output_path), waveform, sample_rate, subtype="PCM_16")
    return output_path


def get_audio_duration(path: Path) -> float:
    """Return the duration of an audio file in seconds.

    Uses `soundfile` for native formats, which reads the header only, and falls
    back to `librosa` for compressed formats.
    """
    path = Path(path)
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return float(info.frames) / float(info.samplerate)
    except (RuntimeError, ImportError):
        import librosa

        return float(librosa.get_duration(path=str(path)))
