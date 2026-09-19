"""Vocal separation built on Demucs, through its native Python API.

No subprocess is used. Two code paths are supported depending on the installed
Demucs build:

- the fast path, `demucs.api.Separator`, available in Demucs 4.0 and later when
  the `api` module ships with the distribution;
- the low level path, `demucs.pretrained` plus `demucs.apply` and `demucs.audio`,
  present in every Demucs 4.x build.

Both paths split the track into isolated vocals (`vocals`) and accompaniment
(`no_vocals`, the sum of the remaining sources).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from .config import Config, DemucsConfig, PostprocessingConfig, PreprocessingConfig
from .exceptions import SeparationError
from .utils.audio import get_audio_duration
from .utils.gpu import resolve_device
from .utils.logger import get_logger

logger = get_logger("separator")


@dataclass
class SeparationResult:
    """Outcome of a Demucs vocal separation."""

    vocals_path: Path
    no_vocals_path: Path
    duration_seconds: float
    processing_time_seconds: float
    preprocess: Any = None  # PreprocessResult or None
    postprocess: Any = None  # PostprocessResult or None


class VocalSeparator:
    """Separate vocals from the instrumental track with Demucs."""

    # Speed and quality presets. When selected through `quality_preset`, they
    # take precedence over the `name` and `shifts` configuration keys.
    QUALITY_PRESETS: ClassVar[dict[str, dict[str, Any]]] = {
        "fast": {"model": "htdemucs", "shifts": 1},
        "balanced": {"model": "htdemucs_ft", "shifts": 2},
        "best": {"model": "htdemucs_ft", "shifts": 4},
    }

    def __init__(self, config: Config) -> None:
        """Read the Demucs settings from the configuration.

        Args:
            config: Loaded project configuration.

        Raises:
            SeparationError: When `quality_preset` names an unknown preset.
        """
        self.config = config
        demucs_cfg: DemucsConfig = config.models.demucs

        preset = demucs_cfg.quality_preset
        if preset:
            if preset not in self.QUALITY_PRESETS:
                raise SeparationError(
                    f"Unknown quality preset: '{preset}'. "
                    f"Available presets: {', '.join(self.QUALITY_PRESETS)}."
                )
            self.quality_preset: str | None = preset
            self.model_name: str = self.QUALITY_PRESETS[preset]["model"]
            self.shifts: int = int(self.QUALITY_PRESETS[preset]["shifts"])
        else:
            self.quality_preset = None
            self.model_name = demucs_cfg.name
            self.shifts = demucs_cfg.shifts

        self.overlap: float = demucs_cfg.overlap
        self.jobs: int = demucs_cfg.jobs
        self.device: str = resolve_device(demucs_cfg.device)
        self._engine: Any = None
        self._use_api: bool | None = None

        self.pre_cfg: PreprocessingConfig = config.pipeline.preprocessing
        self.post_cfg: PostprocessingConfig = config.pipeline.postprocessing

    @staticmethod
    def _has_api() -> bool:
        """Tell whether the convenient `demucs.api` module is importable."""
        try:
            import demucs.api  # noqa: F401

            return True
        except ImportError:
            return False

    def _load_engine(self) -> Any:
        """Load the separation engine once and cache it.

        Returns:
            The `demucs.api.Separator` instance or the low level model.

        Raises:
            SeparationError: When Demucs is missing or cannot be loaded.
        """
        if self._engine is not None:
            return self._engine
        try:
            self._use_api = self._has_api()
            if self._use_api:
                self._engine = self._build_api_separator()
            else:
                self._engine = self._build_lowlevel_model()
        except SeparationError:
            raise
        except ImportError as exc:
            raise SeparationError(
                "The 'demucs' package is not installed. "
                "Install the dependencies with: uv pip install -r requirements.txt"
            ) from exc
        except Exception as exc:
            raise SeparationError(f"Could not load Demucs: {exc}") from exc
        return self._engine

    def _build_api_separator(self) -> Any:
        """Instantiate `demucs.api.Separator`, the fast path."""
        from demucs.api import Separator

        logger.info("Loading Demucs '%s' (api) on %s", self.model_name, self.device)
        return Separator(
            model=self.model_name,
            device=self.device,
            shifts=self.shifts,
            overlap=self.overlap,
            jobs=self.jobs,
        )

    def _build_lowlevel_model(self) -> Any:
        """Load the model through the low level Demucs API."""
        from demucs.pretrained import get_model

        logger.info("Loading Demucs '%s' (low level) on %s", self.model_name, self.device)
        model = get_model(self.model_name)
        model.to(self.device)
        model.eval()
        return model

    def separate(self, audio_path: Path, output_dir: Path) -> SeparationResult:
        """Run Demucs and write the vocals and accompaniment stems.

        Args:
            audio_path: Source audio file.
            output_dir: Directory receiving `*_vocals.wav` and `*_no_vocals.wav`.

        Returns:
            A `SeparationResult` holding the stem paths and timing metrics.

        Raises:
            SeparationError: On CUDA out of memory or any Demucs failure.
        """
        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = audio_path.stem
        duration = get_audio_duration(audio_path)

        demucs_input = audio_path
        preprocess_info = None
        if self.pre_cfg.enabled:
            preprocess_info = self._run_preprocess(audio_path, output_dir)
            demucs_input = preprocess_info.output_path

        engine = self._load_engine()
        vocals_path = output_dir / f"{stem}_vocals.wav"
        no_vocals_path = output_dir / f"{stem}_no_vocals.wav"

        logger.info("Separating vocals from: %s", audio_path.name)
        start = time.perf_counter()
        try:
            if self._use_api:
                self._separate_with_api(engine, demucs_input, vocals_path, no_vocals_path)
            else:
                self._separate_lowlevel(engine, demucs_input, vocals_path, no_vocals_path)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                self._free_cuda()
                raise SeparationError(
                    "Not enough GPU memory (CUDA out of memory) during separation. "
                    "Lower 'shifts' in config.yaml, switch to 'device: cpu', "
                    "or free some VRAM."
                ) from exc
            raise SeparationError(f"Demucs separation failed: {exc}") from exc
        except SeparationError:
            raise
        except Exception as exc:
            raise SeparationError(f"Demucs separation failed: {exc}") from exc

        elapsed = time.perf_counter() - start

        # Non destructive: the raw `*_vocals.wav` stem is kept and the processed
        # version is written next to it, then used as the transcription input.
        postprocess_info = None
        if self.post_cfg.enabled:
            postprocess_info = self._run_postprocess(vocals_path, output_dir)
            if postprocess_info is not None:
                vocals_path = postprocess_info.output_path

        logger.info("Separation done in %.1fs (audio: %.1fs)", elapsed, duration)
        return SeparationResult(
            vocals_path=vocals_path,
            no_vocals_path=no_vocals_path,
            duration_seconds=duration,
            processing_time_seconds=elapsed,
            preprocess=preprocess_info,
            postprocess=postprocess_info,
        )

    def _run_preprocess(self, audio_path: Path, output_dir: Path) -> Any:
        """Apply the configured audio pre processing before separation."""
        from .utils.preprocess import preprocess_audio

        logger.info("Pre processing audio: %s", audio_path.name)
        return preprocess_audio(
            audio_path,
            output_dir,
            resample=self.pre_cfg.resample,
            target_sr=self.pre_cfg.resample_target_sr,
            remove_dc=self.pre_cfg.remove_dc_offset,
            normalize=self.pre_cfg.normalize_loudness,
            target_lufs=self.pre_cfg.target_lufs,
        )

    def _run_postprocess(self, vocals_path: Path, output_dir: Path) -> Any:
        """Apply the configured vocal post processing after separation."""
        from .utils.preprocess import postprocess_vocals

        logger.info("Post processing the isolated vocals.")
        return postprocess_vocals(
            vocals_path,
            output_dir,
            reduce_reverb_=self.post_cfg.reduce_reverb,
            reverb_aggressiveness=self.post_cfg.reverb_aggressiveness,
            enhance=self.post_cfg.enhance_clarity,
        )

    def _separate_with_api(
        self, separator: Any, audio_path: Path, vocals_path: Path, no_vocals_path: Path
    ) -> None:
        """Separate through `demucs.api` and write both stems."""
        from demucs.api import save_audio

        _origin, stems = separator.separate_audio_file(str(audio_path))
        vocals = stems["vocals"]
        no_vocals = sum(t for name, t in stems.items() if name != "vocals")
        save_audio(vocals, str(vocals_path), samplerate=separator.samplerate)
        save_audio(no_vocals, str(no_vocals_path), samplerate=separator.samplerate)

    def _separate_lowlevel(
        self, model: Any, audio_path: Path, vocals_path: Path, no_vocals_path: Path
    ) -> None:
        """Separate through the low level API and write both stems.

        Raises:
            SeparationError: When the model does not produce a `vocals` source.
        """
        import torch
        from demucs.apply import apply_model
        from demucs.audio import AudioFile, save_audio

        wav = AudioFile(str(audio_path)).read(
            streams=0, samplerate=model.samplerate, channels=model.audio_channels
        )
        # Demucs recommends normalizing the input, then denormalizing the stems.
        ref = wav.mean(0)
        std = ref.std()
        if float(std) == 0.0:
            std = torch.tensor(1.0)  # Guard against a fully silent track.
        wav = (wav - ref.mean()) / std

        with torch.no_grad():
            sources = apply_model(
                model,
                wav[None],
                device=self.device,
                shifts=self.shifts,
                overlap=self.overlap,
                progress=True,
                num_workers=self.jobs,
            )[0]
        sources = sources * std + ref.mean()

        names = list(model.sources)
        if "vocals" not in names:
            raise SeparationError(
                f"The Demucs model '{self.model_name}' does not produce a "
                f"'vocals' stem (sources: {names})."
            )
        idx = names.index("vocals")
        vocals = sources[idx]
        no_vocals = torch.stack(
            [sources[i] for i in range(len(names)) if i != idx]
        ).sum(0)

        save_audio(vocals, str(vocals_path), model.samplerate)
        save_audio(no_vocals, str(no_vocals_path), model.samplerate)

    def release(self) -> None:
        """Unload the Demucs model and empty the CUDA cache.

        This is required on an 8 GB GPU: it prevents Demucs and
        HeartTranscriptor from living in VRAM at the same time, which can crash
        the CUDA runtime. The model is reloaded lazily on the next `separate`.
        """
        import gc

        self._engine = None
        self._use_api = None
        gc.collect()
        self._free_cuda()

    @staticmethod
    def _free_cuda() -> None:
        """Try to free the CUDA cache, typically after an out of memory error."""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
