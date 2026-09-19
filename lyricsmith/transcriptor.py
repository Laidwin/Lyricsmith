"""Adapter over the HeartTranscriptor (heartlib) speech recognition engine.

Loads the model from the local checkpoint, runs inference on isolated vocals
and normalizes the engine output into a `RawTranscription`. Honouring
`lazy_load` keeps the pipeline within 8 GB of VRAM by loading the model right
before transcription and releasing it right after.

This module owns no pipeline policy: cleaning, deduplication, quality scoring
and verse reflow are applied afterwards by `lyricsmith.refine`.

Implementation note:
    The public API of `heartlib` keeps evolving, so the concrete inference call
    is isolated in `_run_inference`. If the real signature differs, that is the
    only place to adapt; the rest of the module is engine agnostic.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, ClassVar

from .config import Config, TranscriptorConfig
from .domain import RawTranscription, Segment
from .exceptions import ModelNotFoundError, TranscriptionError
from .utils.gpu import resolve_device
from .utils.logger import get_logger

logger = get_logger("transcriptor")


def _extract_segment_fields(seg: Any) -> tuple[float, float, str]:
    """Extract ``(start, end, text)`` from a heterogeneous segment.

    Accepts a transformers ASR chunk (``{"timestamp": (start, end), "text"}``)
    as well as a dict or object exposing ``start``, ``end`` and ``text``.
    Missing bounds fall back to ``0.0``.

    Args:
        seg: Raw segment returned by the ASR engine.

    Returns:
        The normalized start time, end time and stripped text.
    """

    def _get(obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    text = str(_get(seg, "text", "") or "").strip()
    timestamp = _get(seg, "timestamp")
    if timestamp is not None:
        start = timestamp[0] if len(timestamp) > 0 else None
        end = timestamp[1] if len(timestamp) > 1 else None
    else:
        start = _get(seg, "start")
        end = _get(seg, "end")

    return (
        float(start) if start is not None else 0.0,
        float(end) if end is not None else 0.0,
        text,
    )


class LyricsTranscriptor:
    """Transcribe lyrics from an audio file using HeartTranscriptor."""

    # Mapping from the configured dtype name to a torch dtype attribute.
    _DTYPE_MAP: ClassVar[dict[str, str]] = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
    }

    def __init__(self, config: Config) -> None:
        """Read the transcription settings from the configuration.

        Args:
            config: Loaded project configuration.
        """
        self.settings: TranscriptorConfig = config.models.hearttranscriptor
        self.model_path: Path = config.resolve_path(self.settings.model_path)
        self.device: str = resolve_device(self.settings.device)
        self.lazy_load: bool = self.settings.lazy_load
        self._model: Any = None

    def _ensure_checkpoint(self) -> None:
        """Check that the checkpoint directory exists on disk.

        Raises:
            ModelNotFoundError: When the checkpoint is missing.
        """
        if not self.model_path.exists():
            raise ModelNotFoundError(
                f"HeartTranscriptor checkpoint not found: {self.model_path}. "
                "Run first: python scripts/download_models.py"
            )

    def _resolve_torch_dtype(self) -> Any:
        """Convert the configured dtype name into a `torch.dtype`."""
        import torch

        name = self._DTYPE_MAP.get(self.settings.dtype, self.settings.dtype)
        return getattr(torch, name, torch.float32)

    def _resolve_checkpoint_dir(self) -> Path:
        """Locate the directory holding the Whisper checkpoint (`config.json`).

        Both possible layouts are handled: weights directly inside
        ``model_path``, which is what download_models.py produces, and the
        heartlib convention of a ``HeartTranscriptor-oss`` subdirectory.

        Returns:
            The directory containing the checkpoint.

        Raises:
            ModelNotFoundError: When no checkpoint is found in either layout.
        """
        if (self.model_path / "config.json").exists():
            return self.model_path
        sub = self.model_path / "HeartTranscriptor-oss"
        if (sub / "config.json").exists():
            return sub
        raise ModelNotFoundError(
            f"No Whisper checkpoint (config.json) in {self.model_path} "
            "nor in its 'HeartTranscriptor-oss' subdirectory. "
            "Run again: python scripts/download_models.py"
        )

    @staticmethod
    def _load_pipeline_class() -> Any:
        """Load `HeartTranscriptorPipeline` without running `heartlib/__init__`.

        The heartlib package initializer first imports the music generation
        pipeline (torchtune and native kernels), which triggers a native crash
        (access violation) on Windows. The transcription module has no relative
        import, so it is loaded straight from its file through `importlib`.

        Returns:
            The `HeartTranscriptorPipeline` class.

        Raises:
            TranscriptionError: When heartlib or its transcription module is
                missing or cannot be executed.
        """
        import importlib.util

        spec = importlib.util.find_spec("heartlib")
        if spec is None or not spec.submodule_search_locations:
            raise TranscriptionError(
                "The 'heartlib' package is not installed. "
                "Run: python scripts/download_models.py"
            )
        pkg_dir = Path(next(iter(spec.submodule_search_locations)))
        lyrics_file = pkg_dir / "pipelines" / "lyrics_transcription.py"
        if not lyrics_file.exists():
            raise TranscriptionError(
                f"heartlib transcription module not found: {lyrics_file}"
            )
        mod_spec = importlib.util.spec_from_file_location(
            "heartlib_lyrics_transcription", lyrics_file
        )
        if mod_spec is None or mod_spec.loader is None:
            raise TranscriptionError(f"Could not load the heartlib module: {lyrics_file}")
        module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(module)
        return module.HeartTranscriptorPipeline

    def _load_model(self) -> Any:
        """Load the HeartTranscriptor pipeline, a fine tuned Whisper model.

        Returns:
            The instantiated pipeline.

        Raises:
            TranscriptionError: When the weights cannot be loaded.
        """
        self._ensure_checkpoint()
        import torch
        from transformers import (
            WhisperForConditionalGeneration,
            WhisperProcessor,
        )

        pipeline_cls = self._load_pipeline_class()
        ckpt_dir = self._resolve_checkpoint_dir()
        torch_dtype = self._resolve_torch_dtype()
        device = torch.device(self.device)

        logger.info(
            "Loading HeartTranscriptor (%s, dtype=%s) on %s",
            ckpt_dir,
            self.settings.dtype,
            self.device,
        )
        try:
            model = WhisperForConditionalGeneration.from_pretrained(
                str(ckpt_dir), dtype=torch_dtype, low_cpu_mem_usage=True
            )
            processor = WhisperProcessor.from_pretrained(str(ckpt_dir))
            return pipeline_cls(
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                device=device,
                dtype=torch_dtype,
                chunk_length_s=self.settings.chunk_length_s,
                batch_size=self.settings.batch_size,
            )
        except Exception as exc:
            raise TranscriptionError(f"Could not load the model: {exc}") from exc

    def _get_model(self) -> Any:
        """Return the model, honouring the `lazy_load` policy."""
        if self.lazy_load:
            return self._load_model()
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _release_model(self) -> None:
        """Drop the cached model and hand its VRAM back to the driver.

        The caller must clear its own reference to the model first, otherwise
        the weights stay reachable and emptying the CUDA cache frees nothing.
        """
        if not self.lazy_load:
            return
        import gc

        self._model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _build_generate_kwargs(self) -> dict[str, Any]:
        """Build the anti hallucination `generate_kwargs` from the configuration.

        Configuration options are mapped to the names expected by the Whisper
        generation code in `transformers`. Options left unset in `config.yaml`
        are omitted so that the model defaults apply. The openai-whisper option
        `best_of` has no direct equivalent there and is deliberately omitted.

        Returns:
            The keyword arguments to forward to the generation call.
        """
        settings = self.settings
        gk: dict[str, Any] = {}
        if settings.condition_on_previous_text is not None:
            gk["condition_on_prev_tokens"] = settings.condition_on_previous_text
        if settings.no_speech_threshold is not None:
            gk["no_speech_threshold"] = settings.no_speech_threshold
        if settings.logprob_threshold is not None:
            # Required: the Whisper fallback in transformers reads `logprobs` in
            # the no_speech branch, and that value only exists when
            # logprob_threshold is set, otherwise it raises UnboundLocalError.
            gk["logprob_threshold"] = settings.logprob_threshold
        if settings.compression_ratio_threshold is not None:
            gk["compression_ratio_threshold"] = settings.compression_ratio_threshold
        if settings.beam_size is not None:
            gk["num_beams"] = settings.beam_size
        if settings.temperature is not None:
            temp = settings.temperature
            gk["temperature"] = tuple(temp) if isinstance(temp, list) else float(temp)
        return gk

    def _run_inference(self, model: Any, audio_path: Path) -> RawTranscription:
        """Run ASR inference and normalize the output.

        The 16 kHz mono WAV is read into memory and passed as an array so that
        the pipeline never depends on ffmpeg.

        Args:
            model: Loaded ASR pipeline.
            audio_path: Audio file to transcribe.

        Returns:
            The normalized inference output, without timing information.
        """
        import soundfile as sf

        audio, sample_rate = sf.read(str(audio_path), dtype="float32")
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)

        payload = {"raw": audio, "sampling_rate": int(sample_rate)}
        generate_kwargs = self._build_generate_kwargs()
        try:
            raw = model(payload, return_timestamps=True, generate_kwargs=generate_kwargs)
        except (TypeError, ValueError, UnboundLocalError) as exc:
            # Some generate_kwargs combinations are rejected by older or newer
            # transformers versions, so the call is retried without them.
            logger.warning(
                "Unsupported generate_kwargs (%s), falling back to the model defaults.",
                exc,
            )
            raw = model(payload, return_timestamps=True)
        return self._normalize_output(raw)

    @staticmethod
    def _normalize_output(raw: Any) -> RawTranscription:
        """Normalize a raw engine output into a `RawTranscription`.

        Handles the output of a transformers ASR pipeline (``{"text", "chunks"}``
        where a chunk is ``{"timestamp": (start, end), "text"}``), a plain
        string, and variants exposing ``segments``, ``start`` and ``end``.
        Segments holding no text are discarded.

        Args:
            raw: Raw value returned by the ASR pipeline.

        Returns:
            The normalized transcription.
        """
        if isinstance(raw, str):
            return RawTranscription(lyrics=raw.strip())

        if not isinstance(raw, dict):
            raw = {
                "text": getattr(raw, "text", "") or getattr(raw, "lyrics", ""),
                "language": getattr(raw, "language", "unknown"),
                "chunks": getattr(raw, "chunks", None),
                "segments": getattr(raw, "segments", None),
            }

        segments: list[Segment] = []
        for seg in raw.get("chunks") or raw.get("segments") or []:
            start, end, text = _extract_segment_fields(seg)
            if text:
                segments.append(Segment(start=start, end=end, text=text))

        if segments:
            # One utterance per line reads better than the single Whisper block.
            lyrics = "\n".join(seg.text for seg in segments)
        else:
            lyrics = (raw.get("lyrics") or raw.get("text") or "").strip()

        return RawTranscription(
            lyrics=lyrics,
            segments=segments,
            language=raw.get("language", "unknown"),
        )

    def transcribe(self, audio_path: Path) -> RawTranscription:
        """Transcribe the lyrics of an audio file, ideally isolated vocals.

        Args:
            audio_path: Audio file to transcribe.

        Returns:
            The raw transcription. Text post processing is the caller's job,
            see `lyricsmith.refine.refine_transcription`.

        Raises:
            ModelNotFoundError: When the checkpoint is missing.
            TranscriptionError: When inference fails.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise TranscriptionError(f"Audio file not found: {audio_path}")

        model = self._get_model()
        logger.info("Transcribing: %s", audio_path.name)
        start = time.perf_counter()
        try:
            raw = self._run_inference(model, audio_path)
        except TranscriptionError:
            raise
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise TranscriptionError(
                    "Not enough GPU memory (CUDA out of memory) during transcription. "
                    "Enable 'lazy_load: true' or switch dtype to 'fp16' or 'bf16'."
                ) from exc
            raise TranscriptionError(f"Transcription failed: {exc}") from exc
        except Exception as exc:
            raise TranscriptionError(f"Transcription failed: {exc}") from exc
        finally:
            # Drop the local reference before releasing, so that the weights
            # become unreachable and emptying the CUDA cache actually frees VRAM.
            model = None
            self._release_model()

        raw.processing_time_seconds = time.perf_counter() - start
        logger.info(
            "Transcription done in %.1fs (%d raw segment(s))",
            raw.processing_time_seconds,
            len(raw.segments),
        )
        return raw
