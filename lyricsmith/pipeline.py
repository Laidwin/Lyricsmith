"""Orchestrator chaining vocal separation, transcription and refinement.

This is the use case layer: it sequences the steps, writes the results and
reports progress through a callback. It renders nothing itself, so the same
code serves the CLI, a library caller or a server.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import Config
from .domain import TranscriptionResult
from .exceptions import PipelineError
from .refine import refine_transcription
from .report import (
    build_metadata,
    build_pipeline_info,
    build_pipeline_report,
    format_annotated_lyrics,
)
from .separator import VocalSeparator
from .transcriptor import LyricsTranscriptor
from .utils.audio import convert_to_wav, validate_audio_file
from .utils.logger import get_logger

logger = get_logger("pipeline")

SEPARATION_PHASE = "separation"
TRANSCRIPTION_PHASE = "transcription"


@dataclass
class BatchProgress:
    """Progress of a batch run, handed to the caller's progress callback."""

    phase: str
    completed: int
    total: int
    current: str = ""


ProgressCallback = Callable[[BatchProgress], None]


@dataclass
class PipelineResult:
    """Full outcome of a pipeline run on a single file."""

    input_path: Path
    output_dir: Path
    lyrics_txt_path: Path
    lyrics_json_path: Path
    metadata_path: Path
    transcription: TranscriptionResult
    lyrics_annotated_path: Path | None = None
    vocals_path: Path | None = None
    no_vocals_path: Path | None = None
    duration_seconds: float = 0.0
    total_time_seconds: float = 0.0
    separation_time_seconds: float = 0.0
    transcription_time_seconds: float = 0.0


@dataclass
class _FileJob:
    """Internal state carried from the separation phase to the transcription phase.

    In batch mode every file is separated while Demucs is loaded, then Demucs is
    unloaded and every file is transcribed with HeartTranscriptor. The stems stay
    on disk, inside ``tmp_dir``, between the two passes.
    """

    audio_path: Path
    out_dir: Path
    tmp_dir: Path
    separate: bool
    voice_source: Path
    sep_result: Any = None  # SeparationResult or None, used by the report.
    vocals_path: Path | None = None
    no_vocals_path: Path | None = None
    sep_time: float = 0.0
    duration: float = 0.0


class LyricsPipeline:
    """End to end pipeline turning an audio file into timestamped lyrics."""

    def __init__(self, config: Config) -> None:
        """Build the separator and the transcriptor from the configuration.

        Args:
            config: Loaded project configuration.
        """
        self.config = config
        self.settings = config.pipeline
        self.output_root: Path = config.resolve_path(self.settings.output_dir)
        self.separator = VocalSeparator(config)
        self.transcriptor = LyricsTranscriptor(config)

    def run(self, audio_path: Path, *, separate: bool = True) -> PipelineResult:
        """Run the full pipeline on a single file.

        The steps are validation, optional separation, conversion to 16 kHz WAV,
        transcription, refinement, saving and temporary file cleanup.

        Args:
            audio_path: Input audio file.
            separate: When ``False``, Demucs is skipped because the track is
                already an a cappella.

        Returns:
            A complete `PipelineResult`.

        Raises:
            PipelineError: When validation, separation or transcription fails.
        """
        job = self._prepare_job(audio_path, separate)
        try:
            self._separate_phase(job)
            if separate:
                # Free Demucs from VRAM before loading HeartTranscriptor: the two
                # models must not coexist on an 8 GB GPU.
                self.separator.release()
            return self._transcribe_phase(job)
        finally:
            shutil.rmtree(job.tmp_dir, ignore_errors=True)

    def separate_only(self, audio_path: Path) -> Path:
        """Separate the vocals without transcribing and keep the stems.

        Args:
            audio_path: Input audio file.

        Returns:
            The path of the isolated vocals file (`*_vocals.wav`).

        Raises:
            PipelineError: When validation or separation fails.
        """
        audio_path = Path(audio_path).resolve()
        validate_audio_file(audio_path, self.settings.supported_formats)

        out_dir = self.output_root / audio_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = self.separator.separate(audio_path, out_dir)
        finally:
            self.separator.release()
        logger.info("Isolated vocals written: %s", result.vocals_path)
        return result.vocals_path

    def run_batch(
        self,
        audio_paths: Sequence[Path],
        *,
        separate: bool = True,
        on_progress: ProgressCallback | None = None,
    ) -> list[PipelineResult]:
        """Process several files in two passes.

        Unlike calling `run()` in a loop, each model is loaded once for the whole
        set of files:

        1. Separation pass: Demucs is loaded once and separates every file, with
           the stems kept on disk, then Demucs is unloaded from VRAM.
        2. Transcription pass: HeartTranscriptor transcribes every separated file.

        This avoids loading and unloading each model once per file while
        guaranteeing the two never share VRAM, which matters on an 8 GB GPU.
        Failing files are logged without interrupting the batch.

        Args:
            audio_paths: Audio files to process.
            separate: When ``False``, Demucs is skipped for every file.
            on_progress: Called before each file and once at the end of each
                phase, so a caller can render progress however it wants.

        Returns:
            One `PipelineResult` per successfully processed file.
        """
        jobs = self._separate_batch(audio_paths, separate, on_progress)

        if separate:
            self.separator.release()

        results = self._transcribe_batch(jobs, on_progress)
        logger.info(
            "Batch done: %d/%d files transcribed.", len(results), len(audio_paths)
        )
        return results

    def _separate_batch(
        self,
        audio_paths: Sequence[Path],
        separate: bool,
        on_progress: ProgressCallback | None,
    ) -> list[_FileJob]:
        """Run the separation pass over every file, skipping the failures."""
        jobs: list[_FileJob] = []
        total = len(audio_paths)
        for index, path in enumerate(audio_paths):
            _report(on_progress, SEPARATION_PHASE, index, total, Path(path).name)
            job: _FileJob | None = None
            try:
                job = self._prepare_job(path, separate)
                self._separate_phase(job)
                jobs.append(job)
            except PipelineError as exc:
                logger.error("Separation failed on '%s': %s", Path(path).name, exc)
                if job is not None:
                    shutil.rmtree(job.tmp_dir, ignore_errors=True)
        _report(on_progress, SEPARATION_PHASE, total, total)
        return jobs

    def _transcribe_batch(
        self, jobs: list[_FileJob], on_progress: ProgressCallback | None
    ) -> list[PipelineResult]:
        """Run the transcription pass over every prepared job."""
        results: list[PipelineResult] = []
        total = len(jobs)
        for index, job in enumerate(jobs):
            _report(on_progress, TRANSCRIPTION_PHASE, index, total, job.audio_path.name)
            try:
                results.append(self._transcribe_phase(job))
            except PipelineError as exc:
                logger.error(
                    "Transcription failed on '%s': %s", job.audio_path.name, exc
                )
            finally:
                shutil.rmtree(job.tmp_dir, ignore_errors=True)
        _report(on_progress, TRANSCRIPTION_PHASE, total, total)
        return results

    def _prepare_job(self, audio_path: Path, separate: bool) -> _FileJob:
        """Validate the input, then create the output and temporary directories."""
        audio_path = Path(audio_path).resolve()
        validate_audio_file(audio_path, self.settings.supported_formats)

        out_dir = self.output_root / audio_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(tempfile.mkdtemp(prefix="lyricsmith_"))
        return _FileJob(
            audio_path=audio_path,
            out_dir=out_dir,
            tmp_dir=tmp_dir,
            separate=separate,
            voice_source=audio_path,
        )

    def _separate_phase(self, job: _FileJob) -> None:
        """Run the vocal separation, or bypass it.

        Demucs is deliberately not unloaded here: the caller owns the model life
        cycle, and in batch mode it stays loaded for the whole set of files.
        """
        if not job.separate:
            logger.info("Separation disabled, transcribing the input directly.")
            job.voice_source = job.audio_path
            return

        stems_target = job.out_dir if self.settings.keep_stems else job.tmp_dir
        sep_result = self.separator.separate(job.audio_path, stems_target)
        job.sep_result = sep_result
        job.vocals_path = sep_result.vocals_path
        job.no_vocals_path = sep_result.no_vocals_path
        job.sep_time = sep_result.processing_time_seconds
        job.duration = sep_result.duration_seconds
        job.voice_source = sep_result.vocals_path

    def _transcribe_phase(self, job: _FileJob) -> PipelineResult:
        """Convert to 16 kHz WAV, transcribe, refine, then write every output."""
        song_name = job.audio_path.stem
        phase_start = time.perf_counter()

        wav_for_transcription = job.tmp_dir / f"{song_name}_16k.wav"
        convert_to_wav(
            job.voice_source,
            wav_for_transcription,
            sample_rate=self.settings.target_sample_rate,
            mono=True,
        )

        raw = self.transcriptor.transcribe(wav_for_transcription)
        transcription = refine_transcription(
            raw,
            cleaning=self.settings.cleaning,
            deduplication=self.settings.deduplication,
            line_splitting=self.settings.line_splitting,
            quality=self.settings.quality,
        )

        total_time = job.sep_time + (time.perf_counter() - phase_start)
        return self._write_outputs(job, transcription, total_time)

    def _write_outputs(
        self, job: _FileJob, transcription: TranscriptionResult, total_time: float
    ) -> PipelineResult:
        """Write the lyrics, the metadata and the reports of one file."""
        song_name = job.audio_path.stem
        out_dir = job.out_dir
        base = out_dir / f"{song_name}_lyrics"
        txt_path = base.with_suffix(".txt")
        json_path = base.with_suffix(".json")
        annotated_path = out_dir / f"{song_name}_lyrics_annotated.txt"
        metadata_path = out_dir / f"{song_name}_metadata.json"

        keeping_stems = self.settings.keep_stems and job.separate
        kept_vocals = job.vocals_path if keeping_stems else None
        kept_no_vocals = job.no_vocals_path if keeping_stems else None

        pipeline_info = build_pipeline_info(
            job.audio_path,
            self._describe_separation(job),
            getattr(job.sep_result, "preprocess", None),
            getattr(job.sep_result, "postprocess", None),
            reverb_aggressiveness=self.settings.postprocessing.reverb_aggressiveness,
            duration_seconds=job.duration,
            total_time_seconds=total_time,
        )
        metadata = build_metadata(
            job.audio_path,
            transcription,
            duration_seconds=job.duration,
            separation_used=job.separate,
            separation_time_seconds=job.sep_time,
            total_time_seconds=total_time,
            stems_kept=bool(kept_vocals),
        )

        txt_path.write_text(transcription.lyrics, encoding="utf-8")
        _write_json(json_path, transcription.to_dict())
        annotated_path.write_text(
            format_annotated_lyrics(transcription, pipeline_info), encoding="utf-8"
        )
        _write_json(metadata_path, metadata)

        if self.settings.report.json:
            _write_json(
                out_dir / f"{song_name}_pipeline_report.json",
                build_pipeline_report(
                    job.audio_path,
                    transcription,
                    metadata,
                    separation={
                        "model": self.separator.model_name,
                        "shifts": self.separator.shifts,
                        "quality_preset": self.separator.quality_preset,
                    },
                    preprocess=getattr(job.sep_result, "preprocess", None),
                    postprocess=getattr(job.sep_result, "postprocess", None),
                ),
            )

        logger.info("Pipeline done for '%s' in %.1fs", song_name, total_time)
        return PipelineResult(
            input_path=job.audio_path,
            output_dir=out_dir,
            lyrics_txt_path=txt_path,
            lyrics_json_path=json_path,
            metadata_path=metadata_path,
            transcription=transcription,
            lyrics_annotated_path=annotated_path,
            vocals_path=kept_vocals,
            no_vocals_path=kept_no_vocals,
            duration_seconds=job.duration,
            total_time_seconds=total_time,
            separation_time_seconds=job.sep_time,
            transcription_time_seconds=transcription.processing_time_seconds,
        )

    def _describe_separation(self, job: _FileJob) -> str:
        """Describe the separation settings used for one file."""
        if not job.separate:
            return "disabled"
        return f"{self.separator.model_name}, shifts={self.separator.shifts}"

    @staticmethod
    def result_summary(result: PipelineResult) -> dict[str, object]:
        """Return a serializable summary of a `PipelineResult`."""
        data = asdict(result)
        data.pop("transcription", None)
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in data.items()}


def _report(
    callback: ProgressCallback | None,
    phase: str,
    completed: int,
    total: int,
    current: str = "",
) -> None:
    """Send a progress event when the caller asked for one."""
    if callback is not None:
        callback(BatchProgress(phase=phase, completed=completed, total=total, current=current))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON document, serializing paths as strings."""
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
