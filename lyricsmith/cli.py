"""Command line interface of Lyricsmith.

Subcommands:
    transcribe   Transcribe a single audio file.
    batch        Transcribe every audio file of a directory.
    check-env    Check the GPU, CUDA, VRAM and model availability.

Examples:
    lyricsmith transcribe ./my_song.mp3
    lyricsmith batch ./my_songs/
    lyricsmith check-env

This module is the composition root: it loads the configuration, applies the
command line overrides and renders the results. Everything visual, including
the progress bars, lives here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import typer
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from .config import Config
from .domain import TranscriptionResult
from .exceptions import PipelineError
from .pipeline import (
    SEPARATION_PHASE,
    TRANSCRIPTION_PHASE,
    BatchProgress,
    LyricsPipeline,
)
from .utils.gpu import check_gpu_status
from .utils.logger import console, set_level

app = typer.Typer(
    help="Music lyrics transcription pipeline (Demucs + HeartTranscriptor).",
    no_args_is_help=True,
    add_completion=False,
)

_PREVIEW_CHARS = 500

_PHASE_LABELS = {
    SEPARATION_PHASE: "Separating",
    TRANSCRIPTION_PHASE: "Transcribing",
}


@app.callback()
def main(
    ctx: typer.Context,
    config: Optional[Path] = typer.Option(
        None, "--config", help="Path to an alternative config.yaml."
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", help="Override the output directory."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Detailed logs (DEBUG)."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal logs (WARNING)."),
) -> None:
    """Handle the global options shared by every subcommand."""
    if verbose and quiet:
        raise typer.BadParameter("--verbose and --quiet are mutually exclusive.")
    level = logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    set_level(level)

    try:
        cfg = Config.load(config)
    except (FileNotFoundError, PipelineError) as exc:
        console.print(f"[bold red]Configuration error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    if output_dir is not None:
        cfg.pipeline.output_dir = str(output_dir)
    ctx.obj = cfg


def _config(ctx: typer.Context) -> Config:
    """Return the configuration built by the main callback."""
    if not isinstance(ctx.obj, Config):  # pragma: no cover - defensive
        ctx.obj = Config.load()
    return ctx.obj


@app.command()
def transcribe(
    ctx: typer.Context,
    audio: Path = typer.Argument(..., help="Audio file to transcribe."),
    quality: Optional[str] = typer.Option(
        None,
        "--quality",
        help="Demucs preset: fast, balanced or best. Defaults to config.yaml.",
    ),
    no_separation: bool = typer.Option(
        False, "--no-separation", help="Skip Demucs, for an a cappella track."
    ),
    only_separate: bool = typer.Option(
        False, "--only-separate", help="Only separate the vocals, without transcribing."
    ),
    show_quality_report: bool = typer.Option(
        False, "--show-quality-report", help="Print the quality report."
    ),
    no_clean: bool = typer.Option(
        False, "--no-clean", help="Disable character artifact cleaning."
    ),
    no_dedup: bool = typer.Option(
        False, "--no-dedup", help="Disable deduplication."
    ),
    no_preprocess: bool = typer.Option(
        False, "--no-preprocess", help="Disable audio pre processing."
    ),
    no_linesplit: bool = typer.Option(
        False, "--no-linesplit", help="Disable the reflow of lyrics into verses."
    ),
    no_postprocess: bool = typer.Option(
        False, "--no-postprocess", help="Disable vocal post processing."
    ),
    report_json: bool = typer.Option(
        False, "--report-json", help="Write the JSON report next to the text output."
    ),
    reverb: Optional[float] = typer.Option(
        None, "--reverb", help="Override the reverb reduction aggressiveness (0.0 to 0.6)."
    ),
) -> None:
    """Transcribe a single audio file."""
    cfg = _config(ctx)
    if quality is not None:
        cfg.models.demucs.quality_preset = quality

    if no_clean:
        cfg.pipeline.cleaning.enabled = False
    if no_dedup:
        cfg.pipeline.deduplication.enabled = False
    if no_preprocess:
        cfg.pipeline.preprocessing.enabled = False
    if no_linesplit:
        cfg.pipeline.line_splitting.enabled = False
    if no_postprocess:
        cfg.pipeline.postprocessing.enabled = False
    if report_json:
        cfg.pipeline.report.json = True
    if reverb is not None:
        cfg.pipeline.postprocessing.reverb_aggressiveness = reverb

    try:
        pipeline = LyricsPipeline(cfg)
        if only_separate:
            vocals = pipeline.separate_only(audio)
            console.print(f"[green]Isolated vocals written:[/green] {vocals}")
            return
        result = pipeline.run(audio, separate=not no_separation)
    except PipelineError as exc:
        console.print(f"[bold red]Failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    preview = result.transcription.lyrics[:_PREVIEW_CHARS]
    if len(result.transcription.lyrics) > _PREVIEW_CHARS:
        preview += "..."
    console.print(
        Panel(
            preview or "[dim](no lyrics detected)[/dim]",
            title=f"Lyrics: {result.input_path.name}",
            border_style="green",
        )
    )
    console.print(f"[green]Results written to:[/green] {result.output_dir}")
    report = result.transcription.quality_report or {}
    console.print(
        f"Audio duration: {result.duration_seconds:.1f}s, "
        f"total time: {result.total_time_seconds:.1f}s, "
        f"language: {result.transcription.language}, "
        f"grade: {report.get('quality_grade', '?')}"
    )
    if show_quality_report:
        _print_quality_report(result.transcription)


def _print_quality_report(transcription: TranscriptionResult) -> None:
    """Print the quality report of a transcription in the terminal."""
    report = transcription.quality_report or {}
    table = Table(title="Quality report", show_header=False, border_style="yellow")
    table.add_row("Overall grade", str(report.get("quality_grade", "?")))
    table.add_row("Average confidence", f"{report.get('overall_confidence', 0.0):.2f}")
    table.add_row(
        "Degraded segments",
        f"{report.get('degraded_segments', 0)} / {len(transcription.segments)} "
        f"({report.get('degraded_ratio', 0.0) * 100:.1f}%)",
    )
    table.add_row("Duplicates removed", str(transcription.duplicates_removed))
    console.print(table)


def _progress_renderer(progress: Progress) -> Callable[[BatchProgress], None]:
    """Build a progress callback drawing one rich task per pipeline phase."""
    tasks: dict[str, TaskID] = {}

    def render(event: BatchProgress) -> None:
        label = _PHASE_LABELS.get(event.phase, event.phase)
        task = tasks.get(event.phase)
        if task is None:
            task = progress.add_task(label, total=event.total)
            tasks[event.phase] = task
        description = f"{label}: {event.current}" if event.current else label
        progress.update(task, completed=event.completed, description=description)

    return render


@app.command()
def batch(
    ctx: typer.Context,
    folder: Path = typer.Argument(..., help="Directory holding the audio files."),
    no_separation: bool = typer.Option(
        False, "--no-separation", help="Skip Demucs for every file."
    ),
) -> None:
    """Transcribe every supported audio file of a directory."""
    cfg = _config(ctx)
    folder = Path(folder)
    if not folder.is_dir():
        console.print(f"[bold red]Directory not found:[/bold red] {folder}")
        raise typer.Exit(code=1)

    supported = set(cfg.pipeline.supported_formats)
    audio_files = sorted(
        p for p in folder.iterdir() if p.suffix.lower() in supported and p.is_file()
    )
    if not audio_files:
        console.print(f"[yellow]No supported audio file in:[/yellow] {folder}")
        raise typer.Exit(code=1)

    console.print(f"[cyan]{len(audio_files)} file(s) to process.[/cyan]")
    pipeline = LyricsPipeline(cfg)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        results = pipeline.run_batch(
            audio_files,
            separate=not no_separation,
            on_progress=_progress_renderer(progress),
        )
    console.print(
        f"[green]Done:[/green] {len(results)}/{len(audio_files)} files transcribed."
    )


@app.command(name="check-env")
def check_env(ctx: typer.Context) -> None:
    """Check the environment: GPU, CUDA, VRAM and model availability."""
    cfg = _config(ctx)
    status = check_gpu_status()

    table = Table(title="GPU environment", show_header=False, border_style="cyan")
    table.add_row("CUDA available", "yes" if status["cuda_available"] else "no")
    table.add_row("GPU", str(status["device_name"]))
    table.add_row("Total VRAM", f"{status['vram_total_gb']} GB")
    table.add_row("Free VRAM", f"{status['vram_free_gb']} GB")
    table.add_row(
        "Lazy load recommended",
        "yes" if status["recommended_lazy_load"] else "no",
    )
    console.print(table)

    ckpt = cfg.resolve_path(cfg.models.hearttranscriptor.model_path)
    model_table = Table(title="Models", show_header=False, border_style="magenta")
    model_table.add_row("Demucs", cfg.models.demucs.name)
    model_table.add_row(
        "HeartTranscriptor checkpoint",
        f"found: {ckpt}" if ckpt.exists() else f"missing: {ckpt}",
    )
    console.print(model_table)

    if not ckpt.exists():
        console.print(
            "[yellow]Hint:[/yellow] run "
            "[bold]python scripts/download_models.py[/bold] to fetch the weights."
        )


if __name__ == "__main__":
    app()
