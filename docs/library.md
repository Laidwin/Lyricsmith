# Library

Lyricsmith is usable as a library. The orchestrator renders nothing and writes
only to the configured output directory, so it fits inside a script, a notebook
or a server just as well as inside the CLI.

## Running the pipeline

```python
from pathlib import Path
from lyricsmith import Config, LyricsPipeline

config = Config.load()                      # Reads and validates config.yaml
pipeline = LyricsPipeline(config)
result = pipeline.run(Path("song.mp3"))

print(result.transcription.lyrics)
print(f"Language: {result.transcription.language}")
for seg in result.transcription.segments:
    print(f"[{seg.start:.1f}-{seg.end:.1f}] {seg.text}")
```

`Config.load()` accepts an explicit path, falls back to the `LYRICSMITH_CONFIG`
environment variable, then to `config.yaml` at the project root. The returned
object is a plain value object: mutate it before building the pipeline to
override any setting.

```python
config = Config.load()
config.models.demucs.quality_preset = "fast"
config.pipeline.keep_stems = True
config.pipeline.line_splitting.enabled = False
```

## Batches and progress

`run_batch` loads each model once for the whole set. Progress is reported
through a callback, so nothing is printed unless you print it.

```python
from lyricsmith import LyricsPipeline, BatchProgress

def show(event: BatchProgress) -> None:
    print(f"{event.phase}: {event.completed}/{event.total} {event.current}")

results = LyricsPipeline(config).run_batch(paths, on_progress=show)
```

`BatchProgress` carries `phase` (either `separation` or `transcription`, exposed
as the `SEPARATION_PHASE` and `TRANSCRIPTION_PHASE` constants of
`lyricsmith.pipeline`), `completed`, `total` and `current`, the name of the file
being handled.

## Separation only

```python
vocals_path = LyricsPipeline(config).separate_only(Path("song.mp3"))
```

## Using the steps on their own

Each stage is importable and usable in isolation, which is what makes them
testable without a GPU.

```python
from lyricsmith.refine import refine_transcription
from lyricsmith.domain import RawTranscription, Segment

raw = RawTranscription(
    lyrics="take my hand\ntake my hand\ntake my hand",
    segments=[Segment(0.0, 1.0, "take my hand")],
)
result = refine_transcription(
    raw,
    cleaning=config.pipeline.cleaning,
    deduplication=config.pipeline.deduplication,
    line_splitting=config.pipeline.line_splitting,
    quality=config.pipeline.quality,
)
```

## Public API

Everything below is importable straight from `lyricsmith`.

| Name | Kind | Role |
| --- | --- | --- |
| `Config` | class | Loads and validates `config.yaml` |
| `LyricsPipeline` | class | Orchestrates a run, `run`, `run_batch`, `separate_only` |
| `PipelineResult` | dataclass | Paths and timings of one completed run |
| `BatchProgress` | dataclass | One progress event of a batch |
| `VocalSeparator` | class | Demucs adapter |
| `SeparationResult` | dataclass | Stem paths and separation timings |
| `LyricsTranscriptor` | class | HeartTranscriptor adapter |
| `RawTranscription` | dataclass | Unrefined engine output |
| `TranscriptionResult` | dataclass | Refined transcript, segments and reports |
| `Segment` | dataclass | One timestamped chunk of lyrics |
| `ArtifactReport` | dataclass | What the cleaning removed from one segment |
| `refine_transcription` | function | Applies the text post processing chain |
| `format_annotated_lyrics` | function | Renders the annotated transcript |

## Exceptions

All of them derive from `PipelineError`, so a single `except PipelineError`
catches everything the pipeline raises on purpose.

| Exception | Raised when |
| --- | --- |
| `ConfigError` | `config.yaml` holds an unknown key or an invalid section |
| `AudioFormatError` | The input file is missing or its extension unsupported |
| `SeparationError` | Demucs is missing, fails, or runs out of GPU memory |
| `TranscriptionError` | Inference fails, or heartlib cannot be loaded |
| `ModelNotFoundError` | The HeartTranscriptor checkpoint is absent |
| `GPUError` | CUDA is unavailable or VRAM is insufficient |

```python
from lyricsmith import LyricsPipeline, PipelineError

try:
    result = LyricsPipeline(config).run(path)
except PipelineError as exc:
    print(f"Failed: {exc}")
```
