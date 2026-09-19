# Architecture

## Layers

The package is layered so that every dependency points inward, toward the
entities, and never outward toward the delivery mechanism.

```
cli.py  ->  pipeline.py  ->  separator.py / transcriptor.py  ->  utils/  ->  domain.py
                    \                                                        ^
                     \-> refine.py -> utils/ --------------------------------|
                      \-> report.py ------------------------------------------|
```

| Module | Layer | Responsibility |
| --- | --- | --- |
| `domain.py` | Entities | `Segment`, `RawTranscription`, `TranscriptionResult`, `ArtifactReport`. Imports nothing from the package. |
| `config.py` | Configuration | Parses and validates `config.yaml` into typed dataclasses |
| `utils/` | Services | Audio input and output, cleaning, deduplication, quality, verse reflow, GPU checks, logging |
| `separator.py` | Adapter | Wraps Demucs |
| `transcriptor.py` | Adapter | Wraps HeartTranscriptor, returns raw output and nothing else |
| `refine.py` | Policy | The order of the text post processing steps |
| `report.py` | Presentation | Renders the annotated transcript and the JSON reports |
| `pipeline.py` | Use case | Sequences a run, writes the files, emits progress events |
| `cli.py` | Delivery | Parses arguments, renders tables and progress bars |

Three rules hold, and are worth keeping:

- **`domain.py` imports nothing from the package.** That is what lets
  `utils/cleaner.py` and `utils/quality.py` import `Segment` directly, with no
  circular import workaround.
- **No module below `cli.py` imports `rich`,** except the logger. The
  orchestrator renders nothing, so the library works in a server or a notebook.
- **Adapters own no pipeline policy.** `transcriptor.py` returns a
  `RawTranscription`; what happens to it afterwards is decided in `refine.py`.

## Design decisions

### The 8 GB VRAM constraint

Demucs and HeartTranscriptor do not fit in 8 GB together, and making them
coexist crashes the CUDA runtime rather than raising a Python error. They are
therefore never resident at the same time:

- `LyricsPipeline.run` calls `separator.release()` before the transcriptor
  loads anything;
- the transcriptor honours `lazy_load`, loading the model inside `transcribe`
  and releasing it immediately after.

The release ordering matters more than it looks. `transcribe` clears its own
reference to the model before calling `_release_model`, because emptying the
CUDA cache while any reference is still alive frees nothing at all.

### Two pass batch processing

Calling `run()` in a loop would load and unload each model once per file. A
batch instead works in two passes: Demucs separates every file, with the stems
kept on disk, then it is unloaded and HeartTranscriptor transcribes every
separated file. Each model is loaded once for the whole batch, and the two never
share VRAM.

The trade off is disk: the stems of all files sit in the temporary directory
between the two passes. Failing files are logged and skipped, so one bad file
never interrupts a long batch.

### Progress as events

`run_batch` takes an `on_progress` callback and emits `BatchProgress` events.
The CLI turns them into `rich` progress bars, and a different caller can ignore
them or send them anywhere. This is what keeps the use case layer free of any
rendering dependency.

### Configuration as typed dataclasses

Parsing `config.yaml` into dataclasses means each default is declared exactly
once, instead of being repeated at every `cfg.get(key, default)` call site,
where copies drift apart over time. It also makes an unknown key an error at
startup rather than a silent fallback.

## Platform workarounds

### Loading heartlib without its initializer

The `heartlib` package initializer imports its music generation pipeline first,
which pulls torchtune and native kernels and crashes on import on Windows, with
an access violation rather than a catchable exception. Since the transcription
module has no relative import of its own, it is loaded straight from its file
with `importlib`, bypassing the initializer. That is
`LyricsTranscriptor._load_pipeline_class`.

### Demucs without `demucs.api`

Some Demucs 4.x builds ship without the convenient `demucs.api` module. The
separator probes for it once and falls back to the low level API,
`demucs.pretrained` plus `demucs.apply` and `demucs.audio`, which is present in
every 4.x build. On that path the input is normalized before inference and the
stems denormalized after, as Demucs recommends, with a guard against a fully
silent track.

### Whisper generation options

The configured anti hallucination options are mapped to the names the
`transformers` Whisper generation expects, for instance
`condition_on_previous_text` to `condition_on_prev_tokens` and `beam_size` to
`num_beams`. Options left unset are omitted entirely rather than passed as null.
If a combination is still rejected by the installed version, the call is retried
once with the model defaults and a warning is logged, so a version bump degrades
quality instead of breaking the run.

### UTF-8 on Windows

Windows redirects stdout to cp1252, which raises `UnicodeEncodeError` on the box
drawing characters used by the tables as soon as the output is piped to a file.
The logger reconfigures both streams to UTF-8 at import time.

## Testing strategy

The suite runs without a GPU, without Demucs, without heartlib and without any
model weights. Doubles are injected for the two heavy dependencies, and the
steps that are pure functions are tested directly on `Segment` values. This is
the practical payoff of the layering: `refine.py` holds the step order, so the
order is testable in milliseconds.

See [Contributing](contributing.md) for how to run and extend the suite.
