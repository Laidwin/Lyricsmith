# Command line

Run Lyricsmith from a clone with `python main.py ...`, or install the project
and use the `lyricsmith` command. Both run the same application, defined in
`lyricsmith/cli.py`.

```bash
python main.py --help
```

## Global options

These come before the subcommand.

| Option | Description |
| --- | --- |
| `--config PATH` | Use an alternative `config.yaml` |
| `--output-dir DIR` | Override the output directory for this run |
| `--verbose`, `-v` | Detailed logs (DEBUG) |
| `--quiet`, `-q` | Minimal logs (WARNING) |

`--verbose` and `--quiet` are mutually exclusive. The configuration file can
also be selected through the `LYRICSMITH_CONFIG` environment variable, which
the explicit `--config` flag overrides.

```bash
python main.py --config ./experiments/fast.yaml --quiet transcribe song.mp3
```

## `transcribe`

Transcribes a single audio file.

```bash
python main.py transcribe ./my_song.mp3
```

| Option | Description |
| --- | --- |
| `--quality [preset]` | Demucs preset: `fast`, `balanced` or `best` |
| `--no-separation` | Skip Demucs, for a track that is already an a cappella |
| `--only-separate` | Only isolate the vocals, without transcribing |
| `--show-quality-report` | Print the quality report as a table |
| `--no-preprocess` | Disable the audio pre processing, before Demucs |
| `--no-postprocess` | Disable the vocal post processing, after Demucs |
| `--no-clean` | Disable the character artifact cleaning |
| `--no-dedup` | Disable deduplication, both consecutive and ghost blocks |
| `--no-linesplit` | Disable the reflow of the lyrics into verses |
| `--report-json` | Write `*_pipeline_report.json` |
| `--reverb FLOAT` | Override the reverb reduction aggressiveness, 0.0 to 0.6 |

Every flag overrides the corresponding `config.yaml` value for that run only.

```bash
# Quick draft with the light preset
python main.py transcribe ./song.mp3 --quality fast

# Best separation, with the quality report in the terminal
python main.py transcribe ./song.mp3 --quality best --show-quality-report

# The track is already a vocal stem
python main.py transcribe ./acapella.wav --no-separation

# Only produce the isolated vocals
python main.py transcribe ./song.mp3 --only-separate
```

On success the command prints a preview of the lyrics, the output directory and
a summary line with the duration, the elapsed time, the detected language and
the quality grade. It exits with code 1 on any pipeline error.

## `batch`

Transcribes every supported audio file of a directory, without recursing into
subdirectories.

```bash
python main.py batch ./my_songs/
```

| Option | Description |
| --- | --- |
| `--no-separation` | Skip Demucs for every file |

The batch runs in two passes, separating everything before transcribing
anything, so each model is loaded once for the whole set. See
[Architecture](architecture.md#two-pass-batch-processing) for why. Files that
fail are logged and skipped, and the batch continues.

```
Separating   ---------------------------------------- 12/12 0:04:41
Transcribing ---------------------------------------- 12/12 0:03:07
Done: 12/12 files transcribed.
```

Which extensions are picked up is set by `pipeline.supported_formats`.

## `check-env`

Reports CUDA availability, the GPU model, total and free VRAM, whether lazy
loading is recommended, and whether the HeartTranscriptor checkpoint is present.
It takes no option and changes nothing.

```bash
python main.py check-env
```

## Demucs quality presets

| Preset | Model | Shifts | Use case |
| --- | --- | --- | --- |
| `fast` | `htdemucs` | 1 | Quick draft |
| `balanced` | `htdemucs_ft` | 2 | Default, good trade off |
| `best` | `htdemucs_ft` | 4 | Reverberated vocals, stacked harmonies |

More shifts means better separation, proportionally more time and more VRAM.
A preset set in `config.yaml` or passed through `--quality` takes precedence
over the `name` and `shifts` keys.
