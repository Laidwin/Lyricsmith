# Lyricsmith

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Turn a song into timestamped lyrics. Lyricsmith isolates the vocal track with
[Demucs](https://github.com/facebookresearch/demucs), transcribes it with
**HeartTranscriptor** ([HeartMuLa](https://github.com/HeartMuLa)), then cleans,
deduplicates, scores and reflows the result into readable verses.

It works as a command line tool and as a Python library, and it is built to run
on a single consumer GPU with 8 GB of VRAM.

```
[00:00] They say, "You'll get used to it"
[00:04] But it never goes away
[01:23] [degraded segment: low confidence, confidence 0.31]

---
Quality
  Grade              : B
  Overall confidence : 0.74
  Degraded segments  : 2 / 28 (7.1%)
```

Every transcription is graded from A to D, and the segments the model is least
sure about are flagged rather than silently kept.

## Installation

Requires Python 3.10, an NVIDIA GPU with 8 GB of VRAM, CUDA 12.x and ffmpeg.
It also runs on CPU, slower.

```bash
uv venv --python 3.10
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install -r requirements.txt
python scripts/download_models.py
```

Install PyTorch from the CUDA index first, otherwise the CPU build is pulled in
and everything runs on CPU. Full details in the
[installation guide](docs/installation.md).

## Quick start

```bash
python main.py check-env                  # GPU, CUDA, VRAM and model check
python main.py transcribe ./my_song.mp3   # one file
python main.py batch ./my_songs/          # a whole directory
```

Results land in `./outputs/<song>/`: the plain lyrics, the timestamped segments
as JSON, an annotated transcript and two reports.

As a library:

```python
from pathlib import Path
from lyricsmith import Config, LyricsPipeline

result = LyricsPipeline(Config.load()).run(Path("song.mp3"))
print(result.transcription.lyrics)
```

## Documentation

The full documentation is built with MkDocs Material and lives in
[`docs/`](docs/index.md).

| Page | What it covers |
| --- | --- |
| [Installation](docs/installation.md) | Requirements, environment setup, model download |
| [Command line](docs/cli.md) | Every command, every flag, the quality presets |
| [Library](docs/library.md) | Using Lyricsmith from Python, the public API |
| [Configuration](docs/configuration.md) | Every `config.yaml` key, its type and default |
| [Processing chain](docs/processing-chain.md) | The eight steps, in order, and how to disable them |
| [Output format](docs/output.md) | The files written, their schema, the quality grades |
| [Architecture](docs/architecture.md) | Layering, design decisions, platform workarounds |
| [Troubleshooting](docs/troubleshooting.md) | CUDA out of memory, missing checkpoint, decoding errors |
| [Contributing](docs/contributing.md) | Tests, conventions, how to add a step |

To read it as a site:

```bash
uv pip install -e ".[docs]"
mkdocs serve
```

## Tests

```bash
uv pip install -e ".[dev]"
pytest
```

The suite needs no GPU and no model weights: Demucs and heartlib are replaced by
test doubles.

## License

Released under the [MIT License](LICENSE).
