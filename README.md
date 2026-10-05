<div align="center">

# Lyricsmith

**Turn a song into timestamped lyrics.**

Lyricsmith isolates the voice, transcribes it, then cleans and grades the result,
so you know exactly how much you can trust it.

[![Documentation](https://img.shields.io/badge/docs-laidwin.github.io%2FLyricsmith-blue)](https://laidwin.github.io/Lyricsmith/)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Docs built with Kiln](https://img.shields.io/badge/docs%20built%20with-Kiln-f76b15)](https://github.com/Laidwin/kiln)

[**Documentation**](https://laidwin.github.io/Lyricsmith/) · [**Quick start**](#quick-start)

</div>

```text
[00:00] Walking down the empty road tonight
[00:04] Counting every streetlight on the way
[01:23] [degraded segment: low confidence, confidence 0.31]

---
Quality
  Grade              : B
  Overall confidence : 0.74
  Degraded segments  : 2 / 28 (7.1%)
```

---

> **Part of a duo.** **Lyricsmith** turns a song into timestamped lyrics.
> [Soong](https://github.com/Laidwin/Soong) turns a music video into a lyrics
> video, and calls Lyricsmith when Genius does not know the song.
>
> ```text
> music video ──▶ lyrics (Genius, or Lyricsmith as a fallback) ──▶ word-level sync ──▶ lyrics video
> ```

---

## Highlights

- **Voice first.** The vocals are separated from the music with [Demucs](https://github.com/facebookresearch/demucs) before transcription, for much cleaner results.
- **Readable output.** The raw transcription is cleaned, deduplicated and reflowed into verses.
- **Honest about its limits.** Every song gets a grade from A to D, and doubtful passages are flagged instead of being silently kept.
- **Runs at home.** Designed for a single consumer GPU with 8 GB of memory, and works on CPU, slower.
- **Command line or Python library.** Transcribe one file, a whole folder, or call it from your own code.

---

## Quick start

You need Python 3.10, [uv](https://docs.astral.sh/uv/), Git and ffmpeg.
An NVIDIA GPU with CUDA 12 is recommended.

```bash
git clone https://github.com/Laidwin/Lyricsmith.git
cd Lyricsmith
uv venv --python 3.10
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install -r requirements.txt
python scripts/download_models.py
```

Install PyTorch from the CUDA index first, otherwise everything runs on CPU.

```bash
python main.py check-env                  # check GPU, CUDA and models
python main.py transcribe ./my_song.mp3   # one song
python main.py batch ./my_songs/          # a whole folder
```

Results land in `./outputs/<song>/`: the lyrics, the timestamped segments as JSON,
an annotated transcript and two reports.

From Python:

```python
from pathlib import Path
from lyricsmith import Config, LyricsPipeline

result = LyricsPipeline(Config.load()).run(Path("song.mp3"))
print(result.transcription.lyrics)
```

The [installation guide](https://laidwin.github.io/Lyricsmith/installation/) covers every detail.

---

## Documentation

| | |
| --- | --- |
| **Use it** | [Installation](https://laidwin.github.io/Lyricsmith/installation/) · [Command line](https://laidwin.github.io/Lyricsmith/cli/) · [Library](https://laidwin.github.io/Lyricsmith/library/) · [Configuration](https://laidwin.github.io/Lyricsmith/configuration/) |
| **Understand it** | [Processing chain](https://laidwin.github.io/Lyricsmith/processing-chain/) · [Output format](https://laidwin.github.io/Lyricsmith/output/) · [Troubleshooting](https://laidwin.github.io/Lyricsmith/troubleshooting/) |
| **Build on it** | [Architecture](https://laidwin.github.io/Lyricsmith/architecture/) · [Contributing](https://laidwin.github.io/Lyricsmith/contributing/) |

Built with Python, PyTorch, Demucs and HeartTranscriptor ([HeartMuLa](https://github.com/HeartMuLa)).
The test suite needs no GPU and no model weights: `uv pip install -e ".[dev]" && pytest`.

---

## License

[MIT](LICENSE). Models used at runtime keep their own licenses.
