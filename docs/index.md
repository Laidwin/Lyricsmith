# Lyricsmith

Lyricsmith turns a song into timestamped lyrics. It isolates the vocal track
with [Demucs](https://github.com/facebookresearch/demucs), transcribes it with
**HeartTranscriptor** ([HeartMuLa](https://github.com/HeartMuLa)), then cleans,
deduplicates, scores and reflows the result into readable verses.

It works as a command line tool and as a Python library, and it is designed to
run on a single consumer GPU with 8 GB of VRAM.

## What it produces

For `song.mp3`, a run writes a directory holding the plain lyrics, the
timestamped segments as JSON, an annotated transcript and two reports:

```
[00:00] They say, "You'll get used to it"
[00:04] But it never goes away
[01:23] [degraded segment: low confidence, confidence 0.31]

---
Pipeline report: song.mp3
  Vocal separation   : htdemucs_ft, shifts=2
  Detected language  : unknown
  Audio duration     : 3:21

Quality
  Grade              : B
  Overall confidence : 0.74
  Degraded segments  : 2 / 28 (7.1%)
```

Every transcription is graded from A to D, and the segments the model is least
sure about are flagged rather than silently kept. See
[Output format](output.md).

## Three commands

```bash
python main.py check-env                  # GPU, CUDA, VRAM and model check
python main.py transcribe ./my_song.mp3   # one file
python main.py batch ./my_songs/          # a whole directory
```

## Where to go next

| Page | What it covers |
| --- | --- |
| [Installation](installation.md) | Requirements, environment setup, model download |
| [Command line](cli.md) | Every command, every flag, the quality presets |
| [Library](library.md) | Using Lyricsmith from Python, the public API |
| [Configuration](configuration.md) | Every `config.yaml` key, its type and default |
| [Processing chain](processing-chain.md) | The eight steps, in order, and how to disable them |
| [Output format](output.md) | The files written, their schema, the quality grades |
| [Architecture](architecture.md) | Layering, design decisions, platform workarounds |
| [Troubleshooting](troubleshooting.md) | CUDA out of memory, missing checkpoint, decoding errors |
| [Contributing](contributing.md) | Tests, conventions, how to add a step |

## License

Released under the [MIT License](https://opensource.org/licenses/MIT).
