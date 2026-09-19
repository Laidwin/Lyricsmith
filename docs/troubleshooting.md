# Troubleshooting

## CUDA out of memory

The pipeline reports it explicitly, during separation or transcription:

```
Not enough GPU memory (CUDA out of memory) during separation.
Lower 'shifts' in config.yaml, switch to 'device: cpu', or free some VRAM.
```

Try these in order:

1. Lower `models.demucs.shifts`, or set `quality_preset: fast`.
2. Check that `models.hearttranscriptor.lazy_load` is still `true`.
3. Lower `batch_size` to 8, then 4, then lower `chunk_length_s`.
4. Switch `dtype` to `fp16`.
5. Close whatever else uses the GPU, a browser with hardware acceleration is
   often worth a gigabyte.
6. Set `device: cpu` on either model, which always works and is slow.

## The checkpoint is not found

```
HeartTranscriptor checkpoint not found: .../models/ckpt.
Run first: python scripts/download_models.py
```

Run `python main.py check-env` to confirm what the pipeline is looking at, then
`python scripts/download_models.py`. If the weights live elsewhere, point
`models.hearttranscriptor.model_path` at them. Both layouts are accepted, the
weights directly in the directory or inside a `HeartTranscriptor-oss`
subdirectory.

## The configuration is rejected at startup

```
Configuration error: Unknown key(s) in section 'pipeline.cleaning': enabledd.
```

The key does not exist. The message lists the valid keys for that section; the
full list is in [Configuration](configuration.md#full-reference). This is
deliberate: a silently ignored typo used to mean the setting you thought you
changed never applied.

Two keys were removed in the process, `models.hearttranscriptor.similarity_threshold`
and `max_repeats`. Their replacements are
`pipeline.deduplication.consecutive_similarity` and
`consecutive_max_repeats`.

## mp3 or m4a decoding errors

Install `ffmpeg` and make sure it is on your `PATH`. `librosa` decodes
compressed formats through it. As a check, convert the file to WAV yourself and
transcribe that instead.

## The lyrics repeat the same line over and over

That is a Whisper hallucination loop, and the deduplication catches most of it
after the fact. To attack the cause, in `models.hearttranscriptor`:

- set `condition_on_previous_text: false`, which is the single most effective
  setting;
- set `compression_ratio_threshold: 2.4`, which makes the model reject
  repetitive output during generation;
- keep the `temperature` fallback ladder, so a failed greedy pass is retried.

Check `duplicates_removed` and `ghost_blocks_removed` in the metadata to see how
much was cleaned up after the fact.

## The transcript is graded C or D

A poor grade usually points at the separation, not the transcription. Listen to
the isolated vocals:

```bash
python main.py transcribe ./song.mp3 --only-separate
```

If the vocals are muddy or full of bleed, use `--quality best`, and make sure
`pipeline.preprocessing` is enabled. If they are clean but thin and metallic,
lower `reverb_aggressiveness`, or disable the post processing with
`--no-postprocess` and compare.

## Everything runs on CPU although CUDA is installed

`python main.py check-env` reports `CUDA available: no`. The usual cause is a
CPU build of PyTorch, installed by `requirements.txt` before the CUDA build. See
[Installation](installation.md#setting-up-the-environment). Confirm with:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

A version without a CUDA suffix means a CPU build. Reinstall `torch` and
`torchaudio` from the CUDA index.

## The lyrics come out as one long line

The verse reflow is disabled, or the line is under its threshold. Check
`pipeline.line_splitting.enabled`, and lower `min_words_to_split`. Note that the
reflow only changes the `.txt` file and the `lyrics` field of the JSON, never
the timestamped segments.

## Unicode errors when redirecting the output

The logger forces stdout and stderr to UTF-8 at import time, which covers the
usual cases. If your terminal still refuses, set `PYTHONIOENCODING=utf-8`
before the command.
