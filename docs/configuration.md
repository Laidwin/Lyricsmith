# Configuration

Every tunable parameter lives in `config.yaml` at the project root. Nothing is
hard coded in the Python code, and no setting is read from anywhere else.

## How the file is loaded

The file is resolved in this order:

1. the `--config PATH` command line option, or the argument given to
   `Config.load()`;
2. the `LYRICSMITH_CONFIG` environment variable;
3. `config.yaml` next to the package, at the project root.

It is then parsed into typed dataclasses, one per section. Two consequences
matter in practice:

- **A key you did not declare is an error.** A typo raises `ConfigError` at
  startup instead of silently falling back to a default.

    ```
    Configuration error: Unknown key(s) in section 'pipeline.cleaning': enabledd.
    Known keys: custom_patterns, enabled, min_segment_words, remove_fullwidth,
    remove_repeated_chars.
    ```

- **Any key may be omitted.** Missing sections and missing keys fall back to the
  defaults listed below, which are declared once, in `lyricsmith/config.py`.

Relative paths, such as `output_dir` and `model_path`, are resolved against the
directory holding `config.yaml`, never against the current working directory.

## Full reference

### `models.demucs`

Vocal separation. See the [presets](cli.md#demucs-quality-presets), which
override `name` and `shifts` when set.

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `name` | `str` | `htdemucs` | Any Demucs model name, `htdemucs_ft` is the fine tuned one |
| `device` | `str` | `cuda` | `cuda` or `cpu`, falls back to CPU when CUDA is absent |
| `shifts` | `int` | `1` | More shifts means better quality, more time and VRAM |
| `overlap` | `float` | `0.25` | Overlap between processed chunks |
| `jobs` | `int` | `0` | CPU workers, `0` means auto |
| `quality_preset` | `str` or null | `null` | `fast`, `balanced` or `best` |

### `models.hearttranscriptor`

The ASR engine and its generation options. Every anti hallucination option
defaults to null, meaning "not set", in which case it is left out of the
generation call entirely and the model default applies.

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `model_path` | `str` | `./models/ckpt` | Directory holding the checkpoint |
| `device` | `str` | `cuda` | `cuda` or `cpu` |
| `lazy_load` | `bool` | `true` | Load and unload around each file, required on 8 GB |
| `dtype` | `str` | `bf16` | `bf16`, `fp16` or `fp32` |
| `chunk_length_s` | `int` | `30` | Window fed to Whisper, drives VRAM use |
| `batch_size` | `int` | `16` | Windows processed at once, drives VRAM use |
| `condition_on_previous_text` | `bool` or null | `null` | The main fix against repetition loops, set it to `false` |
| `no_speech_threshold` | `float` or null | `null` | Discards chunks without clear vocals |
| `logprob_threshold` | `float` or null | `null` | Required as soon as `no_speech_threshold` is set |
| `compression_ratio_threshold` | `float` or null | `null` | Detects and rejects repetitive chunks |
| `temperature` | `float` or list | `null` | A list is a fallback ladder, tried in order |
| `beam_size` | `int` or null | `null` | Beam search width, mapped to `num_beams` |

!!! warning "logprob_threshold travels with no_speech_threshold"
    The Whisper fallback in `transformers` reads `logprobs` inside its
    no-speech branch, and that value only exists when `logprob_threshold` is
    set. Setting `no_speech_threshold` alone raises `UnboundLocalError` deep
    inside the library. The shipped `config.yaml` sets both.

The openai-whisper option `best_of` has no equivalent in the `transformers`
generation API and is deliberately unsupported.

### `pipeline`

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `output_dir` | `str` | `./outputs` | One subdirectory per input file is created here |
| `keep_stems` | `bool` | `false` | Keep the Demucs stems next to the lyrics |
| `target_sample_rate` | `int` | `16000` | Sample rate expected by HeartTranscriptor |
| `supported_formats` | list of `str` | `.mp3`, `.wav`, `.flac`, `.m4a`, `.ogg` | Lowercased once at load time |

### `pipeline.preprocessing`

Applied to the source audio, before Demucs. See
[Processing chain](processing-chain.md#1-preprocess).

| Key | Type | Default |
| --- | --- | --- |
| `enabled` | `bool` | `true` |
| `resample` | `bool` | `true` |
| `resample_target_sr` | `int` | `44100` |
| `remove_dc_offset` | `bool` | `true` |
| `normalize_loudness` | `bool` | `true` |
| `target_lufs` | `float` | `-14.0` |

### `pipeline.postprocessing`

Applied to the isolated vocals, after Demucs.

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `enabled` | `bool` | `true` | |
| `reduce_reverb` | `bool` | `true` | |
| `reverb_aggressiveness` | `float` | `0.45` | Clamped to 0.0 to 0.6, above that the vocals turn metallic |
| `enhance_clarity` | `bool` | `true` | 80 Hz high pass plus a 2 to 4 kHz presence boost |

### `pipeline.cleaning`

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `enabled` | `bool` | `true` | |
| `remove_fullwidth` | `bool` | `true` | Fullwidth latin characters and digits |
| `remove_repeated_chars` | `bool` | `true` | A letter repeated four times or more |
| `min_segment_words` | `int` | `1` | A cleaned segment below this is dropped |
| `custom_patterns` | list of `str` | `[]` | Extra regular expressions, no code change needed |

### `pipeline.deduplication`

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `enabled` | `bool` | `true` | |
| `consecutive_max_repeats` | `int` | `2` | Consecutive similar segments kept |
| `consecutive_similarity` | `float` | `0.85` | Ratio above which two segments are the same |
| `ghost_block_min_size` | `int` | `3` | Minimum segments forming a block |
| `ghost_block_time_window` | `float` | `15.0` | Beyond this gap, a repeat is structural and kept |
| `ghost_block_similarity` | `float` | `0.85` | Ratio used to match two blocks |

### `pipeline.quality`

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `low_confidence_threshold` | `float` | `0.4` | Below this, a segment is flagged as degraded |

Quality scoring always runs, since the reports depend on it. There is no
`enabled` key.

### `pipeline.line_splitting`

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `enabled` | `bool` | `true` | |
| `max_words` | `int` | `10` | Length above which a cut on a function word is allowed |
| `hard_max_words` | `int` | `16` | Absolute maximum, forces a cut |
| `min_words_to_split` | `int` | `12` | Shorter lines are never touched |

### `pipeline.report`

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `json` | `bool` | `true` | Also write `*_pipeline_report.json` |

## Tuning for less VRAM

The defaults target 8 GB. To go lower, in order of effect:

```yaml
models:
  demucs:
    shifts: 1            # Or quality_preset: fast
  hearttranscriptor:
    lazy_load: true      # Keep this on
    batch_size: 8        # Then 4
    chunk_length_s: 20
    dtype: fp16
```

Falling back to `device: cpu` on either model always works and is the last
resort. See [Troubleshooting](troubleshooting.md).
