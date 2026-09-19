# Processing chain

A run applies these steps, always in this order:

```
preprocess -> separate -> postprocess -> transcribe -> clean -> dedup -> quality -> linesplit
```

The first three touch audio, the last four touch text. Every step except
quality scoring can be disabled independently, in `config.yaml` or with a CLI
flag.

| Step | Config section | CLI flag to disable |
| --- | --- | --- |
| preprocess | `pipeline.preprocessing` | `--no-preprocess` |
| separate | `models.demucs` | `--no-separation` |
| postprocess | `pipeline.postprocessing` | `--no-postprocess` |
| transcribe | `models.hearttranscriptor` | not optional |
| clean | `pipeline.cleaning` | `--no-clean` |
| dedup | `pipeline.deduplication` | `--no-dedup` |
| quality | `pipeline.quality` | not optional |
| linesplit | `pipeline.line_splitting` | `--no-linesplit` |

## 1. Preprocess

Applied to the source audio, before Demucs, because a clean and calibrated
signal separates better.

- **Resample** to 44.1 kHz, the rate Demucs was trained on. A file already at
  the target rate is passed through untouched, with no needless re-encoding.
- **Remove the DC offset** with a 20 Hz Butterworth high pass filter.
- **Normalize loudness** to a target of -14 LUFS with `pyloudnorm`, the
  streaming standard, with a peak ceiling to avoid clipping.

Each stage is guarded: a missing optional dependency or a failure is logged and
that stage is skipped, rather than aborting the run.

## 2. Separate

Demucs splits the track into isolated vocals and accompaniment, the sum of the
remaining sources. Only the vocals move forward. The stems are written to a
temporary directory and deleted at the end of the run, unless `keep_stems` is
true, in which case they land next to the lyrics.

Two code paths exist, chosen automatically: the convenient `demucs.api` when the
installed build ships it, and the low level `demucs.pretrained` plus
`demucs.apply` otherwise. See
[Architecture](architecture.md#demucs-without-demucsapi).

## 3. Postprocess

Applied to the isolated vocals only.

- **Reverb reduction** through spectral gating with `noisereduce`. The
  aggressiveness is clamped to the 0.0 to 0.6 range, since higher values leave
  metallic artifacts that hurt transcription more than the reverb did.
- **Clarity enhancement**, an 80 Hz high pass plus a 2 to 4 kHz presence boost
  that sharpens consonants.

This step is non destructive: the raw Demucs stem is kept and the processed
version is written beside it, then used as the transcription input.

## 4. Transcribe

The vocals are converted to 16 kHz mono WAV, read into memory and passed to
HeartTranscriptor as an array, so the pipeline never depends on ffmpeg at this
stage. The engine returns a `RawTranscription`: the text, the segments it could
timestamp, and the language when it reports one.

Segments holding no text are dropped here. If the configured generation options
are rejected by the installed `transformers` version, the call is retried once
with the model defaults and a warning is logged.

## 5. Clean

Whisper based models leave residues behind: byte order marks, zero width
spaces, fullwidth characters, CJK punctuation, letters repeated four or more
times, and stray non ASCII characters. Accented latin characters are preserved,
so French, Spanish and German lyrics come through intact.

Every removal is recorded in an `ArtifactReport`, including the original text,
so the cleaning stays auditable. A segment left with fewer than
`min_segment_words` words is dropped from the transcript but kept in the
reports.

## 6. Dedup

Two different problems, handled in two passes.

**Consecutive repetitions**, the classic hallucination loop, where the model
emits the same line over and over. Each segment is compared to the last kept
one, and past `consecutive_max_repeats` similar occurrences the extras are
dropped.

**Ghost blocks**, where a group of at least `ghost_block_min_size` segments
reappears a few seconds later without being adjacent, typically a pre chorus
duplicated in place. The time window is what separates an artifact from a real
structural repeat: a chorus coming back a minute later exceeds
`ghost_block_time_window` and is kept, as it should be.

## 7. Quality

Every surviving segment is scored between 0 and 1 and flagged as degraded when
it looks wrong. Since the `transformers` ASR pipeline returns no per segment
confidence, a heuristic score is derived from the text itself: the share of
normal characters, lexical diversity, and the share of single character tokens.

A segment is degraded when its confidence falls below
`low_confidence_threshold`, when it is repetitive, when it holds an abnormal
share of unusual characters, or when it is very short with a mediocre score.
The per segment scores are then aggregated, weighted by duration, into the
overall grade. See [Output format](output.md#quality-grades).

## 8. Linesplit

HeartTranscriptor often returns everything on a single line: it emits no usable
segment timestamps, and word level timestamps, when present, are contiguous with
no detectable pause. Rhythm is therefore not a usable signal, and the line is
split on three textual ones instead, by priority:

1. **Capital letters**, since the model usually capitalizes the first word of a
   verse. All caps words are treated as acronyms, and the English pronoun "I"
   with its contractions is excluded, as it is capitalized mid sentence.
2. **Function words**, used in fully lowercase passages where the first signal
   is silent. The line is cut before a conjunction or pronoun such as "and",
   "but", "et" or "si", but only once it is long enough, rather than mid
   sentence.
3. **Strong punctuation**, cutting after `.`, `!` or `?`.

A length guard prevents endless verses. Commas are deliberately ignored, since
in lyrics they usually sit inside a verse rather than end one.

This step is presentational: it changes the final text, meaning the `.txt` file
and the `lyrics` field of the JSON, and never the timestamped segments.

## Where each step lives

| Step | Module |
| --- | --- |
| preprocess, postprocess | `lyricsmith/utils/preprocess.py` |
| separate | `lyricsmith/separator.py` |
| transcribe | `lyricsmith/transcriptor.py` |
| clean | `lyricsmith/utils/cleaner.py` |
| dedup | `lyricsmith/utils/dedup.py` |
| quality | `lyricsmith/utils/quality.py` |
| linesplit | `lyricsmith/utils/linesplit.py` |
| the order itself | `lyricsmith/refine.py` and `lyricsmith/pipeline.py` |
