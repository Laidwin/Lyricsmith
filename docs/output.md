# Output format

For `song.mp3`, a run writes into `./outputs/song/`:

| File | Content |
| --- | --- |
| `song_lyrics.txt` | Plain text lyrics, one verse per line |
| `song_lyrics.json` | Lyrics, timestamped segments, confidence, quality report |
| `song_lyrics_annotated.txt` | `[mm:ss]` lyrics, degraded segments, readable report |
| `song_metadata.json` | Duration, language, timings, grade, counters |
| `song_pipeline_report.json` | Structured report of every step, when `report.json` is true |
| `song_vocals.wav` | Isolated vocals, only when `keep_stems` is true |
| `song_no_vocals.wav` | Accompaniment, only when `keep_stems` is true |

The output directory is named after the input file stem, so two files with the
same name overwrite each other.

## `song_lyrics.json`

```json
{
  "lyrics": "first line\nsecond line",
  "language": "unknown",
  "processing_time_seconds": 12.3,
  "duplicates_removed": 7,
  "ghost_blocks_removed": 3,
  "quality_report": {
    "overall_confidence": 0.74,
    "degraded_segments": 2,
    "degraded_ratio": 0.071,
    "quality_grade": "B"
  },
  "degraded_segments": [12, 19],
  "artifact_reports": [
    {
      "source_segment_index": 4,
      "timestamp_start": 18.2,
      "original_text": "oh iiii yeah",
      "cleaned_text": "oh yeah",
      "artifacts_found": ["iiii"],
      "was_emptied": false
    }
  ],
  "segments": [
    {
      "start": 0.0,
      "end": 2.4,
      "text": "first line",
      "confidence": 0.95,
      "degraded_reason": null
    }
  ]
}
```

`degraded_segments` holds indices into `segments`.

!!! note "source_segment_index is not an index into segments"
    Inside `artifact_reports`, `source_segment_index` points at the raw segment
    list returned by the model, before cleaning and deduplication removed
    anything. Use `timestamp_start` to locate the segment in the final
    transcript.

`degraded_reason` is null on a healthy segment, and otherwise one of
`low_confidence`, `repetition` or `noise`.

`language` is `unknown` in most runs. HeartTranscriptor relies on a Whisper ASR
pipeline that does not return the detected language in its standard output. The
transcription itself remains multilingual.

## `song_lyrics_annotated.txt`

The transcript with timestamps, followed by a readable summary of the run.
Degraded segments are replaced by a marker rather than by text the model was
not sure about.

```
[00:00] They say, "You'll get used to it"
[00:04] But it never goes away
[01:23] [degraded segment: low confidence, confidence 0.31]

---
Pipeline report: song.mp3
  Vocal separation   : htdemucs_ft, shifts=2
  Pre processing     : normalized (-14 LUFS), DC offset removed
  Post processing    : reverb reduced (aggressiveness 0.45), clarity enhanced
  Detected language  : unknown
  Audio duration     : 3:21
  Processing time    : 198s

Quality
  Grade              : B
  Overall confidence : 0.74
  Degraded segments  : 2 / 28 (7.1%)

Cleaning
  Artifacts removed  : 5  (iiii, （, ）)
  Segments emptied   : 1

Deduplication
  Consecutive repetitions removed : 7
  Ghost blocks removed            : 3
```

## `song_metadata.json`

A flat document, convenient to collect across many runs.

| Field | Meaning |
| --- | --- |
| `input_file` | Absolute path of the source audio |
| `duration_seconds` | Audio duration, `0.0` when separation was skipped |
| `language` | Detected language, usually `unknown` |
| `separation_used` | Whether Demucs ran |
| `separation_time_seconds` | Time spent in Demucs |
| `transcription_time_seconds` | Time spent in the ASR engine |
| `total_time_seconds` | Whole run, separation included |
| `num_segments` | Segments in the final transcript |
| `duplicates_removed` | Consecutive repetitions dropped |
| `ghost_blocks_removed` | Segments dropped as ghost blocks |
| `quality_grade` | A to D |
| `overall_confidence` | Duration weighted average |
| `degraded_segments` | How many segments are flagged |
| `artifacts_removed` | Characters removed by the cleaning |
| `stems_kept` | Whether the stems were kept |

## `song_pipeline_report.json`

The detailed record of the run: the metadata above, the separation settings, the
full pre and post processing results, the quality report, every artifact report,
and the deduplication counters. This is the file to read when a transcript looks
wrong and you need to know which step did what.

## Quality grades

| Grade | Criteria |
| --- | --- |
| A | Confidence of at least 0.80 and under 5% degraded segments |
| B | Confidence of at least 0.65 and under 15% |
| C | Confidence of at least 0.50 and under 30% |
| D | Anything below |

The overall confidence is a duration weighted average, so a long confident verse
counts more than a short uncertain one. When no segment carries a duration, the
plain mean is used instead.

A grade of C or D usually means the separation struggled rather than the
transcription. Try `--quality best`, and check the isolated vocals with
`--only-separate`.
