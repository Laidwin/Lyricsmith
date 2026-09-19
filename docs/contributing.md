# Contributing

## Running the tests

```bash
uv pip install -e ".[dev]"
pytest
```

The suite runs without a GPU, without Demucs, without heartlib and without any
model weights: the two heavy dependencies are replaced by doubles, defined in
`tests/conftest.py`. A full run takes a few seconds, so there is no reason to
skip it.

A handful of tests exercise the real audio libraries, `soundfile`, `librosa`,
`pyloudnorm` and `noisereduce`. They skip themselves when the dependency is
absent, through `pytest.importorskip`.

| Test module | Covers |
| --- | --- |
| `test_config.py` | Loading, defaults, rejection of unknown keys, coercion |
| `test_pipeline.py` | End to end run with doubles, batch, progress events |
| `test_transcriptor.py` | The ASR adapter, generation options, model release |
| `test_refine.py` | The text post processing chain and its steps |
| `test_separator.py` | Presets and the Demucs adapter |
| `test_cleaner.py` | Character artifact removal |
| `test_dedup.py` | Ghost block detection |
| `test_linesplit.py` | Verse reflow |
| `test_preprocess.py` | Audio pre and post processing, with real dependencies |

## Writing style

These conventions apply to everything in the repository.

- **English everywhere**: identifiers, docstrings, comments, documentation and
  commit messages.
- **No em dashes and no emoji**, anywhere.
- **Comments only when strictly necessary.** Documentation belongs in
  docstrings; a comment explains a constraint that the code cannot express, for
  instance why a reference must be dropped before releasing the model.
- **Docstrings on every public function**, in Google style, with `Args`,
  `Returns` and `Raises` sections when they carry information.
- Line length is 100 characters, `ruff` is the linter, and its settings live in
  `pyproject.toml`.

## Architectural rules

Three invariants keep the layering honest. Breaking one is how this codebase
would rot.

1. **`domain.py` imports nothing from the package.** Entities stay free of
   configuration, models and input or output.
2. **Only `cli.py` and `utils/logger.py` import `rich`.** The pipeline renders
   nothing, and reports progress through the `on_progress` callback instead.
3. **Adapters hold no policy.** `separator.py` and `transcriptor.py` wrap a
   model and return data; step ordering belongs to `refine.py` and
   `pipeline.py`.

A quick check before opening a pull request:

```bash
grep -rn "^from \." lyricsmith/domain.py          # must print nothing
grep -rln "rich" lyricsmith/ | grep -v logger.py  # must print only cli.py
```

## Adding a processing step

Text steps follow the same shape. Taking a hypothetical punctuation fixer:

1. Write the pure function in `lyricsmith/utils/`, taking `Segment` values and
   plain parameters, never a `Config` object. That keeps it testable and
   reusable.
2. Declare its settings as a dataclass in `lyricsmith/config.py`, and add the
   section to `PipelineConfig` plus the nested mapping in `_build_pipeline`.
3. Add the same section to `config.yaml`, with the same defaults.
4. Call it from `lyricsmith/refine.py`, in the right place in the order.
5. Add the CLI flag in `lyricsmith/cli.py` if the step is worth toggling per
   run.
6. Test the function directly, and add a case in `test_refine.py` for the
   enabled and disabled paths.
7. Document it in [Processing chain](processing-chain.md) and
   [Configuration](configuration.md).

Audio steps are the same, except they live in `utils/preprocess.py` and are
called from `separator.py`, before or after the separation.

## Documentation

The site is built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/).

```bash
uv pip install -e ".[docs]"
mkdocs serve          # Live preview on http://127.0.0.1:8000
mkdocs build --strict # Fails on broken internal links
```

Pages live in `docs/`, and the navigation is declared in `mkdocs.yml`. A new
page has to be added to the `nav` section, otherwise `--strict` rejects the
build.

The look is not the stock Material one. The palette is declared as `custom` in
`mkdocs.yml` and defined in `docs/stylesheets/forge.css`: copper accents on a
warm slate background, monospace headings, flat surfaces and no web fonts, since
`font: false` keeps the system stack. Change the colors in that one file.

Keep the configuration reference in sync with the dataclasses in
`lyricsmith/config.py`. It is the one page that silently goes stale, since
nothing checks it.
