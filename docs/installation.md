# Installation

## Requirements

- **Python 3.10**
- **NVIDIA GPU** with at least 8 GB of VRAM, tested on an RTX 3060 Ti
- **CUDA 12.x** and up to date NVIDIA drivers
- **Git**, used to clone `heartlib`
- **ffmpeg**, recommended for decoding compressed formats such as mp3, m4a and ogg

Lyricsmith also runs on CPU, considerably slower. Set `device: cpu` under both
model sections of `config.yaml`, or simply let it be: the code falls back to CPU
automatically when CUDA is unavailable.

## Setting up the environment

The project uses [`uv`](https://github.com/astral-sh/uv), but plain `pip` works
just as well.

```bash
# 1. Create the virtual environment
uv venv --python 3.10
# Windows: .venv\Scripts\activate
# Linux and macOS: source .venv/bin/activate

# 2. Install PyTorch with the CUDA index matching your driver (CUDA 12.1 here)
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install the remaining dependencies
uv pip install -r requirements.txt
```

!!! warning "Install PyTorch first"
    Installing `requirements.txt` before PyTorch pulls the default CPU build
    from PyPI, and the pipeline then runs on CPU whatever `config.yaml` says.
    Install `torch` and `torchaudio` from the CUDA index first.

## Downloading the models

```bash
python scripts/download_models.py
```

The script clones `heartlib` next to the project and installs it in editable
mode, downloads the HeartTranscriptor weights from HuggingFace into
`./models/ckpt`, checks that at least one weight file arrived, then prints a
summary of the GPU environment.

The Demucs weights are not downloaded here: Demucs fetches its own checkpoint on
first use and caches it in your torch hub directory.

Set `HF_TOKEN` in a `.env` file, copied from `.env.example`, only if you need a
private model or higher HuggingFace rate limits.

## Verifying the installation

```bash
python main.py check-env
```

```
                   GPU environment
| CUDA available        | yes                        |
| GPU                   | NVIDIA GeForce RTX 3060 Ti |
| Total VRAM            | 8.0 GB                     |
| Free VRAM             | 6.97 GB                    |
| Lazy load recommended | yes                        |
```

If the checkpoint line reports `missing`, run `scripts/download_models.py`
again.

## Installing as a package

To get the `lyricsmith` command on your `PATH` instead of calling
`python main.py`:

```bash
uv pip install -e .
lyricsmith check-env
```

Both invocations run the same application. The development and documentation
extras are installed with `uv pip install -e ".[dev]"` and
`uv pip install -e ".[docs]"`.
