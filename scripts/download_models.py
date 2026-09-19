"""One shot initialization script.

Steps:
    1. Clone `heartlib` (HeartMuLa) when missing and install it in editable mode.
    2. Download the HeartTranscriptor weights from HuggingFace.
    3. Check the integrity of the downloaded files.
    4. Print a summary of the environment (GPU, VRAM, CUDA).

Usage:
    python scripts/download_models.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Make the `pipeline` package importable when the script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lyricsmith.config import Config  # noqa: E402
from lyricsmith.utils.gpu import check_gpu_status  # noqa: E402
from lyricsmith.utils.logger import console, get_logger  # noqa: E402

logger = get_logger("download_models")

HEARTLIB_REPO = "https://github.com/HeartMuLa/heartlib.git"
HF_MODEL_ID = "HeartMuLa/HeartTranscriptor-oss"


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    """Run an external command and raise when it fails.

    Args:
        cmd: Command and arguments.
        cwd: Working directory for the command.

    Raises:
        RuntimeError: When the command exits with a non zero status.
    """
    logger.info("$ %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def clone_and_install_heartlib(project_root: Path) -> Path:
    """Clone heartlib when needed, then install it in editable mode.

    Args:
        project_root: Directory receiving the `heartlib` clone.

    Returns:
        The path of the heartlib clone.
    """
    target = project_root / "heartlib"
    if target.exists():
        logger.info("heartlib already present: %s", target)
    else:
        console.rule("[cyan]1/4 Cloning heartlib")
        _run(["git", "clone", HEARTLIB_REPO, str(target)])

    console.rule("[cyan]Installing heartlib")
    _run(["uv", "pip", "install", "-e", str(target)])
    return target


def download_weights(model_path: Path) -> Path:
    """Download the HeartTranscriptor weights from HuggingFace.

    Args:
        model_path: Directory receiving the weights.

    Returns:
        The directory holding the downloaded weights.
    """
    console.rule("[cyan]2/4 Downloading the HuggingFace weights")
    model_path.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=HF_MODEL_ID,
            local_dir=str(model_path),
            local_dir_use_symlinks=False,
        )
    except ImportError:
        # Fall back to the `hf` command line when the library is unavailable.
        _run(["hf", "download", "--local_dir", str(model_path), HF_MODEL_ID])
    return model_path


def verify_integrity(model_path: Path) -> bool:
    """Check that at least one weight file was downloaded.

    Args:
        model_path: Directory holding the weights.

    Returns:
        True when the download looks complete.
    """
    console.rule("[cyan]3/4 Checking integrity")
    if not model_path.exists():
        logger.error("Model directory missing: %s", model_path)
        return False

    weight_files = [
        p
        for p in model_path.rglob("*")
        if p.suffix in {".safetensors", ".bin", ".pt", ".ckpt", ".pth"}
    ]
    if not weight_files:
        logger.error("No weight file found in %s", model_path)
        return False

    total_mb = sum(p.stat().st_size for p in weight_files) / (1024 * 1024)
    logger.info("%d weight file(s), %.1f MB in total.", len(weight_files), total_mb)
    return True


def print_environment() -> None:
    """Print a summary of the local environment."""
    console.rule("[cyan]4/4 Environment summary")
    status = check_gpu_status()
    console.print(f"CUDA available : {status['cuda_available']}")
    console.print(f"GPU            : {status['device_name']}")
    console.print(f"Total VRAM     : {status['vram_total_gb']} GB")
    console.print(f"Free VRAM      : {status['vram_free_gb']} GB")

    try:
        import torch

        console.print(f"PyTorch        : {torch.__version__}")
        console.print(f"CUDA (torch)   : {torch.version.cuda}")
    except ImportError:
        console.print("[yellow]PyTorch is not installed.[/yellow]")


def main() -> int:
    """Run every initialization step and return the process exit code."""
    config = Config.load()
    model_path = config.resolve_path(config.models.hearttranscriptor.model_path)

    try:
        clone_and_install_heartlib(config.project_root)
        download_weights(model_path)
        ok = verify_integrity(model_path)
        print_environment()
    except Exception as exc:
        console.print(f"[bold red]Initialization failed:[/bold red] {exc}")
        return 1

    if ok:
        console.print("\n[bold green]Initialization complete.[/bold green]")
        return 0
    console.print("\n[bold red]Initialization incomplete, weights are missing.[/bold red]")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
