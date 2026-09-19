"""GPU environment checks (CUDA and VRAM).

Every function degrades gracefully when PyTorch is missing or when no GPU is
available, so the project stays usable on a CPU only machine.
"""

from __future__ import annotations

from typing import Any

from ..exceptions import GPUError

_LAZY_LOAD_VRAM_THRESHOLD_GB = 10.0
_BYTES_PER_GB = 1024 ** 3


def _import_torch() -> Any | None:
    """Import torch defensively, returning None when it is not installed."""
    try:
        import torch
    except ImportError:
        return None
    return torch


def check_gpu_status() -> dict[str, Any]:
    """Return a summary of the GPU environment.

    Returns:
        A dictionary with the keys ``cuda_available`` (bool), ``device_name``
        (str), ``vram_total_gb`` (float), ``vram_free_gb`` (float) and
        ``recommended_lazy_load`` (bool, true below 10 GB of total VRAM).
    """
    torch = _import_torch()
    status: dict[str, Any] = {
        "cuda_available": False,
        "device_name": "N/A",
        "vram_total_gb": 0.0,
        "vram_free_gb": 0.0,
        "recommended_lazy_load": True,
    }

    if torch is None or not torch.cuda.is_available():
        return status

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    total = props.total_memory / _BYTES_PER_GB

    try:
        free_bytes, _ = torch.cuda.mem_get_info(device)
        free = free_bytes / _BYTES_PER_GB
    except (RuntimeError, AssertionError):
        reserved = torch.cuda.memory_reserved(device) / _BYTES_PER_GB
        free = max(total - reserved, 0.0)

    status.update(
        cuda_available=True,
        device_name=props.name,
        vram_total_gb=round(total, 2),
        vram_free_gb=round(free, 2),
        recommended_lazy_load=total < _LAZY_LOAD_VRAM_THRESHOLD_GB,
    )
    return status


def assert_cuda(min_vram_gb: float = 4.0) -> None:
    """Check that CUDA is available with enough VRAM.

    Args:
        min_vram_gb: Minimum total VRAM required, in gigabytes.

    Raises:
        GPUError: When CUDA is unavailable or total VRAM is insufficient.
    """
    status = check_gpu_status()
    if not status["cuda_available"]:
        raise GPUError(
            "CUDA is not available. Check the NVIDIA drivers, the CUDA runtime "
            "and that PyTorch was installed with GPU support (cuXXX build)."
        )
    if status["vram_total_gb"] < min_vram_gb:
        raise GPUError(
            f"Insufficient VRAM: {status['vram_total_gb']} GB detected, "
            f"{min_vram_gb} GB required on {status['device_name']}."
        )


def resolve_device(requested: str) -> str:
    """Resolve the effective device, falling back to CPU when CUDA is absent.

    Args:
        requested: ``"cuda"`` or ``"cpu"``, usually read from the configuration.

    Returns:
        The device that can actually be used.
    """
    if requested == "cpu":
        return "cpu"
    if check_gpu_status()["cuda_available"]:
        return "cuda"
    return "cpu"
