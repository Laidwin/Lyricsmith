"""Structured logging built on `rich`.

Provides a configurable logger (level driven by `--verbose` / `--quiet`) and a
shared rich `Console` so that panels, tables and progress bars look consistent
across the whole project.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from rich.console import Console
from rich.logging import RichHandler


def _ensure_utf8_streams() -> None:
    """Force stdout and stderr to UTF-8.

    Windows often falls back to cp1252 when the output is redirected, which
    raises `UnicodeEncodeError` on styled characters such as box drawing glyphs.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


_ensure_utf8_streams()

console = Console()

_LOGGER_NAME = "lyricsmith"
_CONFIGURED = False


def setup_logging(level: int | str = logging.INFO) -> logging.Logger:
    """Configure the project root logger with a `rich` handler.

    Idempotent: calling it several times never duplicates handlers.

    Args:
        level: Logging level applied to the project logger.

    Returns:
        The configured project logger.
    """
    global _CONFIGURED

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)

    if not _CONFIGURED:
        handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            show_path=False,
            markup=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
        logger.addHandler(handler)
        logger.propagate = False
        _CONFIGURED = True
    else:
        for handler in logger.handlers:
            handler.setLevel(level)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger of the project logger.

    Args:
        name: Name suffix. When ``None``, the project root logger is returned.

    Returns:
        The requested logger.
    """
    if not _CONFIGURED:
        setup_logging()
    if name is None:
        return logging.getLogger(_LOGGER_NAME)
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")


def set_level(level: int | str) -> None:
    """Change the logging level at runtime, for instance from a CLI flag."""
    setup_logging(level)


def print_panel(content: Any, title: str = "", style: str = "cyan") -> None:
    """Print a rich panel using the shared console."""
    from rich.panel import Panel

    console.print(Panel(content, title=title, border_style=style))
