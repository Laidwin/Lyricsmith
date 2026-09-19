"""Convenience entry point, so the CLI runs from a clone without installing.

`python main.py transcribe song.mp3` and the installed `lyricsmith` command run
the exact same application, defined in `lyricsmith.cli`.
"""

from __future__ import annotations

from lyricsmith.cli import app

if __name__ == "__main__":
    app()
