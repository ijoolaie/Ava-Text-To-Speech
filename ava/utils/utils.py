"""Small helpers shared across the package (local paths only — the project is self-contained)."""

import contextlib
import os
import time
from pathlib import Path

logger = None  # set lazily by callers that need it


def repo_root() -> Path:
    """Root of the Ava repository (two levels above this file)."""
    return Path(__file__).resolve().parent.parent.parent


def resolve_path(path: str | Path, base: Path | None = None) -> Path:
    """Resolve a path from a config or CLI.

    ``~`` is expanded; relative paths are resolved against ``base`` (typically the
    directory of the config file's parent, i.e. the repository root) when the file
    does not exist relative to the current working directory.
    """
    p = Path(os.path.expanduser(str(path)))
    if p.is_absolute() or p.exists():
        return p
    if base is not None:
        candidate = (base / p).resolve()
        if candidate.exists():
            return candidate
        return candidate  # let the caller get a clear FileNotFoundError
    return p


def size_of_dict(state: dict) -> int:
    """Approximate size in bytes of a (nested) dict of tensors."""
    total = 0
    stack = [state]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            stack.extend(item.values())
        elif hasattr(item, "numel"):
            total += item.numel() * item.element_size()
        elif isinstance(item, (int, float)):
            total += 8
    return total


class _Timer:
    elapsed_time_ms: int | None = None


@contextlib.contextmanager
def display_execution_time(title: str, print_output: bool = True):
    """Log how long the wrapped block takes; exposes `.elapsed_time_ms` on the timer."""
    timer = _Timer()
    t = time.monotonic()
    try:
        yield timer
    finally:
        timer.elapsed_time_ms = int((time.monotonic() - t) * 1000)
        if print_output:
            print(f"{title}: {timer.elapsed_time_ms} ms")
