"""Safe file writes: write to a temp file in the same directory, then rename.

os.replace is atomic on the same filesystem, so a disconnect mid-write leaves
either the old file or the new file, never a half-written one.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

TMP_SUFFIX = ".tmp"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def atomic_write_bytes(path: str | os.PathLike, data: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=TMP_SUFFIX)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def atomic_write_json(path: str | os.PathLike, obj: Any, indent: int | None = 2) -> Path:
    data = json.dumps(obj, indent=indent, default=_json_default, sort_keys=True).encode("utf-8")
    return atomic_write_bytes(path, data)


def clean_tmp_files(directory: str | os.PathLike) -> int:
    """Remove leftover *.tmp files from interrupted writes. Returns count removed."""
    directory = Path(directory)
    if not directory.exists():
        return 0
    removed = 0
    for p in directory.rglob(f"*{TMP_SUFFIX}"):
        try:
            p.unlink()
            removed += 1
        except OSError:
            pass
    return removed
