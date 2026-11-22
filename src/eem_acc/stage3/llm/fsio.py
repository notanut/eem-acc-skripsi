"""Small durable-file helpers (Drive FUSE: fsync reaches only the VM's local cache; the runner's completeness checks
handle what this cannot, S3.3a spec section 6)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def atomic_write(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def append_line(path: Path, line: str) -> None:
    """One JSON line per write, flushed and fsynced."""
    assert "\n" not in line
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_jsonl(path: Path) -> list[dict]:
    """Every complete, parseable line; a torn last line is skipped (callers that need strictness use their own reader)."""
    out = []
    p = Path(path)
    if not p.exists():
        return out
    for ln in p.read_text(encoding="utf-8").split("\n"):        # not splitlines(): U+2028 etc. are content
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out
