"""Small durable-file helpers (Drive FUSE: fsync reaches only the VM's local cache; the runner's completeness checks
handle what this cannot, S3.3a spec section 6)."""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any


def canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def log_fingerprint(log: list) -> str:
    """'<entries>:<sha256 prefix>' of an append-only log (S3.4 chain log; second opinion M6)."""
    import hashlib
    return f"{len(log)}:{hashlib.sha256(canon(log).encode()).hexdigest()[:12]}"


class StaleView(RuntimeError):
    """The file on Drive is older than the last bundle: another account's session has not propagated yet."""


def check_expected_log(log: list, expect) -> None:
    """`expect`: n or '<n>:<sha12>' from the last bundle. The log is append-only, so it must have at least n entries
    and (with a sha) start with exactly the bundle's n entries. A longer but still stale view cannot be detected."""
    if expect is None:
        return
    n = int(str(expect).split(":")[0])
    if len(log) < n:
        raise StaleView(f"the chain log on Drive has {len(log)} entries, the last bundle {n}: Drive has not caught "
                        "up (another account's session); nothing was written - wait a few minutes and run again")
    if ":" in str(expect) and log_fingerprint(log[:n]) != str(expect):
        raise ValueError(f"the chain log on Drive does not start with the last bundle's {n} entries: another chain or "
                         "another Drive folder; nothing was written - ask in the chat")


def atomic_write(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")   # unique: concurrent writers
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


def read_lines(path: Path) -> list[str | None]:
    """The file's lines split on b"\\n" only (not splitlines(): U+2028 etc. are content), each decoded on its own; a
    line that is not valid UTF-8 (a write torn inside a multi-byte character) is None. A final empty piece is dropped."""
    raw = Path(path).read_bytes()
    parts = raw.split(b"\n")
    if parts and parts[-1] == b"":
        parts.pop()
    out: list[str | None] = []
    for b in parts:
        try:
            out.append(b.decode("utf-8"))
        except UnicodeDecodeError:
            out.append(None)
    return out


def read_jsonl(path: Path) -> list[dict]:
    """Every complete, parseable line; torn or undecodable lines are skipped (strict callers use their own reader)."""
    out = []
    p = Path(path)
    if not p.exists():
        return out
    for ln in read_lines(p):
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out
