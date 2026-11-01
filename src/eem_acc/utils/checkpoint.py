"""Checkpointing with integrity checks, rotation, and RNG-state capture.

A checkpoint file is: MAGIC (8 bytes) | sha256 of payload (32 bytes) | pickle payload.
`load_latest` verifies the checksum and falls back to the previous checkpoint if
the newest one is corrupt, so a disconnect mid-write can never break resuming.

Pickle is used only for our own checkpoint files (trusted input).
"""

from __future__ import annotations

import hashlib
import pickle
import random
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from .atomic import atomic_write_bytes

MAGIC = b"EEMACC01"
_NAME_RE = re.compile(r"^ckpt_(\d{8})\.pkl$")


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    try:
        import torch

        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
    except ImportError:
        pass
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        try:
            import torch

            torch.set_rng_state(state["torch"])
            if "torch_cuda" in state and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(state["torch_cuda"])
        except ImportError:
            pass


class CheckpointManager:
    """Save/load numbered checkpoints in `root`, keeping the newest `keep_last`."""

    def __init__(self, root: str | Path, keep_last: int = 3):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.keep_last = max(1, keep_last)

    def _path(self, step: int) -> Path:
        return self.root / f"ckpt_{step:08d}.pkl"

    def list_steps(self) -> list[int]:
        steps = []
        for p in self.root.iterdir():
            m = _NAME_RE.match(p.name)
            if m:
                steps.append(int(m.group(1)))
        return sorted(steps)

    def save(self, step: int, state: dict[str, Any], config: dict[str, Any] | None = None) -> Path:
        """Atomically write a checkpoint. `state` holds e.g. EEM store, ACC window,
        progress counters; RNG state and config are added automatically."""
        payload = pickle.dumps(
            {
                "step": step,
                "state": state,
                "config": config,
                "rng": capture_rng_state(),
                "saved_at": time.time(),
            },
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        blob = MAGIC + hashlib.sha256(payload).digest() + payload
        path = atomic_write_bytes(self._path(step), blob)
        self._rotate()
        return path

    def _rotate(self) -> None:
        for step in self.list_steps()[: -self.keep_last]:
            try:
                self._path(step).unlink()
            except OSError:
                pass

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        blob = path.read_bytes()
        if len(blob) < 40 or blob[:8] != MAGIC:
            raise ValueError(f"bad header: {path.name}")
        digest, payload = blob[8:40], blob[40:]
        if hashlib.sha256(payload).digest() != digest:
            raise ValueError(f"checksum mismatch: {path.name}")
        return pickle.loads(payload)

    def load_latest(self, restore_rng: bool = True) -> dict[str, Any] | None:
        """Return the newest valid checkpoint dict (or None). Corrupt files are skipped."""
        for step in reversed(self.list_steps()):
            try:
                ckpt = self._read(self._path(step))
            except Exception as e:  # corrupt / truncated -> try the previous one
                print(f"[checkpoint] skipping unreadable checkpoint {step}: {e}")
                continue
            if restore_rng:
                restore_rng_state(ckpt["rng"])
            return ckpt
        return None
