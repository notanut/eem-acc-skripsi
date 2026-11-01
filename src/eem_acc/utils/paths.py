"""Drive folder layout. Heavy outputs go to Drive, never to the repo."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/Skripsi"
SUBDIRS = ("checkpoints", "results", "logs", "figures", "datasets", "handoff", "code")


class Paths:
    """Resolves the project's Drive folders.

    Root comes from the argument, else $EEM_ACC_DRIVE_ROOT, else DEFAULT_DRIVE_ROOT.
    """

    def __init__(self, root: str | os.PathLike | None = None):
        self.root = Path(root or os.environ.get("EEM_ACC_DRIVE_ROOT", DEFAULT_DRIVE_ROOT))

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def datasets(self) -> Path:
        return self.root / "datasets"

    @property
    def handoff(self) -> Path:
        return self.root / "handoff"

    @property
    def code(self) -> Path:
        return self.root / "code"

    def ensure(self) -> "Paths":
        for d in SUBDIRS:
            (self.root / d).mkdir(parents=True, exist_ok=True)
        return self

    def __str__(self) -> str:
        return "\n".join([f"root: {self.root}"] + [f"  {d}/" for d in SUBDIRS])
