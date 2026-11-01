"""Per-unit result saving: one JSON file per unit (task / run), written atomically,
so partial results are never lost when a Colab session disconnects."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

from .atomic import atomic_write_json

_SAFE = re.compile(r"[^A-Za-z0-9._=-]+")


def _fname(unit_id: str) -> str:
    return _SAFE.sub("_", str(unit_id)) + ".json"


class ResultStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, unit_id: str) -> Path:
        return self.root / _fname(unit_id)

    def save(self, unit_id: str, result: dict[str, Any]) -> Path:
        return atomic_write_json(self.path(unit_id), {"unit_id": str(unit_id), "result": result})

    def exists(self, unit_id: str) -> bool:
        return self.path(unit_id).exists()

    def load(self, unit_id: str) -> dict[str, Any]:
        with open(self.path(unit_id)) as f:
            return json.load(f)["result"]

    def pending(self, unit_ids: Iterable[str]) -> list[str]:
        return [u for u in unit_ids if not self.exists(u)]

    def iter_results(self) -> Iterator[tuple[str, dict[str, Any]]]:
        for p in sorted(self.root.glob("*.json")):
            try:
                with open(p) as f:
                    d = json.load(f)
                yield d["unit_id"], d["result"]
            except (json.JSONDecodeError, KeyError):
                continue  # ignore unreadable file; the unit will simply be re-run
