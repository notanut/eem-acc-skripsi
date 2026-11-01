"""Progress logging with estimated time remaining.

ETA uses the rate measured in the *current session* only (units processed since
this logger started), so resumed runs don't produce a bogus ETA.
"""

from __future__ import annotations

import time
from pathlib import Path


def _fmt(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


class ProgressLogger:
    def __init__(self, total: int, name: str = "run", already_done: int = 0,
                 log_path: str | Path | None = None, min_interval_s: float = 5.0):
        self.total = total
        self.name = name
        self.done = already_done
        self._start_done = already_done
        self._t0 = time.time()
        self._last = 0.0
        self.min_interval_s = min_interval_s
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def eta_seconds(self) -> float | None:
        processed = self.done - self._start_done
        if processed <= 0:
            return None
        rate = processed / max(time.time() - self._t0, 1e-9)
        return (self.total - self.done) / rate

    def line(self) -> str:
        eta = self.eta_seconds()
        pct = 100.0 * self.done / max(self.total, 1)
        return (f"[{self.name}] {self.done}/{self.total} ({pct:.1f}%) "
                f"elapsed {_fmt(time.time() - self._t0)} "
                f"ETA {_fmt(eta) if eta is not None else '?'}")

    def update(self, n: int = 1, force: bool = False) -> None:
        self.done += n
        now = time.time()
        if force or self.done >= self.total or now - self._last >= self.min_interval_s:
            self._last = now
            msg = self.line()
            print(msg, flush=True)
            if self.log_path:
                with open(self.log_path, "a") as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
