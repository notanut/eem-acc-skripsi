"""Virtual-time asyncio event loop.

`loop.time()` is a simulated clock; whenever the loop would block waiting for the
next timer, the clock jumps straight to it. A simulation that "takes" 10 minutes
of API time therefore runs in milliseconds, and (with seeded RNGs) it is fully
deterministic: same seed -> identical event order and identical results.

Only code that uses `asyncio.sleep` / `loop.time()` is simulated. Do NOT use
real I/O (sockets, subprocesses) inside a virtual loop.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Awaitable, TypeVar

T = TypeVar("T")


class _VirtualSelector:
    def __init__(self, loop: "VirtualClockEventLoop", real_selector: Any):
        self._loop = loop
        self._sel = real_selector

    def select(self, timeout: float | None = None):
        if timeout is None:
            # nothing scheduled and nothing ready: a real loop would hang forever
            raise RuntimeError("deadlock in virtual-time loop: no scheduled events")
        if timeout > 0:
            self._loop._vtime += timeout
        return self._sel.select(0)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._sel, name)


class VirtualClockEventLoop(asyncio.SelectorEventLoop):
    def __init__(self) -> None:
        super().__init__()
        self._vtime = 0.0
        self._selector = _VirtualSelector(self, self._selector)

    def time(self) -> float:
        return self._vtime


def _run_here(coro: Awaitable[T]) -> T:
    loop = VirtualClockEventLoop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def run_virtual(coro: Awaitable[T]) -> T:
    """Run `coro` to completion on a fresh virtual-time loop.

    If the calling thread already runs an event loop (Jupyter/Colab cells do), a second loop cannot be started in
    that thread, so the simulation runs on a fresh loop in a short-lived worker thread and this call waits for it.
    The simulation is self-contained (virtual clock, seeded RNGs), so the result is identical either way."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _run_here(coro)
    box: dict[str, Any] = {}

    def work() -> None:
        try:
            box["value"] = _run_here(coro)
        except BaseException as e:      # re-raised in the caller's thread
            box["error"] = e

    t = threading.Thread(target=work, name="run_virtual", daemon=True)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box["value"]
