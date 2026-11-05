"""API gateway: global concurrency window + retry policy (revised Section III-C).

* `AcquireSlot(W)`: at most W attempts in flight across ALL agents (FIFO waiters).
* Retry: 429 / 503 are retried by the gateway with truncated exponential backoff
  and jitter, honouring Retry-After; the slot is released while backing off.
  Retries do not consume agent steps. After `max_retries` the call is returned to
  the agent as an exhausted transient failure.
* Every attempt (including retries) is logged and fed to the controller.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..acc import ACCController
from ..env import MockServer


@dataclass(frozen=True)
class RetryConfig:
    max_retries: int
    backoff_base: float     # s
    backoff_cap: float      # s


@dataclass
class AttemptRecord:
    t_dispatch: float
    status: int
    latency: float          # server-side latency of this attempt
    window: int             # W in force at dispatch
    in_use: int             # slots in use right after acquiring


@dataclass
class CallRecord:
    t_request: float
    t_done: float
    status: int             # final status (200, or last 429/503 if retries exhausted)
    attempts: int
    queue_wait: float       # total time waiting for a slot, summed over attempts
    payload: Any = None

    @property
    def ok(self) -> bool:
        return self.status == 200


class SlotPool:
    """Counting semaphore whose limit can change at run time."""

    def __init__(self, limit: int):
        self.limit = limit
        self.in_use = 0
        self._waiters: deque[asyncio.Future] = deque()

    async def acquire(self) -> None:
        if self.in_use < self.limit and not self._waiters:
            self.in_use += 1
            return
        fut = asyncio.get_running_loop().create_future()
        self._waiters.append(fut)
        await fut

    def release(self) -> None:
        self.in_use -= 1
        self._wake()

    def set_limit(self, limit: int) -> None:
        self.limit = limit
        self._wake()

    def _wake(self) -> None:
        while self._waiters and self.in_use < self.limit:
            fut = self._waiters.popleft()
            if not fut.done():
                self.in_use += 1
                fut.set_result(None)


class Gateway:
    def __init__(self, server: MockServer, controller: ACCController, retry: RetryConfig, seed: int):
        self.server = server
        self.controller = controller
        self.retry = retry
        self.rng = np.random.default_rng(seed)
        self.slots = SlotPool(controller.window)
        self.attempts: list[AttemptRecord] = []
        self.calls: list[CallRecord] = []

    def _backoff(self, k: int) -> float:
        r = self.retry
        base = min(r.backoff_cap, r.backoff_base * (2 ** k))
        return base * float(self.rng.uniform(0.5, 1.0))     # jitter

    async def call(self, request: Any = None) -> CallRecord:
        loop = asyncio.get_running_loop()
        t_request = loop.time()
        queue_wait = 0.0
        n = 0
        while True:
            t_wait = loop.time()
            await self.slots.acquire()
            t_dispatch = loop.time()
            queue_wait += t_dispatch - t_wait
            window_now, in_use = self.slots.limit, self.slots.in_use
            try:
                resp = await self.server.handle(request)
            finally:
                self.slots.release()
            n += 1
            self.attempts.append(AttemptRecord(t_dispatch, resp.status, resp.latency, window_now, in_use))
            if self.controller.record(resp.status, resp.latency, loop.time()):
                self.slots.set_limit(self.controller.window)
            if resp.status == 200:
                break
            if n > self.retry.max_retries:
                break
            delay = max(resp.retry_after or 0.0, self._backoff(n - 1))
            await asyncio.sleep(delay)
        rec = CallRecord(t_request, loop.time(), resp.status, n, queue_wait, resp.payload)
        self.calls.append(rec)
        return rec
