"""API gateway: global concurrency window + retry policy (revised Section III-C).

* `AcquireSlot(W)`: at most W attempts in flight across ALL agents (FIFO waiters).
* Retry: 429 / 503 are retried by the gateway with truncated exponential backoff
  and jitter, honouring Retry-After; the slot is released while backing off.
  Retries do not consume agent steps. After `max_retries` the call is returned to
  the agent as an exhausted transient failure.
* Every attempt (including retries) is logged and fed to the controller.
* Optional client-side token-bucket pacer (baseline); queue_wait then includes pacing delay.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..keyed import KeyedUniforms

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
    window: int             # limit W in force when the slot was granted
    in_use: int             # slots in use right after the grant (always <= window)


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


@dataclass(frozen=True)
class PacerConfig:
    """Client-side token-bucket pacer (baseline): at most `rate` attempts/s with bursts of `burst`."""
    rate: float
    burst: int


class TokenBucketPacer:
    """GCRA token bucket. Each acquire reserves its send time in FIFO order, so it is
    deterministic under the virtual clock. Retries are paced too (every attempt acquires)."""

    def __init__(self, cfg: PacerConfig):
        self.T = 1.0 / cfg.rate                  # emission interval
        self.tau = (cfg.burst - 1) * self.T      # burst tolerance
        self.tat = 0.0                           # theoretical arrival time

    async def acquire(self) -> None:
        now = asyncio.get_running_loop().time()
        tat = max(self.tat, now)
        t_allowed = max(now, tat - self.tau)
        self.tat = tat + self.T
        if t_allowed > now:
            await asyncio.sleep(t_allowed - now)


class SlotPool:
    """Counting semaphore whose limit can change at run time."""

    def __init__(self, limit: int):
        self.limit = limit
        self.in_use = 0
        self._waiters: deque[asyncio.Future] = deque()

    async def acquire(self) -> tuple[int, int]:
        """Wait for a slot. Returns (limit, in_use) as they were when the slot was GRANTED."""
        if self.in_use < self.limit and not self._waiters:
            self.in_use += 1
            return self.limit, self.in_use
        fut = asyncio.get_running_loop().create_future()
        self._waiters.append(fut)
        try:
            return await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():
                self.release()          # the slot had already been granted to us: give it back
            raise

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
                fut.set_result((self.limit, self.in_use))


class Gateway:
    def __init__(self, server: MockServer, controller: ACCController, retry: RetryConfig, seed: int,
                 pacer: PacerConfig | None = None):
        self.server = server
        self.controller = controller
        self.retry = retry
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)      # fallback stream when a call carries no request id
        self._jitter = KeyedUniforms(self.seed, tag=7)
        self.slots = SlotPool(controller.window)
        self.pacer = TokenBucketPacer(pacer) if pacer is not None else None
        self.attempts: list[AttemptRecord] = []
        self.calls: list[CallRecord] = []

    def _backoff(self, k: int, request: tuple | None = None) -> float:
        r = self.retry
        base = min(r.backoff_cap, r.backoff_base * (2 ** k))
        # jitter keyed by (seed, request, retry index) when a request id is given, so that it does not
        # depend on the order in which other calls happen to run (common random numbers across methods)
        if request is None:
            u = float(self.rng.random())
        else:
            agent, call_k = request
            u = self._jitter.get(int(agent), int(call_k) * (r.max_retries + 1) + k)
        return base * (0.5 + 0.5 * u)

    async def call(self, request: tuple | None = None) -> CallRecord:
        """`request` = optional (agent, call_index) identifying the logical call; it keys the server noise
        and the backoff jitter so both are order-independent (common random numbers across methods)."""
        loop = asyncio.get_running_loop()
        t_request = loop.time()
        queue_wait = 0.0
        n = 0
        while True:
            t_wait = loop.time()
            if self.pacer is not None:
                await self.pacer.acquire()
            window_now, in_use = await self.slots.acquire()     # grant-time limit and occupancy
            t_dispatch = loop.time()
            queue_wait += t_dispatch - t_wait
            try:
                resp = await self.server.handle(request)
                n += 1
                self.attempts.append(AttemptRecord(t_dispatch, resp.status, resp.latency, window_now, in_use))
                # Update the window BEFORE freeing this slot, so a shrink takes effect for the very slot being
                # released (otherwise it would be re-granted at the old limit; Algorithm 1 line 8 precedes
                # the next dispatch).
                if self.controller.record(resp.status, resp.latency, loop.time()):
                    self.slots.set_limit(self.controller.window)
            finally:
                self.slots.release()       # always freed, also if the server/controller raises
            if resp.status == 200:
                break
            if n > self.retry.max_retries:
                break
            delay = max(resp.retry_after or 0.0, self._backoff(n - 1, request))
            await asyncio.sleep(delay)
        rec = CallRecord(t_request, loop.time(), resp.status, n, queue_wait, resp.payload)
        self.calls.append(rec)
        return rec
