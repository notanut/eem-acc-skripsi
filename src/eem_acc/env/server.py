"""Mock API server model (in-process, virtual time).

Mechanisms (all parameters are explicit; there are NO defaults on purpose, because
the values are experimental design decisions that the user must confirm):

* Token-bucket rate limit -> HTTP 429 with a Retry-After hint.
* Load-dependent latency: service time grows once in-flight calls exceed `capacity`.
* Hard overload cap -> HTTP 503 when in-flight calls reach `hard_cap`.
* Optional capacity events: at time t, capacity and hard_cap are multiplied by a
  scale factor (e.g. a mid-run capacity drop).

Check order: rate limit (429) first, then overload (503), then service.
Latency is fixed at arrival time from the load at that moment (a documented
simplification).
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ServerConfig:
    base_latency: float          # s, unloaded median service time
    latency_sigma: float         # lognormal sigma of per-call noise (0 = deterministic)
    capacity: int                # in-flight level above which latency degrades
    degradation: float           # latency x (1 + degradation * max(0, inflight/capacity - 1))
    hard_cap: int                # in-flight >= hard_cap -> 503
    rate_limit: float            # token refill rate (calls / s)
    burst: int                   # token bucket size
    retry_after: float           # s, hint attached to 429 responses
    error_latency: float         # s, time to produce a 429/503
    capacity_events: tuple = ()  # ((t, scale), ...) sorted by t; scales capacity and hard_cap


@dataclass
class Response:
    status: int
    latency: float               # server-side time for this attempt (s)
    retry_after: float | None = None
    payload: Any = None


class MockServer:
    def __init__(self, cfg: ServerConfig, seed: int):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.inflight = 0
        self.max_inflight_seen = 0
        self._tokens = float(cfg.burst)
        self._last_refill = 0.0
        self._events = sorted(cfg.capacity_events)
        self.counts = {200: 0, 429: 0, 503: 0}

    def _scale(self, now: float) -> float:
        s = 1.0
        for t, scale in self._events:
            if t <= now:
                s = scale
        return s

    def _refill(self, now: float) -> None:
        dt = max(0.0, now - self._last_refill)
        self._tokens = min(float(self.cfg.burst), self._tokens + dt * self.cfg.rate_limit)
        self._last_refill = now

    async def handle(self, request: Any = None) -> Response:
        cfg = self.cfg
        loop = asyncio.get_running_loop()
        now = loop.time()
        scale = self._scale(now)
        capacity = max(1, int(round(cfg.capacity * scale)))
        hard_cap = max(1, int(round(cfg.hard_cap * scale)))

        self._refill(now)
        if self._tokens < 1.0:
            self.counts[429] += 1
            await asyncio.sleep(cfg.error_latency)
            return Response(429, cfg.error_latency, retry_after=cfg.retry_after)
        if self.inflight >= hard_cap:
            self.counts[503] += 1
            await asyncio.sleep(cfg.error_latency)
            return Response(503, cfg.error_latency)

        self._tokens -= 1.0
        self.inflight += 1
        self.max_inflight_seen = max(self.max_inflight_seen, self.inflight)
        noise = math.exp(self.rng.normal(0.0, cfg.latency_sigma)) if cfg.latency_sigma > 0 else 1.0
        overload = max(0.0, self.inflight / capacity - 1.0)
        latency = cfg.base_latency * noise * (1.0 + cfg.degradation * overload)
        try:
            await asyncio.sleep(latency)
        finally:
            self.inflight -= 1
        self.counts[200] += 1
        return Response(200, latency)
