# Review request: RQ2 simulator, ACC controller and experiment runner (2026-11-14)

You are an independent senior reviewer (distributed systems + experimental methodology). Review the code and
experiment design below for a bachelor thesis / IEEE-style paper. Be skeptical and concrete. Do not praise; find problems.

## 1. Project
Paper: "Learning from API Execution Experience: Self-Reflective Agentic AI with Adaptive Concurrency Control"
(Bina Nusantara University). Two components: EEM (execution-experience memory for LLM agents; NOT part of this
review) and ACC, an AIMD controller of a global concurrency window W shared by all M agents calling one API.
RQ2 asks: does ACC protect the API (fewer 429/503, latency near target) without giving up throughput, compared with
fixed windows and a token-bucket client? RQ2 is evaluated in a deterministic virtual-time asyncio simulation with
scripted (non-LLM) agents. Public repo: https://github.com/notanut/eem-acc-skripsi (the code below is the reviewed state).

## 2. Methodology excerpts (the specification the code must match)
ACC (Section III-C, revised):
- W integer, W_min <= W <= W_max, W_min >= 1; W = max in-flight calls shared by all agents (AcquireSlot = semaphore).
- Updated once per control interval of n_c COMPLETED attempts, using only that interval's attempts.
- F = fraction of the interval's attempts answered 429/503.
- dL = (p95 server latency of successful attempts - L_target)/L_target; dL = 0 if no successful attempt.
- if F > 0: W <- floor(W(1 - beta F)); elif dL > eps: W <- floor(W(1 - alpha dL)); elif dL <= 0: W <- W+1;
  else hold (tolerance band). Clamp to [W_min, W_max]. At most one decrease per interval.
- Retry policy: the gateway retries 429/503 with truncated exponential backoff + jitter honouring Retry-After, up to
  r_max retries; retries consume no agent steps; exhausted retries = one failed step.
- Objective, Eq. (1), REPORTED not tuned on: J = success - lambda * mean_j max(0,(L_j - L_target)/L_target)
  - gamma * mean_j f_j, means over attempts j, f_j = 1 if attempt j got 429/503.

Approved RQ2 design (D10-D15):
- Regimes: CAP (capacity 8, hard cap 16, rate limit 200/s non-binding) and RATE (capacity 32, hard cap 64,
  rate limit 40/s burst 20). Base latency 0.2 s, lognormal sigma 0.25, degradation 1.5, Retry-After 0.5 s,
  error latency 0.02 s.
- Scenarios: CAP-steady, CAP-burst (think 0 during [10,15) s), CAP-drop (capacity x0.5 at 20 s, x1.0 at 40 s),
  RATE-steady; M in {4,16,64}; 150 calls per agent; think time exponential, mean 0.1 s; start jitter 1 s;
  retries <= 6, backoff 0.1 s capped at 4 s. L_target = 0.45 s (1.5 x unloaded p95).
- ACC: W0=8, W_min=1, W_max=64; dev grid alpha,beta in {.25,.5,.75}, eps in {.05,.1,.2}, n_c in {8,16,32}.
- Baselines: fixed W {4,16,64} + best fixed W from {1,2,4,8,16,32,64} (tuned on dev); token bucket + backoff
  (rate tuned on dev from {10,20,40,80}/s, burst 1); ablations no-429 branch, no-latency branch, additive-only.
- Protocol: dev seeds 0-4, test seeds 100-109; one frozen ACC config selected on dev and committed before test;
  selection = maximise per-cell-normalised goodput s.t. mean SLO violation <= 5% and mean call success >= 99%.

## 3. What changed in this revision (please verify each)
1. SlotPool.acquire returns (limit, in_use) at grant time; a cancelled waiter whose slot was already granted returns it.
2. The gateway applies a new window BEFORE releasing the finished attempt's slot; the slot is released in `finally`.
3. Common random numbers: server latency noise keyed by the logical call (agent, k) only; backoff jitter keyed by
   (agent, k, retry index); keyed values come from per-agent cached sequential streams (keyed.py); scripted agents
   draw exactly one think-time variate per call, regardless of phase.
4. floor() in the controller gets +1e-9 against float error (e.g. 12*(1-0.5*25/30) = 6.999999999999999).
5. ResultStore: hash-suffixed filenames when sanitisation changes an id; files must carry their own unit_id and sit
   under their own file name.
6. Metrics: mean_latency_excess_all_attempts (Eq. 1's definition), wasted_attempts_per_ok_call, objective_J().
7. SIM_VERSION and a per-result spec_hash; resume, select and test refuse results/frozen configs from another
   configuration, simulator version or results folder; select refuses after test results exist and refuses to
   overwrite a different frozen file without --force.
Tests: 123 passed, 1 skipped (CPU suite); 2 notebook tests pass separately. A mutation script reverts each of the 14
fixes; every reversion fails at least one test.

## 4. Known open DESIGN issues (pending the author's decision; give your opinion and any better option)
From an earlier internal review with 432 simulation runs:
- H1: in both regimes the throughput-optimal static window is about 8 (CAP capacity 8; RATE 40/s x 0.2 s = 8), and
  W0 = 8. A fixed W=8 then beats ACC on goodput and SLO in almost every cell, so the scenarios cannot show the
  benefit of adapting without knowing the right W.
- H2: the averaged SLO constraint (mean <= 5%) hides about 10% violation in every loaded cell (ACC start config
  sits at 0.0501 mean); with eps > 0 the tolerance band lets p95 sit up to (1+eps) L_target, i.e. above 5% violation
  by construction.
- H3: M=4 cells and CAP-burst do not discriminate (all methods with W >= 4 identical at M=4; closed-loop agents are
  already saturated, so burst = steady).
- M1: events at absolute times while run length depends on the method (makespan 48 s to 1979 s) dilute event effects.
- M2: token-bucket grid {10,20,40,80} is coarse; the SLO constraint pushes it to 20/s (a strawman).
- M3: normalisation by the best dev method per cell, which may be infeasible.
- The simulator has no timeouts, although the paper's retry policy mentions them.

## 5. Checklist
a. Bugs, race conditions, slot leaks or deadlocks in SlotPool/Gateway (set_limit growing/shrinking while waiters
   queue; release while in_use > limit; cancellation).
b. Does the controller implement exactly the spec in section 2 (interval semantics, branch order, clamp, one decrease
   per interval, p95 of successes only, dL = 0 with no success)?
c. Is the CRN pairing correct and complete; can any randomness still depend on scheduling order?
d. Do metrics measure what they claim (goodput, SLO violation, throttle rate, J, wasted attempts)? Is J computed
   exactly as Eq. (1)? Mixed per-call / per-attempt units?
e. Anything that could bias the comparison or leak test information into tuning: unfair or strawman baselines,
   selection criterion, normalisation, unseeded randomness, results from different code mixed together.
f. Reproducibility and resume safety (atomic writes, stamps, refusals that a legitimate workflow might trip over).
g. Tests that would still pass if the code they target were wrong.
h. The open design issues in section 4: which are real, and what concrete design change you would make.

## 6. Output format
Findings ranked by severity (Critical / High / Medium / Low). For each: location (file:line or design item),
explanation, concrete fix. Then: answers to section 4 (one short paragraph each). Then a final verdict:
"OK to run the dev phase" / "run after fixes (list)" / "do not run (why)".

## 7. Source code (reviewed state)

### `src/eem_acc/agent/gateway.py`

```python
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
```

### `src/eem_acc/keyed.py`

```python
"""Order-independent ("keyed") random draws for common random numbers (CRN).

`KeyedNormals(seed).get(agent, k)` returns the k-th standard-normal draw of agent `agent`'s own stream,
whatever order the (agent, k) pairs are requested in. Each agent's stream is one sequential numpy
Generator seeded by (seed, tag, agent) whose values are cached in a table that grows in blocks; this
gives the order independence of a per-draw keyed generator at O(1) amortised cost.
"""

from __future__ import annotations

import numpy as np

_BLOCK = 256


class _KeyedTable:
    def __init__(self, seed: int, tag: int, draw):
        self.seed, self.tag, self._draw = int(seed), int(tag), draw
        self._gens: dict[int, np.random.Generator] = {}
        self._tabs: dict[int, np.ndarray] = {}

    def get(self, agent: int, idx: int) -> float:
        if idx < 0 or agent < 0:
            raise ValueError("agent and index must be non-negative")
        tab = self._tabs.get(agent)
        if tab is None or idx >= len(tab):
            g = self._gens.setdefault(agent, np.random.default_rng([self.seed, self.tag, agent]))
            have = 0 if tab is None else len(tab)
            need = max(_BLOCK, (idx + 1 - have + _BLOCK - 1) // _BLOCK * _BLOCK)
            new = self._draw(g, need)
            tab = new if tab is None else np.concatenate([tab, new])
            self._tabs[agent] = tab
        return float(tab[idx])


class KeyedNormals(_KeyedTable):
    def __init__(self, seed: int, tag: int = 0):
        super().__init__(seed, tag, lambda g, n: g.standard_normal(n))


class KeyedUniforms(_KeyedTable):
    def __init__(self, seed: int, tag: int = 0):
        super().__init__(seed, tag, lambda g, n: g.random(n))
```

### `src/eem_acc/env/server.py`

```python
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

from ..keyed import KeyedNormals


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
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)      # fallback stream when no request id is given
        self._keyed = KeyedNormals(self.seed, tag=1)
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

    def _noise(self, request: Any) -> float:
        """Lognormal multiplicative noise. With a request id (agent, k) the draw is keyed by
        (seed, agent, k): independent of serving order AND of how many 429/503 attempts preceded the
        admitted one, so methods compared on the same seed see the same noise for the same logical call
        (common random numbers). Only admitted attempts draw noise, at most one per call."""
        if self.cfg.latency_sigma <= 0:
            return 1.0
        if request is None:
            z = self.rng.normal(0.0, 1.0)
        else:
            agent, k = request
            z = self._keyed.get(int(agent), int(k))
        return math.exp(self.cfg.latency_sigma * float(z))

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
        noise = self._noise(request)
        overload = max(0.0, self.inflight / capacity - 1.0)
        latency = cfg.base_latency * noise * (1.0 + cfg.degradation * overload)
        try:
            await asyncio.sleep(latency)
        finally:
            self.inflight -= 1
        self.counts[200] += 1
        return Response(200, latency)
```

### `src/eem_acc/acc/controller.py`

```python
"""Adaptive Concurrency Controller (AIMD), implementing revised Section III-C.

Once every `n_c` completed attempts (one control interval) the window W is updated
from that interval's attempts only:

    F   = fraction of attempts that returned 429 or 503
    dL  = (p95 server latency of the interval's *successful* attempts - L_target) / L_target
          (0 if the interval has no successful attempt)

    if use_throttle_branch and F > 0:          W <- max(W_min, floor(W * (1 - beta * F)))
    elif use_latency_branch and dL > epsilon:  W <- max(W_min, floor(W * (1 - alpha * dL)))
    elif (not use_latency_branch) or dL <= 0:  W <- min(W + 1, W_max)
    else:                                      hold   (tolerance band 0 < dL <= epsilon)

IMPLEMENTATION ASSUMPTION (not in the paper text, flagged in HANDOFF.md): the latency
percentile uses successful (HTTP 200) attempts only, because 429/503 responses carry
no service time.

Config flags exist so the same class also implements the ablations/baselines
(no-throttle-branch, no-latency-branch, additive-only, fixed window via `frozen`).
There are NO default values for the hyperparameters: they are experimental decisions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


_FLOOR_EPS = 1e-9   # guards floor() against float error, e.g. 12*(1-0.5*25/30) = 6.999999999999999 (exactly 7)


@dataclass(frozen=True)
class ACCConfig:
    alpha: float
    beta: float
    epsilon: float
    w0: int
    w_min: int
    w_max: int
    n_c: int
    l_target: float
    use_throttle_branch: bool = True
    use_latency_branch: bool = True
    multiplicative_decrease: bool = True   # False -> "additive-only" ablation (never decreases)
    frozen: bool = False                   # True -> fixed window baseline (W stays at w0)

    def __post_init__(self) -> None:
        if not (1 <= self.w_min <= self.w0 <= self.w_max):
            raise ValueError("need 1 <= w_min <= w0 <= w_max")
        if self.n_c < 1:
            raise ValueError("n_c must be >= 1")
        if self.l_target <= 0:
            raise ValueError("l_target must be > 0")


class ACCController:
    def __init__(self, cfg: ACCConfig):
        self.cfg = cfg
        self.window = int(cfg.w0)
        self._buf: list[tuple[int, float]] = []   # (status, server latency) of current interval
        self.n_updates = 0
        self.trace: list[tuple[float, int, float, float]] = []  # (t, W after update, F, dL)

    # -- signals -------------------------------------------------------------
    @staticmethod
    def throttle_rate(buf: list[tuple[int, float]]) -> float:
        return sum(1 for s, _ in buf if s in (429, 503)) / len(buf)

    def latency_degradation(self, buf: list[tuple[int, float]]) -> float:
        ok = [lat for s, lat in buf if s == 200]
        if not ok:
            return 0.0
        p95 = float(np.percentile(ok, 95))
        return (p95 - self.cfg.l_target) / self.cfg.l_target

    # -- update --------------------------------------------------------------
    def next_window(self, w: int, f: float, dl: float) -> int:
        c = self.cfg
        if c.use_throttle_branch and f > 0:
            if not c.multiplicative_decrease:
                return w
            return max(c.w_min, math.floor(w * (1.0 - c.beta * f) + _FLOOR_EPS))
        if c.use_latency_branch and dl > c.epsilon:
            if not c.multiplicative_decrease:
                return w
            return max(c.w_min, math.floor(w * (1.0 - c.alpha * dl) + _FLOOR_EPS))
        if (not c.use_latency_branch) or dl <= 0:
            return min(w + 1, c.w_max)
        return w

    def record(self, status: int, latency: float, now: float = 0.0) -> bool:
        """Feed one completed attempt. Returns True if the window was updated."""
        self._buf.append((status, float(latency)))
        if len(self._buf) < self.cfg.n_c:
            return False
        buf, self._buf = self._buf, []
        f = self.throttle_rate(buf)
        dl = self.latency_degradation(buf)
        if not self.cfg.frozen:
            self.window = self.next_window(self.window, f, dl)
        self.n_updates += 1
        self.trace.append((float(now), self.window, f, dl))
        return True

    # -- checkpointing -------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        return {"window": self.window, "buf": list(self._buf), "n_updates": self.n_updates,
                "trace": list(self.trace)}

    def load_state_dict(self, s: dict[str, Any]) -> None:
        self.window = int(s["window"])
        self._buf = [tuple(x) for x in s["buf"]]
        self.n_updates = int(s["n_updates"])
        self.trace = [tuple(x) for x in s["trace"]]
```

### `src/eem_acc/agent/scripted.py`

```python
"""Scripted (non-LLM) agents for isolating the ACC (RQ2) cheaply.

Each agent runs a sequential loop: think -> API call through the gateway ->
repeat, `calls_per_agent` times. Concurrency arises across the M agents.
`phases` gives the mean think time as a function of simulated time, which lets a
scenario switch between steady load and bursts (think time 0 = back-to-back).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np

from .gateway import Gateway


@dataclass(frozen=True)
class WorkloadConfig:
    n_agents: int
    calls_per_agent: int
    phases: tuple            # ((t_start, think_time_mean_s), ...) sorted by t_start
    start_jitter: float      # agents start at Uniform(0, start_jitter) s


def _think_mean(phases: tuple, now: float) -> float:
    mean = phases[0][1]
    for t, m in phases:
        if t <= now:
            mean = m
    return mean


async def _agent(gw: Gateway, wl: WorkloadConfig, rng: np.random.Generator, agent_id: int) -> None:
    loop = asyncio.get_running_loop()
    if wl.start_jitter > 0:
        await asyncio.sleep(float(rng.uniform(0, wl.start_jitter)))
    for k in range(wl.calls_per_agent):
        u = float(rng.exponential(1.0))                 # ALWAYS one draw per call: the stream never depends on timing
        mean = _think_mean(wl.phases, loop.time())
        if mean > 0:
            await asyncio.sleep(mean * u)
        await gw.call(request=(agent_id, k))


async def run_agents(gw: Gateway, wl: WorkloadConfig, seed: int) -> None:
    children = np.random.SeedSequence(seed).spawn(wl.n_agents)
    await asyncio.gather(*[_agent(gw, wl, np.random.default_rng(c), i) for i, c in enumerate(children)])
```

### `src/eem_acc/sim.py`

```python
"""One simulated scenario run: server + gateway(ACC) + scripted agents, virtual time.

`Scenario` is a plain dataclass tree, so `dataclasses.asdict(scenario)` is a JSON-able
config for `run_resumable` (one unit = one (scenario, method, seed) run).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .acc import ACCConfig, ACCController
from .agent import Gateway, PacerConfig, RetryConfig, WorkloadConfig, run_agents
from .env import MockServer, ServerConfig, run_virtual
from .metrics import compute_metrics


# Bump whenever simulator/gateway/controller/metrics semantics change: it is part of every result's stamp,
# so results produced by different code can never be silently mixed in one results folder.
SIM_VERSION = "3"   # 2: shrink-before-release, grant-time window logging, keyed (CRN) randomness, J metrics
                    # 3: server noise keyed by logical call (agent, k) only (not the attempt index);
                    #    keyed draws served from per-agent cached tables (eem_acc/keyed.py)


@dataclass(frozen=True)
class Scenario:
    server: ServerConfig
    workload: WorkloadConfig
    retry: RetryConfig
    acc: ACCConfig
    seed: int
    pacer: PacerConfig | None = None   # client-side token-bucket baseline (None = off)


async def _simulate(sc: Scenario) -> Gateway:
    ss = np.random.SeedSequence(sc.seed).spawn(3)
    seeds = [int(s.generate_state(1)[0]) for s in ss]
    server = MockServer(sc.server, seeds[0])
    ctrl = ACCController(sc.acc)
    gw = Gateway(server, ctrl, sc.retry, seeds[1], pacer=sc.pacer)
    await run_agents(gw, sc.workload, seeds[2])
    return gw


async def _run(sc: Scenario) -> dict[str, Any]:
    gw = await _simulate(sc)
    return {
        "metrics": compute_metrics(gw, sc.acc.l_target),
        "window_trace": [list(x) for x in gw.controller.trace],
    }


def simulate(sc: Scenario) -> Gateway:
    """Run a scenario and return the finished Gateway (attempt/call logs) for plotting."""
    return run_virtual(_simulate(sc))


def run_scenario(sc: Scenario) -> dict[str, Any]:
    return run_virtual(_run(sc))
```

### `src/eem_acc/metrics.py`

```python
"""Infrastructure metrics from a finished simulation (RQ2).

Latency metrics use *successful* attempts' server-side latency (the quantity the
SLO and the ACC refer to). End-to-end metrics include queueing and retries.
The combined objective J (Eq. 1 of the paper) needs lambda/gamma (undecided), so only its
components are stored; `objective_J` combines them exactly as Eq. 1:
    J = call_success_rate - lambda * mean_latency_excess_all_attempts - gamma * throttle_rate
where the latency excess averages max(0, (L_j - L_target)/L_target) over ALL attempts j (a 429/503
contributes its short error latency, i.e. ~0 excess), as in the paper. `mean_latency_excess` is the
variant over successful attempts only (it does not dilute under heavy throttling); report both and say which is J.
`slo_violation_rate` and lat_p* use successful attempts only.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .agent.gateway import Gateway


def _pct(x: list[float], q: float) -> float | None:
    return float(np.percentile(x, q)) if x else None


def compute_metrics(gw: Gateway, l_target: float) -> dict[str, Any]:
    att, calls = gw.attempts, gw.calls
    n_att = len(att)
    n200 = sum(a.status == 200 for a in att)
    n429 = sum(a.status == 429 for a in att)
    n503 = sum(a.status == 503 for a in att)
    lat = [a.latency for a in att if a.status == 200]
    e2e = [c.t_done - c.t_request for c in calls]
    ok_calls = [c for c in calls if c.ok]
    makespan = (max(c.t_done for c in calls) - min(c.t_request for c in calls)) if calls else 0.0
    w = [a.window for a in att]
    excess_all = [max(0.0, (a.latency - l_target) / l_target) for a in att]
    return {
        "n_calls": len(calls),
        "n_calls_ok": len(ok_calls),
        "call_success_rate": len(ok_calls) / len(calls) if calls else None,
        "n_exhausted_calls": len(calls) - len(ok_calls),
        "n_attempts": n_att,
        "n_200": n200, "n_429": n429, "n_503": n503,
        "throttle_rate": (n429 + n503) / n_att if n_att else None,
        "attempts_per_call": n_att / len(calls) if calls else None,
        "lat_p50": _pct(lat, 50), "lat_p95": _pct(lat, 95), "lat_p99": _pct(lat, 99),
        "slo_violation_rate": (sum(x > l_target for x in lat) / len(lat)) if lat else None,
        "mean_latency_excess": float(np.mean([max(0.0, (x - l_target) / l_target) for x in lat])) if lat else None,
        "mean_latency_excess_all_attempts": float(np.mean(excess_all)) if excess_all else None,   # J's definition
        "wasted_attempts_per_ok_call": (n_att - len(ok_calls)) / len(ok_calls) if ok_calls else None,
        "e2e_p50": _pct(e2e, 50), "e2e_p95": _pct(e2e, 95), "e2e_p99": _pct(e2e, 99),
        "mean_queue_wait": float(np.mean([c.queue_wait for c in calls])) if calls else None,
        "makespan": makespan,
        "goodput": len(ok_calls) / makespan if makespan > 0 else None,
        "window_mean": float(np.mean(w)) if w else None,
        "window_min": min(w) if w else None,
        "window_max": max(w) if w else None,
        "window_final": gw.controller.window,
        "server_max_inflight": gw.server.max_inflight_seen,
    }


def objective_J(m: dict[str, Any], lam: float, gam: float) -> float | None:
    """Eq. (1) of the paper from stored metrics (None if a component is undefined)."""
    parts = (m.get("call_success_rate"), m.get("mean_latency_excess_all_attempts"), m.get("throttle_rate"))
    if any(p is None for p in parts):
        return None
    return parts[0] - lam * parts[1] - gam * parts[2]
```

### `src/eem_acc/scenarios.py`

```python
"""RQ2 scenarios, methods and grids exactly as approved by the user (2026-11-05,
see docs/stage2_design_proposal.md, D10-D15). Changing any value here changes the
experiment: do it only with the user's approval and record it in HANDOFF.md.

Implementation assumptions flagged to the user:
* token-bucket baseline burst = 1 (pure pacing at `rate`);
* all methods use the same retry policy and workload.
"""

from __future__ import annotations

import itertools
from typing import Any

from .acc import ACCConfig
from .agent import PacerConfig, RetryConfig, WorkloadConfig
from .env import ServerConfig
from .sim import Scenario

# ---- D12 -------------------------------------------------------------------
L_TARGET = 0.45                      # 1.5 x unloaded server p95 (0.303 s, measured)

# ---- D10 / D11 -------------------------------------------------------------
_COMMON = dict(base_latency=0.2, latency_sigma=0.25, degradation=1.5, retry_after=0.5, error_latency=0.02)
SERVER_REGIMES = {
    "CAP":  dict(capacity=8, hard_cap=16, rate_limit=200.0, burst=100),
    "RATE": dict(capacity=32, hard_cap=64, rate_limit=40.0, burst=20),
}
# name -> (regime, think-time phases, capacity events)
SCENARIOS = {
    "CAP-steady":  ("CAP", ((0.0, 0.1),), ()),
    "CAP-burst":   ("CAP", ((0.0, 0.1), (10.0, 0.0), (15.0, 0.1)), ()),
    "CAP-drop":    ("CAP", ((0.0, 0.1),), ((20.0, 0.5), (40.0, 1.0))),
    "RATE-steady": ("RATE", ((0.0, 0.1),), ()),
}
M_LEVELS = (4, 16, 64)
CALLS_PER_AGENT = 150
START_JITTER = 1.0
RETRY = RetryConfig(max_retries=6, backoff_base=0.1, backoff_cap=4.0)

# ---- D13 -------------------------------------------------------------------
ACC_FIXED = dict(w0=8, w_min=1, w_max=64)
ACC_GRID = dict(alpha=(0.25, 0.5, 0.75), beta=(0.25, 0.5, 0.75), epsilon=(0.05, 0.1, 0.2), n_c=(8, 16, 32))
ACC_START = dict(alpha=0.5, beta=0.5, epsilon=0.1, n_c=16)     # start values (centre of the grid)

# ---- D14 -------------------------------------------------------------------
FIXED_W_REPORTED = (4, 16, 64)                 # low / mid / high
FIXED_W_DEV = (1, 2, 4, 8, 16, 32, 64)         # candidates for "best global fixed W"
PACER_RATES_DEV = (10.0, 20.0, 40.0, 80.0)     # candidates for the token-bucket baseline
PACER_BURST = 1                                # assumption (see module docstring)
ABLATIONS = {
    "acc_no429":    dict(use_throttle_branch=False),
    "acc_nolat":    dict(use_latency_branch=False),
    "acc_additive": dict(multiplicative_decrease=False),
}

# ---- D15 -------------------------------------------------------------------
DEV_SEEDS = tuple(range(0, 5))
TEST_SEEDS = tuple(range(100, 110))
MAX_SLO_VIOLATION = 0.05
MIN_CALL_SUCCESS = 0.99


# ---- method specs (plain dicts -> JSON-able, hashable via tag) -------------
def acc_method(alpha: float, beta: float, epsilon: float, n_c: int, **flags: Any) -> dict:
    return dict(kind="acc", alpha=alpha, beta=beta, epsilon=epsilon, n_c=n_c, **flags)


def fixed_method(w: int) -> dict:
    return dict(kind="fixed", w=int(w))


def pacer_method(rate: float) -> dict:
    return dict(kind="tokenbucket", rate=float(rate))


def method_tag(m: dict) -> str:
    k = m["kind"]
    if k == "fixed":
        return f"fixed_w{m['w']}"
    if k == "tokenbucket":
        return f"tb_r{m['rate']:g}"
    flags = "_no429" if m.get("use_throttle_branch") is False else ""
    flags += "_nolat" if m.get("use_latency_branch") is False else ""
    flags += "_additive" if m.get("multiplicative_decrease") is False else ""
    return f"acc_a{m['alpha']:g}_b{m['beta']:g}_e{m['epsilon']:g}_n{m['n_c']}{flags}"


def unit_id(scenario: str, m_agents: int, method: dict, seed: int) -> str:
    return f"{scenario}__M{m_agents}__{method_tag(method)}__s{seed}"


def build(scenario: str, m_agents: int, method: dict, seed: int) -> Scenario:
    regime, phases, events = SCENARIOS[scenario]
    server = ServerConfig(**_COMMON, **SERVER_REGIMES[regime], capacity_events=events)
    wl = WorkloadConfig(n_agents=m_agents, calls_per_agent=CALLS_PER_AGENT, phases=phases,
                        start_jitter=START_JITTER)
    k = method["kind"]
    pacer = None
    if k == "acc":
        flags = {f: method[f] for f in ("use_throttle_branch", "use_latency_branch", "multiplicative_decrease")
                 if f in method}
        acc = ACCConfig(alpha=method["alpha"], beta=method["beta"], epsilon=method["epsilon"],
                        n_c=method["n_c"], l_target=L_TARGET, **ACC_FIXED, **flags)
    elif k == "fixed":
        w = method["w"]
        acc = ACCConfig(alpha=0.5, beta=0.5, epsilon=0.1, n_c=16, l_target=L_TARGET,
                        w0=w, w_min=1, w_max=max(64, w), frozen=True)
    elif k == "tokenbucket":
        acc = ACCConfig(alpha=0.5, beta=0.5, epsilon=0.1, n_c=16, l_target=L_TARGET,
                        w0=64, w_min=1, w_max=64, frozen=True)   # window effectively unconstrained
        pacer = PacerConfig(rate=method["rate"], burst=PACER_BURST)
    else:
        raise ValueError(f"unknown method kind: {k}")
    return Scenario(server=server, workload=wl, retry=RETRY, acc=acc, seed=seed, pacer=pacer)


# ---- phase unit lists --------------------------------------------------------
def acc_grid_methods() -> list[dict]:
    keys = list(ACC_GRID)
    return [acc_method(**dict(zip(keys, vals))) for vals in itertools.product(*ACC_GRID.values())]


def dev_methods() -> list[dict]:
    return acc_grid_methods() + [fixed_method(w) for w in FIXED_W_DEV] + [pacer_method(r) for r in PACER_RATES_DEV]


def test_methods(frozen: dict) -> list[dict]:
    """Methods of the test phase, from the frozen dev selection (configs/rq2_frozen.json)."""
    a = frozen["acc"]
    ms = [acc_method(**a)]
    ms += [acc_method(**a, **flags) for flags in ABLATIONS.values()]
    ms += [fixed_method(w) for w in FIXED_W_REPORTED]
    best_w = frozen["best_fixed_w"]
    if best_w not in FIXED_W_REPORTED:
        ms.append(fixed_method(best_w))
    ms.append(pacer_method(frozen["best_pacer_rate"]))
    return ms


def units(methods: list[dict], seeds: tuple[int, ...]) -> list[tuple[str, str, int, dict, int]]:
    """(unit_id, scenario, M, method, seed) in a fixed order (cheap/light cells first)."""
    out = []
    for seed in seeds:
        for scn in SCENARIOS:
            for m in M_LEVELS:
                for meth in methods:
                    out.append((unit_id(scn, m, meth, seed), scn, m, meth, seed))
    return out
```

### `src/eem_acc/analysis.py`

```python
"""Load RQ2 results and select the frozen configuration on the DEV seeds (criterion D15-b).

Criterion (approved): maximise goodput subject to
    mean SLO-violation rate <= MAX_SLO_VIOLATION   and   mean call success >= MIN_CALL_SUCCESS,
means taken over all dev cells (scenario x M), after averaging over dev seeds.

Interpretation made here (flagged to the user): goodput differs by an order of magnitude
across cells (M=4 vs M=64), so it is normalised per cell by the best seed-averaged goodput
any dev method achieves in that cell; a method's score is the mean normalised goodput over
cells. A candidate must have a result in EVERY dev cell to be eligible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from . import scenarios as S
from .utils import ResultStore

METRIC_COLS = ["goodput", "slo_violation_rate", "call_success_rate", "throttle_rate",
               "mean_latency_excess", "makespan", "lat_p95", "n_429", "n_503"]


def load_results(units_dir: str | Path) -> pd.DataFrame:
    rows = []
    for uid, res in ResultStore(units_dir).iter_results():
        meta = res["meta"]
        row = {"unit_id": uid, "scenario": meta["scenario"], "M": meta["M"], "seed": meta["seed"],
               "tag": S.method_tag(meta["method"]), **{f"m_{k}": v for k, v in meta["method"].items()}}
        row.update(res["metrics"])
        rows.append(row)
    return pd.DataFrame(rows)


def cell_means(df: pd.DataFrame) -> pd.DataFrame:
    """Seed-averaged metrics per (scenario, M, method)."""
    d = df.copy()
    d["slo_violation_rate"] = d["slo_violation_rate"].fillna(1.0)   # no successful attempt = worst case
    d["call_success_rate"] = d["call_success_rate"].fillna(0.0)
    keys = ["scenario", "M", "tag"]
    extra = [c for c in d.columns if c.startswith("m_")]
    g = d.groupby(keys)
    out = g[["goodput", "slo_violation_rate", "call_success_rate", "throttle_rate"]].mean()
    out["n_seeds"] = g.size()
    for c in extra:
        out[c] = g[c].first()
    return out.reset_index()


def score_methods(df: pd.DataFrame) -> pd.DataFrame:
    """One row per method: mean normalised goodput, mean SLO violation, mean success, feasibility."""
    cells = cell_means(df)
    best = cells.groupby(["scenario", "M"])["goodput"].transform("max")
    cells["norm_goodput"] = cells["goodput"] / best
    n_cells = cells[["scenario", "M"]].drop_duplicates().shape[0]
    g = cells.groupby("tag")
    out = pd.DataFrame({
        "score": g["norm_goodput"].mean(),
        "mean_slo_violation": g["slo_violation_rate"].mean(),
        "mean_call_success": g["call_success_rate"].mean(),
        "n_cells": g.size(),
    })
    first = g[[c for c in cells.columns if c.startswith("m_")]].first()
    out = out.join(first).reset_index()
    out["complete"] = out["n_cells"] == n_cells
    out["feasible"] = (out["complete"] & (out["mean_slo_violation"] <= S.MAX_SLO_VIOLATION)
                       & (out["mean_call_success"] >= S.MIN_CALL_SUCCESS))
    return out.sort_values(["feasible", "score", "mean_slo_violation", "tag"],
                           ascending=[False, False, True, True]).reset_index(drop=True)


def select_frozen(df: pd.DataFrame) -> dict[str, Any]:
    """Pick the best feasible ACC config, fixed window and pacer rate from dev results.
    Returns {"ok": bool, ...}; if a family has no feasible candidate, ok=False and the
    best-infeasible rows are returned so the user can decide (nothing is silently relaxed)."""
    sc = score_methods(df)
    fams = {"acc": sc[sc["m_kind"] == "acc"], "fixed": sc[sc["m_kind"] == "fixed"],
            "tokenbucket": sc[sc["m_kind"] == "tokenbucket"]}
    result: dict[str, Any] = {"ok": True, "criterion": {
        "objective": "maximise mean per-cell-normalised goodput",
        "max_slo_violation": S.MAX_SLO_VIOLATION, "min_call_success": S.MIN_CALL_SUCCESS},
        "dev_seeds": sorted(int(x) for x in df["seed"].unique()), "n_dev_runs": int(len(df)), "top": {}}
    for name, t in fams.items():
        feasible = t[t["feasible"]]
        result["top"][name] = t.head(5)[["tag", "score", "mean_slo_violation", "mean_call_success", "feasible"]].to_dict("records")
        if feasible.empty:
            result["ok"] = False
            continue
        b = feasible.iloc[0]
        if name == "acc":
            result["acc"] = {k: (int(b[f"m_{k}"]) if k == "n_c" else float(b[f"m_{k}"]))
                             for k in ("alpha", "beta", "epsilon", "n_c")}
        elif name == "fixed":
            result["best_fixed_w"] = int(b["m_w"])
        else:
            result["best_pacer_rate"] = float(b["m_rate"])
    return result
```

### `experiments/run_rq2.py`

```python
"""RQ2 runner (scripted agents, simulated server). Resumable: re-running continues.

Phases (approved design, docs/stage2_design_proposal.md):
  dev     5,520 runs: 81 ACC configs + 7 fixed windows + 4 token-bucket rates, dev seeds 0-4
  select  choose the frozen configuration from dev results -> configs/rq2_frozen.json
  test    frozen ACC + 3 ablations + fixed {4,16,64,best} + tuned token bucket, test seeds 100-109

Usage (repo root):
  PYTHONPATH=src python experiments/run_rq2.py --phase dev    --root <ROOT>
  PYTHONPATH=src python experiments/run_rq2.py --phase select --root <ROOT>
  PYTHONPATH=src python experiments/run_rq2.py --phase test   --root <ROOT>
<ROOT> is the Drive folder in Colab (e.g. /content/drive/MyDrive/Skripsi) or any local folder.
Outputs: <ROOT>/results/rq2/<phase>/{units,checkpoints}/, <ROOT>/logs/rq2_<phase>.log
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

from eem_acc import scenarios as S
from eem_acc.analysis import load_results, score_methods, select_frozen
from eem_acc.sim import SIM_VERSION, run_scenario
from eem_acc.utils import ResultStore, atomic_write_json, run_resumable

import numpy as np

REPO = Path(__file__).resolve().parents[1]
FROZEN_PATH = REPO / "configs" / "rq2_frozen.json"


def phase_units(phase: str, frozen: dict | None):
    if phase == "dev":
        return S.units(S.dev_methods(), S.DEV_SEEDS), False
    if phase == "test":
        assert frozen is not None, "test phase needs configs/rq2_frozen.json (run --phase select first)"
        return S.units(S.test_methods(frozen), S.TEST_SEEDS), True
    raise ValueError(phase)


def spec_hash(scn: str, m: int, meth: dict, seed: int) -> str:
    """Hash of the FULL resolved configuration of one run + simulator version. Stored in every result;
    a resume that would mix results from a different configuration is refused."""
    blob = json.dumps({"sim": SIM_VERSION, "scenario": dataclasses.asdict(S.build(scn, m, meth, seed))},
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def frozen_hash(frozen: dict | None) -> str | None:
    return None if frozen is None else hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()[:16]


_SRC_DIRS = ("sim.py", "metrics.py", "keyed.py", "scenarios.py", "env", "agent", "acc")


def source_hash() -> str:
    """Hash of the simulator source (informational stamp; SIM_VERSION is the enforced one)."""
    h = hashlib.sha256()
    pkg = REPO / "src" / "eem_acc"
    for name in _SRC_DIRS:
        p = pkg / name
        for f in sorted([p] if p.is_file() else p.rglob("*.py")):
            h.update(f.relative_to(pkg).as_posix().encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


def check_results(phase: str, root: Path, frozen: dict | None) -> dict[str, str]:
    """ONE pass over every result file of the phase. Every file must belong to the phase's FULL unit list and
    carry that unit's current spec_hash; otherwise ValueError (mixed code/config versions, or leftovers of an
    older grid that load_results would silently read). Returns {unit_id: spec_hash} of the valid results."""
    units, _ = phase_units(phase, frozen)
    expected = {u[0]: spec_hash(*u[1:]) for u in units}
    store = ResultStore(root / "results" / "rq2" / phase / "units")
    ok, bad = {}, []
    for uid, res in store.iter_results():
        h = res.get("meta", {}).get("spec_hash") if isinstance(res, dict) else None
        if uid in expected and h == expected[uid]:
            ok[uid] = h
        else:
            bad.append(uid)
    if bad:
        raise ValueError(
            f"{len(bad)} result(s) in {store.root} do not match the current configuration/simulator version "
            f"(SIM_VERSION={SIM_VERSION}) or are not part of this phase's grid (e.g. {bad[0]}). "
            f"Use a new --root (recommended) or delete that folder.")
    return ok


def digest(hashes: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(sorted(hashes.items())).encode()).hexdigest()[:16]


def run_phase(phase: str, root: Path, limit: int | None = None, frozen: dict | None = None):
    units, store_trace = phase_units(phase, frozen)
    if limit:
        units = units[:limit]
    index = {u[0]: u for u in units}
    fz = frozen_hash(frozen)

    check_results(phase, root, frozen)   # refuse to resume on top of results from other code/config
    stamp = {"sim_version": SIM_VERSION, "frozen_hash": fz, "numpy": np.__version__, "source_hash": source_hash()}

    def run_unit(uid: str) -> dict:
        _, scn, m, meth, seed = index[uid]
        r = run_scenario(S.build(scn, m, meth, seed))
        out = {"meta": {"scenario": scn, "M": m, "method": meth, "seed": seed,
                        "spec_hash": spec_hash(scn, m, meth, seed), **stamp},
               "metrics": r["metrics"]}
        if store_trace:
            out["window_trace"] = r["window_trace"]
        return out

    config = {"phase": phase, "design": "docs/stage2_design_proposal.md@approved-2026-11-05", "frozen": frozen}
    return run_resumable([u[0] for u in units], run_unit, root / "results" / "rq2" / phase,
                         config=config, stateful=False, checkpoint_every=50,
                         log_path=root / "logs" / f"rq2_{phase}.log", progress_interval_s=15.0)


def do_select(root: Path, force: bool = False) -> int:
    test_units = root / "results" / "rq2" / "test" / "units"
    if test_units.exists() and any(test_units.glob("*.json")):
        print("[select] refusing: test results already exist in this folder. Re-selecting after seeing test "
              "results would invalidate the dev/test split. Use a fresh --root for a new experiment.")
        return 4
    try:
        hashes = check_results("dev", root, None)
    except ValueError as e:
        print(f"[select] refusing: {e}")
        return 5
    expected = len(S.units(S.dev_methods(), S.DEV_SEEDS))
    if len(hashes) != expected:
        print(f"[select] refusing: {len(hashes)} of {expected} dev runs present. Finish --phase dev first.")
        return 2
    df = load_results(root / "results" / "rq2" / "dev" / "units")
    res = select_frozen(df)
    res = {**res, "sim_version": SIM_VERSION, "dev_digest": digest(hashes)}
    print(json.dumps(res, indent=2, default=str))
    if not res["ok"]:
        print("[select] NO feasible candidate in at least one family; nothing written. Ask the user how to proceed.")
        return 3
    if FROZEN_PATH.exists() and not force:
        old = json.loads(FROZEN_PATH.read_text())
        if json.loads(json.dumps(res, default=str)) != old:
            print(f"[select] refusing: {FROZEN_PATH} already exists with DIFFERENT content (another root or "
                  f"an older run). Re-run with --force only if you really mean to replace it.")
            return 6
    FROZEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(FROZEN_PATH, res)
    atomic_write_json(root / "results" / "rq2" / "dev_scores.json", score_methods(df).to_dict("records"))
    atomic_write_json(root / "results" / "rq2" / "rq2_frozen.json", res)   # Drive copy (Colab clones are ephemeral)
    print(f"[select] wrote {FROZEN_PATH} (commit it to the repo BEFORE running the test phase)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["dev", "select", "test"], required=True)
    ap.add_argument("--root", default="outputs", help="Drive folder (Colab) or local output folder")
    ap.add_argument("--limit", type=int, default=None, help="only the first N units (smoke tests)")
    ap.add_argument("--force", action="store_true", help="select: replace a different configs/rq2_frozen.json")
    args = ap.parse_args()
    root = Path(args.root)
    if args.phase == "select":
        return do_select(root, args.force)
    frozen = None
    if args.phase == "test":
        frozen, msg = load_frozen_for_test(root)
        if frozen is None:
            print(f"[test] refusing: {msg}")
            return 7
        print(f"[test] {msg}")
    run_phase(args.phase, root, args.limit, frozen)
    return 0


def load_frozen_for_test(root: Path) -> tuple[dict | None, str]:
    """The test phase uses the root's frozen copy; the repo copy (if any) must be identical, its simulator
    version must be the current one, and it must have been selected from THIS root's dev results."""
    drive = root / "results" / "rq2" / "rq2_frozen.json"
    copies = [p for p in (drive, FROZEN_PATH) if p.exists()]
    if not copies:
        return None, "no frozen configuration found; run --phase select first"
    frozen = json.loads(copies[0].read_text())
    if len(copies) == 2 and json.loads(copies[1].read_text()) != frozen:
        return None, f"{drive} and {FROZEN_PATH} differ; they must be the same selection"
    if frozen.get("sim_version") != SIM_VERSION:
        return None, f"frozen configuration was selected with SIM_VERSION={frozen.get('sim_version')}, now {SIM_VERSION}"
    try:
        dev = check_results("dev", root, None)
    except ValueError as e:
        return None, f"dev results invalid: {e}"
    if frozen.get("dev_digest") != digest(dev):
        return None, "frozen configuration was not selected from this root's (complete, current) dev results"
    return frozen, f"frozen configuration from {copies[0]} (hash {frozen_hash(frozen)}, dev digest ok)"


if __name__ == "__main__":
    sys.exit(main())
```

### `src/eem_acc/utils/results.py`

```python
"""Per-unit result saving: one JSON file per unit (task / run), written atomically,
so partial results are never lost when a Colab session disconnects."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

from .atomic import atomic_write_json

_SAFE = re.compile(r"[^A-Za-z0-9._=-]+")


def _fname(unit_id: str) -> str:
    uid = str(unit_id)
    safe = _SAFE.sub("_", uid)
    if safe != uid:     # sanitising changed the id: add a hash so that e.g. "a/b" and "a_b" cannot collide
        safe += "-" + hashlib.sha1(uid.encode()).hexdigest()[:8]
    return safe + ".json"


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

    def is_valid(self, unit_id: str) -> bool:
        """True only if the result file exists AND parses (a file truncated by a disconnect is not a result)."""
        try:
            with open(self.path(unit_id)) as f:
                d = json.load(f)
            return isinstance(d, dict) and "result" in d and d.get("unit_id") == str(unit_id)
        except (OSError, json.JSONDecodeError):
            return False

    def load(self, unit_id: str) -> dict[str, Any]:
        with open(self.path(unit_id)) as f:
            d = json.load(f)
        if d.get("unit_id") != str(unit_id):
            raise ValueError(f"result file {self.path(unit_id).name} belongs to unit {d.get('unit_id')!r}, not {unit_id!r}")
        return d["result"]

    def pending(self, unit_ids: Iterable[str]) -> list[str]:
        return [u for u in unit_ids if not self.is_valid(u)]

    def iter_results(self) -> Iterator[tuple[str, dict[str, Any]]]:
        for p in sorted(self.root.glob("*.json")):
            try:
                with open(p) as f:
                    d = json.load(f)
                if not isinstance(d, dict) or p.name != _fname(d["unit_id"]):
                    continue    # not a result, or a result copied under another unit's file name
                yield d["unit_id"], d["result"]
            except (json.JSONDecodeError, KeyError, TypeError):
                continue  # ignore unreadable file; the unit will simply be re-run
```

### `src/eem_acc/utils/runner.py`

```python
"""Resumable experiment runner.

Usage
-----
    run_resumable(
        unit_ids, run_unit, run_dir=paths.results / "exp1_eem",
        config=cfg, stateful=True,
        get_state=lambda: {"eem": eem.state_dict(), "acc": acc.state_dict()},
        set_state=lambda s: (eem.load_state_dict(s["eem"]), acc.load_state_dict(s["acc"])),
    )

Re-running the same call after a crash/disconnect continues where it stopped.

stateless (default): units are independent. Any unit with a readable saved result is skipped
    (a corrupted result file is recomputed).
stateful: later units depend on state built by earlier ones (e.g. EEM memory grows
    across tasks). Only units covered by the latest valid checkpoint count as done;
    units finished after it are re-run, so state and results stay consistent.
    Use checkpoint_every=1 to lose nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable

from .atomic import clean_tmp_files
from .checkpoint import CheckpointManager
from .progress import ProgressLogger
from .results import ResultStore


def _canon(cfg: Any) -> str:
    return json.dumps(cfg, sort_keys=True, default=str)


def run_resumable(
    unit_ids: Iterable[str],
    run_unit: Callable[[str], dict[str, Any]],
    run_dir: str | Path,
    *,
    config: dict[str, Any] | None = None,
    stateful: bool = False,
    get_state: Callable[[], Any] | None = None,
    set_state: Callable[[Any], None] | None = None,
    checkpoint_every: int = 1,
    keep_last: int = 3,
    log_path: str | Path | None = None,
    progress_interval_s: float = 5.0,
    allow_config_change: bool = False,
) -> ResultStore:
    run_dir = Path(run_dir)
    ids = [str(u) for u in unit_ids]
    clean_tmp_files(run_dir)  # leftovers from an interrupted write
    store = ResultStore(run_dir / "units")
    ckpt = CheckpointManager(run_dir / "checkpoints", keep_last=keep_last)

    latest = ckpt.load_latest()
    ckpt_done: set[str] = set()
    counter = 0
    if latest is not None:
        if not allow_config_change and _canon(latest["config"]) != _canon(config):
            raise ValueError(
                f"Config differs from the checkpoint in {run_dir}. Use a new run_dir, "
                "or pass allow_config_change=True if the change is harmless."
            )
        ckpt_done = set(latest["state"]["done"])
        counter = latest["step"]
        if set_state is not None and latest["state"].get("user") is not None:
            set_state(latest["state"]["user"])
        print(f"[resume] checkpoint step {counter}: {len(ckpt_done)} units done")

    if stateful:
        done = ckpt_done & set(ids)
    else:
        # a unit counts as done only if its result file is readable; a corrupted file is recomputed
        done = {u for u in ids if store.is_valid(u)}

    progress = ProgressLogger(len(ids), name=run_dir.name, already_done=len(done),
                              log_path=log_path, min_interval_s=progress_interval_s)

    def _checkpoint() -> None:
        ckpt.save(counter, {"done": sorted(done), "user": get_state() if get_state else None},
                  config=config)

    for u in ids:
        if u in done:
            continue
        result = run_unit(u)
        store.save(u, result)
        done.add(u)
        counter += 1
        progress.update()
        if counter % checkpoint_every == 0:
            _checkpoint()

    _checkpoint()  # final state (same step number overwrites atomically)
    return store
```

### `tests/test_review_fixes.py`

```python
"""Tests for the review fixes: CRN, shrink-before-release, grant-time logging, J metrics, result stamping."""

import asyncio
import json

import numpy as np

import pytest

from eem_acc import scenarios as S
from eem_acc.agent import SlotPool
from eem_acc.env import MockServer, run_virtual
from eem_acc.metrics import compute_metrics, objective_J
from eem_acc.sim import SIM_VERSION, run_scenario, simulate
import experiments.run_rq2 as run_rq2  # noqa: E402  (conftest/pyproject put repo root on path via test_reproducibility usage)


def _server(seed=3):
    return MockServer(S.build("CAP-steady", 4, S.fixed_method(4), seed).server, seed)


def test_noise_is_keyed_by_request_not_by_call_order():
    s1, s2 = _server(), _server()
    reqs = [(0, 0), (1, 0), (2, 5), (3, 1)]
    a = [s1._noise(r) for r in reqs]
    b = [s2._noise(r) for r in reversed(reqs)][::-1]
    assert a == b                                   # order independent
    assert len(set(a)) == len(a)                    # different requests -> different noise
    assert s1._noise((0, 0)) == s1._noise((0, 0))   # same request -> same noise


def test_crn_same_call_same_latency_across_methods():
    """The per-request pairing (not just the multiset) must hold: below capacity, logical call (agent, k) gets
    the same latency under two methods even though the calls are served in a different order and with a
    different number of rejected attempts before admission."""
    def lat_by_call(meth, seed=7):
        sc = S.build("CAP-steady", 4, meth, seed)
        seen = {}
        orig = MockServer.handle
        async def spy(self, request=None):
            r = await orig(self, request)
            if r.status == 200:
                seen[request] = r.latency
            return r
        MockServer.handle = spy
        try:
            simulate(sc)
        finally:
            MockServer.handle = orig
        return seen
    a, b = lat_by_call(S.fixed_method(1)), lat_by_call(S.acc_method(**S.ACC_START))
    assert len(a) == len(b) == 4 * 150 and set(a) == set(b)
    assert all(a[k] == pytest.approx(b[k]) for k in a)
    c = lat_by_call(S.fixed_method(1), seed=8)
    assert sum(a[k] != pytest.approx(c[k]) for k in a) > 500


def test_crn_noise_ignores_number_of_rejected_attempts():
    """Same logical call: noise must not depend on how many 429/503 came before admission."""
    s = _server()
    z0 = s._noise((1, 4))
    for r in [(0, 0), (1, 3), (2, 2)] * 3:
        s._noise(r)
    assert s._noise((1, 4)) == z0 == _server()._noise((1, 4))


def test_slot_granted_window_logged_at_grant_time():
    async def go():
        p = SlotPool(1)
        first = await p.acquire()
        t = asyncio.ensure_future(p.acquire())
        await asyncio.sleep(0)
        p.set_limit(3)
        p.release()
        return first, await t
    first, second = run_virtual(go())
    assert first == (1, 1)
    assert second[0] == 3 and second[1] >= 1


def test_cancelled_waiter_does_not_leak_a_slot():
    async def go():
        p = SlotPool(1)
        await p.acquire()
        t = asyncio.ensure_future(p.acquire())
        await asyncio.sleep(0)
        p.release()          # slot handed to the waiter's future
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass
        return p.in_use
    assert run_virtual(go()) == 0


def test_objective_J_matches_eq1_and_handles_none():
    m = {"call_success_rate": 0.9, "mean_latency_excess_all_attempts": 0.2, "throttle_rate": 0.1}
    assert objective_J(m, 0.5, 0.5) == pytest.approx(0.9 - 0.1 - 0.05)
    assert objective_J(m, 1.0, 1.0) == pytest.approx(0.9 - 0.2 - 0.1)
    assert objective_J({**m, "throttle_rate": None}, 0.5, 0.5) is None


def test_metrics_all_attempts_excess_and_waste_consistent():
    r = run_scenario(S.build("RATE-steady", 16, S.fixed_method(16), 2))["metrics"]
    assert r["wasted_attempts_per_ok_call"] == pytest.approx((r["n_attempts"] - r["n_calls_ok"]) / r["n_calls_ok"])
    # error attempts are fast (excess 0), so averaging over all attempts can only dilute the success-only excess
    assert 0 <= r["mean_latency_excess_all_attempts"] <= r["mean_latency_excess"] + 1e-12


def test_spec_hash_changes_with_config_and_is_stable():
    u = ("CAP-steady", 4, S.fixed_method(4), 1)
    assert run_rq2.spec_hash(*u) == run_rq2.spec_hash(*u)
    assert run_rq2.spec_hash(*u) != run_rq2.spec_hash("CAP-steady", 4, S.fixed_method(8), 1)
    assert run_rq2.spec_hash(*u) != run_rq2.spec_hash("CAP-steady", 4, S.fixed_method(4), 2)


def test_resume_refuses_results_from_other_simulator_version(tmp_path, monkeypatch):
    run_rq2.run_phase("dev", tmp_path, limit=2)
    monkeypatch.setattr(run_rq2, "SIM_VERSION", "other")
    with pytest.raises(ValueError, match="do not match the current configuration"):
        run_rq2.run_phase("dev", tmp_path, limit=2)


def test_stamps_present_in_results(tmp_path):
    run_rq2.run_phase("dev", tmp_path, limit=1)
    f = next((tmp_path / "results/rq2/dev/units").glob("*.json"))
    d = json.loads(f.read_text())
    meta = (d.get("result") or d)["meta"]
    assert meta["sim_version"] == SIM_VERSION and len(meta["spec_hash"]) == 16


def test_select_refuses_when_test_results_exist(tmp_path):
    (tmp_path / "results/rq2/test/units").mkdir(parents=True)
    (tmp_path / "results/rq2/test/units/x.json").write_text("{}")
    assert run_rq2.do_select(tmp_path) == 4


# ------------------------------------------------------------------------------------------------ gateway
from eem_acc.acc import ACCConfig, ACCController
from eem_acc.agent import Gateway, RetryConfig, WorkloadConfig
from eem_acc.agent.scripted import _agent
from eem_acc.env.server import Response


class FakeServer:
    """Scripted server: status/latency per request; records dispatch times."""
    def __init__(self, plan=None, raise_on=None):
        self.plan, self.raise_on = plan or {}, raise_on
        self.dispatch, self.max_inflight_seen, self.inflight = {}, 0, 0

    async def handle(self, request=None):
        loop = asyncio.get_running_loop()
        self.dispatch.setdefault(request, []).append(loop.time())
        if request == self.raise_on:
            raise RuntimeError("server blew up")
        status, lat, ra = self.plan.get(request, (200, 1.0, None))
        self.inflight += 1
        self.max_inflight_seen = max(self.max_inflight_seen, self.inflight)
        try:
            await asyncio.sleep(lat)
        finally:
            self.inflight -= 1
        return Response(status, lat, retry_after=ra)


def _acc(**kw):
    base = dict(alpha=0.5, beta=0.5, epsilon=0.1, w0=2, w_min=1, w_max=8, n_c=1, l_target=10.0)
    base.update(kw)
    return ACCController(ACCConfig(**base))


RETRY = RetryConfig(max_retries=2, backoff_base=0.05, backoff_cap=1.0)


def test_shrink_applies_before_the_slot_is_released():
    """W=2, n_c=1: call A's 429 at t=0.01 shrinks W to 1. Releasing A's slot must NOT admit the queued call C
    (in_use stays 1 = B); C may only start when B finishes at t=1.0. Releasing before shrinking admits C at 0.01."""
    srv = FakeServer({(0, 0): (429, 0.01, 5.0)})
    gw = Gateway(srv, _acc(), RETRY, seed=0)

    async def go():
        await asyncio.gather(*(gw.call(request=(a, 0)) for a in range(3)))
    run_virtual(go())
    assert srv.dispatch[(2, 0)][0] >= 1.0 - 1e-9
    assert srv.max_inflight_seen <= 2


def test_attempt_logs_window_at_grant_time_not_at_resume():
    """The limit is changed between the grant and the waiter's resumption; the record must show the grant-time limit."""
    srv = FakeServer()
    gw = Gateway(srv, _acc(w0=1, frozen=True), RETRY, seed=0)
    real_release = gw.slots.release

    def release_then_change_limit():
        real_release()
        gw.slots.limit = 99            # after the grant, before the granted waiter runs
    gw.slots.release = release_then_change_limit

    async def go():
        await asyncio.gather(gw.call(request=(0, 0)), gw.call(request=(1, 0)))
    run_virtual(go())
    assert [a.window for a in gw.attempts] == [1, 1]


@pytest.mark.parametrize("where", ["server", "controller"])
def test_slot_released_when_server_or_controller_raises(where):
    srv = FakeServer(raise_on=(0, 0) if where == "server" else None)
    ctrl = _acc()
    if where == "controller":
        def boom(*a, **k):
            raise RuntimeError("controller blew up")
        ctrl.record = boom
    gw = Gateway(srv, ctrl, RETRY, seed=0)

    async def go():
        with pytest.raises(RuntimeError):
            await gw.call(request=(0, 0))
        return gw.slots.in_use
    assert run_virtual(go()) == 0


def test_backoff_jitter_is_keyed_by_call_and_retry_not_by_order():
    mk = lambda: Gateway(FakeServer(), _acc(), RETRY, seed=11)
    g1, g2 = mk(), mk()
    keys = [(0, (3, 7)), (1, (3, 7)), (0, (0, 0)), (1, (5, 2))]
    v1 = {kk: g1._backoff(*kk) for kk in keys}
    v2 = {kk: g2._backoff(*kk) for kk in reversed(keys)}
    assert v1 == v2
    assert v1[(0, (3, 7))] != v1[(1, (3, 7))] / 2      # retries of one call get distinct jitter
    assert len({round(v / min(1.0, 0.05 * 2 ** k), 12) for (k, _), v in v1.items()}) == len(keys)


def test_think_time_draws_do_not_depend_on_timing():
    """Two runs with different service times cross a zero-think phase after different numbers of calls. The
    k-th think time (where nonzero in both) must still be the same variate: one draw per call, always."""
    phases = ((0.0, 0.1), (1.0, 0.0), (1.2, 0.1))
    wl = WorkloadConfig(n_agents=1, calls_per_agent=60, phases=phases, start_jitter=0.0)

    class G:
        def __init__(self, d):
            self.d, self.log = d, []
        async def call(self, request=None):
            loop = asyncio.get_running_loop()
            t0 = loop.time()
            await asyncio.sleep(self.d)
            self.log.append((t0, loop.time()))

    def thinks(d):
        g = G(d)
        run_virtual(_agent(g, wl, np.random.default_rng(5), 0))
        prev, out = 0.0, []
        for t0, t1 in g.log:
            out.append(t0 - prev)
            prev = t1
        return out
    a, b = thinks(0.01), thinks(0.05)
    both = [k for k in range(60) if a[k] > 1e-12 and b[k] > 1e-12]
    assert len(both) > 20 and any(a[k] <= 1e-12 for k in range(60)) and any(b[k] <= 1e-12 for k in range(60))
    assert all(a[k] == pytest.approx(b[k]) for k in both)


def test_floor_epsilon_recovers_exact_integer():
    """12*(1 - 0.5*25/30) is exactly 7 but evaluates to 6.999999999999999 in floats."""
    ctrl = _acc(w0=12, w_max=64, n_c=30, beta=0.5)
    for i in range(30):
        ctrl.record(429 if i < 25 else 200, 0.1, float(i))
    assert ctrl.window == 7


# ------------------------------------------------------------------------------------------- result store
from eem_acc.utils import ResultStore


def test_sanitised_ids_do_not_collide(tmp_path):
    st = ResultStore(tmp_path)
    st.save("a/b", {"v": 1})
    st.save("a_b", {"v": 2})
    assert st.load("a/b") == {"v": 1} and st.load("a_b") == {"v": 2}


def test_file_of_another_unit_is_not_accepted(tmp_path):
    st = ResultStore(tmp_path)
    st.save("x", {"v": 1})
    st.path("x").replace(st.path("y"))
    assert not st.is_valid("y")
    with pytest.raises(ValueError):
        st.load("y")


# ------------------------------------------------------------------------------------- runner guards
def _tiny(monkeypatch, tmp_path):
    monkeypatch.setattr(S, "ACC_GRID", dict(alpha=(0.25, 0.5), beta=(0.5,), epsilon=(0.1,), n_c=(8,)))
    monkeypatch.setattr(S, "FIXED_W_DEV", (2, 8))
    monkeypatch.setattr(S, "PACER_RATES_DEV", (20.0,))
    monkeypatch.setattr(S, "DEV_SEEDS", (0, 1))
    monkeypatch.setattr(S, "TEST_SEEDS", (100,))
    monkeypatch.setattr(S, "M_LEVELS", (2,))
    monkeypatch.setattr(S, "CALLS_PER_AGENT", 12)
    monkeypatch.setattr(run_rq2, "FROZEN_PATH", tmp_path / "repo_frozen.json")


def test_select_refuses_stale_dev_results(tmp_path, monkeypatch):
    _tiny(monkeypatch, tmp_path)
    run_rq2.run_phase("dev", tmp_path)
    monkeypatch.setattr(run_rq2, "SIM_VERSION", "older")
    assert run_rq2.do_select(tmp_path) == 5


def test_select_refuses_foreign_leftover_file(tmp_path, monkeypatch):
    _tiny(monkeypatch, tmp_path)
    run_rq2.run_phase("dev", tmp_path)
    ResultStore(tmp_path / "results/rq2/dev/units").save("leftover_from_old_grid", {"meta": {}, "metrics": {}})
    assert run_rq2.do_select(tmp_path) == 5


def test_frozen_records_version_and_digest_and_test_phase_checks_them(tmp_path, monkeypatch):
    _tiny(monkeypatch, tmp_path)
    run_rq2.run_phase("dev", tmp_path)
    assert run_rq2.do_select(tmp_path) == 0
    fz = json.loads(run_rq2.FROZEN_PATH.read_text())
    assert fz["sim_version"] == SIM_VERSION and len(fz["dev_digest"]) == 16
    frozen, msg = run_rq2.load_frozen_for_test(tmp_path)
    assert frozen == fz, msg
    # repo copy edited -> copies differ -> refuse
    run_rq2.FROZEN_PATH.write_text(json.dumps({**fz, "dev_digest": "x"}))
    assert run_rq2.load_frozen_for_test(tmp_path)[0] is None
    run_rq2.FROZEN_PATH.write_text(json.dumps(fz))
    # simulator version changed -> refuse
    monkeypatch.setattr(run_rq2, "SIM_VERSION", "newer")
    assert run_rq2.load_frozen_for_test(tmp_path)[0] is None


def test_frozen_selected_on_another_root_is_refused(tmp_path, monkeypatch):
    _tiny(monkeypatch, tmp_path)
    a, b = tmp_path / "A", tmp_path / "B"
    run_rq2.run_phase("dev", a)
    assert run_rq2.do_select(a) == 0
    monkeypatch.setattr(S, "CALLS_PER_AGENT", 13)       # B's dev results differ from A's
    run_rq2.run_phase("dev", b)
    (b / "results/rq2").mkdir(parents=True, exist_ok=True)
    # B has no own selection; only the repo copy (selected on A) exists -> digest mismatch -> refuse
    assert run_rq2.load_frozen_for_test(b)[0] is None


def test_select_refuses_to_overwrite_a_different_frozen_file(tmp_path, monkeypatch):
    _tiny(monkeypatch, tmp_path)
    run_rq2.run_phase("dev", tmp_path)
    run_rq2.FROZEN_PATH.write_text(json.dumps({"something": "else"}))
    assert run_rq2.do_select(tmp_path) == 6
    assert run_rq2.do_select(tmp_path, force=True) == 0
    assert run_rq2.do_select(tmp_path) == 0             # identical content: allowed


def test_iter_results_skips_result_copied_under_another_name(tmp_path):
    st = ResultStore(tmp_path)
    st.save("x", {"v": 1})
    import shutil
    shutil.copy(st.path("x"), tmp_path / "copy_of_x.json")
    assert list(st.iter_results()) == [("x", {"v": 1})]
```
