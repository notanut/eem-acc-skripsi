# Review request: S3.2b implementation (RQ3 wiring) + S3.2a r4 delta, 2026-11-22

You are a strict reviewer of LLM-agent experiments and load-control simulations. Be concrete; no praise.

## Context
Thesis: LLM agents (Qwen2.5-7B, local, greedy) solve multi-call tasks against a synthetic API whose schema drifts.
EEM is a reflection memory of API errors (S3.2a, reviewed twice plus your earlier second opinion). ACC is an AIMD
concurrency window frozen from RQ2 (done). **RQ3** tests both together: 4 LLM agents (20 tasks per seed, 5 each)
share one EEM store and call the API through the RQ2 gateway (ACC or no window) next to 32 scripted background agents
on a simulated CAP-steady server, all in virtual time (LLM generation time is a sleep computed from token counts).
Primary endpoints: R1 = C2 (EEM) vs C0 (no memory) under ACC, LLM task success (non-inferiority one task per seed,
then superiority); R2 = ACC vs no window within C2. The RQ2 simulator code is reused unchanged; RQ3 adds a semantic
server (executes the S3.1 suite at admission for LLM requests, idempotent by call key), a request type that is the
plain (agent, k) tuple for keyed randomness, a dispatcher that turns exhausted retries into a "transient" notice (no
EEM note), until-done background load, a 20 s warm-up and T_max = 3,600 s.

The spec (below) was reviewed by a fresh-context reviewer (2 High, 5 Medium, 9 Low at r1); the code by a second
reviewer (0 High, 6 Medium, 12 Low) and a re-review of the fixes (0 High, 3 Medium, 12 Low), all applied (spec
sections 9-10) with the user's decisions D117-D121. 36 RQ3 tests pass (750 in the repo). A scripted CPU stand-in for the LLM runs everything; its numbers are
plumbing, not evidence (dev seed 5000: C2+ACC 0.95, C2+none 0.85, C0+none 0.45 task success; LLM overload 0.75-0.77
with no window vs 0 under ACC).

Also included: the S3.2a r4 delta after your earlier second opinion: D115 (a moved-operation note also closes on a
success of an id known to be an alias of its operation, learned from any agent's 410 pointer or docs response,
alias chains included) and D116 (remaps of one field shown as one compact item; when line items must be cut, the oldest go).

## Checklist
1. Validity of R1/R2: anything in the wiring that biases C2 vs C0 or ACC vs none other than the memory or the window
   (seeding, background load, warm-up, transient handling, store sharing, T_max censoring, metric definitions)?
2. Are the infrastructure metrics (background traffic only, D117) and the LLM-only metrics (overload, SLO, latency
   excess, J_rq3 = task success - 0.5 x excess - 0.5 x overload) defensible for the paper? Anything mixing traffic?
3. Semantic server correctness: order of 429/503 vs execution, idempotency under client timeouts, hidden executions
   (executed but delivered as transient), semantic 4xx bookkeeping, keyed noise.
4. Shared-store concurrency: timing rules (prompt built at generation start, note visible after its reflection sleep,
   outcome update on the success response), owner rewrites by merges, D119 (fallback reuses the first prompt).
5. Readiness for S3.3 (llama.cpp backend on a free Colab T4, caching/resume, checkpoints of units).
6. D115/D116: any leak or correctness risk in closing via a known alias or in the compact rendering below.

## Output
Findings ranked by severity with concrete fixes; then a verdict on whether S3.3 can build on this.

## The spec (docs/s32b_spec.md, r4)

# S3.2b specification: RQ3 wiring (gateway, server, background load, shared store) (CPU only)

_r4, 2026-11-22 (r3 + the re-review: D120-D121 and clear-cut fixes; r3 = the code review: D117-D119, section 10). Implements design r7 section 7 (RQ3) on top of S3.1 (suite) and S3.2a (agent loop, EEM), reusing
the RQ2 simulator unchanged (`Gateway`, `MockServer`, `ACCController`, frozen ACC config). r1 was reviewed by a
fresh-context subagent (2 High, 5 Medium, 9 Low); the user decided D107-D111 (section 9); the clear-cut findings are
applied below._

## 1. Run = one RQ3 cell on one seed

```python
res = run_rq3_unit(spec, condition, window, backend, embedder, tokenizer, cfg) -> dict
#   spec: an S3.1 rq3 stream (20 tasks; agents [[0,4,..],[1,5,..],[2,6,..],[3,7,..]])
#   condition: "C0" | "C2" | "C2-private"; window: "none" | "acc" | "fixed4"
```

Cells (design section 7): {C0, C2} x {no window, ACC on} plus C2 + fixed W = 4; descriptive: C2 + ACC with private
stores and the 2x2 with LLM times / 3 (both on the first 5 test seeds, 4000-4004). The same warm-up applies to every
cell. Seeds: rq3 test 4000-4009 (planned), rq3 **dev 5000-5099** (D111, an S3.1 change before the freeze: one unit
each of C2+ACC and C0+none in the pilot; only wiring bugs and T_max may change after it).

## 2. Simulator composition (virtual time, one event loop)

- **Configs from one source:** `ServerConfig`, retry policy and the ACC config are taken from
  `scenarios.build("CAP-steady", 32, method, seed)` with `method` = `acc_method(**frozen["acc"])` (ACC),
  `fixed_method(128)` (none; RQ2's unconstrained window, any W >= 32 behaves the same with 36 agents) or
  `fixed_method(4)`. A test asserts equality with those builds.
- **Seeds:** as `sim.py`: `SeedSequence(seed).spawn(3)` -> server, gateway, scripted agents; the scripted agents use
  `run_agents`' `spawn(n_agents + 1)` children. Nothing depends on the cell or condition. Common random numbers hold at
  the level of **draws** (per-agent think times; noise and jitter keyed by (agent, k, admission)); arrival times differ
  between cells because the scripted agents run in a closed loop (Section IV says so).
- **Server:** `SemanticServer(MockServer)` overrides `handle` and keeps RQ2's order: refill -> 429 -> 503 -> token ->
  then:
  - **scripted request** (plain `(agent, k)` tuple): inflight -> noise -> service latency -> 200, exactly as RQ2
    (load only, D110). A test asserts that a scripted-only run through `SemanticServer` gives attempts and metrics
    bit-identical to `MockServer` (`sim.run_scenario`, CAP-steady, M = 32, ACC and W = 128).
  - **LLM request:** executed by the S3.1 suite at admission with `call_key = (agent, k)` (idempotent: a retry after a
    client timeout gets the stored response; the state changes once, before any delay). A semantic **200** (including
    a silent error payload and an idempotent replay of a stored 200) then counts as inflight, draws noise and takes the
    service latency; it is a latency sample for ACC. A semantic **4xx** uses its rate token but is not inflight, draws
    no noise, returns after `error_latency` = 0.02 s (D107), and is neither an overload signal nor a latency sample
    (Section III-C); it has its own key in `counts`.
  - Every `handle` that completes is logged as (request, t_arrival, status, latency); an attempt the gateway abandoned
    at its client timeout (1 s) completes later with latency > T_out and counts as a client timeout. Attempts still in
    flight at the end are left out, as in the gateway. A test checks the log against `gw.attempts`.
- **Request ids:** an LLM request is a tuple subclass that stays exactly the 2-tuple `(agent, k)` (no `__eq__`/
  `__hash__` override; attributes op, args, task_index in `__dict__`), so the RQ2 gateway and server unpack it
  unchanged; the suite gets the plain tuple `(agent, k)`. LLM agents use ids 1000-1003. A test checks that its noise
  and jitter equal those of the plain tuple.
- **Gateway:** `TaggedGateway` (the RQ2 `Gateway` plus a per-attempt request log; no change to timing or draws), RQ2 retry policy (r_max 6, backoff 0.1-4 s, client timeout 1 s), windows as above.
- **Background load:** 32 scripted agents, CAP-steady think times (exponential, mean 0.1 s, start jitter 1 s), in a new
  **until-done** loop (`stage3/rq3.py`; RQ2 code untouched): they run until all 4 LLM agents have finished or until
  **T_max = 3,600 s** (D108); then every scripted call and in-flight attempt is cancelled and `gw.shutdown()` cancels
  abandoned server work (state changes made at admission stay).
- **LLM agents:** 4 S3.2a `Agent`s with ids 1000-1003, each solving its 5 tasks in order, starting after a **20 s
  warm-up** of background load (D109; RQ2 window traces show ACC's window settled by about 6 s, to be confirmed on the
  RQ2 dev seeds). `backend.generate` stays synchronous (no thread or executor), so the virtual clock stands still
  during a generation (test: `loop.time()` equal before and after). The agent then sleeps its t_llm scaled by
  `LoopConfig.llm_time_scale` (1.0; 1/3 in the sensitivity run; applied to action, fallback and reflection calls;
  stamped; RQ1 unchanged).
- **Dispatcher:** `GatewayDispatcher.call(op, args, agent, k, task_index)` -> `gw.call(request)`; a final status in
  `status.RETRYABLE` (exhausted, incl. client timeouts) is **transient** (S3.2a: transient notice, no EEM); otherwise
  the suite response of the admitted attempt (status, body, truth, suite op) is returned.
- **Partial logs:** each agent appends its step records to an external list as they are made, so a task cancelled at
  T_max keeps its steps; tasks never started get a summary with outcome failure (the denominator stays 20); completion
  times of tasks not finished by the end are censored at the end, from the task's start or, if never started, from
  the warm-up end, with `completion_censored` (D118). Never-started tasks carry `never_started` and are excluded from
  completion-time statistics, with their count reported per cell; they still count as failures for task success
  (D120). Cut-off tasks are summarised from their partial steps (drift, agent and format errors, API calls; the call
  number is recorded before the call); a generation cut short by T_max counts its full LLM time and its step counts
  as a step (D121, a caveat for capped units' secondary counts only).

## 3. Store

- **Shared (C2):** one `EEMStore` for the 4 agents, with S3.2a's timing rules (prompt built when a generation starts;
  a note visible after its reflection's t_llm sleep; outcome update and re-check when the success response arrives;
  owner = (agent, task)). The fallback generation after a malformed reply reuses the prompt built for the first
  generation, so a note that became visible in between is not shown to it (D119). A merge by another agent changes the owner, so the original agent's same-task outcome update
  no longer fires; reported as a caveat of the shared store.
- **Private (descriptive):** one store per agent; a test checks no note is visible across agents.

## 4. Hidden executions and drift visibility

- **Hidden execution:** a call the suite executed (`suite.executed(call_key)`, a public method) whose final outcome
  reached the agent as transient. Counted per unit, split into state-changing (stored 200 on a write) and others. In
  CAP-steady they should be near zero (with the hard cap at 8, a 1 s timeout needs noise z > 2.77); the test uses a
  slow server that forces them.
- **Overload hides drift:** drift errors observed and notes created are counted per unit.

## 5. Results per unit

- Per LLM task: the S3.2a summary plus virtual start/finish, completion time (censored), `unfinished_at_t_max`.
- **LLM call success** = final status not in `RETRYABLE` (a semantic 4xx is a delivered response, not an
  infrastructure failure); LLM-only overload rate (`status.OVERLOAD`; a client timeout counts), SLO violation, latency
  p95 over 200s, mean latency excess over all attempts, attempts per call and censored e2e p95, from the gateway's
  attempt records tagged with their request (`TaggedGateway`). Semantic 4xx attempts are in these denominators and in
  ACC's (stated).
- **J for RQ3** (Section III-A): LLM task success rate - 0.5 x mean latency excess - 0.5 x overload rate over the LLM
  attempts (lambda = gamma = 0.5, design section 8); the RQ2 objective J over the background attempts of the primary
  window beside it (D117).
- Infrastructure metrics (`metrics.compute_metrics_v2`, RQ2's 20 s grace) over the **background (scripted) traffic
  only** (D117), over **[warm-up, end]** through `AnalysisConfig.windows` (primary; plus call success with the grace,
  mean latency excess over all attempts, overload rate and the RQ2 objective J with lambda = gamma = 0.5) and over the
  whole run (secondary); goodput; window trace. The queue trace, window (trace, mean, final) and server max in-flight are shared by
  all traffic. The window's
  e2e p95 censors calls requested near the end with no grace (slightly low; stated).
- Hidden executions, drift errors observed, notes created, final notes and note events (C2), the stamp (S3.2a stamp +
  window, condition, store mode, llm_time_scale, T_max, warm-up, SIM_VERSION, **RQ3 wiring version**).

## 6. Analysis hooks

R1 (C2 vs C0 under ACC, NI margin one task per seed, then superiority), R2 (ACC vs no window within C2), interaction
with the floor rule, secondary metrics (design section 8): implemented with the RQ1 analysis in S3.4/S3.5.

## 7. Tests (pytest, CPU, ScriptedBackend)

- Configs equal the `scenarios.build` single source; `none`/`fixed4` frozen at W 128 / 4.
- Scripted-only bit-identity of `SemanticServer` vs `MockServer`.
- Semantic server: order (429/503 before semantics); idempotent retry after a forced client timeout (one state change;
  a retry while the first admission is in flight gets the stored response; both count inflight); 4xx bookkeeping;
  silent 200 is a latency sample; server log vs `gw.attempts`.
- Request subclass: same noise and jitter as the plain tuple; the dispatcher's status equals the suite's.
- Until-done and T_max: the run ends when the LLM agents finish; at T_max unfinished tasks are marked with partial
  logs; no pending tasks after the run; `shutdown` called; no LLM call before the warm-up ends; `get_api_docs` goes
  through the gateway.
- Transient path with a slow server: retries exhausted -> transient notice, hidden executions counted; the scripted
  backend re-issues the call unchanged after a transient notice.
- Shared vs private store; S3.2a timing rules with 4 agents.
- Determinism: same unit twice identical; different PYTHONHASHSEED; the scripted agents' draws identical across cells
  of a seed.
- Manipulation sanity (scripted backend): with no window, LLM call success is not above ACC's.

## 8. Out of scope

The llama.cpp backend, caching and resume (S3.3); the RQ3 analysis code (S3.4/S3.5).

## 9. Decisions (user, 2026-11-22)

| ID | Decision |
|---|---|
| D107 | Semantic 4xx returned after error_latency (0.02 s); its effect on completion time by error style is small and stated |
| D108 | T_max = 3,600 s of virtual time (expected 6-15 virtual minutes per unit; simulation cost is negligible) |
| D109 | LLM agents start after a 20 s warm-up (new relative to design r7; added to the design text) |
| D110 | Background calls are load only (no semantics) |
| D111 | RQ3 dev seeds 5000-5099; one unit each of C2+ACC and C0+none in the pilot; only wiring bugs and T_max may change after it |
| D117 | RQ3 infrastructure metrics over the background calls only; LLM calls are reported by the LLM-only metrics |
| D118 | Tasks not finished by the end: completion time censored at the end (never started: from the warm-up end) with a flag |
| D119 | The fallback generation reuses the prompt built for the first generation (it retries the same decision); with 4 agents a note may have become visible in between; documented |
| D120 | Never-started tasks are flagged and excluded from completion-time statistics (count reported per cell); they count as task failures |
| D121 | Partial summaries of cut-off tasks count a cut generation's full LLM time and its step; stated as a caveat |

## 10. Code review (2026-11-22): clear-cut changes

No High findings. Applied: agent errors end the run at once (`FIRST_EXCEPTION`), with `shutdown` always called
before re-raising (M4); the stamp carries `action_max_tokens`, `grammar_always`, the method, ACC, server and retry
configs and the SHA-256 of `rq2_frozen.json` (M5); LLM metrics from the tagged gateway attempts, so attempts
abandoned at shutdown are counted as the gateway counts them (M3, L6); the primary window gets call success, excess
and J (M2); `LLMRequest` copies and pickles (L4); embedding replay (`embedding_preload`, L5); notes created per unit
(L10); unused imports removed and the shared store built only when used (L12); RQ3 wiring version rq3-v2. Tests
added for every gap in section 7 that the review listed (M6, L11).


## The code: src/eem_acc/stage3/rq3.py

```python
"""RQ3 wiring (S3.2b spec): 4 LLM agents (S3.2a) + 32 scripted background agents through the RQ2 gateway and a
semantic CAP-steady server, in virtual time. The RQ2 simulator code is reused unchanged."""
from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from ..acc import ACCController
from ..agent import Gateway
from ..env import MockServer, Response, run_virtual
from ..metrics import AnalysisConfig, _calls_with_censoring, _pct, _slo_bad, compute_metrics_v2, objective_J
from ..scenarios import L_TARGET, SUCCESS_GRACE, acc_method, build, fixed_method
from ..sim import SIM_VERSION
from ..status import CLIENT_TIMEOUT, OVERLOAD, RETRYABLE
from .agent.loop import Agent, LoopConfig, Outcome, UnitEmbedder, code_commit, task_summary
from .agent.memory import make_memory
from .agent import prompts as P
from .agent.parser import PARSER_VERSION
from .apisuite import ApiSuite, World
from .apisuite.docs import documented_all

RQ3_VERSION = "rq3-v2"
LLM_AGENT_BASE = 1000
N_SCRIPTED = 32
WARMUP = 20.0          # D109
T_MAX = 3600.0         # D108
WRITES = {"create_order", "pay_order", "refund_payment", "cancel_order", "reserve_stock", "create_shipment",
          "create_ticket", "update_ticket", "add_comment"}
FROZEN = Path(__file__).resolve().parents[3] / "configs" / "rq2_frozen.json"


class LLMRequest(tuple):
    """Exactly the 2-tuple (agent, k) for the RQ2 gateway/server (keyed randomness, unpacking); op, args and
    task_index ride along as attributes."""

    def __new__(cls, agent: int, k: int, op: str, args: Any, task_index: int):
        obj = super().__new__(cls, (int(agent), int(k)))
        obj.op, obj.args, obj.task_index = op, args, task_index
        return obj

    def __getnewargs__(self):                     # copy / deepcopy / pickle (S3.3 checkpoints; review L4)
        return (self[0], self[1], self.op, self.args, self.task_index)


def is_llm(request: Any) -> bool:
    return isinstance(request, LLMRequest)


class TaggedGateway(Gateway):
    """RQ2 Gateway that also records the request of every attempt (`attempt_requests[i]` belongs to `attempts[i]`):
    the attempt is appended right after `_send` returns, with no await in between, so the lists stay aligned."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.attempt_requests: list = []

    async def _send(self, request):
        resp = await super()._send(request)
        self.attempt_requests.append(request)
        return resp


def traffic_view(gw: TaggedGateway, keep) -> SimpleNamespace:
    """The gateway's records restricted to requests with keep(request) (duck-typed for metrics.compute_metrics_v2).
    The queue trace, window and server are shared by all traffic."""
    assert len(gw.attempts) == len(gw.attempt_requests)
    return SimpleNamespace(attempts=[a for a, r in zip(gw.attempts, gw.attempt_requests) if keep(r)],
                           calls=[c for c in gw.calls if keep(c.request)],
                           open_calls={i: v for i, v in gw.open_calls.items() if keep(v[1])},
                           queue_trace=gw.queue_trace, controller=gw.controller, server=gw.server)


class SemanticServer(MockServer):
    """RQ2 MockServer with the S3.1 suite executed at admission for LLM requests (S3.2b spec section 2)."""

    def __init__(self, cfg, seed: int, suite: ApiSuite | None):
        super().__init__(cfg, seed)
        self.suite = suite
        self.counts["sem4xx"] = 0
        self.log: list[tuple] = []          # (request, t_arrival, status, latency) of every completed handle

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
            self.log.append((request, now, 429, cfg.error_latency))
            return Response(429, cfg.error_latency, retry_after=cfg.retry_after)
        if self.inflight >= hard_cap:
            self.counts[503] += 1
            await asyncio.sleep(cfg.error_latency)
            self.log.append((request, now, 503, cfg.error_latency))
            return Response(503, cfg.error_latency)

        self._tokens -= 1.0
        sem = None
        if isinstance(request, LLMRequest):
            sem = self.suite.execute(request.op, request.args, request.task_index, call_key=(request[0], request[1]))
            if sem.status != 200:                   # semantic 4xx: not inflight, no noise, fast (D107)
                self.counts["sem4xx"] += 1
                await asyncio.sleep(cfg.error_latency)
                self.log.append((request, now, sem.status, cfg.error_latency))
                return Response(sem.status, cfg.error_latency, payload=sem)
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
        self.log.append((request, now, 200, latency))
        return Response(200, latency, payload=sem)


class GatewayDispatcher:
    def __init__(self, gw: Gateway, suite: ApiSuite):
        self.gw, self.suite = gw, suite
        self.calls: list[dict] = []

    async def call(self, op_id: str, args: Any, agent: int, k: int, task_index: int) -> Outcome:
        rec = await self.gw.call(request=LLMRequest(agent, k, op_id, args, task_index))
        entry = {"agent": agent, "k": k, "op": op_id, "status": rec.status, "attempts": rec.attempts,
                 "t_request": rec.t_request, "t_done": rec.t_done, "queue_wait": rec.queue_wait}
        if rec.status in RETRYABLE:                       # exhausted: transient (no EEM)
            stored = self.suite.executed((agent, k))
            entry["transient"] = True
            if stored is not None:
                entry["hidden_execution"] = True
                entry["hidden_state_change"] = stored.status == 200 and stored.op in WRITES and not (
                    isinstance(stored.body, dict) and stored.body.get("status") == "error")
            self.calls.append(entry)
            return Outcome(rec.status, None, True, {"kind": "transient"}, None)
        sem = rec.payload
        self.calls.append(entry)
        return Outcome(sem.status, sem.body, False, sem.truth.as_json(), sem.op)


async def _scripted_agent(gw: Gateway, think_mean: float, start_jitter: float, rng: np.random.Generator,
                          agent_id: int, stop: asyncio.Event) -> None:
    """RQ2's scripted agent in until-done mode: one exponential draw per call, stop when `stop` is set."""
    if start_jitter > 0:
        await asyncio.sleep(float(rng.uniform(0, start_jitter)))
    k = 0
    while not stop.is_set():
        u = float(rng.exponential(1.0))
        if think_mean > 0:
            await asyncio.sleep(think_mean * u)
        if stop.is_set():
            return
        await gw.call(request=(agent_id, k))
        k += 1


def method_for(window: str) -> dict:
    frozen = json.loads(FROZEN.read_text())
    return {"acc": acc_method(**frozen["acc"]), "none": fixed_method(128), "fixed4": fixed_method(4)}[window]


def _llm_metrics(view: SimpleNamespace, t_end: float) -> dict:
    """LLM-only attempt metrics from the gateway's attempt records (RQ2 accounting: a client timeout is an overload
    signal and an SLO violation; latency percentiles over 200s). Semantic 4xx attempts are in the denominators
    (review L7, stated)."""
    att = view.attempts
    n = len(att)
    calls = _calls_with_censoring(view, t_end)
    if not n:
        return {"n_attempts": 0, "n_calls": len(calls)}
    judged = [a for a in att if a.status in (200, CLIENT_TIMEOUT)]
    lat200 = [a.latency for a in att if a.status == 200]
    return {"n_attempts": n, "n_calls": len(calls),
            "overload_rate": sum(a.status in OVERLOAD for a in att) / n,
            "slo_violation_rate": (sum(_slo_bad(a, L_TARGET) for a in judged) / len(judged)) if judged else None,
            "mean_latency_excess": float(np.mean([max(0.0, (a.latency - L_TARGET) / L_TARGET) for a in att])),
            "lat_p95": _pct(lat200, 95), "attempts_per_call": n / len(calls) if calls else None,
            "e2e_p95_censored": _pct([c[2] for c in calls], 95)}


def _window_extra(view: SimpleNamespace, t0: float, t1: float, grace: float) -> dict:
    """What `_window_stats` lacks for the RQ3 primary window (review M2): call success over calls requested in
    [t0, t1 - grace), latency excess over attempts dispatched in [t0, t1), and the RQ2 objective J (0.5, 0.5)."""
    calls = [c for c in _calls_with_censoring(view, t1) if t0 <= c[0] < t1 - grace]
    att = [a for a in view.attempts if t0 <= a.t_dispatch < t1]
    m = {"call_success_rate": (sum(c[1] for c in calls) / len(calls)) if calls else None,
         "n_calls_success_population": len(calls),
         "mean_latency_excess_all_attempts": float(np.mean([max(0.0, (a.latency - L_TARGET) / L_TARGET)
                                                            for a in att])) if att else None,
         "throttle_rate": (sum(a.status in OVERLOAD for a in att) / len(att)) if att else None}
    m["J"] = objective_J(m, 0.5, 0.5)
    return m


def run_rq3_unit(spec: dict, condition: str, window: str, backend, embedder, tokenizer,
                 cfg: LoopConfig | None = None, warmup: float = WARMUP, t_max: float = T_MAX,
                 server_override: dict | None = None, retry_override: dict | None = None,
                 embedding_preload: list | None = None) -> dict:
    assert spec["kind"] == "rq3" and condition in ("C0", "C2", "C2-private")
    seed = spec["seed"]
    method = method_for(window)
    sc = build("CAP-steady", N_SCRIPTED, method, seed)
    server_cfg = dataclasses.replace(sc.server, **(server_override or {}))
    retry = dataclasses.replace(sc.retry, **(retry_override or {}))
    base_cond = "C2" if condition.startswith("C2") else "C0"
    cfg = dataclasses.replace(cfg or LoopConfig(), condition=base_cond)
    tasks = spec["tasks"]
    if hasattr(backend, "register_tasks"):
        backend.register_tasks(tasks)
    emb = UnitEmbedder(embedder, embedding_preload)

    async def main():
        loop = asyncio.get_running_loop()
        ss = np.random.SeedSequence(seed).spawn(3)                       # sim.py layout
        seeds = [int(s.generate_state(1)[0]) for s in ss]
        world = World.build(spec)
        suite = ApiSuite(world, spec)
        server = SemanticServer(server_cfg, seeds[0], suite)
        gw = TaggedGateway(server, ACCController(sc.acc), retry, seeds[1])
        disp = GatewayDispatcher(gw, suite)
        stop = asyncio.Event()
        children = np.random.SeedSequence(seeds[2]).spawn(N_SCRIPTED + 1)  # run_agents layout
        think = sc.workload.phases[0][1]
        scripted = [asyncio.ensure_future(_scripted_agent(gw, think, sc.workload.start_jitter,
                                                          np.random.default_rng(c), i, stop))
                    for i, c in enumerate(children[:N_SCRIPTED])]
        shared = None if condition == "C2-private" else make_memory(base_cond, emb, tokenizer, cfg.k, cfg.k_s,
                                                                     cfg.theta_link)
        memories = []
        agents = []
        for a in range(4):
            mem = make_memory(base_cond, emb, tokenizer, cfg.k, cfg.k_s, cfg.theta_link) \
                if condition == "C2-private" else shared
            memories.append(mem)
            agents.append(Agent(LLM_AGENT_BASE + a, backend, tokenizer, mem, disp, cfg, lambda ti: documented_all()))
        timing: dict[int, dict] = {}
        results: dict[int, dict] = {}

        async def llm_agent(agent: Agent, task_ids: list[int]):
            await asyncio.sleep(warmup)
            for ti in task_ids:
                timing[ti] = {"t_start": loop.time()}
                results[ti] = await agent.run_task(tasks[ti])
                timing[ti]["t_finish"] = loop.time()

        llm = [asyncio.ensure_future(llm_agent(agents[a], spec["agents"][a])) for a in range(4)]
        # Any agent's error (LLM or scripted) ends the run at once (reviews M4, re-review L1: with llama.cpp it
        # would otherwise surface only after the other agents finish); shutdown always runs (re-review L2).
        abort = loop.create_future()

        def on_done(t: asyncio.Task) -> None:
            if not t.cancelled() and t.exception() is not None and not abort.done():
                abort.set_result(None)
        for t in llm + scripted:
            t.add_done_callback(on_done)
        all_llm = asyncio.ensure_future(asyncio.wait(llm))
        pending: list = []
        try:
            await asyncio.wait({all_llm, abort}, timeout=max(0.0, t_max - loop.time()),
                               return_when=asyncio.FIRST_COMPLETED)
            t_end = loop.time()
            pending = [t for t in llm if not t.done()]
            for t in pending + [all_llm]:
                t.cancel()
            await asyncio.gather(*pending, all_llm, return_exceptions=True)
            stop.set()
            for t in scripted:
                t.cancel()
            await asyncio.gather(*scripted, return_exceptions=True)
        finally:
            await gw.shutdown()
        # errors in agent order (deterministic); an error raised while a task is being cancelled is not swallowed
        errors = [t.exception() for t in llm + scripted if t.done() and not t.cancelled() and t.exception()]
        if errors:
            raise errors[0]
        pending = [t for t in pending if t.cancelled()]
        return world, suite, server, gw, disp, agents, memories, timing, results, t_end, bool(pending)

    world, suite, server, gw, disp, agents, memories, timing, results, t_end, capped = run_virtual(main())
    summaries = []
    for inst in tasks:
        ti = inst["index"]
        agent = next(a for a, ids in enumerate(spec["agents"]) if ti in ids)
        if ti in results:
            res = results[ti]
        else:              # cancelled at T_max or never started: the partial steps are summarised (review L1, L2)
            steps = [r for r in agents[agent].live if r["task"] == ti]
            llm = [c for r in steps for c in r["llm"]]
            res = {"traj": [], "steps": steps, "finished": False, "reason": "unfinished_at_t_max",
                   "n_api": sum(1 for r in steps if "k" in r), "n_llm": len(llm),
                   "tok_in": sum(c["n_in"] for c in llm), "tok_out": sum(c["n_out"] for c in llm),
                   "t_llm": round(sum(c["t_llm"] for c in llm), 6), "max_steps": None}
        s = task_summary(inst, res, world, suite)
        s["agent"] = LLM_AGENT_BASE + agent
        t = timing.get(ti, {})
        s["t_start"] = t.get("t_start")
        s["t_finish"] = t.get("t_finish")
        # D118: a task not finished by the end is censored at the end (never started: from the warm-up end)
        s["completion_time"] = (t["t_finish"] - t["t_start"]) if "t_finish" in t else (
            t_end - t.get("t_start", warmup))
        s["completion_censored"] = "t_finish" not in t
        s["never_started"] = "t_start" not in t          # D120: excluded from completion-time statistics
        s["unfinished_at_t_max"] = ti not in results
        summaries.append(s)
    n_succ = sum(1 for s in summaries if s["success"])
    llm_calls = disp.calls
    llm_m = _llm_metrics(traffic_view(gw, is_llm), t_end)
    # D117: infrastructure metrics over the background (scripted) traffic only; LLM calls are in llm_metrics
    bg = traffic_view(gw, lambda r: not is_llm(r))
    analysis = AnalysisConfig(windows=(("llm_phase", warmup, t_end),), success_grace=SUCCESS_GRACE)
    infra = compute_metrics_v2(bg, L_TARGET, t_end, analysis)
    infra["windows"]["llm_phase"].update(_window_extra(bg, warmup, t_end, SUCCESS_GRACE))
    infra["traffic"] = "background"
    j = (n_succ / len(summaries)) - 0.5 * llm_m.get("mean_latency_excess", 0.0) - 0.5 * llm_m.get("overload_rate", 0.0)
    final_notes = [] if base_cond == "C0" else (memories[0].final_state() if condition == "C2" else
                                                  [m.final_state() for m in memories])
    note_events = [] if base_cond == "C0" else (memories[0].events if condition == "C2" else
                                                  [m.events for m in memories])
    return {
        "stamp": {"rq3_version": RQ3_VERSION, "sim_version": SIM_VERSION, "suite_version": spec["version"],
                  "spec_sha256": spec["sha256"], "condition": condition, "window": window,
                  "llm_time_scale": cfg.llm_time_scale, "t_max": t_max, "warmup": warmup,
                  "params": {"k": cfg.k, "k_s": cfg.k_s, "theta_link": cfg.theta_link,
                             "docs_wording": cfg.docs_wording, "action_max_tokens": cfg.action_max_tokens,
                             "grammar_always": cfg.grammar_always},
                  "method": method, "acc_config": dataclasses.asdict(sc.acc),
                  "server_config": dataclasses.asdict(server_cfg), "retry_config": dataclasses.asdict(retry),
                  "rq2_frozen_sha256": hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
                  "parser_version": PARSER_VERSION,
                  "prompt_version": P.PROMPT_VERSION, "prompt_sha256": P.prompt_hash(),
                  "backend": getattr(backend, "id", type(backend).__name__), "tokenizer": tokenizer.id,
                  "embedder": embedder.id, "code_commit": code_commit(),
                  "server_override": server_override, "retry_override": retry_override},
        "tasks": summaries, "success_rate": n_succ / len(summaries), "t_end": t_end, "capped": capped,
        "llm_calls": llm_calls,
        "llm_call_success_rate": (sum(1 for c in llm_calls if not c.get("transient")) / len(llm_calls))
        if llm_calls else None,
        "hidden_executions": sum(1 for c in llm_calls if c.get("hidden_execution")),
        "hidden_state_changes": sum(1 for c in llm_calls if c.get("hidden_state_change")),
        "drift_errors": sum(s["drift_errors"] for s in summaries),
        "notes_created": sum(1 for evs in ([note_events] if condition == "C2" else note_events)
                             for e in evs if e["event"] == "create"),
        "llm_metrics": llm_m, "J_rq3": j, "infra": infra,
        "window_trace": [list(x) for x in gw.controller.trace],
        "steps": [r for a in agents for r in a.live],
        "final_notes": final_notes, "note_events": note_events, "embedding_cache": emb.export(),
    }
```

## S3.2a r4 delta: note rendering (D116) and outcome update (D115), from src/eem_acc/stage3/agent/memory.py

```python
    @staticmethod
    def display_items(items: list[dict]) -> list[str]:
        """Rendered items for prompts; remaps of one field are shown compactly as one item (D116)."""
        out: list[str] = []
        groups: dict[str, int] = {}
        for it in items:
            if it["kind"] == "remap":
                pair = f"{json.dumps(it['from'])} -> {json.dumps(it['value'])}"
                if it["key"] in groups:
                    out[groups[it["key"]]] += ", " + pair
                    continue
                groups[it["key"]] = len(out)
                out.append(f"'{it['key']}': {pair}")
            else:
                out.append(render_item(it))
        return out

    def note_render(self, n: Note) -> tuple[str, dict]:
        """The note as shown in prompts, at most NOTE_CAP tokens including the "- " prefix. Budget order (D112):
        k_t (<= 30), then the outcome line (<= 40 with its label and mark; when items must be cut the oldest go
        first, D106/D116), then the reflection gets the rest (at most 60). Display order: k_t, reflection, line."""
        t = self.tokenizer
        stats = {"id": n.id, "items_total": len(n.items), "items_shown": 0, "line_shown": False}

        def fits(body: str) -> bool:
            return t.count("- " + body) <= NOTE_CAP

        if self.mode == "a1":
            body = t.truncate(f"{n.kt.api_name}: {n.text_raw}", NOTE_CAP)
            while body and not fits(body):
                c = t.count(body)
                body = t.truncate(body, c - 1)
                if body and t.count(body) >= c:                 # guard: a tokenizer whose re-encoding does not shrink
                    body = body[:-1]
            return "- " + body, stats
        head = t.truncate(n.kt.render(), KT_CAP).rstrip(". ")
        line = ""
        if n.items:
            mark = " (possibly outdated)" if n.possibly_outdated else ""
            for keep in range(len(n.items), 0, -1):
                shown = self.display_items(n.items[-keep:])                     # newest items kept (D116)
                options = ["Last working change: " + ("...; " if keep < len(n.items) else "") + "; ".join(shown)
                           + mark]
                if keep == 1 and len(n.items) > 1:                              # one item fits only unmarked
                    options.append("Last working change: " + "; ".join(shown) + mark)
                cand = next((c for c in options if t.count(c) <= LINE_CAP), None)
                if cand:
                    line = cand
                    stats.update(items_shown=keep, line_shown=True)
                    break
        refl = ""
        if n.reflection:
            label = "" if self.mode == "a2" else "Fix: "
            room = REFL_CAP
            while room > 0:
                cand = t.truncate(n.reflection, room).rstrip(". ")
                if not cand:                                    # nothing left to show: no dangling label
                    break
                if fits(". ".join(x for x in (head, label + cand, line) if x)):
                    refl = label + cand
                    break
                room -= 1
        body = ". ".join(x for x in (head, refl, line) if x)
        assert fits(body), "note over the cap"
        return "- " + body, stats


    def on_success(self, ctx):
        if ctx.called_id == "get_api_docs":
            target = ctx.args.get("operation") if isinstance(ctx.args, dict) else None
            if isinstance(ctx.body, dict) and target and ctx.body.get("operation") not in (None, target):
                self.docs_named[(ctx.agent, ctx.task, target)] = ctx.body["operation"]
                self.alias[ctx.body["operation"]] = target
            return
        if not self.outcome_on:
            return
        op = self.canonical(ctx.called_id)
        at = dict(at=[ctx.agent, ctx.task, ctx.step], t=round(ctx.t, 6), success_args=copy.deepcopy(ctx.args),
                  called_id=ctx.called_id, suite_op=ctx.suite_op)
        for n in sorted(self.notes.values(), key=lambda n: n.id):      # re-check marked lines (any agent)
            if self.canonical(n.api_name) == op and n.possibly_outdated:
                n.items = [it for it in n.items if item_holds(it, ctx.args, ctx.called_id)]
                n.possibly_outdated = False
                self._event("recheck", n, **at)
        for n in sorted(self.notes.values(), key=lambda n: n.id):      # close own open notes on this operation
            if not n.open or n.owner != (ctx.agent, ctx.task) or self.canonical(n.api_name) != op:
                continue
            n.canonical_op = op                                     # aliases learned since creation (r4 review M1)
            if op not in OPS:                                       # documented schema unknown: no diff (leak-safe)
                n.open = False
                self._event("close", n, moved_to=None, known_keys=[], new_items=[], no_schema=True, **at)
                continue
            documented = OPS[op].schema
            if n.error_type in MOVED_TYPES:
                if ctx.called_id == n.api_name:
                    n.open = False                                   # not a move after all: no new line
                    self._event("close", n, moved_to=None, known_keys=[], **at)
                    continue
                # D115: any other id reaching here has the same canonical operation, i.e. an alias (chain) of X
                # that some agent observed (410 pointer or docs response); re-review L9
                moved_to = ctx.called_id
            elif ctx.called_id != n.api_name:
                continue
            else:
                moved_to = None
            known = sorted(self.line_keys.get(op, set()))
            new = last_working_items(n.trigger_args, ctx.args, documented, moved_to=moved_to,
                                     error_keys=n.kt.error_keys, known_keys=frozenset(known))
            kept = list(n.items)                                     # D113: items kept from before a merge
            n.items = kept + [it for it in new if it not in kept]
            for it in n.items:
                if it["kind"] == "rename":
                    self.line_keys.setdefault(op, set()).add(it["new"])
                elif it["kind"] == "added":
                    self.line_keys.setdefault(op, set()).add(it["key"])
            n.open = False
            self._event("close", n, moved_to=moved_to, known_keys=known, new_items=new, kept_items=kept, **at)

```
