"""RQ3 wiring (S3.2b spec): 4 LLM agents (S3.2a) + 32 scripted background agents through the RQ2 gateway and a
semantic CAP-steady server, in virtual time. The RQ2 simulator code is reused unchanged."""
from __future__ import annotations

import asyncio
import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ..acc import ACCController
from ..agent import Gateway
from ..env import MockServer, Response, run_virtual
from ..metrics import AnalysisConfig, compute_metrics_v2
from ..scenarios import L_TARGET, SUCCESS_GRACE, acc_method, build, fixed_method
from ..sim import SIM_VERSION
from ..status import RETRYABLE
from .agent.loop import Agent, LoopConfig, Outcome, UnitEmbedder, code_commit, task_summary
from .agent.memory import make_memory
from .agent import prompts as P
from .agent.parser import PARSER_VERSION
from .apisuite import ApiSuite, World
from .apisuite.docs import documented_all

RQ3_VERSION = "rq3-v1"
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


def _llm_metrics(server: SemanticServer, timeout: float, t0: float) -> dict:
    att = [e for e in server.log if isinstance(e[0], LLMRequest) and e[1] >= t0]
    n = len(att)
    if not n:
        return {"n_attempts": 0}
    overload = sum(1 for _, _, st, lat in att if st in (429, 503) or (st == 200 and lat > timeout))
    lat200 = [min(lat, timeout) for _, _, st, lat in att if st == 200]
    excess = [max(0.0, (min(lat, timeout) - L_TARGET) / L_TARGET) for _, _, st, lat in att]
    return {"n_attempts": n, "overload_rate": overload / n,
            "slo_violation_rate": (sum(1 for x in lat200 if x > L_TARGET) / len(lat200)) if lat200 else None,
            "mean_latency_excess": float(np.mean(excess)),
            "lat_p95": float(np.percentile(lat200, 95)) if lat200 else None}


def run_rq3_unit(spec: dict, condition: str, window: str, backend, embedder, tokenizer,
                 cfg: LoopConfig | None = None, warmup: float = WARMUP, t_max: float = T_MAX,
                 server_override: dict | None = None, retry_override: dict | None = None) -> dict:
    assert spec["kind"] == "rq3" and condition in ("C0", "C2", "C2-private")
    seed = spec["seed"]
    sc = build("CAP-steady", N_SCRIPTED, method_for(window), seed)
    server_cfg = dataclasses.replace(sc.server, **(server_override or {}))
    retry = dataclasses.replace(sc.retry, **(retry_override or {}))
    base_cond = "C2" if condition.startswith("C2") else "C0"
    cfg = dataclasses.replace(cfg or LoopConfig(), condition=base_cond)
    tasks = spec["tasks"]
    if hasattr(backend, "register_tasks"):
        backend.register_tasks(tasks)
    emb = UnitEmbedder(embedder)

    async def main():
        loop = asyncio.get_running_loop()
        ss = np.random.SeedSequence(seed).spawn(3)                       # sim.py layout
        seeds = [int(s.generate_state(1)[0]) for s in ss]
        world = World.build(spec)
        suite = ApiSuite(world, spec)
        server = SemanticServer(server_cfg, seeds[0], suite)
        gw = Gateway(server, ACCController(sc.acc), retry, seeds[1])
        disp = GatewayDispatcher(gw, suite)
        stop = asyncio.Event()
        children = np.random.SeedSequence(seeds[2]).spawn(N_SCRIPTED + 1)  # run_agents layout
        think = sc.workload.phases[0][1]
        scripted = [asyncio.ensure_future(_scripted_agent(gw, think, sc.workload.start_jitter,
                                                          np.random.default_rng(c), i, stop))
                    for i, c in enumerate(children[:N_SCRIPTED])]
        shared = make_memory(base_cond, emb, tokenizer, cfg.k, cfg.k_s, cfg.theta_link)
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
        done, pending = await asyncio.wait(llm, timeout=max(0.0, t_max - loop.time()))
        t_end = loop.time()
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for t in done:
            if t.exception():
                raise t.exception()
        stop.set()
        for t in scripted:
            t.cancel()
        out = await asyncio.gather(*scripted, return_exceptions=True)
        for r in out:
            if isinstance(r, BaseException) and not isinstance(r, asyncio.CancelledError):
                raise r
        await gw.shutdown()
        return world, suite, server, gw, disp, agents, memories, timing, results, t_end, bool(pending)

    world, suite, server, gw, disp, agents, memories, timing, results, t_end, capped = run_virtual(main())
    summaries = []
    for inst in tasks:
        ti = inst["index"]
        agent = next(a for a, ids in enumerate(spec["agents"]) if ti in ids)
        if ti in results:
            res = results[ti]
        else:                                                         # cancelled at T_max or never started
            steps = [r for r in agents[agent].live if r["task"] == ti]
            res = {"traj": [], "steps": steps, "finished": False, "reason": "unfinished_at_t_max", "n_api": sum(
                1 for r in steps if "k" in r), "n_llm": 0, "tok_in": 0, "tok_out": 0, "t_llm": 0.0,
                   "max_steps": None}
        s = task_summary(inst, res, world, suite) if res["max_steps"] is not None else {
            "task": ti, "template": inst["template"], "success": False, "reason": "unfinished_at_t_max",
            "steps": len(res["steps"]), "api_calls": res["n_api"]}
        s["agent"] = LLM_AGENT_BASE + agent
        t = timing.get(ti, {})
        s["t_start"] = t.get("t_start")
        s["t_finish"] = t.get("t_finish")
        s["completion_time"] = (t["t_finish"] - t["t_start"]) if "t_finish" in t else (
            (t_end - t["t_start"]) if "t_start" in t else None)
        s["unfinished_at_t_max"] = ti not in results
        summaries.append(s)
    timeout = gw.retry.timeout
    n_succ = sum(1 for s in summaries if s["success"])
    llm_calls = disp.calls
    llm_m = _llm_metrics(server, timeout, warmup)
    analysis = AnalysisConfig(windows=(("llm_phase", warmup, t_end),), success_grace=SUCCESS_GRACE)
    infra = compute_metrics_v2(gw, L_TARGET, t_end, analysis)
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
                             "docs_wording": cfg.docs_wording}, "parser_version": PARSER_VERSION,
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
        "drift_errors": sum(s.get("drift_errors", 0) for s in summaries),
        "llm_metrics": llm_m, "J_rq3": j, "infra": infra,
        "window_trace": [list(x) for x in gw.controller.trace],
        "steps": [r for a in agents for r in a.live],
        "final_notes": final_notes, "note_events": note_events, "embedding_cache": emb.export(),
    }
