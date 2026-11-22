"""Agent loop (Algorithm 1) and RQ1 units (S3.2a spec sections 2-5, 9)."""
from __future__ import annotations

import asyncio
import copy
import dataclasses
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any

from ..apisuite import ApiSuite, World, check, min_calls
from ..apisuite.docs import documented_all, render
from ..apisuite.oracle import MAX_STEPS_EXTRA, reference_plan
from ..apisuite.ops import OP_NAMES
from ..apisuite.templates import TEMPLATES
from ..feasibility import parse_action
from . import prompts as P
from .backend import GRAMMARS, N_CTX, Gen, t_llm
from .memory import ErrorCtx, Memory, SuccessCtx, make_memory
from .parser import PARSER_VERSION, parse_error
from ..apisuite.drift import effective, get_pair
from .scrub import docs_vocabulary, task_scrubber

READS = {"find_customer", "get_cart", "get_order", "search_products", "check_stock", "track_shipment",
         "list_tickets"}


@dataclass
class Outcome:
    status: int
    body: Any
    transient: bool
    truth: dict
    canonical_op: str | None


class DirectDispatcher:
    """RQ1: the call goes straight to the suite (no overload; the design's generous fixed window)."""

    def __init__(self, suite: ApiSuite):
        self.suite = suite

    async def call(self, op_id: str, args: Any, agent: int, k: int, task_index: int) -> Outcome:
        r = self.suite.execute(op_id, args, task_index, call_key=(agent, k))
        return Outcome(r.status, r.body, False, r.truth.as_json(), r.op)


@dataclass
class LoopConfig:
    condition: str = "C0"
    docs_wording: str = "W1"
    action_max_tokens: int = 320
    grammar_always: bool = False
    reflection_max_tokens: int = 60
    k: int = 3
    k_s: int = 2
    theta_link: float = 0.65
    llm_time_scale: float = 1.0          # RQ3 sensitivity: LLM times / 3 (S3.2b); applied to every LLM call


def is_success(status: int, body: Any) -> bool:
    return status == 200 and not (isinstance(body, dict) and body.get("status") == "error")


class Agent:
    def __init__(self, agent_id: int, backend, tokenizer, memory: Memory, dispatcher, cfg: LoopConfig,
                 docs_for_task):
        self.id, self.backend, self.tok, self.memory, self.disp, self.cfg = (agent_id, backend, tokenizer, memory,
                                                                             dispatcher, cfg)
        self.docs_for_task = docs_for_task       # task_index -> docs text for the system prompt
        self.k = 0                               # running call counter across the stream (D68 guard)
        self.live: list = []                     # every step record as it is made (partial logs at T_max, S3.2b)
        self.vocab: set = set()                  # schema vocabulary this agent has observed (scrub safe list, M2)

    async def _llm(self, messages, kind, grammar, max_tokens, rec):
        assert grammar in GRAMMARS, grammar
        g: Gen = self.backend.generate(messages, kind, grammar, max_tokens)
        dt = t_llm(g) * self.cfg.llm_time_scale
        rec["llm"].append({"kind": kind, "n_in": g.n_in, "n_out": g.n_out, "grammar": g.grammar,
                           "finish_reason": g.finish_reason, "t_llm": round(dt, 6)})
        await asyncio.sleep(dt)
        return g

    async def run_task(self, inst: dict) -> dict:
        cfg = self.cfg
        ti = inst["index"]
        max_steps = min_calls(inst) + MAX_STEPS_EXTRA
        system = P.system_prompt(self.docs_for_task(ti), cfg.docs_wording)
        traj: list[tuple[str, str]] = []
        scrub = task_scrubber(inst, self.vocab)
        steps_log: list[dict] = []
        last_action, last_keys, last_obs = None, [], ""
        reason, finished = "max_steps", False
        n_api = n_llm = tok_in = tok_out = 0
        t_total = 0.0
        echo_values: set = set()
        for step in range(max_steps):
            rec: dict[str, Any] = {"agent": self.id, "task": ti, "step": step, "llm": [],
                                   "t_start": round(asyncio.get_running_loop().time(), 6)}
            self.live.append(rec)
            query = P.state_text(inst["instruction"], last_action, last_keys, last_obs) + "\n" + P.tools_desc()
            rec["visible"] = self.memory.visible()            # retrieval-recall denominator at retrieval time (L5)
            selected = self.memory.retrieve(query)
            lines = self.memory.render(selected)
            rec["retrieved"] = self.memory.selected_ids(selected)
            stats = self.memory.render_stats(selected)
            if stats:
                rec["note_stats"] = stats                          # line_shown, items shown/total (second opinion)
            rec["memory_sha"] = hashlib.sha256("\n".join(lines).encode()).hexdigest()[:16] if lines else None
            messages = P.build_messages(system, inst["instruction"], lines, traj)
            if self.tok.count_messages(messages) + cfg.action_max_tokens > N_CTX:
                reason = "context_overflow"
                rec["event"] = "context_overflow"             # logged, not a step (L3)
                steps_log.append(rec)
                break
            g = await self._llm(messages, "action", "envelope" if cfg.grammar_always else None,
                                cfg.action_max_tokens, rec)
            text, fenced = P.strip_fence(g.text)
            parsed = parse_action(text)
            rec["fence"] = fenced
            rec["fallback"] = False
            if parsed is None and not cfg.grammar_always:
                rec["fallback"] = True
                rec["first_reply"] = g.text[:400]              # the unparsed first reply is kept in the log (L2)
                g = await self._llm(messages, "action", "envelope_cap300", cfg.action_max_tokens, rec)
                text, f2 = P.strip_fence(g.text)
                rec["fallback_fence"] = f2
                parsed = parse_action(text)
            if parsed is None:
                rec["format_error"] = True
                obs = P.FORMAT_ERROR
                traj.append((g.text[:400], obs))
                last_action, last_keys, last_obs = None, [], obs
                steps_log.append(rec)
                continue
            rec["thought_len"] = len(parsed["thought"])
            action, args = parsed["action"], parsed["args"]
            rec["action"], rec["arg_keys"] = action, sorted(map(str, args))
            if action not in ("finish", "get_api_docs") and \
                    any(isinstance(v, str) and PLACEHOLDER_RE.search(v) for v in _leaves(args)):
                rec["placeholder_args"] = True                 # D104
            if action == "finish":
                finished = True
                rec["finish"] = True
                steps_log.append(rec)
                traj.append((text, ""))
                break
            self.k += 1
            n_api += 1
            out = await self.disp.call(action, args, self.id, self.k, ti)
            rec.update({"k": self.k, "status": out.status, "truth": out.truth, "transient": out.transient,
                        "canonical_op": out.canonical_op,
                        "notes_on_op": self.memory.notes_on(out.canonical_op or action)})   # retrieval recall
            if action != "get_api_docs":                       # operation names are never scrub values (H1)
                scrub.add_tree(args)
            if out.transient:
                obs = P.TRANSIENT
            elif is_success(out.status, out.body):
                obs, trunc = P.success_observation(action, out.body)
                rec["truncated"] = trunc
                if action != "get_api_docs":
                    scrub.add_tree(out.body)
                    echo_values |= _enum_like_values(out.body)
                elif isinstance(out.body, dict) and "docs" in out.body:
                    self.vocab |= docs_vocabulary(out.body["docs"])
                self.memory.on_success(SuccessCtx(self.id, ti, step, action, out.canonical_op or action, args,
                                                  out.body, asyncio.get_running_loop().time()))
            else:
                obs = P.error_observation(out.status, out.body)
                api_obs = obs
                if action != "get_api_docs":
                    shown = await self._on_error(inst, step, action, args, out, scrub, rec)
                    if shown:
                        obs += "\n" + shown
                    if _echo_trap(out.truth, args, echo_values):
                        rec["echo_trap"] = True
                traj.append((text, obs))
                last_action, last_keys, last_obs = action, sorted(map(str, args)), api_obs   # D105: no note
                steps_log.append(rec)
                continue
            traj.append((text, obs))
            last_action, last_keys, last_obs = action, sorted(map(str, args)), obs
            steps_log.append(rec)
        for rec in steps_log:
            for c in rec["llm"]:
                n_llm += 1
                tok_in += c["n_in"]
                tok_out += c["n_out"]
                t_total += c["t_llm"]
        return {"traj": traj, "steps": steps_log, "finished": finished, "reason": reason, "n_api": n_api,
                "n_llm": n_llm, "tok_in": tok_in, "tok_out": tok_out, "t_llm": round(t_total, 6),
                "max_steps": max_steps}

    async def _on_error(self, inst, step, action, args, out, scrub, rec):
        kt = parse_error(action, out.status, out.body, args)
        self.vocab |= set(kt.stated_values)
        # defence in depth: the parsed key is scrubbed too (the agent controls action and key names)
        kt = dataclasses.replace(kt, api_name=scrub.scrub(kt.api_name)[0], expected=scrub.scrub(kt.expected)[0],
                                 received=scrub.scrub(kt.received)[0])
        raw = (f"{out.status} " + P.body_text(out.body))[:P.ERROR_CAP]
        mode = self.memory.reflection_mode
        reflection = None
        if mode in ("kt", "raw"):
            err = kt.render() if mode == "kt" else raw
            msgs = P.reflection_messages(inst["instruction"], action, args, err)
            if self.tok.count_messages(msgs) + self.cfg.reflection_max_tokens > N_CTX:
                rec["reflection_skipped"] = True
            else:
                g = await self._llm(msgs, "reflection", None, self.cfg.reflection_max_tokens, rec)
                reflection, n = scrub.scrub(g.text.strip())
                rec["scrubbed"] = n
        raw_s, n2 = scrub.scrub(raw)
        if n2 and mode is None:                            # A1 stores the raw text
            rec["scrubbed"] = rec.get("scrubbed", 0) + n2
        ctx = ErrorCtx(self.id, inst["index"], step, action, out.canonical_op or action, copy.deepcopy(args),
                       out.status, out.body, raw_s, kt, out.truth, asyncio.get_running_loop().time())
        shown = self.memory.on_error(ctx, reflection)
        rec["kt"] = {"error_type": kt.error_type, "expected": kt.expected, "received": kt.received}
        if shown is None:
            return None
        return shown if shown.startswith("Reflection: ") else "Note: " + shown


PLACEHOLDER_RE = re.compile(r"<[a-z_]+>")


def _leaves(x: Any):
    if isinstance(x, dict):
        for v in x.values():
            yield from _leaves(v)
    elif isinstance(x, list):
        for v in x:
            yield from _leaves(v)
    else:
        yield x


def _echo_trap(truth: dict, args: Any, echo_values: set) -> bool:
    """An error from an enum drift (visible or silent) whose sent value of that field is a documented value echoed in
    an earlier success body of the task (M3)."""
    if not isinstance(args, dict):
        return False
    for pid in truth.get("pair_ids", []):
        e = effective(get_pair(pid))
        if e.type == "enum":
            field = e.path.split("[].")[-1]
            if isinstance(args.get(field), str) and args[field] in echo_values:
                return True
    return False


def _enum_like_values(body: Any) -> set:
    out: set = set()
    if isinstance(body, dict):
        for k, v in body.items():
            if isinstance(v, str) and k in ("warehouse", "status", "state", "shipping_method", "category", "method"):
                out.add(v)
            elif isinstance(v, (dict, list)):
                out |= _enum_like_values(v)
    elif isinstance(body, list):
        for v in body:
            out |= _enum_like_values(v)
    return out


def task_summary(inst: dict, res: dict, world: World, suite: ApiSuite) -> dict:
    ok, goals = check(inst, world)
    success = res["finished"] and ok
    reason = "success" if success else ("wrong_state" if res["finished"] else res["reason"])
    drift_errors = sum(1 for s in res["steps"] if s.get("truth", {}).get("kind") in ("drift", "silent_drift"))
    agent_errors = sum(1 for s in res["steps"] if s.get("truth", {}).get("kind") == "agent")
    active = {p.op for p in suite.active_pairs(inst["index"])}
    planned = sorted(TEMPLATES[inst["template"]].covers & active)
    first_call = {}
    for s in res["steps"]:
        op = s.get("canonical_op")
        if op in active and op not in first_call and s.get("action") != "get_api_docs":
            first_call[op] = "ok" if s.get("truth", {}).get("kind") == "ok" else s.get("truth", {}).get("kind")
    plan_reads = [op for op, _ in reference_plan(inst) if op in READS]
    called = [s.get("canonical_op") for s in res["steps"]]
    skipped = sum(max(0, plan_reads.count(op) - called.count(op)) for op in set(plan_reads))
    return {"task": inst["index"], "template": inst["template"], "success": success, "reason": reason,
            "state_ok_at_end": ok, "failed_subgoals": [n for n, g in goals if not g],
            "steps": sum(1 for s in res["steps"] if s.get("event") != "context_overflow"),
            "max_steps": res["max_steps"], "api_calls": res["n_api"],
            "llm_calls": res["n_llm"], "tok_in": res["tok_in"], "tok_out": res["tok_out"], "t_llm": res["t_llm"],
            "drift_errors": drift_errors, "agent_errors": agent_errors,
            "format_errors": sum(1 for s in res["steps"] if s.get("format_error")),
            "fallbacks": sum(1 for s in res["steps"] if s.get("fallback")),
            "exposures_planned": planned, "first_call": first_call, "skipped_reads": skipped,
            "echo_traps": sum(1 for s in res["steps"] if s.get("echo_trap")),
            "scrubbed": sum(s.get("scrubbed", 0) for s in res["steps"]),
            "placeholder_args": sum(1 for s in res["steps"] if s.get("placeholder_args"))}


def run_rq1_unit(spec: dict, condition: str, backend, embedder, tokenizer, cfg: LoopConfig | None = None,
                 n_tasks: int | None = None, embedding_preload: list | None = None) -> dict:
    """One result unit: one condition on one stream (store reset at the start)."""
    from ...env.vtime import run_virtual
    cfg = dataclasses.replace(cfg or LoopConfig(), condition=condition)       # never mutate the caller's config
    world = World.build(spec)
    suite = ApiSuite(world, spec)
    embedder = UnitEmbedder(embedder, embedding_preload)                    # per-unit cache, saved (M9)
    memory = make_memory(condition, embedder, tokenizer, cfg.k, cfg.k_s, cfg.theta_link)
    documented = documented_all()

    def docs_for_task(ti: int) -> str:
        if condition == "R":
            v = suite.views(ti)
            return "\n\n".join(render(v[op]) for op in OP_NAMES)
        return documented

    agent = Agent(0, backend, tokenizer, memory, DirectDispatcher(suite), cfg, docs_for_task)
    tasks = spec["tasks"][:n_tasks] if n_tasks else spec["tasks"]

    async def main():
        out = []
        if hasattr(backend, "register_tasks"):
            backend.register_tasks(tasks)
        for inst in tasks:
            res = await agent.run_task(inst)
            out.append((inst, res))
        return out

    results = run_virtual(main())
    summaries = [task_summary(inst, res, world, suite) for inst, res in results]
    return {"stamp": {"suite_version": spec["version"], "spec_sha256": spec["sha256"], "condition": condition,
                      "params": {"k": cfg.k, "k_s": cfg.k_s, "theta_link": cfg.theta_link,
                                 "docs_wording": cfg.docs_wording, "action_max_tokens": cfg.action_max_tokens,
                                 "grammar_always": cfg.grammar_always, "llm_time_scale": cfg.llm_time_scale},
                      "parser_version": PARSER_VERSION, "prompt_version": P.PROMPT_VERSION,
                      "prompt_sha256": P.prompt_hash(),
                      "backend": getattr(backend, "id", type(backend).__name__), "tokenizer": tokenizer.id,
                      "embedder": embedder.id, "code_commit": code_commit()},
            "tasks": summaries, "steps": [s for _, r in results for s in r["steps"]],
            "trajectories": [r["traj"] for _, r in results],
            "note_events": memory.events, "final_notes": memory.final_state(),
            "embedding_cache": embedder.export()}


class UnitEmbedder:
    """Per-unit view of a (possibly shared) embedder: records every vector the unit used, keyed by text sha256, so the
    unit can be replayed with its own vectors (M9; S3.3 computes cosines from these)."""

    def __init__(self, base, preload: list | None = None):
        """`preload` = a unit's exported cache: on replay its vectors are used instead of recomputing (S3.3)."""
        self.base, self.id, self.used = base, base.id, {}
        self.preload = {k: tuple(v) for k, v in (preload or [])}

    def embed(self, text: str) -> tuple:
        key = hashlib.sha256(text.encode()).hexdigest()
        v = self.preload[key] if key in self.preload else self.base.embed(text)
        self.used[key] = v
        return v

    def export(self) -> list:
        return [[k, list(v)] for k, v in sorted(self.used.items())]


def code_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
                              cwd=__file__.rsplit("/src/", 1)[0]).stdout.strip() or "unknown"
    except Exception:          # noqa: BLE001 - outside a checkout
        return "unknown"
