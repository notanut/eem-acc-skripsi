"""ScriptedBackend: a deterministic rule-based policy that plays the LLM on CPU (S3.2a spec section 8).

It is a pure function of (messages, task instance, strategy): the instance is used only to know the template and its
reference `solve`; everything else (results of earlier calls, docs seen, memory lines) is read from the prompt. It
plans the next call by replaying `solve` against the success bodies parsed from the observations.

Strategies: `naive` (documented form), `docs` (after an error on an operation, look it up with get_api_docs, then
translate with the docs-only translator), `memory` (apply unmarked "Last working change" lines of notes on the
operation, else as `docs`). Results of the `memory` strategy are plumbing tests, never evidence.
"""
from __future__ import annotations

import copy
import json
import re
from decimal import Decimal
from typing import Any

from ..apisuite.docs import documented
from ..apisuite.oracle import docs_translate
from ..apisuite.templates import TEMPLATES
from ..feasibility import parse_action
from .backend import Gen
from .prompts import MEMORY_HEADER, strip_fence


class _Next(Exception):
    def __init__(self, op: str, call_args: dict):
        super().__init__(op)
        self.op, self.call_args = op, call_args


class _Replay:
    def __init__(self, bodies: list):
        self.bodies, self.i = bodies, 0

    def call(self, op: str, args: dict) -> dict:
        if self.i < len(self.bodies):
            b = self.bodies[self.i]
            self.i += 1
            return b
        raise _Next(op, args)


ITEM_RE = re.compile(r"'([^']+)' is now '([^']+)' \((\w+)\)|'([^']+)' added \((\w+)\)(?: = (\S+))?|"
                     r"'([^']+)' is now (\w+)|'([^']+)' = (\S+)|use operation '([^']+)'|'([^']+)' removed")


def parse_line(line: str) -> list[dict]:
    items = []
    for m in ITEM_RE.finditer(line):
        g = m.groups()
        if g[0]:
            items.append({"kind": "rename", "old": g[0], "new": g[1], "type": g[2]})
        elif g[3]:
            it = {"kind": "added", "key": g[3], "type": g[4]}
            if g[5]:
                it["value"] = json.loads(g[5].rstrip(";."))
            items.append(it)
        elif g[6]:
            items.append({"kind": "type", "key": g[6], "type": g[7]})
        elif g[8]:
            items.append({"kind": "value", "key": g[8], "value": json.loads(g[9].rstrip(";."))})
        elif g[10]:
            items.append({"kind": "moved", "op": g[10]})
        elif g[11]:
            items.append({"kind": "removed", "key": g[11]})
    return items


def _to_type(v: Any, t: str) -> Any:
    if t == "string":
        if isinstance(v, bool):
            return str(v).lower()
        if isinstance(v, int):
            return str(v)
        if isinstance(v, float):
            return f"{Decimal(str(v)):.2f}"
        return str(v)
    if t == "integer":
        return int(v)
    if t == "array":
        return [v]
    return v


def apply_items(op: str, args: dict, items: list[dict]) -> tuple[str, dict]:
    route = op
    out = copy.deepcopy(args)
    for it in items:
        k = it["kind"]
        if k == "moved":
            route = it["op"]
        elif k == "rename":
            if "[]." in it["old"]:
                parent, old = it["old"].split("[].")
                new = it["new"].split("[].")[1]
                for item in out.get(parent, []):
                    if old in item:
                        item[new] = item.pop(old)
            elif it["old"] in out:
                out[it["new"]] = out.pop(it["old"])
        elif k == "added" and "[]." not in it["key"]:
            out[it["key"]] = it.get("value", True if it["type"] == "boolean" else out.get(it["key"]))
        elif k == "value":
            out[it["key"]] = it["value"]
        elif k == "type":
            if "[]." in it["key"]:
                parent, sub = it["key"].split("[].")
                for item in out.get(parent, []):
                    if sub in item:
                        item[sub] = _to_type(item[sub], it["type"])
            elif it["key"] in out:
                out[it["key"]] = _to_type(out[it["key"]], it["type"])
        elif k == "removed":
            out.pop(it["key"], None)
    return route, out


class ScriptedBackend:
    def __init__(self, tokenizer, strategy: str = "docs", malformed_every: int = 0, echo_reflections: bool = False):
        self.tok, self.strategy = tokenizer, strategy
        self.malformed_every, self.echo = malformed_every, echo_reflections
        self.id = f"scripted-{strategy}" + (f"-mal{malformed_every}" if malformed_every else "") + \
                  ("-echo" if echo_reflections else "")
        self.task: dict | None = None
        self._calls = 0

    def set_task(self, inst: dict) -> None:
        self.task = inst

    # ---------------------------------------------------------------------------------------------------- parse
    @staticmethod
    def _turns(messages: list[dict]) -> list[tuple[dict, str]]:
        out = []
        for i in range(2, len(messages) - 1, 2):
            a = parse_action(strip_fence(messages[i]["content"])[0])
            out.append((a, messages[i + 1]["content"]))
        return out

    @staticmethod
    def _memory_lines(messages: list[dict]) -> list[str]:
        u = messages[1]["content"]
        if not u.startswith(MEMORY_HEADER):
            return []
        block = u.split("\n\nTask: ")[0]
        return [ln for ln in block.splitlines()[1:] if ln.startswith("- ")]

    # ------------------------------------------------------------------------------------------------- generate
    def generate(self, messages: list[dict], kind: str, grammar: bool, max_tokens: int) -> Gen:
        self._calls += 1
        if kind == "reflection":
            text = self._reflection(messages)
        else:
            text = self._action(messages)
            if self.malformed_every and not grammar and self._calls % self.malformed_every == 0:
                text = "```json\n" + text + "\n```" if (self._calls // self.malformed_every) % 2 else \
                    "I will call the API now. " + text
        n_in = self.tok.count_messages(messages)
        n_out = min(self.tok.count(text), max_tokens)
        return Gen(text, n_in, n_out, "stop", grammar)

    def _reflection(self, messages: list[dict]) -> str:
        c = messages[0]["content"]
        m = re.search(r"Failed call: (\S+) (\{.*\})", c)
        op = m.group(1) if m else "the operation"
        if self.echo and m:
            return f"Call {op} again with these arguments fixed: {m.group(2)}."
        return f"Use the parameter names and types that {op} expects now."

    def _action(self, messages: list[dict]) -> str:
        inst = self.task
        turns = self._turns(messages)
        bodies, docs_seen, failed_ops, last_fail = [], {}, set(), None
        for a, obs in turns:
            if a is None:
                continue
            op = a["action"]
            ok = obs.startswith("Observation: 200")
            if ok and op == "get_api_docs":
                docs_seen[a["args"].get("operation")] = obs.split("\n", 1)[1] if "\n" in obs else ""
                continue
            body = None
            if ok:
                try:
                    body = json.loads(obs[len("Observation: 200 "):].split("\n")[0])
                except json.JSONDecodeError:
                    body = {"status": "error"}
                assert len(obs) < len("Observation: 200 ") + 1500, "success body truncated"
            if ok and not (isinstance(body, dict) and body.get("status") == "error"):
                bodies.append(body)
                last_fail = None
            else:
                failed_ops.add(op)
                last_fail = op
        rp = _Replay(bodies)
        try:
            TEMPLATES[inst["template"]].solve(rp, inst)
        except _Next as nx:
            return self._envelope(*self._translate(nx.op, nx.call_args, docs_seen, failed_ops, last_fail, messages))
        return self._envelope("finish", {"summary": "done"})

    def _translate(self, op, doc_args, docs_seen, failed_ops, last_fail, messages):
        tried_here = last_fail == op                    # the previous attempt of this logical call failed
        if self.strategy == "naive":
            return op, doc_args
        if self.strategy == "prompt_docs":               # condition R: translate with the docs in the system prompt
            block = self._prompt_block(messages[0]["content"], op)
            return docs_translate(documented(op), block, doc_args) if block else (op, doc_args)
        if self.strategy == "memory" and not tried_here:
            items = []
            for ln in self._memory_lines(messages):
                if (ln.startswith(f"- {op} (") and "Last working change:" in ln
                        and "(possibly outdated)" not in ln):
                    items += parse_line(ln.split("Last working change:", 1)[1])
            moved = next((i["op"] for i in items if i["kind"] == "moved"), None)
            if moved:
                for ln in self._memory_lines(messages):
                    if ln.startswith(f"- {moved} (") and "Last working change:" in ln \
                            and "(possibly outdated)" not in ln:
                        items += parse_line(ln.split("Last working change:", 1)[1])
            items = [i for i in items if not (i["kind"] == "value" and i["key"] in doc_args)]
            if items:
                return apply_items(op, doc_args, items)
        # docs strategy (also the memory fallback)
        if op in docs_seen:
            route, args = docs_translate(documented(op), docs_seen[op], doc_args)
            return route, args
        if tried_here or op in failed_ops:
            return "get_api_docs", {"operation": op}
        return op, doc_args

    @staticmethod
    def _prompt_block(system: str, op: str) -> str | None:
        for block in system.split("\n\n"):
            first = block.split("\n", 1)[0]
            if first.startswith(f"{op} - ") or (block.count("\n") and f"(formerly {op})" in block.split("\n")[1]):
                return block
        return None

    @staticmethod
    def _envelope(action: str, args: dict) -> str:
        return json.dumps({"thought": f"Next I call {action}.", "action": action, "args": args})
