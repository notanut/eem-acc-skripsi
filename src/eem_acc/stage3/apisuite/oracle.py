"""Reference solvers and the checker entry points (S3.1 spec sections 1, 7, 11).

* `OracleAPI` knows the current schema exactly (via the suite's views) and translates the template's documented-form
  calls; it raises on any non-success.
* `DocsOnlyAPI` knows only what an agent could read: the rendered current docs (as `get_api_docs` returns them). It
  pairs renamed fields by elimination, maps remapped enum values by position, fills new required fields from the
  docs, and converts type-changed values by the documented kind.
* `NaiveAPI` sends the documented form unchanged (drift is then observable).
* `reference_plan` runs the oracle on a scratch world holding only the task's own entities, without drift: its trace
  is the instance's reference plan and its length is `min_calls` (D75).
"""
from __future__ import annotations

import copy
import re
from decimal import Decimal
from typing import Any

from .drift import to_current
from .ops import OPS
from .suite import ApiResponse, ApiSuite
from .templates import TEMPLATES, goals
from .world import World

MAX_STEPS_EXTRA = 1 + 6          # MaxSteps = min_calls + 1 (finish) + 6


class OracleError(RuntimeError):
    def __init__(self, op: str, resp: ApiResponse):
        super().__init__(f"{op}: {resp.status} {resp.body!r} {resp.truth}")
        self.op, self.resp = op, resp


def _is_success(resp: ApiResponse) -> bool:
    return resp.status == 200 and not (isinstance(resp.body, dict) and resp.body.get("status") == "error")


class _Base:
    def __init__(self, suite: ApiSuite, task_index: int, agent: Any = 0):
        self.suite, self.task_index, self.agent = suite, task_index, agent
        self.trace: list[tuple[str, dict]] = []
        self.responses: list[ApiResponse] = []
        self._k = 0

    def _send(self, op_id: str, args: dict, doc_op: str) -> ApiResponse:
        self._k += 1
        resp = self.suite.execute(op_id, args, self.task_index, call_key=(self.agent, self.task_index, self._k))
        self.trace.append((doc_op, args))
        self.responses.append(resp)
        return resp


class OracleAPI(_Base):
    def call(self, op: str, doc_args: dict) -> dict:
        v = self.suite.views(self.task_index)[op]
        resp = self._send(v.route, to_current(v, copy.deepcopy(doc_args)), op)
        if not _is_success(resp):
            raise OracleError(op, resp)
        return resp.body


class NaiveAPI(_Base):
    """Documented form, no translation. Returns the raw response (does not raise)."""

    def call_raw(self, op: str, doc_args: dict) -> ApiResponse:
        return self._send(op, copy.deepcopy(doc_args), op)


# ------------------------------------------------------------------------------------------------------- docs-only oracle
_LINE = re.compile(r"^(\s+)(\S+)\s+(required|optional)\s+(.*)$")


def parse_docs(text: str) -> tuple[str, dict]:
    """Rendered block -> (route, {path: (required, kind, info)})."""
    lines = text.splitlines()
    route = lines[0].split(" - ")[0].strip()
    fields: dict[str, tuple] = {}
    parent = None
    for ln in lines[1:]:
        m = _LINE.match(ln)
        if not m:
            continue
        indent, name, req, rest = len(m.group(1)), m.group(2), m.group(3) == "required", m.group(4)
        rest = re.sub(r" \(default .*\)$", "", rest)
        if rest.startswith("enum: "):
            kind, info = "enum", tuple(rest[6:].split(" | "))
        elif rest.startswith("constant: "):
            raw = rest[10:]
            kind, info = "const", (True if raw == "true" else False if raw == "false" else
                                   raw.strip('"') if raw.startswith('"') else int(raw))
        elif rest == "array of objects:":
            kind, info = "array_obj", None
        else:
            kind = {"string": "string", "integer": "integer", "number": "number", "boolean": "boolean",
                    "identifier": "id", "string of digits": "digits", "integer (0 or 1)": "int01",
                    "array of strings": "array_str"}.get(rest, "decimal" if rest.startswith("decimal string")
                                                          else rest)
            info = None
        if indent == 2:
            parent = name if kind == "array_obj" else None
            fields[name] = (req, kind, info)
        else:
            fields[f"{parent}[].{name}"] = (req, kind, info)
    return route, fields


def _convert(kind: str, v: Any) -> Any:
    if kind == "digits":
        return str(int(v))
    if kind == "decimal":
        return f"{Decimal(str(v)):.2f}"
    if kind == "int01":
        return 1 if v else 0
    if kind == "array_str":
        return [v]
    return v


def docs_translate(documented: str, current: str, doc_args: dict) -> tuple[str, dict]:
    """Translate documented-form args using only the two rendered blocks."""
    _, dfields = parse_docs(documented)
    route, cfields = parse_docs(current)
    missing = [p for p, (req, _, _) in dfields.items() if req and p not in cfields]
    appeared = [p for p in cfields if p not in dfields]
    rename: dict[str, str] = {}
    if missing:                       # rename + new required never share an operation (D69): appeared = renames
        for m in missing:
            par = m.split("[].")[0] if "[]." in m else None
            cands = [a for a in appeared if (a.split("[].")[0] if "[]." in a else None) == par]
            if len(cands) != 1:
                raise ValueError(f"ambiguous rename for {m}: {cands}")
            rename[m] = cands[0]
            appeared.remove(cands[0])

    def conv(path: str, v: Any) -> tuple[str, Any]:
        cur = rename.get(path, path)
        _, ck, cinfo = cfields[cur]
        _, dk, dinfo = dfields[path]
        if ck == "enum" and dk == "enum" and cinfo != dinfo:
            v = cinfo[dinfo.index(v)]
        elif ck != dk:
            v = _convert(ck, v)
        return cur.split("[].")[-1], v

    out: dict[str, Any] = {}
    for k, v in doc_args.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            items = []
            for it in v:
                ni = {}
                for ik, iv in it.items():
                    nk, nv = conv(f"{k}[].{ik}", iv)
                    ni[nk] = nv
                items.append(ni)
            out[rename.get(k, k)] = items
        else:
            nk, nv = conv(k, v)
            out[nk] = nv
    for a in appeared:               # new required fields
        req, kind, info = cfields[a]
        if req and "[]." not in a:
            out[a] = info if kind == "const" else info[0] if kind == "enum" else True if kind == "boolean" else None
    return route, out


class DocsOnlyAPI(_Base):
    def call(self, op: str, doc_args: dict) -> dict:
        route, args = docs_translate(self.suite.docs(op, "documented"), self.suite.docs(op, "current",
                                                                                       self.task_index), doc_args)
        resp = self._send(route, args, op)
        if not _is_success(resp):
            raise OracleError(op, resp)
        return resp.body


# --------------------------------------------------------------------------------------------------------- entry points
def scratch_suite(inst: dict) -> ApiSuite:
    """A suite over a world with only this task's entities and no drift."""
    w = World()
    w.add_setup(inst["setup"])
    spec = {"styles": {op: "informative" for op in OPS}, "drift_schedule": [[0, []]], "tasks": [inst]}
    return ApiSuite(w, spec)


def reference_plan(inst: dict) -> list[tuple[str, dict]]:
    s = scratch_suite(inst)
    api = OracleAPI(s, 0)
    TEMPLATES[inst["template"]].solve(api, inst)
    return api.trace


def min_calls(inst: dict) -> int:
    return len(reference_plan(inst))


def max_steps(inst: dict) -> int:
    return min_calls(inst) + MAX_STEPS_EXTRA


def check(inst: dict, world: World) -> tuple[bool, list]:
    g = goals(inst, world)
    return all(ok for _, ok in g), g
