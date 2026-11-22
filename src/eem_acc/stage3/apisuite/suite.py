"""The API suite: executes one call against the world under the stream's drift schedule (S3.1 spec sections 1, 4).

Validation order: route -> framework validation (strict; drift items first) -> app-level check of silently drifted
aspects (HTTP 200 + error payload, no state change) -> business rules -> execution. Idempotent by call_key (D68).
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any

from . import errors
from .docs import render
from .drift import OpView, canonicalize, effective, get_pair, op_view
from .ops import OPS, OP_NAMES
from .schema import Field, schema_paths, validate_object
from .world import World

SEMANTIC_STATUSES = frozenset({200, 400, 404, 409, 410, 422})
DOCS_OP = "get_api_docs"
DOCS_SCHEMA = (Field("operation", "string"),)


@dataclass(frozen=True)
class Truth:
    kind: str                              # ok | drift | silent_drift | agent
    pair_ids: tuple = ()                   # drift pairs involved (drift / silent_drift)
    reason: str | None = None              # agent: unknown_operation | unknown_resource | business_rule | documented_schema

    def as_json(self) -> dict:
        d = {"kind": self.kind}
        if self.pair_ids:
            d["pair_ids"] = list(self.pair_ids)
        if self.reason:
            d["reason"] = self.reason
        return d


@dataclass(frozen=True)
class ApiResponse:
    status: int
    body: Any
    truth: Truth
    op: str | None = None                  # the operation the call resolved to (None: unknown route)


class ApiSuite:
    def __init__(self, world: World, spec: dict):
        self.world = world
        self.spec = spec
        self.styles: dict[str, str] = dict(spec["styles"])
        self.schedule: list[tuple[int, list[str]]] = [(int(i), list(ids)) for i, ids in spec["drift_schedule"]]
        self.schedule.sort(key=lambda e: e[0])
        self._views: dict[int, dict[str, OpView]] = {}
        self._done: dict[Any, ApiResponse] = {}
        self.executions = 0                # calls that reached the semantics (not idempotent replays)

    # ------------------------------------------------------------------------------------------------------- views
    def active_pairs(self, task_index: int) -> list:
        ids: list[str] = []
        for start, pids in self.schedule:
            if start <= task_index:
                ids = pids
        return [get_pair(i) for i in ids]

    def views(self, task_index: int) -> dict[str, OpView]:
        key = max((s for s, _ in self.schedule if s <= task_index), default=-1)
        if key not in self._views:
            pairs = self.active_pairs(task_index)
            self._views[key] = {op: op_view(op, pairs) for op in OP_NAMES}
        return self._views[key]

    def docs(self, op: str, which: str = "current", task_index: int = 0) -> str:
        if which == "documented":
            return render(op_view(op, []))
        return render(self.views(task_index)[op])

    # ----------------------------------------------------------------------------------------------------- execute
    def execute(self, op_id: str, args: Any, task_index: int, call_key: Any = None) -> ApiResponse:
        sig = (op_id, json.dumps(args, sort_keys=True, default=str), task_index)
        if call_key is not None and call_key in self._done:
            stored_sig, stored = self._done[call_key]
            if stored_sig != sig:                      # a reused key with a different call is a caller bug
                raise ValueError(f"call_key {call_key!r} reused for a different call")
            return copy.deepcopy(stored)
        self.executions += 1
        resp = self._execute(op_id, args, task_index)
        if call_key is not None:
            self._done[call_key] = (sig, copy.deepcopy(resp))
        return resp

    def _execute(self, op_id: str, args: Any, task_index: int) -> ApiResponse:
        views = self.views(task_index)
        if op_id == DOCS_OP:
            return self._docs_call(args, views)
        by_route = {v.route: op for op, v in views.items()}
        if op_id in by_route:
            op = by_route[op_id]
        elif op_id in views and views[op_id].route != op_id:          # moved operation called by its old id
            v = views[op_id]
            pid = next(p.id for p in v.pairs if effective(p).type == "moved")
            st, body = errors.moved(self.styles[op_id], op_id, v.route)
            return ApiResponse(st, body, Truth("drift", (pid,)), op_id)
        else:
            return ApiResponse(404, errors.ROUTE_404_TERSE, Truth("agent", reason="unknown_operation"), None)
        v = views[op]
        style = self.styles[op]
        items = validate_object(v.schema, args)
        framework = [i for i in items if not v.is_silent(i.path, i.type)]
        silent = [i for i in items if v.is_silent(i.path, i.type)]
        if framework:
            drift_items = [i for i in framework if v.is_drift(i.path, i.type)]
            other = [i for i in framework if not v.is_drift(i.path, i.type)]
            st, body = errors.validation(style, drift_items + other)
            if drift_items:
                pids = tuple(dict.fromkeys(v.drift_paths[i.path] for i in drift_items))
                return ApiResponse(st, body, Truth("drift", pids), op)
            return ApiResponse(st, body, Truth("agent", reason="documented_schema"), op)
        if silent:
            pid = v.drift_paths[silent[0].path]
            return ApiResponse(200, errors.silent_body(self._silent_message(silent[0], style)),
                               Truth("silent_drift", (pid,)), op)
        res = OPS[op].handler(self.world, canonicalize(v, copy.deepcopy(args)))
        if res.kind == "ok":
            return ApiResponse(200, res.body, Truth("ok"), op)
        if res.kind == "not_found":
            st, body = errors.resource_404(style, res.reason)
            return ApiResponse(st, body, Truth("agent", reason="unknown_resource"), op)
        st, body = errors.business(style, res.status, res.reason)
        return ApiResponse(st, body, Truth("agent", reason="business_rule"), op)

    @staticmethod
    def _silent_message(item, style: str) -> str:
        """Mirrors what the 422 would say for the same item (D87): field name + the pydantic message, no more."""
        if style != errors.INFORMATIVE:
            return "Request failed"
        return f"Validation failed: field '{item.loc[-1]}': {item.msg}"

    def _docs_call(self, args: Any, views: dict[str, OpView]) -> ApiResponse:
        items = validate_object(DOCS_SCHEMA, args)
        if items:
            st, body = errors.validation(errors.INFORMATIVE, items)
            return ApiResponse(st, body, Truth("agent", reason="documented_schema"), DOCS_OP)
        name = args["operation"]
        by_route = {v.route: op for op, v in views.items()}
        op = by_route.get(name) or (name if name in views else None)
        if op is None:
            st, body = errors.resource_404(errors.INFORMATIVE, "operation")
            return ApiResponse(st, body, Truth("agent", reason="unknown_resource"), DOCS_OP)
        return ApiResponse(200, {"operation": views[op].route, "docs": render(views[op])}, Truth("ok"), DOCS_OP)
