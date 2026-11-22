"""Memory for every condition (S3.2a spec sections 6.3-6.5, 7): the EEM store (C2, C3, AN, A1, A2, C2-OU), the
reflection list (C1), the FIFO buffer (F) and no memory (C0, R)."""
from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from ..apisuite.leakcheck import item_holds, last_working_items, render_item, render_items
from ..apisuite.ops import OPS
from .backend import Embedder, Tokenizer, cosine
from .parser import KT, moved_pointer
from .prompts import A2_TEMPLATE

NOTE_CAP, KT_CAP, REFL_CAP, LINE_CAP = 120, 30, 60, 40      # D112: the line (<= 40) is budgeted before the reflection
MOVED_TYPES = ("operation_removed", "route_not_found")
# D114: only errors that can signal drift mark other lessons on the operation as possibly outdated
DRIFT_SIGNAL = {"missing_field", "unknown_field", "type_error", "enum_error", "literal_error", "format_error",
                "value_error", "bad_request", "silent_error", "route_not_found", "operation_removed"}


@dataclass
class ErrorCtx:
    agent: int
    task: int
    step: int
    api_name: str                 # the id the agent called
    suite_op: str                 # the suite's operation: for logs only, never used by EEM (second opinion M2)
    args: Any
    status: int
    body: Any
    raw_text: str                 # "<status> <body>" cut at 400 characters
    kt: KT
    truth: dict
    t: float = 0.0                # virtual time (note visibility, S3.2a spec 6.5)


@dataclass
class SuccessCtx:
    agent: int
    task: int
    step: int
    called_id: str
    suite_op: str                 # logs only (M2)
    args: Any
    body: Any
    t: float = 0.0


@dataclass
class Note:
    id: int
    api_name: str
    canonical_op: str
    error_type: str
    kt: KT
    reflection: str | None
    trigger_args: Any
    owner: tuple
    text_raw: str | None = None          # A1
    items: list = field(default_factory=list)
    open: bool = True
    possibly_outdated: bool = False
    links: set = field(default_factory=set)
    z: tuple = ()
    truth: dict = field(default_factory=dict)

    def snapshot(self) -> dict:
        return {"id": self.id, "api_name": self.api_name, "canonical_op": self.canonical_op,
                "error_type": self.error_type, "kt": {"expected": self.kt.expected, "received": self.kt.received,
                                                     "error_keys": sorted(self.kt.error_keys)},
                "reflection": self.reflection, "text_raw": self.text_raw, "trigger_args": self.trigger_args,
                "owner": list(self.owner), "items": copy.deepcopy(self.items), "line": render_items(self.items),
                "open": self.open, "possibly_outdated": self.possibly_outdated, "links": sorted(self.links),
                "truth": self.truth}


class Memory:
    """Base: no memory (C0, R)."""
    reflection_mode: str | None = None   # "kt" | "raw" | "template" | None
    name = "none"

    def __init__(self, embedder: Embedder | None = None, tokenizer: Tokenizer | None = None, k: int = 3):
        self.embedder, self.tokenizer, self.k = embedder, tokenizer, k
        self.events: list[dict] = []

    def retrieve(self, query: str) -> list:
        return []

    def render(self, selected: list) -> list[str]:
        return []

    def render_stats(self, selected: list) -> list:
        return []

    def selected_ids(self, selected: list) -> list:
        return []

    def on_error(self, ctx: ErrorCtx, reflection: str | None) -> str | None:
        return None

    def on_success(self, ctx: SuccessCtx) -> None:
        return None

    def notes_on(self, op: str) -> list[int]:
        return []

    def visible(self) -> list:
        return []

    def final_state(self) -> list[dict]:
        return []


class ReflectionList(Memory):
    """C1: reflections after non-transient errors, no merge; top-k by cosine with the query."""
    reflection_mode = "raw"
    name = "reflection_rag"

    def __init__(self, embedder, tokenizer, k=3):
        super().__init__(embedder, tokenizer, k)
        self.items: list[dict] = []

    def on_error(self, ctx, reflection):
        if not reflection:
            return None
        it = {"id": len(self.items) + 1, "text": reflection, "z": self.embedder.embed(reflection),
              "suite_op": ctx.suite_op, "owner": [ctx.agent, ctx.task]}
        self.items.append(it)
        self.events.append({"event": "create", "id": it["id"], "text": reflection, "truth": ctx.truth,
                            "suite_op": ctx.suite_op, "t": round(ctx.t, 6)})
        return "Reflection: " + self._cap(reflection)

    def _cap(self, text):
        return self.tokenizer.truncate(text, REFL_CAP)

    def retrieve(self, query):
        q = self.embedder.embed(query)
        ranked = sorted(self.items, key=lambda it: (-cosine(q, it["z"]), it["id"]))
        return ranked[:self.k]

    def render(self, selected):
        return [f"- {self._cap(it['text'])}" for it in selected]

    def selected_ids(self, selected):
        return [it["id"] for it in selected]

    def notes_on(self, op):
        return [it["id"] for it in self.items if it["suite_op"] == op]

    def visible(self):
        return [[it["id"], it["suite_op"]] for it in self.items]

    def final_state(self):
        return [{k: v for k, v in it.items() if k != "z"} for it in self.items]


class FifoReflections(ReflectionList):
    """F: the last k reflections, carried across tasks, oldest first (no retrieval)."""
    name = "fifo"

    def retrieve(self, query):
        return self.items[-self.k:]


class EEMStore(Memory):
    """C2 (links), C3 (flat), AN (all notes), A2 (template reflection), C2-OU (no outcome update), A1 (raw text)."""

    def __init__(self, embedder, tokenizer, k=3, k_s=2, theta_link=0.65, *, links=True, outcome=True,
                 all_notes=False, mode="eem", name="eem"):
        super().__init__(embedder, tokenizer, k)
        self.k_s, self.theta = k_s, theta_link
        self.links_on, self.outcome_on, self.all_notes, self.mode, self.name = links, outcome, all_notes, mode, name
        self.reflection_mode = {"eem": "kt", "a2": "template", "a1": None}[mode]
        self.notes: dict[int, Note] = {}
        self.index: dict[tuple, int] = {}
        self.line_keys: dict[str, set] = {}          # canonical op -> undocumented keys ever introduced by a line
        self.docs_named: dict[tuple, str] = {}       # (agent, task, op) -> id named by get_api_docs(op)
        self.alias: dict[str, str] = {}              # observed new id -> documented id (M2)
        self._next = 1

    # ------------------------------------------------------------------------------------------------- aliases
    def canonical(self, api_name: str) -> str:
        """The documented operation an id stands for, from what the agents observed (410 pointers, get_api_docs
        responses, moved items of outcome lines); never from the suite (second opinion M2)."""
        seen = set()
        while api_name in self.alias and api_name not in seen:
            seen.add(api_name)
            api_name = self.alias[api_name]
        return api_name

    # ------------------------------------------------------------------------------------------------- note text
    def _embed_text(self, n: Note) -> str:
        if self.mode == "a1":
            return n.text_raw or ""
        return f"{n.reflection or ''} || {n.kt.render()}"

    def note_render(self, n: Note) -> tuple[str, dict]:
        """The note as shown in prompts, at most NOTE_CAP tokens including the "- " prefix. Budget order (D112):
        k_t (<= 30), then the outcome line (<= 40 with its label and mark, shortened at item boundaries, D106), then
        the reflection gets the rest. Display order: k_t, reflection, line. Returns (text, stats)."""
        t = self.tokenizer
        cap = NOTE_CAP - t.count("-")
        stats = {"id": n.id, "items_total": len(n.items), "items_shown": 0, "line_shown": False}
        if self.mode == "a1":
            return "- " + t.truncate(f"{n.api_name}: {n.text_raw}", cap), stats
        head = t.truncate(n.kt.render(), KT_CAP).rstrip(". ")
        line = ""
        if n.items:
            mark = " (possibly outdated)" if n.possibly_outdated else ""
            rendered = [render_item(it) for it in n.items]
            for keep in range(len(rendered), 0, -1):
                more = " ..." if keep < len(rendered) else ""
                cand = "Last working change: " + "; ".join(rendered[:keep]) + more + mark
                if t.count(cand) <= LINE_CAP:
                    line = cand
                    stats.update(items_shown=keep, line_shown=True)
                    break
        refl = ""
        if n.reflection:
            label = "" if self.mode == "a2" else "Fix: "
            fixed = ". ".join(x for x in (head, label + "x", line) if x)
            room = min(REFL_CAP, cap - t.count(fixed) + 1)
            while room > 0:
                cand = t.truncate(n.reflection, room).rstrip(". ")
                text = ". ".join(x for x in (head, label + cand, line) if x)
                if t.count(text) <= cap:
                    refl = label + cand
                    break
                room -= 1
        return "- " + ". ".join(x for x in (head, refl, line) if x), stats

    def note_line(self, n: Note) -> str:
        return self.note_render(n)[0]

    # ---------------------------------------------------------------------------------------------------- store
    def _relink(self, n: Note) -> None:
        for other_id in list(n.links):
            self.notes[other_id].links.discard(n.id)
        n.links = set()
        if not self.links_on:
            return
        for o in self.notes.values():
            if o.id != n.id and cosine(n.z, o.z) > self.theta:
                n.links.add(o.id)
                o.links.add(n.id)

    def _event(self, kind: str, n: Note, **extra) -> None:
        self.events.append({"event": kind, **n.snapshot(), **extra})

    def on_error(self, ctx, reflection):
        ptr = moved_pointer(ctx.kt)
        if ptr:
            self.alias[ptr] = ctx.api_name
        op = self.canonical(ctx.api_name)
        key = (ctx.api_name, ctx.kt.error_type)
        if self.mode == "a1":
            reflection = None
        if self.mode == "a2":
            reflection = A2_TEMPLATE.format(api=ctx.api_name)
        nid = self.index.get(key)
        at = dict(at=[ctx.agent, ctx.task, ctx.step], t=round(ctx.t, 6), suite_op=ctx.suite_op)
        if nid is not None:
            n = self.notes[nid]
            kept = n.items if self.outcome_on else []
            n.kt, n.reflection, n.trigger_args, n.owner = ctx.kt, reflection, copy.deepcopy(ctx.args), (ctx.agent,
                                                                                                       ctx.task)
            n.text_raw = ctx.raw_text if self.mode == "a1" else None
            # D113: a merge keeps the old items as a possibly-outdated part (re-checked at the next success)
            n.items, n.open, n.possibly_outdated, n.truth = list(kept), True, bool(kept), ctx.truth
            kind = "merge"
            at["items_kept"] = len(kept)
        else:
            n = Note(self._next, ctx.api_name, op, ctx.kt.error_type, ctx.kt, reflection,
                     copy.deepcopy(ctx.args), (ctx.agent, ctx.task),
                     text_raw=ctx.raw_text if self.mode == "a1" else None, truth=ctx.truth)
            self._next += 1
            self.notes[n.id] = n
            self.index[key] = n.id
            kind = "create"
        n.z = self.embedder.embed(self._embed_text(n))
        self._relink(n)
        if self.outcome_on and ctx.kt.error_type in DRIFT_SIGNAL:      # D114
            for o in self.notes.values():
                if o.id != n.id and o.canonical_op == op and o.items and not o.possibly_outdated:
                    o.possibly_outdated = True
                    self._event("mark", o, **at)
        self._event(kind, n, **at)
        return None if self.mode == "a1" else self.note_line(n)[2:]

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
            if n.canonical_op == op and n.possibly_outdated:
                n.items = [it for it in n.items if item_holds(it, ctx.args, ctx.called_id)]
                n.possibly_outdated = False
                self._event("recheck", n, **at)
        for n in sorted(self.notes.values(), key=lambda n: n.id):      # close own open notes on this operation
            if not n.open or n.owner != (ctx.agent, ctx.task) or n.canonical_op != op:
                continue
            documented = OPS[op].schema if op in OPS else ()
            if n.error_type in MOVED_TYPES:
                if ctx.called_id == n.api_name:
                    n.open = False                                   # not a move after all: no new line
                    self._event("close", n, moved_to=None, known_keys=[], **at)
                    continue
                named = moved_pointer(n.kt) or self.docs_named.get((ctx.agent, ctx.task, n.api_name))
                if ctx.called_id != named:
                    continue
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
                elif it["kind"] == "moved":
                    self.alias[it["op"]] = op
            n.open = False
            self._event("close", n, moved_to=moved_to, known_keys=known, new_items=new, **at)   # M5

    # ------------------------------------------------------------------------------------------------ retrieval
    def retrieve(self, query):
        notes = sorted(self.notes.values(), key=lambda n: n.id)
        if self.all_notes:
            return notes
        q = self.embedder.embed(query)
        sim = {n.id: cosine(q, n.z) for n in notes}
        ranked = sorted(notes, key=lambda n: (-sim[n.id], n.id))
        if not self.links_on:
            return ranked[:self.k]
        chosen = ranked[:self.k_s]
        ids = {n.id for n in chosen}
        neigh = sorted({nb for n in chosen for nb in n.links} - ids, key=lambda i: (-sim[i], i))
        for i in neigh:
            if len(chosen) >= self.k:
                break
            chosen.append(self.notes[i])
            ids.add(i)
        for n in ranked:                              # fill to k (D98)
            if len(chosen) >= self.k:
                break
            if n.id not in ids:
                chosen.append(n)
                ids.add(n.id)
        return chosen[:self.k]

    def render(self, selected):
        return [self.note_line(n) for n in selected]

    def render_stats(self, selected):
        return [self.note_render(n)[1] for n in selected]

    def selected_ids(self, selected):
        return [n.id for n in selected]

    def notes_on(self, op):
        return [n.id for n in self.notes.values() if n.canonical_op == op]

    def visible(self):
        return [[n.id, n.canonical_op] for n in sorted(self.notes.values(), key=lambda n: n.id)]

    def final_state(self):
        return [self.notes[i].snapshot() for i in sorted(self.notes)]


CONDITIONS = ("C0", "C1", "C2", "C3", "F", "AN", "R", "A1", "A2", "C2-OU")


def make_memory(condition: str, embedder, tokenizer, k=3, k_s=2, theta_link=0.65) -> Memory:
    if condition in ("C0", "R"):
        return Memory(embedder, tokenizer, k)
    if condition == "C1":
        return ReflectionList(embedder, tokenizer, k)
    if condition == "F":
        return FifoReflections(embedder, tokenizer, k)
    common = dict(k=k, k_s=k_s, theta_link=theta_link)
    if condition == "C2":
        return EEMStore(embedder, tokenizer, name="C2", **common)
    if condition == "C3":
        return EEMStore(embedder, tokenizer, links=False, name="C3", **common)
    if condition == "AN":
        return EEMStore(embedder, tokenizer, all_notes=True, name="AN", **common)
    if condition == "A1":
        return EEMStore(embedder, tokenizer, mode="a1", outcome=False, name="A1", **common)
    if condition == "A2":
        return EEMStore(embedder, tokenizer, mode="a2", name="A2", **common)
    if condition == "C2-OU":
        return EEMStore(embedder, tokenizer, outcome=False, name="C2-OU", **common)
    raise ValueError(condition)
