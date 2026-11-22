"""Memory for every condition (S3.2a spec sections 6.3-6.5, 7): the EEM store (C2, C3, AN, A1, A2, C2-OU), the
reflection list (C1), the FIFO buffer (F) and no memory (C0, R)."""
from __future__ import annotations

import copy
import json
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
    suite_op: str | None = None          # logs only (retrieval-recall denominator, L8)
    # D123: pending writers (agent, task) -> their trigger args and error keys; any writer's same-task success
    # closes it by diffing its own trigger (in a shared store a merge by another agent no longer drops a close)
    writers: dict = field(default_factory=dict)

    def snapshot(self) -> dict:
        return {"id": self.id, "api_name": self.api_name, "canonical_op": self.canonical_op, "suite_op": self.suite_op,
                "error_type": self.error_type, "kt": {"expected": self.kt.expected, "received": self.kt.received,
                                                     "error_keys": sorted(self.kt.error_keys)},
                "reflection": self.reflection, "text_raw": self.text_raw, "trigger_args": self.trigger_args,
                "owner": list(self.owner), "writers": sorted(list(w) for w in self.writers),
                "items": copy.deepcopy(self.items), "line": render_items(self.items),
                # the line as rendered for prompts before any cut (D116 compaction); cuts are in step note_stats
                "line_display": ("Last working change: " + "; ".join(EEMStore.display_items(self.items)))
                if self.items else None,
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
        responses); never from the suite (second opinion M2)."""
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
        if not fits(body):                       # a tokenizer that merges across the joins: degrade, never crash
            stats["render_fallback"] = True
            for parts in ((head, refl), (head,)):
                body = ". ".join(x for x in parts if x)
                if fits(body):
                    break
            while body and not fits(body):
                c = t.count(body)
                body = t.truncate(body, c - 1)
                if body and t.count(body) >= c:
                    body = body[:-1]
            stats["line_shown"] = stats["line_shown"] and line in body
            if not stats["line_shown"]:
                stats["items_shown"] = 0
        return "- " + body, stats

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
            reflection = A2_TEMPLATE.format(api=ctx.kt.api_name)          # the scrubbed name (L7)
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
            # D123: this agent's earlier tasks are over (agents run tasks in order): drop its stale writers
            n.writers = {w: v for w, v in n.writers.items() if w[0] != ctx.agent}
            kind = "merge"
            merge_extra = {"items_kept": len(kept), "kept_items": copy.deepcopy(kept)}
        else:
            n = Note(self._next, ctx.api_name, op, ctx.kt.error_type, ctx.kt, reflection,
                     copy.deepcopy(ctx.args), (ctx.agent, ctx.task),
                     text_raw=ctx.raw_text if self.mode == "a1" else None, truth=ctx.truth)
            self._next += 1
            self.notes[n.id] = n
            self.index[key] = n.id
            kind = "create"
            merge_extra = {}
        n.suite_op = ctx.suite_op                                         # logs only (L8)
        n.writers[(ctx.agent, ctx.task)] = {"trigger_args": copy.deepcopy(ctx.args),
                                            "error_keys": frozenset(ctx.kt.error_keys)}
        n.z = self.embedder.embed(self._embed_text(n))
        self._relink(n)
        if self.outcome_on and ctx.kt.error_type in DRIFT_SIGNAL:      # D114
            for o in self.notes.values():
                if o.id != n.id and self.canonical(o.api_name) == op and o.items and not o.possibly_outdated:
                    o.possibly_outdated = True
                    self._event("mark", o, **at)
        self._event(kind, n, **at, **merge_extra)
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
            if self.canonical(n.api_name) == op and n.possibly_outdated:
                n.items = [it for it in n.items if item_holds(it, ctx.args, ctx.called_id)]
                n.possibly_outdated = False
                self._event("recheck", n, **at)
        me = (ctx.agent, ctx.task)
        for n in sorted(self.notes.values(), key=lambda n: n.id):      # close notes this writer is pending on
            if me not in n.writers or self.canonical(n.api_name) != op:
                continue
            w = n.writers[me]
            wx = dict(writer=list(me), trigger_args=copy.deepcopy(w["trigger_args"]),
                      writer_error_keys=sorted(w["error_keys"]), by_last_owner=(me == n.owner))
            n.canonical_op = op                                     # aliases learned since creation (r4 review M1)
            if op not in OPS:                                       # documented schema unknown: no diff (leak-safe)
                del n.writers[me]
                n.open = bool(n.writers)
                self._event("close", n, moved_to=None, known_keys=[], new_items=[], no_schema=True, **at, **wx)
                continue
            documented = OPS[op].schema
            if n.error_type in MOVED_TYPES:
                if ctx.called_id == n.api_name:
                    del n.writers[me]                                # not a move after all: no new line
                    n.open = bool(n.writers)
                    self._event("close", n, moved_to=None, known_keys=[], **at, **wx)
                    continue
                # D115: any other id reaching here has the same canonical operation, i.e. an alias (chain) of X
                # that some agent observed (410 pointer or docs response); re-review L9
                moved_to = ctx.called_id
            elif ctx.called_id != n.api_name:
                continue
            else:
                moved_to = None
            known = sorted(self.line_keys.get(op, set()))
            new = last_working_items(w["trigger_args"], ctx.args, documented, moved_to=moved_to,
                                     error_keys=w["error_keys"], known_keys=frozenset(known))
            kept = list(n.items)                                     # D113: items kept from before a merge
            n.items = kept + [it for it in new if it not in kept]
            for it in n.items:
                if it["kind"] == "rename":
                    self.line_keys.setdefault(op, set()).add(it["new"])
                elif it["kind"] == "added":
                    self.line_keys.setdefault(op, set()).add(it["key"])
            del n.writers[me]
            n.open = bool(n.writers)
            self._event("close", n, moved_to=moved_to, known_keys=known, new_items=new, kept_items=kept, **at, **wx)

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
        return [n.id for n in self.notes.values() if n.suite_op == op]                 # logs: suite op (L8)

    def visible(self):
        return [[n.id, n.suite_op] for n in sorted(self.notes.values(), key=lambda n: n.id)]

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
