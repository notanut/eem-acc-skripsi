"""S3.5 spec section 8: mechanism checks (not endpoints), from fields the loop already writes.

Ground truth comes from the drift catalogue (`apisuite.drift.get_pair`), not the scrub audit (review M5): a note's
`truth` is the error truth that created or last updated it ({"kind": "drift" | "silent_drift", "pair_ids": [...]})."""
from __future__ import annotations

import json
import re
from collections import defaultdict

from ..apisuite.drift import effective, get_pair
from ..apisuite.leakcheck import JSON_KIND, render_items
from .data import UnitData
from .secondary import active_pairs, op_pairs

DRIFT_KINDS = ("drift", "silent_drift")
EEM_CONDITIONS = ("C2", "C3", "C2-OU", "AN", "A2", "A1")
REFLECTION_CONDITIONS = ("C1", "F")
RETRIEVING = ("C1", "C2", "C3", "A1", "A2", "C2-OU", "F", "AN")


def _last(path: str | None) -> str | None:
    if not path:
        return None
    parts = [p for p in re.split(r"[.\[\]]+", path) if p]
    return parts[-1] if parts else None


def _pair(pid: str):
    p = get_pair(pid)
    return effective(p) if p.type == "silent" else p


def kind_words(new_kind: str) -> list[str]:
    """Words a note may use for a catalogue kind: the kind itself, its JSON kind, and the API errors' word for arrays
    ("list", parser.TYPE_WORD) (code review H2)."""
    words = [new_kind]
    j = JSON_KIND.get(new_kind)
    if j and j not in words:
        words.append(j)
    if j == "array":
        words.append("list")
    return words


def truth_tokens(pid: str) -> list[list[str]]:
    """What a note must name for drift pair `pid`: a list of alternatives, each a list of tokens that must all appear.
    rename / sp: the new key name; type: the field and its new kind; enum: any new value; new_required: the new
    field's name; moved: the new operation id."""
    p = _pair(pid)
    if p.type in ("rename", "sp1", "sp2") or p.new_name:
        return [[p.new_name]]
    if p.type == "type":
        # the API's errors and the notes' items state JSON kinds ("qty" -> string), the catalogue a finer kind
        # ("digits"): either names the new type
        return [[_last(p.path), k] for k in kind_words(p.new_kind)]
    if p.type == "enum":
        return [[new] for _, new in p.mapping]
    if p.type == "new_required":
        return [[p.new_field.name if p.new_field else _last(p.path)]]
    if p.type == "moved":
        return [[p.new_op]]
    raise ValueError(f"unknown pair type {p.type} ({pid})")


def _has(text: str, tok: str) -> bool:
    return re.search(r"(?<![A-Za-z0-9_])" + re.escape(str(tok)) + r"(?![A-Za-z0-9_])", text) is not None


LOOSE_GAP = 4          # D209: at most this many words between the field and the kind, in one clause
CLAUSE_END = ".;!?:\u2026"   # clause separators (with newlines); ':' too, since "field: kind" is the strict rule
NEGATION = r"(?:not|no|never|longer|[A-Za-z]+n't)"       # incl. contractions: isn't, doesn't, can't


def _loose_type(text: str, field: str, words: list[str]) -> bool:
    """Field and kind within LOOSE_GAP words of each other, in the same clause (no '.', ';' or newline between):
    'query must be a list', 'query should be of type list' (Part 1 second opinion M3)."""
    w = r"[^\s" + re.escape(CLAUSE_END) + r"]+"
    q = r"[`'\"(]?"
    f = r"(?<![A-Za-z0-9_])" + re.escape(field) + r"(?![A-Za-z0-9_])"
    k = r"(?<![A-Za-z0-9_])(?:" + "|".join(re.escape(x) for x in words) + r")(?![A-Za-z0-9_])"
    gap = r"[ \t]*((?:" + w + r"[ \t]+){0," + str(LOOSE_GAP) + r"})"
    for m in re.finditer(q + f + r"[`'\")]?" + gap + q + k, text):
        if not re.search(r"(?<![A-Za-z])" + NEGATION + r"(?![A-Za-z])", m.group(1), re.IGNORECASE):
            return True                      # "query must not be a list" is not a statement of the new kind
    return False


def names_truth(text: str, pid: str, items: list | None = None, loose: bool = False) -> bool:
    """Whether a note's text (or structured items) names drift pair `pid`. A type pair needs the field and its new kind
    together: an item {"kind": "type", key: field, type: kind}, or "field: kind" / "'field' is now kind" in the text
    (code review H2: the field and a kind word anywhere in the note are not enough)."""
    p = _pair(pid)
    if p.type == "type":
        field, words = _last(p.path), kind_words(p.new_kind)
        if any(it.get("kind") == "type" and _last(it.get("key")) == field and it.get("type") in words
               for it in (items or [])):
            return True
        pat = (r"(?<![A-Za-z0-9_])" + re.escape(field) + r"""['"]?\s*(?::|is now|->|=)\s*['"]?(?:"""
               + "|".join(re.escape(w) for w in words) + r")(?![A-Za-z0-9_])")
        if re.search(pat, text) is not None:
            return True
        return loose and _loose_type(text, field, words)
    return any(all(_has(text, t) for t in alt) for alt in truth_tokens(pid))


def note_text(note: dict) -> str:
    """What a note states: its rendered line and items, k_t's expected part and error keys, reflection and (A1) raw
    text; for C1/F the reflection text. k_t's `received` (what the agent SENT) is left out (code review H2)."""
    kt = note.get("kt") or {}
    items = note.get("items") or []
    try:
        rendered = render_items(items) or ""
    except (KeyError, ValueError, TypeError):          # a malformed item: its raw form, never a crash
        rendered = json.dumps(items, ensure_ascii=False)
    parts = [note.get("line") or "", note.get("line_display") or "", rendered,
             str(kt.get("expected") or ""),
             " ".join(map(str, kt.get("error_keys") or [])), note.get("reflection") or "", note.get("text_raw") or "",
             note.get("text") or ""]
    return "\n".join(parts)


def note_names_truth(note: dict, loose: bool = False) -> bool | None:
    """True if the note names every drift pair of its truth; None for a note not caused by drift. `loose` (D209)
    also accepts natural phrasing of a type drift."""
    tr = note.get("truth") or {}
    if tr.get("kind") not in DRIFT_KINDS or not tr.get("pair_ids"):
        return None
    text = note_text(note)
    return all(names_truth(text, pid, note.get("items"), loose) for pid in tr["pair_ids"])


def item_belongs(item: dict, pid: str) -> bool:
    """Whether one "Last working change" item states (part of) drift pair `pid`. Items carry full paths
    ("items[].quantity"), so the last segment is compared (code review H1); a new required field may come out as
    "added" or as "value" (leakcheck case D)."""
    p = _pair(pid)
    k = item.get("kind")
    if p.type in ("rename", "sp1", "sp2") or (p.new_name and p.type not in ("type", "enum")):
        return k == "rename" and _last(item.get("new")) == p.new_name
    if p.type == "type":
        return k == "type" and _last(item.get("key")) == _last(p.path) and item.get("type") in kind_words(p.new_kind)
    if p.type == "enum":
        new = {n for _, n in p.mapping}
        return k in ("value", "remap") and _last(item.get("key")) == _last(p.path) and item.get("value") in new
    if p.type == "new_required":
        name = p.new_field.name if p.new_field else _last(p.path)
        return k in ("added", "value") and _last(item.get("key")) == name
    if p.type == "moved":
        return k == "moved" and item.get("op") == p.new_op
    return False


def _units(units, condition, phases):
    """The same population rule as the secondary analyses: C0-C3 on the complete streams only (review L2)."""
    from .secondary import population
    return population(units, condition, phases)


def repeat_exposure(units: list[UnitData], conditions, phases=("s35-T", "s35-S")) -> dict:
    """Among exposures to a drifted operation after its first exposure in the stream, where the operation was called
    (a `first_call` entry exists), the share whose first call failed with the drift error."""
    out = {}
    for c in conditions:
        fail = n = 0
        for u in _units(units, c, phases):
            if not u.spec:
                continue
            seen: set[str] = set()
            for t in sorted(u.tasks, key=lambda t: t["pos"]):
                drifted = op_pairs(u.spec, t["pos"])
                for op, kind in (t.get("first_call") or {}).items():
                    if op not in drifted:
                        continue
                    if op in seen:
                        n += 1
                        fail += kind in DRIFT_KINDS
                    seen.add(op)
        out[c] = {"repeat_exposures": n, "drift_error_rate": (fail / n) if n else None}
    return out


def note_correctness(units: list[UnitData], conditions, phases=("s35-T", "s35-S")) -> dict:
    """Share of drift-caused notes that name their ground truth: before the outcome update (create/merge snapshots),
    after it (close snapshots) and at the end (final_notes). C1/F (reflections) are reported separately; their final
    state has no `truth`, so it is taken from their create events by id (code review L2)."""
    out = {}
    for c in conditions:
        acc = defaultdict(lambda: [0, 0])
        loose_acc = defaultdict(lambda: [0, 0])
        for u in _units(units, c, phases):
            created = {e.get("id"): e.get("truth") for e in u.raw.get("note_events", []) if e.get("event") == "create"}
            for e in u.raw.get("note_events", []):
                stage = {"create": "before_update", "merge": "before_update", "close": "after_update"}.get(e.get("event"))
                if stage is None:
                    continue
                v = note_names_truth(e)
                if v is not None:
                    acc[stage][0] += v
                    acc[stage][1] += 1
                    loose_acc[stage][0] += note_names_truth(e, loose=True)
                    loose_acc[stage][1] += 1
            for nt in u.raw.get("final_notes", []):
                if "truth" not in nt and nt.get("id") in created:
                    nt = {**nt, "truth": created[nt["id"]]}
                v = note_names_truth(nt)
                if v is not None:
                    acc["final"][0] += v
                    acc["final"][1] += 1
                    loose_acc["final"][0] += note_names_truth(nt, loose=True)
                    loose_acc["final"][1] += 1
        out[c] = {"kind": "reflection" if c in REFLECTION_CONDITIONS else "note",
                  **{k: {"n": v[1], "names_truth": v[0] / v[1], "names_truth_loose": loose_acc[k][0] / v[1]}
                     for k, v in sorted(acc.items())}}
    return out


def note_precision(units: list[UnitData], conditions=("C2", "C3", "AN", "A2"), phases=("s35-T", "s35-S")) -> dict:
    """Share of the items in final notes' "Last working change" lines that are correct (D204, both reported):
    - headline `precision_op`: the item belongs to any drift pair active on the note's operation at the stream's last
      task (an operation can carry two pairs, and a note's truth lists only the one that created it);
    - `precision_own_truth`: the item belongs to a pair of the note's own truth."""
    out = {}
    for c in conditions:
        own = op_level = tot = 0
        for u in _units(units, c, phases):
            last = max((t["pos"] for t in u.tasks), default=0)
            active = [pid for pid in (active_pairs(u.spec, last) if u.spec else [])]
            for nt in u.raw.get("final_notes", []):
                tr = nt.get("truth") or {}
                if tr.get("kind") not in DRIFT_KINDS:
                    continue
                on_op = [pid for pid in active if pid.partition(":")[0] == nt.get("suite_op")]
                for it in nt.get("items", []):
                    tot += 1
                    own += any(item_belongs(it, pid) for pid in tr.get("pair_ids", []))
                    op_level += any(item_belongs(it, pid) for pid in (on_op or tr.get("pair_ids", [])))
        out[c] = {"items": tot, "precision_op": (op_level / tot) if tot else None,
                  "precision_own_truth": (own / tot) if tot else None}
    return out


def c1_duplicate_crowding(units: list[UnitData], phases=("s35-T", "s35-S"), condition: str = "C1") -> dict:
    """Design section 7 (code review M1): how often duplicate reflections fill C1's top-k. Per retrieval with at
    least two notes: whether two retrieved reflections have identical text, and the share of slots taken by
    duplicates (slots beyond the first copy of each text)."""
    with_dup = n = 0
    dup_share = []
    for u in _units(units, condition, phases):
        text = {e.get("id"): e.get("text") for e in u.raw.get("note_events", []) if e.get("event") == "create"}
        for s in u.raw.get("steps", []):
            ids = s.get("retrieved") or []
            if len(ids) < 2:
                continue
            texts = [text.get(i) for i in ids]
            n += 1
            extra = len(texts) - len(set(texts))
            with_dup += extra > 0
            dup_share.append(extra / len(texts))
    return {"retrievals": n, "share_with_duplicates": (with_dup / n) if n else None,
            "mean_duplicate_slot_share": (sum(dup_share) / n) if n else None}


def retrieval_recall(units: list[UnitData], conditions=RETRIEVING, phases=("s35-T", "s35-S")) -> dict:
    """When an operation with an existing note is called (`notes_on_op` non-empty), the share of calls whose retrieved
    notes (`retrieved`, same step) include one of them; `visible` is checked as the denominator (every note on the
    operation must have been visible). F: "among the last k"; AN shows every note."""
    out = {}
    for c in conditions:
        hit = n = invisible = 0
        for u in _units(units, c, phases):
            for s in u.raw.get("steps", []):
                on = s.get("notes_on_op") or []
                if not on or "k" not in s:
                    continue
                vis = {v[0] if isinstance(v, (list, tuple)) else v for v in (s.get("visible") or [])}
                pre = [i for i in on if i in vis]          # notes that existed (were visible) when retrieval ran
                invisible += len(on) - len(pre)
                if not pre:
                    continue
                n += 1
                hit += any(i in set(s.get("retrieved") or []) for i in pre)
        out[c] = {"calls": n, "recall": (hit / n) if n else None, "notes_not_yet_visible": invisible,
                  "meaning": "among the last k" if c == "F" else ("all shown" if c == "AN" else "top-k retrieval")}
    return out
