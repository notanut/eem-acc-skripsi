"""Drift catalogue, current schemas, canonicalisation, pools and stream drift sets (S3.1 spec section 6).

A *pair* is one concrete change of one operation. Types: rename, type, new_required, enum, moved, silent (a type 1-4
change of the same operation delivered as HTTP 200 with an error payload, D71), plus the staleness-probe renames
sp1/sp2 (not in the pools). Paths: "amount" or "items[].qty".
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from .ops import OPS, OP_NAMES
from .schema import Field

TYPES = ("rename", "type", "new_required", "enum", "moved", "silent")
TYPE_ERRORS = frozenset({"string_type", "int_type", "float_type", "bool_type", "list_type", "string_pattern_mismatch",
                         "literal_error", "too_short", "finite_number", "string_too_short"})
BASE_TYPES = ("rename", "type", "new_required", "enum")       # usable as silent underlying changes
POOL_SEED = 31_2026_11_21                                     # fixed: the pools are part of the frozen suite


@dataclass(frozen=True)
class Pair:
    id: str
    op: str
    type: str
    path: str | None = None            # rename/type/enum: documented path; new_required: new field name
    new_name: str | None = None        # rename / sp: new key name (last path segment)
    new_kind: str | None = None        # type
    mapping: tuple = ()                # enum: ((old, new), ...)
    new_field: Field | None = None     # new_required
    new_op: str | None = None          # moved
    underlying: str | None = None      # silent: id of the base pair it reuses
    prev_name: str | None = None       # sp2: the stage-1 name (also attributed to drift)

    def as_json(self) -> dict:
        d = {"id": self.id, "op": self.op, "type": self.type}
        for k in ("path", "new_name", "new_kind", "new_op", "underlying", "prev_name"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        if self.mapping:
            d["mapping"] = [list(m) for m in self.mapping]
        if self.new_field is not None:
            f = self.new_field
            d["new_field"] = {"name": f.name, "kind": f.kind, "values": list(f.values), "const": f.const}
        return d


def _p(op, typ, **kw) -> Pair:
    return Pair(id=f"{op}:{typ}", op=op, type=typ, **kw)


def _const(name, value) -> Field:
    return Field(name, "const", const=value)


# One concrete change per legal (operation, type) pair (spec section 6 table).
_ROWS = [
    # find_customer
    _p("find_customer", "rename", path="email", new_name="email_address"),
    _p("find_customer", "new_required", path="include_inactive", new_field=Field("include_inactive", "boolean")),
    _p("find_customer", "moved", new_op="lookup_customer"),
    # get_cart
    _p("get_cart", "rename", path="customer_id", new_name="client_id"),
    _p("get_cart", "new_required", path="currency", new_field=_const("currency", "IDR")),
    _p("get_cart", "moved", new_op="get_basket"),
    # create_order
    _p("create_order", "rename", path="items[].qty", new_name="quantity"),
    _p("create_order", "type", path="items[].qty", new_kind="digits"),
    _p("create_order", "new_required", path="currency", new_field=_const("currency", "IDR")),
    _p("create_order", "enum", path="shipping_method", mapping=(("standard", "regular"), ("express", "priority"))),
    _p("create_order", "moved", new_op="place_order"),
    # get_order
    _p("get_order", "rename", path="order_id", new_name="purchase_id"),
    _p("get_order", "new_required", path="api_version", new_field=_const("api_version", "2")),
    _p("get_order", "moved", new_op="fetch_order"),
    # pay_order
    _p("pay_order", "rename", path="amount", new_name="total"),
    _p("pay_order", "type", path="amount", new_kind="decimal"),
    _p("pay_order", "new_required", path="capture", new_field=Field("capture", "boolean")),
    _p("pay_order", "enum", path="method",
       mapping=(("card", "CARD"), ("transfer", "TRANSFER"), ("wallet", "WALLET"))),
    _p("pay_order", "moved", new_op="submit_payment"),
    # refund_payment
    _p("refund_payment", "rename", path="reason", new_name="refund_reason"),
    _p("refund_payment", "new_required", path="notify_customer", new_field=Field("notify_customer", "boolean")),
    _p("refund_payment", "enum", path="reason",
       mapping=(("damaged", "item_damaged"), ("late", "late_delivery"), ("other", "other_reason"))),
    _p("refund_payment", "moved", new_op="create_refund"),
    # cancel_order
    _p("cancel_order", "rename", path="reason", new_name="cancel_reason"),
    _p("cancel_order", "new_required", path="confirm", new_field=_const("confirm", True)),
    _p("cancel_order", "enum", path="reason",
       mapping=(("customer_request", "requested_by_customer"), ("out_of_stock", "no_stock"))),
    _p("cancel_order", "moved", new_op="void_order"),
    # search_products
    _p("search_products", "rename", path="query", new_name="q"),
    _p("search_products", "type", path="query", new_kind="array_str"),
    _p("search_products", "moved", new_op="find_products"),
    # check_stock
    _p("check_stock", "rename", path="warehouse", new_name="location"),
    _p("check_stock", "enum", path="warehouse",
       mapping=(("JKT", "jakarta"), ("SBY", "surabaya"), ("BDG", "bandung"))),
    _p("check_stock", "moved", new_op="get_inventory"),
    # reserve_stock
    _p("reserve_stock", "rename", path="qty", new_name="quantity"),
    _p("reserve_stock", "type", path="qty", new_kind="digits"),
    _p("reserve_stock", "new_required", path="hold_minutes", new_field=_const("hold_minutes", 30)),
    _p("reserve_stock", "enum", path="warehouse",
       mapping=(("JKT", "jakarta"), ("SBY", "surabaya"), ("BDG", "bandung"))),
    _p("reserve_stock", "moved", new_op="hold_stock"),
    # create_shipment
    _p("create_shipment", "rename", path="carrier", new_name="courier"),
    _p("create_shipment", "new_required", path="signature_required",
       new_field=Field("signature_required", "boolean")),
    _p("create_shipment", "enum", path="carrier",
       mapping=(("jne", "JNE_REG"), ("sicepat", "SICEPAT_REG"), ("pos", "POS_KILAT"))),
    _p("create_shipment", "moved", new_op="ship_order"),
    # track_shipment
    _p("track_shipment", "rename", path="shipment_id", new_name="tracking_id"),
    _p("track_shipment", "new_required", path="include_history", new_field=Field("include_history", "boolean")),
    _p("track_shipment", "moved", new_op="get_shipment_status"),
    # create_ticket
    _p("create_ticket", "rename", path="subject", new_name="title"),
    _p("create_ticket", "new_required", path="channel",
       new_field=Field("channel", "enum", values=("email", "chat", "phone"))),
    _p("create_ticket", "enum", path="category",
       mapping=(("billing", "payments"), ("shipping", "delivery"), ("product", "item"))),
    _p("create_ticket", "moved", new_op="open_ticket"),
    # update_ticket
    _p("update_ticket", "rename", path="status", new_name="state"),
    _p("update_ticket", "enum", path="status",
       mapping=(("open", "OPEN"), ("pending", "ON_HOLD"), ("closed", "RESOLVED"))),
    _p("update_ticket", "moved", new_op="set_ticket_status"),
    # add_comment
    _p("add_comment", "rename", path="text", new_name="body"),
    _p("add_comment", "type", path="public", new_kind="int01"),
    _p("add_comment", "moved", new_op="post_comment"),
    # list_tickets
    _p("list_tickets", "rename", path="customer_id", new_name="owner_id"),
    _p("list_tickets", "new_required", path="page_size", new_field=_const("page_size", 50)),
    _p("list_tickets", "moved", new_op="search_tickets"),
]

CATALOGUE: dict[str, Pair] = {p.id: p for p in _ROWS}

# Staleness-probe renames (enum-valued fields only, D69): op -> (field, stage-1 name, stage-2 name)
# Stage-1 names differ from every catalogue rename (L3 of the code review), so SP test streams never reuse a dev change.
SP_RENAMES = {
    "check_stock": ("warehouse", "depot", "site"),
    "create_shipment": ("carrier", "shipping_partner", "shipper"),
    "refund_payment": ("reason", "refund_category", "refund_cause"),
    "update_ticket": ("status", "ticket_status", "ticket_state"),
    "cancel_order": ("reason", "cancellation_reason", "cancel_cause"),
}


def sp_pairs(op: str) -> tuple[Pair, Pair]:
    f, n1, n2 = SP_RENAMES[op]
    return (Pair(f"{op}:sp1", op, "rename", path=f, new_name=n1),
            Pair(f"{op}:sp2", op, "rename", path=f, new_name=n2, prev_name=n1))


def _silent_pairs() -> dict[str, Pair]:
    rng = random.Random(POOL_SEED + 1)
    out = {}
    for op in OP_NAMES:
        base = [p for p in _ROWS if p.op == op and p.type in BASE_TYPES]
        u = rng.choice(base)
        out[f"{op}:silent"] = Pair(f"{op}:silent", op, "silent", underlying=u.id)
    return out


SILENT: dict[str, Pair] = _silent_pairs()
ALL_PAIRS: dict[str, Pair] = {**CATALOGUE, **SILENT}


def get_pair(pid: str) -> Pair:
    if pid in ALL_PAIRS:
        return ALL_PAIRS[pid]
    op, _, kind = pid.partition(":")
    if kind in ("sp1", "sp2") and op in SP_RENAMES:
        return sp_pairs(op)[0 if kind == "sp1" else 1]
    raise KeyError(pid)


def effective(p: Pair) -> Pair:
    """The schema change a pair makes (a silent pair makes its underlying change)."""
    return CATALOGUE[p.underlying] if p.type == "silent" else p


def fields_touched(p: Pair) -> set[str]:
    """Documented-path-level fields a pair touches (for the two-pair same-field rule)."""
    e = effective(p)
    if e.type == "moved":
        return set()
    return {e.path}


def legal_together(p: Pair, q: Pair) -> bool:
    """Two pairs on one operation (spec section 6, D69)."""
    if p.op != q.op or p.id == q.id or p.type == q.type:
        return False
    if fields_touched(p) & fields_touched(q):
        return False
    tp, tq = effective(p).type, effective(q).type
    if {tp, tq} == {"rename", "new_required"}:
        return False
    if p.type == "silent" and p.underlying == q.id or q.type == "silent" and q.underlying == p.id:
        return False
    return True


# ------------------------------------------------------------------------------------------------- current schema per op
def _rename_in(schema: tuple, path: str, new_name: str) -> tuple:
    if "[]." in path:
        parent, child = path.split("[].")
        return tuple(f.with_(items=tuple(g.with_(name=new_name) if g.name == child else g for g in f.items))
                     if f.name == parent else f for f in schema)
    return tuple(f.with_(name=new_name) if f.name == path else f for f in schema)


def _modify(schema: tuple, path: str, **kw) -> tuple:
    if "[]." in path:
        parent, child = path.split("[].")
        return tuple(f.with_(items=tuple(g.with_(**kw) if g.name == child else g for g in f.items))
                     if f.name == parent else f for f in schema)
    return tuple(f.with_(**kw) if f.name == path else f for f in schema)


def _current_path(path: str, renames: dict[str, str]) -> str:
    """Documented path -> current path given {documented path: new last segment}."""
    if path in renames:
        if "[]." in path:
            return path.split("[].")[0] + "[]." + renames[path]
        return renames[path]
    return path


@dataclass(frozen=True)
class OpView:
    """How one operation looks under the pairs in force."""
    op: str
    route: str                         # current operation id
    schema: tuple                      # current schema
    pairs: tuple                       # the Pair objects in force on this op
    drift_paths: dict                  # current-or-old path -> pair id (paths whose items are drift)
    drift_types: dict                  # path -> error item types that count as drift there (D80)
    silent_paths: frozenset            # paths checked at the app level (silent drift)
    renames: dict                      # documented path -> current path
    kinds: dict                        # documented path -> new kind (type drift)
    enum_maps: dict                    # documented path -> {old: new}
    new_fields: tuple                  # names of new required fields

    def is_drift(self, path: str, typ: str) -> bool:
        """An error item counts as drift only on a drifted path and of the matching kind (D80)."""
        return path in self.drift_paths and typ in self.drift_types[path]

    def is_silent(self, path: str, typ: str) -> bool:
        return path in self.silent_paths and self.is_drift(path, typ)


def op_view(op: str, pairs: list[Pair]) -> OpView:
    schema = OPS[op].schema
    route = op
    drift_paths: dict[str, str] = {}
    drift_types: dict[str, frozenset] = {}
    silent: set[str] = set()
    renames_seg: dict[str, str] = {}
    kinds: dict[str, str] = {}
    enum_maps: dict[str, dict] = {}
    new_fields: list[str] = []
    order = {"moved": 0, "type": 1, "enum": 2, "new_required": 3, "rename": 4}
    mine = [p for p in pairs if p.op == op]
    for p in sorted(mine, key=lambda p: (order[effective(p).type], p.id)):
        e = effective(p)
        paths: dict[str, frozenset] = {}
        if e.type == "moved":
            route = e.new_op
        elif e.type == "type":
            schema = _modify(schema, e.path, kind=e.new_kind)
            kinds[e.path] = e.new_kind
            paths = {e.path: TYPE_ERRORS}
        elif e.type == "enum":
            m = dict(e.mapping)
            schema = _modify(schema, e.path, values=tuple(m[v] for v in _field_at(schema, e.path).values))
            enum_maps[e.path] = m
            paths = {e.path: frozenset({"enum"})}
        elif e.type == "new_required":
            schema = schema + (e.new_field,)
            new_fields.append(e.new_field.name)
            paths = {e.new_field.name: frozenset({"missing"}) | TYPE_ERRORS | {"enum"}}
        elif e.type == "rename":
            schema = _rename_in(schema, e.path, e.new_name)
            renames_seg[e.path] = e.new_name
            newp = _current_path(e.path, renames_seg)
            paths = {e.path: frozenset({"extra_forbidden"}), newp: frozenset({"missing"})}
            if e.prev_name:
                paths[_current_path(e.path, {e.path: e.prev_name})] = frozenset({"extra_forbidden"})
        for q, types in paths.items():
            drift_paths[q] = p.id
            drift_types[q] = types
        if p.type == "silent":
            silent |= set(paths)
    # a type/enum change on a renamed path is attributed under the current name too (not used by legal pairs)
    renames = {d: _current_path(d, renames_seg) for d in renames_seg}
    return OpView(op, route, schema, tuple(mine), drift_paths, drift_types, frozenset(silent), renames, kinds,
                  enum_maps, tuple(new_fields))


def _field_at(schema: tuple, path: str) -> Field:
    if "[]." in path:
        parent, child = path.split("[].")
        f = next(f for f in schema if f.name == parent)
        return next(g for g in f.items if g.name == child)
    return next(f for f in schema if f.name == path)


# ------------------------------------------------------------------------------------------------------ canonicalisation
def _to_number(s: str) -> int | float:
    d = Decimal(s)
    return int(d) if d == d.to_integral_value() else float(d)


def _canon_value(kind: str | None, emap: dict | None, v: Any) -> Any:
    if kind == "digits":
        v = int(v)
    elif kind == "decimal":
        v = _to_number(v)
    elif kind == "int01":
        v = bool(v)
    if emap:
        inv = {new: old for old, new in emap.items()}
        v = inv[v]
    return v


def canonicalize(view: OpView, args: dict) -> dict:
    """Current-schema arguments (already validated) -> documented names, types and enum values."""
    inv_top = {cur: doc for doc, cur in view.renames.items() if "[]." not in doc}
    inv_item = {}
    for doc, cur in view.renames.items():
        if "[]." in doc:
            parent, child = doc.split("[].")
            inv_item[(parent, cur.split("[].")[1])] = child
    out: dict[str, Any] = {}
    for k, v in args.items():
        if k in view.new_fields:
            continue
        doc = inv_top.get(k, k)
        if isinstance(v, list) and v and isinstance(v[0], dict):
            items = []
            for it in v:
                ni = {}
                for ik, iv in it.items():
                    dk = inv_item.get((doc, ik), ik)
                    path = f"{doc}[].{dk}"
                    ni[dk] = _canon_value(view.kinds.get(path), view.enum_maps.get(path), iv)
                items.append(ni)
            out[doc] = items
        else:
            out[doc] = _canon_value(view.kinds.get(doc), view.enum_maps.get(doc), v)
    return out


def to_current(view: OpView, doc_args: dict, new_values: dict | None = None) -> dict:
    """Documented-form arguments -> current-schema arguments (the oracle's translation). New required fields get
    their constant, the first enum value, or True (or `new_values[name]`)."""
    out: dict[str, Any] = {}

    def conv(path: str, v: Any) -> Any:
        k = view.kinds.get(path)
        m = view.enum_maps.get(path)
        if m:
            v = m[v]
        if k == "digits":
            v = str(int(v))
        elif k == "decimal":
            v = f"{Decimal(str(v)):.2f}"
        elif k == "int01":
            v = 1 if v else 0
        elif k == "array_str":
            v = [v]
        return v

    for k, v in doc_args.items():
        cur = view.renames.get(k, k)
        if isinstance(v, list) and v and isinstance(v[0], dict):
            items = []
            for it in v:
                ni = {}
                for ik, iv in it.items():
                    path = f"{k}[].{ik}"
                    ck = view.renames.get(path, path).split("[].")[1]
                    ni[ck] = conv(path, iv)
                items.append(ni)
            out[cur] = items
        else:
            out[cur] = conv(k, v)
    for f in view.schema:
        if f.name in view.new_fields and f.name not in out:
            if new_values and f.name in new_values:
                out[f.name] = new_values[f.name]
            elif f.kind == "const":
                out[f.name] = f.const
            elif f.kind == "enum":
                out[f.name] = f.values[0]
            else:
                out[f.name] = True
    return out


# ------------------------------------------------------------------------------------------------------------------ pools
def build_pools(seed: int = POOL_SEED) -> dict[str, list[str]]:
    """Split the 58 base pairs and 16 silent pairs into disjoint dev/test pools (D86): exactly 8 renames and 8 moved
    per pool (each op's rename and moved go to different pools, so every op is in both), new required 6/6, the odd
    type pair to test (3/2) and the odd enum pair to dev (5/4); silent pairs follow their underlying change."""
    rng = random.Random(seed)
    dev: list[str] = []
    test: list[str] = []
    ops = list(OP_NAMES)
    rng.shuffle(ops)
    rename_dev = set(ops[:8])
    for op in OP_NAMES:
        a, b = f"{op}:rename", f"{op}:moved"
        if op in rename_dev:
            dev.append(a); test.append(b)
        else:
            dev.append(b); test.append(a)
    for typ, odd_to in (("type", "test"), ("new_required", None), ("enum", "dev")):
        ids = sorted(p.id for p in _ROWS if p.type == typ)
        rng.shuffle(ids)
        n_test = len(ids) // 2 + (1 if odd_to == "test" and len(ids) % 2 else 0)
        test += ids[:n_test]
        dev += ids[n_test:]
    for sid, s in sorted(SILENT.items()):
        (dev if s.underlying in dev else test).append(sid)
    return {"dev": sorted(dev), "test": sorted(test)}


POOLS = build_pools()


def _pick(rng: random.Random, cands: list[Pair], usage: dict | None) -> Pair:
    """Uniform choice, or (joint drawing, D86) uniform among the least-used candidates."""
    if usage:
        low = min(usage.get(c.id, 0) for c in cands)
        cands = [c for c in cands if usage.get(c.id, 0) == low]
    return rng.choice(cands)


def draw_drift_set(rng: random.Random, pool: list[str], n_ops: int = 8, n_double: int = 2,
                   exclude_ops: frozenset = frozenset(), max_tries: int = 500,
                   usage: dict | None = None) -> list[Pair]:
    """Stratified drift set (D70): each type at least once where the pool allows, n_ops operations, n_double of them
    with two legal pairs. Seeded constructive search; raises if no set is found. With `usage` (pair id -> times used
    by earlier streams of the same plan), every choice prefers the least-used pairs (joint drawing, D86)."""
    pairs = [get_pair(i) for i in pool if get_pair(i).op not in exclude_ops]
    by_type = {t: [p for p in pairs if p.type == t] for t in TYPES}
    for _ in range(max_tries):
        chosen: list[Pair] = []
        used_ops: set[str] = set()
        types = [t for t in TYPES if by_type[t]]
        rng.shuffle(types)
        ok = True
        for t in types:
            cands = [p for p in by_type[t] if p.op not in used_ops]
            if not cands:
                ok = False
                break
            p = _pick(rng, cands, usage)
            chosen.append(p)
            used_ops.add(p.op)
        if not ok or len(used_ops) > n_ops:
            continue
        while len(used_ops) < n_ops:
            cands = [p for p in pairs if p.op not in used_ops]
            if not cands:
                ok = False
                break
            p = _pick(rng, cands, usage)
            chosen.append(p)
            used_ops.add(p.op)
        if not ok:
            continue
        doubles = 0
        ops_order = sorted(used_ops)
        rng.shuffle(ops_order)
        for op in ops_order:
            if doubles == n_double:
                break
            first = next(c for c in chosen if c.op == op)
            cands = [p for p in pairs if p.op == op and legal_together(first, p)]
            if cands:
                chosen.append(_pick(rng, cands, usage))
                doubles += 1
        if doubles == n_double:
            return sorted(chosen, key=lambda p: p.id)
    raise RuntimeError("no legal drift set found")
