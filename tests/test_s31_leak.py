"""S3.1: the reference "Last working change" diff never records entity values (spec section 10, D69)."""
import copy
import itertools
import json

from eem_acc.stage3.apisuite.drift import (CATALOGUE, SILENT, SP_RENAMES, effective, legal_together, op_view,
                                           sp_pairs, to_current)
from eem_acc.stage3.apisuite.leakcheck import last_working_change
from eem_acc.stage3.apisuite.ops import OPS

from test_s31_suite import DOC_ARGS, _naive_world, _suite


ENTITY_KEYS = ("customer_id", "email", "name", "sku", "order_id", "payment_id", "shipment_id", "ticket_id",
               "subject", "text", "query")


def _entity_strings(world) -> set:
    """Entity values: ids, emails, names, free text, and amounts (categorical enum values are not entities)."""
    out = set()
    for kind in ("customers", "products", "orders", "payments", "shipments", "tickets"):
        for e in getattr(world, kind).values():
            for k, v in e.items():
                if k in ENTITY_KEYS and isinstance(v, str):
                    out.add(v)
                elif isinstance(v, int) and not isinstance(v, bool) and v >= 1000:
                    out.add(str(v))
    for a in DOC_ARGS.values():
        for k, v in a.items():
            if k in ENTITY_KEYS:
                out.add(v)
            elif isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 1000:
                out.add(str(v))
    return out


def _pair_sets():
    by_op = {}
    for p in list(CATALOGUE.values()) + list(SILENT.values()):
        by_op.setdefault(p.op, []).append(p)
    for op, ps in by_op.items():
        for p in ps:
            yield [p]
        for p, q in itertools.combinations(ps, 2):
            if legal_together(p, q):
                yield [p, q]
    for op in SP_RENAMES:
        yield [sp_pairs(op)[0]]
        yield [sp_pairs(op)[1]]


def _triggers(op: str, doc_args: dict, success: dict):
    """Documented call, documented call minus each key, with an undocumented look-alike key, partially translated."""
    yield copy.deepcopy(doc_args)
    for k in doc_args:
        d = copy.deepcopy(doc_args)
        d.pop(k)
        yield d
        d = copy.deepcopy(doc_args)
        d[k + "Id" if not k.endswith("Id") else k + "X"] = d.pop(k)
        yield d
    for k in success:
        d = copy.deepcopy(doc_args)
        if k not in d:
            d[k] = success[k]
            yield d
    yield {}


def test_no_entity_value_in_any_recorded_line():
    checked = 0
    for pairs in _pair_sets():
        op = pairs[0].op
        world = _naive_world(op)
        strings = _entity_strings(world)
        v = op_view(op, pairs)
        success = to_current(v, copy.deepcopy(DOC_ARGS[op]))
        moved = v.route if v.route != op else None
        for trig in _triggers(op, DOC_ARGS[op], success):
            line = last_working_change(trig, success, OPS[op].schema, moved_to=moved)
            if line:
                for s in strings:
                    assert s not in line, (pairs, trig, line, s)
            checked += 1
    assert checked > 500


def test_line_content_examples():
    v = op_view("pay_order", [CATALOGUE["pay_order:rename"]])
    succ = to_current(v, DOC_ARGS["pay_order"])
    line = last_working_change(DOC_ARGS["pay_order"], succ, OPS["pay_order"].schema)
    assert line == "Last working change: 'amount' is now 'total' (integer)"
    v = op_view("get_cart", [CATALOGUE["get_cart:new_required"]])
    succ = to_current(v, DOC_ARGS["get_cart"])
    assert last_working_change(DOC_ARGS["get_cart"], succ, OPS["get_cart"].schema) == \
        "Last working change: 'currency' added (string) = \"IDR\""
    v = op_view("pay_order", [CATALOGUE["pay_order:enum"]])
    succ = to_current(v, DOC_ARGS["pay_order"])
    assert last_working_change(DOC_ARGS["pay_order"], succ, OPS["pay_order"].schema) == \
        "Last working change: 'method' = \"CARD\""
    v = op_view("pay_order", [CATALOGUE["pay_order:type"]])
    succ = to_current(v, DOC_ARGS["pay_order"])
    assert last_working_change(DOC_ARGS["pay_order"], succ, OPS["pay_order"].schema) == \
        "Last working change: 'amount' is now string"
    v = op_view("create_ticket", [CATALOGUE["create_ticket:moved"]])
    succ = to_current(v, DOC_ARGS["create_ticket"])
    assert last_working_change(DOC_ARGS["create_ticket"], succ, OPS["create_ticket"].schema,
                               moved_to="open_ticket") == "Last working change: use operation 'open_ticket'"
    # an identifier rename never records the id
    v = op_view("get_order", [CATALOGUE["get_order:rename"]])
    succ = to_current(v, DOC_ARGS["get_order"])
    line = last_working_change({"orderId": "O-3001"}, succ, OPS["get_order"].schema)
    assert "O-3001" not in line and "purchase_id" in line


def test_no_change_gives_no_line():
    assert last_working_change(DOC_ARGS["pay_order"], DOC_ARGS["pay_order"], OPS["pay_order"].schema) is None


def test_trigger_relative_content():                  # D76
    sch = OPS["pay_order"].schema
    v = op_view("pay_order", [CATALOGUE["pay_order:rename"], CATALOGUE["pay_order:enum"]])
    good = to_current(v, DOC_ARGS["pay_order"])
    # a business-rule note: the trigger already used the current keys, only a value was wrong -> no line
    assert last_working_change(dict(good, total=99999), good, sch) is None
    # identical arguments -> no line
    assert last_working_change(good, good, sch) is None
    # a resource-404 note on a renamed identifier: same keys, other id -> no line
    v2 = op_view("get_order", [CATALOGUE["get_order:rename"]])
    assert last_working_change({"purchase_id": "O-9"}, {"purchase_id": "O-1"}, OPS["get_order"].schema) is None
    # the drift itself is still learnt from a documented-form trigger
    line = last_working_change(DOC_ARGS["pay_order"], good, sch)
    assert "'amount' is now 'total'" in line and '"CARD"' in line
    # new required field already present in the trigger -> no line
    v3 = op_view("pay_order", [CATALOGUE["pay_order:new_required"]])
    s3 = to_current(v3, DOC_ARGS["pay_order"])
    assert last_working_change(s3, s3, sch) is None
