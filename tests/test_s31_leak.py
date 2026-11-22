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


def test_value_only_notes_never_give_a_line_over_all_rows():      # D76 property
    n = 0
    for pairs in _pair_sets():
        op = pairs[0].op
        v = op_view(op, pairs)
        success = to_current(v, copy.deepcopy(DOC_ARGS[op]))
        # trigger = the success with other values (same keys), incl. other enum values of the current schema
        trig = copy.deepcopy(success)
        for f in v.schema:
            if f.name in trig and not isinstance(trig[f.name], list):
                if f.kind == "enum" and len(f.values) > 1:
                    trig[f.name] = next(x for x in f.values if x != trig[f.name])
                elif isinstance(trig[f.name], bool):
                    trig[f.name] = not trig[f.name]
                elif isinstance(trig[f.name], (int, float)):
                    trig[f.name] += 1
                elif isinstance(trig[f.name], str) and f.kind not in ("const",):
                    trig[f.name] = trig[f.name] + "9"
        assert last_working_change(trig, success, OPS[op].schema) is None, (pairs, trig)
        assert last_working_change(success, success, OPS[op].schema) is None
        n += 1
    assert n > 150


def test_drift_is_learnt_from_documented_trigger_over_all_rows():
    for pairs in _pair_sets():
        op = pairs[0].op
        v = op_view(op, pairs)
        success = to_current(v, copy.deepcopy(DOC_ARGS[op]))
        moved = v.route if v.route != op else None
        line = last_working_change(DOC_ARGS[op], success, OPS[op].schema, moved_to=moved)
        assert line, pairs
        for p in pairs:
            e = effective(p)
            if e.type == "rename" and not e.prev_name:
                assert f"'{e.new_name}'" in line or f"[].{e.new_name}'" in line, (p.id, line)
            elif e.type == "new_required":
                assert f"'{e.new_field.name}' added" in line, (p.id, line)
            elif e.type == "enum":
                assert f"'{e.path}' = " in line, (p.id, line)
            elif e.type == "type":
                assert "is now" in line, (p.id, line)
            elif e.type == "moved":
                assert e.new_op in line


def test_sp_stage2_line_names_the_stale_key():                  # D81
    for op, (field, n1, n2) in SP_RENAMES.items():
        v1 = op_view(op, [sp_pairs(op)[0]])
        v2 = op_view(op, [sp_pairs(op)[1]])
        trig = to_current(v1, copy.deepcopy(DOC_ARGS[op]))
        succ = to_current(v2, copy.deepcopy(DOC_ARGS[op]))
        line = last_working_change(trig, succ, OPS[op].schema, known_keys=frozenset({n1}))
        assert line == f"Last working change: '{n1}' is now '{n2}' (string)", line
        # without an earlier line introducing n1, the dropped key is not named (D88)
        assert last_working_change(trig, succ, OPS[op].schema) == f"Last working change: '{n2}' added (string)"


def test_optional_documented_key_dropped_is_not_reported():
    v = op_view("search_products", [CATALOGUE["search_products:rename"]])
    trig = dict(DOC_ARGS["search_products"], limit=5)
    succ = to_current(v, DOC_ARGS["search_products"])
    assert last_working_change(trig, succ, OPS["search_products"].schema) == \
        "Last working change: 'query' is now 'q' (string)"


def test_d85_case_a_trigger_sent_both_keys():
    v = op_view("pay_order", [CATALOGUE["pay_order:rename"]])
    succ = to_current(v, DOC_ARGS["pay_order"])
    trig = dict(DOC_ARGS["pay_order"], total=succ["total"])
    assert last_working_change(trig, succ, OPS["pay_order"].schema) == "Last working change: 'amount' removed"


def test_d85_case_d_constant_fixed_after_named_error():
    sch = OPS["get_cart"].schema
    trig = {"customer_id": "C-1001", "currency": "USD"}
    succ = {"customer_id": "C-1001", "currency": "IDR"}
    assert last_working_change(trig, succ, sch, error_keys=frozenset({"currency"})) == \
        'Last working change: \'currency\' = "IDR"'
    assert last_working_change(trig, succ, sch) is None                       # error did not name it
    # never on a rename operation: a changed amount under 'total' is not recorded even if the error named it
    v = op_view("pay_order", [CATALOGUE["pay_order:rename"]])
    good = to_current(v, DOC_ARGS["pay_order"])
    bad = dict(good, total=99999)
    assert last_working_change(bad, good, OPS["pay_order"].schema, error_keys=frozenset({"total"})) is None


def test_case_d_never_leaks_over_all_rows():
    """Value-only notes with every key named by the error: values recorded only for new required fields."""
    for pairs in _pair_sets():
        op = pairs[0].op
        world = _naive_world(op)
        strings = _entity_strings(world)
        v = op_view(op, pairs)
        success = to_current(v, copy.deepcopy(DOC_ARGS[op]))
        trig = {k: (x + "9" if isinstance(x, str) else x + 1 if isinstance(x, (int, float)) and not isinstance(x, bool)
                    else x) for k, x in success.items()}
        line = last_working_change(trig, success, OPS[op].schema, error_keys=frozenset(success))
        if line:
            for s_ in strings:
                assert s_ not in line, (pairs, line, s_)
