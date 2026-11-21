"""S3.1: validation, error styles, silent drift, idempotency, business rules, get_api_docs (spec sections 1, 3-5)."""
import itertools
import json

import pytest

from eem_acc.stage3.apisuite import errors
from eem_acc.stage3.apisuite.drift import ALL_PAIRS, CATALOGUE, SILENT, effective, get_pair, legal_together
from eem_acc.stage3.apisuite.ops import OPS, OP_NAMES
from eem_acc.stage3.apisuite.schema import Field, validate_object
from eem_acc.stage3.apisuite.suite import ApiSuite
from eem_acc.stage3.apisuite.world import World

CID, SKU, SKU2, OID, OID_PAID, PID, SID, TID = "C-1001", "SKU-2001", "SKU-2002", "O-3001", "O-3002", "P-4001", \
    "S-5001", "T-6001"


def _world() -> World:
    w = World()
    w.add_setup({
        "customers": [{"customer_id": CID, "email": "budi.santoso10@mail.example", "name": "Budi Santoso",
                       "tier": "basic"}],
        "products": [{"sku": SKU, "name": "Nusantara Kopi Arabica 250 g", "price": 50000,
                      "stock": {"JKT": 10, "SBY": 1, "BDG": 0}},
                     {"sku": SKU2, "name": "Nusantara Kopi Arabica 500 g", "price": 90000,
                      "stock": {"JKT": 5, "SBY": 5, "BDG": 5}}],
        "carts": [{"customer_id": CID, "items": [{"sku": SKU, "qty": 2}]}],
        "orders": [{"order_id": OID, "customer_id": CID, "items": [{"sku": SKU, "qty": 2}], "shipping_method": "standard",
                    "total": 100000, "status": "created", "payment_id": None, "shipment_id": None},
                   {"order_id": OID_PAID, "customer_id": CID, "items": [{"sku": SKU2, "qty": 1}],
                    "shipping_method": "express", "total": 90000, "status": "paid", "payment_id": PID,
                    "shipment_id": None}],
        "payments": [{"payment_id": PID, "order_id": OID_PAID, "amount": 90000, "method": "card",
                      "refunded_amount": 0, "status": "captured"}],
        "tickets": [{"ticket_id": TID, "customer_id": CID, "subject": "Question about delivery", "category": "shipping",
                     "priority": "normal", "status": "open", "comments": []}],
    })
    return w


def _suite(pair_ids=(), style="informative", world=None) -> ApiSuite:
    spec = {"styles": {op: style for op in OP_NAMES}, "drift_schedule": [[0, list(pair_ids)]], "tasks": []}
    return ApiSuite(world or _world(), spec)


DOC_ARGS = {
    "find_customer": {"email": "budi.santoso10@mail.example"},
    "get_cart": {"customer_id": CID},
    "create_order": {"customer_id": CID, "items": [{"sku": SKU, "qty": 2}, {"sku": SKU2, "qty": 1}],
                     "shipping_method": "standard"},
    "get_order": {"order_id": OID},
    "pay_order": {"order_id": OID, "amount": 100000, "method": "card"},
    "refund_payment": {"payment_id": PID, "amount": 90000, "reason": "damaged"},
    "cancel_order": {"order_id": OID, "reason": "customer_request"},
    "search_products": {"query": "Nusantara Kopi Arabica 250 g"},
    "check_stock": {"sku": SKU, "warehouse": "JKT"},
    "reserve_stock": {"sku": SKU, "qty": 2, "warehouse": "JKT"},
    "create_shipment": {"order_id": OID_PAID, "warehouse": "JKT", "carrier": "jne"},
    "track_shipment": {"shipment_id": SID},
    "create_ticket": {"customer_id": CID, "subject": "Wrong item received", "category": "product"},
    "update_ticket": {"ticket_id": TID, "status": "pending"},
    "add_comment": {"ticket_id": TID, "text": "Escalated to the logistics team.", "public": True},
    "list_tickets": {"customer_id": CID},
}


def _with_shipment(w: World) -> World:
    w.add_setup({"shipments": [{"shipment_id": SID, "order_id": OID_PAID, "warehouse": "JKT", "carrier": "jne",
                                "insured": False, "status": "delayed"}]})
    return w


# ------------------------------------------------------------------------------------------------------------ validation
@pytest.mark.parametrize("kind,good,bad", [
    ("integer", [0, 5], [True, 1.0, "1"]),
    ("number", [1, 1.5], [True, "1.5"]),
    ("boolean", [True, False], [0, 1, "true"]),
    ("string", ["x"], [1, None]),
    ("digits", ["12"], [12, "1.5", "a"]),
    ("decimal", ["12", "12.50"], [12, "12.505", "x"]),
    ("int01", [0, 1], [True, 2, "1"]),
    ("array_str", [["a"]], ["a", [1], []]),
])
def test_strict_validation(kind, good, bad):
    sch = (Field("x", kind),)
    for v in good:
        assert validate_object(sch, {"x": v}) == [], (kind, v)
    for v in bad:
        assert validate_object(sch, {"x": v}), (kind, v)


def test_all_items_collected_extra_forbidden_and_no_input_echo():
    r = _suite().execute("pay_order", {"order_id": OID, "amount": "100000", "method": "cash", "tip": 5}, 0)
    types = [i["type"] for i in r.body["detail"]]
    assert r.status == 422 and types == ["float_type", "enum", "extra_forbidden"]
    assert all("input" not in i for i in r.body["detail"])
    assert r.truth.kind == "agent" and r.truth.reason == "documented_schema"
    assert "100000" not in json.dumps(r.body) and "cash" not in json.dumps(r.body)


def test_array_items_report_one_item_per_path_and_type():
    r = _suite().execute("create_order", {"customer_id": CID, "shipping_method": "standard",
                                          "items": [{"sku": SKU, "qty": "2"}, {"sku": SKU2, "qty": "1"}]}, 0)
    assert [i["loc"] for i in r.body["detail"]] == [["body", "items", 0, "qty"]]


# ---------------------------------------------------------------------------------------------------------- error styles
def test_route_vs_resource_404_differ_in_both_styles():
    for style in ("informative", "terse"):
        s = _suite(style=style)
        route = s.execute("no_such_op", {}, 0)
        res = s.execute("get_order", {"order_id": "O-9999"}, 0)
        assert route.status == res.status == 404 and route.body != res.body
        assert route.truth.reason == "unknown_operation" and res.truth.reason == "unknown_resource"
        assert "O-9999" not in json.dumps(res.body)                       # D73


def test_terse_bodies_carry_no_detail():
    s = _suite(["pay_order:rename"], style="terse")
    r = s.execute("pay_order", DOC_ARGS["pay_order"], 0)
    assert (r.status, r.body) == (400, {"error": "Bad Request"}) and r.truth.kind == "drift"
    r = s.execute("pay_order", {"order_id": OID, "total": 1, "method": "card"}, 0)
    assert (r.status, r.body) == (422, {"error": "Unprocessable Entity"}) and r.truth.reason == "business_rule"


def test_moved_pointer_only_in_informative_style():   # D74
    r = _suite(["create_ticket:moved"]).execute("create_ticket", DOC_ARGS["create_ticket"], 0)
    assert r.status == 410 and "open_ticket" in r.body["detail"] and r.truth.pair_ids == ("create_ticket:moved",)
    r = _suite(["create_ticket:moved"], style="terse").execute("create_ticket", DOC_ARGS["create_ticket"], 0)
    assert (r.status, r.body) == (404, errors.ROUTE_404_TERSE) and r.truth.kind == "drift"
    r = _suite(["create_ticket:moved"]).execute("open_ticket", DOC_ARGS["create_ticket"], 0)
    assert r.status == 200
    r = _suite().execute("open_ticket", DOC_ARGS["create_ticket"], 0)        # not moved in this stream
    assert r.status == 404 and r.truth.reason == "unknown_operation"


def test_business_messages_carry_no_values():
    s = _suite()
    r = s.execute("pay_order", {"order_id": OID, "amount": 99999, "method": "card"}, 0)
    assert r.status == 422 and "99999" not in json.dumps(r.body) and "100000" not in json.dumps(r.body)


# --------------------------------------------------------------------------------------------------- drift observability
def _naive_world(op):
    return _with_shipment(_world()) if op == "track_shipment" else _world()


@pytest.mark.parametrize("pid", sorted(ALL_PAIRS))
def test_every_pair_breaks_the_documented_call_and_is_tagged(pid):
    p = get_pair(pid)
    s = _suite([pid], world=_naive_world(p.op))
    r = s.execute(p.op, DOC_ARGS[p.op], 0)
    if p.type == "silent":
        assert r.status == 200 and r.body["status"] == "error" and r.truth.kind == "silent_drift"
    else:
        assert r.status in (400, 404, 410, 422) and r.truth.kind == "drift"
    assert pid in r.truth.pair_ids
    assert s.world.created == []                                                   # nothing executed


def test_without_drift_every_documented_call_succeeds():
    for op in OP_NAMES:
        r = _suite(world=_naive_world(op)).execute(op, DOC_ARGS[op], 0)
        assert r.status == 200 and r.truth.kind == "ok", (op, r.body)
        assert not (isinstance(r.body, dict) and r.body.get("status") == "error")


def test_silent_precedence_after_framework_and_no_state_change():
    s = _suite(["pay_order:silent"])
    bad = dict(DOC_ARGS["pay_order"], tip=1)                    # a framework error on another field wins
    r = s.execute("pay_order", bad, 0)
    assert r.status == 422 and [i["loc"] for i in r.body["detail"]] == [["body", "tip"]]   # silent items deferred
    assert r.truth.kind == "agent"
    r = s.execute("pay_order", DOC_ARGS["pay_order"], 0)
    assert r.status == 200 and r.truth.kind == "silent_drift"
    assert s.world.orders[OID]["status"] == "created" and not s.world.payments.keys() - {PID}


def test_silent_messages_informative_vs_terse():
    for sid in SILENT:
        op = SILENT[sid].op
        ri = _suite([sid], world=_naive_world(op)).execute(op, DOC_ARGS[op], 0)
        rt = _suite([sid], style="terse", world=_naive_world(op)).execute(op, DOC_ARGS[op], 0)
        assert ri.body["message"].startswith("Validation failed: ")
        assert rt.body == {"status": "error", "message": "Request failed"}


def _legal_sets():
    by_op = {}
    for p in CATALOGUE.values():
        by_op.setdefault(p.op, []).append(p)
    for s in SILENT.values():
        by_op[s.op].append(s)
    for op, ps in by_op.items():
        for p in ps:
            yield [p]
        for p, q in itertools.combinations(ps, 2):
            if legal_together(p, q):
                yield [p, q]


def test_drift_items_first_and_bodies_under_400_chars():
    n = 0
    for pairs in _legal_sets():
        op = pairs[0].op
        s = _suite([p.id for p in pairs], world=_naive_world(op))
        args = dict(DOC_ARGS[op])
        r = s.execute(op, args, 0)
        txt = r.body if isinstance(r.body, str) else json.dumps(r.body)
        assert len(txt) < 400, (pairs, len(txt))
        if r.status == 422:
            v = s.views(0)[op]
            flags = [(i["loc"][1] if len(i["loc"]) == 2 else f"{i['loc'][1]}[].{i['loc'][3]}") in v.drift_paths
                     for i in r.body["detail"]]
            assert flags == sorted(flags, reverse=True)
        n += 1
    assert n > 74


# ---------------------------------------------------------------------------------------------------------- idempotency
def test_same_call_key_executes_once():                       # D68
    s = _suite()
    a = s.execute("create_order", DOC_ARGS["create_order"], 0, call_key=("agent1", 7))
    b = s.execute("create_order", DOC_ARGS["create_order"], 0, call_key=("agent1", 7))
    assert a == b and len(s.world.orders) == 3 and s.executions == 1
    s.execute("create_order", DOC_ARGS["create_order"], 0, call_key=("agent1", 8))
    assert len(s.world.orders) == 4


# ------------------------------------------------------------------------------------------------------- business rules
def test_business_rules_and_duplicates():
    s = _suite()
    ex = lambda op, **a: s.execute(op, a, 0)
    assert ex("pay_order", order_id=OID_PAID, amount=90000, method="card").status == 409       # already paid
    assert ex("create_shipment", order_id=OID, warehouse="JKT", carrier="jne").status == 409  # unpaid
    assert ex("cancel_order", order_id=OID_PAID, reason="customer_request").status == 409     # paid, not refunded
    assert ex("refund_payment", payment_id=PID, amount=90001, reason="late").status == 422
    assert ex("refund_payment", payment_id=PID, amount=90000, reason="late").status == 200
    assert ex("refund_payment", payment_id=PID, amount=1, reason="late").status == 422        # nothing left
    assert ex("cancel_order", order_id=OID_PAID, reason="customer_request").status == 200
    assert ex("cancel_order", order_id=OID_PAID, reason="customer_request").status == 409     # already cancelled
    assert ex("reserve_stock", sku=SKU, qty=2, warehouse="SBY").status == 409                 # not enough
    assert ex("reserve_stock", sku=SKU, qty=0, warehouse="JKT").status == 422
    assert ex("pay_order", order_id=OID, amount=100000, method="card").status == 200
    assert ex("create_shipment", order_id=OID, warehouse="BDG", carrier="jne").status == 409  # no stock
    assert ex("create_shipment", order_id=OID, warehouse="JKT", carrier="jne").status == 200
    assert ex("create_shipment", order_id=OID, warehouse="JKT", carrier="jne").status == 409  # second shipment
    assert ex("cancel_order", order_id=OID, reason="customer_request").status == 409          # shipped
    assert ex("create_order", customer_id=CID, items=[{"sku": SKU, "qty": 0}], shipping_method="express").status == 422


def test_success_bodies_never_have_status_error():
    s = _suite(world=_with_shipment(_world()))
    for op in OP_NAMES:
        r = s.execute(op, DOC_ARGS[op], 0)
        if r.status == 200:
            assert r.body.get("status") != "error"


# ------------------------------------------------------------------------------------------------------- get_api_docs
def test_get_api_docs_current_moved_unknown():
    s = _suite(["create_ticket:moved", "pay_order:rename"], style="terse")
    r = s.execute("get_api_docs", {"operation": "pay_order"}, 0)
    assert r.status == 200 and " total " in r.body["docs"] and r.truth.kind == "ok"
    r = s.execute("get_api_docs", {"operation": "create_ticket"}, 0)
    assert r.body["operation"] == "open_ticket" and "(formerly create_ticket)" in r.body["docs"]
    r = s.execute("get_api_docs", {"operation": "open_ticket"}, 0)
    assert r.status == 200
    r = s.execute("get_api_docs", {"operation": "nope"}, 0)
    assert r.status == 404 and r.body == {"detail": "Operation not found"}
    r = s.execute("get_api_docs", {}, 0)
    assert r.status == 422


def test_canonical_storage_of_drifted_values():
    s = _suite(["create_order:type", "pay_order:type", "create_order:enum", "add_comment:type"])
    r = s.execute("create_order", {"customer_id": CID, "items": [{"sku": SKU, "qty": "2"}],
                                   "shipping_method": "regular"}, 0)
    o = s.world.orders[r.body["order_id"]]
    assert o["items"] == [{"sku": SKU, "qty": 2}] and o["shipping_method"] == "standard"
    r = s.execute("pay_order", {"order_id": o["order_id"], "amount": "100000.00", "method": "card"}, 0)
    assert r.status == 200 and s.world.payments[r.body["payment_id"]]["amount"] == 100000
    s.execute("add_comment", {"ticket_id": TID, "text": "x", "public": 1}, 0)
    assert s.world.tickets[TID]["comments"][-1]["public"] is True
