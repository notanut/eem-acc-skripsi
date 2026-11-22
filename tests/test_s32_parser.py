"""S3.2a: k_t parser over every S3.1 error shape (spec 6.1)."""
import json

import pytest

from eem_acc.stage3.agent.parser import CODE_TYPES, parse_error, moved_pointer, received_text
from eem_acc.stage3.apisuite.drift import ALL_PAIRS, effective, get_pair

from test_s31_suite import CID, DOC_ARGS, OID, OID_PAID, PID, _naive_world, _suite


@pytest.mark.parametrize("style", ["informative", "terse"])
@pytest.mark.parametrize("pid", sorted(ALL_PAIRS))
def test_every_drift_error_parses(pid, style):
    p = get_pair(pid)
    s = _suite([pid], style=style, world=_naive_world(p.op))
    r = s.execute(p.op, DOC_ARGS[p.op], 0)
    kt = parse_error(p.op, r.status, r.body, DOC_ARGS[p.op])
    e = effective(p)
    if p.type == "silent":
        assert kt.error_type == "silent_error"
        if style == "informative":
            assert kt.expected != "unknown" and kt.error_keys
        else:
            assert kt.expected == "unknown" and not kt.error_keys
    elif e.type == "moved":
        if style == "informative":
            assert kt.error_type == "operation_removed" and moved_pointer(kt) == e.new_op
        else:
            assert kt.error_type == "route_not_found" and moved_pointer(kt) is None
    elif style == "informative":
        assert kt.error_type in set(CODE_TYPES.values()) and kt.error_keys
    else:
        assert kt.error_type == "bad_request" and kt.expected == "unknown"
    assert kt.api_name == p.op


def test_compact_expected_forms():
    s = _suite(["pay_order:enum", "get_cart:new_required"])
    r = s.execute("pay_order", DOC_ARGS["pay_order"], 0)
    kt = parse_error("pay_order", r.status, r.body, DOC_ARGS["pay_order"])
    assert kt.error_type == "enum_error" and kt.expected == "method: one of CARD|TRANSFER|WALLET"
    assert 'method: "card"' in kt.received and "amount: integer" in kt.received
    r = s.execute("get_cart", DOC_ARGS["get_cart"], 0)
    kt = parse_error("get_cart", r.status, r.body, DOC_ARGS["get_cart"])
    assert kt.expected == "currency: missing" and kt.error_type == "missing_field"
    r = s.execute("get_cart", dict(DOC_ARGS["get_cart"], currency="USD"), 0)
    kt = parse_error("get_cart", r.status, r.body, dict(DOC_ARGS["get_cart"], currency="USD"))
    assert kt.error_type == "literal_error" and kt.expected == 'currency: "IDR"' and kt.error_keys == {"currency"}


def test_other_shapes():
    for style in ("informative", "terse"):
        s = _suite(style=style)
        r = s.execute("get_order", {"order_id": "O-9999"}, 0)
        kt = parse_error("get_order", r.status, r.body, {"order_id": "O-9999"})
        assert kt.error_type == "resource_not_found" and kt.expected == "order not found"
        r = s.execute("nope", {}, 0)
        assert parse_error("nope", r.status, r.body, {}).error_type == "route_not_found"
        r = s.execute("pay_order", {"order_id": OID_PAID, "amount": 90000, "method": "card"}, 0)
        kt = parse_error("pay_order", r.status, r.body, {})
        assert kt.error_type == "business_rule"
        assert (kt.expected != "unknown") == (style == "informative")


def test_received_never_carries_entity_values():
    args = {"order_id": "O-1234", "amount": 125000, "method": "card", "email": "a.b12@mail.example",
            "items": [{"sku": "SKU-1", "qty": 2}]}
    t = received_text(args, {"method", "email"})
    for v in ("O-1234", "125000", "a.b12", "SKU-1"):
        assert v not in t
    assert '"card"' in t and "items[].qty: integer" in t
    assert received_text({"method": "bitcoin"}, {"method"}) == "method: string"     # not a schema value (D99)


def test_docs_vocabulary_states_route_ids_enums_and_constants():
    from eem_acc.stage3.agent.scrub import docs_vocabulary
    text = ("open_ticket - Create a support ticket (formerly create_ticket)\n"
            "  method: string, enum: CARD | TRANSFER | EWALLET\n"
            '  currency: string, constant: "IDR"\n'
            "  note: free text with Budi and 125000\n")
    v = docs_vocabulary(text)
    assert {"open_ticket", "create_ticket", "CARD", "TRANSFER", "EWALLET", "IDR"} <= v
    assert not {"Budi", "125000", "note"} & v


def test_stated_values_only_from_enum_and_literal_errors():
    from eem_acc.stage3.agent.parser import parse_error
    from test_s31_suite import DOC_ARGS, _suite
    s = _suite(["pay_order:enum"])
    r = s.execute("pay_order", DOC_ARGS["pay_order"], 0)
    kt = parse_error("pay_order", r.status, r.body, DOC_ARGS["pay_order"])
    assert kt.error_type == "enum_error" and kt.stated_values
    assert all(isinstance(x, str) and x.isupper() for x in kt.stated_values), kt.stated_values
    s2 = _suite(["pay_order:rename"])
    r2 = s2.execute("pay_order", DOC_ARGS["pay_order"], 0)
    kt2 = parse_error("pay_order", r2.status, r2.body, DOC_ARGS["pay_order"])
    assert kt2.error_type != "enum_error" and not kt2.stated_values
