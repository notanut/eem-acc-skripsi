"""S3.2a: store, retrieval, outcome update, possibly-outdated re-check (spec 6.3-6.5, 7)."""
import copy
import json

import pytest

from eem_acc.stage3.agent.backend import BagOfWordsEmbedder, StubTokenizer, cosine
from eem_acc.stage3.agent.memory import (EEMStore, ErrorCtx, FifoReflections, ReflectionList, SuccessCtx,
                                         make_memory)
from eem_acc.stage3.agent.parser import parse_error
from eem_acc.stage3.apisuite.drift import CATALOGUE, op_view, sp_pairs, to_current
from eem_acc.stage3.apisuite.leakcheck import last_working_change
from eem_acc.stage3.apisuite.ops import OPS

from test_s31_suite import DOC_ARGS, OID, _naive_world, _suite

TOK = StubTokenizer()


def _err(mem, s, op, args, agent=0, task=0, step=0, reflection="Use the new field name.", called=None):
    called = called or op
    r = s.execute(called, args, task)
    assert r.status != 200 or r.body.get("status") == "error"
    kt = parse_error(called, r.status, r.body, args)
    ctx = ErrorCtx(agent, task, step, called, r.op or called, copy.deepcopy(args), r.status, r.body,
                   f"{r.status} {r.body}"[:400], kt, r.truth.as_json())
    return mem.on_error(ctx, reflection), r


def _ok(mem, s, called, args, agent=0, task=0, step=1):
    r = s.execute(called, args, task)
    assert r.status == 200 and r.body.get("status") != "error", r.body
    mem.on_success(SuccessCtx(agent, task, step, called, r.op, args, r.body))
    return r


def _ok_noexec(mem, called, args, agent=0, task=0, step=1, canonical=None):
    """Feed a success without executing it again (for repeated successes on state-changing operations)."""
    mem.on_success(SuccessCtx(agent, task, step, called, canonical or called, args, {}))


def _store(**kw):
    return EEMStore(BagOfWordsEmbedder(), TOK, **kw)


def test_merge_same_key_and_reopen():
    s = _suite(["pay_order:rename"])
    m = _store()
    _err(m, s, "pay_order", DOC_ARGS["pay_order"])
    v = op_view("pay_order", [CATALOGUE["pay_order:rename"]])
    _ok(m, s, "pay_order", to_current(v, DOC_ARGS["pay_order"]))
    n = m.notes[1]
    assert not n.open and n.items
    _err(m, s, "pay_order", DOC_ARGS["pay_order"], task=1, reflection="Again.")
    # D113: the merge keeps the old items as a possibly-outdated part
    assert len(m.notes) == 1 and n.open and n.items and n.possibly_outdated and n.reflection == "Again."
    assert n.owner == (0, 1) and m.events[-1]["items_kept"] == 1
    assert [e["event"] for e in m.events] == ["create", "close", "merge"]


def test_line_equals_reference_and_owner_rule():
    s = _suite(["pay_order:rename"])
    m = _store()
    _err(m, s, "pay_order", DOC_ARGS["pay_order"], task=0)
    good = to_current(op_view("pay_order", [CATALOGUE["pay_order:rename"]]), DOC_ARGS["pay_order"])
    _ok_noexec(m, "pay_order", good, task=1)                     # another task: does not close
    assert m.notes[1].open
    _ok_noexec(m, "pay_order", good, agent=1, task=0)            # another agent: does not close
    assert m.notes[1].open
    m2 = _store()
    s2 = _suite(["pay_order:rename"])
    _err(m2, s2, "pay_order", DOC_ARGS["pay_order"])
    _ok(m2, s2, "pay_order", good)
    ref = last_working_change(DOC_ARGS["pay_order"], good, OPS["pay_order"].schema)
    assert m2.final_state()[0]["line"] == ref == "Last working change: 'amount' is now 'total' (integer)"


def test_moved_closed_via_pointer_and_via_docs():
    for style in ("informative", "terse"):
        s = _suite(["create_ticket:moved"], style=style)
        m = _store()
        _err(m, s, "create_ticket", DOC_ARGS["create_ticket"])
        if style == "terse":                                     # no pointer: needs get_api_docs(X) in the same task
            _ok(m, s, "open_ticket", DOC_ARGS["create_ticket"])
            assert m.notes[1].open
            _ok(m, s, "get_api_docs", {"operation": "create_ticket"})
        _ok(m, s, "open_ticket", DOC_ARGS["create_ticket"])
        n = m.final_state()[0]
        assert not n["open"] and n["line"] == "Last working change: use operation 'open_ticket'", (style, n)


def test_moved_plus_second_drift_uses_canonical_schema():
    s = _suite(["create_ticket:moved", "create_ticket:rename"])
    m = _store()
    _err(m, s, "create_ticket", DOC_ARGS["create_ticket"])
    _err(m, s, "open_ticket", DOC_ARGS["create_ticket"])
    v = s.views(0)["create_ticket"]
    _ok(m, s, "open_ticket", to_current(v, DOC_ARGS["create_ticket"]))
    lines = {n["api_name"]: n["line"] for n in m.final_state()}
    assert lines["create_ticket"].startswith("Last working change: use operation 'open_ticket'")
    assert "'subject' is now 'title'" in lines["open_ticket"]


def test_possibly_outdated_mark_and_itemwise_recheck():
    s = _suite(["pay_order:rename", "pay_order:enum"])
    m = _store()
    _err(m, s, "pay_order", DOC_ARGS["pay_order"])
    v = s.views(0)["pay_order"]
    good = to_current(v, DOC_ARGS["pay_order"])
    _ok(m, s, "pay_order", good)
    n1 = m.notes[1]
    assert n1.items
    # a wrong id (resource 404) cannot signal drift: no mark (D114)
    _err(m, s, "pay_order", dict(good, order_id="O-9999"), task=1)
    assert not n1.possibly_outdated
    # a validation error of another kind on the same operation: the line is kept but marked
    _err(m, s, "pay_order", dict(good, tip=1), task=1)
    assert n1.possibly_outdated and n1.items
    assert "(possibly outdated)" in m.note_line(n1)
    # next success anywhere re-checks item by item
    _ok_noexec(m, "pay_order", good, agent=1, task=5)
    assert not n1.possibly_outdated and n1.items


def test_recheck_drops_items_that_no_longer_hold():
    s = _suite(["check_stock:sp1"])
    m = _store()
    _err(m, s, "check_stock", DOC_ARGS["check_stock"])
    v1 = s.views(0)["check_stock"]
    _ok(m, s, "check_stock", to_current(v1, DOC_ARGS["check_stock"]))
    n1 = m.notes[1]
    assert n1.items[0]["new"] == "depot"
    n1.possibly_outdated = True
    v2 = op_view("check_stock", [sp_pairs("check_stock")[1]])
    s2 = _suite([sp_pairs("check_stock")[1].id])
    m.on_success(SuccessCtx(0, 3, 0, "check_stock", "check_stock", to_current(v2, DOC_ARGS["check_stock"]), {}))
    assert n1.items == [] and m.note_line(n1).count("Last working change") == 0


def test_known_keys_history_names_stale_sp_key():
    p1, p2 = sp_pairs("check_stock")
    m = _store()
    s1 = _suite([p1.id])
    _err(m, s1, "check_stock", DOC_ARGS["check_stock"], task=0)
    _ok(m, s1, "check_stock", to_current(s1.views(0)["check_stock"], DOC_ARGS["check_stock"]), task=0)
    s2 = _suite([p2.id], world=s1.world)
    stale = to_current(s1.views(0)["check_stock"], DOC_ARGS["check_stock"])
    _err(m, s2, "check_stock", stale, task=1)                    # merge removes the stage-1 line first
    _ok(m, s2, "check_stock", to_current(s2.views(0)["check_stock"], DOC_ARGS["check_stock"]), task=1)
    assert m.final_state()[0]["line"] == "Last working change: 'depot' is now 'site' (string)"


def test_links_symmetric_threshold_and_retrieval_order():
    m = _store(theta_link=0.1, k=3, k_s=1)
    s = _suite(["pay_order:rename", "get_cart:new_required", "create_order:enum", "check_stock:enum"])
    for op in ("pay_order", "get_cart", "create_order", "check_stock"):
        _err(m, s, op, DOC_ARGS[op], reflection=f"Fix {op} call parameters now.")
    for n in m.notes.values():
        for o in n.links:
            assert n.id in m.notes[o].links and cosine(n.z, m.notes[o].z) > 0.1
    sel = m.retrieve("pay_order amount total")
    assert len(sel) == 3 and sel[0].api_name == "pay_order"
    flat = _store(links=False, k=3)
    for op in ("pay_order", "get_cart", "create_order", "check_stock"):
        _err(flat, s, op, DOC_ARGS[op], reflection=f"Fix {op} call parameters now.")
    assert len(flat.retrieve("pay_order")) == 3
    an = _store(all_notes=True)
    for op in ("pay_order", "get_cart", "create_order", "check_stock"):
        _err(an, s, op, DOC_ARGS[op])
    assert [n.id for n in an.retrieve("x")] == [1, 2, 3, 4]


def test_fill_to_k_when_no_links():
    m = _store(theta_link=0.99, k=3, k_s=2)
    s = _suite(["pay_order:rename", "get_cart:new_required", "create_order:enum"])
    for op in ("pay_order", "get_cart", "create_order"):
        _err(m, s, op, DOC_ARGS[op], reflection=f"distinct words {op}")
    assert all(not n.links for n in m.notes.values())
    assert len(m.retrieve("anything")) == 3


def test_reflection_list_no_merge_and_fifo_order():
    s = _suite(["pay_order:rename"])
    c1 = ReflectionList(BagOfWordsEmbedder(), TOK, k=2)
    for i in range(3):
        _err(c1, s, "pay_order", DOC_ARGS["pay_order"], reflection=f"reflection {i}")
    assert len(c1.items) == 3
    f = FifoReflections(BagOfWordsEmbedder(), TOK, k=2)
    for i in range(3):
        _err(f, s, "pay_order", DOC_ARGS["pay_order"], reflection=f"reflection {i}")
    assert [it["text"] for it in f.retrieve("q")] == ["reflection 1", "reflection 2"]
    assert f.render(f.retrieve("q")) == ["- reflection 1", "- reflection 2"]


def test_note_caps():
    m = _store()
    s = _suite(["create_order:rename"])
    long = "word " * 200
    _err(m, s, "create_order", DOC_ARGS["create_order"], reflection=long)
    line = m.note_line(m.notes[1])
    assert TOK.count(line[2:]) <= 120


def test_no_outcome_arms_never_write_lines():
    for cond in ("C2-OU", "A1"):
        m = make_memory(cond, BagOfWordsEmbedder(), TOK)
        s = _suite(["pay_order:rename"])
        _err(m, s, "pay_order", DOC_ARGS["pay_order"])
        _ok(m, s, "pay_order", to_current(s.views(0)["pay_order"], DOC_ARGS["pay_order"]))
        assert all(n["line"] is None for n in m.final_state())


def test_shown_text_per_condition():                           # D91
    s = _suite(["pay_order:rename"])
    for cond, expect in (("C2", "pay_order (missing_field)"), ("A2", "Fix the arguments of pay_order"),
                         ("C1", "Reflection: "), ("F", "Reflection: "), ("A1", None)):
        m = make_memory(cond, BagOfWordsEmbedder(), TOK)
        shown, _ = _err(m, s, "pay_order", DOC_ARGS["pay_order"], reflection="Use total.")
        if expect is None:
            assert shown is None
        else:
            assert expect in shown, (cond, shown)
    assert make_memory("C0", None, TOK).on_error(None, None) is None


def test_retrieval_ties_by_note_id_and_part_caps():
    m = _store(links=False, k=2)
    s = _suite(["pay_order:rename", "refund_payment:rename"])
    _err(m, s, "pay_order", DOC_ARGS["pay_order"], reflection="same words")
    _err(m, s, "refund_payment", DOC_ARGS["refund_payment"], reflection="same words")
    m.notes[2].z = m.notes[1].z
    assert [n.id for n in m.retrieve("unrelated query")] == [1, 2]
    n = m.notes[1]
    n.reflection = "word " * 100
    n.items = [{"kind": "rename", "old": f"k{i}", "new": f"n{i}", "type": "string"} for i in range(20)]
    line = m.note_line(n)
    head, rest = line[2:].split(". Fix: ", 1)
    refl, lwc = rest.split(". Last working change", 1)
    # caps per D112 (line <= 40 incl. label; this test used the pre-D112 30 and is corrected)
    assert TOK.count(head) <= 30 and TOK.count(refl) <= 60 and TOK.count("Last working change" + lwc) <= 40
    assert TOK.count(line) <= 120
    # D116: when items are cut the oldest go; the cut is marked at the front
    assert lwc.startswith(": ...; ") and "'k19' is now 'n19'" in lwc and "'k0' " not in lwc


def test_scrub_keeps_operation_names_and_glued_numbers():
    from eem_acc.stage3.agent.scrub import Scrubber
    sc = Scrubber(safe_extra={"open_ticket"})            # a moved id is safe once observed (pointer or docs, M2)
    for v, k in (("create_ticket", "operation"), ("open_ticket", "operation"), (125000, "total"), ("O-9761", "id")):
        sc.add(v, k)
    out, _ = sc.scrub("Call open_ticket instead of create_ticket; Rp125000 and 125,000 for order 9761")
    assert "open_ticket" in out and "create_ticket" in out
    assert "125000" not in out and "125,000" not in out and "9761" not in out


def test_scrub_case_insensitive_safe_and_response_words():          # re-review H1, M4
    from eem_acc.stage3.agent.scrub import Scrubber
    sc = Scrubber(safe_extra={"jakarta", "CARD"})        # remapped values stated by an observed error (M2)
    for v in ("Jakarta", "Card", "paid", "Gold", "Shipped"):
        sc.add(v, "x")
    txt = "Input should be 'jakarta' or 'CARD'; the order must be paid before it can be shipped (gold)."
    assert sc.scrub(txt) == (txt, 0)


def test_scrub_normalised_forms():                                  # D103
    from eem_acc.stage3.agent.scrub import Scrubber, task_scrubber
    sc = Scrubber()
    sc.add("dewi.santoso72@mail.example", "email")
    sc.add("Dewi Santoso", "name")
    sc.add(445000, "total")
    sc.add("445000", "amount")
    out, _ = sc.scrub("dewi.santoso72 / dewi_santoso72 / Santoso / dewi_santoso / 445 000 / 445.000 / 445,000")
    for leak in ("dewi.santoso72", "dewi_santoso72", "Santoso", "dewi_santoso", "445 000", "445.000", "445,000"):
        assert leak not in out, (leak, out)
    inst = {"setup": {"products": [{"sku": "SKU-1234", "name": "Nusantara Kopi Arabica 250 g", "price": 50000,
                                    "stock": {}}]}, "params": {}}
    out, _ = task_scrubber(inst).scrub("Buy Nusantara Kopi Arabica 250 g (SKU-1234)")
    assert out == "Buy <product> (<sku>)"


def test_outdated_mark_kept_and_items_never_cut():                  # re-review M1, D106
    m = _store()
    s = _suite(["pay_order:rename"])
    _err(m, s, "pay_order", DOC_ARGS["pay_order"], reflection="word " * 100)
    n = m.notes[1]
    n.kt = n.kt.__class__(n.kt.api_name, "x " * 50, "y " * 50, n.kt.error_type, n.kt.error_keys)
    n.items = [{"kind": "remap", "key": "method", "from": "card", "value": "CARD"},
               {"kind": "rename", "old": "amount", "new": "total", "type": "integer"},
               {"kind": "added", "key": "currency", "type": "string", "value": "IDR"}]
    n.possibly_outdated = True
    line = m.note_line(n)
    assert TOK.count(line) <= 120
    if "Last working change" in line:
        assert line.endswith("(possibly outdated)")
        tail = line.split("Last working change: ", 1)[1]
        assert tail.count('"') % 2 == 0 and "( (" not in tail
    n.reflection = "short"
    line = m.note_line(n)
    assert "Last working change" in line and line.endswith("(possibly outdated)")


def test_remap_recheck_semantics():                                 # re-review M3
    from eem_acc.stage3.apisuite.leakcheck import item_holds
    it = {"kind": "remap", "key": "method", "from": "card", "value": "CARD"}
    assert item_holds(it, {"method": "TRANSFER"}, "pay_order")
    assert not item_holds(it, {"method": "card"}, "pay_order")


def test_eem_reads_no_ground_truth_aliases_from_observations():     # second opinion M2
    s = _suite(["create_ticket:moved", "create_ticket:rename"], style="terse")
    m = _store()
    _err(m, s, "create_ticket", DOC_ARGS["create_ticket"])            # terse: no pointer
    assert m.canonical("open_ticket") == "open_ticket"                # not known yet
    _ok(m, s, "get_api_docs", {"operation": "create_ticket"})
    assert m.canonical("open_ticket") == "create_ticket"              # learned from the docs response
    # the suite op passed in the context is ignored by EEM: a success on the new id whose suite op is the old
    # operation does not close the note while EEM has not observed the alias
    s2 = _suite(["create_ticket:moved"], style="terse")
    m2 = _store()
    _err(m2, s2, "create_ticket", DOC_ARGS["create_ticket"])
    new_id = s2.views(0)["create_ticket"].route
    good = to_current(s2.views(0)["create_ticket"], DOC_ARGS["create_ticket"])
    m2.on_success(SuccessCtx(0, 0, 1, new_id, "create_ticket", good, {}))
    assert m2.notes[1].open and m2.canonical(new_id) == new_id


def test_line_budgeted_before_reflection():                          # D112
    m = _store()
    s = _suite(["create_order:rename", "create_order:new_required"])
    _err(m, s, "create_order", DOC_ARGS["create_order"], reflection="word " * 80)
    n = m.notes[1]
    n.items = [{"kind": "rename", "old": "items[].qty", "new": "items[].quantity", "type": "integer"},
               {"kind": "added", "key": "currency", "type": "string", "value": "IDR"}]
    text, st = m.note_render(n)
    assert st["line_shown"] and st["items_shown"] == 2 and TOK.count(text) <= 120
    assert "Fix: word" in text


def test_business_rule_errors_do_not_mark():                         # D114
    s = _suite(["pay_order:rename"])
    m = _store()
    _err(m, s, "pay_order", DOC_ARGS["pay_order"])
    good = to_current(s.views(0)["pay_order"], DOC_ARGS["pay_order"])
    _ok(m, s, "pay_order", good)
    _err(m, s, "pay_order", good, task=2)                             # already paid -> 409 business rule
    assert not m.notes[1].possibly_outdated


def test_alias_from_410_pointer_and_from_moved_items():               # second opinion M2, r4 review
    s = _suite(["create_ticket:moved"], style="informative")
    m = _store()
    _err(m, s, "create_ticket", DOC_ARGS["create_ticket"])
    assert m.canonical("open_ticket") == "create_ticket"               # learned from the 410 pointer
    # terse: the alias comes from the docs response (a moved item never teaches a new alias: a close already
    # needs the alias, re-review L4; the redundant rule was removed)
    s3 = _suite(["create_ticket:moved"], style="terse")
    m3 = _store()
    _err(m3, s3, "create_ticket", DOC_ARGS["create_ticket"])
    _ok(m3, s3, "get_api_docs", {"operation": "create_ticket"})
    assert m3.alias == {"open_ticket": "create_ticket"}
    _ok(m3, s3, "open_ticket", DOC_ARGS["create_ticket"])
    assert any(it["kind"] == "moved" and it["op"] == "open_ticket" for it in m3.notes[1].items)
    assert m3.alias == {"open_ticket": "create_ticket"}


def test_merge_recheck_close_dedups_kept_and_new_items():             # r4 review H2 regression
    s = _suite(["pay_order:rename"])
    m = _store()
    _err(m, s, "pay_order", DOC_ARGS["pay_order"])
    good = to_current(s.views(0)["pay_order"], DOC_ARGS["pay_order"])
    _ok(m, s, "pay_order", good)
    item = m.notes[1].items
    assert len(item) == 1
    _err(m, s, "pay_order", DOC_ARGS["pay_order"], task=1, reflection="Again.")
    _ok_noexec(m, "pay_order", good, task=1, step=2)
    assert [e["event"] for e in m.events] == ["create", "close", "merge", "recheck", "close"]
    close = m.events[-1]
    assert close["new_items"] == item and close["items"] == item       # kept + new, deduplicated
    assert not m.notes[1].open and not m.notes[1].possibly_outdated


def test_d114_marking_per_error_type():
    import dataclasses
    from eem_acc.stage3.agent.memory import DRIFT_SIGNAL
    types = sorted(DRIFT_SIGNAL) + ["business_rule", "resource_not_found", "unknown"]
    for et in types:
        s = _suite(["pay_order:rename", "create_order:rename"])
        m = _store()
        _err(m, s, "pay_order", DOC_ARGS["pay_order"])
        _ok(m, s, "pay_order", to_current(s.views(0)["pay_order"], DOC_ARGS["pay_order"]))
        r = s.execute("pay_order", DOC_ARGS["pay_order"], 1)
        kt = dataclasses.replace(parse_error("pay_order", r.status, r.body, DOC_ARGS["pay_order"]), error_type=et)
        m.on_error(ErrorCtx(0, 1, 0, "pay_order", "pay_order", DOC_ARGS["pay_order"], r.status, r.body, "x", kt,
                            r.truth.as_json()), "r")
        marked = m.notes[1].possibly_outdated
        if et == m.events[0]["error_type"]:                            # same key: a merge keeps the items (D113)
            assert m.events[-1]["event"] == "merge" and marked, et
            continue
        assert marked == (et in DRIFT_SIGNAL), et
        assert ("mark" in [e["event"] for e in m.events]) == marked, et
    assert {"business_rule", "resource_not_found"}.isdisjoint(DRIFT_SIGNAL)


def test_orphan_close_without_schema_is_logged_and_line_free():
    s = _suite(["pay_order:rename"])
    m = _store()
    r = s.execute("frobnicate", {"x": 1}, 0)
    kt = parse_error("frobnicate", r.status, r.body, {"x": 1})
    m.on_error(ErrorCtx(0, 0, 0, "frobnicate", None, {"x": 1}, r.status, r.body, "x", kt, r.truth.as_json()), "r")
    _ok_noexec(m, "frobnicate", {"x": 1})
    n, e = m.notes[1], m.events[-1]
    assert not n.open and n.items == [] and e["event"] == "close" and e["no_schema"]
    assert "Last working change" not in m.note_line(n)


def test_d115_close_via_known_alias_from_another_task():
    s = _suite(["create_ticket:moved"], style="terse")
    m = _store()
    _err(m, s, "create_ticket", DOC_ARGS["create_ticket"], task=0)     # terse: no pointer
    _ok(m, s, "get_api_docs", {"operation": "create_ticket"}, task=5)  # another task reads the docs
    new_id = s.views(0)["create_ticket"].route
    assert (0, 0, "create_ticket") not in m.docs_named
    _ok(m, s, new_id, to_current(s.views(0)["create_ticket"], DOC_ARGS["create_ticket"]), task=0)
    e = m.events[-1]
    assert e["event"] == "close" and e["moved_to"] == new_id and e["canonical_op"] == "create_ticket"
    assert any(it["kind"] == "moved" for it in e["items"])


def test_d116_compact_same_field_remaps_and_round_trip():
    from eem_acc.stage3.agent.scripted import parse_line
    m = _store()
    s = _suite(["create_shipment:enum"] if "create_shipment:enum" in CATALOGUE else ["pay_order:rename"])
    _err(m, s, *(("create_shipment", DOC_ARGS["create_shipment"]) if "create_shipment:enum" in CATALOGUE
                 else ("pay_order", DOC_ARGS["pay_order"])), reflection="")
    n = m.notes[1]
    n.items = [{"kind": "remap", "key": "carrier", "from": "jne", "value": "JNE_REG"},
               {"kind": "rename", "old": "amount", "new": "total", "type": "integer"},
               {"kind": "remap", "key": "carrier", "from": "sicepat", "value": "SICEPAT_REG"}]
    shown = m.display_items(n.items)
    assert shown[0] == """'carrier': "jne" -> "JNE_REG", "sicepat" -> "SICEPAT_REG\"""" and len(shown) == 2
    text, st = m.note_render(n)
    assert st["items_shown"] == 3 and shown[0] in text
    key = lambda it: json.dumps(it, sort_keys=True)
    got = parse_line(text.split("Last working change: ", 1)[1])
    assert sorted(map(key, got)) == sorted(map(key, n.items))


def test_line_cut_edge_single_item_without_marker():
    m = _store()
    s = _suite(["pay_order:rename"])
    _err(m, s, "pay_order", DOC_ARGS["pay_order"], reflection="")
    n = m.notes[1]
    for w in range(1, 60):                                              # find a key that hits the edge
        long_key = " ".join(["w"] * w)
        last = {"kind": "rename", "old": long_key, "new": "z", "type": "string"}
        one = m.display_items([last])[0]
        if TOK.count("Last working change: ...; " + one) > 40 >= TOK.count("Last working change: " + one):
            break
    else:
        raise AssertionError("no edge case found")
    n.items = [{"kind": "rename", "old": "a", "new": "b", "type": "string"}, last]
    text, st = m.note_render(n)
    assert st["items_shown"] == 1 and ("Last working change: " + one) in text and "..." not in text


def test_canonical_op_resolved_at_close_when_alias_learned_later():   # re-review L3
    s = _suite(["create_ticket:moved", "create_ticket:rename"], style="terse")
    m = _store()
    _err(m, s, "open_ticket", DOC_ARGS["create_ticket"])               # new id, old field name: alias unknown
    assert m.notes[1].canonical_op == "open_ticket"
    _ok(m, s, "get_api_docs", {"operation": "create_ticket"})          # teaches open_ticket -> create_ticket
    _ok(m, s, "open_ticket", to_current(s.views(0)["create_ticket"], DOC_ARGS["create_ticket"]))
    e = m.events[-1]
    assert e["event"] == "close" and e["canonical_op"] == "create_ticket" and not e.get("no_schema")
    assert any(it["kind"] == "rename" for it in e["items"])            # diffed against create_ticket's schema


def test_d115_alias_chain_closes_moved_note():                        # re-review L9
    s = _suite(["create_ticket:moved"], style="terse")
    m = _store()
    _err(m, s, "create_ticket", DOC_ARGS["create_ticket"])
    assert m.notes[1].error_type in ("operation_removed", "route_not_found")
    m.alias.update({"open_ticket_v3": "open_ticket", "open_ticket": "create_ticket"})
    _ok_noexec(m, "open_ticket_v3", DOC_ARGS["create_ticket"])
    e = m.events[-1]
    assert e["event"] == "close" and e["moved_to"] == "open_ticket_v3"


def test_reflection_that_truncates_to_nothing_leaves_no_label():     # re-review L7
    m = _store()
    s = _suite(["pay_order:rename"])
    _err(m, s, "pay_order", DOC_ARGS["pay_order"], reflection="...")
    assert "Fix:" not in m.note_line(m.notes[1])


def test_line_display_logged_as_rendered():                          # re-review L8
    m = _store()
    s = _suite(["pay_order:rename"])
    _err(m, s, "pay_order", DOC_ARGS["pay_order"], reflection="")
    n = m.notes[1]
    n.items = [{"kind": "remap", "key": "method", "from": "card", "value": "CARD"},
               {"kind": "remap", "key": "method", "from": "transfer", "value": "TRANSFER"}]
    snap = n.snapshot()
    assert snap["line_display"] == """Last working change: 'method': "card" -> "CARD", "transfer" -> "TRANSFER\""""
    assert snap["line_display"] in m.note_line(n)


def test_d123_per_writer_triggers_two_agents():
    s = _suite(["pay_order:rename"])
    m = _store()
    _err(m, s, "pay_order", DOC_ARGS["pay_order"], agent=0, task=0)
    other = dict(DOC_ARGS["pay_order"], method="transfer")
    _err(m, s, "pay_order", other, agent=1, task=7, reflection="B")    # another agent merges: owner changes
    n = m.notes[1]
    assert n.owner == (1, 7) and set(n.writers) == {(0, 0), (1, 7)}
    good = to_current(s.views(0)["pay_order"], DOC_ARGS["pay_order"])
    _ok_noexec(m, "pay_order", good, agent=0, task=0)                  # the earlier writer's close is not lost
    e = m.events[-1]
    assert e["event"] == "close" and e["writer"] == [0, 0] and not e["by_last_owner"]
    assert e["trigger_args"] == DOC_ARGS["pay_order"] and n.items and n.open      # (1, 7) still pending
    _ok_noexec(m, "pay_order", to_current(s.views(7)["pay_order"], other), agent=1, task=7)
    assert m.events[-1]["writer"] == [1, 7] and not n.open and not n.writers
    # the same agent's later task replaces its stale writer (RQ1 behaviour)
    _err(m, s, "pay_order", DOC_ARGS["pay_order"], agent=0, task=1)
    _err(m, s, "pay_order", DOC_ARGS["pay_order"], agent=0, task=2)
    assert set(n.writers) == {(0, 2)}


def test_render_fallback_never_crashes():
    class Merging(StubTokenizer):
        """Counts a joined note as more tokens than its parts (a tokenizer that does not split at the joins)."""
        def count(self, text):
            return super().count(text) + (100 if text.startswith("- ") else 0)
    m = EEMStore(BagOfWordsEmbedder(), Merging())
    s = _suite(["pay_order:rename"])
    _err(m, s, "pay_order", DOC_ARGS["pay_order"], reflection="word " * 80)
    n = m.notes[1]
    n.items = [{"kind": "rename", "old": f"k{i}", "new": f"n{i}", "type": "string"} for i in range(5)]
    text, st = m.note_render(n)
    assert st.get("render_fallback") and Merging().count(text) <= 120 and text.startswith("- ")
    assert st["line_shown"] == ("Last working change" in text)
