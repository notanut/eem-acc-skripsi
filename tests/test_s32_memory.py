"""S3.2a: store, retrieval, outcome update, possibly-outdated re-check (spec 6.3-6.5, 7)."""
import copy

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
    assert TOK.count(head) <= 30 and TOK.count(refl) <= 60 and TOK.count("Last working change" + lwc) <= 30
    assert TOK.count(line) <= 120


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
    # the suite op passed in the context is ignored by EEM
    from eem_acc.stage3.agent.memory import SuccessCtx
    m2 = _store()
    m2.on_success(SuccessCtx(0, 0, 0, "open_ticket", "create_ticket", {}, {}))
    assert m2.canonical("open_ticket") == "open_ticket"


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
