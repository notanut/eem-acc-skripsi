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
    assert len(m.notes) == 1 and n.open and not n.items and n.reflection == "Again." and n.owner == (0, 1)
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
    # a new error of another kind on the same operation (wrong id): the line is kept but marked
    _err(m, s, "pay_order", dict(good, order_id="O-9999"), task=1)
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
