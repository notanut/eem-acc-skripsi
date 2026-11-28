"""S3.5 spec sections 6-9 and 16: secondary, descriptive and mechanism analyses, and the report (canonical hash,
pins, the primary part's imports). Hand-built units in the real payload shapes (enum items are "remap", nested item
paths, silent drift, terse notes that state what was sent, never-started tasks with a completion time), each
assertion aimed at a mutant the code review found surviving."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from eem_acc.stage3.analysis import data as D
from eem_acc.stage3.analysis import descriptive as DS
from eem_acc.stage3.analysis import mechanism as M
from eem_acc.stage3.analysis import report as RP
from eem_acc.stage3.analysis import secondary as S

T = "s35-T"
ROOT = Path(__file__).resolve().parents[1]


def tasks_of(solved, n=20, api=5, unsolved_api=None, tok_in=100, tok_out=7, llm=3, **extra):
    return [{"pos": i, "success": i in solved, "api_calls": api if i in solved else (unsolved_api or api),
             "llm_calls": llm, "tok_in": tok_in, "tok_out": tok_out, "template": f"T{i % 3}",
             "reason": "success" if i in solved else "max_steps", **extra} for i in range(n)]


def unit(seed, cond, solved=(), n=20, rq="rq1", kind=None, window=None, phase=T, tag="", tasks=None, spec=None,
         status="done", **raw):
    u = D.UnitData(rq, kind or rq, seed, cond, window, tag, tasks if tasks is not None else tasks_of(set(solved), n),
                   phase=phase, n_tasks=n, raw=raw, status=status)
    u.spec = spec
    return u


def rq1_set(seeds=range(2000, 2004), c0=10, c1=11, c2=12, c3=12, api=(6, 5, 4, 4)):
    us = []
    for s in seeds:
        for c, k, a in zip(("C0", "C1", "C2", "C3"), (c0, c1, c2, c3), api):
            us.append(unit(s, c, tasks=tasks_of(set(range(k)), api=a, unsolved_api=50)))
    return us


# ----------------------------------------------------------------------------------------------------- secondary
def test_c1_vs_c0_direction_and_two_sided_p():
    r = S.c1_vs_c0(rq1_set())
    assert r["success"]["mean"] == pytest.approx(1 / 20) and r["success"]["n"] == 4
    assert r["success"]["p_two_sided"] == pytest.approx(2 * 2 ** -4)
    assert r["api_calls"]["mean"] == pytest.approx(5 - 6)                 # C1 minus C0, jointly solved only


def test_c3_vs_c2_tost_integer_exact_and_directions():
    eq = S.c3_vs_c2_tost(rq1_set(seeds=range(2000, 2008)))                  # C3 == C2 on 8 streams
    assert eq["p_lower"] == pytest.approx(2 ** -8) and eq["p_upper"] == pytest.approx(2 ** -8)
    assert eq["equivalent"] is True and eq["margin"] == pytest.approx(1 / 20) and eq["t_check_equivalent"] is True
    worse = S.c3_vs_c2_tost(rq1_set(seeds=range(2000, 2008), c3=11))      # one task worse: the lower test ties
    assert worse["p_lower"] == 1.0 and worse["p_upper"] == pytest.approx(2 ** -8) and worse["equivalent"] is False
    better = S.c3_vs_c2_tost(rq1_set(seeds=range(2000, 2008), c3=13))     # one task better: the upper test ties
    assert better["p_upper"] == 1.0 and better["p_lower"] == pytest.approx(2 ** -8)
    assert S.c3_vs_c2_tost([])["p_tost"] is None


def test_tost_t_check_needs_both_bounds_inside():
    us = rq1_set(seeds=range(2000, 2006))
    for u in us:                                       # C3 better by 0 or 2 tasks: CI crosses +margin, not -margin
        if u.condition == "C3" and u.seed % 2:
            u.tasks = tasks_of(set(range(14)), api=4)
    r = S.c3_vs_c2_tost(us)
    assert r["ci95"][0] > -r["margin"] and r["ci95"][1] > r["margin"] and r["t_check_equivalent"] is False


def test_integer_tost_agrees_with_stage2_away_from_float_boundaries():
    from eem_acc.analysis import tost_sign_flip
    rng = np.random.default_rng(3)
    for _ in range(30):
        us = []
        for s in range(2000, 2008):
            k2, k3 = int(rng.integers(8, 14)), int(rng.integers(6, 16))
            us += [unit(s, "C0", range(5)), unit(s, "C1", range(5)), unit(s, "C2", range(k2)), unit(s, "C3", range(k3))]
        r = S.c3_vs_c2_tost(us)
        d = [(len([t for t in by.tasks if t["success"]])) for by in us]
        d = np.array([(d[i + 3] - d[i + 2]) / 20 for i in range(0, len(us), 4)])
        if np.any(np.isclose(np.abs(d), 1 / 20)):
            continue                                     # float boundary ties: only the integer version is exact
        assert r["p_tost"] == pytest.approx(tost_sign_flip(d, 1 / 20))


def test_cost_ratio_and_splits():
    us = rq1_set()
    for u in us:
        for t in u.tasks:
            t["tok_out"] = 999
    c = S.cost_per_solved(us, ("C0", "C2"))
    assert c["C0"]["llm_calls_per_solved"] == pytest.approx(4 * 20 * 3 / 40)
    assert c["C0"]["tok_in_per_solved"] == pytest.approx(4 * 20 * 100 / 40)     # prompt tokens, not output tokens
    r = S.ratio_of_sums(us)                         # unsolved tasks have 50 calls: excluded (jointly solved only)
    assert r["C2/C0"]["ratio"] == pytest.approx(4 / 6) and r["C2/C1"]["ratio"] == pytest.approx(4 / 5)
    sp = S.error_splits(us, ("C0", "C2"))
    assert sp["C0"]["budget_exhausted"] == pytest.approx(0.5) and sp["C2"]["budget_exhausted"] == pytest.approx(0.4)
    assert sp["C0"]["tasks"] == 80


def test_population_uses_complete_streams_for_primary_conditions():
    us = rq1_set() + [unit(2009, "C2", range(20))]                    # a lone C2 unit on an incomplete stream
    assert S.cost_per_solved(us, ("C2",))["C2"]["units"] == 4
    assert S.cost_per_solved(us + [unit(2009, "C2-OU", range(3))], ("C2-OU",))["C2-OU"]["units"] == 1


def test_learning_curves():
    lc = S.learning_curves(rq1_set())
    assert lc["C2"][:12] == [1.0] * 12 and lc["C2"][12:] == [0.0] * 8 and len(lc["C0"]) == 20


SPEC = {"drift_schedule": [[0, ["get_cart:rename", "pay_order:type", "pay_order:new_required"]],
                           [10, ["get_cart:rename", "pay_order:type", "pay_order:new_required", "find_customer:moved"]]],
        "styles": {"get_cart": "informative", "pay_order": "terse", "find_customer": "informative"}}


def full_set_with(c2_tasks=None, c2_steps=None, spec=SPEC, **raw):
    us = []
    for c in ("C0", "C1", "C3"):
        us.append(unit(2000, c, range(5), spec=spec))
    us.append(unit(2000, "C2", tasks=c2_tasks, spec=spec, steps=c2_steps or [], **raw))
    return us


def test_active_and_op_pairs():
    assert S.op_pairs(SPEC, 3) == {"get_cart": ["rename"], "pay_order": ["type", "new_required"]}
    assert "find_customer" in S.op_pairs(SPEC, 10) and "find_customer" not in S.op_pairs(SPEC, 9)


def test_exposures_by_type_and_style():
    tasks = [{"pos": 0, "success": True, "first_call": {"pay_order": "ok", "get_cart": "drift"}},
             {"pos": 1, "success": True, "first_call": {"pay_order": "agent", "search_products": "ok"}},
             {"pos": 2, "success": True, "first_call": {"get_cart": "ok"}}]
    e = S.exposures(full_set_with(c2_tasks=tasks), ("C2",))["C2"]
    assert e["by_drift_type"]["type"] == {"n": 2, "first_call_ok": 0.5}            # an agent error is not ok
    assert e["by_drift_type"]["new_required"] == {"n": 2, "first_call_ok": 0.5}    # a two-pair op counts under both
    assert e["by_drift_type"]["rename"] == {"n": 2, "first_call_ok": 0.5}
    assert e["by_style"] == {"informative": {"n": 2, "first_call_ok": 0.5}, "terse": {"n": 2, "first_call_ok": 0.5}}


def test_drifted_call_share():
    steps = [{"k": 1, "task": 0, "canonical_op": "get_cart", "action": "get_cart"},
             {"k": 2, "task": 0, "canonical_op": "search_products", "action": "search_products"},
             {"k": 3, "task": 0, "canonical_op": None, "action": "get_api_docs"},
             {"k": 4, "task": 10, "canonical_op": "find_customer", "action": "find_customer"},
             {"task": 0, "action": "finish"}]
    r = S.drifted_call_share(full_set_with(c2_tasks=tasks_of(set()), c2_steps=steps), ("C2",))["C2"]
    assert r == {"calls": 3, "share_drifted": pytest.approx(2 / 3)}


def test_mixed_model_recovers_a_planted_position_effect():
    rng = np.random.default_rng(1)
    us = []
    for s in range(2000, 2012):
        for c in ("C0", "C1", "C2", "C3"):
            slope = 0.25 if c in ("C2", "C3") else 0.0
            solved = {i for i in range(20) if rng.uniform() < 1 / (1 + np.exp(-(slope * (i - 9.5))))}
            us.append(unit(s, c, solved))
    mm = S.mixed_model(us)
    assert "gee_error" not in mm and "bayes_error" not in mm
    key = "C(condition, Treatment('C0'))[T.C2]:pos_c"
    assert mm["gee"]["fixed"][key]["mean"] > 0.1 and mm["gee"]["fixed"][key]["lo95"] > 0
    assert mm["gee"]["cov_type"] == "bias_reduced"
    assert mm["vb"]["fixed"][key]["mean"] > 0.1 and "vb_map_max_abs_diff" in mm and "warnings" in mm["map"]
    assert S.mixed_model(rq1_set(seeds=[2000]))["error"]


RQ3_TASKS = [{"pos": 0, "success": True, "completion_time": 10.0, "completion_censored": False},
             {"pos": 1, "success": True, "completion_time": 20.0, "completion_censored": False},
             {"pos": 2, "success": False, "completion_time": 50.0, "completion_censored": True,
              "unfinished_at_t_max": True},
             {"pos": 3, "success": False, "completion_time": 270.0, "completion_censored": True,
              "never_started": True, "unfinished_at_t_max": True}]      # rq3.py: t_end - warmup, never started


def test_rq3_completion_versions():
    c = S.completion(unit(4000, "C2", tasks=RQ3_TASKS, rq="rq3", window="acc"))
    assert c == {"finished_mean": 15.0, "censored_mean": pytest.approx(80 / 3), "unfinished": 1, "never_started": 1}


def rq3_unit(seed, cond, window, k, scale=1.0, **kw):
    tasks = [dict(t, completion_time=t["completion_time"] * scale) for t in RQ3_TASKS]
    for i, t in enumerate(tasks):
        t["success"] = i < k
    raw = {"llm_metrics": {"overload_rate": 0.1 * scale, "slo_violation_rate": 0.2, "mean_latency_excess": 0.3},
           "J_rq3": 0.4, "infra": {"windows": {"llm_phase": {"goodput": 5.0 * scale, "J": 0.9, "throttle_rate": 0.05,
                                                             "slo_violation_rate": 0.06}}},
           "drift_errors": 7, "notes_created": 3, "llm_calls": [{"transient": False}] * 30 + [{"transient": True}] * 5,
           "t_end": 120.0, "stamp": {"warmup": 20.0}}
    return unit(seed, cond, tasks=tasks, rq="rq3", window=window, n=20, **{**raw, **kw})   # a unit's n_tasks is 20


def test_rq3_cells_both_traffics():
    cell = S.rq3_secondary([rq3_unit(4000, "C2", "acc", 2)])["cells"]["C2+acc"]
    assert cell["llm_overload_rate"] == pytest.approx(0.1) and cell["llm_slo_violation_rate"] == 0.2
    assert cell["llm_mean_latency_excess"] == 0.3 and cell["llm_J"] == 0.4
    assert cell["background_overload_rate"] == 0.05 and cell["background_slo_violation_rate"] == 0.06
    assert cell["background_goodput"] == 5.0 and cell["background_J"] == 0.9
    assert cell["llm_goodput"] == pytest.approx(30 / 100)
    assert cell["drift_errors"] == 7 and cell["notes_created"] == 3 and cell["unfinished"] == 1


def test_rq3_paired_differences_and_labels():
    us = [rq3_unit(s, "C2", "acc", 2) for s in (4000, 4001)] + \
         [rq3_unit(s, "C2", "fixed4", 1, scale=2.0, phase="s35-S") for s in (4000, 4001)] + \
         [rq3_unit(s, "C0", "acc", 1, scale=1.5) for s in (4000, 4001)]
    p = S.rq3_secondary(us, (T, "s35-S"))["paired"]
    assert p["C2+fixed4 - C2+acc"]["success"]["mean"] == pytest.approx(-1 / 4)
    # "finished" = not censored (tasks 0, 1: 10 and 20 s, times the cell's scale), whether solved or not
    assert p["C2+fixed4 - C2+acc"]["completion_finished"]["mean"] == pytest.approx(30.0 - 15.0)
    assert p["C2+acc - C0+acc"]["completion_finished"]["mean"] == pytest.approx(15.0 - 22.5)
    assert "C2+acc - C2+none" not in p


# ----------------------------------------------------------------------------------------------------- descriptive
def test_arms_streams_population_and_signs():
    us = rq1_set(seeds=[1900, 1901, 1902]) + [unit(1900, "F", tasks=tasks_of(set(range(5)), api=9), phase="s35-S"),
                                               unit(1901, "F", tasks=tasks_of(set(range(7)), api=9), phase="s35-S")]
    a = DS.arms_vs_c2(us)
    assert a["streams"] == [1900, 1901] and a["arms"]["F"]["units"] == 2 and a["arms"]["C0"]["units"] == 2
    assert a["arms"]["F"]["minus_C2_success"]["mean"] == pytest.approx(((5 - 12) + (7 - 12)) / 2 / 20)
    assert a["arms"]["F"]["minus_C2_api_calls"]["mean"] == pytest.approx(9 - 4)          # arm minus C2
    assert a["arms"]["F"]["C2_on_these_streams"]["streams"] == [1900, 1901]


SP_SPEC = {"drift_schedule": [[0, ["create_shipment:sp1", "get_cart:rename"]],
                              [10, ["create_shipment:sp2", "get_cart:rename"]]], "styles": {}, "sp_op": "create_shipment"}


def sp_tasks():
    calls = {10: "drift", 12: "drift", 14: "ok", 3: "drift"}
    return [{"pos": i, "success": i not in (9, 11), "first_call": ({"create_shipment": calls[i]} if i in calls else {})}
            for i in range(20)]


def test_sp_redrift_exposures_and_note_update():
    events = [{"event": "close", "suite_op": "create_shipment", "at": [0, 12, 3],
               "items": [{"kind": "rename", "old": "carrier", "new": "shipper", "type": "string"}]},
              {"event": "create", "suite_op": "create_shipment", "at": [0, 11, 1], "items": []},
              {"event": "mark", "suite_op": "create_shipment", "at": [0, 15, 1]},
              {"event": "close", "suite_op": "create_shipment", "at": [0, 9, 1], "items": []}]
    u = unit(3000, "C2", tasks=sp_tasks(), kind="sp", phase="s35-S", spec=SP_SPEC, note_events=events)
    c0 = unit(3000, "C0", tasks=sp_tasks(), kind="sp", phase="s35-S", spec=SP_SPEC)
    r = DS.sp([u, c0])["conditions"]
    assert r["C2"]["success_before"] == pytest.approx(9 / 10)                  # pos 10 is after the re-drift
    assert r["C2"]["success_after"] == pytest.approx(9 / 10)
    assert r["C2"]["first_call_ok_by_exposure"] == {"1": {"n": 1, "ok": 0.0}, "2": {"n": 1, "ok": 0.0},
                                                    "3": {"n": 1, "ok": 1.0}}
    assert r["C2"]["note_updated_after_redrift"] == {"units": 1, "updated": 1, "updated_naming_new_name": 1}
    assert "note_updated_after_redrift" not in r["C0"]
    only_mark = unit(3000, "C2", tasks=sp_tasks(), kind="sp", phase="s35-S", spec=SP_SPEC,
                     note_events=[events[2], events[3]])
    assert DS.sp([only_mark])["conditions"]["C2"]["note_updated_after_redrift"]["updated"] == 0
    plain = unit(3000, "C2", tasks=sp_tasks(), kind="sp", phase="s35-S", spec=SP_SPEC, note_events=[events[1]])
    assert DS.sp([plain])["conditions"]["C2"]["note_updated_after_redrift"] == \
        {"units": 1, "updated": 1, "updated_naming_new_name": 0}


def test_sp_refuses_a_schedule_that_does_not_redrift_its_op():
    bad = dict(SP_SPEC, sp_op="get_cart")
    with pytest.raises(ValueError):
        DS.sp([unit(3000, "C2", tasks=sp_tasks(), kind="sp", phase="s35-S", spec=bad)])


def test_rq3_extras_keep_the_sensitivity_tag_apart():
    us = []
    for s in (4000, 4001):
        us += [rq3_unit(s, "C2", "acc", 3), rq3_unit(s, "C2-private", "acc", 2, scale=1.2, phase="s35-S")]
        for c, w, k in (("C2", "acc", 3), ("C0", "acc", 1), ("C2", "none", 2), ("C0", "none", 0)):
            us.append(rq3_unit(s, c, w, k, phase="s35-S", tag="sens"))
    r = DS.rq3_extras(us)
    assert r["private_minus_shared"]["success"]["mean"] == pytest.approx(-1 / 4)
    assert r["private_minus_shared"]["completion_finished"]["mean"] == pytest.approx(15 * 1.2 - 15)
    assert r["sensitivity"]["R1_C2_minus_C0_acc"]["mean"] == pytest.approx(2 / 4)
    assert r["sensitivity"]["R2_acc_minus_none_C2"]["mean"] == pytest.approx(1 / 4)


# ----------------------------------------------------------------------------------------------------- mechanism
def test_truth_tokens_every_type():
    assert M.truth_tokens("get_cart:rename") == [["client_id"]]
    assert M.truth_tokens("pay_order:type") == [["amount", "decimal"], ["amount", "string"]]
    assert ["query", "list"] in M.truth_tokens("search_products:type")
    assert M.truth_tokens("reserve_stock:enum") == [["jakarta"], ["surabaya"], ["bandung"]]
    assert M.truth_tokens("get_order:new_required") == [["api_version"]]
    assert M.truth_tokens("cancel_order:moved") == [["void_order"]]
    assert M.truth_tokens("create_shipment:silent") == M.truth_tokens("create_shipment:new_required")
    assert M.truth_tokens("create_shipment:sp2") == [["shipper"]]


def test_names_truth_word_boundaries_and_type_adjacency():
    assert M.names_truth("use client_id now", "get_cart:rename")
    assert M.names_truth("items[].client_id", "get_cart:rename")
    assert not M.names_truth("use client_identifier", "get_cart:rename")
    assert not M.names_truth("use xclient_id", "get_cart:rename")
    assert M.names_truth("'amount' is now string", "pay_order:type")
    assert M.names_truth("amount: decimal", "pay_order:type")
    assert not M.names_truth("amount changed; method: string", "pay_order:type")      # field and kind apart
    assert M.names_truth("x", "pay_order:type", items=[{"kind": "type", "key": "amount", "type": "string"}])
    assert M.names_truth("query: list", "search_products:type")


def test_note_text_leaves_out_what_was_sent():
    terse = {"truth": {"kind": "drift", "pair_ids": ["pay_order:type"]}, "items": [],
             "kt": {"expected": "unknown", "received": "amount: string, method: string", "error_keys": []},
             "line": None, "reflection": "the payment failed"}
    assert M.note_names_truth(terse) is False                     # code review H2: sent types are not a fix


def test_note_names_truth_and_item_belongs():
    good = {"truth": {"kind": "silent_drift", "pair_ids": ["create_order:rename"]},
            "items": [{"kind": "rename", "old": "items[].qty", "new": "items[].quantity", "type": "integer"}],
            "line": "Last working change: 'items[].qty' is now 'items[].quantity' (integer)"}
    assert M.note_names_truth(good) is True
    assert M.note_names_truth({"truth": {"kind": "agent"}}) is None
    two = {"truth": {"kind": "drift", "pair_ids": ["get_cart:rename", "pay_order:type"]}, "reflection": "client_id"}
    assert M.note_names_truth(two) is False                                       # every pair must be named
    assert M.item_belongs({"kind": "rename", "new": "items[].quantity"}, "create_order:rename")    # review H1
    assert not M.item_belongs({"kind": "rename", "new": "items[].qty"}, "create_order:rename")
    assert M.item_belongs({"kind": "remap", "key": "warehouse", "from": "JKT", "value": "jakarta"}, "reserve_stock:enum")
    assert not M.item_belongs({"kind": "remap", "key": "warehouse", "from": "x", "value": "JKT"}, "reserve_stock:enum")
    assert not M.item_belongs({"kind": "remap", "key": "other", "value": "jakarta"}, "reserve_stock:enum")
    assert M.item_belongs({"kind": "added", "key": "api_version"}, "get_order:new_required")
    assert M.item_belongs({"kind": "value", "key": "api_version", "value": "2"}, "get_order:new_required")
    assert not M.item_belongs({"kind": "added", "key": "page"}, "get_order:new_required")
    assert M.item_belongs({"kind": "moved", "op": "void_order"}, "cancel_order:moved")
    assert not M.item_belongs({"kind": "moved", "op": "cancel"}, "cancel_order:moved")
    assert M.item_belongs({"kind": "type", "key": "items[].qty", "type": "string"}, "create_order:type")
    assert not M.item_belongs({"kind": "type", "key": "items[].qty", "type": "integer"}, "create_order:type")
    assert not M.item_belongs({"kind": "type", "key": "price", "type": "string"}, "create_order:type")


def test_repeat_exposure_excludes_first_and_uncalled_and_counts_silent():
    tasks = [{"pos": 0, "success": True, "first_call": {"get_cart": "drift"}},          # first exposure: excluded
             {"pos": 1, "success": True, "first_call": {}},                              # not called: no exposure
             {"pos": 2, "success": True, "first_call": {"get_cart": "silent_drift"}},
             {"pos": 3, "success": True, "first_call": {"get_cart": "ok"}},
             {"pos": 4, "success": True, "first_call": {"get_cart": "agent"}}]
    r = M.repeat_exposure(full_set_with(c2_tasks=tasks), ("C2",), (T,))["C2"]
    assert r == {"repeat_exposures": 3, "drift_error_rate": pytest.approx(1 / 3)}


def test_retrieval_recall_any_and_visibility():
    steps = [{"k": 1, "notes_on_op": [1], "visible": [[1, "get_cart"]], "retrieved": [1]},
             {"k": 2, "notes_on_op": [1, 2], "visible": [[1, "get_cart"]], "retrieved": [3]},
             {"k": 3, "notes_on_op": [1, 4], "visible": [[1, "x"], [4, "x"]], "retrieved": [4]},
             {"k": 4, "notes_on_op": [], "visible": [], "retrieved": []}]
    r = M.retrieval_recall([unit(2000, "C2", steps=steps)], ("C2",), (T,))["C2"]
    assert r == {"calls": 3, "recall": pytest.approx(2 / 3), "notes_not_yet_visible": 1, "meaning": "top-k retrieval"}


def test_note_correctness_stages_precision_and_c1_truth():
    tr = {"kind": "drift", "pair_ids": ["get_cart:rename"]}
    events = [{"event": "create", "truth": tr, "items": []},
              {"event": "merge", "truth": tr, "items": []},
              {"event": "close", "truth": tr, "items": [{"kind": "rename", "old": "customer_id", "new": "client_id",
                                                        "type": "string"}]},
              {"event": "mark", "truth": tr}]
    final = [{"truth": tr, "suite_op": "get_cart",
              "items": [{"kind": "rename", "old": "customer_id", "new": "client_id", "type": "string"},
                        {"kind": "added", "key": "x", "type": "string"}]},
             {"truth": {"kind": "agent"}, "suite_op": "get_cart", "items": [{"kind": "added", "key": "y", "type": "string"}]}]
    u = unit(2000, "C2", spec=SPEC, note_events=events, final_notes=final)
    nc = M.note_correctness([u], ("C2",), (T,))["C2"]
    assert nc["before_update"] == {"n": 2, "names_truth": 0.0} and nc["after_update"] == {"n": 1, "names_truth": 1.0}
    assert nc["final"]["names_truth"] == 1.0
    pr = M.note_precision([u], ("C2",), (T,))["C2"]
    assert pr == {"items": 2, "precision_op": 0.5, "precision_own_truth": 0.5}             # agent note excluded
    c1 = unit(2000, "C1", note_events=[{"event": "create", "id": 7, "truth": tr, "text": "send client_id"}],
              final_notes=[{"id": 7, "text": "send client_id", "suite_op": "get_cart"}])
    assert M.note_correctness([c1], ("C1",), (T,))["C1"]["final"] == {"n": 1, "names_truth": 1.0}


def test_precision_op_level_counts_the_other_pair_on_the_operation():
    note = {"truth": {"kind": "drift", "pair_ids": ["pay_order:type"]}, "suite_op": "pay_order",
            "items": [{"kind": "type", "key": "amount", "type": "string"}, {"kind": "added", "key": "currency"}]}
    from eem_acc.stage3.apisuite.drift import get_pair
    nr = get_pair("pay_order:new_required").new_field.name
    note["items"][1]["key"] = nr
    u = unit(2000, "C2", spec=SPEC, final_notes=[note])
    pr = M.note_precision([u], ("C2",), (T,))["C2"]
    assert pr == {"items": 2, "precision_op": 1.0, "precision_own_truth": 0.5}


def test_c1_duplicate_crowding():
    events = [{"event": "create", "id": i, "text": t} for i, t in enumerate(["a", "a", "b", "c"])]
    steps = [{"retrieved": [0, 1, 2]}, {"retrieved": [2, 3]}, {"retrieved": [0]}]
    r = M.c1_duplicate_crowding([unit(2000, "C1", note_events=events, steps=steps)], (T,))
    assert r == {"retrievals": 2, "share_with_duplicates": 0.5, "mean_duplicate_slot_share": pytest.approx((1 / 3) / 2)}


# ----------------------------------------------------------------------------------------------------- data, report
def test_from_result_keeps_steps_and_flattens_private_stores():
    from eem_acc.stage3.llm.runner import Unit
    u = Unit("s35-S", "rq3", "test", "rq3", 4000, "C2-private", "acc")
    res = {"unit": u.to_json(), "tasks": [], "stamp": {"spec_sha256": u.spec()["sha256"]},
           "steps": [{"k": 1}], "note_events": [[{"event": "create", "id": 1}], [{"event": "create", "id": 1}]]}
    d = D.from_result(res)
    assert d.raw["steps"] == [{"k": 1}]
    assert [e["store_agent"] for e in d.raw["note_events"]] == [0, 1]


def test_canonical_rounding_and_hash():
    a = {"p": 2 ** -14, "x": 0.1 + 0.2, "n": np.int64(3), "t": (1.0, float("nan")), 5: np.bool_(True),
         "u": np.uint64(7), "h": np.float16(0.5), "i8": np.int8(-2)}
    b = {"p": 2 ** -14, "x": 0.3 + 1e-14, "n": 3, "t": [1.0, None], "5": True, "u": 7, "h": 0.5, "i8": -2}
    assert RP.canonical(a)["p"] == 2 ** -14                                   # dyadic p survives
    assert RP.canonical_hash(a) == RP.canonical_hash(b)
    assert RP.canonical_hash({"x": 1.2345678901}) == RP.canonical_hash({"x": 1.2345678904})       # 11th digit
    assert RP.canonical_hash({"x": 1.234567890}) != RP.canonical_hash({"x": 1.234567891})       # 10th digit
    assert RP.canonical_hash({"x": 1.23456789}) != RP.canonical_hash({"x": 1.23456788})         # 9th digit
    with pytest.raises(TypeError):
        RP.canonical({"x": object()})


def test_check_pins(monkeypatch):
    pins = {k: v for k, v in json.loads(RP.PINS.read_text()).items() if not k.startswith("_")}
    monkeypatch.setattr(RP, "versions", lambda mods: {m: "0" for m in mods})
    with pytest.raises(RuntimeError):
        RP.check_pins(strict=True)
    assert set(RP.check_pins(strict=False)["mismatch"]) == set(pins)
    monkeypatch.setattr(RP, "versions", lambda mods: {m: pins.get(m, "1.18.1") for m in mods})
    assert RP.check_pins(strict=True)["mismatch"] == {}                        # matching pins never raise
    assert RP.check_pins(strict=False, expect_scipy="9.9")["mismatch"] == {"scipy": ("1.18.1", "9.9")}


def test_primary_failed_as_unsolved_version_differs():
    us = rq1_set(seeds=range(2000, 2003))
    us = [u for u in us if not (u.seed == 2001 and u.condition == "C1")]
    us.append(D.failed_placeholder("rq1", "rq1", 2001, "C1", None, 20, phase=T))
    p = RP.primary(us)
    assert p["rq1"]["n_achieved"] == 2 and p["rq1_failed_as_unsolved"]["n_achieved"] == 3


def test_epochs_count_t_units_only():
    us = [unit(2000, "C2", phase=T), unit(2000, "F", phase="s35-S")]
    us[0].epoch = 0
    assert RP.epochs(us) == {"C2": {"0": 1}}


def test_write_returns_the_sha_of_the_written_text(tmp_path):
    import hashlib
    sha = RP.write({"a": 1.0}, tmp_path / "x.json")
    assert sha == hashlib.sha256((tmp_path / "x.json").read_bytes()).hexdigest() == RP.canonical_hash({"a": 1.0})
    assert json.loads((tmp_path / "x.json").read_text()) == {"a": 1.0}


# ----------------------------------------------------------------------------------------------------- dry run
@pytest.fixture(scope="module")
def small_dry_run(tmp_path_factory):
    sys.path.insert(0, str(ROOT / "tools"))
    import s35_dry_run as DR
    old = DR.RQ1, DR.RQ3, DR.SP
    DR.RQ1, DR.RQ3, DR.SP = [1900, 1901], [5090], [3050]
    try:
        out_dir = tmp_path_factory.mktemp("dry")
        return DR.main(out_dir), out_dir
    finally:
        DR.RQ1, DR.RQ3, DR.SP = old


def test_dry_run_end_to_end_small(small_dry_run):
    """Spec section 9.3: the dry run on real payloads (scripted agent, non-test seeds), shortened here; pins strict."""
    out, d = small_dry_run
    assert out["primary_hash"] == out["primary_hash_recomputed"]
    full_s = json.loads((d / "s35_dry_run_full_S.json").read_text())
    assert full_s["arms_vs_c2"]["arms"]["F"]["units"] == 2
    assert full_s["sp"]["conditions"]["C2"]["units"] == 1
    prim = json.loads((d / "s35_dry_run_primary.json").read_text())
    assert prim["rq1"]["n_achieved"] == 2 and prim["rq3"]["n_achieved"] == 1


def test_primary_on_real_payloads_needs_only_numpy_and_scipy():
    """D190 (code review L8): from_result + primary on real payloads loads neither pandas nor statsmodels."""
    code = ("import sys, json, tempfile; from pathlib import Path; sys.path[:0] = ['src', 'tests', 'tools']; "
            "import s35_dry_run as DR; DR.RQ1, DR.RQ3 = [1900], [5090]; "
            "u = DR.units(); u['s35-S'] = []; DR.units = lambda: u; "
            "res = DR.run(Path(tempfile.mkdtemp())); "
            "from eem_acc.stage3.analysis import data, report; report.primary([data.from_result(r) for r in res]); "
            "bad = [m for m in ('pandas', 'statsmodels', 'patsy') if m in sys.modules]; print(bad); "
            "sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr


def test_note_text_survives_a_malformed_item():
    note = {"truth": {"kind": "drift", "pair_ids": ["get_cart:rename"]}, "items": [{"kind": "rename", "new": "client_id"}]}
    assert M.note_names_truth(note) is True                     # falls back to the raw item, never crashes


# ------------------------------------------------------------------- mutation survivors (re-run 2026-11-28)
def test_ratio_of_sums_uses_tasks_solved_by_both():
    us = []
    for s in range(2000, 2003):
        us += [unit(s, "C0", tasks=tasks_of(set(range(10)), api=6, unsolved_api=50)),
               unit(s, "C1", tasks=tasks_of(set(range(10)), api=6, unsolved_api=50)),
               unit(s, "C2", tasks=tasks_of(set(range(2, 14)), api=4, unsolved_api=50)),
               unit(s, "C3", tasks=tasks_of(set(range(10)), api=4))]
    assert S.ratio_of_sums(us)["C2/C0"]["ratio"] == pytest.approx(4 / 6)     # joint = 2-9; C2's 0-1 unsolved (50)


def test_budget_exhausted_is_max_steps_only():
    us = rq1_set(seeds=[2000])
    for u in us:
        u.tasks[19]["reason"] = "wrong_state"
    assert S.error_splits(us, ("C0",))["C0"]["budget_exhausted"] == pytest.approx(9 / 20)


def test_from_result_keeps_llm_calls():
    from eem_acc.stage3.llm.runner import Unit
    u = Unit("s35-T", "rq3", "test", "rq3", 4000, "C2", "acc")
    res = {"unit": u.to_json(), "tasks": [], "stamp": {"spec_sha256": u.spec()["sha256"]},
           "llm_calls": [{"transient": False}]}
    assert D.from_result(res).raw["llm_calls"] == [{"transient": False}]
