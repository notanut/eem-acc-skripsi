"""S3.5 spec sections 3, 4 and 6: RQ1/RQ3 assembly on synthetic units with known answers. The fixtures make C0,
C1 and C2 differ, include C2 == comparator streams, unequal solved sets with distinct API calls, a nonzero
interaction and floor edge cases (code review H1)."""
import pytest

from eem_acc.stage3.analysis import data as D
from eem_acc.stage3.analysis import endpoints as E
from eem_acc.stage3.llm.runner import Unit


def unit(rq, seed, cond, solved, api=None, window=None, n=20, status="done", kind=None, tag=""):
    api = api or {}
    tasks = [{"pos": i, "success": i in solved, "api_calls": api.get(i, 5)} for i in range(n)]
    return D.UnitData(rq, kind or rq, seed, cond, window, tag, tasks, status=status, n_tasks=n)


def by_cond(us, seed, cond):
    return next(u for u in us if u.seed == seed and u.condition == cond)


def rq1_campaign(streams=range(2000, 2014)):
    """C0 solves 0-9, C1 solves 0-11, C2 solves 0-12 (C2 - C0 = +3, C2 - C1 = +1 tasks); C3 = C0.
    API calls: C0 6, C1 5, C2 4 on every task."""
    us = []
    for s in streams:
        us += [unit("rq1", s, "C0", set(range(10)), api={i: 6 for i in range(20)}),
               unit("rq1", s, "C1", set(range(12)), api={i: 5 for i in range(20)}),
               unit("rq1", s, "C2", set(range(13)), api={i: 4 for i in range(20)}),
               unit("rq1", s, "C3", set(range(10)))]
    return us


def test_rq1_values_per_comparator():
    r = E.rq1_primary(rq1_campaign())
    h = r["hypotheses"]
    assert h["S1"]["mean_difference"] == pytest.approx(3 / 20) and h["S3"]["mean_difference"] == pytest.approx(1 / 20)
    assert h["NI1"]["tested_values"][0] == pytest.approx(4 / 20) and h["NI3"]["tested_values"][0] == pytest.approx(2 / 20)
    assert h["S1"]["tested_values"][0] == pytest.approx(3 / 20)                # S is not shifted
    assert h["E2"]["mean_difference"] == pytest.approx(-2.0) and h["E4"]["mean_difference"] == pytest.approx(-1.0)
    assert h["E2"]["tested_values"][0] == pytest.approx(2.0)                   # H1: C2 fewer calls -> positive
    assert set(r["graph"]["rejected"]) == {"NI1", "S1", "E2", "NI3", "S3", "E4"}
    assert r["cells"]["C1"]["solved"] == 14 * 12


def test_c2_equal_to_comparator_ni_rejects_superiority_does_not():
    us = []
    for s in range(2000, 2014):
        us += [unit("rq1", s, c, set(range(10))) for c in ("C0", "C1", "C2", "C3")]
    r = E.rq1_primary(us)
    h = r["hypotheses"]
    assert h["NI1"]["p"] == pytest.approx(2 ** -14) and h["NI3"]["p"] == pytest.approx(2 ** -14)
    assert h["S1"]["p"] == 1.0 and h["S1"]["untied"] == 0
    assert set(r["graph"]["rejected"]) == {"NI1", "NI3"}


def test_e2_uses_only_jointly_solved_tasks():
    # C0 solves 0-9 with 6 calls; C2 solves 5-14 with 3 calls on 5-9 and 50 calls on 10-14 (only C2 solved those)
    us = []
    for s in range(2000, 2003):
        us += [unit("rq1", s, "C0", set(range(10)), api={i: 6 for i in range(20)}),
               unit("rq1", s, "C1", set(range(10))), unit("rq1", s, "C3", set(range(10))),
               unit("rq1", s, "C2", set(range(5, 15)), api={**{i: 3 for i in range(5, 10)},
                                                           **{i: 50 for i in range(10, 15)}})]
    h = E.rq1_primary(us)["hypotheses"]["E2"]
    assert h["mean_difference"] == pytest.approx(3 - 6)                       # joint = 5-9 only


def test_one_task_worse_everywhere_is_all_ties():
    us = []
    for s in range(2000, 2014):
        us += [unit("rq1", s, c, set(range(10))) for c in ("C0", "C1", "C3")]
        us.append(unit("rq1", s, "C2", set(range(9))))
    r = E.rq1_primary(us)
    assert r["hypotheses"]["NI1"]["untied"] == 0 and r["hypotheses"]["NI1"]["p"] == 1.0
    assert r["hypotheses"]["NI1"]["inconclusive"] is True
    assert r["graph"]["rejected"] == []


def test_ni_tie_is_exact_zero_from_integers():
    # review M7: (0.35 - 0.40) + 0.05 is not exactly 0 in floating point; the integer path is
    assert (0.35 - 0.40) + 0.05 != 0.0
    a, b = unit("rq1", 2000, "C2", set(range(7))), unit("rq1", 2000, "C0", set(range(8)))
    assert E.ni_value(a, b) == 0.0 and E.success_diff(a, b) == pytest.approx(-0.05)


def test_paired_units_must_have_equal_task_counts():
    with pytest.raises(ValueError):
        E.success_diff(unit("rq1", 2000, "C2", {1}, n=20), unit("rq1", 2000, "C0", {1}, n=19))


def test_completeness_and_d186_sensitivity():
    us = [u for u in rq1_campaign() if not (u.seed == 2003 and u.condition == "C3")]
    us = [u for u in us if not (u.seed == 2005 and u.condition == "C1")]
    us.append(D.failed_placeholder("rq1", "rq1", 2005, "C1", None, 20))
    r = E.rq1_primary(us)
    assert r["n_achieved"] == 12
    assert r["dropped_incomplete"] == {2003: {"C3": "missing"}, 2005: {"C1": "failed"}}
    s = E.rq1_primary(us, include_failed=True)                 # D186: the failed C1 counts as 0 solved
    assert s["n_achieved"] == 13 and s["dropped_incomplete"] == {2003: {"C3": "missing"}}
    i = s["hypotheses"]["S3"]["seeds"].index(2005)
    assert s["hypotheses"]["S3"]["differences"][i] == pytest.approx(13 / 20)
    assert s["hypotheses"]["NI3"]["tested_values"][i] == pytest.approx(14 / 20)
    assert 2005 in s["hypotheses"]["E4"]["left_out_no_difference"]
    assert 2005 not in s["hypotheses"]["E2"]["left_out_no_difference"]


def test_e2_left_out_without_jointly_solved_tasks():
    us = rq1_campaign(streams=range(2000, 2004))
    for t in by_cond(us, 2001, "C0").tasks:
        t["success"] = False
    r = E.rq1_primary(us)
    assert r["hypotheses"]["E2"]["left_out_no_difference"] == [2001]
    assert r["hypotheses"]["E2"]["n"] == 3 and r["hypotheses"]["NI1"]["n"] == 4


def test_duplicate_units_refused_tagged_and_other_lengths_ignored():
    with pytest.raises(ValueError):
        E.rq1_primary(rq1_campaign(streams=[2000]) + [unit("rq1", 2000, "C0", set())])
    us = rq1_campaign(streams=[2000]) + [unit("rq1", 2000, "C2", set(), tag="cfg1"),
                                         unit("rq1", 2000, "C2", set(), n=15)]
    assert E.rq1_primary(us)["n_achieved"] == 1


def test_phase_filter():
    us = rq1_campaign(streams=[2000])
    for u in us:
        u.phase = "s35-T"
    other = unit("rq1", 2000, "C0", set())
    other.phase = "s34-B-r1"
    assert E.rq1_primary(us + [other], phase="s35-T")["n_achieved"] == 1


def rq3_campaign(acc_c2=12, acc_c0=10, none_c2=8, none_c0=7, seeds=range(4000, 4010)):
    us = []
    for s in seeds:
        us += [unit("rq3", s, "C2", set(range(acc_c2)), window="acc"),
               unit("rq3", s, "C0", set(range(acc_c0)), window="acc"),
               unit("rq3", s, "C2", set(range(none_c2)), window="none"),
               unit("rq3", s, "C0", set(range(none_c0)), window="none")]
    return us


def test_rq3_graph_and_interaction():
    r = E.rq3_primary(rq3_campaign())
    h = r["hypotheses"]
    assert h["NI-R1"]["p"] == pytest.approx(2 ** -10) and h["R2"]["mean_difference"] == pytest.approx(4 / 20)
    assert h["S-R1"]["mean_difference"] == pytest.approx(2 / 20)
    assert h["NI-R1"]["tested_values"][0] == pytest.approx(3 / 20)
    assert set(r["graph"]["rejected"]) == {"NI-R1", "S-R1", "R2"}
    # (12 - 10) - (8 - 7) = +1 task per seed
    assert r["interaction"]["mean"] == pytest.approx(1 / 20) and r["interaction"]["uninformative_floor"] is False


def test_rq3_r2_is_within_c2_and_s_r1_unshifted():
    r = E.rq3_primary(rq3_campaign(acc_c2=10, acc_c0=10, none_c2=10, none_c0=2))
    h = r["hypotheses"]
    assert h["R2"]["p"] == 1.0 and h["S-R1"]["p"] == 1.0 and h["NI-R1"]["p"] == pytest.approx(2 ** -10)
    assert r["graph"]["rejected"] == ["NI-R1"]


@pytest.mark.parametrize("c2,c0,floor", [(3, 2, True), (3, 4, False), (4, 2, False), (2, 4, False),
                                         (3, 3, True)])
def test_rq3_floor_rule(c2, c0, floor):
    # floor: mean success under no window < 0.2 (4 of 20) for BOTH C0 and C2; exactly 0.2 is not below
    r = E.rq3_primary(rq3_campaign(none_c2=c2, none_c0=c0))
    assert r["interaction"]["uninformative_floor"] is floor


def test_rq3_short_n_is_inconclusive():
    r = E.rq3_primary(rq3_campaign(seeds=range(4000, 4006)))      # 6 untied seeds: 2^-6 > alpha/2
    assert r["n_achieved"] == 6 and r["hypotheses"]["NI-R1"]["inconclusive"] is True
    assert r["graph"]["rejected"] == []


def test_rq3_cell_counts():
    us = rq3_campaign(seeds=[4000])
    t = by_cond([u for u in us if u.window == "none"], 4000, "C0").tasks
    t[19]["unfinished_at_t_max"] = True
    t[18]["never_started"] = True
    c = E.rq3_primary(us)["cells"]["C0+none"]
    assert c == {"tasks": 20, "solved": 7, "unfinished_at_t_max": 1, "never_started": 1}


def test_empty_campaign():
    r = E.rq1_primary([])
    assert r["n_achieved"] == 0 and r["hypotheses"]["NI1"]["p"] is None and r["graph"]["rejected"] == []


# ------------------------------------------------------------------------------------------------ from_result
def real_result(u: Unit, tasks: list[dict], epoch=0, sha=None):
    return {"unit": u.to_json(), "tasks": tasks,
            "stamp": {"spec_sha256": sha if sha is not None else u.spec()["sha256"], "llm": {"platform_epoch": epoch}},
            "run": {"payload_sha256": "x", "wall_seconds": 1.0}}


def test_from_result_real_layout():
    u = Unit("s35-T", "rq3", "test", "rq3", 4001, "C2", "acc", params=(("k", 5),))
    tasks = [{"task": 3, "success": True, "api_calls": 3}, {"task": 0, "success": False,
                                                           "unfinished_at_t_max": True}]
    d = D.from_result(real_result(u, tasks, epoch=2))
    assert d.cell == "C2+acc" and d.success_rate() == 0.5 and d.params == {"k": 5} and d.phase == "s35-T"
    assert d.epoch == 2                                                  # from stamp.llm (code review M3)
    assert [t["pos"] for t in d.tasks] == [3, 0]                         # position from `task` (L11)


def test_from_result_refuses_a_foreign_stream():
    u = Unit("s35-T", "rq1", "test", "rq1", 2000, "C0")
    with pytest.raises(ValueError):
        D.from_result(real_result(u, [], sha="0" * 64))


def test_from_result_failed_and_not_done():
    u = Unit("s35-T", "rq1", "test", "rq1", 2000, "C1")
    f = D.from_result({"unit": u.to_json(), "status": "failed"})
    assert f.status == "failed" and len(f.tasks) == 20 and f.success_rate() == 0.0
    nd = D.from_result({"unit": u.to_json(), "status": "not_done"})
    assert nd.status == "not_done" and nd.tasks == []
    by = D.index([nd], "rq1", "rq1")
    assert D.complete(by, ("C1",))[1] == {2000: {"C1": "not_done"}}
