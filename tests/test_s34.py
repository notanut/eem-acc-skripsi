"""S3.4 automatic chain (docs/s34_spec.md r2): rules, gate, freeze, chain end to end on a scripted fake."""
import dataclasses
import json
from pathlib import Path

import pytest

from eem_acc.stage3 import s34 as S
from eem_acc.stage3.llm import runner as R
from eem_acc.stage3.llm.t4 import Sections


# ------------------------------------------------------------------------------------------------ helpers
def m(seed=1000, cond="C0", tag="", n=20, solved=10, api=5, tok=1000, first=100, fb=0, actions=None, trunc=0,
      cost=200.0, unfinished=0, rq="rq1", status="done", window=None):
    tasks = [{"pos": i, "success": i < solved, "api_calls": api, "tok_in": tok, "unfinished": i < unfinished}
             for i in range(n)]
    return {"unit_id": f"u-{seed}-{cond}-{tag}-{window}", "rq": rq, "condition": cond, "window": window, "tag": tag,
            "seed": seed, "tasks": tasks, "first_actions": first, "fallbacks": fb,
            "action_replies": actions if actions is not None else first + fb, "truncated_actions": trunc,
            "reflections": 10, "reflections_at_cap": 0, "format_errors": 0, "overflows": 0, "llm_calls": 150,
            "cost_seconds": cost, "status": status}


# ------------------------------------------------------------------------------------------------- rules
def test_choose_wording_ties():
    ms = [m(s, "C0", w, solved=k) for s in (1000, 1001) for w, k in (("W1", 8), ("W2", 10), ("W3", 9))]
    assert S.choose_wording(ms)["wording"] == "W2"
    tie = [m(s, "C0", w, solved=10, api=a) for s in (1000, 1001) for w, a in (("W1", 6), ("W2", 5), ("W3", 5))]
    d = S.choose_wording(tie)
    assert d["wording"] == "W2" and d["decided_by"] == "default"          # W2 = W3 on API calls; W1 not tied
    tie2 = [m(s, "C0", w, solved=10, api=5) for s in (1000, 1001) for w in ("W1", "W2", "W3")]
    assert S.choose_wording(tie2)["wording"] == "W1"
    none = [m(s, "C0", w, solved=0) for s in (1000, 1001) for w in ("W1", "W2", "W3")]
    assert S.choose_wording(none) == {"wording": "W1", "solved": {"W1": 0, "W2": 0, "W3": 0}, "decided_by": "default"}
    by_api = [m(s, "C0", w, solved=10, api=a) for s in (1000, 1001) for w, a in (("W1", 6), ("W2", 4), ("W3", 5))]
    assert S.choose_wording(by_api) == {"wording": "W2", "solved": {"W1": 20, "W2": 20, "W3": 20},
                                        "decided_by": "api_calls"}


@pytest.mark.parametrize("fb0,fb2,tr0,tr2,g,c", [
    (5, 5, 0, 0, False, False),          # 5.0% pooled: not above
    (6, 5, 0, 0, True, False),           # 5.5% pooled
    (0, 3, 0, 0, False, False),          # 3 pp exactly: not above
    (12, 7, 0, 0, False, False),         # C0 12/300 = 4%, C2 7/100 = 7%: 3 pp exactly, pooled 4.75% (floats say
                                         # 0.07 - 0.04 > 0.03; exact arithmetic says not above, review M1)
    (0, 4, 0, 0, True, False),           # 4 pp
    (0, 0, 1, 0, False, False),          # 1% truncation exactly: not above
    (0, 0, 0, 2, False, True)])
def test_switch_boundaries(fb0, fb2, tr0, tr2, g, c):
    n0 = 300 if (fb0, fb2) == (12, 7) else 100
    sw = S.switch_check([m(first=n0, fb=fb0, actions=100, trunc=tr0)], [m(cond="C2", first=100, fb=fb2, actions=100,
                                                                           trunc=tr2)])
    if (fb0, fb2) == (12, 7):
        assert 7 / 100 - 12 / 300 > 0.03                          # the float trap this case guards against
    assert (sw["grammar_always"], sw["cap640"]) == (g, c)


@pytest.mark.parametrize("solved,verdict", [(19, "hard"), (20, "ok"), (80, "ok"), (81, "easy")])
def test_acceptance_boundaries(solved, verdict):
    ms = [m(1002 + i, solved=min(20, max(0, solved - 20 * i))) for i in range(5)]
    assert S.acceptance(ms)["verdict"] == verdict


def test_ladder_steps_and_bounds():
    nat = 0.225806
    s1, nxt, sk = S.ladder_step(S.Settings(), "hard", 0, nat)
    assert s1.long_share == round(nat - 0.1, 6) and nxt == 1 and sk == []
    s2, nxt, _ = S.ladder_step(s1, "hard", 1, nat)
    assert s2.n_distractors == 1 and nxt == 2
    s3, nxt, _ = S.ladder_step(s2, "hard", 2, nat)
    assert s3.n_distractors == 0 and nxt == 3
    assert S.ladder_step(s3, "hard", 3, nat)[0] is None
    edge, nxt, sk = S.ladder_step(S.Settings(long_share=0.05), "hard", 0, nat)     # 0.05 - 0.1 < 0: skipped
    assert sk == [0] and edge.n_distractors == 1 and edge.long_share == 0.05 and nxt == 2
    e1, _, _ = S.ladder_step(S.Settings(), "easy", 0, nat)
    assert e1.long_share == round(nat + 0.1, 6)


def test_select_config_rule():
    def sweep(spec):        # cfg -> (solved per stream, api, tok)
        return [m(1007 + i, "C2", c, solved=s, api=a, tok=t) for c, (s, a, t) in spec.items() for i in range(4)]
    d = S.select_config(sweep({"cfg1": (12, 5, 1000), "cfg2": (10, 5, 1000), "cfg3": (15, 6, 1000)}), 4)
    assert d["config"] == "cfg3" and d["decided_by"] == "success"        # cfg1 is 12 tasks behind (> 4)
    d = S.select_config(sweep({"cfg1": (15, 5, 1000), "cfg3": (14, 4, 1000), "cfg5": (12, 3, 900)}), 4)
    assert d["config"] == "cfg3" and d["decided_by"] == "api_calls"     # cfg5 12 behind; cfg3 within 4 and cheaper
    d = S.select_config(sweep({"cfg1": (15, 5, 1000), "cfg3": (15, 5, 900)}), 4)
    assert d["config"] == "cfg3" and d["decided_by"] == "prompt_tokens"
    d = S.select_config(sweep({"cfg2": (15, 5, 1000), "cfg6": (15, 5, 1000)}), 4)
    assert d["config"] == "cfg2" and d["decided_by"] == "default"
    d = S.select_config(sweep({"cfg4": (15, 5, 1000), "cfg3": (15, 5, 1000)}), 4)
    assert d["config"] == "cfg3"                                          # smaller k first
    d = S.select_config(sweep({"cfg5": (15, 5, 1000), "cfg4": (15, 5, 1000)}), 4)
    assert d["config"] == "cfg5"                                          # same k: theta closest to 0.65
    d = S.select_config(sweep({"cfg1": (15, 5, 1000), "cfg3": (15, 5, 1000)}), 4)
    assert d["config"] == "cfg1"                                          # 0.5 and 0.8 equally close: lower index


def test_wiring_check():
    ok = [m(5001, "C2", rq="rq3", window="acc"), m(5001, "C0", rq="rq3", window="none")]
    assert S.wiring_check(ok, True, 0)["ok"]
    assert not S.wiring_check([m(5001, "C2", rq="rq3", unfinished=1), ok[1]], True, 0)["ok"]
    w = S.wiring_check(ok, False, 0)
    assert not w["ok"] and not w["sane"]
    assert not S.wiring_check(ok, True, 2)["sane"]


# --------------------------------------------------------------------------------------------------- gate
COSTS = S.Costs(c0=40.0, c2=80.0, rq3=90.0, setup_s=600.0, gap_s=10.0, unit_hours_per_session=3.0)


def test_gate_fits_without_cuts():
    g = S.budget_gate(1.0, dataclasses.replace(COSTS, c0=10, c2=20, rq3=20), selection_done=False)
    assert g["fits"] and g["cuts"] == [] and g["n_tasks"] == 20


def test_gate_cuts_in_order_skips_no_saving_and_reduces_tasks():
    full = S.projection(COSTS, S.workload(20, frozenset(), 24, True, 4))["total"]     # no cuts
    g = S.budget_gate(60.0 - full + 0.01, COSTS)                         # just over: the first saving cut suffices
    assert g["cuts"] == [0] and g["fits"]
    g = S.budget_gate(40.0, COSTS, selection_done=True, d60_reserve=False)   # cut 11 saves nothing once F is done
    assert 11 in g["skipped_cuts"] and 11 not in g["cuts"] and list(range(11)) == g["cuts"]
    hard = S.budget_gate(40.0, COSTS)
    assert hard["cuts"] == list(range(12)) and hard["n_tasks"] in (19, 18, 17, 16)
    over = S.budget_gate(58.0, COSTS)
    assert not over["fits"] and over["n_tasks"] == 16 and "D57" in over["pause"]
    only17 = S.budget_gate(40.0, COSTS, exposure_ok=lambda n: n != 19 and n != 18)
    assert only17["n_tasks"] in (17, 16)


def test_gate_keeps_earlier_cuts_and_task_count():
    g = S.budget_gate(0.0, COSTS, cuts=(0, 1), n_tasks=18, selection_done=True, d60_reserve=False)
    assert {0, 1} <= set(g["cuts"]) and g["n_tasks"] <= 18


def test_workload_cut_effects():
    w = {name: (u, t) for name, u, t, _ in S.workload(20, frozenset({0, 2, 7, 10}), 24, True, 4)}
    assert "rq3_private" not in w and "A1" not in w and "F" not in w
    assert w["R"] == (3, 15) and w["C2-OU"] == (5, 20) and w["rq1_C2"] == (14, 20)
    assert w["d60_c0"] == (5, 20) and w["selection"] == (24, 20)


def test_prefix_exposure_ok_on_planned_streams():
    for n in (19, 18, 17, 16):
        assert S.prefix_exposure_ok(n, S.Settings())


# ---------------------------------------------------------------------------------------------- the chain
class Fake:
    """Scripted platform: metrics are a pure function of the unit (its params and stream knobs)."""

    def __init__(self, root, c0=lambda s: 10, fallback=lambda s, u: 0, trunc=lambda s, u: 0, unfinished=0,
                 cfg_solved=None, crash_at=None, used=1.0, c2_cost=80.0):
        self.root, self.c0, self.fallback, self.trunc = Path(root), c0, fallback, trunc
        self.unfinished, self.cfg_solved = unfinished, cfg_solved or {}
        self.crash_at, self.calls, self.phases, self.used, self.c2_cost = crash_at, 0, [], used, c2_cost

    def settings_of(self, u):
        p, st = dict(u.params), dict(u.stream)
        return S.Settings(wording=p.get("docs_wording", "W1"), grammar_always=p.get("grammar_always", False),
                          action_max_tokens=p.get("action_max_tokens", 320),
                          n_distractors=st.get("n_distractors", 2), long_share=st.get("long_share"))

    def metric(self, u):
        s = self.settings_of(u)
        if u.rq == "rq3":
            return m(u.seed, u.condition, rq="rq3", window=u.window, unfinished=self.unfinished, cost=1800)
        if u.condition == "C0":
            solved = self.c0(s) + (1 if u.tag == "W2" else 0)
        else:
            solved = self.cfg_solved.get(u.tag, 12)
        return m(u.seed, u.condition, u.tag, n=u.n_tasks, solved=min(u.n_tasks, solved), first=100,
                 fb=self.fallback(s, u), trunc=self.trunc(s, u), cost=40.0 * u.n_tasks if u.condition == "C0"
                 else self.c2_cost * u.n_tasks / 20)

    def run_units(self, phase, units, meta):
        self.calls += 1
        if self.crash_at and self.calls == self.crash_at:
            raise KeyboardInterrupt("runtime lost")
        if not (self.root / "plans" / f"{phase}.json").exists():          # as the notebook: a plan is written once
            R.write_plan(self.root, phase, units, 1, frozen_commit="c" * 40, meta=meta)
        else:
            assert [u["unit_id"] if "unit_id" in u else R.Unit.from_json(u).unit_id
                    for u in R.read_plan(self.root, phase)["units"]] == [u.unit_id for u in units]
        self.phases.append(phase)
        return [{"unit": u.to_json(), "m": dict(self.metric(u), unit_id=u.unit_id)} for u in units]

    def chain(self):
        sec = Sections(self.root / "chain.json", stamp={"commit": "c" * 40, "nb_version": "v"})
        return S.Chain(sec, self.root, self.run_units, lambda r: r["m"], lambda rs: (True, 0), lambda: self.used,
                       lambda: {"setup_s": 300.0, "gap_s": 10.0, "unit_hours_per_session": 3.0},
                       freeze_extra=lambda s: {"template_set_sha256": S.template_set_sha()},
                       exposure_ok=lambda n, s: True, log=lambda msg: None)


def steps(f):
    return [(e["step"], e.get("round")) for e in json.loads((f.root / "chain.json").read_text())["log"]]


def test_chain_happy_path(tmp_path):
    f = Fake(tmp_path, cfg_solved={"cfg5": 16})
    assert f.chain().run() == "done"
    assert [s for s, _ in steps(f)] == ["A", "B", "C", "D", "E", "F", "D", "G"]
    data = json.loads((tmp_path / "chain.json").read_text())
    assert data["selection"]["config"] == "cfg5" and data["log"][0]["decision"]["wording"] == "W2"
    freeze = json.loads((tmp_path / "s34" / "freeze.json").read_text())
    assert freeze["settings"]["wording"] == "W2" and freeze["commit"] == "c" * 40
    assert R.read_plan(tmp_path, "s34-F-r1")["meta"] == {"freeze_sha": data["freeze_sha"]}   # review M6
    f_units = [R.Unit.from_json(u) for u in R.read_plan(tmp_path, "s34-F-r1")["units"]]
    assert len(f_units) == 24 and all(dict(u.params)["docs_wording"] == "W2" for u in f_units)
    gates = [e["decision"] for e in data["log"] if e["step"] == "D"]          # derived from the log (review L2)
    assert len(gates) == 2 and gates[-1]["fits"] and "gates" not in data


def test_switch_fires_reruns_wording_and_pilot_with_c2(tmp_path):
    f = Fake(tmp_path, fallback=lambda s, u: 0 if s.grammar_always else 10)
    f.chain().run()
    st = steps(f)
    assert st[:5] == [("A", 1), ("B", 1), ("A", 2), ("B", 2), ("C", 1)]
    b2 = [R.Unit.from_json(u) for u in R.read_plan(tmp_path, "s34-B-r2")["units"]]
    assert {u.condition for u in b2} == {"C0", "C2"} and all(dict(u.params)["grammar_always"] for u in b2)
    log = json.loads((tmp_path / "chain.json").read_text())["log"]
    assert "acceptance" not in log[1]["decision"] and log[3]["decision"]["acceptance"]["verdict"] == "ok"


def test_second_switch_after_a_ladder_step_is_still_checked(tmp_path):            # review H1
    f = Fake(tmp_path, c0=lambda s: 2 if s.long_share is None else 10,
             trunc=lambda s, u: 3 if (s.long_share is not None and s.action_max_tokens == 320) else 0)
    f.chain().run()
    log = json.loads((tmp_path / "chain.json").read_text())["log"]
    bs = [e for e in log if e["step"] == "B"]
    assert bs[0]["decision"].get("ladder") and bs[1]["decision"]["fired"] == ["cap640"]
    assert bs[2]["decision"]["acceptance"]["verdict"] == "ok"
    freeze = json.loads((tmp_path / "s34" / "freeze.json").read_text())
    assert freeze["switches"]["action_max_tokens"] == 640 and freeze["knobs"]["long_share"] is not None


def test_ladder_exhausted_pauses(tmp_path):
    f = Fake(tmp_path, c0=lambda s: 1)
    with pytest.raises(S.ChainPause, match="after 3 ladder steps"):
        f.chain().run()
    with pytest.raises(S.ChainPause):                    # a later session stays paused
        f.chain().run()
    assert [s for s, _ in steps(f)] == ["A", "B"] * 4


def test_ladder_overshoot_pauses(tmp_path):
    f = Fake(tmp_path, c0=lambda s: 1 if s.long_share is None else 19)
    with pytest.raises(S.ChainPause, match="overshot"):
        f.chain().run()


def test_rq3_unfinished_pauses(tmp_path):
    f = Fake(tmp_path, unfinished=1)
    with pytest.raises(S.ChainPause, match="unfinished at T_max"):
        f.chain().run()


def test_d60_round_and_reported_acceptance(tmp_path):
    f = Fake(tmp_path, cfg_solved={"cfg2": 15},
             fallback=lambda s, u: 0 if (s.grammar_always or u.phase.startswith("s34-B") or u.condition == "C0")
             else 12)
    assert f.chain().run() == "done"
    st = [s for s, _ in steps(f)]
    assert st == ["A", "B", "C", "D", "E", "F", "H", "D", "G"]
    data = json.loads((tmp_path / "chain.json").read_text())
    d60 = [e["decision"] for e in data["log"] if e["step"] == "H"]
    assert d60[0]["acceptance"]["verdict"] == "ok" and not d60[0]["fired"]
    h = [R.Unit.from_json(u) for u in R.read_plan(tmp_path, "s34-H-r1")["units"]]
    assert {(u.rq, u.condition, u.tag) for u in h} == {("rq1", "C2", "cfg2"), ("rq1", "C0", ""), ("rq3", "C2", ""),
                                                      ("rq3", "C0", "")}
    assert all(dict(u.params).get("grammar_always") for u in h)
    freeze = json.loads((tmp_path / "s34" / "freeze.json").read_text())
    assert freeze["switches"]["grammar_always"] is False                  # the freeze is immutable (review H2)


def test_resume_after_lost_runtime_gives_the_same_chain(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    fa = Fake(a, fallback=lambda s, u: 0 if s.grammar_always else 10)
    fa.chain().run()
    fb = Fake(b, fallback=lambda s, u: 0 if s.grammar_always else 10, crash_at=4)
    with pytest.raises(KeyboardInterrupt):
        fb.chain().run()
    fb.crash_at = None
    assert fb.chain().run() == "done"
    la = json.loads((a / "chain.json").read_text())["log"]
    lb = json.loads((b / "chain.json").read_text())["log"]
    assert [(e["step"], e.get("decision")) for e in la] == [(e["step"], e.get("decision")) for e in lb]


def test_recomputed_decision_that_differs_halts(tmp_path):
    f = Fake(tmp_path, crash_at=3)
    with pytest.raises(KeyboardInterrupt):
        f.chain().run()                                  # A and B recorded, C interrupted
    f.crash_at, f.c0 = None, (lambda s: 11)              # stored results "change"
    with pytest.raises(S.ChainHalt, match="differs"):
        f.chain().run()


def test_failed_unit_pauses_for_a_code_fix(tmp_path):
    f = Fake(tmp_path)
    orig = f.metric
    f.metric = lambda u: dict(orig(u), status="failed") if u.condition == "C2" and u.phase == "s34-B-r1" else orig(u)
    with pytest.raises(S.ChainPause, match="units not done"):
        f.chain().run()


def test_budget_pause_and_cuts_in_chain(tmp_path):
    f = Fake(tmp_path, used=59.5)
    with pytest.raises(S.ChainPause, match="D57"):
        f.chain().run()
    g = [e["decision"] for e in json.loads((tmp_path / "chain.json").read_text())["log"] if e["step"] == "D"][0]
    assert g["cuts"] == list(range(12)) and g["n_tasks"] == 16


def test_write_once(tmp_path):
    p = tmp_path / "x.json"
    sha = S.write_once(p, {"a": 1})
    assert S.write_once(p, {"a": 1}) == sha
    with pytest.raises(S.ChainHalt, match="immutable"):
        S.write_once(p, {"a": 2})


# ------------------------------------------------------------------------------------------- units, knobs
def test_unit_stream_knobs_in_identity_and_spec():
    plain = R.Unit("s34-B-r1", "rq1", "dev", "rq1", 1002, "C0")
    assert "stream" not in plain.to_json() and plain.unit_id == "s34-B-r1-rq1-s1002-C0"
    k = dataclasses.replace(plain, stream=(("n_distractors", 1),))
    assert k.unit_id.startswith("s34-B-r1-rq1-s1002-C0-x") and R.Unit.from_json(k.to_json()) == k
    assert k.spec()["n_distractors"] == 1 and plain.spec()["n_distractors"] == 2
    r3 = R.Unit("s34-C-r1", "rq3", "dev", "rq3", 5001, "C2", "acc", stream=(("long_share", 0.3),))
    assert r3.spec()["long_share"] == 0.3 and r3.spec()["kind"] == "rq3"


def test_audit_uses_the_units_knobs():
    from eem_acc.stage3.llm.audit import audit_result
    u = R.Unit("s34-B-r1", "rq1", "dev", "rq1", 1002, "C2", stream=(("n_distractors", 0),))
    assert audit_result({"unit": u.to_json(), "final_notes": [], "tasks": [], "trajectories": []}) == []


def test_ledger_profile(tmp_path):
    from eem_acc.stage3.llm.ledger import Ledger, profile
    t = [1000.0]
    led = Ledger(tmp_path, "s34-1", "rt1", clock=lambda: t[0])
    led.start(model_load_seconds=20, vm_boot=900.0)                      # VM booted 100 s before the first cell
    t[0] = 1500.0
    led.event("unit", unit="s34-A-r1-x", wall_seconds=300.0)              # unit 1200-1500: setup = 300 s
    t[0] = 1900.0
    led.event("unit", unit="s34-A-r1-y", wall_seconds=350.0)              # unit 1550-1900: gap 50 s
    t[0] = 1960.0
    led.heartbeat()                                                       # 60 s after the last unit: gap
    led.event("unit", unit="smoke-z", wall_seconds=10.0)                  # not S3.4: ignored
    p = profile(tmp_path, "s34-")
    assert p["runtimes"] == 1 and p["setup_s"] == 300.0
    assert p["gap_s"] == (760.0 - 650.0) / 2 and abs(p["unit_hours_per_session"] - 650 / 3600) < 1e-9
    assert profile(tmp_path, "none-")["runtimes"] == 0



def test_resume_after_the_selection_recomputes_it(tmp_path):
    f = Fake(tmp_path, cfg_solved={"cfg4": 17})
    with pytest.raises(KeyboardInterrupt):
        f.crash_at = 5                     # calls: A, B, C, F, then the pilot reload inside F (lost before F's record)
        f.chain().run()
    f.crash_at = None
    assert f.chain().run() == "done"
    data = json.loads((tmp_path / "chain.json").read_text())
    assert data["selection"]["config"] == "cfg4"
    assert [s for s, _ in steps(f)] == ["A", "B", "C", "D", "E", "F", "D", "G"]
    f.cfg_solved = {"cfg4": 10}                           # a later session sees different stored results: halt
    with pytest.raises(S.ChainHalt):
        f.chain().verify_log()



# ------------------------------------------------------------------------------------- code review additions
def test_unit_metrics_from_records():
    u = R.Unit("s34-B-r1", "rq1", "dev", "rq1", 1002, "C2", params=(("docs_wording", "W1"),))
    recs = [{"kind": "action", "grammar": None, "finish_reason": "stop", "n_out": 40},
            {"kind": "action", "grammar": "envelope_cap300", "finish_reason": "length", "n_out": 320},
            {"kind": "action", "grammar": None, "finish_reason": "stop", "n_out": 30},
            {"kind": "reflection", "grammar": None, "finish_reason": "length", "n_out": 60},
            {"kind": "embed"}]
    res = {"unit": u.to_json(), "tasks": [{"task": 0, "success": True, "api_calls": 3, "tok_in": 900,
                                           "format_errors": 1, "reason": "success"},
                                          {"task": 1, "success": False, "api_calls": 4, "tok_in": 800,
                                           "reason": "context_overflow"}],
           "run": {"wall_seconds": 100.0, "calls_fresh": 2, "calls_replayed": 2}}
    mt = S.unit_metrics(res, recs)
    assert (mt["first_actions"], mt["fallbacks"], mt["action_replies"], mt["truncated_actions"]) == (2, 1, 3, 1)
    assert mt["reflections_at_cap"] == 1 and mt["llm_calls"] == 4 and mt["overflows"] == 1 and mt["format_errors"] == 1
    assert mt["cost_seconds"] == 200.0                               # resumed: scaled to all calls
    ga = dataclasses.replace(u, params=(("docs_wording", "W1"), ("grammar_always", True)))
    mt = S.unit_metrics({**res, "unit": ga.to_json()}, recs)
    assert mt["first_actions"] == 3 and mt["fallbacks"] == 0         # grammar always: no fallbacks (review L4)
    assert S.unit_metrics({**res, "run": {"wall_seconds": 1.0, "calls_fresh": 0, "calls_replayed": 9}},
                          recs)["cost_seconds"] is None              # fully replayed: no cost
    r3 = R.Unit("s34-C-r1", "rq3", "dev", "rq3", 5001, "C2", "acc")
    mt = S.unit_metrics({"unit": r3.to_json(), "tasks": [{"task": 0, "success": False, "unfinished_at_t_max": True}],
                         "run": {"wall_seconds": 10.0, "calls_fresh": 5}}, [])
    assert mt["tasks"][0]["unfinished"] and mt["window"] == "acc"
    assert S.condition_summary([dict(mt, cost_seconds=None), m(cost=100.0)])["seconds_per_task"] == 5.0


def test_both_switches_in_successive_rounds(tmp_path):
    f = Fake(tmp_path, fallback=lambda s, u: 0 if s.grammar_always else 10,
             trunc=lambda s, u: 3 if (s.grammar_always and s.action_max_tokens == 320) else 0)
    f.chain().run()
    bs = [e for e in json.loads((tmp_path / "chain.json").read_text())["log"] if e["step"] == "B"]
    assert [b["decision"]["fired"] for b in bs] == [["grammar_always"], ["cap640"], []]
    assert bs[2]["decision"]["acceptance"]["verdict"] == "ok"


def test_two_d60_rounds_then_gate(tmp_path):
    f = Fake(tmp_path, cfg_solved={"cfg2": 15},
             fallback=lambda s, u: 12 if (u.phase.startswith("s34-F") and not s.grammar_always) else 0,
             trunc=lambda s, u: 3 if (u.phase == "s34-H-r1" and u.condition == "C2") else 0)
    orig = f.metric                                       # grammar always is slower: H re-measures C2's cost
    f.metric = lambda u: dict(orig(u), cost_seconds=orig(u)["cost_seconds"] * 1.36) \
        if (u.phase.startswith("s34-H") and u.condition == "C2" and u.rq == "rq1") else orig(u)
    assert f.chain().run() == "done"
    assert [s for s, _ in steps(f)] == ["A", "B", "C", "D", "E", "F", "H", "H", "D", "G"]
    log = json.loads((tmp_path / "chain.json").read_text())["log"]
    hs = [e for e in log if e["step"] == "H"]
    assert hs[0]["decision"]["fired"] == ["cap640"] and hs[1]["decision"]["fired"] == []
    final_gate = [e for e in log if e["step"] == "D"][-1]["decision"]                       # review M4
    assert final_gate["costs"]["c2"] == hs[1]["decision"]["c2_seconds_per_task"] != \
        [e for e in log if e["step"] == "D"][0]["decision"]["costs"]["c2"]
    h2 = [R.Unit.from_json(u) for u in R.read_plan(tmp_path, "s34-H-r2")["units"]]
    assert all(dict(u.params).get("action_max_tokens") == 640 for u in h2)


def test_h_wiring_pause(tmp_path):
    f = Fake(tmp_path, fallback=lambda s, u: 12 if (u.phase.startswith("s34-F") and not s.grammar_always) else 0)
    orig = f.metric
    f.metric = lambda u: dict(orig(u), tasks=[dict(t, unfinished=True) for t in orig(u)["tasks"]]) \
        if (u.rq == "rq3" and u.phase.startswith("s34-H")) else orig(u)
    with pytest.raises(S.ChainPause, match="after D60"):
        f.chain().run()


def test_crash_after_freeze_before_record_still_records_e(tmp_path, monkeypatch):     # review L1
    f = Fake(tmp_path)
    ch = f.chain()
    real = ch.record

    def boom(entry):
        if entry["step"] == "E":
            raise KeyboardInterrupt("lost after the freeze was written")
        real(entry)
    ch.record = boom
    with pytest.raises(KeyboardInterrupt):
        ch.run()
    assert (tmp_path / "s34" / "freeze.json").exists()
    assert f.chain().run() == "done"
    assert [s for s, _ in steps(f)] == ["A", "B", "C", "D", "E", "F", "D", "G"]


def test_failed_checks_pause_c(tmp_path):
    f = Fake(tmp_path)
    sec = Sections(tmp_path / "chain.json", stamp={"commit": "c" * 40, "nb_version": "v"})
    ch = S.Chain(sec, tmp_path, f.run_units, lambda r: r["m"], lambda rs: (False, 0), lambda: 1.0,
                 lambda: {"setup_s": 300.0, "gap_s": 10.0, "unit_hours_per_session": 3.0},
                 exposure_ok=lambda n, s: True, log=lambda msg: None)
    with pytest.raises(S.ChainPause, match="wiring not sane"):
        ch.run()


def test_new_chain_version_after_a_code_fix(tmp_path):                                # review H1
    f = Fake(tmp_path)
    orig = f.metric
    f.metric = lambda u: dict(orig(u), status="failed") if u.phase == "s34-B-r1" and u.condition == "C2" else orig(u)
    with pytest.raises(S.ChainPause, match="units not done"):
        f.chain().run()
    f.metric = orig                                       # the fix
    sec2 = Sections(tmp_path / S.chain_file(2), stamp={"commit": "d" * 40, "nb_version": "v2"})
    ch2 = S.Chain(sec2, tmp_path, f.run_units, lambda r: r["m"], lambda rs: (True, 0), lambda: 1.0,
                  lambda: {"setup_s": 300.0, "gap_s": 10.0, "unit_hours_per_session": 3.0},
                  exposure_ok=lambda n, s: True, log=lambda msg: None, version=2)
    assert ch2.run() == "done"
    assert (tmp_path / "plans" / "s34v2-A-r1.json").exists() and (tmp_path / "s34" / "freeze-v2.json").exists()
    assert not (tmp_path / "s34" / "freeze.json").exists()
    assert json.loads((tmp_path / "chain.json").read_text())["log"][-1].get("pause")      # v1 kept as evidence


def test_save_once_refuses_a_change(tmp_path):
    ch = Fake(tmp_path).chain()
    ch.save_once("selection", {"config": "cfg2"})
    ch.save_once("selection", {"config": "cfg2"})
    with pytest.raises(S.ChainHalt, match="write-once"):
        ch.save_once("selection", {"config": "cfg3"})


def test_prefix_exposure_checks_the_sp_midpoint(monkeypatch):
    from eem_acc.stage3.apisuite import streams as ST
    real = ST.exposures
    assert S.prefix_exposure_ok(17, S.Settings())
    # the re-drifted op keeps a valid start pattern (3 early exposures) but none after the midpoint: only the SP
    # check can fail
    monkeypatch.setattr(ST, "exposures", lambda spec, op: [0, 1, 2] if spec.get("sp_op") == op else real(spec, op))
    assert not S.prefix_exposure_ok(17, S.Settings())
