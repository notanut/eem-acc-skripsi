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
COSTS = S.Costs(c0=40.0, c2=80.0, rq3=90.0, an=96.0, setup_s=600.0, gap_s=10.0, unit_hours_per_session=3.0)


def test_gate_fits_without_cuts():
    g = S.budget_gate(1.0, dataclasses.replace(COSTS, c0=10, c2=20, rq3=20, an=24), selection_done=False)
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
                 cfg_solved=None, crash_at=None, used=1.0, c2_cost=80.0, cfg_cost=None):
        self.root, self.c0, self.fallback, self.trunc = Path(root), c0, fallback, trunc
        self.unfinished, self.cfg_solved = unfinished, cfg_solved or {}
        self.crash_at, self.calls, self.phases, self.used, self.c2_cost = crash_at, 0, [], used, c2_cost
        self.cfg_cost, self.retries, self.fail, self.heal_on_retry = cfg_cost or {}, [], set(), True
        self.statuses = None                               # (phase, unit, retry) -> None | "failed" | "not_done"

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
                 else self.cfg_cost.get(u.tag, self.c2_cost) * u.n_tasks / 20)

    def run_units(self, phase, units, meta, retry=False):
        self.calls += 1
        if self.crash_at and self.calls == self.crash_at:
            raise KeyboardInterrupt("runtime lost")
        if not (self.root / "plans" / f"{phase}.json").exists():          # as the notebook: a plan is written once
            R.write_plan(self.root, phase, units, 1, frozen_commit="c" * 40, meta=meta)
        else:
            assert [u["unit_id"] if "unit_id" in u else R.Unit.from_json(u).unit_id
                    for u in R.read_plan(self.root, phase)["units"]] == [u.unit_id for u in units]
        self.phases.append(phase)
        if retry:
            self.retries.append(phase)
            if self.heal_on_retry:
                self.fail = {k for k in self.fail if k[0] != phase}
        out = []
        for u in units:
            mt = dict(self.metric(u), unit_id=u.unit_id)
            st = self.statuses(phase, u, retry) if self.statuses else None
            if st:
                out.append({"unit": u.to_json(), "m": dict(mt, status=st), "status": st})
            elif (phase, u.condition) in self.fail:        # a transient failure: no result
                out.append({"unit": u.to_json(), "m": dict(mt, status="failed"), "status": "failed"})
            else:
                out.append({"unit": u.to_json(), "m": mt})
        return out

    def chain(self, runtime="rt1", commit="c" * 40, repin=False, checks=None, adopt=None, freeze_extra=None):
        sec = Sections(self.root / "chain.json", stamp={"commit": commit, "nb_version": "v" + commit[:1]}, repin=repin)
        return S.Chain(sec, self.root, self.run_units, lambda r: r["m"], checks or (lambda rs: (True, 0)),
                       lambda: self.used, lambda: {"setup_s": 300.0, "gap_s": 10.0, "unit_hours_per_session": 3.0},
                       freeze_extra=freeze_extra or (lambda s: {"template_set_sha256": S.template_set_sha()}),
                       exposure_ok=lambda n, s: True, log=lambda msg: None, runtime=runtime, adopt=adopt)


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


def test_failed_unit_retries_once_then_pauses_for_a_code_fix(tmp_path):        # second opinion M3
    f = Fake(tmp_path)
    f.heal_on_retry = False
    f.fail = {("s34-B-r1", "C2")}
    with pytest.raises(S.ChainRetry, match="retries them once"):
        f.chain().run()
    with pytest.raises(S.ChainRetry, match="fresh runtime"):      # the same runtime may not retry
        f.chain().run()
    with pytest.raises(S.ChainPause, match="after a retry"):
        f.chain(runtime="rt2").run()
    assert f.retries == ["s34-B-r1"]
    log = json.loads((tmp_path / "chain.json").read_text())["log"]
    assert log[-1]["step"] == "X" and log[-1]["before"] == "B" and log[-1]["failed"] and log[-1]["phase"] == "s34-B-r1"


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
    with pytest.raises(S.ChainRetry):
        f.chain().run()
    with pytest.raises(S.ChainPause, match="units not done"):
        f.chain(runtime="rt2").run()
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


# ------------------------------------------------------------------------- S3.4 code second opinion (D165-D168)
def log_of(f):
    return json.loads((f.root / "chain.json").read_text())["log"]


def answer(f, d, **kw):
    ch = f.chain(**kw)
    return ch.apply_user_decision(d, S.log_fingerprint(ch.entries))


def test_transient_failure_retried_in_a_fresh_runtime_heals(tmp_path):              # second opinion M3
    f = Fake(tmp_path)
    f.fail = {("s34-B-r1", "C2")}
    with pytest.raises(S.ChainRetry):
        f.chain().run()
    assert f.chain(runtime="rt2").run() == "done"
    data = json.loads((tmp_path / "chain.json").read_text())
    (a,) = data["retries"]["s34-B-r1"]
    assert f.retries == ["s34-B-r1"] and a["runtime"] == "rt1" and a["retried_by"] == "rt2" and len(a["units"]) == 5
    assert [s for s, _ in steps(f)] == ["A", "B", "C", "D", "E", "F", "D", "G"]   # no pause, no extra round


def test_failed_units_pause_user_rerun_retries_the_same_phase(tmp_path):            # D165
    f = Fake(tmp_path)
    f.heal_on_retry = False
    f.fail = {("s34-C-r1", "C0")}
    with pytest.raises(S.ChainRetry):
        f.chain().run()
    with pytest.raises(S.ChainPause):
        f.chain(runtime="rt2").run()
    n = log_of(f)[-1]["n"]
    with pytest.raises(ValueError, match="only action is 'rerun'"):
        answer(f, {"answers": n, "action": "accept"})
    f.heal_on_retry = True
    answer(f, {"answers": n, "action": "rerun"})
    assert f.chain(runtime="rt2").run() == "done"
    assert f.retries == ["s34-C-r1", "s34-C-r1"] and not (tmp_path / "plans" / "s34-C-r2.json").exists()
    assert [e["step"] for e in log_of(f)][:5] == ["A", "B", "X", "U", "C"]


def test_repin_clears_a_fix_pause_and_retries(tmp_path):                            # D166
    f = Fake(tmp_path)
    f.heal_on_retry = False
    f.fail = {("s34-B-r1", "C2")}
    with pytest.raises(S.ChainRetry):
        f.chain().run()
    with pytest.raises(S.ChainPause):
        f.chain(runtime="rt2").run()
    with pytest.raises(S.ChainPause):                    # not re-pinned: stays paused
        f.chain(runtime="rt3").run()
    f.fail = set()                                       # the fix
    ch = f.chain(runtime="rt3", commit="d" * 40, repin=True, adopt=lambda ph, u: ("c" * 40, "sha", True))
    assert ch.run() == "done"
    log = log_of(f)
    u = [e for e in log if e["step"] == "U"][0]
    assert u["decision"] == {"action": "repin", "to": "d" * 40} and u["retry_phase"] == "s34-B-r1"
    assert json.loads((tmp_path / "chain.json").read_text())["repins"][0]["to"]["commit"] == "d" * 40


def test_adopt_repinned_replans_and_records(tmp_path):                              # D166
    f = Fake(tmp_path)
    with pytest.raises(S.ChainRetry):
        f.fail = {("s34-C-r1", "C0")}
        f.heal_on_retry = False
        f.chain().run()
    verdict = {"s34-A-r1": True, "s34-B-r1": True, "s34-C-r1": "ReplayDivergence: x"}
    ch = f.chain(commit="d" * 40, repin=True,
                 adopt=lambda ph, u: ("c" * 40, f"sha-{u.unit_id}", verdict[ph]))
    done = ch.adopt_repinned()
    assert [d["phase"] for d in done] == ["s34-A-r1", "s34-B-r1", "s34-C-r1"]
    for ph in ("s34-A-r1", "s34-B-r1", "s34-C-r1"):
        plan = R.read_plan(tmp_path, ph)
        assert plan["frozen_commit"] == "d" * 40 and plan["version"] == 2
    ad = R.adoptions(tmp_path, "s34-A-r1")
    assert len(ad) == 6 and all(v[0]["from"] == "c" * 40 and v[0]["to"] == "d" * 40 for v in ad.values())
    assert R.adoptions(tmp_path, "s34-C-r1") == {} and len(done[2]["rerun"]) == 2
    assert ch.adopt_repinned() == []                     # idempotent: every plan is at the current commit
    assert [a["phase"] for a in json.loads((tmp_path / "chain.json").read_text())["adoptions"]] == \
        ["s34-A-r1", "s34-B-r1", "s34-C-r1"]                                                   # per phase (L2)


def test_repin_that_changes_a_decided_result_halts(tmp_path):                      # D166
    f = Fake(tmp_path, crash_at=3)
    with pytest.raises(KeyboardInterrupt):
        f.chain().run()
    f.crash_at, f.c0 = None, (lambda s: 11)              # the fix changed what the units produce
    ch = f.chain(commit="d" * 40, repin=True, adopt=lambda ph, u: ("c" * 40, "x", False))
    with pytest.raises(S.ChainHalt, match="the code fix changed a decided result"):
        ch.run()


def test_user_accepts_a_paused_pilot_round(tmp_path):                              # D165
    f = Fake(tmp_path, c0=lambda s: 1)
    with pytest.raises(S.ChainPause, match="after 3 ladder steps"):
        f.chain().run()
    p = log_of(f)[-1]
    answer(f, {"answers": p["n"], "action": "accept"})
    assert f.chain().run() == "done"
    log = log_of(f)
    assert [e["step"] for e in log][-8:] == ["B", "U", "C", "D", "E", "F", "D", "G"]
    gate = [e for e in log if e["step"] == "D"][0]["decision"]
    assert gate["costs"]["c0"] == p["decision"]["summary"]["C0"]["seconds_per_task"]     # the accepted round's cost


def test_user_settings_after_a_pause_and_a_second_miss_pauses_again(tmp_path):     # D165
    f = Fake(tmp_path, c0=lambda s: 10 if (s.n_distractors == 0 and s.long_share == 0.4) else 1)
    with pytest.raises(S.ChainPause):
        f.chain().run()
    n = log_of(f)[-1]["n"]
    with pytest.raises(ValueError, match="non-empty subset"):
        answer(f, {"answers": n, "action": "settings", "settings": {"bogus": 1}})
    answer(f, {"answers": n, "action": "settings", "settings": {"n_distractors": 1, "long_share": 0.4}})
    with pytest.raises(S.ChainPause, match="under the user's settings"):
        f.chain().run()
    b = [e for e in log_of(f) if e["step"] == "B"][-1]
    u = R.Unit.from_json(R.read_plan(tmp_path, b["phase"])["units"][0])
    assert dict(u.stream) == {"n_distractors": 1, "long_share": 0.4}
    n = log_of(f)[-1]["n"]                               # a user answer that works: accepted directly
    answer(f, {"answers": n, "action": "settings", "settings": {"n_distractors": 0}})
    assert f.chain().run() == "done"


def test_user_decisions_on_c_and_h_pauses(tmp_path):                               # D165
    f = Fake(tmp_path, unfinished=1)
    with pytest.raises(S.ChainPause, match="unfinished"):
        f.chain().run()
    answer(f, {"answers": log_of(f)[-1]["n"], "action": "rerun"})
    with pytest.raises(S.ChainPause, match="unfinished"):
        f.chain().run()
    assert (tmp_path / "plans" / "s34-C-r2.json").exists()
    answer(f, {"answers": log_of(f)[-1]["n"], "action": "accept"})
    assert f.chain().run() == "done"
    g = Fake(tmp_path / "h", fallback=lambda s, u: 12 if (u.phase.startswith("s34-F") and not s.grammar_always) else 0)
    orig = g.metric
    g.metric = lambda u: dict(orig(u), tasks=[dict(t, unfinished=True) for t in orig(u)["tasks"]]) \
        if (u.rq == "rq3" and u.phase.startswith("s34-H")) else orig(u)
    with pytest.raises(S.ChainPause, match="after D60"):
        g.chain().run()
    answer(g, {"answers": log_of(g)[-1]["n"], "action": "accept"})
    assert g.chain().run() == "done"
    assert [e["step"] for e in log_of(g)][-4:] == ["H", "U", "D", "G"]


def test_user_gate_decision_after_a_d57_pause(tmp_path):                           # D165
    f = Fake(tmp_path, used=59.5)
    with pytest.raises(S.ChainPause, match="D57"):
        f.chain().run()
    n = log_of(f)[-1]["n"]
    with pytest.raises(ValueError, match="cut numbers"):
        answer(f, {"answers": n, "action": "gate", "cuts": [12], "n_tasks": 16})
    answer(f, {"answers": n, "action": "gate", "cuts": list(range(12)), "n_tasks": 16})
    f.used = 1.0
    assert f.chain().run() == "done"
    log = log_of(f)
    d = [e for e in log if e["step"] == "D"]
    assert d[1]["user"] and d[1]["decision"]["n_tasks"] == 16 and not d[1].get("pause")
    freeze = json.loads((tmp_path / "s34" / "freeze.json").read_text())
    assert freeze["n_tasks"] == 16 and freeze["cuts"] == list(range(12))
    assert f.chain().verify_log() is None                # a user gate entry is not recomputed


def test_apply_user_decision_guards(tmp_path):                                      # D165, second opinion M6
    f = Fake(tmp_path)
    with pytest.raises(ValueError, match="not paused"):
        answer(f, {"answers": 0, "action": "accept"})
    g = Fake(tmp_path / "g", unfinished=1)
    with pytest.raises(S.ChainPause):
        g.chain().run()
    n = log_of(g)[-1]["n"]
    ch = g.chain()
    with pytest.raises(ValueError, match="EXPECT_LOG"):
        ch.apply_user_decision({"answers": n, "action": "accept"}, "3:000000000000")
    with pytest.raises(ValueError, match="current pause is n="):
        answer(g, {"answers": n - 1, "action": "accept"})
    with pytest.raises(ValueError, match="not allowed after a C pause"):
        answer(g, {"answers": n, "action": "settings"})
    d = {"answers": n, "action": "accept"}
    e1 = answer(g, d)
    assert g.chain().apply_user_decision(d, None) == e1                     # already recorded: no-op, no check
    assert sum(e["step"] == "U" for e in log_of(g)) == 1


def test_replay_gate_before_the_freeze_and_after_selection(tmp_path):              # D168
    f = Fake(tmp_path)
    bad = {"on": True}
    checks = lambda rs: (not (bad["on"] and rs and rs[0]["unit"]["phase"].startswith(("s34-A", "s34-B"))), 0)
    with pytest.raises(S.ChainPause, match="replay check before E"):
        f.chain(checks=checks).run()
    assert not (tmp_path / "s34" / "freeze.json").exists()
    with pytest.raises(ValueError, match="only action is 'rerun'"):
        answer(f, {"answers": log_of(f)[-1]["n"], "action": "accept"}, checks=checks)
    answer(f, {"answers": log_of(f)[-1]["n"], "action": "rerun"}, checks=checks)      # review M4: transient?
    with pytest.raises(S.ChainPause, match="replay check before E"):
        f.chain(checks=checks).run()                     # still failing: paused again
    bad["on"] = False
    ch = f.chain(commit="d" * 40, repin=True, checks=checks, adopt=lambda ph, u: ("c" * 40, "s", True))
    assert ch.run() == "done"
    g = Fake(tmp_path / "g")
    with pytest.raises(S.ChainPause, match="replay check before F"):
        g.chain(checks=lambda rs: (not rs[0]["unit"]["phase"].startswith("s34-F"), 0)).run()
    assert "selection" not in json.loads((tmp_path / "g" / "chain.json").read_text())


def test_final_gate_prices_c2_at_the_selected_config(tmp_path):                    # D167
    f = Fake(tmp_path, cfg_solved={"cfg5": 16}, cfg_cost={"cfg5": 120.0})
    assert f.chain().run() == "done"
    d = [e["decision"] for e in log_of(f) if e["step"] == "D"]
    assert d[0]["costs"]["c2"] == 4.0 and d[1]["costs"]["c2"] == 6.0             # seconds per task
    assert d[0]["costs"]["an"] == pytest.approx(4.8) and d[1]["costs"]["an"] == pytest.approx(7.2)
    w = {name: per for name, _, _, per in S.workload(20, frozenset(), 0, False, 4)}
    assert w["AN"] == "an" and w["C2-OU"] == "c2"


def test_missing_costs_fall_back(tmp_path):                                         # second opinion M2
    f = Fake(tmp_path)
    orig = f.metric
    f.metric = lambda u: dict(orig(u), cost_seconds=None) if u.rq == "rq3" else orig(u)
    assert f.chain().run() == "done"
    d = [e["decision"] for e in log_of(f) if e["step"] == "D"][0]
    assert d["costs"]["rq3"] == d["costs"]["c2"]
    g = Fake(tmp_path / "g")
    orig_g = g.metric
    g.metric = lambda u: dict(orig_g(u), cost_seconds=None) if u.phase.startswith("s34-B") else orig_g(u)
    with pytest.raises(S.ChainHalt, match="no measurable cost"):
        g.chain().run()


def test_verify_log_h_ignores_timing_and_checks_d_and_freeze(tmp_path):           # second opinion M1, Low 3
    f = Fake(tmp_path, cfg_solved={"cfg2": 15},
             fallback=lambda s, u: 0 if (s.grammar_always or u.phase.startswith("s34-B") or u.condition == "C0")
             else 12)
    assert f.chain().run() == "done"
    f.c2_cost = 999.0                                    # a re-run unit's timing differs: no halt (M1)
    f.chain().verify_log()
    sec = tmp_path / "chain.json"
    data = json.loads(sec.read_text())
    i = [k for k, e in enumerate(data["log"]) if e["step"] == "D"][0]
    data["log"][i]["decision"]["n_tasks"] = 17
    sec.write_text(json.dumps(data))
    with pytest.raises(S.ChainHalt, match="recomputed gate"):
        f.chain().verify_log()
    data["log"][i]["decision"]["n_tasks"] = 20
    sec.write_text(json.dumps(data))
    f.chain().verify_log()
    (tmp_path / "s34" / "freeze.json").write_text("{}")
    with pytest.raises(S.ChainHalt, match="freeze file"):
        f.chain().verify_log()


def test_verify_log_checks_the_d60_switches_of_f(tmp_path):                         # second opinion Low 3
    f = Fake(tmp_path, cfg_solved={"cfg2": 15},
             fallback=lambda s, u: 0 if (s.grammar_always or u.phase.startswith("s34-B") or u.condition == "C0")
             else 12)
    assert f.chain().run() == "done"
    f.fallback = lambda s, u: 0                          # F's stored results no longer fire the switch
    with pytest.raises(S.ChainHalt, match="s34-F-r1"):
        f.chain().verify_log()


def test_h_rq3_unit_runs_the_selected_config(tmp_path):                            # second opinion Low 1
    f = Fake(tmp_path, cfg_solved={"cfg5": 15},
             fallback=lambda s, u: 0 if (s.grammar_always or u.phase.startswith("s34-B") or u.condition == "C0")
             else 12)
    assert f.chain().run() == "done"
    h = {(u.rq, u.condition): u for u in map(R.Unit.from_json, R.read_plan(tmp_path, "s34-H-r1")["units"])}
    p = dict(h[("rq3", "C2")].params)
    assert all(p[k] == v for k, v in S.CONFIGS["cfg5"].items())
    assert not set(S.CONFIGS["cfg5"]) & set(dict(h[("rq3", "C0")].params)) - set(S.Settings().params())


def test_log_fingerprint():
    assert S.log_fingerprint([]) == "0:" + __import__("hashlib").sha256(b"[]").hexdigest()[:12]
    a, b = S.log_fingerprint([{"n": 0, "step": "A"}]), S.log_fingerprint([{"n": 0, "step": "B"}])
    assert a != b and a.startswith("1:")


def test_realised_long_share_reported_per_pilot_round(tmp_path):                 # second opinion Low 7
    nat = S.natural_long_share()
    base = S.realised_long_share(S.pilot_units("s34-B-r1", S.Settings()))
    more = S.realised_long_share(S.pilot_units("s34-B-r2", S.Settings(long_share=round(nat + 0.1, 6))))
    assert 0 < base < 1 and more > base
    f = Fake(tmp_path, c0=lambda s: 2 if s.long_share is None else 10)
    assert f.chain().run() == "done"
    bs = [e["decision"]["realised_long_share"] for e in log_of(f) if e["step"] == "B"]
    assert len(bs) == 2 and bs[1] < bs[0]                # "hard": the ladder lowered the long share


def test_expect_log_refuses_a_stale_or_foreign_chain_before_any_write(tmp_path):    # second opinion M6, review H1
    from eem_acc.stage3.llm.fsio import StaleView
    f = Fake(tmp_path, unfinished=1)
    with pytest.raises(S.ChainPause):
        f.chain().run()
    path = tmp_path / "chain.json"
    log = json.loads(path.read_text())["log"]
    stamp = {"commit": "c" * 40, "nb_version": "vc"}
    for ok in (None, S.log_fingerprint(log), len(log) - 1, S.log_fingerprint(log[:2])):   # a longer log is fine
        Sections(path, stamp=stamp, expect_log=ok)
    before = path.read_bytes()
    with pytest.raises(StaleView, match="not caught up"):
        Sections(path, stamp={"commit": "d" * 40, "nb_version": "vd"}, repin=True, expect_log=f"{len(log) + 1}:abc")
    with pytest.raises(ValueError, match="does not start with"):
        Sections(path, stamp=stamp, expect_log="2:000000000000")
    assert path.read_bytes() == before                    # the re-pin write never happened


# ------------------------------------------------------------------- fresh-context review of the fixes (2026-11-26)
def test_a_recorded_phase_that_fails_on_reload_retries_then_pauses(tmp_path):    # review H2
    f = Fake(tmp_path, crash_at=4)
    with pytest.raises(KeyboardInterrupt):
        f.chain().run()                                  # A-D recorded; lost in E's replay check
    f.crash_at, f.heal_on_retry = None, False
    f.fail = {("s34-A-r1", "C0")}                        # e.g. a lost result is repaired and the GPU fails
    with pytest.raises(S.ChainRetry):
        f.chain().run()                                  # no false "the code fix changed a decided result" halt
    with pytest.raises(S.ChainPause, match="after a retry"):
        f.chain(runtime="rt2").run()
    x = log_of(f)[-1]
    assert x["step"] == "X" and x["phase"] == "s34-A-r1" and x["before"] == "E"
    f.heal_on_retry = True
    answer(f, {"answers": x["n"], "action": "rerun"}, runtime="rt2")
    assert f.chain(runtime="rt2").run() == "done"
    assert [e["step"] for e in log_of(f)][:7] == ["A", "B", "C", "D", "X", "U", "E"]


def test_retry_recorded_only_after_it_ran_and_new_failures_get_their_own(tmp_path):   # review M1
    f = Fake(tmp_path)
    f.fail = {("s34-B-r1", "C2")}
    with pytest.raises(S.ChainRetry):
        f.chain().run()
    real = f.run_units

    def lost(phase, units, meta, retry=False):
        if retry:
            raise KeyboardInterrupt("lost during the retry")
        return real(phase, units, meta, retry)
    f.run_units = lost
    with pytest.raises(KeyboardInterrupt):
        f.chain(runtime="rt2").run()
    f.run_units = real
    (a,) = json.loads((tmp_path / "chain.json").read_text())["retries"]["s34-B-r1"]
    assert "retried_by" not in a                         # the lost retry is not counted
    f.heal_on_retry = False
    f.fail = {("s34-B-r1", "C0")}                        # the retry heals C2, but C0 now fails for the first time
    with pytest.raises(S.ChainRetry, match="units not done"):
        f.chain(runtime="rt3").run()
    f.heal_on_retry = True
    assert f.chain(runtime="rt4").run() == "done"


def test_h_rerun_does_not_use_up_the_d60_rounds(tmp_path):                       # review M3
    f = Fake(tmp_path, fallback=lambda s, u: 12 if (u.phase.startswith("s34-F") and not s.grammar_always) else 0,
             trunc=lambda s, u: 3 if (u.phase == "s34-H-r2" and u.condition == "C2") else 0)
    orig = f.metric
    f.metric = lambda u: dict(orig(u), tasks=[dict(t, unfinished=True) for t in orig(u)["tasks"]]) \
        if (u.rq == "rq3" and u.phase == "s34-H-r1") else orig(u)
    with pytest.raises(S.ChainPause, match="after D60"):
        f.chain().run()
    answer(f, {"answers": log_of(f)[-1]["n"], "action": "rerun"})
    assert f.chain().run() == "done"
    hs = [e for e in log_of(f) if e["step"] == "H"]
    assert [h["decision"]["fired"] for h in hs] == [[], ["cap640"], []] and hs[1]["again"]
    h3 = [R.Unit.from_json(u) for u in R.read_plan(tmp_path, "s34-H-r3")["units"]]
    assert all(dict(u.params).get("action_max_tokens") == 640 for u in h3)       # re-measured under the cap


def test_user_gate_is_monotone_and_its_projection_recomputed(tmp_path, monkeypatch):   # review M5
    real = S.budget_gate

    def gate(*a, **k):                                   # first gate: cuts 0, 1 at 18 tasks; the final gate pauses
        g = real(*a, **k)
        if k.get("budget", S.BUDGET_HOURS) != S.BUDGET_HOURS:
            return g                                     # the user's gate: unchanged
        if not k["selection_done"]:
            return {**g, "cuts": sorted(set(g["cuts"]) | {0, 1}), "n_tasks": min(g["n_tasks"], 18)}
        return {**g, "fits": False, "pause": "the budget does not fit at 16 tasks per stream (D57, D160)"}
    monkeypatch.setattr(S, "budget_gate", gate)
    f = Fake(tmp_path, cfg_solved={"cfg2": 15})
    with pytest.raises(S.ChainPause, match="D57"):
        f.chain().run()
    n = log_of(f)[-1]["n"]
    for bad in ({"cuts": [0], "n_tasks": 16}, {"cuts": [0, 1], "n_tasks": 19}):
        with pytest.raises(ValueError, match="only ever added"):
            answer(f, {"answers": n, "action": "gate", **bad})
    answer(f, {"answers": n, "action": "gate", "cuts": [0, 1, 2], "n_tasks": 16})
    g = [e for e in log_of(f) if e["step"] == "D"][-1]["decision"]
    exp = S.projection(S.Costs(**g["costs"]), S.workload(16, frozenset({0, 1, 2}), 0, False, 4))
    assert g["user"] and g["cuts"] == [0, 1, 2] and g["projected"]["total"] == pytest.approx(exp["total"])
    assert f.chain().run() == "done"


def test_user_settings_are_validated(tmp_path):                                  # review L5
    f = Fake(tmp_path, c0=lambda s: 1)
    with pytest.raises(S.ChainPause):
        f.chain().run()
    n = log_of(f)[-1]["n"]
    for bad in ({"long_share": 1.5}, {"long_share": -0.1}, {"n_distractors": "1"}, {"n_distractors": 5}):
        with pytest.raises(ValueError, match="invalid settings"):
            answer(f, {"answers": n, "action": "settings", "settings": bad})
    for other in ({"grammar_always": True}, {"action_max_tokens": 640}, {"wording": "W2"}):   # 2nd opinion #2 L1
        with pytest.raises(ValueError, match="non-empty subset"):
            answer(f, {"answers": n, "action": "settings", "settings": other})
    answer(f, {"answers": n, "action": "settings", "settings": {"n_distractors": 4}})    # the ladder's "easy" end


def test_freeze_written_before_a_repin_is_kept(tmp_path):                       # review L3
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
    before = (tmp_path / "s34" / "freeze.json").read_bytes()
    ch2 = f.chain(commit="d" * 40, repin=True, adopt=lambda ph, u: ("c" * 40, "s", True))
    assert ch2.run() == "done"
    assert (tmp_path / "s34" / "freeze.json").read_bytes() == before
    e = [x for x in log_of(f) if x["step"] == "E"][0]
    assert e["freeze_sha"] == __import__("hashlib").sha256(before).hexdigest()


# --------------------------------------------------------------------- second opinion #2 on the fixes (2026-11-26)
def test_accept_only_for_a_sane_wiring(tmp_path):                                   # 2nd opinion #2, M3
    f = Fake(tmp_path)
    leaky = lambda rs: (True, 3 if rs and rs[0]["unit"]["phase"].startswith("s34-C") else 0)
    with pytest.raises(S.ChainPause, match="wiring not sane"):
        f.chain(checks=leaky).run()
    n = log_of(f)[-1]["n"]
    with pytest.raises(ValueError, match="not sane"):
        answer(f, {"answers": n, "action": "accept"}, checks=leaky)
    answer(f, {"answers": n, "action": "rerun"}, checks=leaky)
    assert f.chain().run() == "done"                     # the fixed checks pass on the re-run
    g = Fake(tmp_path / "g", fallback=lambda s, u: 12 if (u.phase.startswith("s34-F") and not s.grammar_always) else 0)
    bad_h = lambda rs: (not (rs and rs[0]["unit"]["phase"].startswith("s34-H")), 0)
    with pytest.raises(S.ChainPause, match="after D60"):
        g.chain(checks=bad_h).run()
    with pytest.raises(ValueError, match="not sane"):
        answer(g, {"answers": log_of(g)[-1]["n"], "action": "accept"}, checks=bad_h)


def test_repin_past_the_freeze_checks_the_frozen_material(tmp_path):              # 2nd opinion #2, M2
    f = Fake(tmp_path)
    assert f.chain().run() == "done"
    same = f.chain(commit="d" * 40, repin=True, adopt=lambda ph, u: ("c" * 40, "s", True))
    assert same.run() == "done"
    data = json.loads((tmp_path / "chain.json").read_text())
    (add,) = data["freeze_addenda"]
    assert add["commit"] == "d" * 40 and add["changed"] == {} and add["hashes"]["natural_long_share"] == S.natural_long_share()
    other = lambda s: {"template_set_sha256": "changed-templates"}
    ch = f.chain(commit="e" * 40, repin=True, adopt=lambda ph, u: ("c" * 40, "s", True), freeze_extra=other)
    with pytest.raises(S.ChainPause, match="changes frozen material"):
        ch.run()
    z = log_of(f)[-1]
    assert z["step"] == "Z" and z["changed"]["template_set_sha256"][1] == "changed-templates"
    with pytest.raises(ValueError, match="only action is 'accept'"):
        answer(f, {"answers": z["n"], "action": "rerun"}, commit="e" * 40, freeze_extra=other)
    answer(f, {"answers": z["n"], "action": "accept"}, commit="e" * 40, freeze_extra=other)
    assert f.chain(commit="e" * 40, freeze_extra=other).run() == "done"
    adds = json.loads((tmp_path / "chain.json").read_text())["freeze_addenda"]
    assert adds[-1]["commit"] == "e" * 40 and adds[-1]["user"] and adds[-1]["changed"]
    assert json.loads((tmp_path / "s34" / "freeze.json").read_text())["commit"] == "c" * 40   # never rewritten


def test_units_a_halt_left_unrun_get_their_own_retry(tmp_path):                    # 2nd opinion #2, Low 2
    f = Fake(tmp_path)
    first = {"s34-B-r1-rq1-s1002"}

    def st(phase, u, retry):
        if phase != "s34-B-r1" or u.condition != "C2":
            return None
        key = f"{u.phase}-{u.rq}-s{u.seed}"
        if not f.retries:                                # session 1: seed 1002 fails, the halt leaves the rest unrun
            return "failed" if key in first else "not_done"
        if len(f.retries) == 1:                          # the retry: 1002 succeeds, 1004 fails for the first time
            return "failed" if u.seed == 1004 else None
        return None
    f.statuses = st
    with pytest.raises(S.ChainRetry):
        f.chain().run()
    with pytest.raises(S.ChainRetry, match="units not done"):  # not a pause: 1004 never had its own retry
        f.chain(runtime="rt2").run()
    assert f.chain(runtime="rt3").run() == "done"


def test_paused_chain_shows_a_changed_decision_right_after_the_repin(tmp_path):   # 2nd opinion #2, Low 3
    f = Fake(tmp_path, c0=lambda s: 1)
    with pytest.raises(S.ChainPause, match="after 3 ladder steps"):
        f.chain().run()
    f.c0 = lambda s: 1 if s.long_share is not None or s.n_distractors != 2 else 2      # the fix changes A/B results
    with pytest.raises(S.ChainHalt, match="code fix changed a decided result"):
        f.chain(commit="d" * 40, repin=True, adopt=lambda ph, u: ("c" * 40, "s", True)).run()
    g = Fake(tmp_path / "g", c0=lambda s: 1)
    with pytest.raises(S.ChainPause):
        g.chain().run()
    g.statuses = lambda phase, u, retry: "failed" if phase == "s34-A-r1" and u.tag == "W1" else None
    n_log = len(log_of(g))
    with pytest.raises(S.ChainPause, match="after 3 ladder steps"):   # deferred: nothing recorded during the pause
        g.chain(commit="d" * 40, repin=True, adopt=lambda ph, u: ("c" * 40, "s", True)).run()
    assert len(log_of(g)) == n_log and "retries" not in json.loads((tmp_path / "g" / "chain.json").read_text())


def test_final_gate_scales_rq3_to_the_selected_config(tmp_path):                 # D171
    f = Fake(tmp_path, cfg_solved={"cfg5": 16}, cfg_cost={"cfg5": 120.0})
    assert f.chain().run() == "done"
    d = [e["decision"] for e in log_of(f) if e["step"] == "D"]
    assert d[0]["costs"]["rq3"] == 90.0 and d[1]["costs"]["rq3"] == pytest.approx(90.0 * 6.0 / 4.0)


# --------------------------------------------------------------- fresh-context review of round 2 (2026-11-26)
def test_x_then_repin_with_a_freeze_change_keeps_the_retry(tmp_path):             # fix review M2
    f = Fake(tmp_path, cfg_solved={"cfg2": 15})
    f.heal_on_retry = False
    f.fail = {("s34-F-r1", "C2")}
    with pytest.raises(S.ChainRetry):
        f.chain().run()
    with pytest.raises(S.ChainPause, match="after a retry"):
        f.chain(runtime="rt2").run()
    other = lambda s: {"template_set_sha256": "changed"}
    kw = dict(commit="d" * 40, freeze_extra=other)
    with pytest.raises(S.ChainPause, match="changes frozen material"):
        f.chain(runtime="rt3", repin=True, adopt=lambda ph, u: ("c" * 40, "s", True), **kw).run()
    z = log_of(f)[-1]
    assert z["step"] == "Z" and z["retry_phase"] == "s34-F-r1"
    f.heal_on_retry = True
    answer(f, {"answers": z["n"], "action": "accept"}, runtime="rt3", **kw)
    assert f.chain(runtime="rt3", **kw).run() == "done"
    assert set(f.retries) == {"s34-F-r1"} and len(f.retries) >= 2   # the fixed code retried the units (the F replay
                                                                    # check reloads the phase with the retry flag too)


def test_freeze_kept_across_a_repin_is_checked_before_f(tmp_path):                # fix review L3
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
    with pytest.raises(S.ChainPause, match="changes frozen material"):
        f.chain(commit="d" * 40, repin=True, adopt=lambda ph, u: ("c" * 40, "s", True),
                freeze_extra=lambda s: {"template_set_sha256": "changed"}).run()
    assert [e["step"] for e in log_of(f)][-2:] == ["E", "Z"] and log_of(f)[-1]["before"] == "F"
    assert not (tmp_path / "plans" / "s34-F-r1.json").exists()


def test_accepted_freeze_change_is_the_new_baseline(tmp_path):                    # fix review L6
    f = Fake(tmp_path)
    assert f.chain().run() == "done"
    other = lambda s: {"template_set_sha256": "changed"}
    with pytest.raises(S.ChainPause, match="changes frozen material"):
        f.chain(commit="d" * 40, repin=True, adopt=lambda ph, u: ("c" * 40, "s", True), freeze_extra=other).run()
    answer(f, {"answers": log_of(f)[-1]["n"], "action": "accept"}, commit="d" * 40, freeze_extra=other)
    assert f.chain(commit="d" * 40, freeze_extra=other).run() == "done"
    ch = f.chain(commit="e" * 40, repin=True, adopt=lambda ph, u: ("d" * 40, "s", True), freeze_extra=other)
    assert ch.run() == "done"                            # the same change again: no second pause
    assert json.loads((tmp_path / "chain.json").read_text())["freeze_addenda"][-1]["changed"] == {}


def test_paused_recompute_runs_nothing_and_survives_a_lost_session(tmp_path):     # fix review L4, L7
    f = Fake(tmp_path, c0=lambda s: 1)
    with pytest.raises(S.ChainPause):
        f.chain().run()
    ch = f.chain(commit="d" * 40, repin=True, adopt=lambda ph, u: ("c" * 40, "s", True))
    ch.adopt_repinned()                                  # the session is lost right after the re-plan
    f.c0 = lambda s: 1 if s.long_share is not None or s.n_distractors != 2 else 2
    with pytest.raises(S.ChainHalt, match="code fix changed a decided result"):
        f.chain(commit="d" * 40).run()                   # the next session still recomputes
    g = Fake(tmp_path / "g", c0=lambda s: 1)
    with pytest.raises(S.ChainPause):
        g.chain().run()
    calls = g.calls
    with pytest.raises(S.ChainPause, match="after 3 ladder steps"):
        g.chain(commit="d" * 40, repin=True, adopt=lambda ph, u: ("c" * 40, "s", "ReplayDivergence")).run()
    assert g.calls == calls                              # units to re-run: nothing runs while paused
    adds = json.loads((tmp_path / "g" / "chain.json").read_text())["adoptions"]
    assert all(set(a["reasons"].values()) == {"ReplayDivergence"} for a in adds)          # fix review L8


def test_repin_that_changes_the_natural_long_share_pauses(tmp_path, monkeypatch):  # fix review nit
    f = Fake(tmp_path)
    assert f.chain().run() == "done"
    monkeypatch.setattr(S, "natural_long_share", lambda: 0.3)
    with pytest.raises(S.ChainPause, match="natural_long_share"):
        f.chain(commit="d" * 40, repin=True, adopt=lambda ph, u: ("c" * 40, "s", True)).run()
