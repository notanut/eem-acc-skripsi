"""S3.4 platform (notebook 05 cell 7's callbacks) on the real unit runner with the scripted backend: the chain end to
end, its resume from stored results, a re-pin with adoption by replay (D166), and the platform's own guards."""
import hashlib
import json
from pathlib import Path

import pytest

from eem_acc.stage3 import s34 as S
from eem_acc.stage3.agent import ScriptedBackend
from eem_acc.stage3.llm import runner as R
from eem_acc.stage3.llm.t4 import Sections
from eem_acc.stage3.s34_platform import Platform

from test_s33_runner import TOK, header, runner

C, D = "c" * 40, "d" * 40
EXTRA = {"setup_s": 300.0, "gap_s": 10.0, "unit_hours_per_session": 3.0}


@pytest.fixture
def small(monkeypatch):
    """Fewer seeds (the rules are unchanged): one wording seed, two pilot seeds, one selection seed."""
    monkeypatch.setattr(S, "WORDING_SEEDS", (1000,))
    monkeypatch.setattr(S, "PILOT_SEEDS", (1002, 1003))
    monkeypatch.setattr(S, "SELECTION_SEEDS", (1007,))


class FreshScripted(Platform):
    """The scripted backend keeps a registry of task instructions; a ladder step re-samples a seed's stream, so two
    phases may share an instruction with different tasks. A fresh scripted backend per phase (the llama.cpp backend
    has no such registry)."""

    def run_units(self, phase, units, meta, retry=False):
        self.run.ctx.backend = ScriptedBackend(TOK, "memory")
        return super().run_units(phase, units, meta, retry)


def platform(root, commit=C):
    run = runner(root, ScriptedBackend(TOK, "memory"), hdr={**header(), "code_commit": commit})
    return FreshScripted(root, run, commit, lambda u: (ScriptedBackend(TOK, "memory"), TOK), log=lambda m: None)


def chain(root, commit=C, repin=False):
    pf = platform(root, commit)
    sec = Sections(Path(root) / "s34" / "chain.json", stamp={"commit": commit, "nb_version": commit[:4]}, repin=repin)
    ch = S.Chain(sec, root, pf.run_units, pf.metrics, pf.checks, lambda: 1.0, lambda: EXTRA, pf.freeze_extra,
                 exposure_ok=lambda n, s: True, log=lambda m: None, runtime="rt-" + commit[:1], adopt=pf.adopt)
    return ch, pf


def drive(root, commit=C, repin=False, max_pauses=4):
    """Run the chain to the end, answering each user pause the way the chat's default would (accept / rerun)."""
    for _ in range(max_pauses + 1):
        ch, pf = chain(root, commit, repin)
        try:
            return ch.run(), ch, pf
        except S.ChainPause:
            p = ch.entries[-1]
            act = {"D": "gate"}.get(p.get("before", p["step"]), "accept")
            d = {"answers": p["n"], "action": act}
            if act == "gate":
                d.update(cuts=list(range(12)), n_tasks=16)
            ch.apply_user_decision(d, S.log_fingerprint(ch.entries))
    raise AssertionError("too many pauses")


def result_shas(root):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((Path(root) / "units").rglob("*.json"))}


def test_chain_end_to_end_on_the_real_runner(tmp_path, small):
    state, ch, pf = drive(tmp_path)
    assert state == "done"
    steps = [e["step"] for e in ch.entries]
    assert steps[0] == "A" and steps[-1] == "G" and "E" in steps and "F" in steps
    freeze = json.loads((tmp_path / "s34" / "freeze.json").read_text())
    assert freeze["commit"] == C and freeze["payload_sha256"] and freeze["system_prompt_sha256"]
    sel = ch.S.data["selection"]
    assert sel["config"] in S.CONFIGS and sel["seeds"] == [1007]
    verified = pf.verify_finished({})
    assert verified and all(v["verify_replay"] is True and v["audit_hits"] == [] for v in verified.values())
    # resume: a later session recomputes every decision from the stored results and has nothing left to do
    before = result_shas(tmp_path)
    ch2, _ = chain(tmp_path)
    assert ch2.run() == "done" and result_shas(tmp_path) == before


def test_repin_adopts_every_unit_by_replay(tmp_path, small):
    assert drive(tmp_path)[0] == "done"
    before = result_shas(tmp_path)
    n_log = len(json.loads((tmp_path / "s34" / "chain.json").read_text())["log"])
    ch, pf = chain(tmp_path, D, repin=True)
    assert ch.run() == "done"
    data = ch.S.data
    assert data["stamp"]["commit"] == D and data["repins"][0]["from"]["commit"] == C
    assert data["adoptions"] and all(not ph["rerun"] for ph in data["adoptions"])   # every unit replayed identically
    assert result_shas(tmp_path) == before and len(data["log"]) == n_log          # nothing re-ran, nothing re-decided
    for pf_file in sorted((tmp_path / "plans").glob("s34-*.json")):
        plan = json.loads(pf_file.read_text())
        assert plan["frozen_commit"] == D
        assert all(pf.run.is_done(R.Unit.from_json(u), plan=plan) for u in plan["units"])   # adopted counts as done


def test_repin_reruns_a_unit_that_does_not_replay(tmp_path, small):
    ch, pf = chain(tmp_path)
    with pytest.raises(S.ChainPause):                     # A and B once: enough finished units
        _first_steps(ch)
    plan = R.read_plan(tmp_path, "s34-A-r1")
    u = R.Unit.from_json(plan["units"][0])
    cp = pf.run.cache_path(u)
    lines = cp.read_text().splitlines()
    rec = json.loads(lines[-1])
    lines[-1] = json.dumps({**rec, "zz": 1})              # a cache the fixed code no longer replays identically
    cp.write_text("\n".join(lines) + "\n")
    ch2, pf2 = chain(tmp_path, D, repin=True)
    done = ch2.adopt_repinned()
    a = [d for d in done if d["phase"] == "s34-A-r1"][0]
    assert a["rerun"] == [u.unit_id] and u.unit_id not in R.adoptions(tmp_path, "s34-A-r1")
    assert not pf2.run.is_done(u, plan=R.read_plan(tmp_path, "s34-A-r1"))
    ch2._load("s34-A-r1", "C")                            # the next run re-runs it at the fixed commit ...
    payload = R.split_result(json.loads(pf2.run.result_path(u).read_text()))[0]
    assert payload["stamp"]["llm"]["code_commit"] == D and pf2.run.is_done(u, plan=R.read_plan(tmp_path, "s34-A-r1"))
    ch2.verify_log()                                      # ... and the recorded decisions still hold


def _first_steps(ch):
    """Run A and B once, then stop as a pause would."""
    ch._do_A()
    ch._do_B()
    raise S.ChainPause("stop")


def test_platform_guards(tmp_path, small):
    pf = platform(tmp_path)
    s = S.Settings()
    units = S.wording_units(1, s)
    out = pf.run_units("s34-A-r1", units, None)
    assert all("run" in r for r in out)
    with pytest.raises(S.ChainHalt, match="other units"):
        pf.run_units("s34-A-r1", units[:-1], None)
    pf.run_units("s34-X-r1", units[:1], {"freeze_sha": "a"})
    with pytest.raises(S.ChainHalt, match="meta differs"):
        pf.run_units("s34-X-r1", units[:1], {"freeze_sha": "b"})
    assert pf.adopt("s34-A-r1", units[0])[0] == C and pf.adopt("s34-A-r1", units[0])[2] is True
    missing = R.Unit("s34-A-r1", "rq1", "dev", "rq1", 999, "C0")
    assert pf.adopt("s34-A-r1", missing) == (None, None, None)
    assert pf.checks([{"unit": missing.to_json(), "status": "not_done"}]) == (False, 0)
    v = pf.verify_finished({})
    assert len(v) == len(units)
    assert pf.verify_finished(v) == v                     # finished units are checked once


def test_platform_retry_reruns_failed_units(tmp_path, small):
    class Boom(ScriptedBackend):
        def generate(self, *a, **k):
            raise ValueError("cuda context broken")
    units = S.wording_units(1, S.Settings())[:1]
    pf = Platform(tmp_path, runner(tmp_path, Boom(TOK, "memory")), C, lambda u: (ScriptedBackend(TOK, "memory"), TOK),
                  log=lambda m: None)
    out = pf.run_units("s34-A-r1", units, None)
    assert out[0]["status"] == "failed"
    pf2 = platform(tmp_path)
    assert pf2.run_units("s34-A-r1", units, None)[0]["status"] == "failed"      # skipped while failed
    assert "run" in pf2.run_units("s34-A-r1", units, None, retry=True)[0]


def test_platform_load_respects_the_plans_commit_and_mismatch_halts(tmp_path, small):   # fix review M2
    pf = platform(tmp_path)
    units = S.wording_units(1, S.Settings())
    pf.run_units("s34-A-r1", units, None)
    R.write_plan(tmp_path, "s34-A-r1", units, 2, frozen_commit=D)         # re-planned, nothing adopted
    plan = R.read_plan(tmp_path, "s34-A-r1")
    assert all(pf.load(u) is not None for u in units) and all(pf.load(u, plan) is None for u in units)
    with pytest.raises(S.ChainHalt, match="plan's commit"):
        pf.run_units("s34-A-r1", units, None)                             # the code is still at C


def test_platform_never_returns_a_stale_result_after_a_halt(tmp_path, small, monkeypatch):   # fix review M2
    pf = platform(tmp_path)
    units = S.wording_units(1, S.Settings())
    pf.run_units("s34-A-r1", units, None)
    R.write_plan(tmp_path, "s34-A-r1", units, 2, frozen_commit=D)         # re-planned at D, nothing adopted
    pfd = platform(tmp_path, D)

    def halted(phase, **k):
        raise R.RunHalted("two units in a row failed")
    monkeypatch.setattr(pfd.run, "run_phase", halted)
    out = pfd.run_units("s34-A-r1", units, None)
    assert all(r.get("status") == "not_done" for r in out)                # not the commit-C results


def test_adoption_requires_the_same_platform_header(tmp_path, small):                  # 2nd opinion #2, M1
    pf = platform(tmp_path)
    units = S.wording_units(1, S.Settings())
    pf.run_units("s34-A-r1", units, None)
    same = platform(tmp_path, D)
    assert same.header_diff(units[0]) == [] and same.adopt("s34-A-r1", units[0])[2] is True
    run = runner(tmp_path, ScriptedBackend(TOK, "memory"), hdr={**header(), "code_commit": D, "embedder": "other"})
    other = Platform(tmp_path, run, D, lambda u: (ScriptedBackend(TOK, "memory"), TOK), log=lambda m: None)
    frm, sha, v = other.adopt("s34-A-r1", units[0])
    assert frm == C and v == "header differs: ['embedder']"
    dirty = runner(tmp_path, ScriptedBackend(TOK, "memory"), hdr={**header(), "code_commit": D, "code_dirty": True})
    assert Platform(tmp_path, dirty, D, lambda u: None, log=lambda m: None).header_diff(units[0]) == ["code_dirty"]


def test_verify_outcomes_are_reused_per_payload_and_commit(tmp_path, small):           # 2nd opinion #2, Low 5
    calls = []

    def tools(u):
        calls.append(u.unit_id)
        return ScriptedBackend(TOK, "memory"), TOK
    run = runner(tmp_path, ScriptedBackend(TOK, "memory"))
    pf = Platform(tmp_path, run, C, tools, log=lambda m: None)
    units = S.wording_units(1, S.Settings())
    res = pf.run_units("s34-A-r1", units, None)
    assert pf.checks(res) == (True, 0) and len(calls) == len(units)
    v = pf.verify_finished()
    assert len(calls) == len(units) and all(x["verify_replay"] is True and "audit_hits" in x for x in v.values())
    pf2 = Platform(tmp_path, run, C, tools, log=lambda m: None, verified=v)               # a later session
    pf2.checks(res)
    pf2.verify_finished()
    assert len(calls) == len(units)
    pf3 = Platform(tmp_path, run, D, tools, log=lambda m: None, verified=v)               # another commit: again
    pf3.checks(res)
    assert len(calls) == 2 * len(units)


def test_halted_units_count_as_failed_and_incomplete_memo_entries_are_not_kept(tmp_path, small):  # fix review L5, H1
    pf = platform(tmp_path)
    units = S.wording_units(1, S.Settings())[:2]
    R.write_plan(tmp_path, "s34-A-r1", units, 1, frozen_commit=C)
    pf.run.mark("s34-A-r1", units[0].unit_id, "halted")

    def halted(phase, **k):
        raise R.RunHalted("two units in a row failed")
    real = pf.run.run_phase
    pf.run.run_phase = halted
    out = pf.run_units("s34-A-r1", units, None)
    assert [r.get("status") for r in out] == ["failed", "not_done"]
    pf.run.run_phase = real
    pf.verified[units[1].unit_id] = {"payload_sha256": "x", "commit": C, "verify_replay": "ReplayDivergence"}
    assert units[1].unit_id not in pf.verify_finished()  # no audit outcome: not persisted


def test_adoption_tolerates_not_compat_fields(tmp_path, small):                        # fix review test gap
    pf = platform(tmp_path)
    units = S.wording_units(1, S.Settings())[:1]
    pf.run_units("s34-A-r1", units, None)
    run = runner(tmp_path, ScriptedBackend(TOK, "memory"),
                 hdr={**header(), "code_commit": D, "cpu_model": "another Xeon", "gpu_driver": "x"})
    other = Platform(tmp_path, run, D, lambda u: (ScriptedBackend(TOK, "memory"), TOK), log=lambda m: None)
    assert other.header_diff(units[0]) == [] and other.adopt("s34-A-r1", units[0])[2] is True
