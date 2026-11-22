"""S3.3a: unit runner, resume, completeness, failure policy, plans, lease, ledger, verify_replay, audit (spec 6-8)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from eem_acc.stage3.agent import BagOfWordsEmbedder, ScriptedBackend, StubTokenizer
from eem_acc.stage3.llm import runner as R
from eem_acc.stage3.llm.audit import audit_scrub
from eem_acc.stage3.llm.cache import ReplayDivergence, UnitCache, content_hash_of, crc
from eem_acc.stage3.llm.canary import Session
from eem_acc.stage3.llm.fsio import canon, read_jsonl
from eem_acc.stage3.llm.ledger import Ledger, summarise

REPO = Path(__file__).resolve().parents[1]
TOK = StubTokenizer()


class Crash(BaseException):
    """Like a Colab interrupt: not caught by the runner's failure policy."""


class CrashAfter(ScriptedBackend):
    def __init__(self, n, *a, **k):
        super().__init__(*a, **k)
        self.n, self.k = n, 0

    def generate(self, messages, kind, grammar, max_tokens):
        self.k += 1
        if self.k == self.n:
            raise Crash()
        return super().generate(messages, kind, grammar, max_tokens)


class Suffix(ScriptedBackend):
    """Same id, different outputs: a platform whose generations changed."""

    def generate(self, messages, kind, grammar, max_tokens):
        g = super().generate(messages, kind, grammar, max_tokens)
        return type(g)(g.text + " ", g.n_in, g.n_out, g.finish_reason, g.grammar)


def header():
    """A clean code tree is simulated (the real tree may be dirty while tests run; dirty never resumes, tested below)."""
    h = R.make_header({"backend": "scripted"}, "bow-sha1-256", {"python": "3", "numpy": "2", "gpu": "cpu"}, 0, "fp0")
    return {**h, "code_commit": "c" * 40, "code_dirty": False}


def runner(root, backend, session=None, rerun=None, shard=(0, 1), hdr=None):
    ctx = R.RunContext(backend, BagOfWordsEmbedder(), TOK, hdr or header(), session, rerun)
    return R.Runner(Path(root), ctx, Ledger(root, "s1", "rt1", str(shard[0])), shard=shard, log=lambda m: None)


U1 = R.Unit("pilot", "rq1", "dev", "rq1", 1000, "C2", n_tasks=6)
U3 = R.Unit("pilot", "rq3", "dev", "rq3", 5000, "C2", "acc")


def plan(root, units, **kw):
    return R.write_plan(root, "pilot", units, 1, **kw)


def strategy(u):
    return "docs" if u.condition == "C0" else "memory"


def payload(root, u):
    return R.split_result(json.loads((Path(root) / "units" / u.phase / f"{u.unit_id}.json").read_text()))


@pytest.mark.parametrize("u", [U1, U3], ids=["rq1", "rq3"])
def test_interrupted_and_resumed_equals_uninterrupted(tmp_path, u):
    a, b = tmp_path / "a", tmp_path / "b"
    plan(a, [u])
    plan(b, [u])
    runner(a, ScriptedBackend(TOK, strategy(u))).run_phase("pilot")
    rb = runner(b, CrashAfter(9, TOK, strategy(u)))
    with pytest.raises(Crash):
        rb.run_phase("pilot")
    assert not rb.is_done(u) and (b / "cache" / f"{u.unit_id}.jsonl").exists()
    runner(b, ScriptedBackend(TOK, strategy(u))).run_phase("pilot")
    pa, ra = payload(a, u)
    pb, rbm = payload(b, u)
    assert pa == pb and ra["payload_sha256"] == rbm["payload_sha256"]
    assert rbm["resumed"] and rbm["calls_replayed"] >= 8 and not ra["resumed"]
    assert ra["cache_content_hash"] == rbm["cache_content_hash"]
    assert any(r.get("kind") == "embed" for r in UnitCache(b / "cache" / f"{u.unit_id}.jsonl", {}, "v",
                                                           readonly=True).records)


def test_payload_independent_of_hashseed(tmp_path):
    code = ("import sys,json;sys.path[:0]=['src','tests'];from pathlib import Path;"
            "import test_s33_runner as T;r=Path(sys.argv[1]);T.plan(r,[T.U1]);"
            "T.runner(r,T.ScriptedBackend(T.TOK,'memory')).run_phase('pilot');"
            "print(T.payload(r,T.U1)[1]['payload_sha256'])")
    outs = set()
    for hs in ("0", "77"):
        root = tmp_path / f"h{hs}"
        r = subprocess.run([sys.executable, "-c", code, str(root)], cwd=REPO, capture_output=True, text=True,
                           env=dict(os.environ, PYTHONHASHSEED=hs), check=True)
        outs.add(r.stdout.strip().splitlines()[-1])
    assert len(outs) == 1


def test_done_requires_complete_cache_and_repairs_lost_tail(tmp_path):
    plan(tmp_path, [U1])
    r = runner(tmp_path, ScriptedBackend(TOK, "memory"))
    r.run_phase("pilot")
    sha = payload(tmp_path, U1)[1]["payload_sha256"]
    cp = tmp_path / "cache" / f"{U1.unit_id}.jsonl"
    lines = cp.read_text().splitlines()
    cp.write_text("\n".join(lines[:-6]) + "\n")                  # Drive lost the cache tail after the result
    assert not r.is_done(U1)
    r.run_phase("pilot")
    p, run = payload(tmp_path, U1)
    assert run["payload_sha256"] == sha and run["resumed"] and r.is_done(U1)


def test_result_written_but_manifest_missing_counts_as_done(tmp_path):
    plan(tmp_path, [U1])
    r = runner(tmp_path, ScriptedBackend(TOK, "memory"))
    r.run_phase("pilot")
    for f in (tmp_path / "manifest").rglob("*.jsonl"):
        f.unlink()
    assert r.is_done(U1)
    assert r.run_phase("pilot")["ran"] == 0


def test_stamp_mismatch_archives_and_restarts(tmp_path):
    plan(tmp_path, [U1])
    with pytest.raises(Crash):
        runner(tmp_path, CrashAfter(9, TOK, "memory")).run_phase("pilot")
    h = {**header(), "code_commit": "other"}
    runner(tmp_path, ScriptedBackend(TOK, "memory"), hdr=h).run_phase("pilot")
    p, run = payload(tmp_path, U1)
    assert not run["resumed"] and run["restarts"] and run["restarts"][0].startswith("stamp")
    idx = read_jsonl(tmp_path / "cache" / "archive" / "index.jsonl")
    assert idx[0]["reason"] == "stamp" and "code_commit" in idx[0]["fields"]


def test_test_phase_refuses_other_commit(tmp_path):
    R.write_plan(tmp_path, "test", [R.Unit("test", "rq1", "test", "rq1", 2000, "C2", n_tasks=4)], 1,
                 frozen_commit="abc")
    with pytest.raises(R.TestPhaseCommitMismatch):
        runner(tmp_path, ScriptedBackend(TOK, "memory")).run_phase("test")


def test_divergence_restarts_once_then_halts(tmp_path, monkeypatch):
    plan(tmp_path, [U1])
    with pytest.raises(Crash):
        runner(tmp_path, CrashAfter(9, TOK, "memory")).run_phase("pilot")
    cp = tmp_path / "cache" / f"{U1.unit_id}.jsonl"
    lines = cp.read_text().splitlines()
    rec = json.loads(lines[3])
    rec["key"] = "f" * 64
    rec["crc"] = crc(rec)
    lines[3] = canon(rec)
    cp.write_text("\n".join(lines) + "\n")
    runner(tmp_path, ScriptedBackend(TOK, "memory")).run_phase("pilot")
    p, run = payload(tmp_path, U1)
    assert run["restarts"][0].startswith("divergence") and not run["resumed"]
    orig = UnitCache.call

    def always(self, kind, key, compute, extra):
        raise ReplayDivergence(self.pos, {"kind": kind, "key": key}, kind, key)
    monkeypatch.setattr(UnitCache, "call", always)
    U = R.Unit("pilot", "rq1", "dev", "rq1", 1001, "C2", n_tasks=4)
    plan(tmp_path, [U1, U], ) if False else R.write_plan(tmp_path, "pilot", [U1, U], 2)
    with pytest.raises(R.RunHalted):
        runner(tmp_path, ScriptedBackend(TOK, "memory")).run_phase("pilot")
    monkeypatch.setattr(UnitCache, "call", orig)


def test_two_identical_exceptions_mark_failed_and_are_skipped(tmp_path):
    class Boom(ScriptedBackend):
        def generate(self, *a, **k):
            raise ValueError("boom")
    plan(tmp_path, [U1])
    r = runner(tmp_path, Boom(TOK, "memory"))
    r.run_phase("pilot")
    assert U1.unit_id in r.failed_units("pilot")
    fails = read_jsonl(tmp_path / "failures" / "pilot.jsonl")
    assert len(fails) == 2 and fails[0]["error"] == "ValueError: boom"
    assert r.run_phase("pilot")["ran"] == 0


def test_resume_mismatch_restarts_and_second_enters_replay_only(tmp_path):
    units = [U1, R.Unit("pilot", "rq1", "dev", "rq1", 1001, "C2", n_tasks=4),
             R.Unit("pilot", "rq1", "dev", "rq1", 1002, "C2", n_tasks=4)]
    plan(tmp_path, units)
    for u in units:                                             # every unit interrupted
        R.write_plan(tmp_path, "pilot", [u], 1) if False else None
    for u in units:
        r = runner(tmp_path, CrashAfter(7, TOK, "memory"))
        with pytest.raises(Crash):
            r.run_unit(u, R.read_plan(tmp_path, "pilot"))
    sess = Session("fresh", 0, "pass")
    calls = []
    r = runner(tmp_path, Suffix(TOK, "memory"), session=sess, rerun=lambda: (calls.append(1), (0, "pass", "fp0"))[1])
    r.run_phase("pilot")
    assert sess.mismatches == 2 and sess.replay_only_mode and calls == [1]
    runs = [payload(tmp_path, u)[1] for u in units]
    assert runs[0]["restarts"][0].startswith("resume_mismatch") and not runs[0]["resumed"]
    assert runs[1]["restarts"][0].startswith("resume_mismatch") and not runs[1]["resumed"]   # the trigger restarts
    assert runs[2]["resumed"] and runs[2]["replay_only"]                                    # later: replay only


def test_backend_busy_refused(tmp_path):
    class Busy(ScriptedBackend):
        busy = True
    plan(tmp_path, [U1])
    from eem_acc.stage3.llm.llama_backend import BackendBusy
    with pytest.raises(BackendBusy):
        runner(tmp_path, Busy(TOK, "memory")).run_unit(U1, R.read_plan(tmp_path, "pilot"))


def test_lease_refuses_second_runtime(tmp_path):
    t = [1000.0]
    a = R.Lease(tmp_path, "rtA", clock=lambda: t[0])
    b = R.Lease(tmp_path, "rtB", clock=lambda: t[0])
    a.acquire()
    with pytest.raises(R.LeaseHeld):
        b.acquire()
    t[0] += 500
    b.acquire()                                                  # stale lease taken over
    a2 = R.Lease(tmp_path, "rtA", clock=lambda: t[0])
    with pytest.raises(R.LeaseHeld):
        a2.acquire()


def test_plan_rules_and_unit_ids(tmp_path):
    R.write_plan(tmp_path, "pilot", [U1], 1)
    with pytest.raises(ValueError, match="new plan version"):
        R.write_plan(tmp_path, "pilot", [U1, U3], 1)
    R.write_plan(tmp_path, "pilot", [U1, U3], 2)
    with pytest.raises(ValueError, match="back"):
        R.write_plan(tmp_path, "pilot", [U1], 1)
    with pytest.raises(ValueError, match="duplicate"):
        R.write_plan(tmp_path, "x", [U1, U1], 1)
    ids = {R.Unit("dev", "rq1", "dev", "rq1", 1001, "C2", tag=f"cfg{i}").unit_id for i in range(3)}
    assert len(ids) == 3 and U3.unit_id == "pilot-rq3-s5000-C2-acc"
    assert R.Unit("test", "rq1", "test", "sp", 3000, "C2").unit_id == "test-rq1-sps3000-C2"
    assert R.Unit.from_json(U1.to_json()) == U1


def test_test_phase_order_d63():
    units = R.test_phase_units(list(range(2000, 2014)), list(range(4000, 4010)), 1)
    rq1 = [u for u in units if u.rq == "rq1"]
    rq3 = [u for u in units if u.rq == "rq3"]
    assert len({u.seed for u in rq1}) == 14 and len({u.seed for u in rq3}) == 10
    assert sum(u.condition == "C2-OU" for u in rq1) == 5
    assert {u.seed for u in rq1 if u.condition == "C2-OU"} == set(range(2000, 2005))
    seq = []
    for u in units:
        key = (u.rq, u.seed)
        if not seq or seq[-1] != key:
            seq.append(key)
    kinds = "".join("1" if k[0] == "rq1" else "3" for k in seq)
    assert kinds.startswith("113" * 7) and kinds == "113" * 7 + "333"
    again = R.test_phase_units(list(range(2000, 2014)), list(range(4000, 4010)), 1)
    assert [u.unit_id for u in again] == [u.unit_id for u in units]
    other = R.test_phase_units(list(range(2000, 2014)), list(range(4000, 4010)), 2)
    assert [u.unit_id for u in other] != [u.unit_id for u in units]   # the shuffle depends on the plan version
    assert len({u.unit_id for u in units}) == len(units)


def test_sharding_splits_whole_units(tmp_path):
    units = [R.Unit("pilot", "rq1", "dev", "rq1", s, "C2", n_tasks=3) for s in (1000, 1001, 1002)]
    plan(tmp_path, units)
    r0 = runner(tmp_path, ScriptedBackend(TOK, "memory"), shard=(0, 2))
    r1 = runner(tmp_path, ScriptedBackend(TOK, "memory"), shard=(1, 2))
    assert r0.run_phase("pilot")["total"] == 2 and r1.run_phase("pilot")["total"] == 1
    assert all(r0.is_done(u) for u in units)
    assert (tmp_path / "manifest" / "pilot" / "0.jsonl").exists() and (tmp_path / "manifest" / "pilot" / "1.jsonl").exists()


def test_ledger_gpu_hours_from_heartbeats(tmp_path):
    t = [0.0]
    a = Ledger(tmp_path, "s1", "rtA", "0", clock=lambda: t[0])
    b = Ledger(tmp_path, "s1", "rtA", "1", clock=lambda: t[0])     # second process, same runtime
    a.start(100.0)
    b.start(None)
    t[0] = 1800.0
    a.event("unit", unit="u", calls_fresh=10, calls_replayed=2, llm_seconds=50.0)
    t[0] = 3600.0
    b.heartbeat()
    c = Ledger(tmp_path, "s2", "rtB", "0", clock=lambda: t[0])
    c.start(90.0)
    t[0] = 5400.0
    c.event("discarded", unit="u", reason="epoch", calls=7)
    s = summarise(tmp_path)
    assert s["gpu_hours"] == pytest.approx(1.0 + 0.5) and s["calls_fresh"] == 10
    assert s["seconds_per_fresh_call"] == pytest.approx(5.0) and s["discarded_calls"] == {"epoch": 7}


@pytest.mark.parametrize("u", [U1, U3], ids=["rq1", "rq3"])
def test_verify_replay(tmp_path, u):
    plan(tmp_path, [u])
    runner(tmp_path, ScriptedBackend(TOK, strategy(u))).run_phase("pilot")
    key_source = ScriptedBackend(TOK, strategy(u))
    assert R.verify_replay(tmp_path, "pilot", u.unit_id, key_source, TOK)
    res = json.loads((tmp_path / "units" / "pilot" / f"{u.unit_id}.json").read_text())
    res["success_rate"] = -1.0                                   # a payload that does not match its cache
    (tmp_path / "units" / "pilot" / f"{u.unit_id}.json").write_text(json.dumps(res))
    assert not R.verify_replay(tmp_path, "pilot", u.unit_id, key_source, TOK)


def test_verify_replay_never_generates(tmp_path):
    plan(tmp_path, [U1])
    runner(tmp_path, ScriptedBackend(TOK, "memory")).run_phase("pilot")
    cp = tmp_path / "cache" / f"{U1.unit_id}.jsonl"
    lines = cp.read_text().splitlines()
    cp.write_text("\n".join(lines[:-3]) + "\n")
    before = cp.read_text()
    with pytest.raises(ReplayDivergence):
        R.verify_replay(tmp_path, "pilot", U1.unit_id, ScriptedBackend(TOK, "memory"), TOK)
    assert cp.read_text() == before


def test_scrub_audit_finds_planted_leak(tmp_path):
    plan(tmp_path, [U1])
    runner(tmp_path, ScriptedBackend(TOK, "memory")).run_phase("pilot")
    res = json.loads((tmp_path / "units" / "pilot" / f"{U1.unit_id}.json").read_text())
    clean = audit_scrub([res])
    assert clean["hits"] == [] and clean["units"] == 1
    from eem_acc.stage3.apisuite import sample_stream
    from eem_acc.stage3.llm.audit import stream_entities
    sp = sample_stream(1000, "dev")
    ent = sorted(stream_entities({**sp, "tasks": sp["tasks"][:6]}))[0]
    res["final_notes"][0]["reflection"] = f"use {ent}"
    leak = audit_scrub([res])
    assert leak["hits"] and leak["hits_by_condition"]["C2"] == 1 and leak["hits"][0]["value"] == ent


# ----------------------------------------------------------------------------------------------- code review fixes
def test_busy_backend_halts_run_phase_without_failing_units(tmp_path):                    # review H1
    class Busy(ScriptedBackend):
        busy = True
    units = [R.Unit("pilot", "rq1", "dev", "rq1", s, "C2", n_tasks=3) for s in (1000, 1001)]
    plan(tmp_path, units)
    from eem_acc.stage3.llm.llama_backend import BackendBusy
    with pytest.raises(BackendBusy):
        runner(tmp_path, Busy(TOK, "memory")).run_phase("pilot")
    assert not (tmp_path / "manifest").exists() or not any(
        e.get("status") == "failed" for f in (tmp_path / "manifest").rglob("*.jsonl") for e in read_jsonl(f))
    assert not (tmp_path / "cache" / "archive").exists()


@pytest.mark.parametrize("n", [1, 3, 25, 60])
def test_crash_at_several_points_resumes_identically(tmp_path, n):
    a, b = tmp_path / "a", tmp_path / "b"
    plan(a, [U1])
    plan(b, [U1])
    runner(a, ScriptedBackend(TOK, "memory")).run_phase("pilot")
    try:
        runner(b, CrashAfter(n, TOK, "memory")).run_phase("pilot")
    except Crash:
        pass
    runner(b, ScriptedBackend(TOK, "memory")).run_phase("pilot")
    assert payload(a, U1)[0] == payload(b, U1)[0]


def test_interrupted_resumed_in_subprocess_with_other_hashseed(tmp_path):
    root = tmp_path / "r"
    plan(root, [U1])
    with pytest.raises(Crash):
        runner(root, CrashAfter(12, TOK, "memory")).run_phase("pilot")
    code = ("import sys;sys.path[:0]=['src','tests'];from pathlib import Path;import test_s33_runner as T;"
            "r=Path(sys.argv[1]);T.runner(r,T.ScriptedBackend(T.TOK,'memory')).run_phase('pilot');"
            "print(T.payload(r,T.U1)[1]['payload_sha256'])")
    out = subprocess.run([sys.executable, "-c", code, str(root)], cwd=REPO, capture_output=True, text=True,
                         env=dict(os.environ, PYTHONHASHSEED="99"), check=True).stdout.strip().splitlines()[-1]
    ref = tmp_path / "ref"
    plan(ref, [U1])
    runner(ref, ScriptedBackend(TOK, "memory")).run_phase("pilot")
    assert out == payload(ref, U1)[1]["payload_sha256"] and payload(root, U1)[1]["resumed"]


@pytest.mark.parametrize("field,value", [("gpu", "A100"), ("platform_epoch", 3), ("platform_fingerprint", "fpX"),
                                         ("embedder", "other"), ("backend", "other-model"), ("code_dirty", True)])
def test_stamp_fields_block_resume(tmp_path, field, value):                           # review M13, M5, M6
    plan(tmp_path, [U1])
    with pytest.raises(Crash):
        runner(tmp_path, CrashAfter(9, TOK, "memory")).run_phase("pilot")
    runner(tmp_path, ScriptedBackend(TOK, "memory"), hdr={**header(), field: value}).run_phase("pilot")
    p, run = payload(tmp_path, U1)
    assert not run["resumed"] and run["restarts"][0].startswith("stamp") and field in run["restarts"][0]
    disc = [e for f in (tmp_path / "ledger").glob("*.jsonl") for e in read_jsonl(f) if e["event"] == "discarded"]
    assert disc and disc[0]["reason"] == "stamp"


def test_test_phase_refuses_dirty_tree(tmp_path):
    R.write_plan(tmp_path, "test", [R.Unit("test", "rq1", "test", "rq1", 2000, "C2", n_tasks=4)], 1,
                 frozen_commit="c" * 40)
    with pytest.raises(R.TestPhaseCommitMismatch):
        runner(tmp_path, ScriptedBackend(TOK, "memory"), hdr={**header(), "code_dirty": True}).run_phase("test")
    runner(tmp_path, ScriptedBackend(TOK, "memory")).run_phase("test")          # clean, frozen commit: runs


def test_repair_after_stamp_change_runs_once(tmp_path):                                # review M3
    plan(tmp_path, [U1])
    runner(tmp_path, ScriptedBackend(TOK, "memory")).run_phase("pilot")
    p0, r0 = payload(tmp_path, U1)
    cp = tmp_path / "cache" / f"{U1.unit_id}.jsonl"
    cp.write_text("\n".join(cp.read_text().split("\n")[:-6]) + "\n")
    runner(tmp_path, ScriptedBackend(TOK, "memory"), hdr={**header(), "platform_epoch": 1}).run_phase("pilot")
    p1, r1 = payload(tmp_path, U1)
    assert r1["calls_fresh"] == r0["calls_fresh"] and not r1["resumed"]           # one full run, not two
    assert any(x.startswith("repair: from scratch") for x in r1["restarts"])


def test_repair_resumed_with_different_payload_reruns(tmp_path):
    plan(tmp_path, [U1])
    runner(tmp_path, ScriptedBackend(TOK, "memory")).run_phase("pilot")
    rp = tmp_path / "units" / "pilot" / f"{U1.unit_id}.json"
    res = json.loads(rp.read_text())
    res["tampered"] = True                                       # stored result no longer what the cache gives
    res["run"]["payload_sha256"] = R.payload_sha(R.split_result(res)[0])
    rp.write_text(json.dumps(res))
    cp = tmp_path / "cache" / f"{U1.unit_id}.jsonl"
    cp.write_text("\n".join(cp.read_text().split("\n")[:-6]) + "\n")
    runner(tmp_path, ScriptedBackend(TOK, "memory")).run_phase("pilot")
    p, run = payload(tmp_path, U1)
    assert run["restarts"][0].startswith("repair_mismatch") and "tampered" not in p
    assert len(run["restarts"]) == 1 and not run["resumed"]     # then an ordinary from-scratch run


def test_is_done_checks(tmp_path):                                                    # review M14, M15
    plan(tmp_path, [U1])
    r = runner(tmp_path, ScriptedBackend(TOK, "memory"))
    r.run_phase("pilot")
    assert r.is_done(U1) and r.is_done(U1, full=True)
    cp = tmp_path / "cache" / f"{U1.unit_id}.jsonl"
    lines = cp.read_text().split("\n")
    rec = json.loads(lines[2])
    rec["text"] = (rec["text"] or "") + "X"
    rec["crc"] = crc(rec)
    lines[2] = canon(rec)
    cp.write_text("\n".join(lines))
    assert r.is_done(U1) and not r.is_done(U1, full=True)       # content changes only the full check sees
    rp = tmp_path / "units" / "pilot" / f"{U1.unit_id}.json"
    res = json.loads(rp.read_text())
    res["success_rate"] = 0.123
    rp.write_text(json.dumps(res))
    assert not r.is_done(U1)                                     # payload no longer matches its recorded hash


def test_verify_replay_detects_a_changed_cache(tmp_path):                             # review M21
    plan(tmp_path, [U1])
    runner(tmp_path, ScriptedBackend(TOK, "memory")).run_phase("pilot")
    cp = tmp_path / "cache" / f"{U1.unit_id}.jsonl"
    lines = cp.read_text().split("\n")
    j = next(i for i, ln in enumerate(lines) if ln and json.loads(ln).get("kind") == "action")
    rec = json.loads(lines[j])
    rec["text"] = '{"action": "finish", "args": {"summary": "x"}}'
    rec["crc"] = crc(rec)
    lines[j] = canon(rec)
    cp.write_text("\n".join(lines))
    try:
        ok = R.verify_replay(tmp_path, "pilot", U1.unit_id, ScriptedBackend(TOK, "memory"), TOK)
    except Exception:                                            # a divergence or leftovers also mean "not verified"
        ok = False
    assert not ok


def test_platform_changed_updates_header_and_restarts(tmp_path):                      # review M22-24
    units = [R.Unit("pilot", "rq1", "dev", "rq1", s, "C2", n_tasks=4) for s in (1000, 1001)]
    plan(tmp_path, units)
    for u in units:
        with pytest.raises(Crash):
            runner(tmp_path, CrashAfter(7, TOK, "memory")).run_unit(u, R.read_plan(tmp_path, "pilot"))
    sess = Session("fresh", 0, "pass", "fp0")
    r = runner(tmp_path, Suffix(TOK, "memory"), session=sess, rerun=lambda: (5, "new", "fp5"))
    r.run_phase("pilot")
    assert r.ctx.header["platform_epoch"] == 5 and r.ctx.header["platform_fingerprint"] == "fp5"
    run2 = payload(tmp_path, units[1])[1]
    assert any(x.startswith("platform_changed") for x in run2["restarts"]) and not sess.replay_only_mode


def test_two_units_in_a_row_same_error_halts_and_attempt_cap(tmp_path):              # D130
    class Boom(ScriptedBackend):
        def generate(self, *a, **k):
            raise ValueError("cuda context broken")
    units = [R.Unit("pilot", "rq1", "dev", "rq1", s, "C2", n_tasks=3) for s in (1000, 1001, 1002)]
    plan(tmp_path, units)
    with pytest.raises(R.RunHalted, match="two units in a row"):
        runner(tmp_path, Boom(TOK, "memory")).run_phase("pilot")
    assert not (tmp_path / "units" / "pilot" / f"{units[2].unit_id}.json").exists()
    n = {"k": 0}

    class Varying(ScriptedBackend):
        def generate(self, *a, **k):
            n["k"] += 1
            raise ValueError(f"error at address {n['k']}")
    root = tmp_path / "v"
    plan(root, [U1])
    r = runner(root, Varying(TOK, "memory"))
    r.run_phase("pilot")
    assert U1.unit_id in r.failed_units("pilot") and n["k"] == R.MAX_ATTEMPTS


def test_heartbeat_and_lease_while_running(tmp_path):                                 # review M4
    plan(tmp_path, [U1])
    r = runner(tmp_path, ScriptedBackend(TOK, "memory"))
    r.lease = R.Lease(tmp_path, "rt1")
    r.heartbeat_seconds = 0.05
    r.run_phase("pilot")
    beats = [e for f in (tmp_path / "ledger").glob("*.jsonl") for e in read_jsonl(f) if e["event"] == "heartbeat"]
    assert len(beats) >= 3 and json.loads((tmp_path / "lease.json").read_text())["runtime"] == "rt1"
    other = runner(tmp_path, ScriptedBackend(TOK, "memory"))
    other.lease = R.Lease(tmp_path, "rt2")
    with pytest.raises(R.LeaseHeld):
        other.run_phase("pilot")


def test_ledger_counts_from_vm_boot(tmp_path):
    led = Ledger(tmp_path, "s", "rt", "0", clock=lambda: 1000.0)
    led.start(10.0, vm_boot=400.0)
    led.heartbeat()
    assert summarise(tmp_path)["gpu_hours"] == pytest.approx(600 / 3600, abs=1e-4)


def test_verify_replay_checks_cache_content_not_only_payload(tmp_path):
    plan(tmp_path, [U1])
    runner(tmp_path, ScriptedBackend(TOK, "memory")).run_phase("pilot")
    cp = tmp_path / "cache" / f"{U1.unit_id}.jsonl"
    lines = cp.read_text().split("\n")
    rec = json.loads(lines[1])
    from eem_acc.stage3.llm.cache import zpack
    rec["prompt_z"] = zpack("a different stored prompt")         # payload unaffected, cache content changed
    rec["crc"] = crc(rec)
    lines[1] = canon(rec)
    cp.write_text("\n".join(lines))
    assert not R.verify_replay(tmp_path, "pilot", U1.unit_id, ScriptedBackend(TOK, "memory"), TOK)
