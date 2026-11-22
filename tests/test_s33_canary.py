"""S3.3a: determinism canary, platform epochs, D61/D66 session logic (spec section 5)."""
import json
import threading

import pytest

from eem_acc.stage3.agent.backend import Gen
from eem_acc.stage3.llm.canary import (CanaryRepeatMismatch, EpochStore, FileLock, LockTimeout, SessionNondeterministic,
                                       Session, run_canary)

P = [{"messages": [{"role": "user", "content": f"p{i}"}], "n_tokens": 100 * (i + 1)} for i in range(8)]


def gen_fn(tag=""):
    def g(p):
        t = p["messages"][0]["content"] + tag
        return Gen(t, p["n_tokens"], 5, "stop")
    return g


def test_canary_runs_all_and_repeats_longest_two():
    seen = []

    def g(p):
        seen.append(p["n_tokens"])
        return Gen("x", p["n_tokens"], 1, "stop")
    outs = run_canary(g, P)
    assert len(outs) == 8 and len(seen) == 10 and seen[-2:] == [800, 700]


def test_within_session_repeat_mismatch_halts():
    n = {"k": 0}

    def flaky(p):
        n["k"] += 1
        return Gen("x" if n["k"] <= 8 else "y", p["n_tokens"], 1, "stop")
    with pytest.raises(CanaryRepeatMismatch):
        run_canary(flaky, P)


def test_full_gen_compared_not_only_text():
    n = {"k": 0}

    def g(p):
        n["k"] += 1
        return Gen("same", p["n_tokens"], 1 if n["k"] <= 8 else 2, "stop")
    with pytest.raises(CanaryRepeatMismatch):
        run_canary(g, P)


def test_epochs_pass_new_returned_and_per_path(tmp_path):
    st = EpochStore(tmp_path)
    a = run_canary(gen_fn(), P)
    b = run_canary(gen_fn("!"), P)
    assert st.resolve("fresh", a, {"gpu": "T4"}, "sha") == (0, "new")
    assert st.resolve("fresh", a, {"gpu": "T4"}, "sha") == (0, "pass")
    assert st.resolve("fresh", b, {"gpu": "T4"}, "sha") == (1, "new")
    assert st.resolve("fresh", a, {"gpu": "T4"}, "sha") == (0, "returned")
    assert st.resolve("snapshot", a, {"gpu": "T4"}, "sha") == (0, "new")     # its own series
    assert st.resolve("fresh", a, {"gpu": "T4"}, "other-prompts") == (2, "new")
    data = json.loads((tmp_path / "canary" / "epochs.json").read_text())
    assert [e["epoch"] for e in data["fresh"]] == [0, 1, 2] and data["fresh"][0]["env"] == {"gpu": "T4"}


def test_concurrent_registration_no_duplicate_epochs(tmp_path):
    outs = [run_canary(gen_fn(str(i % 2)), P) for i in range(8)]
    res = []

    def reg(o):
        res.append(EpochStore(tmp_path).resolve("fresh", o, {}, "sha"))
    th = [threading.Thread(target=reg, args=(o,)) for o in outs]
    for t in th:
        t.start()
    for t in th:
        t.join()
    data = json.loads((tmp_path / "canary" / "epochs.json").read_text())
    assert len(data["fresh"]) == 2 and sorted({e for e, _ in res}) == [0, 1]


def test_lock_timeout_and_stale_break(tmp_path):
    lk = tmp_path / "x.lock"
    lk.write_text("1")
    with pytest.raises(LockTimeout):
        with FileLock(lk, timeout=0.3, stale=1e9):
            pass
    import os
    import time
    os.utime(lk, (time.time() - 1000, time.time() - 1000))
    with FileLock(lk, timeout=0.3, stale=300):
        assert lk.exists()
    assert not lk.exists()


def test_session_d61_d66_d131():
    s = Session("fresh", 0, "pass", "fp")
    calls = []
    assert s.on_resume_mismatch(lambda: (calls.append(1), "fp")[1]) == "restart"
    assert not s.replay_only_mode and calls == []                      # first mismatch: restart only
    assert s.on_resume_mismatch(lambda: (calls.append(1), "fp")[1]) == "restart"
    assert s.replay_only_mode and calls == [1]                         # second: same fingerprint -> D66
    s2 = Session("fresh", 0, "returned", "fp")                         # also in a returned epoch
    s2.on_resume_mismatch(lambda: "fp")
    with pytest.raises(SessionNondeterministic):                       # D131: differs within the session -> halt
        s2.on_resume_mismatch(lambda: "other")
    assert not s2.replay_only_mode and s2.epoch == 0
    s3 = Session("fresh", 0, "returned", "fp")
    s3.on_resume_mismatch(lambda: "fp")
    s3.on_resume_mismatch(lambda: "fp")
    assert s3.replay_only_mode
