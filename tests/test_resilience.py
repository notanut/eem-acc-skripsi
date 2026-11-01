import os
import random

import numpy as np
import pytest

from eem_acc.utils import (CheckpointManager, Paths, ProgressLogger, ResultStore,
                           atomic_write_bytes, atomic_write_json, clean_tmp_files,
                           run_resumable, set_seed)


# ---------- atomic writes ----------
def test_atomic_write_leaves_no_tmp(tmp_path):
    atomic_write_json(tmp_path / "a.json", {"x": np.int64(3), "y": np.arange(3)})
    assert [p.name for p in tmp_path.iterdir()] == ["a.json"]


def test_failed_write_keeps_old_file_and_no_tmp(tmp_path, monkeypatch):
    target = tmp_path / "a.bin"
    atomic_write_bytes(target, b"old")

    def boom(*a, **k):
        raise OSError("disconnect")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        atomic_write_bytes(target, b"new")
    monkeypatch.undo()
    assert target.read_bytes() == b"old"
    assert [p.name for p in tmp_path.iterdir()] == ["a.bin"]


def test_clean_tmp_files(tmp_path):
    (tmp_path / "x.json.abc.tmp").write_bytes(b"partial")
    assert clean_tmp_files(tmp_path) == 1


# ---------- checkpoints ----------
def test_checkpoint_roundtrip_and_rotation(tmp_path):
    cm = CheckpointManager(tmp_path, keep_last=2)
    for s in range(1, 5):
        cm.save(s, {"arr": np.arange(s), "s": s}, config={"a": 1})
    assert cm.list_steps() == [3, 4]
    ck = cm.load_latest()
    assert ck["step"] == 4 and ck["config"] == {"a": 1}
    assert np.array_equal(ck["state"]["arr"], np.arange(4))


def test_corrupt_newest_falls_back(tmp_path):
    cm = CheckpointManager(tmp_path, keep_last=3)
    cm.save(1, {"v": "good"})
    cm.save(2, {"v": "newer"})
    p2 = cm._path(2)
    p2.write_bytes(p2.read_bytes()[:50])  # truncated, as after a disconnect
    assert cm.load_latest()["state"]["v"] == "good"
    cm._path(1).write_bytes(b"garbage")
    assert cm.load_latest() is None


def test_rng_state_restored(tmp_path):
    cm = CheckpointManager(tmp_path)
    set_seed(7)
    random.random(); np.random.rand()
    cm.save(1, {})
    expected = (random.random(), np.random.rand())
    random.random(); np.random.rand()  # drift away
    cm.load_latest()
    assert (random.random(), np.random.rand()) == expected


# ---------- results & progress ----------
def test_result_store(tmp_path):
    rs = ResultStore(tmp_path)
    rs.save("task/1", {"ok": True})
    assert rs.exists("task/1") and rs.load("task/1") == {"ok": True}
    assert rs.pending(["task/1", "task/2"]) == ["task/2"]
    (tmp_path / "broken.json").write_text("{not json")
    assert [u for u, _ in rs.iter_results()] == ["task/1"]


def test_progress_eta():
    p = ProgressLogger(10, already_done=5, min_interval_s=1e9)
    assert p.eta_seconds() is None  # nothing processed this session yet
    p.update()
    assert p.eta_seconds() is not None and "ETA" in p.line()


# ---------- resumable runner ----------
def _make(crash_at=None):
    calls = []

    def run_unit(u):
        if crash_at is not None and u == crash_at:
            raise RuntimeError("colab disconnected")
        calls.append(u)
        return {"value": int(u) * 2}

    return run_unit, calls


def test_stateless_resume_skips_finished_units(tmp_path):
    ids = [str(i) for i in range(8)]
    fn, calls = _make(crash_at="5")
    with pytest.raises(RuntimeError):
        run_resumable(ids, fn, tmp_path, config={"c": 1})
    assert calls == ["0", "1", "2", "3", "4"]
    fn2, calls2 = _make()
    store = run_resumable(ids, fn2, tmp_path, config={"c": 1})
    assert calls2 == ["5", "6", "7"]
    assert len(list(store.iter_results())) == 8


def test_stateful_resume_restores_state_and_reruns_post_checkpoint_units(tmp_path):
    ids = [str(i) for i in range(6)]
    state = {"memory": []}

    def run_unit(u):
        state["memory"].append(u)  # state grows with each unit, like the EEM store
        return {"mem_size": len(state["memory"])}

    kw = dict(config={"c": 1}, stateful=True, checkpoint_every=2,
              get_state=lambda: list(state["memory"]),
              set_state=lambda s: state.update(memory=list(s)))
    # first attempt crashes on unit 3; the only checkpoint is at step 2 (units 0 and 1)
    seen = []

    def crashing(u):
        if u == "3":
            raise RuntimeError("disconnect")
        seen.append(u)
        return run_unit(u)

    with pytest.raises(RuntimeError):
        run_resumable(ids, crashing, tmp_path, **kw)
    state["memory"].clear()  # fresh Colab process: in-memory state is gone
    store = run_resumable(ids, run_unit, tmp_path, **kw)
    # unit 2 finished after the step-2 checkpoint, so it was re-run; memory stayed consistent
    assert state["memory"] == ids
    assert [store.load(u)["mem_size"] for u in ids] == [1, 2, 3, 4, 5, 6]


def test_config_mismatch_refuses_to_resume(tmp_path):
    fn, _ = _make()
    run_resumable(["0", "1"], fn, tmp_path, config={"alpha": 0.5})
    with pytest.raises(ValueError):
        run_resumable(["0", "1", "2"], fn, tmp_path, config={"alpha": 0.9})


# ---------- paths ----------
def test_paths_ensure(tmp_path):
    p = Paths(tmp_path / "drive").ensure()
    for d in ("checkpoints", "results", "logs", "figures", "datasets", "handoff", "code"):
        assert (p.root / d).is_dir()
    assert p.checkpoints == p.root / "checkpoints"
