"""S3.3a: per-unit replay cache (spec section 4)."""
import json

import pytest

from eem_acc.stage3.agent.backend import Gen
from eem_acc.stage3.llm.cache import (CacheError, CachedBackend, CachedEmbedder, LeftoverRecords, ReplayDivergence,
                                      ResumeMismatch, UnitCache, archive, content_hash_of, crc, read_header,
                                      zunpack)
from eem_acc.stage3.llm.fsio import canon


class Echo:
    id = "echo"

    def __init__(self, suffix=""):
        self.calls, self.suffix = 0, suffix

    def generate(self, messages, kind, grammar, max_tokens):
        self.calls += 1
        t = messages[-1]["content"] + self.suffix
        return Gen(t, len(t), 3, "stop", grammar)


class Vec:
    id = "vec"

    def __init__(self, bump=0.0):
        self.calls, self.bump = 0, bump

    def embed(self, text):
        self.calls += 1
        return (len(text) / 100 + self.bump, 0.123456789)


M = lambda t: [{"role": "user", "content": t}]


def test_record_format_crc_and_header(tmp_path):
    p = tmp_path / "u.jsonl"
    c = UnitCache(p, {"a": 1}, "w1")
    b = CachedBackend(Echo(), c)
    g = b.generate(M("hi"), "action", None, 10)
    assert g.text == "hi" and g.n_in == 2
    lines = [json.loads(x) for x in p.read_text().splitlines()]
    assert lines[0]["type"] == "header" and lines[0]["a"] == 1 and lines[0]["writer"] == "w1"
    r = lines[1]
    assert r["i"] == 0 and r["crc"] == crc(r) and r["writer"] == "w1" and zunpack(r["prompt_z"]) == canon(M("hi"))
    expect = __import__("hashlib").sha256(canon({k: v for k, v in r.items() if k != "crc"}).encode()).hexdigest()[:16]
    assert r["crc"] == expect
    assert read_header(p)["a"] == 1


def test_strict_replay_and_divergence(tmp_path):
    p = tmp_path / "u.jsonl"
    c = UnitCache(p, {}, "w1")
    b = CachedBackend(Echo(), c)
    for t in ("a", "b", "c"):
        b.generate(M(t), "action", None, 10)
    inner = Echo()
    c2 = UnitCache(p, {}, "w2", resume_check=False)
    b2 = CachedBackend(inner, c2)
    assert b2.generate(M("a"), "action", None, 10).text == "a" and inner.calls == 0
    with pytest.raises(ReplayDivergence):
        b2.generate(M("X"), "action", None, 10)


def test_past_the_end_appends_and_leftovers_detected(tmp_path):
    p = tmp_path / "u.jsonl"
    b = CachedBackend(Echo(), UnitCache(p, {}, "w1"))
    b.generate(M("a"), "action", None, 10)
    b.generate(M("b"), "action", None, 10)
    c2 = UnitCache(p, {}, "w2", resume_check=False)
    b2 = CachedBackend(Echo(), c2)
    b2.generate(M("a"), "action", None, 10)
    with pytest.raises(LeftoverRecords):
        c2.finish()
    b2.generate(M("b"), "action", None, 10)
    b2.generate(M("c"), "action", None, 10)                     # appended by w2
    c2.finish()
    c3 = UnitCache(p, {}, "w3", readonly=True)
    assert [r["writer"] for r in c3.records] == ["w1", "w1", "w2"]


def test_resume_check_last_two_records(tmp_path):
    p = tmp_path / "u.jsonl"
    b = CachedBackend(Echo(), UnitCache(p, {}, "w1"))
    for t in "abc":
        b.generate(M(t), "action", None, 10)
    inner = Echo()
    b2 = CachedBackend(inner, UnitCache(p, {}, "w2"))
    for t in "abc":
        b2.generate(M(t), "action", None, 10)
    assert inner.calls == 2                                      # only the last two records are regenerated
    b3 = CachedBackend(Echo(suffix="!"), UnitCache(p, {}, "w3"))
    b3.generate(M("a"), "action", None, 10)
    with pytest.raises(ResumeMismatch) as e:
        b3.generate(M("b"), "action", None, 10)
    assert e.value.field == "text"


def test_torn_last_line_dropped_mid_corruption_rejected(tmp_path):
    p = tmp_path / "u.jsonl"
    b = CachedBackend(Echo(), UnitCache(p, {}, "w1"))
    for t in "abc":
        b.generate(M(t), "action", None, 10)
    raw = p.read_text()
    p.write_text(raw[:-25])                                      # interrupted last write
    c = UnitCache(p, {}, "w2", resume_check=False)
    assert c.n_records() == 2 and p.read_text().count("\n") == 4  # header, 2 records, resume line
    lines = raw.splitlines()
    rec = json.loads(lines[1])
    rec["text"] = "tampered"
    lines[1] = canon(rec)
    p.write_text("\n".join(lines) + "\n")
    with pytest.raises(CacheError, match="crc"):
        UnitCache(p, {}, "w3")


def test_index_gap_and_foreign_writer_rejected(tmp_path):
    p = tmp_path / "u.jsonl"
    b = CachedBackend(Echo(), UnitCache(p, {}, "w1"))
    for t in "ab":
        b.generate(M(t), "action", None, 10)
    lines = p.read_text().splitlines()
    p.write_text("\n".join([lines[0], lines[2]]) + "\n")
    with pytest.raises(CacheError, match="index gap"):
        UnitCache(p, {}, "w2")
    rec = json.loads(lines[2])
    rec["writer"], rec["i"] = "zombie", 1
    rec["crc"] = crc(rec)
    p.write_text("\n".join([lines[0], lines[1], canon(rec)]) + "\n")
    with pytest.raises(CacheError, match="foreign writer"):
        UnitCache(p, {}, "w2")


def test_embeddings_recorded_memoised_and_replayed(tmp_path):
    p = tmp_path / "u.jsonl"
    v = Vec()
    e = CachedEmbedder(v, UnitCache(p, {}, "w1"))
    a = e.embed("hello")
    assert e.embed("hello") == a and v.calls == 1               # memo within the unit
    assert a == (0.05, 0.12345679)                              # rounded to 1e-8
    v2 = Vec(bump=1e-9)                                          # a different platform's float noise
    e2 = CachedEmbedder(v2, UnitCache(p, {}, "w2", resume_check=False))
    assert e2.embed("hello") == a and v2.calls == 0             # replayed, not recomputed (D129)


def test_content_hash_ignores_writer_and_seconds(tmp_path):
    p1, p2 = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    for p, w in ((p1, "x"), (p2, "y")):
        b = CachedBackend(Echo(), UnitCache(p, {}, w))
        for t in "ab":
            b.generate(M(t), "action", None, 10)
    assert content_hash_of(p1) == content_hash_of(p2) and content_hash_of(p1)[0] == 2


def test_archive_never_deletes_and_indexes(tmp_path):
    p = tmp_path / "cache" / "u1.jsonl"
    b = CachedBackend(Echo(), UnitCache(p, {}, "w1"))
    b.generate(M("a"), "action", None, 10)
    dest = archive(p, "divergence", {"condition": "C2"})
    assert not p.exists() and dest.exists()
    idx = [json.loads(x) for x in (p.parent / "archive" / "index.jsonl").read_text().splitlines()]
    assert idx[0]["unit"] == "u1" and idx[0]["reason"] == "divergence" and idx[0]["records"] == 1
    assert idx[0]["condition"] == "C2"


def test_register_tasks_forwarded_only_when_present(tmp_path):
    from eem_acc.stage3.agent import ScriptedBackend, StubTokenizer
    c = UnitCache(tmp_path / "u.jsonl", {}, "w")
    assert hasattr(CachedBackend(ScriptedBackend(StubTokenizer(), "docs"), c), "register_tasks")
    assert not hasattr(CachedBackend(Echo(), c), "register_tasks")
    assert CachedBackend(Echo(), c).id == "cached(echo)"


def test_readonly_never_writes(tmp_path):
    p = tmp_path / "u.jsonl"
    b = CachedBackend(Echo(), UnitCache(p, {}, "w1"))
    b.generate(M("a"), "action", None, 10)
    before = p.read_text()
    c = UnitCache(p, {}, "v", readonly=True)
    cb = CachedBackend(Echo(), c)
    cb.generate(M("a"), "action", None, 10)
    with pytest.raises(ReplayDivergence):
        cb.generate(M("b"), "action", None, 10)
    assert p.read_text() == before
