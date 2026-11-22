"""LLM backend protocol, modelled LLM time, and the deterministic CPU stand-ins for tokenizer and embedder
(S3.2a spec section 1). The real Qwen tokenizer, MiniLM embedder and llama.cpp backend come in S3.3."""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Protocol

# D49 (S3.0, T4): seconds per token; a = 0
T_LLM = {"free": {"a": 0.0, "b_in": 0.99e-3, "b_out": 44.6e-3},
         "grammar": {"a": 0.0, "b_in": 1.09e-3, "b_out": 70.5e-3}}
N_CTX = 8192


GRAMMARS = (None, "envelope_cap300", "envelope")   # free; fallback grammar (thought <= 300 chars); grammar-always (D65)


@dataclass(frozen=True)
class Gen:
    text: str
    n_in: int
    n_out: int
    finish_reason: str          # "stop" | "length"
    grammar: str | None = None  # one of GRAMMARS


def t_llm(g: Gen) -> float:
    c = T_LLM["grammar" if g.grammar else "free"]
    return c["a"] + c["b_in"] * g.n_in + c["b_out"] * g.n_out


class LLMBackend(Protocol):
    def generate(self, messages: list[dict], kind: str, grammar: str | None, max_tokens: int) -> Gen: ...


class Tokenizer(Protocol):
    id: str

    def count(self, text: str) -> int: ...

    def count_messages(self, messages: list[dict]) -> int: ...

    def truncate(self, text: str, n: int) -> str: ...


class Embedder(Protocol):
    id: str

    def embed(self, text: str) -> tuple: ...


_TOK = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class StubTokenizer:
    """Deterministic stand-in: words and punctuation marks count as tokens; a chat message adds 4 tokens (role and
    template markers). Only for CPU tests; S3.3 uses the Qwen tokenizer with the chat template."""

    id = "stub-regex-v1"

    def count(self, text: str) -> int:
        return len(_TOK.findall(text))

    def count_messages(self, messages: list[dict]) -> int:
        return sum(self.count(m["content"]) + 4 for m in messages) + 2

    def truncate(self, text: str, n: int) -> str:
        if n <= 0:
            return ""
        out, k = [], 0
        for m in re.finditer(r"\w+|[^\w\s]|\s+", text):
            if not m.group().isspace():
                if k == n:
                    break
                k += 1
            out.append(m.group())
        return "".join(out).rstrip()


class BagOfWordsEmbedder:
    """Deterministic hashed bag of words (lower-cased word tokens into `dim` buckets, L2-normalised), so similar texts
    have meaningful cosine similarity in CPU tests (S3.2a spec section 8). Cached by text."""

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.id = f"bow-sha1-{dim}"
        self.cache: dict[str, tuple] = {}

    def embed(self, text: str) -> tuple:
        if text in self.cache:
            return self.cache[text]
        v = [0.0] * self.dim
        for w in re.findall(r"[a-z0-9_]+", text.lower()):
            h = int(hashlib.sha1(w.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out = tuple(round(x / n, 8) for x in v)
        self.cache[text] = out
        return out


def cosine(a: tuple, b: tuple) -> float:
    """Cosine of two stored vectors, rounded to 1e-6 (design section 3: identical on replay)."""
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return round(sum(x * y for x, y in zip(a, b)) / (na * nb), 6)
