"""llama.cpp backend on the T4: rendered GGUF chat template -> tokens -> create_completion, fresh KV cache per call,
grammar cache, guards; the Qwen tokenizer; the static-prefix snapshot lever (S3.3a spec sections 2, 3, 9).

`llm` is a `llama_cpp.Llama` (llama-cpp-python 0.3.36). Nothing here imports llama_cpp at module import time except
inside functions, so CPU code that only needs the cache or runner can import this module without the wheel."""
from __future__ import annotations

import hashlib
import json
import re
import threading
from collections import OrderedDict
from typing import Any

from ..agent.backend import CTX_MARGIN, Gen
from ..feasibility import ENVELOPE_SCHEMA
from .fsio import canon

LLAMA_CPP_VERSION = "0.3.36"
SAMPLING = {"temperature": 0.0, "top_k": 1, "top_p": 0.95, "min_p": 0.05, "repeat_penalty": 1.0, "seed": 0}
SCHEMAS = {
    "envelope_cap300": ENVELOPE_SCHEMA,
    "envelope": {**ENVELOPE_SCHEMA, "properties": {**ENVELOPE_SCHEMA["properties"], "thought": {"type": "string"}}},
}
SNAPSHOT_PROCEDURE = "snap-v1"
# special-token strings of Qwen2.5 that the template itself emits or that would be parsed as control tokens
SPECIAL_TEXT_RE = re.compile(r"<\|[a-z_]+\|>|</?tool_call>|</?tool_response>|</?tools>")


class BackendBusy(RuntimeError):
    """A generation is already running (e.g. left in a daemon thread by an interrupted Colab cell)."""


def make_formatter(llm: Any):
    """The formatter exactly as `Llama.__init__` builds it from the GGUF metadata (llama-cpp-python 0.3.36)."""
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter
    template = llm.metadata["tokenizer.chat_template"]
    eos_id, bos_id = llm.token_eos(), llm.token_bos()
    eos = llm._model.token_get_text(eos_id) if eos_id != -1 else ""
    bos = llm._model.token_get_text(bos_id) if bos_id != -1 else ""
    return Jinja2ChatFormatter(template=template, eos_token=eos, bos_token=bos, stop_token_ids=[eos_id]), template


class LlamaCppBackend:
    def __init__(self, llm: Any, model_digest: str, wheel_sha256: str, load_params: dict,
                 snapshot: bool = False, snapshot_lru: int = 3):
        self.llm = llm
        self.formatter, template = make_formatter(llm)
        self.template_sha256 = hashlib.sha256(template.encode()).hexdigest()
        self.model_digest, self.wheel_sha256 = model_digest, wheel_sha256
        self.load_params = dict(load_params)
        self.n_ctx = llm.n_ctx()
        self.snapshot = snapshot
        self.snapshot_lru = snapshot_lru
        self._states: OrderedDict[str, tuple[list[int], Any]] = OrderedDict()
        self._grammars: dict[str, Any] = {}
        self._lock = threading.Lock()
        self.id = f"llama-{model_digest[:12]}-{'snap' if snapshot else 'fresh'}"

    # ---------------------------------------------------------------------------------------------- identity
    def stamp(self) -> dict:
        """Output-affecting identity of this backend (cache header and result stamps)."""
        return {"model_digest": self.model_digest, "wheel_sha256": self.wheel_sha256,
                "llama_cpp_version": LLAMA_CPP_VERSION, "template_sha256": self.template_sha256,
                "load_params": self.load_params, "sampling": SAMPLING,
                "decoding_path": "snapshot" if self.snapshot else "fresh",
                "snapshot_procedure": SNAPSHOT_PROCEDURE if self.snapshot else None}

    @property
    def busy(self) -> bool:
        return self._lock.locked()

    # ----------------------------------------------------------------------------------------------- render
    def render(self, messages: list[dict]):
        """The chat handler's steps: (prompt text, tokens, stop, stopping criteria)."""
        r = self.formatter(messages=messages)
        tokens = self.llm.tokenize(r.prompt.encode("utf-8"), add_bos=not r.added_special, special=True)
        stop = [] if r.stop is None else (r.stop if isinstance(r.stop, list) else [r.stop])
        return r.prompt, tokens, stop, r.stopping_criteria

    def grammar(self, gid: str | None):
        if gid is None:
            return None, None
        if gid not in self._grammars:
            from llama_cpp.llama_grammar import LlamaGrammar
            g = LlamaGrammar.from_json_schema(json.dumps(SCHEMAS[gid]), verbose=False)   # raises on failure
            self._grammars[gid] = (g, hashlib.sha256(g._grammar.encode()).hexdigest())
        return self._grammars[gid]

    def cache_key(self, messages: list[dict], kind: str, grammar: str | None, max_tokens: int):
        prompt, _, stop, _ = self.render(messages)
        _, gsha = self.grammar(grammar)
        snap = self._snapshot_id(messages) if self.snapshot else None
        parts = {**self.stamp(), "kind": kind, "grammar": grammar, "grammar_gbnf_sha256": gsha,
                 "max_tokens": max_tokens, "stop": stop, "snapshot_id": snap, "prompt": prompt}
        return parts, prompt

    def check_control_tokens(self, messages: list[dict], prompt: str) -> None:
        """Content must not add control tokens: the rendered prompt has exactly the special strings the template adds."""
        bare = self.formatter(messages=[{"role": m["role"], "content": ""} for m in messages]).prompt
        n, n_bare = len(SPECIAL_TEXT_RE.findall(prompt)), len(SPECIAL_TEXT_RE.findall(bare))
        if n != n_bare:
            raise ValueError(f"message content adds {n - n_bare} special-token strings")

    # ------------------------------------------------------------------------------------------- snapshot lever
    def _prefix(self, messages: list[dict], tokens: list[int] | None = None) -> list[int] | None:
        """Tokens up to and including the end token of the system turn (None: no own system turn -> fresh path)."""
        if not messages or messages[0]["role"] != "system":
            return None
        if tokens is None:
            tokens = self.render(messages)[1]
        eos = self.llm.token_eos()
        return tokens[:tokens.index(eos) + 1] if eos in tokens else None

    def _snapshot_id(self, messages: list[dict]) -> str | None:
        p = self._prefix(messages)
        return None if p is None else hashlib.sha256(canon([SNAPSHOT_PROCEDURE, p]).encode()).hexdigest()

    def _load_prefix_state(self, prefix: list[int]) -> None:
        sid = hashlib.sha256(canon([SNAPSHOT_PROCEDURE, prefix]).encode()).hexdigest()
        if sid not in self._states:
            self.llm.reset()
            self.llm.eval(prefix)
            self._states[sid] = (list(prefix), self.llm.save_state())
            while len(self._states) > self.snapshot_lru:
                self._states.popitem(last=False)
        self._states.move_to_end(sid)
        self.llm.load_state(self._states[sid][1])          # before EVERY call (otherwise prefix reuse of the last call)

    # ------------------------------------------------------------------------------------------------ generate
    def generate(self, messages: list[dict], kind: str, grammar: str | None, max_tokens: int) -> Gen:
        if not self._lock.acquire(blocking=False):
            raise BackendBusy("a generation is already running")
        try:
            prompt, tokens, stop, criteria = self.render(messages)
            self.check_control_tokens(messages, prompt)
            if len(tokens) + max_tokens + CTX_MARGIN > self.n_ctx:
                raise ValueError(f"context overflow: {len(tokens)} + {max_tokens} + {CTX_MARGIN} > {self.n_ctx}")
            g, _ = self.grammar(grammar)
            prefix = self._prefix(messages, tokens) if self.snapshot else None
            if prefix is not None:
                assert tokens[:len(prefix)] == prefix
                self._load_prefix_state(prefix)
            else:
                self.llm.reset()                            # D47: fresh KV cache, no prefix reuse
            out = self.llm.create_completion(prompt=tokens, max_tokens=max_tokens, stop=stop,
                                             stopping_criteria=criteria, grammar=g, **SAMPLING)
            usage, choice = out["usage"], out["choices"][0]
            if usage["prompt_tokens"] != len(tokens):
                raise AssertionError(f"n_in {len(tokens)} != usage.prompt_tokens {usage['prompt_tokens']}")
            text = choice["text"]
            n_out = int(usage["completion_tokens"])
            diff = n_out - len(self.llm.tokenize(text.encode("utf-8"), add_bos=False, special=False))
            flags = (f"retok_diff:{diff}",) if diff else ()
            return Gen(text, len(tokens), n_out, choice.get("finish_reason") or "stop", grammar, flags)
        finally:
            self._lock.release()


class LlamaTokenizer:
    """Counts with the model's own tokenizer; `count_messages` is the rendered prompt's length (S3.3a spec section 3)."""

    def __init__(self, backend: LlamaCppBackend):
        self.backend, self.llm = backend, backend.llm
        self.id = f"qwen-gguf-{backend.model_digest[:12]}-llcpp{LLAMA_CPP_VERSION}"

    def count(self, text: str) -> int:
        return len(self.llm.tokenize(text.encode("utf-8"), add_bos=False, special=False)) if text else 0

    def count_messages(self, messages: list[dict]) -> int:
        return len(self.backend.render(messages)[1])

    def truncate(self, text: str, n: int) -> str:
        """The longest character prefix with count <= n (not detokenize: 0.3.36 drops tokens over 32 bytes)."""
        if n <= 0:
            return ""
        if self.count(text) <= n:
            return text
        lo, hi = 0, len(text)                              # count(text[:lo]) <= n < count(text[:hi])
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.count(text[:mid]) <= n:
                lo = mid
            else:
                hi = mid
        return text[:lo].rstrip()
