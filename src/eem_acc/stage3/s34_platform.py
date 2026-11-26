"""S3.4: the chain's platform on a real unit runner (notebook 05's cell 7, kept here so tests run the same code).

`Platform(root, runner, commit, replay_tools)` gives the Chain its callbacks: `run_units` (write a phase's plan once,
run or load it), `metrics`, `checks` (verify_replay + scrub audit), `adopt` (D166: replay a unit made at an earlier
commit with the current code) and `freeze_extra`. `replay_tools(unit) -> (key_source, tokenizer)` builds what
verify_replay needs (on the T4: a vocab-only llama.cpp key source from the unit's cache header).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from . import s34 as S34
from .llm import runner as R


class Platform:
    def __init__(self, root: Path, run: R.Runner, commit: str, replay_tools: Callable[[R.Unit], tuple[Any, Any]],
                 version: int = 1, log: Callable = print, verified: dict | None = None):
        """`verified`: outcomes kept across sessions (chain.json `verified`): a unit whose payload passed
        verify_replay at this commit is not replayed again (second opinion #2, Low 5)."""
        self.root, self.run, self.commit, self.replay_tools = Path(root), run, commit, replay_tools
        self.version, self.log = version, log
        self.verified = dict(verified or {})
        self.prefix = "s34-" if version == 1 else f"s34v{version}-"

    # ------------------------------------------------------------------------------------------------ results
    def load(self, u: R.Unit, plan: dict | None = None) -> dict | None:
        """A finished result; with the plan, only one made (or adopted, D166) at its frozen commit (review M2)."""
        p = self.run.result_path(u)
        if not self.run.is_done(u, plan=plan) or not p.exists():
            return None
        payload, run = R.split_result(json.loads(p.read_text()))
        return {**payload, "run": run}

    def run_units(self, phase: str, units: list[R.Unit], meta: dict | None, retry: bool = False,
                  load_only: bool = False) -> list[dict]:
        """A plan is written once; a reload must name the same units (and meta). `retry`: failed units run once more
        (second opinion M3; halted units are re-enabled by the runner itself). `load_only`: nothing runs (the chain is
        paused; 2nd opinion #3, Low 1)."""
        plan_file = self.root / "plans" / f"{phase}.json"
        if plan_file.exists():
            plan = R.read_plan(self.root, phase)
            if [R.Unit.from_json(u).unit_id for u in plan["units"]] != [u.unit_id for u in units]:
                raise S34.ChainHalt(f"{phase}: the stored plan names other units")
            if meta is not None and plan.get("meta") != meta:                         # review L8
                raise S34.ChainHalt(f"{phase}: the plan's meta differs")
        else:
            R.write_plan(self.root, phase, units, 1, frozen_commit=self.commit, meta=meta)
        try:
            if not load_only:
                self.run.run_phase(phase, retry_failed=retry)
        except R.RunHalted as e:                                   # the chain decides (retry, then a pause)
            self.log(f">>> {type(e).__name__}: {e} <<<")
        except R.TestPhaseCommitMismatch as e:                     # review M2: never "not done", never retried
            raise S34.ChainHalt(f"{phase}: the plan's commit is not the code's ({e}): ask in the chat") from e
        failed = self.run.failed_units(phase) | self.run.halted_marked(phase)   # fix review L5: halted = failed
        plan = R.read_plan(self.root, phase)
        out = []
        for u in units:
            r = self.load(u, plan)
            out.append(r if r is not None else {"unit": u.to_json(),
                                                "status": "failed" if u.unit_id in failed else "not_done"})
        return out

    def metrics(self, r: dict) -> dict:
        from .llm.cache import UnitCache
        if "run" not in r:
            return S34.unit_metrics(r, [])
        u = R.Unit.from_json(r["unit"])
        return S34.unit_metrics(r, UnitCache(self.run.cache_path(u), {}, "s", readonly=True).records)

    def _payload_sha(self, u: R.Unit) -> str | None:
        """The memo key: payload sha256 and the cache's content hash (2nd opinion #3, Low 2: a cache changed later
        without the result changing is verified again)."""
        from .llm.cache import content_hash_of
        try:
            run = R.split_result(json.loads(self.run.result_path(u).read_text()))[1]
            n, h = content_hash_of(self.run.cache_path(u))     # the cache itself (fix review L-b), not the record
        except (OSError, ValueError):
            return None
        return f"{run.get('payload_sha256')}:{h}:{n}"

    def _memo(self, u: R.Unit, sha: str | None) -> dict:
        m = self.verified.get(u.unit_id, {})
        return m if sha and m.get("key") == sha and m.get("commit") == self.commit else {}

    def verify(self, u: R.Unit) -> Any:
        """True, False, or the error text (never raises). Only a pass is remembered (a failure may be transient)."""
        sha = self._payload_sha(u)
        if self._memo(u, sha).get("verify_replay") is True:
            return True
        try:
            ks, tok = self.replay_tools(u)
            v = R.verify_replay(self.root, u.phase, u.unit_id, ks, tok)
        except Exception as e:                                     # noqa: BLE001 - reported, the chain decides
            v = f"{type(e).__name__}: {e}"
        if sha:
            self.verified[u.unit_id] = {**self._memo(u, sha), "key": sha, "payload_sha256": sha.split(":")[0],
                                        "commit": self.commit, "verify_replay": v}
        return v

    def checks(self, results: list[dict]) -> tuple[bool, int]:
        from .llm.audit import audit_result
        ok, hits = True, 0
        for r in results:
            if "run" not in r:
                ok = False
                continue
            v = self.verify(R.Unit.from_json(r["unit"]))
            if v is not True:
                self.log(f"verify_replay {r['unit']}: {v}")
                ok = False
            hits += len(audit_result(r))
        return ok, hits

    # header fields that need not match: the code commit, and what replay re-executes or what changes without a fix
    # (D173, 2nd opinion #3 M2); still required: the backend stamp, the embedder, torch, sentence-transformers, GPU.
    # The prompt, parser, suite, RQ3 and simulator versions are still enforced through the payload stamp that
    # verify_replay recomputes (a bump shows as the reason False, not "header differs")
    ADOPT_FREE = ("code_commit", "code_dirty", "unit", "unit_id", "prompt_version", "prompt_sha256", "parser_version",
                  "suite_version", "rq3_version", "sim_version", "python", "numpy", "platform_epoch",
                  "platform_fingerprint")

    def header_diff(self, u: R.Unit) -> list[str]:
        """Output-affecting header fields of the unit's cache that differ from this session's (second opinion #2,
        M1): a fix to the load cell, the embedder or the platform is never adopted by replay."""
        from .llm.cache import read_header
        h = read_header(self.run.cache_path(u))
        if h is None:
            return ["no cache header"]
        cur = self.run.ctx.header
        if cur.get("code_dirty", True):
            return ["code_dirty"]
        return [k for k in R.Runner.compat(h, cur) if k not in self.ADOPT_FREE]

    def adopt(self, phase: str, u: R.Unit) -> tuple[str | None, str | None, Any]:
        """D166: (the commit the stored result was made at, its payload sha256, verify_replay with the current code);
        (None, None, None) when there is no finished result. The unit's cache header must match this session's on
        every output-affecting field except the code commit."""
        if not self.run.is_done(u):
            return None, None, None
        payload, run = R.split_result(json.loads(self.run.result_path(u).read_text()))
        made = payload.get("stamp", {}).get("llm", {}).get("code_commit")
        diff = self.header_diff(u)
        if diff:
            return made, run["payload_sha256"], f"header differs: {diff}"
        return made, run["payload_sha256"], self.verify(u)

    # ------------------------------------------------------------------------------------------------- freeze
    def freeze_extra(self, s: S34.Settings) -> dict:
        from .agent import prompts as P
        from .apisuite.docs import documented_all
        from .apisuite.streams import SUITE_VERSION
        plans, payloads = {}, {}
        for pf in sorted((self.root / "plans").glob(f"{self.prefix}[ABC]-r*.json")):
            plan = json.loads(pf.read_text())
            plans[pf.stem] = plan["sha256"]
            for uj in plan["units"]:                               # every step A-C result's payload sha256 (review M2)
                u = R.Unit.from_json(uj)
                if self.run.result_path(u).exists():
                    payloads[u.unit_id] = R.split_result(json.loads(self.run.result_path(u).read_text()))[1][
                        "payload_sha256"]
        from ..sim import SIM_VERSION
        from .agent.parser import PARSER_VERSION
        from .rq3 import RQ3_VERSION
        return {"suite_version": SUITE_VERSION, "template_set_sha256": S34.template_set_sha(),
                "prompt_sha256": P.prompt_hash(), "parser_version": PARSER_VERSION, "rq3_version": RQ3_VERSION,
                "sim_version": SIM_VERSION,
                "system_prompt_sha256": hashlib.sha256(P.system_prompt(documented_all(), s.wording).encode())
                .hexdigest(), "plans_sha256": plans, "payload_sha256": payloads}

    # ---------------------------------------------------------------------------------------------- end of session
    def verify_finished(self, verified: dict | None = None) -> dict:
        """Every finished unit of this chain, once per payload and commit (kept across sessions): verify_replay and
        the scrub audit (review M3). Outcomes from this session's checks are reused (second opinion #2, Low 5)."""
        from .llm.audit import audit_result
        for k, v in (verified or {}).items():
            self.verified.setdefault(k, v)
        for pf in sorted((self.root / "plans").glob(f"{self.prefix}*.json")):
            for uj in json.loads(pf.read_text())["units"]:
                u = R.Unit.from_json(uj)
                if not self.run.is_done(u):
                    continue
                payload, run = R.split_result(json.loads(self.run.result_path(u).read_text()))
                m = self._memo(u, self._payload_sha(u))
                if "audit_hits" in m and m.get("verify_replay") is True:
                    continue
                v = self.verify(u)
                self.verified[u.unit_id] = {**self.verified.get(u.unit_id, {}), "verify_replay": v,   # Low 3
                                            "audit_hits": audit_result({**payload, "run": run})[:5]}
        # only complete outcomes are kept (fix review H1: a unit replayed for adoption but not finished has none)
        return {k: v for k, v in self.verified.items() if "audit_hits" in v and "verify_replay" in v}
