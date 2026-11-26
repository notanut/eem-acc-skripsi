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
                 version: int = 1, log: Callable = print):
        self.root, self.run, self.commit, self.replay_tools = Path(root), run, commit, replay_tools
        self.version, self.log = version, log
        self.prefix = "s34-" if version == 1 else f"s34v{version}-"

    # ------------------------------------------------------------------------------------------------ results
    def load(self, u: R.Unit, plan: dict | None = None) -> dict | None:
        """A finished result; with the plan, only one made (or adopted, D166) at its frozen commit (review M2)."""
        p = self.run.result_path(u)
        if not self.run.is_done(u, plan=plan) or not p.exists():
            return None
        payload, run = R.split_result(json.loads(p.read_text()))
        return {**payload, "run": run}

    def run_units(self, phase: str, units: list[R.Unit], meta: dict | None, retry: bool = False) -> list[dict]:
        """A plan is written once; a reload must name the same units (and meta). `retry`: failed units run once more
        (second opinion M3; halted units are re-enabled by the runner itself)."""
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
            self.run.run_phase(phase, retry_failed=retry)
        except R.RunHalted as e:                                   # the chain decides (retry, then a pause)
            self.log(f">>> {type(e).__name__}: {e} <<<")
        except R.TestPhaseCommitMismatch as e:                     # review M2: never "not done", never retried
            raise S34.ChainHalt(f"{phase}: the plan's commit is not the code's ({e}): ask in the chat") from e
        failed = self.run.failed_units(phase)
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

    def verify(self, u: R.Unit) -> Any:
        """True, False, or the error text (never raises)."""
        try:
            ks, tok = self.replay_tools(u)
            return R.verify_replay(self.root, u.phase, u.unit_id, ks, tok)
        except Exception as e:                                     # noqa: BLE001 - reported, the chain decides
            return f"{type(e).__name__}: {e}"

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

    def adopt(self, phase: str, u: R.Unit) -> tuple[str | None, str | None, Any]:
        """D166: (the commit the stored result was made at, its payload sha256, verify_replay with the current code);
        (None, None, None) when there is no finished result."""
        if not self.run.is_done(u):
            return None, None, None
        payload, run = R.split_result(json.loads(self.run.result_path(u).read_text()))
        return payload.get("stamp", {}).get("llm", {}).get("code_commit"), run["payload_sha256"], self.verify(u)

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
        return {"suite_version": SUITE_VERSION, "template_set_sha256": S34.template_set_sha(),
                "system_prompt_sha256": hashlib.sha256(P.system_prompt(documented_all(), s.wording).encode())
                .hexdigest(), "plans_sha256": plans, "payload_sha256": payloads}

    # ---------------------------------------------------------------------------------------------- end of session
    def verify_finished(self, verified: dict) -> dict:
        """Every finished unit of this chain, once (kept across sessions): verify_replay and the scrub audit (review
        M3). A unit re-run after a re-pin is checked again (its payload sha256 changed)."""
        from .llm.audit import audit_result
        out = dict(verified)
        for pf in sorted((self.root / "plans").glob(f"{self.prefix}*.json")):
            for uj in json.loads(pf.read_text())["units"]:
                u = R.Unit.from_json(uj)
                if not self.run.is_done(u):
                    continue
                payload, run = R.split_result(json.loads(self.run.result_path(u).read_text()))
                if out.get(u.unit_id, {}).get("payload_sha256") == run["payload_sha256"]:
                    continue
                out[u.unit_id] = {"verify_replay": self.verify(u), "payload_sha256": run["payload_sha256"],
                                  "audit_hits": audit_result({**payload, "run": run})[:5]}
        return out
