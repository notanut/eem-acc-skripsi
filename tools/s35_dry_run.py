"""S3.5 spec section 9.3: the dry run of the frozen analysis on a synthetic campaign (CPU, scripted backend).

The units have the real test-phase structure (blocks, arms, tags, params; plan phases s35-T and s35-S) but run on
seeds outside the pre-registered test seeds, so no held-out stream is touched by any agent: RQ1 dev seeds
1900-1913, RQ3 dev seeds 5090-5099, and SP seeds 3050-3054 (SP has only a test range; 3000-3004 are the registered
ones). The output is evidence that every analysis runs end to end on real payloads; its numbers describe the scripted
agent, not the LLM.

    python tools/s35_dry_run.py [out_dir]
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from eem_acc.stage3.agent import ScriptedBackend                      # noqa: E402
from eem_acc.stage3.analysis import data as D, report                # noqa: E402
from eem_acc.stage3.llm import runner as R                            # noqa: E402
from test_s33_runner import TOK, runner                               # noqa: E402

CFG5 = (("k", 5), ("k_s", 3), ("theta_link", 0.65))
RQ1 = list(range(1900, 1914))
RQ3 = list(range(5090, 5100))
SP = list(range(3050, 3055))


def units() -> dict[str, list[R.Unit]]:
    T, S = [], []
    for i, s in enumerate(RQ1):
        for c in ("C0", "C1", "C2", "C3") + (("C2-OU",) if i < 5 else ()):
            T.append(R.Unit("s35-T", "rq1", "dev", "rq1", s, c, params=CFG5))
    for s in RQ3:
        for c, w in (("C0", "acc"), ("C2", "acc"), ("C0", "none"), ("C2", "none")):
            T.append(R.Unit("s35-T", "rq3", "dev", "rq3", s, c, w, params=CFG5))
    for s in RQ1[:5]:
        for c in ("F", "AN", "R", "A1", "A2"):
            S.append(R.Unit("s35-S", "rq1", "dev", "rq1", s, c, params=CFG5))
    for s in SP:
        for c in ("C0", "C2"):
            S.append(R.Unit("s35-S", "rq1", "test", "sp", s, c, params=CFG5))
    for s in RQ3:
        S.append(R.Unit("s35-S", "rq3", "dev", "rq3", s, "C2", "fixed4", params=CFG5))
    for s in RQ3[:5]:
        S.append(R.Unit("s35-S", "rq3", "dev", "rq3", s, "C2-private", "acc", params=CFG5))
        for c, w in (("C0", "acc"), ("C2", "acc"), ("C0", "none"), ("C2", "none")):
            S.append(R.Unit("s35-S", "rq3", "dev", "rq3", s, c, w, params=CFG5 + (("llm_time_scale", 1 / 3),),
                            tag="sens"))
    return {"s35-T": T, "s35-S": S}


class PerUnitScripted(ScriptedBackend):
    """The scripted backend asserts that task instructions are unique across everything it has seen; different
    streams reuse instructions. Units run one at a time, so each unit's tasks replace the registry."""

    def register_tasks(self, tasks):
        self.by_instruction = {t["instruction"]: t for t in tasks}


def run(root: Path) -> list[dict]:
    results = []
    for phase, us in units().items():
        run_ = runner(root, PerUnitScripted(TOK, "memory"))
        R.write_plan(root, phase, us, 1)
        run_.run_phase(phase)
        for u in us:
            payload, meta = R.split_result(json.loads(run_.result_path(u).read_text()))
            results.append({**payload, "unit": u.to_json(), "run": meta})
    return results


def main(out_dir: Path, strict: bool = True) -> dict:
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        results = run(Path(tmp))
    units_ = [D.from_result(r) for r in results]
    prim = report.primary(units_)
    again = report.primary([D.from_result(r) for r in results])
    full_t = report.full_T(units_, strict_pins=strict)
    full_s = report.full_S(units_, strict_pins=strict)
    out = {"what": "S3.5 dry run on a synthetic campaign (scripted agent, non-test seeds); not results",
           "units": len(results), "seconds": round(time.time() - t0, 1),
           "primary_hash": report.canonical_hash(prim), "primary_hash_recomputed": report.canonical_hash(again),
           "versions": report.versions()}
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, obj in (("primary", prim), ("full_T", full_t), ("full_S", full_s)):
        out[f"{name}_sha256"] = report.write(obj, out_dir / f"s35_dry_run_{name}.json")
    (out_dir / "s35_dry_run_summary.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs" / "evidence"), indent=1))
