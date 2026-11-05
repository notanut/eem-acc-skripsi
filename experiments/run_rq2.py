"""RQ2 runner (scripted agents, simulated server). Resumable: re-running continues.

Phases (approved design, docs/stage2_design_proposal.md):
  dev     5,520 runs: 81 ACC configs + 7 fixed windows + 4 token-bucket rates, dev seeds 0-4
  select  choose the frozen configuration from dev results -> configs/rq2_frozen.json
  test    frozen ACC + 3 ablations + fixed {4,16,64,best} + tuned token bucket, test seeds 100-109

Usage (repo root):
  PYTHONPATH=src python experiments/run_rq2.py --phase dev    --root <ROOT>
  PYTHONPATH=src python experiments/run_rq2.py --phase select --root <ROOT>
  PYTHONPATH=src python experiments/run_rq2.py --phase test   --root <ROOT>
<ROOT> is the Drive folder in Colab (e.g. /content/drive/MyDrive/Skripsi) or any local folder.
Outputs: <ROOT>/results/rq2/<phase>/{units,checkpoints}/, <ROOT>/logs/rq2_<phase>.log
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eem_acc import scenarios as S
from eem_acc.analysis import load_results, score_methods, select_frozen
from eem_acc.sim import run_scenario
from eem_acc.utils import atomic_write_json, run_resumable

REPO = Path(__file__).resolve().parents[1]
FROZEN_PATH = REPO / "configs" / "rq2_frozen.json"


def phase_units(phase: str, frozen: dict | None):
    if phase == "dev":
        return S.units(S.dev_methods(), S.DEV_SEEDS), False
    if phase == "test":
        assert frozen is not None, "test phase needs configs/rq2_frozen.json (run --phase select first)"
        return S.units(S.test_methods(frozen), S.TEST_SEEDS), True
    raise ValueError(phase)


def run_phase(phase: str, root: Path, limit: int | None = None, frozen: dict | None = None):
    units, store_trace = phase_units(phase, frozen)
    if limit:
        units = units[:limit]
    index = {u[0]: u for u in units}

    def run_unit(uid: str) -> dict:
        _, scn, m, meth, seed = index[uid]
        r = run_scenario(S.build(scn, m, meth, seed))
        out = {"meta": {"scenario": scn, "M": m, "method": meth, "seed": seed}, "metrics": r["metrics"]}
        if store_trace:
            out["window_trace"] = r["window_trace"]
        return out

    config = {"phase": phase, "design": "docs/stage2_design_proposal.md@approved-2026-11-05", "frozen": frozen}
    return run_resumable([u[0] for u in units], run_unit, root / "results" / "rq2" / phase,
                         config=config, stateful=False, checkpoint_every=50,
                         log_path=root / "logs" / f"rq2_{phase}.log", progress_interval_s=15.0)


def do_select(root: Path) -> int:
    df = load_results(root / "results" / "rq2" / "dev" / "units")
    expected = len(S.units(S.dev_methods(), S.DEV_SEEDS))
    if len(df) != expected:
        print(f"[select] refusing: {len(df)} of {expected} dev runs present. Finish --phase dev first.")
        return 2
    res = select_frozen(df)
    print(json.dumps(res, indent=2, default=str))
    if not res["ok"]:
        print("[select] NO feasible candidate in at least one family; nothing written. Ask the user how to proceed.")
        return 3
    FROZEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(FROZEN_PATH, res)
    atomic_write_json(root / "results" / "rq2" / "dev_scores.json", score_methods(df).to_dict("records"))
    atomic_write_json(root / "results" / "rq2" / "rq2_frozen.json", res)   # Drive copy (Colab clones are ephemeral)
    print(f"[select] wrote {FROZEN_PATH} (commit it to the repo BEFORE running the test phase)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["dev", "select", "test"], required=True)
    ap.add_argument("--root", default="outputs", help="Drive folder (Colab) or local output folder")
    ap.add_argument("--limit", type=int, default=None, help="only the first N units (smoke tests)")
    args = ap.parse_args()
    root = Path(args.root)
    if args.phase == "select":
        return do_select(root)
    frozen = None
    if args.phase == "test":
        src = FROZEN_PATH if FROZEN_PATH.exists() else root / "results" / "rq2" / "rq2_frozen.json"
        print(f"[test] frozen configuration from {src}")
        frozen = json.loads(src.read_text())
    run_phase(args.phase, root, args.limit, frozen)
    return 0


if __name__ == "__main__":
    sys.exit(main())
