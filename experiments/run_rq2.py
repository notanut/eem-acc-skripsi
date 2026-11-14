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
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

from eem_acc import scenarios as S
from eem_acc.analysis import load_results, score_methods, select_frozen
from eem_acc.sim import SIM_VERSION, run_scenario
from eem_acc.utils import ResultStore, atomic_write_json, run_resumable

import numpy as np

REPO = Path(__file__).resolve().parents[1]
FROZEN_PATH = REPO / "configs" / "rq2_frozen.json"


def phase_units(phase: str, frozen: dict | None):
    if phase == "dev":
        return S.units(S.dev_methods(), S.DEV_SEEDS), False
    if phase == "test":
        assert frozen is not None, "test phase needs configs/rq2_frozen.json (run --phase select first)"
        return S.units(S.test_methods(frozen), S.TEST_SEEDS), True
    raise ValueError(phase)


def spec_hash(scn: str, m: int, meth: dict, seed: int) -> str:
    """Hash of the FULL resolved configuration of one run + simulator version. Stored in every result;
    a resume that would mix results from a different configuration is refused."""
    blob = json.dumps({"sim": SIM_VERSION, "scenario": dataclasses.asdict(S.build(scn, m, meth, seed))},
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def frozen_hash(frozen: dict | None) -> str | None:
    return None if frozen is None else hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()[:16]


_SRC_DIRS = ("sim.py", "metrics.py", "keyed.py", "scenarios.py", "env", "agent", "acc")


def source_hash() -> str:
    """Hash of the simulator source (informational stamp; SIM_VERSION is the enforced one)."""
    h = hashlib.sha256()
    pkg = REPO / "src" / "eem_acc"
    for name in _SRC_DIRS:
        p = pkg / name
        for f in sorted([p] if p.is_file() else p.rglob("*.py")):
            h.update(f.relative_to(pkg).as_posix().encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


def check_results(phase: str, root: Path, frozen: dict | None) -> dict[str, str]:
    """ONE pass over every result file of the phase. Every file must belong to the phase's FULL unit list and
    carry that unit's current spec_hash; otherwise ValueError (mixed code/config versions, or leftovers of an
    older grid that load_results would silently read). Returns {unit_id: spec_hash} of the valid results."""
    units, _ = phase_units(phase, frozen)
    expected = {u[0]: spec_hash(*u[1:]) for u in units}
    store = ResultStore(root / "results" / "rq2" / phase / "units")
    ok, bad = {}, []
    for uid, res in store.iter_results():
        h = res.get("meta", {}).get("spec_hash") if isinstance(res, dict) else None
        if uid in expected and h == expected[uid]:
            ok[uid] = h
        else:
            bad.append(uid)
    if bad:
        raise ValueError(
            f"{len(bad)} result(s) in {store.root} do not match the current configuration/simulator version "
            f"(SIM_VERSION={SIM_VERSION}) or are not part of this phase's grid (e.g. {bad[0]}). "
            f"Use a new --root (recommended) or delete that folder.")
    return ok


def digest(hashes: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(sorted(hashes.items())).encode()).hexdigest()[:16]


def run_phase(phase: str, root: Path, limit: int | None = None, frozen: dict | None = None):
    units, store_trace = phase_units(phase, frozen)
    if limit:
        units = units[:limit]
    index = {u[0]: u for u in units}
    fz = frozen_hash(frozen)

    check_results(phase, root, frozen)   # refuse to resume on top of results from other code/config
    stamp = {"sim_version": SIM_VERSION, "frozen_hash": fz, "numpy": np.__version__, "source_hash": source_hash()}

    def run_unit(uid: str) -> dict:
        _, scn, m, meth, seed = index[uid]
        r = run_scenario(S.build(scn, m, meth, seed))
        out = {"meta": {"scenario": scn, "M": m, "method": meth, "seed": seed,
                        "spec_hash": spec_hash(scn, m, meth, seed), **stamp},
               "metrics": r["metrics"]}
        if store_trace:
            out["window_trace"] = r["window_trace"]
        return out

    config = {"phase": phase, "design": "docs/stage2_design_proposal.md@approved-2026-11-05", "frozen": frozen}
    return run_resumable([u[0] for u in units], run_unit, root / "results" / "rq2" / phase,
                         config=config, stateful=False, checkpoint_every=50,
                         log_path=root / "logs" / f"rq2_{phase}.log", progress_interval_s=15.0)


def do_select(root: Path, force: bool = False) -> int:
    test_units = root / "results" / "rq2" / "test" / "units"
    if test_units.exists() and any(test_units.glob("*.json")):
        print("[select] refusing: test results already exist in this folder. Re-selecting after seeing test "
              "results would invalidate the dev/test split. Use a fresh --root for a new experiment.")
        return 4
    try:
        hashes = check_results("dev", root, None)
    except ValueError as e:
        print(f"[select] refusing: {e}")
        return 5
    expected = len(S.units(S.dev_methods(), S.DEV_SEEDS))
    if len(hashes) != expected:
        print(f"[select] refusing: {len(hashes)} of {expected} dev runs present. Finish --phase dev first.")
        return 2
    df = load_results(root / "results" / "rq2" / "dev" / "units")
    res = select_frozen(df)
    res = {**res, "sim_version": SIM_VERSION, "dev_digest": digest(hashes)}
    print(json.dumps(res, indent=2, default=str))
    if not res["ok"]:
        print("[select] NO feasible candidate in at least one family; nothing written. Ask the user how to proceed.")
        return 3
    if FROZEN_PATH.exists() and not force:
        old = json.loads(FROZEN_PATH.read_text())
        outdated = "sim_version" in old and old["sim_version"] != SIM_VERSION   # unusable anyway: test refuses it
        if outdated:
            print(f"[select] replacing {FROZEN_PATH}: it was selected with SIM_VERSION={old['sim_version']} (now {SIM_VERSION})")
        elif json.loads(json.dumps(res, default=str)) != old:
            print(f"[select] refusing: {FROZEN_PATH} already exists with DIFFERENT content (another root or "
                  f"an older run). Re-run with --force only if you really mean to replace it.")
            return 6
    FROZEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(FROZEN_PATH, res)
    atomic_write_json(root / "results" / "rq2" / "dev_scores.json", score_methods(df).to_dict("records"))
    atomic_write_json(root / "results" / "rq2" / "rq2_frozen.json", res)   # Drive copy (Colab clones are ephemeral)
    print(f"[select] wrote {FROZEN_PATH} and {root / 'results' / 'rq2' / 'rq2_frozen.json'}. Locally: commit the repo "
          f"copy BEFORE the test phase. In Colab the clone cannot push: the Drive copy is the record (download it "
          f"or send it to be committed).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["dev", "select", "test"], required=True)
    ap.add_argument("--root", default="outputs", help="Drive folder (Colab) or local output folder")
    ap.add_argument("--limit", type=int, default=None, help="only the first N units (smoke tests)")
    ap.add_argument("--force", action="store_true", help="select: replace a different configs/rq2_frozen.json")
    args = ap.parse_args()
    root = Path(args.root)
    if args.phase == "select":
        return do_select(root, args.force)
    frozen = None
    if args.phase == "test":
        frozen, msg = load_frozen_for_test(root)
        if frozen is None:
            print(f"[test] refusing: {msg}")
            return 7
        print(f"[test] {msg}")
    run_phase(args.phase, root, args.limit, frozen)
    return 0


def load_frozen_for_test(root: Path) -> tuple[dict | None, str]:
    """The test phase uses the root's frozen copy; the repo copy (if any) must be identical, its simulator
    version must be the current one, and it must have been selected from THIS root's dev results."""
    drive = root / "results" / "rq2" / "rq2_frozen.json"
    copies = [p for p in (drive, FROZEN_PATH) if p.exists()]
    if not copies:
        return None, "no frozen configuration found; run --phase select first"
    frozen = json.loads(copies[0].read_text())
    if len(copies) == 2 and json.loads(copies[1].read_text()) != frozen:
        return None, f"{drive} and {FROZEN_PATH} differ; they must be the same selection"
    if frozen.get("sim_version") != SIM_VERSION:
        return None, f"frozen configuration was selected with SIM_VERSION={frozen.get('sim_version')}, now {SIM_VERSION}"
    try:
        dev = check_results("dev", root, None)
    except ValueError as e:
        return None, f"dev results invalid: {e}"
    if frozen.get("dev_digest") != digest(dev):
        return None, "frozen configuration was not selected from this root's (complete, current) dev results"
    return frozen, f"frozen configuration from {copies[0]} (hash {frozen_hash(frozen)}, dev digest ok)"


if __name__ == "__main__":
    sys.exit(main())
