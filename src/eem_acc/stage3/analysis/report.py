"""S3.5 spec sections 9 and 16: primary.json (exact primary tests: numpy + scipy only), full.json (everything
else), and the canonical hash the notebook records and Claude must reproduce (D190)."""
from __future__ import annotations

import hashlib
import importlib
import json
import math
from pathlib import Path

from . import descriptive, endpoints, mechanism, secondary
from .data import PRIMARY_RQ1, UnitData

SIG_DIGITS = 10
PINS = Path(__file__).resolve().parents[4] / "configs" / "s35_pins.json"
T_PHASE, S_PHASE = "s35-T", "s35-S"
ALL_RQ1 = PRIMARY_RQ1 + descriptive.ARMS


def canonical(obj):
    """Floats rounded to 10 significant digits (nan/inf -> None), tuples -> lists, keys -> strings (sorted when
    dumped). The exact sign-flip p-values are dyadic and survive unchanged."""
    import numpy as np
    if isinstance(obj, np.generic):                 # every numpy scalar (bool, int8, uint64, float16, ...; review L1)
        obj = obj.item()
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(f"{obj:.{SIG_DIGITS}g}")
    if isinstance(obj, dict):
        return {str(k): canonical(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [canonical(v) for v in obj]
    raise TypeError(f"cannot canonicalise {type(obj).__name__}")


def dumps(obj) -> str:
    return json.dumps(canonical(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_hash(obj) -> str:
    return hashlib.sha256(dumps(obj).encode()).hexdigest()


FLOAT_KEYS = ("ci95", "mdd")                       # computed with scipy's t quantiles: compared at a tolerance
RTOL = 1e-8          # ~4 orders above the ~1e-12 scipy noise seen; above the 10-digit rounding of canonical()


def exact_part(obj):
    """D210: the part of primary.json that does not depend on scipy: exact values (p-values, counts, flags,
    rejections, levels, adjusted p) and deterministic floating-point values computed with numpy alone (the E2/E4
    differences and their means, the interaction means). Everything but the scipy-derived t intervals and MDDs. Its
    canonical hash is the notebook's record and must be reproduced on any scipy."""
    if isinstance(obj, dict):
        return {k: exact_part(v) for k, v in obj.items() if k not in FLOAT_KEYS}
    if isinstance(obj, (list, tuple)):
        return [exact_part(v) for v in obj]
    return obj


def primary_hash(prim: dict) -> str:
    return canonical_hash(exact_part(prim))


def float_mismatches(a, b, rtol: float = RTOL, path: str = "") -> list[str]:
    """D210: the scipy-derived floats (FLOAT_KEYS) of two primary.json versions agree to a relative RTOL (1e-8).
    Compare the recorded file with the UNROUNDED recomputation (report.primary output), not two rounded files: two
    rounded values can straddle a 10-digit rounding edge (review L2)."""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k in FLOAT_KEYS:
                out += _close(a.get(k), b.get(k), rtol, f"{path}/{k}")
            elif isinstance(a.get(k), (dict, list)):
                out += float_mismatches(a.get(k), b.get(k), rtol, f"{path}/{k}")
    elif (a is None) != (b is None) or type(a) is not type(b):
        out.append(path or "/")
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            out += float_mismatches(x, y, rtol, f"{path}[{i}]")
    return out


def _close(x, y, rtol, path):
    xs, ys = (x if isinstance(x, list) else [x]), (y if isinstance(y, list) else [y])
    if len(xs) != len(ys):
        return [path]
    for u, v in zip(xs, ys):
        if (u is None) != (v is None):
            return [path]
        if u is not None and not math.isclose(float(u), float(v), rel_tol=rtol, abs_tol=1e-15):
            return [path]
    return []


def versions(mods=("numpy", "scipy", "statsmodels", "patsy", "pandas")) -> dict:
    out = {}
    for m in mods:
        try:
            out[m] = importlib.import_module(m).__version__
        except Exception:                                   # noqa: BLE001 - a module absent on Colab is recorded
            out[m] = None
    return out


def check_pins(strict: bool = True, expect_scipy: str | None = None) -> dict:
    """The full analysis refuses other versions of the pinned packages (spec section 9.4) unless strict=False.
    `expect_scipy`: the scipy version the notebook recorded with primary.json (the reproduction must use it)."""
    pins = {k: v for k, v in json.loads(PINS.read_text()).items() if not k.startswith("_")}
    if expect_scipy:
        pins["scipy"] = expect_scipy
    have = versions(tuple(pins))
    bad = {k: (have[k], v) for k, v in pins.items() if have[k] != v}
    if bad and strict:
        raise RuntimeError(f"analysis environment differs from configs/s35_pins.json: {bad}")
    return {"pins": pins, "have": have, "mismatch": bad}


def primary(units: list[UnitData]) -> dict:
    """AP: the RQ1 and RQ3 hypotheses, both graphs, interaction and floor rule, completeness, and the D186
    sensitivity versions. numpy + scipy only; hashed canonically."""
    return {"rq1": endpoints.rq1_primary(units, T_PHASE),
            "rq1_failed_as_unsolved": endpoints.rq1_primary(units, T_PHASE, include_failed=True),
            "rq3": endpoints.rq3_primary(units, T_PHASE),
            "rq3_failed_as_unsolved": endpoints.rq3_primary(units, T_PHASE, include_failed=True),
            "epochs": epochs(units)}


def epochs(units: list[UnitData], phase: str = T_PHASE) -> dict:
    """Platform epochs by cell, over the T units only (so the hash never depends on what else is passed in, L4)."""
    out: dict = {}
    for u in units:
        if u.phase != phase:
            continue
        out.setdefault(u.cell, {}).setdefault(str(u.epoch), 0)
        out[u.cell][str(u.epoch)] += 1
    return out


def full_T(units: list[UnitData], strict_pins: bool = True, expect_scipy: str | None = None) -> dict:
    """Everything after AP that uses T units only (spec section 16)."""
    env = check_pins(strict_pins, expect_scipy)
    return {"environment": env,
            "c1_vs_c0": secondary.c1_vs_c0(units, T_PHASE),
            "c3_vs_c2_tost": secondary.c3_vs_c2_tost(units, T_PHASE),
            "c2ou_vs_c2": descriptive.arms_vs_c2(units, (T_PHASE,), arms=("C2-OU",)),
            "cost_per_solved": secondary.cost_per_solved(units, PRIMARY_RQ1 + ("C2-OU",), T_PHASE),
            "ratio_of_sums": secondary.ratio_of_sums(units, T_PHASE),
            "error_splits": secondary.error_splits(units, PRIMARY_RQ1 + ("C2-OU",), T_PHASE),
            "exposures": secondary.exposures(units, PRIMARY_RQ1 + ("C2-OU",), T_PHASE),
            "drifted_call_share": secondary.drifted_call_share(units, PRIMARY_RQ1, T_PHASE),
            "learning_curves": secondary.learning_curves(units, T_PHASE),
            "mixed_model": secondary.mixed_model(units, T_PHASE),
            "rq3_secondary": secondary.rq3_secondary(units, (T_PHASE,)),
            "mechanism": {"repeat_exposure": mechanism.repeat_exposure(units, PRIMARY_RQ1 + ("C2-OU",), (T_PHASE,)),
                          "note_correctness": mechanism.note_correctness(units, ("C1", "C2", "C3", "C2-OU"), (T_PHASE,)),
                          "note_precision": mechanism.note_precision(units, ("C2", "C3"), (T_PHASE,)),
                          "retrieval_recall": mechanism.retrieval_recall(units, ("C1", "C2", "C3", "C2-OU"),
                                                                         (T_PHASE,)),
                          "c1_duplicate_crowding": mechanism.c1_duplicate_crowding(units, (T_PHASE,))}}


def full_S(units: list[UnitData], strict_pins: bool = True, expect_scipy: str | None = None) -> dict:
    """The descriptive part (after S): arms, SP, RQ3 extras, and the mechanism checks over every arm."""
    env = check_pins(strict_pins, expect_scipy)
    both = (T_PHASE, S_PHASE)
    return {"environment": env,
            "arms_vs_c2": descriptive.arms_vs_c2(units, both),
            "sp": descriptive.sp(units, (S_PHASE,)),
            "rq3_extras": descriptive.rq3_extras(units, both),
            "rq3_secondary": secondary.rq3_secondary(units, both),
            "cost_per_solved": secondary.cost_per_solved(units, ALL_RQ1, both),
            "error_splits": secondary.error_splits(units, ALL_RQ1, both),
            "mechanism": {"repeat_exposure": mechanism.repeat_exposure(units, ALL_RQ1, both),
                          "note_correctness": mechanism.note_correctness(units, ("C1", "F") + mechanism.EEM_CONDITIONS,
                                                                         both),
                          "note_precision": mechanism.note_precision(units, ("C2", "C3", "AN", "A2"), both),
                          "retrieval_recall": mechanism.retrieval_recall(units, mechanism.RETRIEVING, both)}}


def write(obj, path: Path) -> str:
    """Write the canonical JSON and return the sha256 of the whole file. Not the D190/D210 record: the notebook
    records `primary_hash` (the exact part)."""
    text = dumps(obj)
    Path(path).write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()
