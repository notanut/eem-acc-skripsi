"""Load RQ2 results and select the frozen configuration on the DEV seeds (criterion D15-b).

Criterion (approved): maximise goodput subject to
    mean SLO-violation rate <= MAX_SLO_VIOLATION   and   mean call success >= MIN_CALL_SUCCESS,
means taken over all dev cells (scenario x M), after averaging over dev seeds.

Interpretation made here (flagged to the user): goodput differs by an order of magnitude
across cells (M=4 vs M=64), so it is normalised per cell by the best seed-averaged goodput
any dev method achieves in that cell; a method's score is the mean normalised goodput over
cells. A candidate must have a result in EVERY dev cell to be eligible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from . import scenarios as S
from .utils import ResultStore

METRIC_COLS = ["goodput", "slo_violation_rate", "call_success_rate", "throttle_rate",
               "mean_latency_excess", "makespan", "lat_p95", "n_429", "n_503"]


def load_results(units_dir: str | Path) -> pd.DataFrame:
    rows = []
    for uid, res in ResultStore(units_dir).iter_results():
        meta = res["meta"]
        row = {"unit_id": uid, "scenario": meta["scenario"], "M": meta["M"], "seed": meta["seed"],
               "tag": S.method_tag(meta["method"]), **{f"m_{k}": v for k, v in meta["method"].items()}}
        row.update(res["metrics"])
        rows.append(row)
    return pd.DataFrame(rows)


def cell_means(df: pd.DataFrame) -> pd.DataFrame:
    """Seed-averaged metrics per (scenario, M, method)."""
    d = df.copy()
    d["slo_violation_rate"] = d["slo_violation_rate"].fillna(1.0)   # no successful attempt = worst case
    d["call_success_rate"] = d["call_success_rate"].fillna(0.0)
    keys = ["scenario", "M", "tag"]
    extra = [c for c in d.columns if c.startswith("m_")]
    g = d.groupby(keys)
    out = g[["goodput", "slo_violation_rate", "call_success_rate", "throttle_rate"]].mean()
    out["n_seeds"] = g.size()
    for c in extra:
        out[c] = g[c].first()
    return out.reset_index()


def score_methods(df: pd.DataFrame) -> pd.DataFrame:
    """One row per method: mean normalised goodput, mean SLO violation, mean success, feasibility."""
    cells = cell_means(df)
    best = cells.groupby(["scenario", "M"])["goodput"].transform("max")
    cells["norm_goodput"] = cells["goodput"] / best
    n_cells = cells[["scenario", "M"]].drop_duplicates().shape[0]
    g = cells.groupby("tag")
    out = pd.DataFrame({
        "score": g["norm_goodput"].mean(),
        "mean_slo_violation": g["slo_violation_rate"].mean(),
        "mean_call_success": g["call_success_rate"].mean(),
        "n_cells": g.size(),
    })
    first = g[[c for c in cells.columns if c.startswith("m_")]].first()
    out = out.join(first).reset_index()
    out["complete"] = out["n_cells"] == n_cells
    out["feasible"] = (out["complete"] & (out["mean_slo_violation"] <= S.MAX_SLO_VIOLATION)
                       & (out["mean_call_success"] >= S.MIN_CALL_SUCCESS))
    return out.sort_values(["feasible", "score", "mean_slo_violation", "tag"],
                           ascending=[False, False, True, True]).reset_index(drop=True)


def select_frozen(df: pd.DataFrame) -> dict[str, Any]:
    """Pick the best feasible ACC config, fixed window and pacer rate from dev results.
    Returns {"ok": bool, ...}; if a family has no feasible candidate, ok=False and the
    best-infeasible rows are returned so the user can decide (nothing is silently relaxed)."""
    sc = score_methods(df)
    fams = {"acc": sc[sc["m_kind"] == "acc"], "fixed": sc[sc["m_kind"] == "fixed"],
            "tokenbucket": sc[sc["m_kind"] == "tokenbucket"]}
    result: dict[str, Any] = {"ok": True, "criterion": {
        "objective": "maximise mean per-cell-normalised goodput",
        "max_slo_violation": S.MAX_SLO_VIOLATION, "min_call_success": S.MIN_CALL_SUCCESS},
        "dev_seeds": sorted(int(x) for x in df["seed"].unique()), "n_dev_runs": int(len(df)), "top": {}}
    for name, t in fams.items():
        feasible = t[t["feasible"]]
        result["top"][name] = t.head(5)[["tag", "score", "mean_slo_violation", "mean_call_success", "feasible"]].to_dict("records")
        if feasible.empty:
            result["ok"] = False
            continue
        b = feasible.iloc[0]
        if name == "acc":
            result["acc"] = {k: (int(b[f"m_{k}"]) if k == "n_c" else float(b[f"m_{k}"]))
                             for k in ("alpha", "beta", "epsilon", "n_c")}
        elif name == "fixed":
            result["best_fixed_w"] = int(b["m_w"])
        else:
            result["best_pacer_rate"] = float(b["m_rate"])
    return result
