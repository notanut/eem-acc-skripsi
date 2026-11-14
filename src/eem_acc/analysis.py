"""RQ2 analysis for design v2 (docs/rq2_design_v2.md sections 5-8; decisions D20-D23).

Dev selection (per family: ACC, fixed W, token bucket):
* loaded cells are fixed a priori (scenarios.loaded_cells());
* a cell result is feasible if SLO violation <= 5%, call success >= 99% and throttle rate <= 10% (seed means);
* a candidate is feasible if it is feasible in EVERY loaded cell;
* score = mean over scored loaded cells of goodput / denominator, denominator = best goodput of any feasible dev
  method in that cell (frozen with the selection; cells with no feasible method are excluded from scoring);
* ties: score >= (1 - TIE_TOLERANCE) * best; then lower mean SLO violation, lower mean throttle rate, then a fixed
  lexicographic order. Constraints are never relaxed for ACC (no feasible ACC -> ok = False) nor for the per-scenario
  best fixed W. D34: a GLOBAL fixed-W / token-bucket baseline with no feasible value is the closest-to-feasible value
  (most loaded cells feasible, then score), recorded in `baseline_fallback` and reported as infeasible.
Test analysis: robustness matrix, pre-registered primary comparisons (exact paired sign-flip test over seeds, Holm),
leave-one-scenario-out (on dev data of the left-out scenario), Pareto front.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import scenarios as S
from .utils import ResultStore

METRICS = ["goodput", "slo_violation_rate", "call_success_rate", "throttle_rate"]
FAMILY_ORDER = {  # last-resort lexicographic tie-break: kappa desc (prefer the paper's base value), then n_c desc,
                  # alpha/beta/eps asc; fixed W and rate ascending (smaller first)
    "acc": [("m_kappa", False), ("m_n_c", False), ("m_alpha", True), ("m_beta", True), ("m_epsilon", True)],
    "fixed": [("m_w", True)],
    "tokenbucket": [("m_rate", True)],
}


# ------------------------------------------------------------------------------------------------ loading
def load_results(units_dir: str | Path) -> pd.DataFrame:
    rows = []
    for uid, res in ResultStore(units_dir).iter_results():
        meta, m = res["meta"], res["metrics"]
        row = {"unit_id": uid, "scenario": meta["scenario"], "M": meta["M"], "seed": meta["seed"],
               "tag": S.method_tag(meta["method"]), **{f"m_{k}": v for k, v in meta["method"].items()}}
        row.update({k: v for k, v in m.items() if not isinstance(v, dict)})
        for wname, ws in (m.get("windows") or {}).items():
            row.update({f"win_{wname}_{k}": v for k, v in ws.items()})
        if m.get("recovery"):
            row["recovery_status"] = m["recovery"]["status"]
            row["recovery_time"] = m["recovery"]["time"]
        rows.append(row)
    return pd.DataFrame(rows)


def _worst_fill(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["slo_violation_rate"] = d["slo_violation_rate"].fillna(1.0)   # nothing judged = worst case
    d["call_success_rate"] = d["call_success_rate"].fillna(0.0)
    d["throttle_rate"] = d["throttle_rate"].fillna(1.0)
    return d


def cell_means(df: pd.DataFrame) -> pd.DataFrame:
    """Seed-averaged core metrics per (scenario, M, tag), with the method parameters."""
    d = _worst_fill(df)
    keys = ["scenario", "M", "tag"]
    g = d.groupby(keys)
    out = g[METRICS].mean()
    out["n_seeds"] = g.size()
    params = [c for c in d.columns if c.startswith("m_")]
    if params:
        out = out.join(g[params].first())
    out = out.reset_index()
    out["cell_feasible"] = ((out["slo_violation_rate"] <= S.MAX_SLO_VIOLATION)
                            & (out["call_success_rate"] >= S.MIN_CALL_SUCCESS)
                            & (out["throttle_rate"] <= S.MAX_THROTTLE_RATE))
    return out


def _loaded_mask(cells: pd.DataFrame, loaded: list[tuple[str, int]]) -> pd.Series:
    keys = set(loaded)
    return pd.Series([(s, int(m)) in keys for s, m in zip(cells["scenario"], cells["M"])], index=cells.index)


def denominators(cells: pd.DataFrame, loaded: list[tuple[str, int]]) -> dict[str, float]:
    """Best goodput of any cell-feasible method per loaded cell ('scenario|M' -> value). Cells with no feasible
    method are absent (excluded from scoring and reported)."""
    out = {}
    for (s, m) in loaded:
        c = cells[(cells["scenario"] == s) & (cells["M"] == m) & cells["cell_feasible"]]
        if len(c):
            out[f"{s}|{int(m)}"] = float(c["goodput"].max())
    return out


def score_methods(df: pd.DataFrame, loaded: list[tuple[str, int]] | None = None,
                  denoms: dict[str, float] | None = None) -> pd.DataFrame:
    """One row per method over the loaded cells: score (mean normalised goodput over scored cells), feasibility
    (feasible in every loaded cell, all loaded cells present), mean SLO violation / success / throttle."""
    loaded = S.loaded_cells() if loaded is None else loaded
    if df.empty or not loaded:
        raise ValueError("score_methods needs results and at least one loaded cell")
    cells = cell_means(df)
    cells = cells[_loaded_mask(cells, loaded)].copy()
    denoms = denominators(cells, loaded) if denoms is None else denoms
    cells["cell_key"] = cells["scenario"] + "|" + cells["M"].astype(int).astype(str)
    cells["denom"] = cells["cell_key"].map(denoms)
    cells["norm_goodput"] = cells["goodput"] / cells["denom"]
    g = cells.groupby("tag")
    out = pd.DataFrame({
        "score": g["norm_goodput"].mean(),             # NaN denominators (unscored cells) are skipped
        "mean_slo_violation": g["slo_violation_rate"].mean(),
        "mean_call_success": g["call_success_rate"].mean(),
        "mean_throttle_rate": g["throttle_rate"].mean(),
        "n_cells": g.size(),
        "n_feasible_cells": g["cell_feasible"].sum(),
    })
    params = [c for c in cells.columns if c.startswith("m_")]
    out = out.join(g[params].first()).reset_index()
    out["complete"] = out["n_cells"] == len(loaded)
    out["feasible"] = out["complete"] & (out["n_feasible_cells"] == len(loaded))
    return out.sort_values(["feasible", "score", "mean_slo_violation", "tag"],
                           ascending=[False, False, True, True]).reset_index(drop=True)


def choose(scored: pd.DataFrame, family: str) -> pd.Series | None:
    """Best feasible candidate of a family with the documented tie rules; None if none is feasible."""
    t = scored[(scored["m_kind"] == family) & scored["feasible"]]
    if t.empty:
        return None
    best = t["score"].max()
    tied = t[t["score"] >= (1.0 - S.TIE_TOLERANCE) * best]
    cols = ["mean_slo_violation", "mean_throttle_rate"] + [c for c, _ in FAMILY_ORDER[family]]
    asc = [True, True] + [a for _, a in FAMILY_ORDER[family]]
    cols_present = [c for c in cols if c in tied.columns]
    asc_present = [a for c, a in zip(cols, asc) if c in tied.columns]
    return tied.sort_values(cols_present, ascending=asc_present).iloc[0]


def _params(row: pd.Series, family: str) -> Any:
    if family == "acc":
        return {"alpha": float(row["m_alpha"]), "beta": float(row["m_beta"]), "epsilon": float(row["m_epsilon"]),
                "n_c": int(row["m_n_c"]), "kappa": float(row["m_kappa"])}
    if family == "fixed":
        return int(row["m_w"])
    return float(row["m_rate"])


def secondary_global_baselines(df: pd.DataFrame, loaded: list[tuple[str, int]], denoms: dict) -> dict[str, Any]:
    """D35: the closest-to-feasible global fixed W / token bucket counted per SCENARIO (a scenario is passed when the
    value is feasible in all its loaded cells), then score; reported as a pre-registered secondary P1 comparison,
    because counting per cell (D34) depends on how many loaded cells each scenario happens to have."""
    cells = cell_means(df)
    cells = cells[_loaded_mask(cells, loaded)]
    scored = score_methods(df, loaded, denoms).set_index("tag")
    scns = sorted({s for s, _ in loaded})
    out = {}
    for fam in ("fixed", "tokenbucket"):
        rows = []
        for tag, r in scored[scored["m_kind"] == fam].iterrows():
            if not r["complete"]:
                continue
            ok_scn = sum(bool(cells[(cells["tag"] == tag) & (cells["scenario"] == s)]["cell_feasible"].all()) for s in scns)
            rows.append({"tag": tag, "n_feasible_scenarios": ok_scn, "score": float(r["score"]),
                         "mean_slo_violation": float(r["mean_slo_violation"]), **{c: r[c] for c, _ in FAMILY_ORDER[fam]}})
        if not rows:
            continue
        t = pd.DataFrame(rows)
        order = [c for c, _ in FAMILY_ORDER[fam]]
        best = t.sort_values(["n_feasible_scenarios", "score", "mean_slo_violation"] + order,
                             ascending=[False, False, True] + [a for _, a in FAMILY_ORDER[fam]]).iloc[0]
        out[fam] = {"value": int(best["m_w"]) if fam == "fixed" else float(best["m_rate"]), "tag": best["tag"],
                    "n_feasible_scenarios": int(best["n_feasible_scenarios"]), "n_scenarios": len(scns),
                    "feasible": bool(best["n_feasible_scenarios"] == len(scns)), "score": float(best["score"])}
    return out


def dev_margins(df: pd.DataFrame, tag: str, loaded: list[tuple[str, int]]) -> dict[str, Any]:
    """M4 (pre-registered 2026-11-14): per loaded cell, the frozen ACC's dev seed-mean and worst-seed values of the three
    constraint metrics and the margins to the limits. If the frozen ACC is infeasible in a TEST cell, that is
    reported as-is; there is no re-selection."""
    d = _worst_fill(df[df["tag"] == tag])
    out = {}
    for s, m in loaded:
        c = d[(d["scenario"] == s) & (d["M"] == m)]
        if c.empty:
            continue
        out[f"{s}|{m}"] = {
            "slo_mean": float(c["slo_violation_rate"].mean()), "slo_worst_seed": float(c["slo_violation_rate"].max()),
            "success_mean": float(c["call_success_rate"].mean()), "success_worst_seed": float(c["call_success_rate"].min()),
            "overload_mean": float(c["throttle_rate"].mean()), "overload_worst_seed": float(c["throttle_rate"].max()),
            "margin_slo": S.MAX_SLO_VIOLATION - float(c["slo_violation_rate"].mean()),
            "margin_success": float(c["call_success_rate"].mean()) - S.MIN_CALL_SUCCESS,
            "margin_overload": S.MAX_THROTTLE_RATE - float(c["throttle_rate"].mean())}
    return out


def boundary_report(dev: pd.DataFrame, boundary: pd.DataFrame, frozen: dict) -> pd.DataFrame:
    """D37: frozen ACC vs the one-step-past-the-edge variants on DEV seeds (sensitivity only)."""
    loaded = [(s, int(m)) for s, m in (x.split("|") for x in frozen["criterion"]["loaded_cells"])]
    both = pd.concat([dev[dev["tag"] == S.method_tag(S.acc_method(**frozen["acc"]))], boundary])
    sc = score_methods(both, loaded, frozen["denominators"])
    return sc[["tag", "score", "feasible", "n_feasible_cells", "mean_slo_violation", "mean_throttle_rate",
               "mean_call_success"]].reset_index(drop=True)


def pareto_front(scored: pd.DataFrame) -> list[dict[str, Any]]:
    """Non-dominated methods on (higher score, lower mean SLO violation), complete methods only."""
    t = scored[scored["complete"]].dropna(subset=["score"])
    pts = t[["tag", "score", "mean_slo_violation", "feasible"]].to_dict("records")
    front = [p for p in pts if not any(q["score"] >= p["score"] and q["mean_slo_violation"] <= p["mean_slo_violation"]
                                       and (q["score"] > p["score"] or q["mean_slo_violation"] < p["mean_slo_violation"])
                                       for q in pts)]
    return sorted(front, key=lambda p: -p["score"])


def _select(df: pd.DataFrame, loaded: list[tuple[str, int]]) -> dict[str, Any]:
    cells = cell_means(df)
    denoms = denominators(cells[_loaded_mask(cells, loaded)], loaded)
    scored = score_methods(df, loaded, denoms)
    res: dict[str, Any] = {"ok": True, "denominators": denoms,
                           "unscored_cells": [f"{s}|{m}" for s, m in loaded if f"{s}|{m}" not in denoms],
                           "top": {}}
    for fam, key in (("acc", "acc"), ("fixed", "best_fixed_w"), ("tokenbucket", "best_pacer_rate")):
        res["top"][fam] = scored[scored["m_kind"] == fam].head(5)[
            ["tag", "score", "mean_slo_violation", "mean_throttle_rate", "mean_call_success", "feasible"]].to_dict("records")
        row = choose(scored, fam)
        if row is not None:
            res[key] = _params(row, fam)
            continue
        if fam == "acc":
            res["ok"] = False                  # the frozen ACC must be feasible: never a fallback
            continue
        # D34: global fixed-W / token-bucket baseline with no feasible value -> closest to feasible
        fb = closest_to_feasible(scored, fam, loaded)
        if fb is None:
            res["ok"] = False
        else:
            res[key] = fb["value"]
            res.setdefault("baseline_fallback", {})[fam] = fb
    return res, scored


def closest_to_feasible(scored: pd.DataFrame, family: str, loaded: list[tuple[str, int]]) -> dict[str, Any] | None:
    """D34 (user, 2026-11-14): when no value of a baseline family is feasible in every loaded cell, take the value
    feasible in the most loaded cells, then the highest score, then the lowest mean SLO violation, then the tag;
    complete candidates only. It is reported as infeasible."""
    t = scored[(scored["m_kind"] == family) & scored["complete"]]
    if t.empty:
        return None
    order = [c for c, _ in FAMILY_ORDER[family] if c in t.columns]
    asc = [a for c, a in FAMILY_ORDER[family] if c in t.columns]
    row = t.sort_values(["n_feasible_cells", "score", "mean_slo_violation"] + order,
                        ascending=[False, False, True] + asc).iloc[0]
    score = float(row["score"]) if pd.notna(row["score"]) else None
    return {"value": _params(row, family), "tag": row["tag"], "feasible": False,
            "n_feasible_cells": int(row["n_feasible_cells"]), "n_loaded_cells": len(loaded),
            "score": score, "mean_slo_violation": float(row["mean_slo_violation"])}


def select_frozen(df: pd.DataFrame) -> dict[str, Any]:
    """Frozen configuration from DEV results (all loaded cells), plus the per-scenario best fixed W (oracle for the
    robustness matrix), the Pareto front and leave-one-scenario-out selections."""
    loaded = S.loaded_cells()
    res, scored = _select(df, loaded)
    res["criterion"] = {"objective": "maximise mean normalised goodput over loaded cells",
                        "max_slo_violation": S.MAX_SLO_VIOLATION, "min_call_success": S.MIN_CALL_SUCCESS,
                        "max_throttle_rate": S.MAX_THROTTLE_RATE, "tie_tolerance": S.TIE_TOLERANCE,
                        "loaded_cells": [f"{s}|{m}" for s, m in loaded]}
    res["dev_seeds"] = sorted(int(x) for x in df["seed"].unique())
    res["n_dev_runs"] = int(len(df))
    per = {}
    for scn in S.SCENARIOS:
        sub = [(s, m) for s, m in loaded if s == scn]
        if not sub:
            continue                  # no loaded cell: the window choice does not matter there (no oracle needed)
        r, _ = _select(df[df["scenario"] == scn], sub)
        if "best_fixed_w" in r and "fixed" not in r.get("baseline_fallback", {}):   # the oracle must be feasible
            per[scn] = r["best_fixed_w"]
        else:
            res["ok"] = False
    res["best_fixed_w_per_scenario"] = per
    # D27: the best kappa = 1 ACC config (the paper's base algorithm at its best); if none is feasible, the best-scoring
    # one is taken and marked infeasible (it is a reference method, not a candidate for the frozen ACC)
    k1 = scored[(scored["m_kind"] == "acc") & (scored["m_kappa"] == 1.0)]
    if len(k1):
        row = choose(k1, "acc")
        feasible = row is not None
        if row is None:                    # closest to feasible first (not the most aggressive), complete rows only
            c = k1[k1["complete"]] if k1["complete"].any() else k1
            row = c.sort_values(["n_feasible_cells", "score", "mean_slo_violation", "tag"],
                                ascending=[False, False, True, True]).iloc[0]
        res["acc_kappa1"] = {"config": _params(row, "acc"), "feasible": bool(feasible), "score": float(row["score"]),
                             "n_feasible_cells": int(row["n_feasible_cells"]), "n_loaded_cells": len(loaded)}
    res["secondary_global_baselines"] = secondary_global_baselines(df, loaded, res["denominators"])
    if "acc" in res:
        res["acc_dev_margins"] = dev_margins(df, S.method_tag(S.acc_method(**res["acc"])), loaded)
    res["pareto_front"] = pareto_front(scored)
    res["loso"] = leave_one_scenario_out(df)
    return res


def leave_one_scenario_out(dev: pd.DataFrame) -> dict[str, Any]:
    """D23. For each scenario s: select on the other scenarios' loaded cells, then evaluate the chosen configs on s's
    loaded cells using s's DEV results (s was not seen by that selection; every dev candidate has dev results)."""
    out = {}
    loaded = S.loaded_cells()
    for s in S.SCENARIOS:
        train = [(a, m) for a, m in loaded if a != s]
        held = [(a, m) for a, m in loaded if a == s]
        if not train or not held:
            out[s] = {"skipped": "no loaded cells left to train on" if not train else "no loaded cells in this scenario"}
            continue
        sel, _ = _select(dev[dev["scenario"] != s], train)
        ev = {}
        held_df = dev[dev["scenario"] == s]
        cells = cell_means(held_df)
        cells = cells[_loaded_mask(cells, held)]
        dn = denominators(cells, held)
        for fam, key in (("acc", "acc"), ("fixed", "best_fixed_w"), ("tokenbucket", "best_pacer_rate")):
            if key not in sel:
                ev[fam] = {"config": None, "feasible_on_training": False,
                           "reason": ("no feasible ACC configuration on the training scenarios" if fam == "acc"
                                      else "no complete candidate on the training scenarios")}
                continue
            tag = S.method_tag(S.acc_method(**sel[key]) if fam == "acc" else
                               S.fixed_method(sel[key]) if fam == "fixed" else S.pacer_method(sel[key]))
            c = cells[cells["tag"] == tag].copy()
            c["norm"] = [g / dn.get(f"{a}|{int(m)}", np.nan) for g, a, m in zip(c["goodput"], c["scenario"], c["M"])]
            fb = sel.get("baseline_fallback", {}).get(fam)
            ev[fam] = {"config": sel[key], "tag": tag, "feasible_on_training": fb is None,
                       "training_fallback": fb,
                       "score_on_left_out": float(c["norm"].mean()) if len(c) else None,
                       "feasible_on_left_out": bool(c["cell_feasible"].all()) if len(c) else None,
                       "mean_slo_violation": float(c["slo_violation_rate"].mean()) if len(c) else None,
                       "mean_throttle_rate": float(c["throttle_rate"].mean()) if len(c) else None}
        out[s] = ev
    return out


# ------------------------------------------------------------------------------------------------ test phase
def robustness_matrix(test: pd.DataFrame, frozen: dict) -> pd.DataFrame:
    """Each per-scenario best fixed W (and ACC, the global fixed W, the token bucket) evaluated in every scenario:
    mean over that scenario's loaded cells of goodput normalised by the frozen dev denominators, SLO, throttle."""
    loaded = [tuple(x.split("|")) for x in frozen["criterion"]["loaded_cells"]]
    loaded = [(s, int(m)) for s, m in loaded]
    cells = cell_means(test)
    cells = cells[_loaded_mask(cells, loaded)].copy()
    dn = frozen["denominators"]
    cells["norm"] = [g / dn.get(f"{s}|{int(m)}", np.nan) for g, s, m in zip(cells["goodput"], cells["scenario"], cells["M"])]
    fb = frozen.get("baseline_fallback", {})
    def _lab(f):
        return (f" (global, closest to feasible; dev: feasible in {fb[f]['n_feasible_cells']}/{fb[f]['n_loaded_cells']}"
                f" loaded cells)") if f in fb else " (global)"
    w_lab, r_lab = _lab("fixed"), _lab("tokenbucket")
    rows = {"ACC (frozen)": S.method_tag(S.acc_method(**frozen["acc"])),
            f"fixed W={frozen['best_fixed_w']}{w_lab}": S.method_tag(S.fixed_method(frozen["best_fixed_w"])),
            f"token bucket {frozen['best_pacer_rate']:g}/s{r_lab}": S.method_tag(S.pacer_method(frozen["best_pacer_rate"]))}
    a = frozen["acc"]
    if a.get("kappa", 1.0) != 1.0:
        rows["ACC (frozen, kappa = 1: ablation)"] = S.method_tag(S.acc_method(**{**a, "kappa": 1.0}))
    if frozen.get("acc_kappa1"):
        rows["ACC (best kappa = 1 on dev)"] = S.method_tag(S.acc_method(**frozen["acc_kappa1"]["config"]))
    sec = frozen.get("secondary_global_baselines", {})
    for f, mk in (("fixed", S.fixed_method), ("tokenbucket", S.pacer_method)):
        if f in sec:
            t = S.method_tag(mk(sec[f]["value"]))
            name = f"fixed W={sec[f]['value']}" if f == "fixed" else f"token bucket {sec[f]['value']:g}/s"
            lab = (f"{name} (secondary global, D35; dev: feasible in {sec[f]['n_feasible_scenarios']}/"
                   f"{sec[f]['n_scenarios']} scenarios)")
            rows[lab] = t
    by_tag = {t: lab for lab, t in rows.items()}
    for scn, w in frozen["best_fixed_w_per_scenario"].items():
        t = S.method_tag(S.fixed_method(w))
        if t in by_tag:                                   # same method as a row above: one row, merged label
            old = by_tag[t]
            rows = {(f"{k}; best for {scn}" if k == old else k): v for k, v in rows.items()}   # keep row order
            by_tag[t] = f"{old}; best for {scn}"
        else:
            rows[f"fixed W={w} (best for {scn})"] = t
            by_tag[t] = f"fixed W={w} (best for {scn})"
    out = []
    for label, tag in rows.items():
        for scn in S.SCENARIOS:
            c = cells[(cells["tag"] == tag) & (cells["scenario"] == scn)]
            if len(c):
                out.append({"method": label, "scenario": scn, "norm_goodput": c["norm"].mean(),
                            "slo_violation": c["slo_violation_rate"].mean(), "throttle_rate": c["throttle_rate"].mean(),
                            "feasible_all_loaded_cells": bool(c["cell_feasible"].all())})
    return pd.DataFrame(out)


def test_descriptive(test: pd.DataFrame, frozen: dict) -> pd.DataFrame:
    """D36 (frozen before the test phase): every test method scored like the dev selection, but with the FROZEN dev
    denominators: score (mean normalised goodput over scored loaded cells), number of loaded cells in which it is
    feasible on the TEST seeds, mean and worst-cell SLO violation, mean overload rate, mean call success. Descriptive
    only (no significance tests); this is the static frontier shown next to the frozen ACC."""
    loaded = [(s, int(m)) for s, m in (x.split("|") for x in frozen["criterion"]["loaded_cells"])]
    cells = cell_means(test)
    cells = cells[_loaded_mask(cells, loaded)].copy()
    dn = frozen["denominators"]
    cells["norm"] = [g / dn.get(f"{s}|{int(m)}", np.nan) for g, s, m in zip(cells["goodput"], cells["scenario"], cells["M"])]
    g = cells.groupby("tag")
    out = pd.DataFrame({"score": g["norm"].mean(), "n_feasible_cells": g["cell_feasible"].sum().astype(int),
                        "n_cells": g.size(), "mean_slo_violation": g["slo_violation_rate"].mean(),
                        "worst_cell_slo_violation": g["slo_violation_rate"].max(),
                        "mean_overload_rate": g["throttle_rate"].mean(), "mean_call_success": g["call_success_rate"].mean()})
    out["feasible_all_loaded_cells"] = out["n_feasible_cells"] == len(loaded)
    out["kind"] = g["m_kind"].first() if "m_kind" in cells.columns else None
    return out.reset_index().sort_values(["kind", "score"], ascending=[True, False]).reset_index(drop=True)


def sign_flip_test(d: np.ndarray) -> float:
    """Exact two-sided paired randomisation test of mean(d) = 0: all 2^n sign patterns (n <= 20)."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n == 0 or n > 20:
        raise ValueError("need 1..20 paired differences")
    obs = abs(d.mean())
    signs = np.array(list(itertools.product((1.0, -1.0), repeat=n)))
    means = np.abs((signs * d).mean(axis=1))
    return float(np.mean(means >= obs - 1e-12))


_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
         11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
         20: 2.086, 25: 2.060, 30: 2.042}


def tost_sign_flip(d: np.ndarray, margin: float) -> float:
    """Equivalence (TOST) with one-sided sign-flip tests: H0a mean <= -margin, H0b mean >= +margin.
    p = max(p_a, p_b); equivalence is shown at level a if p <= a. Exact only under SYMMETRY of the paired differences
    about the boundary (unlike the test of no difference, which only needs exchangeable pairs); the paired-t TOST
    (90% CI inside +-margin) is reported next to it as a robustness check."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n == 0 or n > 20:
        raise ValueError("need 1..20 paired differences")
    signs = np.array(list(itertools.product((1.0, -1.0), repeat=n)))
    up, dn = d + margin, d - margin
    p_a = float(np.mean((signs * up).mean(axis=1) >= up.mean() - 1e-12))    # small if mean(d) clearly > -margin
    p_b = float(np.mean((signs * dn).mean(axis=1) <= dn.mean() + 1e-12))    # small if mean(d) clearly < +margin
    return max(p_a, p_b)


_T95 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812,
        11: 1.796, 12: 1.782, 13: 1.771, 14: 1.761, 15: 1.753, 16: 1.746, 17: 1.740, 18: 1.734, 19: 1.729,
        20: 1.725, 25: 1.708, 30: 1.697}


def t_ci90(d: np.ndarray) -> tuple[float, float]:
    """90% paired t interval (the TOST at alpha = 0.05 shows equivalence iff it lies inside +-margin)."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n < 2:
        return (math.nan, math.nan)
    df = n - 1
    q = _T95.get(df) or (_T95[max(k for k in _T95 if k <= df)] if df < 30 else 1.645)
    h = q * d.std(ddof=1) / math.sqrt(n)
    return (float(d.mean() - h), float(d.mean() + h))


def t_ci(d: np.ndarray) -> tuple[float, float]:
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n < 2:
        return (math.nan, math.nan)
    df = n - 1
    q = _T975.get(df) or (_T975[max(k for k in _T975 if k <= df)] if df < 30 else 1.96)
    h = q * d.std(ddof=1) / math.sqrt(n)
    return (float(d.mean() - h), float(d.mean() + h))


def holm(pvals: dict[str, float]) -> dict[str, float]:
    keys = sorted(pvals, key=pvals.get)
    m, out, running = len(keys), {}, 0.0
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        out[k] = running
    return out


def _per_seed(test: pd.DataFrame, tag: str, cells_: list[tuple[str, int]], dn: dict, metric: str) -> pd.Series:
    t = _worst_fill(test[test["tag"] == tag])
    t = t[[(s, int(m)) in set(cells_) for s, m in zip(t["scenario"], t["M"])]].copy()
    if metric == "norm_goodput":
        t["v"] = [g / dn.get(f"{s}|{int(m)}", np.nan) for g, s, m in zip(t["goodput"], t["scenario"], t["M"])]
    else:
        t["v"] = t[metric]
    return t.groupby("seed")["v"].mean()


def primary_comparisons(test: pd.DataFrame, frozen: dict) -> pd.DataFrame:
    """Pre-registered endpoints (spec section 8). Difference = ACC - baseline, per test seed (paired via CRN):
    P1a/P1b normalised goodput over all scored loaded cells vs the global fixed W / tuned token bucket;
    P2-<scenario> normalised goodput over that scenario's loaded cells vs that scenario's best fixed W;
    P2-RATE-steady-throttle: throttle rate in RATE-steady's loaded cells vs its best fixed W.
    Exact sign-flip p-values, Holm-adjusted over all endpoints; 95% paired t CI of the mean difference."""
    loaded = [(s, int(m)) for s, m in (x.split("|") for x in frozen["criterion"]["loaded_cells"])]
    scored = [c for c in loaded if f"{c[0]}|{c[1]}" in frozen["denominators"]]
    dn = frozen["denominators"]
    acc = S.method_tag(S.acc_method(**frozen["acc"]))
    ends = {
        "P1a ACC - global fixed W": (S.method_tag(S.fixed_method(frozen["best_fixed_w"])), scored, "norm_goodput", "ACC higher"),
        "P1b ACC - token bucket": (S.method_tag(S.pacer_method(frozen["best_pacer_rate"])), scored, "norm_goodput", "ACC higher"),
    }
    expect = {"CAP-steady": "ACC lower", "RATE-steady": "about equal", "CAP-drop": "ACC higher", "CAP-burst": "ACC lower"}
    for scn, w in frozen["best_fixed_w_per_scenario"].items():
        sub = [c for c in scored if c[0] == scn]
        ends[f"P2 {scn}: ACC - best fixed W={w}"] = (S.method_tag(S.fixed_method(w)), sub, "norm_goodput", expect.get(scn, ""))
    if "RATE-steady" in frozen["best_fixed_w_per_scenario"]:
        w = frozen["best_fixed_w_per_scenario"]["RATE-steady"]
        sub = [c for c in loaded if c[0] == "RATE-steady"]
        ends[f"P2 RATE-steady throttle: ACC - best fixed W={w}"] = (S.method_tag(S.fixed_method(w)), sub, "throttle_rate", "ACC lower")
    rows, pv = [], {}
    for name, (base, cells_, metric, exp) in ends.items():
        a, b = _per_seed(test, acc, cells_, dn, metric), _per_seed(test, base, cells_, dn, metric)
        d = (a - b).dropna()
        equivalence = exp == "about equal"                      # D25: TOST with +-EQUIVALENCE_MARGIN
        if not len(d):
            p = math.nan
        elif equivalence:
            p = tost_sign_flip(d.values, S.EQUIVALENCE_MARGIN)
        else:
            p = sign_flip_test(d.values)
        pv[name] = p
        lo, hi = t_ci(d.values)
        row = {"endpoint": name, "expected": exp, "test": f"TOST +-{S.EQUIVALENCE_MARGIN:g}" if equivalence else "two-sided",
               "n_seeds": len(d), "mean_diff": float(d.mean()) if len(d) else math.nan,
               "ci95_low": lo, "ci95_high": hi, "p_value": p}
        if equivalence:                     # a TOST rejection at 0.05 means "equivalent"; t robustness check
            l90, h90 = t_ci90(d.values)
            row.update(ci90_low=l90, ci90_high=h90,
                       t_tost_equivalent=bool(-S.EQUIVALENCE_MARGIN < l90 and h90 < S.EQUIVALENCE_MARGIN))
        rows.append(row)
    adj = holm({k: v for k, v in pv.items() if not math.isnan(v)})
    for r in rows:
        r["p_holm"] = adj.get(r["endpoint"], math.nan)
        r["family"] = "primary"
    # D35: pre-registered SECONDARY P1 comparisons (outside the Holm family), per-scenario-counted global baselines
    sec = frozen.get("secondary_global_baselines", {})
    for key, mk, lab in (("fixed", S.fixed_method, "P1a' ACC - secondary global fixed W"),
                         ("tokenbucket", S.pacer_method, "P1b' ACC - secondary token bucket")):
        if key not in sec:
            continue
        base = S.method_tag(mk(sec[key]["value"]))
        a, b = _per_seed(test, acc, scored, dn, "norm_goodput"), _per_seed(test, base, scored, dn, "norm_goodput")
        d = (a - b).dropna()
        lo, hi = t_ci(d.values)
        rows.append({"endpoint": f"{lab}={sec[key]['value']:g}", "expected": "ACC higher", "test": "two-sided",
                     "n_seeds": len(d), "mean_diff": float(d.mean()) if len(d) else math.nan, "ci95_low": lo,
                     "ci95_high": hi, "p_value": sign_flip_test(d.values) if len(d) else math.nan,
                     "p_holm": math.nan, "family": "secondary"})
    return pd.DataFrame(rows)
