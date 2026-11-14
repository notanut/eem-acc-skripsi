# Review request: RQ2 dev-phase selection and the D34 fallback, 2026-11-14

You are an independent reviewer (experimental methodology). Be concrete; no praise.

## Context
Thesis RQ2 (https://github.com/notanut/eem-acc-skripsi, design docs/rq2_design_v2.md). The dev phase (16,200 simulation
runs, dev seeds 0-4) finished. Pre-registered selection: a candidate is feasible if, in EVERY loaded cell, SLO
violation <= 5%, call success >= 99% and overload rate <= 10%; per family the feasible candidate with the highest
normalised goodput is chosen; if a family has none, stop and ask. Result: 56 of 243 ACC configs are feasible (all with
kappa = 0.8); NO fixed window and NO token-bucket rate is feasible in all 8 loaded cells (small values keep the SLO in
the small-capacity scenario but cannot clear the open-loop burst backlog; large values break the SLO). The author
decided D34: the GLOBAL fixed-window and token-bucket baselines fall back to the closest-to-feasible value (most loaded
cells feasible, then highest score), reported as infeasible; no fallback for ACC or for the per-scenario best fixed
window. The frozen configuration (committed before any test-seed run) is below, followed by the code diff.

## Checklist
1. Is D34 a fair, defensible rule, or does it hand ACC an easy win in comparison P1 (W=4 is the most cautious window)?
   What should the paper say, and what extra descriptive comparison would you add?
2. All feasible ACC configs have kappa = 0.8, a value first found in calibration. How should this be disclosed?
3. Code correctness of the fallback and its reporting (LOSO folds, robustness matrix labels).
4. Anything that should change BEFORE the test phase (test seeds 100-109) is run.

## Output
Findings by severity (location, explanation, fix), then a verdict on starting the test phase.

## Frozen configuration (excerpt of configs/rq2_frozen.json)
```json
{
 "acc": {
  "alpha": 0.25,
  "beta": 0.25,
  "epsilon": 0.05,
  "kappa": 0.8,
  "n_c": 32
 },
 "acc_kappa1": {
  "config": {
   "alpha": 0.75,
   "beta": 0.5,
   "epsilon": 0.05,
   "kappa": 1.0,
   "n_c": 32
  },
  "feasible": false,
  "score": 0.9185247464231556
 },
 "best_fixed_w": 4,
 "best_pacer_rate": 15.0,
 "baseline_fallback": {
  "fixed": {
   "feasible": false,
   "mean_slo_violation": 0.0006473142485043991,
   "n_feasible_cells": 6,
   "n_loaded_cells": 8,
   "score": 0.503187444735263,
   "tag": "fixed_w4",
   "value": 4
  },
  "tokenbucket": {
   "feasible": false,
   "mean_slo_violation": 0.003831210412064019,
   "n_feasible_cells": 6,
   "n_loaded_cells": 8,
   "score": 0.38976920259478764,
   "tag": "tb_r15",
   "value": 15.0
  }
 },
 "best_fixed_w_per_scenario": {
  "CAP-burst": 24,
  "CAP-drop": 12,
  "CAP-steady": 4,
  "RATE-steady": 20
 },
 "denominators": {
  "CAP-burst|32": 105.30499999999999,
  "CAP-burst|96": 116.10333333333332,
  "CAP-drop|32": 78.91833333333332,
  "CAP-drop|96": 81.88666666666667,
  "CAP-steady|32": 19.330000000000002,
  "CAP-steady|8": 19.225,
  "CAP-steady|96": 19.333333333333336,
  "RATE-steady|96": 99.72
 },
 "criterion": {
  "loaded_cells": [
   "CAP-steady|8",
   "CAP-steady|32",
   "CAP-steady|96",
   "RATE-steady|96",
   "CAP-drop|32",
   "CAP-drop|96",
   "CAP-burst|32",
   "CAP-burst|96"
  ],
  "max_slo_violation": 0.05,
  "max_throttle_rate": 0.1,
  "min_call_success": 0.99,
  "objective": "maximise mean normalised goodput over loaded cells",
  "tie_tolerance": 0.005
 },
 "dev_seeds": [
  0,
  1,
  2,
  3,
  4
 ],
 "n_dev_runs": 16200,
 "sim_version": "6",
 "dev_digest": "9994e60265900428"
}
```

## Code diff
```diff
diff --git a/experiments/run_rq2.py b/experiments/run_rq2.py
index 773ea39..6fef3d2 100644
--- a/experiments/run_rq2.py
+++ b/experiments/run_rq2.py
@@ -149,6 +149,9 @@ def do_select(root: Path, force: bool = False) -> int:
     df = load_results(root / "results" / "rq2" / "dev" / "units")
     res = select_frozen(df)
     res = {**res, "sim_version": SIM_VERSION, "dev_digest": digest(hashes)}
+    for fam, fb in res.get("baseline_fallback", {}).items():
+        print(f"[select] NOTE (D34): no {fam} value is feasible in every loaded cell; using the closest to feasible, "
+              f"{fb['tag']} (feasible in {fb['n_feasible_cells']} of {fb['n_loaded_cells']} loaded cells), reported as infeasible.")
     print(json.dumps(res, indent=2, default=str))
     if not res["ok"]:
         print("[select] NO feasible candidate in at least one family; nothing written. Ask the user how to proceed.")
diff --git a/src/eem_acc/analysis.py b/src/eem_acc/analysis.py
index 173bca6..89b6d5e 100644
--- a/src/eem_acc/analysis.py
+++ b/src/eem_acc/analysis.py
@@ -7,7 +7,9 @@ Dev selection (per family: ACC, fixed W, token bucket):
 * score = mean over scored loaded cells of goodput / denominator, denominator = best goodput of any feasible dev
   method in that cell (frozen with the selection; cells with no feasible method are excluded from scoring);
 * ties: score >= (1 - TIE_TOLERANCE) * best; then lower mean SLO violation, lower mean throttle rate, then a fixed
-  lexicographic order (conservative first). Constraints are never relaxed: no feasible candidate -> ok = False.
+  lexicographic order. Constraints are never relaxed for ACC (no feasible ACC -> ok = False) nor for the per-scenario
+  best fixed W. D34: a GLOBAL fixed-W / token-bucket baseline with no feasible value is the closest-to-feasible value
+  (most loaded cells feasible, then score), recorded in `baseline_fallback` and reported as infeasible.
 Test analysis: robustness matrix, pre-registered primary comparisons (exact paired sign-flip test over seeds, Holm),
 leave-one-scenario-out (on dev data of the left-out scenario), Pareto front.
 """
@@ -166,13 +168,37 @@ def _select(df: pd.DataFrame, loaded: list[tuple[str, int]]) -> dict[str, Any]:
         res["top"][fam] = scored[scored["m_kind"] == fam].head(5)[
             ["tag", "score", "mean_slo_violation", "mean_throttle_rate", "mean_call_success", "feasible"]].to_dict("records")
         row = choose(scored, fam)
-        if row is None:
+        if row is not None:
+            res[key] = _params(row, fam)
+            continue
+        if fam == "acc":
+            res["ok"] = False                  # the frozen ACC must be feasible: never a fallback
+            continue
+        # D34: global fixed-W / token-bucket baseline with no feasible value -> closest to feasible
+        fb = closest_to_feasible(scored, fam, loaded)
+        if fb is None:
             res["ok"] = False
         else:
-            res[key] = _params(row, fam)
+            res[key] = fb["value"]
+            res.setdefault("baseline_fallback", {})[fam] = fb
     return res, scored
 
 
+def closest_to_feasible(scored: pd.DataFrame, family: str, loaded: list[tuple[str, int]]) -> dict[str, Any] | None:
+    """D34 (user, 2026-11-14): when no value of a baseline family is feasible in every loaded cell, take the value
+    feasible in the most loaded cells, then the highest score, then the lowest mean SLO violation, then the tag;
+    complete candidates only. It is reported as infeasible."""
+    t = scored[(scored["m_kind"] == family) & scored["complete"]]
+    if t.empty:
+        return None
+    row = t.sort_values(["n_feasible_cells", "score", "mean_slo_violation", "tag"],
+                        ascending=[False, False, True, True]).iloc[0]
+    score = float(row["score"]) if pd.notna(row["score"]) else None
+    return {"value": _params(row, family), "tag": row["tag"], "feasible": False,
+            "n_feasible_cells": int(row["n_feasible_cells"]), "n_loaded_cells": len(loaded),
+            "score": score, "mean_slo_violation": float(row["mean_slo_violation"])}
+
+
 def select_frozen(df: pd.DataFrame) -> dict[str, Any]:
     """Frozen configuration from DEV results (all loaded cells), plus the per-scenario best fixed W (oracle for the
     robustness matrix), the Pareto front and leave-one-scenario-out selections."""
@@ -190,7 +216,7 @@ def select_frozen(df: pd.DataFrame) -> dict[str, Any]:
         if not sub:
             continue                  # no loaded cell: the window choice does not matter there (no oracle needed)
         r, _ = _select(df[df["scenario"] == scn], sub)
-        if "best_fixed_w" in r:
+        if "best_fixed_w" in r and "fixed" not in r.get("baseline_fallback", {}):   # the oracle must be feasible
             per[scn] = r["best_fixed_w"]
         else:
             res["ok"] = False
@@ -230,13 +256,16 @@ def leave_one_scenario_out(dev: pd.DataFrame) -> dict[str, Any]:
         dn = denominators(cells, held)
         for fam, key in (("acc", "acc"), ("fixed", "best_fixed_w"), ("tokenbucket", "best_pacer_rate")):
             if key not in sel:
-                ev[fam] = None
+                ev[fam] = {"config": None, "training_ok": bool(sel["ok"]),
+                           "reason": "no feasible candidate on the training scenarios"}
                 continue
             tag = S.method_tag(S.acc_method(**sel[key]) if fam == "acc" else
                                S.fixed_method(sel[key]) if fam == "fixed" else S.pacer_method(sel[key]))
             c = cells[cells["tag"] == tag].copy()
             c["norm"] = [g / dn.get(f"{a}|{int(m)}", np.nan) for g, a, m in zip(c["goodput"], c["scenario"], c["M"])]
-            ev[fam] = {"config": sel[key], "tag": tag, "score_on_left_out": float(c["norm"].mean()) if len(c) else None,
+            ev[fam] = {"config": sel[key], "tag": tag, "training_ok": bool(sel["ok"]),
+                       "training_fallback": sel.get("baseline_fallback", {}).get(fam),
+                       "score_on_left_out": float(c["norm"].mean()) if len(c) else None,
                        "feasible_on_left_out": bool(c["cell_feasible"].all()) if len(c) else None,
                        "mean_slo_violation": float(c["slo_violation_rate"].mean()) if len(c) else None,
                        "mean_throttle_rate": float(c["throttle_rate"].mean()) if len(c) else None}
@@ -254,16 +283,27 @@ def robustness_matrix(test: pd.DataFrame, frozen: dict) -> pd.DataFrame:
     cells = cells[_loaded_mask(cells, loaded)].copy()
     dn = frozen["denominators"]
     cells["norm"] = [g / dn.get(f"{s}|{int(m)}", np.nan) for g, s, m in zip(cells["goodput"], cells["scenario"], cells["M"])]
+    fb = frozen.get("baseline_fallback", {})
+    w_lab = " (global, closest to feasible: infeasible in some loaded cells)" if "fixed" in fb else " (global)"
+    r_lab = " (closest to feasible: infeasible in some loaded cells)" if "tokenbucket" in fb else ""
     rows = {"ACC (frozen)": S.method_tag(S.acc_method(**frozen["acc"])),
-            f"fixed W={frozen['best_fixed_w']} (global)": S.method_tag(S.fixed_method(frozen["best_fixed_w"])),
-            f"token bucket {frozen['best_pacer_rate']:g}/s": S.method_tag(S.pacer_method(frozen["best_pacer_rate"]))}
+            f"fixed W={frozen['best_fixed_w']}{w_lab}": S.method_tag(S.fixed_method(frozen["best_fixed_w"])),
+            f"token bucket {frozen['best_pacer_rate']:g}/s{r_lab}": S.method_tag(S.pacer_method(frozen["best_pacer_rate"]))}
     a = frozen["acc"]
     if a.get("kappa", 1.0) != 1.0:
         rows["ACC (frozen, kappa = 1: ablation)"] = S.method_tag(S.acc_method(**{**a, "kappa": 1.0}))
     if frozen.get("acc_kappa1"):
         rows["ACC (best kappa = 1 on dev)"] = S.method_tag(S.acc_method(**frozen["acc_kappa1"]["config"]))
+    by_tag = {t: lab for lab, t in rows.items()}
     for scn, w in frozen["best_fixed_w_per_scenario"].items():
-        rows[f"fixed W={w} (best for {scn})"] = S.method_tag(S.fixed_method(w))
+        t = S.method_tag(S.fixed_method(w))
+        if t in by_tag:                                   # same method as a row above: one row, merged label
+            old = by_tag[t]
+            rows[f"{old}; best for {scn}"] = rows.pop(old)
+            by_tag[t] = f"{old}; best for {scn}"
+        else:
+            rows[f"fixed W={w} (best for {scn})"] = t
+            by_tag[t] = f"fixed W={w} (best for {scn})"
     out = []
     for label, tag in rows.items():
         for scn in S.SCENARIOS:
```
