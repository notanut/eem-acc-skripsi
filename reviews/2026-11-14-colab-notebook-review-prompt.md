# Review request: Colab live notebook, per-run results folders (2026-11-14)

You are an independent reviewer. Be skeptical and concrete.

## Context
Thesis project (adaptive concurrency control for LLM agents calling APIs). RQ2 runs a deterministic simulation in
Google Colab (free CPU runtime) from `notebooks/02_live.ipynb`; results go to Google Drive. The runner
`experiments/run_rq2.py` (public repo https://github.com/notanut/eem-acc-skripsi) has phases dev (tuning, 5,520
runs), select (choose a frozen configuration from dev results only) and test (held-out seeds). It stamps every result
with a config hash + SIM_VERSION and refuses to resume, select or test on results/frozen files from another version,
configuration or results folder. `do_select` writes the frozen config both to `configs/rq2_frozen.json` in the repo
clone and to `<root>/results/rq2/rq2_frozen.json`.

## The change (diff below)
1. Each experiment writes to `Drive/Skripsi/rq2_runs/<RUN_NAME>/` (default "trial") instead of the Drive root, so
   trial runs never mix with or block real runs.
2. Background worker threads are kept across cell re-runs and keyed by (phase, results folder); a warning lists other
   runs still working. (Stopping a cell only stops the live display, not the run.)
3. Part C explains every exit code of do_select and offers FORCE=True for code 6 (a different frozen file in the
   clone). do_select now replaces a frozen file automatically if it records an older SIM_VERSION.
4. Part D uses `run_rq2.load_frozen_for_test(ROOT)` (checks repo/Drive copies agree, version, and that the selection
   came from this folder's dev results).

## Checklist
- Can a trial run still contaminate, block or be confused with a real run? Is FORCE a foot-gun (could it let a
  selection from one run be used to test another)?
- Thread safety: two workers on one folder; a worker still running after RUN_NAME changes; Colab disconnects.
- Is auto-replacing an older-SIM_VERSION frozen file safe for the dev/test protocol (no peeking at test results)?
- Anything that breaks in Colab specifically (Drive mount, re-running cells, git pull on a dirty clone).

## Output
Findings ranked by severity (Critical/High/Medium/Low) with location, explanation, fix; then a verdict.

## Diff
```diff
diff --git a/experiments/run_rq2.py b/experiments/run_rq2.py
index eb63b0d..c40e425 100644
--- a/experiments/run_rq2.py
+++ b/experiments/run_rq2.py
@@ -145,7 +145,10 @@ def do_select(root: Path, force: bool = False) -> int:
         return 3
     if FROZEN_PATH.exists() and not force:
         old = json.loads(FROZEN_PATH.read_text())
-        if json.loads(json.dumps(res, default=str)) != old:
+        outdated = "sim_version" in old and old["sim_version"] != SIM_VERSION   # unusable anyway: test refuses it
+        if outdated:
+            print(f"[select] replacing {FROZEN_PATH}: it was selected with SIM_VERSION={old['sim_version']} (now {SIM_VERSION})")
+        elif json.loads(json.dumps(res, default=str)) != old:
             print(f"[select] refusing: {FROZEN_PATH} already exists with DIFFERENT content (another root or "
                   f"an older run). Re-run with --force only if you really mean to replace it.")
             return 6
@@ -153,7 +156,9 @@ def do_select(root: Path, force: bool = False) -> int:
     atomic_write_json(FROZEN_PATH, res)
     atomic_write_json(root / "results" / "rq2" / "dev_scores.json", score_methods(df).to_dict("records"))
     atomic_write_json(root / "results" / "rq2" / "rq2_frozen.json", res)   # Drive copy (Colab clones are ephemeral)
-    print(f"[select] wrote {FROZEN_PATH} (commit it to the repo BEFORE running the test phase)")
+    print(f"[select] wrote {FROZEN_PATH} and {root / 'results' / 'rq2' / 'rq2_frozen.json'}. Locally: commit the repo "
+          f"copy BEFORE the test phase. In Colab the clone cannot push: the Drive copy is the record (download it "
+          f"or send it to be committed).")
     return 0
 
 
diff --git a/notebooks/02_live.ipynb b/notebooks/02_live.ipynb
index 13e6176..980f8df 100644
--- a/notebooks/02_live.ipynb
+++ b/notebooks/02_live.ipynb
@@ -93,7 +93,9 @@
     "5,520 simulated runs (81 ACC settings + fixed windows + token bucket, 4 scenarios, 3 agent counts, 5 seeds). About 15 minutes on a fast machine; plan for roughly 15-40 minutes on Colab's CPU. The ETA is rough: the first runs are the light 4-agent ones, so it grows as heavier runs start.\n",
     "Top: progress and ETA. Tiles: runs finished per scenario and agent count. Table: **partial** ranking of methods from the runs finished so far (it only becomes meaningful when the phase is complete).\n",
     "\n",
-    "Set `LIMIT` to a small number (e.g. 60) for a quick trial; keep `None` for the full run. If you stop the cell, run it again to continue."
+    "Set `LIMIT` to a small number (e.g. 60) for a quick trial; keep `None` for the full run. If you stop the cell, run it again to continue.\n",
+    "\n",
+    "Results go to `Skripsi/rq2_runs/<RUN_NAME>/` on your Drive (set `RUN_NAME` in the cell below)."
    ],
    "id": "cell-04"
   },
@@ -111,12 +113,22 @@
     "from eem_acc.monitor import snapshot\n",
     "from eem_acc.viz import monitor_figure\n",
     "\n",
-    "ROOT = Path(DRIVE_ROOT)\n",
+    "# Each experiment version gets its OWN results folder, so trial runs never mix with (or block) real runs.\n",
+    "# Use a new RUN_NAME for a trial or after the simulator/design changes; reuse the same name to resume.\n",
+    "RUN_NAME = \"trial\"\n",
+    "ROOT = Path(DRIVE_ROOT) / \"rq2_runs\" / RUN_NAME\n",
     "LIMIT = None          # e.g. 60 for a quick trial\n",
-    "_WORKERS = {}         # one background worker per phase (re-running a cell never starts a duplicate)\n",
+    "print(\"results for this run go to:\", ROOT)\n",
+    "# background workers survive re-running this cell (stopping the cell only stops the DISPLAY, not the run)\n",
+    "_WORKERS = globals().get(\"_WORKERS\", {})\n",
     "\n",
     "def watch(phase, frozen=None, limit=None, refresh=5.0):\n",
-    "    w = _WORKERS.get(phase)\n",
+    "    key = (phase, str(ROOT))\n",
+    "    others = [k for k, v in _WORKERS.items() if k != key and v[\"thread\"].is_alive()]\n",
+    "    if others:\n",
+    "        print(\"WARNING: other runs are still working in the background:\", others,\n",
+    "              \"\\nThey keep using CPU. Restart the runtime (Runtime -> Restart session) to stop them.\")\n",
+    "    w = _WORKERS.get(key)\n",
     "    if w is None or not w[\"thread\"].is_alive():\n",
     "        w = {\"err\": []}\n",
     "        def work():\n",
@@ -126,7 +138,7 @@
     "                w[\"err\"].append(e)\n",
     "        w[\"thread\"] = threading.Thread(target=work, daemon=True)\n",
     "        w[\"thread\"].start()\n",
-    "        _WORKERS[phase] = w\n",
+    "        _WORKERS[key] = w\n",
     "    udir = ROOT / \"results\" / \"rq2\" / phase / \"units\"\n",
     "    while True:\n",
     "        alive = w[\"thread\"].is_alive()\n",
@@ -165,8 +177,13 @@
    "execution_count": null,
    "outputs": [],
    "source": [
-    "code_ = run_rq2.do_select(ROOT)\n",
-    "print(\"exit code:\", code_, \"(0 = configuration chosen and saved; 2 = dev phase incomplete; 3 = no feasible candidate)\")"
+    "FORCE = False   # set True ONLY if select reported code 6 and you really want to replace the old frozen file\n",
+    "code_ = run_rq2.do_select(ROOT, force=FORCE)\n",
+    "print(\"exit code:\", code_)\n",
+    "print(\"0 = chosen and saved | 2 = dev phase incomplete (expected after a LIMIT trial) | 3 = no feasible candidate\\n\"\n",
+    "      \"4 = test results already exist | 5 = dev results from another code version/grid\\n\"\n",
+    "      \"6 = configs/rq2_frozen.json in this Colab clone holds a DIFFERENT selection (e.g. from another RUN_NAME):\\n\"\n",
+    "      \"    set FORCE = True and re-run this cell to replace it with this run's selection\")\n"
    ],
    "id": "cell-07"
   },
@@ -185,14 +202,10 @@
    "execution_count": null,
    "outputs": [],
    "source": [
-    "frozen_path = (Path(REPO_DIR) / \"configs\" / \"rq2_frozen.json\")\n",
-    "if not frozen_path.exists():\n",
-    "    frozen_path = ROOT / \"results\" / \"rq2\" / \"rq2_frozen.json\"\n",
-    "if frozen_path.exists():\n",
-    "    frozen = json.loads(frozen_path.read_text())\n",
-    "    watch(\"test\", frozen=frozen, limit=LIMIT)\n",
-    "else:\n",
-    "    print(\"No frozen configuration yet: finish Part B and run Part C first.\")"
+    "frozen, msg = run_rq2.load_frozen_for_test(ROOT)   # checks version, copies and that it was selected on THIS run's dev results\n",
+    "print(msg)\n",
+    "if frozen is not None:\n",
+    "    watch(\"test\", frozen=frozen, limit=LIMIT)\n"
    ],
    "id": "cell-09"
   }
@@ -211,4 +224,4 @@
  },
  "nbformat": 4,
  "nbformat_minor": 5
-}
\ No newline at end of file
+}
```
