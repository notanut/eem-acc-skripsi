# Review request: Colab "No module named 'eem_acc'" fix (2026-11-14)

You are an independent reviewer. Be concrete.

## Context
Colab notebooks of https://github.com/notanut/eem-acc-skripsi (src layout, setuptools build backend) clone the repo
and run `pip install -q -r requirements.txt -e <repo>` via subprocess inside the running kernel, then
`from eem_acc.utils import ...`. In Colab this raised ModuleNotFoundError: the editable install's .pth file is only
read at interpreter start. Fix: after the install, at top level, add `<repo>/src` to sys.path (guarded against
duplicates) and call importlib.invalidate_caches(); 02_live also warns to restart the runtime if git pull brought
new code while eem_acc is already imported, and guards its experiments/ path insert. Static regression tests
(stdlib json) check all three notebooks.

## Checklist
- Is the diagnosis right? Does the fix work on first run, on re-runs, and after a pull? Any case where an old
  installed copy of eem_acc could shadow the repo's src/?
- Is the restart warning reliable (HEAD comparison, sys.modules check)?
- Are the tests meaningful (they fail on the old notebooks)?

## Output
Findings by severity (location, explanation, fix), then a verdict.

## Diff
```diff
diff --git a/notebooks/00_setup.ipynb b/notebooks/00_setup.ipynb
index 164d5f1..d8996fd 100644
--- a/notebooks/00_setup.ipynb
+++ b/notebooks/00_setup.ipynb
@@ -61,7 +61,12 @@
    "source": [
     "# Cell 3: install the package (editable) and dependencies\n",
     "subprocess.run([sys.executable, \"-m\", \"pip\", \"install\", \"-q\", \"-r\", f\"{REPO_DIR}/requirements.txt\"], check=True)\n",
-    "subprocess.run([sys.executable, \"-m\", \"pip\", \"install\", \"-q\", \"-e\", REPO_DIR], check=True)"
+    "subprocess.run([sys.executable, \"-m\", \"pip\", \"install\", \"-q\", \"-e\", REPO_DIR], check=True)\n",
+    "# Make the code importable NOW: an editable pip install only takes effect after a runtime restart in Colab.\n",
+    "import importlib, sys\n",
+    "if f\"{REPO_DIR}/src\" not in sys.path:\n",
+    "    sys.path.insert(0, f\"{REPO_DIR}/src\")\n",
+    "importlib.invalidate_caches()\n"
    ]
   },
   {
@@ -117,4 +122,4 @@
  },
  "nbformat": 4,
  "nbformat_minor": 5
-}
\ No newline at end of file
+}
diff --git a/notebooks/01_rq2.ipynb b/notebooks/01_rq2.ipynb
index a32e949..53991c4 100644
--- a/notebooks/01_rq2.ipynb
+++ b/notebooks/01_rq2.ipynb
@@ -38,6 +38,11 @@
     "else:\n",
     "    git(\"clone\", REPO_URL, REPO_DIR)\n",
     "subprocess.run([sys.executable, \"-m\", \"pip\", \"install\", \"-q\", \"-r\", f\"{REPO_DIR}/requirements.txt\", \"-e\", REPO_DIR], check=True)\n",
+    "# Make the code importable NOW: an editable pip install only takes effect after a runtime restart in Colab.\n",
+    "import importlib\n",
+    "if f\"{REPO_DIR}/src\" not in sys.path:\n",
+    "    sys.path.insert(0, f\"{REPO_DIR}/src\")\n",
+    "importlib.invalidate_caches()\n",
     "os.environ[\"EEM_ACC_DRIVE_ROOT\"] = DRIVE_ROOT\n",
     "from eem_acc.utils import Paths, set_seed\n",
     "set_seed(SEED); paths = Paths().ensure(); print(paths)"
@@ -89,4 +94,4 @@
  },
  "nbformat": 4,
  "nbformat_minor": 5
-}
\ No newline at end of file
+}
diff --git a/notebooks/02_live.ipynb b/notebooks/02_live.ipynb
index 980f8df..940a08d 100644
--- a/notebooks/02_live.ipynb
+++ b/notebooks/02_live.ipynb
@@ -33,15 +33,25 @@
     "    def git(*args, cwd=None):\n",
     "        subprocess.run([\"git\", *args], cwd=cwd, check=True)\n",
     "    if os.path.isdir(os.path.join(REPO_DIR, \".git\")):\n",
+    "        head = lambda: subprocess.run([\"git\", \"rev-parse\", \"HEAD\"], cwd=REPO_DIR, capture_output=True, text=True).stdout\n",
+    "        before = head()\n",
     "        git(\"pull\", \"--ff-only\", cwd=REPO_DIR)\n",
+    "        if head() != before and \"eem_acc\" in sys.modules:\n",
+    "            print(\"NEW CODE WAS PULLED, but the old version is still loaded.\\n\"\n",
+    "                  \">>> Runtime -> Restart session, then run this cell again. <<<\")\n",
     "    else:\n",
     "        git(\"clone\", REPO_URL, REPO_DIR)\n",
     "    subprocess.run([sys.executable, \"-m\", \"pip\", \"install\", \"-q\", \"-r\", f\"{REPO_DIR}/requirements.txt\", \"-e\", REPO_DIR], check=True)\n",
     "else:                                   # running outside Colab (local test): use this checkout + a temp folder\n",
     "    REPO_DIR = str(Path.cwd().parent) if Path.cwd().name == \"notebooks\" else str(Path.cwd())\n",
     "    DRIVE_ROOT = tempfile.mkdtemp(prefix=\"eem_acc_\")\n",
+    "# Make the code importable NOW: an editable pip install only takes effect after a runtime restart in Colab.\n",
+    "import importlib\n",
+    "if f\"{REPO_DIR}/src\" not in sys.path:\n",
     "    sys.path.insert(0, f\"{REPO_DIR}/src\")\n",
-    "sys.path.insert(0, f\"{REPO_DIR}/experiments\")\n",
+    "importlib.invalidate_caches()\n",
+    "if f\"{REPO_DIR}/experiments\" not in sys.path:\n",
+    "    sys.path.insert(0, f\"{REPO_DIR}/experiments\")\n",
     "os.environ[\"EEM_ACC_DRIVE_ROOT\"] = DRIVE_ROOT\n",
     "from eem_acc.utils import Paths, set_seed\n",
     "set_seed(42); Paths().ensure()\n",
diff --git a/tests/test_notebooks.py b/tests/test_notebooks.py
index 22f7437..3b0ef95 100644
--- a/tests/test_notebooks.py
+++ b/tests/test_notebooks.py
@@ -25,3 +25,4 @@ def test_live_notebook_executes_with_small_limit():
                             resources={"metadata": {"path": str(NB_DIR)}}).execute()
     errors = [o for c in nb.cells if c.cell_type == "code" for o in c.outputs if o.output_type == "error"]
     assert not errors
+
```
