# Review request: "Cannot run the event loop while another loop is running" in Colab (2026-11-14)

You are an independent reviewer. Be concrete.

## Context
https://github.com/notanut/eem-acc-skripsi runs RQ2 as a deterministic virtual-time asyncio simulation
(`src/eem_acc/env/vtime.py`: a SelectorEventLoop whose clock jumps to the next timer; only asyncio.sleep /
loop.time and seeded numpy RNGs are used inside). In Google Colab (Python 3.13), a notebook cell that calls the
simulator synchronously failed because a loop is already running in the kernel's main thread. Fix: run_virtual
detects a running loop and then runs the simulation on a fresh virtual loop in a worker thread, waits for it, and
re-raises its exception. Tests run the simulator from inside `asyncio.run(...)` (reproducing the notebook
condition): they fail on the old code and pass now, and check identical results inside vs outside a running loop.

## Checklist
- Determinism and correctness of running the virtual loop off the main thread (signal handling, thread-local state,
  numpy/matplotlib thread safety, asyncio policies).
- Can blocking the notebook's main thread with join() deadlock or freeze anything? Behaviour on cell interrupt.
- Would nest_asyncio or making the API async be a better fix? Trade-offs.
- Are the tests meaningful?

## Output
Findings by severity (location, explanation, fix), then a verdict.

## Diff
```diff
diff --git a/src/eem_acc/env/vtime.py b/src/eem_acc/env/vtime.py
index 90123ae..c52f3ba 100644
--- a/src/eem_acc/env/vtime.py
+++ b/src/eem_acc/env/vtime.py
@@ -12,6 +12,7 @@ real I/O (sockets, subprocesses) inside a virtual loop.
 from __future__ import annotations
 
 import asyncio
+import threading
 from typing import Any, Awaitable, TypeVar
 
 T = TypeVar("T")
@@ -44,10 +45,35 @@ class VirtualClockEventLoop(asyncio.SelectorEventLoop):
         return self._vtime
 
 
-def run_virtual(coro: Awaitable[T]) -> T:
-    """Run `coro` to completion on a fresh virtual-time loop."""
+def _run_here(coro: Awaitable[T]) -> T:
     loop = VirtualClockEventLoop()
     try:
         return loop.run_until_complete(coro)
     finally:
         loop.close()
+
+
+def run_virtual(coro: Awaitable[T]) -> T:
+    """Run `coro` to completion on a fresh virtual-time loop.
+
+    If the calling thread already runs an event loop (Jupyter/Colab cells do), a second loop cannot be started in
+    that thread, so the simulation runs on a fresh loop in a short-lived worker thread and this call waits for it.
+    The simulation is self-contained (virtual clock, seeded RNGs), so the result is identical either way."""
+    try:
+        asyncio.get_running_loop()
+    except RuntimeError:
+        return _run_here(coro)
+    box: dict[str, Any] = {}
+
+    def work() -> None:
+        try:
+            box["value"] = _run_here(coro)
+        except BaseException as e:      # re-raised in the caller's thread
+            box["error"] = e
+
+    t = threading.Thread(target=work, name="run_virtual", daemon=True)
+    t.start()
+    t.join()
+    if "error" in box:
+        raise box["error"]
+    return box["value"]
```

## New tests
```python
"""Regression (2026-11-14, user hit 'Cannot run the event loop while another loop is running' in Colab Part A):
Jupyter/Colab cells execute while an asyncio loop is running in the main thread. run_virtual must still work there."""

import asyncio

import pytest

from eem_acc import scenarios as S
from eem_acc.env import run_virtual
from eem_acc.sim import run_scenario


def _inside_running_loop(fn):
    async def host():
        return fn()          # synchronous call made while a loop is running, exactly like a notebook cell
    return asyncio.run(host())


def test_run_virtual_works_when_a_loop_is_already_running():
    async def sleepy():
        await asyncio.sleep(3600)
        return asyncio.get_running_loop().time()
    assert _inside_running_loop(lambda: run_virtual(sleepy())) == pytest.approx(3600.0)


def test_scenario_result_identical_inside_and_outside_a_running_loop():
    sc = S.build("CAP-drop", 16, S.acc_method(**S.ACC_START), 3)
    assert _inside_running_loop(lambda: run_scenario(sc)) == run_scenario(sc)


def test_errors_propagate_from_the_worker_thread():
    async def boom():
        raise ValueError("inside the simulation")
    with pytest.raises(ValueError, match="inside the simulation"):
        _inside_running_loop(lambda: run_virtual(boom()))


def test_demo_figure_works_inside_running_loop():
    import matplotlib
    matplotlib.use("Agg")
    from eem_acc.viz import demo_figure
    fig = _inside_running_loop(lambda: demo_figure("CAP-drop", 16, seed=0, fixed_w=16))
    assert fig is not None
```
