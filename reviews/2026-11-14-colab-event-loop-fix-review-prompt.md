# Review request: "Cannot run the event loop while another loop is running" in Colab (2026-11-14)

You are an independent reviewer. Be concrete.

## Context
https://github.com/notanut/eem-acc-skripsi runs RQ2 as a deterministic virtual-time asyncio simulation
(`src/eem_acc/env/vtime.py`: a SelectorEventLoop whose clock jumps to the next timer; only asyncio.sleep /
loop.time and seeded numpy RNGs are used inside). In Google Colab (Python 3.13), a notebook cell that calls the
simulator synchronously failed because a loop is already running in the kernel's main thread. Fix: run_virtual
detects a running loop and then runs the simulation on a fresh virtual loop in a worker thread, waits for it, and
re-raises its exception; if the wait is interrupted (cell stopped) the simulation is cancelled. Tests run the simulator from inside `asyncio.run(...)` (reproducing the notebook
condition): they fail on the old code and pass now, and check identical results inside vs outside a running loop.

## Checklist
- Determinism and correctness of running the virtual loop off the main thread (signal handling, thread-local state,
  numpy/matplotlib thread safety, asyncio policies).
- Can blocking the notebook's main thread with join() deadlock or freeze anything? Behaviour on cell interrupt.
- Would nest_asyncio or making the API async be a better fix? Trade-offs.
- Are the tests meaningful?

## Output
Findings by severity (location, explanation, fix), then a verdict.

## Diff (against main, final state incl. working tree)
```diff
diff --git a/src/eem_acc/env/vtime.py b/src/eem_acc/env/vtime.py
index 90123ae..a64209e 100644
--- a/src/eem_acc/env/vtime.py
+++ b/src/eem_acc/env/vtime.py
@@ -12,7 +12,8 @@ real I/O (sockets, subprocesses) inside a virtual loop.
 from __future__ import annotations
 
 import asyncio
-from typing import Any, Awaitable, TypeVar
+import threading
+from typing import Any, Coroutine, TypeVar
 
 T = TypeVar("T")
 
@@ -44,10 +45,57 @@ class VirtualClockEventLoop(asyncio.SelectorEventLoop):
         return self._vtime
 
 
-def run_virtual(coro: Awaitable[T]) -> T:
-    """Run `coro` to completion on a fresh virtual-time loop."""
+def _run_here(coro: Coroutine[Any, Any, T], box: dict[str, Any] | None = None) -> T:
     loop = VirtualClockEventLoop()
     try:
-        return loop.run_until_complete(coro)
+        task = loop.create_task(coro)
+        if box is not None:
+            box["loop"], box["task"] = loop, task
+            box["started"].set()
+        return loop.run_until_complete(task)
     finally:
         loop.close()
+
+
+def _join(t: threading.Thread) -> None:
+    t.join()
+
+
+def run_virtual(coro: Coroutine[Any, Any, T]) -> T:
+    """Run the coroutine `coro` to completion on a fresh virtual-time loop.
+
+    If the calling thread already runs an event loop (Jupyter/Colab cells do), a second loop cannot be started in
+    that thread, so the simulation runs on a fresh loop in a short-lived worker thread and this call waits for it.
+    The simulation is self-contained (virtual clock, seeded RNGs), so the result is identical either way.
+    Requirements: pass a coroutine (not a Future/Task bound to another loop), and the simulation must never need the
+    caller's loop (it would deadlock while this call waits). If the wait is interrupted (e.g. the user stops the
+    cell), the simulation is cancelled instead of running on in the background."""
+    try:
+        asyncio.get_running_loop()
+    except RuntimeError:
+        return _run_here(coro)
+    box: dict[str, Any] = {"started": threading.Event()}
+
+    def work() -> None:
+        try:
+            box["value"] = _run_here(coro, box)
+        except BaseException as e:      # re-raised in the caller's thread
+            box["error"] = e
+
+    t = threading.Thread(target=work, name="run_virtual", daemon=True)
+    t.start()
+    try:
+        _join(t)
+    except BaseException:
+        box["started"].wait(1.0)
+        loop, task = box.get("loop"), box.get("task")
+        if loop is not None and not loop.is_closed():
+            try:
+                loop.call_soon_threadsafe(task.cancel)
+            except RuntimeError:        # loop closed in the meantime: nothing left to cancel
+                pass
+        t.join(5.0)
+        raise
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


def test_interrupting_the_wait_cancels_the_simulation(monkeypatch):
    """User stops the cell while a long simulation runs: the worker must be cancelled, not run on in the background."""
    import threading
    import time
    from eem_acc.env import vtime

    progress = {"steps": 0}

    async def endless():
        while True:
            await asyncio.sleep(1.0)
            progress["steps"] += 1
            time.sleep(0.001)              # real time passes, so the worker is still busy when we interrupt

    def interrupted_join(t):
        time.sleep(0.2)
        raise KeyboardInterrupt

    monkeypatch.setattr(vtime, "_join", interrupted_join)
    with pytest.raises(KeyboardInterrupt):
        _inside_running_loop(lambda: run_virtual(endless()))
    workers = [t for t in threading.enumerate() if t.name == "run_virtual"]
    assert not workers, "simulation thread still running after the interrupt"
    steps = progress["steps"]
    time.sleep(0.1)
    assert progress["steps"] == steps and steps > 0
```
