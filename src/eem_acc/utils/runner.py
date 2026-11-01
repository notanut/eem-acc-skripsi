"""Resumable experiment runner.

Usage
-----
    run_resumable(
        unit_ids, run_unit, run_dir=paths.results / "exp1_eem",
        config=cfg, stateful=True,
        get_state=lambda: {"eem": eem.state_dict(), "acc": acc.state_dict()},
        set_state=lambda s: (eem.load_state_dict(s["eem"]), acc.load_state_dict(s["acc"])),
    )

Re-running the same call after a crash/disconnect continues where it stopped.

stateless (default): units are independent. Any unit with a saved result is skipped.
stateful: later units depend on state built by earlier ones (e.g. EEM memory grows
    across tasks). Only units covered by the latest valid checkpoint count as done;
    units finished after it are re-run, so state and results stay consistent.
    Use checkpoint_every=1 to lose nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable

from .atomic import clean_tmp_files
from .checkpoint import CheckpointManager
from .progress import ProgressLogger
from .results import ResultStore


def _canon(cfg: Any) -> str:
    return json.dumps(cfg, sort_keys=True, default=str)


def run_resumable(
    unit_ids: Iterable[str],
    run_unit: Callable[[str], dict[str, Any]],
    run_dir: str | Path,
    *,
    config: dict[str, Any] | None = None,
    stateful: bool = False,
    get_state: Callable[[], Any] | None = None,
    set_state: Callable[[Any], None] | None = None,
    checkpoint_every: int = 1,
    keep_last: int = 3,
    log_path: str | Path | None = None,
    progress_interval_s: float = 5.0,
    allow_config_change: bool = False,
) -> ResultStore:
    run_dir = Path(run_dir)
    ids = [str(u) for u in unit_ids]
    clean_tmp_files(run_dir)  # leftovers from an interrupted write
    store = ResultStore(run_dir / "units")
    ckpt = CheckpointManager(run_dir / "checkpoints", keep_last=keep_last)

    latest = ckpt.load_latest()
    ckpt_done: set[str] = set()
    counter = 0
    if latest is not None:
        if not allow_config_change and _canon(latest["config"]) != _canon(config):
            raise ValueError(
                f"Config differs from the checkpoint in {run_dir}. Use a new run_dir, "
                "or pass allow_config_change=True if the change is harmless."
            )
        ckpt_done = set(latest["state"]["done"])
        counter = latest["step"]
        if set_state is not None and latest["state"].get("user") is not None:
            set_state(latest["state"]["user"])
        print(f"[resume] checkpoint step {counter}: {len(ckpt_done)} units done")

    if stateful:
        done = ckpt_done & set(ids)
    else:
        done = (ckpt_done | {u for u in ids if store.exists(u)}) & set(ids)

    progress = ProgressLogger(len(ids), name=run_dir.name, already_done=len(done),
                              log_path=log_path, min_interval_s=progress_interval_s)

    def _checkpoint() -> None:
        ckpt.save(counter, {"done": sorted(done), "user": get_state() if get_state else None},
                  config=config)

    for u in ids:
        if u in done:
            continue
        result = run_unit(u)
        store.save(u, result)
        done.add(u)
        counter += 1
        progress.update()
        if counter % checkpoint_every == 0:
            _checkpoint()

    _checkpoint()  # final state (same step number overwrites atomically)
    return store
