"""S3.5 spec section 3: normalised per-unit data, pairing and completeness."""
from __future__ import annotations

from dataclasses import dataclass, field

PRIMARY_RQ1 = ("C0", "C1", "C2", "C3")
# payload parts kept for the secondary, descriptive and mechanism analyses (spec sections 6-8); caches are never read
RAW_KEYS = ("note_events", "final_notes", "steps", "llm_metrics", "infra", "J_rq3", "drift_errors", "notes_created",
            "success_rate", "t_end", "stamp", "llm_calls")


def _flat(v):
    """RQ3 C2-private stores one event/notes list per agent (a list of lists): flattened, each item tagged with its
    agent, so no later code mistakes a list for an event (code review L7)."""
    if isinstance(v, list) and v and all(isinstance(x, list) for x in v):
        return [{**e, "store_agent": a} for a, xs in enumerate(v) for e in xs]
    return v
PRIMARY_RQ3 = (("C0", "acc"), ("C2", "acc"), ("C0", "none"), ("C2", "none"))


@dataclass
class UnitData:
    rq: str
    kind: str
    seed: int
    condition: str
    window: str | None
    tag: str
    tasks: list[dict]
    status: str = "done"                    # done | failed (accepted as failed, D186)
    epoch: int | None = None
    params: dict = field(default_factory=dict)
    phase: str = ""
    n_tasks: int = 20
    raw: dict = field(default_factory=dict)     # payload parts the secondary/mechanism analyses read (RAW_KEYS)
    spec: dict | None = None                     # the rebuilt stream (drift schedule, styles), when checked

    @property
    def cell(self) -> str:
        return self.condition if self.rq == "rq1" else f"{self.condition}+{self.window}"

    def success_rate(self) -> float:
        return sum(bool(t["success"]) for t in self.tasks) / len(self.tasks) if self.tasks else float("nan")


def from_result(result: dict, check_spec: bool = True) -> UnitData:
    """A unit as `Platform.run_units` / `load` returns it -> UnitData.
    - status: `result.get("status", "done")`; a failed unit becomes a D186 placeholder (every task not solved), a
      not-done unit an empty unit with status "not_done" (dropped by `complete`) (code review M4);
    - the platform epoch from the payload's `stamp.llm` (code review M3);
    - check_spec: the unit's stream is rebuilt (`Unit.spec`, with its knobs) and must have the payload's
      `stamp.spec_sha256` (spec section 3, code review M5).
    Tasks unfinished at T_max are already `success = False` in the RQ3 payload; the flag is kept."""
    from ..llm.runner import Unit
    u = result["unit"]
    status = result.get("status", "done")
    window = u.get("window")
    if status == "failed":
        return failed_placeholder(u["rq"], u.get("kind", u["rq"]), int(u["seed"]), u["condition"], window,
                                  int(u.get("n_tasks") or 20), tag=u.get("tag", ""), phase=u.get("phase", ""))
    if status != "done":
        return UnitData(rq=u["rq"], kind=u.get("kind", u["rq"]), seed=int(u["seed"]), condition=u["condition"],
                        window=window, tag=u.get("tag", ""), tasks=[], status="not_done", phase=u.get("phase", ""))
    stamp = result.get("stamp", {})
    spec = None
    if check_spec:
        spec = Unit.from_json(u).spec()
        if stamp.get("spec_sha256") != spec["sha256"]:
            raise ValueError(f"{Unit.from_json(u).unit_id}: payload spec_sha256 {stamp.get('spec_sha256')} != "
                             f"rebuilt stream {spec['sha256']}")
    tasks = [{**t, "pos": t.get("task", i), "success": bool(t.get("success"))}
             for i, t in enumerate(result.get("tasks", []))]
    return UnitData(rq=u["rq"], kind=u.get("kind", u["rq"]), seed=int(u["seed"]), condition=u["condition"],
                    window=window, tag=u.get("tag", ""), tasks=tasks, status="done",
                    epoch=(stamp.get("llm") or {}).get("platform_epoch"),
                    params=dict(tuple(p) for p in u.get("params", ())), phase=u.get("phase", ""),
                    n_tasks=int(u.get("n_tasks") or 20), raw={k: (_flat(result[k]) if k in ("note_events", "final_notes") else result[k])
                         for k in RAW_KEYS if k in result},
                    spec=spec)


def failed_placeholder(rq: str, kind: str, seed: int, condition: str, window: str | None, n_tasks: int,
                       tag: str = "", phase: str = "") -> UnitData:
    """D186: a unit that failed for good. Complete-case analysis drops it; the sensitivity analysis counts every
    task as not solved (API calls unknown, so it never enters E2/E4)."""
    return UnitData(rq, kind, seed, condition, window, tag, [{"pos": i, "success": False, "api_calls": None}
                                                            for i in range(n_tasks)], status="failed",
                    phase=phase, n_tasks=n_tasks)


def index(units: list[UnitData], rq: str, kind: str, phase=None, n_tasks: int = 20, tag: str = ""
          ) -> dict[int, dict[str, UnitData]]:
    """Units of one rq/kind with one tag (default: untagged), optionally of one plan phase or a tuple of phases (code
    review L12), by seed and cell."""
    phases = None if phase is None else ((phase,) if isinstance(phase, str) else tuple(phase))
    out: dict[int, dict[str, UnitData]] = {}
    for u in units:
        if (u.rq == rq and u.kind == kind and (u.tag or "") == tag and (phases is None or u.phase in phases)
                and u.n_tasks == n_tasks):
            if u.cell in out.setdefault(u.seed, {}):
                raise ValueError(f"duplicate unit {rq} {kind} {u.seed} {u.cell}")
            out[u.seed][u.cell] = u
    return out


def complete(by_seed: dict[int, dict[str, UnitData]], cells: tuple[str, ...], include_failed: bool = False
             ) -> tuple[list[int], dict[int, dict[str, str]]]:
    """Seeds where every cell is done (or, for the D186 sensitivity, done or failed). Returns (kept, dropped), where
    dropped maps each dropped seed to the status of each cell that kept it out ('missing', 'failed', 'not_done')
    (spec section 3: reported by condition, code review M7)."""
    kept, dropped = [], {}
    ok_status = ("done", "failed") if include_failed else ("done",)
    for s in sorted(by_seed):
        bad = {c: ("missing" if c not in by_seed[s] else by_seed[s][c].status) for c in cells
               if c not in by_seed[s] or by_seed[s][c].status not in ok_status}
        if bad:
            dropped[s] = bad
        else:
            kept.append(s)
    return kept, dropped
