"""S3.0 feasibility check for Stage 3 (design `docs/stage3_design.md` r4, section 10).

Runs on a Colab T4 from `notebooks/03_s30_feasibility.ipynb`. Everything here except the `generate` callables is pure
Python and CPU-testable. It measures, for the local LLM (Qwen2.5-7B-Instruct Q4 GGUF via llama-cpp-python):

* the timing model t_llm = a + b_in * n_in + b_out * n_out (least squares), with and without the JSON-envelope grammar;
* JSON-envelope validity on sample agent prompts;
* determinism of repeated greedy generations (informs the "replay reproducibility" claim);
* c_step and behaviour on a toy 3-operation task with one drifted operation (a smoke test of the agent format, NOT a
  result: the real environment is built in S3.1).
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

# ----------------------------------------------------------------------------------------------------------------------
# Action envelope (design section 6, S6): grammar for the envelope only, `args` free.
THOUGHT_MAX_CHARS = 300  # ~60 Qwen tokens
ENVELOPE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "thought": {"type": "string", "maxLength": THOUGHT_MAX_CHARS},
        "action": {"type": "string"},
        "args": {"type": "object"},
    },
    "required": ["thought", "action", "args"],
}


def parse_action(text: str) -> dict | None:
    """Return the action dict if `text` is exactly one valid envelope, else None (a format error)."""
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict) or set(obj) != {"thought", "action", "args"}:
        return None
    if not isinstance(obj["thought"], str) or not isinstance(obj["action"], str) or not isinstance(obj["args"], dict):
        return None
    if not obj["action"].strip():
        return None
    return obj


# ----------------------------------------------------------------------------------------------------------------------
# Toy environment: 3 operations + docs tool + finish. One operation (create_order) drifts: `currency` was renamed to
# `currency_code` and the old key is rejected (unknown keys are rejected, design section 4).
DOCS_OLD = {
    "find_customer": {"description": "Find a customer by e-mail.", "args": {"email": "string, required"},
                      "returns": "{customer_id}"},
    "list_cart": {"description": "List the items in a customer's cart.",
                  "args": {"customer_id": "string (identifier), required"}, "returns": "{items: [{sku, qty}]}"},
    "create_order": {"description": "Create an order from items.",
                     "args": {"customer_id": "string (identifier), required",
                              "items": "array of {sku: string, qty: integer}, required",
                              "currency": "enum [IDR, USD], required"},
                     "returns": "{order_id}"},
}
DOCS_CURRENT = json.loads(json.dumps(DOCS_OLD))
DOCS_CURRENT["create_order"]["args"].pop("currency")
DOCS_CURRENT["create_order"]["args"]["currency_code"] = "enum [IDR, USD], required"

SYSTEM_RULES = """You are an agent that completes tasks by calling API operations.
Reply with exactly one JSON object and nothing else: {"thought": "<short reasoning>", "action": "<operation>", "args": {...}}.
Use one operation per reply. Operations:
{docs}
- get_api_docs(operation): returns the current documentation of an operation. Use it if a call fails and the error does not say how to fix it.
- finish(answer): call when the task is complete.
After each call you receive the API response. The documentation above may be out of date."""


def _docs_text(docs: dict) -> str:
    return "\n".join(f"- {op}: {d['description']} args: {json.dumps(d['args'])} returns: {d['returns']}"
                     for op, d in docs.items())


@dataclass
class ToyInstance:
    email: str
    customer_id: str
    items: list[dict]
    currency: str
    error_style: str  # "informative" | "terse"

    @property
    def task(self) -> str:
        return (f"Place an order in {self.currency} for everything in the cart of the customer with e-mail "
                f"{self.email}, then finish with the order id.")


def make_instances(n: int, seed: int = 0) -> list[ToyInstance]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        items = [{"sku": f"SKU-{rng.randint(100, 999)}", "qty": rng.randint(1, 3)} for _ in range(rng.randint(1, 3))]
        out.append(ToyInstance(email=f"user{rng.randint(1000, 9999)}@example.com", customer_id=f"C-{rng.randint(100, 999)}",
                               items=items, currency=rng.choice(["IDR", "USD"]),
                               error_style="informative" if i % 2 == 0 else "terse"))
    return out


class ToyShop:
    """Executes operations for one instance and checks the final state."""

    def __init__(self, inst: ToyInstance):
        self.inst = inst
        self.orders: list[dict] = []

    def call(self, op: str, args: dict) -> tuple[int, Any]:
        i = self.inst
        if op == "get_api_docs":
            name = args.get("operation")
            if isinstance(name, str) and name in DOCS_CURRENT:
                return 200, {name: DOCS_CURRENT[name]}
            return 404, {"detail": "unknown operation"}
        if op == "find_customer":
            if set(args) != {"email"}:
                return 422, {"detail": "expected exactly: email"}
            if args["email"] != i.email:
                return 404, {"detail": "customer not found"}
            return 200, {"customer_id": i.customer_id}
        if op == "list_cart":
            if set(args) != {"customer_id"}:
                return 422, {"detail": "expected exactly: customer_id"}
            if args["customer_id"] != i.customer_id:
                return 404, {"detail": "customer not found"}
            return 200, {"items": i.items}
        if op == "create_order":
            expected = {"customer_id", "items", "currency_code"}
            if set(args) != expected:
                if i.error_style == "terse":
                    return 400, "Bad Request"
                missing, extra = sorted(expected - set(args)), sorted(set(args) - expected)
                return 422, {"detail": [{"loc": ["body", k], "msg": "field required"} for k in missing]
                             + [{"loc": ["body", k], "msg": "extra fields not permitted"} for k in extra]}
            if args["customer_id"] != i.customer_id:
                return 404, {"detail": "customer not found"}
            if args["currency_code"] not in ("IDR", "USD"):
                return 422, {"detail": "currency_code must be one of IDR, USD"}
            oid = f"O-{len(self.orders) + 1:03d}"
            self.orders.append({"order_id": oid, **args})
            return 200, {"order_id": oid}
        if op in DOCS_OLD:
            return 422, {"detail": "invalid arguments"}
        return 404, "Not Found"  # route-level 404: unknown operation

    def success(self, answer: Any) -> bool:
        def norm(items):
            try:
                return sorted((str(x["sku"]), int(x["qty"])) for x in items)
            except (TypeError, KeyError, ValueError):
                return None
        ok = [o for o in self.orders if o["customer_id"] == self.inst.customer_id
              and o["currency_code"] == self.inst.currency and norm(o["items"]) == norm(self.inst.items)]
        return len(ok) == 1 and len(self.orders) == 1 and str(answer).find(ok[0]["order_id"]) >= 0


def build_messages(task: str, trajectory: list[tuple[str, str]], docs: dict = DOCS_OLD, memory: str = "") -> list[dict]:
    """Prompt layout of design section 6: static prefix -> memory -> task -> trajectory (assistant/user turns)."""
    system = SYSTEM_RULES.replace("{docs}", _docs_text(docs))
    msgs = [{"role": "system", "content": system}]
    user = (f"Lessons from earlier tasks:\n{memory}\n\n" if memory else "") + f"Task: {task}"
    msgs.append({"role": "user", "content": user})
    for action_json, observation in trajectory:
        msgs.append({"role": "assistant", "content": action_json})
        msgs.append({"role": "user", "content": f"Observation: {observation}"})
    return msgs


def observation_text(status: int, body: Any, limit: int = 400) -> str:
    text = f"HTTP {status} " + (body if isinstance(body, str) else json.dumps(body))
    return text[:limit]


def filler_docs(n_ops: int, seed: int = 0) -> dict:
    """Deterministic extra operations so the docs block has the size of the real 16-operation suite."""
    rng = random.Random(seed)
    nouns = ["shipment", "ticket", "invoice", "refund", "product", "warehouse", "coupon", "address", "review",
             "subscription", "payment", "return", "supplier"]
    docs = {}
    for i in range(n_ops):
        noun = nouns[i % len(nouns)]
        verb = ["get", "list", "update", "cancel"][i % 4]
        docs[f"{verb}_{noun}"] = {"description": f"{verb.capitalize()} a {noun} record and its related metadata.",
                                  "args": {f"{noun}_id": "string (identifier), required",
                                           "include_history": "boolean, optional",
                                           "status": f"enum [{', '.join(rng.sample(['open', 'closed', 'pending', 'hold', 'done'], 3))}], optional"},
                                  "returns": f"{{{noun}_id, status, updated_at}}"}
    return docs


def sample_prompts(n: int = 20, seed: int = 0, long_share: float = 0.5) -> list[list[dict]]:
    """Agent prompts for timing and validity checks. Trajectories of 0-9 turns, optional memory, and for `long_share`
    of them a 16-operation docs block, so n_in spans roughly the range of real Stage 3 prompts (a few hundred to a few
    thousand tokens)."""
    rng = random.Random(seed)
    insts = make_instances(n, seed)
    big_docs = {**DOCS_OLD, **filler_docs(13, seed)}
    prompts = []
    for k, inst in enumerate(insts):
        traj = []
        steps = [("find_customer", {"email": inst.email}, 200, {"customer_id": inst.customer_id}),
                 ("list_cart", {"customer_id": inst.customer_id}, 200, {"items": inst.items}),
                 ("create_order", {"customer_id": inst.customer_id, "items": inst.items, "currency": inst.currency},
                  400 if inst.error_style == "terse" else 422,
                  "Bad Request" if inst.error_style == "terse" else {"detail": [{"loc": ["body", "currency_code"],
                                                                                "msg": "field required"}]}),
                 ("get_api_docs", {"operation": "create_order"}, 200, {"create_order": DOCS_CURRENT["create_order"]})]
        long = k < round(n * long_share)
        n_turns = (k % (len(steps) + 1)) if not long else rng.randint(4, 9)
        seq = steps + [steps[2], steps[3], steps[2], steps[3], steps[2]]          # repeated failures for long ones
        for op, args, st, body in seq[:n_turns]:
            traj.append((json.dumps({"thought": "next step", "action": op, "args": args}), observation_text(st, body)))
        memory = ""
        if long:   # up to 8 notes of about the 120-token cap (design section 7)
            memory = "\n".join(
                f"- [{op} / {kind}] k_t=(API_Name={op}, Schema_Expected={{...{op}_field: string}}, Schema_Received="
                f"{{...old_field: number}}, Error_Type={kind}). Reflection: when calling {op}, send the field under its "
                f"new name and as a string; check the error detail before retrying, and do not resend the old key. "
                f"Last working change: old_field -> {op}_field (string)."
                for op, kind in [(rng.choice(list(big_docs)), rng.choice(["missing_field", "type_error", "enum_error"]))
                                 for _ in range(rng.randint(3, 8))])
        elif k % 3 == 2:
            memory = "\n".join(f"- [create_order / missing_field] Use currency_code instead of currency. "
                               f"Last working change: currency -> currency_code (string)." for _ in range(rng.randint(1, 5)))
        prompts.append(build_messages(inst.task, traj, docs=big_docs if long else DOCS_OLD, memory=memory))
    return prompts


# ----------------------------------------------------------------------------------------------------------------------
# Timing model fit (design section 3): t = a + b_in * n_in + b_out * n_out
def fit_timing(records: list[dict]) -> dict:
    """Least squares on records with keys seconds, n_in, n_out. Returns a, b_in, b_out, r2, n, and max residual."""
    import numpy as np
    if len(records) < 4:
        raise ValueError("need at least 4 timing records")
    X = np.array([[1.0, r["n_in"], r["n_out"]] for r in records])
    y = np.array([r["seconds"] for r in records])
    coef, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    out = {"a": float(coef[0]), "b_in": float(coef[1]), "b_out": float(coef[2]),
           "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"), "n": len(records),
           "max_abs_residual_s": float(np.abs(y - pred).max()), "rank": int(rank),
           "condition_number": float(np.linalg.cond(X)),
           "n_in_range": [int(X[:, 1].min()), int(X[:, 1].max())], "n_out_range": [int(X[:, 2].min()), int(X[:, 2].max())]}
    dof = len(records) - 3
    if rank == 3 and dof > 0:
        cov = (ss_res / dof) * np.linalg.inv(X.T @ X)
        out["se"] = {k: float(math.sqrt(max(cov[i, i], 0.0))) for i, k in enumerate(("a", "b_in", "b_out"))}
    else:
        out["se"] = None   # collinear inputs: coefficients are not identified (lstsq gives a minimum-norm answer)
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Toy episode: the agent loop of Algorithm 1 without EEM (C0), used only to measure c_step and check the format.
GenerateFn = Callable[[list[dict]], dict]  # returns {"text", "n_in", "n_out", "seconds"}


@dataclass
class EpisodeResult:
    success: bool
    steps: int
    api_calls: int
    format_errors: int
    docs_calls: int
    finished: bool
    llm_seconds: float
    llm_calls: int
    n_in: list[int] = field(default_factory=list)
    n_out: list[int] = field(default_factory=list)
    transcript: list[dict] = field(default_factory=list)


TOY_MAX_STEPS = 3 + 1 + 6   # minimal steps (3 calls + finish) + 6, design section 6


def run_toy_episode(generate: GenerateFn, inst: ToyInstance, max_steps: int = TOY_MAX_STEPS) -> EpisodeResult:
    """Steps are LLM generations (including `finish` and format errors)."""
    shop = ToyShop(inst)
    traj: list[tuple[str, str]] = []
    res = EpisodeResult(False, 0, 0, 0, 0, False, 0.0, 0)
    for _ in range(max_steps):
        out = generate(build_messages(inst.task, traj))
        res.steps += 1
        res.llm_calls += 1
        res.llm_seconds += out["seconds"]
        res.n_in.append(out["n_in"])
        res.n_out.append(out["n_out"])
        step = {"raw": out["text"][:2000], "seconds": out["seconds"], "n_in": out["n_in"], "n_out": out["n_out"],
                "finish_reason": out.get("finish_reason")}
        act = parse_action(out["text"])
        if act is None:
            res.format_errors += 1
            traj.append((out["text"][:400], "invalid action format"))
            res.transcript.append({**step, "obs": "invalid action format"})
            continue
        if act["action"] == "finish":
            res.finished = True
            res.success = shop.success(act["args"].get("answer", act["args"]))
            res.transcript.append({**step, "obs": "finish"})
            break
        status, body = shop.call(act["action"], act["args"])
        res.api_calls += 1
        res.docs_calls += act["action"] == "get_api_docs"
        obs = observation_text(status, body)
        traj.append((out["text"], obs))          # the model's raw text, so the next prompt extends this one exactly
        res.transcript.append({**step, "obs": obs})
    return res


def summarise_episodes(results: list[EpisodeResult]) -> dict:
    n = len(results)
    calls = sum(r.llm_calls for r in results)
    return {"episodes": n, "success_rate": sum(r.success for r in results) / n if n else math.nan,
            "mean_steps": sum(r.steps for r in results) / n if n else math.nan,
            "format_error_rate": sum(r.format_errors for r in results) / calls if calls else math.nan,
            "docs_calls": sum(r.docs_calls for r in results),
            "c_step_s": sum(r.llm_seconds for r in results) / calls if calls else math.nan,
            "mean_n_in": sum(sum(r.n_in) for r in results) / calls if calls else math.nan,
            "mean_n_out": sum(sum(r.n_out) for r in results) / calls if calls else math.nan}


def timed(fn: Callable[[], dict]) -> dict:
    t0 = time.perf_counter()
    out = fn()
    out["seconds"] = time.perf_counter() - t0
    return out


# ----------------------------------------------------------------------------------------------------------------------
# llama-cpp-python glue (the `llm` object is a llama_cpp.Llama on the T4; tests pass a fake with the same method).
def make_generate(llm: Any, grammar: bool, reset: bool = True, max_tokens: int = 160) -> GenerateFn:
    """Greedy chat completion. `reset=True` clears the KV cache first, so no prefix is reused (worst case)."""
    kwargs: dict[str, Any] = {"temperature": 0.0, "top_k": 1, "max_tokens": max_tokens, "seed": 0}
    if grammar:
        kwargs["response_format"] = {"type": "json_object", "schema": ENVELOPE_SCHEMA}

    def generate(messages: list[dict]) -> dict:
        if reset and hasattr(llm, "reset"):
            llm.reset()
        t0 = time.perf_counter()
        out = llm.create_chat_completion(messages=messages, **kwargs)
        sec = time.perf_counter() - t0
        usage = out.get("usage", {})
        return {"text": out["choices"][0]["message"]["content"] or "", "n_in": int(usage.get("prompt_tokens", 0)),
                "n_out": int(usage.get("completion_tokens", 0)), "seconds": sec,
                "finish_reason": out["choices"][0].get("finish_reason")}
    return generate


def bench_prompts(generate: GenerateFn, prompts: list[list[dict]]) -> list[dict]:
    recs = []
    for i, msgs in enumerate(prompts):
        out = generate(msgs)
        recs.append({"prompt": i, "seconds": out["seconds"], "n_in": out["n_in"], "n_out": out["n_out"],
                     "valid_envelope": parse_action(out["text"]) is not None, "finish_reason": out.get("finish_reason"),
                     "text": out["text"][:500]})
    return recs


def determinism_check(generate: GenerateFn, prompts: list[list[dict]], repeats: int = 2) -> dict:
    """Same prompt, fresh KV cache each time: are greedy outputs identical? (Expected yes; the informative check is
    `replay_compare`.)"""
    same = 0
    for msgs in prompts:
        texts = {generate(msgs)["text"] for _ in range(repeats)}
        same += len(texts) == 1
    return {"prompts": len(prompts), "identical": same}


def replay_compare(generate_fresh: GenerateFn, inst: ToyInstance, transcript: list[dict]) -> dict:
    """Re-generate each step of a recorded episode (run with KV-prefix reuse) from the same messages with a fresh KV
    cache, and compare the full raw texts. This is the case that matters for resume-by-replay (design section 9)."""
    traj: list[tuple[str, str]] = []
    identical, compared = 0, 0
    first_diff = None
    for i, step in enumerate(transcript):
        out = generate_fresh(build_messages(inst.task, traj))
        compared += 1
        if out["text"][:2000] == step["raw"]:
            identical += 1
        elif first_diff is None:
            first_diff = i
        traj.append((step["raw"] if step["obs"] != "invalid action format" else step["raw"][:400], step["obs"]))
        if step["obs"] == "finish":
            break
    return {"steps": compared, "identical": identical, "first_difference_at_step": first_diff}
