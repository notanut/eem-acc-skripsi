"""S3.0 feasibility helpers (CPU only; the LLM is replaced by scripted or fake generators)."""
import json

import pytest

from eem_acc.stage3 import feasibility as F


def scripted_policy(messages):
    """A deterministic 'agent' that reads the trajectory and acts like a competent C0 agent."""
    turns = [m for m in messages if m["role"] in ("assistant", "user")][1:]
    obs = [m["content"] for m in turns if m["role"] == "user"]
    task = messages[1]["content"]
    email = task.split("e-mail ")[1].split(",")[0]
    cur = "IDR" if " in IDR " in task else "USD"
    acts = [json.loads(m["content"]) for m in turns if m["role"] == "assistant"]
    last_obs = obs[-1] if obs else ""
    def emit(op, args):
        return {"text": json.dumps({"thought": "t", "action": op, "args": args}), "n_in": 100 + 10 * len(turns),
                "n_out": 20, "seconds": 0.01}
    if not acts:
        return emit("find_customer", {"email": email})
    cid = next((json.loads(o.split(" ", 3)[3])["customer_id"] for o in obs if '"customer_id"' in o and o.startswith("Observation: HTTP 200")), None)
    if acts[-1]["action"] == "find_customer":
        return emit("list_cart", {"customer_id": cid})
    items = next(json.loads(o.split(" ", 3)[3])["items"] for o in obs if '"items"' in o)
    if acts[-1]["action"] == "list_cart":
        return emit("create_order", {"customer_id": cid, "items": items, "currency": cur})
    if acts[-1]["action"] == "create_order" and "HTTP 200" in last_obs:
        oid = json.loads(last_obs.split(" ", 3)[3])["order_id"]
        return emit("finish", {"answer": oid})
    if acts[-1]["action"] == "create_order" and "currency_code" not in last_obs:
        return emit("get_api_docs", {"operation": "create_order"})
    return emit("create_order", {"customer_id": cid, "items": items, "currency_code": cur})


def test_parse_action_accepts_only_exact_envelopes():
    ok = '{"thought": "x", "action": "find_customer", "args": {"email": "a"}}'
    assert F.parse_action(ok)["action"] == "find_customer"
    for bad in ["", "not json", '{"thought": "x", "action": "a"}', '{"thought": "x", "action": "a", "args": []}',
                '{"thought": "x", "action": " ", "args": {}}', '[1]',
                '{"thought": "x", "action": "a", "args": {}, "extra": 1}']:
        assert F.parse_action(bad) is None, bad


def test_scripted_agent_solves_both_error_styles_and_terse_needs_docs():
    for inst in F.make_instances(6, seed=3):
        res = F.run_toy_episode(scripted_policy, inst)
        assert res.success and res.finished, inst
        assert res.docs_calls == (1 if inst.error_style == "terse" else 0)
        assert res.steps == (6 if inst.error_style == "terse" else 5)
        assert res.format_errors == 0


def test_old_schema_is_rejected_and_unknown_keys_too():
    inst = F.make_instances(1)[0]
    shop = F.ToyShop(inst)
    st, body = shop.call("create_order", {"customer_id": inst.customer_id, "items": inst.items, "currency": inst.currency})
    assert st == 422 and "currency_code" in json.dumps(body) and "extra fields" in json.dumps(body)
    st, _ = shop.call("create_order", {"customer_id": inst.customer_id, "items": inst.items,
                                       "currency_code": inst.currency, "note": "x"})
    assert st == 422 and not shop.orders
    assert shop.call("no_such_op", {}) == (404, "Not Found")          # route-level 404 differs from resource 404
    assert shop.call("find_customer", {"email": "nobody@x"})[1] == {"detail": "customer not found"}


def test_checker_rejects_wrong_currency_duplicate_orders_and_missing_answer():
    inst = F.make_instances(1)[0]
    shop = F.ToyShop(inst)
    other = "USD" if inst.currency == "IDR" else "IDR"
    shop.call("create_order", {"customer_id": inst.customer_id, "items": inst.items, "currency_code": other})
    assert not shop.success("O-001")
    shop2 = F.ToyShop(inst)
    args = {"customer_id": inst.customer_id, "items": inst.items, "currency_code": inst.currency}
    shop2.call("create_order", args)
    assert shop2.success("order O-001") and not shop2.success("done")
    shop2.call("create_order", args)
    assert not shop2.success("O-001")


def test_format_errors_are_steps_without_api_calls():
    calls = iter(["garbage"] * 9)
    res = F.run_toy_episode(lambda m: {"text": next(calls), "n_in": 10, "n_out": 1, "seconds": 0.0},
                            F.make_instances(1)[0])
    assert res.steps == 9 and res.format_errors == 9 and res.api_calls == 0 and not res.success


def test_prompt_layout_and_sample_prompts_grow():
    msgs = F.build_messages("T", [('{"a":1}', "obs")], memory="- note")
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[1]["content"].startswith("Lessons from earlier tasks:") and msgs[1]["content"].endswith("Task: T")
    assert "currency" in msgs[0]["content"] and "currency_code" not in msgs[0]["content"]   # docs are the OLD schema
    ps = F.sample_prompts(20)
    assert len(ps) == 20 and len({len(p) for p in ps}) >= 4
    assert F.sample_prompts(20) == ps                                                     # deterministic


def test_fit_timing_recovers_known_coefficients():
    recs = [{"n_in": n_in, "n_out": n_out, "seconds": 0.3 + 0.001 * n_in + 0.04 * n_out}
            for n_in in (200, 800, 1500, 3000) for n_out in (5, 40, 120)]
    fit = F.fit_timing(recs)
    assert fit["a"] == pytest.approx(0.3) and fit["b_in"] == pytest.approx(0.001) and fit["b_out"] == pytest.approx(0.04)
    assert fit["r2"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        F.fit_timing(recs[:3])


class FakeLlama:
    def __init__(self):
        self.resets = 0
        self.kwargs = None

    def reset(self):
        self.resets += 1

    def create_chat_completion(self, messages, **kw):
        self.kwargs = kw
        return {"choices": [{"message": {"content": '{"thought":"t","action":"finish","args":{}}'},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 50 * len(messages), "completion_tokens": 12}}


def test_make_generate_uses_grammar_greedy_and_resets_cache():
    llm = FakeLlama()
    gen = F.make_generate(llm, grammar=True)
    out = gen(F.build_messages("T", []))
    assert llm.resets == 1 and out["n_in"] == 100 and out["n_out"] == 12
    assert llm.kwargs["temperature"] == 0.0 and llm.kwargs["response_format"]["schema"] == F.ENVELOPE_SCHEMA
    F.make_generate(llm, grammar=False)(F.build_messages("T", []))
    assert "response_format" not in llm.kwargs
    recs = F.bench_prompts(F.make_generate(llm, grammar=False, reset=False), F.sample_prompts(4))
    assert len(recs) == 4 and all(r["valid_envelope"] for r in recs) and llm.resets == 2
    assert F.determinism_check(gen, F.sample_prompts(3)) == {"prompts": 3, "identical": 3}


def test_summarise_episodes():
    res = [F.run_toy_episode(scripted_policy, i) for i in F.make_instances(4, seed=1)]
    s = F.summarise_episodes(res)
    assert s["episodes"] == 4 and s["success_rate"] == 1.0 and s["format_error_rate"] == 0.0 and s["c_step_s"] > 0
