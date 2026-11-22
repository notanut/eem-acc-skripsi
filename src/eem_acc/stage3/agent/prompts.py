"""Prompt texts and builders (S3.2a spec sections 3, 4, 6.2). Versioned: any change bumps PROMPT_VERSION."""
from __future__ import annotations

import json
import re
from typing import Any

PROMPT_VERSION = "p-v1"

DOCS_WORDING = {   # D94; the pilot keeps the variant with the highest C0 dev success
    "W1": "If a call fails because the API may have changed, call get_api_docs with the operation name to see its "
          "current schema.",
    "W2": "The documentation above may be out of date; get_api_docs(operation) returns the current schema of an "
          "operation.",
    "W3": "Before retrying a failed call, check get_api_docs(operation) for the current parameters.",
    "W4": "get_api_docs(operation) returns the current schema of an operation.",
}

RULES = """You are an agent that completes a task by calling the operations of a shop API, one call per reply.
Reply with exactly one JSON object and nothing else:
{"thought": "<one or two short sentences>", "action": "<operation>", "args": {<arguments>}}
Use the ids returned by earlier calls. After each call you receive the API response as an observation.
When the task is complete, reply with {"thought": "...", "action": "finish", "args": {"summary": "<what you did>"}}."""

EXTRA_TOOLS = """get_api_docs - Return the current documentation of an operation.
  operation         required  string
  returns: operation, docs

finish - End the task.
  summary           required  string"""

MEMORY_HEADER = "Notes from earlier API errors:"
REFLECT_INSTR = ("A call to an API operation failed. Write one imperative instruction of at most 40 words that tells "
                 "how to fix the call next time. Reply with the instruction only.")
A2_TEMPLATE = "Fix the arguments of {api} as stated above."
TRANSIENT = "Observation: the API is temporarily unavailable (the gateway retried and gave up). You may try again."
FORMAT_ERROR = "Observation: invalid action format"
SUCCESS_CAP = 1500
ERROR_CAP = 400
FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n(.*?)\n?```\s*$", re.DOTALL)


def prompt_hash() -> str:
    import hashlib
    blob = json.dumps([RULES, EXTRA_TOOLS, DOCS_WORDING, MEMORY_HEADER, REFLECT_INSTR, A2_TEMPLATE, TRANSIENT,
                       FORMAT_ERROR, SUCCESS_CAP, ERROR_CAP], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def system_prompt(docs_text: str, wording: str) -> str:
    return f"{RULES}\n\nOperations:\n\n{docs_text}\n\n{EXTRA_TOOLS}\n\n{DOCS_WORDING[wording]}"


def user_prompt(instruction: str, note_lines: list[str]) -> str:
    mem = (MEMORY_HEADER + "\n" + "\n".join(note_lines) + "\n\n") if note_lines else ""
    return f"{mem}Task: {instruction}"


def build_messages(system: str, instruction: str, note_lines: list[str],
                   trajectory: list[tuple[str, str]]) -> list[dict]:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user_prompt(instruction, note_lines)}]
    for assistant, observation in trajectory:
        msgs.append({"role": "assistant", "content": assistant})
        msgs.append({"role": "user", "content": observation})
    return msgs


def strip_fence(text: str) -> tuple[str, bool]:
    m = FENCE_RE.match(text)
    return (m.group(1).strip(), True) if m else (text.strip(), False)


def body_text(body: Any) -> str:
    return body if isinstance(body, str) else json.dumps(body)


def success_observation(op: str, body: Any) -> tuple[str, bool]:
    """Returns (text, truncated). get_api_docs bodies are shown verbatim (multi-line)."""
    if op == "get_api_docs" and isinstance(body, dict) and "docs" in body:
        return "Observation: 200\n" + body["docs"], False
    t = json.dumps(body, separators=(",", ":"))
    return "Observation: 200 " + t[:SUCCESS_CAP], len(t) > SUCCESS_CAP


def error_observation(status: int, body: Any) -> str:
    return f"Observation: {status} " + body_text(body)[:ERROR_CAP]


def reflection_messages(instruction: str, action: str, args: Any, error_text: str) -> list[dict]:
    """error_text = the uncapped k_t rendering (EEM arms) or the raw status and body (C1, F)."""
    content = (f"Task: {instruction}\nFailed call: {action} {json.dumps(args)}\nError: {error_text}\n\n"
               f"{REFLECT_INSTR}")
    return [{"role": "user", "content": content}]


def state_text(instruction: str, last_action: str | None, last_arg_keys: list[str], last_obs: str) -> str:
    """s_t for the retrieval query (D98)."""
    act = f"{last_action}({', '.join(last_arg_keys)})" if last_action else "none"
    return f"{instruction}\nLast action: {act}\nLast observation: {last_obs[:200]}"


def tools_desc() -> str:
    from ..apisuite.ops import OP_NAMES
    return " ".join(list(OP_NAMES) + ["get_api_docs", "finish"])
