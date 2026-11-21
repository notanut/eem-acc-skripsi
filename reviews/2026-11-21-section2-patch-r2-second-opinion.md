# Second opinion on Section II patch r2 (external LLM, brought by the user 2026-11-21)

_Archived summary of the user's doc "Section II Patch Review" (claude.ai artifact Doi5TR8W1AWz6JLZmDWVgM). Wording
condensed; every finding kept. Applied in `docs/section2_patch.md` r3 (one deviation noted there)._

**Verdict:** revise R1, R4, R5, R6 before inserting; descriptions mostly accurate, problems are positioning.

- **H1:** the EEM claim is not narrowed against ToolEVO (tool-update after a successful new invocation adds the new
  usage to the prompt's tool descriptions: a precedent for the outcome update; ToolQA-D's server-side changes overlap
  drift types 1-3 and 5) and MemToolAgent. Proposed EEM claim: training-free, per-error unit merged on (API_Name,
  Error_Type), deterministic update from the argument change that worked, shared by concurrent agents, benefit only as
  far as RQ1 shows. Section IV should say drift types 1-3 and 5 follow ToolQA-D.
- **H2:** the remaining ACC claim (kappa) is a design choice (the SRE margin), not a contribution; AIMD against a
  delay target is standard (Breakwater). Proposed: ACC applies an established mechanism to the agents' tool-API
  traffic, which HiveMind/AgentRM do not regulate; contribution = RQ2 comparison + RQ3 joint evaluation.
- **M1:** HiveMind and AgentRM regulate calls to the LLM provider, not tool calls (HiveMind Sec. 7.2). New R4 and R5
  wording proposed.
- **M2:** MemToolAgent entries are per past user query; retrieval cuts at the first sudden similarity drop. Call C1 "a
  reflection-retrieval baseline", not MemToolAgent-like.
- **M3:** SchemaShield-Lite shows the changed interface to the model; our docs are stale and the server enforces the
  new schema.
- **M4:** cite ReliabilityBench's finding that ReAct is more robust than Reflexion under stress.
- **M5:** add Breakwater (OSDI 2020) and G-Memory (NeurIPS 2025).
- **Low:** ATCC wording ("RL policy deciding when to switch"); R2 vagueness; en dash; DOI for ContDa; proceedings URL
  for ToolEVO; "et al." for more than six authors.
