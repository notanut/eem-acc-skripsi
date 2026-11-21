# Novelty and overlap check, 2026-11-21

Purpose: check that the thesis (EEM + ACC, Stage 3 design r4) does not duplicate published work, and list what must
be cited. Method: web searches of arXiv, ACL Anthology, ICLR, IEEE and engineering sources (10 queries, standard and
extended), plus reading the abstracts/HTML of the closest papers. A search cannot prove that nothing similar exists.
Repeat it before submission.

## Closest work found, per component

| Our component | Closest published work | Overlap | Difference (what is ours) |
|---|---|---|---|
| EEM: reflection on API errors, stored and retrieved for later tasks | **MemToolAgent** (Er et al., AWS AI, arXiv:2606.07909, Jun 2026): reflections from failed tool calls + environment/user feedback, stored and retrieved by embedding similarity; WorkBench, NESTFUL, PEToolBench | High: same core idea (error -> LLM reflection -> memory -> retrieval) | No parsed error tuple k_t, no merge on (API, error type), no links, no outcome update, no shared store across concurrent agents, no schema drift, no concurrency/rate limits (checked in the paper's HTML) |
| | Reflexion, ExpeL, A-MEM (already cited); ToolReflection (ACL REALM 2025); structured reflection for tool calls (arXiv:2509.18847, trains the model) | Medium | Training-free; cross-task memory under drift |
| Adapting to API/schema drift | **ToolEVO** (Chen et al., ICLR 2025): MCTS + self-reflection + a tool-update module that rewrites API usage in the prompt; ToolQA-D benchmark with API changes | High in goal; the tool-update module is close to our outcome update | ToolEVO fine-tunes the model on MCTS data; ours is training-free, deterministic, per-note, shared |
| | **ContDa** (Wu, Meij, Yilmaz, Findings of ACL 2026): agents update tool documentation as toolsets change | Medium-high (updating usage knowledge) | Doc rewriting by exploration; no error memory, no concurrency |
| | **SchemaShield-Lite** (arXiv:2610.23342, Oct 2026): validation + one-shot repair under interface drift (DICE, BFCL) | Medium | Inference-time repair, no memory across tasks |
| Drift benchmarks | ToolQA-D; Assidiqi et al., IEEE Access 2026 (schema/toolset/policy drift); ProEvolve; DRIFT-BENCH; ReliabilityBench (rate limits AND schema changes as production faults, arXiv:2601.06112) | Medium: our drift operators (rename, type change, new field, moved endpoint) are standard | Our environment couples drift with a calibrated overload model and measures the interaction |
| ACC: AIMD window on 429/503/timeouts + latency | **HiveMind** (arXiv:2604.17111): AIMD backpressure + circuit breaker for concurrent LLM agents (LLM provider APIs); **AgentRM** (arXiv:2603.13110): token bucket + AIMD backoff + admission control; industry libraries (NVIDIA NeMo DataDesigner, many AIMD-on-429 limiters); TCP AIMD (RFC 5681) | High: AIMD for agent API calls is known practice, and HiveMind already decreases on errors and when average latency exceeds a target, updating per interval (checked in its PDF, Eq. 2) | Only the controller target kappa*L_target set below the SLO (so the tolerance band does not lie above it), the timeout-as-overload signal set, and the pre-registered evaluation against tuned static baselines (RQ2) |
| Concurrency control for agents | ATCC (arXiv:2603.13906) is **database** transaction CC (OCC/PCC switching by RL); CoAgent (arXiv:2606.15376) is write-conflict CC between agents | Low | Different problem (consistency of shared state, not API load) |
| Combination EEM + ACC | None found. AgentRM pairs scheduling with *context-length* management, not learning from errors (checked). No paper found that evaluates error-learning memory and adaptive concurrency together under simultaneous drift and overload | - | This is the main defensible contribution (RQ3) |

## Consequences
1. **Claims must be narrowed.** Not new: reflection memory from API errors; adapting to drift by updating usage
   knowledge; AIMD concurrency for agents, including a latency target (HiveMind). Defensible: (a) the joint system and its evaluation under simultaneous
   drift and overload, incl. the interaction (RQ3); (b) EEM's specific structure (k_t, merge, links, deterministic
   outcome update, store shared by concurrent agents) tested against a MemToolAgent-like baseline (our C1,
   reflection RAG); (c) ACC's headroom target below the SLO and its rigorous pre-registered comparison (RQ2), not the AIMD rule itself.
2. **Must cite (new):** MemToolAgent, ToolEVO/ToolQA-D, ContDa, SchemaShield-Lite, ReliabilityBench (Assidiqi et
   al. 2026 once its details are checked in IEEE Xplore); keep HiveMind and AgentRM (already in the proposal) and describe them accurately.
3. **Proposal text to fix:** ATCC is database transaction concurrency control; Section III's "extending adaptive
   transaction management paradigms [18], [28]" overstates the link.
4. **Plagiarism in the strict sense** (copied text or code): all text and code in the repo were written for this
   project; nothing was copied from these papers. The public repository will match the paper text in a similarity
   checker (self-match).
