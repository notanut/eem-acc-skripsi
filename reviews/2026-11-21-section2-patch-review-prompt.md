# Review request: Related Works patch (Section II) after a novelty check, 2026-11-21

You are a strict IEEE reviewer and thesis examiner. Be concrete; no praise.

## Context
Thesis paper "Learning from API Execution Experience: Self-Reflective Agentic AI with Adaptive Concurrency Control"
(Bina Nusantara University). EEM: non-transient API errors are parsed into a tuple (API_Name, Schema_Expected,
Schema_Received, Error_Type), an LLM writes a fix instruction, notes are merged on (API_Name, Error_Type), linked
Zettelkasten-style, updated deterministically with the argument change that later worked, and shared by concurrent
agents. ACC: an AIMD concurrency window driven by overload signals (429/503/504/timeouts) and interval p95 latency
against a controller target kappa*L_target below the SLO. A novelty check found close prior work for each component
(MemToolAgent, ToolEVO, ContDa, SchemaShield-Lite, ReliabilityBench; HiveMind and AgentRM already use AIMD for agent
API calls, HiveMind with a latency target) and no work combining both. The patch below adds this work to Related
Works, corrects descriptions of ATCC/HiveMind/AgentRM, and states the gap. The original proposal text is quoted as
"Old" in the patch.

## Checklist
1. Is every statement about a cited paper accurate? (Check the papers.)
2. Is the gap statement (R5) defensible and worded with the right caution? Is anything overclaimed or missing that
   an examiner would expect (other close work on agent memory for tool use, API drift, or adaptive concurrency)?
3. Do the narrowed contribution claims (R6) follow from the related work?
4. IEEE style and clarity of the new text and references.

## Output
Findings ranked by severity with a concrete rewording; then a verdict.

## The patch (docs/section2_patch.md)

# Section II (Related Works) patch, revision 2, 2026-11-21

_Draft for the user's review. It follows the novelty check `docs/novelty_check_2026-11-21.md`. The user decided
(2026-11-21): no extra comparison arm; ToolEVO, ContDa and the other close work go into Related Work. Each change
below gives the exact proposal text (Old) and the replacement (New). Facts about cited papers were checked against
their abstracts, HTML or PDF on 2026-11-21. r2 applies an internal review of r1 (HiveMind's AIMD also uses a latency
target; reference style; MemToolAgent, ToolEVO and ATCC wording; scope of the gap statement). New references take the next free numbers; the final numbering is done when the
paper is assembled, together with the references added in Section III ([RFC5681], [HF96], [SRE-SLO])._

## R1. Section II-B, new bullet after "Agentic Memory Evolution"

**New (insert after the A-MEM bullet):**

> - Tool Memory and Evolving APIs: MemToolAgent [31] stores tool-call trajectories with environment and user
>   feedback, adds an LLM-written reflection to the failed ones, and retrieves entries by embedding similarity,
>   without fine-tuning. Other work targets APIs that change after deployment. ToolEVO [32] combines Monte Carlo Tree
>   Search with a self-reflection module and a tool-update module that updates the tool descriptions in the prompt,
>   fine-tunes the model on the resulting trajectories, and introduces ToolQA-D, a benchmark of tool variability.
>   ContDa [33] lets agents update tool documentation as toolsets evolve, and SchemaShield-Lite [34] repairs invalid
>   calls under interface drift at inference time through schema validation, retries for missing tool names and
>   one-shot constrained repair. ReliabilityBench [35] evaluates ReAct and Reflexion agents under injected faults
>   that include rate limits, timeouts and schema drift.

## R2. Section II-B, closing sentence

**Old:**
> While self-reflection and memory evolution allow agents to recover from parameter or logic errors, these cognitive
> mechanisms operate in isolation from low-level network and system dynamics.

**New:**
> These methods adapt what the agent knows about its tools, but none of them regulates how many calls the agents
> issue: they operate in isolation from low-level network and system dynamics. ReliabilityBench [35] exposes agents
> to rate-limit faults but proposes no mechanism to regulate load.

## R3. Section II-C, "Agentic Concurrency & Transaction Management" bullet, first sentence

**Old:**
> To handle unpredictable agent transaction bursts, ATCC [18] introduced adaptive concurrency control specifically
> tailored for agentic transactions.

**New:**
> In databases, ATCC [18] adapts transaction concurrency control to the unforeseen transactions issued by LLM-based
> data agents, switching each transaction between optimistic and pessimistic execution with a reinforcement-learning
> policy.

_Why:_ ATCC controls consistency of database transactions, not the load an agent puts on an API. The old wording
could be read as prior work on our problem.

## R4. Section II-C, "OS-Inspired Agent Schedulers" bullet, second sentence

**Old:**
> HiveMind [20] and She [21] further proposed OS-inspired resource schedulers for concurrent multi-agent workloads,
> while DynTaskMAS [12] implemented a dynamic task graph framework for asynchronous, parallel execution.

**New:**
> HiveMind [20] and AgentRM [21] further proposed OS-inspired resource managers for concurrent agent workloads.
> HiveMind adjusts the number of concurrent LLM API calls with additive-increase/multiplicative-decrease (AIMD),
> decreasing on provider errors and when the average latency exceeds a target, and adds a circuit breaker; AgentRM
> combines rate-limit-aware admission control with AIMD backoff. DynTaskMAS [12] implemented a dynamic task graph
> framework for asynchronous, parallel execution.

_Why:_ AIMD for agent API calls, including a latency target, is prior work; the paper must say so before presenting
ACC. ACC's claim is narrowed accordingly (R6). "She [21]" is the author's surname; the system is AgentRM.

## R5. Section II-C, new closing paragraph (after the "Agent Governance & Exchange Protocols" bullet)

**New:**
> The agent schedulers above [20]-[22] regulate load but do not learn from the content of failed calls; the memory component of AgentRM
> [21] manages context length, not execution lessons. Conversely, the memory-based methods of Section II-B do not
> regulate load. To our knowledge, no prior work evaluates error-driven experiential memory and adaptive concurrency
> control together, under simultaneous interface drift and overload; this is the gap addressed here.

_Note:_ "to our knowledge" rests on the search of 2026-11-21; it is repeated before submission.

## R6. Related changes outside Section II (for later patches; listed so they are not forgotten)

- **Section III-C, first sentence** (goes into Section III patch r8): "extending adaptive transaction management
  paradigms [18], [28]" overstates the link to ATCC. Proposed: "following adaptive concurrency limits from congestion
  control [RFC5681] and microservice concurrency adaptation [28]".
- **Introduction:** "(ATCC) [18]" already noted in `docs/section3_revised.md`; the contribution list should claim
  the joint system and its evaluation (RQ3), and EEM's structure against a MemToolAgent-like baseline. For ACC, AIMD
  with a latency target is prior work [20]; the claim is limited to regulating to a target kappa*L_target below the
  SLO (so the tolerance band does not lie above it) and the pre-registered comparison against tuned static limits
  (RQ2). The last sentence of the Introduction states a result
  ("achieves high task completion rates ...") and is rewritten after Chapter IV.

- **Not cited:** Assidiqi et al. (IEEE Access, 2026), a drift benchmark found in the novelty check. Its full title,
  volume and DOI could not be retrieved from here (IEEE Xplore did not load); ToolQA-D [32] and ReliabilityBench [35]
  cover the benchmark side. It can be added once the details are checked in IEEE Xplore.

## New references (in the style of the proposal's existing list)

- [31] S. A. Er, D. Ribeiro, Y. Virkar, S. Lakew, A. Kalyanpur, J. Gung, T. Delteil, and A. Gupta, "MemToolAgent:
  Leveraging Memory for Tool Using Agents Based on Environment and User Feedback," Jun. 2026, [Online]. Available:
  http://arxiv.org/abs/2606.07909
- [32] G. Chen, Z. Zhang, X. Cong, F. Guo, Y. Wu, Y. Lin, W. Feng, and Y. Wang, "Learning Evolving Tools for Large
  Language Models," in Proc. International Conference on Learning Representations (ICLR), Singapore, Apr. 2025.
- [33] B. Wu, E. Meij, and E. Yilmaz, "Beyond Static Toolsets: Self-Evolving LLM Tool Agents via Continual
  Documentation Adaptation," in Findings of the Association for Computational Linguistics: ACL 2026, San Diego,
  CA, USA, Jul. 2026, pp. 21519-21539.
- [34] A. Chattopadhyay, K. Halder, and P. S. Roy, "Robust Function Calling for LLM Agents Under Interface Drift,"
  Oct. 2026, [Online]. Available: http://arxiv.org/abs/2610.23342
- [35] A. Gupta, "ReliabilityBench: Evaluating LLM Agent Reliability Under Production-Like Stress Conditions,"
  Jan. 2026, [Online]. Available: http://arxiv.org/abs/2601.06112
