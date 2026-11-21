# Review request: Related Works patch (Section II, revision 4) after a novelty check, 2026-11-21

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

## The patch (docs/section2_patch.md, revision 4)

# Section II (Related Works) patch, revision 4, 2026-11-21

_Draft for the user's review. It follows the novelty check `docs/novelty_check_2026-11-21.md`. The user decided
(2026-11-21): no extra comparison arm; ToolEVO, ContDa and the other close work go into Related Work. Each change
gives the exact proposal text (Old) and the replacement (New)._

_History:_
- _r2 applied an internal review of r1._
- _r3 applies the user's second opinion on r2 ("Section II Patch Review"): the EEM claim narrowed against ToolEVO and
  MemToolAgent; the ACC claim reduced to applying an established mechanism to tool-API traffic plus RQ2/RQ3, with
  kappa as a design choice; HiveMind and AgentRM described as regulating calls to LLM providers (verified: HiveMind
  Sec. 7.2, "HiveMind manages API access but does not coordinate agents' filesystem operations or tool calls");
  MemToolAgent and SchemaShield-Lite described more exactly; ReliabilityBench's finding on Reflexion; Breakwater
  and G-Memory added; IEEE style._
- _Changes that differ from that review: (1) drift types 1-3 and 5 are said to overlap with ToolQA-D's modifications,
  not to "follow" them; they were fixed earlier (D5) without reference to ToolQA-D. (2) MemToolAgent is said to
  choose how many entries to retrieve "from the similarity distribution", not "at the first sudden drop": it uses the
  first or the second drop depending on the benchmark._
- _r4 applies an internal review of r3: R1 split into two bullets; G-Memory described more exactly; ReliabilityBench
  stated once; the client-side adaptive limiter [17] added to the ACC claim; Breakwater's below-SLO delay target
  cited for kappa; reference style aligned with the proposal's list; ToolEVO's evaluation setting noted._

Facts about cited papers were checked against the papers on 2026-11-21. New references take the next free numbers;
the final numbering is done when the paper is assembled, together with the references added in Section III
([RFC5681], [HF96], [SRE-SLO]).

## R1. Section II-B, two new bullets after "Agentic Memory Evolution"

**New (insert two bullets after the A-MEM bullet):**

> - Tool and Multi-Agent Memory: MemToolAgent [31] stores, for each past user query, the tool calls, binary
>   environment or user feedback and, for failed executions, an LLM-written reflection; it retrieves entries by
>   query-embedding similarity and chooses how many from the similarity distribution, without fine-tuning. G-Memory
>   [36] maintains a three-tier graph memory (insight, query and interaction graphs) of a multi-agent system's past
>   collaborations and retrieves memories for each agent.
> - Evolving APIs and Interface Drift: ToolEVO [32] combines Monte Carlo Tree Search with a self-reflection module
>   and a tool-update module that, after a new tool has been invoked successfully, adds a summary of its usage to the
>   tool descriptions in the prompt; it fine-tunes the model on the resulting trajectories and introduces ToolQA-D, a
>   benchmark in which API names, parameters and invocation formats change on the server side. ContDa [33] lets agents
>   update tool documentation as toolsets evolve. SchemaShield-Lite [34] assumes the changed interface is shown to the
>   model and repairs invalid calls at inference time through schema validation, retries for missing tool names and
>   one-shot constrained repair. ReliabilityBench [35] evaluates ReAct and Reflexion agents under injected faults that
>   include rate limits, timeouts and schema drift, and finds that the simpler ReAct agents outperform Reflexion under
>   stress.

## R2. Section II-B, closing sentence

**Old:**
> While self-reflection and memory evolution allow agents to recover from parameter or logic errors, these cognitive
> mechanisms operate in isolation from low-level network and system dynamics.

**New:**
> These methods adapt what the agent knows about its tools, but none of them observes or regulates the load the
> agents place on the API.

## R3. Section II-C, "Agentic Concurrency & Transaction Management" bullet, first sentence

**Old:**
> To handle unpredictable agent transaction bursts, ATCC [18] introduced adaptive concurrency control specifically
> tailored for agentic transactions.

**New:**
> In databases, ATCC [18] adapts transaction concurrency control to the unforeseen transactions issued by LLM-based
> data agents, switching each transaction between optimistic and pessimistic execution, with a reinforcement-learning
> policy deciding when to switch.

_Why:_ ATCC controls the consistency of database transactions, not the load an agent puts on an API.

## R4. Section II-C, "OS-Inspired Agent Schedulers" bullet, second sentence

**Old:**
> HiveMind [20] and She [21] further proposed OS-inspired resource schedulers for concurrent multi-agent workloads,
> while DynTaskMAS [12] implemented a dynamic task graph framework for asynchronous, parallel execution.

**New:**
> HiveMind [20] and AgentRM [21] further proposed OS-inspired resource managers for concurrent agents' calls to LLM
> provider APIs. HiveMind adjusts the number of concurrent provider calls with additive-increase/multiplicative-
> decrease (AIMD), decreasing on provider errors and when the mean latency over a sliding window exceeds a target, and
> adds a circuit breaker; it does not coordinate the agents' tool calls. AgentRM schedules agent tasks with a
> multi-level feedback queue and protects model-API rate limits with per-model token buckets, admission control and
> AIMD backoff. DynTaskMAS [12] implemented a dynamic task graph framework for asynchronous, parallel execution.

_Why:_ AIMD against a latency target is prior work, for agents' LLM calls [20] and in server overload control [37];
the paper must say so before presenting ACC. "She [21]" is the author's surname; the system is AgentRM.

## R5. Section II-C, "Adaptive Rate Control & Microservices" bullet, added sentence

**New (append to the bullet, after the DRACO sentence):**
> In server overload control, Breakwater [37] adjusts a pool of credits with an AIMD rule every round trip, using
> the server's queueing delay as the congestion signal.

## R6. Section II-C, new closing paragraph (after the "Agent Governance & Exchange Protocols" bullet)

**New:**
> HiveMind [20] and AgentRM [21] regulate the agents' calls to LLM providers, not the tool calls that reach backend
> APIs, and none of the schedulers [20]–[22] learns from the content of failed calls; the memory components of AIOS
> [22] and AgentRM [21] manage context, not execution lessons. Conversely, the memory-based methods of Section II-B do
> not regulate load. ReliabilityBench [35] comes closest by injecting schema drift and rate limits together, but it
> evaluates existing agents and proposes no mechanism for either problem. To our knowledge, no prior work combines a
> cross-task memory of API errors with adaptive concurrency control of the agents' tool calls and evaluates the two
> together under simultaneous interface drift and overload.

_Note:_ "to our knowledge" rests on the search of 2026-11-21; it is repeated before submission.

## R7. Contribution claims and related changes outside Section II (for later patches)

**Claims (Introduction contribution list; Section IV/V wording):**
- **EEM:** a training-free memory used at inference time. Its unit is an error on an operation, merged on
  (API_Name, Error_Type), not a past task as in MemToolAgent [31]. It is updated deterministically from the argument
  change that later worked, whereas ToolEVO [32] has the LLM summarise the new usage and fine-tunes on successful
  trajectories and was evaluated with the original tool descriptions in the prompt (the paper does not describe
  carrying updates across questions). EEM's store persists across tasks and is shared at run time among concurrent
  agents (G-Memory [36] also shares memory within a multi-agent system, but organises it by collaboration trajectory).
  Its benefit is claimed only as far as RQ1 shows it, against a reflection-retrieval baseline (C1; per-error
  reflections with a fixed k, not a reimplementation of MemToolAgent).
- **ACC:** applies an established mechanism, AIMD concurrency control against a latency target [20], [37], to the
  agents' tool-API traffic, which [20] and [21] do not regulate. It differs from client-side adaptive rate limiting
  of HTTP APIs [17], which adapts a token-bucket rate on 429 responses only, by limiting concurrency and using
  latency as well as overload signals. The claimed contribution is the pre-registered
  comparison against tuned static limits (RQ2) and the joint evaluation with EEM (RQ3). kappa is a design choice
  justified in Section III-C, not a contribution; Breakwater [37] likewise sets its delay target below the SLO
  (40% of it), which Section III-C should cite.
- **Joint system:** the combination and its evaluation under simultaneous drift and overload (RQ3).

**Elsewhere:**
- **Section III-C, first sentence** (goes into Section III patch r8): "extending adaptive transaction management
  paradigms [18], [28]" overstates the link to ATCC. Proposed: "applying AIMD concurrency control against a latency
  target, as used for agents' LLM calls [20] and in server overload control [37], to the agents' tool calls".
- **Section IV (setup):** state that drift types 1-3 and 5 overlap with the API modifications of ToolQA-D [32]
  (renamed APIs and parameters, new parameters, changed parameter formats), and that enum changes and silent errors
  are additions. Unlike the setting of SchemaShield-Lite [34], the agent's documentation stays stale while the server
  enforces the new schema.
- **Section IV/V (discussion):** relate the RQ1 result for reflection to ReliabilityBench's finding that Reflexion is
  outperformed by ReAct under stress [35]; EEM never reflects on 429, 503, 504 or timeouts (Section III-B).
- **Introduction:** "(ATCC) [18]" already noted in `docs/section3_revised.md`; the last sentence states a result
  ("achieves high task completion rates ...") and is rewritten after Chapter IV.
- **Not cited:** Assidiqi et al. (IEEE Access, 2026), a drift benchmark found in the novelty check. Its full title,
  volume and DOI could not be retrieved from here (IEEE Xplore did not load); ToolQA-D [32] and ReliabilityBench [35]
  cover the benchmark side. It can be added once the details are checked in IEEE Xplore.

## New references (in the style of the proposal's existing list; "et al." for more than six authors)

- [31] S. A. Er et al., "MemToolAgent: Leveraging Memory for Tool Using Agents Based on Environment and User
  Feedback," Jun. 2026, [Online]. Available: http://arxiv.org/abs/2606.07909
- [32] G. Chen et al., "Learning Evolving Tools for Large Language Models," in Proceedings of the International Conference on
  Learning Representations (ICLR), Apr. 2025. [Online]. Available:
  https://proceedings.iclr.cc/paper_files/paper/2025/hash/d8d1d325f822e07d16866780cc23deb2-Abstract-Conference.html
- [33] B. Wu, E. Meij, and E. Yilmaz, "Beyond Static Toolsets: Self-Evolving LLM Tool Agents via Continual
  Documentation Adaptation," in Findings of the Association for Computational Linguistics: ACL 2026, Jul. 2026,
  pp. 21519–21539. doi: 10.18653/v1/2026.findings-acl.1082.
- [34] A. Chattopadhyay, K. Halder, and P. S. Roy, "Robust Function Calling for LLM Agents Under Interface Drift,"
  Oct. 2026, [Online]. Available: http://arxiv.org/abs/2610.23342
- [35] A. Gupta, "ReliabilityBench: Evaluating LLM Agent Reliability Under Production-Like Stress Conditions,"
  Jan. 2026, [Online]. Available: http://arxiv.org/abs/2601.06112
- [36] G. Zhang, M. Fu, K. Wang, F. Wan, M. Yu, and S. Yan, "G-Memory: Tracing Hierarchical Memory for Multi-Agent
  Systems," in Advances in Neural Information Processing Systems 38 (NeurIPS), Dec. 2025.
- [37] I. Cho, A. Saeed, J. Fried, S. J. Park, M. Alizadeh, and A. Belay, "Overload Control for µs-scale RPCs with
  Breakwater," in Proceedings of the 14th USENIX Symposium on Operating Systems Design and Implementation (OSDI), Nov. 2020,
  pp. 299–314.

_Notes on references:_ [36] uses the author list and order of the NeurIPS proceedings page; arXiv:2506.07398 lists
"Guancheng Wan" and a different order. [17] (already in the proposal) may have a conference version (IEEE CCNC 2026,
reported by the internal review, not checked here); check IEEE Xplore and cite the published version if it exists.
