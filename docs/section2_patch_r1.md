# Section II (Related Works) patch, revision 1, 2026-11-21

_Draft for the user's review. It follows the novelty check `docs/novelty_check_2026-11-21.md`. The user decided
(2026-11-21): no extra comparison arm; ToolEVO, ContDa and the other close work go into Related Work. Each change
below gives the exact proposal text (Old) and the replacement (New). Facts about cited papers were checked against
their abstracts or HTML on 2026-11-21. New references take the next free numbers; the final numbering is done when the
paper is assembled, together with the references added in Section III ([RFC5681], [HF96], [SRE-SLO])._

## R1. Section II-B, new bullet after "Agentic Memory Evolution"

**New (insert after the A-MEM bullet):**

> - Tool Memory and Evolving APIs: MemToolAgent [31] distils failed tool calls, together with environment and user
>   feedback, into reflection-based memory entries that are retrieved by embedding similarity, without fine-tuning.
>   Other work targets APIs that change after deployment. ToolEVO [32] combines Monte Carlo Tree Search with
>   self-reflection and a tool-update module that rewrites outdated API usage, fine-tunes the model on the resulting
>   trajectories, and introduces the ToolQA-D benchmark of tool variability. ContDa [33] lets agents update tool
>   documentation as toolsets evolve, and SchemaShield-Lite [34] repairs invalid calls under interface drift at
>   inference time through schema validation, retries for missing tool names and one-shot constrained repair. ReliabilityBench [35] evaluates ReAct and
>   Reflexion agents under injected faults that include rate limits, timeouts and schema drift.

## R2. Section II-B, closing sentence

**Old:**
> While self-reflection and memory evolution allow agents to recover from parameter or logic errors, these cognitive
> mechanisms operate in isolation from low-level network and system dynamics.

**New:**
> These methods adapt what the agent knows about its tools, but none of them regulates how many calls the agents
> issue: they operate in isolation from low-level network and system dynamics, and ReliabilityBench [35] measures
> rate-limit faults without controlling them.

## R3. Section II-C, "Agentic Concurrency & Transaction Management" bullet, first sentence

**Old:**
> To handle unpredictable agent transaction bursts, ATCC [18] introduced adaptive concurrency control specifically
> tailored for agentic transactions.

**New:**
> In databases, ATCC [18] adapts transaction concurrency control to the long, irregular transactions issued by data
> agents, switching each transaction between optimistic and pessimistic execution with a reinforcement-learning policy.

_Why:_ ATCC controls consistency of database transactions, not the load an agent puts on an API. The old wording
could be read as prior work on our problem.

## R4. Section II-C, "OS-Inspired Agent Schedulers" bullet, second sentence

**Old:**
> HiveMind [20] and She [21] further proposed OS-inspired resource schedulers for concurrent multi-agent workloads,
> while DynTaskMAS [12] implemented a dynamic task graph framework for asynchronous, parallel execution.

**New:**
> HiveMind [20] and AgentRM [21] further proposed OS-inspired resource managers for concurrent agent workloads; both
> combine admission control with additive-increase/multiplicative-decrease (AIMD) backoff when the LLM provider
> signals rate limits. DynTaskMAS [12] implemented a dynamic task graph framework for asynchronous, parallel
> execution.

_Why:_ AIMD for agent API calls is prior practice; the paper must say so before presenting ACC. "She [21]" is the
author's surname; the system is AgentRM.

## R5. Section II-C, new closing paragraph (after the "Agent Governance & Exchange Protocols" bullet)

**New:**
> These systems regulate load but do not learn from the content of failed calls; the memory component of AgentRM
> [21] manages context length, not execution lessons. Conversely, the memory-based methods of Section II-B do not
> regulate load. To our knowledge, no prior work evaluates error-driven experiential memory and adaptive concurrency
> control together, under simultaneous interface drift and overload; this is the gap addressed here.

_Note:_ "to our knowledge" rests on the search of 2026-11-21; it is repeated before submission.

## R6. Related changes outside Section II (for later patches; listed so they are not forgotten)

- **Section III-C, first sentence** (goes into Section III patch r8): "extending adaptive transaction management
  paradigms [18], [28]" overstates the link to ATCC. Proposed: "following adaptive concurrency limits from congestion
  control [RFC5681] and microservice concurrency adaptation [28]".
- **Introduction:** "(ATCC) [18]" already noted in `docs/section3_revised.md`; the contribution list should claim
  the joint system and its evaluation (RQ3), EEM's structure against a MemToolAgent-like baseline, and the
  SLO-targeted ACC, not reflection memory or AIMD as such. The last sentence of the Introduction states a result
  ("achieves high task completion rates ...") and is rewritten after Chapter IV.

## New references

- [31] S. A. Er, D. Ribeiro, Y. Virkar, S. Lakew, A. Kalyanpur, J. Gung, T. Delteil, and A. Gupta, "MemToolAgent:
  Leveraging memory for tool using agents based on environment and user feedback," arXiv:2606.07909, Jun. 2026.
- [32] G. Chen, Z. Zhang, X. Cong, F. Guo, Y. Wu, Y. Lin, W. Feng, and Y. Wang, "Learning evolving tools for large
  language models," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2025.
- [33] B. Wu, E. Meij, and E. Yilmaz, "Beyond static toolsets: Self-evolving LLM tool agents via continual
  documentation adaptation," in *Findings Assoc. Comput. Linguistics: ACL 2026*, San Diego, CA, USA, 2026,
  pp. 21519-21539.
- [34] A. Chattopadhyay, K. Halder, and P. S. Roy, "Robust function calling for LLM agents under interface drift,"
  arXiv:2610.23342, Oct. 2026.
- [35] A. Gupta, "ReliabilityBench: Evaluating LLM agent reliability under production-like stress conditions,"
  arXiv:2601.06112, Jan. 2026.
