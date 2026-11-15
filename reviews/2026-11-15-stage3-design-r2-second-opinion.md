# Stage 3 Design Review (r2)

Nov 15, 2026 · @Natasya Fernanda Josy

## Verdict

r2 is not ready to approve. I compared it with `docs/stage3_design.md` at commit `2fde691` (identical) and with the RQ2 numbers in HANDOFF.

- **Can start now:** S3.0 (the T4 feasibility notebook) and the core of S3.1 (the API layer on CPU). Neither depends on the open choices.
- **Decide before S3.2:** H1 and S5, because they change how EEM and the baselines are implemented.
- **Fix before the analysis is frozen:** H3, H4 and M1.

## High

### H1 (§6, III-B; §11): the reflection is written before the fix is known

- `LLM_reflect(s_t, a_t, k_t)` runs at the moment of failure. For terse errors, and for "endpoint moved" without a pointer, the note can only say "check the docs".
- Merge-on-key refreshes a note only when a new error arrives, so a note never records what actually worked.
- So the pre-registered "advantage larger for terse errors" goes against the mechanism. The gap between C2 and C1 would mostly come down to retrieval.
- **Decided: (a), the outcome update** (a Section III change, to go into patch r8 under S12). The options were:
  - **(a) Outcome update (my recommendation):** when a later call to the same API\_Name succeeds in the same task, update the note deterministically with the corrected argument diff (no LLM call). This fits "learning from execution experience".
  - **(b) Keep it as is,** and revise the terse-error expectation.
- Either way, add **note correctness** as a mechanism metric. The ground truth is known, so check automatically whether the stored note names the new field, type, enum value or path.

### H2 (§7, S5): the baselines don't isolate EEM's parts, and C1 is the weakest memory baseline you could pick

- 10–14 notes of at most 40 words is about 700–1,000 tokens, so the whole store fits in the prompt. An examiner will ask why retrieve at all, and what parsing, merge and links add over plain retrieval of reflections.
- A FIFO of k=3 over about 12 notes throws away relevant notes. The r2 enlargement of the store widens that gap, so E3/E4 favour EEM by construction.
- **Fix:**
  - **C1 := "reflection RAG":** the same reflections embedded and retrieved by top-k cosine with the same query, the same k and the same memory-token budget, with no k\_t, merge or links.
  - Move FIFO and **"all notes in prompt"** (the C2 store with no retrieval) to descriptive arms on 5 streams.
- Then C1 − C0 measures cross-task memory, and C2 − C1 measures parsing + merge + links.

### H3 (§8, E2/E4): "API calls per solved task" conditions on the outcome

- If C2 solves harder tasks that C0 fails, C2's calls per solved task go up. EEM can then look worse because it is better.
- Tasks are identical across conditions within a stream, so use that pairing.
- **Fix:**
  - E2/E4 = mean API calls (gateway attempts, including `get_api_docs`) on the tasks **solved by both conditions** in that stream.
  - Secondary: the ratio of sums, plus **LLM calls and prompt tokens per solved task**. EEM's own cost (reflection calls, longer prompts) is not API calls, and the current endpoints hide it.

### H4 (§8, §11): the tests don't match the stated claims, and power is unknown

- §11 expects "at least equal success", which is a non-inferiority claim. E1/E3 are two-sided superiority tests, and a non-significant result can't support "at least equal".
- **Granularity:** per-stream success moves in steps of 0.05 (20 tasks). Streams with zero difference carry no information in a sign-flip test. With 6 non-zero streams the smallest p is 2/64 = 0.031, which Holm over 4 endpoints turns into 0.125.
- **Non-inferiority at 0.05:** with 10 streams it is shown only if the SD of the per-stream differences is below about 0.086 (when the true difference is 0).
- **RQ3 interaction:** the contrast spans 4 cells, so its variance is roughly double. With 20 LLM tasks per seed per cell, non-inferiority at 0.05 will almost certainly be inconclusive.
- **Fix:**
  - Make E1/E3 one-sided non-inferiority tests with a margin fixed now. Keep E2/E4 as superiority tests.
  - Before freezing, run a **power simulation by resampling the pilot's per-stream differences**, and pre-register the number of streams (10 or 14) from it.
  - For RQ3, either justify a wider margin, or report the interaction descriptively (estimate + CI) and make "EEM effect under ACC on" the primary endpoint.
  - At n = 10, use t intervals or sign-flip-inversion intervals. Percentile bootstrap intervals undercover at that size (D4 currently says bootstrap).

## Medium

### M1 (§7 RQ3, S4/S7): ACC hardly touches the endpoint, and the non-inferiority test can pass for free

- MaxSteps counts steps, not time. So ACC changes LLM-task success only through exhausted retries.
- If the no-window arm floors both C0 and C2, the EEM effect without ACC is about 0, and the non-inferiority test passes trivially.
- The real interaction channel is that **overload hides drift**: overload checks run first, so a call whose retries are exhausted never reaches EEM. Count the drift errors observed and the notes created in each cell.
- **Fixes:**
  - **(i) Manipulation check from existing RQ2 dev data, now:** call success and attempts per call for W ≥ 32 in CAP-steady at M = 32. If call success is about 0.9 or higher, the ACC main effect on success will be near zero.
  - **(ii) Add virtual task completion time** as an RQ3 endpoint, since that is where ACC acts. ACC on may well be slower, because LLM calls queue FIFO behind 32 scripted agents at W ≈ 4. Report it whichever way it goes.
  - **(iii) Pre-register a floor rule:** if success in the off arm is below 0.2 for both C0 and C2, report the interaction as uninformative.
  - **(iv)** Count tasks still unfinished at T\_max as failures, and report how many there are.

### M2 (§5, S3): the environment favours memory by construction

- Half the endpoints drift from task 1, the docs stay stale, every drifted endpoint gets at least 3 exposures, and the drift never changes again.
- So EEM's main risk, a stale or wrong note, is never exercised.
- **Fix:**
  - Keep S3(a), and add a small descriptive **staleness probe**: one endpoint re-drifts at the stream midpoint, run with C0 and C2 on 5 streams.
  - Freeze and publish the suite, templates, error formats and drift pool before any EEM dev run.
  - Report the share of calls that hit a drifted endpoint, and state the drift density as a limitation.

### M3 (§4, §6): two author-set knobs control C0's strength

- **Docs-tool wording:** state now that it is chosen to maximise C0's dev success.
- **MaxSteps:** recovering from a drifted endpoint costs about 2 steps (the failed call, the docs lookup, the retry). So "min + 6" binds mostly for C0, and E1 partly measures efficiency under a step budget.
- Report, for the pilot and the test, what share of failures are budget exhaustion versus a wrong final state.

### M4 (§9): replay can diverge silently

- **(i) Embeddings:** they are recomputed on replay. GPU vs CPU or batch-size numerics can flip near-tied cosine ranks, which changes the prompt and causes a cache miss. Cache embeddings by text hash (or compute them on CPU), round the cosine and break ties by note id.
- **(ii) Strict replay:** a cache miss before the end of a unit's cache should be an error, not a fresh generation.
- **(iii) Token counts:** store n\_in and n\_out in the cache, so t\_llm is computed from cached counts.
- **(iv) Scope of the resume test:** the "interrupted = uninterrupted" test holds for the simulator and for replay, not for fresh generations. Say so.

### M5 (§8): dev selection

- 80 dev tasks per configuration with a 0.02 tie tolerance (1.6 tasks) is mostly noise. Use a tie tolerance of about 0.05, then break ties by API calls on jointly solved tasks, then by tokens.
- Give C1 the frozen C2's k and token budget.
- Report the dev sweep itself as the EEM part of RQ4, and drop A3 on test streams.

## Low

- **§4 references:** MirrorAPI weights are on Hugging Face ([model card](https://huggingface.co/stabletoolbench/MirrorAPI): MIT licence, a fine-tune of Qwen2.5-7B-Instruct). S2 still stands: StableToolBench's pass rate uses a GPT judge, and a second 8B model would double T4 time.
- **BFCL:** the [dataset card](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard) states Apache-2.0. "Checked by state, no judge model" still rests on a secondary source, so check the gorilla repo before option (b).
- **Retrieval recall:** retrieval runs before the action is chosen, and A\_t = A. Log whether the existing note for a drifted endpoint was in M\_rel when that endpoint was called. Otherwise a null result can't be told apart from "learned but not retrieved".
- **k\_t parser:** it is accurate by construction, since the same author writes the error formats and the parser. State this as a limitation; A1 covers it partly.
- **Budget:** cap the length of `thought` in the grammar, since decoding dominates c\_step on a T4. Measure the grammar's overhead in S3.0. Report the format-error rate per condition, since longer EEM prompts may raise it.
- **Mixed model:** add a position-in-stream × condition term for the learning curve.

## S1–S12

I agree with 10 of the 12 recommendations, mostly with conditions. S5 is a new option and S11 is modified.

| ID | My choice |
| --- | --- |
| S1 | (a) |
| S2 | (a), with the suite frozen before dev (M2); (b) only after RQ1 |
| S3 | (a) plus the descriptive staleness probe |
| S4 | (a), with t\_llm constants fixed a priori from a cited serving benchmark, plus completion time and a T\_max rule (M1) |
| S5 | **(d) new:** C1 = reflection RAG with the same k and budget; FIFO and all-notes as descriptive arms; drop C1w |
| S6 | (a); report format errors per condition |
| S7 | (a), after the manipulation check |
| S8 | (c). Pre-register the cut order now: A3 on test, then R to 3 streams, then C1w, then A1/A2 to 3 streams. Never cut C0/C1/C2 or the 2×2 |
| S9 | (a); (c) only if c\_step > 6 s |
| S10 | (a), plus the budget-exhaustion share (M3) |
| S11 | (a), modified: E1/E3 as non-inferiority, E2/E4 on jointly solved tasks. Keep the mechanism checks secondary, and add note correctness and retrieval recall |
| S12 | (a), including H1's edit if you choose option (a) |

## Sources

- [MirrorAPI model card](https://huggingface.co/stabletoolbench/MirrorAPI) (Hugging Face)
- [BFCL dataset card](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard) (Hugging Face)
- [eem-acc-skripsi repository](https://github.com/notanut/eem-acc-skripsi): `docs/stage3_design.md` and `HANDOFF.md` at commit `2fde691`
