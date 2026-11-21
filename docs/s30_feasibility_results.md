# S3.0 feasibility results (Colab T4), 2026-11-21

Source: `docs/evidence/s30_results_2026-11-21.json` (the user's run of `notebooks/03_s30_feasibility.ipynb`; also on
Drive `Skripsi/stage3/s30/s30_results.json`). Setup: Tesla T4 15 GB, Python 3.13, llama-cpp-python 0.3.36 built for
sm_75 (wheel sha256 ab93c390...ef99), Qwen2.5-7B-Instruct Q4_K_M (2 GGUF parts, sha256 recorded), n_ctx 8192,
n_batch 512, greedy decoding. **Not a thesis result**: these numbers set the Stage 3 scale and the t_llm constants.

## Measurements

| Item | Value |
|---|---|
| Model load | 102 s; GPU memory 5.2 GB after load, 5.3 GB after the timing runs (of 15.4 GB) |
| Valid JSON envelope | 20/20 with the grammar, 20/20 without it (prompts of 278-2,627 tokens, incl. memory notes); the free and grammar-decoded texts were identical in 20/20, as expected by construction under greedy decoding (a grammar mask changes only tokens that would be invalid), so this adds no independent support; 0/20 invalid still allows an invalid rate up to about 14% (one-sided 95% bound) |
| Mean time per call, 20 prompts, fresh KV cache | 6.29 s with the grammar, 4.25 s without (mean 1,273 input / 70 output tokens) |
| Same, reusing the shared system-prompt prefix | 5.56 s (grammar) |
| Timing fit, grammar (35 records) | a = -0.36 s (SE 0.21), b_in = 1.15 ms/token (SE 0.10), b_out = 74.3 ms/token (SE 3.2), R^2 0.966 |
| Timing fit, free decoding (35 records) | a = -0.09 s (SE 0.15), b_in = 1.01 ms/token (SE 0.08), b_out = 45.3 ms/token (SE 1.9), R^2 0.970 |
| Fit with a = 0 (intercept not distinguishable from 0) | grammar b_in 1.09 ms, b_out 70.5 ms (RMSE 0.48 s); free b_in 0.99 ms, b_out 44.6 ms (RMSE 0.38 s) |
| Toy task (3 operations, `currency` renamed to `currency_code`, KV-prefix reuse, grammar) | 5/6 solved; mean 6.8 steps (max 10); 6 docs look-ups; 0 format errors; c_step 4.98 s at a mean 589 input / 61 output tokens |
| Toy steps: actual vs predicted fresh-cache time | 204.3 s vs 202.8 s: KV-prefix reuse saved nothing measurable at these prompt sizes |
| Determinism, same prompt twice with a fresh cache | 8/8 identical |
| Determinism, regenerating the toy steps (recorded with KV reuse) with a fresh cache, same messages (teacher-forced) | fully identical in 1 of 6 episodes; the other 5 diverge at steps 2-6 (1-based); 31/41 steps identical in total, 25/41 before the first divergence |

## What this means

1. **Output tokens dominate the cost.** With the grammar each output token costs about 74 ms against 45 ms without
   (+64%); the "thought" field is about 73 characters on average. At realistic Stage 3 prompt sizes (1,500-2,500
   input tokens, about 60 output tokens) a call costs about 5.9-7.0 s with the grammar and 4.2-5.2 s without (a = 0
   fits).
2. **Greedy output depends on how the KV cache was filled.** Fresh-cache generation is reproducible (8/8), but output
   generated with prefix reuse differs from fresh output for the same prompt in 5 of 6 episodes. Resuming a run
   therefore reproduces it exactly only if (a) every call uses a fresh cache, or (b) the run replays its own cache
   and accepts that new calls after the resume point may differ (the design's current claim).
3. **The cost of giving up prefix reuse is not measured at Stage 3 prompt sizes.** On the toy task (about 600 input
   tokens) there was no measurable saving; reusing only the system prompt saved about 12% (grammar decoding) or about
   17% (the same 0.73 s on free decoding) at a 1,273-token average. At 2,000 input tokens prefill is about 2.0 of
   4.7 s, so reusing the whole trajectory could save up to about 40%. The t_llm model assumes a fresh cache in either
   case, so this affects GPU time, not the validity of the simulation.
4. **The 7B model is usable as an agent here:** it reads informative errors, looks up docs after terse errors, and
   fails some tasks (1/6: it read the current docs twice and still sent the old key). So a memory-less agent is not
   at ceiling, which the RQ1 design needs.
5. **Pre-registered model rule (S9):** keep 7B unless c_step > 6 s. The toy c_step was 4.98 s; at real prompt sizes
   it would be about 6-7 s with the grammar, about 4.5-5 s without.

## Budget at the design's scale (about 38k LLM calls, design r4 section 9)

| Decoding | Approx. s/call at 2,000 in / 60 out | Approx. GPU hours |
|---|---|---|
| Grammar, fresh cache | 6.4 | 68 |
| Free decoding, fresh cache | 4.7 | 49 |
| Free decoding, prefix reuse | uncertain: about 2.8-4.1 (up to about 40% less) | about 30-43 |

Estimates; the number of calls per task is still a design estimate (9) until the pilot. They leave out fallback
calls, the model load (about 100 s per session) and prompts above 2,000 tokens (at 3,000 tokens: about 5.6 s per call,
about 59 h).

## Limits of this evidence
- The determinism checks and the toy episodes ran **with** the grammar; free decoding was tested on 20 single-step
  prompts only. The pilot reports the fallback rate and repeats the determinism check with free decoding.
- Determinism across Colab sessions and GPU types is untested.
- Provenance: the timing section was measured in the 2026-11-18 session (stamped with commit e4594b9, before the
  measurement-code hash existed) and reused in the 2026-11-21 session; the other sections are stamped with the
  feasibility.py hash cb2e4dcf.
