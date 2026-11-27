# Second-opinion review request: S3.5 spec r2 (frozen analysis and blinded test phase)

Repository: https://github.com/notanut/eem-acc-skripsi (branch `main`, the commit named in the chat).

**Contract:** `docs/s35_spec.md` r2. **Pre-registered design:** `docs/stage3_design.md`, sections 7-11.
**Machinery reused:** `docs/s34_spec.md` r11, `src/eem_acc/stage3/s34.py`, `s34_platform.py` and `llm/runner.py`.

**What S3.4 produced:** wording W1, no switches, cfg5 (k = 5, k_s = 3, theta_link = 0.65), 20 tasks, no cuts,
and 11.41 + 46.15 = 57.56 of 60 GPU-hours projected.

A fresh-context review of r1 found 4 High, 7 Medium and 10 Low issues, all applied in r2 (spec section 19).
A fresh-context review of the first Part 1 code (mutation-tested) found 2 High, 5 Medium and 7 Low, all
applied (spec section 20).

**First Part 1 code, on the same commit:**
- `src/eem_acc/stage3/analysis/graphical.py`: the BMBP09 procedure, intersection weights, exact closed-test
  adjusted p-values, and the RQ1 and RQ3 graphs;
- `stats.py`: the one-sided sign-flip test, MDD, t interval and bootstrap;
- `data.py`: normalisation and completeness;
- `endpoints.py`: RQ1 E1-E4 and RQ3 R1/R2 with integer-based NI values, plus the interaction and floor rule;
- the tests `tests/test_s35_graphical.py` and `tests/test_s35_endpoints.py`.

Not yet written: secondary, descriptive, mechanism, report, `s35.py` (chain), notebook 06.

## The user decided (2026-11-27)

- **D179:** no deadline. The test phase stops early only through the budget accounting.
- **D180:** one spec in two parts. Part 1 (the analysis) is frozen before any test unit runs.
- **D181:** reuse the S3.4 chain machinery: retries, pauses, re-pin with adoption by replay, and the freeze check.

## Calls Claude made on the user's behalf (the user was away and said "do anything you recommend")

Please challenge each one. Say whether you agree, and if not, which option you would pick and why.

| ID | Question | Options | Claude chose |
|----|----------|---------|--------------|
| D182 | How is the analysis frozen? | (a) analysis modules join the test chain's protocol files; a freeze record (sha256 of every protocol file) goes into every test plan's `meta`; GitHub release `s35-freeze` for a server timestamp; (b) git tag only | (a) |
| D183 | In what order do the descriptive units run? | (a) 11 tiers in the reverse of the pre-registered cut order (spec section 11 table), so a stop at a tier boundary leaves at least a pre-registered cut configuration; (b) arm by arm | (a) |
| D184 | What stops the test phase on budget? | (a) checked before each T block and S tier against the frozen S3.4 cost model; in T a pause (`stop` or `extend`, blind); in S an automatic stop at the tier boundary; (b) hard stop at 60 h | (a) |
| D185 | When are results unblinded? | (a) everything at the end; (b) primary results after T, with every S rule (budget stop, X retry then accept_failed, epoch) fixed in the freeze now; (c) primary only after S | (b) |
| D186 | What happens to a primary unit that fails for good? | (a) its stream or seed is left out for every condition (complete case), plus a sensitivity analysis counting its tasks as failures; (b) count as failures only | (a) |
| D187 | How is the mixed-effects logistic model (secondary) fitted? | (a) statsmodels `BinomialBayesMixedGLM.fit_vb` with crossed random intercepts for stream and template, priors stated, GEE alongside; (b) R lme4; (c) GEE only | (a) |
| D188 | Which code changes are allowed between S3.4 and the test phase? | (a) only if every S3.4 unit still replays identically and no S3.4 behaviour module or data changes (else a pause); none planned; (b) none | (a) |
| D189 | Are costs and timing blinded? | (a) no per-unit progress lines; progress and ETA (from the frozen cost model) only at block, tier and session ends; bundle ledger = total GPU-hours only; (b) per-unit lines | (a) |
| D190 | Where does the analysis run? | (a) all on Colab with new pins; (b) only on Claude's machine; (c) hybrid: the notebook computes the exact primary tests with numpy and scipy (nothing installed) and records a canonical hash (floats rounded to 10 significant digits) in the chain log; Claude runs the full analysis on the export and must reproduce that hash | (c) |
| D191 | How is the code structured? | (a) `s35.py` with `TestChain(s34.Chain)` and `TestPlatform(Platform)` overriding the S3.4-specific parts (spec section 12); `s34.py`, `s34_platform.py` and `runner.py` unchanged; (b) refactor a shared core | (a) |

## Questions

1. **Fidelity.** Is the spec faithful to design section 8? That covers:
   - E1-E4, the NI margin of one task per stream, and the graph weights and transitions;
   - the RQ3 graph and the floor rule;
   - the early-stop completeness rule, D63 and D64.

   Is anything silently dropped or added?
2. **Statistics.**
   - Is the one-sided exact sign-flip test with integer NI values correct?
   - Is the symmetry assumption stated correctly?
   - Is the closed-test formula for adjusted p-values right for these graphs?
   - Are the BMBP09 update formulas in `graphical.py` right?
3. **Blinding.** Is there any path by which outcomes by condition could reach the user or Claude during T? Consider
   the bundle whitelist, block-level progress, error signatures, `X`/`Z` pauses and the budget rule. Does D185(b)
   really leave no outcome-dependent decision after unblinding?
4. **D183 tiers.** Does every prefix of whole tiers give at least the matching cut configuration (design section 9,
   cut steps 0-11)?
5. **Budget.** Can pause B deadlock or leak anything? Is the per-block projection sound?
6. **Code.** Anything wrong in the Part 1 modules or their tests?

Please classify each finding as High, Medium or Low, and give a concrete fix for each.
