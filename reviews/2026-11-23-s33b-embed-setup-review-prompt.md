# Second-opinion review request: S3.3b embedder setup for session 2 (D142-D145)

Repository: https://github.com/notanut/eem-acc-skripsi (branch `main`). Please review critically; report findings as
High / Medium / Low with file:line and a concrete failure scenario, and separate clear-cut bugs from judgment calls.
Read the code rather than trusting this summary.

## Context
This follows your previous review (`reviews/2026-11-23-s33b-session2-prep-second-opinion.md`). Colab's preinstalled
torch (CUDA build) does not import, so session 2 replaces it with PyTorch's CPU-only build plus sentence-transformers
for the MiniLM embedder (D142). The user then decided:
- **D143:** pin the CPU code paths before torch is imported: `MKL_CBWR=COMPATIBLE` (amended from `AVX2,STRICT` after
  the internal review: Intel documents only AUTO/COMPATIBLE as working on non-Intel CPUs), `ATEN_CPU_CAPABILITY=avx2`,
  `ONEDNN_MAX_CPU_ISA=AVX2`; part of the embedder id; a probe-vector sha recorded per runtime.
- **D144:** cache, on Drive with sha256 manifests, the wheels of every distribution the install added or changed, plus
  the MiniLM snapshot; later runtimes (fresh VMs after deliberate kills) install offline.
- **D145:** record why Colab's torch fails before uninstalling it; re-run 4 S3.0 equality records with torch loaded.

## Where to look
- `src/eem_acc/stage3/llm/embed_setup.py` (`ensure_embed` and helpers): the main subject.
- `src/eem_acc/stage3/llm/embed.py` (CPU path env, local-path loading, `--probe` CLI).
- `tools/make_nb04.py` (cell `# 1) Setup`, cell `# 8) SESSION 2: open`, bench C), which generates
  `notebooks/04_stage3.ipynb`.
- `tests/test_s33b_embed_setup.py` (a fake pip and Colab image), `docs/s33b_spec.md` sections 11 and 12.

## Questions
1. The pip mechanics on Colab (Python 3.13): `pip list` before/after as the set to cache; `pip download --no-deps
   --only-binary :all:` per changed distribution (CPU torch from the CPU index); `--no-index --find-links` with every
   recorded version pinned. Could any step fail on real pip, or cache something different from what was installed?
2. Drive as the cache medium: copy under a temporary name, manifest last, rename; a manifest with missing files is
   set aside and rebuilt. Are there FUSE behaviours this misses?
3. The mode logic and refusals: a runtime that already attempted the install is refused (H1); the loaded-module
   guard; `in_place` re-runs. Any path that loops or locks the session?
4. D143: will these variables do what is intended on Colab's CPUs (Intel and possibly AMD)? Is the recorded evidence
   (ATen capability, CPU vendor/flags, probe-vector sha) enough to tell whether vectors match across CPU types?
5. Does anything here threaten the smoke run, the resume test or the D53 benchmark in session 2?
6. Test fidelity: where is the fake more permissive than real pip or Drive?

Internal reviews found and fixed: first round 0H/2M/7L, second round 1H/2M/6L (see spec section 12). The full
suite passes locally; Colab-marked tests and the real pip / Drive behaviour run only in session 2.
