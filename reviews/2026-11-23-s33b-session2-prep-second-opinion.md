**Verdict:** no High findings. Session 2 can run once one Medium is fixed (a small change) and one judgment call is decided, because that call changes every embedding vector. Everything else is Low or optional.

**What I checked:**
- **Code:** `ensure_embed_libraries`, the setup cell (`make_nb04.py`) and the open-runtime cell (notebook cell 8).
- **Tests:** after pulling `a022b6b` and rebuilding the tiny GGUF with the real template, 162 S3.3 tests pass. That covers `test_s33b_t4`, `test_s33_llama`, the notebook static tests, runner, cache and canary.
- **Import check:** in a plain interpreter, `from eem_acc.stage3.llm import t4` plus `runtime_id` import none of the nine embed libraries.
- **Package names:** on Python 3.13, `importlib.metadata` resolves `huggingface-hub`, `huggingface_hub` and `Python.DateUtil` alike, and sees new installs immediately.
- **Version compatibility:** sentence-transformers 6.3.0 on PyPI declares `transformers>=5,<6`, `huggingface-hub>=1.3,<3` and `torch>=2.2`. So with Colab's transformers 5.22.1 and hub 1.33.0 held, pip will not fall back to an old sentence-transformers.
- **Not testable here:** the PyTorch CPU index is blocked from this sandbox, so the real pip run was not executed.

## Bugs

**M1 (Medium). The library versions become binding before anyone has embedded with them.**
- **Where:** the check only imports the libraries (`t4.py:262`), `libraries` is saved straight after (`t4.py:347`), and the real MiniLM test runs later in cell 8 (`make_nb04.py:205–217`).
- **Failure:**
  1. The first runtime's set imports fine, but loading MiniLM at the pinned revision or calling `encode` fails, for example on an API transformers 5.x removed.
  2. Every later runtime installs exactly the recorded set, so it fails the same way.
  3. `session2.json` also pins the commit, so even a code fix needs a manual Drive edit (renaming `session2.json`, which loses the runtime records). That recovery isn't documented anywhere.
- **Fix (either):**
  - Make the check embed in a fresh interpreter: `MiniLMEmbedder(rev).embed("pay_order rename amount")` with one thread, recording the vector's sha and `torch.get_num_threads()`.
  - Or keep `libraries` provisional until `minilm_test` passes, and on a failure move it to `libraries_rejected` so the next runtime resolves versions again.

**L1. The "already imported" guard covers 3 of the 9 libraries.**
- **Where:** `make_nb04.py:106–107`, `t4.py:329`.
- **Scenario:** on a later runtime, Colab's image has a different numpy than the recorded one, and numpy was already imported in the kernel. I couldn't verify whether Colab's kernel preloads numpy, for example through the inline matplotlib backend.
- **Consequence:**
  1. pip replaces numpy's files under the loaded module.
  2. `session_env()` (`canary.py:216–217`) records the version still in memory.
  3. The header's `numpy` field then differs from runtime 1, so every resume across that boundary is archived as a stamp mismatch. The resume test reads as failed.
- **Fix:**
  - Map package names to import names (`scikit-learn` → `sklearn`, `huggingface-hub` → `huggingface_hub`, …).
  - Refuse when any library the plan would change is already in `sys.modules`.
  - Record which ones were loaded.

**L2. torch is not held in the second install.**
- **Where:** `t4.py:292` and `:295`.
- **Scenario:** if any package in that install declares a torch constraint the CPU build doesn't meet, pip pulls CUDA torch plus the nvidia-* packages (several GB) from PyPI. The CPU check catches it, but only after the download.
- **Fix:** add `torch==<installed CPU version>` to the second install. pip sees it as satisfied and leaves torch alone.

**L3. A failed check can loop when the versions already look installed.**
- **Where:** `t4.py:328–336`, `make_nb04.py:97–100`.
- **Scenario:** a Jupyter interrupt during pip reaches only the kernel. `subprocess.run` then kills pip outright after 0.25 s, with no rollback. The installed-package records can show the recorded versions while files are missing. Every re-run then skips pip and fails the check, and the error message doesn't tell the user to delete the runtime.
- **Fix:** on a check failure, retry once with `--force-reinstall --no-deps` of the recorded set, or say "Runtime → Disconnect and delete runtime".

**L4. Evidence gaps.**
- `before` is recorded only after a successful run (`t4.py:332`, `:349`). After a half-finished install, the retry never records Colab's original versions.
- No pip wall time is recorded, although every campaign runtime now pays for the install.
- `freeze` can be `None` silently.
- `pip check` output isn't kept.
- **Fix:** append a `libraries_attempts` entry (before, plan, start time) before running pip, and record durations.

## Judgment calls

**J1. Make embeddings identical across CPU types; decide before epoch 0.**
- **Problem:**
  - Colab T4 VMs don't all have the same CPU. Session 1 got "Xeon @ 2.00GHz"; other VMs can be Broadwell without AVX-512, or AMD.
  - MKL and ATen pick code paths by CPU instruction set, so MiniLM vectors differ at about 1e-8 across CPU types.
  - That makes the smoke run's interrupted-vs-reference comparison fail or come out inconclusive whenever a resumed runtime lands on another CPU.
- **Proposal:** since D142 now controls the torch install, set these in the setup cell before any torch import (workers inherit them) and put them in the embedder id:
  - `MKL_CBWR=COMPATIBLE` (or `AVX2`)
  - `ATEN_CPU_CAPABILITY=avx2`
  - `ONEDNN_MAX_CPU_ISA=AVX2`
- **Caveats:** I couldn't verify this here; check it on two runtimes with different `cpu_model`. It changes the vectors, so adopt it now or never.

**J2. Cache the embedder wheels and the MiniLM snapshot on Drive, with sha256s.**
- Do it the way the llama wheel is already cached. Later runtimes would then install with `--no-index --find-links`.
- **Benefits:**
  - After a kill, a runtime no longer depends on the PyTorch index, PyPI or the Hugging Face Hub being up (about 200 MB+ per runtime today).
  - It pins the exact builds, not just version strings.
  - It cuts setup time on every campaign runtime.

**J3. Turn "torch doesn't affect llama.cpp" into evidence.**
- **The gap:** session 1 measured S3.0 equality and D47 without torch, while epoch 0 will be registered with torch loaded.
- **Why I expect no change:**
  - All model layers run on the GPU.
  - Greedy decoding on the CPU is a plain argmax.
  - llama_cpp is loaded before torch in both the kernel and the worker (`unit_worker.py:36` before `:56`).
- **Cheap evidence (about 30 s):**
  - Re-run four S3.0 equality records (2 free, 2 grammar) after `MiniLMEmbedder` is constructed.
  - Re-record `ldd` of the llama libraries after the torch swap.

**J4 (optional). Record why Colab's torch fails before uninstalling it.**
- About 10 s: `import torch` in a clean interpreter, `pip show nvidia-nccl-cu12`, and `ldd libtorch_cuda.so | grep nccl`.
- This documents in the thesis why D142 was needed.

## Answers to the questions

1. **Can the pip sequence fail or silently change the environment?**
   - `--index-url` pointing at the CPU index is PyTorch's own install command. torch's dependencies are mirrored there, and installed ones that already satisfy are kept, so failures are loud resolution errors.
   - Uninstalling first is necessary, because an installed torch would otherwise satisfy the requirement.
   - The second install holds the four Colab libraries. scipy, scikit-learn and safetensors aren't held on the first runtime, but `before`/`versions` record any change.
   - Package-name normalisation is fine.
   - `pip list` is good evidence of what is installed, not of which builds (see J2).
2. **Is anything imported before the setup step?** Not by the repo's code, in either session path. Only a possible Colab kernel preload is unverified (L1).
3. **Recovery:**
   - A failure halfway: the re-run recomputes the plan.
   - A re-run after success: pip is skipped.
   - A kill during install: a fresh VM gives a clean environment.
   - An interrupt during install: can leave a half-install (L3).
   - Committed pins win over the session record. Once pins are committed, add a test that they equal session 2's `libraries.versions`, like `test_session1_evidence_matches_the_template_fixture`.
4. **Does CPU torch threaten any measurement?** I see no mechanism that changes GPU outputs. The real costs are host RAM per process and setup time per runtime. The D53 benchmark comparisons are unaffected; J3 covers the evidence.
5. **Are the tests meaningful; is any weakened?**
   - The fake-pip tests are meaningful for the branching. They cannot exercise real pip resolution; only session 2 can.
   - Nothing is weakened: the template tests now assert equality with S3.0, and the notebook static test checks more than before.
   - The canary rebuild test is now stronger: it checks the committed file against the real template.