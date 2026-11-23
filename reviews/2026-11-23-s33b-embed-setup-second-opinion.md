Let me probe the recovery paths with the repo's own fake Colab VM.

The prompt you attached is the same as last time, but `main` has moved to `acdc83e` (D143–D145), so I reviewed the current code.

**Verdict:** session 2 can run. Nothing is High. One Medium affects session 2 itself: a broken Drive cache, most likely after one of the deliberate kills, stops every later runtime until someone deletes a Drive folder by hand. The same problem will matter much more once the pins and cache are reused in later sessions.

**What I checked:**
- **Tests:** the 70 tests in `test_s33b_embed_setup`, `test_s33b_t4` and the notebook static tests pass.
- **Previous review:**
  - M1 is fixed: the check now embeds in a fresh interpreter, and `libraries` is saved only after it passes.
  - L1–L4 are fixed.
  - J1–J4 were adopted as D143–D145.
- **Recovery probes:** I ran three failure scenarios through the repo's own fake Colab VM (`World` in the tests); results below.

## Bugs

**M1 (Medium). A broken cache blocks every new runtime, and some failures leave the libraries half-installed.**

Mode selection (`embed_setup.py:310–315`) looks only at whether a manifest exists and its wheel files exist. So whatever is wrong with the cache is wrong again on every fresh VM. I hit three versions of this:

- **(a) Cut MiniLM upload.** The MiniLM snapshot has a manifest, but `model.safetensors` is missing because a kill cut the upload.
  - `copy_verified` raises "cache file missing" (`:166–167`). The wheel cache gets set aside and rebuilt in this case, but the MiniLM snapshot does not.
  - It fails after pip has already installed the libraries, so each attempt wastes the install.
  - Probe: two fresh VMs in a row failed identically.
- **(b) Damaged wheel.** A wheel exists but its sha256 doesn't match (a damaged file).
  - It raises at `:172–174` with "delete … on Drive to rebuild it", while the appended hint says to delete the runtime.
  - Deleting the runtime alone doesn't help.
- **(c) Colab image changed.** A held library that isn't cached changed in Colab's image (your own test, `test_offline_install_fails_loudly_when_colab_image_changed`).
  - `offline_plan` uninstalls Colab's torch first (`:125–127`), then the `--no-index` install fails.
  - Every new VM ends up half-installed in the same way.
  - This is unlikely within session 2's two hours, but likely over weeks once the pins and cache are reused in later sessions.

**Fix:** run pre-flight checks before pip touches anything.
1. Verify the local copy of every wheel and MiniLM file.
2. Check that each held library not in the cache is already at the recorded version (`before[n] == rec[n]`).
3. If any check fails, set the cache aside (recorded in evidence) and continue in the same call with `network_exact`, or re-download MiniLM. No pip has run yet, so this is safe.
4. Also cache wheels for all nine libraries, not only the changed ones, so offline installs depend less on Colab's image.

**L1. The "already attempted" refusal also fires when nothing was installed.**
- **Where:** the guard at `:306–308` refuses any runtime that has an earlier attempt, and that record is written before `try` (`:318`).
- **Probe:**
  1. An offline run failed the sha256 check with zero pip commands run; Colab's torch was untouched.
  2. I deleted the cache as instructed and re-ran in the same runtime.
  3. It was refused anyway, forcing a new VM and a fresh install.
- **Fix:** record "pip started" in the attempts file before each pip command, and refuse only if an earlier attempt started pip.

**L2. Spec and code disagree on the CUDA-library check.**
- **Spec (section 12):** a difference in llama.cpp's linked CUDA libraries stops the session.
- **Code:** `torch_llama_check` only records `cuda_libraries()` (`make_nb04.py:230`); only S3.0 equality stops the session (`:232`).
- **Note:** the `ldd` lines include load addresses `(0x…)` that change in every process, so a raw comparison would always differ.
- **Fix:** strip the addresses and compare the paths with session 1's `stamps.cuda_libraries.linked`, or drop that clause from the spec.

## Judgment calls

**J1. Compare the kernel's embedder with the fresh-interpreter check.**
- In cell 8, after the warm-up (`:223`), embed `PROBE_TEXTS` with `EMB` and require the vectors' sha256 to equal `LIBS["probe"]["vectors_sha256"]`.
- The check runs in a fresh interpreter, but production embeds inside the llama.cpp process.
- This costs milliseconds and shows directly that the check stands for production on the same VM. It catches a stray thread count, a missing environment setting, or interference from llama.cpp's OpenMP runtime.

**J2. The setup cell shows no output for about 5 minutes.**
- `_sh` captures all output (`make_nb04.py:97`).
- A user who thinks it has hung and interrupts it hits the refusal and needs a new VM. Interrupting also kills pip outright mid-install.
- Print one line per step (uninstall, torch, sentence-transformers, wheel cache, MiniLM, check).

**J3. A transient failure after pip finished also costs a new VM.**
- Examples: a Hugging Face Hub download error for MiniLM, or a Drive copy error.
- **Option:** log "pip completed, versions X" in the attempts file, and accept the in-place re-run in the same runtime when the installed versions equal X.

**J4. Record the cost of `MKL_CBWR=COMPATIBLE`.**
- It forces MKL's generic (SSE2-level) code paths.
- For MiniLM's tiny matrix multiplications the slowdown is probably negligible, but per-embed seconds in the check would put it in the budget evidence instead of leaving it assumed.

## Answers to the questions (current code)

1. **pip sequence.** Sound:
   - torch is held in the second install.
   - The first runtime holds all seven Colab libraries.
   - `network_exact` pins all nine.
   - `--index-url` for the CPU torch alone is PyTorch's own command.
   - Remaining risk: the offline install depends on Colab's image for libraries that aren't cached (M1c).
2. **Imports before setup.** None:
   - `t4`, `embed_setup` and `embed` import none of the libraries.
   - The guard now covers all eleven import names and refuses only when a loaded module would be replaced.
3. **Recovery:**
   - A half-installed VM is now refused, and failed attempts no longer pin the commit.
   - Gaps: cache failures repeat on every VM (M1), and the refusal is broader than needed (L1, J3).
4. **CPU torch and measurements.**
   - D145 re-checks S3.0 equality with torch loaded, after a warm-up, before epoch 0. That answers the question with evidence.
   - J1 would close the remaining gap between the check and the in-process embedder.
   - D143 only produces evidence once two runtimes land on different CPU types (recorded in `equal_first`).
5. **Tests.**
   - Meaningful for the branching and caching logic; real pip resolution still runs only on Colab.
   - Nothing is weakened.
   - Two tests (the damaged-cache and changed-image tests) lock in the "fail loudly with no automatic recovery" behaviour that M1 questions. Update them along with the fix.