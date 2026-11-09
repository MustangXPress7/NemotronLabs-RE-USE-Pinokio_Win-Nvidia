# Phase 1: NVIDIA install and GPU truth - Context

**Gathered:** 2026-11-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver a reproducible, NVIDIA-only Pinokio launcher for RE-USE whose CUDA support is
explicit, verified on an RTX 4070, and documented without unverified claims. The phase
covers auditing the prior uncommitted edits, hardening install/reset, making GPU support
fail-fast and verifiable, correcting documentation, and producing a verification report.

It does not add AMD ROCm, Apple MPS, or CPU inference. It does not validate on Quadro RTX
6000 hardware; that card's compatibility is reasoned from the CUDA 12.8 wheel architecture
list. It does not benchmark every sample rate; any number published must come from a run
that actually happened.
</domain>

<decisions>
## Implementation Decisions

### Platform and GPU scope
- **D-01:** NVIDIA-only. `pinokio.json` already declares `"gpu": ["nvidia"]`; keep that.
- **D-02:** Remove the dead AMD/Windows-DirectML, AMD/ROCm, Apple Silicon, Intel Mac, and
  generic CPU branches from `torch.js`. They contradict the declared scope and the README's
  "no CPU fallback" claim.
- **D-03:** If `gpu !== 'nvidia'` on Windows/Linux, `torch.js` must fail with an actionable
  error instead of silently installing CPU PyTorch. A silent CPU install guarantees a runtime
  crash because `app.py` uses `device = "cuda"`.
- **D-04:** Keep a Windows/Linux NVIDIA cu128 installation path. Do not reintroduce the old
  "Force CUDA on Windows" fallback that installed CUDA even when detection failed; instead,
  verify CUDA after install and fail with a clear message so detection problems are visible.

### Validation scope
- **D-05:** Real validation only on the available RTX 4070 (12 GB, compute capability 8.9).
- **D-06:** Quadro RTX 6000 (Turing, compute capability 7.5) compatibility is documented as
  *reasoned*: the installed `torch.cuda.get_arch_list()` contains `sm_75`. It is never
  presented as measured or tested.
- **D-07:** No fabricated benchmark table. Remove the existing "Measured on a Quadro RTX
  6000 (22.5 GB)" table (the card has 24 GB and the measurement never happened). If a number
  is published, it must be labeled with the machine and the command that produced it.

### Environment and install shape
- **D-08:** Keep the conda environment at `.env` (the existing working setup). Do not switch
  to a plain `venv`; the current app and logs prove conda works.
- **D-09:** Use Pinokio's declarative conda creation instead of a separate
  `conda install python=3.10 --yes` step: `conda: { path: ".env", python: "python=3.10" }`.
  This creates the environment with Python 3.10 and avoids relying on whatever Python the
  base environment happens to have.
- **D-10:** Use relative paths in launcher scripts (`.env`, `mamba`, `.`) rather than
  `path.resolve(__dirname, ...)` for `conda`/`path` attributes. The one exception is
  `update.js`'s existing `git pull` path, which is not part of this phase unless it breaks.
- **D-11:** Keep `requires: { bundle: "ai" }` on `install.js`.

### PyTorch family
- **D-12:** Pin the mutually compatible set `torch==2.10.0`, `torchvision==0.25.0`,
  `torchaudio==2.10.0` from the cu128 index. The old `torchvision==0.15.0` pin was wrong and
  must not return.
- **D-13:** Windows installs `triton-windows==3.3.1.post19`; Linux installs `triton`.
  Keep the existing optional xformers/sageattention/flashattention branches, but they remain
  disabled by default because this app uses none of them.

### Mamba and selective scan
- **D-14:** Keep the pinned `mamba` tag `v2.3.2.post1` and install it as a pure-Python
  package with `MAMBA_SKIP_CUDA_BUILD=TRUE`.
- **D-15:** Remove `MAMBA_KEEP_CUDA_BUILD` and `MAMBA_FORCE_BUILD` from `install.js` and the
  README/manual instructions. The pinned tag `v2.3.2.post1` reads
  `MAMBA_SKIP_CUDA_BUILD`, not `MAMBA_KEEP_CUDA_BUILD`; the current `MAMBA_KEEP_CUDA_BUILD`
  line is dead and misleading. (Newer upstream commits switched to `MAMBA_KEEP_CUDA_BUILD`,
  so this must be pinned deliberately.)
- **D-16:** Keep `app.py`'s runtime patch that turns the hard
  `import selective_scan_cuda` into an optional import and routes
  `mamba_simple.selective_scan_fn` to `selective_scan_ref`. Do not add a second patching
  mechanism. Clean up the dead `_install_mamba_pure_python()` no-op only if it is safe.
- **D-17:** Do not install `causal-conv1d`. `mamba_ssm.modules.mamba_simple` already guards
  its import with `try/except ImportError`; the successful RTX 4070 run happened with a
  broken `causal_conv1d` present. `reset.js` may keep removing the legacy `causal-conv1d`
  source folder so old half-installs are cleaned, and this must be documented as legacy
  cleanup rather than a current install artifact.

### Verification
- **D-18:** Add a repeatable `verify_gpu.py` at the project root. It must check:
  `torch.cuda.is_available()`, device name, compute capability,
  `torch.cuda.get_arch_list()` contains `sm_75` and `sm_86`, `torchvision`/`torchaudio`
  import, `mamba_ssm` import, and `selective_scan_cuda is None` after the app patch path is
  exercised. Print a compact JSON summary and exit non-zero with a clear message on failure.
- **D-19:** `install.js` runs `python verify_gpu.py` as its final step. A failed verification
  fails the install, not a later run.
- **D-20:** Add `verify.js` to re-run the same check without reinstalling, and expose it in
  `menu.js` as "Verify GPU" once `.env` exists.
- **D-21:** `run.js` keeps its current URL-capture pattern:
  `/(http:\/\/[0-9.:]+)/` with `local.set` of `{{ input.event[1] }}`. This matches the
  AGENTS-mandated pattern and already works in the RTX 4070 log.
- **D-22:** End the phase with a verification report that records the exact commands,
  output summary, and every claim that remains unverified.

### App behavior
- **D-23:** Add an early runtime guard in `app.py`: if `torch.cuda.is_available()` is false,
  exit with a clear NVIDIA/CUDA message instead of a later cryptic kernel error. Import
  `torch` before the guard. Do not add a CPU fallback.
- **D-24:** Change `demo.queue().launch(share=True)` to `launch(share=False)`. The public
  Gradio tunnel is unnecessary behind Pinokio's local/HTTPS proxy, and the RTX 4070 log shows
  the share-link attempt failing with a `h11` protocol error. The local URL still works.
- **D-25:** Remove the broken "Video Upload (.mp4, .mov)" tab from `app.py` and make the UI
  audio-only. `soundfile` cannot demux video, and the README already tells users to convert
  first. Video transcoding via ffmpeg is deferred; ffmpeg remains installed because other
  audio formats and future work use it.
- **D-26:** README must state: RTX 4070 tested; Quadro RTX 6000 reasoned from `sm_75`;
  Linux expected but not tested on this machine; AMD/Apple/CPU unsupported.

### Documentation
- **D-27:** Replace the fabricated "Measured GPU usage" section with an honest statement:
  no VRAM benchmark was run for this launcher; VRAM grows with sample rate and clip length;
  pre-resample to 16 kHz for small GPUs. If a real measurement is taken during verification,
  add it with machine, command, sample rate, and clip length.
- **D-28:** Add a "What we learned / What we unlearned" section covering:
  CUDA 12.8 wheel architecture coverage (`sm_70`…`sm_120`, including `sm_75` and `sm_86`);
  the wrong `torchvision==0.15.0` pin; the pinned mamba tag's `MAMBA_SKIP_CUDA_BUILD` flag;
  the fact that `causal-conv1d` is optional; the `.m4a`/video `soundfile` limitation; and the
  fact that prior changes improved install robustness but added no new GPU architecture.
- **D-29:** Keep `README.md` as the full source of truth and `README_PINOKIO.md` as the short
  launcher note. Do not duplicate stale numbers in either.

### Audit and change control
- **D-30:** Treat the current uncommitted edits as untrusted until the audit plan maps each
  change to evidence. Keep changes that are correct, fix those that are incomplete, and
  revert anything that weakens GPU behavior. Do not commit the prior edits as-is.
- **D-31:** Follow `AGENTS.md` and `PINOKIO.md` for every launcher edit. Reference examples:
  `system/examples/mochi/start.js` (URL capture),
  `system/examples/accdiffusion/torch.js` (torch branches),
  `system/examples/AllTalk-TTS/install.js` (conda path),
  `system/examples/openaudio/install.js` (`requires.bundle`).

### the agent's Discretion
- Exact wording of error messages and README prose.
- Whether `verify.js` uses `shell.run` directly or wraps `verify_gpu.py`.
- Whether `verify_gpu.py` prints JSON only or JSON plus a human summary.
- Whether the legacy `_install_mamba_pure_python()` no-op in `app.py` is removed or left
  with a clarifying comment, as long as there is only one active patch path.
- Whether `update.js` is touched at all, as long as it does not contradict the new flow.
</decisions>

<specifics>
## Specific Ideas

- The verification output should be easy to paste into the final report:
  `{"cuda": true, "device": "NVIDIA GeForce RTX 4070", "capability": [8, 9], "arch_list": ["sm_70", ...], "mamba": "2.3.2.post1", "selective_scan_cuda": null}`.
- Error text for a missing GPU should name the requirement, not just say "CUDA error":
  "RE-USE requires an NVIDIA GPU with a CUDA 12.8-capable driver. Pinokio detected: <gpu>."
- The README's compatibility table should use three states only: Tested, Reasoned, Unsupported.
- Do not turn the phase into a generic refactor. Every change must map to a requirement ID.
</specifics>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### GSD state
- `.planning/PROJECT.md` — project context, validated/active requirements, constraints, decisions
- `.planning/REQUIREMENTS.md` — REQ-IDs and traceability
- `.planning/ROADMAP.md` — Phase 1 goal, requirements, and success criteria
- `.planning/STATE.md` — current position and blockers

### Launcher code
- `install.js` — current install flow; conda creation, torch delegation, requirements, mamba build, completion input
- `torch.js` — current PyTorch installer; NVIDIA cu128 branches plus dead AMD/Apple/CPU branches
- `reset.js` — deletes `.env`, `cache`, `logs`, `output`, `causal-conv1d`, `mamba`
- `run.js` — daemon start and URL capture
- `menu.js` — dynamic menu; install/run/update/reset visibility
- `pinokio.json` — version, platform, arch, GPU declaration, icon
- `app.py` — hardcoded `device = "cuda"`, runtime mamba patch, chunked inference, Gradio UI, `share=True`
- `requirements_full.txt` — Python dependencies; `torchcodec` already removed
- `requirements.txt` — core list; check whether it is still used

### Documentation to correct
- `README.md` — full install/run/requirements/compatibility/troubleshooting; contains the unverified Quadro table and the dead mamba flag
- `README_PINOKIO.md` — short launcher notes; repeats the dead mamba flag and the unverified VRAM summary

### Evidence (read before changing anything)
- `logs/api/torch.js/1794214330492` — original torch install failed with "No virtual environment found" because it ran in `base`, not `.env`
- `logs/api/install.js/1794214330492` — `uv pip install -r requirements_full.txt` worked inside the `.env` conda environment with Python 3.10.22
- `logs/api/install.js/1794215211886` — original mamba clone/build failure
- `logs/api/run.js/1794218407913` — successful model load and 72.9 s inference on the RTX 4070; also shows the failed Gradio share link / `h11` error
- `logs/api/run.js/1794217166729` — `.m4a` "Format not recognised" failure from `soundfile`

### Pinokio API and rules
- `AGENTS.md` — mandatory workflow, destination rules, URL-capture pattern, examples-first policy
- `D:\pinokio\prototype\PINOKIO.md` — `conda` config (§3711), `gpu`/`gpus`/`gpu_target` (§7728), `on` (§3822), `fs.rm` (§4768)
- `D:\pinokio\prototype\system\examples\mochi\start.js` — canonical URL capture block
- `D:\pinokio\prototype\system\examples\accdiffusion\torch.js` — canonical NVIDIA cu128 torch branch shape
- `D:\pinokio\prototype\system\examples\AllTalk-TTS\install.js` — conda path example
- `D:\pinokio\prototype\system\examples\openaudio\install.js` — `requires: { bundle: "ai" }` example
</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `run.js` already implements the required `on` + `local.set` URL capture; reuse it unchanged.
- `menu.js` already switches between install and run based on `info.exists('.env')`; add the
  verify entry in the installed branch.
- `reset.js` already uses `fs.rm` correctly per Pinokio docs; extend or annotate it rather
  than rewriting it.
- The installed `.env` is a working Python 3.10.22 conda environment with
  `torch 2.10.0+cu128`, `torchvision 0.25.0+cu128`, `mamba_ssm 2.3.2.post1`; use it as the
  reference state for verification.

### Established Patterns
- Pinokio scripts are CommonJS modules exporting `{ run: [...] }`; `daemon: true` is required
  for the Gradio server.
- `shell.run` uses `conda: { path, python }` for env creation/activation and `path` for cwd.
- `next: null` terminates a branch in `torch.js`.
- `when: "{{...}}"` expressions gate steps; `gpu`, `platform`, `arch`, `which`, `exists`,
  `running`, and `local` are available.
- `script.start` passes `params` to another script as `args`.

### Integration Points
- `install.js` → `script.start torch.js` → returns to `install.js` → `requirements_full.txt`
  → mamba clone/build → `verify_gpu.py`.
- `run.js` → `python app.py` → Gradio local URL → `local.url` → `menu.js`.
- `menu.js` → `verify.js` → `.env` Python → `verify_gpu.py` → terminal output.
- `reset.js` removes the artifacts created by `install.js` plus legacy build folders.
- `pinokio.json` controls whether Pinokio offers the app on NVIDIA-only machines.
</code_context>

<deferred>
## Deferred Ideas

- AMD ROCm support — no hardware, no validated Mamba/Triton story.
- Apple Silicon / MPS — app is CUDA-only by design.
- CPU inference — requires an app-level device abstraction and performance work.
- Real Quadro RTX 6000 validation — hardware unavailable.
- Video transcoding via ffmpeg — removed the broken video tab in this phase; a future phase
  can add `ffmpeg -vn` extraction properly and test it.
- Per-sample-rate VRAM benchmark — only publish if a real measurement is taken; otherwise
  keep the honest "not benchmarked" statement.
- Upstream mamba version bump (the local checkout is at `e9594ce`, nine commits past the
  pinned tag) — do not bump during this phase; the pinned tag is part of the proven setup.
</deferred>

---

*Phase: 01-nvidia-install-and-gpu-truth*
*Context gathered: 2026-11-09*
