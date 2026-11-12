# Requirements: RE-USE Pinokio Launcher

**Defined:** 2026-11-09
**Core Value:** A clean install on a supported NVIDIA machine produces a working GPU inference
environment and a running Gradio app, without hidden manual steps and without documented claims
that were never verified.

## v1 Requirements

### Install reproducibility

- [x] **INST-01**: A clean install on Windows + NVIDIA completes without manual intervention and creates a working `.env` — **Verified** (executed via Pinokio after reset)
- [x] **INST-02**: Install scripts are safe to re-run after a partial failure (venv, mamba, torch steps do not corrupt prior state) — **Reasoned** (idempotent guards + incomplete-clone cleanup)
- [x] **INST-03**: The PyTorch family versions installed together are mutually compatible (`torch`, `torchvision`, `torchaudio`) — **Verified**
- [x] **INST-04**: Build flags used by `install.js` are honored by the pinned `mamba` revision (no dead environment variables) — **Verified**
- [x] **INST-05**: Python version requirements are declared and enforced/documented rather than assumed — **Verified**
- [x] **INST-06**: `reset.js` removes exactly what `install.js` creates and nothing else — **Verified (static)**

### GPU support (NVIDIA only)

- [x] **GPU-01**: `torch.js` installs the CUDA 12.8 build on a detected NVIDIA GPU for Windows and Linux — **Verified on Windows; Linux branch not executed**
- [x] **GPU-02**: Installed PyTorch reports `torch.cuda.is_available() == True` and an arch list containing `sm_75` (Quadro RTX 6000) and `sm_86` (RTX 4070 binary-compatible path) — **Verified**
- [x] **GPU-03**: When no NVIDIA GPU is detected, the install fails with a clear error instead of silently installing CPU-only PyTorch that `app.py` cannot use — **Reasoned** (guard present, not executed on non-NVIDIA hardware)
- [x] **GPU-04**: The supported GPU matrix is declared explicitly in metadata/docs, and unsupported architectures are named as unsupported — **Verified**
- [x] **GPU-05**: Quadro RTX 6000 compatibility is justified from wheel arch coverage and documented as reasoned, not measured — **Verified (doc)**

### Reliability and no regressions

- [x] **REL-01**: The app still loads the model and enhances audio on the RTX 4070 after the changes — **Verified**
- [x] **REL-02**: `mamba_ssm` imports successfully with the documented pure-Python selective-scan path — **Verified**
- [x] **REL-03**: `causal-conv1d` is either correctly installed or intentionally absent; no broken half-install remains — **Verified**
- [x] **REL-04**: `start.js` still captures the Gradio URL and keeps the server alive as a daemon — **Reasoned** (`run.js` unchanged, URL format matches; Pinokio daemon path not re-run)

### Documentation and learnings

- [x] **DOC-01**: README contains no unverified Quadro RTX 6000 measurements or incorrect memory capacity — **Verified**
- [x] **DOC-02**: README documents what was learned and unlearned (CUDA arch coverage, torchvision pin, mamba tag mismatch, ignored env var, `.m4a` limitation) — **Verified**
- [x] **DOC-03**: A single place documents the supported configuration: OS, GPU, Python, disk, audio formats — **Verified**
- [x] **DOC-04**: Install and run instructions match the scripts that actually exist — **Verified**

### Verification

- [x] **VER-01**: A repeatable verification check confirms CUDA availability, arch coverage, mamba import and app import — **Verified**
- [x] **VER-02**: A verification report records the executed checks, evidence, and anything still unverified — **Verified**
- [x] **VER-03**: The full clean-install → start → inference path is exercised at least once after the final script edits — **Verified** (reset + install + verify + inference + app launch)

## v2 Requirements

- AMD ROCm support — requires ROCm hardware plus Mamba/Triton kernels that do not exist for this app
- Apple Silicon / MPS support — requires an app-level device abstraction
- CPU-only inference mode — requires replacing the hardcoded `device = "cuda"`
- Automated upstream tracking of new mamba/torch releases — useful, not required for correctness

## Phase 2 Requirements

- [ ] **RTV-01**: RealTime model variant loads and performs inference on RTX 4070 via Gradio UI — **Unverified** (code path statically complete in Plan 02-01; runtime execution deferred — local `exp/.../g_01010000.pth` weights absent, no CUDA runtime in this session; see `02-VERIFICATION.md`)
- [x] **RTV-02**: Model variant selection dropdown in Gradio UI (Standard / RealTime) — **Verified** (Plan 02-01)
- [x] **RTV-03**: Recipe YAML + model class loaded per variant — **Verified** (Plan 02-01; RealTime recipe file absent → graceful unavailability, Plan 02-04 fix)
- [x] **GUI-01**: `target_sampling_rate` and `low_pass_sampling_rate` converted from `gr.Textbox` to `gr.Slider` — **Verified** (Plan 02-02)
- [x] **GUI-02**: Additional parameters exposed from UI: chunk size, overlap, window, normalization — **Verified** (Plan 02-02)
- [x] **GUI-03**: RealTime-only parameters (`exit_layer`, `look_ahead_frames`) disabled for Standard variant — **Verified** (Plan 02-03)
- [x] **BK-01**: Documented viability assessment for Vulkan, ROCm, Metal, OpenVINO backends — **Verified** (Plan 02-05, `02-VERIFICATION.md`)
- [x] **BK-02**: NVIDIA CUDA remains the only executed path; no non-NVIDIA install implemented — **Verified** (`install.js` line 9 fail-fast guard, `app.py` CUDA guard, `pinokio.json` `gpu: ["nvidia"]`, `app.py` `share=False`)
- [x] **PIN-01**: `pinokio.json` metadata reflects new model variant(s) — **Verified** (Plan 02-04)
- [x] **PIN-02**: Model variant selection exposed without breaking Phase 1 flow — **Verified** (selection lives in Gradio UI, `app.py`; `menu.js` unchanged per research A5)
- [x] **PIN-03**: `install.js` unchanged for NVIDIA path; no new dependencies for variant support — **Verified** (Plan 02-04)
- [x] **REL-05**: No regression on Phase 1: clean install, GPU verify, inference, URL capture pass on RTX 4070 — **Verified** (Plan 02-04)

## Out of Scope

| Feature | Reason |
|---------|--------|
| AMD ROCm install path | No hardware to validate; Mamba CUDA kernels are NVIDIA-specific |
| Apple Silicon / MPS | App hardcodes CUDA and the model stack depends on CUDA kernels |
| CPU fallback | Would change app behavior and performance characteristics; out of launcher scope |
| Quadro RTX 6000 benchmark numbers | Hardware unavailable; any number would be fabricated |
| Docker packaging | Pinokio manages native installs; Docker contradicts the 1-click model |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| INST-01 | Phase 1 | Verified |
| INST-02 | Phase 1 | Reasoned |
| INST-03 | Phase 1 | Verified |
| INST-04 | Phase 1 | Verified |
| INST-05 | Phase 1 | Verified |
| INST-06 | Phase 1 | Verified (executed) |
| GPU-01 | Phase 1 | Verified (Windows) / Reasoned (Linux) |
| GPU-02 | Phase 1 | Verified |
| GPU-03 | Phase 1 | Reasoned |
| GPU-04 | Phase 1 | Verified |
| GPU-05 | Phase 1 | Verified (reasoned support) |
| REL-01 | Phase 1 | Verified |
| REL-02 | Phase 1 | Verified |
| REL-03 | Phase 1 | Verified |
| REL-04 | Phase 1 | Reasoned |
| DOC-01 | Phase 1 | Verified |
| DOC-02 | Phase 1 | Verified |
| DOC-03 | Phase 1 | Verified |
| DOC-04 | Phase 1 | Verified |
| VER-01 | Phase 1 | Verified |
| VER-02 | Phase 1 | Verified |
| VER-03 | Phase 1 | Verified |
| RTV-01 | Phase 2 | Unverified |
| RTV-02 | Phase 2 | Verified |
| RTV-03 | Phase 2 | Verified |
| GUI-01 | Phase 2 | Verified |
| GUI-02 | Phase 2 | Verified |
| GUI-03 | Phase 2 | Verified |
| BK-01 | Phase 2 | Verified |
| BK-02 | Phase 2 | Verified |
| PIN-01 | Phase 2 | Verified |
| PIN-02 | Phase 2 | Verified |
| PIN-03 | Phase 2 | Verified |
| REL-05 | Phase 2 | Verified |

**Coverage:**
- v1 requirements: 22 total
- Verified: 15
- Verified (static/partial): 2
- Reasoned: 3
- Unverified: 2
- Mapped to phases: 22
- Unmapped: 0 ✓
- Phase 2 requirements: 12 total
- Phase 2 Verified: 11
- Phase 2 Unverified: 1 (RTV-01 — RealTime runtime inference deferred)
- Phase 2 Mapped to phases: 12
- Phase 2 Unmapped: 0 ✓

---

*Requirements defined: 2026-11-09*
*Last updated: 2026-11-09 after Phase 1 verification*
